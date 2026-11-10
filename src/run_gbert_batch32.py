"""Reproduce the GBERT physical-batch-32 controls with RNG-preserving checkpointing."""

import argparse
import gc
import hashlib
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import train
import predict
from get_loggers import get_logger
from classification_loss import compute_classification_loss
from dataset import load_gahd
from evaluate_predictions import compute_binary_metrics as score_metrics
from gold_labels import load_gold_labels

logger = get_logger("gbert_batch32")


def get_training_arguments(path_run: Path, seed: int) -> TrainingArguments:
    """Return the physical-batch-32 control recipe used in the saved runs."""
    return TrainingArguments(
        output_dir=str(path_run),
        num_train_epochs=3.0,
        learning_rate=2e-05,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1-macro",
        save_total_limit=3,
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "preserve_rng_state": True,
        },
    )


def main(args: argparse.Namespace) -> None:
    """Train each control seed and predict all three Swiss German variants."""
    logger.info(json.dumps(vars(args), default=str))
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide distinct training seeds")
    for seed in args.seeds:
        config = json.loads(
            (args.path_configs / f"gahd_gbert_raw_rtx3080_run{seed}.json").read_text(
                encoding="utf-8"
            )
        )["train"]
        config.update(
            experiment_name="gahd_gbert_batch32_control",
            batch_size=32,
            gradient_accumulation_steps=1,
            resume=False,
            epochs=3.0,
            learning_rate=2e-5,
            weight_decay=0.01,
            patience=3,
            log_interval=50,
            max_length=None,
            model_family="bert",
            no_fp16=True,
            path_runs_dir=str(args.path_runs),
            path_gahd=str(args.path_data / "gahd.csv"),
        )
        run = args.path_runs / config["experiment_name"] / f"run{seed}"
        train.check_empty_output_dir(str(run))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        train_set, dev_set, test_set = load_gahd(config["path_gahd"])
        tokenizer = BertTokenizer.from_pretrained(
            config["model_name"], local_files_only=True
        )
        datasets = [
            train.encode_dataset(df[["text", "label"]], tokenizer, None)
            for df in [train_set, dev_set, test_set]
        ]
        model = train.load_model(config["model_name"], "bert")
        training_args = get_training_arguments(run, seed)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets[0],
            eval_dataset=datasets[1],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=train.compute_metrics,
            compute_loss_func=compute_classification_loss,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        metadata = {
            **config,
            "gradient_checkpointing": True,
            "checkpoint_rng_preserved": True,
            "python_version": platform.python_version(),
            "package_versions": {
                name: version(name)
                for name in [
                    "torch",
                    "transformers",
                    "datasets",
                    "pandas",
                    "scikit-learn",
                ]
            },
            "hardware": torch.cuda.get_device_name(0),
            "device": str(training_args.device),
            "effective_batch_size": 32,
            "optimizer": str(training_args.optim),
            "gahd_sha256": hashlib.sha256(
                Path(config["path_gahd"]).read_bytes()
            ).hexdigest(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "loss_normalization": "sum_cross_entropy_over_accumulated_examples",
            "purpose": "Controlled full-batch-32 comparison; memory checkpointing was verified to preserve gradients exactly on the probe.",
        }
        (run / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        logger.info("Start training seed %s.", seed)
        result = trainer.train()
        trainer.save_state()
        logger.info("Finished training seed %s.", seed)
        best = trainer.state.best_model_checkpoint
        metadata.update(
            best_model_checkpoint=best,
            best_dev_macro_f1=trainer.state.best_metric,
            train_runtime=result.metrics["train_runtime"],
        )
        (run / "experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        test = trainer.predict(datasets[2])
        train.write_test_results(test_set, test.predictions, str(run))
        del trainer, model, datasets
        gc.collect()
        torch.cuda.empty_cache()
        gold = load_gold_labels(args.path_data)
        metrics = {}
        for variant, path in [
            ("text", args.path_data / "eval_sentences.csv"),
            ("translation_haiku", args.path_data / "pred_input_haiku.csv"),
            ("translation_apertus8b", args.path_data / "pred_input_apertus8b.csv"),
        ]:
            predict.main(
                argparse.Namespace(
                    experiment_name=config["experiment_name"],
                    model_name=config["model_name"],
                    max_length=None,
                    path_checkpoint=best,
                    path_eval_set=str(path),
                    input_column=variant,
                    batch_size=32,
                    no_fp16=True,
                    path_runs_dir=str(args.path_runs),
                    seed=seed,
                )
            )
            scores = (
                pd.read_csv(run / f"predictions_swiss_german_{variant}.csv")
                .set_index("sentence_id")
                .loc[gold.index, "prob_hate"]
                .to_numpy()
            )
            metrics[variant] = score_metrics(gold.label.to_numpy(), scores, 0.5)
            logger.info("Seed %s, %s: %s", seed, variant, json.dumps(metrics[variant]))
        (run / "swiss_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=4), encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Training seeds; each reads its existing local GBERT config.",
    )
    parser.add_argument(
        "--path_configs",
        type=Path,
        default=Path("configs"),
        help="Directory containing the local GBERT configs.",
    )
    parser.add_argument(
        "--path_runs",
        type=Path,
        default=Path("runs"),
        help="Output root; existing runs are refused.",
    )
    parser.add_argument(
        "--path_data",
        type=Path,
        default=Path("data"),
        help="GAHD, annotations and prepared prediction inputs.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
