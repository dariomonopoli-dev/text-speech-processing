"""Export the local GBERT and corrected SwissBERT comparison with provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_eval_sentences import load_annotated_sentences
from evaluate_predictions import (
    BOOTSTRAP_METRICS,
    compute_binary_metrics,
    load_predictions,
)
from get_loggers import get_logger
from gold_labels import load_gold_labels

logger = get_logger("export_comparison")
VARIANTS = [
    ("approach_1", "GBERT, raw Swiss German", "gahd_gbert_raw", "text"),
    ("approach_2_haiku", "GBERT, Haiku", "gahd_gbert_raw", "translation_haiku"),
    (
        "approach_2_apertus8b",
        "GBERT, Apertus-8B",
        "gahd_gbert_raw",
        "translation_apertus8b",
    ),
    (
        "approach_3",
        "SwissBERT, raw Swiss German",
        "gahd_swissbert_raw_lossfixed",
        "text",
    ),
]


def compute_file_hash(path_input: Path) -> str:
    with path_input.open("rb") as fin:
        return hashlib.file_digest(fin, "sha256").hexdigest()


def main(args: argparse.Namespace) -> None:
    logger.info(json.dumps(vars(args), default=str))
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Comparison requires at least two distinct seeds")
    if args.path_output_dir.exists() and any(args.path_output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.path_output_dir}")
    gold = load_gold_labels(args.path_annotations)
    gold.index = gold.index.map(str)
    sentences = load_annotated_sentences(args.path_annotations).set_index("sentence_id")
    predictions = sentences.join(gold, validate="one_to_one")
    if predictions.isna().any().any():
        raise ValueError("Gold labels and annotated sentence texts do not align")
    all_rows, summary_rows, metadata = [], [], {}
    reference_environment = None
    for variant, title, experiment, input_column in VARIANTS:
        scores_per_seed = {}
        for seed in args.seeds:
            path_run = args.path_runs / experiment / f"run{seed}"
            run_metadata = json.loads((path_run / "experiment.json").read_text())
            if (
                run_metadata.get("loss_normalization")
                != "sum_cross_entropy_over_accumulated_examples"
            ):
                raise ValueError(f"Missing corrected loss normalization in {path_run}")
            environment = {
                key: run_metadata[key]
                for key in [
                    "hardware",
                    "device",
                    "python_version",
                    "package_versions",
                    "gahd_sha256",
                ]
            }
            if reference_environment is None:
                reference_environment = environment
            elif environment != reference_environment:
                raise ValueError(f"Environment or training data differs in {path_run}")
            path_predictions = path_run / f"predictions_swiss_german_{input_column}.csv"
            scores = load_predictions(path_predictions, "prob_hate")
            scores.index = scores.index.map(str)
            if not scores.index.is_unique or set(scores.index) != set(
                predictions.index
            ):
                raise ValueError(
                    f"Prediction IDs differ from gold in {path_predictions}"
                )
            if not np.isfinite(scores).all() or not scores.between(0, 1).all():
                raise ValueError(f"Invalid probabilities in {path_predictions}")
            scores = scores.loc[predictions.index]
            predictions[f"{variant}_prob_hate_seed{seed}"] = scores
            scores_per_seed[seed] = scores
            metadata_key = f"{experiment}/run{seed}"
            if metadata_key not in metadata:
                state = json.loads((path_run / "trainer_state.json").read_text())
                path_checkpoint = Path(state["best_model_checkpoint"])
                if not path_checkpoint.is_absolute():
                    path_checkpoint = Path.cwd() / path_checkpoint
                metadata[metadata_key] = {
                    **run_metadata,
                    "checkpoint_sha256": compute_file_hash(
                        path_checkpoint / "model.safetensors"
                    ),
                    "base_weights_sha256": compute_file_hash(
                        Path(run_metadata["model_name"]) / "model.safetensors"
                    ),
                }
            for subset in ["all", "shared"]:
                part = (
                    predictions
                    if subset == "all"
                    else predictions.loc[predictions.is_shared]
                )
                metrics = compute_binary_metrics(
                    part.label.to_numpy(), scores.loc[part.index].to_numpy(), 0.5
                )
                stored = json.loads(
                    (
                        path_run / f"swiss_german_{input_column}_{subset}_metrics.json"
                    ).read_text()
                )
                if any(abs(metrics[key] - stored[key]) > 1e-12 for key in metrics):
                    raise ValueError(
                        f"Recomputed metrics differ in {path_run}: {subset}"
                    )
                all_rows.append(
                    {
                        "variant": variant,
                        "title": title,
                        "subset": subset,
                        "seed": seed,
                        **metrics,
                    }
                )
        predictions[f"{variant}_mean_prob_hate"] = pd.concat(
            scores_per_seed, axis=1
        ).mean(axis=1)
        for subset in ["all", "shared"]:
            rows = pd.DataFrame(
                [
                    row
                    for row in all_rows
                    if row["variant"] == variant and row["subset"] == subset
                ]
            )
            path_aggregate = (
                args.path_runs / experiment / f"swiss_german_{input_column}_{subset}"
            )
            bootstrap = json.loads(
                (path_aggregate / f"seed_avg_bootstrap_{subset}.json").read_text()
            )
            if bootstrap["num_resamples"] != 1000 or bootstrap["seed"] != 42:
                raise ValueError(f"Unexpected bootstrap settings in {path_aggregate}")
            for metric in BOOTSTRAP_METRICS:
                summary_rows.append(
                    {
                        "variant": variant,
                        "title": title,
                        "subset": subset,
                        "metric": metric,
                        "mean": rows[metric].mean(),
                        "stddev": rows[metric].std(),
                        "ci_low": bootstrap[f"{metric}-ci_low"],
                        "ci_high": bootstrap[f"{metric}-ci_high"],
                        "num_runs": len(rows),
                        "num_examples": int(rows.num_examples.iloc[0]),
                    }
                )
    args.path_output_dir.mkdir(parents=True, exist_ok=True)
    raw, summary = pd.DataFrame(all_rows), pd.DataFrame(summary_rows)
    raw.to_csv(args.path_output_dir / "by_seed.csv", index=False)
    summary.to_csv(args.path_output_dir / "summary.csv", index=False)
    predictions.to_csv(args.path_output_dir / "predictions.csv")
    manifest = {
        "evaluation_date": args.evaluation_date,
        "seeds": args.seeds,
        "threshold": 0.5,
        "gold_rule": "Majority on shared sentences; original label on single_annotator sentences.",
        "num_examples": len(predictions),
        "num_true_hate": int(predictions.label.sum()),
        "annotation_sha256": {
            path.name: compute_file_hash(path)
            for path in sorted(
                args.path_annotations.glob("annotation_*_stratified.csv")
            )
        },
        "translation_sha256": {
            name: compute_file_hash(args.path_annotations / name)
            for name in ["haiku_translations_700.csv", "eval_sentences_apertus8b.csv"]
        },
        "bootstrap": {
            "num_resamples": 1000,
            "seed": 42,
            "method": "Sentence bootstrap of seed-mean metrics, excluding resamples without positives.",
        },
        "runs": metadata,
    }
    quality = {}
    for name, filename in [
        ("haiku", "haiku_translations_700_audited.csv"),
        ("apertus8b", "eval_sentences_apertus8b.csv"),
    ]:
        audited = pd.read_csv(args.path_annotations / filename)
        quality[name] = {
            column: int(audited[column].astype("string").str.lower().eq("true").sum())
            for column in ["translation_unchanged", "translation_has_anomaly"]
        }
    manifest["translation_quality_counts"] = quality
    lines = [
        "# Local comparison of Approaches 1, 2 and 3",
        "",
        f"Completed {args.evaluation_date}. Classifier training and inference used the same NVIDIA GeForce RTX 3080 and Python environment. The existing Haiku and Apertus translations were reused; their generation was not repeated on this machine.",
        "",
        f"All variants use the same {len(predictions)} gold sentences ({int(predictions.label.sum())} hate), seeds {args.seeds}, threshold 0.5, and corrected accumulation-loss normalization. Checkpoints were selected using GAHD development macro-F1. Test labels were used only for evaluation.",
        "",
        "| Variant | Macro-F1 (%) | Hate F1 (%) | Precision (%) | Recall (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, title, _, _ in VARIANTS:
        part = summary.loc[
            (summary.variant == variant) & (summary.subset == "all")
        ].set_index("metric")
        cells = [
            f"{100*part.loc[m, 'mean']:.2f} ± {100*part.loc[m, 'stddev']:.2f}"
            for m in ["f1-macro", "f1-pos", "precision", "recall"]
        ]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    constant = compute_binary_metrics(
        predictions.label.to_numpy(), np.zeros(len(predictions)), 0.5
    )
    lines += [
        "",
        f"For context, always predicting non-hate gives macro-F1 {100*constant['f1-macro']:.2f}%, hate F1 0% and accuracy {100*constant['acc']:.2f}%. The low positive prevalence makes accuracy alone misleading.",
    ]
    path_old_manifest = (
        args.path_annotations.parent / "results/swissbert_raw_metrics.json"
    )
    path_old_scores = args.path_annotations.parent / "results/swissbert_raw_by_seed.csv"
    if path_old_manifest.is_file() and path_old_scores.is_file():
        old_manifest = json.loads(path_old_manifest.read_text(encoding="utf-8"))
        if old_manifest["annotation_sha256"] == manifest["annotation_sha256"]:
            old = pd.read_csv(path_old_scores)
            old = old.loc[(old.subset == "all") & old.seed.isin(args.seeds)]
            new = raw.loc[(raw.variant == "approach_3") & (raw.subset == "all")]
            if len(old) == len(new) == len(args.seeds):
                changes = {
                    metric: {
                        "original": old[metric].mean(),
                        "corrected": new[metric].mean(),
                        "difference": new[metric].mean() - old[metric].mean(),
                    }
                    for metric in ["f1-macro", "f1-pos"]
                }
                manifest["swissbert_original_comparison"] = changes
                lines += [
                    "",
                    "SwissBERT before and after corrected training on the same labels:",
                    "",
                    "| Metric | Original (%) | Corrected (%) | Change (percentage points) |",
                    "| --- | ---: | ---: | ---: |",
                ]
                for metric, values in changes.items():
                    lines.append(
                        f"| {metric} | {100*values['original']:.2f} | {100*values['corrected']:.2f} | {100*values['difference']:+.2f} |"
                    )
    lines += [
        "",
        f"Values are mean ± sample SD over the {len(args.seeds)} seeds. The summary CSV also includes accuracy, ROC-AUC, PR-AUC, 95% bootstrap intervals and the shared-only subset. Mean probabilities in the prediction export are for inspection; reported metrics are averages of per-seed metrics, not ensemble metrics.",
        "",
        "GBERT uses three epochs and maximum length 512; SwissBERT uses five epochs and maximum length 128. Both use learning rate 2e-5, FP32 and effective batch size 32 (8 × 4 accumulation steps). SwissBERT freezes embeddings and language adapters, trains with de_CH and predicts with gsw. Their established recipes were retained; this is not a comparison under identical training hyperparameters.",
        "",
        "The original SwissBERT runs had a fourfold gradient-scale defect in the installed Transformers loss handling. Those checkpoints and scores remain available separately. Corrected training is reported here; this does not establish that the loss defect caused the original low scores, because clipping/Adam can offset a uniform gradient scale and final uneven batches also affect weighting.",
        "",
        "The [SwissBERT model card](https://huggingface.co/ZurichNLP/swissbert#training-details) states that its gsw adapter was pretrained on SwissCrawl. This test set also comes from SwissCrawl, so possible exposure to unlabeled test text should be disclosed. Sentence-level pretraining overlap has not been established.",
        "",
        f"The main comparison retains all translated sentences, including unchanged/flagged inputs. Haiku has {quality['haiku']['translation_unchanged']} unchanged translations and {quality['haiku']['translation_has_anomaly']} anomaly flags; Apertus has {quality['apertus8b']['translation_unchanged']} unchanged translations and {quality['apertus8b']['translation_has_anomaly']} anomaly flags. These counts describe translation quality checks, not confirmed translation errors.",
        "",
        "[Per-seed results](by_seed.csv) · [Means, SDs and confidence intervals](summary.csv) · [All sentence scores](predictions.csv) · [Environment and checkpoint hashes](metadata.json)",
        "",
        f"All exported per-seed metrics were independently recomputed from the stored probabilities and matched their evaluation JSON files. All prediction files contain exactly one finite probability per gold sentence. Training-data hashes, hardware and package versions match across the {len(metadata)} trained models.",
        "",
    ]
    (args.path_output_dir / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    (args.path_output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote verified comparison to {args.path_output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path_runs",
        type=Path,
        default=Path("runs"),
        help="Root containing both experiment folders.",
    )
    parser.add_argument(
        "--path_annotations",
        type=Path,
        default=Path("data"),
        help="Annotation and translation directory.",
    )
    parser.add_argument(
        "--path_output_dir",
        type=Path,
        required=True,
        help="Empty output directory for the comparison.",
    )
    parser.add_argument(
        "--evaluation_date",
        required=True,
        help="Client-local completion date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Training seeds to compare.",
    )
    cmd_args = parser.parse_args()
    main(cmd_args)
