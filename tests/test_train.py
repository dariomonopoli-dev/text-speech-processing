import argparse
import json
import os

import numpy as np
import pandas as pd
import pytest
from transformers import XmodConfig, XmodForSequenceClassification

import delete_checkpoints
import exec_scripts
import train
from config import get_run_name, get_run_path

PATH_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_binary_metrics_on_mixed_predictions():
    metrics = train.compute_binary_metrics(
        np.array([1, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 1])
    )

    assert metrics == pytest.approx(
        {
            "acc": 0.6,
            "f1-macro": (0.5 + 2 / 3) / 2,
            "f1-pos": 0.5,
            "f1-neg": 2 / 3,
            "precision": 0.5,
            "recall": 0.5,
            "num_examples": 5,
            "num_true_hate": 2,
            "num_pred_hate": 2,
            "true_hs_ratio": 0.4,
            "pred_hs_ratio": 0.4,
        }
    )


def test_binary_metrics_keep_both_labels_when_split_has_one_class():
    metrics = train.compute_binary_metrics(
        np.zeros(4, dtype=int), np.zeros(4, dtype=int)
    )

    assert metrics["f1-neg"] == 1.0
    assert metrics["f1-pos"] == 0.0
    assert metrics["f1-macro"] == 0.5


def test_metrics_are_json_serialisable():
    metrics = train.compute_binary_metrics(np.array([1, 0]), np.array([1, 1]))

    assert json.loads(json.dumps(metrics)) == metrics


def test_run_names_end_in_the_seed():
    assert get_run_name("gahd_gbert_raw", 43) == "gahd_gbert_raw_run43"
    assert get_run_path("runs", "gahd_gbert_raw", 43) == os.path.join(
        "runs", "gahd_gbert_raw", "run43"
    )


def test_output_dir_is_created_when_missing(tmp_path):
    path_out_dir = tmp_path / "gahd_gbert_raw" / "run42"

    train.check_empty_output_dir(str(path_out_dir))

    assert path_out_dir.is_dir()


def test_non_empty_output_dir_is_refused(tmp_path):
    (tmp_path / "gahd_test_metrics.json").write_text("{}")

    with pytest.raises(FileExistsError, match="not empty"):
        train.check_empty_output_dir(str(tmp_path))


def test_non_positive_batch_size_is_refused():
    args = train.get_argparser().parse_args(
        ["--experiment_name", "x", "--batch_size", "0"]
    )

    with pytest.raises(ValueError, match="--batch_size"):
        train.check_args(args)


def test_non_positive_gradient_accumulation_is_refused():
    args = train.get_argparser().parse_args(
        ["--experiment_name", "x", "--gradient_accumulation_steps", "0"]
    )

    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train.check_args(args)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_local_config_preserves_effective_batch_size(seed):
    path_config = os.path.join(
        PATH_CONFIGS_DIR, f"gahd_gbert_raw_rtx3080_run{seed}.json"
    )
    with open(path_config) as fin:
        config = json.load(fin)

    exec_scripts.check_config_section("train", config["train"], train)
    assert (
        config["train"]["batch_size"] * config["train"]["gradient_accumulation_steps"]
        == 32
    )
    assert config["train"]["path_gahd"] == "data/gahd.csv"
    assert config["train"]["seed"] == seed


def test_missing_gahd_columns_are_listed():
    with pytest.raises(KeyError, match="gahd_id"):
        train.check_columns(
            pd.DataFrame({"text": ["a"], "label": [0]}),
            train.GAHD_COLUMNS,
            "GAHD test split",
        )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_train_config_keys_equal_train_flags(seed):
    path_config = os.path.join(PATH_CONFIGS_DIR, f"gahd_gbert_raw_run{seed}.json")
    with open(path_config) as fin:
        config = json.load(fin)

    assert config["train"].keys() == exec_scripts.get_step_dests(train)
    assert config["train"]["seed"] == seed


def test_config_section_with_missing_key_is_refused():
    section = {"experiment_name": "gahd_gbert_raw"}

    with pytest.raises(KeyError, match="seed"):
        exec_scripts.check_config_section("train", section, train)


def test_config_section_with_unknown_key_is_refused():
    section = dict.fromkeys(exec_scripts.get_step_dests(train)) | {"lr": 2e-5}

    with pytest.raises(ValueError, match="lr"):
        exec_scripts.check_config_section("train", section, train)


def write_fake_run(path_run_dir, eval_scores):
    steps = [10 * (i + 1) for i in range(len(eval_scores))]
    for step in steps:
        (path_run_dir / f"checkpoint-{step}").mkdir()
    log_history = [{"step": step, "loss": 0.5} for step in steps] + [
        {"step": step, "eval_f1-macro": score}
        for step, score in zip(steps, eval_scores)
    ]
    (path_run_dir / f"checkpoint-{steps[-1]}" / "trainer_state.json").write_text(
        json.dumps({"log_history": log_history})
    )


def test_only_best_checkpoint_survives(tmp_path):
    write_fake_run(tmp_path, [0.70, 0.80, 0.75])
    (tmp_path / "gahd_test_metrics.json").write_text("{}")

    delete_checkpoints.main(argparse.Namespace(path_run_dir=str(tmp_path)))

    assert sorted(os.listdir(tmp_path)) == ["checkpoint-20", "gahd_test_metrics.json"]
    assert delete_checkpoints.get_checkpoint_path(str(tmp_path)) == str(
        tmp_path / "checkpoint-20"
    )


def test_first_checkpoint_wins_a_tie(tmp_path):
    write_fake_run(tmp_path, [0.80, 0.80, 0.75])

    assert delete_checkpoints.get_best_checkpoint(str(tmp_path), [10, 20, 30]) == 10


def test_run_without_checkpoints_is_refused(tmp_path):
    with pytest.raises(ValueError, match="No checkpoints"):
        delete_checkpoints.main(argparse.Namespace(path_run_dir=str(tmp_path)))


def test_swissbert_load_model_freezes_adapters_and_selects_german(tmp_path):
    model = XmodForSequenceClassification(
        XmodConfig(
            vocab_size=20,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            languages=["de_CH", "gsw"],
            num_labels=2,
        )
    )
    model.save_pretrained(tmp_path)

    loaded = train.load_model(str(tmp_path), "swissbert")

    assert loaded.config.default_language == "de_CH"
    assert loaded.config.label2id == {"not_hate": 0, "hate": 1}
    assert not any(p.requires_grad for p in loaded.roberta.embeddings.parameters())
    assert not any(
        p.requires_grad
        for name, p in loaded.named_parameters()
        if ".adapter_modules." in name
    )
    assert all(p.requires_grad for p in loaded.classifier.parameters())
    assert loaded.roberta.encoder.layer[0].attention.self.query.weight.requires_grad


@pytest.fixture
def resumable_run(tmp_path):
    args = train.get_argparser().parse_args(["--experiment_name", "x", "--resume"])
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                **vars(args),
                "loss_normalization": "sum_cross_entropy_over_accumulated_examples",
            }
        )
    )
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    for name in [
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ]:
        (checkpoint / name).touch()
    return args, tmp_path, checkpoint


def test_resume_keeps_matching_checkpoint_state(resumable_run):
    args, run_path, checkpoint = resumable_run
    assert train.get_resume_checkpoint(args, str(run_path)) == str(checkpoint.resolve())


def test_resume_refuses_changed_hyperparameters(resumable_run):
    args, run_path, _ = resumable_run
    args.learning_rate = 1e-3
    with pytest.raises(ValueError, match="learning_rate"):
        train.get_resume_checkpoint(args, str(run_path))


def test_resume_refuses_missing_optimizer_state(resumable_run):
    args, run_path, checkpoint = resumable_run
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError, match="optimizer.pt"):
        train.get_resume_checkpoint(args, str(run_path))
