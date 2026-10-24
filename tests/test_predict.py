import json
import os

import pandas as pd
import pytest

import exec_scripts
import predict

PATH_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
RUN_CONFIGS = [
    "gahd_gbert_raw_run42.json",
    "gahd_gbert_raw_run43.json",
    "gahd_gbert_raw_run44.json",
]


@pytest.mark.parametrize("config_name", RUN_CONFIGS)
def test_predict_config_keys_equal_predict_flags(config_name):
    with open(os.path.join(PATH_CONFIGS_DIR, config_name)) as fin:
        config = json.load(fin)
    predict_sections = [config[key] for key in config if key.startswith("predict")]

    assert [section["input_column"] for section in predict_sections] == [
        "text",
        "translation",
    ]
    for section in predict_sections:
        assert section.keys() == exec_scripts.get_step_dests(predict)
        assert section["seed"] == config["train"]["seed"]
        assert section["experiment_name"] == config["train"]["experiment_name"]


def test_predictions_file_is_named_after_the_input_column():
    assert predict.get_predictions_path("runs/gahd_gbert_raw/run42", "translation") == (
        os.path.join(
            "runs/gahd_gbert_raw/run42", "predictions_swiss_german_translation.csv"
        )
    )


def test_explicit_checkpoint_overrides_the_run_checkpoint(tmp_path):
    assert predict.get_model_path(str(tmp_path), "runs/none") == str(tmp_path)


def test_missing_explicit_checkpoint_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        predict.get_model_path(str(tmp_path / "nope"), "runs/none")


def test_eval_set_keeps_id_and_input_column(tmp_path):
    path_eval_set = tmp_path / "master.csv"
    pd.DataFrame(
        {"text": ["grüezi"], "sentence_id": [7], "translation": ["hallo"], "url": ["u"]}
    ).to_csv(path_eval_set, index=False)

    eval_set = predict.load_eval_set(str(path_eval_set), "translation")

    assert eval_set.to_dict("records") == [{"sentence_id": 7, "translation": "hallo"}]


def test_missing_eval_set_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        predict.load_eval_set(str(tmp_path / "missing.csv"), "text")


def test_eval_set_without_input_column_is_refused(tmp_path):
    path_eval_set = tmp_path / "master.csv"
    pd.DataFrame({"text": ["grüezi"], "sentence_id": [7]}).to_csv(
        path_eval_set, index=False
    )

    with pytest.raises(KeyError, match="translation"):
        predict.load_eval_set(str(path_eval_set), "translation")


def test_eval_set_with_empty_texts_is_refused(tmp_path):
    path_eval_set = tmp_path / "master.csv"
    pd.DataFrame({"text": ["grüezi", None], "sentence_id": [7, 8]}).to_csv(
        path_eval_set, index=False
    )

    with pytest.raises(ValueError, match="1 rows"):
        predict.load_eval_set(str(path_eval_set), "text")
