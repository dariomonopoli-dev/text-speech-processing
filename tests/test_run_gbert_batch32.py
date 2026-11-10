import argparse
import json

import pytest

import run_gbert_batch32


def test_existing_control_run_is_refused_before_model_loading(tmp_path, monkeypatch):
    path_configs = tmp_path / "configs"
    path_configs.mkdir()
    (path_configs / "gahd_gbert_raw_rtx3080_run42.json").write_text(
        json.dumps({"train": {"model_name": "unused"}}), encoding="utf-8"
    )
    path_run = tmp_path / "runs" / "gahd_gbert_batch32_control" / "run42"
    path_run.mkdir(parents=True)
    path_saved = path_run / "experiment.json"
    path_saved.write_text("existing result", encoding="utf-8")

    def fail_loading(*args, **kwargs):
        pytest.fail("Existing run must be refused before data or model loading")

    monkeypatch.setattr(run_gbert_batch32, "load_gahd", fail_loading)
    args = argparse.Namespace(
        seeds=[42],
        path_configs=path_configs,
        path_runs=tmp_path / "runs",
        path_data=tmp_path / "data",
    )
    with pytest.raises(FileExistsError):
        run_gbert_batch32.main(args)
    assert path_saved.read_text(encoding="utf-8") == "existing result"


@pytest.mark.parametrize("seeds", [[], [42, 42]])
def test_invalid_seeds_fail_before_reading_inputs(seeds):
    with pytest.raises(ValueError, match="distinct training seeds"):
        run_gbert_batch32.main(argparse.Namespace(seeds=seeds))
