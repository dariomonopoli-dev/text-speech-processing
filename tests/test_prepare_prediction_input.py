import argparse

import pandas as pd
import pytest

import prepare_prediction_input
from pipeline_io import PipelineError, write_csv


@pytest.fixture
def args(tmp_path) -> argparse.Namespace:
    write_csv(
        pd.DataFrame(
            {
                "sentence_id": pd.array(["11", "12"], dtype="string"),
                "text": ["das isch guet", "mir gönd hei"],
                "translation": ["das ist gut", "wir gehen heim"],
                "translation_status": ["success", "success"],
            }
        ),
        tmp_path / "translations.csv",
    )
    return argparse.Namespace(
        path_translations=tmp_path / "translations.csv",
        input_column="translation_haiku",
        path_output=tmp_path / "pred_input.csv",
        overwrite=False,
    )


def test_main_keeps_only_the_ids_and_the_renamed_translation_column(args):
    prepare_prediction_input.main(args)

    written = pd.read_csv(args.path_output)
    assert list(written.columns) == ["sentence_id", "translation_haiku"]
    assert written["translation_haiku"].tolist() == ["das ist gut", "wir gehen heim"]


def test_main_writes_the_file_without_a_byte_order_mark(args):
    prepare_prediction_input.main(args)

    assert not args.path_output.read_bytes().startswith(b"\xef\xbb\xbf")


def test_main_rejects_a_missing_translation(args, tmp_path):
    write_csv(
        pd.DataFrame({"sentence_id": ["11", "12"], "translation": ["gut", " "]}),
        args.path_translations,
    )

    with pytest.raises(ValueError, match="1 rows without a translation"):
        prepare_prediction_input.main(args)


def test_main_rejects_duplicate_sentence_ids(args):
    write_csv(
        pd.DataFrame({"sentence_id": ["11", "11"], "translation": ["a", "b"]}),
        args.path_translations,
    )

    with pytest.raises(ValueError, match="unique"):
        prepare_prediction_input.main(args)


def test_main_rejects_a_file_without_the_translation_column(args):
    write_csv(pd.DataFrame({"sentence_id": ["11"]}), args.path_translations)

    with pytest.raises(KeyError, match="translation"):
        prepare_prediction_input.main(args)


def test_main_refuses_to_overwrite_the_input_file(args):
    args.path_output = args.path_translations
    args.overwrite = True

    with pytest.raises(PipelineError, match="different file"):
        prepare_prediction_input.main(args)


def test_main_refuses_an_existing_output_without_overwrite(args):
    args.path_output.write_text("old")

    with pytest.raises(PipelineError, match="--overwrite"):
        prepare_prediction_input.main(args)
