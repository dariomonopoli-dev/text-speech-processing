import argparse

import pandas as pd
import pytest

import audit_translations
from pipeline_io import PipelineError, load_csv, write_csv
from translation import compute_unchanged, flag_unchanged


@pytest.fixture
def translations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sentence_id": pd.array(["1", "2", "3"], dtype="string"),
            "text": ["das isch guet", "ok", "mir gönd hei"],
            "translation": ["das ist gut", "OK ", "wir gehen heim"],
            "translation_status": ["success", "success", "success"],
        }
    )


@pytest.fixture
def args(tmp_path, translations) -> argparse.Namespace:
    write_csv(translations, tmp_path / "translations.csv")
    return argparse.Namespace(
        path_translations=tmp_path / "translations.csv",
        path_output=tmp_path / "audited.csv",
        overwrite=False,
    )


def test_compute_unchanged_ignores_case_and_surrounding_spaces(translations):
    assert compute_unchanged(translations).tolist() == [False, True, False]


def test_compute_unchanged_is_false_for_a_missing_translation(translations):
    translations.loc[0, "translation"] = None

    assert compute_unchanged(translations).tolist() == [False, True, False]


def test_flag_unchanged_marks_the_row_for_manual_review():
    audited = pd.DataFrame(
        {
            "text": ["ok", "das isch guet"],
            "translation": ["ok", "das ist gut"],
            "translation_has_anomaly": [False, False],
            "translation_status": ["success", "success"],
        }
    )

    flagged = flag_unchanged(audited)

    assert flagged["translation_status"].tolist() == ["manual_review", "success"]
    assert flagged["translation_has_anomaly"].tolist() == [True, False]


def test_audit_translations_keeps_the_translations_unchanged(translations):
    audited = audit_translations.audit_translations(translations)

    assert audited["translation"].tolist() == translations["translation"].tolist()
    assert audited["translation_unchanged"].tolist() == [False, True, False]
    assert "translation_input_tokens" not in audited.columns


def test_main_writes_a_new_file_and_leaves_the_input_untouched(args):
    original = args.path_translations.read_bytes()

    audit_translations.main(args)

    assert args.path_translations.read_bytes() == original
    audited = load_csv(args.path_output)
    assert audited["translation_unchanged"].sum() == 1


def test_main_refuses_to_overwrite_the_input_file(args):
    args.path_output = args.path_translations
    args.overwrite = True

    with pytest.raises(PipelineError, match="different file"):
        audit_translations.main(args)


def test_main_refuses_an_existing_output_without_overwrite(args):
    args.path_output.write_text("old")

    with pytest.raises(PipelineError, match="--overwrite"):
        audit_translations.main(args)


def test_main_rejects_a_file_without_the_translation_column(args, translations):
    write_csv(translations.drop(columns="translation"), args.path_translations)

    with pytest.raises(KeyError, match="translation"):
        audit_translations.main(args)
