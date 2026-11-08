import argparse

import pandas as pd
import pytest

import build_eval_sentences
from pipeline_io import PipelineError

ANNOTATION_ROWS = {
    "dario": [(1, "eins"), (2, "zwei"), (3, "drei")],
    "marjorie": [(1, "eins"), (2, "zwei"), (4, "vier")],
    "yannis": [(1, "eins"), (2, "zwei"), (5, "fünf")],
}


@pytest.fixture
def args(tmp_path) -> argparse.Namespace:
    for annotator, rows in ANNOTATION_ROWS.items():
        pd.DataFrame(rows, columns=["sentence_id", "text"]).assign(label=0).to_csv(
            tmp_path / f"annotation_{annotator}_stratified.csv", index=False
        )
    return argparse.Namespace(
        path_annotations=tmp_path,
        path_output=tmp_path / "eval_sentences.csv",
        overwrite=False,
    )


def test_load_annotated_sentences_returns_each_sentence_once(args):
    sentences = build_eval_sentences.load_annotated_sentences(args.path_annotations)

    assert sentences["sentence_id"].tolist() == ["1", "2", "3", "4", "5"]
    assert sentences["text"].tolist() == ["eins", "zwei", "drei", "vier", "fünf"]


def test_load_annotated_sentences_rejects_conflicting_texts(args, tmp_path):
    pd.DataFrame([(1, "anders")], columns=["sentence_id", "text"]).assign(label=0).to_csv(
        tmp_path / "annotation_yannis_stratified.csv", index=False
    )

    with pytest.raises(ValueError, match="different texts"):
        build_eval_sentences.load_annotated_sentences(args.path_annotations)


def test_load_annotated_sentences_names_a_missing_file(args, tmp_path):
    (tmp_path / "annotation_yannis_stratified.csv").unlink()

    with pytest.raises(FileNotFoundError, match="yannis"):
        build_eval_sentences.load_annotated_sentences(args.path_annotations)


def test_load_annotated_sentences_names_a_missing_column(args, tmp_path):
    pd.DataFrame({"sentence_id": [1]}).to_csv(
        tmp_path / "annotation_dario_stratified.csv", index=False
    )

    with pytest.raises(KeyError, match="text"):
        build_eval_sentences.load_annotated_sentences(args.path_annotations)


def test_main_writes_the_sentences(args):
    build_eval_sentences.main(args)

    written = pd.read_csv(args.path_output)
    assert list(written.columns) == ["sentence_id", "text"]
    assert len(written) == 5


def test_main_refuses_an_existing_output_without_overwrite(args):
    args.path_output.write_text("old")

    with pytest.raises(PipelineError, match="--overwrite"):
        build_eval_sentences.main(args)
