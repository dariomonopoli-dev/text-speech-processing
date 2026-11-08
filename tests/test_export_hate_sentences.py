import argparse

import pandas as pd
import pytest

import export_hate_sentences

SENTENCE_IDS = [1, 2, 3, 4, 5]
HATE_IDS = {2, 4}
VARIANT_OFFSETS = {"text": 0.0, "translation_haiku": 0.2, "translation_apertus8b": 0.1}


@pytest.fixture
def args(tmp_path) -> argparse.Namespace:
    for annotator in ["dario", "marjorie", "yannis"]:
        pd.DataFrame(
            {
                "sentence_id": SENTENCE_IDS,
                "label": [int(i in HATE_IDS) for i in SENTENCE_IDS],
            }
        ).to_csv(tmp_path / f"annotation_{annotator}_stratified.csv", index=False)

    texts = [f"satz {i}" for i in SENTENCE_IDS]
    pd.DataFrame(
        {
            "sentence_id": SENTENCE_IDS,
            "text": texts,
            "translation": [f"haiku {i}" for i in SENTENCE_IDS],
        }
    ).to_csv(tmp_path / "haiku.csv", index=False)
    pd.DataFrame(
        {
            "sentence_id": SENTENCE_IDS,
            "text": texts,
            "translation": [f"apertus {i}" for i in SENTENCE_IDS],
        }
    ).to_csv(tmp_path / "apertus.csv", index=False)

    for seed in [42, 43]:
        run_path = tmp_path / "runs" / f"run{seed}"
        run_path.mkdir(parents=True)
        for variant, offset in VARIANT_OFFSETS.items():
            seed_shift = 0.0 if seed == 42 else 0.1
            pd.DataFrame(
                {
                    "sentence_id": SENTENCE_IDS,
                    "prob_hate": [i / 10 + offset + seed_shift for i in SENTENCE_IDS],
                }
            ).to_csv(run_path / f"predictions_swiss_german_{variant}.csv", index=False)

    return argparse.Namespace(
        path_annotations=tmp_path,
        path_haiku=tmp_path / "haiku.csv",
        path_apertus=tmp_path / "apertus.csv",
        path_runs=tmp_path / "runs",
        score_column="prob_hate",
        subset="gold_hate",
        path_output=tmp_path / "hate.csv",
        overwrite=False,
    )


def test_main_exports_only_the_gold_hate_sentences_with_all_columns(args):
    export_hate_sentences.main(args)

    exported = pd.read_csv(args.path_output, encoding="utf-8-sig")
    assert list(exported.columns) == [
        "sentence_id",
        "text",
        "gold_label",
        "translation_haiku",
        "translation_apertus8b",
        "mean_prob_hate_raw",
        "mean_prob_hate_apertus8b",
        "mean_prob_hate_haiku",
    ]
    assert exported["sentence_id"].tolist() == [2, 4]
    assert exported["gold_label"].tolist() == [1, 1]
    assert exported["translation_haiku"].tolist() == ["haiku 2", "haiku 4"]
    assert exported["translation_apertus8b"].tolist() == ["apertus 2", "apertus 4"]


def test_main_exports_every_sentence_with_its_gold_label(args):
    args.subset = "all"

    export_hate_sentences.main(args)

    exported = pd.read_csv(args.path_output, encoding="utf-8-sig")
    assert exported["sentence_id"].tolist() == SENTENCE_IDS
    assert exported["gold_label"].tolist() == [0, 1, 0, 1, 0]


def test_main_averages_the_scores_over_the_seeds(args):
    export_hate_sentences.main(args)

    exported = pd.read_csv(args.path_output, encoding="utf-8-sig")
    assert exported["mean_prob_hate_raw"].tolist() == pytest.approx([0.25, 0.45])
    assert exported["mean_prob_hate_apertus8b"].tolist() == pytest.approx([0.35, 0.55])
    assert exported["mean_prob_hate_haiku"].tolist() == pytest.approx([0.45, 0.65])


def test_main_rejects_different_texts_in_the_two_translation_files(args, tmp_path):
    haiku = pd.read_csv(args.path_haiku)
    haiku.loc[haiku["sentence_id"] == 2, "text"] = "anderer satz"
    haiku.to_csv(args.path_haiku, index=False)

    with pytest.raises(ValueError, match="different texts"):
        export_hate_sentences.main(args)


def test_main_names_a_hate_sentence_missing_from_a_translation_file(args):
    apertus = pd.read_csv(args.path_apertus)
    apertus[apertus["sentence_id"] != 4].to_csv(args.path_apertus, index=False)

    with pytest.raises(KeyError, match="4"):
        export_hate_sentences.main(args)


def test_main_names_a_translation_file_without_the_translation_column(args):
    pd.read_csv(args.path_haiku).drop(columns="translation").to_csv(
        args.path_haiku, index=False
    )

    with pytest.raises(KeyError, match="translation"):
        export_hate_sentences.main(args)


def test_main_refuses_an_existing_output_without_overwrite(args):
    args.path_output.write_text("old")

    with pytest.raises(FileExistsError, match="--overwrite"):
        export_hate_sentences.main(args)
