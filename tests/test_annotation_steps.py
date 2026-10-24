import argparse

import pytest

import build_annotation_files
import verify_testset
from annotation_assignment import (
    assign_annotations,
    get_annotation_paths,
    get_workbook_paths,
)
from pipeline_io import PipelineError, load_csv, load_json, write_csv
from sampling import sample_fixed_strata
from translation import add_translations


@pytest.fixture
def path_master(tmp_path, scored_candidates):
    sample = sample_fixed_strata(scored_candidates, seed=42)
    master = add_translations(sample, sample["text"].tolist(), attempts=1)
    write_csv(master, tmp_path / "master.csv")
    return tmp_path / "master.csv"


@pytest.fixture
def annotation_args(tmp_path, path_master) -> argparse.Namespace:
    return argparse.Namespace(
        path_master=path_master,
        path_annotation_dir=tmp_path / "annotations",
        seed=42,
        resume=False,
    )


@pytest.fixture
def needs_openpyxl():
    pytest.importorskip("openpyxl")


@pytest.fixture
def verify_args(tmp_path, path_master) -> argparse.Namespace:
    path_annotations = get_annotation_paths(tmp_path / "annotations")
    for annotator, annotation_set in assign_annotations(load_csv(path_master)).items():
        write_csv(annotation_set, path_annotations[annotator])
    return argparse.Namespace(
        path_master=path_master,
        path_annotation_dir=tmp_path / "annotations",
        path_verification_report=tmp_path / "verification_report.json",
        resume=False,
    )


def test_annotation_step_writes_csv_and_xlsx_per_annotator(
    annotation_args, needs_openpyxl
):
    build_annotation_files.main(annotation_args)

    path_annotations = get_annotation_paths(annotation_args.path_annotation_dir)
    path_workbooks = get_workbook_paths(annotation_args.path_annotation_dir)
    assert all(len(load_csv(path)) == 300 for path in path_annotations.values())
    assert all(path.is_file() for path in path_workbooks.values())


def test_annotation_step_refuses_to_overwrite_without_resume(
    annotation_args, needs_openpyxl
):
    build_annotation_files.main(annotation_args)

    with pytest.raises(PipelineError, match="pass --resume"):
        build_annotation_files.main(annotation_args)


def test_annotation_step_with_resume_preserves_existing_files(
    annotation_args, needs_openpyxl
):
    path_dario = get_annotation_paths(annotation_args.path_annotation_dir)["dario"]
    path_dario.parent.mkdir(parents=True)
    path_dario.write_text("labelled by hand\n")
    annotation_args.resume = True

    build_annotation_files.main(annotation_args)

    assert path_dario.read_text() == "labelled by hand\n"
    assert get_workbook_paths(annotation_args.path_annotation_dir)["dario"].is_file()


def test_verify_step_writes_a_passing_report(verify_args):
    verify_testset.main(verify_args)

    report = load_json(verify_args.path_verification_report)
    assert report["status"] == "passed"
    assert report["single_annotator_counts"] == {
        "dario": 200,
        "marjorie": 200,
        "yannis": 200,
    }


def test_verify_step_requires_every_annotation_file(verify_args):
    get_annotation_paths(verify_args.path_annotation_dir)["yannis"].unlink()

    with pytest.raises(PipelineError, match="annotation_yannis_stratified.csv"):
        verify_testset.main(verify_args)


def test_verify_step_refuses_to_overwrite_without_resume(verify_args):
    verify_testset.main(verify_args)

    with pytest.raises(PipelineError, match="pass --resume"):
        verify_testset.main(verify_args)
