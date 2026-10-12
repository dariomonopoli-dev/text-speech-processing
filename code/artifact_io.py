"""Atomic artifact writers and spreadsheet integrity checks."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def verify_annotation_workbooks(
    annotations: Mapping[str, pd.DataFrame],
    workbook_paths: Mapping[str, Path],
) -> None:
    """Require every annotator workbook to match its canonical CSV frame."""
    from openpyxl import load_workbook

    if set(workbook_paths) != set(annotations):
        raise ValueError("Workbook annotators do not match annotation frames.")
    for annotator, frame in annotations.items():
        path = Path(workbook_paths[annotator])
        if not path.is_file():
            raise ValueError(f"{annotator} annotation workbook does not exist: {path}")
        sheet = load_workbook(path, read_only=True, data_only=False).active
        rows = list(sheet.iter_rows())
        if len(rows) != len(frame) + 1:
            raise ValueError(f"{annotator} workbook row count does not match its CSV.")
        actual_headers = [cell.value for cell in rows[0]]
        if actual_headers != frame.columns.tolist():
            raise ValueError(f"{annotator} workbook columns do not match its CSV.")
        expected_rows = frame.fillna("").astype(str).itertuples(index=False, name=None)
        for row_number, (cells, expected) in enumerate(
            zip(rows[1:], expected_rows, strict=True), start=2
        ):
            if any(cell.data_type == "f" for cell in cells):
                raise ValueError(
                    f"{annotator} workbook contains a formula in row {row_number}."
                )
            actual = tuple(
                "" if cell.value is None else str(cell.value) for cell in cells
            )
            if actual != expected:
                raise ValueError(
                    f"{annotator} workbook row {row_number} does not match its CSV."
                )


def _atomic_text_path(path: Path) -> tuple[Any, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace a UTF-8 CSV only after its complete content reaches disk."""
    target = Path(path)
    handle, temporary = _atomic_text_path(target)
    try:
        with handle:
            handle.write("\ufeff")
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_xlsx(frame: pd.DataFrame, path: Path) -> None:
    """Write a workbook whose text cells cannot be interpreted as formulas."""
    from openpyxl import Workbook

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".tmp.xlsx",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "annotations"
        sheet.append([str(column) for column in frame.columns])
        for row in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(value) else value for value in row])
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    cell.data_type = "s"
        workbook.save(temporary)
        with temporary.open("rb") as saved:
            os.fsync(saved.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Replace a JSON file only after its complete content reaches disk."""
    target = Path(path)
    handle, temporary = _atomic_text_path(target)
    try:
        with handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
