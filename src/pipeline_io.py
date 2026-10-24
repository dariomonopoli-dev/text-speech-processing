"""File helpers shared by the SwissCrawl test-set steps."""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


class PipelineError(RuntimeError):
    pass


def check_inputs_exist(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise PipelineError("Required input does not exist: " + ", ".join(missing))


def check_outputs_absent(paths: Sequence[Path], resume: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not resume:
        raise PipelineError(
            "Output already exists; pass --resume to reuse or replace it: "
            + ", ".join(existing)
        )


def check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise PipelineError(f"--{name} must be positive.")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sentence_id": "string"})


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fin:
        return json.load(fin)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write JSON atomically, so an interrupted run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as fout:
        fout.write(content)
        fout.flush()
        os.fsync(fout.fileno())
        path_temporary = Path(fout.name)
    path_temporary.replace(path)


def write_xlsx(frame: pd.DataFrame, path: Path) -> None:
    from openpyxl import Workbook

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "annotations"
    sheet.append(frame.columns.tolist())
    for values in frame.itertuples(index=False, name=None):
        sheet.append([None if pd.isna(value) else value for value in values])
    # openpyxl stores strings starting with "=" as formulas unless forced to text.
    for row in sheet.iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
    workbook.save(path)
