from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, data: Any) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return output


def write_text(path: str | Path, text: str) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return output


def write_frame(path: str | Path, frame: pd.DataFrame) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    frame.to_csv(output, index=False)
    return output


def read_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
