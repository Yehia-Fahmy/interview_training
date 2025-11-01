from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path(relative_path))


def read_jsonl(relative_path: str) -> Iterable[Dict[str, Any]]:
    p = data_path(relative_path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


