"""Placeholder tests for retrieval module."""

import pathlib

from data_ml_coding.intermediate.pipeline import retrieve


def test_retrieve_returns_list(tmp_path: pathlib.Path) -> None:
    kb_path = tmp_path / "kb.json"
    kb_path.write_text("[]")
    results = retrieve("sample query", k=3, kb_path=kb_path)
    assert isinstance(results, list)

