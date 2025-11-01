"""Baseline text classification pipeline template for the foundational ML exercise."""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class ExperimentConfig:
    data_path: pathlib.Path
    report_path: pathlib.Path
    random_seed: int = 8090


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    """Load raw data. Replace with dataset-specific logic."""

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    texts = df["text"].astype(str).str.lower()
    labels = df["label"].astype(str).values
    return texts.values, labels


def train_model(texts: np.ndarray, labels: np.ndarray, seed: int) -> Dict[str, object]:
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=5_000, n_jobs=-1)
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "report": report,
        "confusion_matrix": matrix,
        "labels": np.unique(labels).tolist(),
    }


def save_report(artifacts: Dict[str, object], path: pathlib.Path) -> None:
    payload = {
        "metrics": artifacts["report"],
        "labels": artifacts["labels"],
        "confusion_matrix": artifacts["confusion_matrix"].tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline ticket triage classifier")
    parser.add_argument("data", type=pathlib.Path, help="Path to CSV dataset")
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("report.json"))
    parser.add_argument("--seed", type=int, default=8090)
    args = parser.parse_args()

    cfg = ExperimentConfig(data_path=args.data, report_path=args.report, random_seed=args.seed)
    df = load_dataset(cfg.data_path)
    texts, labels = preprocess(df)
    artifacts = train_model(texts, labels, cfg.random_seed)
    save_report(artifacts, cfg.report_path)


if __name__ == "__main__":
    main()

