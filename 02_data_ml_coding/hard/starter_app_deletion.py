"""
Minimal starter for the App Deletion Risk Prediction challenge.

You are expected to implement the entire pipeline yourself. This file only
defines the high-level scaffolding so you have consistent entry points for
experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
# from PIL.TiffImagePlugin import idx
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


@dataclass
class ExperimentConfig:
    """Holds the knobs you'll likely need while iterating."""

    data_path: Path = Path("data/data_small.csv")
    seed: int = 42
    batch_size: int = 256
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: Optional[str] = None


def resolve_device(preferred: Optional[str] = None) -> str:
    """Pick the best available accelerator: CUDA -> Apple MPS -> CPU."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    mps_is_available = getattr(torch.backends, "mps", None)
    if mps_is_available and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="App deletion risk training entry point.")
    parser.add_argument("--data-path", type=Path, default=ExperimentConfig.data_path)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.max_epochs)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Override automatic device selection (default: auto-detect).",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        data_path=args.data_path,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )


def load_snapshot_table(path: Path) -> Any:
    return pd.read_csv(path)


def train_val_test_split(raw_table: Any, seed: int) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    X_values = raw_table.drop(columns=['label', 'user_id'])
    Y_values = raw_table['label']
    groups = raw_table['user_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, test_indices = next(gss.split(X_values, Y_values, groups=groups))

    X_test_pd = X_values.iloc[test_indices]
    y_test_pd = Y_values.iloc[test_indices]
    groups_test = groups.iloc[test_indices]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    test_indices, val_indices = next(gss2.split(X_test_pd, y_test_pd, groups=groups_test))

    X_train_pd = X_values.iloc[train_indices]
    y_train_pd = Y_values.iloc[train_indices]

    X_test_pd = X_values.iloc[test_indices]
    y_test_pd = Y_values.iloc[test_indices]

    X_val_pd = X_values.iloc[val_indices]
    y_val_pd = Y_values.iloc[val_indices]

    return ((X_train_pd, y_train_pd), (X_test_pd, y_test_pd), (X_val_pd, y_val_pd))

class CustomDataset(Dataset):
    def __init__(self, X: pd.DataFrame, Y: pd.Series,
                feature_scaler=None, categorical_encoders=None,
                numeric_imputer=None, categorical_imputer=None):
        self.X = X.copy()
        self.Y = Y.values.astype(np.float32)
        self.feature_scaler = feature_scaler
        self.categorical_encoders = categorical_encoders or {}
        self.numeric_imputer = numeric_imputer
        self.categorical_imputer = categorical_imputer

        self.numeric_cols = X.select_dtypes([np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(['object', 'category']).columns.tolist()
        self.total_features = len(self.numeric_cols) + len(self.categorical_cols)

        self._impute_missing_values()

        self._process_features()

    def _impute_missing_values(self):
        if self.numeric_cols:
            X_numeric = self.X[self.numeric_cols]

            if self.numeric_imputer:
                X_numeric_imputed = self.numeric_imputer.transform(X_numeric)
            else:
                self.numeric_imputer = SimpleImputer(strategy='median')
                X_numeric_imputed = self.numeric_imputer.fit_transform(X_numeric)
            
            self.X[self.numeric_cols] = X_numeric_imputed

        if self.categorical_cols:
            X_categorical = self.X[self.categorical_cols]

            if self.categorical_imputer:
                X_categorical_imputed = self.categorical_imputer.transform(X_categorical)
            else:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                X_categorical_imputed = self.categorical_imputer.fit_transform(X_categorical)

            X_cat_df = pd.DataFrame(X_categorical_imputed, columns=self.categorical_cols, index=self.X.index)
            self.X[self.categorical_cols] = X_cat_df

    def _process_features(self):
        processed_features = []

        if self.numeric_cols:
            X_numeric = self.X[self.numeric_cols].values.astype(np.float32)

            if self.feature_scaler:
                X_numeric = self.feature_scaler.transform(X_numeric)
            else:
                self.feature_scaler = StandardScaler()
                X_numeric = self.feature_scaler.fit_transform(X_numeric)

            processed_features.append(X_numeric)

        if self.categorical_cols:
            X_categorical = []
            for col in self.categorical_encoders:
                if col in self.categorical_cols:
                    encoder = self.categorical_encoders[col]
                    encoded_values = encoder.transform(self.X[col].values).reshape(-1, 1)
                else:
                    encoder = LabelEncoder()
                    encoded_values = encoder.fit_transform(self.X[col].values).reshape(-1, 1)
                    self.categorical_encoders[col] = encoder
                X_categorical.append(encoded_values.astype(np.float32))
            
            if X_categorical:
                processed_features.append(np.hstack(X_categorical))

        self.features = np.hstack(processed_features).astype(np.float32)
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return {
            'features': torch.tensor(self.features[index]),
            'label': torch.tensor(self.Y[index])
        }

def build_datasets(
    train_data: Any, test_data: Any, val_data: Any
) -> Dict[str, torch.utils.data.Dataset]:
    (X_train_pd, y_train_pd) = train_data
    (X_test_pd, y_test_pd) = test_data
    (X_val_pd, y_val_pd) = val_data

    train_dataset = CustomDataset(X_train_pd, y_train_pd)
    test_dataset = CustomDataset(X_test_pd, y_test_pd, train_dataset.feature_scaler,
        train_dataset.categorical_encoders, train_dataset.numeric_imputer, train_dataset.categorical_imputer)
    val_dataset = CustomDataset(X_val_pd, y_val_pd, train_dataset.feature_scaler, train_dataset.categorical_encoders,
        train_dataset.numeric_imputer, train_dataset.categorical_imputer)
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset,
    }


class DeletionModel(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(features))
        x = self.dropout(x)
        x = self.relu(self.fc3(features))
        return self.sigmoid(x)

def train_loop(
    model: DeletionModel,
    datasets: Dict[str, torch.utils.data.Dataset],
    config: ExperimentConfig,
) -> Dict[str, float]:
    """
    TODO: implement everything needed for training:
      * DataLoader creation
      * Loss / metrics configuration
      * Optimization loop (or Lightning Trainer)
      * Checkpointing + logging
    Return any metrics you want to log for monitoring.
    """
    train_dataloader = DataLoader(datasets['train'], config.batch_size, shuffle=True)
    test_dataloader = DataLoader(datasets['test'], config.batch_size, shuffle=True)
    val_dataloader = DataLoader(datasets['val'], config.batch_size, shuffle=True)
    model.to(config.device)
    optimizer = Adam(model.parameters(), config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(config.max_epochs):
        # train
        model.train()
        correct, total = 0, 0
        for batch in train_dataloader:
            features, labels = batch['features'].to(config.device), batch['label'].to(config.device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels.float())
            loss.backward()
            optimizer.step()
            total += len(labels)
            correct += np.sum([output == labels.float()])
        train_acc = correct / total
        print(f"Training accuracy for epoch #{epoch+1}: {train_acc}")

        # test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_dataloader:
                features, labels = batch['features'].to(config.device), batch['label'].to(config.device)
                output = model(features)
                total += len(labels)
                correct += np.sum([output == labels.float()])
            test_acc = correct / total
            print(f"Testing accuracy for epoch #{epoch+1}: {test_acc}")

    # TODO sperform validation



def main() -> None:
    config = parse_args()
    config.device = resolve_device(config.device)
    print(f"Starting experiment with config: {config}")

    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    raw_table = load_snapshot_table(config.data_path)
    train_part, val_part, test_part = train_val_test_split(raw_table, config.seed)
    datasets = build_datasets(train_part, val_part, test_part)

    model = DeletionModel(datasets['test'].total_features).to(config.device)
    metrics = train_loop(model, datasets, config)
    print(f"Finished training. Metrics: {metrics}")


if __name__ == "__main__":
    main()

