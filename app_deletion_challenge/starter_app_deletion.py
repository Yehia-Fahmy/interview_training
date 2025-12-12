"""
Starter Code: App Deletion Risk Prediction

This is minimal starter code for the ML interview challenge.
Your task is to build a binary classification model that predicts
whether a user will delete the app within the next n weeks.

See challenge_app_deletion.md for full problem description.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path: str = "data.csv") -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(path)


# =============================================================================
# TODO: Implement your solution below
# =============================================================================

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset.
    
    TODO:
    - Handle missing values
    - Encode categorical variables
    - Scale/normalize numeric features
    - Split features and target
    """
    pass


def train_model(X_train, y_train):
    """
    Train a classification model.
    
    TODO:
    - Choose and implement a model
    - Handle class imbalance if needed
    - Tune hyperparameters
    """
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model.
    
    TODO:
    - Calculate appropriate metrics (ROC-AUC, F1, etc.)
    - Generate confusion matrix
    - Analyze results
    """
    pass


# =============================================================================
# Main
# =============================================================================

def main():
    # Load data
    data_path = Path(__file__).parent / "data.csv"
    df = load_data(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nClass distribution:\n{df['Label'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # TODO: Implement your solution
    # 1. Preprocess data
    # 2. Split into train/test
    # 3. Train model
    # 4. Evaluate model


if __name__ == "__main__":
    main()
