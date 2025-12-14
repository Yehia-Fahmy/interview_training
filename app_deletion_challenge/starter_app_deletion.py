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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, classification_report
import torch
from torch import float32, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset


# =============================================================================
# Data Loading
# =============================================================================

class DeletionModel(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.linear1 = nn.Linear(input_dim, 20)

        self.bn2 = nn.BatchNorm1d(20)
        self.linear2 = nn.Linear(20, 10)

        self.bn3 = nn.BatchNorm1d(10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(self.bn1(x))
        x = self.relu(x)

        x = self.linear2(self.bn2(x))
        x = self.relu(x)

        x = self.linear3(self.bn3(x))
        x = self.sig(x)

        return x

class AppDeletion:
    def __init__(self, data_path) -> None:
        self.dataset = self.load_data(data_path)
        self.labels = self.dataset['Label']
        self.dataset = self.dataset.drop(columns=['Label']) 

        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.categorical_features = []
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.numeric_features = []
        self.numeric_imputer = SimpleImputer(strategy="median")

        self.num_epochs = 10
        self.lr = 0.001
        self.batch_size = 32
        self.input_dim = 0
        self.opt = None
        self.loss = nn.BCELoss()

        self.train_data_loader = None
        self.test_data_loader = None

    def load_data(self, path: str = "data.csv") -> pd.DataFrame:
        """Load the dataset from CSV."""
        return pd.read_csv(path)

    def split_data(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.labels, test_size=test_size, stratify=self.labels, random_state=42)
        return X_train, X_test, y_train, y_test

    def preprocess_data(self):
        """
        Preprocess the dataset.
        """
        print("============ Start of preprocessing data ============")
        X_train, X_test, y_train, y_test = self.split_data()
        # dynamicaly figure out which features are categorical
        for idx, col in enumerate(X_train.columns):
            if X_train[col].dtype == "object":
                self.categorical_features.append(col)
            else:
                self.numeric_features.append(col)
        # Impute numeric data
        X_train[self.numeric_features] = X_train[self.numeric_features].replace('', np.nan)
        X_test[self.numeric_features] = X_test[self.numeric_features].replace('', np.nan)
        X_train[self.numeric_features] = self.numeric_imputer.fit_transform(X_train[self.numeric_features])
        X_test[self.numeric_features] = self.numeric_imputer.transform(X_test[self.numeric_features])
        
        # Impute categorical data
        X_train[self.categorical_features] = X_train[self.categorical_features].replace('', np.nan)
        X_test[self.categorical_features] = X_test[self.categorical_features].replace('', np.nan)
        X_train[self.categorical_features] = self.categorical_imputer.fit_transform(X_train[self.categorical_features])
        X_test[self.categorical_features] = self.categorical_imputer.transform(X_test[self.categorical_features])
        
        # Encode categorical data
        train_encoded_values = self.one_hot_encoder.fit_transform(X_train[self.categorical_features])
        # need to get encoded names after fitting
        self.encoded_col_names = self.one_hot_encoder.get_feature_names_out(self.categorical_features)
        train_encoded_df = pd.DataFrame(train_encoded_values, columns=self.encoded_col_names, index=X_train.index)
        X_train = X_train.drop(columns=self.categorical_features)
        X_train = pd.concat([X_train, train_encoded_df], axis=1)
        test_encoded_values = self.one_hot_encoder.transform(X_test[self.categorical_features])
        test_encoded_df = pd.DataFrame(test_encoded_values, columns=self.encoded_col_names, index=X_test.index)
        X_test = X_test.drop(columns=self.categorical_features)
        X_test = pd.concat([X_test, test_encoded_df], axis=1)
        
        self.input_dim = len(X_train.columns)
        return X_train, X_test, y_train, y_test

    def create_data_loaders(self, X_train, X_test, y_train, y_test):
        train_tensor_dataset = TensorDataset(
            torch.tensor(X_train.values, dtype=float32),
            torch.tensor(y_train.values, dtype=float32).unsqueeze(1),
        )

        test_tensor_dataset = TensorDataset(
            torch.tensor(X_test.values, dtype=float32),
            torch.tensor(y_test.values, dtype=float32).unsqueeze(1),
        )

        self.train_data_loader = DataLoader(train_tensor_dataset, self.batch_size, True)
        self.test_data_loader = DataLoader(test_tensor_dataset, self.batch_size, False)

    def prepare_model(self):
        self.model = DeletionModel(self.input_dim)

        self.opt = Adam(self.model.parameters(), self.lr)

    def train_model(self):
        """
        Train a classification model.
        
        TODO:
        - Choose and implement a model
        - Handle class imbalance if needed
        - Tune hyperparameters
        """
        # Initialize tracking metrics
        for epoch in range(1, self.num_epochs + 1):
            cumulative_loss, total_inferences, correct_inferences = 0, 0, 0
            print(f"Training Epoch #{epoch}")
            self.model.train()
            for inputs, labels in self.train_data_loader:
                # reset gradients
                self.opt.zero_grad()
                # forward pass
                out = self.model(inputs)
                predictions = (out >= 0.5).float()
                # calculate loss / metrics
                loss = self.loss(out, labels)
                cumulative_loss += loss
                total_inferences += len(labels)
                correct_inferences += (predictions == labels).sum().item()
                # backwards pass
                loss.backward()
                self.opt.step()
            # Print epoch summary
            avg_loss = cumulative_loss / len(self.train_data_loader)
            accuracy = correct_inferences / total_inferences * 100
            print(f"Epoch {epoch}/{self.num_epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")


    def evaluate_model(self):
        print("\n============ Model Evaluation ============")
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in self.test_data_loader:
                # Forward pass
                outputs = self.model(inputs)
                probabilities = outputs.squeeze().cpu().numpy()
                predictions = (outputs >= 0.5).float().squeeze().cpu().numpy()
                labels_np = labels.squeeze().cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_labels.extend(labels_np)
        
        # Convert to numpy arrays
        all_probabilities = np.array(all_probabilities)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        roc_auc = roc_auc_score(all_labels, all_probabilities)
        pr_auc = average_precision_score(all_labels, all_probabilities)
        f1 = f1_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Print results
        print(f"\nTest Set Performance:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Precision-Recall AUC: {pr_auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Retain  Delete")
        print(f"  Actual Retain  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"         Delete  {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        print(f"\nDetailed Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (Sensitivity): {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Retain', 'Delete']))
        
        print("=" * 45)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy
        }


# =============================================================================
# Main
# =============================================================================

def main():
    # Load data
    data_path = Path(__file__).parent / "data.csv"
    # Load raw data first for exploration
    raw_df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {raw_df.shape}")
    print(f"\nColumn names: {raw_df.columns.tolist()}")
    print(f"\nClass distribution:\n{raw_df['Label'].value_counts()}")
    print(f"\nMissing values:\n{raw_df.isnull().sum()}")
    
    # Now create AppDeletion object (which will drop Label column)
    app_deletion = AppDeletion(data_path)
    
    X_train, X_test, y_train, y_test = app_deletion.preprocess_data()
    app_deletion.create_data_loaders(X_train, X_test, y_train, y_test)
    app_deletion.prepare_model()
    app_deletion.train_model()
    app_deletion.evaluate_model()


if __name__ == "__main__":
    main()
