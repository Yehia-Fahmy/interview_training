"""
Baseline Solution for Exercise 5: PyTorch Deep Learning Pipeline

This is a simple, naive implementation that serves as a baseline.
It works but lacks many best practices and optimizations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class BaselineDataset(Dataset):
    """Simple baseline dataset - minimal implementation"""
    
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        # Simple scaling - fit on this data (data leakage for test, but baseline)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(features.astype(np.float32))
        self.targets = targets.astype(np.float32) if targets is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.targets is not None:
            return features, torch.tensor(self.targets[idx], dtype=torch.float32)
        return features,


class BaselineModel(nn.Module):
    """Simple baseline model - single hidden layer, no regularization"""
    
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def baseline_train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    """Baseline training and evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets (with data leakage - baseline doesn't prevent it)
    train_dataset = BaselineDataset(X_train, y_train)
    val_dataset = BaselineDataset(X_val, y_val)
    test_dataset = BaselineDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Simple model
    model = BaselineModel(X_train.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Simple training - no early stopping, no validation tracking
    model.train()
    for epoch in range(20):  # Fixed epochs
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features).squeeze().cpu().numpy()
            test_predictions.extend(outputs)
            test_targets.extend(targets.numpy())
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    mae = mean_absolute_error(test_targets, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    r2 = r2_score(test_targets, test_predictions)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'test_loss': mean_squared_error(test_targets, test_predictions)
    }

