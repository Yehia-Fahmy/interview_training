"""
Solution for Exercise 5: PyTorch Deep Learning Pipeline

This file contains the reference solution.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 89
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class CustomDataset(Dataset):
    """Custom PyTorch Dataset for tabular data."""
    
    def __init__(self, 
                 features: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = True):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        
        # Handle scaling
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided when fit_scaler=False")
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return features, target
        else:
            return features,
    
    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler


class NeuralNetwork(nn.Module):
    """Custom neural network architecture."""
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64, 32],
                 output_size: int = 1,
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 task_type: str = 'regression'):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.task_type = task_type
        
        # Activation function mapping
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        self.layers = nn.ModuleList()
        
        # Input to first hidden layer
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            self.layers.append(self.activation)
            
            # Dropout (except for last layer)
            if i < len(hidden_sizes) - 1 or task_type == 'classification':
                self.layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Output activation based on task type
        if task_type == 'classification':
            if output_size == 1:
                self.output_activation = nn.Sigmoid()
            else:
                self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.output_activation(x)
        
        return x


class ModelTrainer:
    """Training loop and model management."""
    
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: Optional[torch.device] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(dataloader):
            features, targets = batch
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            
            # Handle different output shapes
            if outputs.dim() > 1 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              early_stopping_patience: int = 10,
              verbose: bool = True) -> Dict[str, List[float]]:
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Val R2: {val_metrics.get('r2', 0):.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def _compute_metrics(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, float]:
        # Concatenate all predictions and targets
        pred_array = torch.cat(predictions).numpy()
        target_array = torch.cat(targets).numpy()
        
        metrics = {}
        
        # Determine task type from model
        task_type = getattr(self.model, 'task_type', 'regression')
        
        if task_type == 'regression':
            metrics['mae'] = mean_absolute_error(target_array, pred_array)
            metrics['rmse'] = np.sqrt(mean_squared_error(target_array, pred_array))
            metrics['r2'] = r2_score(target_array, pred_array)
        else:
            # Classification
            if pred_array.ndim > 1 and pred_array.shape[1] > 1:
                pred_classes = np.argmax(pred_array, axis=1)
            else:
                pred_classes = (pred_array > 0.5).astype(int).flatten()
            
            target_classes = target_array.astype(int).flatten()
            
            metrics['accuracy'] = accuracy_score(target_classes, pred_classes)
            metrics['precision'] = precision_score(target_classes, pred_classes, average='binary', zero_division=0)
            metrics['recall'] = recall_score(target_classes, pred_classes, average='binary', zero_division=0)
            metrics['f1'] = f1_score(target_classes, pred_classes, average='binary', zero_division=0)
        
        return metrics
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple):
                    features = batch[0]
                else:
                    features = batch
                
                features = features.to(self.device)
                outputs = self.model(features)
                
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                all_predictions.append(outputs.cpu())
        
        return torch.cat(all_predictions).numpy()
    
    def save_model(self, filepath: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def create_model(input_size: int,
                 task_type: str = 'regression',
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 **kwargs) -> nn.Module:
    output_size = 1 if task_type == 'regression' else 2
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        task_type=task_type,
        **kwargs
    )
    
    return model


def create_data_loaders(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       batch_size: int = 32,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CustomDataset(X_train, y_train, fit_scaler=True)
    val_dataset = CustomDataset(X_val, y_val, scaler=train_dataset.get_scaler(), fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.1).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )
    
    model = create_model(
        input_size=n_features,
        task_type='regression',
        hidden_sizes=[64, 32, 16],
        dropout_rate=0.2
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    print("\nTraining model...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=10,
        verbose=True
    )
    
    test_dataset = CustomDataset(X_test, y_test, scaler=train_loader.dataset.get_scaler(), fit_scaler=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"\n=== Test Results ===")
    print(f"Test Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

