"""
Exercise 5: PyTorch Deep Learning Pipeline

Build a complete PyTorch deep learning pipeline from scratch.
Implement custom dataset, neural network architecture, training loop, and evaluation.
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
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 89
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for tabular data.
    
    Handles feature scaling, target encoding, and data loading.
    Should be reusable for different datasets.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = True):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target vector (n_samples,) or None for inference
            scaler: Pre-fitted scaler (for test/inference) or None
            fit_scaler: Whether to fit scaler on this data (True for train, False for test)
        """
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        
        # Implement feature scaling
        # 1. If fit_scaler is True, create and fit a StandardScaler
        # 2. If scaler is provided, use it (for test/inference)
        # 3. Transform features using the scaler
        # 4. Store the scaler for later use
        
        if fit_scaler:
            # Create and fit scaler on training data
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided when fit_scaler=False")
            # Use provided scaler and transform features
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, target) if targets exist, else (features,)
        """
        # Get features at index idx and convert to torch tensor
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # If targets exist, return (features, target), else just (features,)
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (features, target)
        else:
            return (features,)
    
    def get_scaler(self) -> Optional[StandardScaler]:
        """Get the fitted scaler"""
        return self.scaler


class NeuralNetwork(nn.Module):
    """
    Custom neural network architecture.
    
    Configurable depth and width with dropout for regularization.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64, 32],
                 output_size: int = 1,
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 task_type: str = 'regression'):  # 'regression' or 'classification'
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units
            activation: Activation function ('relu', 'tanh', 'gelu')
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            task_type: 'regression' or 'classification'
        """
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.task_type = task_type
        
        # TODO: Build the network architecture
        # 1. Create layers list to store all layers
        # 2. Add input layer -> first hidden layer
        # 3. For each hidden layer:
        #    - Add linear layer
        #    - Add batch normalization (if enabled)
        #    - Add activation function
        #    - Add dropout
        # 4. Add output layer (no activation for regression, sigmoid/softmax for classification)
        # 5. Use nn.Sequential or nn.ModuleList to organize layers
        
        self.layers = nn.ModuleList()
        
        # Input to first hidden
        # TODO: Add first layer with batch norm, activation, dropout
        
        # Hidden layers
        # TODO: Add intermediate hidden layers
        
        # Output layer
        # TODO: Add output layer (consider task_type for activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_size)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        # TODO: Implement forward pass through all layers
        pass


class ModelTrainer:
    """
    Training loop and model management.
    
    Handles training, validation, early stopping, and model checkpointing.
    """
    
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: Optional[torch.device] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to run on (cuda/cpu)
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # TODO: Implement training loop
        # 1. Iterate over batches
        # 2. Move data to device
        # 3. Zero gradients
        # 4. Forward pass
        # 5. Compute loss
        # 6. Backward pass
        # 7. Update weights
        # 8. Track loss and predictions for metrics
        
        for batch_idx, batch in enumerate(dataloader):
            # TODO: Extract features and targets from batch
            # TODO: Move to device
            # TODO: Forward pass
            # TODO: Compute loss
            # TODO: Backward pass and optimization
            pass
        
        avg_loss = total_loss / len(dataloader)
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # TODO: Implement validation loop
        # 1. Set model to eval mode
        # 2. Use torch.no_grad() context
        # 3. Iterate over batches
        # 4. Forward pass and compute loss
        # 5. Track predictions and targets
        
        with torch.no_grad():
            # TODO: Implement validation loop
            pass
        
        avg_loss = total_loss / len(dataloader)
        metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              early_stopping_patience: int = 10,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
        
        Returns:
            Dictionary with training history
        """
        # TODO: Implement training loop with early stopping
        # 1. Loop over epochs
        # 2. Train for one epoch
        # 3. Validate
        # 4. Update learning rate scheduler if provided
        # 5. Track best model (save state when validation loss improves)
        # 6. Early stopping (stop if no improvement for patience epochs)
        # 7. Print progress if verbose
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # TODO: Train and validate
            # TODO: Update scheduler
            # TODO: Check for best model
            # TODO: Early stopping logic
            pass
        
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
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of prediction tensors
            targets: List of target tensors
        
        Returns:
            Dictionary of metric names and values
        """
        # TODO: Concatenate predictions and targets
        # TODO: Convert to numpy if needed
        # TODO: Compute metrics based on task type:
        #   - Regression: MAE, RMSE, R2
        #   - Classification: Accuracy, Precision, Recall, F1
        
        pred_array = None  # TODO: Concatenate and convert
        target_array = None  # TODO: Concatenate and convert
        
        metrics = {}
        
        # TODO: Compute metrics based on task_type
        # For regression: MAE, RMSE, R2
        # For classification: Accuracy, Precision, Recall, F1
        
        return metrics
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            dataloader: Data loader
        
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        all_predictions = []
        
        # TODO: Implement prediction loop
        # 1. Set model to eval mode
        # 2. Use torch.no_grad()
        # 3. Iterate over batches
        # 4. Collect predictions
        
        with torch.no_grad():
            # TODO: Implement prediction
            pass
        
        # TODO: Concatenate and return as numpy array
        return None
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
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
        """Load model checkpoint"""
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
    """
    Factory function to create model.
    
    Args:
        input_size: Number of input features
        task_type: 'regression' or 'classification'
        hidden_sizes: Hidden layer sizes
        dropout_rate: Dropout rate
        **kwargs: Additional arguments for NeuralNetwork
    
    Returns:
        Initialized model
    """
    output_size = 1 if task_type == 'regression' else 2  # Binary classification
    
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
    """
    Create train and validation data loaders.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size
        shuffle: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create train dataset with fit_scaler=True
    train_dataset = CustomDataset(
        features=X_train,
        targets=y_train,
        fit_scaler=True
    )
    
    # Get scaler from train dataset
    scaler = train_dataset.get_scaler()
    
    # Create validation dataset with scaler from train_dataset
    val_dataset = CustomDataset(
        features=X_val,
        targets=y_val,
        scaler=scaler,
        fit_scaler=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle validation data
    )
    
    return train_loader, val_loader


def evaluate_student_solution(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluate student solution and compare to baseline and ideal solutions.
    
    Returns:
        Dictionary with results from student, baseline, and ideal solutions
    """
    results = {
        'student': None,
        'baseline': None,
        'ideal': None,
        'errors': []
    }
    
    print("\n" + "="*70)
    print("EVALUATING YOUR SOLUTION")
    print("="*70)
    
    # Try to run student solution
    print("\n[1/3] Running your solution...")
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=32
        )
        
        # Create model
        model = create_model(
            input_size=X_train.shape[1],
            task_type='regression',
            hidden_sizes=[64, 32, 16],
            dropout_rate=0.2
        )
        
        # Create trainer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        trainer = ModelTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Train (with fewer epochs for evaluation)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=30,
            early_stopping_patience=7,
            verbose=False
        )
        
        # Evaluate on test set
        test_dataset = CustomDataset(X_test, y_test, scaler=train_loader.dataset.get_scaler(), fit_scaler=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        test_loss, test_metrics = trainer.validate(test_loader)
        
        results['student'] = {
            'test_loss': test_loss,
            'metrics': test_metrics,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'epochs_trained': len(history['train_loss'])
        }
        print(f"✓ Your solution completed successfully!")
        print(f"  Test Loss: {test_loss:.4f}")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
            
    except Exception as e:
        error_msg = f"Student solution error: {str(e)}"
        results['errors'].append(error_msg)
        print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
    
    # Run baseline solution
    print("\n[2/3] Running baseline solution (simple implementation)...")
    try:
        from baseline_05 import baseline_train_and_evaluate
        baseline_results = baseline_train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
        results['baseline'] = baseline_results
        print(f"✓ Baseline solution completed!")
        print(f"  Test Loss: {baseline_results['test_loss']:.4f}")
        print(f"  MAE: {baseline_results['mae']:.4f}")
        print(f"  RMSE: {baseline_results['rmse']:.4f}")
        print(f"  R²: {baseline_results['r2']:.4f}")
    except Exception as e:
        error_msg = f"Baseline solution error: {str(e)}"
        results['errors'].append(error_msg)
        print(f"✗ {error_msg}")
    
    # Run ideal solution
    print("\n[3/3] Running ideal solution (reference implementation)...")
    try:
        import importlib.util
        import sys
        from pathlib import Path
        
        solution_path = Path(__file__).parent / 'solution_05.py'
        spec = importlib.util.spec_from_file_location("solution_05", solution_path)
        if spec and spec.loader:
            ideal_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ideal_module)
            
            # Run ideal solution
            ideal_train_loader, ideal_val_loader = ideal_module.create_data_loaders(
                X_train, y_train, X_val, y_val, batch_size=32
            )
            
            ideal_model = ideal_module.create_model(
                input_size=X_train.shape[1],
                task_type='regression',
                hidden_sizes=[64, 32, 16],
                dropout_rate=0.2
            )
            
            ideal_criterion = nn.MSELoss()
            ideal_optimizer = optim.Adam(ideal_model.parameters(), lr=0.001)
            ideal_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ideal_optimizer, mode='min', factor=0.5, patience=5)
            
            ideal_trainer = ideal_module.ModelTrainer(
                model=ideal_model,
                criterion=ideal_criterion,
                optimizer=ideal_optimizer,
                scheduler=ideal_scheduler
            )
            
            ideal_history = ideal_trainer.train(
                train_loader=ideal_train_loader,
                val_loader=ideal_val_loader,
                epochs=30,
                early_stopping_patience=7,
                verbose=False
            )
            
            ideal_test_dataset = ideal_module.CustomDataset(
                X_test, y_test, 
                scaler=ideal_train_loader.dataset.get_scaler(), 
                fit_scaler=False
            )
            ideal_test_loader = DataLoader(ideal_test_dataset, batch_size=32, shuffle=False)
            
            ideal_test_loss, ideal_test_metrics = ideal_trainer.validate(ideal_test_loader)
            
            results['ideal'] = {
                'test_loss': ideal_test_loss,
                'metrics': ideal_test_metrics,
                'final_val_loss': ideal_history['val_loss'][-1] if ideal_history['val_loss'] else None,
                'epochs_trained': len(ideal_history['train_loss'])
            }
            print(f"✓ Ideal solution completed!")
            print(f"  Test Loss: {ideal_test_loss:.4f}")
            for metric, value in ideal_test_metrics.items():
                print(f"  {metric.upper()}: {value:.4f}")
        else:
            raise ImportError("Could not load solution_05.py")
            
    except Exception as e:
        error_msg = f"Ideal solution error: {str(e)}"
        results['errors'].append(error_msg)
        print(f"✗ {error_msg}")
    
    return results


def print_comparison(results: Dict):
    """Print comparison between student, baseline, and ideal solutions"""
    print("\n" + "="*70)
    print("COMPARISON: YOUR SOLUTION vs BASELINE vs IDEAL")
    print("="*70)
    
    if results['student'] is None:
        print("\n⚠ Your solution could not be evaluated. Please fix errors above.")
        return
    
    student = results['student']
    baseline = results.get('baseline')
    ideal = results.get('ideal')
    
    print("\n📊 Performance Metrics:")
    print("-" * 70)
    
    # Test Loss comparison
    print(f"\nTest Loss (MSE):")
    print(f"  Your Solution:  {student['test_loss']:.4f}")
    if baseline:
        improvement_over_baseline = ((baseline['test_loss'] - student['test_loss']) / baseline['test_loss']) * 100
        print(f"  Baseline:        {baseline['test_loss']:.4f} ({improvement_over_baseline:+.1f}% vs yours)")
    if ideal:
        gap_from_ideal = ((student['test_loss'] - ideal['test_loss']) / ideal['test_loss']) * 100
        print(f"  Ideal:           {ideal['test_loss']:.4f} ({gap_from_ideal:+.1f}% gap from ideal)")
    
    # Metrics comparison
    if 'metrics' in student:
        print(f"\nR² Score:")
        print(f"  Your Solution:  {student['metrics'].get('r2', 'N/A'):.4f}")
        if baseline:
            print(f"  Baseline:        {baseline.get('r2', 'N/A'):.4f}")
        if ideal and 'r2' in ideal.get('metrics', {}):
            print(f"  Ideal:           {ideal['metrics']['r2']:.4f}")
        
        print(f"\nMAE:")
        print(f"  Your Solution:  {student['metrics'].get('mae', 'N/A'):.4f}")
        if baseline:
            print(f"  Baseline:        {baseline.get('mae', 'N/A'):.4f}")
        if ideal and 'mae' in ideal.get('metrics', {}):
            print(f"  Ideal:           {ideal['metrics']['mae']:.4f}")
        
        print(f"\nRMSE:")
        print(f"  Your Solution:  {student['metrics'].get('rmse', 'N/A'):.4f}")
        if baseline:
            print(f"  Baseline:        {baseline.get('rmse', 'N/A'):.4f}")
        if ideal and 'rmse' in ideal.get('metrics', {}):
            print(f"  Ideal:           {ideal['metrics']['rmse']:.4f}")
    
    # Training efficiency
    print(f"\n📈 Training Efficiency:")
    print("-" * 70)
    print(f"  Epochs Trained: {student.get('epochs_trained', 'N/A')}")
    if ideal:
        print(f"  Ideal Epochs:    {ideal.get('epochs_trained', 'N/A')}")
    
    # Overall assessment
    print(f"\n🎯 Assessment:")
    print("-" * 70)
    
    if baseline and ideal:
        student_loss = student['test_loss']
        baseline_loss = baseline['test_loss']
        ideal_loss = ideal['test_loss']
        
        # Calculate performance relative to baseline and ideal
        if baseline_loss > 0:
            improvement_pct = ((baseline_loss - student_loss) / baseline_loss) * 100
            if student_loss < baseline_loss:
                print(f"✓ Your solution performs {improvement_pct:.1f}% better than baseline!")
            else:
                print(f"⚠ Your solution performs {abs(improvement_pct):.1f}% worse than baseline.")
        
        if ideal_loss > 0:
            gap_pct = ((student_loss - ideal_loss) / ideal_loss) * 100
            if gap_pct < 5:
                print(f"🎉 Excellent! You're within 5% of the ideal solution!")
            elif gap_pct < 15:
                print(f"👍 Good! You're within 15% of the ideal solution.")
            elif gap_pct < 30:
                print(f"💪 Getting there! You're within 30% of the ideal solution.")
            else:
                print(f"📚 Keep working! There's room for improvement.")
    
    print("\n" + "="*70)


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    print("="*70)
    print("PyTorch Deep Learning Pipeline - Solution Evaluation")
    print("="*70)
    print("\nGenerating synthetic data...")
    np.random.seed(RANDOM_STATE)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Create regression target with some non-linearity
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.1).astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Evaluate solutions
    results = evaluate_student_solution(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print comparison
    print_comparison(results)

