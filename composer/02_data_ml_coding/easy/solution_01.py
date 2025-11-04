"""
Solution for Exercise 1: Classification Model from Scratch

This file contains the reference solution. It's kept separate so you
can't see it while working on your implementation.
"""

import numpy as np
from typing import Optional


class LogisticRegression:
    """
    Logistic Regression classifier from scratch.
    
    This implementation uses gradient descent for optimization.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 regularization: float = 0.0):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iter: Maximum number of iterations
            regularization: L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        n = len(y)
        z = X @ self.weights + self.bias
        predictions = self._sigmoid(z)
        
        # Avoid log(0) by clipping predictions
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Add L2 regularization
        if self.regularization > 0:
            loss += self.regularization * np.sum(self.weights ** 2)
        
        return loss
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Compute gradients for weights and bias."""
        n = len(y)
        z = X @ self.weights + self.bias
        predictions = self._sigmoid(z)
        
        # Gradient for weights
        gradient_weights = (1 / n) * X.T @ (predictions - y)
        if self.regularization > 0:
            gradient_weights += 2 * self.regularization * self.weights
        
        # Gradient for bias
        gradient_bias = (1 / n) * np.sum(predictions - y)
        
        return gradient_weights, gradient_bias
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,), binary {0, 1}
        
        Returns:
            self for method chaining
        """
        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Compute gradients
            gradient_weights, gradient_bias = self._compute_gradient(X, y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias
            
            # Track loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Check convergence
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                    break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates."""
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

