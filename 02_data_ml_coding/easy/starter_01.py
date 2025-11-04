"""
Exercise 1: Classification Model from Scratch

Implement a logistic regression classifier from scratch (without using sklearn's LogisticRegression, 
but you can use NumPy). The focus is on:
1. Clean, readable code with proper structure
2. Explaining design choices
3. Proper documentation
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
        # Clip to prevent overflow in exp()
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        y_pred = X @ self.weights + self.bias
        p = self._sigmoid(y_pred)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        n = len(y_pred)
        loss = (-1 / n) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        if self.regularization > 0:
            loss += (self.regularization / 2) * np.sum(self.weights ** 2)
        return loss
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Compute gradients for weights and bias."""
        n = len(y)
        y_pred = self._sigmoid(X @ self.weights + self.bias)
        g_w = 1 / n * X.T @ (y_pred - y) + self.regularization * self.weights
        g_b = 1 / n * np.sum(y_pred - y)
        return (g_w, g_b)
    
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
            g_w, g_b = self._compute_gradient(X, y)
            self.weights -= g_w * self.learning_rate
            self.bias -= g_b * self.learning_rate
            # Track loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Check convergence (optional)
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                    break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates."""
        return self._sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        return (self.predict_proba(X) > 0.5).astype(int)


# Test your implementation
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")

