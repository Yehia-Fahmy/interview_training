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
        # TODO: Implement sigmoid function
        # Hint: sigmoid(z) = 1 / (1 + exp(-z))
        # Be careful with numerical stability (clip z values)
        pass
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        # TODO: Implement loss function
        # Hint: loss = -1/n * sum(y*log(p) + (1-y)*log(1-p))
        # where p = sigmoid(X @ weights + bias)
        # Don't forget L2 regularization if self.regularization > 0
        pass
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Compute gradients for weights and bias."""
        # TODO: Implement gradient computation
        # Hint: 
        # - gradient_weights = 1/n * X.T @ (predictions - y) + regularization * weights
        # - gradient_bias = 1/n * sum(predictions - y)
        # where predictions = sigmoid(X @ weights + bias)
        pass
    
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
            # TODO: Compute gradients and update weights/bias
            # Use self._compute_gradient() and self._compute_loss()
            
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
        # TODO: Compute probabilities using sigmoid
        # Return shape: (n_samples,)
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        # TODO: Use predict_proba() and threshold at 0.5
        # Return shape: (n_samples,) with values in {0, 1}
        pass


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

