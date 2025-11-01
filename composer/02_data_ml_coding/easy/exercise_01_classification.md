# Exercise 1: Classification Model from Scratch

**Difficulty:** Easy  
**Time Limit:** 45 minutes (with AI assistance allowed)  
**Focus:** Implementing ML algorithms, code quality, design rationale

## Problem

Implement a logistic regression classifier from scratch (without using sklearn's LogisticRegression, but you can use NumPy). The focus is on:

1. **Clean, readable code** with proper structure
2. **Explaining design choices**
3. **Proper documentation**

## Requirements

1. Implement logistic regression with:
   - Gradient descent optimization
   - L2 regularization (optional)
   - Convergence detection
   
2. Structure your code with:
   - Clear class structure
   - Well-documented methods
   - Separation of concerns

3. Include:
   - Training method
   - Prediction method
   - Probability estimation

## Solution Template

```python
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
        # Your implementation
        pass
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        # Your implementation
        pass
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Compute gradients for weights and bias."""
        # Your implementation
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
            # Compute gradients and update
            # Your implementation
            
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
        # Your implementation
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        # Your implementation
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
```

## Key Learning Points

1. **Understanding ML Fundamentals:** Implement core algorithm logic
2. **Code Organization:** Classes, methods, documentation
3. **Design Decisions:** Explain why you made certain choices

## Design Considerations to Explain

- Why use sigmoid for binary classification?
- Why gradient descent vs other optimizers?
- How does learning rate affect convergence?
- Why include regularization?

