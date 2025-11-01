"""
Solution: Linear Regression from Scratch

This implementation demonstrates production-quality ML code with:
- Proper vectorization
- Input validation
- Comprehensive documentation
- Error handling
- Testing
"""

import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression using Gradient Descent
    
    This implementation uses batch gradient descent to optimize the
    mean squared error loss function.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent. Too large may cause divergence,
        too small may result in slow convergence.
    n_iterations : int, default=1000
        Number of gradient descent iterations
    
    Attributes:
    -----------
    weights : np.ndarray, shape (n_features,)
        Model weights (coefficients) after fitting
    bias : float
        Model bias (intercept) after fitting
    loss_history : List[float]
        Loss value at each iteration during training
        
    Example:
    --------
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 6, 8])
    >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    >>> model.fit(X, y)
    >>> predictions = model.predict(np.array([[5]]))
    >>> print(predictions)  # Should be close to 10
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        # Validate parameters
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Will be set during fitting
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Train the linear regression model using gradient descent
        
        The algorithm:
        1. Initialize weights to zeros
        2. For each iteration:
           a. Compute predictions: ŷ = Xw + b
           b. Compute loss: MSE = (1/2m) Σ(ŷ - y)²
           c. Compute gradients: dw = (1/m)Xᵀ(ŷ - y), db = (1/m)Σ(ŷ - y)
           d. Update parameters: w = w - α·dw, b = b - α·db
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : LinearRegression
            Fitted model (for method chaining)
        """
        # Input validation
        X = self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            y_pred = self._forward(X)
            
            # Compute and store loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Compute gradients
            # dL/dw = (1/m) * X^T * (y_pred - y)
            # dL/db = (1/m) * sum(y_pred - y)
            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict on
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
            
        Raises:
        -------
        ValueError
            If model hasn't been fitted yet
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Validate feature count
        if X.shape[1] != len(self.weights):
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{len(self.weights)} features"
            )
        
        return self._forward(X)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute forward pass: y = Xw + b
        
        This is separated into its own method for clarity and reuse.
        """
        return X @ self.weights + self.bias
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss
        
        MSE = (1/2m) Σ(ŷ - y)²
        
        The factor of 1/2 makes the derivative cleaner but doesn't
        affect optimization (just scales the loss value).
        """
        y_pred = self._forward(X)
        return np.mean((y_pred - y) ** 2) / 2
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess input data
        
        Returns:
        --------
        X : np.ndarray
            Validated and possibly reshaped X
        """
        # Convert to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Ensure y is 1D
        if y.ndim != 1:
            y = y.ravel()
        
        # Check shapes match
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        # Check for NaN or inf
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input contains infinite values")
        
        return X
    
    def plot_loss_history(self, figsize=(10, 6)):
        """
        Plot the loss history during training
        
        Useful for diagnosing training issues:
        - Loss should decrease monotonically
        - If loss increases, learning rate may be too high
        - If loss plateaus early, may need more iterations or higher learning rate
        """
        if not self.loss_history:
            raise ValueError("No training history available. Fit the model first.")
        
        plt.figure(figsize=figsize)
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss History')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination)
        
        R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)²
        
        R² = 1 means perfect predictions
        R² = 0 means model is no better than predicting the mean
        R² < 0 means model is worse than predicting the mean
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def test_linear_regression():
    """Comprehensive test suite for LinearRegression"""
    
    print("="*70)
    print("TESTING LINEAR REGRESSION IMPLEMENTATION")
    print("="*70)
    
    # Test 1: Simple linear relationship
    print("\nTest 1: Simple linear relationship (y = 3x + 2)")
    print("-" * 70)
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1
    
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    print(f"Learned weights: {model.weights[0]:.3f} (expected: 3.0)")
    print(f"Learned bias: {model.bias:.3f} (expected: 2.0)")
    print(f"R² score: {model.score(X, y):.4f}")
    
    assert abs(model.weights[0] - 3.0) < 0.5, "Weight should be close to 3"
    assert abs(model.bias - 2.0) < 0.5, "Bias should be close to 2"
    print("✓ Test 1 passed")
    
    # Test 2: Multiple features
    print("\nTest 2: Multiple features")
    print("-" * 70)
    X = np.random.randn(100, 3)
    true_weights = np.array([2, -1, 0.5])
    true_bias = 1
    y = X @ true_weights + true_bias + np.random.randn(100) * 0.1
    
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    print(f"Learned weights: {model.weights}")
    print(f"Expected weights: {true_weights}")
    print(f"Learned bias: {model.bias:.3f} (expected: {true_bias})")
    print(f"R² score: {model.score(X, y):.4f}")
    
    assert np.allclose(model.weights, true_weights, atol=0.5), "Weights should be close"
    print("✓ Test 2 passed")
    
    # Test 3: Loss decreases
    print("\nTest 3: Loss should decrease during training")
    print("-" * 70)
    assert model.loss_history[0] > model.loss_history[-1], "Loss should decrease"
    print(f"Initial loss: {model.loss_history[0]:.6f}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")
    print(f"Loss reduction: {(1 - model.loss_history[-1]/model.loss_history[0])*100:.2f}%")
    print("✓ Test 3 passed")
    
    # Test 4: Prediction shape
    print("\nTest 4: Prediction shapes")
    print("-" * 70)
    X_test = np.array([[1.0, 2.0, 3.0]])
    pred = model.predict(X_test)
    assert pred.shape == (1,), f"Expected shape (1,), got {pred.shape}"
    print(f"Single prediction shape: {pred.shape} ✓")
    
    X_test = np.random.randn(10, 3)
    pred = model.predict(X_test)
    assert pred.shape == (10,), f"Expected shape (10,), got {pred.shape}"
    print(f"Multiple predictions shape: {pred.shape} ✓")
    print("✓ Test 4 passed")
    
    # Test 5: Error handling
    print("\nTest 5: Error handling")
    print("-" * 70)
    
    try:
        model_new = LinearRegression()
        model_new.predict(X)
        assert False, "Should raise error when predicting before fitting"
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    try:
        X_wrong = np.random.randn(10, 5)  # Wrong number of features
        model.predict(X_wrong)
        assert False, "Should raise error for wrong feature count"
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    print("✓ Test 5 passed")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


def demo_with_visualization():
    """
    Demonstrate the model with visualization
    """
    print("\n" + "="*70)
    print("DEMONSTRATION WITH VISUALIZATION")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2.5 * X.squeeze() + 1.5 + np.random.randn(100) * 1.5
    
    # Train model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Data and fitted line
    axes[0].scatter(X, y, alpha=0.5, label='Data')
    axes[0].plot(X, y_pred, 'r-', linewidth=2, label='Fitted line')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'Linear Regression Fit\nw={model.weights[0]:.2f}, b={model.bias:.2f}, R²={model.score(X, y):.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    axes[1].plot(model.loss_history)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Training Loss History')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'linear_regression_demo.png'")
    plt.show()


if __name__ == "__main__":
    # Run tests
    test_linear_regression()
    
    # Run demonstration
    demo_with_visualization()
    
    # Print key takeaways
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Vectorization is crucial for performance
   - Use NumPy operations instead of Python loops
   - X @ weights is much faster than manual multiplication

2. Input validation prevents bugs
   - Check shapes, NaN, inf values
   - Provide clear error messages

3. Gradient descent requires tuning
   - Learning rate too high → divergence
   - Learning rate too low → slow convergence
   - Monitor loss to diagnose issues

4. Production code needs:
   - Comprehensive docstrings
   - Type hints
   - Error handling
   - Tests
   - Clear variable names

5. R² score is more interpretable than MSE
   - R² = 1 means perfect fit
   - R² = 0 means model = mean
   - R² < 0 means model worse than mean
    """)

