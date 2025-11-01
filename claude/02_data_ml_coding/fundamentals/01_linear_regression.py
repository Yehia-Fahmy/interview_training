"""
Problem: Implement Linear Regression from Scratch

Difficulty: Fundamentals
Time: 30-45 minutes

Description:
Implement a linear regression model from scratch using gradient descent.
Your implementation should support:
- Training with gradient descent
- Making predictions
- Computing loss (MSE)
- Proper vectorization using NumPy

Requirements:
1. Implement the LinearRegression class with fit() and predict() methods
2. Use gradient descent for optimization
3. Support learning rate and number of iterations as parameters
4. Track loss history during training
5. Add proper input validation
6. Include comprehensive docstrings

Evaluation Criteria:
- Correctness: Model trains and predicts accurately
- Code quality: Clean, readable, well-documented
- Vectorization: Efficient NumPy operations (no Python loops for computation)
- Error handling: Validate inputs, handle edge cases
- Testing: Comprehensive test cases

Learning Objectives:
- Understand gradient descent optimization
- Practice NumPy vectorization
- Implement ML algorithm from first principles
- Write production-quality ML code

Mathematical Background:
- Hypothesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
- Cost function: J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- Gradient: ∂J/∂θⱼ = (1/m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
- Update rule: θⱼ := θⱼ - α(∂J/∂θⱼ)
"""

import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression using Gradient Descent
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of gradient descent iterations
    
    Attributes:
    -----------
    weights : np.ndarray
        Model weights (coefficients)
    bias : float
        Model bias (intercept)
    loss_history : List[float]
        Loss value at each iteration
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        # TODO: Initialize parameters
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Train the linear regression model using gradient descent
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : LinearRegression
            Fitted model
        """
        # TODO: Implement gradient descent
        pass
    
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
        """
        # TODO: Implement prediction
        pass
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True values
            
        Returns:
        --------
        loss : float
            MSE loss
        """
        # TODO: Implement MSE calculation
        pass
    
    def plot_loss_history(self):
        """Plot the loss history during training"""
        # TODO: Implement plotting
        pass


def test_linear_regression():
    """Test the LinearRegression implementation"""
    
    # Test 1: Simple linear relationship
    print("Test 1: Simple linear relationship")
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1
    
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Check if weights are close to true values (3 and 2)
    print(f"Learned weights: {model.weights}, bias: {model.bias}")
    print(f"Expected: weights ≈ [3], bias ≈ 2")
    
    # Test predictions
    X_test = np.array([[1.0], [2.0], [3.0]])
    predictions = model.predict(X_test)
    print(f"Predictions for [1, 2, 3]: {predictions}")
    
    # Test 2: Multiple features
    print("\nTest 2: Multiple features")
    X = np.random.randn(100, 3)
    true_weights = np.array([2, -1, 0.5])
    true_bias = 1
    y = X @ true_weights + true_bias + np.random.randn(100) * 0.1
    
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    print(f"Learned weights: {model.weights}")
    print(f"Expected: {true_weights}")
    print(f"Learned bias: {model.bias:.2f}, Expected: {true_bias}")
    
    # Test 3: Check loss decreases
    print("\nTest 3: Loss should decrease during training")
    assert model.loss_history[0] > model.loss_history[-1], "Loss should decrease"
    print(f"Initial loss: {model.loss_history[0]:.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
    print("✓ Loss decreased during training")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_linear_regression()

