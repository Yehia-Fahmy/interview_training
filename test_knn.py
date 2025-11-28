import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D classification data
# Create two distinct clusters
n_samples_per_class = 50

# Class 0: centered around (0, 0)
X_class0 = np.random.randn(n_samples_per_class, 2) * 1.5 + np.array([0, 0])
Y_class0 = np.zeros(n_samples_per_class, dtype=int)

# Class 1: centered around (4, 4)
X_class1 = np.random.randn(n_samples_per_class, 2) * 1.5 + np.array([4, 4])
Y_class1 = np.ones(n_samples_per_class, dtype=int)

# Class 2: centered around (0, 4)
X_class2 = np.random.randn(n_samples_per_class, 2) * 1.5 + np.array([0, 4])
Y_class2 = np.full(n_samples_per_class, 2, dtype=int)

# Combine all classes
X_train = np.vstack([X_class0, X_class1, X_class2])
Y_train = np.hstack([Y_class0, Y_class1, Y_class2])

# Shuffle the data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
Y_train = Y_train[shuffle_idx]

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {Y_train.shape}")
print(f"Number of classes: {len(np.unique(Y_train))}")
print(f"Class distribution: {np.bincount(Y_train)}")

# Create KNN classifier
k = 5
knn = KNN(k=k, X_train=X_train, Y_train=Y_train)

# Test points
test_points = np.array([
    [0, 0],      # Should be class 0
    [4, 4],      # Should be class 1
    [0, 4],      # Should be class 2
    [2, 2],      # Border point
    [-1, -1],    # Edge case
    [5, 5],      # Edge case
])

print("\n" + "="*50)
print("Testing KNN predictions:")
print("="*50)

for i, test_point in enumerate(test_points):
    result = knn.inference(test_point)
    # stats.mode returns a ModeResult object with .mode attribute
    prediction = result.mode
    print(f"Test point {i+1}: {test_point} -> Predicted class: {prediction}")

# Optional: Visualize the results
try:
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Training data
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 
                c='red', label='Class 0', alpha=0.6)
    plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 
                c='blue', label='Class 1', alpha=0.6)
    plt.scatter(X_train[Y_train == 2, 0], X_train[Y_train == 2, 1], 
                c='green', label='Class 2', alpha=0.6)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Test predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 
                c='red', label='Class 0', alpha=0.3)
    plt.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 
                c='blue', label='Class 1', alpha=0.3)
    plt.scatter(X_train[Y_train == 2, 0], X_train[Y_train == 2, 1], 
                c='green', label='Class 2', alpha=0.3)
    
    # Plot test points with predictions
    colors = ['red', 'blue', 'green']
    for test_point in test_points:
        result = knn.inference(test_point)
        prediction = result.mode
        plt.scatter(test_point[0], test_point[1], 
                   c=colors[prediction], marker='*', s=200, 
                   edgecolors='black', linewidths=2)
    
    plt.title(f'Test Predictions (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_test_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'knn_test_results.png'")
    plt.show()
    
except Exception as e:
    print(f"\nCould not generate visualization: {e}")
    print("(This is okay - the predictions above are still valid)")

