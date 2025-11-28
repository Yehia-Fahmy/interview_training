import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D clustering data
# Create three distinct clusters
n_samples_per_cluster = 50

# Cluster 1: centered around (0, 0)
X_cluster1 = np.random.randn(n_samples_per_cluster, 2) * 1.5 + np.array([0, 0])

# Cluster 2: centered around (4, 4)
X_cluster2 = np.random.randn(n_samples_per_cluster, 2) * 1.5 + np.array([4, 4])

# Cluster 3: centered around (0, 4)
X_cluster3 = np.random.randn(n_samples_per_cluster, 2) * 1.5 + np.array([0, 4])

# Combine all clusters
X_train = np.vstack([X_cluster1, X_cluster2, X_cluster3])

# Shuffle the data
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]

print(f"Training data shape: {X_train.shape}")
print(f"Number of samples: {len(X_train)}")
print(f"Number of features: {X_train.shape[1]}")

# Create KMeans model
k = 3
epochs = 10
n_features = 2

kmeans = KMeans(epochs=epochs, k_values=k, n_features=n_features)

print(f"\nInitializing KMeans with k={k}, epochs={epochs}")
print("Initial centroids:")
for i, centroid in enumerate(kmeans.centroids):
    print(f"  Centroid {i}: {centroid.center}")

# Train the model
print("\nTraining KMeans...")
kmeans.train(X_train)

print("\nFinal centroids after training:")
for i, centroid in enumerate(kmeans.centroids):
    print(f"  Centroid {i}: {centroid.center}")
    print(f"    Number of samples: {len(centroid.samples)}")

# Predict on training data to see cluster assignments
print("\n" + "="*50)
print("Predicting cluster assignments for training data:")
print("="*50)

predictions = np.array([kmeans.predict(x) for x in X_train])
unique, counts = np.unique(predictions, return_counts=True)
print(f"Cluster distribution: {dict(zip(unique, counts))}")

# Test points
test_points = np.array([
    [0, 0],      # Should be near cluster 0
    [4, 4],      # Should be near cluster 1
    [0, 4],      # Should be near cluster 2
    [2, 2],      # Border point
    [-1, -1],    # Edge case
    [5, 5],      # Edge case
])

print("\n" + "="*50)
print("Testing KMeans predictions on new points:")
print("="*50)

for i, test_point in enumerate(test_points):
    cluster = kmeans.predict(test_point)
    # Calculate distance to assigned centroid
    assigned_centroid = kmeans.centroids[cluster].center
    distance = np.sqrt((test_point - assigned_centroid).T.dot(test_point - assigned_centroid))
    print(f"Test point {i+1}: {test_point} -> Cluster: {cluster} (distance to centroid: {distance:.3f})")

# Optional: Visualize the results
try:
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data (before clustering)
    plt.subplot(1, 3, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c='gray', alpha=0.6, s=30)
    plt.title('Original Data (Unlabeled)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Clustered data with centroids
    plt.subplot(1, 3, 2)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i in range(k):
        cluster_points = X_train[predictions == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], label=f'Cluster {i}', 
                       alpha=0.6, s=30)
        # Plot centroid
        centroid = kmeans.centroids[i].center
        plt.scatter(centroid[0], centroid[1], 
                   c=colors[i % len(colors)], marker='X', s=300, 
                   edgecolors='black', linewidths=2, label=f'Centroid {i}')
    plt.title(f'KMeans Clustering Results (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Test predictions
    plt.subplot(1, 3, 3)
    # Plot all clusters
    for i in range(k):
        cluster_points = X_train[predictions == i]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], alpha=0.2, s=20)
        # Plot centroid
        centroid = kmeans.centroids[i].center
        plt.scatter(centroid[0], centroid[1], 
                   c=colors[i % len(colors)], marker='X', s=200, 
                   edgecolors='black', linewidths=2, alpha=0.7)
    
    # Plot test points with predictions
    for test_point in test_points:
        cluster = kmeans.predict(test_point)
        plt.scatter(test_point[0], test_point[1], 
                   c=colors[cluster % len(colors)], marker='*', s=300, 
                   edgecolors='black', linewidths=2)
    
    plt.title(f'Test Point Predictions (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_test_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'kmeans_test_results.png'")
    plt.show()
    
except Exception as e:
    print(f"\nCould not generate visualization: {e}")
    print("(This is okay - the predictions above are still valid)")

# Additional analysis: Calculate within-cluster sum of squares (WCSS)
print("\n" + "="*50)
print("Cluster Quality Metrics:")
print("="*50)

wcss = 0
for i in range(k):
    cluster_points = X_train[predictions == i]
    if len(cluster_points) > 0:
        centroid = kmeans.centroids[i].center
        cluster_wcss = np.sum([(point - centroid).T.dot(point - centroid) 
                              for point in cluster_points])
        wcss += cluster_wcss
        print(f"Cluster {i}: {len(cluster_points)} points, WCSS: {cluster_wcss:.2f}")

print(f"\nTotal Within-Cluster Sum of Squares (WCSS): {wcss:.2f}")

