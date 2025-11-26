from dataclasses import dataclass
from turtle import distance
import numpy as np

'''
Implement K means

Init
1. initialize the centroids with random centers over the space

Train
1. each iteration assign the samples to the closest centroid
2. find the mean of the centroid
3. re-assign the centroid
4. repeat until training is done

Predict
1. find the closest centroid
'''

@dataclass
class Centroid:
    center: np.ndarray
    samples: np.ndarray

class KMeans:
    def __init__(self, epochs, k_values, n_features):
        self.epochs = epochs
        self.n_features = n_features
        self.centroids = [
            Centroid(
                center=np.random.randn(n_features),
                samples=np.empty((0, n_features))
            ) for _ in range(k_values)
        ]

    def _distance(self, x_i, y):
        return (x_i - y).T.dot(x_i - y)

    def train(self, X):
        for e in range(self.epochs):
            for centroid in self.centroids:
                centroid.samples = np.empty((0, self.n_features))
            for x_i in X:
                distances = [self._distance(x_i, centroid.center) for centroid in self.centroids]
                prediction_idx = distances.index(min(distances))
                self.centroids[prediction_idx].samples = np.vstack([self.centroids[prediction_idx].samples, x_i])
            for centroid in self.centroids:
                centroid.center = np.mean(centroid.samples, axis=0)

    def predict(self, x_i):
        distances = [self._distance(x_i, centroid.center) for centroid in self.centroids]
        prediction_idx = distances.index(min(distances))
        return prediction_idx
