import numpy as np
from scipy import stats
# KNN works by placing the point in a space with all the trianing data, then chosing the nearest 
# neighbors from the training set and using that to vote on what the classification of the point should be 

class KNN:
    def __init__(self, k, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = k

    def distance(self, x_i, y):
        return (x_i - y).T.dot(x_i - y)

    def inference(self, X_new):
        # distances from each point
        distances = [self.distance(X_new, X) for X in self.X_train]

        sorted_indecies = np.argsort(distances)
        top_k = sorted_indecies[:self.k]
        votes = np.array([self.Y_train[vote] for vote in top_k])
        return stats.mode(votes)
