import numpy as np

class SoftKMeans:
    def __init__(self, k, iter, beta):
        self.k = k
        self.iter = iter
        self.beta = beta

    def assign(self, X):
        distance = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        weights = np.exp(-self.beta * distance ** 2)
        weights /= weights.sum(axis=1)[:, np.newaxis]
        self.weights = weights
        return weights

    def update(self, X, weights):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            if np.sum(weights[:, i]) != 0:
                new_centroids[i] = np.sum(weights[:, i][:, np.newaxis] * X, axis=0) / np.sum(weights[:, i])
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def train(self, X):
        samples, features = X.shape

        indices = np.arange(samples)
        np.random.shuffle(indices)
        self.centroids = X[indices[:self.k]]

        for i in range(self.iter):
            weights = self.assign(X)
            new_centroids = self.update(X, weights)

            if (self.centroids == new_centroids).all():
                break

            self.centroids = new_centroids

    def predict(self, X):
        return np.argmax(self.assign(X), axis=1)

    def normalize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def J(self, X):
        # caculate the distance of each point to every centroid
        distance = ((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2) # size: (samples, k)
        distance = distance * self.weights # size: (samples, k)
        J = np.sum(distance) # size: (1)
        return J