import numpy as np

class KMeans:
    def __init__(self, k, iter):
        self.k = k
        self.iter = iter
        self.centroids = None
        
    def assign(self, X):
        distance = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distance, axis=1)
    
    def update(self, X, labels):
        new_centroids = np.empty_like(self.centroids)
        for i in range(self.k):
            new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids
    
    def train(self, X):
        numbers, features = X.shape
        
        self.centroids = X[np.random.choice(numbers, self.k, replace=False)]
        
        for i in range(self.iter):
            labels = self.assign(X)
            new_centroids = self.update(X, labels)
            
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids