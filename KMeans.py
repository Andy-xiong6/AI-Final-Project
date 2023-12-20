import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, iter):
        self.k = k
        self.iter = iter
        self.centroids = None # size: (k, features)
        
    def assign(self, X):
        # X size: (1, features)
        # centroids size: (k, features)
        distance = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2)) # sum
        return np.argmin(distance, axis=0) 
    
    def update(self, X, labels):
        new_centroids = np.empty_like(self.centroids)
        for i in range(self.k):
            new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids
    
    def train(self, X):
        numbers, features = X.shape
        
        #self.centroids = X[np.random.choice(numbers, self.k, replace=False)]
        indices = np.arange(numbers)
        np.random.shuffle(indices)
        self.centroids = X[indices[:self.k]]
        
        for i in range(self.iter):
            labels = self.assign(X)
            new_centroids = self.update(X, labels)
            
            if (self.centroids == new_centroids).all():
                break
            
            self.centroids = new_centroids
    
        
    def predict(self, X):
        return self.assign(X)
    
    def normalize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def distance(self, X):
        distance = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        #add every distance of each cluster
        return distance.sum(axis=0).sum(axis=0)
    