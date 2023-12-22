import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, iter):
        self.k = k
        self.iter = iter
        
    def assign(self, X):
        # X size: (1, features)
        # centroids size: (k, features)
        distance = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2) 
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
            #caculate the distance of each point to its centroid
            distance = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2)
            J = np.sum(np.min(distance, axis=0))
            print("iter: ", i)
            print("J: ", J)
            if (self.centroids == new_centroids).all():
                break
            
            self.centroids = new_centroids
        
    def predict(self, X):
        return self.assign(X)
    
    def normalize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    