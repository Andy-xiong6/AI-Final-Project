import numpy as np

class KMeans:
    def __init__(self, k, iter):
        self.k = k
        self.iter = iter
        self.array = []
        
    def assign(self, X):
        # X size: (1, features)
        # centroids size: (k, features)
        distance = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2) 
        return np.argmin(distance, axis=0) 
    
    def update(self, X, labels):
        new_centroids = np.empty_like(self.centroids)
        for i in range(self.k):
            if len(X[labels == i]) == 0:
                new_centroids[i] = self.centroids[i]
            else:
                new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids
    
    def train(self, X):
        numbers, features = X.shape
        
        # randomly choose k points as centroids
        index = np.arange(numbers)
        np.random.shuffle(index)
        self.centroids = X[index[:self.k]]
        
        for i in range(self.iter):
            labels = self.assign(X)
            new_centroids = self.update(X, labels)
            #caculate the distance of each point to its centroid
            distance = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2)
            J = np.sum(np.min(distance, axis=0))
            self.array.append(J)
            if (self.centroids == new_centroids).all():
                break # stop when the centroids don't change
            
            self.centroids = new_centroids
            
    def modified_train(self, X):
        # add non-local split-and-merge moves
        numbers, features = X.shape
        
        index = np.arange(numbers)
        np.random.shuffle(index)
        self.centroids = X[index[:self.k]]
        
        for i in range(self.iter):
            labels = self.assign(X)
            new_centroids = self.update(X, labels)
            #caculate the distance of each point to its centroid
            distance = ((X - self.centroids[:, np.newaxis])**2).sum(axis=2)
            J = np.sum(np.min(distance, axis=0))
            self.array.append(J)
            if (self.centroids == new_centroids).all():
                break # stop when the centroids don't change
            
            for i in range(self.k):
                for j in range(i+1, self.k):
                    if np.linalg.norm(self.centroids[i] - self.centroids[j]) < 0.5: # this maybe a hyperparameter which need to be find the best value
                        self.centroids[i] = (self.centroids[i] + self.centroids[j]) / 2
                        self.centroids[j] = self.centroids[i]
            for i in range(self.k):
                if len(X[labels == i]) > 2: # this also maybe a hyperparameter
                    self.centroids[i] = X[labels == i].mean(axis=0)
                    self.centroids = np.append(self.centroids, [self.centroids[i]], axis=0)
                    
            self.centroids = new_centroids
        
        
    def predict(self, X):
        return self.assign(X)
    
    def normalize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
        
