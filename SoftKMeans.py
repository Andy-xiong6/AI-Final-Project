import numpy as np

class SoftKMeans:
    def __init__(self, k, iter, beta):
        self.k = k
        self.iter = iter
        self.beta = beta
        self.array = []

    def assign(self, X):
        distance = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        weights = np.exp(-self.beta * distance ** 2)
        weights /= weights.sum(axis=1)[:, np.newaxis]
        self.weights = weights
        return weights

    def update(self, X, weights):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            if np.sum(weights[:, i]) != 0: # check if the cluster is empty
                new_centroids[i] = np.sum(weights[:, i][:, np.newaxis] * X, axis=0) / np.sum(weights[:, i])
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def train(self, X):
        samples, features = X.shape

        # randomly choose k points as centroids
        index = np.arange(samples)
        np.random.shuffle(index)
        self.centroids = X[index[:self.k]]

        for _ in range(self.iter):
            weights = self.assign(X)
            new_centroids = self.update(X, weights)
            self.array.append(self.J(X))
            if (self.centroids == new_centroids).all():
                break # stop when the centroids don't change

            self.centroids = new_centroids

    def modified_train(self, X):
        samples, features = X.shape

        indices = np.arange(samples)
        np.random.shuffle(indices)
        self.centroids = X[indices[:self.k]]

        for _ in range(self.iter):
            weights = self.assign(X)
            new_centroids = self.update(X, weights)
            self.array.append(self.J(X))
            if (self.centroids == new_centroids).all():
                break
            # add non-local split-and-merge moves
            for i in range(self.k):
                for j in range(i+1, self.k):
                    if np.linalg.norm(new_centroids[i] - new_centroids[j]) < 0.5: # this maybe a hyperparameter which need to be find the best value
                        new_centroids[i] = (new_centroids[i] + new_centroids[j]) / 2
                        new_centroids[j] = new_centroids[i]
            for i in range(self.k):
                if np.sum(weights[:, i]) > 0.5: # this also maybe a hyperparameter
                    new_centroids[i] = np.sum(weights[:, i][:, np.newaxis] * X, axis=0) / np.sum(weights[:, i])
                    
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
    
    


