import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k
        
    def normalize(self, X):
        self.X = X
        X_norm = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        return X_norm
    
    def train(self, X):
        X_norm = self.normalize(X)
        covariance_matrix = np.cov(X_norm.T)
        U, S, V = np.linalg.svd(covariance_matrix)
        self.S = S
        self.V = V
        self.U = U
        self.Z = X_norm @ U[:, :self.k]
        return self.Z
    
    def predict(self, X):
        # This function is used to predict the new data
        X_norm = self.normalize(X)
        return X_norm @ self.U[:, :self.k]
    
    def recover(self, Z):
        # This function is used to recover the original data
        return Z @ self.U[:, :self.k].T
    
    def J(self, X):
        X_norm = self.normalize(X)
        return np.sum((X_norm - self.recover(self.predict(X)))**2)