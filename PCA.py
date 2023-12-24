import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k
        # k is the number of principal components
        
    def normalize(self, X):
        self.X = X
        X_norm = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        return X_norm
    
    def train(self, X):
        covariance_matrix = np.cov(X.T)
        U, S, V = np.linalg.svd(covariance_matrix)
        self.S = S
        self.V = V
        self.U = U
        self.Z = X @ U[:, :self.k]
        return self.Z
    
    def predict(self, X):
        return self.normalize(X)@ self.U[:, :self.k]
    
    def recover(self, Z):
        return Z @ self.U[:, :self.k].T
    
    def J(self, X):
        return np.sum((X- self.recover(self.predict(X)))**2)