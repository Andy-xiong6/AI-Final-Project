import numpy as np
from KMeans import KMeans

# Load data
dataset = np.loadtxt('seeds_dataset.txt')
labels = dataset[:, -1]
X = dataset[:, :-1]

# KMeans algorithm
kmeans = KMeans(3, 1000)
X_norm = kmeans.normalize(X)
kmeans.train(X)
print(kmeans.predict(X))

kmeans.train(X_norm)
print(kmeans.predict(X_norm))

