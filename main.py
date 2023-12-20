import numpy as np
from KMeans import KMeans
from SoftKMeans import SoftKMeans
    
# Load data
dataset = np.loadtxt('seeds_dataset.txt')
labels = dataset[:, -1]
X = dataset[:, :-1]

print("Kmeans with k = 3: ")
def precision(X):
    cluster1 = X[0:70]
    cluster2 = X[71:140]
    cluster3 = X[141:210]
    precision_cluster1 = np.unique(cluster1, return_counts=True)[1].max()/70
    precision_cluster2 = np.unique(cluster2, return_counts=True)[1].max()/70
    precision_cluster3 = np.unique(cluster3, return_counts=True)[1].max()/70
    print("cluster 1: ", precision_cluster1)
    print("cluster 2: ", precision_cluster2)
    print("cluster 3: ", precision_cluster3)
    
# KMeans algorithm
kmeans = KMeans(3, 1000)
X_norm = kmeans.normalize(X)
kmeans.train(X)
print(kmeans.predict(X))
print("precision: ")
precision(kmeans.predict(X))
print("distance: ", kmeans.distance(X))

print("Soft Kmeans with k = 3: ")
# SoftKMeans algorithm
softkmeans = SoftKMeans(3, 1000, 0.2)
softkmeans.train(X)
print(softkmeans.predict(X))
print("precision: ")
precision(softkmeans.predict(X))


