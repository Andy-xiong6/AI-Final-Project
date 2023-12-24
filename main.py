import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from KMeans import KMeans
from SoftKMeans import SoftKMeans
from PCA import PCA
from Linearautoencoder import linearautoencoder

# Load data
dataset = np.loadtxt('seeds_dataset.txt')
labels = dataset[:, -1]
X = dataset[:, :-1]

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
    
# KMeans algorithm and Soft KMeans algorithm
print("KMeans with k = 3: ")
kmeans = KMeans(3, 10)
X_norm = kmeans.normalize(X)
kmeans.train(X)

print("precision: ")
precision(kmeans.predict(X))
print("Soft KMeans with k = 3: ")
softkmeans = SoftKMeans(3, 10, 10)
softkmeans.train(X)

print("precision: ")
precision(softkmeans.predict(X))
print("J: ", softkmeans.J(X))
#plot the kmeans with k=3 and soft kmeans with k = 3 in a figure
plt.plot(softkmeans.array, label='Soft KMeans')
plt.plot(kmeans.array, label='KMeans')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('KMeans and Soft KMeans with k = 3')
plt.legend()
plt.show()

# KMeans with K=10
kmeans = KMeans(10, 10)
kmeans.train(X)
# modified KMeans algorithm with K = 10
kmeans_modified = KMeans(10,20)
kmeans_modified.modified_train(X)
# plot the kmeans with k=10 and modified kmeans with k = 10 in a figure
plt.plot(kmeans_modified.array, label='modified KMeans')
plt.plot(kmeans.array, label='KMeans')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('KMeans and modified KMeans with k = 10')
plt.legend()
plt.show()

#Soft KMeans with k=10
softkmeans = SoftKMeans(10, 15, 15)
softkmeans.train(X)
# modified Soft KMeans algorithm with K = 10
skmeans_modified = SoftKMeans(10, 25, 15)
skmeans_modified.modified_train(X)
# plot the soft kmeans with k=10 and modified soft kmeans with k = 10 in a figure
plt.plot(skmeans_modified.array, label='modified Soft KMeans')
plt.plot(softkmeans.array, label='Soft KMeans')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('Soft KMeans and modified Soft KMeans with k = 10')
plt.legend()
plt.show()

# plot the kmeans with k=10 and soft kmeans with k = 10 in a figure
plt.plot(kmeans.array, label='KMeans')
plt.plot(softkmeans.array, label='Soft KMeans')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('KMeans and Soft KMeans with k = 10')
plt.legend()
plt.show()

#plot the modified kmeans and modified soft kmeans with k=10 in a figure
plt.plot(kmeans_modified.array, label='modified KMeans')
plt.plot(skmeans_modified.array, label='modified Soft KMeans')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('modified KMeans and modified Soft KMeans with k = 10')
plt.legend()
plt.show()

# PCA algorithm
raw_image = Image.open('flower.png')
image = np.array(raw_image)
image = image/255  # normalize
pca = PCA(60)
Z = pca.train(image)
recover_image = pca.recover(Z)
plt.imshow(image, cmap='gray')
plt.title('original image')
plt.show()
plt.imshow(recover_image, cmap='gray')
plt.title('PCA recover image')
plt.show()

# KMeans on image
print("KMeans on image: ")
kmeans = KMeans(60, 100)
X_norm = kmeans.normalize(image)
kmeans.train(image)
recover_image = kmeans.centroids[kmeans.predict(image)]
plt.imshow(recover_image, cmap='gray')
plt.title('KMeans recover image')
plt.show()

# Linear Auto Encoder
lae = linearautoencoder([30], 100, 100)
lae.train(image, image, 0.00001, 1000000)
print("J: ", lae.loss(image, image))
recover_image = lae.forward_propagation(image)[-1]
plt.imshow(recover_image, cmap='gray')
plt.title('Linear Auto Encoder recover image')
plt.show()



