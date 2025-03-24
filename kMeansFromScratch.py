import numpy as np
import matplotlib.pyplot as plt

# Creating clusterable data
x= np.arange(0,30,1).reshape(-1,1)
y=[]
y.extend(np.random.randn(10,))
y.extend(np.random.randn(10,)+5)
y.extend(np.random.randn(10,)+10)
y=np.array(y)
y=y.reshape(-1,1)
x=np.concatenate((x,y),axis=1)
labels = [0 for _ in range(10)]
labels.extend([1 for _ in range(10)])
labels.extend([2 for _ in range(10)])
plt.scatter(x[:,0],x[:,1],c=labels)
plt.show()


#k Means

#Define number of clusters
num_clusters = 3

#Define 3 random points from the dataset to be centroids
centroids = [x[i,:] for i in np.random.randint(0,30,(3,))]

#Initialize empty clusters
clusters={0:[],1:[],2:[]}

num_points=30
cluster_of=[0]*30

# Run the whole algo for 10 iterations
for _ in range(10):

  for i in range(num_points):
    distances = [np.sqrt(np.sum((x[i]-centroids[j])**2)) for j in range(3)]

    # print(distances)
    # print("\n")
    distances =np.array(distances)
    min_cluster = distances.argmin()
    clusters[min_cluster].append(x[i])
    cluster_of[i]=min_cluster

  for j in range(num_clusters):
    if clusters[j]!=[]:
      new_centroid = np.mean(clusters[j])
      centroids[j] = new_centroid

plt.scatter(x[:,0],x[:,1],c=cluster_of)


