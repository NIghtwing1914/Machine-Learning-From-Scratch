import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

def generate_clusterable_data(n_samples=500, n_features=2, n_clusters=4, cluster_std=1.0, random_state=42):

    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                       cluster_std=cluster_std, random_state=random_state)
    return X

def plot_clusters(X, labels):
    
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k')
    plt.title("Clustered Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def agglomerative_clustering_from_scratch(X, n_clusters=4):
   
    clusters = {i: [i] for i in range(len(X))}  
    distances = cdist(X, X)  
    np.fill_diagonal(distances, np.inf) 
    
    while len(clusters) > n_clusters:
        # Find closest pair of clusters
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        c1, c2 = min_idx
        
        # Merge them
        clusters[c1].extend(clusters[c2])
        del clusters[c2]
        
        
        for i in clusters:
            if i != c1:
                new_dist = np.min([distances[p1, p2] for p1 in clusters[c1] for p2 in clusters[i]])
                distances[c1, i] = distances[i, c1] = new_dist
        
        distances[:, c2] = distances[c2, :] = np.inf  
    
    # Assign final labels
    labels = np.zeros(len(X), dtype=int)
    for cluster_id, points in enumerate(clusters.values()):
        for point in points:
            labels[point] = cluster_id
    
    return labels


data = generate_clusterable_data()

cluster_labels = agglomerative_clustering_from_scratch(data)

# Plot results
plot_clusters(data, cluster_labels)
