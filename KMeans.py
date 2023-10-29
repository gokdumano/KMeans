import numpy as np

class KMeans:
	def __init__(self, n_clusters=8, max_iters=300):
		self.n_clusters = n_clusters
		self.max_iters = max_iters
	def fit(self, X):
		# 1. Randomly initialize the centroids
		centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
		
		for n_iter in range(self.max_iters):
            # 2. Assign each data point to the closest centroid
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1), axis=-1)
            
            # 3. Recalculate the centroids as the mean of all data points assigned to each centroid
            new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(self.n_clusters)])
            
            # 4. Check for convergence
            if np.all(centroids == new_centroids): break
            
            centroids = new_centroids
            
        self.labels_ = labels
        self.n_iter_ = n_iter
        self.cluster_centers_ = centroids
        
    def predict(self, X):
        if not isinstance(X, np.ndarray): X = np.array(X)
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=-1), axis=-1)
        return  labels

if __name__ == '__main__':
  X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
  km = KMeans(2, 100)
  km.fit(X)
  
  print(km.labels_)
  print(km.n_iter_)
  print(km.cluster_centers_)
  
  print(km.predict([[0, 0], [12, 3]]))
