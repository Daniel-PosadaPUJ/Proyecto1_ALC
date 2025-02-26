import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, distance_func=None, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_func = distance_func if distance_func else self._default_distance

    def fit(self, X):
        i = 0
        change = np.inf
        self.centroids = self._initialize_centroids(X)
        while i < self.max_iter and change > self.tol:
            self.labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            change = np.mean(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            i += 1

    def _initialize_centroids(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        min_vals = X.min().values
        max_vals = X.max().values
        aleatory_points = np.random.rand(self.n_clusters, X.shape[1])
        centroids = min_vals + (max_vals - min_vals) * aleatory_points
        return centroids


    def _assign_clusters(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                value = X.iloc[i].values
                centroid = self.centroids[j]
                distances[i, j] = self.distance_func(value, centroid)
        return np.argmin(distances, axis=1)


    def _default_distance(self, x, y):
        return np.linalg.norm(x - y)

    def _compute_centroids(self, X):
        centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                centroids.append(cluster_points.mean(axis=0))
            else:
                # If a cluster is empty, keep the previous centroid
                centroids.append(self.centroids[i])
        return np.array(centroids)

    def predict(self, X):
       return self._assign_clusters(X)

    def inertia(self, X):
        distancias = np.array([self.distance_func(X, centroid) for centroid in self.centroids])
        return np.sum(np.min(distancias, axis=0))


