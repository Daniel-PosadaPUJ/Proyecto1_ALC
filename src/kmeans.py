import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, distance_func=None, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_func = distance_func if distance_func else self._p_distance

    def fit(self, X):
        i = 0
        change = np.inf
        self.centroids = self._initialize_centroids(X)
        while i < self.max_iter and change > self.tol:
            self.labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            self.centroids = new_centroids
            change = np.mean(np.abs(new_centroids - self.centroids, axis=1))
            i += 1

    def _initialize_centroids(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        return min_vals + (max_vals - min_vals) * np.random.rand(self.n_clusters, X.shape[1])

    def _assign_clusters(self, X):
        distances = self.distance_func(X[:, np.newaxis], self.centroids)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        centroids = []
        for i in range(self.n_clusters):
            centroids.append(X[self.labels == i].mean(axis=0))
        return np.array(centroids)

    def predict(self, X):
        return self._assign_clusters(X)

    def inertia(self, X):
        distancias = self.distance_func(X, self.centroids[self.labels])
        return np.sum(distancias)

    def _p_distance_to_inertia(self, X, centroids, p=2):
        return np.sum(np.abs(X - centroids) ** p, axis=2)