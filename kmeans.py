import numpy as np
import random
from typing import Self, TypeAlias

import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.float64]
Labels: TypeAlias = npt.NDArray[np.int_]

class Kmeans:
    def __init__(self, k: int, tol: float = 1e-4, max_iter: int = 100) -> None:
        self.k: int = k
        self.tol: float = tol
        self.max_iter: int = max_iter
        self.centroids: Array | None = None
        self.labels_: Labels | None = None

    def fit(self, data: Array) -> Self:
        self.centroids = self._initialize_centroids(data)

        for _ in range(self.max_iter):
            self.labels_ = self._assign_clusters(data)
            
            new_centroids = self._update_centroids(data)

            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
            
            self.centroids = new_centroids
        
        return self

    def _initialize_centroids(self, data: Array) -> Array:
        initial_indices = random.sample(range(data.shape[0]), self.k)
        centroids = data[initial_indices]
        return centroids

    def _assign_clusters(self, data: Array) -> Labels:
        if self.centroids is None:
            raise RuntimeError("Fit method must be called before assigning clusters.")
        
        distances = np.zeros((data.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(data - centroid, axis=1)
        
        labels = np.argmin(distances, axis=1).astype(np.int_)
        return labels

    def _update_centroids(self, data: Array) -> Array:
        if self.labels_ is None:
            raise RuntimeError("Labels not found. Cannot update centroids.")

        new_centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            cluster_points = data[self.labels_ == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
        return new_centroids

    def predict(self, data: Array) -> Labels:
        return self._assign_clusters(data)

def load_iris_data(path: str) -> tuple[Array, npt.NDArray[np.str_]]:
    points: list[list[float]] = []
    classes: list[str] = []
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            points.append([float(v) for v in parts[:-1]])
            classes.append(parts[-1])
    
    return np.array(points, dtype=np.float64), np.array(classes)

if __name__ == '__main__':
    iris_data, iris_classes = load_iris_data('iris/iris.data')
    
    kmeans = Kmeans(k=3)
    
    kmeans.fit(iris_data)
    
    if kmeans.centroids is not None:
        clusters = kmeans.predict(iris_data)
        
        print("Centroides finais encontrados:")
        print(np.round(kmeans.centroids, 2))
        
        print("\nExemplo de associação de pontos a clusters:")
        for i in range(10):
            print(f"Ponto {i} (Classe Original: {iris_classes[i]}) -> Cluster: {clusters[i]}")
