from typing import Self

import numpy as np

from utils import Array, Labels, load_iris_data

NOISE_LABEL = -1
UNCLASSIFIED_LABEL = -2

class DBSCAN:
    def __init__(self, eps: float, min_pts: int) -> None:
        self.eps: float = eps
        self.min_pts: int = min_pts
        self.labels_: Labels | None = None

    def fit(self, data: Array) -> Self:
        n_points: int = data.shape[0]
        self.labels_ = np.full(n_points, UNCLASSIFIED_LABEL, dtype=np.int_)
        cluster_id = 0

        for point_idx in range(n_points):
            if self.labels_[point_idx] != UNCLASSIFIED_LABEL:
                continue

            neighbors_indices = self._region_query(data, point_idx)

            if len(neighbors_indices) < self.min_pts:
                self.labels_[point_idx] = NOISE_LABEL
                continue

            self._expand_cluster(data, point_idx, neighbors_indices, cluster_id)
            cluster_id += 1
        
        return self

    def _expand_cluster(self, data: Array, start_point_idx: int, neighbors_indices: list[int], cluster_id: int) -> None:
        if self.labels_ is None:
            raise Exception('labels não pode ser null')
        
        self.labels_[start_point_idx] = cluster_id
        
        queue = neighbors_indices
        head = 0
        
        while head < len(queue):
            current_point_idx = queue[head]
            head += 1

            if self.labels_[current_point_idx] == NOISE_LABEL:
                self.labels_[current_point_idx] = cluster_id

            if self.labels_[current_point_idx] != UNCLASSIFIED_LABEL:
                continue

            self.labels_[current_point_idx] = cluster_id
            
            new_neighbors_indices = self._region_query(data, current_point_idx)

            if len(new_neighbors_indices) >= self.min_pts:
                queue.extend(new_neighbors_indices)

    def _region_query(self, data: Array, point_idx: int) -> list[int]:
        distances = np.linalg.norm(data - data[point_idx], axis=1)
        return np.where(distances < self.eps)[0].tolist()

    def fit_predict(self, data: Array) -> Labels:
        self.fit(data)
        if self.labels_ is None:
            raise RuntimeError("Fit method failed to produce labels.")
        return self.labels_



if __name__ == '__main__':
    iris_data, iris_classes = load_iris_data('iris/iris.data')
    min_pts: int = 5
    eps: float = 0.5

    dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    clusters = dbscan.fit_predict(iris_data)

    n_clusters = len(set(clusters)) - (1 if NOISE_LABEL in clusters else 0)
    n_noise = np.sum(clusters == NOISE_LABEL)

    print(f"Parâmetros: eps={eps}, min_pts={min_pts}")
    print(f"Número de clusters encontrados: {n_clusters}")
    print(f"Número de pontos de ruído (noise): {n_noise}")

    print("\nExemplo de associação de pontos a clusters:")
    for i in range(15):
        print(f"Ponto {i:03d} (Classe: {iris_classes[i]:<15}) -> Cluster: {clusters[i]}")