from typing import Self

import numpy as np
import numpy.typing as npt

from utils import Array, Labels, load_iris_data


class SOM:
    def __init__(self, map_height: int, map_width: int, n_epochs: int, learning_rate: float, sigma: float) -> None:
        self.map_height: int = map_height
        self.map_width: int = map_width
        self.n_epochs: int = n_epochs
        self.init_learning_rate: float = learning_rate
        self.init_sigma: float = sigma
        
        self.weights: Array | None = None
        self.neuron_locations: npt.NDArray[np.int_] | None = None

    def fit(self, data: Array) -> Self:
        shape: tuple[int, int] = data.shape
        n_samples, input_dim = shape
        
        self.weights = np.random.rand(self.map_height, self.map_width, input_dim)
        
        locations = np.indices((self.map_height, self.map_width)).T.reshape(-1, 2)
        self.neuron_locations = locations.reshape(self.map_height, self.map_width, 2)
        
        total_iterations = n_samples * self.n_epochs

        for epoch in range(self.n_epochs):
            shuffled_data = data.copy()
            np.random.shuffle(shuffled_data)
            
            for i, input_vector in enumerate(shuffled_data):
                current_iter = epoch * n_samples + i
                
                bmu_coords = self._find_bmu(input_vector)
                
                learning_rate = self.init_learning_rate * np.exp(-current_iter / total_iterations)
                sigma = self.init_sigma * np.exp(-current_iter / (total_iterations / 2))
                
                self._update_weights(input_vector, bmu_coords, learning_rate, sigma)

        return self

    def _find_bmu(self, input_vector: Array) -> tuple[int, int]:
        if self.weights is None:
            raise RuntimeError("Fit method must be called before finding BMU.")
        
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        bmu_flat_idx = np.argmin(distances)
        unraveled_coords = np.unravel_index(bmu_flat_idx, (self.map_height, self.map_width))
        return (int(unraveled_coords[0]), int(unraveled_coords[1]))

    def _update_weights(self, input_vector: Array, bmu_coords: tuple[int, int], learning_rate: float, sigma: float) -> None:
        if self.weights is None or self.neuron_locations is None:
            raise RuntimeError("Fit method must be called before updating weights.")
        
        grid_dist = np.linalg.norm(self.neuron_locations - bmu_coords, axis=2)
        influence = np.exp(-grid_dist**2 / (2 * sigma**2))
        influence_expanded = np.expand_dims(influence, axis=-1)
        
        self.weights += influence_expanded * learning_rate * (input_vector - self.weights)
        
    def predict(self, data: Array) -> Labels:
        if self.weights is None:
            raise RuntimeError("Fit method must be called before prediction.")
        
        bmu_indices = np.apply_along_axis(self._find_bmu, 1, data)
        return bmu_indices[:, 0] * self.map_width + bmu_indices[:, 1]


if __name__ == '__main__':
    iris_data, iris_classes = load_iris_data('iris/iris.data')
    map_height: int = 1
    map_width: int = 3
    n_epochs: int = 100
    learning_rate: float = 0.5
    sigma: float = 1.0

    som = SOM(map_height=map_height, map_width=map_width, n_epochs=n_epochs, learning_rate=learning_rate, sigma=sigma)
    som.fit(iris_data)
    clusters = som.predict(iris_data)

    n_clusters = len(np.unique(clusters))

    print(f"Parâmetros: Grade={map_height}x{map_width}, Épocas={n_epochs}, LR={learning_rate}, Sigma={sigma}")
    print(f"Número de clusters (neurônios ativados): {n_clusters}")

    print("\nExemplo de associação de pontos a clusters (neurônios):")
    for i in range(15):
        print(f"Ponto {i:03d} (Classe: {iris_classes[i]:<15}) -> Cluster (Neurônio ID): {clusters[i]}")
