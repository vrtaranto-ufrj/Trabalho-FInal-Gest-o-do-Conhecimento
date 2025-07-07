import os

import numpy as np
import numpy.typing as npt

from dbscan import DBSCAN
from kmeans import Kmeans
from som import SOM
from utils import Array, Labels, load_iris_data, plot_clusters


def main() -> None:
    """
    Script principal para executar a análise comparativa dos algoritmos
    de agrupamento K-means, DBSCAN e SOM no dataset Iris.
    """
    output_dir: str = 'graficos'
    os.makedirs(output_dir, exist_ok=True)

    # Carrega os dados uma única vez
    iris_data_tuple: tuple[Array, npt.NDArray[np.str_]] = load_iris_data('iris/iris.data')
    iris_data, iris_classes = iris_data_tuple

    # --- Visualização dos Dados Originais (Ground Truth) ---
    print("Gerando gráfico das classes reais do dataset Iris...")
    _, ground_truth_labels = np.unique(iris_classes, return_inverse=True)
    
    title: str = 'Visualização do Dataset Iris (Classes Reais)'
    filename: str = os.path.join(output_dir, 'iris_classes_ground_truth.png')
    plot_clusters(iris_data, ground_truth_labels, title, filename)

    print("Iniciando análise do K-means...")
    k_values: list[int] = [2, 3, 4, 5]
    for k in k_values:
        print(f"  Testando K-means com k={k}...")
        kmeans: Kmeans = Kmeans(k=k)
        labels: Labels = kmeans.fit(iris_data).predict(iris_data)
        
        title: str = f'K-means com k={k}'
        filename: str = os.path.join(output_dir, f'kmeans_k_{k}.png')
        plot_clusters(iris_data, labels, title, filename)

    print("\nIniciando análise do DBSCAN...")
    # Experimento 1: Variando eps, min_pts fixo
    min_pts_fixed: int = 3
    eps_values: list[float] = [0.4, 0.5, 0.7]
    for eps in eps_values:
        print(f"  Testando DBSCAN com eps={eps} e min_pts={min_pts_fixed}...")
        dbscan: DBSCAN = DBSCAN(eps=eps, min_pts=min_pts_fixed)
        labels: Labels = dbscan.fit_predict(iris_data)

        n_clusters: int = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise: int = sum(1 for label in labels if label == -1)
        
        title: str = f'DBSCAN com eps={eps}, min_pts={min_pts_fixed}\n(Clusters: {n_clusters}, Ruído: {n_noise})'
        filename: str = os.path.join(output_dir, f'dbscan_eps_{eps}_minpts_{min_pts_fixed}.png')
        plot_clusters(iris_data, labels, title, filename)


    # Experimento 2: Variando min_pts, eps fixo
    eps_fixed: float = 0.5
    min_pts_values: list[int] = [3, 5, 10]
    for min_pts in min_pts_values:
        print(f"  Testando DBSCAN com eps={eps_fixed} e min_pts={min_pts}...")
        dbscan: DBSCAN = DBSCAN(eps=eps_fixed, min_pts=min_pts)
        labels: Labels = dbscan.fit_predict(iris_data)

        n_clusters: int = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise: int = sum(1 for label in labels if label == -1)

        title: str = f'DBSCAN com eps={eps_fixed}, min_pts={min_pts}\n(Clusters: {n_clusters}, Ruído: {n_noise})'
        filename: str = os.path.join(output_dir, f'dbscan_eps_{eps_fixed}_minpts_{min_pts}.png')
        plot_clusters(iris_data, labels, title, filename)

    print("\nIniciando análise do SOM...")
    grid_sizes: list[tuple[int, int]] = [(1, 2), (1, 3), (2, 2), (3, 3)]
    n_epochs: int = 100
    learning_rate: float = 0.5
    sigma: float = 1.0

    for grid in grid_sizes:
        h, w = grid
        print(f"  Testando SOM com grade {h}x{w}...")
        som: SOM = SOM(map_height=h, map_width=w, n_epochs=n_epochs, learning_rate=learning_rate, sigma=sigma)
        labels: Labels = som.fit(iris_data).predict(iris_data)

        title: str = f'SOM com Grade {h}x{w}'
        filename: str = os.path.join(output_dir, f'som_grid_{h}x{w}.png')
        plot_clusters(iris_data, labels, title, filename)
        
    print("\nAnálise concluída! Verifique a pasta 'graficos' para os resultados.")


if __name__ == '__main__':
    main()
