from typing import TypeAlias

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

Array: TypeAlias = npt.NDArray[np.float64]
Labels: TypeAlias = npt.NDArray[np.int_]

def load_iris_data(path: str) -> tuple[Array, npt.NDArray[np.str_]]:
    points: list[list[float]] = []
    classes: list[str] = []
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            parts: list[str] = line.strip().split(',')
            points.append([float(v) for v in parts[:-1]])
            classes.append(parts[-1])
    
    return np.array(points, dtype=np.float64), np.array(classes)


def plot_clusters(data: Array, labels: Labels, title: str, filename: str | None = None) -> None:
    pca: PCA = PCA(n_components=2)
    data_2d = pca.fit_transform(data)  # type: ignore

    unique_labels = np.unique(labels)
    
    plt.figure(figsize=(10, 7))  # type: ignore
    
    for label in unique_labels:
        if label == -1:
            color = 'k'
            point_label = 'Ruído'
        else:
            color = plt.cm.viridis(float(label) / np.max(unique_labels))  # type: ignore
            point_label = f'Cluster {label}'
            
        plt.scatter(   # type: ignore
            data_2d[labels == label, 0], 
            data_2d[labels == label, 1],
            color=color, 
            label=point_label,
            alpha=0.8,
            edgecolors='k',
            linewidth=0.5
        )

    plt.title(title, fontsize=16)  # type: ignore
    plt.xlabel('Componente Principal 1')  # type: ignore
    plt.ylabel('Componente Principal 2')  # type: ignore
    plt.legend()  # type: ignore
    plt.grid(True, linestyle='--', alpha=0.6)  # type: ignore
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # type: ignore

