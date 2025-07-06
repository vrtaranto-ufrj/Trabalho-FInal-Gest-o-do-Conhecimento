from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.float64]
Labels: TypeAlias = npt.NDArray[np.int_]

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