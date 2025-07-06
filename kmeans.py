from random import sample, shuffle

class Kmeans:
    NUM_IRIS: int = 150
    NUM_ATRIBUTOS: int = 4
    def __init__(self, iris_path: str, k: int) -> None:
        
        self.k = k
        self.pontos: list[list[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.NUM_IRIS)]
        self.classes: list[str] = ['' for _ in range(self.NUM_IRIS)]

        linha: int = 0
        with open(iris_path) as file:
            for line in file:
                valores: list[str] = line.split(',')
                self.classes[linha] = valores[-1]
                for i in range(self.NUM_ATRIBUTOS):
                    self.pontos[linha][i] = float(valores[i])
                linha += 1

        dados: list[tuple[list[float], str]] = list(zip(self.pontos, self.classes))
        shuffle(dados)
        pontos, classes = zip(*dados)
        self.pontos = list(map(list, pontos))
        self.classes = list(classes)

    def fit_predict(self) -> None:
        self.centroides: list[int] = sample(range(150), self.k)
        # self.centroides: list[int] = [25, 75, 125]
        rodadas: int = 1
        while True:
            self.grupos: list[list[int]] = [[] for _ in range(self.k)]
            init: list[int] = self.centroides.copy()

            for iris in range(self.NUM_IRIS):
                min_dist: float = float('inf')
                min_idx: int = -1
                for centroide in self.centroides:
                    dist: float = self._calculate_dist(self.pontos[iris], self.pontos[centroide])
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = centroide
                self.grupos[self.centroides.index(min_idx)].append(iris)

            for grupo_idx, grupo in enumerate(self.grupos):
                min_dist: float = float('inf')
                min_idx: int = -1
                for iris1 in grupo:
                    dist: float = 0.0
                    for iris2 in grupo:
                        dist += self._calculate_dist(self.pontos[iris1], self.pontos[iris2])
                
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = iris1

                self.centroides[grupo_idx] = min_idx

            if init == self.centroides:
                print(f'Convergiu com {rodadas}')
                break
            rodadas += 1

        return


    @classmethod
    def _calculate_dist(cls, iris1: list[float], iris2: list[float]) -> float:
        dist: float = 0.0
        for i in range(cls.NUM_ATRIBUTOS):
            dist += (iris1[i] - iris2[i]) ** 2

        return dist ** 0.5



a = Kmeans('iris/iris.data', 3)
a.fit_predict()
classes = [a.classes[i] for grupo in a.grupos for i in grupo]

for e, classe in enumerate(classes):
    if e % 50 == 0:
        print()
    print(classe)
