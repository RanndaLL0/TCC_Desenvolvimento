import numpy as np

from database.index import pegar_dados
from src.utils.split import split_data
from src.utils.symbols import to_indices


class FirstOrder:
    def __init__(self):
        self.matrix = np.zeros((3, 3), int)
        self.prob = None
        self.data_off_set = 1  # off_set para converter discretização em indice

    # função auxiliar para ver se a prob das linhas soma 1
    def _prob_lines_all_equals_one(self, data):
        sums = data.sum(axis=1, keepdims=True)
        if not np.allclose(sums, 1):
            raise ValueError(
                f"the line {np.flatnonzero(sums != 1)} did not sum probability to 1"
            )

        return True

    ## para a matriz de transição vou somar um nos dados vindos
    # para ficar com o índice correto na matriz
    # ent a coluna 0 é na verade quando houve um retorno a baixo
    def fit(self, data):
        self.matrix = np.zeros((3, 3), int)
        for i in range(1, len(data)):
            prev = data[i - 1]
            crr = data[i]
            self.matrix[prev + self.data_off_set][crr + self.data_off_set] += 1

        sums = self.matrix.sum(axis=1, keepdims=True)
        if np.any(sums == 0):
            raise ValueError(
                f"the transition for symbol {np.flatnonzero(sums == 0) - 1} was not found"
            )

        self.prob = self.matrix / sums

        if not self._prob_lines_all_equals_one(self.prob):
            raise ValueError(
                "failed on training the model. The sums of lines did not converged to one"
            )

    def _calculate_likelihood(self):
        if self.prob is None:
            raise AttributeError(
                "_calculate_likelihood should be called after function 'fit'"
            )

        ll = 0
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                ll += self.matrix[i][j] * np.log(self.prob[i][j])

        return ll

    def calculate_likelihood(self, symbols_sequence):
        if self.prob is None:
            raise AttributeError(
                "calculate_likelihood should be called after function 'fit'"
            )

        indices = to_indices(symbols_sequence)

        if len(indices) < 2:
            raise ValueError("the sequence of symbols must have at least two elements")

        sample = indices[:-1]
        target = indices[1:]

        return float(np.log(self.prob[sample, target]).sum())

    def predict_proba(self, symbols):
        if self.prob is None:
            raise AttributeError("predict_prob should be called after function 'fit'")

        indices = to_indices(symbols)
        matrix = []

        for i in indices:
            matrix.append(self.prob[i])

        return np.asarray(matrix, dtype=float)


if __name__ == "__main__":
    data = pegar_dados()
    data = np.ravel(data)
    data = np.array(data).astype(int)
    (train, test) = split_data(data, 95, 100)
    f = FirstOrder()
    f.fit(train)
    print(f.calculate_likelihood(train))
    print(f._calculate_likelihood())
    print(f.predict_proba(test))
