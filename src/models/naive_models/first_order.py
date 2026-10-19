import numpy as np

from src.utils.symbols import to_indices


class FirstOrder:
    def __init__(self):
        self.matrix = np.zeros((3, 3), int)
        self.prob = None

    # função auxiliar para ver se a prob das linhas soma 1
    def _prob_lines_all_equals_one(self, data):
        sums = data.sum(axis=1, keepdims=True)
        if not np.allclose(sums, 1):
            elements = np.any(~np.isclose(sums, 1), axis=1)
            raise ValueError(
                f"the line {np.flatnonzero(elements)} did not sum probability to 1"
            )

        return True

    ## para a matriz de transição vou somar um nos dados vindos
    # para ficar com o índice correto na matriz
    # ent a coluna 0 é na verade quando houve um retorno a baixo
    def fit(self, data):
        arr = to_indices(data)
        prev = arr[:-1]
        next = arr[1:]

        code = prev * 3 + next
        count = np.bincount(code, minlength=9)
        matrix = count.reshape(3, 3)

        sums = matrix.sum(axis=1, keepdims=True)
        if np.any(sums == 0):
            raise ValueError(
                f"the transition for symbol {np.flatnonzero(sums == 0) - 1} was not found"
            )

        prob = matrix / sums

        if not self._prob_lines_all_equals_one(prob):
            raise ValueError(
                "failed on training the model. The sums of lines did not converged to one"
            )

        self.matrix = matrix
        self.prob = prob

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
            raise AttributeError("predict_proba should be called after function 'fit'")

        arr = np.asarray(symbols)

        if arr.size == 0:
            raise ValueError("the parameter 'symbols' must have at least one element")

        indices = to_indices(arr)
        matrix = self.prob[indices]

        return np.asarray(matrix, dtype=float)
