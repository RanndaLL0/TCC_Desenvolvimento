import numpy as np

from src.utils.symbols import SYMBOLS, to_indices


## Baseline aleatório: todo símbolo tem probabilidade 1/3, independente do contexto.
# Serve de piso de comparação: LL = (N - 1) * log(1/3) para N símbolos
class Uniform:
    def __init__(self):
        self.prob = None

    def fit(self, data):
        if len(to_indices(data)) < 2:
            raise ValueError("the sequence of symbols must have at least two elements")

        n = len(SYMBOLS)
        self.prob = np.full((n, n), 1 / n)

    def calculate_likelihood(self, symbols_sequence):
        if self.prob is None:
            raise AttributeError(
                "calculate_likelihood should be called after function 'fit'"
            )

        indices = to_indices(symbols_sequence)
        if len(indices) < 2:
            raise ValueError("the sequence of symbols must have at least two elements")

        context = indices[:-1]
        target = indices[1:]

        return float(np.log(self.prob[context, target]).sum())

    # linha i é a distribuição do símbolo que vem depois de data[i]
    # (N linhas para N símbolos; a última não tem resposta conhecida no teste)
    def predict_prob(self, data):
        if self.prob is None:
            raise AttributeError("predict_prob should be called after function 'fit'")

        return self.prob[to_indices(data)]
