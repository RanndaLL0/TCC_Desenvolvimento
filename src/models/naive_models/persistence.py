import numbers

import numpy as np

from src.utils.symbols import SYMBOLS, to_indices


## Baseline de persistência: prevê que o próximo símbolo é igual ao atual.
# A distribuição pura seria one-hot (log(0) quando erra), então suavizamos:
# P(igual ao atual) = 1 - 2 * epsilon e P(cada um dos outros dois) = epsilon.
# O epsilon é uma escolha metodológica e afeta a LL e o BIC
class Persistence:
    def __init__(self, epsilon=0.01):
        if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
            raise TypeError("epsilon should be a real number")

        # 1 - 2 * epsilon > epsilon exige epsilon < 1/3 (em 1/3 vira o uniforme)
        if not 0 < epsilon < 1 / 3:
            raise ValueError("epsilon should be in the interval (0, 1/3)")

        self.epsilon = float(epsilon)
        self.prob = None

    def fit(self, data):
        if len(to_indices(data)) < 2:
            raise ValueError("the sequence of symbols must have at least two elements")

        n = len(SYMBOLS)
        prob = np.full((n, n), self.epsilon)
        np.fill_diagonal(prob, 1 - 2 * self.epsilon)
        self.prob = prob

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
