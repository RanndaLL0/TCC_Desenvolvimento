import numbers
import warnings

import numpy as np
from numba import get_num_threads, njit, prange

from src.utils.symbols import SYMBOLS, to_indices

M = len(SYMBOLS)


## MTD (Mixture Transition Distribution) de ordem l com alfabeto de M símbolos:
#   P(X_t = j | x_{t-1}, ..., x_{t-l}) = sum_{g=1..l} lam[g-1] * Q[x_{t-g}, j]
# Q é UMA matriz de transição MxM (linhas somam 1) compartilhada por todos os
# lags e lam é o vetor de pesos dos lags (simplex). Com l = 1 o modelo é
# exatamente a cadeia de primeira ordem. O treino é feito por EM, tratando
# "qual lag gerou X_t" como variável latente (Lèbre & Bourguignon, 2008)


## Passo E sobre as transições t em [start, end). Para cada t calcula as
# responsabilidades r_g = lam[g] * Q[x[t-1-g], j] / P e acumula em
# counts[i, j] (contagem ponderada por par lag->alvo) e em lam_num[g].
# Devolve o log da verossimilhança do trecho
@njit(cache=True)
def _accumulate(x, lam, Q, order, start, end, counts, lam_num):
    ll = 0.0
    r = np.empty(order)

    for t in range(start, end):
        j = x[t]
        s = 0.0
        for g in range(order):
            r[g] = lam[g] * Q[x[t - 1 - g], j]
            s += r[g]

        ll += np.log(s)
        inv = 1.0 / s
        for g in range(order):
            w = r[g] * inv
            counts[x[t - 1 - g], j] += w
            lam_num[g] += w

    return ll


## Divide as transições (t = order .. T-1) em blocos independentes, cada um com
# seus acumuladores locais, e soma no final. Mesmo padrão do hmm_numba.py
@njit(parallel=True, cache=True)
def e_step(x, lam, Q, order, n_blocks):
    T = x.shape[0]
    m = Q.shape[0]

    counts_loc = np.zeros((n_blocks, m, m))
    lam_loc = np.zeros((n_blocks, order))
    ll_loc = np.zeros(n_blocks)

    size = (T - order + n_blocks - 1) // n_blocks

    for p in prange(n_blocks):
        start = order + p * size
        end = min(start + size, T)
        if start < end:
            ll_loc[p] = _accumulate(
                x, lam, Q, order, start, end, counts_loc[p], lam_loc[p]
            )

    counts = np.zeros((m, m))
    lam_num = np.zeros(order)
    ll = 0.0
    for p in range(n_blocks):
        counts += counts_loc[p]
        lam_num += lam_loc[p]
        ll += ll_loc[p]

    return ll, counts, lam_num


## Passo M: lam é a média das responsabilidades e Q é a contagem ponderada
# normalizada por linha. O piso funciona como pseudocontagem: evita linha de Q
# zerada quando um símbolo nunca aparece como lag e mantém P > 0 (log finito)
def m_step(counts, lam_num, n_obs, floor):
    m = counts.shape[0]
    lam = lam_num / n_obs
    Q = (counts + floor) / (counts.sum(axis=1, keepdims=True) + m * floor)
    return lam, Q


## Roda o EM a partir de (lam, Q) e devolve (lam, Q, ll, info).
# x: índices (0..M-1). max_iter conta passos M. A LL devolvida é sempre a dos
# parâmetros devolvidos; se um passo fizer a LL cair além da tolerância, avisa
# e devolve os parâmetros anteriores
def em(
    x,
    lam,
    Q,
    order,
    max_iter=1000,
    tolerance=1e-8,
    n_blocks=None,
    floor=1e-10,
    verbose=False,
):
    x = np.ascontiguousarray(x, dtype=np.int64)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)

    if n_blocks is None:
        n_blocks = get_num_threads()

    n_obs = x.shape[0] - order

    ll, counts, lam_num = e_step(x, lam, Q, order, n_blocks)
    history = [ll]
    converged = False
    decreased = False

    for it in range(max_iter):
        new_lam, new_Q = m_step(counts, lam_num, n_obs, floor)
        new_ll, new_counts, new_lam_num = e_step(x, new_lam, new_Q, order, n_blocks)

        if verbose:
            print(f"iteration {it:3d} ll = {new_ll:.4f}")

        # a variação é relativa, como no Baum-Welch
        variation = (new_ll - ll) / abs(ll)
        if variation < -tolerance:
            warnings.warn(
                f"the log-likelihood decreased at iteration {it} "
                f"({ll:.6f} -> {new_ll:.6f}); the EM should never get worse"
            )
            decreased = True
            break

        lam, Q, ll = new_lam, new_Q, new_ll
        counts, lam_num = new_counts, new_lam_num
        history.append(ll)

        if variation < tolerance:
            converged = True
            break

    info = {
        "iterations": len(history) - 1,
        "converged": converged,
        "decreased": decreased,
        "history": np.array(history),
    }
    return lam, Q, ll, info


## Ponto de partida do reinício 0: lam uniforme e Q igual à cadeia de primeira
# ordem empírica (determinístico). Os demais reinícios sorteiam de uma Dirichlet
def initial_guess(x, order, restart, seed_sequence, floor, concentration=5.0):
    if restart == 0:
        counts = np.bincount(x[:-1] * M + x[1:], minlength=M * M).reshape(M, M)
        Q = (counts + floor) / (counts.sum(axis=1, keepdims=True) + M * floor)
        return np.full(order, 1.0 / order), Q

    rng = np.random.default_rng(seed_sequence)
    lam = rng.dirichlet(np.ones(order) * concentration)
    Q = rng.dirichlet(np.ones(M) * concentration, size=M)
    return lam, Q


class MTD:
    def __init__(
        self,
        order,
        n_restarts=5,
        seed=0,
        max_iter=1000,
        tolerance=1e-8,
        floor=1e-10,
        n_blocks=None,
    ):
        for name, value in (("order", order), ("n_restarts", n_restarts)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} should be an integer")
            if value < 1:
                raise ValueError(f"{name} should be at least 1")

        if floor <= 0:
            raise ValueError("floor should be positive")

        self.order = int(order)
        self.n_restarts = int(n_restarts)
        self.seed = seed
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.floor = floor
        self.n_blocks = n_blocks

        self.lam = None
        self.Q = None
        self.ll = None
        self.n_obs = None
        self.info = None

    ## m(m-1) da matriz Q mais (l-1) pesos livres (lam soma 1)
    @property
    def n_params(self):
        return M * (M - 1) + (self.order - 1)

    def _check_fitted(self, caller):
        if self.Q is None:
            raise AttributeError(f"{caller} should be called after function 'fit'")

    ## Os atributos só são atribuídos no final: se o fit falhar, o estado
    # anterior continua consistente
    def fit(self, data):
        x = to_indices(data)

        if len(x) <= self.order:
            raise ValueError(
                f"the sequence must have more than {self.order} symbols (the order)"
            )

        seeds = np.random.SeedSequence(self.seed).spawn(self.n_restarts)

        best = None
        restarts = []
        for r, ss in enumerate(seeds):
            lam0, Q0 = initial_guess(x, self.order, r, ss, self.floor)
            lam, Q, ll, info = em(
                x,
                lam0,
                Q0,
                self.order,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                n_blocks=self.n_blocks,
                floor=self.floor,
            )
            restarts.append(
                {
                    "ll": ll,
                    "iterations": info["iterations"],
                    "converged": info["converged"],
                }
            )
            if best is None or ll > best[2]:
                best = (lam, Q, ll, {**info, "restart": r})

        lam, Q, ll, info = best
        info["restarts"] = restarts

        self.lam, self.Q, self.ll, self.info = lam, Q, ll, info
        self.n_obs = len(x) - self.order

    ## LL de uma sequência: soma sobre as transições t = order .. N-1, ou seja,
    # N - order termos (os primeiros `order` símbolos só servem de contexto)
    def calculate_likelihood(self, symbols_sequence):
        self._check_fitted("calculate_likelihood")

        x = to_indices(symbols_sequence)
        n = len(x)

        if n <= self.order:
            raise ValueError(
                f"the sequence must have more than {self.order} symbols (the order)"
            )

        proba = np.zeros(n - self.order)
        for g in range(self.order):
            context = x[self.order - 1 - g : n - 1 - g]
            proba += self.lam[g] * self.Q[context, x[self.order :]]

        return float(np.log(proba).sum())

    ## A linha i é a distribuição do símbolo que vem depois de symbols[i],
    # usando os lags symbols[i], symbols[i-1], ... (mesma convenção do FirstOrder).
    # Nas primeiras order-1 linhas faltam lags: usa só os lags disponíveis, com os
    # pesos renormalizados. Se esses pesos forem todos zero, devolve a uniforme
    def predict_proba(self, symbols):
        self._check_fitted("predict_proba")

        arr = np.asarray(symbols)
        if arr.size == 0:
            raise ValueError("the parameter 'symbols' must have at least one element")

        x = to_indices(arr)
        n = len(x)

        proba = np.zeros((n, M))
        weight = np.zeros(n)
        for g in range(1, min(self.order, n) + 1):
            proba[g - 1 :] += self.lam[g - 1] * self.Q[x[: n - g + 1]]
            weight[g - 1 :] += self.lam[g - 1]

        empty = weight == 0
        weight[empty] = 1.0
        proba /= weight[:, None]
        proba[empty] = 1.0 / M

        return proba
