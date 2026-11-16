import numpy as np
def markov_table(y, k, alpha=1.0):
    y = np.asarray(y, dtype=np.int64)
    if k == 0:
        c = np.bincount(y, minlength=3) + alpha; return c / c.sum()
    code = np.zeros(y.size - k, dtype=np.int64)
    for j in range(k): code = code * 3 + y[j:y.size - k + j]
    cnt = np.bincount(code * 3 + y[k:], minlength=3 ** (k + 1)).reshape(-1, 3) + alpha
    return cnt / cnt.sum(1, keepdims=True)

def markov_pred(tab, k, ctx, y):
    """distribuições previstas para cada y_t do teste, contexto inicial = fim do treino."""
    if k == 0: return np.tile(tab, (len(y), 1))
    full = np.concatenate([np.asarray(ctx[-k:], dtype=np.int64), np.asarray(y, dtype=np.int64)])
    code = np.zeros(len(y), dtype=np.int64)
    for j in range(k): code = code * 3 + full[j:j + len(y)]
    return tab[code]

def ll_of(P, y): return np.log(P[np.arange(len(y)), y])
def nparams_markov(k): return 2 * 3 ** k
