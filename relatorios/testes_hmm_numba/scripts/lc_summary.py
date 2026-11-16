# Resume a curva de aprendizado (seção 13): melhor reinício de cada N e modelos de referência
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import pickle
from common import *
from baselines import *
res = pickle.load(open(OUT / "pool_lc.pkl", "rb")); S = pickle.load(open(OUT / "states_summary.pkl", "rb"))
d = get(); tr, te = d["train"], d["test"]
L = {}
for N in [2000, 10000, 50000, 100000, 300000, 905654]:
    row = {}; t = tr[-N:]
    row["iid"] = ll_of(markov_pred(markov_table(t, 0), 0, t, te), te).mean()
    # contagem pura (como o FirstOrder) quando há dados; +1 nas contagens com poucos dados
    row["mk1"] = ll_of(markov_pred(markov_table(t, 1, alpha=0 if N > 10000 else 1), 1, t, te), te).mean()
    for n in [2, 3, 4]:
        if N == 905654:
            row[f"hmm{n}"] = S[n]["ll_te_s"]; continue
        rr = res[f"lc{N}_n{n}"]; b = max(rr.values(), key=lambda v: v["ll"])
        row[f"hmm{n}"] = b["llte"] / te.size
        row[f"hmm{n}_dwell"] = np.sort(1 / (1 - np.diag(b["A"])) * 5 / 60).round(1).tolist()
    L[N] = row; print(N, row)
save("curva_aprendizado", L)
