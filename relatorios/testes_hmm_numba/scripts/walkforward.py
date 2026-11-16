import os, sys
os.environ["NUMBA_NUM_THREADS"] = "1"
from multiprocessing import Pool
from common import *
from common import _forward, _backward
from baselines import *

d = get(); y_all = np.concatenate([d["train"], d["test"]]); t_all = d["times"]
mon = (t_all.astype("datetime64[m]").astype("datetime64[M]"))
months = np.unique(mon)

def window(i0, i1):
    hi = months[i1] if i1 < len(months) else months[-1] + np.timedelta64(1, "M")
    return np.flatnonzero((mon >= months[i0]) & (mon < hi))

def job(i):
    # treino: 12 meses [i-12, i); teste: 3 meses [i, i+3)
    tr = y_all[window(i - 12, i)]; te = y_all[window(i, min(i + 3, len(months)))]
    best = None
    for r in range(8):
        m = HMMDiscreto(3, 3, n_reinicios=1, max_iter=3000, n_blocos=1, semente=1000 * i + r).fit(tr)
        if best is None or m.ll_ > best.ll_: best = m
    lls = [r_ll for r_ll in [best.ll_]]
    p, A, B, o = canon(best)
    _, _, llte = _forward(np.ascontiguousarray(te), stationary(A), A, B)
    out = dict(inicio_teste=str(months[i]), n_tr=int(tr.size), n_te=int(te.size), hmm_tr=best.ll_ / tr.size, hmm_te=llte / te.size,
               A=A, B=B, dwell_h=(1 / (1 - np.diag(A)) * 5 / 60), stat=stationary(A))
    for k in [0, 1]:
        tab = markov_table(tr, k); out[f"mk{k}_te"] = ll_of(markov_pred(tab, k, tr, te), te).mean()
    print(out["inicio_teste"], round(out["hmm_te"], 4), round(out["mk1_te"], 4), round(out["mk0_te"], 4), np.round(out["dwell_h"], 1), flush=True)
    return out

if __name__ == "__main__":
    HMMDiscreto.aquecer_jit()
    idx = [i for i in range(12, len(months), 3) if i + 3 <= len(months)]
    with Pool(int(sys.argv[1])) as p: res = p.map(job, idx)
    save("walkforward", res)
