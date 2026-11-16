import os, sys, pickle
os.environ["NUMBA_NUM_THREADS"] = "1"
from multiprocessing import Pool
from common import *
from common import _forward
import walkforward as W
def job(a):
    i, r = a
    tr = W.y_all[W.window(i - 12, i)]
    m = HMMDiscreto(3, 3, max_iter=3000, n_blocos=1)
    pi0, A0, B0 = m._inicializar(np.random.default_rng(np.random.SeedSequence(5000 + i).spawn(r + 1)[r]))
    pi, A, B, ll, info = m._baum_welch(np.ascontiguousarray(tr), pi0, A0, B0)
    te = W.y_all[W.window(i, i + 3)]
    _, _, llte = _forward(np.ascontiguousarray(te), stationary(A), A, B)
    return i, r, dict(ll=ll, llte=llte / te.size, dwell_h=np.sort(1 / (1 - np.diag(A)) * 5 / 60), ntr=tr.size)
if __name__ == "__main__":
    ms = [str(m) for m in W.months]
    idx = [ms.index(x) for x in ["2018-09", "2019-12", "2020-03"]]
    with Pool(int(sys.argv[1])) as p: out = p.map(job, [(i, r) for i in idx for r in range(24)])
    R = {}
    for i, r, v in out: R.setdefault(ms[i], []).append(v)
    pickle.dump(R, open(OUT / "wf_recheck.pkl", "wb"))
    for k, v in R.items():
        v = sorted(v, key=lambda x: -x["ll"])
        print(k, [(round(x["ll"]), round(x["llte"], 4), np.round(x["dwell_h"], 1).tolist()) for x in v[:4]], "| pior:", round(v[-1]["ll"]))
