# taxa de sucesso dos reinícios quando os dados vêm de um HMM verdadeiro de 3 e 4 estados
import os, sys, json, pickle
os.environ["NUMBA_NUM_THREADS"] = "1"
from multiprocessing import Pool
from common import *
from common import _forward
def job(a):
    key, A, B, T, r = a
    A = np.array(A); B = np.array(B); n = A.shape[0]
    t = HMMDiscreto(n, 3); t.pi_, t.A_, t.B_ = stationary(A), A, B
    _, y = t.sample(T, semente=5)
    m = HMMDiscreto(n, 3, max_iter=4000, n_blocos=1)
    pi0, A0, B0 = m._inicializar(np.random.default_rng(np.random.SeedSequence(77).spawn(r + 1)[r]))
    pi, A_, B_, ll, info = m._baum_welch(np.ascontiguousarray(y), pi0, A0, B0)
    return key, r, dict(ll=ll, ll_true=t.score(y), it=info["iteracoes"], conv=info["convergiu"], B=B_, A=A_)
if __name__ == "__main__":
    res = pickle.load(open(OUT / "pool_states.pkl", "rb"))
    jobs = []
    for n in [3, 4]:
        best = max(res[f"b3_n{n}"].values(), key=lambda v: v["ll"])
        for r in range(20): jobs.append((f"n{n}", best["A"].tolist(), best["B"].tolist(), 300_000, r))
    with Pool(int(sys.argv[1])) as p: out = p.map(job, jobs)
    R = {}
    for k, r, v in out: R.setdefault(k, {})[r] = v
    pickle.dump(R, open(OUT / "synth_multi.pkl", "wb"))
    for k, v in R.items():
        lls = np.array([x["ll"] for x in v.values()]); print(k, "true", list(v.values())[0]["ll_true"], np.sort(lls)[::-1].round(1))
