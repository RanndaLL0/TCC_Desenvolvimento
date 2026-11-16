import os, sys, time
from common import *
from common import _forward, _backward
from src.models.hmm.hmm_numba import _acumular_paralelo, _m_step, _viterbi
from numba import get_num_threads, set_num_threads
HMMDiscreto.aquecer_jit()
y = get()["train"]; R = {"threads_max": get_num_threads()}
def tm(f, rep=5):
    f(); ts = []
    for _ in range(rep):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))
rng = np.random.default_rng(0)
def params(n):
    return rng.dirichlet(np.ones(n) * 5), rng.dirichlet(np.ones(n) * 5, size=n), rng.dirichlet(np.ones(3) * 5, size=n)
# decomposição por kernel
comp = {}
for n in [2, 3, 4, 6, 8]:
    pi, A, B = params(n)
    alpha, c, _ = _forward(y, pi, A, B); beta = _backward(y, A, B, c)
    row = dict(forward=tm(lambda: _forward(y, pi, A, B)), backward=tm(lambda: _backward(y, A, B, c)))
    for th in [1, 2, 4, 8, 12]:
        set_num_threads(th); row[f"acumular_{th}t"] = tm(lambda: _acumular_paralelo(y, A, B, alpha, beta, th))
    set_num_threads(R["threads_max"])
    xi, Bn = _acumular_paralelo(y, A, B, alpha, beta, 12)
    row["m_step"] = tm(lambda: _m_step(alpha, beta, xi, Bn, 1e-10))
    lp, lA, lB = np.log(pi), np.log(A), np.log(B)
    row["viterbi"] = tm(lambda: _viterbi(y, lp, lA, lB), 3)
    row["iter_1t"] = row["forward"] + row["backward"] + row["acumular_1t"] + row["m_step"]
    row["iter_12t"] = row["forward"] + row["backward"] + row["acumular_12t"] + row["m_step"]
    comp[n] = row; print(n, {k: round(v * 1000, 1) for k, v in row.items()}, flush=True)
R["kernels_s"] = comp
# escala com T
sc = {}
pi, A, B = params(3)
for T in [10_000, 100_000, 300_000, 905_654]:
    yy = y[:T]
    m = HMMDiscreto(3, 3, n_reinicios=1, max_iter=20, tolerancia=0)
    sc[T] = tm(lambda: m.fit(yy), 3) / 20
    print(T, sc[T], flush=True)
R["por_iter_vs_T"] = sc
# hmmlearn
from hmmlearn.hmm import CategoricalHMM
hl = {}
for n in [2, 3, 4]:
    pi, A, B = params(n)
    h = CategoricalHMM(n, init_params="", n_iter=10, tol=-np.inf, implementation="scaling")
    h.startprob_, h.transmat_, h.emissionprob_, h.n_features = pi, A, B, 3
    t = time.perf_counter(); h.fit(y.reshape(-1, 1)); thl = (time.perf_counter() - t) / 10
    h2 = CategoricalHMM(n, init_params="", n_iter=10, tol=-np.inf, implementation="log")
    h2.startprob_, h2.transmat_, h2.emissionprob_, h2.n_features = pi, A, B, 3
    t = time.perf_counter(); h2.fit(y.reshape(-1, 1)); thl2 = (time.perf_counter() - t) / 10
    m = HMMDiscreto(n, 3, n_reinicios=1, max_iter=10, tolerancia=0)
    t = time.perf_counter(); m.fit(y); tnb = (time.perf_counter() - t) / 10
    hl[n] = dict(hmmlearn_scaling=thl, hmmlearn_log=thl2, numba=tnb); print(n, hl[n], flush=True)
R["hmmlearn_vs_numba_s_por_iter"] = hl
# tempo de compilação JIT (cache em disco desligado: mede via nova assinatura float32? usa cache)
save("bench", R)
