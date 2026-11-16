import os, sys, itertools, time, warnings
os.environ["NUMBA_NUM_THREADS"] = "4"
from common import *
from common import _forward, _backward
from src.models.hmm.hmm_numba import _viterbi, _acumular_paralelo, _m_step
from hmmlearn.hmm import CategoricalHMM
HMMDiscreto.aquecer_jit()
R = {}
rng = np.random.default_rng(42)

# (a) força bruta
errs = []
for trial in range(20):
    n, m, T = rng.integers(2, 4), 3, int(rng.integers(3, 9))
    pi = rng.dirichlet(np.ones(n)); A = rng.dirichlet(np.ones(n), size=n); B = rng.dirichlet(np.ones(m), size=n)
    y = rng.integers(0, m, T)
    tot = 0.0; post = np.zeros((T, n)); best = (-1, None)
    for path in itertools.product(range(n), repeat=T):
        p = pi[path[0]] * B[path[0], y[0]]
        for t in range(1, T): p *= A[path[t-1], path[t]] * B[path[t], y[t]]
        tot += p
        for t in range(T): post[t, path[t]] += p
        if p > best[0]: best = (p, path)
    post /= tot
    mod = HMMDiscreto(n, m); mod.pi_, mod.A_, mod.B_ = pi, A, B
    errs.append(dict(ll=abs(mod.score(y) - np.log(tot)), gamma=np.abs(mod.predict_proba(y) - post).max(),
                     vit=bool((mod.predict(y) == np.array(best[1])).all()), T=T, n=int(n)))
R["forca_bruta"] = dict(casos=len(errs), max_err_ll=max(e["ll"] for e in errs), max_err_gamma=max(e["gamma"] for e in errs),
                       viterbi_ok=sum(e["vit"] for e in errs))
print(R["forca_bruta"])

# (b) hmmlearn nos dados reais, com os parâmetros do modelo de 3 estados salvo
import json
J3 = json.load(open(ROOT / "relatorios/hmm_2026-10-01_blocks_k3.json"))["results"][0]
pi, A, B = (np.array(J3[k]) for k in ("pi", "A", "B"))
pi_s = stationary(A)
y = get()["train"]
mod = HMMDiscreto(3, 3); mod.pi_, mod.A_, mod.B_ = pi_s, A, B
hl = CategoricalHMM(3, init_params="", params="ste", n_iter=1, tol=-np.inf, implementation="scaling")
hl.startprob_, hl.transmat_, hl.emissionprob_, hl.n_features = pi_s, A, B, 3
X = y.reshape(-1, 1)
t = time.perf_counter(); ll_hl = hl.score(X); t_hl_score = time.perf_counter() - t
t = time.perf_counter(); ll_nb = mod.score(y); t_nb_score = time.perf_counter() - t
g_hl = hl.predict_proba(X); g_nb = mod.predict_proba(y)
v_hl = hl.predict(X); v_nb = mod.predict(y)
# um passo de EM
alpha, c, _ = _forward(y, pi_s, A, B); beta = _backward(y, A, B, c)
xi, Bn = _acumular_paralelo(y, A, B, alpha, beta, 4)
p1, A1, B1 = _m_step(alpha, beta, xi, Bn, 0.0)
hl.fit(X)
R["hmmlearn"] = dict(T=int(y.size), ll_numba=ll_nb, ll_hmmlearn=ll_hl, dif_ll=abs(ll_nb - ll_hl),
                     max_dif_gamma=float(np.abs(g_hl - g_nb).max()), viterbi_iguais=float((v_hl == v_nb).mean()),
                     max_dif_A_apos_1_EM=float(np.abs(hl.transmat_ - A1).max()),
                     max_dif_B_apos_1_EM=float(np.abs(hl.emissionprob_ - B1).max()),
                     t_score_hmmlearn=t_hl_score, t_score_numba=t_nb_score)
print(R["hmmlearn"])

# (c) determinismo / n_blocos
lls = {}
for nb in [1, 2, 7, 12, 64]:
    mm = HMMDiscreto(3, 3, n_reinicios=1, max_iter=200, tolerancia=0, n_blocos=nb, semente=3).fit(y)
    lls[nb] = mm.ll_
mm2 = HMMDiscreto(3, 3, n_reinicios=1, max_iter=200, tolerancia=0, n_blocos=12, semente=3).fit(y)
R["determinismo"] = dict(ll_por_n_blocos=lls, amplitude=max(lls.values()) - min(lls.values()), repeticao_identica=bool(mm2.ll_ == lls[12]))
print(R["determinismo"])

# (d) recuperação sintética: verdadeiro = modelo de 3 estados salvo
true = HMMDiscreto(3, 3); true.pi_, true.A_, true.B_ = pi_s, A, B
order_t = np.argsort(-B[:, 1]); At, Bt = A[np.ix_(order_t, order_t)], B[order_t]
_, yy_hold = true.sample(200_000, semente=999)
ll_true_hold = true.score(yy_hold) / yy_hold.size
rec = []
for T in [1_000, 5_000, 20_000, 100_000, 300_000, 905_654]:
    for rep in range(3):
        st, yy = true.sample(T, semente=100 + rep)
        f = HMMDiscreto(3, 3, n_reinicios=6, max_iter=3000, semente=rep).fit(yy)
        p_, A_, B_, o = canon(f)
        dwell_t = 1 / (1 - np.diag(At)); dwell_f = 1 / (1 - np.clip(np.diag(A_), 0, 1 - 1e-12))
        # acerto de estado por posteriores
        g = f.predict_proba(yy)[:, o].argmax(1)
        st_c = np.argsort(order_t)[st]
        rec.append(dict(T=T, rep=rep, errB=float(np.abs(B_ - Bt).max()), errA=float(np.abs(A_ - At).max()),
                        dwell_true=dwell_t.tolist(), dwell_fit=dwell_f.tolist(),
                        acerto_estado=float((g == st_c).mean()),
                        ll_true_train=true.score(yy) / T, ll_fit_train=f.ll_ / T,
                        ll_hold_fit=f.score(yy_hold) / yy_hold.size, ll_hold_true=ll_true_hold,
                        sucesso_reinicios=sum(abs(r["ll"] - f.ll_) < 1e-3 * abs(f.ll_) * 1e-3 for r in f.info_["reinicios"])))
        print(rec[-1]["T"], rep, round(rec[-1]["errB"], 4), round(rec[-1]["errA"], 5), np.round(dwell_f, 1), rec[-1]["acerto_estado"], flush=True)
R["recuperacao"] = rec
# acerto de estado com os parâmetros verdadeiros (teto)
st, yy = true.sample(300_000, semente=7)
gt = true.predict_proba(yy).argmax(1)
R["acerto_estado_parametros_verdadeiros"] = float((gt == st).mean())
R["acerto_estado_viterbi_verdadeiros"] = float((true.predict(yy) == st).mean())
save("validacao", R)
print("ok")
