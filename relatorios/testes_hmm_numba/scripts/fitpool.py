"""Executa muitos reinícios (job = config + índice de reinício) em processos paralelos.
Guarda, para cada reinício: ll final, iterações, convergência, parâmetros, ll de teste."""
import os, sys, json, time, pickle
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from multiprocessing import Pool
from common import *
from common import _forward, _backward

def run(job):
    key, cfg, r = job
    d = get(cfg["mode"], cfg["k"], cfg.get("interval", 1))
    y = d["train"]; te = d["test"]
    if cfg.get("last"):  # usa só os últimos N símbolos do treino
        y = y[-cfg["last"]:]
    m = HMMDiscreto(cfg["n"], 3, max_iter=cfg.get("max_iter", 3000), tolerancia=cfg.get("tol", 1e-8),
                    concentracao=cfg.get("conc", 5.0), n_blocos=1, semente=cfg.get("seed", 0))
    ss = np.random.SeedSequence(cfg.get("seed", 0)).spawn(r + 1)[r]
    pi0, A0, B0 = m._inicializar(np.random.default_rng(ss))
    t0 = time.perf_counter()
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pi, A, B, ll, info = m._baum_welch(np.ascontiguousarray(y, dtype=np.int64), pi0, A0, B0)
    el = time.perf_counter() - t0
    _, _, llte = _forward(np.ascontiguousarray(te, dtype=np.int64), stationary(A), A, B)
    return key, r, dict(ll=ll, llte=float(llte), it=info["iteracoes"], conv=info["convergiu"], sec=el,
                        warn=len(w), pi=pi, A=A, B=B, hist=info["historico"][::10], T=int(y.size), Tte=int(te.size))

if __name__ == "__main__":
    name = sys.argv[1]; workers = int(sys.argv[2])
    cfgs = json.loads(open(sys.argv[3]).read())  # {key: {cfg..., restarts: R}}
    out = OUT / f"pool_{name}.pkl"
    res = pickle.load(open(out, "rb")) if out.exists() else {}
    jobs = [(k, c, r) for k, c in cfgs.items() for r in range(c["restarts"]) if r not in res.get(k, {})]
    # maior custo primeiro
    jobs.sort(key=lambda j: (j[1].get("prio", 0), j[2]))
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with Pool(workers) as p:
        for i, (k, r, rr) in enumerate(p.imap_unordered(run, jobs)):
            res.setdefault(k, {})[r] = rr
            print(f"[{time.time()-t0:7.0f}s] {i+1}/{len(jobs)} {k} r{r} ll={rr['ll']:.1f} te={rr['llte']:.1f} it={rr['it']} conv={rr['conv']}", flush=True)
            if (i + 1) % 4 == 0:
                pickle.dump(res, open(out, "wb"))
    pickle.dump(res, open(out, "wb"))
    print("FIM", flush=True)
