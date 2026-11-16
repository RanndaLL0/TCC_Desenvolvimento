import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]  # raiz do repositório
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parents[1] / "dados"; OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)  # sequências discretizadas (regeneráveis, não versionar)
import numpy as np
from src.preprocess.discretize import discretize, load_snapshot
from src.training.hmm_training import split_sequences, n_free_parameters
from src.models.hmm.hmm_numba import HMMDiscreto, _forward, _backward
SNAP = ROOT / "snapshot/btc_usdt_2026-10-01.parquet"

def get(mode="blocks", k=3, interval=1, train_pct=95, full=False):
    f = CACHE / f"seq_{mode}_k{k}_i{interval}_t{train_pct}.npz"
    if not f.exists():
        ot, close, src = load_snapshot(SNAP)
        r = discretize(ot, close, k, mode=mode, interval_months=interval, train_pct=train_pct)
        tr, te = split_sequences(r)
        v = r.valid
        np.savez(f, train=tr, test=te, times=ot[v].astype("datetime64[m]").astype(np.int64),
                 close=np.asarray(close)[v], cut=r.cut, ntrain=tr.size,
                 seg=r.segment[v])
    d = np.load(f)
    return d

def canon(m):
    """ordena estados por P(0) decrescente (calmo primeiro)."""
    o = np.argsort(-m.B_[:, 1])
    return m.pi_[o], m.A_[np.ix_(o, o)], m.B_[o], o

def stationary(A):
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(abs(w - 1))]); return p / p.sum()

def filt_pred(y, pi, A, B):
    """P(y_t | y_<t) para cada t (um passo à frente), começando em pi."""
    alpha, c, ll = _forward(np.ascontiguousarray(y, dtype=np.int64), pi, A, B)
    pred_state = np.vstack([pi[None, :], alpha[:-1] @ A])
    return pred_state @ B, ll

def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
