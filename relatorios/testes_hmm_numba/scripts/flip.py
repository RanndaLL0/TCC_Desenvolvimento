# decompõe o F1 direcional: "quando há movimento" × "para onde"
import os, sys, pickle
os.environ["NUMBA_NUM_THREADS"] = "2"
from common import *
from baselines import *
d = get(); tr, te = d["train"], d["test"]
res = pickle.load(open(OUT / "pool_states.pkl", "rb"))
prior = np.bincount(tr, minlength=3) / tr.size
def f1s(pred, y):
    out = {}
    for c, lab in [(0, "-1"), (2, "+1")]:
        tp = ((pred == c) & (y == c)).sum(); pp = (pred == c).sum(); rp = (y == c).sum()
        p = tp / pp if pp else 0; r = tp / rp
        out[lab] = dict(prec=p, rec=r, f1=2 * p * r / (p + r) if p + r else 0)
    out["macro"] = (out["-1"]["f1"] + out["+1"]["f1"]) / 2
    return out
rng = np.random.default_rng(1)
R = {}
for n in [3, 4, 5]:
    b = max(res[f"b3_n{n}"].values(), key=lambda v: v["ll"])
    P, _ = filt_pred(te, stationary(b["A"]), b["A"], b["B"])
    pred = (P / prior).argmax(1)
    flip = np.where(pred == 0, 2, np.where(pred == 2, 0, 1))
    mov = pred != 1
    rnd = np.where(mov, rng.choice([0, 2], te.size), 1)
    # oráculo de direção: acerta o sinal sempre que prevê movimento
    orac = np.where(mov & (te != 1), te, np.where(mov, rng.choice([0, 2], te.size), 1))
    R[n] = {"original": f1s(pred, te), "sinal invertido": f1s(flip, te), "sinal sorteado": f1s(rnd, te),
            "sinal perfeito (oráculo)": f1s(orac, te), "frac_mov_prevista": mov.mean()}
    for k, v in R[n].items():
        if isinstance(v, dict): print(n, f"{k:26s} F1(-1)={v['-1']['f1']:.4f} F1(+1)={v['+1']['f1']:.4f} macro={v['macro']:.4f} prec(-1)={v['-1']['prec']:.4f} prec(+1)={v['+1']['prec']:.4f}")
# precisão direcional condicional: entre instantes em que o modelo prevê movimento e houve movimento
save("flip", R)
