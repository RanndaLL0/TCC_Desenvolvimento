import os, sys, pickle
os.environ["NUMBA_NUM_THREADS"] = "4"
from common import *
from common import _forward
from baselines import *

def auc(score, lab):
    o = np.argsort(score, kind="mergesort"); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    # empates: média dos postos
    s_sorted = score[o]; _, inv, cnt = np.unique(s_sorted, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = cs - (cnt - 1) / 2; r[o] = avg[inv]
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]

def best_model(res, key):
    r = max(res[key].values(), key=lambda v: v["ll"]); return r["pi"], r["A"], r["B"]

d = get(); tr, te = d["train"], d["test"]; nt = int(d["ntrain"])
mins = d["times"]; how_all = ((mins // 1440 + 3) % 7) * 24 + (mins // 60) % 24
how_tr, how_te = how_all[:nt], how_all[nt:]
res = pickle.load(open(OUT / "pool_states.pkl", "rb"))
R = {}
P = {}
# modelos de referência do repositório (FirstOrder na versão do HEAD)
from first_order_head import FirstOrder  # gerado por: git show e1adb76:src/models/naive_models/first_order.py > scripts/first_order_head.py
from src.models.naive_models.uniform import Uniform
from src.models.naive_models.persistence import Persistence
from src.utils.symbols import to_symbols
full_sym = to_symbols(np.concatenate([tr[-1:], te]))
fo = FirstOrder(); fo.fit(to_symbols(tr)); P["Markov 1"] = fo.predict_proba(full_sym)[:-1]
un = Uniform(); un.fit(to_symbols(tr[:10])); P["Uniforme"] = un.predict_prob(full_sym)[:-1]
pe = Persistence(); pe.fit(to_symbols(tr[:10])); P["Persistência"] = pe.predict_prob(full_sym)[:-1]
assert abs(ll_of(P["Markov 1"], te).sum() - fo.calculate_likelihood(full_sym)) < 1e-6
P["iid"] = markov_pred(markov_table(tr, 0), 0, tr, te)
# sazonais
cnt = np.zeros((168, 3)) + 1; np.add.at(cnt, (how_tr, tr), 1); tab_how = cnt / cnt.sum(1, keepdims=True)
P["Hora-da-semana"] = tab_how[how_te]
cnt = np.zeros((168, 3, 3)) + 1; np.add.at(cnt, (how_tr[1:], tr[:-1], tr[1:]), 1); tab_hm = cnt / cnt.sum(2, keepdims=True)
prev = np.concatenate([[tr[-1]], te[:-1]]); P["Hora-da-semana + Markov 1"] = tab_hm[how_te, prev]
NS = [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6, 7]
for n in NS:
    key = f"b3_n{n}"
    if key not in res: continue
    pi, A, B = best_model(res, key)
    P[f"HMM {n}"], _ = filt_pred(te, stationary(A), A, B)

iid = P["iid"][0]
for n in [3, 4]:
    poe = P[f"HMM {n}"] * P["Hora-da-semana"] / iid; P[f"HMM {n} × sazonal (PoE)"] = poe / poe.sum(1, keepdims=True)
mv = te != 1
tab = {}
for name, p in P.items():
    l = ll_of(p, te)
    dirm = mv & (p[:, 0] != p[:, 2])
    tab[name] = dict(ll=l.mean(), auc_mov=auc(1 - p[:, 1], mv), acc=(p.argmax(1) == te).mean(),
                     dir=((p[:, 2] > p[:, 0]) == (te == 2))[mv].mean(), brier=((p - np.eye(3)[te]) ** 2).sum(1).mean())
    # AUC estratificada por hora da semana
    a = []; w = []
    for h in range(168):
        s = how_te == h
        if mv[s].all() or (~mv[s]).all(): continue
        a.append(auc(1 - p[s, 1], mv[s])); w.append(s.sum())
    tab[name]["auc_estrat"] = float(np.average(a, weights=w))
    print(f"{name:32s} ll={tab[name]['ll']:.5f} auc={tab[name]['auc_mov']:.4f} aucE={tab[name]['auc_estrat']:.4f} acc={tab[name]['acc']:.4f} dir={tab[name]['dir']:.4f}")
R["preditivo"] = tab
# fração da AUC "explicada" pela sazonalidade
# bootstrap por dia
day = (d["times"][nt:] // 1440); _, dix = np.unique(day, return_inverse=True); nd = dix.max() + 1
contrib = {name: np.bincount(dix, weights=ll_of(p, te), minlength=nd) for name, p in P.items()}
cntd = np.bincount(dix, minlength=nd)
rng = np.random.default_rng(0)
pairs = [("HMM 3", "Markov 1"), ("HMM 3", "Hora-da-semana + Markov 1"), ("HMM 4", "HMM 3"), ("HMM 3", "HMM 2"),
         ("HMM 3 × sazonal (PoE)", "HMM 3"), ("HMM 5", "HMM 4"), ("HMM 6", "HMM 4"), ("HMM 7", "HMM 4"), ("HMM 2", "Markov 1")]
boot = {}
idx = rng.integers(0, nd, (4000, nd))
for a_, b_ in pairs:
    if a_ not in contrib or b_ not in contrib: continue
    dd = contrib[a_] - contrib[b_]
    s = dd[idx].sum(1) / cntd[idx].sum(1)
    boot[f"{a_} − {b_}"] = dict(media=dd.sum() / cntd.sum(), ic95=np.percentile(s, [2.5, 97.5]), frac_dias_melhor=(dd > 0).mean())
    print(a_, "-", b_, boot[f"{a_} − {b_}"])
R["bootstrap_dias"] = boot; R["n_dias_teste"] = int(nd)
# mês a mês no teste
mon = (d["times"][nt:].astype("datetime64[m]").astype("datetime64[M]")); um = np.unique(mon)
R["por_mes"] = {str(m_): {name: ll_of(P[name][mon == m_], te[mon == m_]).mean() for name in ["iid", "Markov 1", "Hora-da-semana", "HMM 3", "HMM 4", "HMM 5"]} for m_ in um}

# previsão de volatilidade: |r| médio da próxima 1 h
close = d["close"][nt - 1:]; r = np.abs(np.diff(np.log(close)))  # r[t] = |retorno| do símbolo te[t]
H = 12
fut = np.convolve(r, np.ones(H), "valid")[1:] / H  # média de |r| em t+1..t+H
Tn = fut.size
pi, A, B = best_model(res, "b3_n3")
alpha, c, _ = _forward(np.ascontiguousarray(te), stationary(A), A, B)
acc_ = np.zeros(len(te)); st = alpha.copy()
for h in range(H):
    st = st @ A; acc_ += 1 - (st @ B)[:, 1]
hmm_vol = acc_[:Tn]
def ewma(x, hl):
    a = 1 - 0.5 ** (1 / hl); o = np.empty_like(x, dtype=float); s = x[:200].mean()
    for i, v in enumerate(x): s = a * v + (1 - a) * s; o[i] = s
    return o
mv_f = mv.astype(float)
preds = {"HMM 3 (P(mov.) nas próximas 12)": hmm_vol,
         "EWMA indicador de movimento (meia-vida 1 h)": ewma(mv_f, 12)[:Tn],
         "EWMA indicador de movimento (meia-vida 6 h)": ewma(mv_f, 72)[:Tn],
         "EWMA |r| (meia-vida 1 h, usa preço)": ewma(r, 12)[:Tn],
         "EWMA |r| (meia-vida 6 h, usa preço)": ewma(r, 72)[:Tn],
         "Hora-da-semana (P(mov.))": 1 - P["Hora-da-semana"][1:Tn + 1, 1]}
R["volatilidade_1h"] = {k: dict(spearman=spearman(v, fut), pearson=np.corrcoef(v, fut)[0, 1]) for k, v in preds.items()}
for k, v in R["volatilidade_1h"].items(): print(k, v)

# ---------- classificação: matriz de confusão, precisão/recall/F1 por classe ----------
rng2 = np.random.default_rng(123)
Pc = dict(P)
prior = np.bincount(tr, minlength=3) / tr.size
def prf(pred, yt):
    cm = np.zeros((3, 3), int); np.add.at(cm, (yt, pred), 1)
    out = {"cm": cm, "acc": np.trace(cm) / cm.sum()}
    for c, lab in zip(range(3), ["-1", "0", "+1"]):
        tp = cm[c, c]; pp = cm[:, c].sum(); rp = cm[c, :].sum()
        p_ = tp / pp if pp else 0.0; r_ = tp / rp if rp else 0.0
        out[lab] = dict(precisao=p_, recall=r_, f1=2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0, previstos=int(pp))
    out["f1_macro_direcional"] = (out["-1"]["f1"] + out["+1"]["f1"]) / 2
    return out
cls = {}
cls["Aleatório (sorteio uniforme)"] = {"argmax": prf(rng2.integers(0, 3, te.size), te)}
for name, p in Pc.items():
    cls[name] = {"argmax": prf(p.argmax(1), te), "argmax P/prior": prf((p / prior).argmax(1), te)}
for name, v in cls.items():
    for rule, o in v.items():
        print(f"{name:32s} {rule:14s} acc={o['acc']:.4f} F1(-1)={o['-1']['f1']:.4f} F1(+1)={o['+1']['f1']:.4f} prec(-1)={o['-1']['precisao']:.4f} prec(+1)={o['+1']['precisao']:.4f} prev={[o[l]['previstos'] for l in ['-1','0','+1']]}")
R["classificacao"] = cls
R["prior_treino"] = prior

save("sazonal_preditivo", R)
