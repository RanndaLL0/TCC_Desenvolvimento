import os, sys, pickle
os.environ["NUMBA_NUM_THREADS"] = "2"
from common import *
from baselines import *
def auc(score, lab):
    o = np.argsort(score, kind="mergesort"); s_sorted = score[o]
    _, inv, cnt = np.unique(s_sorted, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt); avg = cs - (cnt - 1) / 2; r = np.empty(len(score)); r[o] = avg[inv]
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
disc = pickle.load(open(OUT / "pool_disc.pkl", "rb")); st = pickle.load(open(OUT / "pool_states.pkl", "rb"))
VARS = [("b3", "blocks", 3, 1), ("b4", "blocks", 4, 1), ("b5", "blocks", 5, 1), ("g3", "global", 3, 1), ("g4", "global", 4, 1), ("g5", "global", 5, 1), ("b3i3", "blocks", 3, 3)]
out = {}
for tag, mode, k, iv in VARS:
    d = get(mode, k, iv); tr, te = d["train"], d["test"]
    row = dict(mode=mode, k=k, interval=iv, n_tr=int(tr.size), n_te=int(te.size),
               freq_tr=(np.bincount(tr, minlength=3) / tr.size), freq_te=(np.bincount(te, minlength=3) / te.size))
    row["iid"] = ll_of(markov_pred(markov_table(tr, 0), 0, tr, te), te).mean()
    P1 = markov_pred(markov_table(tr, 1, alpha=0), 1, tr, te); row["mk1"] = ll_of(P1, te).mean()
    mv = te != 1
    row["mk1_auc"] = auc(1 - P1[:, 1], mv); row["mk1_dir"] = ((P1[:, 2] > P1[:, 0]) == (te == 2))[mv].mean()
    for n in [2, 3, 4]:
        src = st[f"b3_n{n}"] if tag == "b3" else disc[f"{tag}_n{n}"]
        if tag == "b3": src = {r: v for r, v in src.items() if r < {2: 4, 3: 6, 4: 6}[n]}  # mesmo nº de reinícios
        lls = np.array([v["ll"] for v in src.values()]); b = max(src.values(), key=lambda v: v["ll"])
        o = np.argsort(-b["B"][:, 1]); A = b["A"][np.ix_(o, o)]; B = b["B"][o]
        P, _ = filt_pred(te, stationary(A), A, B)
        p = n_free_parameters(n, 3)
        row[f"hmm{n}"] = dict(ll_te=b["llte"] / te.size, ll_tr=b["ll"], bic=p * np.log(tr.size) - 2 * b["ll"],
                             sucesso=f"{int((lls > b['ll'] - 20).sum())}/{len(lls)}", B=B, A=A,
                             dwell_h=1 / (1 - np.diag(A)) * 5 / 60, assim=float(np.abs(B[:, 2] - B[:, 0]).max()),
                             auc=auc(1 - P[:, 1], mv), dir=((P[:, 2] > P[:, 0]) == (te == 2))[mv].mean(), stat=stationary(A))
    out[tag] = row
    h3 = row["hmm3"]
    print(f"{tag:5s} freq_te={np.round(row['freq_te'],3)} iid={row['iid']:.4f} mk1={row['mk1']:.4f} " +
          " ".join(f"h{n}={row[f'hmm{n}']['ll_te']:.4f}({row[f'hmm{n}']['sucesso']})" for n in [2, 3, 4]) +
          f" | ganho h3-mk1={h3['ll_te']-row['mk1']:.4f} h3-iid={h3['ll_te']-row['iid']:.4f} auc h3={h3['auc']:.3f} mk1={row['mk1_auc']:.3f} dir h3={h3['dir']:.3f} mk1={row['mk1_dir']:.3f}")
    print("      P(0) h3:", np.round(h3["B"][:, 1], 3), "assim h2..4:", [round(row[f"hmm{n}"]["assim"], 3) for n in [2, 3, 4]], "dwell h3:", np.round(h3["dwell_h"], 1),
          "BIC best n:", min([2, 3, 4], key=lambda n: row[f"hmm{n}"]["bic"]))
pickle.dump(out, open(OUT / "disc_summary.pkl", "wb"))
