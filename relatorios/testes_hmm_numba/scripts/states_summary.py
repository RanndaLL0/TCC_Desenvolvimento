import os, sys, pickle
os.environ["NUMBA_NUM_THREADS"] = "1"
from common import *
from common import _forward
res = pickle.load(open(OUT / "pool_states.pkl", "rb"))
d = get(); tr, te = d["train"], d["test"]; nt = tr.size
TOL = 20.0  # nats: reinícios a menos disso do melhor contam como "mesmo ótimo"
S = {}
for n in [2, 3, 4, 5]:
    rr = res[f"b3_n{n}"]; lls = np.array([v["ll"] for v in rr.values()])
    b = max(rr.values(), key=lambda v: v["ll"])
    p = n_free_parameters(n, 3)
    o = np.argsort(-b["B"][:, 1]); A = b["A"][np.ix_(o, o)]; B = b["B"][o]
    st = stationary(A)
    # famílias de ótimos (agrupando por 20 nats)
    srt = np.sort(lls)[::-1]; fam = []
    for v in srt:
        if fam and fam[-1][0] - v < TOL: fam[-1][1] += 1
        else: fam.append([v, 1])
    m = HMMDiscreto(n, 3); m.pi_, m.A_, m.B_ = st, A, B
    gam = m.predict_proba(np.concatenate([tr, te]))
    S[n] = dict(p=p, reinicios=len(rr), ll_tr=b["ll"], ll_tr_s=b["ll"] / nt, ll_te=b["llte"], ll_te_s=b["llte"] / te.size,
                aic=2 * p - 2 * b["ll"], bic=p * np.log(nt) - 2 * b["ll"],
                sucesso=int((lls > b["ll"] - TOL).sum()), conv=int(sum(v["conv"] for v in rr.values())),
                it_med=float(np.median([v["it"] for v in rr.values()])), seg_med=float(np.median([v["sec"] for v in rr.values()])),
                seg_tot=float(sum(v["sec"] for v in rr.values())),
                familias=fam, lls=lls, llte_all=np.array([v["llte"] for v in rr.values()]),
                A=A, B=B, stat=st, dwell_h=1 / (1 - np.diag(A)) * 5 / 60, ocup_post=gam.mean(0))
    print(n, p, round(b["ll"], 1), round(b["llte"] / te.size, 5), round(S[n]["bic"], 1), f"{S[n]['sucesso']}/{len(rr)}", S[n]["conv"], len(fam), np.round(S[n]["dwell_h"], 2))
    print("   P(0) por estado:", np.round(B[:, 1], 3), " P(+1)-P(-1):", np.round(B[:, 2] - B[:, 0], 3), " ocupação:", np.round(st, 3))
    # perfil por hora (UTC) e fim de semana
    mins = d["times"]; hour = (mins // 60) % 24; dow = (mins // 1440 + 3) % 7
    S[n]["por_hora"] = np.array([gam[hour == h].mean(0) for h in range(24)])
    S[n]["por_dia"] = np.array([gam[dow == k].mean(0) for k in range(7)])
pickle.dump(S, open(OUT / "states_summary.pkl", "wb"))
