import sys, json, pickle
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "dados"
FIG = Path(__file__).resolve().parents[1] / "figuras"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .3, "axes.formatter.use_locale": False})
C = ["#2a6f97", "#e07a5f", "#81b29a", "#9b5de5", "#f2cc8f", "#3d405b", "#c1121f"]
REG = ["#81b29a", "#f2cc8f", "#e07a5f", "#c1121f", "#6d597a", "#3d405b", "#2a6f97"]
def br(ax):  # vírgula decimal
    import matplotlib.ticker as mt
    for a in (ax.xaxis, ax.yaxis):
        if isinstance(a.get_major_formatter(), mt.ScalarFormatter):
            a.set_major_formatter(mt.FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", bbox_inches="tight"); plt.close(fig); print(name)

which = sys.argv[1:] or ["all"]
def want(n): return "all" in which or n in which

if want("recuperacao"):
    R = json.load(open(OUT / "validacao.json"))["recuperacao"]
    Ts = sorted({r["T"] for r in R})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for key, lab, c in [("errB", "erro máx. em B", C[0]), ("errA", "erro máx. em A", C[1])]:
        v = [[r[key] for r in R if r["T"] == T] for T in Ts]
        ax[0].plot(Ts, [np.median(x) for x in v], "o-", color=c, label=lab)
        for T, x in zip(Ts, v): ax[0].scatter([T] * len(x), x, color=c, s=8, alpha=.4)
    ax[0].plot(Ts, 0.6 / np.sqrt(Ts), "--", color="gray", lw=1, label=r"referência $\propto 1/\sqrt{T}$")
    ax[0].set(xscale="log", yscale="log", xlabel="tamanho da série sintética (T)", ylabel="erro absoluto máximo", title="Recuperação dos parâmetros")
    ax[0].legend(fontsize=8)
    acc = [[r["acerto_estado"] for r in R if r["T"] == T] for T in Ts]
    ax[1].plot(Ts, [np.median(x) for x in acc], "o-", color=C[2], label="modelo estimado")
    for T, x in zip(Ts, acc): ax[1].scatter([T] * len(x), x, color=C[2], s=8, alpha=.4)
    ax[1].axhline(json.load(open(OUT / "validacao.json"))["acerto_estado_parametros_verdadeiros"], ls="--", color="gray", label="parâmetros verdadeiros (teto)")
    ax[1].set(xscale="log", xlabel="tamanho da série sintética (T)", ylabel="fração de estados corretos", title="Decodificação dos estados ocultos", ylim=(0.25, 1))
    ax[1].legend(fontsize=8, loc="lower right"); br(ax[1])
    save(fig, "01_recuperacao_sintetica")

if want("walkforward"):
    R = json.load(open(OUT / "walkforward.json"))
    x = np.array([np.datetime64(r["inicio_teste"]) for r in R]).astype("datetime64[D]").astype("O")
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    g1 = [r["hmm_te"] - r["mk1_te"] for r in R]; g0 = [r["hmm_te"] - r["mk0_te"] for r in R]
    ax[0].bar(x, g0, width=60, color=C[2], alpha=.6, label="HMM 3 − frequências (iid)")
    ax[0].bar(x, g1, width=60, color=C[0], label="HMM 3 − Markov ordem 1")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set(ylabel="ganho (nats/símbolo)", title="Walk-forward: treino de 12 meses, teste nos 3 meses seguintes"); ax[0].legend(fontsize=8); br(ax[0])
    dw = np.array([sorted(r["dwell_h"]) for r in R])
    for j, lab in enumerate(["menor", "intermediária", "maior"]):
        ax[1].plot(x, dw[:, j], "o-", ms=3, color=REG[j], label=f"permanência {lab}")
    ax[1].set(yscale="log", ylabel="permanência esperada (h)", title="Permanência dos 3 estados em cada janela"); ax[1].legend(fontsize=8); 
    save(fig, "10_walkforward")

NAMES = {2: ["calmo", "agitado"], 3: ["calmo", "intermediário", "agitado"],
         4: ["muito calmo", "calmo", "intermediário", "agitado"],
         5: ["muito calmo", "calmo", "interm. (+)", "interm. (−)", "agitado"]}
if want("estados"):
    S = pickle.load(open(OUT / "states_summary.pkl", "rb")); ns = [2, 3, 4, 5]
    # 1) paisagem dos reinícios
    fig, ax = plt.subplots(figsize=(8, 3.8))
    rng = np.random.default_rng(0)
    for i, n in enumerate(ns):
        l = S[n]["lls"] / 1e3; best = l.max()
        ok = l > best - 0.02
        ax.scatter(i + rng.uniform(-.18, .18, l.size), l, s=14, c=np.where(ok, C[0], C[1]), alpha=.75)
        ax.text(i, best + 1.0, f"{ok.sum()}/{l.size}\nno melhor", ha="center", fontsize=8)
    ax.set_xticks(range(len(ns)), [f"{n} estados" for n in ns])
    ax.set(ylabel="log-verossimilhança final no treino (mil nats)", title="Resultado de cada reinício do Baum-Welch (azul: melhor ótimo encontrado)")
    ax.set_ylim(top=max(S[n]["lls"].max() for n in ns) / 1e3 + 4); br(ax)
    save(fig, "03_reinicios_otimos_locais")
    # 2) seleção do número de estados
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ns, [S[n]["bic"] / 1e6 for n in ns], "o-", color=C[0], label="BIC")
    ax[0].plot(ns, [S[n]["aic"] / 1e6 for n in ns], "s--", color=C[1], label="AIC")
    ax[0].set(xlabel="número de estados ocultos", ylabel="critério (milhões)", title="Critérios de informação (treino)", xticks=ns); ax[0].legend(); br(ax[0])
    ax[1].plot(ns, [S[n]["ll_te_s"] for n in ns], "o-", color=C[2], label="teste")
    ax[1].plot(ns, [S[n]["ll_tr_s"] for n in ns], "s--", color=C[5], label="treino")
    ax[1].set(xlabel="número de estados ocultos", ylabel="log-veross. por símbolo (nats)", title="Ajuste por símbolo", xticks=ns); ax[1].legend(); br(ax[1])
    save(fig, "04_selecao_estados")
    # 3) emissões
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[3, 4, 5]))
    for a, n in zip(ax, [3, 4, 5]):
        B = S[n]["B"]; x = np.arange(n); w = .27
        for j, (lab, c) in enumerate(zip(["−1 (queda)", "0 (estável)", "+1 (alta)"], [C[1], "#9a9a9a", C[0]])):
            a.bar(x + (j - 1) * w, B[:, j], w, color=c, label=lab)
        a.set_xticks(x, NAMES[n], rotation=20, fontsize=8); a.set(ylim=(0, 1), title=f"{n} estados"); br(a)
        for i in range(n): a.text(i, 0.93, f"{S[n]['stat'][i]*100:.0f}%", ha="center", fontsize=7, color="k")
    ax[0].set_ylabel("P(símbolo | estado)"); ax[0].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1, .88))
    fig.suptitle("Matriz de emissão B (rótulo no topo: fração estacionária do tempo no estado)", fontsize=10)
    save(fig, "05_emissoes")
    # 4) transições (heatmap log)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[3, 4, 5]))
    for a, n in zip(ax, [3, 4, 5]):
        A = S[n]["A"]; im = a.imshow(np.log10(np.clip(A, 1e-6, 1)), cmap="Blues", vmin=-6, vmax=0)
        for i in range(n):
            for j in range(n):
                a.text(j, i, f"{A[i,j]:.3f}".replace(".", ",") if A[i, j] >= 5e-4 else "≈0", ha="center", va="center", fontsize=7,
                       color="w" if A[i, j] > .3 else "k")
        a.set_xticks(range(n), NAMES[n], rotation=30, fontsize=7); a.set_yticks(range(n), NAMES[n], fontsize=7); a.grid(False)
        a.set_title(f"{n} estados")
    fig.suptitle("Matriz de transição A (linha = estado atual, coluna = próximo)", fontsize=10)
    save(fig, "06_transicoes")
    # 5) perfil por hora e dia
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[3, 1.3]))
    ph = S[3]["por_hora"]; bottom = np.zeros(24)
    for i in range(3):
        ax[0].bar(range(24), ph[:, i], bottom=bottom, color=REG[[0, 1, 2][i]], label=NAMES[3][i]); bottom += ph[:, i]
    ax[0].set(xlabel="hora (UTC)", ylabel="fração média da posterior", title="Regimes do HMM de 3 estados por hora do dia", xticks=range(0, 24, 2), ylim=(0, 1))
    ax[0].axvspan(12.6, 17.4, color="k", alpha=.06); ax[0].text(15, .03, "abertura EUA", ha="center", fontsize=7)
    ax[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0)); br(ax[0])
    pdw = S[3]["por_dia"]; bottom = np.zeros(7)
    for i in range(3):
        ax[1].bar(range(7), pdw[:, i], bottom=bottom, color=REG[i]); bottom += pdw[:, i]
    ax[1].set(xticks=range(7), ylim=(0, 1), title="por dia da semana"); ax[1].set_xticklabels(["seg", "ter", "qua", "qui", "sex", "sáb", "dom"], fontsize=8); br(ax[1])
    save(fig, "07_sazonalidade_regimes")

if want("preditivo"):
    R = json.load(open(OUT / "sazonal_preditivo.json")); T = R["preditivo"]
    mods = ["Uniforme", "Persistência", "iid", "Markov 1", "Hora-da-semana", "Hora-da-semana + Markov 1", "HMM 2", "HMM 3", "HMM 4", "HMM 5"]
    lab = {"iid": "Frequências (iid)", "Hora-da-semana + Markov 1": "Hora-sem. + Markov 1", "Hora-da-semana": "Hora da semana"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ms = [m for m in mods if m != "Persistência"]
    g = [T[m]["ll"] + np.log(3) for m in ms]
    cols = [C[0] if m.startswith("HMM") else (C[4] if "Hora" in m else "#9a9a9a") for m in ms]
    ax[0].barh(range(len(ms)), g, color=cols); ax[0].set_yticks(range(len(ms)), [lab.get(m, m) for m in ms], fontsize=8); ax[0].invert_yaxis()
    ax[0].set(xlabel="ganho sobre o uniforme (nats/símbolo, teste)", title="Log-verossimilhança fora da amostra"); br(ax[0])
    for i, v in enumerate(g): ax[0].text(v + .001, i, f"{v:.4f}".replace(".", ","), va="center", fontsize=7)
    ax[0].set_xlim(0, max(g) * 1.2)
    ms2 = ["Markov 1", "Persistência", "Hora-da-semana", "Hora-da-semana + Markov 1", "HMM 2", "HMM 3", "HMM 4", "HMM 5"]
    y = np.arange(len(ms2))
    ax[1].barh(y - .2, [T[m]["auc_mov"] for m in ms2], .4, color=C[0], label="AUC geral")
    ax[1].barh(y + .2, [T[m]["auc_estrat"] for m in ms2], .4, color=C[1], label="AUC dentro da mesma hora da semana")
    ax[1].set_yticks(y, [lab.get(m, m) for m in ms2], fontsize=8); ax[1].invert_yaxis(); ax[1].axvline(.5, color="k", lw=.8)
    ax[1].set(xlim=(.45, .72), xlabel="AUC (movimento × candle parado)", title="Prever SE haverá movimento"); ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2); br(ax[1])
    save(fig, "08_preditivo_ll_auc")
    # F1 direcional: decomposição
    F = json.load(open(OUT / "flip.json")); cls = R["classificacao"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    groups = ["original", "sinal invertido", "sinal sorteado"]; x = np.arange(3); w = .25
    for j, gname in enumerate(groups):
        ax.bar(x + (j - 1) * w, [F[str(n)][gname]["macro"] for n in [3, 4, 5]], w, color=[C[0], C[1], "#9a9a9a"][j], label=gname)
    for v, labl, ls, dy in [(cls["Aleatório (sorteio uniforme)"]["argmax"]["f1_macro_direcional"], "aleatório", ":", .0015),
                        (cls["Persistência"]["argmax"]["f1_macro_direcional"], "persistência", "--", -.0045),
                        (cls["Markov 1"]["argmax P/prior"]["f1_macro_direcional"], "Markov 1 (regra P/prior)", "-.", .0015)]:
        ax.axhline(v, color="k", lw=.9, ls=ls); ax.text(2.45, v + dy, labl, fontsize=7, ha="right")
    ax.set_xticks(x, [f"HMM {n} estados" for n in [3, 4, 5]]); ax.set_ylim(.25, .37)
    ax.set(ylabel="F1 médio das classes −1 e +1", title="F1 direcional: quanto vem do 'quando' e quanto do 'para onde'"); ax.legend(fontsize=8, loc="upper left"); br(ax)
    save(fig, "09_f1_direcional_decomposicao")

if want("curva"):
    L = json.load(open(OUT / "curva_aprendizado.json")); Ns = sorted(int(k) for k in L)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for key, labl, c, mk in [("iid", "Frequências (iid)", "#9a9a9a", "s"), ("mk1", "Markov ordem 1", C[5], "D"),
                             ("hmm2", "HMM 2 estados", C[2], "o"), ("hmm3", "HMM 3 estados", C[0], "o"), ("hmm4", "HMM 4 estados", C[1], "o")]:
        ax.plot(Ns, [L[str(N)][key] for N in Ns], marker=mk, color=c, label=labl)
    ax.set(xscale="log", xlabel="símbolos de treino (os mais recentes antes do teste)", ylabel="log-veross. por símbolo no teste", title="Curva de aprendizado")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / 288, lambda d: d * 288)); sec.set_xlabel("dias de dados", fontsize=8)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, .5)); br(ax)
    save(fig, "11_curva_aprendizado")

if want("disc"):
    D = pickle.load(open(OUT / "disc_summary.pkl", "rb"))
    tags = ["b3", "b4", "b5", "b3i3", "g3", "g4", "g5"]
    lab = {"b3": "blocos\ns/3 *", "b4": "blocos\ns/4", "b5": "blocos\ns/5", "b3i3": "blocos 3m\ns/3",
           "g3": "global\ns/3", "g4": "global\ns/4", "g5": "global\ns/5"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(tags)); w = .26
    for j, n in enumerate([2, 3, 4]):
        ax[0].bar(x + (j - 1) * w, [D[t][f"hmm{n}"]["ll_te"] - D[t]["mk1"] for t in tags], w, color=[C[2], C[0], C[1]][j], label=f"HMM {n} estados")
    ax[0].set_xticks(x, [lab[t] for t in tags], fontsize=7); ax[0].legend(fontsize=8)
    ax[0].set(ylabel="ganho sobre a Markov 1 (nats/símbolo)", title="Ganho no teste em cada discretização"); br(ax[0])
    ax[1].bar(x - .2, [D[t]["hmm3"]["auc"] for t in tags], .4, color=C[0], label="HMM 3: AUC de movimento")
    ax[1].bar(x + .2, [D[t]["mk1_auc"] for t in tags], .4, color=C[5], label="Markov 1: AUC de movimento")
    ax[1].plot(x, [D[t]["hmm3"]["dir"] for t in tags], "o", color=C[1], label="HMM 3: acerto de direção")
    ax[1].plot(x, [D[t]["mk1_dir"] for t in tags], "D", color="#9a9a9a", label="Markov 1: acerto de direção")
    ax[1].axhline(.5, color="k", lw=.8); ax[1].set_ylim(.45, .74)
    ax[1].set_xticks(x, [lab[t] for t in tags], fontsize=7); ax[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.18))
    ax[1].set(title="Movimento × direção em cada discretização"); br(ax[1])
    save(fig, "12_sensibilidade_discretizacao")

if want("bench"):
    Bq = json.load(open(OUT / "bench.json")); K = Bq["kernels_s"]; ns = sorted(int(n) for n in K)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    parts = [("forward", "forward", C[0]), ("backward", "backward", C[2]), ("acumular_1t", "passo E (1 thread)", C[1])]
    bottom = np.zeros(len(ns))
    for key, labl, c in parts:
        v = np.array([K[str(n)][key] * 1000 for n in ns]); ax[0].bar(range(len(ns)), v, bottom=bottom, color=c, label=labl); bottom += v
    ax[0].plot(range(len(ns)), [K[str(n)]["iter_12t"] * 1000 for n in ns], "k_", ms=22, mew=2, label="iteração com 12 threads")
    ax[0].set_xticks(range(len(ns)), [str(n) for n in ns]); ax[0].set(xlabel="número de estados", ylabel="ms por iteração (T = 905.654)", title="Tempo de uma iteração do Baum-Welch")
    ax[0].legend(fontsize=7); br(ax[0])
    th = [1, 2, 4, 8, 12]
    for n, c in zip([3, 6], [C[0], C[1]]):
        base = K[str(n)]["acumular_1t"]; ax[1].plot(th, [base / K[str(n)][f"acumular_{t}t"] for t in th], "o-", color=c, label=f"passo E, {n} estados")
        it1 = K[str(n)]["iter_1t"]
        ax[1].plot(th, [it1 / (K[str(n)]["forward"] + K[str(n)]["backward"] + K[str(n)][f"acumular_{t}t"]) for t in th], "s--", color=c, alpha=.6, label=f"iteração inteira, {n} estados")
    ax[1].plot(th, th, ":", color="gray", label="ideal"); ax[1].set_ylim(0, 4)
    ax[1].set(xlabel="threads", ylabel="aceleração (× 1 thread)", title="Ganho da paralelização"); ax[1].legend(fontsize=7); br(ax[1])
    save(fig, "13_desempenho")
