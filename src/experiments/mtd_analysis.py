import argparse
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.experiments.mtd_report import (
    directional_accuracy,
    gap_contamination,
    num,
    pct,
    split_symbols,
    thousands,
)
from src.models.hmm.hmm_numba import _forward
from src.models.mtd.mtd import MTD
from src.models.naive_models.first_order import FirstOrder
from src.models.naive_models.persistence import Persistence
from src.preprocess import discretize as d
from src.utils.bic import bic
from src.utils.symbols import to_symbols

# Análise do MTD com as métricas que o TCC promete (sem métricas sem referência no
# texto): log-verossimilhança e BIC, acurácia, precisão, recall e F1 por classe,
# matriz de confusão e o acerto de direção (acurácia restrita aos candles ±1),
# mais os baselines do escopo (uniforme, persistência, cadeia de 1ª ordem) e as
# frequências do treino (iid). Também: estrutura do MTD, efeito das lacunas e
# estabilidade dos reinícios.
#
# Convenções:
#   - tudo é avaliado nas MESMAS transições: treino = as últimas
#     N - ordem_máxima; teste = todas, com o contexto vindo do fim do treino;
#   - `pred[t]` é a distribuição de x[t] dado SÓ x[:t] (linhas: -1, 0, +1);
#   - o HMM é filtrado ao longo de treino + teste, então ele também começa o
#     teste com o estado do fim do treino.

ROOT = Path(__file__).resolve().parents[2]
STD_FOLDER = ROOT / "relatorios"
LN3 = math.log(3)
M = 3

# Reinícios que ficam a menos de 1 nat do melhor contam como "a mesma solução"
SAME_SOLUTION_NATS = 1.0


## ---------------------------------------------------------------- modelos


## Baseline de 1ª ordem (FirstOrder, do escopo) ajustado no treino. pred[t] é a
# distribuição de x[t] dado x[t-1]; a linha 0 não tem contexto e fica uniforme
def first_order_pred(train_symbols, x):
    model = FirstOrder()
    model.fit(train_symbols)
    return _shift_one(model.predict_proba(to_symbols(x)), len(x))


## Baseline de persistência (ε = 0,01, como em Persistence) ajustado no treino
def persistence_pred(train_symbols, x):
    model = Persistence()
    model.fit(train_symbols)
    return _shift_one(model.predict_prob(to_symbols(x)), len(x))


## predict_proba devolve, na linha i, a distribuição do que vem DEPOIS de x[i];
# a distribuição de x[t] é a linha t-1
def _shift_one(after, n):
    pred = np.full((n, M), 1.0 / M)
    pred[1:] = after[:-1]
    return pred


## Frequências do treino (ordem 0): a mesma linha para toda posição
def marginal_pred(x, n_train):
    freq = np.bincount(x[:n_train], minlength=M) / n_train
    return np.tile(freq, (len(x), 1))


def uniform_pred(x):
    return np.full((len(x), M), 1.0 / M)


## Reconstrói um MTD ajustado a partir de (lam, Q), sem refazer o EM
def restore_mtd(order, lam, Q, n_obs=None, info=None):
    model = MTD(order=order)
    model.lam = np.asarray(lam, dtype=float)
    model.Q = np.asarray(Q, dtype=float)
    model.n_obs = n_obs
    model.info = info
    return model


## pred[t] = distribuição de x[t] dado x[:t] no MTD. predict_proba devolve, na
# linha i, a distribuição do que vem DEPOIS de x[i]; basta deslocar uma linha.
# As `order` primeiras linhas usam lags parciais e não devem ser avaliadas
def mtd_pred(model, x):
    pred = np.full((len(x), M), 1.0 / M)
    pred[1:] = model.predict_proba(to_symbols(x))[:-1]
    return pred


## Predição de um passo do HMM, filtrada (só o passado): pred[t] =
# alpha[t-1] @ A @ B, com pred[0] = pi @ B. Devolve também o alpha de cada t
def hmm_pred(pi, A, B, y):
    y = np.ascontiguousarray(y, dtype=np.int64)
    alpha, _, _ = _forward(y, pi, A, B)
    pred = np.empty((len(y), B.shape[1]))
    pred[0] = pi @ B
    pred[1:] = (alpha[:-1] @ A) @ B
    return pred, alpha


## --------------------------------------------------------------- métricas


## Matriz de confusão (linhas = real, colunas = previsto; ordem −1, 0, +1) e
# métricas por classe para a decisão argmax. Também separa as duas perguntas:
# "vai ficar parado?" (0) e "vai se mexer?" (−1 ou +1). Precisão ou recall sem
# nenhuma previsão (ou sem nenhum caso real) saem como None; o F1 vale 0 nesses
# casos (a convenção do scikit-learn com zero_division=0)
def confusion_metrics(chosen, target):
    cm = np.zeros((M, M), dtype=int)
    np.add.at(cm, (target, chosen), 1)
    total = int(cm.sum())

    def ratio(num_, den):
        return float(num_ / den) if den else None

    per_class = []
    for c, symbol in enumerate((-1, 0, 1)):
        tp = int(cm[c, c])
        precision = ratio(tp, int(cm[:, c].sum()))
        recall = ratio(tp, int(cm[c].sum()))
        both = precision is not None and recall is not None and precision + recall > 0
        per_class.append(
            {
                "symbol": symbol,
                "support": int(cm[c].sum()),
                "predicted": int(cm[:, c].sum()),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if both else 0.0,
            }
        )

    real_moves = int(cm[0].sum() + cm[2].sum())
    predicted_zero = int(cm[:, 1].sum())
    predicted_move = total - predicted_zero
    hit_moves = int(cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2])
    return {
        "matrix": cm.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean([c["f1"] for c in per_class])),
        "movement": {
            "base_zero": ratio(int(cm[1].sum()), total),
            "predicted_zero": predicted_zero,
            "precision_zero": ratio(int(cm[1, 1]), predicted_zero),
            "recall_zero": ratio(int(cm[1, 1]), int(cm[1].sum())),
            "predicted_move": predicted_move,
            "precision_move": ratio(hit_moves, predicted_move),
            "recall_move": ratio(hit_moves, real_moves),
        },
    }


## Métricas de previsão de um passo sobre as posições `rows` (índices de x).
# pred: (n, 3); x: índices 0..2 (símbolo + 1)
def evaluate_predictions(pred, x, rows):
    p = pred[rows]
    t = x[rows]
    symbols = t - 1
    ll = float(np.log(p[np.arange(len(t)), t]).sum())

    chosen = p.argmax(axis=1)
    non_zero_pred = chosen != 1

    return {
        "n": int(len(t)),
        "ll": ll,
        "ll_per_symbol": ll / len(t),
        "accuracy": float((chosen == t).mean()),
        "direction": directional_accuracy(p, symbols),
        "confusion": confusion_metrics(chosen, t),
        "zero_share": float((chosen == 1).mean()),
        "non_zero_pred_share": float(non_zero_pred.mean()),
        "non_zero_pred_accuracy": (
            float((chosen[non_zero_pred] == t[non_zero_pred]).mean())
            if non_zero_pred.any()
            else None
        ),
    }


## Acurácia de "sempre 0" e de "repete o último símbolo" sobre `rows`
def naive_accuracy(x, rows):
    t = x[rows]
    return {
        "always_zero": float((t == 1).mean()),
        "persistence": float((x[rows - 1] == t).mean()),
    }


## Distribuição estacionária (autovetor à esquerda, autovalor 1) de uma matriz
# estocástica por linhas
def stationary_distribution(Q):
    values, vectors = np.linalg.eig(np.asarray(Q).T)
    i = int(np.argmin(np.abs(values - 1.0)))
    v = np.real(vectors[:, i])
    return v / v.sum()


## Memória do MTD. Como P(x_t = j | passado) é LINEAR nos indicadores dos lags,
# o desvio da marginal em relação à estacionária obedece
# m_t = sum_g lam_g m_{t-g} Q. Na direção de um autovetor de Q com autovalor mu,
# isso vira r^l = mu * sum_g lam_g r^(l-g). A raiz dominante de módulo < 1 dá o
# decaimento: tempo de relaxação = -1/ln|r| passos. Devolve uma lista, uma
# entrada por autovalor de Q diferente de 1
def memory_times(lam, Q):
    lam = np.asarray(lam, dtype=float)
    order = len(lam)
    values = np.linalg.eigvals(np.asarray(Q))
    values = values[np.argsort(-np.abs(values))][1:]  # tira o autovalor 1

    out = []
    for mu in values:
        # r^l - mu * (lam_1 r^(l-1) + ... + lam_l) = 0
        coefficients = np.concatenate(([1.0], -mu * lam))
        roots = np.roots(coefficients)
        r = float(np.max(np.abs(roots)))
        out.append(
            {
                "eigenvalue": complex(mu).real
                if abs(complex(mu).imag) < 1e-12
                else str(complex(mu)),
                "root_modulus": r,
                "relaxation_steps": (-1.0 / math.log(r)) if 0 < r < 1 else None,
            }
        )
    return out


## ------------------------------------------------------------------ HMM


## Lê pi, A e B do relatório do HMM (hmm_training.py) para o mesmo modo e k.
# Confere se a discretização do JSON é a mesma desta execução
def load_hmm(path, n_states, cut, n_train, n_test):
    report = json.loads(Path(path).read_text())
    disc = report["discretization"]
    if (
        disc["first_test_row"] != cut
        or disc["train_symbols"] != n_train
        or disc["test_symbols"] != n_test
    ):
        raise ValueError(
            f"{path} was trained on another split: {disc} vs "
            f"cut={cut}, train={n_train}, test={n_test}"
        )
    for entry in report["results"]:
        if entry["n_states"] == n_states:
            return entry
    raise ValueError(f"{path} has no result with {n_states} states")


## ------------------------------------------------------------- ajuste


## Ajusta o MTD de cada ordem (com cache em JSON, porque 10 reinícios por ordem
# levam minutos). A chave do cache inclui o SHA do snapshot e os tamanhos
def fit_orders(train, max_order, restarts, seed, tag, cache_dir):
    models = {}
    for order in range(1, max_order + 1):
        cache = None
        if cache_dir is not None:
            cache = Path(cache_dir) / f"mtd_{tag}_o{order}_r{restarts}_s{seed}.json"

        if cache is not None and cache.exists():
            saved = json.loads(cache.read_text())
            model = restore_mtd(
                order, saved["lam"], saved["Q"], saved["n_obs"], saved["info"]
            )
            model.seconds = saved["seconds"]
            model.ll = saved["ll"]
            model.ll_decreased_warnings = saved["ll_decreased_warnings"]
            print(f"  order {order:2d}: cache {cache.name}", flush=True)
        else:
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = MTD(order=order, n_restarts=restarts, seed=seed)
                model.fit(train)
            model.seconds = time.perf_counter() - t0
            model.ll_decreased_warnings = len(caught)
            info = {k: v for k, v in model.info.items() if k != "history"}
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(
                    json.dumps(
                        {
                            "lam": model.lam.tolist(),
                            "Q": model.Q.tolist(),
                            "ll": model.ll,
                            "n_obs": model.n_obs,
                            "info": info,
                            "seconds": model.seconds,
                            "ll_decreased_warnings": model.ll_decreased_warnings,
                        }
                    )
                )
            print(
                f"  order {order:2d}: fitted in {model.seconds:.0f}s "
                f"(ll {model.ll:.1f})",
                flush=True,
            )
        models[order] = model
    return models


def restart_summary(model):
    restarts = model.info["restarts"]
    lls = np.array([r["ll"] for r in restarts])
    best = float(lls.max())
    return {
        "restarts": len(restarts),
        "same_as_best": int((lls >= best - SAME_SOLUTION_NATS).sum()),
        "converged": int(sum(r["converged"] for r in restarts)),
        "worst_gap": float(best - lls.min()),
        "best_restart": int(model.info["restart"]),
        "iterations_best": int(model.info["iterations"]),
        "ll_decreased_warnings": int(model.ll_decreased_warnings),
        "all": [
            {"ll": r["ll"], "iterations": r["iterations"], "converged": r["converged"]}
            for r in restarts
        ],
    }


## ---------------------------------------------------------- configuração


def analyze_config(
    open_times,
    close,
    mode,
    k,
    max_order,
    restarts,
    seed,
    cache_dir,
    hmm_path,
    sha,
    interval_months=1,
):
    result = d.discretize(
        open_times, close, k=k, mode=mode, interval_months=interval_months
    )
    train_sym, test_sym, train_seg, test_seg = split_symbols(result)
    n_train, n_test = len(train_sym), len(test_sym)
    n_total = n_train + n_test
    x = np.concatenate([train_sym, test_sym]) + 1
    segment = np.concatenate([train_seg, test_seg])

    valid = result.valid
    times = np.concatenate(
        [
            open_times[: result.cut][valid[: result.cut]],
            open_times[result.cut :][valid[result.cut :]],
        ]
    )

    rows_train = np.arange(max_order, n_train)
    rows_test = np.arange(n_train, n_total)
    n_train_trans = len(rows_train)
    print(f"{mode} s/{k}: train {n_train}, test {n_test}", flush=True)

    # o cache do bloco de 1 mês mantém o nome antigo; os demais incluem o intervalo
    suffix = f"_i{interval_months}" if mode == "blocks" and interval_months != 1 else ""
    tag = f"{mode}_k{k}{suffix}_{sha[:8]}_{n_train}"
    models = fit_orders(train_sym, max_order, restarts, seed, tag, cache_dir)

    # ------------------------------------------------ modelos e métricas
    def full_metrics(name, n_params, pred):
        tr = evaluate_predictions(pred, x, rows_train)
        te = evaluate_predictions(pred, x, rows_test)
        return {
            "name": name,
            "n_params": n_params,
            "train": tr,
            "test": te,
            "aic": 2 * n_params - 2 * tr["ll"],
            "bic": bic(tr["ll"], n_params, n_train_trans) if n_params else None,
        }

    entries = []
    entries.append(full_metrics("uniform", 0, uniform_pred(x)))
    entries.append(full_metrics("marginal", M - 1, marginal_pred(x, n_train)))
    entries.append(
        full_metrics("persistence", 0, persistence_pred(train_sym, x))
    )
    entries.append(
        full_metrics("first_order", M * (M - 1), first_order_pred(train_sym, x))
    )

    for order, model in models.items():
        pred = mtd_pred(model, x)
        entries.append(full_metrics(f"mtd{order}", model.n_params, pred))

    hmm = None
    if hmm_path is not None and Path(hmm_path).exists():
        hmm = load_hmm(hmm_path, 3, result.cut, n_train, n_test)
        pi, A, B = (np.array(hmm[key]) for key in ("pi", "A", "B"))
        pred, _ = hmm_pred(pi, A, B, x)
        entries.append(full_metrics("hmm3", hmm["free_parameters"], pred))

    by_name = {e["name"]: e for e in entries}
    for e in entries:
        e["gain_vs_uniform_test"] = LN3 + e["test"]["ll_per_symbol"]
        e["gain_vs_marginal_test"] = (
            e["test"]["ll_per_symbol"] - by_name["marginal"]["test"]["ll_per_symbol"]
        )

    # ------------------------------------------------ reversão no teste
    after = {}
    for label, symbol in (("-1", 0), ("+1", 2)):
        mask = x[rows_test - 1] == symbol
        nxt = x[rows_test][mask]
        after[label] = {
            "n": int(mask.sum()),
            "next_minus": float((nxt == 0).mean()),
            "next_zero": float((nxt == 1).mean()),
            "next_plus": float((nxt == 2).mean()),
        }

    # ---------------------------------------------- estrutura do MTD
    structure = {}
    for order, model in models.items():
        stationary = stationary_distribution(model.Q)
        structure[order] = {
            "lam": model.lam.tolist(),
            "Q": model.Q.tolist(),
            "stationary": stationary.tolist(),
            "memory": memory_times(model.lam, model.Q),
            "restart_summary": restart_summary(model),
            "seconds": model.seconds,
        }

    # ------------------------------------------------------ lacunas
    gaps = {}
    for order in sorted({1, 6, max_order} & set(models)):
        pred = mtd_pred(models[order], x)
        crossing = segment[rows_train] != segment[rows_train - order]
        p = pred[rows_train, :][np.arange(len(rows_train)), x[rows_train]]
        logp = np.log(p)
        gaps[order] = {
            "crossing": int(crossing.sum()),
            "total": int(len(rows_train)),
            "ll_crossing": float(logp[crossing].sum()),
            "ll_all": float(logp.sum()),
            "ll_per_symbol_all": float(logp.mean()),
            "ll_per_symbol_clean": float(logp[~crossing].mean()),
        }
    gap_counts = [
        gap_contamination(train_seg, o) for o in sorted({1, 6, max_order})
    ]

    return {
        "mode": mode,
        "k": k,
        "interval_months": result.interval_months,
        "n_train": n_train,
        "n_test": n_test,
        "n_train_transitions": n_train_trans,
        "segments_train": int(len(np.unique(train_seg))),
        "class_share_train": (np.bincount(x[:n_train], minlength=M) / n_train).tolist(),
        "class_share_test": (np.bincount(x[n_train:], minlength=M) / n_test).tolist(),
        "test_start": str(times[n_train]),
        "test_end": str(times[-1]),
        "hmm_available": hmm is not None,
        "hmm_reported": (
            {
                key: hmm[key]
                for key in (
                    "ll_train",
                    "ll_test_per_symbol",
                    "bic",
                    "converged",
                    "best_restart",
                )
            }
            if hmm is not None
            else None
        ),
        "models": entries,
        "naive_accuracy_test": naive_accuracy(x, rows_test),
        "after_symbol_test": after,
        "structure": structure,
        "gaps": gaps,
        "gap_counts": gap_counts,
    }


## ------------------------------------------------------------ relatório


def label_of(name):
    if name == "uniform":
        return "Uniforme"
    if name == "marginal":
        return "Frequências do treino (iid)"
    if name == "persistence":
        return "Persistência (ε = 0,01)"
    if name == "first_order":
        return "Cadeia de 1ª ordem"
    if name.startswith("mtd"):
        return f"MTD ordem {name[3:]}"
    if name == "hmm3":
        return "HMM 3 estados"
    return name


## Intervalo do bloco de uma configuração (relatórios antigos não gravavam o
# campo: eram todos de 1 mês). O modo global não tem blocos
def config_interval(cfg):
    if cfg["mode"] != "blocks":
        return None
    return cfg.get("interval_months") or 1


## Resumo entre discretizações: uma linha por configuração, com o MTD da ordem de
# melhor BIC. BIC e LL só se comparam DENTRO da mesma discretização; a coluna
# "Ganho s/ iid" desconta o que uma tabela de frequências já explica
def render_summary(configs):
    out = [
        "## Resumo entre discretizações",
        "",
        "**Atenção:** BIC e LL só são comparáveis **dentro** da mesma discretização, porque "
        "cada uma tem outra distribuição de símbolos (e, no modo `global`, o teste tem muito "
        "mais zeros que o treino). Entre linhas, vale comparar o ganho sobre o iid, a acurácia "
        "contra \"sempre 0\", o F1 das classes ±1 e a direção. Cada linha mostra o MTD da ordem "
        "de menor BIC; \"Melhor constante\" é o melhor acerto de direção de um modelo que prevê "
        "sempre o mesmo sinal, escolhido olhando o teste; \"F1 de ±1\" é a média do F1 das "
        "classes −1 e +1.",
        "",
        "| Modo | Bloco (meses) | s/k | Zeros treino | Zeros teste | Ordem (menor BIC) | "
        "LL/sím. teste | Ganho s/ iid | Acurácia (sempre 0) | F1 de ±1 | "
        "Direção teste (IC 95%) | Melhor constante |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for cfg in sorted(
        configs, key=lambda c: (c["mode"], config_interval(c) or 0, c["k"])
    ):
        mtd = [e for e in cfg["models"] if e["name"].startswith("mtd")]
        best = min(mtd, key=lambda e: e["bic"])
        names = {e["name"]: e for e in cfg["models"]}
        rate = names["marginal"]["test"]["direction"]["rate"]
        ci = best["test"]["direction"]["ci"]
        per_class = best["test"]["confusion"]["per_class"]
        f1_moves = (per_class[0]["f1"] + per_class[2]["f1"]) / 2
        out.append(
            f"| {cfg['mode']} | {config_interval(cfg) or '—'} | {cfg['k']} | "
            f"{pct(cfg['class_share_train'][1])} | "
            f"{pct(cfg['class_share_test'][1])} | {best['name'][3:]} | "
            f"{num(best['test']['ll_per_symbol'], 5)} | {num(best['gain_vs_marginal_test'], 5)} | "
            f"{pct(best['test']['accuracy'])} ({pct(cfg['naive_accuracy_test']['always_zero'])}) | "
            f"{num(f1_moves, 3)} | "
            f"{pct(best['test']['direction']['rate'])} ({pct(ci[0])} a {pct(ci[1])}) | "
            f"{pct(max(rate, 1 - rate))} |"
        )
    out.append("")
    return out


def pct_or_dash(value):
    return "—" if value is None else pct(value)


## Seção do relatório: matriz de confusão e métricas por classe (teste, argmax)
def render_confusion(cfg):
    names = {e["name"]: e for e in cfg["models"]}
    mtd = [e["name"] for e in cfg["models"] if e["name"].startswith("mtd")]
    best = min(mtd, key=lambda n: names[n]["bic"])
    chosen = [n for n in ("first_order", "mtd6", best, "hmm3") if n in names]
    chosen = list(dict.fromkeys(chosen))  # sem repetir (ex.: ordem 6 = melhor BIC)

    zeros = names["marginal"]["test"]["confusion"]["movement"]["base_zero"]
    sym = {-1: "−1", 0: "0", 1: "+1"}
    out = [
        "",
        "#### Zero e movimento (teste, decisão argmax)",
        "",
        "Duas perguntas separadas: \"vai ficar parado?\" (símbolo 0) e \"vai se mexer?\" "
        "(−1 ou +1). Base: "
        f"{pct(zeros)} dos candles de teste ficaram parados e {pct(1 - zeros)} se moveram. "
        "O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.",
        "",
        "| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | "
        "Precisão do movimento | Recall do movimento |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Sempre 0 | 100,00% | {pct(zeros)} | 100,00% | 0,00% | — | 0,00% |",
    ]
    for n in chosen:
        t = names[n]["test"]["confusion"]
        m = t["movement"]
        total = names[n]["test"]["n"]
        out.append(
            f"| {label_of(n)} | {pct(m['predicted_zero'] / total)} | "
            f"{pct_or_dash(m['precision_zero'])} | {pct_or_dash(m['recall_zero'])} | "
            f"{pct(m['predicted_move'] / total)} | {pct_or_dash(m['precision_move'])} | "
            f"{pct_or_dash(m['recall_move'])} |"
        )

    out += [
        "",
        "#### Métricas por classe (teste, decisão argmax)",
        "",
        "Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o "
        "modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de "
        "direção separa as duas.",
        "",
        "| Modelo | −1 | 0 | +1 | F1 médio |",
        "|---|---|---|---|---:|",
    ]
    for n in chosen:
        c = names[n]["test"]["confusion"]
        cells = [
            f"{pct_or_dash(k['precision'])} / {pct_or_dash(k['recall'])} / {num(k['f1'], 3)}"
            for k in c["per_class"]
        ]
        out.append(f"| {label_of(n)} | " + " | ".join(cells) + f" | {num(c['macro_f1'], 3)} |")

    for n in dict.fromkeys([best] + (["hmm3"] if "hmm3" in names else [])):
        matrix = names[n]["test"]["confusion"]["matrix"]
        out += [
            "",
            f"Matriz de confusão, {label_of(n)} (linhas = real, colunas = previsto):",
            "",
            "| Real \\ Previsto | −1 | 0 | +1 | Total |",
            "|---|---:|---:|---:|---:|",
        ]
        for c, row in zip((-1, 0, 1), matrix):
            out.append(
                f"| {sym[c]} | " + " | ".join(thousands(v) for v in row) + f" | {thousands(sum(row))} |"
            )
    return out


## Configurações com seção detalhada no relatório (as demais ficam só nas matrizes
# de resumo e no JSON). Formato "modo:k" ou "modo:k:bloco", por exemplo
# "blocks:3:1" (blocos de 1 mês, s/3) e "global:3"
DEFAULT_DETAIL = ("blocks:3:1", "global:3")


def parse_detail(specs):
    keys = set()
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in (2, 3) or parts[0] not in d.MODES:
            raise ValueError(f"detail should be mode:k or mode:k:interval, got {spec!r}")
        mode, k = parts[0], int(parts[1])
        interval = None
        if mode == "blocks":
            interval = int(parts[2]) if len(parts) == 3 else 1
        keys.add((mode, k, interval))
    return keys


def is_detail(cfg, keys):
    return (cfg["mode"], cfg["k"], config_interval(cfg)) in keys


## MTD da ordem de menor BIC de uma configuração (é o que as matrizes mostram)
def best_mtd(cfg):
    mtd = [e for e in cfg["models"] if e["name"].startswith("mtd")]
    return min(mtd, key=lambda e: e["bic"])


def signed(value, digits=2):
    return f"{value:+.{digits}f}".replace(".", ",")


## Valores de cada célula das matrizes de resumo, para um modelo `entry`
# (hoje o melhor MTD; o relatório do HMM usará o mesmo contrato). Cada função
# recebe (cfg, entry) e devolve o texto da célula
def zeros_cell(cfg, entry):
    return f"{pct(cfg['class_share_train'][1])} → {pct(cfg['class_share_test'][1])}"


def ll_cell(cfg, entry):
    return num(entry["test"]["ll_per_symbol"], 4)


def gain_cell(cfg, entry):
    return num(entry["gain_vs_marginal_test"], 4)


def accuracy_cell(cfg, entry):
    margin = 100 * (entry["test"]["accuracy"] - cfg["naive_accuracy_test"]["always_zero"])
    return f"{pct(entry['test']['accuracy'])} ({signed(margin, 1)} p.p.)"


def f1_moves_cell(cfg, entry):
    per_class = entry["test"]["confusion"]["per_class"]
    return num((per_class[0]["f1"] + per_class[2]["f1"]) / 2, 3)


def best_constant(cfg):
    names = {e["name"]: e for e in cfg["models"]}
    rate = names["marginal"]["test"]["direction"]["rate"]
    return max(rate, 1 - rate)


def direction_cell(cfg, entry):
    return pct(entry["test"]["direction"]["rate"])


## Margem do acerto de direção sobre a melhor constante, em pontos percentuais.
# O asterisco marca os casos em que o limite inferior do IC de 95% já passa da
# melhor constante (comparação conservadora: a constante é escolhida olhando o teste)
def margin_cell(cfg, entry):
    t = entry["test"]["direction"]
    constant = best_constant(cfg)
    mark = "*" if t["ci"][0] > constant else ""
    return f"{signed(100 * (t['rate'] - constant), 2)}{mark}"


GRIDS = (
    ("Zeros: treino → teste", zeros_cell),
    ("Log-verossimilhança por símbolo no teste (nats)", ll_cell),
    ("Ganho de LL sobre o iid (nats por símbolo)", gain_cell),
    ("Acurácia no teste (diferença para \"sempre 0\")", accuracy_cell),
    ("F1 médio das classes −1 e +1", f1_moves_cell),
    ("Acerto de direção no teste", direction_cell),
    ("Margem da direção sobre a melhor constante (p.p.; * = IC 95% acima dela)", margin_cell),
)


def grid_column(cfg):
    return "global" if cfg["mode"] == "global" else config_interval(cfg)


def months_label(n):
    return f"{n} mês" if n == 1 else f"{n} meses"


def grid_label(column):
    return "*Global*" if column == "global" else f"*Blocks* {months_label(column)}"


## Matrizes de resumo: uma linha por k, uma coluna por modo e tamanho de bloco.
# Cada matriz mostra o MTD da ordem de menor BIC; células sem configuração
# calculada ficam "—"
def render_grids(configs, model_of=best_mtd):
    cells = {(c["k"], grid_column(c)): c for c in configs}
    ks = sorted({k for k, _ in cells})
    columns = sorted({col for _, col in cells}, key=lambda c: (c != "global", c if c != "global" else 0))
    out = [
        "## Matrizes de resumo",
        "",
        "Cada célula é uma configuração de discretização: **linha = k** (limiar s/k), "
        "**coluna = modo** (*Global* ou *Blocks* com o tamanho do bloco em meses). O modelo "
        "mostrado é o MTD da ordem de menor BIC de cada configuração. **LL e BIC só são "
        "comparáveis dentro da mesma configuração**, porque cada uma tem outra distribuição "
        "de símbolos; entre células, vale olhar o ganho sobre o iid, a acurácia contra "
        "\"sempre 0\", o F1 e a direção.",
        "",
    ]
    for title, cell in GRIDS:
        out += [
            f"### {title}",
            "",
            "| k | " + " | ".join(grid_label(c) for c in columns) + " |",
            "|---:|" + "---:|" * len(columns),
        ]
        for k in ks:
            row = []
            for col in columns:
                cfg = cells.get((k, col))
                row.append("—" if cfg is None else cell(cfg, model_of(cfg)))
            out.append(f"| s/{k} | " + " | ".join(row) + " |")
        out.append("")
    return out


## Verificações agregadas sobre todas as configurações (ajuste, ordem e lacunas)
def render_checks(configs, max_order, restarts):
    def label(cfg):
        interval = config_interval(cfg)
        where = "global" if interval is None else f"blocks {interval}m"
        return f"{where} s/{cfg['k']}"

    border = [label(c) for c in configs if int(best_mtd(c)["name"][3:]) != max_order]
    unstable, converged_all, worst = [], True, 0.0
    gap_effect = 0.0
    for c in configs:
        for order, st in c["structure"].items():
            r = st["restart_summary"]
            worst = max(worst, r["worst_gap"])
            if r["same_as_best"] != restarts:
                unstable.append(f"{label(c)}, ordem {order}")
            if r["converged"] != restarts:
                converged_all = False
        for g in c["gaps"].values():
            gap_effect = max(
                gap_effect, abs(g["ll_per_symbol_all"] - g["ll_per_symbol_clean"])
            )

    out = [
        "## Verificações em todas as configurações",
        "",
        f"- **Configurações calculadas:** {len(configs)}.",
        f"- **Ordem de menor BIC:** é a ordem máxima testada ({max_order}) em "
        + (
            f"todas as {len(configs)} configurações; o BIC não encontra um mínimo interno."
            if not border
            else f"{len(configs) - len(border)} de {len(configs)}; exceções: {', '.join(border)}."
        ),
        f"- **Reinícios do EM ({restarts} por ordem):** "
        + (
            "todos chegam à mesma solução (±1 nat) em todas as ordens de todas as configurações"
            if not unstable
            else f"há {len(unstable)} casos sem concordância total, por exemplo {', '.join(unstable[:3])}"
        )
        + f"; a maior diferença entre o melhor e o pior reinício é de {num(worst, 3)} nats"
        + ("; todos convergiram." if converged_all else "; nem todos convergiram."),
        "- **Lacunas:** a maior diferença na LL por símbolo entre incluir e excluir as "
        f"transições que atravessam uma lacuna é de {num(gap_effect, 6)} nats.",
        "",
    ]
    return out


def render_markdown(report, detail=DEFAULT_DETAIL):
    out = [
        "# Relatório final do MTD",
        "",
        f"Gerado em {report['generated_at']} (UTC) por `src/experiments/mtd_analysis.py`. "
        "Métricas: log-verossimilhança e BIC, acurácia, precisão, *recall*, F1, matriz de "
        "confusão e acerto de direção.",
        "",
        "## Fonte e parâmetros",
        "",
        f"- Snapshot: `{report['source']['file']}` (SHA-256 `{report['source']['sha256']}`), "
        f"{thousands(report['source']['rows'])} candles",
        f"- MTD: ordens 1 a {report['max_order']}, **{report['restarts']} reinícios por ordem**, "
        f"EM com tolerância relativa 1e-8, `max_iter` 1000, piso 1e-10, semente {report['seed']}",
        "- Treino e teste: corte de 95% das linhas de candle. **Todos os modelos são avaliados nas "
        "mesmas transições**: treino = as últimas `N − ordem_máxima`; teste = todas, com o "
        "contexto vindo do fim do treino",
        "- O HMM (quando aparece) usa os parâmetros do JSON de `hmm_training.py` e é **filtrado "
        "ao longo de treino e teste**, ou seja, começa o teste com o estado do fim do treino",
        "- Unidade: log-verossimilhança por símbolo, em nats. Uniforme = −ln 3 = −1,0986",
        "",
        "## Como ler",
        "",
        "1. As **matrizes de resumo** mostram, para cada k e cada modo de discretização, as "
        "métricas principais do MTD.",
        "2. As **verificações** resumem o ajuste (reinícios, ordem de menor BIC, lacunas) "
        "nas configurações todas.",
        "3. A **tabela completa** traz uma linha por configuração, com os números exatos.",
        "4. As **seções detalhadas** (baselines, matriz de confusão, estrutura do MTD) existem "
        "só para as configurações centrais; as demais estão no `.json`.",
        "",
    ]

    keys = parse_detail(detail)
    configs = report["configs"]
    if configs:
        out += render_grids(configs)
        out += render_checks(configs, report["max_order"], report["restarts"])
        out += ["## Tabela completa (uma linha por configuração)", ""]
        out += render_summary(configs)[2:]

    shown = [c for c in configs if is_detail(c, keys)]
    for cfg in shown:
        names = {e["name"]: e for e in cfg["models"]}
        title = f"Discretização `{cfg['mode']}`, s/{cfg['k']}"
        if cfg["mode"] == "blocks":
            title += f", blocos de {months_label(config_interval(cfg))}"
        out += [
            f"## {title}",
            "",
            f"- Símbolos: treino {thousands(cfg['n_train'])}, teste {thousands(cfg['n_test'])} "
            f"({cfg['test_start'][:16]} a {cfg['test_end'][:16]}); "
            f"{cfg['segments_train']} segmentos no treino",
            f"- Classes treino (−1/0/+1): {' / '.join(pct(v) for v in cfg['class_share_train'])}",
            f"- Classes teste (−1/0/+1): {' / '.join(pct(v) for v in cfg['class_share_test'])}",
            "",
            "### 1. Baselines, MTD e HMM: log-verossimilhança e BIC",
            "",
            "| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | "
            "Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for e in cfg["models"]:
            best = min(
                (m["bic"] for m in cfg["models"] if m["bic"] is not None)
            )
            mark = " **←**" if e["bic"] is not None and e["bic"] == best else ""
            bic_text = "—" if e["bic"] is None else num(e["bic"], 1) + mark
            out.append(
                f"| {label_of(e['name'])} | {thousands(e['n_params'])} | "
                f"{num(e['train']['ll_per_symbol'], 5)} | {num(e['test']['ll_per_symbol'], 5)} | "
                f"{bic_text} | {num(e['gain_vs_uniform_test'], 5)} | "
                f"{num(e['gain_vs_marginal_test'], 5)} |"
            )
        if cfg["hmm_reported"]:
            h = cfg["hmm_reported"]
            out += [
                "",
                f"HMM no relatório original: LL de treino {num(h['ll_train'], 1)} (soma de T termos), "
                f"BIC {num(h['bic'], 1)}, teste {num(h['ll_test_per_symbol'], 5)} por símbolo "
                "(começando na distribuição estacionária). A linha acima recalcula nas mesmas "
                "transições do MTD.",
            ]

        out += [
            "",
            "### 2. Previsão de um passo no teste",
            "",
            "| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | Prev. 0 |",
            "|---|---:|---:|---|---:|",
        ]
        naive = cfg["naive_accuracy_test"]
        out.append(
            f"| Sempre 0 | — | {pct(naive['always_zero'])} | — | 100,00% |"
        )
        for e in cfg["models"]:
            if e["name"] in ("uniform", "marginal"):
                continue
            t = e["test"]
            ci = t["direction"]["ci"]
            out.append(
                f"| {label_of(e['name'])} | {num(t['ll_per_symbol'], 5)} | {pct(t['accuracy'])} | "
                f"{pct(t['direction']['rate'])} ({pct(ci[0])} a {pct(ci[1])}) | "
                f"{pct(t['zero_share'])} |"
            )
        majority = names["marginal"]["test"]["direction"]
        out += [
            "",
            "Direção: entre os candles de teste que se moveram "
            f"({thousands(majority['n'])}), fração em que o modelo deu mais probabilidade ao sinal "
            "correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p.",
            "",
            "Depois de um símbolo, o que vem no teste:",
            "",
            "| Depois de | n | próximo −1 | próximo 0 | próximo +1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for label in ("-1", "+1"):
            a = cfg["after_symbol_test"][label]
            out.append(
                f"| {label} | {thousands(a['n'])} | {pct(a['next_minus'])} | "
                f"{pct(a['next_zero'])} | {pct(a['next_plus'])} |"
            )

        out += render_confusion(cfg)

        out += ["", "### 3. Estrutura do MTD", ""]
        out += [
            "| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | "
            "Relaxação (passos) | Relaxação (min) |",
            "|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        for order, s in cfg["structure"].items():
            lam = s["lam"]
            mem = s["memory"][0]
            relax = mem["relaxation_steps"]
            out.append(
                f"| {order} | {num(lam[0], 3)} | "
                f"{num(lam[1], 3) if len(lam) > 1 else '—'} | {num(lam[-1], 3)} | "
                f"{' / '.join(pct(v) for v in s['stationary'])} | "
                f"{num(mem['eigenvalue'], 3) if isinstance(mem['eigenvalue'], float) else mem['eigenvalue']} | "
                f"{num(relax, 1) if relax else '—'} | {num(relax * 5, 0) if relax else '—'} |"
            )
        top = max(cfg["structure"], key=int)
        s = cfg["structure"][top]
        out += [
            "",
            f"`Q` da ordem {top} (linha = símbolo no lag, coluna = próximo símbolo):",
            "",
            "| de \\ para | −1 | 0 | +1 |",
            "|---|---:|---:|---:|",
        ]
        for label, row in zip(("−1", "0", "+1"), s["Q"]):
            out.append(f"| {label} | " + " | ".join(num(v, 4) for v in row) + " |")
        out += [
            "",
            f"Pesos por lag (ordem {top}, lag 1 primeiro): "
            + ", ".join(num(v, 3) for v in s["lam"]),
        ]

        out += [
            "",
            "### 4. Estabilidade dos reinícios",
            "",
            "| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | "
            "Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for order, s in cfg["structure"].items():
            r = s["restart_summary"]
            out.append(
                f"| {order} | {r['restarts']} | {r['same_as_best']} | {r['converged']} | "
                f"{num(r['worst_gap'], 2)} | {r['best_restart']} | {r['iterations_best']} | "
                f"{r['ll_decreased_warnings']} | {s['seconds']:.0f}s |"
            )

        out += [
            "",
            "### 5. Lacunas (treino)",
            "",
            "Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. "
            "Abaixo, o que elas valem na LL do modelo ajustado:",
            "",
            "| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for order, g in cfg["gaps"].items():
            out.append(
                f"| {order} | {thousands(g['crossing'])} | {pct(g['crossing'] / g['total'])} | "
                f"{num(g['ll_crossing'], 1)} | {num(g['ll_per_symbol_all'], 5)} | "
                f"{num(g['ll_per_symbol_clean'], 5)} |"
            )
        out.append("")

    others = [c for c in configs if not is_detail(c, keys)]
    if others:
        out += [
            "## Configurações sem seção detalhada",
            "",
            "Os números completos de cada uma estão no `.json`: "
            + ", ".join(
                f"{'global' if c['mode'] == 'global' else 'blocks ' + str(config_interval(c)) + 'm'} s/{c['k']}"
                for c in others
            )
            + ".",
            "",
        ]
    return "\n".join(out)


def jsonable(value):
    """Converte chaves int em str e arrays do numpy, para o json.dumps."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


## Relatórios anteriores traziam métricas que saíram do escopo (AUC, Brier,
# calibração, horizontes, sazonalidade, simulação): ao reaproveitar uma
# configuração, esses campos são descartados para o relatório novo não os carregar
REMOVED_CONFIG_KEYS = ("horizon", "calibration", "seasonal", "adequacy")
REMOVED_METRIC_KEYS = ("auc_movement", "brier")


def drop_removed_metrics(cfg):
    cfg = {k: v for k, v in cfg.items() if k not in REMOVED_CONFIG_KEYS}
    models = []
    for e in cfg.get("models", []):
        e = dict(e)
        for part in ("train", "test"):
            if part in e:
                e[part] = {
                    k: v for k, v in e[part].items() if k not in REMOVED_METRIC_KEYS
                }
        models.append(e)
    cfg["models"] = models
    return cfg


## Configurações já calculadas num relatório anterior (--reuse). Só vale se a
# fonte (SHA-256) e os parâmetros do MTD forem os mesmos; senão os números
# misturariam execuções diferentes. Devolve {(modo, k): configuração}
def load_reusable(path, source, max_order, restarts, seed):
    previous = json.loads(Path(path).read_text())
    for key, value in (
        ("max_order", max_order),
        ("restarts", restarts),
        ("seed", seed),
    ):
        if previous[key] != value:
            raise ValueError(
                f"{path} was generated with {key}={previous[key]}, not {value}"
            )
    if previous["source"]["sha256"] != source["sha256"]:
        raise ValueError(f"{path} was generated from another snapshot")
    # configurações de relatórios anteriores à matriz de confusão não têm os campos
    # que o relatório de hoje exige: são recalculadas em vez de reaproveitadas
    reusable = {}
    for c in previous["configs"]:
        if not all(
            "confusion" in e.get("test", {"confusion": 1}) for e in c.get("models", [])
        ):
            continue
        reusable[(c["mode"], c["k"], config_interval(c))] = drop_removed_metrics(c)
    return reusable


def main():
    parser = argparse.ArgumentParser(
        description="Full MTD analysis, with the same metrics as the HMM analysis"
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=d.MODES, default=["blocks", "global"])
    parser.add_argument("--k", nargs="+", type=int, default=[3])
    parser.add_argument(
        "--interval",
        nargs="+",
        type=int,
        default=[1],
        help="months per block (blocks mode only; global ignores it). The HMM "
        "comparison is only available for 1, the interval it was trained with",
    )
    parser.add_argument(
        "--hmm-dir",
        type=Path,
        default=STD_FOLDER,
        help="folder with hmm_<date>_<mode>_k<k>.json (hmm_training.py); modes without it skip the HMM",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="folder to cache the fitted MTDs (a rerun then skips the EM)",
    )
    parser.add_argument(
        "--reuse",
        nargs="+",
        type=Path,
        default=[],
        help="previous reports (.json): their configurations (same snapshot and MTD "
        "parameters) are copied instead of recomputed; with several files, the "
        "last one that has a configuration wins",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="appended to the output name (mtd_<date>_completa<suffix>), so a new "
        "report does not overwrite an existing one",
    )
    parser.add_argument(
        "--detail",
        nargs="*",
        default=list(DEFAULT_DETAIL),
        help="configurations that get a detailed section, as mode:k or "
        "mode:k:interval (default: blocks:3:1 global:3)",
    )
    parser.add_argument("--output", type=Path, default=STD_FOLDER)
    args = parser.parse_args()
    parse_detail(args.detail)  # valida antes de calcular qualquer coisa

    open_times, close, source = d.load_snapshot(args.snapshot)

    reusable = {}
    origin = {}
    for path in args.reuse:
        found = load_reusable(path, source, args.max_order, args.restarts, args.seed)
        reusable.update(found)
        origin.update({key: path.name for key in found})

    configs = []
    for mode in args.mode:
        # o modo global não tem blocos: uma única configuração por k
        intervals = args.interval if mode == "blocks" else [None]
        for k in args.k:
            for interval in intervals:
                if (mode, k, interval) in reusable:
                    print(
                        f"{mode} s/{k} i={interval}: reused from {origin[(mode, k, interval)]}",
                        flush=True,
                    )
                    configs.append(reusable[(mode, k, interval)])
                    continue
                hmm_path = None
                if interval in (None, 1):
                    hmm_path = (
                        args.hmm_dir
                        / f"hmm_{source['until_exclusive']}_{mode}_k{k}.json"
                    )
                configs.append(
                    analyze_config(
                        open_times,
                        close,
                        mode,
                        k,
                        args.max_order,
                        args.restarts,
                        args.seed,
                        args.cache,
                        hmm_path,
                        source["sha256"],
                        interval_months=interval or 1,
                    )
                )

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "max_order": args.max_order,
        "restarts": args.restarts,
        "seed": args.seed,
        "configs": configs,
    }
    report = jsonable(report)

    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / f"mtd_{source['until_exclusive']}_completa{args.suffix}"
    base.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    base.with_suffix(".md").write_text(render_markdown(report, args.detail))
    print(f"{base.with_suffix('.md')}\n{base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
