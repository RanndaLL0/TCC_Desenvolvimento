import argparse
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.mtd.mtd import MTD
from src.models.naive_models.first_order import FirstOrder
from src.models.naive_models.persistence import Persistence
from src.models.naive_models.uniform import Uniform
from src.preprocess import discretize as d
from src.utils.bic import bic

# Relatório do MTD sobre o snapshot: para cada discretização (modo, k) ajusta o
# MTD nas ordens 1..max_order e compara com os baselines no treino e no teste.
# Os números citados na documentação saem do JSON gerado aqui.

ROOT = Path(__file__).resolve().parents[2]
STD_FOLDER = ROOT / "relatorios"
LN3 = math.log(3)


## Regra de decisão que ignora o 0: escolhe entre -1 e +1 pela maior
# probabilidade (empate fica em -1, como o argmax). Colunas na ordem -1, 0, +1
def direction_rule(proba):
    return np.where(proba[:, 2] > proba[:, 0], 1, -1)


## Acerto de direção só nas posições em que o real é -1 ou +1.
# Devolve (acertos, total, taxa, intervalo de Wilson de 95%)
def directional_accuracy(proba, targets):
    nonzero = targets != 0
    n = int(nonzero.sum())
    hits = int((direction_rule(proba)[nonzero] == targets[nonzero]).sum())
    p = hits / n
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"hits": hits, "n": n, "rate": p, "ci": [center - half, center + half]}


## Separa treino e teste pelo corte da discretização (r.cut, em linhas de
# candle) e só depois descarta as posições sem símbolo. Filtrar antes e cortar
# com split_data daria outro ponto de corte
def split_symbols(result):
    train = result.symbols[: result.cut][result.valid[: result.cut]].astype(int)
    test = result.symbols[result.cut :][result.valid[result.cut :]].astype(int)
    train_segment = result.segment[: result.cut][result.valid[: result.cut]]
    test_segment = result.segment[result.cut :][result.valid[result.cut :]]
    return train, test, train_segment, test_segment


## Transições cujo contexto (as `order` posições anteriores) atravessa uma lacuna
def gap_contamination(segment, order):
    crossing = int((segment[order:] != segment[:-order]).sum())
    total = len(segment) - order
    return {"order": order, "crossing": crossing, "total": total}


def evaluate(train, test, max_order, n_restarts, seed):
    # contexto do teste = últimos max_order símbolos do treino + o teste. Cada
    # modelo usa a cauda de tamanho certa para avaliar exatamente as mesmas
    # transições (as len(test) do teste e as len(train) - max_order do treino)
    context = np.concatenate([train[-max_order:], test])
    n_train = len(train) - max_order

    train_nonzero = train[max_order:]
    test_nonzero = test

    rows = []
    for order in range(1, max_order + 1):
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = MTD(order=order, n_restarts=n_restarts, seed=seed)
            model.fit(train)
        elapsed = time.perf_counter() - t0

        ll_train = model.calculate_likelihood(train[max_order - order :])
        ll_test = model.calculate_likelihood(context[max_order - order :])

        proba_train = model.predict_proba(train)[max_order - 1 : -1]
        proba_test = model.predict_proba(context)[max_order - 1 : -1]

        rows.append(
            {
                "order": order,
                "n_params": model.n_params,
                "ll_train": ll_train,
                "bic": bic(ll_train, model.n_params, n_train),
                "ll_test_per_symbol": ll_test / len(test),
                "gain_vs_uniform_test": LN3 + ll_test / len(test),
                "dir_train": directional_accuracy(proba_train, train_nonzero),
                "dir_test": directional_accuracy(proba_test, test_nonzero),
                "zero_share_test": float((proba_test.argmax(axis=1) == 1).mean()),
                "iterations": model.info["iterations"],
                "converged": model.info["converged"],
                "ll_decreased_warnings": len(caught),
                "seconds": elapsed,
                "lam": model.lam.tolist(),
                "Q": model.Q.tolist(),
                "restarts": model.info["restarts"],
                "history": model.info["history"].tolist(),
            }
        )
        print(
            f"  order {order:2d}  bic {rows[-1]['bic']:.1f}  "
            f"test ll/sym {rows[-1]['ll_test_per_symbol']:.5f}  {elapsed:.1f}s",
            flush=True,
        )

    baselines = {}
    for name, model in (
        ("uniform", Uniform()),
        ("persistence", Persistence()),
        ("first_order", FirstOrder()),
    ):
        model.fit(train)
        baselines[name] = {
            "ll_test_per_symbol": model.calculate_likelihood(
                np.concatenate([train[-1:], test])
            )
            / len(test)
        }

    # baseline de ordem 0: frequências do treino, sem contexto. Separa "o 0 é
    # frequente" (e a distribuição do teste difere do treino) de "há estrutura"
    freq = np.bincount(train + 1, minlength=3) / len(train)
    baselines["marginal"] = {
        "ll_test_per_symbol": float(np.log(freq[test + 1]).mean())
    }

    # baseline de direção: sempre o sinal majoritário (entre os não-zero) do treino
    majority = 1 if (train == 1).sum() > (train == -1).sum() else -1
    nonzero = test != 0
    baselines["majority_direction"] = {
        "symbol": majority,
        "rate": float((test[nonzero] == majority).mean()),
        "n": int(nonzero.sum()),
    }
    return rows, baselines


def run_config(open_times, close, mode, k, max_order, n_restarts, seed):
    result = d.discretize(open_times, close, k=k, mode=mode)
    train, test, train_segment, test_segment = split_symbols(result)
    print(f"{mode} k={k}: train {len(train)}, test {len(test)}", flush=True)

    rows, baselines = evaluate(train, test, max_order, n_restarts, seed)

    return {
        "mode": mode,
        "k": k,
        "n_train": len(train),
        "n_test": len(test),
        "segments_train": int(len(np.unique(train_segment))),
        "boundary_in_same_segment": bool(train_segment[-1] == test_segment[0]),
        "class_share_train": (np.bincount(train + 1) / len(train)).tolist(),
        "class_share_test": (np.bincount(test + 1) / len(test)).tolist(),
        "gaps": [
            gap_contamination(train_segment, o)
            for o in sorted({1, min(6, max_order), max_order})
        ],
        "baselines": baselines,
        "orders": rows,
    }


def pct(value):
    return f"{100 * value:.2f}".replace(".", ",") + "%"


def thousands(value):
    return f"{value:,}".replace(",", ".")


def num(value, digits=4):
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def render_markdown(report):
    out = [
        "# Relatório do MTD",
        "",
        f"Gerado em {report['generated_at']} (UTC) por `src/experiments/mtd_report.py`. "
        "Notas do fluxo de uso: `docs/mtd-notas-de-fluxo.md`.",
        "",
        "## Fonte",
        "",
        f"- Snapshot: `{report['source']['file']}` "
        f"(SHA-256 `{report['source']['sha256']}`), {thousands(report['source']['rows'])} candles",
        f"- Ordens: 1 a {report['max_order']}; EM com tolerância relativa 1e-8, "
        f"piso 1e-10, semente {report['seed']}",
        "- Treino e teste: corte de 95% das linhas de candle (`Discretization.cut`)",
        "- **Todas as ordens são avaliadas nas mesmas transições** (treino: as últimas "
        "`N - ordem_máxima`; teste: todas, com o contexto vindo da cauda do treino), "
        "senão o BIC e a LL não são comparáveis entre ordens",
        "- Direção: acerto entre `-1` e `+1` ignorando o 0, só nas posições em que o "
        "real não é 0; intervalo de Wilson de 95%",
        "",
    ]

    for cfg in report["configs"]:
        b = cfg["baselines"]
        out += [
            f"## Discretização `{cfg['mode']}`, s/{cfg['k']}",
            "",
            f"- Símbolos: treino {thousands(cfg['n_train'])}, teste {thousands(cfg['n_test'])}",
            f"- Classes treino (-1/0/+1): {' / '.join(pct(v) for v in cfg['class_share_train'])}",
            f"- Classes teste (-1/0/+1): {' / '.join(pct(v) for v in cfg['class_share_test'])}",
            f"- Último símbolo do treino e primeiro do teste no mesmo segmento: "
            f"{'sim' if cfg['boundary_in_same_segment'] else 'não'}",
            "",
            "### Baselines (LL por símbolo no teste)",
            "",
            "| Modelo | LL/símbolo (teste) | Ganho sobre o uniforme |",
            "|---|---:|---:|",
        ]
        for name, label in (
            ("uniform", "uniforme"),
            ("persistence", "persistência (ε=0,01)"),
            ("marginal", "frequências do treino (ordem 0)"),
            ("first_order", "cadeia de 1ª ordem"),
        ):
            v = b[name]["ll_test_per_symbol"]
            out.append(f"| {label} | {num(v, 5)} | {num(LN3 + v, 5)} |")

        out += [
            "",
            f"Direção, sempre `{b['majority_direction']['symbol']:+d}` (sinal majoritário do "
            f"treino): **{pct(b['majority_direction']['rate'])}** "
            f"em {thousands(b['majority_direction']['n'])} posições não-zero do teste.",
            "",
            "### MTD por ordem",
            "",
            "| Ordem | Parâmetros | LL treino | BIC | LL/símbolo (teste) | Ganho s/ uniforme | "
            "Direção treino | Direção teste (IC 95%) | Prev. 0 (teste) | Iter. | Tempo |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
        best_bic = min(r["bic"] for r in cfg["orders"])
        for r in cfg["orders"]:
            mark = " **←**" if r["bic"] == best_bic else ""
            ci = r["dir_test"]["ci"]
            out.append(
                f"| {r['order']} | {r['n_params']} | {num(r['ll_train'], 1)} | "
                f"{num(r['bic'], 1)}{mark} | {num(r['ll_test_per_symbol'], 5)} | "
                f"{num(r['gain_vs_uniform_test'], 5)} | {pct(r['dir_train']['rate'])} | "
                f"{pct(r['dir_test']['rate'])} ({pct(ci[0])} a {pct(ci[1])}) | "
                f"{pct(r['zero_share_test'])} | {r['iterations']} | {r['seconds']:.0f}s |"
            )

        out += [
            "",
            "### Contaminação por lacunas (treino)",
            "",
            "Transições cujo contexto atravessa uma lacuna, contadas hoje como normais "
            "(D6 pendente em `docs/discretizacao.md`).",
            "",
            "| Ordem | Transições afetadas | Total | Fração |",
            "|---:|---:|---:|---:|",
        ]
        for g in cfg["gaps"]:
            out.append(
                f"| {g['order']} | {thousands(g['crossing'])} | {thousands(g['total'])} | "
                f"{pct(g['crossing'] / g['total'])} |"
            )
        out.append("")

    main = report["configs"][0]
    best = max(main["orders"], key=lambda r: r["order"])
    out += [
        f"## Detalhe: `{main['mode']}` s/{main['k']}, ordem {best['order']}",
        "",
        "Pesos por lag (`lam`, lag 1 primeiro) e matriz `Q` (linha = símbolo no lag, "
        "coluna = próximo símbolo; ordem -1, 0, +1).",
        "",
        "| Lag | " + " | ".join(str(i + 1) for i in range(best["order"])) + " |",
        "|---|" + "---:|" * best["order"],
        "| peso | " + " | ".join(num(v, 4) for v in best["lam"]) + " |",
        "",
        "| de \\ para | -1 | 0 | +1 |",
        "|---|---:|---:|---:|",
    ]
    for label, row in zip(("-1", "0", "+1"), best["Q"]):
        out.append(f"| {label} | " + " | ".join(num(v, 4) for v in row) + " |")

    out += [
        "",
        "### Estabilidade dos reinícios",
        "",
        "| Reinício | LL treino | Iterações | Convergiu |",
        "|---:|---:|---:|---|",
    ]
    for i, r in enumerate(best["restarts"]):
        out.append(
            f"| {i} | {num(r['ll'], 2)} | {r['iterations']} | {'sim' if r['converged'] else 'não'} |"
        )

    history = best["history"]
    final = history[-1]
    out += [
        "",
        "### Curva da LL no melhor reinício",
        "",
        "| Iteração | LL | Falta para a final |",
        "|---:|---:|---:|",
    ]
    for it in sorted({0, 1, 5, 10, 25, 50, 100, len(history) - 1}):
        if it < len(history):
            out.append(
                f"| {it} | {num(history[it], 2)} | {num(final - history[it], 2)} |"
            )

    out += [
        "",
        "## Resumo entre discretizações (melhor BIC por configuração)",
        "",
        "**Atenção:** BIC e LL só são comparáveis **dentro** da mesma discretização. "
        "Cada linha abaixo tem um alfabeto com outra distribuição (e, no modo `global`, o "
        "teste tem muito mais zeros que o treino), então comparar os valores entre linhas "
        "não mede qual discretização é melhor. A coluna \"Ganho s/ ordem 0\" desconta o que "
        "uma tabela de frequências já explica.",
        "",
    ]
    out += [
        "| Modo | s/k | Melhor BIC | Ordem | LL/símbolo (teste) | Ganho s/ ordem 0 | Direção teste | Sempre `+1` |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for cfg in report["configs"]:
        r = min(cfg["orders"], key=lambda x: x["bic"])
        out.append(
            f"| {cfg['mode']} | {cfg['k']} | {num(r['bic'], 1)} | {r['order']} | "
            f"{num(r['ll_test_per_symbol'], 5)} | "
            f"{num(r['ll_test_per_symbol'] - cfg['baselines']['marginal']['ll_test_per_symbol'], 5)} | "
            f"{pct(r['dir_test']['rate'])} | "
            f"{pct(cfg['baselines']['majority_direction']['rate'])} |"
        )
    out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Fits the MTD and writes the report")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument(
        "--main-restarts",
        type=int,
        default=5,
        help="restarts of the first configuration (stability check)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", nargs="+", choices=d.MODES, default=["blocks", "global"])
    parser.add_argument("--k", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--output", type=Path, default=STD_FOLDER)
    args = parser.parse_args()

    open_times, close, source = d.load_snapshot(args.snapshot)

    configs = []
    for mode in args.mode:
        for k in args.k:
            restarts = args.main_restarts if not configs else args.restarts
            configs.append(
                run_config(
                    open_times, close, mode, k, args.max_order, restarts, args.seed
                )
            )

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "max_order": args.max_order,
        "seed": args.seed,
        "configs": configs,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / f"mtd_{source['until_exclusive']}"
    base.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    base.with_suffix(".md").write_text(render_markdown(report))
    print(f"{base.with_suffix('.md')}\n{base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
