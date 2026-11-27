import argparse
import json
import math
from pathlib import Path
from statistics import median

# Tabelas LaTeX do capítulo de resultados do TCC, geradas direto do JSON do
# relatório final do MTD (mtd_analysis.py). As tabelas saem direto do JSON; os números
# citados no texto vêm dessas tabelas ou do relatório em .md.

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "relatorios" / "mtd_2026-10-01_completa_final.json"
DEFAULT_OUT = ROOT / "relatorios" / "mtd_tabelas_tcc.tex"

# Configurações mostradas nas tabelas de exemplo: (modo, meses do bloco, k)
REPRESENTATIVE = [
    ("blocks", 1, 3),
    ("blocks", 3, 3),
    ("blocks", 12, 3),
    ("blocks", 3, 5),
    ("blocks", 3, 10),
    ("global", None, 3),
    ("global", None, 5),
]
ORDERS_SHOWN = (1, 2, 3, 5, 10)


## ------------------------------------------------------------ formatação


## Número no padrão brasileiro: vírgula decimal e ponto de milhar
def num(x, decimals=2, sign=False):
    text = f"{x:{'+' if sign else ''},.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, decimals=2):
    return num(100 * x, decimals) + "\\%"


## Cabeçalho de duas linhas, alinhado à direita, para caber na margem
def stack(top, bottom):
    return f"\\shortstack[r]{{{top}\\\\{bottom}}}"


def pp(x, decimals=2):
    return num(100 * x, decimals, sign=True)


## -------------------------------------------------------------- extração


def model_of(cfg, name):
    return next(m for m in cfg["models"] if m["name"] == name)


## MTD da ordem de menor BIC (o mesmo que o relatório em .md mostra)
def best_mtd(cfg):
    mtds = [m for m in cfg["models"] if m["name"].startswith("mtd")]
    return min(mtds, key=lambda m: m["bic"])


def find(report, mode, months, k):
    return next(
        c
        for c in report["configs"]
        if c["mode"] == mode and c["k"] == k and c.get("interval_months") == months
    )


## Melhor constante de direção: a maior fração entre −1 e +1 nos candles que se
# moveram no teste (a matriz de confusão tem o real nas linhas)
def best_constant(model):
    matrix = model["test"]["confusion"]["matrix"]
    down, up = sum(matrix[0]), sum(matrix[2])
    return max(down, up) / (down + up)


## p-valor unilateral (aproximação normal da binomial) do acerto contra p0
def one_sided_p(hits, n, p0):
    z = (hits - n * p0) / math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


## Métricas de uma configuração, todas para o MTD da ordem de menor BIC
def config_metrics(cfg):
    mtd = best_mtd(cfg)
    first = model_of(cfg, "first_order")
    marginal = model_of(cfg, "marginal")
    persistence = model_of(cfg, "persistence")
    direction = mtd["test"]["direction"]
    const = best_constant(mtd)
    zero_train, zero_test = cfg["class_share_train"][1], cfg["class_share_test"][1]
    majority = max(cfg["class_share_test"])
    ll_mtd = mtd["test"]["ll_per_symbol"]
    ll_iid = marginal["test"]["ll_per_symbol"]
    return {
        "mode": cfg["mode"],
        "months": cfg.get("interval_months"),
        "k": cfg["k"],
        "zero_train": zero_train,
        "zero_test": zero_test,
        "move_base": 1 - zero_test,
        "move_precision": mtd["test"]["confusion"]["movement"]["precision_move"],
        "accuracy": mtd["test"]["accuracy"],
        "majority": majority,
        "gain_iid": ll_mtd - ll_iid,
        "entropy_share": (ll_mtd - ll_iid) / -ll_iid,
        "direction": direction["rate"],
        "direction_first": first["test"]["direction"]["rate"],
        "direction_persistence": persistence["test"]["direction"]["rate"],
        "constant": const,
        "margin": direction["rate"] - const,
        "ci_above_constant": direction["ci"][0] > const,
        "p_value": one_sided_p(direction["hits"], direction["n"], const),
    }


## -------------------------------------------------------------- tabelas


def table_order(report):
    cfg = find(report, "blocks", 1, 3)
    rows = [("Frequências do treino (iid)", "marginal")]
    rows += [(f"MTD de ordem {o}", f"mtd{o}") for o in ORDERS_SHOWN]
    rows += [("HMM com 3 estados", "hmm3")]
    iid_ll = model_of(cfg, "marginal")["test"]["ll_per_symbol"]
    lines = []
    for label, name in rows:
        m = model_of(cfg, name)
        d = m["test"]["direction"]
        lines.append(
            f"    {label} & {m['n_params']} & {num(m['test']['ll_per_symbol'], 4)} "
            f"& {num(m['test']['ll_per_symbol'] - iid_ll, 4)} & {num(m['bic'], 0)} "
            f"& {pct(d['rate'])} \\\\"
        )
    return TABLE.format(
        caption="Mais memória aumenta a verossimilhança, mas não a direção "
        "(\\textit{Blocks}, $n=1$ mês, limiar $s/3$)",
        label="tab:mtd_ordem",
        spec="lrrrrr",
        size="\\small",
        header="Modelo & Parâm. & "
        + " & ".join(stack(a, b) for a, b in [("LL/símbolo", "(teste)"), ("Ganho", "s/ iid"), ("BIC", "(treino)"), ("Direção", "(teste)")])
        + " \\\\",
        body="\n".join(lines),
        legend="Fonte: Elaborado pelos autores (2026). LL em nats por símbolo; ganho sobre as "
        "frequências do treino (iid).",
    )


def table_summary(metrics):
    def spread(key, fmt):
        values = [m[key] for m in metrics]
        return f"{fmt(min(values))} & {fmt(median(values))} & {fmt(max(values))}"

    n = len(metrics)
    order_bic = sum(1 for m in metrics if m["_best_order"] == m["_max_order"])
    ci_above = sum(m["ci_above_constant"] for m in metrics)
    sig = sum(m["p_value"] < 0.05 for m in metrics)
    bonferroni = sum(m["p_value"] < 0.05 / n for m in metrics)
    over_first = sum(m["direction"] > m["direction_first"] for m in metrics)
    gap_first = [m["direction"] - m["direction_first"] for m in metrics]

    def f4(x):
        return num(x, 4)

    body = "\n".join(
        [
            f"    Ganho de LL sobre o iid (nats/símbolo) & {spread('gain_iid', f4)} \\\\",
            f"    Fração da entropia do iid explicada & {spread('entropy_share', pct)} \\\\",
            "    \\midrule",
            f"    Direção do MTD & {spread('direction', pct)} \\\\",
            f"    Direção da cadeia de 1ª ordem & {spread('direction_first', pct)} \\\\",
            f"    Direção da persistência & {spread('direction_persistence', pct)} \\\\",
            f"    Melhor constante (sempre $-1$ ou sempre $+1$) & {spread('constant', pct)} \\\\",
            f"    Margem do MTD sobre a melhor constante (p.p.) & {spread('margin', pp)} \\\\",
            f"    MTD menos cadeia de 1ª ordem (p.p.) & {num(100 * min(gap_first), 2, True)} "
            f"& {num(100 * median(gap_first), 2, True)} & {num(100 * max(gap_first), 2, True)} \\\\",
            "    \\midrule",
            f"    Ordem de menor BIC igual à máxima testada (10) & \\multicolumn{{3}}{{c}}{{{order_bic} de {n}}} \\\\",
            f"    IC 95\\% do MTD acima da melhor constante & \\multicolumn{{3}}{{c}}{{{ci_above} de {n}}} \\\\",
            f"    Significativo contra a melhor constante ($p<0{{,}}05$) & \\multicolumn{{3}}{{c}}{{{sig} de {n}}} \\\\",
            f"    Idem, com correção de Bonferroni ($p<0{{,}}05/{n}$) & \\multicolumn{{3}}{{c}}{{{bonferroni} de {n}}} \\\\",
            f"    MTD com direção acima da cadeia de 1ª ordem & \\multicolumn{{3}}{{c}}{{{over_first} de {n}}} \\\\",
        ]
    )
    return TABLE.format(
        caption=f"Resumo do MTD nas {n} configurações de discretização",
        label="tab:mtd_resumo",
        spec="lrrr",
        size="\\small",
        header="Medida & Mínimo & Mediana & Máximo \\\\",
        body=body,
        legend="Fonte: Elaborado pelos autores (2026). MTD da ordem de menor BIC em cada configuração. "
        "Teste unilateral (aproximação normal) do acerto de direção contra a melhor constante.",
    )


def table_volatility_direction(report):
    lines = []
    for mode, months, k in REPRESENTATIVE:
        m = config_metrics(find(report, mode, months, k))
        label = (
            f"\\textit{{Blocks}}, {months} {'mês' if months == 1 else 'meses'}, $s/{k}$"
            if mode == "blocks"
            else f"\\textit{{Global}}, $s/{k}$"
        )
        lines.append(
            f"    {label} & {pct(m['move_base'], 1)} & {pct(m['move_precision'], 1)} "
            f"& {pct(m['accuracy'], 1)} & {pct(m['majority'], 1)} "
            f"& {pct(m['direction'])} & {pct(m['direction_first'])} & {pct(m['constant'])} \\\\"
        )
    return TABLE.format(
        caption="Onde o MTD acerta: volatilidade e não direção",
        label="tab:mtd_volatilidade_direcao",
        spec="lrrrrrrr",
        size="\\footnotesize",
        header="Configuração & "
        + " & ".join(
            stack(a, b)
            for a, b in [
                ("Mov.", "real"),
                ("Prec.", "mov."),
                ("Acur.", "MTD"),
                ("Classe", "maj."),
                ("Dir.", "MTD"),
                ("Dir. 1ª", "ordem"),
                ("Melhor", "const."),
            ]
        )
        + " \\\\",
        body="\n".join(lines),
        legend="Fonte: Elaborado pelos autores (2026). Mov. real: fração dos \\textit{candles} de teste "
        "com símbolo $\\pm1$. Prec. mov.: fração das previsões de movimento do MTD que de fato foram "
        "movimento. Dir.: acerto de direção entre os \\textit{candles} que se moveram.",
    )


def table_stationarity(report):
    rows = [("global", None), ("blocks", 1), ("blocks", 2), ("blocks", 3), ("blocks", 6), ("blocks", 12)]
    lines = []
    for mode, months in rows:
        m = config_metrics(find(report, mode, months, 3))
        label = "\\textit{Global}" if mode == "global" else f"\\textit{{Blocks}}, {months} {'mês' if months == 1 else 'meses'}"
        lines.append(
            f"    {label} & {pct(m['zero_train'])} & {pct(m['zero_test'])} "
            f"& {pp(m['zero_test'] - m['zero_train'])} & {num(m['gain_iid'], 4)} \\\\"
        )
    return TABLE.format(
        caption="A volatilidade do teste difere da do treino (limiar $s/3$)",
        label="tab:mtd_estacionariedade",
        spec="lrrrr",
        size="\\small",
        header="Discretização & Zeros (treino) & Zeros (teste) & Variação (p.p.) & Ganho de LL s/ iid \\\\",
        body="\n".join(lines),
        legend="Fonte: Elaborado pelos autores (2026). Ganho de LL em nats por símbolo, do MTD da ordem de menor BIC.",
    )


def table_robustness(report):
    configs = report["configs"]
    restarts = sum(c["structure"][o]["restart_summary"]["restarts"] for c in configs for o in c["structure"])
    same = sum(c["structure"][o]["restart_summary"]["same_as_best"] for c in configs for o in c["structure"])
    converged = sum(c["structure"][o]["restart_summary"]["converged"] for c in configs for o in c["structure"])
    worst = max(c["structure"][o]["restart_summary"]["worst_gap"] for c in configs for o in c["structure"])
    gap = max(
        abs(g["ll_per_symbol_all"] - g["ll_per_symbol_clean"]) for c in configs for g in c["gaps"].values()
    )
    body = "\n".join(
        [
            f"    Reinícios do EM na mesma solução (±1 nat) & {num(same, 0)} de {num(restarts, 0)} \\\\",
            f"    Reinícios que convergiram & {num(converged, 0)} de {num(restarts, 0)} \\\\",
            f"    Maior diferença entre o melhor e o pior reinício & {num(worst, 3)} nats \\\\",
            f"    Efeito das transições que atravessam lacunas (LL/símbolo) & $\\leq$ {num(gap, 6)} nats \\\\",
        ]
    )
    return TABLE.format(
        caption="Verificações de que o resultado não vem de falha de ajuste",
        label="tab:mtd_robustez",
        spec="lr",
        size="\\small",
        header="Verificação & Resultado \\\\",
        body=body,
        legend=f"Fonte: Elaborado pelos autores (2026). {len(configs)} configurações, ordens 1 a 10, 10 reinícios por ordem.",
    )


TABLE = """\\begin{{table}}[htb]
  \\centering
  \\caption{{{caption}}}
  \\label{{{label}}}
  {size}
  \\begin{{tabular}}{{{spec}}}
    \\toprule
    {header}
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
  \\legend{{{legend}}}
\\end{{table}}
"""


## ---------------------------------------------------------------- geral


def all_metrics(report):
    out = []
    for cfg in report["configs"]:
        m = config_metrics(cfg)
        orders = [int(x[3:]) for x in (mm["name"] for mm in cfg["models"]) if x.startswith("mtd")]
        m["_best_order"] = int(best_mtd(cfg)["name"][3:])
        m["_max_order"] = max(orders)
        out.append(m)
    return out


def build(report):
    metrics = all_metrics(report)
    parts = [
        table_order(report),
        table_summary(metrics),
        table_volatility_direction(report),
        table_stationarity(report),
        table_robustness(report),
    ]
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = json.loads(args.json.read_text(encoding="utf-8"))
    args.out.write_text(build(report), encoding="utf-8")
    print(f"escrito: {args.out}")


if __name__ == "__main__":
    main()
