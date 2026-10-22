import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Discretização dos retornos logarítmicos do fechamento em {-1, 0, +1}.
# As decisões e o porquê de cada uma estão em docs/discretizacao.md.
#
# Nada aqui conhece a fonte dos dados: as funções recebem arrays (open_time e
# close). O snapshot é só um carregador (load_snapshot), então uma fonte "ao
# vivo" poderá reaproveitar o mesmo núcleo.

RAIZ = Path(__file__).resolve().parents[2]
PASTA_PADRAO = RAIZ / "relatorios"

PASSO = np.timedelta64(5, "m")
MODOS = ("blocos", "global")

# Posição sem símbolo: primeiro candle, retorno que atravessa uma lacuna e, no
# modo "blocos", o primeiro bloco (sem histórico). O valor está fora de
# {-1, 0, 1} de propósito: se vazar para um modelo, to_indices levanta erro em
# vez de contar um símbolo que não existe
SEM_SIMBOLO = np.iinfo(np.int8).min


@dataclass(frozen=True)
class Discretization:
    mode: str
    k: float
    train_pct: int
    interval_months: int | None
    cut: int  # primeira linha do teste (as linhas [0, cut) são o treino)
    symbols: np.ndarray  # int8, uma posição por candle; SEM_SIMBOLO quando não há
    segment: (
        np.ndarray
    )  # int32, trecho contínuo (sem lacunas) de cada símbolo; -1 sem símbolo
    block: np.ndarray | None  # int32, bloco de cada candle (só no modo "blocos")
    stats: dict  # parâmetros usados (mu, sigma) para o relatório

    @property
    def valid(self):
        return self.symbols != SEM_SIMBOLO


## Valida e devolve os open_time: estritamente crescentes e múltiplos de 5 minutos
def _validate_times(open_times):
    times = np.asarray(open_times)

    if times.ndim != 1:
        raise ValueError("open_times must be one-dimensional")

    if not np.issubdtype(times.dtype, np.datetime64):
        raise TypeError(f"open_times must be datetime64, got dtype {times.dtype}")

    deltas = np.diff(times)

    if (deltas <= np.timedelta64(0, "m")).any():
        raise ValueError("open_times must be strictly increasing, without duplicates")

    if (deltas % PASSO != np.timedelta64(0, "m")).any():
        raise ValueError(
            "the spacing between open_times must be a multiple of 5 minutes"
        )

    return times


## Lacunas: posições onde o candle anterior NÃO está a 5 minutos (open_time[t] -
# open_time[t-1] > 5 min). Devolve a posição do primeiro candle depois do buraco
# e quantos candles faltam em cada um
def find_gaps(open_times):
    times = _validate_times(open_times)

    deltas = np.diff(times)
    before = np.flatnonzero(deltas > PASSO)

    return before + 1, (deltas[before] // PASSO) - 1


## Retorno logarítmico do fechamento entre candles ADJACENTES (5 minutos).
# Fica NaN no primeiro candle e em todo candle logo depois de uma lacuna: o
# retorno que atravessa um buraco não é um retorno de 5 minutos
def log_returns(open_times, close):
    times = _validate_times(open_times)
    prices = np.asarray(close, dtype=float)

    if prices.shape != times.shape:
        raise ValueError("open_times and close must have the same length")

    if not (np.isfinite(prices).all() and (prices > 0).all()):
        raise ValueError("close must be finite and positive")

    returns = np.full(len(prices), np.nan)
    returns[1:] = np.log(prices[1:] / prices[:-1])
    returns[find_gaps(times)[0]] = np.nan

    return returns


## Primeira linha do teste. Mesma conta do split_data (data_manager.py), para o
# corte da discretização e o dos modelos serem o mesmo
def cut_index(n, train_pct):
    if not isinstance(train_pct, int) or isinstance(train_pct, bool):
        raise TypeError("train_pct should be an int like 95")

    if train_pct < 1 or train_pct > 99:
        raise ValueError("train_pct should be an int between 1 and 99")

    cut = (n * train_pct) // 100

    if not (1 < cut < n):
        raise ValueError(
            "train_pct and the series length leave the train or test empty"
        )

    return cut


## Média e desvio padrão AMOSTRAL (ddof=1, como no TCC) dos retornos válidos
def fit_stats(returns, context="returns"):
    values = np.asarray(returns, dtype=float)

    if values.size < 2:
        raise ValueError(f"{context}: at least 2 returns are needed, got {values.size}")

    mu = float(values.mean())
    sigma = float(values.std(ddof=1))

    if not np.isfinite(sigma) or sigma == 0:
        raise ValueError(f"{context}: the standard deviation is zero or not finite")

    return mu, sigma


## Limiar s/k: -1 se (r - mu) < -s/k, +1 se (r - mu) > s/k e 0 caso contrário
# (os pontos exatamente em ±s/k ficam em 0, como na equação do TCC). Note que k
# divide o desvio: k=3 corta em s/3, e não em 3 desvios
def classify(returns, mu, sigma, k):
    if (
        isinstance(k, bool)
        or not isinstance(k, (int, float, np.integer, np.floating))
        or k <= 0
    ):
        raise ValueError("k must be a positive number (the threshold is sigma/k)")

    deviation = (
        np.asarray(returns, dtype=float) - mu
    )  # desvio padrão em relação a média
    cut = sigma / k

    return np.where(deviation < -cut, -1, np.where(deviation > cut, 1, 0)).astype(
        np.int8
    )


## Bloco de cada candle: meses de calendário contados a partir do mês do
# primeiro candle e agrupados de `interval_months` em `interval_months`
def block_index(open_times, interval_months):
    if not isinstance(interval_months, int) or isinstance(interval_months, bool):
        raise TypeError("interval_months should be an int like 1")

    if interval_months < 1:
        raise ValueError("interval_months should be at least 1")

    times = _validate_times(open_times)

    if times.size == 0:
        raise ValueError("open_times is empty")

    months = times.astype("datetime64[M]").astype(int)

    return ((months - months[0]) // interval_months).astype(np.int32)


## Discretiza a série inteira de uma vez.
#
# modo "blocos": o bloco b é classificado com a média e o desvio do bloco b-1
#   (só dados anteriores, então nenhum símbolo depende do futuro). O primeiro
#   bloco só serve de histórico e fica sem símbolos.
# modo "global": uma única média e desvio, estimados só nas linhas de treino
#   ([0, cut)), aplicados a treino e teste (a definição literal do TCC).
#
# Como o corte treino/teste é feito DEPOIS, sobre o resultado, o modo "blocos"
# produz os mesmos símbolos de treino com ou sem o teste
def discretize(open_times, close, k, mode="blocos", interval_months=1, train_pct=95):
    if mode not in MODOS:
        raise ValueError(f"mode must be one of {MODOS}, got {mode!r}")

    times = _validate_times(open_times)
    returns = log_returns(times, close)
    n = len(returns)
    cut = cut_index(n, train_pct)
    finite = np.isfinite(returns)

    # valida o k antes de qualquer trabalho
    classify(np.empty(0), 0.0, 1.0, k)

    symbols = np.full(n, SEM_SIMBOLO, dtype=np.int8)
    block = None

    if mode == "global":
        train = returns[:cut]
        mu, sigma = fit_stats(train[np.isfinite(train)], "train partition")
        symbols[finite] = classify(returns[finite], mu, sigma, k)
        stats = {"mu": mu, "sigma": sigma}
    else:
        block = block_index(times, interval_months)
        present, starts = np.unique(block, return_index=True)
        ends = np.append(starts[1:], n)
        # mapeia: bloco -> (start, end)
        bounds = dict(zip(present.tolist(), zip(starts.tolist(), ends.tolist())))

        size = int(present[-1]) + 1
        mu = np.full(size, np.nan)
        sigma = np.full(size, np.nan)
        used = np.zeros(size, dtype=int)

        # usa as estatísticas dos blocos anteriores para discretiozar o bloco atual
        for b in present[1:].tolist():
            if b - 1 not in bounds:
                raise ValueError(
                    f"block {b - 1} has no candles, so the statistics of block {b} cannot be estimated"
                )

            start, end = bounds[b - 1]
            previous = returns[start:end]
            previous = previous[np.isfinite(previous)]
            mu[b], sigma[b] = fit_stats(previous, f"block {b - 1}")
            used[b] = previous.size

            start, end = bounds[b]
            current = returns[start:end]
            ok = np.isfinite(current)
            symbols[start:end][ok] = classify(current[ok], mu[b], sigma[b], k)

        stats = {"mu": mu, "sigma": sigma, "n": used}

    gap_positions, _ = find_gaps(times)
    new_segment = np.zeros(n, dtype=np.int32)
    new_segment[gap_positions] = 1
    segment = np.where(symbols != SEM_SIMBOLO, np.cumsum(new_segment), -1).astype(
        np.int32
    )

    return Discretization(
        mode=mode,
        k=k,
        train_pct=train_pct,
        interval_months=interval_months if mode == "blocos" else None,
        cut=cut,
        symbols=symbols,
        segment=segment,
        block=block,
        stats=stats,
    )


## Contagem dos símbolos -1, 0 e +1 (ignora as posições sem símbolo)
def class_counts(symbols):
    values = np.asarray(symbols)
    values = values[values != SEM_SIMBOLO]

    return np.bincount(values.astype(int) + 1, minlength=3)


def _distribution(counts):
    total = int(counts.sum())
    labels = ("-1", "0", "1")

    return {
        "total": total,
        **{label: int(counts[i]) for i, label in enumerate(labels)},
        "pct": {
            label: round(100 * int(counts[i]) / total, 2) if total else None
            for i, label in enumerate(labels)
        },
    }


def _by_month(open_times, symbols):
    valid = symbols != SEM_SIMBOLO
    months, inverse = np.unique(
        open_times[valid].astype("datetime64[M]"), return_inverse=True
    )
    codes = inverse * 3 + (symbols[valid].astype(int) + 1)
    counts = np.bincount(codes, minlength=len(months) * 3).reshape(-1, 3)

    rows = []
    for month, row in zip(months, counts):
        total = int(row.sum())
        rows.append(
            {
                "mes": str(month),
                "total": total,
                "-1": int(row[0]),
                "0": int(row[1]),
                "1": int(row[2]),
                "pct_zero": round(100 * int(row[1]) / total, 2),
            }
        )

    return rows


def _by_year(by_month):
    years = {}

    for row in by_month:
        year = years.setdefault(row["mes"][:4], [0, 0])
        year[0] += row["total"]
        year[1] += row["0"]

    return {
        year: round(100 * zeros / total, 2) for year, (total, zeros) in years.items()
    }


def _timestamp(value):
    return np.datetime_as_string(value, unit="s").replace("T", " ")


def _blocks_table(result, open_times):
    stats = result.stats
    first_month = open_times[0].astype("datetime64[M]")
    rows = []

    for b in range(len(stats["mu"])):
        has_stats = bool(np.isfinite(stats["mu"][b]))
        month = first_month + int(b * result.interval_months)
        rows.append(
            {
                "bloco": b,
                "primeiro_mes": str(month),
                "retornos_do_bloco_anterior": int(stats["n"][b]),
                "mu": float(stats["mu"][b]) if has_stats else None,
                "sigma": float(stats["sigma"][b]) if has_stats else None,
            }
        )

    return rows


## Relatório de uma execução: registros, lacunas, corte, parâmetros e a
# distribuição de classes por limiar (geral, treino e teste) e por mês
def build_report(open_times, results, source):
    times = _validate_times(open_times)
    n = len(times)
    positions, missing = find_gaps(times)

    details = []
    for position, absent in zip(positions.tolist(), missing.tolist()):
        gap = times[position] - times[position - 1]
        details.append(
            {
                "de": _timestamp(times[position - 1]),
                "ate": _timestamp(times[position]),
                "duracao": str(timedelta(seconds=int(gap / np.timedelta64(1, "s")))),
                "candles_faltantes": int(absent),
            }
        )

    largest = (
        max(details, key=lambda gap: gap["candles_faltantes"]) if details else None
    )
    cut = results[0].cut

    report = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonte": source,
        "registros": n,
        "primeiro_open_time": _timestamp(times[0]),
        "ultimo_open_time": _timestamp(times[-1]),
        # o primeiro candle e o candle depois de cada lacuna ficam sem retorno
        "retornos_validos": n - 1 - len(details),
        "lacunas": {
            "quantidade": len(details),
            "candles_faltantes": int(missing.sum()),
            "segmentos": len(details) + 1,
            "maior": largest,
            "detalhe": details,
        },
        "corte": {
            "treino_pct": results[0].train_pct,
            "primeira_linha_do_teste": cut,
            "open_time_da_primeira_linha_do_teste": _timestamp(times[cut]),
        },
        "resultados": [],
    }

    for result in results:
        by_month = _by_month(times, result.symbols)
        entry = {
            "modo": result.mode,
            "k": result.k,
            "limiar": f"s/{result.k}",
            "intervalo_meses": result.interval_months,
            "simbolos_gerados": int(result.valid.sum()),
            "segmentos_com_simbolos": int(len(np.unique(result.segment[result.valid]))),
            "distribuicao": {
                "geral": _distribution(class_counts(result.symbols)),
                "treino": _distribution(class_counts(result.symbols[:cut])),
                "teste": _distribution(class_counts(result.symbols[cut:])),
            },
            "pct_zero_por_ano": _by_year(by_month),
            "por_mes": by_month,
        }

        if result.mode == "global":
            entry["parametros"] = {
                "mu": result.stats["mu"],
                "sigma": result.stats["sigma"],
            }
        else:
            entry["blocos"] = _blocks_table(result, times)

        report["resultados"].append(entry)

    return report


# o relatório em Markdown usa o formato brasileiro (milhar com ponto, decimal com
# vírgula); o JSON guarda os números puros
def _pct(value):
    return "-" if value is None else f"{value:.2f}%".replace(".", ",")


def render_markdown(report):
    gaps = report["lacunas"]
    fonte = report["fonte"]
    corte = report["corte"]
    lines = [
        "# Relatório da discretização",
        "",
        f"Gerado em {report['gerado_em']} (UTC). Decisões: `docs/discretizacao.md`.",
        "",
        "## Fonte",
        "",
        f"- Arquivo: `{fonte['arquivo']}` (SHA-256 `{fonte['sha256']}`)",
        f"- Corte do snapshot: `open_time` anterior a {fonte['ate_exclusivo']}",
        f"- Candles: {report['registros']:,} ({report['primeiro_open_time']} a {report['ultimo_open_time']})".replace(
            ",", "."
        ),
        f"- Retornos válidos (entre candles a 5 minutos): {report['retornos_validos']:,}".replace(
            ",", "."
        ),
        "",
        "## Lacunas",
        "",
        f"- Lacunas (`open_time[t] - open_time[t-1]` diferente de 5 minutos): **{gaps['quantidade']}**",
        f"- Candles faltantes: **{gaps['candles_faltantes']:,}**".replace(",", "."),
        f"- Segmentos contínuos: **{gaps['segmentos']}**",
        "- O retorno que atravessa uma lacuna não gera símbolo; o candle logo depois dela começa um novo segmento.",
    ]

    if gaps["maior"]:
        maior = gaps["maior"]
        lines.append(
            f"- Maior lacuna: {maior['duracao']} ({maior['candles_faltantes']} candles), de {maior['de']} a {maior['ate']}"
        )

    lines += [
        "",
        "As lacunas, uma a uma, estão no arquivo JSON do relatório.",
        "",
        "## Corte treino/teste",
        "",
        f"- Treino: {corte['treino_pct']}% das linhas; o teste começa na linha {corte['primeira_linha_do_teste']:,} ({corte['open_time_da_primeira_linha_do_teste']})".replace(
            ",", "."
        ),
        "",
        "## Distribuição de classes",
        "",
        "Percentuais de -1 / 0 / +1.",
        "",
        "| Modo | Limiar | Símbolos | Geral | Treino | Teste |",
        "|---|---|---:|---|---|---|",
    ]

    for entry in report["resultados"]:
        cells = []
        for part in ("geral", "treino", "teste"):
            pct = entry["distribuicao"][part]["pct"]
            cells.append(" / ".join(_pct(pct[label]) for label in ("-1", "0", "1")))

        lines.append(
            f"| {entry['modo']} | {entry['limiar']} | {entry['simbolos_gerados']:,} | ".replace(
                ",", "."
            )
            + " | ".join(cells)
            + " |"
        )

    years = sorted(
        {year for entry in report["resultados"] for year in entry["pct_zero_por_ano"]}
    )
    labels = [f"{entry['modo']} {entry['limiar']}" for entry in report["resultados"]]

    lines += [
        "",
        "## Percentual de símbolo 0 por ano",
        "",
        "A volatilidade do retorno muda ao longo dos anos; é o que compara um limiar único (global) com o limiar por bloco.",
        "",
        "| Ano | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]

    for year in years:
        row = [
            _pct(entry["pct_zero_por_ano"].get(year)) for entry in report["resultados"]
        ]
        lines.append(f"| {year} | " + " | ".join(row) + " |")

    lines += [
        "",
        "O detalhe por mês e as estatísticas de cada bloco estão no arquivo JSON do relatório.",
        "",
    ]

    return "\n".join(lines)


## Carrega open_time e close de um snapshot (database/snapshot.py), conferindo o
# SHA-256 do manifesto antes de usar os dados
def load_snapshot(parquet):
    from database.snapshot import verify_checksum

    manifest = verify_checksum(parquet)
    table = pq.read_table(parquet, columns=["open_time", "close"])

    open_times = table.column("open_time").to_numpy()
    close = table.column("close").to_numpy()

    source = {
        "arquivo": Path(parquet).name,
        "sha256": manifest["sha256"],
        "ate_exclusivo": manifest["ate_exclusivo"],
        "linhas": manifest["linhas"],
    }

    return open_times, close, source


def main():
    parser = argparse.ArgumentParser(
        description="Discretiza o fechamento de um snapshot e grava o relatório"
    )
    parser.add_argument(
        "--snapshot", required=True, type=Path, help="caminho do Parquet"
    )
    parser.add_argument("--modo", nargs="+", choices=MODOS, default=list(MODOS))
    parser.add_argument(
        "--k", nargs="+", type=int, default=[3, 4, 5], help="limiar s/k"
    )
    parser.add_argument(
        "--intervalo", type=int, default=1, help="meses por bloco (modo blocos)"
    )
    parser.add_argument("--treino", type=int, default=95, help="percentual de treino")
    parser.add_argument("--saida", type=Path, default=PASTA_PADRAO)
    args = parser.parse_args()

    open_times, close, source = load_snapshot(args.snapshot)

    results = [
        discretize(
            open_times,
            close,
            k,
            mode=mode,
            interval_months=args.intervalo,
            train_pct=args.treino,
        )
        for mode in args.modo
        for k in args.k
    ]
    report = build_report(open_times, results, source)

    args.saida.mkdir(parents=True, exist_ok=True)
    base = args.saida / f"discretizacao_{source['ate_exclusivo']}"
    base.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    base.with_suffix(".md").write_text(render_markdown(report))

    print(f"{base.with_suffix('.md')}\n{base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
