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

ROOT = Path(__file__).resolve().parents[2]
STD_FOLDER = ROOT / "relatorios"

OFF_SET = np.timedelta64(5, "m")
MODES = ("blocks", "global")

# Posição sem símbolo: primeiro candle, retorno que atravessa uma lacuna e, no
# modo "blocks", o primeiro bloco (sem histórico). O valor está fora de
# {-1, 0, 1} de propósito: se vazar para um modelo, to_indices levanta erro em
# vez de contar um símbolo que não existe
NO_SYMBOL = np.iinfo(np.int8).min


@dataclass(frozen=True)
class Discretization:
    """Resultado de uma discretização da série inteira.

    Attributes:
        mode: "blocks" ou "global".
        k: divisor do limiar (o corte é sigma/k).
        train_pct: percentual de treino usado no corte.
        interval_months: meses por bloco (None no modo "global").
        cut: primeira linha do teste; as linhas [0, cut) são o treino.
        symbols: int8, uma posição por candle; NO_SYMBOL quando não há símbolo.
        segment: int32, id do trecho contínuo (sem lacunas) de cada símbolo;
            -1 onde não há símbolo.
        block: int32, bloco de cada candle (None no modo "global").
        stats: parâmetros usados (mu, sigma e, nos blocos, n) para o relatório.
    """

    mode: str
    k: float
    train_pct: int
    interval_months: int | None
    cut: int
    symbols: np.ndarray
    segment: np.ndarray
    block: np.ndarray | None
    stats: dict

    @property
    def valid(self):
        """Máscara booleana das posições que têm símbolo."""
        return self.symbols != NO_SYMBOL


def _validate_times(open_times):
    """Valida e devolve os open_time como array datetime64.

    Args:
        open_times: sequência 1D de datetime64.

    Returns:
        np.ndarray datetime64, estritamente crescente e com espaçamento múltiplo
        de 5 minutos.

    Raises:
        ValueError: se não for 1D, tiver duplicados/ordem errada ou espaçamento
            que não seja múltiplo de 5 minutos.
        TypeError: se o dtype não for datetime64.
    """
    times = np.asarray(open_times)

    if times.ndim != 1:
        raise ValueError("open_times must be one-dimensional")

    if not np.issubdtype(times.dtype, np.datetime64):
        raise TypeError(f"open_times must be datetime64, got dtype {times.dtype}")

    deltas = np.diff(times)

    if (deltas <= np.timedelta64(0, "m")).any():
        raise ValueError("open_times must be strictly increasing, without duplicates")

    if (deltas % OFF_SET != np.timedelta64(0, "m")).any():
        raise ValueError(
            "the spacing between open_times must be a multiple of 5 minutes"
        )

    return times


def find_gaps(open_times):
    """Encontra as lacunas: onde open_time[t] - open_time[t-1] > 5 minutos.

    Args:
        open_times: sequência 1D de datetime64 (validada por _validate_times).

    Returns:
        Tupla (positions, missing). positions tem a posição do primeiro candle
        depois de cada buraco; missing tem quantos candles faltam em cada um.
    """
    times = _validate_times(open_times)

    deltas = np.diff(times)
    before = np.flatnonzero(deltas > OFF_SET)

    return before + 1, (deltas[before] // OFF_SET) - 1


def log_returns(open_times, close):
    """Calcula o retorno logarítmico do fechamento entre candles adjacentes.

    O retorno que atravessa um buraco não é um retorno de 5 minutos, então
    fica NaN o primeiro candle e todo candle logo depois de uma lacuna.

    Args:
        open_times: sequência 1D de datetime64, um por candle.
        close: preços de fechamento, mesmo tamanho de open_times.

    Returns:
        np.ndarray float com um retorno por candle (NaN onde não há).

    Raises:
        ValueError: se os tamanhos diferirem ou close não for finito e positivo.
    """
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


def cut_index(n, train_pct):
    """Calcula a primeira linha do teste.

    Usa a mesma conta do split_data (data_manager.py), para o corte da
    discretização e o dos modelos serem o mesmo.

    Args:
        n: número de linhas da série.
        train_pct: percentual de treino, inteiro entre 1 e 99.

    Returns:
        int com o índice de corte; as linhas [0, cut) são o treino.

    Raises:
        TypeError: se train_pct não for int.
        ValueError: se train_pct estiver fora de 1..99 ou deixar treino ou
            teste vazio.
    """
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


def fit_stats(returns, context="returns"):
    """Estima a média e o desvio padrão amostral (ddof=1, como no TCC).

    Args:
        returns: retornos válidos (sem NaN).
        context: nome usado nas mensagens de erro.

    Returns:
        Tupla (mu, sigma) de floats.

    Raises:
        ValueError: se houver menos de 2 retornos ou o desvio for zero ou não
            finito.
    """
    values = np.asarray(returns, dtype=float)

    if values.size < 2:
        raise ValueError(f"{context}: at least 2 returns are needed, got {values.size}")

    mu = float(values.mean())
    sigma = float(values.std(ddof=1))

    if not np.isfinite(sigma) or sigma == 0:
        raise ValueError(f"{context}: the standard deviation is zero or not finite")

    return mu, sigma


def classify(returns, mu, sigma, k):
    """Discretiza retornos em {-1, 0, +1} pelo limiar sigma/k.

    -1 se (r - mu) < -sigma/k, +1 se (r - mu) > sigma/k e 0 caso contrário. Os
    pontos exatamente em ±sigma/k ficam em 0, como na equação do TCC. Note que
    k divide o desvio: k=3 corta em sigma/3, e não em 3 desvios.

    Args:
        returns: retornos logarítmicos (array 1D, sem NaN).
        mu: média usada para centrar os retornos.
        sigma: desvio padrão que define a escala do limiar.
        k: número positivo; o limiar é sigma/k.

    Returns:
        np.ndarray int8 com um símbolo (-1, 0 ou +1) por retorno.

    Raises:
        ValueError: se k não for um número positivo.
    """
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


def block_index(open_times, interval_months):
    """Atribui um bloco a cada candle.

    Os blocos são meses de calendário contados a partir do mês do primeiro
    candle e agrupados de `interval_months` em `interval_months`.

    Args:
        open_times: sequência 1D de datetime64, um por candle.
        interval_months: meses por bloco, inteiro >= 1.

    Returns:
        np.ndarray int32 com o número do bloco de cada candle (começa em 0).

    Raises:
        TypeError: se interval_months não for int.
        ValueError: se interval_months < 1 ou open_times estiver vazio.
    """
    if not isinstance(interval_months, int) or isinstance(interval_months, bool):
        raise TypeError("interval_months should be an int like 1")

    if interval_months < 1:
        raise ValueError("interval_months should be at least 1")

    times = _validate_times(open_times)

    if times.size == 0:
        raise ValueError("open_times is empty")

    months = times.astype("datetime64[M]").astype(int)

    return ((months - months[0]) // interval_months).astype(np.int32)


def discretize(open_times, close, k, mode="blocks", interval_months=1, train_pct=95):
    """Discretiza a série inteira de uma vez.

    Modo "blocks": o bloco b é classificado com a média e o desvio do bloco b-1
    (só dados anteriores, então nenhum símbolo depende do futuro). O primeiro
    bloco só serve de histórico e fica sem símbolos.

    Modo "global": uma única média e desvio, estimados só nas linhas de treino
    ([0, cut)), aplicados a treino e teste (a definição literal do TCC).

    Como o corte treino/teste é feito DEPOIS, sobre o resultado, o modo
    "blocks" produz os mesmos símbolos de treino com ou sem o teste.

    Args:
        open_times: sequência 1D de datetime64, um por candle.
        close: preços de fechamento, mesmo tamanho de open_times.
        k: divisor do limiar (o corte é sigma/k).
        mode: "blocks" ou "global".
        interval_months: meses por bloco (usado só no modo "blocks").
        train_pct: percentual de treino, inteiro entre 1 e 99.

    Returns:
        Discretization com os símbolos, o segmento de cada candle e os
        parâmetros usados.

    Raises:
        ValueError: se mode, k ou train_pct forem inválidos, ou se um mês do
            modo "blocks" não tiver candles para estimar o bloco seguinte.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    times = _validate_times(open_times)
    returns = log_returns(times, close)
    n = len(returns)
    cut = cut_index(n, train_pct)
    finite = np.isfinite(returns)

    # valida o k antes de qualquer trabalho
    classify(np.empty(0), 0.0, 1.0, k)

    symbols = np.full(n, NO_SYMBOL, dtype=np.int8)
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
    segment = np.where(symbols != NO_SYMBOL, np.cumsum(new_segment), -1).astype(
        np.int32
    )

    return Discretization(
        mode=mode,
        k=k,
        train_pct=train_pct,
        interval_months=interval_months if mode == "blocks" else None,
        cut=cut,
        symbols=symbols,
        segment=segment,
        block=block,
        stats=stats,
    )


def class_counts(symbols):
    """Conta os símbolos -1, 0 e +1, ignorando as posições sem símbolo.

    Args:
        symbols: array de símbolos, possivelmente com NO_SYMBOL.

    Returns:
        np.ndarray com 3 contagens, na ordem (-1, 0, +1).
    """
    values = np.asarray(symbols)
    values = values[values != NO_SYMBOL]

    return np.bincount(values.astype(int) + 1, minlength=3)


def _distribution(counts):
    """Monta a distribuição de classes a partir das contagens.

    Args:
        counts: contagens na ordem (-1, 0, +1), como em class_counts.

    Returns:
        dict com "total", a contagem de cada rótulo ("-1", "0", "1") e "pct"
        (percentual de cada um, None se o total for zero).
    """
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
    """Conta os símbolos de cada mês de calendário.

    Args:
        open_times: datetime64, um por candle.
        symbols: símbolos de cada candle, possivelmente com NO_SYMBOL.

    Returns:
        Lista de dicts, um por mês com símbolos: "month", "total", as contagens
        "-1", "0", "1" e "pct_zero".
    """
    valid = symbols != NO_SYMBOL
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
                "month": str(month),
                "total": total,
                "-1": int(row[0]),
                "0": int(row[1]),
                "1": int(row[2]),
                "pct_zero": round(100 * int(row[1]) / total, 2),
            }
        )

    return rows


def _by_year(by_month):
    """Calcula o percentual de símbolo 0 de cada ano.

    Args:
        by_month: lista devolvida por _by_month.

    Returns:
        dict ano -> percentual de zeros (float, 2 casas).
    """
    years = {}

    for row in by_month:
        year = years.setdefault(row["month"][:4], [0, 0])
        year[0] += row["total"]
        year[1] += row["0"]

    return {
        year: round(100 * zeros / total, 2) for year, (total, zeros) in years.items()
    }


def _timestamp(value):
    """Formata um datetime64 como "AAAA-MM-DD HH:MM:SS".

    Args:
        value: datetime64 escalar.

    Returns:
        str com a data e a hora, sem o "T" e com precisão de segundos.
    """
    return np.datetime_as_string(value, unit="s").replace("T", " ")


def _blocks_table(result, open_times):
    """Monta a tabela de estatísticas de cada bloco (modo "blocks").

    Args:
        result: Discretization do modo "blocks".
        open_times: datetime64, um por candle.

    Returns:
        Lista de dicts, um por bloco: "block", "first_month",
        "previous_block_returns", "mu" e "sigma" (None no primeiro bloco,
        que não tem histórico).
    """
    stats = result.stats
    first_month = open_times[0].astype("datetime64[M]")
    rows = []

    for b in range(len(stats["mu"])):
        has_stats = bool(np.isfinite(stats["mu"][b]))
        month = first_month + np.timedelta64(int(b * result.interval_months), "M")
        rows.append(
            {
                "block": b,
                "first_month": str(month),
                "previous_block_returns": int(stats["n"][b]),
                "mu": float(stats["mu"][b]) if has_stats else None,
                "sigma": float(stats["sigma"][b]) if has_stats else None,
            }
        )

    return rows


def build_report(open_times, results, source):
    """Monta o relatório de uma execução.

    Reúne registros, lacunas, corte, parâmetros e a distribuição de classes por
    limiar (geral, treino e teste) e por mês.

    Args:
        open_times: datetime64, um por candle.
        results: lista de Discretization da mesma série e do mesmo corte.
        source: dict da fonte dos dados, como devolvido por load_snapshot.

    Returns:
        dict pronto para virar JSON ou para render_markdown.
    """
    times = _validate_times(open_times)
    n = len(times)
    positions, missing = find_gaps(times)

    details = []
    for position, absent in zip(positions.tolist(), missing.tolist()):
        gap = times[position] - times[position - 1]
        details.append(
            {
                "from": _timestamp(times[position - 1]),
                "to": _timestamp(times[position]),
                "duration": str(timedelta(seconds=int(gap / np.timedelta64(1, "s")))),
                "missing_candles": int(absent),
            }
        )

    largest = (
        max(details, key=lambda gap: gap["missing_candles"]) if details else None
    )
    cut = results[0].cut

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "records": n,
        "first_open_time": _timestamp(times[0]),
        "last_open_time": _timestamp(times[-1]),
        # o primeiro candle e o candle depois de cada lacuna ficam sem retorno
        "valid_returns": n - 1 - len(details),
        "gaps": {
            "count": len(details),
            "missing_candles": int(missing.sum()),
            "segments": len(details) + 1,
            "largest": largest,
            "details": details,
        },
        "cut_info": {
            "train_pct": results[0].train_pct,
            "first_test_row": cut,
            "first_test_open_time": _timestamp(times[cut]),
        },
        "results": [],
    }

    for result in results:
        by_month = _by_month(times, result.symbols)
        entry = {
            "mode": result.mode,
            "k": result.k,
            "threshold": f"s/{result.k}",
            "interval_months": result.interval_months,
            "symbols_generated": int(result.valid.sum()),
            "segments_with_symbols": int(len(np.unique(result.segment[result.valid]))),
            "distribution": {
                "overall": _distribution(class_counts(result.symbols)),
                "train": _distribution(class_counts(result.symbols[:cut])),
                "test": _distribution(class_counts(result.symbols[cut:])),
            },
            "pct_zero_by_year": _by_year(by_month),
            "by_month": by_month,
        }

        if result.mode == "global":
            entry["parameters"] = {
                "mu": result.stats["mu"],
                "sigma": result.stats["sigma"],
            }
        else:
            entry["blocks"] = _blocks_table(result, times)

        report["results"].append(entry)

    return report


def _pct(value):
    """Formata um percentual no padrão brasileiro (decimal com vírgula).

    O relatório em Markdown usa o formato brasileiro; o JSON guarda os números
    puros.

    Args:
        value: percentual (float) ou None.

    Returns:
        str como "12,34%", ou "-" quando value é None.
    """
    return "-" if value is None else f"{value:.2f}%".replace(".", ",")


def render_markdown(report):
    """Renderiza o relatório em Markdown.

    Args:
        report: dict devolvido por build_report.

    Returns:
        str com o texto Markdown (milhar com ponto, decimal com vírgula).
    """
    gaps = report["gaps"]
    source = report["source"]
    cut_info = report["cut_info"]
    lines = [
        "# Relatório da discretização",
        "",
        f"Gerado em {report['generated_at']} (UTC). Decisões: `docs/discretizacao.md`.",
        "",
        "## Fonte",
        "",
        f"- Arquivo: `{source['file']}` (SHA-256 `{source['sha256']}`)",
        f"- Corte do snapshot: `open_time` anterior a {source['until_exclusive']}",
        f"- Candles: {report['records']:,} ({report['first_open_time']} a {report['last_open_time']})".replace(
            ",", "."
        ),
        f"- Retornos válidos (entre candles a 5 minutos): {report['valid_returns']:,}".replace(
            ",", "."
        ),
        "",
        "## Lacunas",
        "",
        f"- Lacunas (`open_time[t] - open_time[t-1]` diferente de 5 minutos): **{gaps['count']}**",
        f"- Candles faltantes: **{gaps['missing_candles']:,}**".replace(",", "."),
        f"- Segmentos contínuos: **{gaps['segments']}**",
        "- O retorno que atravessa uma lacuna não gera símbolo; o candle logo depois dela começa um novo segmento.",
    ]

    if gaps["largest"]:
        largest = gaps["largest"]
        lines.append(
            f"- Maior lacuna: {largest['duration']} ({largest['missing_candles']} candles), de {largest['from']} a {largest['to']}"
        )

    lines += [
        "",
        "As lacunas, uma a uma, estão no arquivo JSON do relatório.",
        "",
        "## Corte treino/teste",
        "",
        f"- Treino: {cut_info['train_pct']}% das linhas; o teste começa na linha {cut_info['first_test_row']:,} ({cut_info['first_test_open_time']})".replace(
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

    for entry in report["results"]:
        cells = []
        for part in ("overall", "train", "test"):
            pct = entry["distribution"][part]["pct"]
            cells.append(" / ".join(_pct(pct[label]) for label in ("-1", "0", "1")))

        lines.append(
            f"| {entry['mode']} | {entry['threshold']} | {entry['symbols_generated']:,} | ".replace(
                ",", "."
            )
            + " | ".join(cells)
            + " |"
        )

    years = sorted(
        {year for entry in report["results"] for year in entry["pct_zero_by_year"]}
    )
    labels = [f"{entry['mode']} {entry['threshold']}" for entry in report["results"]]

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
            _pct(entry["pct_zero_by_year"].get(year)) for entry in report["results"]
        ]
        lines.append(f"| {year} | " + " | ".join(row) + " |")

    lines += [
        "",
        "O detalhe por mês e as estatísticas de cada bloco estão no arquivo JSON do relatório.",
        "",
    ]

    return "\n".join(lines)


def load_snapshot(parquet):
    """Carrega open_time e close de um snapshot (database/snapshot.py).

    Confere o SHA-256 do manifesto antes de usar os dados.

    Args:
        parquet: caminho do arquivo Parquet do snapshot.

    Returns:
        Tupla (open_times, close, source): os dois arrays e um dict com o nome
        do arquivo, o SHA-256, o corte (until_exclusive) e o número de linhas.
    """
    from database.snapshot import verify_checksum

    manifest = verify_checksum(parquet)
    table = pq.read_table(parquet, columns=["open_time", "close"])

    open_times = table.column("open_time").to_numpy()
    close = table.column("close").to_numpy()

    source = {
        "file": Path(parquet).name,
        "sha256": manifest["sha256"],
        "until_exclusive": manifest["ate_exclusivo"],
        "rows": manifest["linhas"],
    }

    return open_times, close, source


def main():
    """Ponto de entrada: discretiza um snapshot e grava o relatório.

    Gera um relatório .json e um .md para cada combinação de modo e k pedida
    na linha de comando, e imprime os caminhos dos arquivos criados.
    """
    parser = argparse.ArgumentParser(
        description="Discretizes the close price of a snapshot and writes the report"
    )
    parser.add_argument(
        "--snapshot", required=True, type=Path, help="path to the Parquet file"
    )
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--k", nargs="+", type=int, default=[3, 4, 5], help="threshold s/k"
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="months per block (blocks mode)"
    )
    parser.add_argument("--train", type=int, default=95, help="train percentage")
    parser.add_argument("--output", type=Path, default=STD_FOLDER)
    args = parser.parse_args()

    open_times, close, source = load_snapshot(args.snapshot)

    results = [
        discretize(
            open_times,
            close,
            k,
            mode=mode,
            interval_months=args.interval,
            train_pct=args.train,
        )
        for mode in args.mode
        for k in args.k
    ]
    report = build_report(open_times, results, source)

    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / f"discretizacao_{source['until_exclusive']}"
    base.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    base.with_suffix(".md").write_text(render_markdown(report))

    print(f"{base.with_suffix('.md')}\n{base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
