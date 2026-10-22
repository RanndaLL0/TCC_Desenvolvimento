import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from database import snapshot
from src.preprocess import discretizar as d
from src.utils.data_manager import split_data
from src.utils.symbols import to_indices

PASSO = np.timedelta64(5, "m")


def grid(start, n):
    return np.datetime64(start, "m") + np.arange(n) * PASSO


def prices_from_returns(returns):
    # o primeiro candle não tem retorno: entra com 0 e o preço inicial é 100
    shifted = np.asarray(returns, dtype=float).copy()
    shifted[0] = 0.0
    return 100.0 * np.exp(np.cumsum(shifted))


# 4 meses de calendário completos (jan a abr de 2020), a cada 5 minutos, com a
# volatilidade de cada mês escolhida pelo teste
def four_months(vols=(0.001, 0.01, 0.01, 0.01), seed=0):
    times = grid("2020-01-01T00:00", (31 + 29 + 31 + 30) * 288)
    month = times.astype("datetime64[M]").astype(int) - times[0].astype("datetime64[M]").astype(int)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, np.asarray(vols)[month])
    return times, returns, month


# ---------- retornos e lacunas ----------


def test_log_returns_between_adjacent_candles():
    times = grid("2020-01-01T00:00", 4)
    returns = d.log_returns(times, [100.0, 110.0, 121.0, 133.1])

    assert np.isnan(returns[0])
    assert returns[1:] == pytest.approx(np.log(1.1))


def test_log_returns_is_nan_after_a_gap():
    times = np.concatenate([grid("2020-01-01T00:00", 3), grid("2020-01-01T00:25", 2)])
    returns = d.log_returns(times, [100.0, 101.0, 102.0, 103.0, 104.0])

    # nan no primeiro candle e logo depois do buraco; o seguinte já é adjacente
    assert np.isnan(returns).tolist() == [True, False, False, True, False]


def test_find_gaps_counts_the_missing_candles():
    times = np.concatenate(
        [grid("2020-01-01T00:00", 3), grid("2020-01-01T00:25", 2), grid("2020-01-01T01:00", 1)]
    )
    positions, missing = d.find_gaps(times)

    assert positions.tolist() == [3, 5]
    assert missing.tolist() == [2, 5]


def test_regular_grid_has_no_gaps():
    positions, missing = d.find_gaps(grid("2020-01-01T00:00", 50))

    assert len(positions) == 0 and len(missing) == 0


def test_unsorted_or_duplicated_times_raise():
    with pytest.raises(ValueError, match="increasing"):
        d.find_gaps(grid("2020-01-01T00:00", 3)[::-1])

    with pytest.raises(ValueError, match="increasing"):
        d.find_gaps(np.array(["2020-01-01T00:00", "2020-01-01T00:00"], dtype="datetime64[m]"))


def test_spacing_must_be_a_multiple_of_five_minutes():
    times = np.array(["2020-01-01T00:00", "2020-01-01T00:07"], dtype="datetime64[m]")

    with pytest.raises(ValueError, match="multiple of 5 minutes"):
        d.find_gaps(times)


def test_times_must_be_datetime64():
    with pytest.raises(TypeError):
        d.find_gaps([1, 2, 3])


@pytest.mark.parametrize("close", [[100.0, 0.0, 101.0], [100.0, -1.0, 101.0], [100.0, np.nan, 101.0]])
def test_non_positive_or_nan_prices_raise(close):
    with pytest.raises(ValueError, match="positive"):
        d.log_returns(grid("2020-01-01T00:00", 3), close)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        d.log_returns(grid("2020-01-01T00:00", 3), [100.0, 101.0])


# ---------- corte, estatísticas e limiar ----------


@pytest.mark.parametrize("n, pct", [(100, 95), (1000, 80), (957_853, 95), (10, 50)])
def test_cut_index_matches_split_data(n, pct):
    # o corte da discretização tem que ser o mesmo dos modelos
    assert d.cut_index(n, pct) == len(split_data(np.arange(n), pct)[0])


@pytest.mark.parametrize("pct, error", [(0, ValueError), (100, ValueError), (95.0, TypeError), (True, TypeError)])
def test_cut_index_validates_the_percentage(pct, error):
    with pytest.raises(error):
        d.cut_index(100, pct)


def test_fit_stats_uses_the_sample_standard_deviation():
    mu, sigma = d.fit_stats([1.0, 2.0, 3.0, 4.0])

    assert mu == 2.5
    assert sigma == pytest.approx(np.sqrt(5 / 3))  # ddof=1 (com ddof=0 seria sqrt(1.25))


@pytest.mark.parametrize("values", [[], [1.0], [2.0, 2.0, 2.0]])
def test_fit_stats_rejects_too_few_or_constant_values(values):
    with pytest.raises(ValueError):
        d.fit_stats(values)


def test_classify_cuts_at_sigma_over_k_and_keeps_the_boundary_in_zero():
    # sigma=3 e k=3 dão corte exato em 1.0
    out = d.classify([1.0, -1.0, 1.0000001, -1.0000001, 0.0], mu=0.0, sigma=3.0, k=3)

    assert out.tolist() == [0, 0, 1, -1, 0]
    assert out.dtype == np.int8


def test_classify_centers_on_the_mean():
    # com mu=0.5 o ponto 1.5 está a 1.0 da média, exatamente no corte
    assert d.classify([1.5, 1.6, -0.5, -0.6], mu=0.5, sigma=3.0, k=3).tolist() == [0, 1, 0, -1]


def test_larger_k_gives_fewer_zeros():
    returns = np.random.default_rng(0).normal(0, 1, 5000)
    zeros = [(d.classify(returns, 0.0, 1.0, k) == 0).sum() for k in (3, 4, 5)]

    assert zeros[0] > zeros[1] > zeros[2]


@pytest.mark.parametrize("k", [0, -3, True, "3"])
def test_invalid_k_raises(k):
    with pytest.raises((ValueError, TypeError)):
        d.classify([0.0], 0.0, 1.0, k)


# ---------- blocos ----------


def test_block_index_groups_calendar_months_from_the_first_candle():
    times = grid("2020-01-15T00:00", 3)
    times = np.concatenate([times, grid("2020-02-03T00:00", 1), grid("2020-04-01T00:00", 1)])

    assert d.block_index(times, 1).tolist() == [0, 0, 0, 1, 3]
    assert d.block_index(times, 3).tolist() == [0, 0, 0, 0, 1]


def test_block_index_validates_the_interval():
    times = grid("2020-01-01T00:00", 3)

    with pytest.raises(ValueError):
        d.block_index(times, 0)

    with pytest.raises(TypeError):
        d.block_index(times, 1.0)


def test_first_block_has_no_symbols():
    times, returns, month = four_months()
    result = d.discretize(times, prices_from_returns(returns), k=3)

    assert (result.symbols[month == 0] == d.SEM_SIMBOLO).all()
    # do segundo bloco em diante todo retorno válido recebe símbolo
    assert (result.symbols[month >= 1] != d.SEM_SIMBOLO).all()


def test_each_block_uses_the_statistics_of_the_previous_block():
    times, returns, month = four_months()
    close = prices_from_returns(returns)
    result = d.discretize(times, close, k=3)
    recovered = d.log_returns(times, close)

    for b in (1, 2, 3):
        previous = recovered[month == b - 1]
        mu, sigma = d.fit_stats(previous[np.isfinite(previous)])
        current = month == b
        valid = current & np.isfinite(recovered)

        assert result.stats["mu"][b] == mu
        assert result.stats["sigma"][b] == sigma
        assert result.symbols[valid].tolist() == d.classify(recovered[valid], mu, sigma, 3).tolist()


def test_blocks_adapt_to_a_change_in_volatility():
    times, returns, month = four_months()
    result = d.discretize(times, prices_from_returns(returns), k=3)
    zeros = lambda b: (result.symbols[month == b] == 0).mean()

    # mês 1 (vol 10x maior) medido com a régua do mês 0: quase tudo vira ±1
    assert zeros(1) < 0.10
    # mês 2 já é medido com a régua do mês 1 (mesma vol): ~26% de zeros em s/3
    assert 0.20 < zeros(2) < 0.32


def test_no_leakage_altering_the_future_does_not_change_the_past():
    times, returns, month = four_months()
    close = prices_from_returns(returns)
    baseline = d.discretize(times, close, k=3)

    altered = returns.copy()
    altered[month == 3] *= 5  # mexe só no último mês (volatilidade 5x maior)
    other = d.discretize(times, prices_from_returns(altered), k=3)

    before_last = month < 3
    assert baseline.symbols[before_last].tolist() == other.symbols[before_last].tolist()
    # nenhuma estatística olha para o bloco seguinte, então o último bloco não influencia nenhuma
    for name in ("mu", "sigma"):
        assert np.array_equal(baseline.stats[name][:3], other.stats[name][:3], equal_nan=True)


def test_altering_a_block_only_changes_the_next_block_statistics():
    times, returns, month = four_months()
    baseline = d.discretize(times, prices_from_returns(returns), k=3)

    altered = returns.copy()
    altered[month == 1] *= 3
    other = d.discretize(times, prices_from_returns(altered), k=3)

    assert baseline.stats["sigma"][1] == other.stats["sigma"][1]  # vem do bloco 0
    assert baseline.stats["sigma"][2] != other.stats["sigma"][2]  # vem do bloco 1
    assert baseline.stats["sigma"][3] == other.stats["sigma"][3]  # vem do bloco 2


def test_train_symbols_do_not_depend_on_the_test_partition():
    # é o que permite discretizar a série inteira e só depois cortar em treino e teste
    times, returns, month = four_months()
    close = prices_from_returns(returns)
    full = d.discretize(times, close, k=4)

    train_only = d.discretize(times[: full.cut], close[: full.cut], k=4)

    assert full.symbols[: full.cut].tolist() == train_only.symbols.tolist()


def test_zero_sigma_in_the_previous_block_raises():
    times, returns, month = four_months()
    flat = returns.copy()
    flat[month == 0] = 0.0  # preço constante no bloco 0

    with pytest.raises(ValueError, match="block 0"):
        d.discretize(times, prices_from_returns(flat), k=3)


def test_empty_previous_block_raises():
    times = np.concatenate([grid("2020-01-01T00:00", 400), grid("2020-03-01T00:00", 400)])
    returns = np.random.default_rng(0).normal(0, 0.001, len(times))

    with pytest.raises(ValueError, match="block 1 has no candles"):
        d.discretize(times, prices_from_returns(returns), k=3)


def test_blocks_ignore_the_returns_that_cross_a_gap_when_estimating():
    times, returns, month = four_months()
    keep = np.ones(len(times), dtype=bool)
    keep[500:520] = False  # buraco de 20 candles no bloco 0
    result = d.discretize(times[keep], prices_from_returns(returns)[keep], k=3)

    recovered = d.log_returns(times[keep], prices_from_returns(returns)[keep])
    month_kept = month[keep]
    previous = recovered[month_kept == 0]
    mu, sigma = d.fit_stats(previous[np.isfinite(previous)])

    assert result.stats["n"][1] == np.isfinite(previous).sum()
    assert result.stats["sigma"][1] == sigma


# ---------- lacunas e segmentos nos símbolos ----------


def test_gaps_remove_the_symbol_and_start_a_new_segment():
    times, returns, month = four_months()
    keep = np.ones(len(times), dtype=bool)
    keep[9000:9010] = False  # buraco dentro do bloco 1 (jan = 0, fev = 1)
    close = prices_from_returns(returns)
    result = d.discretize(times[keep], close[keep], k=3)

    gap_position = int(d.find_gaps(times[keep])[0][0])
    assert result.symbols[gap_position] == d.SEM_SIMBOLO
    assert result.symbols[gap_position - 1] != d.SEM_SIMBOLO
    assert result.symbols[gap_position + 1] != d.SEM_SIMBOLO
    # o segmento muda entre o candle antes do buraco e o candle depois dele
    assert result.segment[gap_position + 1] == result.segment[gap_position - 1] + 1
    assert (result.segment[result.symbols == d.SEM_SIMBOLO] == -1).all()


def test_symbols_with_a_value_are_accepted_by_to_indices_and_the_marker_is_not():
    times, returns, month = four_months()
    result = d.discretize(times, prices_from_returns(returns), k=3)

    assert set(to_indices(result.symbols[result.valid]).tolist()) <= {0, 1, 2}
    # o marcador nunca passa despercebido: o modelo recusa a sequência com ele
    with pytest.raises(ValueError):
        to_indices(result.symbols)


# ---------- modo global ----------


def test_global_mode_estimates_only_on_the_train_partition():
    times, returns, month = four_months()
    close = prices_from_returns(returns)
    result = d.discretize(times, close, k=3, mode="global", train_pct=95)

    recovered = d.log_returns(times, close)
    train = recovered[: result.cut]
    mu, sigma = d.fit_stats(train[np.isfinite(train)])

    assert result.stats == {"mu": mu, "sigma": sigma}
    assert result.symbols[0] == d.SEM_SIMBOLO  # o primeiro candle não tem retorno
    assert (result.symbols[1:] != d.SEM_SIMBOLO).all()  # treino e teste recebem símbolo


def test_global_mode_ignores_the_test_partition_for_the_parameters():
    times, returns, month = four_months()
    baseline = d.discretize(times, prices_from_returns(returns), k=3, mode="global")

    altered = returns.copy()
    altered[baseline.cut :] *= 5  # só o teste (volatilidade 5x maior)
    other = d.discretize(times, prices_from_returns(altered), k=3, mode="global")

    assert baseline.stats == other.stats
    assert baseline.symbols[: baseline.cut].tolist() == other.symbols[: baseline.cut].tolist()

    altered_train = returns.copy()
    altered_train[: baseline.cut] *= 2  # mexer no treino muda a régua
    third = d.discretize(times, prices_from_returns(altered_train), k=3, mode="global")

    assert third.stats["sigma"] != baseline.stats["sigma"]


def test_global_mode_has_no_blocks_and_blocks_mode_has_them():
    times, returns, month = four_months()
    close = prices_from_returns(returns)

    assert d.discretize(times, close, k=3, mode="global").block is None
    assert d.discretize(times, close, k=3, mode="blocos").block is not None


def test_invalid_mode_raises():
    times, returns, month = four_months()

    with pytest.raises(ValueError, match="mode"):
        d.discretize(times, prices_from_returns(returns), k=3, mode="outro")


# ---------- relatório ----------


def report_for(keep=None):
    times, returns, month = four_months()
    close = prices_from_returns(returns)
    if keep is not None:
        times, close = times[keep], close[keep]

    results = [d.discretize(times, close, k, mode=mode) for mode in d.MODOS for k in (3, 4, 5)]
    source = {"arquivo": "x.parquet", "sha256": "abc", "ate_exclusivo": "2020-05-01", "linhas": len(times)}
    return d.build_report(times, results, source), times


def test_report_counts_records_gaps_and_classes():
    keep = np.ones((31 + 29 + 31 + 30) * 288, dtype=bool)
    keep[3000:3004] = False  # 4 candles faltando
    keep[20000:20010] = False  # 10 candles faltando
    report, times = report_for(keep)

    assert report["registros"] == len(times)
    assert report["lacunas"]["quantidade"] == 2
    assert report["lacunas"]["candles_faltantes"] == 14
    assert report["lacunas"]["segmentos"] == 3
    assert report["lacunas"]["maior"]["candles_faltantes"] == 10
    assert report["retornos_validos"] == len(times) - 1 - 2
    assert len(report["resultados"]) == 6  # 2 modos x 3 limiares

    for entry in report["resultados"]:
        geral = entry["distribuicao"]["geral"]
        assert geral["total"] == entry["simbolos_gerados"]
        assert geral["-1"] + geral["0"] + geral["1"] == geral["total"]
        # treino e teste somam o geral
        parts = entry["distribuicao"]
        assert parts["treino"]["total"] + parts["teste"]["total"] == geral["total"]
        assert sum(row["total"] for row in entry["por_mes"]) == geral["total"]


def test_report_is_json_serializable_and_markdown_has_the_key_sections():
    report, _ = report_for()

    json.dumps(report)  # não pode levantar (NaN de bloco sem histórico vira None)
    text = d.render_markdown(report)

    for title in ("# Relatório da discretização", "## Lacunas", "## Distribuição de classes", "## Percentual de símbolo 0 por ano"):
        assert title in text
    assert "s/3" in text and "s/5" in text


def test_blocks_report_lists_the_statistics_of_each_block():
    report, _ = report_for()
    blocks_entry = next(e for e in report["resultados"] if e["modo"] == "blocos")

    assert [b["bloco"] for b in blocks_entry["blocos"]] == [0, 1, 2, 3]
    assert blocks_entry["blocos"][0]["sigma"] is None  # o primeiro bloco não tem histórico
    assert blocks_entry["blocos"][1]["sigma"] is not None


# ---------- carregador do snapshot ----------


def write_tiny_snapshot(tmp_path, n=20):
    rows = []
    for i in range(n):
        t = datetime(2020, 1, 1) + timedelta(minutes=5 * i)
        rows.append((t, 1.0, 1.0, 1.0, 100.0 + i, 1.0, t, 1.0, 1, 1.0, 1.0, "BTCUSDT", "5m"))

    parquet = tmp_path / "btc_usdt_2020-02-01.parquet"
    snapshot.write_parquet([rows], parquet)
    manifest = {
        "tabela": "btc_usdt",
        "ate_exclusivo": "2020-02-01",
        "linhas": n,
        "sha256": snapshot.sha256_arquivo(parquet),
    }
    snapshot.caminho_manifesto(parquet).write_text(json.dumps(manifest))
    return parquet


def test_load_snapshot_returns_times_prices_and_source(tmp_path):
    parquet = write_tiny_snapshot(tmp_path)
    open_times, close, source = d.load_snapshot(parquet)

    assert open_times[0] == np.datetime64("2020-01-01T00:00") and len(open_times) == 20
    assert close.tolist() == [100.0 + i for i in range(20)]
    assert source["ate_exclusivo"] == "2020-02-01" and source["arquivo"] == parquet.name
    assert source["sha256"] == snapshot.sha256_arquivo(parquet)


def test_load_snapshot_rejects_a_corrupted_file(tmp_path):
    parquet = write_tiny_snapshot(tmp_path)
    parquet.write_bytes(parquet.read_bytes() + b"x")

    with pytest.raises(ValueError, match="Checksum"):
        d.load_snapshot(parquet)
