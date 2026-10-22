import numpy as np
import pytest

from src.utils.data_manager import discretize_data
from src.utils.symbols import to_indices


def monthly(n_months, per_month=2):
    return np.array(
        [np.datetime64("2020-01") + np.timedelta64(m, "M") for m in range(n_months) for _ in range(per_month)]
    )


def test_first_block_is_only_history():
    ts = monthly(6)
    data = np.arange(len(ts), dtype=float)
    symbols, mask = discretize_data(data, ts, interval=3)

    assert mask.sum() == len(symbols) == 6  # 2 blocos de 3 meses, só o 2º sai
    assert not mask[:6].any()


def test_output_is_valid_for_to_indices():
    ts = monthly(9)
    rng = np.random.default_rng(0)
    symbols, _ = discretize_data(rng.normal(size=len(ts)), ts, interval=3)
    to_indices(symbols)  # não pode levantar


def test_no_leakage_future_does_not_change_past():
    ts = monthly(12)
    rng = np.random.default_rng(1)
    data = rng.normal(size=len(ts))
    full, _ = discretize_data(data, ts, interval=3)

    altered = data.copy()
    altered[-6:] += 1000  # mexe só nos 3 últimos meses
    other, _ = discretize_data(altered, ts, interval=3)

    # blocos 1 e 2 (meses 3-8) não podem depender do bloco 3
    assert full[:12].tolist() == other[:12].tolist()


def test_stats_are_recomputed_per_block():
    ts = monthly(9)
    # baixa volatilidade e depois alta: o mesmo valor +-1.0 é extremo no bloco 1
    # (histórico calmo) mas comum no bloco 2 (histórico já viu a volatilidade)
    data = np.concatenate([np.tile([-0.1, 0.1], 3), np.tile([-1.0, 1.0], 3), np.tile([-1.0, 1.0], 3)])
    symbols, _ = discretize_data(data, ts, interval=3, threshold=2)

    assert set(symbols[:6]) == {-1, 1}
    assert set(symbols[6:]) == {0}


def test_threshold_controls_amount_of_zeros():
    ts = monthly(6)
    data = np.random.default_rng(2).normal(size=len(ts))
    narrow, _ = discretize_data(data, ts, interval=3, threshold=0.1)
    wide, _ = discretize_data(data, ts, interval=3, threshold=10)

    assert (wide == 0).all()
    assert (narrow == 0).sum() <= (wide == 0).sum()


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_interval_raises(interval):
    with pytest.raises(ValueError):
        discretize_data([1.0], [np.datetime64("2020-01")], interval)


def test_non_int_interval_raises():
    with pytest.raises(TypeError):
        discretize_data([1.0], [np.datetime64("2020-01")], 3.0)


def test_unsorted_timestamps_raise():
    ts = np.array(["2020-02", "2020-01"], dtype="datetime64[M]")
    with pytest.raises(ValueError):
        discretize_data([1.0, 2.0], ts, 1)


def test_nan_raises():
    with pytest.raises(ValueError):
        discretize_data([1.0, np.nan], np.array(["2020-01", "2020-02"], dtype="datetime64[M]"), 1)


def test_empty_returns_empty():
    symbols, mask = discretize_data([], np.array([], dtype="datetime64[M]"), 3)
    assert len(symbols) == 0 and len(mask) == 0
