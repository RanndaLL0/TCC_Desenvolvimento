import math

import numpy as np
import pytest

from src.models.mtd.mtd import MTD
from src.models.naive_models.first_order import FirstOrder
from src.utils.bic import bic

TRUE_LAM = np.array([0.6, 0.3, 0.1])
TRUE_Q = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])


def simulate_mtd(lam, Q, size, seed):
    rng = np.random.default_rng(seed)
    order = len(lam)
    lags = rng.choice(order, size=size, p=lam)
    u = rng.random(size)
    cum = np.cumsum(Q, axis=1)
    x = np.empty(size, dtype=int)
    x[:order] = rng.integers(0, 3, size=order)
    for t in range(order, size):
        x[t] = min(int(np.searchsorted(cum[x[t - 1 - lags[t]]], u[t])), 2)
    return x - 1


# ---------------------------------------------------------------- fórmula


def test_bic_matches_hand_computation():
    assert bic(-100.0, 6, 1000) == pytest.approx(200 + 6 * math.log(1000))


def test_bic_zero_likelihood_is_pure_penalty():
    assert bic(0.0, 4, 100) == pytest.approx(4 * math.log(100))


def test_bic_single_observation_has_no_penalty():
    assert bic(-3.0, 5, 1) == pytest.approx(6.0)


def test_bic_penalty_grows_with_parameters_and_with_n():
    assert bic(-50.0, 7, 500) > bic(-50.0, 6, 500)
    assert bic(-50.0, 6, 5000) > bic(-50.0, 6, 500)


def test_bic_better_fit_gives_lower_bic():
    assert bic(-90.0, 6, 500) < bic(-100.0, 6, 500)


def test_bic_accepts_numpy_scalars():
    assert bic(np.float64(-10.0), np.int64(6), np.int32(100)) == pytest.approx(
        20 + 6 * math.log(100)
    )


def test_bic_minus_infinity_is_discarded():
    assert bic(-math.inf, 6, 100) == math.inf


@pytest.mark.parametrize("ll", [math.nan, math.inf])
def test_bic_rejects_invalid_log_likelihood(ll):
    with pytest.raises(ValueError):
        bic(ll, 6, 100)


@pytest.mark.parametrize("ll", ["-1", None, True])
def test_bic_rejects_non_numeric_log_likelihood(ll):
    with pytest.raises(TypeError):
        bic(ll, 6, 100)


@pytest.mark.parametrize("p, n", [(6.0, 100), (6, 100.0), ("6", 100), (6, True)])
def test_bic_rejects_non_integer_counts(p, n):
    with pytest.raises(TypeError):
        bic(-1.0, p, n)


@pytest.mark.parametrize("p, n", [(0, 100), (-1, 100), (6, 0), (6, -5)])
def test_bic_rejects_non_positive_counts(p, n):
    with pytest.raises(ValueError):
        bic(-1.0, p, n)


# ------------------------------------------------- integração com os modelos


def test_bic_of_first_order_toy():
    # n = N - 1 = 9 transições e p = 6 parâmetros (referência fixada no first_order)
    train = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])
    m = FirstOrder()
    m.fit(train)
    expected = -2 * (-7.4547) + 6 * math.log(9)
    assert bic(m.calculate_likelihood(train), 6, len(train) - 1) == pytest.approx(
        expected, abs=1e-3
    )


def test_bic_of_mtd_order_one_equals_first_order():
    sequence = np.random.default_rng(1).choice([-1, 0, 1], size=2000, p=(0.3, 0.4, 0.3))
    first = FirstOrder()
    first.fit(sequence)
    mtd = MTD(order=1, n_restarts=1)
    mtd.fit(sequence)
    assert bic(mtd.ll, mtd.n_params, mtd.n_obs) == pytest.approx(
        bic(first.calculate_likelihood(sequence), 6, len(sequence) - 1), abs=1e-3
    )


def aligned_bic(model, sequence, max_order):
    # todos os candidatos avaliados sobre as mesmas transições (t >= max_order):
    # descarta do começo o contexto que o candidato de menor ordem não precisa
    tail = sequence[max_order - model.order :]
    return bic(model.calculate_likelihood(tail), model.n_params, len(sequence) - max_order)


def test_bic_selects_true_order_among_mtd_candidates():
    sequence = simulate_mtd(TRUE_LAM, TRUE_Q, size=40000, seed=3)
    max_order = 5
    scores = {}
    for order in range(1, max_order + 1):
        m = MTD(order=order, n_restarts=2)
        m.fit(sequence)
        scores[order] = aligned_bic(m, sequence, max_order)

    assert min(scores, key=scores.get) == 3
