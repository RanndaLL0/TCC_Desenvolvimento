import math
import warnings

import numpy as np
import pytest

from src.models.mtd.mtd import MTD, em, initial_guess
from src.models.naive_models.first_order import FirstOrder
from src.utils.decision import predict

SYMBOLS = (-1, 0, 1)
SEEDS = [0, 1, 2]

# sequência do brinquedo usado nos outros modelos
TRAIN = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])

# parâmetros verdadeiros para simular um MTD de ordem 3 com Q bem informativa
TRUE_LAM = np.array([0.6, 0.3, 0.1])
TRUE_Q = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])


def random_sequence(seed, size=1000, p=(1 / 3, 1 / 3, 1 / 3)):
    return np.random.default_rng(seed).choice(SYMBOLS, size=size, p=p)


def simulate_mtd(lam, Q, size, seed):
    # gera o MTD de verdade: sorteia o lag que "manda" e depois o símbolo a
    # partir da linha de Q desse lag (índices 0..2, devolve símbolos -1..+1)
    rng = np.random.default_rng(seed)
    order = len(lam)
    lags = rng.choice(order, size=size, p=lam)
    u = rng.random(size)
    cum = np.cumsum(Q, axis=1)
    x = np.empty(size, dtype=int)
    x[:order] = rng.integers(0, 3, size=order)
    for t in range(order, size):
        row = cum[x[t - 1 - lags[t]]]
        x[t] = min(int(np.searchsorted(row, u[t])), 2)
    return x - 1


# ---- referências em Python puro, sem numpy, para não repetir a lógica do modelo


def reference_probability(lam, Q, context, target):
    # context[g] é o símbolo no lag g+1
    return sum(lam[g] * Q[context[g]][target] for g in range(len(lam)))


def reference_log_likelihood(lam, Q, indices):
    order = len(lam)
    total = 0.0
    for t in range(order, len(indices)):
        context = [indices[t - 1 - g] for g in range(order)]
        total += math.log(reference_probability(lam, Q, context, indices[t]))
    return total


def reference_em(indices, lam, Q, iterations, floor=1e-10):
    lam = [float(v) for v in lam]
    Q = [[float(v) for v in row] for row in Q]
    order = len(lam)
    n_obs = len(indices) - order

    for _ in range(iterations):
        counts = [[0.0] * 3 for _ in range(3)]
        lam_num = [0.0] * order
        for t in range(order, len(indices)):
            j = indices[t]
            weights = [lam[g] * Q[indices[t - 1 - g]][j] for g in range(order)]
            total = sum(weights)
            for g in range(order):
                r = weights[g] / total
                counts[indices[t - 1 - g]][j] += r
                lam_num[g] += r
        lam = [v / n_obs for v in lam_num]
        Q = [
            [(counts[i][j] + floor) / (sum(counts[i]) + 3 * floor) for j in range(3)]
            for i in range(3)
        ]

    return lam, Q


@pytest.fixture(scope="module")
def simulated():
    return simulate_mtd(TRUE_LAM, TRUE_Q, size=60000, seed=7)


@pytest.fixture(scope="module")
def fitted_order_3(simulated):
    m = MTD(order=3, n_restarts=3)
    m.fit(simulated)
    return m


# ---------------------------------------------------------------- construtor


def test_initial_state():
    m = MTD(order=2)
    assert m.lam is None and m.Q is None and m.ll is None and m.info is None


@pytest.mark.parametrize("order", [0, -1])
def test_constructor_rejects_non_positive_order(order):
    with pytest.raises(ValueError):
        MTD(order=order)


@pytest.mark.parametrize("order", [1.0, "2", True, None])
def test_constructor_rejects_non_integer_order(order):
    with pytest.raises(TypeError):
        MTD(order=order)


def test_constructor_rejects_bad_floor_and_restarts():
    with pytest.raises(ValueError):
        MTD(order=2, floor=0)
    with pytest.raises(ValueError):
        MTD(order=2, n_restarts=0)


@pytest.mark.parametrize("order, expected", [(1, 6), (2, 7), (3, 8), (6, 11)])
def test_n_params_follows_the_bic_formula(order, expected):
    # m(m-1) + (l-1), com m = 3
    assert MTD(order=order).n_params == expected


# ------------------------------------------------------- ordem 1 = primeira ordem


@pytest.mark.parametrize("seed", SEEDS)
def test_order_one_matches_first_order(seed):
    sequence = random_sequence(seed, size=3000, p=(0.3, 0.4, 0.3))

    first = FirstOrder()
    first.fit(sequence)

    m = MTD(order=1, n_restarts=2)
    m.fit(sequence)

    assert np.allclose(m.lam, [1.0])
    assert np.allclose(m.Q, first.prob, atol=1e-6)
    assert m.calculate_likelihood(sequence) == pytest.approx(
        first.calculate_likelihood(sequence), abs=1e-4
    )
    assert np.allclose(m.predict_proba(sequence), first.predict_proba(sequence))


# ------------------------------------------------------------------------ EM


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("iterations", [1, 3, 10])
def test_em_matches_pure_python_reference(order, iterations):
    sequence = random_sequence(11, size=400, p=(0.3, 0.4, 0.3))
    indices = (sequence + 1).tolist()

    rng = np.random.default_rng(5)
    lam0 = rng.dirichlet(np.ones(order))
    Q0 = rng.dirichlet(np.ones(3), size=3)

    lam, Q, _, _ = em(
        np.array(indices), lam0, Q0, order, max_iter=iterations, tolerance=0.0
    )
    ref_lam, ref_Q = reference_em(indices, lam0, Q0, iterations)

    assert np.allclose(lam, ref_lam)
    assert np.allclose(Q, ref_Q)


@pytest.mark.parametrize("seed", SEEDS)
def test_log_likelihood_never_decreases(seed):
    sequence = simulate_mtd(TRUE_LAM, TRUE_Q, size=5000, seed=seed)
    indices = sequence + 1

    lam0, Q0 = initial_guess(indices, 3, 1, np.random.SeedSequence(seed), 1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # o aviso de queda viraria erro
        _, _, _, info = em(indices, lam0, Q0, 3, max_iter=50)

    history = info["history"]
    assert len(history) > 2
    assert np.all(np.diff(history) >= -1e-9)
    assert not info["decreased"]


def test_em_ll_matches_calculate_likelihood(fitted_order_3, simulated):
    assert fitted_order_3.ll == pytest.approx(
        fitted_order_3.calculate_likelihood(simulated), abs=1e-6
    )


def test_em_ll_matches_reference_log_likelihood(fitted_order_3, simulated):
    expected = reference_log_likelihood(
        fitted_order_3.lam.tolist(), fitted_order_3.Q.tolist(), (simulated + 1).tolist()
    )
    assert fitted_order_3.ll == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("n_blocks", [1, 2, 7])
def test_result_does_not_depend_on_number_of_blocks(n_blocks):
    sequence = random_sequence(3, size=2000, p=(0.3, 0.4, 0.3))
    base = MTD(order=3, n_restarts=1, n_blocks=1)
    base.fit(sequence)
    other = MTD(order=3, n_restarts=1, n_blocks=n_blocks)
    other.fit(sequence)

    assert np.allclose(base.lam, other.lam)
    assert np.allclose(base.Q, other.Q)
    assert base.ll == pytest.approx(other.ll)


def test_more_blocks_than_transitions_still_works():
    sequence = random_sequence(4, size=12)
    m = MTD(order=2, n_restarts=1, n_blocks=64, max_iter=5)
    m.fit(sequence)
    assert m.ll == pytest.approx(m.calculate_likelihood(sequence), abs=1e-9)


# --------------------------------------------------------- recuperação de parâmetros


def test_fit_recovers_true_parameters(fitted_order_3):
    assert np.allclose(fitted_order_3.lam, TRUE_LAM, atol=0.05)
    assert np.allclose(fitted_order_3.Q, TRUE_Q, atol=0.04)


def test_fit_beats_first_order_on_a_true_mtd(fitted_order_3, simulated):
    first = FirstOrder()
    first.fit(simulated)
    # compara nas mesmas transições (t >= 3), então a LL da cadeia de ordem 1
    # precisa ser recalculada a partir de t = 3
    first_ll = first.calculate_likelihood(simulated[2:])
    assert fitted_order_3.ll > first_ll


def test_irrelevant_lags_get_small_weight():
    # ordem verdadeira 1 ajustada com ordem 3: os lags extras devem ficar perto de 0
    sequence = simulate_mtd(np.array([1.0]), TRUE_Q, size=40000, seed=9)
    m = MTD(order=3, n_restarts=3)
    m.fit(sequence)
    assert m.lam[0] > 0.9
    assert np.allclose(m.Q, TRUE_Q, atol=0.04)


# -------------------------------------------------------- invariantes do ajuste


@pytest.mark.parametrize("order", [1, 2, 4])
def test_parameters_are_valid_distributions(order):
    m = MTD(order=order, n_restarts=2)
    m.fit(random_sequence(1, size=2000, p=(0.2, 0.6, 0.2)))

    assert m.lam.shape == (order,)
    assert m.Q.shape == (3, 3)
    assert m.lam.sum() == pytest.approx(1.0)
    assert np.all(m.lam >= 0)
    assert np.allclose(m.Q.sum(axis=1), 1)
    assert np.all(m.Q > 0)


def test_fit_is_deterministic_for_the_same_seed():
    sequence = random_sequence(2, size=1500)
    a = MTD(order=3, n_restarts=4, seed=5)
    a.fit(sequence)
    b = MTD(order=3, n_restarts=4, seed=5)
    b.fit(sequence)
    assert np.array_equal(a.lam, b.lam)
    assert np.array_equal(a.Q, b.Q)


def test_best_restart_is_the_one_with_highest_ll():
    m = MTD(order=3, n_restarts=5)
    m.fit(random_sequence(6, size=3000, p=(0.3, 0.4, 0.3)))
    lls = [r["ll"] for r in m.info["restarts"]]
    assert len(lls) == 5
    assert m.ll == max(lls)
    assert m.info["restarts"][m.info["restart"]]["ll"] == m.ll


def test_fit_does_not_modify_input():
    data = TRAIN.copy()
    MTD(order=2, n_restarts=1).fit(data)
    assert np.array_equal(data, TRAIN)


def test_fit_accepts_lists_and_integer_dtypes():
    sequence = random_sequence(8, size=500)
    reference = MTD(order=2, n_restarts=1)
    reference.fit(sequence)
    for data in (sequence.tolist(), sequence.astype(np.int8), sequence.astype(np.int32)):
        m = MTD(order=2, n_restarts=1)
        m.fit(data)
        assert np.allclose(m.Q, reference.Q)


def test_refit_replaces_previous_state():
    a = random_sequence(1, size=800)
    b = random_sequence(2, size=800, p=(0.2, 0.6, 0.2))
    m = MTD(order=2, n_restarts=1)
    m.fit(a)
    m.fit(b)
    fresh = MTD(order=2, n_restarts=1)
    fresh.fit(b)
    assert np.allclose(m.Q, fresh.Q)
    assert np.allclose(m.lam, fresh.lam)


def test_n_obs_is_n_minus_order():
    m = MTD(order=4, n_restarts=1)
    m.fit(random_sequence(1, size=500))
    assert m.n_obs == 496


def test_symbol_never_seen_as_lag_does_not_break_fit():
    # só aparecem 0 e +1: a linha do -1 de Q fica só com o piso (uniforme)
    sequence = random_sequence(1, size=500, p=(0.0, 0.5, 0.5))
    m = MTD(order=2, n_restarts=1)
    m.fit(sequence)
    assert np.allclose(m.Q.sum(axis=1), 1)
    assert np.allclose(m.Q[0], 1 / 3)
    assert np.isfinite(m.ll)


@pytest.mark.parametrize("data", [[], [0], [0, 1], [0, 1, -1]])
def test_fit_raises_when_sequence_is_not_longer_than_order(data):
    with pytest.raises(ValueError, match="more than 3"):
        MTD(order=3).fit(data)


def test_failed_fit_keeps_previous_state_consistent():
    m = MTD(order=2, n_restarts=1)
    m.fit(random_sequence(1, size=500))
    lam, Q, ll = m.lam.copy(), m.Q.copy(), m.ll
    with pytest.raises(ValueError):
        m.fit([0, 1])
    assert np.array_equal(m.lam, lam)
    assert np.array_equal(m.Q, Q)
    assert m.ll == ll


@pytest.mark.parametrize(
    "data",
    [[-1, 0, 1, -2, 0, -1, 1], [-1, 0, 1, 2, 0, -1], np.array([-1.0, 0.0, 1.0, 0.0, -1.0])],
    ids=["symbol_-2", "symbol_2", "float"],
)
def test_fit_rejects_invalid_symbols(data):
    with pytest.raises((ValueError, TypeError)):
        MTD(order=2).fit(data)


# ----------------------------------------------------------- verossimilhança


def test_log_likelihood_on_new_sequence_matches_reference(fitted_order_3):
    other = random_sequence(21, size=300, p=(0.3, 0.4, 0.3))
    expected = reference_log_likelihood(
        fitted_order_3.lam.tolist(), fitted_order_3.Q.tolist(), (other + 1).tolist()
    )
    assert fitted_order_3.calculate_likelihood(other) == pytest.approx(expected)


def test_log_likelihood_is_non_positive_and_beats_uniform_on_training(fitted_order_3, simulated):
    ll = fitted_order_3.calculate_likelihood(simulated)
    assert ll <= 0
    assert ll > (len(simulated) - 3) * math.log(1 / 3)


def test_log_likelihood_is_additive_over_overlapping_pieces(fitted_order_3):
    # trechos que se sobrepõem em `order` símbolos de contexto somam o todo
    sequence = random_sequence(5, size=60)
    k = 30
    order = fitted_order_3.order
    whole = fitted_order_3.calculate_likelihood(sequence)
    parts = fitted_order_3.calculate_likelihood(
        sequence[: k + 1]
    ) + fitted_order_3.calculate_likelihood(sequence[k + 1 - order :])
    assert whole == pytest.approx(parts)


def test_log_likelihood_requires_fit():
    with pytest.raises(AttributeError):
        MTD(order=2).calculate_likelihood([0, 1, 0])


@pytest.mark.parametrize("data", [[], [0], [0, 1], [0, 1, -1]])
def test_log_likelihood_requires_more_symbols_than_order(fitted_order_3, data):
    with pytest.raises(ValueError, match="more than 3"):
        fitted_order_3.calculate_likelihood(data)


def test_log_likelihood_rejects_invalid_symbols(fitted_order_3):
    with pytest.raises(ValueError, match="symbols must be in"):
        fitted_order_3.calculate_likelihood([0, 1, 0, 2, 0])
    with pytest.raises(TypeError):
        fitted_order_3.calculate_likelihood([0.0, 1.0, 0.0, 1.0, 0.0])


# ------------------------------------------------------------- predict_proba


def test_predict_proba_requires_fit():
    with pytest.raises(AttributeError):
        MTD(order=2).predict_proba([0])


def test_predict_proba_shape_and_rows_sum_to_one(fitted_order_3):
    window = random_sequence(1, size=40)
    result = fitted_order_3.predict_proba(window)
    assert result.shape == (40, 3)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.allclose(result.sum(axis=1), 1)
    assert np.all(result >= 0)


def test_predict_proba_full_context_row_matches_reference(fitted_order_3):
    window = np.array([-1, 0, 1, 1, 0])
    indices = (window + 1).tolist()
    row = fitted_order_3.predict_proba(window)[-1]
    # lags 1, 2, 3 do último símbolo: window[-1], window[-2], window[-3]
    context = [indices[-1], indices[-2], indices[-3]]
    expected = [
        reference_probability(
            fitted_order_3.lam.tolist(), fitted_order_3.Q.tolist(), context, j
        )
        for j in range(3)
    ]
    assert np.allclose(row, expected)


def test_predict_proba_short_context_uses_renormalised_lags(fitted_order_3):
    lam, Q = fitted_order_3.lam, fitted_order_3.Q
    window = np.array([1, -1, 0, 0])
    result = fitted_order_3.predict_proba(window)

    # linha 0: só o lag 1; linha 1: lags 1 e 2 renormalizados
    assert np.allclose(result[0], Q[2])
    expected_1 = (lam[0] * Q[0] + lam[1] * Q[2]) / (lam[0] + lam[1])
    assert np.allclose(result[1], expected_1)


@pytest.mark.parametrize("seed", SEEDS)
def test_predict_proba_uses_only_the_past(fitted_order_3, seed):
    test = random_sequence(seed + 100, size=30)
    full = fitted_order_3.predict_proba(test)
    for n in range(1, len(test) + 1):
        assert np.allclose(fitted_order_3.predict_proba(test[:n])[-1], full[n - 1])


def test_predict_proba_is_consistent_with_log_likelihood(fitted_order_3):
    sequence = random_sequence(12, size=200, p=(0.3, 0.4, 0.3))
    proba = fitted_order_3.predict_proba(sequence)
    order = fitted_order_3.order
    targets = sequence[order:] + 1
    total = np.log(proba[order - 1 : -1][np.arange(len(targets)), targets]).sum()
    assert total == pytest.approx(fitted_order_3.calculate_likelihood(sequence))


def test_predict_proba_with_all_early_lags_at_zero_weight_is_uniform():
    m = MTD(order=2, n_restarts=1)
    m.fit(random_sequence(1, size=300))
    m.lam = np.array([0.0, 1.0])
    result = m.predict_proba([1, 0, -1])
    assert np.allclose(result[0], 1 / 3)
    assert np.allclose(result.sum(axis=1), 1)


def test_predict_proba_shorter_than_order_still_works(fitted_order_3):
    result = fitted_order_3.predict_proba([0])
    assert result.shape == (1, 3)
    assert np.allclose(result[0], fitted_order_3.Q[1])


def test_predict_proba_returns_a_copy(fitted_order_3):
    before = fitted_order_3.Q.copy()
    result = fitted_order_3.predict_proba([0])
    result[0, 0] = 9.0
    assert np.array_equal(fitted_order_3.Q, before)


def test_predict_proba_rejects_invalid_input(fitted_order_3):
    with pytest.raises(ValueError, match="symbols must be in"):
        fitted_order_3.predict_proba([0, 3])
    with pytest.raises(ValueError, match="one-dimensional"):
        fitted_order_3.predict_proba([[0, 1]])
    with pytest.raises(TypeError):
        fitted_order_3.predict_proba([0.0, 1.0])


def test_predict_proba_rejects_empty_input(fitted_order_3):
    with pytest.raises(ValueError, match="at least one element"):
        fitted_order_3.predict_proba([])


# ------------------------------------------------------ integração com a decisão


def test_predict_on_mtd_probabilities_returns_valid_symbols(fitted_order_3):
    proba = fitted_order_3.predict_proba(random_sequence(1, size=100))
    predictions = predict(proba)
    assert predictions.shape == (100,)
    assert set(predictions.tolist()) <= set(SYMBOLS)
