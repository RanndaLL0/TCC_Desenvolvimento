from collections import Counter

import numpy as np
import pytest

from src.models.naive_models.first_order import FirstOrder
from src.utils.decision import predict

SYMBOLS = (-1, 0, 1)

# sequências do brinquedo usado na validação do first_order
TRAIN = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])
TEST = np.array([0, 1, -1, 0, 0, 1])

# contagens e probabilidades do treino acima (linhas e colunas na ordem -1, 0, +1)
TOY_COUNTS = [[0, 2, 0], [1, 1, 2], [1, 1, 1]]
TOY_PROB = [[0, 1, 0], [0.25, 0.25, 0.5], [1 / 3, 1 / 3, 1 / 3]]

SEEDS = [0, 1, 2]


def random_sequence(seed, size=1000, p=(1 / 3, 1 / 3, 1 / 3)):
    return np.random.default_rng(seed).choice(SYMBOLS, size=size, p=p)


def reference_probabilities(sequence):
    # referência em Python puro, sem numpy, para não repetir a lógica do modelo
    sequence = [int(s) for s in sequence]
    counts = Counter(zip(sequence[:-1], sequence[1:]))
    matrix = []
    for prev in SYMBOLS:
        total = sum(counts[(prev, nxt)] for nxt in SYMBOLS)
        matrix.append([counts[(prev, nxt)] / total for nxt in SYMBOLS])
    return matrix


@pytest.fixture
def model():
    m = FirstOrder()
    m.fit(TRAIN)
    return m


# ---------------------------------------------------------------- estado inicial


def test_initial_state():
    m = FirstOrder()
    assert m.prob is None
    assert m.matrix.shape == (3, 3)
    assert not m.matrix.any()


# ------------------------------------------------------------------------- fit


def test_fit_counts_toy_sequence(model):
    assert model.matrix.tolist() == TOY_COUNTS
    assert np.issubdtype(model.matrix.dtype, np.integer)


def test_fit_probabilities_toy_sequence(model):
    assert np.allclose(model.prob, TOY_PROB)


@pytest.mark.parametrize("seed", SEEDS)
def test_fit_counts_total_is_n_minus_one(seed):
    sequence = random_sequence(seed)
    m = FirstOrder()
    m.fit(sequence)
    assert m.matrix.sum() == len(sequence) - 1


@pytest.mark.parametrize("seed", SEEDS)
def test_fit_rows_sum_to_one_and_are_non_negative(seed):
    m = FirstOrder()
    m.fit(random_sequence(seed))
    assert np.allclose(m.prob.sum(axis=1), 1)
    assert np.all(m.prob >= 0)


@pytest.mark.parametrize("seed", SEEDS)
def test_fit_matches_reference_implementation(seed):
    sequence = random_sequence(seed, p=(0.2, 0.6, 0.2))
    m = FirstOrder()
    m.fit(sequence)
    assert np.allclose(m.prob, reference_probabilities(sequence))


@pytest.mark.parametrize(
    "data",
    [
        TRAIN.tolist(),
        TRAIN.astype(np.int8),
        TRAIN.astype(np.int32),
        TRAIN.astype(np.int64),
    ],
    ids=["list", "int8", "int32", "int64"],
)
def test_fit_accepts_lists_and_integer_dtypes(data):
    m = FirstOrder()
    m.fit(data)
    assert m.matrix.tolist() == TOY_COUNTS


def test_fit_does_not_modify_input():
    data = TRAIN.copy()
    FirstOrder().fit(data)
    assert np.array_equal(data, TRAIN)


def test_refit_replaces_previous_state(model):
    other = np.array([1, 1, 1, 0, 0, -1, -1, 1, 0, -1, 0, 1])
    model.fit(other)
    fresh = FirstOrder()
    fresh.fit(other)
    assert np.array_equal(model.matrix, fresh.matrix)
    assert np.allclose(model.prob, fresh.prob)


# o último símbolo da sequência nunca é contexto, então não conta como estado visto
@pytest.mark.parametrize(
    "data, missing",
    [
        ([0, 0, 1, 1], -1),
        ([-1, -1, 1, 1], 0),
        ([-1, 0, 0, -1], 1),
        ([-1, 0, 1], 1),
    ],
)
def test_fit_raises_when_a_state_has_no_transition(data, missing):
    with pytest.raises(ValueError, match=rf"\[\s*{missing}\s*\]"):
        FirstOrder().fit(data)


@pytest.mark.parametrize("data", [[], [0]], ids=["empty", "single"])
def test_fit_raises_on_sequence_too_short(data):
    with pytest.raises(ValueError):
        FirstOrder().fit(data)


@pytest.mark.xfail(
    strict=True,
    reason="pendência 2: self.matrix é atribuído antes da verificação; se o fit "
    "falha, matrix fica novo e prob antigo",
)
def test_failed_fit_keeps_previous_state_consistent(model):
    matrix_before = model.matrix.copy()
    prob_before = model.prob.copy()
    with pytest.raises(ValueError):
        model.fit([0, 0, 1, 1])
    assert np.array_equal(model.matrix, matrix_before)
    assert np.allclose(model.prob, prob_before)


@pytest.mark.xfail(
    strict=True,
    reason="o fit não valida os símbolos: -2 vira índice negativo e passa em "
    "silêncio, 2 e floats estouram IndexError sem mensagem clara",
)
@pytest.mark.parametrize(
    "data",
    [
        [-1, 0, 1, -2, 0, -1, 1],
        [-1, 0, 1, 2, 0, -1],
        np.array([-1.0, 0.0, 1.0, 0.0, -1.0, 1.0]),
    ],
    ids=["symbol_-2", "symbol_2", "float"],
)
def test_fit_rejects_invalid_symbols(data):
    with pytest.raises((ValueError, TypeError)):
        FirstOrder().fit(data)


# ----------------------------------------------------------------- verossimilhança


def test_log_likelihood_of_training_toy_sequence(model):
    assert model.calculate_likelihood(TRAIN) == pytest.approx(-7.4547, abs=1e-4)


@pytest.mark.xfail(
    strict=True,
    reason="_calculate_likelihood devolve nan: 0 * log(0) sem a máscara matrix > 0",
)
def test_counting_likelihood_matches_pairwise_likelihood(model):
    assert model._calculate_likelihood() == pytest.approx(
        model.calculate_likelihood(TRAIN)
    )


def test_log_likelihood_of_test_toy_sequence(model):
    # contextos 0,+1,-1,0,0 dão ao real as probabilidades .5, 1/3, 1, .25, .5
    expected = np.log(0.5) + np.log(1 / 3) + np.log(1) + np.log(0.25) + np.log(0.5)
    assert model.calculate_likelihood(TEST) == pytest.approx(expected)
    assert model.calculate_likelihood(TEST) / (len(TEST) - 1) == pytest.approx(
        -0.7742, abs=1e-4
    )


def test_log_likelihood_is_additive_over_overlapping_pieces(model):
    k = 3
    whole = model.calculate_likelihood(TRAIN)
    parts = model.calculate_likelihood(TRAIN[: k + 1]) + model.calculate_likelihood(
        TRAIN[k:]
    )
    assert whole == pytest.approx(parts)


@pytest.mark.parametrize("seed", SEEDS)
def test_log_likelihood_is_non_positive_and_beats_uniform_on_training(seed):
    sequence = random_sequence(seed, p=(0.2, 0.6, 0.2))
    m = FirstOrder()
    m.fit(sequence)
    ll = m.calculate_likelihood(sequence)
    uniform = (len(sequence) - 1) * np.log(1 / 3)
    assert ll <= 0
    # a máxima verossimilhança no treino nunca perde para a matriz uniforme
    assert ll >= uniform - 1e-9


def test_log_likelihood_is_minus_infinity_for_impossible_transition(model):
    # no treino, depois de -1 só aparece 0; -1 -> -1 tem probabilidade zero
    with pytest.warns(RuntimeWarning):
        assert model.calculate_likelihood([-1, -1]) == -np.inf


def test_log_likelihood_requires_fit():
    with pytest.raises(AttributeError):
        FirstOrder().calculate_likelihood([0, 1])


@pytest.mark.parametrize("data", [[], [0]], ids=["empty", "single"])
def test_log_likelihood_requires_two_symbols(model, data):
    with pytest.raises(ValueError, match="at least two"):
        model.calculate_likelihood(data)


def test_log_likelihood_rejects_invalid_symbols(model):
    with pytest.raises(ValueError, match="symbols must be in"):
        model.calculate_likelihood([0, 2])
    with pytest.raises(TypeError):
        model.calculate_likelihood([0.0, 1.0])


# ------------------------------------------------------------------ predict_proba


def test_predict_proba_requires_fit():
    with pytest.raises(AttributeError):
        FirstOrder().predict_proba([0])


def test_predict_proba_has_one_row_per_symbol(model):
    result = model.predict_proba(TEST)
    assert result.shape == (len(TEST), 3)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.allclose(result.sum(axis=1), 1)


def test_predict_proba_rows_follow_the_current_symbol(model):
    result = model.predict_proba([0, 1, -1])
    assert np.allclose(result[0], TOY_PROB[1])
    assert np.allclose(result[1], TOY_PROB[2])
    assert np.allclose(result[2], TOY_PROB[0])


def test_predict_proba_single_symbol(model):
    result = model.predict_proba([1])
    assert result.shape == (1, 3)
    assert np.allclose(result[0], TOY_PROB[2])


def test_predict_proba_same_symbol_gives_same_row(model):
    result = model.predict_proba([0, 1, 0, 0, -1, 0])
    for position in (2, 3, 5):
        assert np.allclose(result[position], result[0])


@pytest.mark.parametrize("seed", SEEDS)
def test_predict_proba_uses_only_the_past(seed):
    # prever tudo de uma vez deve coincidir com prever por prefixos
    sequence = random_sequence(seed, size=200)
    m = FirstOrder()
    m.fit(sequence)
    test = random_sequence(seed + 100, size=50)
    full = m.predict_proba(test)
    for n in range(1, len(test) + 1):
        assert np.allclose(m.predict_proba(test[:n])[-1], full[n - 1])


def test_predict_proba_last_row_is_the_live_prediction(model):
    window = np.array([-1, 0, 1])
    assert np.allclose(model.predict_proba(window)[-1], model.prob[1 + 1])


def test_predict_proba_returns_a_copy(model):
    result = model.predict_proba([0])
    result[0, 0] = 9.0
    assert model.prob[1, 0] == pytest.approx(0.25)


def test_predict_proba_rejects_invalid_input(model):
    with pytest.raises(ValueError, match="symbols must be in"):
        model.predict_proba([0, 3])
    with pytest.raises(ValueError, match="one-dimensional"):
        model.predict_proba([[0, 1]])
    with pytest.raises(TypeError):
        model.predict_proba([0.0, 1.0])


@pytest.mark.xfail(
    strict=True,
    reason="entrada vazia devolve shape (0,) em vez de (0, 3), contrato do "
    "predict_proba é matriz (N, 3)",
)
def test_predict_proba_empty_input_keeps_three_columns(model):
    assert model.predict_proba([]).shape == (0, 3)


# ------------------------------------------------------- predict (decisão) integrado


def test_predict_on_fitted_matrix_toy(model):
    # -1 -> 0; 0 -> +1; +1 -> empate entre os três, resolvido para -1
    assert predict(model.prob).tolist() == [0, 1, -1]


def test_predict_one_step_toy_accuracy_and_log_score(model):
    predictions = predict(model.predict_proba(TEST))
    assert predictions.tolist() == [1, -1, 0, 1, 1, -1]

    # modo A: previsão da posição i conferida contra o símbolo i+1
    hits = predictions[:-1] == TEST[1:]
    assert hits.tolist() == [True, True, True, False, True]
    assert hits.mean() == pytest.approx(0.8)


@pytest.mark.parametrize("seed", SEEDS)
def test_predict_returns_a_most_probable_symbol(seed):
    sequence = random_sequence(seed, p=(0.3, 0.4, 0.3))
    m = FirstOrder()
    m.fit(sequence)
    proba = m.predict_proba(sequence)
    predictions = predict(proba)
    assert predictions.shape == (len(sequence),)
    assert set(predictions.tolist()) <= set(SYMBOLS)
    chosen = proba[np.arange(len(sequence)), predictions + 1]
    assert np.allclose(chosen, proba.max(axis=1))


@pytest.mark.parametrize("seed", SEEDS)
def test_predict_collapses_to_zero_when_zero_dominates(seed):
    # com 0 em ~60% e símbolos independentes, o argmax prevê sempre 0;
    # é o comportamento esperado que motiva a regra de decisão sem o 0
    sequence = random_sequence(seed, size=5000, p=(0.2, 0.6, 0.2))
    m = FirstOrder()
    m.fit(sequence)
    predictions = predict(m.predict_proba(sequence))
    assert set(predictions.tolist()) == {0}
