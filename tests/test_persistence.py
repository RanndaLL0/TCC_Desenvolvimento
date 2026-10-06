import numpy as np
import pytest

from src.models.naive_models.persistence import Persistence

# sequências do brinquedo usado na validação do first_order
TRAIN = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])
TEST = np.array([0, 1, -1, 0, 0, 1])
EPS = 0.05


@pytest.fixture
def model():
    m = Persistence(epsilon=EPS)
    m.fit(TRAIN)
    return m


def test_prob_matrix_structure(model):
    expected = np.full((3, 3), EPS)
    np.fill_diagonal(expected, 1 - 2 * EPS)
    assert np.allclose(model.prob, expected)
    assert np.allclose(model.prob.sum(axis=1), 1)
    assert np.all(model.prob > 0)


def test_predict_prob_rows_follow_current_symbol(model):
    result = model.predict_prob([0, 1, -1])
    assert result.shape == (3, 3)
    # colunas na ordem -1, 0, +1; o símbolo atual concentra 1 - 2*eps
    assert np.allclose(result[0], [EPS, 1 - 2 * EPS, EPS])
    assert np.allclose(result[1], [EPS, EPS, 1 - 2 * EPS])
    assert np.allclose(result[2], [1 - 2 * EPS, EPS, EPS])


def test_predict_prob_has_one_row_per_symbol(model):
    result = model.predict_prob(TEST)
    assert result.shape == (len(TEST), 3)
    assert np.allclose(result.sum(axis=1), 1)


def test_argmax_of_each_row_is_the_current_symbol(model):
    predicted = model.predict_prob(TEST).argmax(axis=1) - 1
    assert predicted.tolist() == TEST.tolist()


def test_likelihood_by_hand_on_test_toy(model):
    # pares: 0->+1 erra, +1->-1 erra, -1->0 erra, 0->0 acerta, 0->+1 erra
    expected = 4 * np.log(EPS) + np.log(1 - 2 * EPS)
    assert model.calculate_likelihood(TEST) == pytest.approx(expected, abs=1e-12)


def test_likelihood_by_hand_on_train_toy(model):
    # pares: -1->0, 0->0, 0->+1, +1->-1, -1->0, 0->+1, +1->+1, +1->0, 0->-1
    # acertos (iguais): 0->0 e +1->+1; os outros 7 erram
    expected = 2 * np.log(1 - 2 * EPS) + 7 * np.log(EPS)
    assert model.calculate_likelihood(TRAIN) == pytest.approx(expected, abs=1e-12)


def test_likelihood_is_finite_even_when_always_wrong(model):
    assert np.isfinite(model.calculate_likelihood([0, 1, -1, 0, 1]))


def test_likelihood_consistent_with_predict_prob(model):
    probs = model.predict_prob(TEST)[:-1]
    targets = TEST[1:] + 1
    from_probs = np.log(probs[np.arange(len(targets)), targets]).sum()
    assert model.calculate_likelihood(TEST) == pytest.approx(from_probs, abs=1e-12)


def test_default_epsilon_is_valid():
    m = Persistence()
    m.fit(TRAIN)
    assert np.allclose(m.prob.sum(axis=1), 1)
    assert 0 < m.epsilon < 1 / 3


@pytest.mark.parametrize("epsilon", [0, -0.1, 1 / 3, 0.5, 1])
def test_epsilon_out_of_range_raises(epsilon):
    with pytest.raises(ValueError):
        Persistence(epsilon=epsilon)


@pytest.mark.parametrize("epsilon", ["0.1", None, True])
def test_epsilon_wrong_type_raises(epsilon):
    with pytest.raises(TypeError):
        Persistence(epsilon=epsilon)


def test_methods_require_fit():
    m = Persistence()
    with pytest.raises(AttributeError):
        m.predict_prob(TEST)
    with pytest.raises(AttributeError):
        m.calculate_likelihood(TEST)


def test_fit_requires_at_least_two_symbols():
    with pytest.raises(ValueError):
        Persistence().fit([1])


def test_fit_does_not_set_state_when_it_fails():
    m = Persistence()
    with pytest.raises(ValueError):
        m.fit([0, 7])
    assert m.prob is None


def test_likelihood_requires_at_least_two_symbols(model):
    with pytest.raises(ValueError):
        model.calculate_likelihood([0])


@pytest.mark.parametrize("invalid", [[0, 2], [0, -2], [0.0, 1.0]])
def test_invalid_symbols_are_rejected(model, invalid):
    with pytest.raises((ValueError, TypeError)):
        model.predict_prob(invalid)
    with pytest.raises((ValueError, TypeError)):
        model.calculate_likelihood(invalid)
