import numpy as np
import pytest

from src.models.naive_models.uniform import Uniform

# sequências do brinquedo usado na validação do first_order
TRAIN = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])
TEST = np.array([0, 1, -1, 0, 0, 1])


@pytest.fixture
def model():
    m = Uniform()
    m.fit(TRAIN)
    return m


def test_prob_matrix_is_uniform_and_rows_sum_to_one(model):
    assert model.prob.shape == (3, 3)
    assert np.allclose(model.prob, 1 / 3)
    assert np.allclose(model.prob.sum(axis=1), 1)


def test_predict_prob_has_one_row_per_symbol(model):
    result = model.predict_prob(TEST)
    assert result.shape == (len(TEST), 3)
    assert np.allclose(result, 1 / 3)
    assert np.allclose(result.sum(axis=1), 1)


def test_predict_prob_empty_sequence(model):
    assert model.predict_prob([]).shape == (0, 3)


def test_likelihood_is_n_minus_one_times_log_one_third(model):
    expected = (len(TEST) - 1) * np.log(1 / 3)
    assert model.calculate_likelihood(TEST) == pytest.approx(expected, abs=1e-12)


def test_likelihood_of_train_toy_sequence(model):
    # 9 pares: 9 * log(1/3) = -9.8875...
    assert model.calculate_likelihood(TRAIN) == pytest.approx(
        -9.887510598012987, abs=1e-9
    )


def test_likelihood_consistent_with_predict_prob(model):
    probs = model.predict_prob(TEST)[:-1]
    targets = TEST[1:] + 1
    from_probs = np.log(probs[np.arange(len(targets)), targets]).sum()
    assert model.calculate_likelihood(TEST) == pytest.approx(from_probs, abs=1e-12)


def test_methods_require_fit():
    m = Uniform()
    with pytest.raises(AttributeError):
        m.predict_prob(TEST)
    with pytest.raises(AttributeError):
        m.calculate_likelihood(TEST)


def test_fit_requires_at_least_two_symbols():
    with pytest.raises(ValueError):
        Uniform().fit([0])


def test_fit_does_not_set_state_when_it_fails():
    m = Uniform()
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
