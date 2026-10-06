import numpy as np
import pytest

from src.utils.symbols import to_indices


def test_converts_symbols_to_indices():
    assert to_indices([-1, 0, 1, 0]).tolist() == [0, 1, 2, 1]


def test_accepts_numpy_integer_arrays():
    assert to_indices(np.array([1, -1], dtype=np.int8)).tolist() == [2, 0]


def test_empty_sequence_returns_empty_indices():
    assert len(to_indices([])) == 0


@pytest.mark.parametrize("invalid", [[2], [-2], [0, 1, 5], [0, -1, 3, 0]])
def test_symbol_out_of_range_raises(invalid):
    # sem a validação, -2 viraria o índice -1 (última linha) silenciosamente
    with pytest.raises(ValueError):
        to_indices(invalid)


@pytest.mark.parametrize("invalid", [[0.0, 1.0], [True, False], ["0", "1"]])
def test_non_integer_dtype_raises(invalid):
    with pytest.raises(TypeError):
        to_indices(invalid)


def test_non_one_dimensional_raises():
    with pytest.raises(ValueError):
        to_indices([[0, 1], [1, 0]])
