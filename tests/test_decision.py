import re

import numpy as np
import pytest

from src.utils.decision import predict

GOOD = [0.2, 0.3, 0.5]  # linha válida, prevê +1


def lines_msg(*lines):
    # formato que o numpy usa ao imprimir os índices, ex.: "[1 3]"
    return re.escape(str(np.array(lines)))


def test_predict_toy_matrix():
    matrix = np.array([[0.3, 0.4, 0.3], [0.2, 0.7, 0.1], [0.3, 0.2, 0.5]])
    # linha 0: empate entre -1 e +1 com 0 no topo -> 0; linha 1 -> 0; linha 2 -> +1
    assert predict(matrix).tolist() == [0, 0, 1]


def test_predict_columns_map_to_symbols():
    matrix = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])
    assert predict(matrix).tolist() == [-1, 0, 1]


def test_predict_tie_picks_first_symbol():
    # np.argmax devolve o primeiro índice no empate, ou seja, o símbolo -1
    assert predict(np.full((1, 3), 1 / 3)).tolist() == [-1]
    assert predict(np.array([[0.0, 0.5, 0.5]])).tolist() == [0]


def test_predict_output_is_integer_vector_with_one_symbol_per_row():
    matrix = np.array([GOOD, [0.6, 0.2, 0.2], [0.1, 0.8, 0.1], GOOD])
    result = predict(matrix)
    assert result.shape == (4,)
    assert np.issubdtype(result.dtype, np.integer)
    assert set(result.tolist()) <= {-1, 0, 1}


def test_predict_accepts_nested_list_and_float32():
    assert predict([GOOD, [0.6, 0.2, 0.2]]).tolist() == [1, -1]
    matrix = np.array([GOOD, [0.1, 0.1, 0.8]], dtype=np.float32)
    assert predict(matrix).tolist() == [1, 1]


def test_predict_does_not_modify_input():
    matrix = np.array([GOOD, [0.1, 0.1, 0.8]])
    before = matrix.copy()
    predict(matrix)
    assert np.array_equal(matrix, before)


def test_predict_empty_matrix_returns_empty():
    result = predict(np.empty((0, 3)))
    assert result.shape == (0,)


def test_predict_tolerates_float_noise_in_row_sums():
    assert predict(np.array([[0.1, 0.2, 0.7 + 1e-9]])).tolist() == [1]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.3, 0.4, 0.3]),
        np.ones((2, 3, 3)) / 3,
    ],
)
def test_predict_rejects_non_2d(bad):
    with pytest.raises(ValueError, match="two-dimensional"):
        predict(bad)


def test_predict_rejects_wrong_number_of_columns():
    with pytest.raises(ValueError, match="N-3"):
        predict(np.ones((2, 4)) / 4)


@pytest.mark.parametrize(
    "bad",
    [np.array([[1, 0, 0]]), np.empty((0, 3), dtype=int)],
)
def test_predict_rejects_integer_dtype(bad):
    with pytest.raises(TypeError, match="floating"):
        predict(bad)


# a linha ruim nunca é a 0 e há mais de uma, para provar que o índice
# reportado é o da linha e não uma posição da matriz achatada
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"]
)
def test_predict_reports_rows_with_invalid_numbers(bad_value):
    matrix = np.array([GOOD, [bad_value, 0.5, 0.5], GOOD, [0.5, 0.5, bad_value]])
    with pytest.raises(ValueError, match=lines_msg(1, 3)):
        predict(matrix)


def test_predict_reports_invalid_number_outside_first_column():
    matrix = np.array([GOOD, GOOD, [0.5, np.nan, 0.5]])
    with pytest.raises(ValueError, match=lines_msg(2)):
        predict(matrix)


def test_predict_reports_rows_with_negative_values():
    matrix = np.array([GOOD, [1.5, -0.5, 0.0], GOOD, [1.2, 0.0, -0.2]])
    with pytest.raises(ValueError, match=lines_msg(1, 3)):
        predict(matrix)


def test_predict_reports_rows_that_do_not_sum_to_one():
    matrix = np.array([GOOD, [0.3, 0.3, 0.3], GOOD, [0.9, 0.9, 0.9]])
    with pytest.raises(ValueError, match=lines_msg(1, 3)):
        predict(matrix)


def test_predict_reports_single_bad_sum_row():
    matrix = np.array([GOOD, GOOD, [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match=lines_msg(2)):
        predict(matrix)


def test_predict_checks_invalid_numbers_before_negatives_and_sums():
    matrix = np.array([GOOD, [np.nan, -1.0, 0.0]])
    with pytest.raises(ValueError, match="valid number"):
        predict(matrix)
