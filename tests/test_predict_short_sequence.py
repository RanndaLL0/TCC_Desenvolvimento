from collections import Counter

import numpy as np
import pytest

from src.models.naive_models.first_order import FirstOrder
from src.models.naive_models.persistence import Persistence
from src.utils.decision import predict
from src.utils.split import split_data

SYMBOLS = (-1, 0, 1)

# sequência curta e determinística de 100 elementos: o brinquedo da validação
# do first_order repetido 10 vezes. Os 4 primeiros símbolos já contêm os três
# estados, então o fit funciona mesmo com apenas 5 elementos de treino
BASE = [-1, 0, 0, 1, -1, 0, 1, 1, 0, -1]
SEQUENCE = np.array(BASE * 10)

# 5% de treino (95% de teste) e, para conferência, o inverso
SPLITS = [5, 95]


def reference_transition_matrix(train):
    # referência em Python puro, sem numpy, para não repetir a lógica do modelo
    counts = Counter(zip(train[:-1], train[1:]))
    matrix = []
    for prev in SYMBOLS:
        total = sum(counts[(prev, nxt)] for nxt in SYMBOLS)
        matrix.append([counts[(prev, nxt)] / total for nxt in SYMBOLS])
    return matrix


def reference_predict(matrix, symbols):
    # argmax por linha; no empate fica o primeiro símbolo, como np.argmax
    result = []
    for symbol in symbols:
        row = matrix[SYMBOLS.index(symbol)]
        result.append(SYMBOLS[row.index(max(row))])
    return result


@pytest.mark.parametrize("train_percentage", SPLITS)
def test_split_of_short_sequence_has_expected_sizes(train_percentage):
    train, test = split_data(SEQUENCE, train_percentage, 100)
    assert len(train) == train_percentage
    assert len(test) == 100 - train_percentage
    assert np.array_equal(np.concatenate([train, test]), SEQUENCE)


@pytest.mark.parametrize("train_percentage", SPLITS)
def test_first_order_predict_matches_reference(train_percentage):
    train, test = split_data(SEQUENCE, train_percentage, 100)
    model = FirstOrder()
    model.fit(train)

    reference = reference_transition_matrix(train.tolist())
    assert np.allclose(model.prob, reference)

    proba = model.predict_proba(test)
    assert proba.shape == (len(test), 3)
    assert np.allclose(proba.sum(axis=1), 1)

    predictions = predict(proba)
    assert predictions.shape == (len(test),)
    assert predictions.tolist() == reference_predict(reference, test.tolist())


def test_first_order_five_percent_train_toy_values():
    # treino = [-1, 0, 0, 1, -1] -> P: (-1)->[0,1,0], 0->[0,.5,.5], (+1)->[1,0,0]
    train, test = split_data(SEQUENCE, 5, 100)
    model = FirstOrder()
    model.fit(train)
    assert np.allclose(model.prob, [[0, 1, 0], [0, 0.5, 0.5], [1, 0, 0]])

    # -1 -> 0; 0 -> empate entre 0 e +1 resolvido para 0; +1 -> -1
    mapping = dict(zip(SYMBOLS, predict(model.prob)))
    assert mapping == {-1: 0, 0: 0, 1: -1}
    assert predict(model.predict_proba(test)).tolist() == [mapping[s] for s in test]


@pytest.mark.parametrize("train_percentage", SPLITS)
def test_one_step_pairing_matches_manual_count(train_percentage):
    # modo A: a previsão da posição i é conferida contra o símbolo i+1
    train, test = split_data(SEQUENCE, train_percentage, 100)
    model = FirstOrder()
    model.fit(train)
    predictions = predict(model.predict_proba(test))

    hits = predictions[:-1] == test[1:]
    manual = sum(1 for i in range(len(test) - 1) if predictions[i] == test[i + 1])
    assert hits.sum() == manual
    assert len(hits) == len(test) - 1


@pytest.mark.parametrize("train_percentage", SPLITS)
def test_predictions_use_only_the_past(train_percentage):
    # prever o teste inteiro de uma vez deve dar o mesmo que prever prefixos:
    # se divergir, alguma linha está usando informação do futuro
    train, test = split_data(SEQUENCE, train_percentage, 100)
    model = FirstOrder()
    model.fit(train)

    full = model.predict_proba(test)
    for n in range(1, len(test) + 1):
        prefix = model.predict_proba(test[:n])
        assert np.allclose(prefix[-1], full[n - 1])


def test_predict_is_agnostic_to_the_model():
    # a mesma função de decisão serve a outro modelo sem alterações:
    # a persistência prevê que o próximo símbolo é igual ao atual
    train, test = split_data(SEQUENCE, 5, 100)
    model = Persistence(epsilon=0.05)
    model.fit(train)
    assert predict(model.predict_prob(test)).tolist() == test.tolist()
