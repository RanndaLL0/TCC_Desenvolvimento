import numpy as np

SYMBOLS = (-1, 0, 1)
DATA_OFF_SET = 1  # off_set para converter discretização em indice


## Valida uma sequência de símbolos e a converte para índices de linha/coluna
# das matrizes de transição (símbolo + off_set). Centraliza a validação para que
# um valor fora de {-1, 0, 1} nunca vire um índice negativo silencioso do numpy
def to_indices(data):
    arr = np.asarray(data)

    if arr.ndim != 1:
        raise ValueError("the sequence of symbols must be one-dimensional")

    if arr.size == 0:
        return np.empty(0, dtype=int)

    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"the symbols must be integers, got dtype {arr.dtype}")

    invalid = ~np.isin(arr, SYMBOLS)
    if invalid.any():
        raise ValueError(
            f"the symbols must be in {SYMBOLS}, found {np.unique(arr[invalid])}"
        )

    return arr.astype(int) + DATA_OFF_SET
