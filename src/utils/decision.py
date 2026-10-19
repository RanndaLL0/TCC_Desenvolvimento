import numpy as np
from src.utils.symbols import to_symbols


def predict(data):
    arr = np.asarray(data)

    if arr.ndim != 2:
        raise ValueError("the data must be two-dimensional")

    if arr.shape[1] != 3:
        raise ValueError("the data must be a martrix N-3")

    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"the indices must be floating, got dtype {arr.dtype}")

    if arr.size == 0:
        return np.empty(0, dtype=int)

    if not np.isfinite(arr).all():
        elements = np.any(~np.isfinite(arr), axis=1)
        raise ValueError(f"the lines {np.flatnonzero(elements)} was not a valid number")

    if np.any(arr < 0):
        elements = np.any(arr < 0, axis=1)
        raise ValueError(
            f"the lines {np.flatnonzero(elements)} are negative. The elements must be zero or positive"
        )

    sums = arr.sum(axis=1, keepdims=True)
    if not np.allclose(sums, 1):
        elements = ~np.isclose(sums, 1)
        raise ValueError(
            f"the line {np.flatnonzero(elements)} did not sum probability to 1"
        )

    indices = np.argmax(arr, axis=1)
    return to_symbols(indices)
