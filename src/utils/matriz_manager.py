"""
    Este arquivo serve como utilitário para o gerenciamento das matrizes Numpy pré treinadas geradas pelos modelos preditivos
    não triviais, o armazenamento destas matrizes é realizada através de arquivos .pickle aonde estes podem ser carregados e listados.
"""

import pickle
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


class InfoModelo(TypedDict):
    ll: float | None
    metadados: dict[str, Any]


def para_numpy(matriz: ArrayLike, dtype: DTypeLike = np.float64) -> NDArray[Any]:
    return np.ascontiguousarray(matriz, dtype=dtype)

def preparar_caminho(caminho: str | Path) -> Path:
    caminho = Path(caminho)
    if caminho.suffix not in (".pickle", ".pkl"):
        caminho = caminho.with_suffix(".pickle")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho

def salvar_matriz(matriz: ArrayLike, caminho: str | Path) -> Path:
    caminho = preparar_caminho(caminho=caminho)
    with open(caminho, "wb") as f:
        pickle.dump(para_numpy(matriz), f, protocol=pickle.HIGHEST_PROTOCOL)
    return caminho

def carregar_matriz(caminho: str | Path, dtype: DTypeLike = np.float64) -> NDArray[Any]:
    with open(caminho, "rb") as f:
        obj = pickle.load(f)
    return para_numpy(obj, dtype=dtype)


"""
    Este modulo é dedicado para salvar as matrizes que são geradas pelo modelo HMM
        PI: Vetor de probabilidade inicial
        A: Matriz de probabilidades de transição
        B: Matriz de probabilidades de Emissão
        ll: Log de verossimilhança do modelo
        Metadados se houver
"""
def salvar_modelo(
    caminho: str | Path,
    pi: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    ll: float | None = None,
    **metadados: Any,
) -> Path:
    pi, A, B = para_numpy(pi), para_numpy(A), para_numpy(B)

    pacote = {
        "pi": pi,
        "A": A,
        "B": B,
        "ll": None if ll is None else float(ll),
        "metadados": metadados
    }
    caminho = preparar_caminho(caminho)
    with open(caminho, "wb") as f:
        pickle.dump(pacote, f, protocol=pickle.HIGHEST_PROTOCOL)
    return caminho

def carregar_modelo(
    caminho: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], InfoModelo]:

    with open(caminho, "rb") as f:
        pacote = pickle.load(f)

    pi = para_numpy(pacote["pi"])
    A = para_numpy(pacote["A"])
    B = para_numpy(pacote["B"])

    info: InfoModelo = {"ll": pacote.get("ll"), "metadados": pacote.get("metadados", {})}
    return pi, A, B, info
