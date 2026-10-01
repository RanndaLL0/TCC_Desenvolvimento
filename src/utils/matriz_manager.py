"""
    Este arquivo serve como utilitário para o gerenciamento das matrizes Numpy pré treinadas geradas pelos modelos preditivos
    não triviais, o armazenamento destas matrizes é realizada através de arquivos .pickle aonde estes podem ser carregados e listados.
"""

import pickle
from pathlib import Path
import numpy as np

def para_numpy(matriz, dtype=np.float64) -> np.ndarray:
    np.ascontiguousarray(matriz, dtype=dtype)

def preparar_caminho(caminho: str) -> str:
    caminho = Path(caminho)
    if caminho.suffix not in (".pickle", ".pkl"):
        caminho = caminho.with_suffix(".pickle")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho

def salvar_matriz(matriz, caminho: str) -> str:
    caminho = preparar_caminho(caminho=caminho)
    with open(caminho, "wb") as f:
        pickle.dump(para_numpy(matriz), f, protocol=pickle.HIGHEST_PROTOCOL)
    return caminho

def carregar_matriz(caminho, dtype=np.float64):
    with open(caminho, "rb") as f:
        obj = pickle.load(f)
    return para_numpy(obj, dtype=dtype)