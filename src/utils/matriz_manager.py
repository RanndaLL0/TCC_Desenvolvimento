"""
    Este arquivo serve como utilitário para o gerenciamento das matrizes Numpy pré treinadas geradas pelos modelos preditivos
    não triviais, o armazenamento destas matrizes é realizada através de arquivos .pickle aonde estes podem ser carregados e listados.
"""

import pickle
from pathlib import Path
import numpy as np

def preparar_caminho(caminho: str) -> str:
    caminho = Path(caminho)
    if caminho.suffix not in (".pickle", ".pkl"):
        caminho = caminho.with_suffix(".pickle")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return caminho

