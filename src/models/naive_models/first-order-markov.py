import sys
import numpy as np

sys.path.append(".")
from database.index import create_connection


def pegar_dados(): 
    conn = create_connection()

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT close_normalized FROM normalize_data ORDER BY open_time LIMIT 1000")
            linhas = cur.fetchall();
            return linhas
    except:
        print("deu ruim no select do first order")
        
## Para a matriz de transição vou somar um nos dados vindos
# para ficar com o índice correto na matriz
# ent a coluna 0 é na verade quando houve um retorno a baixo
def montar_matriz():
    matriz_contagem = np.zeros((3,3), float)
    dados = pegar_dados()
    dados = np.ravel(dados)
    vetor = np.array(dados).astype(int)
    for i in range(1, len(vetor)):
        anterior= vetor[i - 1]
        atual = vetor[i] 
        matriz_contagem[anterior + 1][atual + 1] += 1 ## soma pra corrigir o indice da matriz

    return matriz_contagem

def verossimilhanca(matriz):
    ll = 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
          ll += matriz[i][j] * np.log(matriz[i][j]/matriz[i].sum())  

    return ll

if __name__ == "__main__":
    matriz = montar_matriz()
    print(matriz)
    
    ll = verossimilhanca(matriz)    
    print(ll)
