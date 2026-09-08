import sys
import numpy as np

sys.path.append(".")
from database.index import create_connection

N = 3
M = 3
T = 200_000

PI = np.array([0.33, 0.33, 0.34])

A = np.array([[0.33, 0.33, 0.34],
              [0.33, 0.33, 0.34],
              [0.33, 0.33, 0.34]])

B = np.array([[0.33, 0.33, 0.34],
              [0.33, 0.33, 0.34],
              [0.33, 0.33, 0.34]])

# Gera uma sequencia de observações de tamanho T (tamanho da base)
# Ira servir como uma função de debug caso os dados gere dificuldades
# para a interpretacao do que esta acontecendo.
def generator(T, pi=PI, A=A, B=B):
    rng = np.random.default_rng()
    y = np.empty(T, dtype=np.int64)
    states = np.empty(T, dtype=np.int64)

    q = rng.choice(N, p=pi)
    for t in range(T):
        y[t] = rng.choice(M, p=B[q])
        states[t] = q
        q = rng.choice(N, p=A[q])

    return y, states

def test_conexao():
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT close FROM btc_usdt ORDER BY open_time LIMIT 10")
            linhas = cur.fetchall();
            return linhas
    except:
        print("Deu ruim no select");

if __name__ == "__main__":
    test_conexao()
    print("ok")