import sys
import numpy as np

N = 3
M = 3
T = 972_817

PI = np.array([0.33, 0.33, 0.34])

A = np.array([[0.80, 0.10, 0.10],
              [0.10, 0.80, 0.10],
              [0.10, 0.10, 0.80]])

B = np.array([[0.70, 0.20, 0.10],
              [0.10, 0.70, 0.20],
              [0.20, 0.10, 0.70]])

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


""" 
    tentei seguir a logica do escalonamento que o Jurafsky usa no livro para 
    Y = Sequencia de estados ocultos observados em uma sequencia
    pi = Vetor de probabilidades iniciais para cada estado
    A = Matriz de probabilidade para a transicao entre os estados
    B = Matriz de probabilidade de emisao da observacao dado os estados ocultos
"""
def forward(y, pi, A, B):
    alpha = np.empty((len(y), len(pi)))
    c = np.empty(len(y))

    alpha[0] = pi * B[:, y[0]]

    """ Fator de escalonamento (É o inverso da soma das probabilidades da emisão da observação) """
    c[0] = 1.0 / alpha[0].sum()
    """ Normalizacao da probabilidade inicial (Evita que os valores ultrapassem 1.0 i.e 100%) """
    alpha[0] *= c[0]

    for t in range(1, len(y)):
        """
            Itera calculando a probabilidade de emisão (alpha) do estado obervado ter sido gerada pelo modelo
            dado as condicoes iniciais do sistema.
        """
        a = (alpha[t - 1] @ A) * B[:, y[t]]
        c[t] = 1.0 / a.sum()
        alpha[t] = a * c[t]

    """
        -np.log(c).sum() é o log de verossimilhança, o que é essencialmente o que o algoritmo forward calcula,
        a probabilidade da sequencia de observacoes terem sido geradas pelo modelo i.e P(Y|lambda).
    """
    return alpha, c, -np.log(c).sum()


def backward(y, A, B, c):
    beta = np.empty((len(y), A.shape[0]))

    """
        'Para cada estado que eu poderia estar agora, qual a chance de ir para todos os outros estados possiveis
        tendo emitido a observacao que eu sei que acontece'
    """
    beta[-1] = c[-1]
    for t in range(len(y) - 2, -1, -1):
        beta[t] = (A @ (B[:, y[t + 1]] * beta[t + 1])) * c[t]

    return beta

def e_step(y, pi, A, B):
    alpha, c, ll = forward(y, pi, A, B)
    beta = backward(y, A, B, c)

    """ O gamma essencialmente é a probabilidade de estar em cada estado em cada instante de tempo da sequencia """
    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    """ 
        O XI é o numero estimado de transições entre cada par de estados ocultos durante toda observacao 
        (Aqui dói a cabeça)
    """
    xi = A * (alpha[:-1].T @ (B[:, y[1:]].T * beta[1:]))

    return gamma, xi, ll

def m_step(y, gamma, xi, n_simbolos=M):

    pi = gamma[0].copy()
    A = xi / xi.sum(axis=1, keepdims=True)

    B = np.zeros((gamma.shape[1], n_simbolos))
    for k in range(n_simbolos):
        B[:, k] = gamma[y == k].sum(axis=0)
    B /= B.sum(axis=1, keepdims=True)

    return pi, A, B

def baum_welch(y, pi, A, B, max_iter=100, tol=1e-4):
    ll_anterior = -np.inf

    for i in range(max_iter):
        gamma, xi, ll = e_step(y, pi, A, B)
        pi, A, B = m_step(y, gamma, xi, B.shape[1])

        if ll - ll_anterior < tol:
            break
        ll_anterior = ll

    return pi, A, B, ll


if __name__ == "__main__":
    y, _ = generator(5000, PI, A, B)

    rng = np.random.default_rng(0)
    pi0 = rng.dirichlet(np.ones(N) * 5)
    A0 = rng.dirichlet(np.ones(N) * 5, size=N)
    B0 = rng.dirichlet(np.ones(M) * 5, size=N)

    pi_est, A_est, B_est, ll = baum_welch(y, pi0, A0, B0, max_iter=10000)

    print(f"log-verossimilhanca final: {ll:.4f}")

    print("\nA real:")
    print(A)
    print("A estimado:")
    print(np.round(A_est, 3))

    print("\nB real:")
    print(B)
    print("B estimado:")
    print(np.round(B_est, 3))
