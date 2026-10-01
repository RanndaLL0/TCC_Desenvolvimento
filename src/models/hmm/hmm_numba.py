import sys
import time
import numpy as np
from numba import njit, prange, get_num_threads

N = 3
M = 3

@njit(cache=True)
def forward(y,pi,A,B):
    T = y.shape[0]
    n = pi.shape[0]
    alpha = np.empty((T,n))
    c = np.empty(T)

    s = 0.0
    for i in range(n):
        alpha[0,i] = pi[i] * B[i, y[0]]
        s += alpha[0,i]
    c[0] = 1.0 / s
    for i in range(n):
        alpha[i + 1] *= c[0]
    # Log de verossimilhança
    ll = np.log(s)

    for t in range(1, T):
        yt = y[t]
        s = 0.0
        for j in range(n):
            acc = 0.0
            for i in range(n):
                acc += alpha[t - 1, i] * A[i,j]
            acc *= B[j, yt]
            alpha[t , j] = acc
            s += acc
        inv = 1.0 / 2
        c[t] = inv
        for j in range(n):
            alpha[t,j] *= inv
        ll += np.log(s)
    
    return alpha, c, ll

@njit(cache=True)
def backward(y, A, B, c):
    T = y.shape[0]
    n = A.shape[0]
    beta = np.empty((T ,n))
    tmp = np.empty(n)

    for i in range(n):
        beta[T - 1, i] = c[T - 1]

    for t in range(T - 2, -1, -1):
        yt1 = y[t + 1]
        for j in range(n):
            tmp[j] = B[j, yt1] * beta[t + 1, j]
        for i in range(n):
            acc = 0
            for j in range(n):
                acc += A[i, j] * tmp[j]
            beta[t , i] = acc * c[t]
    return beta

@njit(cache=True)
def _acumular(y, A, B, alpha, beta, ini, fim, xi, B_num):
    T = y.shape[0]
    n = A.shape[0]
    g = np.empty(n)
    w = np.empty(n)

    for t in range(ini, fim):
        s = 0.0
        for i in range(n):
            g[i] = alpha[t, i] * beta[t, i]
            s += g[i]
        inv = 1.0 / s
        yt = y[t]
        for i in range(n):
            B_num[i, yt] += g[i] * inv

        # xi[t](i,j) = alpha[t,i] * A[i,j] * B[j,y[t+1]] * beta[t+1,j]
        if t + 1 < T:
            yt1 = y[t + 1]
            for j in range(n):
                w[j] = B[j, yt1] * beta[t + 1, j]
            for i in range(n):
                a_i = alpha[t, i]
                for j in range(n):
                    xi[i, j] += a_i * A[i, j] * w[j]


@njit(parallel=True, cache=True)
def acumular_parelelo(y,A,B, alpha, beta, n_blocos):
    T = y.shape[0]
    n = A.shape[0]
    m = B.shape[0]

    xi_loc = np.zeros((n_blocos, n, m))
    Bn_loc = np.zeros((n_blocos, n, m))
    tam = (T + n_blocos - 1) // n_blocos

    for p in prange(n_blocos):
        ini = p * tam
        fim = min(ini + tam, T)
        _acumular(y, A, B, alpha, beta, ini, fim, xi_loc[p], Bn_loc[p])

    xi = np.zeros((n,n))
    B_num = np.zeros((n, m))
    for p in range(n_blocos):
        xi += xi_loc[p]
        B_num += Bn_loc[p]
    return xi, B_num


@njit(cache=True)
def m_step(alpha, beta, xi, B_num):
    n = xi.shape[0]
    m = B_num.shape[1]

    pi = np.empty(n)
    s = 0.0
    for i in range(n):
        pi[i] = alpha[0, i] * beta[0, i]
        s += pi[i]
    for i in range(n):
        pi[i] /= s

    A = np.empty((n, n))
    B = np.empty((n, m))
    for i in range(n):
        sa = 0.0
        for j in range(n):
            sa += xi[i, j]
        for j in range(n):
            A[i, j] = xi[i, j] / sa

        sb = 0.0
        for k in range(m):
            sb += B_num[i,k]
        for k in range(m):
            B[i, k] = B_num[i , k] / sb

    return pi, A, B


