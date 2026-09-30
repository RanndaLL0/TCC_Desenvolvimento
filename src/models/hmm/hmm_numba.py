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