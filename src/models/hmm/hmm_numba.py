import warnings

import numpy as np
from numba import njit, prange, get_num_threads

# HMM com emissões discretas (categóricas), ajustado por Baum-Welch.
#
# Os laços pesados (forward, backward, acumulação das contagens esperadas,
# passo M e Viterbi) são kernels compilados com Numba; a classe HMMDiscreto só
# orquestra os reinícios e guarda os parâmetros. O forward e o backward usam a
# normalização de Rabiner (um fator c[t] por instante), então nada underflowa
# em séries longas e a log-verossimilhança sai como soma de logs.
#
# O modelo recebe ÍNDICES de símbolo em [0, n_simbolos), não os símbolos
# {-1, 0, +1} da discretização: a conversão é do src/utils/symbols.py
# (to_indices). Notação usada nos comentários: T é o tamanho da sequência,
# n o número de estados e m o número de símbolos.


@njit(cache=True)
def _forward(y, pi, A, B):
    """Passo forward normalizado.

    alpha[t, i] é P(estado i em t | y[0..t]): cada linha é dividida pela sua
    soma, e o inverso dessa soma fica em c[t] para o backward reaproveitar.

    Args:
        y: int64 (T,), índices dos símbolos observados.
        pi: (n,), distribuição do estado inicial.
        A: (n, n), A[i, j] = P(estado j em t+1 | estado i em t).
        B: (n, m), B[i, k] = P(símbolo k | estado i).

    Returns:
        Tupla (alpha, c, ll): alpha (T, n) normalizado por linha, c (T,) com os
        fatores de escala e ll, a log-verossimilhança log P(y) (log natural).
    """
    T = y.shape[0]
    n = pi.shape[0]
    alpha = np.empty((T, n))
    c = np.empty(T)

    s = 0.0
    for i in range(n):
        alpha[0, i] = pi[i] * B[i, y[0]]
        s += alpha[0, i]
    c[0] = 1.0 / s
    for i in range(n):
        alpha[0, i] *= c[0]
    ll = np.log(s)

    for t in range(1, T):
        yt = y[t]
        s = 0.0
        for j in range(n):
            acc = 0.0
            for i in range(n):
                acc += alpha[t - 1, i] * A[i, j]
            acc *= B[j, yt]
            alpha[t, j] = acc
            s += acc
        inv = 1.0 / s
        c[t] = inv
        for j in range(n):
            alpha[t, j] *= inv
        ll += np.log(s)

    return alpha, c, ll


@njit(cache=True)
def _backward(y, A, B, c):
    """Passo backward, normalizado com os mesmos fatores do forward.

    Usar os c[t] do forward faz alpha[t] * beta[t] ser proporcional à
    posterior dos estados em t, e torna as contagens de transição de
    _acumular já normalizadas, sem dividir por P(y).

    Args:
        y: int64 (T,), índices dos símbolos observados.
        A: (n, n), matriz de transição.
        B: (n, m), matriz de emissão.
        c: (T,), fatores de escala devolvidos por _forward com os mesmos
            parâmetros.

    Returns:
        np.ndarray beta (T, n), normalizado.
    """
    T = y.shape[0]
    n = A.shape[0]
    beta = np.empty((T, n))
    tmp = np.empty(n)

    for i in range(n):
        beta[T - 1, i] = c[T - 1]

    for t in range(T - 2, -1, -1):
        yt1 = y[t + 1]
        for j in range(n):
            tmp[j] = B[j, yt1] * beta[t + 1, j]
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += A[i, j] * tmp[j]
            beta[t, i] = acc * c[t]
    return beta


@njit(cache=True)
def _acumular(y, A, B, alpha, beta, ini, fim, xi, B_num):
    """Soma as contagens esperadas (passo E) dos instantes [ini, fim).

    Para cada t, soma a posterior dos estados (gamma) na coluna do símbolo
    observado em B_num e a posterior de cada transição i -> j entre t e t+1
    em xi. Graças à normalização compartilhada de alpha e beta, o termo
    alpha[t, i] * A[i, j] * B[j, y[t+1]] * beta[t+1, j] já é a probabilidade
    da transição, sem dividir por P(y).

    Args:
        y: int64 (T,), índices dos símbolos observados.
        A: (n, n), matriz de transição usada no forward/backward.
        B: (n, m), matriz de emissão usada no forward/backward.
        alpha: (T, n), devolvido por _forward.
        beta: (T, n), devolvido por _backward.
        ini: primeiro instante do trecho.
        fim: instante seguinte ao último do trecho.
        xi: (n, n), acumulador das transições esperadas; alterado no lugar.
        B_num: (n, m), acumulador das emissões esperadas; alterado no lugar.
    """
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

        if t + 1 < T:
            yt1 = y[t + 1]
            for j in range(n):
                w[j] = B[j, yt1] * beta[t + 1, j]
            for i in range(n):
                a_i = alpha[t, i]
                for j in range(n):
                    xi[i, j] += a_i * A[i, j] * w[j]


@njit(parallel=True, cache=True)
def _acumular_paralelo(y, A, B, alpha, beta, n_blocos):
    """Executa _acumular em paralelo, dividindo a série em trechos contíguos.

    Cada thread acumula num par de matrizes próprio (sem disputa de escrita) e
    as parciais são somadas no fim, sempre na mesma ordem. Os "blocos" aqui são
    só a divisão do trabalho entre threads; não têm relação com os blocos de
    meses da discretização.

    A soma em ponto flutuante depende de como a série foi dividida: o mesmo
    n_blocos dá sempre o mesmo resultado, mas n_blocos diferentes (por exemplo,
    máquinas com outro número de threads) podem diferir nas últimas casas.

    Args:
        y: int64 (T,), índices dos símbolos observados.
        A: (n, n), matriz de transição.
        B: (n, m), matriz de emissão.
        alpha: (T, n), devolvido por _forward.
        beta: (T, n), devolvido por _backward.
        n_blocos: número de trechos em que a série é dividida.

    Returns:
        Tupla (xi, B_num): transições esperadas (n, n) e emissões esperadas
        (n, m), somadas sobre a série inteira.
    """
    T = y.shape[0]
    n = A.shape[0]
    m = B.shape[1]

    xi_loc = np.zeros((n_blocos, n, n))
    Bn_loc = np.zeros((n_blocos, n, m))
    tam = (T + n_blocos - 1) // n_blocos

    for p in prange(n_blocos):
        ini = p * tam
        fim = min(ini + tam, T)
        _acumular(y, A, B, alpha, beta, ini, fim, xi_loc[p], Bn_loc[p])

    xi = np.zeros((n, n))
    B_num = np.zeros((n, m))
    for p in range(n_blocos):
        xi += xi_loc[p]
        B_num += Bn_loc[p]
    return xi, B_num


@njit(cache=True)
def _m_step(alpha, beta, xi, B_num, piso):
    """Passo M: reestima pi, A e B a partir das contagens esperadas.

    pi é a posterior dos estados em t=0. Cada linha de A e de B é a contagem
    esperada normalizada, com `piso` somado a cada célula como pseudocontagem:
    nenhuma probabilidade vira exatamente zero e um estado que não recebeu
    massa não divide por zero (fica com a linha uniforme).

    Args:
        alpha: (T, n), devolvido por _forward.
        beta: (T, n), devolvido por _backward.
        xi: (n, n), transições esperadas.
        B_num: (n, m), emissões esperadas.
        piso: pseudocontagem somada a cada célula de A e B.

    Returns:
        Tupla (pi, A, B) com os novos parâmetros; as linhas somam 1.
    """
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
            A[i, j] = (xi[i, j] + piso) / (sa + n * piso)

        sb = 0.0
        for k in range(m):
            sb += B_num[i, k]
        for k in range(m):
            B[i, k] = (B_num[i, k] + piso) / (sb + m * piso)

    return pi, A, B


@njit(cache=True)
def _viterbi(y, log_pi, log_A, log_B):
    """Algoritmo de Viterbi no espaço log.

    Em caso de empate entre predecessores, fica o de menor índice.

    Args:
        y: int64 (T,), índices dos símbolos observados.
        log_pi: (n,), log da distribuição inicial.
        log_A: (n, n), log da matriz de transição.
        log_B: (n, m), log da matriz de emissão.

    Returns:
        np.ndarray int64 (T,) com o estado do caminho mais provável em cada t.
    """
    T = y.shape[0]
    n = log_pi.shape[0]
    delta = np.empty((T, n))
    psi = np.zeros((T, n), dtype=np.int64)

    for i in range(n):
        delta[0, i] = log_pi[i] + log_B[i, y[0]]

    for t in range(1, T):
        yt = y[t]
        for j in range(n):
            melhor = -np.inf
            arg = 0
            for i in range(n):
                v = delta[t - 1, i] + log_A[i, j]
                if v > melhor:
                    melhor = v
                    arg = i
            delta[t, j] = melhor + log_B[j, yt]
            psi[t, j] = arg

    caminho = np.empty(T, dtype=np.int64)
    caminho[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        caminho[t] = psi[t + 1, caminho[t + 1]]
    return caminho


class HMMDiscreto:
    """HMM com emissões categóricas.

    Hiperparâmetros vão no construtor; parâmetros aprendidos ficam em
    atributos com "_" no final (pi_, A_, B_, ll_, info_), como no scikit-learn.

    Args:
        n_estados: número de estados ocultos (n).
        n_simbolos: tamanho do alfabeto (m); as observações são índices em
            [0, n_simbolos).
        n_reinicios: quantas inicializações aleatórias o fit testa; fica a de
            maior log-verossimilhança.
        max_iter: máximo de iterações do Baum-Welch em cada reinício.
        tolerancia: critério de parada; o ajuste para quando a melhora
            relativa da log-verossimilhança fica abaixo dela.
        piso: pseudocontagem somada a cada célula de A e B no passo M.
        concentracao: parâmetro da Dirichlet simétrica que sorteia os
            parâmetros iniciais; quanto maior, mais perto do uniforme.
        n_blocos: trechos em que o passo E é dividido entre threads; None usa
            o número de threads do Numba. Fixe para resultados idênticos em
            máquinas diferentes (ver _acumular_paralelo).
        semente: semente dos reinícios; a mesma semente reproduz o ajuste.
        verbose: imprime a log-verossimilhança de cada iteração e reinício.

    Attributes:
        pi_: (n,), distribuição do estado inicial.
        A_: (n, n), A_[i, j] = P(estado j em t+1 | estado i em t).
        B_: (n, m), B_[i, k] = P(símbolo k | estado i).
        ll_: log-verossimilhança da sequência de treino no modelo escolhido.
        info_: dict do ajuste: "iteracoes", "convergiu" e "historico" (array
            das log-verossimilhanças) do melhor reinício, "reinicio" (qual
            foi), "semente" e "reinicios" (ll, iterações e convergência de
            cada um).
    """

    def __init__(
        self,
        n_estados,
        n_simbolos,
        n_reinicios=10,
        max_iter=1000,
        tolerancia=1e-8,
        piso=1e-10,
        concentracao=5.0,
        n_blocos=None,
        semente=0,
        verbose=False,
    ):
        self.n_estados = n_estados
        self.n_simbolos = n_simbolos
        self.n_reinicios = n_reinicios
        self.max_iter = max_iter
        self.tolerancia = tolerancia
        self.piso = piso
        self.concentracao = concentracao
        self.n_blocos = n_blocos
        self.semente = semente
        self.verbose = verbose

        self.pi_ = None
        self.A_ = None
        self.B_ = None
        self.ll_ = None
        self.info_ = None


    def __repr__(self):
        estado = "ajustado" if self.ajustado else "nao ajustado"
        return (
            f"HMMDiscreto(n_estados={self.n_estados}, "
            f"n_simbolos={self.n_simbolos}, {estado})"
        )

    @property
    def ajustado(self):
        """True depois que o fit rodou."""
        return self.pi_ is not None

    def _checar_ajustado(self):
        """Garante que o modelo já foi ajustado.

        Raises:
            RuntimeError: se o fit ainda não foi chamado.
        """
        if not self.ajustado:
            raise RuntimeError("chame fit() antes de usar o modelo")

    def _validar_y(self, y):
        """Valida a sequência observada e a converte para o formato dos kernels.

        Args:
            y: sequência 1D de índices de símbolo.

        Returns:
            np.ndarray int64 contíguo, com valores em [0, n_simbolos).

        Raises:
            ValueError: se y não for 1D, estiver vazio ou tiver índice fora de
                [0, n_simbolos) (inclusive o NO_SYMBOL da discretização).
        """
        y = np.ascontiguousarray(y, dtype=np.int64)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y deve ser um vetor 1D nao vazio")
        # O Numba não checa limites de índice: um símbolo fora de [0, M)
        # leria memória inválida em B sem dar erro.
        if y.min() < 0 or y.max() >= self.n_simbolos:
            raise ValueError(
                f"simbolos devem estar em [0, {self.n_simbolos}); "
                f"recebido [{y.min()}, {y.max()}]"
            )
        return y

    def _inicializar(self, rng):
        """Sorteia pi, A e B iniciais de Dirichlets simétricas.

        Args:
            rng: np.random.Generator do reinício.

        Returns:
            Tupla (pi, A, B), com cada linha somando 1.
        """
        n, m, k = self.n_estados, self.n_simbolos, self.concentracao
        pi = rng.dirichlet(np.ones(n) * k)
        A = rng.dirichlet(np.ones(n) * k, size=n)
        B = rng.dirichlet(np.ones(m) * k, size=n)
        return pi, A, B

    @staticmethod
    def aquecer_jit():
        """Compila os kernels num exemplo minúsculo (tira o JIT da medição de tempo)."""
        modelo = HMMDiscreto(2, 2, n_reinicios=1, max_iter=1)
        y = np.array([0, 1, 0, 1, 1], dtype=np.int64)
        modelo.fit(y)
        modelo.predict(y)


    def _baum_welch(self, y, pi, A, B):
        """Roda o Baum-Welch a partir de uma inicialização.

        Para quando a melhora relativa da log-verossimilhança fica abaixo da
        tolerância ou ao atingir max_iter. Se a log-verossimilhança cair mais
        que a tolerância (o que o EM não deveria fazer, então indica problema
        numérico), emite um aviso, descarta a iteração e devolve os parâmetros
        anteriores.

        Args:
            y: int64 (T,), já validado por _validar_y.
            pi: (n,), distribuição inicial de partida.
            A: (n, n), matriz de transição de partida.
            B: (n, m), matriz de emissão de partida.

        Returns:
            Tupla (pi, A, B, ll, info): os parâmetros finais, a
            log-verossimilhança deles e um dict com "iteracoes" (iterações
            aceitas), "convergiu" (False se parou por max_iter ou por queda) e
            "historico" (log-verossimilhança inicial e após cada iteração).
        """
        n_blocos = self.n_blocos or get_num_threads()

        alpha, c, ll = _forward(y, pi, A, B)
        historico = [ll]
        convergiu = False

        for it in range(self.max_iter):
            beta = _backward(y, A, B, c)
            xi, B_num = _acumular_paralelo(y, A, B, alpha, beta, n_blocos)
            novo_pi, novo_A, novo_B = _m_step(alpha, beta, xi, B_num, self.piso)

            novo_alpha, novo_c, novo_ll = _forward(y, novo_pi, novo_A, novo_B)

            # Diferença relativa
            variacao = (novo_ll - ll) / abs(ll)
            if variacao < -self.tolerancia:
                warnings.warn(
                    f"a log-verossimilhanca diminuiu na iteracao {it} "
                    f"({ll:.6f} -> {novo_ll:.6f}); mantendo os parametros anteriores"
                )
                break

            pi, A, B = novo_pi, novo_A, novo_B
            alpha, c, ll = novo_alpha, novo_c, novo_ll
            historico.append(ll)

            if self.verbose:
                print(f"  iteracao {it:3d} ll = {ll:.4f}")

            if variacao < self.tolerancia:
                convergiu = True
                break

        info = {
            "iteracoes": len(historico) - 1,
            "convergiu": convergiu,
            "historico": np.array(historico),
        }
        return pi, A, B, ll, info

    def fit(self, y):
        """Ajusta o modelo, ficando com o melhor de n_reinicios.

        Cada reinício parte de parâmetros sorteados com uma semente derivada de
        `semente` (SeedSequence.spawn), então o ajuste é reprodutível.

        y é tratado como uma única sequência contínua: transições que
        atravessam uma lacuna da série são contadas como se fossem vizinhas
        (pendência D6 em docs/discretizacao.md).

        Args:
            y: sequência 1D de índices de símbolo em [0, n_simbolos).

        Returns:
            self, com pi_, A_, B_, ll_ e info_ preenchidos.

        Raises:
            ValueError: se y for inválido (ver _validar_y).
        """
        y = self._validar_y(y)
        sementes = np.random.SeedSequence(self.semente).spawn(self.n_reinicios)

        melhor = None
        reinicios = []
        for r, ss in enumerate(sementes):
            pi0, A0, B0 = self._inicializar(np.random.default_rng(ss))
            pi, A, B, ll, info = self._baum_welch(y, pi0, A0, B0)

            reinicios.append(
                {"ll": ll, "iteracoes": info["iteracoes"], "convergiu": info["convergiu"]}
            )
            if self.verbose:
                print(
                    f"reinicio {r:2d} ll = {ll:.4f} iteracoes = {info['iteracoes']} "
                    f"convergiu = {info['convergiu']}"
                )

            if melhor is None or ll > melhor[3]:
                melhor = (pi, A, B, ll, {**info, "reinicio": r})

        self.pi_, self.A_, self.B_, self.ll_, info = melhor
        info["semente"] = self.semente
        info["reinicios"] = reinicios
        self.info_ = info
        return self

    def score(self, y):
        """Calcula a log-verossimilhança de y sob o modelo ajustado.

        Args:
            y: sequência 1D de índices de símbolo em [0, n_simbolos).

        Returns:
            float com log P(y) (log natural, somado sobre a sequência; não é
            média por símbolo).

        Raises:
            RuntimeError: se o modelo não foi ajustado.
            ValueError: se y for inválido.
        """
        self._checar_ajustado()
        y = self._validar_y(y)
        _, _, ll = _forward(y, self.pi_, self.A_, self.B_)
        return ll

    def predict_proba(self, y):
        """Calcula a posterior de cada estado em cada instante (forward-backward).

        Args:
            y: sequência 1D de índices de símbolo em [0, n_simbolos).

        Returns:
            np.ndarray (T, n_estados); a linha t é P(estado em t | y inteiro) e
            soma 1. Usa a sequência toda, inclusive instantes depois de t.

        Raises:
            RuntimeError: se o modelo não foi ajustado.
            ValueError: se y for inválido.
        """
        self._checar_ajustado()
        y = self._validar_y(y)
        alpha, c, _ = _forward(y, self.pi_, self.A_, self.B_)
        beta = _backward(y, self.A_, self.B_, c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def predict(self, y):
        """Encontra a sequência de estados mais provável (Viterbi).

        Args:
            y: sequência 1D de índices de símbolo em [0, n_simbolos).

        Returns:
            np.ndarray int64 (T,) com o estado de cada instante.

        Raises:
            RuntimeError: se o modelo não foi ajustado.
            ValueError: se y for inválido.
        """
        self._checar_ajustado()
        y = self._validar_y(y)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.pi_)
            log_A = np.log(self.A_)
            log_B = np.log(self.B_)
        return _viterbi(y, log_pi, log_A, log_B)

    def sample(self, T, semente=None):
        """Gera uma sequência sintética a partir do modelo ajustado.

        Args:
            T: tamanho da sequência, inteiro >= 1.
            semente: semente do gerador; None gera uma sequência diferente a
                cada chamada.

        Returns:
            Tupla (estados, simbolos), dois np.ndarray int64 (T,). Os símbolos
            são índices em [0, n_simbolos), não {-1, 0, +1}.

        Raises:
            RuntimeError: se o modelo não foi ajustado.
        """
        self._checar_ajustado()
        rng = np.random.default_rng(semente)
        estados = np.empty(T, dtype=np.int64)
        simbolos = np.empty(T, dtype=np.int64)
        estados[0] = rng.choice(self.n_estados, p=self.pi_)
        simbolos[0] = rng.choice(self.n_simbolos, p=self.B_[estados[0]])
        for t in range(1, T):
            estados[t] = rng.choice(self.n_estados, p=self.A_[estados[t - 1]])
            simbolos[t] = rng.choice(self.n_simbolos, p=self.B_[estados[t]])
        return estados, simbolos



if __name__ == "__main__":
    # Ajusta um HMM de 3 estados na base inteira (python -m src.models.hmm.hmm_numba)
    import time
    from database.index import pegar_dados
    from src.utils.symbols import SYMBOLS

    N = 3
    M = len(SYMBOLS)

    btc_usdt = pegar_dados()
    y = np.array([int(d[0]) + 1 for d in btc_usdt], dtype=np.int64)

    HMMDiscreto.aquecer_jit()

    t0 = time.perf_counter()
    modelo = HMMDiscreto(N, M, n_reinicios=10, semente=0, verbose=True).fit(y)
    print(f"tempo: {time.perf_counter() - t0:.3f}s  ({get_num_threads()} threads)")

    print(f"melhor reinicio: {modelo.info_['reinicio']}  convergiu: {modelo.info_['convergiu']}")
    print(f"log-verossimilhanca final: {modelo.ll_:.4f}")
    print("A estimado:")
    print(np.round(modelo.A_, 3))
    print("B estimado:")
    print(np.round(modelo.B_, 3))

    estados = modelo.predict(y)
    print("ultimos 20 estados (Viterbi):", estados[-20:])
