import math
import numbers


## BIC = -2 * LL + p * log(n). Menor é melhor, e só faz sentido comparar
# candidatos avaliados sobre as MESMAS n transições (a LL de cada um precisa ser
# somada sobre o mesmo conjunto de observações). Não é uma medida absoluta
# de qualidade
def bic(log_likelihood, n_params, n_obs):
    if isinstance(log_likelihood, bool) or not isinstance(
        log_likelihood, numbers.Real
    ):
        raise TypeError("log_likelihood should be a real number")

    for name, value in (("n_params", n_params), ("n_obs", n_obs)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} should be an integer")
        if value < 1:
            raise ValueError(f"{name} should be at least 1")

    if math.isnan(log_likelihood) or log_likelihood == math.inf:
        raise ValueError("log_likelihood should be a number or -infinity")

    # LL = -inf (transição impossível) dá BIC = +inf: candidato descartado
    return -2.0 * float(log_likelihood) + int(n_params) * math.log(int(n_obs))
