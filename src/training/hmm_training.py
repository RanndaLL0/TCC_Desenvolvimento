import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.hmm.hmm_numba import HMMDiscreto
from src.preprocess.discretize import MODES, discretize, load_snapshot
from src.utils.symbols import SYMBOLS, to_indices

# Treino do HMM discreto sobre o snapshot discretizado.
#
# O fluxo é: load_snapshot -> discretize -> corte treino/teste (o mesmo cut da
# discretização) -> to_indices -> HMMDiscreto.fit. O teste só é usado para
# medir a log-verossimilhança fora da amostra; nada dele entra no ajuste.
#
# As posições sem símbolo (NO_SYMBOL) são descartadas antes do ajuste, então
# os segmentos separados por lacunas são concatenados e o HMM os trata como
# uma sequência contínua (pendência D6 em docs/discretizacao.md).

ROOT = Path(__file__).resolve().parents[2]
STD_FOLDER = ROOT / "relatorios"


def split_sequences(result):
    """Separa os símbolos de treino e de teste de uma discretização.

    Args:
        result: Discretization devolvida por discretize.

    Returns:
        Tupla (train, test) de np.ndarray int64 com ÍNDICES de símbolo em
        [0, len(SYMBOLS)), já sem as posições NO_SYMBOL.

    Raises:
        ValueError: se o treino ou o teste ficarem sem nenhum símbolo.
    """
    valid = result.valid
    cut = result.cut

    train = result.symbols[:cut][valid[:cut]]
    test = result.symbols[cut:][valid[cut:]]

    if train.size == 0 or test.size == 0:
        raise ValueError("the train or test partition has no symbols")

    return to_indices(train).astype(np.int64), to_indices(test).astype(np.int64)


def n_free_parameters(n_states, n_symbols):
    """Conta os parâmetros livres do HMM (pi, A e B, cada linha soma 1).

    Args:
        n_states: número de estados ocultos (n).
        n_symbols: tamanho do alfabeto (m).

    Returns:
        int com (n - 1) + n(n - 1) + n(m - 1).
    """
    return (n_states - 1) + n_states * (n_states - 1) + n_states * (n_symbols - 1)


def train_hmm(train, test, n_states, **hmm_kwargs):
    """Ajusta um HMM no treino e avalia no treino e no teste.

    Args:
        train: índices de símbolo do treino.
        test: índices de símbolo do teste.
        n_states: número de estados ocultos.
        **hmm_kwargs: hiperparâmetros repassados para HMMDiscreto
            (n_reinicios, max_iter, tolerancia, semente, ...).

    Returns:
        Tupla (model, entry): o HMMDiscreto ajustado e um dict com as
        log-verossimilhanças, AIC, BIC, dados do ajuste e os parâmetros.
    """
    n_symbols = len(SYMBOLS)
    model = HMMDiscreto(n_states, n_symbols, **hmm_kwargs)

    t0 = time.perf_counter()
    model.fit(train)
    elapsed = time.perf_counter() - t0

    ll_train = float(model.ll_)
    ll_test = float(model.score(test))
    p = n_free_parameters(n_states, n_symbols)

    entry = {
        "n_states": n_states,
        "free_parameters": p,
        "ll_train": ll_train,
        "ll_train_per_symbol": ll_train / train.size,
        "ll_test": ll_test,
        "ll_test_per_symbol": ll_test / test.size,
        "aic": 2 * p - 2 * ll_train,
        "bic": p * np.log(train.size) - 2 * ll_train,
        "fit_seconds": round(elapsed, 3),
        "iterations": model.info_["iteracoes"],
        "converged": model.info_["convergiu"],
        "best_restart": model.info_["reinicio"],
        "restarts": model.info_["reinicios"],
        "pi": model.pi_.tolist(),
        "A": model.A_.tolist(),
        "B": model.B_.tolist(),
    }

    return model, entry


def main():
    """Ponto de entrada: discretiza um snapshot, treina o HMM e grava o relatório.

    Gera um .json para cada combinação de modo e k pedida, com um resultado
    para cada número de estados, e imprime os caminhos dos arquivos criados.
    """
    parser = argparse.ArgumentParser(
        description="Discretizes a snapshot and trains the discrete HMM on it"
    )
    parser.add_argument(
        "--snapshot", required=True, type=Path, help="path to the Parquet file"
    )
    parser.add_argument("--mode", nargs="+", choices=MODES, default=["blocks"])
    parser.add_argument(
        "--k", nargs="+", type=int, default=[3], help="threshold s/k"
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="months per block (blocks mode)"
    )
    parser.add_argument("--train", type=int, default=95, help="train percentage")
    parser.add_argument(
        "--states", nargs="+", type=int, default=[3], help="number of hidden states"
    )
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="E-step chunks; fix it for identical results across machines",
    )
    parser.add_argument("--output", type=Path, default=STD_FOLDER)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    open_times, close, source = load_snapshot(args.snapshot)
    hmm_kwargs = {
        "n_reinicios": args.restarts,
        "max_iter": args.max_iter,
        "tolerancia": args.tol,
        "n_blocos": args.blocks,
        "semente": args.seed,
        "verbose": args.verbose,
    }

    HMMDiscreto.aquecer_jit()
    args.output.mkdir(parents=True, exist_ok=True)

    for mode in args.mode:
        for k in args.k:
            result = discretize(
                open_times,
                close,
                k,
                mode=mode,
                interval_months=args.interval,
                train_pct=args.train,
            )
            train, test = split_sequences(result)

            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "source": source,
                "discretization": {
                    "mode": result.mode,
                    "k": result.k,
                    "threshold": f"s/{result.k}",
                    "interval_months": result.interval_months,
                    "train_pct": result.train_pct,
                    "first_test_row": result.cut,
                    "train_symbols": int(train.size),
                    "test_symbols": int(test.size),
                },
                "hmm": {
                    "symbols": list(SYMBOLS),
                    "restarts": args.restarts,
                    "max_iter": args.max_iter,
                    "tolerance": args.tol,
                    "seed": args.seed,
                    "e_step_blocks": args.blocks,
                },
                "results": [],
            }

            for n_states in args.states:
                print(f"{mode} s/{k}: training HMM with {n_states} states")
                _, entry = train_hmm(train, test, n_states, **hmm_kwargs)
                report["results"].append(entry)
                print(
                    f"  ll/symbol train = {entry['ll_train_per_symbol']:.6f}  "
                    f"test = {entry['ll_test_per_symbol']:.6f}  "
                    f"bic = {entry['bic']:.1f}  converged = {entry['converged']}"
                )

            path = args.output / f"hmm_{source['until_exclusive']}_{mode}_k{k}.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(path)


if __name__ == "__main__":
    main()
