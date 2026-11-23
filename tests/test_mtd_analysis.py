import itertools
import json
import math
from datetime import date

import numpy as np
import pytest

from src.experiments import mtd_analysis as ma
from src.models.mtd.mtd import MTD
from src.utils.symbols import to_symbols

LAM = np.array([0.5, 0.3, 0.2])
Q = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.25, 0.25, 0.5]])


def fitted_mtd(lam=LAM, q=Q):
    return ma.restore_mtd(len(lam), lam, q)


def mixture(x, t, lam, q):
    """P(x_t | x_{t-1}, ...) do MTD, escrito direto da definição."""
    return sum(lam[g] * q[x[t - 1 - g]] for g in range(len(lam)))


# --------------------------------------------- métricas de previsão


def test_evaluate_predictions_by_hand():
    # contexto fictício: 4 posições, alvos -1, 0, +1, +1 (índices 0, 1, 2, 2)
    x = np.array([1, 0, 1, 2, 2])
    pred = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],  # linha 0 não é avaliada
            [0.5, 0.3, 0.2],  # alvo -1: argmax -1, direção -1 (certo)
            [0.1, 0.8, 0.1],  # alvo 0: argmax 0
            [0.2, 0.2, 0.6],  # alvo +1: argmax +1, direção +1 (certo)
            [0.4, 0.1, 0.5],  # alvo +1: argmax +1, direção +1 (certo)
        ]
    )
    rows = np.arange(1, 5)
    r = ma.evaluate_predictions(pred, x, rows)

    ll = math.log(0.5) + math.log(0.8) + math.log(0.6) + math.log(0.5)
    assert r["n"] == 4
    assert r["ll"] == pytest.approx(ll)
    assert r["ll_per_symbol"] == pytest.approx(ll / 4)
    assert r["accuracy"] == 1.0
    assert r["direction"]["hits"] == 3 and r["direction"]["n"] == 3
    assert r["zero_share"] == 0.25
    assert r["non_zero_pred_share"] == 0.75
    assert r["non_zero_pred_accuracy"] == 1.0


def test_naive_accuracy():
    x = np.array([1, 1, 0, 0, 2, 1])
    rows = np.arange(1, 6)
    r = ma.naive_accuracy(x, rows)
    assert r["always_zero"] == pytest.approx(2 / 5)  # alvos 1,0,0,2,1
    assert r["persistence"] == pytest.approx(2 / 5)  # repetem as posições 1 e 3


# ---------------------------------------------------------- baselines


def test_first_order_pred_matches_hand_count():
    symbols = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])
    x = symbols + 1
    pred = ma.first_order_pred(symbols, x)

    counts = np.zeros((3, 3))
    for t in range(1, len(x)):
        counts[x[t - 1], x[t]] += 1
    table = counts / counts.sum(axis=1, keepdims=True)

    assert np.allclose(pred[0], 1 / 3)  # sem contexto
    for t in range(1, len(x)):
        assert pred[t] == pytest.approx(table[x[t - 1]])
    assert np.allclose(pred.sum(axis=1), 1)


def test_persistence_pred_puts_most_mass_on_the_last_symbol():
    symbols = np.array([-1, 0, 0, 1, -1, 0, 1, 1, 0, -1])
    x = symbols + 1
    pred = ma.persistence_pred(symbols, x)
    for t in range(1, len(x)):
        expected = np.full(3, 0.01)
        expected[x[t - 1]] = 0.98
        assert pred[t] == pytest.approx(expected)
    assert np.allclose(pred[0], 1 / 3)


def test_first_order_pred_uses_train_only():
    rng = np.random.default_rng(1)
    symbols = rng.integers(-1, 2, size=200)
    other = symbols.copy()
    other[150:] = (other[150:] + 2) % 3 - 1  # só o "teste" muda
    a = ma.first_order_pred(symbols[:150], symbols + 1)
    b = ma.first_order_pred(other[:150], other + 1)
    # as previsões das posições de treino não podem mudar com o teste
    assert np.allclose(a[1:150], b[1:150])


def test_marginal_uses_train_only():
    x = np.array([0, 0, 1, 2, 2, 2, 2, 2])
    pred = ma.marginal_pred(x, 4)
    assert np.allclose(pred[0], [0.5, 0.25, 0.25])
    assert np.allclose(pred, pred[0])


# ----------------------------------------------------------------- MTD


def test_mtd_pred_matches_definition():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 3, size=60)
    pred = ma.mtd_pred(fitted_mtd(), x)
    for t in range(3, len(x)):
        assert pred[t] == pytest.approx(mixture(x, t, LAM, Q))


def test_mtd_pred_log_sum_matches_calculate_likelihood():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 3, size=200)
    model = fitted_mtd()
    pred = ma.mtd_pred(model, x)
    ll = np.log(pred[3:][np.arange(len(x) - 3), x[3:]]).sum()
    assert ll == pytest.approx(model.calculate_likelihood(to_symbols(x)))


# ----------------------------------------------------------------- HMM


def test_hmm_pred_matches_path_enumeration():
    pi = np.array([0.6, 0.4])
    A = np.array([[0.8, 0.2], [0.3, 0.7]])
    B = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    y = np.array([0, 2, 1, 2])

    def joint(seq):
        total = 0.0
        for states in itertools.product(range(2), repeat=len(seq)):
            p = pi[states[0]] * B[states[0], seq[0]]
            for i in range(1, len(seq)):
                p *= A[states[i - 1], states[i]] * B[states[i], seq[i]]
            total += p
        return total

    pred, alpha = ma.hmm_pred(pi, A, B, y)
    assert np.allclose(pred.sum(axis=1), 1)
    assert pred[0, y[0]] == pytest.approx(joint(y[:1]))
    for t in range(1, len(y)):
        assert pred[t, y[t]] == pytest.approx(joint(y[: t + 1]) / joint(y[:t]))
    assert np.allclose(alpha.sum(axis=1), 1)


# ------------------------------------------------------- estrutura de Q


def test_stationary_distribution():
    pi = ma.stationary_distribution(Q)
    assert pi.sum() == pytest.approx(1)
    assert np.all(pi > 0)
    assert pi @ Q == pytest.approx(pi)


def test_memory_order1_is_the_eigenvalue():
    mem = ma.memory_times(np.array([1.0]), Q)
    values = np.sort(np.abs(np.linalg.eigvals(Q)))[::-1][1:]
    assert mem[0]["root_modulus"] == pytest.approx(values[0])
    assert mem[0]["relaxation_steps"] == pytest.approx(-1 / math.log(values[0]))


def test_memory_matches_the_decay_of_the_exact_recursion():
    # a raiz dominante tem que ser a taxa de decaimento da marginal exata.
    # Q com autovalores distintos: com autovalor repetido o decaimento é
    # k * r^k e a razão entre passos converge devagar
    q = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    assert len(set(np.round(np.linalg.eigvals(q), 6))) == 3

    order = len(LAM)
    stationary = ma.stationary_distribution(q)
    window = np.zeros((1, order, 3))
    window[0, 0] = [1.0, 0.0, 0.0]
    window[0, 1] = [0.0, 1.0, 0.0]
    window[0, 2] = [0.0, 0.0, 1.0]

    deviations = []
    for _ in range(60):
        m = np.tensordot(LAM, window, axes=(0, 1)) @ q
        deviations.append(np.abs(m[0] - stationary).sum())
        window = np.concatenate([m[:, None, :], window[:, :-1, :]], axis=1)

    # mede a taxa antes de o desvio chegar ao piso numérico (~1e-16)
    assert deviations[40] > 1e-14
    rate = deviations[40] / deviations[39]
    expected = ma.memory_times(LAM, q)[0]["root_modulus"]
    assert rate == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------- reinícios, cache


def test_restart_summary_counts():
    class Fake:
        ll_decreased_warnings = 0
        info = {
            "restart": 1,
            "iterations": 40,
            "restarts": [
                {"ll": -100.0, "iterations": 1000, "converged": False},
                {"ll": -50.0, "iterations": 40, "converged": True},
                {"ll": -50.4, "iterations": 45, "converged": True},
            ],
        }

    r = ma.restart_summary(Fake())
    assert r["restarts"] == 3
    assert r["same_as_best"] == 2
    assert r["converged"] == 2
    assert r["worst_gap"] == pytest.approx(50.0)
    assert r["best_restart"] == 1


def sample_mtd(lam, q, n, seed):
    """Sequência do MTD (índices 0..2): sorteia o lag e depois o símbolo da linha de Q."""
    rng = np.random.default_rng(seed)
    order = len(lam)
    x = np.ones(n, dtype=int)
    for i in range(order, n):
        lag = rng.choice(order, p=lam)
        x[i] = rng.choice(3, p=q[x[i - 1 - lag]])
    return x


def test_fit_orders_cache_roundtrip(tmp_path):
    train = sample_mtd(LAM, Q, 3000, seed=0)  # índices 0..2
    train_symbols = train - 1

    first = ma.fit_orders(train_symbols, 2, 2, 0, "t", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mtd_t_o1_r2_s0.json",
        "mtd_t_o2_r2_s0.json",
    ]
    second = ma.fit_orders(train_symbols, 2, 2, 0, "t", tmp_path)
    for order in (1, 2):
        assert np.array_equal(first[order].lam, second[order].lam)
        assert np.array_equal(first[order].Q, second[order].Q)
        assert second[order].ll == pytest.approx(first[order].ll)
        assert second[order].n_params == first[order].n_params

    # o modelo restaurado prevê igual ao original
    x = train[:200]
    assert np.allclose(ma.mtd_pred(first[2], x), ma.mtd_pred(second[2], x))


# ----------------------------------------------------- ponta a ponta


def synthetic_candles(months=14, seed=0):
    rng = np.random.default_rng(seed)
    n = months * 30 * 288
    times = np.datetime64("2020-01-01T00:00") + np.arange(n) * np.timedelta64(5, "m")
    vol = np.exp(0.5 * np.sin(np.arange(n) / 700.0))  # volatilidade que oscila
    close = 100 * np.exp(np.cumsum(rng.normal(size=n) * 1e-3 * vol))
    return times, close


def test_analyze_config_smoke_and_render(tmp_path):
    times, close = synthetic_candles()
    cfg = ma.analyze_config(
        times, close, "blocks", 3, 3, 2, 0, tmp_path, tmp_path / "nope.json", "0" * 64
    )

    assert cfg["hmm_available"] is False
    names = [e["name"] for e in cfg["models"]]
    assert names == [
        "uniform",
        "marginal",
        "persistence",
        "first_order",
        "mtd1",
        "mtd2",
        "mtd3",
    ]
    by = {e["name"]: e for e in cfg["models"]}
    assert by["uniform"]["test"]["ll_per_symbol"] == pytest.approx(-math.log(3))
    assert by["uniform"]["bic"] is None
    # mais parâmetros nunca pioram o ajuste de treino (nos mesmos dados)
    assert by["mtd3"]["train"]["ll"] >= by["mtd1"]["train"]["ll"] - 1e-6
    # o MTD de ordem 1 é exatamente a cadeia de 1ª ordem
    assert by["mtd1"]["train"]["ll_per_symbol"] == pytest.approx(
        by["first_order"]["train"]["ll_per_symbol"], abs=1e-8
    )
    assert by["mtd1"]["test"]["ll_per_symbol"] == pytest.approx(
        by["first_order"]["test"]["ll_per_symbol"], abs=1e-8
    )
    assert by["mtd1"]["n_params"] == 6 and by["first_order"]["n_params"] == 6
    # a persistência com ε = 0,01 é pior que o uniforme em LL
    assert by["persistence"]["test"]["ll_per_symbol"] < -math.log(3)

    assert cfg["gaps"][1]["crossing"] == 0  # série sintética sem lacunas

    report = ma.jsonable(
        {
            "generated_at": "x",
            "source": {"file": "f", "sha256": "0" * 64, "rows": len(times)},
            "max_order": 3,
            "restarts": 2,
            "seed": 0,
            "configs": [cfg],
        }
    )
    json.dumps(report)  # serializa
    text = ma.render_markdown(report)
    for fragment in (
        "Cadeia de 1ª ordem",
        "Persistência",
        "MTD ordem 3",
        "Lacunas",
        "Matrizes de resumo",
        "Tabela completa",
        "Matriz de confusão",
        "Precisão do 0",
    ):
        assert fragment in text
    # métricas fora do escopo não aparecem no relatório nem no JSON
    for removed in ("AUC", "Brier", "Calibração", "Horizonte", "Sazonalidade", "Adequação"):
        assert removed not in text
    for key in ("horizon", "calibration", "seasonal", "adequacy"):
        assert key not in cfg
    for e in cfg["models"]:
        assert "auc_movement" not in e["test"] and "brier" not in e["test"]


# --------------------------------------------- reaproveitar relatório anterior


SOURCE = {"sha256": "a" * 64}


def previous_report(tmp_path, **overrides):
    report = {
        "source": SOURCE,
        "max_order": 10,
        "restarts": 10,
        "seed": 0,
        "configs": [
            {"mode": "blocks", "k": 3, "marker": "b3"},
            {"mode": "global", "k": 3, "marker": "g3"},
        ],
    }
    report.update(overrides)
    path = tmp_path / "previous.json"
    path.write_text(json.dumps(report))
    return path


def test_load_reusable_indexes_configs_by_mode_and_k(tmp_path):
    reusable = ma.load_reusable(previous_report(tmp_path), SOURCE, 10, 10, 0)
    # relatórios antigos não gravavam o intervalo: blocks = 1 mês, global = sem blocos
    assert set(reusable) == {("blocks", 3, 1), ("global", 3, None)}
    assert reusable[("global", 3, None)]["marker"] == "g3"


def test_config_interval_defaults_and_explicit_values():
    assert ma.config_interval({"mode": "global", "k": 3}) is None
    assert ma.config_interval({"mode": "global", "k": 3, "interval_months": 6}) is None
    assert ma.config_interval({"mode": "blocks", "k": 3}) == 1
    assert ma.config_interval({"mode": "blocks", "k": 3, "interval_months": 1}) == 1
    assert ma.config_interval({"mode": "blocks", "k": 3, "interval_months": 6}) == 6


def test_analyze_config_with_another_block_size(tmp_path):
    times, close = synthetic_candles(months=14)
    cfg = ma.analyze_config(
        times, close, "blocks", 3, 2, 2, 0, tmp_path, None, "0" * 64,
        interval_months=3,
    )
    assert cfg["interval_months"] == 3
    # blocos maiores têm um primeiro bloco maior sem símbolos: menos símbolos de treino
    base = ma.analyze_config(
        times, close, "blocks", 3, 2, 2, 0, tmp_path, None, "0" * 64
    )
    assert base["interval_months"] == 1
    assert cfg["n_train"] < base["n_train"]
    assert cfg["n_test"] == base["n_test"]
    # o cache de 1 mês mantém o nome antigo; o de 3 meses inclui o intervalo
    names = sorted(p.name for p in tmp_path.iterdir())
    assert any("_i3_" in n for n in names)
    assert any("blocks_k3_" in n and "_i" not in n for n in names)
    title = ma.render_markdown(
        ma.jsonable(
            {
                "generated_at": "x",
                "source": {"file": "f", "sha256": "0" * 64, "rows": len(times)},
                "max_order": 2,
                "restarts": 2,
                "seed": 0,
                "configs": [cfg],
            }
        ),
        detail=("blocks:3:3",),
    )
    assert "blocos de 3 meses" in title


@pytest.mark.parametrize(
    "args",
    [(SOURCE, 6, 10, 0), (SOURCE, 10, 5, 0), (SOURCE, 10, 10, 1)],
)
def test_load_reusable_refuses_other_mtd_parameters(tmp_path, args):
    with pytest.raises(ValueError):
        ma.load_reusable(previous_report(tmp_path), *args)


def test_load_reusable_refuses_other_snapshot(tmp_path):
    with pytest.raises(ValueError):
        ma.load_reusable(previous_report(tmp_path), {"sha256": "b" * 64}, 10, 10, 0)


# ------------------------------------------------------------------ resumo


def fake_config(mode, k, zeros_train, zeros_test, rates):
    """Configuração mínima: só o que render_summary lê."""

    def direction(rate):
        return {"rate": rate, "ci": [rate - 0.006, rate + 0.006]}

    def mtd(order, bic, ll, rate):
        return {
            "name": f"mtd{order}",
            "bic": bic,
            "gain_vs_marginal_test": 0.05,
            "test": {
                "ll_per_symbol": ll,
                "accuracy": 0.4711,
                "direction": direction(rate),
                "confusion": {
                    "per_class": [
                        {"f1": 0.2},  # −1
                        {"f1": 0.6},  # 0
                        {"f1": 0.4},  # +1
                    ]
                },
            },
        }

    return {
        "mode": mode,
        "k": k,
        "class_share_train": [0.3, zeros_train, 0.3],
        "class_share_test": [0.3, zeros_test, 0.3],
        "naive_accuracy_test": {"always_zero": 0.4383},
        "models": [
            {"name": "marginal", "bic": 1.0, "test": {"direction": direction(0.4914)}},
            mtd(2, 200.0, -1.05, rates[0]),
            mtd(10, 100.0, -1.02, rates[1]),  # menor BIC
        ],
    }


def test_render_summary_picks_the_lowest_bic_order_and_best_constant():
    configs = [
        fake_config("blocks", 3, 0.4233, 0.4383, (0.5153, 0.5117)),
        fake_config("global", 4, 0.4135, 0.5745, (0.5000, 0.5050)),
    ]
    lines = ma.render_summary(configs)
    text = "\n".join(lines)
    rows = [l for l in lines if l.startswith("| blocks") or l.startswith("| global")]
    assert len(rows) == 2

    first = rows[0].split("|")
    assert first[1].strip() == "blocks" and first[2].strip() == "1"  # bloco de 1 mês
    assert first[3].strip() == "3"  # s/k
    assert first[6].strip() == "10"  # ordem de menor BIC
    assert "51,17%" in rows[0]  # direção da ordem 10, e não a da ordem 2
    assert "50,86%" in rows[0]  # melhor constante = max(0,4914; 0,5086)
    assert "42,33%" in rows[0] and "43,83%" in rows[0]
    assert "global" in rows[1] and "57,45%" in rows[1]
    assert rows[1].split("|")[2].strip() == "—"  # global não tem blocos
    assert "47,11% (43,83%)" in rows[0]  # acurácia do MTD e a de "sempre 0"
    assert "0,300" in rows[0]  # F1 de ±1 = média de 0,2 e 0,4
    assert "AUC" not in text
    assert "Atenção" in text


# ------------------------------------------------- matriz de confusão


def test_confusion_metrics_by_hand():
    # alvos:    -1  0 +1  0  0 -1   (índices 0, 1, 2, 1, 1, 0)
    # previsto: -1  0  0  0 +1  0
    target = np.array([0, 1, 2, 1, 1, 0])
    chosen = np.array([0, 1, 1, 1, 2, 1])
    r = ma.confusion_metrics(chosen, target)

    assert r["matrix"] == [[1, 1, 0], [0, 2, 1], [0, 1, 0]]
    minus, zero, plus = r["per_class"]
    assert (minus["precision"], minus["recall"]) == (1.0, 0.5)
    assert minus["f1"] == pytest.approx(2 / 3)
    assert zero["precision"] == pytest.approx(2 / 4)
    assert zero["recall"] == pytest.approx(2 / 3)
    assert zero["f1"] == pytest.approx(4 / 7)
    assert (plus["precision"], plus["recall"], plus["f1"]) == (0.0, 0.0, 0.0)
    assert r["macro_f1"] == pytest.approx((2 / 3 + 4 / 7 + 0) / 3)

    m = r["movement"]
    assert m["base_zero"] == pytest.approx(3 / 6)
    assert m["predicted_zero"] == 4 and m["predicted_move"] == 2
    assert m["precision_zero"] == pytest.approx(2 / 4)
    assert m["recall_zero"] == pytest.approx(2 / 3)
    # dos 2 previstos como movimento (−1 certo; +1 era um 0), 1 foi movimento real
    assert m["precision_move"] == pytest.approx(1 / 2)
    assert m["recall_move"] == pytest.approx(1 / 3)  # 3 movimentos reais, 1 achado


def test_confusion_metrics_never_predicting_a_class():
    # o "sempre 0": precisão do −1 e do +1 não existem (None) e o F1 vale 0
    target = np.array([0, 1, 2, 1])
    r = ma.confusion_metrics(np.ones(4, dtype=int), target)
    minus, zero, plus = r["per_class"]
    assert minus["precision"] is None and minus["recall"] == 0.0 and minus["f1"] == 0.0
    assert plus["precision"] is None and plus["f1"] == 0.0
    assert zero["precision"] == pytest.approx(0.5) and zero["recall"] == 1.0
    assert r["movement"]["precision_move"] is None
    assert r["movement"]["recall_move"] == 0.0


def test_confusion_is_consistent_with_accuracy_and_zero_share():
    rng = np.random.default_rng(10)
    x = rng.integers(0, 3, size=500)
    pred = rng.dirichlet(np.ones(3), size=500)
    rows = np.arange(1, 500)
    r = ma.evaluate_predictions(pred, x, rows)
    c = r["confusion"]
    cm = np.array(c["matrix"])
    assert cm.sum() == r["n"]
    assert np.trace(cm) / cm.sum() == pytest.approx(r["accuracy"])
    assert c["movement"]["predicted_zero"] / r["n"] == pytest.approx(r["zero_share"])
    # a linha do real soma o suporte; a coluna soma o previsto
    for k, row, col in zip(c["per_class"], cm, cm.T):
        assert k["support"] == row.sum() and k["predicted"] == col.sum()


def test_load_reusable_skips_configs_without_confusion(tmp_path):
    report = {
        "source": SOURCE,
        "max_order": 10,
        "restarts": 10,
        "seed": 0,
        "configs": [
            {"mode": "blocks", "k": 3, "models": [{"name": "mtd1", "test": {}}]},
            {
                "mode": "global",
                "k": 3,
                "models": [{"name": "mtd1", "test": {"confusion": {}}}],
            },
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(report))
    reusable = ma.load_reusable(path, SOURCE, 10, 10, 0)
    assert set(reusable) == {("global", 3, None)}  # a antiga é recalculada


# ----------------------------------------- métricas removidas do reaproveitamento


def old_config_with_removed_metrics():
    return {
        "mode": "blocks",
        "k": 3,
        "horizon": {"mtd1": {}},
        "calibration": {},
        "seasonal": {},
        "adequacy": {},
        "models": [
            {
                "name": "mtd1",
                "train": {"ll": -1.0, "auc_movement": 0.6, "brier": 0.6},
                "test": {"ll": -1.0, "auc_movement": 0.6, "brier": 0.6, "confusion": {}},
            }
        ],
    }


def test_drop_removed_metrics_keeps_the_rest():
    original = old_config_with_removed_metrics()
    cfg = ma.drop_removed_metrics(original)
    for key in ("horizon", "calibration", "seasonal", "adequacy"):
        assert key not in cfg
    model = cfg["models"][0]
    for part in ("train", "test"):
        assert "auc_movement" not in model[part] and "brier" not in model[part]
        assert model[part]["ll"] == -1.0
    assert model["test"]["confusion"] == {}
    assert cfg["mode"] == "blocks" and cfg["k"] == 3
    # não altera o original
    assert "auc_movement" in original["models"][0]["test"]


def test_load_reusable_returns_configs_without_removed_metrics(tmp_path):
    report = {
        "source": SOURCE,
        "max_order": 10,
        "restarts": 10,
        "seed": 0,
        "configs": [old_config_with_removed_metrics()],
    }
    path = tmp_path / "old_with_confusion.json"
    path.write_text(json.dumps(report))
    cfg = ma.load_reusable(path, SOURCE, 10, 10, 0)[("blocks", 3, 1)]
    assert "horizon" not in cfg and "auc_movement" not in cfg["models"][0]["test"]


# ----------------------------------------------------- relatório final (matrizes)


def test_parse_detail_and_is_detail():
    keys = ma.parse_detail(["blocks:3:1", "global:3", "blocks:5"])
    assert keys == {("blocks", 3, 1), ("global", 3, None), ("blocks", 5, 1)}
    assert ma.parse_detail(["global:4:6"]) == {("global", 4, None)}  # global ignora o bloco
    for bad in ("blocks", "mode:3", "blocks:3:1:9", "blocks:x"):
        with pytest.raises(ValueError):
            ma.parse_detail([bad])

    keys = {("blocks", 3, 1), ("global", 3, None)}
    assert ma.is_detail({"mode": "blocks", "k": 3}, keys)  # sem campo = bloco de 1 mês
    assert ma.is_detail({"mode": "blocks", "k": 3, "interval_months": 1}, keys)
    assert not ma.is_detail({"mode": "blocks", "k": 3, "interval_months": 2}, keys)
    assert ma.is_detail({"mode": "global", "k": 3}, keys)
    assert not ma.is_detail({"mode": "global", "k": 4}, keys)


def test_signed_uses_comma_and_explicit_sign():
    assert ma.signed(0.954, 2) == "+0,95"
    assert ma.signed(-0.081, 2) == "-0,08"
    assert ma.signed(3.3, 1) == "+3,3"
    assert ma.signed(0, 1) == "+0,0"


def test_months_label_singular_and_plural():
    assert ma.months_label(1) == "1 mês"
    assert ma.months_label(12) == "12 meses"


def grid_config(mode, k, interval, rate=0.5117, lower=0.5058, zeros=(0.42, 0.44)):
    """Configuração com o que as matrizes de resumo leem."""

    def entry(name, bic, ll, gain):
        return {
            "name": name,
            "bic": bic,
            "gain_vs_marginal_test": gain,
            "test": {
                "ll_per_symbol": ll,
                "accuracy": 0.47,
                "direction": {"rate": rate, "ci": [lower, rate + 0.006]},
                "confusion": {"per_class": [{"f1": 0.3}, {"f1": 0.6}, {"f1": 0.1}]},
            },
        }

    cfg = {
        "mode": mode,
        "k": k,
        "class_share_train": [0.3, zeros[0], 0.3],
        "class_share_test": [0.3, zeros[1], 0.3],
        "naive_accuracy_test": {"always_zero": 0.44},
        "models": [
            {"name": "marginal", "bic": 1.0, "test": {"direction": {"rate": 0.4914, "ci": [0, 1]}}},
            entry("mtd2", 200.0, -1.05, 0.04),
            entry("mtd10", 100.0, -1.02, 0.05),
        ],
        "structure": {
            "2": {"restart_summary": {"same_as_best": 10, "converged": 10, "worst_gap": 0.01}},
            "10": {"restart_summary": {"same_as_best": 10, "converged": 10, "worst_gap": 0.02}},
        },
        "gaps": {"2": {"ll_per_symbol_all": -1.0, "ll_per_symbol_clean": -1.00001}},
    }
    if mode == "blocks":
        cfg["interval_months"] = interval
    return cfg


def test_render_grids_layout_order_and_missing_cells():
    configs = [
        grid_config("blocks", 3, 2),
        grid_config("global", 3, None),
        grid_config("blocks", 1, 1),
        grid_config("blocks", 3, 1),
    ]
    lines = ma.render_grids(configs)
    text = "\n".join(lines)

    # uma matriz por medida
    assert text.count("### ") == len(ma.GRIDS)

    zeros = lines[lines.index("### Zeros: treino → teste") + 2 :]
    header = zeros[0].split("|")
    labels = [c.strip() for c in header[2:-1]]
    assert labels == ["*Global*", "*Blocks* 1 mês", "*Blocks* 2 meses"]  # global primeiro, blocos crescentes
    rows = [l for l in zeros[2:6] if l.startswith("| s/")]
    assert [r.split("|")[1].strip() for r in rows] == ["s/1", "s/3"]  # k crescente
    cells_k1 = [c.strip() for c in rows[0].split("|")[2:-1]]
    assert cells_k1[0] == "—" and cells_k1[2] == "—"  # k=1 só existe com bloco de 1 mês
    assert cells_k1[1] == "42,00% → 44,00%"


def test_margin_cell_marks_ci_above_the_best_constant():
    # melhor constante = max(0,4914; 0,5086) = 0,5086
    above = grid_config("blocks", 3, 1, rate=0.5167, lower=0.5105)
    entry = ma.best_mtd(above)
    assert ma.best_constant(above) == pytest.approx(0.5086)
    assert ma.margin_cell(above, entry) == "+0,81*"  # IC inferior 0,5105 > 0,5086
    inside = grid_config("blocks", 3, 1, rate=0.5117, lower=0.5058)
    assert ma.margin_cell(inside, ma.best_mtd(inside)) == "+0,31"  # IC cruza a constante


def test_render_checks_reports_exceptions():
    ok = [grid_config("blocks", 3, 1), grid_config("global", 3, None)]
    text = "\n".join(ma.render_checks(ok, max_order=10, restarts=10))
    assert "**Configurações calculadas:** 2" in text
    assert "todas as 2 configurações" in text
    assert "todos chegam à mesma solução" in text and "todos convergiram" in text

    bad = grid_config("blocks", 4, 1)
    bad["models"][2]["bic"] = 300.0  # agora o menor BIC é a ordem 2
    bad["structure"]["10"]["restart_summary"]["same_as_best"] = 7
    bad["structure"]["10"]["restart_summary"]["converged"] = 8
    text = "\n".join(ma.render_checks(ok + [bad], max_order=10, restarts=10))
    assert "2 de 3" in text and "blocks 1m s/4" in text
    assert "1 casos sem concordância" in text and "ordem 10" in text
    assert "nem todos convergiram" in text


def test_render_markdown_details_only_for_selected_configs(tmp_path):
    times, close = synthetic_candles(months=14)
    configs = [
        ma.analyze_config(times, close, "blocks", 3, 2, 2, 0, tmp_path, None, "0" * 64, interval_months=i)
        for i in (1, 2)
    ]
    report = ma.jsonable(
        {
            "generated_at": "x",
            "source": {"file": "f", "sha256": "0" * 64, "rows": len(times)},
            "max_order": 2,
            "restarts": 2,
            "seed": 0,
            "configs": configs,
        }
    )
    text = ma.render_markdown(report, detail=("blocks:3:1",))
    assert "## Discretização `blocks`, s/3, blocos de 1 mês" in text
    assert "blocos de 2 meses" not in text  # sem seção detalhada
    assert "## Configurações sem seção detalhada" in text
    assert "blocks 2m s/3" in text.split("## Configurações sem seção detalhada")[1]
    assert "## Matrizes de resumo" in text and "## Verificações em todas as configurações" in text
    assert "## Tabela completa" in text
    # sem nenhuma seleção válida, não há seção detalhada
    only_tables = ma.render_markdown(report, detail=("global:9",))
    assert "## Discretização" not in only_tables
