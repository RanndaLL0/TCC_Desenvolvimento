import json
from pathlib import Path

import pytest

from src.experiments import mtd_thesis_tables as t

REPORT = Path(__file__).resolve().parents[1] / "relatorios" / "mtd_2026-10-01_completa_final.json"

needs_report = pytest.mark.skipif(not REPORT.exists(), reason="relatório final ausente")


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_number_format_is_brazilian():
    assert t.num(1234567.891, 1) == "1.234.567,9"
    assert t.num(0.05176, 4) == "0,0518"
    assert t.pct(0.5117) == "51,17\\%"
    assert t.pp(0.0082) == "+0,82"
    assert t.pp(-0.0385) == "-3,85"


def test_one_sided_p_is_half_at_the_null_and_small_far_above():
    assert t.one_sided_p(500, 1000, 0.5) == pytest.approx(0.5)
    assert t.one_sided_p(560, 1000, 0.5) < 1e-3
    assert t.one_sided_p(440, 1000, 0.5) > 0.999


@needs_report
def test_config_metrics_match_the_markdown_report(report):
    # valores impressos em mtd_2026-10-01_completa_final.md (blocks, 1 mês, s/3)
    m = t.config_metrics(t.find(report, "blocks", 1, 3))
    assert round(100 * m["direction"], 2) == 51.17
    assert round(100 * m["constant"], 2) == 50.86
    assert round(m["gain_iid"], 5) == 0.05175
    assert round(100 * m["accuracy"], 2) == 47.11
    assert round(100 * m["majority"], 2) == 43.83


@needs_report
def test_best_mtd_is_order_ten_everywhere(report):
    assert {t.best_mtd(c)["name"] for c in report["configs"]} == {"mtd10"}


@needs_report
def test_hmm_row_matches_the_report(report):
    cfg = t.find(report, "blocks", 1, 3)
    hmm = t.model_of(cfg, "hmm3")
    assert hmm["n_params"] == 14
    assert round(hmm["test"]["ll_per_symbol"], 5) == -1.01545


@needs_report
def test_best_constant_is_at_least_half(report):
    for cfg in report["configs"]:
        assert t.best_constant(t.best_mtd(cfg)) >= 0.5


@needs_report
def test_robustness_numbers_match_the_report(report):
    tex = t.table_robustness(report)
    assert "6.000 de 6.000" in tex
    assert "0,108 nats" in tex
    assert "0,000116" in tex


@needs_report
def test_build_has_five_balanced_tables(report):
    tex = t.build(report)
    assert tex.count("\\begin{table}") == tex.count("\\end{table}") == 5
    for label in ("ordem", "resumo", "volatilidade_direcao", "estacionariedade", "robustez"):
        assert f"\\label{{tab:mtd_{label}}}" in tex
    assert "nan" not in tex.lower().replace("\\textit", "")
