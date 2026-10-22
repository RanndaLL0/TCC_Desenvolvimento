import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from database import snapshot


def candle(i):
    t = datetime(2020, 1, 1) + timedelta(minutes=5 * i)
    return (
        t,
        100.0 + i,
        101.0 + i,
        99.0 + i,
        100.5 + i,
        1.5,
        t + timedelta(minutes=5),
        150.25,
        10 + i,
        0.7,
        70.1,
        "BTCUSDT",
        "5m",
    )


class FakeCursor:
    def __init__(self, lotes, executed):
        self.lotes = list(lotes)
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchmany(self, size):
        return self.lotes.pop(0) if self.lotes else []


class FakeConn:
    def __init__(self, lotes):
        self.lotes = lotes
        self.executed = []
        self.read_only = False
        self.cursor_names = []

    def cursor(self, name=None):
        self.cursor_names.append(name)
        return FakeCursor(self.lotes, self.executed)


PASSADO = date(2026, 1, 1)


def test_parquet_round_trip_keeps_values_and_schema(tmp_path):
    lotes = [[candle(i) for i in range(0, 3)], [candle(i) for i in range(3, 5)]]
    caminho = tmp_path / "x.parquet"

    linhas, primeiro, ultimo = snapshot.write_parquet(lotes, caminho)

    assert linhas == 5
    assert primeiro == candle(0)[0]
    assert ultimo == candle(4)[0]

    lido = pq.ParquetFile(caminho)
    assert lido.schema_arrow.equals(snapshot.SCHEMA)

    tabela = lido.read()
    volta = list(zip(*(coluna.to_pylist() for coluna in tabela.columns)))
    assert volta == [candle(i) for i in range(5)]


def test_batches_are_merged_into_larger_row_groups_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "GRUPO_LINHAS", 4)
    # lotes de 3, 2, 3 e 1 linhas: descarrega ao chegar a 5 (3+2) e a 4 (3+1)
    tamanhos = [3, 2, 3, 1]
    inicio = [sum(tamanhos[:i]) for i in range(len(tamanhos))]
    lotes = [[candle(i) for i in range(a, a + n)] for a, n in zip(inicio, tamanhos)]
    caminho = tmp_path / "x.parquet"

    linhas, _, _ = snapshot.write_parquet(lotes, caminho)

    lido = pq.ParquetFile(caminho)
    assert linhas == 9
    assert [lido.metadata.row_group(i).num_rows for i in range(lido.num_row_groups)] == [5, 4]

    tabela = lido.read()
    assert list(zip(*(c.to_pylist() for c in tabela.columns))) == [candle(i) for i in range(9)]


def test_sha256_changes_when_the_file_changes(tmp_path):
    caminho = tmp_path / "a.bin"
    caminho.write_bytes(b"abc")
    antes = snapshot.sha256_arquivo(caminho)

    caminho.write_bytes(b"abd")

    assert snapshot.sha256_arquivo(caminho) != antes
    assert len(antes) == 64


def test_export_writes_parquet_and_manifest(tmp_path):
    conn = FakeConn([[candle(i) for i in range(4)], [candle(4), candle(5)]])

    parquet, manifesto = snapshot.export_snapshot(conn, PASSADO, tmp_path)

    assert parquet.name == "btc_usdt_2026-01-01.parquet"
    assert conn.read_only is True
    assert conn.cursor_names == ["snapshot_export"]  # cursor no servidor, sem carregar tudo
    assert conn.executed[0][1] == (PASSADO,)

    assert manifesto["linhas"] == 6
    assert manifesto["ate_exclusivo"] == "2026-01-01"
    assert manifesto["primeiro_open_time"] == "2020-01-01 00:00:00"
    assert manifesto["ultimo_open_time"] == "2020-01-01 00:25:00"
    assert manifesto["sha256"] == snapshot.sha256_arquivo(parquet)
    assert json.loads(snapshot.caminho_manifesto(parquet).read_text()) == manifesto


def test_export_without_rows_raises_and_leaves_no_file(tmp_path):
    with pytest.raises(ValueError, match="Nenhuma linha"):
        snapshot.export_snapshot(FakeConn([]), PASSADO, tmp_path)

    assert list(tmp_path.glob("*.parquet")) == []


def test_export_refuses_a_date_in_the_future(tmp_path):
    futuro = date.today() + timedelta(days=2)

    # a validação vem antes de qualquer acesso ao banco: conn nem é usado
    with pytest.raises(ValueError, match="futuro"):
        snapshot.export_snapshot(None, futuro, tmp_path)


def test_verify_checksum_accepts_an_intact_snapshot(tmp_path):
    parquet, manifesto = snapshot.export_snapshot(FakeConn([[candle(0)]]), PASSADO, tmp_path)

    assert snapshot.verify_checksum(parquet) == manifesto


def test_verify_checksum_detects_a_corrupted_file(tmp_path):
    parquet, _ = snapshot.export_snapshot(FakeConn([[candle(0)]]), PASSADO, tmp_path)

    dados = bytearray(parquet.read_bytes())
    dados[len(dados) // 2] ^= 0xFF
    parquet.write_bytes(bytes(dados))

    with pytest.raises(ValueError, match="Checksum"):
        snapshot.verify_checksum(parquet)


def test_verify_checksum_requires_the_manifest(tmp_path):
    parquet, _ = snapshot.export_snapshot(FakeConn([[candle(0)]]), PASSADO, tmp_path)
    snapshot.caminho_manifesto(parquet).unlink()

    with pytest.raises(FileNotFoundError):
        snapshot.verify_checksum(parquet)


def test_restore_checks_the_checksum_before_touching_the_database(tmp_path):
    parquet, _ = snapshot.export_snapshot(FakeConn([[candle(0)]]), PASSADO, tmp_path)
    parquet.write_bytes(parquet.read_bytes() + b"x")

    # conn=None: qualquer acesso ao banco levantaria AttributeError, não ValueError
    with pytest.raises(ValueError, match="Checksum"):
        snapshot.restore_snapshot(None, parquet)


def test_columns_follow_the_database_schema_order():
    assert snapshot.COLUNAS == (
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
        "symbol", "interval",
    )
