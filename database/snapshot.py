import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from database.index import close_connection, create_connection

RAIZ = Path(__file__).resolve().parent.parent
SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"
PASTA_PADRAO = RAIZ / "snapshot"

TABELA = "btc_usdt"
TAMANHO_LOTE = 100_000

# Grupos de linhas maiores comprimem melhor: com 100 mil linhas o arquivo fica com
# ~69 MiB e com 500 mil, ~46 MiB (zstd 19, medido em 957 mil linhas). Os lotes
# lidos do banco sao juntados ate este tamanho antes de gravar
GRUPO_LINHAS = 500_000
NIVEL_ZSTD = 19

# Mesma ordem e tipos de database/schema.sql. A ordem importa: o restore usa COPY
SCHEMA = pa.schema(
    [
        pa.field("open_time", pa.timestamp("us"), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("close_time", pa.timestamp("us")),
        pa.field("quote_volume", pa.float64()),
        pa.field("trades", pa.int32()),
        pa.field("taker_buy_base", pa.float64()),
        pa.field("taker_buy_quote", pa.float64()),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
    ]
)
COLUNAS = tuple(SCHEMA.names)
COLUNAS_SQL = ", ".join(f'"{coluna}"' for coluna in COLUNAS)


def sha256_arquivo(caminho):
    resumo = hashlib.sha256()

    with open(caminho, "rb") as arquivo:
        while bloco := arquivo.read(1024 * 1024):
            resumo.update(bloco)

    return resumo.hexdigest()


def caminho_manifesto(parquet):
    parquet = Path(parquet)
    return parquet.with_name(parquet.stem + ".manifest.json")


def _gravar_pendentes(escritor, pendentes):
    if pendentes:
        escritor.write_table(pa.concat_tables(pendentes))
        pendentes.clear()


# Escreve os lotes (listas de tuplas na ordem de COLUNAS) em um unico Parquet.
# Devolve a quantidade de linhas e o primeiro e o ultimo open_time
def write_parquet(lotes, caminho):
    linhas = 0
    primeiro = ultimo = None
    pendentes = []

    with pq.ParquetWriter(
        caminho, SCHEMA, compression="zstd", compression_level=NIVEL_ZSTD
    ) as escritor:
        for lote in lotes:
            colunas = [
                pa.array(valores, type=campo.type)
                for valores, campo in zip(zip(*lote), SCHEMA)
            ]
            pendentes.append(pa.Table.from_arrays(colunas, schema=SCHEMA))

            linhas += len(lote)
            if primeiro is None:
                primeiro = lote[0][0]
            ultimo = lote[-1][0]

            if sum(tabela.num_rows for tabela in pendentes) >= GRUPO_LINHAS:
                _gravar_pendentes(escritor, pendentes)

        _gravar_pendentes(escritor, pendentes)

    return linhas, primeiro, ultimo


def verify_checksum(parquet):
    manifesto_path = caminho_manifesto(parquet)
    if not manifesto_path.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado: {manifesto_path}")

    manifesto = json.loads(manifesto_path.read_text())

    obtido = sha256_arquivo(parquet)
    if obtido != manifesto["sha256"]:
        raise ValueError(
            f"Checksum diferente do manifesto para {Path(parquet).name}: "
            f"esperado {manifesto['sha256']}, obtido {obtido}"
        )

    return manifesto


# Exporta btc_usdt ate a data final (exclusiva), em modo somente leitura.
# A data final e obrigatoria: sem ela o resultado mudaria a cada execucao, ja que
# a coleta continua
def export_snapshot(conn, ate, pasta):
    if ate > datetime.now(timezone.utc).date():
        raise ValueError(f"A data final {ate} esta no futuro: o snapshot nao seria fixo")

    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    parquet = pasta / f"{TABELA}_{ate.isoformat()}.parquet"

    conn.read_only = True

    def lotes():
        with conn.cursor(name="snapshot_export") as cur:
            cur.execute(
                f'SELECT {COLUNAS_SQL} FROM {TABELA} WHERE open_time < %s '
                'ORDER BY open_time, symbol, "interval"',
                (ate,),
            )
            while lote := cur.fetchmany(TAMANHO_LOTE):
                yield lote

    linhas, primeiro, ultimo = write_parquet(lotes(), parquet)

    if linhas == 0:
        parquet.unlink()
        raise ValueError(f"Nenhuma linha de {TABELA} anterior a {ate}")

    manifesto = {
        "tabela": TABELA,
        "ate_exclusivo": ate.isoformat(),
        "linhas": linhas,
        "primeiro_open_time": primeiro.isoformat(sep=" "),
        "ultimo_open_time": ultimo.isoformat(sep=" "),
        "colunas": list(COLUNAS),
        "formato": "parquet/zstd",
        "sha256": sha256_arquivo(parquet),
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    caminho_manifesto(parquet).write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n"
    )

    return parquet, manifesto


# Carrega o Parquet em btc_usdt. So aceita tabela vazia: create_connection usa o
# config.ini, que pode apontar para a producao, e assim nada existente e alterado
def restore_snapshot(conn, parquet):
    manifesto = verify_checksum(parquet)

    arquivo = pq.ParquetFile(parquet)
    if not arquivo.schema_arrow.equals(SCHEMA):
        raise ValueError(f"O esquema de {Path(parquet).name} difere do esperado")

    with conn.transaction():
        conn.execute(SCHEMA_SQL.read_text())

        existentes = conn.execute(f"SELECT count(*) FROM {TABELA}").fetchone()[0]
        if existentes:
            raise RuntimeError(
                f"{TABELA} ja tem {existentes} linhas: o restore so carrega em tabela vazia"
            )

        with conn.cursor() as cur:
            with cur.copy(f"COPY {TABELA} ({COLUNAS_SQL}) FROM STDIN") as copia:
                for lote in arquivo.iter_batches(batch_size=TAMANHO_LOTE):
                    for linha in zip(*(coluna.to_pylist() for coluna in lote.columns)):
                        copia.write_row(linha)

        total, primeiro, ultimo = conn.execute(
            f"SELECT count(*), min(open_time), max(open_time) FROM {TABELA}"
        ).fetchone()

        # a excecao desfaz a transacao inteira
        if (
            total != manifesto["linhas"]
            or primeiro.isoformat(sep=" ") != manifesto["primeiro_open_time"]
            or ultimo.isoformat(sep=" ") != manifesto["ultimo_open_time"]
        ):
            raise RuntimeError(
                "O banco restaurado nao confere com o manifesto: "
                f"{total} linhas de {primeiro} a {ultimo}"
            )

    return manifesto


def main():
    parser = argparse.ArgumentParser(description="Snapshot da tabela btc_usdt em Parquet")
    comandos = parser.add_subparsers(dest="comando", required=True)

    exportar = comandos.add_parser("export", help="gera o Parquet e o manifesto")
    exportar.add_argument(
        "--ate",
        required=True,
        type=date.fromisoformat,
        help="data final EXCLUSIVA, AAAA-MM-DD (candles com open_time anterior a ela)",
    )
    exportar.add_argument("--saida", type=Path, default=PASTA_PADRAO)

    restaurar = comandos.add_parser("restore", help="carrega o Parquet em btc_usdt vazia")
    restaurar.add_argument("parquet", type=Path)

    args = parser.parse_args()

    conn = create_connection()
    try:
        if args.comando == "export":
            parquet, manifesto = export_snapshot(conn, args.ate, args.saida)
            tamanho = parquet.stat().st_size / 1024 / 1024
            print(f"{parquet} ({tamanho:.1f} MiB)")
        else:
            manifesto = restore_snapshot(conn, args.parquet)
            print(f"Restauradas {manifesto['linhas']} linhas em {TABELA}")

        print(json.dumps(manifesto, indent=2, ensure_ascii=False))
    finally:
        close_connection(conn)


if __name__ == "__main__":
    main()
