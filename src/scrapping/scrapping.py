import json
import configparser
import os
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE_URL = 'https://api.binance.com/api/v3/klines'
LIMITE_REQUISICAO = 1000
FECHAMENTO = 300_000

# A nomenclatura das colunas segue a mesma da API
CABECALHO = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
]

SYMBOL = "BTCUSDT"
INTERVALO = "5m"
INICIO = "2017-08-17"

# Evitei ao maximo de utilizar bibliotecas externas como o PSYCOPG
# Entao as conexões e os comandos executados no banco pelo scrapping sao todos
# Comandos nativos que rodam no psql
config = configparser.ConfigParser(inline_comment_prefixes=None)
config.read("../config.ini")

#O arquivo de configuracoes TOML nao foi posto no repositorio
PG_HOST = config.get("postgres", "host")
PG_PORT = config.get("postgres", "port")
PG_USER = config.get("postgres", "user")
PG_PASSWORD = config.get("postgres", "password")
PG_DB = config.get("postgres", "database")

def rodar_sql(comando_poggers):
    env = os.environ.copy()
    env["PGPASSWORD"] = PG_PASSWORD

    try:
        subprocess.run(
            [
                "psql",
                "-h", PG_HOST,
                "-p", PG_PORT,
                "-U", PG_USER,
                "-d", PG_DB,
                "-v", "ON ERROR_STOP=1",
                "-c", comando_poggers,
            ],
            env = env,
            check = True,
            capture_output = True,
            text = True
        )
    except subprocess.CalledProcessError as e:
        print(f"{e.stderr}")
        raise e


# Como foi removido o teste com o CSV estas funcoes serao removidas no futuro
# No momento as mesmas servem apenas para o comparativo de tempo
def agora_milisegundos() -> int:
    return int(time.time() * 1000)


def para_ms(texto):
    dt = datetime.strptime(texto, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

#Utilitario para formatar a data no formato de TIMESTAMP no banco
def para_iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def buscar(inicio_ms):
    parametros = urllib.parse.urlencode({
        "symbol": SYMBOL,
        "interval": INTERVALO,
        "startTime": inicio_ms,
        "limit": LIMITE_REQUISICAO,
    })
    with urllib.request.urlopen(f"{BASE_URL}?{parametros}", timeout=30) as resposta:
        return json.loads(resposta.read())

def inserir_batch(linhas_poggers):
    env = os.environ.copy()
    env["PGPASSWORD"] = PG_PASSWORD

    script_sql = (
        "CREATE TEMP TABLE tmp_candles AS SELECT * FROM btc_usdt WITH NO DATA;\n"
        "\\copy tmp_candles FROM STDIN WITH (FORMAT text, DELIMITER E'\\t');\n"
        f"{linhas_poggers}\n"
        "\\.\n"
        "INSERT INTO btc_usdt SELECT * FROM tmp_candles "
        "ON CONFLICT (symbol, interval, open_time) DO NOTHING;\n"
    )

    subprocess.run(
        [
            "psql",
            "-h", PG_HOST,
            "-p", PG_PORT,
            "-U", PG_USER,
            "-d", PG_DB,
            "-v", "ON_ERROR_STOP=1",
        ],
        input=script_sql,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def main():
    passo_ms = FECHAMENTO
    cursor = para_ms(INICIO)
    agora = agora_milisegundos()


    while cursor < agora:
        lote = buscar(cursor)
        if not lote:
            break

        linhas = []

        for candle in lote:
            if int(candle[6]) < agora:
                valores = [
                    para_iso(int(candle[0])),
                    str(candle[1]),
                    str(candle[2]),
                    str(candle[3]),
                    str(candle[4]),
                    str(candle[5]),
                    para_iso(int(candle[6])),
                    str(candle[7]),
                    str(candle[8]),
                    str(candle[9]),
                    str(candle[10]),
                    SYMBOL,
                    INTERVALO,
                ]
                linhas.append("\t".join(valores))

        if linhas:
            inserir_batch("\n".join(linhas))

        cursor = int(lote[-1][0]) + passo_ms

        #Folga para evitar timeout
        time.sleep(0.25)


if __name__ == "__main__":
    main()