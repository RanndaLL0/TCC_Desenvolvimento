import json
import configparser
import http.client
import os
import random
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = 'https://api.binance.com/api/v3/klines'
LIMITE_REQUISICAO = 1000
FECHAMENTO = 300_000

# Retry com backoff exponencial para as requisicoes na API
TENTATIVAS_MAXIMAS = 5
ESPERA_INICIAL_SEGUNDOS = 1
ESPERA_MAXIMA_SEGUNDOS = 60

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

CAMINHO_CONFIG = Path(__file__).resolve().parents[2] / "config.ini"

config = configparser.ConfigParser(inline_comment_prefixes=None)
if not config.read(CAMINHO_CONFIG):
    raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {CAMINHO_CONFIG}")

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

def para_iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def erro_temporario(erro):
    if isinstance(erro, urllib.error.HTTPError):
        return erro.code in (418, 429) or erro.code >= 500
    return True


def tempo_espera(tentativa, erro):
    if isinstance(erro, urllib.error.HTTPError):
        retry_after = erro.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return int(retry_after)

    # Backoff exponencial
    espera = min(ESPERA_MAXIMA_SEGUNDOS, ESPERA_INICIAL_SEGUNDOS * 2 ** tentativa)
    return espera + random.uniform(0, espera / 2)


def buscar(inicio_ms):
    parametros = urllib.parse.urlencode({
        "symbol": SYMBOL,
        "interval": INTERVALO,
        "startTime": inicio_ms,
        "limit": LIMITE_REQUISICAO,
    })
    url = f"{BASE_URL}?{parametros}"

    for tentativa in range(TENTATIVAS_MAXIMAS):
        try:
            with urllib.request.urlopen(url, timeout=30) as resposta:
                return json.loads(resposta.read())
        except (OSError, http.client.HTTPException) as erro:
            if not erro_temporario(erro) or tentativa == TENTATIVAS_MAXIMAS - 1:
                raise

            espera = tempo_espera(tentativa, erro)
            print(f"Falha ao buscar candles ({erro}), tentativa {tentativa + 1}/{TENTATIVAS_MAXIMAS}, nova tentativa em {espera:.1f}s")
            time.sleep(espera)

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

# O mais foi posto dentro de sincronizar para o processo
# de busca e insercao estar visivel para o servico em background
def sincronizar(cursor_ms, agora_ms=None):
    if agora_ms is None:
        agora_ms = agora_milisegundos()

    while cursor_ms < agora_ms:
        lote = buscar(cursor_ms)
        if not lote:
            break

        linhas = []

        for candle in lote:
            if int(candle[6]) < agora_ms:
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

        cursor_ms = int(lote[-1][0]) + FECHAMENTO

        #Folga para evitar timeout
        time.sleep(0.25)

def main():
    sincronizar(para_ms(INICIO))


if __name__ == "__main__":
    main()
