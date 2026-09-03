import csv
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE_URL = 'https://api.binance.com/api/v3/klines'
LIMITE_REQUISICAO = 1000

INTERVALOS_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

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
    "ignore"
]

SYMBOL = "BTCUSDT"
INTERVALO = "5m"
INICIO = "2023-01-01"
SAIDA = "dados/btcusdt.csv"


# Essas funcoes sao so utilitarios para converter tempo para milisegundos
# ( A API exige )

def agora_milisegundos() -> int:
    return int(time.time() * 1000)


def para_ms(texto):
    dt = datetime.strptime(texto, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


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


def main():
    passo_ms = INTERVALOS_MS[INTERVALO]
    cursor = para_ms(INICIO)
    agora = agora_milisegundos()

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(CABECALHO)

        while cursor < agora:
            lote = buscar(cursor)
            if not lote:
                break
            for candle in lote:
                if int(candle[6]) < agora:
                    escritor.writerow([
                        para_iso(int(candle[0])),
                        *candle[1:6],
                        para_iso(int(candle[6])),
                        *candle[7:11],
                    ])
            cursor = int(lote[-1][0]) + passo_ms

            #Folga para evitar timeout
            time.sleep(0.25)


if __name__ == "__main__":
    main()
