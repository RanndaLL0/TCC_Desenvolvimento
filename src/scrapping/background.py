import os
import subprocess
import time
from datetime import datetime, timezone
import scrapping

INTERVALO_EXECUCAO_SEGUNDOS = 5 * 60

# Aqui pego o ultimo open time para consultar tudo o que falta até o momento
def obter_cursor_ms():
    comando = (
        "SELECT COALESCE(MAX(open_time)::text, '') FROM btc_usdt;"
    )

    env = os.environ.copy()
    env["PGPASSWORD"] = scrapping.PG_PASSWORD

    resultado = subprocess.run(
        [
            "psql",
            "-h", scrapping.PG_HOST,
            "-p", scrapping.PG_PORT,
            "-U", scrapping.PG_USER,
            "-d", scrapping.PG_DB,
            "-v", "ON_ERROR_STOP=1",
            "-t", "-A",
            "-c", comando,
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    ultimo = resultado.stdout.strip()
    if not ultimo:
        return scrapping.para_ms(scrapping.INICIO)

    dt = datetime.strptime(ultimo, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000) + 1


def ciclo():
    cursor = obter_cursor_ms()
    scrapping.sincronizar(cursor)

def main():
    while True:
        ciclo()
        time.sleep(INTERVALO_EXECUCAO_SEGUNDOS)


if __name__ == "__main__":
    main()
