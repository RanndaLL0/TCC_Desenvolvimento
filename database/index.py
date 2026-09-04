import configparser
from pathlib import Path
import psycopg
import paramiko

paramiko.DSSKey = getattr(paramiko, "DSSKey", paramiko.Ed25519Key)
from sshtunnel import SSHTunnelForwarder

CONFIG = configparser.ConfigParser(inline_comment_prefixes=None)
CONFIG.read(Path(__file__).resolve().parent.parent / "config.ini")

def _valor(secao, chave):
    return CONFIG.get(secao, chave).strip().strip('"')


def create_connection():
    tunnel = SSHTunnelForwarder(
        (_valor("SSH_CONNECTION", "SSH_HOST"), 22),
        ssh_username=_valor("SSH_CONNECTION", "SSH_USER"),
        ssh_pkey=_valor("SSH_CONNECTION", "SSH_KEY"),
        remote_bind_address=(_valor("postgres", "host"), int(_valor("postgres", "port"))),
    )
    tunnel.start()

    conn = psycopg.connect(
        host="127.0.0.1",
        port=tunnel.local_bind_port,
        user=_valor("postgres", "user"),
        password=_valor("postgres", "password"),
        dbname=_valor("postgres", "database"),
    )
    conn._tunnel = tunnel
    return conn


if __name__ == "__main__":
    conn = create_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        print("Versão do banco:", cur.fetchone()[0])
    conn.close()
