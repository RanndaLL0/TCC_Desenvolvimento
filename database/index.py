import configparser
from pathlib import Path
import psycopg
import paramiko

# Compatibilidade: o paramiko >= 4.0 removeu o suporte a chaves DSA (paramiko.DSSKey),
# mas o sshtunnel 0.4.0 (sem manutencao) ainda referencia essa classe ao carregar as chaves.
paramiko.DSSKey = getattr(paramiko, "DSSKey", paramiko.Ed25519Key)
from sshtunnel import SSHTunnelForwarder

CAMINHO_CONFIG = Path(__file__).resolve().parent.parent / "config.ini"

CONFIG = configparser.ConfigParser(inline_comment_prefixes=None)
CONFIG.read(CAMINHO_CONFIG)

CHAVES_SSH = ("SSH_HOST", "SSH_USER", "SSH_KEY")
CHAVES_POSTGRES = ("host", "port", "user", "database")


def _valor(secao, chave):
    return CONFIG.get(secao, chave).strip().strip('"')


# Secao ou chave ausente e valor vazio contam como "nao configurado"
def _valor_ou_none(secao, chave):
    if not CONFIG.has_option(secao, chave):
        return None

    return _valor(secao, chave) or None


# Devolve as credenciais SSH se as tres chaves estiverem preenchidas e None se
# nenhuma estiver. Preenchida pela metade e erro: cair no modo direto em silencio
# esconderia um config.ini com defeito
def _credenciais_ssh():
    valores = {chave: _valor_ou_none("SSH_CONNECTION", chave) for chave in CHAVES_SSH}

    if not any(valores.values()):
        return None

    faltando = [chave for chave, valor in valores.items() if valor is None]
    if faltando:
        raise ValueError(
            f"[SSH_CONNECTION] incompleta no config.ini, faltando: {', '.join(faltando)}"
        )

    return valores


def _dados_postgres():
    faltando = [c for c in CHAVES_POSTGRES if _valor_ou_none("postgres", c) is None]
    if faltando:
        raise ValueError(
            f"[postgres] incompleta no config.ini, faltando: {', '.join(faltando)}"
        )

    return {
        "host": _valor("postgres", "host"),
        "port": int(_valor("postgres", "port")),
        "user": _valor("postgres", "user"),
        "password": _valor_ou_none("postgres", "password") or "",
        "dbname": _valor("postgres", "database"),
    }


# Em producao o Postgres so aceita conexoes vindas da VM, entao host/port sao o
# endereco do banco visto a partir dela
def _connect_via_tunnel(ssh):
    dados = _dados_postgres()

    tunnel = SSHTunnelForwarder(
        (ssh["SSH_HOST"], 22),
        ssh_username=ssh["SSH_USER"],
        ssh_pkey=ssh["SSH_KEY"],
        remote_bind_address=(dados["host"], dados["port"]),
    )
    tunnel.start()

    try:
        conn = psycopg.connect(
            **{**dados, "host": "127.0.0.1", "port": tunnel.local_bind_port}
        )
    except Exception:
        tunnel.close()
        raise

    conn._tunnel = tunnel
    return conn


# Sem credenciais SSH: host/port sao o endereco do banco visto desta maquina
def _connect_local():
    dados = _dados_postgres()

    try:
        conn = psycopg.connect(**dados)
    except psycopg.OperationalError as erro:
        raise ConnectionError(
            "Sem credenciais SSH no config.ini e nenhum PostgreSQL acessivel em "
            f"{dados['host']}:{dados['port']}"
        ) from erro

    conn._tunnel = None
    return conn


def create_connection():
    if not CONFIG.sections():
        raise FileNotFoundError(
            f"Arquivo de configuracao nao encontrado ou vazio: {CAMINHO_CONFIG}"
        )

    ssh = _credenciais_ssh()
    if ssh is not None:
        return _connect_via_tunnel(ssh)

    return _connect_local()


def close_connection(conn):
    conn.close()

    if conn._tunnel is not None:
        conn._tunnel.close()


def pegar_dados():
    conn = create_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT close_normalized FROM normalize_data ORDER BY open_time;"
            )
            linhas = cur.fetchall()
            return linhas
    finally:
        close_connection(conn)


if __name__ == "__main__":
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print("Versão do banco:", cur.fetchone()[0])
    finally:
        close_connection(conn)
