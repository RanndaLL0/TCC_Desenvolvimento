from types import SimpleNamespace

import configparser
import psycopg
import pytest

from database import index

SSH = {
    "SSH_HOST": '"vm.exemplo"',
    "SSH_USER": '"ubuntu"',
    "SSH_KEY": '"/home/u/.ssh/id"',
}

POSTGRES = {
    "host": "10.0.0.5",
    "port": "5433",
    "user": "app",
    "password": "segredo",
    "database": "btc",
}


def config_text(ssh=None, postgres=POSTGRES):
    lines = ["[postgres]"] + [f"{k} = {v}" for k, v in postgres.items()]
    if ssh is not None:
        lines += ["", "[SSH_CONNECTION]"] + [f"{k} = {v}" for k, v in ssh.items()]
    return "\n".join(lines)


@pytest.fixture
def load_config(monkeypatch):
    def load(text):
        config = configparser.ConfigParser(inline_comment_prefixes=None)
        config.read_string(text)
        monkeypatch.setattr(index, "CONFIG", config)

    return load


class FakeTunnel:
    def __init__(self, address, ssh_username, ssh_pkey, remote_bind_address):
        self.address = address
        self.ssh_username = ssh_username
        self.ssh_pkey = ssh_pkey
        self.remote_bind_address = remote_bind_address
        self.local_bind_port = 40000
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


# nenhum teste abre SSH ou fala com um banco de verdade: o túnel e o psycopg
# são trocados por dublês que só registram como foram chamados
@pytest.fixture
def tunnels(monkeypatch):
    created = []

    def factory(address, **kwargs):
        tunnel = FakeTunnel(address, **kwargs)
        created.append(tunnel)
        return tunnel

    monkeypatch.setattr(index, "SSHTunnelForwarder", factory)
    return created


@pytest.fixture
def connects(monkeypatch):
    calls = []

    def fake_connect(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(index.psycopg, "connect", fake_connect)
    return calls


def test_ssh_credentials_open_tunnel_and_connect_through_it(
    load_config, tunnels, connects
):
    load_config(config_text(ssh=SSH))
    conn = index.create_connection()

    tunnel = tunnels[0]
    assert tunnel.started
    assert tunnel.address == ("vm.exemplo", 22)
    assert tunnel.ssh_username == "ubuntu"
    assert tunnel.ssh_pkey == "/home/u/.ssh/id"
    assert tunnel.remote_bind_address == ("10.0.0.5", 5433)

    assert connects == [
        {
            "host": "127.0.0.1",
            "port": 40000,
            "user": "app",
            "password": "segredo",
            "dbname": "btc",
        }
    ]
    assert conn._tunnel is tunnel


def test_without_ssh_section_connects_directly(load_config, tunnels, connects):
    load_config(config_text())
    conn = index.create_connection()

    assert tunnels == []
    assert connects == [
        {
            "host": "10.0.0.5",
            "port": 5433,
            "user": "app",
            "password": "segredo",
            "dbname": "btc",
        }
    ]
    assert conn._tunnel is None


def test_empty_ssh_values_connect_directly(load_config, tunnels, connects):
    load_config(config_text(ssh={chave: "" for chave in SSH}))
    conn = index.create_connection()

    assert tunnels == []
    assert conn._tunnel is None


@pytest.mark.parametrize("missing", list(SSH))
def test_partial_ssh_credentials_raise_instead_of_going_local(
    load_config, tunnels, connects, missing
):
    load_config(config_text(ssh={**SSH, missing: ""}))

    with pytest.raises(ValueError, match=missing):
        index.create_connection()

    assert tunnels == []
    assert connects == []


@pytest.mark.parametrize("missing", ["host", "port", "user", "database"])
def test_incomplete_postgres_section_raises(load_config, tunnels, connects, missing):
    load_config(config_text(postgres={k: v for k, v in POSTGRES.items() if k != missing}))

    with pytest.raises(ValueError, match=missing):
        index.create_connection()

    assert connects == []


def test_password_is_optional(load_config, tunnels, connects):
    load_config(config_text(postgres={k: v for k, v in POSTGRES.items() if k != "password"}))
    index.create_connection()

    assert connects[0]["password"] == ""


def test_no_local_postgres_raises_connection_error(load_config, tunnels, monkeypatch):
    def refuse(**kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(index.psycopg, "connect", refuse)
    load_config(config_text())

    with pytest.raises(ConnectionError, match="10.0.0.5:5433") as info:
        index.create_connection()

    assert isinstance(info.value.__cause__, psycopg.OperationalError)
    assert tunnels == []


def test_tunnel_is_closed_when_database_connection_fails(
    load_config, tunnels, monkeypatch
):
    def refuse(**kwargs):
        raise psycopg.OperationalError("password authentication failed")

    monkeypatch.setattr(index.psycopg, "connect", refuse)
    load_config(config_text(ssh=SSH))

    with pytest.raises(psycopg.OperationalError):
        index.create_connection()

    assert tunnels[0].closed


def test_empty_config_raises_file_not_found(load_config):
    load_config("")

    with pytest.raises(FileNotFoundError):
        index.create_connection()


def test_close_connection_closes_the_tunnel_when_there_is_one():
    tunnel = FakeTunnel(("h", 22), "u", "k", ("h", 5432))
    closed = []
    conn = SimpleNamespace(close=lambda: closed.append(True), _tunnel=tunnel)

    index.close_connection(conn)

    assert closed == [True]
    assert tunnel.closed


def test_close_connection_without_tunnel():
    closed = []
    conn = SimpleNamespace(close=lambda: closed.append(True), _tunnel=None)

    index.close_connection(conn)

    assert closed == [True]
