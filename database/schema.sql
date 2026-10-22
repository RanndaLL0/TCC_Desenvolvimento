-- Esquema do banco de dados do projeto (PostgreSQL 16+).
--
-- Uso (banco vazio):
--   psql -h <host> -U <usuario> -d <banco> -f database/schema.sql
--   psql -h <host> -U <usuario> -d <banco> -f database/normalize_btc.sql
--
-- O DDL abaixo reproduz o banco de producao (extraido de information_schema e
-- pg_constraint em 2026-10-21).

-- Candles de 5 minutos coletados da Binance (src/scrapping/scrapping.py).
--
-- A ORDEM DAS COLUNAS E PARTE DO CONTRATO: o scrapper carrega os lotes com
-- "\copy tmp_candles" e depois "INSERT INTO btc_usdt SELECT * FROM tmp_candles",
-- onde tmp_candles herda as colunas desta tabela. Alterar a ordem quebra a coleta.
--
-- A chave primaria e o que o "ON CONFLICT (symbol, interval, open_time)" do
-- scrapper usa para ignorar candles repetidos; o PostgreSQL resolve o alvo do
-- conflito pelo conjunto de colunas, nao pela ordem em que aparecem na chave.
CREATE TABLE IF NOT EXISTS btc_usdt (
    open_time        timestamp        NOT NULL,
    open             double precision,
    high             double precision,
    low              double precision,
    close            double precision,
    volume           double precision,
    close_time       timestamp,
    quote_volume     double precision,
    trades           integer,
    taker_buy_base   double precision,
    taker_buy_quote  double precision,
    symbol           varchar          NOT NULL DEFAULT 'BTCUSDT',
    "interval"       varchar          NOT NULL DEFAULT '5m',
    CONSTRAINT btc_usdt_pkey PRIMARY KEY (open_time, symbol, "interval")
);

-- Serie discretizada em {-1, 0, +1}, lida por database/index.py (pegar_dados).
--
-- No banco de producao e uma TABELA (nao uma view), sem indices, recriada do
-- zero por database/normalize_btc.sql. Ela precisa existir antes da primeira
-- chamada da procedure, que comeca com um DROP TABLE sem IF EXISTS.
CREATE TABLE IF NOT EXISTS normalize_data (
    open_normalized   double precision,
    high_normalized   double precision,
    low_normalized    double precision,
    close_normalized  double precision,
    open_time         timestamp
);
