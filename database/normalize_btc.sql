-- Procedure LEGADA de discretizacao: retorno logarítmico -> {-1, 0, +1}.
--
-- Mantida aqui apenas para reproduzir o estado atual da tabela normalize_data
-- (requer database/schema.sql e dados em btc_usdt). Uso:
--   psql -h <host> -U <usuario> -d <banco> -f database/normalize_btc.sql
--   psql -h <host> -U <usuario> -d <banco> -c "CALL normalize_btc();"
--
-- Limitacoes conhecidas, a serem substituidas pelo modulo Python de discretizacao:
--   * media e desvio (stddev_samp) calculados sobre a serie INTEIRA, sem separar
--     treino e teste (vazamento de informacao do teste para o treino);
--   * limiar fixo em s/3, sem parametrizar s/4 ou s/5;
--   * o primeiro candle nao tem LAG, seu retorno e NULL e, no CASE, cai no ELSE:
--     recebe o simbolo 0 em vez de ficar sem simbolo;
--   * o LAG atravessa as lacunas da serie (candles ausentes);
--   * recria a tabela a cada execucao (DROP TABLE) e le todo o btc_usdt, sem data
--     final fixa; a coleta continua, entao o resultado muda com o tempo.
CREATE OR REPLACE PROCEDURE public.normalize_btc()
LANGUAGE 'plpgsql'
AS $BODY$
DECLARE
    open_avg DOUBLE PRECISION;
    high_avg DOUBLE PRECISION;
    low_avg DOUBLE PRECISION;
    close_avg DOUBLE PRECISION;

    open_std DOUBLE PRECISION;
    high_std DOUBLE PRECISION;
    low_std DOUBLE PRECISION;
    close_std DOUBLE PRECISION;
BEGIN

    DROP TABLE normalize_data;
    CREATE TABLE IF NOT EXISTS normalize_data(
        open_normalized double precision,
        high_normalized double precision,
        low_normalized double precision,
        close_normalized double precision,
        open_time TIMESTAMP
    );

    INSERT INTO normalize_data(
        open_normalized,
        high_normalized,
        low_normalized,
        close_normalized,
        open_time
    ) SELECT
        LN(NULLIF(open, 0) / NULLIF(LAG(open) OVER (ORDER BY open_time), 0)) as open_log,
        LN(NULLIF(high, 0) / NULLIF(LAG(high) OVER (ORDER BY open_time), 0 )) as high_log,
        LN(NULLIF(low, 0) / NULLIF(LAG(low) OVER (ORDER BY open_time), 0 )) as low_log,
        LN(NULLIF(close, 0) / NULLIF(LAG(close) OVER (ORDER BY open_time), 0 )),
        open_time
    FROM btc_usdt;

    SELECT
        avg(open_normalized), stddev_samp(open_normalized),
        avg(high_normalized), stddev_samp(high_normalized),
        avg(low_normalized),  stddev_samp(low_normalized),
        avg(close_normalized), stddev_samp(close_normalized)
    INTO
        open_avg, open_std,
        high_avg, high_std,
        low_avg,  low_std,
        close_avg, close_std
    FROM normalize_data;

    UPDATE normalize_data
    SET
        open_normalized = CASE
            WHEN (open_normalized - open_avg) < -(open_std / 3.0) THEN -1
            WHEN (open_normalized - open_avg) > (open_std / 3.0) THEN 1
            ELSE 0
        END,
        high_normalized = CASE
            WHEN (high_normalized - high_avg) < -(high_std / 3.0) THEN -1
            WHEN (high_normalized - high_avg) > (high_std / 3.0) THEN 1
            ELSE 0
        END,
        low_normalized = CASE
            WHEN (low_normalized - low_avg) < -(low_std / 3.0) THEN -1
            WHEN (low_normalized - low_avg) > (low_std / 3.0) THEN 1
            ELSE 0
        END,
        close_normalized = CASE
            WHEN (close_normalized - close_avg) < -(close_std / 3.0) THEN -1
            WHEN (close_normalized - close_avg) > (close_std / 3.0) THEN 1
            ELSE 0
        END;
END;
$BODY$;
