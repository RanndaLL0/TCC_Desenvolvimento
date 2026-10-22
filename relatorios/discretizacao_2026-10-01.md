# Relatório da discretização

Gerado em 2026-10-22T12:56:44+00:00 (UTC). Decisões: `docs/discretizacao.md`.

## Fonte

- Arquivo: `btc_usdt_2026-10-01.parquet` (SHA-256 `4c71d14999175cf62f037af5880db2554c03617228db82989fe76fdeb175bd4d`)
- Corte do snapshot: `open_time` anterior a 2026-10-01
- Candles: 957.853 (2017-08-17 04:00:00 a 2026-09-30 23:55:00)
- Retornos válidos (entre candles a 5 minutos): 957.818

## Lacunas

- Lacunas (`open_time[t] - open_time[t-1]` diferente de 5 minutos): **34**
- Candles faltantes: **1.715**
- Segmentos contínuos: **35**
- O retorno que atravessa uma lacuna não gera símbolo; o candle logo depois dela começa um novo segmento.
- Maior lacuna: 1 day, 9:35:00 (402 candles), de 2018-02-08 00:25:00 a 2018-02-09 10:00:00

As lacunas, uma a uma, estão no arquivo JSON do relatório.

## Corte treino/teste

- Treino: 95% das linhas; o teste começa na linha 909.960 (2026-04-17 16:55:00)

## Distribuição de classes

Percentuais de -1 / 0 / +1.

| Modo | Limiar | Símbolos | Geral | Treino | Teste |
|---|---|---:|---|---|---|
| blocos | s/3 | 953.547 | 28,77% / 42,41% / 28,82% | 28,78% / 42,33% / 28,89% | 28,57% / 43,83% / 27,61% |
| blocos | s/4 | 953.547 | 33,21% / 33,51% / 33,28% | 33,23% / 33,41% / 33,36% | 32,84% / 35,36% / 31,80% |
| blocos | s/5 | 953.547 | 36,15% / 27,64% / 36,22% | 36,17% / 27,53% / 36,29% | 35,67% / 29,56% / 34,77% |
| global | s/3 | 957.818 | 23,98% / 51,87% / 24,15% | 24,39% / 51,02% / 24,58% | 16,18% / 67,96% / 15,85% |
| global | s/4 | 957.818 | 28,86% / 42,15% / 28,98% | 29,25% / 41,35% / 29,41% | 21,62% / 57,45% / 20,94% |
| global | s/5 | 957.818 | 32,27% / 35,36% / 32,37% | 32,62% / 34,61% / 32,76% | 25,59% / 49,59% / 24,82% |

## Percentual de símbolo 0 por ano

A volatilidade do retorno muda ao longo dos anos; é o que compara um limiar único (global) com o limiar por bloco.

| Ano | blocos s/3 | blocos s/4 | blocos s/5 | global s/3 | global s/4 | global s/5 |
|---|---:|---:|---:|---:|---:|---:|
| 2017 | 42,26% | 35,85% | 31,72% | 31,47% | 27,83% | 25,44% |
| 2018 | 46,16% | 36,45% | 29,97% | 43,44% | 34,82% | 29,20% |
| 2019 | 45,27% | 35,80% | 29,41% | 51,74% | 41,61% | 34,55% |
| 2020 | 41,08% | 32,49% | 26,75% | 49,37% | 39,13% | 32,17% |
| 2021 | 37,10% | 28,34% | 22,81% | 33,15% | 25,12% | 20,08% |
| 2022 | 45,37% | 36,14% | 30,00% | 51,51% | 41,59% | 34,67% |
| 2023 | 43,73% | 34,82% | 28,83% | 67,73% | 56,90% | 48,59% |
| 2024 | 38,88% | 30,09% | 24,50% | 54,37% | 43,54% | 36,02% |
| 2025 | 40,88% | 32,02% | 26,16% | 62,19% | 51,19% | 43,19% |
| 2026 | 43,59% | 35,08% | 29,40% | 63,88% | 53,64% | 46,17% |

O detalhe por mês e as estatísticas de cada bloco estão no arquivo JSON do relatório.
