# Discretização dos retornos: decisões e uso

Registro das decisões da discretização do fechamento do BTC/USDT em `{-1, 0, +1}`, implementada em `src/preprocess/discretize.py`. Este documento é documentação do sistema: a justificativa formal e a redação da metodologia ficam para a monografia.

- **Escrito em:** 2026-10-22
- **Dados:** snapshot `btc_usdt_2026-10-01` (957.853 candles de 5 minutos, de 2017-08-17 04:00 a 2026-09-30 23:55). Ver o README, etapa "Reconstruir a base a partir do snapshot".
- **Números citados** vêm do relatório gerado sobre esse snapshot: `relatorios/discretizacao_2026-10-01.md` (resumo) e `.json` (detalhe por mês, por bloco e cada lacuna).

## Resumo das decisões

| # | Decisão | Status |
|---|---|---|
| D1 | Retorno logarítmico do fechamento, só entre candles adjacentes (5 min). | Adotada |
| D2 | Limiar `s/k` parametrizável (`k` = 3, 4, 5), desvio amostral. | Adotada |
| D3 | Modo `blocks`: o bloco `b` usa a média e o desvio do bloco `b-1`. | Adotada pela dupla; justificativa formal na monografia |
| D4 | Modo `global`: uma média e um desvio só do treino. | Adotada (referência; é o que o TCC descreve hoje) |
| D5 | Corte treino/teste em 95% das linhas; discretiza a série inteira e depois corta. | Adotada |
| D6 | Lacunas detectadas e contadas; o retorno que as atravessa não gera símbolo e cada trecho contínuo é um segmento. | Detecção adotada. **Tratamento nas transições pendente** (ver D6) |
| D7 | Os experimentos do TCC rodam sobre o snapshot; o núcleo não depende da fonte. | Adotada. Módulo "live" é proposta futura |
| D8 | Posição sem símbolo marcada com um valor fora de `{-1, 0, 1}`. | Adotada |

## Contexto

A série bruta vem em candles de 5 minutos. O TCC modela a sequência de símbolos que resume o retorno de cada candle: queda significativa (`-1`), oscilação pequena (`0`) e alta significativa (`+1`).

Havia uma discretização anterior, a procedure `normalize_btc()` (`database/normalize_btc.sql`, mantida como legado). Os problemas dela, todos confirmados:

1. Média e desvio da **série inteira**, sem separar treino e teste: o teste influencia a régua do treino.
2. Limiar fixo em `s/3`, sem como testar `s/4` e `s/5`.
3. O primeiro candle, sem retorno, caía no `ELSE` e virava símbolo `0`.
4. O retorno usava `LAG` sem olhar lacunas: um retorno de várias horas era tratado como retorno de 5 minutos.

**Por que não um limiar único para a série toda.** A variância do retorno não é estacionária. O desvio do retorno de 5 minutos, medido por mês, foi de cerca de **0,0052** no primeiro bloco (agosto de 2017) para **0,0011** no último (agosto de 2026). Com média e desvio globais, os primeiros anos concentram `±1` e os últimos concentram `0`: no relatório, o percentual de símbolo `0` com `s/3` global vai de 31% em 2017 a 68% em 2023, e o **teste** (últimos 5%) fica com 68% de zeros contra 51% no treino. Com limiar por bloco (também `s/3`), o percentual de zeros fica entre 37% e 46% em todos os anos. O mesmo valor de retorno significa coisas muito diferentes em 2017 e em 2026.

## Decisões

### D1. Retorno logarítmico do fechamento, entre candles adjacentes

`r[t] = ln(close[t] / close[t-1])`, só quando `open_time[t] - open_time[t-1]` é exatamente 5 minutos. O primeiro candle e o candle logo depois de uma lacuna ficam sem retorno.

- **Por quê:** é o retorno que o TCC define, e o retorno que atravessa uma lacuna mede outra escala de tempo (D6).
- **Só o `close`:** a procedure legada também normalizava `open`, `high` e `low`; o TCC usa o fechamento.

### D2. Limiar `s/k`

Com `mu` e `s` a média e o desvio **amostral** (`ddof=1`, como o TCC diz) dos retornos de referência:

```
(r - mu) < -s/k         ->  -1
-s/k <= (r - mu) <= s/k ->   0
(r - mu) >  s/k         ->  +1
```

- **`k` divide o desvio.** `k=3` corta em `s/3`, não em 3 desvios. Esse é o parâmetro da issue (`s/3`, `s/4`, `s/5`) e substitui o `threshold` em múltiplos de desvio que o primeiro rascunho dos testes usava.
- Os pontos exatamente em `±s/k` ficam em `0`, como na equação do TCC.
- `k` é qualquer número positivo; os valores do TCC são 3, 4 e 5.

### D3. Modo `blocks` (causal)

Os candles são agrupados em blocos de `interval` meses de calendário (padrão 1), contados a partir do mês do primeiro candle. **O bloco `b` é classificado com a média e o desvio do bloco `b-1`**, calculados só com os retornos válidos do `b-1`.

- **Por quê:** acompanha a mudança de volatilidade (Contexto) sem usar nenhum dado do futuro. Classificar um bloco com as estatísticas do próprio bloco daria a cada símbolo informação de candles posteriores, e impediria rodar o mesmo procedimento ao vivo, onde o desvio do mês corrente ainda não existe.
- **O primeiro bloco só serve de histórico e fica sem símbolos.** No snapshot é agosto de 2017, que começa no dia 17: 4.271 símbolos a menos, 0,45% da série.
- **Alternativas consideradas:**
  - Estatísticas do próprio bloco: descartada pelo vazamento acima.
  - Janela de histórico com os últimos N blocos: descartada por ora; usa só o bloco anterior. Um histórico maior reage mais devagar.
- **Limitações assumidas:**
  - A régua atrasa um bloco: numa mudança brusca de volatilidade, os primeiros dias do bloco novo são medidos com a régua do anterior.
  - Se o bloco anterior tiver desvio zero, ou menos de 2 retornos, ou estiver vazio, a discretização **falha com erro**, e não inventa um valor.
  - Nos blocos de teste, as estatísticas vêm de blocos anteriores que também são de teste. É causal, mas **não é literalmente "só do treino"**; essa é a divergência em relação ao texto atual do TCC, e é por isso que o modo `global` existe (D4).
- **Tamanho do bloco:** o padrão de 1 mês é uma escolha inicial, ainda sem análise de sensibilidade. `--interval` permite testar outros.

### D4. Modo `global`

Uma média e um desvio, estimados **só nas linhas de treino** (`[0, corte)`) e aplicados a treino e teste: a definição que o TCC traz hoje e que a issue pede.

- **Por quê:** é a referência. Permite comparar, no mesmo relatório, o limiar único com o limiar por bloco.
- No snapshot, `mu = 3,09e-06` e `sigma = 0,002415` (treino). Com a série inteira seriam `3,11e-06` e `0,00237`.

### D5. Corte treino/teste

O treino são as primeiras 95% das **linhas de candles** (`(n * 95) // 100`, a mesma conta do `split_data` em `src/utils/data_manager.py`; um teste garante que as duas coincidem). No snapshot, o teste começa na linha 909.960 (2026-04-17 16:55).

**Discretiza-se a série inteira e só depois corta-se.** No modo `blocks` isso é seguro porque cada símbolo depende só de dados anteriores a ele: os símbolos de treino saem idênticos com ou sem o teste (há um teste automatizado para isso, alterando o futuro). No modo `global`, o corte define quais linhas estimam `mu` e `s`.

### D6. Lacunas

Uma lacuna é um `open_time` que não está a 5 minutos do anterior. O snapshot tem **34 lacunas, 1.715 candles faltantes e 35 segmentos contínuos**. A maior tem 1 dia e 9h35 (402 candles), de 2018-02-08 00:25 a 2018-02-09 10:00.

- **Já implementado:** detectar e contar; não gerar símbolo para o retorno que atravessa a lacuna; registrar um identificador de segmento para cada símbolo (`Discretization.segment`); não usar esses retornos para estimar média e desvio.
- **Pendente:** tratar as lacunas **nas transições**, isto é, os modelos contarem pares (e, em ordens maiores, janelas de `n+1` símbolos) só dentro de um mesmo segmento. Hoje o `FirstOrder.fit` recebe uma única sequência e conta como vizinhos dois símbolos separados por uma lacuna. O efeito numérico no primeiro-ordem é no máximo 34 pares em 957 mil (0,0036%), mas o critério da issue pede o tratamento e ele passa a importar nas ordens maiores. A dupla vai acompanhar essa mudança.

### D7. Fonte dos dados e módulo live

As funções de `discretize.py` trabalham sobre arrays (`open_time` e `close`). O snapshot entra apenas por `load_snapshot`, que confere o SHA-256 do manifesto antes de usar o arquivo. Os experimentos do TCC rodam sobre o snapshot para serem reprodutíveis; **a apresentação ao vivo precisará de uma fonte com os dados mais recentes**, e o desenho aproveita o mesmo núcleo (ver "Proposta futura").

### D8. Marcador de posição sem símbolo

`NO_SYMBOL = -128` (o menor `int8`). Fica fora de `{-1, 0, 1}` de propósito: se um array com marcador chegar a um modelo, `to_indices` levanta `ValueError` em vez de contar um símbolo que não existe. Os modelos devem receber `resultado.symbols[resultado.valid]`, ou, depois da adaptação da D6, cada segmento.

## Como usar

Relatório sobre o snapshot, com os dois modos e `k` igual a 3, 4 e 5:

```
python -m src.preprocess.discretize --snapshot snapshot/btc_usdt_2026-10-01.parquet
```

| Opção | Padrão | Significado |
|---|---|---|
| `--snapshot` | (obrigatória) | Parquet do snapshot, com o manifesto ao lado. |
| `--mode` | `blocks global` | Um ou os dois modos. |
| `--k` | `3 4 5` | Limiares `s/k`. |
| `--interval` | `1` | Meses por bloco (modo `blocks`). |
| `--train` | `95` | Percentual de treino. |
| `--output` | `relatorios/` | Pasta dos arquivos `discretizacao_<data>.md` e `.json`. |

Como biblioteca:

```python
from src.preprocess import discretize as d

open_times, close, fonte = d.load_snapshot("snapshot/btc_usdt_2026-10-01.parquet")
r = d.discretize(open_times, close, k=3, mode="blocks")

r.symbols[r.valid]   # os símbolos, sem as posições sem retorno
r.segment            # segmento contínuo de cada símbolo (-1 onde não há símbolo)
r.cut                # primeira linha do teste
```

## Verificações feitas

- **Reprodução do legado.** Com a série inteira, `log_returns` + `fit_stats` + `classify` reproduzem a `normalize_data` gerada pela procedure em **100%** das 957.852 linhas comparáveis. O retorno e o limiar estão corretos.
- **Testes** (`tests/test_discretize.py`, 52 testes): ausência de vazamento (alterar o futuro não muda o passado, e o treino não depende do teste), estatísticas do bloco anterior, modo global só com o treino, lacunas e segmentos, `ddof=1`, fronteira do limiar, erros de entrada, relatório e carregador do snapshot.
- **Mutação.** Cinco defeitos injetados de propósito foram pegos pelos testes: bloco usando as próprias estatísticas, `ddof=0`, modo global usando a série toda, fronteira do limiar invertida e retorno atravessando a lacuna.

## Referências

- **Lida por inteiro:** Araújo e Barbosa (2024), *Reconstructing Cryptocurrency Processes via Markov Chains*, Computational Economics 64(4) ([arXiv 2308.07626](https://arxiv.org/pdf/2308.07626)). Série horária de 7 meses, retorno log, alfabeto codificado por média e desvio; não trata a não estacionariedade da variância. Antes de citar os cortes do alfabeto de 3 símbolos, confirmar no PDF (a extração de texto truncou as frações).
- **Consultadas só pelo resumo (ler antes de citar):** Tino, Schittenkopf e Dorffner (2000), quantização de séries financeiras não estacionárias; *Entropy and efficiency of the ETF market* ([arXiv 1609.04199](https://arxiv.org/pdf/1609.04199)), limiar adaptativo à volatilidade; *Stylized Facts of High-Frequency Bitcoin Time Series* ([arXiv 2402.11930](https://arxiv.org/html/2402.11930v2)), mudança de regime de volatilidade; Bouri et al. (2019), quebras estruturais na volatilidade do Bitcoin; Shen, Urquhart e Wang (2020), previsão de volatilidade do Bitcoin com quebras.

## Pendências

1. **Lacunas nas transições (D6):** adaptar o `FirstOrder` para contar por segmento; HMM e MTD seguem o mesmo contrato quando forem tratados.
2. **Sensibilidade ao tamanho do bloco** (`--interval`).
3. **Texto do TCC:** a seção "Estratégia de discretização" descreve média e desvio só do treino, uma vez. Ao adotar o modo `blocks` como método principal, a metodologia precisa ser reescrita e justificada na monografia.
4. **Gravar as discretizações no Postgres** com os parâmetros congelados (plano de arquitetura), em vez de só gerá-las em memória.
5. **Módulo live** (abaixo).

## Proposta futura: módulo live

**Não implementado.** Registrado aqui para não ficar como ponta solta.

**Objetivo.** Rodar um modelo já treinado com os dados mais atuais possíveis, para a apresentação do TCC, e comparar cada previsão com o que de fato aconteceu no candle seguinte.

**O que o desenho atual já permite.**

- O modo `blocks` é causal: o símbolo de um candle novo só precisa da média e do desvio do bloco anterior, que já existem. O mesmo procedimento do treino funciona ao vivo.
- O núcleo recebe arrays; falta só um carregador com os candles mais recentes. O serviço `src/scrapping/background.py` já sincroniza a tabela `btc_usdt` a cada 5 minutos.

**Peças previstas.**

1. **Fonte de candles recentes**, lendo só candles **fechados** (`close_time` anterior ao momento atual).
2. **Estado:** a média e o desvio do bloco anterior, recalculados na virada do mês.
3. **Modelo treinado**, carregado de arquivo (serialização já prevista no plano dos modos de avaliação).
4. **Laço:** candle fechado, retorno, símbolo, previsão do próximo símbolo; quando o próximo candle fecha, compara previsão e desfecho.
5. **Registro** de previsões e desfechos, para calcular as métricas no fim.

**Questões em aberto.**

- Lacunas ao vivo (candle atrasado ou ausente): não prever com uma janela que atravesse a lacuna; depende da D6.
- Modelo congelado no treino ou reajustado periodicamente.
- Reprodutibilidade: guardar os candles que entraram em cada previsão.
- No modo `global`, o live precisaria dos parâmetros do treino congelados e persistidos.
- Relógio e atraso da coleta: definir o que fazer se o último candle ainda não chegou.
