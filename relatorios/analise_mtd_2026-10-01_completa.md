# Análise completa do MTD — snapshot 2026-10-01, blocos e global, s/3

Esta análise lê `relatorios/mtd_2026-10-01_completa.md` (tabelas) e `relatorios/mtd_2026-10-01_completa.json`, gerados por `src/experiments/mtd_analysis.py`. Ela refaz para o MTD as medidas de `relatorios/analise_hmm_2026-10-01_completa.md`, para que os dois modelos possam ser comparados:

- **Baselines do escopo:** uniforme, persistência, frequências (iid) e cadeia de Markov de 1ª ordem, ajustados no mesmo treino e avaliados no mesmo teste.
- **Ordem do MTD:** de 1 a 10, com BIC. É o análogo do "número de estados" do HMM.
- **Previsão no teste:** um passo à frente, com calibração, e horizontes de até 1 dia.
- **Estrutura:** pesos por lag, matriz `Q`, distribuição implicada e tempo de memória.
- **Sazonalidade:** a P(0) prevista por hora e por dia da semana, contra a observada.
- **Estabilidade do EM:** 10 reinícios por ordem, o mesmo número do HMM de 3 estados.
- **Adequação e lacunas:** autocorrelação real contra a de dados simulados pelo modelo, e o efeito das lacunas.

O HMM entra na comparação com os parâmetros de `relatorios/hmm_2026-10-01_blocks_k3.json`, **sem retreinar**. Só existe HMM treinado no modo `blocks`; **no modo `global` a comparação com o HMM não foi feita** (ver seção 10).

Unidade: log-verossimilhança **por símbolo, em nats**. Um modelo uniforme em 3 símbolos fica em `-ln 3` = -1,0986. Este documento é análise do sistema, não texto da monografia.

## O que mudou em relação à análise anterior do MTD

A análise `analise_mtd_2026-10-01.md` e o relatório `mtd_2026-10-01.md` rodaram o MTD com 5 reinícios na primeira configuração e 2 nas demais, sem baselines de ordem alta, AUC, calibração, horizontes, sazonalidade nem simulação, e comparavam com o HMM pelos números do relatório dele. Agora:

| Item | Antes | Agora |
|---|---|---|
| Reinícios por ordem | 5 (1ª configuração), 2 (demais) | **10**, como no HMM de 3 estados |
| Baselines | uniforme, iid, Markov de ordem 1 | uniforme, iid, persistência, Markov de ordem 1, com BIC onde há parâmetros ajustados |
| Métricas de previsão | LL e direção | LL, acurácia, direção, **AUC de movimento, Brier**, calibração por decis |
| Horizonte | só 5 min | 5 min, 15 min, 1 h, 4 h, 1 dia |
| Comparação com o HMM | números citados, transições diferentes | **mesmas transições**: o HMM é filtrado ao longo de treino e teste, como o MTD |
| Estrutura | `Q` e `lam` | `Q`, `lam`, distribuição implicada, tempo de memória |
| Sazonalidade, simulação, lacunas | não havia | incluídas |
| Discretizações | 6 (blocos e global, s/3, s/4, s/5) | `blocks` e `global`, só s/3 (a configuração do HMM) |

Parâmetros que **não mudam** e já eram iguais aos do HMM: snapshot, corte de 95%, tolerância relativa 1e-8, `max_iter` 1000, piso 1e-10, semente 0.

Duas conferências de que o procedimento reproduz o do HMM: a autocorrelação real (0,157; 0,154; 0,148; 0,140; 0,115; 0,101) e a LL da cadeia de 1ª ordem (-1,0600) coincidem com as do relatório do HMM, e o HMM recalculado dá -1,01545 no teste, contra -1,01544 no original.

---

## 1. Resumo

1. **No modo `blocks`, o HMM ainda ganha do MTD, mas por pouco, e o MTD ganha da cadeia de 1ª ordem em todas as ordens.** Teste: HMM -1,0154 (14 parâmetros), MTD de ordem 10 -1,0236 (15), MTD de ordem 6 -1,0321 (11), cadeia de 1ª ordem -1,0600 (6), iid -1,0753. A persistência (ε = 0,01) fica em -2,7432, pior que o uniforme (-1,0986), e serve só como baseline de acerto.
2. **O MTD e o HMM contam a mesma história: o que se aprende é aglomeração de volatilidade.** `Q` diz que "calma puxa calma" (P(0|0) = 0,78) e "movimento puxa movimento". Nenhum dos dois modelos acerta direção de forma convincente.
3. **Direção: nenhum modelo tem vantagem sobre uma constante.** O MTD acerta 51,2% a 51,5% (ordens 2 a 10) no `blocks`. A melhor constante (sempre −1) acerta 50,9% e o HMM, 50,1%. A diferença entre o MTD e a constante é de 1 a 2 erros padrão, e a constante foi escolhida olhando o teste. No `global` o MTD fica em 50,4% a 51,0%, dentro do ruído.
4. **O BIC do MTD não tem mínimo: está sempre na borda da busca (ordem 10), nos dois modos.** O HMM de 3 estados tem BIC 12.874 menor que o do MTD de ordem 10.
5. **A memória do MTD é de minutos; a do HMM é de horas.** Com ordem 10, o tempo de relaxação é de 57 min (`blocks`). Nos dados simulados, o MTD de ordem 10 reproduz a autocorrelação do movimento até cerca de 30 min, caindo a 0,084 em 1 h (real: 0,140) e a zero em 4 h (real: 0,115). O HMM chegava a 0,088 em 4 h.
6. **O EM do MTD é muito mais estável que o do HMM.** Em todas as ordens e nos dois modos, **10 de 10 reinícios** chegam à mesma solução (diferença ≤ 0,04 nats) e convergem. No HMM de 3 estados, foram 3 de 10, e a solução ruim ficava 7.892 nats abaixo.
7. **A sazonalidade aparece nos dois modelos, e no MTD aparece mais.** A P(0) média prevista por hora acompanha a observada com correlação 0,987 (MTD ordem 6) e 0,972 (ordem 10), contra 0,865 do HMM. Todos subestimam a amplitude (previsto 7% a 10%, observado 14,5%), e o HMM acerta melhor o nível do fim de semana.
8. **O modo `global` mistura estrutura com mudança de distribuição.** O teste tem 68% de zeros (treino: 51%). O iid sozinho "ganha" 0,191 nats sobre o uniforme, e a calibração do MTD subestima o 0 em todos os decis. O `blocks` é a referência honesta; o `global` serve como contraste.
9. **As lacunas não importam para o MTD.** Até 340 transições (0,04%) atravessam uma lacuna, e retirá-las muda a LL por símbolo em menos de 5·10⁻⁵.

---

## 2. Dados e configuração

| Item | `blocks` | `global` |
|---|---|---|
| Snapshot | `btc_usdt_2026-10-01.parquet` (SHA-256 `4c71d149…bd4d`), 957.853 candles | idem |
| Discretização | blocos de 1 mês, limiar s/3, treino 95% | μ e σ só do treino, s/3, treino 95% |
| Treino | 905.654 símbolos, 35 segmentos | 909.925 símbolos, 35 segmentos |
| Teste | 47.893 símbolos, de 2026-04-17 16:55 a 2026-09-30 23:55 | idem |
| Frequências treino (−1/0/+1) | 28,78% / 42,33% / 28,89% | 24,39% / 51,02% / 24,58% |
| Frequências teste (−1/0/+1) | 28,57% / 43,83% / 27,61% | 16,18% / 67,96% / 15,85% |
| MTD | ordens 1 a 10, 10 reinícios, tolerância 1e-8, `max_iter` 1000, piso 1e-10, semente 0 | idem |

Todos os modelos são avaliados nas **mesmas transições**: no treino, as últimas `N − 10` (905.644 no `blocks`); no teste, todas as 47.893, com o contexto vindo do fim do treino. Sem isso, o BIC e a LL não seriam comparáveis entre ordens.

---

## 3. Quanto o MTD explica: baselines (`blocks`)

| Modelo | Parâmetros | Treino | Teste | BIC (treino) |
|---|---:|---:|---:|---:|
| Uniforme | 0 | -1,0986 | -1,0986 | — |
| Frequências (iid) | 2 | -1,0811 | -1,0753 | 1.958.146,0 |
| Persistência (ε = 0,01) | 0 | -2,8196 | -2,7432 | — |
| Cadeia de 1ª ordem (= MTD ordem 1) | 6 | -1,0678 | -1,0600 | 1.934.137,9 |
| MTD ordem 3 | 8 | -1,0521 | -1,0441 | 1.905.688,5 |
| MTD ordem 6 | 11 | -1,0402 | -1,0321 | 1.884.217,2 |
| MTD ordem 10 | 15 | -1,0328 | -1,0236 | 1.870.878,8 |
| **HMM 3 estados** | 14 | **-1,0257** | **-1,0154** | **1.858.005,0** |

- **Sanidade.** O MTD de ordem 1 é exatamente a cadeia de Markov de 1ª ordem (mesma LL e mesmo BIC), como deveria ser.
- **Persistência.** Com ε = 0,01 ela dá -2,74 nats por símbolo no teste, 1,64 pior que o uniforme. É um baseline de acerto (acerta 40,6% das posições), não de verossimilhança.
- **De onde vem o ganho.** No treino, o MTD de ordem 10 ganha 0,0658 nats sobre o uniforme. Desse total, 0,0175 (27%) vêm só das frequências e 0,0483 (73%) da dinâmica. No HMM, a divisão é 24% / 76%. No teste, o ganho de dinâmica sobre o iid é de 0,0518 (MTD ordem 10) e 0,0599 (HMM).
- **Cada ordem a mais melhora a LL do teste e o BIC.** De -1,0600 (ordem 1) a -1,0236 (ordem 10), com apenas 9 parâmetros a mais (6 → 15). O BIC cai de 1.934.138 para 1.870.879, e a penalidade `p·ln n` fica abaixo de 206.
- **O HMM fica 0,0082 nats por símbolo à frente do MTD de ordem 10 no teste** (cerca de 390 nats no total), com um parâmetro a menos. No BIC, a vantagem é de 12.874.
- **Comparabilidade.** Agora os dois estão nas mesmas transições e o HMM começa o teste com o estado do fim do treino. O BIC do HMM sai 1.858.005,0 aqui, contra 1.858.021,8 no relatório dele, porque lá a LL de treino soma `T` termos e aqui somam-se as mesmas 905.644 transições do MTD.
- **O teste "mais fácil" que o treino não é sobreajuste, nem aqui.** Todos os modelos melhoram do treino para o teste (o iid melhora 0,0057, o MTD de ordem 10, 0,0092). O período de teste tem mais zeros (43,8% contra 42,3%).

---

## 4. Escolha da ordem

| Ordem | Parâmetros | BIC (treino) | Teste | Ganho do teste em relação à ordem anterior |
|---:|---:|---:|---:|---:|
| 1 | 6 | 1.934.137,9 | -1,0600 | — |
| 2 | 7 | 1.917.558,6 | -1,0506 | 0,0094 |
| 4 | 9 | 1.897.243,1 | -1,0390 | 0,0050 (de 3 para 4) |
| 6 | 11 | 1.884.217,2 | -1,0321 | 0,0033 (de 5 para 6) |
| 8 | 13 | 1.876.738,6 | -1,0274 | 0,0018 (de 7 para 8) |
| 10 | 15 | **1.870.878,8** | **-1,0236** | 0,0018 (de 9 para 10) |

- **O BIC cai em toda ordem, nos dois modos.** A penalidade `p·ln n` vale no máximo 206 e cada ordem a mais ganha milhares de nats de LL. O BIC aqui quase não penaliza. O resultado honesto é que **o BIC pede mais lags do que o escopo cobre**: escolher a ordem por BIC entre 2 e 6 devolve 6 por construção.
- **O ganho por lag é decrescente** (0,0094 de 1 para 2; 0,0018 de 9 para 10), mas não chegou a zero. O MTD não encontrou um "joelho" em 50 min de janela.
- **Contraste com o HMM.** No HMM, 4 estados já era o ponto em que o ganho caía a 0,0036 por símbolo e 5 estados falhou por otimização. No MTD, a otimização nunca falha (seção 7), então o que limita a ordem é só o custo de ajuste (de 14 a 55 s por ordem, a partir da ordem 2, com 10 reinícios).
- **Hipótese (não testada):** como `Q` é única para todos os lags, aumentar a ordem é a única forma de o MTD "lembrar mais", e ele paga com mais lags uma memória que o HMM guarda em um estado oculto. Uma variante com uma `Q` por lag (MTDg) permitiria separar a reversão de curto prazo da aglomeração de volatilidade. Não foi testado.

---

## 5. Estrutura do modelo (`blocks`, ordem 10)

`Q` (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,345 | 0,137 | **0,518** |
| 0 | 0,107 | **0,782** | 0,110 |
| +1 | **0,492** | 0,189 | 0,320 |

Pesos por lag: 0,134, 0,121, 0,111, 0,099, 0,097, 0,100, 0,083, 0,085, 0,089, 0,083.

- **Magnitude.** Depois de um 0, a chance de outro 0 é 78%; depois de ±1, é de 14% a 19%. É a assinatura da aglomeração de volatilidade, a mesma que o HMM representa com a escada calmo–intermediário–agitado (P(0) de 0,77 a 0,22).
- **Memória longa e quase plana.** Os pesos decaem devagar, de 0,134 (lag 1) a 0,083 (lag 10). O modelo usa a janela inteira.
- **Distribuição implicada.** A distribuição estacionária de `Q` (28,5% / 42,8% / 28,7% na ordem 10) é próxima da do treino (28,8% / 42,3% / 28,9%), mas **se afasta um pouco a cada ordem** (o 0 sobe de 42,4% na ordem 2 para 42,8% na 10). No HMM, a distribuição implicada coincidia com o treino até a 4ª casa. É um sinal pequeno de que o MTD, que usa uma `Q` única, distorce levemente as frequências marginais.
- **Direção (reversão).** `Q` tem a reversão de curto prazo: depois de −1, +1 (0,518) é mais provável que −1 (0,345), e o inverso vale depois de +1. Os dados de teste confirmam o sinal: depois de −1, +1 vem em 32,6% e −1 em 31,0%; depois de +1, −1 vem em 34,1% e +1 em 29,9%.
- **Tempo de memória.** Como a mistura é linear nos indicadores dos lags, o desvio da marginal em relação à estacionária obedece a uma recorrência linear, e o seu decaimento sai das raízes de `r^l = μ·Σ λ_g r^(l−g)`, com `μ` o segundo autovalor de `Q`:

| Ordem | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| Relaxação, `blocks` (min) | 3 | 6 | 15 | 27 | 41 | 57 |
| Relaxação, `global` (min) | 3 | 7 | 19 | 35 | 55 | 77 |

  Para comparar: o HMM tem tempos de relaxação de 8,0 h e 3,9 h. **Mesmo com 10 lags, o MTD esquece o contexto em cerca de 1 hora.** Isso confirma o diagnóstico da seção 4 e explica o resultado da seção 9.

---

## 6. Desempenho preditivo no teste (um passo à frente, `blocks`)

| Modelo | LL/símbolo | Acurácia | Acerto de direção (IC 95%) | AUC movimento | Brier |
|---|---:|---:|---|---:|---:|
| Sempre 0 | — | 43,83% | — | — | — |
| Persistência | — | 40,61% | — | — | — |
| Persistência (ε = 0,01) | -2,7432 | 40,61% | 48,79% (48,2 a 49,4) | 0,587 | 1,153 |
| Cadeia de 1ª ordem (= MTD ordem 1) | -1,0600 | 43,83% | 51,21% (50,6 a 51,8) | 0,587 | 0,639 |
| MTD ordem 2 | -1,0506 | 45,23% | 51,53% (50,9 a 52,1) | 0,622 | 0,632 |
| MTD ordem 6 | -1,0321 | 46,51% | 51,29% (50,7 a 51,9) | 0,665 | 0,619 |
| MTD ordem 10 | -1,0236 | 47,11% | 51,17% (50,6 a 51,8) | 0,679 | 0,613 |
| **HMM 3 estados** | **-1,0154** | **47,61%** | 50,13% (49,5 a 50,7) | **0,695** | **0,608** |

### 6.1 Movimento: o HMM ainda prevê melhor

- **AUC de movimento** (separar "o candle sai da faixa ±s/3" de "fica em 0"): 0,695 (HMM), 0,679 (MTD ordem 10), 0,665 (MTD ordem 6), 0,587 (cadeia de 1ª ordem). O MTD sobe a cada ordem e não alcança o HMM.
- **Calibração.** A P(0) prevista pelo MTD é bem calibrada, exceto no último decil. Com ordem 10: decil 5, previsto 0,402 e observado 0,402; decil 9, 0,608 e 0,613; **decil 10, previsto 0,716 e observado 0,758**. O MTD subestima os momentos de calma extrema (P(0) vai de 0,23 no p10 a 0,65 no p90, contra 0,23 a 0,72 do HMM). Isso é consistente com a janela curta: o MTD só sabe que "os últimos 50 min foram calmos", enquanto o HMM sabe que "o regime é calmo há horas".

### 6.2 Direção: sem vantagem sobre uma constante

Só as posições em que o real é −1 ou +1 (26.903 no teste). Regra de decisão: maior entre −1 e +1.

| | Acerto |
|---|---:|
| Sempre `+1` (majoritário do treino) | 49,14% |
| Sempre `−1` (melhor constante **no teste**) | 50,86% |
| MTD, ordens 2 a 10, teste | 51,17% a 51,53% |
| MTD, ordens 2 a 10, treino | cerca de 52,6% a 52,8% (ordens 6 e 10) |
| HMM 3 estados, teste | 50,13% |

- O erro padrão é de cerca de 0,3 p.p. O MTD fica 0,3 a 0,7 p.p. acima da melhor constante, a 1 ou 2 erros padrão, e a constante foi escolhida olhando o teste. **Não há evidência de vantagem direcional.**
- O acerto **não cresce com a ordem** (51,5% na ordem 2 e 51,2% na 10). A ordem melhora a LL, que é magnitude, e não melhora a direção. É o oposto do que ocorreria se mais contexto trouxesse direção.
- A diferença para o HMM (50,1% contra 51,2%–51,5%) é a reversão que o MTD capta e o HMM de 3 estados não: todos os estados do HMM são simétricos. O HMM está na faixa do acaso, e o MTD está um pouco acima, mas dentro do que um único trecho de teste não separa de uma constante.
- No treino o MTD acerta cerca de 52,7% e no teste 51,2%; com 15 parâmetros e 905 mil símbolos isso não é sobreajuste, é o período do teste com menos reversão que a média.
- **Acurácia por argmax.** O MTD ordem 10 prevê 0 em 68,6% dos candles e acerta 47,1%; o HMM prevê 0 em 67,7% e acerta 47,6%. Quando prevê um símbolo diferente de 0, o MTD acerta cerca de 37,5% (o HMM, 37,8%). Esse patamar é a frequência do símbolo, não habilidade. A acurácia continua dominada pela classe 0 e não separa os modelos.
- Acerto de direção em candles de 5 minutos não mede o mercado "sobe ou desce" (que inclui os candles em 0), e custos e spread não foram considerados.

---

## 7. Superfície de otimização: o EM do MTD não tem armadilhas

| | MTD (cada ordem) | HMM 3 estados |
|---|---|---|
| Reinícios | 10 | 10 |
| Chegam à melhor solução (±1 nat) | **10 de 10**, em todas as ordens, nos dois modos | 3 de 10 |
| Convergiram | 10 de 10 | 3 de 10 (7 pararam em `max_iter`) |
| Pior reinício a menos de | 0,04 nats (`blocks`), 0,03 (`global`) | 7.894 nats |
| Passos do EM com LL caindo | 0 | — |

- Isso era esperado: o MTD tem poucos parâmetros (`Q`, `lam`) e o passo E tem uma estrutura simples. Aqui os reinícios aleatórios não trazem risco de reprodutibilidade, ao contrário do HMM, em que 7 de 10 reinícios ficaram presos numa família de soluções "direcionais" 7.892 nats abaixo.
- O reinício 0 usa `lam` uniforme e `Q` igual à cadeia de ordem 1 empírica, e a ordem 1 converge em 1 iteração (é o ótimo em forma fechada). Os melhores reinícios das outras ordens variam (1, 3, 8), mas a diferença entre eles é de centésimos de nat.
- **Consequência prática:** para o MTD, 2 a 3 reinícios bastam. Os 10 desta análise não eram necessários. Para o HMM, são.

---

## 8. Sazonalidade: o que o MTD "sabe" sobre o relógio

Nenhum dos dois modelos recebe o horário. A tabela compara a P(0) média prevista com a frequência observada de 0, por hora UTC (série inteira, treino e teste):

| Hora UTC | Observada | MTD ordem 6 | MTD ordem 10 | HMM 3 estados |
|---:|---:|---:|---:|---:|
| 00 | 37,69% | 40,79% | 40,95% | 41,09% |
| 05 | **49,00%** | 45,99% | 46,76% | 46,37% |
| 08 | 44,58% | 43,87% | 44,42% | 45,41% |
| 14 | 34,55% | 38,69% | 38,11% | 39,13% |
| **15** | **35,35%** | 38,59% | 37,72% | 37,61% |
| 17 | 39,69% | 40,65% | 39,97% | 37,99% |
| 20 | 41,56% | 42,09% | 41,98% | 41,08% |

| Modelo | Correlação com a observada (24 horas) | Amplitude prevista | Amplitude observada |
|---|---:|---:|---:|
| MTD ordem 6 | **0,987** | 7,4% | 14,5% |
| MTD ordem 10 | 0,972 | 9,0% | 14,5% |
| HMM 3 estados | 0,865 | 9,5% | 14,5% |

- **O MTD recupera o desenho do dia** (calma de madrugada UTC, agitação das 13h às 17h UTC, a abertura dos EUA) tão bem quanto o HMM, e com correlação maior. Mas **subestima a amplitude** (prevê metade da variação que existe), e o HMM também. A correlação alta com amplitude baixa quer dizer: acerta a forma, não a escala.
- **Fim de semana:** observado 52,3% (sábado) e 49,7% (domingo) de zeros. O HMM prevê 49,6% e 48,4%; o MTD de ordem 6, 47,5% e 46,3%. O HMM acerta melhor o nível, porque o regime calmo é persistente por horas. A correlação por dia da semana (1,000 para o MTD) é inflada por haver só 7 pontos, dois deles muito diferentes dos outros.
- **Interpretação cuidadosa:** esta é uma medida na amostra inteira (treino e teste). Ela mostra que a sazonalidade está embutida na sequência de símbolos, não que o modelo a use como previsão. Um modelo com a hora como variável (ver recomendações) separaria "sazonalidade" de "dinâmica própria".

---

## 9. Horizonte e adequação

### 9.1 Horizonte (mesmos 47.893 alvos; informação até t, alvo em t + h)

| Horizonte | HMM AUC | HMM LL | MTD 10 AUC | MTD 10 LL | MTD 6 AUC | MTD 6 LL | iid LL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,695 | -1,0154 | 0,679 | -1,0236 | 0,665 | -1,0321 | -1,0753 |
| 15 min | 0,690 | -1,0187 | 0,669 | -1,0303 | 0,652 | -1,0415 | -1,0753 |
| 1 h | 0,679 | -1,0259 | 0,655 | -1,0465 | 0,638 | -1,0646 | -1,0753 |
| 4 h | 0,644 | -1,0448 | 0,618 | -1,0738 | 0,606 | -1,0752 | -1,0753 |
| 1 dia | 0,614 | -1,0724 | 0,596 | -1,0751 | 0,520 | -1,0752 | -1,0753 |

- **O HMM perde a previsão devagar; o MTD a perde rápido.** O ganho de LL sobre o iid, no HMM, vai de 0,060 (5 min) a 0,049 (1 h) e 0,030 (4 h). No MTD de ordem 10, vai de 0,052 a 0,029 (1 h) e **0,0015** (4 h). A 1 hora de distância o HMM (-1,0259) prevê quase tão bem quanto o MTD de ordem 10 a 5 minutos (-1,0236).
- **Cuidado com o AUC a partir de 4 h no MTD.** A previsão do MTD converge para a distribuição estacionária e, em 1 dia, ela é **constante até 10⁻¹² (ordem 10) ou 10⁻¹⁶ (ordem 6)**. O AUC de 0,52 (ordem 6, `blocks`) e de 0,48 (ordem 6, `global`) é ordenação de diferenças do tamanho do ruído de ponto flutuante, não informação. O AUC só mede a ordem das previsões, e não a sua escala; a LL, que não sai do nível do iid (-1,0751 contra -1,0753), é a leitura correta. A 4 h, o desvio padrão da P(0) prevista é 3·10⁻⁵ na ordem 6 e 3·10⁻³ na ordem 10. No HMM, o AUC de 0,614 em 1 dia vem com ganho de LL real (0,003 nats), porque o regime ainda carrega informação.
- Esse cálculo é exato, não uma simulação: como a mistura é linear nos lags, a marginal de x[t+h] vem de uma recorrência sobre as marginais dos `ordem` passos anteriores (testada contra a enumeração de todos os caminhos futuros).

### 9.2 Autocorrelação real contra simulada (treino, `blocks`)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,157 | 0,154 | 0,148 | 0,140 | 0,115 | 0,101 |
| Movimento, MTD ordem 6 | 0,160 | 0,155 | 0,145 | 0,037 | 0,000 | 0,001 |
| Movimento, MTD ordem 10 | 0,164 | 0,158 | 0,149 | 0,084 | 0,002 | −0,001 |
| Movimento, HMM (relatório do HMM, semente 1) | 0,144 | 0,144 | 0,136 | 0,130 | 0,088 | 0,008 |
| Símbolo, real | **−0,047** | −0,031 | −0,007 | 0,000 | 0,002 | −0,002 |
| Símbolo, MTD ordem 1 | −0,047 | 0,002 | 0,000 | −0,002 | −0,001 | 0,000 |
| Símbolo, MTD ordem 10 | −0,020 | −0,020 | −0,015 | 0,001 | 0,000 | 0,000 |

- **O MTD reproduz a aglomeração de volatilidade até o fim da sua janela** (5 min a 30 min) e a perde depois: a 1 hora ele tem 0,037 (ordem 6) e 0,084 (ordem 10), contra 0,140 real. O HMM mantém 0,130 em 1 h e 0,088 em 4 h. A memória real é de dias, e **nenhum dos dois** chega a 1 dia (real 0,101; HMM 0,008; MTD zero).
- **A reversão do sinal no lag 1 é o ponto fraco do MTD de ordem alta.** O MTD de ordem 1 reproduz −0,047 exatamente, mas as ordens 6 e 10 dão −0,024 e −0,020. Como `Q` é única, a reversão do lag 1 é diluída entre todos os lags. Este é o sintoma da limitação estrutural levantada na seção 4: ou o modelo acerta a reversão (ordem 1) ou a volatilidade (ordem alta), e não os dois.
- O HMM, por outro lado, simula símbolo com autocorrelação ≈ 0 em todos os lags (não tem reversão).
- Observação: a simulação do HMM é a do relatório dele, que não foi reexecutada.

### 9.3 Lacunas

O treino concatena 35 segmentos, e o `fit` do MTD conta como normais as transições cujo contexto atravessa uma lacuna.

| Ordem | Transições afetadas | Fração | LL por símbolo (todas / sem as afetadas) |
|---:|---:|---:|---|
| 1 | 34 | 0,00% | -1,06778 / -1,06777 |
| 6 | 204 | 0,02% | -1,04018 / -1,04017 |
| 10 | 340 | 0,04% | -1,03279 / -1,03277 |

A diferença é de no máximo 2·10⁻⁵ por símbolo (`blocks`) e 5·10⁻⁵ (`global`). Isto mede a LL do modelo ajustado com e sem essas transições; **não** refaz o ajuste excluindo-as, o que exigiria o `fit` aceitar segmentos (pendência D6). Dado o tamanho do efeito, não há motivo para esperar diferença relevante.

---

## 10. Modo `global`: o contraste

O `global` usa μ e σ do treino para todo o período. O teste (2026) tem volatilidade muito menor que a média do treino e, com o limiar fixo, **68% dos símbolos de teste são 0** (51% no treino).

| | Treino | Teste |
|---|---:|---:|
| Zeros | 51,02% | 67,96% |
| iid (LL por símbolo) | -1,0324 | -0,9081 |
| MTD ordem 6 | -0,9660 | -0,8143 |
| MTD ordem 10 | -0,9560 | -0,8021 |

- **O ganho de teste do MTD ordem 10 sobre o uniforme é 0,296 nats, mas 0,191 vêm só da mudança de distribuição** (o iid já "ganha" isso) e 0,106 vêm de estrutura. No `blocks`, o ganho sobre o iid é 0,052. Por isso as LL e os BIC **não se comparam entre os dois modos**.
- **A calibração mostra o desvio de distribuição.** Em todos os decis a frequência observada de 0 é maior que a prevista (ordem 10: decil 1, previsto 0,321 e observado 0,418; decil 5, 0,628 e 0,659). O modelo, treinado em um período mais agitado, subestima a calma. No `blocks`, em que o σ do bloco anterior acompanha a escala da volatilidade, não há esse viés. (No `global`, os decis 9 e 10 têm P(0) prevista idêntica na ordem 6, 0,809: as previsões saturam no valor "todos os lags em 0" e o empate é desfeito pela ordem das posições.)
- **O MTD no `global` leva a mesma leitura.** BIC mínimo na ordem 10 (1.740.003,1, contra 1.835.462,6 da cadeia de 1ª ordem), LL de teste de -0,8143 (ordem 6) contra -0,8707 (1ª ordem), AUC de movimento de 0,684 (ordem 6) e 0,697 (ordem 10), relaxação de 35 a 77 min, 10 de 10 reinícios iguais.
- **Direção no `global`:** 50,4% a 51,0% (ordens 2 a 10), contra a melhor constante de 50,5% (sempre −1). Sem sinal.
- **O HMM não foi treinado no `global`.** Sem ele, não há comparação entre os modelos nesse modo. Isso é o que falta para fechar a comparação (ver seção 12).

---

## 11. Conclusões

1. **Comparação com o HMM (`blocks`).** Em previsão de um passo, o HMM é melhor em tudo que importa para volatilidade (LL -1,0154 contra -1,0236; AUC 0,695 contra 0,679; Brier 0,608 contra 0,613) e perde na única coisa que o MTD capta, a reversão de curto prazo (direção 50,1% contra 51,2%, ambos dentro do ruído). A distância é de 0,008 nats por símbolo no teste e de 12.874 no BIC. **A distância cresce com o horizonte**: o HMM prevê 1 hora à frente quase tão bem quanto o MTD prevê 5 minutos.
2. **Os dois modelos concordam no diagnóstico.** A estrutura da série de 5 minutos é aglomeração de volatilidade: o MTD a representa com uma janela de lags, o HMM com estados persistentes, e ambos recuperam a sazonalidade diária e semanal sem receber o horário. Dois modelos de famílias diferentes chegando à mesma leitura reforçam o resultado.
3. **O MTD cumpre o papel no escopo.** É a família de ordem alta da comparação: melhora sobre todos os baselines do escopo (uniforme, persistência, iid e cadeia de 1ª ordem) em LL, AUC e Brier, tem EM estável e produz um BIC que melhora até a borda da busca. Mas "ordem escolhida por BIC" não tem resposta interna (sempre a maior); só dá para dizer que o BIC pede mais lags do que 10.
4. **O MTD é um modelo de memória curta.** Mesmo com 10 lags, o tempo de relaxação é de cerca de 1 hora e a autocorrelação simulada cai a zero em 4 h. É a limitação que o HMM, com estados que duram horas, não tem.
5. **Nenhum dos dois serve para prever direção.** Acerto de direção em 50,1% a 51,5% nos dois modos, igual ou perto da melhor constante. A reversão captada pelo MTD (depois de −1, +1 em 32,6% e −1 em 31,0%) é real, mas fraca.
6. **Nada aqui é leitura financeira.** Custos, spread e o payoff do mercado de previsão não entraram.

## 12. Limitações e próximos passos

- **HMM no `global`:** treinar com `python -m src.training.hmm_training --snapshot snapshot/btc_usdt_2026-10-01.parquet --mode global` (cerca de 5 minutos) e rodar `mtd_analysis` de novo; o script já inclui o HMM sozinho se encontrar `relatorios/hmm_2026-10-01_global_k3.json`. Sem isso, o modo `global` só tem o MTD.
- **Só s/3.** Os limiares s/4 e s/5 do MTD estão em `analise_mtd_2026-10-01.md`, sem as métricas novas. O HMM também só foi treinado em s/3.
- **Um único trecho de teste** (47.893 símbolos, cerca de 166 dias). Diferenças menores que 0,3 p.p. de acerto ou 0,005 nats não são distinguíveis.
- **A sazonalidade é medida na série inteira**, com o modelo ajustado no treino. Ela mostra o que o modelo reproduz, não o que ele prevê fora da amostra.
- **A simulação do HMM (seção 9.2) é a citada do relatório dele**, com outra semente e implementação; a do MTD é nova.
- **Lacunas (D6):** o `fit` do MTD e do HMM ainda não aceita segmentos. O efeito medido é desprezível.
- **Combinação e MTDg:** os caminhos da seção 7 de `analise_mtd_2026-10-01.md` (HMM diz *quanto*, MTD diz *para onde*; direção condicionada ao regime; modelo único com regime e lags) seguem em aberto e dependem do filtro preditivo do HMM, que agora existe em `mtd_analysis.hmm_pred` (só como função da análise, não na classe `HMMDiscreto`).

## Apêndice: reprodução

```
python -m src.experiments.mtd_analysis --snapshot snapshot/btc_usdt_2026-10-01.parquet \
    --mode blocks global --k 3 --max-order 10 --restarts 10 --cache <pasta de cache>
```

- Código: `src/experiments/mtd_analysis.py`; testes: `tests/test_mtd_analysis.py` (36 testes). As partes numéricas mais delicadas foram conferidas contra cálculo direto: AUC contra contagem de pares (com empates), marginais de horizonte contra enumeração de todos os caminhos futuros, filtro do HMM contra enumeração de todos os caminhos de estados, tempo de memória contra a recorrência exata, simulador contra a mistura teórica.
- O `--cache` guarda `lam` e `Q` por ordem, para refazer só as métricas sem repetir o EM (cerca de 10 minutos). A chave do cache inclui o SHA do snapshot e o tamanho do treino.
- Todos os modelos usam a semente 0 e o mesmo critério de parada do HMM. O EM do MTD é determinístico dada a semente.
- Ambiente: Python 3.14, numpy 2.3.2, numba 0.67, 8 threads. Suíte completa: 350 testes passando.
