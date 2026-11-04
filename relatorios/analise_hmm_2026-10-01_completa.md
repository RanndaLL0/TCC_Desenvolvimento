# Análise completa do HMM — snapshot 2026-10-01, blocos, s/3

Esta análise parte de `relatorios/hmm_2026-10-01_blocks_k3.json` (HMM de 3 estados, 10 reinícios) e acrescenta experimentos para responder ao que aquele arquivo, sozinho, não responde:

- **Baselines:** uniforme, frequências e cadeias de Markov de ordem 1 a 6, ajustadas no mesmo treino e avaliadas no mesmo teste.
- **Outros números de estados:** HMMs de 2, 4 e 5 estados, com 6 reinícios cada.
- **Regimes nos dados:** a decodificação dos regimes (Viterbi e posteriores) cruzada com o retorno, a hora, o dia da semana e o ano.
- **Previsão no teste:** um passo à frente, com calibração e horizontes de até 1 dia.
- **Reinício preso:** o reinício 2, que parou em `max_iter`, continuado até a convergência.
- **Adequação e lacunas:** a autocorrelação dos dados reais contra a de dados simulados pelo modelo, e o efeito das lacunas.

Nenhum código do repositório foi alterado. Os experimentos usaram `src/training/hmm_training.py`, `src/preprocess/discretize.py` e `src/models/hmm/hmm_numba.py` como estão, e scripts de análise avulsos (ver Apêndice). A análise anterior, restrita ao JSON, continua em `relatorios/analise_hmm_2026-10-01_blocks_k3.md`. Este documento a amplia e, em alguns pontos, corrige.

Unidade: log-verossimilhança **por símbolo, em nats** (log natural). Quanto mais perto de zero, melhor. Para ter uma referência, um modelo que não sabe nada sobre 3 símbolos fica em `-ln 3` = -1,0986.

---

## 1. Resumo

1. **O HMM explica bem mais que as cadeias de Markov.** No teste, o HMM de 3 estados (14 parâmetros) atinge -1,0154 por símbolo. A melhor cadeia de Markov testada, a de ordem 6 com 1.458 parâmetros, fica em -1,0315. Cerca de 76% do ganho do HMM sobre o uniforme vem da dinâmica, e não apenas de o símbolo 0 ser mais frequente.
2. **Os estados são regimes de volatilidade, não de direção.** Calmo, intermediário e agitado diferem na chance de o candle sair da faixa ±s/3, e em nenhum deles alta e queda se distinguem. O retorno absoluto médio no regime agitado é 3,5 vezes o do calmo.
3. **A dinâmica tem forma de escada**, sem saltos diretos entre calmo e agitado. A mesma estrutura reaparece, de forma independente, no modelo de 4 estados (tridiagonal). É o resultado mais robusto da análise.
4. **Os regimes seguem o relógio do mercado.** O regime agitado domina das 13h às 17h UTC (abertura dos EUA), e o calmo, de madrugada UTC e nos fins de semana. O modelo descobriu essa sazonalidade sem ter acesso ao horário.
5. **O HMM é um bom previsor de *se* haverá movimento, mas não de *para onde*:**
   - **Movimento:** AUC de 0,70 para separar movimento de candle parado, contra 0,59 da Markov de ordem 1. As probabilidades são bem calibradas.
   - **Direção:** acerto de 50,1%, igual ao acaso, contra 51,2% a 51,5% das cadeias de Markov, que captam uma reversão de curto prazo que o HMM de 3 estados ignora.
6. **O BIC prefere 4 estados a 3**, mas o ganho prático é pequeno (0,0036 nats por símbolo no teste). O ajuste de 5 estados não convergiu e ficou *pior* que o de 4, o que só pode ser falha de otimização: não dá para concluir nada sobre 5 estados.
7. **O EM tem pelo menos duas famílias de ótimos locais.** Uma é a "escada de volatilidade", a boa. A outra é uma solução "direcional" que fica 7.892 nats abaixo e, continuada, **converge** lá (não é só lentidão). Com 3 estados, 7 de 10 reinícios caíram na solução ruim; com 4 estados, 5 de 6.
8. **Limitações do modelo:** a memória da volatilidade nos dados dura dias, e a do HMM some em cerca de 8 h. O HMM também não reproduz a reversão do sinal. As lacunas e a distribuição inicial `pi` degenerada têm efeito numérico desprezível.

---

## 2. Dados e configuração

| Item | Valor |
|---|---|
| Snapshot | `btc_usdt_2026-10-01.parquet` (SHA-256 `4c71d149…bd4d`), 957.853 candles de 5 min, de 2017-08-17 a 2026-09-30 |
| Discretização | modo `blocks` (1 mês), limiar s/3, treino 95% |
| Treino | 905.654 símbolos (linhas 0 a 909.959; 4.306 posições sem símbolo: primeiro bloco, primeiro candle e lacunas) |
| Teste | 47.893 símbolos, de 2026-04-17 16:55 a 2026-09-30 (cerca de 166 dias, sem lacunas) |
| Frequências (−1 / 0 / +1) | treino 28,78% / 42,33% / 28,89%; teste 28,57% / 43,83% / 27,61% |
| HMM | tolerância relativa 1e-8, `max_iter` 1000, `piso` 1e-10, semente 0, 12 threads |

---

## 3. Quanto o HMM explica: comparação com baselines

Os modelos de Markov foram ajustados por contagem no treino, com suavização de Laplace (+1). No teste, o contexto inicial vem do fim do treino. O BIC usa o número de parâmetros livres de cada modelo.

| Modelo | Parâmetros | Treino | Teste | BIC (treino) |
|---|---:|---:|---:|---:|
| Uniforme | 0 | -1,0986 | -1,0986 | — |
| Frequências do treino (iid) | 2 | -1,0811 | -1,0753 | — |
| Markov ordem 1 | 6 | -1,0678 | -1,0600 | 1.934.153,6 |
| Markov ordem 2 | 18 | -1,0583 | -1,0502 | 1.917.112,5 |
| Markov ordem 3 | 54 | -1,0510 | -1,0433 | 1.904.356,1 |
| Markov ordem 4 | 162 | -1,0459 | -1,0382 | 1.896.621,3 |
| Markov ordem 6 | 1.458 | -1,0372 | -1,0315 | 1.898.630,2 |
| **HMM 3 estados** | **14** | **-1,0257** | **-1,0154** | **1.858.021,8** |

- **De onde vem o ganho.** No treino, o HMM ganha 0,0729 nats por símbolo sobre o uniforme. Desse total, 0,0175 (24%) vêm só das frequências, isto é, de o 0 ser mais comum. Os outros 0,0554 (76%) vêm da dinâmica. No teste, os números são 0,0832 de ganho total e 0,0599 sobre o iid.
- **O HMM bate a Markov de ordem 6 por larga margem.** São 0,016 nats por símbolo no teste, com 1% dos parâmetros. A Markov de ordem *k* só enxerga os últimos *k* candles (30 min, na ordem 6). O HMM resume no estado oculto uma memória de horas, e é ela que importa aqui.
- **O teste "mais fácil" que o treino não é sobreajuste.** *Todos* os modelos melhoram no teste: o iid melhora 0,0057, a Markov de ordem 1, 0,0078, e o HMM, 0,0102. O período de teste é ligeiramente mais concentrado no 0 (43,8% contra 42,3%), e o efeito vem do período, não do modelo.

---

## 4. Escolha do número de estados

Os modelos de 2, 4 e 5 estados foram treinados com 6 reinícios cada. O de 3 estados é o do relatório original, com 10 reinícios.

| Estados | Parâmetros | Treino | Teste | AIC | BIC | Convergiu | Reinícios na melhor solução | Tempo |
|---:|---:|---:|---:|---:|---:|---|---|---:|
| 2 | 7 | -1,03502 | -1,02524 | 1.874.744,3 | 1.874.826,3 | sim | 4 de 6 | 68 s |
| 3 | 14 | -1,02568 | -1,01544 | 1.857.857,8 | 1.858.021,8 | sim | 3 de 10 | 283 s |
| **4** | 23 | **-1,02275** | **-1,01182** | **1.852.551,4** | **1.852.820,9** | sim | **1 de 6** | 278 s |
| 5 | 34 | -1,02435 | -1,01534 | 1.855.484,6 | 1.855.883,0 | **não** | 0 de 6 convergiram | 526 s |

- **De 2 para 3 estados** o ganho é grande: ΔBIC = −16.804, e 0,0098 nats por símbolo no teste.
- **De 3 para 4 estados** o BIC ainda cai bastante (−5.201), mas o ganho é de apenas 0,0029 nats por símbolo no treino e 0,0036 no teste. Com quase 1 milhão de símbolos, o BIC aceita quase qualquer melhoria. O ganho de 4 estados é estatisticamente sólido, mas modesto na prática.
- **O modelo de 4 estados repete a escada.** Os estados ordenados por P(0) são: muito calmo (P(0) 0,84, 8% do tempo), calmo (0,58, 30%), intermediário (0,36, 41%) e agitado (0,18, 21%). A matriz de transição é **tridiagonal**: só há trocas entre vizinhos, e as transições que pulariam um degrau são zero ou ≤ 10⁻⁵. O modelo de 3 estados não é um acidente: a volatilidade se move de degrau em degrau.
- **O resultado de 5 estados não é válido como comparação.** Um HMM de 5 estados contém o de 4 como caso particular, então no ótimo global a sua log-verossimilhança nunca seria menor. Ela ficou menor (-927.708 contra -926.253), e nenhum dos 6 reinícios convergiu. A otimização falhou, e não dá para dizer se 5 estados seriam melhores. Ainda assim, a solução encontrada é reveladora (seção 8): ela criou subestados **direcionais** de curtíssima duração.
- **Para o TCC:** 3 estados é a escolha mais interpretável e já captura o grosso da estrutura. 4 estados é a escolha pelo BIC. Os dois contam a mesma história.

---

## 5. Estrutura do modelo de 3 estados

Os estados abaixo estão rotulados pelo papel que exercem; no JSON, calmo = 2, intermediário = 0 e agitado = 1.

| Regime | P(−1) | P(0) | P(+1) | Permanência esperada | Fração do tempo (estacionária) |
|---|---:|---:|---:|---:|---:|
| Calmo | 0,114 | 0,767 | 0,119 | 6,1 h | 16,0% |
| Intermediário | 0,267 | 0,462 | 0,271 | 8,0 h | 48,5% |
| Agitado | 0,395 | 0,215 | 0,390 | 10,4 h | 35,5% |

Transições:

```
calmo  <-- 0,0136 ---  intermediário  --- 0,0059 -->  agitado
       --- 0,0045 -->                 <-- 0,0080 ---
            (calmo <-> agitado: 4·10⁻⁸ e 5·10⁻¹⁶, ou seja, zero)
```

- Em todos os regimes, |P(+1) − P(−1)| < 0,005: **nenhum estado tem direção.**
- A distribuição de símbolos implicada pelo modelo (28,78% / 42,33% / 28,89%) reproduz as frequências do treino até a 4ª casa decimal.
- Os autovalores de `A` (1; 0,9896; 0,9784) dão tempos de relaxação de cerca de 8,0 h e 3,9 h. Depois de ~1 dia, o modelo já "esqueceu" o regime atual.

---

## 6. O que os regimes significam nos dados

Para esta seção, a série inteira (treino e teste) foi decodificada com o modelo de 3 estados, por Viterbi e por posteriores (forward-backward).

### 6.1 Magnitude do retorno

| Regime | Ocupação (Viterbi) | \|r\| médio | \|r\| mediano | \|r\| / σ do bloco |
|---|---:|---:|---:|---:|
| Calmo | 14,1% | 0,00056 | 0,00035 | 0,24 |
| Intermediário | 52,3% | 0,00109 | 0,00070 | 0,50 |
| Agitado | 33,6% | 0,00195 | 0,00122 | 1,09 |

O retorno absoluto cresce de forma monotônica ao longo da escada. O regime agitado tem 3,5 vezes o retorno típico do calmo, e a razão em relação ao σ do bloco vai de 0,24 a 1,09. A leitura como regimes de volatilidade é direta.

A posterior média por estado (16,1% / 48,5% / 35,4%) coincide com a distribuição estacionária. Em 66% dos instantes, o modelo tem mais de 90% de certeza sobre o regime.

### 6.2 Hora do dia (UTC)

Fração da posterior em cada regime, por hora (horas selecionadas):

| Hora UTC | Calmo | Intermediário | Agitado | % de símbolos 0 |
|---:|---:|---:|---:|---:|
| 00 | 0,134 | 0,510 | 0,356 | 37,7% |
| 04 | 0,230 | 0,504 | 0,266 | 48,3% |
| **05** | **0,235** | 0,507 | 0,258 | **49,0%** |
| 08 | 0,195 | 0,527 | 0,278 | 44,6% |
| 12 | 0,131 | 0,490 | 0,379 | 40,4% |
| 14 | 0,103 | 0,421 | 0,476 | 34,6% |
| **15** | 0,102 | 0,414 | **0,484** | 35,3% |
| 17 | 0,132 | 0,430 | 0,438 | 39,7% |
| 20 | 0,154 | 0,468 | 0,378 | 41,6% |

- O regime agitado tem pico entre **13h e 17h UTC**, a janela da abertura do mercado americano (9h30 em Nova York = 13h30 ou 14h30 UTC, conforme o horário de verão) e da divulgação de dados macroeconômicos dos EUA.
- O calmo tem pico entre **3h e 7h UTC**, a madrugada americana e o fim do pregão asiático.
- O modelo não recebeu o horário em nenhum momento. Ele recuperou essa sazonalidade apenas a partir da sequência de símbolos. Isso também explica por que as permanências estimadas ficam entre 6 e 10 horas: elas têm a escala de um "turno" do mercado.

### 6.3 Dia da semana

| | Seg–Sex (média) | Sábado | Domingo |
|---|---:|---:|---:|
| Calmo | 0,107 | **0,317** | 0,278 |
| Intermediário | 0,484 | 0,488 | 0,487 |
| Agitado | 0,409 | **0,195** | 0,234 |
| % de símbolos 0 | 38,9% | 52,3% | 49,7% |

No fim de semana, o regime agitado cai pela metade e o calmo triplica. É o mesmo efeito da hora do dia: o cripto opera 24/7, mas o fluxo institucional, não.

### 6.4 Ano

| Ano | Calmo | Intermediário | Agitado |
|---|---:|---:|---:|
| 2017 | 0,122 | 0,569 | 0,309 |
| 2018 | 0,192 | 0,567 | 0,242 |
| 2019 | 0,213 | 0,489 | 0,298 |
| 2020 | 0,179 | 0,393 | **0,428** |
| 2021 | **0,063** | 0,515 | 0,422 |
| 2022 | 0,195 | 0,505 | 0,300 |
| 2023 | 0,194 | 0,457 | 0,350 |
| 2024 | 0,116 | 0,456 | **0,428** |
| 2025 | 0,139 | 0,475 | 0,386 |
| 2026 | 0,182 | 0,478 | 0,340 |

Como o limiar é recalculado a cada mês com o σ do mês anterior, a escala absoluta da volatilidade já foi removida. O que sobra é a variação *dentro* do mês e a mudança de um mês para o outro. Os anos com mais tempo no regime agitado (2020, 2021 e 2024) foram anos de alta forte e de volatilidade crescente. Nesses anos, o σ do mês anterior tende a subestimar o mês corrente. É uma hipótese plausível, não testada aqui. O teste (2026) fica com 18,2% / 48,1% / 33,7%, perto da distribuição estacionária, sem sinal de que o período de teste seja atípico para o modelo.

### 6.5 Episódios (Viterbi)

| Regime | Episódios | Duração média | Mediana | P90 | Máximo |
|---|---:|---:|---:|---:|---:|
| Calmo | 646 | 17,4 h | 9,7 h | 40,7 h | 295 h |
| Intermediário | 1.460 | 28,4 h | 16,4 h | 64,9 h | 259 h |
| Agitado | 814 | 32,8 h | 12,3 h | 75,3 h | 733 h (cerca de 30 dias) |

- **Nenhuma troca direta entre calmo e agitado em todo o caminho de Viterbi.** As 2.919 trocas passam todas pelo intermediário.
- As durações do Viterbi são 2 a 3 vezes as permanências esperadas da matriz `A`. Isso é esperado: o Viterbi escolhe o caminho mais provável como um todo e "alisa" excursões curtas de regime. Para estimar durações, as posteriores são mais fiéis que o Viterbi.

---

## 7. Desempenho preditivo no teste (um passo à frente)

A previsão de cada símbolo usa apenas o passado: filtro forward iniciado na distribuição estacionária, com `P(y_{t+1} | y_1..y_t) = α_t · A · B`.

| Modelo | Log-veross. | Acurácia | Acerto de direção\* | AUC movimento\*\* | Brier\*\*\* |
|---|---:|---:|---:|---:|---:|
| Sempre 0 | — | 43,83% | — | — | — |
| Persistência (repete o último) | — | 40,61% | — | — | — |
| Markov ordem 1 | -1,0600 | 43,83% | 51,21% | 0,587 | 0,639 |
| Markov ordem 2 | -1,0502 | 45,23% | 51,53% | 0,624 | 0,632 |
| Markov ordem 4 | -1,0382 | 45,90% | 51,50% | 0,651 | 0,624 |
| HMM 2 estados | -1,0252 | 47,19% | 50,62% | 0,690 | 0,614 |
| **HMM 3 estados** | **-1,0154** | **47,62%** | 50,12% | **0,696** | **0,608** |
| HMM 4 estados | -1,0118 | 47,68% | 49,85% | 0,697 | 0,606 |
| HMM 5 estados (não convergido) | -1,0153 | 48,10% | **51,63%** | 0,696 | 0,608 |

\* Entre os candles de teste que de fato se moveram (26.901), a fração em que o modelo atribuiu mais probabilidade ao sinal correto. O erro padrão é de cerca de 0,3 p.p., então 51,2% está cerca de 4 erros padrão acima de 50%.
\*\* Capacidade de separar "o próximo candle sai da faixa ±s/3" de "fica em 0". 0,5 é o acaso.
\*\*\* Erro quadrático da distribuição prevista. Um modelo uniforme tem 0,667.

### 7.1 O HMM prevê bem a volatilidade

- **Separação.** O HMM de 3 estados separa candles com e sem movimento com AUC 0,696, bem acima da Markov de ordem 4 (0,651) e da de ordem 1 (0,587).
- **Calibração.** As probabilidades são bem calibradas. Agrupando a P(0) prevista em decis:

| Decil | P(0) prevista | Frequência observada de 0 |
|---:|---:|---:|
| 1 | 0,224 | 0,194 |
| 2 | 0,239 | 0,256 |
| 3 | 0,281 | 0,315 |
| 4 | 0,355 | 0,329 |
| 5 | 0,419 | 0,402 |
| 6 | 0,450 | 0,451 |
| 7 | 0,469 | 0,493 |
| 8 | 0,508 | 0,539 |
| 9 | 0,637 | 0,624 |
| 10 | 0,746 | 0,778 |

  A P(0) prevista varia de 0,23 (p10) a 0,72 (p90). Quando o modelo diz "75% de chance de o candle ficar parado", isso acontece em 78% das vezes.

- **Horizonte.** A previsão se degrada devagar com a distância:

| Horizonte | AUC movimento | Log-veross. |
|---|---:|---:|
| 5 min | 0,695 | -1,0155 |
| 15 min | 0,690 | -1,0187 |
| 1 h | 0,679 | -1,0259 |
| 4 h | 0,644 | -1,0448 |
| 1 dia | 0,615 | -1,0726 |

  A 1 hora de distância, o HMM ainda prevê melhor (-1,0259) que a Markov de ordem 4 a 5 minutos (-1,0382).

### 7.2 O HMM não prevê direção

- **Direção.** O acerto é de 50,1% com 3 estados e 49,9% com 4, o que é acaso.
- **Reversão.** As cadeias de Markov acertam 51,2% a 51,5% porque captam uma **reversão de curto prazo**. No teste, depois de um −1, o +1 vem em 32,6% das vezes e o −1 em 31,0%. Depois de um +1, o −1 vem em 34,1% e o +1 em 29,9%. O HMM de até 4 estados não tem como representar isso, porque todos os seus estados são simétricos.
- **Acurácia.** A acurácia de 47,6% do HMM (3,8 p.p. acima de "sempre 0") vem apenas de saber *quando* prever 0. Quando a maior probabilidade não é o 0 (regime agitado), o modelo escolhe sempre −1 (32% das previsões) e nunca +1. A escolha é decidida por uma assimetria de 0,004 em `B`, ou seja, ruído de estimação: as previsões de −1 acertam 37,8%, e prever +1 nesses mesmos instantes acertaria 36,2%.
- **5 estados.** É o único HMM com direção acima do acaso (51,6%), justamente porque seus subestados rápidos são direcionais (seção 8).
- **Sem leitura financeira.** Um acerto de direção de 51% em candles de 5 minutos é estatisticamente detectável, mas não indica, por si só, nada explorável, pois custos e spread não foram considerados.

---

## 8. Superfície de otimização: duas famílias de soluções

O relatório original mostrava 7 de 10 reinícios presos em cerca de -936.820 depois de 1000 iterações. Para saber se era lentidão ou outro ótimo, o **reinício 2** (mesma semente, `SeedSequence(0).spawn(10)[2]`) foi continuado com `max_iter = 6000`:

| Iteração | 0 | 100 | 500 | 1000 | 1500 | 2000 | 2491 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Log-veross. | -1.014.204 | -937.318 | -936.861 | -936.833 | -936.820 | -936.812 | **-936.807 (convergiu)** |

O reinício **convergiu para um ótimo local diferente**, 7.892 nats abaixo do melhor. Mais iterações não o levariam à solução boa. A estrutura dessa solução é outra:

| Estado | P(−1) | P(0) | P(+1) | Permanência | Leitura |
|---|---:|---:|---:|---:|---|
| a | 0,178 | 0,640 | 0,181 | 86 passos (7,2 h) | calmo persistente |
| b | **0,600** | 0,278 | 0,122 | 2,3 passos | "queda" |
| c | 0,001 | 0,277 | **0,723** | 1,5 passos | "alta" |

Os estados **b** e **c** alternam rapidamente (b → c com 0,43; c → b com 0,64): o modelo gastou dois de seus três estados para representar a reversão de sinal do curto prazo. Portanto, há duas explicações concorrentes para os dados:

| Família | O que captura | Log-veross. (3 estados) |
|---|---|---:|
| Escada de volatilidade | aglomeração de volatilidade (horas) | **-928.915** |
| Direcional | 1 regime calmo + reversão de sinal (minutos) | -936.807 |
| (referência: HMM de 2 estados) | só calmo × agitado | -937.365 |

- A explicação de volatilidade vale cerca de 15 vezes mais que a direcional (8.450 contra 558 nats de ganho sobre o modelo de 2 estados).
- Com 3 estados, não cabem as duas. Por isso o EM escolhe uma ou outra, dependendo de onde começa.
- Com 5 estados, a solução encontrada tentou ficar com as duas: 1 calmo e 1 agitado persistentes, mais 3 subestados rápidos, dois deles quase puramente direcionais (P(−1) = 0,53 com P(+1) ≈ 0, e P(+1) = 0,55 com P(−1) ≈ 0). Isso explica o melhor acerto de direção do modelo de 5 estados e sugere que, bem otimizado, ele pode superar o de 4.

**Consequência prática.** A taxa de sucesso dos reinícios é baixa e cai com o número de estados: 4 de 6 com 2 estados, 3 de 10 com 3 estados, 1 de 6 com 4 estados. Com 3 estados e 10 reinícios, a chance de nenhum chegar à escada é de cerca de 0,7¹⁰ ≈ 3%. Com 4 estados e 6 reinícios, o modelo obtido depende muito da sorte. Os reinícios com 4 estados que não chegaram ao melhor ficaram em -928,6 mil, a qualidade de um modelo de 3 estados, ou em -936,3 mil, a família direcional.

---

## 9. Adequação do modelo

Uma sequência do tamanho do treino foi simulada pelo HMM de 3 estados (`sample`, semente 1) e comparada com os dados reais por autocorrelação.

| Defasagem | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Indicador de movimento (≠ 0), real | 0,157 | 0,154 | 0,148 | 0,140 | 0,115 | **0,101** |
| Indicador de movimento, HMM | 0,144 | 0,144 | 0,136 | 0,130 | 0,088 | **0,008** |
| Símbolo (−1/0/+1), real | **−0,047** | −0,031 | −0,007 | 0,000 | 0,002 | −0,002 |
| Símbolo, HMM | 0,001 | −0,003 | −0,003 | −0,002 | 0,000 | 0,002 |

- **Até cerca de 1 hora, o HMM reproduz bem a aglomeração de volatilidade.**
- **A memória real é muito mais longa.** A autocorrelação do movimento ainda é 0,10 a 1 dia, enquanto a do HMM é praticamente zero. Parte disso é a sazonalidade diária (seção 6.2), que se repete a cada 288 passos e que um HMM homogêneo não consegue gerar. Parte é a memória longa típica da volatilidade.
- **O HMM não gera a autocorrelação negativa do sinal** (−0,047 na defasagem 1), que é a reversão da seção 7.

Detalhes que **não** afetam as conclusões:

- **Distribuição inicial `pi`.** Ela é degenerada (100% no regime calmo) por ser estimada a partir de um único instante. Recalcular o teste com `pi` igual à distribuição estacionária muda o resultado em 2,5·10⁻⁵ nats por símbolo, cerca de 1,2 nats no total.
- **Lacunas (pendência D6).** O treino concatena 35 segmentos. Avaliar cada segmento separadamente, reiniciando o filtro, muda a log-verossimilhança de treino em −126 nats (1,4·10⁻⁴ por símbolo), e a maior parte disso é só o custo de reiniciar o filtro 35 vezes. O efeito das 34 junções é desprezível para o HMM.

---

## 10. Conclusões para o TCC

1. **O HMM é um modelo de volatilidade.** O texto deve apresentá-lo assim: ele identifica regimes de agitação do mercado com estrutura interpretável (escada, sazonalidade intradiária e semanal) e prevê bem *se* haverá movimento. Não deve ser apresentado como um modelo de direção do preço.
2. **Como métrica principal, use a log-verossimilhança no teste, a calibração ou a AUC de movimento.** A acurácia por argmax esconde o que o modelo faz bem e é dominada pela classe 0.
3. **A comparação com os modelos ingênuos é favorável e robusta.** O HMM de 3 estados supera cadeias de Markov de até ordem 6 com uma fração dos parâmetros, tanto no treino quanto no teste.
4. **Se a direção for um objetivo, nenhum HMM de 3 ou 4 estados serve.** As alternativas são uma Markov de ordem baixa para o sinal, um HMM com mais estados e bem otimizado, ou um modelo que combine os dois.

## 11. Recomendações (sem alterar o código agora)

1. **Mais reinícios, e registrar quantos chegaram à melhor solução.** Para 3 estados, pelo menos 20; para 4 ou mais, de 30 a 50, ou uma inicialização a partir da solução de n−1 estados com um estado dividido.
2. **Subir `max_iter` para 5 estados ou mais.** Nenhum reinício convergiu em 1000 iterações, e o reinício estudado precisou de 2.491 com 3 estados.
3. **Ordenar os estados por P(0) ao salvar**, para que os rótulos sejam estáveis entre execuções.
4. **Incluir no relatório de treino os baselines** (iid e Markov de ordem 1 a 4), a AUC de movimento e o acerto de direção, com o teste iniciado na distribuição estacionária.
5. **Fixar `--blocks`** para que os números do texto sejam reproduzíveis em outra máquina.
6. **Rodar a mesma bateria em `global` e em s/4 e s/5.** Com s/4, os três símbolos ficam equilibrados (cerca de 33% cada) e a interpretação dos regimes pode mudar.
7. **Investigar a sazonalidade.** Um HMM com transições que dependem da hora, ou uma comparação simples com um modelo "hora do dia", mostraria quanto do poder preditivo do HMM é sazonalidade e quanto é dinâmica própria.

---

## Apêndice: reprodução

Treinos (o primeiro é o relatório original; os demais foram gravados fora do repositório):

```
python -m src.training.hmm_training --snapshot snapshot/btc_usdt_2026-10-01.parquet
python -m src.training.hmm_training --snapshot snapshot/btc_usdt_2026-10-01.parquet --states 2 4 5 --restarts 6 --output <pasta>
```

Os cálculos das seções 3 e 6 a 9 usaram os parâmetros salvos nos JSONs, mais as funções existentes: `load_snapshot`, `discretize`, `split_sequences`, `HMMDiscreto.predict`, `predict_proba`, `score`, `sample` e `_forward`.

- **Markov:** contagem com Laplace +1.
- **AUC:** pelo método de postos (Mann-Whitney).
- **Reinício continuado:** `HMMDiscreto(3, 3, max_iter=6000)._baum_welch` a partir de `_inicializar(default_rng(SeedSequence(0).spawn(10)[2]))`.

Ambiente: Python 3.13, numpy 2.5, numba 0.68, 12 threads.
