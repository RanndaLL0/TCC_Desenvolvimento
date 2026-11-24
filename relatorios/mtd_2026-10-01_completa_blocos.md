# Relatório completo do MTD (tabelas)

Gerado em 2026-11-23 02:50 (UTC) por `src/experiments/mtd_analysis.py`. Mesmas métricas de `relatorios/analise_hmm_2026-10-01_completa.md`. A leitura dos números está em `relatorios/analise_mtd_2026-10-01_completa.md`.

## Fonte e parâmetros

- Snapshot: `btc_usdt_2026-10-01.parquet` (SHA-256 `4c71d14999175cf62f037af5880db2554c03617228db82989fe76fdeb175bd4d`), 957.853 candles
- MTD: ordens 1 a 10, **10 reinícios por ordem**, EM com tolerância relativa 1e-8, `max_iter` 1000, piso 1e-10, semente 0 (os mesmos valores do treino do HMM)
- Treino e teste: corte de 95% das linhas de candle. **Todos os modelos são avaliados nas mesmas transições**: treino = as últimas `N − ordem_máxima`; teste = todas, com o contexto vindo do fim do treino
- O HMM (quando aparece) usa os parâmetros do JSON de `hmm_training.py` e é **filtrado ao longo de treino e teste**, ou seja, começa o teste com o estado do fim do treino
- Unidade: log-verossimilhança por símbolo, em nats. Uniforme = −ln 3 = −1,0986

## Resumo entre discretizações

**Atenção:** BIC e LL só são comparáveis **dentro** da mesma discretização, porque cada uma tem outra distribuição de símbolos (e, no modo `global`, o teste tem muito mais zeros que o treino). Entre linhas, vale comparar o ganho sobre o iid, o AUC e a direção. Cada linha mostra o MTD da ordem de menor BIC; "Melhor constante" é o melhor acerto de direção de um modelo que prevê sempre o mesmo sinal, escolhido olhando o teste.

| Modo | Bloco (meses) | s/k | Zeros treino | Zeros teste | Ordem (menor BIC) | LL/sím. teste | Ganho s/ iid | AUC movimento | Direção teste (IC 95%) | Melhor constante |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| blocks | 1 | 3 | 42,33% | 43,83% | 10 | -1,02360 | 0,05175 | 0,679 | 51,17% (50,58% a 51,77%) | 50,86% |
| blocks | 2 | 3 | 44,03% | 48,35% | 10 | -0,98745 | 0,06692 | 0,699 | 51,37% (50,75% a 52,00%) | 50,42% |
| blocks | 3 | 3 | 44,83% | 50,65% | 10 | -0,98895 | 0,05298 | 0,672 | 51,24% (50,60% a 51,87%) | 50,14% |
| blocks | 6 | 3 | 47,58% | 47,98% | 10 | -1,00661 | 0,04635 | 0,669 | 51,67% (51,05% a 52,29%) | 50,13% |
| blocks | 12 | 3 | 49,90% | 50,68% | 10 | -0,98966 | 0,04539 | 0,670 | 51,57% (50,94% a 52,21%) | 50,71% |

## Discretização `blocks`, s/3, blocos de 1 mês(es)

- Símbolos: treino 905.654, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 35 segmentos no treino
- Classes treino (−1/0/+1): 28,78% / 42,33% / 28,89%
- Classes teste (−1/0/+1): 28,57% / 43,83% / 27,61%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,02327 |
| Frequências do treino (iid) | 2 | -1,08106 | -1,07534 | 1.958.146,0 | 0,02327 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,81962 | -2,74315 | — | -1,64454 | -1,66781 |
| Cadeia de 1ª ordem | 6 | -1,06778 | -1,05997 | 1.934.137,9 | 0,03864 | 0,01538 |
| MTD ordem 1 | 6 | -1,06778 | -1,05997 | 1.934.137,9 | 0,03864 | 0,01538 |
| MTD ordem 2 | 7 | -1,05862 | -1,05058 | 1.917.558,6 | 0,04803 | 0,02476 |
| MTD ordem 3 | 8 | -1,05206 | -1,04405 | 1.905.688,5 | 0,05456 | 0,03130 |
| MTD ordem 4 | 9 | -1,04739 | -1,03904 | 1.897.243,1 | 0,05958 | 0,03631 |
| MTD ordem 5 | 10 | -1,04355 | -1,03544 | 1.890.298,8 | 0,06317 | 0,03990 |
| MTD ordem 6 | 11 | -1,04018 | -1,03209 | 1.884.217,2 | 0,06652 | 0,04325 |
| MTD ordem 7 | 12 | -1,03799 | -1,02919 | 1.880.267,4 | 0,06942 | 0,04615 |
| MTD ordem 8 | 13 | -1,03604 | -1,02742 | 1.876.738,6 | 0,07119 | 0,04792 |
| MTD ordem 9 | 14 | -1,03419 | -1,02541 | 1.873.404,9 | 0,07320 | 0,04993 |
| MTD ordem 10 | 15 | -1,03279 | -1,02360 | 1.870.878,8 | 0,07501 | 0,05175 |
| HMM 3 estados | 14 | -1,02569 | -1,01545 | 1.858.005,0 **←** | 0,08317 | 0,05990 |

HMM no relatório original: LL de treino -928.914,9 (soma de T termos), BIC 1.858.021,8, teste -1,01544 por símbolo (começando na distribuição estacionária). A linha acima recalcula nas mesmas transições do MTD.

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 43,83% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,74315 | 40,61% | 48,79% (48,19% a 49,38%) | 0,587 | 1,153 | 43,82% |
| Cadeia de 1ª ordem | -1,05997 | 43,83% | 51,21% (50,62% a 51,81%) | 0,587 | 0,639 | 100,00% |
| MTD ordem 1 | -1,05997 | 43,83% | 51,21% (50,62% a 51,81%) | 0,587 | 0,639 | 100,00% |
| MTD ordem 2 | -1,05058 | 45,23% | 51,53% (50,93% a 52,13%) | 0,622 | 0,632 | 64,16% |
| MTD ordem 3 | -1,04405 | 45,66% | 51,41% (50,81% a 52,01%) | 0,640 | 0,628 | 75,53% |
| MTD ordem 4 | -1,03904 | 45,77% | 51,37% (50,77% a 51,97%) | 0,651 | 0,624 | 75,10% |
| MTD ordem 5 | -1,03544 | 46,18% | 51,51% (50,91% a 52,10%) | 0,658 | 0,622 | 64,60% |
| MTD ordem 6 | -1,03209 | 46,51% | 51,29% (50,69% a 51,89%) | 0,665 | 0,619 | 71,10% |
| MTD ordem 7 | -1,02919 | 46,79% | 51,39% (50,79% a 51,99%) | 0,669 | 0,617 | 70,04% |
| MTD ordem 8 | -1,02742 | 46,85% | 51,32% (50,72% a 51,91%) | 0,673 | 0,616 | 67,13% |
| MTD ordem 9 | -1,02541 | 47,04% | 51,22% (50,62% a 51,81%) | 0,676 | 0,615 | 68,93% |
| MTD ordem 10 | -1,02360 | 47,11% | 51,17% (50,58% a 51,77%) | 0,679 | 0,613 | 68,62% |
| HMM 3 estados | -1,01545 | 47,61% | 50,13% (49,53% a 50,73%) | 0,695 | 0,608 | 67,67% |

Direção: entre os candles de teste que se moveram (26.903), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 13.683 | 31,04% | 36,37% | 32,60% |
| +1 | 13.221 | 34,10% | 36,02% | 29,88% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 43,83% dos candles de teste ficaram parados e 56,17% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 43,83% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 100,00% | 43,83% | 100,00% | 0,00% | — | 0,00% |
| MTD ordem 6 | 71,10% | 50,33% | 81,64% | 28,90% | 72,16% | 37,13% |
| MTD ordem 10 | 68,62% | 51,48% | 80,61% | 31,38% | 72,92% | 40,74% |
| HMM 3 estados | 67,67% | 52,32% | 80,78% | 32,33% | 73,94% | 42,56% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | — / 0,00% / 0,000 | 43,83% / 100,00% / 0,609 | — / 0,00% / 0,000 | 0,203 |
| MTD ordem 6 | 37,47% / 16,96% / 0,234 | 50,33% / 81,64% / 0,623 | 36,86% / 21,32% / 0,270 | 0,375 |
| MTD ordem 10 | 38,35% / 18,06% / 0,246 | 51,48% / 80,61% / 0,628 | 36,95% / 24,00% / 0,291 | 0,388 |
| HMM 3 estados | 37,77% / 42,75% / 0,401 | 52,32% / 80,78% / 0,635 | — / 0,00% / 0,000 | 0,345 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 2.471 | 8.090 | 3.121 | 13.682 |
| 0 | 1.777 | 16.919 | 2.294 | 20.990 |
| +1 | 2.195 | 7.853 | 3.173 | 13.221 |

Matriz de confusão, HMM 3 estados (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 5.849 | 7.833 | 0 | 13.682 |
| 0 | 4.035 | 16.955 | 0 | 20.990 |
| +1 | 5.600 | 7.621 | 0 | 13.221 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. | HMM 3 estados prev. | HMM 3 estados obs. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,213 | 0,234 | 0,199 | 0,224 | 0,224 | 0,194 |
| 2 | 0,287 | 0,298 | 0,268 | 0,266 | 0,239 | 0,256 |
| 3 | 0,314 | 0,309 | 0,315 | 0,313 | 0,280 | 0,315 |
| 4 | 0,372 | 0,372 | 0,358 | 0,356 | 0,354 | 0,331 |
| 5 | 0,390 | 0,388 | 0,402 | 0,402 | 0,419 | 0,400 |
| 6 | 0,454 | 0,447 | 0,444 | 0,442 | 0,450 | 0,452 |
| 7 | 0,474 | 0,482 | 0,489 | 0,480 | 0,469 | 0,493 |
| 8 | 0,543 | 0,528 | 0,542 | 0,528 | 0,508 | 0,538 |
| 9 | 0,594 | 0,592 | 0,608 | 0,613 | 0,636 | 0,623 |
| 10 | 0,681 | 0,733 | 0,716 | 0,758 | 0,746 | 0,779 |
| P(0) p10 a p90 | 0,28 a 0,64 |  | 0,23 a 0,65 |  | 0,23 a 0,72 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | HMM 3 estados AUC | HMM 3 estados LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0753 | 0,695 | -1,0154 | 0,587 | -1,0600 | 0,665 | -1,0321 | 0,679 | -1,0236 |
| 15 min | 0,500 | -1,0753 | 0,690 | -1,0187 | 0,582 | -1,0747 | 0,652 | -1,0415 | 0,669 | -1,0303 |
| 1 h | 0,500 | -1,0753 | 0,679 | -1,0259 | 0,577 | -1,0753 | 0,638 | -1,0646 | 0,655 | -1,0465 |
| 4 h | 0,500 | -1,0753 | 0,644 | -1,0448 | 0,500 | -1,0753 | 0,606 | -1,0752 | 0,618 | -1,0738 |
| 1 dia | 0,500 | -1,0753 | 0,614 | -1,0724 | 0,500 | -1,0753 | 0,520 | -1,0752 | 0,596 | -1,0751 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 28,78% / 42,33% / 28,89% | 0,157 | 0,5 | 3 |
| 2 | 0,515 | 0,485 | 0,485 | 28,76% / 42,37% / 28,87% | 0,268 | 1,2 | 6 |
| 3 | 0,352 | 0,331 | 0,317 | 28,73% / 42,44% / 28,83% | 0,352 | 2,0 | 10 |
| 4 | 0,273 | 0,257 | 0,226 | 28,70% / 42,50% / 28,80% | 0,414 | 3,0 | 15 |
| 5 | 0,226 | 0,210 | 0,182 | 28,67% / 42,56% / 28,78% | 0,465 | 4,1 | 20 |
| 6 | 0,193 | 0,179 | 0,156 | 28,63% / 42,62% / 28,74% | 0,509 | 5,4 | 27 |
| 7 | 0,172 | 0,159 | 0,118 | 28,61% / 42,68% / 28,72% | 0,543 | 6,7 | 33 |
| 8 | 0,157 | 0,142 | 0,106 | 28,58% / 42,72% / 28,69% | 0,571 | 8,1 | 41 |
| 9 | 0,144 | 0,130 | 0,098 | 28,56% / 42,77% / 28,67% | 0,598 | 9,7 | 49 |
| 10 | 0,134 | 0,121 | 0,083 | 28,54% / 42,81% / 28,65% | 0,620 | 11,4 | 57 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3453 | 0,1372 | 0,5175 |
| 0 | 0,1074 | 0,7823 | 0,1102 |
| +1 | 0,4916 | 0,1885 | 0,3199 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,134, 0,121, 0,111, 0,099, 0,097, 0,100, 0,083, 0,085, 0,089, 0,083

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 8 | 88 | 0 | 18s |
| 3 | 10 | 10 | 10 | 0,01 | 1 | 207 | 0 | 26s |
| 4 | 10 | 10 | 10 | 0,01 | 3 | 142 | 0 | 27s |
| 5 | 10 | 10 | 10 | 0,01 | 3 | 138 | 0 | 33s |
| 6 | 10 | 10 | 10 | 0,02 | 3 | 131 | 0 | 35s |
| 7 | 10 | 10 | 10 | 0,03 | 3 | 138 | 0 | 44s |
| 8 | 10 | 10 | 10 | 0,03 | 1 | 174 | 0 | 38s |
| 9 | 10 | 10 | 10 | 0,03 | 1 | 173 | 0 | 53s |
| 10 | 10 | 10 | 10 | 0,04 | 1 | 172 | 0 | 55s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 | HMM 3 estados |
|---:|---:|---:|---:|---:|
| 00 | 37,69% | 40,79% | 40,95% | 41,09% |
| 04 | 48,34% | 45,37% | 45,93% | 45,09% |
| 05 | 49,00% | 45,99% | 46,76% | 46,37% |
| 08 | 44,58% | 43,87% | 44,42% | 45,41% |
| 12 | 40,35% | 41,98% | 42,33% | 43,33% |
| 14 | 34,55% | 38,69% | 38,11% | 39,13% |
| 15 | 35,35% | 38,59% | 37,72% | 37,61% |
| 17 | 39,69% | 40,65% | 39,97% | 37,99% |
| 20 | 41,56% | 42,09% | 41,98% | 41,08% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 7,41% | 14,45% | 1,000 |
| MTD ordem 10 | 0,972 | 9,03% | 14,45% | 1,000 |
| HMM 3 estados | 0,865 | 9,53% | 14,45% | 0,998 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 | HMM 3 estados |
|---|---:|---:|---:|---:|
| Seg | 39,34% | 40,94% | 40,64% | 39,91% |
| Ter | 38,84% | 40,70% | 40,35% | 39,48% |
| Qua | 38,92% | 40,74% | 40,40% | 39,57% |
| Qui | 38,60% | 40,56% | 40,19% | 39,27% |
| Sex | 39,13% | 40,82% | 40,47% | 39,54% |
| Sáb | 52,27% | 47,50% | 48,60% | 49,55% |
| Dom | 49,72% | 46,31% | 47,23% | 48,42% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,157 | 0,154 | 0,148 | 0,140 | 0,115 | 0,101 |
| Movimento, MTD ordem 1 | 0,156 | 0,023 | 0,001 | -0,001 | 0,001 | 0,001 |
| Movimento, MTD ordem 6 | 0,160 | 0,155 | 0,145 | 0,037 | 0,000 | 0,001 |
| Movimento, MTD ordem 10 | 0,164 | 0,158 | 0,149 | 0,084 | 0,002 | -0,001 |
| Símbolo, real | -0,047 | -0,031 | -0,007 | 0,000 | 0,002 | -0,002 |
| Símbolo, MTD ordem 1 | -0,047 | 0,002 | 0,000 | -0,002 | -0,001 | -0,000 |
| Símbolo, MTD ordem 6 | -0,024 | -0,023 | -0,019 | -0,002 | -0,001 | -0,000 |
| Símbolo, MTD ordem 10 | -0,020 | -0,020 | -0,015 | 0,001 | -0,000 | 0,000 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 0,00% | -40,8 | -1,06778 | -1,06777 |
| 6 | 204 | 0,02% | -221,9 | -1,04018 | -1,04017 |
| 10 | 340 | 0,04% | -363,6 | -1,03279 | -1,03277 |

## Discretização `blocks`, s/3, blocos de 2 mês(es)

- Símbolos: treino 897.098, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 34 segmentos no treino
- Classes treino (−1/0/+1): 27,93% / 44,03% / 28,04%
- Classes teste (−1/0/+1): 26,04% / 48,35% / 25,60%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,04425 |
| Frequências do treino (iid) | 2 | -1,07394 | -1,05436 | 1.926.865,2 | 0,04425 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,76502 | -2,61803 | — | -1,51942 | -1,56366 |
| Cadeia de 1ª ordem | 6 | -1,05826 | -1,03363 | 1.898.790,3 | 0,06498 | 0,02073 |
| MTD ordem 1 | 6 | -1,05826 | -1,03363 | 1.898.790,3 | 0,06498 | 0,02073 |
| MTD ordem 2 | 7 | -1,04739 | -1,02161 | 1.879.299,0 | 0,07700 | 0,03275 |
| MTD ordem 3 | 8 | -1,03987 | -1,01290 | 1.865.814,7 | 0,08572 | 0,04147 |
| MTD ordem 4 | 9 | -1,03442 | -1,00680 | 1.856.060,3 | 0,09182 | 0,04757 |
| MTD ordem 5 | 10 | -1,03016 | -1,00187 | 1.848.418,3 | 0,09674 | 0,05250 |
| MTD ordem 6 | 11 | -1,02628 | -0,99734 | 1.841.472,5 | 0,10128 | 0,05703 |
| MTD ordem 7 | 12 | -1,02382 | -0,99407 | 1.837.083,4 | 0,10454 | 0,06030 |
| MTD ordem 8 | 13 | -1,02162 | -0,99167 | 1.833.143,6 | 0,10695 | 0,06270 |
| MTD ordem 9 | 14 | -1,01963 | -0,98922 | 1.829.589,7 | 0,10939 | 0,06514 |
| MTD ordem 10 | 15 | -1,01807 | -0,98745 | 1.826.799,6 **←** | 0,11117 | 0,06692 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 48,35% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,61803 | 43,34% | 48,54% (47,92% a 49,17%) | 0,599 | 1,100 | 48,35% |
| Cadeia de 1ª ordem | -1,03363 | 48,35% | 51,56% (50,94% a 52,19%) | 0,598 | 0,620 | 100,00% |
| MTD ordem 1 | -1,03363 | 48,35% | 51,56% (50,94% a 52,19%) | 0,598 | 0,620 | 100,00% |
| MTD ordem 2 | -1,02161 | 48,47% | 51,32% (50,70% a 51,94%) | 0,636 | 0,611 | 68,39% |
| MTD ordem 3 | -1,01290 | 49,39% | 51,48% (50,85% a 52,10%) | 0,657 | 0,605 | 79,06% |
| MTD ordem 4 | -1,00680 | 49,65% | 51,19% (50,57% a 51,81%) | 0,668 | 0,601 | 81,90% |
| MTD ordem 5 | -1,00187 | 49,82% | 51,65% (51,02% a 52,27%) | 0,677 | 0,597 | 70,42% |
| MTD ordem 6 | -0,99734 | 50,32% | 51,38% (50,76% a 52,00%) | 0,684 | 0,594 | 75,88% |
| MTD ordem 7 | -0,99407 | 50,50% | 51,41% (50,79% a 52,04%) | 0,689 | 0,592 | 76,26% |
| MTD ordem 8 | -0,99167 | 50,72% | 51,74% (51,12% a 52,37%) | 0,693 | 0,590 | 72,84% |
| MTD ordem 9 | -0,98922 | 50,85% | 51,49% (50,86% a 52,11%) | 0,696 | 0,589 | 74,41% |
| MTD ordem 10 | -0,98745 | 50,75% | 51,37% (50,75% a 52,00%) | 0,699 | 0,588 | 74,45% |

Direção: entre os candles de teste que se moveram (24.735), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 12.473 | 29,35% | 39,04% | 31,61% |
| +1 | 12.263 | 32,62% | 38,56% | 28,83% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 48,35% dos candles de teste ficaram parados e 51,65% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 48,35% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 100,00% | 48,35% | 100,00% | 0,00% | — | 0,00% |
| MTD ordem 6 | 75,88% | 54,65% | 85,75% | 24,12% | 71,44% | 33,37% |
| MTD ordem 10 | 74,45% | 55,42% | 85,34% | 25,55% | 72,25% | 35,74% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | — / 0,00% / 0,000 | 48,35% / 100,00% / 0,652 | — / 0,00% / 0,000 | 0,217 |
| MTD ordem 6 | 36,89% / 15,63% / 0,220 | 54,65% / 85,75% / 0,668 | 36,56% / 18,69% / 0,247 | 0,378 |
| MTD ordem 10 | 38,00% / 16,18% / 0,227 | 55,42% / 85,34% / 0,672 | 36,49% / 20,61% / 0,263 | 0,387 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 2.018 | 7.986 | 2.468 | 12.472 |
| 0 | 1.465 | 19.762 | 1.931 | 23.158 |
| +1 | 1.827 | 7.908 | 2.528 | 12.263 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,222 | 0,250 | 0,204 | 0,228 |
| 2 | 0,293 | 0,313 | 0,283 | 0,297 |
| 3 | 0,348 | 0,351 | 0,340 | 0,348 |
| 4 | 0,386 | 0,403 | 0,389 | 0,384 |
| 5 | 0,440 | 0,440 | 0,436 | 0,442 |
| 6 | 0,477 | 0,480 | 0,485 | 0,485 |
| 7 | 0,533 | 0,526 | 0,537 | 0,546 |
| 8 | 0,574 | 0,589 | 0,595 | 0,596 |
| 9 | 0,653 | 0,688 | 0,665 | 0,699 |
| 10 | 0,728 | 0,793 | 0,765 | 0,811 |
| P(0) p10 a p90 | 0,28 a 0,66 |  | 0,25 a 0,70 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0544 | 0,598 | -1,0336 | 0,684 | -0,9973 | 0,699 | -0,9874 |
| 15 min | 0,500 | -1,0544 | 0,595 | -1,0534 | 0,673 | -1,0079 | 0,690 | -0,9946 |
| 1 h | 0,500 | -1,0544 | 0,590 | -1,0544 | 0,659 | -1,0381 | 0,676 | -1,0147 |
| 4 h | 0,500 | -1,0544 | 0,463 | -1,0544 | 0,623 | -1,0537 | 0,636 | -1,0512 |
| 1 dia | 0,500 | -1,0544 | 0,471 | -1,0544 | 0,498 | -1,0537 | 0,613 | -1,0534 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 27,93% / 44,03% / 28,04% | 0,172 | 0,6 | 3 |
| 2 | 0,507 | 0,493 | 0,493 | 27,90% / 44,09% / 28,01% | 0,291 | 1,3 | 6 |
| 3 | 0,346 | 0,336 | 0,318 | 27,86% / 44,18% / 27,97% | 0,379 | 2,2 | 11 |
| 4 | 0,269 | 0,258 | 0,230 | 27,82% / 44,25% / 27,93% | 0,443 | 3,2 | 16 |
| 5 | 0,223 | 0,211 | 0,182 | 27,78% / 44,33% / 27,89% | 0,494 | 4,4 | 22 |
| 6 | 0,190 | 0,179 | 0,159 | 27,75% / 44,40% / 27,85% | 0,539 | 5,9 | 29 |
| 7 | 0,169 | 0,159 | 0,120 | 27,72% / 44,46% / 27,82% | 0,572 | 7,3 | 36 |
| 8 | 0,153 | 0,143 | 0,108 | 27,69% / 44,51% / 27,80% | 0,600 | 8,9 | 44 |
| 9 | 0,140 | 0,131 | 0,098 | 27,67% / 44,56% / 27,77% | 0,626 | 10,7 | 53 |
| 10 | 0,130 | 0,121 | 0,084 | 27,65% / 44,59% / 27,76% | 0,647 | 12,5 | 62 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3509 | 0,1311 | 0,5179 |
| 0 | 0,0968 | 0,8043 | 0,0989 |
| +1 | 0,4910 | 0,1838 | 0,3252 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,130, 0,121, 0,110, 0,100, 0,096, 0,102, 0,083, 0,086, 0,088, 0,084

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 3 | 139 | 0 | 18s |
| 3 | 10 | 10 | 10 | 0,01 | 3 | 143 | 0 | 27s |
| 4 | 10 | 10 | 10 | 0,01 | 6 | 122 | 0 | 27s |
| 5 | 10 | 10 | 10 | 0,02 | 1 | 166 | 0 | 33s |
| 6 | 10 | 10 | 10 | 0,01 | 7 | 126 | 0 | 35s |
| 7 | 10 | 10 | 10 | 0,02 | 1 | 163 | 0 | 43s |
| 8 | 10 | 10 | 10 | 0,03 | 1 | 162 | 0 | 42s |
| 9 | 10 | 10 | 10 | 0,03 | 1 | 162 | 0 | 55s |
| 10 | 10 | 10 | 10 | 0,03 | 1 | 162 | 0 | 55s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 39,55% | 42,46% | 42,65% |
| 04 | 50,34% | 47,47% | 48,02% |
| 05 | 51,08% | 48,10% | 48,85% |
| 08 | 46,33% | 45,74% | 46,31% |
| 12 | 42,07% | 43,70% | 44,05% |
| 14 | 36,39% | 40,25% | 39,69% |
| 15 | 36,98% | 40,08% | 39,24% |
| 17 | 41,51% | 42,32% | 41,64% |
| 20 | 43,56% | 43,95% | 43,83% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 8,02% | 14,69% | 1,000 |
| MTD ordem 10 | 0,973 | 9,61% | 14,69% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 41,03% | 42,57% | 42,27% |
| Ter | 40,70% | 42,41% | 42,08% |
| Qua | 40,70% | 42,40% | 42,07% |
| Qui | 40,44% | 42,27% | 41,91% |
| Sex | 41,05% | 42,56% | 42,23% |
| Sáb | 54,21% | 49,65% | 50,74% |
| Dom | 51,61% | 48,37% | 49,29% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,172 | 0,170 | 0,165 | 0,155 | 0,129 | 0,116 |
| Movimento, MTD ordem 1 | 0,171 | 0,027 | 0,001 | -0,001 | 0,001 | 0,001 |
| Movimento, MTD ordem 6 | 0,173 | 0,170 | 0,162 | 0,046 | -0,001 | 0,001 |
| Movimento, MTD ordem 10 | 0,177 | 0,174 | 0,166 | 0,098 | 0,004 | 0,000 |
| Símbolo, real | -0,045 | -0,031 | -0,007 | 0,000 | 0,001 | -0,002 |
| Símbolo, MTD ordem 1 | -0,046 | 0,002 | -0,000 | -0,002 | -0,002 | -0,000 |
| Símbolo, MTD ordem 6 | -0,025 | -0,024 | -0,020 | -0,001 | 0,000 | 0,001 |
| Símbolo, MTD ordem 10 | -0,021 | -0,019 | -0,015 | 0,001 | -0,001 | 0,001 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 33 | 0,00% | -39,6 | -1,05826 | -1,05826 |
| 6 | 198 | 0,02% | -219,0 | -1,02628 | -1,02626 |
| 10 | 330 | 0,04% | -358,9 | -1,01807 | -1,01804 |

## Discretização `blocks`, s/3, blocos de 3 mês(es)

- Símbolos: treino 888.170, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 34 segmentos no treino
- Classes treino (−1/0/+1): 27,53% / 44,83% / 27,65%
- Classes teste (−1/0/+1): 24,74% / 50,65% / 24,60%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,05668 |
| Frequências do treino (iid) | 2 | -1,07022 | -1,04193 | 1.901.077,0 | 0,05668 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,72002 | -2,61631 | — | -1,51769 | -1,57438 |
| Cadeia de 1ª ordem | 6 | -1,05136 | -1,02588 | 1.867.627,0 | 0,07274 | 0,01605 |
| MTD ordem 1 | 6 | -1,05136 | -1,02588 | 1.867.627,0 | 0,07274 | 0,01605 |
| MTD ordem 2 | 7 | -1,03853 | -1,01583 | 1.844.852,8 | 0,08279 | 0,02610 |
| MTD ordem 3 | 8 | -1,02982 | -1,00968 | 1.829.395,6 | 0,08893 | 0,03225 |
| MTD ordem 4 | 9 | -1,02382 | -1,00435 | 1.818.752,5 | 0,09426 | 0,03758 |
| MTD ordem 5 | 10 | -1,01906 | -1,00087 | 1.810.307,4 | 0,09774 | 0,04106 |
| MTD ordem 6 | 11 | -1,01482 | -0,99734 | 1.802.797,4 | 0,10127 | 0,04459 |
| MTD ordem 7 | 12 | -1,01213 | -0,99447 | 1.798.039,8 | 0,10414 | 0,04746 |
| MTD ordem 8 | 13 | -1,00973 | -0,99265 | 1.793.777,4 | 0,10596 | 0,04927 |
| MTD ordem 9 | 14 | -1,00751 | -0,99055 | 1.789.859,0 | 0,10806 | 0,05138 |
| MTD ordem 10 | 15 | -1,00576 | -0,98895 | 1.786.763,1 **←** | 0,10966 | 0,05298 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 50,65% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,61631 | 43,38% | 48,50% (47,86% a 49,13%) | 0,583 | 1,099 | 50,65% |
| Cadeia de 1ª ordem | -1,02588 | 50,65% | 51,33% (50,69% a 51,97%) | 0,583 | 0,615 | 100,00% |
| MTD ordem 1 | -1,02588 | 50,65% | 51,33% (50,69% a 51,97%) | 0,583 | 0,615 | 100,00% |
| MTD ordem 2 | -1,01583 | 49,34% | 51,31% (50,67% a 51,94%) | 0,618 | 0,607 | 71,48% |
| MTD ordem 3 | -1,00968 | 50,49% | 51,05% (50,41% a 51,69%) | 0,635 | 0,603 | 82,07% |
| MTD ordem 4 | -1,00435 | 50,80% | 51,20% (50,56% a 51,83%) | 0,646 | 0,599 | 82,43% |
| MTD ordem 5 | -1,00087 | 50,29% | 51,36% (50,72% a 52,00%) | 0,652 | 0,596 | 73,71% |
| MTD ordem 6 | -0,99734 | 51,05% | 51,25% (50,61% a 51,89%) | 0,659 | 0,594 | 80,13% |
| MTD ordem 7 | -0,99447 | 51,31% | 51,41% (50,77% a 52,04%) | 0,663 | 0,592 | 80,01% |
| MTD ordem 8 | -0,99265 | 51,26% | 51,54% (50,90% a 52,17%) | 0,667 | 0,591 | 77,03% |
| MTD ordem 9 | -0,99055 | 51,54% | 51,25% (50,61% a 51,89%) | 0,670 | 0,589 | 79,11% |
| MTD ordem 10 | -0,98895 | 51,60% | 51,24% (50,60% a 51,87%) | 0,672 | 0,588 | 79,09% |

Direção: entre os candles de teste que se moveram (23.633), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 11.850 | 27,59% | 42,45% | 29,97% |
| +1 | 11.784 | 30,65% | 41,99% | 27,36% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 50,65% dos candles de teste ficaram parados e 49,35% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 50,65% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 100,00% | 50,65% | 100,00% | 0,00% | — | 0,00% |
| MTD ordem 6 | 80,13% | 55,10% | 87,16% | 19,87% | 67,27% | 27,08% |
| MTD ordem 10 | 79,09% | 55,79% | 87,11% | 20,91% | 68,78% | 29,15% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | — / 0,00% / 0,000 | 50,65% / 100,00% / 0,672 | — / 0,00% / 0,000 | 0,224 |
| MTD ordem 6 | 34,14% / 12,63% / 0,184 | 55,10% / 87,16% / 0,675 | 35,25% / 15,35% / 0,214 | 0,358 |
| MTD ordem 10 | 36,40% / 13,30% / 0,195 | 55,79% / 87,11% / 0,680 | 35,29% / 17,02% / 0,230 | 0,368 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 1.576 | 8.356 | 1.917 | 11.849 |
| 0 | 1.365 | 21.133 | 1.762 | 24.260 |
| +1 | 1.389 | 8.389 | 2.006 | 11.784 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,233 | 0,311 | 0,222 | 0,294 |
| 2 | 0,299 | 0,344 | 0,306 | 0,328 |
| 3 | 0,377 | 0,395 | 0,363 | 0,386 |
| 4 | 0,399 | 0,420 | 0,415 | 0,421 |
| 5 | 0,471 | 0,487 | 0,463 | 0,478 |
| 6 | 0,489 | 0,522 | 0,509 | 0,517 |
| 7 | 0,563 | 0,545 | 0,558 | 0,560 |
| 8 | 0,592 | 0,595 | 0,613 | 0,611 |
| 9 | 0,669 | 0,671 | 0,677 | 0,675 |
| 10 | 0,745 | 0,774 | 0,778 | 0,795 |
| P(0) p10 a p90 | 0,29 a 0,68 |  | 0,27 a 0,71 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0419 | 0,583 | -1,0259 | 0,659 | -0,9973 | 0,672 | -0,9889 |
| 15 min | 0,500 | -1,0419 | 0,577 | -1,0409 | 0,645 | -1,0069 | 0,661 | -0,9960 |
| 1 h | 0,500 | -1,0419 | 0,571 | -1,0419 | 0,630 | -1,0258 | 0,646 | -1,0075 |
| 4 h | 0,500 | -1,0419 | 0,529 | -1,0419 | 0,595 | -1,0406 | 0,606 | -1,0374 |
| 1 dia | 0,500 | -1,0419 | 0,520 | -1,0419 | 0,490 | -1,0406 | 0,580 | -1,0399 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 27,53% / 44,83% / 27,65% | 0,189 | 0,6 | 3 |
| 2 | 0,507 | 0,493 | 0,493 | 27,48% / 44,92% / 27,60% | 0,317 | 1,4 | 7 |
| 3 | 0,345 | 0,336 | 0,319 | 27,41% / 45,06% / 27,53% | 0,407 | 2,4 | 12 |
| 4 | 0,268 | 0,259 | 0,227 | 27,35% / 45,19% / 27,46% | 0,472 | 3,5 | 17 |
| 5 | 0,222 | 0,213 | 0,182 | 27,29% / 45,30% / 27,41% | 0,524 | 4,8 | 24 |
| 6 | 0,189 | 0,182 | 0,159 | 27,23% / 45,42% / 27,35% | 0,568 | 6,4 | 32 |
| 7 | 0,169 | 0,161 | 0,120 | 27,19% / 45,51% / 27,30% | 0,601 | 7,9 | 40 |
| 8 | 0,153 | 0,145 | 0,108 | 27,15% / 45,59% / 27,26% | 0,629 | 9,7 | 48 |
| 9 | 0,141 | 0,132 | 0,099 | 27,11% / 45,67% / 27,22% | 0,654 | 11,7 | 58 |
| 10 | 0,131 | 0,123 | 0,085 | 27,07% / 45,74% / 27,19% | 0,675 | 13,8 | 69 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3561 | 0,1223 | 0,5216 |
| 0 | 0,0877 | 0,8236 | 0,0887 |
| +1 | 0,4936 | 0,1750 | 0,3314 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,131, 0,123, 0,111, 0,097, 0,095, 0,100, 0,082, 0,085, 0,089, 0,085

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 8 | 80 | 0 | 16s |
| 3 | 10 | 10 | 10 | 0,01 | 6 | 118 | 0 | 24s |
| 4 | 10 | 10 | 10 | 0,01 | 6 | 112 | 0 | 26s |
| 5 | 10 | 10 | 10 | 0,02 | 1 | 152 | 0 | 31s |
| 6 | 10 | 10 | 10 | 0,02 | 3 | 114 | 0 | 33s |
| 7 | 10 | 10 | 10 | 0,02 | 1 | 151 | 0 | 41s |
| 8 | 10 | 10 | 10 | 0,03 | 1 | 151 | 0 | 39s |
| 9 | 10 | 10 | 10 | 0,03 | 1 | 151 | 0 | 49s |
| 10 | 10 | 10 | 10 | 0,03 | 1 | 151 | 0 | 51s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 40,48% | 43,31% | 43,56% |
| 04 | 51,16% | 48,51% | 49,08% |
| 05 | 52,07% | 49,31% | 50,07% |
| 08 | 47,31% | 46,80% | 47,38% |
| 12 | 43,11% | 44,73% | 45,12% |
| 14 | 37,05% | 40,85% | 40,30% |
| 15 | 37,75% | 40,75% | 39,94% |
| 17 | 42,26% | 43,11% | 42,42% |
| 20 | 44,46% | 44,89% | 44,79% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,988 | 8,56% | 15,02% | 1,000 |
| MTD ordem 10 | 0,973 | 10,13% | 15,02% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 42,11% | 43,51% | 43,25% |
| Ter | 41,38% | 43,13% | 42,81% |
| Qua | 41,45% | 43,16% | 42,84% |
| Qui | 41,24% | 43,03% | 42,69% |
| Sex | 41,93% | 43,41% | 43,10% |
| Sáb | 55,16% | 50,92% | 52,03% |
| Dom | 52,59% | 49,58% | 50,53% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,189 | 0,188 | 0,181 | 0,172 | 0,146 | 0,133 |
| Movimento, MTD ordem 1 | 0,188 | 0,034 | 0,001 | -0,001 | 0,001 | 0,001 |
| Movimento, MTD ordem 6 | 0,192 | 0,189 | 0,178 | 0,056 | 0,001 | 0,001 |
| Movimento, MTD ordem 10 | 0,194 | 0,193 | 0,182 | 0,116 | 0,008 | 0,000 |
| Símbolo, real | -0,044 | -0,032 | -0,007 | 0,000 | 0,002 | -0,002 |
| Símbolo, MTD ordem 1 | -0,045 | 0,002 | -0,000 | -0,002 | -0,001 | -0,000 |
| Símbolo, MTD ordem 6 | -0,023 | -0,021 | -0,020 | -0,002 | -0,000 | 0,000 |
| Símbolo, MTD ordem 10 | -0,020 | -0,017 | -0,015 | 0,001 | -0,002 | 0,002 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 33 | 0,00% | -41,1 | -1,05136 | -1,05135 |
| 6 | 198 | 0,02% | -221,1 | -1,01482 | -1,01480 |
| 10 | 330 | 0,04% | -358,7 | -1,00576 | -1,00573 |

## Discretização `blocks`, s/3, blocos de 6 mês(es)

- Símbolos: treino 861.723, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 30 segmentos no treino
- Classes treino (−1/0/+1): 26,17% / 47,58% / 26,25%
- Classes teste (−1/0/+1): 26,08% / 47,98% / 25,95%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,04565 |
| Frequências do treino (iid) | 2 | -1,05533 | -1,05297 | 1.818.808,5 | 0,04565 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,66439 | -2,68705 | — | -1,58844 | -1,63409 |
| Cadeia de 1ª ordem | 6 | -1,03616 | -1,03959 | 1.785.830,5 | 0,05902 | 0,01337 |
| MTD ordem 1 | 6 | -1,03616 | -1,03959 | 1.785.830,5 | 0,05902 | 0,01337 |
| MTD ordem 2 | 7 | -1,02309 | -1,03151 | 1.763.308,1 | 0,06710 | 0,02146 |
| MTD ordem 3 | 8 | -1,01399 | -1,02659 | 1.747.651,5 | 0,07202 | 0,02637 |
| MTD ordem 4 | 9 | -1,00779 | -1,02188 | 1.736.978,4 | 0,07673 | 0,03109 |
| MTD ordem 5 | 10 | -1,00296 | -1,01817 | 1.728.670,1 | 0,08044 | 0,03479 |
| MTD ordem 6 | 11 | -0,99852 | -1,01518 | 1.721.020,7 | 0,08343 | 0,03778 |
| MTD ordem 7 | 12 | -0,99585 | -1,01212 | 1.716.431,8 | 0,08649 | 0,04084 |
| MTD ordem 8 | 13 | -0,99333 | -1,01036 | 1.712.115,0 | 0,08825 | 0,04261 |
| MTD ordem 9 | 14 | -0,99111 | -1,00832 | 1.708.302,4 | 0,09029 | 0,04465 |
| MTD ordem 10 | 15 | -0,98932 | -1,00661 | 1.705.233,2 **←** | 0,09200 | 0,04635 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 47,98% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,68705 | 41,83% | 48,31% (47,69% a 48,93%) | 0,582 | 1,129 | 47,97% |
| Cadeia de 1ª ordem | -1,03959 | 47,98% | 51,51% (50,88% a 52,13%) | 0,581 | 0,625 | 100,00% |
| MTD ordem 1 | -1,03959 | 47,98% | 51,51% (50,88% a 52,13%) | 0,581 | 0,625 | 100,00% |
| MTD ordem 2 | -1,03151 | 47,26% | 51,24% (50,62% a 51,86%) | 0,615 | 0,619 | 68,85% |
| MTD ordem 3 | -1,02659 | 48,22% | 51,38% (50,76% a 52,00%) | 0,632 | 0,615 | 79,94% |
| MTD ordem 4 | -1,02188 | 48,69% | 51,51% (50,89% a 52,13%) | 0,641 | 0,612 | 85,09% |
| MTD ordem 5 | -1,01817 | 48,42% | 51,85% (51,23% a 52,47%) | 0,649 | 0,609 | 74,58% |
| MTD ordem 6 | -1,01518 | 48,87% | 51,51% (50,88% a 52,13%) | 0,655 | 0,607 | 77,77% |
| MTD ordem 7 | -1,01212 | 49,27% | 51,80% (51,18% a 52,42%) | 0,659 | 0,605 | 80,55% |
| MTD ordem 8 | -1,01036 | 49,16% | 51,75% (51,13% a 52,37%) | 0,663 | 0,604 | 76,60% |
| MTD ordem 9 | -1,00832 | 49,32% | 51,44% (50,82% a 52,06%) | 0,666 | 0,602 | 77,04% |
| MTD ordem 10 | -1,00661 | 49,39% | 51,67% (51,05% a 52,29%) | 0,669 | 0,601 | 78,15% |

Direção: entre os candles de teste que se moveram (24.916), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 12.491 | 28,42% | 40,41% | 31,17% |
| +1 | 12.426 | 31,89% | 39,86% | 28,25% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 47,98% dos candles de teste ficaram parados e 52,02% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 47,98% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 100,00% | 47,98% | 100,00% | 0,00% | — | 0,00% |
| MTD ordem 6 | 77,77% | 52,72% | 85,47% | 22,23% | 68,64% | 29,33% |
| MTD ordem 10 | 78,15% | 53,00% | 86,33% | 21,85% | 70,00% | 29,40% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | — / 0,00% / 0,000 | 47,98% / 100,00% / 0,648 | — / 0,00% / 0,000 | 0,216 |
| MTD ordem 6 | 35,20% / 14,02% / 0,201 | 52,72% / 85,47% / 0,652 | 35,53% / 16,22% / 0,223 | 0,358 |
| MTD ordem 10 | 37,64% / 13,33% / 0,197 | 53,00% / 86,33% / 0,657 | 35,62% / 17,32% / 0,233 | 0,362 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 1.665 | 8.759 | 2.066 | 12.490 |
| 0 | 1.317 | 19.837 | 1.823 | 22.977 |
| +1 | 1.442 | 8.832 | 2.152 | 12.426 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,234 | 0,288 | 0,218 | 0,268 |
| 2 | 0,302 | 0,336 | 0,300 | 0,321 |
| 3 | 0,370 | 0,372 | 0,356 | 0,369 |
| 4 | 0,400 | 0,414 | 0,405 | 0,394 |
| 5 | 0,457 | 0,435 | 0,449 | 0,451 |
| 6 | 0,494 | 0,487 | 0,497 | 0,484 |
| 7 | 0,544 | 0,502 | 0,545 | 0,516 |
| 8 | 0,592 | 0,567 | 0,599 | 0,560 |
| 9 | 0,666 | 0,636 | 0,670 | 0,649 |
| 10 | 0,751 | 0,760 | 0,781 | 0,786 |
| P(0) p10 a p90 | 0,29 a 0,69 |  | 0,27 a 0,71 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0530 | 0,581 | -1,0396 | 0,655 | -1,0152 | 0,669 | -1,0066 |
| 15 min | 0,500 | -1,0530 | 0,574 | -1,0520 | 0,642 | -1,0239 | 0,658 | -1,0132 |
| 1 h | 0,500 | -1,0530 | 0,571 | -1,0530 | 0,628 | -1,0391 | 0,642 | -1,0235 |
| 4 h | 0,500 | -1,0530 | 0,524 | -1,0530 | 0,595 | -1,0529 | 0,605 | -1,0505 |
| 1 dia | 0,500 | -1,0530 | 0,517 | -1,0530 | 0,509 | -1,0530 | 0,580 | -1,0530 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 26,17% / 47,58% / 26,25% | 0,191 | 0,6 | 3 |
| 2 | 0,504 | 0,496 | 0,496 | 26,12% / 47,68% / 26,20% | 0,319 | 1,4 | 7 |
| 3 | 0,342 | 0,335 | 0,323 | 26,04% / 47,85% / 26,12% | 0,412 | 2,4 | 12 |
| 4 | 0,265 | 0,259 | 0,229 | 25,96% / 48,00% / 26,04% | 0,478 | 3,5 | 18 |
| 5 | 0,220 | 0,211 | 0,182 | 25,89% / 48,14% / 25,97% | 0,529 | 4,9 | 24 |
| 6 | 0,187 | 0,180 | 0,161 | 25,82% / 48,29% / 25,89% | 0,574 | 6,5 | 33 |
| 7 | 0,167 | 0,160 | 0,118 | 25,76% / 48,41% / 25,84% | 0,606 | 8,1 | 40 |
| 8 | 0,151 | 0,143 | 0,109 | 25,70% / 48,51% / 25,78% | 0,635 | 9,9 | 50 |
| 9 | 0,139 | 0,131 | 0,099 | 25,66% / 48,61% / 25,73% | 0,660 | 11,9 | 60 |
| 10 | 0,129 | 0,121 | 0,086 | 25,61% / 48,70% / 25,69% | 0,681 | 14,1 | 70 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3543 | 0,1296 | 0,5161 |
| 0 | 0,0822 | 0,8364 | 0,0813 |
| +1 | 0,4878 | 0,1809 | 0,3313 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,129, 0,121, 0,113, 0,098, 0,095, 0,102, 0,081, 0,087, 0,089, 0,086

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 0 | 37 | 0 | 17s |
| 3 | 10 | 10 | 10 | 0,01 | 4 | 87 | 0 | 24s |
| 4 | 10 | 10 | 10 | 0,01 | 3 | 117 | 0 | 27s |
| 5 | 10 | 10 | 10 | 0,01 | 1 | 150 | 0 | 32s |
| 6 | 10 | 10 | 10 | 0,01 | 3 | 112 | 0 | 36s |
| 7 | 10 | 10 | 10 | 0,02 | 1 | 149 | 0 | 42s |
| 8 | 10 | 10 | 10 | 0,02 | 1 | 149 | 0 | 37s |
| 9 | 10 | 10 | 10 | 0,03 | 1 | 150 | 0 | 47s |
| 10 | 10 | 10 | 10 | 0,03 | 1 | 150 | 0 | 50s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 42,64% | 45,85% | 46,12% |
| 04 | 54,13% | 51,50% | 52,12% |
| 05 | 54,73% | 52,11% | 52,91% |
| 08 | 49,87% | 49,56% | 50,14% |
| 12 | 45,71% | 47,43% | 47,82% |
| 14 | 39,02% | 43,18% | 42,62% |
| 15 | 39,81% | 43,11% | 42,20% |
| 17 | 44,70% | 45,73% | 45,00% |
| 20 | 46,99% | 47,55% | 47,44% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 9,01% | 15,71% | 1,000 |
| MTD ordem 10 | 0,971 | 10,72% | 15,71% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 44,40% | 46,04% | 45,74% |
| Ter | 43,79% | 45,71% | 45,37% |
| Qua | 43,88% | 45,75% | 45,41% |
| Qui | 43,65% | 45,63% | 45,26% |
| Sex | 44,23% | 45,93% | 45,59% |
| Sáb | 57,89% | 53,76% | 54,88% |
| Dom | 55,33% | 52,42% | 53,38% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,191 | 0,190 | 0,184 | 0,173 | 0,147 | 0,135 |
| Movimento, MTD ordem 1 | 0,190 | 0,035 | 0,001 | -0,001 | 0,001 | -0,000 |
| Movimento, MTD ordem 6 | 0,196 | 0,193 | 0,183 | 0,058 | -0,001 | 0,001 |
| Movimento, MTD ordem 10 | 0,197 | 0,194 | 0,184 | 0,117 | 0,011 | -0,002 |
| Símbolo, real | -0,045 | -0,032 | -0,008 | 0,001 | 0,002 | -0,002 |
| Símbolo, MTD ordem 1 | -0,045 | 0,002 | 0,001 | -0,002 | -0,001 | -0,000 |
| Símbolo, MTD ordem 6 | -0,023 | -0,022 | -0,020 | -0,000 | 0,001 | 0,000 |
| Símbolo, MTD ordem 10 | -0,017 | -0,017 | -0,014 | 0,000 | -0,000 | 0,001 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 29 | 0,00% | -35,2 | -1,03616 | -1,03616 |
| 6 | 174 | 0,02% | -196,4 | -0,99852 | -0,99849 |
| 10 | 290 | 0,03% | -317,5 | -0,98932 | -0,98929 |

## Discretização `blocks`, s/3, blocos de 12 mês(es)

- Símbolos: treino 810.244, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 24 segmentos no treino
- Classes treino (−1/0/+1): 25,04% / 49,90% / 25,06%
- Classes teste (−1/0/+1): 25,01% / 50,68% / 24,31%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,06356 |
| Frequências do treino (iid) | 2 | -1,04039 | -1,03506 | 1.685.942,4 | 0,06356 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,57760 | -2,62080 | — | -1,52219 | -1,58575 |
| Cadeia de 1ª ordem | 6 | -1,01778 | -1,02231 | 1.649.355,1 | 0,07631 | 0,01275 |
| MTD ordem 1 | 6 | -1,01778 | -1,02231 | 1.649.355,1 | 0,07631 | 0,01275 |
| MTD ordem 2 | 7 | -1,00288 | -1,01419 | 1.625.226,6 | 0,08442 | 0,02086 |
| MTD ordem 3 | 8 | -0,99216 | -1,00944 | 1.607.875,9 | 0,08918 | 0,02562 |
| MTD ordem 4 | 9 | -0,98502 | -1,00465 | 1.596.318,0 | 0,09396 | 0,03041 |
| MTD ordem 5 | 10 | -0,97935 | -1,00140 | 1.587.142,8 | 0,09721 | 0,03365 |
| MTD ordem 6 | 11 | -0,97439 | -0,99814 | 1.579.110,6 | 0,10048 | 0,03692 |
| MTD ordem 7 | 12 | -0,97134 | -0,99508 | 1.574.187,9 | 0,10353 | 0,03998 |
| MTD ordem 8 | 13 | -0,96856 | -0,99337 | 1.569.699,3 | 0,10524 | 0,04168 |
| MTD ordem 9 | 14 | -0,96609 | -0,99124 | 1.565.709,4 | 0,10737 | 0,04381 |
| MTD ordem 10 | 15 | -0,96421 | -0,98966 | 1.562.669,7 **←** | 0,10895 | 0,04539 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 50,68% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,62080 | 43,28% | 48,81% (48,18% a 49,45%) | 0,582 | 1,101 | 50,68% |
| Cadeia de 1ª ordem | -1,02231 | 50,68% | 51,68% (51,04% a 52,31%) | 0,581 | 0,612 | 100,00% |
| MTD ordem 1 | -1,02231 | 50,68% | 51,68% (51,04% a 52,31%) | 0,581 | 0,612 | 100,00% |
| MTD ordem 2 | -1,01419 | 49,12% | 51,33% (50,69% a 51,96%) | 0,615 | 0,606 | 71,58% |
| MTD ordem 3 | -1,00944 | 50,39% | 50,89% (50,26% a 51,53%) | 0,632 | 0,603 | 82,25% |
| MTD ordem 4 | -1,00465 | 51,00% | 51,52% (50,89% a 52,16%) | 0,642 | 0,599 | 87,77% |
| MTD ordem 5 | -1,00140 | 50,50% | 51,48% (50,84% a 52,12%) | 0,650 | 0,597 | 79,32% |
| MTD ordem 6 | -0,99814 | 50,89% | 51,38% (50,74% a 52,01%) | 0,656 | 0,595 | 80,76% |
| MTD ordem 7 | -0,99508 | 51,29% | 51,43% (50,79% a 52,06%) | 0,660 | 0,593 | 84,35% |
| MTD ordem 8 | -0,99337 | 51,26% | 51,66% (51,02% a 52,29%) | 0,663 | 0,591 | 80,42% |
| MTD ordem 9 | -0,99124 | 51,43% | 51,40% (50,76% a 52,03%) | 0,667 | 0,590 | 80,39% |
| MTD ordem 10 | -0,98966 | 51,57% | 51,57% (50,94% a 52,21%) | 0,670 | 0,589 | 82,16% |

Direção: entre os candles de teste que se moveram (23.622), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 11.979 | 27,61% | 42,88% | 29,52% |
| +1 | 11.644 | 30,99% | 41,87% | 27,15% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 50,68% dos candles de teste ficaram parados e 49,32% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 50,68% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 100,00% | 50,68% | 100,00% | 0,00% | — | 0,00% |
| MTD ordem 6 | 80,76% | 54,89% | 87,47% | 19,24% | 66,99% | 26,13% |
| MTD ordem 10 | 82,16% | 55,00% | 89,18% | 17,84% | 69,25% | 25,04% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | — / 0,00% / 0,000 | 50,68% / 100,00% / 0,673 | — / 0,00% / 0,000 | 0,224 |
| MTD ordem 6 | 34,09% / 11,88% / 0,176 | 54,89% / 87,47% / 0,674 | 34,15% / 14,78% / 0,206 | 0,352 |
| MTD ordem 10 | 36,36% / 11,05% / 0,170 | 55,00% / 89,18% / 0,680 | 35,32% / 14,87% / 0,209 | 0,353 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 1.324 | 8.978 | 1.676 | 11.978 |
| 0 | 1.133 | 21.644 | 1.494 | 24.271 |
| +1 | 1.184 | 8.729 | 1.731 | 11.644 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,242 | 0,318 | 0,228 | 0,294 |
| 2 | 0,312 | 0,345 | 0,318 | 0,342 |
| 3 | 0,396 | 0,404 | 0,377 | 0,384 |
| 4 | 0,416 | 0,433 | 0,431 | 0,433 |
| 5 | 0,497 | 0,489 | 0,481 | 0,475 |
| 6 | 0,511 | 0,504 | 0,527 | 0,519 |
| 7 | 0,593 | 0,537 | 0,579 | 0,545 |
| 8 | 0,620 | 0,593 | 0,637 | 0,601 |
| 9 | 0,706 | 0,665 | 0,706 | 0,669 |
| 10 | 0,788 | 0,779 | 0,815 | 0,805 |
| P(0) p10 a p90 | 0,30 a 0,72 |  | 0,28 a 0,74 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0351 | 0,581 | -1,0223 | 0,656 | -0,9981 | 0,670 | -0,9897 |
| 15 min | 0,500 | -1,0351 | 0,574 | -1,0338 | 0,643 | -1,0072 | 0,658 | -0,9966 |
| 1 h | 0,500 | -1,0351 | 0,569 | -1,0351 | 0,628 | -1,0189 | 0,642 | -1,0043 |
| 4 h | 0,500 | -1,0351 | 0,477 | -1,0351 | 0,592 | -1,0349 | 0,603 | -1,0315 |
| 1 dia | 0,500 | -1,0351 | 0,483 | -1,0351 | 0,529 | -1,0350 | 0,578 | -1,0352 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 25,04% / 49,90% / 25,06% | 0,208 | 0,6 | 3 |
| 2 | 0,507 | 0,493 | 0,493 | 24,95% / 50,07% / 24,97% | 0,343 | 1,5 | 7 |
| 3 | 0,341 | 0,330 | 0,329 | 24,82% / 50,34% / 24,84% | 0,441 | 2,6 | 13 |
| 4 | 0,262 | 0,255 | 0,232 | 24,70% / 50,58% / 24,72% | 0,509 | 3,9 | 19 |
| 5 | 0,215 | 0,207 | 0,186 | 24,57% / 50,83% / 24,59% | 0,562 | 5,4 | 27 |
| 6 | 0,183 | 0,175 | 0,161 | 24,46% / 51,06% / 24,48% | 0,607 | 7,2 | 36 |
| 7 | 0,163 | 0,155 | 0,120 | 24,36% / 51,26% / 24,38% | 0,639 | 9,1 | 45 |
| 8 | 0,148 | 0,139 | 0,109 | 24,27% / 51,45% / 24,28% | 0,668 | 11,1 | 56 |
| 9 | 0,135 | 0,127 | 0,100 | 24,18% / 51,63% / 24,20% | 0,692 | 13,5 | 68 |
| 10 | 0,126 | 0,117 | 0,084 | 24,09% / 51,80% / 24,11% | 0,712 | 15,9 | 80 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3624 | 0,1248 | 0,5127 |
| 0 | 0,0702 | 0,8614 | 0,0684 |
| +1 | 0,4862 | 0,1730 | 0,3407 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,126, 0,117, 0,114, 0,099, 0,098, 0,102, 0,083, 0,087, 0,090, 0,084

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 5 | 102 | 0 | 13s |
| 3 | 10 | 10 | 10 | 0,01 | 1 | 148 | 0 | 19s |
| 4 | 10 | 10 | 10 | 0,00 | 3 | 107 | 0 | 20s |
| 5 | 10 | 10 | 10 | 0,01 | 3 | 107 | 0 | 25s |
| 6 | 10 | 10 | 10 | 0,01 | 3 | 104 | 0 | 27s |
| 7 | 10 | 10 | 10 | 0,01 | 1 | 140 | 0 | 33s |
| 8 | 10 | 10 | 10 | 0,02 | 1 | 141 | 0 | 32s |
| 9 | 10 | 10 | 10 | 0,02 | 1 | 142 | 0 | 41s |
| 10 | 10 | 10 | 10 | 0,02 | 1 | 143 | 0 | 44s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 45,12% | 48,39% | 48,79% |
| 04 | 56,63% | 54,32% | 54,93% |
| 05 | 57,15% | 54,94% | 55,80% |
| 08 | 52,32% | 52,15% | 52,81% |
| 12 | 48,12% | 50,05% | 50,56% |
| 14 | 41,06% | 45,16% | 44,63% |
| 15 | 41,94% | 45,13% | 44,23% |
| 17 | 46,51% | 47,82% | 47,05% |
| 20 | 49,33% | 49,89% | 49,75% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,986 | 9,81% | 16,10% | 1,000 |
| MTD ordem 10 | 0,971 | 11,57% | 16,10% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 46,33% | 48,17% | 47,88% |
| Ter | 45,93% | 47,94% | 47,60% |
| Qua | 46,10% | 48,05% | 47,74% |
| Qui | 45,87% | 47,90% | 47,57% |
| Sex | 46,52% | 48,28% | 47,98% |
| Sáb | 60,76% | 56,90% | 58,09% |
| Dom | 58,11% | 55,45% | 56,49% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,208 | 0,207 | 0,202 | 0,191 | 0,164 | 0,151 |
| Movimento, MTD ordem 1 | 0,207 | 0,043 | 0,001 | -0,000 | 0,002 | -0,000 |
| Movimento, MTD ordem 6 | 0,214 | 0,212 | 0,205 | 0,074 | 0,001 | 0,001 |
| Movimento, MTD ordem 10 | 0,219 | 0,217 | 0,210 | 0,142 | 0,017 | 0,000 |
| Símbolo, real | -0,042 | -0,030 | -0,008 | 0,001 | 0,001 | -0,002 |
| Símbolo, MTD ordem 1 | -0,042 | 0,001 | 0,000 | -0,002 | -0,001 | -0,001 |
| Símbolo, MTD ordem 6 | -0,018 | -0,019 | -0,018 | -0,001 | 0,001 | 0,001 |
| Símbolo, MTD ordem 10 | -0,015 | -0,015 | -0,013 | 0,001 | 0,001 | 0,000 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 23 | 0,00% | -27,0 | -1,01778 | -1,01777 |
| 6 | 138 | 0,02% | -149,1 | -0,97439 | -0,97437 |
| 10 | 230 | 0,03% | -244,1 | -0,96421 | -0,96418 |
