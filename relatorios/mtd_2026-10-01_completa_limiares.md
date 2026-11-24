# Relatório completo do MTD (tabelas)

Gerado em 2026-11-23 02:42 (UTC) por `src/experiments/mtd_analysis.py`. Mesmas métricas de `relatorios/analise_hmm_2026-10-01_completa.md`. A leitura dos números está em `relatorios/analise_mtd_2026-10-01_completa.md`.

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
| blocks | 1 | 4 | 33,41% | 35,36% | 10 | -1,05527 | 0,04329 | 0,669 | 51,15% (50,59% a 51,70%) | 50,80% |
| blocks | 1 | 5 | 27,53% | 29,56% | 10 | -1,06074 | 0,03563 | 0,658 | 51,28% (50,75% a 51,82%) | 50,64% |
| global | — | 3 | 51,02% | 67,96% | 10 | -0,80213 | 0,10595 | 0,697 | 50,44% (49,65% a 51,23%) | 50,52% |
| global | — | 4 | 41,35% | 57,45% | 10 | -0,93281 | 0,09656 | 0,681 | 50,91% (50,23% a 51,60%) | 50,80% |
| global | — | 5 | 34,61% | 49,59% | 10 | -1,00330 | 0,08640 | 0,669 | 51,71% (51,08% a 52,34%) | 50,76% |

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

## Discretização `blocks`, s/4, blocos de 1 mês(es)

- Símbolos: treino 905.654, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 35 segmentos no treino
- Classes treino (−1/0/+1): 33,23% / 33,41% / 33,36%
- Classes teste (−1/0/+1): 32,84% / 35,36% / 31,80%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,00005 |
| Frequências do treino (iid) | 2 | -1,09861 | -1,09856 | 1.989.926,1 | 0,00005 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,94994 | -2,86895 | — | -1,77033 | -1,77038 |
| Cadeia de 1ª ordem | 6 | -1,08923 | -1,08662 | 1.972.996,3 | 0,01199 | 0,01194 |
| MTD ordem 1 | 6 | -1,08923 | -1,08662 | 1.972.996,3 | 0,01199 | 0,01194 |
| MTD ordem 2 | 7 | -1,08229 | -1,07961 | 1.960.439,8 | 0,01900 | 0,01895 |
| MTD ordem 3 | 8 | -1,07743 | -1,07375 | 1.951.643,3 | 0,02486 | 0,02481 |
| MTD ordem 4 | 9 | -1,07387 | -1,06932 | 1.945.217,2 | 0,02929 | 0,02924 |
| MTD ordem 5 | 10 | -1,07095 | -1,06617 | 1.939.939,9 | 0,03245 | 0,03240 |
| MTD ordem 6 | 11 | -1,06827 | -1,06304 | 1.935.089,3 | 0,03558 | 0,03553 |
| MTD ordem 7 | 12 | -1,06656 | -1,06033 | 1.932.015,7 | 0,03829 | 0,03824 |
| MTD ordem 8 | 13 | -1,06486 | -1,05854 | 1.928.951,1 | 0,04007 | 0,04002 |
| MTD ordem 9 | 14 | -1,06338 | -1,05679 | 1.926.283,0 | 0,04183 | 0,04177 |
| MTD ordem 10 | 15 | -1,06219 | -1,05527 | 1.924.137,6 **←** | 0,04334 | 0,04329 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 35,36% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,86895 | 37,87% | 48,96% (48,40% a 49,52%) | 0,578 | 1,206 | 35,36% |
| Cadeia de 1ª ordem | -1,08662 | 39,57% | 51,04% (50,48% a 51,60%) | 0,576 | 0,658 | 35,36% |
| MTD ordem 1 | -1,08662 | 39,57% | 51,04% (50,48% a 51,60%) | 0,576 | 0,658 | 35,36% |
| MTD ordem 2 | -1,07961 | 40,33% | 50,76% (50,20% a 51,31%) | 0,607 | 0,654 | 54,66% |
| MTD ordem 3 | -1,07375 | 40,55% | 51,50% (50,94% a 52,05%) | 0,626 | 0,649 | 40,31% |
| MTD ordem 4 | -1,06932 | 41,54% | 51,60% (51,04% a 52,16%) | 0,638 | 0,646 | 42,99% |
| MTD ordem 5 | -1,06617 | 41,89% | 51,69% (51,13% a 52,25%) | 0,646 | 0,644 | 49,10% |
| MTD ordem 6 | -1,06304 | 41,89% | 51,19% (50,63% a 51,75%) | 0,652 | 0,642 | 43,23% |
| MTD ordem 7 | -1,06033 | 42,33% | 51,48% (50,93% a 52,04%) | 0,658 | 0,640 | 44,55% |
| MTD ordem 8 | -1,05854 | 42,63% | 51,48% (50,93% a 52,04%) | 0,662 | 0,639 | 47,01% |
| MTD ordem 9 | -1,05679 | 42,56% | 51,50% (50,95% a 52,06%) | 0,666 | 0,638 | 45,29% |
| MTD ordem 10 | -1,05527 | 42,62% | 51,15% (50,59% a 51,70%) | 0,669 | 0,637 | 45,72% |

Direção: entre os candles de teste que se moveram (30.959), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 15.728 | 34,11% | 30,22% | 35,67% |
| +1 | 15.232 | 37,12% | 29,50% | 33,38% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 35,36% dos candles de teste ficaram parados e 64,64% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 35,36% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 35,36% | 45,40% | 45,39% | 64,64% | 70,13% | 70,13% |
| MTD ordem 6 | 43,23% | 46,98% | 57,45% | 56,77% | 73,49% | 64,54% |
| MTD ordem 10 | 45,72% | 47,37% | 61,25% | 54,28% | 74,76% | 62,78% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | 37,12% / 35,95% / 0,365 | 45,40% / 45,39% / 0,454 | 35,67% / 36,83% / 0,362 | 0,394 |
| MTD ordem 6 | 38,69% / 30,20% / 0,339 | 46,98% / 57,45% / 0,517 | 37,47% / 36,67% / 0,371 | 0,409 |
| MTD ordem 10 | 39,14% / 29,99% / 0,340 | 47,37% / 61,25% / 0,534 | 38,16% / 34,94% / 0,365 | 0,413 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 4.717 | 5.845 | 5.165 | 15.727 |
| 0 | 3.102 | 10.372 | 3.460 | 16.934 |
| +1 | 4.232 | 5.678 | 5.322 | 15.232 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,180 | 0,195 | 0,159 | 0,171 |
| 2 | 0,216 | 0,219 | 0,206 | 0,199 |
| 3 | 0,256 | 0,269 | 0,248 | 0,259 |
| 4 | 0,271 | 0,264 | 0,278 | 0,279 |
| 5 | 0,325 | 0,326 | 0,314 | 0,322 |
| 6 | 0,340 | 0,352 | 0,349 | 0,344 |
| 7 | 0,386 | 0,375 | 0,385 | 0,373 |
| 8 | 0,416 | 0,425 | 0,432 | 0,440 |
| 9 | 0,480 | 0,485 | 0,491 | 0,497 |
| 10 | 0,569 | 0,626 | 0,600 | 0,651 |
| P(0) p10 a p90 | 0,18 a 0,50 |  | 0,19 a 0,53 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0986 | 0,576 | -1,0866 | 0,652 | -1,0630 | 0,669 | -1,0553 |
| 15 min | 0,500 | -1,0986 | 0,574 | -1,0982 | 0,643 | -1,0705 | 0,660 | -1,0605 |
| 1 h | 0,500 | -1,0986 | 0,569 | -1,0986 | 0,627 | -1,0918 | 0,645 | -1,0771 |
| 4 h | 0,500 | -1,0986 | 0,500 | -1,0986 | 0,601 | -1,0984 | 0,613 | -1,0978 |
| 1 dia | 0,500 | -1,0986 | 0,500 | -1,0986 | 0,503 | -1,0984 | 0,588 | -1,0983 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 33,23% / 33,41% / 33,36% | 0,130 | 0,5 | 2 |
| 2 | 0,508 | 0,492 | 0,492 | 33,22% / 33,44% / 33,34% | 0,228 | 1,1 | 5 |
| 3 | 0,352 | 0,336 | 0,312 | 33,20% / 33,48% / 33,32% | 0,304 | 1,8 | 9 |
| 4 | 0,274 | 0,261 | 0,224 | 33,17% / 33,53% / 33,30% | 0,362 | 2,6 | 13 |
| 5 | 0,226 | 0,216 | 0,178 | 33,15% / 33,56% / 33,28% | 0,410 | 3,5 | 18 |
| 6 | 0,194 | 0,183 | 0,155 | 33,13% / 33,61% / 33,26% | 0,453 | 4,6 | 23 |
| 7 | 0,172 | 0,163 | 0,115 | 33,12% / 33,64% / 33,24% | 0,486 | 5,7 | 28 |
| 8 | 0,156 | 0,146 | 0,108 | 33,10% / 33,68% / 33,23% | 0,516 | 6,9 | 35 |
| 9 | 0,143 | 0,133 | 0,096 | 33,08% / 33,70% / 33,21% | 0,542 | 8,3 | 41 |
| 10 | 0,132 | 0,124 | 0,083 | 33,07% / 33,73% / 33,20% | 0,565 | 9,6 | 48 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3459 | 0,1244 | 0,5297 |
| 0 | 0,1413 | 0,7116 | 0,1471 |
| +1 | 0,5080 | 0,1691 | 0,3229 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,132, 0,124, 0,111, 0,099, 0,096, 0,100, 0,081, 0,088, 0,087, 0,083

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 9 | 115 | 0 | 23s |
| 3 | 10 | 10 | 10 | 0,01 | 1 | 257 | 0 | 33s |
| 4 | 10 | 10 | 10 | 0,02 | 1 | 233 | 0 | 29s |
| 5 | 10 | 10 | 10 | 0,02 | 3 | 168 | 0 | 37s |
| 6 | 10 | 10 | 10 | 0,02 | 1 | 210 | 0 | 40s |
| 7 | 10 | 10 | 10 | 0,05 | 1 | 207 | 0 | 47s |
| 8 | 10 | 10 | 10 | 0,04 | 1 | 202 | 0 | 44s |
| 9 | 10 | 10 | 10 | 0,04 | 1 | 199 | 0 | 60s |
| 10 | 10 | 10 | 10 | 0,05 | 1 | 196 | 0 | 58s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 29,45% | 32,23% | 32,33% |
| 04 | 38,47% | 35,73% | 36,22% |
| 05 | 39,36% | 36,31% | 36,97% |
| 08 | 35,27% | 34,54% | 34,98% |
| 12 | 31,65% | 33,12% | 33,35% |
| 14 | 26,96% | 30,74% | 30,26% |
| 15 | 27,36% | 30,57% | 29,84% |
| 17 | 31,35% | 32,21% | 31,66% |
| 20 | 32,88% | 33,31% | 33,26% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 5,74% | 12,40% | 1,000 |
| MTD ordem 10 | 0,971 | 7,13% | 12,40% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 30,68% | 32,28% | 32,00% |
| Ter | 30,29% | 32,09% | 31,77% |
| Qua | 30,52% | 32,21% | 31,92% |
| Qui | 30,15% | 32,03% | 31,70% |
| Sex | 30,58% | 32,22% | 31,90% |
| Sáb | 42,26% | 37,50% | 38,48% |
| Dom | 40,06% | 36,60% | 37,43% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,130 | 0,130 | 0,125 | 0,118 | 0,097 | 0,083 |
| Movimento, MTD ordem 1 | 0,128 | 0,016 | 0,001 | -0,000 | 0,001 | 0,000 |
| Movimento, MTD ordem 6 | 0,134 | 0,131 | 0,119 | 0,023 | 0,002 | -0,000 |
| Movimento, MTD ordem 10 | 0,135 | 0,132 | 0,122 | 0,061 | 0,001 | 0,001 |
| Símbolo, real | -0,047 | -0,033 | -0,007 | 0,000 | 0,002 | -0,001 |
| Símbolo, MTD ordem 1 | -0,047 | 0,002 | -0,000 | -0,002 | -0,001 | 0,000 |
| Símbolo, MTD ordem 6 | -0,026 | -0,024 | -0,020 | -0,002 | -0,002 | 0,000 |
| Símbolo, MTD ordem 10 | -0,021 | -0,022 | -0,015 | -0,000 | -0,001 | -0,000 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 0,00% | -38,4 | -1,08923 | -1,08923 |
| 6 | 204 | 0,02% | -220,4 | -1,06827 | -1,06826 |
| 10 | 340 | 0,04% | -364,1 | -1,06219 | -1,06219 |

## Discretização `blocks`, s/5, blocos de 1 mês(es)

- Símbolos: treino 905.654, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 35 segmentos no treino
- Classes treino (−1/0/+1): 36,17% / 27,53% / 36,29%
- Classes teste (−1/0/+1): 35,67% / 29,56% / 34,77%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,00224 |
| Frequências do treino (iid) | 2 | -1,09079 | -1,09638 | 1.975.771,1 | 0,00224 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,97655 | -2,90753 | — | -1,80891 | -1,81115 |
| Cadeia de 1ª ordem | 6 | -1,08352 | -1,08669 | 1.962.653,3 | 0,01192 | 0,00969 |
| MTD ordem 1 | 6 | -1,08352 | -1,08669 | 1.962.653,3 | 0,01192 | 0,00969 |
| MTD ordem 2 | 7 | -1,07814 | -1,08123 | 1.952.910,3 | 0,01738 | 0,01515 |
| MTD ordem 3 | 8 | -1,07446 | -1,07616 | 1.946.265,2 | 0,02245 | 0,02022 |
| MTD ordem 4 | 9 | -1,07158 | -1,07252 | 1.941.059,5 | 0,02609 | 0,02385 |
| MTD ordem 5 | 10 | -1,06925 | -1,06985 | 1.936.852,2 | 0,02876 | 0,02653 |
| MTD ordem 6 | 11 | -1,06711 | -1,06713 | 1.932.999,7 | 0,03148 | 0,02925 |
| MTD ordem 7 | 12 | -1,06572 | -1,06489 | 1.930.490,0 | 0,03372 | 0,03149 |
| MTD ordem 8 | 13 | -1,06434 | -1,06347 | 1.928.004,0 | 0,03515 | 0,03291 |
| MTD ordem 9 | 14 | -1,06311 | -1,06189 | 1.925.795,9 | 0,03673 | 0,03449 |
| MTD ordem 10 | 15 | -1,06207 | -1,06074 | 1.923.918,7 **←** | 0,03787 | 0,03563 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 29,56% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,90753 | 37,03% | 48,77% (48,23% a 49,30%) | 0,571 | 1,222 | 29,56% |
| Cadeia de 1ª ordem | -1,08669 | 38,88% | 51,23% (50,70% a 51,77%) | 0,567 | 0,659 | 29,56% |
| MTD ordem 1 | -1,08669 | 38,88% | 51,23% (50,70% a 51,77%) | 0,567 | 0,659 | 29,56% |
| MTD ordem 2 | -1,08123 | 38,70% | 51,44% (50,91% a 51,97%) | 0,597 | 0,655 | 11,68% |
| MTD ordem 3 | -1,07616 | 39,80% | 51,44% (50,91% a 51,97%) | 0,616 | 0,652 | 23,60% |
| MTD ordem 4 | -1,07252 | 39,95% | 51,26% (50,72% a 51,79%) | 0,628 | 0,650 | 26,27% |
| MTD ordem 5 | -1,06985 | 40,28% | 51,68% (51,14% a 52,21%) | 0,636 | 0,648 | 21,45% |
| MTD ordem 6 | -1,06713 | 40,46% | 51,43% (50,90% a 51,97%) | 0,643 | 0,646 | 27,04% |
| MTD ordem 7 | -1,06489 | 40,72% | 51,55% (51,02% a 52,08%) | 0,648 | 0,645 | 26,53% |
| MTD ordem 8 | -1,06347 | 40,81% | 51,55% (51,02% a 52,09%) | 0,652 | 0,644 | 25,40% |
| MTD ordem 9 | -1,06189 | 41,15% | 51,71% (51,18% a 52,25%) | 0,656 | 0,643 | 27,25% |
| MTD ordem 10 | -1,06074 | 40,91% | 51,28% (50,75% a 51,82%) | 0,658 | 0,642 | 27,39% |

Direção: entre os candles de teste que se moveram (33.735), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 17.085 | 36,29% | 25,91% | 37,80% |
| +1 | 16.650 | 39,47% | 24,86% | 35,67% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 29,56% dos candles de teste ficaram parados e 70,44% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 29,56% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 29,56% | 39,50% | 39,50% | 70,44% | 74,61% | 74,61% |
| MTD ordem 6 | 27,04% | 44,12% | 40,35% | 72,96% | 75,83% | 78,55% |
| MTD ordem 10 | 27,39% | 45,27% | 41,94% | 72,61% | 76,36% | 78,72% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | 39,47% / 38,47% / 0,390 | 39,50% / 39,50% / 0,395 | 37,80% / 38,78% / 0,383 | 0,389 |
| MTD ordem 6 | 39,54% / 38,42% / 0,390 | 44,12% / 40,35% / 0,422 | 38,71% / 42,65% / 0,406 | 0,406 |
| MTD ordem 10 | 39,78% / 38,05% / 0,389 | 45,27% / 41,94% / 0,435 | 38,80% / 42,96% / 0,408 | 0,411 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 6.501 | 3.614 | 6.969 | 17.084 |
| 0 | 3.906 | 5.938 | 4.314 | 14.158 |
| +1 | 5.934 | 3.564 | 7.153 | 16.651 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,159 | 0,167 | 0,132 | 0,138 |
| 2 | 0,166 | 0,176 | 0,175 | 0,179 |
| 3 | 0,217 | 0,222 | 0,199 | 0,191 |
| 4 | 0,230 | 0,240 | 0,231 | 0,253 |
| 5 | 0,248 | 0,248 | 0,256 | 0,258 |
| 6 | 0,292 | 0,293 | 0,286 | 0,288 |
| 7 | 0,306 | 0,316 | 0,317 | 0,319 |
| 8 | 0,352 | 0,358 | 0,354 | 0,346 |
| 9 | 0,391 | 0,395 | 0,407 | 0,421 |
| 10 | 0,481 | 0,541 | 0,511 | 0,562 |
| P(0) p10 a p90 | 0,16 a 0,43 |  | 0,14 a 0,44 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0964 | 0,567 | -1,0867 | 0,643 | -1,0671 | 0,658 | -1,0607 |
| 15 min | 0,500 | -1,0964 | 0,568 | -1,0962 | 0,634 | -1,0737 | 0,650 | -1,0655 |
| 1 h | 0,500 | -1,0964 | 0,564 | -1,0964 | 0,622 | -1,0919 | 0,641 | -1,0797 |
| 4 h | 0,500 | -1,0964 | 0,547 | -1,0964 | 0,594 | -1,0963 | 0,607 | -1,0959 |
| 1 dia | 0,500 | -1,0964 | 0,537 | -1,0964 | 0,499 | -1,0963 | 0,579 | -1,0962 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 36,17% / 27,53% / 36,29% | 0,113 | 0,5 | 2 |
| 2 | 0,512 | 0,488 | 0,488 | 36,16% / 27,55% / 36,29% | 0,200 | 1,0 | 5 |
| 3 | 0,360 | 0,337 | 0,304 | 36,15% / 27,58% / 36,27% | 0,268 | 1,6 | 8 |
| 4 | 0,278 | 0,263 | 0,223 | 36,13% / 27,61% / 36,26% | 0,323 | 2,4 | 12 |
| 5 | 0,233 | 0,217 | 0,175 | 36,12% / 27,64% / 36,25% | 0,368 | 3,2 | 16 |
| 6 | 0,199 | 0,185 | 0,151 | 36,11% / 27,66% / 36,23% | 0,409 | 4,1 | 21 |
| 7 | 0,177 | 0,165 | 0,114 | 36,09% / 27,69% / 36,22% | 0,441 | 5,0 | 25 |
| 8 | 0,160 | 0,148 | 0,106 | 36,08% / 27,71% / 36,21% | 0,470 | 6,1 | 30 |
| 9 | 0,146 | 0,135 | 0,094 | 36,07% / 27,73% / 36,20% | 0,497 | 7,3 | 36 |
| 10 | 0,136 | 0,125 | 0,083 | 36,06% / 27,75% / 36,19% | 0,520 | 8,5 | 42 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3481 | 0,1143 | 0,5376 |
| 0 | 0,1691 | 0,6526 | 0,1782 |
| +1 | 0,5199 | 0,1524 | 0,3277 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,136, 0,125, 0,109, 0,101, 0,096, 0,098, 0,081, 0,086, 0,086, 0,083

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 0 | 90 | 0 | 30s |
| 3 | 10 | 10 | 10 | 0,01 | 1 | 313 | 0 | 43s |
| 4 | 10 | 10 | 10 | 0,02 | 3 | 208 | 0 | 43s |
| 5 | 10 | 10 | 10 | 0,03 | 6 | 175 | 0 | 48s |
| 6 | 10 | 10 | 10 | 0,03 | 1 | 246 | 0 | 50s |
| 7 | 10 | 10 | 10 | 0,05 | 1 | 240 | 0 | 62s |
| 8 | 10 | 10 | 10 | 0,05 | 1 | 233 | 0 | 53s |
| 9 | 10 | 10 | 10 | 0,05 | 1 | 228 | 0 | 72s |
| 10 | 10 | 10 | 10 | 0,05 | 1 | 223 | 0 | 66s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 24,11% | 26,56% | 26,60% |
| 04 | 31,99% | 29,36% | 29,79% |
| 05 | 32,79% | 29,81% | 30,38% |
| 08 | 28,96% | 28,32% | 28,67% |
| 12 | 26,00% | 27,29% | 27,45% |
| 14 | 22,02% | 25,49% | 25,10% |
| 15 | 22,55% | 25,40% | 24,78% |
| 17 | 25,82% | 26,61% | 26,16% |
| 20 | 27,15% | 27,50% | 27,49% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 4,41% | 10,77% | 1,000 |
| MTD ordem 10 | 0,971 | 5,59% | 10,77% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 25,07% | 26,61% | 26,35% |
| Ter | 24,81% | 26,48% | 26,20% |
| Qua | 24,92% | 26,54% | 26,28% |
| Qui | 24,75% | 26,47% | 26,19% |
| Sex | 25,14% | 26,61% | 26,35% |
| Sáb | 35,36% | 30,79% | 31,66% |
| Dom | 33,38% | 30,05% | 30,77% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,113 | 0,112 | 0,107 | 0,102 | 0,084 | 0,072 |
| Movimento, MTD ordem 1 | 0,111 | 0,011 | 0,000 | 0,001 | 0,002 | 0,001 |
| Movimento, MTD ordem 6 | 0,116 | 0,112 | 0,101 | 0,016 | 0,001 | 0,000 |
| Movimento, MTD ordem 10 | 0,117 | 0,113 | 0,101 | 0,048 | 0,000 | 0,000 |
| Símbolo, real | -0,047 | -0,034 | -0,007 | 0,001 | 0,002 | -0,001 |
| Símbolo, MTD ordem 1 | -0,047 | 0,002 | -0,000 | -0,002 | -0,001 | -0,000 |
| Símbolo, MTD ordem 6 | -0,027 | -0,026 | -0,020 | -0,003 | -0,002 | -0,000 |
| Símbolo, MTD ordem 10 | -0,023 | -0,023 | -0,015 | -0,000 | -0,001 | -0,000 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 0,00% | -37,1 | -1,08352 | -1,08352 |
| 6 | 204 | 0,02% | -216,4 | -1,06711 | -1,06711 |
| 10 | 340 | 0,04% | -360,3 | -1,06207 | -1,06207 |

## Discretização `global`, s/3

- Símbolos: treino 909.925, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 35 segmentos no treino
- Classes treino (−1/0/+1): 24,39% / 51,02% / 24,58%
- Classes teste (−1/0/+1): 16,18% / 67,96% / 15,85%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,19054 |
| Frequências do treino (iid) | 2 | -1,03241 | -0,90807 | 1.878.840,2 | 0,19054 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,55369 | -1,98743 | — | -0,88882 | -1,07936 |
| Cadeia de 1ª ordem | 6 | -1,00854 | -0,87066 | 1.835.462,6 | 0,22795 | 0,03741 |
| MTD ordem 1 | 6 | -1,00854 | -0,87066 | 1.835.462,6 | 0,22795 | 0,03741 |
| MTD ordem 2 | 7 | -0,99326 | -0,85003 | 1.807.655,2 | 0,24858 | 0,05804 |
| MTD ordem 3 | 8 | -0,98303 | -0,83667 | 1.789.051,1 | 0,26194 | 0,07140 |
| MTD ordem 4 | 9 | -0,97601 | -0,82688 | 1.776.295,6 | 0,27173 | 0,08119 |
| MTD ordem 5 | 10 | -0,97066 | -0,82040 | 1.766.573,7 | 0,27822 | 0,08768 |
| MTD ordem 6 | 11 | -0,96596 | -0,81431 | 1.758.026,8 | 0,28430 | 0,09376 |
| MTD ordem 7 | 12 | -0,96288 | -0,81024 | 1.752.436,6 | 0,28838 | 0,09784 |
| MTD ordem 8 | 13 | -0,96015 | -0,80710 | 1.747.493,2 | 0,29151 | 0,10097 |
| MTD ordem 9 | 14 | -0,95775 | -0,80406 | 1.743.127,5 | 0,29455 | 0,10401 |
| MTD ordem 10 | 15 | -0,95602 | -0,80213 | 1.740.003,1 **←** | 0,29649 | 0,10595 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 67,96% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -1,98743 | 57,09% | 49,05% (48,25% a 49,84%) | 0,594 | 0,833 | 67,96% |
| Cadeia de 1ª ordem | -0,87066 | 67,96% | 50,95% (50,16% a 51,75%) | 0,593 | 0,502 | 100,00% |
| MTD ordem 1 | -0,87066 | 67,96% | 50,95% (50,16% a 51,75%) | 0,593 | 0,502 | 100,00% |
| MTD ordem 2 | -0,85003 | 64,74% | 50,91% (50,12% a 51,70%) | 0,633 | 0,488 | 85,66% |
| MTD ordem 3 | -0,83667 | 66,83% | 50,69% (49,90% a 51,48%) | 0,654 | 0,479 | 92,72% |
| MTD ordem 4 | -0,82688 | 67,51% | 50,99% (50,20% a 51,78%) | 0,667 | 0,473 | 95,55% |
| MTD ordem 5 | -0,82040 | 66,95% | 50,67% (49,88% a 51,47%) | 0,676 | 0,468 | 92,25% |
| MTD ordem 6 | -0,81431 | 67,20% | 50,62% (49,82% a 51,41%) | 0,684 | 0,465 | 93,32% |
| MTD ordem 7 | -0,81024 | 67,57% | 50,86% (50,07% a 51,65%) | 0,688 | 0,462 | 94,80% |
| MTD ordem 8 | -0,80710 | 67,36% | 50,97% (50,18% a 51,77%) | 0,692 | 0,460 | 93,73% |
| MTD ordem 9 | -0,80406 | 67,41% | 50,57% (49,78% a 51,36%) | 0,696 | 0,458 | 93,95% |
| MTD ordem 10 | -0,80213 | 67,48% | 50,44% (49,65% a 51,23%) | 0,697 | 0,457 | 94,58% |

Direção: entre os candles de teste que se moveram (15.343), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 7.752 | 21,50% | 55,75% | 22,74% |
| +1 | 7.592 | 24,12% | 54,74% | 21,14% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 67,96% dos candles de teste ficaram parados e 32,04% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 67,96% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 100,00% | 67,96% | 100,00% | 0,00% | — | 0,00% |
| MTD ordem 6 | 93,32% | 69,84% | 95,89% | 6,68% | 58,17% | 12,14% |
| MTD ordem 10 | 94,58% | 69,61% | 96,86% | 5,42% | 60,69% | 10,27% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | — / 0,00% / 0,000 | 67,96% / 100,00% / 0,809 | — / 0,00% / 0,000 | 0,270 |
| MTD ordem 6 | 30,20% / 5,81% / 0,097 | 69,84% / 95,89% / 0,808 | 30,51% / 6,88% / 0,112 | 0,339 |
| MTD ordem 10 | 31,14% / 3,91% / 0,069 | 69,61% / 96,86% / 0,810 | 29,93% / 6,40% / 0,105 | 0,328 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 303 | 6.955 | 493 | 7.751 |
| 0 | 376 | 31.529 | 645 | 32.550 |
| +1 | 294 | 6.812 | 486 | 7.592 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,319 | 0,433 | 0,321 | 0,418 |
| 2 | 0,430 | 0,514 | 0,442 | 0,496 |
| 3 | 0,505 | 0,576 | 0,514 | 0,565 |
| 4 | 0,564 | 0,614 | 0,575 | 0,629 |
| 5 | 0,609 | 0,673 | 0,628 | 0,659 |
| 6 | 0,659 | 0,706 | 0,674 | 0,708 |
| 7 | 0,708 | 0,749 | 0,723 | 0,760 |
| 8 | 0,742 | 0,793 | 0,774 | 0,799 |
| 9 | 0,809 | 0,849 | 0,822 | 0,854 |
| 10 | 0,809 | 0,890 | 0,860 | 0,909 |
| P(0) p10 a p90 | 0,40 a 0,81 |  | 0,40 a 0,86 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -0,9081 | 0,593 | -0,8707 | 0,684 | -0,8143 | 0,697 | -0,8021 |
| 15 min | 0,500 | -0,9081 | 0,582 | -0,9057 | 0,668 | -0,8292 | 0,684 | -0,8120 |
| 1 h | 0,500 | -0,9081 | 0,578 | -0,9081 | 0,650 | -0,8733 | 0,665 | -0,8385 |
| 4 h | 0,500 | -0,9081 | 0,500 | -0,9081 | 0,601 | -0,9028 | 0,612 | -0,8938 |
| 1 dia | 0,500 | -0,9081 | 0,500 | -0,9081 | 0,480 | -0,9030 | 0,597 | -0,9003 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 24,39% / 51,02% / 24,58% | 0,213 | 0,6 | 3 |
| 2 | 0,511 | 0,489 | 0,489 | 24,34% / 51,14% / 24,53% | 0,349 | 1,5 | 7 |
| 3 | 0,348 | 0,334 | 0,318 | 24,25% / 51,31% / 24,44% | 0,443 | 2,6 | 13 |
| 4 | 0,270 | 0,257 | 0,228 | 24,16% / 51,49% / 24,35% | 0,510 | 3,8 | 19 |
| 5 | 0,225 | 0,211 | 0,181 | 24,08% / 51,66% / 24,27% | 0,561 | 5,3 | 27 |
| 6 | 0,194 | 0,179 | 0,158 | 24,01% / 51,79% / 24,20% | 0,604 | 7,1 | 35 |
| 7 | 0,172 | 0,160 | 0,121 | 23,95% / 51,91% / 24,14% | 0,637 | 8,9 | 44 |
| 8 | 0,157 | 0,144 | 0,109 | 23,90% / 52,02% / 24,08% | 0,664 | 10,9 | 55 |
| 9 | 0,144 | 0,132 | 0,099 | 23,84% / 52,12% / 24,03% | 0,688 | 13,2 | 66 |
| 10 | 0,135 | 0,123 | 0,081 | 23,80% / 52,21% / 23,99% | 0,707 | 15,4 | 77 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3578 | 0,1219 | 0,5203 |
| 0 | 0,0702 | 0,8601 | 0,0697 |
| +1 | 0,4843 | 0,1836 | 0,3321 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,135, 0,123, 0,111, 0,098, 0,094, 0,099, 0,084, 0,086, 0,089, 0,081

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 5 | 99 | 0 | 14s |
| 3 | 10 | 10 | 10 | 0,01 | 4 | 81 | 0 | 19s |
| 4 | 10 | 10 | 10 | 0,01 | 3 | 107 | 0 | 21s |
| 5 | 10 | 10 | 10 | 0,01 | 5 | 109 | 0 | 26s |
| 6 | 10 | 10 | 10 | 0,01 | 1 | 138 | 0 | 28s |
| 7 | 10 | 10 | 10 | 0,02 | 1 | 139 | 0 | 33s |
| 8 | 10 | 10 | 10 | 0,02 | 7 | 114 | 0 | 33s |
| 9 | 10 | 10 | 10 | 0,02 | 7 | 116 | 0 | 43s |
| 10 | 10 | 10 | 10 | 0,03 | 1 | 141 | 0 | 44s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 47,32% | 49,97% | 50,41% |
| 04 | 58,12% | 55,48% | 56,13% |
| 05 | 59,01% | 56,21% | 57,02% |
| 08 | 54,46% | 53,70% | 54,38% |
| 12 | 49,70% | 51,27% | 51,87% |
| 14 | 42,59% | 46,36% | 45,85% |
| 15 | 43,58% | 46,39% | 45,50% |
| 17 | 48,46% | 49,19% | 48,51% |
| 20 | 51,06% | 51,29% | 51,21% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 9,85% | 16,42% | 1,000 |
| MTD ordem 10 | 0,972 | 11,52% | 16,42% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 48,68% | 49,89% | 49,68% |
| Ter | 48,24% | 49,65% | 49,41% |
| Qua | 48,43% | 49,76% | 49,54% |
| Qui | 48,12% | 49,57% | 49,32% |
| Sex | 48,70% | 49,89% | 49,67% |
| Sáb | 61,66% | 57,72% | 58,83% |
| Dom | 59,22% | 56,37% | 57,33% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,213 | 0,210 | 0,201 | 0,190 | 0,161 | 0,149 |
| Movimento, MTD ordem 1 | 0,212 | 0,044 | 0,001 | -0,001 | 0,001 | 0,000 |
| Movimento, MTD ordem 6 | 0,218 | 0,214 | 0,201 | 0,071 | -0,001 | -0,000 |
| Movimento, MTD ordem 10 | 0,223 | 0,216 | 0,208 | 0,138 | 0,016 | -0,000 |
| Símbolo, real | -0,051 | -0,031 | -0,007 | -0,000 | 0,003 | -0,002 |
| Símbolo, MTD ordem 1 | -0,052 | 0,002 | -0,001 | -0,003 | -0,001 | -0,000 |
| Símbolo, MTD ordem 6 | -0,024 | -0,022 | -0,019 | -0,000 | -0,001 | 0,001 |
| Símbolo, MTD ordem 10 | -0,017 | -0,017 | -0,014 | 0,000 | 0,000 | 0,000 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 0,00% | -44,3 | -1,00854 | -1,00853 |
| 6 | 204 | 0,02% | -229,2 | -0,96596 | -0,96592 |
| 10 | 340 | 0,04% | -373,1 | -0,95602 | -0,95597 |

## Discretização `global`, s/4

- Símbolos: treino 909.925, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 35 segmentos no treino
- Classes treino (−1/0/+1): 29,25% / 41,35% / 29,41%
- Classes teste (−1/0/+1): 21,62% / 57,45% / 20,94%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,06925 |
| Frequências do treino (iid) | 2 | -1,08465 | -1,02936 | 1.973.897,9 | 0,06925 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,79426 | -2,40923 | — | -1,31062 | -1,37987 |
| Cadeia de 1ª ordem | 6 | -1,06658 | -0,99757 | 1.941.083,2 | 0,10104 | 0,03179 |
| MTD ordem 1 | 6 | -1,06658 | -0,99757 | 1.941.083,2 | 0,10104 | 0,03179 |
| MTD ordem 2 | 7 | -1,05446 | -0,97913 | 1.919.025,0 | 0,11948 | 0,05023 |
| MTD ordem 3 | 8 | -1,04632 | -0,96658 | 1.904.241,8 | 0,13203 | 0,06278 |
| MTD ordem 4 | 9 | -1,04052 | -0,95730 | 1.893.701,7 | 0,14132 | 0,07207 |
| MTD ordem 5 | 10 | -1,03598 | -0,95058 | 1.885.443,2 | 0,14803 | 0,07878 |
| MTD ordem 6 | 11 | -1,03191 | -0,94504 | 1.878.046,1 | 0,15357 | 0,08433 |
| MTD ordem 7 | 12 | -1,02931 | -0,94075 | 1.873.341,9 | 0,15786 | 0,08861 |
| MTD ordem 8 | 13 | -1,02695 | -0,93804 | 1.869.045,2 | 0,16057 | 0,09133 |
| MTD ordem 9 | 14 | -1,02486 | -0,93484 | 1.865.254,6 | 0,16378 | 0,09453 |
| MTD ordem 10 | 15 | -1,02326 | -0,93281 | 1.862.367,0 **←** | 0,16580 | 0,09656 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 57,45% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,40923 | 47,89% | 49,01% (48,33% a 49,70%) | 0,587 | 1,011 | 57,44% |
| Cadeia de 1ª ordem | -0,99757 | 49,01% | 51,63% (50,94% a 52,31%) | 0,587 | 0,594 | 57,44% |
| MTD ordem 1 | -0,99757 | 49,01% | 51,63% (50,94% a 52,31%) | 0,587 | 0,594 | 57,44% |
| MTD ordem 2 | -0,97913 | 54,82% | 51,39% (50,70% a 52,07%) | 0,622 | 0,580 | 77,63% |
| MTD ordem 3 | -0,96658 | 56,60% | 51,18% (50,50% a 51,87%) | 0,641 | 0,571 | 87,00% |
| MTD ordem 4 | -0,95730 | 55,47% | 51,27% (50,58% a 51,96%) | 0,653 | 0,565 | 78,37% |
| MTD ordem 5 | -0,95058 | 56,16% | 50,99% (50,30% a 51,67%) | 0,661 | 0,560 | 81,01% |
| MTD ordem 6 | -0,94504 | 57,06% | 51,05% (50,37% a 51,74%) | 0,668 | 0,556 | 85,57% |
| MTD ordem 7 | -0,94075 | 56,78% | 51,24% (50,55% a 51,92%) | 0,673 | 0,553 | 82,33% |
| MTD ordem 8 | -0,93804 | 56,97% | 51,36% (50,68% a 52,05%) | 0,675 | 0,552 | 82,90% |
| MTD ordem 9 | -0,93484 | 57,25% | 50,88% (50,20% a 51,57%) | 0,679 | 0,549 | 84,77% |
| MTD ordem 10 | -0,93281 | 57,19% | 50,91% (50,23% a 51,60%) | 0,681 | 0,548 | 84,04% |

Direção: entre os candles de teste que se moveram (20.380), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 10.354 | 25,32% | 47,72% | 26,96% |
| +1 | 10.027 | 28,25% | 47,12% | 24,62% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 57,45% dos candles de teste ficaram parados e 42,55% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 57,45% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 57,44% | 64,87% | 64,87% | 42,56% | 52,57% | 52,58% |
| MTD ordem 6 | 85,57% | 61,05% | 90,94% | 14,43% | 63,92% | 21,67% |
| MTD ordem 10 | 84,04% | 61,69% | 90,26% | 15,96% | 64,92% | 24,34% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | 28,25% / 27,36% / 0,278 | 64,87% / 64,87% / 0,649 | 26,96% / 27,83% / 0,274 | 0,400 |
| MTD ordem 6 | 33,59% / 9,64% / 0,150 | 61,05% / 90,94% / 0,731 | 33,29% / 13,07% / 0,188 | 0,356 |
| MTD ordem 10 | 34,71% / 10,78% / 0,165 | 61,69% / 90,26% / 0,733 | 32,62% / 14,40% / 0,200 | 0,366 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 1.116 | 7.810 | 1.427 | 10.353 |
| 0 | 1.125 | 24.832 | 1.556 | 27.513 |
| +1 | 974 | 7.609 | 1.444 | 10.027 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,242 | 0,352 | 0,241 | 0,327 |
| 2 | 0,336 | 0,414 | 0,339 | 0,404 |
| 3 | 0,380 | 0,446 | 0,402 | 0,456 |
| 4 | 0,454 | 0,521 | 0,455 | 0,499 |
| 5 | 0,474 | 0,552 | 0,502 | 0,558 |
| 6 | 0,546 | 0,580 | 0,548 | 0,579 |
| 7 | 0,566 | 0,617 | 0,599 | 0,639 |
| 8 | 0,640 | 0,690 | 0,650 | 0,675 |
| 9 | 0,676 | 0,740 | 0,711 | 0,755 |
| 10 | 0,743 | 0,832 | 0,786 | 0,853 |
| P(0) p10 a p90 | 0,28 a 0,74 |  | 0,30 a 0,74 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0294 | 0,587 | -0,9976 | 0,668 | -0,9450 | 0,681 | -0,9328 |
| 15 min | 0,500 | -1,0294 | 0,581 | -1,0277 | 0,656 | -0,9586 | 0,670 | -0,9417 |
| 1 h | 0,500 | -1,0294 | 0,575 | -1,0294 | 0,638 | -1,0021 | 0,653 | -0,9690 |
| 4 h | 0,500 | -1,0294 | 0,500 | -1,0294 | 0,598 | -1,0255 | 0,609 | -1,0194 |
| 1 dia | 0,500 | -1,0294 | 0,500 | -1,0294 | 0,535 | -1,0255 | 0,584 | -1,0234 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 29,25% / 41,35% / 29,41% | 0,184 | 0,6 | 3 |
| 2 | 0,511 | 0,489 | 0,489 | 29,20% / 41,44% / 29,36% | 0,308 | 1,3 | 7 |
| 3 | 0,350 | 0,335 | 0,315 | 29,13% / 41,57% / 29,29% | 0,397 | 2,3 | 11 |
| 4 | 0,272 | 0,259 | 0,227 | 29,07% / 41,70% / 29,23% | 0,462 | 3,4 | 17 |
| 5 | 0,225 | 0,212 | 0,181 | 29,01% / 41,82% / 29,17% | 0,513 | 4,6 | 23 |
| 6 | 0,193 | 0,180 | 0,158 | 28,95% / 41,94% / 29,11% | 0,557 | 6,2 | 31 |
| 7 | 0,171 | 0,160 | 0,119 | 28,90% / 42,04% / 29,06% | 0,590 | 7,7 | 38 |
| 8 | 0,155 | 0,144 | 0,108 | 28,85% / 42,13% / 29,01% | 0,619 | 9,4 | 47 |
| 9 | 0,143 | 0,132 | 0,098 | 28,81% / 42,21% / 28,97% | 0,644 | 11,3 | 56 |
| 10 | 0,133 | 0,123 | 0,082 | 28,78% / 42,28% / 28,94% | 0,665 | 13,2 | 66 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3493 | 0,1199 | 0,5308 |
| 0 | 0,0969 | 0,8064 | 0,0967 |
| +1 | 0,5056 | 0,1636 | 0,3309 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,133, 0,123, 0,109, 0,099, 0,096, 0,101, 0,083, 0,087, 0,088, 0,082

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 5 | 116 | 0 | 17s |
| 3 | 10 | 10 | 10 | 0,01 | 6 | 122 | 0 | 25s |
| 4 | 10 | 10 | 10 | 0,01 | 6 | 115 | 0 | 27s |
| 5 | 10 | 10 | 10 | 0,02 | 7 | 123 | 0 | 31s |
| 6 | 10 | 10 | 10 | 0,01 | 1 | 155 | 0 | 32s |
| 7 | 10 | 10 | 10 | 0,02 | 1 | 155 | 0 | 40s |
| 8 | 10 | 10 | 10 | 0,03 | 1 | 155 | 0 | 37s |
| 9 | 10 | 10 | 10 | 0,03 | 1 | 155 | 0 | 47s |
| 10 | 10 | 10 | 10 | 0,03 | 1 | 155 | 0 | 52s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 37,85% | 40,37% | 40,76% |
| 04 | 48,15% | 45,35% | 45,99% |
| 05 | 48,87% | 45,83% | 46,67% |
| 08 | 44,35% | 43,46% | 44,06% |
| 12 | 40,13% | 41,60% | 42,09% |
| 14 | 33,94% | 37,57% | 37,09% |
| 15 | 34,53% | 37,49% | 36,68% |
| 17 | 39,10% | 39,81% | 39,17% |
| 20 | 41,50% | 41,64% | 41,60% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,987 | 8,34% | 14,93% | 1,000 |
| MTD ordem 10 | 0,971 | 9,99% | 14,93% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 38,85% | 40,19% | 39,96% |
| Ter | 38,50% | 40,02% | 39,77% |
| Qua | 38,80% | 40,18% | 39,96% |
| Qui | 38,45% | 39,99% | 39,72% |
| Sex | 39,01% | 40,27% | 40,04% |
| Sáb | 51,93% | 47,47% | 48,62% |
| Dom | 49,50% | 46,25% | 47,25% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,184 | 0,183 | 0,176 | 0,167 | 0,140 | 0,129 |
| Movimento, MTD ordem 1 | 0,183 | 0,032 | 0,001 | -0,001 | 0,001 | 0,001 |
| Movimento, MTD ordem 6 | 0,187 | 0,183 | 0,169 | 0,051 | -0,000 | -0,001 |
| Movimento, MTD ordem 10 | 0,189 | 0,184 | 0,178 | 0,110 | 0,010 | 0,002 |
| Símbolo, real | -0,051 | -0,034 | -0,007 | 0,000 | 0,004 | -0,002 |
| Símbolo, MTD ordem 1 | -0,051 | 0,003 | 0,001 | -0,002 | -0,001 | -0,000 |
| Símbolo, MTD ordem 6 | -0,025 | -0,024 | -0,021 | -0,001 | -0,001 | 0,001 |
| Símbolo, MTD ordem 10 | -0,021 | -0,019 | -0,016 | 0,000 | 0,000 | 0,001 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 0,00% | -40,3 | -1,06658 | -1,06658 |
| 6 | 204 | 0,02% | -221,1 | -1,03191 | -1,03190 |
| 10 | 340 | 0,04% | -369,3 | -1,02326 | -1,02324 |

## Discretização `global`, s/5

- Símbolos: treino 909.925, teste 47.893 (2026-04-17T16:55 a 2026-09-30T23:55); 35 segmentos no treino
- Classes treino (−1/0/+1): 32,62% / 34,61% / 32,76%
- Classes teste (−1/0/+1): 25,59% / 49,59% / 24,82%

### 1. Baselines, MTD e HMM: log-verossimilhança e BIC

| Modelo | Parâmetros | LL/sím. treino | LL/sím. teste | BIC (treino) | Ganho s/ uniforme (teste) | Ganho s/ iid (teste) |
|---|---:|---:|---:|---:|---:|---:|
| Uniforme | 0 | -1,09861 | -1,09861 | — | 0,00000 | -0,00891 |
| Frequências do treino (iid) | 2 | -1,09824 | -1,08971 | 1.998.643,8 | 0,00891 | 0,00000 |
| Persistência (ε = 0,01) | 0 | -2,90106 | -2,65067 | — | -1,55206 | -1,56097 |
| Cadeia de 1ª ordem | 6 | -1,08438 | -1,06258 | 1.973.472,5 | 0,03603 | 0,02713 |
| MTD ordem 1 | 6 | -1,08438 | -1,06258 | 1.973.472,5 | 0,03603 | 0,02713 |
| MTD ordem 2 | 7 | -1,07459 | -1,04568 | 1.955.661,1 | 0,05294 | 0,04403 |
| MTD ordem 3 | 8 | -1,06796 | -1,03517 | 1.943.610,2 | 0,06344 | 0,05454 |
| MTD ordem 4 | 9 | -1,06307 | -1,02698 | 1.934.733,0 | 0,07163 | 0,06272 |
| MTD ordem 5 | 10 | -1,05917 | -1,02100 | 1.927.644,5 | 0,07761 | 0,06871 |
| MTD ordem 6 | 11 | -1,05577 | -1,01585 | 1.921.481,7 | 0,08276 | 0,07386 |
| MTD ordem 7 | 12 | -1,05354 | -1,01140 | 1.917.423,4 | 0,08721 | 0,07830 |
| MTD ordem 8 | 13 | -1,05152 | -1,00864 | 1.913.759,4 | 0,08997 | 0,08106 |
| MTD ordem 9 | 14 | -1,04972 | -1,00570 | 1.910.510,2 | 0,09291 | 0,08400 |
| MTD ordem 10 | 15 | -1,04826 | -1,00330 | 1.907.865,5 **←** | 0,09531 | 0,08640 |

### 2. Previsão de um passo no teste

| Modelo | LL/sím. | Acurácia | Direção (IC 95%) | AUC movimento | Brier | Prev. 0 |
|---|---:|---:|---|---:|---:|---:|
| Sempre 0 | — | 49,59% | — | — | — | 100,00% |
| Persistência (ε = 0,01) | -2,65067 | 42,63% | 48,66% (48,03% a 49,29%) | 0,582 | 1,114 | 49,59% |
| Cadeia de 1ª ordem | -1,06258 | 44,21% | 51,79% (51,16% a 52,42%) | 0,580 | 0,641 | 49,59% |
| MTD ordem 1 | -1,06258 | 44,21% | 51,79% (51,16% a 52,42%) | 0,580 | 0,641 | 49,59% |
| MTD ordem 2 | -1,04568 | 48,35% | 51,42% (50,79% a 52,05%) | 0,615 | 0,628 | 70,51% |
| MTD ordem 3 | -1,03517 | 47,99% | 51,37% (50,74% a 52,00%) | 0,632 | 0,620 | 66,05% |
| MTD ordem 4 | -1,02698 | 47,99% | 51,73% (51,10% a 52,36%) | 0,642 | 0,614 | 62,79% |
| MTD ordem 5 | -1,02100 | 49,31% | 51,79% (51,16% a 52,42%) | 0,649 | 0,610 | 71,73% |
| MTD ordem 6 | -1,01585 | 49,13% | 51,48% (50,85% a 52,11%) | 0,655 | 0,607 | 67,32% |
| MTD ordem 7 | -1,01140 | 49,51% | 51,63% (51,00% a 52,26%) | 0,660 | 0,604 | 67,81% |
| MTD ordem 8 | -1,00864 | 50,00% | 51,78% (51,15% a 52,41%) | 0,663 | 0,602 | 71,45% |
| MTD ordem 9 | -1,00570 | 50,07% | 51,55% (50,92% a 52,18%) | 0,666 | 0,600 | 70,23% |
| MTD ordem 10 | -1,00330 | 50,30% | 51,71% (51,08% a 52,34%) | 0,669 | 0,598 | 70,62% |

Direção: entre os candles de teste que se moveram (24.142), fração em que o modelo deu mais probabilidade ao sinal correto (regra que ignora o 0; empate fica em −1). Erro padrão de cerca de 0,3 p.p. AUC movimento: separar "sai da faixa ±s/3" de "fica em 0" (0,5 = acaso). Brier: uniforme = 0,667.

Depois de um símbolo, o que vem no teste:

| Depois de | n | próximo −1 | próximo 0 | próximo +1 |
|---|---:|---:|---:|---:|
| -1 | 12.256 | 28,02% | 41,93% | 30,05% |
| +1 | 11.887 | 31,61% | 41,05% | 27,34% |

#### Zero e movimento (teste, decisão argmax)

Duas perguntas separadas: "vai ficar parado?" (símbolo 0) e "vai se mexer?" (−1 ou +1). Base: 49,59% dos candles de teste ficaram parados e 50,41% se moveram. O acerto de direção (acima) é a terceira pergunta: dado que se moveu, para que lado.

| Modelo | Previu 0 | Precisão do 0 | Recall do 0 | Previu movimento | Precisão do movimento | Recall do movimento |
|---|---:|---:|---:|---:|---:|---:|
| Sempre 0 | 100,00% | 49,59% | 100,00% | 0,00% | — | 0,00% |
| Cadeia de 1ª ordem | 49,59% | 57,82% | 57,82% | 50,41% | 58,50% | 58,50% |
| MTD ordem 6 | 67,32% | 56,67% | 76,93% | 32,68% | 64,99% | 42,13% |
| MTD ordem 10 | 70,62% | 56,63% | 80,63% | 29,38% | 67,31% | 39,24% |

#### Métricas por classe (teste, decisão argmax)

Cada célula: precisão / recall / F1. A precisão de ±1 mistura a volatilidade (o modelo prevê movimento quando ele é mais provável) com a direção; só o acerto de direção separa as duas.

| Modelo | −1 | 0 | +1 | F1 médio |
|---|---|---|---|---:|
| Cadeia de 1ª ordem | 31,61% / 30,66% / 0,311 | 57,82% / 57,82% / 0,578 | 30,05% / 30,98% / 0,305 | 0,398 |
| MTD ordem 6 | 34,15% / 18,86% / 0,243 | 56,67% / 76,93% / 0,653 | 33,20% / 24,81% / 0,284 | 0,393 |
| MTD ordem 10 | 35,96% / 18,38% / 0,243 | 56,63% / 80,63% / 0,665 | 34,40% / 22,60% / 0,273 | 0,394 |

Matriz de confusão, MTD ordem 10 (linhas = real, colunas = previsto):

| Real \ Previsto | −1 | 0 | +1 | Total |
|---|---:|---:|---:|---:|
| −1 | 2.252 | 7.408 | 2.595 | 12.255 |
| 0 | 2.071 | 19.151 | 2.529 | 23.751 |
| +1 | 1.939 | 7.261 | 2.687 | 11.887 |

### 3. Calibração da P(0) prevista (teste, decis)

| Decil | MTD ordem 6 prev. | MTD ordem 6 obs. | MTD ordem 10 prev. | MTD ordem 10 obs. |
|---:|---:|---:|---:|---:|
| 1 | 0,201 | 0,303 | 0,197 | 0,281 |
| 2 | 0,259 | 0,341 | 0,274 | 0,334 |
| 3 | 0,330 | 0,392 | 0,326 | 0,376 |
| 4 | 0,349 | 0,429 | 0,373 | 0,428 |
| 5 | 0,413 | 0,464 | 0,415 | 0,467 |
| 6 | 0,432 | 0,502 | 0,456 | 0,497 |
| 7 | 0,490 | 0,523 | 0,502 | 0,538 |
| 8 | 0,520 | 0,580 | 0,552 | 0,582 |
| 9 | 0,595 | 0,658 | 0,614 | 0,663 |
| 10 | 0,664 | 0,768 | 0,712 | 0,796 |
| P(0) p10 a p90 | 0,25 a 0,61 |  | 0,25 a 0,65 | 

### 4. Horizonte (teste; informação até t, alvo em t + h)

Mesmos alvos (os 47.893 símbolos do teste) em todos os horizontes.

| Horizonte | Frequências do treino (iid) AUC | Frequências do treino (iid) LL | MTD ordem 1 AUC | MTD ordem 1 LL | MTD ordem 6 AUC | MTD ordem 6 LL | MTD ordem 10 AUC | MTD ordem 10 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0,500 | -1,0897 | 0,580 | -1,0626 | 0,655 | -1,0158 | 0,669 | -1,0033 |
| 15 min | 0,500 | -1,0897 | 0,574 | -1,0887 | 0,642 | -1,0295 | 0,657 | -1,0123 |
| 1 h | 0,500 | -1,0897 | 0,570 | -1,0897 | 0,626 | -1,0693 | 0,641 | -1,0394 |
| 4 h | 0,500 | -1,0897 | 0,471 | -1,0897 | 0,591 | -1,0870 | 0,601 | -1,0831 |
| 1 dia | 0,500 | -1,0897 | 0,479 | -1,0897 | 0,503 | -1,0870 | 0,577 | -1,0854 |

### 5. Estrutura do MTD

| Ordem | Lag 1 | Lag 2 | Último lag | π de Q (−1/0/+1) | 2º autovalor de Q | Relaxação (passos) | Relaxação (min) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1,000 | — | 1,000 | 32,62% / 34,61% / 32,76% | 0,160 | 0,5 | 3 |
| 2 | 0,506 | 0,494 | 0,494 | 32,59% / 34,67% / 32,73% | 0,275 | 1,2 | 6 |
| 3 | 0,349 | 0,337 | 0,314 | 32,55% / 34,77% / 32,69% | 0,358 | 2,1 | 10 |
| 4 | 0,269 | 0,261 | 0,228 | 32,50% / 34,86% / 32,64% | 0,421 | 3,0 | 15 |
| 5 | 0,223 | 0,214 | 0,182 | 32,46% / 34,94% / 32,60% | 0,472 | 4,2 | 21 |
| 6 | 0,191 | 0,182 | 0,155 | 32,41% / 35,03% / 32,56% | 0,515 | 5,5 | 27 |
| 7 | 0,169 | 0,162 | 0,118 | 32,38% / 35,10% / 32,52% | 0,549 | 6,8 | 34 |
| 8 | 0,154 | 0,145 | 0,106 | 32,34% / 35,17% / 32,49% | 0,577 | 8,3 | 41 |
| 9 | 0,141 | 0,133 | 0,096 | 32,32% / 35,22% / 32,46% | 0,603 | 9,9 | 49 |
| 10 | 0,131 | 0,123 | 0,083 | 32,29% / 35,28% / 32,43% | 0,625 | 11,6 | 58 |

`Q` da ordem 10 (linha = símbolo no lag, coluna = próximo símbolo):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,3461 | 0,1148 | 0,5391 |
| 0 | 0,1215 | 0,7572 | 0,1212 |
| +1 | 0,5188 | 0,1498 | 0,3314 |

Pesos por lag (ordem 10, lag 1 primeiro): 0,131, 0,123, 0,110, 0,100, 0,097, 0,099, 0,083, 0,086, 0,087, 0,083

### 6. Estabilidade dos reinícios

| Ordem | Reinícios | Na melhor solução (±1 nat) | Convergiram | Pior a menos de (nats) | Melhor reinício | Iterações (melhor) | Avisos de LL caindo | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 | 0,00 | 0 | 1 | 0 | 0s |
| 2 | 10 | 10 | 10 | 0,01 | 2 | 115 | 0 | 20s |
| 3 | 10 | 10 | 10 | 0,01 | 6 | 142 | 0 | 29s |
| 4 | 10 | 10 | 10 | 0,01 | 3 | 143 | 0 | 30s |
| 5 | 10 | 10 | 10 | 0,01 | 1 | 178 | 0 | 36s |
| 6 | 10 | 10 | 10 | 0,01 | 1 | 175 | 0 | 38s |
| 7 | 10 | 10 | 10 | 0,03 | 1 | 174 | 0 | 48s |
| 8 | 10 | 10 | 10 | 0,03 | 1 | 172 | 0 | 47s |
| 9 | 10 | 10 | 10 | 0,03 | 1 | 171 | 0 | 57s |
| 10 | 10 | 10 | 10 | 0,04 | 1 | 170 | 0 | 61s |

### 7. Sazonalidade: P(0) prevista contra observada

| Hora UTC | observada | MTD ordem 6 | MTD ordem 10 |
|---:|---:|---:|---:|
| 00 | 31,52% | 33,88% | 34,19% |
| 04 | 41,03% | 38,05% | 38,67% |
| 05 | 41,44% | 38,37% | 39,17% |
| 08 | 37,43% | 36,41% | 36,95% |
| 12 | 33,77% | 34,88% | 35,30% |
| 14 | 27,93% | 31,50% | 31,06% |
| 15 | 28,67% | 31,45% | 30,72% |
| 17 | 32,70% | 33,36% | 32,77% |
| 20 | 34,83% | 34,91% | 34,88% |

Ajuste às 24 horas e aos 7 dias da semana (correlação entre a P(0) média prevista e a frequência observada; amplitude = máximo − mínimo):

| Modelo | Correlação (hora) | Amplitude prevista (hora) | Amplitude observada (hora) | Correlação (dia) |
|---|---:|---:|---:|---:|
| MTD ordem 6 | 0,986 | 6,92% | 13,51% | 1,000 |
| MTD ordem 10 | 0,970 | 8,45% | 13,51% | 1,000 |

| Dia | observada | MTD ordem 6 | MTD ordem 10 |
|---|---:|---:|---:|
| Seg | 32,22% | 33,57% | 33,34% |
| Ter | 31,96% | 33,44% | 33,20% |
| Qua | 32,15% | 33,55% | 33,33% |
| Qui | 31,85% | 33,39% | 33,14% |
| Sex | 32,41% | 33,65% | 33,42% |
| Sáb | 44,60% | 39,93% | 41,04% |
| Dom | 42,33% | 38,87% | 39,83% |

### 8. Adequação: autocorrelação real contra simulada (treino)

| Série | 5 min | 10 min | 30 min | 1 h | 4 h | 1 dia |
|---|---:|---:|---:|---:|---:|---:|
| Movimento (≠ 0), real | 0,161 | 0,161 | 0,155 | 0,146 | 0,124 | 0,114 |
| Movimento, MTD ordem 1 | 0,159 | 0,025 | 0,000 | -0,000 | 0,001 | 0,000 |
| Movimento, MTD ordem 6 | 0,162 | 0,161 | 0,145 | 0,039 | 0,001 | -0,001 |
| Movimento, MTD ordem 10 | 0,165 | 0,161 | 0,153 | 0,089 | 0,007 | 0,002 |
| Símbolo, real | -0,050 | -0,035 | -0,008 | 0,001 | 0,002 | -0,002 |
| Símbolo, MTD ordem 1 | -0,050 | 0,003 | -0,000 | -0,002 | -0,001 | 0,000 |
| Símbolo, MTD ordem 6 | -0,026 | -0,026 | -0,021 | -0,001 | -0,002 | 0,001 |
| Símbolo, MTD ordem 10 | -0,022 | -0,020 | -0,017 | 0,000 | 0,000 | -0,000 |

### 9. Lacunas (treino)

Transições cujo contexto atravessa uma lacuna entram no ajuste como normais. Abaixo, o que elas valem na LL do modelo ajustado:

| Ordem | Transições afetadas | Fração | LL nelas | LL/sím. com todas | LL/sím. sem elas |
|---:|---:|---:|---:|---:|---:|
| 1 | 34 | 0,00% | -38,4 | -1,08438 | -1,08438 |
| 6 | 204 | 0,02% | -220,1 | -1,05577 | -1,05577 |
| 10 | 340 | 0,04% | -367,4 | -1,04826 | -1,04825 |
