# Relatório completo do MTD (tabelas)

Gerado em 2026-11-23 02:41 (UTC) por `src/experiments/mtd_analysis.py`. Mesmas métricas de `relatorios/analise_hmm_2026-10-01_completa.md`. A leitura dos números está em `relatorios/analise_mtd_2026-10-01_completa.md`.

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
| global | — | 3 | 51,02% | 67,96% | 10 | -0,80213 | 0,10595 | 0,697 | 50,44% (49,65% a 51,23%) | 50,52% |

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
