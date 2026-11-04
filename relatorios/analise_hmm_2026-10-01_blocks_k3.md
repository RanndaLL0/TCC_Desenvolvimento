# Análise do treinamento do HMM — blocos, s/3, 3 estados

Análise de `relatorios/hmm_2026-10-01_blocks_k3.json` (gerado em 2026-10-22 19:24 UTC). Tudo aqui sai dos números desse arquivo. As grandezas derivadas, como a distribuição estacionária, as durações e as entropias, foram calculadas a partir das matrizes `pi`, `A` e `B` que ele traz. Nenhum outro dado ou experimento foi usado.

## Resumo

- O ajuste convergiu (265 iterações, melhor reinício 9) com log-verossimilhança de treino **-928.914,9**, ou **-1,0257 nats por símbolo**.
- Os três estados são **regimes de volatilidade**: calmo, intermediário e agitado. Nenhum deles distingue alta de queda. Em todos, `P(-1)` e `P(+1)` diferem em menos de 0,005.
- A matriz de transição é muito persistente (diagonal ≥ 0,986) e tem estrutura **em escada**. O regime calmo e o agitado não se comunicam diretamente; toda troca entre eles passa pelo intermediário.
- O teste (-1,0154 por símbolo) não ficou pior que o treino, então não há sinal de sobreajuste. Com 14 parâmetros para 905 mil símbolos, isso era esperado.
- **Ponto de atenção:** 7 dos 10 reinícios pararam em `max_iter` (1000) cerca de 7.894 nats abaixo do melhor. Só 3 chegaram à solução boa.

## 1. Dados do ajuste

| Item | Valor |
|---|---|
| Discretização | blocos de 1 mês, limiar s/3, treino 95% |
| Símbolos de treino | 905.654 (linhas 0 a 909.959) |
| Símbolos de teste | 47.893 (cerca de 166 dias de candles de 5 min) |
| Parâmetros livres | 14 = 2 (`pi`) + 6 (`A`) + 6 (`B`) |
| Tempo de ajuste | 283 s para 10 reinícios e 7.728 iterações (cerca de 37 ms por iteração) |

O treino cobre 909.960 linhas, mas só 905.654 têm símbolo. As 4.306 restantes são o primeiro bloco, que só serve de histórico no modo `blocks`, mais o primeiro candle e os retornos que atravessam lacunas. No teste, todas as 47.893 linhas têm símbolo.

## 2. Qualidade do ajuste

| | Treino | Teste |
|---|---:|---:|
| Log-verossimilhança total | -928.914,91 | -48.632,51 |
| Por símbolo (nats) | -1,0257 | -1,0154 |
| Perplexidade (`exp(-ll/T)`) | 2,789 | 2,761 |
| Ganho sobre o uniforme (`ln 3` = 1,0986) | 0,0729 | 0,0832 |

- **Perplexidade.** Um modelo que não sabe nada sobre 3 símbolos tem perplexidade 3. O HMM fica em cerca de 2,78, ou seja, reduz a incerteza a algo como "2,8 símbolos igualmente prováveis". É um ganho real, mas modesto. O retorno de 5 minutos tem pouca estrutura previsível, e boa parte do ganho pode vir só de o símbolo 0 ser mais frequente. O relatório não traz um baseline que separe essas duas fontes (ver seção 7).
- **Teste melhor que o treino.** A diferença é de 0,0102 nats por símbolo. Isso descarta sobreajuste, mas não quer dizer que o modelo generalize "melhor do que aprendeu". O teste é um único trecho de cerca de 5,5 meses no fim da série, e a log-verossimilhança depende de quão previsível foi aquele período. O relatório não tem informação para explicar a diferença.
- **AIC e BIC.** A penalidade do BIC é `14 · ln(905.654)` = 192,0, e a do AIC é 28. Os dois somem diante de uma log-verossimilhança da ordem de 10⁶. Sozinhos, os valores absolutos (AIC 1.857.857,8; BIC 1.858.021,8) não dizem nada: só servem para comparar com ajustes de outro número de estados **na mesma sequência**, e este relatório tem apenas `n_states = 3`.

## 3. Os estados: regimes de volatilidade

Matriz de emissão `B` (linhas: estados; colunas: símbolos):

| Estado | P(-1) | P(0) | P(+1) | P(+1) − P(-1) | Entropia (nats) | Leitura |
|---|---:|---:|---:|---:|---:|---|
| 2 | 0,114 | **0,767** | 0,119 | +0,005 | 0,704 | **calmo** |
| 0 | 0,267 | 0,462 | 0,271 | +0,004 | 1,063 | **intermediário** |
| 1 | 0,395 | 0,215 | 0,390 | −0,004 | 1,065 | **agitado** |

- O que distingue os estados é **a chance de o candle sair da faixa ±s/3**, ou seja, a magnitude do movimento. Essa chance vai de 23% no calmo a 79% no agitado.
- **Nenhum estado carrega direção.** Em todos, `P(-1)` e `P(+1)` são praticamente iguais. Por isso, o modelo não tem como antecipar se o próximo movimento será de alta ou de queda, apenas *se* haverá movimento. Para o TCC isso é central. Se a métrica for acertar o símbolo, as previsões do HMM tendem a escolher o `0` nos regimes calmo e intermediário. No agitado, a escolha entre `-1` e `+1` é decidida por uma diferença de 0,004, que é ruído de estimação e não sinal.
- O regime calmo é o único com emissão concentrada (entropia 0,70, contra o máximo de 1,10). Os outros dois ficam perto do uniforme, e é aí que o modelo menos ajuda a prever o símbolo.
- Os rótulos 0, 1 e 2 são arbitrários e podem trocar de lugar em outro ajuste ou outra semente (troca de rótulos do EM). Para comparar ajustes, ordene os estados por `P(0)`.

## 4. Dinâmica: matriz de transição

| de \ para | 0 (interm.) | 1 (agitado) | 2 (calmo) |
|---|---:|---:|---:|
| 0 (interm.) | 0,98963 | 0,00586 | 0,00451 |
| 1 (agitado) | 0,00801 | 0,99199 | **5,5·10⁻¹⁶** |
| 2 (calmo) | 0,01365 | **4,4·10⁻⁸** | 0,98635 |

**Estrutura em escada: calmo ↔ intermediário ↔ agitado.** As transições diretas entre calmo e agitado são numericamente zero. O valor de 5,5·10⁻¹⁶ é só o efeito da pseudocontagem `piso = 1e-10` sobre uma contagem esperada nula. O modelo aprendeu sozinho que a volatilidade sobe e desce de forma gradual, sem pular do calmo para o agitado em 5 minutos. É uma estrutura interpretável e um bom resultado para o texto do TCC.

Durações dos regimes (permanência geométrica, `1/(1 − a_ii)` passos de 5 min):

| Estado | Duração esperada | Mediana | Saída preferida |
|---|---:|---:|---|
| agitado | 124,8 passos (**10,4 h**) | 7,2 h | só para o intermediário |
| intermediário | 96,5 passos (**8,0 h**) | 5,5 h | agitado (56%) ou calmo (44%) |
| calmo | 73,3 passos (**6,1 h**) | 4,2 h | só para o intermediário |

- Os regimes duram horas, não minutos. Isso é coerente com a aglomeração de volatilidade dos retornos de alta frequência. Também mostra que o HMM capta uma dependência de longo alcance que uma cadeia de Markov de ordem 1 sobre os símbolos não captaria.
- Os autovalores de `A` são 1; 0,9896; 0,9784. Os tempos de relaxação correspondentes, cerca de 8,0 h e 3,9 h, indicam em quanto tempo o modelo "esquece" o regime atual. Para prever além de ~1 dia, a distribuição do modelo já é praticamente a estacionária.
- A permanência no HMM é sempre geométrica: a duração mais provável de qualquer regime é 1 passo. Se a duração real dos regimes tiver outra forma, o modelo não a reproduz. Isso é uma limitação conhecida do HMM, e não algo que este relatório permita medir.

**Distribuição estacionária** (fração de tempo a longo prazo em cada regime): intermediário **48,5%**, agitado **35,5%**, calmo **16,0%**. A distribuição de símbolos implicada pelo modelo é `(-1, 0, +1)` = (28,78%, 42,33%, 28,89%): simétrica e com o `0` como símbolo mais frequente, mas não majoritário.

## 5. Distribuição inicial `pi`

`pi = (2·10⁻¹⁵¹, 4·10⁻²⁰⁰, 1,0)`: o modelo diz que a série começa no regime calmo com certeza. Não é um achado, é um artefato. Com uma única sequência, `pi` é estimado a partir de **um único instante** (a posterior em `t = 0`), e o EM o empurra para um vértice. Consequências:

- `pi` não deve ser interpretado nem reportado como resultado.
- O `ll_test` foi calculado começando o teste do zero com esse `pi`, ou seja, supondo que o teste começa no regime calmo. O efeito se limita às primeiras horas do teste, porque a cadeia se mistura em cerca de 8 h (seção 4). Mesmo assim, o certo é começar o teste pela distribuição estacionária, ou continuar o filtro a partir do fim do treino.

## 6. Convergência e reinícios

| Grupo | Reinícios | Log-verossimilhança | Iterações | Convergiu |
|---|---|---|---:|---|
| Solução boa | 9, 1, 0 | -928.914,9 a -928.915,3 | 221–265 | sim |
| Solução ruim | 2, 3, 4, 5, 6, 7, 8 | -936.808,9 a -936.837,7 | 1000 (limite) | não |

- **Os três reinícios bons chegaram ao mesmo ponto.** A diferença entre eles (0,41 nats) é da ordem do critério de parada, já que a tolerância relativa de `1e-8` equivale a cerca de 0,009 nats por iteração. Os parâmetros do reinício 9 são, portanto, um resultado estável.
- **Os sete ruins estão todos presos na mesma região.** Ficam a cerca de **7.894 nats** do melhor (0,0087 por símbolo) e dentro de uma faixa de apenas 29 nats entre si. Como nenhum convergiu, o relatório não diz se é outro ótimo local ou uma região plana que o EM atravessaria com mais iterações. Uma hipótese comum nesse padrão é uma solução em que dois estados ficam quase iguais, um modelo de 2 estados "disfarçado". Não dá para confirmar só com este arquivo, porque ele não guarda os parâmetros nem o histórico dos reinícios descartados.
- **7.894 nats não é uma diferença pequena.** É 41 vezes a penalidade total do BIC deste modelo. Escolher um desses reinícios levaria a conclusões bem diferentes sobre os regimes.
- **Risco de reprodutibilidade.** Só 30% dos reinícios encontraram a solução boa. Com 10 reinícios, a chance de nenhum encontrá-la seria de cerca de 0,7¹⁰ ≈ 2,8%, supondo que os reinícios sejam independentes com essa taxa. Com 3 reinícios, seria de cerca de 34%. Reduzir `--restarts` para acelerar testes pode facilmente produzir um modelo pior sem nenhum aviso.
- `e_step_blocks` está `null`, então a divisão do passo E seguiu o número de threads da máquina. Em outra máquina, os números podem diferir nas últimas casas decimais (ver `_acumular_paralelo`).

## 7. O que este relatório não permite concluir

- **Se o HMM vale a pena.** Não há baseline no arquivo: uniforme, frequências do treino, cadeia de Markov de ordem 1 ou mais. O ganho de 0,073 nats sobre o uniforme mistura o efeito de "o 0 é mais comum" com o efeito da dinâmica de regimes. Só a comparação com baselines medidos na mesma sequência separa os dois.
- **Qual o número de estados adequado.** Há um único `n_states`, então AIC e BIC não têm com o que ser comparados.
- **Se as lacunas afetaram o ajuste.** Os 35 segmentos foram concatenados numa única sequência (pendência D6). Pela escala (34 junções em 905 mil símbolos) o efeito deve ser desprezível, mas o relatório não mede isso.
- **Como o modelo se sai em `global` ou em outros `k`.** Só a configuração `blocks`, s/3 foi treinada.

## 8. Recomendações

1. **Ordenar os estados por `P(0)`** ao salvar o relatório, para os rótulos terem significado fixo (calmo, intermediário, agitado) entre execuções.
2. **Aumentar `--restarts`** (por exemplo, 20 a 30) ou o `--max-iter`, e registrar no relatório quantos reinícios chegaram à melhor solução. Guardar também os parâmetros do melhor reinício *não convergido* ajudaria a diagnosticar a segunda solução.
3. **Treinar `--states 2 3 4 5`** na mesma configuração para que o BIC sirva de critério de escolha.
4. **Incluir baselines no relatório** (uniforme, frequências do treino, Markov de ordem 1), com a mesma métrica de log-verossimilhança por símbolo no teste.
5. **Calcular o `ll_test` a partir da distribuição estacionária** (ou do estado filtrado no fim do treino) em vez do `pi` degenerado.
6. **Fixar `--blocks`** nos experimentos que vão para o texto do TCC, para que os números sejam reproduzíveis em qualquer máquina.
7. **Na redação:** apresentar o HMM como um modelo de **regimes de volatilidade**, e não de direção. Os resultados mostram que ele indica *quando* o mercado deve se mexer, mas não *para onde*.
