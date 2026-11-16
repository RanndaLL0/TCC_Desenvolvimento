# Análise do MTD: blocos, s/3, ordens 1 a 10

Análise de `relatorios/mtd_2026-10-01.json` (tabelas completas em `relatorios/mtd_2026-10-01.md`, gerado por `src/experiments/mtd_report.py`). Os números do HMM vêm de `relatorios/analise_hmm_2026-10-01_blocks_k3.md` e **não foram reexecutados**. Este documento é análise do sistema, não texto da monografia. Os itens marcados como **hipótese** não foram testados.

## Resumo

- O MTD **aprende estrutura real e fora da amostra**: a LL por símbolo no teste melhora a cada ordem (−1,0600 na ordem 1, −1,0321 na 6, −1,0236 na 10) e o ganho sobre o uniforme vai de 0,039 a 0,075 nats.
- Essa estrutura é de **magnitude, não de direção**. A matriz `Q` diz que "movimento puxa movimento" e "calma puxa calma"; o acerto de direção (−1 contra +1) fica em 51,1% a 51,5% no teste, praticamente igual à melhor constante (50,9%).
- **O BIC não encontra mínimo.** Em todas as seis discretizações o melhor BIC está na ordem 10, a maior testada. Dentro do escopo (ordens até 6) ele escolheria 6 por construção, e isso não diz nada sobre a ordem "verdadeira".
- **O HMM do parceiro ganha do MTD com menos parâmetros** (teste −1,0154 com 14 parâmetros, contra −1,0236 com 15). As duas famílias convergem na mesma leitura: a estrutura da série é aglomeração de volatilidade.
- O ajuste é estável: os 5 reinícios da ordem 10 chegam à mesma LL (diferença de 0,04 nats em 935 mil) e nenhum passo do EM fez a LL cair.

## 1. O que o MTD capturou

`Q` e `lam` da ordem 10 (`blocks` s/3, treino):

| de \ para | −1 | 0 | +1 |
|---|---:|---:|---:|
| −1 | 0,345 | 0,137 | 0,518 |
| 0 | 0,107 | **0,782** | 0,110 |
| +1 | 0,492 | 0,189 | 0,320 |

Pesos por lag: 0,134, 0,121, 0,111, 0,099, 0,097, 0,100, 0,083, 0,085, 0,089, 0,083.

- **Magnitude.** Depois de um 0, a chance de outro 0 é 78%. Depois de ±1, é de 14% a 19%. É a assinatura de aglomeração de volatilidade, a mesma que os estados do HMM mostram.
- **Memória longa.** Os pesos decaem devagar (de 0,134 no lag 1 para 0,083 no lag 10) e a LL ainda melhora de uma ordem para a seguinte: da 9 para a 10, o BIC cai 2.526 contra uma penalidade de 13,7. O MTD está usando uma janela de ordem 10 (50 minutos) para estimar algo que o HMM guarda em um estado oculto que dura de 6 a 10 horas.
- **Direção.** Há um sinal de reversão em `Q`: depois de −1, +1 (0,518) é bem mais provável que −1 (0,345), e o inverso vale depois de +1. Ele aparece também na ordem 1, mas com a metade da força (condicionado a haver movimento, 53,7% contra 60% na ordem 10).
- **Hipótese (limitação estrutural do MTD):** como `Q` é única e compartilhada entre os lags, o modelo não consegue representar "o lag 1 inverte a direção e os lags 2 a 10 só informam volatilidade". A reversão e a aglomeração ficam misturadas na mesma matriz. Uma variante com uma `Q` por lag (chamada MTDg, descrita por Berchtold e Raftery, 2002; a conferir antes de citar) separaria os dois efeitos. Não foi testado.

## 2. Direção: sem vantagem sobre uma constante

Só as posições em que o real é −1 ou +1 (26.903 no teste). Regra de decisão: maior entre −1 e +1.

| | Acerto |
|---|---:|
| Sempre `+1` (majoritário do treino) | 49,14% |
| Sempre `−1` (melhor constante **no teste**) | 50,86% |
| MTD, ordens 1 a 10, teste | 51,10% a 51,51% |
| MTD, ordens 1 a 10, treino | 52,34% a 52,83% |

- O erro padrão do acerto é cerca de 0,3 ponto percentual. A diferença entre o MTD e a melhor constante (de 0,2 a 0,7 ponto) está dentro de 1 a 2 erros padrão, e a ordem com o melhor valor foi escolhida olhando o teste. **Não há evidência de vantagem direcional.**
- O acerto **não cresce com a ordem** (51,1% a 51,5%, sem tendência). A ordem melhora a LL, que é magnitude, e não melhora a direção.
- No treino o acerto é 52,6%, no teste 51,2%. Com 15 parâmetros e 905 mil símbolos não é sobreajuste; é o período do teste (cerca de 166 dias, fim da série) com menos reversão que a média do treino. Um único trecho de teste não permite dizer se isso é regime ou acaso.
- Em `blocks` s/3 a regra "ignorar o 0" ainda importa: o `argmax` puro prevê 0 em 64% a 76% das posições (100% na ordem 1).
- Acerto de direção sozinho não decide aposta: para o Polymarket, o retorno depende do preço pago (payoff `S·(1−p)/p`), então 51% pode ser nada depois de custos. Ver a extensão Polymarket.

## 3. Ordem e BIC

| Ordem | Parâmetros | BIC | LL/símbolo (teste) |
|---:|---:|---:|---:|
| 1 | 6 | 1.934.137,9 | −1,05997 |
| 3 | 8 | 1.905.688,5 | −1,04405 |
| 6 | 11 | 1.884.217,2 | −1,03209 |
| 10 | 15 | 1.870.878,8 | −1,02360 |

- O BIC cai em toda ordem e a LL do teste acompanha, então a seleção por BIC está **na borda da busca**. Para o TCC, "escolher a ordem por BIC" entre 2 e 6 sempre devolverá 6; o resultado honesto é "o BIC pede mais lags do que o escopo cobre".
- A penalidade `p·ln n` (≤ 206) é desprezível diante de ganhos de milhares de nats. O BIC aqui quase não penaliza; a ordem só é limitada pelo custo do ajuste.
- Tempo: de 10 a 23 s por ordem (5 reinícios na primeira configuração, 2 nas demais). Ordens muito maiores são viáveis, mas o ganho por lag é decrescente (da 9 para a 10, 0,0018 nats por símbolo).

## 4. Comparação com o HMM do parceiro (mesmos dados)

| | Parâmetros | LL/símbolo (teste) | Ganho s/ uniforme | Ganho s/ ordem 0 |
|---|---:|---:|---:|---:|
| Ordem 0 (frequências do treino) | 2 | −1,0753 | 0,0233 | 0 |
| Cadeia de 1ª ordem | 6 | −1,0600 | 0,0386 | 0,0154 |
| MTD ordem 6 | 11 | −1,0321 | 0,0665 | 0,0432 |
| MTD ordem 10 | 15 | −1,0236 | 0,0750 | 0,0518 |
| HMM, 3 estados | 14 | −1,0154 | 0,0832 | 0,0599 |

- O HMM fica 0,008 nats por símbolo à frente do MTD de ordem 10 (cerca de 390 nats no teste), com um parâmetro a menos. Em BIC de treino (1.858.021,8 contra 1.870.878,8), a diferença é de 12.857 a favor do HMM.
- **Ressalvas de comparabilidade** (detalhe no item 11 de `docs/mtd-notas-de-fluxo.md`): o HMM começa o teste do zero (sem contexto do treino) e soma o primeiro símbolo; o MTD usa o contexto. O número exato não é comparável, a ordem de grandeza sim. A LL do HMM de treino também soma `T` termos, contra `N − 10` do MTD.
- A leitura é a mesma nos dois: o ganho é de volatilidade. O HMM a representa com 3 estados persistentes, o MTD com uma janela de lags.

## 5. Cuidado com a discretização `global`

No modo `global` o teste tem 68% de zeros (treino 51%), porque a volatilidade caiu. Só a tabela de frequências do treino já ganha 0,191 nats sobre o uniforme no teste, e o MTD de ordem 10 chega a 0,296. Do ganho do MTD, **0,106 vem de estrutura e 0,191 da mudança de distribuição**. Por isso as LL e os BIC **não se comparam entre discretizações** (o resumo do relatório avisa) e o modo `blocks`, que mantém o teste com 44% de zeros, é a referência mais honesta para avaliar o modelo.

Em `blocks`, aumentar `k` reduz o ganho (s/4: ordem 10 com 0,043 sobre a ordem 0; s/5: 0,036). Limiares mais largos deixam menos símbolos informativos. O acerto de direção é igual em todos (51,1% a 51,3% na ordem 10).

## 6. Limitações

- **Um único trecho de teste** (47.893 símbolos, cerca de 166 dias). Qualquer diferença menor que 0,3 ponto de acerto ou 0,005 nats não é distinguível aqui.
- **Lacunas ignoradas** no `fit`: 0,04% das transições do treino na ordem 10 atravessam uma lacuna (340 de 905.644). Efeito desprezível, mas D6 segue pendente.
- **EM parado por tolerância rígida** (1e-8): as últimas dezenas de iterações ganham menos de 50 nats.
- **Só modo A** (um passo, com contexto real). Previsão recursiva e prequential não foram avaliadas.
- Direção medida só onde o real é ±1; não é a mesma métrica do mercado "sobe ou desce" (que inclui candles que ficaram em 0).

## 7. Como alinhar HMM e MTD (trabalho futuro)

A divisão natural é: **o HMM diz *quanto* vai se mexer; o MTD (ou algo parecido) diz *para onde***. Três caminhos, do mais barato ao mais caro:

1. **Combinação pós-ajuste (barata).** Decompor o símbolo em magnitude (`0` contra `±1`) e direção (`−1` contra `+1` dado `±1`): `P(0) = a`, `P(±1) = (1−a)·q(±1)`. O `a` vem do HMM (probabilidade preditiva de "movimento") e o `q` do MTD renormalizado entre −1 e +1. Avalia-se por LL no teste, contra o HMM e o MTD sozinhos. **Pré-requisito que hoje falta:** o `HMMDiscreto.predict_proba` devolve a posterior *suavizada* dos estados (usa a sequência toda, inclusive o futuro). Servir de entrada para previsão exigiria a distribuição **filtrada** (só o passado) e a preditiva do próximo símbolo; senão há vazamento. Ver item 5 de `docs/mtd-notas-de-fluxo.md`.
2. **Direção condicionada ao regime (barata e informativa).** Medir o acerto de direção e a força da reversão **separadamente em cada regime filtrado do HMM**. Se a reversão for forte só no regime agitado (ou só no calmo), a regra de decisão passa a ser "apostar só nesse regime". Responde, com os modelos que já existem, se o regime ajuda a direção.
3. **Modelo único com regime e lags (cara).** Um HMM cujo estado escolhe os parâmetros de um MTD: `P(x_t | contexto, regime r) = Σ_g λ_{r,g} Q_r[x_{t−g}, x_t]`. O EM combina o forward-backward do HMM com as responsabilidades de lag do MTD, em cima do código que já existe. Esse tipo de modelo aparece na literatura como *double chain Markov model* (Berchtold, 1999; a conferir antes de citar) e como autorregressão com troca de regime. É o desenho mais elegante e o mais caro, com mais parâmetros e mais risco de mínimos locais. Só vale se o caminho 2 mostrar que o regime muda a direção.

Recomendação: **2, depois 1, e 3 só se 2 mostrar algo.** O 2 não exige código novo de modelo, só o filtro do HMM.

## 8. O MTD ajuda no trabalho?

Sim, mas não como previsor de direção:

- **Cumpre o escopo:** é a família de ordem alta da comparação, com seleção de ordem por BIC, e agora há um número para cada ordem em seis discretizações.
- **Dá um segundo ponto de vista sobre a mesma estrutura.** Dois modelos de famílias diferentes (estados ocultos e lags) concluem que o que existe na série de 5 minutos é aglomeração de volatilidade. Isso fortalece o resultado do HMM, em vez de depender de um modelo só.
- **Entrega um resultado negativo útil:** mais contexto não compra direção. O acerto fica em ~51% para todas as ordens e discretizações, e isso vale como baseline sério contra o qual comparar qualquer método mais sofisticado.
- **Não supera o HMM em LL**, e a única pista de direção (a reversão em `Q`) é fraca e instável entre treino e teste.
