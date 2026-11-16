# MTD: notas do fluxo de uso

Anotações feitas ao usar o projeto "como um usuário novo": carregar o snapshot, discretizar, separar treino e teste, ajustar o MTD, comparar com os baselines e escrever o relatório (`relatorios/mtd_2026-10-01.md`, gerado por `src/experiments/mtd_report.py`). Cada item é um atrito encontrado no caminho, para corrigir depois. Nada aqui foi corrigido ainda, exceto onde indicado.

- **Escrito em:** 2026-11-03
- **Ordem:** na sequência em que os atritos apareceram.
- **Prioridade:** A = atrapalha a comparação entre modelos ou pode gerar número errado; B = atrapalha quem usa; C = limpeza.

## Ambiente

### 1. Dois ambientes virtuais e nenhum completo (B)

O `../.venv` (o que usamos) tinha `numba` e `numpy`, mas **não tinha `pyarrow`**: `load_snapshot` falhava no `import`. Instalei `pyarrow==26.0.0` nele. O `.venv` de dentro de `TCC_Desenvolvimento/` não tem nem `numba` nem `pyarrow`.

- O `requirements.txt` **não lista `numba`**, que o HMM e o MTD exigem, e fixa `numpy==2.5.3`, enquanto o ambiente em uso tem `2.3.2`.
- `test_discretize.py` e `test_snapshot.py` não passam a coleta sem `pyarrow`, então um `pytest` "limpo" quebrava antes de testar qualquer coisa.
- **Sugestão:** um único ambiente documentado; `numba` no `requirements.txt`; conferir que `pip install -r requirements.txt && pytest` passa do zero.

### 2. `import src` só funciona dentro do `pytest` (B)

O README manda rodar `pip install -e .`, mas o `../.venv` não o tinha. Sem isso, um script fora da raiz falha com `No module named 'src'`; dentro da raiz funciona com `python -m ...` ou `PYTHONPATH=.`. O `pytest` esconde o problema porque o `pytest.ini` já põe a raiz no `pythonpath`.

- **Sugestão:** instalar em modo editável no ambiente padrão e documentar que scripts rodam sempre com `python -m` a partir da raiz (já é o que o README faz para os outros comandos).

## Dados

### 3. Cortar a série filtrada com `split_data` dá outro ponto de corte (A)

A seção "Como usar" de `docs/discretizacao.md` sugere `r.symbols[r.valid]`. O caminho natural depois disso é `split_data(simbolos, 95)`. Isso **não** reproduz o corte da discretização (`r.cut`, em linhas de candle):

| Modo | Treino pelo `r.cut` | Treino por `split_data` sobre a série filtrada |
|---|---:|---:|
| blocks, s/3 | 905.654 | 905.869 |
| global, s/3 | 909.925 | 909.927 |

No modo `blocks`, 215 símbolos que a discretização trata como teste entram no treino. Não é vazamento de estatística (a régua do bloco é causal), mas é um corte diferente do que o relatório da discretização e o do HMM usam, e tornaria as comparações entre modelos incomparáveis sem aviso.

- O caminho certo é cortar **antes** de filtrar: `symbols[:cut][valid[:cut]]`.
- **Duplicação:** `src/training/hmm_training.py` tem `split_sequences` (devolve índices) e `src/experiments/mtd_report.py` tem `split_symbols` (devolve símbolos). São a mesma coisa.
- **Sugestão:** um único `split` em `src/preprocess/discretize.py`, usado por todos, que devolva também o segmento de cada símbolo.

### 4. Lacunas: o `MTD.fit` recebe uma sequência única (B, vira A em ordens altas)

É a pendência D6 da discretização. No treino do `blocks` s/3 há 35 segmentos, e as janelas de contexto que atravessam uma lacuna são contadas como normais. O efeito medido no relatório é pequeno (≈0,02% das transições na ordem 6), mas cresce com a ordem e o `fit` deveria aceitar segmentos. O HMM tem o mesmo problema.

## Modelos

### 5. Três contratos diferentes para os mesmos nomes de método (A)

| Modelo | Recebe | `predict_proba` devolve | Verossimilhança |
|---|---|---|---|
| `FirstOrder`, `MTD` | símbolos `-1/0/+1` | distribuição do **próximo símbolo** | `calculate_likelihood` |
| `Uniform`, `Persistence` | símbolos | (não existe; o nome é `predict_prob`) | `calculate_likelihood` |
| `HMMDiscreto` | **índices** `0/1/2` | **posterior dos estados** (suavizada, usa a sequência toda) | `score` |

- Não dá para trocar um modelo por outro num avaliador. O mesmo nome `predict_proba` significa coisas diferentes (próximo símbolo vs. estado oculto).
- O HMM **não tem previsão do próximo símbolo** (distribuição preditiva do filtro). Sem ela, a comparação de acerto de direção entre MTD e HMM não é possível; só dá para comparar LL.
- `Uniform` e `Persistence` usam `predict_prob`, sem o `a`.
- `FirstOrder`, `Uniform` e `Persistence` não têm `n_params` nem `n_obs`; só o `MTD` tem. O BIC dos baselines é calculado à mão.
- **Sugestão:** fechar o contrato (a pendência 3 de 2026-10-05): `fit`, `calculate_likelihood`/`score`, `predict_proba` = próximo símbolo, `n_params`, e entrada sempre em símbolos (a conversão para índice fica dentro do modelo).

### 6. Não há como avaliar no teste usando o contexto do treino (A)

`calculate_likelihood` e `predict_proba` só enxergam a sequência que recebem. Para avaliar o teste sem perder as primeiras `l` transições, e para comparar ordens diferentes nas **mesmas** transições, foi preciso concatenar a cauda do treino e fatiar à mão (`context[max_order - order:]`), duas vezes (BIC e LL do teste). Errar a fatia desloca a janela e o resultado muda sem erro nenhum. É a pendência 6 de 2026-10-05 (fronteira treino/teste).

- **Sugestão:** `model.score(sequence, context=None)` e `model.predict_proba(sequence, context=None)`, com o contexto prefixado internamente e só as transições de `sequence` contadas.

### 7. A regra de decisão sem o 0 não está no código (A)

Decidimos em 2026-10-18 que a previsão de direção é o maior entre `-1` e `+1`. `src/utils/decision.py` ainda tem só o `argmax` puro, que no MTD (`blocks` s/3) prevê `0` em 100% das posições do teste na ordem 1 e em 64% a 76% nas ordens 2 a 10. A regra ficou como `direction_rule` dentro de `mtd_report.py`. Também falta fixar o desempate (usei: empate fica em `-1`, como o `argmax`).

### 8. Persistência com ε=0,01 é um baseline fraco em LL (C)

A persistência suavizada dá −2,74 nats por símbolo no teste, pior que o uniforme em 1,64. Serve como baseline de **acerto**, mas não de verossimilhança. Vale decidir se o ε entra como parâmetro estimado ou se a persistência só é usada na métrica de direção.

### 9. Não há como salvar e carregar o MTD (B)

`salvar_modelo`/`carregar_modelo` em `src/utils/matriz_manager.py` guardam `pi`, `A`, `B` (HMM). O MTD precisa de `lam` e `Q`, e o plano do teste ao vivo conta com modelos em pickle.

## Ajuste (EM)

### 10. Tolerância de 1e-8 gasta muitas iterações para ganho mínimo (C)

O critério herdado do Baum-Welch (variação relativa < 1e-8 sobre uma LL da ordem de 10⁶) faz cada ajuste rodar de 50 a 313 iterações. Na ordem 10 do `blocks` s/3, a LL na iteração 50 ainda estava 1.004 nats abaixo da final (0,1%) e na 100 faltavam 46 (0,005%); o ajuste só parou na 172. Cada ordem leva de 3 a 23 s (2 a 5 reinícios). Não afeta a conclusão, mas `1e-6` cortaria boa parte do tempo. Decisão adiada: não mexi na tolerância para o relatório ser comparável com o HMM.

### 11. A LL por símbolo do HMM e a do MTD não são exatamente comparáveis (B)

O treino do HMM (`src/training/hmm_training.py`) chama `model.score(test)`, que roda o forward desde o início do teste, com `pi`, sem o contexto do treino, e soma `T` termos (inclui o primeiro símbolo). O MTD do relatório soma exatamente as `len(test)` transições, com o contexto vindo do treino. A diferença é de poucas transições em ~48 mil no teste, então a ordem de grandeza se compara, mas **o número exato não**: o HMM começa o teste "do zero" e o MTD não. Deve sair de um avaliador comum (item 6).

## Documentação

### 12. O README descreve uma estrutura que já não existe (C)

A árvore do README cita `first-order-markov.py` e `hmm.py`; os arquivos hoje são `first_order.py` e `hmm_numba.py`, e há pastas novas (`training/`, `experiments/`). A seção "Estado atual dos modelos" também fala do HMM antigo.
