# Roadmap: do código do TCC a uma biblioteca Python publicada

Levantamento do que falta, e do que deve mudar em arquitetura e código, para transformar o repositório em uma biblioteca publicável no PyPI.

- **Escrito em:** 2026-11-22
- **Base:** estado do repositório nesta data (suíte com 350 testes passando; `src/`, `database/`, `tests/`, `docs/`, `relatorios/`).
- **Ressalva:** o build de wheel não foi executado. Os itens de empacotamento seguem o comportamento conhecido do setuptools e devem ser confirmados com `python -m build`.
- **Prioridades:** **P0** bloqueia a publicação; **P1** necessário para uma versão 1.0 sólida; **P2** desejável.

## Diagnóstico

O código tem boa base numérica: os modelos são validados contra cálculo direto e a suíte é extensa. Mas o repositório ainda é de **pesquisa**, não de biblioteca. Ele mistura coleta de dados, banco, túnel SSH, relatórios e modelos no mesmo pacote, e os modelos não compartilham um contrato de uso.

---

## A. Decisões de escopo (antes de mexer em código)

1. **[P0] O que a biblioteca é.** Hoje o alfabeto está fixo em `SYMBOLS = (-1, 0, 1)` e `M = 3` em vários módulos. Duas opções:
   - modelos de Markov para séries discretizadas de retorno (3 símbolos); ou
   - modelos de Markov para sequências categóricas quaisquer (alfabeto de tamanho `m`, com rótulos).

   A segunda é mais publicável e custa pouco, pois o código já usa `M` e índices internamente.
2. **[P0] Nome do pacote.** `tcc-desenvolvimento` não serve. Escolher um nome livre no PyPI (a conferir antes de adotar).
3. **[P0] Licença.** O `LICENSE` atual é GPL-3, que obriga quem usar a biblioteca a liberar o próprio código e desencoraja a adoção. Avaliar MIT, BSD ou Apache-2.0. A decisão é da dupla e, provavelmente, das regras de propriedade intelectual do SENAC e dos orientadores.
4. **[P0] Idioma da documentação pública.** No PyPI o padrão é inglês. A convenção atual (README e comentários em PT) pode continuar no repositório, mas docstrings e documentação da API pública precisam de uma decisão explícita.
5. **[P1] Escopo da versão 1.0.** Definir o que entra: modelos ingênuos (uniforme, persistência, Markov de 1ª ordem), MTD e HMM; ou também o HMM bayesiano e o MTDg (uma `Q` por lag), que são objetivos do TCC e ainda não existem. Não publicar o que não existe.
6. **[P1] Separar em dois repositórios.**
   - **Biblioteca:** modelos, discretização, métricas, seleção de modelo.
   - **Experimentos do TCC:** coleta Binance, Postgres, SSH, `schema.sql`, snapshot, `relatorios/` e `docs/`, dependendo da biblioteca.

---

## B. Empacotamento

7. **[P0] Pacotes `src` e `database` no topo.** Um `pip install` instalaria `import src` e `import database` no site-packages de qualquer usuário, o que garante colisão. Adotar o layout `src/<nome_do_pacote>/…`.
8. **[P0] Não existe nenhum `__init__.py` no projeto.** Os pacotes são namespace implícitos, e `setuptools.packages.find` não os encontra: um wheel construído hoje sairia vazio ou incompleto. Criar os `__init__.py`.
9. **[P0] `[project]` sem `dependencies`.** O `pyproject.toml` não declara nenhuma dependência. O `requirements.txt` lista `numpy`, `psycopg`, `paramiko` e outros, mas **não lista `numba`**, que o HMM e o MTD exigem. Um `pip install` limpo quebraria no primeiro `import`.
10. **[P0] Separar as dependências em extras.**
    - Núcleo: `numpy`, `numba`.
    - `[parquet]`: `pyarrow`.
    - `[db]`: `psycopg`, `sshtunnel`, `paramiko` (ou sair da biblioteca, item 6).
    - Remover `cryptography`, `bcrypt`, `PyNaCl`, `cffi` e `invoke` do núcleo.
11. **[P0] Versões fixadas com `==`.** Biblioteca declara faixas (`numpy>=1.26`), não versões exatas. O `requirements.txt` fixa `numpy==2.5.3`, e o ambiente em uso tem 2.3.2.
12. **[P1] `pyarrow` importado no topo de `discretize.py`.** Com `pyarrow` opcional (item 10), o import precisa ser tardio, dentro de `load_snapshot`.
13. **[P1] Metadados faltando no `pyproject.toml`:** `description`, `readme`, `authors`, `license` (expressão SPDX), `classifiers`, `keywords`, `urls`, versão em um lugar só.
14. **[P1] Extra `gpu` com `cupy` aponta para um `hmm.py` que não existe mais.** Só há `hmm_numba.py`. Remover o extra e as menções no README, ou reimplementar.
15. **[P1] Matriz de versões de Python.** O `requires-python` declara `>=3.12`, mas só se desenvolve na 3.14. Testar as versões declaradas e ajustar o `requires-python` ao que for testado, conferindo o suporte do numba a cada uma.
16. **[P1] `pytest.ini` e `[tool.pytest.ini_options]` duplicados.** Manter só um.
17. **[P2] Incluir `py.typed`** e type hints no pacote (item 37).

---

## C. Arquitetura

18. **[P0] Caminhos derivados do local do arquivo.** `ROOT = Path(__file__).resolve().parents[2]` e `STD_FOLDER = ROOT / "relatorios"` aparecem em `discretize.py`, `hmm_training.py`, `mtd_report.py` e `mtd_analysis.py`. Com a biblioteca instalada, isso aponta para dentro do site-packages. Os padrões de saída devem ser o diretório atual (`Path.cwd()`) ou um argumento obrigatório.
19. **[P0] Código que não é biblioteca dentro do pacote.**
    - `src/scrapping/`, `database/` e `normalize_btc.sql` saem (item 6).
    - O bloco `__main__` de `hmm_numba.py` importa `database.index`. É acoplamento morto e deve sair.
20. **[P1] Camadas propostas:**
    ```
    <pacote>/
      alphabet.py        # alfabeto e conversão símbolo <-> índice (hoje symbols.py)
      preprocessing/     # discretização (blocks/global), lacunas, segmentos, split
      models/            # base.py + uniform, persistence, first_order, mtd, hmm
      selection/         # bic, aic, escolha de ordem
      evaluation/        # métricas, avaliador comum, horizontes, calibração
      simulation/        # amostragem, autocorrelação
      io/                # salvar/carregar modelos (sem pickle)
      cli.py             # opcional
    ```
21. **[P0] Mover para a biblioteca o que hoje está em `experiments/`.** São funções que um usuário esperaria encontrar na própria biblioteca:
    - matriz de confusão, precisão, *recall* e F1 por classe, e acerto de direção com intervalo de Wilson;
    - filtro preditivo do HMM (`hmm_pred`);
    - `memory_times` e `stationary_distribution`;
    - as previsões dos baselines do escopo (`first_order_pred`, `persistence_pred`), que hoje só existem no script;
    - `direction_rule`, ainda em `mtd_report.py`.

    Os scripts de relatório ficam só como exemplos de uso.
22. **[P1] Três funções de split diferentes** (`split_data`, `split_sequences`, `split_symbols`). O `split_data` dá outro ponto de corte que o `Discretization.cut` (905.869 contra 905.654 símbolos de treino, segundo as notas de fluxo). Manter uma só, no módulo de pré-processamento, que devolva também o segmento de cada símbolo.
23. **[P1] Avaliador comum.** Hoje cada script concatena a cauda do treino à mão e fatia com `context[max_order - order:]`, um erro silencioso à espera de acontecer. A biblioteca precisa de `evaluate(model, train, test)`, ou `score(seq, context=...)`, com o contexto tratado internamente.
24. **[P1] Previsão incremental para uso ao vivo** (objetivo do TCC). Uma interface com `update(symbol)` e `predict_next()`, sem recomputar a sequência inteira.

---

## D. API dos modelos

25. **[P0] Contrato único, hoje inconsistente:**

    | Modelo | `fit` devolve | `predict_proba` devolve | Verossimilhança | Entrada |
    |---|---|---|---|---|
    | `FirstOrder`, `MTD` | `None` | próximo símbolo | `calculate_likelihood` | símbolos |
    | `Uniform`, `Persistence` | `None` | (o método se chama `predict_prob`) | `calculate_likelihood` | símbolos |
    | `HMMDiscreto` | `self` | **posterior dos estados** | `score` | **índices** |

    Decidir: `fit(X)` devolve `self`; verossimilhança com um nome só (`score`); `predict_proba` = distribuição do próximo símbolo; `n_params_`; entrada sempre em símbolos, com a conversão para índice dentro do modelo. Formalizar com uma classe base ou `Protocol`.
26. **[P0] O mesmo nome com significados diferentes.** `predict` é Viterbi no HMM e `argmax` em `utils/decision.py`. O HMM não tem previsão do próximo símbolo; a que existe foi escrita no `mtd_analysis`.
27. **[P0] Nomes em português em código público.** `HMMDiscreto`, `n_estados`, `n_simbolos`, `n_reinicios`, `tolerancia`, `piso`, `concentracao`, `semente`, `aquecer_jit`, `pegar_dados` e todo o `matriz_manager.py` (`salvar_matriz`, `InfoModelo`…). Isso contraria a convenção "código em inglês". Renomear junto com o resto da API (por exemplo `n_states`, `n_restarts`, `tol`, `seed`).
28. **[P1] Convenção scikit-learn completa:** hiperparâmetros no `__init__`, `get_params`/`set_params`, atributos aprendidos com `_` no final. Hoje o HMM usa `pi_`, `A_`, `B_`, mas o MTD usa `lam`, `Q`, `ll`, `info`.
29. **[P1] `n_params` e `n_obs` em todos os modelos.** Só o MTD os tem, e o BIC dos baselines é calculado à mão no script. Deve haver um método único (`model.bic(seq)`).
30. **[P1] Treino com várias sequências (lacunas, pendência D6).** `fit` recebe uma única sequência e as lacunas são concatenadas. Aceitar uma lista de segmentos nos três modelos. Isso também resolve o `pi` degenerado do HMM, estimado a partir de um único instante.
31. **[P1] `FirstOrder`, `Persistence` e `Uniform`:**
    - `FirstOrder` levanta erro quando um símbolo não aparece no treino e não tem suavização. Decidir se um parâmetro opcional de suavização entra na API. Pelo escopo do TCC, **não** generalizar para ordem k: tudo de ordem ≥ 2 é o MTD.
    - A persistência com ε fixo de 0,01 é um baseline fraco em LL (−2,74 nats por símbolo, pior que o uniforme). Decidir se serve só como baseline de acerto.
    - `Uniform.fit` não aprende nada. Manter como baseline, documentado como tal.
32. **[P1] HMM como produto:**
    - Rótulos estáveis: ordenar os estados por `P(0)` ao final do ajuste (recomendado nas duas análises).
    - `max_iter=1000` não bastou a partir de 5 estados.
    - Inicialização por divisão de estado (n → n+1), como alternativa aos reinícios aleatórios.
    - O resultado depende do número de threads (`n_blocos=None`). O padrão deveria ser determinístico.
    - Só 3 de 10 reinícios chegam à melhor solução com 3 estados. Expor `n_restarts` com esse aviso na documentação.
33. **[P1] Regra de decisão sem o 0.** Mover `direction_rule` para `utils/decision.py` e fixar o desempate (hoje, o empate vai para −1). Resolver o conflito de nome com `predict` (item 26).
34. **[P2] Salvar e carregar modelos sem `pickle`.** O `pickle` é inseguro de carregar e frágil entre versões. Usar `.npz` ou JSON com versão do formato, alfabeto e parâmetros, com `to_dict`/`from_dict` por modelo. O MTD hoje não tem como ser salvo, e o plano do teste ao vivo depende disso.

---

## E. Qualidade do código

35. **[P0] Corrigir erros de digitação** em mensagens e docstrings que viram API pública: "seuquence", "indeces", "martrix", "verade", "discretiozar".
36. **[P1] Docstrings reais.** `mtd.py`, `bic.py`, `symbols.py`, os modelos ingênuos e `decision.py` usam comentários `##` antes da função, então `help()` e o Sphinx não os enxergam. `discretize.py` e `hmm_numba.py` já usam docstrings em estilo Google. Padronizar em um estilo.
37. **[P1] Type hints** (hoje só `matriz_manager.py` os tem) e checagem com `mypy` ou `pyright` no CI.
38. **[P1] Lint e formatação:** `ruff` (lint e format), `pre-commit` e configuração no `pyproject.toml`.
39. **[P1] Validação de entrada uniforme.** Cada modelo valida de um jeito, com mensagens diferentes. Centralizar em uma função.
40. **[P1] Numba como dependência obrigatória.**
    - A primeira chamada compila (segundos).
    - `cache=True` escreve em `__pycache__` dentro do site-packages, o que falha em instalações somente-leitura.
    - Documentar `NUMBA_CACHE_DIR` e decidir se haverá um modo sem numba (mais lento, mas instalável em qualquer lugar).
41. **[P2] Desempenho:** benchmarks versionados, uso de memória do forward (`alpha` de T×n) e `float32` opcional.

---

## F. Testes e CI

42. **[P0] O HMM não tem arquivo de teste.** Em `tests/` há testes de MTD, Markov, BIC, discretização e outros, mas nenhum `test_hmm*`. O `hmm_numba.py` (650 linhas, o modelo central do TCC) e o `hmm_training.py` não têm cobertura, e `matriz_manager.py` e `data_manager.py` também não. Para o HMM, seguir o padrão do MTD: comparar com enumeração de caminhos em casos pequenos e, como referência externa, com o `hmmlearn` (dependência só de desenvolvimento).
43. **[P1] CI no GitHub Actions:** matriz de sistemas operacionais (Linux, macOS, Windows) e de versões de Python; `pytest`, `ruff`, `mypy`; cobertura (`pytest-cov`) com meta mínima.
44. **[P1] Testes com dependências opcionais** (`test_discretize.py` e `test_snapshot.py` precisam de `pyarrow`) devem ser marcados (`@pytest.mark.parquet`) e pulados quando a dependência não estiver instalada.
45. **[P1] Testes de regressão numérica** com valores fixados (por exemplo, os números dos relatórios), para detectar mudança de resultado entre versões.
46. **[P2] Testes baseados em propriedades** (`hypothesis`): linhas somando 1, LL monótona no EM, invariância a permutação de rótulos.

---

## G. Documentação

47. **[P0] README novo.** O atual descreve VM AWS, túnel SSH, `\copy` e `psql`, e a árvore de arquivos cita arquivos que não existem (`first-order-markov.py`, `hmm.py`). Para a biblioteca: instalação, exemplo mínimo de cerca de 10 linhas, lista de modelos e links.
48. **[P1] Documentação de API gerada** (Sphinx ou MkDocs) e publicada (Read the Docs ou GitHub Pages).
49. **[P1] Páginas de teoria e referências:** MTD (Raftery, 1985; Lèbre e Bourguignon, 2008), HMM, BIC, discretização (modos `blocks` e `global` e o argumento do vazamento). São páginas de documentação técnica, escritas pela dupla; o texto da monografia continua fora do escopo do assistente.
50. **[P1] Tutorial executável** (notebook) com dados sintéticos pequenos embutidos, para o usuário rodar sem baixar o snapshot de 46 MiB.
51. **[P1] `CHANGELOG.md`, `CONTRIBUTING.md` e `CITATION.cff`** (útil em contexto acadêmico), mais a política de versionamento (0.x até a API estabilizar).
52. **[P1] Alinhar `docs/` ao código.** O registro de decisões da discretização cita `discretizar.py`, mas o arquivo hoje é `discretize.py`.

---

## H. Publicação e segurança

53. **[P0] Auditar o histórico do git antes de abrir ao público.** O `config.ini` está no `.gitignore`, mas é preciso confirmar se algum commit antigo teve credenciais ou chaves SSH. Se tiver, limpar o histórico ou publicar a partir de um repositório novo.
54. **[P0] Verificar as regras do SENAC e dos orientadores** sobre a publicação de código derivado do TCC.
55. **[P1] Fluxo de release:** `python -m build`, `twine check`, ensaio no TestPyPI, depois PyPI com "trusted publishing" (GitHub Actions, sem token), tag e release no GitHub, versão em um lugar só.
56. **[P1] Dados fora do pacote.** O snapshot continua como asset de release do repositório de experimentos. `load_snapshot` assume o esquema de candles da Binance e deve documentar o esquema esperado (colunas `open_time` e `close`).
57. **[P2] Higiene do repositório.** Há `.pytest_cache` na árvore e dois ambientes virtuais (`../.venv` e o de dentro do projeto). Manter um ambiente documentado.

---

## Ordem sugerida

1. **Decisões** (itens 1 a 6): nome, licença, escopo e idioma. Sem elas, o resto vira retrabalho.
2. **Reestruturar** (7 a 10, 18, 19, 20): layout `src/<pacote>/`, `__init__.py`, dependências e caminhos.
3. **Testes do HMM** (42): antes de renomear o HMM (27), para não renomear código sem rede de segurança.
4. **Contrato dos modelos** (25 a 29): é a mudança que quebra mais código e testes. Fazer cedo, com os testes existentes como rede de segurança.
5. **Mover o que está em `experiments/`** (21 a 23), depois lint, tipos e CI (37, 38, 43).
6. **Documentação e release** (47 a 55).

## Decisões em aberto (da dupla)

| Decisão | Itens afetados |
|---|---|
| Nome do pacote | 2, 7, 55 |
| Licença | 3, 54 |
| Alfabeto geral ou só 3 símbolos | 1, 20, 25 |
| Idioma da documentação pública | 4, 36, 47 |
| Escopo da 1.0 (HMM bayesiano, MTDg) | 5 |
| Um repositório ou dois | 6, 19, 56 |

## Fora do alcance do assistente sem pedido explícito

- `FirstOrder` e as lacunas por segmento (item 30), que a dupla quer acompanhar de perto.
- Qualquer texto da monografia.
