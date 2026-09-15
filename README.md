# TCC — Modelagem Probabilística de Séries Temporais de Criptomoedas

Trabalho de Conclusão de Curso cujo objetivo é coletar dados históricos de negociação do par **BTC/USDT** e comparar diferentes modelos probabilísticos (modelos ingênuos, Cadeias de Markov e Modelos Ocultos de Markov) na tarefa de modelar/prever o comportamento da série temporal.

## Tecnologias

- **Python 3.14**
- **NumPy** — geração de dados sintéticos e operações matriciais dos modelos
- **psycopg 3** (`psycopg` + `psycopg-binary`) — driver de conexão com o PostgreSQL
- **paramiko** + **sshtunnel** — abertura de túnel SSH para acesso seguro ao banco remoto
- **cryptography**, **bcrypt**, **PyNaCl**, **cffi** — dependências de suporte do `paramiko`/`sshtunnel` para autenticação SSH
- **PostgreSQL** — armazenamento dos dados de mercado
- **psql (CLI)** — usado diretamente pelo serviço de scrapping para inserção em lote via `\copy`, evitando overhead de bibliotecas externas nesse ponto específico
- **Binance API (REST)** — fonte dos dados históricos de candles

## Serviços em execução

| Serviço | Descrição | Local |
|---|---|---|
| **VM (AWS EC2 / Ubuntu)** | Hospeda o PostgreSQL e roda o serviço de scrapping em background | Acesso via SSH |
| **PostgreSQL** | Banco relacional com os dados de candles do BTC/USDT | Porta `5432`, acessível apenas via túnel SSH a partir da VM |
| **`background.py`** | Loop contínuo (a cada 5 minutos) que consulta o último `open_time` salvo e sincroniza os dados faltantes junto à API da Binance | Roda como processo de longa duração na VM |

A conexão da aplicação (scripts de modelagem) com o banco nunca é feita diretamente: é aberto um túnel SSH até a VM e, a partir dele, uma conexão local ao PostgreSQL (ver `database/index.py`).

## Como os dados são armazenados

- Os candles de 5 minutos do par `BTCUSDT` são coletados da Binance (endpoint `/api/v3/klines`) desde `2017-08-17` e persistidos na tabela `btc_usdt`, com as colunas originais da API (`open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`) mais `symbol` e `interval`.
- A inserção é feita em lote: o serviço monta um arquivo `\copy` em formato texto (TSV) e insere numa tabela temporária, aplicando `INSERT ... ON CONFLICT (symbol, interval, open_time) DO NOTHING` para evitar duplicatas.
- O serviço de background (`src/scrapping/background.py`) roda em loop, buscando periodicamente `MAX(open_time)` no banco e sincronizando (`sincronizar`) apenas o intervalo ainda não coletado, mantendo a base sempre atualizada.
- Existe também uma tabela/view `normalize_data`, com a coluna `close_normalized`, utilizada pelos modelos como série discretizada (queda / estável / alta) de entrada para as cadeias de Markov.

## Estrutura do projeto

```
database/
  index.py               # conexão com o Postgres via túnel SSH + pegar_dados() (lê a série normalizada)
src/
  scrapping/
    scrapping.py          # coleta e inserção dos candles da Binance
    background.py         # serviço contínuo de sincronização
  models/
    naive_models/
      first-order-markov.py   # cadeia de Markov de 1ª ordem (matriz de transição + log-verossimilhança)
    hmm/
      hmm.py                  # HMM: forward-backward (com escalonamento) e Baum-Welch (EM) sobre a série real
  utils/
    bic.py                    # (a implementar) critério de informação bayesiano
```

## Como executar

1. Crie e ative um ambiente virtual e instale as dependências:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Configure o `config.ini` (não versionado) com as credenciais do PostgreSQL e os dados de acesso SSH à VM.
3. Rode o serviço de scrapping/sincronização (na VM) ou os scripts de modelagem localmente, apontando para o banco via túnel SSH.

## Estado atual dos modelos

- **Cadeia de Markov de 1ª ordem** (`first-order-markov.py`): monta a matriz de transição (3x3: queda/estável/alta) a partir de `pegar_dados()` e calcula a log-verossimilhança do modelo ajustado.
- **HMM** (`hmm.py`): implementa o algoritmo *forward-backward* com escalonamento (baseado em Jurafsky) e o treinamento via **Baum-Welch** (EM), rodando sobre a série real (`close_normalized`) obtida do banco. Ainda em fase de teste/validação dos resultados (`max_iter` reduzido, sem critério de parada refinado).

## Próximos passos

- [ ] Finalizar a implementação dos modelos ingênuos (modelo aleatório e modelo de persistência)
- [ ] Validar a convergência e os resultados do Baum-Welch com a base histórica completa
- [ ] Implementar o cálculo do BIC (Critério de Informação Bayesiano) para seleção de modelos
- [ ] Implementar o MTD (Mixture Transition Distribution)
- [ ] Gerar uma análise detalhada dos modelos e suas performances (métricas de comparação, verossimilhança, BIC entre modelos, previsão fora da amostra, etc.)
