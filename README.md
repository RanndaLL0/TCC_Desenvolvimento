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

| Serviço                   | Descrição                                                                                                                       | Local                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **VM (AWS EC2 / Ubuntu)** | Hospeda o PostgreSQL e roda o serviço de scrapping em background                                                                | Acesso via SSH                                              |
| **PostgreSQL**            | Banco relacional com os dados de candles do BTC/USDT                                                                            | Porta `5432`, acessível apenas via túnel SSH a partir da VM |
| **`background.py`**       | Loop contínuo (a cada 5 minutos) que consulta o último `open_time` salvo e sincroniza os dados faltantes junto à API da Binance | Roda como processo de longa duração na VM                   |

A conexão da aplicação (scripts de modelagem) com o banco, em `database/index.py`, tem dois modos. Com as credenciais da seção `[SSH_CONNECTION]` do `config.ini`, abre um túnel SSH até a VM e conecta ao PostgreSQL através dele (é o caso da produção). Sem elas, conecta direto ao PostgreSQL da seção `[postgres]`, por exemplo uma base local restaurada a partir do snapshot. Se não houver credenciais SSH nem PostgreSQL acessível, a conexão falha com erro.

## Como os dados são armazenados

- Os candles de 5 minutos do par `BTCUSDT` são coletados da Binance (endpoint `/api/v3/klines`) desde `2017-08-17` e persistidos na tabela `btc_usdt`, com as colunas originais da API (`open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`) mais `symbol` e `interval`.
- A inserção é feita em lote: o serviço monta um arquivo `\copy` em formato texto (TSV) e insere numa tabela temporária, aplicando `INSERT ... ON CONFLICT (symbol, interval, open_time) DO NOTHING` para evitar duplicatas.
- O serviço de background (`src/scrapping/background.py`) roda em loop, buscando periodicamente `MAX(open_time)` no banco e sincronizando (`sincronizar`) apenas o intervalo ainda não coletado, mantendo a base sempre atualizada.
- Existe também a tabela `normalize_data`, com a coluna `close_normalized`, utilizada pelos modelos como série discretizada (queda / estável / alta) de entrada para as cadeias de Markov. Ela é recriada pela procedure `normalize_btc()` (`database/normalize_btc.sql`), que é a discretização **legada**: as limitações conhecidas estão listadas no cabeçalho do arquivo.

## Estrutura do projeto

```
database/
  index.py               # conexão com o Postgres (túnel SSH ou direta) + pegar_dados() (lê a série normalizada)
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

### Pré-requisitos

| Item                               | Para quê                                                                           | Observação                                                                                                                                                      |
| ---------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python 3.14**                    | Todo o projeto                                                                     | Versão usada no desenvolvimento; o `pyproject.toml` declara `>=3.12`.                                                                                           |
| **`psql`** (cliente do PostgreSQL) | O scrapping insere os candles chamando o `psql` via `subprocess` (`\copy` em lote) | Precisa estar no `PATH`. Ubuntu/Debian: `sudo apt install postgresql-client`.                                                                                   |
| **PostgreSQL 16+**                 | Só se for hospedar o banco                                                         | A produção usa o 16; o esquema também foi aplicado no 18.                                                                                                       |
| **`cupy`** (opcional)              | Apenas o HMM em GPU, `src/models/hmm/hmm.py`                                       | Exige driver NVIDIA + CUDA. Instale com `pip install -e ".[gpu]"` (CUDA 12; para CUDA 13 use `cupy-cuda13x`). O `hmm_numba.py` roda em CPU e dispensa o `cupy`. |

### 1. Instalação

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

O `pip install -e .` instala o projeto em modo editável, permitindo que os imports `src...` e `database...` funcionem a partir de qualquer diretório.

### 2. Configuração (`config.ini`)

O `config.ini` não é versionado (contém credenciais). Copie o modelo e preencha:

```
cp config.ini.example config.ini
```

| Seção              | Chave                          | Significado                                                                                                                                                    |
| ------------------ | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[postgres]`       | `host`, `port`                 | Endereço do PostgreSQL. Com túnel SSH, **visto a partir da VM** (o scrapping, que roda na VM, usa os mesmos valores no `psql -h/-p`); na conexão direta, **visto a partir desta máquina**. |
| `[postgres]`       | `user`, `password`, `database` | Credenciais e nome do banco. Escreva os valores **sem aspas** (o scrapping não as remove). A `password` pode ficar vazia (por exemplo, um PostgreSQL local com autenticação `trust`).                                                                     |
| `[SSH_CONNECTION]` | `SSH_HOST`                     | IP ou nome da VM. A porta SSH é fixa em 22 (`database/index.py`). **A seção é opcional**: remova-a ou deixe as três chaves vazias para conectar direto. Preencher só parte delas é erro.                                                                                              |
| `[SSH_CONNECTION]` | `SSH_USER`                     | Usuário SSH na VM.                                                                                                                                             |
| `[SSH_CONNECTION]` | `SSH_KEY`                      | Caminho da chave privada SSH. As aspas são opcionais.                                                                                                          |

### 3. Banco de dados

Em um banco vazio, crie as tabelas e a procedure de discretização legada:

```
psql -h <host> -U <usuario> -d <banco> -f database/schema.sql
psql -h <host> -U <usuario> -d <banco> -f database/normalize_btc.sql
```

O `schema.sql` cria `btc_usdt` (candles, com a chave primária que o `ON CONFLICT` do scrapping usa) e `normalize_data`. Depois de carregar os candles (etapa abaixo), gere a série discretizada com:

```
psql -h <host> -U <usuario> -d <banco> -c "CALL normalize_btc();"
```

### 4. Coleta dos candles

Os dois comandos usam o `psql` e o `config.ini` e devem rodar na máquina que enxerga o PostgreSQL (a VM):

```
python src/scrapping/scrapping.py     # carga histórica completa, desde 2017-08-17
python src/scrapping/background.py    # serviço contínuo: sincroniza a cada 5 minutos
```

### 5. Testes

```
python -m pytest
```

## Estado atual dos modelos

- **Cadeia de Markov de 1ª ordem** (`first-order-markov.py`): monta a matriz de transição (3x3: queda/estável/alta) a partir de `pegar_dados()` e calcula a log-verossimilhança do modelo ajustado.
- **HMM** (`hmm.py`): implementa o algoritmo _forward-backward_ com escalonamento (baseado em Jurafsky) e o treinamento via **Baum-Welch** (EM), rodando sobre a série real (`close_normalized`) obtida do banco. Ainda em fase de teste/validação dos resultados (`max_iter` reduzido, sem critério de parada refinado).

## Próximos passos

- [ ] Finalizar a implementação dos modelos ingênuos (modelo aleatório e modelo de persistência)
- [ ] Validar a convergência e os resultados do Baum-Welch com a base histórica completa
- [ ] Implementar o cálculo do BIC (Critério de Informação Bayesiano) para seleção de modelos
- [ ] Implementar o MTD (Mixture Transition Distribution)
- [ ] Gerar uma análise detalhada dos modelos e suas performances (métricas de comparação, verossimilhança, BIC entre modelos, previsão fora da amostra, etc.)
