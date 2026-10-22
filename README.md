# TCC — Modelagem Probabilística de Séries Temporais de Criptomoedas

Trabalho de Conclusão de Curso cujo objetivo é coletar dados históricos de negociação do par **BTC/USDT** e comparar diferentes modelos probabilísticos (modelos ingênuos, Cadeias de Markov e Modelos Ocultos de Markov) na tarefa de modelar/prever o comportamento da série temporal.

## Tecnologias

- **Python 3.14**
- **NumPy** — geração de dados sintéticos e operações matriciais dos modelos
- **psycopg 3** (`psycopg` + `psycopg-binary`) — driver de conexão com o PostgreSQL
- **paramiko** + **sshtunnel** — abertura de túnel SSH para acesso seguro ao banco remoto
- **pyarrow** — leitura e escrita do snapshot da base em Parquet (`database/snapshot.py`)
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

| Seção              | Chave                          | Significado                                                                                                                                                                                |
| ------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[postgres]`       | `host`, `port`                 | Endereço do PostgreSQL. Com túnel SSH, **visto a partir da VM** (o scrapping, que roda na VM, usa os mesmos valores no `psql -h/-p`); na conexão direta, **visto a partir desta máquina**. |
| `[postgres]`       | `user`, `password`, `database` | Credenciais e nome do banco. Escreva os valores **sem aspas** (o scrapping não as remove). A `password` pode ficar vazia (por exemplo, um PostgreSQL local com autenticação `trust`).      |
| `[SSH_CONNECTION]` | `SSH_HOST`                     | IP ou nome da VM. A porta SSH é fixa em 22 (`database/index.py`). **A seção é opcional**: remova-a ou deixe as três chaves vazias para conectar direto. Preencher só parte delas é erro.   |
| `[SSH_CONNECTION]` | `SSH_USER`                     | Usuário SSH na VM.                                                                                                                                                                         |
| `[SSH_CONNECTION]` | `SSH_KEY`                      | Caminho da chave privada SSH. As aspas são opcionais.                                                                                                                                      |

### 3. Banco de dados

Em um banco vazio, crie as tabelas e a procedure de discretização legada:

```
psql -h <host> -U <usuario> -d <banco> -f database/schema.sql
psql -h <host> -U <usuario> -d <banco> -f database/normalize_btc.sql
```

O `schema.sql` cria `btc_usdt` (candles, com a chave primária que o `ON CONFLICT` do scrapping usa) e `normalize_data`. Depois de carregar os candles (pela coleta ou pelo snapshot, nas etapas abaixo), gere a série discretizada com:

```
psql -h <host> -U <usuario> -d <banco> -c "CALL normalize_btc();"
```

### 4. Reconstruir a base a partir do snapshot

Sem acesso à VM, a base pode ser montada localmente a partir de um snapshot da tabela `btc_usdt`: 957.853 candles, de 2017-08-17 04:00 a 2026-09-30 23:55 (46 MiB). O arquivo Parquet é publicado como anexo do [release `snapshot-2026-10-01`](https://github.com/RanndaLL0/TCC_Desenvolvimento/releases/tag/snapshot-2026-10-01), e o manifesto (`snapshot/btc_usdt_2026-10-01.manifest.json`) já vem no clone do repositório. Detalhes do formato e dos arquivos na subseção "Sobre o snapshot", abaixo.

1. Baixe o Parquet para a pasta `snapshot/`, ao lado do manifesto:
   ```
   curl -L -o snapshot/btc_usdt_2026-10-01.parquet \
     https://github.com/RanndaLL0/TCC_Desenvolvimento/releases/download/snapshot-2026-10-01/btc_usdt_2026-10-01.parquet
   ```
   Com o GitHub CLI, o equivalente é `gh release download snapshot-2026-10-01 --dir snapshot`. Para conferir o download, o resultado de `sha256sum snapshot/btc_usdt_2026-10-01.parquet` deve ser igual ao campo `sha256` do manifesto (o `restore` faz essa mesma conferência sozinho).
2. Crie um banco PostgreSQL vazio e aponte o `config.ini` para ele **sem a seção `[SSH_CONNECTION]`** (conexão direta, ver etapa 2).
3. Restaure:
   ```
   python -m database.snapshot restore snapshot/btc_usdt_2026-10-01.parquet
   ```
   O comando confere o checksum **antes** de tocar no banco, aplica o `schema.sql`, carrega os candles e, ao final, confere linhas e período contra o manifesto (se algo não bater, a transação é desfeita). Ele só carrega em `btc_usdt` **vazia**: se o `config.ini` apontar para um banco que já tem dados, como a produção, nada é alterado.
4. Gere a série discretizada legada com os comandos da etapa 3 (`normalize_btc.sql` e `CALL normalize_btc();`).
5. Confira a conexão: `python -m database.index`.

#### Sobre o snapshot

**Parquet.** Formato de arquivo para tabelas, binário, comprimido (zstd) e organizado por colunas. Diferente do CSV, guarda o tipo de cada coluna (`timestamp`, `double`, `int32`, texto), então os dados não precisam ser reinterpretados na carga. É lido e escrito com o `pyarrow`.

**Manifesto.** JSON com o mesmo nome base do Parquet, que descreve o conteúdo do arquivo:

| Campo                                    | Significado                                                            |
| ---------------------------------------- | ---------------------------------------------------------------------- |
| `tabela`                                 | Tabela de origem (`btc_usdt`).                                         |
| `ate_exclusivo`                          | Data final do corte: entram os candles com `open_time` anterior a ela. |
| `linhas`                                 | Número de linhas do Parquet.                                           |
| `primeiro_open_time`, `ultimo_open_time` | Período coberto.                                                       |
| `colunas`                                | Colunas, na ordem da tabela.                                           |
| `formato`                                | `parquet/zstd`.                                                        |
| `sha256`                                 | Impressão digital do Parquet: qualquer byte alterado muda o valor.     |
| `gerado_em`                              | Momento da exportação (UTC).                                           |

O manifesto serve para detectar um Parquet corrompido ou trocado antes de restaurar e para conferir o banco depois da carga (linhas e período).

**Nomes dos arquivos.** Cada snapshot é um par `<tabela>_<data final>.parquet` e `<tabela>_<data final>.manifest.json`, na mesma pasta. O `restore` encontra o manifesto derivando o nome a partir do Parquet; por isso os dois devem andar juntos e, se um for renomeado, o outro também. Um snapshot com outra data final gera um novo par e não apaga os anteriores, já a exportação repetida com a mesma data sobrescreve os dois arquivos. O `restore` nunca escolhe "o mais recente": recebe sempre o caminho explícito.

**Comandos.** Rodam a partir da raiz do projeto, com o ambiente virtual ativo, e usam o banco definido no `config.ini`.

| Comando                                                               | O que faz                                                                                                                                                                                                               |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python -m database.snapshot export --ate AAAA-MM-DD [--saida PASTA]` | Lê `btc_usdt` (somente leitura) e grava o par Parquet + manifesto em `PASTA` (padrão: `snapshot/`). O `--ate` é obrigatório e exclusivo: como a coleta continua, sem uma data final o conteúdo mudaria a cada execução. |
| `python -m database.snapshot restore <parquet>`                       | Confere o checksum, aplica o `schema.sql` e carrega o Parquet em `btc_usdt`, que precisa estar vazia.                                                                                                                   |

A ajuda de cada comando está em `python -m database.snapshot --help`, `... export --help` e `... restore --help`. As datas seguem o formato ISO (`2026-10-01`).

**Publicar um snapshot novo** (para quem mantém a base, com acesso ao banco com os dados):

1. Exporte: `python -m database.snapshot export --ate AAAA-MM-DD`.
2. Publique o Parquet em um release:
   ```
   gh release create snapshot-AAAA-MM-DD snapshot/btc_usdt_AAAA-MM-DD.parquet \
     --title "Snapshot btc_usdt até AAAA-MM-DD" \
     --notes "SHA-256 no manifesto: snapshot/btc_usdt_AAAA-MM-DD.manifest.json"
   ```

### 5. Discretização

Discretiza o fechamento do snapshot em `{-1, 0, +1}` e grava o relatório (registros, lacunas, parâmetros e distribuição de classes por limiar, partição e mês):

```
python -m src.preprocess.discretizar --snapshot snapshot/btc_usdt_2026-10-01.parquet
```

Por padrão roda os dois modos (`blocos` e `global`) com os limiares `s/3`, `s/4` e `s/5`, e escreve `relatorios/discretizacao_2026-10-01.md` e `.json`. As opções (`--modo`, `--k`, `--intervalo`, `--treino`, `--saida`), as decisões e os resultados estão em [`docs/discretizacao.md`](docs/discretizacao.md).

### 6. Coleta dos candles

Os dois comandos usam o `psql` e o `config.ini` e devem rodar na máquina que enxerga o PostgreSQL (a VM):

```
python src/scrapping/scrapping.py     # carga histórica completa, desde 2017-08-17
python src/scrapping/background.py    # serviço contínuo: sincroniza a cada 5 minutos
```

### 7. Testes

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
