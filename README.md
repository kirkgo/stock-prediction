

# Documentação do Projeto Stock Prediction

## Autor

- Kirk Patrick (MLET1 - Grupo 66)
- Você pode entrar em contato com o autor pelo LinkedIn: [https://www.linkedin.com/in/kirkgo/](https://www.linkedin.com/in/kirkgo/)

## Link do Video

[Vídeo de Apresentação](https://www.linkedin.com/in/kirkgo/)

# Documentação do Projeto Stock Prediction

## Sumário
- [Visão Geral](#visão-geral)
- [Requisitos](#requisitos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [API](#api)
- [Modelo LSTM](#modelo-lstm)
- [Monitoramento](#monitoramento)
- [Testes](#testes)
- [Docker](#docker)

## Visão Geral

O Stock Prediction é um projeto que implementa um modelo de Deep Learning (LSTM - Long Short-Term Memory) para predição de preços de ações. O projeto inclui uma API REST para servir as predições e um sistema completo de monitoramento usando Prometheus e Grafana.

### Principais Funcionalidades
- Predição de preços de ações usando redes neurais LSTM
- API REST para acesso às predições
- Sistema de monitoramento e métricas
- Interface para visualização de dados históricos
- Treinamento automatizado de modelos
- Cache de modelos para melhor performance
- Análise e avaliação de performance dos modelos

## Requisitos

- Python 3.11+
- Docker e Docker Compose
- Bibliotecas Python (principais):
  - TensorFlow 2.18.0+
  - FastAPI 0.115.5+
  - Pandas 2.2.3+
  - NumPy 2.0.2+
  - yfinance 0.2.49+
  - Prometheus Client 0.21.1+

## Estrutura do Projeto

```
stock-prediction-project/
├── src/
│   ├── models/         # Implementação dos modelos
│   ├── api/           # Código da API REST
│   └── utils/         # Utilidades e funções auxiliares
├── notebooks/        # Jupyter notebooks para análises
├── tests/           # Testes automatizados
├── data/            # Dados do projeto
├── models/          # Modelos salvos
│   ├── checkpoints/ # Checkpoints durante treinamento
│   ├── saved_models/# Modelos finalizados
├── configs/         # Arquivos de configuração
├── docs/           # Documentação
└── scripts/        # Scripts utilitários
```

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd stock-prediction-project
```

2. Usando Docker (recomendado):
```bash
docker-compose up -d
```

3. Instalação local (desenvolvimento):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Configuração

### Configuração do Modelo (training_config.yaml)
```yaml
training:
  start_date: '2018-01-01'
  end_date: '2024-07-20'
  epochs: 200
  batch_size: 64
  validation_split: 0.2
  early_stopping_patience: 20

model:
  sequence_length: 30
  lstm_units: [128, 64]
  dropout_rate: 0.1
  learning_rate: 0.001
```

### Variáveis de Ambiente
- `PROMETHEUS_PORT`: Porta para métricas (padrão: 8000)
- `MODEL_CACHE_SIZE`: Tamanho do cache de modelos (padrão: 5)
- `LOG_LEVEL`: Nível de logging (padrão: INFO)

## Uso

### Treinamento do Modelo

Para treinar um novo modelo:
```bash
python scripts/train_model.py --symbol AAPL --config configs/training_config.yaml
```

Parâmetros disponíveis:
- `--symbol`: Símbolo da ação (ex: AAPL)
- `--config`: Caminho para arquivo de configuração
- `--output-dir`: Diretório para salvar o modelo

## API

A API REST fornece os seguintes endpoints:

### Endpoints

#### GET /health
Verifica o status da API.

#### POST /predict
Realiza predições de preços.

Exemplo de requisição:
```json
{
    "symbol": "AAPL",
    "days": 30
}
```

#### POST /train
Inicia o treinamento de um novo modelo.

Exemplo de requisição:
```json
{
    "symbol": "AAPL",
    "epochs": 100,
    "batch_size": 32
}
```

#### GET /model/info/{symbol}
Retorna informações sobre o modelo de uma ação específica.

#### GET /historical/{symbol}
Retorna dados históricos da ação.

## Modelo LSTM

O modelo implementa uma rede neural LSTM com as seguintes características:

- Arquitetura: 3 camadas LSTM com dropout
- Features utilizadas:
  - Preço de fechamento
  - Média móvel de 5 períodos
  - Média móvel de 20 períodos
  - RSI (Relative Strength Index)
  - Volatilidade

### Processo de Treinamento

O treinamento do modelo LSTM é realizado através da classe `StockPredictor` e segue as seguintes etapas:

1. **Coleta de Dados**
   - Dados são obtidos através da API do Yahoo Finance (yfinance)
   - Período configurável através dos parâmetros `start_date` e `end_date`
   - Features extraídas automaticamente:
     - Preço de fechamento (Close)
     - Média Móvel de 5 períodos (MA5)
     - Média Móvel de 20 períodos (MA20)
     - RSI (Relative Strength Index)
     - Volatilidade (desvio padrão móvel de 20 períodos)

2. **Pré-processamento**
   - Normalização dos dados usando MinMaxScaler
   - Preenchimento de valores ausentes (forward fill e backward fill)
   - Criação de sequências temporais para treinamento
   - Divisão treino/validação (80/20 por padrão)

3. **Arquitetura do Modelo**
   ```python
   model = Sequential([
       Input(shape=input_shape),
       LSTM(units=50, return_sequences=True),
       Dropout(0.2),
       LSTM(units=50, return_sequences=True),
       Dropout(0.2),
       LSTM(units=50),
       Dropout(0.2),
       Dense(units=1)
   ])
   ```

4. **Parâmetros de Treinamento**
   - Otimizador: Adam
   - Loss Function: Mean Squared Error
   - Métricas monitoradas: MAE, MAPE
   - Early Stopping com paciência de 10 épocas
   - Model Checkpoints salvos durante o treinamento

5. **Callbacks e Monitoramento**
   - EarlyStopping: Evita overfitting
   - ModelCheckpoint: Salva melhores modelos
   - Logging detalhado do processo
   - Métricas Prometheus para monitoramento em tempo real

### Avaliação e Métricas

1. **Métricas de Performance**
   - MAE (Mean Absolute Error): Erro médio absoluto
   - RMSE (Root Mean Square Error): Raiz do erro quadrático médio
   - MAPE (Mean Absolute Percentage Error): Erro percentual médio absoluto

2. **Relatórios de Avaliação**
   ```python
   {
       "basic_metrics": {
           "mae": float,
           "rmse": float,
           "mape": float
       },
       "error_statistics": {
           "mean_error": float,
           "std_error": float,
           "median_error": float,
           "min_error": float,
           "max_error": float
       },
       "percentage_error_statistics": {
           "mean_percentage_error": float,
           "std_percentage_error": float,
           "median_percentage_error": float
       }
   }
   ```

3. **Visualizações Geradas**
   - Gráficos de loss e métricas durante treinamento
   - Comparação entre valores previstos e reais
   - Distribuição dos erros
   - Box plots de erros percentuais
   - Scatter plots de correlação

4. **Interpretação de Performance**
   - MAPE < 5%: Excelente
   - MAPE < 10%: Bom
   - MAPE < 15%: Regular
   - MAPE > 15%: Necessita Melhorias

5. **Recomendações Automáticas**
   - Baseadas nas métricas obtidas
   - Sugestões para melhorias no modelo
   - Ajustes de hiperparâmetros
   - Considerações sobre tamanho do dataset

### Métricas de Avaliação em Tempo Real
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)

## Monitoramento

### Prometheus
Métricas coletadas:
- `prediction_time_seconds`: Tempo gasto em predições
- `predictions_total`: Total de predições realizadas
- `training_time_seconds`: Tempo de treinamento
- `model_errors_total`: Total de erros do modelo

### Grafana
- Dashboard disponível em `http://localhost:3000`
- Credenciais padrão: admin/admin
- Dashboards pré-configurados para monitoramento do modelo

## Testes

Execute os testes usando:
```bash
pytest
```

Marcadores disponíveis:
- `slow`: Testes mais demorados
```bash
pytest -m "not slow"  # Executa apenas testes rápidos
```

## Docker

### Configuração dos Serviços

#### API (stock-prediction-api)
- Porta: 8001 (API REST)
- Porta: 8000 (Métricas Prometheus)
- Volumes:
  - `./models`: Modelos treinados
  - `./src`: Código fonte
  - `./scripts`: Scripts de utilidade
- Variáveis de Ambiente:
  - `PROMETHEUS_PORT=8000`
  - `API_PORT=8001`
  - `MODEL_CACHE_SIZE=5`
  - `LOG_LEVEL=INFO`
- Recursos:
  - Limites: 1.0 CPU, 2GB RAM
  - Reservas: 0.5 CPU, 1GB RAM
- Healthcheck: Verifica endpoint `/health` a cada 30 segundos

#### Prometheus
- Porta: 9090
- Volumes:
  - `prometheus_data`: Dados persistentes
  - `./configs/prometheus.yml`: Configuração
- Healthcheck: Verifica endpoint `/-/healthy` a cada 30 segundos

#### Grafana
- Porta: 3000
- Credenciais padrão: admin/admin
- Volumes:
  - `grafana_data`: Dados persistentes
  - `./configs/grafana`: Configurações e dashboards
- Plugins: grafana-piechart-panel

#### Node Exporter
- Porta: 9100
- Volumes:
  - `/proc`: Métricas do sistema (somente leitura)
  - `/sys`: Métricas do sistema (somente leitura)
  - `/`: Sistema de arquivos root (somente leitura)

### Dockerfile

O Dockerfile está otimizado para segurança e performance:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libhdf5-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    wget \
    git \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Usuário não-root e diretórios
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/models/saved_models /app/models/checkpoints \
    && chown -R appuser:appuser /app
```

### Segurança

- Execução como usuário não-root (appuser)
- Volumes montados com permissões apropriadas
- Healthchecks em todos os serviços críticos
- Controle de recursos por container
- Configurações isoladas por serviço

### Portas Utilizadas

- 8001: API REST
- 8000: Métricas Prometheus da API
- 9090: Interface Prometheus
- 3000: Interface Grafana
- 9100: Node Exporter

### Gestão de Containers

1. **Construir e Iniciar:**
   ```bash
   docker-compose up --build
   ```

2. **Parar Serviços:**
   ```bash
   docker-compose down
   ```

3. **Visualizar Logs:**
   ```bash
   docker-compose logs -f api
   docker-compose logs -f prometheus
   docker-compose logs -f grafana
   ```

4. **Reiniciar Serviço Específico:**
   ```bash
   docker-compose restart api
   ```

5. **Verificar Status:**
   ```bash
   docker-compose ps
   ```

### Volumes e Persistência

- Dados do Prometheus: `/prometheus`
- Dados do Grafana: `/var/lib/grafana`
- Modelos treinados: `/app/models`
- Configurações: `/app/configs`

### Troubleshooting Docker

1. **Problema de Permissão:**
   ```bash
   sudo chown -R 1000:1000 ./models
   sudo chown -R 1000:1000 ./configs
   ```

2. **Conflito de Portas:**
   - Verificar portas em uso: `netstat -tulpn`
   - Modificar portas no docker-compose.yml se necessário

3. **Problemas de Memória:**
   - Aumentar limites no docker-compose.yml
   - Monitorar uso: `docker stats`

4. **Container não Inicia:**
   - Verificar logs: `docker-compose logs [serviço]`
   - Verificar healthcheck: `docker inspect [container]`

## Segurança

- Execução com usuário não-root (appuser)
- Configurações de ambiente isoladas
- Healthchecks implementados
- Limitação de recursos por container

## Troubleshooting

### Sistema de Monitoramento

## Configuração do Grafana

### Estrutura de Diretórios
```
configs/
└── grafana/
    ├── provisioning/
    │   ├── dashboards/
    │   │   └── dashboard.yml
    │   └── datasources/
    │       └── datasource.yml
    └── dashboards/
        └── stock_prediction_dashboard.json
```

### Datasources
O Prometheus é configurado como fonte de dados padrão através do arquivo `datasource.yml`:
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### Dashboard Principal
O dashboard principal inclui:

1. **Métricas de Predição**
   - Tempo de processamento
   - Total de predições realizadas
   - Gráfico temporal de performance

2. **Métricas do Modelo**
   - Taxa de erro
   - Tempo de treinamento
   - Performance em tempo real

3. **Métricas do Sistema**
   - Uso de CPU e memória
   - Latência da API
   - Status dos containers

### Acesso ao Grafana
- URL: http://localhost:3000
- Credenciais padrão:
  - Usuário: admin
  - Senha: admin

### Personalização
1. Acesse o Grafana Dashboard
2. Clique em "Edit" no dashboard desejado
3. Adicione ou modifique painéis
4. Configure alertas se necessário
5. Salve as alterações

## Troubleshooting

### Problemas Comuns

1. **Erro ao carregar modelo:**
   - Verifique se existe um modelo treinado em `models/saved_models`
   - Verifique as permissões dos arquivos
   - Logs: `docker logs stock-prediction-api`

2. **Erro de memória durante treinamento:**
   - Ajuste o `batch_size` no arquivo de configuração
   - Verifique os limites de recursos no Docker
   - Monitore através do Grafana

3. **API indisponível:**
   - Verifique logs: `docker logs stock-prediction-api`
   - Verifique o status do healthcheck
   - Consulte métricas no Grafana

4. **Grafana/Prometheus não inicializa:**
   - Verifique se as portas estão disponíveis
   - Confirme as permissões dos volumes
   - Verifique a configuração do datasource

### Verificação de Logs

1. **Logs da API:**
```bash
docker logs stock-prediction-api
```

2. **Logs do Prometheus:**
```bash
docker logs prometheus
```

3. **Logs do Grafana:**
```bash
docker logs grafana
```

### Reinicialização de Serviços

1. **Reiniciar todos os serviços:**
```bash
docker-compose down
docker-compose up -d
```

2. **Reiniciar serviço específico:**
```bash
docker-compose restart <service-name>
```

3. **Reconstruir serviços:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Verificação de Métricas

1. **Prometheus Targets:**
   - Acesse: http://localhost:9090/targets
   - Verifique status dos endpoints

2. **Grafana Datasources:**
   - Acesse: http://localhost:3000/datasources
   - Verifique conexão com Prometheus

3. **Métricas em Tempo Real:**
   - Dashboard principal do Grafana
   - Alertas configurados
   - Histórico de performance
