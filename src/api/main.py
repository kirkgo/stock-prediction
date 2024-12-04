from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import logging
import time
from prometheus_client import start_http_server, Summary, Counter, Histogram
import os
import sys
from pathlib import Path
import numpy as np
import yfinance as yf
import pandas as pd

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    Metrics,
    TrainModelRequest,
    ModelInfoResponse,
    HistoricalDataResponse
)
from src.models.lstm_model import StockPredictor

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas Prometheus
PREDICTION_TIME = Histogram(
    'prediction_time_seconds', 'Time spent making predictions')
PREDICTIONS_COUNTER = Counter(
    'predictions_total', 'Total number of predictions made', ['symbol'])
TRAINING_TIME = Histogram('training_time_seconds',
                          'Time spent training models')
MODEL_ERRORS = Counter('model_errors_total',
                       'Total number of model errors', ['type'])

app = FastAPI(
    title="Stock Price Prediction API",
    description="API para predição de preços de ações usando LSTM",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache para os modelos
model_cache = {}


def get_predictor(symbol: str, force_new: bool = False):
    """Obtém ou cria uma instância do StockPredictor"""
    if force_new or symbol not in model_cache:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        predictor = StockPredictor(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        model_dir = Path('models/saved_models')
        if not force_new and model_dir.exists():
            models = list(model_dir.glob(f'{symbol}_model_*.keras'))
            if models:
                latest_model = max(models, key=lambda x: x.stat().st_mtime)
                logger.info(f"Carregando modelo: {latest_model}")
                predictor.load_model(str(latest_model))

        model_cache[symbol] = predictor
    return model_cache[symbol]


async def train_model_task(symbol: str, epochs: int, batch_size: int):
    """Tarefa em background para treinar o modelo"""
    try:
        with TRAINING_TIME.time():
            predictor = get_predictor(symbol, force_new=True)
            predictor.train(epochs=epochs, batch_size=batch_size)

            # Salva o modelo treinado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = Path('models/saved_models')
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{symbol}_model_{timestamp}.keras"
            predictor.save_model(str(model_path))

    except Exception as e:
        MODEL_ERRORS.labels(type='training').inc()
        logger.error(f"Erro no treinamento do modelo: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Inicializa o servidor de monitoramento na inicialização"""
    try:
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", 8000))
        start_http_server(prometheus_port)
        logger.info(f"Servidor de monitoramento iniciado na porta {prometheus_port}")
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor de monitoramento: {str(e)}")


@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """Prediz preços futuros de ações"""
    try:
        logger.info(f"Requisição de predição recebida para {request.symbol}")
        start_time = time.time()

        predictor = get_predictor(request.symbol)

        with PREDICTION_TIME.time():
            predictions = predictor.predict(days=request.days)

        PREDICTIONS_COUNTER.labels(symbol=request.symbol).inc()

        dates = [
            (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(request.days)
        ]

        prediction_results = [
            {"date": date, "price": float(price)}
            for date, price in zip(dates, predictions)
        ]

        processing_time = time.time() - start_time

        return PredictionResponse(
            symbol=request.symbol,
            predictions=prediction_results,
            metrics=Metrics(processing_time=processing_time)
        )

    except Exception as e:
        MODEL_ERRORS.labels(type='prediction').inc()
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Inicia o treinamento de um novo modelo em background"""
    try:
        background_tasks.add_task(
            train_model_task,
            request.symbol,
            request.epochs,
            request.batch_size
        )

        return {
            "message": f"Treinamento iniciado para {request.symbol}",
            "status": "training"
        }
    except Exception as e:
        logger.error(f"Erro ao iniciar treinamento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info/{symbol}", response_model=ModelInfoResponse)
async def get_model_info(symbol: str):
    """Obtém informações sobre o modelo para um símbolo específico"""
    try:
        model_dir = Path('models/saved_models')
        models = list(model_dir.glob(f'{symbol}_model_*.keras'))

        if not models:
            return ModelInfoResponse(
                symbol=symbol,
                model_exists=False,
                last_training=None,
                total_models=0
            )

        latest_model = max(models, key=lambda x: x.stat().st_mtime)
        last_training = datetime.fromtimestamp(latest_model.stat().st_mtime)

        return ModelInfoResponse(
            symbol=symbol,
            model_exists=True,
            last_training=last_training,
            total_models=len(models)
        )
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical/{symbol}", response_model=HistoricalDataResponse)
async def get_historical_data(symbol: str, days: int = 30):
    """Obtém dados históricos para uma ação"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        historical_data = [
            {
                "date": date.strftime('%Y-%m-%d'),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            }
            for date, row in df.iterrows()
        ]

        return HistoricalDataResponse(
            symbol=symbol,
            data=historical_data,
            period_start=start_date.strftime('%Y-%m-%d'),
            period_end=end_date.strftime('%Y-%m-%d')
        )
    except Exception as e:
        logger.error(f"Erro ao obter dados históricos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
