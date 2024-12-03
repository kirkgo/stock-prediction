from src.api.schemas import PredictionRequest, PredictionResponse, Metrics
from src.models.lstm_model import StockPredictor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import logging
import time
from prometheus_client import start_http_server, Summary, Counter, Histogram
import os
import sys
from pathlib import Path
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_TIME = Histogram(
    'prediction_time_seconds', 'Time spent making predictions')
PREDICTIONS_COUNTER = Counter(
    'predictions_total', 'Total number of predictions made', ['symbol'])

app = FastAPI(
    title="Stock Price Prediction API",
    description="API para predição de preços de ações usando LSTM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_cache = {}


def get_predictor(symbol: str):
    """Get or create a StockPredictor instance"""
    if symbol not in model_cache:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        predictor = StockPredictor(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        model_dir = Path('models/saved_models')
        model_loaded = False

        if model_dir.exists():
            models = list(model_dir.glob(f'{symbol}_model_*.keras'))
            if models:
                try:
                    latest_model = max(models, key=lambda x: x.stat().st_mtime)
                    logger.info(f"Carregando modelo: {latest_model}")
                    predictor.load_model(str(latest_model))
                    model_loaded = True
                except Exception as e:
                    logger.warning(
                        f"Não foi possível carregar o modelo salvo: {str(e)}")

        # Se não conseguiu carregar um modelo, treina um novo
        if not model_loaded:
            logger.info(f"Treinando novo modelo para {symbol}")
            # Treinamento rápido inicial
            predictor.train(epochs=50, batch_size=32)

            # Salva o modelo treinado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"{symbol}_model_{timestamp}.keras"
            model_dir.mkdir(parents=True, exist_ok=True)
            predictor.save_model(str(model_path))

        model_cache[symbol] = predictor
    return model_cache[symbol]


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup"""
    start_http_server(8000)
    logger.info("Monitoring server started on port 8000")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# @app.post("/predict", response_model=PredictionResponse)
# async def predict_prices(request: PredictionRequest):
#     """Predict future stock prices"""
#     try:
#         logger.info(f"Received prediction request for {request.symbol}")
#         start_time = time.time()

#         predictor = get_predictor(request.symbol)

#         # Faz as predições
#         with PREDICTION_TIME.time():
#             predictions = predictor.predict(days=request.days)

#         if not isinstance(predictions, (list, np.ndarray)):
#             raise ValueError("Predictions must be a list or numpy array")

#         PREDICTIONS_COUNTER.labels(symbol=request.symbol).inc()

#         if isinstance(predictions, np.ndarray):
#             predictions = predictions.tolist()

#         start_date = datetime.now()
#         dates = [
#             (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
#             for i in range(request.days)
#         ]

#         processing_time = time.time() - start_time
#         logger.info(f"Prediction completed in {processing_time:.2f} seconds")

#         return PredictionResponse(
#             symbol=request.symbol,
#             predictions=predictions,
#             dates=dates,
#             metrics=Metrics(processing_time=processing_time)
#         )

#     except Exception as e:
#         logger.error(f"Error making prediction: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict", response_model=dict)  # Alterado para JSON genérico
async def predict_prices(request: PredictionRequest):
    """Predict future stock prices"""
    try:
        logger.info(f"Received prediction request for {request.symbol}")
        start_time = time.time()

        predictor = get_predictor(request.symbol)

        # Faz as predições
        with PREDICTION_TIME.time():
            predictions = predictor.predict(days=request.days)

        if not isinstance(predictions, (list, np.ndarray)):
            raise ValueError("Predictions must be a list or numpy array")

        PREDICTIONS_COUNTER.labels(symbol=request.symbol).inc()

        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        # Gera as datas associadas às predições
        start_date = datetime.now()
        dates = [
            (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(request.days)
        ]

        # Combina datas e previsões
        results = [
            {"date": date, "predicted_price": round(price, 2)}
            for date, price in zip(dates, predictions)
        ]

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Prediction completed in {processing_time} seconds")

        return {
            "symbol": request.symbol,
            "predictions": results,
            "metrics": {
                "processing_time": processing_time
            }
        }

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info/{symbol}")
async def get_model_info(symbol: str):
    """Get information about the model for a specific symbol"""
    try:
        predictor = get_predictor(symbol)
        return {
            "symbol": symbol,
            "model_loaded": predictor.model is not None,
            "last_training": None
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
