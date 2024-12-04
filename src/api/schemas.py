from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Símbolo da ação (ex: AAPL, GOOGL)")
    days: int = Field(default=30, ge=1, le=365,
                      description="Número de dias para predição")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "days": 30
            }
        }


class TrainModelRequest(BaseModel):
    symbol: str = Field(...,
                        description="Símbolo da ação para treinar o modelo")
    epochs: int = Field(default=100, ge=1,
                        description="Número de épocas de treinamento")
    batch_size: int = Field(
        default=32, ge=1, description="Tamanho do batch para treinamento")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "epochs": 100,
                "batch_size": 32
            }
        }


class Metrics(BaseModel):
    processing_time: float = Field(...,
                                   description="Tempo de processamento em segundos")


class PriceData(BaseModel):
    date: str = Field(..., description="Data da predição")
    price: float = Field(..., description="Preço predito")


class PredictionResponse(BaseModel):
    symbol: str = Field(..., description="Símbolo da ação")
    predictions: List[PriceData] = Field(..., description="Lista de predições")
    metrics: Optional[Metrics] = Field(
        None, description="Métricas de performance")


class ModelInfoResponse(BaseModel):
    symbol: str = Field(..., description="Símbolo da ação")
    model_exists: bool = Field(...,
                               description="Indica se existe um modelo treinado")
    last_training: Optional[datetime] = Field(
        None, description="Data do último treinamento")
    total_models: int = Field(...,
                              description="Número total de modelos salvos")


class HistoricalData(BaseModel):
    date: str = Field(..., description="Data")
    close: float = Field(..., description="Preço de fechamento")
    volume: int = Field(..., description="Volume de negociação")


class HistoricalDataResponse(BaseModel):
    symbol: str = Field(..., description="Símbolo da ação")
    data: List[HistoricalData] = Field(..., description="Dados históricos")
    period_start: str = Field(..., description="Início do período")
    period_end: str = Field(..., description="Fim do período")
