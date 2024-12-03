from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, GOOGL)")
    days: Optional[int] = Field(
        30, ge=1, le=365, description="Number of days to predict")
    start_date: Optional[str] = Field(
        None, description="Start date for historical data")
    end_date: Optional[str] = Field(
        None, description="End date for historical data")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "days": 30
            }
        }


class Metrics(BaseModel):
    processing_time: float = Field(...,
                                   description="Time taken to process request in seconds")


class PredictionResponse(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    predictions: List[float] = Field(...,
                                     description="List of predicted prices")
    dates: List[str] = Field(..., description="List of dates for predictions")
    metrics: Optional[Metrics] = Field(None, description="Performance metrics")
