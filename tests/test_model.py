from src.models.lstm_model import StockPredictor
import os
from datetime import datetime, timedelta
import numpy as np
import pytest


@pytest.fixture
def stock_predictor():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    return StockPredictor(
        symbol='AAPL',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )


def test_stock_predictor_initialization():
    """Test StockPredictor initialization"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    predictor = StockPredictor(
        symbol='AAPL',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    assert predictor.symbol == 'AAPL'
    assert predictor.model is None
    assert predictor.scaler is not None


def test_fetch_data(stock_predictor):
    """Test data fetching and preprocessing"""
    data = stock_predictor.fetch_data()
    assert isinstance(data, np.ndarray)
    assert data.shape[1] == 5
    assert not np.isnan(data).any()
    assert not np.isinf(data).any()


def test_prepare_sequences(stock_predictor):
    """Test sequence preparation for LSTM"""
    data = stock_predictor.fetch_data()
    X_train, X_val, y_train, y_val = stock_predictor.prepare_sequences(data)

    # Verificar dimensões
    assert len(X_train.shape) == 3
    assert len(y_train.shape) == 1
    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0

    # Verificar divisão treino/validação
    assert len(X_train) > len(X_val)
    assert X_train.shape[1:] == X_val.shape[1:]


def test_build_model(stock_predictor):
    """Test model building"""
    model = stock_predictor.build_model((60, 5))
    assert model is not None

    config = model.get_config()
    assert len(config['layers']) > 0

    assert config['layers'][0]['class_name'] == 'InputLayer'
    assert config['layers'][-1]['class_name'] == 'Dense'


@pytest.mark.slow
def test_model_training(stock_predictor):
    """Test complete model training"""
    data = stock_predictor.fetch_data()
    history = stock_predictor.train(epochs=2, batch_size=32)

    assert history is not None
    assert 'loss' in history.history
    assert 'val_loss' in history.history
    assert len(history.history['loss']) > 0


def test_predict(stock_predictor):
    """Test prediction functionality"""
    data = stock_predictor.fetch_data()
    X_train, X_val, y_train, y_val = stock_predictor.prepare_sequences(data)
    stock_predictor.model = stock_predictor.build_model((60, 5))
    stock_predictor.model.fit(
        X_train, y_train, epochs=1, batch_size=32, verbose=0)

    days_to_test = [5, 10, 30]
    for days in days_to_test:
        predictions = stock_predictor.predict(days=days)
        assert len(predictions) == days
        assert all(isinstance(x, (int, float)) for x in predictions)
        assert not np.isnan(predictions).any()


def test_model_save_load(stock_predictor, tmp_path):
    """Test model saving and loading"""
    data = stock_predictor.fetch_data()
    X_train, X_val, y_train, y_val = stock_predictor.prepare_sequences(data)
    stock_predictor.model = stock_predictor.build_model((60, 5))
    stock_predictor.model.fit(
        X_train, y_train, epochs=1, batch_size=32, verbose=0)

    model_path = tmp_path / "test_model.keras"
    stock_predictor.save_model(model_path)
    assert os.path.exists(model_path)

    new_predictor = StockPredictor('AAPL', '2020-01-01', '2021-01-01')
    new_predictor.load_model(model_path)
    assert new_predictor.model is not None

    predictions = new_predictor.predict(days=5)
    assert len(predictions) == 5


def test_evaluate(stock_predictor):
    """Test model evaluation metrics"""
    data = stock_predictor.fetch_data()
    X_train, X_test, y_train, y_test = stock_predictor.prepare_sequences(data)

    stock_predictor.model = stock_predictor.build_model((60, 5))
    stock_predictor.model.fit(
        X_train, y_train, epochs=1, batch_size=32, verbose=0)

    metrics = stock_predictor.evaluate(X_test, y_test)

    required_metrics = ['MAE', 'RMSE', 'MAPE']
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
        assert metrics[metric] >= 0


@pytest.mark.parametrize("invalid_symbol", ['', 'INVALID', '123'])
def test_invalid_stock_symbol(invalid_symbol):
    """Test handling of invalid stock symbols"""
    with pytest.raises(Exception):
        predictor = StockPredictor(
            symbol=invalid_symbol,
            start_date='2020-01-01',
            end_date='2021-01-01'
        )
        predictor.fetch_data()
