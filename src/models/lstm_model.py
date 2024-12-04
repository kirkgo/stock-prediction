import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fetch_data(self):
        """Fetch and preprocess stock data"""
        logger.info(f"Fetching data for {self.symbol}")
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date)

        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'].values)
        df['Volatility'] = df['Close'].rolling(window=20).std()

        df = df.ffill()
        df = df.bfill()

        logger.info("Shape of data after preprocessing: %s", df.shape)

        features = ['Close', 'MA5', 'MA20', 'RSI', 'Volatility']
        dataset = df[features].values

        scaled_data = self.scaler.fit_transform(dataset)
        return scaled_data

    def prepare_sequences(self, data, seq_length=60):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 0])

        X = np.array(X)
        y = np.array(y)

        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:]
        y_val = y[train_size:]

        return X_train, X_val, y_train, y_val

    def build_model(self, input_shape):
        """Build LSTM model"""
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

        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae', 'mape']
        )

        return model

    def train(self, epochs=100, batch_size=32, seq_length=60):
        """Train the model"""
        logger.info("Starting model training")

        data = self.fetch_data()
        X_train, X_val, y_train, y_val = self.prepare_sequences(
            data, seq_length)

        self.model = self.build_model((seq_length, X_train.shape[2]))

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                'models/checkpoints/model_{epoch:02d}.keras',
                save_best_only=True
            )
        ]

        # Treina
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)

        predictions_inv = self.scaler.inverse_transform(
            np.concatenate([predictions, np.zeros(
                (len(predictions), 4))], axis=1)
        )[:, 0]

        y_test_inv = self.scaler.inverse_transform(
            np.concatenate(
                [y_test.reshape(-1, 1), np.zeros((len(y_test), 4))], axis=1)
        )[:, 0]

        mae = np.mean(np.abs(y_test_inv - predictions_inv))
        rmse = np.sqrt(np.mean((y_test_inv - predictions_inv) ** 2))
        mape = np.mean(
            np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    def predict(self, days=30):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("No model is loaded. Cannot make predictions.")

        logger.info(f"Predicting next {days} days")

        data = self.fetch_data()
        last_sequence = data[-60:]

        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(days):
            current_batch = current_sequence[-60:].reshape((1, 60, 5))

            next_pred = self.model.predict(current_batch)[0]
            predictions.append(next_pred[0])

            new_row = np.zeros((1, 5))
            new_row[0, 0] = next_pred[0]
            current_sequence = np.vstack([current_sequence, new_row])

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(
            np.concatenate([predictions, np.zeros(
                (len(predictions), 4))], axis=1)
        )[:, 0]

        return predictions

    def save_model(self, filepath):
        """Save the model"""
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.error("No model to save")
            raise ValueError("Cannot save model: No model is defined")

    def load_model(self, filepath):
        """Load a saved model"""
        try:
            self.model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI using numpy operations"""
        prices = np.array(prices).flatten()

        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period]) if len(gains) > 0 else 0
        avg_loss = np.mean(losses[:period]) if len(losses) > 0 else 0

        rsi = np.zeros_like(prices)

        if avg_loss == 0:
            rsi[:period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[:period] = 100 - (100 / (1 + rs))

        for i in range(period, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100

        return rsi
