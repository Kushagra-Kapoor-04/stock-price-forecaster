import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

DATA_DIR = 'data'


def fetch_stock_data(ticker='RELIANCE.NS',
                     start='2018-01-01',
                     end='2024-12-31'):
    """
    Download historical stock data from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g. 'RELIANCE.NS' for NSE, 'AAPL' for US)
        start:  Start date string (YYYY-MM-DD)
        end:    End date string   (YYYY-MM-DD)

    Returns:
        DataFrame with a single 'Close' column, indexed by date.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Fetching {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Keep only the closing price, drop any NaN rows
    df = df[['Close']].dropna()
    df.columns = ['Close']
    print(f"Downloaded {len(df)} trading days")
    return df


def preprocess(df, test_split=0.2):
    """
    Scale prices to [0, 1] and split chronologically.

    Why MinMaxScaler (not StandardScaler)?
      LSTM's sigmoid/tanh activations saturate with large values.
      Scaling to [0, 1] keeps gradients stable.

    Why no shuffling?
      Shuffling breaks temporal order — that's data leakage.
      Always split chronologically for time series.

    Returns:
        train_scaled, test_scaled, fitted scaler
    """
    prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    split = int(len(scaled) * (1 - test_split))
    train = scaled[:split]
    test = scaled[split:]

    print(f"Train: {len(train)} | Test: {len(test)}")
    return train, test, scaler


def create_sequences(data, lookback=30, predict_returns=False):
    """
    Sliding window: each sample is `lookback` days → next day.

    Args:
        data:            Scaled price array of shape (N, 1) or (N,)
        lookback:        Number of past days to use as input
        predict_returns: If True, target = daily change (data[i] - data[i-1])
                         instead of absolute price. This makes the LSTM
                         predict the price CHANGE anchored to the last
                         known price, similar to ARIMA's d=1 differencing.

    Returns:
        X: np.ndarray of shape (num_samples, lookback)
        y: np.ndarray of shape (num_samples,)
    """
    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        if predict_returns:
            # Target = daily change in scaled space
            y.append(data[i, 0] - data[i - 1, 0])
        else:
            y.append(data[i, 0])
    return np.array(X), np.array(y)
