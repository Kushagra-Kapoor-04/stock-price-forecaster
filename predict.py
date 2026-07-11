"""
Live prediction for any ticker.
Usage: python predict.py --ticker TCS.NS --days 30
"""
import argparse
import os
import sys
import torch
import numpy as np
import pickle
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import StockLSTM


def predict_live(ticker, days=30):
    """
    Fetch the most recent 6 months of data for `ticker`,
    load the trained LSTM, and forecast `days` into the future.

    Uses return prediction: each step predicts the daily change,
    which is added to the last known price.
    """
    model_path = 'models/lstm_model.pth'
    scaler_path = 'models/scaler.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("ERROR: No trained model found.")
        print("Run main.py first to train the model.")
        sys.exit(1)

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load model
    device = torch.device('cpu')
    model = StockLSTM()
    model.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    model.eval()

    # Fetch recent data
    print(f"Fetching latest data for {ticker}...")
    df = yf.download(ticker, period='6mo', progress=False)[['Close']]
    df.columns = ['Close']
    prices = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(prices)  # use same scaler as training

    seq = scaled.flatten()
    future = []
    lookback = 30

    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(seq[-lookback:], dtype=torch.float32) \
                     .unsqueeze(0).unsqueeze(-1)
            pred_return = model(x).item()
            # Reconstruct: new price = last price + predicted change
            pred_scaled = seq[-1] + pred_return
            future.append(pred_scaled)
            seq = np.append(seq, pred_scaled)

    future_prices = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten()

    print(f"\n{ticker} — Next {days} day forecast:")
    for i, p in enumerate(future_prices, 1):
        print(f"  Day {i:2d}: ₹{p:.2f}")

    return future_prices


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Live stock price prediction')
    ap.add_argument('--ticker', default='RELIANCE.NS',
                    help='Stock ticker symbol (default: RELIANCE.NS)')
    ap.add_argument('--days', type=int, default=30,
                    help='Number of days to forecast (default: 30)')
    args = ap.parse_args()
    predict_live(args.ticker, args.days)
