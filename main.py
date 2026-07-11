"""
Stock Price Forecaster — Full Pipeline
=======================================
Fetches live data → trains LSTM (return prediction) → runs ARIMA baseline
→ compares metrics → plots.

Key design decisions:
  - Return prediction: LSTM predicts daily CHANGE, not absolute price.
    This anchors each prediction to the last known price, matching
    ARIMA's d=1 differencing advantage.
  - Full-data sequences: test windows include training-period context
    at the boundary, so no test days are wasted as warmup.

Usage:
    python main.py
"""
import os
import sys
import torch
import numpy as np
import pickle

# Ensure the project root is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import fetch_stock_data, preprocess, create_sequences
from src.dataset import get_loaders
from src.model import StockLSTM
from src.train import train_model
from src.arima import run_arima
from src.evaluate import (compute_metrics, plot_predictions,
                           plot_loss, plot_future_forecast)

# ── Configuration ──────────────────────────────────────────────────
TICKER   = 'RELIANCE.NS'
LOOKBACK = 30
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models',  exist_ok=True)

    # ── Step 1: Fetch data ─────────────────────────────────────────
    print("Step 1/6 — Fetching data...")
    df = fetch_stock_data(TICKER)

    # ── Step 2: Preprocess ─────────────────────────────────────────
    print("\nStep 2/6 — Preprocessing...")
    train_scaled, test_scaled, scaler = preprocess(df)

    # ── Step 3: Create sequences from FULL data ────────────────────
    # Critical: build sequences from train+test combined so that
    # test windows at the boundary include training-period context.
    # This also means we predict ALL test days (no warmup waste).
    print("\nStep 3/6 — Creating sequences (return prediction)...")
    full_scaled = np.vstack([train_scaled, test_scaled])
    X_all, y_all = create_sequences(full_scaled, LOOKBACK,
                                     predict_returns=True)

    # Split: train sequences have targets in the training period
    # y[i] corresponds to full_scaled[i + LOOKBACK]
    # Training period ends at index len(train_scaled) - 1
    n_train_seq = len(train_scaled) - LOOKBACK
    X_train, y_train = X_all[:n_train_seq], y_all[:n_train_seq]
    X_test,  y_test  = X_all[n_train_seq:], y_all[n_train_seq:]
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    train_loader, test_loader = get_loaders(X_train, y_train,
                                             X_test,  y_test)

    # ── Step 4: Train LSTM ─────────────────────────────────────────
    print(f"\nStep 4/6 — Training LSTM (return prediction) on {DEVICE}...")
    model = StockLSTM(input_size=1, hidden_size=64,
                      num_layers=2, dropout=0.2)
    history = train_model(model, train_loader, test_loader,
                          DEVICE, num_epochs=100, lr=0.001)
    plot_loss(history)

    # Save model + scaler
    torch.save(model.state_dict(), 'models/lstm_model.pth')
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved models/lstm_model.pth + models/scaler.pkl")

    # ── Step 5: Evaluate LSTM ──────────────────────────────────────
    print("\nStep 5/6 — Evaluating LSTM...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = model(X_batch.to(DEVICE)).cpu().numpy()
            all_preds.extend(pred)

    # Reconstruct prices: predicted_price = last_actual + predicted_return
    lstm_pred_returns = np.array(all_preds).flatten()
    last_actual_scaled = X_test[:, -1]   # last price in each input window
    lstm_pred_scaled = last_actual_scaled + lstm_pred_returns
    lstm_pred = scaler.inverse_transform(
        lstm_pred_scaled.reshape(-1, 1)
    ).flatten()

    # Actual prices: actual = last_actual + actual_return
    actual_scaled = last_actual_scaled + y_test
    actual = scaler.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).flatten()

    # ── Step 6: Run ARIMA baseline ─────────────────────────────────
    print("\nStep 6/6 — Running ARIMA baseline...")
    train_prices = scaler.inverse_transform(train_scaled).flatten()
    arima_pred = run_arima(train_prices, actual)

    # ── Metrics comparison ─────────────────────────────────────────
    lstm_metrics  = compute_metrics(actual, lstm_pred,  'LSTM')
    arima_metrics = compute_metrics(actual, arima_pred, 'ARIMA')

    print("\n" + "=" * 40)
    winner = "LSTM" if lstm_metrics['rmse'] < arima_metrics['rmse'] else "ARIMA"
    print(f"Winner by RMSE: {winner}")
    print("=" * 40)

    # ── Plots ──────────────────────────────────────────────────────
    plot_predictions(actual, lstm_pred, arima_pred)

    # Future forecast uses the last portion of scaled data
    plot_future_forecast(full_scaled.flatten(), model, scaler,
                         DEVICE, LOOKBACK)

    print("\nDone. Check outputs/ for plots.")


if __name__ == '__main__':
    main()
