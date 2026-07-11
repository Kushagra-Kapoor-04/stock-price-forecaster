import numpy as np
import matplotlib
matplotlib.use('Agg')        # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import os
import torch

OUTPUTS_DIR = 'outputs'


def compute_metrics(actual, predicted, name='Model'):
    """
    Compute and print RMSE, MAE, and MAPE.

    All values should be in original price scale (after inverse_transform).
    """
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"\n{name}:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return {'rmse': rmse, 'mae': mae, 'mape': mape}


def plot_predictions(actual, lstm_pred, arima_pred, dates=None):
    """Plot LSTM vs ARIMA predictions against actual prices."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    plt.figure(figsize=(14, 5))
    x_axis = dates if dates is not None else range(len(actual))
    plt.plot(x_axis, actual,     label='Actual',  linewidth=1.5)
    plt.plot(x_axis, lstm_pred,  label='LSTM',    linewidth=1.2)
    plt.plot(x_axis, arima_pred, label='ARIMA',   linewidth=1.2, linestyle='--')
    plt.title('LSTM vs ARIMA — Stock Price Prediction')
    plt.xlabel('Trading Days')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/lstm_vs_arima.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/lstm_vs_arima.png")


def plot_loss(history):
    """Plot training and validation loss curves."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'],   label='Val', linestyle='--')
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved outputs/loss_curve.png")


def plot_future_forecast(last_sequence, model, scaler, device,
                         lookback=30, days=30):
    """
    Autoregressive future forecast with return-prediction model.

    At each step:
      1. Feed last `lookback` scaled prices to the model
      2. Model predicts the daily CHANGE (return) in scaled space
      3. New scaled price = last scaled price + predicted change
      4. Append and repeat

    Error compounds at each step — forecasts beyond ~5-10 days are
    inherently uncertain.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    model.eval()
    seq = last_sequence.copy()
    future_preds_scaled = []

    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(seq[-lookback:], dtype=torch.float32) \
                     .unsqueeze(0).unsqueeze(-1).to(device)
            pred_return = model(x).item()
            # Reconstruct: new price = last price + predicted change
            pred_price_scaled = seq[-1] + pred_return
            future_preds_scaled.append(pred_price_scaled)
            seq = np.append(seq, pred_price_scaled)

    # Inverse-transform back to real prices
    future_prices = scaler.inverse_transform(
        np.array(future_preds_scaled).reshape(-1, 1)
    ).flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(future_prices)), future_prices,
             marker='o', markersize=3, color='#2196F3')
    plt.title(f'LSTM Forecast — Next {days} Trading Days')
    plt.xlabel('Days from today')
    plt.ylabel('Predicted Price (INR)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/future_forecast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved outputs/future_forecast.png")
    return future_prices
