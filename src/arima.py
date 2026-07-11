import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


def run_arima(train_prices, test_prices, order=(5, 1, 0)):
    """
    Walk-forward ARIMA baseline.

    ARIMA(p, d, q):
      p=5 : use last 5 observations (autoregressive terms)
      d=1 : first difference to make the series stationary
      q=0 : no moving-average terms

    Walk-forward validation:
      Fitting ARIMA once and predicting the entire test set is
      unrealistic — in production you'd retrain as new data arrives.
      Walk-forward simulates this:
        1. Predict one day ahead
        2. Observe the actual value
        3. Append actual to history
        4. Repeat

    Args:
        train_prices: Array of training prices (original scale)
        test_prices:  Array of test prices (original scale)
        order:        ARIMA (p, d, q) order

    Returns:
        np.ndarray of ARIMA predictions aligned with test_prices.
    """
    history = list(train_prices.flatten())
    predictions = []

    print(f"Running ARIMA{order} walk-forward on {len(test_prices)} days...")
    for i, actual in enumerate(test_prices.flatten()):
        model = ARIMA(history, order=order)
        fit = model.fit()
        pred = fit.forecast(steps=1)[0]
        predictions.append(pred)
        history.append(actual)  # add actual value for next prediction

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_prices)} done")

    return np.array(predictions)
