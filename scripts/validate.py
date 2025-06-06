import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def walk_forward_validate(train, test, order, seasonal_order):
    """
    Perform walk-forward validation on SARIMA.
    Returns a DataFrame of predictions and test values.
    """
    preds = []
    history = train.copy()

    for t in range(len(test)):
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        preds.append(yhat)
        history = pd.concat([history, pd.Series(test.iloc[t], index=[test.index[t]])])

    preds_series = pd.Series(preds, index=test.index)
    df_pred = pd.DataFrame({
        'y_test': test,
        'y_pred': preds_series
    })
    return df_pred

def evaluate_forecast(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, mae

def save_walk_forward_results(df_pred):
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    filename = f"{timestamp}_walk_forward_results.csv"
    df_pred.to_csv(os.path.join(RESULTS_DIR, filename), index=True)
    print(f"Walk-forward validation results saved as {filename}")
    return filename

def main(train, test):
    order = (2, 1, 3)
    seasonal_order = (1, 0, 1, 28)

    df_pred = walk_forward_validate(train, test, order, seasonal_order)
    mse, mae = evaluate_forecast(df_pred['y_test'], df_pred['y_pred'])
    print(f"Walk-forward Validation - MSE: {mse:.2f}, MAE: {mae:.2f}")

    save_walk_forward_results(df_pred)
    return df_pred, mse, mae
