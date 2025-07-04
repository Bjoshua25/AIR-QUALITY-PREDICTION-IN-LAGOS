import os
import joblib
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scripts.config import load_config
from scripts.wrangle import load_combined_series

cfg = load_config()
RESULTS_DIR = cfg["paths"]["results_folder"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load train and test data
y_train, y_test = load_combined_series()


def walk_forward_validate(train=y_train, test=y_test, order=cfg["model"]["order"], seasonal_order=cfg["model"]["seasonal_order"]):
    """
    Perform walk-forward validation using SARIMA.
    Returns a DataFrame with actual and predicted test values.
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


def save_walk_forward_results(df_pred):
    """
    Save walk-forward validation results to CSV.
    """
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    filename = f"{timestamp}_walk_forward_results.csv"
    df_pred.to_csv(os.path.join(RESULTS_DIR, filename), index=True)
    print(f"Walk-forward validation results saved as {filename}")
    return filename


def load_latest_wfv_results():
    """
    Load the most recent walk-forward validation CSV file.
    """
    pattern = os.path.join(RESULTS_DIR, "*_walk_forward_results.csv")
    wfv_files = sorted(glob.glob(pattern))

    if not wfv_files:
        raise FileNotFoundError("No walk-forward results found in results directory.")

    latest_file = wfv_files[-1]
    df = pd.read_csv(latest_file, index_col="date", parse_dates=True)
    print(f"Loaded WFV results: {os.path.basename(latest_file)}")
    return df


def evaluate_forecast(y_true, y_pred):
    """
    Compute MSE and MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae


def main():
    df_pred = walk_forward_validate()
    save_walk_forward_results(df_pred)
    mse, mae = evaluate_forecast(df_pred['y_test'], df_pred['y_pred'])
    print(f"Walk-forward Validation - MSE: {mse:.2f}, MAE: {mae:.2f}")


if __name__ == "__main__":
    main()
