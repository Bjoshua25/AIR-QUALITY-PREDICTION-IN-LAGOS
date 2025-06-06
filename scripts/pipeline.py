import pandas as pd
import os
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

from scripts.wrangle import wrangle
from scripts.model import train_sarima, forecast
from scripts.validate import walk_forward_validation
from scripts.evaluate import evaluate_forecast, plot_forecast


def main():
    # File paths
    data_path = "data/Air_Quality_Lagos_Combined.csv"
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)

    # Load and resample data
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    df = df.asfreq("6H")
    df = df.interpolate(method="time")

    # Train/test split
    train = df.loc[:"2024-11-24"]
    test = df.loc["2024-11-25":]

    # Train SARIMA model
    sarima_result = train_sarima(train, order=(2, 1, 3), seasonal_order=(1, 0, 1, 28))

    # Initial prediction on test set
    y_pred = forecast(sarima_result, test.index.min(), test.index.max())

    # Evaluation
    mse, mae = evaluate_forecast(test, y_pred)
    print(f"SARIMA Forecast -> MSE: {mse:.2f}, MAE: {mae:.2f}")

    # Walk-forward validation
    y_pred_wfv = walk_forward_validation(train, test, order=(2, 1, 3), seasonal_order=(1, 0, 1, 28))

    # Save outputs
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    model_path = os.path.join(result_dir, f"{timestamp}_sarima_model.pkl")
    wfv_path = os.path.join(result_dir, f"{timestamp}_walk_forward_results.csv")
    summary_path = os.path.join(result_dir, f"{timestamp}_sarima_summary.txt")

    joblib.dump(sarima_result, model_path)

    pd.DataFrame({
        "y_test": test.values.flatten(),
        "y_pred": y_pred_wfv.values
    }, index=test.index).to_csv(wfv_path)

    with open(summary_path, "w") as f:
        f.write(str(sarima_result.summary()))

    # Plot
    plot_forecast(test, y_pred_wfv, title="Walk-Forward Validation Prediction vs Actual")


if __name__ == "__main__":
    main()
