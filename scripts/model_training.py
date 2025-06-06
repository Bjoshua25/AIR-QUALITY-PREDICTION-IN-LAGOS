import os
import joblib
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_sarima_model(train_series, order=(2, 1, 3), seasonal_order=(1, 0, 1, 28)):
    """
    Trains a SARIMA model on the given training series.

    Parameters:
    -----------
    train_series : pd.Series
        Time series data for training.
    order : tuple
        ARIMA order (p, d, q).
    seasonal_order : tuple
        Seasonal order (P, D, Q, m).

    Returns:
    --------
    model_fit : SARIMAXResultsWrapper
        Trained SARIMA model.
    """
    model = SARIMAX(train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    model_fit = model.fit()
    return model_fit


def save_model(model, result_dir="../results/", model_name="sarima_model"):
    """
    Saves the model to the result directory with a timestamp.

    Parameters:
    -----------
    model : SARIMAXResultsWrapper
        Trained SARIMA model.
    result_dir : str
        Directory to save the model.
    model_name : str
        Prefix for the model filename.
    """
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    model_filename = f"{timestamp}_{model_name}.pkl"
    joblib.dump(model, os.path.join(result_dir, model_filename))
    print(f"Model saved to {os.path.join(result_dir, model_filename)}")


def save_model_summary(model, result_dir="../results/", model_name="sarima_summary"):
    """
    Saves the model summary to a text file.

    Parameters:
    -----------
    model : SARIMAXResultsWrapper
        Trained SARIMA model.
    result_dir : str
        Directory to save the summary.
    model_name : str
        Prefix for the summary filename.
    """
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    summary_filename = f"{timestamp}_{model_name}.txt"
    with open(os.path.join(result_dir, summary_filename), "w") as f:
        f.write(str(model.summary()))
    print(f"Model summary saved to {os.path.join(result_dir, summary_filename)}")
