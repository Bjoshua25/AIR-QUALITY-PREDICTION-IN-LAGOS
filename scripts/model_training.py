import os
import joblib
import glob
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import load_config
from wrangle import load_combined_series

# load train and test split
y_train, y_test = load_combined_series()

# Forecast using model
start_date = y_test.index.min()
end_date = y_test.index.max()

# load config
cfg = load_config()
order = cfg["model"]["order"]
seasonal_order = cfg["model"]["seasonal_order"]


def train_sarima_model(train_series=y_train, order=order, seasonal_order=seasonal_order):
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


def save_model(model, result_dir=cfg["paths"]["models_folder"], model_name="sarima_model"):
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


def save_model_summary(model, result_dir=cfg["paths"]["results_folder"], model_name="sarima_summary"):
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


def load_latest_model():
    """
    Load the latest saved SARIMA model from the results directory.

    Returns:
        SARIMAXResultsWrapper: Trained SARIMA model.
    """
    pattern = os.path.join(cfg["paths"]["models_folder"], "*_sarima_model.pkl")
    model_files = sorted(glob.glob(pattern))

    if not model_files:
        raise FileNotFoundError("No SARIMA models found in results directory.")

    latest_model_file = model_files[-1]
    print(f"Loaded model: {os.path.basename(latest_model_file)}")
    return joblib.load(latest_model_file)

    

def forecast_with_model(model, y_test=y_test):
    
    """
    Forecast future values using the SARIMA model and test index.

    Parameters:
        model: Fitted SARIMAXResults model object.
        y_test (pd.Series): Test series with datetime index.

    Returns:
        pd.Series: Forecasted PM2.5 values aligned with test index.
    """
    return model.predict(start=y_test.index.min(), end=y_test.index.max())

