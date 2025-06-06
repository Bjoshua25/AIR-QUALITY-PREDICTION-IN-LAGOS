# evaluate.py

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px


def evaluate_forecast(y_true, y_pred):
    """
    Compute evaluation metrics for forecast.

    Parameters:
        y_true (pd.Series): Actual values
        y_pred (pd.Series): Predicted values

    Returns:
        dict: Dictionary containing MSE and MAE
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"MSE": mse, "MAE": mae}


def plot_forecast(y_true, y_pred, title="Forecast vs Actual"):
    """
    Plot forecasted vs actual values.

    Parameters:
        y_true (pd.Series): Actual values
        y_pred (pd.Series): Predicted values
        title (str): Title of the plot

    Returns:
        plotly.graph_objs._figure.Figure: Line plot
    """
    df = pd.DataFrame({"y_test": y_true, "y_pred": y_pred}, index=y_true.index)
    fig = px.line(
        df,
        title=title,
        labels={"value": "PM2.5", "index": "Date"},
    )
    return fig
