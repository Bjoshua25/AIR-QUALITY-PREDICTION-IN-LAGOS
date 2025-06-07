import streamlit as st
import pandas as pd
import plotly.express as px

from scripts.config import load_config
from scripts.model_training import load_latest_model, train_sarima_model, save_model, save_model_summary
from scripts.validate import load_latest_wfv_results, evaluate_forecast
from scripts.wrangle import load_combined_series

# Load configuration
cfg = load_config()

st.set_page_config(page_title="Lagos Air Quality Forecast", layout="wide")

st.title("🌍 Lagos Air Quality Forecast (PM2.5)")
st.markdown("This app forecasts PM2.5 levels using SARIMA time series models based on sensor data from OpenAfrica.")

# Load data
y_train, y_test = load_combined_series()

# Sidebar
st.sidebar.header("Actions")

# Option to retrain model
if st.sidebar.button("📈 Train New Model"):
    with st.spinner("Training SARIMA model..."):
        model = train_sarima_model()
        save_model(model)
        save_model_summary(model)
        st.success("Model trained and saved!")

# Load model
try:
    model = load_latest_model()
    st.sidebar.success("✅ Model Loaded Successfully")
except FileNotFoundError:
    st.error("No saved SARIMA model found. Please train the model first.")
    st.stop()

# Load WFV results instead of running
st.subheader("🚶 Walk-Forward Validation Results")

try:
    df_pred = load_latest_wfv_results()
    mse_wfv, mae_wfv = evaluate_forecast(df_pred["y_test"], df_pred["y_pred"])
    st.markdown(f"**WFV Mean Squared Error (MSE)**: {mse_wfv:.2f}  \n**WFV Mean Absolute Error (MAE)**: {mae_wfv:.2f}")

    # Plot actual vs WFV forecast
    df_long = pd.DataFrame()

    if "y_train" in df_pred:
        df_long = pd.concat([
            pd.DataFrame({"Date": df_pred.index, "PM2.5": df_pred["y_train"], "Type": "Train"}),
            pd.DataFrame({"Date": df_pred.index, "PM2.5": df_pred["y_test"], "Type": "Test"}),
            pd.DataFrame({"Date": df_pred.index, "PM2.5": df_pred["y_pred"], "Type": "Predicted"})
        ])

    fig_wfv = px.line(
        df_long,
        x="Date",
        y="PM2.5",
        color="Type",
        title="PM2.5: Train, Test, and Walk-Forward Forecast"
    )

    st.plotly_chart(fig_wfv, use_container_width=True)

except Exception as e:
    st.warning("⚠️ No walk-forward validation results found. Run it offline and save results first.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io)")
