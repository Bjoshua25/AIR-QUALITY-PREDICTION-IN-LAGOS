import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from io import BytesIO

from scripts.config import load_config
from scripts.model_training import load_latest_model, train_sarima_model, save_model, save_model_summary
from scripts.validate import load_latest_wfv_results, evaluate_forecast
from scripts.wrangle import load_combined_series

# Load configuration
cfg = load_config()
st.set_page_config(page_title="Lagos Air Quality Forecast", layout="wide")

st.title("🌍 Lagos Air Quality Forecast (PM2.5)")
st.markdown("This app forecasts PM2.5 levels using SARIMA time series models based on sensor data from OpenAfrica.")

# Sidebar
st.sidebar.header("Actions")
if st.sidebar.button("📈 Train New Model"):
    with st.spinner("Training SARIMA model..."):
        model = train_sarima_model()
        save_model(model)
        save_model_summary(model)
        st.success("Model trained and saved!")

# Tabs
tab1, tab2 = st.tabs(["📈 Forecasting", "📊 EDA"])

# ===== FORECASTING TAB =====
with tab1:
    try:
        model = load_latest_model()
        st.sidebar.success("✅ Model Loaded Successfully")
    except FileNotFoundError:
        st.error("No saved SARIMA model found. Please train the model first.")
        st.stop()

    st.subheader("🚶 Walk-Forward Validation Results")

    try:
        df_pred = load_latest_wfv_results()
        mse_wfv, mae_wfv = evaluate_forecast(df_pred["y_test"], df_pred["y_pred"])
        st.markdown(f"**WFV Mean Squared Error (MSE)**: {mse_wfv:.2f}  \n**WFV Mean Absolute Error (MAE)**: {mae_wfv:.2f}")

        df_long = pd.concat([
            pd.DataFrame({"Date": df_pred.index, "PM2.5": df_pred["y_test"], "Type": "Test"}),
            pd.DataFrame({"Date": df_pred.index, "PM2.5": df_pred["y_pred"], "Type": "Predicted"})
        ])
        fig_wfv = px.line(df_long, x="Date", y="PM2.5", color="Type", title="PM2.5: Actual vs Walk-Forward Forecast")
        st.plotly_chart(fig_wfv, use_container_width=True)

    except Exception:
        st.warning("⚠️ No walk-forward validation results found. Run it offline and save results first.")

    # Forecast Horizon
    st.markdown("### 🔮 Forecast Future PM2.5 Levels")
    forecast_options = {
        "1 Day (4 steps)": 4,
        "1 Week (28 steps)": 28,
        "1 Month (120 steps)": 120,
        "3 Months (360 steps)": 360,
        "6 Months (720 steps)": 720,
        "1 Year (1440 steps)": 1440
    }
    selected_forecast = st.radio("Select Forecast Horizon", list(forecast_options.keys()))
    steps = forecast_options[selected_forecast]

    if st.button("Run Forecast"):
        future_forecast = model.forecast(steps=steps)
        future_forecast.index.name = "Date"

        st.markdown(f"#### Forecasted PM2.5 Levels — {selected_forecast}")
        st.line_chart(future_forecast)
        st.dataframe(future_forecast.rename("Forecast").reset_index())

        # Download
        buffer = BytesIO()
        forecast_df = future_forecast.rename("Forecast").reset_index()
        forecast_df.to_csv(buffer, index=False)
        st.download_button("📥 Download Forecast as CSV", data=buffer.getvalue(), file_name="forecast_pm25.csv", mime="text/csv")

# ===== EDA TAB =====
with tab2:
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    y, *_ = load_combined_series()
    y.name = "PM2.5"

    # Radio to pick duration
    st.markdown("### Select Time Range for EDA")
    eda_ranges = {
        "1 Day": 4,
        "1 Week": 28,
        "1 Month": 120,
        "3 Months": 360,
        "6 Months": 720,
        "1 Year": 1440,
        "All Data": len(y)
    }
    selected_range = st.radio("Time Range", list(eda_ranges.keys()), index=list(eda_ranges.keys()).index("All Data"))
    n_points = eda_ranges[selected_range]
    if n_points < len(y):
        y = y[-n_points:]

    # Line Chart
    st.markdown(f"### PM2.5 Over Time — Last {selected_range}")
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(y)
    ax.set_title(f"PM2.5 Values Over Time — Last {selected_range}")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    st.pyplot(fig)

    # Rolling Stats
    st.markdown("### Rolling Mean & Standard Deviation (120-step window)")
    rolling_mean = y.rolling(window=120).mean()
    rolling_std = y.rolling(window=120).std()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(rolling_mean, label="Rolling Mean")
    ax.plot(rolling_std, label="Rolling Std")
    ax.set_title("Rolling Statistics (Window=120)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")
    ax.legend()
    st.pyplot(fig)

    # Histogram & Boxplot
    st.markdown("### Distribution and Outliers")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(y, kde=True, bins=30, color="skyblue", ax=axes[0])
    axes[0].set_title("PM2.5 Distribution")
    axes[0].set_ylabel("Frequency")
    sns.boxplot(x=y, color="lightcoral", ax=axes[1])
    axes[1].set_title("Boxplot (Outlier Detection)")
    st.pyplot(fig)

    # ACF & PACF
    st.markdown("### Autocorrelation (ACF)")
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_acf(y, ax=ax)
    st.pyplot(fig)

    st.markdown("### Partial Autocorrelation (PACF)")
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_pacf(y, ax=ax)
    st.pyplot(fig)

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "Made with ❤️ using [Streamlit](https://streamlit.io)  \n"
    "Project by [Joshua Bolaji](https://github.com/Bjoshua25/AIR-QUALITY-PREDICTION-IN-LAGOS/tree/main) | "
    "Data: [OpenAfrica](https://open.africa/dataset/sensorsafrica-airquality-archive-lagos)"
)
