# Air Quality Prediction in Lagos

This project focuses on forecasting air quality in Lagos, Nigeria, by predicting PM2.5 levels using time series modeling techniques — primarily SARIMA. It includes full preprocessing, model training, evaluation, and walk-forward validation pipelines. The data source is Sensor Africa, accessed via OpenAfrica.

---

## 📁 Project Structure
.
├── config.yaml                  # Central configuration file
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
│
├── data/
│   ├── monthly\_record/          # Raw monthly CSVs from Sensor Africa
│   └── combined\_dataset.csv     # Merged and resampled dataset
│
├── models/                      # Trained SARIMA models (.pkl files)
├── results/                     # Evaluation results, model summaries, WFV CSVs
├── notebooks/
│   ├── exploratory.ipynb        # Exploratory data analysis
│   ├── combine\_dataset.ipynb    # Dataset merging and cleaning
│   └── models.ipynb             # Model experimentation
│
├── scripts/
│   ├── config.py                # Load YAML configuration
│   ├── wrangle.py               # Data loading, cleaning, merging
│   ├── model\_training.py        # SARIMA training, saving, forecasting
│   ├── evaluate.py              # Evaluation metrics and visualizations
│   └── validate.py              # Walk-forward validation framework


---

## 🔍 Project Objectives

- Clean and consolidate hourly air quality readings
- Resample and interpolate missing data
- Train and validate SARIMA models
- Save model artifacts and summaries
- Evaluate model predictions using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- Perform walk-forward validation

---

## ⚙️ Configuration

All project parameters (file paths, SARIMA settings, etc.) are managed in a central `config.yaml` file.

Example:
```yaml
data:
  monthly_data_folder: data/monthly_record/
  combined_output_csv: data/combined_dataset.csv

model:
  order: [1, 1, 1]
  seasonal_order: [1, 1, 1, 4]

paths:
  models_folder: models/
  results_folder: results/
````

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Merge & preprocess data

```bash
python scripts/wrangle.py
```

### 3. Train and save a SARIMA model

```bash
python scripts/model_training.py
```

### 4. Walk-forward validation

```bash
python scripts/validate.py
```

---

## 📈 Example Output

* `models/`: Timestamped `.pkl` files of trained SARIMA models
* `results/`:

  * Model summaries (text)
  * WFV results (CSV)
  * Evaluation metrics printed to console

---

## 📚 Dependencies

* Python 3.10+
* pandas
* numpy
* scikit-learn
* statsmodels
* joblib
* plotly
* pyyaml

See [`requirements.txt`](./requirements.txt) for exact versions.

---

## 📌 Data Source

Sensor Africa (via [OpenAfrica.net](https://open.africa/dataset/sensorsafrica-airquality-archive-lagos))


