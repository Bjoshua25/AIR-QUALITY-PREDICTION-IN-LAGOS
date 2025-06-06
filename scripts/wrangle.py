import os
import pandas as pd
import glob
from config import load_config

cfg = load_config()
folder_path = cfg["data"]["monthly_data_folder"]
output_csv_path = cfg["data"]["combined_output_csv"]


def wrangle(filepath):
    """
    Loads and cleans a single monthly PM2.5 reading file from Sensor Africa.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'PM2.5' column and datetime index.
    """
    df = pd.read_csv(filepath, delimiter=';', index_col='timestamp')

    # Filter for PM2.5 only
    df = df[df['value_type'] == 'P2']

    # Convert index to datetime
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]  # Drop rows with invalid timestamps

    # Localize to UTC if not already localized
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Convert to Lagos time zone
    df.index = df.index.tz_convert('Africa/Lagos')
    
    # Rename and clean columns
    df.rename(columns={'value': 'PM2.5'}, inplace=True)

    # resampling the data to 1 hour interval
    df.index.name = 'date'
    df.sort_index(inplace=True)

    return df

def merge_monthly_files(folder_path = folder_path, output_csv_path = output_csv_path):
    """
    Merges all monthly CSVs in a folder into a single combined DataFrame and saves it.

    Parameters:
        folder_path (str): Directory containing the monthly CSV files.
        output_csv_path (str): Path to save the combined CSV file.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    dataframes = []
    for file in csv_files:
        try:
            df = wrangle(file)
            dataframes.append(df)
            print(f"Processed {os.path.basename(file)}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

    combined_df = pd.concat(dataframes)
    combined_df.to_csv(output_csv_path)
    print(f"\nCombined shape: {combined_df.shape}")

    return combined_df


def load_combined_series(filepath=None):
    """
    Loads the combined CSV into a Series with datetime index.

    Returns:
        pd.Series: The PM2.5 values with datetime index.
    """
    filepath = cfg["data"]["combined_output_csv"]

    df = pd.read_csv(filepath, index_col="date", parse_dates=True)
    df = df["PM2.5"].resample('6h').mean().interpolate(method = "time").to_frame()
    df.sort_index(inplace=True)

    if "PM2.5" not in df.columns:
        raise ValueError("Expected 'PM2.5' column not found.")

    return df["PM2.5"]