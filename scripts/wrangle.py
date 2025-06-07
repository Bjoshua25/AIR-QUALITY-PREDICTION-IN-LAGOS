import os
import pandas as pd
import glob
from scripts.config import load_config

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
    print(f"Reading files from: {folder_path}")
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


def load_combined_series(split=True, train_ratio=0.8):
    """
    Load the combined Air Quality CSV, convert to a time series,
    and optionally split into train and test series.

    Parameters:
        split (bool): Whether to split the data into train and test sets.
        train_ratio (float): Proportion of data to use for training.

    Returns:
        tuple or pd.Series:
            If split=True: (train_series, test_series)
            If split=False: full time series as pd.Series
    """
    combined_path = cfg["data"]["combined_output_csv"]
    df = pd.read_csv(combined_path, index_col="date", parse_dates=True)
    series = df["PM2.5"].resample("6h").mean().interpolate(method="time")

    if split:
        cutoff = int(len(series) * train_ratio)
        y_train = series.iloc[:cutoff] 
        y_test = series.iloc[cutoff:]
        return y_train, y_test
    else:
        return series


def main():
    merge_monthly_files()

if __name__ == "__main__":
    main()

