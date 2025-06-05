import os
import glob
import pandas as pd

def wrangle(filepath):
    """
    A function that load monthly PM2.5 readings from sensor africa, clean it, and preprocess it.

    Parameter:
    ---------
        filepath: str
            file directory or file location
        
    Return:
    ------
        DataFrame: 
            contains only one columns, "PM2.5".
            A properly formated datetime index.

    Example:
    --------
        wrangle("..data/november_2023.csv")s
    """
    
    # Load the dataset (semicolon-separated)
    df = pd.read_csv(filepath, delimiter=';', index_col='timestamp')

    # Filter only PM2.5 readings
    df = df[df['value_type'] == 'P2']

    # Convert index to datetime
    df.index = pd.to_datetime(df.index, errors='coerce')

    # Drop rows with invalid (NaT) timestamps
    df = df[~df.index.isna()]

    # Localize and convert time zone
    df.index = df.index.tz_convert('Africa/Lagos')

    # Rename column for clarity
    df.rename(columns={'value': 'PM2.5'}, inplace=True)

    # Resample to hourly frequency, forward fill
    df = df['PM2.5'].resample('1h').mean().ffill().to_frame()

    # Rename index to 'date'
    df.index.name = 'date'

    # Sort by date ascending
    df.sort_index(inplace=True)

    return df
