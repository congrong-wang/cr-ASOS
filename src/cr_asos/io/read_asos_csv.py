import numpy as np
import pandas as pd
from cr_asos.utils import calculate_relative_humidity, fahrenheit_to_celsius


def read_asos_1min_csv(file_path):
    """
    Reads an ASOS CSV file and returns a DataFrame.

    Parameters:
    file_path (str): Path to the ASOS CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the ASOS data.
    """

    # Read pabi1min.csv file
    df = pd.read_csv(
        file_path,
        parse_dates=["valid(America/Anchorage)"],
        index_col="valid(America/Anchorage)",
        na_values=["M"],  # Treat 'M' as missing values
    )

    # Convert all numeric columns to proper numeric types
    numeric_columns = [
        "tmpf",
        "dwpf",
        "sknt",
        "drct",
        "mslp",
        "p01i",
        "vsby",
        "gust",
        "precip",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert temperatures to Celsius
    df["temp_c"] = fahrenheit_to_celsius(df["tmpf"])
    df["dewpoint_c"] = fahrenheit_to_celsius(df["dwpf"])

    # Calculate relative humidity
    df["rh"] = calculate_relative_humidity(df["temp_c"], df["dewpoint_c"])

    # Convert inches to millimeters for precipitation
    df["precip_mm"] = df["precip"] * 25.4

    # Convert wind speed from knots to meters per second
    df["sknt_ms"] = df["sknt"] * 0.514444

    # FOR DEBUG
    # save df to CSV
    # df.to_csv("debug_asos_1min.csv")

    return df


def read_asos_5min_csv(file_path):
    """
    Reads an ASOS CSV file and returns a DataFrame.

    Parameters:
    file_path (str): Path to the ASOS CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the ASOS data.
    """

    # Read pabi1min.csv file
    df = pd.read_csv(
        file_path,
        parse_dates=["valid"],
        index_col="valid",
        na_values=["M"],  # Treat 'M' as missing values
    )

    # Convert all numeric columns to proper numeric types
    numeric_columns = [
        "tmpf",
        "dwpf",
        "relh",
        "drct",
        "sknt",
        "p01i",
        "mslp",
        "vsby",
        "gust",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert temperatures to Celsius
    df["temp_c"] = fahrenheit_to_celsius(df["tmpf"])
    # df["dewpoint_c"] = fahrenheit_to_celsius(df["dwpf"])

    df.rename(columns={"relh": "rh"}, inplace=True)

    def get_hour_block(t):
        if t.minute > 53:
            # 如果在 xx:54 ~ xx:59 的话，它属于下一小时的周期
            return t.floor("h") + pd.Timedelta(hours=1)
        else:
            return (t - pd.Timedelta(hours=1)).floor("h") + pd.Timedelta(hours=1)

    df["hour_block"] = df.index.map(get_hour_block)

    # Calculate precipitation in mm for each hour block
    # Because p01i is cumulative within 1h, we need to calculate the difference
    df["precip_in"] = df.groupby("hour_block")["p01i"].diff()
    df["precip_in"] = df["precip_in"].fillna(df["p01i"])  # add the first value
    df["precip_mm"] = df["precip_in"] * 25.4

    # Convert wind speed from knots to meters per second
    df["sknt_ms"] = df["sknt"] * 0.514444

    # FOR DEBUG
    # save df to CSV
    # df.to_csv("debug_asos_5min.csv")

    return df


def read_asos_1h_precip_csv(file_path):
    df = pd.read_csv(
        file_path,
        parse_dates=["valid"],
        index_col="valid",
    )
    # Convert all numeric columns to proper numeric types
    numeric_columns = [
        "precip_in",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert inches to millimeters
    df["precip_mm"] = df["precip_in"] * 25.4

    # FOR DEBUG
    # save df to CSV
    # df.to_csv("debug_asos_1h_precip.csv")

    return df
