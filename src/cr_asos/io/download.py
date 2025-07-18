import requests
from datetime import datetime, timedelta


def parse_time_input(time_input):
    """
    Parse the time input for downloading ASOS data.
    Args:
        time_input: Either a single date string (e.g., "2025-6-25") for single day,
                    or a list/tuple of two date strings (e.g., ["2025-6-1", "2025-6-30"]) for date range
    """
    # 如果是列表或元组, 且长度为2, 则表示日期范围
    if isinstance(time_input, (list, tuple)) and len(time_input) == 2:
        # Date range
        start_date = datetime.strptime(time_input[0], "%Y-%m-%d")
        end_date = datetime.strptime(time_input[1], "%Y-%m-%d") + timedelta(
            days=1
        )  # Next day 00:00:00
        filename_suffix = f"{start_date.strftime('%Y-%m-%d')}_{(end_date - timedelta(days=1)).strftime('%Y-%m-%d')}"
    # 如果是字符串, 则表示单个日期
    else:
        # Single date
        if isinstance(time_input, str):
            single_date = datetime.strptime(time_input, "%Y-%m-%d")
        else:
            raise ValueError(
                "Invalid time format. Use string 'YYYY-M-D' or list ['YYYY-M-D', 'YYYY-M-D']"
            )
        start_date = single_date
        end_date = single_date + timedelta(days=1)  # Next day 00:00:00
        filename_suffix = single_date.strftime("%Y-%m-%d")

    return start_date, end_date, filename_suffix


def download_asos_1min_data(time, station="PABI", timezone="America/Anchorage"):
    """
    Download ASOS 1-minute data for PABI station.
    Website: https://mesonet.agron.iastate.edu/request/asos/1min.phtml

    Args:
        time: Either a single date string (e.g., "2025-6-25") for single day,
              or a list/tuple of two date strings (e.g., ["2025-6-1", "2025-6-30"]) for date range
        station: Station code (default: "PABI")
        timezone: Timezone (default: "America/Anchorage")
    """
    start_date, end_date, filename_suffix = parse_time_input(time)

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"

    # Define the parameters for the request
    params = {
        "station": station,
        "year1": str(start_date.year),
        "month1": str(start_date.month),
        "day1": str(start_date.day),
        "hour1": "0",
        "minute1": "0",
        "year2": str(end_date.year),
        "month2": str(end_date.month),
        "day2": str(end_date.day),
        "hour2": "0",
        "minute2": "0",
        "sample": "1min",
        "tz": timezone,
        "what": "download",
        "format": "comma",
        "vars": [
            "tmpf",
            "dwpf",
            "sknt",
            "drct",
            "gust_drct",
            "gust_sknt",
            "vis1_coeff",
            "vis1_nd",
            "vis2_coeff",
            "vis2_nd",
            "vis3_coeff",
            "vis3_nd",
            "ptype",
            "precip",
            "pres1",
            "pres2",
            "pres3",
        ],
    }

    response = requests.get(url, params=params)

    # Generate filename based on input type
    filename = f"{station}_ASOS_1min_{filename_suffix}.csv"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Download successful: {filename}")


def download_asos_5min_data(
    time, network="AK_ASOS", station="PABI", timezone="America/Anchorage"
):
    """
    Download ASOS 5-minute data for PABI station.
    Website: https://mesonet.agron.iastate.edu/request/download.phtml

    Args:
        time: Either a single date string (e.g., "2025-6-25") for single day,
              or a list/tuple of two date strings (e.g., ["2025-6-1", "2025-6-30"]) for date range
        station: Station code (default: "PABI")
        timezone: Timezone (default: "America/Anchorage")
    """
    start_date, end_date, filename_suffix = parse_time_input(time)

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "network": network,
        "station": station,
        "data": "all",
        "year1": str(start_date.year),
        "month1": str(start_date.month),
        "day1": str(start_date.day),
        "year2": str(end_date.year),
        "month2": str(end_date.month),
        "day2": str(end_date.day),
        "tz": timezone,
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "yes",  # Directly download to the file
        "report_type": ["1", "3", "4"],
        # 1: MADIS HFMETAR / 5 Minute ASOS
        # 3: Routine / Once Hourly
        # 4: Specials
    }
    response = requests.get(url, params=params)

    # Generate filename based on input type
    filename = f"{station}_ASOS_5min_{filename_suffix}.csv"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Download successful: {filename}")


def download_hourly_precip_data(
    time, network="AK_ASOS", station="PABI", timezone="America/Anchorage"
):
    """
    Download hourly precipitation data for specific station(s).
    Website: https://mesonet.agron.iastate.edu/request/asos/hourlyprecip.phtml

    Args:
        time: Either a single date string (e.g., "2025-6-25") for single day,
              or a list/tuple of two date strings (e.g., ["2025-6-1", "2025-6-30"]) for date range
        network: Network name (default: "AK_ASOS")
        station: Station code (default: "PABI")
        timezone: Timezone (default: "America/Anchorage")
    """

    start_date, end_date, filename_suffix = parse_time_input(time)

    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py"

    params = {
        "station": station,
        "year1": str(start_date.year),
        "month1": str(start_date.month),
        "day1": str(start_date.day),
        "hour1": "0",
        "minute1": "0",
        "year2": str(end_date.year),
        "month2": str(end_date.month),
        "day2": str(end_date.day),
        "hour2": "0",
        "minute2": "0",
        "tz": timezone,
        "network": network,
    }

    response = requests.get(url, params=params)

    # Generate filename based on input type
    filename = f"{station}_ASOS_1h_precip_{filename_suffix}.csv"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Download successful: {filename}")
