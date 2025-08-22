# Installation

Run

```bash
pip install git+https://github.com/congrong-wang/cr-ASOS.git
```



# Functions

## Download data

There are three types of data for download:

- 1 min weather data
- 5 min weather data
- hourly precipitation data

You can import download functions using:

```python
from cr_asos.io import (
    download_asos_1min_data,
    download_asos_5min_data,
    download_asos_hourly_precip_data,
)
```

These functions are documented with docstring. Call `help(function_name)` to read it.



## Read downloaded files

There are also three functions to read downloaded CSV files:

```python
from cr_asos.io import (
    read_asos_1min_csv,
    read_asos_5min_csv,
    read_asos_hourly_precip_csv,
)
```

These functions are also documented.



## Plot

There is currently only one plotting function:

```python
from cr_asos.plotting import daily_plot_w_SMPS
```

![](/home/wcr/ASOS3.11.12/assets/PABI_daily_w_SMPS_2025-06-25.png)
