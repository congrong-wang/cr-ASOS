import numpy as np


# Function to calculate relative humidity from temperature and dew point
def calculate_relative_humidity(temp_c, dewpoint_c):
    """Calculate relative humidity from temperature and dew point in Celsius"""

    # Magnus formula for saturation vapor pressure
    def saturation_vapor_pressure(temp):
        return 6.112 * np.exp((17.67 * temp) / (temp + 243.5))

    es_temp = saturation_vapor_pressure(
        temp_c
    )  # Saturation vapor pressure at temperature
    es_dew = saturation_vapor_pressure(
        dewpoint_c
    )  # Saturation vapor pressure at dew point

    rh = (es_dew / es_temp) * 100
    return np.clip(rh, 0, 100)  # Ensure RH is between 0 and 100%
