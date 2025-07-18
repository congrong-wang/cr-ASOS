import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


matplotlib.rcParams["axes.unicode_minus"] = False

# Read data
df = pd.read_csv("PABI.csv")

# Convert time column to datetime format
df["valid"] = pd.to_datetime(df["valid"])

# Convert some numerical columns from string to float (e.g. sknt, mslp, drct)
df["sknt"] = pd.to_numeric(df["sknt"], errors="coerce")  # Wind speed in knots
df["mslp"] = pd.to_numeric(df["mslp"], errors="coerce")  # Sea Level Pressure in hPa
df["drct"] = pd.to_numeric(df["drct"], errors="coerce")  # Wind direction


# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    """Convert temperature from Fahrenheit to Celsius"""
    return (fahrenheit - 32) * 5 / 9


# Convert temperature columns from Fahrenheit to Celsius
df["tmpc"] = fahrenheit_to_celsius(pd.to_numeric(df["tmpf"], errors="coerce"))
df["dwpc"] = fahrenheit_to_celsius(pd.to_numeric(df["dwpf"], errors="coerce"))


# Date range configuration for plotting
# Set to None to plot all data, or specify as tuple of strings ('start_date', 'end_date')
# Example: plot_date_range = ('2025-06-01', '2025-06-15')
plot_date_range = ("2025-6-9", "2025-7-1")  # Change this to limit the date range

# Filter data based on date range if specified


# Remove outliers from wind speed data using IQR method for y-axis scaling only
def get_wind_speed_ylim(data, column):
    """Calculate appropriate y-axis limits by removing outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 5 * IQR
    upper_bound = Q3 + 5 * IQR

    # Count outliers for reporting
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Found {len(outliers)} outliers in wind speed data")
    print(
        f"Wind speed range: {data[column].min():.1f} - {data[column].max():.1f} knots"
    )

    # Get clean data range for y-axis limits
    clean_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    clean_max = clean_data[column].max()
    print(f"Setting y-axis range: 0 - {clean_max:.1f} knots (excluding outliers)")

    return clean_max


# Get appropriate y-axis limit for wind speed (without changing the data)
wind_upper_limit = get_wind_speed_ylim(df, "sknt")

# Create subplots with additional space for wind direction color bar
fig = plt.figure(figsize=(16, 11))  # Slightly taller to accommodate color bar

# Create subplots with custom spacing
ax1 = plt.subplot(5, 1, 1)  # Temperature
ax2 = plt.subplot(5, 1, 2)  # Humidity
ax3 = plt.subplot(5, 1, 3)  # Pressure
ax_wind_dir = plt.subplot(5, 1, 4)  # Wind direction color bar (thin)
ax4 = plt.subplot(5, 1, 5)  # Wind speed

# Adjust subplot heights - make wind direction bar thinner
fig.subplots_adjust(hspace=0, top=0.93, bottom=0.15)  # More bottom space for legend
pos1 = ax1.get_position()
pos2 = ax2.get_position()
pos3 = ax3.get_position()
pos_wd = ax_wind_dir.get_position()
pos4 = ax4.get_position()

# Make wind direction bar thicker
wind_dir_height = 0.08  # Thicker bar for better visibility
wind_speed_height = 0.12  # Shorter height for wind speed plot

# Calculate new positions to maintain tight spacing
total_height = 0.78  # Reduced available height to leave space for legend (from bottom=0.15 to top=0.93)
remaining_height = total_height - wind_dir_height - wind_speed_height
each_main_plot_height = remaining_height / 3  # For temp, humidity, pressure

# Set positions from top to bottom
y_top = 0.93
ax1.set_position(
    [pos1.x0, y_top - each_main_plot_height, pos1.width, each_main_plot_height]
)  # Temperature

y_temp_bottom = y_top - each_main_plot_height
ax2.set_position(
    [pos2.x0, y_temp_bottom - each_main_plot_height, pos2.width, each_main_plot_height]
)  # Humidity

y_humidity_bottom = y_temp_bottom - each_main_plot_height
ax3.set_position(
    [
        pos3.x0,
        y_humidity_bottom - each_main_plot_height,
        pos3.width,
        each_main_plot_height,
    ]
)  # Pressure

y_pressure_bottom = y_humidity_bottom - each_main_plot_height
ax_wind_dir.set_position(
    [pos_wd.x0, y_pressure_bottom - wind_dir_height, pos_wd.width, wind_dir_height]
)  # Wind direction

y_wind_dir_bottom = y_pressure_bottom - wind_dir_height
ax4.set_position(
    [pos4.x0, y_wind_dir_bottom - wind_speed_height, pos4.width, wind_speed_height]
)  # Wind speed

# Readjust other plots to maintain tight spacing

# Set the time index
df_indexed = df.set_index("valid")  # Use original data

# Temperature subplot
# Handle missing data by breaking the line when time gaps are too large
temp_data = df_indexed["tmpc"].copy()
time_index = df_indexed.index

# Find large time gaps (more than 2 hours) and insert NaN to break the line
time_diffs = time_index.to_series().diff()
large_gaps = time_diffs > pd.Timedelta(hours=2)

# Insert NaN values where there are large time gaps
for i in range(1, len(time_index)):
    if large_gaps.iloc[i]:
        temp_data.iloc[i] = np.nan

ax1.plot(time_index, temp_data, color="red", linewidth=2)
ax1.set_ylabel("Temperature (°C)", color="red")
ax1.tick_params(axis="y", labelcolor="red")
ax1.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels

# Relative Humidity subplot
# Handle missing data by breaking the line when time gaps are too large
humidity_data = df_indexed["relh"].copy()

# Insert NaN values where there are large time gaps (reuse the same gap detection)
for i in range(1, len(time_index)):
    if large_gaps.iloc[i]:
        humidity_data.iloc[i] = np.nan

ax2.plot(time_index, humidity_data, color="blue", linewidth=2)
ax2.set_ylabel("Relative Humidity (%)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
ax2.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
ax2.set_ylim(0, 100)  # Set y-axis from 0 to 100% for relative humidity

# Sea Level Pressure subplot
ax3.plot(df_indexed.index, df_indexed["mslp"], color="green", linewidth=2)
ax3.set_ylabel("Sea Level Pressure (hPa)", color="green")
ax3.tick_params(axis="y", labelcolor="green")
ax3.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels

# Wind Direction Color Bar - Simplified approach
wind_dir_data = df_indexed["drct"].values
wind_speed_data = df_indexed["sknt"].values
time_data = df_indexed.index

# Simple color mapping: white for calm wind, HSV for wind direction
colors = []
for i in range(len(wind_dir_data)):
    if pd.isna(wind_dir_data[i]) or wind_speed_data[i] <= 0.1:
        colors.append("white")
    else:
        # Convert wind direction to HSV color
        hue = wind_dir_data[i] / 360.0
        colors.append(plt.cm.hsv(hue))

# Plot using bar chart for simplicity and alignment
ax_wind_dir.bar(
    time_data,
    height=1,
    width=pd.Timedelta(hours=1),
    color=colors,
    align="center",
    edgecolor="none",
)
ax_wind_dir.set_ylim(0, 1)
ax_wind_dir.set_ylabel("Wind Dir", fontsize=10, color="purple")
ax_wind_dir.tick_params(axis="y", labelcolor="purple", labelsize=8)
ax_wind_dir.tick_params(axis="x", labelbottom=False)
ax_wind_dir.set_yticks([])

print(
    f"Wind direction color bar created with {len([c for c in colors if c == 'white'])} calm periods"
)

# Wind Speed subplot
# Handle missing data by breaking the line when time gaps are too large
wind_speed_data = df_indexed["sknt"].copy()
time_index = df_indexed.index

# Find large time gaps (more than 2 hours) and insert NaN to break the line
time_diffs = time_index.to_series().diff()
large_gaps = time_diffs > pd.Timedelta(hours=2)

# Insert NaN values where there are large time gaps
for i in range(1, len(time_index)):
    if large_gaps.iloc[i]:
        # Insert NaN at the point after a large gap to break the line
        wind_speed_data.iloc[i] = np.nan

ax4.plot(time_index, wind_speed_data, color="orange", linewidth=2)
ax4.set_ylabel("Wind Speed (knots)", color="orange")
ax4.tick_params(axis="y", labelcolor="orange")
ax4.set_xlabel("Time (America/Anchorage)")  # Set x-axis label with timezone
ax4.set_ylim(
    0, wind_upper_limit
)  # Set y-axis from 0 to upper limit after removing outliers

# Configure x-axis ticks and formatting for the bottom subplot only
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as MM-DD
ax4.tick_params(axis="x", rotation=45)  # Rotate labels for better readability

# Add vertical grid lines at each major tick (daily) and ensure wind direction ax is included
for ax in [ax1, ax2, ax3, ax_wind_dir, ax4]:  # Include wind direction axis
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.grid(True, alpha=0.3, which="major", axis="both")
    ax.grid(True, alpha=0.1, which="minor", axis="x")
    # Ensure all axes have the same x-axis limits
    if plot_date_range is not None:
        start_date, end_date = plot_date_range
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    else:
        ax.set_xlim(df_indexed.index.min(), df_indexed.index.max())

plt.suptitle("PABI Weather Data", fontsize=16)

# Add circular wind direction legend at the bottom
# Create a separate axis for the wind direction legend
legend_ax = fig.add_axes(
    [0.45, 0.01, 0.1, 0.1]
)  # [left, bottom, width, height] - smaller square area

# Create circular wind rose
theta = np.linspace(0, 2 * np.pi, 37)  # 36 segments + 1 to close the circle
radius_inner = 0.3
radius_outer = 0.8

# Create the circular segments
for i in range(36):
    # Calculate angles for this segment (remember: 0° is North, clockwise)
    angle_start = (
        i * 10 * np.pi / 180 - np.pi / 2
    )  # Convert to math convention (counter-clockwise from East)
    angle_end = (i + 1) * 10 * np.pi / 180 - np.pi / 2

    # Create wedge coordinates
    angles = np.linspace(angle_start, angle_end, 10)

    # Outer arc
    x_outer = radius_outer * np.cos(angles)
    y_outer = radius_outer * np.sin(angles)

    # Inner arc (reversed for proper polygon)
    x_inner = radius_inner * np.cos(angles[::-1])
    y_inner = radius_inner * np.sin(angles[::-1])

    # Combine to form the wedge
    x_wedge = np.concatenate([x_outer, x_inner])
    y_wedge = np.concatenate([y_outer, y_inner])

    # Get color from HSV colormap
    color_value = i / 36.0
    from matplotlib.cm import hsv

    color = hsv(color_value)

    # Draw the wedge
    legend_ax.fill(x_wedge, y_wedge, color=color, edgecolor="white", linewidth=0.5)

# Add direction labels
legend_ax.text(
    0, radius_outer + 0.15, "N", ha="center", va="center", fontsize=8, fontweight="bold"
)
legend_ax.text(
    radius_outer + 0.15, 0, "E", ha="center", va="center", fontsize=8, fontweight="bold"
)
legend_ax.text(
    0,
    -(radius_outer + 0.15),
    "S",
    ha="center",
    va="center",
    fontsize=8,
    fontweight="bold",
)
legend_ax.text(
    -(radius_outer + 0.15),
    0,
    "W",
    ha="center",
    va="center",
    fontsize=8,
    fontweight="bold",
)

# Set equal aspect ratio and clean up axes
legend_ax.set_xlim(-1.2, 1.2)
legend_ax.set_ylim(-1.2, 1.2)
legend_ax.set_aspect("equal")
legend_ax.axis("off")  # Hide axes
legend_ax.text(0, -1.1, "Wind Direction", ha="center", va="center", fontsize=9)

# Don't call subplots_adjust again as it will override our custom positioning
plt.savefig("PABI_weather_plot.png", dpi=300, bbox_inches="tight")
plt.show()
