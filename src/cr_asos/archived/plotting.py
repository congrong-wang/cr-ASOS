import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import cmocean


# Function to smooth wind direction (circular/angular data)
def smooth_wind_direction(
    directions, window=5
):  # Keep original window for 5-min wind data
    """Smooth wind direction data using circular statistics"""
    # Convert degrees to radians
    directions_rad = np.deg2rad(directions)

    # Calculate sin and cos components
    sin_components = np.sin(directions_rad)
    cos_components = np.cos(directions_rad)

    # Apply rolling average to components
    sin_smoothed = (
        pd.Series(sin_components)
        .rolling(window=window, center=True, min_periods=3)
        .mean()
    )
    cos_smoothed = (
        pd.Series(cos_components)
        .rolling(window=window, center=True, min_periods=3)
        .mean()
    )

    # Convert back to angles
    smoothed_rad = np.arctan2(sin_smoothed, cos_smoothed)
    smoothed_deg = np.rad2deg(smoothed_rad)

    # Ensure angles are in 0-360 range
    smoothed_deg = (smoothed_deg + 360) % 360

    return smoothed_deg


def plot_1min_daily(dt, plot_date):

    # # Date configuration for single day plotting
    # plot_date = "2025-06-23"  # Change this to the date you want to plot

    matplotlib.rcParams["axes.unicode_minus"] = False

    # Function to smooth wind direction (circular/angular data)
    def smooth_wind_direction(directions, window=5):
        """Smooth wind direction data using circular statistics"""
        # Convert degrees to radians
        directions_rad = np.deg2rad(directions)

        # Calculate sin and cos components
        sin_components = np.sin(directions_rad)
        cos_components = np.cos(directions_rad)

        # Apply rolling average to components
        sin_smoothed = (
            pd.Series(sin_components)
            .rolling(window=window, center=True, min_periods=3)
            .mean()
        )
        cos_smoothed = (
            pd.Series(cos_components)
            .rolling(window=window, center=True, min_periods=3)
            .mean()
        )

        # Convert back to angles
        smoothed_rad = np.arctan2(sin_smoothed, cos_smoothed)
        smoothed_deg = np.rad2deg(smoothed_rad)

        # Ensure angles are in 0-360 range
        smoothed_deg = (smoothed_deg + 360) % 360

        return smoothed_deg

    # Filter data for the specific date
    date_mask = dt.index.date == pd.to_datetime(plot_date).date()
    daily_data = dt[date_mask].copy()

    print(f"Plotting data for {plot_date}")
    print(f"Found {len(daily_data)} data points for this day")

    if len(daily_data) == 0:
        print(f"No data found for {plot_date}")
        return

    # # save daily data to CSV to see if it looks correct
    # daily_data.to_csv(f"pabi1min_daily_{plot_date}.csv")

    # Create the multi-variable plot with stacked layout (no gaps) - 1:1 aspect ratio with elongated subplots
    fig = plt.figure(figsize=(12, 12))  # 1:1 aspect ratio for overall figure

    # Create subplots with custom spacing (5 subplots including wind direction bar)
    ax1 = plt.subplot(5, 1, 1)  # Temperature
    ax2 = plt.subplot(5, 1, 2)  # Humidity
    ax3 = plt.subplot(5, 1, 3)  # Precipitation (moved up)
    ax4 = plt.subplot(5, 1, 4)  # Wind Speed (moved down)
    ax_wind_dir = plt.subplot(5, 1, 5)  # Wind direction color bar (at bottom)

    # Adjust subplot heights - no spacing between subplots
    fig.subplots_adjust(
        hspace=0, top=0.92, bottom=0.25, left=0.05, right=0.95, wspace=0
    )  # Reduced top from 0.95 to 0.92 to give more space for top x-axis labels
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()
    pos_wd = ax_wind_dir.get_position()

    # Calculate heights - make wind direction bar thinner (10:1 ratio) compared to main plots (6:1 ratio each)
    wind_dir_height = 0.04  # Very thin for 10:1 aspect ratio
    total_height = 0.70  # Increased available height (from top=0.95 to bottom=0.25)
    remaining_height = total_height - wind_dir_height
    each_main_plot_height = (
        remaining_height / 4
    )  # For temp, humidity, precip, wind speed (6:1 ratio each)

    # Set positions from top to bottom with no gaps
    y_top = 0.93  # Adjusted to match the new top margin
    ax1.set_position(
        [pos1.x0, y_top - each_main_plot_height, pos1.width, each_main_plot_height]
    )

    y_second = y_top - each_main_plot_height  # Directly adjacent to ax1
    ax2.set_position(
        [pos2.x0, y_second - each_main_plot_height, pos2.width, each_main_plot_height]
    )

    y_third = y_second - each_main_plot_height  # Directly adjacent to ax2
    ax3.set_position(
        [pos3.x0, y_third - each_main_plot_height, pos3.width, each_main_plot_height]
    )

    y_fourth = y_third - each_main_plot_height  # Directly adjacent to ax3
    ax4.set_position(
        [pos4.x0, y_fourth - each_main_plot_height, pos4.width, each_main_plot_height]
    )

    y_wind_dir = y_fourth - each_main_plot_height  # Directly adjacent to ax4
    ax_wind_dir.set_position(
        [pos_wd.x0, y_wind_dir - wind_dir_height, pos_wd.width, wind_dir_height]
    )

    # Handle missing data gaps - break line if time gap > 10 minutes for 1-minute data
    time_diffs = daily_data.index.to_series().diff()
    large_gaps = time_diffs > pd.Timedelta(minutes=10)

    # Temperature subplot with 5-minute smoothing
    temp_data = daily_data["temp_c"].copy()
    for i in range(1, len(daily_data)):
        if large_gaps.iloc[i]:
            temp_data.iloc[i] = np.nan

    # Apply 5-minute rolling average smoothing (window=5 for 1-minute data)
    temp_smoothed = temp_data.rolling(window=5, center=True, min_periods=3).mean()

    # Plot both original (faint) and smoothed (bold) lines
    ax1.plot(
        daily_data.index,
        temp_data,
        color="red",
        linewidth=0.5,
        alpha=0.3,
        label="Original",
    )
    ax1.plot(
        daily_data.index,
        temp_smoothed,
        color="red",
        linewidth=1.5,
        label="5-min smoothed",
    )
    ax1.set_ylabel("Temperature (°C)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax1.grid(True, alpha=0.3)

    # Relative Humidity subplot with 5-minute smoothing
    rh_data = daily_data["rh"].copy()
    for i in range(1, len(daily_data)):
        if large_gaps.iloc[i]:
            rh_data.iloc[i] = np.nan

    # Apply 5-minute rolling average smoothing (window=5 for 1-minute data)
    rh_smoothed = rh_data.rolling(window=5, center=True, min_periods=3).mean()

    # Plot both original (faint) and smoothed (bold) lines
    ax2.plot(
        daily_data.index,
        rh_data,
        color="blue",
        linewidth=0.5,
        alpha=0.3,
        label="Original",
    )
    ax2.plot(
        daily_data.index,
        rh_smoothed,
        color="blue",
        linewidth=1.5,
        label="5-min smoothed",
    )
    ax2.set_ylabel("Relative\nHumidity (%)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # Precipitation subplot - Cumulative precipitation curve (moved to 3rd position)
    if "precip" in daily_data.columns:
        # Convert precipitation from inches to millimeters

        # Calculate cumulative precipitation from start of day
        cumulative_precip = daily_data["precip_mm"].cumsum()

        # Plot cumulative precipitation as a curve
        ax3.plot(daily_data.index, cumulative_precip, color="green", linewidth=2)

        # Fill the area under the curve for better visualization
        ax3.fill_between(daily_data.index, cumulative_precip, alpha=0.3, color="green")

        ax3.set_ylabel("Cumulative\nPrecipitation (mm)", color="green")
        ax3.tick_params(axis="y", labelcolor="green")
        ax3.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
        ax3.grid(True, alpha=0.3)

        # Set y-axis to start from 0
        ax3.set_ylim(bottom=0)

        # Add final total as text
        daily_total = cumulative_precip.iloc[-1] if len(cumulative_precip) > 0 else 0
        daily_total_inches = daily_data["precip"].sum()  # Original value in inches
        ax3.text(
            0.01,
            0.94,
            f"Daily Total: {daily_total:.2f} mm ({daily_total_inches:.3f} in)",
            transform=ax3.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            verticalalignment="top",
            horizontalalignment="left",
        )

        print(
            f"Daily precipitation total: {daily_total:.2f} mm ({daily_total_inches:.3f} inches)"
        )

    else:
        ax3.text(
            0.5,
            0.5,
            "No precipitation data available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_ylabel("Precipitation (No Data)", color="green")
        ax3.tick_params(axis="x", labelbottom=False)

    # Wind Speed subplot with 5-minute smoothing (moved to 4th position)
    wind_speed_data = daily_data["sknt"].copy()
    for i in range(1, len(daily_data)):
        if large_gaps.iloc[i]:
            wind_speed_data.iloc[i] = np.nan

    # Convert wind speed from knots to m/s
    wind_speed_ms = daily_data["sknt_ms"].copy()

    # Apply 5-minute rolling average smoothing for wind speed
    wind_speed_smoothed = wind_speed_ms.rolling(
        window=5, center=True, min_periods=3
    ).mean()

    # Plot both original (faint) and smoothed (bold) lines
    ax4.plot(
        daily_data.index,
        wind_speed_ms,
        color="orange",
        linewidth=0.5,
        alpha=0.3,
        label="Original",
    )
    ax4.plot(
        daily_data.index,
        wind_speed_smoothed,
        color="orange",
        linewidth=1.5,
        label="5-min smoothed",
    )
    ax4.set_ylabel("Wind Speed (m/s)", color="orange")
    ax4.tick_params(axis="y", labelcolor="orange")
    ax4.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax4.grid(True, alpha=0.3)

    # Wind Direction Color Bar with smoothing (at the bottom)
    if "drct" in daily_data.columns:
        wind_dir_data = daily_data["drct"].values
        wind_speed_for_dir = daily_data["sknt"].values
        time_data = daily_data.index

        # Apply 5-minute smoothing to wind direction using circular statistics
        wind_dir_smoothed = smooth_wind_direction(daily_data["drct"], window=5)

        # Color mapping: white for calm wind, phase colormap for wind direction (using smoothed direction)
        colors = []
        for i in range(len(wind_dir_data)):
            if pd.isna(wind_dir_data[i]) or wind_speed_for_dir[i] <= 0.1:
                colors.append("white")
            else:
                # Use smoothed wind direction for color mapping
                smoothed_dir = (
                    wind_dir_smoothed.iloc[i]
                    if not pd.isna(wind_dir_smoothed.iloc[i])
                    else wind_dir_data[i]
                )
                hue = smoothed_dir / 360.0
                colors.append(cmocean.cm.phase(hue))

        # Plot using bar chart for alignment with time series
        ax_wind_dir.bar(
            time_data,
            height=1,
            width=pd.Timedelta(minutes=1),
            color=colors,
            align="center",
            edgecolor="none",
        )
        ax_wind_dir.set_ylim(0, 1)
        ax_wind_dir.set_ylabel(
            "Wind\nDir",
            fontsize=10,
            color="purple",
            rotation=0,
            ha="right",
            va="center",
        )
        ax_wind_dir.tick_params(axis="y", labelcolor="purple", labelsize=8)
        ax_wind_dir.set_xlabel(
            "Time (America/Anchorage)"
        )  # Show x-axis labels on bottom subplot
        ax_wind_dir.set_yticks([])

        calm_periods = len([c for c in colors if c == "white"])
        print(f"Wind direction color bar created with {calm_periods} calm periods")
    else:
        ax_wind_dir.text(
            0.5,
            0.5,
            "No wind direction data available",
            ha="center",
            va="center",
            transform=ax_wind_dir.transAxes,
        )
        ax_wind_dir.set_ylabel(
            "Wind\nDir",
            fontsize=10,
            color="purple",
            rotation=0,
            ha="right",
            va="center",
        )
        ax_wind_dir.set_xlabel("Time (America/Anchorage)")

    # Set same x-axis limits for all subplots and add grid
    start_of_day = daily_data.index.min().replace(hour=0, minute=0, second=0)
    end_of_day = daily_data.index.max().replace(hour=23, minute=59, second=59)

    # Set x-axis formatting for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax_wind_dir]:
        ax.set_xlim(start_of_day, end_of_day)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Every hour
        ax.grid(
            True, alpha=0.3, which="major", axis="x", linestyle="--"
        )  # Vertical dashed grid lines for x-axis only

        # For bottom subplot, show x-axis labels
        if ax == ax_wind_dir:
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%H")
            )  # Only hour numbers
            ax.tick_params(
                axis="x", rotation=0, labelsize=10
            )  # No rotation needed for short labels
            ax.set_xlabel(
                "Time (America/Anchorage)"
            )  # Show x-axis labels on bottom subplot
        else:
            # For upper subplots, hide x-axis labels
            ax.tick_params(axis="x", labelbottom=False)

    # Add top x-axis only for the topmost subplot (temperature)
    ax1_top = ax1.twiny()  # Create twin axis at top for temperature subplot only
    ax1_top.set_xlim(start_of_day, end_of_day)
    ax1_top.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1_top.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # Only hour numbers
    ax1_top.tick_params(
        axis="x", rotation=0, labelsize=10
    )  # Same font size as bottom labels

    # Ensure no spacing between subplots by resetting positions after adding twin axis
    ax1.set_position(
        [pos1.x0, y_top - each_main_plot_height, pos1.width, each_main_plot_height]
    )
    ax2.set_position(
        [pos2.x0, y_top - 2 * each_main_plot_height, pos2.width, each_main_plot_height]
    )
    ax3.set_position(
        [pos3.x0, y_top - 3 * each_main_plot_height, pos3.width, each_main_plot_height]
    )
    ax4.set_position(
        [pos4.x0, y_top - 4 * each_main_plot_height, pos4.width, each_main_plot_height]
    )
    ax_wind_dir.set_position(
        [
            pos_wd.x0,
            y_top - 4 * each_main_plot_height - wind_dir_height,
            pos_wd.width,
            wind_dir_height,
        ]
    )

    plt.suptitle(f"PABI Weather Data - {plot_date}", fontsize=16)

    # Bottom area divided into three sections: left (wind direction legend), center (for wind rose), right (reserved)
    bottom_height = 0.18  # More space for bottom area
    bottom_y = 0.02  # Close to bottom edge

    # Left section: Wind direction legend (1/3 of width) - adjusted for 1:1 overall ratio
    legend_ax_left = fig.add_axes(
        [0.05, bottom_y, 0.28, bottom_height]
    )  # Adjusted for 1:1 ratio with elongated content

    radius_inner = 0.28  # Inner radius
    radius_outer = 0.55  # Outer radius

    # Create the circular segments for wind direction color legend
    for i in range(36):
        # Calculate angles for this segment (0° is North, clockwise)
        # Fixed: Need to rotate clockwise from North, so subtract from π/2
        angle_start = np.pi / 2 - i * 10 * np.pi / 180  # Start from North, go clockwise
        angle_end = np.pi / 2 - (i + 1) * 10 * np.pi / 180

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

        # Get color from phase colormap - match the actual wind direction angle
        # Note: Wind direction is "from" direction, but we want to show it at the correct compass position
        # i=0 is at North position (-π/2), and should show 0° wind (North wind) color
        # But we need to map the visual position to the wind direction correctly
        compass_angle = (
            i * 10
        )  # The compass position in degrees (0=North, 90=East, etc.)
        # For wind direction color mapping, we want the color that would appear at this compass position
        color_value = compass_angle / 360.0
        color = cmocean.cm.phase(color_value)

        # Draw the wedge
        legend_ax_left.fill(
            x_wedge, y_wedge, color=color, edgecolor="white", linewidth=0.5
        )

    # Add direction labels with consistent font style
    label_distance = radius_outer + 0.15  # Closer labels to avoid overlap
    legend_ax_left.text(
        0,
        label_distance,
        "N",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    legend_ax_left.text(
        label_distance,
        0,
        "E",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    legend_ax_left.text(
        0,
        -label_distance,
        "S",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    legend_ax_left.text(
        -label_distance,
        0,
        "W",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )

    # Set equal aspect ratio and clean up axes
    legend_ax_left.set_xlim(-0.9, 0.9)  # Smaller bounds to reduce overlap
    legend_ax_left.set_ylim(-0.9, 0.9)
    legend_ax_left.set_aspect("equal")
    legend_ax_left.axis("off")  # Hide axes
    legend_ax_left.text(
        0, -0.85, "Wind Direction", ha="center", va="center", fontsize=9
    )

    # Center section: Wind rose (1/3 of width)
    legend_ax_center = fig.add_axes(
        [0.36, bottom_y, 0.28, bottom_height]
    )  # Adjusted positioning for 1:1 ratio

    # Create 16-sector wind rose
    if "drct" in daily_data.columns and "sknt" in daily_data.columns:
        # Define 16 sectors (22.5° each)
        sectors = 16
        sector_angle = 360.0 / sectors

        # Initialize arrays for wind rose data
        sector_counts = np.zeros(sectors)
        sector_avg_speed = np.zeros(sectors)

        # Get valid wind data (exclude calm periods and NaN)
        valid_mask = (
            (~pd.isna(daily_data["drct"]))
            & (~pd.isna(daily_data["sknt"]))
            & (daily_data["sknt"] > 0.1)
        )
        valid_directions = daily_data["drct"][valid_mask].values
        valid_speeds = daily_data["sknt"][valid_mask].values

        # Classify data into sectors
        for direction, speed in zip(valid_directions, valid_speeds):
            # Convert direction to sector index (0° = North)
            sector_idx = int((direction + sector_angle / 2) % 360 / sector_angle)
            sector_counts[sector_idx] += 1
            sector_avg_speed[sector_idx] += speed

        # Calculate average speeds (avoid division by zero)
        for i in range(sectors):
            if sector_counts[i] > 0:
                sector_avg_speed[i] /= sector_counts[i]

        # Normalize counts to percentages
        total_valid_data = np.sum(sector_counts)
        if total_valid_data > 0:
            sector_percentages = (sector_counts / total_valid_data) * 100
        else:
            sector_percentages = np.zeros(sectors)

        # Create polar plot for wind rose
        theta = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        # Offset by 90° to put North at top and rotate clockwise
        theta_plot = (np.pi / 2 - theta) % (2 * np.pi)

        # Create the wind rose bars
        max_percentage = (
            np.max(sector_percentages) if np.max(sector_percentages) > 0 else 1
        )

        for i in range(sectors):
            if sector_percentages[i] > 0:
                # Radial length proportional to frequency (reduced to 90% size)
                radius = (
                    sector_percentages[i] / max_percentage * 0.72
                )  # Changed from 0.8 to 0.72 (0.8 * 0.9)

                # Color based on wind direction (same as wind direction color band and color ring)
                # Calculate the center angle of this sector in degrees
                sector_center_angle = i * sector_angle  # 0° = North, 90° = East, etc.
                # Use same phase color mapping as wind direction color band
                color_value = sector_center_angle / 360.0
                color = cmocean.cm.phase(color_value)

                # Calculate sector edges
                angle_start = theta_plot[i] - np.pi / sectors
                angle_end = theta_plot[i] + np.pi / sectors

                # Create wedge coordinates
                angles = np.linspace(angle_start, angle_end, 10)
                x_outer = radius * np.cos(angles)
                y_outer = radius * np.sin(angles)
                x_inner = np.zeros_like(angles)
                y_inner = np.zeros_like(angles)

                # Combine to form the wedge
                x_wedge = np.concatenate([x_inner, x_outer[::-1]])
                y_wedge = np.concatenate([y_inner, y_outer[::-1]])

                # Draw the wedge
                legend_ax_center.fill(
                    x_wedge,
                    y_wedge,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.8,
                )

        # Add direction labels
        direction_labels = [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ]
        label_radius = (
            0.81  # Reduced from 0.9 to 0.81 (0.9 * 0.9) to match the smaller wind rose
        )

        for i, label in enumerate(direction_labels):
            # Skip the "N" label (i=0) to avoid covering frequency numbers
            if i == 0:  # Skip North label
                continue

            angle = theta_plot[i]
            x_label = label_radius * np.cos(angle)
            y_label = label_radius * np.sin(angle)
            legend_ax_center.text(
                x_label,
                y_label,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="normal",  # Changed from "bold" to "normal" to match other labels
            )

        # Add concentric circles for percentage scale (reduced to 90% size)
        for r in [
            0.18,
            0.36,
            0.54,
            0.72,
        ]:  # Changed from [0.2, 0.4, 0.6, 0.8] to 90% size
            circle = plt.Circle(
                (0, 0), r, fill=False, color="gray", alpha=0.3, linewidth=0.5
            )
            legend_ax_center.add_patch(circle)

        # Add percentage labels
        max_pct = max_percentage if max_percentage > 0 else 1
        for i, r in enumerate(
            [0.18, 0.36, 0.54, 0.72]
        ):  # Updated to match new circle radii
            pct_value = (r / 0.72) * max_pct  # Changed denominator from 0.8 to 0.72
            legend_ax_center.text(
                0,
                r + 0.05,
                f"{pct_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

        # Set equal aspect ratio and clean up axes
        legend_ax_center.set_xlim(-1.1, 1.1)
        legend_ax_center.set_ylim(-1.1, 1.1)
        legend_ax_center.set_aspect("equal")
        legend_ax_center.axis("off")

        # Add title
        legend_ax_center.text(
            0,
            -1.05,
            "Wind Rose",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="normal",  # Changed from "bold" to "normal"
        )

        print(f"Wind rose created with {total_valid_data} valid data points")

    else:
        # Fallback if no wind data
        legend_ax_center.text(
            0.5,
            0.5,
            "Wind Rose\n(No Data)",
            ha="center",
            va="center",
            transform=legend_ax_center.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )
        legend_ax_center.axis("off")

    # Right section: Smoothing information (1/3 of width)
    legend_ax_right = fig.add_axes(
        [0.67, bottom_y, 0.28, bottom_height]
    )  # Adjusted positioning for 1:1 ratio
    legend_ax_right.text(
        0.5,
        0.5,
        "Data Smoothing\n\n• Temperature: 5-min\n  moving average\n• Humidity: 5-min\n  moving average\n• Wind Speed: 5-min\n  moving average\n• Wind Direction: 5-min\n  circular average",
        ha="center",
        va="center",
        transform=legend_ax_right.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3),
    )
    legend_ax_right.axis("off")

    plt.savefig(f"PABI_daily_{plot_date}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Plot saved as PABI_daily_{plot_date}.png")


def plot_5min_daily(dt, precip_dt, plot_date):

    matplotlib.rcParams["axes.unicode_minus"] = False  # 防止负号显示为方块

    # 选取指定日期的数据
    date_mask = dt.index.date == pd.to_datetime(plot_date).date()
    daily_data = dt[date_mask].copy()
    precip_mask = precip_dt.index.date == pd.to_datetime(plot_date).date()
    daily_precip = precip_dt[precip_mask].copy()

    # 打印数据情况
    print(f"Plotting data for {plot_date}")
    print(f"Found {len(daily_data)} data points for this day")
    if len(daily_data) == 0:
        print(f"No data found for {plot_date}")
        return

    # Separate hourly data (with temperature) from 5-minute data (wind only)
    temp_rh_data = daily_data.dropna(
        subset=["temp_c", "rh"]
    )  # Only rows with temperature
    wind_data = daily_data[["drct", "sknt_ms"]].dropna()  # All rows with wind data
    precip_data = daily_precip.dropna(
        subset=["precip_mm"]
    )  # Only rows with precipitation

    print(f"Found {len(temp_rh_data)} hourly observations")
    print(f"Found {len(wind_data)} wind observations")

    if len(temp_rh_data) == 0 and len(wind_data) == 0:
        print(f"No usable data found for {plot_date}")
        return

    # FOR DEBUGGING: save daily data to CSV to see if it looks correct
    # daily_data.to_csv(f"pabi5min_daily_{plot_date}.csv")

    # 创建画布
    fig = plt.figure(figsize=(12, 12))

    # 创建四个子图
    ax1 = plt.subplot(4, 1, 1)  # Temperature
    ax2 = plt.subplot(4, 1, 2)  # Humidity
    ax3 = plt.subplot(4, 1, 3)  # Precipitation
    ax4 = plt.subplot(4, 1, 4)  # Wind Speed + Direction (combined)

    # 依次画每一个子图
    # Temperature subplot - hourly data with markers
    if len(temp_rh_data) > 0:
        # # Handle missing data gaps for hourly data - break line if time gap > 2 hours
        # time_diffs = hourly_data.index.to_series().diff()
        # large_gaps = time_diffs > pd.Timedelta(hours=2)

        temp_data = temp_rh_data["temp_c"].copy()
        # for i in range(1, len(hourly_data)):
        #     if large_gaps.iloc[i]:
        #         temp_data.iloc[i] = np.nan

        ax1.plot(
            temp_rh_data.index,
            temp_data,
            color="red",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Temperature",
        )
    else:
        ax1.text(
            0.5,
            0.5,
            "No temperature data available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    ax1.set_ylabel("Temperature (°C)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax1.grid(True, alpha=0.3)

    # 相对湿度
    if len(temp_rh_data):
        # time_diffs = hourly_data.index.to_series().diff()
        # large_gaps = time_diffs > pd.Timedelta(hours=2)

        rh_data = temp_rh_data["rh"].copy()
        # for i in range(1, len(hourly_data)):
        # if large_gaps.iloc[i]:
        #     rh_data.iloc[i] = np.nan

        # Plot only smoothed line with markers (no need for original since it's hourly data)
        ax2.plot(
            temp_rh_data.index,
            rh_data,
            color="blue",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Humidity",
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No humidity data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    ax2.set_ylabel("Relative\nHumidity (%)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # 每小时降水量图 - 累计降水曲线
    if len(precip_data) > 0 and "precip_mm" in precip_data.columns:

        # Calculate cumulative precipitation from start of day
        cumulative_precip = precip_data["precip_mm"].cumsum()

        # Plot cumulative precipitation as a curve with markers
        ax3.plot(
            precip_data.index,
            cumulative_precip,
            color="green",
            linewidth=2,
            marker="o",
            markersize=2,
        )

        # Fill the area under the curve for better visualization
        ax3.fill_between(precip_data.index, cumulative_precip, alpha=0.3, color="green")

        ax3.set_ylabel("Cumulative\nPrecipitation (mm)", color="green")
        ax3.tick_params(axis="y", labelcolor="green")
        ax3.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
        ax3.grid(True, alpha=0.3)

        # Set y-axis to start from 0
        ax3.set_ylim(bottom=0)

        # Add final total as text
        daily_total = cumulative_precip.iloc[-1] if len(cumulative_precip) > 0 else 0
        daily_total_inches = precip_data["precip_in"].sum()
        ax3.text(
            0.01,
            0.94,
            f"Daily Total: {daily_total:.2f} mm ({daily_total_inches:.3f} in)",
            transform=ax3.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            verticalalignment="top",
            horizontalalignment="left",
        )

        print(
            f"Daily precipitation total: {daily_total:.2f} mm ({daily_total_inches:.3f} inches)"
        )

    else:
        ax3.text(
            0.5,
            0.5,
            "No precipitation data available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_ylabel("Precipitation (No Data)", color="green")
        ax3.tick_params(axis="x", labelbottom=False)

    # 风速风向图
    if (
        len(wind_data) > 0
        and "sknt_ms" in wind_data.columns
        and "drct" in wind_data.columns
    ):
        # 准备数据
        wind_speed = wind_data["sknt_ms"].copy()
        wind_dir = wind_data["drct"].copy()

        # Apply smoothing
        wind_speed_smoothed = wind_speed.rolling(
            window=5, center=True, min_periods=3
        ).mean()
        wind_dir_smoothed = smooth_wind_direction(wind_dir, window=5)

        # 原始风向的颜色
        colors_original = []
        for i in range(len(wind_dir)):
            if (
                pd.isna(wind_dir.iloc[i])
                or pd.isna(wind_speed.iloc[i])
                or wind_speed.iloc[i] <= 0.1
            ):
                colors_original.append("lightgray")  # 无风时显示灰色
            else:
                hue = wind_dir.iloc[i] / 360.0
                colors_original.append(cmocean.cm.phase(hue))

        # 画原始风向风速-小圆点
        valid_mask = ~pd.isna(wind_speed) & ~pd.isna(wind_dir)  # 修正这里
        ax4.scatter(
            wind_data.index[valid_mask],
            wind_speed[valid_mask],
            c=[
                colors_original[i]
                for i in range(len(colors_original))
                if valid_mask.iloc[i]
            ],
            s=8,  # Small size
            alpha=0.3,
            edgecolors="none",
            label="Original data",
        )

        # 平滑风向的颜色
        colors_smoothed = []
        for i in range(len(wind_dir_smoothed)):
            if (
                pd.isna(wind_dir_smoothed.iloc[i])
                or pd.isna(wind_speed_smoothed.iloc[i])
                or wind_speed_smoothed.iloc[i] <= 0.1
            ):
                colors_smoothed.append("lightgray")  # 无风时显示灰色
            else:
                hue = wind_dir_smoothed.iloc[i] / 360.0
                colors_smoothed.append(cmocean.cm.phase(hue))

        # 画平滑后的风向风速-大圆点
        valid_mask_smooth = ~pd.isna(wind_speed_smoothed) & ~pd.isna(
            wind_dir_smoothed
        )  # 修正这里
        ax4.scatter(
            wind_data.index[valid_mask_smooth],
            wind_speed_smoothed[valid_mask_smooth],
            c=[
                colors_smoothed[i]
                for i in range(len(colors_smoothed))
                if valid_mask_smooth.iloc[i]
            ],
            s=20,  # Larger size
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            label="5 min smoothed data",
        )

        # Add a subtle line connecting the smoothed points for better continuity
        ax4.plot(
            wind_data.index,
            wind_speed_smoothed,
            color="gray",
            linewidth=0.5,
            alpha=0.3,
            zorder=0,
        )

        print(f"Wind scatter plot created with {len(wind_data)} data points")
    else:
        ax4.text(
            0.5,
            0.5,
            "No wind data available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    ax4.set_ylabel("Wind Speed(m/s)", color="purple")
    ax4.tick_params(axis="y", labelcolor="purple")
    ax4.set_xlabel("Time (America/Anchorage)")
    ax4.grid(True, alpha=0.3)

    start_of_day = daily_data.index.min().replace(hour=0, minute=0, second=0)
    end_of_day = daily_data.index.max().replace(hour=23, minute=59, second=59)

    # 为所有子图设置相同的x轴格式
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(start_of_day, end_of_day)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(True, alpha=0.3, which="major", axis="x", linestyle="--")

        # 仅在最下面的子图显示x轴标签
        if ax == ax4:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.tick_params(axis="x", rotation=0, labelsize=10)
            ax.set_xlabel("Time (America/Anchorage)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    # 在最上面的子图的上方添加x轴标签
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(start_of_day, end_of_day)
    ax1_top.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax1_top.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax1_top.tick_params(axis="x", rotation=0, labelsize=10)

    # 调整位置
    # 调整子图在画布中的分布
    fig.subplots_adjust(
        # hspace=0,  # 子图之间的垂直间距为0
        # wspace=0,  # 子图之间的水平间距为0
        top=0.92,
        bottom=0.25,
        left=0.05,
        right=0.95,
    )

    # 每个子图位置的句柄
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()

    # 计算每个子图的高度
    total_height = 0.70
    subplot_h = total_height / 4

    # 设置子图位置
    y_top = 0.93
    ax1.set_position([pos1.x0, y_top - subplot_h, pos1.width, subplot_h])
    ax2.set_position([pos2.x0, y_top - 2 * subplot_h, pos2.width, subplot_h])
    ax3.set_position([pos3.x0, y_top - 3 * subplot_h, pos3.width, subplot_h])
    ax4.set_position([pos4.x0, y_top - 4 * subplot_h, pos4.width, subplot_h])

    plt.suptitle(f"PABI Weather Data - {plot_date} (Mixed Resolution)", fontsize=16)

    # 底部区域
    bottom_height = 0.18  # More space for bottom area
    bottom_y = 0.02  # Close to bottom edge

    # 左边：风向图例
    legend_ax_left = fig.add_axes(
        [0.05, bottom_y, 0.28, bottom_height]
    )  # Adjusted for 1:1 ratio with elongated content

    radius_inner = 0.28  # Inner radius
    radius_outer = 0.55  # Outer radius

    # Create the circular segments for wind direction color legend
    for i in range(36):
        # Calculate angles for this segment (0° is North, clockwise)
        # Fixed: Need to rotate clockwise from North, so subtract from π/2
        angle_start = np.pi / 2 - i * 10 * np.pi / 180  # Start from North, go clockwise
        angle_end = np.pi / 2 - (i + 1) * 10 * np.pi / 180

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

        # Get color from phase colormap - match the actual wind direction angle
        # Note: Wind direction is "from" direction, but we want to show it at the correct compass position
        # i=0 is at North position (-π/2), and should show 0° wind (North wind) color
        # But we need to map the visual position to the wind direction correctly
        compass_angle = (
            i * 10
        )  # The compass position in degrees (0=North, 90=East, etc.)
        # For wind direction color mapping, we want the color that would appear at this compass position
        color_value = compass_angle / 360.0
        color = cmocean.cm.phase(color_value)

        # Draw the wedge
        legend_ax_left.fill(
            x_wedge, y_wedge, color=color, edgecolor="white", linewidth=0.5
        )

    # Add direction labels with consistent font style
    label_distance = radius_outer + 0.15  # Closer labels to avoid overlap
    legend_ax_left.text(
        0,
        label_distance,
        "N",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    legend_ax_left.text(
        label_distance,
        0,
        "E",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    legend_ax_left.text(
        0,
        -label_distance,
        "S",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    legend_ax_left.text(
        -label_distance,
        0,
        "W",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )

    # Set equal aspect ratio and clean up axes
    legend_ax_left.set_xlim(-0.9, 0.9)  # Smaller bounds to reduce overlap
    legend_ax_left.set_ylim(-0.9, 0.9)
    legend_ax_left.set_aspect("equal")
    legend_ax_left.axis("off")  # Hide axes
    legend_ax_left.text(
        0, -0.85, "Wind Direction", ha="center", va="center", fontsize=9
    )

    # 中间：风玫瑰图
    legend_ax_center = fig.add_axes(
        [0.36, bottom_y, 0.28, bottom_height]
    )  # Adjusted positioning for 1:1 ratio

    # Create 16-sector wind rose using 5-minute wind data
    if (
        len(wind_data) > 0
        and "drct" in wind_data.columns
        and "sknt_ms" in wind_data.columns
    ):
        # Define 16 sectors (22.5° each)
        sectors = 16
        sector_angle = 360.0 / sectors

        # Initialize arrays for wind rose data
        sector_counts = np.zeros(sectors)
        sector_avg_speed = np.zeros(sectors)

        # Get valid wind data (exclude calm periods and NaN)
        valid_mask = (
            (~pd.isna(wind_data["drct"]))
            & (~pd.isna(wind_data["sknt_ms"]))
            & (wind_data["sknt_ms"] > 0.1)
        )
        valid_directions = wind_data["drct"][valid_mask].values
        valid_speeds = wind_data["sknt_ms"][valid_mask].values

        # Classify data into sectors
        for direction, speed in zip(valid_directions, valid_speeds):
            # Convert direction to sector index (0° = North)
            sector_idx = int((direction + sector_angle / 2) % 360 / sector_angle)
            sector_counts[sector_idx] += 1
            sector_avg_speed[sector_idx] += speed

        # Calculate average speeds (avoid division by zero)
        for i in range(sectors):
            if sector_counts[i] > 0:
                sector_avg_speed[i] /= sector_counts[i]

        # Normalize counts to percentages
        total_valid_data = np.sum(sector_counts)
        if total_valid_data > 0:
            sector_percentages = (sector_counts / total_valid_data) * 100
        else:
            sector_percentages = np.zeros(sectors)

        # Create polar plot for wind rose
        theta = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        # Offset by 90° to put North at top and rotate clockwise
        theta_plot = (np.pi / 2 - theta) % (2 * np.pi)

        # Create the wind rose bars
        max_percentage = (
            np.max(sector_percentages) if np.max(sector_percentages) > 0 else 1
        )

        for i in range(sectors):
            if sector_percentages[i] > 0:
                # Radial length proportional to frequency (reduced to 90% size)
                radius = (
                    sector_percentages[i] / max_percentage * 0.72
                )  # Changed from 0.8 to 0.72 (0.8 * 0.9)

                # Color based on wind direction (same as wind direction color band and color ring)
                # Calculate the center angle of this sector in degrees
                sector_center_angle = i * sector_angle  # 0° = North, 90° = East, etc.
                # Use same phase color mapping as wind direction color band
                color_value = sector_center_angle / 360.0
                color = cmocean.cm.phase(color_value)

                # Calculate sector edges
                angle_start = theta_plot[i] - np.pi / sectors
                angle_end = theta_plot[i] + np.pi / sectors

                # Create wedge coordinates
                angles = np.linspace(angle_start, angle_end, 10)
                x_outer = radius * np.cos(angles)
                y_outer = radius * np.sin(angles)
                x_inner = np.zeros_like(angles)
                y_inner = np.zeros_like(angles)

                # Combine to form the wedge
                x_wedge = np.concatenate([x_inner, x_outer[::-1]])
                y_wedge = np.concatenate([y_inner, y_outer[::-1]])

                # Draw the wedge
                legend_ax_center.fill(
                    x_wedge,
                    y_wedge,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.8,
                )

        # Add direction labels
        direction_labels = [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ]
        label_radius = (
            0.81  # Reduced from 0.9 to 0.81 (0.9 * 0.9) to match the smaller wind rose
        )

        for i, label in enumerate(direction_labels):
            # Skip the "N" label (i=0) to avoid covering frequency numbers
            if i == 0:  # Skip North label
                continue

            angle = theta_plot[i]
            x_label = label_radius * np.cos(angle)
            y_label = label_radius * np.sin(angle)
            legend_ax_center.text(
                x_label,
                y_label,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="normal",  # Changed from "bold" to "normal" to match other labels
            )

        # Add concentric circles for percentage scale (reduced to 90% size)
        for r in [
            0.18,
            0.36,
            0.54,
            0.72,
        ]:  # Changed from [0.2, 0.4, 0.6, 0.8] to 90% size
            circle = plt.Circle(
                (0, 0), r, fill=False, color="gray", alpha=0.3, linewidth=0.5
            )
            legend_ax_center.add_patch(circle)

        # Add percentage labels
        max_pct = max_percentage if max_percentage > 0 else 1
        for i, r in enumerate(
            [0.18, 0.36, 0.54, 0.72]
        ):  # Updated to match new circle radii
            pct_value = (r / 0.72) * max_pct  # Changed denominator from 0.8 to 0.72
            legend_ax_center.text(
                0,
                r + 0.05,
                f"{pct_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

        # Set equal aspect ratio and clean up axes
        legend_ax_center.set_xlim(-1.1, 1.1)
        legend_ax_center.set_ylim(-1.1, 1.1)
        legend_ax_center.set_aspect("equal")
        legend_ax_center.axis("off")

        # Add title
        legend_ax_center.text(
            0,
            -1.05,
            "Wind Rose",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="normal",  # Changed from "bold" to "normal"
        )

        print(f"Wind rose created with {total_valid_data} valid data points")

    else:
        # Fallback if no wind data
        legend_ax_center.text(
            0.5,
            0.5,
            "Wind Rose\n(No Data)",
            ha="center",
            va="center",
            transform=legend_ax_center.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )
        legend_ax_center.axis("off")

    # 右边：数据说明
    legend_ax_right = fig.add_axes(
        [0.67, bottom_y, 0.28, bottom_height]
    )  # Adjusted positioning for 1:1 ratio
    legend_ax_right.text(
        0.5,
        0.5,
        f"Data Information\n\n• Mixed Resolution Data\n• Hourly: {len(temp_rh_data)} obs\n• Wind (5-min): {len(wind_data)} obs\n• Temp/Humidity: hourly\n• Wind: 5-minute resolution\n• Wind: Speed=position,\n  Direction=color",
        ha="center",
        va="center",
        transform=legend_ax_right.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.3),
    )
    legend_ax_right.axis("off")

    plt.savefig(
        f"PABI_daily_1h_windsmoothed_{plot_date}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print(f"Plot saved as PABI_daily_1h_windsmoothed_{plot_date}.png")
