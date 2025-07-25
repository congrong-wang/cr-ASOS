import cmocean
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd

from cr_smps.analysis import _plot_heatmap


# Function to smooth wind direction (circular/angular data)
def _smooth_wind_direction(
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


def new_daily_plot_w_SMPS(
    dt,
    dt_smps,
    plot_date,
    output_dir="./plots/",
    SMPS_plotting_timezone=None,
):
    # Create 5 axis for the daily plot
    matplotlib.rcParams["axes.unicode_minus"] = False  # Prevent minus sign from showing
    fig, axs = plt.subplots(
        5,
        1,
        figsize=(12, 12),
        # sharex=True,
    )

    date_mask = dt.index.date == pd.to_datetime(plot_date).date()
    daily_data = dt[date_mask].copy()

    _plot_heatmap(
        ax=axs[0],
        dataset=dt_smps,
        time_range=plot_date,
        output_time_zone=SMPS_plotting_timezone,
    )
    temp_data = daily_data["temp_c"].copy()
    _plot_temperature(axs[1], temp_data)
    rh_data = daily_data["rh"].copy()
    _plot_humidity(axs[2], rh_data)
    precip_data = daily_data["precip_mm"].copy()
    _plot_precipitation(axs[3], precip_data)
    wind_data = daily_data[["sknt_ms", "drct"]].copy()
    _plot_wind(axs[4], wind_data)

    start_of_day = daily_data.index.min().replace(hour=0, minute=0, second=0)
    end_of_day = daily_data.index.max().replace(hour=23, minute=59, second=59)

    # 为所有子图设置相同的x轴格式
    for ax in axs:
        if ax != axs[0]:
            ax.set_xlim(start_of_day, end_of_day)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(True, alpha=0.3, which="major", axis="x", linestyle="--")

        # 仅在最下面的子图显示x轴标签
        if ax == axs[4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.tick_params(axis="x", rotation=0, labelsize=10)
            ax.set_xlabel("Time (America/Anchorage)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    # 在最上面的子图的上方添加x轴标签
    ax1_top = axs[0].twiny()
    # ax1_top.set_xlim(start_of_day, end_of_day)
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
    pos1 = axs[0].get_position()
    pos2 = axs[1].get_position()
    pos3 = axs[2].get_position()
    pos4 = axs[3].get_position()
    pos5 = axs[4].get_position()

    # 计算每个子图的高度
    total_height = 0.70
    subplot_h = total_height / 5  # 改为5个子图，而不是4个

    # 设置子图位置
    y_top = 0.93
    axs[0].set_position([pos1.x0, y_top - subplot_h, pos1.width, subplot_h])
    axs[1].set_position([pos2.x0, y_top - 2 * subplot_h, pos2.width, subplot_h])
    axs[2].set_position([pos3.x0, y_top - 3 * subplot_h, pos3.width, subplot_h])
    axs[3].set_position([pos4.x0, y_top - 4 * subplot_h, pos4.width, subplot_h])
    axs[4].set_position([pos5.x0, y_top - 5 * subplot_h, pos5.width, subplot_h])

    plt.suptitle(f"SMPS Heatmap with PABI Weather Data - {plot_date}", fontsize=16)

    # 底部区域
    bottom_height = 0.18  # More space for bottom area
    bottom_y = 0.02  # Close to bottom edge

    # 左边：风向图例
    legend_ax_left = fig.add_axes(
        [0.05, bottom_y, 0.28, bottom_height]
    )  # Adjusted for 1:1 ratio with elongated content
    _plot_wind_dir_legend(legend_ax_left)

    # 中间：风玫瑰图
    legend_ax_center = fig.add_axes(
        [0.36, bottom_y, 0.28, bottom_height]
    )  # Adjusted positioning for 1:1 ratio
    _plot_wind_rose(legend_ax_center, wind_data)

    # Save the figure
    fname = f"PABI_daily_{plot_date}_new.png"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, fname)
    else:
        save_path = fname
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def new_daily_plot(dt, plot_date):
    # Create 4 axis for the daily plot
    matplotlib.rcParams["axes.unicode_minus"] = False  # Prevent minus sign from showing
    fig, axs = plt.subplots(4, 1, figsize=(12, 24), sharex=True)

    date_mask = dt.index.date == pd.to_datetime(plot_date).date()
    daily_data = dt[date_mask].copy()

    temp_data = daily_data["temp_c"].copy()
    _plot_temperature(axs[0], temp_data)
    rh_data = daily_data["rh"].copy()
    _plot_humidity(axs[1], rh_data)
    precip_data = daily_data["precip_mm"].copy()
    _plot_precipitation(axs[2], precip_data)
    wind_data = daily_data[["sknt_ms", "drct"]].copy()
    _plot_wind(axs[3], wind_data)

    start_of_day = daily_data.index.min().replace(hour=0, minute=0, second=0)
    end_of_day = daily_data.index.max().replace(hour=23, minute=59, second=59)

    # 为所有子图设置相同的x轴格式
    for ax in axs:
        ax.set_xlim(start_of_day, end_of_day)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(True, alpha=0.3, which="major", axis="x", linestyle="--")

        # 仅在最下面的子图显示x轴标签
        if ax == axs[3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.tick_params(axis="x", rotation=0, labelsize=10)
            ax.set_xlabel("Time (America/Anchorage)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    # 在最上面的子图的上方添加x轴标签
    ax1_top = axs[0].twiny()
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
    pos1 = axs[0].get_position()
    pos2 = axs[1].get_position()
    pos3 = axs[2].get_position()
    pos4 = axs[3].get_position()

    # 计算每个子图的高度
    total_height = 0.70
    subplot_h = total_height / 5  # 改为5个子图，而不是4个

    # 设置子图位置
    y_top = 0.93
    axs[0].set_position([pos1.x0, y_top - subplot_h, pos1.width, subplot_h])
    axs[1].set_position([pos2.x0, y_top - 2 * subplot_h, pos2.width, subplot_h])
    axs[2].set_position([pos3.x0, y_top - 3 * subplot_h, pos3.width, subplot_h])
    axs[3].set_position([pos4.x0, y_top - 4 * subplot_h, pos4.width, subplot_h])

    plt.suptitle(f"PABI Weather Data - {plot_date}", fontsize=16)

    # 底部区域
    bottom_height = 0.18  # More space for bottom area
    bottom_y = 0.02  # Close to bottom edge

    # 左边：风向图例
    legend_ax_left = fig.add_axes(
        [0.05, bottom_y, 0.28, bottom_height]
    )  # Adjusted for 1:1 ratio with elongated content
    _plot_wind_dir_legend(legend_ax_left)

    # 中间：风玫瑰图
    legend_ax_center = fig.add_axes(
        [0.36, bottom_y, 0.28, bottom_height]
    )  # Adjusted positioning for 1:1 ratio
    _plot_wind_rose(legend_ax_center, wind_data)

    # save whole plot for debugging
    plt.savefig(f"PABI_daily_{plot_date}_new.png", dpi=300, bbox_inches="tight")


def _plot_temperature(ax, dt):
    dt = dt.dropna()  # Remove NaN to avoid discontinuities in the line plot
    if len(dt) > 0:
        ax.plot(
            dt.index,
            dt,
            color="red",
            linewidth=1,
            marker="o",
            markersize=1,
            label="Temperature",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No temperature data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_ylabel("Temperature (°C)", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax.grid(True, alpha=0.3)


def _plot_humidity(ax, dt):
    dt = dt.dropna()  # Remove NaN to avoid discontinuities in the line plot
    if len(dt) > 0:
        ax.plot(
            dt.index,
            dt,
            color="blue",
            linewidth=1,
            marker="o",
            markersize=1,
            label="Humidity",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No humidity data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_ylabel("Relative\nHumidity (%)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)


def _plot_precipitation(ax, dt):
    """
    Arguments:
    ax: matplotlib axis object to plot on
    dt: Precipitation data in mm
    """
    dt = dt.dropna()  # Remove NaN to avoid discontinuities in the line plot

    if len(dt) > 0:
        # Calculate cumulative precipitation from start of day
        cumulative_precip = dt.cumsum()
        # Plot cumulative precipitation as a curve with markers
        ax.plot(
            dt.index,
            cumulative_precip,
            color="green",
            linewidth=2,
            # marker="o",
            # markersize=2,
        )

        # Fill the area under the curve for better visualization
        ax.fill_between(dt.index, cumulative_precip, alpha=0.3, color="green")

        ax.set_ylabel("Cumulative\nPrecipitation (mm)", color="green")
        ax.tick_params(axis="y", labelcolor="green")
        ax.tick_params(axis="x", labelbottom=False)  # Hide x-axis labels
        ax.grid(True, alpha=0.3)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Add final total as text
        daily_total = cumulative_precip.iloc[-1] if len(cumulative_precip) > 0 else 0
        ax.text(
            0.01,
            0.94,
            f"Daily Total: {daily_total:.2f} mm",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            verticalalignment="top",
            horizontalalignment="left",
        )

        print(f"Daily precipitation total: {daily_total:.2f} mm")

    else:
        ax.text(
            0.5,
            0.5,
            "No precipitation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_ylabel("Precipitation (No Data)", color="green")
        ax.tick_params(axis="x", labelbottom=False)


def _plot_wind(ax, dt):
    # 风速风向图
    if len(dt) > 0 and "sknt_ms" in dt.columns and "drct" in dt.columns:
        # 准备数据
        wind_speed = dt["sknt_ms"].copy()
        wind_dir = dt["drct"].copy()

        # Apply smoothing
        wind_speed_smoothed = wind_speed.rolling(
            window=5, center=True, min_periods=3
        ).mean()
        wind_dir_smoothed = _smooth_wind_direction(wind_dir, window=5)

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
        ax.scatter(
            dt.index[valid_mask],
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
        valid_mask_smooth = ~pd.isna(wind_speed_smoothed) & ~pd.isna(wind_dir_smoothed)
        ax.scatter(
            dt.index[valid_mask_smooth],
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
        ax.plot(
            dt.index,
            wind_speed_smoothed,
            color="gray",
            linewidth=0.5,
            alpha=0.3,
            zorder=0,
        )

        print(f"Wind scatter plot created with {len(dt)} data points")
    else:
        ax.text(
            0.5,
            0.5,
            "No wind data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_ylabel("Wind Speed(m/s)", color="purple")
    ax.tick_params(axis="y", labelcolor="purple")
    ax.set_xlabel("Time (America/Anchorage)")
    ax.grid(True, alpha=0.3)


def _plot_wind_dir_legend(ax):

    r_in = 0.28  # Inner radius
    r_out = 0.55  # Outer radius

    # Create the circular segments for wind direction color legend
    for i in range(36):
        # Calculate angles for this segment (0° is North, clockwise)
        # Fixed: Need to rotate clockwise from North, so subtract from π/2
        angle_start = np.pi / 2 - i * 10 * np.pi / 180  # Start from North, go clockwise
        angle_end = np.pi / 2 - (i + 1) * 10 * np.pi / 180

        # Create wedge coordinates
        angles = np.linspace(angle_start, angle_end, 10)

        # Outer arc
        x_outer = r_out * np.cos(angles)
        y_outer = r_out * np.sin(angles)

        # Inner arc (reversed for proper polygon)
        x_inner = r_in * np.cos(angles[::-1])
        y_inner = r_in * np.sin(angles[::-1])

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
        ax.fill(x_wedge, y_wedge, color=color, edgecolor="white", linewidth=0.5)

    # Add direction labels with consistent font style
    label_distance = r_out + 0.15  # Closer labels to avoid overlap
    ax.text(
        0,
        label_distance,
        "N",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    ax.text(
        label_distance,
        0,
        "E",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    ax.text(
        0,
        -label_distance,
        "S",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )
    ax.text(
        -label_distance,
        0,
        "W",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="normal",
    )

    # Set equal aspect ratio and clean up axes
    ax.set_xlim(-0.9, 0.9)  # Smaller bounds to reduce overlap
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes
    ax.text(0, -0.85, "Wind Direction", ha="center", va="center", fontsize=9)


def _plot_wind_rose(ax, dt):

    # Create 16-sector wind rose
    if "drct" in dt.columns and "sknt_ms" in dt.columns:
        # Define 16 sectors (22.5° each)
        sectors = 16
        sector_angle = 360.0 / sectors

        # Initialize arrays for wind rose data
        sector_counts = np.zeros(sectors)
        sector_avg_speed = np.zeros(sectors)

        # Get valid wind data (exclude calm periods and NaN)
        valid_mask = (
            (~pd.isna(dt["drct"])) & (~pd.isna(dt["sknt_ms"])) & (dt["sknt_ms"] > 0.1)
        )
        valid_directions = dt["drct"][valid_mask].values
        valid_speeds = dt["sknt_ms"][valid_mask].values

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
                ax.fill(
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
            ax.text(
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
            ax.add_patch(circle)

        # Add percentage labels
        max_pct = max_percentage if max_percentage > 0 else 1
        for i, r in enumerate(
            [0.18, 0.36, 0.54, 0.72]
        ):  # Updated to match new circle radii
            pct_value = (r / 0.72) * max_pct  # Changed denominator from 0.8 to 0.72
            ax.text(
                0,
                r + 0.05,
                f"{pct_value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

        # Set equal aspect ratio and clean up axes
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add title
        ax.text(
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
        ax.text(
            0.5,
            0.5,
            "Wind Rose\n(No Data)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )
        ax.axis("off")
