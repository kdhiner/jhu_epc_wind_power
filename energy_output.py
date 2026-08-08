'''
This code will calculate the annual energy output of a roof-mounted system of vertical axis wind turbines (VAWT)

We will consider the roof-mounted VAWTs are deployed in Washington, D.C.
We will download weather data to provide the wind speeds 
We will consider a specific wind turbine by Urban Green Energy, the 4K 

Based on the actual wind speed data and the actual power curve of the UGE-4K, 
we will calculate the hypothetical power output for the time period under consideration.
We will plot the power curve over the course of the time periods under consideration. 
'''

import numpy as np
import pandas as pd
from scipy.stats import weibull_min, kstest

import matplotlib.pyplot as plt

LOCATION = "Seattle, WA"  # Change this to "Washington, D.C." or "Chicago, IL" as needed
print(f"\nCalculating energy output for {LOCATION}")

if LOCATION == "Washington, D.C.":
    WEATHER_FILE = 'open-meteo-38.91N77.07W12m_2025_edit.csv' #Wasington, D.C. weather data for 2025
elif LOCATION == "Chicago, IL":
    WEATHER_FILE = 'open-meteo-41.86N87.65W179m_2025_edit.csv' #Chicago, IL weather data for 2025
elif LOCATION == "Seattle, WA":
    WEATHER_FILE = 'open-meteo-47.63N122.32W59m_2025_edit.csv' #Seattle, WA weather data for 2025
elif LOCATION == "Minneapolis, MN":
    WEATHER_FILE = 'open-meteo-44.96N93.21W261m_2025_edit.csv' #Minneapolis, MN weather data for 2025

def speed_to_power_rating(speed: float, lookup: dict = None) -> float:
    """Translate a wind speed to a power rating using the bin dictionary.
    
    Args:
        speed (float): The wind speed in m/s.
        lookup (dict): A dictionary mapping wind speed ranges to power ratings.
    
    Returns:
        float: The power rating corresponding to the wind speed.
    """
    for (low, high), power in lookup.items():
        if low <= speed < high:
            return power
    return 0

def plot_power_output(weather_data: pd.DataFrame, outpath: str = 'power_output_over_time_2025.png', month: str = None) -> None:
    """Plot power output time series for 10m and 100m and save to file.

    Args:
        weather_data (pd.DataFrame): DataFrame containing 'time', 'power_output_10m', and 'power_output_100m'.
        outpath (str): File path to save the figure.
        month (str): Optional month name to display in the title.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(weather_data['time'], weather_data['power_output_10m'], label='Power Output at 10m', color='blue')
    plt.plot(weather_data['time'], weather_data['power_output_100m'], label='Power Output at 100m', color='orange')
    title = f'Power Output Over Time {LOCATION} {month} 2025' if month else f'Power Output Over Time {LOCATION} 2025'
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Power Output (kW)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(outpath)

def fit_weibull_distribution(weather_data: pd.DataFrame, height: str = '10m') -> tuple:
    """Fit a Weibull distribution to wind speed data and perform a KS test.

    Returns: (shape, loc, scale, ks_stat, ks_pvalue)
    """
    if height not in ('10m', '100m'):
        raise ValueError("height must be '10m' or '100m'")

    column = f'wind_speed_{height}'
    wind_data = weather_data[column].dropna()

    shape, loc, scale = weibull_min.fit(wind_data, floc=0)
    cdf_func = lambda x: weibull_min.cdf(x, shape, loc, scale)
    ks_stat, ks_pvalue = kstest(wind_data, cdf_func)

    print(f"Weibull parameters for {height}: shape={round(shape, 2)}, loc={loc}, scale={round(scale, 2)}")
    print(f"KS test for {height}: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
    alpha = 0.001
    if ks_pvalue < alpha:
        print(f"Reject null hypothesis that the distributions are identical at alpha={alpha}: Weibull is NOT a good fit for {height}.")
    else:
        print(f"Fail to reject null hypothesis that the distributions are identical at alpha={alpha}: Weibull is a plausible fit for {height}.")

    return shape, loc, scale, ks_stat, ks_pvalue


def plot_wind_speed_histogram(weather_data: pd.DataFrame, outpath: str = 'wind_speed_histogram_2025.png', height: str = '10m', bins: int = 20) -> None:
    """Plot the histogram distribution of wind speeds with the fitted Weibull and save to file.

    This function calls fit_weibull_distribution to obtain parameters to plot the fit.
    """
    if height not in ('10m', '100m'):
        raise ValueError("height must be '10m' or '100m'")

    print(f"\nPlotting wind speed histogram for {height} at {LOCATION} 2025")

    column = f'wind_speed_{height}'
    wind_data = weather_data[column].dropna()

    shape, loc, scale, ks_stat, ks_pvalue = fit_weibull_distribution(weather_data, height=height)

    x = np.linspace(0, wind_data.max(), 100)
    pdf = weibull_min.pdf(x, shape, loc, scale)

    plt.figure(figsize=(10, 6))
    plt.hist(wind_data, bins=bins, density=True, color='green', edgecolor='black', alpha=0.7)

    # Report the bin width and the number of bins used in the histogram
    bin_width = (wind_data.max() - wind_data.min()) / bins
    print(f"Histogram for {height}: bin width={bin_width:.2f}, number of bins={bins}")

    # Draw the weibull distribution fit line
    plt.plot(x, pdf, 'r-', lw=2, label='Weibull Fit')
    # Annotate the plot with the Weibull parameters
    plt.text(0.95, 0.95, f'Weibull\nShape: {shape:.2f}\nScale: {scale:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.legend()
    plt.title(f'Wind Speed Distribution at {height} {LOCATION} 2025')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath)


if __name__ == "__main__":
    weather_data = pd.read_csv(WEATHER_FILE)
    weather_data['time'] = pd.to_datetime(weather_data['time'])

    # Display the distribution of wind speeds at both 10m and 100m
    print(f"\nWind speed distribution at 10m for {LOCATION} 2025:")
    print(weather_data['wind_speed_10m'].describe())
    print(f"\nWind speed distribution at 100m for {LOCATION} 2025:")
    print(weather_data['wind_speed_100m'].describe())

    # I need a binning for the power curve so that any wind speed can be converted to a power unit
    # I am going to build this manually by examining the documentation for the UGE-4K wind turbine. The power curve is as follows:
    speed_to_power = {
        (0, 3.5): 0,
        (3.5, 4): 0.1,
        (4, 5): 0.3,
        (5, 6): 0.5,
        (6, 7): 0.75,
        (7, 8): 1,
        (8, 9): 1.5,
        (9, 10): 2,
        (10, 11): 2.75,
        (11, 12): 3.5,
        (12, 30): 4,
        (30, float('inf')): 0  # Above cut-out speed, power output is 0
    }

    # Plot the wind speed histogram for both 10m and 100m
    plot_wind_speed_histogram(weather_data, outpath=f'{LOCATION.lower().replace(", ", "_")}_wind_speed_histogram_10m_2025.png', height='10m')
    plot_wind_speed_histogram(weather_data, outpath=f'{LOCATION.lower().replace(", ", "_")}_wind_speed_histogram_100m_2025.png', height='100m')

    # Apply the speed_to_power_rating function to the wind speed data
    weather_data['power_output_10m'] = weather_data['wind_speed_10m'].apply(speed_to_power_rating, lookup=speed_to_power)
    weather_data['power_output_100m'] = weather_data['wind_speed_100m'].apply(speed_to_power_rating, lookup=speed_to_power)

    # Calculate the total energy output over the time period
    total_energy_10m = weather_data['power_output_10m'].sum() * 1  # Assuming each row represents 1 hour output units are kWh
    total_energy_100m = weather_data['power_output_100m'].sum() * 1  # Assuming each row represents 1 hour output units are kWh

    # Print the total energy output
    print(f"\nTotal energy output at 10m: {total_energy_10m} kWh")
    print(f"Total energy output at 100m: {total_energy_100m} kWh")   


    outpath_prefix = f'{LOCATION.lower().replace(", ", "_")}_power_output_'

    # Plot the power output over time
    plot_power_output(weather_data, outpath=outpath_prefix+'2025.png')

    # Plot only the data for January 2025
    january_data = weather_data[weather_data['time'].dt.month == 1]
    plot_power_output(january_data, outpath=outpath_prefix+'january_2025.png', month='January')

    # Plot only the data for April 2025
    april_data = weather_data[weather_data['time'].dt.month == 4]
    plot_power_output(april_data, outpath=outpath_prefix+'april_2025.png', month='April')

    # Plot only the data for July 2025
    july_data = weather_data[weather_data['time'].dt.month == 7]
    plot_power_output(july_data, outpath=outpath_prefix+'july_2025.png', month='July')

    # Plot only the data for October 2025
    october_data = weather_data[weather_data['time'].dt.month == 10]
    plot_power_output(october_data, outpath=outpath_prefix+'october_2025.png', month='October')
