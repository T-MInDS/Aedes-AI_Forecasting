"""Create the data for each step in the forecasting process

    Starts with raw weather files from NOAA and outputs
    """

# autopep8: off

import sys, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json, glob
import matplotlib.pyplot as plt
import pandas as pd
import utils.format_data_utils as data_utils
from utils import gen_utils

# autopep8: on


def format_observed(weather_path: str):
    """Process raw NOAA weather data and generate cleaned daily summaries for San Juan.

    Reads the raw weather CSV, computes average temperature and humidity, cleans the data,
    renames columns, and saves processed data in pickle format for further analysis.

    Args:
        weather_path (str): Path to the directory containing the raw weather CSV files.

    Returns:
        pd.DataFrame: Cleaned and processed weather data for San Juan.
    """
    raw_weather = '{}/sj_asos.csv'.format(weather_path)
    data = pd.read_csv(raw_weather)
    data['Avg_Temp'] = (data['max_temp_f'] + data['min_temp_f']) / 2
    data['Humidity'] = (data['min_rh'] + data['max_rh']) / 2
    # data['Humidity'] = data['Humidity'].fillna()
    data['Humidity'] = data['Humidity'].fillna(data['avg_rh'])
    data.rename(columns={'precip_in': 'Precip_in',
                'day': 'Datetime'}, inplace=True)
    data.Datetime = pd.to_datetime(data.Datetime)

    _ = data_utils.cleanDailySummaries(data, 'San_Juan')

    data = pd.read_csv('{}/San_Juan.csv'.format(weather_path))
    data['Precip'] = 25.4 * data['Precip_in']
    data = data[['Location', 'Year', 'Month', 'Day',
                 'Avg_Temp', 'Precip', 'Humidity', 'Ref']]
    data.to_pickle('{}/San_Juan.pd'.format(weather_path))

    data['Precip'] = data['Precip']/10
    data.to_pickle('{}/San_Juan_daily.pd'.format(weather_path))

    return data


def format_forecast(weather_path: str):
    """    Process raw NOAA weather forecast data and generate cleaned daily summaries for Ceiba.

    Collects and cleans forecast data, computes average temperature and humidity, renames columns,
    and saves processed data in pickle format for further analysis.

    Args:
        weather_path (str): Path to the directory containing the raw weather forecast CSV files.

    Returns:
        pd.DataFrame: Cleaned and processed weather forecast data for Ceiba.
    """
    data_utils.collect_ceiba()
    raw_weather = '{}/DailySummaries_Ceiba.csv'.format(weather_path)
    _ = data_utils.cleanDailySummaries(pd.read_csv(raw_weather), loc='Ceiba')

    data = pd.read_csv('{}/Ceiba.csv'.format(weather_path))
    data['Precip'] = 25.4 * data['Precip_in']
    data = data[['Location', 'Year', 'Month', 'Day',
                 'Avg_Temp', 'Precip', 'Humidity', 'Ref']]
    data.to_pickle('{}/Ceiba.pd'.format(weather_path))

    data['Precip'] = data['Precip']/10
    data.to_pickle('{}/Ceiba_daily.pd'.format(weather_path))

    return data


def process_mols(raw_mols_path: str):
    """Process MoLS prediction files by applying burn-in and burn-out periods.

    Reads all MoLS prediction CSV files from the specified directory, removes the first 10 days (burn-in)
    and the last 90 days (burn-out) from each file, and saves the cleaned data back to the same files.

    Note:
        This function assumes that the MoLS prediction files are named with the pattern '*_MoLS.csv'
        and are located in the directory specified by `raw_mols_path`.
    """
    # Save the daily weather as csv files for MoLS
    data_utils.save_weather_mols()

    # Manual process: Obtain corresponding MoLS predictions for loc_daily.pd files and store them in raw_mols_path
    # Finally, 90 day burn in and burn out

    fils = glob.glob('{}/*_MoLS.csv'.format(raw_mols_path))
    for fil in fils:
        data = pd.read_csv(fil)
        data = data.iloc[10:-90].reset_index(drop=True)
        data.to_csv(fil, index=False)
    return


def main():
    weather_path, _, _, _, raw_mols_path = gen_utils.load_paths(
        '../fpaths_config.json')
    observed = format_observed(weather_path)
    forecasted = format_forecast(weather_path)
    process_mols(raw_mols_path)


if __name__ == "__main__":
    main()
