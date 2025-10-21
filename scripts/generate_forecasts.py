# autopep8: off

import os, sys, json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import utils.forecasting_utils as forecast_utils
import utils.gen_utils as gen_utils


# autopep8: on


def configure_data(raw_mols_path, nn_abundance_predictions):
    observed_weather, forecasted_weather = forecast_utils.load_weather_data(
        raw_mols_path)
    observed_weather['Datetime'] = pd.to_datetime(
        observed_weather[['Year', 'Month', 'Day']])
    forecasted_weather['Datetime'] = pd.to_datetime(
        forecasted_weather[['Year', 'Month', 'Day']])

    samples, t0_list = [], []
    for t0 in observed_weather.Datetime.iloc[0:120]:
        sample = forecast_utils.configure_sample(t0=t0, observed_weather=observed_weather,
                                                 forecasted_weather=forecasted_weather, date_col='Datetime')
        if sample is not None:
            samples.append(sample)
            t0_list.append(t0)

    print(len(samples))
    asdf

    forecast_utils.save_samples_to_hdf5(
        samples, t0_list, '{}/mixed_samples.h5'.format(nn_abundance_predictions))


def main():
    weather_path, mols_path, nn_preds_path, model_files_path, raw_mols_path = gen_utils.load_paths(
        '../fpaths_config.json')

    configure_data(raw_mols_path, nn_preds_path)


if __name__ == "__main__":
    main()
