# autopep8: off

import os, sys
from tqdm.auto import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import utils.forecasting_utils as forecast_utils
import utils.gen_utils as gen_utils
import utils.predictions as predictions

# autopep8: on

def _is_ok(v, allow_inf=True):
    if pd.isna(v):
        return True
    if np.isscalar(v) and isinstance(v, (int, float, np.integer, np.floating)):
        return True if allow_inf else np.isfinite(v)
    return False


def configure_data(raw_mols_path, samples_path, model, scaler, data_shape):
    observed_weather, forecasted_weather = forecast_utils.load_weather_data(
        raw_mols_path)
    observed_weather['Datetime'] = pd.to_datetime(
        observed_weather[['Year', 'Month', 'Day']])
    forecasted_weather['Datetime'] = pd.to_datetime(
        forecasted_weather[['Year', 'Month', 'Day']])

    observed_weather.rename(columns={'Ref': 'MoLS'}, inplace=True)
    forecasted_weather.rename(columns={'Ref': 'MoLS'}, inplace=True)
    
    # --- select only start-of-week dates (weeks ending Saturday) ---
    observed_weather['Week'] = observed_weather['Datetime'].dt.to_period('W-WED')
    # get first date of each weekly period (start of that week)
    week_starts = (
        observed_weather.groupby('Week')['Datetime']
        .min()
        .sort_values()
        .reset_index(drop=True)
    )
    
    # Drop to avoid Period dtype leaking into downstream scalers
    observed_weather.drop(columns=['Week'], inplace=True)

    samples, t0_list = [], []
    for t0 in tqdm(week_starts, desc="Building samples", unit="week"):
        sample = forecast_utils.configure_sample(
            t0=t0,
            observed_weather=observed_weather,
            forecasted_weather=forecasted_weather,
            date_col='Datetime')
        if sample is not None:
            temp = sample.drop(
                columns=['Datetime'])
            
            if 'Ref' in temp.columns:
                temp = temp.rename(columns={'Ref': 'MoLS'})
            temp['Location'] = 'San_Juan'
    
            results = create_nn_predictions(temp,
                                            model,
                                            scaler,
                                            data_shape)
            results['Datetime'] = pd.to_datetime(results[['Year', 'Month', 'Day']])
            
            if 'MoLS' in results.columns:
                results.drop(columns=['MoLS'], inplace=True)
            
            merged = pd.merge(sample, results[['Datetime', 'Neural Network']], on=[
                              'Datetime'], how='outer')
            
            merged = merged[merged['Neural Network'].notna()].reset_index()
            
            observed = forecast_utils.convert_to_weekly(merged[merged.Location == 'San_Juan'])
            forecast = forecast_utils.convert_to_weekly(merged[merged.Location == 'Ceiba'])

            weekly = pd.concat([forecast, observed]).sort_values(by='Datetime').reset_index(drop=True)
            weekly.loc[weekly['Location'] == 'San_Juan', 'Location'] = 'Observed'
            weekly.loc[weekly['Location'] == 'Ceiba', 'Location'] = 'Forecast'

            #Ensure consistent sample length
            weekly = weekly.iloc[0:17]
                        
            bad_mask = ~weekly['Neural Network'].map(_is_ok)
            if int(bad_mask.sum()) > 0:
                print(weekly)
                raise TypeError('Invalid nn values')         
            
            samples.append(weekly)
            t0_list.append(t0)

    forecast_utils.save_samples_to_hdf5(
        samples, t0_list, samples_path)


def create_nn_predictions(sample, model, scaler, data_shape):
    results = predictions.gen_preds(model, sample, data_shape, scaler, fit_scaler=False)
    results = pd.DataFrame(
        results,
        columns=['Location', 'Year', 'Month', 'Day', 'MoLS', 'Neural Network']
    )
    return results


def main():
    _, _, nn_preds_path, model_files_path, raw_mols_path = gen_utils.load_input_paths(
        '../fpaths_config.json')

    # Load model files
    model, config, scaler = gen_utils.load_finetune_model_fils(
        model_files_path, 'finetune_config.json')
    data_shape = config['data']['data_shape']

    samples_path = '{}/nn_mixed_samples.h5'.format(nn_preds_path)

    if not os.path.exists(samples_path):
        configure_data(raw_mols_path, samples_path, model, scaler, data_shape)


if __name__ == "__main__":
    main()
