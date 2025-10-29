# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import csv

import pandas as pd
import numpy as np
import utils.forecasting_utils as forecast_utils
import utils.gen_utils as gen_utils
import utils.analysis_utils as analysis_utils


# autopep8: on

#---------------Quantile functions
def calculate_coverage(l_quant, u_quant, forecast):
    trial_cols = forecast_utils.trial_names()
    coverages = []
    for i in range(0, len(forecast)):
        rto = ((forecast[trial_cols].iloc[i] >= l_quant[i]) &
               (forecast[trial_cols].iloc[i] <= u_quant[i])).sum() / len(trial_cols)
        coverages.append(rto)
    return coverages

#---------------Saving results functions
def add_result_line(coverages, prefix, result_fil):
    with open(result_fil, 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(prefix + coverages)
    return

def prepare_result_fil(dist, opath):
    result_fil = f'{opath}/{dist}_coverages.csv'
    header = ['dist', 't_0', 'alpha', 'wk1', 'wk2', 'wk3', 'wk4']
    with open(result_fil, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
    return result_fil

#---------------Processing samples function
def process_samples(samples, t0_list, dist, output_path):
    result_fil = prepare_result_fil(dist, output_path)

    for sample, t0 in zip(samples, t0_list):
        forecast = sample[sample.Location == 'Forecast']
        t0_str = pd.to_datetime(t0).strftime("%Y-%m-%d")
        for ci in np.arange(0, 1.1, step=0.1):
            prefix = [dist, t0_str, ci]
            if dist == 'poisson':
                l_quant, u_quant = analysis_utils.poisson_quant(forecast['Point_predictions'], ci)
            if dist == 'negbin':
                l_quant, u_quant = analysis_utils.negbin_quant(forecast['ns'], forecast['mean_p'], ci)
            coverages = calculate_coverage(l_quant, u_quant, forecast)
            add_result_line(coverages, prefix, result_fil)
    print(f'{dist} coverages saved in {result_fil}')
    return


def main():
    fpaths_config = '../fpaths_config.json'
    _, _, nn_preds_path, _, _ = gen_utils.load_input_paths(
        fpaths_config)
    
    _, output_path = gen_utils.load_output_paths(fpaths_config)

    #Poisson
    if True:
        poisson_forecast_fil = f'{nn_preds_path}/poisson_forecasts.h5'
        poissons, t0_list = forecast_utils.load_samples_hdf5(poisson_forecast_fil)
        process_samples(poissons, t0_list, 'poisson', output_path)

    #Neg Bin
    if True:
        negbin_forecast_fil = f'{nn_preds_path}/negbin_forecasts.h5'
        negbins, t0_list = forecast_utils.load_samples_hdf5(negbin_forecast_fil)
        process_samples(negbins, t0_list, 'negbin', output_path)
    

if __name__ == "__main__":
    main()
