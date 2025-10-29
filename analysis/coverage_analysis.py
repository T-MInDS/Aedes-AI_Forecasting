# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import utils.forecasting_utils as forecast_utils
import utils.gen_utils as gen_utils

from scipy.stats import nbinom, poisson

# autopep8: on

#---------------Quantile functions
def poisson_quant(predictions, ci):
    u_quant = poisson.ppf(0.5 + ci / 2, mu=predictions)
    l_quant = poisson.ppf(0.5 - ci / 2, mu=predictions)
    return l_quant, u_quant

def negbin_quant(ns, mean_p, ci):
    u_quant = nbinom.ppf(0.5 + ci / 2, n=ns, p=mean_p)
    l_quant = nbinom.ppf(0.5 - ci / 2, n=ns, p=mean_p)
    return l_quant, u_quant

def calculate_coverage(l_quant, u_quant, forecast):
    trial_cols = forecast_utils.trial_names()
    coverages = []
    for i in range(0, len(forecast)):
        rto = ((forecast[trial_cols].iloc[i] >= l_quant[i]) &
               (forecast[trial_cols].iloc[i] <= u_quant[i])).sum() / len(trial_cols)
        coverages.append(rto)
    return coverages

#---------------Saving results functions
def add_result_line(coverages, result, result_fil):
    for cov in coverages:
        result += f'{cov}\t'
    result += '\n'
    with open(result_fil, 'a') as f:
        f.write(result)
        f.close()
    return

def prepare_result_fil(dist, opath):
    result_fil = f'{opath}/{dist}_coverages.csv'
    header = 't_0\talpha\twk1\twk2\twk3\twk4'
    header += '\n'
    with open(result_fil, 'w') as f:
        f.write(header)
        f.close()
    return result_fil


#---------------Processing samples functions
def process_poisson_samples(samples, t0_list, output_path):
    result_fil = prepare_result_fil('poisson', output_path)

    for sample, t0 in zip(samples, t0_list):
        forecast = sample[sample.Location == 'Forecast']
        for ci in np.arange(0, 1.1, step=0.1):
            result = f"{t0}\t{ci}\t"
            l_quant, u_quant = poisson_quant(forecast['Point_predictions'], ci)
            coverages = calculate_coverage(l_quant, u_quant, forecast)
            add_result_line(coverages, result, result_fil)
    print(f'Poisson coverages saved in {result_fil}')
    return

def process_negbin_samples(samples, t0_list, output_path):
    result_fil = prepare_result_fil('negbin', output_path)

    for sample, t0 in zip(samples, t0_list):
        forecast = sample[sample.Location == 'Forecast']
        for ci in np.arange(0, 1.1, step=0.1):
            result = f"{t0}\t{ci}\t"
            l_quant, u_quant = negbin_quant(forecast['ns'], forecast['mean_p'], ci)
            coverages = calculate_coverage(l_quant, u_quant, forecast)
            add_result_line(coverages, result, result_fil)
    print(f'Neg. Bin. coverages saved in {result_fil}')
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
        process_poisson_samples(poissons, t0_list, output_path)

    #Neg Bin
    if True:
        negbin_forecast_fil = f'{nn_preds_path}/negbin_forecasts.h5'
        negbins, t0_list = forecast_utils.load_samples_hdf5(negbin_forecast_fil)
        process_negbin_samples(negbins, t0_list, output_path)
    

if __name__ == "__main__":
    main()
