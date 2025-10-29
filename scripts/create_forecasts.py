# autopep8: off

import os, sys, json
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

import matplotlib.pyplot as plt

from scipy.stats import nbinom, poisson

# autopep8: on

#---------------Point prediction functions
def scale_point_predictions(sample):
    observed = sample[sample.Location == 'Observed']
    scaler = scale_rto(observed)
    sample['Point_predictions'] = scaler * sample['Neural Network']
    return sample

def scale_rto(data):
    return data.describe().loc['mean']['Ref'] / data.describe().loc['mean']['Neural Network']

#---------------Negative Binomial parameters
def negbin_avg_p(observed):
    mu = observed.Ref.values.astype('float')
    sig = observed.Ref_sd.values.astype('float')
    ps = np.clip(np.divide(mu, sig**2), 0, 1)
    mean_p = np.mean(ps)

    return mean_p

def negbin_params(observed, predictions):
    mean_p = negbin_avg_p(observed)
    # var: sig^2 = mu/p
    var = np.divide(predictions, mean_p)

    # n (num successes): n=mu^2/sig^2 - mu
    ns = np.divide(predictions**2, var-predictions)
    
    return ns, mean_p


#---------------Create forecasts functions
def create_poisson_forecasts(samples):
    forecasts = []
    for sample in samples:
        sample = scale_point_predictions(sample)
        forecasts.append(sample)
    
    return forecasts

def create_negbin_forecasts(samples):
    forecasts = []
    for sample in samples:
        sample = scale_point_predictions(sample)
        sample.Datetime = pd.to_datetime(sample.Datetime)

        observed = sample[sample.Location == 'Observed']
        ns, mean_p = negbin_params(observed, sample['Point_predictions'])
        sample = sample.copy()
        sample['ns'] = ns
        sample['mean_p'] = mean_p
        forecasts.append(sample)
    
    return forecasts


def load_poissons(nn_preds_path, samples_path, trap_catch_path):
    poisson_path = '{}/poisson_samples.h5'.format(nn_preds_path)
    if not os.path.exists(poisson_path):
        samples, t0_list = forecast_utils.load_samples_hdf5(samples_path)

        poisson_traps = pd.read_csv(f'{trap_catch_path}/poisson_traps.csv')
        poissons = forecast_utils.merge_nn_traps(samples, poisson_traps)

        forecast_utils.save_samples_to_hdf5(poissons, t0_list, poisson_path)
    
    else:
        poissons, t0_list = forecast_utils.load_samples_hdf5(poisson_path)
    
    return poissons, t0_list

def load_negbins(nn_preds_path, samples_path, trap_catch_path):
    negbin_path = '{}/negbin_samples.h5'.format(nn_preds_path)
    if not os.path.exists(negbin_path):
        samples, t0_list = forecast_utils.load_samples_hdf5(samples_path)

        negbin_traps = pd.read_csv(f'{trap_catch_path}/negbin_traps.csv')
        negbins = forecast_utils.merge_nn_traps(samples, negbin_traps)

        forecast_utils.save_samples_to_hdf5(negbins, t0_list, negbin_path)
    
    else:
        negbins, t0_list = forecast_utils.load_samples_hdf5(negbin_path)
    
    return negbins, t0_list


def main():
    fpaths_config = '../fpaths_config.json'
    _, trap_catch_path, nn_preds_path, _, _ = gen_utils.load_input_paths(
        fpaths_config)
    
    _, output_path = gen_utils.load_output_paths(fpaths_config)

    samples_path = '{}/nn_mixed_samples.h5'.format(nn_preds_path)

    #Run Poisson
    if True:
        poissons, t0_list = load_poissons(nn_preds_path, samples_path, trap_catch_path)
        poisson_forecasts = create_poisson_forecasts(poissons)
        opath = f'{nn_preds_path}/poisson_forecasts.h5'
        forecast_utils.save_samples_to_hdf5(poisson_forecasts, t0_list, opath)

    #Run NegBin
    if True:
        negbin, t0_list = load_negbins(nn_preds_path, samples_path, trap_catch_path)
        negbin_forecasts = create_negbin_forecasts(negbin)
        opath = f'{nn_preds_path}/negbin_forecasts.h5'
        forecast_utils.save_samples_to_hdf5(negbin_forecasts, t0_list, opath)
    
    

if __name__ == "__main__":
    main()
