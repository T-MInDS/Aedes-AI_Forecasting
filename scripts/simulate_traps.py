# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.gen_utils as gen_utils
import utils.forecasting_utils as forecasting_utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import binom, nbinom

# autopep8: on

def create_trap_cols(data):
    weekly = forecasting_utils.convert_to_weekly(data)

    # Replicate MoLS into mols_run_0 ... mols_run_49
    for i in range(50):
        weekly[f'mols_run_{i}'] = weekly['MoLS']

    return weekly

def poisson_traps(df, poisson_p=0.00048):
    data = df.copy()
    trial_cols = forecasting_utils.trial_names()
    for col_name in trial_cols:
        col = data[col_name]
        for j in range(0,len(col)):
            total = data.MoLS.iloc[j]
            trials = np.random.random(size=int(total))
            catches = len(np.where(trials < poisson_p)[0])

            data.loc[j, col_name] = catches
    
    data['Ref'] = data[trial_cols].mean(axis=1)
    data['Ref_sd'] = np.sqrt(data[trial_cols].var(axis=1))

    mu, u_forecast_quant, l_forecast_quant = true_poisson_params(poisson_p, data)

    data['True_mu'] = mu
    data['True_u_forecast_quant'] = u_forecast_quant
    data['True_l_forecast_quant'] = l_forecast_quant
    
    return data

def negbin_traps(data, prop=8, negbin_p=0.5):
    seen_prop = prop/np.average(data.MoLS)
    run_cols = forecasting_utils.trial_names()
    ns = []
    for col_name in run_cols:
        col = data[col_name]
        for j in range(0, len(col)):
            total = data.MoLS.iloc[j]
            trials = np.random.random(size=int(total))
            n = seen_prop * total
            ns.append(n)

            negbin_catches = 0
            negbin_successes = 0
            for trial in trials:
                if trial <= negbin_p:
                    negbin_successes += 1
                else:
                    negbin_catches += 1

                if negbin_successes >= n:
                    break

            data.loc[j, col_name] = negbin_catches

    data['Ref'] = data[run_cols].mean(axis=1)
    data['Ref_sd'] = np.sqrt(data[run_cols].var(axis=1))

    mu, u_forecast_quant, l_forecast_quant = true_negbin_params(negbin_p, seen_prop*data.MoLS)

    data['True_mu'] = mu
    data['True_u_forecast_quant'] = u_forecast_quant
    data['True_l_forecast_quant'] = l_forecast_quant
    return data

def true_poisson_params(p, data):
    #Note this function is called true_poisson to differentiate between poisson and negbin, but it's actually calculating the parameters of the bin. distribution used to generate the data
    ns = np.floor(data.MoLS)
    mu = binom.stats(ns, p, moments='m')
    u_forecast_quant = binom.ppf(0.5 + 0.68 / 2, ns, p)
    l_forecast_quant = binom.ppf(0.5 - 0.68 / 2, ns, p)
    return mu, u_forecast_quant, l_forecast_quant

def true_negbin_params(p, seen):
    ns = np.floor(seen)

    mu = nbinom.stats(ns, p, moments='m')
    u_forecast_quant = nbinom.ppf(0.5 + 0.68 / 2, ns, p)
    l_forecast_quant = nbinom.ppf(0.5 - 0.68 / 2, ns, p)
    return mu, u_forecast_quant, l_forecast_quant


def main():

    _, trap_catch_path, _, _, raw_mols_path = gen_utils.load_input_paths(
        '../fpaths_config.json')

    observed, _ = forecasting_utils.load_weather_data(raw_mols_path)
    observed.rename(columns={'Ref': 'MoLS'}, inplace=True)
    weekly = create_trap_cols(observed)
    
    poisson = poisson_traps(weekly)
    poisson = poisson[poisson.Year>2016]
    poisson.to_csv(f'{trap_catch_path}/poisson_traps.csv', index=False)
    
    negbin = negbin_traps(weekly)
    negbin = negbin[negbin.Year>2016]
    negbin.to_csv(f'{trap_catch_path}/negbin_traps.csv', index=False)

    print(f'Traps simulations saved in {trap_catch_path}')


if __name__ == "__main__":
    main()
