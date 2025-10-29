# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


import utils.gen_utils as gen_utils
import utils.forecasting_utils as forecast_utils

# autopep8: on


#---------------Functions for RMSE+apprx err table
def calculate_rmse_mu_sig(trap_path, output_path, dist, results_file):
    """
    Calculate RMSE mean and std for a given distribution and append results to a text file.
    """
    trap = pd.read_csv(f'{trap_path}/{dist}_traps.csv')
    trial_cols = forecast_utils.trial_names()

    trap_avg = trap[trial_cols].mean().mean()
    rmses = pd.read_csv(f'{output_path}/{dist}_rmses.csv', sep='\t')

    rmse_mu = rmses.RMSE.mean()
    rmse_sig = rmses.RMSE.std()

    approx_err_mu = rmse_mu / trap_avg
    approx_err_sig = rmse_sig / trap_avg

    # Prepare text output
    lines = [
        f'Distribution: {dist}',
        f'Avg. trap catch: {trap_avg:.4f}',
        f'RMSE mu: {rmse_mu:.4f}; RMSE sig: {rmse_sig:.4f}',
        f'Approx err mu: {approx_err_mu:.4f}; Approx err sig: {approx_err_sig:.4f}',
        '-' * 50,
        ''
    ]

    # Print to console
    for line in lines:
        print(line)

    # Append results to file
    with open(results_file, 'a') as f:
        f.write('\n'.join(lines))
    
    return

def tabular_rmses(trap_catch_path, output_path):
    results_file = os.path.join(output_path, 'rmse_summary.txt')
    if os.path.exists(results_file):
        os.remove(results_file)

    # Run for both distributions
    for dist in ['poisson', 'negbin']:
        calculate_rmse_mu_sig(trap_catch_path, output_path, dist, results_file)

    print(f'\nSummary saved to: {results_file}')
    return

#---------------Functions for Avg. RMSE + abs err plot
def rmse_plot(output_path):
    
    return


def main():
    config = '../fpaths_config.json'
    _, trap_catch_path, _, _, _ = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)

    tabular_rmses(trap_catch_path, output_path)


if __name__ == "__main__":
    main()