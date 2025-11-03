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
import utils.plotting_utils as plt_utils

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

#---------------Functions for err plot
def err_plot(output_path):
    poisson = pd.read_csv(f'{output_path}/poisson_rmses.csv', sep='\t')
    negbin = pd.read_csv(f'{output_path}/negbin_rmses.csv', sep='\t')

    labels = [
        r'RMSE (wks. 1–4)',
        r'$|$Error$|$ (wk 1)',
        r'$|$Error$|$ (wk 2)',
        r'$|$Error$|$ (wk 3)',
        r'$|$Error$|$ (wk 4)'
    ]

    score_cols = ['RMSE', 'wk1_abs_err', 'wk2_abs_err',
                  'wk3_abs_err', 'wk4_abs_err']

    fig, axs = plt.subplots(2, figsize=(8,4.5), sharex=True, sharey=True)

    dists = {'Poisson': poisson, 'Negative Binomial': negbin}
    for ax, (dist, scores) in zip(axs, dists.items()):
        avgs = scores[score_cols].mean()
        stds = scores[score_cols].std()
        plt_utils.format_single_bar_plot(ax, avgs, stds, labels)
        #ax.annotate(f'{dist}', xy=(0, 0.5), xycoords='axes fraction', 
        #            xytext=(-50, 0), textcoords='offset points',
        #            ha='right', va='center', fontsize='large', rotation=90, 
        #            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', lw=0.8))
        ax.set_title(f'{dist} (Avg. Trap = {np.average(scores.Avg_trap):.2f})', fontsize=12)

    
    axs[1].set_xticklabels(labels, rotation=0)
    #axs[0].set_title('Forecast Point Prediction Scores', fontsize='medium')

    fig.tight_layout()
    fig.savefig(f'{output_path}/point_prediction_errors.png', dpi=300, bbox_inches='tight')        
    return


def main():
    config = '../fpaths_config.json'
    _, trap_catch_path, _, _, _ = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)

    if True:
        tabular_rmses(trap_catch_path, output_path)
    
    if True:
        err_plot(output_path)


if __name__ == "__main__":
    main()