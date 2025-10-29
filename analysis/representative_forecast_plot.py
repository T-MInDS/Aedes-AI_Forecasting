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


#---------------Processing samples and find indices functions
def find_skill_t0s(rmses):
    #Reminder p25 indicates higher skill RMSE and p75 indicates lower skill
    p25 = np.percentile(rmses.RMSE, 25)
    p75 = np.percentile(rmses.RMSE, 75)
    
    #Find index of the closest values to those percentiles
    i25 = np.argmin(np.abs(rmses.RMSE - p25))
    i75 = np.argmin(np.abs(rmses.RMSE - p75))

    return i25, i75

def process_rmses(output_path, nn_preds_path, dist):
    rmses = pd.read_csv(f"{output_path}/{dist}_rmses.csv", sep="\t", parse_dates=["t_0"])
    i25, i75 = find_skill_t0s(rmses)

    samples_fil = f'{nn_preds_path}/{dist}_forecasts.h5'
    samples, t0_list = forecast_utils.load_samples_hdf5(samples_fil)

    return samples[i25], samples[i75]


#---------------Functions to control main plots
def high_low_skill(poisson25, poisson75, negbin25, negbin75, output_path):
    fig, axs = plt.subplots(2,2, figsize=(8,4.5))
    axs[0,0] = plt_utils.format_single_plot(axs[0,0], poisson25, 'poisson')
    axs[0,1] = plt_utils.format_single_plot(axs[0,1], poisson75, 'poisson')

    axs[1,0] = plt_utils.format_single_plot(axs[1,0], negbin25, 'negbin')
    axs[1,1] = plt_utils.format_single_plot(axs[1,1], negbin75, 'negbin')

    axs[0,0].set_ylabel('Trap Counts')
    axs[1,0].set_ylabel('Trap Counts')

    axs[0,0].set_title('(Higher skill)', fontsize='medium')
    axs[0,1].set_title('(Lower skill)', fontsize='medium')

    axs[0,0].annotate('{}'.format('Poisson'), xy=(0, 0.5), xycoords='axes fraction',
        xytext=(-50, 0), textcoords='offset points',
        ha='right', va='center', fontsize='large', rotation=90, bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', lw=0.8))
    
    axs[1,0].annotate('{}'.format('Negative Binomial'), xy=(0, 0.5), xycoords='axes fraction',
        xytext=(-50, 0), textcoords='offset points',
        ha='right', va='center', fontsize='large', rotation=90, bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', lw=0.8))

    #Add custom legend
    fig = plt_utils.create_legend(fig)
    
    fig.tight_layout()

    fig.savefig(f'{output_path}/representative_forecasts.png', bbox_inches='tight', dpi=300)
    return

def 

def main():
    config = '../fpaths_config.json'
    _, _, nn_preds_path, _, _ = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)
    
    poisson25, poisson75 = process_rmses(output_path, nn_preds_path, 'poisson')
    negbin25, negbin75 = process_rmses(output_path, nn_preds_path, 'negbin')

    high_low_skill(poisson25, poisson75, negbin25, negbin75, output_path)





if __name__ == "__main__":
    main()

