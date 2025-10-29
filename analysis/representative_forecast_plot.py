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

def monthly_indices(t0_list, year=2017):
    # Generate the first of each month
    first_of_months = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='MS')
    t0_series = pd.Series(t0_list)

    #Find closest t0 values to first_of_months
    closest_indices = []
    closest_values = []

    for target in first_of_months:
        # Compute absolute time difference
        diffs = (t0_series - target).abs()
        idx = diffs.idxmin()  # index of closest value
        closest_indices.append(idx)
        closest_values.append(t0_series[idx])
    return closest_indices, closest_values


#---------------Functions to control main plots
def high_low_skill(poisson25, poisson75, negbin25, negbin75, output_path):
    #Create the representative forecasts of high and low skill
    fig, axs = plt.subplots(2,2, figsize=(8,4.5))
    axs[0,0] = plt_utils.format_forecast_single_plot(axs[0,0], poisson25, 'poisson')
    axs[0,1] = plt_utils.format_forecast_single_plot(axs[0,1], poisson75, 'poisson')

    axs[1,0] = plt_utils.format_forecast_single_plot(axs[1,0], negbin25, 'negbin')
    axs[1,1] = plt_utils.format_forecast_single_plot(axs[1,1], negbin75, 'negbin')

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
    bottom_margin=0.08
    bbox_to_anchor=(0.52, -0.0)
    fig = plt_utils.create_legend(fig, bottom_margin=bottom_margin, bbox_to_anchor=bbox_to_anchor)
    fig.tight_layout(rect=(0, bottom_margin+0.01, 1, 1))
    fig.savefig(f'{output_path}/representative_forecasts.png', dpi=300)
    return

def forecast_examples(nn_preds_path, output_path):
    #Create forecast plots for the start of every month

    poisson_forecast_fil = f'{nn_preds_path}/poisson_forecasts.h5'
    poissons, t0_list = forecast_utils.load_samples_hdf5(poisson_forecast_fil)

    negbin_forecast_fil = f'{nn_preds_path}/negbin_forecasts.h5'
    negbins, t0_list = forecast_utils.load_samples_hdf5(negbin_forecast_fil)
    
    closest_indices, closest_values = monthly_indices(t0_list, year=2019)

    #Plot 6 months at a time
    idx_dict = {'a': closest_indices[:6], 'b': closest_indices[6:]}
    for subset, idxs in idx_dict.items():
        fig, axs = plt.subplots(6, 2, figsize=(7, 10), sharey='row')
        
        for row, idx in enumerate(idxs):
            #Add Poisson forecasts
            ax = axs[row, 0]
            sample = poissons[idx]
            plt_utils.format_forecast_single_plot(ax, sample, 'poisson')
            ax.set_ylabel('Trap Counts')
        
            #Add Neg Bin forecasts
            ax = axs[row, 1]
            sample = negbins[idx]
            plt_utils.format_forecast_single_plot(ax, sample, 'negbin')

        axs[0,0].set_title('Poisson')
        axs[0,1].set_title('Negative Binomial')

        bottom_margin=0.045
        bbox_to_anchor=(0.52, 0.01)
        fig = plt_utils.create_legend(fig, bottom_margin=bottom_margin, bbox_to_anchor=bbox_to_anchor)
        fig.tight_layout(rect=(0, bottom_margin+0.01, 1, 1))
        fig.savefig(f'{output_path}/forecast_examples_{subset}.png', dpi=300)
    
    return


    

def main():
    config = '../fpaths_config.json'
    _, _, nn_preds_path, _, _ = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)
    
    if True:
        poisson25, poisson75 = process_rmses(output_path, nn_preds_path, 'poisson')
        negbin25, negbin75 = process_rmses(output_path, nn_preds_path, 'negbin')

        high_low_skill(poisson25, poisson75, negbin25, negbin75, output_path)
    
    if True:
        forecast_examples(nn_preds_path, output_path)




if __name__ == "__main__":
    main()

