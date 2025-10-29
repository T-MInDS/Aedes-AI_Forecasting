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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines


# autopep8: on

def raw_data_plot(poisson, negbin, output_path):
    trial_cols = forecast_utils.trial_names()

    fig, axs = plt.subplots(2, figsize=(7,4), sharex=True, sharey=True)

    s = 2
    alpha = 0.65
    # First plot Poisson traps
    [axs[0].scatter(poisson.Datetime, poisson[col], s=s, color='tab:orange', alpha=0.1) for col in trial_cols]
    axs[0].plot(poisson.Datetime, poisson['True_mu'], color='tab:green', alpha=alpha, label='True means')
    axs[0].plot(poisson.Datetime, poisson['True_u_forecast_quant'], linestyle='--', color='tab:green', alpha=alpha, label='True 68% prediction intervals')#,\n{}'.format(r'$([l_{w,0.68}, u_{w,0.68}])$'))
    axs[0].plot(poisson.Datetime, poisson['True_l_forecast_quant'], linestyle='--', color='tab:green', alpha=alpha)
    
    axs[0].set_ylabel('Poisson\ntrap counts')

    alpha = 0.7
    [axs[1].scatter(negbin.Datetime, negbin[col], s=s, color='tab:orange', alpha=0.1) for col in trial_cols]
    axs[1].plot(negbin.Datetime, negbin['True_mu'], color='tab:green', alpha=alpha, label='True means')
    axs[1].plot(negbin.Datetime, negbin['True_u_forecast_quant'], linestyle='--', color='tab:green', alpha=alpha, label='True 68% prediction intervals')#,\n{}'.format(r'$([l_{w,0.68}, u_{w,0.68}])$'))
    axs[1].plot(negbin.Datetime, negbin['True_l_forecast_quant'], linestyle='--', color='tab:green', alpha=alpha)
    
    axs[1].set_ylabel('negbin\ntrap counts')

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Hide every other x tick label
    for j, label in enumerate(axs[1].get_xticklabels()):
        if j % 2 != 0:
            label.set_visible(False)

    # Create a custom legend handle for 'Trap samples' (larger and darker)
    trap_sample_handle = mlines.Line2D([], [], marker='o', color='tab:orange', linestyle='None', markerfacecolor='tab:orange',
                            markeredgewidth=0, markersize=8, label='Samples')#,\n{}'.format(r'$(c_{i,w})$'))

    # Collect handles and labels from one of the axes
    handles, labels = axs[0].get_legend_handles_labels()

    # Add the custom legend handle to the list
    handles = [trap_sample_handle] + handles
    labels = ['Samples'] + labels #,\n{}'.format(r'$(c_{i,w})$')] + labels

    # Add a single legend to the right of all subplots
    #fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.17, -0.05), fontsize='medium', ncol=3)
    axs[0].legend(handles, labels, loc='upper center', ncol=3)           

    fig.tight_layout()
    fig.savefig(f'{output_path}/simulated_traps_raw.png', dpi=300, bbox_inches='tight')
    return

def sample_statistics(poisson, negbin, output_path):
    fig, axs = plt.subplots(2, figsize=(7,4), sharex=True, sharey=True)

    axs[0].plot(poisson.Datetime, poisson.Ref, color='tab:blue', alpha=0.65, label='Sample means')#, {}'.format(r'$\mu(C_w)$'))
    axs[0].plot(poisson.Datetime, poisson.Ref_sd**2, color='tab:orange', alpha=0.65, label='Sample variances')#, {}'.format(r'$\sigma^2(C_w)$'))
    axs[0].set_ylabel('Poisson\ntrap counts')

    axs[1].plot(negbin.Datetime, negbin.Ref, color='tab:blue', alpha=0.65)
    axs[1].plot(negbin.Datetime, negbin.Ref_sd**2, color='tab:orange', alpha=0.65)
    axs[1].set_ylabel('Negative Binomial\ntrap counts')
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for j, label in enumerate(axs[1].get_xticklabels()):
        if j % 2 != 0:
            label.set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(loc='upper center', ncol=2)

    fig.tight_layout()
    fig.savefig(f'{output_path}/simulated_trap_sample_stats.png', dpi=300, bbox_inches='tight')
    return

def main():
    config = '../fpaths_config.json'
    _, trap_catch_path, nn_preds_path, _, _ = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)

    poisson = pd.read_csv(f'{trap_catch_path}/poisson_traps.csv')
    poisson['Datetime'] = pd.to_datetime(poisson['Datetime'])
    
    negbin = pd.read_csv(f'{trap_catch_path}/negbin_traps.csv')
    negbin['Datetime'] = pd.to_datetime(negbin['Datetime'])
    
    raw_data_plot(poisson, negbin, output_path)
    sample_statistics(poisson, negbin, output_path)

if __name__ == "__main__":
    main()