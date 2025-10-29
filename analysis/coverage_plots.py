# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import utils.gen_utils as gen_utils

# autopep8: on


def single_scatter(alpha_grouped, ax, cmap):
    for alpha in alpha_grouped.index:
        color = cmap(1 - alpha)
        vals = 100 * alpha_grouped[alpha_grouped.index == alpha].values[0]

        good_idxs = np.where(vals >= 100 * alpha)[0]
        bad_idxs = np.where(vals < 100 * alpha)[0]

        wks = np.arange(1, len(vals) + 1)
        ax.axhline(y=alpha * 100, xmin=0, xmax=len(vals), color=color, alpha=0.3)
        marker_size = 4
        ax.scatter(wks[good_idxs], vals[good_idxs], color=color, alpha=0.8, s=marker_size)
        ax.scatter(wks[bad_idxs], vals[bad_idxs], color=color, alpha=1, s=15)

    ax.set_yticks(100 * alpha_grouped.index.values)
    ax.set_xticks(wks)
    ax.set_ylabel('Coverages', fontsize=10)
    ax.set_xlabel('Forecast Week', fontsize=10)


def coverage_plot(poissons, negbins, outpath):
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8, 6))
    outer_gs = GridSpec(3,1, height_ratios=[1, 1, 0.08], figure=fig, hspace=0.4)  # autopep8: off
    int_vals = [str(i) for i in range(0, 101, 10)]
    int_vals[-1] = 99

    cov_cols = ['alpha', 'wk1', 'wk2', 'wk3', 'wk4']

    dist_dict = {'Poisson': poissons, 'Negative Binomial': negbins}
    for row_idx, (dist_name, coverages) in enumerate(dist_dict.items()):
        row_gs = GridSpecFromSubplotSpec(
            1, 2, width_ratios=[
                3, 2], subplot_spec=outer_gs[row_idx], wspace=0.2)

        ax = fig.add_subplot(row_gs[0])

        alpha_grouped = coverages[cov_cols].groupby('alpha').mean()
        single_scatter(alpha_grouped, ax, cmap)


    plt.show()
    asdf


def main():
    config = '../fpaths_config.json'
    _, output_path = gen_utils.load_output_paths(config)

    poissons = pd.read_csv(f'{output_path}/poisson_coverages.csv', sep='\t')
    negbins = pd.read_csv(f'{output_path}/negbin_coverages.csv', sep='\t')

    coverage_plot(poissons, negbins, output_path)

    return


if __name__ == "__main__":
    main()
