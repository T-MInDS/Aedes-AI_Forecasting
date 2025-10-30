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
import matplotlib as mpl

import utils.gen_utils as gen_utils

# autopep8: on

# ---------------Combined scatterplot functions


def single_scatter(alpha_grouped, ax, cmap):
    for alpha in alpha_grouped.index:
        color = cmap(1 - alpha)
        vals = 100 * alpha_grouped[alpha_grouped.index == alpha].values[0]

        good_idxs = np.where(vals >= 100 * alpha)[0]
        bad_idxs = np.where(vals < 100 * alpha)[0]

        wks = np.arange(1, len(vals) + 1)
        ax.axhline(y=alpha * 100, xmin=0, xmax=len(vals), color=color, alpha=0.3)
        marker_size_good = 10
        marker_size_bad = 30
        ax.scatter(wks[good_idxs], vals[good_idxs], color=color, alpha=0.8, s=marker_size_good)
        ax.scatter(wks[bad_idxs], vals[bad_idxs], color=color, alpha=1, s=marker_size_bad)

    yticks = 100 * alpha_grouped.index.values
    yticks[-1] = 99
    ax.set_yticks(yticks)
    ax.set_xticks(wks, ['wk 1', 'wk 2', 'wk 3', 'wk 4'])
    ax.set_ylabel('Coverages', fontsize=10)
    ax.set_xlabel('Forecast Week', fontsize=10)
    for k, label in enumerate(ax.yaxis.get_ticklabels()):
        if k % 2 != 0:
            label.set_visible(False)
    return


# ---------------Weekly plot functions
def single_small(ax_small, alphas, coverages_for_week, cmap):
    scatter_colors = [cmap(1 - alpha) for alpha in alphas]
    sizes = [10 if c >= a * 100 else 40 for a, c in zip(alphas, coverages_for_week)]

    ax_small.scatter(alphas, coverages_for_week, c=scatter_colors, s=sizes)
    ax_small.axline((0, 0), slope=100, linestyle='--', color='gray', alpha=0.8)

    ax_small.set_xlim([-0.05, 1.05])
    ax_small.set_ylim([-5, 105])

    ax_small.set_xticks(np.arange(0, 1.1, step=0.1))
    ax_small.set_yticks(np.arange(0, 101, step=10))

    return


def small_plots(fig, small_gs, alpha_grouped, cmap):
    int_vals = [str(i) for i in range(0, 101, 10)]
    int_vals[-1] = 99
    for idx in np.arange(alpha_grouped.shape[-1]):
        i, j = divmod(idx, 2)
        ax_small = fig.add_subplot(small_gs[i, j])
        alphas = sorted(alpha_grouped.index.values)
        coverages_for_week = [100 * alpha_grouped.loc[alpha][idx] for alpha in alphas]

        single_small(ax_small, alphas, coverages_for_week, cmap)

        ax_small.set_title(f"Week {idx+1}", fontsize=10, pad=2)
        if i != 1:
            ax_small.set_xticklabels([])
        else:
            ax_small.set_xticklabels(int_vals)
            # ax_small.set_xlabel(r'$100\cdot(1-\alpha)$', fontsize=10)
            ax_small.set_xlabel('Prediction interval', fontsize=10)
            for k, label in enumerate(ax_small.xaxis.get_ticklabels()):
                if k % 2 != 0:
                    label.set_visible(False)

        if j != 0:
            ax_small.set_yticklabels([])
        else:
            ax_small.set_yticklabels(int_vals, fontsize=10)
            ax_small.set_ylabel('Coverages', fontsize=10)
            for k, label in enumerate(ax_small.yaxis.get_ticklabels()):
                if k % 2 != 0:
                    label.set_visible(False)

    return


def colorbar(cb_ax):
    cmap = plt.get_cmap('viridis_r')

    # Horizontal colorbar at the bottom
    bounds = np.linspace(0, 110, 12)  # 11 bins: 0-10, 10-20, ..., 100-110
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Compute midpoints for centered labels
    tick_locs = 0.5 * (bounds[:-1] + bounds[1:])  # Midpoints of each bin
    tick_labels = [str(int(t)) for t in np.linspace(0, 100, 11)]  # Label from 0 to 100
    tick_labels[-1] = 99

    cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap, norm=norm,
                                   ticks=tick_locs, boundaries=bounds,
                                   orientation='horizontal')
    cb.ax.set_xticklabels(tick_labels)
    cb.ax.set_xlabel(r'$100\cdot(1-\alpha)$', fontsize=12)

# ---------------Main plot functions


def coverage_plot(poissons, negbins, output_path):
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(9, 9))
    height_ratios = [1, 0.2, 1, 0.05, 0.08]
    outer_gs = GridSpec(len(height_ratios), 1, height_ratios=height_ratios, figure=fig)

    cov_cols = ['alpha', 'wk1', 'wk2', 'wk3', 'wk4']

    dist_dict = {'Poisson': poissons, 'Negative Binomial': negbins}
    row_indices = [0, 2]
    for row_idx, (dist_name, coverages) in zip(row_indices, dist_dict.items()):
        row_gs = GridSpecFromSubplotSpec(
            1, 2, width_ratios=[
                1.5, 3], subplot_spec=outer_gs[row_idx], wspace=0.25)

        ax = fig.add_subplot(row_gs[0])

        alpha_grouped = coverages[cov_cols].groupby('alpha').mean()
        single_scatter(alpha_grouped, ax, cmap)
        ax.annotate('{}'.format(dist_name), xy=(0, 0.5), xycoords='axes fraction',
                    xytext=(-55, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=12, rotation=90,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', lw=0.8))

        # Add 2x2 subplot grid to the right of the main plots
        small_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=row_gs[1], wspace=0.1, hspace=0.3)
        small_plots(fig, small_gs, alpha_grouped, cmap)

    cb_ax = fig.add_subplot(outer_gs[-1])
    colorbar(cb_ax)

    fig.savefig(f'{output_path}/coverages.png', dpi=300, bbox_inches='tight')
    return


def main():
    config = '../fpaths_config.json'
    _, output_path = gen_utils.load_output_paths(config)

    poissons = pd.read_csv(f'{output_path}/poisson_coverages.csv', sep='\t')
    negbins = pd.read_csv(f'{output_path}/negbin_coverages.csv', sep='\t')

    coverage_plot(poissons, negbins, output_path)

    return


if __name__ == "__main__":
    main()
