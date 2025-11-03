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

# ---------------Weekly plot functions
def single_small(ax_small, alphas, coverages_for_week, cmap):
    scatter_colors = np.array([cmap(1 - alpha) for alpha in alphas])
    alphas_arr = np.array(alphas)

    nominal = np.array(alphas)*100
    coverages = np.array(coverages_for_week)

    good_mask = coverages >= nominal
    broken_mask = ~good_mask

    if np.any(good_mask):
        ax_small.scatter(
            alphas_arr[good_mask],
            coverages[good_mask],
            c=scatter_colors[good_mask, :],
            s=15,                   # moderate size
            marker='o',
            alpha=0.9,
            zorder=3,
        )

    if np.any(broken_mask):
        ax_small.scatter(
            alphas_arr[broken_mask],
            coverages[broken_mask],
            facecolors='none',      # hollow marker for clear visual contrast
            edgecolors=scatter_colors[broken_mask, :],
            s=20,                   # slightly larger but not too large
            marker='v',
            linewidth=1.2,
            zorder=4,
        )
    ax_small.axline((0, 0), slope=100, linestyle='--', color='gray', alpha=0.8)

    ax_small.set_xlim([-0.05, 1.05])
    ax_small.set_ylim([-5, 105])

    ax_small.set_xticks(np.arange(0, 1.1, step=0.1))
    ax_small.set_yticks(np.arange(0, 101, step=10))

    return


def small_plots(fig, small_gs, alpha_grouped, cmap):
    int_vals = [str(i) for i in range(0, 101, 10)]
    int_vals[-1] = 99

    axs_block = []
    alphas = sorted(alpha_grouped.index.values)


    for idx in np.arange(alpha_grouped.shape[-1]):
        i, j = divmod(idx, 2)
        ax_small = fig.add_subplot(small_gs[i, j])
        coverages_for_week = [100 * alpha_grouped.loc[alpha][idx] for alpha in alphas]
        print(i, j, coverages_for_week)

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
        axs_block.append(ax_small)

    return axs_block


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
    cb.ax.set_xlabel(r'$100\cdot(1-\alpha)$', fontsize=10)

# ---------------Main plot functions


def coverage_plot(poissons, negbins, output_path):
    cmap = plt.get_cmap('viridis')


    fig = plt.figure(figsize=(9, 4))
    height_ratios = [1, 0.05, 0.08]
    outer_gs = GridSpec(
        nrows=len(height_ratios),
        ncols=1,
        height_ratios=height_ratios,
        figure=fig
    )
    cov_cols = ['alpha', 'wk1', 'wk2', 'wk3', 'wk4']

    dist_dict = {'Poisson': poissons, 'Negative Binomial': negbins}
    
    dist_gs = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer_gs[0],
        wspace=0.25
    )
    for i, (dist_name, coverages) in enumerate(dist_dict.items()):
        alpha_grouped = coverages[cov_cols].groupby('alpha').mean()

        # Add 2x2 subplot grid to the right of the main plots
                # 2x2 sub-block for weeks
        small_gs = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=dist_gs[0, i],
            wspace=0.1,
            hspace=0.3
        )
        print(dist_name)
        axs_block = small_plots(fig, small_gs, alpha_grouped, cmap)    
        
        # Add a block label ("Poisson", etc.) above Week 1 panel
        # axs_block[0] = Week 1 (row 0, col 0)
        # axs_block[1] = Week 2 (row 0, col 1)
        ax_left  = axs_block[0]
        ax_right = axs_block[1]

        pos_left = ax_left.get_position(fig)
        pos_right = ax_right.get_position(fig)

        # horizontal center between left edge of left subplot and right edge of right subplot
        x_center = 0.5 * (pos_left.x0 + pos_right.x1)

        # vertical position just above the top row
        y_top = max(pos_left.y1, pos_right.y1)
        y_text = y_top + 0.05  # bump up a little

        fig.text(
            x_center,
            y_text,
            dist_name,
            ha='center',
            va='bottom',
            fontsize=10
        )
        
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
