# autopep8: off

import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import utils.gen_utils as gen_utils
import utils.forecasting_utils as forecast_utils

size = 7
plt.rcParams.update({
    'font.size': size,
    'axes.titlesize': size,
    'axes.labelsize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'legend.fontsize': 6,
})

# autopep8: on

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple

# ---------------Custom legend
class HandlerLineOverWideBand(HandlerTuple):
    """
    Legend handler that draws a centered, thinner translucent band
    with a line across its middle.
    """
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        line, band = orig_handle

        # make band slightly thinner (height factor < 1.5)
        band_height = 1.0 * height

        # center vertically with text
        band_y = ydescent + (height - band_height) / 2 - 0.2 * height

        # shorten the handle width a bit for better balance
        band_width = 0.8 * width
        x_start = xdescent + 0.1 * width  # center horizontally

        # rectangle (uncertainty band)
        band_artist = Rectangle(
            (x_start, band_y), band_width, band_height,
            facecolor=band.get_facecolor(),
            alpha=band.get_alpha(),
            edgecolor='none',
            transform=trans,
        )

        # line across the center
        line_y = band_y + band_height / 2
        line_artist = Line2D(
            [x_start, x_start + band_width],
            [line_y, line_y],
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            transform=trans,
        )

        return [band_artist, line_artist]
    
class HandlerDotOverWideBand(HandlerTuple):
    """
    Legend handler for trap counts: light band + centered dot.
    """
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        dot, band = orig_handle

        # match proportions to the blue one for consistency
        band_height = 1.1 * height
        band_y = ydescent + (height - band_height) / 2 - 0.2 * height

        band_width = 1.0 * width
        x_start = xdescent + 0 * width

        # pale orange spread band
        band_artist = Rectangle(
            (x_start, band_y), band_width, band_height,
            facecolor=band.get_facecolor(),
            alpha=band.get_alpha(),
            edgecolor='none',
            transform=trans,
        )


        # centered smaller dot
        dot_x = x_start + band_width / 2.0
        dot_y = band_y + band_height / 2.0
        dot_artist = Line2D(
            [dot_x], [dot_y],
            marker=dot.get_marker(),
            markersize=dot.get_markersize() * 0.8,  # 20% smaller
            markerfacecolor=dot.get_markerfacecolor(),
            markeredgecolor=dot.get_markeredgecolor(),
            markeredgewidth=dot.get_markeredgewidth(),
            linestyle='None',
            transform=trans,
        )

        return [band_artist, dot_artist]


# ---------------Subplot creators
def weather_plot(ax, weather):
    # Panel (a): weather
    ax.plot(weather.Datetime, weather.Avg_Temp,
            color='tab:orange', label='T', zorder=1)
    ax.plot(weather.Datetime, weather.Precip,
            color='tab:green', label='P', zorder=2)
    ax.plot(weather.Datetime, weather.Humidity,
            color='tab:purple', label='RH', zorder=3)
    ax.set_ylim([-10, 110])

    # red boxes for observed windows
    rec_start = weather.Datetime.iloc[0]
    rec_end = weather.Datetime.iloc[89]
    rect = Rectangle((rec_start, -5), rec_end - rec_start, 100,
                     facecolor='none', edgecolor='tab:red', zorder=4)
    ax.add_patch(rect)

    rec_start = weather.Datetime.iloc[104]
    rec_end = weather.Datetime.iloc[194]
    rect = Rectangle((rec_start, -5), rec_end - rec_start, 100,
                     facecolor='none', edgecolor='tab:red', zorder=5)
    ax.add_patch(rect)


def abundance_curve(ax, prediction):
    # Panel (b): unscaled abundance (Aedes-AI curve)
    ax.plot(
        prediction.Datetime,
        prediction['Neural Network'],
        color='tab:blue',
        label=r'$Aedes$-$AI$ curve, $\mathbf{Y}$',
        zorder=1
    )

    # red dots for callouts
    ax.scatter(
        prediction.Datetime.iloc[0],
        prediction['Neural Network'].iloc[0],
        color='tab:red', s=40, zorder=2, clip_on=False
    )
    ax.scatter(
        prediction.Datetime.iloc[-2],
        prediction['Neural Network'].iloc[-2],
        color='tab:red', s=40, zorder=3, clip_on=False
    )

    ax.set_ylim([0, 19900])


def point_preds(ax, forecast):
    # Panel (c): observed trap counts, model point preds

    # first ~13 weeks of observed data
    dt = forecast.Datetime.iloc[0:13]
    ref = forecast.Ref.iloc[0:13]
    std = forecast.Ref_sd.iloc[0:13]

    # blue model line (point predictions)
    ax.plot(
        forecast.Datetime,
        forecast.Point_predictions,
        color='tab:blue',
        label=r'Point predictions, $\hat{\mu}(\mathbf{X})$'
    )

    # orange observed trap means
    ax.scatter(
        dt,
        ref,
        color='tab:orange',
        label=r'Trap counts, $\hat{\mu}(\mathbf{C})$ and $\hat{\sigma}^2(\mathbf{C})$'
    )

    # orange observed variability band
    ax.fill_between(
        dt,
        ref + std,
        ref - std,
        color='tab:orange',
        alpha=0.15
    )

    ax.set_ylim([0, 12])

    # highlight observed window in red box
    rec_start = forecast.Datetime.iloc[0]
    rec_end = forecast.Datetime.iloc[12]
    rect = Rectangle(
        (rec_start, 1),         # (x,y)
        rec_end - rec_start,    # width = timedelta
        11,                     # height
        facecolor='none',
        edgecolor='tab:red',
        zorder=5
    )
    ax.add_patch(rect)

    # y ticks as ints, no offset
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.yaxis.set_major_formatter('{:.0f}'.format)


def forecasts(ax, forecast):
    # Panel (d): forecasts with predictive uncertainty

    # observed portion
    dt = forecast.Datetime.iloc[0:13]
    ref = forecast.Ref.iloc[0:13]
    std = forecast.Ref_sd.iloc[0:13]

    # forecast horizon portion (last ~5 weeks)
    pred_dt = forecast.Datetime.iloc[-5:]
    pred_ref = forecast.Point_predictions.iloc[-5:]
    pred_std = np.sqrt(pred_ref)

    # blue model line
    ax.plot(
        forecast.Datetime,
        forecast.Point_predictions,
        color='tab:blue',
    )

    # blue forecast uncertainty band (future only)
    ax.fill_between(
        pred_dt,
        pred_ref + pred_std,
        pred_ref - pred_std,
        color='tab:blue',
        alpha=0.15
    )

    # orange observed trap means
    ax.scatter(
        dt,
        ref,
        color='tab:orange',
    )

    # orange observed variability band (past only)
    ax.fill_between(
        dt,
        ref + std,
        ref - std,
        color='tab:orange',
        alpha=0.15
    )

    ax.set_ylim([0, 12])


# ---------------Formatting helpers
def format_plots(fig, axs):
    # Legends per panel, spine cleanup, and hiding x-ticks for top 3 panels

    for i, ax in enumerate(axs):
        # panel (a): weather legend (T, P, RH)
        if i == 0:
            ax.legend(
                loc='lower left',
                handlelength=0.75,
                handletextpad=0.4,
                handleheight=1.2,
                markerscale=0.8,
                ncol=3,
                columnspacing=1.1
            )

        # panel (b): Aedes-AI curve legend
        elif i == 1:
            # panel (b): Aedes-AI curve legend with thicker line in legend only
            line_handle = Line2D(
                            [], [],
                            color='tab:blue',
                            linewidth=2,   # legend line thickness only
                            linestyle='-',
                        )
            ax.legend(
                [line_handle],
                [r'$Aedes$-$AI$ curve, $\mathbf{Y}$'],
                loc='lower left',
                handlelength=1.2,
                handletextpad=0.6,
                handleheight=1.2,
                markerscale=0.8,
                frameon=True,
                fancybox=True,
                edgecolor='0.7',
            )

        # panels (c): model vs trap means
        elif i == 2:
            # panel (c): legend
            #   Row 1: Point predictions -> blue line over a transparent band
            #   Row 2: Trap counts -> orange dot over orange band

            # blue line
            point_line = Line2D(
                [], [],
                color='tab:blue',
                linewidth=2,
                linestyle='-',
            )
            # "band" for point predictions: fully transparent so it doesn't look like uncertainty
            point_band = Rectangle(
                (0, 0), 1, 1,
                facecolor='tab:blue',
                alpha=0.0,       # invisible band, just for consistent legend geometry
                edgecolor='none',
            )
            point_handle = (point_line, point_band)

            # orange band + dot for trap counts
            trap_dot = Line2D(
                [], [], marker='o', linestyle='None',
                markerfacecolor='tab:orange',
                markeredgecolor='tab:orange',
                markeredgewidth=0,
                markersize=6,
            )
            trap_band = Rectangle(
                (0, 0), 1, 1,
                facecolor='tab:orange',
                alpha=0.15,
                edgecolor='none',
            )
            trap_handle = (trap_dot, trap_band)

            handles = [
                point_handle,
                trap_handle,
            ]
            labels = [
                r'Point predictions, $\hat{\mu}(\mathbf{X})$',
                r'Trap counts, $\hat{\mu}(\mathbf{C})$ and $\hat{\sigma}^2(\mathbf{C})$',
            ]

            ax.legend(
                handles,
                labels,
                handler_map={
                    point_handle: HandlerLineOverWideBand(),  # draw line centered in band
                    trap_handle: HandlerDotOverWideBand(),    # draw dot centered in band
                },
                loc='lower left',
                handlelength=1.4,    # match panel (d)
                handletextpad=0.6,
                handleheight=1.2,
                markerscale=0.8,
                frameon=True,
                fancybox=True,
                edgecolor='0.7',
            )
        

        elif i == 3:
            # (1) blue line + blue band combo
            fcst_line = Line2D([], [], color='tab:blue', linewidth=2)
            fcst_band = Rectangle(
                (0, 0), 1, 1,
                facecolor='tab:blue',
                alpha=0.15,          # match fill_between alpha
                edgecolor='none'
            )
            fcst_handle = (fcst_line, fcst_band)

            # (2) orange dots for trap means
            trap_dot = Line2D(
                [], [], marker='o', linestyle='None',
                markerfacecolor='tab:orange',
                markeredgecolor='tab:orange',
                markeredgewidth=0,
                markersize=6,
            )
            trap_band = Rectangle(
                (0, 0), 1, 1,
                facecolor='tab:orange',
                alpha=0.15,          # match fill_between alpha
                edgecolor='none'
            )
            trap_handle = (trap_dot, trap_band)

            handles = [
                fcst_handle,
                trap_handle,
            ]
            labels = [
                r'Probabilistic forecasts, $\hat{\mu}(\mathbf{X})$ and $\hat{\sigma}^2(\mathbf{X})$',
                r'Trap counts, $\hat{\mu}(\mathbf{C})$ and $\hat{\sigma}^2(\mathbf{C})$'             
                ]

            axs[3].legend(
                handles,
                labels,
                handler_map={
                    fcst_handle: HandlerLineOverWideBand(),
                    trap_handle: HandlerDotOverWideBand(),
                },
                loc='lower left',
                handlelength=1.4,    # was 1.8
                handletextpad=0.6,
                handleheight=1.2,
                markerscale=0.8,
                frameon=True,
                fancybox=True,
                edgecolor='0.7',
            )

        # style: remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # y ticks as ints, no offset
        ax.yaxis.get_major_formatter()
        ax.yaxis.set_major_formatter('{:.0f}'.format)

    # hide x ticks for top three panels so only bottom has dates
    for ax in axs[0:3]:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0)

    ax = axs[-1]
    # x-axis ticks: 1st of each month, nice format
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.tick_params(axis='x', rotation=0)




def add_line(fig, axs, t0):
    """
    Draw one vertical dashed line at t0 that spans all panels.
    """
    # finalize layout so transforms are valid
    fig.canvas.draw()

    # convert x position (t0) from data coords in bottom axis to figure coords
    x_data = mdates.date2num(t0)
    x_disp = axs[-1].transData.transform((x_data, 0))[0]
    x_fig = fig.transFigure.inverted().transform((x_disp, 0))[0]

    # add a single vertical line across the figure
    line = plt.Line2D(
        (x_fig, x_fig),    # x coords in figure fraction
        (0.03, 0.88),      # y coords in figure fraction
        transform=fig.transFigure,
        color='k',
        linestyle='--',
        alpha=0.7,
        linewidth=1
    )
    fig.lines.append(line)


# ---------------Main plotting function
def main_plot(forecast, prediction, weather, output_path):
    forecast.Datetime = pd.to_datetime(forecast.Datetime)
    prediction.Datetime = pd.to_datetime(prediction.Datetime)

    fig, axs = plt.subplots(
        4,
        figsize=(6, 4),
        sharex=True,
        gridspec_kw={
            'height_ratios': [0.75, 0.75, 1.25, 1.25],
            'hspace': 0.05
        }
    )

    # build each panel
    weather_plot(axs[0], weather)
    abundance_curve(axs[1], prediction)
    point_preds(axs[2], forecast)
    forecasts(axs[3], forecast)

    # legends, cleanup, ticks
    format_plots(fig, axs)

    # vertical dashed "forecasting week t0" line
    t0 = forecast.Datetime.iloc[12]
    add_line(fig, axs, t0)

    plt.subplots_adjust(bottom=0.12)
    fig.savefig(f'{output_path}/process.png', dpi=300, bbox_inches='tight')


def text_plot(output_path):
    """
    Generates the text overlay figure (your callout labels a/b/c/d etc.).
    """
    fig, axs = plt.subplots(figsize=(6, 6))
    axs.set_axis_off()

    fig.text(0.25, 0.9, "Forecasting\nweek $t_0$", fontsize=8)
    fig.text(0.5, 0.9, "90 day weather sample,\n$100\%$ observed", fontsize=8)
    fig.text(0.25, 0.7, "Unscaled\nabundance\nprediction", fontsize=8)
    fig.text(0.5, 0.7, "13 weeks of\nobserved trap\ncounts", fontsize=8)
    fig.text(0.5, 0.6, "90 day weather sample,\n$77\%$ observed and $23\%$ forecasted", fontsize=8)

    fig.text(
        0.25, 0.4,
        "(a)\n\n   \nWeather",
        fontsize=8,
        horizontalalignment='center',
        verticalalignment='center',
    )
    fig.text(
        0.5, 0.4,
        "(b)\n\nUnscaled\nAbundance",
        fontsize=8,
        horizontalalignment='center',
        verticalalignment='center',
    )
    fig.text(
        0.7, 0.4,
        "(c)\n\nTrap\nCounts",
        fontsize=8,
        horizontalalignment='center',
        verticalalignment='center',
    )
    fig.text(
        0.85, 0.4,
        "(d)\n\nTrap\nCounts",
        fontsize=8,
        horizontalalignment='center',
        verticalalignment='center',
    )

    fig.savefig(f'{output_path}/text.png', dpi=300, bbox_inches='tight')


def main():
    config = '../fpaths_config.json'
    _, _, nn_preds_path, _, raw_mols_path = gen_utils.load_input_paths(config)
    _, output_path = gen_utils.load_output_paths(config)

    # Load weekly forecast samples
    forecast_path = f'{nn_preds_path}/poisson_forecasts.h5'
    forecasts_list, t0_list = forecast_utils.load_samples_hdf5(forecast_path)

    idx = 90
    t0 = t0_list[idx]   # (you use t0 for weather selection below)
    forecast = forecasts_list[idx]

    # Load NN predictions
    prediction_path = f'{nn_preds_path}/nn_mixed_samples.h5'
    predictions_list, _ = forecast_utils.load_samples_hdf5(prediction_path)
    prediction = predictions_list[idx]

    # Load / slice weather
    weather_path = f'{raw_mols_path}/San_Juan_MoLS_test.csv'
    weathers = pd.read_csv(weather_path)
    weathers['Datetime'] = pd.to_datetime(weathers[['Year', 'Month', 'Day']])

    weather_idx = weathers[weathers.Datetime == t0].index.values[0]
    weather = weathers.iloc[weather_idx-180:weather_idx+30]
    weather = weather[weather.Datetime <= forecast.Datetime.iloc[-1]]

    # Make the main process figure
    if False:
        main_plot(forecast, prediction, weather, output_path)

    # Make the text labels figure
    if True:
        text_plot(output_path)


if __name__ == "__main__":
    main()
