import numpy as np
import pandas as pd
import os, sys, json
sys.path.append( os.path.abspath(os.path.join('..')) )
import utils.utils as utils
import methods.forecast as forecast
from scipy.stats import nbinom as nbinom
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Opening config file
f = open('fpaths_config.json')
paths = json.load(f)

raw_data_path = paths['raw_data']
smoothed_data_path = paths['smoothed_data']



def still_shot(airport_flag, site, idx, scaler_win, forecast_win, avg_mos):
    if airport_flag:
        nn_fil_name = '{}/{}_airport_raw_weekly_predictions.csv'.format(raw_data_path,site)
    else:
        nn_fil_name = '{}/{}_site_raw_weekly_predictions.csv'.format(raw_data_path,site)
    
    
    site_nn_data = utils.load_csv(nn_fil_name)
    site_nn_data.Datetime = pd.to_datetime(site_nn_data.Datetime)

    site_nn_subset = site_nn_data.copy(deep=True)
    site_nn_subset = site_nn_subset.iloc[idx:idx+scaler_win+forecast_win]

    date_list = site_nn_subset.Datetime.values

    _, _, ns, mean_p, scaled_nn_preds = forecast.main_analysis(scaler_win, forecast_win, 0, site_nn_subset)

    x_scale = date_list[0:-forecast_win]
    x_forecast = date_list[-forecast_win:]
    
    ref = site_nn_subset['Ref']
    ref_sd = site_nn_subset['Ref_sd']
        
    ref = ref/avg_mos
    ref_sd = ref_sd/avg_mos
    
    trap_up_err = ref + ref_sd    
    trap_lo_err = ref - ref_sd
    trap_lo_err[trap_lo_err<0] = 0
    
    nn_scale = scaled_nn_preds.iloc[:-forecast_win]  
    nn_forecast = scaled_nn_preds.iloc[-forecast_win:]
        
    #Plot (and label on graph) 68% quantile
    sig = np.sqrt(nbinom.stats(ns, mean_p, moments='v'))
    
    u_forecast_sd = nn_forecast+sig
    l_forecast_sd = nn_forecast-sig


    u_forecast_sd = u_forecast_sd/avg_mos
    l_forecast_sd = l_forecast_sd/avg_mos
    nn_scale = nn_scale/avg_mos
    nn_forecast = nn_forecast/avg_mos

    return x_forecast, u_forecast_sd, l_forecast_sd, x_scale, nn_scale, nn_forecast, ref, trap_lo_err, trap_up_err, date_list

def still_add_to_ax(ax, x_forecast, u_forecast_sd, l_forecast_sd, x_scale, nn_scale, nn_forecast, ref, trap_lo_err, trap_up_err, date_list, forecast_win):
    trap_col, nn_col = 'tab:orange', 'tab:blue'
    #set alpha value for scaler points
    scaler_alpha=0.35

    ax.fill_between(x_forecast, u_forecast_sd, l_forecast_sd, alpha=0.25, color=nn_col)

    #Mark start of forecast window
    ax.axvline(x_forecast[0], linestyle='--', color='k', alpha=0.5)

    #Plot nn predictions (split to make scaling window lighter)
    ax.plot(x_scale, nn_scale, alpha=scaler_alpha, color=nn_col)
    ax.plot(x_forecast, nn_forecast, color=nn_col, label='Aedes-AI NN')

    #Plot weekly trap data and standard error
    ax.scatter(x_scale, ref[0:-forecast_win], alpha=scaler_alpha, color=trap_col)
    ax.scatter(x_forecast, ref[-forecast_win:], color=trap_col, label='AGO')
    ax.fill_between(date_list, trap_lo_err, trap_up_err, alpha=0.15, color=trap_col)
    
    ax.set_ylim(bottom=0)

    return ax

def formatting_single_plt(ax, row_idx, col_idx, site, rto, ref, nn_forecast, forecast_win):
    if row_idx == 0:
        if col_idx==0:   
            ax.legend(loc='upper left')     
            ax.set_title('(Higher skill)', size='medium')
        elif col_idx==1:
            ax.annotate("Forecasts (Local weather)", xy=(-0.05,1.25), xytext=(-0.05,1.35), size='large', ha='center', va='bottom', xycoords='axes fraction', 
                        arrowprops=dict(arrowstyle='-[, widthB=10.0, lengthB=0.5', lw=1.0, color='k'))
            ax.set_title('(Lower skill)', size='medium')
        elif col_idx==2:
            ax.set_title('(Higher skill)', size='medium')
        else:
            ax.annotate("Forecasts (Airport weather)", xy=(-0.05,1.25), xytext=(-0.05,1.35), size='large', ha='center', va='bottom', xycoords='axes fraction', 
                        arrowprops=dict(arrowstyle='-[, widthB=10.0, lengthB=0.5', lw=1.0, color='k'))
            ax.set_title('(Lower skill)', size='medium')
        
    if col_idx==0:
        ax.set_ylabel('Scaled Trap\nAbundance')
        ax.text(ax.get_xlim()[0]-150, rto*ax.get_ylim()[-1], site.replace('_',' '), bbox=dict(boxstyle="square", fc="white"), rotation=90, size='large')
    else:
        ax.axes.get_yaxis().set_ticklabels([])
    
        
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b'%y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    for label,label_idx in zip(ax.xaxis.get_ticklabels(),range(len(ax.xaxis.get_ticklabels()))):
        if not ((label_idx % 3) == 0):
            label.set_visible(False)
        
    rmse = round(mse(ref[-forecast_win:], nn_forecast, squared=False), ndigits=2)
    ax.text(0.9915*ax.get_xlim()[-1], 0.87*ax.get_ylim()[-1], 'RMSE: {}'.format(rmse))
    
    return ax

def formatting_hi(ax, col_idx, site, rto, ref, nn_forecast, forecast_win):
    if site.lower()=='villodas':
        lims = [3.5,15]
    elif site.lower()=='la_margarita':
        lims = [5,20]
    else:
        lims = [3.5,7]
    ax.set_ylim(lims)

    if col_idx>0:
        ax.axes.get_yaxis().set_ticklabels([])
        
    rmse = round(mse(ref[-forecast_win:], nn_forecast, squared=False), ndigits=2)
    if site.lower()=='villodas':
        ax.text(0.9917*ax.get_xlim()[-1], 0.64*ax.get_ylim()[-1], 'RMSE: {}'.format(rmse))    
    elif site.lower()=='la_margarita':
        ax.text(0.9917*ax.get_xlim()[-1], 0.65*ax.get_ylim()[-1], 'RMSE: {}'.format(rmse))    
    else:
        ax.text(0.9917*ax.get_xlim()[-1], 0.74*ax.get_ylim()[-1], 'RMSE: {}'.format(rmse))    

    ax.spines.bottom.set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)
    

    return ax

def formatting_lo(ax, row_idx, col_idx, site, rto, ref, nn_forecast, forecast_win):
    if site.lower()=='villodas':
        lims = [0,2.5]
    elif site.lower()=='la_margarita':
        lims = [0,3]
    else:
        lims = [0,3.1]

    ax.set_ylim(lims)
    if col_idx==0:
        ax.text(ax.get_xlim()[0]-150, rto*ax.get_ylim()[-1], site.replace('_',' '), bbox=dict(boxstyle="square", fc="white"), rotation=90, size='large')
        if site.lower()=='villodas':
            ax.text(ax.get_xlim()[0]-100, (rto-0.05)*ax.get_ylim()[-1], 'Scaled Trap\nAbundance', rotation=90, size='medium')
        elif site.lower()=='la_margarita':
            ax.text(ax.get_xlim()[0]-100, (rto+0.1)*ax.get_ylim()[-1], 'Scaled Trap\nAbundance', rotation=90, size='medium')
        else:
            ax.text(ax.get_xlim()[0]-100, (rto-0.15)*ax.get_ylim()[-1], 'Scaled Trap\nAbundance', rotation=90, size='medium')
    else:
        ax.axes.get_yaxis().set_ticklabels([])
        
    ax.spines.top.set_visible(False)
    ax.xaxis.tick_bottom()

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b'%y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    ax.set_ylim(lims)

    for label,label_idx in zip(ax.xaxis.get_ticklabels(),range(len(ax.xaxis.get_ticklabels()))):
        if not ((label_idx % 3) == 0):
            label.set_visible(False)

    if row_idx==3:
        if col_idx==0:
            ax.set_xlabel('(a)')
        elif col_idx==1:
            ax.set_xlabel('(b)')
        elif col_idx==2:
            ax.set_xlabel('(c)')
        else:
            ax.set_xlabel('(d)')

    return ax