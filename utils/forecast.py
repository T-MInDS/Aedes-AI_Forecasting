import pandas as pd
import numpy as np
import os, glob
import cv2
import matplotlib.dates as mdates
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import utils.utils as utils

from scipy.fft import fft, fftfreq

from sklearn.metrics import mean_squared_error as mse
from scipy.stats import nbinom as nbinom 

def scale_rto(data):
    return data.describe().loc['mean']['Ref'] / data.describe().loc['mean']['Neural Network']

def calculate_avg_p(subset):
    #Get Neg. Bin. parameters from trap data
    mu = subset.Ref.values.astype('float')
    sig = subset.Ref_sd.values.astype('float')
    ps = np.minimum(np.divide(mu, sig**2),1)
    mean_p = np.mean(ps)
    
    return mean_p

def calculate_quant(nn_preds, mean_p, ci):
    #var: sig^2 = mu/p
    var = np.divide(nn_preds, mean_p)
    #n (num successes): n=mu^2/sig^2-mu
    ns = np.divide(nn_preds**2, var - nn_preds)
    
    u_forecast_quant = nbinom.ppf(0.5+ci/2, n=ns, p=mean_p)
    l_forecast_quant = nbinom.ppf(0.5-ci/2, n=ns, p=mean_p)
    
    return ns, l_forecast_quant, u_forecast_quant

def moving_avg(a, n=7):
    ret = pd.DataFrame(a).rolling(n, min_periods=1).mean()
    return ret

def coverage_analysis(site_info, site_nn_data, scaler_forecasts, ofil):
    site, weather_flag, smoothing_flag = site_info
    for scaler_forecast in scaler_forecasts:
        scaler_win, forecast_win = scaler_forecast
              
        for i in range(0, len(site_nn_data) - (scaler_win+forecast_win)):
            cis = np.arange(0,1.1,step=0.1)
            cis[-1] = 0.99
            
            site_nn_subset = site_nn_data.copy(deep=True)
            site_nn_subset = site_nn_subset.iloc[i:i+scaler_win+forecast_win,:]
        
            for ci in cis:
                results = "{}\t{}\t{}\t".format(site, site_nn_subset.Datetime.iloc[-forecast_win], ci)
        
                l_quant, u_quant, ns, mean_p, scaled_nn_preds = main_analysis(scaler_win, forecast_win, ci, site_nn_subset)
                forecast_days = site_nn_subset.iloc[-forecast_win:,:].copy(deep=True)
                
                forecast_days['Lower_quant'] = l_quant
                forecast_days['Upper_quant'] = u_quant
                
                for k in range(0,len(forecast_days)):
                    if (forecast_days.Lower_quant.iloc[k]<=forecast_days.Ref.iloc[k]<=forecast_days.Upper_quant.iloc[k]):
                        results += '1\t'
                    else:
                        results += '0\t'
                results += '\n'
                with open(ofil, 'a') as f:
                    f.write(results)
                    f.close()
    return 

def scaler_analysis(site_nn_data, scaler_wins):
    
    to_return = pd.DataFrame(columns = ['t_0', 'Scaler_rto'])
    for scaler_win in scaler_wins:
        for i in range(0, len(site_nn_data) - (scaler_win)):
            #Get scaler_win+forecast_win days of nn predictions
            site_nn_subset = site_nn_data.copy(deep=True)
            site_nn_subset = site_nn_subset.iloc[i:i+scaler_win,:]                    
            date_list = site_nn_subset.Datetime.values
        
            #Scaler calculated on scaling window 
            scaler = scale_rto(site_nn_subset)   
            to_return = to_return.append({'t_0':date_list[-1], 'Scaler_rto': scaler}, ignore_index=True)
            
    return to_return

def t0_metric_analysis(site_info, site_nn_data, scaler_forecasts, metric_fil, amt = 1):   
    site, weather_flag, smoothing_flag = site_info
    for scaler_forecast in scaler_forecasts:
        scaler_win, forecast_win = scaler_forecast
                
        for i in range(0, len(site_nn_data) - (scaler_win+forecast_win)):
            to_write = '{}\t{}\t{}\t{}\t{}\t'.format(site, weather_flag, scaler_win, forecast_win, smoothing_flag)

            #Get scaler_win+forecast_win days of nn predictions
            site_nn_subset = site_nn_data.copy(deep=True)
            site_nn_subset = site_nn_subset.iloc[i:i+scaler_win+forecast_win,:]
            
            l_quant, u_quant, ns, mean_p, scaled_nn_preds = main_analysis(scaler_win, forecast_win, 99, site_nn_subset)
           
            site_nn_subset['Neural Network'] = scaled_nn_preds                    
            
            forecast_days = site_nn_subset.iloc[-forecast_win:,:]
                        
            trap = forecast_days.Ref.values
            nn = forecast_days['Neural Network'].values
                        
            #Performance metrics on smoothed trap: rmse, nrmse, rel sq err
            total_rmse = mse(trap, nn,  squared=False)
           
            to_write = to_write + '{}\t{}\t'.format(site_nn_subset.Datetime.iloc[scaler_win].date(), total_rmse)
           

            for j in range(0,len(trap), amt):
                wk_rmse = mse(trap[j:j+amt], nn[j:j+amt], squared=False)
                to_write = to_write + '{}\t'.format(wk_rmse)

            to_write = to_write + '\n'
            
            with open(metric_fil, 'a') as f:
                f.write(to_write)
                f.close()

    return 


def avg_metric_analysis(site_info, site_nn_data, scaler_forecasts, metric_fil, write_to_csv=True):
    site, weather_flag, smoothing_flag = site_info
    
    for scaler_forecast in scaler_forecasts:
        scaler_win, forecast_win = scaler_forecast
                
        rmses = []
        for i in range(0, len(site_nn_data) - (scaler_win+forecast_win)):
            #Get scaler_win+forecast_win days of nn predictions
            site_nn_subset = site_nn_data.copy(deep=True)
            site_nn_subset = site_nn_subset.iloc[i:i+scaler_win+forecast_win,:]                    

            l_quant, u_quant, ns, mean_p, scaled_nn_preds = main_analysis(scaler_win, forecast_win, 99, site_nn_subset)
            
            site_nn_subset['Neural Network'] = scaled_nn_preds                    
            
            forecast_days = site_nn_subset.iloc[-forecast_win:,:]
                        
            trap = forecast_days.Ref.values
            nn = forecast_days['Neural Network'].values
            
            #Performance metrics on smoothed trap: rmse
            rmse = mse(trap, nn, squared=False)
            rmses.append(rmse)
            
        if ((scaler_win==13) & (forecast_win==52)):
            quantiles = np.quantile(rmses, [0.25, 0.75])
            first_quantile = np.nanargmin(np.abs(rmses - quantiles[0]))
            third_quantile = np.nanargmin(np.abs(rmses - quantiles[-1]))
            print(site_info, first_quantile, third_quantile)

        avg_rmse = np.average(rmses)               
        sig_rmse = np.std(rmses)

        if write_to_csv:
            with open(metric_fil, 'a') as f:
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(site, weather_flag, scaler_win, forecast_win,
                                                                       smoothing_flag, avg_rmse, sig_rmse))
                
    return 

def main_analysis(scaler_win, forecast_win, ci, site_nn_subset):
    #Scaler calculated on scaling window 
    scaler = scale_rto(site_nn_subset.iloc[:scaler_win,:])   

    #Scale s.t. avg. nn is avg. trap
    scaled_nn_preds = scaler*site_nn_subset['Neural Network']

    #Get Neg. Bin. parameters from trap data in scaling window
    mean_p = calculate_avg_p(site_nn_subset.iloc[:scaler_win,:])     
    
    #Calculate conf int using predictions in forecasting window
    ns, l_quant, u_quant = calculate_quant(scaled_nn_preds[-forecast_win:], mean_p, ci)

    return l_quant, u_quant, ns, mean_p, scaled_nn_preds

def plot_analysis(site_info, site_nn_data, scaler_forecasts, path, movie=True):
    site, weather_flag, smoothing_flag = site_info

    for scaler_forecast in scaler_forecasts:
        scaler_win, forecast_win = scaler_forecast
        
        for i in range(0, len(site_nn_data) - (scaler_win+forecast_win)):
            #Get scaler_win+forecast_win days of nn predictions
            site_nn_subset = site_nn_data.copy(deep=True)
            site_nn_subset = site_nn_subset.iloc[i:i+scaler_win+forecast_win,:]                    
    
            date_list = site_nn_subset.Datetime.values

            _,_, ns, mean_p, scaled_nn_preds = main_analysis(scaler_win, forecast_win, 0, site_nn_subset)

            #Plot forecasts with t_0<plt_date
            plt_end_date = np.datetime64('2014-04-01')
            plt_st_date = np.datetime64('2013-04-01')
            
            if movie:
                if ((date_list[0]> plt_st_date) & (date_list[scaler_win] < plt_end_date)):
                    plot_forecast_ints(site_info, i, date_list, forecast_win, scaled_nn_preds, site_nn_subset, mean_p, ns, movie, path)
                elif date_list[scaler_win] >= plt_end_date:
                    break
            if not movie:
                dict_idxs = dict(arboleda = [], playa = [], la_margarita = [], villodas = [])
                if (smoothing_flag==True):
                    if (weather_flag=='site'):
                        dict_idxs = dict(arboleda = [235, 384], playa = [415, 333], la_margarita = [1693, 7], villodas = [166, 1608])
                    else:
                        dict_idxs = dict(arboleda = [765], playa = [1516], la_margarita = [1727], villodas = [762])

                if (((i % 25) == 0) or (i in dict_idxs[site.lower()])):
                    plot_forecast_ints(site_info, i, date_list, forecast_win, scaled_nn_preds, site_nn_subset, mean_p, ns, movie, path)
    return 



def plot_forecast_ints(site_info, i, date_list, forecast_win, scaled_nn_preds, site_nn_subset, p , n, movie, path):   
    site, weather_flag, smoothing_flag = site_info
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [20,1]}, num=i, figsize=(6,4))
    forecast_ax = axs[0]
    axs[1].axis('off')
    
    #Set colors for plot
    maria_col, trap_col, nn_col = 'cornflowerblue', 'tab:orange', 'tab:blue'
    
    if 'Arboleda' in site:
        ymax = 36
    elif (('Villodas' in site) or ('La_Margarita' in site)):
        ymax = 10
    elif 'Playa' in site:
        ymax = 60
    ymin=0

    x_scale = date_list[0:-forecast_win]
    x_forecast = date_list[-forecast_win:]
    
    ref = site_nn_subset['Ref']
    ref_sd = site_nn_subset['Ref_sd']
    
    trap_up_err = ref + ref_sd    
    trap_lo_err = ref - ref_sd
    trap_lo_err[trap_lo_err<0] = 0
    
    scaled_nn_preds = moving_avg(scaled_nn_preds, n=15)
    scaled_nn_preds = moving_avg(scaled_nn_preds, n=15)

    nn_scale = scaled_nn_preds.iloc[:-forecast_win]  
    nn_forecast = scaled_nn_preds.iloc[-forecast_win:]
    
    #set alpha value for scaler points
    scaler_alpha=0.35
    
    #Plot (and label on graph) 68% quantile
    ci=.68
    u_forecast_quant = nbinom.ppf(0.5+ci/2, n=n, p=p)
    l_forecast_quant = nbinom.ppf(0.5-ci/2, n=n, p=p)

    u_forecast_quant = moving_avg(u_forecast_quant, n=7).values[-forecast_win:,0]
    l_forecast_quant = moving_avg(l_forecast_quant, n=7).values[-forecast_win:,0]
        
    forecast_ax.fill_between(x_forecast, u_forecast_quant, l_forecast_quant, alpha=0.25, color=nn_col)
    
    #Mark start of forecast window
    forecast_ax.axvline(x_forecast[0], linestyle='--', color='k', alpha=0.5)

    #Plot nn predictions (split to make scaling window lighter)
    forecast_ax.plot(x_scale, nn_scale, alpha=scaler_alpha, color=nn_col)
    forecast_ax.plot(x_forecast, nn_forecast, color=nn_col, label='Aedes-AI NN')
        
    #Plot weekly trap data and standard error
    forecast_ax.scatter(x_scale[3::7], ref[0:-forecast_win][3::7], alpha=scaler_alpha, color=trap_col)
    forecast_ax.scatter(x_forecast[3::7], ref[-forecast_win:][3::7], color=trap_col, label='Trap')
    forecast_ax.fill_between(date_list[3::7], trap_lo_err[3::7], trap_up_err[3::7], alpha=0.15, color=trap_col)

    forecast_ax.legend(loc='upper left')
    forecast_ax.set_ylabel('Abundance')
    forecast_ax.legend(loc='upper left')
    #forecast_ax.set_title(site.replace('_',' '))

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter("%b'%y")
    forecast_ax.xaxis.set_major_locator(locator)
    forecast_ax.xaxis.set_major_formatter(fmt)

    forecast_ax.set_ylim([ymin, ymax])

    for label in forecast_ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    if not movie:
        nn_forecast = nn_forecast.values[:,0]       
        rmse = round(mse(ref[-forecast_win:], nn_forecast, squared=False), ndigits=2)
        forecast_ax.text(0.9945*forecast_ax.get_xlim()[-1], 0.95*forecast_ax.get_ylim()[-1], 'RMSE: {}'.format(rmse))
    
    if smoothing_flag:
        opath = '{}/{}_{}_42k'.format(path, site, weather_flag)
    else:
        opath = '{}/{}_{}_raw'.format(path, site, weather_flag)

    if not os.path.exists(opath):
        os.mkdir(opath)
    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(opath, i), dpi=300, bbox_inches='tight')
    plt.close('all')
    return

