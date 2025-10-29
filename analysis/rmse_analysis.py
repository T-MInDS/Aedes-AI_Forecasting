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

from sklearn.metrics import mean_squared_error as mse


# autopep8: on

#---------------Error calculation functions
def point_prediction_error(predictions, Ref):
    results = []
    for i in range(len(Ref)):
        abs_err = np.abs(predictions.iloc[i] - Ref.iloc[i]) / Ref.iloc[i]
        results.append(abs_err)

    rmse = np.sqrt(mse(predictions, Ref))
    results.append(rmse)
    
    return results

#---------------Saving results functions
def add_result_line(scores, result, result_fil):
    for score in scores:
        result += f'{score}\t'
    result += '\n'
    with open(result_fil, 'a') as f:
        f.write(result)
        f.close()
    return

def prepare_result_fil(dist, opath):
    result_fil = f'{opath}/{dist}_rmses.csv'
    header = 'dist\tt_0\trmse\twk1_abs_err\twk2_abs_err\twk3_abs_err\twk4_abs_err'
    header += '\n'
    with open(result_fil, 'w') as f:
        f.write(header)
        f.close()
    return result_fil


#---------------Processing samples functions
def process_samples(samples, t0_list, output_path, dist):
    result_fil = prepare_result_fil(dist, output_path)

    for sample, t0 in zip(samples, t0_list):
        result = f'{dist}\t{t0}'
        forecast = sample[sample.Location == 'Forecast']
        scores = point_prediction_error(forecast['Point_predictions'], forecast['Ref'])
        add_result_line(scores, result, result_fil)
    print(f'{dist} point prediction scores saved in {result_fil}')       
    return


def main():
    fpaths_config = '../fpaths_config.json'
    _, _, nn_preds_path, _, _ = gen_utils.load_input_paths(
        fpaths_config)
    
    _, output_path = gen_utils.load_output_paths(fpaths_config)

    #Poisson
    if True:
        poisson_forecast_fil = f'{nn_preds_path}/poisson_forecasts.h5'
        poissons, t0_list = forecast_utils.load_samples_hdf5(poisson_forecast_fil)
        process_samples(poissons, t0_list, output_path, 'poisson')

    #Neg Bin
    if True:
        negbin_forecast_fil = f'{nn_preds_path}/negbin_forecasts.h5'
        negbins, t0_list = forecast_utils.load_samples_hdf5(negbin_forecast_fil)
        process_samples(negbins, t0_list, output_path, 'negbin')
    

if __name__ == "__main__":
    main()
