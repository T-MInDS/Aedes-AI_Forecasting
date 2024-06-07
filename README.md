# Directory for mosquito abundance forecasting with *Aedes-AI* neural networks

This directory contains the methodology used to generate and analyze forecasts. Jupyter notebooks in the ```../notebooks/``` control the workflow, and we provide descriptions of these notebooks here, listed in the order they should be run. The ```fpaths_config.json``` file is referenced throughout the notebooks and controls the file storage of all results.
## ```Aedes_AI-GRU_Average_Temperature.ipynb```
Python code to train the ANN model using average temperature, precipitation, and relative humidity as inputs.
* An overview of the file structure, together with plots for various locations in the testing set, are provided in Aedes_AI-GRU_Average_Temperature.pdf.
* The model was retrained on 11/12/2022 because of updates in sklearn. See https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations for more information.


## ```PR_Mosquito_Abundance.ipynb```
Python code that reads and formats the PR weather input data so that it can be used to create input files for the ANN model.
* Examples of input files are created for 4 locations in Puerto Rico.
* The resulting ANN output is compared to surveillance trap data.
* An overview of the file structure is provided in PR_Mosquito_Abundance.pdf.
* Because the ANN had to be retrained, this file was run again on 11/12/2022.

## ```preprocess_and_smooth.ipynb```
Python code that converts daily predictions to weekly means and smoothes trap data using a 3wk moving average. Results paths are stored in variables ```raw_data``` and ```smoothed_data``` from ```fpaths_config.json```.

## ```calculate_metrics.ipynb```
Python code that scales and forecasts for each day with data availabe, then calculates the overall metrics, interval metrics, and scaling ratios. It does this for the ANN predictions generated using local weather and the airport weather. Results paths are stored in ```mean_metric_fil```, ```amt_metrics```, ```scaling_rto_fil```, and ```baseline_scaling_rto_fil``` from ```fpaths_config.json```.

## ```calculate_coverages.ipynb```
Python code that scales and forecasts for each day with data available, then calculates the confidence intervals for $1-\alpha=0, 0.1, ..., 0.99$. Finally, reports whether the trap data point for each week in the forecast was captured by the corresponding confidence intervals. Results paths is stored in ```coverages_path``` from ```fpaths_config.json```.

## ```mean_metric_figures.ipynb```
Python code to produce figures associated with the mean metrics:
* justify scaling window byu showing RMSE for varying scaler windows
* show the average RMSE for baselines, forecasts with local weather, and forecasts with airport weather
The results path is stored in variable ```metric_figures_path``` from ```fpaths_config.json```.

## ```interval_metric_figures.ipynb```
Python code to produce a figure comparing the RMSE scores for the whole-year forecast to the interval RMSEs. Results path is stored in variable ```metric_figures_path``` from ```fpaths_config.json```.

## ```coverage_figures.ipynb```
Python code to produce a figure showing the coverages of all confidence intervals for all locations with local and airport weather. Results path is stored in variable ```metric_figures_path``` from ```fpaths_config.json```.

## ```representative_forecasts.ipynb```
Python code to produce the representative forecasts for all locations, showing high and low skill forecasts generated with local weather and airport weather. Results path is stored in variable ```misc_figures_path``` from ```fpaths_config.json```.

## ```misc_figures.ipynb```
Python code to produce miscellaneous figures:
* scatter plot of abundance curves generated using local weather and weather from the San Juan airport
* comparison of local weather and San Juan airport weather
* time evolution of $p_w$ and $n_w$ over time
* comparison of scaling ratio for baselines and forecasts for local and airport weather
Results path is stored in variable ```misc_figures_path``` from ```fpaths_config.json```.

