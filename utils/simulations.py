import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import nbinom


def format_mols_data(data_path, model):
    run_cols = sim_cols(50)
    
    daily = pd.read_csv('{}/{}_predictions.csv'.format(data_path, model))
    daily['Datetime'] = pd.to_datetime(daily[['Year', 'Month', 'Day']])
    daily['week'] = daily['Datetime'].dt.isocalendar().week
    daily['year'] = daily['Datetime'].dt.isocalendar().year

    daily['Week'] = daily['Datetime'].dt.to_period('W-SAT')  # Grouping by week ending on Saturday
    week_counts = daily.groupby('Week')['Datetime'].count()
    
    # Keep only weeks with exactly 7 days
    valid_weeks = week_counts[week_counts == 7].index

    # Filter original data to keep only valid weeks
    daily = daily[daily['Datetime'].dt.to_period('W-SAT').isin(valid_weeks)]

    # Group by week and sum the values
    weekly = daily.groupby(pd.Grouper(key='Datetime', freq='W-SAT')).agg({
        'Avg_Temp': 'mean',
        'Humidity': 'mean',
        'Precip': 'sum',
        'MoLS': 'sum',
        'Neural Network': 'sum'
    })

    # Keep only necessary columns
    weekly = weekly.reset_index()[['Datetime', 'Avg_Temp', 'Humidity', 'Precip', 'MoLS', 'Neural Network']]
    for i in range(0,50):
        weekly[f'mols_run_{i}'] = weekly.MoLS

    return weekly


def sim_cols(n):
    run_cols = []
    [run_cols.append('mols_run_{}'.format(i)) for i in range(0,n)]
    return run_cols

def true_poisson_params(p, df):
    ns = np.floor(df.MoLS)
    mu = binom.stats(ns, p, moments='m')
    u_forecast_quant = binom.ppf(0.5+0.68/2, ns, p)
    l_forecast_quant = binom.ppf(0.5-0.68/2, ns, p)
    return mu, u_forecast_quant, l_forecast_quant


def negbin_params(p, seen):
    ns = np.floor(seen)

    mu = nbinom.stats(ns, p, moments='m')
    u_forecast_quant = nbinom.ppf(0.5+0.68/2, ns, p)
    l_forecast_quant = nbinom.ppf(0.5-0.68/2, ns, p)
    return mu, u_forecast_quant, l_forecast_quant



def poisson_sim(p, df):
    run_cols = sim_cols(50)

    for i in range(0,len(run_cols)):
        col = df['mols_run_{}'.format(i)]
        for j in range(0, len(col)):
            total = df.MoLS.iloc[j]
            trials = np.random.random(size = int(total))
            catches = len(np.where(trials < p)[0])

            df['mols_run_{}'.format(i)].iloc[j] = catches
    
    df['Ref'] = df[run_cols].mean(axis=1)
    df['Ref_sd'] = np.sqrt(df[run_cols].var(axis=1))

    return df


def negbin_sim(p, seen_prop, df):
    run_cols = sim_cols(50)
    ns = []
    for i in range(0,len(run_cols)):
        col = df['mols_run_{}'.format(i)]
        for j in range(0, len(col)):
            total = df.MoLS.iloc[j]
            trials = np.random.random(size = int(total))
            n = seen_prop * total
            ns.append(n)

            negbin_catches = 0
            negbin_successes = 0
            for trial in trials:
                if trial<p:
                    negbin_catches += 1
                else:
                    negbin_successes += 1
                
                if negbin_successes>=n:
                    break

            df['mols_run_{}'.format(i)].iloc[j] = negbin_catches

    df['Ref'] = df[run_cols].mean(axis=1)
    df['Ref_sd'] = np.sqrt(df[run_cols].var(axis=1))

    return df


