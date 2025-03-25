import pandas as pd
import numpy as np


def format_mols_data(loc, data_path):
    run_cols = sim_cols(50)
    
    daily = pd.read_csv('{}/{}_gru_avg_temp_predictions.csv'.format(data_path, loc))
    daily['Datetime'] = pd.to_datetime(daily[['Year', 'Month', 'Day']])
    #daily['mols_avg'] = np.average(daily[run_cols], axis=1)
    weekly = daily.groupby(pd.Grouper(key='Datetime', freq='W-SAT')).sum()
    #90 day burn in
    weekly = weekly.iloc[13:]
    weekly = weekly.reset_index()[['Datetime', 'MoLS', 'Neural Network']]
    
    for i in range(0,50):
        weekly[f'mols_run_{i}'] = weekly.MoLS

    return weekly


def sim_cols(n):
    run_cols = []
    [run_cols.append('mols_run_{}'.format(i)) for i in range(0,n)]
    return run_cols



def poisson_sim(p, df):
    run_cols = sim_cols(50)

    for i in range(0,len(run_cols)):
        col = df['mols_run_{}'.format(i)]
        for j in range(0, len(col)):
            total = df.MoLS.iloc[j]
            trials = np.random.random(size = int(total))
            catches = len(np.where(trials < p)[0])

            df['mols_run_{}'.format(i)].iloc[j] = catches
    
    df['weekly_mean'] = df[run_cols].mean(axis=1)
    df['weekly_var'] = df[run_cols].var(axis=1)

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

    df['weekly_mean'] = df[run_cols].mean(axis=1)
    df['weekly_var'] = df[run_cols].var(axis=1)

    return df


