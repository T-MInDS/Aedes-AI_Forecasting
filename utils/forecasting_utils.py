# autopep8: off

import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# autopep8: on

def merge_nn_traps(samples, traps):
    traps['Datetime'] = pd.to_datetime(traps['Datetime'])
    
    for idx in range(len(samples)):
        sample = samples[idx]
        sample['Datetime'] = pd.to_datetime(sample['Datetime'])

        cols = trial_names()
        [cols.append(col) for col in ['Ref', 'Ref_sd', 'True_mu', 'True_u_forecast_quant', 'True_l_forecast_quant']]

        for col in cols:
            sample[col] = sample['Datetime'].map(traps.set_index('Datetime')[col])

        samples[idx] = sample
        
    return samples
        
    

def trial_names(n=50):
    run_cols = []
    [run_cols.append('mols_run_{}'.format(i)) for i in range(0, n)]
    return run_cols

def convert_to_weekly(data):
    data = data.copy()
    data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

    # Define week buckets ending on Saturday
    data['Week'] = data['Datetime'].dt.to_period('W-WED')


    sum_cols = [c for c in data.columns if c in ['Precip', 'MoLS', 'Ref', 'Neural Network']]

    agg_dict = {'days_in_week': ('Datetime', 'count'),
                'Avg_Temp': ('Avg_Temp', 'mean'),
                'Humidity': ('Humidity', 'mean'),
                'Location': ('Location', 'first')
                }
    for c in sum_cols:
        agg_dict[c] = (c, 'sum')

    # Group by week and sum the values
    weekly = data.groupby('Week').agg(**agg_dict)

    # Scale sums up to a 7-day equivalent if the week has < 7 days
    # (cap at 1.0 if somehow there are duplicated days and count > 7)
    n = weekly['days_in_week'].astype(float).clip(lower=1)  # guard against 0
    
    factor = np.where(n >= 7, 1.0, 7.0 / n)

    for col in sum_cols:
        weekly[col] = weekly[col] * factor

    weekly['Datetime'] = weekly.index.to_timestamp(how='start').normalize()

    # Now these work fine:
    weekly['Year'] = weekly['Datetime'].dt.year
    weekly['Month'] = weekly['Datetime'].dt.month
    weekly['Day'] = weekly['Datetime'].dt.day

    # Reorder and drop unnecessary columns
    out_cols = ['Location', 'Datetime', 'Year', 'Month', 'Day', 'Avg_Temp', 'Humidity'] + sum_cols
    weekly = weekly.reset_index(drop=True)[out_cols]
    
    return weekly



def load_weather_data(raw_mols):
    """Load weather data.
    Only load the test data, since <2017 used to finetune
    Args:
        raw_mols (str): Path to the raw MoLS data file.

    Returns:
        observed_weather (pd.DataFrame): DataFrame containing the San Juan weather data.
        forecasted_weather (pd.DataFrame): DataFrame containing the Ceiba weather data.
    """

    observed_weather = pd.read_csv('{}/San_Juan_MoLS_test.csv'.format(raw_mols))
    forecasted_weather = pd.read_csv('{}/Ceiba_MoLS_test.csv'.format(raw_mols))

    return observed_weather, forecasted_weather


def configure_sample(
    t0: pd.Timestamp,
    observed_weather: pd.DataFrame,
    forecasted_weather: pd.DataFrame,
    date_col: str = "Datetime",
    scale_win: int = 180,   # observed window length: [t0-scale_win, t0-1]
    f_win: int = 30,       # forecast window length: [t0, t0+f_win-1]
    ref_col: str = "Ref",
):
    """
    Configure one sample consisting of:
      - observed scale_win-day window:  [t0-scale_win, t0-1]
      - forecast f_win-day window:  [t0, t0+f_win]
    The function:
      1. Verifies both windows have *no missing calendar days*.
      2. Combines observed and forecast windows into one DataFrame.
      3. Adds a 'Source' column ("observed" or "forecasted").
    Returns the combined DataFrame, or None if either window is incomplete.
    """

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        if date_col not in df.columns:
            raise KeyError(f"Expected '{date_col}' column.")
        out = df.copy()
        out[date_col] = pd.to_datetime(
            out[date_col]).dt.normalize()
        out = out.sort_values(date_col)
        out = out[~out[date_col].duplicated(keep="last")]
        return out.reset_index(drop=True)

    obs = prep(observed_weather)
    fct = prep(forecasted_weather)

    # --- Observed window: t0-scale_win .. t0-1
    scale_start = pd.to_datetime(t0).floor("D") - pd.Timedelta(days=scale_win)
    scale_end = pd.to_datetime(t0).floor("D") - pd.Timedelta(days=1)
    idx_obs = pd.date_range(scale_start, scale_end, freq="D")

    # --- Forecast window indices
    f_start = pd.to_datetime(t0).floor("D")
    f_end = f_start + pd.Timedelta(days=f_win - 1)
    idx_fct = pd.date_range(f_start, f_end, freq="D")

    # Slice and reindex to validate completeness (no gaps)
    obs_win = (obs.set_index(date_col)
                  .loc[scale_start:scale_end]
                  .reindex(idx_obs))
    fct_win = (fct.set_index(date_col)
                  .loc[f_start:f_end]
                  .reindex(idx_fct))

    # If either window has any missing rows after reindex, bail
    if obs_win.shape[0] != scale_win or obs_win.index.hasnans or obs_win.isna().all(axis=None):
        return None
    if fct_win.shape[0] != f_win or fct_win.index.hasnans or fct_win.isna().all(axis=None):
        return None

    if (obs_win.isna().any().any()) or (fct_win.isna().any().any()):
        return None

    # --- Combine

    combined = pd.concat([obs_win, fct_win], axis=0)
    combined.index.name = date_col
    combined = combined.reset_index().sort_values(date_col).reset_index(drop=True)

    # ---- Map observed Ref to ALL combined dates (even forecast rows)
    if ref_col in obs.columns:
        ref_series = obs.set_index(date_col)[ref_col]
        combined[ref_col] = combined[date_col].map(ref_series)
        # Interpolate only for numeric refs; otherwise leave as is
        if pd.api.types.is_numeric_dtype(ref_series):
            combined[ref_col] = (combined[ref_col]
                                 .interpolate(method="linear", limit_direction="both")
                                 .ffill()
                                 .bfill())

    return combined


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    COL_ORDER = df.columns
    # required columns present?
    missing = [c for c in COL_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # datatypes
    df['Datetime'] = pd.to_datetime(df['Datetime']).astype('datetime64[ns]')
    df['Location'] = df['Location'].astype('string')    # safer than category for PyTables table
    for c in ['Year', 'Month', 'Day', 'Avg_Temp', 'Precip', 'Humidity', 'Ref', 'Neural Network', 'MoLS']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # keep Y/M/D as Int64 (nullable) if you want integers with NaN support, else leave float64
    for c in ['Year', 'Month', 'Day']:
        if c in df.columns:
            df[c] = df[c].astype('Int64') if df[c].isna().any() else df[c].astype('int64')

    # strictly float for continuous cols
    for c in ['Avg_Temp', 'Precip', 'Humidity', 'Ref', 'Neural Network']:
        if c in df.columns:
            df[c] = df[c].astype('float64')
            df.loc[~np.isfinite(df[c]), c] = np.nan  # remove infs

    # final guard: NN must be float or NaN only
    bad = df['Neural Network'].map(
        lambda v: not (
            pd.isna(v) or isinstance(
                v, (float, np.floating))))
    if bad.any():
        ex = df.loc[bad, ['Datetime', 'Neural Network']].head()
        raise TypeError(f"'Neural Network' has non-floats after coercion. Examples:\n{ex}")

    return df


def save_samples_to_hdf5(samples, t0_list, opath):
    """
    Save a list of DataFrames to an HDF5 file.
    Each sample is stored under a key: /sample_<YYYYMMDD>
    """
    if os.path.exists(opath):
        os.remove(opath)

    with pd.HDFStore(opath, mode="w") as store:
        for df, t0 in zip(samples, t0_list):
            key = f"sample_{pd.to_datetime(t0).strftime('%Y%m%d')}"
            df_clean = _coerce_schema(df)
            # allows efficient chunked access
            store.put(key, df_clean, format="table")

    print(f"Saved {len(samples)} samples to {opath}")
    return


def load_samples_hdf5(fil):
    samples, t0_list = [], []

    with pd.HDFStore(fil, mode="r") as store:
        for key in store.keys():
            df = store[key]
            samples.append(df)
            # extract date from key
            t0 = pd.to_datetime(key.replace("/sample_", ""))
            t0_list.append(t0)
    print(f'Loaded {len(samples)} samples from {fil}')
    return samples, t0_list
