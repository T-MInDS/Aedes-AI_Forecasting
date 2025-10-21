# autopep8: on

import pandas as pd
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# autopep8: off

def load_weather_data(raw_mols):
    """Load weather data.
    Args:
        raw_mols (str): Path to the raw MoLS data file.

    Returns:
        observed_weather (pd.DataFrame): DataFrame containing the San Juan weather data.
        forecasted_weather (pd.DataFrame): DataFrame containing the Ceiba weather data.
    """

    observed_weather = pd.read_csv('{}/San_Juan_MoLS.csv'.format(raw_mols))
    forecasted_weather = pd.read_csv('{}/Ceiba_MoLS.csv'.format(raw_mols))

    return observed_weather, forecasted_weather


def configure_sample(
    t0: pd.Timestamp,
    observed_weather: pd.DataFrame,
    forecasted_weather: pd.DataFrame,
    date_col: str = "Datetime",
    scale_win: int = 90,   # observed window length: [t0-scale_win, t0-1]
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
    scale_start = t0 - pd.Timedelta(days=scale_win)
    scale_end = t0 - pd.Timedelta(days=1)
    expected_obs = pd.date_range(scale_start, scale_end, freq="D")

    obs_mask = obs[date_col].between(scale_start, scale_end, inclusive="both")
    obs_win = obs.loc[obs_mask].copy().sort_values(date_col)

    if len(obs_win) != scale_win or not pd.Index(obs_win[date_col].values).equals(expected_obs):
        return None

    # --- Forecast window: t0 .. t0+f_win
    f_start = t0
    f_end = t0 + pd.Timedelta(days=f_win-1)
    expected_fct = pd.date_range(f_start, f_end, freq="D")

    fct_mask = fct[date_col].between(f_start, f_end, inclusive="both")
    fct_win = fct.loc[fct_mask].copy().sort_values(date_col)

    if len(fct_win) != f_win or not pd.Index(fct_win[date_col].values).equals(expected_fct):
        return None

    # --- Combine

    combined = pd.concat([obs_win, fct_win], ignore_index=True)

    # ---- Map observed Ref to ALL combined dates (even forecast rows)
    if ref_col in obs.columns:
        ref_map = dict(zip(obs[date_col], obs[ref_col]))
        combined[ref_col] = combined[date_col].map(ref_map)
        combined[ref_col] = combined[ref_col].interpolate().ffill().bfill()
    return combined


def save_samples_to_hdf5(samples, t0_list, opath):
    """
    Save a list of DataFrames to an HDF5 file.
    Each sample is stored under a key: /sample_<YYYY-MM-DD>
    """
    if os.path.exists(opath):
        os.remove(opath)

    with pd.HDFStore(opath, mode="w") as store:
        for df, t0 in zip(samples, t0_list):
            key = f"sample_{pd.to_datetime(t0).strftime('%Y-%m-%d')}"
            # allows efficient chunked access
            store.put(key, df, format="table")
    print(f"Saved {len(samples)} samples to {opath}")
    return
