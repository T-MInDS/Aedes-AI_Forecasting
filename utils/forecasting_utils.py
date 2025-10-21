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
    scale_start = pd.to_datetime(t0).floor("D") - pd.Timedelta(days=scale_win)
    scale_end   = pd.to_datetime(t0).floor("D") - pd.Timedelta(days=1)
    idx_obs = pd.date_range(scale_start, scale_end, freq="D")

    # --- Forecast window indices
    f_start = pd.to_datetime(t0).floor("D")
    f_end   = f_start + pd.Timedelta(days=f_win - 1)
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
            # allows efficient chunked access
            store.put(key, df, format="table")
    print(f"Saved {len(samples)} samples to {opath}")
    return
