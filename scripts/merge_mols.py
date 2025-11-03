"""Create the data for each step in the forecasting process

    Starts with raw weather files from NOAA and outputs
    """

# autopep8: off

import sys, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json, glob
import matplotlib.pyplot as plt
import pandas as pd
import utils.format_data_utils as data_utils
from utils import gen_utils

# autopep8: on


def process_mols(raw_mols_path: str):
    """Process MoLS prediction files by applying burn-in and burn-out periods.

    Reads all MoLS prediction CSV files from the specified directory, removes the first 10 days (burn-in)
    and the last 90 days (burn-out) from each file, and saves the cleaned data back to the same files.

    Note:
        This function assumes that the MoLS prediction files are named with the pattern '*_MoLS.csv'
        and are located in the directory specified by `raw_mols_path`.
    """
    # Manual process: Obtain corresponding MoLS predictions for loc_daily.pd files and store them in raw_mols_path
    # Finally, 180 day burn in and 90 day burn out

    fils = glob.glob('{}/*_MoLS.csv'.format(raw_mols_path))
    for fil in fils:
        data = pd.read_csv(fil)
        data = data.iloc[180:-90].reset_index(drop=True)
        data.to_csv(fil, index=False)

        test = data[data.Year>2016]
        test_fil = fil.replace('MoLS', 'MoLS_test')
        test.to_csv(test_fil, index=False)
    return

def split_test(raw_mols_path: str):
    """Process MoLS prediction files by applying burn-in and burn-out periods.

    Reads all MoLS prediction CSV files from the specified directory, removes the first 10 days (burn-in)
    and the last 90 days (burn-out) from each file, and saves the cleaned data back to the same files.

    Note:
        This function assumes that the MoLS prediction files are named with the pattern '*_MoLS.csv'
        and are located in the directory specified by `raw_mols_path`.
    """
    # Manual process: Obtain corresponding MoLS predictions for loc_daily.pd files and store them in raw_mols_path
    # Finally, 180 day burn in and 90 day burn out

    fils = glob.glob('{}/*_MoLS.csv'.format(raw_mols_path))
    for fil in fils:
        data = pd.read_csv(fil)

        test = data[data.Year>2016]
        test_fil = fil.replace('MoLS', 'MoLS_test')
        test.to_csv(test_fil, index=False)
    return


def main():
    paths = '../fpaths_config.json'
    _, _, _, _, raw_mols_path = gen_utils.load_input_paths(paths)

    #Manual merge from MoLS
    process_mols(raw_mols_path)
    split_test(raw_mols_path)


if __name__ == "__main__":
    main()
