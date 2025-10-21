import json
import os
from typing import Tuple


def load_paths(fpath: str) -> Tuple[str, str, str, str]:
    """Read in path names from json file.

    Args:
        fpath: file path to json file containing paths

    Returns:
        paths to weather data, raw mols data, nn predictions, and model files
    """
    with open(os.path.expanduser(fpath), 'r') as f:
        paths = json.load(f)

    weather_path = paths["weather_data"]
    mols_path = paths["raw_mols"]
    nn_preds_path = paths["nn_abundance_predictions"]
    model_files_path = paths["model_files"]
    raw_mols_path = paths["raw_mols"]

    return weather_path, mols_path, nn_preds_path, model_files_path, raw_mols_path


def load_output_paths(fpath: str) -> Tuple[str, str, str]:
    """Read in output path names from json file.

    Args:
        fpath: file path to json file containing paths

    Returns:
        paths to figures, processed data, and results
    """
    with open(os.path.expanduser(fpath), 'r') as f:
        paths = json.load(f)

    figures_path = paths["figures"]
    processed_data_path = paths["processed_data"]
    results_path = paths["results"]

    return figures_path, processed_data_path, results_path
