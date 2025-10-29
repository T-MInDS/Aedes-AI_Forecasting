"""
Aedes-AI GRU Model with Mean Temperature as Input

This code trains and tests a GRU model that estimates Ae. aegypti abundance from:
- Daily mean temperature (°C)
- Daily precipitation (cm)
- Daily relative humidity (%)

The GRU is trained to reproduce MoLS outputs.
"""

# autopep8: off

import sys, os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple
import pickle, json, argparse
from utils import training as training_utils
from utils import predictions
from utils import models
from utils import gen_utils

# autopep8: on

# ------------------------- IO HELPERS -------------------------


def update_training_config(config_path: str, model_files_path: str) -> None:
    """Update model training config file with correct paths and data shape."""
    config_path = os.path.expanduser(config_path)
    model_files_path = os.path.expanduser(model_files_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    config['data']['data_shape'] = [90, 3]
    config['files']['model'] = f'{model_files_path}/gru_avg_temp.keras'
    config['files']['training'] = f'{model_files_path}/train_avg_data.pd'
    config['files']['validation'] = f'{model_files_path}/val_avg_data.pd'
    config['files']['testing'] = f'{model_files_path}/test_avg_data.pd'

    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=2)


# ------------------------- TRAINING -------------------------

def train(model_files_path: str, config_name: str = "gru_avg_temp_config.json") -> None:
    """ Train and save a GRU model for Ae. aegypti abundance prediction. """
    # GPU setup (safe defaults)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # Optional: cap memory (in MiB). Comment says 3GB; set 3072 to match.
            # Remove this block if memory growth alone is sufficient.
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=3072)]
            )
        except RuntimeError as e:
            print("GPU config warning:", e)

    np.random.seed(14)
    tf.random.set_seed(14)

    # load model config
    cfg_path = os.path.join(os.path.expanduser(model_files_path), config_name)
    with open(cfg_path) as fp:
        config = json.load(fp)

    model_file = os.path.expanduser(config['files']['model'])

    if os.path.exists(model_file):
        model = tf.keras.models.load_model(
            model_file, custom_objects={'r2_keras': predictions.r2_keras}
        )
        print(
            f'{model_file} already exists; will continue training based on this checkpoint.')
    else:
        # Build the model using your registry in utils.models
        model_ctor = getattr(models, config['model'])
        model = model_ctor(config['data']['data_shape'])

    # Load seasonal location lists
    hi_locs_csv = os.path.expanduser(f"{model_files_path}/hi_locs.csv")
    if os.path.exists(hi_locs_csv):
        # robust load in case csv has header/index
        hi_df = pd.read_csv(hi_locs_csv)
        # assume first column holds the string locations if unknown
        summer_cities = set(hi_df.iloc[:, 0].astype(str).tolist())
    else:
        summer_cities = set()

    winter_cities = {
        "Dane,Wisconsin", "Milwaukee,Wisconsin", "New Haven,Connecticut",
        "Bronx,New York", "Kings,New York", "Monmouth,New Jersey",
        "Mono,California", "Monterey,California", "Morris,New Jersey",
        "Napa,California", "Nassau,New York", "New Hanover,North Carolina",
        "New River,Arizona", "Okaloosa,Florida", "Orange,California",
        "Oro Valley,Arizona", "Prescott,Arizona", "Rio Rico,Arizona",
        "Rockland,New York", "Sacramento,California"
    }

    # get the data
    train_df = pd.read_pickle(os.path.expanduser(config['files']['training']))
    val_df = pd.read_pickle(os.path.expanduser(config['files']['validation']))
    test_df = pd.read_pickle(os.path.expanduser(config['files']['testing']))

    train_df, scaler = training_utils.format_data(
        train_df,
        config['data']['data_shape'],
        config['data']['samples_per_city'],
        scaler=MinMaxScaler(),
        fit_scaler=True,
        summer_samples=config['data'].get('summer_samples'),
        winter_samples=config['data'].get('winter_samples'),
        summer_cities=summer_cities,
        winter_cities=winter_cities
    )

    # persist scaler
    with open(os.path.expanduser(f'{model_files_path}/avg_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    val_df = training_utils.format_data(
        val_df,
        config['data']['data_shape'],
        config['data']['samples_per_city'],
        scaler=scaler
    )
    test_df = training_utils.format_data(
        test_df,
        config['data']['data_shape'],
        config['data']['samples_per_city'],
        scaler=scaler
    )

    X_train, y_train = training_utils.split_and_shuffle(train_df)
    X_val, y_val = training_utils.split_and_shuffle(val_df)
    X_test, y_test = training_utils.split_and_shuffle(test_df)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # compile
    opt_name = config['compile']['optimizer']
    lr = config['compile']['learning_rate']
    optimizer_cls = getattr(tf.keras.optimizers, opt_name)
    optimizer = optimizer_cls(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=config['compile']['loss'],
        metrics=[predictions.r2_keras]
    )

    # callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.EarlyStopping(
            patience=15, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        **config['fit'],
        callbacks=callbacks
    )

    # save
    model.save(model_file)
    print('Model saved to ' + model_file)
    save_history_plots(history)
    return


# ------------------------- TEST / PREDICT -------------------------

def test_predictions(
        model_files_path: str,
        config_name: str = "gru_avg_temp_config.json") -> pd.DataFrame:
    """Generate and return predictions from a trained GRU model on the test dataset."""
    cfg_path = os.path.join(os.path.expanduser(model_files_path), config_name)
    with open(cfg_path) as fp:
        config = json.load(fp)

    data_shape = config['data']['data_shape']
    test_path = os.path.expanduser(config['files']['testing'])
    model_file = os.path.expanduser(config['files']['model'])

    test_data = pd.read_pickle(test_path)

    model = tf.keras.models.load_model(
        model_file, custom_objects={'r2_keras': predictions.r2_keras}
    )

    with open(os.path.expanduser(f'{model_files_path}/avg_scaler.pkl'), 'rb') as f:
        loaded_scaler = pickle.load(f)

    results = predictions.gen_preds(
        model, test_data, data_shape, loaded_scaler, fit_scaler=False
    )
    results = pd.DataFrame(
        results,
        columns=['Location', 'Year', 'Month', 'Day', 'MoLS', 'Neural Network']
    )
    return results


# ------------------------- PLOTTING -------------------------

def plot_predictions(results: pd.DataFrame, location_substr: str = 'Avondale') -> None:
    """Plot MoLS vs NN predictions for the subset of rows whose Location contains location_substr."""
    results = results.copy()
    results['Datetime'] = pd.to_datetime(results[['Year', 'Month', 'Day']])
    subset = results[results.Location.astype(
        str).str.contains(location_substr, na=False)]

    if subset.empty:
        print(f"No rows matched location substring: {location_substr}")
        return

    fig, axs = plt.subplots(figsize=(8, 5))  # you can adjust figure size if desired

    axs.plot(subset.Datetime, subset.MoLS, label='MoLS')
    axs.plot(subset.Datetime, subset['Neural Network'], label='Neural Network')

    axs.set_xlabel('Date')
    axs.set_ylabel('Abundance (units of MoLS output)')
    axs.set_title(f'Predictions for {location_substr}')
    axs.legend()

    fig.tight_layout()
    fig.savefig('../output/training/base_testing.png', dpi=300, bbox_inches='tight')
    return


def save_history_plots(history, out_dir="../output/training", prefix="base"):
    """Save training curves from a Keras History object, plus a CSV."""
    os.makedirs(out_dir, exist_ok=True)

    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, hist["loss"], label="train")
    if "val_loss" in hist:
        ax.plot(epochs, hist["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    fig.tight_layout()

    loss_path = os.path.join(out_dir, f"{prefix}_loss.png")
    fig.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Save raw history to CSV for reproducibility ---
    hist_df = pd.DataFrame(hist)
    csv_path = os.path.join(out_dir, f"{prefix}_history.csv")
    hist_df.to_csv(csv_path, index=False)

    print(f"Saved: {loss_path}")
    print(f"Saved: {csv_path}")
    return


def run_all(model_files_path) -> None:
    cfg_name = "gru_avg_temp_config.json"

    # Update config
    update_training_config(os.path.join(
        model_files_path, cfg_name), model_files_path)

    # Train
    train(model_files_path, config_name=cfg_name)

    # Test and plot
    results = test_predictions(model_files_path, config_name=cfg_name)
    print(results.head())
    plot_predictions(results, location_substr="Avondale")


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train/test GRU for Aedes-AI with mean temp input."
    )
    parser.add_argument(
        "--paths",
        default="../fpaths_config.json",
        help="Path to JSON with {weather_data, raw_mols, nn_abundance_predictions, model_files}."
    )
    parser.add_argument(
        "--config-name",
        default="gru_avg_temp_config.json",
        help="Name of the model config file inside model_files_path."
    )
    parser.add_argument("--update-config", action="store_true",
                        help="Update the training config paths and data shape to [90, 3].")
    parser.add_argument("--train", action="store_true",
                        help="Train the model.")
    parser.add_argument("--test", action="store_true",
                        help="Run test predictions and print head().")
    parser.add_argument(
        "--plot",
        metavar="LOCATION_SUBSTR",
        nargs="?",
        const="Avondale",
        help="Plot predictions for locations containing this substring (default: Avondale).")

    args = parser.parse_args()

    _, _, _, model_files_path, _ = gen_utils.load_paths(args.paths)
    cfg_path = os.path.join(os.path.expanduser(model_files_path), args.config_name)

    did_any = False
    if args.update_config:
        update_training_config(cfg_path, model_files_path)
        print(f"Updated config at {cfg_path}")
        did_any = True

    if args.train:
        train(model_files_path, config_name=args.config_name)
        did_any = True

    if args.test or args.plot is not None:
        df = test_predictions(model_files_path, config_name=args.config_name)
        if args.test:
            print(df.head())
        if args.plot is not None:
            plot_predictions(df, location_substr=args.plot)
        did_any = True

    if not did_any:
        run_all(model_files_path)


if __name__ == "__main__":
    main()
