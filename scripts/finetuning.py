"""
Aedes-AI GRU Model with Mean Temperature as Input

This code trains and tests a GRU model that estimates Ae. aegypti abundance from:
- Daily mean temperature (°C)
- Daily precipitation (cm)
- Daily relative humidity (%)

The GRU is trained to reproduce MoLS outputs.
"""

# autopep8: off

import sys, os, json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import gen_utils
from utils import predictions
from utils import format_data_utils

import matplotlib.pyplot as plt

# autopep8: on


def get_finetuning_data(raw_mols_path):
    """_summary_

    Args:
        raw_mols_path (_type_): _description_
    """
    data = pd.read_csv('{}/San_Juan_MoLS.csv'.format(raw_mols_path))
    data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day']])

    train = data[data.Year <= 2015]
    train = train.iloc[180:, :].reset_index(drop=True)

    val = data[data.Year == 2016].reset_index(drop=True)

    test = data[data.Year >= 2017].reset_index(drop=True)

    return train, val, test


def load_model_fils(model_files_path):
    """Load model files from specified directory.

    Args:
        model_files_path (str): Path to the directory containing model files.
    """

    # config
    with open('{}/gru_avg_temp_config.json'.format(model_files_path), 'r') as f:
        config = json.load(f)

    model_fil = config['files'].get('model')
    base_model = tf.keras.models.load_model(
        model_fil, custom_objects={'r2_keras': predictions.r2_keras})

    scaler = pd.read_pickle('{}/avg_scaler.pkl'.format(model_files_path))

    return base_model, config, scaler


def get_finetuning_samples(train, val, test, scaler):
    """Get finetuning samples from training, validation, and test datasets.

    """
    X_train, y_train, locs_train = format_data_utils.format_finetuning_samples(
        train, scaler)

    X_val, y_val, locs_val = format_data_utils.format_finetuning_samples(
        val, scaler)

    X_test, y_test, locs_test = format_data_utils.format_finetuning_samples(
        test, scaler)

    # Shuffle train and val data
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    locs_train = locs_train[permutation]

    permutation = np.random.permutation(X_val.shape[0])
    X_val = X_val[permutation]
    y_val = y_val[permutation]
    locs_val = locs_val[permutation]

    return X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test


def train_finetune_model(base_model, X_train, y_train, X_val, y_val, config):
    """Fine-tune the base model using the provided training and validation data.

    Args:
        base_model: Pre-trained Keras model to be fine-tuned.
        X_train: Training input data.
        y_train: Training target data.
        X_val: Validation input data.
        y_val: Validation target data.
        config: Configuration dictionary containing training parameters.
    """

    finetuned_model = tf.keras.models.clone_model(base_model)
    finetuned_model.set_weights(base_model.get_weights())

    # Freeze all layers except the last two
    for layer in finetuned_model.layers[:-2]:
        layer.trainable = False

    finetuned_model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), loss='mse', metrics=[predictions.r2_keras])

    history_finetune = finetuned_model.fit(X_train, y_train, validation_data=(X_val, y_val), **config['fit'],
                                           callbacks=[tf.keras.callbacks.TensorBoard(), tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])

    return history_finetune, finetuned_model


def plot_finetune_history(history, model_files_path):
    """Plot the training and validation loss over epochs.

    Args:
        history: Keras History object containing training history.
    """
    fig, axs = plt.subplots()
    axs.plot(history.history['loss'], label='train')
    axs.plot(history.history['val_loss'], label='validation')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.legend()
    fig.savefig('{}/gru_avg_temp_finetune_history.png'.format(model_files_path),
                bbox_inches='tight', dpi=300)
    return


def main():
    fils_config = "../fpaths_config.json"
    _, _, _, model_files_path, raw_mols_path = gen_utils.load_paths(
        fils_config)

    base_model, config, scaler = load_model_fils(
        model_files_path)

    train, val, test = get_finetuning_data(raw_mols_path)

    X_train, y_train, locs_train, X_val, y_val, locs_val, X_test, y_test, locs_test = get_finetuning_samples(
        train, val, test, scaler)

    history_finetune, finetuned_model = train_finetune_model(
        base_model, X_train, y_train, X_val, y_val, config)
    plot_finetune_history(history_finetune, model_files_path)

    finetuned_model.save(
        '{}/gru_avg_temp_finetuned.h5'.format(model_files_path))
    print('Model saved to {}/gru_avg_temp_finetuned.h5'.format(model_files_path))
    return


if __name__ == "__main__":
    main()
