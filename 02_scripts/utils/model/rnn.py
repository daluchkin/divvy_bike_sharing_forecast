import itertools

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tqdm.notebook import tqdm

from utils.model.seed import set_seed


def train_validate_rnn(train, test, y_feature_name, H, verbose=1):
    """
    Trains an LSTM-based RNN on time series data with exogenous features,
    performs validation using a time-based split, and returns inverse-transformed predictions.

    Parameters:
        train (pd.DataFrame): Training dataset containing the target and exogenous features.
        test (pd.DataFrame): Test dataset with the same structure as the training set.
        y_feature_name (str): Name of the target variable to be predicted.
        H (int): Number of time steps (lags) used as input sequence length.
        verbose (int): Verbosity level for model training and prediction.

    Returns:
        tuple:
            history (tf.keras.callbacks.History): Training history object.
            model (tf.keras.Model): Trained Keras model.
            features (list): List of features used in training and prediction.
            norm_encoder (MinMaxScaler): Scaler fitted on training data for inverse transformation.
            pred_y_inv (np.ndarray): Inverse-scaled predictions on the test set.
    """

    # Define the feature set: target + exogenous variables
    features = [y_feature_name, 'is_holiday', 'year',
       'quarter', 'month', 'day', 'season', 'weekday', 'temp', 'feelslike',
       'humidity', 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth',
       'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover',
       'visibility', 'Clear', 'Freezing Drizzle/Freezing Rain', 'Ice',
       'Overcast', 'Partially cloudy', 'Rain', 'Snow']

    # Extract training data and scale to [0, 1]
    df_train = train[features]
    norm_encoder = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = norm_encoder.fit_transform(df_train.values)

    # Create sequences of lagged target values for LSTM input
    X_train = []
    y_train = []
    for i in range(H, len(df_train)):
        X_train.append(df_train_scaled[i - H: i, 0])  # lagged y values
        y_train.append(df_train_scaled[i, 0])         # next y value

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # shape: (samples, time steps, 1 feature)

    # Static features (exogenous variables) from the same period
    X_train_static = np.array(df_train_scaled[H:, 1:29])  # skip target (index 0)

    # Define RNN model
    input_seq = Input(shape=(H, 1))         # time-dependent input (lags)
    input_static = Input(shape=(28,))       # time-independent exogenous input

    x = LSTM(64, return_sequences=True)(input_seq)
    x = Dropout(0.15)(x)

    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.25)(x)

    x = Concatenate()([x, input_static])    # merge sequence and static inputs
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation="softplus")(x)  # softplus ensures non-negative output

    model = Model(inputs=[input_seq, input_static], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='mean_squared_error')

    # Learning rate scheduling and early stopping
    lr_scheduler = ReduceLROnPlateau(monitor="loss", factor=0.8, patience=15, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with 10% of the data used as validation set (last part of training set)
    history = model.fit([X_train, X_train_static], y_train,
                        epochs=100, batch_size=64,
                        verbose=verbose, callbacks=[early_stop, lr_scheduler])

    # -------- Prediction Phase -------- #

    # Combine train and test to create test lags
    df_test = test[features]
    data_total = pd.concat((df_train, df_test), axis=0)
    inputs = data_total[len(data_total) - len(df_test) - H:].values
    inputs = norm_encoder.transform(inputs)

    # Build lagged input sequences for test set
    X_test = []
    for i in range(H, H + len(df_test)):
        X_test.append(inputs[i - H:i, 0])  # target only

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    X_test_static = np.array(inputs[H:, 1:29])  # corresponding exogenous features

    # Predict and inverse transform to original scale
    pred_y = model.predict([X_test, X_test_static], verbose=verbose)
    full_pred = np.hstack([pred_y, X_test_static])  # concatenate for inverse scaling
    original_scaled = norm_encoder.inverse_transform(full_pred)
    pred_y_inv = original_scaled[:, 0]  # extract only target column

    return (history, model, features, norm_encoder, pred_y_inv)


def test_rnn_window_size(train, test, target_name, window_sizes, seeds):
    """"
    Evaluates the performance of an RNN/LSTM model across multiple window sizes and random seeds,
    and selects the best-performing window size based on average MAE.

    Parameters:
        train (pd.DataFrame): Training dataset including the target and all necessary features.
        test (pd.DataFrame): Test dataset with the same structure as the training set.
        target_name (str): Name of the target variable to forecast.
        window_sizes (list of int): List of window sizes (number of lag steps) to evaluate.
        seeds (list of int): List of random seeds to assess robustness and stability.

    Returns:
        tuple:
            best_window_size (int): The window size with the lowest average MAE.
            best_preds (list of np.ndarray): List of predictions (one per seed) for the best window size.
            fun_out (list of dict): Detailed results for each window size, containing:
                - 'window_size': int, the window size tested
                - 'mae_mean': float, mean MAE across seeds
                - 'mae_std': float, standard deviation of MAE
                - 'forecast_mean': np.ndarray, average forecast across seeds
    """
    rnn_forecasts = {ws: {"mae": [], "y_pred": []} for ws in window_sizes}
    param_grid = list(itertools.product(seeds, window_sizes))
    i = 0
    for seed, window_size in tqdm(param_grid):

        if i == 0:
            tqdm.write("")
            tqdm.write(f"BEST RNN/LSTM WINDOW SIZE FOR `{target_name}`")
            tqdm.write('')
            header = f"| {"N.":<2} | {'Random Seed':<20} | {"Window Size":<20} | {"MAE":<10} |"
            tqdm.write('-' * len(header))
            tqdm.write(header)
            tqdm.write('-' * len(header))
        
        set_seed(seed, verbose=False)
        _, _, _, _, rnn_pred_y = train_validate_rnn(train, test, target_name, H=window_size, verbose=0)
        mae = mean_absolute_error(test[target_name], rnn_pred_y)
        rnn_forecasts[window_size]["mae"].append(mae)
        rnn_forecasts[window_size]["y_pred"].append(rnn_pred_y)

        tqdm.write(f"| {(i+1):<2} | {seed:<20} | {window_size:<20} | {mae:<10.2f} |")
        tqdm.write('-' * len(header))
        
        i += 1

    print("\nResults:\n")
    header = f"   | {"Window Size":<20} | {"Mean MAE":<30} |"
    tqdm.write('-' * len(header))
    tqdm.write(header)
    tqdm.write('-' * len(header))

    fun_out = []
    
    for window_size, item in rnn_forecasts.items():
        mean_mae = np.mean(item["mae"])
        mae_std = np.std(item["mae"])
        mean_y_pred = np.mean(item["y_pred"], axis=0)
        
        fun_out.append({"window_size": window_size,
                        "mae_mean": mean_mae,
                        "mae_std": mae_std,
                        "forecast_mean": mean_y_pred})

    best_window_size = min(fun_out, key=lambda x: x["mae_mean"])["window_size"]

    for item in sorted(fun_out, key=lambda x: x["mae_mean"]):
        mark = " "
        if item["window_size"] == best_window_size:
            mark = "*"
        mae = f"{item["mae_mean"]:.2f}±{item["mae_std"]:.2f}"
        print(f" {mark} | {item["window_size"]:<20} | {mae:<30} |")
        print("-" * len(header))

    print(f"\nBest window size for {target_name}: {best_window_size}\n")
    best_preds = rnn_forecasts[best_window_size]["y_pred"]

    return best_window_size, best_preds, fun_out
