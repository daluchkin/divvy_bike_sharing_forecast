import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.preprocessing import MinMaxScaler


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
