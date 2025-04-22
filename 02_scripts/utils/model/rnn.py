import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler


def train_validate_rnn(train, test, y_feature_name, H, verbose=1):
    """
    Trains an LSTM-based RNN on time series data with exogenous features,
    then validates it on a test set and returns inverse-transformed predictions.

    Parameters:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        y_feature_name (str): Target column name.
        H (int): Number of lag steps.
        verbose (int): Verbosity level.

    Returns:
        tuple: (model, inverse-scaled test predictions)
    """
    
    features = [y_feature_name, 'is_holiday', 'year',
       'quarter', 'month', 'day', 'season', 'weekday', 'temp', 'feelslike',
       'humidity', 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth',
       'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover',
       'visibility', 'Clear', 'Freezing Drizzle/Freezing Rain', 'Ice',
       'Overcast', 'Partially cloudy', 'Rain', 'Snow']
    df_train = train[features]
    # scaling
    norm_encoder = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = norm_encoder.fit_transform(df_train.values)
    # timesteps
    X_train = []
    y_train = []
    
    for i in range(H, len(df_train)):
        X_train.append(df_train_scaled[i - H: i, 0])
        y_train.append(df_train_scaled[i, 0])
    
    X_train, y_train =  np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0],
                               X_train.shape[1],
                               1))   
    # exogenous predictors
    X_train_static = np.array(df_train_scaled[H:, 1:29])
    
    # RNN
    input_seq = Input(shape=(H, 1))
    input_static = Input(shape=(28,))
    
    x = LSTM(32, return_sequences=True)(input_seq)
    x = Dropout(0.2)(x)
    
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(256, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    
    x = Concatenate()([x, input_static])
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    
    output = Dense(1, activation="softplus")(x) # avoid negative values
    
    model = Model(inputs=[input_seq, input_static], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='mean_squared_error')
    
    lr_scheduler = ReduceLROnPlateau(monitor="loss",
                                 factor=0.8,
                                 patience=15,
                                 min_lr=1e-6)
    
    # training the model
    history = model.fit([X_train, X_train_static], y_train, epochs=100, batch_size=64, verbose=verbose,
                       callbacks=[lr_scheduler])

    # validation
    df_test = test[features]
    df_test_scaled = norm_encoder.fit_transform(df_test.values)
    # test lags
    data_total = pd.concat((df_train[features], df_test[features]), axis=0)
    inputs = data_total[len(data_total) - len(df_test) - H:].values
    inputs = norm_encoder.transform(inputs)
    
    X_test = []
    
    for i in range(H, H+len(df_test)):
        X_test.append(inputs[i-H:i, 0])
    
    X_test =  np.array(X_test)
    
    X_test_static = np.array(df_test_scaled[:, 1:29])

    # prediction
    pred_y = model.predict([X_test, X_test_static], verbose=verbose)
    full_pred = np.hstack([pred_y, X_test_static])
    original_scaled = norm_encoder.inverse_transform(full_pred)
    pred_y_inv = original_scaled[:, 0] 

    return (model, pred_y_inv)