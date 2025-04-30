import warnings
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima import auto_arima

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.metrics import mean_absolute_error


def stationary_test(ts):
    """
    Performs stationarity tests and visualizes autocorrelation for a given time series.

    Tests performed:
    - Augmented Dickey–Fuller (ADF) test
    - Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test

    Parameters:
        ts : pd.Series or np.ndarray
            The time series data to be tested for stationarity.

    Behavior:
        - Prints the ADF statistic and p-value.
            - If p-value < 0.05, the series is considered stationary by the ADF test.
        - Prints the KPSS statistic and p-value.
            - If p-value < 0.05, the series is considered non-stationary by the KPSS test.
        - Plots:
            - Autocorrelation function (ACF)
            - Partial autocorrelation function (PACF)
    """
    print("-- Dickey–Fuller test --")
    result = adfuller(ts, maxlag=None, regression='c', autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    
    if result[1] < 0.05:
        print("Time series is stationary")
    else:
        print("Time series is non stationary")
        
    print("-- KPSS test --")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result = kpss(ts, regression='c')
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        
        if result[1] < 0.05:
            print("Time series is non stationary")
        else:
            print("Time series is stationary")
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(ts, ax=ax[0])
    plot_pacf(ts, ax=ax[1])

    
def make_harmonic_features(t, periods, harmonics_per_periods):
    """
    Generates harmonic (Fourier) features for time series modeling based on specified periods and harmonic orders.

    Parameters
        t : np.ndarray or list
            Time indices (e.g., a sequence of integers representing time steps).
    
        periods : list of floats
            List of periods for which to generate harmonic terms (e.g., [7, 30.44, 365.25] for weekly, monthly, and yearly seasonality).
    
        harmonics_per_periods : list of ints
            List specifying the number of harmonics (sine and cosine terms) to generate for each period.
            Each element corresponds to a period in the `periods` list.

    Returns
        features : np.ndarray
            A 2D array where each column is a sine or cosine harmonic feature.
            The shape is (len(t), total number of harmonic features).
    """
    features = []
    i = 0
    for P in periods:
        for k in range(1, harmonics_per_periods[i] + 1):
            features.append(np.sin(2 * np.pi * k * t / P))
            features.append(np.cos(2 * np.pi * k * t / P))
        i += 1
    return np.column_stack(features)


def safe_boxcox(series, lmbda=None, shift=None):
    """
    Applies a Box-Cox transformation to a time series, safely handling zero and negative values.

    Parameters
        series : pd.Series or np.ndarray
            Input time series data to be transformed.
        
        lmbda : float, optional (default=None)
            The lambda parameter for the Box-Cox transformation.
            If None, the optimal lambda will be automatically estimated.
    
        shift : float, optional (default=None)
            A shift value to adjust the series so all values are positive.
            If None, the shift is computed as (1 - minimum value) if the minimum is less than or equal to zero.

    Returns
        transformed : np.ndarray
            The transformed series after applying the Box-Cox transformation.
    
        lmbda : float
            The lambda value used for the transformation.
    
        shift_tr : float
            The applied shift value to ensure all inputs were positive before transformation.
    """
    if shift is None:
        min_val = series.min()
        shift_tr = 1 - min_val if min_val <= 0 else 0
    else:
        shift_tr = shift

    if lmbda is None:
        transformed, lmbda = boxcox(series + shift_tr)
    else:
        transformed = boxcox(series + shift_tr, lmbda=lmbda)

    return transformed, lmbda, shift_tr


def train_validate_harmonic_arima(train, test, feature_name, 
                                  harmonic_periods, harmonics_per_periods, exog_features, 
                                  forecast_type="direct", steps_ahead=14, d=None):
    """
    Train and validate a non-seasonal ARIMA model with harmonic features and exogenous variables.

    This function supports two forecast types:
    - "direct": trains on the entire training set and forecasts all future steps at once.
    - "iterative": performs recursive forecasting, updating the model step-by-step using actual values.

    Parameters:
        train : pd.DataFrame
            Training dataset with time series values and exogenous features.
        test : pd.DataFrame
            Test dataset used for forecasting.
        feature_name : str
            Name of the target feature to forecast.
        harmonic_periods : list of int
            Periods for which harmonics will be generated (e.g., [7, 365] for weekly/yearly).
        harmonics_per_periods : list of int
            Number of harmonics to generate for each period.
        exog_features : list of str
            List of column names to be used as exogenous variables.
        forecast_type : str, optional
            "direct" or "iterative". Determines forecast strategy. Default is "direct".
        steps_ahead : int, optional
            Number of steps ahead for forecasting (used only in "iterative" mode). Default is 14.
        d : int or None, optional
            Degree of differencing to force in ARIMA model. If None, selected automatically.

    Returns:
        model : ARIMA model
            The trained ARIMA model.
        harmonic_forecast : np.ndarray
            The forecasted values (inverse Box-Cox transformed).
    """
    print(f"\n--- {feature_name} ---")
    # box-cox transformation
    train_transformed, train_lambda, shift = safe_boxcox(train[feature_name])
    # farmonic features
    t = np.arange(len(train))
    train_harmonics = make_harmonic_features(t, 
                                             periods=harmonic_periods, 
                                             harmonics_per_periods=harmonics_per_periods)
    # searching the model
    print("Searching the best model...")
    match forecast_type:
        case "iterative":
            train_data = train_transformed[:-steps_ahead]
            X = np.column_stack([train_harmonics[:-steps_ahead], train.iloc[:-steps_ahead][exog_features].values])
        case "direct":
            train_data = train_transformed
            X = np.column_stack([train_harmonics, train[exog_features].values])
        case _:
            raise ValueError(f"Unknown value for 'forecast_type': {forecast_type}")
            
    model = auto_arima(train_data, X=X, 
                            seasonal=False, 
                            stepwise=False,
                            suppress_warnings=True,
                            max_p=6, max_q=6,
                            max_order=None,
                            information_criterion="bic",
                            test="adf", d=d, 
                            error_action="ignore",
                            trace=False,
                            out_of_sample_size=7,
                            scoring="mae")
    print(f"The best model for {feature_name} has order {model.order}")
    
    sarima_test_preds = []

    match forecast_type:
        case "iterative":
            bar_length = 50
            # warm up the model
            print("Warming up the model...")
            for i in range(len(train_transformed) - steps_ahead, len(train_transformed)):
                exog = np.hstack([train_harmonics[i], train.iloc[i][exog_features].values]).reshape(1, -1)
                pred = model.predict(n_periods=1, X=exog)[0]
                #sarima_test_preds.append(pred) # remove it
                model.update(train_transformed[i], X=exog) # update the model
            
            # prediction
            print("Iterative Forecasting...")
            test_transformed, _, _ = safe_boxcox(test[feature_name], train_lambda, shift)
            for i in range(len(train_transformed), len(train_transformed) + len(test)):
                test_harmonics = make_harmonic_features(np.arange(i, i + 1), 
                                                         periods=harmonic_periods, 
                                                        harmonics_per_periods=harmonics_per_periods)
                exog = np.hstack([test_harmonics, test.iloc[i-len(train_transformed)][exog_features].values.reshape(1, -1)])
                y_pred = model.predict(1, X=exog)[0]
                sarima_test_preds.append(y_pred)
                model.update(test_transformed[i-len(train_transformed)], X=exog)
        
                percent = (i - len(train_transformed) + 1) / len(test)
                bar = "#" * int(bar_length * percent) + "-" * (bar_length - int(bar_length * percent))
                sys.stdout.write(f"\r[{bar}] {int(percent * 100)}%")
                sys.stdout.flush()
            print("\n")
        case "direct":
            print("Direct Forecasting...")
            test_harmonic = make_harmonic_features(np.arange(len(train), len(train) + len(test)), 
                                       periods=harmonic_periods, 
                                       harmonics_per_periods=harmonics_per_periods)
            X_exog = np.column_stack([test_harmonic, test[exog_features].values])
            sarima_test_preds = model.predict(n_periods=len(test), X=X_exog)
        case _:
            raise ValueError(f"Unknown value for 'forecast_type': {forecast_type}")
            
    print("Boxcox Invertion...")
    harmonic_forecast = inv_boxcox(sarima_test_preds, train_lambda) - shift

    return (model, harmonic_forecast)


def run_harimax(train, test, target_name, features, harmonic_periods, harmonics_per_period):
    """
    Run harmonic ARIMA (HARIMAX) forecasts with different configurations and compare MAE.

    This function performs four different training and forecasting procedures using
    a harmonic-enhanced ARIMA model with exogenous features (HARIMAX):
    1. Direct forecast with automatic differencing.
    2. Iterative forecast with automatic differencing.
    3. Direct forecast with fixed differencing (d=1).
    4. Iterative forecast with fixed differencing (d=1).

    For each configuration, the model is trained on the training set and used to predict
    the target variable over the test period. The performance is evaluated using mean absolute error (MAE).

    Parameters:
        train : pd.DataFrame
            The training dataset, including the target and exogenous variables.
        test : pd.DataFrame
            The test dataset, used for forecasting evaluation.
        target_name : str
            The name of the target variable to forecast.
        features : list of str
            List of column names to use as exogenous variables.
        harmonic_periods : list of int
            List of base periods (e.g., [7, 365]) for generating harmonic features.
        harmonics_per_period : list of int
            Number of harmonics to generate per each base period.

    Returns:
        results : list of dict
            List of dictionaries, each containing:
            - 'model_order': ARIMA (p,d,q) order used,
            - 'd': degree of differencing used (None or 1),
            - 'forecast_type': "direct" or "iterative",
            - 'mae': mean absolute error for this forecast,
            - 'forecast': forecasted values as a NumPy array.
    """
    # direct forecast w/auto d
    model1, harimax_pred1 = train_validate_harmonic_arima(train, test, 
                                                          harmonic_periods=harmonic_periods,
                                                          harmonics_per_periods=harmonics_per_period, 
                                                          exog_features=features, 
                                                          forecast_type="direct", 
                                                          feature_name=target_name)
    harimax_mae1 = mean_absolute_error(test[target_name], harimax_pred1)
    print(f"MAE of direct forecast of {target_name}: {harimax_mae1:.2f}")
    print("\n")
    
    # iterative forecast w/auto d
    model2, harimax_pred2 = train_validate_harmonic_arima(train, test, 
                                                          harmonic_periods=harmonic_periods,
                                                          harmonics_per_periods=harmonics_per_period, 
                                                          exog_features=features, 
                                                          forecast_type="iterative", 
                                                          feature_name=target_name)
    harimax_mae2 = mean_absolute_error(test[target_name], harimax_pred2)
    print(f"MAE of iterative forecast of {target_name}: {harimax_mae2:.2f}")
    print("\n")
    
    # direct forecast w/ d = 1
    model3, harimax_pred3 = train_validate_harmonic_arima(train, test, 
                                                          harmonic_periods=harmonic_periods,
                                                          harmonics_per_periods=harmonics_per_period, 
                                                          exog_features=features, 
                                                          forecast_type="direct", 
                                                          feature_name=target_name, d=1)
    harimax_mae3 = mean_absolute_error(test[target_name], harimax_pred3)
    print(f"MAE of direct forecast w/ differencing of {target_name}: {harimax_mae3:.2f}")
    print("\n")
    
    # iterative forecast w/ d = 1
    model4, harimax_pred4 = train_validate_harmonic_arima(train, test, 
                                                          harmonic_periods=harmonic_periods,
                                                          harmonics_per_periods=harmonics_per_period, 
                                                          exog_features=features, 
                                                          forecast_type="iterative", 
                                                          feature_name=target_name, d=1)
    harimax_mae4 = mean_absolute_error(test[target_name], harimax_pred4)
    print(f"MAE of iterative forecast w/ differencing of {target_name}: {harimax_mae4:.2f}")

    results = [
        {"model_order": model1.order, "d": None, "forecast_type": "direct", "mae": harimax_mae1, "forecast": harimax_pred1},
        {"model_order": model2.order, "d": None, "forecast_type": "iterative", "mae": harimax_mae2, "forecast": harimax_pred2},
        {"model_order": model3.order, "d": 1, "forecast_type": "direct", "mae": harimax_mae3, "forecast": harimax_pred3},
        {"model_order": model4.order, "d": 1, "forecast_type": "iterative", "mae": harimax_mae4, "forecast": harimax_pred4},
    ]
    
    return results


def get_best_model(models_info):
    """
    Selects the model entry with the lowest MAE from a list of model results.

    Parameters:
        models_info : list of dicts
            Each dictionary should contain at least the keys: 'mae', 'model_order', 'd', 'forecast_type', 'forecast'.

    Returns:
        best_model_info : dict
            The dictionary corresponding to the model with the best (lowest) MAE.
    """
    if not models_info:
        raise ValueError("The models_info list is empty.")
    
    best_model_info = min(models_info, key=lambda x: x['mae'])
    return best_model_info