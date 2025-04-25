import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


def add_lags(df, y_col_name, lags=[1, 7, 12, 30]):
    """
    Adds lag features to a time series column in a DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the time series data.

    y_col_name : str
        The name of the target column to generate lags for.

    lags : list of int, optional (default = [1, 7, 12, 30])
        A list of lag intervals (in number of time steps) to create lag features for.

    Returns:
    -------
    df : pandas.DataFrame
        The DataFrame with new lag columns added, named as 'lag_<n>'.
    """
    
    for lag in lags:
        # Create a new column for each lag, shifting the target column by 'lag' steps
        df[f"lag_{lag}"] = df[y_col_name].shift(lag)

    return df


def make_features(target_name):
    """
    Returns a list of feature column names for a time series forecasting dataset,
    including the specified target variable and various timestamps, holidays and weather indicators.

    Parameters:
        target_name (str): The name of the target variable to include in the feature list.

    Returns:
        list: A list of column names representing features used for model training.
    """
    
    cols = [target_name, "year", "quarter", "month", "day", "season", "weekday", "is_holiday", 'temp', 'feelslike',
       'humidity', 'precip', 'precipprob', 'precipcover', 'snow', 'snowdepth',
       'windgust', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover',
       'visibility', 'Clear', 'Freezing Drizzle/Freezing Rain', 'Ice',
       'Overcast', 'Partially cloudy', 'Rain', 'Snow']
    
    return cols


def test_transformations(train, test, target_name, features, train_validate_func, add_lags, 
                         transformations=[None, "log", "log1p", "boxcox"], verbose=True):
    """
    Evaluates the impact of multiple target transformations on model performance.

    For each transformation type (e.g., None, log, log1p, boxcox), this function:
        - Transforms the target variable accordingly.
        - Calls a user-defined training and validation function to fit the model.
        - Computes MAE on the test set using inverse-transformed predictions.
        - Stores the trained model, forecast, MAE, and related metadata.

    Parameters:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
        target_name (str): Name of the target variable to be predicted.
        features (list[str]): List of feature column names to be used for training.
        train_validate_func (function): Custom function to train and validate a model.
            Must accept arguments:
                (train_set, test_set, features, target_name, addlags, transformation, verbose)
            and return:
                (model, model_params, forecast (np.ndarray or pd.Series), lambda (float or None), lags)
        add_lags (list[int] or None): Optional list of lag indices to be added as features.
        transformations (list): List of transformation types to test.
            Supported values: [None, "log", "log1p", "boxcox"]
        verbose (bool): Whether to print transformation progress and training logs.

    Returns:
        dict: A dictionary with the `target_name` as the key and values as another dict,
              where each inner key is an integer index and value is a dict containing:
                - 'model': Trained model object
                - 'params': Training metadata or hyperparameters
                - 'transformation': Name of the transformation used
                - 'test_mae': MAE on the test set (after inverse transform)
                - 'test_forecast': Forecast on the test set in original scale
                - 'lags': List of lag features used (or None)
                - 'lambda': Box-Cox λ parameter (only present if transformation == "boxcox")
    """
    
    results = {}
    for  index, transformation in enumerate(transformations):
        if verbose:
            print(f"Transformation: {transformation}")
            
        xgb_model, xgb_model_params, xgb_forecast, xgb_lmbda, xgb_lags = train_validate_func(train, 
                                                                       test, 
                                                                       features, 
                                                                       target_name, 
                                                                       addlags=add_lags,
                                                                       transformation=transformation, 
                                                                       verbose=verbose)

        xgb_mae = mean_absolute_error(test[target_name], xgb_forecast)

        result = {
                    "model": xgb_model,
                    "params": xgb_model_params,
                    "transformation": transformation,
                    "test_mae": xgb_mae,
                    "test_forecast": xgb_forecast,
                    "lags": xgb_lags
                }
        if transformation == "boxcox":
            result["lambda"] = xgb_lmbda
            
        results[index] = result
        
    return {target_name: results}


def get_best_result(results, target_name):
    """
    Returns the result with the lowest test MAE for the specified target.

    Parameters:
        results (dict): Dictionary of results structured as 
                        {target_name: {index: {
                            "model": @model,
                            "params": @params,
                            "transformation": @transformation,
                            "test_mae": @mae,
                            "test_forecast": @forecast,
                            "lags": @lags,
                            "lambda": @lambda (is case of transformation = "boxcox")
                        }}}.
        target_name (str): Name of the target variable to search in.

    Returns:
        dict: The result dictionary with the minimal "test_mae" value.
    """
    
    return min(results[target_name].values(), key=lambda d: d["test_mae"])


def get_transformations(train, test, target_name):
    """
    Returns a list of applicable target transformations based on the presence of zeros in the data.

    If the target column in either the training or test set contains zero values,
    only transformations that can safely handle zeros are returned: None and 'log1p'.
    
    Otherwise, a full set of transformations is returned, including:
    - None: no transformation
    - 'log': natural logarithm (requires all values > 0)
    - 'log1p': log(1 + x), safe for zero values
    - 'boxcox': Box-Cox power transformation (requires all values > 0)

    Parameters:
        train (pd.DataFrame): Training dataset containing the target column.
        test (pd.DataFrame): Test dataset containing the target column.
        target_name (str): Name of the target column to be transformed.

    Returns:
        List[str or None]: List of transformation names that are safe to apply.
    """
    if (train[target_name] == 0).any() or (test[target_name] == 0).any():
        return [None, "log1p"]
    return [None, "log", "log1p", "boxcox"]


def get_best_model(combos, target_name):
    """
    Identifies the best-performing model(s) across all feature group combinations based on the lowest test MAE.

    Parameters:
        combos (List[Dict]): List of model evaluation results, each containing:
                             - 'combination' (tuple): feature group names
                             - 'models' (list): results of model runs with various transformations
        target_name (str): The name of the target variable used to extract the best result from each combination.

    Returns:
        Tuple[float, Dict[Tuple[str], Dict]]:
            - The lowest test MAE value (`min_val`)
            - A dictionary mapping feature combinations (tuples of group names)
              to their corresponding best result dictionaries (with transformation, MAE, etc.).
    """
    best_models = {item["combination"]: get_best_result(item["models"], target_name) for item in combos}
    min_val = min(model["test_mae"] for model in best_models.values())
    results = {k: v for k, v in best_models.items() if v["test_mae"] == min_val}
    min_combo_len = min(len(key) for key in results.keys())
    results = {k: v for k, v in results.items() if len(k) == min_combo_len}

    return min_val, results


