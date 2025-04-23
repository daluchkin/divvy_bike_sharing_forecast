import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from scipy.special import inv_boxcox

from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

from itertools import combinations
from statsmodels.tsa.stattools import pacf

import xgboost as xgb
from xgboost import XGBRegressor

from utils.model.tools import add_lags, get_transformations, get_best_result, test_transformations, get_best_model

from tqdm.notebook import tqdm

from utils.plotting.viz import plot_trips, plot_feature_importance, distplot, check_residuals, plot_combos


def train_validate_xgb(train, test, features, target_name, addlags, transformation=None, verbose=True):
    """
    Trains and validates an XGBoost regressor using time series cross-validation 
    and a grid search for hyperparameter tuning. Also adds lag features to improve 
    predictive performance.

    Parameters:
    ----------
    train : pandas.DataFrame
        Training dataset containing the target and feature columns.

    test : pandas.DataFrame
        Test dataset used for final validation.

    features : list of str
        List of column names to be used as features.

    target_name : str
        The name of the target variable column.

    transformation (str or None): Optional transformation to apply to `target_name`.
            Supported values:
                - None: no transformation
                - "log": apply np.log
                - "log1p": apply np.log1p
                - "boxcox": apply scipy.stats.boxcox

    Returns:
    -------
    best_model : xgboost.XGBRegressor
        The trained XGBoost model with the best parameters.

    predictors : pandas.Index
        The list of feature column names used in training.

    best_params : dict
        The best set of hyperparameters found during grid search.

    y_pred : numpy.ndarray
        Predicted values for the test set in original scaling.

    lambda : float
        (in case of transformation = "boxcox") Estimated Box-Cox lambda that best normalizes the data.
    """
    # Upd: 1
    if transformation in ["log", "boxcox"] and (np.any(train[target_name] <= 0) or np.any(test[target_name] <= 0)):
        raise ValueError(f"{transformation} transformation requires all {target_name} in training and test sets > 0")

    
    train_transformed = train.copy()
    lambda_bc = None
    if transformation == "log":
        train_transformed[target_name] = np.log(train_transformed[target_name])
    elif transformation == "log1p":
        train_transformed[target_name] = np.log1p(train_transformed[target_name])
    elif transformation == "boxcox":
        train_transformed[target_name], lambda_bc = boxcox(train_transformed[target_name])

    # end Upd 1
    
    train_xgb = pd.DataFrame(train_transformed[features]) # Upd 1
    
    if addlags is not None:
        train_xgb = add_lags(train_xgb, target_name, lags=addlags)

    X_train = train_xgb.drop(columns=target_name)
    y_train = train_xgb[target_name]

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    param_grid = {
        'n_estimators': [50, 100, 200], # 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    cv = TimeSeriesSplit(n_splits=5)
    regressor = XGBRegressor(objective='reg:squarederror', tree_method='hist')
    #regressor = XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist', predictor='gpu_predictor')
    #grid_search = GridSearchCV(regressor, param_grid, scoring=mae_scorer, cv=cv, n_jobs=-1, verbose=int(verbose))
    grid_search = RandomizedSearchCV(regressor, param_grid, n_iter=100, scoring=mae_scorer, cv=cv, n_jobs=-1, random_state=42, verbose=int(verbose))
    grid_search.fit(X_train, y_train)

    if verbose:
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best MAE ({target_name}) on training set: {abs(grid_search.best_score_):.2f}")    

    # Validate on test set

    # Upd 1
    test_transformed = test.copy()

    if transformation == "log":
        test_transformed[target_name] = np.log(test_transformed[target_name])
    elif transformation == "log1p":
        test_transformed[target_name] = np.log1p(test_transformed[target_name])
    elif transformation == "boxcox":
        test_transformed[target_name] = boxcox(test_transformed[target_name], lmbda=lambda_bc)

    # end Upd 1
    
    test_xgb = pd.DataFrame(test_transformed[features])
    
    if addlags is not None:
        test_xgb = pd.concat([train_xgb, test_xgb]) # concat train set to get true lags for test set
        test_xgb = add_lags(test_xgb, target_name, lags=addlags)
        test_xgb = test_xgb[len(train):]
        
    X_test = test_xgb.drop(columns=target_name)
    y_test = test_xgb[target_name]

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Compute and print test metrics
    test_mae = mean_absolute_error(y_test, y_pred)

    if verbose:
        print(f"MAE on test set:  {test_mae:.2f}")

    # Upd 1: invert prediction
    
    if transformation == "log":
        y_pred = np.exp(y_pred)
    elif transformation == "log1p":
        y_pred = np.expm1(y_pred)
    elif transformation == "boxcox":
        y_pred = inv_boxcox(y_pred, lambda_bc)
    
    # end Upd 1
    
    return (best_model,            
            grid_search.best_params_, 
            y_pred,
            lambda_bc,
            addlags)        


def find_best_combo(train, test, target_name):
    """
    Iterates through all valid combinations of predefined feature groups,
    applies applicable transformations to the target variable, trains XGBoost models,
    and returns the results for each combination.

    Workflow:
    1. Defines logical feature groups such as time index, lags, weather, holidays, and weather features.
    2. Generates all possible combinations of 2 or more feature groups.
    3. Computes PACF (Partial Autocorrelation Function) on the target variable
       to identify informative lags (i.e., lags with strong autocorrelation).
    4. For each combination:
        - Constructs the full feature list (including the target)
        - Determines whether to include lag features
        - Selects valid transformations for the target (e.g., None, log, log1p, boxcox)
        - Trains and evaluates models using each transformation
        - Selects the best-performing transformation by lowest test MAE
        - Logs and stores results for analysis

    Parameters:
        train (pd.DataFrame): The training dataset, including target and predictor columns.
        test (pd.DataFrame): The test dataset, used to evaluate model generalization.
        target_name (str): Name of the target variable to be predicted.

    Returns:
        List[Dict]: A list of dictionaries, one per feature group combination, each containing:
                    - index (int): position of the combination in the list
                    - combination (Tuple[str]): names of the feature groups used
                    - models (List[Dict]): results from all transformations tried on this combination
    """
    elements = {"time_index": {"features": ["time_index"], "add_lags": False}, 
                "lags": {"features": [], "add_lags": True}, 
                "timestamps": {"features": ["year", "quarter", "month", "day", "season", "weekday"], "add_lags": False}, 
                "holidays": {"features": ["is_holiday"], "add_lags": False}, 
                "weather": {"features": ["temp", "feelslike",
                                           "humidity", "precip", "precipprob", "precipcover", "snow", "snowdepth",
                                           "windgust", "windspeed", "winddir", "sealevelpressure", "cloudcover",
                                           "visibility", "Clear", "Freezing Drizzle/Freezing Rain", "Ice",
                                           "Overcast", "Partially cloudy", "Rain", "Snow"], "add_lags": False},
               }
    
    all_combinations = []
    
    for r in range(2, len(elements)+1):
        all_combinations.extend(combinations(elements.keys(), r))
    
    train["time_index"] = np.arange(len(train))
    test["time_index"] = np.arange(len(train), len(train) + len(test))
    
    combination_results = []

    pacf_vals = pacf(train[target_name], nlags=31)
    informative_lags = [i for i, val in enumerate(pacf_vals) if abs(val) > 0.2 and i != 0]
    
    for index, combo in tqdm(enumerate(all_combinations), total=len(all_combinations)):
        
        if index == 0:
            tqdm.write("")
            tqdm.write(f"BEST MODELS FOR `{target_name}`")
            tqdm.write('')

            tqdm.write(f"Total combinations: {len(all_combinations)}")
            tqdm.write(f'Informative Lags: {informative_lags} (threshold = 0.2)')
            
            tqdm.write('')
            header = f"| {"N.":<2} | {'Combination':<50} | {"Transformation":<20} | {"Lags":<10} | {"MAE":<10} |"
            tqdm.write('-' * len(header))
            tqdm.write(header)
            tqdm.write('-' * len(header))
            
        features = []
        add_lags = False
        for item in combo:
            features.extend(elements[item]["features"])
            add_lags |= elements[item]["add_lags"]
    
        features = [target_name] + features
        
        transformations = get_transformations(train, test, target_name)
        
        lags = None
        if add_lags:
            lags = informative_lags
            
        results = test_transformations(train, 
                                       test, 
                                       target_name, 
                                       features, 
                                       train_validate_xgb, 
                                       add_lags=lags, 
                                       transformations=transformations, verbose=False)    
        
        best_result = get_best_result(results, target_name)

        trans = best_result["transformation"]
        if trans is not None:
            trans_str = f"{trans} (λ={best_result.get('lambda', 0):.4f})" if trans == "boxcox" else trans
        else:
            trans_str = "-"
        lags_str = 'Yes' if lags else "No"
        tqdm.write(f"| {(index+1):<2} | {', '.join(combo):<50} | {trans_str:<20} | {lags_str:<10} | {best_result["test_mae"]:<10.2f} |")
        tqdm.write('-' * len(header))
        
        combination_results.append({"index": index, "combination": combo, "models": results})

    best_mae, best_models = get_best_model(combination_results, target_name)
    best_combo = list(best_models.keys())[0]
    best_model = list(best_models.values())[0]
    print(f"\nBest MAE on test set: {best_mae:.2f}")
    print(f"Best combinations: {best_combo}")
    print(f"Lags: {best_model['lags']}")
    transformation = best_model['transformation']
    transformation = transformation if transformation != "boxcox" else transformation + f" (λ={best_model.get('lambda', 0):.4f})"
    print(f"Transformation: {transformation}\n")
    
    return combination_results


def xgb_run(train, test, target_name):
    """
    Runs the full modeling pipeline using XGBoost to forecast the target variable.
    Includes exploratory analysis, feature selection, transformation testing,
    model training, evaluation, and visualization.

    Parameters:
        train (pd.DataFrame): Training dataset with features and target.
        test (pd.DataFrame): Test dataset with the same structure.
        target_name (str): Name of the target column to forecast.

    Returns:
        Tuple:
            - best_combo (tuple[str]): Best feature group combination.
            - best_model (dict): Model metadata including test forecast, MAE, transformation, feature importance, etc.
    """
    # analise the target distribution
    print(f"\nStep 1: Analyzing distribution of `{target_name}`")
    distplot(train, [target_name])
    # run searching the best combination of predictors
    print("\nStep 2: Searching for best feature + transformation combo...")
    combo = find_best_combo(train, test, target_name)
    # plot feature importance
    print("\nStep 3: Visualizing feature group performance")
    plot_combos(combo, target_name)
    # get the best model
    print("\nStep 4: Extracting best model based on MAE")
    _, best_models = get_best_model(combo, target_name)
    best_combo = list(best_models.keys())[0]
    best_model = list(best_models.values())[0]
    # Viz the forecast
    print("\nStep 5: Forecast visualization and feature importance")
    _, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_trips(ax[0], train[target_name], test[target_name], 
               forecast=pd.DataFrame(best_model["test_forecast"], index=test.index),
               title=f"`{target_name}` forecast")
    
    plot_feature_importance(ax[1], best_model["model"])
    
    plt.suptitle(f"XGBoost (MAE={best_model["test_mae"]:.2f})", fontsize=18)
    plt.tight_layout()
    plt.show() 
    # analise the residuals
    print("\nStep 6: Residual analysis")
    check_residuals(best_model["test_forecast"], test[target_name], target_name)

    return best_combo, best_model