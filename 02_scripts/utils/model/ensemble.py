import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_absolute_error


def weighted_ensemble_many(predictions, errors, use_squared_weights=True, verbose=True):
    """
    Compute a weighted ensemble forecast from multiple model predictions.

    Parameters:
        predictions : list of array-like
            A list of prediction arrays from different models. Each array must have the same shape.
        
        errors : array-like
            A list or array of error metrics (e.g., MAE, RMSE) for each corresponding model.
            Lower errors result in higher weights.
        
        use_squared_weights : bool, default=True
            If True, weights are calculated as the inverse of squared errors (1 / error²).
            If False, weights are calculated as the inverse of errors (1 / error).
        
        verbose : bool, default=True
            If True, prints the normalized weights used in the ensemble.

    Returns:
        final_forecast : ndarray
            The final ensemble prediction as a weighted average across all models.
            Shape is the same as an individual prediction array.
    """
    predictions = [np.array(p) for p in predictions]
    errors = np.array(errors)

    if use_squared_weights:
        weights = 1 / errors**2
    else:
        weights = 1 / errors

    weights /= weights.sum()  
    if verbose:
        print(f"weights: {weights}")

    stacked = np.vstack(predictions)
    final_forecast = np.dot(weights, stacked)
    
    return final_forecast, weights
    

def compare_lists_strict(a, b):
    """
    Compare two lists element-wise for strict equality.

    Parameters:
        a : list
            The first list to compare.
        b : list
            The second list to compare.

    Returns:
        bool
            True if both lists are of the same length and contain exactly the same elements 
            in the same order. False otherwise.
    """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))

    
def find_better_ensemble(test, target_name, forecasts, errors, verbose=True):
    """
    Search for the best weighted ensemble of model forecasts by evaluating all model combinations.

    Parameters:
        test : pandas.DataFrame
            Ground truth data containing the target variable.
    
        target_name : str
            Column name in `test` corresponding to the target variable for evaluation.
    
        forecasts : pandas.DataFrame
            DataFrame with model forecasts. Rows represent samples or target categories,
            columns represent model names.
    
        errors : pandas.DataFrame
            DataFrame with corresponding error metrics (e.g. MAE or RMSE) for each model.
            Must have the same columns and structure as `forecasts`.
    
        verbose : bool, default=True
            If True, prints progress, weights, and MAE values for each ensemble combination.

    Returns:
        best_combo : tuple
            The combination of model names that produced the lowest MAE.
    
        best_mae : float
            The lowest MAE score obtained among all combinations.
    
        best_forecast : numpy.ndarray
            The ensemble forecast corresponding to `best_combo`.

    Raises:
        ValueError
            If `forecasts.columns` and `errors.columns` do not match exactly.
    """
    if not compare_lists_strict(forecasts.columns, errors.columns):
        raise ValueError(f"Column names do not match")

    elements = forecasts.columns
    all_combinations = []
        
    for r in range(1, len(elements)+1):
            all_combinations.extend(combinations(elements, r))
    
    results = {}
    for index, combo in enumerate(all_combinations):
        combo_list = list(combo)
        if verbose:
            print(f"\nModels: {combo_list}")
        ensemble_forecast, ensemble_weights = weighted_ensemble_many(forecasts.loc[target_name, combo_list], 
                                                   errors.loc[target_name, combo_list],
                                                  verbose=verbose)
        ensemble_mae = mean_absolute_error(test[target_name], ensemble_forecast)
        if verbose:
            print(f"MAE: {ensemble_mae}")
        results[combo] = {"mae": ensemble_mae,
                          "forecast": ensemble_forecast,
                          "weights": ensemble_weights}

    if verbose:
        print("\n")
        df = pd.DataFrame([(k, v["mae"]) for k, v in results.items()], columns=['Models', 'MAE'])
        df.sort_values(["MAE"], inplace=True, ignore_index=True)
        print(df)
    
    best_combo, best_item = min(results.items(), key=lambda item: item[1]["mae"])

    if verbose:
        print("\n")
        print(f"The best ensemble {best_combo} with MAE = {best_item["mae"]}\n")

    return (best_combo, best_item["mae"], best_item["forecast"], best_item["weights"])
    