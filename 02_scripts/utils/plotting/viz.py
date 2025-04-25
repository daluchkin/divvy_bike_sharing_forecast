from utils.plotting.notebook_setup import TRAIN_COLOR, TEST_COLOR, FORECAST_COLOR
from utils.model.tools import get_best_result

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

def plot_trips(ax, train, test, forecast=None, title=""):
    """
    Plots bike trip time series for train, test, and forecast.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes for plotting.
        train (pd.Series): Training data.
        test (pd.Series): Test data.
        forecast (pd.Series, optional): Forecast data to overlay, if provided.
        title (str): Plot title.

    Returns:
        matplotlib.axes.Axes: The axes with plotted data.
    """
    ax.plot(train, lw=1, color=TRAIN_COLOR, label="Train Set")
    ax.plot(test, lw=1, color=TEST_COLOR, label="Test Set")
    if forecast is not None:
        ax.plot(forecast, lw=1, linestyle='--', color=FORECAST_COLOR, label="Forecast")

    ax.set_title(f"{title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of bike trips")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend()
    ax.grid(linestyle='--', color="lightgray")

    return ax


def plot_feature_importance(ax, xgb_model):
    """
    Plots normalized feature importance (by 'gain') from an XGBoost model.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the feature importances.

    xgb_model : xgboost.XGBModel
        A trained XGBoost model from which to extract feature importances.
    """
    # Get raw feature importance by 'gain' (how much each feature improves the model)
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    
    # Sum total gain to normalize values
    total_gain = sum(importance.values())
    
    # Normalize gain so the values sum to 1 (proportional importance)
    normalized_gain = {k: v / total_gain for k, v in importance.items()}
    
    # Sort features by their normalized importance in descending order
    sorted_importance = dict(sorted(normalized_gain.items(), key=lambda item: item[1], reverse=True))
    
    ax.bar(sorted_importance.keys(), sorted_importance.values(), color=TRAIN_COLOR)
    
    ax.set_title("XGBoost: Proportional feature impact (sum = 1)")
    ax.set_ylabel("Normalized Gain")
    ax.set_xlabel("Features")
    
    ax.set_xticklabels(labels=sorted_importance, rotation=90)


def distplot(df, columns, by=None, plots='all'):
    """
        Builds distribution plot (histogram, box plot, violine plot)

        Args:
            df (pandas.DataFrame): The dataset
            columns ([str]): The list of column names for which the dist plots will be built
            by ([str]): The column names to group the plots by
            plots (str: "h|b|v|all"): Determines which plots will be built: 
                                      h - histogram, b - boxplot, v - violine, all - all plots.
                                      The combinations are possible.

        Usage:
            # Build all distribution plots (histogram, boxplot, violine plot)
            distplot(df, ['distance_km', 'duration_min'])

            # Build histogram plot only by rider_type
            distplot(df, ['distance_km', 'duration_min'], by='rider_type', plots='h')

            # Build histogram and box plots by bike_type
            distplot(df, ['distance_km', 'duration_min'], by='bike_type', plots='hb')
    """
    
    for var_name in columns:
        print(f'\n[`{var_name}`]')
        if by is not None:
            print(f'Skewness:\n{np.round(df.groupby(by, observed=True)[var_name].skew(), 2)}')
        else:
            print(f'Skewness: {round(df[var_name].skew(), 2)}')
        print("""
        Note:
        1. Approximately symmetric: -0.5 < skewness < 0.5
        2. Moderately skewed:
                              -1 < skewness < -0.5 - left skewed, 
                             0.5 < skewness < 1    - right skewed
        3. Highly skewed: 
                            skewness < -1  - left skewed,
                            1 < skewness   - right skewed
        """)
        
        if by is not None:
            print(f'Kurtosis:\n{np.round(df.groupby(by, observed=True)[var_name].apply(pd.Series.kurt), 2)}')
        else:
            print(f'Kurtosis: {round(df[var_name].kurt(), 2)}')
        print("""
        Note:
        1. Mesokurtic: kurtosis = 3
           A normal distribution.
        2. Leptokurtic: kurtosis > 3
           A distribution has fatter tails and a sharper peak than the normal distribution.
        3. Platykurtic: kurtosis < 3
           A distribution has thinner tails and a flatter peak than the normal distribution.
    
        Excess Kurtosis: Kurtosis - 3
        """)
        
        plots_list = list('hbv' if plots == 'all' else plots)
        plots_count = len(plots_list)
        plt.figure(figsize = (6*plots_count, 6))
        bins = int(np.ceil(np.sqrt(df.shape[0])))
        for index, plot in zip(range(1, len(plots)+1), plots_list):
            if plot == 'h':
                plt.subplot(1, plots_count, index)
                sns.histplot(data=df, x=var_name, kde=True, bins=bins, hue=by)
                plt.ylabel('Frequence')
                plt.xlabel(var_name)
                plt.title(f'Distribution of {var_name}')
            if  plot == 'b':
                plt.subplot(1, plots_count, index)
                sns.boxplot(data=df, y=var_name, hue=by)
                plt.title(f'Box plot of {var_name}')
            if  plot == 'v':
                plt.subplot(1, plots_count, index)
                sns.violinplot(data=df, y=var_name, hue=by)
                plt.title(f'Violin plot of {var_name}')
            
        plt.show()


def check_residuals(test_y, predictions, model_name):
    """
    Performs a full residual diagnostics analysis for a regression or forecasting model.

    This function computes residuals (test_y - predictions) and visualizes them through:
        - Time series plot of residuals
        - Histogram with KDE
        - ACF and PACF plots (autocorrelation analysis)
        - Q-Q plot for normality check
        - Scatter plot: residuals vs. predictions

    It also performs statistical tests:
        - Ljung–Box test for autocorrelation
        - Augmented Dickey–Fuller (ADF) test for stationarity
        - KPSS test for confirming or rejecting stationarity

    Parameters:
        test_y (array-like): True values from the test set.
        predictions (array-like): Model predictions for the test set.
        model_name (str): Name of the model (for plot titles).

    Returns:
        None. Displays diagnostic plots and prints statistical test results.
    """   
    
    residuals = test_y - predictions
    
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    
    ax[0, 0].plot(residuals)
    ax[0, 0].set_title(f"Residuals of the model")
    ax[0, 0].axhline(0, linestyle='--', color='black')
    
    
    sns.histplot(residuals, kde=True, ax=ax[0, 1])
    ax[0, 1].set_title("Residuals distribution")
    
    sm.graphics.tsa.plot_acf(residuals, lags=30, ax=ax[1, 0])
    sm.graphics.tsa.plot_pacf(residuals, lags=30, ax=ax[1, 1])
    ax[1, 0].set_title("ACF of residuals")
    ax[1, 1].set_title("PACF of residuals")

    sm.qqplot(residuals, line='s', ax=ax[2, 0])
    ax[2, 0].set_title("Q-Q plot residuals")

    sns.regplot(ax=ax[2, 1], x=predictions, y=residuals, lowess=False, ci=95,
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

    ax[2, 1].axhline(0, color='gray', linestyle='--')
    ax[2, 1].set_xlabel("Predictions")
    ax[2, 1].set_ylabel("Residuals")
    ax[2, 1].set_title("Residuals vs Predictions")

    plt.suptitle(f"Residuals Analysis of {model_name}")
    plt.tight_layout()
    plt.show()
    
    print("\n--- Ljung-Box Test ---")

    results = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(results)
    pval = results["lb_pvalue"]
    if all(pval > 0.05):
        print("Residuals do not show significant autocorrelation")
    else:
        print("Residuals are autocorrelated")

    print("\n-- Dickey–Fuller test --")
    result = adfuller(residuals, maxlag=None, regression='c', autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    
    if result[1] < 0.05:
        print("Time series is stationary")
    else:
        print("Time series is non stationary")
        
    print("\n-- KPSS test --")
    result = kpss(residuals, regression='ct')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    
    if result[1] < 0.05:
        print("Time series is non stationary")
    else:
        print("Time series is stationary")


def plot_combos(combo, target_name):
    """
    Creates a horizontal bar plot showing the test MAE for each feature group combination,
    with a visual distinction between combinations that use lag features and those that don't.

    Parameters:
        combo (List[Dict]): List of results produced by `find_best_combo`, where each item contains:
                            - 'combination' (tuple): names of feature groups
                            - 'models' (list): results of models trained with different transformations
        target_name (str): The name of the target variable (used for labeling the plot).

    Returns:
        None: Displays the plot inline using matplotlib.
    """
    best_models = {item["combination"]: get_best_result(item["models"], target_name) for item in combo}
    best_models_data = [(str(key), item["test_mae"], "lags" in key) for key, item in best_models.items()]
    
    df = pd.DataFrame(best_models_data, columns=["Combination", "MAE", "Lags"])
    df = df.sort_values(by="MAE", ascending=True)
    # Визуализация
    plt.figure(figsize=(12, 7))
    colors = df["Lags"].map({True: '#ff9999', False: '#99ccff'})
    bars = plt.barh(df["Combination"], df["MAE"], color=colors)
    for bar in bars:
        width = bar.get_width()
        plt.text(width - 0.5,             
                 bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}',         
                 va='center')            

    plt.xlabel("MAE (lower is better)")
    plt.title(f"MAE per Predictor Combination for `{target_name}`")
    plt.gca().invert_yaxis()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', label='Uses Lags'),
        Patch(facecolor='#99ccff', label='No Lags')
    ]
    plt.legend(handles=legend_elements, title="Feature Type")
    
    plt.tight_layout()
    plt.show()


def plot_window_size(results, target_name):
    """
    Plot the mean absolute error (MAE) with error bars for different window sizes of an RNN forecast.

    Parameters
    ----------
    results : list of dict
        Each dict should contain:
        - "window_size" (int or float): The input window length used in the RNN.
        - "mae_mean" (float): The mean MAE computed over the test folds or runs.
        - "mae_std" (float): The standard deviation of the MAE.
    target_name : str
        Human-readable name of the target variable, used in the plot title.

    Returns
    -------
    None
        Displays a Matplotlib plot showing the MAE means with error bars for each window size.
    """
    plt.figure(figsize=(8, 4))
    plt.errorbar([str(item["window_size"]) for item in results], [item["mae_mean"] for item in results], 
                 yerr=[item["mae_std"] for item in results], fmt='o', 
                 capsize=5, color='blue', ecolor='gray', elinewidth=2)
    plt.title(f"RNN: The best window size selection for {target_name}")
    plt.xlabel("Window Size")
    plt.ylabel("Mean MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()