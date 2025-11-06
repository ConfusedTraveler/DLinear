# Built-in libraries
import math
import numpy as np
import pandas as pd

# Sklearn
from sklearn import metrics


def smape(A, F):
    try:
        return (100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))))
    except:
        return np.NaN


def rmse(A, F):
    try:
        return math.sqrt(metrics.mean_squared_error(A, F))
    except:
        return np.NaN


def RegressionEvaluation(Prices):
    '''
    Parameters
    ----------
    Prices : pd.DataFrame
        包含真实值与预测值两列，列名分别为真实值和预测值。

    Returns
    -------
    MAE   : 平均绝对误差 (Mean Absolute Error)
    RMSE  : 均方根误差 (Root Mean Square Error)
    MAPE  : 平均绝对百分比误差 (Mean Absolute Percentage Error)
    SMAPE : 对称平均百分比误差 (Symmetric MAPE)
    R2    : 判定系数 (R-squared)
    MSE   : 均方误差 (Mean Squared Error)
    '''

    SeriesName = Prices.columns[0]
    Prediction = Prices.columns[1]

    Y = Prices[SeriesName].to_numpy()
    Pred = Prices[Prediction].to_numpy()

    # Metrics
    MAE = metrics.mean_absolute_error(Y, Pred)
    MSE = metrics.mean_squared_error(Y, Pred)
    RMSE = math.sqrt(MSE)

    try:
        MAPE = np.mean(np.abs((Y - Pred) / Y)) * 100.0
    except:
        MAPE = np.NaN

    SMAPE = smape(Y, Pred)
    R2 = metrics.r2_score(Y, Pred)

    return (MAE, RMSE, MAPE, SMAPE, R2, MSE)
