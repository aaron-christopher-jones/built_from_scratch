import numpy as np


def metric_maep(y, yhat):
    """
    The mean absolute error as percentage evaluation metric.

    Parameters
    ----------
    y : array
    yhat : array
    """
    return np.round(np.mean(np.abs(y - yhat)) / np.mean(y), 4)


def metric_rmse(y, yhat):
    """
    The root mean squared error evaluation metric.

    Parameters
    ----------
    y : array
    yhat : array
    """
    return np.round(np.sqrt(np.mean(np.power(y - yhat, 2))), 4)
