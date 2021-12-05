from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_maep(y, yhat):
    return np.round(np.mean(np.abs(y - yhat)) / np.mean(y), 4)


def metric_rmse(y, yhat):
    return np.round(np.sqrt(np.mean(np.power(y - yhat, 2))), 4)


def build_confusion_matrix(y, yhat, as_percentage=False):
    assert len(y) == len(yhat), "Different lengths."
    output = pd.DataFrame({"Y": y, "YHAT": yhat})
    confusion = pd.crosstab(output.Y, output.YHAT)
    return (confusion / output.shape[0] if as_percentage else confusion)
