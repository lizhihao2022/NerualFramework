from time import time

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np


class AverageRecord(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    def __init__(self, split="valid"):
        self.start_time = time()
        self.split = split

        self.y_pred = None
        self.y_true = None

        self.mae = None
        self.rmse = None
        self.r2 = None

    def update(self, y_pred, y_true):
        if self.y_pred is None:
            self.y_pred = y_pred
            self.y_true = y_true
        else:
            self.y_pred = np.concatenate((self.y_pred, y_pred))
            self.y_true = np.concatenate((self.y_true, y_true))
        self.compute_metrics()

    def compute_metrics(self):
        self.mae = mean_absolute_error(self.y_true, self.y_pred)
        self.rmse = sqrt(mean_squared_error(self.y_true, self.y_pred))
        self.r2 = r2_score(self.y_true, self.y_pred)
        self.mape = mean_absolute_percentage_error(self.y_true, self.y_pred)

    def format_metrics(self):
        result = "MAE: {:.8f} | ".format(self.mae)
        result += "MAPE: {:.4f} | ".format(self.mape)
        result += "RMSE: {:.4f} | ".format(self.rmse)
        result += "R2: {:.2%} | ".format(self.r2)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result

    def to_dict(self):
        return {
            "{} MAE".format(self.split): self.mae,
            "{} MAPE".format(self.split): self.mape,
            "{} RMSE".format(self.split): self.rmse,
            "{} R2".format(self.split): self.r2
        }
        
    def __repr__(self):
        return self.mae
    
    def __str__(self):
        return self.format_metrics()
