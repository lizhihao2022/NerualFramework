from time import time
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


METRIC_DICT = {
    'mae': mean_absolute_error,
    'MAE': mean_absolute_error,
    
    'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    'RMSE': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)),
    
    'r2': r2_score,
    'R2': r2_score,
    
    'mape': mean_absolute_percentage_error,
    'MAPE': mean_absolute_percentage_error,
    
    'mse': mean_squared_error,
    'MSE': mean_squared_error,
}
VALID_METRICS = list(METRIC_DICT.keys())


class AverageRecord(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    def __init__(self, metrics=['mae', 'r2'], split="valid"):
        self.metric_list = metrics
        self.start_time = time()
        self.split = split
        self.metrics = {metric: AverageRecord() for metric in self.metric_list}

    def update(self, y_pred, y_true):
        for metric in self.metric_list:
            self.metrics[metric].update(METRIC_DICT[metric](y_true, y_pred))

    def compute_metrics(self):
        for metric in self.metric_list:
            self.metrics[metric] = METRIC_DICT[metric](self.y_true, self.y_pred)

    def format_metrics(self):
        result = ""
        for metric in self.metric_list:
            result += "{}: {:.4f} | ".format(metric.upper(), self.metrics[metric].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result

    def to_dict(self):
        return {
            metric: self.metrics[metric].avg for metric in self.metric_list
        }

    def __repr__(self):
        return self.metrics[self.metric_list[0]].avg
    
    def __str__(self):
        return self.format_metrics()


class LossRecord:
    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def format_metrics(self):
        result = ""
        for loss in self.loss_list:
            result += "{}: {:.4f} | ".format(loss, self.loss_dict[loss].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result
    
    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
