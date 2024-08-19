import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score


def calAUC(y_true, y_pred, multi_class=False):
    if not multi_class:
        return roc_auc_score(y_true, y_pred)
    else:
        return roc_auc_score(y_true, y_pred, multi_class='ovr')


def calF1Macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def calF1Micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def calMAPE(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def calMAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def calRMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calAccuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy


def calHitRate(y_hat, y, rate):
    relative_error = np.abs((y_hat - y) / y)
    hit = np.sum(relative_error < rate)
    total = len(y)
    return hit / (total + 1e-8)
