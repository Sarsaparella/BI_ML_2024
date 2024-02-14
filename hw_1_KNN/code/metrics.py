import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    true_p = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    false_p = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    false_n = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    true_n = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    
    accuracy = (true_p + true_n) / (true_p + false_p + false_n + true_n)
    precision = true_p / (true_p + false_p)
    recall = true_p / (true_p + false_n)
    f1 = 2 * precision * recall / (precision + recall)
    
    answer = f'Precision is {precision}. Recall is {recall}. F1 is {f1}. Accuracy is {accuracy}.'
    
    return answer


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    true_pos_neg = np.sum(y_pred == y_true)
    accuracy = true_pos_neg / y_pred.shape[0]
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    sse = sum(np.square(y_true - y_pred))
    sst = sum(np.square(y_true - np.mean(y_true)))
    
    r2 = 1 - sse/sst 
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = sum((y_true - y_pred)**2) / y_pred.shape[0]
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = sum(np.abs(y_true - y_pred)) / y_pred.shape[0]
    
    return mae
    