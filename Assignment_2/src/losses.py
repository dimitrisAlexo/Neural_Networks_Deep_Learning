import numpy as np


def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.

    Returns:
    Mean Squared Error.
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """
    Derivative of Mean Squared Error (MSE) loss function.

    Parameters:
    - y_true: True values.
    - y_pred: Predicted values.

    Returns:
    Derivative of Mean Squared Error.
    """
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss function.

    Parameters:
    - y_true: True binary labels (0 or 1).
    - y_pred: Predicted probabilities.

    Returns:
    Binary Cross-Entropy.
    """
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    """
    Derivative of Binary Cross-Entropy loss function.

    Parameters:
    - y_true: True binary labels (0 or 1).
    - y_pred: Predicted probabilities.

    Returns:
    Derivative of Binary Cross-Entropy.
    """
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
