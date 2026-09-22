import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    """

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    """
    Derivative of sigmoid.

    The derivative is calculated from
    the sigmoid output:

        sigmoid'(x) = output * (1 - output)
    """

    return output * (1 - output)
