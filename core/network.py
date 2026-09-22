import numpy as np

from core.layer import Layer
from core.activation import sigmoid, sigmoid_derivative
from core.backprop import Backprop


class Network:
    """
    A simple feed-forward neural network.

    Architecture:

        Input
          ↓
        Layer 1
          ↓
        Sigmoid
          ↓
        Layer 2
          ↓
        Sigmoid
          ↓
        Output

    Each layer controls whether it participates
    in gradient tracking through `gradients`.
    """

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_layer = Layer(
            input_size,
            hidden_size,
            sigmoid,
            sigmoid_derivative,

            # Gradient tracking enabled.
            gradients=True
        )

        self.output_layer = Layer(
            hidden_size,
            output_size,
            sigmoid,
            sigmoid_derivative,

            # Gradient tracking enabled.
            gradients=True
        )

        self.layers = [
            self.hidden_layer,
            self.output_layer
        ]

        # Backpropagation operates on this network.
        self.backprop = Backprop(self)

    def forward(self, X):
        """
        Perform a sequential forward pass.
        """

        output = np.asarray(X)

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def predict(self, X):
        """
        Generate binary predictions.
        """

        output = self.forward(X)

        return (output >= 0.5).astype(int)
