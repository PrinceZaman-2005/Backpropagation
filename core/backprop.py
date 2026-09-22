import numpy as np


class Backprop:
    """
    Backpropagation algorithm.

    Backpropagation:
    - calculates the loss
    - calculates the loss gradient
    - propagates the correction backward
    - stores gradients inside each layer
    - stops when a layer has gradients=False
    """

    def __init__(self, network):
        self.network = network
        self.loss = None

    def calculate_loss(self, output, target):
        """
        Calculate mean squared error.
        """

        return np.mean(
            (output - target) ** 2
        )

    def backward(self, y):
        """
        Propagate the loss backward through the network.

        Each layer stores its own gradient state.

        Backward flow:

            Loss
              ↓
            Layer 2
              ↓
            Layer 1
              ↓
            Input

        If gradients=False is encountered,
        backward propagation stops.
        """

        layers = self.network.layers

        # Get the final output from the forward pass.
        output = layers[-1].output

        # Calculate loss.
        self.loss = self.calculate_loss(
            output,
            y
        )

        # Clear old gradient state before
        # calculating the new gradients.
        for layer in layers:
            layer.gradient_signal = None
            layer.weight_gradient = None
            layer.bias_gradient = None

        # Derivative of MSE:
        #
        # dL/dOutput = 2(Output - Target) / N
        #
        error = (
            2 * (output - y)
            / y.size
        )

        # Walk backward through the network.
        for index in reversed(range(len(layers))):

            layer = layers[index]

            # Gradient tracking disabled.
            if not layer.gradients:
                break

            # Local correction:
            #
            # dL/dZ =
            # dL/dA * dA/dZ
            #
            delta = (
                error
                * layer.activation_derivative(
                    layer.output
                )
            )

            # Store the gradient signal.
            layer.gradient_signal = delta

            # Calculate weight gradient.
            layer.weight_gradient = np.dot(
                layer.input.T,
                delta
            )

            # Calculate bias gradient.
            layer.bias_gradient = np.sum(
                delta,
                axis=0,
                keepdims=True
            )

            # Propagate the correction to
            # the previous layer.
            if index > 0:
                error = np.dot(
                    delta,
                    layer.weights.T
                )

        return self.loss
