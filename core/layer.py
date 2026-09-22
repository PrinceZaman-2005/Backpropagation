import numpy as np


class Layer:
    """
    A simple fully-connected neural-network layer.

    Each layer contains:
    - weights
    - bias
    - activation function
    - activation derivative
    - gradient tracking state
    - forward-pass values needed for backpropagation
    """

    def __init__(
        self,
        input_size,
        output_size,
        activation,
        activation_derivative,
        gradients=True
    ):
        self.weights = np.random.randn(
            input_size,
            output_size
        ) * 0.7 # set higher, sigmoid(0) ≈ 0.5

        self.bias = np.zeros((1, output_size))

        self.activation = activation
        self.activation_derivative = activation_derivative

        # Controls whether this layer participates
        # in gradient tracking and backpropagation.
        self.gradients = gradients

        # Values stored during the forward pass.
        # Backpropagation will use these later.
        self.input = None
        self.linear_output = None
        self.output = None

        # Gradient state.
        # These will be filled during backpropagation.
        self.gradient_signal = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, inputs):
        """
        Perform the forward computation.

        z = XW + b
        a = activation(z)
        """

        self.input = inputs

        # Linear transformation
        self.linear_output = (
            np.dot(inputs, self.weights)
            + self.bias
        )

        # Activation
        self.output = self.activation(
            self.linear_output
        )

        return self.output
