import numpy as np


class Trainer:
    """
    Handles the training process of a neural network.

    The Trainer is responsible for:
    - running training epochs
    - requesting backpropagation
    - updating trainable parameters
    - tracking loss and accuracy

    Backpropagation calculates and stores gradients
    inside each layer.
    """

    def __init__(
        self,
        network,
        learning_rate=0.1,
        epochs=10000,
        print_every=1000
    ):
        self.network = network

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.print_every = print_every

        self.history = {
            "loss": [],
            "accuracy": []
        }

    def calculate_accuracy(self, predictions, targets):
        """
        Calculate binary classification accuracy.
        """

        predicted_classes = (
            predictions >= 0.5
        ).astype(int)

        correct = (
            predicted_classes == targets
        )

        return np.mean(correct)

    def update_parameters(self):
        """
        Update parameters using gradients stored
        inside each layer.

        Layers with gradients=False are skipped.
        """

        for layer in self.network.layers:

            if not layer.gradients:
                continue

            if layer.weight_gradient is None:
                continue

            layer.weights -= (
                self.learning_rate
                * layer.weight_gradient
            )

            layer.bias -= (
                self.learning_rate
                * layer.bias_gradient
            )

    def fit(self, X, y):
        """
        Train the network.

        Returns training history containing
        loss and accuracy for each epoch.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        for epoch in range(1, self.epochs + 1):

            # Forward pass
            predictions = self.network.forward(X)

            # Backpropagation calculates the loss
            # and stores gradients inside each layer.
            loss = self.network.backprop.backward(y)

            # Calculate accuracy
            accuracy = self.calculate_accuracy(
                predictions,
                y
            )

            # Store metrics
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)

            # Update parameters using stored gradients.
            self.update_parameters()

            # Optional training progress
            if (
                epoch == 1
                or epoch % self.print_every == 0
                or epoch == self.epochs
            ):
                print(
                    f"Epoch {epoch}/{self.epochs} "
                    f"| Loss: {loss:.6f} "
                    f"| Accuracy: {accuracy:.2%}"
                )

        return self.history
