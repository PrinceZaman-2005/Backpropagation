from core.network import Network

from gate.xor_gate import get_xor_data
from training.training import Trainer


def display_results(X, y, predictions):
    """
    Display the final predictions.
    """

    print("\nFinal Results")
    print("=" * 30)

    for inputs, target, prediction in zip(
        X,
        y,
        predictions
    ):
        print(
            f"{inputs} -> "
            f"target: {target}, "
            f"prediction: {prediction}"
        )


def main():
    # Load experiment data
    X, y = get_xor_data()

    # Create the network.
    # The network internally owns Backprop.
    network = Network(
        input_size=2,
        hidden_size=2,
        output_size=1
    )

    # Create training module
    trainer = Trainer(
        network=network,
        learning_rate=0.4,
        epochs=10000,
        print_every=1000
    )

    print("Training Neural Network")
    print("=" * 30)

    # Train the network
    trainer.fit(X, y)

    # Generate final predictions
    predictions = network.predict(X)

    # Display results
    display_results(
        X,
        y,
        predictions
    )


if __name__ == "__main__":
    main()
