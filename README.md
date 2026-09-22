# 🔁 Backpropagation

Backpropagation is a learning algorithm that helps a neural network adjust its weights by working backward from the prediction error.

This project builds on the **Nested Perceptron** from Day 2 and introduces the idea of allowing the network to **learn its weights instead of manually configuring them**.

### How it works

The basic learning process is:

**Input → Forward Pass → Prediction → Error → Backward Pass → Weight Update**

The network first makes a prediction.

Then, the prediction is compared with the expected target to calculate an error.

Backpropagation sends information about that error backward through the network so that the weights can be adjusted.

Over multiple iterations, the network gradually improves its predictions.

### What's included

* NumPy-based neural network implementation
* Forward propagation
* Prediction error calculation
* Backward propagation
* Weight updates
* Layered network structure
* XOR gate experiment
* In-memory result routing through `main.py`

### Why this project?

In Day 2, the Nested Perceptron could solve XOR, but its weights were manually configured.

Backpropagation introduces a more important idea:

> **The network can learn how its internal weights should change from its mistakes.**

Instead of manually deciding how the hidden layer should behave:

**Manually configured weights → Prediction**

we can move toward:

**Prediction → Error → Backpropagation → Learned weights**

This is one of the fundamental ideas behind training neural networks.

### Day 3

This is **Day 3** of my **10 Days of Modeling AI for Beginners** series.

The goal of this series is not to build a state-of-the-art model, but to make fundamental AI mechanisms visible, understandable, and experimentable.

> Make the mistake. Trace the error. Adjust the weights and Learn!
