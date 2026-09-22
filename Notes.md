# 📙 Notes

## About This Implementation

This project is a simplified implementation of backpropagation using NumPy.

The goal is not to reproduce a production-grade automatic differentiation system.

The goal is to make the core mechanism visible:

```text
Forward Pass
     ↓
Prediction
     ↓
Loss
     ↓
Loss Gradient
     ↓
Backward Pass
     ↓
Layer Gradients
     ↓
Parameter Update
```

The implementation intentionally keeps the mathematics and architecture small enough to inspect and experiment with.

---

## 1. Simplified Backpropagation

Our `Backprop` implementation is designed around a small sequential neural network.

The backward process starts from the output and moves through the layers in reverse:

```text
Loss
 ↓
Output Layer
 ↓
Hidden Layer
 ↓
Input
```

Each layer stores its own gradient-related state:

* `gradient_signal`
* `weight_gradient`
* `bias_gradient`

This makes the backward process explicit instead of hiding it behind an automatic differentiation framework.

### Gradient Tracking

Each layer has:

```python
gradients = True
```

or:

```python
gradients = False
```

When `gradients=False`, backpropagation stops at that layer.

Example:

```text
Loss
 ↓
Layer 2       gradients=True
 ↓
Layer 1       gradients=False
 X
```

This is an experimental gradient boundary.

It is useful for understanding that backward propagation requires every participating computational layer to pass the gradient signal backward.

---

## 2. The Loss Function

This implementation uses **Mean Squared Error (MSE)**:

```text
MSE = mean((prediction - target)²)
```

Its derivative is:

```text
dL/dprediction =
    2(prediction - target) / N
```

Backpropagation starts from this loss gradient and propagates it through the network.

### Important Limitation

The current Backprop implementation is specifically simplified around MSE.

Therefore:

> **This implementation should not be assumed to work correctly for every loss function or every neural-network architecture.**

Changing the loss function requires changing the corresponding loss gradient.

For example, using a different loss without updating its derivative would make the backward calculation mathematically incorrect.

---

## 3. Sigmoid

The network currently uses the sigmoid activation function:

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

Sigmoid produces values between:

```text
0 and 1
```

This makes it convenient for the XOR experiment because the targets are:

```text
0 or 1
```

Its derivative can be calculated from its output:

```text
sigmoid'(x) = output × (1 - output)
```

During backpropagation:

```text
dL/dZ =
    dL/dA × sigmoid'(Z)
```

### Important Limitation

Sigmoid is meaningful for this small binary experiment, but it is not automatically the best activation function for every problem.

Different architectures and tasks may require different activation functions and corresponding derivatives.

---

## 4. Learning Rate

The learning rate controls how strongly the calculated gradients change the parameters.

The basic update is:

```text
weight = weight - learning_rate × weight_gradient
```

and:

```text
bias = bias - learning_rate × bias_gradient
```

For this experiment, a learning rate of:

```text
0.4
```

was used.

This value was chosen experimentally for the small XOR network.

A learning rate is not universally meaningful.

### If the learning rate is too small

Training may become very slow.

```text
small step
→ small parameter changes
→ slow learning
```

### If the learning rate is too large

Training can become unstable.

```text
large step
→ overshooting
→ unstable loss
```

Therefore, the learning rate should be treated as an experimental hyperparameter rather than a fixed constant that works everywhere.

---

## 5. Weight Initialization

The current experiment uses:

```python
np.random.randn(input_size, output_size) * 0.7
```

The initial experiment used a smaller scale of:

```python
* 0.1
```

With the smaller initialization, the network started close to:

```text
z ≈ 0
```

which gives:

```text
sigmoid(z) ≈ 0.5
```

The network then struggled to make useful progress on the XOR experiment.

Increasing the initialization scale produced more useful initial activations for this small experiment.

### Important Limitation

This does **not** mean that `0.7` is a generally correct initialization strategy.

Proper initialization depends on the architecture and activation function.

This project uses the value as an experimental simplification.

---

## 6. XOR Is an Experiment, Not the Definition of Backpropagation

The network is tested using XOR:

```text
0 0 → 0
0 1 → 1
1 0 → 1
1 1 → 0
```

XOR is useful because a single Perceptron cannot solve it, while a small multi-layer network can.

However:

> Backpropagation is not an XOR-specific algorithm.

XOR is simply the small experiment used to observe the mechanism.

---

## 7. What We Successfully Demonstrated

With:

```text
Layer 1 gradients = True
Layer 2 gradients = True
```

the network successfully learned XOR.

The observed experiment reached:

```text
Accuracy = 100%
Loss ≈ 0.0009
```

With Layer 1 disabled:

```text
Layer 1 gradients = False
Layer 2 gradients = True
```

backpropagation stopped at Layer 1.

The network remained around:

```text
Loss ≈ 0.25
Accuracy ≈ 50%
```

This demonstrates the effect of restricting the backward gradient path.

---

## 8. Why This Implementation Can Break

This project intentionally simplifies many things.

The current implementation may fail or behave poorly when changing:

* the loss function
* the activation function
* the network architecture
* the number of layers
* the number of neurons
* the learning rate
* the weight initialization
* the dataset
* the output representation
* the optimization method

In particular:

> **The current simplified backpropagation is designed and tested around MSE + sigmoid + a small feed-forward network.**

It should therefore be treated as an educational model of backpropagation, not a universal implementation.

---

## 9. The Main Lesson

The purpose of this project is to make the mechanism visible.

Instead of:

```python
model.fit(X, y)
```

and hiding the entire process, we explicitly observe:

```text
Forward computation
       ↓
Activation
       ↓
Prediction
       ↓
MSE Loss
       ↓
Loss Gradient
       ↓
Backward propagation
       ↓
Local layer correction
       ↓
Weight/Bias gradients
       ↓
Parameter update
```

The implementation is intentionally imperfect in the sense that it is **not generalized for every possible neural-network configuration**.

That is acceptable for this experiment.

The purpose is understanding the mechanism first, then increasing generality later.
