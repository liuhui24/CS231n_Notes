# CS231n (Spring 2025) — Lecture 4: Neural Networks and Backpropagation

> **Instructor:** Ehsan Adeli
> **Date:** April 10, 2025

---

## 1. Recap: From Linear Classifier to Optimization

We have:
- **Score function:** $f(x, W) = Wx + b$
- **Loss function:** Softmax (cross-entropy) or SVM (hinge), plus regularization
- **Optimization:** Gradient descent — follow the negative gradient to minimize loss

### Gradient Descent Variants

| Method | Key Idea |
|--------|----------|
| **SGD** | Update using minibatch gradient estimate |
| **SGD + Momentum** | Accumulate velocity for smoother trajectory |
| **RMSProp** | Per-parameter adaptive learning rates |
| **Adam** | Combines Momentum + RMSProp + bias correction |

### Learning Rate Schedules

| Schedule | Behavior |
|----------|----------|
| **Step** | Reduce LR at fixed epochs (e.g., ×0.1 at epoch 30, 60, 90) |
| **Cosine** | $\alpha_t = \frac{\alpha_0}{2}(1 + \cos(\frac{t\pi}{T}))$ |
| **Linear** | Linear decay from $\alpha_0$ to 0 |
| **Inverse Sqrt** | $\alpha_t = \alpha_0 / \sqrt{t}$ |
| **Linear Warmup** | Ramp up LR initially, then decay |

**Gradient checking:** Always verify analytic gradients with numerical gradients before full training.

---

## 2. Why Deep Learning?

The lecture opens with motivation from modern systems:
- **DALL-E 2/3:** Text-to-image generation with remarkable compositionality
- **GPT-4:** Multimodal reasoning across text and images
- **SAM (Segment Anything Model):** Zero-shot image segmentation
- **Sora:** Video generation as world simulation — scaling compute dramatically improves quality

These systems are built from the neural network fundamentals covered today.

---

## 3. Neural Networks: Beyond Linear Classifiers

### From Linear to 2-Layer Network

**Linear classifier:** $f = Wx$

**2-Layer Neural Network:** $f = W_2 \cdot \max(0, W_1 \cdot x)$

Where $\max(0, \cdot)$ is the **activation function** (ReLU).

### Why Non-linearity Is Essential

Without a non-linear activation, stacking linear layers collapses:

$$f = W_2(W_1 x) = (W_2 W_1)x = W' x$$

No matter how many layers, you still have a linear classifier. **The activation function is what gives the network its representational power.**

Geometric intuition: some datasets (e.g., concentric circles) are not linearly separable in the input space. A non-linear feature transform can map them to a space where they become separable.

### Terminology

"Neural Network" is a broad term; these are more precisely called **fully-connected networks** or **multi-layer perceptrons (MLPs)**.

### Architecture Notation

| Name | Layers | Hidden Layers |
|------|--------|---------------|
| 2-layer NN | Input → Hidden → Output | 1 |
| 3-layer NN | Input → Hidden → Hidden → Output | 2 |

### Hierarchical Computation

```
x (3072)  →  W1  →  h (100)  →  W2  →  s (10)
```

The hidden layer learns **100 templates** (not just the 10 class templates of a linear classifier). These templates are **shared across classes** — each hidden unit can contribute to multiple output classes, enabling more efficient representation.

---

## 4. Activation Functions

| Function | Formula | Notes |
|----------|---------|-------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Squashes to [0,1]; historically popular but saturates → kills gradients |
| **Tanh** | $\tanh(x)$ | Squashes to [-1,1]; zero-centered, but still saturates |
| **ReLU** | $\max(0, x)$ | Default choice; efficient, non-saturating in positive region; converges ~6× faster than sigmoid |
| **Leaky ReLU** | $\max(\alpha x, x)$ | Addresses "dead neuron" issue of ReLU |
| **ELU** | $x$ if $x>0$ else $\alpha(e^x-1)$ | Zero-centered, robust, but uses exp() |

**ReLU** is the recommended default for most problems.

---

## 5. Backpropagation

### The Chain Rule on Computation Graphs

Every differentiable operation can be represented as a node in a computation graph. Backpropagation applies the chain rule systematically:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$

> **Downstream gradient = Upstream gradient × Local gradient**

### Gradient Patterns for Common Gates

| Gate | Forward | Backward Rule |
|------|---------|---------------|
| **Add** | $x + y$ | Gradient is distributed unchanged to both inputs |
| **Multiply** | $x \times y$ | Gradient is swapped: $\partial L/\partial x = y \cdot \partial L/\partial z$ |
| **Copy / Split** | → multiple outputs | Sum gradients from all consumers |
| **Max / ReLU** | $\max(x, y)$ | Gradient routes only to the "winning" input; zero to others |
| **Sigmoid** | $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ — computable from forward pass alone |

### Backprop with Vectors and Matrices

For a matrix-vector multiply $z = Wx$:
- $\frac{\partial L}{\partial x} = W^T \frac{\partial L}{\partial z}$
- $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot x^T$

**Key rule:** The gradient of a variable always has the **same shape** as the variable itself. Use dimension analysis to derive backward passes — never construct the full Jacobian explicitly.

---

## 6. Practical Training Recipe for a 2-Layer Network

1. Define input/output dimensions and hidden layer size
2. Initialize weights (He/Kaiming initialization for ReLU networks)
3. **Forward pass:** Compute hidden activations → scores → loss
4. **Backward pass:** Compute gradients for all trainable parameters
5. **Gradient check:** Verify analytic gradients against numerical gradients
6. **Train:** Iterate SGD with minibatches until convergence

---

## Core Takeaways

1. **Non-linearity is mandatory** — without it, deep networks collapse to linear models.
2. Neural networks learn hierarchical templates: hidden layer templates are shared across output classes.
3. **ReLU** is the default activation function — fast, simple, effective.
4. **Backpropagation = chain rule applied to computation graphs.** Memorize the patterns: add distributes, multiply swaps, max routes.
5. Every modern deep learning framework (PyTorch, JAX, TensorFlow) is built on automatic differentiation using these exact principles.

---

*Notes compiled from CS231n Spring 2025 Lecture 4 slides and course materials.*
