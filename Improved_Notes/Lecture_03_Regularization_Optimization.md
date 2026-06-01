# CS231n (Spring 2025) — Lecture 3: Regularization and Optimization

> **Instructor:** Zane Durante
> **Date:** April 8, 2025

---

## 1. Recap: Where We Are

From Lecture 2, we now have:
- A dataset of examples $\{(x_i, y_i)\}$
- A **score function**: $f(x, W) = Wx + b$
- A **loss function** that tells us how good our current classifier is:

$$L(W) = \frac{1}{N} \sum_i L_i(f(x_i, W), y_i)$$

For a given W on 3 training examples with 3 classes, we get per-class scores. The loss quantifies the mismatch between these scores and the true labels.

---

## 2. Regularization

### The Full Objective

$$L(W) = \underbrace{\frac{1}{N}\sum_i L_i}_{\text{Data Loss}} + \underbrace{\lambda R(W)}_{\text{Regularization}}$$

- **Data Loss:** Model predictions should match training data
- **Regularization:** Prevent the model from doing *too well* on training data
- **$\lambda$:** Regularization strength (hyperparameter)

### Why Regularize?

Motivation comes from **Occam's Razor**: among competing hypotheses, the simplest is best.

Intuition: Given training points, many functions can fit the data perfectly. Regularization pushes against fitting noise — we prefer simpler models that generalize better.

Three concrete reasons:
1. **Express preferences** over weights
2. **Keep the model simple** so it works on test data
3. **Improve optimization** by adding curvature to the loss landscape

### Common Regularizers

| Method | Formula | Effect |
|--------|---------|--------|
| **L2** | $R(W) = \sum_k \sum_l W_{k,l}^2$ | Prefers small, **spread-out** weights |
| **L1** | $R(W) = \sum_k \sum_l |W_{k,l}|$ | Encourages **sparse** solutions |
| **Elastic Net** | $R(W) = \alpha R_{L1} + \beta R_{L2}$ | Combines both |

**L2 intuition:** Given input $x = [1,1,1,1]$ and two weight vectors $w_1 = [1,0,0,0]$ vs. $w_2 = [0.25,0.25,0.25,0.25]$, both give the same dot product, but L2 prefers $w_2$ — it spreads influence across all input dimensions.

**L1 intuition:** L1 encourages weight vectors where most entries are exactly zero, acting as implicit feature selection.

More complex regularizers (covered later): Dropout, Batch Normalization, Stochastic Depth, Fractional Pooling.

---

## 3. Optimization

### Goal

$$\arg\min_W L(W)$$

How do we find the W that minimizes the loss?

### Strategy #1: Random Search

Randomly sample W matrices, keep the best one.
- **CIFAR-10 result:** ~15.5% accuracy
- Sounds surprisingly good, but SOTA is ~99.7%

### Strategy #2: Follow the Slope (Gradient Descent)

**Gradient:** The vector of partial derivatives. The direction of steepest descent is the **negative gradient**.

$$W \leftarrow W - \alpha \nabla_W L(W)$$

where $\alpha$ is the learning rate.

**Numerical Gradient:**
- Approximate using finite differences: $\frac{\partial L}{\partial w} \approx \frac{L(w+h) - L(w)}{h}$
- Slow, approximate, but easy to write and useful for debugging

**Analytic Gradient:**
- Derive the exact gradient formula using calculus
- Fast, exact, but error-prone to derive

**Best Practice:** Derive analytic gradients, then verify correctness using numerical gradient checks.

### Stochastic Gradient Descent (SGD)

Full loss: $L(W) = \frac{1}{N}\sum_i L_i + \lambda R(W)$ — computing the sum over all N examples is expensive when N is large.

**Solution:** Approximate the gradient using a **minibatch** (e.g., 32, 64, 128 examples):

$$L(W) \approx \frac{1}{|B|} \sum_{i \in B} L_i + \lambda R(W)$$

---

## 4. Loss Functions (Detailed)

### Multiclass SVM (Hinge) Loss

$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

- Wants the correct class score to exceed all incorrect class scores by at least a margin $\Delta$
- $\Delta$ is typically set to 1 (its exact value is absorbed by the scale of W through regularization)
- The loss is 0 if the margin condition is satisfied

### Softmax (Cross-Entropy) Loss

$$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

- Interprets scores as unnormalized log-probabilities
- The softmax function converts scores to a probability distribution: $P(y=k|x) = e^{s_k} / \sum_j e^{s_j}$
- Minimizing cross-entropy = maximizing the probability of the correct class
- Loss ranges from 0 (perfect) to $\infty$ (completely wrong)

---

## Core Takeaways

1. **Regularization** is not optional — it prevents overfitting and encodes preferences for simpler models. L2 spreads weights out; L1 encourages sparsity.
2. **Gradient descent** is the workhorse optimization algorithm. SGD with minibatches makes it practical for large datasets.
3. **Numerical gradients** are for checking; **analytic gradients** are for training.
4. Two standard loss functions: **Hinge loss** (focus on margins) and **Cross-entropy** (probabilistic interpretation).
5. The learning rate and regularization strength $\lambda$ are the two most important hyperparameters.

---

*Notes compiled from CS231n Spring 2025 Lecture 3 slides and course materials.*
