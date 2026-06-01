# CS231n (Spring 2025) — Lecture 2: Image Classification with Linear Classifiers

> **Instructor:** Ehsan Adeli
> **Date:** April 3, 2025

---

## 1. Image Classification: The Core Task

**Input:** An image (a tensor of integers in [0, 255], e.g., 800 × 600 × 3 for RGB)
**Output:** A label from a fixed set of categories {dog, cat, truck, plane, ...}

### The Semantic Gap

What the computer sees (a grid of numbers) has no inherent connection to the semantic concept a human perceives. Bridging this gap is the fundamental challenge.

---

## 2. Seven Challenges of Image Classification

| Challenge | Description |
|-----------|-------------|
| **Viewpoint Variation** | All pixels change when the camera moves |
| **Illumination** | Same object under different lighting yields dramatically different pixel values |
| **Background Clutter** | Object may blend into a complex background |
| **Occlusion** | Only a small portion of the object may be visible |
| **Deformation** | Non-rigid objects can change shape |
| **Intraclass Variation** | One category contains enormous visual diversity (e.g., many dog breeds) |
| **Context** | The same visual pattern may be interpreted differently depending on surrounding cues |

Subtext: there is no obvious way to hard-code an algorithm that recognizes a cat. Unlike sorting numbers, the rules are not explicit.

---

## 3. The Data-Driven Approach

Instead of hand-crafting rules, we:

1. **Collect** a dataset of images and labels
2. **Train** a classifier using machine learning on that data
3. **Evaluate** the classifier on new, unseen images

---

## 4. K-Nearest Neighbors (K-NN)

### Nearest Neighbor Classifier

- **Train:** Memorize all training data and labels — O(1)
- **Predict:** For each test image, find the most similar training image and output its label — O(N)

This is backwards from what we want: training should be slow (acceptable), but prediction should be fast.

### Distance Metrics

**L1 (Manhattan) distance:**

$$d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|$$

Measures distance by moving along grid lines (like walking through city blocks).

**L2 (Euclidean) distance:**

$$d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$$

Measures the straight-line (as-the-crow-flies) distance.

The choice of distance metric is a **hyperparameter** — it depends on the problem and dataset.

### K-Nearest Neighbors

Instead of copying the label from the single nearest neighbor, take a **majority vote** from the K closest points.

- **K=1:** Perfectly memorizes training data; tends to overfit
- **Larger K:** Smoother decision boundaries, more robust to outliers

---

## 5. Hyperparameter Tuning

Hyperparameters (K value, distance metric) are choices about the algorithm itself, not learned from data.

### The Correct Protocol

| Approach | Verdict |
|----------|---------|
| Choose hyperparams that work best on training data | ❌ Bad — K=1 always wins on training data |
| Choose hyperparams that work best on test data | ❌ Bad — information leak; test set is no longer representative |
| **Split data into train/val/test; tune on val** | ✅ Correct |
| **Cross-validation** (for small datasets) | ✅ Correct |

**Golden Rule:** The test set is used exactly once — for final evaluation only.

---

## 6. Linear Classifier

### Score Function

$$f(x, W, b) = Wx + b$$

- **W:** Weight matrix [K × D] — each row is a classifier template for one class
- **b:** Bias vector [K × 1] — class-specific offsets
- **x:** Flattened input image [D × 1]

### Two Geometric Interpretations

**Template Matching View:** Each row of W is a learned "template" for one category. The dot product measures how well the input matches each template. CIFAR-10 templates reveal averaged prototypes per class (e.g., a blurry "average car," a "two-headed horse" from averaging left-facing and right-facing horses).

**Hyperplane View:** In high-dimensional space, the linear classifier draws linear decision boundaries (hyperplanes) separating categories. A linear classifier can only separate classes that are **linearly separable**.

### Limitations

- Cannot solve non-linearly separable problems (e.g., XOR)
- One template per class — cannot handle multimodal class distributions

> The linear classifier is the fundamental building block of neural networks. Stacking linear layers with nonlinear activation functions enables learning complex, nonlinear decision boundaries.

---

## 7. Loss Function (Preview)

A loss function quantifies how good (or bad) our current classifier is. Given a dataset of examples $\{(x_i, y_i)\}$, the loss is an average over individual per-example losses.

Two main loss functions (detailed in Lecture 3):
- **Multiclass SVM (Hinge) Loss:** Penalizes incorrect classes whose score exceeds the correct class score minus a margin
- **Softmax (Cross-Entropy) Loss:** Interprets scores as unnormalized log-probabilities; penalizes the negative log-likelihood of the correct class

---

## Core Takeaways

1. Image classification is hard because of the **semantic gap** between pixels and meaning.
2. K-NN is a simple baseline: no training, but slow at inference; provides geometric intuition.
3. **Always split train/val/test** — tuning hyperparameters on the test set is the cardinal sin of ML.
4. The **linear classifier** $f(x,W) = Wx + b$ is the simplest parametric model and the foundation for neural networks.
5. A linear classifier learns one template per class and draws linear decision boundaries.

---

*Notes compiled from CS231n Spring 2025 Lecture 2 slides and course materials.*
