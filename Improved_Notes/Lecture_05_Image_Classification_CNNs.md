# CS231n (Spring 2025) — Lecture 5: Image Classification with CNNs

> **Instructor:** Justin Johnson
> **Date:** April 15, 2025

---

## 1. The Problem with Fully-Connected Layers

**Recap — 2-layer Neural Network:**

```
x (3072)  →  W1  →  h (100)  →  W2  →  s (10)
```

The first layer $W_1$ is a [3072 × 100] matrix. Every hidden unit computes a weighted sum over **all** input pixels. This destroys the **2D spatial structure** of the image — neighboring pixels that form an edge are treated no differently than pixels on opposite corners.

**Recall limitations of linear classifiers:**
- Visual view: Learn one template per class (e.g., one averaged "horse")
- Geometric view: Can only draw linear decision boundaries

Neural networks solve the decision boundary problem (via non-linearity), but fully-connected layers still don't leverage spatial structure.

---

## 2. Motivation: Hand-Crafted Image Features (Pre-2012)

Before end-to-end deep learning, the standard approach was:
1. Extract hand-designed features from images
2. Train a linear classifier on those features

| Feature | How It Works |
|---------|-------------|
| **Color Histogram** | Count the frequency of each color in the image; simple but discards all spatial information |
| **Histogram of Oriented Gradients (HoG)** | Divide image into 8×8 pixel regions, quantize edge direction into 9 bins per region. A 320×240 image → 30×40×9 = 10,800-dimensional feature vector |
| **Bag of Words** | Extract random patches, cluster them to form a "codebook" of visual words, then encode each image by counting which visual words appear |

**Key insight:** These features all involve **local** operations (edges, gradients in small neighborhoods), not global averages. CNNs automate this feature extraction through learnable convolution.

---

## 3. The Convolutional Neural Network

CNN = Feature extraction (convolution + pooling) + Classifier (fully-connected layers at the end)

Trained **end-to-end** with backpropagation and gradient descent.

### Core Idea

Convolution and pooling operators extract features while **respecting 2D image structure**. The fully-connected layers form an MLP at the end to predict class scores.

---

## 4. Convolution Layer

### Operation

A **filter** (kernel) of size $K_h \times K_w \times C_{in}$ slides over the input volume spatially. At each position, it computes a dot product between the filter weights and the local patch:

$$\text{Output}[i, j] = \sum_{c=0}^{C_{in}-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} W[c, u, v] \cdot X[c, i+u, j+v] + b$$

Multiple filters produce multiple output channels (activation maps).

### Spatial Dimensions

$$W' = \frac{W - K + 2P}{S} + 1$$

| Parameter | Meaning |
|-----------|---------|
| $W$ | Input spatial size |
| $K$ | Filter size |
| $P$ | Padding |
| $S$ | Stride |
| $W'$ | Output spatial size |

**Common configuration:** 3×3 filters, stride 1, padding 1 → output size = input size.

### Three Inductive Biases

| Bias | Meaning | Why Useful |
|------|---------|------------|
| **Locality** | Filters only look at small spatial neighborhoods | Natural images have local structure |
| **Weight Sharing** | Same filter is reused across all spatial positions | Drastically reduces parameters; enables translation equivariance |
| **Translation Equivariance** | Shifting the input shifts the output identically | A cat in the top-left is still a cat in the bottom-right |

---

## 5. Pooling Layer

Pooling reduces spatial dimensions, providing downsampling and some translation invariance.

| Type | Operation |
|------|-----------|
| **Max Pooling** | Take the maximum value in each window — introduces non-linearity |
| **Average Pooling** | Take the mean value in each window — linear operation |

**Typical configuration:** 2×2 filter, stride 2 → halves spatial dimensions.

---

## 6. Learned Filter Hierarchy

| Layer Depth | What Filters Detect |
|-------------|---------------------|
| **Layer 1** | Oriented edges, color blobs, Gabor-like patterns |
| **Middle layers** | Textures, corners, simple patterns |
| **Deep layers** | Object parts: eyes, wheels, text patterns, semantic concepts |

The hierarchy emerges naturally from training — the network discovers that edges → textures → parts → objects is the most efficient representation.

---

## 7. Typical CNN Architecture Pattern

```
INPUT → [[CONV → ReLU]*N → POOL?]*M → [FC → ReLU]*K → FC → SOFTMAX
```

- **Body (CONV + ReLU + POOL):** Hierarchical feature extraction, gradually reducing spatial dimensions while increasing channel depth
- **Head (FC layers):** Small classifier mapping extracted features to class scores
- **All parameters learned end-to-end** via backpropagation

### Convolution Variants

| Dimension | Application |
|-----------|-------------|
| **1D Conv** | Sequences, time-series, text |
| **2D Conv** | Images (standard) |
| **3D Conv** | Volumetric data, video (spatiotemporal) |
| **1×1 Conv** | Channel-wise projection (dimensionality reduction/expansion) |

---

## Core Takeaways

1. Fully-connected layers destroy spatial structure; **CNNs preserve it** through local connectivity and weight sharing.
2. Convolution provides three inductive biases: **locality, weight sharing, and translation equivariance** — all well-suited to natural images.
3. Hand-crafted features (HoG, Bag of Words) provided the intuition; CNNs automate feature learning end-to-end.
4. Learned filters form a hierarchy: **edges → textures → parts → objects**.
5. A standard CNN alternates convolution+ReLU blocks with occasional pooling, followed by FC classifiers at the end.

---

*Notes compiled from CS231n Spring 2025 Lecture 5 slides and course materials.*
