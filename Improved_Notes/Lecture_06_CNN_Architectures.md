# CS231n (Spring 2025) — Lecture 6: Training CNNs and CNN Architectures

> **Instructor:** Zane Durante
> **Date:** April 17, 2025

---

## 1. Lecture Overview: Two Broad Themes

| How to Build CNNs | How to Train CNNs |
|--------------------|-------------------|
| Layers in CNNs | Data Preprocessing |
| Activation Functions | Data Augmentation |
| CNN Architectures | Transfer Learning |
| Weight Initialization | Hyperparameter Selection |

---

## 2. CNN Component Review

### Core Building Blocks

| Component | Role |
|-----------|------|
| **Convolution Layer** | Local feature extraction with shared filters |
| **Pooling Layer** | Spatial downsampling (typically 2×2, stride 2) |
| **Fully-Connected Layer** | Global classifier mapping features to class scores |
| **Normalization Layer** | Stabilize and accelerate training |
| **Activation Function** | Introduce non-linearity |
| **Dropout** | Regularization |

### Normalization Layers

High-level idea: learn parameters $\gamma, \beta$ that scale and shift normalized input data.

1. Normalize input: $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$
2. Scale and shift: $y = \gamma \hat{x} + \beta$

| Type | Normalization Over | Best For |
|------|-------------------|----------|
| **Batch Norm** | Batch dimension (N) | Large batches, CNNs |
| **Layer Norm** | Feature dimension (D) | Small batches, Transformers |
| **Instance Norm** | Spatial dimensions (H, W) | Style transfer |
| **Group Norm** | Groups of channels | Small batch sizes |

Edge case: Normalization doesn't always make sense and doesn't fully resolve poor weight initialization — it helps but doesn't eliminate optimization issues.

---

## 3. Regularization: Dropout

### Mechanism

In each forward pass, randomly set a fraction $p$ of neurons to zero (common choice: $p = 0.5$).

### Why This Works

1. **Prevents co-adaptation:** Neurons can't rely on specific other neurons being present → forces redundant, robust representations.

   Example: Instead of one neuron detecting "ear" and another detecting "tail" that always co-fire, the network learns multiple independent detectors for each feature.

2. **Ensemble interpretation:** Each forward pass uses a different binary mask → effectively training $2^n$ different networks that share parameters. An FC layer with 4096 units has ~$10^{1233}$ possible masks (more than atoms in the universe).

### Test-Time Behavior

At test time, **all neurons are active**. To compensate, multiply activations by $p$ (the keep-probability) to match the expected value during training.

**Summary:** Drop during training; scale at test time.

---

## 4. Activation Functions

| Function | Formula | Key Properties |
|----------|---------|----------------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Squashes to [0,1]; historically popular as a "firing rate" analogy; **saturates at extremes** → kills gradients; many sigmoid layers → vanishing gradients |
| **Tanh** | $\tanh(x)$ | Zero-centered [-1,1]; still saturates |
| **ReLU** | $\max(0, x)$ | **Default choice.** Does not saturate in positive region; computationally efficient; converges ~6× faster than sigmoid in practice [Krizhevsky et al., 2012] |
| **Leaky ReLU** | $\max(\alpha x, x)$ | Small gradient for negative inputs → prevents dead neurons |
| **ELU** | $x$ if $x > 0$, else $\alpha(e^x - 1)$ | Smooth, zero-centered, but uses expensive exp() |

**ReLU is the recommended default** for most problems. Sigmoid is now rarely used in hidden layers due to vanishing gradient issues.

---

## 5. CNN Architectures: Historical Progression

### AlexNet (2012) — The Breakthrough
- **ImageNet Top-5 Error:** ~15.4%
- **Architecture:** 5 conv layers + 3 FC layers, ~60M parameters
- **Key innovations:** ReLU activation, Dropout for FC layers, data augmentation (random crops + flips + PCA color augmentation), GPU training (split across 2 GPUs)

### VGG (2014) — Deep and Uniform
- **VGG16/19, Top-5 Error:** ~7.3%
- **Design philosophy:** Use only 3×3 convolutions (stride 1, pad 1) with periodic 2×2 max-pooling
- **Why stack small filters?** Three 3×3 conv layers have the same receptive field as one 7×7, but with:
  - More non-linearities (3 ReLUs vs 1)
  - Fewer parameters ($3 \times 3^2 \times C^2 = 27C^2$ vs $7^2 \times C^2 = 49C^2$)

### GoogLeNet (2014) — Inception Module
- **Top-5 Error:** ~6.7%
- **Key idea:** Inception modules use parallel convolution branches (1×1, 3×3, 5×5) + pooling, with 1×1 bottleneck layers to reduce computation

### ResNet (2015) — Skip Connections
- **ResNet-152, Top-5 Error:** ~3.6%
- **The degradation problem:** Deeper plain networks have higher training *and* test error — this is not overfitting, it's an optimization problem.
- **Solution: Residual learning.** Output = $F(x) + x$
  - The skip connection provides a gradient "highway" during backpropagation
  - L2 regularization naturally drives residual block weights toward zero → identity mapping as default behavior
  - Enables training of 152+ layer networks

**Bottleneck Block (ResNet-50+):** 1×1 (reduce) → 3×3 (process) → 1×1 (expand) — efficient for very deep networks.

**Training recipe:** BN after every CONV, He initialization, SGD + Momentum (0.9), initial LR 0.1 with step decay, weight decay 1e-5, **no Dropout** (BN handles regularization).

---

## 6. Training Practices

### Data Augmentation

- Geometric: Random horizontal flip, random resized crop, rotation
- Color: Color jitter (brightness, contrast, saturation), PCA color augmentation
- Cutout / Random Erasing: Mask out random patches

**Test-time augmentation (TTA):** Average predictions over multiple crops/flips → ~1-2% accuracy boost.

### Transfer Learning

| Dataset Size | Strategy |
|--------------|----------|
| **Small** (hundreds) | Freeze pretrained features, train only new classifier layer |
| **Medium** (thousands) | Fine-tune later layers, freeze early layers |
| **Large** (tens of thousands+) | Fine-tune all layers (slower learning rate) |

### Hyperparameter Tuning Strategy

1. First, overfit on a tiny subset to verify implementation correctness
2. Coarse search for learning rate
3. **Random search** (not grid search) over hyperparameter space
4. Monitor train/val loss curves throughout training

---

## 7. Weight Initialization

Proper initialization is critical — poorly initialized weights can cause vanishing/exploding activations and gradients.

- **He (Kaiming) Initialization:** Weights ~ $\mathcal{N}(0, \sqrt{2/n_{in}})$ — designed for ReLU networks
- **Xavier (Glorot) Initialization:** Weights ~ $\mathcal{N}(0, \sqrt{1/n_{in}})$ — designed for tanh/sigmoid

---

## Core Takeaways

1. **Normalization (especially Batch Norm) + ReLU + Dropout** form the standard CNN training stack.
2. Small 3×3 filters stacked deep > large filters — more non-linearity and fewer parameters.
3. **Residual connections** are the most important architectural innovation since AlexNet — they make very deep networks trainable.
4. **Data augmentation** and **transfer learning** are essential when labeled data is limited.
5. Architecture progression: AlexNet → VGG → GoogLeNet → ResNet — each generation solved a specific limitation of the previous.

---

*Notes compiled from CS231n Spring 2025 Lecture 6 slides and course materials.*
