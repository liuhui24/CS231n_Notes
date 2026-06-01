# CS231n (Spring 2025) — Lecture 9: Detection, Segmentation, Visualization & Understanding

> **Instructor:** Ehsan Adeli
> **Date:** April 29, 2025

---

## 1. Three Sequence Processing Primitives: A Comparison

| | RNN | 1D Convolution | Self-Attention |
|------|-----|---------------|----------------|
| **Domain** | 1D ordered sequences | N-dimensional grids | Sets of vectors |
| **Long sequences** | Theoretically good: $O(N)$ compute & memory | Bad: need many layers for large receptive field | Great: each output depends directly on all inputs |
| **Parallelism** | Poor: must compute sequentially | Good: outputs parallelizable | Excellent: just 4 matrix multiplications |
| **Cost** | $O(N)$ | $O(kN)$ | **$O(N^2)$** — the tradeoff for global interaction |

---

## 2. Vision Transformer (ViT)

Proposed by Dosovitskiy et al. (ICLR 2021): apply the standard NLP Transformer to images.

### Pipeline

1. **Patchify:** Split input image (e.g., 224×224×3) into $N$ non-overlapping patches (e.g., 16×16×3 each → 196 patches)

2. **Linear Projection:** Flatten each patch and project to a $D$-dimensional vector → these are the Transformer input tokens

3. **Add Positional Embedding:** Add a learned $D$-dim vector per spatial position → tells the Transformer where each patch came from (since self-attention is permutation-invariant)

4. **Transformer Encoder:** Process through standard Transformer blocks — **no masking**; each patch can attend to all other patches

5. **Classification:**
   - **`[CLS]` token approach:** Prepend a special learned token; use its output vector → linear projection to class scores
   - **Pooling approach:** Average-pool all output vectors → linear layer → class scores

### Key Insight

The architecture is nearly identical to NLP Transformers — demonstrating that the Transformer's set-processing capability is modality-agnostic.

---

## 3. Tweaks to the Transformer Architecture

The Transformer architecture has been remarkably stable since 2017. A few changes have become common:

- **Pre-LayerNorm:** Apply LayerNorm **before** attention/FFN (not after) — improves training stability
- **Learned positional embeddings** instead of sinusoidal
- Different configurations of encoder-only (ViT, BERT), decoder-only (GPT), or full encoder-decoder

---

## 4. Object Detection

### Task Definition

**Input:** Image
**Output:** Set of bounding boxes + class labels for each object

### Two-Stage Detectors (R-CNN Family)

| Method | Key Idea |
|--------|----------|
| **R-CNN (2014)** | Generate region proposals → warp each region → run CNN → classify. Slow: ~2000 forward passes per image |
| **Fast R-CNN (2015)** | Run CNN once on full image → extract features for each proposed region (ROI Pooling) → classify. Much faster |
| **Faster R-CNN (2015)** | Learn a **Region Proposal Network (RPN)** to generate proposals from CNN features → end-to-end trainable |

### Single-Stage Detectors

| Method | Key Idea |
|--------|----------|
| **YOLO (2016)** | Divide image into grid → each cell predicts bounding boxes and class scores in one shot. Very fast |
| **SSD** | Predict boxes at multiple feature map scales → handles objects of different sizes |
| **DETR (2020)** | Transformer-based: CNN backbone → Transformer encoder-decoder → predicts a fixed set of objects. No anchors, no NMS |

### Evaluation: Mean Average Precision (mAP)

Measures the area under precision-recall curve, averaged across classes and IoU thresholds.

---

## 5. Image Segmentation

| Type | Input → Output | Description |
|------|---------------|-------------|
| **Semantic Segmentation** | Image → Per-pixel class label | Each pixel gets a class; no distinction between instances (e.g., all "dogs" are the same label) |
| **Instance Segmentation** | Image → Per-instance pixel mask | Each object instance gets a unique mask |
| **Panoptic Segmentation** | Image → Unified segmentation | Combines semantic (stuff: sky, grass) + instance (things: car, person) |

### FCN (Fully Convolutional Network, 2015)

Replace FC layers with 1×1 convolutions → output a spatial heatmap instead of a single class score. Upsample to input resolution.

### U-Net

Encoder-decoder with **skip connections** between corresponding encoder and decoder layers → preserves fine spatial details.

---

## 6. Visualizing and Understanding CNNs

### Feature Visualization

**What does a neuron detect?** Find input images that maximally activate that neuron through gradient ascent on the input.

- **Layer 1:** Gabor-like edges and color blobs
- **Middle layers:** Textures, patterns
- **Deep layers:** Object parts, complex structures

### Nearest Neighbors in Feature Space

Instead of pixel-space neighbors (which are meaningless), find images whose **CNN feature vectors** are close — they share semantic content even if visually different.

### Saliency Maps

Compute the gradient of the class score with respect to input pixels → highlights which pixels most influence the prediction.

### Adversarial Examples

Adding a small, imperceptible perturbation to an image can cause the model to misclassify it with high confidence. This reveals that CNNs learn decision boundaries that are very different from human perception.

### DeepDream & Style Transfer

- **DeepDream:** Amplify features detected at a chosen layer by gradient ascent → hallucinatory imagery
- **Neural Style Transfer:** Match content (from one image) with style (from another) by optimizing a combined loss

---

## Core Takeaways

1. **ViT** applies standard NLP Transformers to images by treating patches as tokens — proving the architecture's generality.
2. Object detection evolved from slow two-stage methods (R-CNN → Faster R-CNN) to fast single-stage detectors (YOLO) and Transformer-based approaches (DETR).
3. Segmentation tasks form a hierarchy: **semantic** (per-pixel class) → **instance** (per-object mask) → **panoptic** (unified).
4. Visualization techniques reveal that CNNs learn a **hierarchy** of features (edges → textures → parts) and are vulnerable to adversarial perturbations.
5. **Attention maps** in Transformers and saliency maps in CNNs provide interpretability of model decisions.

---

*Notes compiled from CS231n Spring 2025 Lecture 9 slides and course materials.*
