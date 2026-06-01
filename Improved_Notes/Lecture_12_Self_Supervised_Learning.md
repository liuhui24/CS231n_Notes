# CS231n (Spring 2025) — Lecture 12: Self-Supervised Learning

> **Instructor:** Ehsan Adeli
> **Date:** May 8, 2025

---

## 1. The Labeling Bottleneck

### The Problem

Training deep neural networks requires massive labeled datasets. Manually labeling millions of images is:
- **Expensive** and time-consuming
- **Non-scalable** to new domains
- **Biased** by annotator perspective

**Is there a way to train neural networks without huge manually labeled datasets?**

### Learned Representations Already Work

Recall: Nearest neighbors in CNN **feature space** find semantically similar images, even when pixel-space neighbors are meaningless. This suggests the learned representation captures something fundamental — can we learn such representations without labels?

---

## 2. Self-Supervised Learning: The Framework

### Pretext Task + Downstream Task

**Pretext Task:**
- A task defined purely from the data itself
- Requires **no manual annotation**
- Labels are automatically generated from the data
- The model learns by solving this task with standard supervised objectives (classification, regression)

**Downstream Task:**
- The actual application you care about
- You typically have a **small labeled dataset** for it
- Transfer the features learned from the pretext task

### Pipeline

```
Unlabeled Data → Encoder → Learned Representation → Decoder/Classifier → Auto-generated Labels
                                                                            (from data itself)

Then:

Encoder (frozen/pretrained) → Shallow Classifier → Small Labeled Dataset → Downstream Task
```

---

## 3. Pretext Tasks from Image Transformations

The core idea: **transform** the image in a known way, and train the model to predict the transformation.

| Pretext Task | Description | What the Model Learns |
|-------------|-------------|----------------------|
| **Rotation Prediction** | Rotate image by 0°, 90°, 180°, or 270°; predict which rotation was applied (4-way classification) | "Visual commonsense" — objects have canonical orientations |
| **Image Inpainting** | Mask out a region; predict the missing pixels | Object structure and contextual reasoning |
| **Jigsaw Puzzle** | Shuffle image patches; predict the correct arrangement | Spatial relationships between object parts |
| **Colorization** | Grayscale input → predict color channels | Semantic understanding of object colors (sky is blue, grass is green) |

### Rotation Prediction Example (Gidaris et al., 2018)

A model can recognize the correct rotation of an object only if it has learned what the object should look like unperturbed — this requires understanding the object's typical appearance, not just memorizing pixels.

**Evaluation:** Pretrain with rotation prediction on full CIFAR-10 (no labels used). Freeze conv1 + conv2 layers. Train conv3 + linear classifier on a **subset** of labeled CIFAR-10 data. Performance approaches fully supervised training.

---

## 4. Contrastive Representation Learning

### Core Intuition

- **Pull together:** Different augmentations of the **same** image should have similar features
- **Push apart:** Features of **different** images should be dissimilar

### SimCLR (Chen et al., 2020)

1. Take a batch of images
2. Apply **random augmentations** to each image twice → two views per image
3. Extract features via encoder
4. **Contrastive loss:** Maximize similarity between views of the same image; minimize similarity between views of different images

**Problem:** Requires **large batch size** to have enough negative samples. The contrastive signal improves with more negatives in the batch.

### MoCo (He et al., 2020)

- Maintain a **running queue** of keys (negative samples) — decouples batch size from number of negatives
- **Momentum encoder:** The key encoder is updated via exponential moving average (slowly), not gradient descent
- Gradients flow only through the query encoder → stable training with many negatives

### DINO (Caron et al., 2021)

- Uses **Vision Transformers** instead of ResNets
- Matches features using **KL divergence** (not dot product)
- Teacher-student framework with momentum update
- **Emergent property:** Attention maps naturally segment objects without any segmentation supervision

### DINOv2 (Oquab et al., 2023)

- Scales training from 1M (ImageNet) to **142M images**
- Produces very strong, general-purpose image features widely used in practice
- PCA visualization reveals clean feature clustering

---

## 5. How to Evaluate Self-Supervised Methods

| Evaluation Method | Description |
|-------------------|-------------|
| **Pretext Task Performance** | How well the model solves the pretext task itself |
| **Linear Evaluation** | Freeze encoder → train linear classifier on labeled data → measure accuracy |
| **Clustering** | Measure how well representations cluster by class |
| **t-SNE Visualization** | Visual inspection of representation separability |
| **Transfer Learning** | Fine-tune on downstream tasks (detection, segmentation) |
| **Robustness** | Test generalization to different distributions |

---

## 6. Reconstruction-Based Learning: MAE

**Masked Autoencoder (He et al., 2022):**
- Mask a large portion (~75%) of image patches
- Train an encoder-decoder to reconstruct the missing patches
- **Asymmetric design:** Encoder only sees unmasked patches (efficient); decoder (lightweight) reconstructs from encoded features + mask tokens
- Very effective for Vision Transformer pretraining

---

## Core Takeaways

1. Self-supervised learning eliminates the labeling bottleneck by creating **pretext tasks** from the data itself.
2. Two main paradigms: **pretext task prediction** (rotation, jigsaw, inpainting) and **contrastive learning** (SimCLR, MoCo, DINO).
3. Contrastive learning pulls same-image views together and pushes different images apart — the number and quality of negatives matter.
4. **MoCo** uses momentum encoder + queue; **DINO** uses ViT + KL divergence, with emergent segmentation abilities.
5. Evaluation is done via **linear probing** or **transfer learning** on downstream tasks — the quality of the learned representation is what matters.

---

*Notes compiled from CS231n Spring 2025 Lecture 12 slides and course materials.*
