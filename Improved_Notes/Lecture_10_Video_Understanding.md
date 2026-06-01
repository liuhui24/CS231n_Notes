# CS231n (Spring 2025) — Lecture 10: Video Understanding

> **Instructor:** Ruohan Gao
> **Date:** May 1, 2025

---

## 1. Video = 2D + Time

A video is a sequence of images: a **4D tensor** of shape $T \times 3 \times H \times W$ (or $3 \times T \times H \times W$).

| Images | Videos |
|--------|--------|
| Recognize objects (dog, cat, truck) | Recognize **actions** (swimming, running, jumping) |
| Static scene understanding | Temporal dynamics, motion, causality |

---

## 2. The Scale Problem

Videos are typically ~30 frames per second (fps). Uncompressed size (3 bytes per pixel):

| Resolution | Size per Minute |
|------------|-----------------|
| SD (640 × 480) | ~1.5 GB |
| HD (1920 × 1080) | ~10 GB |

**Solution:** Train on **short clips** with low fps and low spatial resolution — e.g., $T = 16$, $H = W = 112$ (~3.2 seconds at 5 fps, ~588 KB).

**Evaluation:** Run the model on multiple clips from the same video and **average predictions**.

---

## 3. Video Classification Architectures

### (a) Single-Frame CNN (Baseline)

Train a standard 2D CNN to classify each frame independently. At test time, average predicted probabilities across all frames.

> This is often a **surprisingly strong baseline** — static appearance alone captures a lot of signal.

### (b) Late Fusion (with FC Layers)

1. Run 2D CNN on each frame independently → per-frame feature maps
2. Concatenate frame features: $T \times D \times H' \times W'$ → flatten → MLP → class scores

**Limitation:** Features are combined only at the highest level → hard to compare low-level motion between frames.

### (c) Late Fusion (with Pooling)

Same as above, but **average-pool** features across space and time before a linear classifier. Still suffers from the same late-combination limitation.

### (d) Early Fusion

Reshape the first convolutional layer's input to process all frames at once:

- Input: $T \times 3 \times H \times W$ → reshape → $3T \times H \times W$
- First conv collapses all temporal information immediately
- Rest of network is standard 2D CNN

**Limitation:** One layer of temporal processing may not be sufficient.

### (e) 3D CNN

Use **3D convolution and 3D pooling** throughout the entire network:

- Each layer operates on 4D tensors: $D \times T \times H \times W$
- 3D filters (e.g., $3 \times 3 \times 3$) slide in all three dimensions
- Temporal information is **gradually fused** as the network deepens

This is the most natural extension of 2D CNNs to video — but 3D convolutions are computationally expensive.

### (f) Two-Stream Networks

Combine:
- **Spatial stream:** Processes individual RGB frames (appearance)
- **Temporal stream:** Processes **optical flow** (explicit motion) computed between consecutive frames

Fuse predictions from both streams. Optical flow provides strong motion cues, but computing it is expensive.

---

## 4. Beyond Simple Classification

### Multimodal Video Understanding

Videos contain:
- **Visual** information (frames)
- **Audio** information (speech, environmental sounds)
- **Text** information (subtitles, metadata)

Joint modeling of these modalities enables richer understanding — e.g., knowing that a sizzling sound + frying pan visual = "cooking," even if no one is visible.

### Temporal Action Localization

Beyond classifying the whole video: detect **when** specific actions start and end within a long untrimmed video.

---

## Core Takeaways

1. Video = 2D images + time. The key challenge is **modeling temporal dynamics** efficiently.
2. Videos are **huge** → train on short clips with low resolution and fps.
3. A **single-frame 2D CNN** is a strong baseline — appearance alone is highly informative.
4. Early fusion (process frames together from the start) vs. late fusion (combine high-level frame features) vs. **3D CNNs** (gradually fuse temporal information throughout the network).
5. **3D CNNs** are the natural extension of 2D CNNs, but expensive. Two-stream networks with optical flow provide an alternative.

---

*Notes compiled from CS231n Spring 2025 Lecture 10 slides and course materials.*
