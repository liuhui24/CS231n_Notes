# CS231n (Spring 2025) — Lecture 1: Introduction

> **Instructors:** Fei-Fei Li, Ehsan Adeli, Zane Durante, Justin Johnson
> **Guest Lecturers:** Jiajun Wu, Ranjay Krishna, Ruohan Gao, Yunzhu Li
> **Date:** April 1, 2025
> **Note:** CS231n 10th Anniversary

---

## 1. Course Positioning

CS231n sits at the intersection of **Computer Vision**, **Machine Learning**, and **Deep Learning** — all within Artificial Intelligence.

The field draws from multiple disciplines: Mathematics, Neuroscience, Computer Science, Physics, Biology, and Psychology.

---

## 2. A Brief History of Computer Vision

### Biological Origins

- **Cambrian Explosion (~530-540 million years ago):** The emergence of vision is hypothesized to have triggered the rapid evolution of species diversity. Vision allowed organisms to actively perceive and interact with their environment.

### Early Optical Devices

- **Camera Obscura (16th century):** Documented by Leonardo da Vinci and later Gemma Frisius (1545). The pinhole camera principle — light passing through a small hole projects an inverted image — laid the physical foundation for modern cameras.

### The Birth of Computational Vision

| Year | Milestone |
|------|-----------|
| **1959** | **Hubel & Wiesel** — Recorded neural activity in cat visual cortex. Discovered **simple cells** (respond to specific edge orientation/position) and **complex cells** (respond to motion, exhibit translation invariance). This hierarchical organization directly inspired layered neural networks. |
| **1963** | **Larry Roberts** — "Machine Perception of Three-Dimensional Solids." Extracted edges from images to reconstruct 3D block structures. The first explicit attempt at computational visual understanding. |
| **1966** | MIT AI Lab launched a summer project to solve computer vision — vastly underestimated the problem's difficulty. |

### The Deep Learning Era

- **1998:** LeCun's CNN for handwritten digit recognition (banking/postal systems)
- **2009:** ImageNet released — millions of labeled images, 1000 categories
- **2012:** AlexNet won ImageNet by a large margin — the **turning point** that launched the deep learning revolution in computer vision

---

## 3. CS231n Course Overview: Four Themes

### Theme 1: Deep Learning Basics
- Image Classification → Linear Classifier → Regularization & Optimization → Neural Networks
- These are the foundational building blocks for everything that follows.

### Theme 2: Perceiving and Understanding the Visual World

**Tasks beyond image classification:**

| Task | Description |
|------|-------------|
| **Semantic Segmentation** | Per-pixel class labels (no object distinction) |
| **Object Detection** | Bounding boxes + class labels |
| **Instance Segmentation** | Per-instance pixel masks |
| **Video Classification** | Action/event labels from video |
| **Multimodal Video Understanding** | Joint visual + audio understanding |
| **Visualization & Understanding** | What models actually learn |

**Models beyond MLP:**
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Attention Mechanisms / Transformers

### Theme 3: Generative and Interactive Visual Intelligence

- **Self-supervised Learning:** Learning without manual labels
- **Generative Modeling:** Style transfer, text-to-image (DALL-E 2), diffusion models
- **Vision-Language Models:** CLIP (contrastive pre-training across image-text pairs)
- **3D Vision:** Reconstructing and understanding 3D structure

### Theme 4: Large-Scale Training & Human-Centered AI

- **Distributed Training:**
  - Data Parallelism: copy model to all workers, split data
  - Model Parallelism: split model across devices
  - Synchronous vs. asynchronous gradient updates
- **Human-Centered Applications & Ethical Implications**

---

## 4. Assignment 3 Highlight

In Assignment 3, students implement a **diffusion-based generative model** that generates emojis from text inputs — connecting theory to hands-on practice with modern generative AI.

---

## Core Takeaways

1. Computer vision evolved from biological inspiration (Hubel & Wiesel) through hand-crafted features to end-to-end deep learning.
2. CS231n covers the full pipeline: basic classifiers → perception tasks → generative models → human-centered implications.
3. The 2012 AlexNet moment marked the transition to deep learning dominance — the field has not looked back since.
4. Modern CV is increasingly multimodal: vision + language, generation, and 3D understanding.

---

*Notes compiled from CS231n Spring 2025 Lecture 1 slides (Parts 1 & 2) and course materials.*
