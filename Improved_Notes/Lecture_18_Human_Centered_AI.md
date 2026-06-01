# CS231n (Spring 2025) — Lecture 18: Human-Centered AI

> **Instructor:** Fei-Fei Li
> **Date:** June 3, 2025

---

## 1. Three Guiding Questions

This final lecture reframes computer vision through a human-centered lens:

1. **Seeing what humans see** — Replicating human perceptual capabilities
2. **Seeing what humans do not see** — Going beyond human perception
3. **Seeing what humans want to see** — Connecting perception to human-valued action

> Technological progress should be measured not just by algorithmic metrics, but by alignment with human values, privacy preservation, bias mitigation, and real-world utility.

---

## 2. Biological & Historical Foundations

### Vision as the Primary Sense
- Vision appeared early in animal evolution, catalyzing ecological diversification (Cambrian Explosion)
- Humans achieve **~150ms category-level distinction** (animal vs. non-animal) — remarkably fast
- Specialized neuroanatomical regions: FFA (faces), PPA (places), EBA (bodies)
- Humans recognize **tens of thousands** of visual categories → large-scale datasets needed

### The Deep Learning Revolution
- **ImageNet + CNN + GPU** → qualitative leap in object recognition (2012 milestone)
- Deep ConvNets automatically learn hierarchical features from scale
- **Remaining challenges:** Long-tail rare categories, domain shift, challenging conditions (occlusion, lighting, weather)

---

## 3. Beyond Object Recognition: Scenes and Relationships

- **Scene Graphs:** Objects as nodes, relationships (spatial, action, attribute) as edges → compositional representations
- **Visual Genome, Dense Captioning:** Combine detection with language models for narrative scene understanding
- **Dynamic Activity Understanding:** Requires temporal reasoning — *who is doing what, when* — remains largely unsolved
- Human visual understanding is inherently **relational** and **contextual**, not just a bag of objects

---

## 4. Beyond Human Perception (Superhuman Vision)

| Capability | Example |
|------------|---------|
| **Fine-Grained Recognition** | Distinguishing thousands of bird species, car makes/model years |
| **Population-Scale Analysis** | Street-view imagery → socioeconomic patterns, voting behavior, environmental signals |
| **Medical Imaging** | Detecting subtle patterns in radiology that human eyes miss |

These capabilities act as **sociological lenses**, but raise important ethical questions about surveillance, consent, and appropriate use.

---

## 5. Human Perceptual Limitations & High-Stakes Failures

- **Stroop interference** and **change blindness** reveal fundamental limits of human attention
- In healthcare: these limitations contribute to diagnostic and procedural errors — **medical error is a leading cause of death**
- Manual mitigation (checklists, auditors) is fragile and subject to fatigue
- **Automated visual monitoring** can provide consistent alerts, but requires careful validation, privacy preservation, and clinical integration

---

## 6. Bias, Fairness, and Privacy

### Bias in Vision Systems

- Training data encodes societal biases → **disparate performance across demographic groups**
- Historically problematic in facial recognition applications
- Requires: detection (auditing), measurement (disaggregated evaluation), and mitigation strategies

### Privacy-Preserving Techniques

| Technique | Description |
|-----------|-------------|
| **Blurring & Masking** | Obscure identities in imagery |
| **Dimensionality Reduction** | Compress away sensitive attributes |
| **Federated Learning** | Train without centralizing data |
| **Encryption** | Protect data in transit and at rest |
| **Hybrid HW/SW** | Filter raw imagery at capture time (edge processing) |

**Goal:** Preserve task-relevant information (e.g., activity recognition) while obscuring identity.

---

## 7. AI as Augmentation (Not Replacement)

### Healthcare & Ambient Intelligence

- **Ambient Intelligence:** Distributed sensors + ML for continuous, privacy-aware monitoring
- **Hand Hygiene Monitoring:** Depth sensors → classify actions more reliably than sparse human audits → reduce hospital-acquired infections
- **ICU Monitoring:** Track patient mobility, predict risk of deterioration
- **Aging in Place:** Thermal sensing, mobility tracking, sleep/eating monitoring → early infection detection in elderly care settings

### Closing the Perception-Action Loop

```
LLM translates human instruction → Program code / action plan
VLM detects objects, handles, affordances → Grounds the plan
Semantic instructions → Motion planning heatmaps (safe/goal regions)
Parameterized execution (gripper rotation, velocity profiles)
```

This modular approach enables **compositional generalization**, reducing dependence on closed-world pretraining.

---

## 8. Ecological Robotics Benchmarks

### Human-Centered Task Selection
- Solicit community input on what tasks people actually want robots to help with at home
- Aggregate thousands of everyday activities into a prioritized task list
- Examples: folding laundry, loading dishwasher, tidying rooms

### Realistic Benchmark Requirements
- High-fidelity virtual environments from real-world scans
- Large sets of articulated/deformable 3D objects
- Physically accurate simulation (cloth, liquids, transparency)
- Platforms like **NVIDIA Omniverse** provide the simulation infrastructure

### Reality Check

> Under **no privileged information**, the success rate of general-purpose robots on everyday household tasks approaches **zero**. Performance only improves with privileged assumptions — exposing fundamental brittleness in current approaches.

---

## 9. Brain-Computer Interfaces (BCI)

- **Non-invasive BCI** (EEG) + pretrained action primitives → **thought-driven robot control**
- Demonstration: EEG-triggered primitive sequences to complete complex tasks like meal preparation
- **Requirements:** Supervised calibration, per-user/domain primitive-level pretraining
- **Impact:** A pathway toward assistive robotics for individuals with severe paralysis

---

## 10. Course Summary: CS231n Spring 2025

CS231n 2025 traced a journey from fundamentals to frontiers:

```
Deep Learning Basics           Perceiving & Understanding       Generative & Interactive
(Linear Classifiers → CNNs     (Classification, Detection,      (VAE, GAN, Diffusion,
 → Transformers)               Segmentation, Video)             Self-Supervised Learning)
                                                                      ↓
                                                         Human-Centered AI
                                                   (3D Vision, Vision-Language,
                                                   Robot Learning, Ethics & Fairness)
```

### Guiding Principles

| Principle | Description |
|-----------|-------------|
| **Human-Centered Values** | Guide development through fairness/privacy measurement, transparent limitation reporting, cross-disciplinary engagement |
| **Augmentation Over Replacement** | Prioritize systems that extend human capabilities, especially in areas of workforce shortage |
| **Safety-First Deployment** | Continuous collaboration between technical teams and domain stakeholders |
| **Open Research Problems** | Long-tail recognition, robust sim-to-real transfer, generalization without privileged information, bias auditing |

### Core Thread

**From pixels to understanding. From perception to action. From tools to augmenting human capability.**

---

*Notes compiled from CS231n Spring 2025 Lecture 18 and course materials.*
