# CS231n (Spring 2025) — Lecture 16: Multi-Modal Foundation Models

> **Instructor:** Ranjay Krishna
> **Date:** May 27, 2025

---

## 1. From Task-Specific Models to Foundation Models

### The Old Paradigm

Train a **specialized model** for each task on each domain:

```
Domain 1 → Model 1 → Task 1
Domain 2 → Model 2 → Task 2
Domain 3 → Model 3 → Task 3
...
```

### The Foundation Model Paradigm

**Pretrain one model** on large-scale diverse data, then adapt it to many downstream tasks:

```
Large-Scale Diverse Data → Foundation Model → Task 1, Task 2, Task 3, Task 4
                            (Pretraining)    (Fine-tuning / Zero-shot / Few-shot)
```

### What Makes a Foundation Model?

| Always present | Often present |
|----------------|---------------|
| General / robust to many different tasks | Large number of parameters |
| | Large amount of training data |
| | Self-supervised pretraining objective |

---

## 2. The Multimodal Foundation Model Landscape

| Category | Examples |
|----------|----------|
| **Language Models** | ELMo, BERT, GPT, T5 |
| **Classification (Vision-Text)** | **CLIP**, CoCa |
| **LM + Vision (VLMs)** | **LLaVA**, Flamingo, GPT-4V, Gemini, Molmo |
| **Other** | Segment Anything (SAM), Whisper, DALL-E, Stable Diffusion, Imagen |
| **Chaining** | LMs + CLIP, Visual Programming |

This lecture focuses on **vision-language** foundation models.

---

## 3. CLIP: Contrastive Language-Image Pretraining

### From SimCLR to CLIP

**SimCLR:** Image encoder + contrastive loss → pulls together different augmentations of the same image, pushes apart different images.

**CLIP:** Replace one image encoder with a **text encoder**. Now:
- **Positive pair:** An image and its corresponding caption
- **Negative pairs:** All other image-text combinations in the batch

### Architecture

```
Image → Image Encoder → Image Vector  ┐
                                       ├── Contrastive Loss (pull matching, push non-matching)
Text  → Text Encoder  → Text Vector   ┘
```

### Training Data

400 million **(image, alt-text)** pairs scraped from the internet. No human annotation needed — alt-text is naturally occurring supervision.

### Zero-Shot Classification

CLIP enables classification on **any** dataset without retraining:

1. Create a text vector for each possible category by encoding prompts: "A photo of a [category]"
2. Encode the test image
3. Find the most similar text vector → predict that category

This is essentially **1-Nearest Neighbor** in a learned joint embedding space, where the "training set" is the text embeddings of category names.

### Prompt Engineering Matters

| Prompt Strategy | ImageNet Gain |
|-----------------|---------------|
| Just the category name | Baseline |
| "A photo of a [category]" | +1.3% |
| Ensemble of multiple prompts per category (average embeddings) | +5% |

### CLIP's Remarkable Results

- **Matches** the accuracy of a ResNet-101 **fully supervised** on ImageNet — using **zero human labels**
- Far more **robust** to distribution shifts (ObjectNet with unusual viewpoints, sketches, graphic images, adversarial datasets)

### Why Does CLIP Work So Well?

**Scale.**

| | ResNet on ImageNet | CLIP |
|------|--------------------|------|
| **Parameters** | 44.5 Million | 307 Million (Transformer) |
| **Training Images** | 1.28 Million (labeled) | 400 Million (image-text pairs) |

More data + larger model → better representations, even without manual labels.

### CoCa: Adding Generation

CoCa (Contrastive Captioners) extends CLIP by adding a **decoder** with a captioning loss — the model learns both contrastive matching and text generation.

---

## 4. CLIP's Limitations

| Limitation | Description |
|------------|-------------|
| **Reliance on large batches** | Fine-grained concepts (e.g., "Welsh Corgi" vs. "dog") require very large batch sizes to have enough contrastive signal |
| **Compositionality failures** | Cannot distinguish "a mug in some grass" from "some grass in a mug" — the bag-of-words nature of contrastive pretraining ignores word order |
| **Image-level supervision is coarse** | An image caption "living room" doesn't tell the model which pixels correspond to the couch vs. the plant |
| **Data quality** | Web-scraped data contains biases, noise, and harmful content — intentional data curation and filtering is critical |

---

## 5. Vision-Language Models (VLMs): LLaVA

### Motivation

Language models (LLMs) trained on next-token prediction can be applied to a wide variety of downstream tasks zero-shot. Can we build a model that accepts **images and text** as input and outputs **text**?

### Historical Context

Vision-Language Models didn't start with LLaVA — ViLBERT (2019) was an early example. But earlier VLMs required **task-specific fine-tuning** with non-trivial methods for each task (e.g., Mask-RCNN bounding box re-ranking for referring expression comprehension).

### LLaVA's Key Idea

Add visual information to an LLM by prepending visual tokens:

```
Image → Vision Encoder → [V1, V2, ..., VN] → [Cats, are, ...] → LLM → "Cute" → "so"
        (e.g., CLIP)    Visual Tokens     Text Tokens          Autoregressive generation
```

The LLM's autoregressive nature enables:
- **Visual question answering:** "What color is the cat?"
- **Image captioning:** Describe the image
- **Visual reasoning:** "Is the person in the photo likely to be cold?"
- **Conversation about images:** Multi-turn dialogue with visual context

### The Paradigm Shift

Unlike earlier VLMs that required task-specific architectures and fine-tuning procedures, LLaVA-style models handle diverse visual tasks through a **unified text generation interface** — the same way LLMs handle diverse language tasks.

---

## Core Takeaways

1. **Foundation models** replace task-specific models: pretrain once on large-scale data, adapt to many downstream tasks through zero-shot, few-shot, or fine-tuning.
2. **CLIP** bridges vision and language through contrastive pretraining on 400M image-text pairs — achieving zero-shot classification competitive with supervised models.
3. CLIP's success is driven by **scale** — more parameters and more data, even without manual labels.
4. CLIP struggles with **compositionality** and fine-grained concepts — batch-based contrastive learning has inherent limits.
5. **LLaVA** and similar VLMs combine vision encoders with LLMs, enabling visual understanding through a unified text-generation interface.

---

*Notes compiled from CS231n Spring 2025 Lecture 16 slides and course materials.*
