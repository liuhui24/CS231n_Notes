# CS231n (Spring 2025) — Lecture 11: Large-Scale Distributed Training

> **Instructor:** Justin Johnson
> **Date:** May 6, 2025

---

## 1. Motivation: Training at Scale

**Running Example: Llama 3 (405B parameters)**

In contrast to GPT-4's closed approach ("no further details about architecture, hardware, or training"), Meta's Llama 3 (April 2024) provides open access to model weights and publishes extensive training details.

The goal of this lecture: understand the hardware and algorithms that make training models at this scale possible.

---

## 2. GPU Hardware Deep Dive: NVIDIA H100

### H100 Die

| Component | Specification |
|-----------|---------------|
| **GPU Memory** | 80 GB HBM (High Bandwidth Memory) |
| **Memory Bandwidth** | 3,352 GB/sec to compute cores |
| **L2 Cache** | 50 MB |
| **Streaming Multiprocessors (SMs)** | 132 (144 physically, 132 enabled for yield) |

### H100 Streaming Multiprocessor (SM)

Each SM is analogous to a CPU core with vector instructions:

| SM Component | Specification |
|--------------|---------------|
| **L1 Cache** | 256 KB |
| **Registers** | 256 KB |
| **FP32 Cores** | 128 — each computes $a \times x + b$ per clock cycle (2 FLOPs) → 256 FLOP/cycle per SM |
| **Tensor Cores** | 4 — each computes $AX + B$ per clock cycle as a matrix operation $[16 \times 4] \times [4 \times 8] + [16 \times 8]$ → 1,024 FLOPs → 4,096 FLOP/cycle per SM |
| **Mixed Precision** | Supports FP16/BF16 compute with FP32 accumulate |

**Why Tensor Cores matter:** For matrix operations (the core of deep learning), Tensor Cores provide ~16× more throughput than FP32 cores.

### GPU Evolution: 1000× Speedup Since 2013

| GPU | Year | FP32 TFLOPS | Tensor Core TFLOPS |
|-----|------|-------------|---------------------|
| K40 | 2013 | 5 | — |
| P100 | 2016 | 10.6 | — |
| V100 | 2017 | 15.7 | 125 |
| A100 | 2020 | 19.5 | 312 |
| H100 | 2022 | 67 | 989 |
| B200 | 2024 | 83.3 | 5,000 |

A ~1,000× speedup in just over 10 years. And we can also use **more than 1 GPU**.

---

## 3. From GPU to Cluster

### Bandwidth Hierarchy

| Level | Configuration | Interconnect Speed |
|-------|--------------|-------------------|
| **GPU** | 1 × H100 | 3,352 GB/sec (internal) |
| **Server (Node)** | 8 × GPU | ~900 GB/sec between GPUs |
| **Rack** | 2 servers = 16 GPUs | — |
| **Pod** | 192 racks = 3,072 GPUs | ~50 GB/sec between GPUs |
| **Cluster** | 8 pods = 24,576 GPUs | < 50 GB/sec between GPUs |

### Llama 3 Cluster Stats (Meta)

| Metric | Value |
|--------|-------|
| **Total GPUs** | 24,576 H100s |
| **GPU Memory** | 1.875 PB (Petabytes) |
| **FP32 Cores** | 415 Million |
| **Tensor Cores** | 13 Million |
| **Peak Compute** | 24.3 EFLOP/sec ($24.3 \times 10^{18}$ operations per second) |

**The bandwidth hierarchy drives everything:** data movement is the bottleneck. Computation is cheap; communication is expensive.

---

## 4. Parallelism Strategies

### Data Parallelism

- Each GPU holds a **full copy** of the model
- Data is split across GPUs (each processes a different minibatch)
- After each forward/backward pass: **all-reduce** to average gradients
- Implication: model must fit on a single GPU's memory

### Model Parallelism

- The model is **split across GPUs** — different layers on different devices
- Each GPU processes its portion for every input
- Pipeline parallelism: micro-batches flow through the pipeline
- Enables training models larger than any single GPU's memory

### Hybrid Approaches (3D Parallelism)

Large models (like Llama 3-405B) combine:
- **Data parallelism** across pods
- **Pipeline model parallelism** within pods
- **Tensor parallelism** (split individual layers) within a server

### Synchronous vs. Asynchronous Updates

| Mode | How It Works | Tradeoff |
|------|-------------|----------|
| **Synchronous** | All workers finish before averaging gradients | Consistent gradients, but stragglers bottleneck |
| **Asynchronous** | Workers update independently | Faster iteration, but gradient staleness |

Modern large-scale training almost always uses **synchronous** updates.

---

## 5. Mixed Precision Training

Training in FP16/BF16 reduces memory and increases throughput. Key techniques:

- **FP16 forward + backward:** Compute in half precision for speed
- **FP32 master weights:** Keep a master copy in FP32 for numerical stability
- **Loss scaling:** Multiply loss by a large factor before backward to prevent gradient underflow (FP16's limited dynamic range)

---

## 6. Fault Tolerance

At 24,576 GPUs, failures are **expected**, not exceptional:
- GPUs fail, network links drop, storage degrades
- **Checkpointing:** Periodically save model state + optimizer state
- **Resume from checkpoint:** Resume training from the last good state
- Meta reported ~90% uptime during Llama 3 training

---

## Core Takeaways

1. Modern GPU clusters are **massive parallel computers**: 24K+ GPUs, petabytes of memory, exaFLOPs of compute.
2. Communication bandwidth falls off dramatically across the hierarchy — **data locality** is critical.
3. Three parallelism strategies: **data** (split data), **model** (split layers), **tensor** (split individual operations). Large models use all three.
4. GPUs have gotten **1,000× faster** in 10 years, largely due to Tensor Cores specialized for matrix operations.
5. At scale, **fault tolerance** and **efficient communication** are as important as the model architecture itself.

---

*Notes compiled from CS231n Spring 2025 Lecture 11 slides and course materials.*
