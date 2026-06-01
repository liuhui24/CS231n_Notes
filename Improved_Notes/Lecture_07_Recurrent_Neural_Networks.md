# CS231n (Spring 2025) — Lecture 7: Recurrent Neural Networks

> **Instructor:** Zane Durante
> **Date:** April 22, 2025

---

## 1. Beyond Fixed-Size Inputs

All models so far (linear classifier, MLP, CNN) assume **fixed-size inputs** producing **fixed-size outputs**. But many real-world problems involve **sequences** of varying length.

### Sequence Task Paradigms

| Paradigm | Input → Output | Example |
|----------|---------------|---------|
| **One-to-One** | Fixed → Fixed | Image classification |
| **One-to-Many** | Fixed → Sequence | Image → Caption |
| **Many-to-One** | Sequence → Fixed | Video → Action label, Sentiment analysis |
| **Many-to-Many (aligned)** | Sequence → Sequence (same length) | Per-frame video classification |
| **Many-to-Many (unaligned)** | Sequence → Sequence (different length) | Machine translation |

---

## 2. Recurrent Neural Networks: Core Idea

RNNs have an **"internal state"** that is updated as each element of the sequence is processed.

### The Recurrence Formula

$$h_t = f_W(h_{t-1}, x_t)$$

- $h_t$: new hidden state
- $h_{t-1}$: previous hidden state
- $x_t$: current input vector
- $f_W$: function with parameters $W$ — **same function, same parameters at every time step**

### Output Generation

$$y_t = f_{W_{hy}}(h_t)$$

### Unrolled View

```
h0 → RNN → h1 → RNN → h2 → RNN → h3 → ... → hT
      ↑           ↑           ↑              ↑
     x1          x2          x3             xT
      ↓           ↓           ↓              ↓
     y1          y2          y3             yT
```

Unrolling across time reveals that an RNN is a deep feed-forward network **with shared weights across layers**.

---

## 3. Vanilla RNN

The simplest form (also called "Elman RNN"):

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

| Matrix | Role |
|--------|------|
| $W_{xh}$ | Input → Hidden |
| $W_{hh}$ | Hidden → Hidden (the recurrent connection) |
| $W_{hy}$ | Hidden → Output |

### Concrete Example: Detecting Repeated 1s

**Task:** For a binary input sequence, output 1 only when two consecutive 1s appear.

```
Input:  0 1 0 1 1 1 0 1 1
Output: 0 0 0 0 1 1 0 0 1
```

The hidden state must capture: "was the previous input a 1?" This shows how even a simple RNN state serves as a form of memory.

---

## 4. Training RNNs: Backpropagation Through Time (BPTT)

The loss is computed over all time steps, and gradients flow backward through the entire unrolled computation graph:

$$L = \sum_{t=1}^T L_t(y_t, \text{target}_t)$$

**Truncated BPTT:** For very long sequences, backpropagate through a fixed-size window rather than the full sequence — reduces memory and computation.

---

## 5. The Vanishing/Exploding Gradient Problem

When backpropagating through many time steps, gradients are multiplied by the recurrent weight matrix $W_{hh}$ repeatedly:

| Condition | Result |
|-----------|--------|
| Largest singular value of $W_{hh}$ < 1 | **Gradient vanishes** — early inputs cannot influence later outputs |
| Largest singular value of $W_{hh}$ > 1 | **Gradient explodes** — training becomes unstable |

**Solutions:**
- **Gradient explosion → Gradient clipping:** Scale gradients if their norm exceeds a threshold
- **Gradient vanishing → Architecture change:** LSTM, GRU

### Long-Range Dependency Example

> "The **cat**, which already ate several fish that were bought at the market this morning, **was** full."

A vanilla RNN struggles to connect "cat" (early) with "was" (late) — the gradient signal decays exponentially across the intervening words.

---

## 6. LSTM (Long Short-Term Memory)

LSTMs introduce a separate **cell state** $c_t$ that provides an additive gradient highway through time.

### Four Gates

| Gate | Formula | Purpose |
|------|---------|---------|
| **Forget Gate** $f_t$ | $\sigma(W_f[h_{t-1}, x_t] + b_f)$ | What to erase from memory |
| **Input Gate** $i_t$ | $\sigma(W_i[h_{t-1}, x_t] + b_i)$ | What to write to memory |
| **Candidate** $g_t$ | $\tanh(W_g[h_{t-1}, x_t] + b_g)$ | New candidate values |
| **Output Gate** $o_t$ | $\sigma(W_o[h_{t-1}, x_t] + b_o)$ | What to expose from memory |

### Cell State Update

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = o_t \odot \tanh(c_t)$$

**Why this helps:** The cell state update is **additive** — when $f_t \approx 1$, information passes through unchanged. Gradients flow backward through the cell state without being repeatedly multiplied, analogous to ResNet's skip connections.

### GRU (Gated Recurrent Unit)

A simplified LSTM with only 2 gates (reset + update) and no separate cell state. Similar performance, fewer parameters.

---

## 7. RNN Limitations and the Path to Transformers

| RNN Strengths | RNN Weaknesses |
|---------------|----------------|
| Theoretically infinite context | **Sequential computation** — cannot parallelize across time |
| Parameter-efficient (shared weights) | Hidden state is a fixed-size bottleneck |
| Naturally handles variable-length sequences | Training instability (vanishing/exploding) |

The sequential nature of RNNs — each hidden state must be computed before the next can begin — is the fundamental bottleneck that motivates **Transformers** (Lecture 8).

---

## Core Takeaways

1. RNNs process sequences by maintaining and updating an **internal hidden state** through a recurrence relation with **shared parameters across time**.
2. BPTT unrolls the RNN through time and applies the chain rule; **truncation** handles very long sequences.
3. **Vanilla RNNs suffer from vanishing gradients** — they cannot learn long-range dependencies effectively.
4. **LSTMs** solve this with gated cell states that act as additive gradient highways.
5. The **sequential computation** bottleneck of RNNs is their fundamental limitation, motivating attention and Transformer architectures.

---

*Notes compiled from CS231n Spring 2025 Lecture 7 slides and course materials.*
