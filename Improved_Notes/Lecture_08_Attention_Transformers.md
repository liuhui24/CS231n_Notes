# CS231n (Spring 2025) — Lecture 8: Attention and Transformers

> **Instructor:** Justin Johnson
> **Date:** April 24, 2025

---

## 1. Motivation: The RNN Bottleneck in Seq2Seq

### Encoder-Decoder with RNNs

For machine translation (English → Italian):

**Encoder:** $h_t = f_W(x_t, h_{t-1})$ — processes the entire input sequence
**Decoder:** $s_t = g_U(y_{t-1}, s_{t-1}, c)$ — generates the output sequence

The encoder's final hidden state provides:
- Initial decoder state $s_0$
- **Context vector $c$** (typically $c = h_T$)

**The problem:** The entire input sequence must be compressed into one fixed-size vector $c$. For long sequences (e.g., $T = 1000$), information from early inputs is severely diluted.

### The Solution: Attention (Bahdanau et al., 2015)

On **each** decoder step, look back at the **entire** input sequence and dynamically select relevant parts.

---

## 2. RNN with Attention: Mechanism

For each decoder time step $t$:

1. **Compute alignment scores:** $e_{t,i} = f_{\text{att}}(s_{t-1}, h_i)$ — a small linear layer scoring how relevant each encoder hidden state is to the current decoder state

2. **Normalize to attention weights:** $a_{t,i} = \text{softmax}(e_t)_i$ — weights sum to 1

3. **Compute context vector:** $c_t = \sum_i a_{t,i} \cdot h_i$ — weighted combination of encoder states

4. **Decode:** $s_t = g_U(y_{t-1}, s_{t-1}, c_t)$ — use the dynamically-computed context

The entire mechanism is **fully differentiable** and trained end-to-end. The attention weights are interpretable: they show which source words the model focused on when generating each target word.

---

## 3. Generalized Attention: Query, Key, Value

The attention mechanism can be abstracted into three roles:

| Component | Role | Analogy |
|-----------|------|---------|
| **Query (Q)** | "What am I looking for?" | A search request |
| **Key (K)** | "What do I contain?" | Database index |
| **Value (V)** | "What information do I provide?" | Database content |

**Attention =** compute similarity between Query and all Keys, then use those similarities to take a weighted sum of Values.

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Why scale by $\sqrt{d_k}$?** Without scaling, large $d_k$ produces large dot products → softmax saturates to one-hot → gradients vanish. Scaling keeps softmax in a well-behaved regime.

---

## 4. Self-Attention

In self-attention, Q, K, V all come from the **same input**:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

- Each position in the sequence can attend to **all** positions (including itself)
- Each output depends directly on all inputs — **global receptive field in a single layer**
- The operation is **permutation equivariant** — shuffling inputs shuffles outputs identically
- **Positional encoding** is required to inject order information

### Masked Self-Attention

For autoregressive tasks (generating sequences), future positions are masked by setting their attention logits to $-\infty$ before softmax, ensuring each position can only attend to itself and earlier positions.

### Multi-Head Self-Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $\text{head}_i = \text{Attention}(QW_{Q}^i, KW_{K}^i, VW_{V}^i)$

- Typical: $h = 8$ heads, $d_k = d_v = d_{\text{model}} / h = 64$
- Different heads learn to attend to different types of relationships (syntactic, semantic, positional, etc.)

---

## 5. Comparing Sequence Processing Primitives

| Primitive | Parallelism | Receptive Field | Compute | Key Limitation |
|-----------|-------------|-----------------|---------|----------------|
| **RNN** | None (sequential) | Long but attenuated | $O(n \cdot d^2)$ | Can't parallelize across time |
| **1D Convolution** | Good | Local (needs depth for global) | $O(k \cdot n \cdot d^2)$ | Long-range needs many layers |
| **Self-Attention** | Excellent | Global in one layer | $O(n^2 \cdot d)$ | Quadratic in sequence length |

---

## 6. The Transformer (Vaswani et al., 2017)

### Transformer Block

Each block consists of two sublayers, each with a **residual connection + LayerNorm**:

```
x → Multi-Head Self-Attention → Add(x, ·) → LayerNorm → FFN → Add(x, ·) → LayerNorm
```

- **Self-Attention sublayer:** Global context aggregation
- **Position-wise FFN:** A 2-layer MLP applied identically to each position independently (adds per-position capacity)

Blocks are stacked $N = 6$ times (in the original paper).

### Encoder-Decoder Architecture

**Encoder ($N$ blocks):**
- Processes the full input sequence (bidirectional attention — no masking)
- Outputs contextualized representations for each position

**Decoder ($N$ blocks, each with 3 sublayers):**
1. **Masked Self-Attention** — causal, can only see past positions
2. **Cross-Attention** — Q from decoder, K & V from encoder output
3. **Position-wise FFN**

### Positional Encoding

Since self-attention is permutation-invariant, positional information is added:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Sinusoidal functions allow the model to extrapolate to sequence lengths not seen during training.

---

## 7. The Transformer Revolution

- **NLP:** GPT series (decoder-only), BERT (encoder-only) — scaled to hundreds of billions of parameters
- **Vision:** Vision Transformer (ViT) — image patches as tokens, standard Transformer encoder
- **Multimodal:** CLIP, DALL-E — Transformers bridging vision and language
- **Key advantage:** The architecture has remained largely unchanged since 2017 — it scales remarkably well with more data and compute

---

## Core Takeaways

1. **Attention** emerged as a fix for the RNN Seq2Seq bottleneck — allowing each decoder step to dynamically access the full input.
2. The **Q, K, V abstraction** unifies all attention variants: $\text{softmax}(QK^T/\sqrt{d}) \cdot V$.
3. **Self-attention** gives every position direct access to every other position in a single layer — global context without depth.
4. **Multi-head attention** learns multiple relationship types in parallel.
5. **Transformer = Self-Attention + FFN + Residuals + LayerNorm**, repeated and stacked. The $O(n^2)$ cost is the price of global interaction.

---

*Notes compiled from CS231n Spring 2025 Lecture 8 slides and course materials.*
