# CS231n (Spring 2025) — Lecture 13: Generative Models (Part 1)

> **Instructor:** Justin Johnson
> **Date:** May 15, 2025

---

## 1. Supervised vs. Unsupervised Learning

| | Supervised Learning | Unsupervised Learning |
|------|---------------------|-----------------------|
| **Data** | $(x, y)$ — data with labels | $x$ — just data, no labels |
| **Goal** | Learn a function mapping $x \rightarrow y$ | Learn hidden structure in the data |
| **Examples** | Classification, regression, detection, segmentation, captioning | Clustering (K-means), dimensionality reduction (PCA), density estimation |

---

## 2. Discriminative vs. Generative Models

| Model | Learns | Notation |
|-------|--------|----------|
| **Discriminative** | Probability of label given data | $p(y \mid x)$ |
| **Generative** | Probability distribution over data | $p(x)$ |
| **Conditional Generative** | Probability distribution over data given label | $p(x \mid y)$ |

### Density Function Refresher

- $p(x)$ assigns a positive number to each possible $x$ — higher = more likely
- Normalized: $\int p(x) \,dx = 1$
- Different values of $x$ **compete** for probability density

For a discriminative classifier: $P(\text{cat} \mid \text{image}) + P(\text{dog} \mid \text{image}) + \dots = 1$. Different labels compete for probability.

---

## 3. Taxonomy of Generative Models

```
                    Generative Models
                         │
        ┌────────────────┴────────────────┐
        │                                 │
  Explicit Density                 Implicit Density
  (can compute p(x))              (can only sample)
        │                                 │
  ┌─────┴─────┐                    ┌──────┴──────┐
  │           │                    │             │
Tractable   Approximate        GANs        Diffusion
  p(x)        p(x)           (Direct)     (Indirect)
  │           │
Autoregressive  VAE
  Models
```

---

## 4. Autoregressive Models

Treat data as a sequence and factor the joint distribution using the chain rule:

$$p(x) = p(x_1, x_2, \ldots, x_T) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2) \cdots = \prod_{t=1}^T p(x_t \mid x_1, \ldots, x_{t-1})$$

- For images: treat pixels as a sequence (e.g., row-by-row, pixel-by-pixel)
- Model each conditional $p(x_t \mid x_{<t})$ with an RNN or Transformer
- **Advantage:** Directly maximizes the likelihood of training data — principled
- **Disadvantage:** Sequential sampling is slow — must generate one pixel/token at a time
- Examples: PixelRNN, PixelCNN, ImageGPT

---

## 5. Variational Autoencoders (VAEs)

### Core Idea

Instead of directly modeling $p(x)$, introduce a **latent variable** $z$ and learn:
- **Encoder** $q_\phi(z \mid x)$: maps input $x$ to a distribution over latent $z$
- **Decoder** $p_\theta(x \mid z)$: reconstructs $x$ from latent $z$

### The ELBO (Evidence Lower Bound)

Directly maximizing $\log p(x)$ is intractable. Instead, maximize the **variational lower bound**:

$$\log p(x) \geq \mathbb{E}_{z \sim q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \parallel p(z))$$

- **First term:** Reconstruction loss — how well can we reconstruct $x$ from $z$?
- **Second term:** KL divergence — keeps the latent distribution close to a simple prior $p(z)$ (usually $\mathcal{N}(0, I)$)

### Reparameterization Trick

To backpropagate through sampling: sample $\epsilon \sim \mathcal{N}(0, I)$, then $z = \mu + \sigma \odot \epsilon$. This makes the sampling operation differentiable.

### Characteristics

- **Advantage:** Principled probabilistic framework; provides a latent space for interpolation and manipulation
- **Disadvantage:** Generated samples tend to be blurry — the Gaussian decoder assumption averages over possible images

---

## 6. Preview: Next Lecture

- **Generative Adversarial Networks (GANs):** Give up on explicit density, focus on generating realistic samples via adversarial training
- **Diffusion Models:** Iteratively denoise random noise into structured images — the current state of the art

---

## Core Takeaways

1. Generative models learn the data distribution $p(x)$ — a fundamentally different goal from discriminative models $p(y \mid x)$.
2. **Autoregressive models** factor $p(x)$ into a product of conditionals — principled but slow to sample.
3. **VAEs** learn a latent variable model via the ELBO — provides a structured latent space but produces blurry samples.
4. The taxonomy: explicit density (autoregressive, VAE) vs. implicit density (GANs, diffusion) — each with different tradeoffs.
5. Modern generative AI (DALL-E, Stable Diffusion) builds on all of these foundations.

---

*Notes compiled from CS231n Spring 2025 Lecture 13 slides and course materials.*
