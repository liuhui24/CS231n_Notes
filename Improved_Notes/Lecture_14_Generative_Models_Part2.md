# CS231n (Spring 2025) — Lecture 14: Generative Models (Part 2)

> **Instructor:** Justin Johnson
> **Date:** May 20, 2025

---

## 1. Taxonomy Recap

From Part 1, the two branches of generative models:

| Branch | Can compute $p(x)$? | Can sample? |
|--------|---------------------|-------------|
| **Explicit Density** (Autoregressive, VAE) | Yes (or approximate) | Yes |
| **Implicit Density** (GANs, Diffusion) | No | Yes |

**Part 2 focus:** GANs and Diffusion Models.

---

## 2. Generative Adversarial Networks (GANs)

### Core Idea (Goodfellow et al., 2014)

Instead of modeling $p(x)$ explicitly, train a **Generator** $G$ to produce samples that are indistinguishable from real data — by competing against a **Discriminator** $D$.

### Setup

- Real data: $x \sim p_{\text{data}}(x)$
- Latent variable: $z \sim p(z)$ — simple prior (e.g., unit Gaussian)
- **Generator:** $G(z)$ — maps latent $z$ to fake image $\hat{x}$
- **Discriminator:** $D(x)$ — classifies whether $x$ is real (1) or fake (0)
- **Goal:** $p_G \rightarrow p_{\text{data}}$

### The Minimax Game

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

| Component | Objective |
|-----------|-----------|
| **Discriminator** wants | $D(x) = 1$ for real data, $D(x) = 0$ for fake data |
| **Generator** wants | $D(G(z)) = 1$ — fool the discriminator |
| **The first term** ($\mathbb{E}[\log D(x)]$) | Does not depend on $G$; only the second term matters for the generator |

### Training Procedure

Alternate between:
1. **Train D:** Take a minibatch of real data and a minibatch of fake (generated) data; update D to maximize the objective
2. **Train G:** Generate fake data; update G to maximize $\log D(G(z))$ — i.e., make D classify fakes as real

### GAN Challenges

| Challenge | Description |
|-----------|-------------|
| **Training Instability** | The minimax game can oscillate or diverge; balancing G and D is delicate |
| **Mode Collapse** | Generator produces only a few varieties of samples, ignoring other modes of the data distribution |
| **Evaluation** | No single metric; sample quality is often judged visually or through Inception Score (IS) / Frechet Inception Distance (FID) |

### GAN Evolution

- **DCGAN:** CNN-based GAN architecture; established stable architectural patterns
- **StyleGAN:** Progressive growing + style-based generator → unprecedented quality and latent space control
- **Conditional GANs:** $G(z, y)$ — generate images conditioned on class labels or text

---

## 3. Diffusion Models

### Core Intuition

Instead of generating an image in one shot (like GANs), diffusion models learn to **iteratively denoise** random noise into a structured image.

### Two Processes

**Forward (Diffusion) Process:**
- Start with a real image $x_0$
- Gradually add Gaussian noise over $T$ timesteps:
  $$x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- After enough steps ($T \approx 1000$), $x_T$ is essentially pure noise

**Reverse (Denoising) Process:**
- Train a neural network $\epsilon_\theta(x_t, t)$ to **predict the noise** added at each timestep
- The training objective (simplified):
  $$L = \mathbb{E}_{t, x_0, \epsilon}[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 ]$$
- At sampling time: start from $x_T \sim \mathcal{N}(0, I)$, iteratively apply the learned denoising:
  $$x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z$$

### Why Diffusion Models Work So Well

| Property | Benefit |
|----------|---------|
| **Iterative refinement** | Each step is a small, easy-to-learn transformation |
| **Simple loss** | Just predict the noise — no adversarial training, no mode collapse |
| **Stable training** | The objective is a straightforward regression problem |
| **Scalable** | Works remarkably well with more compute and larger models |

### Key Architectures

- **U-Net:** The dominant backbone — encoder-decoder with skip connections, well-suited for per-pixel prediction
- **DDPM (2020):** The original pixel-space diffusion model
- **Latent Diffusion / Stable Diffusion:** Apply diffusion in a **compressed latent space** (from a pretrained VAE) instead of pixel space — dramatically more efficient
- **Text conditioning:** Cross-attention from text embeddings (CLIP) into the U-Net → text-to-image generation

### Classifier-Free Guidance

At sampling time, interpolate between conditional and unconditional predictions:

$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$

Higher guidance scale $w$ → stronger adherence to the conditioning, but potentially less diversity.

---

## 4. Comparison of Generative Approaches

| Approach | Sample Quality | Sample Speed | Training Stability | Latent Space |
|----------|---------------|-------------|-------------------|--------------|
| **Autoregressive** | Good | Very slow (sequential) | Stable | No |
| **VAE** | Blurry | Fast (one forward pass) | Stable | Yes (meaningful) |
| **GAN** | Sharp (best for a while) | Fast (one forward pass) | Unstable | Yes |
| **Diffusion** | Excellent (current SOTA) | Slow (iterative, ~50-1000 steps) | Stable | No (but can add via latent diffusion) |

---

## Core Takeaways

1. **GANs** frame generation as a minimax game between generator and discriminator — produces sharp samples but suffers from training instability and mode collapse.
2. **Diffusion models** learn to reverse a gradual noising process — stable training with a simple regression objective, currently state-of-the-art for image generation.
3. Diffusion works by predicting the noise added at each timestep; sampling requires iterative denoising from pure noise.
4. **Latent diffusion** (Stable Diffusion) applies diffusion in compressed latent space for efficiency; text conditioning via cross-attention.
5. The field has converged on diffusion models for quality-critical image generation, while GANs remain useful where fast single-pass generation is needed.

---

*Notes compiled from CS231n Spring 2025 Lecture 14 slides and course materials.*
