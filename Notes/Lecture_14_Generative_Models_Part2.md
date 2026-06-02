# Lecture 14: Generative Models (Part 2 — Diffusion Models)

---

## 1. 扩散模型核心直觉

1. 从简单噪声分布开始 (通常 $\mathcal{N}(0, I)$)
2. 在不同噪声水平下破坏真实数据: $x_t = \alpha_t x_0 + \sigma_t \epsilon$
   - $t=0$: 干净数据
   - $t=1$: 纯噪声
3. 训练网络**去除少量噪声** (或预测 velocity / score)
4. 推理: 采样噪声 $x_1 \sim \mathcal{N}(0,I)$ → 多步迭代去噪 → 干净样本 $x_0$

---

## 2. Rectified Flow (最简洁的表述)

### 训练
- **直线插值**数据与噪声: $x_t = (1-t)x_0 + t\epsilon$
- 网络学习预测**恒定速度**: $v = \epsilon - x_0$
- Loss: $\mathbb{E}_{t, x_0, \epsilon} \left[ \| v_\theta(x_t, t) - (\epsilon - x_0) \|^2 \right]$

### 采样
- N 步 (通常 ~50), 从 $x_1 \sim \mathcal{N}(0,I)$ 开始
- Euler 步反向: $x_{t-\Delta t} = x_t - v_\theta(x_t, t) \cdot \Delta t$

### 条件 Rectified Flow
- 模型接受条件 c: $v_\theta(x_t, t, c)$

---

## 3. Classifier-Free Guidance (CFG)

**关键技巧** (Ho & Salimans 2022):

- 训练时**随机丢弃条件** $c \rightarrow \emptyset$ ~50% 时间
- 推理时评估两个速度并**外推**:

$$v_{\text{guided}} = v_\theta(x_t, t, \emptyset) + \gamma \cdot (v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset))$$

- $\gamma > 1$: 更强地推向条件
- **代价:** 每次采样加倍计算。但 CFG 在高质量条件生成中**必不可少**。

---

## 4. 噪声调度

### 核心问题
- 许多 $(x_0, \epsilon)$ 对可产生相同 $x_t$ — 网络必须**平均**
- $t=0$ (无噪声): 最优预测 = $x_0$ 的均值 — 容易
- $t=1$ (全噪声): 最优预测 = $\epsilon$ 的均值 — 容易
- **中等噪声水平最难** (最模糊)

### 解决方案
- 强调**中间 t**，使用非均匀采样 (如 **logit-normal sampling**)
- 高分辨率数据还需向更高噪声偏移

---

## 5. Latent Diffusion Models (LDMs)

**关键洞察:** 像素空间扩散无法扩展到高分辨率。

### 现代 Pipeline
1. 训练 **VAE + GAN autoencoder** 压缩图像 (3×H×W → C×H/8×W/8 latent)
2. **在隐空间中扩散**，非像素空间
3. **Stable Diffusion** 及大多数现代系统的标准做法

---

## 6. Diffusion Transformer (DiT)

Peebles & Xie (ICCV 2023): 用标准 **Transformer 块**替代 U-Net。

### 条件注入方法

| 方法 | 场景 |
|------|------|
| **adaLN-Zero** (预测 scale/shift) | 时间步 t (最常用) |
| **Cross-attention / Joint attention** | 文本和图像条件 |
| **序列维度拼接** | 替代方案 |

### 真实案例: FLUX.1 [dev]
- 文本 encoder: T5 + CLIP
- VAE: 8× 下采样
- 扩散主干: **12B 参数 DiT**

---

## 7. Text-to-Video

- 相同配方 + **时间轴**: latents = (T, C, H, W)
- **Meta MovieGen (2024):** 30B DiT, patchify → 76K tokens
- 2024 视频模型爆发: Sora, Gen-3, Dream Machine, MovieGen, CogVideoX, Veo 2 等

---

## 8. 扩散蒸馏

**问题:** 采样需要 ~30-50 次模型评估/图像 — 太慢。

| 方法 | 年份 |
|------|------|
| Progressive Distillation | 2022 |
| Consistency Models | 2023 |
| Adversarial Diffusion Distillation | 2024 |
| Multistep Distillation via Moment Matching | 2025 |

> 可减少到几步甚至 **1 步**，且可将 CFG 烘焙进学生模型。

---

## 9. 广义扩散模板

所有扩散变体共享一个模板，仅系数函数不同:

| 变体 | $\alpha_t$ | $\sigma_t$ |
|------|-----------|-----------|
| **Rectified Flow** | $1-t$ | $t$ |
| **Variance Preserving (VP)** | 变化 ($\alpha_t^2 + \sigma_t^2 = 1$) | 变化 |
| **Variance Exploding (VE)** | 1 | 变化 ($\sigma_t = t$) |

### 预测目标

| 名称 | 目标 | 公式 |
|------|------|------|
| **x-prediction** | 干净数据 | $x_0$ |
| **ε-prediction (DDPM)** | 噪声 | $\epsilon$ |
| **v-prediction** | 速度 | $\alpha_t \epsilon - \sigma_t x_0$ |

---

## 10. 扩散的八大等价视角

1. **Autoencoders** — 多噪声水平的去噪自编码器
2. **深层隐变量模型** — VAE 的变分下界
3. **Score Function** — 指向高密度区域的向量场
4. **逆向 SDE** — 逆向随机微分方程
5. **Flow-based Models** — Flow Matching, 连续归一化流
6. **递归神经网络** — 时间上的迭代函数
7. **自回归模型** — 在噪声水平上自回归
8. **条件期望估计** — 干净数据给定噪声观测的条件期望

---

## 11. 自回归模型回击

- 像素级 AR 因 token 太长被放弃
- 但在**离散隐空间**上有效:
  - VQ-VAE → 量化隐空间 → AR Transformer over latent tokens
  - VQ-VAE-2, VQGAN, Parti
