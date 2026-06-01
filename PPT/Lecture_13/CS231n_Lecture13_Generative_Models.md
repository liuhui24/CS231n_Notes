# CS231n Lecture 13: Generative Models (生成模型) — 详细课堂记录

**课程**: CS231n: Convolutional Neural Networks for Visual Recognition (Spring 2017)
**讲师**: Justin Johnson (Serena Yeung 和 Fei-Fei Li 也参与本课程)
**主题**: 无监督学习与生成模型
**视频**: https://youtu.be/zbHXQRUNlH0
**课件**: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf

---

## 第一部分：从监督学习到无监督学习

### 1.1 监督学习回顾

在前12节课中，课程一直在讨论**监督学习（Supervised Learning）**：

- 数据形式: (x, y)，其中 x 是输入（如图像），y 是标签
- 目标: 学习从 x 到 y 的映射函数 f: x → y
- 典型任务: 图像分类、目标检测、语义分割、图像描述（Image Captioning）等
- 所有这些任务都依赖人工标注的标签

### 1.2 无监督学习（Unsupervised Learning）

Justin 引出本节课的主题——**无监督学习**：

- 数据形式: 只有 x，没有标签 y
- 优势: 无标签数据极其丰富且廉价（互联网上的图片、视频、音频等）
- 目标: 从数据本身发现隐藏的结构和模式

无监督学习的典型任务包括：

| 任务 | 描述 | 例子 |
|------|------|------|
| **聚类 (Clustering)** | 将相似的数据点分组 | K-means, 层次聚类 |
| **降维 (Dimensionality Reduction)** | 在保留结构的前提下减少数据维度 | PCA, t-SNE |
| **特征学习 (Feature Learning)** | 自动学习数据的有效表示 | Autoencoders, 稀疏编码 |
| **密度估计 (Density Estimation)** | 估计数据的概率分布 | **本节课的核心** |

### 1.3 什么是生成模型（Generative Models）？

**定义**: 给定从真实分布 p_data(x) 中采样的训练数据，学习一个模型 p_model(x)，使其能生成与训练数据来自同一分布的新样本。

Justin 用一个简单的方式描述：
> "Given training data, generate new samples from the same distribution."

生成模型的重要性体现在多个方面：

- **生成逼真的图像、艺术作品**
- **超分辨率重建（Super-resolution）**
- **图像上色（Colorization）**
- **强化学习中的模拟与规划**: 生成时间序列数据用于仿真环境，帮助 agent 进行规划
- **学习有用的潜在特征表示**用于下游任务
- **填补缺失数据**

---

## 第二部分：生成模型的分类法（Taxonomy）

Justin 在课堂上将生成模型按照是否显式建模概率密度 p(x) 进行分类：

```
生成模型 (Generative Models)
│
├── Explicit Density (显式密度估计)
│   ├── Tractable Density (可解密度)
│   │   └── PixelRNN / PixelCNN (完全可见信念网络 FVBN)
│   │
│   └── Approximate Density (近似密度)
│       ├── Variational (变分方法)
│       │   └── Variational Autoencoders (VAE)
│       │
│       └── Markov Chain (马尔可夫链方法)
│           └── Boltzmann Machines (简要提及)
│
└── Implicit Density (隐式密度估计)
    └── Direct (直接生成样本)
        └── Generative Adversarial Networks (GAN)
```

这节课重点介绍三种最重要的生成模型：**PixelRNN/PixelCNN**、**VAE**、**GAN**。

---

## 第三部分：PixelRNN 和 PixelCNN（显式可解密度模型）

### 3.1 核心思想：链式法则

PixelRNN 和 PixelCNN 属于**完全可见信念网络（Fully Visible Belief Networks, FVBNs）**。

基本思路是利用概率的**链式法则（Chain Rule）**，将图像的联合分布分解为一维条件分布的乘积：

$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_1, x_2, \ldots, x_{i-1})$$

其中 x_i 是图像中的单个像素值。这意味着：
- 将图像中所有 n 个像素的联合分布分解为每个像素依赖于其之前所有像素的条件概率的乘积
- 每个条件概率 $p(x_i \mid x_1, ..., x_{i-1})$ 由一个神经网络建模
- 训练目标就是最大化训练数据的似然（likelihood）

**生成过程**: 按照预设的顺序逐个生成像素——从图像的某个角落（如左上角）开始，一行一行地生成，每个新像素都依赖于之前已经生成的所有像素。

Justin 在课上展示了这一生成过程的图示：先生成第一个像素，然后在这个像素的条件下生成第二个，以此类推，直到整个图像完成。

### 3.2 PixelRNN (van der Oord et al., NIPS 2016)

**架构特点**:
- 按照空间顺序（例如从左上角到右下角）扫描图像像素
- 使用 **RNN（具体是 LSTM）** 来建模像素间的依赖关系
- 每个时间步基于之前所有像素的隐藏状态来预测当前像素

**优点**:
- 自然适合序列建模，RNN 设计之初就是为了处理序列依赖

**缺点**:
- **训练非常慢**: 因为是序列模型，前向传播必须从第一个像素串行计算到最后一个像素，无法并行化
- 每个像素的预测都依赖于前一个像素的隐藏状态

### 3.3 PixelCNN (van der Oord et al., NIPS 2016)

**架构特点**:
- 同样从角落开始逐像素生成
- 但使用 **CNN 处理上下文区域（Context Region）**，即当前像素之前的所有已生成像素
- 关键技术：**掩码卷积（Masked Convolution）**

**掩码卷积的工作原理**:
- 在普通卷积核上施加掩码，使得当前像素位置只能看到它"之前"的像素（即上下文区域内的像素）
- 具体做法：将卷积核中对应"未来"像素位置的权重设为零
- 对于中心像素，使用两种掩码：
  - Mask A: 当前像素也不能看到自己（用于第一层）
  - Mask B: 当前像素可以看到自己（用于后续层）

**训练时的优势**:
- 训练时，因为训练图像的所有像素值都是已知的，所有像素的条件概率可以**并行计算**
- 只需要一次前向传播，就能得到所有像素位置的预测
- 使用 **Softmax 损失函数**: 将每个像素视为 256 类（对于 8-bit 图像）的分类问题

**测试/生成时的劣势**:
- 生成过程仍然是**串行的**——必须先生成第一个像素，然后第二个……因为每个新像素都依赖于之前生成的像素
- 速度仍然很慢

Justin 在课上强调了一个关键点：
> "训练可以并行化是因为我们有 ground truth 的像素值，但生成新图像时，你只能一个一个像素地生成，这就是为什么 PixelCNN 在生成时和 PixelRNN 一样慢。"

### 3.4 PixelRNN/PixelCNN 总结

| 优点 | 缺点 |
|------|------|
| 显式计算似然 p(x)，可直接评估对数似然 | **生成极慢**（逐个像素串行生成） |
| 在训练集上可得到可优化的精确似然 | 对于高分辨率图像，生成时间不可接受 |
| 样本质量在当时（2016-2017）表现不错 | 没有显式的潜在表示 |
| 训练过程稳定，直接优化对数似然 | |

**后续改进**:
- PixelCNN++ (Salimans et al., 2017): 通过改进架构和离散逻辑混合分布来提升性能
- WaveNet (van der Oord et al., 2016): 类似的思路用于音频生成，极大推动了语音合成领域的发展

---

## 第四部分：变分自编码器（Variational Autoencoders, VAE）

### 4.1 背景：传统自编码器（Autoencoders）

在引入 VAE 之前，Justin 首先回顾了传统的自编码器。

**传统自编码器架构**:

```
输入 x → [Encoder (编码器)] → 潜在特征 z → [Decoder (解码器)] → 重建输出 x̂
```

- **编码器 (Encoder)**: 将高维输入 x（如 28×28 = 784 维的图像）映射到低维潜在特征 z（如 20-100 维）
- **解码器 (Decoder)**: 从低维特征 z 重建出与原始输入 x 尺寸相同的 x̂
- **训练损失**: L2 重建损失 ||x - x̂||²（或其他重建损失）
- **无监督特性**: 训练不需要标签，因为目标就是让输出等于输入

**编码器的架构**（对于图像数据）:
- 通常是 CNN，逐步下采样（通过 stride 或 pooling）
- 最后通过全连接层输出潜在向量 z

**解码器的架构**:
- 与编码器对称（镜像结构）
- 使用上采样（反卷积/转置卷积）逐步恢复空间分辨率
- 最终输出与输入相同尺寸

**传统自编码器的用途**:
- 降维: 将数据压缩到低维表示
- 特征学习: 编码器学到的特征可用于初始化监督学习模型
- 预训练: 当标注数据稀缺时，先用无标签数据训练自编码器，然后用学到的编码器初始化分类网络

**传统自编码器的致命缺陷**:

Justin 在课上明确指出：
> "传统自编码器不能用来生成新的图像。"

原因分析：
- 传统 AE 学习的是一个**确定性的 1 对 1 映射**
- z 空间（潜在空间）没有结构化——某些区域的 z 值解码后可能生成无意义的输出
- 如果随机取一个 z 值输入解码器，输出通常是噪声，无法生成有意义的图像
- 解码器只在训练时见过编码器产生的 z 值，对于随机的 z 值缺乏泛化能力

### 4.2 VAE 的直观理解

VAE 的核心创新：**将编码器输出从确定性值改为概率分布的参数**。

Justin 的解释思路：
> "我们不直接输出 z，而是让编码器输出一个分布的参数——均值 μ 和方差 σ。然后从这个分布中采样得到 z。"

这意味着：
- 不再是：x → z（确定性的点）
- 而是：x → (μ(x), σ(x)) → 从 N(μ, σ²) 中采样 z

**直觉**: 由于采样引入了随机性，相近的 z 值会被解码为相似的图像。这使得 z 空间被"平滑化"了，可以支持插值（interpolation）和生成。

### 4.3 VAE 的概率形式化

VAE 将生成模型定义为概率模型：

**生成过程**:
1. 从先验 p(z) 中采样潜在变量 z（通常选择标准正态分布 N(0, I)）
2. 从条件分布 p_θ(x|z) 中采样数据 x（由参数为 θ 的解码器神经网络建模）

**核心问题——数据似然的计算**:

$$p_\theta(x) = \int p(z) \cdot p_\theta(x \mid z) \, dz$$

这个积分是**不可解的（intractable）**：
- 需要对所有可能的 z 进行积分
- z 是多维连续变量，穷举所有 z 值是不可能的
- 因此无法直接最大化似然 p_θ(x)

**后验概率也是不可解的**:

根据贝叶斯公式：

$$p_\theta(z \mid x) = \frac{p_\theta(x \mid z) \cdot p(z)}{p_\theta(x)} = \frac{p_\theta(x \mid z) \cdot p(z)}{\int p(z) \, p_\theta(x \mid z) \, dz}$$

- 分母 p_θ(x) 不可解 → 后验 p_θ(z|x) 也不可解
- 但我们需要后验来知道"给定图像 x，什么样的 z 最可能产生它"

### 4.4 VAE 的解决方案：变分推断

**关键思路**: 引入一个额外的**编码器网络** q_φ(z|x)（参数为 φ）来**近似**真实后验 p_θ(z|x)。

- q_φ(z|x) 也称为**识别网络（Recognition Network）**或**推断网络（Inference Network）**
- q_φ(z|x) 是一个高斯分布: N(μ_φ(x), σ_φ(x)²·I)，其中 μ 和 σ 由编码器网络输出

### 4.5 变分下界（ELBO — Evidence Lower BOund）

Justin 在课上完整推导了对数似然的下界：

$$\log p_\theta(x) = \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x)]$$

$$= \mathbb{E}_z \left[ \log \frac{p_\theta(x, z)}{p_\theta(z \mid x)} \right]$$

$$= \mathbb{E}_z \left[ \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right]$$

$$= \mathbb{E}_z \left[ \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] + \mathbb{E}_z \left[ \log \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right]$$

$$= \mathbb{E}_z [\log p_\theta(x, z) - \log q_\phi(z \mid x)] + D_{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

$$= \mathcal{L}(x, \theta, \phi) + D_{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))$$

其中：
- 第一项 $\mathcal{L}(x, \theta, \phi)$ 是 **ELBO（Evidence Lower BOund）**
- 第二项是 KL 散度，恒 ≥ 0

由于第二项 KL 散度 ≥ 0，我们有：

$$\log p_\theta(x) \geq \mathcal{L}(x, \theta, \phi)$$

ELBO 可以展开为：

$$\mathcal{L}(x, \theta, \phi) = \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \| p(z))$$

### 4.6 VAE 损失函数的两项含义

**第一项——重建损失（Reconstruction Loss）**:
$$\mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x \mid z)]$$
- 从编码器分布 q_φ(z|x) 中采样 z
- 最大化用 z 重建出 x 的对数似然
- 确保解码器能根据 z 还原出好的 x
- 在实际实现中，通常使用 L2 损失或二元交叉熵（对于像素值归一化到 [0,1] 的图像）

**第二项——KL 正则化项（KL Regularization）**:
$$D_{KL}(q_\phi(z \mid x) \| p(z))$$
- 使近似后验 q_φ(z|x) 接近先验 p(z)（即标准高斯 N(0, I)）
- 起到**正则化**作用，防止编码器为每个训练样本输出互不重叠的分布
- 确保潜在空间是"良好结构化的"——相似的 x 映射到相似区域的 z

当先验 p(z) = N(0, I)，且 q_φ(z|x) = N(μ, σ²I) 时，KL 散度有闭式解：

$$D_{KL}(N(\mu, \sigma^2) \| N(0, 1)) = -\frac{1}{2} \sum_{j=1}^{d} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$$

其中 d 是潜在空间 z 的维度。

当 μ = 0, σ = 1 时，KL 散度 = 0（即 q = p 时，KL = 0）。

### 4.7 重参数化技巧（Reparameterization Trick）

**问题**: 采样操作 z ~ N(μ, σ²) 是不可微的——梯度无法流过随机采样操作！

**解决方案——重参数化技巧**:

将采样操作改写为：

$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 ⊙ 表示逐元素乘法。

这样：
- μ 和 σ 是可学习的参数（由编码器网络输出），梯度可以通过
- ε 是来自标准正态分布的随机噪声，与模型参数无关
- 反向传播时，ε 被视为常数，梯度可以通过 μ 和 σ 流动

Justin 在课上强调这是 VAE 能够成功训练的关键技巧。

### 4.8 VAE 的完整训练流程

**前向传播**:
1. 输入图像 x 通过编码器网络
2. 编码器输出 μ_φ(x) 和 log(σ_φ(x)²)（通常输出 log 方差以保证数值稳定性）
3. 计算 KL 散度损失: D_KL(q_φ(z|x) || p(z))
4. 采样 ε ~ N(0, I)
5. 计算 z = μ + σ ⊙ ε
6. z 通过解码器网络输出重建 x̂
7. 计算重建损失（L2 或交叉熵）

**反向传播**:
- 梯度从重建损失和 KL 损失回传
- 通过重参数化技巧，梯度顺畅流经采样操作
- 同时更新编码器参数 φ 和解码器参数 θ

**训练目标**: 最小化总损失 = 重建损失 + KL 正则化项

### 4.9 VAE 的生成过程

训练完成后：

1. **丢弃编码器**（不需要了）
2. 从先验 p(z) = N(0, I) 中采样 z
3. 将 z 输入解码器网络
4. 解码器输出生成的图像

这就是 VAE 作为生成模型的工作方式——我们可以随机采样 z，然后生成新的图像，而不需要输入任何真实图像！

### 4.10 潜在空间的性质

Justin 在课上展示了一些 VAE 的有趣特性：

**连续插值（Smooth Interpolation）**:
- 在 z 空间中两个点之间进行线性插值
- 解码出的图像会平滑地从一个变为另一个
- 这表明 z 空间是连续的、良好结构化的

**解耦的潜在因子（Disentangled Factors）**:
- 不同的 z 维度倾向于编码不同的语义因素
- 例如在人脸 VAE 中：z_1 控制微笑程度，z_2 控制头部姿态，z_3 控制光照方向
- 这使得 VAE 的潜在空间具有可解释性

**潜在空间算术**:
- 可以通过对 z 向量进行加减操作来实现语义操作
- 类似于 word2vec 中的 "国王 - 男人 + 女人 = 女王"

### 4.11 VAE 总结

| 优点 | 缺点 |
|------|------|
| 基于概率原则的方法论 | 优化的是似然的下界（ELBO），而非真实似然 |
| 编码器 q(z\|x) 提供有用的特征表示，可用于下游任务 | 生成样本**比较模糊（blurry）**，不如 GAN 清晰 |
| 平滑、可解释的潜在空间 | 解码器通常假设输出为高斯分布或伯努利分布，表达能力有限 |
| 天然支持推断查询——给定图像找出其潜在表示 | 对复杂分布拟合能力有限 |

**为什么 VAE 生成的图像比较模糊？**

Justin 解释：
- VAE 优化的是对数似然的下界，而非真实似然
- 解码器输出通常假设为高斯分布（连续值）的均值，这导致取"平均值"的模糊效果
- 模型为每个像素预测均值，而非从多模态分布中采样
- 相比之下，GAN 直接从数据分布隐式学习，能生成更锐利的样本

---

## 第五部分：生成对抗网络（Generative Adversarial Networks, GAN）

### 5.1 引言与核心思想

Justin 引用了 Yann LeCun 的名言：
> "GANs are the most interesting idea in the last 10 years of machine learning."
>（GAN 是过去 10 年机器学习领域最有趣的想法。）

**GAN 的根本区别**:
- PixelRNN/CNN 和 VAE 都试图**显式地建模数据密度 p(x)**
- GAN **完全放弃显式密度建模**
- 改为使用**博弈论方法（Game-Theoretic Approach）**: 通过一个两人博弈来学习如何生成

### 5.2 两个玩家

GAN 由两个神经网络组成，它们相互对抗：

**生成器（Generator, G）**:
- 输入: 随机噪声 z ~ p(z)（通常从标准正态分布或均匀分布采样）
- 输出: 伪造的图像 G(z)
- 目标: **欺骗判别器**，让判别器认为伪造的图像是真实的
- 类比: 假币制造者，试图制造看起来像真币的假币

**判别器（Discriminator, D）**:
- 输入: 一张图像（可能是真实的训练样本，也可能是 G 生成的伪造品）
- 输出: 一个标量概率 D(x) ∈ [0, 1]，表示该图像是真实的概率
- 目标: **正确区分**真实图像和伪造图像
- D(x) → 1 表示"这是真图像"
- D(G(z)) → 0 表示"这是假图像"
- 类比: 警察，试图辨别真币和假币

### 5.3 极小极大目标函数（Minimax Objective）

$$\min_{\theta_g} \max_{\theta_d} \left[ \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p(z)} [\log(1 - D(G(z)))] \right]$$

**Justin 对这个公式的逐项解释**:

**判别器的最大化目标**:
- $\mathbb{E}_{x \sim p_{data}} [\log D(x)]$: 对真实图像的期望 —— D 希望 D(x) 接近 1，所以 log D(x) 接近 0
- $\mathbb{E}_{z \sim p(z)} [\log(1 - D(G(z)))]$: 对伪造图像的期望 —— D 希望 D(G(z)) 接近 0，所以 log(1-D(G(z))) 接近 0
- 两个部分的和越大，判别器表现越好

**生成器的最小化目标**:
- G 希望 D(G(z)) 接近 1（让判别器被欺骗）
- 所以 G 希望最小化 log(1 - D(G(z)))
- 当 D(G(z)) → 1 时，log(1 - D(G(z))) → -∞，这是 G 的"胜利"

**博弈均衡**:
- 在最优解处，生成器产生与真实数据完全无法区分的样本
- 此时 D(x) = D(G(z)) = 0.5（判别器只能随机猜测）
- p_g = p_data（生成分布等于数据分布）

### 5.4 实际训练技巧

Justin 强调了训练 GAN 的一个关键实际问题：

**原始生成器损失的问题**:
$$\min_{\theta_g} \log(1 - D(G(z)))$$

- 在训练初期，生成器很差（生成的图像很容易被判别器识别为假）
- 此时 D(G(z)) ≈ 0，log(1 - D(G(z))) ≈ 0
- 这意味着损失函数的**梯度非常平坦**——生成器无法获得有效的梯度信号
- 生成器很难从这个状态中学习进步

**解决方案——非饱和生成器损失（Non-saturating Generator Loss）**:

将生成器的优化目标改为：

$$\max_{\theta_g} \log D(G(z)) \quad \text{（等价于} \min_{\theta_g} -\log D(G(z)) \text{）}$$

这样做的优势：
- 当生成器表现差时（D(G(z)) ≈ 0），梯度非常大（-log 的导数是 -1/x，x 接近 0 时很大）
- 给生成器强烈的学习信号
- 当生成器表现好时（D(G(z)) ≈ 1），梯度减小，生成器可以精细化调整

**两个版本的对比**:

| 原始版本 | 改进版本 |
|---------|---------|
| min log(1 - D(G(z))) | max log D(G(z)) |
| 初始梯度平坦，训练困难 | 初始梯度大，利于学习 |
| 后期梯度可能过大 | 后期梯度适中，稳定 |

### 5.5 完整的训练算法

Justin 在课上详细描述了 GAN 的训练过程：

```
在每个训练迭代中：

步骤 1: 训练判别器 (k 次更新，通常 k=1)
  a. 从先验噪声分布中采样一个 mini-batch 的 z
  b. 用生成器生成伪造图像 G(z)
  c. 从训练数据中采样一个 mini-batch 的真实图像 x
  d. 计算判别器损失:
     - 真实图像: log D(x)（希望最大化）
     - 伪造图像: log(1 - D(G(z)))（希望最大化）
  e. 对判别器参数 θ_d 执行梯度上升（Gradient Ascent）

步骤 2: 训练生成器 (1 次更新)
  a. 从先验噪声分布中采样一个新的 mini-batch 的 z
  b. 用生成器生成伪造图像 G(z)
  c. 计算生成器损失:
     - 改进版: max log D(G(z))
     - 等价于 min -log D(G(z))
  d. 对生成器参数 θ_g 执行梯度下降（Gradient Descent）

重复以上步骤直到收敛
```

**重要细节**:
- 训练判别器时，生成器的参数**固定不变**
- 训练生成器时，判别器的参数**固定不变**
- 这是一个交替优化过程，而非同时优化
- k 是每个训练周期中判别器的更新次数（通常 k=1，有时对判别器进行更多次更新）

### 5.6 DCGAN — 深度卷积生成对抗网络

Justin 介绍了 DCGAN（Radford et al., ICLR 2016）——将 CNN 架构引入 GAN，极大提升了生成质量。

**DCGAN 的架构原则**:

1. **用跨步卷积替代池化层**:
   - 判别器: 使用 strided convolution（步长 > 1）进行下采样
   - 生成器: 使用 fractional-strided convolution（转置卷积）进行上采样

2. **在网络中使用 Batch Normalization**:
   - 判别器和生成器都使用 BN
   - 但生成器输出层和判别器输入层不使用 BN

3. **移除全连接层**:
   - 使用全卷积架构
   - 生成器的第一层将噪声 z 通过全连接 reshape 为特征图后，后续都是卷积

4. **激活函数选择**:
   - 生成器: 中间层使用 ReLU，**输出层使用 Tanh**
   - 判别器: 使用 **LeakyReLU**（通常 slope = 0.2）而非普通 ReLU

**DCGAN 生成器的典型架构**（从噪声到 64×64 图像）:

```
z ∈ R^100 (噪声向量)
  → 全连接 + reshape → 4×4×1024 特征图
  → 转置卷积 5×5, stride=2 → 8×8×512 (BN + ReLU)
  → 转置卷积 5×5, stride=2 → 16×16×256 (BN + ReLU)
  → 转置卷积 5×5, stride=2 → 32×32×128 (BN + ReLU)
  → 转置卷积 5×5, stride=2 → 64×64×3 (Tanh)
```

**DCGAN 的重要成果**:

1. **高质量图像生成**: 卧室、人脸等图像质量大幅提升
2. **潜在空间向量算术**: 展示了 z 空间的语义属性

   Justin 展示的经典例子：
   > "戴墨镜的男人" - "不戴墨镜的男人" + "不戴墨镜的女人" = "戴墨镜的女人"

   这表明 z 空间中存在可操作的语义方向。

3. **潜在空间平滑插值**: 在两个 z 之间插值，生成的图像平滑过渡

### 5.7 GAN 的进展与应用（截至 2017 年）

Justin 在讲座末尾展示了 GAN 的几个前沿方向：

**更好的训练方法**:
- Wasserstein GAN (WGAN): 使用 Earth Mover's Distance，训练更稳定
- LSGAN (Least Squares GAN): 使用最小二乘损失
- Improved training techniques (Salimans et al., 2016)

**条件 GAN（Conditional GAN, cGAN）**:
- 在生成器和判别器中加入条件信息（如类别标签、文本描述）
- 可以从指定类别生成图像

**图像到图像翻译（Image-to-Image Translation）**:
- **pix2pix** (Isola et al., 2017): 使用配对训练数据的图像翻译
- **CycleGAN** (Zhu et al., 2017): 不需要配对数据的图像翻译
  - 🐴 ↔ 🦓: 马变斑马
  - 🍎 ↔ 🍊: 苹果变橙子
  - ☀️ ↔ ❄️: 夏天变冬天
  - 这些结果让课堂上的同学感到非常兴奋

**文本到图像（Text-to-Image）**:
- 基于文本描述生成对应图像
- 如 "a small bird with a red crown" → 生成对应的鸟类图片

**超分辨率**:
- SRGAN: 从低分辨率图像生成高分辨率图像

### 5.8 GAN 总结

| 优点 | 缺点 |
|------|------|
| **最先进的样本质量**: 生成图像最清晰、最逼真 | **没有显式密度**: 无法计算 p(x)，没有对数似然评估 |
| 生成速度快: 只需一次前向传播 | **训练极其不稳定**: G 和 D 需要精心平衡 |
| 不需要马尔可夫链，只用反向传播 | 可能发生**模式坍塌 (Mode Collapse)**: 生成器只能生成少数几种模式 |
| 可集成任意可微函数作为 G 和 D | 难以生成离散数据（如文本） |
| 应用灵活多变 | 没有推断机制: 不能计算 q(z\|x) |

**模式坍塌（Mode Collapse）**:
- 生成器学会只生成少数几种不同类型的样本
- 例如，在 MNIST 上训练时，G 可能只能生成数字 "1" 和 "7"，而忽略了其他数字
- 这是 GAN 训练中的一个经典难题

---

## 第六部分：总结对比

### 6.1 三种模型的核心对比

| 特性 | PixelRNN/PixelCNN | VAE | GAN |
|------|-------------------|-----|-----|
| **密度类型** | Explicit, Tractable | Explicit, Approximate (lower bound) | Implicit (no density) |
| **优化目标** | 最大化 log p(x)（直接）| 最大化 ELBO | 博弈论 minimax 目标 |
| **生成速度** | 非常慢（串行逐像素）| 快（单次前向传播）| 快（单次前向传播）|
| **样本质量** | 不错 | 模糊/平滑 | **最好，最清晰** |
| **训练难度** | 简单 | 中等 | 困难（不稳定）|
| **推断查询** | 无 | **支持** q(z\|x) | 无 |
| **模型评估** | **容易**（对数似然）| 较易（ELBO）| **困难**（无 p(x)）|
| **潜在表示** | 无 | 有（结构化）| 有（非显式）|

### 6.2 使用建议

Justin 在课上给出的实践指导：

- **需要好的评估指标**: 选择 PixelRNN/CNN（直接对数似然）
- **需要特征表示/推断**: 选择 VAE（编码器 q(z|x) 提供特征）
- **追求最好的视觉质量**: 选择 GAN（但要做好调试困难的准备）
- **探索潜在空间/插值**: VAE（平滑、结构化）或 GAN（需要更多技巧）

### 6.3 未来方向

Justin 提到，研究界正在尝试**结合这些模型的优势**：

- **VAE+GAN 混合**: 利用 VAE 的稳定训练和推断能力 + GAN 的高质量生成
  - 例如: 在 VAE 训练中加入对抗性损失（Adversarial Autoencoder）
- **更好的 GAN 训练技术**: WGAN, WGAN-GP, SN-GAN 等
- **渐进式生成**: ProGAN (Karras et al., 2018)，从低分辨率开始逐步增加分辨率

### 6.4 课堂尾声

Justin 以一种反思性的语调结束了这堂课：

- 生成模型是无监督学习中非常激动人心的方向
- 短短几年内（2014-2017），这个领域从一个学术想法发展到可以生成以假乱真的图像
- 但我们对这些模型的理论理解仍然有限（尤其是 GAN）
- 鼓励同学们去尝试这些模型，亲身体验它们的优缺点

---

## 附录：关键论文列表

### PixelRNN / PixelCNN 系列
- van der Oord et al., "Pixel Recurrent Neural Networks", ICML 2016
- van der Oord et al., "Conditional Image Generation with PixelCNN Decoders", NIPS 2016
- Salimans et al., "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications", ICLR 2017

### VAE 系列
- Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
- Rezende et al., "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", ICML 2014

### GAN 系列
- Goodfellow et al., "Generative Adversarial Nets", NIPS 2014
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", ICLR 2016
- Arjovsky et al., "Wasserstein GAN", ICML 2017
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)", CVPR 2017
- Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)", ICCV 2017

### 综合
- Goodfellow, "NIPS 2016 Tutorial: Generative Adversarial Networks", NIPS 2016

---

## 附录：核心数学公式汇总

### 链式法则分解（PixelRNN/CNN）
$$p(x) = \prod_{i=1}^{n^2} p(x_i \mid x_1, \ldots, x_{i-1})$$

### VAE 证据下界（ELBO）
$$\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \| p(z))$$

### 重参数化技巧
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### GAN 极小极大目标
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p(z)} [\log(1 - D(G(z)))]$$

### VAE KL 散度闭式解（高斯先验）
$$D_{KL}(N(\mu, \sigma^2) \| N(0, 1)) = -\frac{1}{2}\sum_{j} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

### 非饱和 GAN 生成器损失
$$\max_{\theta_g} \mathbb{E}_{z \sim p(z)} [\log D(G(z))]$$

---

> **说明**: 本笔记综合了 CS231n 2017 Spring Lecture 13 (Justin Johnson 主讲) 的视频内容、课程幻灯片、官方课程笔记以及多个相关学习资料，力求完整还原课堂内容。笔记中的引号内容为 Justin Johnson 在课堂上的口语表述的意译。
