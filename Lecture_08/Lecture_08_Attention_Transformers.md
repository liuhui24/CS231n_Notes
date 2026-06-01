# CS231n (Spring 2025) — Lecture 8: Attention and Transformers

> **课程:** Stanford CS231n: Deep Learning for Computer Vision
> **整理日期:** 2026-05-18

---

## 1. 历史动机：从 RNN 瓶颈到 Attention

### Seq2Seq 瓶颈
一个固定大小向量必须表示一切。长序列的重要细节被丢失或稀释。

**解决方案:** 让 Decoder 在每个解码步灵活访问任意 Encoder 隐藏状态。

---

## 2. RNN 中的 Attention (Bahdanau et al., 2014)

每个 Decoder 时间步 i:

1. **对齐得分:** $e_{i,j} = a(s_{i-1}, h_j)$ — 小 MLP
2. **Softmax:** $\alpha_{i,j} = \text{softmax}_j(e_{i,j})$
3. **上下文向量:** $c_i = \sum_j \alpha_{i,j} \cdot h_j$
4. **解码:** $s_i = f(s_{i-1}, y_{i-1}, c_i)$

**关键特性:** 完全可微，端到端训练，对齐模式自动学习，可解释。

---

## 3. 抽象 Attention: Query, Key, Value

| 角色 | 含义 |
|------|------|
| **Query (Q)** | "我需要什么信息？" |
| **Key (K)** | 与 Q 匹配产生权重 |
| **Value (V)** | 通过权重组合产生输出 |

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**为什么除以 $\sqrt{d_k}$？** 防止大维度内积使 softmax 过于"尖锐" → 梯度消失。

---

## 4. Self-Attention

$$Q = XW_Q^T,\quad K = XW_K^T,\quad V = XW_V^T$$

- 每个位置关注所有其他位置
- 排列等变 → 需要位置编码
- **Masked Self-Attention:** 将未来位置 logit 设为 $-\infty$ → 自回归建模

---

## 5. Multi-Head Self-Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O$$

- **典型值:** $h=8$, $d_k=d_v=d_\text{model}/h=64$
- 不同头关注不同表征子空间

---

## 6. 序列处理原语对比

| 原语 | 并行性 | 感受野 | 复杂度 | 关键局限 |
|------|--------|--------|--------|---------|
| **RNN** | 差 | 长但衰减 | $O(n \cdot d^2)$ | 难以并行化 |
| **卷积** | 好 | 局部 | $O(k \cdot n \cdot d^2)$ | 远距离需深层 |
| **Self-Attention** | 极好 | 单层全局 | $O(n^2 \cdot d)$ | 序列长度二次方 |

---

## 7. Transformer 架构 (Vaswani et al., 2017)

### Transformer Block (重复 N=6 次)

```
1. Multi-Head Self-Attention
2. Add & LayerNorm (残差连接)
3. Position-wise FFN (MLP, 每位置独立)
4. Add & LayerNorm (残差连接)
```

### 完整 Encoder-Decoder

**Encoder (N=6):** 处理整个输入 → 输出作为 Decoder 的 K, V

**Decoder (N=6, 每块三层):**
1. Masked Multi-Head Self-Attention (因果)
2. Cross-Attention — Q 来自 Decoder, K&V 来自 Encoder
3. Position-wise FFN

### 位置编码

$$PE_{(t, 2i)} = \sin\!\left(\frac{t}{10000^{2i/d_\text{model}}}\right), \quad PE_{(t, 2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d_\text{model}}}\right)$$

---

## 8. 关键公式速查

| 概念 | 公式 |
|------|------|
| Scaled Dot-Product | $\text{softmax}(QK^T / \sqrt{d_k}) \cdot V$ |
| Multi-Head | $\text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$ |
| Transformer Block | $\text{LN}(x + \text{MHA}(x)) \rightarrow \text{LN}(x + \text{FFN}(x))$ |

---

## 9. 现代语境

- Transformer 用于所有模态: NLP (GPT, BERT), 视觉 (ViT), 多模态 (CLIP, DALL·E)
- ViT: 图像 → patches → token 序列 → 标准 Transformer Encoder
- 规模化: GPT-3 (175B 参数), 涌现少样本能力
- 预训练范式: 大规模无监督预训练 → 下游任务微调

---

## 核心要点

1. **Attention** 从 RNN 补丁演变为独立序列处理原语
2. **Q, K, V 抽象** — Attention = softmax(QKᵀ/√d)·V
3. **Self-Attention** 提供单层全局交互；代价是 O(n²)
4. **Multi-Head** 增加容量和表达力
5. **Transformer = Self-Attention + FFN + Residuals + LayerNorm**

---

*笔记整理自 [CS231n Spring 2025 Lecture 8](https://youtu.be/PTypu6GqEd4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) 及课程资料*
