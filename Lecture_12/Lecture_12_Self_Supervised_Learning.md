# CS231n (Spring 2025) — Lecture 12: Self-Supervised Learning

> **课程:** Stanford CS231n: Deep Learning for Computer Vision
> **整理日期:** 2026-05-18

---

## 1. 动机：标注瓶颈

- 高质量特征编码器 (ResNet, ViT) 赋能分类、检测、分割
- 但以监督目标大规模训练编码器需要**海量人工标注**
- 尤其语义分割需要逐像素标注 — 成本极高

### SSL 的承诺

> 在**无人工标签**的情况下获得同样高质量的特征编码器

**方法:** 在大规模无标注数据上预训练 → 下游任务迁移

---

## 2. 定义与核心框架

**Self-Supervised Learning:** 在无标注数据上定义辅助 **pretext task**，伪标签从数据自身自动生成。

### 三个关键组件

| 组件 | 角色 |
|------|------|
| **Encoder** | 通过优化 pretext loss 学习表征 |
| **Decoder / Projection Head** | 将表征转换为 pretext 输出 |
| **Pretext Loss** | 自动从数据派生的监督信号 |

### 迁移流程

```
1. 在 pretext task 上端到端训练 encoder + decoder
2. 丢弃 decoder
3. 将 encoder 迁移到下游任务
4. 冻结 (linear probing) 或微调
```

---

## 3. Pretext 任务三大类

### A. 基于变换的 Pretext 任务

对图像施加合成变换，让模型预测变换参数或重建原始内容。

| 任务 | 描述 | 学到什么 |
|------|------|---------|
| **Rotation Prediction** | 判断旋转了 {0°, 90°, 180°, 270°} 中的哪个 | 全局物体方向、粗形状线索 |
| **Jigsaw Puzzles** | 预测 patch 的排列索引 (如 3×3 网格) | 空间布局、部件关系 |
| **Colorization** | 从 Lab 的 L 通道预测 (A,B) 颜色通道 | 物体身份、材质属性 |
| **Inpainting** | 遮盖区域，重建缺失像素 | 上下文和语义理解 |

> **设计原则:** 任务必须**足够难**以迫使丰富特征学习，但**足够通用**以广泛迁移。

### B. 掩码重建 (MAE — Masked Auto encoders)

主导性的现代重建方法 (Kaiming He et al.):

- **架构:** ViT encoder + 轻量 decoder
- **掩码:** 随机掩码**高比例**的 patch (如 **75%**)
- **Encoder:** 仅处理**未掩码** patch → 效率提升
- **Decoder:** 接收编码的可见 patch + 可学习**掩码 token** → 重建像素
- **Loss:** 仅在掩码位置计算 L2 重建损失

### C. 对比学习 (Contrastive Learning)

将表征学习框架为: **最大化同图不同视图的一致性 (正样本)**，同时**最小化与其他图像的相似性 (负样本)**。

### InfoNCE Loss

$$L = -\log \frac{\exp(s(x, x^+) / \tau)}{\sum_i \exp(s(x, x_i) / \tau)}$$

- $s(\cdot,\cdot)$: 相似度函数 (通常余弦相似度)
- $x^+$: 正样本对 (同图不同增强)
- $x_i$: 负样本 (其他图像)
- $\tau$: 温度超参数
- 更多负样本 → 更紧的下界 → 更好性能

### SimCLR

- 每图两增强视图 → 共享 encoder → MLP 投影头 → 对比损失
- **投影头至关重要:** 将对比目标隔离到子空间
- **要求:** 大 batch size (提供足够 in-batch 负样本)

### MoCo (Momentum Contrast)
- 解决 SimCLR 大 batch 问题
- 维护**动态队列**存储编码后的负样本
- **Key encoder** 通过 query encoder 的**指数移动平均**更新
- 历史负样本**不在当前计算图中**

### 其他对比学习范式
- **DINO / DINOv2:** 师生自蒸馏，无需负样本

---

## 4. 评估维度

| 维度 | 衡量什么 |
|------|---------|
| **Pretext 准确率** | 模型解决辅助任务的能力 |
| **Linear Probing 准确率** | 冻结 encoder → 训练线性分类器 — 表征质量代理指标 |
| **鲁棒性** | 扰动和分布偏移下的性能 |
| **下游任务性能** | 全微调后的准确率/样本效率 — **最决定性的指标** |
| **可视化** | t-SNE / UMAP 检查语义结构 |

---

## 5. 架构模式总结

| 模式 | 关键组件 |
|------|---------|
| **Encoder 中心** | ResNet, ViT — 保留用于下游的部分 |
| **Decoder / 投影头** | 预训练后丢弃；分类用小 FC, 重建用大 decoder |
| **掩码 Token (MAE)** | 代表被掩码位置的可学习嵌入 |
| **动量队列 (MoCo)** | 外部记忆 + 动量更新的 key encoder |

---

## 6. 实践指导

- **匹配 pretext 与数据和算力:** MAE 配合 ViT + 激进掩码扩展性好; 对比方法需要大量负样本或队列
- **关键超参数:** 掩码比例、投影头深度、温度 τ、batch size、队列长度、动量系数
- **先 Linear Probing 验证**，再全微调
- **数据增强**对对比方法至关重要

---

## 核心要点

1. SSL 通过从数据自身派生监督来消除标注瓶颈
2. 三类 pretext 家族提供**互补的归纳偏置**: 空间布局 / 上下文重建 / 实例判别
3. **MAE + ViT** 和 **MoCo/SimCLR** 代表视觉 SSL 的现代 SOTA
4. **Encoder 才是重要的** — decoder 和投影头是预训练后丢弃的脚手架
5. **下游微调性能是终极指标**; linear probing 是有用代理
6. SSL 在许多 benchmark 上已匹敌或超越监督 ImageNet 预训练

---

*笔记整理自 [CS231n Spring 2025 Lecture 12](https://youtu.be/PTypu6GqEd4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) 及课程资料*
