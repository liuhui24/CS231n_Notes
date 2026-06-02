# Lecture 3: Regularization and Optimization

---

## 1. 损失函数回顾

### Multiclass SVM (合页损失)
$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

### Softmax + 交叉熵
$$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$$

---

## 2. 正则化

### 完整目标函数
$$L(W) = \underbrace{\frac{1}{N}\sum_i L_i}_{\text{数据损失}} + \underbrace{\lambda R(W)}_{\text{正则化}}$$

| 方法 | 公式 | 效果 |
|------|------|------|
| **L2** | $R(W) = \sum W^2$ | 权重分散、小而均匀 |
| **L1** | $R(W) = \sum \|W\|$ | 稀疏解、特征选择 |
| **Elastic Net** | L1 + L2 | 两者结合 |

### λ 的作用

| λ 值 | 效果 |
|------|------|
| 太大 | 欠拟合 (权重约束过强) |
| 太小 | 过拟合 (模型过于复杂) |

---

## 3. 优化方法

| 方法 | 思路 | CIFAR-10 |
|------|------|----------|
| **随机搜索** | 随机尝试 W | ~15.5% |
| **梯度下降** | $W \leftarrow W - \alpha \nabla L$ | 有效 |
| **Mini-batch SGD** | 在小批量上计算梯度 | 标准做法 |

### 高级优化器

| 优化器 | 核心思想 |
|--------|---------|
| **SGD + Momentum** | 累积速度，平滑轨迹 |
| **RMSProp** | 每参数自适应学习率 |
| **Adam** | Momentum + RMSProp + 偏差修正 |
| **AdamW** | 解耦权重衰减 (**推荐**) |

### 优化中的常见问题

| 问题 | 解法 |
|------|------|
| **病态条件** | Momentum, Adam |
| **鞍点** | 二阶优化或动量 |
| **学习率过大** | 降低 lr / warmup |
| **学习率过小** | 提高 lr |

---

## 4. 学习率调度

```
Linear Warmup → Cosine Decay (推荐)
或 Step Decay (每 N epoch 衰减)
```

> **线性缩放法则:** batch size 翻倍 → lr 翻倍

---

## 5. 实践建议

| 建议 | 说明 |
|------|------|
| **默认优化器** | AdamW + linear warmup + cosine decay |
| **备选** | SGD + Momentum |
| **正则化选择** | L2 用于分布式表征; L1 用于稀疏性 |
| **超参搜索** | 随机搜索优于网格搜索 |
| **梯度检查** | 数值梯度验证解析梯度 |

