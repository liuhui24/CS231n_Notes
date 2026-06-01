# CS231n (Spring 2025) — Lecture 7: Recurrent Neural Networks

> **课程:** Stanford CS231n: Deep Learning for Computer Vision
> **整理日期:** 2026-05-18

---

## 1. 序列建模：为什么需要 RNN？

### 五种序列任务范式

| 范式 | 示例 |
|------|------|
| **One-to-One** | 标准图像分类 |
| **One-to-Many** | 图像描述 (Image Captioning) |
| **Many-to-One** | 情感分析、视频 → 单标签 |
| **Many-to-Many** (对齐) | 逐帧视频分类 |
| **Many-to-Many** (不对齐) | 机器翻译 |

---

## 2. Vanilla RNN

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

| 矩阵 | 含义 |
|------|------|
| $W_{xh}$ | 输入到隐藏 |
| $W_{hh}$ | 隐藏到隐藏 (循环连接) |
| $W_{hy}$ | 隐藏到输出 |

### 关键特性
- **参数时间共享:** 处理任意长度序列
- **展开 (Unrolling):** 沿时间展开为前馈图 → 支持 BPTT
- **$h_0$:** 可学习或初始化为零

---

## 3. 时间反向传播 (BPTT)

- 沿所有时间步应用链式法则
- 截断 BPTT: 固定大小滑动窗口 → 减少内存/计算

---

## 4. 梯度消失与爆炸

| 条件 | 结果 |
|------|------|
| $W_{hh}$ 最大奇异值 < 1 | **梯度消失** |
| $W_{hh}$ 最大奇异值 > 1 | **梯度爆炸** |

| 问题 | 解决方法 |
|------|---------|
| **梯度爆炸** | 梯度裁剪 (Gradient Clipping) |
| **梯度消失** | 架构改进: **LSTM**, **GRU** |

### 具体例子
> "The **cat**, which already ate …, **was** full."
> "The **cats**, which already ate …, **were** full."

普通 RNN 难以捕捉这种长距离依赖——梯度信号指数衰减。

---

## 5. 字符级 RNN 语言模型

- 输入: one-hot 字符 → Embedding → 稠密向量
- 预测: 每步预测下一个字符 (softmax over vocab)
- Teacher Forcing: 训练时用 ground-truth 前一 token
- 推理: Greedy / Sampling / Beam Search

### 隐藏激活的可解释性
单个隐藏单元学到: 引号检测、行长度预测、缩进计数、注释检测、嵌套深度跟踪

---

## 6. RNN 的权衡

| 优势 | 劣势 |
|------|------|
| 理论上无限上下文 | 顺序计算 — 无法时间并行 |
| 参数效率高 | 固定隐藏状态瓶颈 |
| 自然适合变长序列 | 训练不稳定 |

---

## 7. LSTM (长短期记忆)

### 四个门 + 细胞状态

| 组件 | 公式 | 作用 |
|------|------|------|
| **遗忘门** $f_t$ | $\sigma(W_f [h_{t-1}, x_t] + b_f)$ | 控制丢弃什么 |
| **输入门** $i_t$ | $\sigma(W_i [h_{t-1}, x_t] + b_i)$ | 控制写入什么 |
| **候选细胞** $g_t$ | $\tanh(W_g [h_{t-1}, x_t] + b_g)$ | 候选新值 |
| **输出门** $o_t$ | $\sigma(W_o [h_{t-1}, x_t] + b_o)$ | 控制暴露什么 |

### 更新方程

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
$$h_t = o_t \odot \tanh(c_t)$$

### 为什么缓解梯度消失？
- 细胞状态更新是**加法性**的
- 当 $f_t \approx 1$, 信息不变地通过
- 梯度通过细胞状态路径回传时衰减最小 — 类似 ResNet 跳跃连接

---

## 8. 视觉-语言应用

- **图像描述:** CNN → LSTM → 自回归生成文字
- **视觉问答 (VQA):** RNN 编码问题 + CNN 编码图像
- **视觉导航:** RNN 隐藏状态作为观察记忆

---

## 核心要点

1. RNN 通过时间共享权重处理变长序列
2. BPTT 沿时间展开计算梯度
3. 梯度消失是 RNN 的阿喀琉斯之踵 → 驱动 LSTM 和 Transformer
4. LSTM 通过门控细胞状态提供加法梯度高速公路
5. 顺序计算是 RNN 的根本瓶颈

---

*笔记整理自 [CS231n Spring 2025 Lecture 7](https://youtu.be/PTypu6GqEd4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) 及课程资料*
