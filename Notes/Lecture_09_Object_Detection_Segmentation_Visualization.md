# Lecture 9: Object Detection, Image Segmentation, Visualizing

---

## 1. Transformer 基础回顾

| 组件 | 作用 |
|------|------|
| **Multi-Head Self-Attention** | 每个 token 并行关注序列中所有其他 token |
| **Layer Normalization** | 稳定训练过程 |
| **Feedforward MLP Blocks** | 在 attention 后对每个 token 独立应用 |
| **Encoder-Decoder 范式** | Encoder 生成表示，Decoder 用于生成/预测 |

---

## 2. Vision Transformer (ViT)

```
Image → Split into Patches → Linear Projection → + Positional Embedding → Transformer Encoder → Classification
```

- **Patchification:** 移除显式的 2D 位置信息
- **位置编码:** 1D 序列索引编码 或 2D (x, y) 坐标编码
- **分类方式:** 可学习 `[CLS]` token → softmax；或全局池化 → 分类器

---

## 3. Transformer 架构优化技巧

| 技术 | 原理 | 优势 |
|------|------|------|
| **Pre-Norm** | LayerNorm 放在 Attention 和 MLP 之前 | 训练更稳定 |
| **RMSNorm** | 去掉均值中心化的轻量归一化 | 大模型更稳定高效 |
| **Gated MLP (GLU)** | 线性投影间引入乘法门控 | 更多非线性，参数更少 |
| **Mixture-of-Experts (MoE)** | 学习路由器将 token 分发给少量专家 | 海量参数 + 稀疏计算 |

---

## 4. 计算机视觉核心任务总览

| 任务 | 输出形式 |
|------|---------|
| 图像分类 | 单类别标签 |
| 目标检测 | 边界框 + 类别标签 |
| 语义分割 | 逐像素类别标签 |
| 实例分割 | 逐实例掩码 + 边界框 |
| 可视化/可解释性 | 归因热力图 |

---

## 5. 语义分割 (Semantic Segmentation)

**目标:** 为每个像素分配一个类别标签。

- 使用 CNN 或 ViT 一次性消费整张图像 → 输出稠密标签图
- 监督信号：**逐像素交叉熵损失**
- 需要像素级标注数据，成本高

---

## 6. 全卷积网络 (FCN)

### Encoder（下采样）
- 步长卷积、池化 → 增大感受野，增加通道容量

### Decoder（上采样）
| 方法 | 类型 |
|------|------|
| 最近邻插值 | 非学习 |
| 最大池化索引 | 非学习 |
| **转置卷积** | **可学习** |

---

## 7. UNet 架构

```
输入 → [收缩路径: 下采样] → [扩张路径: 上采样] → 输出
              ↓  skip connections  ↑
```

### 关键创新：Skip Connections
- 编码器特征图拼接/相加到对应解码器层
- 保留细粒度空间结构
- **医学图像分割**中表现突出

---

## 8. 转置卷积 (Transpose Convolution)

标准卷积的"空间逆操作"——从低分辨率输入产生更大尺寸输出。
- 可学习滤波器，重叠感受野贡献求和
- 无缝集成到 FCN 和 UNet 的 Decoder

---

## 9. 目标检测 (Object Detection)

**目标:** 输出每个对象的类别标签 + 边界框坐标 `(x, y, w, h)`。

---

## 10. R-CNN 系列演进

```
滑动窗口 → R-CNN → Fast R-CNN → Faster R-CNN
  慢                               快
```

| 方法 | 思路 |
|------|------|
| **R-CNN** | 提取区域提议 → 每个提议独立跑 CNN (慢但准) |
| **Fast R-CNN** | 整图一次 CNN → 共享特征图上提取特征 |
| **Faster R-CNN** | 引入 RPN 端到端生成提议 |

---

## 11. 区域提议网络 (RPN)

对每个 anchor 预测:
1. **Objectness Score** — 是否包含物体？
2. **Bounding Box Offsets** — 精调坐标

---

## 12. 单阶段检测器 (YOLO)

- S × S 网格，每格预测 B 个框 + 类别 + 置信度
- 一次前向传播完成所有预测
- NMS 去冗余
- **速度优先**

---

## 13. DETR (Detection Transformer)

```
图像特征 → Transformer Encoder → 上下文表示
                                        ↓
               Learned Object Queries → Transformer Decoder → Class + Box
```

| 组件 | 作用 |
|------|------|
| **Learned Object Queries** | 固定数量可学习向量，每个"问"模型预测一个物体 |
| **Bipartite Matching Loss** | 匈牙利算法匹配预测与 GT |
| **无需 NMS** | 优雅的端到端检测范式 |

---

## 14. 实例分割: Mask R-CNN

```
RoI Feature
    ├── Classification Branch → 类别得分
    ├── Box Regression Branch → 精调边界框
    └── Mask Branch → 二值分割掩码 (逐像素 BCE)
```

---

## 15. 可视化与可解释性

| 方法 | 原理 |
|------|------|
| **Saliency Maps** | 类别得分对输入像素的梯度 |
| **CAM** | 特征图 + 分类器权重的线性组合 |
| **Grad-CAM** | 用梯度空间平均值加权特征图 (通用) |
| **Attention Maps** | Transformer 的注意力权重矩阵 |

---

## 16. 核心要点总结

```
Transformers for Modern CV
├── ViT: Patch Tokenization → [CLS] or Pooling → Classification
├── 语义分割: FCN Encoder-Decoder → UNet Skip Connections
├── 目标检测: R-CNN → YOLO → DETR
├── 实例分割: Mask R-CNN = 检测 + Mask Head
├── 架构优化: Pre-Norm, RMSNorm, GLU, MoE
└── 可解释性: Saliency → CAM → Grad-CAM → Attention Maps
```

> **从分类到检测再到分割，视觉任务的空间精度逐步提升；Transformer 统一了建模范式；可视化方法贯穿始终。**
