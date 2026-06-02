# Lecture 11: Large Scale Distributed Training

---

## 1. 为什么需要分布式训练？

十年前，模型在单个 GPU 上训练。今天，训练通常跨越数万设备。

### 案例: Llama 3 (405B 参数)

| 组件 | 规格 |
|------|------|
| GPU | 24,576 个 H100 |
| 集群 | 8 个 Pod, 每个 192 机架 |
| 训练时间 | 数月 |

---

## 2. GPU 硬件深度解析 (NVIDIA H100)

| 组件 | 规格 | 备注 |
|------|------|------|
| **HBM** | 80 GB, ~3 TB/s 带宽 | 高容量高延迟 |
| **L2 Cache** | ~50 MB | 片上大容量低延迟缓存 |
| **SM** | 132 个 | 包含 FP32 核心 + Tensor Cores |
| **Tensor Cores** | 专用矩阵乘加单元 | 混合精度运算 (低精输入, FP32 累加) |

> 软件若**未将张量转为 16-bit** → 回退到慢速 FP32 核心 → 大幅减速。始终使用混合精度！

---

## 3. 集群内存与互连层级

带宽在每一层急剧下降:

```
GPU 内部 (HBM ↔ Compute):  ~3 TB/s
GPU 间 (NVLink):            ~900 GB/s
Pod 内 (192 机架, 3072 GPU): ~50 GB/s
跨 Pod:                     < 50 GB/s
```

**核心原则:** 算法必须最小化跨节点流量，将重通信放在快速本地链路上。

---

## 4. 并行化问题陈述

Transformer 的计算是一个 **3D 张量: (Batch × Sequence × Hidden-Dim)**，跨 L 层堆叠。

### 四种并行化轴

| 轴 | 策略 | 切分对象 |
|------|------|---------|
| Batch | **Data Parallelism (DP)** | Mini-batch 分到各设备 |
| Sequence | **Context Parallelism (CP)** | 长序列分到各设备 |
| Hidden-Dim | **Tensor Parallelism (TP)** | 权重矩阵分到各设备 |
| Layers | **Pipeline Parallelism (PP)** | 模型层分到各设备 |

**目标:** 分区这些轴，将通信映射到硬件拓扑，重叠通信与计算。

---

## 5. Data Parallelism (DP) — 最简单策略

```
每 GPU 复制完整模型
→ 分配不同 mini-batch 分片
→ 独立前向 + 反向
→ All-Reduce 平均梯度
→ 同步权重更新
```

**数学正确性:** 梯度平均是精确的——梯度是线性的。

**优化:** 重叠反向计算与梯度通信 (`DistributedDataParallel`)

**局限:** 模型必须完全放入**单个 GPU** (参数 + 梯度 + 优化器状态)

---

## 6. Fully Sharded Data Parallelism (FSDP)

**核心思想:** 将模型参数、梯度、优化器状态**分片**到所有 GPU。

### 前向传播 (每层 i)
1. **All-Gather:** 参数所有者广播 W_i 给所有 GPU
2. **计算:** 所有 GPU 用各自数据分片计算激活
3. **丢弃:** 非所有者删除 W_i 节省内存
4. **预取优化:** 计算层 i 时异步获取 W_{i+1}

### 反向传播 (每层 i)
1. **All-Gather:** 所有者重新广播 W_i
2. **计算:** 所有 GPU 计算本地梯度
3. **Reduce-Scatter:** 将梯度发回所有者聚合
4. **更新:** 所有者更新其参数分片

### DP vs FSDP

| | DP | FSDP |
|------|------|------|
| 模型存储 | 每 GPU 完整副本 | 分片到所有 GPU |
| 内存瓶颈 | 单 GPU VRAM | 所有 GPU VRAM 之和 |
| 通信频率 | 每次迭代一次 (梯度) | 每层前向+反向 (权重+梯度) |
| 适用场景 | 模型能放进单 GPU | 模型太大无法放入单 GPU |

---

## 7. Hybrid Sharded DP (HSDP)

将 GPU 排列成 **2D 网格**:
- **组内 FSDP** (快速 NVLink): 权重分片
- **组间 DP** (较慢链路): 宏批量梯度 All-Reduce

> 重 FSDP 通信留在快速 NVLink 上; 轻 DP All-Reduce 跨较慢链路。数百 GPU 以上扩展的关键。

---

## 8. Activation Checkpointing

### 问题
激活值主导 GPU 内存。标准训练存储**所有**中间激活值 (N 层 → O(N) 内存)。

### 策略

| 策略 | 计算复杂度 | 内存 | 说明 |
|------|-----------|------|------|
| 标准 | O(N) | O(N) | 存储所有激活值 |
| 全部重算 | O(N²) | O(1) | 每层从头重算 |
| 每 C 层 checkpoint | O(N²/C) | O(C) | 存储 checkpoint, 段内重算 |
| **√N checkpoints (最优)** | **O(N√N)** | **O(√N)** | C ≈ √N, 最佳平衡 |

> 用 ~30-50% 额外计算换大幅内存减少，使更大模型和更长序列成为可能。

---

## 9. 扩展方案分级

| 阶段 | 策略 | 使用条件 |
|------|------|---------|
| 1 | **简单 DP** | ~128 GPU, ~1B 参数 |
| 2 | **FSDP** | 模型参数超过单 GPU 内存 |
| 3 | **Activation Checkpointing** | 激活值主导内存 |
| 4 | **HSDP** | 数百 GPU |
| 5 | **TP + PP + CP** | 极大模型 (100B+ 参数) |

---

## 10. 性能指标: MFU

### Model FLOPS Utilization (核心指标)

$$\text{MFU} = \frac{\text{每次迭代的理论 FLOP 数}}{\text{设备峰值 FLOPS × 实测迭代时间}}$$

- 好: **>30%**
- 优秀: **>40%**
- Llama 3 在 24K GPU 上达到 ~38-42% MFU
- MFU 是调优 batch size、并行维度、checkpointing 的**首要指标**

---

## 11. Context Parallelism (CP)

沿序列维度分区，每个 GPU 处理一个子序列。

| 层类型 | 并行难度 | 原因 |
|--------|---------|------|
| LayerNorm / Residual | 容易 | 逐 token 操作 |
| MLP / QKV 投影 | 容易 | Token 独立 |
| **Attention** | **困难** | 每个 Q 需要所有 K, V |

### 解决方案
- **Ring Attention:** 循环数据流 — 每个 GPU 固定 Q 块，循环接收 K/V 块
- **Ulysses:** 沿 head 维度分割而非序列维度

---

## 12. Pipeline Parallelism (PP)

沿层维度分区: GPU k 拥有层 `[k·L/P, (k+1)·L/P]`。

### Bubble 问题
朴素 PP 使大多数 GPU 空闲 — 最大 MFU 上限为 **1/P**。

### 解决方案: Micro-Batching
将 batch 分成 M 个 micro-batch，交错注入 pipeline → 填满空闲时间。

---

## 13. Tensor Parallelism (TP)

在权重矩阵内进行层内并行。

### 连续层配对技巧
- 层 1: 列式切分 W
- 层 2: 行式切分 U
- 结果: 通信被**延迟**到末尾，只需一次 reduce-scatter

---

## 14. N 维并行 — 全部组合

```
总 GPU 数 N = n_DP × n_PP × n_CP × n_TP
```

每个 GPU 有**4D 坐标** — 同时属于 DP 组、PP 阶段、CP 切片、TP 组。

### 典型生产方案 (Llama 3)
- **节点内 TP** (8 GPU via NVLink)
- **节点间 PP**
- **全局 DP** 复制 pipeline
- **CP 在需要时** 用于极长序列
