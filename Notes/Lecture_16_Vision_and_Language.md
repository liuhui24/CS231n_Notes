#  Lecture 16: Vision and Language

---

## 1. 多模态基础模型

视觉-语言模型是**多模态基础模型**，位于 CV、NLP、机器人学和 HCI 的交叉点。

### 什么是基础模型？

大规模模型，在多样化数据上预训练，以最少微调广泛迁移到下游任务。

**核心属性:**
- 大参数量 + 可扩展架构 (Transformer)
- 互联网规模训练语料
- 通用预训练目标
- 灵活适配: 微调 / few-shot / zero-shot

---

## 2. CLIP: 对比多模态预训练

### 核心洞察
> **文本只是图像的另一种视图** — "一只可爱的毛茸茸的猫" 的嵌入应靠近猫图像。

### CLIP 架构

```
图像 → Image Encoder (ViT/ResNet) → z_i (归一化向量)
文本 → Text Encoder (Transformer)  → t_i (归一化向量)
          ↓
对称对比损失 (InfoNCE)
```

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{\text{img→text}} + \mathcal{L}_{\text{text→img}})$$

- 对称性**至关重要** — 必须双向约束以避免几何坍缩
- **数据优势:** 互联网图文对丰富且廉价 (alt-text, 标题)

### Zero-Shot 分类

1. 填入 prompt 模板: `"a photo of a {label}"`
2. Text Encoder 编码 → 类别原型向量
3. Image Encoder 编码测试图像
4. 最近邻分类: $\text{logits} = z_{\text{test}}^T W_{\text{zs}}$

### Prompt Engineering 至关重要

- `"a photo of a dog"` 远优于 `"dog"`
- **Ensemble** 多个模板平均嵌入 → 进一步提分

### CLIP 的局限

| 局限 | 原因 |
|------|------|
| Batch 敏感性 | 对比损失需要大量负样本 |
| 组合性失败 | "杯子在草地上" vs "草在杯子里" → Bag-of-words 行为 |
| 无空间定位 | 图像级标题缺乏像素级监督 |
| 无生成输出 | 只能产生相似度分数 |

---

## 3. CoCa: 对比 + 描述目标

扩展 CLIP，添加**生成式描述 decoder**:

$$\mathcal{L}_{\text{CoCa}} = \lambda_{\text{con}}\mathcal{L}_{\text{contrastive}} + \lambda_{\text{cap}}\mathcal{L}_{\text{captioning}}$$

- 生成描述迫使模型学习更丰富的属性 (颜色、形状、空间关系)

---

## 4. 多模态 LLM: 将视觉注入语言模型

### 核心思路

$$\text{Input} = [\text{Visual Tokens}, \text{Text Tokens}]$$

### LLaVA
- 使用 CLIP ViT**倒数第二层**特征 (保留空间细节)
- 简单**线性投影** (connector) 映射视觉特征到 LLM 嵌入空间
- **大部分权重冻结** — 仅训练 connector

### Flamingo: 深度 Cross-Attention 融合

| 组件 | 功能 |
|------|------|
| **Perceiver Resampler** | 压缩可变大小视觉特征为固定数量 token |
| **Gated XATTN-DENSE** | 在**每个** LLM 层插入 Cross-Attention |
| **Tanh Gating** | 门初始化为 0 → 从纯语言开始，逐渐融入视觉 |

> 支持**上下文多模态学习** — 类比 GPT-3 的 in-context learning 但用于视觉任务。

---

## 5. Momo: 缩小开源差距

2024-2025 年，闭源模型远超开源模型 (LLaVA ~43% vs 70-80%)。

### Momo 的贡献

1. **数据质量 > 数量:** 70 万密集标注样本匹配了训练于 60 亿对的模型
2. **像素级定位:** 模型必须输出**指向坐标** — 先有视觉证据再生成文本 → 减少幻觉
3. **密集描述标签:** 包含位置、材质、形状、空间关系
4. **口语标注收集:** 说话而非打字 → 更丰富、更少刻板的描述
5. **完全开源:** 权重 + 代码 + 数据

---

## 6. SAM: Segment Anything Model

将分割框架化为**可 prompt 的接口**:

### 架构
- **Image Encoder** (重, 运行一次): 产生稠密特征图 F
- **Prompt Encoder** (轻): 编码点/框/文本 → prompt tokens P
- **Mask Decoder** (轻): $(F, P) \rightarrow$ 分割掩码

### 处理模糊性: 一键三掩码
- 输出**3 个掩码** (整体/部件/子部件粒度)
- **"Winner-take-all" 训练:** 只反传最接近 GT 的掩码
- **IoU 预测头:** 对 3 个掩码按质量排序

### 数据引擎: Human-in-the-Loop
```
人工种子标注 → 训练模型 → 模型提议 → 人工修正 → 重复
```
→ 指数级吞吐量提升 → **SA-1B: 11M 图像, 1.1B 掩码**

---

## 7. 模型链式组合

### Knowledge Chaining
LLM + CLIP: 对 CLIP 未见的稀有类别，用 GPT 生成视觉描述 → CLIP 文本 encoder 做 zero-shot 分类。

### Visual Programming (VisProg)
对复杂多步查询:
- LLM **生成可执行代码**调用专用视觉 API (检测器、OCR、SAM)
- LLM 作为**任务分解器和工具调度器**，而非直接感知器
- 通过外包感知给专用模型消除幻觉

---

## 核心要点

| 主题 | 洞察 |
|------|------|
| **对比对齐** | CLIP 的对称 InfoNCE 创建共享图文嵌入空间 |
| **Prompt Engineering** | 模板设计和 ensemble 显著影响 CLIP 性能 |
| **生成目标** | 添加描述 (CoCa) 产生比纯对比更丰富的特征 |
| **架构权衡** | LLaVA 简单 prefix vs Flamingo 深度门控 Cross-Attention |
| **数据中心转向** | Momo 证明策展质量可胜过原始规模 |
| **可 prompt 接口** | SAM 的设计模式 (重 encoder + 轻 prompt decoder) |
| **组合** | LLM 编排的基础模型链解决单一模型无法处理的任务 |
