<div align="center">

# 🧠 深度学习架构可视化

**从神经元到 Transformer 推理，一条可交互的学习主线 —— 每个概念都能调参数、看真实计算结果。**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-贡献)

</div>

---

跑的是**真实的数学运算**，不是预录的动画。真的 softmax、真的反向传播、真的 KV Cache 增长曲线 —— 改参数就能看到数值实时变化。面向想系统理解大模型底层逻辑的人，尤其是训推方向的 AI 产品经理。

## 目录

- [特性](#-特性)
- [快速开始](#-快速开始)
- [学习路线](#-学习路线)
- [项目结构](#-项目结构)
- [自定义](#-自定义)
- [技术栈](#-技术栈)
- [贡献](#-贡献)
- [许可证](#-许可证)

## ✨ 特性

- **真实计算驱动** —— 每个可视化背后都是实打实的数学运算，改参数 → 看真实数值变化。
- **完整学习主线** —— 从单个神经元一路串到大模型的训推全生命周期，左侧导航顺序即建议学习顺序。
- **纯前端零后端** —— Vite + React，原生 SVG 画图，无图表库依赖，开箱即跑。
- **PM 友好** —— 训练侧关注数据/算力/收敛，推理侧关注首字延迟(TTFT)/吞吐(TPS)/显存。
- **明暗主题** —— 一键切换，偏好自动持久化。
- **易于二次开发** —— 纯 JavaScript（无 TypeScript），模块互相独立，改一处立即热更新。

## 🚀 快速开始

```bash
npm install      # 安装依赖（首次较慢）
npm run dev      # 启动开发服务器，自动打开 http://localhost:5180
```

其他命令：

```bash
npm run build    # 打包到 dist/
npm run preview  # 本地预览打包结果
```

> 需要 Node.js 18+。

## 🗺 学习路线

分成两大块：前半段理解**架构怎么一步步演进到 Transformer**，后半段走完一条真实大模型从**原始文本到线上服务**的完整生命周期。

### 基础与架构演进

| 编号 | 模块 | 你会学到 / 能动手做什么 |
|:----:|------|------------------------|
| 00 | **架构演变史** | 从感知机到大模型的时间线 + 生命周期预览图 |
| 01 | **神经元与前向传播** | 拖输入、换激活函数，看数值在网络里实时流动 |
| 02 | **反向传播 · 优化器** | 真实训练 MLP；调学习率、切换 SGD/Momentum/Adam 看收敛差异 |
| 03 | **CNN 卷积网络** | 画图案、选卷积核，看特征图实时响应 |
| 04 | **RNN / LSTM** | 时序展开；亲眼看 RNN 梯度消失、LSTM 门控 |
| 05 | **注意力机制** | 改句子，看真实的 Q·Kᵀ/√d → softmax 注意力热图 |
| 06 | **Transformer 整体架构** | 点击数据流图每个组件；对比 Decoder-Only 与 Encoder-Decoder |

### 大模型生命周期（训推全流程）

| 编号 | 模块 | 你会学到 / 能动手做什么 |
|:----:|------|------------------------|
| 07 | **分词 Tokenization** | 对比字符/词/子词粒度；看真实 BPE 合并过程 |
| 08 | **训练全流程** | 预训练 → SFT → RLHF/DPO 对齐，每阶段的数据/目标/成本/PM 视角 |
| 09 | **微调 · LoRA / QLoRA** | 调 rank 看可训练参数量骤降；全量/LoRA/QLoRA 选型决策 |
| 10 | **推理 · Prefill 与 KV Cache** | 自回归生成动画；对比有/无 KV Cache 的算力差距 |

> 这条链路刻意覆盖了 PM 最该懂的全貌：**文本怎么变数字 → 怎么训出 base 模型 → 怎么对齐成助手 → 怎么低成本微调 → 怎么高效推理**。

## 📁 项目结构

```
src/
├── App.jsx              # 导航 + 模块路由，想增删模块改这里
├── lib/mathx.js         # 共享数学库：softmax、矩阵乘、激活函数、权重初始化
│                        #   👉 改这里，所有模块的计算会跟着变
├── components/ui.jsx    # 共享 UI 组件：Slider / Card / Callout / 热力色
├── styles/global.css    # 全局样式与配色变量（顶部 :root 改主题色）
└── modules/             # 11 个教学模块，每个独立、互不依赖
    ├── Overview.jsx         # 演变时间线 + 生命周期预览（改 TIMELINE 数组即可加节点）
    ├── NeuralNet.jsx        # 改 LAYERS 改网络结构
    ├── Backprop.jsx         # 改 TARGETS 加拟合目标；改 OPTIMIZERS 改优化器；改 H 改隐藏层宽度
    ├── CNN.jsx              # 改 KERNELS 加新卷积核
    ├── RNNLSTM.jsx          # 改 LSTM 门权重看门控行为
    ├── Attention.jsx        # 改 D（向量维度）、DEFAULT_TOKENS
    ├── TransformerArch.jsx  # 改 COMPONENTS 改组件说明
    ├── Tokenizer.jsx        # 改 GRANULARITY；runBPE 是真实的 BPE 实现
    ├── TrainingPipeline.jsx # 改 STAGES 改训练三阶段的说明
    ├── FineTuning.jsx       # 改 dim/rank 看 LoRA 参数量；公式在 stats 里
    └── Inference.jsx        # 改 PROMPT / GEN 改演示序列
```

## 🔧 自定义

几个改起来最直观、改完立刻在页面看到效果的入口：

- **加一个新的激活函数** —— 在 `src/lib/mathx.js` 的 `activations` 对象里加一项 `{ fn, d, label, tex }`，「神经元」模块的下拉框会自动出现它。
- **加一个新的拟合目标** —— 在 `src/modules/Backprop.jsx` 的 `TARGETS` 里加 `{ fn: (x) => ..., label: '...' }`，训练模块下拉框自动出现。
- **加一个新的卷积核** —— 在 `src/modules/CNN.jsx` 的 `KERNELS` 里加一个 3×3 矩阵。
- **改注意力的输入句子** —— 页面上直接在输入框改，或改 `Attention.jsx` 的 `DEFAULT_TOKENS`。
- **改主题配色** —— 改 `src/styles/global.css` 顶部 `:root` 的 CSS 变量。

改完保存，Vite 会热更新，浏览器立即刷新 —— 改代码 → 看流程变化的闭环就是这么转的。

## 🛠 技术栈

**Vite 6** + **React 18**，纯 JavaScript（无 TypeScript），可视化用原生 SVG，零图表库依赖。整个项目刻意保持轻量，方便你读懂每一行、随手改。

## 🤝 贡献

欢迎 Issue 和 PR！无论是补充新模块、修正某个计算细节，还是改进交互，都很受欢迎。

1. Fork 本仓库并新建分支：`git checkout -b feature/your-idea`
2. 提交改动：`git commit -m 'Add some feature'`
3. 推送分支：`git push origin feature/your-idea`
4. 发起 Pull Request

新增可视化模块时，建议保持「真实计算」的原则 —— 数值应来自真实运算而非硬编码，这样调参数才有意义。

## 📄 许可证

本项目基于 [Apache License 2.0](./LICENSE) 开源。
