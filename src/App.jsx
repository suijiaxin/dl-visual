import React, { useState, useEffect } from 'react'
import Overview from './modules/Overview.jsx'
import NeuralNet from './modules/NeuralNet.jsx'
import Backprop from './modules/Backprop.jsx'
import CNN from './modules/CNN.jsx'
import RNNLSTM from './modules/RNNLSTM.jsx'
import Attention from './modules/Attention.jsx'
import TransformerArch from './modules/TransformerArch.jsx'
import Tokenizer from './modules/Tokenizer.jsx'
import TrainingPipeline from './modules/TrainingPipeline.jsx'
import FineTuning from './modules/FineTuning.jsx'
import Inference from './modules/Inference.jsx'

// 模块按学习顺序排列：先讲为什么演变，再从基础往上搭，
// 再到 Transformer 核心，最后走一遍大模型的完整生命周期（数据→训练→微调→推理）
const NAV = [
  {
    group: '总览',
    items: [{ id: 'overview', num: '00', label: '架构演变史', comp: Overview }],
  },
  {
    group: '神经网络基础',
    items: [
      { id: 'nn', num: '01', label: '神经元与前向传播', comp: NeuralNet },
      { id: 'backprop', num: '02', label: '反向传播 · 优化器', comp: Backprop },
    ],
  },
  {
    group: '经典架构',
    items: [
      { id: 'cnn', num: '03', label: 'CNN 卷积网络', comp: CNN },
      { id: 'rnn', num: '04', label: 'RNN / LSTM 时序网络', comp: RNNLSTM },
    ],
  },
  {
    group: 'Transformer 核心',
    items: [
      { id: 'attention', num: '05', label: '注意力机制', comp: Attention },
      { id: 'transformer', num: '06', label: 'Transformer 整体架构', comp: TransformerArch },
    ],
  },
  {
    group: '大模型生命周期',
    items: [
      { id: 'tokenizer', num: '07', label: '分词 Tokenization', comp: Tokenizer },
      { id: 'training', num: '08', label: '训练全流程 · Pretrain→对齐', comp: TrainingPipeline },
      { id: 'finetune', num: '09', label: '微调 · LoRA / QLoRA', comp: FineTuning },
      { id: 'inference', num: '10', label: '推理 · Prefill 与 KV Cache', comp: Inference },
    ],
  },
]

const ALL = NAV.flatMap((g) => g.items)

// 读取保存的主题；默认深色（夜晚）
function getInitialTheme() {
  if (typeof localStorage !== 'undefined') {
    const saved = localStorage.getItem('dlv-theme')
    if (saved === 'light' || saved === 'dark') return saved
  }
  return 'dark'
}

export default function App() {
  const [active, setActive] = useState('overview')
  const [theme, setTheme] = useState(getInitialTheme)
  const Current = ALL.find((i) => i.id === active)?.comp || Overview

  // 把主题写到 <html data-theme> 上，CSS 变量随之切换；并持久化
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('dlv-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>深度学习架构可视化</h1>
          <p>从神经元到 Transformer 推理 · 边调边学</p>
        </div>
        {NAV.map((g) => (
          <div key={g.group}>
            <div className="nav-group-label">{g.group}</div>
            {g.items.map((it) => (
              <div
                key={it.id}
                className={`nav-item ${active === it.id ? 'active' : ''}`}
                onClick={() => setActive(it.id)}
              >
                <span className="nav-num">{it.num}</span>
                {it.label}
              </div>
            ))}
          </div>
        ))}
      </aside>
      <main className="main">
        <div className="main-inner">
          <Current key={active} onNav={setActive} />
        </div>
      </main>
      <button
        className="theme-toggle"
        onClick={toggleTheme}
        title={theme === 'dark' ? '切换到白天模式' : '切换到夜晚模式'}
        aria-label="切换主题"
      >
        <span className="toggle-icon">{theme === 'dark' ? '☀️' : '🌙'}</span>
      </button>
    </>
  )
}
