import React, { useState, useEffect, useRef } from 'react'
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

import MLOverview from './modules/ml/MLOverview.jsx'
import LinearRegression from './modules/ml/LinearRegression.jsx'
import LogisticRegression from './modules/ml/LogisticRegression.jsx'
import DecisionTree from './modules/ml/DecisionTree.jsx'
import RandomForest from './modules/ml/RandomForest.jsx'
import Boosting from './modules/ml/Boosting.jsx'
import KNN_SVM from './modules/ml/KNN_SVM.jsx'
import Clustering from './modules/ml/Clustering.jsx'
import DimReduction from './modules/ml/DimReduction.jsx'
import TimeSeries from './modules/ml/TimeSeries.jsx'
import Scorecard from './modules/ml/Scorecard.jsx'
import FeatureEngineering from './modules/ml/FeatureEngineering.jsx'
import IrisClassification from './modules/ml/IrisClassification.jsx'
import Titanic from './modules/ml/Titanic.jsx'
import BostonHousing from './modules/ml/BostonHousing.jsx'
import MallCustomers from './modules/ml/MallCustomers.jsx'
import AirPassengers from './modules/ml/AirPassengers.jsx'

// 深度学习导航：按学习顺序排列 —— 先讲为什么演变，再从基础往上搭，
// 再到 Transformer 核心，最后走一遍大模型的完整生命周期（数据→训练→微调→推理）
const DL_NAV = [
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
      { id: 'rnn', num: '04', label: 'RNN / LSTM / GRU 时序网络', comp: RNNLSTM },
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

// 机器学习导航：从架构演进 → 监督学习（回归/分类）→ 集成 → 无监督 → 时序 → 风控应用
const ML_NAV = [
  {
    group: '总览',
    items: [{ id: 'ml-overview', num: '00', label: '演进史 · 场景地图', comp: MLOverview }],
  },
  {
    group: '监督学习 · 基础',
    items: [
      { id: 'ml-linreg', num: '01', label: '线性回归', comp: LinearRegression },
      { id: 'ml-logreg', num: '02', label: '逻辑回归 · 分类', comp: LogisticRegression },
    ],
  },
  {
    group: '树模型与集成',
    items: [
      { id: 'ml-tree', num: '03', label: '决策树', comp: DecisionTree },
      { id: 'ml-forest', num: '04', label: '随机森林 · Bagging', comp: RandomForest },
      { id: 'ml-boost', num: '05', label: '梯度提升 · GBDT/XGBoost', comp: Boosting },
    ],
  },
  {
    group: '距离与边界',
    items: [
      { id: 'ml-knnsvm', num: '06', label: 'KNN 与 SVM', comp: KNN_SVM },
    ],
  },
  {
    group: '无监督学习',
    items: [
      { id: 'ml-cluster', num: '07', label: 'K-Means 聚类', comp: Clustering },
      { id: 'ml-pca', num: '08', label: 'PCA 降维', comp: DimReduction },
    ],
  },
  {
    group: '时序与应用',
    items: [
      { id: 'ml-ts', num: '09', label: '时序预测', comp: TimeSeries },
      { id: 'ml-scorecard', num: '10', label: '评分卡 · 模型评估', comp: Scorecard },
    ],
  },
  {
    group: '特征工程与实战',
    items: [
      { id: 'ml-feateng', num: '11', label: '特征工程', comp: FeatureEngineering },
      { id: 'ml-iris', num: '12', label: '实战 · 鸢尾花分类', comp: IrisClassification },
      { id: 'ml-titanic', num: '13', label: '实战 · 泰坦尼克生还', comp: Titanic },
      { id: 'ml-boston', num: '14', label: '实战 · 波士顿房价', comp: BostonHousing },
      { id: 'ml-mall', num: '15', label: '实战 · 客户分群', comp: MallCustomers },
      { id: 'ml-air', num: '16', label: '实战 · 航空客流', comp: AirPassengers },
    ],
  },
]

const DOMAINS = {
  dl: { title: '深度学习架构可视化', nav: DL_NAV, first: 'overview' },
  ml: { title: '机器学习架构可视化', nav: ML_NAV, first: 'ml-overview' },
}

// 读取保存的主题；默认深色（夜晚）
function getInitialTheme() {
  if (typeof localStorage !== 'undefined') {
    const saved = localStorage.getItem('dlv-theme')
    if (saved === 'light' || saved === 'dark') return saved
  }
  return 'dark'
}

export default function App() {
  const [domain, setDomain] = useState('dl')
  const [active, setActive] = useState(DOMAINS.dl.first)
  const [theme, setTheme] = useState(getInitialTheme)
  const [menuOpen, setMenuOpen] = useState(false)
  const brandRef = useRef(null)

  const nav = DOMAINS[domain].nav
  const all = nav.flatMap((g) => g.items)
  const Current = all.find((i) => i.id === active)?.comp || DOMAINS[domain].nav[0].items[0].comp

  // 把主题写到 <html data-theme> 上，CSS 变量随之切换；并持久化
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('dlv-theme', theme)
  }, [theme])

  // 点击品牌区下拉之外的地方时关闭菜单
  useEffect(() => {
    if (!menuOpen) return
    const onClick = (e) => {
      if (brandRef.current && !brandRef.current.contains(e.target)) setMenuOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [menuOpen])

  const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

  const switchDomain = (d) => {
    setDomain(d)
    setActive(DOMAINS[d].first)
    setMenuOpen(false)
  }

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-brand" ref={brandRef}>
          <button
            className="brand-switcher"
            onClick={() => setMenuOpen((o) => !o)}
            aria-haspopup="true"
            aria-expanded={menuOpen}
          >
            <h1>{DOMAINS[domain].title}</h1>
            <span className={`brand-caret ${menuOpen ? 'open' : ''}`}>▾</span>
          </button>
          <a
            className="brand-github"
            href="https://github.com/suijiaxin/dl-visual"
            target="_blank"
            rel="noopener noreferrer"
            title="在 GitHub 上查看源码"
            aria-label="在 GitHub 上查看源码"
          >
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </a>

          {menuOpen && (
            <div className="brand-menu" role="menu">
              {Object.entries(DOMAINS).map(([key, d]) => (
                <button
                  key={key}
                  className={`brand-menu-item ${domain === key ? 'active' : ''}`}
                  onClick={() => switchDomain(key)}
                  role="menuitem"
                >
                  <span className="brand-menu-dot" />
                  {d.title}
                  {domain === key && <span className="brand-menu-check">✓</span>}
                </button>
              ))}
            </div>
          )}
        </div>
        <nav className="sidebar-nav">
          {nav.map((g) => (
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
        </nav>
      </aside>
      <main className="main">
        <div className="main-inner">
          <Current key={`${domain}-${active}`} onNav={setActive} />
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
