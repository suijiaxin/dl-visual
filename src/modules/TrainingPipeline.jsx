import React, { useState } from 'react'
import { PageHeader, Card, Callout, Formula } from '../components/ui.jsx'

// 大模型训练全流程：预训练 → SFT → 偏好对齐(RLHF/DPO)。
// 这是"一堆网络权重"变成"会聊天、听指令的助手"的完整旅程。

const STAGES = [
  {
    id: 'pretrain',
    name: '① 预训练 Pre-training',
    color: 'var(--accent)',
    goal: '学会语言本身：语法、事实、推理的"底子"',
    data: '海量无标注文本（数万亿 token，整个互联网级别）',
    objective: '预测下一个 token（self-supervised，自监督）',
    output: 'Base 模型：知识渊博，但只会"续写"，不会听指令',
    cost: '最贵的一步：数千张 GPU 训练数周~数月，烧掉百万~千万美元',
    pmNote: '决定模型的知识上限和基础能力。绝大多数公司不做这步，直接用开源 base 模型。',
  },
  {
    id: 'sft',
    name: '② 监督微调 SFT',
    color: 'var(--green)',
    goal: '学会"听话"：把续写能力对齐到"按指令回答"的格式',
    data: '人工编写的高质量「指令-回答」对（几千~几十万条）',
    objective: '同样是预测下一个 token，但只在"回答"部分算 loss',
    output: 'Instruct 模型：能理解并遵循指令，像个助手了',
    cost: '相对便宜：几张~几十张 GPU，几小时~几天',
    pmNote: '这是大多数企业落地的主战场。数据质量 >> 数据数量。LoRA 微调通常用在这一步。',
  },
  {
    id: 'align',
    name: '③ 偏好对齐 RLHF / DPO',
    color: 'var(--accent-2)',
    goal: '学会"符合人类偏好"：更有用、更诚实、更无害',
    data: '人类对模型多个回答的偏好排序（A 比 B 好）',
    objective: 'RLHF：训奖励模型 + 强化学习；DPO：直接用偏好对优化（更简单）',
    output: 'Chat 模型：回答风格、安全性、价值观对齐人类期望（如 ChatGPT、Claude）',
    cost: '中等，但数据采集（人类标注偏好）很贵',
    pmNote: '决定产品的"调性"和安全底线。幻觉、拒答、语气都在这步被塑造。',
  },
]

export default function TrainingPipeline() {
  const [sel, setSel] = useState('pretrain')
  const [nextTokenDemo, setNextTokenDemo] = useState('注意力机制是 Transformer 的')
  const s = STAGES.find((x) => x.id === sel)

  // 模拟 next-token 预测的候选概率（写死一组直观的）
  const predictions = [
    { tok: '核心', p: 0.62 },
    { tok: '基础', p: 0.18 },
    { tok: '关键', p: 0.11 },
    { tok: '一部分', p: 0.05 },
    { tok: '玩具', p: 0.01 },
  ]

  return (
    <div>
      <PageHeader
        eyebrow="08 · TRAINING PIPELINE"
        title="训练全流程：base 模型如何变成 ChatGPT"
        lead="反向传播只是「怎么更新一次权重」的机制。而把一个随机初始化的网络，训练成像 ChatGPT 那样会聊天的助手，要经历三个性质完全不同的阶段。理解这三段，你就理解了大模型团队在做什么、钱花在哪、以及你的产品该介入哪一段。"
      />

      {/* 三阶段流程条 */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        {STAGES.map((st, i) => (
          <React.Fragment key={st.id}>
            <div
              onClick={() => setSel(st.id)}
              style={{
                flex: 1, minWidth: 200, cursor: 'pointer',
                background: sel === st.id ? st.color : 'var(--bg-card)',
                color: sel === st.id ? '#0b0e14' : 'var(--text)',
                border: `1.5px solid ${st.color}`, borderRadius: 10, padding: '14px 16px',
                transition: 'all 0.15s',
              }}
            >
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{st.name}</div>
              <div style={{ fontSize: 11.5, opacity: 0.85 }}>{st.goal}</div>
            </div>
            {i < STAGES.length - 1 && (
              <div style={{ alignSelf: 'center', color: 'var(--text-faint)', fontSize: 20 }}>→</div>
            )}
          </React.Fragment>
        ))}
      </div>

      <Card title={s.name} sub="点上面的阶段切换">
        <div style={{ height: 4, background: s.color, borderRadius: 2, marginBottom: 16 }} />
        <div className="grid-2">
          <div>
            <Row label="🎯 目标" val={s.goal} />
            <Row label="📦 数据" val={s.data} />
            <Row label="⚙️ 训练目标" val={s.objective} />
          </div>
          <div>
            <Row label="📤 产出" val={s.output} />
            <Row label="💰 成本" val={s.cost} />
          </div>
        </div>
        <Callout type="key">
          <b>给 PM 的视角：</b> {s.pmNote}
        </Callout>
      </Card>

      {sel === 'pretrain' && (
        <Card title="预训练的核心任务：预测下一个 token" sub="所有能力都从这一个简单目标中涌现">
          <Formula>给定前文，最大化 P(下一个 token | 前面所有 token)</Formula>
          <div style={{ marginBottom: 14 }}>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>输入前文（试着改改）：</div>
            <input type="text" value={nextTokenDemo} onChange={(e) => setNextTokenDemo(e.target.value)}
              style={{ width: '100%' }} />
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
            模型预测的下一个 token 概率分布（示意）：
          </div>
          {predictions.map((p, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <span style={{ width: 64, fontSize: 13, fontFamily: 'var(--mono)', color: i === 0 ? 'var(--green)' : 'var(--text-dim)', textAlign: 'right' }}>{p.tok}</span>
              <div style={{ flex: 1, height: 18, background: 'var(--bg-elev)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ width: `${p.p * 100}%`, height: '100%', background: i === 0 ? 'var(--green)' : 'var(--accent)' }} />
              </div>
              <span style={{ width: 44, fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-faint)' }}>{(p.p * 100).toFixed(0)}%</span>
            </div>
          ))}
          <Callout>
            就这么简单——把网上的文本盖住后半句，让模型猜。猜错了就反向传播调权重（回到模块 02 的机制）。
            重复几万亿次后，模型为了"猜得准"，被迫学会了语法、事实、翻译、推理……
            这就是"<b>预测下一个词</b>"为何能催生出通用智能的惊人之处。
          </Callout>
        </Card>
      )}

      {sel === 'sft' && (
        <Card title="SFT：从「续写」到「应答」" sub="同样的网络，换一种数据，就学会了听指令">
          <div className="grid-2">
            <div className="stat" style={{ borderColor: 'var(--orange)' }}>
              <div style={{ fontSize: 12, color: 'var(--orange)', marginBottom: 6 }}>Base 模型（仅预训练）</div>
              <div style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
                输入：「中国的首都是哪里？」<br />
                输出：「中国的首都是哪里？这道题很多人答错……」<br />
                <span style={{ color: 'var(--red)' }}>→ 它在"续写"，而不是回答你</span>
              </div>
            </div>
            <div className="stat" style={{ borderColor: 'var(--green)' }}>
              <div style={{ fontSize: 12, color: 'var(--green)', marginBottom: 6 }}>SFT 后（指令微调）</div>
              <div style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
                输入：「中国的首都是哪里？」<br />
                输出：「中国的首都是北京。」<br />
                <span style={{ color: 'var(--green)' }}>→ 它理解了"这是一个要回答的问题"</span>
              </div>
            </div>
          </div>
          <Callout>
            关键：SFT 用的训练数据是大量「<b>指令 → 理想回答</b>」对，且只在"回答"部分计算 loss。
            模型本质没变，只是被引导到"看到指令格式就进入应答模式"。
          </Callout>
        </Card>
      )}

      {sel === 'align' && (
        <Card title="RLHF vs DPO：让模型符合人类偏好" sub="SFT 教会了「怎么答」，对齐教会「怎么答得更好」">
          <div className="grid-2">
            <div>
              <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--accent-2)' }}>RLHF（经典三步）</div>
              <div className="prose" style={{ fontSize: 12.5 }}>
                <p>1. 让人对模型的多个回答打分排序</p>
                <p>2. 用这些排序训练一个「奖励模型」打分</p>
                <p>3. 用强化学习（PPO）优化主模型，让奖励最大化</p>
                <p style={{ color: 'var(--text-faint)' }}>强大但流程复杂、训练不稳定</p>
              </div>
            </div>
            <div>
              <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--green)' }}>DPO（更简单的新方法）</div>
              <div className="prose" style={{ fontSize: 12.5 }}>
                <p>跳过奖励模型和强化学习</p>
                <p>直接用「人类更喜欢 A 而非 B」的偏好对，</p>
                <p>用一个分类式损失直接优化模型</p>
                <p style={{ color: 'var(--text-faint)' }}>更稳定、更省资源，近年广泛采用</p>
              </div>
            </div>
          </div>
          <Formula>偏好数据：(prompt, 较好回答 y_w, 较差回答 y_l) → 让模型更倾向 y_w</Formula>
          <Callout type="warn">
            这一步直接塑造产品"性格"：是否健谈、是否轻易拒答、安全边界在哪、价值观如何。
            产品和安全团队的诉求，最终都要落到这一阶段的偏好数据里。
          </Callout>
        </Card>
      )}

      <Callout type="key">
        <b>完整心智模型：</b> 分词(01) 把文本变 token → 预训练让模型「博览群书」学到知识 →
        SFT 让它「学会听话」→ 对齐让它「符合人类偏好」。三段都在用模块 02 的反向传播，
        只是<b>数据和目标不同</b>。训练完成后，模型权重冻结，进入推理阶段(模块 10)。
        而企业落地最常碰的，是用 <b>LoRA 高效微调</b>(下一模块)在自己的数据上做 SFT。
      </Callout>
    </div>
  )
}

function Row({ label, val }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, color: 'var(--text-faint)', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 13, color: 'var(--text)' }}>{val}</div>
    </div>
  )
}
