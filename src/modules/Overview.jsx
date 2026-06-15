import React, { useState } from 'react'
import { PageHeader, Card, Callout } from '../components/ui.jsx'

// 架构演变时间线 —— 每个节点的核心是"上一代有什么痛点，这一代怎么解决"
const TIMELINE = [
  {
    year: '1958',
    name: '感知机 Perceptron',
    color: 'var(--text-faint)',
    idea: '单个神经元：输入加权求和，过一个阈值输出 0/1。',
    pain: '只能学线性可分问题，连 XOR 都解不了。',
    nav: 'nn',
  },
  {
    year: '1986',
    name: '多层感知机 MLP + 反向传播',
    color: 'var(--green)',
    idea: '堆叠隐藏层 + 非线性激活，用反向传播算法高效求梯度，让深层网络可训练。',
    pain: '全连接对图像/序列效率极低，参数爆炸，且不利用数据的空间/时序结构。',
    nav: 'backprop',
  },
  {
    year: '1998 / 2012',
    name: 'CNN 卷积网络',
    color: 'var(--accent)',
    idea: '用共享的小卷积核扫描图像，捕捉局部特征 + 参数共享。2012 年 AlexNet 引爆深度学习。',
    pain: '擅长空间结构（图像），但不擅长处理变长的时序依赖（语言）。',
    nav: 'cnn',
  },
  {
    year: '1990s / 1997 / 2014',
    name: 'RNN → LSTM → GRU',
    color: 'var(--orange)',
    idea: 'RNN 用隐状态沿时间步传递记忆；LSTM 加入三道门控解决梯度消失；GRU 把门精简成两道，更轻更快。',
    pain: '必须按时间步串行计算，无法并行；超长序列依然吃力。',
    nav: 'rnn',
  },
  {
    year: '2017',
    name: 'Transformer · Attention Is All You Need',
    color: 'var(--accent-2)',
    idea: '彻底丢掉循环，用自注意力让每个 token 直接看到所有其他 token，可大规模并行训练。',
    pain: '注意力计算量随序列长度平方增长；推理时 KV cache 占显存。',
    nav: 'attention',
  },
  {
    year: '2018 →',
    name: 'GPT / BERT 及大模型时代',
    color: 'var(--pink)',
    idea: '基于 Transformer 堆叠 + 海量数据预训练。Decoder-only 的 GPT 路线主导了生成式大模型。',
    pain: '推理成本、上下文长度、对齐与幻觉 —— 当下工程与研究的主战场。',
    nav: 'inference',
  },
]

export default function Overview({ onNav }) {
  const [open, setOpen] = useState(4) // 默认展开 Transformer 节点

  return (
    <div>
      <PageHeader
        eyebrow="00 · OVERVIEW"
        title="架构演变史：每一代都在补上一代的短板"
        lead="深度学习不是一蹴而就的。理解每个架构「为了解决什么问题而生」，比记住它的结构更重要。下面这条主线，会把你提到的所有名词串起来。点击任一节点展开，或直接跳到对应模块动手实验。"
      />

      <Card title="一句话主线" sub="把整条脉络浓缩成一句话">
        <div className="prose">
          <p>
            <strong>感知机</strong>证明了「神经元」可以学习 → <strong>MLP + 反向传播</strong>
            让深层网络能训练 → <strong>CNN</strong> 高效处理空间结构、
            <strong>RNN/LSTM</strong> 处理时序 → <strong>Transformer</strong>{' '}
            用注意力机制取代循环、实现大规模并行 → 大模型时代的核心矛盾从「能不能学」变成了
            <strong>「推理怎么又快又省」</strong>（这就是 prefill / KV cache 要解决的）。
          </p>
        </div>
      </Card>

      <div className="card">
        <div className="card-title">演变时间线</div>
        <div className="card-sub">点击节点展开「核心思想」与「遗留痛点」——痛点正是下一代的出发点</div>
        <div style={{ position: 'relative', paddingLeft: 8 }}>
          {TIMELINE.map((t, i) => (
            <div key={i} style={{ display: 'flex', gap: 16, marginBottom: 4 }}>
              {/* 左侧时间轴线 */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div
                  style={{
                    width: 16,
                    height: 16,
                    borderRadius: '50%',
                    background: t.color,
                    flexShrink: 0,
                    marginTop: 16,
                    boxShadow: `0 0 0 4px ${t.color}22`,
                  }}
                />
                {i < TIMELINE.length - 1 && (
                  <div style={{ width: 2, flex: 1, background: 'var(--border)', minHeight: 24 }} />
                )}
              </div>
              {/* 右侧卡片 */}
              <div
                style={{
                  flex: 1,
                  background: open === i ? 'var(--bg-elev)' : 'transparent',
                  border: '1px solid',
                  borderColor: open === i ? 'var(--border)' : 'transparent',
                  borderRadius: 10,
                  padding: open === i ? '14px 16px' : '12px 16px',
                  marginBottom: 10,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onClick={() => setOpen(open === i ? -1 : i)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span className="tag" style={{ color: t.color, borderColor: t.color }}>
                    {t.year}
                  </span>
                  <strong style={{ fontSize: 14.5 }}>{t.name}</strong>
                </div>
                {open === i && (
                  <div className="fade-in" style={{ marginTop: 12, fontSize: 13 }}>
                    <p style={{ color: 'var(--text-dim)', marginBottom: 8 }}>
                      <span style={{ color: 'var(--green)' }}>💡 核心思想：</span>
                      {t.idea}
                    </p>
                    <p style={{ color: 'var(--text-dim)', marginBottom: 12 }}>
                      <span style={{ color: 'var(--orange)' }}>⚠️ 遗留痛点：</span>
                      {t.pain}
                    </p>
                    <button
                      className="btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        onNav(t.nav)
                      }}
                    >
                      去对应模块动手 →
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">大模型的完整生命周期</div>
        <div className="card-sub">理解了结构演变后，最后一个分区会带你走完一条真实大模型从「原始文本」到「线上服务」的链路</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          {[
            { t: '分词', d: '文本→token', c: 'var(--cyan)', nav: 'tokenizer' },
            { t: '预训练', d: '学知识', c: 'var(--accent)', nav: 'training' },
            { t: 'SFT', d: '学听话', c: 'var(--green)', nav: 'training' },
            { t: '对齐', d: '符合偏好', c: 'var(--accent-2)', nav: 'training' },
            { t: '微调', d: 'LoRA 适配业务', c: 'var(--orange)', nav: 'finetune' },
            { t: '推理', d: 'prefill+KV cache', c: 'var(--pink)', nav: 'inference' },
          ].map((x, i, arr) => (
            <React.Fragment key={i}>
              <div onClick={() => onNav(x.nav)} style={{
                cursor: 'pointer', textAlign: 'center', padding: '10px 14px',
                background: 'var(--bg-elev)', border: `1.5px solid ${x.c}`, borderRadius: 9, minWidth: 92,
              }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: x.c }}>{x.t}</div>
                <div style={{ fontSize: 10.5, color: 'var(--text-dim)', marginTop: 2 }}>{x.d}</div>
              </div>
              {i < arr.length - 1 && <span style={{ color: 'var(--text-faint)' }}>→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <Callout type="key">
        <b>给产品经理的视角：</b>{' '}
        训练阶段关心的是「反向传播 / 梯度更新 / 优化器」——模型怎么从数据里学到参数；推理阶段关心的是「prefill /
        KV cache」——模型部署后怎么用学到的参数高效地一个字一个字往外蹦。这两套流程用的是
        <b>同一套网络结构</b>，但目标和瓶颈完全不同。后面的模块会分别拆开讲。
      </Callout>

      <div className="grid-3">
        <div className="stat">
          <div className="stat-val" style={{ color: 'var(--green)' }}>训练</div>
          <div className="stat-label">前向 → 算 loss → 反向传播 → 更新权重，循环百万次</div>
        </div>
        <div className="stat">
          <div className="stat-val" style={{ color: 'var(--accent)' }}>推理</div>
          <div className="stat-label">权重冻结，只做前向；自回归一个个生成 token</div>
        </div>
        <div className="stat">
          <div className="stat-val" style={{ color: 'var(--accent-2)' }}>注意力</div>
          <div className="stat-label">贯穿训练与推理的核心算子，也是算力 / 显存瓶颈所在</div>
        </div>
      </div>
    </div>
  )
}
