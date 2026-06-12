import React, { useState } from 'react'
import { PageHeader, Card, Callout, Formula } from '../components/ui.jsx'

// Transformer 整体架构图。可点击每个组件查看它在做什么。
// 同时讲清 encoder-decoder 经典结构 与 decoder-only（GPT）的区别。

const COMPONENTS = {
  embed: {
    title: '词嵌入 Token Embedding',
    color: 'var(--cyan)',
    desc: '把离散的 token（词/子词）映射成稠密向量。相近语义的词向量在空间上也相近。这是模型理解文字的第一步——文字变数字。',
    detail: '词表里每个 token 对应一行可学习的向量，查表即可。维度通常 512~12288。',
  },
  pos: {
    title: '位置编码 Positional Encoding',
    color: 'var(--yellow)',
    desc: '注意力本身不区分顺序（打乱输入结果一样）。位置编码把「第几个位置」的信息加进向量，让模型知道词序。',
    detail: '经典用 sin/cos 不同频率的波形；现代模型多用 RoPE 旋转位置编码，对长度外推更友好。',
  },
  mha: {
    title: '多头自注意力 Multi-Head Self-Attention',
    color: 'var(--accent)',
    desc: '本层的核心。每个 token 通过 Q/K/V 机制关注序列里其他所有 token，汇聚相关信息。多个头并行捕捉不同关系。',
    detail: '见「注意力机制」模块。这是 Transformer 唯一做 token 间信息交换的地方。',
  },
  addnorm: {
    title: '残差连接 + LayerNorm',
    color: 'var(--green)',
    desc: '残差(x + Sublayer(x))让梯度能直接穿过深层网络、缓解退化；LayerNorm 把每个向量归一化、稳定训练。',
    detail: '公式：LayerNorm(x + Sublayer(x))。深层 Transformer 能堆几十上百层，全靠这个结构。',
  },
  ffn: {
    title: '前馈网络 Feed-Forward (FFN)',
    color: 'var(--orange)',
    desc: '对每个位置独立地过一个两层 MLP（先升维再降维，中间非线性）。负责对注意力汇聚来的信息做非线性变换/加工。',
    detail: '通常中间层是输入的 4 倍宽。大模型里 FFN 占了绝大部分参数量。',
  },
  out: {
    title: '输出层 Linear + Softmax',
    color: 'var(--pink)',
    desc: '把最后一层的向量投影回词表大小的维度，softmax 得到下一个 token 的概率分布。',
    detail: '推理时从这个分布里采样（贪心/top-k/温度采样）得到下一个词。',
  },
}

function Block({ id, label, color, onClick, active, sub }) {
  return (
    <div
      onClick={() => onClick(id)}
      style={{
        background: active ? color : 'var(--bg-elev)',
        color: active ? '#0b0e14' : 'var(--text)',
        border: `1.5px solid ${color}`,
        borderRadius: 8,
        padding: '10px 14px',
        cursor: 'pointer',
        textAlign: 'center',
        fontSize: 13,
        fontWeight: 600,
        transition: 'all 0.15s',
        position: 'relative',
      }}
    >
      {label}
      {sub && <div style={{ fontSize: 10, fontWeight: 400, opacity: 0.8, marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const Arrow = () => (
  <div style={{ textAlign: 'center', color: 'var(--text-faint)', fontSize: 16, lineHeight: 1 }}>↓</div>
)

export default function TransformerArch() {
  const [sel, setSel] = useState('mha')
  const [arch, setArch] = useState('decoder') // decoder-only(GPT) vs full
  const c = COMPONENTS[sel]
  const Nx = 'N×'

  return (
    <div>
      <PageHeader
        eyebrow="06 · TRANSFORMER"
        title="整体架构：把所有零件组装起来"
        lead="前面的注意力、FFN、残差都是零件。Transformer 把它们按固定模式堆成一个「块(Block)」，再把 N 个块串起来。点击图里任意组件看它的职责。可切换经典 Encoder-Decoder 与 GPT 式 Decoder-Only 两种结构。"
      />

      <div className="btn-row" style={{ marginBottom: 16 }}>
        <button className={`btn ${arch === 'decoder' ? '' : 'secondary'}`} onClick={() => setArch('decoder')}>
          Decoder-Only（GPT 类，主流大模型）
        </button>
        <button className={`btn ${arch === 'full' ? '' : 'secondary'}`} onClick={() => setArch('full')}>
          Encoder-Decoder（原版，翻译类）
        </button>
      </div>

      <div className="grid-2">
        <Card title={arch === 'decoder' ? 'Decoder-Only 数据流' : 'Encoder-Decoder 数据流'}
          sub="自下而上看：token 进，概率分布出">
          {arch === 'decoder' ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <Block id="out" label="Linear + Softmax → 下一个词概率" color={COMPONENTS.out.color} onClick={setSel} active={sel === 'out'} />
              <Arrow />
              <div style={{ border: '1px dashed var(--border)', borderRadius: 10, padding: 12 }}>
                <div style={{ fontSize: 10, color: 'var(--text-faint)', marginBottom: 8, textAlign: 'center' }}>Transformer Block（重复 {Nx}）</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <Block id="addnorm" label="Add & LayerNorm" color={COMPONENTS.addnorm.color} onClick={setSel} active={sel === 'addnorm'} />
                  <Arrow />
                  <Block id="ffn" label="Feed-Forward (FFN)" color={COMPONENTS.ffn.color} onClick={setSel} active={sel === 'ffn'} />
                  <Arrow />
                  <Block id="addnorm" label="Add & LayerNorm" color={COMPONENTS.addnorm.color} onClick={setSel} active={sel === 'addnorm'} />
                  <Arrow />
                  <Block id="mha" label="带因果掩码的多头自注意力" color={COMPONENTS.mha.color} onClick={setSel} active={sel === 'mha'} sub="只能看到自己和左边的词" />
                </div>
              </div>
              <Arrow />
              <Block id="pos" label="+ 位置编码" color={COMPONENTS.pos.color} onClick={setSel} active={sel === 'pos'} />
              <Arrow />
              <Block id="embed" label="词嵌入 Embedding" color={COMPONENTS.embed.color} onClick={setSel} active={sel === 'embed'} />
              <Arrow />
              <div style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>输入 tokens</div>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: 12 }}>
              {/* Encoder */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ fontSize: 11, color: 'var(--accent)', textAlign: 'center', fontWeight: 600 }}>Encoder {Nx}</div>
                <Block id="addnorm" label="Add & Norm" color={COMPONENTS.addnorm.color} onClick={setSel} active={sel === 'addnorm'} />
                <Block id="ffn" label="FFN" color={COMPONENTS.ffn.color} onClick={setSel} active={sel === 'ffn'} />
                <Block id="mha" label="自注意力" color={COMPONENTS.mha.color} onClick={setSel} active={sel === 'mha'} sub="可看全文" />
                <Arrow />
                <Block id="pos" label="+位置编码" color={COMPONENTS.pos.color} onClick={setSel} active={sel === 'pos'} />
                <Block id="embed" label="Embedding" color={COMPONENTS.embed.color} onClick={setSel} active={sel === 'embed'} />
                <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-faint)' }}>源句(如英文)</div>
              </div>
              {/* Decoder */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ fontSize: 11, color: 'var(--pink)', textAlign: 'center', fontWeight: 600 }}>Decoder {Nx}</div>
                <Block id="out" label="Linear+Softmax" color={COMPONENTS.out.color} onClick={setSel} active={sel === 'out'} />
                <Block id="ffn" label="FFN" color={COMPONENTS.ffn.color} onClick={setSel} active={sel === 'ffn'} />
                <Block id="mha" label="交叉注意力" color={COMPONENTS.mha.color} onClick={setSel} active={sel === 'mha'} sub="Q来自译文,K/V来自Encoder" />
                <Block id="mha" label="掩码自注意力" color={COMPONENTS.mha.color} onClick={setSel} active={sel === 'mha'} />
                <Arrow />
                <Block id="pos" label="+位置编码" color={COMPONENTS.pos.color} onClick={setSel} active={sel === 'pos'} />
                <Block id="embed" label="Embedding" color={COMPONENTS.embed.color} onClick={setSel} active={sel === 'embed'} />
                <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-faint)' }}>已生成译文</div>
              </div>
            </div>
          )}
        </Card>

        {/* 组件详情 */}
        <div>
          <Card title={c.title} sub="点左侧图里的组件切换">
            <div style={{ height: 4, background: c.color, borderRadius: 2, marginBottom: 14 }} />
            <div className="prose">
              <p>{c.desc}</p>
            </div>
            <Callout>
              <b>细节：</b> {c.detail}
            </Callout>
          </Card>

          {sel === 'addnorm' && <Formula>output = LayerNorm( x + Sublayer(x) )</Formula>}
          {sel === 'ffn' && <Formula>FFN(x) = W₂ · ReLU( W₁·x + b₁ ) + b₂</Formula>}
          {sel === 'pos' && <Formula>PE(pos, 2i) = sin( pos / 10000^(2i/d) )</Formula>}
          {sel === 'out' && <Formula>P(下一个词) = softmax( W_out · h_last )</Formula>}
        </div>
      </div>

      <div className="grid-2">
        <Card title="因果掩码 Causal Mask（生成的关键）">
          <div className="prose">
            <p>
              GPT 在预测第 t 个词时，<strong>不能偷看</strong>第 t 个及之后的词（否则就是抄答案）。
              因果掩码把注意力矩阵的「上三角」置为 -∞，softmax 后变 0，确保每个位置只能关注它左边的词。
            </p>
            <CausalMaskViz />
          </div>
        </Card>
        <Card title="两种结构怎么选">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>Encoder-Decoder：</strong>原版 Transformer，适合「输入→输出」明确的任务，如翻译、摘要。Encoder 双向理解全文，Decoder 生成。</p>
            <p><strong>Decoder-Only (GPT)：</strong>只保留带掩码的 decoder，统一用「预测下一个词」建模一切任务。结构简单、易 scale，是当今大模型主流（GPT、LLaMA、Claude…）。</p>
            <p><strong>Encoder-Only (BERT)：</strong>只要 encoder，做理解类任务（分类、检索），不擅长生成。</p>
          </div>
        </Card>
      </div>

      <Callout type="key">
        <b>承上启下：</b> 到这里你已经理解了 Transformer 的<b>结构</b>。接下来「大模型生命周期」分区会把这个结构真正用起来：
        文本怎么<b>分词</b>进模型、怎么<b>训练</b>(预训练→SFT→对齐)、怎么低成本<b>微调</b>(LoRA)、
        最后怎么高效<b>推理</b>(Prefill 与 KV Cache)。走完那一圈，整条训推链路就闭环了。
      </Callout>
    </div>
  )
}

function CausalMaskViz() {
  const n = 5
  const words = ['我', '爱', '吃', '苹', '果']
  const cell = 30
  return (
    <svg width={cell * n + 50} height={cell * n + 30} style={{ marginTop: 8 }}>
      {words.map((w, j) => (
        <text key={`c${j}`} x={50 + j * cell + cell / 2} y={16} textAnchor="middle" fontSize="10" fill="var(--text-faint)">{w}</text>
      ))}
      {words.map((w, i) => (
        <g key={i}>
          <text x={44} y={26 + i * cell + cell / 2} textAnchor="end" fontSize="10" fill="var(--text-faint)">{w}</text>
          {words.map((_, j) => (
            <rect key={j} x={50 + j * cell} y={26 + i * cell} width={cell - 2} height={cell - 2} rx={3}
              fill={j <= i ? 'var(--accent)' : 'var(--bg-elev)'}
              opacity={j <= i ? 0.7 : 1}
              stroke="var(--border)" strokeWidth={0.5} />
          ))}
        </g>
      ))}
      <text x={50} y={cell * n + 26} fontSize="10" fill="var(--text-faint)">蓝=可关注，暗=被掩盖</text>
    </svg>
  )
}
