import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Formula, Legend, heatColor, f2, Slider } from '../components/ui.jsx'
import { mulberry32, randn, softmax, dot } from '../lib/mathx.js'

// 真实的 scaled dot-product attention。给一串 token，每个有一个 d 维向量，
// 真实算 Q·Kᵀ/√d → softmax → 加权求和 V，画出注意力矩阵热图。

const DEFAULT_TOKENS = ['猫', '坐在', '垫子', '上', '它', '很', '舒服']
const D = 4 // 向量维度（演示用小维度）

function buildProjections(seed) {
  const rng = mulberry32(seed)
  const mk = () => Array.from({ length: D }, () => Array.from({ length: D }, () => randn(rng) * 0.6))
  return { Wq: mk(), Wk: mk(), Wv: mk() }
}

// 给每个 token 一个确定的 embedding（按字符派生，保证可复现）
function embed(tokens, seed) {
  const rng = mulberry32(seed + 99)
  // 每个唯一 token 一个固定向量
  const map = {}
  return tokens.map((tk) => {
    if (!map[tk]) {
      const base = Array.from({ length: D }, () => randn(rng))
      map[tk] = base
    }
    return map[tk]
  })
}

function project(vec, M) {
  // vec(1×D) · M(D×D)
  return M[0].map((_, j) => vec.reduce((s, v, i) => s + v * M[i][j], 0))
}

export default function Attention() {
  const [tokensStr, setTokensStr] = useState(DEFAULT_TOKENS.join(' '))
  const [seed, setSeed] = useState(5)
  const [temp, setTemp] = useState(1) // 缩放温度，演示 √d 的作用
  const [selected, setSelected] = useState(0)

  const tokens = useMemo(() => tokensStr.trim().split(/\s+/).filter(Boolean).slice(0, 9), [tokensStr])
  const { Wq, Wk, Wv } = useMemo(() => buildProjections(seed), [seed])
  const embeds = useMemo(() => embed(tokens, seed), [tokens, seed])

  const { Q, K, V, scores, attn, out } = useMemo(() => {
    const Q = embeds.map((e) => project(e, Wq))
    const K = embeds.map((e) => project(e, Wk))
    const V = embeds.map((e) => project(e, Wv))
    const scale = Math.sqrt(D) * temp
    const scores = Q.map((q) => K.map((k) => dot(q, k) / scale))
    const attn = scores.map((row) => softmax(row))
    const out = attn.map((w) => {
      const o = new Array(D).fill(0)
      w.forEach((a, j) => V[j].forEach((v, d) => (o[d] += a * v)))
      return o
    })
    return { Q, K, V, scores, attn, out }
  }, [embeds, Wq, Wk, Wv, temp])

  const n = tokens.length
  const cell = Math.min(48, 360 / Math.max(n, 1))

  return (
    <div>
      <PageHeader
        eyebrow="05 · ATTENTION"
        title="注意力机制：让每个词直接「看」到相关的词"
        lead="这是 Transformer 的心脏。核心思想：对每个 token，去衡量它和序列里所有 token 的「相关度」，然后按相关度加权汇总信息。相关度通过 Query 和 Key 的点积算出，再用 softmax 归一成权重。下面是真实计算，改 token 或参数看注意力矩阵怎么变。"
      />

      <Formula>Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V</Formula>

      <Card title="self-attention 实时计算" sub="点矩阵某一行，看那个 token 把注意力分给了谁">
        <div className="controls">
          <div className="control" style={{ minWidth: 320, flex: 1 }}>
            <label><span>输入序列（空格分词，最多 9 个）</span></label>
            <input type="text" value={tokensStr} onChange={(e) => setTokensStr(e.target.value)} />
          </div>
          <Slider label="温度 (放大=√dₖ 作用)" value={temp} min={0.3} max={3} step={0.1} onChange={setTemp} fmt={f2} />
          <div className="control">
            <label><span>投影矩阵</span></label>
            <button className="btn secondary" onClick={() => setSeed((s) => s + 1)}>🎲 重新随机</button>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 36, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {/* 注意力矩阵热图 */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
              注意力权重矩阵 (行=Query 当前词，列=Key 被关注的词)
            </div>
            <svg width={cell * n + 70} height={cell * n + 30}>
              {/* 列标题 */}
              {tokens.map((t, j) => (
                <text key={`c${j}`} x={70 + j * cell + cell / 2} y={20} textAnchor="middle" fontSize="11" fill="var(--text-dim)">{t}</text>
              ))}
              {attn.map((row, i) => (
                <g key={i}>
                  <text x={64} y={30 + i * cell + cell / 2 + 4} textAnchor="end" fontSize="11"
                    fill={i === selected ? 'var(--accent)' : 'var(--text-dim)'} fontWeight={i === selected ? 700 : 400}>{tokens[i]}</text>
                  {row.map((w, j) => (
                    <g key={j}>
                      <rect x={70 + j * cell} y={30 + i * cell} width={cell - 2} height={cell - 2}
                        fill={heatColor(w, 0, 1)} rx={3}
                        stroke={i === selected ? 'var(--accent)' : 'transparent'} strokeWidth={i === selected ? 2 : 0}
                        style={{ cursor: 'pointer' }} onClick={() => setSelected(i)} />
                      <text x={70 + j * cell + cell / 2 - 1} y={30 + i * cell + cell / 2 + 4} textAnchor="middle"
                        fontSize="9.5" fill={w > 0.5 ? '#111' : '#fff'} fontFamily="var(--mono)" pointerEvents="none">{f2(w)}</text>
                    </g>
                  ))}
                </g>
              ))}
            </svg>
            <Legend items={[
              { color: heatColor(0.05), label: '低注意力' },
              { color: heatColor(0.95), label: '高注意力' },
            ]} />
          </div>

          {/* 选中行解读 */}
          <div style={{ flex: 1, minWidth: 240 }}>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
              「<b style={{ color: 'var(--accent)' }}>{tokens[selected]}</b>」的注意力分配
            </div>
            {tokens.map((t, j) => (
              <div key={j} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                <span style={{ width: 46, fontSize: 12, color: 'var(--text-dim)', textAlign: 'right' }}>{t}</span>
                <div style={{ flex: 1, height: 14, background: 'var(--bg-elev)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${(attn[selected]?.[j] || 0) * 100}%`, height: '100%', background: 'var(--accent)' }} />
                </div>
                <span style={{ width: 38, fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-faint)' }}>{f2(attn[selected]?.[j] || 0)}</span>
              </div>
            ))}
            <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 10 }}>
              这些权重加起来 = 1（softmax 保证）。输出向量 = 用这些权重对所有 V 向量加权平均。
            </div>
          </div>
        </div>
      </Card>

      <div className="grid-3">
        <Card title="Q / K / V 是什么">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p><strong style={{ color: 'var(--accent)' }}>Query：</strong>我在找什么样的信息</p>
            <p><strong style={{ color: 'var(--green)' }}>Key：</strong>我能提供什么样的信息</p>
            <p><strong style={{ color: 'var(--orange)' }}>Value：</strong>我实际携带的内容</p>
            <p>每个 token 的向量分别乘三个可学习矩阵 W_q/W_k/W_v 得到。Q 和所有 K 点积 = 匹配度。</p>
          </div>
        </Card>
        <Card title="为什么要除以 √dₖ">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p>
              点积会随维度 d 增大而数值变大，导致 softmax 进入「饱和区」——一个词几乎独占全部注意力，梯度消失。
            </p>
            <p>除以 √dₖ 把方差拉回稳定区间。把上面的<strong>温度调大</strong>模拟去掉缩放：注意力会变得极端尖锐。</p>
          </div>
        </Card>
        <Card title="多头注意力 Multi-Head">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p>
              实际用<strong>多组</strong> Q/K/V 并行做注意力（多个「头」），每个头关注不同的关系模式
              （语法、指代、位置…），最后拼接。这里展示的是单头。
            </p>
            <p>头之间完全独立 → 又是天然可并行。</p>
          </div>
        </Card>
      </div>

      <Callout type="key">
        <b>对比 RNN 的关键飞跃：</b> RNN 里「猫」要影响「它」，信息得逐步沿时间步传递、容易衰减；
        注意力里「它」可以<b>一步直接</b>给「猫」分配高权重，无论隔多远。而且所有 token 的注意力可
        <b>同时并行</b>计算——这就是 Transformer 能吃下海量数据、训练超大模型的根本原因。
      </Callout>
    </div>
  )
}
