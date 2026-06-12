import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Slider, Formula, f2 } from '../components/ui.jsx'

// RNN 时序展开 + LSTM 门控。用真实的标量递推演示隐状态如何沿时间步传递，
// 以及为什么 RNN 会梯度消失、LSTM 的门控如何缓解。

const sigmoid = (x) => 1 / (1 + Math.exp(-x))

export default function RNNLSTM() {
  const [seq, setSeq] = useState([0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1])
  const [wh, setWh] = useState(0.6) // RNN 隐到隐权重
  const [wx, setWx] = useState(0.8)
  const [mode, setMode] = useState('rnn')

  // RNN 前向：h_t = tanh(wh*h_{t-1} + wx*x_t)
  const rnnStates = useMemo(() => {
    let h = 0
    const out = [{ h: 0, contrib: [] }]
    seq.forEach((x) => {
      h = Math.tanh(wh * h + wx * x)
      out.push({ h })
    })
    return out
  }, [seq, wh, wx])

  // LSTM 前向（简化标量版，固定一组门权重，演示门控行为）
  const lstmStates = useMemo(() => {
    let h = 0, c = 0
    const out = [{ h: 0, c: 0, f: 0, i: 0, o: 0 }]
    seq.forEach((x) => {
      const f = sigmoid(1.2 * x + 0.8 * h + 0.5) // 遗忘门
      const ig = sigmoid(1.0 * x + 0.6 * h - 0.2) // 输入门
      const o = sigmoid(0.9 * x + 0.7 * h + 0.3) // 输出门
      const cTilde = Math.tanh(1.1 * x + 0.5 * h) // 候选记忆
      c = f * c + ig * cTilde
      h = o * Math.tanh(c)
      out.push({ h, c, f, i: ig, o, cTilde })
    })
    return out
  }, [seq])

  // 梯度消失演示：RNN 中早期输入对最终隐状态的影响 ≈ wh^t * (1-h^2)连乘
  const gradFlow = useMemo(() => {
    const out = []
    let g = 1
    for (let t = seq.length; t >= 1; t--) {
      const h = rnnStates[t].h
      g *= wh * (1 - h * h) // d tanh
      out.unshift(Math.abs(g))
    }
    return out
  }, [rnnStates, wh, seq])

  const states = mode === 'rnn' ? rnnStates : lstmStates
  const W = 720, H = 200
  const stepX = W / (seq.length + 1)
  const hY = (h) => H / 2 - h * 60

  const setSeqVal = (i, v) => {
    const s = [...seq]; s[i] = v; setSeq(s)
  }

  return (
    <div>
      <PageHeader
        eyebrow="04 · RNN / LSTM"
        title="时序网络：让信息沿时间步流动"
        lead="处理序列（文字、语音、股价）需要「记忆」。RNN 引入一个隐状态 h，每个时间步把当前输入和上一步的记忆揉在一起，再传给下一步。但朴素 RNN 记不住远处的信息——梯度会指数衰减。LSTM 用三道门控精细管理记忆，是 Transformer 之前序列建模的王者。"
      />

      <div className="btn-row" style={{ marginBottom: 16 }}>
        <button className={`btn ${mode === 'rnn' ? '' : 'secondary'}`} onClick={() => setMode('rnn')}>朴素 RNN</button>
        <button className={`btn ${mode === 'lstm' ? '' : 'secondary'}`} onClick={() => setMode('lstm')}>LSTM 门控</button>
      </div>

      <Card
        title={mode === 'rnn' ? 'RNN 沿时间展开' : 'LSTM 沿时间展开'}
        sub={mode === 'rnn'
          ? 'h_t = tanh(W_h · h_{t-1} + W_x · x_t)，每个圆是一个时间步的隐状态'
          : '记忆细胞 c_t 像一条传送带，三道门决定写入/遗忘/输出多少'}
      >
        {mode === 'rnn' && (
          <Formula>hₜ = tanh( W_h · hₜ₋₁ + W_x · xₜ )</Formula>
        )}
        {mode === 'lstm' && (
          <Formula>fₜ=σ(...)　iₜ=σ(...)　oₜ=σ(...)　cₜ = fₜ·cₜ₋₁ + iₜ·c̃ₜ　hₜ = oₜ·tanh(cₜ)</Formula>
        )}

        {mode === 'rnn' && (
          <div className="controls">
            <Slider label="W_h (记忆权重)" value={wh} min={-1.2} max={1.2} step={0.05} onChange={setWh} fmt={f2} />
            <Slider label="W_x (输入权重)" value={wx} min={0} max={1.5} step={0.05} onChange={setWx} fmt={f2} />
          </div>
        )}

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
          <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="var(--border)" strokeDasharray="3 3" />
          {seq.map((x, t) => {
            const cx = stepX * (t + 1)
            const st = states[t + 1]
            const prevX = stepX * t
            return (
              <g key={t}>
                {/* 隐状态传递箭头 */}
                {t > 0 && (
                  <line x1={prevX + 22} y1={hY(states[t].h)} x2={cx - 22} y2={hY(st.h)}
                    stroke="var(--accent-2)" strokeWidth={2} markerEnd="url(#arrow)" />
                )}
                {/* 输入 */}
                <line x1={cx} y1={H - 14} x2={cx} y2={hY(st.h) + 22} stroke="var(--green)" strokeWidth={1.5} />
                <text x={cx} y={H - 2} textAnchor="middle" fontSize="11" fill="var(--green)" fontFamily="var(--mono)">x={f2(x)}</text>
                {/* 隐状态圆 */}
                <circle cx={cx} cy={hY(st.h)} r={20}
                  fill={st.h >= 0 ? `rgba(124,92,255,${0.25 + Math.abs(st.h) * 0.6})` : `rgba(255,92,122,${0.25 + Math.abs(st.h) * 0.6})`}
                  stroke="var(--border)" strokeWidth={1.5} />
                <text x={cx} y={hY(st.h) + 4} textAnchor="middle" fontSize="10" fill="var(--text)" fontFamily="var(--mono)">{f2(st.h)}</text>
                <text x={cx} y={hY(st.h) - 28} textAnchor="middle" fontSize="9" fill="var(--text-faint)">t={t + 1}</text>
                {/* LSTM 门控指示 */}
                {mode === 'lstm' && (
                  <g>
                    <rect x={cx - 18} y={hY(st.h) + 26} width={36} height={6} rx={3} fill="var(--border)" />
                    <rect x={cx - 18} y={hY(st.h) + 26} width={36 * st.f} height={6} rx={3} fill="var(--orange)" />
                  </g>
                )}
              </g>
            )
          })}
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent-2)" />
            </marker>
          </defs>
        </svg>
        {mode === 'lstm' && (
          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 6 }}>
            橙色条 = 遗忘门开度 fₜ（满格=完全保留上一刻记忆，空=完全遗忘）
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>编辑输入序列：</div>
          <div className="controls" style={{ gap: 10 }}>
            {seq.map((x, i) => (
              <Slider key={i} label={`x${i + 1}`} value={x} min={-1} max={1} step={0.1} onChange={(v) => setSeqVal(i, v)} fmt={f2} />
            ))}
          </div>
        </div>
      </Card>

      {mode === 'rnn' && (
        <Card title="梯度消失 / 爆炸演示" sub="第 t 步输入对最终隐状态的影响强度（反向传播时梯度连乘的结果）">
          <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', height: 100 }}>
            {gradFlow.map((g, i) => {
              const mag = Math.min(1, g / (Math.max(...gradFlow) || 1))
              return (
                <div key={i} style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: 80 }}>
                    <div style={{ height: `${mag * 100}%`, background: g < 0.05 ? 'var(--red)' : 'var(--accent)', borderRadius: '3px 3px 0 0', minHeight: 2 }} />
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>t{i + 1}</div>
                </div>
              )
            })}
          </div>
          <Callout type="warn">
            把 <b>W_h 调小</b>（比如 0.3）：早期时间步的梯度（左侧柱子）几乎归零——这就是<b>梯度消失</b>，
            模型学不到长距离依赖。把 <b>W_h 调大</b>（&gt;1）：梯度可能爆炸。朴素 RNN 卡在这个两难里，
            而 LSTM 的记忆细胞 c 提供了一条「梯度高速公路」缓解了它。
          </Callout>
        </Card>
      )}

      <div className="grid-2">
        <Card title="LSTM 的三道门">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong style={{ color: 'var(--orange)' }}>遗忘门 f：</strong> 决定上一刻的记忆 c 保留多少（0=全忘，1=全留）。</p>
            <p><strong style={{ color: 'var(--green)' }}>输入门 i：</strong> 决定当前新信息写入记忆多少。</p>
            <p><strong style={{ color: 'var(--accent)' }}>输出门 o：</strong> 决定记忆中有多少暴露为当前输出 h。</p>
            <p>记忆细胞 c 的更新是<strong>加法</strong>为主（c = f·c + i·c̃），不像 RNN 全是连乘，所以梯度能沿 c 较稳定地往回传。</p>
          </div>
        </Card>
        <Card title="RNN/LSTM 的根本瓶颈">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>
              无论 RNN 还是 LSTM，都必须<strong>严格按时间步串行</strong>计算——算第 t 步必须先算完 t-1 步。
              这意味着无法利用 GPU 的大规模并行，训练长序列极慢。
            </p>
            <p>
              2017 年 Transformer 的洞察是：<strong>抛弃循环</strong>，让序列里每个位置通过「注意力」
              一步直接看到所有其他位置，既解决长依赖，又能完全并行。这就是下一模块的核心 →
            </p>
          </div>
        </Card>
      </div>
    </div>
  )
}
