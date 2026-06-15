import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, Formula, f2 } from '../../components/ui.jsx'
import { sigmoid } from '../../lib/mathx.js'

// 逻辑回归：2D 两类点，sigmoid 把线性得分压成概率，决策边界随权重移动。
// 调权重看决策面如何旋转/平移。

const W = 480, H = 380, PAD = 40
const RANGE = 6 // 坐标范围 [-3, 3] 左右

// 两类样本（蓝=0，橙=1），大致线性可分
const CLASS0 = [[-2, -1.5], [-2.4, 0.2], [-1.6, -2.2], [-1, -0.8], [-2.8, -1], [-0.6, -2], [-1.8, 0.8], [-2.2, -2.4]]
const CLASS1 = [[1.6, 1.2], [2.2, 0.4], [1, 2], [2.6, 1.8], [1.4, -0.4], [2, 2.6], [0.8, 0.9], [2.8, -0.2]]

export default function LogisticRegression() {
  const [w1, setW1] = useState(1.2)
  const [w2, setW2] = useState(0.9)
  const [b, setB] = useState(-0.3)

  const all = useMemo(() => [
    ...CLASS0.map((p) => ({ x: p[0], y: p[1], label: 0 })),
    ...CLASS1.map((p) => ({ x: p[0], y: p[1], label: 1 })),
  ], [])

  // 当前模型的预测与准确率
  const stats = useMemo(() => {
    let correct = 0, lossSum = 0
    all.forEach((p) => {
      const z = w1 * p.x + w2 * p.y + b
      const prob = sigmoid(z)
      const pred = prob >= 0.5 ? 1 : 0
      if (pred === p.label) correct++
      const eps = 1e-9
      lossSum += -(p.label * Math.log(prob + eps) + (1 - p.label) * Math.log(1 - prob + eps))
    })
    return { acc: correct / all.length, loss: lossSum / all.length }
  }, [all, w1, w2, b])

  const sx = (x) => PAD + ((x + RANGE / 2) / RANGE) * (W - 2 * PAD)
  const sy = (y) => H - PAD - ((y + RANGE / 2) / RANGE) * (H - 2 * PAD)

  // 决策边界 w1·x + w2·y + b = 0  → y = -(w1·x + b)/w2
  const boundary = useMemo(() => {
    if (Math.abs(w2) < 1e-6) return null
    const x1 = -RANGE / 2, x2 = RANGE / 2
    return [
      [x1, -(w1 * x1 + b) / w2],
      [x2, -(w1 * x2 + b) / w2],
    ]
  }, [w1, w2, b])

  // 概率热力背景网格
  const grid = useMemo(() => {
    const cells = []
    const G = 24
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const x = -RANGE / 2 + (i + 0.5) * (RANGE / G)
        const y = -RANGE / 2 + (j + 0.5) * (RANGE / G)
        cells.push({ x, y, prob: sigmoid(w1 * x + w2 * y + b) })
      }
    return { cells, G }
  }, [w1, w2, b])

  const cellW = (W - 2 * PAD) / grid.G
  const cellH = (H - 2 * PAD) / grid.G

  return (
    <div>
      <PageHeader
        eyebrow="02 · 逻辑回归"
        title="逻辑回归：把线性得分压成概率来分类"
        lead="名字带「回归」，干的却是分类的活。它先算一个线性得分 z = w·x + b，再用 sigmoid 把 z 压到 0~1 之间当作「属于正类的概率」。z=0 的那条线就是决策边界。调三个权重，看边界怎么旋转平移、准确率怎么变。"
      />

      <Formula>z = w₁·x₁ + w₂·x₂ + b　　P(y=1) = σ(z) = 1/(1+e⁻ᶻ)　　σ(z)≥0.5 判为正类</Formula>

      <Card title="决策边界与概率场" sub="背景颜色 = 模型认为该位置属于橙类的概率；白线是 50% 决策边界">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          {/* 概率热力背景 */}
          {grid.cells.map((c, i) => {
            // prob 0→蓝(class0)，1→橙(class1)
            const r = Math.round(91 + c.prob * (255 - 91))
            const g = Math.round(157 + c.prob * (159 - 157))
            const bl = Math.round(255 - c.prob * (255 - 67))
            return (
              <rect key={i} x={sx(c.x) - cellW / 2} y={sy(c.y) - cellH / 2} width={cellW + 1} height={cellH + 1}
                fill={`rgb(${r},${g},${bl})`} opacity={0.25} />
            )
          })}

          {/* 坐标轴 */}
          <line x1={sx(-RANGE / 2)} y1={sy(0)} x2={sx(RANGE / 2)} y2={sy(0)} stroke="var(--border)" strokeDasharray="3 3" />
          <line x1={sx(0)} y1={sy(-RANGE / 2)} x2={sx(0)} y2={sy(RANGE / 2)} stroke="var(--border)" strokeDasharray="3 3" />

          {/* 决策边界 */}
          {boundary && (
            <line x1={sx(boundary[0][0])} y1={sy(boundary[0][1])} x2={sx(boundary[1][0])} y2={sy(boundary[1][1])}
              stroke="var(--text)" strokeWidth={2.5} />
          )}

          {/* 样本点 */}
          {all.map((p, i) => {
            const prob = sigmoid(w1 * p.x + w2 * p.y + b)
            const pred = prob >= 0.5 ? 1 : 0
            const wrong = pred !== p.label
            return (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={7}
                fill={p.label === 1 ? 'var(--orange)' : 'var(--accent)'}
                stroke={wrong ? 'var(--red)' : 'var(--bg)'} strokeWidth={wrong ? 3 : 2} />
            )
          })}
        </svg>

        <Legend items={[
          { color: 'var(--accent)', label: '类别 0' },
          { color: 'var(--orange)', label: '类别 1' },
          { color: 'var(--red)', label: '红圈=分错' },
        ]} />

        <div className="controls" style={{ marginTop: 14 }}>
          <Slider label="权重 w₁ (x 方向)" value={w1} min={-3} max={3} step={0.1} onChange={setW1} fmt={f2} />
          <Slider label="权重 w₂ (y 方向)" value={w2} min={-3} max={3} step={0.1} onChange={setW2} fmt={f2} />
          <Slider label="偏置 b" value={b} min={-3} max={3} step={0.1} onChange={setB} fmt={f2} />
        </div>
      </Card>

      <div className="grid-2">
        <Card title="当前模型表现">
          <div style={{ display: 'flex', gap: 24, justifyContent: 'center', padding: '8px 0' }}>
            <div style={{ textAlign: 'center' }}>
              <div className="formula" style={{ fontSize: 26, color: stats.acc === 1 ? 'var(--green)' : 'var(--orange)' }}>{Math.round(stats.acc * 100)}%</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>准确率</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div className="formula" style={{ fontSize: 26, color: 'var(--accent)' }}>{f2(stats.loss)}</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>交叉熵损失</div>
            </div>
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', textAlign: 'center', marginTop: 4 }}>
            试着调权重让准确率到 100%——真实训练就是用梯度下降自动找这组最优权重。
          </div>
        </Card>
        <Card title="为什么用 sigmoid 和交叉熵？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>sigmoid</strong> 把任意实数压到 (0,1)，正好当概率用；z 越大越接近 1，z 越小越接近 0。</p>
            <p><strong>交叉熵损失</strong>比平方误差更适合分类：预测越自信又越错，惩罚越重，梯度也更健康，收敛更快。</p>
            <p>多分类时把 sigmoid 换成 <strong>softmax</strong>，就是 <code>LogisticRegression(multi_class='multinomial')</code>。</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>框架：</b> <code>from sklearn.linear_model import LogisticRegression</code>；<code>LogisticRegression(C=1.0)</code> 的 <code>C</code> 是正则化强度的倒数，越小正则越强、边界越平滑。逻辑回归因为<strong>可解释（每个特征的权重正负即影响方向）</strong>，至今是风控评分卡的核心模型——见 10 号模块。
      </Callout>
    </div>
  )
}
