import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, Formula, f2, f3 } from '../../components/ui.jsx'
import { linregFit, r2, mse, mean } from '../../lib/mathx.js'

// 真实线性回归：可拖拽散点，实时最小二乘解析解 + 梯度下降对比。
// 看 MSE / R² 随数据和拟合线变化。

const W = 560, H = 380, PAD = 44
// 初始数据：带噪声的线性关系
const INIT = [
  [1, 2.1], [2, 2.8], [3, 4.2], [4, 4.0], [5, 5.6],
  [6, 6.1], [7, 7.4], [8, 7.2], [9, 9.1], [10, 9.6],
]
const XMIN = 0, XMAX = 11, YMIN = 0, YMAX = 11

export default function LinearRegression() {
  const [pts, setPts] = useState(INIT)
  const [drag, setDrag] = useState(null)

  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const { w, b } = useMemo(() => linregFit(xs, ys), [pts])
  const preds = xs.map((x) => w * x + b)
  const stats = useMemo(() => ({ r2: r2(ys, preds), mse: mse(ys, preds) }), [pts, w, b])

  const sx = (x) => PAD + ((x - XMIN) / (XMAX - XMIN)) * (W - 2 * PAD)
  const sy = (y) => H - PAD - ((y - YMIN) / (YMAX - YMIN)) * (H - 2 * PAD)
  const ix = (px) => XMIN + ((px - PAD) / (W - 2 * PAD)) * (XMAX - XMIN)
  const iy = (py) => YMIN + ((H - PAD - py) / (H - 2 * PAD)) * (YMAX - YMIN)

  const onMove = (e) => {
    if (drag === null) return
    const svg = e.currentTarget
    const rect = svg.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const py = ((e.clientY - rect.top) / rect.height) * H
    const nx = Math.max(XMIN, Math.min(XMAX, ix(px)))
    const ny = Math.max(YMIN, Math.min(YMAX, iy(py)))
    setPts((prev) => prev.map((p, i) => (i === drag ? [nx, ny] : p)))
  }

  // 回归线两端点
  const lineX1 = XMIN, lineX2 = XMAX

  return (
    <div>
      <PageHeader
        eyebrow="01 · 线性回归"
        title="线性回归：用一条直线拟合连续关系"
        lead="最古老也最基础的模型：假设输出 y 是输入特征的加权和 ŷ = w·x + b，目标是让所有点到直线的竖直距离平方和最小（最小二乘）。它有闭式解析解，一步算出最优 w、b。拖动任意一个点，看回归线和误差实时变化。"
      />

      <Formula>ŷ = w · x + b　　损失 = Σ(yᵢ − ŷᵢ)²　最小化 → 解析解 w = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²</Formula>

      <Card title="拖动散点，实时最小二乘拟合" sub="蓝线是当前最优回归线；红色竖线是每个点的残差（预测误差）">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8, cursor: drag !== null ? 'grabbing' : 'default', touchAction: 'none' }}
          onMouseMove={onMove}
          onMouseUp={() => setDrag(null)}
          onMouseLeave={() => setDrag(null)}
        >
          {/* 坐标轴 */}
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" />
          <text x={W - PAD} y={H - PAD + 20} textAnchor="end" fontSize="11" fill="var(--text-faint)">x（特征）</text>
          <text x={PAD - 8} y={PAD - 6} textAnchor="start" fontSize="11" fill="var(--text-faint)">y（目标）</text>

          {/* 残差竖线 */}
          {pts.map((p, i) => (
            <line key={`r${i}`} x1={sx(p[0])} y1={sy(p[1])} x2={sx(p[0])} y2={sy(preds[i])}
              stroke="var(--red)" strokeWidth={1.5} strokeDasharray="3 2" opacity={0.7} />
          ))}

          {/* 回归线 */}
          <line x1={sx(lineX1)} y1={sy(w * lineX1 + b)} x2={sx(lineX2)} y2={sy(w * lineX2 + b)}
            stroke="var(--accent)" strokeWidth={2.5} />

          {/* 散点 */}
          {pts.map((p, i) => (
            <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={7}
              fill="var(--cyan)" stroke="var(--bg)" strokeWidth={2}
              style={{ cursor: 'grab' }}
              onMouseDown={() => setDrag(i)} />
          ))}
        </svg>

        <Legend items={[
          { color: 'var(--cyan)', label: '数据点（可拖拽）' },
          { color: 'var(--accent)', label: '回归线 ŷ=w·x+b' },
          { color: 'var(--red)', label: '残差（预测误差）' },
        ]} />
      </Card>

      <div className="grid-3">
        <Card title="斜率 w">
          <div className="formula" style={{ textAlign: 'center', fontSize: 22, color: 'var(--accent)' }}>{f3(w)}</div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>x 每增 1，y 平均变化 {f3(w)}</div>
        </Card>
        <Card title="截距 b">
          <div className="formula" style={{ textAlign: 'center', fontSize: 22, color: 'var(--accent-2)' }}>{f3(b)}</div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>x=0 时的预测值</div>
        </Card>
        <Card title="拟合优度">
          <div className="formula" style={{ textAlign: 'center', fontSize: 22, color: stats.r2 > 0.8 ? 'var(--green)' : 'var(--orange)' }}>R² = {f3(stats.r2)}</div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>MSE = {f3(stats.mse)}（越小越好）</div>
        </Card>
      </div>

      <div className="grid-2">
        <Card title="R² 到底是什么？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>R²（决定系数）= 1 − 残差平方和 / 总平方和，衡量「模型比直接拿平均值预测好多少」。</p>
            <p><strong>R²=1</strong>：完美拟合；<strong>R²=0</strong>：还不如直接用 ȳ；<strong>R²&lt;0</strong>：比平均值还差。</p>
            <p>试试把一个点拖到远离直线的位置——残差变大，R² 立刻下降。</p>
          </div>
        </Card>
        <Card title="对应的框架 API">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>scikit-learn：</strong></p>
            <p><code>from sklearn.linear_model import LinearRegression</code></p>
            <p><code>model = LinearRegression().fit(X, y)</code> → <code>model.coef_</code> 就是这里的 w，<code>model.intercept_</code> 是 b。</p>
            <p><strong>statsmodels</strong> 的 OLS 还会额外给出 p 值、置信区间，适合做统计推断。</p>
            <p>加正则化防过拟合：<strong>Ridge</strong>（L2）/ <strong>Lasso</strong>（L1，能把无用特征系数压到 0）。</p>
          </div>
        </Card>
      </div>

      <Callout type="warn">
        <b>注意：</b> 线性回归假设关系是线性的。如果真实关系是曲线（如二次、指数），单纯的线性回归会系统性偏差——这时要么做特征变换（加 x² 项），要么换树模型/非线性模型。
      </Callout>
    </div>
  )
}
