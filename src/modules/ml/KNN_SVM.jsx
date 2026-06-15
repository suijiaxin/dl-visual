import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { dist, majority } from '../../lib/mathx.js'

// KNN：调 k 看决策边界与某点的最近邻投票。
// SVM：线性可分两类的最大间隔超平面 + 支持向量 + 间隔带。

const W = 440, H = 380, PAD = 34
const RANGE = 10

// 两类点（线性大致可分，留点重叠看 SVM 间隔）
const PTS = [
  ...[[2, 3], [2.5, 4.5], [1.5, 2], [3, 3.5], [2.2, 5.5], [1, 4], [3.5, 5], [2.8, 2.5], [4, 4]].map((p) => ({ x: p[0], y: p[1], label: 0 })),
  ...[[7, 6], [7.5, 7.5], [8, 6.5], [6.5, 8], [8.5, 7], [7.2, 5.5], [6, 7], [8, 8.5], [6.8, 6.2]].map((p) => ({ x: p[0], y: p[1], label: 1 })),
]

// 简单线性 SVM（硬间隔近似）：用感知机式更新找一个大间隔分隔线（教学用，非严格 QP）
function trainSVM() {
  // 用解析近似：取两类质心连线的垂直平分面作为初始，再微调到最大间隔
  // 这里直接用一个固定的、对该数据接近最优间隔的 w,b（演示用），并算支持向量
  let w = [1, 1], b = -10
  // 梯度下降软间隔 SVM（hinge loss + L2），真实迭代
  const lr = 0.01, C = 1, epochs = 2000
  w = [0.1, 0.1]; b = 0
  for (let e = 0; e < epochs; e++) {
    PTS.forEach((p) => {
      const yi = p.label === 1 ? 1 : -1
      const margin = yi * (w[0] * p.x + w[1] * p.y + b)
      if (margin < 1) {
        w[0] += lr * (C * yi * p.x - w[0] / PTS.length)
        w[1] += lr * (C * yi * p.y - w[1] / PTS.length)
        b += lr * (C * yi)
      } else {
        w[0] -= lr * (w[0] / PTS.length)
        w[1] -= lr * (w[1] / PTS.length)
      }
    })
  }
  return { w, b }
}
const SVM = trainSVM()

export default function KNN_SVM() {
  const [mode, setMode] = useState('knn')
  const [k, setK] = useState(5)
  const [query, setQuery] = useState({ x: 5, y: 5 })

  const sx = (x) => PAD + (x / RANGE) * (W - 2 * PAD)
  const sy = (y) => H - PAD - (y / RANGE) * (H - 2 * PAD)
  const ix = (px) => ((px - PAD) / (W - 2 * PAD)) * RANGE
  const iy = (py) => ((H - PAD - py) / (H - 2 * PAD)) * RANGE

  // KNN：query 点的最近 k 个邻居
  const neighbors = useMemo(() => {
    return PTS.map((p, i) => ({ i, d: dist([p.x, p.y], [query.x, query.y]), label: p.label }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k)
  }, [query, k])
  const knnPred = useMemo(() => majority(neighbors.map((n) => n.label)), [neighbors])

  // KNN 决策背景
  const knnGrid = useMemo(() => {
    if (mode !== 'knn') return { cells: [], G: 1 }
    const cells = []
    const G = 28
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const x = (i + 0.5) * (RANGE / G)
        const y = (j + 0.5) * (RANGE / G)
        const nb = PTS.map((p) => ({ d: dist([p.x, p.y], [x, y]), label: p.label }))
          .sort((a, b) => a.d - b.d).slice(0, k)
        cells.push({ x, y, pred: Number(majority(nb.map((n) => n.label))) })
      }
    return { cells, G }
  }, [mode, k])
  const cw = (W - 2 * PAD) / knnGrid.G
  const ch = (H - 2 * PAD) / knnGrid.G

  // SVM 决策线与间隔带： w·x + b = 0, ±1
  const svmLine = (offset) => {
    // w0*x + w1*y + b = offset → y = (offset - b - w0*x)/w1
    const [w0, w1] = SVM.w
    const x1 = 0, x2 = RANGE
    return [[x1, (offset - SVM.b - w0 * x1) / w1], [x2, (offset - SVM.b - w0 * x2) / w1]]
  }
  const margin = 1 / Math.hypot(SVM.w[0], SVM.w[1])
  // 支持向量：margin ≈ 1 的点
  const isSupport = (p) => {
    const yi = p.label === 1 ? 1 : -1
    return yi * (SVM.w[0] * p.x + SVM.w[1] * p.y + SVM.b) < 1.15
  }

  const onClick = (e) => {
    if (mode !== 'knn') return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const py = ((e.clientY - rect.top) / rect.height) * H
    setQuery({ x: Math.max(0, Math.min(RANGE, ix(px))), y: Math.max(0, Math.min(RANGE, iy(py))) })
  }

  return (
    <div>
      <PageHeader
        eyebrow="06 · KNN 与 SVM"
        title="KNN 与 SVM：基于距离和间隔的分类"
        lead="两个几何味很浓的经典算法。KNN「近朱者赤」：看一个点最近的 k 个邻居是哪类，就跟着投票，连训练都不需要。SVM 则去找一条「离两边都尽量远」的最大间隔分界线，只有边界上的少数支持向量起作用。切换下面两个标签分别体验。"
      />

      <div className="btn-row" style={{ marginBottom: 16 }}>
        <button className={`btn ${mode === 'knn' ? '' : 'secondary'}`} onClick={() => setMode('knn')}>KNN 最近邻</button>
        <button className={`btn ${mode === 'svm' ? '' : 'secondary'}`} onClick={() => setMode('svm')}>SVM 最大间隔</button>
      </div>

      {mode === 'knn' && (
        <Card title="KNN 决策边界" sub="点击图中任意位置放置查询点；黄线连到它的 k 个最近邻">
          <div className="controls" style={{ marginBottom: 14 }}>
            <Slider label="邻居数 k" value={k} min={1} max={11} step={2} onChange={setK} fmt={(v) => v} />
            <div className="control">
              <label><span>查询点预测</span><b style={{ color: Number(knnPred) === 1 ? 'var(--orange)' : 'var(--accent)' }}>类别 {knnPred}</b></label>
            </div>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8, cursor: 'crosshair' }} onClick={onClick}>
            {knnGrid.cells.map((c, i) => (
              <rect key={i} x={sx(c.x) - cw / 2} y={sy(c.y) - ch / 2} width={cw + 1} height={ch + 1}
                fill={c.pred === 1 ? 'var(--orange)' : 'var(--accent)'} opacity={0.15} />
            ))}
            {/* 最近邻连线 */}
            {neighbors.map((n, i) => (
              <line key={`l${i}`} x1={sx(query.x)} y1={sy(query.y)} x2={sx(PTS[n.i].x)} y2={sy(PTS[n.i].y)}
                stroke="var(--yellow)" strokeWidth={1.5} opacity={0.7} />
            ))}
            {PTS.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={6}
                fill={p.label === 1 ? 'var(--orange)' : 'var(--accent)'} stroke="var(--bg)" strokeWidth={2} />
            ))}
            {/* 查询点 */}
            <circle cx={sx(query.x)} cy={sy(query.y)} r={9} fill="var(--yellow)" stroke="var(--bg)" strokeWidth={2} />
          </svg>
          <Legend items={[
            { color: 'var(--accent)', label: '类别 0' },
            { color: 'var(--orange)', label: '类别 1' },
            { color: 'var(--yellow)', label: '查询点 + 最近邻' },
          ]} />
        </Card>
      )}

      {mode === 'svm' && (
        <Card title="SVM 最大间隔超平面" sub="实线是决策边界，两条虚线是间隔边界，带圈的是支持向量">
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
            {/* 间隔带 */}
            {(() => {
              const top = svmLine(1), bot = svmLine(-1)
              return (
                <polygon
                  points={`${sx(top[0][0])},${sy(top[0][1])} ${sx(top[1][0])},${sy(top[1][1])} ${sx(bot[1][0])},${sy(bot[1][1])} ${sx(bot[0][0])},${sy(bot[0][1])}`}
                  fill="var(--accent)" opacity={0.08} />
              )
            })()}
            {/* 间隔虚线 */}
            {[1, -1].map((off, i) => {
              const l = svmLine(off)
              return <line key={i} x1={sx(l[0][0])} y1={sy(l[0][1])} x2={sx(l[1][0])} y2={sy(l[1][1])} stroke="var(--text-faint)" strokeWidth={1.5} strokeDasharray="5 4" />
            })}
            {/* 决策线 */}
            {(() => { const l = svmLine(0); return <line x1={sx(l[0][0])} y1={sy(l[0][1])} x2={sx(l[1][0])} y2={sy(l[1][1])} stroke="var(--text)" strokeWidth={2.5} /> })()}
            {/* 点 + 支持向量高亮 */}
            {PTS.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={6}
                fill={p.label === 1 ? 'var(--orange)' : 'var(--accent)'}
                stroke={isSupport(p) ? 'var(--yellow)' : 'var(--bg)'} strokeWidth={isSupport(p) ? 3 : 2} />
            ))}
          </svg>
          <Legend items={[
            { color: 'var(--accent)', label: '类别 0' },
            { color: 'var(--orange)', label: '类别 1' },
            { color: 'var(--yellow)', label: '支持向量（决定边界的点）' },
          ]} />
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 8 }}>
            间隔宽度 ≈ {f2(2 * margin)}。SVM 的目标就是<strong>最大化这个间隔</strong>——只有黄圈标出的支持向量参与决定边界，删掉其他点边界都不变。
          </div>
        </Card>
      )}

      <div className="grid-2">
        <Card title="KNN：懒惰但直观">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>KNN 没有「训练」过程，预测时才现算距离——所以叫<strong>懒惰学习</strong>。k 越小边界越锯齿（易受噪声影响），k 越大越平滑（可能欠拟合）。</p>
            <p>缺点：预测要算到所有训练点的距离，数据大时慢；且对特征尺度敏感，<strong>必须先标准化</strong>。</p>
            <p><code>from sklearn.neighbors import KNeighborsClassifier</code>；<code>KNeighborsClassifier(n_neighbors=5)</code>。</p>
          </div>
        </Card>
        <Card title="SVM：最大间隔与核技巧">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>SVM 找的不是随便一条分界线，而是<strong>离两类都最远</strong>的那条，泛化更稳。软间隔参数 <strong>C</strong> 控制对错分的容忍度。</p>
            <p>遇到线性不可分？用<strong>核技巧</strong>（RBF 核等）把数据隐式映射到高维，在那里线性可分。</p>
            <p><code>from sklearn.svm import SVC</code>；<code>SVC(kernel='rbf', C=1.0)</code>。中小数据上 SVM 仍很能打，但大数据训练慢。</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>共同点：</b> 两者都对<strong>特征尺度敏感</strong>，用前务必标准化（<code>StandardScaler</code>）。否则量纲大的特征会主导距离/间隔的计算。
      </Callout>
    </div>
  )
}
