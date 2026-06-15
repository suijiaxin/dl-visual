import React, { useState, useMemo, useEffect } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { kmeansStep, mulberry32, dist } from '../../lib/mathx.js'

// K-Means：单步「分配→更新质心」迭代动画，实时收敛。
// 调 k；肘部法选最优 k。无监督：没有标签，只看几何聚集。

const W = 440, H = 380, PAD = 34
const RANGE = 10
const COLORS = ['var(--accent)', 'var(--orange)', 'var(--green)', 'var(--pink)', 'var(--cyan)']

// 生成 3 个自然簇 + 噪声（无标签，仅坐标）
function genPoints() {
  const rng = mulberry32(11)
  const centers = [[2.5, 3], [7, 2.5], [5, 7.5]]
  const pts = []
  centers.forEach((c) => {
    for (let i = 0; i < 18; i++) {
      pts.push([
        Math.max(0.2, Math.min(9.8, c[0] + (rng() - 0.5) * 3)),
        Math.max(0.2, Math.min(9.8, c[1] + (rng() - 0.5) * 3)),
      ])
    }
  })
  return pts
}
const POINTS = genPoints()

// 初始质心（用种子，可复现）
function initCentroids(k) {
  const rng = mulberry32(100 + k)
  const cs = []
  for (let i = 0; i < k; i++) cs.push([1 + rng() * 8, 1 + rng() * 8])
  return cs
}

// 跑到收敛，记录每一步（用于肘部法的 inertia）
function runToConvergence(k) {
  let centroids = initCentroids(k)
  const steps = [{ centroids: centroids.map((c) => [...c]), assign: null }]
  let prevInertia = Infinity
  for (let iter = 0; iter < 30; iter++) {
    const { assign, newCentroids, inertia } = kmeansStep(POINTS, centroids)
    steps.push({ centroids: centroids.map((c) => [...c]), assign, inertia })
    if (Math.abs(prevInertia - inertia) < 1e-6) break
    prevInertia = inertia
    centroids = newCentroids
  }
  return steps
}

export default function Clustering() {
  const [k, setK] = useState(3)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  const steps = useMemo(() => runToConvergence(k), [k])
  const maxStep = steps.length - 1
  const curStep = Math.min(step, maxStep)
  const cur = steps[curStep]

  // 重置 step 当 k 改变
  useEffect(() => { setStep(0); setPlaying(false) }, [k])

  // 自动播放
  useEffect(() => {
    if (!playing) return
    if (curStep >= maxStep) { setPlaying(false); return }
    const t = setTimeout(() => setStep((s) => Math.min(s + 1, maxStep)), 700)
    return () => clearTimeout(t)
  }, [playing, curStep, maxStep])

  const sx = (x) => PAD + (x / RANGE) * (W - 2 * PAD)
  const sy = (y) => H - PAD - (y / RANGE) * (H - 2 * PAD)

  // 肘部法：每个 k 的最终 inertia
  const elbow = useMemo(() => {
    const out = []
    for (let kk = 1; kk <= 6; kk++) {
      const s = runToConvergence(kk)
      out.push({ k: kk, inertia: s[s.length - 1].inertia })
    }
    return out
  }, [])
  const maxInertia = Math.max(...elbow.map((e) => e.inertia))

  return (
    <div>
      <PageHeader
        eyebrow="07 · K-Means 聚类"
        title="K-Means：没有标签，也能把数据分群"
        lead="前面都是有标签的监督学习。聚类是无监督的——只有坐标，没人告诉你谁是哪类。K-Means 的算法极简：① 随机放 k 个质心 ② 每个点归到最近的质心 ③ 质心移到各自簇的中心 ④ 重复②③直到不再变。点「单步」或「自动播放」看它怎么收敛。"
      />

      <Card title="K-Means 迭代过程" sub="× 是质心；点的颜色 = 当前归属的簇。看质心怎么一步步移到簇中心">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="簇数 k" value={k} min={1} max={5} step={1} onChange={setK} fmt={(v) => v} />
          <div className="control">
            <label><span>迭代步</span><b>{curStep} / {maxStep}</b></label>
          </div>
        </div>
        <div className="btn-row" style={{ marginBottom: 12 }}>
          <button className="btn secondary" onClick={() => { setStep(0); setPlaying(false) }}>重置</button>
          <button className="btn ghost" onClick={() => setStep((s) => Math.max(0, s - 1))}>← 上一步</button>
          <button className="btn ghost" onClick={() => setStep((s) => Math.min(maxStep, s + 1))}>下一步 →</button>
          <button className="btn" onClick={() => { if (curStep >= maxStep) setStep(0); setPlaying((p) => !p) }}>{playing ? '⏸ 暂停' : '▶ 自动播放'}</button>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          {/* 点 */}
          {POINTS.map((p, i) => {
            const c = cur.assign ? cur.assign[i] : null
            return (
              <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={5.5}
                fill={c !== null ? COLORS[c % COLORS.length] : 'var(--text-faint)'}
                opacity={c !== null ? 0.85 : 0.5} stroke="var(--bg)" strokeWidth={1.5} />
            )
          })}
          {/* 质心 */}
          {cur.centroids.map((c, i) => (
            <g key={i}>
              <line x1={sx(c[0]) - 8} y1={sy(c[1]) - 8} x2={sx(c[0]) + 8} y2={sy(c[1]) + 8} stroke={COLORS[i % COLORS.length]} strokeWidth={3.5} />
              <line x1={sx(c[0]) - 8} y1={sy(c[1]) + 8} x2={sx(c[0]) + 8} y2={sy(c[1]) - 8} stroke={COLORS[i % COLORS.length]} strokeWidth={3.5} />
              <circle cx={sx(c[0])} cy={sy(c[1])} r={13} fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth={2} />
            </g>
          ))}
        </svg>

        <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 8 }}>
          {curStep === 0 ? '第 0 步：质心随机初始化，点还未分配。' :
            cur.assign && curStep < maxStep ? `第 ${curStep} 步：簇内平方和(惯性) = ${f2(cur.inertia)}，还在下降。` :
              `已收敛（${maxStep} 步）：惯性 = ${f2(cur.inertia)}，质心不再移动。`}
        </div>
      </Card>

      <div className="grid-2">
        <Card title="肘部法：到底该选几个簇？" sub="惯性随 k 增大而下降，「拐点」就是合适的 k">
          <svg viewBox="0 0 360 200" style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
            <polyline
              points={elbow.map((e) => `${30 + ((e.k - 1) / 5) * 300},${180 - (e.inertia / maxInertia) * 150}`).join(' ')}
              fill="none" stroke="var(--accent)" strokeWidth={2.5} />
            {elbow.map((e, i) => {
              const x = 30 + ((e.k - 1) / 5) * 300
              const y = 180 - (e.inertia / maxInertia) * 150
              const isElbow = e.k === 3
              return (
                <g key={i}>
                  <circle cx={x} cy={y} r={isElbow ? 7 : 4.5} fill={isElbow ? 'var(--green)' : 'var(--accent)'} stroke="var(--bg)" strokeWidth={2}
                    style={{ cursor: 'pointer' }} onClick={() => setK(e.k)} />
                  <text x={x} y={195} textAnchor="middle" fontSize="10" fill="var(--text-faint)">k={e.k}</text>
                </g>
              )
            })}
            <text x={30} y={14} fontSize="10" fill="var(--text-faint)">惯性（簇内平方和）</text>
          </svg>
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 6 }}>
            绿点 k=3 处曲线明显「拐弯」——再加簇收益骤减。这份数据正好是 3 个自然簇。点任意点切换 k 验证。
          </div>
        </Card>
        <Card title="K-Means 的脾气与框架">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>要自己定 k：</strong> 算法不会告诉你该分几类，得靠肘部法 / 轮廓系数等辅助判断。</p>
            <p><strong>对初始化敏感：</strong> 质心初始位置不好可能陷入局部最优，实践中用 <code>k-means++</code> 智能初始化 + 多次重启取最优。</p>
            <p><strong>假设簇是「球形、大小相近」：</strong> 遇到环形/条形簇会失灵，那时换 DBSCAN / 谱聚类。</p>
            <p><code>from sklearn.cluster import KMeans</code>；<code>KMeans(n_clusters=3, init='k-means++', n_init=10)</code>。</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>典型应用：</b> 用户分层（按行为聚成高/中/低价值群）、图像颜色量化、异常检测（离所有质心都远的点）。聚类结果常作为后续监督模型的一个特征。
      </Callout>
    </div>
  )
}
