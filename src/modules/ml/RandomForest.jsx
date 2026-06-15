import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { gini, mulberry32, majority } from '../../lib/mathx.js'

// 随机森林：对数据做 bootstrap 重采样 + 随机特征，训练多棵浅树，投票。
// 调树数量，看决策边界从单树的锯齿/孤岛变平滑、方差下降。

const W = 420, H = 360, PAD = 30
const RANGE = 10

// 同一份带噪声的两类数据（圆环内外 + 一些噪声点）
function genData() {
  const rng = mulberry32(7)
  const data = []
  for (let i = 0; i < 60; i++) {
    const x = rng() * RANGE
    const y = rng() * RANGE
    const dx = x - 5, dy = y - 5
    const r = Math.sqrt(dx * dx + dy * dy)
    let label = r < 2.8 ? 0 : 1
    if (rng() < 0.12) label = 1 - label // 12% 噪声
    data.push({ x, y, label })
  }
  return data
}
const DATA = genData()

function bestSplit(data, featSubset) {
  const parentGini = gini(data.map((d) => d.label))
  let best = null
  for (const feat of featSubset) {
    const vals = [...new Set(data.map((d) => d[feat]))].sort((a, b) => a - b)
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2
      const left = data.filter((d) => d[feat] <= thr)
      const right = data.filter((d) => d[feat] > thr)
      if (!left.length || !right.length) continue
      const wG = (left.length * gini(left.map((d) => d.label)) + right.length * gini(right.map((d) => d.label))) / data.length
      const gain = parentGini - wG
      if (!best || gain > best.gain) best = { feat, thr, gain, left, right }
    }
  }
  return best
}

function buildTree(data, depth, maxDepth, rng) {
  const labels = data.map((d) => d.label)
  const pred = majority(labels)
  const node = { pred: Number(pred), leaf: true }
  if (depth >= maxDepth || gini(labels) < 1e-6 || data.length < 3) return node
  // 随机特征子集（这里 2 维，随机选 1~2 个）
  const feats = rng() < 0.5 ? ['x', 'y'] : (rng() < 0.5 ? ['x'] : ['y'])
  const split = bestSplit(data, feats)
  if (!split || split.gain < 1e-6) return node
  node.leaf = false
  node.feat = split.feat
  node.thr = split.thr
  node.left = buildTree(split.left, depth + 1, maxDepth, rng)
  node.right = buildTree(split.right, depth + 1, maxDepth, rng)
  return node
}

function predict(node, p) {
  if (node.leaf) return node.pred
  return p[node.feat] <= node.thr ? predict(node.left, p) : predict(node.right, p)
}

function buildForest(nTrees) {
  const rng = mulberry32(42)
  const trees = []
  for (let t = 0; t < nTrees; t++) {
    // bootstrap：有放回采样
    const sample = []
    for (let i = 0; i < DATA.length; i++) sample.push(DATA[Math.floor(rng() * DATA.length)])
    trees.push(buildTree(sample, 0, 4, rng))
  }
  return trees
}

export default function RandomForest() {
  const [nTrees, setNTrees] = useState(15)
  const forest = useMemo(() => buildForest(nTrees), [nTrees])

  const forestPredict = (p) => {
    const votes = forest.map((t) => predict(t, p))
    return majority(votes)
  }
  // 投票比例（用于显示置信度过渡）
  const forestProb = (p) => {
    const votes = forest.map((t) => predict(t, p))
    return votes.filter((v) => v === 1).length / votes.length
  }

  const acc = useMemo(() => {
    const correct = DATA.filter((d) => Number(forestPredict(d)) === d.label).length
    return correct / DATA.length
  }, [forest])

  const sx = (x) => PAD + (x / RANGE) * (W - 2 * PAD)
  const sy = (y) => H - PAD - (y / RANGE) * (H - 2 * PAD)

  const grid = useMemo(() => {
    const cells = []
    const G = 32
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const x = (i + 0.5) * (RANGE / G)
        const y = (j + 0.5) * (RANGE / G)
        cells.push({ x, y, prob: forestProb({ x, y }) })
      }
    return { cells, G }
  }, [forest])
  const cw = (W - 2 * PAD) / grid.G
  const ch = (H - 2 * PAD) / grid.G

  return (
    <div>
      <PageHeader
        eyebrow="04 · 随机森林"
        title="随机森林：很多棵「弱树」投票，强过单棵树"
        lead="单棵决策树方差大、易过拟合。随机森林的思路是 Bagging：对数据做有放回重采样、每次分裂只看随机的特征子集，训练出很多棵互不相同的树，最后投票。一棵树的随机错误会被其他树抵消，整体更稳、更平滑。拖动树的数量，看决策边界从毛糙变顺滑。"
      />

      <Card title="森林的决策边界" sub="背景颜色深浅 = 投票给该类的树比例；树越多边界越平滑">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="树的数量 n_estimators" value={nTrees} min={1} max={60} step={1} onChange={setNTrees} fmt={(v) => v} />
          <div className="control">
            <label><span>训练准确率</span><b style={{ color: 'var(--green)' }}>{Math.round(acc * 100)}%</b></label>
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 420, background: 'var(--bg-elev)', borderRadius: 8 }}>
          {grid.cells.map((c, i) => {
            // prob 0(蓝)→1(橙)
            const op = Math.abs(c.prob - 0.5) * 0.5 + 0.06
            return (
              <rect key={i} x={sx(c.x) - cw / 2} y={sy(c.y) - ch / 2} width={cw + 1} height={ch + 1}
                fill={c.prob >= 0.5 ? 'var(--orange)' : 'var(--accent)'} opacity={op} />
            )
          })}
          {DATA.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={5.5}
              fill={p.label === 1 ? 'var(--orange)' : 'var(--accent)'} stroke="var(--bg)" strokeWidth={1.5} />
          ))}
        </svg>

        <Legend items={[
          { color: 'var(--accent)', label: '类别 0（圆环内）' },
          { color: 'var(--orange)', label: '类别 1（圆环外）' },
        ]} />
      </Card>

      <div className="grid-2">
        <Card title="Bagging 的两层随机性">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>① 样本随机（Bootstrap）：</strong> 每棵树训练在一份有放回重采样的数据上，看到的样本各不相同。</p>
            <p><strong>② 特征随机：</strong> 每次分裂只从随机的特征子集里挑最优，避免所有树都被某个强特征「带偏」，进一步去相关。</p>
            <p>多棵去相关的树平均后，<strong>方差大幅下降、偏差基本不变</strong>——这就是集成降方差的核心。把树的数量从 1 调到 60，边界肉眼可见地变干净。</p>
          </div>
        </Card>
        <Card title="框架与调参">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><code>from sklearn.ensemble import RandomForestClassifier</code></p>
            <p><code>RandomForestClassifier(n_estimators=100, max_depth=None, max_features='sqrt')</code></p>
            <p>关键超参：<strong>n_estimators</strong>（树越多越稳，但收益递减）、<strong>max_features</strong>（每次分裂看几个特征，控制去相关程度）。</p>
            <p>随机森林还能免费给出<strong>特征重要性</strong>（<code>feature_importances_</code>），是特征筛选的常用工具。</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>Bagging vs Boosting：</b> 随机森林是「并行造很多独立的树再投票」（降方差）；下一模块的 GBDT 是「串行造树、每棵专门纠正上一棵的错」（降偏差）。两条路线统治了表格数据。
      </Callout>
    </div>
  )
}
