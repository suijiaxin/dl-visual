import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { gini } from '../../lib/mathx.js'

// 真实决策树：按基尼增益递归在 x/y 上找最优阈值切分 2D 平面。
// 调最大深度，看决策边界从欠拟合到过拟合，以及树结构。

const W = 420, H = 360, PAD = 30
const RANGE = 10 // [0,10]

// 一个非线性可分的两类数据（异或式分布，单层切不开）
const DATA = [
  // 类 0：左下 + 右上
  ...[[2, 2], [3, 1.5], [1.5, 3], [2.5, 2.8], [1, 1], [3.5, 2.2]].map((p) => ({ x: p[0], y: p[1], label: 0 })),
  ...[[7.5, 8], [8, 7], [8.5, 8.5], [7, 7.5], [9, 8], [8.2, 6.8]].map((p) => ({ x: p[0], y: p[1], label: 0 })),
  // 类 1：左上 + 右下
  ...[[2, 8], [3, 7.5], [1.5, 7], [2.5, 8.5], [1, 8], [3.5, 7.8]].map((p) => ({ x: p[0], y: p[1], label: 1 })),
  ...[[8, 2], [7.5, 3], [8.5, 1.5], [7, 2.5], [9, 2], [8.2, 3.2]].map((p) => ({ x: p[0], y: p[1], label: 1 })),
]

// 在某个数据子集上找最优分裂（特征 + 阈值），返回基尼增益最大的切分
function bestSplit(data) {
  const parentGini = gini(data.map((d) => d.label))
  let best = null
  for (const feat of ['x', 'y']) {
    const vals = [...new Set(data.map((d) => d[feat]))].sort((a, b) => a - b)
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2
      const left = data.filter((d) => d[feat] <= thr)
      const right = data.filter((d) => d[feat] > thr)
      if (!left.length || !right.length) continue
      const wGini = (left.length * gini(left.map((d) => d.label)) + right.length * gini(right.map((d) => d.label))) / data.length
      const gain = parentGini - wGini
      if (!best || gain > best.gain) best = { feat, thr, gain, left, right }
    }
  }
  return best
}

// 递归建树
function buildTree(data, depth, maxDepth) {
  const labels = data.map((d) => d.label)
  const g = gini(labels)
  const counts = [labels.filter((l) => l === 0).length, labels.filter((l) => l === 1).length]
  const pred = counts[1] > counts[0] ? 1 : 0
  const node = { gini: g, n: data.length, pred, counts, leaf: true }
  if (depth >= maxDepth || g < 1e-6 || data.length < 2) return node
  const split = bestSplit(data)
  if (!split || split.gain < 1e-6) return node
  node.leaf = false
  node.feat = split.feat
  node.thr = split.thr
  node.left = buildTree(split.left, depth + 1, maxDepth)
  node.right = buildTree(split.right, depth + 1, maxDepth)
  return node
}

// 对一个点用树预测
function predict(node, p) {
  if (node.leaf) return node.pred
  return p[node.feat] <= node.thr ? predict(node.left, p) : predict(node.right, p)
}

export default function DecisionTree() {
  const [maxDepth, setMaxDepth] = useState(3)

  const tree = useMemo(() => buildTree(DATA, 0, maxDepth), [maxDepth])

  const acc = useMemo(() => {
    const correct = DATA.filter((d) => predict(tree, d) === d.label).length
    return correct / DATA.length
  }, [tree])

  const sx = (x) => PAD + (x / RANGE) * (W - 2 * PAD)
  const sy = (y) => H - PAD - (y / RANGE) * (H - 2 * PAD)

  // 决策区域背景网格
  const grid = useMemo(() => {
    const cells = []
    const G = 30
    for (let i = 0; i < G; i++)
      for (let j = 0; j < G; j++) {
        const x = (i + 0.5) * (RANGE / G)
        const y = (j + 0.5) * (RANGE / G)
        cells.push({ x, y, pred: predict(tree, { x, y }) })
      }
    return { cells, G }
  }, [tree])
  const cw = (W - 2 * PAD) / grid.G
  const ch = (H - 2 * PAD) / grid.G

  // 简单树结构布局（递归算 x 坐标）
  const treeLayout = useMemo(() => {
    const nodes = []
    const edges = []
    let leafX = 0
    const place = (node, depth) => {
      let x
      if (node.leaf) {
        x = leafX++
      } else {
        const lx = place(node.left, depth + 1)
        const rx = place(node.right, depth + 1)
        x = (lx + rx) / 2
      }
      node._x = x
      node._d = depth
      nodes.push(node)
      if (!node.leaf) {
        edges.push([node, node.left])
        edges.push([node, node.right])
      }
      return x
    }
    place(tree, 0)
    return { nodes, edges, leaves: leafX }
  }, [tree])

  const TW = 560, depthMax = Math.max(...treeLayout.nodes.map((n) => n._d), 1)
  const TH = (depthMax + 1) * 70 + 30
  const tnx = (n) => 40 + (n._x / Math.max(1, treeLayout.leaves - 1)) * (TW - 80)
  const tny = (n) => 30 + n._d * 70

  return (
    <div>
      <PageHeader
        eyebrow="03 · 决策树"
        title="决策树：一连串 if-else 切分特征空间"
        lead="决策树像玩「20 个问题」：每次挑一个特征和阈值，把数据切成两半，让每半尽量「纯」（同一类）。纯度用基尼系数衡量。这样递归下去，特征空间被切成一块块矩形区域。拖动最大深度，看边界怎么从粗糙变精细——以及怎么开始过拟合。"
      />

      <Card title="决策边界 + 树结构" sub="左：树把平面切成的矩形决策区域；右：真实生成的树（按基尼增益分裂）">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="最大深度 max_depth" value={maxDepth} min={1} max={6} step={1} onChange={setMaxDepth} fmt={(v) => v} />
          <div className="control">
            <label><span>训练准确率</span><b style={{ color: acc === 1 ? 'var(--green)' : 'var(--orange)' }}>{Math.round(acc * 100)}%</b></label>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {/* 决策区域 */}
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 380, background: 'var(--bg-elev)', borderRadius: 8 }}>
            {grid.cells.map((c, i) => (
              <rect key={i} x={sx(c.x) - cw / 2} y={sy(c.y) - ch / 2} width={cw + 1} height={ch + 1}
                fill={c.pred === 1 ? 'var(--orange)' : 'var(--accent)'} opacity={0.18} />
            ))}
            {DATA.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={6}
                fill={p.label === 1 ? 'var(--orange)' : 'var(--accent)'} stroke="var(--bg)" strokeWidth={2} />
            ))}
          </svg>

          {/* 树结构 */}
          <svg viewBox={`0 0 ${TW} ${TH}`} style={{ width: '100%', maxWidth: 560, background: 'var(--bg-elev)', borderRadius: 8 }}>
            {treeLayout.edges.map(([a, b], i) => (
              <line key={i} x1={tnx(a)} y1={tny(a)} x2={tnx(b)} y2={tny(b)} stroke="var(--border)" strokeWidth={1.5} />
            ))}
            {treeLayout.nodes.map((n, i) => (
              <g key={i}>
                {n.leaf ? (
                  <>
                    <circle cx={tnx(n)} cy={tny(n)} r={16}
                      fill={n.pred === 1 ? 'var(--orange)' : 'var(--accent)'} opacity={0.85} stroke="var(--bg)" strokeWidth={2} />
                    <text x={tnx(n)} y={tny(n) + 4} textAnchor="middle" fontSize="10" fill="#fff" fontFamily="var(--mono)">{n.counts[0]}/{n.counts[1]}</text>
                  </>
                ) : (
                  <>
                    <rect x={tnx(n) - 34} y={tny(n) - 14} width={68} height={28} rx={6}
                      fill="var(--bg-card)" stroke="var(--border)" strokeWidth={1.5} />
                    <text x={tnx(n)} y={tny(n) - 1} textAnchor="middle" fontSize="10" fill="var(--text)" fontFamily="var(--mono)">{n.feat} ≤ {f2(n.thr)}</text>
                    <text x={tnx(n)} y={tny(n) + 10} textAnchor="middle" fontSize="8" fill="var(--text-faint)">gini={f2(n.gini)}</text>
                  </>
                )}
              </g>
            ))}
          </svg>
        </div>

        <Legend items={[
          { color: 'var(--accent)', label: '类别 0 / 预测为 0 的区域' },
          { color: 'var(--orange)', label: '类别 1 / 预测为 1 的区域' },
        ]} />
      </Card>

      <div className="grid-2">
        <Card title="基尼系数：怎么挑最优分裂？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>基尼 = 1 − Σpᵢ²，衡量一个节点里类别的「混乱度」：全是一类时 gini=0（最纯），两类各半时 gini=0.5（最乱）。</p>
            <p>每次分裂，树会<strong>枚举所有特征的所有阈值</strong>，挑「分裂后加权基尼下降最多」的那个。这就是上图右侧每个方块节点的由来。</p>
            <p>深度越大，叶子越纯、训练准确率越高——但也越容易把噪声也学进去。</p>
          </div>
        </Card>
        <Card title="过拟合与框架">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>把深度调到 5~6：训练准确率冲到 100%，但边界出现一些只为单个点存在的「孤岛」——这就是<strong>过拟合</strong>，换新数据就翻车。</p>
            <p>实践中用 <code>max_depth</code>、<code>min_samples_leaf</code> 等剪枝控制复杂度。</p>
            <p><code>from sklearn.tree import DecisionTreeClassifier</code>；<code>DecisionTreeClassifier(max_depth=3, criterion='gini')</code>。单棵树不稳，所以才有了下面的<strong>随机森林</strong>和<strong>梯度提升</strong>。</p>
          </div>
        </Card>
      </div>

      <Callout type="warn">
        <b>关键直觉：</b> 单棵决策树方差大——数据变一点，整棵树可能长得完全不同。集成多棵树（森林/提升）正是为了治这个病。
      </Callout>
    </div>
  )
}
