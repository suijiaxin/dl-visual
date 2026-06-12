import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Slider, Formula, Legend, f2 } from '../components/ui.jsx'
import { activations, mulberry32, randn } from '../lib/mathx.js'

// 一个 2-3-1 的小网络，真实前向传播。节点颜色随激活值变化，连线粗细随权重大小变化。
const LAYERS = [2, 3, 1] // 输入2，隐藏3，输出1

function buildWeights(seed) {
  const rng = mulberry32(seed)
  const W = []
  const B = []
  for (let l = 1; l < LAYERS.length; l++) {
    const rows = LAYERS[l]
    const cols = LAYERS[l - 1]
    W.push(Array.from({ length: rows }, () => Array.from({ length: cols }, () => randn(rng) * 0.9)))
    B.push(Array.from({ length: rows }, () => randn(rng) * 0.3))
  }
  return { W, B }
}

export default function NeuralNet() {
  const [x0, setX0] = useState(0.8)
  const [x1, setX1] = useState(-0.5)
  const [actName, setActName] = useState('relu')
  const [seed, setSeed] = useState(7)

  const { W, B } = useMemo(() => buildWeights(seed), [seed])
  const act = activations[actName]

  // 前向传播，记录每层的加权和(z)和激活后(a)，用于可视化
  const forward = useMemo(() => {
    const input = [x0, x1]
    const layers = [{ a: input, z: input }]
    let a = input
    for (let l = 0; l < W.length; l++) {
      const z = W[l].map((row, i) => row.reduce((s, w, j) => s + w * a[j], 0) + B[l][i])
      // 输出层用线性，隐藏层用所选激活函数
      const isOut = l === W.length - 1
      const aNext = isOut ? z : z.map(act.fn)
      layers.push({ z, a: aNext })
      a = aNext
    }
    return layers
  }, [x0, x1, W, B, act])

  // 布局坐标
  const width = 640
  const height = 320
  const colX = LAYERS.map((_, i) => 90 + (i * (width - 180)) / (LAYERS.length - 1))
  const nodeY = (layer, i) => {
    const n = LAYERS[layer]
    const gap = height / (n + 1)
    return gap * (i + 1)
  }

  const allAct = forward.flatMap((l) => l.a)
  const maxAbs = Math.max(0.5, ...allAct.map(Math.abs))

  return (
    <div>
      <PageHeader
        eyebrow="01 · NEURAL NETWORK"
        title="神经元与前向传播：一切的最小单元"
        lead="一个神经元做的事就三步：把输入加权求和、加偏置、过一个非线性激活函数。把成千上万个这样的神经元连起来，就是神经网络。拖动输入或换激活函数，看数值如何在网络里流动。"
      />

      <Formula>z = w₁·x₁ + w₂·x₂ + ... + b　　a = activation(z)</Formula>

      <Card title="实时前向传播" sub="节点亮度 = 激活值大小；连线粗细 = 权重绝对值；红=负 蓝=正">
        <div className="controls">
          <Slider label="输入 x₁" value={x0} min={-2} max={2} step={0.1} onChange={setX0} fmt={f2} />
          <Slider label="输入 x₂" value={x1} min={-2} max={2} step={0.1} onChange={setX1} fmt={f2} />
          <div className="control">
            <label><span>激活函数</span></label>
            <select value={actName} onChange={(e) => setActName(e.target.value)}>
              {Object.entries(activations).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>
          <div className="control">
            <label><span>权重随机种子</span></label>
            <button className="btn secondary" onClick={() => setSeed((s) => s + 1)}>
              🎲 重新初始化
            </button>
          </div>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 10 }}>
          {/* 连线 */}
          {W.map((mat, l) =>
            mat.map((row, i) =>
              row.map((w, j) => {
                const x1c = colX[l]
                const y1c = nodeY(l, j)
                const x2c = colX[l + 1]
                const y2c = nodeY(l + 1, i)
                const col = w >= 0 ? '91,157,255' : '255,92,122'
                return (
                  <line
                    key={`${l}-${i}-${j}`}
                    x1={x1c} y1={y1c} x2={x2c} y2={y2c}
                    stroke={`rgba(${col},${Math.min(0.9, 0.15 + Math.abs(w) * 0.4)})`}
                    strokeWidth={0.5 + Math.abs(w) * 2}
                  />
                )
              })
            )
          )}
          {/* 节点 */}
          {LAYERS.map((n, l) =>
            Array.from({ length: n }).map((_, i) => {
              const aVal = forward[l].a[i]
              const intensity = Math.min(1, Math.abs(aVal) / maxAbs)
              const fill = aVal >= 0
                ? `rgba(91,157,255,${0.2 + intensity * 0.8})`
                : `rgba(255,92,122,${0.2 + intensity * 0.8})`
              const labels = ['输入层', '隐藏层', '输出层']
              return (
                <g key={`${l}-${i}`}>
                  <circle cx={colX[l]} cy={nodeY(l, i)} r={20} fill={fill} stroke="var(--border)" strokeWidth={1.5} />
                  <text x={colX[l]} y={nodeY(l, i) + 4} textAnchor="middle" fontSize="11" fill="var(--text)" fontFamily="var(--mono)">
                    {f2(aVal)}
                  </text>
                  {i === 0 && (
                    <text x={colX[l]} y={24} textAnchor="middle" fontSize="11" fill="var(--text-faint)">
                      {labels[l]}
                    </text>
                  )}
                </g>
              )
            })
          )}
        </svg>
        <Legend items={[
          { color: 'rgba(91,157,255,0.9)', label: '正激活值 / 正权重' },
          { color: 'rgba(255,92,122,0.9)', label: '负激活值 / 负权重' },
        ]} />
      </Card>

      <div className="grid-2">
        <Card title="为什么需要激活函数？">
          <div className="prose">
            <p>
              如果没有非线性激活，无论堆多少层，整个网络都等价于<strong>一个线性变换</strong>——
              再深也只能学直线。激活函数（ReLU、Sigmoid、Tanh）引入非线性，让网络能逼近任意复杂的函数。
            </p>
            <p>
              试试把激活函数切到不同选项，观察隐藏层节点的输出变化：ReLU 会把负值直接砍成 0（你会看到一些节点变暗）。
            </p>
          </div>
        </Card>
        <Card title="当前激活函数">
          <Formula>{act.label}(x) = {act.tex}</Formula>
          <ActPlot act={act} />
        </Card>
      </div>

      <Callout type="key">
        <b>串联下一步：</b> 现在网络的权重是随机的，所以输出毫无意义。
        模型怎么把这些随机权重「调」到能正确预测？答案就是下一模块的
        <b> 反向传播 + 梯度更新</b>。
      </Callout>
    </div>
  )
}

// 画激活函数曲线
function ActPlot({ act }) {
  const W = 280, H = 120, pad = 10
  const xs = []
  for (let x = -4; x <= 4; x += 0.1) xs.push(x)
  const toX = (x) => pad + ((x + 4) / 8) * (W - 2 * pad)
  const toY = (y) => H / 2 - y * 28
  const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'} ${toX(x)} ${toY(act.fn(x))}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
      <line x1={pad} y1={H / 2} x2={W - pad} y2={H / 2} stroke="var(--border)" strokeWidth={1} />
      <line x1={W / 2} y1={pad} x2={W / 2} y2={H - pad} stroke="var(--border)" strokeWidth={1} />
      <path d={path} fill="none" stroke="var(--cyan)" strokeWidth={2} />
    </svg>
  )
}
