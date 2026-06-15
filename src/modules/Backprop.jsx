import React, { useState, useRef, useEffect, useMemo } from 'react'
import { PageHeader, Card, Callout, Slider, Formula, f3 } from '../components/ui.jsx'
import { mulberry32, randn } from '../lib/mathx.js'

// 真实训练：一个 1→8→8→1 的 MLP 用梯度下降拟合目标函数。
// 全部前向 + 反向传播手写，你能看到 loss 真实下降、权重真实更新。

const TARGETS = {
  sine: { fn: (x) => Math.sin(x * 3), label: 'sin(3x)' },
  abs: { fn: (x) => Math.abs(x) * 1.2 - 0.6, label: '|x| 折线' },
  step: { fn: (x) => (x > 0 ? 0.6 : -0.6), label: '阶跃' },
}
const H = 8 // 隐藏层宽度

function tanh(x) { return Math.tanh(x) }
function dtanh(y) { return 1 - y * y } // 输入是 tanh 输出

function initNet(seed) {
  const rng = mulberry32(seed)
  const sc = 0.8
  return {
    W1: Array.from({ length: H }, () => randn(rng) * sc),
    b1: Array.from({ length: H }, () => randn(rng) * 0.1),
    W2: Array.from({ length: H }, () => randn(rng) * sc),
    b2: randn(rng) * 0.1,
  }
}

// 单样本前向，返回中间量供反向使用
function forward(net, x) {
  const z1 = net.W1.map((w, i) => w * x + net.b1[i])
  const a1 = z1.map(tanh)
  const y = a1.reduce((s, a, i) => s + a * net.W2[i], net.b2)
  return { z1, a1, y }
}

// 一步梯度下降（在整个 batch 上）
function trainStep(net, data, lr) {
  const gW1 = new Array(H).fill(0), gb1 = new Array(H).fill(0)
  const gW2 = new Array(H).fill(0)
  let gb2 = 0
  let loss = 0
  for (const [x, t] of data) {
    const { a1, y } = forward(net, x)
    const err = y - t // dL/dy for MSE (1/2)
    loss += 0.5 * err * err
    // 输出层梯度
    for (let i = 0; i < H; i++) gW2[i] += err * a1[i]
    gb2 += err
    // 反传到隐藏层
    for (let i = 0; i < H; i++) {
      const da1 = err * net.W2[i]
      const dz1 = da1 * dtanh(a1[i])
      gW1[i] += dz1 * x
      gb1[i] += dz1
    }
  }
  const n = data.length
  // 把所有梯度摊平成一组，交给优化器统一处理
  const grad = {
    W1: gW1.map((g) => g / n),
    b1: gb1.map((g) => g / n),
    W2: gW2.map((g) => g / n),
    b2: gb2 / n,
  }
  return { grad, loss: loss / n, grads: { gW1, gW2 } }
}

// ===== 优化器 =====
// SGD：最朴素，直接沿梯度反方向走
// Momentum：累积"惯性"速度，能冲过小坑、加速收敛
// Adam：自适应学习率 + 动量，深度学习框架(PyTorch/TF)的默认选择
const OPTIMIZERS = {
  sgd: { label: 'SGD（朴素梯度下降）' },
  momentum: { label: 'Momentum（带动量）' },
  adam: { label: 'Adam（自适应，框架默认）' },
}

function initOptState(net) {
  const zeros = (arr) => (Array.isArray(arr) ? arr.map(() => 0) : 0)
  return {
    m: { W1: zeros(net.W1), b1: zeros(net.b1), W2: zeros(net.W2), b2: 0 }, // 一阶矩 / 速度
    v: { W1: zeros(net.W1), b1: zeros(net.b1), W2: zeros(net.W2), b2: 0 }, // 二阶矩(Adam)
    t: 0,
  }
}

// 对单个参数张量应用一步更新；返回 [新参数, 新m, 新v]
function applyUpdate(param, grad, m, v, opt, lr, t) {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8, mom = 0.9
  const step = (p, g, mi, vi) => {
    if (opt === 'sgd') return [p - lr * g, mi, vi]
    if (opt === 'momentum') {
      const nm = mom * mi + g
      return [p - lr * nm, nm, vi]
    }
    // adam
    const nm = beta1 * mi + (1 - beta1) * g
    const nv = beta2 * vi + (1 - beta2) * g * g
    const mhat = nm / (1 - Math.pow(beta1, t))
    const vhat = nv / (1 - Math.pow(beta2, t))
    return [p - (lr * mhat) / (Math.sqrt(vhat) + eps), nm, nv]
  }
  if (Array.isArray(param)) {
    const np = [], nm = [], nv = []
    for (let i = 0; i < param.length; i++) {
      const [a, b, c] = step(param[i], grad[i], m[i], v[i])
      np.push(a); nm.push(b); nv.push(c)
    }
    return [np, nm, nv]
  }
  return step(param, grad, m, v)
}

function optimizerStep(net, grad, optState, opt, lr) {
  const t = optState.t + 1
  const next = {}, m = {}, v = {}
  for (const key of ['W1', 'b1', 'W2', 'b2']) {
    const [p, nm, nv] = applyUpdate(net[key], grad[key], optState.m[key], optState.v[key], opt, lr, t)
    next[key] = p; m[key] = nm; v[key] = nv
  }
  return { next, optState: { m, v, t } }
}

export default function Backprop() {
  const [targetName, setTargetName] = useState('sine')
  const [lr, setLr] = useState(0.5)
  const [opt, setOpt] = useState('sgd')
  const [seed, setSeed] = useState(3)
  const [running, setRunning] = useState(false)
  const [net, setNet] = useState(() => initNet(3))
  const [step, setStep] = useState(0)
  const [lossHist, setLossHist] = useState([])
  const [lastGrad, setLastGrad] = useState(null)
  const raf = useRef(null)
  const optStateRef = useRef(initOptState(initNet(3)))

  const data = useMemo(() => {
    const pts = []
    for (let x = -1.5; x <= 1.5; x += 0.06) pts.push([x, TARGETS[targetName].fn(x)])
    return pts
  }, [targetName])

  // 重置
  const reset = (newSeed = seed) => {
    cancelAnimationFrame(raf.current)
    setRunning(false)
    const fresh = initNet(newSeed)
    setNet(fresh)
    optStateRef.current = initOptState(fresh)
    setStep(0)
    setLossHist([])
    setLastGrad(null)
  }

  useEffect(() => { reset() }, [targetName])
  // 换优化器时重置优化器内部状态（动量/二阶矩），公平对比
  useEffect(() => {
    optStateRef.current = initOptState(net)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opt])

  // 训练循环
  useEffect(() => {
    if (!running) return
    let cur = net
    const tick = () => {
      // 每帧跑若干步，加快收敛观感
      let loss = 0, grads = null
      for (let k = 0; k < 4; k++) {
        const r = trainStep(cur, data, lr)
        const upd = optimizerStep(cur, r.grad, optStateRef.current, opt, lr)
        cur = upd.next
        optStateRef.current = upd.optState
        loss = r.loss
        grads = r.grads
      }
      setNet(cur)
      setStep((s) => s + 4)
      setLossHist((h) => [...h.slice(-300), loss])
      setLastGrad(grads)
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [running, lr, opt, data])

  // 画拟合曲线
  const W = 460, Hh = 240, pad = 24
  const toX = (x) => pad + ((x + 1.5) / 3) * (W - 2 * pad)
  const toY = (y) => Hh / 2 - y * (Hh / 2 - pad) * 0.85
  const targetPath = data.map(([x, t], i) => `${i ? 'L' : 'M'} ${toX(x)} ${toY(t)}`).join(' ')
  const predPath = data.map(([x], i) => `${i ? 'L' : 'M'} ${toX(x)} ${toY(forward(net, x).y)}`).join(' ')

  const curLoss = lossHist[lossHist.length - 1] ?? null

  return (
    <div>
      <PageHeader
        eyebrow="02 · BACKPROPAGATION"
        title="反向传播与梯度更新：模型如何「学」"
        lead="这里真实训练的就是一个 MLP（多层感知机，1→8→8→1 的全连接 DNN）。训练 = 反复做四件事：① 前向算出预测 ② 用损失函数衡量预测和真实值差多少 ③ 反向传播算出每个权重对误差的「责任」(梯度) ④ 沿梯度反方向微调权重。点开始，亲眼看 loss 下降、曲线逼近目标。"
      />

      <Formula>
        前向: ŷ = f(x; W)　　损失: L = ½(ŷ − y)²　　梯度: ∂L/∂W　　更新: W ← W − lr · ∂L/∂W
      </Formula>

      <Card title="实时训练一个神经网络" sub="蓝线是网络预测，灰线是目标函数。它们重合时，说明模型学会了">
        <div className="controls">
          <div className="control">
            <label><span>目标函数</span></label>
            <select value={targetName} onChange={(e) => setTargetName(e.target.value)}>
              {Object.entries(TARGETS).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>
          <Slider label="学习率 lr" value={lr} min={0.05} max={3} step={0.05} onChange={setLr} fmt={f3} />
          <div className="control">
            <label><span>优化器 Optimizer</span></label>
            <select value={opt} onChange={(e) => setOpt(e.target.value)}>
              {Object.entries(OPTIMIZERS).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>
          <div className="control" style={{ justifyContent: 'flex-end' }}>
            <div className="btn-row">
              <button className="btn" onClick={() => setRunning((r) => !r)}>
                {running ? '⏸ 暂停' : '▶ 开始训练'}
              </button>
              <button className="btn secondary" onClick={() => reset()}>↻ 重置</button>
              <button className="btn ghost" onClick={() => { const s = seed + 1; setSeed(s); reset(s) }}>🎲 换初始权重</button>
            </div>
          </div>
        </div>

        <div className="grid-2">
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>拟合效果</div>
            <svg viewBox={`0 0 ${W} ${Hh}`} style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
              <line x1={pad} y1={Hh / 2} x2={W - pad} y2={Hh / 2} stroke="var(--border)" strokeWidth={1} />
              <path d={targetPath} fill="none" stroke="var(--text-faint)" strokeWidth={2.5} strokeDasharray="5 4" />
              <path d={predPath} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>损失曲线 (越低越好)</div>
            <LossChart hist={lossHist} W={W} H={Hh} />
          </div>
        </div>

        <div className="grid-3" style={{ marginTop: 16 }}>
          <div className="stat">
            <div className="stat-val" style={{ color: 'var(--green)' }}>{step}</div>
            <div className="stat-label">训练步数 (steps)</div>
          </div>
          <div className="stat">
            <div className="stat-val" style={{ color: 'var(--orange)' }}>{curLoss != null ? f3(curLoss) : '—'}</div>
            <div className="stat-label">当前损失 Loss</div>
          </div>
          <div className="stat">
            <div className="stat-val" style={{ color: 'var(--accent)' }}>{(H * 2 + H + 1)}</div>
            <div className="stat-label">可训练参数量</div>
          </div>
        </div>
      </Card>

      <div className="grid-2">
        <Card title="优化器：怎么用梯度更聪明地走" sub="同样的梯度，不同的更新策略，收敛速度差很多">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p><strong style={{ color: 'var(--text-dim)' }}>SGD：</strong> 每步直接沿梯度反方向走 lr 步。简单，但遇到平坦区或震荡方向会很慢。</p>
            <p><strong style={{ color: 'var(--accent)' }}>Momentum：</strong> 累积历史梯度成"速度/惯性"，像小球滚下山，能冲过小坑、在一致方向上加速。</p>
            <p><strong style={{ color: 'var(--green)' }}>Adam：</strong> 动量 + 给每个参数自适应调整学习率（梯度大的参数步子自动变小）。<strong>PyTorch / TensorFlow 训练大模型的默认选择。</strong></p>
            <p style={{ color: 'var(--text-faint)' }}>切换优化器后重新训练，对比 loss 曲线下降的速度和平滑度——通常 Adam 最快最稳。</p>
          </div>
          <Formula>Adam: m←β₁m+(1-β₁)g　v←β₂v+(1-β₂)g²　W←W − lr·m̂/(√v̂+ε)</Formula>
        </Card>
        <Card title="学习率的影响" sub="梯度更新的步长">
          <div className="prose">
            <p>
              <strong>太小</strong>（如 0.05）：每步挪一点点，收敛极慢。<br />
              <strong>适中</strong>（如 0.5）：稳步下降。<br />
              <strong>太大</strong>（如 2.5+）：可能在最低点附近来回横跳甚至发散，loss 曲线会剧烈抖动。
            </p>
            <p>试着把学习率拉到最大再训练，观察 loss 曲线是否变得不稳定。</p>
          </div>
        </Card>
        <Card title="输出层梯度强度" sub="每个隐藏神经元对输出权重的梯度大小 (绝对值)">
          {lastGrad ? (
            <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', height: 90 }}>
              {lastGrad.gW2.map((g, i) => {
                const mag = Math.min(1, Math.abs(g) / (Math.max(...lastGrad.gW2.map(Math.abs)) || 1))
                return (
                  <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: '100%' }}>
                    <div style={{ height: `${mag * 100}%`, background: 'var(--accent-2)', borderRadius: '3px 3px 0 0', minHeight: 2 }} />
                  </div>
                )
              })}
            </div>
          ) : (
            <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>开始训练后显示梯度</div>
          )}
          <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 8 }}>
            梯度大 = 该权重对当前误差「责任」大，会被调得更多。随着 loss 下降，梯度整体会变小。
          </div>
        </Card>
      </div>

      <Callout type="key">
        <b>这就是「训练」的全部本质。</b> GPT 这样的大模型，训练时做的也是同一件事——只是参数从这里的{' '}
        {H * 2 + H + 1} 个变成几千亿个，目标函数从拟合曲线变成「预测下一个词」，数据从几十个点变成整个互联网的文本。
        反向传播算法让这套流程在任意深的网络上都能高效进行。
      </Callout>
    </div>
  )
}

function LossChart({ hist, W, H }) {
  const pad = 24
  if (hist.length < 2) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
        <text x={W / 2} y={H / 2} textAnchor="middle" fill="var(--text-faint)" fontSize="13">开始训练后显示</text>
      </svg>
    )
  }
  const max = Math.max(...hist)
  const min = Math.min(...hist)
  const toX = (i) => pad + (i / (hist.length - 1)) * (W - 2 * pad)
  const toY = (v) => H - pad - ((v - min) / (max - min || 1)) * (H - 2 * pad)
  const path = hist.map((v, i) => `${i ? 'L' : 'M'} ${toX(i)} ${toY(v)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
      <path d={path} fill="none" stroke="var(--orange)" strokeWidth={2} />
      <text x={W - pad} y={16} textAnchor="end" fill="var(--text-faint)" fontSize="10" fontFamily="var(--mono)">max {f3(max)}</text>
      <text x={W - pad} y={H - 6} textAnchor="end" fill="var(--text-faint)" fontSize="10" fontFamily="var(--mono)">min {f3(min)}</text>
    </svg>
  )
}
