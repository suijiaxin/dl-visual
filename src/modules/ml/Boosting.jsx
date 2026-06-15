import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, Formula, f2, f3 } from '../../components/ui.jsx'
import { mean, mse } from '../../lib/mathx.js'

// 梯度提升（GBDT）：在 1D 回归上逐轮用「树桩」拟合残差，
// 看预测怎么一步步逼近目标。调学习率和轮数。

const W = 600, H = 340, PAD = 44
const XMIN = 0, XMAX = 10

// 目标函数：非线性（带拐弯），单个树桩拟合不了，需要多轮叠加
const TRUE_FN = (x) => 2 * Math.sin(x * 0.8) + 0.3 * x + 3
// 训练样本
const XS = Array.from({ length: 24 }, (_, i) => 0.2 + i * (9.6 / 23))
const YS = XS.map((x) => TRUE_FN(x))
const YMIN = Math.min(...YS) - 1.5
const YMAX = Math.max(...YS) + 1.5

// 一个回归树桩：找单个最优分裂点，左右各输出均值（拟合当前残差）
function fitStump(xs, residuals) {
  let best = null
  const sorted = [...xs].sort((a, b) => a - b)
  for (let i = 0; i < sorted.length - 1; i++) {
    const thr = (sorted[i] + sorted[i + 1]) / 2
    const leftIdx = xs.map((x, k) => (x <= thr ? k : -1)).filter((k) => k >= 0)
    const rightIdx = xs.map((x, k) => (x > thr ? k : -1)).filter((k) => k >= 0)
    if (!leftIdx.length || !rightIdx.length) continue
    const lv = mean(leftIdx.map((k) => residuals[k]))
    const rv = mean(rightIdx.map((k) => residuals[k]))
    // 分裂后的平方误差
    let err = 0
    leftIdx.forEach((k) => (err += (residuals[k] - lv) ** 2))
    rightIdx.forEach((k) => (err += (residuals[k] - rv) ** 2))
    if (!best || err < best.err) best = { thr, lv, rv, err }
  }
  return best
}

// 跑 GBDT：返回每轮之后的预测序列（在密集网格上，用于画曲线）
function runGBDT(nRounds, lr) {
  const GRID = Array.from({ length: 120 }, (_, i) => XMIN + (i / 119) * (XMAX - XMIN))
  let preds = YS.map(() => mean(YS)) // 初始预测 = 均值
  let gridPreds = GRID.map(() => mean(YS))
  const history = [{ gridPreds: [...gridPreds], preds: [...preds], mse: mse(YS, preds) }]
  const stumps = []
  for (let r = 0; r < nRounds; r++) {
    const residuals = YS.map((y, k) => y - preds[k])
    const stump = fitStump(XS, residuals)
    if (!stump) break
    stumps.push(stump)
    preds = preds.map((p, k) => p + lr * (XS[k] <= stump.thr ? stump.lv : stump.rv))
    gridPreds = gridPreds.map((p, k) => p + lr * (GRID[k] <= stump.thr ? stump.lv : stump.rv))
    history.push({ gridPreds: [...gridPreds], preds: [...preds], mse: mse(YS, preds), thr: stump.thr })
  }
  return { GRID, history }
}

export default function Boosting() {
  const [nRounds, setNRounds] = useState(30)
  const [lr, setLr] = useState(0.3)
  const [step, setStep] = useState(8)

  const { GRID, history } = useMemo(() => runGBDT(nRounds, lr), [nRounds, lr])
  const curStep = Math.min(step, history.length - 1)
  const cur = history[curStep]

  const sx = (x) => PAD + ((x - XMIN) / (XMAX - XMIN)) * (W - 2 * PAD)
  const sy = (y) => H - PAD - ((y - YMIN) / (YMAX - YMIN)) * (H - 2 * PAD)

  const truePath = GRID.map((x) => `${sx(x)},${sy(TRUE_FN(x))}`).join(' ')
  const predPath = GRID.map((x, i) => `${sx(x)},${sy(cur.gridPreds[i])}`).join(' ')

  return (
    <div>
      <PageHeader
        eyebrow="05 · 梯度提升"
        title="梯度提升 GBDT：每棵树专门纠正上一棵的残差"
        lead="和随机森林的「并行投票」相反，Boosting 是串行的：先用一个简单模型预测，算出还差多少（残差），再训练下一棵树专门拟合这个残差，一轮轮加上去。预测曲线一步步逼近真实目标。这就是 XGBoost / LightGBM 的内核。拖动「当前轮数」单步观察，或调学习率看收敛快慢。"
      />

      <Formula>Fₘ(x) = Fₘ₋₁(x) + lr · hₘ(x)　其中 hₘ 拟合残差 rᵢ = yᵢ − Fₘ₋₁(xᵢ)</Formula>

      <Card title="逐轮逼近目标" sub="灰线是真实目标；蓝线是当前轮的累计预测；红竖线是每个样本当前的残差">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="查看第几轮" value={step} min={0} max={nRounds} step={1} onChange={setStep} fmt={(v) => Math.min(v, history.length - 1)} />
          <Slider label="学习率 learning_rate" value={lr} min={0.05} max={1} step={0.05} onChange={setLr} fmt={f2} />
          <Slider label="总轮数 n_estimators" value={nRounds} min={1} max={80} step={1} onChange={setNRounds} fmt={(v) => v} />
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" />

          {/* 残差竖线 */}
          {XS.map((x, i) => (
            <line key={`r${i}`} x1={sx(x)} y1={sy(YS[i])} x2={sx(x)} y2={sy(cur.preds[i])}
              stroke="var(--red)" strokeWidth={1.3} opacity={0.6} />
          ))}

          {/* 真实目标 */}
          <polyline points={truePath} fill="none" stroke="var(--text-faint)" strokeWidth={2} strokeDasharray="5 4" />
          {/* 当前预测 */}
          <polyline points={predPath} fill="none" stroke="var(--accent)" strokeWidth={2.5} />

          {/* 样本点 */}
          {XS.map((x, i) => (
            <circle key={i} cx={sx(x)} cy={sy(YS[i])} r={4} fill="var(--cyan)" stroke="var(--bg)" strokeWidth={1.5} />
          ))}

          <text x={W - PAD} y={PAD - 8} textAnchor="end" fontSize="12" fill="var(--accent)">第 {curStep} 轮预测</text>
        </svg>

        <Legend items={[
          { color: 'var(--cyan)', label: '训练样本' },
          { color: 'var(--text-faint)', label: '真实目标函数' },
          { color: 'var(--accent)', label: '累计预测 Fₘ(x)' },
          { color: 'var(--red)', label: '当前残差' },
        ]} />
      </Card>

      <div className="grid-3">
        <Card title="当前轮数">
          <div className="formula" style={{ textAlign: 'center', fontSize: 24, color: 'var(--accent)' }}>{curStep} / {nRounds}</div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>第 0 轮 = 只用均值预测</div>
        </Card>
        <Card title="训练 MSE">
          <div className="formula" style={{ textAlign: 'center', fontSize: 24, color: cur.mse < 0.3 ? 'var(--green)' : 'var(--orange)' }}>{f3(cur.mse)}</div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center' }}>每加一轮，误差下降</div>
        </Card>
        <Card title="学习率作用">
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', padding: '4px 0' }}>
            lr 越小，每棵树贡献越少，需要更多轮才收敛，但泛化通常更好（小步慢走不容易过拟合）。试试 lr=0.05 配 80 轮 vs lr=1 配 5 轮。
          </div>
        </Card>
      </div>

      <div className="grid-2">
        <Card title="为什么叫「梯度」提升？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>对平方损失，残差 r = y − F(x) 恰好是损失对预测的<strong>负梯度</strong>。所以「拟合残差」本质就是「沿损失的负梯度方向，用一棵树走一步」——梯度下降的函数空间版本。</p>
            <p>换别的损失函数（如分类用的对数损失），拟合的就是对应的负梯度，框架自动处理。</p>
          </div>
        </Card>
        <Card title="XGBoost vs LightGBM">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>XGBoost：</strong> 在 GBDT 上加了二阶泰勒近似 + 正则项 + 缺失值处理，精度高、稳，竞赛常胜将军。</p>
            <p><strong>LightGBM：</strong> 用直方图分裂 + leaf-wise 生长 + 单边采样，大数据上<strong>更快更省内存</strong>，工业界海量特征首选。</p>
            <p><code>import xgboost as xgb</code> / <code>import lightgbm as lgb</code>，API 都兼容 sklearn 的 fit/predict。关键参数同样是 <code>learning_rate</code> 和 <code>n_estimators</code>，二者要配合调。</p>
          </div>
        </Card>
      </div>

      <Callout type="warn">
        <b>过拟合警告：</b> 把轮数拖到很大、学习率也大时，蓝线会开始穿过每个样本点、偏离平滑的灰线——它在拟合噪声。实践中用早停（early_stopping）+ 验证集来选最优轮数。
      </Callout>
    </div>
  )
}
