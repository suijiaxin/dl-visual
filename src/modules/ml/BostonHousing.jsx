import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2, f3 } from '../../components/ui.jsx'
import { mean, std, r2 } from '../../lib/mathx.js'

// 波士顿房价：多元线性回归实战。真实风格子集，4 个特征预测房价中位数(MEDV)。
// 真实梯度下降训练，标准化，看系数、预测vs真实、R²。顶部标注数据集已弃用。

// [RM(房间数), LSTAT(低收入人群%), PTRATIO(师生比), CRIM(犯罪率), MEDV(房价 千美元)]
const RAW = [
  [6.575, 4.98, 15.3, 0.006, 24.0], [6.421, 9.14, 17.8, 0.027, 21.6], [7.185, 4.03, 17.8, 0.027, 34.7],
  [6.998, 2.94, 18.7, 0.032, 33.4], [7.147, 5.33, 18.7, 0.069, 36.2], [6.430, 5.21, 18.7, 0.030, 28.7],
  [6.012, 12.43, 15.2, 0.088, 22.9], [6.172, 19.15, 15.2, 0.145, 27.1], [5.631, 29.93, 15.2, 0.211, 16.5],
  [6.004, 17.10, 15.2, 0.170, 18.9], [6.377, 20.45, 15.2, 0.225, 15.0], [6.009, 13.27, 15.2, 0.117, 18.9],
  [5.889, 15.71, 15.2, 0.094, 21.7], [5.949, 8.26, 21.0, 0.630, 20.4], [6.096, 10.26, 21.0, 0.638, 18.2],
  [5.834, 8.47, 21.0, 0.627, 19.9], [5.935, 6.58, 21.0, 1.054, 23.1], [5.990, 14.67, 21.0, 0.784, 17.5],
  [5.456, 11.69, 21.0, 0.803, 20.2], [5.727, 11.28, 21.0, 0.726, 18.2], [6.286, 21.02, 21.0, 1.252, 13.6],
  [6.279, 13.83, 21.0, 0.852, 19.6], [6.142, 18.72, 21.0, 1.234, 15.2], [5.813, 19.88, 21.0, 0.988, 14.5],
  [7.416, 6.07, 19.2, 0.150, 31.1], [6.781, 7.74, 19.2, 0.180, 26.6], [7.875, 3.32, 14.7, 0.014, 50.0],
  [7.610, 3.16, 14.7, 0.020, 42.3], [7.853, 3.81, 14.7, 0.034, 48.5], [6.030, 8.10, 17.4, 0.135, 22.0],
  [6.404, 5.64, 17.4, 0.094, 23.9], [8.069, 4.21, 14.7, 0.025, 38.7], [7.820, 3.57, 14.7, 0.061, 43.8],
  [5.569, 13.51, 20.2, 0.541, 17.4], [5.642, 14.66, 20.2, 0.404, 17.8], [6.066, 10.50, 20.2, 0.346, 19.4],
  [6.286, 9.54, 17.8, 0.040, 25.0], [6.625, 7.56, 17.8, 0.052, 28.4], [6.163, 14.34, 20.2, 0.452, 19.1],
  [7.420, 5.25, 17.6, 0.072, 33.2], [6.849, 6.12, 17.6, 0.084, 30.1], [5.966, 9.97, 20.2, 0.337, 20.3],
]

const FEAT = ['RM 房间数', 'LSTAT 低收入%', 'PTRATIO 师生比', 'CRIM 犯罪率']

export default function BostonHousing() {
  const [iterShown, setIterShown] = useState(400)

  // 标准化 4 个特征（目标 MEDV 不标准化，便于读分）
  const { X, y, scaler } = useMemo(() => {
    const feats = RAW.map((r) => r.slice(0, 4))
    const scaler = [0, 1, 2, 3].map((c) => {
      const col = feats.map((r) => r[c])
      return { mu: mean(col), sd: std(col) }
    })
    const X = feats.map((r) => r.map((v, c) => (v - scaler[c].mu) / scaler[c].sd))
    const y = RAW.map((r) => r[4])
    return { X, y, scaler }
  }, [])

  // 真实多元线性回归：梯度下降最小化 MSE
  const training = useMemo(() => {
    const n = X.length, d = X[0].length
    let w = new Array(d).fill(0), b = mean(y)
    const lr = 0.05
    const history = []
    for (let iter = 0; iter <= 600; iter++) {
      const gw = new Array(d).fill(0); let gb = 0
      const preds = X.map((xi) => xi.reduce((s, x, j) => s + x * w[j], b))
      for (let i = 0; i < n; i++) {
        const err = preds[i] - y[i]
        for (let j = 0; j < d; j++) gw[j] += err * X[i][j]
        gb += err
      }
      if (iter % 20 === 0) {
        history.push({ iter, w: [...w], b, mse: mean(y.map((yi, i) => (yi - preds[i]) ** 2)), preds: [...preds] })
      }
      for (let j = 0; j < d; j++) w[j] -= lr * 2 * gw[j] / n
      b -= lr * 2 * gb / n
    }
    return history
  }, [X, y])

  const curIdx = Math.min(Math.floor(iterShown / 20), training.length - 1)
  const cur = training[curIdx]
  const curR2 = r2(y, cur.preds)

  // 预测 vs 真实散点
  const PW = 380, PH = 340, PAD = 44
  const vmin = Math.min(...y, ...cur.preds) - 2
  const vmax = Math.max(...y, ...cur.preds) + 2
  const px = (v) => PAD + ((v - vmin) / (vmax - vmin)) * (PW - 2 * PAD)
  const py = (v) => PH - PAD - ((v - vmin) / (vmax - vmin)) * (PH - 2 * PAD)

  const maxW = Math.max(...cur.w.map(Math.abs), 0.1)

  return (
    <div>
      <PageHeader
        eyebrow="14 · 实战 · 波士顿房价"
        title="波士顿房价：多元线性回归实战"
        lead="经典的回归入门数据集：用社区特征（房间数、低收入人群比例、师生比、犯罪率等）预测房价中位数。这里走多元线性回归全流程——标准化、真实梯度下降训练、读系数、用预测 vs 真实图和 R² 评估拟合质量。"
      />

      <Callout type="warn">
        <b>⚠️ 数据集伦理说明：</b> 原始波士顿房价数据集含一个基于种族的特征（B），设计上隐含歧视性假设，<b>scikit-learn 1.2 起已正式移除</b>（<code>load_boston</code> 不再可用）。官方推荐用 <b>加州房价</b>（<code>fetch_california_housing</code>）替代。本页仅用其中无伦理争议的房屋/社区特征做教学演示，建议实际项目改用加州房价数据集。
      </Callout>

      <Card title="① 训练多元线性回归（真实梯度下降）" sub="拖动看训练进程：MSE 下降、R² 上升、系数逐渐稳定">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="训练轮数 iteration" value={iterShown} min={0} max={600} step={20} onChange={setIterShown} fmt={(v) => v} />
          <div className="control"><label><span>R²（拟合优度）</span><b style={{ color: curR2 > 0.7 ? 'var(--green)' : 'var(--orange)' }}>{f3(curR2)}</b></label></div>
          <div className="control"><label><span>MSE</span><b>{f2(cur.mse)}</b></label></div>
        </div>

        <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginBottom: 8 }}>各特征系数（标准化后，可直接比较重要性；蓝=推高房价，橙=拉低房价）：</div>
        {FEAT.map((name, j) => (
          <div key={j} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <div style={{ width: 110, fontSize: 12, flexShrink: 0, textAlign: 'right' }}>{name}</div>
            <div style={{ flex: 1, height: 16, position: 'relative', background: 'var(--slot)', borderRadius: 4 }}>
              <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--border)' }} />
              <div style={{
                position: 'absolute', height: '100%', borderRadius: 4,
                left: cur.w[j] >= 0 ? '50%' : `${50 - Math.abs(cur.w[j]) / maxW * 50}%`,
                width: `${Math.abs(cur.w[j]) / maxW * 50}%`,
                background: cur.w[j] >= 0 ? 'var(--accent)' : 'var(--orange)',
              }} />
            </div>
            <div style={{ width: 50, fontSize: 11, fontFamily: 'var(--mono)', color: cur.w[j] >= 0 ? 'var(--accent)' : 'var(--orange)' }}>{f2(cur.w[j])}</div>
          </div>
        ))}
      </Card>

      <div className="grid-2">
        <Card title="② 预测 vs 真实" sub="点越靠近对角线，预测越准">
          <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', maxWidth: PW, background: 'var(--bg-elev)', borderRadius: 8 }}>
            {/* 对角线 y=x */}
            <line x1={px(vmin)} y1={py(vmin)} x2={px(vmax)} y2={py(vmax)} stroke="var(--text-faint)" strokeDasharray="5 4" />
            {/* 坐标轴 */}
            <line x1={PAD} y1={PH - PAD} x2={PW - PAD} y2={PH - PAD} stroke="var(--border)" />
            <line x1={PAD} y1={PAD} x2={PAD} y2={PH - PAD} stroke="var(--border)" />
            <text x={PW - PAD} y={PH - PAD + 20} textAnchor="end" fontSize="11" fill="var(--text-faint)">真实房价</text>
            <text x={PAD - 8} y={PAD - 8} fontSize="11" fill="var(--text-faint)">预测房价</text>
            {y.map((yi, i) => (
              <circle key={i} cx={px(yi)} cy={py(cur.preds[i])} r={4.5} fill="var(--cyan)" opacity={0.75} stroke="var(--bg)" strokeWidth={1.2} />
            ))}
          </svg>
          <Legend items={[
            { color: 'var(--cyan)', label: '每套房（真实, 预测）' },
            { color: 'var(--text-faint)', label: '完美预测线 y=x' },
          ]} />
        </Card>
        <Card title="系数怎么读？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><b style={{ color: 'var(--accent)' }}>RM（房间数）</b>系数为正且大：房间越多房价越高，符合常识。</p>
            <p><b style={{ color: 'var(--orange)' }}>LSTAT（低收入人群比例）</b>系数为负且大：社区低收入比例越高，房价越低——通常是最强的负向特征。</p>
            <p>因为特征都<b>标准化</b>到同一尺度，系数绝对值可以<b>直接比较重要性</b>。没标准化的话，系数大小会被单位制扭曲（如犯罪率和房间数量纲完全不同）。</p>
          </div>
        </Card>
      </div>

      <Card title="完整 sklearn 代码（用加州房价替代）">
        <div className="formula" style={{ fontSize: 12, lineHeight: 1.9, textAlign: 'left', whiteSpace: 'pre-wrap' }}>
{`from sklearn.datasets import fetch_california_housing   # 替代已弃用的 load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
print(r2_score(y_te, model.predict(X_te)))`}
        </div>
      </Card>

      <Callout>
        <b>想提升 R²？</b> 线性回归只能拟合线性关系。房价对特征常是非线性的——加多项式/交互特征，或直接换 <b>随机森林 / GBDT 回归</b>（<code>RandomForestRegressor</code> / <code>XGBRegressor</code>），在这类表格数据上通常能把 R² 再推高一截。流程（划分→预处理→训练→评估）完全不变，只换模型那一行。
      </Callout>
    </div>
  )
}
