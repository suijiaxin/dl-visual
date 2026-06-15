import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { movingAverage, expSmoothing, mean } from '../../lib/mathx.js'

// 时序：构造「趋势+季节+噪声」序列，展示分解，并对比移动平均/指数平滑预测。

const W = 640, H = 260, PAD = 40
const N = 48 // 历史点数

// 真实成分
const trend = (t) => 10 + 0.35 * t
const season = (t) => 4 * Math.sin((2 * Math.PI * t) / 12) // 周期 12
function genSeries() {
  // 固定噪声（可复现）
  const noiseSeed = [0.5, -1.2, 0.8, -0.3, 1.1, -0.7, 0.2, 0.9, -1.0, 0.4, -0.5, 1.3, -0.9, 0.6, -0.2, 0.7,
    -1.1, 0.3, 0.95, -0.6, 1.0, -0.4, 0.15, -0.85, 0.55, -1.15, 0.75, -0.35, 1.05, -0.65, 0.25, 0.88,
    -0.98, 0.45, -0.55, 1.25, -0.92, 0.62, -0.22, 0.72, -1.08, 0.32, 0.9, -0.58, 0.98, -0.42, 0.18, -0.8]
  return Array.from({ length: N }, (_, t) => trend(t) + season(t) + noiseSeed[t] * 1.2)
}
const SERIES = genSeries()

export default function TimeSeries() {
  const [window, setWindow] = useState(6)
  const [alpha, setAlpha] = useState(0.4)
  const [horizon, setHorizon] = useState(12)

  const ma = useMemo(() => movingAverage(SERIES, window), [window])
  const es = useMemo(() => expSmoothing(SERIES, alpha), [alpha])

  // 分解：趋势(线性拟合) + 季节(去趋势后按周期平均) + 残差
  const decomp = useMemo(() => {
    const ts = Array.from({ length: N }, (_, t) => t)
    const mt = mean(ts), my = mean(SERIES)
    let num = 0, den = 0
    ts.forEach((t, i) => { num += (t - mt) * (SERIES[i] - my); den += (t - mt) ** 2 })
    const slope = num / den, intercept = my - slope * mt
    const trendFit = ts.map((t) => slope * t + intercept)
    const detrended = SERIES.map((v, i) => v - trendFit[i])
    // 季节：按 mod 12 平均
    const period = 12
    const seasonAvg = new Array(period).fill(0)
    const counts = new Array(period).fill(0)
    detrended.forEach((v, i) => { seasonAvg[i % period] += v; counts[i % period]++ })
    for (let i = 0; i < period; i++) seasonAvg[i] /= counts[i]
    const seasonal = ts.map((t) => seasonAvg[t % period])
    const resid = SERIES.map((v, i) => v - trendFit[i] - seasonal[i])
    return { trendFit, seasonal, resid, slope, intercept, seasonAvg }
  }, [])

  // 预测：用趋势+季节外推（简单的分解预测法）
  const forecast = useMemo(() => {
    const out = []
    for (let h = 0; h < horizon; h++) {
      const t = N + h
      out.push(decomp.slope * t + decomp.intercept + decomp.seasonAvg[t % 12])
    }
    return out
  }, [decomp, horizon])

  const total = N + horizon
  const allVals = [...SERIES, ...forecast, ...ma, ...es]
  const ymin = Math.min(...allVals) - 1
  const ymax = Math.max(...allVals) + 1
  const sx = (t) => PAD + (t / (total - 1)) * (W - 2 * PAD)
  const sy = (v) => H - PAD - ((v - ymin) / (ymax - ymin)) * (H - 2 * PAD)

  const path = (arr, offset = 0) => arr.map((v, i) => `${sx(i + offset)},${sy(v)}`).join(' ')

  return (
    <div>
      <PageHeader
        eyebrow="09 · 时序预测"
        title="时间序列：分解趋势与季节，再外推未来"
        lead="时序数据（流量、销量、电量）通常由三部分叠加：长期趋势 + 周期性季节波动 + 随机残差。把它们拆开，就能分别理解、分别建模。最朴素的预测就是把趋势线和季节模式向未来延伸。下面看真实分解 + 三种方法的对比。"
      />

      <Card title="原始序列 + 平滑 + 预测" sub="实线=历史；虚线=向未来外推的预测；移动平均/指数平滑用于平滑历史">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="移动平均窗口" value={window} min={2} max={12} step={1} onChange={setWindow} fmt={(v) => v} />
          <Slider label="指数平滑 α" value={alpha} min={0.05} max={0.95} step={0.05} onChange={setAlpha} fmt={f2} />
          <Slider label="预测步数" value={horizon} min={3} max={18} step={1} onChange={setHorizon} fmt={(v) => v} />
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
          {/* 历史/预测分界 */}
          <line x1={sx(N - 1)} y1={PAD} x2={sx(N - 1)} y2={H - PAD} stroke="var(--border)" strokeDasharray="4 4" />
          <text x={sx(N - 1) + 4} y={PAD + 4} fontSize="10" fill="var(--text-faint)">↓ 预测</text>

          {/* 移动平均 */}
          <polyline points={path(ma)} fill="none" stroke="var(--orange)" strokeWidth={1.6} opacity={0.8} />
          {/* 指数平滑 */}
          <polyline points={path(es)} fill="none" stroke="var(--green)" strokeWidth={1.6} opacity={0.8} />
          {/* 原始 */}
          <polyline points={path(SERIES)} fill="none" stroke="var(--accent)" strokeWidth={2} />
          {SERIES.map((v, i) => <circle key={i} cx={sx(i)} cy={sy(v)} r={2.3} fill="var(--accent)" />)}
          {/* 预测 */}
          <polyline points={`${sx(N - 1)},${sy(SERIES[N - 1])} ${path(forecast, N)}`} fill="none" stroke="var(--pink)" strokeWidth={2.5} strokeDasharray="6 4" />
          {forecast.map((v, i) => <circle key={i} cx={sx(N + i)} cy={sy(v)} r={2.6} fill="var(--pink)" />)}
        </svg>

        <Legend items={[
          { color: 'var(--accent)', label: '原始序列' },
          { color: 'var(--orange)', label: `移动平均(窗口${window})` },
          { color: 'var(--green)', label: `指数平滑(α=${f2(alpha)})` },
          { color: 'var(--pink)', label: '趋势+季节预测' },
        ]} />
      </Card>

      <Card title="时序分解：趋势 + 季节 + 残差" sub="同一条序列拆成三层，各自的规律一目了然">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {[
            { name: '趋势 Trend', data: decomp.trendFit, color: 'var(--accent)', desc: `斜率 ${f2(decomp.slope)}/期，长期${decomp.slope > 0 ? '上升' : '下降'}` },
            { name: '季节 Seasonal', data: Array.from({ length: N }, (_, t) => decomp.seasonal[t]), color: 'var(--orange)', desc: '周期=12，规律性波动' },
            { name: '残差 Residual', data: decomp.resid, color: 'var(--text-faint)', desc: '去掉趋势和季节后剩下的随机噪声' },
          ].map((row, ri) => {
            const rmin = Math.min(...row.data), rmax = Math.max(...row.data)
            const h = 56
            const ry = (v) => h - 6 - ((v - rmin) / (rmax - rmin || 1)) * (h - 12)
            return (
              <div key={ri} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 110, flexShrink: 0 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: row.color }}>{row.name}</div>
                  <div style={{ fontSize: 10.5, color: 'var(--text-faint)' }}>{row.desc}</div>
                </div>
                <svg viewBox={`0 0 ${W} ${h}`} style={{ flex: 1, background: 'var(--bg-elev)', borderRadius: 6, height: h }}>
                  <polyline points={row.data.map((v, i) => `${4 + (i / (N - 1)) * (W - 8)},${ry(v)}`).join(' ')}
                    fill="none" stroke={row.color} strokeWidth={1.6} />
                </svg>
              </div>
            )
          })}
        </div>
      </Card>

      <div className="grid-2">
        <Card title="三种方法的取舍">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>移动平均：</strong> 窗口越大越平滑，但越滞后。只能平滑历史，不擅长外推。</p>
            <p><strong>指数平滑：</strong> α 越大越看重近期。Holt-Winters 版本能同时建模趋势和季节，是经典预测主力。</p>
            <p><strong>分解外推：</strong> 把趋势线和季节模式直接延伸到未来，简单透明，适合规律性强的序列。</p>
          </div>
        </Card>
        <Card title="进阶方法与框架">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>ARIMA：</strong> 经典统计模型，建模自相关；<code>statsmodels</code> 的 <code>ARIMA</code> / <code>SARIMAX</code>（带季节）。</p>
            <p><strong>Prophet：</strong> Facebook 开源，自动处理趋势+多重季节+节假日，调参友好，业务预测常用。</p>
            <p><strong>树模型/深度学习：</strong> 把时序转成监督学习（滞后特征 + 滑窗统计）后喂给 LightGBM；或用 LSTM/Transformer 做长序列预测。</p>
            <p>评估别用随机划分——时序必须<strong>按时间切分</strong>训练/验证集，防止「未来信息泄漏」。</p>
          </div>
        </Card>
      </div>

      <Callout type="warn">
        <b>关键纪律：</b> 时序的验证集必须在训练集「之后」的时间段。任何用到未来数据的特征（如全局均值、未来滚动统计）都会造成乐观的假象，上线即翻车。
      </Callout>
    </div>
  )
}
