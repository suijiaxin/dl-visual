import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { mean } from '../../lib/mathx.js'

// 航空客流预测：Box & Jenkins 的经典时序数据集 AirPassengers。
// 1949-1960 每月国际航空乘客数(千)，144 个点，趋势 + 乘性季节都很典型。
// 留出最后 24 个月做测试，用「趋势+乘性季节」模型预测，看 MAPE。

// 真实 AirPassengers 数据：1949.01 - 1960.12，单位千人次
const DATA = [
  112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
  115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
  145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
  171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
  196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
  204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
  242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
  284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
  315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
  340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
  360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
  417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]
const N = DATA.length // 144
const TEST_N = 24 // 留最后两年做测试
const TRAIN_N = N - TEST_N

const W = 680, H = 300, PAD = 46

export default function AirPassengers() {
  const [horizon, setHorizon] = useState(TEST_N)

  // 用训练集拟合「趋势(线性) + 乘性季节(月度因子)」模型
  const model = useMemo(() => {
    const train = DATA.slice(0, TRAIN_N)
    const ts = Array.from({ length: TRAIN_N }, (_, i) => i)
    // 线性趋势拟合
    const mt = mean(ts), my = mean(train)
    let num = 0, den = 0
    ts.forEach((t, i) => { num += (t - mt) * (train[i] - my); den += (t - mt) ** 2 })
    const slope = num / den, intercept = my - slope * mt
    const trend = (t) => slope * t + intercept
    // 乘性季节因子：实际/趋势，按月平均
    const ratios = Array.from({ length: 12 }, () => [])
    train.forEach((v, i) => ratios[i % 12].push(v / trend(i)))
    const seasonal = ratios.map((r) => mean(r))
    // 归一化季节因子使均值为 1
    const sMean = mean(seasonal)
    const seasonalNorm = seasonal.map((s) => s / sMean)
    return { slope, intercept, trend, seasonal: seasonalNorm }
  }, [])

  // 预测：趋势 × 季节因子
  const forecast = (t) => model.trend(t) * model.seasonal[t % 12]

  // 拟合值（训练段）+ 预测值（测试段及之后）
  const fitted = Array.from({ length: TRAIN_N }, (_, t) => forecast(t))
  const predicted = Array.from({ length: horizon }, (_, h) => forecast(TRAIN_N + h))

  // 测试集 MAPE（只在有真实值的部分）
  const mape = useMemo(() => {
    const m = Math.min(horizon, TEST_N)
    let s = 0
    for (let h = 0; h < m; h++) {
      const actual = DATA[TRAIN_N + h]
      s += Math.abs((actual - predicted[h]) / actual)
    }
    return (s / m) * 100
  }, [predicted, horizon])

  const total = TRAIN_N + horizon
  const allVals = [...DATA, ...predicted]
  const ymin = Math.min(...allVals) * 0.9
  const ymax = Math.max(...allVals) * 1.05
  const sx = (t) => PAD + (t / (Math.max(N, total) - 1)) * (W - 2 * PAD)
  const sy = (v) => H - PAD - ((v - ymin) / (ymax - ymin)) * (H - 2 * PAD)

  // 年份刻度
  const yearTicks = Array.from({ length: 12 }, (_, i) => ({ t: i * 12, year: 1949 + i }))

  return (
    <div>
      <PageHeader
        eyebrow="16 · 实战 · 航空客流"
        title="航空客流预测：时序分析的「教科书」数据集"
        lead="Box & Jenkins 1976 年用来讲 ARIMA 的 AirPassengers 数据集：1949–1960 年每月国际航班乘客数。它同时具备教科书级的「上升趋势」和「逐年放大的季节波动」（夏季出行高峰），是检验时序方法的标准样本。我们留出最后两年做测试，用『趋势 × 乘性季节』模型预测，并用 MAPE 评估。"
      />

      <Card title="历史客流 + 预测" sub="蓝实线=训练数据；灰=真实测试值；粉虚线=模型预测。注意季节波动逐年放大">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="预测月数" value={horizon} min={6} max={36} step={1} onChange={setHorizon} fmt={(v) => `${v} 月`} />
          <div className="control"><label><span>测试集 MAPE</span><b style={{ color: mape < 10 ? 'var(--green)' : 'var(--orange)' }}>{f2(mape)}%</b></label></div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" />
          {/* 年份刻度 */}
          {yearTicks.map((tk) => (
            <text key={tk.t} x={sx(tk.t)} y={H - PAD + 16} textAnchor="middle" fontSize="9" fill="var(--text-faint)">{tk.year}</text>
          ))}
          {/* 训练/测试分界 */}
          <line x1={sx(TRAIN_N - 1)} y1={PAD} x2={sx(TRAIN_N - 1)} y2={H - PAD} stroke="var(--border)" strokeDasharray="4 4" />
          <text x={sx(TRAIN_N - 1) + 4} y={PAD + 4} fontSize="10" fill="var(--text-faint)">↓ 预测区</text>

          {/* 训练拟合线 */}
          <polyline points={fitted.map((v, i) => `${sx(i)},${sy(v)}`).join(' ')} fill="none" stroke="var(--orange)" strokeWidth={1.4} opacity={0.6} />
          {/* 真实测试值 */}
          <polyline points={DATA.slice(TRAIN_N).map((v, i) => `${sx(TRAIN_N + i)},${sy(v)}`).join(' ')} fill="none" stroke="var(--text-faint)" strokeWidth={2} />
          {/* 训练真实值 */}
          <polyline points={DATA.slice(0, TRAIN_N).map((v, i) => `${sx(i)},${sy(v)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={2} />
          {/* 预测 */}
          <polyline points={predicted.map((v, i) => `${sx(TRAIN_N + i)},${sy(v)}`).join(' ')} fill="none" stroke="var(--pink)" strokeWidth={2.5} strokeDasharray="6 4" />
        </svg>
        <Legend items={[
          { color: 'var(--accent)', label: '训练数据(前10年)' },
          { color: 'var(--orange)', label: '训练段拟合' },
          { color: 'var(--text-faint)', label: '真实测试值' },
          { color: 'var(--pink)', label: '模型预测' },
        ]} />
      </Card>

      <div className="grid-2">
        <Card title="① 趋势：长期增长" sub="线性拟合捕捉总体上升">
          <div style={{ textAlign: 'center', padding: '6px 0' }}>
            <div className="formula" style={{ fontSize: 22, color: 'var(--accent)' }}>+{f2(model.slope)} 千/月</div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>平均每月增长约 {f2(model.slope)} 千人次</div>
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 6 }}>
            十年间客流从 ~120 涨到 ~430，反映战后航空业的高速扩张。
          </div>
        </Card>
        <Card title="② 季节：月度因子" sub="乘性季节——夏季是淡季的多少倍">
          <svg viewBox="0 0 340 120" style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 6 }}>
            {model.seasonal.map((s, i) => {
              const bh = (s - 0.7) / 0.6 * 90
              const peak = s >= 1.1
              return (
                <g key={i}>
                  <rect x={12 + i * 27} y={105 - bh} width={20} height={bh} fill={peak ? 'var(--orange)' : 'var(--accent)'} opacity={0.8} rx={2} />
                  <text x={22 + i * 27} y={117} textAnchor="middle" fontSize="8" fill="var(--text-faint)">{i + 1}</text>
                </g>
              )
            })}
            <line x1={12} y1={105 - (1 - 0.7) / 0.6 * 90} x2={336} y2={105 - (1 - 0.7) / 0.6 * 90} stroke="var(--text-faint)" strokeDasharray="3 3" />
          </svg>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>
            橙色 7、8 月（暑期）客流是平均的 {f2(Math.max(...model.seasonal))}×，11 月最低。虚线=均值 1.0。
          </div>
        </Card>
      </div>

      <div className="grid-2">
        <Card title="为什么用「乘性」季节？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>看图：早期季节波动小，后期波动越来越大——波动幅度<strong>随整体水平按比例放大</strong>。这是<strong>乘性季节</strong>：实际 = 趋势 × 季节因子。</p>
            <p>如果波动幅度恒定（不随水平变化），才用<strong>加性季节</strong>：实际 = 趋势 + 季节。选错模型会系统性偏差。</p>
            <p>经典处理：对乘性序列<strong>取对数</strong>，就转成了加性，可直接用 SARIMA 等线性模型。</p>
          </div>
        </Card>
        <Card title="进阶方法与框架">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>这里用的是最朴素的「趋势×季节」分解预测，MAPE 已不错。工业级方法：</p>
            <p><strong>SARIMA：</strong> <code>statsmodels</code> 的 <code>SARIMAX(order, seasonal_order)</code>，专治带季节的时序。</p>
            <p><strong>Holt-Winters：</strong> <code>ExponentialSmoothing(trend='add', seasonal='mul', seasonal_periods=12)</code>，正是为这类数据设计。</p>
            <p><strong>Prophet：</strong> 自动处理趋势+季节+节假日，调参友好。</p>
          </div>
        </Card>
      </div>

      <Callout type="warn">
        <b>时序铁律（再次强调）：</b> 测试集必须是训练集<strong>之后</strong>的时间段（这里留最后 24 个月），绝不能随机打乱。MAPE（平均绝对百分比误差）是时序最常用的评估指标，越小越好，{f2(mape)}% 意味着预测平均偏离真实值约 {f2(mape)}%。
      </Callout>
    </div>
  )
}
