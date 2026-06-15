import React, { useState, useMemo, useEffect } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { kmeansStep, mean, std } from '../../lib/mathx.js'

// 商场客户分群：K-Means 最经典的商业实战案例（Mall Customer Segmentation）。
// 用「年收入 × 消费分数」对客户分群，真实迭代收敛，给每个群贴营销标签。

// 真实风格子集：[年收入(k$), 消费分数(1-100)]，呈现经典的 5 簇结构
const CUSTOMERS = [
  // 低收入低消费（节俭型）
  [15, 39], [16, 6], [17, 40], [18, 6], [19, 3], [20, 14], [21, 35], [23, 29], [24, 35], [25, 5],
  // 低收入高消费（冲动型）
  [16, 77], [17, 76], [18, 94], [19, 72], [20, 99], [21, 66], [23, 98], [24, 73], [25, 73], [27, 89],
  // 中收入中消费（普通型）
  [40, 55], [42, 49], [43, 56], [44, 51], [46, 50], [48, 59], [50, 48], [52, 54], [54, 53], [55, 47],
  // 高收入低消费（谨慎型）
  [70, 29], [71, 35], [73, 5], [74, 10], [75, 5], [77, 12], [78, 22], [85, 26], [88, 13], [93, 14],
  // 高收入高消费（目标型）
  [69, 91], [70, 77], [71, 95], [73, 88], [75, 93], [78, 76], [79, 83], [86, 95], [89, 75], [98, 88],
]

const COLORS = ['var(--accent)', 'var(--orange)', 'var(--green)', 'var(--pink)', 'var(--cyan)']
const W = 480, H = 400, PAD = 48

// 标准化后的点（K-Means 对尺度敏感）
function standardize() {
  const inc = CUSTOMERS.map((c) => c[0]), spend = CUSTOMERS.map((c) => c[1])
  const mu = [mean(inc), mean(spend)], sd = [std(inc), std(spend)]
  return { pts: CUSTOMERS.map((c) => [(c[0] - mu[0]) / sd[0], (c[1] - mu[1]) / sd[1]]), mu, sd }
}

// 按 k 跑到收敛（标准化空间），种子化初始质心保证可复现
function runKMeans(k, pts) {
  // 用前 k 个分散的点做初始质心（确定性）
  const step = Math.floor(pts.length / k)
  let centroids = Array.from({ length: k }, (_, i) => [...pts[i * step]])
  let prev = Infinity, assign = null
  for (let it = 0; it < 50; it++) {
    const r = kmeansStep(pts, centroids)
    assign = r.assign
    if (Math.abs(prev - r.inertia) < 1e-9) break
    prev = r.inertia
    centroids = r.newCentroids
  }
  return { assign, centroids, inertia: prev }
}

const SEGMENT_LABELS = {
  // 根据质心在「收入,消费」原始空间的位置贴标签
  hiInc_hiSpend: { name: '🎯 目标客户', desc: '高收入高消费，VIP，重点维护、专属权益', color: 'var(--green)' },
  hiInc_loSpend: { name: '💼 谨慎型', desc: '高收入低消费，有潜力，用优质商品和信任建立唤醒', color: 'var(--accent)' },
  loInc_hiSpend: { name: '⚡ 冲动型', desc: '低收入高消费，对促销敏感，主推性价比和限时折扣', color: 'var(--orange)' },
  loInc_loSpend: { name: '🪙 节俭型', desc: '低收入低消费，价格敏感，靠基础刚需和会员积分维系', color: 'var(--pink)' },
  mid: { name: '🛒 普通型', desc: '中等收入消费，主力人群，标准化运营、培养忠诚度', color: 'var(--cyan)' },
}

function labelSegment(incScore, spendScore) {
  const hiInc = incScore > 55, loInc = incScore < 40
  const hiSpend = spendScore > 60, loSpend = spendScore < 40
  if (hiInc && hiSpend) return SEGMENT_LABELS.hiInc_hiSpend
  if (hiInc && loSpend) return SEGMENT_LABELS.hiInc_loSpend
  if (loInc && hiSpend) return SEGMENT_LABELS.loInc_hiSpend
  if (loInc && loSpend) return SEGMENT_LABELS.loInc_loSpend
  return SEGMENT_LABELS.mid
}

export default function MallCustomers() {
  const [k, setK] = useState(5)
  const { pts, mu, sd } = useMemo(() => standardize(), [])
  const result = useMemo(() => runKMeans(k, pts), [k, pts])

  // 质心还原到原始空间
  const centroidsOrig = result.centroids.map((c) => [c[0] * sd[0] + mu[0], c[1] * sd[1] + mu[1]])

  // 肘部法
  const elbow = useMemo(() => {
    const out = []
    for (let kk = 1; kk <= 8; kk++) out.push({ k: kk, inertia: runKMeans(kk, pts).inertia })
    return out
  }, [pts])
  const maxInertia = Math.max(...elbow.map((e) => e.inertia))

  // 坐标映射（原始空间：收入 0-100，消费 0-100）
  const sx = (inc) => PAD + (inc / 110) * (W - 2 * PAD)
  const sy = (sp) => H - PAD - (sp / 105) * (H - 2 * PAD)

  // 每个簇的标签 + 人数
  const segInfo = useMemo(() => {
    return centroidsOrig.map((c, i) => {
      const n = result.assign.filter((a) => a === i).length
      return { ...labelSegment(c[0], c[1]), n, centroid: c, idx: i }
    })
  }, [result, k])

  return (
    <div>
      <PageHeader
        eyebrow="15 · 实战 · 客户分群"
        title="商场客户分群：K-Means 最经典的商业落地"
        lead="无监督学习最广为人知的实战：商场会员数据里没有「客户类型」标签，但运营想做精细化营销。用「年收入 × 消费分数」两个维度跑 K-Means，机器自动把客户分成若干群，再给每群贴上营销标签——目标客户、冲动型、谨慎型……这是 Mall Customer Segmentation 数据集的经典玩法。"
      />

      <Card title="客户分群结果" sub="× 是各群中心；颜色 = 所属客户群。调 k 看分群粒度变化">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="分群数 k" value={k} min={2} max={6} step={1} onChange={setK} fmt={(v) => v} />
          <div className="control"><label><span>客户总数</span><b>{CUSTOMERS.length} 人</b></label></div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          {/* 坐标轴 */}
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" />
          <text x={W - PAD} y={H - PAD + 22} textAnchor="end" fontSize="11" fill="var(--text-faint)">年收入 (k$) →</text>
          <text x={PAD - 10} y={PAD - 10} fontSize="11" fill="var(--text-faint)">消费分数 ↑</text>
          {/* 象限参考线 */}
          <line x1={sx(50)} y1={PAD} x2={sx(50)} y2={H - PAD} stroke="var(--border-soft)" strokeDasharray="3 3" />
          <line x1={PAD} y1={sy(50)} x2={W - PAD} y2={sy(50)} stroke="var(--border-soft)" strokeDasharray="3 3" />

          {/* 客户点 */}
          {CUSTOMERS.map((c, i) => {
            const cl = result.assign[i]
            return <circle key={i} cx={sx(c[0])} cy={sy(c[1])} r={5.5} fill={COLORS[cl % COLORS.length]} opacity={0.8} stroke="var(--bg)" strokeWidth={1.5} />
          })}
          {/* 质心 */}
          {centroidsOrig.map((c, i) => (
            <g key={i}>
              <line x1={sx(c[0]) - 9} y1={sy(c[1]) - 9} x2={sx(c[0]) + 9} y2={sy(c[1]) + 9} stroke={COLORS[i % COLORS.length]} strokeWidth={4} />
              <line x1={sx(c[0]) - 9} y1={sy(c[1]) + 9} x2={sx(c[0]) + 9} y2={sy(c[1]) - 9} stroke={COLORS[i % COLORS.length]} strokeWidth={4} />
              <circle cx={sx(c[0])} cy={sy(c[1])} r={15} fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth={2} />
            </g>
          ))}
        </svg>
        <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 6 }}>
          虚线把图分成四象限：右上=高收入高消费，左上=低收入高消费，依此类推。
        </div>
      </Card>

      <Card title="各客户群的营销画像" sub={`当前分成 ${k} 群，每群一个运营策略`}>
        <div className="grid-2" style={{ gap: 12 }}>
          {segInfo.map((s) => (
            <div key={s.idx} style={{ padding: '12px 14px', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-elev)', borderLeft: `3px solid ${COLORS[s.idx % COLORS.length]}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                <span style={{ fontSize: 13.5, fontWeight: 600, color: COLORS[s.idx % COLORS.length] }}>{s.name}</span>
                <span style={{ fontSize: 11.5, color: 'var(--text-faint)' }}>{s.n} 人 · 收入≈{Math.round(s.centroid[0])}k · 消费≈{Math.round(s.centroid[1])}</span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4 }}>{s.desc}</div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid-2">
        <Card title="肘部法：该分几群？" sub="点任意点切换 k；拐点 k=5 是这份数据的自然分群数">
          <svg viewBox="0 0 360 200" style={{ width: '100%', background: 'var(--bg-elev)', borderRadius: 8 }}>
            <polyline points={elbow.map((e) => `${30 + ((e.k - 1) / 7) * 300},${180 - (e.inertia / maxInertia) * 150}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
            {elbow.map((e, i) => {
              const x = 30 + ((e.k - 1) / 7) * 300, y = 180 - (e.inertia / maxInertia) * 150
              const isElbow = e.k === 5
              return (
                <g key={i}>
                  <circle cx={x} cy={y} r={isElbow ? 7 : 4.5} fill={isElbow ? 'var(--green)' : 'var(--accent)'} stroke="var(--bg)" strokeWidth={2} style={{ cursor: 'pointer' }} onClick={() => setK(Math.min(6, e.k))} />
                  <text x={x} y={195} textAnchor="middle" fontSize="10" fill="var(--text-faint)">{e.k}</text>
                </g>
              )
            })}
            <text x={30} y={14} fontSize="10" fill="var(--text-faint)">簇内平方和（惯性）</text>
          </svg>
        </Card>
        <Card title="从分群到行动 & 框架">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>聚类的价值不在「分出了几类」，而在<strong>每一类该怎么对待</strong>：右上角目标客户给 VIP 权益，左上角冲动型推限时折扣，右下角谨慎型靠信任唤醒。</p>
            <p>实战要点：① 跑前必须<strong>标准化</strong>（收入和消费分尺度不同）；② 用肘部法/轮廓系数定 k；③ 用 <code>k-means++</code> 初始化避免局部最优。</p>
            <p className="formula" style={{ fontSize: 11.5, textAlign: 'left' }}>
{`KMeans(n_clusters=5, init='k-means++',
       n_init=10).fit(StandardScaler()
       .fit_transform(X))`}
            </p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>为什么是经典案例：</b> 它把抽象的「无监督聚类」直接对应到人人能懂的「客户分层运营」，输入只有两维、结果肉眼可验证，是理解 K-Means 商业价值的最佳入口。真实项目里维度更多（RFM：最近消费、频次、金额），但思路完全一致。
      </Callout>
    </div>
  )
}
