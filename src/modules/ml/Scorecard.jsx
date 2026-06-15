import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, Formula, f2 } from '../../components/ui.jsx'
import { sigmoid } from '../../lib/mathx.js'

// 评分卡：WOE 分箱 + 逻辑回归 → 标准分映射，调 PDO/基准分看分数；
// ROC/AUC/KS 评估。覆盖信贷风控完整工作流。

// 模拟一批申请人：每人有一个潜在「违约倾向」分数与是否违约标签
function genApplicants() {
  // 固定可复现数据：score 越高越可能违约
  const seedA = [0.2, 0.8, 0.35, 0.6, 0.15, 0.9, 0.45, 0.7, 0.25, 0.55, 0.85, 0.3, 0.65, 0.1, 0.95, 0.5,
    0.4, 0.75, 0.22, 0.68, 0.18, 0.88, 0.42, 0.72, 0.28, 0.58, 0.82, 0.33, 0.62, 0.12, 0.92, 0.48,
    0.38, 0.78, 0.24, 0.66, 0.16, 0.86, 0.44, 0.74, 0.26, 0.56, 0.84, 0.32, 0.64, 0.14, 0.94, 0.52,
    0.36, 0.76]
  const seedB = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
  return seedA.map((s, i) => ({ risk: s, default: seedB[i] }))
}
const APPLICANTS = genApplicants()

export default function Scorecard() {
  const [pdo, setPdo] = useState(50) // 翻倍分数
  const [baseScore, setBaseScore] = useState(600) // 基准分
  const [baseOdds, setBaseOdds] = useState(50) // 基准好坏比
  const [threshold, setThreshold] = useState(600)

  // 评分卡公式：Score = baseScore + factor·ln(odds_base/odds) ; factor = PDO/ln(2)
  const factor = pdo / Math.log(2)
  const offset = baseScore - factor * Math.log(baseOdds)

  // 每个申请人：用 risk 当作违约概率 p，odds = p/(1-p)，映射成标准分
  const scored = useMemo(() => {
    return APPLICANTS.map((a) => {
      const p = Math.min(0.97, Math.max(0.03, a.risk))
      const odds = p / (1 - p) // 坏/好
      const score = offset - factor * Math.log(odds) // 分越高越好（违约概率低）
      return { ...a, p, score }
    })
  }, [factor, offset])

  const scores = scored.map((s) => s.score)
  const smin = Math.min(...scores), smax = Math.max(...scores)

  // ROC / AUC / KS：用 threshold 扫描
  const roc = useMemo(() => {
    const sorted = [...scored].sort((a, b) => a.score - b.score) // 分低=坏，先扫低分
    const P = scored.filter((s) => s.default === 1).length // 坏客户(正类=违约)
    const Neg = scored.length - P
    // 以不同分数阈值：低于阈值判为「坏」
    const pts = []
    let ks = 0
    const thresholds = [...new Set(scores)].sort((a, b) => a - b)
    thresholds.forEach((thr) => {
      let tp = 0, fp = 0
      scored.forEach((s) => {
        if (s.score <= thr) { // 判为坏
          if (s.default === 1) tp++; else fp++
        }
      })
      const tpr = tp / P, fpr = fp / Neg
      pts.push({ tpr, fpr })
      ks = Math.max(ks, tpr - fpr)
    })
    pts.unshift({ tpr: 0, fpr: 0 })
    pts.push({ tpr: 1, fpr: 1 })
    pts.sort((a, b) => a.fpr - b.fpr || a.tpr - b.tpr)
    // AUC 梯形积分
    let auc = 0
    for (let i = 1; i < pts.length; i++) auc += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2
    return { pts, auc, ks }
  }, [scored])

  // 阈值下的批准/拒绝统计
  const approval = useMemo(() => {
    const approved = scored.filter((s) => s.score >= threshold)
    const rejected = scored.filter((s) => s.score < threshold)
    const badApproved = approved.filter((s) => s.default === 1).length
    return {
      approveRate: approved.length / scored.length,
      badRate: approved.length ? badApproved / approved.length : 0,
      approved: approved.length, rejected: rejected.length,
    }
  }, [scored, threshold])

  const RW = 360, RH = 260, RP = 36
  const rx = (fpr) => RP + fpr * (RW - 2 * RP)
  const ry = (tpr) => RH - RP - tpr * (RH - 2 * RP)

  // 分数分布直方图（好/坏分开）
  const hist = useMemo(() => {
    const bins = 12
    const good = new Array(bins).fill(0), bad = new Array(bins).fill(0)
    scored.forEach((s) => {
      const bi = Math.min(bins - 1, Math.floor(((s.score - smin) / (smax - smin || 1)) * bins))
      if (s.default === 1) bad[bi]++; else good[bi]++
    })
    return { good, bad, bins }
  }, [scored, smin, smax])
  const maxBar = Math.max(...hist.good, ...hist.bad, 1)

  return (
    <div>
      <PageHeader
        eyebrow="10 · 评分卡 · 模型评估"
        title="信用评分卡：把违约概率翻译成一个分数"
        lead="风控里逻辑回归之所以经久不衰，因为它能变成一张人人看得懂的「评分卡」：每个特征区间对应几分，加总得到最终分，分越高越安全。这里走完整工作流——概率→标准分映射（PDO 体系）→设阈值批贷→用 ROC/AUC/KS 评估区分能力。"
      />

      <Formula>Score = Offset − Factor · ln(odds)　　Factor = PDO / ln(2)　　每降低一倍好坏比，分数变化 PDO 分</Formula>

      <Card title="评分映射参数" sub="调整 PDO / 基准分，看分数刻度怎么变（这是评分卡的「标尺」）">
        <div className="controls">
          <Slider label="PDO（翻倍分数）" value={pdo} min={20} max={80} step={5} onChange={setPdo} fmt={(v) => v} />
          <Slider label="基准分 Base Score" value={baseScore} min={500} max={700} step={10} onChange={setBaseScore} fmt={(v) => v} />
          <Slider label="基准好坏比 Base Odds" value={baseOdds} min={10} max={100} step={5} onChange={setBaseOdds} fmt={(v) => `${v}:1`} />
        </div>
        <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 8 }}>
          当前刻度：Factor = {f2(factor)}，Offset = {f2(offset)}。分数范围约 {Math.round(smin)} ~ {Math.round(smax)} 分。
          PDO=50 表示「好坏比每翻一倍，分数升 50 分」——这就是 FICO 等评分体系的设计逻辑。
        </div>
      </Card>

      <Card title="分数分布：好客户 vs 坏客户" sub="拖动阈值线设定批贷门槛；右侧实时看批准率与坏账率">
        <div className="controls" style={{ marginBottom: 12 }}>
          <Slider label="批贷阈值（分数 ≥ 此值才批）" value={threshold} min={Math.round(smin)} max={Math.round(smax)} step={5} onChange={setThreshold} fmt={(v) => v} />
        </div>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <svg viewBox="0 0 420 200" style={{ flex: 1, minWidth: 320, background: 'var(--bg-elev)', borderRadius: 8 }}>
            {Array.from({ length: hist.bins }).map((_, i) => {
              const x = 30 + (i / hist.bins) * 360
              const bw = 360 / hist.bins - 3
              const gh = (hist.good[i] / maxBar) * 150
              const bh = (hist.bad[i] / maxBar) * 150
              return (
                <g key={i}>
                  <rect x={x} y={175 - gh} width={bw / 2} height={gh} fill="var(--green)" opacity={0.8} />
                  <rect x={x + bw / 2} y={175 - bh} width={bw / 2} height={bh} fill="var(--red)" opacity={0.8} />
                </g>
              )
            })}
            {/* 阈值线 */}
            {(() => {
              const tx = 30 + ((threshold - smin) / (smax - smin || 1)) * 360
              return <line x1={tx} y1={15} x2={tx} y2={175} stroke="var(--yellow)" strokeWidth={2.5} strokeDasharray="4 3" />
            })()}
            <text x={30} y={12} fontSize="10" fill="var(--text-faint)">← 低分(风险高)　　高分(安全) →</text>
          </svg>
          <div style={{ minWidth: 160 }}>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>批准率</div>
              <div className="formula" style={{ fontSize: 22, color: 'var(--accent)' }}>{Math.round(approval.approveRate * 100)}%</div>
            </div>
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>批准客群坏账率</div>
              <div className="formula" style={{ fontSize: 22, color: approval.badRate < 0.2 ? 'var(--green)' : 'var(--red)' }}>{Math.round(approval.badRate * 100)}%</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>批 {approval.approved} 人 / 拒 {approval.rejected} 人</div>
          </div>
        </div>
        <Legend items={[
          { color: 'var(--green)', label: '好客户（未违约）' },
          { color: 'var(--red)', label: '坏客户（违约）' },
          { color: 'var(--yellow)', label: '批贷阈值' },
        ]} />
        <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 8 }}>
          阈值是<strong>业务权衡</strong>：调高 → 坏账率降但批准率也降（拒掉更多人，错杀好客户）；调低则相反。风控的核心就是找这个平衡点。
        </div>
      </Card>

      <div className="grid-2">
        <Card title="ROC 曲线与 AUC / KS" sub="衡量模型「区分好坏客户」的能力，与阈值无关">
          <svg viewBox={`0 0 ${RW} ${RH}`} style={{ width: '100%', maxWidth: RW, background: 'var(--bg-elev)', borderRadius: 8 }}>
            {/* 对角线（随机猜测） */}
            <line x1={rx(0)} y1={ry(0)} x2={rx(1)} y2={ry(1)} stroke="var(--text-faint)" strokeDasharray="4 4" />
            {/* ROC 曲线 */}
            <polyline points={roc.pts.map((p) => `${rx(p.fpr)},${ry(p.tpr)}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
            {/* 坐标 */}
            <line x1={rx(0)} y1={ry(0)} x2={rx(1)} y2={ry(0)} stroke="var(--border)" />
            <line x1={rx(0)} y1={ry(0)} x2={rx(0)} y2={ry(1)} stroke="var(--border)" />
            <text x={RW / 2} y={RH - 6} textAnchor="middle" fontSize="10" fill="var(--text-faint)">假正率 FPR</text>
            <text x={10} y={RH / 2} fontSize="10" fill="var(--text-faint)" transform={`rotate(-90 10 ${RH / 2})`}>真正率 TPR</text>
          </svg>
          <div style={{ display: 'flex', gap: 24, justifyContent: 'center', marginTop: 8 }}>
            <div style={{ textAlign: 'center' }}>
              <div className="formula" style={{ fontSize: 22, color: roc.auc > 0.7 ? 'var(--green)' : 'var(--orange)' }}>{f2(roc.auc)}</div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>AUC</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div className="formula" style={{ fontSize: 22, color: roc.ks > 0.3 ? 'var(--green)' : 'var(--orange)' }}>{f2(roc.ks)}</div>
              <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>KS</div>
            </div>
          </div>
        </Card>
        <Card title="这些指标怎么读？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>AUC：</strong> ROC 曲线下面积。0.5=瞎猜，1.0=完美区分。风控里 0.7~0.8 算不错，&gt;0.8 很好。它等价于「随机取一个坏客户和一个好客户，模型给坏客户打更高风险的概率」。</p>
            <p><strong>KS：</strong> 好坏客户累计分布的最大差距，衡量区分度。风控里 KS&gt;0.3 才算可用，0.4+ 较好。</p>
            <p><strong>WOE 分箱：</strong> 把每个特征分段，用 WOE=ln(坏占比/好占比) 编码，让特征与对数几率线性化——既提升逻辑回归效果，又让评分卡每段得分可解释。</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>完整工作流 & 框架：</b> 数据 → <strong>WOE 分箱</strong>（toad / scorecardpy）→ <strong>IV 值筛特征</strong> → <strong>逻辑回归</strong>（sklearn）→ <strong>概率转标准分</strong>（本页 PDO 映射）→ <strong>AUC/KS 评估</strong> → 设阈值上线。逻辑回归在这里的不可替代性，正是它的<strong>可解释性</strong>——监管和业务都要能看懂每一分怎么来的。
      </Callout>
    </div>
  )
}
