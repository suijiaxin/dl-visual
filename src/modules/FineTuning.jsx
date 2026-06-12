import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Formula, Slider, Legend } from '../components/ui.jsx'

// 微调方法：全量 vs LoRA vs QLoRA。
// 核心可视化：LoRA 用两个小矩阵 A、B 的乘积 (低秩) 来近似权重更新 ΔW，
// 可调 rank r，实时看可训练参数量从"百万级"骤降到"千级"。

export default function FineTuning() {
  const [dim, setDim] = useState(1024) // 权重矩阵维度 d×d
  const [rank, setRank] = useState(8) // LoRA 秩 r
  const [method, setMethod] = useState('lora')

  const stats = useMemo(() => {
    const full = dim * dim // 全量：整个 ΔW
    const lora = 2 * dim * rank // LoRA：A(d×r) + B(r×d)
    const ratio = (lora / full) * 100
    return { full, lora, ratio }
  }, [dim, rank])

  const fmt = (n) => {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
    return n.toString()
  }

  return (
    <div>
      <PageHeader
        eyebrow="09 · FINE-TUNING"
        title="微调方法：用小代价让大模型学会你的任务"
        lead="全量微调要更新模型全部几十亿参数，显存和算力都吃不消，普通团队玩不起。LoRA 是当前最流行的解法：冻结原模型，只训练两个「小补丁」矩阵。下面用真实的矩阵维度算给你看，参数量能省到什么程度。"
      />

      <div className="btn-row" style={{ marginBottom: 16 }}>
        <button className={`btn ${method === 'full' ? '' : 'secondary'}`} onClick={() => setMethod('full')}>全量微调 Full</button>
        <button className={`btn ${method === 'lora' ? '' : 'secondary'}`} onClick={() => setMethod('lora')}>LoRA</button>
        <button className={`btn ${method === 'qlora' ? '' : 'secondary'}`} onClick={() => setMethod('qlora')}>QLoRA</button>
      </div>

      <Card title="LoRA 的核心思想：低秩分解" sub="不直接改 W，而是给它加一个用两个瘦长矩阵拼出来的「增量」ΔW = B·A">
        <Formula>W_new = W (冻结) + ΔW　　其中 ΔW = B·A，A 是 r×d，B 是 d×r，r ≪ d</Formula>

        <div className="controls">
          <Slider label="权重矩阵维度 d" value={dim} min={256} max={4096} step={256} onChange={setDim} fmt={(v) => v} />
          <Slider label="LoRA 秩 r" value={rank} min={1} max={64} step={1} onChange={setRank} fmt={(v) => v} />
        </div>

        {/* 矩阵尺寸示意 */}
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center', margin: '20px 0' }}>
          <MatrixBox w={120} h={120} label={`W (冻结)`} sub={`${dim}×${dim}`} color="var(--text-faint)" frozen />
          <div style={{ fontSize: 24, color: 'var(--text-faint)' }}>+</div>
          <MatrixBox w={120} h={28} label="B" sub={`${dim}×${rank}`} color="var(--accent-2)" />
          <div style={{ fontSize: 24, color: 'var(--text-faint)' }}>×</div>
          <MatrixBox w={28} h={120} label="A" sub={`${rank}×${dim}`} color="var(--green)" />
          <div style={{ fontSize: 20, color: 'var(--text-faint)' }}>=</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>只训练 A、B</div>
            <div style={{ fontSize: 11, color: 'var(--text-faint)' }}>两个小矩阵</div>
          </div>
        </div>
        <Legend items={[
          { color: 'var(--text-faint)', label: 'W 原始权重（冻结，不训练）' },
          { color: 'var(--accent-2)', label: 'B（可训练）' },
          { color: 'var(--green)', label: 'A（可训练）' },
        ]} />
      </Card>

      <Card title="参数量对比（单个权重矩阵）" sub="拖上面的滑块看变化。真实模型有成百上千个这样的矩阵">
        <div className="grid-3">
          <div className="stat" style={{ borderColor: 'var(--red)' }}>
            <div className="stat-val" style={{ color: 'var(--red)' }}>{fmt(stats.full)}</div>
            <div className="stat-label">全量微调：要训练的参数（d×d）</div>
          </div>
          <div className="stat" style={{ borderColor: 'var(--green)' }}>
            <div className="stat-val" style={{ color: 'var(--green)' }}>{fmt(stats.lora)}</div>
            <div className="stat-label">LoRA：要训练的参数（2×d×r）</div>
          </div>
          <div className="stat" style={{ borderColor: 'var(--accent)' }}>
            <div className="stat-val" style={{ color: 'var(--accent)' }}>{stats.ratio.toFixed(2)}%</div>
            <div className="stat-label">LoRA 只需训练这么点比例</div>
          </div>
        </div>
        <div style={{ marginTop: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <span style={{ width: 70, fontSize: 12, color: 'var(--text-dim)' }}>全量</span>
            <div style={{ flex: 1, height: 22, background: 'var(--bg-elev)', borderRadius: 5, overflow: 'hidden' }}>
              <div style={{ width: '100%', height: '100%', background: 'var(--red)' }} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ width: 70, fontSize: 12, color: 'var(--text-dim)' }}>LoRA</span>
            <div style={{ flex: 1, height: 22, background: 'var(--bg-elev)', borderRadius: 5, overflow: 'hidden' }}>
              <div style={{ width: `${Math.max(stats.ratio, 0.3)}%`, height: '100%', background: 'var(--green)' }} />
            </div>
          </div>
        </div>
        <Callout type="key">
          <b>为什么能这么省？</b> 研究发现：微调时权重的"改变量 ΔW"其实是<b>低秩</b>的——
          它的有效信息能用一个很瘦的矩阵表达。所以不必更新整个 d×d，用 r=8 这样的小秩就够了。
          rank 越大，表达能力越强但参数越多；r=8~32 是常见甜点。
        </Callout>
      </Card>

      <div className="grid-3">
        <Card title="全量微调 Full">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p>更新模型<strong>所有</strong>参数。</p>
            <p>✅ 效果上限最高<br />❌ 显存/算力需求巨大，每个任务都要存一份完整模型副本</p>
            <p style={{ color: 'var(--text-faint)' }}>适合资源充足、追求极致效果的场景</p>
          </div>
        </Card>
        <Card title="LoRA">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p>冻结原模型，只训练 A、B 小矩阵。</p>
            <p>✅ 可训练参数降到 &lt;1%，显存大降<br />✅ 一个 base 模型 + 多个小 LoRA 适配器，按需切换<br />❌ 极限效果略逊全量</p>
            <p style={{ color: 'var(--text-faint)' }}>企业落地最常用</p>
          </div>
        </Card>
        <Card title="QLoRA">
          <div className="prose" style={{ fontSize: 12.5 }}>
            <p>= 4-bit <strong>量化</strong>的 base 模型 + LoRA。</p>
            <p>✅ 把冻结的底座压成 4bit，显存再砍一大截<br />✅ 单张消费级显卡就能微调几十亿参数模型<br />❌ 量化带来轻微精度损失</p>
            <p style={{ color: 'var(--text-faint)' }}>资源受限时的利器</p>
          </div>
        </Card>
      </div>

      {method === 'qlora' && (
        <Callout type="warn">
          <b>QLoRA 的「量化」是什么：</b> 把模型权重从 16 位浮点压缩成 4 位整数存储，显存占用降到约 1/4。
          底座量化后冻结、只在其上训练 LoRA 小矩阵。这让"在一张 24GB 显卡上微调 30B 模型"成为可能——
          量化这个概念在推理部署里也极其重要（模型权重和 KV cache 都能量化）。
        </Callout>
      )}

      <Callout type="key">
        <b>PM 决策视角：</b> 要不要微调？先试 Prompt 工程和 RAG（检索增强），不行再考虑微调。
        要微调，优先 LoRA/QLoRA——成本低、可维护多个适配器。全量微调留给少数有充足算力、且对效果有极致要求的核心场景。
        微调用的"指令-回答"数据，对应的就是上一模块的 <b>SFT 阶段</b>。
      </Callout>
    </div>
  )
}

function MatrixBox({ w, h, label, sub, color, frozen }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        width: w, height: h, background: frozen ? 'var(--bg-elev)' : `${color}22`,
        border: `1.5px solid ${color}`, borderRadius: 6, margin: '0 auto',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 16, fontWeight: 700, color, position: 'relative',
      }}>
        {label}
        {frozen && <span style={{ position: 'absolute', top: 2, right: 4, fontSize: 12 }}>🔒</span>}
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-faint)', marginTop: 4, fontFamily: 'var(--mono)' }}>{sub}</div>
    </div>
  )
}
