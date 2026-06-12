import React, { useState, useEffect, useRef, useMemo } from 'react'
import { PageHeader, Card, Callout, Formula, Legend } from '../components/ui.jsx'

// 推理流程：prefill + 自回归 decode + KV cache。
// 用动画演示「无缓存每步重算 O(n²)」vs「有缓存每步只算新 token O(n)」的差距。

const PROMPT = ['请', '介绍', '一下', '注意力']
const GEN = ['机制', '是', 'Transformer', '的', '核心'] // 模拟生成的词

export default function Inference() {
  const [useCache, setUseCache] = useState(true)
  const [phase, setPhase] = useState('idle') // idle | prefill | decode | done
  const [genCount, setGenCount] = useState(0) // 已生成多少个词
  const [running, setRunning] = useState(false)
  const timer = useRef(null)

  const promptLen = PROMPT.length
  const totalSeq = promptLen + genCount
  const allTokens = [...PROMPT, ...GEN.slice(0, genCount)]

  // 计算量统计
  const stats = useMemo(() => {
    // prefill: 一次性处理 promptLen 个 token，注意力 ~ promptLen²
    let withCache = promptLen * promptLen // prefill
    let without = promptLen * promptLen
    for (let s = 1; s <= GEN.length; s++) {
      const seqLen = promptLen + s
      // 有 cache：只为新 token 算它对前面所有 key 的注意力 ~ seqLen
      withCache += seqLen
      // 无 cache：每步把整个序列重新跑一遍 ~ seqLen²
      without += seqLen * seqLen
    }
    return { withCache, without }
  }, [promptLen])

  const reset = () => {
    clearTimeout(timer.current)
    setRunning(false)
    setPhase('idle')
    setGenCount(0)
  }

  const start = () => {
    reset()
    setRunning(true)
    setPhase('prefill')
    // prefill 阶段
    timer.current = setTimeout(() => {
      setPhase('decode')
      stepDecode(0)
    }, 1100)
  }

  const stepDecode = (i) => {
    if (i >= GEN.length) {
      setPhase('done')
      setRunning(false)
      return
    }
    setGenCount(i + 1)
    timer.current = setTimeout(() => stepDecode(i + 1), 850)
  }

  useEffect(() => () => clearTimeout(timer.current), [])

  // KV cache 可视化：每个 token 有 K 和 V 两个槽
  const cacheSlots = useCache ? totalSeq : 0

  return (
    <div>
      <PageHeader
        eyebrow="10 · INFERENCE"
        title="推理：Prefill 与 KV Cache"
        lead="训练完的模型部署后，要一个词一个词地往外生成（自回归）。这里有个核心矛盾：每生成一个新词，注意力都要用到前面所有词的 Key/Value。如果每步都重算，计算量是序列长度的平方。KV Cache 把已算好的 K/V 存起来复用，是 LLM 推理最重要的优化。点播放看全过程。"
      />

      <Card title="自回归生成动画" sub="Prefill 一次性处理提示词 → Decode 逐个生成，每个新词追加到 KV Cache">
        <div className="btn-row" style={{ marginBottom: 16 }}>
          <button className="btn" onClick={start} disabled={running}>▶ 播放生成过程</button>
          <button className="btn secondary" onClick={reset}>↻ 重置</button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-dim)', cursor: 'pointer' }}>
            <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
            启用 KV Cache
          </label>
          <span className="tag" style={{
            color: phase === 'prefill' ? 'var(--yellow)' : phase === 'decode' ? 'var(--green)' : 'var(--text-dim)',
          }}>
            阶段: {phase === 'idle' ? '待机' : phase === 'prefill' ? 'PREFILL 预填充' : phase === 'decode' ? 'DECODE 解码' : '完成'}
          </span>
        </div>

        {/* token 序列 */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 20 }}>
          {allTokens.map((t, i) => {
            const isPrompt = i < promptLen
            const isNewest = i === totalSeq - 1 && phase === 'decode'
            return (
              <div key={i} style={{
                padding: '8px 12px',
                borderRadius: 8,
                background: isPrompt ? 'rgba(255,210,63,0.15)' : isNewest ? 'var(--green)' : 'rgba(61,220,132,0.15)',
                border: `1.5px solid ${isPrompt ? 'var(--yellow)' : 'var(--green)'}`,
                color: isNewest ? '#0b0e14' : 'var(--text)',
                fontSize: 14, fontWeight: 600,
                transition: 'all 0.2s',
              }}>
                {t}
              </div>
            )
          })}
          {phase === 'decode' && <div style={{ padding: '8px 4px', color: 'var(--green)', animation: 'fadeIn 0.5s infinite alternate' }}>▍</div>}
        </div>

        {/* KV Cache 可视化 */}
        <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>
          KV Cache 显存（每个已处理 token 缓存一份 Key + Value）
        </div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', minHeight: 56, padding: 12, background: 'var(--bg-elev)', borderRadius: 8 }}>
          {cacheSlots === 0 && phase !== 'idle' && (
            <div style={{ color: 'var(--red)', fontSize: 13 }}>未启用缓存 —— 每步都要为全部 token 重新计算 K/V（红色警示见下方算力对比）</div>
          )}
          {cacheSlots === 0 && phase === 'idle' && <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>点播放后这里会逐格填满</div>}
          {Array.from({ length: cacheSlots }).map((_, i) => {
            const isPrompt = i < promptLen
            const justAdded = i === cacheSlots - 1 && phase === 'decode'
            return (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <div style={{
                  width: 30, height: 18, borderRadius: 3,
                  background: justAdded ? 'var(--green)' : isPrompt ? 'rgba(255,210,63,0.5)' : 'rgba(61,220,132,0.45)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 8, color: '#fff',
                }}>K{i}</div>
                <div style={{
                  width: 30, height: 18, borderRadius: 3,
                  background: justAdded ? 'var(--green)' : isPrompt ? 'rgba(255,210,63,0.5)' : 'rgba(61,220,132,0.45)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 8, color: '#fff',
                }}>V{i}</div>
              </div>
            )
          })}
        </div>
        <Legend items={[
          { color: 'var(--yellow)', label: 'Prompt token（prefill 阶段一次性算完）' },
          { color: 'rgba(61,220,132,0.6)', label: '已生成 token 的缓存' },
          { color: 'var(--green)', label: '本步新增' },
        ]} />
      </Card>

      <div className="grid-2">
        <Card title="Prefill 阶段" sub="处理用户输入的提示词">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>
              用户的整段 prompt <strong>一次性、并行</strong>喂进模型（能充分利用 GPU）。这一步计算出
              prompt 里每个 token 的 K/V 并<strong>存入 cache</strong>，同时产出第一个新 token。
            </p>
            <p>Prefill 是<strong>计算密集型</strong>（compute-bound）：prompt 越长，这步越慢，对应你体感的「首字延迟 TTFT」。</p>
          </div>
        </Card>
        <Card title="Decode 阶段" sub="逐个生成后续 token">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>
              之后每一步<strong>只输入上一个新生成的 token</strong>，算出它的 Q，去和 cache 里所有历史 K/V 做注意力，
              产出下一个词，再把自己的 K/V 追加进 cache。
            </p>
            <p>Decode 是<strong>访存密集型</strong>（memory-bound）：瓶颈在反复读取庞大的 KV cache 和权重，对应「每秒生成 token 数 TPS」。</p>
          </div>
        </Card>
      </div>

      <Card title="算力对比：KV Cache 到底省了多少" sub={`生成 ${GEN.length} 个词、prompt 长 ${promptLen} 的累计注意力计算量（相对值）`}>
        <div className="grid-2">
          <div className="stat">
            <div className="stat-val" style={{ color: 'var(--green)' }}>{stats.withCache}</div>
            <div className="stat-label">✅ 有 KV Cache（每步只算新 token，≈ O(n)）</div>
          </div>
          <div className="stat">
            <div className="stat-val" style={{ color: 'var(--red)' }}>{stats.without}</div>
            <div className="stat-label">❌ 无缓存（每步重算整个序列，≈ O(n²)）</div>
          </div>
        </div>
        <div style={{ marginTop: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <span style={{ width: 70, fontSize: 12, color: 'var(--text-dim)' }}>有缓存</span>
            <div style={{ flex: 1, height: 22, background: 'var(--bg-elev)', borderRadius: 5, overflow: 'hidden' }}>
              <div style={{ width: `${(stats.withCache / stats.without) * 100}%`, height: '100%', background: 'var(--green)' }} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ width: 70, fontSize: 12, color: 'var(--text-dim)' }}>无缓存</span>
            <div style={{ flex: 1, height: 22, background: 'var(--bg-elev)', borderRadius: 5, overflow: 'hidden' }}>
              <div style={{ width: '100%', height: '100%', background: 'var(--red)' }} />
            </div>
          </div>
        </div>
        <Callout type="warn">
          序列越长，差距越夸张（平方 vs 线性）。这就是为什么所有 LLM 推理框架（vLLM、TensorRT-LLM 等）都必做 KV Cache。
          代价是<b>显存</b>：cache 大小 = 2 × 层数 × 注意力头数 × 头维度 × 序列长度 × batch。长上下文场景下，
          KV cache 能吃掉比模型权重还多的显存，于是又催生了 <b>PagedAttention、MQA/GQA、量化 cache</b> 等优化。
        </Callout>
      </Card>

      <Formula>KV Cache 显存 ≈ 2 · L层 · H头 · d头维 · S序列长 · B批量 · 字节数</Formula>

      <Callout type="key">
        <b>全流程闭环：</b> 「架构演变 → 神经元 → 反向传播训练出权重 → CNN/RNN 的局限 → 注意力 → 组装成 Transformer → 部署后用 prefill+KV cache 高效推理」。
        作为大模型训推产品经理，训练侧你关注<b>数据/算力/收敛</b>，推理侧你关注<b>首字延迟(TTFT)、吞吐(TPS)、显存占用、并发</b>——
        而 KV Cache 正是连接「上下文长度」「成本」「速度」这三个产品核心指标的技术枢纽。
      </Callout>
    </div>
  )
}
