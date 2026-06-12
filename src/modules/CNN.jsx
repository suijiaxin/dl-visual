import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, f2 } from '../components/ui.jsx'

// 真实卷积：可编辑的 8×8 输入图，3×3 卷积核滑动，实时算出特征图。
// 点输入格子可画图案，选不同卷积核看它检测什么特征。

const N = 8 // 输入尺寸
const K = 3 // 卷积核尺寸

const KERNELS = {
  edgeV: { label: '竖直边缘', m: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
  edgeH: { label: '水平边缘', m: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
  sharpen: { label: '锐化', m: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  blur: { label: '模糊(均值)', m: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] },
}

// 初始图案：一个对角线 + 方块，便于看出边缘检测效果
function initGrid() {
  const g = Array.from({ length: N }, () => new Array(N).fill(0))
  for (let i = 0; i < N; i++) g[i][i] = 1
  for (let i = 2; i < 5; i++) for (let j = 4; j < 7; j++) g[i][j] = 1
  return g
}

function convolve(grid, kernel) {
  const out = Array.from({ length: N - K + 1 }, () => new Array(N - K + 1).fill(0))
  for (let r = 0; r < N - K + 1; r++)
    for (let c = 0; c < N - K + 1; c++) {
      let s = 0
      for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) s += grid[r + i][c + j] * kernel[i][j]
      out[r][c] = s
    }
  return out
}

export default function CNN() {
  const [grid, setGrid] = useState(initGrid)
  const [kName, setKName] = useState('edgeV')
  const [pos, setPos] = useState({ r: 0, c: 0 }) // 卷积核当前位置

  const kernel = KERNELS[kName].m
  const feat = useMemo(() => convolve(grid, kernel), [grid, kernel])
  const featMax = Math.max(1, ...feat.flat().map(Math.abs))

  // 当前窗口的点乘明细
  const detail = useMemo(() => {
    let s = 0
    const terms = []
    for (let i = 0; i < K; i++)
      for (let j = 0; j < K; j++) {
        const v = grid[pos.r + i][pos.c + j]
        const w = kernel[i][j]
        s += v * w
        terms.push({ v, w })
      }
    return { s, terms }
  }, [grid, kernel, pos])

  const cell = 34
  const toggle = (r, c) => {
    const g = grid.map((row) => [...row])
    g[r][c] = g[r][c] ? 0 : 1
    setGrid(g)
  }

  return (
    <div>
      <PageHeader
        eyebrow="03 · CNN"
        title="卷积网络：用小窗口扫描，捕捉局部特征"
        lead="全连接层把图像每个像素都连到下一层，参数爆炸且无视空间结构。CNN 换个思路：用一个共享的小卷积核在图上滑动，每次只看一个局部窗口做加权求和。同一个核扫全图（参数共享），专门检测某种局部模式（如边缘）。点格子画图，看卷积核怎么响应。"
      />

      <Card title="卷积运算可视化" sub="左：输入图 (点击格子翻转)；中：卷积核；右：输出特征图。黄框是当前滑动窗口">
        <div className="controls">
          <div className="control">
            <label><span>卷积核类型</span></label>
            <select value={kName} onChange={(e) => setKName(e.target.value)}>
              {Object.entries(KERNELS).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </div>
          <div className="control">
            <label><span>清空 / 重置</span></label>
            <div className="btn-row">
              <button className="btn ghost" onClick={() => setGrid(Array.from({ length: N }, () => new Array(N).fill(0)))}>清空</button>
              <button className="btn secondary" onClick={() => setGrid(initGrid())}>重置图案</button>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {/* 输入图 */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>输入图 8×8 (可点击)</div>
            <svg width={N * cell} height={N * cell} style={{ background: 'var(--bg-elev)', borderRadius: 6 }}>
              {grid.map((row, r) =>
                row.map((v, c) => (
                  <rect
                    key={`${r}-${c}`}
                    x={c * cell} y={r * cell} width={cell - 1.5} height={cell - 1.5}
                    fill={v ? 'var(--accent)' : 'var(--slot)'}
                    style={{ cursor: 'pointer' }}
                    onClick={() => toggle(r, c)}
                  />
                ))
              )}
              {/* 滑动窗口高亮 */}
              <rect
                x={pos.c * cell} y={pos.r * cell} width={K * cell - 1.5} height={K * cell - 1.5}
                fill="none" stroke="var(--yellow)" strokeWidth={2.5} pointerEvents="none"
              />
            </svg>
          </div>

          {/* 卷积核 */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>卷积核 3×3</div>
            <svg width={K * cell} height={K * cell} style={{ background: 'var(--bg-elev)', borderRadius: 6 }}>
              {kernel.map((row, i) =>
                row.map((w, j) => (
                  <g key={`${i}-${j}`}>
                    <rect x={j * cell} y={i * cell} width={cell - 1.5} height={cell - 1.5}
                      fill={w > 0 ? 'rgba(61,220,132,0.25)' : w < 0 ? 'rgba(255,92,122,0.25)' : 'var(--slot)'} />
                    <text x={j * cell + cell / 2} y={i * cell + cell / 2 + 4} textAnchor="middle" fontSize="10" fill="var(--text)" fontFamily="var(--mono)">
                      {f2(w)}
                    </text>
                  </g>
                ))
              )}
            </svg>
            <div style={{ marginTop: 14, fontSize: 12, color: 'var(--text-dim)' }}>移动窗口：</div>
            <div className="btn-row" style={{ marginTop: 6 }}>
              <button className="btn ghost" onClick={() => setPos((p) => ({ ...p, c: Math.max(0, p.c - 1) }))}>←</button>
              <button className="btn ghost" onClick={() => setPos((p) => ({ ...p, c: Math.min(N - K, p.c + 1) }))}>→</button>
              <button className="btn ghost" onClick={() => setPos((p) => ({ ...p, r: Math.max(0, p.r - 1) }))}>↑</button>
              <button className="btn ghost" onClick={() => setPos((p) => ({ ...p, r: Math.min(N - K, p.r + 1) }))}>↓</button>
            </div>
          </div>

          {/* 特征图 */}
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 8 }}>输出特征图 6×6</div>
            <svg width={(N - K + 1) * cell} height={(N - K + 1) * cell} style={{ background: 'var(--bg-elev)', borderRadius: 6 }}>
              {feat.map((row, r) =>
                row.map((v, c) => {
                  const t = Math.abs(v) / featMax
                  const fill = v >= 0 ? `rgba(61,220,132,${t})` : `rgba(255,92,122,${t})`
                  const isCur = r === pos.r && c === pos.c
                  return (
                    <g key={`${r}-${c}`}>
                      <rect x={c * cell} y={r * cell} width={cell - 1.5} height={cell - 1.5} fill={fill}
                        stroke={isCur ? 'var(--yellow)' : 'none'} strokeWidth={isCur ? 2.5 : 0}
                        style={{ cursor: 'pointer' }} onClick={() => setPos({ r, c })} />
                      <text x={c * cell + cell / 2} y={r * cell + cell / 2 + 4} textAnchor="middle" fontSize="9" fill="var(--text-dim)" fontFamily="var(--mono)" pointerEvents="none">
                        {f2(v)}
                      </text>
                    </g>
                  )
                })
              )}
            </svg>
          </div>
        </div>

        <Legend items={[
          { color: 'var(--accent)', label: '输入像素=1' },
          { color: 'rgba(61,220,132,0.8)', label: '特征响应为正' },
          { color: 'rgba(255,92,122,0.8)', label: '特征响应为负' },
          { color: 'var(--yellow)', label: '当前窗口' },
        ]} />
      </Card>

      <Card title="当前窗口的点乘明细" sub={`输出特征图位置 (${pos.r}, ${pos.c}) 是这样算出来的`}>
        <div className="formula" style={{ lineHeight: 2 }}>
          {detail.terms.map((t, i) => (
            <span key={i}>
              {i > 0 && ' + '}
              <span style={{ color: 'var(--accent)' }}>{t.v}</span>×
              <span style={{ color: t.w >= 0 ? 'var(--green)' : 'var(--red)' }}>{f2(t.w)}</span>
            </span>
          ))}
          {' = '}
          <span style={{ color: 'var(--yellow)', fontWeight: 700 }}>{f2(detail.s)}</span>
        </div>
        <div style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>
          这就是「卷积」的全部：窗口内逐元素相乘再求和。窗口滑过整张图，就得到完整特征图。
        </div>
      </Card>

      <div className="grid-2">
        <Card title="CNN 的两个关键思想">
          <div className="prose">
            <p><strong>① 局部连接：</strong> 每个输出只看输入的一个小窗口，符合「图像的语义来自局部」这一先验。</p>
            <p><strong>② 参数共享：</strong> 同一个卷积核扫全图，无论特征出现在哪都能检测到，参数量大幅下降。</p>
            <p>真实 CNN 会堆叠多层：浅层学边缘/纹理，深层组合成物体部件乃至完整物体。中间穿插<strong>池化</strong>下采样、缩小尺寸、扩大感受野。</p>
          </div>
        </Card>
        <Card title="为什么语言模型不用 CNN？">
          <div className="prose">
            <p>
              卷积核的感受野是<strong>固定且局部</strong>的——它天生擅长「邻近像素相关」的图像，但语言里
              「这个代词指的是三句话之前的某个名词」这种<strong>长距离依赖</strong>，靠堆很多层卷积才能勉强覆盖，效率低。
            </p>
            <p>处理序列、捕捉任意距离的依赖，是 RNN/LSTM 和后来 Transformer 的主场。下一站 →</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>动手试试：</b> 选「竖直边缘」核，然后清空画一条竖线和一条横线，对比特征图——竖线处响应强烈，横线几乎无响应。这就是一个卷积核「只对特定方向的边缘敏感」的直观体现。
      </Callout>
    </div>
  )
}
