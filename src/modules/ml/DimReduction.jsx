import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { pca2d, mulberry32 } from '../../lib/mathx.js'

// PCA：2D 点云算主成分方向，投影到第一主成分（1D），看方差解释比例。
// 调相关性强度，看主成分方向与解释比例怎么变。

const W = 440, H = 380, PAD = 40
const RANGE = 8 // [-4, 4]

// 生成相关的 2D 数据：corr 控制 x,y 相关性
function genPoints(corr) {
  const rng = mulberry32(23)
  const pts = []
  for (let i = 0; i < 50; i++) {
    const t = (rng() - 0.5) * 6 // 沿主方向
    const n = (rng() - 0.5) * 2.5 // 噪声
    const x = t
    const y = corr * t + (1 - Math.abs(corr)) * n * 1.5
    pts.push([x, y])
  }
  return pts
}

export default function DimReduction() {
  const [corr, setCorr] = useState(0.8)
  const pts = useMemo(() => genPoints(corr), [corr])
  const pca = useMemo(() => pca2d(pts), [pts])

  const sx = (x) => PAD + ((x + RANGE / 2) / RANGE) * (W - 2 * PAD)
  const sy = (y) => H - PAD - ((y + RANGE / 2) / RANGE) * (H - 2 * PAD)

  const { mean: m, v1, v2, ratio } = pca

  // 把点投影到第一主成分上的坐标
  const projected = useMemo(() => {
    return pts.map((p) => {
      const dx = p[0] - m[0], dy = p[1] - m[1]
      const t = dx * v1[0] + dy * v1[1] // 在 v1 上的投影长度
      return { orig: p, proj: [m[0] + t * v1[0], m[1] + t * v1[1]], t }
    })
  }, [pts, pca])

  // 主成分轴线段（延伸）
  const axisLine = (v, len) => [
    [m[0] - v[0] * len, m[1] - v[1] * len],
    [m[0] + v[0] * len, m[1] + v[1] * len],
  ]
  const pc1 = axisLine(v1, 4.5)
  const pc2 = axisLine(v2, 4.5 * Math.sqrt(Math.max(0.02, pca.l2 / pca.l1)))

  return (
    <div>
      <PageHeader
        eyebrow="08 · PCA 降维"
        title="PCA：找到数据方差最大的方向，压缩维度"
        lead="高维数据难可视化、还可能维度冗余。PCA 的洞察：数据往往集中在少数几个方向上，沿这些「主成分」方向，信息（方差）最多。把数据投影到前几个主成分，就能用更少的维度保留大部分信息。下面是 2D→1D 的最小例子：调相关性，看第一主成分方向和它能解释多少方差。"
      />

      <Card title="主成分方向与投影" sub="长轴 PC1 = 方差最大方向；把点投到 PC1（灰色虚线）就完成 2D→1D 降维">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="特征相关性" value={corr} min={0} max={0.98} step={0.02} onChange={setCorr} fmt={f2} />
          <div className="control">
            <label><span>PC1 方差解释比例</span><b style={{ color: ratio > 0.8 ? 'var(--green)' : 'var(--orange)' }}>{Math.round(ratio * 100)}%</b></label>
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          {/* 坐标轴 */}
          <line x1={sx(-RANGE / 2)} y1={sy(0)} x2={sx(RANGE / 2)} y2={sy(0)} stroke="var(--border)" strokeDasharray="3 3" />
          <line x1={sx(0)} y1={sy(-RANGE / 2)} x2={sx(0)} y2={sy(RANGE / 2)} stroke="var(--border)" strokeDasharray="3 3" />

          {/* 投影连线 */}
          {projected.map((p, i) => (
            <line key={`pl${i}`} x1={sx(p.orig[0])} y1={sy(p.orig[1])} x2={sx(p.proj[0])} y2={sy(p.proj[1])}
              stroke="var(--text-faint)" strokeWidth={1} opacity={0.4} />
          ))}

          {/* PC2 次轴 */}
          <line x1={sx(pc2[0][0])} y1={sy(pc2[0][1])} x2={sx(pc2[1][0])} y2={sy(pc2[1][1])} stroke="var(--orange)" strokeWidth={2} opacity={0.7} />
          {/* PC1 主轴 */}
          <line x1={sx(pc1[0][0])} y1={sy(pc1[0][1])} x2={sx(pc1[1][0])} y2={sy(pc1[1][1])} stroke="var(--accent)" strokeWidth={3} />

          {/* 原始点 */}
          {pts.map((p, i) => (
            <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={5} fill="var(--cyan)" opacity={0.7} stroke="var(--bg)" strokeWidth={1.2} />
          ))}
          {/* 投影后的点（落在 PC1 上） */}
          {projected.map((p, i) => (
            <circle key={`pp${i}`} cx={sx(p.proj[0])} cy={sy(p.proj[1])} r={3.5} fill="var(--accent)" />
          ))}

          {/* 标注 */}
          <text x={sx(pc1[1][0])} y={sy(pc1[1][1]) - 6} fontSize="11" fill="var(--accent)" fontWeight="700">PC1</text>
          <text x={sx(pc2[1][0]) + 4} y={sy(pc2[1][1])} fontSize="11" fill="var(--orange)">PC2</text>
        </svg>

        <Legend items={[
          { color: 'var(--cyan)', label: '原始 2D 点' },
          { color: 'var(--accent)', label: 'PC1 主成分 + 投影点' },
          { color: 'var(--orange)', label: 'PC2 次成分' },
        ]} />
      </Card>

      <div className="grid-2">
        <Card title="方差解释比例说明了什么？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>相关性拖到 0.9 以上：点几乎排成一条线，PC1 就能解释 90%+ 的方差——说明<strong>只保留 PC1（降到 1 维）几乎不丢信息</strong>。</p>
            <p>相关性拖到 0：点云是个圆，两个方向方差差不多，PC1 只解释约 50%——这时降维会丢一半信息，不划算。</p>
            <p>所以 PCA 适合<strong>特征间高度相关、有冗余</strong>的场景。实践中看「累计方差解释比例」达到 90~95% 需要几个主成分，就保留几个。</p>
          </div>
        </Card>
        <Card title="用途与框架">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>① 可视化：</strong> 把几十维数据降到 2~3 维画出来看分布。</p>
            <p><strong>② 去冗余/降噪：</strong> 丢掉方差极小的方向（多半是噪声），加速后续模型、缓解维度灾难。</p>
            <p><code>from sklearn.decomposition import PCA</code>；<code>PCA(n_components=2).fit_transform(X)</code>；<code>explained_variance_ratio_</code> 就是这里的解释比例。</p>
            <p>注意 PCA 是<strong>线性</strong>降维；非线性结构用 t-SNE / UMAP 做可视化效果更好（但不可逆、不能用于建模特征）。</p>
          </div>
        </Card>
      </div>

      <Callout type="warn">
        <b>务必先标准化：</b> PCA 基于方差，量纲大的特征会霸占主成分。用前先 <code>StandardScaler</code> 把每个特征缩放到同一尺度，否则结果由单位制决定而非真实结构。
      </Callout>
    </div>
  )
}
