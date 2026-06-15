import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, f2 } from '../../components/ui.jsx'
import { mean, std, dist } from '../../lib/mathx.js'

// 鸢尾花分类：经典入门数据集。真实的 45 行子集（每类 15 朵），
// 4 个特征选 2 个画散点，训练最近质心分类器，看决策区域 + 混淆矩阵。

// 真实鸢尾花数据子集 [萼长, 萼宽, 瓣长, 瓣宽, 类别] 0=setosa 1=versicolor 2=virginica
const IRIS = [
  [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0], [4.6, 3.1, 1.5, 0.2, 0], [5.0, 3.6, 1.4, 0.2, 0],
  [5.4, 3.9, 1.7, 0.4, 0], [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0], [4.4, 2.9, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.1, 0],
  [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0], [4.8, 3.0, 1.4, 0.1, 0], [4.3, 3.0, 1.1, 0.1, 0], [5.8, 4.0, 1.2, 0.2, 0],
  [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1], [5.5, 2.3, 4.0, 1.3, 1], [6.5, 2.8, 4.6, 1.5, 1],
  [5.7, 2.8, 4.5, 1.3, 1], [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1], [6.6, 2.9, 4.6, 1.3, 1], [5.2, 2.7, 3.9, 1.4, 1],
  [5.9, 3.0, 4.2, 1.5, 1], [6.0, 2.2, 4.0, 1.0, 1], [6.1, 2.9, 4.7, 1.4, 1], [5.6, 2.9, 3.6, 1.3, 1], [6.7, 3.1, 4.4, 1.4, 1],
  [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2], [6.3, 2.9, 5.6, 1.8, 2], [6.5, 3.0, 5.8, 2.2, 2],
  [7.6, 3.0, 6.6, 2.1, 2], [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2], [6.7, 2.5, 5.8, 1.8, 2], [7.2, 3.6, 6.1, 2.5, 2],
  [6.5, 3.2, 5.1, 2.0, 2], [6.4, 2.7, 5.3, 1.9, 2], [6.8, 3.0, 5.5, 2.1, 2], [5.7, 2.5, 5.0, 2.0, 2], [5.8, 2.8, 5.1, 2.4, 2],
]

const FEATURES = ['萼片长', '萼片宽', '花瓣长', '花瓣宽']
const SPECIES = ['Setosa', 'Versicolor', 'Virginica']
const COLORS = ['var(--accent)', 'var(--orange)', 'var(--green)']
const W = 440, H = 380, PAD = 44

export default function IrisClassification() {
  const [fx, setFx] = useState(2) // 花瓣长
  const [fy, setFy] = useState(3) // 花瓣宽

  // 训练/测试划分：每类前 10 训练，后 5 测试（按索引确定，可复现）
  const { train, test } = useMemo(() => {
    const train = [], test = []
    IRIS.forEach((row, i) => {
      const within = i % 15 // 每类 15 个
      if (within < 10) train.push(row); else test.push(row)
    })
    return { train, test }
  }, [])

  // 在训练集上算每个特征的标准化参数（防泄漏：只用训练集）
  const scaler = useMemo(() => {
    return FEATURES.map((_, f) => {
      const col = train.map((r) => r[f])
      return { mu: mean(col), sd: std(col) }
    })
  }, [train])
  const z = (row) => row.slice(0, 4).map((v, f) => (v - scaler[f].mu) / scaler[f].sd)

  // 最近质心分类器：在标准化后的 4 维空间算每类质心
  const centroids = useMemo(() => {
    return [0, 1, 2].map((c) => {
      const members = train.filter((r) => r[4] === c).map(z)
      return [0, 1, 2, 3].map((f) => mean(members.map((m) => m[f])))
    })
  }, [train, scaler])

  const classify = (row) => {
    const zr = z(row)
    let best = 0, bestD = Infinity
    centroids.forEach((c, i) => { const d = dist(zr, c); if (d < bestD) { bestD = d; best = i } })
    return best
  }

  // 测试集评估 + 混淆矩阵
  const { acc, confusion } = useMemo(() => {
    const conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    let correct = 0
    test.forEach((r) => {
      const pred = classify(r)
      conf[r[4]][pred]++
      if (pred === r[4]) correct++
    })
    return { acc: correct / test.length, confusion: conf }
  }, [test, centroids])

  // 散点图坐标范围（选中的两个特征）
  const xvals = IRIS.map((r) => r[fx]), yvals = IRIS.map((r) => r[fy])
  const xmin = Math.min(...xvals) - 0.3, xmax = Math.max(...xvals) + 0.3
  const ymin = Math.min(...yvals) - 0.3, ymax = Math.max(...yvals) + 0.3
  const sx = (x) => PAD + ((x - xmin) / (xmax - xmin)) * (W - 2 * PAD)
  const sy = (y) => H - PAD - ((y - ymin) / (ymax - ymin)) * (H - 2 * PAD)

  return (
    <div>
      <PageHeader
        eyebrow="12 · 实战 · 鸢尾花"
        title="鸢尾花分类：机器学习的「Hello World」"
        lead="Fisher 1936 年的鸢尾花数据集：150 朵花（这里取 45 朵子集），4 个测量特征，分 3 个品种。它是检验分类算法的标准入门题。我们走完整流程——划分训练/测试集、在训练集上标准化（防泄漏）、训练最近质心分类器、在测试集上看准确率和混淆矩阵。"
      />

      <Card title="选两个特征看类别分布" sub="点按钮换坐标轴特征。花瓣长×花瓣宽几乎能完美分开三类">
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>X 轴特征</div>
            <div className="btn-row">{FEATURES.map((f, i) => <button key={i} className={`btn ${fx === i ? '' : 'secondary'}`} onClick={() => setFx(i)} style={{ fontSize: 12 }}>{f}</button>)}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>Y 轴特征</div>
            <div className="btn-row">{FEATURES.map((f, i) => <button key={i} className={`btn ${fy === i ? '' : 'secondary'}`} onClick={() => setFy(i)} style={{ fontSize: 12 }}>{f}</button>)}</div>
          </div>
        </div>

        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, background: 'var(--bg-elev)', borderRadius: 8 }}>
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--border)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--border)" />
          <text x={W - PAD} y={H - PAD + 20} textAnchor="end" fontSize="11" fill="var(--text-faint)">{FEATURES[fx]}</text>
          <text x={PAD - 6} y={PAD - 8} fontSize="11" fill="var(--text-faint)">{FEATURES[fy]}</text>
          {IRIS.map((r, i) => {
            const isTest = i % 15 >= 10
            return (
              <circle key={i} cx={sx(r[fx])} cy={sy(r[fy])} r={isTest ? 6 : 5}
                fill={COLORS[r[4]]} opacity={isTest ? 1 : 0.5}
                stroke={isTest ? 'var(--text)' : 'var(--bg)'} strokeWidth={isTest ? 2 : 1.5} />
            )
          })}
        </svg>
        <Legend items={[
          { color: COLORS[0], label: 'Setosa' },
          { color: COLORS[1], label: 'Versicolor' },
          { color: COLORS[2], label: 'Virginica' },
        ]} />
        <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 6 }}>实心描边=测试集，半透明=训练集</div>
      </Card>

      <div className="grid-2">
        <Card title="测试集表现">
          <div style={{ textAlign: 'center', padding: '6px 0' }}>
            <div className="formula" style={{ fontSize: 30, color: acc >= 0.9 ? 'var(--green)' : 'var(--orange)' }}>{Math.round(acc * 100)}%</div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>测试准确率（{test.length} 朵未参与训练的花）</div>
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 8 }}>
            模型在<b>没见过的数据</b>上的表现才算数。这就是为什么要留出测试集——在训练集上再高的准确率都可能是过拟合的假象。
          </div>
        </Card>
        <Card title="混淆矩阵" sub="行=真实类别，列=预测类别，对角线=正确">
          <table className="ml-map" style={{ fontSize: 12, textAlign: 'center' }}>
            <thead>
              <tr><th></th>{SPECIES.map((s, i) => <th key={i} style={{ textAlign: 'center' }}>预测{s.slice(0, 4)}</th>)}</tr>
            </thead>
            <tbody>
              {confusion.map((row, i) => (
                <tr key={i}>
                  <td><b style={{ color: COLORS[i] }}>真{SPECIES[i].slice(0, 4)}</b></td>
                  {row.map((v, j) => (
                    <td key={j} style={{ textAlign: 'center', fontFamily: 'var(--mono)', fontWeight: i === j ? 700 : 400, color: i === j ? 'var(--green)' : (v > 0 ? 'var(--red)' : 'var(--text-faint)'), background: i === j && v > 0 ? 'rgba(61,220,132,0.1)' : 'transparent' }}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 8 }}>
            非对角线的数字就是分错的样本。Setosa 通常 100% 分对（它离另两类很远）；Versicolor 和 Virginica 偶有混淆（它们有重叠）。
          </div>
        </Card>
      </div>

      <Card title="完整流程对应的 sklearn 代码">
        <div className="formula" style={{ fontSize: 12.5, lineHeight: 1.9, textAlign: 'left', whiteSpace: 'pre-wrap' }}>
{`from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33, stratify=y)
clf = make_pipeline(StandardScaler(), LogisticRegression())  # 标准化只 fit 训练集
clf.fit(X_tr, y_tr)
print(clf.score(X_te, y_te))`}
        </div>
      </Card>

      <Callout>
        <b>关键点：</b> <code>make_pipeline</code> 把标准化和模型绑在一起——<code>fit</code> 时 StandardScaler 只学训练集的均值方差，<code>predict</code> 时自动复用，天然防数据泄漏。这正是 11 号特征工程模块强调的「两条铁律」的落地方式。
      </Callout>
    </div>
  )
}
