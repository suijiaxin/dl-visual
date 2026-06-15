import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, f2 } from '../../components/ui.jsx'
import { sigmoid, mean, std } from '../../lib/mathx.js'

// 泰坦尼克生还预测：侧重「脏数据 → 预处理 → 特征工程 → 逻辑回归训练」全流程。
// 真实风格子集，含缺失 Age、类别 Sex/Embarked。真实梯度下降训练。

// [Pclass, Sex(0男1女), Age(可空), SibSp, Parch, Fare, Embarked(S/C/Q), Survived]
const RAW = [
  [3, 0, 22, 1, 0, 7.25, 'S', 0], [1, 1, 38, 1, 0, 71.3, 'C', 1], [3, 1, 26, 0, 0, 7.92, 'S', 1],
  [1, 1, 35, 1, 0, 53.1, 'S', 1], [3, 0, 35, 0, 0, 8.05, 'S', 0], [3, 0, null, 0, 0, 8.46, 'Q', 0],
  [1, 0, 54, 0, 0, 51.9, 'S', 0], [3, 0, 2, 3, 1, 21.07, 'S', 0], [3, 1, 27, 0, 2, 11.13, 'S', 1],
  [2, 1, 14, 1, 0, 30.07, 'C', 1], [3, 1, 4, 1, 1, 16.7, 'S', 1], [1, 1, 58, 0, 0, 26.55, 'S', 1],
  [3, 0, 20, 0, 0, 8.05, 'S', 0], [3, 0, 39, 1, 5, 31.27, 'S', 0], [3, 1, 14, 0, 0, 7.85, 'S', 0],
  [2, 1, 55, 0, 0, 16.0, 'S', 1], [3, 0, 2, 4, 1, 29.12, 'Q', 0], [2, 0, null, 0, 0, 13.0, 'S', 1],
  [3, 1, 31, 1, 0, 18.0, 'S', 0], [3, 1, null, 0, 0, 7.22, 'C', 1], [1, 0, 35, 0, 0, 26.55, 'S', 0],
  [3, 0, 34, 0, 0, 8.05, 'S', 0], [1, 1, 15, 0, 1, 211.3, 'S', 1], [1, 0, 28, 0, 0, 35.5, 'S', 1],
  [3, 1, 8, 3, 1, 21.07, 'S', 0], [3, 1, 38, 1, 5, 31.39, 'S', 0], [3, 0, null, 0, 0, 7.75, 'Q', 0],
  [2, 0, 19, 0, 0, 13.0, 'S', 0], [1, 1, 40, 0, 0, 153.46, 'S', 1], [1, 1, 50, 0, 1, 247.52, 'C', 1],
  [3, 0, 33, 0, 0, 7.9, 'S', 0], [2, 1, 24, 0, 2, 27.0, 'S', 1], [3, 0, 30, 0, 0, 7.23, 'C', 0],
  [1, 0, 42, 1, 0, 52.0, 'S', 0], [3, 1, 18, 0, 1, 9.35, 'S', 1], [2, 0, 31, 0, 0, 10.5, 'S', 0],
  [1, 1, 49, 1, 0, 76.73, 'C', 1], [3, 0, 25, 0, 0, 7.05, 'S', 0], [2, 1, 36, 0, 0, 13.0, 'S', 1],
  [3, 0, 45, 0, 0, 8.05, 'S', 0],
]

const FEAT_NAMES = ['Pclass(舱等)', 'Sex(性别)', 'Age(年龄)', 'FamilySize(家庭)', 'Fare(票价)', 'Embarked_C', 'Embarked_Q']

export default function Titanic() {
  const [iterShown, setIterShown] = useState(300)

  // 预处理：Age 缺失用训练集中位数填充
  const ages = RAW.map((r) => r[2]).filter((v) => v !== null).sort((a, b) => a - b)
  const ageMedian = ages[Math.floor(ages.length / 2)]
  const nMissing = RAW.filter((r) => r[2] === null).length

  // 特征工程 + 标准化（数值列）
  const { X, y, scaler } = useMemo(() => {
    // 构造原始特征矩阵
    const rows = RAW.map((r) => {
      const age = r[2] === null ? ageMedian : r[2]
      const familySize = r[3] + r[4] + 1 // SibSp + Parch + 自己
      const embC = r[6] === 'C' ? 1 : 0
      const embQ = r[6] === 'Q' ? 1 : 0
      return { feats: [r[0], r[1], age, familySize, r[5], embC, embQ], y: r[7] }
    })
    // 标准化连续列（索引 0,2,3,4），0/1 列不动
    const contIdx = [0, 2, 3, 4]
    const scaler = {}
    contIdx.forEach((c) => {
      const col = rows.map((r) => r.feats[c])
      scaler[c] = { mu: mean(col), sd: std(col) }
    })
    const X = rows.map((r) => r.feats.map((v, c) => contIdx.includes(c) ? (v - scaler[c].mu) / scaler[c].sd : v))
    const y = rows.map((r) => r.y)
    return { X, y, scaler }
  }, [])

  // 真实逻辑回归：梯度下降训练，记录每轮权重和损失
  const training = useMemo(() => {
    const n = X.length, d = X[0].length
    let w = new Array(d).fill(0), b = 0
    const lr = 0.1
    const history = []
    for (let iter = 0; iter <= 500; iter++) {
      const gw = new Array(d).fill(0); let gb = 0, loss = 0
      for (let i = 0; i < n; i++) {
        const zi = X[i].reduce((s, x, j) => s + x * w[j], b)
        const p = sigmoid(zi)
        const err = p - y[i]
        for (let j = 0; j < d; j++) gw[j] += err * X[i][j]
        gb += err
        const eps = 1e-9
        loss += -(y[i] * Math.log(p + eps) + (1 - y[i]) * Math.log(1 - p + eps))
      }
      if (iter % 10 === 0) {
        let correct = 0
        for (let i = 0; i < n; i++) {
          const p = sigmoid(X[i].reduce((s, x, j) => s + x * w[j], b))
          if ((p >= 0.5 ? 1 : 0) === y[i]) correct++
        }
        history.push({ iter, loss: loss / n, acc: correct / n, w: [...w], b })
      }
      for (let j = 0; j < d; j++) w[j] -= lr * gw[j] / n
      b -= lr * gb / n
    }
    return { history, w, b }
  }, [X, y])

  const curIdx = Math.min(Math.floor(iterShown / 10), training.history.length - 1)
  const cur = training.history[curIdx]

  // 分组生还率统计（用原始数据，直观）
  const survBy = (pred) => {
    const groups = {}
    RAW.forEach((r) => {
      const k = pred(r)
      if (!groups[k]) groups[k] = { n: 0, s: 0 }
      groups[k].n++; groups[k].s += r[7]
    })
    return groups
  }
  const bySex = survBy((r) => r[1] === 1 ? '女性' : '男性')
  const byClass = survBy((r) => `${r[0]}等舱`)

  const maxW = Math.max(...cur.w.map(Math.abs), 0.1)

  return (
    <div>
      <PageHeader
        eyebrow="13 · 实战 · 泰坦尼克"
        title="泰坦尼克生还预测：脏数据实战的代表作"
        lead="Kaggle 入门赛的经典题：根据乘客信息预测谁能在沉船中生还。它之所以是绝佳教材，是因为数据「脏」得很真实——年龄有缺失、性别和登船港是文字、还能从亲属数衍生出「家庭规模」。我们走一遍预处理 → 特征工程 → 真实梯度下降训练逻辑回归 → 看哪些特征最关键。"
      />

      <div className="grid-2">
        <Card title="① 预处理：处理缺失值">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>40 名乘客中有 <b style={{ color: 'var(--red)' }}>{nMissing} 人</b>年龄缺失。直接删行会丢样本，这里用训练集 <b>年龄中位数 {ageMedian} 岁</b>填充（中位数比均值更抗极端值）。</p>
            <p>Fare、Embarked 偶有缺失时同理：数值用中位数、类别用众数填充。</p>
          </div>
        </Card>
        <Card title="② 特征工程：衍生与编码">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><b>衍生：</b> FamilySize = SibSp + Parch + 1，把「兄弟姐妹数」和「父母子女数」合成更有意义的「家庭规模」。</p>
            <p><b>编码：</b> Sex 映射 0/1；Embarked（S/C/Q）做 One-Hot 拆成 Embarked_C、Embarked_Q 两列（S 为基准）。</p>
            <p><b>标准化：</b> Pclass/Age/FamilySize/Fare 这些连续列标准化到同一尺度。</p>
          </div>
        </Card>
      </div>

      <Card title="③ 分组生还率：先用肉眼找规律" sub="建模前的探索性分析，往往能预判哪些特征重要">
        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
          {[{ title: '按性别', data: bySex }, { title: '按舱等', data: byClass }].map((blk, bi) => (
            <div key={bi} style={{ flex: 1, minWidth: 240 }}>
              <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginBottom: 8 }}>{blk.title}</div>
              {Object.entries(blk.data).sort().map(([k, v]) => {
                const rate = v.s / v.n
                return (
                  <div key={k} style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span>{k}</span><span style={{ color: rate > 0.5 ? 'var(--green)' : 'var(--red)' }}>{Math.round(rate * 100)}% 生还</span>
                    </div>
                    <div style={{ height: 8, background: 'var(--slot)', borderRadius: 4, overflow: 'hidden', marginTop: 2 }}>
                      <div style={{ width: `${rate * 100}%`, height: '100%', background: rate > 0.5 ? 'var(--green)' : 'var(--red)' }} />
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
        <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginTop: 10 }}>
          「女士和孩子优先」+「头等舱靠近甲板」——性别和舱等的生还率差异巨大，预示它们会是最强特征。下面训练验证这个直觉。
        </div>
      </Card>

      <Card title="④ 训练逻辑回归（真实梯度下降）" sub="拖动看训练进程：损失下降、准确率上升">
        <div className="controls" style={{ marginBottom: 14 }}>
          <Slider label="训练轮数 iteration" value={iterShown} min={0} max={500} step={10} onChange={setIterShown} fmt={(v) => v} />
          <div className="control"><label><span>训练准确率</span><b style={{ color: 'var(--green)' }}>{Math.round(cur.acc * 100)}%</b></label></div>
          <div className="control"><label><span>交叉熵损失</span><b>{f2(cur.loss)}</b></label></div>
        </div>

        <div style={{ fontSize: 12.5, color: 'var(--text-dim)', marginBottom: 8 }}>各特征权重（绝对值越大越重要；蓝=利于生还，橙=不利）：</div>
        {FEAT_NAMES.map((name, j) => (
          <div key={j} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 5 }}>
            <div style={{ width: 120, fontSize: 12, flexShrink: 0, textAlign: 'right' }}>{name}</div>
            <div style={{ flex: 1, height: 16, position: 'relative', background: 'var(--slot)', borderRadius: 4 }}>
              <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--border)' }} />
              <div style={{
                position: 'absolute', height: '100%', borderRadius: 4,
                left: cur.w[j] >= 0 ? '50%' : `${50 - Math.abs(cur.w[j]) / maxW * 50}%`,
                width: `${Math.abs(cur.w[j]) / maxW * 50}%`,
                background: cur.w[j] >= 0 ? 'var(--accent)' : 'var(--orange)',
              }} />
            </div>
            <div style={{ width: 44, fontSize: 11, fontFamily: 'var(--mono)', color: cur.w[j] >= 0 ? 'var(--accent)' : 'var(--orange)' }}>{f2(cur.w[j])}</div>
          </div>
        ))}
      </Card>

      <div className="grid-2">
        <Card title="模型学到了什么？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p>训练收敛后，<b style={{ color: 'var(--accent)' }}>Sex（性别）</b>权重最大且为正——女性生还概率显著更高，和探索分析吻合。</p>
            <p><b style={{ color: 'var(--orange)' }}>Pclass</b> 标准化后为负权重：舱等数字越大（3 等舱）越不利。<b>Fare</b> 越高（往往是高等舱）越有利。</p>
            <p>这正是逻辑回归的价值：不仅能预测，<b>权重的正负和大小直接告诉你每个因素怎么影响结果</b>，可解释性极强。</p>
          </div>
        </Card>
        <Card title="完整 sklearn Pipeline">
          <div className="formula" style={{ fontSize: 11.5, lineHeight: 1.85, textAlign: 'left', whiteSpace: 'pre-wrap' }}>
{`from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num = ['Pclass','Age','FamilySize','Fare']
cat = ['Embarked']
pre = ColumnTransformer([
  ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                    ('sc', StandardScaler())]), num),
  ('cat', OneHotEncoder(), cat)])
clf = Pipeline([('pre', pre),
                ('lr', LogisticRegression())]).fit(X_tr, y_tr)`}
          </div>
        </Card>
      </div>

      <Callout>
        <b>真实成绩参考：</b> 这套「中位数填充 + One-Hot + 衍生 FamilySize + 逻辑回归」在完整泰坦尼克数据上约 78~80% 准确率。想再提升就换随机森林/XGBoost、做更多特征（如从姓名里提取 Title 头衔）——但流程框架完全一样。
      </Callout>
    </div>
  )
}
