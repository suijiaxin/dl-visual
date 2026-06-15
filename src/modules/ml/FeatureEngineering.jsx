import React, { useState, useMemo } from 'react'
import { PageHeader, Card, Callout, Legend, Slider, Formula, f2 } from '../../components/ui.jsx'
import { mean, std } from '../../lib/mathx.js'

// 特征工程：在一份带缺失值/类别/不同尺度的小数据上，逐步演示
// 缺失值填充 → 类别编码 → 标准化 → 分箱 → 多项式特征。全部真实计算。

// 原始「脏」数据：年龄(有缺失) / 城市(类别) / 收入(尺度大) / 标签
const RAW = [
  { age: 25, city: '北京', income: 8000, y: 0 },
  { age: 38, city: '上海', income: 25000, y: 1 },
  { age: null, city: '北京', income: 12000, y: 0 },
  { age: 52, city: '广州', income: 30000, y: 1 },
  { age: 29, city: '上海', income: null, y: 0 },
  { age: 45, city: '广州', income: 42000, y: 1 },
  { age: 33, city: '北京', income: 15000, y: 0 },
  { age: null, city: '上海', income: 22000, y: 1 },
]

const STEPS = ['raw', 'impute', 'encode', 'scale', 'bin', 'poly']
const STEP_LABEL = {
  raw: '① 原始数据',
  impute: '② 缺失值填充',
  encode: '③ 类别编码',
  scale: '④ 标准化',
  bin: '⑤ 分箱',
  poly: '⑥ 多项式特征',
}

export default function FeatureEngineering() {
  const [step, setStep] = useState('raw')

  // 计算各列统计量（用于填充/标准化）
  const ages = RAW.map((r) => r.age).filter((v) => v !== null)
  const incomes = RAW.map((r) => r.income).filter((v) => v !== null)
  const ageMean = mean(ages)
  const incMean = mean(incomes)

  // 逐步构造处理后的数据
  const processed = useMemo(() => {
    return RAW.map((r) => {
      const row = { ...r }
      // 填充
      const ageF = r.age === null ? ageMean : r.age
      const incF = r.income === null ? incMean : r.income
      row._ageFilled = r.age === null
      row._incFilled = r.income === null
      row.age = ageF
      row.income = incF
      // 编码（one-hot）
      row.city_bj = r.city === '北京' ? 1 : 0
      row.city_sh = r.city === '上海' ? 1 : 0
      row.city_gz = r.city === '广州' ? 1 : 0
      // 标准化
      row.ageZ = (ageF - ageMean) / std(ages)
      row.incZ = (incF - incMean) / std(incomes)
      // 分箱（年龄段）
      row.ageBin = ageF < 30 ? '青年' : ageF < 45 ? '中年' : '资深'
      // 多项式（age × income 交互项，标准化后）
      row.ageXinc = row.ageZ * row.incZ
      return row
    })
  }, [])

  const stepIdx = STEPS.indexOf(step)
  const showImpute = stepIdx >= 1
  const showEncode = stepIdx >= 2
  const showScale = stepIdx >= 3
  const showBin = stepIdx >= 4
  const showPoly = stepIdx >= 5

  return (
    <div>
      <PageHeader
        eyebrow="11 · 特征工程"
        title="特征工程：模型好坏，一半在这里决定"
        lead="「数据和特征决定了机器学习的上限，模型只是逼近这个上限。」真实数据往往是脏的——有缺失、有文字类别、量纲悬殊。喂给模型前必须清洗和变换。下面用一份 8 行的脏数据，一步步走完最常用的特征工程流水线，每步都是真实计算。"
      />

      <div className="btn-row" style={{ marginBottom: 16, flexWrap: 'wrap' }}>
        {STEPS.map((s) => (
          <button key={s} className={`btn ${step === s ? '' : 'secondary'}`} onClick={() => setStep(s)}>{STEP_LABEL[s]}</button>
        ))}
      </div>

      <Card title={STEP_LABEL[step]} sub="黄色高亮 = 当前步骤新增/改变的列">
        <div style={{ overflowX: 'auto' }}>
          <table className="ml-map" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>age</th>
                <th>city</th>
                <th>income</th>
                {showEncode && <th style={{ color: 'var(--yellow)' }}>北京/上海/广州</th>}
                {showScale && <th style={{ color: 'var(--yellow)' }}>age_z</th>}
                {showScale && <th style={{ color: 'var(--yellow)' }}>inc_z</th>}
                {showBin && <th style={{ color: 'var(--yellow)' }}>age_bin</th>}
                {showPoly && <th style={{ color: 'var(--yellow)' }}>age×inc</th>}
                <th>y</th>
              </tr>
            </thead>
            <tbody>
              {processed.map((r, i) => {
                const rawAge = RAW[i].age
                const rawInc = RAW[i].income
                return (
                  <tr key={i}>
                    <td style={{ color: showImpute && r._ageFilled ? 'var(--yellow)' : 'var(--text)' }}>
                      {!showImpute && rawAge === null ? <span style={{ color: 'var(--red)' }}>缺失</span> : (showImpute ? f2(r.age) : rawAge)}
                      {showImpute && r._ageFilled && ' ←填充'}
                    </td>
                    <td>{RAW[i].city}</td>
                    <td style={{ color: showImpute && r._incFilled ? 'var(--yellow)' : 'var(--text)' }}>
                      {!showImpute && rawInc === null ? <span style={{ color: 'var(--red)' }}>缺失</span> : (showImpute ? Math.round(r.income) : rawInc)}
                      {showImpute && r._incFilled && ' ←填充'}
                    </td>
                    {showEncode && <td style={{ fontFamily: 'var(--mono)' }}>{r.city_bj}/{r.city_sh}/{r.city_gz}</td>}
                    {showScale && <td style={{ fontFamily: 'var(--mono)', color: r.ageZ >= 0 ? 'var(--accent)' : 'var(--orange)' }}>{f2(r.ageZ)}</td>}
                    {showScale && <td style={{ fontFamily: 'var(--mono)', color: r.incZ >= 0 ? 'var(--accent)' : 'var(--orange)' }}>{f2(r.incZ)}</td>}
                    {showBin && <td>{r.ageBin}</td>}
                    {showPoly && <td style={{ fontFamily: 'var(--mono)' }}>{f2(r.ageXinc)}</td>}
                    <td><b style={{ color: r.y === 1 ? 'var(--green)' : 'var(--text-dim)' }}>{r.y}</b></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div style={{ fontSize: 13, color: 'var(--text-dim)', marginTop: 14, lineHeight: 1.8 }}>
          {step === 'raw' && <>原始数据有三个问题：<b style={{ color: 'var(--red)' }}>age/income 有缺失</b>、<b>city 是文字</b>模型不认、<b>income 数值远大于 age</b>会主导距离类模型。逐步点上面的按钮处理。</>}
          {step === 'impute' && <>缺失值用该列<b>均值</b>填充（age→{f2(ageMean)}，income→{Math.round(incMean)}）。也可用中位数（抗异常值）、众数（类别列）或模型预测。<b>切忌直接删行</b>——会丢信息。</>}
          {step === 'encode' && <>city 做 <b>One-Hot 编码</b>：拆成 3 个 0/1 列。对无序类别用 One-Hot；有序类别（如学历）可用序号编码；高基数类别（如邮编）用目标编码 / 频数编码。</>}
          {step === 'scale' && <>标准化 z = (x − μ) / σ，让每列均值 0、方差 1。<b>KNN/SVM/线性模型/PCA 必须做</b>，否则量纲大的 income 会碾压 age。树模型则不需要。</>}
          {step === 'bin' && <>把连续的 age 切成<b>青年/中年/资深</b>三段。分箱能引入非线性、增强鲁棒性（抗异常值），是评分卡 WOE 的基础。代价是损失一些信息粒度。</>}
          {step === 'poly' && <>构造<b>交互特征</b> age×income：捕捉「年龄和收入的组合效应」。多项式特征能让线性模型表达非线性关系，但维度会爆炸，需配合特征选择。</>}
        </div>
      </Card>

      <Formula>标准化 z = (x − μ) / σ　　归一化 x' = (x − min) / (max − min)　　One-Hot: 类别 → 多个 0/1 列</Formula>

      <div className="grid-2">
        <Card title="特征工程常用武器库">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>缺失值：</strong> <code>SimpleImputer(strategy='mean'/'median'/'most_frequent')</code></p>
            <p><strong>类别编码：</strong> <code>OneHotEncoder</code> / <code>OrdinalEncoder</code> / 目标编码（category_encoders）</p>
            <p><strong>缩放：</strong> <code>StandardScaler</code>（标准化）/ <code>MinMaxScaler</code>（归一化）/ <code>RobustScaler</code>（抗异常值）</p>
            <p><strong>构造：</strong> <code>PolynomialFeatures</code>、分箱 <code>KBinsDiscretizer</code>、日期拆解、文本 TF-IDF</p>
          </div>
        </Card>
        <Card title="两条铁律">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>① 防数据泄漏：</strong> 标准化/填充的统计量（μ、σ、均值）<b>只能在训练集上算</b>，再用同一组参数变换验证/测试集。否则测试信息「泄漏」进训练，线下虚高、上线翻车。</p>
            <p><strong>② 用 Pipeline 封装：</strong> <code>sklearn.pipeline.Pipeline</code> 把预处理和模型串成一个对象，<code>fit</code>/<code>predict</code> 一气呵成，自动保证训练和预测用同一套变换，杜绝泄漏。</p>
          </div>
        </Card>
      </div>

      <Callout>
        <b>接下来：</b> 12～14 号是三个完整实战案例（鸢尾花 / 泰坦尼克 / 波士顿房价），把这里的特征工程套到真实数据集上，走完「预处理 → 特征 → 训练 → 评估」全流程。
      </Callout>
    </div>
  )
}
