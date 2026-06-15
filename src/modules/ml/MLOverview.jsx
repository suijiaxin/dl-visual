import React from 'react'
import { PageHeader, Card, Callout } from '../../components/ui.jsx'

// 机器学习总览：演进时间线 + 「场景 → 算法 → 框架」对照地图。
// 改 TIMELINE / SCENARIOS 数组即可增删条目。

const TIMELINE = [
  {
    year: '1950s–60s',
    name: '线性模型 · 感知机',
    color: 'var(--accent)',
    idea: '最小二乘线性回归、逻辑回归、感知机——用一条直线/超平面拟合或分割数据。',
    pain: '只能学线性关系，复杂边界无能为力。',
  },
  {
    year: '1980s–90s',
    name: '决策树 · SVM',
    color: 'var(--cyan)',
    idea: '决策树用 if-else 递归切分特征空间；SVM 用核技巧在高维找最大间隔超平面。',
    pain: '单棵树易过拟合；SVM 在大数据上训练慢、调核难。',
  },
  {
    year: '2000s',
    name: '集成学习',
    color: 'var(--green)',
    idea: 'Bagging（随机森林）降方差，Boosting（GBDT）逐轮纠错降偏差，成为表格数据霸主。',
    pain: '可解释性下降，超参数变多。',
  },
  {
    year: '2014–至今',
    name: 'XGBoost / LightGBM',
    color: 'var(--orange)',
    idea: '工程化的梯度提升：二阶近似、直方图加速、正则化，至今仍是结构化数据竞赛与风控首选。',
    pain: '面对图像/文本/语音等非结构化数据，仍让位于深度学习。',
  },
]

const SCENARIOS = [
  {
    scene: '回归 · 预测连续值',
    examples: '房价、销量、寿命预测',
    algos: '线性回归、岭/Lasso、GBDT 回归',
    frame: 'sklearn · statsmodels · XGBoost',
    color: 'var(--accent)',
    nav: 'ml-linreg',
  },
  {
    scene: '分类 · 预测类别',
    examples: '垃圾邮件、违约判定、图像标签',
    algos: '逻辑回归、决策树、随机森林、SVM、XGBoost',
    frame: 'sklearn · XGBoost · LightGBM',
    color: 'var(--green)',
    nav: 'ml-logreg',
  },
  {
    scene: '聚类 · 无标签分群',
    examples: '用户分层、异常检测',
    algos: 'K-Means、层次聚类、DBSCAN',
    frame: 'sklearn',
    color: 'var(--cyan)',
    nav: 'ml-cluster',
  },
  {
    scene: '降维 · 压缩与可视化',
    examples: '特征压缩、可视化高维数据',
    algos: 'PCA、t-SNE、UMAP',
    frame: 'sklearn',
    color: 'var(--accent-2)',
    nav: 'ml-pca',
  },
  {
    scene: '时序 · 预测未来',
    examples: '流量、电量、股价、销量',
    algos: '移动平均、指数平滑、ARIMA、Prophet',
    frame: 'statsmodels · Prophet',
    color: 'var(--orange)',
    nav: 'ml-ts',
  },
  {
    scene: '风控 · 评分卡',
    examples: '信贷评分、反欺诈',
    algos: 'WOE 分箱 + 逻辑回归 → 标准分',
    frame: 'sklearn · toad · scorecardpy',
    color: 'var(--pink)',
    nav: 'ml-scorecard',
  },
]

export default function MLOverview({ onNav }) {
  return (
    <div>
      <PageHeader
        eyebrow="00 · 机器学习总览"
        title="经典机器学习：从一条直线到表格数据霸主"
        lead="深度学习之外，结构化/表格数据的世界由经典机器学习统治。它们计算开销小、可解释性强、上线快——风控、推荐、运营、预测里随处可见。这条线从最简单的线性模型，一路演进到今天竞赛与风控仍在用的 XGBoost / LightGBM。下面先看演进脉络，再看「什么场景该用什么算法、配什么框架」。"
      />

      <Card title="演进时间线" sub="每一步都在解决上一步的痛点">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {TIMELINE.map((t, i) => (
            <div key={i} style={{ display: 'flex', gap: 16 }}>
              {/* 时间轴竖线 + 节点 */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 16 }}>
                <div style={{ width: 14, height: 14, borderRadius: '50%', background: t.color, flexShrink: 0, marginTop: 4 }} />
                {i < TIMELINE.length - 1 && (
                  <div style={{ width: 2, flex: 1, background: 'var(--border)', minHeight: 40 }} />
                )}
              </div>
              <div style={{ paddingBottom: 22 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-faint)' }}>{t.year}</span>
                  <strong style={{ fontSize: 15, color: t.color }}>{t.name}</strong>
                </div>
                <p style={{ fontSize: 13, color: 'var(--text-dim)', marginTop: 4 }}>{t.idea}</p>
                <p style={{ fontSize: 12.5, color: 'var(--text-faint)', marginTop: 2 }}>痛点 → {t.pain}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="场景 → 算法 → 框架 对照地图" sub="点任意一行跳到对应模块">
        <div style={{ overflowX: 'auto' }}>
          <table className="ml-map">
            <thead>
              <tr>
                <th>场景</th>
                <th>典型例子</th>
                <th>主流算法</th>
                <th>常用框架</th>
              </tr>
            </thead>
            <tbody>
              {SCENARIOS.map((s, i) => (
                <tr key={i} onClick={() => onNav && onNav(s.nav)} style={{ cursor: 'pointer' }}>
                  <td>
                    <span className="ml-map-dot" style={{ background: s.color }} />
                    <strong>{s.scene}</strong>
                  </td>
                  <td style={{ color: 'var(--text-dim)' }}>{s.examples}</td>
                  <td style={{ color: 'var(--text-dim)' }}>{s.algos}</td>
                  <td style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-faint)' }}>{s.frame}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid-2">
        <Card title="经典 ML vs 深度学习：怎么选？">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong style={{ color: 'var(--accent)' }}>数据是表格/结构化的</strong>（几十到几千列特征、行数有限）→ 优先 XGBoost / LightGBM，往往比神经网络又快又准。</p>
            <p><strong style={{ color: 'var(--accent-2)' }}>数据是图像/文本/语音</strong>（高维、有空间或时序结构）→ 深度学习主场（见另一个领域）。</p>
            <p><strong style={{ color: 'var(--green)' }}>要可解释、要快速上线、样本少</strong> → 经典 ML 几乎总是更省心。</p>
          </div>
        </Card>
        <Card title="三大主流框架定位">
          <div className="prose" style={{ fontSize: 13 }}>
            <p><strong>scikit-learn：</strong> 经典算法的「瑞士军刀」，统一的 fit/predict API，覆盖回归/分类/聚类/降维/预处理。</p>
            <p><strong>XGBoost / LightGBM：</strong> 专攻梯度提升树，表格数据竞赛与工业风控的事实标准；LightGBM 在大数据上更快更省内存。</p>
            <p><strong>statsmodels：</strong> 偏统计建模，提供 p 值、置信区间、ARIMA 等，做时序与因果分析时不可替代。</p>
          </div>
        </Card>
      </div>

      <Card title="特征工程与实战案例" sub="光懂算法不够，真实项目要走完整流程。点卡片直达">
        <div className="grid-2" style={{ gap: 12 }}>
          {[
            { nav: 'ml-feateng', tag: '11', title: '特征工程', desc: '缺失值/编码/标准化/分箱/交互特征，模型上限由这里决定', color: 'var(--accent-2)' },
            { nav: 'ml-iris', tag: '12', title: '鸢尾花分类', desc: '入门经典：划分→标准化→分类→混淆矩阵', color: 'var(--green)' },
            { nav: 'ml-titanic', tag: '13', title: '泰坦尼克生还', desc: '脏数据实战：缺失填充→特征衍生→逻辑回归', color: 'var(--orange)' },
            { nav: 'ml-boston', tag: '14', title: '波士顿房价', desc: '多元线性回归：标准化→训练→预测vs真实→R²', color: 'var(--cyan)' },
            { nav: 'ml-mall', tag: '15', title: '客户分群', desc: 'K-Means 经典商业落地：客户分层→营销画像', color: 'var(--pink)' },
            { nav: 'ml-air', tag: '16', title: '航空客流', desc: '时序教科书数据集：趋势×乘性季节→预测+MAPE', color: 'var(--accent)' },
          ].map((c) => (
            <div key={c.nav} onClick={() => onNav && onNav(c.nav)}
              style={{ cursor: 'pointer', padding: '12px 14px', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-elev)', display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: c.color, fontWeight: 700, flexShrink: 0 }}>{c.tag}</span>
              <div>
                <div style={{ fontSize: 13.5, fontWeight: 600, color: c.color }}>{c.title}</div>
                <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 2 }}>{c.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Callout>
        <b>建议顺序：</b> 左侧菜单从 01 线性回归往下走，01–10 讲算法原理，11–14 是特征工程与完整实战案例。每个模块都能调参数看真实计算结果。想随时切回深度学习，点左上角标题下拉即可。
      </Callout>
    </div>
  )
}
