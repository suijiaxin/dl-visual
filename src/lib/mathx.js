// 共享的纯函数数学库 —— 所有模块的可视化都跑真实计算，不是假动画。
// 你可以在这里改激活函数、初始化方式等，所有模块会跟着变。

export function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export function matVec(M, v) {
  return M.map((row) => dot(row, v))
}

// 矩阵乘法 A(m×k) · B(k×n) = (m×n)
export function matMul(A, B) {
  const m = A.length
  const k = A[0].length
  const n = B[0].length
  const C = Array.from({ length: m }, () => new Array(n).fill(0))
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let p = 0; p < k; p++) s += A[i][p] * B[p][j]
      C[i][j] = s
    }
  return C
}

export function transpose(M) {
  return M[0].map((_, j) => M.map((row) => row[j]))
}

// 数值稳定的 softmax
export function softmax(arr) {
  const max = Math.max(...arr)
  const exps = arr.map((x) => Math.exp(x - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

// 激活函数
export const activations = {
  relu: { fn: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), label: 'ReLU', tex: 'max(0, x)' },
  sigmoid: {
    fn: (x) => 1 / (1 + Math.exp(-x)),
    d: (x) => {
      const s = 1 / (1 + Math.exp(-x))
      return s * (1 - s)
    },
    label: 'Sigmoid',
    tex: '1 / (1 + e^-x)',
  },
  tanh: { fn: (x) => Math.tanh(x), d: (x) => 1 - Math.tanh(x) ** 2, label: 'Tanh', tex: 'tanh(x)' },
}

// 一个可复现的伪随机数发生器（带种子），保证刷新页面结果一致、可调试
export function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// 标准正态采样（Box-Muller），用于权重初始化
export function randn(rng) {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

// 向量加 / 缩放
export const vAdd = (a, b) => a.map((x, i) => x + b[i])
export const vScale = (a, s) => a.map((x) => x * s)

// LayerNorm（单个向量）
export function layerNorm(v, gamma = 1, beta = 0, eps = 1e-5) {
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  const variance = v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length
  return v.map((x) => gamma * ((x - mean) / Math.sqrt(variance + eps)) + beta)
}

// ===== 经典机器学习通用函数（同样跑真实计算）=====

export const sigmoid = (x) => 1 / (1 + Math.exp(-x))

export const mean = (arr) => arr.reduce((a, b) => a + b, 0) / (arr.length || 1)

export function std(arr) {
  const m = mean(arr)
  return Math.sqrt(mean(arr.map((x) => (x - m) ** 2)))
}

// 欧氏距离（任意维）
export function dist(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
  return Math.sqrt(s)
}

// 一元最小二乘解析解：返回 { w, b } 使 y ≈ w·x + b
export function linregFit(xs, ys) {
  const n = xs.length
  const mx = mean(xs)
  const my = mean(ys)
  let num = 0, den = 0
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my)
    den += (xs[i] - mx) ** 2
  }
  const w = den === 0 ? 0 : num / den
  const b = my - w * mx
  return { w, b }
}

// 决定系数 R²
export function r2(ys, preds) {
  const my = mean(ys)
  let ssRes = 0, ssTot = 0
  for (let i = 0; i < ys.length; i++) {
    ssRes += (ys[i] - preds[i]) ** 2
    ssTot += (ys[i] - my) ** 2
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot
}

// 均方误差
export const mse = (ys, preds) => mean(ys.map((y, i) => (y - preds[i]) ** 2))

// 基尼不纯度：labels 为类别数组
export function gini(labels) {
  if (labels.length === 0) return 0
  const counts = {}
  labels.forEach((l) => (counts[l] = (counts[l] || 0) + 1))
  let imp = 1
  for (const k in counts) imp -= (counts[k] / labels.length) ** 2
  return imp
}

// 信息熵
export function entropy(labels) {
  if (labels.length === 0) return 0
  const counts = {}
  labels.forEach((l) => (counts[l] = (counts[l] || 0) + 1))
  let h = 0
  for (const k in counts) {
    const p = counts[k] / labels.length
    h -= p * Math.log2(p)
  }
  return h
}

// 多数投票，返回出现最多的标签
export function majority(labels) {
  const counts = {}
  labels.forEach((l) => (counts[l] = (counts[l] || 0) + 1))
  let best = null, bestN = -1
  for (const k in counts) if (counts[k] > bestN) { bestN = counts[k]; best = k }
  return best
}

// K-Means 单步迭代：给定点集和质心，返回新分配与新质心
// points: [[x,y],...]  centroids: [[x,y],...]
export function kmeansStep(points, centroids) {
  const assign = points.map((p) => {
    let best = 0, bestD = Infinity
    centroids.forEach((c, i) => {
      const d = dist(p, c)
      if (d < bestD) { bestD = d; best = i }
    })
    return best
  })
  const newCentroids = centroids.map((c, i) => {
    const members = points.filter((_, idx) => assign[idx] === i)
    if (members.length === 0) return c
    const dim = c.length
    const sum = new Array(dim).fill(0)
    members.forEach((p) => p.forEach((v, d) => (sum[d] += v)))
    return sum.map((s) => s / members.length)
  })
  // 簇内平方和（惯性），用于肘部法
  let inertia = 0
  points.forEach((p, idx) => (inertia += dist(p, centroids[assign[idx]]) ** 2))
  return { assign, newCentroids, inertia }
}

// 2D PCA：返回主成分方向（单位向量）与方差解释比例
// points: [[x,y],...]
export function pca2d(points) {
  const mx = mean(points.map((p) => p[0]))
  const my = mean(points.map((p) => p[1]))
  let cxx = 0, cyy = 0, cxy = 0
  points.forEach(([x, y]) => {
    cxx += (x - mx) ** 2
    cyy += (y - my) ** 2
    cxy += (x - mx) * (y - my)
  })
  const n = points.length || 1
  cxx /= n; cyy /= n; cxy /= n
  // 2×2 对称矩阵 [[cxx,cxy],[cxy,cyy]] 的特征值/向量解析解
  const tr = cxx + cyy
  const det = cxx * cyy - cxy * cxy
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det))
  const l1 = tr / 2 + disc // 较大特征值
  const l2 = tr / 2 - disc
  // 主成分方向：特征向量
  let v1
  if (Math.abs(cxy) > 1e-9) v1 = [l1 - cyy, cxy]
  else v1 = cxx >= cyy ? [1, 0] : [0, 1]
  const norm = Math.hypot(v1[0], v1[1]) || 1
  v1 = [v1[0] / norm, v1[1] / norm]
  const v2 = [-v1[1], v1[0]]
  const ratio = l1 + l2 === 0 ? 0 : l1 / (l1 + l2)
  return { mean: [mx, my], v1, v2, l1, l2, ratio }
}

// 简单移动平均
export function movingAverage(series, window) {
  return series.map((_, i) => {
    const start = Math.max(0, i - window + 1)
    const slice = series.slice(start, i + 1)
    return mean(slice)
  })
}

// 指数平滑（一次）：s_t = α·x_t + (1-α)·s_{t-1}
export function expSmoothing(series, alpha) {
  const out = []
  let s = series[0]
  series.forEach((x, i) => {
    s = i === 0 ? x : alpha * x + (1 - alpha) * s
    out.push(s)
  })
  return out
}
