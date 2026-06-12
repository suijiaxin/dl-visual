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
