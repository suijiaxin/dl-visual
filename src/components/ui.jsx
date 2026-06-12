import React from 'react'

// 一组可复用的小组件，所有模块共享，减少重复代码、方便你统一调风格

export function Slider({ label, value, min, max, step = 1, onChange, fmt = (v) => v }) {
  return (
    <div className="control">
      <label>
        <span>{label}</span>
        <b>{fmt(value)}</b>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  )
}

export function Card({ title, sub, children, right }) {
  return (
    <div className="card fade-in">
      {(title || right) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            {title && <div className="card-title">{title}</div>}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {right}
        </div>
      )}
      {children}
    </div>
  )
}

export function Callout({ children, type }) {
  return <div className={`callout ${type || ''}`}>{children}</div>
}

export function Formula({ children }) {
  return <div className="formula">{children}</div>
}

export function PageHeader({ eyebrow, title, lead }) {
  return (
    <div className="page-header fade-in">
      {eyebrow && <div className="eyebrow">{eyebrow}</div>}
      <h2>{title}</h2>
      {lead && <p className="lead">{lead}</p>}
    </div>
  )
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <div className="legend-item" key={i}>
          <span className="legend-dot" style={{ background: it.color }} />
          {it.label}
        </div>
      ))}
    </div>
  )
}

// 把数值映射成蓝->白->红的热力色，用于注意力矩阵 / 权重可视化
export function heatColor(v, min = 0, max = 1) {
  const t = max === min ? 0.5 : (v - min) / (max - min)
  // 0 -> 深蓝, 0.5 -> 中性, 1 -> 暖橙
  const clamp = (x) => Math.max(0, Math.min(255, Math.round(x)))
  if (t < 0.5) {
    const k = t / 0.5
    return `rgb(${clamp(30 + k * 150)}, ${clamp(60 + k * 150)}, ${clamp(120 + k * 120)})`
  }
  const k = (t - 0.5) / 0.5
  return `rgb(${clamp(180 + k * 75)}, ${clamp(210 - k * 90)}, ${clamp(240 - k * 190)})`
}

// 工具：格式化小数
export const f2 = (x) => (Math.round(x * 100) / 100).toFixed(2)
export const f3 = (x) => (Math.round(x * 1000) / 1000).toFixed(3)
