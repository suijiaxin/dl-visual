import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite 配置：纯 React + JS，无 TypeScript，方便直接改代码
export default defineConfig({
  plugins: [react()],
  base: './',  // 使用相对路径，支持部署到任意路径（GitHub Pages / ProtoHub 等）
  server: {
    port: 5180,
    open: true,
  },
})
