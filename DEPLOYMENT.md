# 部署指南

## ProtoHub / Vercel / Netlify / GitHub Pages

项目已配置为使用相对路径，支持部署到任意路径。

### 快速部署

1. **构建项目**
   ```bash
   npm run build
   ```

2. **部署 `dist/` 目录**
   
   - **ProtoHub**: 直接上传 `dist` 目录的所有文件
   - **Vercel**: 
     - 连接 GitHub 仓库
     - Framework Preset: `Vite`
     - Build Command: `npm run build`
     - Output Directory: `dist`
   
   - **Netlify**: 同 Vercel 配置
   
   - **GitHub Pages**:
     ```bash
     npm run build
     cd dist
     git init
     git add -A
     git commit -m 'deploy'
     git push -f git@github.com:你的用户名/仓库名.git main:gh-pages
     ```
     然后在仓库 Settings → Pages 中选择 `gh-pages` 分支

### 本地预览构建结果

```bash
npm run preview
# 访问 http://localhost:4173
```

### 验证清单

部署后检查：
- ✅ 页面正常加载（没有白屏）
- ✅ 左侧导航栏显示正常
- ✅ 点击不同模块能切换内容
- ✅ 右上角主题切换按钮工作
- ✅ favicon 图标显示

### 常见问题

**Q: 部署后白屏？**
A: 检查浏览器控制台，如果报 404 错误，确认：
- `vite.config.js` 中 `base: './'` 已配置
- 重新运行 `npm run build`

**Q: 部署到子路径（如 `/my-project/`）?**
A: 使用相对路径 `base: './'` 会自动适配，无需修改

**Q: favicon 不显示？**
A: 清除浏览器缓存，或强制刷新（Cmd+Shift+R / Ctrl+Shift+R）

### 推荐平台

- **Vercel** - 最简单，连接 GitHub 自动部署
- **Netlify** - 同样简单，拖拽 `dist` 目录即可
- **ProtoHub** - 适合快速演示原型
- **GitHub Pages** - 完全免费，适合开源项目
