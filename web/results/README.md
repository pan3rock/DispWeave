# 本地结果页

用浏览器直接打开 `index.html`。无需构建、联网或 JavaScript。

优先直接展示 NUTS 模型区间、频散拟合和轨迹，其次是 LM 对照与合成数据。
其余 6 张图及报告在“补充图与报告”中，所有图片可点击打开原图或 PDF。
NUTS 三链图仍为诊断结果，原四链验收未通过。

## 更新

- 原位更新 `results/` 中的图，刷新网页即可。
- 新实验目录或结果说明有变化时，只改 `index.html` 的图片路径、链接和文字。
- 不必维护图集配置、交互图或生成页面；本页未注册或发布远程站点。
- 只有实际采样验收通过后，才更新 NUTS 的状态说明。

图表通过相对路径读取，需保留整个项目的目录结构。生成图继续由 `.gitignore` 排除，
新克隆的仓库需要恢复或生成结果。`scripts/plot_crust_target.py` 用于目标对照补图，
不需要每次修改网页都执行。

如需 HTTP 查看，在项目根目录执行：

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory .
```

打开 `http://127.0.0.1:8000/web/results/`，结束时 Ctrl+C。

本轮简化仅改展示页面，删除旧图集脚本和独立样式文件，没有重新反演、采样或绘图。
已在断网、禁用 JavaScript 的 Chromium 中检查 5 张主图、11 组 PNG/PDF 链接、
原图打开和桌面/手机布局；记录位于 `results/results-web-simple-validation/`。
