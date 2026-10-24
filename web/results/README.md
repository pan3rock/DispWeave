# 本地结果页

直接用浏览器打开 `index.html`，无需构建、联网或 JavaScript，不发布站点。

主区按算例展示三层低速夹层和两层实验的结构与数据拟合，固定层地壳及六层结果折叠保留。
目标使用绿色；其余颜色以各图图例为准。LM/NUTS固定界面与RJ变层数的条件在图旁注明。
完整后验未验收和结构非唯一性在主区保留，不以拟合接近代替混合验收。

[阶段结论](../../docs/rj-predictive-closeout.md#当前阶段总览) 与
[复现索引](../../docs/rj-results-index.json) 包含配置、种子、输入hash和运行命令。
`history.html` 保留旧页面全部诊断；其中历史“下一步”不代表当前待办。

## 更新和查看

已有图片更新后刷新页面即可，无需重建或重新打包。
`results/` 为本地忽略目录，新克隆需要恢复结果；网页不是包含原始数据的独立归档。
地壳图可用 `scripts/refresh_crust_results.py` 从已保存缓存更新，无需采样或正演。

如需HTTP查看，从项目根目录执行：

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory .
```

打开 `http://127.0.0.1:8000/web/results/`，结束时Ctrl+C。
本次整理不启动服务、不发布、不运行新链；校验记录保存于 `results/rj-stage-closeout/`。
