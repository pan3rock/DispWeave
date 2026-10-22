#!/usr/bin/env python3
"""Create an offline M7 results page from saved diagnostics and rendered figures."""
import argparse
import base64
import html
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--results', type=Path, default=Path('results/m7'))
p.add_argument('--output', type=Path, default=Path('web/m7-demo/dist/index.html'))
a = p.parse_args()
sections = []
for name, title in [('long', '等权层数先验'), ('prior-contrast', '较小层数优先的先验')]:
    report = json.loads((a.results / f'{name}-diagnostics.json').read_text())
    config = (a.results / f'{name}-rj/config.toml').read_text()
    metadata = (a.results / f'{name}-rj/metadata.txt').read_text()
    rows = []
    for x in report['layer_probabilities']:
        d = x['diagnostics']
        rhat = '不可估计' if d['rhat'] is None else f"{d['rhat']:.4f}"
        ess = '不可估计' if d['ess_bulk'] is None else f"{d['ess_bulk']:.0f}"
        rows.append(f"<tr><td>{x['k']}</td><td>{x['probability']:.2%}</td><td>{rhat}</td><td>{ess}</td></tr>")
    failures = sum(s['forward_failures'] + s['numerical_failures'] for s in report['sampler'])
    figures = []
    for suffix, caption in [('summary', '层数分布、逐深度 Vs 可信区间与界面占据概率'), ('predictive', '潜在频散曲线与重复观测的逐点预测区间')]:
        content = base64.b64encode((a.results / f'{name}-{suffix}.png').read_bytes()).decode('ascii')
        figures.append(f'<figure><img src="data:image/png;base64,{content}" alt="{title}：{caption}"><figcaption>{caption}</figcaption></figure>')
    sections.append(f'''<section id="{name}"><h2>{title}</h2>
<p class="warning">诊断：{html.escape(report['status'])} · 正式采样中 {failures:,} 次目标评估失败</p>
<p>{len(report['chains'])} 条链，每链保留 {report['draws_per_chain']:,} 个样本。K 包含无限半空间。</p>
<table><thead><tr><th>层数 K</th><th>估计后验概率</th><th>R-hat</th><th>bulk ESS</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(figures)}
<details><summary>查看本次实验的完整配置与运行元数据</summary><pre>{html.escape(config + chr(10) + metadata)}</pre></details></section>''')
page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DispWeave · M7 跨维反演实验</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f5;color:#163338;font:17px/1.75 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:40px 24px}header{border-top:5px solid #147b78;padding:24px 0}h1{font-size:clamp(28px,5vw,44px);line-height:1.2}h2{font-size:26px}a{color:#086c70}nav{display:flex;gap:24px;flex-wrap:wrap}section{background:white;margin:32px 0;padding:28px;border:1px solid #d7e2e2;border-radius:12px}.eyebrow{letter-spacing:.12em;font-size:13px;font-weight:700}.warning{background:#fff1dc;color:#7a4300;padding:12px 16px;border-left:4px solid #ce8217}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}figure{margin:28px 0}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:14px;color:#52666a;text-align:center}summary{cursor:pointer;color:#086c70}pre{overflow:auto;font-size:13px;background:#f3f5f5;padding:16px}footer{font-size:14px;color:#52666a}@media(max-width:600px){main{padding:20px 12px}section{padding:16px}td,th{padding:6px;font-size:14px}}@media print{section{break-inside:avoid}}
</style></head><body><main><header><div class="eyebrow">DISPWEAVE / M7 / SYNTHETIC EXPERIMENT</div><h1>层数也作为未知量</h1><p>独立 RJMCMC 的合成频散实验：同时探索层数、界面位置和 Vs，并检查结果对层数先验的依赖。</p><nav><a href="#long">等权先验</a><a href="#prior-contrast">先验对照</a><a href="#limits">如何解读</a></nav></header>
<section><h2>同一组数据，两种层数先验</h2><p>真值为 1.5 km 厚、Vs=1.0 km/s 的上层与 Vs=2.3 km/s 的半空间。六个基阶频散观测覆盖 0.15–0.9 Hz，边缘误差为 0.03 km/s，并保留误差相关性。</p><p>两组实验固定 Vp/Vs=1.8、密度 2.3 g/cm³，允许 1–4 层和非单调 Vs。第二组只将层数先验权重从 [1,1,1,1] 改为 [1,1,0.5,0.25]，其他设置保持一致。</p></section>
''' + ''.join(sections) + '''
<section id="limits"><h2>如何解读这些结果</h2><p>层数概率随先验发生明显变化。这组基阶观测对结构复杂度的约束有限，拟合频散曲线并不意味着层数唯一。</p><p class="warning">这些实验仍有正演失败提议。失败已记录并拒绝，不能把它们全部解释成物理模态不存在，也不能将结果宣称为整个先验支持域上的已认证后验。</p><ul><li>K=1 未被正式样本访问；表中的 0% 不是零后验质量的证明。</li><li>R-hat 与 ESS 未触发阈值不代表已排除未发现的后验模态。</li><li>Vs 区间是逐深度可信区间，不是深度分辨率，也不是同时置信带。</li><li>界面图表示有限深度区间内至少有一个界面的概率，不是概率密度。</li><li>每组后验预测取 900 个保存模型，均有效；成功预测不能抵消提议阶段的正演失败。</li></ul></section>
<footer>本页由保存的实验文件生成，所有图片嵌入 HTML，可离线打开。图表为合成实验结果；没有上传或发布。重新生成：python scripts/build_rj_demo_page.py</footer></main></body></html>'''
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(page)
print(a.output)
