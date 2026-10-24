#!/usr/bin/env python3
"""Plot terminal-adaptation costs and conditional efficiency, retaining failures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark_nuts_warmup import progress_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    root=a.directory;s=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    labels=[v[0] for v in protocol['variants']];colors=['#777777','#0072b2','#d55e00']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for j,(label,color) in enumerate(zip(labels,colors)):
        for r in [r for r in s['records'] if r['label']==label]:
            rows,_=progress_rows(root/f"{label}-{r['replicate']}/chain_0_progress.txt",45)
            axes[0].plot(np.arange(len(rows)),[float(v[6]) for v in rows],color=color,alpha=.6,
                label=label if r['replicate']==0 else None)
            formal=r['production']
            if formal['transitions']:
                axes[1].scatter(j+(r['replicate']-1.5)*.06,formal['mean_evaluations'],color=color,
                    marker='o' if r['complete'] else 'x')
        g=s['groups'][label]
        if g['complete']:
            if protocol.get('compare_metrics'):
                axes[2].bar(j-.18,g['min_bulk_per_1000_total'],width=.35,color=color,label='Bulk' if j==0 else None)
                axes[2].bar(j+.18,g['min_tail_per_1000_total'],width=.35,color=color,alpha=.45,hatch='//',label='Tail' if j==0 else None)
            else:axes[2].bar(j,g['min_bulk_per_1000_total'],color=color)
        else:axes[2].text(j,0,'Incomplete',rotation=90,ha='center',va='bottom')
    axes[0].set(yscale='log',xlabel='Transition (warmup + formal)',ylabel='Step size');axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Formal evaluations / state');axes[2].set(ylabel='Minimum bulk ESS / 1,000 total evaluations')
    for ax in axes[1:]:
        ax.set_xticks(range(len(labels)),labels,rotation=15);ax.set_xlim(-.5,len(labels)-.5)
    if not any(g['complete'] for g in s['groups'].values()):
        axes[2].set_ylim(0,1);axes[2].set_yticks([])
    if protocol.get('compare_metrics'):
        axes[2].set_ylabel('Minimum ESS / 1,000 total evaluations');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2,axis='y')
    title='Same checkpoint: before / after mass update' if protocol.get('compare_metrics') else 'Same checkpoint and final mass'
    fig.suptitle(f"{title}; 4 random streams per setting, up to {protocol['draws']} formal states each")
    fig.text(.5,.015,'Crosses: incomplete runs. Total cost includes search and warmup; incomplete groups have no pooled ESS.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.92));fig.savefig(root/'terminal-step.png',dpi=160);fig.savefig(root/'terminal-step.pdf')


if __name__=='__main__':main()
