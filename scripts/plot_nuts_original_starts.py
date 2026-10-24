#!/usr/bin/env python3
"""Plot completed-transition costs, including partial original-start runs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark_nuts_warmup import progress_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for c,ax in enumerate(axes.flat):
        for label,path,color in [('Reference target 0.99',a.directory/f'inputs/chain_{c}_progress.txt','#777777'),
                                 ('Candidate target 0.95',a.directory/f'chain-{c}/nuts/chain_0_progress.txt','#0072b2')]:
            rows,_=progress_rows(path,45);rows=rows[:2000]
            if rows:
                y=np.cumsum([int(r[5]) for r in rows])/1000
                ax.plot(np.arange(1,len(rows)+1),y,label=label,color=color,lw=1.6)
                ax.scatter(len(rows),y[-1],color=color,s=20)
        for boundary in [450,950]:ax.axvline(boundary,color='#aaaaaa',ls=':',lw=.8)
        ax.axvline(1000,color='black',ls='--',lw=1)
        ax.set(title=f'Original chain {c}',xlabel='Completed transitions',ylabel='Cumulative evaluations (thousands)',xlim=(0,2000))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Original-start slice NUTS: 1,000 warmup + up to 1,000 formal states')
    fig.text(.5,.015,'Dashed line: end of warmup. Dotted lines: 450 / 950. Endpoints retain incomplete runs; initialization costs excluded.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(a.directory/'original-start-cost.png',dpi=160)
    fig.savefig(a.directory/'original-start-cost.pdf')


if __name__=='__main__':main()
