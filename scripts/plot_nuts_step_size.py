#!/usr/bin/env python3
"""Plot the fixed-metric acceptance-target pilot without hiding flagged groups."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    summary=json.loads((a.directory/'summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for kernel,color in [('slice','#0072b2'),('multinomial','#d55e00')]:
        x=[];steps=[];lo=[];hi=[];calls=[];ess_cost=[];ess_time=[];flags=[]
        for target in [90,95,99]:
            label=f'{kernel}-{target}';g=summary['comparison'][label]
            if not g['complete']:continue
            chains=[r for r in summary['chains'] if r['label']==label]
            eps=[r['frozen_step'] for r in chains];x.append(target/100)
            steps.append(np.median(eps));lo.append(min(eps));hi.append(max(eps))
            calls.append(g['formal_evaluations']/800)
            ess_cost.append(g['min_bulk_ess_per_1000_formal_evaluations'])
            ess_time.append(g['min_bulk_ess_per_total_chain_second']);flags.append(not g['numerical_passed'])
        values=[steps,calls,ess_cost,ess_time]
        for ax,y in zip(axes.flat,values):
            ax.plot(x,y,'o-',color=color,label=kernel,lw=1.8)
            for xx,yy,flag in zip(x,y,flags):
                if flag:ax.scatter(xx,yy,s=130,marker='x',color='black',zorder=5)
        if x:axes[0,0].vlines(x,lo,hi,color=color,alpha=.4,lw=3)
    titles=['Frozen step: median and range over 4 chains','Formal target evaluations per retained state',
            'Minimum bulk ESS / 1,000 formal evaluations','Minimum bulk ESS / summed chain seconds']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Target acceptance');ax.set_xticks([.9,.95,.99]);ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Fixed-metric posterior-start pilot: 4 x 200 draws per setting')
    fig.text(.5,.018,'ESS minima cover 45 Vs + 4 derived quantities. Short-chain estimates; time includes 100 adaptation steps.\nBlack crosses mark formal numerical/divergence/depth or movement flags; no flags does not establish convergence.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(a.directory/'step-size-comparison.png',dpi=160)
    fig.savefig(a.directory/'step-size-comparison.pdf')


if __name__=='__main__':main()
