#!/usr/bin/env python3
"""Plot the fixed-metric acceptance-target pilot without hiding flagged groups."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_confirmation(directory,summary):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    fields=['mean_difference_in_combined_mcse','rhat','ess_bulk','ess_tail']
    titles=['Absolute mean shift / combined MCSE (descriptive)',
            'Rank / folded split R-hat','Bulk ESS','Tail ESS']
    for (label,g),color in zip(summary['comparison'].items(),['#d55e00','#0072b2']):
        if not g['complete']:continue
        for ax,field in zip(axes.flat,fields):
            ax.plot(np.arange(len(g['parameters'])),[d[field] for d in g['parameters']],
                    '.-',color=color,label=label,lw=1,markersize=4)
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Vs layer index; last four are derived quantities')
        ax.axvspan(44.5,48.5,color='grey',alpha=.1);ax.grid(alpha=.2)
    axes[0,1].axhline(1.01,color='black',ls='--',lw=1)
    for ax in axes[1]:ax.axhline(400,color='black',ls='--',lw=1)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Fixed-metric candidate confirmation: 4 x 1,000 draws per setting; new seed')
    fig.text(.5,.018,'Reference states are reused: mean comparisons are descriptive. Passing these checks does not validate dispersed-start adaptation.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(directory/'step-size-confirmation.png',dpi=160)
    fig.savefig(directory/'step-size-confirmation.pdf')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    summary=json.loads((a.directory/'summary.json').read_text())
    if summary['protocol'].get('confirm_candidates',False):
        plot_confirmation(a.directory,summary)
        return
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
