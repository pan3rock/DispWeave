#!/usr/bin/env python3
"""Read-only structural diagnostics of existing K=4 PT; no forward calls or sampling."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark_crust_depth import digest
from joint_results import save_json
from posterior_diagnostics import diagnose


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/rj-pt-k4'))
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    paths=[source/f'rj/chain_{c}_models.txt' for c in range(4)]
    a=np.stack([np.loadtxt(p) for p in paths]);assert a.shape==(4,25000,11)
    x=a[:,:,4:];z=x[:,:,:3];vs=x[:,:,3:];delta=np.diff(vs,axis=2)
    # Descriptive structural label, NOT an inferred number of posterior modes.
    branch=np.argmax(np.abs(delta),axis=2)
    dominant_z=np.take_along_axis(z,branch[:,:,None],axis=2)[:,:,0]
    means={}
    tops=np.concatenate([np.zeros((*z.shape[:2],1)),z],axis=2)
    bottoms=np.concatenate([z,np.full((*z.shape[:2],1),5.)],axis=2)
    for lo,hi in [(0,1),(1,2),(2,5)]:
        length=np.maximum(0,np.minimum(bottoms,hi)-np.maximum(tops,lo))
        means[f'mean_vs_{lo}_{hi}_km']=(length*vs).sum(axis=2)/(hi-lo)
    features=dict(log_likelihood=a[:,:,2],dominant_interface_depth=dominant_z,**means)
    for j in range(3):features[f'dominant_jump_at_interface_{j+1}']=(branch==j).astype(float)
    # Fixed pooled 5% cutoff exposes the existing interface-3 tail-ESS failure.
    cutoff=float(np.quantile(z[:,:,2],.05));features['interface_3_lower_5pct']=(z[:,:,2]<=cutoff).astype(float)
    report=dict(source=str(source),hashes={str(p.relative_to(source)):digest(p) for p in paths},
        interpretation='Exploratory diagnostics of unaccepted chains; branch labels are largest absolute Vs contrast, not certified modes or posterior masses.',
        interface_3_tail_cutoff_km=cutoff,features={},parameters=[],temperatures=[],branches=[])
    for name,val in features.items():
        report['features'][name]=dict(**diagnose(val),chain_means=val.mean(axis=1).tolist(),
            block_means=val.reshape(4,10,2500).mean(axis=2).tolist())
    names=[f'interface_{j+1}_km' for j in range(3)]+[f'vs_{j+1}' for j in range(4)]
    for j,name in enumerate(names):
        report['parameters'].append(dict(name=name,**diagnose(x[:,:,j]),chain_means=x[:,:,j].mean(axis=1).tolist()))
    for j in range(3):
        selected=x[branch==j];corr=np.corrcoef(selected,rowvar=False)
        indices=np.triu_indices(7,1);order=np.argsort(np.abs(corr[indices]))[::-1][:3]
        report['branches'].append(dict(interface=j+1,count=len(selected),chain_fractions=(branch==j).mean(axis=1).tolist(),
            mean_model=selected.mean(axis=0).tolist(),median_log_likelihood=float(np.median(a[:,:,2][branch==j])),
            strongest_correlations=[dict(a=names[indices[0][k]],b=names[indices[1][k]],r=float(corr[indices][k])) for k in order]))
    for c in range(4):
        d=source/f'rj/ladder_{c}';states=np.stack([np.loadtxt(d/f'temperature_{t}_models.txt') for t in range(4)])
        before_swaps=states.copy();swap_rows=np.loadtxt(d/'swaps.txt')
        for row in swap_rows:
            i=int(row[0])-2500
            if i>=0 and row[9]:
                lo,hi=map(int,row[2:4]);before_swaps[[lo,hi],i]=before_swaps[[hi,lo],i]
        for t in range(4):
            post=states[t];pre=before_swaps[t];tr=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1)[2500:]
            pb=np.argmax(np.abs(np.diff(post[:,7:],axis=1)),axis=1)
            lb=np.argmax(np.abs(np.diff(pre[:,7:],axis=1)),axis=1)
            operations={}
            for move in ['interface','velocity']:
                mask=tr[:,1]==move;accepted=mask&(tr[:,5]=='1')
                operations[move]=dict(proposals=int(mask.sum()),acceptance=float(accepted.sum()/mask.sum()),
                    outside_support=int(np.sum(mask&(tr[:,2]=='outside_support'))))
            # Exclude first retained transition because previous warmup state is not saved.
            local_changes=int(np.sum(lb[1:]!=pb[:-1]));exchange_changes=int(np.sum(pb[1:]!=lb[1:]))
            oldtail=post[:-1,6]<=cutoff;localtail=pre[1:,6]<=cutoff;posttail=post[1:,6]<=cutoff
            report['temperatures'].append(dict(ladder=c,temperature=[1,2,4,8][t],moves=operations,
                branch_fractions=[float(np.mean(pb==j)) for j in range(3)],
                local_branch_changes=local_changes,swap_branch_changes=exchange_changes,
                local_tail_changes=int(np.sum(localtail!=oldtail)),swap_tail_changes=int(np.sum(posttail!=localtail))))
    save_json(out/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');colors=['#0077bb','#aa4499','#ee7733','#777777']
    for c,color in enumerate(colors):
        axes[0,0].plot(np.arange(10)*2500+1250,z[c,:,2].reshape(10,2500).mean(axis=1),color=color,label=f'Chain {c}')
        axes[0,1].plot(np.arange(10)*2500+1250,report['features']['interface_3_lower_5pct']['block_means'][c],color=color)
        axes[1,0].bar(np.arange(3)+(c-1.5)*.18,[(branch[c]==j).mean() for j in range(3)],width=.18,color=color)
        axes[1,1].plot(np.arange(10)*2500+1250,features['mean_vs_2_5_km'][c].reshape(10,2500).mean(axis=1),color=color)
    axes[0,0].set(ylabel='Interface 3 block mean (km)',xlabel='Retained iteration');axes[0,0].legend(fontsize=8)
    axes[0,1].set(ylabel=f'Fraction: interface 3 <= {cutoff:.3f} km',xlabel='Retained iteration')
    axes[1,0].set(xticks=range(3),xticklabels=['Interface 1','Interface 2','Interface 3'],ylabel='Observed fraction',xlabel='Location of largest absolute Vs jump')
    axes[1,1].set(ylabel='2–5 km average Vs block mean (km/s)',xlabel='Retained iteration')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('K=4 PT remaining bottleneck | exploratory, posterior NOT accepted')
    fig.savefig(out/'bottleneck.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ['hashes','temperatures']},indent=2))

if __name__=='__main__':main()
