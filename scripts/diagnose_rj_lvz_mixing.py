#!/usr/bin/env python3
"""Saved-trajectory LVZ bottleneck analysis; consumes verified RNG reconstruction."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_rj_remaining import runs
from joint_results import clean


source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
check=json.loads((out/'reconstruction.json').read_text())
assert check['transitions_verified']==440000 and check['forward_calls']==check['new_samples']==0
hashes={}
for pattern in ['rj.toml','rj/ladder_*/temperature_*_models.txt','rj/ladder_*/temperature_*_warmup_models.txt',
                'rj/ladder_*/temperature_*_failed_models.txt','rj/ladder_*/temperature_*_local_trace.txt','rj/ladder_*/swaps.txt']:
    for path in source.glob(pattern):
        with path.open('rb') as f:hashes[str(path.relative_to(source))]=hashlib.file_digest(f,'sha256').hexdigest()
def read(path):
    data=path.read_bytes();hashes[str(path.relative_to(source))]=hashlib.sha256(data).hexdigest()
    return [s.split() for s in data.decode().splitlines() if s.strip() and not s.startswith('#')]
def describe(x):
    x=np.asarray(x)
    return dict(count=len(x),mean=float(np.mean(x)),quantiles=np.quantile(x,[.05,.5,.95]).tolist()) if len(x) else None
read(source/'rj.toml')
chains=[[np.array(r,float) for r in read(source/f'rj/chain_{c}_models.txt')] for c in range(4)]
layers=np.array([[int(r[1]) for r in chain] for chain in chains])
groups=[];scatter={};residence=[];blocks=[];events=[]
for c,chain in enumerate(chains):
    rs=runs(layers[c]);residence.append(dict(chain=c,by_k={str(k):dict(lengths=describe([r['length'] for r in rs if r['k']==k]),
        longest=sorted([r for r in rs if r['k']==k],key=lambda r:-r['length'])[:5]) for k in [3,4]}))
    for b,idx in enumerate(np.array_split(np.arange(25000),5)):
        blocks.append(dict(chain=c,block=b,k3=float(np.mean(layers[c,idx]==3))))
    for k in [3,4]:
        vals=[];indices=[];ll=[];state_indices=[];coordinates=[]
        for i,r in enumerate(chain):
            if r[1]!=k:continue
            coordinates.append(r[4:]);v=r[3+k:];z=r[4:3+k]
            strength=np.minimum(v[:-2]-v[1:-1],v[2:]-v[1:-1]);eligible=np.flatnonzero(strength>=.1)
            if not len(eligible):continue
            j=eligible[np.argmax(strength[eligible])]+1
            vals.append([z[j-1],z[j],z[j]-z[j-1],v[j]])
            indices.append(int(j));ll.append(r[2]);state_indices.append(i)
        x=np.array(vals);scatter[c,k]=x
        groups.append(dict(chain=c,k=k,k_count=len(coordinates),dip_count=len(x),
            variables=['top_km','bottom_km','thickness_km','dip_vs_km_s'],
            mean=x.mean(axis=0).tolist(),quantiles=np.quantile(x,[.025,.5,.975],axis=0).tolist(),
            pearson=np.corrcoef(x,rowvar=False).tolist(),
            all_coordinate_correlation=np.corrcoef(np.array(coordinates),rowvar=False).tolist(),
            dip_index_fractions={str(j):float(np.mean(np.array(indices)==j)) for j in range(1,k-1)},
            log_likelihood=describe(ll),
            chronological_block_means=[np.mean(x[(np.array(state_indices)>=b*5000)&(np.array(state_indices)<(b+1)*5000)],axis=0).tolist() for b in range(5)]))
    folder=source/f'rj/ladder_{c}'
    tr=np.array(read(folder/'temperature_0_local_trace.txt'))[2500:]
    before=tr[:,6].astype(int);local=tr[:,7].astype(int);after=tr[:,10].astype(int)
    np.testing.assert_array_equal(after,layers[c]);np.testing.assert_array_equal(before[1:],layers[c,:-1])
    events.append(dict(chain=c,local_3_to_4=int(np.sum((before==3)&(local==4))),local_4_to_3=int(np.sum((before==4)&(local==3))),
        swap_3_to_4=int(np.sum((local==3)&(after==4))),swap_4_to_3=int(np.sum((local==4)&(after==3)))))

reconstructed=np.loadtxt(out/'proposals.tsv',dtype=str,skiprows=1)
formal=reconstructed[reconstructed[:,3]=='0'];stats=[]
for t in [1,2,4,8]:
    for move,k in [('birth',3),('death',4),*[("correlated",k) for k in range(1,5)]]:
        for chain in [-1,0,1,2,3]:
            mask=(formal[:,1].astype(float)==t)&(formal[:,4]==move)&(formal[:,8]==str(k))
            if chain>=0:mask &= formal[:,0]==str(chain)
            a=formal[mask];valid=a[:,6]=='valid';accept=a[:,7]=='1';reject=valid&~accept
            outside=a[:,5]=='outside_support';unavailable=a[:,5]=='unavailable'
            failure=np.isin(a[:,6],['forward_failure','numerical_failure'])
            assert len(a)==np.sum(outside)+np.sum(unavailable)+np.sum(failure)+np.sum(valid)
            dll=a[valid,10].astype(float);lp=a[valid,11].astype(float);h=a[valid,12].astype(float);ratio=a[valid,9].astype(float)
            np.testing.assert_allclose(dll+lp+h,ratio,atol=1e-10,rtol=1e-12)
            rejected=a[reject];dr=rejected[:,10].astype(float);hr=rejected[:,12].astype(float)
            if move=='correlated':assert np.max(abs(lp),initial=0)<1e-12
            stats.append(dict(temperature=t,chain=chain,move=move,from_k=k,attempts=len(a),accepted=int(accept.sum()),
                outside=int(outside.sum()),unavailable=int(unavailable.sum()),forward_failure=int(np.sum(a[:,6]=='forward_failure')),
                numerical_failure=int(np.sum(a[:,6]=='numerical_failure')),valid=int(valid.sum()),valid_rejected=int(reject.sum()),
                valid_delta_logL_over_T=describe(dll),valid_prior=describe(lp),valid_log_q_ratio=describe(h),valid_total_log_ratio=describe(ratio),
                rejected_delta_logL_over_T=describe(dr),rejected_log_q_ratio=describe(hr),
                rejected_likelihood_negative=int(np.sum(dr<0)),rejected_likelihood_below_minus5=int(np.sum(dr<-5)),
                rejected_likelihood_loss_dominates_negative_q=int(np.sum(np.maximum(-dr,0)>np.maximum(-hr,0))),
                rejected_q_more_negative_than_likelihood=int(np.sum(np.maximum(-hr,0)>np.maximum(-dr,0))),
                rejected_likelihood_nonnegative_but_q_negative=int(np.sum((dr>=0)&(hr<0))),
                rejected_q_nonnegative=int(np.sum(hr>=0))))

# Reconcile every reconstructed aggregate with the original text traces.
for s in stats:
    if s['chain']<0:continue
    t=[1,2,4,8].index(s['temperature']);c=s['chain']
    tr=np.array(read(source/f'rj/ladder_{c}/temperature_{t}_local_trace.txt'))[2500:]
    a=tr[(tr[:,1]==s['move'])&(tr[:,6]==str(s['from_k']))]
    assert len(a)==s['attempts'] and int(np.sum(a[:,5]=='1'))==s['accepted']
    assert int(np.sum(a[:,2]=='outside_support'))==s['outside']
    assert int(np.sum(a[:,4]=='forward_failure'))==s['forward_failure']
for path,sha in hashes.items():assert hashlib.sha256((source/path).read_bytes()).hexdigest()==sha
summary=json.loads((source/'summary.json').read_text())
assert sum(s['attempts'] for s in stats if s['chain']==-1 and s['move']=='correlated')==sum(s['production']['moves']['correlated']['proposals'] for s in summary['sampler'])
assert sum(s['accepted'] for s in stats if s['chain']==-1 and s['move']=='correlated')==sum(s['production']['moves']['correlated']['accepted'] for s in summary['sampler'])
report=dict(groups=groups,blocks=blocks,residence=residence,cold_transition_sources=events,proposal_stats=stats,
    reconstruction=check,input_hashes=hashes,new_forward_evaluations=0,new_samples=0,
    interpretation='Existing-production exploratory diagnosis. Conditional dip summaries retain repeats, no compressed-time Rhat. Rejected-candidate likelihood change is reconstructed algebraically from recorded ratio and RNG-reconstructed q, not newly evaluated. Negative-term comparisons are descriptive, not causal ablations; Hastings term must remain.')
(out/'diagnosis.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(2,3,figsize=(15,8.5),constrained_layout=True)
colors=['#0072B2','#CC79A7','#E69F00','#555555']
for i,k in enumerate([3,4]):
    for c,color in enumerate(colors):
        x=scatter[c,k];sample=x[np.linspace(0,len(x)-1,min(len(x),600),dtype=int)]
        axes[i,0].scatter(sample[:,0],sample[:,1],s=5,color=color,alpha=.23,label=f'Chain {c}')
        axes[i,1].scatter(sample[:,2],sample[:,3],s=5,color=color,alpha=.23)
    axes[i,0].scatter([.8],[1.8],marker='x',s=80,color='#228833',lw=2,label='Target')
    axes[i,1].scatter([1.],[1.],marker='x',s=80,color='#228833',lw=2)
    axes[i,0].set(title=f'K={k}, conditional on >=0.1 km/s dip',xlabel='Dip top (km)',ylabel='Dip bottom (km)')
    axes[i,1].set(title=f'K={k}: thickness / speed trade-off',xlabel='Dip thickness (km)',ylabel='Dip Vs (km/s)')
axes[0,0].legend(fontsize=8,markerscale=2)
for c,color in enumerate(colors):
    axes[0,2].plot(range(1,6),[x['k3'] for x in blocks if x['chain']==c],'o-',color=color,label=f'Chain {c}')
axes[0,2].set(title='K=3 fraction in consecutive blocks',xlabel='5000-draw block',ylabel='Fraction');axes[0,2].legend(fontsize=8)
categories=['accepted','outside','forward_failure','valid_rejected'];labels=['Accepted','Outside support','Forward failure','Valid, MH rejected'];category_colors=['#228833','#999999','#E69F00','#7B3294']
for i,t in enumerate([1,2,4,8]):
    parts=[s for s in stats if s['temperature']==t and s['chain']==-1 and s['move']=='correlated'];total=sum(s['attempts'] for s in parts);left=0
    for name,label,color in zip(categories,labels,category_colors):
        fraction=sum(s[name] for s in parts)/total;axes[1,2].barh(i,fraction,left=left,color=color,label=label if i==0 else None);left+=fraction
axes[1,2].set(yticks=range(4),yticklabels=['T=1','T=2','T=4','T=8'],xlabel='Fraction of correlated attempts',title='Joint proposal outcomes');axes[1,2].legend(fontsize=7)
for ax in axes.flat:ax.grid(alpha=.15)
fig.suptitle('LVZ mixing diagnosis | existing trajectories + verified proposal reconstruction | no new forwards')
fig.savefig(out/'diagnosis.png',dpi=160);plt.close(fig)
print('Conditional geometry, layer transitions and reconstructed rejection contributions verified.')
