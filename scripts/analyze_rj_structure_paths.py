#!/usr/bin/env python3
"""Existing-state comparison of K4 LVZ representations and replica-following routes."""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from joint_results import clean


out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
sources={'transdim':Path('results/rj-lvz-transfer/production'),
         'fixed_independent':Path('results/rj-local-k4/independent'),
         'fixed_local':Path('results/rj-local-k4/local')}
hashes={}
def text(path):
    data=path.read_bytes();hashes[str(path.resolve())]=hashlib.sha256(data).hexdigest();return data.decode()
def rows(path):return [np.fromstring(s,sep=' ') for s in text(path).splitlines() if s and not s.startswith('#')]
def family(r):
    k=int(r[1])
    if k!=4:return f'K{k}'
    v=r[7:];strength=np.minimum(v[:-2]-v[1:-1],v[2:]-v[1:-1])
    if strength.max()<.1:return 'K4_no_dip'
    return 'A' if strength.argmax()==0 else 'B'
def describe(x):
    return dict(count=len(x),mean=float(np.mean(x)),quantiles=np.quantile(x,[.025,.5,.975]).tolist()) if len(x) else None
depth=np.linspace(0,5,61);reports=[]
for name,source in sources.items():
    cold=[rows(source/f'rj/chain_{c}_models.txt') for c in range(4)]
    groups=[]
    for label in ['A','B']:
        values=[];top=[];bottom=[];vs=[];extra_z=[];extra_jump=[];ll=[];counts=[];chain_summaries=[]
        for c,chain in enumerate(cold):
            subset=[r for r in chain if family(r)==label];counts.append(len(subset))
            chain_summaries.append(dict(chain=c,count=len(subset),log_likelihood=describe([r[2] for r in subset])))
            for r in subset:
                z=r[4:7];v=r[7:];j=1 if label=='A' else 2
                values.append(v[np.searchsorted(z,depth,side='right')]);top.append(z[j-1]);bottom.append(z[j]);vs.append(v[j]);ll.append(r[2])
                extra_z.append(z[2] if label=='A' else z[0]);extra_jump.append(abs(v[3]-v[2]) if label=='A' else abs(v[1]-v[0]))
        groups.append(dict(family=label,per_chain_count=counts,log_likelihood=describe(ll),top=describe(top),bottom=describe(bottom),dip_vs=describe(vs),
            extra_interface_depth=describe(extra_z),extra_velocity_jump=describe(extra_jump),
            profiles=np.quantile(values,[.025,.5,.975],axis=0).tolist(),per_chain=chain_summaries))
    direct=Counter();bridges=Counter();arrivals=Counter();examples=[];route_durations={};replica_counts=[]
    for c in range(4):
        folder=source/f'rj/ladder_{c}'
        states=[rows(folder/f'temperature_{t}_models.txt') for t in range(4)]
        initial=[rows(folder/f'temperature_{t}_warmup_models.txt')[-1] for t in range(4)]
        labels=np.array([[family(r) for r in s] for s in states]);pre=[list(s) for s in states]
        for a,b in zip(states[0],cold[c]):np.testing.assert_array_equal(a,b)
        swaps=rows(folder/'swaps.txt')
        for r in reversed(swaps):
            i=int(r[0])-2500
            if i>=0 and r[9]:
                lo,hi=map(int,r[2:4]);pre[lo][i],pre[hi][i]=pre[hi][i],pre[lo][i]
        for t in range(4):
            trace=[s.split() for s in text(folder/f'temperature_{t}_local_trace.txt').splitlines() if s and not s.startswith('#')][2500:]
            before=[initial[t],*states[t][:-1]]
            for i,tr in enumerate(trace):
                a,b=family(before[i]),family(pre[t][i])
                assert int(tr[6])==before[i][1] and int(tr[7])==pre[t][i][1]
                if tr[5]=='0':np.testing.assert_array_equal(before[i][1:],pre[t][i][1:])
                if {a,b}=={'A','B'}:
                    assert tr[5]=='1';direct[f'T{[1,2,4,8][t]}:{tr[1]}:{a}->{b}']+=1
        ids=np.array(rows(folder/'replicas.txt'),int)
        assert ids.shape==(27500,6)
        for t in range(4):
            previous=family(initial[t]);last=previous if previous in ['A','B'] else None
            # Replica id at warmup end can differ from its temperature slot.
            replica=int(ids[2499,2+t]);last_i=-1;intermediate=set();count=0
            for i in range(25000):
                slot=int(np.flatnonzero(ids[2500+i,2:]==replica)[0]);label=labels[slot,i]
                if label in ['A','B']:
                    if last is not None and label!=last:
                        kind=('via_K3' if 'K3' in intermediate else 'via_other_K' if any(x in intermediate for x in ['K1','K2'])
                              else 'via_K4_no_dip' if intermediate else 'direct_K4')
                        key=f'{kind}:{last}->{label}';bridges[key]+=1;arrivals[f'{kind}:T{[1,2,4,8][slot]}']+=1;count+=1;route_durations.setdefault(key,[]).append(i-last_i)
                        if len([e for e in examples if e['route']==kind])<4:
                            examples.append(dict(ladder=c,replica=replica,from_family=last,to_family=label,route=kind,
                                from_draw=last_i,to_draw=i,arrival_temperature=[1,2,4,8][slot],intermediate_labels=sorted(intermediate)))
                    last=label;last_i=i;intermediate=set()
                else:intermediate.add(label)
            replica_counts.append(dict(ladder=c,replica=replica,family_switches=count))
    # A direct family switch must be the same event whether followed locally or by replica.
    assert sum(direct.values())==sum(v for k,v in bridges.items() if k.startswith('direct_K4:'))
    reports.append(dict(source=name,groups=groups,local_direct_switches=dict(direct),replica_routes=dict(bridges),
        route_durations={k:describe(v) for k,v in route_durations.items()},route_arrival_temperatures=dict(arrivals),examples=examples,per_replica=replica_counts))

# Reuse the already computed, evenly spaced transdimensional predictive sample.
predict=Path('results/rj-lvz-transfer/predict');models=rows(predict/'selected_models.txt')
status=[s.split() for s in text(predict/'draws.txt').splitlines() if s and not s.startswith('#')]
latent=np.array(rows(predict/'latent.txt'));obs=np.array(rows(predict/'observations.txt'));cov=np.array(rows(predict/'covariance.txt'))
prediction=[];quantiles={}
for label in ['A','B']:
    selected=[i for i,r in enumerate(models) if family(r)==label]
    failed=[i for i in selected if status[i][3]!='valid']
    item=dict(family=label,selected_count=len(selected),failed_rows=failed,
        per_chain_count=[sum(int(status[i][1])==c for i in selected) for c in range(4)])
    if selected and not failed:
        q=np.quantile(latent[selected],[.025,.5,.975],axis=0);quantiles[label]=q
        item.update(latent_quantiles=q.tolist(),observed_chi2=describe([float(status[i][4]) for i in selected]))
    prediction.append(item)
distance={}
if len(quantiles)==2:
    delta=quantiles['A'][1]-quantiles['B'][1];sigma=np.sqrt(np.diag(cov))
    distance=dict(median_curve_difference_max_sigma=float(np.max(abs(delta)/sigma)),
        median_curve_difference_rms_sigma=float(np.sqrt(np.mean((delta/sigma)**2))),
        median_curve_mahalanobis2=float(delta@np.linalg.solve(cov,delta)),
        pointwise_intervals_overlap=int(np.sum(np.maximum(quantiles['A'][0],quantiles['B'][0])<=np.minimum(quantiles['A'][2],quantiles['B'][2]))))
for path,sha in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
report=dict(stages=reports,predictions=prediction,prediction_comparison=distance,depth=depth.tolist(),input_hashes=hashes,
    new_samples=0,new_forward_calls=0,
    definitions='K4 A: strongest >=.1 km/s dip is first interior layer; B: second. Remaining interface is below dip for A, above dip for B. These are labels, not proven isolated modes. Paths follow replicas through temperature swaps; all 16 replicas included. Route ends at next A/B after leaving previous A/B; initial incomplete paths excluded.',
    limitations='Conditional summaries retain repeats; no filtered-time Rhat. Prediction subset comes only from existing transdimensional run, not a new balanced comparison and not a claim of predictive equivalence. Fitted curves alone cannot certify structure or convergence.')
(out/'summary.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(1,4,figsize=(16,5.2),constrained_layout=True)
colors={'A':'#0072B2','B':'#7B3294'}
for i,r in enumerate(reports):
    for g in r['groups']:
        q=np.array(g['profiles']);color=colors[g['family']]
        axes[i].fill_betweenx(depth,q[0],q[2],color=color,alpha=.12)
        axes[i].plot(q[1],depth,color=color,label=f'Family {g["family"]}')
    axes[i].plot([1.6,1.6,1,1,2.3,2.3],[0,.8,.8,1.8,1.8,5],color='#228833',ls=':',label='Target')
    axes[i].set(title=r['source'],xlabel='Conditional Vs (km/s)',ylabel='Depth (km)',ylim=(5,0))
for label,q in quantiles.items():
    axes[3].fill_between(obs[:,0],q[0],q[2],color=colors[label],alpha=.12)
    count=next(p['selected_count'] for p in prediction if p['family']==label)
    axes[3].plot(obs[:,0],q[1],color=colors[label],label=f'Family {label}, n={count} existing draws')
axes[3].errorbar(obs[:,0],obs[:,1],yerr=np.sqrt(np.diag(cov)),fmt='k.',label='Observed +/- sigma')
axes[3].set(title='Existing transdimensional predictions',xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)')
for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Different K4 structure representations | conditional summaries, not certified modes | no new forwards')
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
print(json.dumps(clean(dict(predictions=prediction,distance=distance,routes=[dict(source=r['source'],routes=r['replica_routes']) for r in reports])),indent=2))
