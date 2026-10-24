#!/usr/bin/env python3
"""Audit saved PT initialization/warmup states without any sampling replay."""
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np
from benchmark_crust_depth import forward
from joint_results import read_config,write_config,save_json
from rj_correlated_density import FrozenDensities

out=Path(sys.argv[1]).resolve();work=out/'warmup-audit';work.mkdir(exist_ok=False)
cfg=read_config(out/'rj.toml');density=FrozenDensities(cfg);options=cfg['rjmcmc'];warm=options['warmup'];cold=options['tempering'].get('cold_start_steps',0);temps=options['tempering']['temperatures']
truth=read_config(out/'truth.toml');obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt');_,ld=np.linalg.slogdet(cov)
def rows(path):return [np.fromstring(line,sep=' ') for line in path.read_text().splitlines() if line and not line.startswith('#')]
def insertion(r):return np.maximum(np.diff(np.r_[0,r[4:3+int(r[1])],5])-.6,0).sum()
checks=0;accepts=0;rejected=0;error=0;stages=[];loglikes=[]
for c in range(4):
 d=out/f'rj/ladder_{c}';state=[rows(d/f'temperature_{t}_warmup_models.txt') for t in range(4)]
 permutations=np.tile(np.arange(4),(warm,1));swaps=np.loadtxt(d/'swaps.txt');assert not np.any(swaps[:,0]<cold)
 for row in swaps:
  i=int(row[0])
  if i>=warm:continue
  l,h=map(int,row[2:4]);expected=[row[7],row[6]] if row[9] else [row[6],row[7]]
  np.testing.assert_array_equal([state[l][i+1][2],state[h][i+1][2]],expected)
  if row[9]:permutations[i,l],permutations[i,h]=h,l
 for t,temp in enumerate(temps):
  series=state[t];assert len(series)==warm+1
  initial=cfg['rj_starts'][c];np.testing.assert_array_equal(series[0][4:],initial['interfaces']+initial['vs'])
  for i,r in enumerate(series):
   k=int(r[1]);assert r[0]==i-1 and len(r)==2*k+3 and np.isfinite(r).all() and 1<=k<=4
   assert np.all(np.diff(np.r_[0,r[4:3+k],5])>=.3) and np.all((r[3+k:]>=.8)&(r[3+k:]<=2.7))
   prior=-math.log(4)+math.lgamma(k)-(k-1)*math.log(5-.3*k)-k*math.log(1.9)
   assert abs(prior-r[3])<1e-12;checks+=1
  tr=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1)
  for i in range(warm):
   before=series[i];local=state[permutations[i,t]][i+1];post=series[i+1];trace=tr[i]
   assert int(trace[6])==before[1] and int(trace[7])==local[1] and int(trace[10])==post[1]
   if trace[5]=='0':np.testing.assert_array_equal(before[1:],local[1:]);rejected+=1
   else:
    k,kk=int(before[1]),int(local[1]);h=0;move=trace[1]
    if move=='birth':assert kk==k+1;h=math.log(insertion(before)*1.9/k)
    elif move=='death':assert kk==k-1;h=-math.log(insertion(local)*1.9/kk)
    elif move=='correlated':assert kk==k;h=density.logq(before)-density.logq(local)
    else:assert kk==k
    beta=1 if i<cold else 1/temp
    value=beta*(local[2]-before[2])+local[3]-before[3]+h
    assert abs(value-float(trace[8]))<1e-8;accepts+=1
  # Warmup/formal boundary: inverse any exchange at first formal iteration.
  first=[rows(d/f'temperature_{j}_models.txt')[0] for j in range(4)]
  for swap in swaps[swaps[:,0]==warm]:
   if swap[9]:l,h=map(int,swap[2:4]);first[l],first[h]=first[h],first[l]
  if tr[warm,5]=='0':np.testing.assert_array_equal(series[-1][1:],first[t][1:])
  # Independent physical likelihood at the end of the cold initialization phase.
  probe=cold if cold else warm
  row=series[probe];k=int(row[1]);v=row[3+k:];config=copy.deepcopy(truth)
  config['model']=dict(depth=[0.,*row[4:3+k]],vs=v.tolist(),vp=(1.8*v).tolist(),rho=[2.3]*k)
  name=f'c{c}-t{t}-cold-end';write_config(work/f'{name}.toml',config);pred=forward(out/'dispweave',work/f'{name}.toml',work/name)
  assert all(r[4]=='valid' for r in pred)
  delta=np.array([float(r[3]) for r in pred])-obs[:,1];ll=-.5*(delta@np.linalg.solve(cov,delta)+ld+12*np.log(2*np.pi))
  error=max(error,abs(ll-row[2]));assert abs(ll-row[2])<1e-7
  stages.append(dict(ladder=c,temperature=temp,initial_logL=float(series[0][2]),initialization_end_logL=float(series[cold][2]),probe_iteration=probe-1,probe_logL=float(row[2]),warmup_end_logL=float(series[-1][2]),probe_K=k,
   cold_failures=int(np.sum((tr[:cold,3]=='1')&(tr[:cold,4]=='forward_failure'))),tempered_warmup_failures=int(np.sum((tr[cold:warm,3]=='1')&(tr[cold:warm,4]=='forward_failure')))))
 loglikes.append([[float(r[2]) for r in slot] for slot in state])
report=dict(passed=True,validated_states_including_initial=checks,accepted_warmup_ratios_verified=accepts,rejected_warmup_states_verified=rejected,
 probe_likelihood_recomputations=16,maximum_log_likelihood_error=error,stages=stages,interpretation='All warmup excluded from posterior; initialization changes only discarded transitions, no state selection or resetting.')
save_json(out/'warmup-audit.json',report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
for c,ax in enumerate(axes.flat):
 for t,temp in enumerate(temps):ax.plot(np.arange(-1,warm),loglikes[c][t],label=f'T={temp} slot',lw=.8)
 ax.axvline(cold-.5,color='#222222',ls='--');ax.set(title=f'Ladder {c}',xlabel='Warmup iteration',ylabel='Raw log likelihood',ylim=(-250,40));ax.grid(alpha=.15)
axes[0,0].legend(fontsize=7);fig.suptitle((f'Initialization audit: first {cold} steps all T=1' if cold else 'PT warmup: fixed ladder, no cold initialization')+'; plots clipped at logL=-250')
fig.savefig(out/'warmup.png',dpi=160);plt.close(fig)
print(report)
