#!/usr/bin/env python3
"""Saved-state transdimensional PT audit, independent acceptance-ratio reconstruction."""
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np
from benchmark_crust_depth import forward
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose

out=Path(sys.argv[1]).resolve();work=out/'audit';work.mkdir(exist_ok=False)
truth=read_config(out/'truth.toml');obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt')
_,ld=np.linalg.slogdet(cov);checks=0;error=0;accepted_checks=0;cold=[];failure_categories={}
def load(p):return [np.fromstring(s,sep=' ') for s in p.read_text().splitlines() if s and not s.startswith('#')]
def insertion(r):
 k=int(r[1]);return np.maximum(np.diff(np.r_[0,r[4:3+k],5])-.6,0).sum()
for c in range(4):
 d=out/f'rj/ladder_{c}';states=[load(d/f'temperature_{t}_models.txt') for t in range(4)];cold.append(states[0]);pre=[list(s) for s in states]
 for row in np.loadtxt(d/'swaps.txt'):
  i=int(row[0])-2500
  if i>=0 and row[9]:
   l,h=map(int,row[2:4]);pre[l][i],pre[h][i]=pre[h][i],pre[l][i]
 for t,temp in enumerate([1,2,4,8]):
  tr=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1)
  for i in range(1,25000):
   if tr[2500+i,5]!='1':continue
   a,b=states[t][i-1],pre[t][i];move=tr[2500+i,1];ka,kb=int(a[1]),int(b[1]);h=0
   if move=='birth':
    assert kb==ka+1;h=math.log(insertion(a)*1.9/ka)
   elif move=='death':
    assert kb==ka-1;h=-math.log(insertion(b)*1.9/kb)
   else:assert kb==ka
   ratio=(b[2]-a[2])/temp+b[3]-a[3]+h
   assert abs(ratio-float(tr[2500+i,8]))<1e-10;accepted_checks+=1
  # Check every failed evaluation has all 12 observation rows and a non-valid row.
  failed=[s.split() for s in (d/f'temperature_{t}_failed_models.txt').read_text().splitlines() if s and not s.startswith('#')]
  rows=[s.split() for s in (d/f'temperature_{t}_failed_rows.txt').read_text().splitlines() if s and not s.startswith('#')]
  groups={}
  for r in rows:groups.setdefault(int(r[0]),[]).append(r)
  assert set(groups)=={int(r[0]) for r in failed}
  for f in failed:
   rs=groups[int(f[0])];assert len(rs)==12 and [int(r[1]) for r in rs]==list(range(12))
   bad=[r[2] for r in rs if r[2]!='valid'];assert bad
   for status in bad:failure_categories[status]=failure_categories.get(status,0)+1
  for i in [0,12500,24999]:
   r=states[t][i];k=int(r[1]);v=r[3+k:];config=copy.deepcopy(truth)
   config['model']=dict(depth=[0.,*r[4:3+k]],vs=v.tolist(),vp=(1.8*v).tolist(),rho=[2.3]*k)
   name=f'c{c}-t{t}-i{i}';write_config(work/f'{name}.toml',config);prediction=forward(out/'dispweave',work/f'{name}.toml',work/name)
   assert all(row[4]=='valid' for row in prediction)
   residual=np.array([float(row[3]) for row in prediction])-obs[:,1]
   ll=-.5*(residual@np.linalg.solve(cov,residual)+ld+12*np.log(2*np.pi))
   error=max(error,abs(ll-r[2]));assert abs(ll-r[2])<1e-7;checks+=1
summary=json.loads((out/'summary.json').read_text());layers=np.array([[r[1] for r in chain] for chain in cold])
for p in summary['layer_probabilities']:
 diag=diagnose((layers==p['k']).astype(float))
 for key in ['rhat','ess_bulk','ess_tail']:
  if p['diagnostics'][key] is None:assert not np.isfinite(diag[key])
  else:assert diag[key]==p['diagnostics'][key]
for p in summary['depth_profiles']:
 # Explicit interval scan, independent of searchsorted in the analysis.
 val=np.empty((4,25000))
 for c,chain in enumerate(cold):
  for i,r in enumerate(chain):
   k=int(r[1]);z=[0.,*r[4:3+k],math.inf];v=r[3+k:]
   val[c,i]=next(v[j] for j in range(k) if z[j]<=p['depth']<z[j+1])
 diag=diagnose(val)
 for key in ['rhat','ess_bulk','ess_tail']:assert diag[key]==p[key]
report=dict(passed=True,accepted_local_ratios_verified=accepted_checks,raw_likelihood_recomputations=checks,maximum_log_likelihood_error=error,
 failed_observation_status_counts=failure_categories,all_failure_rows_matched=True,all_layer_and_depth_diagnostics_recomputed=True,
 limitation='Recomputed retained states verify caches, not global root completeness; failed states retain unresolved physical/numerical support status.')
save_json(out/'independent-audit.json',report);print(report)
