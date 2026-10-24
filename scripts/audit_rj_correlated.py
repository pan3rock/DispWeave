#!/usr/bin/env python3
"""Verify frozen empirical mixtures and each retained correlated transition, no new targets."""
import json
from pathlib import Path
import sys
import numpy as np
from benchmark_crust_depth import digest
from joint_results import read_config,save_json
from rj_correlated_density import FrozenDensities

out=Path(sys.argv[1]).resolve();cfg=read_config(out/'rj.toml');training=json.loads((out/'training.json').read_text());source=Path(training['source']);q=FrozenDensities(cfg)
def rows(p):return [np.fromstring(s,sep=' ') for s in p.read_text().splitlines() if s and not s.startswith('#')]
for name,h in training['source_hashes'].items():assert digest(source/name)==h
assert digest(out/'rj.toml')==training['mixture_config_sha256']
chains=[rows(source/f'rj/chain_{c}_models.txt') for c in range(4)];fitted=0
for model in cfg['rj_correlated_models']:
 k=model['layers'];scales=np.array([5.]*(k-1)+[1.9]*k);parts=model['components']
 if k==1:assert len(parts)==1 and parts[0]['weight']==1
 else:
  assert len(parts)==5
  for c in range(4):
   x=np.array([r[4:] for r in chains[c] if r[1]==k]);mean=x.sum(axis=0)/len(x);centered=x-mean
   covariance=centered.T@centered/(len(x)-1);expected=.9*covariance+.1*np.diag(np.diag(covariance))+1e-6*np.diag(scales**2)
   np.testing.assert_allclose(parts[c]['mean'],mean,atol=1e-13,rtol=1e-13);np.testing.assert_allclose(parts[c]['covariance'],expected,atol=1e-13,rtol=1e-12);assert parts[c]['weight']==.225;fitted+=1
 broad=parts[-1];np.testing.assert_allclose(broad['covariance'],np.diag(scales**2/12),atol=1e-14,rtol=0)
 np.testing.assert_array_equal(broad['mean'],[5.*(j+1)/k for j in range(k-1)]+[1.75]*k)
 assert broad['weight']==(1 if k==1 else .1)
 assert abs(sum(c['weight'] for c in parts)-1)<1e-15
stats=[];max_error=0;rescues=[]
for c in range(4):
 d=out/f'rj/ladder_{c}';state=[rows(d/f'temperature_{t}_warmup_models.txt')+rows(d/f'temperature_{t}_models.txt') for t in range(4)]
 swaps=np.loadtxt(d/'swaps.txt');perm=np.tile(np.arange(4),(len(state[0])-1,1))
 for r in swaps:
  if r[9]:i,l,h=map(int,r[[0,2,3]]);perm[i,l],perm[i,h]=h,l
 for t,temp in enumerate([1,2,4,8]):
  trace=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1)
  for phase,lo,hi in [('warmup',0,2500),('production',2500,len(trace))]:
   attempts=evaluated=accepted=outside=fail=both_changed=0
   for i in range(lo,hi):
    tr=trace[i]
    if tr[1]!='correlated':continue
    attempts+=1;outside+=tr[2]=='outside_support';evaluated+=tr[3]=='1';fail+=(tr[3]=='1' and tr[4]=='forward_failure')
    if tr[2]=='outside_support':assert tr[3]=='0' and tr[5]=='0'
    before=state[t][i];after=state[perm[i,t]][i+1]
    if tr[5]!='1':np.testing.assert_array_equal(before[1:],after[1:]);continue
    accepted+=1;k=int(before[1]);assert after[1]==k
    error=abs(float(tr[8])-((after[2]-before[2])/temp+after[3]-before[3]+q.logq(before)-q.logq(after)))
    max_error=max(max_error,error);assert error<1e-8
    both_changed+=k>1 and np.any(before[4:3+k]!=after[4:3+k]) and np.any(before[3+k:]!=after[3+k:])
    if before[2]<-50 and after[2]>0:rescues.append(dict(ladder=c,temperature=temp,iteration=i,before_logL=float(before[2]),after_logL=float(after[2]),k=k))
   stats.append(dict(ladder=c,temperature=temp,phase=phase,proposals=attempts,evaluated=int(evaluated),accepted=accepted,outside_support=int(outside),forward_failures=int(fail),both_interface_and_velocity_changed=int(both_changed)))
report=dict(passed=True,empirical_components_recomputed=fitted,training_hashes_unchanged=True,maximum_log_accept_ratio_error=max_error,events=stats,low_to_high_fit_events=rescues,
 interpretation='Finite proposal fit is not a posterior estimate. Rejections preserved; full mixture q correction remains untempered.')
save_json(out/'correlated-audit.json',report);print(json.dumps(report,indent=2))
