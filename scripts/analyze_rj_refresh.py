#!/usr/bin/env python3
"""Audit joint-refresh events in saved warmup and production, no new target calls."""
import json
from pathlib import Path
import sys
import numpy as np
from joint_results import read_config,save_json

out=Path(sys.argv[1]).resolve();cfg=read_config(out/'rj.toml');warm=cfg['rjmcmc']['warmup'];temps=cfg['rjmcmc']['tempering']['temperatures'];stats=[];rescues=[]
def load(file):return [np.fromstring(s,sep=' ') for s in file.read_text().splitlines() if s and not s.startswith('#')]
for c in range(4):
 d=out/f'rj/ladder_{c}';states=[load(d/f'temperature_{t}_warmup_models.txt')+load(d/f'temperature_{t}_models.txt') for t in range(4)]
 swaps=np.loadtxt(d/'swaps.txt');perm=np.tile(np.arange(4),(len(states[0])-1,1))
 for r in swaps:
  if r[9]:i,l,h=map(int,r[[0,2,3]]);perm[i,l],perm[i,h]=h,l
 for t,temp in enumerate(temps):
  tr=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1)
  for phase,left,right in [('warmup',0,warm),('production',warm,len(tr))]:
   attempts=accepted=failed=surface_switches=0
   for i in range(left,right):
    if tr[i,1]!='velocity_refresh':continue
    attempts+=1;assert tr[i,2]=='valid' and tr[i,3]=='1'
    failed+=tr[i,4]=='forward_failure';before=states[t][i];after=states[perm[i,t]][i+1]
    if tr[i,5]!='1':continue
    accepted+=1;k=int(before[1]);assert k==after[1]
    np.testing.assert_array_equal(before[4:3+k],after[4:3+k]);assert np.all(before[3+k:]!=after[3+k:]);assert before[3]==after[3]
    np.testing.assert_allclose(float(tr[i,8]),(after[2]-before[2])/temp,atol=1e-10,rtol=1e-10)
    surface_switches+=before[3+k]>2 and after[3+k]<1.5
    if before[2]<-50 and after[2]>0:rescues.append(dict(ladder=c,temperature=temp,iteration=i,warmup=i<warm,before_logL=float(before[2]),after_logL=float(after[2]),before_surface_vs=float(before[3+k]),after_surface_vs=float(after[3+k]),k=k))
   stats.append(dict(ladder=c,temperature=temp,phase=phase,proposals=attempts,accepted=accepted,forward_failures=int(failed),high_to_low_surface_switches=int(surface_switches)))
report=dict(passed=True,events=stats,low_to_high_fit_events=rescues,
 interpretation='Descriptive transitions from raw logL<-50 to >0, and surface Vs>2 to <1.5; not independently certified posterior modes or evidence of convergence.')
save_json(out/'refresh-audit.json',report);print(json.dumps(report,indent=2))
