#!/usr/bin/env python3
"""Independent saved-state audit of the fixed-K4 PT experiment (no sampling)."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
from joint_results import read_config, write_config, save_json
from benchmark_crust_depth import forward
from posterior_diagnostics import diagnose
out=Path(sys.argv[1]).resolve();work=out/'audit';work.mkdir(exist_ok=False)
cfg=read_config(out/'rj.toml');base=read_config(Path('results/rj-fixed-k4/rj.toml'))
restored=copy.deepcopy(cfg);restored['rjmcmc']=base['rjmcmc'];assert restored==base
assert cfg['rjmcmc']==dict(base['rjmcmc'],warmup=2500,draws=25000,tempering=dict(temperatures=[1.,2.,4.,8.],swap_every=10))
for name in ['observations.txt','covariance.txt','truth.toml']:assert (out/name).read_bytes()==(Path('results/rj-fixed-k4')/name).read_bytes()
truth=read_config(out/'truth.toml');obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt');sign,ld=np.linalg.slogdet(cov);assert sign==1
summary=json.loads((out/'summary.json').read_text());cold=[];checks=0;max_error=0
for c in range(4):
 d=out/f'rj/ladder_{c}';a=np.stack([np.loadtxt(d/f'temperature_{t}_models.txt') for t in range(4)]);cold.append(a[0,:,4:]);pre=a.copy()
 for row in np.loadtxt(d/'swaps.txt'):
  i=int(row[0])-2500
  if i>=0 and row[9]:
   l,h=map(int,row[2:4]);pre[[l,h],i]=pre[[h,l],i]
 for t,temp in enumerate([1,2,4,8]):
  trace=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1);tr=trace[2501:];reject=tr[:,5]=='0'
  np.testing.assert_array_equal(pre[t,1:,1:][reject],a[t,:-1,1:][reject])
  accepted=tr[:,5]=='1';expected=(pre[t,1:,2]-a[t,:-1,2])/temp
  np.testing.assert_allclose(tr[accepted,8].astype(float),expected[accepted],atol=1e-11,rtol=1e-11)
  fail=trace[(trace[:,3]=='1')&np.isin(trace[:,4],['forward_failure','numerical_failure']),9].astype(int)
  saved=[int(line.split()[0]) for line in (d/f'temperature_{t}_failed_models.txt').read_text().splitlines() if not line.startswith('#')]
  np.testing.assert_array_equal(fail,saved)
  for i in [0,12500,24999]:
   model=copy.deepcopy(truth);row=a[t,i];vs=row[7:];model['model']=dict(depth=[0.,*row[4:7]],vs=vs.tolist(),vp=(1.8*vs).tolist(),rho=[2.3]*4)
   name=f'c{c}-t{t}-i{i}';write_config(work/f'{name}.toml',model)
   pred=forward(out/'dispweave',work/f'{name}.toml',work/name)
   assert all(r[4]=='valid' for r in pred),pred
   residual=np.array([float(r[3]) for r in pred])-obs[:,1]
   logl=-.5*(residual@np.linalg.solve(cov,residual)+ld+len(residual)*np.log(2*np.pi))
   max_error=max(max_error,abs(logl-row[2]));assert abs(logl-row[2])<1e-7;checks+=1
x=np.stack(cold)
for i,d in enumerate(summary['parameters']):
 actual=diagnose(x[:,:,i]);assert all(actual[key]==d[key] for key in actual)
for d in summary['depth_profiles']:
 value=np.zeros(x.shape[:2])
 for layer in range(4):
  mask=np.ones(x.shape[:2],bool)
  if layer:mask &= d['depth']>=x[:,:,layer-1]
  if layer<3:mask &= d['depth']<x[:,:,layer]
  value[mask]=x[:,:,3+layer][mask]
 actual=diagnose(value);assert all(actual[key]==d[key] for key in actual)
report=dict(passed=True,only_declared_changes=True,post_swap_rejections_and_accepted_ratios_verified=True,failed_evaluation_ids_verified=True,
 raw_likelihood_recomputations=checks,maximum_log_likelihood_error=max_error,all_diagnostics_recomputed=True)
save_json(out/'independent-audit.json',report);print(report)
