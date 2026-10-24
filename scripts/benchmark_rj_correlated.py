#!/usr/bin/env python3
"""Freeze per-K correlated mixtures from existing PT, then run one bounded comparison."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from benchmark_rj_pt_transdim import analyze
from joint_results import read_config,write_config,save_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=Path('results/rj-nuts-comparison').resolve();training=Path('results/rj-pt-transdim').resolve();begin=time.perf_counter()
cfg=read_config(source/'rj.toml');traincfg=read_config(training/'rj.toml')
assert cfg['rj_prior']==traincfg['rj_prior'] and cfg['rj_physics']==traincfg['rj_physics']
for name in ['observations.txt','covariance.txt','truth.toml']:
 assert (source/name).read_bytes()==(training/name).read_bytes();shutil.copy2(source/name,out/name)
shutil.copy2('build/dispweave',out/'dispweave');shutil.copy2(__file__,out/'run-script.py');shutil.copy2('scripts/benchmark_rj_pt_transdim.py',out/'analysis-script.py')
# Save the design before fitting or running. No outcomes used to select components/scales.
save_json(out/'design.json',dict(total_local_proposals=440000,training=str(training),component_scheme='one empirical component per old cold chain for each visited K, weights .225 each; broad defensive Gaussian weight .1; unvisited K uses broad Gaussian only',
 covariance='0.9 empirical +0.1 diagonal(empirical) +1e-6 diag([depth_limit]*(K-1)+[Vs_width]*K)^2',
 defensive_covariance='diagonal prior-range squared /12, interfaces centered at j*depth_limit/K; full Gaussian, no truncation',
 proposal_rate='conditional velocity probability .4 => .1 of all proposals; prior-uniform refresh disabled',
 warmup=2500,draws=25000,temperatures=[1,2,4,8],cold_start_steps=0,no_retries_tuning_or_extensions=True,
 training_interpretation='Unaccepted old samples shape q only, never counted as new posterior; all old sampling costs reported separately and included in end-to-end comparison.'))
chains=[];training_hashes={}
for c in range(4):
 file=training/f'rj/chain_{c}_models.txt';training_hashes[str(file.relative_to(training))]=digest(file)
 chains.append([np.fromstring(line,sep=' ') for line in file.read_text().splitlines() if line and not line.startswith('#')])
models=[];counts=[]
for k in range(1,5):
 n=2*k-1;scale=np.array([5.]*(k-1)+[1.9]*k);mean=np.array([5.*(j+1)/k for j in range(k-1)]+[1.75]*k)
 broad=dict(mean=mean.tolist(),covariance=np.diag(scale**2/12).tolist(),weight=.1)
 components=[]
 for c,chain in enumerate(chains):
  x=np.array([row[4:] for row in chain if row[1]==k]);counts.append(dict(k=k,chain=c,count=len(x)))
  if len(x):
   assert len(x)>n+1
   covariance=np.atleast_2d(np.cov(x,rowvar=False));regularized=.9*covariance+.1*np.diag(np.diag(covariance))+1e-6*np.diag(scale**2)
   np.linalg.cholesky(regularized)
   components.append(dict(mean=x.mean(axis=0).tolist(),covariance=regularized.tolist(),weight=.225))
 assert len(components) in [0,4]
 if not components:broad['weight']=1.
 components.append(broad);models.append(dict(layers=k,components=components))
cfg['rj_correlated_models']=models
cfg['rjmcmc'].update(warmup=2500,draws=25000,correlated_probability=.4,tempering=dict(temperatures=[1.,2.,4.,8.],swap_every=10,save_warmup=True))
write_config(out/'rj.toml',cfg)
training_cost=json.loads((training/'summary.json').read_text())['total_target_evaluations']
save_json(out/'training.json',dict(source=str(training),source_hashes=training_hashes,counts=counts,old_target_evaluations=training_cost,
 fitting_wall_seconds=time.perf_counter()-begin,mixture_config_sha256=digest(out/'rj.toml'),interpretation='Shared frozen proposal across fresh independent random streams; old posterior accuracy not assumed.'))
save_json(out/'protocol.json',dict(source=str(source),total_local_proposals=440000,baseline_total_local_proposals=440000,ladders=4,warmup=2500,draws=25000,temperatures=[1,2,4,8],swap_every=10,no_retries_tuning_or_extensions=True,
 acceptance='All K indicators and 31 depths Rhat<=1.01 and bulk/tail ESS>=400; unvisited K unresolved; forward failures separately audited and prevent overall acceptance',
 hashes={name:digest(out/name) for name in ['dispweave','rj.toml','observations.txt','covariance.txt','truth.toml','run-script.py','analysis-script.py','design.json','training.json']}))
call=invoke(out/'dispweave',out,'rj',['rjmcmc','--config',out/'rj.toml','--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',out/'rj'],300)
if call['returncode'] not in [0,5]:raise RuntimeError('Incomplete run; no sampling retries')
analyze(out,source)
report=json.loads((out/'summary.json').read_text())
save_json(out/'costs.json',dict(new_target_evaluations=report['total_target_evaluations'],training_target_evaluations=training_cost,end_to_end_target_evaluations=training_cost+report['total_target_evaluations'],
 interpretation='New-run budget matched; old pilot samples have a real cost. Independent audit evaluations excluded from both sampling budgets.'))
