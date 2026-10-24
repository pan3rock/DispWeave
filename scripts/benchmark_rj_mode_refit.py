#!/usr/bin/env python3
"""Refit q from existing augmented-data states, freeze, and run one fresh stream."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
from benchmark_crust_depth import digest, rows
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, save_json

SOURCE=Path('results/rj-added-mode/augmented')

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    for n in ['dispweave','observations.txt','covariance.txt','truth.toml']:shutil.copy2(SOURCE/n,out/n)
    shutil.copy2(__file__,out/'run-script.py')
    cfg=read_config(SOURCE/'rj.toml')
    hashes={n:digest(SOURCE/n) for n in ['rj.toml','summary.json','observations.txt','covariance.txt','truth.toml',*[f'rj/chain_{c}_models.txt' for c in range(4)]]}
    save_json(out/'design.json',dict(source=str(SOURCE.resolve()),seed=2026103101,old_seed=cfg['rjmcmc']['seed'],
        total_proposals=440000,warmup=2500,draws=25000,ladders=4,temperatures=[1,2,4,8],
        changed='Only rj_correlated_models and fresh sampler seed. Data/prior/starts/proposal rate/temperature/budget unchanged.',
        training='All formal cold states, repeats retained; per K one component per chain with count > dimension (2K-1). Weight .9 split across eligible chains; .1 broad defensive, or broad weight1 if no empirical components.',
        covariance='.9 S + .1 diag(S) + 1e-6 diag([5]*(K-1)+[1.9]*K)^2; broad diagonal range squared/12; no tuning',
        primary='B indicator and worst depth Rhat excess above1 reduce >=25%; B bulk and tail ESS, depth minimum bulk and tail ESS all >=2x previous. All conditions required for clear improvement; acceptance rate secondary.',
        acceptance='Full posterior still requires Rhat<=1.01, bulk/tail ESS>=400, no unresolved support failures and no unresolved unvisited K. Improvement is not acceptance.',
        limitations='Training run not converged; fit is a proposal, not posterior truth. One old/new seed pair cannot isolate Monte Carlo variability. Source samples never pooled with new posterior.',
        no_retries_tuning_extensions=True,audit_model_forwards=48,no_new_predictive_forwards=True,
        source_hashes=hashes,hashes={n:digest(out/n) for n in ['dispweave','observations.txt','covariance.txt','truth.toml','run-script.py']}))
    chains=[[np.array(r,float) for r in rows(SOURCE/f'rj/chain_{c}_models.txt')] for c in range(4)]
    models=[];counts=[]
    for k in range(1,5):
        scale=np.array([5.]*(k-1)+[1.9]*k);parts=[]
        for c,chain in enumerate(chains):
            x=np.array([r[4:] for r in chain if int(r[1])==k]);eligible=len(x)>2*k-1
            counts.append(dict(k=k,chain=c,count=len(x),eligible=eligible))
            if eligible:
                cov=np.atleast_2d(np.cov(x,rowvar=False));cov=.9*cov+.1*np.diag(np.diag(cov))+1e-6*np.diag(scale**2)
                np.linalg.cholesky(cov);parts.append(dict(mean=x.mean(axis=0).tolist(),covariance=cov.tolist()))
        for part in parts:part['weight']=.9/len(parts)
        parts.append(dict(mean=[5.*j/k for j in range(1,k)]+[1.75]*k,covariance=np.diag(scale**2/12).tolist(),weight=.1 if parts else 1.))
        models.append(dict(layers=k,components=parts))
    cfg['rj_correlated_models']=models;cfg['rjmcmc']['seed']=2026103101
    write_config(out/'rj.toml',cfg)
    restored=copy.deepcopy(cfg);old=read_config(SOURCE/'rj.toml');restored['rj_correlated_models']=old['rj_correlated_models'];restored['rjmcmc']['seed']=old['rjmcmc']['seed'];assert restored==old
    save_json(out/'training.json',dict(source=str(SOURCE.resolve()),source_hashes=hashes,counts=counts,mixture_config_sha256=digest(out/'rj.toml'),old_target_evaluations=json.loads((SOURCE/'summary.json').read_text())['total_target_evaluations']))
    save_json(out/'protocol.json',dict(total_local_proposals=440000,hashes={n:digest(out/n) for n in ['dispweave','observations.txt','covariance.txt','truth.toml','rj.toml','design.json','training.json','run-script.py']}))
    call=invoke(out/'dispweave',out,'rj',['rjmcmc','--config',out/'rj.toml','--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',out/'rj'],300)
    if call['returncode'] not in [0,5]:raise RuntimeError('Incomplete sampling, no retry')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
