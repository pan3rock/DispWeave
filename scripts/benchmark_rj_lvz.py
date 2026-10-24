#!/usr/bin/env python3
"""Preregistered three-layer LVZ transfer: one new pilot and one frozen-q run."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from benchmark_crust_depth import digest, forward, rows
from benchmark_rj_nuts import invoke
from benchmark_rj_pt_transdim import analyze
from joint_results import read_config, write_config, save_json


def physical(depth, vs):
    return dict(depth=depth, vs=vs, vp=[1.8*v for v in vs], rho=[2.3]*len(vs))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    old = Path('results/rj-pt-correlated')
    shutil.copy2(old/'dispweave', out/'dispweave')
    shutil.copy2(__file__, out/'run-script.py')
    shutil.copy2('scripts/benchmark_rj_pt_transdim.py', out/'analysis-script.py')
    cfg = read_config(old/'rj.toml')
    cfg.pop('rj_correlated_models')
    cfg['rjmcmc'].pop('correlated_probability')
    cfg['rjmcmc']['seed'] = 2026102801
    query = dict(frequency_hz=np.geomspace(.15,.9,12).tolist(), mode=[0]*12)
    write_config(out/'truth.toml', dict(model=physical([0.,.8,1.8],[1.6,1.,2.3]), queries=query))
    write_config(out/'filled.toml', dict(model=physical([0.,.8,1.8],[1.6,1.6,2.3]), queries=query))
    lm = dict(model=physical([0.,1.8],[1.6,2.3]), inversion=dict(lower_vs=[.8,.8],upper_vs=[2.7,2.7],
        vp_vs_ratio=1.8, starts_vs=[[1.1,2.2],[.9,2.4],[1.2,2.1],[1.,2.5]], sensitivity='implicit',
        max_iterations=150,gradient_tolerance=1e-5,regularization=dict(type='none',**{'lambda':0.})))
    write_config(out/'noiseless-lm.toml', lm)
    write_config(out/'pilot-base.toml', cfg)
    save_json(out/'design.json',dict(truth='K=3; z=[.8,1.8] km; Vs=[1.6,1.0,2.3] km/s',
        pilot_seed=2026102801,production_seed=2026102802,noise_seed=2026102800,predictive_noise_seed=2026102803,
        sigma=.03,frequencies=query['frequency_hz'],total_local_proposals_per_stage=440000,total_stages=2,
        no_retries_extensions_noise_reselection_or_tuning=True,
        detectability='Before sampling: target minus filled-LVZ Mahalanobis separation; fixed-z=1.8 two-layer unregularized multistart fit to noiseless target. Not global identifiability against all non-LVZ models. Weak contrast does not stop or change the case.',
        proposal_training='Only new pilot cold states, repeats retained. Per K one component per chain with count > 2K-1; empirical total weight .9 divided equally among eligible chains, defensive weight .1; if none, broad weight 1. Same .9 S+.1 diag(S)+1e-6 diag(scale^2), scales=[5]*(K-1)+[1.9]*K. No component selected by fit quality.',
        production='Same starts, steps, temperatures, warmup/draws and 10% total correlated moves as old case. Frozen q before new run.',
        lvz_event='An interior layer at least .1 km/s slower than both neighbors; strongest qualifying dip used only for conditional location summaries.',
        depth_contrast='Vs(.4)-Vs(1.2) and Vs(2.5)-Vs(1.2); both >.1 defines fixed-depth dip event',
        predictive='100 evenly spaced retained draws per cold chain, no replacement or dropping failures',
        acceptance='All K indicators and 31 depths Rhat<=1.01 and bulk/tail ESS>=400; LVZ indicators/contrasts also checked. Constants/unvisited K and unresolved forward failures prevent full certification. Recovery need not mean P(K=3) near 1.',
        hashes={n:digest(out/n) for n in ['dispweave','run-script.py','analysis-script.py','truth.toml','filled.toml','noiseless-lm.toml','pilot-base.toml']}))
    truth = forward(out/'dispweave',out/'truth.toml',out/'truth-forward.txt')
    filled = forward(out/'dispweave',out/'filled.toml',out/'filled-forward.txt')
    if not all(r[4]=='valid' for r in truth+filled):
        save_json(out/'precheck-failure.json',dict(reason='Unresolved target/control roots; no sampling or retuning'))
        raise RuntimeError('LVZ detectability precheck failed')
    y=np.array([float(r[3]) for r in truth]); null=np.array([float(r[3]) for r in filled])
    covariance=np.eye(12)*.03**2
    np.savetxt(out/'covariance.txt',covariance,fmt='%.17g')
    np.savetxt(out/'noiseless.txt',np.c_[query['frequency_hz'],y,np.zeros(12),np.full(12,.03)],fmt='%.17g')
    noise=np.random.default_rng(2026102800).normal(0,.03,12)
    np.savetxt(out/'noise.txt',noise,fmt='%.17g')
    np.savetxt(out/'observations.txt',np.c_[query['frequency_hz'],y+noise,np.zeros(12),np.full(12,.03)],fmt='%.17g')
    call=invoke(out/'dispweave',out,'noiseless-lm',['invert','--config',out/'noiseless-lm.toml','--data',out/'noiseless.txt','--covariance',out/'covariance.txt','--output',out/'noiseless-lm'])
    meta=dict(line.split('=',1) for line in (out/'noiseless-lm/metadata.txt').read_text().splitlines())
    best=int(meta['best_run']); lm_rows=rows(out/'noiseless-lm/runs.txt')
    separation=dict(filled_chi2=float(np.sum(((y-null)/.03)**2)),filled_max_abs_delta_sigma=float(np.max(abs(y-null))/.03),
        fixed_geometry_lm_converged=best>=0,lm_best_run=best,lm_evaluations=sum(int(r[3]) for r in lm_rows),
        lm_residual_chi2=2*float(lm_rows[best][2]) if best>=0 else None,
        interpretation='Mahalanobis distances between noiseless curves, not a p-value or proof of global structural identifiability.')
    save_json(out/'detectability.json',separation);print('DETECTABILITY',separation,flush=True)
    for stage,seed in [('pilot',2026102801),('production',2026102802)]:
        folder=out/stage;folder.mkdir()
        for name in ['dispweave','truth.toml','observations.txt','covariance.txt']:
            shutil.copy2(out/name,folder/name)
        cfg=copy.deepcopy(cfg);cfg['rjmcmc']['seed']=seed
        if stage=='production':
            source=out/'pilot'; chains=[];hashes={}
            for c in range(4):
                name=f'rj/chain_{c}_models.txt';hashes[name]=digest(source/name)
                chains.append([np.array(r,float) for r in rows(source/name)])
            models=[];counts=[]
            for k in range(1,5):
                scale=np.array([5.]*(k-1)+[1.9]*k);components=[]
                for c,chain in enumerate(chains):
                    x=np.array([r[4:] for r in chain if r[1]==k]);eligible=len(x)>2*k-1
                    counts.append(dict(k=k,chain=c,count=len(x),eligible=eligible))
                    if eligible:
                        cov=np.atleast_2d(np.cov(x,rowvar=False));cov=.9*cov+.1*np.diag(np.diag(cov))+1e-6*np.diag(scale**2)
                        components.append(dict(mean=x.mean(axis=0).tolist(),covariance=cov.tolist()))
                for component in components:component['weight']=.9/len(components)
                components.append(dict(mean=[5.*j/k for j in range(1,k)]+[1.75]*k,
                    covariance=np.diag(scale**2/12).tolist(),weight=.1 if components else 1.))
                models.append(dict(layers=k,components=components))
            cfg['rj_correlated_models']=models;cfg['rjmcmc']['correlated_probability']=.4
            training=dict(source=str(source),source_hashes=hashes,counts=counts,old_target_evaluations=json.loads((source/'summary.json').read_text())['total_target_evaluations'])
        write_config(folder/'rj.toml',cfg)
        if stage=='production':
            training['mixture_config_sha256']=digest(folder/'rj.toml');save_json(folder/'training.json',training)
        save_json(folder/'protocol.json',dict(stage=stage,seed=seed,total_local_proposals=440000,
            hashes={n:digest(folder/n) for n in ['dispweave','truth.toml','observations.txt','covariance.txt','rj.toml']},
            interpretation='New LVZ data only; pilot and production remain separate. No former-case posterior samples.'))
        call=invoke(folder/'dispweave',folder,'rj',['rjmcmc','--config',folder/'rj.toml','--data',folder/'observations.txt','--covariance',folder/'covariance.txt','--output',folder/'rj'])
        if call['returncode'] not in [0,5]:raise RuntimeError('Stage incomplete; no retry')
        analyze(folder)
    call=invoke(out/'dispweave',out,'predict',['rj-predict','--run',out/'production/rj','--output',out/'predict','--draws-per-chain',100,'--seed',2026102803])
    if call['returncode'] not in [0,5]:raise RuntimeError('Prediction incomplete; no retry')
    costs={stage:json.loads((out/stage/'summary.json').read_text())['total_target_evaluations'] for stage in ['pilot','production']}
    save_json(out/'costs.json',dict(**costs,total_sampling=sum(costs.values()),predictive_models=400,detectability_forwards=2,
        detectability_lm_evaluations=separation['lm_evaluations'],interpretation='Pilot charged fully; independent audits separate; no old training samples reused.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
