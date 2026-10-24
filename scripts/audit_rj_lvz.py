#!/usr/bin/env python3
"""Check new-case training provenance, mixture fits, LVZ definitions and predictions."""
import copy
import json
from pathlib import Path
import sys

import numpy as np

from benchmark_crust_depth import digest, rows
from joint_results import read_config, save_json


out=Path(sys.argv[1]).resolve()
design=json.loads((out/'design.json').read_text())
for name,sha in design['hashes'].items():assert digest(out/name)==sha
for stage in ['pilot','production']:
    for name,sha in json.loads((out/stage/'protocol.json').read_text())['hashes'].items():assert digest(out/stage/name)==sha
    for name in ['dispweave','observations.txt','covariance.txt','truth.toml']:
        assert digest(out/stage/name)==digest(out/name)
pilot=read_config(out/'pilot/rj.toml');production=read_config(out/'production/rj.toml')
old=read_config(Path('results/rj-pt-correlated/rj.toml'))
old.pop('rj_correlated_models');old['rjmcmc'].pop('correlated_probability');old['rjmcmc']['seed']=2026102801
assert old==pilot
restored=copy.deepcopy(production);restored.pop('rj_correlated_models');restored['rjmcmc'].pop('correlated_probability');restored['rjmcmc']['seed']=2026102801
assert restored==pilot and production['rjmcmc']['correlated_probability']==.4
training=json.loads((out/'production/training.json').read_text())
assert Path(training['source']).resolve()==out/'pilot'
assert training['mixture_config_sha256']==digest(out/'production/rj.toml')
for name,sha in training['source_hashes'].items():assert digest(out/'pilot'/name)==sha
chains=[[np.array(r,float) for r in rows(out/f'pilot/rj/chain_{c}_models.txt')] for c in range(4)]
components=0
for m in production['rj_correlated_models']:
    k=m['layers'];scales=np.array([5.]*(k-1)+[1.9]*k);eligible=[]
    for c,chain in enumerate(chains):
        x=np.array([r[4:] for r in chain if int(r[1])==k])
        count=next(v for v in training['counts'] if v['chain']==c and v['k']==k)
        assert count['count']==len(x) and count['eligible']==(len(x)>2*k-1)
        if len(x)>2*k-1:eligible.append(x)
    assert len(m['components'])==len(eligible)+1
    for x,part in zip(eligible,m['components'][:-1]):
        mean=x.sum(axis=0)/len(x);center=x-mean;cov=center.T@center/(len(x)-1)
        expected=.9*cov+.1*np.diag(np.diag(cov))+1e-6*np.diag(scales**2)
        np.testing.assert_allclose(part['mean'],mean,atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(part['covariance'],expected,atol=1e-13,rtol=1e-12)
        assert part['weight']==.9/len(eligible);components+=1
    broad=m['components'][-1]
    np.testing.assert_array_equal(broad['mean'],[5.*j/k for j in range(1,k)]+[1.75]*k)
    np.testing.assert_allclose(broad['covariance'],np.diag(scales**2/12),atol=1e-14,rtol=0)
    assert broad['weight']==(.1 if eligible else 1.)
report=json.loads((out/'transfer-summary.json').read_text())
for entry in report['stages']:
    count=0;all_locs=[]
    for c in range(4):
        chain=rows(out/f'{entry["stage"]}/rj/chain_{c}_models.txt');local=[]
        for row in chain:
            k=int(row[1]);v=list(map(float,row[3+k:]));z=list(map(float,row[4:3+k]))
            candidates=[j for j in range(1,k-1) if v[j-1]-v[j]>=.1 and v[j+1]-v[j]>=.1]
            if candidates:
                j=max(candidates,key=lambda j:min(v[j-1]-v[j],v[j+1]-v[j]))
                local.append([z[j-1],z[j],v[j],min(v[j-1]-v[j],v[j+1]-v[j])])
        assert len(local)==entry['conditional_lvz']['per_chain'][c]['conditional_count']
        count+=len(local);all_locs.extend(local)
    assert count==entry['conditional_lvz']['count']
    assert count/100000==entry['lvz_indicator']['mean']
    np.testing.assert_allclose(np.quantile(all_locs,[.025,.5,.975],axis=0),entry['conditional_lvz']['quantiles'],atol=1e-14,rtol=0)
selected=rows(out/'predict/selected_models.txt');trace=rows(out/'predict/draws.txt')
for c in range(4):
    chain=rows(out/f'production/rj/chain_{c}_models.txt')
    for j in range(100):
        i=c*100+j
        assert list(map(int,trace[i][:3]))==[i,c,j*250]
        assert selected[i][1:]==chain[j*250][1:]
assert sum(t[3]!='valid' for t in trace)==report['predictive']['failed_count']
obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt');truth=np.array([float(r[3]) for r in rows(out/'truth-forward.txt')]);filled=np.array([float(r[3]) for r in rows(out/'filled-forward.txt')])
np.testing.assert_allclose(obs[:,1]-truth,np.random.default_rng(2026102800).normal(0,.03,12),atol=1e-15,rtol=0)
delta=truth-filled
assert abs(delta@np.linalg.solve(cov,delta)-report['detectability']['filled_chi2'])<1e-9
save_json(out/'transfer-audit.json',dict(passed=True,empirical_components_recomputed=components,
    source_is_new_pilot_only=True,frozen_q_config_hash_verified=True,sampler_settings_preserved=True,
    lvz_event_and_location_counts_verified=True,predictive_indices_verified=400,new_forward_evaluations=0))
print('New-case provenance, frozen mixtures, events and predictive selection verified.')
