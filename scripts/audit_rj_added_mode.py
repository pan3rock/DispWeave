#!/usr/bin/env python3
"""Independent matched-input, noise, structural-summary and failure attribution checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import numpy as np

out=Path(sys.argv[1]);design=json.loads((out/'design.json').read_text());report=json.loads((out/'comparison-summary.json').read_text())
def rows(p):return [line.split() for line in p.read_text().splitlines() if line and not line.startswith('#')]
def quantiles(data):
    arr=np.asarray(data);result=[]
    for p in [.025,.5,.975]:
        ordered=np.sort(arr,axis=0);j=(len(arr)-1)*p;lo=int(np.floor(j));hi=int(np.ceil(j));result.append(ordered[lo]+(ordered[hi]-ordered[lo])*(j-lo))
    return np.asarray(result)
configs=[tomllib.loads((out/a/'rj.toml').read_text()) for a in ['baseline','augmented']]
assert configs[0]==configs[1]
old=tomllib.loads(Path('results/rj-lvz-transfer/production/rj.toml').read_text())
restored=json.loads(json.dumps(configs[0]));restored['rjmcmc']['seed']=old['rjmcmc']['seed'];restored['rj_starts']=old['rj_starts'];assert restored==old
for start,record in zip(configs[0]['rj_starts'],design['shared_starts']):assert start==dict(interfaces=record['model'][4:7],vs=record['model'][7:])
obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt');assert obs.shape==(14,4) and cov.shape==(14,14)
np.testing.assert_array_equal(obs[:12],np.loadtxt(out/'baseline/observations.txt'));np.testing.assert_array_equal(cov[:12,:12],np.loadtxt(out/'baseline/covariance.txt'))
np.testing.assert_array_equal(obs,np.loadtxt(out/'augmented/observations.txt'));np.testing.assert_array_equal(cov,np.loadtxt(out/'augmented/covariance.txt'))
np.testing.assert_array_equal(obs[12:,[0,2]],np.array([[.6,1],[.65,1]]))
expected_noise=np.random.default_rng(design['noise_seed']).normal(0,.03,2)
np.testing.assert_array_equal(expected_noise,np.loadtxt(out/'added-noise.txt'))
truth=np.array([float(r[3]) for r in rows(out/'truth-forward.txt')]);np.testing.assert_array_equal(obs[12:,1],truth[12:]+expected_noise)
assert np.all(cov[:12,12:]==0) and np.all(cov[12:,:12]==0);np.testing.assert_array_equal(cov[12:,12:],np.eye(2)*.03**2)
failures={};widths=[]
for result in report['arms']:
    name=result['arm'];folder=out/name
    counts=Counter();chain_counts=[];locations=[];chains=[]
    for c in range(4):
        chain=rows(folder/f'rj/chain_{c}_models.txt');assert len(chain)==25000;chains.append(chain);cc=Counter()
        for r in chain:
            k=int(r[1]);z=list(map(float,r[4:3+k]));v=list(map(float,r[3+k:]));dips=[min(v[j-1]-v[j],v[j+1]-v[j]) for j in range(1,k-1)]
            label=f'K{k}'
            if dips and max(dips)>=.1:
                j=dips.index(max(dips))+1;locations.append([z[j-1],z[j],v[j]])
                if k==4:label='A' if j==1 else 'B'
            elif k==4:label='K4_no_dip'
            cc[label]+=1
        counts.update(cc);chain_counts.append(cc)
    for g in result['groups']:
        assert g['fraction']==counts[g['label']]/100000
        assert g['per_chain_fraction']==[c[g['label']]/25000 for c in chain_counts]
    q=quantiles(locations);np.testing.assert_allclose(q,result['conditional_lvz']['quantiles'],rtol=1e-13,atol=1e-13)
    assert len(locations)==result['conditional_lvz']['count'];widths.append(q[2]-q[0])
    selected=json.loads((folder/'comparison-predict/selection.json').read_text());assert len(selected)==100
    for i,s in enumerate(selected):
        assert s['row']==i and s['chain']==i//25 and s['draw']==i%25*1000
        r=chains[s['chain']][s['draw']];k=int(r[1]);cfg=tomllib.loads((folder/f'comparison-predict/{i:03d}.toml').read_text());v=list(map(float,r[3+k:]))
        assert cfg['model']==dict(depth=[0.]+list(map(float,r[4:3+k])),vs=v,vp=[1.8*x for x in v],rho=[2.3]*k)
        assert cfg['queries']==tomllib.loads((out/'truth.toml').read_text())['queries']
        assert s['log_likelihood']==float(r[2])
    output_rows=[rows(folder/f'comparison-predict/{i:03d}.txt') for i in range(100)]
    expected_failures=[]
    for j in range(14):
        for i,rr in enumerate(output_rows):
            if rr[j][4]!='valid':expected_failures.append(dict(row=i,query=j,status=rr[j][4]))
        valid=all(rr[j][4]=='valid' for rr in output_rows)
        expected=result['predictions']['latent_quantiles'][j]
        if valid:np.testing.assert_allclose(quantiles([float(rr[j][3]) for rr in output_rows]),expected,rtol=1e-13,atol=1e-13)
        else:assert expected is None
    assert sorted(expected_failures,key=lambda x:(x['row'],x['query']))==result['predictions']['failures']
    categories=Counter()
    for c in range(4):
        for t in range(4):
            d=folder/f'rj/ladder_{c}';trace=rows(d/f'temperature_{t}_local_trace.txt')
            formal={int(r[9]) for r in trace if r[0]=='0' and r[3]=='1' and r[4] in ['forward_failure','numerical_failure']}
            bad={i:[] for i in formal}
            for r in rows(d/f'temperature_{t}_failed_rows.txt'):
                if int(r[0]) in formal and r[2]!='valid':bad[int(r[0])].append(int(r[1]))
            for indices in bad.values():
                assert indices;lo=any(i<12 for i in indices);hi=any(i>=12 for i in indices)
                categories['both' if lo and hi else 'original_only' if lo else 'added_only']+=1
    assert sum(categories.values())==result['base_summary']['formal_forward_failures']+result['base_summary']['formal_numerical_failures']
    failures[name]=dict(categories)
np.testing.assert_allclose(widths[1]/widths[0],report['conditional_width_ratio_augmented_over_baseline'],rtol=1e-13)
for path,h in design['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
for n,h in design['hashes'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h
result=dict(passed=True,paired_configs_identical=True,only_old_config_changes=['seed','common starts'],original_data_unchanged=True,
    fixed_noise_verified=True,all_200000_cold_states_classified=True,conditional_quantiles_verified=True,prediction_source_models_verified=200,
    formal_failure_attribution=failures,source_hashes_unchanged=True,new_forward_calls=0)
(out/'comparison-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
