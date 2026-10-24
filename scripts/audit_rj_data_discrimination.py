#!/usr/bin/env python3
"""Independently verify selection, physical inputs and screening arithmetic; no forwards."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib
import numpy as np

out=Path(sys.argv[1]);protocol=json.loads((out/'protocol.json').read_text())
selected=json.loads((out/'selection.json').read_text());summary=json.loads((out/'summary.json').read_text())
def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith('#')]
def label(r):
    if int(r[1])!=4:return None
    a,b,c,d=map(float,r[7:])
    if a-b>=.1 and c-b>=.1:return 'A'
    if b-c>=.1 and d-c>=.1:return 'B'
    return None
def quantile(x,p):
    x=sorted(x);k=(len(x)-1)*p;lo=math.floor(k);hi=math.ceil(k)
    return x[lo]+(x[hi]-x[lo])*(k-lo)
def variance(x):
    mean=sum(x)/len(x)
    return sum((v-mean)**2 for v in x)/(len(x)-1)
source=Path('results/rj-lvz-transfer/production')
assert len(selected)==64
for c in range(4):
    chain=rows(source/f'rj/chain_{c}_models.txt')
    for f in ['A','B']:
        available=[i for i,r in enumerate(chain) if label(r)==f]
        expected=[available[len(available)*j//8] for j in range(8)]
        actual=[x for x in selected if x['chain']==c and x['family']==f]
        assert [x['draw'] for x in actual]==expected
        for s in actual:
            assert s['model']==list(map(float,chain[s['draw']]))
            model=tomllib.loads((out/f'model-{s["row"]:02d}.toml').read_text())['model']
            assert model['depth']==[0.]+s['model'][4:7]
            assert model['vs']==s['model'][7:]
            assert model['vp']==[v*1.8 for v in model['vs']] and model['rho']==[2.3]*4
truth=tomllib.loads((source/'truth.toml').read_text())['model']
assert tomllib.loads((out/'target.toml').read_text())['model']==truth
outputs=[rows(out/f'model-{i:02d}.txt') for i in range(64)]+[rows(out/'target.txt')]
assert all(len(x)==38 for x in outputs)
assert len(json.loads((out/'commands.json').read_text()))==65
sigma=protocol['sigma'];candidates=[]
for j,(q,r) in enumerate(zip(protocol['queries'],summary['queries'])):
    assert all(int(x[j][2])==q['mode'] and abs(float(x[j][1])-q['frequency'])<1e-12 for x in outputs)
    assert dict(Counter(x[j][4] for x in outputs))==r['statuses']
    eligible=all(x[j][4]=='valid' for x in outputs);assert eligible==r['eligible']
    candidate=False
    if eligible:
        a=[float(outputs[s['row']][j][3]) for s in selected if s['family']=='A']
        b=[float(outputs[s['row']][j][3]) for s in selected if s['family']=='B']
        for f,x in [('A',a),('B',b)]:np.testing.assert_allclose([quantile(x,p) for p in [.025,.5,.975]],r['quantiles'][f],rtol=1e-13)
        gap=quantile(a,.5)-quantile(b,.5);adjusted=abs(gap)/math.sqrt((variance(a)+variance(b))/2+sigma**2)
        np.testing.assert_allclose([abs(gap)/sigma,adjusted],[r['gap_sigma'],r['dispersion_adjusted_gap']],rtol=1e-13)
        gaps=[]
        for c in range(4):
            groups=[[float(outputs[s['row']][j][3]) for s in selected if s['family']==f and s['chain']==c] for f in ['A','B']]
            gaps.append(quantile(groups[0],.5)-quantile(groups[1],.5))
        np.testing.assert_allclose(gaps,r['chain_gaps'],rtol=1e-13)
        candidate=q['band']!='existing' and abs(gap)/sigma>=1 and adjusted>=1 and all(g*gap>0 for g in gaps)
    assert candidate==r['candidate'];candidates.append(candidate)
bands=[]
for band in ['low','high','mode1','mode2']:
    indices=[i for i,q in enumerate(protocol['queries']) if q['band']==band]
    pairs=[[protocol['queries'][a]['frequency'],protocol['queries'][b]['frequency']] for a,b in zip(indices[:-1],indices[1:]) if candidates[a] and candidates[b]]
    if pairs:bands.append(dict(band=band,adjacent_pairs=pairs))
assert bands==summary['promising_bands']
for path,h in protocol['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
for name,h in protocol['hashes'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
report=dict(passed=True,selection_and_chain_provenance_verified=64,physical_models_verified=65,
    query_results_checked=2470,independent_quantiles_variance_candidates=True,source_hashes_unchanged=True,
    new_forward_calls=0,new_samples=0,root_completeness_certified=False)
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
