#!/usr/bin/env python3
"""Read-only independent provenance, conditional-statistic and root-order audit."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tomllib

out=Path(sys.argv[1]);protocol=json.loads((out/'protocol.json').read_text())
selected=json.loads((out/'selection.json').read_text());summary=json.loads((out/'summary.json').read_text())
source=Path('results/rj-lvz-transfer/production');old=Path('results/rj-data-discrimination')
def rows(p):return [line.split() for line in p.read_text().splitlines() if line and not line.startswith('#')]
def label(r):
    if int(r[1])!=4:return None
    vs=list(map(float,r[7:]));strength=[min(vs[i-1]-vs[i],vs[i+1]-vs[i]) for i in [1,2]]
    if max(strength)<.1:return None
    return ['A','B'][strength.index(max(strength))]
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-13),(a,b)
def q(x,p):
    v=sorted(x);j=(len(v)-1)*p;lo=math.floor(j);hi=math.ceil(j)
    return v[lo]+(v[hi]-v[lo])*(j-lo)
old_selection=json.loads((old/'selection.json').read_text())
assert len(selected)==128
assert [{k:v for k,v in s.items() if k!='cohort'} for s in selected[:64]]==old_selection
excluded={tuple(s['model'][4:]) for s in old_selection}
assert not excluded.intersection(tuple(s['model'][4:]) for s in selected[64:])
for c in range(4):
    chain=rows(source/f'rj/chain_{c}_models.txt')
    for f in ['A','B']:
        eligible=[i for i,r in enumerate(chain) if label(r)==f and tuple(map(float,r[4:])) not in excluded]
        chosen=[s for s in selected[64:] if s['chain']==c and s['family']==f]
        assert [s['draw'] for s in chosen]==[eligible[math.floor((j+.5)*len(eligible)/8)] for j in range(8)]
    for s in selected:
        if s['chain']==c:assert s['model']==list(map(float,chain[s['draw']]))
frequencies=protocol['frequencies'];tables=[]
for i in range(129):
    cfg=tomllib.loads((out/f'model-{i:03d}.toml').read_text())
    assert cfg['queries']==dict(frequency_hz=frequencies,mode=[1]*10)
    if i<128:
        r=selected[i]['model'];m=dict(depth=[0.]+r[4:7],vs=r[7:],vp=[1.8*v for v in r[7:]],rho=[2.3]*4)
    else:m=tomllib.loads((source/'truth.toml').read_text())['model']
    assert cfg['model']==m
    table=rows(out/f'model-{i:03d}.txt');assert len(table)==10
    for r,f in zip(table,frequencies):assert int(r[2])==1;close(float(r[1]),f)
    tables.append(table)
passed={};gaps={}
for cohort in ['original','replication']:
    chosen=[s for s in selected if s['cohort']==cohort];assert Counter(s['family'] for s in chosen)=={'A':32,'B':32}
    passed[cohort]=[];gaps[cohort]=[]
    for j,r in enumerate(summary['cohorts'][cohort]):
        eligible=all(tables[s['row']][j][4]=='valid' for s in chosen) and tables[128][j][4]=='valid'
        assert eligible==r['eligible'];assert dict(Counter(tables[s['row']][j][4] for s in chosen))==r['statuses']
        candidate=False;gap=0.
        if eligible:
            a,b=[[float(tables[s['row']][j][3]) for s in chosen if s['family']==f] for f in ['A','B']]
            for f,x in [('A',a),('B',b)]:
                for actual,expected in zip([q(x,p) for p in [.025,.5,.975]],r['quantiles'][f]):close(actual,expected)
            gap=statistics.median(a)-statistics.median(b)
            d=abs(gap)/math.sqrt((statistics.variance(a)+statistics.variance(b))/2+protocol['sigma']**2)
            close(gap,r['gap']);close(d,r['D']);close(abs(gap)/protocol['sigma'],r['gap_sigma'])
            chain_gaps=[]
            for c in range(4):
                ca,cb=[[float(tables[s['row']][j][3]) for s in chosen if s['family']==f and s['chain']==c] for f in ['A','B']]
                chain_gaps.append(statistics.median(ca)-statistics.median(cb))
            for actual,expected in zip(chain_gaps,r['chain_gaps']):close(actual,expected)
            candidate=abs(gap)/protocol['sigma']>=1 and d>=1 and all(gap*g>0 for g in chain_gaps)
        assert candidate==r['candidate'];passed[cohort].append(candidate);gaps[cohort].append(gap)
joint=[passed['original'][i] and passed['replication'][i] and gaps['original'][i]*gaps['replication'][i]>0 for i in range(10)]
pairs=[[frequencies[i],frequencies[i+1]] for i in range(9) if joint[i] and joint[i+1]]
assert pairs==summary['replicated_adjacent_pairs']
expected_reps=[s['row'] for s in selected[64:] if s['selection_rank']==4]+[128]
assert expected_reps==protocol['scan_representatives']
scan_models=rows(out/'scan-models.txt');assert len(scan_models)==9
for ident,r in zip(expected_reps,scan_models):
    m=tomllib.loads((out/f'model-{ident:03d}.toml').read_text())['model']
    assert int(r[0])==ident and int(r[2])==len(m['vs'])
    assert list(map(float,r[3:]))==m['depth'][1:]+m['vs']
root_checks=[];root_issues=[]
for f in protocol['scan_frequencies']:
    coarse,fine=[rows(out/f'scan-{n}-{f:.2f}.txt') for n in protocol['scan_segments']]
    assert len(coarse)==len(fine)==9
    for ident,a,b in zip(expected_reps,coarse,fine):
        assert int(a[0])==int(b[0])==ident
        ar,br=[list(map(float,r[6:])) for r in [a,b]]
        assert len(ar)==int(a[4]) and len(br)==int(b[4])
        assert all(x<y for rt in [ar,br] for x,y in zip(rt[:-1],rt[1:]))
        agreement=len(ar)==len(br) and all(abs(x-y)<=1e-6 for x,y in zip(ar,br))
        backend=tables[ident][frequencies.index(f)]
        error=abs(float(backend[3])-br[1]) if backend[4]=='valid' and len(br)>=2 else None
        passed_root=agreement and min(len(ar),len(br))>=2 and int(a[5])==int(b[5])==0 and error is not None and error<=1e-5
        r=next(r for r in summary['root_checks'] if r['model']==ident and r['frequency']==f)
        assert r['passed']==passed_root and r['resolution_agrees']==agreement
        assert r['coarse_roots']==ar and r['fine_roots']==br
        if error is not None:close(error,r['mode1_error'])
        root_checks.append(passed_root)
        if not passed_root:root_issues.append((ident,f))
assert summary['root_checks_passed']==all(root_checks)
assert len(summary['root_issues'])==len(root_issues)
assert summary['ready_for_bounded_added_data_inversion']==bool(pairs and all(root_checks))
commands=json.loads((out/'commands.json').read_text())
assert len(commands)==139 and sum(c['kind']=='prediction' for c in commands)==129
for p,h in protocol['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
for name,h in protocol['hashes'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
report=dict(passed=True,selection_and_physics_verified=129,query_results_verified=1290,
    disjoint_physical_cohorts=True,independent_statistics_verified=True,root_order_checks_verified=45,
    adjacent_pairs=pairs,source_hashes_unchanged=True,new_forward_calls=0,new_samples=0,
    limitations='Finite independent scans on nine representatives and five frequencies; no global completeness or statistically independent cohorts.')
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
