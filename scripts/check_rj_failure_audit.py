#!/usr/bin/env python3
"""Independent selection and full-response provenance check; no new root evaluation."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np
from benchmark_crust_depth import digest,rows
from joint_results import read_config

out=Path(sys.argv[1]);protocol=json.loads((out/'protocol.json').read_text());source=Path(protocol['source'])
selected=json.loads((out/'selection.json').read_text());summary=json.loads((out/'summary.json').read_text())
strata=defaultdict(list);model_lookup={};response_lookup={};total=0;temperature_failures=Counter()
for c in range(4):
    chosen_by_slot=defaultdict(set)
    for s in selected:
        if s['ladder']==c:chosen_by_slot[s['temperature_slot']].add(s['evaluation'])
    for t in range(4):
        folder=source/f'rj/ladder_{c}'
        trace=rows(folder/f'temperature_{t}_local_trace.txt')
        formal={int(r[9]) for r in trace if r[0]=='0' and r[3]=='1' and r[4] in ['forward_failure','numerical_failure']}
        models={int(r[0]):r for r in rows(folder/f'temperature_{t}_failed_models.txt') if int(r[0]) in formal}
        assert set(models)==formal;total+=len(formal);temperature_failures[[1,2,4,8][t]]+=len(formal)
        for r in rows(folder/f'temperature_{t}_failed_rows.txt'):
            ev=int(r[0]);q=int(r[1])
            if ev not in formal:continue
            if r[2]!='valid':strata[c,int(models[ev][2]),q].append((t,ev))
            if ev in chosen_by_slot[t]:response_lookup[c,t,ev,q]=r
        for ev in chosen_by_slot[t]:model_lookup[c,t,ev]=models[ev]
assert total==json.loads((source/'summary.json').read_text())['formal_forward_failures']
expected=[]
for q in range(14):
    for c in range(4):
        found=False
        for offset in range(4):
            k=(q+c+offset)%4+1
            candidates=strata.get((c,k,q),[])
            if candidates:
                t,ev=sorted(candidates)[len(candidates)//2];expected.append((c,t,ev,k,q,len(candidates)));found=True;break
assert expected==[(s['ladder'],s['temperature_slot'],s['evaluation'],s['k'],s['query'],s['stratum_count']) for s in selected]
counts=Counter();status_comparisons=0;max_phase_error=0.;positive_control_errors=[]
truth=read_config(source/'truth.toml')
for s,rec in zip(selected,summary['records']):
    i=s['row'];c=s['ladder'];t=s['temperature_slot'];ev=s['evaluation'];k=s['k'];q=s['query'];m=int(protocol['queries'][q][1])
    raw=model_lookup[c,t,ev];state=list(map(float,raw[3:]));assert state==s['model_state']
    cfg=read_config(out/f'model-{i:02d}.toml');v=state[k-1:]
    assert cfg['model']==dict(depth=[0.]+state[:k-1],vs=v,vp=[1.8*x for x in v],rho=[2.3]*k)
    assert cfg['queries']==truth['queries']
    fresh=rows(out/f'forward-{i:02d}.txt');assert len(fresh)==14
    for j,r in enumerate(fresh):
        original=response_lookup[c,t,ev,j];assert r[4]==original[2];status_comparisons+=1
        assert int(r[2])==int(protocol['queries'][j][1]);assert abs(float(r[1])-protocol['queries'][j][0])<1e-12
        if r[4]=='valid':max_phase_error=max(max_phase_error,abs(float(r[3])-float(original[4])))
    assert max_phase_error<=1e-8
    saved=response_lookup[c,t,ev,q];assert saved[2]==s['saved_status'] and int(saved[3])==s['saved_roots_found']
    rs=[rows(out/f'scan-{i:02d}-{n}.txt')[0] for n in [5000,10000]]
    roots=[list(map(float,r[6:])) for r in rs]
    assert all(int(r[0])==i and int(r[1])==k and int(r[4])==len(rt) for r,rt in zip(rs,roots))
    assert all(a<b for rt in roots for a,b in zip(rt[:-1],rt[1:]))
    agree=len(roots[0])==len(roots[1]) and all(abs(a-b)<=1e-6 for a,b in zip(*roots))
    failures=[int(r[5]) for r in rs]
    category='reproduction_discrepancy' if fresh[q][4]!=saved[2] else 'reference_inconclusive' if not agree or any(failures) else 'suspected_missed_root' if len(roots[1])>=m+1 else 'no_requested_root_found'
    assert category==rec['category'] and roots==rec['roots'] and failures==rec['scan_failures'];counts[category]+=1
    for r,rt in zip(rs,roots):
        if rt and r[2]=='valid':positive_control_errors.append(abs(float(r[3])-rt[0]))
assert dict(counts)==summary['categories']
for p,h in protocol['source_hashes'].items():assert digest(Path(p))==h
for name,h in protocol['hashes'].items():assert digest(out/name)==h
commands=json.loads((out/'commands.json').read_text());assert len(commands)==3*len(selected)
report=dict(passed=True,selected_pairs_verified=len(selected),formal_failed_candidates=total,formal_failures_by_temperature=dict(temperature_failures),
    all_saved_response_statuses_verified=status_comparisons,maximum_saved_valid_phase_error=max_phase_error,
    auxiliary_fundamental_matches_with_positive_root=len(positive_control_errors),max_auxiliary_first_root_error=max(positive_control_errors,default=None),
    categories_independently_verified=dict(counts),source_hashes_unchanged=True,new_forward_calls=0,
    limitations='No selected T=1 candidates; stratified scan is not a random prevalence estimate or root absence certificate.')
(out/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
