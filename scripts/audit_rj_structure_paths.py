#!/usr/bin/env python3
"""Independent categorical route count and cached prediction provenance checks."""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

out=Path(sys.argv[1]).resolve();report=json.loads((out/'summary.json').read_text())
sources={'transdim':Path('results/rj-lvz-transfer/production'),
         'fixed_independent':Path('results/rj-local-k4/independent'),
         'fixed_local':Path('results/rj-local-k4/local')}
def rows(path):return [np.fromstring(s,sep=' ') for s in path.read_text().splitlines() if s and not s.startswith('#')]
def label(r):
    k=int(r[1])
    if k!=4:return f'K{k}'
    a,b,c,d=r[7:]
    if a-b>=.1 and c-b>=.1:return 'A'
    if b-c>=.1 and d-c>=.1:return 'B'
    return 'K4_no_dip'
for stage in report['stages']:
    source=sources[stage['source']];counts=Counter();arrivals=Counter()
    for c in range(4):
        folder=source/f'rj/ladder_{c}';ids=np.loadtxt(folder/'replicas.txt',dtype=int)[2499:,2:]
        states=[ [rows(folder/f'temperature_{t}_warmup_models.txt')[-1]]+rows(folder/f'temperature_{t}_models.txt') for t in range(4)]
        cats=np.array([[label(r) for r in s] for s in states]).T
        for family in ['A','B']:
            observed=int(np.sum(cats[1:,0]==family));expected=next(g for g in stage['groups'] if g['family']==family)['per_chain_count'][c]
            assert observed==expected
        for replica in range(4):
            slot=np.argmax(ids==replica,axis=1)
            path=cats[np.arange(len(cats)),slot]
            known=np.flatnonzero(np.isin(path,['A','B']))
            for a,b in zip(known[:-1],known[1:]):
                if path[a]==path[b]:continue
                middle=set(path[a+1:b])
                kind='via_K3' if 'K3' in middle else 'via_other_K' if middle & {'K1','K2'} else 'via_K4_no_dip' if middle else 'direct_K4'
                counts[f'{kind}:{path[a]}->{path[b]}']+=1
                arrivals[f'{kind}:T{[1,2,4,8][slot[b]]}']+=1
    assert dict(counts)==stage['replica_routes']
    assert dict(arrivals)==stage['route_arrival_temperatures']
prediction=Path('results/rj-lvz-transfer/predict');selected=rows(prediction/'selected_models.txt');trace=np.loadtxt(prediction/'draws.txt',dtype=str,skiprows=1)
for c in range(4):
    chain=rows(sources['transdim']/f'rj/chain_{c}_models.txt')
    for i in np.flatnonzero(trace[:,1].astype(int)==c):
        np.testing.assert_array_equal(selected[i][1:],chain[int(trace[i,2])][1:])
for group in report['predictions']:
    chosen=[i for i,r in enumerate(selected) if label(r)==group['family']]
    assert len(chosen)==group['selected_count']
    assert [i for i in chosen if trace[i,3]!='valid']==group['failed_rows']
for path,sha in report['input_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
(out/'audit.json').write_text(json.dumps(dict(passed=True,replica_route_counts_independently_verified=True,
    cold_family_counts_verified=True,existing_prediction_source_indices_verified=400,
    input_hashes_unchanged=True,new_forward_calls=0,new_samples=0),indent=2)+'\n')
print('Replica paths, cold groups, prediction provenance and input hashes verified.')
