#!/usr/bin/env python3
"""At most 16 cold-slot failure pairs, one per occupied ladder/K; fixed mode rotation."""
import argparse
from pathlib import Path
from benchmark_crust_depth import rows,digest
from audit_rj_mode_failures import run,analyze


def select_cold(source):
    strata={};hashes={};total=0
    for c in range(4):
        folder=source/f'rj/ladder_{c}'
        paths={n:folder/f'temperature_0_{n}.txt' for n in ['local_trace','failed_models','failed_rows']}
        for p in paths.values():hashes[str(p.resolve())]=digest(p)
        formal={int(r[9]) for r in rows(paths['local_trace']) if r[0]=='0' and r[3]=='1' and r[4] in ['forward_failure','numerical_failure']}
        models={int(r[0]):r for r in rows(paths['failed_models']) if int(r[0]) in formal};assert set(models)==formal;total+=len(formal)
        for r in rows(paths['failed_rows']):
            ev=int(r[0]);q=int(r[1])
            if ev not in formal or r[2]=='valid':continue
            model=models[ev];k=int(model[2])
            strata.setdefault((c,k,q),[]).append(dict(ladder=c,temperature_slot=0,temperature=1,evaluation=ev,k=k,query=q,saved_status=r[2],saved_roots_found=int(r[3]),model_state=list(map(float,model[3:]))))
    selected=[];empty=[]
    for c in range(4):
        for k in range(1,5):
            qs=sorted(q for cc,kk,q in strata if cc==c and kk==k)
            if not qs:empty.append(dict(ladder=c,k=k));continue
            preferred=(c+k)%2
            candidates=[q for q in qs if int(q>=12)==preferred]
            if not candidates:candidates=qs
            q=candidates[(c+k)%len(candidates)]
            available=sorted(strata[c,k,q],key=lambda s:s['evaluation']);s=available[len(available)//2]
            selected.append(dict(s,row=len(selected),stratum_count=len(available),preferred_mode=preferred,eligible_queries=candidates))
    assert all(s['temperature_slot']==0 for s in selected) and len(selected)<=16
    inventory=[dict(ladder=c,k=k,query=q,count=len(v)) for (c,k,q),v in sorted(strata.items())]
    return selected,inventory,empty,hashes,total


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if a.analyze_only:analyze(a.output.resolve())
    else:run(a.output.resolve(),selector=select_cold,maximum_pairs=16,selection_kind='cold_by_ladder_K',selector_script=Path(__file__),
        selection_description='T=1 formal failures only. One pair per occupied (ladder c,K); prefer mode=(c+K)%2, fallback only if no failures in that mode. Among sorted failed query indices in that mode choose index (c+K)%len; within query select floor(N/2) ordered evaluation. Empty K not filled. No likelihood selection or retries.')
