#!/usr/bin/env python3
"""Bounded stratified scan of rejected 14-point RJ candidates. Never samples."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmark_crust_depth import digest, rows
from joint_results import read_config,write_config,clean

SOURCE=Path('results/rj-mode-refit-replication')

def save(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def select(source):
    strata={};hashes={};inventory=[];total=0
    for c in range(4):
        for t in range(4):
            folder=source/f'rj/ladder_{c}'
            paths={k:folder/f'temperature_{t}_{k}.txt' for k in ['local_trace','failed_models','failed_rows']}
            for p in paths.values():hashes[str(p.resolve())]=digest(p)
            trace=rows(paths['local_trace'])
            formal={int(r[9]) for r in trace if r[0]=='0' and r[3]=='1' and r[4] in ['forward_failure','numerical_failure']}
            models={int(r[0]):r for r in rows(paths['failed_models']) if int(r[0]) in formal}
            assert set(models)==formal;total+=len(formal)
            for r in rows(paths['failed_rows']):
                ev=int(r[0]);q=int(r[1])
                if ev not in formal or r[2]=='valid':continue
                m=models[ev];k=int(m[2]);item=dict(ladder=c,temperature_slot=t,temperature=[1,2,4,8][t],evaluation=ev,k=k,query=q,
                    saved_status=r[2],saved_roots_found=int(r[3]),model_state=list(map(float,m[3:])))
                strata.setdefault((c,k,q),[]).append(item)
    selected=[];empty=[]
    for q in range(14):
        for c in range(4):
            priority=[1+(q+c+j)%4 for j in range(4)]
            available=next((strata[(c,k,q)] for k in priority if (c,k,q) in strata),[])
            if not available:empty.append(dict(query=q,ladder=c));continue
            available=sorted(available,key=lambda x:(x['temperature_slot'],x['evaluation']))
            chosen=dict(available[len(available)//2],row=len(selected),stratum_count=len(available),K_priority=priority)
            selected.append(chosen)
    for (c,k,q),items in sorted(strata.items()):inventory.append(dict(ladder=c,k=k,query=q,count=len(items)))
    return selected,inventory,empty,hashes,total


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    for src,name in [(SOURCE/'dispweave','dispweave'),(Path('build/rj_forward_audit'),'rj_forward_audit'),(Path(__file__),'run-script.py')]:shutil.copy2(src,out/name)
    selected,inventory,empty,hashes,total=select(SOURCE)
    for p in [SOURCE/'observations.txt',SOURCE/'rj.toml',SOURCE/'truth.toml',SOURCE/'summary.json',Path('benchmarks/rj_forward_audit.cpp'),Path('tests/independent_reference.hpp')]:hashes[str(p.resolve())]=digest(p)
    obs=np.loadtxt(SOURCE/'observations.txt');assert obs.shape==(14,4)
    save(out/'selection.json',selected);save(out/'inventory.json',dict(formal_failed_models=total,strata=inventory,empty_query_ladder_cells=empty))
    protocol=dict(source=str(SOURCE.resolve()),maximum_pairs=56,actual_pairs=len(selected),queries=obs[:,[0,2]].tolist(),scan_segments=[5000,10000],
        selection='One failed model per observed query and ladder; cyclic K priority 1+(q+c+j)%4, j=0..3; first occupied K, sorted (temperature slot,evaluation), choose floor(N/2). Formal failures only; no fit selection, no fill of empty cells.',
        budget=dict(full14_prediction_calls=len(selected),independent_model_frequency_scans=2*len(selected),scan_auxiliary_fundamental_evaluations=2*len(selected),new_samples=0),
        comparison='Use saved failure and fresh full-query production status at requested mode; compare required root ordinal (mode+1) to independently scanned ordered roots. Scanner auxiliary backend output is mode0 and is NOT the requested-mode check.',
        classify='If fresh status differs, mark reproduction discrepancy. If either scan reports failure or root counts/locations disagree >1e-6, mark reference inconclusive. Stable clean scans with enough roots are suspected production miss, not automatically proven. Otherwise no requested root found by finite scan, never physical-absence proof.',
        scan_domain='.5*min(Vs) to halfspace Vs*(1-1e-8); finite real trapped-wave sign-change scan, no global completeness guarantee.',
        stopping='Fixed sample, no widening or additional chains. If suspected mismatch occurs, inspect it before any solver change; no unsupported patch.',
        source_hashes=hashes,hashes={n:digest(out/n) for n in ['dispweave','rj_forward_audit','run-script.py','selection.json','inventory.json']})
    save(out/'protocol.json',protocol);truth=read_config(SOURCE/'truth.toml');commands=[]
    for s in selected:
        i=s['row'];k=s['k'];z=s['model_state'][:k-1];v=np.array(s['model_state'][k-1:])
        model=dict(depth=[0.]+z,vs=v.tolist(),vp=(1.8*v).tolist(),rho=[2.3]*k)
        write_config(out/f'model-{i:02d}.toml',dict(model=model,queries=truth['queries']))
        cmd=[str(out/'dispweave'),'forward','--config',str(out/f'model-{i:02d}.toml'),'--output',str(out/f'forward-{i:02d}.txt'),'--root-accuracy','refined','--sensitivity','none']
        begin=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        commands.append(dict(kind='production_recheck',row=i,argv=cmd,returncode=p.returncode,seconds=time.perf_counter()-begin,stderr=p.stderr));save(out/'commands.json',commands)
        if p.returncode not in [0,2,3]:raise RuntimeError('Incomplete recheck; no retry')
        (out/f'scan-model-{i:02d}.txt').write_text(' '.join(map(str,[i,'failed',k,*s['model_state']]))+'\n')
        for n in [5000,10000]:
            cmd=[str(out/'rj_forward_audit'),str(out/f'scan-model-{i:02d}.txt'),'1.8','2.3',str(obs[s['query'],0]),'1',str(n)]
            begin=time.perf_counter()
            with (out/f'scan-{i:02d}-{n}.txt').open('w') as stdout,(out/f'scan-{i:02d}-{n}.log').open('w') as stderr:
                p=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=120)
            commands.append(dict(kind='independent_scan',row=i,segments=n,argv=cmd,returncode=p.returncode,seconds=time.perf_counter()-begin));save(out/'commands.json',commands)
            if p.returncode:raise RuntimeError('Incomplete scan; no retry')
    analyze(out)


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selection.json').read_text());records=[]
    for p,h in protocol['source_hashes'].items():assert digest(Path(p))==h
    for n,h in protocol['hashes'].items():assert digest(out/n)==h
    for s in selection:
        i=s['row'];q=s['query'];f,m=protocol['queries'][q];m=int(m)
        fresh=rows(out/f'forward-{i:02d}.txt');assert len(fresh)==14
        for r,(expected_f,expected_m) in zip(fresh,protocol['queries']):assert abs(float(r[1])-expected_f)<1e-12 and int(r[2])==expected_m
        scans=[rows(out/f'scan-{i:02d}-{n}.txt') for n in [5000,10000]];assert all(len(x)==1 for x in scans)
        scans=[x[0] for x in scans];roots=[list(map(float,x[6:])) for x in scans]
        for r,rr in zip(scans,roots):assert int(r[0])==i and int(r[1])==s['k'] and int(r[4])==len(rr)
        agree=len(roots[0])==len(roots[1]) and np.allclose(*roots,atol=1e-6,rtol=0)
        clean_scan=all(int(x[5])==0 for x in scans);reproduced=fresh[q][4]==s['saved_status']
        enough=len(roots[1])>m
        category=('reproduction_discrepancy' if not reproduced else 'reference_inconclusive' if not (agree and clean_scan)
                  else 'suspected_missed_root' if enough else 'no_requested_root_found')
        records.append(dict(**s,frequency=f,mode=m,fresh_status=fresh[q][4],root_counts=list(map(len,roots)),roots=roots,
            scan_failures=[int(x[5]) for x in scans],resolution_agrees=bool(agree),category=category))
    counts=dict(Counter(r['category'] for r in records));commands=json.loads((out/'commands.json').read_text())
    summary=dict(records=records,categories=counts,selected_pairs=len(records),unique_candidate_models=len({(r['ladder'],r['temperature_slot'],r['evaluation']) for r in records}),
        physical_models=len({(r['k'],*r['model_state']) for r in records}),sampled_K_counts=dict(Counter(r['k'] for r in records)),sampled_temperature_counts=dict(Counter(r['temperature'] for r in records)),
        original_query_pairs=sum(r['mode']==0 for r in records),added_query_pairs=sum(r['mode']==1 for r in records),
        full14_prediction_calls=sum(c['kind']=='production_recheck' for c in commands),independent_model_frequency_scans=sum(c['kind']=='independent_scan' for c in commands),
        recorded_process_seconds=sum(c['seconds'] for c in commands),new_samples=0,
        interpretation='Stratified diagnostic sample, not random frequency-weighted prevalence estimate. No requested root found is not proof of absence. Finite scans may miss close or tangential roots or suffer conditioning.')
    save(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    labels=['no_requested_root_found','reference_inconclusive','suspected_missed_root','reproduction_discrepancy']
    for idx,mode in enumerate([0,1]):
        axes[0].bar(np.arange(4)+(idx-.5)*.35,[sum(r['mode']==mode and r['category']==label for r in records) for label in labels],width=.33,label=f'mode {mode}')
    axes[0].set(xticks=range(4),xticklabels=['No requested\nroot found','Reference\ninconclusive','Suspected\nmiss','Reproduction\ndifference'],ylabel='Selected query-model pairs');axes[0].legend()
    for mode,color in [(0,'#0072B2'),(1,'#7B3294')]:
        rs=[r for r in records if r['mode']==mode]
        axes[1].scatter([r['frequency'] for r in rs],[r['root_counts'][1] for r in rs],s=45,alpha=.6,color=color,label=f'mode {mode}: scanned roots')
    axes[1].set(xlabel='Frequency (Hz)',ylabel='Fine-scan root count',yticks=range(5));axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Rejected-candidate audit | fixed stratified sample | finite scans are NOT absence certificates')
    fig.savefig(out/'failure-audit.png',dpi=150);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if a.analyze_only:analyze(a.output.resolve())
    else:run(a.output.resolve())
