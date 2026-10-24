#!/usr/bin/env python3
"""Fixed-budget dense mode-1 prediction replication and independent root scan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joint_results import read_config, write_config

SOURCE=Path('results/rj-lvz-transfer/production')
OLD=Path('results/rj-data-discrimination')
FREQUENCIES=[.45,.50,.55,.60,.65,.70,.75,.80,.85,.90]
SCAN_FREQUENCIES=[.45,.55,.60,.65,.90]
SIGMA=.03


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith('#')]


def family(r):
    if int(r[1])!=4:return None
    a,b,c,d=map(float,r[7:])
    if min(a-b,c-b)>=.1:return 'A'
    if min(b-c,d-c)>=.1:return 'B'
    return None


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    for src,name in [(SOURCE/'dispweave','dispweave'),(Path('build/rj_forward_audit'),'rj_forward_audit'),(Path(__file__),'run-script.py')]:shutil.copy2(src,out/name)
    old=json.loads((OLD/'selection.json').read_text())
    assert len(old)==64
    excluded={tuple(s['model'][4:]) for s in old}
    selection=[dict(**s,cohort='original') for s in old]
    for c in range(4):
        chain=rows(SOURCE/f'rj/chain_{c}_models.txt')
        for label in ['A','B']:
            eligible=[i for i,r in enumerate(chain) if family(r)==label and tuple(map(float,r[4:])) not in excluded]
            assert len(eligible)>=8
            for j in range(8):
                i=eligible[(2*j+1)*len(eligible)//16]
                selection.append(dict(row=len(selection),cohort='replication',chain=c,draw=i,family=label,
                    eligible_count=len(eligible),selection_rank=j,model=list(map(float,chain[i]))))
    save(out/'selection.json',selection)
    representatives=[s['row'] for s in selection if s['cohort']=='replication' and s['selection_rank']==4]+[128]
    truth=read_config(SOURCE/'truth.toml')['model']
    scan_lines=['# id kind K interfaces Vs']
    for i in representatives:
        if i==128:scan_lines.append(' '.join(map(str,[i,'target',3,*truth['depth'][1:],*truth['vs']])))
        else:
            s=selection[i];scan_lines.append(' '.join(map(str,[i,s['family'],4,*s['model'][4:]])))
    (out/'scan-models.txt').write_text('\n'.join(scan_lines)+'\n')
    source_files=[SOURCE/'truth.toml',SOURCE/'rj.toml',OLD/'selection.json',OLD/'protocol.json',
        Path('benchmarks/rj_forward_audit.cpp'),Path('tests/independent_reference.hpp')]
    source_files += [SOURCE/f'rj/chain_{c}_models.txt' for c in range(4)]
    source_files += [OLD/f'model-{i:02d}.txt' for i in range(64)]+[OLD/'target.txt']
    protocol=dict(frequencies=FREQUENCIES,mode=1,sigma=SIGMA,scan_frequencies=SCAN_FREQUENCIES,scan_segments=[5000,10000],
        prediction_models=129,prediction_queries=1290,scan_representatives=representatives,
        independent_model_frequency_scans=90,independent_initial_grid_points=9*5*(5001+10001),
        scan_backend_fundamental_evaluations=90,new_samples=0,
        selection='Original 64 retained; new cohort excludes every original physical model, then each chain/family selects midpoint ranks floor((j+.5)*N/8), j=0..7. No fit selection; repeats within new cohort retained.',
        interpretation='Disjoint physical models do not imply independent samples; same unaccepted chains, balanced family/chain diagnostic, not posterior-weighted probabilities.',
        representatives='New-cohort rank j=4 for each chain/family plus target, fixed before new predictions.',
        point_rule='All 64 cohort states and target valid; abs median gap/sigma>=1; D=abs gap/sqrt((sample var A+sample var B)/2+sigma^2)>=1; four chain median gaps same sign.',
        band_rule='Two adjacent frequencies pass in BOTH cohorts, with the same A-B sign across cohorts at each point. No pooling cohorts to pass.',
        root_rule='All 45 representative/frequency pairs: both finite scans have >=2 ordered roots, no reported scan failures, counts and root locations agree within 1e-6 km/s; production mode1 agrees with second root within 1e-5 km/s. Any issue prevents full validation.',
        next_stage_rule='Only replicated adjacent band AND root_rule passing supports a bounded added-data inversion; otherwise stop. No new data or chains in this run.',
        failure_policy='Keep every status, no replacement or retries. Incomplete query cannot score. Finite sign scans are not global completeness or continuous-mode proofs.',
        source_hashes={str(p):sha(p) for p in source_files},
        hashes={name:sha(out/name) for name in ['dispweave','rj_forward_audit','run-script.py','selection.json','scan-models.txt']})
    save(out/'protocol.json',protocol)
    commands=[]
    for i in range(129):
        if i==128:model=truth
        else:
            r=selection[i]['model'];v=np.array(r[7:])
            model=dict(depth=[0.]+r[4:7],vs=v.tolist(),vp=(1.8*v).tolist(),rho=[2.3]*4)
        config=out/f'model-{i:03d}.toml';write_config(config,dict(model=model,queries=dict(frequency_hz=FREQUENCIES,mode=[1]*10)))
        cmd=[str(out/'dispweave'),'forward','--config',str(config),'--output',str(out/f'model-{i:03d}.txt'),'--root-accuracy','refined','--sensitivity','none']
        begin=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        commands.append(dict(kind='prediction',model=i,argv=cmd,returncode=p.returncode,seconds=time.perf_counter()-begin,stderr=p.stderr))
        save(out/'commands.json',commands)
        if p.returncode not in [0,2,3]:raise RuntimeError(f'Prediction incomplete: {i}: {p.stderr}')
    for segments in [5000,10000]:
        for f in SCAN_FREQUENCIES:
            name=f'scan-{segments}-{f:.2f}'
            cmd=[str(out/'rj_forward_audit'),str(out/'scan-models.txt'),'1.8','2.3',str(f),'9',str(segments)]
            begin=time.perf_counter()
            with (out/f'{name}.txt').open('w') as stdout,(out/f'{name}.log').open('w') as stderr:
                p=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=120)
            commands.append(dict(kind='independent_scan',segments=segments,frequency=f,argv=cmd,returncode=p.returncode,seconds=time.perf_counter()-begin))
            save(out/'commands.json',commands)
            if p.returncode:raise RuntimeError(f'Scan incomplete: {name}')
            print(f'Completed {name}',flush=True)
    analyze(out)


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text());selected=json.loads((out/'selection.json').read_text())
    for path,h in protocol['source_hashes'].items():assert sha(Path(path))==h
    for name,h in protocol['hashes'].items():assert sha(out/name)==h
    outputs=[rows(out/f'model-{i:03d}.txt') for i in range(129)]
    values=np.full((129,10),np.nan)
    for i,rs in enumerate(outputs):
        assert len(rs)==10
        for j,(r,f) in enumerate(zip(rs,FREQUENCIES)):
            assert abs(float(r[1])-f)<1e-12 and int(r[2])==1
            if r[4]=='valid':values[i,j]=float(r[3]);assert np.isfinite(values[i,j])
    old_error=0.
    for i in range(65):
        old=rows(OLD/('target.txt' if i==64 else f'model-{i:02d}.txt'));new=outputs[128 if i==64 else i]
        for r in old:
            if int(r[2])==1 and float(r[1]) in [.45,.6,.9]:
                n=new[FREQUENCIES.index(float(r[1]))];assert r[4]==n[4]
                if r[4]=='valid':old_error=max(old_error,abs(float(r[3])-float(n[3])))
    assert old_error<1e-8
    cohorts={}
    for cohort in ['original','replication']:
        chosen=[s for s in selected if s['cohort']==cohort];stats=[]
        for j,f in enumerate(FREQUENCIES):
            status=Counter(outputs[s['row']][j][4] for s in chosen)
            eligible=all(np.isfinite(values[s['row'],j]) for s in chosen) and np.isfinite(values[128,j])
            r=dict(frequency=f,statuses=dict(status),eligible=bool(eligible),candidate=False)
            if eligible:
                a,b=[values[[s['row'] for s in chosen if s['family']==label],j] for label in ['A','B']]
                aq,bq=[np.quantile(x,[.025,.5,.975]) for x in [a,b]];gap=float(aq[1]-bq[1])
                d=abs(gap)/np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2+SIGMA**2)
                gaps=[float(np.median(values[[s['row'] for s in chosen if s['family']=='A' and s['chain']==c],j])-np.median(values[[s['row'] for s in chosen if s['family']=='B' and s['chain']==c],j])) for c in range(4)]
                r.update(quantiles={'A':aq.tolist(),'B':bq.tolist()},gap=gap,gap_sigma=abs(gap)/SIGMA,D=float(d),chain_gaps=gaps,
                    candidate=bool(abs(gap)/SIGMA>=1 and d>=1 and all(g*gap>0 for g in gaps)),
                    intervals_overlap=bool(max(aq[0],bq[0])<=min(aq[2],bq[2])))
            stats.append(r)
        cohorts[cohort]=stats
    joint=[a['candidate'] and b['candidate'] and a['gap']*b['gap']>0 for a,b in zip(cohorts['original'],cohorts['replication'])]
    pairs=[[FREQUENCIES[j],FREQUENCIES[j+1]] for j in range(9) if joint[j] and joint[j+1]]
    root_records=[];issues=[]
    for f in SCAN_FREQUENCIES:
        tables={n:rows(out/f'scan-{n}-{f:.2f}.txt') for n in [5000,10000]}
        assert all(len(t)==9 for t in tables.values())
        for j,ident in enumerate(protocol['scan_representatives']):
            coarse,fine=[tables[n][j] for n in [5000,10000]]
            assert int(coarse[0])==int(fine[0])==ident
            roots=[np.array(r[6:],float) for r in [coarse,fine]]
            assert all(len(rt)==int(row[4]) for rt,row in zip(roots,[coarse,fine]))
            agree=len(roots[0])==len(roots[1]) and np.allclose(*roots,rtol=0,atol=1e-6)
            backend=values[ident,FREQUENCIES.index(f)]
            error=float(abs(backend-roots[1][1])) if np.isfinite(backend) and len(roots[1])>=2 else None
            passed=bool(agree and min(map(len,roots))>=2 and int(coarse[5])==int(fine[5])==0 and error is not None and error<=1e-5)
            record=dict(model=ident,frequency=f,coarse_roots=roots[0].tolist(),fine_roots=roots[1].tolist(),
                scan_failures=[int(coarse[5]),int(fine[5])],resolution_agrees=bool(agree),mode1_error=error,passed=passed)
            root_records.append(record)
            if not passed:issues.append(record)
    commands=json.loads((out/'commands.json').read_text())
    summary=dict(cohorts=cohorts,replicated_adjacent_pairs=pairs,root_checks=root_records,root_issues=issues,
        root_checks_passed=not issues,ready_for_bounded_added_data_inversion=bool(pairs and not issues),
        original_overlap_max_phase_error=old_error,prediction_calls=sum(c['kind']=='prediction' for c in commands),
        independent_scan_processes=sum(c['kind']=='independent_scan' for c in commands),independent_model_frequency_scans=90,
        seconds_by_kind={k:sum(c['seconds'] for c in commands if c['kind']==k) for k in ['prediction','independent_scan']},
        new_samples=0,replication_unique_models=len({tuple(s['model'][4:]) for s in selected if s['cohort']=='replication'}))
    save(out/'summary.json',summary)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(cohort,stats) in zip(axes[0],cohorts.items()):
        for label,color in [('A','#0072B2'),('B','#7B3294')]:
            q=np.array([r['quantiles'][label] if r['eligible'] else [np.nan]*3 for r in stats])
            ax.plot(FREQUENCIES,q[:,1],'.-',color=color,label=f'{label} median')
            ax.fill_between(FREQUENCIES,q[:,0],q[:,2],color=color,alpha=.12)
        ax.plot(FREQUENCIES,values[128],':',color='#228833',label='Target')
        ax.set(title=f'{cohort}: 32 states per family',ylabel='Mode 1 phase velocity (km/s)');ax.legend(fontsize=8)
    for cohort,stats in cohorts.items():
        axes[1,0].plot(FREQUENCIES,[r.get('gap_sigma',np.nan) for r in stats],'.-',label=cohort)
        axes[1,1].plot(FREQUENCIES,[r.get('D',np.nan) for r in stats],'.-',label=cohort)
    axes[1,0].set(ylabel='Absolute median gap / sigma');axes[1,1].set(ylabel='Gap / sqrt(pooled variance + sigma squared)')
    for ax in axes[1]:ax.axhline(1,color='black',ls='--');ax.legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Frequency (Hz)');ax.grid(alpha=.2)
    fig.suptitle(f'Dense mode-1 check | balanced, correlated states | sigma=.03 km/s\nReplicated adjacent pairs: {len(pairs)}; root checks passed: {45-len(issues)}/45; no new sampling')
    fig.savefig(out/'followup.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(pairs=pairs,root_issues=len(issues),ready=summary['ready_for_bounded_added_data_inversion'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if a.analyze_only:analyze(a.output.resolve())
    else:prepare(a.output.resolve())
