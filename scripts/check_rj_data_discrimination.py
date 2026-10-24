#!/usr/bin/env python3
"""Bounded, preregistered prediction-only screening of existing K4 LVZ families."""
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
from joint_results import read_config, write_config, save_json

SOURCE = Path('results/rj-lvz-transfer/production')
SIGMA = .03
PER_CHAIN = 8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith('#')]


def family(r):
    if int(r[1]) != 4:
        return None
    v = np.asarray(r[7:], float)
    dips = np.minimum(v[:-2]-v[1:-1], v[2:]-v[1:-1])
    return ('A' if np.argmax(dips) == 0 else 'B') if dips.max() >= .1 else None


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE/'dispweave', out/'dispweave')
    shutil.copy2(__file__, out/'run-script.py')
    truth = read_config(SOURCE/'truth.toml')
    queries = [dict(frequency=f, mode=0, band='existing') for f in truth['queries']['frequency_hz']]
    queries += [dict(frequency=f, mode=0, band='low') for f in [.05,.075,.10,.125]]
    queries += [dict(frequency=f, mode=0, band='high') for f in [1.1,1.4,1.8,2.3,3.,4.]]
    queries += [dict(frequency=f, mode=m, band=f'mode{m}') for m in [1,2]
                for f in [.3,.45,.6,.9,1.2,1.8,2.5,3.5]]
    selected = []
    source_hashes = {str(SOURCE/n):sha(SOURCE/n) for n in ['truth.toml','observations.txt','covariance.txt','rj.toml']}
    for c in range(4):
        path = SOURCE/f'rj/chain_{c}_models.txt'
        source_hashes[str(path)] = sha(path)
        chain = rows(path)
        for label in ['A','B']:
            indices = [i for i,r in enumerate(chain) if family(r) == label]
            assert len(indices) >= PER_CHAIN
            for j in range(PER_CHAIN):
                i = indices[j*len(indices)//PER_CHAIN]
                selected.append(dict(row=len(selected),chain=c,draw=i,family=label,
                                     family_available=len(indices),model=list(map(float,chain[i]))))
    save_json(out/'selection.json', selected)
    protocol = dict(queries=queries, sigma=SIGMA, per_family_per_chain=PER_CHAIN,
        models=64, target_models=1, model_forward_calls=65, frequency_mode_queries=65*len(queries),
        selection='For each family within each cold chain, ranks floor(j*N/8), j=0..7. Repeats retained; no fit filtering.',
        interpretation='Balanced conditional diagnostic, NOT posterior-weighted prediction; chains remain unaccepted and samples correlated.',
        failure_policy='Every status retained. A frequency is scored only if all 64 states and target are valid. Missing/unresolved modes are not evidence of discrimination; no resampling or retries.',
        criterion='At an added query: |median A - median B|/sigma >= 1 AND |gap|/sqrt((varA+varB)/2+sigma^2) >= 1; all four chain median gaps have the same sign. Recommend a band only for two adjacent eligible queries meeting all criteria. Descriptive heuristic, not calibrated hypothesis test or expected information gain.',
        noise='Hypothetical constant independent sigma=.03 km/s for screening, not a measured higher-mode error model. No synthetic noise or data generated.',
        source_hashes=source_hashes, hashes={n:sha(out/n) for n in ['dispweave','run-script.py','selection.json']},
        new_samples=0, fixed_budget=True)
    save_json(out/'protocol.json',protocol)
    execute(out)


def execute(out):
    protocol=json.loads((out/'protocol.json').read_text())
    selected=json.loads((out/'selection.json').read_text())
    queries=protocol['queries'];truth=read_config(SOURCE/'truth.toml')
    for path,h in protocol['source_hashes'].items():assert sha(Path(path))==h
    for name,h in protocol['hashes'].items():assert sha(out/name)==h
    query_doc=dict(frequency_hz=[q['frequency'] for q in queries],mode=[q['mode'] for q in queries])
    commands=json.loads((out/'commands.json').read_text()) if (out/'commands.json').exists() else []
    for i,item in enumerate(selected+[None]):
        if item is None:
            model=truth['model'];name='target'
        else:
            r=np.array(item['model']);v=r[7:];name=f'model-{i:02d}'
            model=dict(depth=[0.]+r[4:7].tolist(),vs=v.tolist(),vp=(1.8*v).tolist(),rho=[2.3]*4)
        if (out/f'{name}.txt').exists():
            assert len(rows(out/f'{name}.txt'))==len(queries)
            if not any(c['name']==name for c in commands):
                commands.append(dict(name=name,seconds=None,returncode=None,recovered_existing_output=True))
            continue
        write_config(out/f'{name}.toml',dict(model=model,queries=query_doc))
        cmd=[str((out/'dispweave').resolve()),'forward','--config',str(out/f'{name}.toml'),
             '--output',str(out/f'{name}.txt'),'--root-accuracy','refined','--sensitivity','none']
        start=time.perf_counter()
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        commands.append(dict(name=name,argv=cmd,returncode=p.returncode,seconds=time.perf_counter()-start,stdout=p.stdout,stderr=p.stderr))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if p.returncode not in [0,2,3]:
            raise RuntimeError(f'{name} did not complete; no retries: {p.stderr}')
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    analyze(out)


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text());selected=json.loads((out/'selection.json').read_text())
    for path,h in protocol['source_hashes'].items():assert sha(Path(path))==h
    for name,h in protocol['hashes'].items():assert sha(out/name)==h
    queries=protocol['queries'];n=len(queries)
    values=np.full((65,n),np.nan);statuses=[]
    for i in range(65):
        table=rows(out/('target.txt' if i==64 else f'model-{i:02d}.txt'))
        assert len(table)==n
        statuses.append([r[4] for r in table])
        for j,(r,q) in enumerate(zip(table,queries)):
            assert abs(float(r[1])-q['frequency'])<1e-12 and int(r[2])==q['mode']
            if r[4]=='valid':
                values[i,j]=float(r[3]);assert np.isfinite(values[i,j])
    valid=np.isfinite(values).all(axis=0)
    # Check the original likelihood to ensure physics and retained model reconstruction match.
    obs=np.loadtxt(SOURCE/'observations.txt');cov=np.loadtxt(SOURCE/'covariance.txt')
    assert np.isfinite(values[:64,:12]).all()
    inv=np.linalg.inv(cov);norm=np.linalg.slogdet(cov)[1]+12*np.log(2*np.pi)
    ll=np.array([-.5*((v[:12]-obs[:,1])@inv@(v[:12]-obs[:,1])+norm) for v in values[:64]])
    ll_error=float(np.max(abs(ll-np.array([s['model'][2] for s in selected]))))
    assert ll_error<1e-6,ll_error
    report=[]
    for j,q in enumerate(queries):
        row=dict(**q,eligible=bool(valid[j]),statuses=dict(Counter(s[j] for s in statuses)),
                 target_valid=statuses[-1][j]=='valid',valid_per_family={f:sum(statuses[i][j]=='valid' for i,s in enumerate(selected) if s['family']==f) for f in ['A','B']},candidate=False)
        if valid[j]:
            groups={f:values[[i for i,s in enumerate(selected) if s['family']==f],j] for f in ['A','B']}
            a,b=groups['A'],groups['B'];aq,bq=[np.quantile(v,[.025,.5,.975]) for v in [a,b]]
            gap=float(aq[1]-bq[1]);denom=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2+SIGMA**2)
            chain_gaps=[float(np.median(values[[i for i,s in enumerate(selected) if s['chain']==c and s['family']=='A'],j])-np.median(values[[i for i,s in enumerate(selected) if s['chain']==c and s['family']=='B'],j])) for c in range(4)]
            consistent=all(g*gap>0 for g in chain_gaps)
            row.update(quantiles={'A':aq.tolist(),'B':bq.tolist()},gap_sigma=abs(gap)/SIGMA,
                dispersion_adjusted_gap=float(abs(gap)/denom),chain_gaps=chain_gaps,chain_sign_consistent=consistent,
                intervals_overlap=bool(max(aq[0],bq[0])<=min(aq[2],bq[2])),
                candidate=bool(q['band']!='existing' and abs(gap)>=SIGMA and abs(gap)/denom>=1 and consistent))
        report.append(row)
    bands=[]
    for band in ['low','high','mode1','mode2']:
        r=[x for x in report if x['band']==band]
        pairs=[[a['frequency'],b['frequency']] for a,b in zip(r[:-1],r[1:]) if a['candidate'] and b['candidate']]
        if pairs:bands.append(dict(band=band,adjacent_pairs=pairs))
    commands=json.loads((out/'commands.json').read_text())
    summary=dict(queries=report,promising_bands=bands,baseline_log_likelihood_max_error=ll_error,
                 model_forward_calls=len(commands),frequency_mode_queries=len(commands)*n,
                 recorded_seconds=sum(c['seconds'] or 0 for c in commands),timing_incomplete=any(c['seconds'] is None for c in commands),new_samples=0,
                 physical_model_duplicates=64-len({tuple(s['model'][4:]) for s in selected}),
                 interpretation='Screening only; balanced, correlated samples from incomplete posterior. No claim of identifiability, posterior convergence, calibrated significance or practical measurability.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,band in zip(axes.flat,['existing','low','high','mode1','mode2']):
        r=[(i,x) for i,x in enumerate(report) if x['band']==band]
        for f,color in [('A','#0072B2'),('B','#7B3294')]:
            fs=[x['frequency'] for _,x in r]
            quant=np.array([x['quantiles'][f] if x['eligible'] else [np.nan]*3 for _,x in r])
            ax.plot(fs,quant[:,1],color=color,label=f'Family {f} median')
            ax.fill_between(fs,quant[:,0],quant[:,2],color=color,alpha=.12)
        target=[values[64,i] for i,_ in r]
        ax.plot(fs,target,color='#228833',ls=':',label='Target')
        for _,x in r:
            if not x['eligible']:ax.plot(x['frequency'],.04,marker='x',color='black',transform=ax.get_xaxis_transform())
            if x['candidate']:ax.axvline(x['frequency'],color='#999999',ls=':',alpha=.6)
        ax.set(title=band,xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    ax=axes.flat[-1]
    for band in ['existing','low','high','mode1','mode2']:
        r=[x for x in report if x['band']==band and x['eligible']]
        ax.plot([x['frequency'] for x in r],[x['dispersion_adjusted_gap'] for x in r],'.-',label=band)
    ax.axhline(1,color='black',ls='--');ax.set(xlabel='Frequency (Hz)',ylabel='Gap / sqrt(pooled variance + noise variance)',title='Dispersion-adjusted separation');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Existing K4 LVZ families | 32 states each, balanced by chain | sigma = 0.03 km/s\nPointwise empirical 95% bands; x = missing/unresolved response, not scored; no new sampling')
    fig.savefig(out/'discrimination.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(promising_bands=bands,eligible_queries=int(sum(valid)),calls=len(commands),baseline_error=ll_error)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyze-only',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.analyze_only:analyze(args.output.resolve())
    elif args.resume:execute(args.output.resolve())
    else:prepare(args.output.resolve())
