#!/usr/bin/env python3
"""Bounded restart ablation at saved states; not continuous warmup or posterior validation."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from joint_results import read_config, write_config, save_json
from audit_nuts_window import digest, spectrum


def averaged_step(rows):
    # Reconstruct the original DA state since the accepted update at iteration 100.
    mu=np.log(10.)+np.log(float(rows[100,6])); average=np.log(float(rows[100,6])); hbar=0.
    for t,row in enumerate(rows[100:150],1):
        eta=1/(t+10.); hbar=(1-eta)*hbar+eta*(.99-float(row[7]))
        log_step=mu-np.sqrt(t)*hbar/.05; weight=t**(-.75)
        average=weight*log_step+(1-weight)*average
    return float(np.exp(average))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir();source=a.checkpoint.resolve()
    exe=out/'dispweave';shutil.copy2(a.exe.resolve(),exe)
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    inputs=out/'inputs';inputs.mkdir()
    for name in ['observations.txt','covariance.txt']:shutil.copyfile(source/'inputs'/name,inputs/name)
    jobs=[];geometry=[]
    for c in [0,1]:
        base=read_config(source/f'chain-{c}/local-gn.toml')
        rows=np.loadtxt(source/f'inputs/progress-{c}.txt',dtype=str,skiprows=1)
        step=averaged_step(rows)
        masses={k:np.loadtxt(source/f'chain-{c}/mass-{k}.txt') for k in ['before','after']}
        g=np.loadtxt(source/f'chain-{c}/local-gn/chain_0_initial_mass.txt')
        ratio=spectrum(masses['after'],g)['max_frequency_squared']/spectrum(masses['before'],g)['max_frequency_squared']
        geometry.append(dict(chain=c,step=step,previous_step=float(rows[149,6]),stiffness_ratio=ratio,accepted=ratio<=4))
        for variant in ['standard','conservative','conservative-guard']:
            metric='before' if variant=='conservative-guard' and ratio>4 else 'after'
            for seed in [17,2026]:
                cfg=copy.deepcopy(base);opt=cfg['nuts'];opt.pop('initial_metric')
                opt.update(warmup=20,draws=10,max_depth=8,step_size=step,adapt_step_size=True,
                    metric_adaptation='none',find_initial_step_size=True,conservative_warmup=variant!='standard',
                    mass=masses[metric].tolist(),seed=seed)
                path=inputs/f'{c}-{variant}-{seed}.toml';write_config(path,cfg)
                jobs.append((c,variant,seed,path,out/path.stem))
    save_json(out/'protocol.json',dict(geometry=geometry,seeds=[17,2026],warmup=20,draws=10,workers=2,
        timeout_seconds=30,max_depth=8,binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        inputs={p.name:digest(p) for p in inputs.iterdir()},
        interpretation='Restarted RNG/DA at fixed checkpoints; guard preselects saved metric using saved local GN, then metric is frozen. Not a full adaptive guard benchmark or posterior validation.'))
    def run(job):
        c,variant,seed,path,dest=job;start=time.perf_counter()
        with dest.with_suffix('.log').open('w') as log:
            try:
                r=subprocess.run([str(exe),'nuts','--config',str(path),'--data',str(inputs/'observations.txt'),
                    '--covariance',str(inputs/'covariance.txt'),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=30)
                code=r.returncode
            except subprocess.TimeoutExpired:code='timeout'
        record=dict(chain=c,variant=variant,seed=seed,returncode=code,seconds=time.perf_counter()-start)
        path=dest/'chain_0_progress.txt'
        lines=path.read_text().splitlines()[1:] if path.exists() else []
        rows=np.array([line.split() for line in lines if len(line.split())==53])
        for phase,flag in [('warmup','1'),('formal','0')]:
            r=rows[rows[:,1]==flag] if len(rows) else []
            record[phase]=dict(count=len(r),evaluations=sum(int(x[5]) for x in r),divergences=sum(int(x[3]) for x in r),
                invalid=sum(x[2] not in ['valid','outside_support'] for x in r),depth8=sum(int(x[4])==8 for x in r))
            if len(r):record[phase].update(mean_acceptance=float(np.mean([float(x[7]) for x in r])),last_step=float(r[-1][6]))
        if code in [0,5]:
            record['cost']={k:int(v) for k,v in (line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines())}
        save_json(dest.with_suffix('.json'),record)
        print(record,flush=True);return record
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(run,jobs))
    save_json(out/'summary.json',dict(geometry=geometry,runs=records,posterior_validation=False))


if __name__=='__main__':main()
