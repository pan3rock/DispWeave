#!/usr/bin/env python3
"""Bounded prior sensitivity run: only physical-Vs prior sigma changes."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmark_nuts_warmup import digest,progress_rows,phase_stats
from benchmark_crust_nuts import analyze
from joint_results import read_config,write_config,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=a.reference.resolve();out=a.output.resolve();base=read_config(ref/'nuts.toml')
    assert json.loads((ref/'acceptance.json').read_text())['accepted']
    assert base['nuts']['prior_sigma_vs']==[.1]*45 and base['nuts']['find_initial_step_size']
    assert base['nuts']['target_accept']==.95 and base['nuts']['warmup']==base['nuts']['draws']==1000
    assert base['nuts']['initial_metric']=='start_gauss_newton' and len(base['inversion']['starts_vs'])==4
    out.mkdir();inputs=out/'inputs';inputs.mkdir()
    for n in ['nuts.toml','observations.txt','covariance.txt','diagnostics.json','predictive-summary.json']:
        shutil.copyfile(ref/n,inputs/n)
        if n in ['observations.txt','covariance.txt']:shutil.copyfile(ref/n,out/n)
    for c in range(4):shutil.copyfile(ref/f'nuts/chain_{c}_vs.txt',inputs/f'chain_{c}_vs.txt')
    cfg=copy.deepcopy(base);cfg['nuts']['prior_sigma_vs']=[.2]*45;write_config(out/'nuts.toml',cfg)
    exe=out/'dispweave';shutil.copy2(a.exe.resolve(),exe);shutil.copyfile(Path(__file__),out/Path(__file__).name)
    jobs=[]
    for c,start in enumerate(cfg['inversion']['starts_vs']):
        folder=out/f'chain-{c}';folder.mkdir();one=copy.deepcopy(cfg)
        one['inversion']['starts_vs']=[start];one['nuts']['chain_id_offset']=c;write_config(folder/'nuts.toml',one);jobs.append((c,folder))
    save_json(out/'protocol.json',dict(interpretation='Prior sensitivity, not sampler speed comparison',reference=str(ref),
        only_change='prior_sigma_vs: 0.1 -> 0.2 km/s',bounds='unchanged: reference +/-0.6 km/s intersect [0.5,6]',
        initial_metric='recomputed at each original start with new prior curvature',chains=4,warmup=1000,draws=1000,
        seed=cfg['nuts']['seed'],workers=4,timeout_per_chain=1800,no_retries_or_extensions=True,
        incomplete_policy='No pooled posterior comparison if any chain incomplete; do not discard failed chains',
        acceptance=dict(max_rhat=1.01,min_bulk_ess=400,min_tail_ess=400,formal_divergences=0,formal_invalid=0,formal_depth_caps=0),
        posterior_checks=dict(fd_state=500,scales=2,predictive_states_per_chain=100),
        comparison='Layer and thickness-weighted band medians, 95% widths, actual bounded prior SD, predictive fit and boundary proximity',
        boundary_neighborhood_km_s=.05,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        inputs={f.name:digest(f) for f in inputs.iterdir()},configs={str(f.relative_to(out)):digest(f) for f in out.glob('chain-*/nuts.toml')},
        helpers={n:digest(Path(__file__).parent/n) for n in ['benchmark_crust_nuts.py','posterior_diagnostics.py','joint_results.py','benchmark_nuts_warmup.py']}))
    save_json(out/'provenance.json',dict(binary_sha256=digest(exe),files={n:digest(out/n) for n in ['nuts.toml','observations.txt','covariance.txt']}))
    def run(job):
        c,folder=job;start=time.perf_counter();dest=folder/'nuts'
        with (folder/'nuts.log').open('w') as log:
            try:code=subprocess.run([str(exe),'nuts','--config',str(folder/'nuts.toml'),'--data',str(out/'observations.txt'),'--covariance',str(out/'covariance.txt'),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code='timeout'
        elapsed=time.perf_counter()-start;r,partial=progress_rows(dest/'chain_0_progress.txt',45)
        rec=dict(chain=c,returncode=code,seconds=elapsed,complete=code in [0,5] and len(r)==2000,partial_lines=partial,
            warmup=phase_stats(r[:1000]),production=phase_stats(r[1000:]))
        if rec['complete']:
            cost=dict((k,int(v)) for k,v in (line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines()))
            assert cost['target_evaluations']==1+cost['transition_evaluations']+cost['step_search_evaluations']
            assert cost['transition_evaluations']==sum(int(row[5]) for row in r) and cost['metric_evaluations']==1
            rec['cost']=cost;rec['total_evaluations']=cost['target_evaluations']+cost['preflight_evaluations']+cost['metric_evaluations']
        save_json(folder/'sampler-command.json',rec);print(c,rec['complete'],round(elapsed,1),len(r),flush=True);return rec
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,jobs))
    save_json(out/'sampling-summary.json',dict(chains=records,all_complete=all(r['complete'] for r in records)))
    if not all(r['complete'] for r in records):
        save_json(out/'acceptance.json',dict(accepted=False,reason='Incomplete declared chain; no pooled comparison'));return
    merged=out/'nuts';merged.mkdir();shutil.copyfile(out/'nuts.toml',merged/'config.toml')
    for name in ['observations.txt','covariance.txt','mass.txt','prior_parameters.txt','parameter_bounds.txt']:
        first=out/'chain-0/nuts'/name
        for c in range(1,4):assert first.read_bytes()==(out/f'chain-{c}/nuts'/name).read_bytes()
        shutil.copyfile(first,merged/name)
    for c in range(4):
        for suffix in ['vs','z','trace','mass','metric_updates','initial_mass','progress','cost','failed_evaluations','step_size_searches']:
            shutil.copyfile(out/f'chain-{c}/nuts/chain_0_{suffix}.txt',merged/f'chain_{c}_{suffix}.txt')
    print('Analysis exit status:',analyze(out,exe,100,make_plots=False),flush=True)


if __name__=='__main__':main()
