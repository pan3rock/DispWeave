#!/usr/bin/env python3
"""Conditional, bounded terminal step adaptation comparison; no posterior acceptance."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmark_nuts_warmup import digest, progress_rows, phase_stats
from benchmark_nuts_step_size import statistics
from joint_results import read_config, write_config, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-metrics',action='store_true',help='Compare pre/post update masses with 100 warmup + 300 draws, 300s per run')
    a=p.parse_args();out=a.output.resolve();out.mkdir();src=a.source.resolve()
    inputs=out/'inputs';inputs.mkdir()
    for name in ['nuts.toml','observations.txt','covariance.txt']:
        shutil.copyfile(src/name,inputs/name)
    shutil.copyfile(src/'chain-0/nuts/chain_0_progress.txt',inputs/'progress.txt')
    base=read_config(inputs/'nuts.toml');rows,partial=progress_rows(inputs/'progress.txt',45)
    assert not partial and len(rows)>=1001 and base['nuts']['target_accept']==.95
    # Progress is emitted after the transition and before covariance adaptation.
    x=np.array([r[8:] for r in rows[450:950]],float)
    cov=500/505*np.cov(x.T)+.005/505*np.eye(45)
    mass=np.linalg.solve(cov,np.eye(45));mass=(mass+mass.T)/2
    np.testing.assert_allclose(mass,np.loadtxt(src/'last-metric-audit/mass-after.txt'),rtol=1e-10,atol=1e-10)
    np.savetxt(inputs/'mass.txt',mass,fmt='%.17g')
    v=np.array(base['model']['vs']);inv=base['inversion']
    lo=np.log(np.maximum(inv['vs_min'],v-inv['vs_width']/2));hi=np.log(np.minimum(inv['vs_max'],v+inv['vs_width']/2))
    z=np.array(rows[949][8:],float);start=np.exp(lo+(hi-lo)/(1+np.exp(-z)))
    np.savetxt(inputs/'start_vs.txt',start,fmt='%.17g')
    step=float(rows[950][6])  # step entering the first transition under the new mass
    exe=out/'dispweave';shutil.copy2(a.exe.resolve(),exe)
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    variants=[('ordinary-50',False,50),('search-50',True,50),('search-150',True,150)]
    masses={label:mass for label,_,_ in variants}
    draws,timeout,seed=200,120,20261026
    if a.compare_metrics:
        variants=[('before',True,100),('after',True,100)]
        x=np.array([r[8:] for r in rows[250:450]],float)
        cov=200/205*np.cov(x.T)+.005/205*np.eye(45)
        before=np.linalg.solve(cov,np.eye(45));before=(before+before.T)/2
        np.testing.assert_allclose(before,np.loadtxt(src/'last-metric-audit/mass-before.txt'),rtol=1e-10,atol=1e-10)
        masses={'before':before,'after':mass}
        np.savetxt(inputs/'mass-before.txt',before,fmt='%.17g')
        draws,timeout,seed=300,300,20261027
    jobs=[]
    for c in range(4):
        order=variants[c%len(variants):]+variants[:c%len(variants)]
        for label,search,warmup in order:
            cfg=copy.deepcopy(base);cfg['inversion']['starts_vs']=[start.tolist()]
            cfg['nuts'].update(initial_metric='supplied',mass=masses[label].tolist(),kernel='slice',
                metric_adaptation='none',metric_shrinkage=0.,metric_max_stiffness_ratio=0.,
                conservative_warmup=False,conservative_metric_restart=False,
                find_initial_step_size=search,step_size=step,warmup=warmup,draws=draws,
                seed=seed,chain_id_offset=c,adapt_step_size=True)
            path=inputs/f'{label}-{c}.toml';write_config(path,cfg)
            jobs.append((label,c,warmup,path))
    protocol=dict(checkpoint='chain 0 state after transition 949; reconstructed post-950 mass',
        variants=variants,compare_metrics=a.compare_metrics,repetitions=4,common_start=True,seed=seed,draws=draws,
        workers=4,timeout_per_job_seconds=timeout,nominal_worker_sampling_budget_seconds=len(variants)*timeout,
        initial_step=step,ranking='minimum bulk ESS / all target evaluations including search, warmup, preflight',
        interpretation='Short conditional restart experiment; identical starting point, not dispersed-start convergence validation; ESS provisional',
        incomplete_policy='No retries/extensions or pooled ESS for incomplete group',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        inputs={f.name:digest(f) for f in inputs.iterdir()},
        helpers={n:digest(Path(__file__).parent/n) for n in ['benchmark_nuts_warmup.py','benchmark_nuts_step_size.py','posterior_diagnostics.py','joint_results.py']})
    save_json(out/'protocol.json',protocol)

    def run(job):
        label,c,warmup,path=job;dest=out/f'{label}-{c}';begin=time.perf_counter()
        with (out/f'{label}-{c}.log').open('w') as log:
            try:
                code=subprocess.run([str(exe),'nuts','--config',str(path),'--data',str(inputs/'observations.txt'),
                    '--covariance',str(inputs/'covariance.txt'),'--output',str(dest)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired:code='timeout'
        elapsed=time.perf_counter()-begin;r,partial=progress_rows(dest/'chain_0_progress.txt',45)
        record=dict(label=label,replicate=c,seconds=elapsed,returncode=code,complete=code in (0,5) and len(r)==warmup+draws,
            partial_lines=partial,warmup=phase_stats(r[:warmup]),production=phase_stats(r[warmup:]))
        if len(r)>warmup:
            assert len(set(row[6] for row in r[warmup:]))==1
            record['frozen_step']=float(r[warmup][6])
        if record['complete']:
            cost=dict((k,int(v)) for k,v in (line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines()))
            assert cost['transition_evaluations']==sum(int(row[5]) for row in r)
            assert cost['target_evaluations']==1+cost['transition_evaluations']+cost['step_search_evaluations']
            assert cost['metric_evaluations']==0
            record['cost']=cost;record['total_evaluations']=cost['target_evaluations']+cost['preflight_evaluations']
            np.testing.assert_allclose(np.loadtxt(dest/'chain_0_mass.txt'),masses[label],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(np.loadtxt(dest/'chain_0_mass.txt'),np.loadtxt(dest/'chain_0_initial_mass.txt'))
            np.testing.assert_array_equal(np.loadtxt(dest/'prior_parameters.txt')[:,1],base['model']['vs'])
            t=np.loadtxt(dest/'chain_0_trace.txt',dtype=str)
            for key,part in [('warmup',t[:warmup]),('production',t[warmup:])]:
                record[key].update(depth_caps=int(part[:,9].astype(int).sum()),max_energy_error=float(part[:,5].astype(float).max()))
        save_json(out/f'{label}-{c}-cost.json',record)
        print(label,c,record['complete'],round(elapsed,1),record.get('frozen_step'),flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,jobs))
    groups={}
    for label,_,_ in variants:
        rr=[r for r in records if r['label']==label]
        if not all(r['complete'] for r in rr):
            groups[label]=dict(complete=False,reason='No pooled ESS: incomplete declared replicate');continue
        samples=np.stack([np.loadtxt(out/f'{label}-{c}/chain_0_vs.txt') for c in range(4)])
        total=sum(r['total_evaluations'] for r in rr);formal=sum(r['production']['completed_transition_evaluations'] for r in rr)
        stats=statistics(samples,base)
        groups[label]=dict(complete=True,total_evaluations=total,formal_evaluations=formal,
            min_bulk_ess=min(s['ess_bulk'] for s in stats),min_tail_ess=min(s['ess_tail'] for s in stats),
            max_rhat=max(s['rhat'] for s in stats),
            min_bulk_per_1000_total=min(s['ess_bulk'] for s in stats)/total*1000,
            min_tail_per_1000_total=min(s['ess_tail'] for s in stats)/total*1000,
            parameters=stats,halves=[statistics(part,base) for part in [samples[:,:draws//2],samples[:,draws//2:]]])
    save_json(out/'summary.json',dict(records=records,groups=groups,posterior_accepted=False))


if __name__=='__main__':main()
