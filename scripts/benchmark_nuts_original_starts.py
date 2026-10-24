#!/usr/bin/env python3
"""Bounded four-original-start validation of slice NUTS at target acceptance .95.

Keep the accepted reference's 1000-step warmup schedule, initial metrics, seed,
and target. Compare with its first 1000 retained states; never pool partial runs.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmark_nuts_warmup import digest,progress_rows,phase_stats
from benchmark_nuts_step_size import statistics
from benchmark_crust_nuts import analyze
from joint_results import read_config,write_config,save_json


WINDOWS=[(0,250),(250,450),(450,950),(950,1000)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--with-step-search',action='store_true',help='Search at startup and after each mass update')
    p.add_argument('--timeout',type=float,default=600,help='Per-chain sampling time limit in seconds')
    a=p.parse_args()
    if not 0<a.timeout<=3600:p.error('timeout must be in (0, 3600] seconds')
    out=a.output.resolve();out.mkdir();ref=a.reference.resolve();inputs=out/'inputs';inputs.mkdir()
    base=read_config(ref/'nuts.toml')
    assert len(base['model']['vs'])==45 and len(base['inversion']['starts_vs'])==4
    assert base['nuts']['warmup']==1000 and base['nuts']['target_accept']==.99
    assert base['nuts']['initial_metric']=='start_gauss_newton' and base['nuts']['metric_adaptation']=='dense'
    assert base['nuts'].get('kernel','slice')=='slice' and not base['nuts'].get('find_initial_step_size',False)
    config=copy.deepcopy(base);config['nuts'].update(target_accept=.95,draws=1000)
    if a.with_step_search:config['nuts']['find_initial_step_size']=True
    write_config(out/'nuts.toml',config)
    for name in ['nuts.toml','observations.txt','covariance.txt']:
        shutil.copyfile(ref/name,inputs/name)
        if name!='nuts.toml':shutil.copyfile(ref/name,out/name)
    exe=out/'dispweave';shutil.copy2(a.exe.resolve(),exe)
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    jobs=[]
    for c,start in enumerate(config['inversion']['starts_vs']):
        path=out/f'chain-{c}';path.mkdir();cfg=copy.deepcopy(config)
        cfg['inversion']['starts_vs']=[start];cfg['nuts']['chain_id_offset']=c
        write_config(path/'nuts.toml',cfg);jobs.append((c,path))
        for suffix in ['trace','vs','progress','initial_mass']:
            shutil.copyfile(ref/f'nuts/chain_{c}_{suffix}.txt',inputs/f'chain_{c}_{suffix}.txt')
    protocol=dict(interpretation='Original-start validation at target .95, optional reasonable-step search, formal budget 1000',
        kernel='slice',target_accept=.95,seed=config['nuts']['seed'],warmup=1000,draws=1000,chains=4,
        workers=4,timeout_per_chain=a.timeout,nominal_sampling_wall_budget_seconds=a.timeout,no_retries=True,
        find_initial_step_size=a.with_step_search,
        initial_metric='start_gauss_newton',metric_adaptation='dense',new_experimental_options='conservative policies, covariance persistence and stiffness guard disabled',
        reference_comparison='Stored target .99, all 1000 warmup + FIRST 1000 formal states per chain; no matched old wall timing available',
        acceptance=dict(max_rhat=1.01,min_bulk_ess=400,min_tail_ess=400,formal_divergences=0,formal_invalid=0,formal_depth_caps=0),
        posterior_checks=dict(derivatives='Middle retained state of each chain, two finite-difference scales and refined forward checks',predictive_states_per_chain=100),
        incomplete_policy='Keep all declared chains; no pooled ESS or acceptance if any run incomplete',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        helpers={name:digest(Path(__file__).parent/name) for name in ['benchmark_crust_nuts.py','benchmark_nuts_step_size.py','benchmark_nuts_warmup.py','posterior_diagnostics.py','joint_results.py']},
        inputs={f.name:digest(f) for f in inputs.iterdir()},
        configs={str(path.relative_to(out)/'nuts.toml'):digest(path/'nuts.toml') for _,path in jobs})
    save_json(out/'protocol.json',protocol)
    save_json(out/'provenance.json',dict(binary_sha256=digest(exe),files={name:digest(out/name) for name in ['nuts.toml','observations.txt','covariance.txt']}))
    reference_samples=np.stack([np.loadtxt(inputs/f'chain_{c}_vs.txt')[:1000] for c in range(4)])
    reference_stats=statistics(reference_samples,base)
    old_costs=[]
    for c in range(4):
        r,_=progress_rows(inputs/f'chain_{c}_progress.txt',45)
        t=np.loadtxt(inputs/f'chain_{c}_trace.txt',dtype=str)
        assert t.shape==(3000,10)
        old_costs.append(dict(chain=c,warmup_evaluations=int(t[:1000,7].astype(int).sum()),
            formal_evaluations=int(t[1000:2000,7].astype(int).sum()),
            total_evaluations=int(t[:2000,7].astype(int).sum())+3,
            segments=[dict(begin=b,end=e,**phase_stats(r[b:e])) for b,e in WINDOWS]))
    save_json(out/'reference-matched.json',dict(parameters=reference_stats,costs=old_costs,
        initial_preflight_and_gn_calls_per_chain=3,interpretation='Historical first 1000 formal states; no historical wall-time ratio'))

    def run(job):
        c,folder=job;begin=time.perf_counter()
        cmd=[str(exe),'nuts','--config',str(folder/'nuts.toml'),'--data',str(out/'observations.txt'),
             '--covariance',str(out/'covariance.txt'),'--output',str(folder/'nuts')]
        with (folder/'nuts.log').open('w') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:code='timeout'
        elapsed=time.perf_counter()-begin;dest=folder/'nuts'
        r,partial=progress_rows(dest/'chain_0_progress.txt',45)
        warm=[x for x in r if x[1]=='1'];formal=[x for x in r if x[1]=='0']
        record=dict(chain=c,returncode=code,seconds=elapsed,complete=code in (0,5) and len(r)==2000,
            warmup=phase_stats(warm),production=phase_stats(formal),partial_progress_lines=partial,
            segments=[dict(begin=b,end=e,**phase_stats(warm[b:e])) for b,e in WINDOWS])
        if formal:
            assert len(set(x[6] for x in formal))==1;record['frozen_step']=float(formal[0][6])
        if record['complete']:
            cost={k:int(v) for k,v in (line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines())}
            assert (cost['step_search_evaluations']>0)==a.with_step_search and cost['metric_evaluations']==1
            assert cost['target_evaluations']==1+cost['transition_evaluations']+cost['step_search_evaluations']
            assert cost['transition_evaluations']==sum(int(x[5]) for x in r)
            record['cost']=cost
            record['total_evaluations']=cost['target_evaluations']+cost['preflight_evaluations']+cost['metric_evaluations']
            t=np.loadtxt(dest/'chain_0_trace.txt',dtype=str)
            for name,part in [('warmup',t[:1000]),('production',t[1000:])]:
                record[name]['depth_caps']=int(part[:,9].astype(int).sum())
            np.testing.assert_allclose(np.loadtxt(dest/'chain_0_initial_mass.txt'),np.loadtxt(inputs/f'chain_{c}_initial_mass.txt'),rtol=1e-12,atol=1e-12)
        save_json(folder/'sampler-command.json',record);print(record,flush=True);return record

    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,jobs))
    save_json(out/'sampling-summary.json',dict(chains=records,all_complete=all(r['complete'] for r in records)))
    if not all(r['complete'] for r in records):
        save_json(out/'acceptance.json',dict(accepted=False,reason='One or more declared chains incomplete; no pooled ESS'))
        return
    merged=out/'nuts';merged.mkdir();shutil.copyfile(out/'nuts.toml',merged/'config.toml')
    for name in ['observations.txt','covariance.txt','mass.txt','prior_parameters.txt','parameter_bounds.txt']:
        first=out/'chain-0/nuts'/name
        for c in range(1,4):assert (out/f'chain-{c}/nuts'/name).read_bytes()==first.read_bytes()
        shutil.copyfile(first,merged/name)
    for c in range(4):
        for suffix in ['vs','z','trace','mass','metric_updates','initial_mass','progress','cost','failed_evaluations']+(['step_size_searches'] if a.with_step_search else []):
            shutil.copyfile(out/f'chain-{c}/nuts/chain_0_{suffix}.txt',merged/f'chain_{c}_{suffix}.txt')
    x=np.stack([np.loadtxt(merged/f'chain_{c}_vs.txt') for c in range(4)])
    new_stats=statistics(x,config)
    comparison={}
    for label,stats,costs in [('reference_099',reference_stats,old_costs),('candidate_095',new_stats,records)]:
        formal=sum(r['formal_evaluations'] if label=='reference_099' else r['production']['completed_transition_evaluations'] for r in costs)
        total=sum(r['total_evaluations'] for r in costs)
        comparison[label]=dict(parameters=stats,formal_evaluations=formal,total_evaluations=total,
            min_bulk_ess=min(d['ess_bulk'] for d in stats),min_tail_ess=min(d['ess_tail'] for d in stats),max_rhat=max(d['rhat'] for d in stats),
            min_bulk_ess_per_1000_formal_evaluations=min(d['ess_bulk'] for d in stats)/formal*1000,
            min_bulk_ess_per_1000_total_evaluations=min(d['ess_bulk'] for d in stats)/total*1000)
    save_json(out/'efficiency-comparison.json',comparison)
    code=analyze(out,exe,100,make_plots=False)
    print('Full analysis exit status',code,flush=True)


if __name__=='__main__':main()
