#!/usr/bin/env python3
"""Fixed-metric acceptance-target comparison from saved posterior states.

Four declared chains per kernel/target; never pool incomplete groups or silently
remove failed trajectories. The default pilot uses 200 draws; confirmation uses
1000 draws and a new seed, still conditional on frozen posterior geometry.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from benchmark_nuts_warmup import digest, progress_rows, phase_stats
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose


def observables(x, cfg):
    values=[x[:,:,j] for j in range(x.shape[2])]
    names=[f'Vs[{j}]' for j in range(x.shape[2])]
    depth=np.array(cfg['model']['depth'])
    for low,high in [(0.,20.),(20.,100.),(100.,float(depth[-1]))]:
        weights=np.maximum(0,np.minimum(np.r_[depth[1:],high],high)-np.maximum(depth,low))/(high-low)
        values.append(x@weights);names.append(f'mean_vs_{low:g}_{high:g}_km')
    delta=(x-np.array(cfg['model']['vs']))/np.array(cfg['nuts']['prior_sigma_vs'])
    values.append(.5*np.sum(delta**2,axis=2));names.append('reference_quadratic')
    return names,values


def statistics(x,cfg):
    names,values=observables(x,cfg)
    return [dict(name=name,**diagnose(value)) for name,value in zip(names,values)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--confirm-candidates',action='store_true',
        help='Confirm multinomial 0.90 and slice 0.95 with 1000 draws/chain and seed 20261025')
    a=p.parse_args();out=a.output.resolve();out.mkdir();ref=a.reference.resolve()
    draws=1000 if a.confirm_candidates else 200
    timeout=300 if a.confirm_candidates else 150
    seed=20261025 if a.confirm_candidates else 20261024
    inputs=out/'inputs';inputs.mkdir()
    base=read_config(ref/'nuts.toml')
    assert len(base['model']['vs'])==45 and len(base['inversion']['starts_vs'])==4
    assert base['nuts']['prior']=='reference_gaussian_vs' and base['nuts']['mode_support']=='trapped_modes'
    assert np.loadtxt(ref/'observations.txt').shape==(125,4)
    for name in ['nuts.toml','observations.txt','covariance.txt']:
        shutil.copyfile(ref/name,inputs/name)
    mass=np.loadtxt(ref/'nuts/chain_0_mass.txt');np.linalg.cholesky(mass)
    step=float(np.loadtxt(ref/'nuts/chain_0_trace.txt',dtype=str)[-1,3])
    shutil.copyfile(ref/'nuts/chain_0_mass.txt',inputs/'mass.txt')
    reference=np.stack([np.loadtxt(ref/f'nuts/chain_{c}_vs.txt') for c in range(4)])
    starts=reference[:,-1,:]
    np.savetxt(inputs/'starts_vs.txt',starts,fmt='%.17g')
    np.save(inputs/'reference_vs.npy',reference)
    reference_stats=statistics(reference,base)
    save_json(inputs/'reference_statistics.json',reference_stats)
    exe=out/'dispweave';shutil.copy2(a.exe.resolve(),exe)
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    groups=[(kernel,target) for target in [.90,.95,.99] for kernel in ['slice','multinomial']]
    if a.confirm_candidates:groups=[('multinomial',.90),('slice',.95)]
    jobs=[]
    # Rotate the settings over the four starting states to distribute order effects.
    for c in range(4):
        offset=c%len(groups);order=groups[offset:]+groups[:offset]
        if c%2:order=list(reversed(order))
        for kernel,target in order:
            label=f'{kernel}-{round(100*target)}';cfg=copy.deepcopy(base)
            cfg['inversion']['starts_vs']=[starts[c].tolist()]
            opt=cfg['nuts'];opt.pop('initial_metric',None)
            opt.update(kernel=kernel,target_accept=target,step_size=step,mass=mass.tolist(),
                metric_adaptation='none',metric_shrinkage=0.,metric_max_stiffness_ratio=0.,
                conservative_warmup=False,conservative_metric_restart=False,find_initial_step_size=False,
                warmup=100,draws=draws,max_depth=10,adapt_step_size=True,seed=seed,
                chain_id_offset=c,record_progress=True,record_failures=True)
            path=inputs/f'{label}-{c}.toml';write_config(path,cfg)
            jobs.append((label,kernel,target,c,path))
    protocol=dict(interpretation='Fixed mass, posterior starts, step-size adaptation only; exploratory ESS, not new posterior acceptance',
        groups=[dict(kernel=k,target_accept=t) for k,t in groups],chains=4,warmup=100,draws=draws,
        confirm_candidates=a.confirm_candidates,seed=seed,workers=4,timeout_per_chain=timeout,
        total_wall_budget_seconds=len(groups)*timeout,
        conditional_diagnostic_thresholds=dict(max_rhat=1.01,min_bulk_ess=400,min_tail_ess=400,formal_divergences=0,formal_invalid=0,formal_depth_caps=0),
        initial_step=step,common_mass='Accepted reference chain 0 final momentum covariance',
        common_starts='Last physical Vs state of each of the four accepted reference chains',
        adaptation='Ordinary dual averaging; no initial search; frozen metric; all experimental options disabled',
        ranking='Report failures first, then minimum bulk/tail ESS over all 45 Vs + 4 derived quantities; no automatic default change',
        uncertainty=f'{draws} draws/chain: report first/second-half diagnostics and per-observable efficiency; no significance claim',
        timing='ESS/s uses summed complete chain wall seconds INCLUDING warmup; no formal-only wall timer',
        cost='Both formal-only and including-warmup ESS/evaluation denominators; incomplete totals are lower bounds',
        reference_mean_check='Combined MCSE only descriptive: starting states reused from reference, 49 correlated comparisons',
        incomplete_policy='No retries, extension, truncation to fastest chains, or pooled ESS from incomplete groups',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        helper_sha256={name:digest(Path(__file__).parent/name) for name in ['posterior_diagnostics.py','joint_results.py','benchmark_nuts_warmup.py']},
        inputs={f.name:digest(f) for f in inputs.iterdir()},job_order=[f'{label}-{c}' for label,_,_,c,_ in jobs])
    save_json(out/'protocol.json',protocol)

    def run(job):
        label,kernel,target,c,path=job;dest=out/f'{label}-{c}';begin=time.perf_counter()
        cmd=[str(exe),'nuts','--config',str(path),'--data',str(inputs/'observations.txt'),
            '--covariance',str(inputs/'covariance.txt'),'--output',str(dest)]
        with (out/f'{label}-{c}.log').open('w') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired:code='timeout'
        elapsed=time.perf_counter()-begin
        rows,partial=progress_rows(dest/'chain_0_progress.txt',45)
        warm=[r for r in rows if r[1]=='1'];formal=[r for r in rows if r[1]=='0']
        assert len(warm)==min(100,len(rows))
        record=dict(label=label,kernel=kernel,target_accept=target,chain=c,returncode=code,seconds=elapsed,
            complete=code in (0,5) and len(warm)==100 and len(formal)==draws,
            warmup=phase_stats(warm),production=phase_stats(formal),partial_progress_lines=partial)
        if formal:
            steps=[float(r[6]) for r in formal];assert len(set(steps))==1
            record['frozen_step']=steps[0]
        if record['complete']:
            trace=np.loadtxt(dest/'chain_0_trace.txt',dtype=str,ndmin=2)
            samples=np.loadtxt(dest/'chain_0_vs.txt',ndmin=2)
            assert trace.shape==(100+draws,10) and samples.shape==(draws,45) and np.isfinite(samples).all()
            assert np.array_equal(np.loadtxt(dest/'chain_0_mass.txt'),np.loadtxt(dest/'chain_0_initial_mass.txt'))
            np.testing.assert_allclose(np.loadtxt(dest/'chain_0_mass.txt'),mass,rtol=1e-14,atol=1e-14)
            np.testing.assert_array_equal(np.loadtxt(dest/'prior_parameters.txt')[:,1],base['model']['vs'])
            costs={k:int(v) for k,v in (line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines())}
            assert costs['target_evaluations']==1+costs['transition_evaluations']
            assert costs['step_search_evaluations']==costs['metric_evaluations']==costs.get('guard_evaluations',0)==0
            assert costs['transition_evaluations']==sum(int(r[5]) for r in rows)
            record['total_evaluations']=costs['target_evaluations']+costs['preflight_evaluations']
            for phase,t in [('warmup',trace[:100]),('production',trace[100:])]:
                record[phase]['depth_caps']=int(t[:,9].astype(int).sum())
                record[phase]['candidate_replacements']=int(t[:,1].astype(int).sum())
                record[phase]['max_energy_error']=float(t[:,5].astype(float).max())
        save_json(out/f'{label}-{c}-cost.json',record)
        print(dict(label=label,chain=c,complete=record['complete'],seconds=round(elapsed,2),
            warmup=len(warm),formal=len(formal),step=record.get('frozen_step'),production=record['production']),flush=True)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(run,jobs))
    comparison={}
    for kernel,target in groups:
        label=f'{kernel}-{round(100*target)}';r=[r for r in records if r['label']==label]
        assert len(r)==4
        if not all(row['complete'] for row in r):
            comparison[label]=dict(complete=False,reason='All four declared chains required; no pooled ESS')
            continue
        x=np.stack([np.loadtxt(out/f'{label}-{c}/chain_0_vs.txt') for c in range(4)])
        stats=statistics(x,base)
        seconds=sum(row['seconds'] for row in r);total=sum(row['total_evaluations'] for row in r)
        formal=sum(row['production']['completed_transition_evaluations'] for row in r)
        for d,refstat in zip(stats,reference_stats):
            d.update(bulk_ess_per_1000_formal_evaluations=d['ess_bulk']/formal*1000,
                bulk_ess_per_1000_total_evaluations=d['ess_bulk']/total*1000,
                bulk_ess_per_total_chain_second=d['ess_bulk']/seconds,
                mean_difference_in_combined_mcse=abs(d['mean']-refstat['mean'])/np.hypot(d['mcse_mean'],refstat['mcse_mean']))
        invalid=sum(row['production']['invalid_transitions'] for row in r)
        div=sum(row['production']['divergences'] for row in r)
        caps=sum(row['production']['depth_caps'] for row in r)
        numeric=invalid==div==caps==0 and all(row['production']['candidate_replacements']>0 for row in r)
        mixing=all(np.isfinite([d['rhat'],d['ess_bulk'],d['ess_tail']]).all() and d['rhat']<=1.01
            and min(d['ess_bulk'],d['ess_tail'])>=400 for d in stats)
        comparison[label]=dict(complete=True,parameters=stats,
            conditional_diagnostics_passed=numeric and mixing,mixing_passed=mixing,
            numerical_passed=numeric,
            formal_divergences=div,formal_invalid=invalid,formal_depth_caps=caps,
            formal_support_rejections=sum(row['production']['support_rejections'] for row in r),
            max_rhat=max(d['rhat'] for d in stats),min_bulk_ess=min(d['ess_bulk'] for d in stats),
            min_tail_ess=min(d['ess_tail'] for d in stats),total_chain_seconds=seconds,
            total_evaluations=total,formal_evaluations=formal,
            min_bulk_ess_per_1000_formal_evaluations=min(d['ess_bulk'] for d in stats)/formal*1000,
            min_bulk_ess_per_1000_total_evaluations=min(d['ess_bulk'] for d in stats)/total*1000,
            min_bulk_ess_per_total_chain_second=min(d['ess_bulk'] for d in stats)/seconds,
            max_mean_difference_in_combined_mcse=max(d['mean_difference_in_combined_mcse'] for d in stats),
            halves=[dict(parameters=statistics(x[:,begin:end],base),
                formal_evaluations=sum(int(np.loadtxt(out/f'{label}-{c}/chain_0_trace.txt',dtype=str)[100+begin:100+end,7].astype(int).sum()) for c in range(4)))
                for begin,end in [(0,draws//2),(draws//2,draws)]])
    save_json(out/'summary.json',dict(protocol=protocol,chains=records,comparison=comparison,new_posterior_accepted=False))
    print({label:{k:v for k,v in d.items() if k not in ['parameters','halves']} for label,d in comparison.items()},flush=True)


if __name__=='__main__':main()
