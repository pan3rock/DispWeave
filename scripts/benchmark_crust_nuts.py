#!/usr/bin/env python3
"""Fixed-reference crustal NUTS with an explicit bounded physical-Vs Gaussian prior.

The input case is a prepared benchmark_crust_reference.py directory. Initial
metric uses local Gauss-Newton curvature only for preconditioning. The target,
reference, observations and bounds are never altered in response to diagnostics.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import shutil
from statistics import NormalDist
import subprocess
import sys

import numpy as np

from benchmark_crust_depth import brocher, digest, forward, audit_terminal
from benchmark_crust_regularization import fd_passed
from check_qedisp_examples import check_queries, run, forward_rows
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose

PRIOR_SD = .1
SEED = 20261023


def prepare(case, exe, output, warmup, draws, chains, adaptation, target_accept=.95,
            mode_support='strict_forward', initial_metric='reference'):
    if warmup < 100 or draws < 8 or chains not in (1, 4):
        raise ValueError('Need warmup >=100, draws >=8 and one pilot or four production chains')
    if not np.isfinite(target_accept) or not 0<target_accept<1:raise ValueError('Target acceptance must be in (0,1)')
    config = read_config(case/'inversion.toml');inv = config['inversion'];model = config['model']
    if (inv.get('vs2model') != 'brocher05' or inv.get('fixed_vs_layers')
            or inv.get('regularization') != dict(type='reference_damping', **{'lambda':100.})
            or len(inv['starts_vs']) != 5):
        raise ValueError('Need the free-Vs Brocher05 reference-damping lambda=100 case with five declared starts')
    data = np.loadtxt(case/'observations.txt', ndmin=2)
    covariance = np.loadtxt(case/'lm/covariance.txt', ndmin=2)
    if data.shape[1] != 4 or covariance.shape != (len(data), len(data)):
        raise ValueError('Malformed source data/covariance')
    output.mkdir();(output/'metric').mkdir();(output/'scripts').mkdir()
    for path in Path(__file__).resolve().parent.glob('*.py'):shutil.copyfile(path, output/'scripts'/path.name)
    shutil.copyfile(case/'inversion.toml', output/'source-inversion.toml')
    shutil.copyfile(case/'observations.txt', output/'observations.txt')
    shutil.copyfile(case/'lm/covariance.txt', output/'covariance.txt')
    vs = np.array(model['vs']);n = len(vs);vp,rho,dvp,drho = brocher(vs)
    # Refined local derivatives set a constant initial momentum covariance;
    # ordinary sampler evaluations continue to use their existing standard roots.
    probe = dict(model=dict(depth=model['depth'], vs=vs.tolist(), vp=vp.tolist(), rho=rho.tolist()),
                 queries=dict(frequency_hz=data[:,0].tolist(), mode=data[:,2].astype(int).tolist()))
    write_config(output/'metric/forward.toml', probe)
    rows = forward(exe, output/'metric/forward.toml', output/'metric/forward.txt', 'implicit')
    check_queries(rows, data[:,0], data[:,2])
    if any(r[4]!='valid' or r[7]!='valid' for r in rows):raise ValueError('Reference metric has unresolved roots/gradients')
    j = np.array([r[12:12+n] for r in rows], float)
    j += np.array([r[12+n:12+2*n] for r in rows], float)*dvp
    j += np.array([r[12+2*n:12+3*n] for r in rows], float)*drho
    if 'vs_width' in inv:
        lower=np.maximum(inv['vs_min'],vs-inv['vs_width']/2);upper=np.minimum(inv['vs_max'],vs+inv['vs_width']/2)
    else:lower=np.array(inv['lower_vs']);upper=np.array(inv['upper_vs'])
    q=np.log(vs);lo=np.log(lower);hi=np.log(upper);s=(q-lo)/(hi-lo)
    dv_dz=vs*(hi-lo)*s*(1-s)
    white=np.linalg.solve(np.linalg.cholesky(covariance),j*dv_dz)
    mass=white.T@white+np.diag((dv_dz/PRIOR_SD)**2)
    np.linalg.cholesky(mass)
    np.savetxt(output/'metric/initial_mass.txt',mass,fmt='%.17g')
    np.savetxt(output/'metric/physical_jacobian.txt',j,fmt='%.17g')
    inv['regularization']=dict(type='none',**{'lambda':0.})
    inv['starts_vs']=inv['starts_vs'][1:1+chains]
    config['nuts']=dict(prior='reference_gaussian_vs',prior_sigma_vs=[PRIOR_SD]*n,
        warmup=warmup,draws=draws,max_depth=10,step_size=.1,target_accept=target_accept,
        adapt_step_size=True,metric_adaptation=adaptation,seed=SEED,mass=mass.tolist(),record_failures=True)
    config['nuts'].update(mode_support=mode_support,record_progress=True)
    if initial_metric=='start':
        del config['nuts']['mass']
        config['nuts']['initial_metric']='start_gauss_newton'
    write_config(output/'nuts.toml',config)
    provenance=dict(source=str(case),files={name:digest(output/name) for name in
        ['source-inversion.toml','observations.txt','covariance.txt','nuts.toml']},
        binary=str(exe),binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        prior_mean='model.vs',prior_sd_km_s=PRIOR_SD,
        initial_metric=f'D * (J_vs.T C^-1 J_vs + prior_precision) * D at {initial_metric}; D=dVs/dz',
        mode_support=mode_support,
        initial_metric_eigenvalues=np.linalg.eigvalsh(mass).tolist(),
        warmup=warmup,draws=draws,chains=chains,metric_adaptation=adaptation,target_accept=target_accept,
        acceptance=dict(max_rhat=1.01,min_bulk_ess=400,min_tail_ess=400,
                        production_divergences=0,production_invalid=0,production_depth_limits=0),
        interpretation='Conditional on fixed interfaces, reference Gaussian prior, Vs bounds, Brocher05 and fixed data covariance. No automatic prior tuning.')
    save_json(output/'provenance.json',provenance)
    return config


def prior_statistics(config):
    model=config['model'];inv=config['inversion'];mean=np.array(model['vs'])
    sd=np.array(config['nuts']['prior_sigma_vs'])
    if 'vs_width' in inv:
        lower=np.maximum(inv['vs_min'],mean-inv['vs_width']/2);upper=np.minimum(inv['vs_max'],mean+inv['vs_width']/2)
    else:lower=np.array(inv['lower_vs']);upper=np.array(inv['upper_vs'])
    normal=NormalDist();result=[]
    for j in range(len(mean)):
        a=(lower[j]-mean[j])/sd[j];b=(upper[j]-mean[j])/sd[j]
        mass=.5*(math.erf(b/math.sqrt(2))-math.erf(a/math.sqrt(2)))
        pa=math.exp(-.5*a*a)/math.sqrt(2*math.pi);pb=math.exp(-.5*b*b)/math.sqrt(2*math.pi)
        shift=(pa-pb)/mass
        variance=sd[j]**2*(1+(a*pa-b*pb)/mass-shift**2)
        quantiles=[mean[j]+sd[j]*normal.inv_cdf(normal.cdf(a)+p*mass) for p in [.025,.5,.975]]
        result.append(dict(layer=j,mean=float(mean[j]+sd[j]*shift),sd=float(math.sqrt(variance)),
            quantiles=quantiles,untruncated_mean=float(mean[j]),untruncated_sd=float(sd[j]),
            lower=float(lower[j]),upper=float(upper[j]),normalizing_mass=mass))
    return result


def diagnostics(config, samples, traces):
    """All declared chains and all formal states are required for acceptance."""
    options=config['nuts'];count=len(config['inversion']['starts_vs']);n=len(config['model']['vs'])
    if samples.shape!=(count,options['draws'],n) or not np.isfinite(samples).all() or len(traces)!=count:
        raise ValueError('Missing, unequal or nonfinite posterior chains')
    priors=prior_statistics(config)
    if (np.any(samples<=np.array([p['lower'] for p in priors]))
            or np.any(samples>=np.array([p['upper'] for p in priors]))):
        raise ValueError('Posterior samples outside declared strict support')
    sampler=[]
    for c,trace in enumerate(traces):
        if trace.shape!=(options['warmup']+options['draws'],10):raise ValueError('Incomplete NUTS trace')
        if (not np.all(trace[:options['warmup'],0]=='1') or not np.all(trace[options['warmup']:,0]=='0')):
            raise ValueError('Warmup flags disagree with declared schedule')
        record=dict(chain=c)
        for label,part in [('warmup',trace[:options['warmup']]),('production',trace[options['warmup']:])]:
            record[label]=dict(divergences=int(part[:,2].astype(int).sum()),
                invalid_evaluations=int(np.count_nonzero(~np.isin(part[:,6],['valid','outside_support']))),
                support_rejections=int(np.count_nonzero(part[:,6]=='outside_support')),
                depth_limits=int(part[:,9].astype(int).sum()),
                accepted=int(part[:,1].astype(int).sum()),gradient_evaluations=int(part[:,7].astype(int).sum()),
                mean_acceptance_probability=float(part[:,4].astype(float).mean()),
                mean_tree_depth=float(part[:,8].astype(float).mean()),
                maximum_tree_depth=int(part[:,8].astype(int).max()),
                statuses={s:int(np.count_nonzero(part[:,6]==s)) for s in np.unique(part[:,6])})
        production=trace[options['warmup']:]
        step=production[:,3].astype(float)
        if not np.isfinite(step).all() or np.any(step<=0) or np.ptp(step)!=0:raise ValueError('Production step is not finite, positive and frozen')
        record['frozen_step_size']=float(step[0]);sampler.append(record)
    result=dict(chains=count,draws=options['draws'],sampler=sampler,priors=priors,
                parameters=[],derived=[],accepted=False)
    if count<2:
        result['reason']='Pilot only: at least two chains required for mixing diagnostics';return result
    for j in range(n):
        d=diagnose(samples[:,:,j]);d['layer']=j
        d['posterior_to_prior_sd']=d['sd']/priors[j]['sd']
        d['mean_shift_in_prior_sd']=(d['mean']-priors[j]['mean'])/priors[j]['sd']
        result['parameters'].append(d)
    z=np.array(config['model']['depth']);edges=sorted(set([0.,*[v for v in [20.,100.] if v<z[-1]],float(z[-1])]))
    for low,high in zip(edges[:-1],edges[1:]):
        weights=np.maximum(0,np.minimum(np.r_[z[1:],high],high)-np.maximum(z,low))/(high-low)
        d=diagnose(samples@weights);d.update(name=f'mean_vs_{low:g}_{high:g}_km',low_km=low,high_km=high)
        result['derived'].append(d)
    delta=(samples-np.array(config['model']['vs']))/np.array(options['prior_sigma_vs'])
    d=diagnose(.5*np.sum(delta**2,axis=2));d['name']='reference_quadratic';result['derived'].append(d)
    all_diagnostics=result['parameters']+result['derived']
    numeric=all(s['production']['divergences']==0 and s['production']['invalid_evaluations']==0
                and s['production']['depth_limits']==0 and s['production']['accepted']>0 for s in sampler)
    mixing=all(d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400 for d in all_diagnostics)
    result.update(accepted=count==4 and numeric and mixing,numerical_passed=numeric,mixing_passed=mixing,
                  maximum_rhat=max(d['rhat'] for d in all_diagnostics),
                  minimum_bulk_ess=min(d['ess_bulk'] for d in all_diagnostics),
                  minimum_tail_ess=min(d['ess_tail'] for d in all_diagnostics),
                  correlation=np.corrcoef(samples.reshape(-1,n),rowvar=False).tolist())
    return result


def verify_inputs(output,exe):
    provenance=json.loads((output/'provenance.json').read_text())
    if digest(exe)!=provenance['binary_sha256']:raise ValueError('Executable differs from prepared executable')
    for name,expected_hash in provenance['files'].items():
        if digest(output/name)!=expected_hash:raise ValueError(f'Input changed after preparation: {name}')


def analyze(output, exe, per_chain, make_plots=True):
    config=read_config(output/'nuts.toml');folder=output/'nuts';count=len(config['inversion']['starts_vs'])
    verify_inputs(output,exe)
    if read_config(folder/'config.toml')!=config:raise ValueError('Sampled config differs from declared config')
    for name in ['observations.txt','covariance.txt']:
        np.testing.assert_array_equal(np.loadtxt(folder/name,ndmin=2),np.loadtxt(output/name,ndmin=2))
    expected={f'chain_{c}_vs.txt' for c in range(count)}
    if {p.name for p in folder.glob('chain_*_vs.txt')}!=expected:raise ValueError('Missing or unexpected chain files')
    samples=np.stack([np.loadtxt(folder/f'chain_{c}_vs.txt',ndmin=2) for c in range(count)])
    traces=[np.loadtxt(folder/f'chain_{c}_trace.txt',dtype=str,ndmin=2,skiprows=1) for c in range(count)]
    result=diagnostics(config,samples,traces)
    for c in range(count):
        mass=np.loadtxt(folder/f'chain_{c}_mass.txt',ndmin=2);np.linalg.cholesky(mass)
        lines=(folder/f'chain_{c}_metric_updates.txt').read_text().splitlines()
        updates=[i for i,line in enumerate(lines) if line.startswith('# ') and line[2:].split()[0].isdigit()]
        if config['nuts']['metric_adaptation']=='none':
            if updates:raise ValueError('Unexpected metric updates')
            initial=folder/f'chain_{c}_initial_mass.txt'
            np.testing.assert_allclose(mass,np.loadtxt(initial if initial.exists() else folder/'mass.txt'),rtol=1e-12,atol=1e-12)
        else:
            if not updates:raise ValueError('Missing declared metric adaptation')
            for start in updates:
                if int(lines[start].split()[1])>=config['nuts']['warmup']:raise ValueError('Metric changed in production')
            last=np.array([line.split() for line in lines[updates[-1]+1:updates[-1]+1+mass.shape[0]]],float)
            np.testing.assert_allclose(last,mass,rtol=1e-12,atol=1e-12)
    save_json(output/'diagnostics.json',result)
    if count==1:
        print(json.dumps(result['sampler'],indent=2));return 4
    audits=audit_posterior(output,exe,config,samples)
    if per_chain<1 or per_chain>config['nuts']['draws']:raise ValueError('Predictive draws must be within retained budget')
    call=run([exe,'predict','--run',folder,'--output',output/'predictive',
              '--draws-per-chain',per_chain,'--seed',SEED+1],output/'predictive.log')
    save_json(output/'predictive-command.json',call)
    if call['returncode'] not in [0,5]:raise RuntimeError('Predictive invocation failed')
    latent=np.loadtxt(output/'predictive/latent.txt',ndmin=2)
    replicated=np.loadtxt(output/'predictive/replicated.txt',ndmin=2)
    selected=np.loadtxt(output/'predictive/draws.txt',dtype=str,ndmin=2,skiprows=1)
    data=np.loadtxt(output/'observations.txt',ndmin=2)
    if latent.shape!=(count*per_chain,len(data)) or replicated.shape!=latent.shape or len(selected)!=len(latent):
        raise ValueError('Missing predictive rows')
    predictive=dict(failures=int(np.count_nonzero(selected[:,3]!='valid')),selected=len(latent),by_mode=[])
    if predictive['failures']==0 and np.isfinite(latent).all() and np.isfinite(replicated).all():
        predictive.update(valid=True,posterior_predictive_p=float(np.mean(selected[:,5].astype(float)>=selected[:,4].astype(float))),
            mean_observed_chi2=float(np.mean(selected[:,4].astype(float))),mean_replicated_chi2=float(np.mean(selected[:,5].astype(float))))
        for mode in np.unique(data[:,2]).astype(int):
            mask=data[:,2]==mode;residual=latent[:,mask]-data[mask,1]
            interval=np.quantile(replicated[:,mask],[.025,.975],axis=0)
            predictive['by_mode'].append(dict(mode=int(mode),count=int(mask.sum()),
                rms_of_posterior_mean_km_s=float(np.sqrt(np.mean(residual.mean(axis=0)**2))),
                latent_rms_quantiles=np.quantile(np.sqrt(np.mean(residual**2,axis=1)),[.025,.5,.975]).tolist(),
                observations_inside_pointwise_predictive_95=int(np.sum((data[mask,1]>=interval[0])&(data[mask,1]<=interval[1])))))
    else:predictive['valid']=False
    save_json(output/'predictive-summary.json',predictive)
    result['accepted']=result['accepted'] and predictive['valid'] and all(a['passed'] for a in audits)
    save_json(output/'acceptance.json',dict(accepted=result['accepted'],mixing_passed=result['mixing_passed'],
        numerical_passed=result['numerical_passed'],predictive_valid=predictive['valid'],
        posterior_derivatives_passed=all(a['passed'] for a in audits)))
    write_report(output,config,result,predictive)
    if make_plots:plot(output,config,samples,result,data,latent,replicated,predictive)
    print(f"accepted={result['accepted']} Rhat={result['maximum_rhat']:.5f} bulk={result['minimum_bulk_ess']:.1f} tail={result['minimum_tail_ess']:.1f}",flush=True)
    return 0 if result['accepted'] else 4


def audit_posterior(output,exe,config,samples):
    """Predeclared middle retained draw of every chain; never select by audit outcome."""
    work=output/'posterior-audit';work.mkdir();(work/'lm').mkdir()
    data=np.loadtxt(output/'observations.txt',ndmin=2);n=samples.shape[2];records=[]
    index=samples.shape[1]//2
    for c in range(len(samples)):
        vs=samples[c,index];vp,rho,dvp,drho=brocher(vs)
        np.savetxt(work/f'lm/run_{c}_model.txt',np.c_[config['model']['depth'],config['model']['vs'],
            config['inversion']['starts_vs'][c],vs,vp,rho],fmt='%.17g')
        audit=audit_terminal(exe,config,data,work,c)
        directory=work/f'audit-{c}'
        call=run([exe,'forward','--config',directory/'forward.toml','--sensitivity','implicit',
                  '--output',directory/'standard.txt'],directory/'standard.log')
        if call['returncode'] not in [0,2,3]:raise RuntimeError('Posterior precision check invocation failed')
        standard=forward_rows(directory/'standard.txt');refined=forward_rows(directory/'forward.txt')
        check_queries(standard,data[:,0],data[:,2])
        precision_ok=all(r[4]=='valid' and r[7]=='valid' for r in standard+refined)
        record=dict(chain=c,source_draw=index,refined_fd=audit,passed=False)
        if precision_ok:
            delta=np.abs(np.array([float(r[3]) for r in standard])-np.array([float(r[3]) for r in refined]))
            def jac(rows):
                return vs*(np.array([r[12:12+n] for r in rows],float)
                    +np.array([r[12+n:12+2*n] for r in rows],float)*dvp
                    +np.array([r[12+2*n:12+3*n] for r in rows],float)*drho)
            js,jr=jac(standard),jac(refined)
            relative=np.abs(js-jr)/(2e-6+1e-3*np.maximum(np.abs(js),np.abs(jr)))
            record.update(max_phase_difference_km_s=float(delta.max()),
                max_phase_difference_over_sigma=float(np.max(delta/data[:,3])),
                max_scaled_jacobian_difference=float(relative.max()))
            record['passed']=bool(fd_passed(audit) and record['max_phase_difference_over_sigma']<=1e-4 and relative.max()<=1)
        records.append(record)
    save_json(output/'posterior-audit.json',records)
    return records


def sample_separate_chains(output,exe,workers,resume=False):
    """Independent CLI processes with exactly the serial CLI's seed/chain-id streams."""
    config=read_config(output/'nuts.toml');count=len(config['inversion']['starts_vs'])
    verify_inputs(output,exe)
    if (output/'nuts').exists():raise ValueError('Merged samples already exist; use --analyze-only')
    jobs=[]
    for c,start in enumerate(config['inversion']['starts_vs']):
        directory=output/f'chain-{c}'
        one=copy.deepcopy(config);one['inversion']['starts_vs']=[start];one['nuts']['chain_id_offset']=c
        if directory.exists():
            if not resume:raise ValueError('Existing chain directory; use --resume-sampling')
            if read_config(directory/'nuts.toml')!=one:raise ValueError('Prepared chain configuration changed')
            completed=directory/'sampler-command.json'
            if completed.exists():
                call=json.loads(completed.read_text())
                if call['returncode'] not in [0,5]:raise ValueError('Completed invocation failed; inspect before retrying')
                jobs.append((c,directory));continue
            attempt=1
            while (output/f'chain-{c}-interrupted-{attempt}').exists():attempt+=1
            directory.rename(output/f'chain-{c}-interrupted-{attempt}')
        directory.mkdir()
        write_config(directory/'nuts.toml',one)
        jobs.append((c,directory))
    def execute(job):
        c,directory=job
        if (directory/'sampler-command.json').exists():
            print(f'Chain {c}: retaining completed invocation',flush=True)
            return json.loads((directory/'sampler-command.json').read_text())
        call=run([exe,'nuts','--config',directory/'nuts.toml','--data',output/'observations.txt',
                  '--covariance',output/'covariance.txt','--output',directory/'nuts'],directory/'nuts.log')
        save_json(directory/'sampler-command.json',call)
        print(f'Chain {c} finished: code {call["returncode"]}, {call["seconds"]:.1f} s',flush=True)
        return call
    # Numerical worker processes only; no agents or delegated development.
    with ThreadPoolExecutor(max_workers=workers) as pool:commands=list(pool.map(execute,jobs))
    save_json(output/'sampler-command.json',dict(workers=workers,chains=commands))
    if any(c['returncode'] not in [0,5] for c in commands):raise RuntimeError('A chain invocation failed; individual outputs preserved')
    merged=output/'nuts';merged.mkdir();shutil.copyfile(output/'nuts.toml',merged/'config.toml')
    common=['observations.txt','covariance.txt','mass.txt','prior_parameters.txt','parameter_bounds.txt']
    for filename in common:
        original=output/'chain-0/nuts'/filename
        for c in range(1,count):
            if original.read_bytes()!=(output/f'chain-{c}/nuts'/filename).read_bytes():raise ValueError('Common chain inputs differ')
        shutil.copyfile(original,merged/filename)
    summaries=[]
    for c,_ in jobs:
        source=output/f'chain-{c}/nuts'
        if read_config(source/'config.toml')!=read_config(output/f'chain-{c}/nuts.toml'):
            raise ValueError('Executed chain configuration differs from prepared configuration')
        for suffix in ['vs','z','trace','mass','metric_updates']:
            shutil.copyfile(source/f'chain_0_{suffix}.txt',merged/f'chain_{c}_{suffix}.txt')
        for suffix in ['initial_mass','progress']:
            if (source/f'chain_0_{suffix}.txt').exists():
                shutil.copyfile(source/f'chain_0_{suffix}.txt',merged/f'chain_{c}_{suffix}.txt')
        if (source/'chain_0_failed_evaluations.txt').exists():
            shutil.copyfile(source/'chain_0_failed_evaluations.txt',merged/f'chain_{c}_failed_evaluations.txt')
        row=(source/'chains.txt').read_text().splitlines()[1].split();row[0]=str(c);summaries.append(' '.join(row))
    header=(output/'chain-0/nuts/chains.txt').read_text().splitlines()[0]
    (merged/'chains.txt').write_text(header+'\n'+'\n'.join(summaries)+'\n')
    (merged/'metadata.txt').write_text('sampler=slice_nuts_hoffman_gelman_2014\nlayout=independent_cli_processes\n'
        'streams=identical_to_serial_multi_chain_config\nindividual_metadata=../chain-N/nuts/metadata.txt\n'
        'posterior=conditional on supplied parameterization\n')


def write_report(output,config,result,predictive):
    lines=['# 地壳参考先验 NUTS 验证','',f"整体验收：{'通过' if result['accepted'] else '未通过，区间只作诊断用途'}。",
        f"{result['chains']} 条链，每链预热 {config['nuts']['warmup']}，正式 {result['draws']}；包含拒绝后的重复状态。",
        f"全部参数与派生量：最大 R-hat={result['maximum_rhat']:.6g}，最低 bulk ESS={result['minimum_bulk_ess']:.6g}，最低 tail ESS={result['minimum_tail_ess']:.6g}。",'',
        '| 链 | 正式发散 | 正式无效 | 树深截断 | 正式平均树深 | 冻结步长 |',
        '|---:|---:|---:|---:|---:|---:|']
    for chain in result['sampler']:
        p=chain['production'];lines.append(f"| {chain['chain']} | {p['divergences']} | {p['invalid_evaluations']} | {p['depth_limits']} | {p['mean_tree_depth']:.4g} | {chain['frozen_step_size']:.6g} |")
    lines+=['','## 逐层先验与后验','',
        '速度单位 km/s；区间是固定界面、物性映射、参考先验和误差模型下的逐点 95% 区间。',
        '标准差比值小于 1 表示本次条件后验相对先验收窄，不等于深度分辨率。','',
        '| 层 | 顶深 km | 先验 95% | 后验 95% | 后验均值 | SD比 | R-hat | bulk ESS | tail ESS |',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for p,d,z in zip(result['priors'],result['parameters'],config['model']['depth']):
        lines.append(f"| {d['layer']} | {z:g} | [{p['quantiles'][0]:.6f}, {p['quantiles'][2]:.6f}] | "
            f"[{d['quantiles'][0]:.6f}, {d['quantiles'][2]:.6f}] | {d['mean']:.6f} | {d['posterior_to_prior_sd']:.4f} | "
            f"{d['rhat']:.5f} | {d['ess_bulk']:.1f} | {d['ess_tail']:.1f} |")
    lines+=['','## 后验预测','',f"选取 {predictive['selected']} 个状态，预测失败 {predictive['failures']}。"]
    if predictive['valid']:
        lines += [f"总体 PPC 超越比例={predictive['posterior_predictive_p']:.5f}；使用拟合过的同一数据，不是经典均匀 p 值。",'',
            '| 模态 | 点数 | 后验均值曲线 RMS km/s | 逐点预测区间包含观测数 |','|---:|---:|---:|---:|']
        for m in predictive['by_mode']:lines.append(f"| {m['mode']} | {m['count']} | {m['rms_of_posterior_mean_km_s']:.6f} | {m['observations_inside_pointwise_predictive_95']} |")
    lines+=['','所有失败、预热、正式轨迹和参数诊断均保留。混合诊断通过不证明遍历了所有后验模态；区间不包含层数、界面、参考中心或物性关系的不确定性。','']
    (output/'report.md').write_text('\n'.join(lines))


def plot(output,config,samples,result,data,latent,replicated,predictive):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(output/f'{name}.{ext}',dpi=160)
        plt.close(fig)
    tops=np.array(config['model']['depth']);bottom=max(250.,1.2*tops[-1]);z=np.r_[tops,bottom]
    prior=np.array([d['quantiles'] for d in result['priors']]);post=np.array([d['quantiles'] for d in result['parameters']])
    fig,axes=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
    for ax in axes[:2]:
        ax.fill_betweenx(z,np.r_[prior[:,0],prior[-1,0]],np.r_[prior[:,2],prior[-1,2]],step='post',color='gray',alpha=.25,label='Prior 95%')
        ax.fill_betweenx(z,np.r_[post[:,0],post[-1,0]],np.r_[post[:,2],post[-1,2]],step='post',color='C0',alpha=.4,label='Posterior 95%')
        ax.stairs(post[:,1],z,orientation='horizontal',baseline=None,color='C0',label='Posterior median')
        ax.stairs(config['model']['vs'],z,orientation='horizontal',baseline=None,color='black',label='Reference')
        ax.set(xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(bottom if ax==axes[0] else 40,0));ax.legend(fontsize=8)
    axes[2].stairs([d['posterior_to_prior_sd'] for d in result['parameters']],z,orientation='horizontal',baseline=None)
    axes[2].axvline(1,color='gray',linestyle=':');axes[2].set(xlabel='Posterior SD / prior SD',ylabel='Depth (km)',ylim=(bottom,0))
    for ax in axes:ax.grid(alpha=.2)
    label='Validation passed' if result['accepted'] else f'VALIDATION FAILED ({len(samples)}/4 chains): diagnostic distributions only'
    fig.suptitle(label+'; pointwise intervals conditional on fixed interfaces and declared prior')
    save(fig,'prior-posterior')
    worst_rhat=int(np.argmax([d['rhat'] for d in result['parameters']]))
    worst_ess=int(np.argmin([min(d['ess_bulk'],d['ess_tail']) for d in result['parameters']]))
    shifted=int(np.argmax([abs(d['mean_shift_in_prior_sd']) for d in result['parameters']]))
    weak=int(np.argmax([d['posterior_to_prior_sd'] for d in result['parameters']]))
    chosen=list(dict.fromkeys([worst_rhat,worst_ess,shifted,weak,len(tops)-1]))
    fig,axes=plt.subplots(len(chosen),1,figsize=(12,2.2*len(chosen)),squeeze=False,constrained_layout=True)
    for ax,j in zip(axes.flat,chosen):
        for c in range(len(samples)):ax.plot(samples[c,:,j],linewidth=.5,alpha=.7,label=f'Chain {c}')
        ax.set(xlabel='Retained draw',ylabel='Vs (km/s)',title=f'Layer {j}, top {tops[j]:g} km');ax.grid(alpha=.2)
    axes.flat[0].legend(ncol=4);fig.suptitle(label+'; worst R-hat/ESS, largest shift/SD ratio and halfspace')
    save(fig,'traces')
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    im=ax.imshow(result['correlation'],vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax,label='Posterior correlation')
    ax.set(xlabel='Layer',ylabel='Layer',title=label+'; conditional Vs correlation');save(fig,'correlation')
    if predictive['valid']:
        modes=np.unique(data[:,2]).astype(int)
        fig,axes=plt.subplots(2,len(modes),figsize=(5*len(modes),8),squeeze=False,constrained_layout=True)
        for j,mode in enumerate(modes):
            idx=np.flatnonzero(data[:,2]==mode);idx=idx[np.argsort(data[idx,0])];f=data[idx,0]
            q=np.quantile(latent[:,idx],[.025,.5,.975],axis=0);rep=np.quantile(replicated[:,idx],[.025,.5,.975],axis=0)
            ax=axes[0,j];ax.fill_between(f,rep[0],rep[2],alpha=.15,color='C0',label='Replicated data 95%')
            ax.fill_between(f,q[0],q[2],alpha=.45,color='C0',label='Latent curve 95%');ax.plot(f,q[1],color='C0')
            ax.errorbar(f,data[idx,1],yerr=data[idx,3],fmt='k.',elinewidth=.5,label='Observed, assumed sigma')
            ax.set(xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)',title=f'Mode {mode}');ax.legend(fontsize=7)
            axes[1,j].fill_between(f,(q[0]-data[idx,1])/data[idx,3],(q[2]-data[idx,1])/data[idx,3],alpha=.4)
            axes[1,j].plot(f,(q[1]-data[idx,1])/data[idx,3]);axes[1,j].axhline(0,color='black',linewidth=.6)
            axes[1,j].set(xlabel='Frequency (Hz)',ylabel='Latent residual / assumed sigma')
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle(label+'; predictive intervals are pointwise, not simultaneous bands');save(fig,'predictive')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--chains',type=int,choices=[1,4],default=4)
    parser.add_argument('--metric-adaptation',choices=['none','diagonal','dense'],default='dense')
    parser.add_argument('--target-accept',type=float,default=.95)
    parser.add_argument('--mode-support',choices=['strict_forward','trapped_modes'],default='strict_forward')
    parser.add_argument('--initial-metric',choices=['reference','start'],default='reference')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--resume-sampling',action='store_true',help='Preserve completed chains and restart interrupted chains with the same streams')
    parser.add_argument('--predictive-per-chain',type=int,default=200)
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--workers',type=int,choices=[1,2,4],default=4)
    args=parser.parse_args();exe=args.exe.resolve();output=args.output.resolve()
    if sum([args.analyze_only,args.resume_sampling,args.prepare_only])>1:parser.error('Choose one execution mode')
    if args.analyze_only:return analyze(output,exe,args.predictive_per_chain,not args.no_plot)
    if args.resume_sampling:
        sample_separate_chains(output,exe,args.workers,resume=True)
        return analyze(output,exe,args.predictive_per_chain,not args.no_plot)
    if args.case is None:parser.error('--case is required for a new run')
    prepare(args.case.resolve(),exe,output,args.warmup,args.draws,args.chains,args.metric_adaptation,args.target_accept,
            args.mode_support,args.initial_metric)
    if args.prepare_only:return 0
    sample_separate_chains(output,exe,args.workers)
    return analyze(output,exe,args.predictive_per_chain,not args.no_plot)


if __name__=='__main__':sys.exit(main())
