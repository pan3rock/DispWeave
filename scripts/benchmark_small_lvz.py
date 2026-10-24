#!/usr/bin/env python3
"""One preregistered six-layer consistency check; no tuning or noise reselection."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmark_crust_depth import brocher, forward, audit_terminal, digest, rows
from benchmark_crust_regularization import fd_passed
from benchmark_crust_nuts import analyze
from benchmark_nuts_warmup import progress_rows, phase_stats
from check_qedisp_examples import check_queries
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose


def model(depth, vs):
    vp, rho, _, _ = brocher(np.array(vs))
    return dict(depth=depth, vs=vs, vp=vp.tolist(), rho=rho.tolist())


def invoke(exe, out, name, args, timeout):
    start = time.perf_counter()
    with (out/f'{name}.log').open('w') as log:
        try:
            code = subprocess.run([str(exe), *map(str, args)], stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 'timeout'
    rec = dict(returncode=code, seconds=time.perf_counter()-start)
    save_json(out/f'{name}-command.json', rec)
    return rec


def prepare(out, binary):
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'dispweave'; shutil.copy2(binary, exe)
    shutil.copyfile(Path(__file__), out/Path(__file__).name)
    depth = [0., 2., 5., 9., 15., 25.]
    truth = [2.8, 3.3, 3.05, 3.65, 4., 4.4]
    ref = np.array([2.9, 3.15, 3.35, 3.65, 3.95, 4.3])
    frequencies = np.geomspace(.03, .5, 36)
    query = dict(frequency_hz=frequencies.tolist(), mode=[0]*36)
    starts = [(ref+d).tolist() for d in [-.15, .15,
              .15*np.sin(np.arange(6)*np.pi/3), .15*np.cos(np.arange(6)*np.pi/3)]]
    cfg = dict(model=model(depth, ref.tolist()), inversion=dict(
        lower_vs=(ref-.6).tolist(), upper_vs=(ref+.6).tolist(),
        vs2model='brocher05', starts_vs=starts, fixed_vs_layers=[], sensitivity='implicit',
        max_iterations=200, gradient_tolerance=1e-5, initial_damping=.001,
        regularization=dict(type='none', **{'lambda':0.})),
        nuts=dict(prior='reference_gaussian_vs', prior_sigma_vs=[.1]*6,
        warmup=1000, draws=1000, max_depth=10, step_size=.1, target_accept=.95,
        adapt_step_size=True, metric_adaptation='dense', seed=20261024,
        mode_support='trapped_modes', record_progress=True, record_failures=True,
        initial_metric='start_gauss_newton', find_initial_step_size=True))
    write_config(out/'truth.toml', dict(model=model(depth, truth), queries=query))
    write_config(out/'reference.toml', dict(model=cfg['model'], queries=query))
    write_config(out/'nuts.toml', cfg)
    lm = copy.deepcopy(cfg); del lm['nuts']
    lm['inversion']['regularization'] = dict(type='reference_damping', **{'lambda':100.})
    write_config(out/'inversion.toml', lm)
    save_json(out/'protocol.json', dict(
        design='Single fixed-truth matched-parameterization consistency check, not coverage estimation or independent solver validation',
        truth=truth, reference=ref.tolist(), depth=depth, modes=[0], frequencies=frequencies.tolist(),
        noise=dict(distribution='independent Gaussian', sigma_km_s=.02, numpy_seed=2026102401),
        prior_sd_km_s=.1, lm_lambda=100., chains=4, warmup=1000, draws=1000,
        timeout_seconds_per_chain=600, lm_timeout_seconds=120, workers=4,
        no_tuning_retries_or_extensions=True, predictive_states_per_chain=100, fd_state=500,
        extra_diagnostic='Vs[2]-Vs[1], low velocity contrast',
        acceptance=dict(max_rhat=1.01, min_bulk_ess=400, min_tail_ess=400,
                        formal_divergences=0, formal_invalid=0, formal_depth_caps=0),
        binary_sha256=digest(exe), script_sha256=digest(Path(__file__)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
    curves = {}
    for name in ['truth', 'reference']:
        r = forward(exe, out/f'{name}.toml', out/f'{name}-forward.txt')
        check_queries(r, frequencies, np.zeros(36))
        if any(v[4]!='valid' for v in r):
            raise RuntimeError(f'{name}: invalid declared queries; no frequency replacement')
        curves[name] = np.array([float(v[3]) for v in r])
    noise = np.random.default_rng(2026102401).normal(0, .02, 36)
    np.savetxt(out/'noise.txt', noise, fmt='%.17g')
    np.savetxt(out/'observations.txt', np.c_[frequencies, curves['truth']+noise,
                                           np.zeros(36), np.full(36,.02)], fmt='%.17g')
    np.savetxt(out/'covariance.txt', np.eye(36)*.02**2, fmt='%.17g')
    save_json(out/'provenance.json', dict(binary_sha256=digest(exe), files={
        n:digest(out/n) for n in ['truth.toml','reference.toml','inversion.toml','nuts.toml',
                                  'observations.txt','covariance.txt','noise.txt','protocol.json']}))
    call = invoke(exe, out, 'lm', ['invert','--config',out/'inversion.toml',
        '--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',out/'lm'],120)
    if call['returncode'] not in [0,4]:
        raise RuntimeError('LM failed; outputs retained')
    jobs = []
    for c, start in enumerate(starts):
        folder = out/f'chain-{c}'; folder.mkdir()
        one = copy.deepcopy(cfg); one['inversion']['starts_vs']=[start]; one['nuts']['chain_id_offset']=c
        write_config(folder/'nuts.toml', one); jobs.append((c,folder))
    def sample(job):
        c, folder = job
        rec = invoke(exe, folder, 'sampler', ['nuts','--config',folder/'nuts.toml',
            '--data',out/'observations.txt','--covariance',out/'covariance.txt',
            '--output',folder/'nuts'],600)
        r, partial = progress_rows(folder/'nuts/chain_0_progress.txt', 6)
        rec.update(chain=c, complete=rec['returncode'] in [0,5] and len(r)==2000,
                   partial_lines=partial, warmup=phase_stats(r[:1000]), production=phase_stats(r[1000:]))
        print(f'chain {c}: {rec}', flush=True)
        return rec
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(sample, jobs))
    save_json(out/'sampling-summary.json', dict(chains=records, all_complete=all(r['complete'] for r in records)))
    if not all(r['complete'] for r in records):
        save_json(out/'acceptance.json', dict(accepted=False, reason='Incomplete declared chains; no pooled analysis'))
        return False
    merged = out/'nuts'; merged.mkdir(); shutil.copyfile(out/'nuts.toml', merged/'config.toml')
    for name in ['observations.txt','covariance.txt','mass.txt','prior_parameters.txt','parameter_bounds.txt']:
        source = out/'chain-0/nuts'/name
        for c in range(1,4):
            assert source.read_bytes()==(out/f'chain-{c}/nuts'/name).read_bytes()
        shutil.copyfile(source, merged/name)
    for c in range(4):
        for suffix in ['vs','z','trace','mass','metric_updates','initial_mass','progress','cost','failed_evaluations','step_size_searches']:
            shutil.copyfile(out/f'chain-{c}/nuts/chain_0_{suffix}.txt', merged/f'chain_{c}_{suffix}.txt')
    return True


def summarize(out):
    exe=out/'dispweave'; analyze(out, exe, 100, make_plots=False)
    cfg=read_config(out/'nuts.toml'); truth=np.array(read_config(out/'truth.toml')['model']['vs'])
    x=np.stack([np.loadtxt(out/f'nuts/chain_{c}_vs.txt') for c in range(4)])
    diagnostic=json.loads((out/'diagnostics.json').read_text())
    acceptance=json.loads((out/'acceptance.json').read_text())
    contrast=diagnose(x[:,:,2]-x[:,:,1])
    contrast_ok=contrast['status']=='no_flag' and contrast['rhat']<=1.01 and min(contrast['ess_bulk'],contrast['ess_tail'])>=400
    data=np.loadtxt(out/'observations.txt'); costs=np.loadtxt(out/'lm/costs.txt',ndmin=2)
    reports=[]; predictions=[]; models=[]
    lm_cfg=read_config(out/'inversion.toml')
    for row in rows(out/'lm/runs.txt'):
        i=int(row[0]); audit=audit_terminal(exe,lm_cfg,data,out,i)
        vs=np.loadtxt(out/f'lm/run_{i}_model.txt')[:,3]; models.append(vs)
        pred=np.array([float(r[3]) for r in rows(out/f'audit-{i}/forward.txt')]); predictions.append(pred)
        penalty=.5*100*np.sum((vs-np.array(cfg['model']['vs']))**2)
        np.testing.assert_allclose(costs[i,2],penalty,rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(costs[i,1],.5*np.sum(((pred-data[:,1])/.02)**2),rtol=1e-6,atol=1e-5)
        reports.append(dict(run=i,converged=row[1]=='1',termination=row[-2],reason=row[-1],
            fd_passed=fd_passed(audit),data_cost=float(costs[i,1]),prior_cost=float(costs[i,2]),
            rms=float(np.sqrt(np.mean((pred-data[:,1])**2))),vs=vs.tolist()))
    best=int(np.argmin([r['data_cost']+r['prior_cost'] for r in reports]))
    latent=np.loadtxt(out/'predictive/latent.txt'); replicated=np.loadtxt(out/'predictive/replicated.txt')
    quant=np.quantile(x.reshape(-1,6),[.025,.5,.975],axis=0)
    costs_nuts=[]
    for c in range(4):
        d={k:int(v) for k,v in (line.split('=') for line in (out/f'nuts/chain_{c}_cost.txt').read_text().splitlines())}
        assert d['target_evaluations']==1+d['transition_evaluations']+d['step_search_evaluations']
        d['total_evaluations']=d['target_evaluations']+d['preflight_evaluations']+d['metric_evaluations']; costs_nuts.append(d)
    report=dict(accepted=acceptance['accepted'] and contrast_ok and all(r['converged'] and r['fd_passed'] for r in reports),
        nuts_accepted=acceptance['accepted'] and contrast_ok,contrast=contrast,
        probability_low_velocity=float(np.mean(x[:,:,2]<x[:,:,1])),
        truth_contrast=float(truth[2]-truth[1]),lm_runs=reports,best_lm=best,
        truth_in_pointwise95=((truth>=quant[0])&(truth<=quant[2])).tolist(),
        posterior_quantiles=quant.tolist(),noise_rms=float(np.sqrt(np.mean(np.loadtxt(out/'noise.txt')**2))),
        posterior_mean_prediction_rms=float(np.sqrt(np.mean((latent.mean(axis=0)-data[:,1])**2))),
        posterior_sd_over_prior=[d['posterior_to_prior_sd'] for d in diagnostic['parameters']],
        maximum_rhat=max(diagnostic['maximum_rhat'],contrast['rhat']),
        minimum_bulk_ess=min(diagnostic['minimum_bulk_ess'],contrast['ess_bulk']),
        minimum_tail_ess=min(diagnostic['minimum_tail_ess'],contrast['ess_tail']),costs=costs_nuts,
        interpretation='One realization, pointwise intervals, no coverage-rate claim; identical forward family for generation and inference')
    save_json(out/'summary.json',report)
    plot(out)
    print(json.dumps(report,indent=2),flush=True)


def plot(out):
    cfg=read_config(out/'nuts.toml')
    truth=np.array(read_config(out/'truth.toml')['model']['vs'])
    report=json.loads((out/'summary.json').read_text()); best=report['best_lm']
    quant=np.array(report['posterior_quantiles']); models=[r['vs'] for r in report['lm_runs']]
    predictions=[np.array([float(r[3]) for r in rows(out/f'audit-{i}/forward.txt')]) for i in range(4)]
    data=np.loadtxt(out/'observations.txt'); latent=np.loadtxt(out/'predictive/latent.txt')
    replicated=np.loadtxt(out/'predictive/replicated.txt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained',gridspec_kw={'width_ratios':[1,1.35,1.1]})
    z=np.r_[cfg['model']['depth'],35.]
    extend=lambda v:np.r_[v,v[-1]]
    ax=axs[0];ax.fill_betweenx(z,extend(quant[0]),extend(quant[2]),step='post',color='#0077bb',alpha=.10,label='NUTS pointwise 95%')
    for values,color,style,label in [(truth,'#228833','-','Target'),(cfg['model']['vs'],'#888888','--','Reference'),(quant[1],'#0077bb','-','NUTS median'),(models[best],'#cc3377','--','LM')]:
        ax.plot(np.repeat(values,2),np.repeat(z,2)[1:-1],color=color,ls=style,lw=2,label=label)
    ax.invert_yaxis();ax.set(xlabel='Vs (km/s)',ylabel='Depth (km)',title='Six-layer model (last layer: halfspace)');ax.legend(fontsize=8)
    f=data[:,0];ax=axs[1];q=np.quantile(replicated,[.025,.975],axis=0)
    ax.fill_between(f,*q,color='#0077bb',alpha=.10,label='Predictive 95% (with noise)')
    ax.errorbar(f,data[:,1],yerr=.02,fmt='.',color='#333333',alpha=.6,label='Data +/- 1 sigma')
    for pred,color,style,label in [(np.array([float(r[3]) for r in rows(out/'truth-forward.txt')]),'#228833','-','Target'),(latent.mean(axis=0),'#0077bb','-','NUTS mean'),(predictions[best],'#cc3377','--','LM')]:
        ax.plot(f,pred,color=color,ls=style,lw=2,label=label)
    ax.set(xscale='log',xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)',title='Fundamental Rayleigh dispersion');ax.legend(fontsize=8)
    ax=axs[2];ax.axhline(0,color='#888888',lw=1)
    ax.axhspan(-2,2,color='#888888',alpha=.08)
    ax.plot(f,(predictions[best]-data[:,1])/.02,'--',color='#cc3377',label='LM')
    ax.plot(f,(latent.mean(axis=0)-data[:,1])/.02,color='#0077bb',label='NUTS mean')
    ax.set(xscale='log',xlabel='Frequency (Hz)',ylabel='(Prediction - data) / sigma',title='Residuals; known sigma = 0.02');ax.legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle('Independent small case | '+('sampling and numerical checks passed' if report['accepted'] else 'diagnostic results: validation incomplete'))
    for ext in ['png','pdf']:fig.savefig(out/f'comparison.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,default=Path('build/dispweave'))
    p.add_argument('--output',type=Path,required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--plot-only',action='store_true',help='Redraw saved results without forward evaluation or sampling')
    modes.add_argument('--analyze-only',action='store_true',help='Analyze completed samples once; never rerun sampling')
    a=p.parse_args(); out=a.output.resolve()
    if a.plot_only:plot(out)
    elif a.analyze_only or prepare(out,a.exe.resolve()):summarize(out)


if __name__=='__main__':main()
