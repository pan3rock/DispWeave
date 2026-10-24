#!/usr/bin/env python3
"""One fixed-budget comparison of fixed-interface NUTS and variable-layer RJMCMC.
Same observations, Vp/Vs, density and conditional physical-Vs prior; different
structural assumptions. No tuning, noise reselection or automatic extensions.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from benchmark_crust_depth import digest, forward, rows
from check_qedisp_examples import check_queries
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose
from rj_diagnostics import analyze as analyze_rj, residence
from rj_results import load_run, depth_values

SEED=20261025
NOISE_SEED=2026102501
SIGMA=.03


def invoke(exe, out, name, args, timeout=300):
    start=time.perf_counter()
    with (out/f'{name}.log').open('w') as f:
        try: code=subprocess.run([str(exe),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired: code='timeout'
    record=dict(returncode=code,seconds=time.perf_counter()-start)
    save_json(out/f'{name}-command.json',record)
    print(name,record,flush=True)
    return record


def physical(depth,vs):
    return dict(depth=list(depth),vs=list(vs),vp=(1.8*np.array(vs)).tolist(),rho=[2.3]*len(vs))


def prepare(out,binary):
    out.mkdir(parents=True,exist_ok=False);exe=out/'dispweave';shutil.copy2(binary,exe)
    shutil.copyfile(Path(__file__),out/'run-script.py')
    f=np.geomspace(.15,.9,12);query=dict(frequency_hz=f.tolist(),mode=[0]*len(f))
    truth=physical([0.,1.5],[1.,2.3]);ref=physical([0.,1.5],[1.2,2.1])
    write_config(out/'truth.toml',dict(model=truth,queries=query))
    write_config(out/'reference.toml',dict(model=ref,queries=query))
    rj=dict(rj_prior=dict(min_layers=1,max_layers=4,depth_limit=5.,min_spacing=.3,
        vs_lower=.8,vs_upper=2.7,layer_weights=[1.,1.,1.,1.]),
        rj_physics=dict(vp_vs_ratio=1.8,density=2.3),
        rjmcmc=dict(warmup=10000,draws=100000,vs_step=.08,interface_step=.2,seed=SEED),
        rj_starts=[dict(interfaces=[],vs=[1.8]),dict(interfaces=[1.8],vs=[1.2,2.1]),
                   dict(interfaces=[1.,2.5],vs=[1.1,1.6,2.4]),
                   dict(interfaces=[.8,1.8,3.2],vs=[1.,1.4,1.9,2.4])])
    write_config(out/'rj.toml',rj)
    # Save the experiment and stopping rules before generating data or sampling.
    save_json(out/'protocol.json',dict(design=__doc__,frequencies=f.tolist(),noise_sigma=SIGMA,
        noise_seed=NOISE_SEED,sampler_seed=SEED,rj_warmup=10000,rj_draws=100000,
        nuts_warmup=1000,nuts_draws=1000,chains=4,rj_timeout_seconds=300,nuts_timeout_seconds=120,
        no_retries_tuning_or_extensions=True,predictive_per_chain=100,
        nuts_metric='Supplied GN at reference, then standard dense adaptation; automatic start GN requires Gaussian prior in current CLI',
        rj_mixing='K indicators with pooled mass >= 1% and all 31 depth-grid Vs: Rhat<=1.01, bulk/tail ESS>=400. Unvisited K remains unverified.',
        structural_checks='Per-chain K probabilities, residence/transition counts, 2-to-4-to-2 round trips; interface-bin occupancy diagnostics',
        forward_audit='First/middle/last failed model per chain; retained indices 25000/50000/75000 per chain. All 12 frequencies, independent 5000-segment scan.',
        failure_policy='Keep all failures; no full physical posterior certification with unresolved proposals, even if mixing passes',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=digest(exe),script_sha256=digest(Path(__file__))))
    target=forward(exe,out/'truth.toml',out/'truth-forward.txt')
    reference=forward(exe,out/'reference.toml',out/'reference-forward.txt','implicit')
    for r in [target,reference]:
        check_queries(r,f,np.zeros(len(f)))
        assert all(row[4]=='valid' for row in r)
    assert all(row[7]=='valid' for row in reference)
    c=np.array([float(row[3]) for row in target])
    noise=np.random.default_rng(NOISE_SEED).normal(0,SIGMA,len(f))
    np.savetxt(out/'noise.txt',noise,fmt='%.17g')
    np.savetxt(out/'observations.txt',np.c_[f,c+noise,np.zeros(len(f)),np.full(len(f),SIGMA)],fmt='%.17g')
    np.savetxt(out/'covariance.txt',np.eye(len(f))*SIGMA**2,fmt='%.17g')
    v=np.array(ref['vs']);q=(np.log(v)-np.log(.8))/(np.log(2.7)-np.log(.8))
    dv=v*np.log(2.7/.8)*q*(1-q)
    j=np.array([r[12:14] for r in reference],float)+1.8*np.array([r[14:16] for r in reference],float)
    mass=(j*dv/SIGMA).T@(j*dv/SIGMA);np.linalg.cholesky(mass)
    nuts=dict(model=ref,inversion=dict(lower_vs=[.8,.8],upper_vs=[2.7,2.7],vp_vs_ratio=1.8,
        starts_vs=[[1.1,2.2],[.9,2.4],[1.2,2.1],[1.,2.5]],sensitivity='implicit',
        fixed_vs_layers=[],regularization=dict(type='none',**{'lambda':0.})),
        nuts=dict(prior='uniform_vs',warmup=1000,draws=1000,max_depth=10,step_size=.1,
        target_accept=.95,adapt_step_size=True,metric_adaptation='dense',seed=SEED,
        find_initial_step_size=True,mode_support='trapped_modes',record_progress=True,
        record_failures=True,mass=mass.tolist()))
    write_config(out/'nuts.toml',nuts)
    save_json(out/'provenance.json',dict(binary_sha256=digest(exe),files={n:digest(out/n) for n in
        ['truth.toml','reference.toml','rj.toml','nuts.toml','observations.txt','covariance.txt','noise.txt','protocol.json']},
        helpers={n:digest(Path(__file__).parent/n) for n in ['benchmark_crust_depth.py','check_qedisp_examples.py',
                  'joint_results.py','posterior_diagnostics.py','rj_diagnostics.py','rj_results.py']}))
    calls={}
    for name,command,config,limit in [('nuts','nuts','nuts.toml',120),('rj','rjmcmc','rj.toml',300)]:
        calls[name]=invoke(exe,out,name,[command,'--config',out/config,'--data',out/'observations.txt',
                '--covariance',out/'covariance.txt','--output',out/name],limit)
    complete=True
    for name in ['nuts','rj']:
        if calls[name]['returncode'] not in [0,5]:complete=False
    save_json(out/'sampling-summary.json',dict(calls=calls,complete=complete))
    return complete


def fd_audit(out,x):
    exe=out/'dispweave';query=read_config(out/'truth.toml')['queries'];records=[]
    for c in range(4):
        folder=out/f'nuts-audit-{c}';folder.mkdir();v=x[c,500]
        cfg=dict(model=physical([0.,1.5],v),queries=query);write_config(folder/'center.toml',cfg)
        r=forward(exe,folder/'center.toml',folder/'center.txt','implicit')
        assert all(t[4]=='valid' and t[7]=='valid' for t in r)
        j=(np.array([t[12:14] for t in r],float)+1.8*np.array([t[14:16] for t in r],float))*v
        errors=[]
        for h in [1e-4,5e-5]:
            jac=[]
            for k in range(2):
                pred=[]
                for sign in [-1,1]:
                    w=v.copy();w[k]*=np.exp(sign*h)
                    doc=dict(model=physical([0.,1.5],w),queries=query)
                    name=f'h{h}-j{k}-s{sign}';write_config(folder/f'{name}.toml',doc)
                    rr=forward(exe,folder/f'{name}.toml',folder/f'{name}.txt')
                    assert all(t[4]=='valid' for t in rr);pred.append(np.array([float(t[3]) for t in rr]))
                jac.append((pred[1]-pred[0])/(2*h))
            jf=np.array(jac).T
            errors.append(float(np.max(np.abs(j-jf)/(2e-6+1e-3*np.maximum(np.abs(j),np.abs(jf))))))
        records.append(dict(chain=c,draw=500,max_scaled_errors=errors,passed=max(errors)<=1))
    save_json(out/'nuts-derivatives.json',records)
    return all(r['passed'] for r in records)


def summarize(out):
    assert json.loads((out/'sampling-summary.json').read_text())['complete']
    provenance=json.loads((out/'provenance.json').read_text())
    for name,h in provenance['files'].items():assert digest(out/name)==h
    assert digest(out/'dispweave')==provenance['binary_sha256']
    x=np.stack([np.loadtxt(out/f'nuts/chain_{c}_vs.txt') for c in range(4)])
    assert x.shape==(4,1000,2)
    nd=[diagnose(x[:,:,k]) for k in range(2)];numerics=[]
    for c in range(4):
        t=np.loadtxt(out/f'nuts/chain_{c}_trace.txt',dtype=str,skiprows=1)
        assert t.shape==(2000,10) and (t[:1000,0]=='1').all() and (t[1000:,0]=='0').all()
        f=t[1000:];assert np.ptp(f[:,3].astype(float))==0
        numerics.append(dict(chain=c,divergences=int(f[:,2].astype(int).sum()),
            failures=int(np.count_nonzero(~np.isin(f[:,6],['valid','outside_support']))),
            support_rejections=int(np.count_nonzero(f[:,6]=='outside_support')),depth_caps=int(f[:,9].astype(int).sum()),
            evaluations=int(f[:,7].astype(int).sum())))
    derivative_passed=fd_audit(out,x)
    def good(d):return d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400
    nuts_pass=all(good(d) for d in nd) and all(not any(r[k] for k in ['divergences','failures','depth_caps']) for r in numerics) and derivative_passed
    save_json(out/'nuts-diagnostics.json',dict(accepted=nuts_pass,parameters=nd,sampler=numerics,derivatives_passed=derivative_passed))
    r=analyze_rj(out/'rj',31);save_json(out/'rj-diagnostics.json',r)
    config,paths,chains=load_run(out/'rj');assert len(chains)==4 and all(len(c)==100000 for c in chains)
    layers=np.array([[s['k'] for s in c] for c in chains]);depth=np.linspace(0,5,31)
    # Structural diagnostics include common-depth bins, never cross-K layer indices.
    bin_occupancy=np.zeros((4,100000,30),dtype=np.uint8)
    for c,chain in enumerate(chains):
        for i,state in enumerate(chain):bin_occupancy[c,i]=np.histogram(state['interfaces'],depth)[0]>0
    bins=[diagnose(bin_occupancy[:,:,i].astype(float)) for i in range(30)]
    relevant=[d for p,d in zip(r['interface_bins'],bins) if .01<=p['probability_any']<=.99]
    main_k=[p['diagnostics'] for p in r['layer_probabilities'] if p['probability']>=.01]
    mixing=all(good(d) for d in main_k+r['depth_profiles'])
    structural=all(good(d) for d in relevant)
    trips=[residence(k,2,4)['endpoint_roundtrips'] for k in layers]
    predicted={}
    for name,command in [('nuts','predict'),('rj','rj-predict')]:
        call=invoke(out/'dispweave',out,f'{name}-predict',[command,'--run',out/name,'--output',out/f'{name}-predictive',
            '--draws-per-chain',100,'--seed',SEED+1],120)
        if call['returncode'] not in [0,5]:raise RuntimeError('Prediction failed; outputs retained')
        latent=np.loadtxt(out/f'{name}-predictive/latent.txt');repl=np.loadtxt(out/f'{name}-predictive/replicated.txt')
        selected=np.loadtxt(out/f'{name}-predictive/draws.txt',dtype=str,skiprows=1)
        valid=latent.shape==(400,12) and np.isfinite(latent).all() and np.isfinite(repl).all() and (selected[:,3]=='valid').all()
        item=dict(valid=bool(valid),selected=len(latent))
        if valid:
            obs=np.loadtxt(out/'observations.txt')[:,1];q=np.quantile(repl,[.025,.975],axis=0)
            item.update(rms=float(np.sqrt(np.mean((latent.mean(axis=0)-obs)**2))),
                observations_in_predictive95=int(np.sum((obs>=q[0])&(obs<=q[1]))))
        predicted[name]=item
    # Save a deterministic subset for independent root audit, without outcome selection.
    audit_lines=['# evaluation status K interfaces vs'];selected_models=[]
    for c in range(4):
        failed=rows(out/f'rj/chain_{c}_failed_models.txt')
        for index in sorted(set([0,len(failed)//2,len(failed)-1])) if failed else []:
            row=failed[index];ident=len(selected_models);audit_lines.append(' '.join([str(ident),*row[1:]]))
            selected_models.append(dict(id=ident,kind='failed',chain=c,source_evaluation=int(row[0])))
        for index in [25000,50000,75000]:
            state=chains[c][index];ident=len(selected_models)
            audit_lines.append(' '.join(map(str,[ident,'retained',state['k'],*state['interfaces'],*state['vs']])))
            selected_models.append(dict(id=ident,kind='retained',chain=c,source_draw=index))
    (out/'audit-models.txt').write_text('\n'.join(audit_lines)+'\n');save_json(out/'audit-selection.json',selected_models)
    report=dict(nuts_accepted=nuts_pass,rj_diagnostic_status=r['status'],rj_main_mixing_passed=mixing,
        interface_bin_mixing_passed=structural,interface_bin_diagnostics=bins,
        rj_unvisited_layers=[p['k'] for p in r['layer_probabilities'] if p['probability']==0],
        rj_2_4_2_roundtrips=trips,predictive=predicted,
        physical_posterior_certified=False,
        interpretation='RJMCMC changes both K and interfaces; fixed-interface NUTS knows the true interface. No sampler-speed or unconditional coverage comparison.')
    save_json(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='interface_bin_diagnostics'},indent=2),flush=True)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'rj-diagnostics.json').read_text());n=json.loads((out/'nuts-diagnostics.json').read_text())
    summary=json.loads((out/'summary.json').read_text());truth=read_config(out/'truth.toml')['model']
    fig,axes=plt.subplots(1,4,figsize=(16,5),layout='constrained')
    ax=axes[0];k=[p['k'] for p in r['layer_probabilities']]
    ax.bar(k,[p['probability'] for p in r['layer_probabilities']],color='#dddddd',label='Pooled RJ')
    for c in range(4):ax.plot(k,[p['per_chain'][c] for p in r['layer_probabilities']],'o-',lw=1,ms=3,label=f'Chain {c}')
    ax.set(xlabel='Layers (including halfspace)',ylabel='Probability',xticks=k,ylim=(0,1));ax.legend(fontsize=8)
    ax=axes[1];z=np.array([p['depth'] for p in r['depth_profiles']]);q=np.array([p['quantiles'] for p in r['depth_profiles']])
    ax.fill_betweenx(z,q[:,0],q[:,2],alpha=.10,color='#aa4499',label='RJ pointwise 95%')
    ax.plot(q[:,1],z,color='#aa4499',label='RJ median')
    nq=np.array([p['quantiles'] for p in n['parameters']]);edges=np.array([0.,1.5,5.]);extend=lambda v:np.r_[v,v[-1]]
    ax.fill_betweenx(edges,extend(nq[:,0]),extend(nq[:,2]),step='post',alpha=.10,color='#0077bb',label='NUTS pointwise 95%')
    ax.plot(np.repeat(nq[:,1],2),np.repeat(edges,2)[1:-1],'--',color='#0077bb',label='Fixed-interface NUTS')
    ax.plot(np.repeat(truth['vs'],2),np.repeat(edges,2)[1:-1],color='#228833',lw=2,label='Target')
    ax.set(xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(5,0));ax.legend(fontsize=7)
    ax=axes[2];bins=r['interface_bins'];centers=[.5*(b['lower']+b['upper']) for b in bins]
    ax.barh(centers,[b['probability_any'] for b in bins],height=5/30,color='#aa4499',alpha=.6)
    ax.axhline(1.5,color='#228833',label='Target interface');ax.set(xlabel='P(interface in 0.167 km bin)',ylabel='Depth (km)',ylim=(5,0));ax.legend(fontsize=8)
    ax=axes[3];data=np.loadtxt(out/'observations.txt');ax.errorbar(data[:,0],data[:,1],yerr=SIGMA,fmt='.',color='#333333',label='Data +/- sigma')
    for name,color,style in [('rj','#aa4499','-'),('nuts','#0077bb','--')]:
        if not summary['predictive'][name]['valid']:continue
        latent=np.loadtxt(out/f'{name}-predictive/latent.txt');q=np.quantile(latent,[.025,.975],axis=0)
        ax.fill_between(data[:,0],*q,color=color,alpha=.10);ax.plot(data[:,0],latent.mean(axis=0),style,color=color,label=name.upper())
    ax.set(xscale='log',xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)');ax.legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('RJMCMC vs fixed-interface NUTS | RJ diagnostic status: '+r['status']+'\nShaded: pointwise 95%; RJ not converged: diagnostic estimates only',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(out/f'comparison.{ext}',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exe',type=Path,default=Path('build/dispweave'))
    group=p.add_mutually_exclusive_group();group.add_argument('--analyze-only',action='store_true');group.add_argument('--plot-only',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.plot_only:plot(out)
    elif a.analyze_only or prepare(out,a.exe.resolve()):summarize(out)


if __name__=='__main__':main()
