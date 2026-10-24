#!/usr/bin/env python3
"""Fixed-K=2 isolation of existing RJ within-model moves; no proposal tuning."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=False)
    for name in ['dispweave','observations.txt','covariance.txt','truth.toml']:
        shutil.copy2(source/name,out/name)
    shutil.copyfile(Path(__file__),out/'run-script.py')
    cfg=read_config(source/'rj.toml');original=copy.deepcopy(cfg)
    cfg['rj_prior'].update(min_layers=2,max_layers=2,layer_weights=[1.])
    cfg['rj_starts']=[dict(interfaces=[z],vs=v) for z,v in zip([1.,1.3,1.8,2.5],
        [[1.1,2.2],[.9,2.4],[1.2,2.1],[1.,2.5]])]
    write_config(out/'rj.toml',cfg)
    assert cfg['rjmcmc']==original['rjmcmc'] and cfg['rj_physics']==original['rj_physics']
    save_json(out/'protocol.json',dict(source=str(source),fixed_layers=2,
        question='Does the existing one-coordinate interface/velocity kernel mix at K=2 without cross-dimensional moves?',
        changed_fields=['rj_prior.min_layers','rj_prior.max_layers','rj_prior.layer_weights','rj_starts'],
        conditional_prior='Same as source given K=2: interface Uniform(.3,4.7), independent Vs Uniform(.8,2.7)',
        action_schedule='Unchanged: birth/death/interface/velocity each 1/4. Birth/death unavailable self-transitions retained, not renormalized.',
        warmup=10000,draws=100000,chains=4,seed=cfg['rjmcmc']['seed'],timeout_seconds=300,
        starts=cfg['rj_starts'],no_retries_tuning_or_extensions=True,
        diagnostics='Interface depth, both Vs, and same 31 fixed depth points; Rhat<=1.01 and bulk/tail ESS>=400',
        comparison='Old cross-K depth diagnostics are different estimands; old conditional K=2 subsets only descriptive, not convergence or speedup comparisons',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=digest(out/'dispweave'),script_sha256=digest(Path(__file__)),
        inputs={name:digest(out/name) for name in ['observations.txt','covariance.txt','truth.toml','rj.toml']},
        source_config_sha256=digest(source/'rj.toml')))
    call=invoke(out/'dispweave',out,'rj',['rjmcmc','--config',out/'rj.toml','--data',out/'observations.txt',
                 '--covariance',out/'covariance.txt','--output',out/'rj'],300)
    if call['returncode'] not in [0,5]:
        save_json(out/'summary.json',dict(accepted=False,reason='Incomplete invocation; no pooled analysis'))
        return False
    return True


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text());source=Path(protocol['source'])
    for name,h in protocol['inputs'].items():assert digest(out/name)==h
    assert digest(out/'dispweave')==protocol['binary_sha256']
    cfg=read_config(out/'rj.toml');assert read_config(out/'rj/config.toml')==cfg
    for name in ['observations.txt','covariance.txt']:
        np.testing.assert_array_equal(np.loadtxt(out/name),np.loadtxt(out/'rj'/name))
    chains=[];sampler=[];cost=0
    expected_prior=-np.log(4.4)-2*np.log(1.9)
    for c in range(4):
        a=np.loadtxt(out/f'rj/chain_{c}_models.txt');assert a.shape==(100000,7)
        np.testing.assert_array_equal(a[:,0],np.arange(100000));assert (a[:,1]==2).all()
        np.testing.assert_allclose(a[:,3],expected_prior,rtol=0,atol=1e-12)
        assert np.all((a[:,4]>=.3)&(a[:,4]<=4.7)) and np.all((a[:,5:]>=.8)&(a[:,5:]<=2.7))
        chains.append(a[:,4:])
        t=np.loadtxt(out/f'rj/chain_{c}_trace.txt',dtype=str,skiprows=1);assert t.shape==(110000,10)
        assert (t[:10000,0]=='1').all() and (t[10000:,0]=='0').all() and (t[:,6:8]=='2').all()
        unavailable=np.isin(t[:,1],['birth','death'])
        assert (t[unavailable,2]=='unavailable').all() and (t[unavailable,3]=='0').all() and (t[unavailable,5]=='0').all()
        # All actual target calls + one preflight and one initial call per chain.
        evaluations=int(np.sum(t[:,3]=='1'));cost+=evaluations+2
        f=t[10000:];phases={}
        for name,phase in [('warmup',t[:10000]),('production',f)]:
            phases[name]=dict(evaluations=int(np.sum(phase[:,3]=='1')),
                forward_failures=int(np.sum((phase[:,3]=='1')&(phase[:,4]=='forward_failure'))),
                numerical_failures=int(np.sum((phase[:,3]=='1')&(phase[:,4]=='numerical_failure'))),
                moves={move:dict(proposals=int(np.sum(phase[:,1]==move)),accepted=int(np.sum((phase[:,1]==move)&(phase[:,5]=='1')))) for move in ['birth','death','interface','velocity']})
        sampler.append(dict(chain=c,**phases))
    x=np.stack(chains);ds=[dict(name=name,**diagnose(x[:,:,i])) for i,name in enumerate(['interface_km','upper_vs','halfspace_vs'])]
    depth=np.linspace(0,5,31);profiles=[]
    for z in depth:
        v=np.where(z<x[:,:,0],x[:,:,1],x[:,:,2]);profiles.append(dict(depth=float(z),**diagnose(v)))
    def good(d):return d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400
    mixing=all(good(d) for d in ds+profiles)
    numeric=all(not r['production']['forward_failures'] and not r['production']['numerical_failures'] for r in sampler)
    old=json.loads((source/'rj-diagnostics.json').read_text())
    report=dict(accepted=mixing and numeric,mixing_passed=mixing,numerical_passed=numeric,
        parameters=ds,depth_profiles=profiles,sampler=sampler,total_target_evaluations=cost,
        formal_target_evaluations=sum(r['production']['evaluations'] for r in sampler),
        maximum_rhat=max(d['rhat'] for d in ds+profiles),minimum_bulk_ess=min(d['ess_bulk'] for d in ds+profiles),
        minimum_tail_ess=min(d['ess_tail'] for d in ds+profiles),
        chain_means=x.mean(axis=1).tolist(),correlation=np.corrcoef(x.reshape(-1,3),rowvar=False).tolist(),
        baseline_depth_max_rhat=max(d['rhat'] for d in old['depth_profiles']),
        baseline_depth_min_bulk=min(d['ess_bulk'] for d in old['depth_profiles']),
        baseline_depth_min_tail=min(d['ess_tail'] for d in old['depth_profiles']),
        interpretation='Conditional K=2 check only. Passing does not validate K=3/4 mixing, cross-K weights or full physical support; no speedup claim.')
    save_json(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['parameters','depth_profiles','sampler']},indent=2),flush=True)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'summary.json').read_text());fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    x=np.stack([np.loadtxt(out/f'rj/chain_{c}_models.txt')[:,4:] for c in range(4)])
    names=['Interface depth (km)','Upper Vs (km/s)','Halfspace Vs (km/s)'];truth=[1.5,1.,2.3]
    for i,ax in enumerate(axes):
        for c in range(4):
            q=np.quantile(x[c,:,i],[.025,.5,.975]);ax.errorbar(c,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='o',capsize=4,color=['#0077bb','#aa4499','#ee7733','#777777'][c],label=f'Chain {c}')
        ax.axhline(truth[i],color='#228833',label='Target')
        ax.set(xlabel='Chain',ylabel=names[i],xticks=range(4));ax.grid(alpha=.15)
        d=r['parameters'][i];ax.set_title(f"R-hat {d['rhat']:.4f}; bulk/tail {d['ess_bulk']:.0f}/{d['ess_tail']:.0f}",fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle('Fixed K=2, free interface | '+('mixing checks passed' if r['mixing_passed'] else 'mixing checks failed')+'\nPer-chain median and 95% posterior interval; same proposals and budget',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(out/f'fixed-k-summary.{ext}',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('results/rj-nuts-comparison'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot-only',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.plot_only:plot(out)
    elif prepare(a.source.resolve(),out):analyze(out)


if __name__=='__main__':main()
