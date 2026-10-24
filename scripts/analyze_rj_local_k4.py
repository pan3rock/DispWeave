#!/usr/bin/env python3
"""Compare the two fixed-K4 arms without selecting or merging samples."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_crust_depth import digest
from joint_results import read_config,clean
from posterior_diagnostics import diagnose


out=Path(sys.argv[1]).resolve();design=json.loads((out/'design.json').read_text())
for name,sha in design['hashes'].items():assert digest(out/name)==sha
a=read_config(out/'independent/rj.toml');b=read_config(out/'local/rj.toml')
assert b['rjmcmc'].pop('correlated_local_scale')==.2 and a==b
result=[];depth=np.linspace(0,5,31)
for mode in ['independent','local']:
    folder=out/mode;s=json.loads((folder/'summary.json').read_text())
    rows=np.array([np.loadtxt(folder/f'rj/chain_{c}_models.txt') for c in range(4)])
    x=rows[:,:,4:];assert x.shape==(4,25000,7) and np.all(rows[:,:,1]==4)
    for j,param in enumerate(s['parameters']):
        d=diagnose(x[:,:,j]);assert all(d[k]==param[k] for k in ['rhat','ess_bulk','ess_tail'])
    def at(z):return np.take_along_axis(x[:,:,3:],np.sum(x[:,:,:3]<=z,axis=2)[:,:,None],axis=2)[:,:,0]
    profiles=np.stack([at(z) for z in depth],axis=2)
    upper=at(.4)-at(1.2);lower=at(2.5)-at(1.2)
    strengths=np.minimum(x[:,:,3:5]-x[:,:,4:6],x[:,:,5:7]-x[:,:,4:6]);dip=np.max(strengths,axis=2)>=.1
    index=np.argmax(strengths,axis=2);second=dip&(index==1)
    extra=[dict(name='upper_depth_contrast',**diagnose(upper)),dict(name='lower_depth_contrast',**diagnose(lower)),
           dict(name='LVZ_present',**diagnose(dip.astype(float))),dict(name='LVZ_at_second_interior_layer',**diagnose(second.astype(float)))]
    conditional=[]
    for c in range(4):
        vals=[]
        for i in np.flatnonzero(dip[c]):
            j=index[c,i]+1;vals.append([x[c,i,j-1],x[c,i,j],x[c,i,j]-x[c,i,j-1],x[c,i,3+j]])
        conditional.append(dict(chain=c,LVZ_fraction=float(dip[c].mean()),second_interior_fraction=float(second[c].mean()),
            conditional_quantiles=np.quantile(vals,[.025,.5,.975],axis=0).tolist()))
    all_d=s['parameters']+s['depth_profiles']+extra
    good=lambda d:d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400
    groups=[]
    reconstruction=json.loads((folder/'reconstruction.json').read_text())
    assert reconstruction['transitions_verified']==440000 and reconstruction['max_state_error']==0
    replay=np.array([r.split() for r in (folder/'proposals.tsv').read_text().splitlines() if r and not r.startswith('#') and r.split()[3]=='0' and r.split()[4]=='correlated'])
    for t in range(4):
        tr=np.concatenate([np.loadtxt(folder/f'rj/ladder_{c}/temperature_{t}_local_trace.txt',dtype=str,skiprows=1)[2500:] for c in range(4)])
        joint=tr[tr[:,1]=='correlated']
        valid=joint[:,4]=='valid';accepted=joint[:,5]=='1'
        recovered=replay[(replay[:,1].astype(float)==[1,2,4,8][t])&(replay[:,6]=='valid')]
        assert len(recovered)==np.sum(valid)
        if mode=='local':assert np.all(recovered[:,12].astype(float)==0)
        groups.append(dict(temperature=[1,2,4,8][t],attempts=len(joint),accepted=int(accepted.sum()),
            outside=int(np.sum(joint[:,2]=='outside_support')),forward_failures=int(np.sum(joint[:,4]=='forward_failure')),
            valid_rejected=int(np.sum(valid&~accepted)),median_valid_log_accept_ratio=float(np.median(joint[valid,8].astype(float))),
            median_valid_delta_logL_over_T=float(np.median(recovered[:,10].astype(float)))))
    core=s['parameters']+s['depth_profiles']
    min_bulk=min(d['ess_bulk'] for d in core);min_tail=min(d['ess_tail'] for d in core)
    result.append(dict(mode=mode,parameters=s['parameters'],depth_profiles=s['depth_profiles'],extra_diagnostics=extra,
        max_rhat=max(d['rhat'] for d in core),min_bulk_ess=min_bulk,min_tail_ess=min_tail,
        coordinates_passed=sum(good(d) for d in s['parameters']),depths_passed=sum(good(d) for d in s['depth_profiles']),
        all_checks_passed=all(good(d) for d in all_d),accepted=all(good(d) for d in all_d) and s['formal_forward_failures']==s['formal_numerical_failures']==0,
        evaluations=s['total_target_evaluations'],formal_evaluations=s['formal_evaluations'],forward_failures=s['formal_forward_failures'],numerical_failures=s['formal_numerical_failures'],
        min_bulk_per_100k_evaluations=min_bulk/s['total_target_evaluations']*1e5,min_tail_per_100k_evaluations=min_tail/s['total_target_evaluations']*1e5,
        proposal_groups=groups,conditional_lvz=conditional,
        profile_quantiles=np.quantile(profiles.reshape(-1,len(depth)),[.025,.5,.975],axis=0).tolist()))
report=dict(arms=result,costs=json.loads((out/'costs.json').read_text()),
    interpretation='Same fixed-K4 target and proposal-count budget, different actual evaluation costs. No speedup/convergence claim from acceptance alone; all draws retained, no scale scan. Conditional LVZ statistics not used as compressed-time diagnostics.')
(out/'comparison-summary.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(1,4,figsize=(16,5.5),constrained_layout=True)
for j,(r,color) in enumerate(zip(result,['#0072B2','#7B3294'])):
    q=np.array(r['profile_quantiles']);ax=axes[j]
    ax.fill_betweenx(depth,q[0],q[2],color=color,alpha=.2,label='Pointwise 95%')
    ax.plot(q[1],depth,color=color,lw=2,label=r['mode'])
    for c in range(4):ax.plot([p['chain_quantiles'][c][1] for p in r['depth_profiles']],depth,color=color,alpha=.5,lw=.8)
    ax.plot([1.6,1.6,1.,1.,2.3,2.3],[0,.8,.8,1.8,1.8,5],color='#228833',ls=':',lw=2,label='Target (K=3)')
    ax.set(title=f'{r["mode"]}, fixed K=4',xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(5,0))
    axes[2].plot([d['rhat'] for d in r['depth_profiles']],depth,color=color,label=r['mode'])
    axes[3].plot([min(d['ess_bulk'],d['ess_tail'])/r['evaluations']*1e5 for d in r['depth_profiles']],depth,color=color,label=r['mode'])
axes[2].axvline(1.01,color='#333333',ls='--');axes[2].set(xlabel='Depth Vs R-hat',ylabel='Depth (km)',ylim=(5,0))
axes[3].set(xlabel='min(bulk,tail) ESS / 100k target calls',ylabel='Depth (km)',ylim=(5,0))
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('One fixed-K4 comparison | independent vs local correlated increments (scale 0.2) | diagnostics, not certification')
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
print(json.dumps(clean({r['mode']:{k:r[k] for k in ['max_rhat','min_bulk_ess','min_tail_ess','coordinates_passed','depths_passed','all_checks_passed','evaluations','proposal_groups','extra_diagnostics']} for r in result}),indent=2))
