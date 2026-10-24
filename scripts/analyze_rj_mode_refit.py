#!/usr/bin/env python3
"""Independent q-fit audit and predeclared structural-mixing comparison; no forwards."""
import copy
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark_crust_depth import digest, rows
from joint_results import read_config, clean
from posterior_diagnostics import diagnose

out=Path(sys.argv[1]).resolve();design=json.loads((out/'design.json').read_text());source=Path(design['source'])
for n,h in design['source_hashes'].items():assert digest(source/n)==h
for n,h in design['hashes'].items():assert digest(out/n)==h
cfg=read_config(out/'rj.toml');oldcfg=read_config(source/'rj.toml');restored=copy.deepcopy(cfg)
restored['rjmcmc']['seed']=oldcfg['rjmcmc']['seed'];restored['rj_correlated_models']=oldcfg['rj_correlated_models'];assert restored==oldcfg
for name in ['dispweave','observations.txt','covariance.txt','truth.toml']:assert digest(out/name)==digest(source/name)
training=json.loads((out/'training.json').read_text());assert training['mixture_config_sha256']==digest(out/'rj.toml')
chains=[[np.array(r,float) for r in rows(source/f'rj/chain_{c}_models.txt')] for c in range(4)]
fitted=0;counts=[]
for model in cfg['rj_correlated_models']:
    k=model['layers'];scale=np.array([5.]*(k-1)+[1.9]*k);parts=[]
    for c,chain in enumerate(chains):
        x=np.array([r[4:] for r in chain if int(r[1])==k]);eligible=len(x)>len(scale)
        counts.append(dict(k=k,chain=c,count=len(x),eligible=eligible))
        if eligible:
            mean=x.sum(axis=0)/len(x);centered=x-mean;s=centered.T@centered/(len(x)-1)
            expected=.9*s+.1*np.diag(np.diag(s))+1e-6*np.diag(scale*scale)
            part=model['components'][len(parts)];np.testing.assert_allclose(part['mean'],mean,atol=1e-13,rtol=1e-12);np.testing.assert_allclose(part['covariance'],expected,atol=1e-13,rtol=1e-12)
            parts.append(part);fitted+=1
    assert len(model['components'])==len(parts)+1
    for part in parts:assert part['weight']==.9/len(parts)
    broad=model['components'][-1];assert broad['weight']==(.1 if parts else 1.)
    np.testing.assert_array_equal(broad['mean'],[5.*j/k for j in range(1,k)]+[1.75]*k)
    np.testing.assert_allclose(broad['covariance'],np.diag(scale*scale/12),atol=1e-14,rtol=0)
assert counts==training['counts']
reports=[];depth=np.linspace(0,5,31)
for folder,label in [(source,'old_q'),(out,'refit_q')]:
    base=json.loads((folder/'summary.json').read_text());cs=[[np.array(r,float) for r in rows(folder/f'rj/chain_{c}_models.txt')] for c in range(4)]
    indicators={key:np.zeros((4,25000)) for key in ['A','B','LVZ']};profiles=[];locations=[]
    for c,chain in enumerate(cs):
        assert len(chain)==25000;vs_depth=[]
        for i,r in enumerate(chain):
            k=int(r[1]);z=r[4:3+k];v=r[3+k:];vs_depth.append(v[np.searchsorted(z,depth,side='right')])
            d=np.minimum(v[:-2]-v[1:-1],v[2:]-v[1:-1]);yes=len(d)>0 and max(d)>=.1
            if yes:
                j=int(np.argmax(d))+1;indicators['LVZ'][c,i]=1;locations.append([z[j-1],z[j],v[j]])
                if k==4:indicators['A' if j==1 else 'B'][c,i]=1
        profiles.append(vs_depth)
    profiles=np.array(profiles);groups={k:dict(per_chain_fraction=v.mean(axis=1).tolist(),**diagnose(v)) for k,v in indicators.items()}
    if label=='old_q':
        saved=json.loads(Path('results/rj-added-mode/comparison-summary.json').read_text())['arms'][1]
        for k in ['A','B']:
            previous=next(g for g in saved['groups'] if g['label']==k);assert previous['per_chain_fraction']==groups[k]['per_chain_fraction']
            for key in ['rhat','ess_bulk','ess_tail']:assert groups[k][key]==previous['diagnostics'][key]
    ds=base['depth_profiles'];correlated=[s['production']['moves']['correlated'] for s in base['sampler']]
    proposals=sum(r['proposals'] for r in correlated);accepted=sum(r['accepted'] for r in correlated)
    reports.append(dict(label=label,source=str(folder),base=base,groups=groups,vs_quantiles=np.quantile(profiles.reshape(-1,31),[.025,.5,.975],axis=0).tolist(),
        conditional_lvz_quantiles=np.quantile(locations,[.025,.5,.975],axis=0).tolist() if locations else None,
        depth_max_rhat=max(d['rhat'] for d in ds),depth_min_bulk=min(d['ess_bulk'] for d in ds),depth_min_tail=min(d['ess_tail'] for d in ds),
        depth_passed=sum(d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400 for d in ds),
        correlated=dict(proposals=proposals,accepted=accepted,rate=accepted/proposals)))
a,b=reports
# The exact predeclared improvement gate; no post-hoc acceptance-rate substitution.
checks=dict(B_rhat_excess_25pct=b['groups']['B']['rhat']-1<=.75*(a['groups']['B']['rhat']-1),
    B_bulk_2x=b['groups']['B']['ess_bulk']>=2*a['groups']['B']['ess_bulk'],B_tail_2x=b['groups']['B']['ess_tail']>=2*a['groups']['B']['ess_tail'],
    depth_rhat_excess_25pct=b['depth_max_rhat']-1<=.75*(a['depth_max_rhat']-1),depth_bulk_2x=b['depth_min_bulk']>=2*a['depth_min_bulk'],depth_tail_2x=b['depth_min_tail']>=2*a['depth_min_tail'])
oldpilot=json.loads(Path('results/rj-lvz-transfer/pilot/summary.json').read_text())['total_target_evaluations']
costs=dict(new_run_target_evaluations=b['base']['total_target_evaluations'],source_training_run_evaluations=a['base']['total_target_evaluations'],earlier_q_training_evaluations=oldpilot,
    new_plus_source_training=b['base']['total_target_evaluations']+a['base']['total_target_evaluations'],lineage_total=b['base']['total_target_evaluations']+a['base']['total_target_evaluations']+oldpilot,independent_audit_model_forwards=48,new_predictive_forwards=0)
summary=dict(runs=reports,clear_improvement_checks=checks,clear_improvement=all(checks.values()),depth=depth.tolist(),costs=costs,
    interpretation='One fresh seed, proposal learned from unaccepted source states; no pooling of samples, no causal proof from a single seed comparison and no posterior certification from improvement.')
(out/'refit-summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
(out/'training-audit.json').write_text(json.dumps(dict(passed=True,empirical_components_verified=fitted,only_changes=['frozen mixture','fresh seed'],source_hashes_unchanged=True,old_structural_diagnostics_reproduced=True,new_forward_calls=0),indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
truth=read_config(out/'truth.toml')['model'];tz=np.array(truth['depth']+[5.]);tv=np.array(truth['vs'])
for i,(r,color) in enumerate(zip(reports,['#0072B2','#7B3294'])):
    q=np.array(r['vs_quantiles']);ax=axes[0,i];ax.fill_betweenx(depth,q[0],q[2],color=color,alpha=.15);ax.plot(q[1],depth,color=color,label=r['label']);ax.plot(np.repeat(tv,2),np.repeat(tz,2)[1:-1],':',color='#228833',label='Target');ax.set(title=r['label']+' | same 14 data',ylabel='Depth (km)',xlabel='Vs (km/s)',ylim=(5,0))
    axes[1,0].plot([d['rhat'] for d in r['base']['depth_profiles']],depth,color=color,label=r['label'])
    axes[1,1].bar(np.arange(4)+(i-.5)*.32,r['groups']['B']['per_chain_fraction'],width=.3,color=color,label=r['label'])
axes[1,0].axvline(1.01,color='black',ls='--');axes[1,0].set(title='Depth chain agreement',xlabel='R-hat',ylabel='Depth (km)',ylim=(5,0))
axes[1,1].set(title='Family B fractions by chain',xlabel='Chain',xticks=range(4),ylabel='Observed fraction, not certified probability')
for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(f'Frozen proposal refit | fixed budget, fresh seed | improvement gate: {all(checks.values())}\nFull posterior NOT certified: remaining mixing and unresolved-support limits')
fig.savefig(out/'refit.png',dpi=150);plt.close(fig)
print(json.dumps(clean(dict(checks=checks,costs=costs,metrics=[dict(label=r['label'],B=r['groups']['B'],depth_rhat=r['depth_max_rhat'],bulk=r['depth_min_bulk'],tail=r['depth_min_tail'],depth_passed=r['depth_passed'],correlated=r['correlated']) for r in reports])),indent=2))
