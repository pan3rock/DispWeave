#!/usr/bin/env python3
"""Read-only comparison and independent B-label audit of frozen-q seed replication."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark_crust_depth import digest
from joint_results import read_config, clean
from posterior_diagnostics import diagnose

out=Path(sys.argv[1]).resolve();protocol=json.loads((out/'protocol.json').read_text());source=Path(protocol['replication_source'])
for n,h in protocol['source_hashes'].items():assert digest(source/n)==h
for n,h in protocol['hashes'].items():assert digest(out/n)==h
cfg=read_config(out/'rj.toml');oldcfg=read_config(source/'rj.toml')
assert cfg['rjmcmc']['seed']==2026103102 and oldcfg['rjmcmc']['seed']==2026103101
cfg['rjmcmc']['seed']=oldcfg['rjmcmc']['seed'];assert cfg==oldcfg
for name in ['observations.txt','covariance.txt','truth.toml','dispweave']:assert digest(out/name)==digest(source/name)
reports=[json.loads((p/'refit-summary.json').read_text()) for p in [source,out]]
assert reports[0]['runs'][0]==reports[1]['runs'][0]
runs=[r['runs'][1] for r in reports]
for folder,r in zip([source,out],runs):
    b=[]
    for c in range(4):
        indicator=[]
        for line in (folder/f'rj/chain_{c}_models.txt').read_text().splitlines():
            if not line or line.startswith('#'):continue
            row=line.split();value=False
            if int(row[1])==4:
                v=list(map(float,row[7:]));value=v[1]-v[2]>=.1 and v[3]-v[2]>=.1
            indicator.append(float(value))
        assert len(indicator)==25000;b.append(indicator)
    b=np.asarray(b);assert b.mean(axis=1).tolist()==r['groups']['B']['per_chain_fraction']
    diag=diagnose(b)
    for key in ['rhat','ess_bulk','ess_tail','mean']:assert diag[key]==r['groups']['B'][key]
    previous=reports[0]['runs'][0]
    checks=[(diag['rhat']-1)/(previous['groups']['B']['rhat']-1)<=.75,
            diag['ess_bulk']/previous['groups']['B']['ess_bulk']>=2,diag['ess_tail']/previous['groups']['B']['ess_tail']>=2,
            (r['depth_max_rhat']-1)/(previous['depth_max_rhat']-1)<=.75,
            r['depth_min_bulk']/previous['depth_min_bulk']>=2,r['depth_min_tail']/previous['depth_min_tail']>=2]
    stored=reports[0 if folder==source else 1]
    assert checks==list(stored['clear_improvement_checks'].values()) and all(checks)==stored['clear_improvement']
metrics=[dict(seed=seed,B=r['groups']['B'],depth_max_rhat=r['depth_max_rhat'],depth_min_bulk=r['depth_min_bulk'],depth_min_tail=r['depth_min_tail'],depth_passed=r['depth_passed'],
    accepted=r['base']['accepted'],formal_forward_failures=r['base']['formal_forward_failures'],formal_evaluations=r['base']['formal_evaluations'],
    correlated=r['correlated'],conditional_lvz_quantiles=r['conditional_lvz_quantiles'],layer_probabilities=r['base']['layer_probabilities']) for seed,r in zip([2026103101,2026103102],runs)]
training=reports[0]['costs'];new_cost=runs[1]['base']['total_target_evaluations'];first_cost=runs[0]['base']['total_target_evaluations']
costs=dict(new_run_target_evaluations=new_cost,first_refit_run_evaluations=first_cost,source_training_run_evaluations=training['source_training_run_evaluations'],earlier_q_training_evaluations=training['earlier_q_training_evaluations'],
    two_runs_plus_training=new_cost+first_cost+training['source_training_run_evaluations']+training['earlier_q_training_evaluations'],new_audit_model_forwards=48,new_predictive_forwards=0)
summary=dict(metrics=metrics,improvement_checks=[r['clear_improvement_checks'] for r in reports],improvement_repeated=all(r['clear_improvement'] for r in reports),
    B_fraction_difference=runs[1]['groups']['B']['mean']-runs[0]['groups']['B']['mean'],costs=costs,
    full_posterior_accepted=all(r['base']['accepted'] for r in runs),
    interpretation='Two seeds, same frozen mixture and unaccepted training sample; both runs kept separate. Repeated relative improvement does not certify exact structure probabilities or physical posterior completeness.')
(out/'replication-summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
(out/'replication-audit.json').write_text(json.dumps(dict(passed=True,only_seed_changed=True,mixture_and_data_identical=True,full_chain_B_indicators_verified=200000,diagnostics_and_improvement_gates_verified=True,source_hashes_unchanged=True,new_forward_calls=0),indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);depth=np.array(reports[0]['depth'])
truth=read_config(out/'truth.toml')['model'];tz=np.array(truth['depth']+[5.]);tv=np.array(truth['vs'])
for i,(r,seed,color) in enumerate(zip(runs,[2026103101,2026103102],['#0072B2','#7B3294'])):
    q=np.array(r['vs_quantiles']);ax=axes[0,i];ax.fill_betweenx(depth,q[0],q[2],color=color,alpha=.15);ax.plot(q[1],depth,color=color,label=f'Seed {seed}');ax.plot(np.repeat(tv,2),np.repeat(tz,2)[1:-1],':',color='#228833',label='Target');ax.set(title=f'Seed {seed} | separate posterior summary',ylabel='Depth (km)',xlabel='Vs (km/s)',ylim=(5,0))
    axes[1,0].plot([d['rhat'] for d in r['base']['depth_profiles']],depth,color=color,label=str(seed))
    axes[1,1].bar(np.arange(4)+(i-.5)*.32,r['groups']['B']['per_chain_fraction'],width=.3,color=color,label=str(seed))
axes[1,0].axvline(1.01,color='black',ls='--');axes[1,0].set(title='Depth chain agreement',xlabel='R-hat',ylabel='Depth (km)',ylim=(5,0))
axes[1,1].set(title='Family B fractions: not certified probabilities',xlabel='Chain',xticks=range(4),ylabel='Observed fraction')
for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(f'Frozen-q replication | improvement repeated: {summary["improvement_repeated"]}\nNo pooled chains; full posterior NOT certified')
fig.savefig(out/'replication.png',dpi=150);plt.close(fig)
print(json.dumps(clean(summary),indent=2))
