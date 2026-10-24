#!/usr/bin/env python3
"""LVZ transfer summaries; conditional dip locations are not convergence tests."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_crust_depth import rows
from joint_results import clean
from posterior_diagnostics import diagnose
from rj_results import load_run, depth_values


out=Path(sys.argv[1]).resolve();reports=[]
depth=np.linspace(0,5,61)
for stage in ['pilot','production']:
    cfg,paths,chains=load_run(out/stage/'rj')
    summary=json.loads((out/stage/'summary.json').read_text())
    values=depth_values(chains,depth)
    points=depth_values(chains,np.array([.4,1.2,2.5]))
    upper=points[:,:,0]-points[:,:,1];lower=points[:,:,2]-points[:,:,1]
    fixed=(upper>.1)&(lower>.1)
    lvz=np.zeros((4,25000));locations=[];per_chain=[]
    for c,chain in enumerate(chains):
        loc=[]
        for i,r in enumerate(chain):
            v=r['vs'];z=r['interfaces']
            contrast=np.minimum(v[:-2]-v[1:-1],v[2:]-v[1:-1])
            eligible=np.flatnonzero(contrast>=.1)
            if len(eligible):
                j=eligible[np.argmax(contrast[eligible])]+1
                lvz[c,i]=1;loc.append([z[j-1],z[j],v[j],contrast[j-1]])
        locations.extend(loc)
        per_chain.append(dict(chain=c,probability=float(lvz[c].mean()),
            conditional_count=len(loc),conditional_quantiles=np.quantile(loc,[.025,.5,.975],axis=0).tolist() if loc else None))
    q=np.quantile(values.reshape(-1,len(depth)),[.025,.5,.975],axis=0)
    reports.append(dict(stage=stage,summary={k:summary[k] for k in ['accepted','layer_probabilities','depth_profiles','total_target_evaluations','formal_evaluations','formal_forward_failures','formal_numerical_failures','exchange']},
        depth=depth.tolist(),vs_quantiles=q.tolist(),per_chain_vs_quantiles=np.quantile(values,[.025,.5,.975],axis=1).transpose(1,0,2).tolist(),
        lvz_indicator=diagnose(lvz),fixed_depth_indicator=diagnose(fixed.astype(float)),
        upper_contrast=diagnose(upper),lower_contrast=diagnose(lower),
        conditional_lvz=dict(columns=['top_km','bottom_km','vs_km_s','min_neighbor_contrast_km_s'],
            count=len(locations),quantiles=np.quantile(locations,[.025,.5,.975],axis=0).tolist() if locations else None,
            per_chain=per_chain),
        interpretation='LVZ indicator retains all draws. Location/velocity quantiles condition on at least one >=.1 km/s dip; choose strongest when multiple. No Rhat on filtered samples.'))

obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt');factor=np.linalg.cholesky(cov)
truth=np.array([float(r[3]) for r in rows(out/'truth-forward.txt')]);filled=np.array([float(r[3]) for r in rows(out/'filled-forward.txt')])
det=json.loads((out/'detectability.json').read_text())
lm=np.array([float(r[3]) for r in rows(out/f'noiseless-lm/run_{det["lm_best_run"]}_prediction.txt')]) if det['lm_best_run']>=0 else None
tr=np.loadtxt(out/'predict/draws.txt',dtype=str,skiprows=1);latent=np.loadtxt(out/'predict/latent.txt');rep=np.loadtxt(out/'predict/replicated.txt')
failures=[dict(row=int(r[0]),chain=int(r[1]),draw=int(r[2]),status=r[3]) for r in tr if r[3]!='valid']
predictive=dict(selected=len(tr),failures=failures,failed_count=len(failures))
if not failures:
    lq=np.quantile(latent,[.025,.5,.975],axis=0);rq=np.quantile(rep,[.025,.5,.975],axis=0)
    chi=np.sum(np.linalg.solve(factor,(latent-obs[:,1]).T)**2,axis=0)
    repchi=np.sum(np.linalg.solve(factor,(rep-latent).T)**2,axis=0)
    np.testing.assert_allclose(chi,tr[:,4].astype(float),atol=1e-8,rtol=1e-10)
    np.testing.assert_allclose(repchi,tr[:,5].astype(float),atol=1e-8,rtol=1e-10)
    chosen=rows(out/'predict/selected_models.txt')
    ll=-.5*(chi+np.linalg.slogdet(cov)[1]+12*np.log(2*np.pi))
    np.testing.assert_allclose(ll,[float(r[2]) for r in chosen],atol=1e-7,rtol=0)
    predictive.update(latent_quantiles=lq.tolist(),replicated_quantiles=rq.tolist(),
        observation_coverage=int(np.sum((obs[:,1]>=rq[0])&(obs[:,1]<=rq[2]))),
        truth_coverage=int(np.sum((truth>=lq[0])&(truth<=lq[2]))),
        mean_chi2=float(chi.mean()),posterior_predictive_p=float(np.mean(repchi>=chi)))
report=dict(stages=reports,predictive=predictive,detectability=det,costs=json.loads((out/'costs.json').read_text()),
    interpretation='New-case pilot and production are separate. In-sample predictive fit and conditional LVZ location do not establish convergence or global identifiability; unresolved forward support remains.')
(out/'transfer-summary.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
f=obs[:,0];sigma=np.sqrt(np.diag(cov))
axes[0,0].plot(f,truth,color='#228833',lw=2,label='LVZ target')
axes[0,0].plot(f,filled,color='#777777',ls=':',label='Filled LVZ control')
if lm is not None:axes[0,0].plot(f,lm,color='#D55E00',ls='--',label='Best fixed-geometry 2-layer fit to truth')
axes[0,0].errorbar(f,obs[:,1],yerr=sigma,fmt='k.',label='Observed +/- sigma')
axes[0,0].set(title='Signal check (comparators are restricted)',xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)')
axes[0,1].errorbar(f,obs[:,1],yerr=sigma,fmt='k.',label='Observed +/- sigma',zorder=5)
axes[0,1].plot(f,truth,color='#228833',ls=':',lw=2,label='LVZ target')
if not failures:
    axes[0,1].fill_between(f,rq[0],rq[2],color='#7B3294',alpha=.10,label='95% replicated observations')
    axes[0,1].fill_between(f,lq[0],lq[2],color='#7B3294',alpha=.25,label='95% latent curves')
    axes[0,1].plot(f,lq[1],color='#7B3294',label='Production median')
else:axes[0,1].text(.5,.5,'Prediction failures; intervals withheld',transform=axes[0,1].transAxes,ha='center')
axes[0,1].set(title=f'Production predictive check; failures {len(failures)}/400',xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)')
for i,(r,color) in enumerate(zip(reports,['#0072B2','#7B3294'])):
    q=np.array(r['vs_quantiles']);ax=axes[1,i]
    ax.fill_betweenx(depth,q[0],q[2],color=color,alpha=.2,label='Pointwise 95%')
    ax.plot(q[1],depth,color=color,lw=2,label=r['stage']+' median')
    for cq in r['per_chain_vs_quantiles']:ax.plot(cq[1],depth,color=color,alpha=.5,lw=.8)
    ax.plot([1.6,1.6,1,1,2.3,2.3],[0,.8,.8,1.8,1.8,5],color='#228833',ls=':',lw=2,label='Target')
    ax.set(title=r['stage']+' | all 100000 cold states',xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(5,0),xlim=(.8,2.7))
    probs=r['summary']['layer_probabilities']
    axes[0,2].bar(np.arange(1,5)+(i-.5)*.3,[p['probability'] for p in probs],width=.28,color=color,label=r['stage'])
    axes[1,2].plot([d['rhat'] for d in r['summary']['depth_profiles']],np.linspace(0,5,31),color=color,label=r['stage'])
axes[0,2].set(title='Layer count: truth K=3',xticks=[1,2,3,4],xlabel='K (including halfspace)',ylabel='Observed fraction')
axes[1,2].axvline(1.01,color='#333333',ls='--',label='R-hat 1.01')
axes[1,2].set(title='Chain agreement',xlabel='Depth Vs R-hat',ylabel='Depth (km)',ylim=(5,0))
for ax in axes.flat:ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('New 3-layer low-velocity-zone case | fresh pilot -> frozen q | NOT full posterior certification')
fig.savefig(out/'transfer.png',dpi=160);plt.close(fig)
print(json.dumps(clean(dict(predictive=predictive,events=[dict(stage=r['stage'],lvz=r['lvz_indicator'],fixed_depth=r['fixed_depth_indicator'],conditional=r['conditional_lvz']) for r in reports])),indent=2))
