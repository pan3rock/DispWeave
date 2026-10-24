#!/usr/bin/env python3
"""Compare fixed-budget added-mode arms; 200 preregistered prediction forwards only."""
import argparse
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark_crust_depth import forward, rows, digest
from joint_results import read_config, write_config, clean
from posterior_diagnostics import diagnose
from rj_results import load_run, depth_values


def save(p,d):p.write_text(json.dumps(clean(d),indent=2,allow_nan=False)+'\n')

def main(out,predict):
    design=json.loads((out/'design.json').read_text())
    for path,h in design['source_hashes'].items():assert digest(Path(path))==h
    for name,h in design['hashes'].items():assert digest(out/name)==h
    truth=read_config(out/'truth.toml');obs=np.loadtxt(out/'observations.txt');cov=np.loadtxt(out/'covariance.txt')
    reports=[];depth=np.linspace(0,5,31)
    for name in ['baseline','augmented']:
        folder=out/name;cfg,paths,chains=load_run(folder/'rj');base=json.loads((folder/'summary.json').read_text())
        assert len(chains)==4 and all(len(c)==25000 for c in chains)
        values=depth_values(chains,depth);labels=[];loc=[];chain_loc=[];lvz=[]
        for chain in chains:
            labs=[];local=[];events=[]
            for r in chain:
                v=np.array(r['vs']);z=r['interfaces'];contrast=np.minimum(v[:-2]-v[1:-1],v[2:]-v[1:-1])
                yes=len(contrast)>0 and max(contrast)>=.1;events.append(float(yes))
                if yes:
                    j=int(np.argmax(contrast))+1;local.append([z[j-1],z[j],v[j]])
                labs.append(('A' if j==1 else 'B') if yes and r['k']==4 else ('K4_no_dip' if r['k']==4 else f'K{r["k"]}'))
            labels.append(labs);lvz.append(events);loc+=local;chain_loc.append(dict(count=len(local),quantiles=np.quantile(local,[.025,.5,.975],axis=0).tolist() if local else None))
        labels=np.array(labels);groups=[]
        for label in ['K1','K2','K3','A','B','K4_no_dip']:
            indicator=(labels==label).astype(float)
            groups.append(dict(label=label,fraction=float(indicator.mean()),per_chain_fraction=indicator.mean(axis=1).tolist(),diagnostics=diagnose(indicator)))
        q=np.quantile(values.reshape(-1,len(depth)),[.025,.5,.975],axis=0)
        pred=folder/'comparison-predict'
        if predict:pred.mkdir(exist_ok=False)
        selected=[];begin=time.perf_counter()
        for c,chain in enumerate(chains):
            for i in range(0,25000,1000):
                r=chain[i];ident=len(selected);selected.append(dict(row=ident,chain=c,draw=i,log_likelihood=r['log_likelihood']))
                if predict:
                    v=np.array(r['vs']);model=dict(depth=[0.,*r['interfaces']],vs=v.tolist(),vp=(1.8*v).tolist(),rho=[2.3]*r['k'])
                    write_config(pred/f'{ident:03d}.toml',dict(model=model,queries=truth['queries']))
                    forward(folder/'dispweave',pred/f'{ident:03d}.toml',pred/f'{ident:03d}.txt')
        if predict:save(pred/'selection.json',selected);save(pred/'cost.json',dict(model_forwards=100,seconds=time.perf_counter()-begin))
        assert json.loads((pred/'selection.json').read_text())==selected
        latent=np.full((100,14),np.nan);failures=[]
        for s in selected:
            rs=rows(pred/f'{s["row"]:03d}.txt');assert len(rs)==14
            for j,r in enumerate(rs):
                assert abs(float(r[1])-obs[j,0])<1e-12 and int(r[2])==int(obs[j,2])
                if r[4]=='valid':latent[s['row'],j]=float(r[3])
                else:failures.append(dict(row=s['row'],query=j,status=r[4]))
        n=12 if name=='baseline' else 14;assert np.isfinite(latent[:,:n]).all()
        delta=latent[:,:n]-obs[:n,1];chis=np.einsum('ij,jk,ik->i',delta,np.linalg.inv(cov[:n,:n]),delta)
        ll=-.5*(chis+np.linalg.slogdet(cov[:n,:n])[1]+n*np.log(2*np.pi))
        err=float(max(abs(ll-np.array([s['log_likelihood'] for s in selected]))));assert err<1e-7
        pq=[]
        for j in range(14):pq.append(np.quantile(latent[:,j],[.025,.5,.975]).tolist() if np.isfinite(latent[:,j]).all() else None)
        report=dict(arm=name,base_summary=base,depth=depth.tolist(),vs_quantiles=q.tolist(),
            per_chain_vs_quantiles=np.quantile(values,[.025,.5,.975],axis=1).transpose(1,0,2).tolist(),groups=groups,
            lvz_fraction=float(np.mean(lvz)),lvz_diagnostics=diagnose(np.array(lvz)),conditional_lvz=dict(count=len(loc),columns=['top','bottom','Vs'],quantiles=np.quantile(loc,[.025,.5,.975],axis=0).tolist() if loc else None,per_chain=chain_loc),
            predictions=dict(model_count=100,failures=failures,latent_quantiles=pq,own_data_log_likelihood_max_error=err,own_data_chi2_mean=float(chis.mean()),interpretation='Baseline added-mode predictions held out; augmented predictions in sample. Pointwise latent intervals exclude measurement noise.'))
        reports.append(report)
    q0,q1=[np.array(r['conditional_lvz']['quantiles']) for r in reports]
    widths=(q1[2]-q1[0])/(q0[2]-q0[0])
    pilot=json.loads(Path('results/rj-lvz-transfer/pilot/summary.json').read_text())['total_target_evaluations']
    costs=dict(new_sampling_target_evaluations=sum(r['base_summary']['total_target_evaluations'] for r in reports),shared_old_training_evaluations=pilot,
        prediction_model_calls=200,target_model_calls=1,planned_retained_audit_model_calls=96)
    costs['sampling_plus_shared_training']=costs['new_sampling_target_evaluations']+pilot
    summary=dict(arms=reports,conditional_width_ratio_augmented_over_baseline=widths.tolist(),costs=costs,
        interpretation='One fixed seed and noise draw. Frozen proposal unchanged; start states paired. No certified structural probabilities or convergence claim from interval shrinkage; missing modes restrict evaluated support.')
    save(out/'comparison-summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    colors=['#0072B2','#7B3294']
    target=truth['model'];tz=np.array(target['depth']+[5.]);tv=np.array(target['vs'])
    for i,(r,color) in enumerate(zip(reports,colors)):
        q=np.array(r['vs_quantiles']);ax=axes[0,i]
        ax.fill_betweenx(depth,q[0],q[2],color=color,alpha=.15);ax.plot(q[1],depth,color=color,label=r['arm'])
        for cq in r['per_chain_vs_quantiles']:ax.plot(cq[1],depth,color=color,alpha=.25,lw=.7)
        ax.plot(np.repeat(tv,2),np.repeat(tz,2)[1:-1],color='#228833',ls=':',label='Target')
        ax.set(title=r['arm']+' | pointwise 95%',xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(5,0))
        axes[0,2].bar(np.arange(6)+(i-.5)*.32,[g['fraction'] for g in r['groups']],width=.3,color=color,label=r['arm'])
        axes[1,2].plot([p['rhat'] for p in r['base_summary']['depth_profiles']],depth,color=color,label=r['arm'])
        for ax,indices in [(axes[1,0],range(12)),(axes[1,1],range(12,14))]:
            pq=np.array([r['predictions']['latent_quantiles'][j] or [np.nan]*3 for j in indices]);f=obs[list(indices),0]
            ax.fill_between(f,pq[:,0],pq[:,2],color=color,alpha=.15);ax.plot(f,pq[:,1],'.-',color=color,label=r['arm'])
    target_curve=np.array([float(r[3]) for r in rows(out/'truth-forward.txt')])
    for ax,indices,title in [(axes[1,0],list(range(12)),'Original fundamental data'),(axes[1,1],[12,13],'Added mode 1: held-out vs in-sample')]:
        ax.errorbar(obs[indices,0],obs[indices,1],yerr=.03,fmt='k.',label='Observed +/- sigma');ax.plot(obs[indices,0],target_curve[indices],':',color='#228833',label='Target');ax.set(title=title,xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)')
    axes[0,2].set(title='Observed state fractions (not certified probabilities)',xticks=range(6),xticklabels=[g['label'] for g in reports[0]['groups']])
    axes[1,2].axvline(1.01,ls='--',color='black');axes[1,2].set(title='Depth chain agreement',xlabel='R-hat',ylabel='Depth (km)',ylim=(5,0))
    for ax in axes.flat:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Paired data comparison | identical frozen sampler, starts and proposal budget | full posterior NOT certified')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(width_ratio=widths.tolist(),costs=costs)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--predict',action='store_true');a=p.parse_args();main(a.output.resolve(),a.predict)
