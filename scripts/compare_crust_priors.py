#!/usr/bin/env python3
"""Compare physical posterior summaries under two priors; never compare ESS as speedup."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from statistics import NormalDist
import numpy as np
from joint_results import read_config,save_json
from benchmark_crust_nuts import prior_statistics
from benchmark_nuts_warmup import digest
from posterior_diagnostics import diagnose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();roots=[a.baseline.resolve(),a.candidate.resolve()];out=a.output.resolve()
    configs=[read_config(r/'nuts.toml') for r in roots];check=read_config(roots[1]/'nuts.toml');check['nuts']['prior_sigma_vs']=configs[0]['nuts']['prior_sigma_vs'];assert check==configs[0]
    assert json.loads((roots[1]/'sampling-summary.json').read_text())['all_complete']
    for name in ['observations.txt','covariance.txt']:assert (roots[0]/name).read_bytes()==(roots[1]/name).read_bytes()
    acceptance=[json.loads((r/'acceptance.json').read_text()) for r in roots];accepted=all(d['accepted'] for d in acceptance)
    x=[np.stack([np.loadtxt(r/f'nuts/chain_{c}_vs.txt') for c in range(4)]) for r in roots]
    assert all(v.shape==(4,1000,45) and np.isfinite(v).all() for v in x)
    prior=[prior_statistics(c) for c in configs];z=np.array(configs[0]['model']['depth']);n=len(z);normal=NormalDist();near=.05
    layers=[];bands=[]
    def describe(samples,pm,psd):
        d=diagnose(samples);return dict(posterior=d,prior_mean=pm,prior_sd=psd,sd_ratio=d['sd']/psd,width95=d['quantiles'][2]-d['quantiles'][0])
    for j in range(n):
        variants=[]
        for k in range(2):
            pr=prior[k][j];d=describe(x[k][:,:,j],pr['mean'],pr['sd'])
            lower,upper=pr['lower'],pr['upper'];mu=pr['untruncated_mean'];sigma=pr['untruncated_sd'];cdf=lambda v:normal.cdf((v-mu)/sigma)
            d.update(near_boundary_fraction=float(np.mean((x[k][:,:,j]<lower+near)|(x[k][:,:,j]>upper-near))),
                prior_near_boundary_fraction=(cdf(lower+near)-cdf(lower)+cdf(upper)-cdf(upper-near))/(cdf(upper)-cdf(lower)))
            variants.append(d)
        b,c=variants;layers.append(dict(layer=j,top_km=float(z[j]),bottom_km=float(z[j+1]) if j+1<n else None,baseline=b,candidate=c,
            median_change_km_s=c['posterior']['quantiles'][1]-b['posterior']['quantiles'][1],
            median_change_baseline_sd=(c['posterior']['quantiles'][1]-b['posterior']['quantiles'][1])/b['posterior']['sd'],width_ratio=c['width95']/b['width95']))
    for low,high in [(0,3),(3,10),(10,20),(20,50),(50,100),(100,float(z[-1])),(float(z[-1]),None)]:
        if high is None:w=np.zeros(n);w[-1]=1;name=f'>={low:g} (halfspace)'
        else:w=np.maximum(0,np.minimum(np.r_[z[1:],high],high)-np.maximum(z,low))/(high-low);name=f'{low:g}-{high:g}'
        variants=[]
        for k in range(2):
            pm=float(w@np.array([d['mean'] for d in prior[k]]));psd=float(np.sqrt(np.sum((w*np.array([d['sd'] for d in prior[k]]))**2)))
            variants.append(describe(x[k]@w,pm,psd))
        b,c=variants;bands.append(dict(name=name,low_km=low,high_km=high,weights=w.tolist(),baseline=b,candidate=c,
            median_change_km_s=c['posterior']['quantiles'][1]-b['posterior']['quantiles'][1],width_ratio=c['width95']/b['width95']))
    predictive=[json.loads((r/'predictive-summary.json').read_text()) for r in roots]
    result=dict(accepted=accepted,acceptance=acceptance,layers=layers,bands=bands,predictive=predictive,boundary_neighborhood_km_s=near,
        interpretation='Different statistical targets. If acceptance fails, comparisons are diagnostic only. Fixed boundaries and parameterization remain part of inference.')
    out.mkdir();save_json(out/'summary.json',result)
    with (out/'layers.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['layer','top_km','median_010','median_020','median_change','width95_010','width95_020','width_ratio','boundary_fraction_010','boundary_fraction_020'])
        for d in layers:w.writerow([d['layer'],d['top_km'],d['baseline']['posterior']['quantiles'][1],d['candidate']['posterior']['quantiles'][1],d['median_change_km_s'],d['baseline']['width95'],d['candidate']['width95'],d['width_ratio'],d['baseline']['near_boundary_fraction'],d['candidate']['near_boundary_fraction']])
    with (out/'bands.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['depth_km','median_010','median_020','median_change','width95_010','width95_020','width_ratio','sd_ratio_010','sd_ratio_020'])
        for d in bands:w.writerow([d['name'],d['baseline']['posterior']['quantiles'][1],d['candidate']['posterior']['quantiles'][1],d['median_change_km_s'],d['baseline']['width95'],d['candidate']['width95'],d['width_ratio'],d['baseline']['sd_ratio'],d['candidate']['sd_ratio']])
    shutil.copyfile(Path(__file__),out/'comparison-script.py')
    files=[r/n for r in roots for n in ['nuts.toml','observations.txt','covariance.txt','acceptance.json','predictive-summary.json']]+[r/f'nuts/chain_{c}_vs.txt' for r in roots for c in range(4)]
    save_json(out/'provenance.json',dict(inputs={str(f):digest(f) for f in files},script_sha256=digest(Path(__file__)),no_new_sampling=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0077bb','#aa4499'];edges=np.r_[z,350.]
    fig,axes=plt.subplots(1,3,figsize=(14,6),constrained_layout=True)
    for k,label in enumerate(['Prior sigma=0.1','Prior sigma=0.2']):
        q=np.quantile(x[k].reshape(-1,n),[.025,.5,.975],axis=0)
        for ax,depth in zip(axes[:2],[350,25]):
            ax.fill_betweenx(edges,np.r_[q[0],q[0,-1]],np.r_[q[2],q[2,-1]],step='post',color=colors[k],alpha=.10)
            ax.stairs(q[1],edges,orientation='horizontal',baseline=None,color=colors[k],ls='-' if k==0 else '--',lw=1.8,label=label)
            ax.set(ylim=(depth,0),xlabel='Vs (km/s)',ylabel='Depth (km)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].stairs([d['width_ratio'] for d in layers],edges,orientation='horizontal',baseline=None,color='#228833')
    axes[2].axvline(1,color='gray',ls=':');axes[2].axvline(2,color='gray',ls='--');axes[2].set(ylim=(350,0),xlabel='95% width ratio (sigma 0.2 / 0.1)',ylabel='Depth (km)');axes[2].grid(alpha=.2)
    fig.suptitle(('Both posterior validations passed' if accepted else 'DIAGNOSTIC ONLY: candidate validation not passed')+'\nPrior sensitivity with identical data, reference and physical bounds')
    for ext in ['png','pdf']:fig.savefig(out/f'prior-sensitivity.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True);pos=np.arange(len(bands))
    for k,key in enumerate(['baseline','candidate']):
        q=np.array([d[key]['posterior']['quantiles'] for d in bands]);axes[0].errorbar(q[:,1],pos+(k-.5)*.15,xerr=[q[:,1]-q[:,0],q[:,2]-q[:,1]],fmt='o',ms=3,color=colors[k],label=f'Prior sigma={configs[k]["nuts"]["prior_sigma_vs"][0]}')
    axes[0].set_yticks(pos,[d['name'] for d in bands]);axes[0].invert_yaxis();axes[0].set(xlabel='Depth-average Vs: median and 95% interval');axes[0].legend()
    axes[1].barh(pos,[d['width_ratio'] for d in bands],color='#228833');axes[1].set_yticks(pos,[d['name'] for d in bands]);axes[1].invert_yaxis();axes[1].axvline(1,color='gray',ls=':');axes[1].axvline(2,color='gray',ls='--');axes[1].set(xlabel='95% width ratio (sigma 0.2 / 0.1)')
    for ax in axes:ax.grid(alpha=.2,axis='x')
    fig.suptitle('Thickness-weighted depth averages; intervals remain conditional on model and bounds')
    for ext in ['png','pdf']:fig.savefig(out/f'band-sensitivity.{ext}',dpi=160)
    print('Saved prior comparison:',out,'accepted=',accepted)


if __name__=='__main__':main()
