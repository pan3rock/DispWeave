#!/usr/bin/env python3
"""Describe conditional prior/posterior contraction using saved physical Vs samples."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from joint_results import read_config,save_json
from benchmark_nuts_warmup import digest
from benchmark_crust_nuts import prior_statistics
from posterior_diagnostics import diagnose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.output.resolve()
    assert json.loads((run/'acceptance.json').read_text())['accepted']
    cfg=read_config(run/'nuts.toml');priors=prior_statistics(cfg);z=np.array(cfg['model']['depth']);n=len(z)
    x=np.stack([np.loadtxt(run/f'nuts/chain_{c}_vs.txt') for c in range(4)])
    assert x.shape==(4,cfg['nuts']['draws'],n) and np.isfinite(x).all()
    prior_mean=np.array([d['mean'] for d in priors]);prior_sd=np.array([d['sd'] for d in priors])
    flat=x.reshape(-1,n);cov=np.cov(flat,rowvar=False);corr=np.corrcoef(flat,rowvar=False)
    old=json.loads((run/'diagnostics.json').read_text())
    np.testing.assert_allclose(corr,old['correlation'],rtol=0,atol=1e-12)
    layers=[]
    for j in range(n):
        d=diagnose(x[:,:,j]);ratio=d['sd']/prior_sd[j];shift=(d['mean']-prior_mean[j])/prior_sd[j]
        np.testing.assert_allclose([ratio,shift],[old['parameters'][j]['posterior_to_prior_sd'],old['parameters'][j]['mean_shift_in_prior_sd']],rtol=0,atol=1e-12)
        layers.append(dict(layer=j,top_km=float(z[j]),bottom_km=float(z[j+1]) if j+1<n else None,
            prior_mean=float(prior_mean[j]),prior_sd=float(prior_sd[j]),posterior_mean=d['mean'],posterior_sd=d['sd'],
            sd_ratio=ratio,shift_in_prior_sd=shift,posterior_q025=d['quantiles'][0],posterior_median=d['quantiles'][1],posterior_q975=d['quantiles'][2],
            half_sd_ratios=[float(np.std(v[:,:,j],ddof=1)/prior_sd[j]) for v in np.array_split(x,2,axis=1)]))
    bands=[]
    for low,high in [(0,3),(3,10),(10,20),(20,50),(50,100),(100,float(z[-1])),(float(z[-1]),None)]:
        if high is None:
            w=np.zeros(n);w[-1]=1;name=f'>= {low:g} (halfspace)'
        else:
            w=np.maximum(0,np.minimum(np.r_[z[1:],high],high)-np.maximum(z,low))/(high-low);name=f'{low:g}-{high:g}'
        assert abs(w.sum()-1)<1e-12
        psd=float(np.sqrt(np.sum((w*prior_sd)**2)));pm=float(w@prior_mean);d=diagnose(x@w)
        np.testing.assert_allclose(d['sd']**2,w@cov@w,rtol=1e-12,atol=1e-15)
        bands.append(dict(name=name,low_km=low,high_km=high,weights=w.tolist(),prior_mean=pm,prior_sd=psd,
            posterior=d,sd_ratio=d['sd']/psd,shift_in_prior_sd=(d['mean']-pm)/psd))
    pairs=sorted([(i,j) for i in range(n) for j in range(i)],key=lambda ij:-abs(corr[ij]))[:10]
    spectrum=np.linalg.eigvalsh(cov/prior_sd[:,None]/prior_sd[None,:]);assert spectrum.min()>0
    result=dict(source_run=str(run),chains=4,draws_per_chain=x.shape[1],layers=layers,bands=bands,
        strongest_correlations=[dict(layer_i=i,layer_j=j,top_i=float(z[i]),top_j=float(z[j]),correlation=float(corr[i,j])) for i,j in pairs],
        standardized_covariance_eigen_sd_ratios=np.sqrt(spectrum).tolist(),
        interpretation='Conditional contraction relative to independent bounded Gaussian physical-Vs prior; not a data-only resolution matrix or prior-sensitivity experiment',
        prior_sensitivity_decision='Worth testing sigma 0.2: many thin-layer marginals barely contract and LM uses the same quadratic reference scale; no new sampling in this analysis')
    out.mkdir();save_json(out/'summary.json',result)
    with (out/'layers.csv').open('w',newline='') as f:
        fields=[k for k in layers[0] if k!='half_sd_ratios'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:v for k,v in d.items() if k in fields} for d in layers)
    with (out/'bands.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['depth_km','prior_sd','posterior_sd','sd_ratio','posterior_mean','mean_shift_prior_sd','rhat','bulk_ess','tail_ess'])
        for b in bands:writer.writerow([b['name'],b['prior_sd'],b['posterior']['sd'],b['sd_ratio'],b['posterior']['mean'],b['shift_in_prior_sd'],b['posterior']['rhat'],b['posterior']['ess_bulk'],b['posterior']['ess_tail']])
    np.savetxt(out/'posterior-correlation.txt',corr,fmt='%.17g')
    shutil.copyfile(Path(__file__),out/'analysis-script.py')
    files=[run/'nuts.toml',run/'acceptance.json',run/'diagnostics.json']+[run/f'nuts/chain_{c}_vs.txt' for c in range(4)]
    save_json(out/'provenance.json',dict(inputs={str(f):digest(f) for f in files},script_sha256=digest(Path(__file__)),helpers={n:digest(Path(__file__).parent/n) for n in ['posterior_diagnostics.py','benchmark_crust_nuts.py','joint_results.py']},no_sampling_or_forward_calls=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edges=np.r_[z,350.];ratios=np.array([d['sd_ratio'] for d in layers]);shift=np.array([d['shift_in_prior_sd'] for d in layers])
    fig,axes=plt.subplots(1,3,figsize=(13,6),constrained_layout=True)
    for ax,depth in zip(axes[:2],[350,20]):
        ax.stairs(ratios,edges,orientation='horizontal',baseline=None,color='#0077bb',lw=1.8)
        ax.axvline(1,color='gray',ls='--');ax.set(ylim=(depth,0),xlim=(0,1.15),xlabel='Posterior SD / prior SD',ylabel='Depth (km)',title='Full depth' if depth==350 else 'Shallow detail');ax.grid(alpha=.2)
    positions=np.arange(len(bands));axes[2].barh(positions-.18,[b['prior_sd'] for b in bands],height=.35,color='#999999',label='Prior')
    axes[2].barh(positions+.18,[b['posterior']['sd'] for b in bands],height=.35,color='#0077bb',label='Posterior')
    axes[2].set_yticks(positions,[b['name'] for b in bands]);axes[2].invert_yaxis();axes[2].set(xlabel='SD of depth-average Vs (km/s)',title='Thickness-weighted band averages');axes[2].legend();axes[2].grid(axis='x',alpha=.2)
    fig.suptitle('Individual thin layers and broad averages have different constraints\nConditional on fixed interfaces, Brocher05, prior and assumed observation noise')
    for ext in ['png','pdf']:fig.savefig(out/f'depth-constraints.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    for ax,depth in zip(axes[:2],[350,25]):
        ax.stairs(shift,edges,orientation='horizontal',baseline=None,color='#0077bb');ax.axvline(0,color='gray',ls='--');ax.set(ylim=(depth,0),xlabel='(Posterior mean - prior mean) / prior SD',ylabel='Depth (km)',title='Mean shift' if depth==350 else 'Shallow detail');ax.grid(alpha=.2)
    im=axes[2].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=axes[2],label='Correlation');axes[2].set(xlabel='Layer index',ylabel='Layer index',title='Posterior parameter trade-offs')
    fig.suptitle('Mean shifts and correlations: not a standalone measure of physical resolution')
    for ext in ['png','pdf']:fig.savefig(out/f'mean-shift-correlation.{ext}',dpi=160)
    print(json.dumps({k:v for k,v in result.items() if k not in ['layers','bands']},indent=2))


if __name__=='__main__':main()
