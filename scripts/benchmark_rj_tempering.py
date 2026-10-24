#!/usr/bin/env python3
"""One predeclared equal-proposal-budget fixed-K4 parallel-tempering experiment."""
import argparse
import json
import math
from pathlib import Path
import shutil
import numpy as np
from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose


def analyze(out, source):
    cfg=read_config(out/'rj.toml');assert cfg==read_config(out/'rj/config.toml')
    protocol=json.loads((out/'protocol.json').read_text())
    for name,h in protocol['hashes'].items():assert digest(out/name)==h
    warm=cfg['rjmcmc']['warmup'];draws=cfg['rjmcmc']['draws'];total=warm+draws
    temps=cfg['rjmcmc']['tempering']['temperatures'];chains=[];cost=4;failures=[];exchanges=[]
    for c in range(4):
        d=out/f'rj/ladder_{c}';states=[]
        for t in range(4):
            a=np.loadtxt(d/f'temperature_{t}_models.txt');assert a.shape==(draws,11)
            np.testing.assert_array_equal(a[:,0],np.arange(draws));assert (a[:,1]==4).all()
            np.testing.assert_allclose(a[:,3],math.lgamma(4)-3*math.log(3.8)-4*math.log(1.9),atol=1e-12,rtol=0)
            assert np.all(np.diff(np.c_[np.zeros(draws),a[:,4:7],np.full(draws,5.)],axis=1)>=.3)
            assert ((a[:,7:]>=.8)&(a[:,7:]<=2.7)).all();assert np.isfinite(a).all();states.append(a)
            tr=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str);assert tr.shape==(total,11)
            assert (tr[:warm,0]=='1').all() and (tr[warm:,0]=='0').all();assert (tr[:,[6,7,10]]=='4').all()
            evaluated=tr[:,3]=='1';cost+=1+int(evaluated.sum())
            np.testing.assert_array_equal(tr[evaluated,9].astype(int),np.arange(1,evaluated.sum()+1))
            failures.append(dict(ladder=c,temperature=temps[t],evaluations=int(evaluated.sum()),
                warmup_failures=int(np.sum(evaluated[:warm]&np.isin(tr[:warm,4],['forward_failure','numerical_failure']))),
                production_failures=int(np.sum(evaluated[warm:]&np.isin(tr[warm:,4],['forward_failure','numerical_failure'])))))
        np.testing.assert_array_equal(states[0],np.loadtxt(out/f'rj/chain_{c}_models.txt'));chains.append(states[0][:,4:])
        swaps=np.loadtxt(d/'swaps.txt');ids=np.loadtxt(d/'replicas.txt',dtype=int);assert ids.shape==(total,6)
        origins=list(range(4));cursor=0;phase=[0]*4;trips=[0]*4
        for i in range(total):
            expected_pairs=([(0,1),(2,3)] if ((i+1)//10-1)%2==0 else [(1,2)]) if (i+1)%10==0 else []
            for lo,hi in expected_pairs:
                row=swaps[cursor];cursor+=1
                assert list(row[:6].astype(int))==[i,int(i<warm),lo,hi,origins[lo],origins[hi]]
                assert np.isclose(row[8],(1/temps[lo]-1/temps[hi])*(row[7]-row[6]),rtol=1e-12,atol=1e-12)
                if row[9]:origins[lo],origins[hi]=origins[hi],origins[lo]
                # Invert disjoint swaps to verify stored raw likelihood caches at both slots.
                if i>=warm:
                    left,right=(row[7],row[6]) if row[9] else (row[6],row[7])
                    np.testing.assert_allclose([states[lo][i-warm,2],states[hi][i-warm,2]],[left,right],atol=1e-12,rtol=0)
            assert list(ids[i])==[i,int(i<warm),*origins]
            if i==warm:phase=[0]*4
            if i>=warm:
                cold,hot=origins[0],origins[-1]
                if phase[cold]==2:trips[cold]+=1
                phase[cold]=1
                if phase[hot]==1:phase[hot]=2
        assert cursor==len(swaps)
        rates=[]
        for lo in range(3):
            s=swaps[(swaps[:,1]==0)&(swaps[:,2]==lo)];rates.append(float(s[:,9].mean()))
        exchanges.append(dict(ladder=c,production_swap_rates=rates,production_cold_hot_cold_roundtrips=trips))
    meta=dict(line.split('=',1) for line in (out/'rj/metadata.txt').read_text().splitlines());assert cost==int(meta['total_target_evaluations'])
    x=np.stack(chains);names=[f'interface_{i+1}_km' for i in range(3)]+[f'vs_{i+1}' for i in range(4)]
    ds=[dict(name=n,**diagnose(x[:,:,i])) for i,n in enumerate(names)];profiles=[]
    for z in np.linspace(0,5,31):
        index=(x[:,:,:3]<=z).sum(axis=2);v=np.take_along_axis(x[:,:,3:],index[:,:,None],axis=2)[:,:,0]
        profiles.append(dict(depth=float(z),chain_quantiles=np.quantile(v,[.025,.5,.975],axis=1).transpose(1,0).tolist(),**diagnose(v)))
    all_d=ds+profiles;mixing=all(d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400 for d in all_d)
    old=json.loads((source/'summary.json').read_text())
    r=dict(fixed_layers=4,accepted=mixing and not any(f['production_failures'] for f in failures),mixing_passed=mixing,
        parameters=ds,depth_profiles=profiles,failures=failures,exchanges=exchanges,total_target_evaluations=cost,
        maximum_rhat=max(d['rhat'] for d in all_d),minimum_bulk_ess=min(d['ess_bulk'] for d in all_d),minimum_tail_ess=min(d['ess_tail'] for d in all_d),
        baseline={k:old[k] for k in ['maximum_rhat','minimum_bulk_ess','minimum_tail_ess','total_target_evaluations']},
        interpretation='Fixed K=4 conditional diagnostic experiment only; hot chains excluded from posterior diagnostics; no cross-K convergence claim.')
    r['minimum_bulk_ess_per_1000_calls']=r['minimum_bulk_ess']/cost*1000
    r['baseline']['minimum_bulk_ess_per_1000_calls']=old['minimum_bulk_ess']/old['total_target_evaluations']*1000
    save_json(out/'summary.json',r)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmark_rj_fixed_k import plot_higher
    plot_higher(out,r)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');y=np.arange(7)
    ax.barh(y-.18,[d['rhat'] for d in old['parameters']],.35,label='Original K=4',color='#0077bb')
    ax.barh(y+.18,[d['rhat'] for d in ds],.35,label='PT cold chains',color='#aa4499')
    ax.axvline(1.01,color='#222222',ls='--');ax.set(yticks=y,yticklabels=names,xlabel='R-hat',xlim=(.99,1.75))
    ax.legend();ax.set_title('Equal total proposal budget | '+('PT passed' if r['accepted'] else 'PT NOT accepted'))
    fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in r.items() if k not in ['parameters','depth_profiles','failures']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=Path('results/rj-fixed-k4').resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ['observations.txt','covariance.txt','truth.toml']:shutil.copy2(source/name,out/name)
    shutil.copy2('build/dispweave',out/'dispweave');shutil.copy2(__file__,out/'run-script.py')
    cfg=read_config(source/'rj.toml');cfg['rjmcmc'].update(warmup=2500,draws=25000,tempering=dict(temperatures=[1.,2.,4.,8.],swap_every=10));write_config(out/'rj.toml',cfg)
    save_json(out/'protocol.json',dict(source=str(source),warmup=2500,draws=25000,ladders=4,temperatures=[1,2,4,8],
        total_local_proposals=440000,baseline_total_local_proposals=440000,no_retries_tuning_or_extensions=True,
        acceptance='All 7 physical parameters +31 depth values: Rhat<=1.01, bulk/tail ESS>=400; zero formal numerical/forward failures',
        costs='All temperatures including warmup, initialization and preflight; swaps reuse caches',
        hashes={name:digest(out/name) for name in ['dispweave','rj.toml','observations.txt','covariance.txt','truth.toml','run-script.py']}))
    call=invoke(out/'dispweave',out,'rj',['rjmcmc','--config',out/'rj.toml','--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',out/'rj'],300)
    if call['returncode'] not in [0,5]:raise RuntimeError('PT run incomplete; no retries')
    analyze(out,source)
if __name__=='__main__':main()
