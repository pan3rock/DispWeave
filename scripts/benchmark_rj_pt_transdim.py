#!/usr/bin/env python3
"""One frozen-budget K=1..4 PT comparison, with all-temperature accounting."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose
from rj_diagnostics import residence


def read_models(path, prior, draws):
    rows=[np.fromstring(line,sep=' ') for line in path.read_text().splitlines() if line and not line.startswith('#')]
    assert len(rows)==draws
    for i,r in enumerate(rows):
        k=int(r[1]);assert r[0]==i and r[1]==k and 1<=k<=4 and len(r)==2*k+3 and np.isfinite(r).all()
        assert np.all(np.diff(np.r_[0.,r[4:k+3],5.])>=.3) and np.all((r[k+3:]>=.8)&(r[k+3:]<=2.7))
        lp=-math.log(4)+math.lgamma(k)-(k-1)*math.log(5-.3*k)-k*math.log(1.9)
        assert abs(lp-r[3])<1e-12
    return rows


def analyze(out,source):
    cfg=read_config(out/'rj.toml');assert cfg==read_config(out/'rj/config.toml')
    restored=copy.deepcopy(cfg);restored['rjmcmc']=read_config(source/'rj.toml')['rjmcmc'];assert restored==read_config(source/'rj.toml')
    protocol=json.loads((out/'protocol.json').read_text())
    for name,h in protocol['hashes'].items():assert digest(out/name)==h
    start_steps=cfg['rjmcmc']['tempering'].get('cold_start_steps',0)
    n=25000;warm=2500;temps=[1,2,4,8];cost=4;sampler=[];exchange=[];cold=[]
    for c in range(4):
        d=out/f'rj/ladder_{c}';states=[read_models(d/f'temperature_{t}_models.txt',cfg['rj_prior'],n) for t in range(4)]
        assert (out/f'rj/chain_{c}_models.txt').read_bytes()==(d/'temperature_0_models.txt').read_bytes();cold.append(states[0])
        swaps=np.loadtxt(d/'swaps.txt');ids=np.loadtxt(d/'replicas.txt',dtype=int);assert ids.shape==(warm+n,6)
        permutation=np.tile(np.arange(4),(n,1));origins=list(range(4));cursor=0;phase=[0]*4;trips=[0]*4
        for i in range(warm+n):
            elapsed=i-start_steps+1
            pairs=([(0,1),(2,3)] if (elapsed//10-1)%2==0 else [(1,2)]) if elapsed>0 and elapsed%10==0 else []
            for lo,hi in pairs:
                row=swaps[cursor];cursor+=1;assert row[:6].astype(int).tolist()==[i,int(i<warm),lo,hi,origins[lo],origins[hi]]
                np.testing.assert_allclose(row[8],(1/temps[lo]-1/temps[hi])*(row[7]-row[6]),rtol=1e-12,atol=1e-12)
                if row[9]:
                    origins[lo],origins[hi]=origins[hi],origins[lo]
                    if i>=warm:permutation[i-warm,lo],permutation[i-warm,hi]=hi,lo
                if i>=warm:
                    values=[states[lo][i-warm][2],states[hi][i-warm][2]]
                    np.testing.assert_allclose(values,row[[7,6]] if row[9] else row[[6,7]],atol=1e-12,rtol=0)
            assert ids[i].tolist()==[i,int(i<warm),*origins]
            if i==warm:phase=[0]*4
            if i>=warm:
                a,b=origins[0],origins[-1]
                if phase[a]==2:trips[a]+=1
                phase[a]=1
                if phase[b]==1:phase[b]=2
        assert cursor==len(swaps)
        exchange.append(dict(ladder=c,roundtrips=trips,swap_rates=[float(swaps[(swaps[:,1]==0)&(swaps[:,2]==j),9].mean()) for j in range(3)]))
        for t in range(4):
            tr=np.loadtxt(d/f'temperature_{t}_local_trace.txt',dtype=str,skiprows=1);assert tr.shape==(warm+n,11)
            assert (tr[:warm,0]=='1').all() and (tr[warm:,0]=='0').all()
            ev=tr[:,3]=='1';calls=int(ev.sum())+1;cost+=calls
            np.testing.assert_array_equal(tr[ev,9].astype(int),np.arange(1,calls));assert (tr[~ev,9]=='-1').all()
            failed=ev&np.isin(tr[:,4],['forward_failure','numerical_failure'])
            saved=[line.split() for line in (d/f'temperature_{t}_failed_models.txt').read_text().splitlines() if line and not line.startswith('#')]
            np.testing.assert_array_equal(tr[failed,9].astype(int),[int(r[0]) for r in saved])
            for i in range(n):
                post=states[t][i];pre=states[permutation[i,t]][i];trace=tr[warm+i]
                assert int(trace[10])==int(post[1]) and int(trace[7])==int(pre[1])
                if i:
                    assert int(trace[6])==int(states[t][i-1][1])
                    if trace[5]=='0':np.testing.assert_array_equal(pre[1:],states[t][i-1][1:])
            phases={}
            for name,tt in [('warmup',tr[:warm]),('production',tr[warm:])]:
                phases[name]=dict(evaluations=int(np.sum(tt[:,3]=='1')),forward_failures=int(np.sum((tt[:,3]=='1')&(tt[:,4]=='forward_failure'))),numerical_failures=int(np.sum((tt[:,3]=='1')&(tt[:,4]=='numerical_failure'))),
                    moves={m:dict(proposals=int(np.sum(tt[:,1]==m)),accepted=int(np.sum((tt[:,1]==m)&(tt[:,5]=='1')))) for m in ['birth','death','interface','velocity']})
            k=np.array([int(r[1]) for r in states[t]])
            sampler.append(dict(ladder=c,temperature=temps[t],total_evaluations=calls,**phases,**residence(k,1,4),visited_2_4=residence(k,2,4) if np.all(k>=2) else None))
    meta=dict(line.split('=',1) for line in (out/'rj/metadata.txt').read_text().splitlines());assert cost==int(meta['total_target_evaluations'])
    layers=np.array([[int(r[1]) for r in chain] for chain in cold]);depth=np.linspace(0,5,31)
    values=np.array([[r[3+int(r[1]):][np.searchsorted(r[4:3+int(r[1])],depth,side='right')] for r in chain] for chain in cold])
    probs=[dict(k=k,probability=float(np.mean(layers==k)),per_chain=np.mean(layers==k,axis=1).tolist(),diagnostics=diagnose((layers==k).astype(float))) for k in range(1,5)]
    profiles=[dict(depth=float(z),chain_quantiles=np.quantile(values[:,:,j],[.025,.5,.975],axis=1).T.tolist(),**diagnose(values[:,:,j])) for j,z in enumerate(depth)]
    structural=[]
    for index in range(3):
        indicator=np.array([[float(int(r[1])>index+1 and np.argmax(np.abs(np.diff(r[3+int(r[1]):])))==index) for r in chain] for chain in cold])
        structural.append(dict(jump_at_interface=index+1,per_chain=indicator.mean(axis=1).tolist(),**diagnose(indicator)))
    old=json.loads((source/'rj-diagnostics.json').read_text());oldcost=8
    for c in range(4):oldcost+=int(np.sum(np.loadtxt(source/f'rj/chain_{c}_trace.txt',dtype=str,skiprows=1)[:,3]=='1'))
    def good(d):return d['status']=='no_flag' and d['rhat']<=1.01 and min(d['ess_bulk'],d['ess_tail'])>=400
    ds=[p['diagnostics'] for p in probs]+profiles
    report=dict(accepted=False,mixing_passed=all(good(d) for d in ds),visited_K_mixing_passed=all(good(p['diagnostics']) for p in probs if p['probability']>0),depth_mixing_passed=all(good(p) for p in profiles),
        layer_probabilities=probs,depth_profiles=profiles,structural_indicators=structural,sampler=sampler,exchange=exchange,total_target_evaluations=cost,
        formal_evaluations=sum(s['production']['evaluations'] for s in sampler),formal_forward_failures=sum(s['production']['forward_failures'] for s in sampler),formal_numerical_failures=sum(s['production']['numerical_failures'] for s in sampler),
        baseline=dict(total_target_evaluations=oldcost,layer_probabilities=old['layer_probabilities'],depth_profiles=old['depth_profiles'],formal_evaluations=sum(s['target_evaluations'] for s in old['sampler']),formal_forward_failures=sum(s['forward_failures'] for s in old['sampler'])),
        interpretation='Unvisited K is not certified zero probability. Hot states are excluded from posterior diagnostics. Forward failures remain an unresolved support limitation.')
    report['accepted']=report['mixing_passed'] and report['formal_forward_failures']==0 and report['formal_numerical_failures']==0
    if start_steps:
        previous=json.loads(Path('results/rj-pt-transdim/summary.json').read_text())
        report['previous_pt']={k:previous[k] for k in ['layer_probabilities','depth_profiles','exchange','total_target_evaluations','formal_forward_failures','formal_evaluations']}
    save_json(out/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained');colors=['#0077bb','#aa4499','#ee7733','#777777']
    for c,color in enumerate(colors):
        axes[0].plot(range(1,5),[p['per_chain'][c] for p in probs],marker='o',color=color,label=f'Cold chain {c}')
        q=np.array([p['chain_quantiles'][c] for p in profiles]);axes[1].plot(q[:,1],depth,color=color)
    axes[0].set(xticks=range(1,5),xlabel='K (includes halfspace)',ylabel='Observed fraction');axes[0].legend(fontsize=8)
    axes[1].plot([1,1,2.3,2.3],[0,1.5,1.5,5],color='#228833',ls='--',label='Target');axes[1].set(xlabel='Per-chain median Vs (km/s)',ylabel='Depth (km)',ylim=(5,0));axes[1].legend(fontsize=8)
    axes[2].plot([p['rhat'] for p in old['depth_profiles']],depth,label='Original RJ',color='#0077bb')
    axes[2].plot([p['rhat'] for p in profiles],depth,label='RJ + PT',color='#aa4499');axes[2].axvline(1.01,color='#222222',ls='--');axes[2].set(xlabel='Depth Vs R-hat',ylim=(5,0));axes[2].legend(fontsize=8)
    if start_steps:
        axes[2].plot([p['rhat'] for p in report['previous_pt']['depth_profiles']],depth,label='Previous PT',color='#777777',ls=':');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('K=1–4, equal total proposal budget | '+('passed' if report['accepted'] else 'NOT accepted; diagnostic estimates only'))
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ['sampler','baseline','depth_profiles','structural_indicators']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--analyze-only',action='store_true');parser.add_argument('--cold-start-steps',type=int,default=0);parser.add_argument('--save-warmup',action='store_true');a=parser.parse_args()
    if not 0<=a.cold_start_steps<2500:parser.error('cold-start-steps must be in [0,2500)')
    source=Path('results/rj-nuts-comparison').resolve();out=a.output.resolve()
    if a.analyze_only:analyze(out,source);return
    out.mkdir(parents=True,exist_ok=False)
    for name in ['observations.txt','covariance.txt','truth.toml']:shutil.copy2(source/name,out/name)
    shutil.copy2('build/dispweave',out/'dispweave');shutil.copy2(__file__,out/'run-script.py')
    cfg=read_config(source/'rj.toml');cfg['rjmcmc'].update(warmup=2500,draws=25000,tempering=dict(temperatures=[1.,2.,4.,8.],swap_every=10))
    if a.cold_start_steps:cfg['rjmcmc']['tempering']['cold_start_steps']=a.cold_start_steps
    if a.save_warmup:cfg['rjmcmc']['tempering']['save_warmup']=True
    write_config(out/'rj.toml',cfg)
    save_json(out/'protocol.json',dict(source=str(source),total_local_proposals=440000,baseline_total_local_proposals=440000,ladders=4,warmup=2500,draws=25000,temperatures=[1,2,4,8],swap_every=10,no_retries_tuning_or_extensions=True,cold_start_steps=a.cold_start_steps,save_warmup=a.save_warmup,initialization='Independent T=1 updates without exchanges for cold_start_steps within warmup; no resets or best-state selection',
        acceptance='All K=1..4 indicator and 31 depth diagnostics Rhat<=1.01 and bulk/tail ESS>=400; constants/unvisited K remain unresolved; separately report all forward/numerical failures. Overall acceptance requires both mixing and zero formal failures.',
        hashes={name:digest(out/name) for name in ['dispweave','rj.toml','observations.txt','covariance.txt','truth.toml','run-script.py']}))
    call=invoke(out/'dispweave',out,'rj',['rjmcmc','--config',out/'rj.toml','--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',out/'rj'],300)
    if call['returncode'] not in [0,5]:raise RuntimeError('Incomplete run; no sampling retries')
    analyze(out,source)
if __name__=='__main__':main()
