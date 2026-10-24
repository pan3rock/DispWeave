#!/usr/bin/env python3
"""One matched, frozen-proposal experiment adding two validated mode-1 observations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from benchmark_crust_depth import forward, rows
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, clean

SOURCE=Path('results/rj-lvz-transfer/production')
FOLLOW=Path('results/rj-mode1-followup')
SEED=2026103001
NOISE_SEED=2026103000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(SOURCE/'dispweave',out/'dispweave');shutil.copy2(__file__,out/'run-script.py')
    cfg=read_config(SOURCE/'rj.toml');cfg['rjmcmc']['seed']=SEED
    selection=json.loads((FOLLOW/'selection.json').read_text())
    starts=[next(s for s in selection if s['cohort']=='replication' and s['chain']==c and s['family']==('A' if c%2==0 else 'B') and s['selection_rank']==4) for c in range(4)]
    cfg['rj_starts']=[dict(interfaces=s['model'][4:7],vs=s['model'][7:]) for s in starts]
    truth=read_config(SOURCE/'truth.toml')
    augmented=copy.deepcopy(truth)
    augmented['queries']['frequency_hz'] += [.6,.65];augmented['queries']['mode'] += [1,1]
    write_config(out/'truth.toml',augmented)
    save(out/'design.json',dict(source=str(SOURCE),sampler_seed=SEED,noise_seed=NOISE_SEED,added_frequencies=[.6,.65],added_mode=1,sigma=.03,
        total_proposals_per_arm=440000,arms=['baseline','augmented'],warmup=2500,draws=25000,ladders=4,temperatures=[1,2,4,8],
        shared_starts=starts,start_reason='Old K1 start cannot support observed mode1. Both arms use identical preselected, already validated retained K4 states; no truth or new-noise fitting.',
        proposal='Same old LVZ frozen q in both arms, no retraining; full density ratio retained. Old pilot cost charged once. New posterior may fit proposal less well.',
        data='First 12 data/covariance entries unchanged. Two target responses + one fixed Gaussian noise draw, independent variance .03^2. No noise reselection.',
        primary='Compare conditional LVZ top/bottom/Vs widths, unconditional depth intervals, K and A/B fractions and full-chain indicator diagnostics. Width shrinkage or changed fractions do not certify convergence.',
        mixing='Rhat<=1.01 and bulk/tail ESS>=400, status no_flag, for depths/K/structure. Unvisited K and unresolved forward failures prevent full posterior certification.',
        prediction='25 evenly spaced formal states per cold chain per arm, indices 0,1000,...24000; full 14-query latent predictions. Keep failures, no resampling. Baseline high-mode predictions are held-out; augmented high-mode predictions are in-sample.',
        audits='Existing formal acceptance/state/diagnostic audit, 48 retained model forwards per arm on its own data; 1 truth forward, 200 comparison forwards. No independent root scans added.',
        no_retries_tuning_extensions=True,new_sampling_stages=2,
        source_hashes={str(p):sha(p) for p in [SOURCE/'rj.toml',SOURCE/'truth.toml',SOURCE/'observations.txt',SOURCE/'covariance.txt',FOLLOW/'selection.json',FOLLOW/'summary.json',FOLLOW/'audit.json']},
        hashes={n:sha(out/n) for n in ['dispweave','run-script.py','truth.toml']}))
    predictions=forward(out/'dispweave',out/'truth.toml',out/'truth-forward.txt')
    assert len(predictions)==14 and all(r[4]=='valid' for r in predictions)
    old=np.loadtxt(SOURCE/'observations.txt');cov=np.loadtxt(SOURCE/'covariance.txt')
    noise=np.random.default_rng(NOISE_SEED).normal(0,.03,2);np.savetxt(out/'added-noise.txt',noise,fmt='%.17g')
    extra=np.c_[[.6,.65],np.array([float(r[3]) for r in predictions[-2:]])+noise,[1,1],[.03,.03]]
    full=np.vstack([old,extra]);fullcov=np.zeros((14,14));fullcov[:12,:12]=cov;fullcov[12:,12:]=np.eye(2)*.03**2
    np.savetxt(out/'observations.txt',full,fmt='%.17g');np.savetxt(out/'covariance.txt',fullcov,fmt='%.17g')
    # Both arm configs and input hashes are frozen before either sampler starts.
    for name in ['baseline','augmented']:
        folder=out/name;folder.mkdir();shutil.copy2(out/'dispweave',folder/'dispweave')
        write_config(folder/'rj.toml',cfg);write_config(folder/'truth.toml',truth if name=='baseline' else augmented)
        if name=='baseline':
            for n in ['observations.txt','covariance.txt']:shutil.copy2(SOURCE/n,folder/n)
        else:
            for n in ['observations.txt','covariance.txt']:shutil.copy2(out/n,folder/n)
        save(folder/'protocol.json',dict(arm=name,seed=SEED,total_local_proposals=440000,
            hashes={n:sha(folder/n) for n in ['dispweave','rj.toml','truth.toml','observations.txt','covariance.txt']}))
    assert (out/'baseline/rj.toml').read_bytes()==(out/'augmented/rj.toml').read_bytes()
    for name in ['baseline','augmented']:
        folder=out/name
        call=invoke(folder/'dispweave',folder,'rj',['rjmcmc','--config',folder/'rj.toml','--data',folder/'observations.txt','--covariance',folder/'covariance.txt','--output',folder/'rj'],300)
        if call['returncode'] not in [0,5]:raise RuntimeError('Incomplete stage; no sampling retry')
    print('Both arms completed. Analyze and audit separately; do not rerun sampling.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
