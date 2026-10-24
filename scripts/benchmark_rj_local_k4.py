#!/usr/bin/env python3
"""One frozen-scale fixed-K=4 comparison: independent versus local correlated moves."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from benchmark_rj_pt_transdim import analyze
from joint_results import read_config, write_config, save_json


parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
source=Path('results/rj-lvz-transfer/production').resolve()
cfg=read_config(source/'rj.toml')
cfg['rj_prior'].update(min_layers=4,max_layers=4,layer_weights=[1.])
cfg['rj_correlated_models']=[m for m in cfg['rj_correlated_models'] if m['layers']==4]
cfg['rjmcmc']['seed']=2026102901
cfg['rj_starts']=[dict(interfaces=z,vs=v) for z,v in [
    ([.6,1.4,3.],[1.1,1.3,2.2,2.5]),([1.,2.,4.],[1.,1.8,2.3,2.5]),
    ([.8,1.8,3.2],[1.,1.4,1.9,2.4]),([1.3,2.7,4.2],[.9,2.2,2.4,2.5])]]
shutil.copy2('build/dispweave',out/'dispweave');shutil.copy2(__file__,out/'run-script.py')
shutil.copy2('scripts/benchmark_rj_pt_transdim.py',out/'analysis-script.py')
for name in ['truth.toml','observations.txt','covariance.txt']:shutil.copy2(source/name,out/name)
write_config(out/'baseline.toml',cfg)
local=copy.deepcopy(cfg);local['rjmcmc']['correlated_local_scale']=.2;write_config(out/'local.toml',local)
save_json(out/'design.json',dict(source=str(source),seed=2026102901,local_scale=.2,
    proposal='Use identical frozen per-K=4 component weights/covariances. Local increments .2*L_j*z centered on current state; component chosen independently of current state. Full mixture symmetric; reject support violations once.',
    compared_modes=['independent','local'],no_scale_sweep_retraining_extensions_or_sampling_retries=True,
    action_schedule='25% unavailable birth, 25% unavailable death, 25% one-interface, 15% one-velocity, 10% correlated. No renormalization at fixed K.',
    budget='Each arm four ladders x four temperatures x (2500 warmup +25000 production) =440000 proposals. Actual target costs differ and are reported; compare ESS per target evaluation, not raw acceptance alone.',
    conditional_target='Original LVZ prior conditioned on K=4, same data/physics. Fixed K is deliberate isolation, not a test of original layer probabilities.',
    starts='Four historical fixed-K4 starts, identical in both arms, no truth-based or best-state selection.',
    acceptance='All seven physical coordinates and31depths Rhat<=1.01 and bulk/tail ESS>=400; additionally depth contrasts, LVZ event and structural representation. Forward failures remain separate and prevent full acceptance.',
    source_config_sha256=digest(source/'rj.toml'),training_target_evaluations=371844,
    hashes={n:digest(out/n) for n in ['dispweave','run-script.py','analysis-script.py','truth.toml','observations.txt','covariance.txt','baseline.toml','local.toml']}))

# Short old-versus-new replay with the default (independent) kernel; no selected outcomes.
replay=Path(tempfile.mkdtemp(prefix='dw-local-default-replay-'));short=copy.deepcopy(cfg);short['rjmcmc'].update(warmup=100,draws=200)
write_config(replay/'config.toml',short)
for name,exe in [('old',source/'dispweave'),('new',out/'dispweave')]:
    call=invoke(exe,replay,name,['rjmcmc','--config',replay/'config.toml','--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',replay/name])
    if call['returncode'] not in [0,5]:raise RuntimeError('Default replay failed')
count=0
for path in (replay/'old').rglob('*'):
    if path.is_file() and path.name!='metadata.txt':
        assert path.read_bytes()==(replay/'new'/path.relative_to(replay/'old')).read_bytes(),path;count+=1
save_json(out/'default-replay.json',dict(passed=True,identical_files=count,directory=str(replay),budget='Short regression check, separate from both scientific arms'))
for mode,settings in [('independent',cfg),('local',local)]:
    folder=out/mode;folder.mkdir()
    for name in ['dispweave','truth.toml','observations.txt','covariance.txt']:shutil.copy2(out/name,folder/name)
    write_config(folder/'rj.toml',settings)
    save_json(folder/'protocol.json',dict(mode=mode,hashes={n:digest(folder/n) for n in ['dispweave','truth.toml','observations.txt','covariance.txt','rj.toml']}))
    call=invoke(folder/'dispweave',folder,'rj',['rjmcmc','--config',folder/'rj.toml','--data',folder/'observations.txt','--covariance',folder/'covariance.txt','--output',folder/'rj'])
    if call['returncode'] not in [0,5]:raise RuntimeError('Incomplete arm; no sampling retries')
    analyze(folder)
costs={mode:json.loads((out/mode/'summary.json').read_text())['total_target_evaluations'] for mode in ['independent','local']}
save_json(out/'costs.json',dict(**costs,new_comparison_total=sum(costs.values()),shared_pilot_cost=371844,
    comparison_plus_training=sum(costs.values())+371844,interpretation='No new training. Old pilot charged once; prior exploratory production not part of this matched comparison. Regression/audit work reported separately.'))
