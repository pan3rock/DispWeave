#!/usr/bin/env python3
"""One seed-only replication of the frozen 14-point proposal; no refitting."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, save_json

SOURCE=Path('results/rj-mode-refit')

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    previous=json.loads((SOURCE/'protocol.json').read_text())
    for n,h in previous['hashes'].items():assert digest(SOURCE/n)==h
    for n in ['dispweave','observations.txt','covariance.txt','truth.toml']:shutil.copy2(SOURCE/n,out/n)
    shutil.copy2(__file__,out/'run-script.py')
    cfg=read_config(SOURCE/'rj.toml');assert cfg['rjmcmc']['seed']==2026103101
    cfg['rjmcmc']['seed']=2026103102;write_config(out/'rj.toml',cfg)
    check=copy.deepcopy(cfg);check['rjmcmc']['seed']=2026103101;assert check==read_config(SOURCE/'rj.toml')
    design=json.loads((SOURCE/'design.json').read_text())
    design.update(seed=2026103102,replication_source=str(SOURCE.resolve()),replication_original_seed=2026103101,
        changed='Only seed relative to frozen refit run; no refitting or mixture changes.',
        replication_rule='Reuse exact six original improvement thresholds against old-q augmented baseline; compare both new-q runs separately, no sample pooling. Retain configuration if improvement repeats, but do not certify full posterior or precise family probabilities.',
        fresh_training=False,hashes={n:digest(out/n) for n in ['dispweave','observations.txt','covariance.txt','truth.toml','run-script.py']})
    save_json(out/'design.json',design)
    training=json.loads((SOURCE/'training.json').read_text());training['original_mixture_config_sha256']=training['mixture_config_sha256'];training['mixture_config_sha256']=digest(out/'rj.toml');training['replication_note']='Same frozen components copied; no new fit.'
    save_json(out/'training.json',training)
    save_json(out/'protocol.json',dict(total_local_proposals=440000,replication_source=str(SOURCE.resolve()),only_change='rjmcmc.seed',source_hashes={n:digest(SOURCE/n) for n in ['rj.toml','refit-summary.json','training.json','summary.json','independent-audit.json']},
        hashes={n:digest(out/n) for n in ['dispweave','observations.txt','covariance.txt','truth.toml','rj.toml','run-script.py','design.json','training.json']}))
    call=invoke(out/'dispweave',out,'rj',['rjmcmc','--config',out/'rj.toml','--data',out/'observations.txt','--covariance',out/'covariance.txt','--output',out/'rj'],300)
    if call['returncode'] not in [0,5]:raise RuntimeError('Incomplete replication; no retry')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output.resolve())
