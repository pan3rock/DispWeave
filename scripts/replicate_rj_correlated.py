#!/usr/bin/env python3
"""One seed-only replication of frozen correlated PT. Never fit q or retry sampling."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from benchmark_crust_depth import digest
from benchmark_rj_nuts import invoke
from benchmark_rj_pt_transdim import analyze
from joint_results import read_config, write_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = Path('results/rj-pt-correlated').resolve()
    baseline = Path('results/rj-nuts-comparison').resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original_protocol = json.loads((source/'protocol.json').read_text())
    for name, sha in original_protocol['hashes'].items():
        assert digest(source/name) == sha
    original = read_config(source/'rj.toml')
    assert original['rjmcmc']['seed'] == 20261025
    cfg = copy.deepcopy(original)
    cfg['rjmcmc']['seed'] = 20261026
    for name in ['dispweave', 'observations.txt', 'covariance.txt', 'truth.toml']:
        shutil.copy2(source/name, out/name)
    shutil.copy2(__file__, out/'run-script.py')
    shutil.copy2('scripts/benchmark_rj_pt_transdim.py', out/'analysis-script.py')
    write_config(out/'rj.toml', cfg)
    restored = read_config(out/'rj.toml')
    restored['rjmcmc']['seed'] = original['rjmcmc']['seed']
    assert restored == original
    training = json.loads((source/'training.json').read_text())
    # This hash binds the same mixture to its new seed-only configuration, not a refit.
    training['original_mixture_config_sha256'] = training['mixture_config_sha256']
    training['mixture_config_sha256'] = digest(out/'rj.toml')
    training['replication_note'] = 'No fitting performed; original components copied exactly; only sampler seed changed.'
    save_json(out/'training.json', training)
    save_json(out/'protocol.json', dict(source=str(source), seed=20261026, original_seed=20261025,
        only_config_change='rjmcmc.seed', frozen_binary_sha256=digest(source/'dispweave'),
        source_hashes={n: digest(source/n) for n in ['rj.toml', 'training.json', 'summary.json']},
        total_local_proposals=440000, ladders=4, warmup=2500, draws=25000,
        temperatures=[1, 2, 4, 8], swap_every=10,
        no_retries_tuning_or_extensions=True, refit=False,
        acceptance=original_protocol['acceptance'],
        comparison='Report both runs separately, all retained draws, K indicators and conditional/depth distributions. No selection of the better seed. Unvisited K and forward failures remain unresolved.',
        hashes={n: digest(out/n) for n in ['dispweave', 'rj.toml', 'observations.txt', 'covariance.txt',
                                        'truth.toml', 'run-script.py', 'analysis-script.py', 'training.json']}))
    call = invoke(out/'dispweave', out, 'rj', ['rjmcmc', '--config', out/'rj.toml',
        '--data', out/'observations.txt', '--covariance', out/'covariance.txt', '--output', out/'rj'], 300)
    if call['returncode'] not in [0, 5]:
        raise RuntimeError('Incomplete replication; no sampling retries')
    analyze(out, baseline)
    current = json.loads((out/'summary.json').read_text())
    previous = json.loads((source/'summary.json').read_text())
    pilot = training['old_target_evaluations']
    save_json(out/'costs.json', dict(new_target_evaluations=current['total_target_evaluations'],
        first_run_target_evaluations=previous['total_target_evaluations'], training_target_evaluations=pilot,
        two_runs_plus_shared_training=pilot+previous['total_target_evaluations']+current['total_target_evaluations'],
        interpretation='Pilot charged once across both fresh runs; audit forward calls accounted separately. No previous samples merged into this run.'))


if __name__ == '__main__':
    main()
