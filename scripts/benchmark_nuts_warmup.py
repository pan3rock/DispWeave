#!/usr/bin/env python3
"""Fixed-budget warmup ablation from two original crustal initial states.

Keep timeouts and partial progress. Do not label warmup states as posterior
draws, or infer a frozen step size from the last adaptive transition.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from joint_results import read_config, write_config, save_json


VARIANTS = [('slice', 'slice', False), ('slice-search', 'slice', True),
            ('multinomial-search', 'multinomial', True)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress_rows(path, dimensions):
    if not path.exists():
        return [], 0
    rows, partial = [], 0
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        row = line.split()
        if len(row) != 8 + dimensions:
            partial += 1
            continue
        assert int(row[0]) == len(rows), 'Nonsequential progress'
        rows.append(row)
    return rows, partial


def phase_stats(rows):
    if not rows:
        return dict(transitions=0)
    evaluations = [int(r[5]) for r in rows]
    return dict(transitions=len(rows), completed_transition_evaluations=sum(evaluations),
                mean_evaluations=float(np.mean(evaluations)),
                median_evaluations=float(np.median(evaluations)),
                mean_acceptance=float(np.mean([float(r[7]) for r in rows])),
                divergences=sum(int(r[3]) for r in rows),
                invalid_transitions=sum(r[2] not in ('valid', 'outside_support') for r in rows),
                support_rejections=sum(r[2] == 'outside_support' for r in rows),
                last_step_used=float(rows[-1][6]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--draws', type=int, default=100)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--metric-shrinkage', type=float, default=0,
                   help='Pseudo-count retaining previous position covariance; zero is the original experiment')
    p.add_argument('--modern-only', action='store_true', help='Run only multinomial plus step search, both original starts')
    args = p.parse_args()
    if args.warmup < 150 or args.draws < 1 or not 0 < args.timeout <= 120:
        p.error('Need warmup >=150, draws >=1, timeout in (0,120] seconds')
    if not np.isfinite(args.metric_shrinkage) or args.metric_shrinkage < 0:
        p.error('metric-shrinkage must be finite and nonnegative')
    ref = args.reference.resolve(); out = args.output.resolve(); out.mkdir()
    base = read_config(ref/'nuts.toml')
    assert len(base['model']['vs']) == 45 and len(base['inversion']['starts_vs']) == 4
    assert base['nuts']['initial_metric'] == 'start_gauss_newton'
    assert base['nuts']['metric_adaptation'] == 'dense' and base['nuts']['target_accept'] == .99
    assert np.loadtxt(ref/'observations.txt').shape == (125, 4)
    inputs = out/'inputs'; inputs.mkdir()
    for name in ['nuts.toml', 'observations.txt', 'covariance.txt']:
        shutil.copyfile(ref/name, inputs/name)
    exe = out/'dispweave'; shutil.copy2(args.exe.resolve(), exe)
    shutil.copyfile(Path(__file__), out/Path(__file__).name)
    jobs = []
    variants = VARIANTS[-1:] if args.modern_only else VARIANTS
    for label, kernel, search in variants:
        for chain in [0, 1]:
            cfg = copy.deepcopy(base)
            cfg['inversion']['starts_vs'] = [base['inversion']['starts_vs'][chain]]
            cfg['nuts'].update(kernel=kernel, find_initial_step_size=search,
                               warmup=args.warmup, draws=args.draws, chain_id_offset=chain,
                               record_progress=True, record_failures=True)
            if args.metric_shrinkage:
                cfg['nuts']['metric_shrinkage'] = args.metric_shrinkage
            path = out/f'{label}-{chain}.toml'; write_config(path, cfg)
            jobs.append((label, chain, path))
    protocol = dict(interpretation='Warmup ablation pilot, not a new posterior acceptance run',
        original_chains={'0': 'previous normal chain, reference Vs minus 0.2',
                         '1': 'previous slow chain, reference Vs plus 0.2'},
        warmup=args.warmup, draws=args.draws, timeout_seconds_per_chain=args.timeout, workers=2,
        seed=base['nuts']['seed'], target_accept=base['nuts']['target_accept'],
        initial_step=base['nuts']['step_size'], initial_metric=base['nuts']['initial_metric'],
        metric_adaptation=base['nuts']['metric_adaptation'],
        variants=[dict(label=a, kernel=b, step_search=c) for a, b, c in variants],
        metric_shrinkage=args.metric_shrinkage,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        binary_sha256=digest(exe), script_sha256=digest(Path(__file__)),
        inputs={name:digest(inputs/name) for name in ['nuts.toml', 'observations.txt', 'covariance.txt']},
        configs={path.name:digest(path) for _, _, path in jobs},
        timeout_accounting='Completed-transition evaluation sums are lower bounds: exclude searches, initial calls, and interrupted transition',
        incomplete_policy='Report all declared jobs; no retries, budget extensions, pooled ESS or posterior acceptance')
    save_json(out/'protocol.json', protocol)

    def run(job):
        label, chain, config = job; dest = out/f'{label}-{chain}'
        command = [str(exe), 'nuts', '--config', str(config), '--data', str(inputs/'observations.txt'),
                   '--covariance', str(inputs/'covariance.txt'), '--output', str(dest)]
        start = time.perf_counter()
        with (out/f'{label}-{chain}.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        elapsed = time.perf_counter()-start
        rows, partial = progress_rows(dest/'chain_0_progress.txt', 45)
        warmup = [r for r in rows if r[1] == '1']; retained = [r for r in rows if r[1] == '0']
        assert len(warmup) == min(len(rows), args.warmup)
        record = dict(variant=label, original_chain=chain, returncode=code, seconds=elapsed,
            complete=code in (0, 5) and len(rows) == args.warmup+args.draws,
            warmup=phase_stats(warmup), production=phase_stats(retained),
            last_25_warmup=phase_stats(warmup[-25:]), partial_progress_lines=partial,
            completed_transition_evaluations_lower_bound=sum(int(r[5]) for r in rows),
            frozen_step=None, exact_total_evaluations=None, step_search_evaluations=None)
        if retained:
            steps = [float(r[6]) for r in retained]
            assert len(set(steps)) == 1, 'Production step changed'
            record['frozen_step'] = steps[0]
        if record['complete']:
            costs = {k:int(v) for k, v in (line.split('=') for line in
                     (dest/'chain_0_cost.txt').read_text().splitlines())}
            assert costs['target_evaluations'] == 1+costs['transition_evaluations']+costs['step_search_evaluations']
            assert costs['transition_evaluations'] == record['completed_transition_evaluations_lower_bound']
            record['exact_total_evaluations'] = costs['target_evaluations']+costs['preflight_evaluations']+costs['metric_evaluations']
            record['step_search_evaluations'] = costs['step_search_evaluations']
            samples = np.loadtxt(dest/'chain_0_vs.txt', ndmin=2)
            assert samples.shape == (args.draws, 45) and np.isfinite(samples).all()
        if label == 'slice' and args.warmup == 300 and args.metric_shrinkage == 0:
            old, _ = progress_rows(ref/f'nuts/chain_{chain}_progress.txt', 45)
            count = min(250, len(rows))
            # The 300 and 1000 warmup schedules share their first three windows.
            assert rows[:count] == old[:count], 'Legacy warmup prefix differs from accepted reference run'
            record['original_run_identical_prefix_transitions'] = count
        save_json(out/f'{label}-{chain}-summary.json', record)
        print(record, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run, jobs))
    assert len(records) == 2*len(variants)
    save_json(out/'summary.json', dict(protocol=protocol, chains=records,
        complete_chains=sum(r['complete'] for r in records), accepted_as_new_posterior=False))


if __name__ == '__main__':
    main()
