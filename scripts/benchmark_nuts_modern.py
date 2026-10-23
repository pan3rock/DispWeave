#!/usr/bin/env python3
"""Bounded paired kernel comparison; reuses a validated posterior and frozen metric.

This isolates tree construction, not warmup or convergence from dispersed starts.
Both kernels use the same binary, target, starts, metric and step. No retuning.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
from joint_results import read_config, write_config, save_json
from posterior_diagnostics import diagnose


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=200)
    p.add_argument('--timeout', type=float, default=120)
    args = p.parse_args()
    if args.draws < 8 or args.timeout <= 0:
        p.error('Need >=8 draws and positive timeout')
    exe = args.exe.resolve(); ref = args.reference.resolve(); out = args.output.resolve()
    out.mkdir()
    base = read_config(ref/'nuts.toml')
    metric = np.loadtxt(ref/'nuts/chain_0_mass.txt')
    step = float(np.loadtxt(ref/'nuts/chain_0_trace.txt', dtype=str)[-1, 3])
    starts = [np.loadtxt(ref/f'nuts/chain_{c}_vs.txt')[-1].tolist() for c in range(4)]
    paths = [ref/'nuts.toml', ref/'observations.txt', ref/'covariance.txt',
             ref/'nuts/chain_0_mass.txt', ref/'nuts/chain_0_trace.txt']
    paths += [ref/f'nuts/chain_{c}_vs.txt' for c in range(4)]
    save_json(out/'protocol.json', dict(
        interpretation='Kernel-only pilot from existing posterior states; NOT dispersed-start or warmup validation',
        draws=args.draws, warmup=0, chains=4, workers=2, timeout_seconds_per_chain=args.timeout,
        common_metric='reference chain 0 final mass', common_step=step, seed=20261024,
        binary=str(exe), binary_sha256=digest(exe), script_sha256=digest(Path(__file__)),
        inputs={str(path):digest(path) for path in paths}))
    jobs = []
    # Alternate ordering across chain pairs to reduce ordering/timing effects.
    for c in range(4):
        for kernel in (['slice', 'multinomial'] if c % 2 == 0 else ['multinomial', 'slice']):
            cfg = copy.deepcopy(base)
            cfg['inversion']['starts_vs'] = [starts[c]]
            opt = cfg['nuts']; opt.pop('initial_metric', None)
            opt.update(kernel=kernel, find_initial_step_size=False, warmup=0, draws=args.draws,
                       adapt_step_size=False, metric_adaptation='none', mass=metric.tolist(),
                       step_size=step, chain_id_offset=c, seed=20261024)
            config = out/f'{kernel}-{c}.toml'; write_config(config, cfg)
            jobs.append((kernel, c, config))

    def run(job):
        kernel, c, config = job; dest = out/f'{kernel}-{c}'
        command = [str(exe), 'nuts', '--config', str(config), '--data', str(ref/'observations.txt'),
                   '--covariance', str(ref/'covariance.txt'), '--output', str(dest)]
        start = time.perf_counter()
        with (out/f'{kernel}-{c}.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        record = dict(kernel=kernel, chain=c, seconds=time.perf_counter()-start, returncode=code)
        if code in (0, 5) and (dest/'chain_0_vs.txt').exists():
            trace = np.loadtxt(dest/'chain_0_trace.txt', dtype=str, ndmin=2)
            samples = np.loadtxt(dest/'chain_0_vs.txt', ndmin=2)
            assert trace.shape == (args.draws, 10) and samples.shape == (args.draws, 45)
            costs = dict(line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines())
            record.update(complete=True, transition_evaluations=int(trace[:, 7].astype(int).sum()),
                          total_evaluations=sum(int(costs[k]) for k in
                              ['target_evaluations', 'preflight_evaluations', 'metric_evaluations']),
                          divergences=int(trace[:, 2].astype(int).sum()),
                          depth_caps=int(trace[:, 9].astype(int).sum()),
                          invalid=int(np.count_nonzero(~np.isin(trace[:, 6], ['valid', 'outside_support']))),
                          support_rejections=int(np.count_nonzero(trace[:, 6] == 'outside_support')))
        else:
            record['complete'] = False
        save_json(out/f'{kernel}-{c}-cost.json', record)
        print(record, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run, jobs))
    summary = dict(chains=records, comparison={})
    reference = np.stack([np.loadtxt(ref/f'nuts/chain_{c}_vs.txt') for c in range(4)])
    reference_stats = [diagnose(reference[:, :, j]) for j in range(45)]
    for kernel in ['slice', 'multinomial']:
        costs = [r for r in records if r['kernel'] == kernel]
        if not all(r['complete'] for r in costs):
            summary['comparison'][kernel] = dict(complete=False, reason='All declared chains required')
            continue
        x = np.stack([np.loadtxt(out/f'{kernel}-{c}/chain_0_vs.txt') for c in range(4)])
        stats = [diagnose(x[:, :, j]) for j in range(45)]
        seconds = sum(r['seconds'] for r in costs)
        evaluations = sum(r['total_evaluations'] for r in costs)
        ess = min(s['ess_bulk'] for s in stats)
        summary['comparison'][kernel] = dict(complete=True, parameters=stats,
            max_rhat=max(s['rhat'] for s in stats), min_bulk_ess=ess,
            min_tail_ess=min(s['ess_tail'] for s in stats), summed_chain_seconds=seconds,
            total_evaluations=evaluations, min_bulk_ess_per_chain_second=ess/seconds,
            min_bulk_ess_per_1000_evaluations=ess/evaluations*1000,
            max_mean_difference_in_pooled_mcse=max(abs(s['mean']-r['mean']) /
                np.hypot(s['mcse_mean'], r['mcse_mean']) for s, r in zip(stats, reference_stats)))
    save_json(out/'summary.json', summary)


if __name__ == '__main__':
    main()
