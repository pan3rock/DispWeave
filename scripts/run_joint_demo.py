#!/usr/bin/env python3
"""M8d reproducible single/joint, conflicting-data and prior sensitivity suite.

Python >=3.11, NumPy; plots additionally require Matplotlib. Outputs are local.
Each run directory is immutable. Failed CLI calls stop the suite and keep logs.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import numpy as np
from joint_results import read_config, write_config, save_json, summarize


def run(exe, command, config, output):
    with output.with_suffix('.log').open('x') as log:
        p = subprocess.run([str(exe), command, '--config', str(config), '--output', str(output)],
                           stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'{command} returned {p.returncode}; see {output.with_suffix(".log")}')


def check_receiver(exe, doc, work):
    """Use observed -1..12 s; all prior corners plus generating models."""
    work.mkdir()
    b = next(b for b in doc['observations'] if b['observable'] == 'p_receiver_function')
    processing = b['processing']
    lower, upper = doc['parameters']['lower_vs'], doc['parameters']['upper_vs']
    points = [doc['synthetic']['vs'], [2.9, 3.6]] + [[a, z] for a in [lower[0], upper[0]] for z in [lower[1], upper[1]]]
    results = []
    for j, vs in enumerate(points):
        waves = []
        for tag, count, dt in [('base', 1024, .1), ('long', 2048, .1), ('fine', 2048, .05)]:
            cfg = dict(model=deepcopy(doc['model']), receiver=deepcopy(processing))
            cfg['model']['vs'] = vs
            cfg['receiver'].update(samples=count, dt=dt)
            path = work/f'{j}-{tag}.toml';write_config(path, cfg)
            output = work/f'{j}-{tag}';run(exe, 'receiver', path, output)
            wave = np.loadtxt(output/'receiver.txt', usecols=(0, 1))
            ids = np.array(b['sample_indices'])*(2 if tag == 'fine' else 1)
            waves.append(wave[ids, 1])
        errors = dict(vs=vs, window_error=float(np.max(np.abs(waves[0]-waves[1]))),
                      sampling_error=float(np.max(np.abs(waves[0]-waves[2]))))
        results.append(errors)
    save_json(work/'summary.json', results)
    if max(max(r['window_error'], r['sampling_error']) for r in results) > 1e-5:
        raise RuntimeError('RF discretization audit exceeded 1e-5; inspect audit before inference')
    return results


def prepare(exe, source, output, warmup, draws):
    output.mkdir()
    doc = read_config(source)
    doc['sampler'].update(warmup=warmup, draws=draws)
    receiver = doc['observations'][-1]
    times = np.array(receiver['sample_indices'])*receiver['processing']['dt']
    sigma = np.array(receiver.pop('sigma'))
    # Stated synthetic covariance, not estimated from these simulated residuals.
    receiver['covariance'] = (sigma[:, None]*sigma[None, :]*np.exp(-np.abs(times[:, None]-times[None, :])/.35)).tolist()
    check_receiver(exe, doc, output/'receiver-audit')
    source = output/'consistent-source.toml';write_config(source, doc)
    run(exe, 'joint-data', source, output/'consistent-data')
    conflict = deepcopy(doc)
    conflict['observations'][-1]['synthetic_vs'] = [2.9, 3.6]
    path = output/'conflict-source.toml';write_config(path, conflict)
    run(exe, 'joint-data', path, output/'conflict-data')
    consistent = read_config(output/'consistent-data/observations.toml')
    conflicting = read_config(output/'conflict-data/observations.toml')
    # Same seed/order means conflict changes only RF signal, not residual noise.
    for a, b in zip(consistent['observations'][:2], conflicting['observations'][:2]):
        if a['values'] != b['values']:
            raise ValueError('Conflict unexpectedly changed non-RF data')
    cases = []

    def case(name, base, select=None, log_vs=False, noise_sd=None, fixed=False):
        cfg = deepcopy(base)
        if select:
            cfg['observations'] = [b for b in cfg['observations'] if b['name'] in select]
        if log_vs:
            cfg['prior']['measure'] = 'log_vs'
        for block in cfg['observations']:
            if noise_sd is not None:
                block['noise']['sd_log'] = noise_sd
            if fixed:
                block['noise'] = dict(kind='fixed', scale=1.)
        cfg['sampler']['proposal_sd'] = [.02]*len(cfg['model']['vs']) + [.15]*sum(b['noise']['kind'] != 'fixed' for b in cfg['observations'])
        path = output/f'{name}.toml';write_config(path, cfg)
        cases.append(name)

    for name in ['dispersion', 'ellipticity', 'receiver']:
        case(name, consistent, [name])
    case('joint', consistent)
    case('joint-log-prior', consistent, log_vs=True)
    case('conflict', conflicting)
    case('conflict-receiver', conflicting, ['receiver'])
    case('conflict-wide-noise', conflicting, noise_sd=1.)
    case('conflict-fixed-noise', conflicting, fixed=True)
    save_json(output/'manifest.json', dict(cases=cases, warmup=warmup, draws=draws,
              interpretation='Same consistent observations reused for all subsets; conflict changes only RF generating Vs. Fixed interfaces/Vp/density.'))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=Path('build/dispweave'))
    parser.add_argument('--config', type=Path, default=Path('examples/joint/synthetic.toml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=4000)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true', help='Run every case in an existing prepared manifest')
    parser.add_argument('--jobs', type=int, default=1, help='Independent CLI processes; each chain keeps its own seed')
    parser.add_argument('--case', help='Run one previously prepared case; does not regenerate data')
    args = parser.parse_args()
    exe = args.exe.resolve()
    if args.jobs < 1 or (args.case and args.run_prepared):
        parser.error('Invalid jobs or mutually exclusive case/run-prepared')
    if args.case or args.run_prepared:
        import json
        manifest = json.loads((args.output/'manifest.json').read_text())
        if args.case and args.case not in manifest['cases']:
            parser.error('Unknown prepared case')
        cases = [args.case] if args.case else manifest['cases']
    else:
        cases = prepare(exe, args.config, args.output, args.warmup, args.draws)
    if args.prepare_only:
        return
    def execute(name):
        print(f'Running {name}', flush=True)
        run(exe, 'joint', args.output/f'{name}.toml', args.output/name)
        summary = summarize(args.output/name)
        save_json(args.output/name/'diagnostics.json', summary)
        print(name, {k: (round(v['rhat'], 4), round(v['ess_bulk']), v['status']) for k, v in summary['parameters'].items()}, flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(execute, cases))


if __name__ == '__main__':
    main()
