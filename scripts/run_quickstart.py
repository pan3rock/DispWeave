#!/usr/bin/env python3
"""Short, self-contained LM/NUTS/RJMCMC workflow; not statistical acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rj_results import load_run, depth_values

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(out):
    cfg = tomllib.loads((out / 'config.toml').read_text())
    meta = dict(line.split('=', 1) for line in (out / 'lm/metadata.txt').read_text().splitlines())
    best = int(meta['best_run'])
    if best < 0:
        raise ValueError('No converged LM run; inspect lm/runs.txt')
    model = np.loadtxt(out / f'lm/run_{best}_model.txt', ndmin=2)
    depth = np.linspace(0, cfg['rj_prior']['depth_limit'], 501)
    indices = np.searchsorted(cfg['model']['depth'][1:], depth, side='right')
    fixed = np.concatenate([np.loadtxt(p, ndmin=2) for p in sorted((out / 'nuts').glob('chain_*_vs.txt'))])
    _, _, chains = load_run(out / 'rj')
    varying = depth_values(chains, depth).reshape(-1, len(depth))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for samples, label, color in [(fixed[:, indices], 'NUTS', '#0072B2'), (varying, 'RJMCMC', '#7851A9')]:
        lo, mid, hi = np.quantile(samples, [.05, .5, .95], axis=0)
        axes[0].fill_betweenx(depth, lo, hi, color=color, alpha=.10)
        axes[0].plot(mid, depth, color=color, label=label + ' sample median')
    axes[0].plot(np.array(cfg['model']['vs'])[indices], depth, color='#008550', lw=2.5, label='Target', zorder=5)
    axes[0].plot(model[:, 3][indices], depth, '--', color='#D000A0', lw=2, label='LM', zorder=6)
    axes[0].invert_yaxis()
    axes[0].set(xlabel='Vs (km/s)', ylabel='Depth (km)', title='Shading: 5–95% short-run sample range')
    data = np.loadtxt(out / 'lm/observations.txt', ndmin=2)
    axes[1].errorbar(data[:, 0], data[:, 1], yerr=data[:, 3], fmt='ko', label='Noisy data (1 sigma)')
    rows = [shlex.split(line) for line in (out / f'lm/run_{best}_prediction.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
    if any(row[4] != 'valid' for row in rows):
        raise ValueError('Invalid LM prediction')
    axes[1].plot(data[:, 0], [float(row[3]) for row in rows], '--', color='#D000A0', label='LM fit')
    for name, color in [('nuts', '#0072B2'), ('rj', '#7851A9')]:
        pred = out / (name + '_predict')
        statuses = [line.split()[3] for line in (pred / 'draws.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
        if any(s != 'valid' for s in statuses):
            raise ValueError(f'Invalid {name} predictive draw; no failed draws discarded')
        latent = np.loadtxt(pred / 'latent.txt', ndmin=2)
        if not np.isfinite(latent).all():
            raise ValueError('Nonfinite prediction')
        axes[1].plot(data[:, 0], np.median(latent, axis=0), color=color, label=name.upper() + ' prediction median')
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Phase velocity (km/s)', title='Fundamental mode; 8 predictive draws per chain')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('WORKFLOW SMOKE ONLY — no convergence claim\nLM/NUTS: fixed true interface; RJMCMC: variable layers and interfaces', fontsize=11)
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=ROOT / 'build/dispweave')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/quickstart')
    parser.add_argument('--plot-only', action='store_true', help='Read saved results; perform no forward evaluations or sampling')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.plot_only:
        plot(out)
        print(out / 'comparison.png')
        return
    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error(f'Build executable first: {exe}')
    if out.exists():
        parser.error('Output already exists; use a new directory or --plot-only')
    out.mkdir(parents=True)
    cfg = out / 'config.toml'
    shutil.copyfile(ROOT / 'examples/quickstart/synthetic.toml', cfg)
    report = dict(purpose='workflow_smoke_only', statistical_acceptance=False,
                  executable=str(exe), executable_sha256=digest(exe), config_sha256=digest(cfg),
                  budgets=tomllib.loads(cfg.read_text()), commands=[], workflow_complete=False)

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(name, command, allowed=(0,)):
        command = list(map(str, command))
        print(f'{name}: {shlex.join(command)}', flush=True)
        entry = dict(name=name, command=command)
        report['commands'].append(entry)
        start = time.monotonic()
        try:
            with (out / (name + '.log')).open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            entry['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            entry['returncode'] = 'timeout'
            raise
        finally:
            entry['seconds'] = round(time.monotonic() - start, 3)
            save()
        if result.returncode not in allowed:
            raise RuntimeError(f'{name} failed ({result.returncode}); see {out / (name + ".log")}')

    save()
    # synthetic generates data and runs LM; avoid a redundant second inversion.
    run('lm', [exe, 'synthetic', '--config', cfg, '--output', out / 'lm'])
    data_args = ['--data', out / 'lm/observations.txt', '--covariance', out / 'lm/covariance.txt']
    for name, subcommand in [('nuts', 'nuts'), ('rj', 'rjmcmc')]:
        run(name, [exe, subcommand, '--config', cfg, *data_args, '--output', out / name], (0, 5))
        diagnostic = 'posterior_diagnostics.py' if name == 'nuts' else 'rj_diagnostics.py'
        run(name + '_diagnostics', [sys.executable, ROOT / 'scripts' / diagnostic, out / name, '--output', out / (name + '_diagnostics.json')])
        run(name + '_predict', [exe, 'predict' if name == 'nuts' else 'rj-predict', '--run', out / name, '--output', out / (name + '_predict'), '--draws-per-chain', '8', '--seed', '2026'])
    plot(out)
    nuts_diagnostics = json.loads((out / 'nuts_diagnostics.json').read_text())
    rj_diagnostics = json.loads((out / 'rj_diagnostics.json').read_text())
    report['diagnostic_status'] = {
        'nuts_parameters': [p['status'] for p in nuts_diagnostics['parameters']],
        'rj': rj_diagnostics['status'],
    }
    report['input_sha256'] = {p: digest(out / 'lm' / p) for p in ['observations.txt', 'covariance.txt']}
    report['workflow_complete'] = True
    save()
    print(f'Workflow complete; short chains are NOT statistically accepted. See {out / "comparison.png"}')


if __name__ == '__main__':
    main()
