#!/usr/bin/env python3
"""Fixed-truth layered repeated-noise coverage and paired prior sensitivity.
Uses the validated invLM forward for synthesis AND inference: a consistency
experiment, not independent validation of the physical model or global coverage.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from posterior_diagnostics import diagnose
from nuts_coverage import wilson


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--replicates', type=int, default=12)
    a = p.parse_args()
    if a.replicates < 2:
        raise ValueError('Need at least two replicates')
    a.output.mkdir(parents=True, exist_ok=False)
    executable = a.executable.resolve()
    template = (Path(__file__).resolve().parents[1] / 'examples/nuts/synthetic.toml').read_text()
    truth = np.array([1., 2., 3.])
    records = []

    def run(folder, *args):
        result = subprocess.run([str(executable), *map(str, args)], capture_output=True, text=True)
        (folder / 'stdout.txt').write_text(result.stdout + result.stderr)
        return result.returncode

    for replicate in range(a.replicates):
        folder = a.output / f'replicate_{replicate:03d}'
        folder.mkdir()
        sigma = .02 if replicate % 2 == 0 else .05
        content = template.replace('seed = 2026', f'seed = {87000+replicate}').replace('sigma_km_s = 0.02', f'sigma_km_s = {sigma}')
        config = folder / 'synthetic.toml'
        config.write_text(content)
        # Synthetic also computes an LM fit. A stalled LM does not erase valid data.
        code = run(folder, 'synthetic', '--config', config, '--output', folder / 'data')
        if code not in (0, 4):
            raise RuntimeError(f'Synthesis failed, code {code}, {folder}')
        if not (folder / 'data/observations.txt').exists() or not (folder / 'data/covariance.txt').exists():
            raise RuntimeError(f'Synthesis failed, code {code}, {folder}')
        for prior in ('uniform_vs', 'uniform_log_vs'):
            job = folder / prior
            job.mkdir()
            config = job / 'config.toml'
            config.write_text(content.replace('prior = "uniform_vs"', f'prior = "{prior}"'))
            code = run(job, 'nuts', '--config', config, '--data', folder / 'data/observations.txt',
                       '--covariance', folder / 'data/covariance.txt', '--output', job / 'run')
            if code not in (0, 5):
                raise RuntimeError(f'Sampling failed, code {code}, {job}')
            chains = np.stack([np.loadtxt(job / f'run/chain_{i}_vs.txt') for i in range(3)])
            diagnostics = [diagnose(chains[:, :, j]) for j in range(3)]
            counts = np.loadtxt(job / 'run/chains.txt')
            records.append(dict(replicate=replicate, sigma=sigma, prior=prior, diagnostics=diagnostics,
                                covered=[bool(d['quantiles'][0] <= truth[j] <= d['quantiles'][2]) for j, d in enumerate(diagnostics)],
                                divergences=int(counts[:, 3].sum()), invalid=int(counts[:, 4].sum()),
                                depth_limits=int(counts[:, 6].sum()), sampler_issues=code == 5))
            # Persist every finished job, including warnings, even if a later run fails.
            (a.output / 'records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
            print(f'replicate {replicate+1}/{a.replicates} {prior}: {[d["status"] for d in diagnostics]}', flush=True)
    aggregate = []
    for prior in ('uniform_vs', 'uniform_log_vs'):
        for sigma in (.02, .05):
            rows = [r for r in records if r['prior'] == prior and r['sigma'] == sigma]
            count = np.sum([r['covered'] for r in rows], axis=0).astype(int)
            aggregate.append(dict(prior=prior, sigma=sigma, replicates=len(rows), covered=count.tolist(),
                                  wilson95=[wilson(int(n), len(rows)) for n in count],
                                  warning_runs=sum(any(d['status'] != 'no_flag' for d in r['diagnostics']) for r in rows),
                                  divergences=sum(r['divergences'] for r in rows),
                                  invalid=sum(r['invalid'] for r in rows), depth_limits=sum(r['depth_limits'] for r in rows)))
    shifts = np.array([[records[i+1]['diagnostics'][j]['mean']-records[i]['diagnostics'][j]['mean'] for j in range(3)] for i in range(0, len(records), 2)])
    report = dict(design=__doc__, truth=truth.tolist(), aggregate=aggregate,
                  mean_prior_shift=shifts.mean(axis=0).tolist(), maximum_absolute_prior_shift=np.abs(shifts).max(axis=0).tolist(), records=records)
    (a.output / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
