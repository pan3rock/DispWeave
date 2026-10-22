#!/usr/bin/env python3
"""Repeated halfspace inversion, independent Rayleigh cubic truth and quadrature posterior.
Truths follow uniform physical Vs. Only the matching prior has prior-predictive
95% coverage as a calibration target; the log-Vs prior is a sensitivity experiment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from posterior_diagnostics import diagnose


def quadrature(observed, sigma, factor, prior, size):
    x = np.linspace(1., 3., size)
    density = np.exp(-.5*((factor*x-observed)/sigma)**2)
    if prior == 'uniform_log_vs':
        density /= x
    area = .5*(density[1:]+density[:-1])*np.diff(x)
    cdf = np.r_[0., np.cumsum(area)]
    normalization = cdf[-1]
    cdf /= normalization
    mean = np.sum(.5*(x[1:]*density[1:]+x[:-1]*density[:-1])*np.diff(x))/normalization
    return float(mean), np.interp([.025, .5, .975], cdf, x)


def wilson(successes, n):
    z = 1.959963984540054
    center = (successes/n + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(successes/n*(1-successes/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--replicates', type=int, default=40)
    parser.add_argument('--draws', type=int, default=8000)
    args = parser.parse_args()
    if args.draws < 400 or args.draws % 40:
        raise ValueError('Draw count must be >=400 and divisible by 40')
    if args.replicates < 2:
        raise ValueError('Need at least two replicates')
    args.output.mkdir(parents=True, exist_ok=False)
    exe = args.executable.resolve()
    beta = 1/1.8**2
    roots = np.roots([1., -8., 24-16*beta, -16*(1-beta)])
    physical = [r.real for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1]
    if len(physical) != 1:
        raise ValueError('Ambiguous independent Rayleigh cubic root')
    factor = np.sqrt(physical[0])
    rng = np.random.default_rng(20261022)
    records = []
    for replicate in range(args.replicates):
        truth = float(rng.uniform(1., 3.))
        sigma = .2 if replicate % 2 == 0 else .5
        observed = float(factor*truth + rng.normal(0, sigma))
        for prior in ('uniform_vs', 'uniform_log_vs'):
            folder = args.output / f'replicate_{replicate:03d}_{prior}'
            folder.mkdir()
            config = folder / 'input.toml'
            config.write_text(f'''[model]
depth = [0.0]
vs = [2.0]
vp = [3.6]
rho = [2.5]
[inversion]
lower_vs = [1.0]
upper_vs = [3.0]
starts_vs = [[1.4], [2.6]]
vp_vs_ratio = 1.8
sensitivity = "implicit"
[nuts]
prior = "{prior}"
warmup = 800
draws = {args.draws}
target_accept = 0.95
metric_adaptation = "diagonal"
max_depth = 10
seed = {2026+replicate}
''')
            data = folder / 'observations.txt'
            data.write_text(f'# frequency_hz phase_km_s mode sigma_km_s\n0.5 {observed:.17g} 0 {sigma}\n')
            run = folder / 'run'
            result = subprocess.run([str(exe), 'nuts', '--config', str(config), '--data', str(data), '--output', str(run)], capture_output=True, text=True)
            (folder / 'stdout.txt').write_text(result.stdout + result.stderr)
            if result.returncode not in (0, 5):
                raise RuntimeError(result.stderr)
            chains = np.stack([np.loadtxt(run / f'chain_{i}_vs.txt') for i in range(2)])
            summary = diagnose(chains)
            exact_mean, exact_quantiles = quadrature(observed, sigma, factor, prior, 40001)
            coarse_mean, coarse_quantiles = quadrature(observed, sigma, factor, prior, 20001)
            if abs(exact_mean-coarse_mean) > 1e-6 or np.max(np.abs(exact_quantiles-coarse_quantiles)) > 1e-6:
                raise RuntimeError('Independent quadrature not stable')
            gate = max(.04*summary['sd'], 6*summary['mcse_mean'])
            passed = abs(summary['mean']-exact_mean) <= gate and summary['mcse_mean'] < .1*summary['sd']
            probability_checks = []
            for quantile, probability in zip(exact_quantiles, [.025, .5, .975]):
                batches = (chains <= quantile).reshape(2, 40, args.draws//40).mean(axis=2)
                mcse = float(np.sqrt(batches.var(axis=1, ddof=1).sum()/40)/2)
                estimate = float(batches.mean())
                probability_checks.append(dict(expected=probability, observed=estimate, mcse=mcse,
                                               passed=abs(estimate-probability) <= max(.005, 6*mcse)))
            lo, _, hi = summary['quantiles']
            records.append(dict(replicate=replicate, prior=prior, truth=truth, observed=observed, sigma=sigma,
                                diagnostics=summary, sampler_issues=result.returncode == 5,
                                probability_checks=probability_checks, exact_mean=exact_mean, exact_quantiles=exact_quantiles.tolist(), mean_gate_pass=passed,
                                covered=bool(lo <= truth <= hi), exact_covered=bool(exact_quantiles[0] <= truth <= exact_quantiles[2])))
        print(f'replicate {replicate+1}/{args.replicates}', flush=True)
    aggregate = {}
    for prior in ('uniform_vs', 'uniform_log_vs'):
        rows = [r for r in records if r['prior'] == prior]
        count = sum(r['covered'] for r in rows)
        aggregate[prior] = dict(covered=count, replicates=len(rows), coverage=count/len(rows),
                               coverage_wilson95=wilson(count, len(rows)),
                               exact_covered=sum(r['exact_covered'] for r in rows),
                               mean_gate_failures=sum(not r['mean_gate_pass'] for r in rows),
                               probability_gate_failures=sum(not c['passed'] for r in rows for c in r['probability_checks']),
                               sampler_issue_runs=sum(r['sampler_issues'] for r in rows),
                               diagnostic_warning_runs=sum(r['diagnostics']['status'] != 'no_flag' for r in rows),
                               maximum_quantile_error=max(float(np.max(np.abs(np.array(r['diagnostics']['quantiles'])-r['exact_quantiles']))) for r in rows))
    shifts = [records[i+1]['diagnostics']['mean']-records[i]['diagnostics']['mean'] for i in range(0, len(records), 2)]
    report = dict(design='Prior-predictive uniform physical Vs in [1,3], alternating sigma .2/.5, independent halfspace truth; two priors on identical observations',
                  seed=20261022, draws_per_chain=args.draws, rayleigh_factor=float(factor), aggregate=aggregate,
                  mean_prior_shift=float(np.mean(shifts)), maximum_absolute_prior_shift=float(np.max(np.abs(shifts))), records=records)
    with (args.output / 'analysis.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(aggregate, indent=2))
    if any(x['mean_gate_failures'] or x['probability_gate_failures'] for x in aggregate.values()):
        raise SystemExit('Some means or cumulative probabilities failed independent quadrature comparison; all runs retained')


if __name__ == '__main__':
    main()
