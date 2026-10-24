#!/usr/bin/env python3
"""Independent saved-table checks; no new forwards or sampling."""
import json
from pathlib import Path
import sys

import numpy as np

from benchmark_crust_depth import digest, rows
from joint_results import save_json


out = Path(sys.argv[1]).resolve()
report = json.loads((out/'summary.json').read_text())
protocol = json.loads((out/'protocol.json').read_text())
obs = np.loadtxt(out/'observations.txt')
covariance = np.loadtxt(out/'covariance.txt')
inverse = np.linalg.inv(covariance)
max_chi2_error = 0.
for run, source_name in zip(report['runs'], protocol['sources']):
    source = Path(source_name)
    prior_summary = json.loads((source/'summary.json').read_text())
    folder = out/f'predict-{run["seed"]}'
    for directory in [source/'rj', folder, out/'lm']:
        for name in ['observations.txt', 'covariance.txt']:
            # CLI reserializes text (scientific versus default float formatting).
            np.testing.assert_array_equal(np.loadtxt(directory/name), np.loadtxt(out/name))
    chosen = rows(folder/'selected_models.txt')
    trace = rows(folder/'draws.txt')
    latent = np.loadtxt(folder/'latent.txt')
    replicated = np.loadtxt(folder/'replicated.txt')
    failures = sum(t[3] != 'valid' for t in trace)
    assert failures == run['predictive']['failed_count']
    for c in range(4):
        chain = rows(source/f'rj/chain_{c}_models.txt')
        for j in range(100):
            idx = c*100+j
            assert chosen[idx][1:] == chain[j*250][1:]
        # Compare every second grid point to separately computed previous chain quantiles.
        q = np.array(run['per_chain_vs_quantiles'])[c, :, ::2].T
        expected = np.array([d['chain_quantiles'][c] for d in prior_summary['depth_profiles']])
        np.testing.assert_allclose(q, expected, atol=1e-13, rtol=0)
        # Count interfaces explicitly by interval, without histogram or depth indexing helpers.
        interfaces = np.array([float(z) for row in chain for z in row[4:3+int(row[1])]])
        edges = run['interface_edges']
        counts = np.array([np.sum((interfaces >= lo) & (interfaces < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
        np.testing.assert_allclose(counts/len(chain), run['per_chain_interface_occupancy'][c], atol=1e-14, rtol=0)
        assert counts.sum() == sum(int(row[1])-1 for row in chain)
    for i, tr in enumerate(trace):
        if tr[3] != 'valid':
            continue
        observed = latent[i]-obs[:, 1]
        noise = replicated[i]-latent[i]
        chi = float(observed @ inverse @ observed)
        rep_chi = float(noise @ inverse @ noise)
        error = max(abs(chi-float(tr[4])), abs(rep_chi-float(tr[5])))
        max_chi2_error = max(max_chi2_error, error)
        assert error < 1e-8
    if not failures:
        p = sum(float(t[5]) >= float(t[4]) for t in trace)/len(trace)
        assert p == run['predictive']['posterior_predictive_p']
        for name, matrix in [('latent_quantiles',latent), ('replicated_quantiles',replicated)]:
            # Explicit order-statistic interpolation, independent of np.quantile.
            sorted_values = np.sort(matrix, axis=0)
            expected = []
            for p in [.025,.5,.975]:
                index = (len(matrix)-1)*p; lo = int(index); frac = index-lo
                expected.append((1-frac)*sorted_values[lo]+frac*sorted_values[min(lo+1,len(matrix)-1)])
            np.testing.assert_allclose(expected, run['predictive'][name], atol=1e-13, rtol=0)
for name, sha in protocol['source_hashes'].items():
    assert digest(Path(name)) == sha
for name, sha in protocol['hashes'].items():
    assert digest(out/name) == sha
save_json(out/'independent-audit.json', dict(passed=True,selected_models_verified=800,
    data_identical=True, cached_previous_depth_quantiles_verified=True, interface_counts_verified=True,
    predictive_quantiles_verified=True, maximum_chi2_error=max_chi2_error,
    input_hashes_unchanged=True, additional_forward_evaluations=0,
    limitation='Stored prediction consistency only; no claim of held-out calibration, convergence or root completeness.'))
print('Predictive/model-space audit passed; no additional forwards.')
