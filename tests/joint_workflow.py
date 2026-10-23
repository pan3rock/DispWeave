"""CLI/data/PPC contracts; deliberately short chains are not convergence evidence."""
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import numpy as np

exe, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(root/'scripts'))
from joint_results import summarize, write_config, read_config


def replace_array(text, key, value):
    start = text.index(key+' =')
    left = text.index('[', start)
    level = 0
    for end in range(left, len(text)):
        level += (text[end] == '[')-(text[end] == ']')
        if level == 0:
            return text[:start]+key+' = '+json.dumps(value)+text[end+1:]
    raise ValueError('Unterminated array')


def run(command, config, output, expected=0):
    process = subprocess.run([str(exe), command, '--config', str(config), '--output', str(output)], capture_output=True, text=True)
    assert process.returncode == expected, (process.returncode, process.stdout, process.stderr)


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    text = (root/'examples/joint/synthetic.toml').read_text()
    for key, value in [('chains', 2), ('warmup', 80), ('draws', 80), ('predictive_per_chain', 8), ('samples', 256)]:
        text = re.sub(rf'(?m)^{key}\s*=.*$', f'{key} = {value}', text)
    text = text.replace('enabled = true', 'enabled = false')
    text = replace_array(text, 'initial_vs', [[2.4, 3.5], [2.8, 3.8]])
    # A short processing window keeps this I/O test fast. M8b and the science
    # experiments, not this test, establish waveform/window convergence.
    start = text.index('name = "receiver"')
    n = len(tomllib.loads(text)['observations'][-1]['sample_indices'])
    covariance = .008**2*.4**np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    text = text[:start]+re.sub(r'(?m)^sigma =.*$', 'covariance = '+json.dumps(covariance.tolist()), text[start:])
    source = work/'source.toml';source.write_text(text)
    data, again = work/'data', work/'data-repeat'
    run('joint-data', source, data);run('joint-data', source, again)
    assert (data/'observations.toml').read_bytes() == (again/'observations.toml').read_bytes()
    assert (data/'config.toml').read_bytes() == source.read_bytes()
    config = data/'observations.toml'
    document = tomllib.loads(config.read_text())
    assert [len(b['values']) for b in document['observations']] == [8, 6, 27]
    first, second = work/'inference', work/'replay'
    run('joint', config, first);run('joint', config, second)
    assert 'target_evaluations_including_warmup=322\n' in (first/'metadata.txt').read_text()
    assert 'failed_target_evaluations_including_warmup=0\n' in (first/'metadata.txt').read_text()
    assert len((first/'failed_evaluations.txt').read_text().splitlines()) == 1
    for chain in range(2):
        stem = f'chain_{chain}'
        assert (first/f'{stem}_state.txt').read_bytes() == (second/f'{stem}_state.txt').read_bytes()
        state = np.loadtxt(first/f'{stem}_state.txt', ndmin=2)
        assert state.shape == (80, 5) and np.isfinite(state).all()
        np.testing.assert_allclose(np.exp(state[:, :2]), np.loadtxt(first/f'{stem}_vs.txt', ndmin=2), atol=1e-14)
        trace = np.loadtxt(first/f'{stem}_trace.txt', dtype=str, ndmin=2, skiprows=1)
        assert trace.shape == (160, 5)
        assert (trace[:80, 0] == '1').all() and (trace[80:, 0] == '0').all()
        assert np.ptp(trace[80:, 3].astype(float)) == 0
    states = [line.split()[:3] for line in (first/'predictive_status.txt').read_text().splitlines() if not line.startswith('#')]
    assert len(states) == 16 and all(row[2] == 'valid' for row in states)
    assert len({tuple(row[:2]) for row in states}) == 16
    for block in document['observations']:
        name = block['name']
        latent = np.loadtxt(first/f'prediction_{name}.txt')
        replicated = np.loadtxt(first/f'replicate_{name}.txt')
        misfit = np.loadtxt(first/f'predictive_misfit_{name}.txt')
        assert latent.shape == (16, 2+len(block['values']))
        assert np.array_equal(latent[:, :2], replicated[:, :2]) and np.array_equal(latent[:, :2], misfit[:, :2])
        cov = np.loadtxt(first/f'covariance_{name}.txt', ndmin=2)
        factor = np.linalg.cholesky(cov)
        observed = np.linalg.solve(factor, (latent[:, 2:]-block['values']).T).T/misfit[:, 4, None]
        repeat = np.linalg.solve(factor, (replicated[:, 2:]-latent[:, 2:]).T).T/misfit[:, 4, None]
        np.testing.assert_allclose(np.sum(observed**2, axis=1), misfit[:, 2], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(np.sum(repeat**2, axis=1), misfit[:, 3], rtol=1e-11, atol=1e-11)
    summary = summarize(first)
    assert len(summary['parameters']) == 5 and len(summary['blocks']) == 3
    assert summary['numerical_proposals_including_warmup'] == 0
    assert all(c['retained']['transitions'] == 80 for c in summary['sampler'])
    round_trip = work/'round-trip.toml';write_config(round_trip, document)
    assert read_config(round_trip) == document
    # A missing prediction or explicit failure must invalidate analysis, not
    # produce tighter intervals by silently removing the row.
    status_file = first/'predictive_status.txt';original_status = status_file.read_text()
    for bad in ['\n'.join(original_status.splitlines()[:-1])+'\n', original_status.replace('valid', 'forward_failure', 1)]:
        status_file.write_text(bad)
        try:
            summarize(first)
        except ValueError:
            pass
        else:
            raise AssertionError('Analysis silently accepted incomplete/failed predictions')
    status_file.write_text(original_status)
    run('joint-data', source, data, 1)
    # Presence of a synthetic table must not cause the inference command to
    # regenerate or substitute missing observed values.
    run('joint', source, work/'missing-values', 1)
    assert not (work/'missing-values').exists()
    for name, bad in [('dependence', text.replace('independent_blocks = true', 'independent_blocks = false')),
                      ('hv', text.replace('rayleigh_modal_ellipticity', 'ambient_hv')),
                      ('units', text.replace('unit = "km/s"', 'unit = "m/s"'))]:
        source.write_text(bad);run('joint-data', source, work/name, 1)
        assert not (work/name).exists()
print('Joint CLI data replay, posterior/PPC dimensions, covariance/noise scaling and failure contracts passed')
