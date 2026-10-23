"""CLI contract checks; distribution correctness is tested independently in C++."""
import pathlib
import importlib.util
import json
import subprocess
import sys
import tempfile

exe = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2])
algorithm = sys.argv[3] if len(sys.argv) > 3 else 'hmc'

def run(*args, codes=(0,)):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode in codes, (p.returncode, p.stdout, p.stderr)
    return p

with tempfile.TemporaryDirectory() as tmp:
    work = pathlib.Path(tmp)
    config = work / 'input.toml'
    content = (root / f'examples/{algorithm}/synthetic.toml').read_text().replace('warmup = 1000', 'warmup = 100').replace('draws = 4000', 'draws = 80')
    config.write_text(content)
    data = work / 'data'
    run('synthetic', '--config', config, '--output', data)
    def sample(output, cfg=config, observations=data / 'observations.txt', codes=(0, 5)):
        return run(algorithm, '--config', cfg, '--data', observations, '--covariance', data / 'covariance.txt', '--output', output, codes=codes)
    a, b = work / 'a', work / 'b'
    sample(a)
    sample(b)
    for chain in range(3):
        for suffix in ('vs', 'z', 'trace'):
            name = f'chain_{chain}_{suffix}.txt'
            assert (a / name).read_bytes() == (b / name).read_bytes(), 'same seed must replay'
        rows = [l.split() for l in (a / f'chain_{chain}_vs.txt').read_text().splitlines() if not l.startswith('#')]
        assert len(rows) == 80
        for row in rows:
            assert len(row) == 3
            assert all(lo < float(x) < hi for x, lo, hi in zip(row, [.7, 1.5, 2.5], [1.4, 2.5, 3.5]))
        trace = [l.split() for l in (a / f'chain_{chain}_trace.txt').read_text().splitlines() if not l.startswith('#')]
        assert len(trace) == 180
        assert [int(r[0]) for r in trace] == [1] * 100 + [0] * 80
        if algorithm == 'nuts':
            header = (a / f'chain_{chain}_trace.txt').read_text().splitlines()[0].lstrip('# ').split()
            for row in trace:
                values = dict(zip(header, row))
                assert 1 <= int(values['depth']) <= 10
                assert 1 <= int(values['evaluations']) <= 2**int(values['depth']) - 1
                assert values['max_depth_reached'] in ('0', '1')
    assert (a / 'chain_0_vs.txt').read_bytes() != (a / 'chain_1_vs.txt').read_bytes()
    for name in ('observations.txt', 'covariance.txt'):
        assert (a / name).read_bytes() == (data / name).read_bytes()
    assert 'conditional on supplied parameterization' in (a / 'metadata.txt').read_text()
    for dest in ('predictive-a', 'predictive-b'):
        run('predict', '--run', a, '--output', work / dest, '--draws-per-chain', 8, '--seed', 17)
    for name in ('latent.txt', 'replicated.txt', 'draws.txt', 'selected_log_vs.txt'):
        assert (work / 'predictive-a' / name).read_bytes() == (work / 'predictive-b' / name).read_bytes()
    predicted_rows = [r.split() for r in (work / 'predictive-a/latent.txt').read_text().splitlines() if not r.startswith('#')]
    assert len(predicted_rows) == 24 and all(len(r) == 14 for r in predicted_rows)
    selection = [r.split() for r in (work / 'predictive-a/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert [(int(r[1]), int(r[2])) for r in selection] == [(c, j*10) for c in range(3) for j in range(8)]
    run('predict', '--run', a, '--output', work / 'predictive-a', codes=(1,))
    run('predict', '--run', a, '--output', work / 'too-many', '--draws-per-chain', 81, codes=(1,))
    adaptive = work / 'adaptive.toml'
    adaptive.write_text(content if 'metric_adaptation' in content else content.replace('prior = "uniform_vs"', 'prior = "uniform_vs"\nmetric_adaptation = "diagonal"'))
    sample(work / 'adaptive', cfg=adaptive)
    for chain in range(3):
        mass = [list(map(float, row.split())) for row in (work / f'adaptive/chain_{chain}_mass.txt').read_text().splitlines()]
        assert len(mass) == 3 and all(mass[i][i] > 0 for i in range(3))
        assert all(mass[i][j] == 0 for i in range(3) for j in range(3) if i != j)
        assert '# 90 75' in (work / f'adaptive/chain_{chain}_metric_updates.txt').read_text()
    if importlib.util.find_spec('numpy') is not None:
        report = work / 'diagnostics.json'
        subprocess.run([sys.executable, str(root / 'scripts/posterior_diagnostics.py'), str(a), '--output', str(report)], check=True, capture_output=True)
        diagnostics = json.loads(report.read_text())
        assert len(diagnostics['sampler']) == 3
        for chain, record in enumerate(diagnostics['sampler']):
            lines = (a / f'chain_{chain}_trace.txt').read_text().splitlines()
            header = lines[0].lstrip('# ').split()
            retained = [dict(zip(header, row.split())) for row in lines[1:] if row.split()[0] == '0']
            assert record['gradient_evaluations'] == sum(int(r['evaluations']) for r in retained)
            if algorithm == 'nuts':
                assert record['max_depth_reached'] == sum(int(r['max_depth_reached']) for r in retained)
                assert record['maximum_tree_depth'] == max(int(r['depth']) for r in retained)
    sample(a, codes=(1,))
    if algorithm == 'nuts':
        modern = work / 'modern.toml'
        modern.write_text(content.replace('[nuts]', '[nuts]\nkernel = "multinomial"\nfind_initial_step_size = true'))
        sample(work / 'modern-a', cfg=modern)
        sample(work / 'modern-b', cfg=modern)
        assert 'sampler=multinomial_nuts' in (work / 'modern-a/metadata.txt').read_text()
        for chain in range(3):
            prefix = f'chain_{chain}_'
            for suffix in ('vs', 'z', 'trace', 'cost', 'step_size_searches'):
                assert (work / f'modern-a/{prefix}{suffix}.txt').read_bytes() == (work / f'modern-b/{prefix}{suffix}.txt').read_bytes()
            costs = dict(line.split('=') for line in (work / f'modern-a/{prefix}cost.txt').read_text().splitlines())
            searches = [line.split() for line in (work / f'modern-a/{prefix}step_size_searches.txt').read_text().splitlines() if not line.startswith('#')]
            updates = [line.split()[1] for line in (work / f'modern-a/{prefix}metric_updates.txt').read_text().splitlines() if line.startswith('# ') and not line.startswith('# each')]
            assert [s[0] for s in searches] == ['0'] + updates
            assert int(costs['step_search_evaluations']) == sum(int(s[1]) for s in searches)
            assert int(costs['target_evaluations']) == 1 + int(costs['transition_evaluations']) + int(costs['step_search_evaluations'])
        bad_kernel = work / 'bad-kernel.toml'
        bad_kernel.write_text(content.replace('[nuts]', '[nuts]\nkernel = "unknown"'))
        assert 'Unknown NUTS kernel' in sample(work / 'bad-kernel', cfg=bad_kernel, codes=(1,)).stderr
        shrink = work / 'shrink.toml'
        shrink.write_text(modern.read_text().replace('[nuts]', '[nuts]\nmetric_shrinkage = 25.0'))
        sample(work / 'shrink', cfg=shrink)
        assert 'metric_shrinkage=25' in (work / 'shrink/metadata.txt').read_text()
        assert (work / 'shrink/chain_0_mass.txt').read_bytes() != (work / 'modern-a/chain_0_mass.txt').read_bytes()
        shrink.write_text(shrink.read_text().replace('metric_shrinkage = 25.0', 'metric_shrinkage = -1.0'))
        assert 'metric_shrinkage' in sample(work / 'bad-shrink', cfg=shrink, codes=(1,)).stderr
        capped = work / 'capped.toml'
        capped.write_text(content.replace('max_depth = 10', 'max_depth = 1').replace('step_size = 0.03', 'step_size = 0.000001').replace('adapt_step_size = true', 'adapt_step_size = false').replace('metric_adaptation = "diagonal"', 'metric_adaptation = "none"'))
        sample(work / 'capped', cfg=capped, codes=(5,))
        metadata = (work / 'capped/metadata.txt').read_text()
        assert 'sampler=slice_nuts_hoffman_gelman_2014' in metadata
        assert 'leapfrog_steps=' not in metadata and 'step_jitter=' not in metadata
        incompatible = work / 'incompatible.toml'
        incompatible.write_text(content.replace('max_depth = 10', 'max_depth = 10\nleapfrog_steps = 12'))
        sample(work / 'incompatible', cfg=incompatible, codes=(1,))
    invalid = work / 'invalid.toml'
    invalid.write_text(content.replace('[1.1, 1.8, 2.8]', '[0.7, 1.8, 2.8]'))
    sample(work / 'invalid', cfg=invalid, codes=(1,))
    fixed = work / 'fixed.toml'
    fixed.write_text(content.replace('[inversion]', '[inversion]\nfixed_vs_layers = [2]'))
    rejected = sample(work / 'fixed', cfg=fixed, codes=(1,))
    assert 'fixed_vs_layers currently supported only by invert/synthetic' in rejected.stderr
    missing = work / 'missing.txt'
    lines = (data / 'observations.txt').read_text().splitlines()
    rows = []
    for line in lines:
        if line.startswith('#'):
            rows.append(line)
        else:
            row = line.split(); row[2] = '100'; rows.append(' '.join(row))
    missing.write_text('\n'.join(rows) + '\n')
    sample(work / 'missing', observations=missing, codes=(1,))
    (a / 'observations.txt').write_bytes(missing.read_bytes())
    run('predict', '--run', a, '--output', work / 'failed-predictive', '--draws-per-chain', 2, codes=(5,))
    failed = [r.split() for r in (work / 'failed-predictive/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert len(failed) == 6 and all(r[3] == 'forward_failure' for r in failed)
print(f'{algorithm} workflow: replay, chain streams, retained draws, bounds, covariance, failures passed')
