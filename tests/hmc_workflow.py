"""CLI contract checks; distribution correctness is tested independently in C++."""
import pathlib
import subprocess
import sys
import tempfile

exe = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2])

def run(*args, codes=(0,)):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode in codes, (p.returncode, p.stdout, p.stderr)
    return p

with tempfile.TemporaryDirectory() as tmp:
    work = pathlib.Path(tmp)
    config = work / 'input.toml'
    content = (root / 'examples/hmc/synthetic.toml').read_text().replace('warmup = 1000', 'warmup = 100').replace('draws = 4000', 'draws = 80')
    config.write_text(content)
    data = work / 'data'
    run('synthetic', '--config', config, '--output', data)
    def sample(output, cfg=config, observations=data / 'observations.txt', codes=(0, 5)):
        return run('hmc', '--config', cfg, '--data', observations, '--covariance', data / 'covariance.txt', '--output', output, codes=codes)
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
    adaptive.write_text(content.replace('prior = "uniform_vs"', 'prior = "uniform_vs"\nmetric_adaptation = "diagonal"'))
    sample(work / 'adaptive', cfg=adaptive)
    for chain in range(3):
        mass = [list(map(float, row.split())) for row in (work / f'adaptive/chain_{chain}_mass.txt').read_text().splitlines()]
        assert len(mass) == 3 and all(mass[i][i] > 0 for i in range(3))
        assert all(mass[i][j] == 0 for i in range(3) for j in range(3) if i != j)
        assert '# 90 75' in (work / f'adaptive/chain_{chain}_metric_updates.txt').read_text()
    sample(a, codes=(1,))
    invalid = work / 'invalid.toml'
    invalid.write_text(content.replace('[1.1, 1.8, 2.8]', '[0.7, 1.8, 2.8]'))
    sample(work / 'invalid', cfg=invalid, codes=(1,))
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
print('HMC workflow: replay, chain streams, retained draws, bounds, covariance, failures passed')
