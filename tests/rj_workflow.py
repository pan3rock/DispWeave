"""CLI data/state contract checks, separate from distribution validation."""
from pathlib import Path
import math
import subprocess
import sys
import tempfile
exe = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])

def run(*args, codes=(0,)):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert result.returncode in codes, (result.returncode, result.stdout, result.stderr)
    return result

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    cfg = work / 'input.toml'
    content = (root / 'examples/rjmcmc/synthetic.toml').read_text().replace('warmup = 10000', 'warmup = 60').replace('draws = 50000', 'draws = 150')
    cfg.write_text(content)
    data = work / 'data'
    run('synthetic', '--config', cfg, '--output', data, codes=(0, 4))
    def sample(dest, config=cfg, observations=data / 'observations.txt', codes=(0, 5)):
        return run('rjmcmc', '--config', config, '--data', observations, '--covariance', data / 'covariance.txt', '--output', dest, codes=codes)
    a, b = work / 'a', work / 'b'
    sample(a); sample(b)
    for chain in range(3):
        for suffix in ('models', 'trace'):
            name = f'chain_{chain}_{suffix}.txt'
            assert (a / name).read_bytes() == (b / name).read_bytes(), 'Seed replay'
        rows = [list(map(float, r.split())) for r in (a / f'chain_{chain}_models.txt').read_text().splitlines() if not r.startswith('#')]
        assert len(rows) == 150
        for index, row in enumerate(rows):
            k = int(row[1])
            assert row[0] == index and row[1] == k and 1 <= k <= 4
            assert len(row) == 2*k+3 and all(math.isfinite(x) for x in row)
            interfaces, vs = row[4:4+k-1], row[4+k-1:]
            depths = [0, *interfaces, 5]
            assert all(y-x >= .3 for x, y in zip(depths[:-1], depths[1:]))
            assert all(.8 <= x <= 2.7 for x in vs)
        trace = [r.split() for r in (a / f'chain_{chain}_trace.txt').read_text().splitlines() if not r.startswith('#')]
        assert len(trace) == 210 and [int(r[0]) for r in trace] == [1]*60 + [0]*150
        assert all(r[1] in ('birth', 'death', 'interface', 'velocity') for r in trace)
        assert all(r[4] == 'not_evaluated' for r in trace if r[3] == '0')
        assert [int(r[7]) for r in trace[60:]] == [int(r[1]) for r in rows]
        failures = [r.split() for r in (a / f'chain_{chain}_failed_models.txt').read_text().splitlines() if not r.startswith('#')]
        expected = {int(r[9]): r[4] for r in trace if r[3] == '1' and r[4] != 'valid'}
        assert {int(r[0]): r[1] for r in failures} == expected
    for dest in ('predict-a', 'predict-b'):
        run('rj-predict', '--run', a, '--output', work / dest, '--draws-per-chain', 8, '--seed', 17)
    for name in ('latent.txt', 'replicated.txt', 'draws.txt', 'selected_models.txt'):
        assert (work / 'predict-a' / name).read_bytes() == (work / 'predict-b' / name).read_bytes()
    selected = [r.split() for r in (work / 'predict-a/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert [(int(r[1]), int(r[2])) for r in selected] == [(c, j*150//8) for c in range(3) for j in range(8)]
    run('rj-predict', '--run', a, '--output', work / 'predict-a', codes=(1,))
    run('rj-predict', '--run', a, '--output', work / 'too-many', '--draws-per-chain', 151, codes=(1,))
    for name in ('observations.txt', 'covariance.txt'):
        assert (a / name).read_bytes() == (data / name).read_bytes()
    assert 'posterior=transdimensional' in (a / 'metadata.txt').read_text()
    sample(a, codes=(1,))
    invalid = work / 'bad.toml'
    invalid.write_text(content.replace('vp_vs_ratio = 1.8', 'vp_vs_ratio = 1.0'))
    sample(work / 'bad', config=invalid, codes=(1,))
    missing = work / 'missing.txt'
    lines = []
    for line in (data / 'observations.txt').read_text().splitlines():
        if line.startswith('#'): lines.append(line)
        else:
            row = line.split(); row[2] = '100'; lines.append(' '.join(row))
    missing.write_text('\n'.join(lines)+'\n')
    sample(work / 'missing', observations=missing, codes=(1,))
    (a / 'observations.txt').write_bytes(missing.read_bytes())
    run('rj-predict', '--run', a, '--output', work / 'failed-predict', '--draws-per-chain', 2, codes=(5,))
    failed = [r.split() for r in (work / 'failed-predict/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert len(failed) == 6 and all(r[3] == 'forward_failure' for r in failed)
print('RJ workflow replay, ragged states, diagnostics, covariance and failures passed')
