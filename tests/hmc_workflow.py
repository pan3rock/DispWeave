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
print('HMC workflow: replay, chain streams, retained draws, bounds, covariance, failures passed')
