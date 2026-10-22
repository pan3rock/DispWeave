"""CLI contracts: preserve root/field distinction, failed rows and physical definition."""
from pathlib import Path
import subprocess
import sys
import shlex
import tempfile
import math
exe, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / 'values.txt'
    def run(config, code):
        p = subprocess.run([str(exe), 'ellipticity', '--config', str(config), '--output', str(out)], capture_output=True, text=True)
        assert p.returncode == code, (p.returncode, p.stdout, p.stderr)
        text = out.read_text()
        assert 'NOT ambient-noise H/V' in text and 'rtol=8*double_epsilon' in text
        return [shlex.split(l) for l in text.splitlines() if l and not l.startswith('#')]
    rows = run(root / 'examples/ellipticity/layered.toml', 0)
    assert len(rows) == 10 and all(r[4] == r[6] == 'valid' for r in rows)
    assert all(math.isfinite(float(r[5])) and float(r[5]) >= 0 for r in rows)
    rows = run(root / 'examples/forward/missing.toml', 2)
    assert len(rows) == 2 and rows[1][6] == 'root_unavailable' and math.isnan(float(rows[1][5]))
    config = Path(tmp) / 'pole.toml'
    config.write_text('''[model]
depth=[0.0,0.1]
vs=[0.4,2.5]
vp=[0.8,4.5]
rho=[2.0,2.5]
[queries]
frequency_hz=[1.0175858928204988,1.0174,1.0178]
mode=[0,0,0]
''')
    rows = run(config, 2)
    assert len(rows) == 3 and all(r[4] == 'valid' for r in rows)
    assert rows[0][6] == 'near_pole' and math.isnan(float(rows[0][5]))
    assert all(r[6] == 'valid' for r in rows[1:])
    original = config.read_bytes()
    p = subprocess.run([str(exe), 'ellipticity', '--config', str(config), '--output', str(config)], capture_output=True)
    assert p.returncode == 1 and config.read_bytes() == original
print('Ellipticity CLI physical label, output ordering, missing/pole rows and input protection passed')
