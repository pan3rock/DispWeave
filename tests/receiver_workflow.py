"""Compare CLI RF waveforms to independent state-space spectra and NumPy FFT."""
from pathlib import Path
import tempfile
import subprocess
import sys
import numpy as np
exe, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    for name, water, width, shift, custom_source in [('layered', .01, 2.5, 0., False), ('layered', .2, 4., .37, True), ('low_velocity', .05, 1.5, -.23, False)]:
        raw = np.loadtxt(root / f'tests/fixtures/receiver/{name}.txt')
        fields = (root / f'tests/fixtures/receiver/{name}-input.txt').read_text().splitlines()
        layers, _, slow = fields[0].split();layers = int(layers)
        model = np.array([list(map(float, r.split())) for r in fields[1:1+layers]])
        source = np.zeros(512);source[0] = 1;source[3] = .4;source[8] = -.2
        cfg = work / 'input.toml'
        text = f'''[model]
depth = {np.r_[0, np.cumsum(model[:-1, 0])].tolist()}
vs = {model[:, 1].tolist()}
vp = {model[:, 2].tolist()}
rho = {model[:, 3].tolist()}
[receiver]
incident = "P"
slowness_s_km = {slow}
samples = 512
dt = 0.1
gaussian_width = {width}
water_level = {water}
pre_arrival = 5.0
time_shift = {shift}
'''
        if custom_source:
            text += f'source = {source.tolist()}\n'
        cfg.write_text(text)
        out = work / f'{name}-{water}'
        p = subprocess.run([str(exe), 'receiver', '--config', str(cfg), '--output', str(out)], capture_output=True, text=True)
        assert p.returncode == 0, (p.stdout, p.stderr)
        got = np.loadtxt(out / 'receiver.txt', usecols=(0, 1))
        w = 2*np.pi*raw[:, 0]
        stf = np.fft.rfft(source).conj() if custom_source else np.ones(len(w))
        radial = (raw[:, 1] + 1j*raw[:, 2])*stf
        vertical = (raw[:, 3] + 1j*raw[:, 4])*stf
        power = np.abs(vertical)**2
        denominator = np.maximum(power, water*power.max())
        gaussian = np.exp(-w*w/(4*width*width))
        normalization = np.fft.irfft(power/denominator*gaussian, n=512)[0]
        transfer = radial*vertical.conj()/denominator*gaussian/normalization
        expected = np.fft.irfft(transfer.conj()*np.exp(-1j*w*(5+shift)), n=512)
        np.testing.assert_allclose(got[:, 0], np.arange(512)*.1-5, atol=1e-14)
        np.testing.assert_allclose(got[:, 1], expected, atol=2e-11, rtol=2e-9)
        print(name, water, 'max independent RF error', np.max(np.abs(got[:, 1]-expected)))
        assert (out / 'config.toml').read_bytes() == cfg.read_bytes()
        meta = (out / 'metadata.txt').read_text()
        assert 'incident=P_upgoing' in meta and 'finite_periodic_DFT' in meta
        replay = subprocess.run([str(exe), 'receiver', '--config', str(cfg), '--output', str(out)], capture_output=True)
        assert replay.returncode == 1
    # Critical angle: all frequency bins and RF samples must remain explicit.
    cfg.write_text('''[model]
depth=[0.0,20.0]
vs=[5.0,4.5]
vp=[10.0,8.1]
rho=[2.7,3.3]
[receiver]
slowness_s_km=0.1
samples=64
dt=0.1
pre_arrival=1.0
''')
    failed = work / 'failed'
    p = subprocess.run([str(exe), 'receiver', '--config', str(cfg), '--output', str(failed)], capture_output=True)
    assert p.returncode == 2
    values = np.loadtxt(failed / 'receiver.txt', usecols=(0, 1))
    assert values.shape == (64, 2) and np.isnan(values[:, 1]).all()
    rows = [r for r in (failed / 'spectrum.txt').read_text().splitlines() if not r.startswith('#')]
    assert len(rows) == 33 and all('numerical_failure' in r for r in rows)
    cfg.write_text(cfg.read_text().replace('[receiver]', '[receiver]\nincident="SV"'))
    p = subprocess.run([str(exe), 'receiver', '--config', str(cfg), '--output', str(work / 'unsupported')], capture_output=True)
    assert p.returncode == 1 and not (work / 'unsupported').exists()
print('Independent RF waveforms, source/filter/shift conventions, failure retention and output protection passed')
