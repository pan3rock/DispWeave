#!/usr/bin/env python3
"""Freeze independent state-space spectra and audit Telewavesim 0.2.1.
Supply unpacked source and a LAPACK/OpenBLAS library; SciPy is validation only.
No downloads/install; the external recursion issue is recorded, not hidden.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import scipy
from receiver_state_reference import spectrum
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--lapack-library', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
a.source = a.source.resolve();a.lapack_library = a.lapack_library.resolve()
if a.output.exists() and any(a.output.iterdir()):
    raise FileExistsError('Reference destination must be new or empty')
a.output.mkdir(parents=True, exist_ok=True)
# h(km), Vs/Vp(km/s), rho(g/cm^3); halfspace thickness ignored.
cases = {
    'homogeneous_cap': ([[1, 3.5, 6.3, 2.7], [0, 3.5, 6.3, 2.7]], .06, [0, .1, 1, 5]),
    'layered': ([[20, 3.5, 6.3, 2.7], [0, 4.5, 8.1, 3.3]], .06, [k/51.2 for k in range(257)]),
    'low_velocity': ([[3, 3.4, 6.0, 2.6], [7, 2.8, 5.0, 2.4], [0, 4.3, 7.6, 3.2]], .04, [k/51.2 for k in range(257)]),
    'evanescent': ([[.1, 12., 22., 3.5], [0, 4.5, 8.1, 3.3]], .1, [0, .1, 1, 5]),
}
sources = [a.source / 'src/rmat.f90', a.source / 'src/rmat_sub.f90']
expected = {
    'rmat.f90': '80cb1ee8f7728f476b02b59b66496330d4f6ecea3322bff768d7cca0d9ca5930',
    'rmat_sub.f90': 'f8863df1b70b89a104d1d774307423944b78167bd4c9b43a34a241372e1cdb98',
}
for path in sources:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]:
        raise ValueError(f'Expected unmodified Telewavesim 0.2.1: {path}')
with tempfile.TemporaryDirectory() as tmp:
    executable = Path(tmp) / 'reference'
    subprocess.run(['gfortran', '-O2', *map(str, sources), str(root / 'benchmarks/receiver_reference_driver.f90'),
                    str(a.lapack_library), '-Wl,-rpath,' + str(a.lapack_library.parent), '-o', str(executable)], cwd=tmp, check=True)
    audit = a.output / 'telewavesim-audit'
    audit.mkdir()
    # Audit-only patch: the published addit computes reverbi but then uses
    # reverb in four recursion products. Keep original output and exact diff.
    original = sources[1].read_text()
    start = original.index('          tusw = MATMUL')
    end = original.index('!\n! copy over', start)
    corrected = original[:start]+original[start:end].replace('MATMUL(reverb,', 'MATMUL(reverbi,')+original[end:]
    import difflib
    (audit / 'addit.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), corrected.splitlines(True), fromfile='rmat_sub.f90', tofile='rmat_sub.f90')))
    patched = Path(tmp) / 'rmat_sub.f90'
    patched.write_text(corrected)
    patched_executable = Path(tmp) / 'corrected-reference'
    subprocess.run(['gfortran', '-O2', str(sources[0]), str(patched), str(root / 'benchmarks/receiver_reference_driver.f90'),
                    str(a.lapack_library), '-Wl,-rpath,' + str(a.lapack_library.parent), '-o', str(patched_executable)], cwd=tmp, check=True)
    errors = {}
    for name, (model, slow, frequencies) in cases.items():
        text = f'{len(model)} {len(frequencies)} {slow}\n' + ''.join(' '.join(map(str, row))+'\n' for row in model)
        text += ''.join(f'{f:.17g}\n' for f in frequencies)
        (a.output / f'{name}-input.txt').write_text(text)
        reference = spectrum(model, slow, frequencies)
        np.savetxt(a.output / f'{name}.txt', reference, fmt='%.17e', header='Independent state-space matrix exponential\nf_Hz R_real R_imag Z_up_real Z_up_imag')
        if name == 'evanescent':  # External isotroc uses real square roots.
            continue
        import io
        errors[name] = {}
        for label, binary in [('unmodified', executable), ('addit-corrected', patched_executable)]:
            run = subprocess.run([str(binary)], input=text, capture_output=True, text=True, check=True)
            (audit / f'{name}-{label}.txt').write_text(run.stdout)
            values = np.loadtxt(io.StringIO(run.stdout))
            error = float(np.max(np.abs(values[:, 1:]-reference[:, 1:])))
            errors[name][label] = error
            if label == 'addit-corrected' or name != 'low_velocity':
                assert error < 2e-11, (name, label, error)
metadata = dict(reference='State-space ODE matrix exponential (SciPy)', scipy=scipy.__version__,
                external_audit=dict(package='Telewavesim', version='0.2.1', source='https://pypi.org/project/telewavesim/0.2.1/',
                license='MIT, Copyright 2019 Pascal Audet', source_hashes=expected,
                archive_sha256='4dd1e1a81a55ab6193736b50ee76fe277c06c518d8a90a49e20edccd91db0da6', errors=errors),
                convention='Incident upgoing P, x along horizontal propagation, Z positive up, exp(-i*omega*t). Real omega: no plane_land 0.001i attenuation.',
                production_dependency=False,
                limitations='Matrix exponentials are only validated at moderate optical thickness. Telewavesim 0.2.1 returns zero transmission for one layer and its addit uses reverb instead of reverbi in four products. Unmodified multi-interface results are not an acceptance oracle; original and explicitly patched audit outputs are preserved. Its isotropic real-q routine cannot validate evanescent layers.')
metadata['generator_hashes'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), root/'benchmarks/receiver_state_reference.py', root/'benchmarks/receiver_reference_driver.f90']}
metadata['fixture_hashes'] = {str(path.relative_to(a.output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(a.output.rglob('*')) if path.is_file()}
(a.output / 'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(a.output)
