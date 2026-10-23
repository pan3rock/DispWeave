"""Reproduce the M8d noisy minima at which standard root accuracy stalled LM."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np

exe, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(root/'scripts'))
from joint_results import read_config, write_config

observations = json.loads((root/'tests/fixtures/joint/lm_observations.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    for name, cost in [('joint', 15.37240483206), ('conflict', 276.32276885825)]:
        doc = read_config(root/'examples/joint/synthetic.toml')
        # Tiny chains exercise the CLI only; these are not sampling validation.
        doc['sampler'].update(warmup=0, draws=8, predictive_per_chain=1,
                              adapt_scale=False, adapt_covariance=False)
        for b in doc['observations']:
            b['values'] = observations[name][b['name']]
        rf = doc['observations'][-1]
        times = np.array(rf['sample_indices'])*rf['processing']['dt']
        sigma = np.array(rf.pop('sigma'))
        rf['covariance'] = (sigma[:, None]*sigma[None, :]*np.exp(-np.abs(times[:, None]-times[None, :])/.35)).tolist()
        cfg = work/f'{name}.toml';write_config(cfg, doc)
        output = work/name
        subprocess.run([str(exe), 'joint', '--config', str(cfg), '--output', str(output)], check=True)
        summary = dict(line.split('=', 1) for line in (output/'lm_summary.txt').read_text().splitlines())
        assert summary['termination'] == 'converged', summary
        assert float(summary['projected_gradient']) <= 1e-5
        assert abs(float(summary['cost'])-cost) < 1e-8
        assert summary['dispersion_root_accuracy'] == 'refined'
        assert summary['forward_failures'] == summary['gradient_failures'] == '0'
        assert summary['used_for_chain_initialization'] == 'false'
        if name == 'conflict':
            # This constrained optimum is at the lower halfspace Vs bound.
            assert abs(np.loadtxt(output/'lm_vs.txt')[1]-3.45) < 1e-12
print('Noisy joint LM interior and active-boundary minima converge without relaxing tolerances')
