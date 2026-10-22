import json
import tempfile
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rj_results import load_run, depth_values
from rj_diagnostics import analyze, residence

r = residence(np.array([1, 1, 2, 3, 2, 1, 2, 3, 1]), 1, 3)
assert r['changes'] == 7 and r['endpoint_roundtrips'] == 2
assert r['transition_counts'][0][0] == 1
assert r['residence']['1']['maximum'] == 2
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / 'config.toml').write_text('''[rj_prior]
min_layers=1
max_layers=3
depth_limit=4.0
min_spacing=0.1
vs_lower=0.5
vs_upper=3.0
''')
    for c in range(2):
        k = 2+c
        state = '1.0 1.0 2.0' if c == 0 else '.75 1.25 1.0 1.5 2.0'
        (root / f'chain_{c}_models.txt').write_text('# draw K loglike logprior interfaces vs\n'+''.join(f'{i} {k} -1 -2 {state}\n' for i in range(8)))
        (root / f'chain_{c}_trace.txt').write_text('# warmup move proposal_status evaluated evaluation_status accepted layers_before layers_after log_accept_ratio\n'+''.join(f'0 velocity valid 1 valid 0 {k} {k} -1\n' for i in range(8)))
    (root / 'chain_0_failed_models.txt').write_text('1 forward_failure 2 1.0 2.0 1.0\n')
    report = analyze(root, 3)
    assert report['status'] == 'warning'
    assert [p['probability'] for p in report['layer_probabilities']] == [0, .5, .5]
    assert report['layer_probabilities'][1]['diagnostics']['status'] == 'constant_chain'
    assert report['interface_bins'][0]['probability_any'] == 1
    assert report['interface_bins'][0]['expected_count'] == 1.5
    assert all(c['changes'] == 0 for c in report['sampler'])
    json.dumps(report, allow_nan=False)
    _, _, chains = load_run(root)
    values = depth_values(chains, np.array([0., 1., 4.]))
    assert np.all(values[0, :, 1] == 2.)  # Exact interface uses lower layer.
    bad = root / 'chain_1_models.txt'
    bad.write_text(bad.read_text().replace('0 3 -1 -2', '0 4 -1 -2', 1))
    try:
        load_run(root)
        raise AssertionError('Malformed dimension accepted')
    except ValueError:
        pass
print('RJ reader, residence, categorical nonmixing, depth mapping and interface occupancy passed')
