"""Distribution/edge-case checks; optional --reference compares against ArviZ 0.21.0."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from posterior_diagnostics import diagnose, rank_normalize, split


def cases():
    rng = np.random.default_rng(716)
    iid = rng.normal(size=(4, 1001))
    ar = rng.normal(size=(4, 1001))
    anti = ar.copy()
    for i in range(1, ar.shape[1]):
        ar[:, i] = .95*ar[:, i-1] + np.sqrt(1-.95**2)*ar[:, i]
        anti[:, i] = -.7*anti[:, i-1] + np.sqrt(1-.7**2)*anti[:, i]
    yield 'iid', iid
    yield 'autocorrelated', ar
    yield 'antithetic', anti
    yield 'ties', np.round(iid, 1)
    yield 'heavy_tail', rng.standard_cauchy((4, 1001))
    yield 'location_shift', iid + np.arange(4)[:, None]*2
    yield 'scale_shift', iid * np.array([1, 1, 1, 5])[:, None]
    yield 'constant_within_different_between', np.tile(np.arange(4)[:, None], (1, 1001))


if __name__ == '__main__':
    fixture = json.loads((Path(__file__).parent / 'fixtures/diagnostics_arviz.json').read_text())
    reference = '--reference' in sys.argv
    if reference:
        import arviz as az
        assert az.__version__ == fixture['arviz_version']
    for name, x in cases():
        actual = diagnose(x)
        expected = fixture['cases'][name]
        for key, value in expected.items():
            assert np.isclose(actual[key], float(value), rtol=1e-9, atol=1e-10), (name, key, actual[key], value)
        if reference:
            for key, value in [('rhat', az.rhat(x)), ('ess_bulk', az.ess(x)), ('ess_tail', az.ess(x, method='tail')), ('mcse_mean', az.mcse(x))]:
                assert np.isclose(actual[key], float(value), rtol=1e-9, atol=1e-10), (name, key)
        if name in ('location_shift', 'scale_shift'):
            assert actual['rhat'] > 1.1 and actual['status'] == 'warning'
        if name == 'autocorrelated':
            assert actual['ess_bulk'] < x.size/10
        if name == 'antithetic':
            assert actual['ess_bulk'] > x.size
        print(name, actual['rhat'], actual['ess_bulk'], actual['ess_tail'], actual['status'])
    constant = diagnose(np.ones((4, 40)))
    assert constant['status'] == 'constant_chain' and np.isnan(constant['rhat']) and np.isnan(constant['ess_bulk'])
    for x in [np.ones((1, 100)), np.ones((4, 7)), np.full((4, 100), np.nan), np.full((4, 100), np.inf)]:
        try:
            diagnose(x)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed chains accepted')
    x = np.array([[1, 1, 2, 3], [1, 2, 3, 3]])
    z = rank_normalize(x)
    assert z[0, 0] == z[0, 1] == z[1, 0]
    assert split(np.arange(9)[None, :]).tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    print('diagnostics tests passed' + (' with independent ArviZ comparison' if reference else ''))
