"""Analytic checks of the independent marginal-posterior integration reference."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_joint_grid import log_scale_integral, reference, validation_issues
from posterior_diagnostics import diagnose

noise = dict(kind='lognormal', mean_log=.2, sd_log=.6)
actual = log_scale_integral(np.array([0.]), 3, noise, 192)
np.testing.assert_allclose(actual, -3*.2+.5*9*.6**2, atol=1e-12)
_, scale_mean = log_scale_integral(np.array([0.]), 3, noise, 192, with_scale=True)
np.testing.assert_allclose(scale_mean, np.exp(.2+(0.5-3)*.6**2), atol=1e-12)
x, y = np.meshgrid(np.linspace(1, 3, 301), np.linspace(2, 5, 301), indexing='ij')
grid = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
config = dict(prior=dict(kind='uniform', measure='physical_vs'), observations=[])
np.testing.assert_allclose(reference(grid, ['unused'], config)['mean'], [2, 3.5], atol=1e-12)
config['prior']['measure'] = 'log_vs'
np.testing.assert_allclose(reference(grid, ['unused'], config)['mean'], [2/np.log(3), 3/np.log(2.5)], atol=1e-5)
# Fixed-scale, separated Gaussian factors; support is >6 sd from each mean.
grid[:, 2] = ((grid[:, 0]-2)/.12)**2+((grid[:, 1]-3.5)/.2)**2
config = dict(prior=dict(kind='uniform', measure='physical_vs'),
              observations=[dict(name='gaussian', values=[0., 0.], noise=dict(kind='fixed', scale=1.))])
result = reference(grid, ['gaussian'], config)
np.testing.assert_allclose(result['mean'], [2, 3.5], atol=1e-11)
np.testing.assert_allclose(result['sd'], [.12, .2], atol=1e-10)
# Chains trapped in different modes can have such a large MCSE that an
# inaccurate pooled mean still passes a mean-only comparison. Reject it.
rng = np.random.default_rng(72163)
stuck = rng.normal(0., .1, size=(4, 2000))+np.array([-2., -2., -2., 2.])[:, None]
row = diagnose(stuck)
assert row['status'] == 'warning' and abs(row['mean']) < 5*row['mcse_mean']
comparison = dict(within_5mcse_and_2grid_error=True, scales={})
summary = dict(parameters={'Vs[0]': row}, numerical_proposals_including_warmup=0)
assert validation_issues(comparison, summary)
summary['parameters']['Vs[0]'] = diagnose(rng.normal(size=(4, 2000)))
assert not validation_issues(comparison, summary)
summary['numerical_proposals_including_warmup'] = 1
assert validation_issues(comparison, summary)
summary['numerical_proposals_including_warmup'] = 0
comparison['scales']['scale:dispersion'] = dict(within_5mcse_and_2grid_error=False)
assert validation_issues(comparison, summary)
print('Independent quadrature and rejection of poorly mixed/inaccurate posterior comparisons passed')
