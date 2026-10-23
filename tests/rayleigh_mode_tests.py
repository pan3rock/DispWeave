"""Independent root count: analytic roots, critical speeds and close branches."""
from pathlib import Path
import io
import subprocess
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
from rayleigh_mode_audit import BoundarySystem
from rayleigh_boundary_reference import root as travelling_root, boundary_matrix


class ModeAudit(unittest.TestCase):
    def test_full_trapped_interval(self):
        for name in ['formal', 'warmup']:
            m = np.loadtxt(ROOT/f'tests/fixtures/rayleigh/crust-cutoff-{name}.txt')
            self.assertEqual(BoundarySystem(m, .1612).trapped_count()['count'], 2)
            m[-1, 3] += .025
            self.assertEqual(BoundarySystem(m, .1612).trapped_count()['count'], 3)
        m = np.array([[0., 0., 2.7, 3., 3*np.sqrt(2)]])
        self.assertEqual(BoundarySystem(m, .1).trapped_count()['count'], 1)

    def test_homogeneous_cubic_and_subdivision(self):
        ratio = 1.8; b = ratio**-2
        roots = np.roots([1., -8., 24-16*b, -16*(1-b)])
        expected = 2*np.sqrt(next(x.real for x in roots if abs(x.imag) < 1e-12 and 0 < x.real < 1))
        for tops in [[0.], [0., .5, 5.]]:
            m = np.c_[np.arange(len(tops)), tops, [2.5]*len(tops), [2.]*len(tops), [3.6]*len(tops)]
            s = BoundarySystem(m, .37); audit = s.audit(.8, 2*(1-1e-10))
            self.assertEqual(audit['count'], 1)
            self.assertAlmostEqual(audit['roots'][0]['phase_km_s'], expected, places=10)
            self.assertLess(audit['roots'][0]['relative_singular_value'], 1e-10)
            empty = s.audit(.8, 1.5)
            self.assertEqual(empty['count'], 0)
            self.assertEqual(empty['roots'], [])

    def test_internal_critical_speeds_are_not_roots(self):
        m = np.array([[0., 0., 2.1, 1., 1.8], [1., 1.3, 2.5, 3., 5.4]])
        s = BoundarySystem(m, .37)
        for critical in [1., 1.8]:
            a = s.matrix(critical)
            self.assertTrue(np.isfinite(a).all())
            self.assertLess(np.max(abs(a.imag)), 1e-14)
            self.assertGreater(np.linalg.svd(a, compute_uv=False)[-1], 1e-5)
            self.assertEqual(s.sign(critical*(1-1e-8)), s.sign(critical*(1+1e-8)))
        with self.assertRaises(ValueError): s.count(.5, 3., (.02,))
        with self.assertRaises(ValueError): s.count(.5, 2.9, (.02,), 128, 64)
        with self.assertRaises(ValueError): BoundarySystem(m, 0.)

    def test_crust_roots_against_unchanged_travelling_reference(self):
        m = np.loadtxt(ROOT/'tests/fixtures/rayleigh/crust-cutoff-formal.txt')
        s = BoundarySystem(m, .1612); audit = s.audit(.5*m[:, 3].min(), m[-1, 3]*(1-1e-10))
        self.assertEqual(audit['count'], 2)
        expanded = s.audit(.25*m[:, 3].min(), m[-1, 3]*(1-1e-10))
        self.assertEqual(expanded['count'], audit['count'])
        for r, expected in zip(audit['roots'], [3.0717024062236975, 3.939309043031118]):
            c = r['phase_km_s']; independent = travelling_root(m, .1612, c-1e-6, c+1e-6)
            self.assertAlmostEqual(c, expected, places=9)
            self.assertAlmostEqual(c, independent['phase_km_s'], places=9)

    def test_close_pair_not_merged(self):
        m = np.loadtxt(io.StringIO(subprocess.check_output([FIXTURE, '--dump-fixture'], text=True)))
        audit = BoundarySystem(m, .392).audit(3.93763, 3.93774, (1e-4, 5e-5))
        self.assertEqual(audit['count'], 2)
        np.testing.assert_allclose([r['phase_km_s'] for r in audit['roots']],
                                   [3.937656639506221, 3.937716231795848], rtol=0, atol=2e-9)

    def test_difficult_warmup_and_cutoff_approach(self):
        m = np.loadtxt(ROOT/'tests/fixtures/rayleigh/crust-cutoff-warmup.txt')
        audit = BoundarySystem(m, .1612).audit(.25*m[:, 3].min(), m[-1, 3]*(1-1e-10))
        self.assertEqual(audit['count'], 2)
        for r in audit['roots']:
            c = r['phase_km_s']
            # At this difficult model's zeros, the old complex determinant phase
            # becomes ill-conditioned. Check its original matching matrix itself,
            # without weakening that solver's phase guard or its root tolerance.
            self.assertGreater(np.abs(c-m[:, 3:]).min(), 1e-5)
            residuals = []
            for offset in [-1e-5, 0., 1e-5]:
                singular = np.linalg.svd(boundary_matrix(m, .1612, c+offset), compute_uv=False)
                residuals.append(singular[-1]/singular[0])
            self.assertLess(residuals[1], 1e-10)
            self.assertLess(residuals[1], 1e-3*min(residuals[0], residuals[2]))
        m = np.loadtxt(ROOT/'tests/fixtures/rayleigh/crust-cutoff-formal.txt')
        v = 4.43921
        vp = .9409+2.0947*v-.8206*v**2+.2683*v**3-.0251*v**4
        rho = 1.6612*vp-.4721*vp**2+.0671*vp**3-.0043*vp**4+.000106*vp**5
        m[-1, 2:] = [rho, v, vp]
        audit = BoundarySystem(m, .1612).audit(.25*m[:, 3].min(), v*(1-1e-12))
        self.assertEqual(audit['count'], 3)
        self.assertAlmostEqual(audit['roots'][-1]['phase_km_s'], 4.439194324297, places=9)


if __name__ == '__main__':
    FIXTURE = sys.argv.pop(1)
    unittest.main()
