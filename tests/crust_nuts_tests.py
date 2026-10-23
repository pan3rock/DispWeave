"""Acceptance must reject stuck chains and numerical failures without hiding them."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_crust_nuts import prior_statistics, diagnostics


class CrustDiagnostics(unittest.TestCase):
    def config(self):
        return dict(model=dict(depth=[0.,1.,4.],vs=[3.,4.,5.]),
            inversion=dict(lower_vs=[2.5,3.5,4.5],upper_vs=[3.5,4.5,5.5],starts_vs=[[3.,4.,5.]]*4),
            nuts=dict(warmup=100,draws=2000,prior_sigma_vs=[.1,.1,.1]))

    def traces(self):
        t=np.tile(np.array(['0','1','0','.1','.95','0','valid','15','4','0'],dtype='U32'),(2100,1))
        t[:100,0]='1'
        return [t.copy() for _ in range(4)]

    def test_prior_half_normal_moments(self):
        config=self.config();config['inversion']['lower_vs']=[3.,4.,5.]
        config['inversion']['upper_vs']=[3.8,4.8,5.8]
        p=prior_statistics(config)[0]
        # Upper bound is eight sigma; remaining tail mass is below 1e-14.
        self.assertAlmostEqual(p['mean'],3+.1*np.sqrt(2/np.pi),places=6)
        self.assertAlmostEqual(p['sd'],.1*np.sqrt(1-2/np.pi),places=6)
        self.assertGreater(p['quantiles'][0],3.)
        self.assertLess(p['quantiles'][2],3.8)

    def test_independent_vs_stuck_chains_and_numerical_failures(self):
        config=self.config();rng=np.random.default_rng(173)
        x=rng.normal(config['model']['vs'],.1,(4,2000,3));traces=self.traces()
        good=diagnostics(config,x,traces)
        self.assertTrue(good['accepted'])
        traces[0][111,6]='outside_support'
        boundary=diagnostics(config,x,traces)
        self.assertTrue(boundary['accepted'])
        self.assertEqual(boundary['sampler'][0]['production']['support_rejections'],1)
        # A warmup failure is retained but does not falsely label production as failed.
        traces[0][0,6]='forward_failure';warmup=diagnostics(config,x,traces)
        self.assertTrue(warmup['accepted'])
        self.assertEqual(warmup['sampler'][0]['warmup']['invalid_evaluations'],1)
        traces[0][110,6]='forward_failure';bad=diagnostics(config,x,traces)
        self.assertFalse(bad['accepted']);self.assertFalse(bad['numerical_passed'])
        self.assertEqual(bad['sampler'][0]['production']['invalid_evaluations'],1)
        stuck=x*.01+np.array(config['model']['vs'])*.99
        stuck[:2]+=.1;stuck[2:]-=.1
        bad=diagnostics(config,stuck,self.traces())
        self.assertFalse(bad['mixing_passed']);self.assertFalse(bad['accepted'])
        self.assertGreater(bad['maximum_rhat'],1.1)
        with self.assertRaises(ValueError):diagnostics(config,x[:3],self.traces()[:3])
        changed=self.traces();changed[0][110,3]='.2'
        with self.assertRaises(ValueError):diagnostics(config,x,changed)
        outside=x.copy();outside[0,0,0]=2.5
        with self.assertRaises(ValueError):diagnostics(config,outside,self.traces())


if __name__=='__main__':unittest.main()
