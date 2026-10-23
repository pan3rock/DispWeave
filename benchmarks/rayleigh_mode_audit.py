#!/usr/bin/env python3
"""Independent, validation-only P/SV matching and numerical contour root count.

Finite-layer cosh and sinh(q*z)/q potentials remove the travelling-wave basis
degeneracy at internal P/S critical speeds. No invLM code or search candidates
are used. Contour refinement supplies numerical evidence, not an interval-
arithmetic proof of root completeness or a production absence classification.
"""
import numpy as np


class BoundarySystem:
    def __init__(self, model, frequency):
        self.model = m = np.asarray(model, float)
        if (m.ndim != 2 or m.shape[1] != 5 or len(m) < 1 or not np.isfinite(m).all()
                or m[0, 1] != 0 or np.any(np.diff(m[:, 1]) <= 0)
                or np.any(m[:, 2:] <= 0) or np.any(m[:, 4]**2 <= 4*m[:, 3]**2/3)
                or not np.isfinite(frequency) or frequency <= 0):
            raise ValueError('Need a finite, stable solid model and positive frequency')
        self.omega = 2*np.pi*frequency
        self.mu = m[:, 2]*m[:, 3]**2
        self.half_h = np.diff(m[:, 1])/2
        self.cache = {}

    def matrix(self, phase, halfspace_s=None):
        if not np.isfinite(phase) or phase == 0:
            raise ValueError('Nonfinite or zero phase velocity')
        m = self.model; n = len(m); k = self.omega/phase; mu = self.mu
        d = 2*mu*k*k-m[:, 2]*self.omega**2
        p = np.sqrt((k*k-(self.omega/m[:, 4])**2).astype(complex))
        s = np.sqrt((k*k-(self.omega/m[:, 3])**2).astype(complex))
        if halfspace_s is not None:
            s[-1] = halfspace_s
        a = np.zeros((4*n-2, 4*n-2), complex)
        # State is [ux/i, uz, txz/i, tzz]; multiply SV potential amplitudes by i.
        # Thus the matrix is real on the trapped real-c axis, across layer speeds.
        top = np.empty((n-1, 4, 4), complex); bottom = np.empty_like(top)
        for shear, q in [(False, p[:-1]), (True, s[:-1])]:
            v = q*self.half_h; scale = np.abs(q.real)*self.half_h
            ep, em = np.exp(v-scale), np.exp(-v-scale)
            c = (ep+em)/2
            t = np.empty_like(q); small = np.abs(v) < 1e-4
            t[small] = self.half_h[small]*(1+v[small]**2/6+v[small]**4/120
                                          +v[small]**6/5040)*np.exp(-scale[small])
            t[~small] = (ep[~small]-em[~small])/(2*q[~small])
            for target, direction in [(top, -1), (bottom, 1)]:
                for j, (f, fp) in enumerate([(c, direction*q*q*t), (direction*t, c)]):
                    fields = ([-fp, -k*f, -d[:-1]*f, -2*mu[:-1]*k*fp] if shear
                              else [k*f, fp, 2*mu[:-1]*k*fp, d[:-1]*f])
                    target[:, :, 2*int(shear)+j] = np.array(fields).T
        halfspace = np.array([[k, s[-1]], [-p[-1], -k],
                             [-2*mu[-1]*k*p[-1], -d[-1]], [d[-1], 2*mu[-1]*k*s[-1]]])
        for i in range(n):
            t = top[i] if i < n-1 else halfspace
            if i == 0: a[:2, :t.shape[1]] = t[2:]
            else: a[4*i-2:4*i+2, 4*i:4*i+t.shape[1]] = -t
            if i < n-1: a[4*i+2:4*i+6, 4*i:4*i+4] = bottom[i]
        # Strictly positive real scalings change neither zeros nor determinant phase.
        for axis in [0, 1]:
            scale = np.max(np.abs(a), axis=axis)
            if not np.isfinite(a).all() or np.any(scale == 0):
                raise ValueError('Unresolved matching matrix')
            a /= scale[None, :] if axis == 0 else scale[:, None]
        return a

    def determinant(self, phase):
        key = complex(phase)
        if key not in self.cache:
            value, magnitude = np.linalg.slogdet(self.matrix(phase))
            # Remove a zero-free analytic exp(2*k*H) background. Its winding is
            # zero on our contours (c=0 lies outside), while its phase can wind
            # many times on an individual low-c edge and defeat coarse sampling.
            exponent = 2*(self.omega/phase)*self.model[-1, 1]
            value *= np.exp(-1j*complex(exponent).imag)
            self.cache[key] = complex(value), float(magnitude-complex(exponent).real)
        return self.cache[key]

    def trapped_count(self, max_segments=2048):
        """Full physical interval using the halfspace-S uniformizing variable.

        Independent dense LAPACK determinant, for checking the native banded LU.
        Numerical evidence, not an interval arithmetic certificate.
        """
        m = self.model
        lambda_mu = m[:, 2]*(m[:, 4]**2-m[:, 3]**2)
        lower = .95*np.sqrt((3-np.sqrt(5))*np.minimum(self.mu, lambda_mu).min()/m[:, 2].max())
        high = np.sqrt(1-(lower/m[-1, 3])**2); cache = {}; records = []
        def phase(t):
            if t not in cache:
                c = m[-1, 3]*np.sqrt(1-complex(t)**2); k = self.omega/c
                a = self.matrix(c, halfspace_s=k*t)
                sign, _ = np.linalg.slogdet(a)
                if not np.isfinite(sign) or sign == 0: raise ValueError('Singular contour')
                cache[t] = sign*np.exp(-1j*(2*k*m[-1, 1]).imag)
            return cache[t]
        for t in [0., high]:
            c = m[-1, 3]*np.sqrt(1-t*t); k = self.omega/c
            singular = np.linalg.svd(self.matrix(c, halfspace_s=k*t), compute_uv=False)
            if singular[-1]/singular[0] < 1e-12: raise ValueError('Unresolved endpoint')
        agreed = None
        for height in [.02, .01]:
            n = 64; previous = None
            while n <= max_segments:
                x = np.unique(np.r_[np.linspace(0, high, n+1), np.geomspace(1e-12, high, n//2+1)])
                y = np.minimum(height, .25*x+1e-10); right = min(height, .25*high+1e-10)
                points = np.r_[x-1j*y, high+1j*np.linspace(-right, right, n//2+1),
                               x[::-1]+1j*y[::-1], 1j*np.linspace(1e-10, -1e-10, 17)]
                values = np.array([phase(t) for t in points])
                delta = np.angle(np.roll(values, -1)/values); winding = delta.sum()/(2*np.pi)
                current = dict(count=int(round(winding)), maximum_phase_increment=float(abs(delta).max()),
                               integer_error=float(abs(winding-round(winding))), height=height, segments=n)
                records.append(current)
                if (previous is not None and current['count'] == previous['count'] and current['count'] >= 0
                        and max(current['maximum_phase_increment'], previous['maximum_phase_increment']) < np.pi/4
                        and max(current['integer_error'], previous['integer_error']) < 1e-8):
                    if agreed is not None and agreed != current['count']: raise ValueError('Contour heights disagree')
                    agreed = current['count']; break
                previous = current; n *= 2
            else: raise ValueError('Full-interval count budget exhausted')
        return dict(count=agreed, lower_speed=float(lower), evaluations=len(cache), contours=records)

    def sign(self, phase):
        value, _ = self.determinant(float(phase))
        if not np.isfinite(value) or abs(value.imag) > 1e-9:
            raise ValueError('Reference determinant lost its real-axis phase')
        return np.sign(value.real)

    def root(self, lower, upper):
        left, right = self.sign(lower), self.sign(upper)
        if not left*right < 0:
            raise ValueError('Reference root must have a strict sign bracket')
        bracket = [float(lower), float(upper)]
        for _ in range(80):
            middle = (lower+upper)/2
            if upper-lower <= 2e-12 or middle in (lower, upper): break
            sign = self.sign(middle)
            if sign == 0: lower = upper = middle; break
            if sign != left: upper = middle
            else: lower = middle
        phase = (lower+upper)/2
        singular = np.linalg.svd(self.matrix(phase), compute_uv=False)
        return dict(phase_km_s=float(phase), bracket=bracket,
                    relative_singular_value=float(singular[-1]/singular[0]))

    def contour(self, lower, upper, height, segments):
        if (not 0 < lower < upper < min(self.model[-1, 3:5]) or height <= 0
                or not np.isfinite([lower, upper, height]).all()
                or segments < 16 or segments % 4):
            raise ValueError('Contour must avoid zero and halfspace branch points')
        # A uniform vertical mesh cannot resolve a root approaching the halfspace
        # branch point. Cluster symmetrically at the real axis, retaining y=0.
        positive = np.geomspace(height*1e-12, height, segments//8+1)
        vertical = np.unique(np.r_[-positive[::-1], 0., positive,
                                    np.linspace(-height, height, segments//4+1)])
        points = np.r_[np.linspace(lower, upper, segments, endpoint=False)-1j*height,
                       upper+1j*vertical,
                       np.linspace(upper, lower, segments, endpoint=False)+1j*height,
                       lower-1j*vertical]
        values = np.array([self.determinant(c)[0] for c in points])
        if not np.isfinite(values).all() or np.any(values == 0):
            raise ValueError('Singular or nonfinite contour evaluation')
        changes = np.angle(np.roll(values, -1)/values)
        winding = float(changes.sum()/(2*np.pi)); rounded = int(round(winding))
        return dict(segments=segments, evaluations=len(points), height=height,
                    winding=winding, count=rounded, integer_error=abs(winding-rounded),
                    maximum_phase_increment=float(np.abs(changes).max()))

    def count(self, lower, upper, heights=(.02, .01), initial_segments=128, max_segments=8192):
        records = []; counts = []
        for height in heights:
            previous = None; segments = initial_segments
            while segments <= max_segments:
                result = self.contour(lower, upper, height, segments); records.append(result)
                if (previous is not None and result['count'] == previous['count']
                        and result['count'] >= 0
                        and max(result['integer_error'], previous['integer_error']) < 1e-8
                        and max(result['maximum_phase_increment'], previous['maximum_phase_increment']) < np.pi/4):
                    counts.append(result['count']); break
                previous = result; segments *= 2
            else: raise ValueError('Contour refinement did not converge')
        if not counts or len(set(counts)) != 1:
            raise ValueError('Contour heights disagree; complex zeros or unresolved count')
        return dict(count=counts[0], lower=float(lower), upper=float(upper), contours=records,
                    interpretation='Converged floating-point contour count on this finite rectangle; not a rigorous global absence certificate')

    def audit(self, lower, upper, heights=(.02, .01), initial_segments=128):
        count = self.count(lower, upper, heights, initial_segments)
        segments = 512
        while segments <= 16384:
            grid = np.linspace(lower, upper, segments+1)
            signs = np.array([self.sign(c) for c in grid]); roots = []
            if np.any(signs == 0):
                raise ValueError('Scan hit an exact zero; use a shifted interval')
            for i in np.flatnonzero(signs[:-1]*signs[1:] < 0):
                roots.append(self.root(grid[i], grid[i+1]))
            if len(roots) == count['count']: break
            segments *= 2
        else: raise ValueError('Real root isolation disagrees with contour count')
        if any(r['relative_singular_value'] > 1e-9 for r in roots):
            raise ValueError('Root residual failed independent boundary matching')
        return dict(**count, roots=roots, scan_segments=segments)


if __name__ == '__main__':
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='index depth rho Vs Vp table')
    parser.add_argument('--frequency', type=float, required=True)
    parser.add_argument('--lower', type=float)
    parser.add_argument('--upper', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); model = np.loadtxt(args.model, ndmin=2)
    system = BoundarySystem(model, args.frequency)
    result = system.audit(args.lower if args.lower is not None else .25*model[:, 3].min(),
                          args.upper if args.upper is not None else model[-1, 3]*(1-1e-10))
    result.update(model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), frequency_hz=args.frequency)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Numerical count={result['count']}; roots={[r['phase_km_s'] for r in result['roots']]}")
