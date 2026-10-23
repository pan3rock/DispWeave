#!/usr/bin/env python3
"""Validation-only P/SV boundary matching, independent of invLM/Dunkin.

Finite-layer waves are anchored at the endpoint towards which they grow,
avoiding exponentially growing propagation factors. The full displacement /
traction continuity system is assembled explicitly. This costly reference is
not a production backend or a certificate of modal completeness.
Requires NumPy only; use OPENBLAS_NUM_THREADS=1 for these small matrices.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def boundary_matrix(model, frequency, phase):
    omega = 2*np.pi*frequency;k = omega/phase;n = len(model)
    matrix = np.zeros((4*n-2, 4*n-2), complex)
    for i, (_, depth, rho, vs, vp) in enumerate(model):
        mu = rho*vs**2;lam = rho*vp**2-2*mu
        p = np.sqrt(complex(k*k-(omega/vp)**2))
        s = np.sqrt(complex(k*k-(omega/vs)**2))
        vertical = np.array([-p, -s, p, s])
        field = np.empty((4, 4), complex)
        for j, z in enumerate(vertical):
            if j % 2 == 0:
                field[:, j] = [1j*k, z, 2j*mu*k*z, (lam+2*mu)*z*z-lam*k*k]
            else:
                field[:, j] = [-z, 1j*k, -mu*(z*z+k*k), 2j*mu*k*z]
        if i == n-1:
            top = field[:, :2]  # Downward-decaying halfspace waves only.
        else:
            h = model[i+1, 1]-depth
            top = field*np.r_[np.ones(2), np.exp(-vertical[2:]*h)]
            bottom = field*np.r_[np.exp(vertical[:2]*h), np.ones(2)]
            matrix[2+4*i:6+4*i, 4*i:4*i+4] = bottom
        if i == 0:
            matrix[:2, :top.shape[1]] = top[2:]
        else:
            matrix[2+4*(i-1):6+4*(i-1), 4*i:4*i+top.shape[1]] = -top
    # Positive row scaling leaves the zeros and determinant sign unchanged.
    scale = np.max(np.abs(matrix), axis=1)
    if not np.isfinite(matrix).all() or (scale == 0).any():
        raise ValueError('Degenerate reference matching matrix')
    return matrix/scale[:, None]


def root(model, frequency, lower, upper):
    def phase(c):
        value, _ = np.linalg.slogdet(boundary_matrix(model, frequency, c))
        # Anchoring upward waves at the bottom multiplies their columns by
        # exp(-(p+s)*h). Remove its known phase before comparing determinant
        # signs. The remaining constant phase depends on the wave basis.
        omega = 2*np.pi*frequency;k = omega/c
        p = np.sqrt((k*k-(omega/model[:-1, 4])**2).astype(complex))
        s = np.sqrt((k*k-(omega/model[:-1, 3])**2).astype(complex))
        return value*np.exp(1j*np.sum((p+s).imag*np.diff(model[:, 1])))
    reference_phase = phase(lower)
    def sign(c):
        value = phase(c)/reference_phase
        if not np.isfinite(value) or abs(value.imag) > 1e-3:
            raise ValueError('Invalid/ill-conditioned reference determinant phase')
        return np.sign(value.real)
    lo_sign = sign(lower)
    if lo_sign*sign(upper) >= 0:
        raise ValueError('Reference interval does not bracket a zero')
    for _ in range(60):
        mid = (lower+upper)/2
        if upper-lower < 2e-12 or mid == lower or mid == upper:
            break
        value = sign(mid)
        if value == 0:
            lower = upper = mid;break
        if value != lo_sign:
            upper = mid
        else:
            lower = mid
    phase = (lower+upper)/2
    singular = np.linalg.svd(boundary_matrix(model, frequency, phase), compute_uv=False)
    return dict(phase_km_s=phase, smallest_relative_singular_value=float(singular[-1]/singular[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='index depth_km rho_g_cm3 Vs_km_s Vp_km_s')
    parser.add_argument('--frequency', type=float, required=True)
    parser.add_argument('--bracket', type=float, nargs=2, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();model = np.loadtxt(args.model, ndmin=2)
    if model.shape[1] != 5 or not np.isfinite(model).all() or (model[:, 2:] <= 0).any() or (np.diff(model[:, 1]) <= 0).any():
        raise ValueError('Invalid solid model')
    if args.frequency <= 0 or not np.isfinite(args.frequency) or any(not 0 < lo < hi < model[-1, 3] for lo, hi in args.bracket):
        raise ValueError('Need positive frequency and trapped-wave brackets')
    result = dict(model=str(args.model), frequency_hz=args.frequency,
        roots=[root(model, args.frequency, *b) for b in args.bracket])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
