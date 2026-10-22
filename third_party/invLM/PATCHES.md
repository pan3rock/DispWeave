# DispWeave local changes to invLM

Base commit: `86ad6fa5aaec3f44c6ddcac3d69b27a2f13c2e88`.
All files except `src/dispersion.cpp` are copied without modification.
`provenance.json` preserves both original and local file hashes.
The original invLM working tree is unchanged.

## Rayleigh search lower bound

Date: 2026-10-21.

The original `get_samples()` starts coarse sampling at `0.8 * vs_min_` and
inserts a Newton estimate of the surface-layer Rayleigh velocity. For a solid
halfspace with Vs=0.6, Vp=0.72, rho=2, this can miss the root
c=0.449352742970114 km/s: the Newton sample lies just on the same side of the
root as the other samples. It is not a physical cutoff.

The local change starts coarse sampling at `0.5 * vs_min_`.
For a stable isotropic elastic halfspace, Vp/Vs > sqrt(4/3). Its physical
Rayleigh solution has c/Vs greater than approximately 0.689; thus 0.5 Vs lies
below this halfspace limit, including auxetic solids. This is a bracket change,
not an approximate replacement for a missing phase velocity. It does not
constitute a completeness proof for general layered spectra.

Tests cover independent halfspace cubic roots at ratios 1.155, 1.2, 1.41, 1.73,
3 and 10, at 0.001, 1 and 1000 Hz. Existing layered regressions, independent
P/SV boundary matching and layer-subdivision checks also run after this change.
No sensitivity kernels, LM, HMC or RJMCMC code was included in M1.

## M2 kernel import

The original `dispersion_kernel.cpp`, `dunkin_derivative.cpp`, their headers and
`model.hpp` are now imported from the same base commit without modifications.
Validation and separate root/gradient status handling live in DispWeave adapters.
No upstream kernel tolerance or root algorithm was changed in M2.

## Trapped Rayleigh search domain (2026-10-22)

For a fast surface layer over a slower halfspace, the surface-layer Newton
estimate and the samples around minimum Vs can lie at or above halfspace Vs.
The upstream real secular continuation can then produce candidates outside
its trapped-wave domain. The DispWeave adapter rejected these candidates,
but they could contaminate a search that also requested higher modes.

Rayleigh sampling now removes nonfinite/nonpositive points and points at or
above halfspace Vs before secular evaluation. This is a domain restriction,
not an absence certificate or a substitute phase velocity. The lower bracket,
interior sampling, root refinement, kernels and Love search are unchanged.
Saved RJ regression models cover an unresolved fast cap and an independently
bracketed near-cutoff trapped root, using Dunkin and fast-scalar backends.
See `docs/forward-domain-validation.md` for the broader audit and chain replay.
