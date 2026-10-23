# DispWeave local changes to invLM

Base commit: `86ad6fa5aaec3f44c6ddcac3d69b27a2f13c2e88`.
The DispWeave local modifications described here affect `src/dispersion.cpp`,
`include/dispersion.hpp` and the Config adapter in `regularization.hpp/.cpp`;
other imported files are unchanged by these patches.
`provenance.json` preserves both original and local file hashes.
The original invLM tree is modified only where explicitly stated below.

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

## M8a optional root refinement

`Dispersion::search` now accepts absolute and relative refinement tolerances,
with the existing defaults 1e-12 and 1e-14. The mode search, coarse brackets,
ordering and deduplication are unchanged. TOMS748 error status is checked for
Rayleigh as well as Love; a failed refinement is not an accepted phase.
`InvLMForward(method, RootAccuracy::refined)` uses zero absolute tolerance and relative
8*double_epsilon for modal ellipticity, whose ratio can amplify tiny phase errors.
The default forward/inversion path retains its old tolerances. No kernel changes.
See `docs/m8a-validation.md` for pole-side independent verification and limitations.

## Clustered Rayleigh candidate intervals (2026-10-23)

As in QEDispInv, `locate_extremum` now attempts a parabolic vertex for every
adjacent sample triple, retaining only vertices strictly inside the interval.
The previous middle-sample-extremum precondition could skip an interior vertex
even for a monotone triple, hiding a pair of sign changes. Accumulated extrema,
zero samples, root tolerances and Love root counting are preserved.
The same minimal change is applied to the local invLM repository.
DispWeave's fixed-layer LM now opts into the existing refined root tolerances,
matching joint LM; the backend's default tolerances are unchanged.

A saved crustal model, independent P/SV boundary matching, modal-prefix and
perturbation regressions demonstrate the repair for the default Dunkin backend.
FAST_SCALAR still fails this fixture; this change is not a completeness proof.
See `docs/clustered-root-repair.md` for validation, overhead and remaining issues.

## Elastic relations and explicit regularization (2026-10-23)

Imported `model_builder.hpp/.cpp` and `regularization.hpp/.cpp` from the recorded
base commit 86ad6fa (identical to these files in local invLM 5fd7f27).
ModelBuilder formulas and analytical derivatives are unchanged. From
Regularization only the native Config-dependent constructor/conversion helper
and its include/forward declarations were removed; the Settings constructor,
residuals and derivatives are unchanged. No Ceres/configuration dependency was
introduced. The DispWeave adapter exposes explicit single-weight reference,
perturbation-smooth and Vs-smooth penalties; it does not expose automatic weight
selection or interface-adaptive weighting. See `docs/material-regularization.md`.
