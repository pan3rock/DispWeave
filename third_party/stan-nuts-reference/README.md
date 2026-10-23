# Stan NUTS algorithm reference

Repository: https://github.com/stan-dev/stan

Pinned commit: `c3db9b995dae2f2f9e785cc55c6e96ab1127cdab`.
The unmodified files under `upstream/` retain their original basenames; the
original paths and SHA-256 hashes are recorded in `provenance.json`.
`LICENSE` is the upstream BSD-3-Clause license, including attribution.

`src/nuts_multinomial.hpp` adapts the multinomial weights, biased progressive
outer selection, and three generalized U-turn checks from `base_nuts.hpp`.
Step-size bracketing and refresh after metric updates are informed by
`base_hmc.hpp` and `adapt_dense_e_nuts.hpp`, respectively.

The production library has no Stan dependency. The `nuts_stan_reference` test
compiles this exact upstream `base_nuts.hpp` with small test-only adapters in
`tests/stan_nuts_adapter/`. These provide DispWeave's target, metric, integrator
and random-number stream. This independently checks tree construction and
selection against upstream source; it does **not** independently validate
Stan automatic differentiation, Stan integrators, or Stan warmup adaptation.

Intentional differences: DispWeave reports attempted depth, treats verified
physical support rejections separately from numerical divergences, and bounds
its warmup step-size search to 50 probes. Reference replay compares candidate
states, leapfrog counts and acceptance statistics, not those depth/divergence
diagnostics. No source patch is applied to the upstream tree implementation.
