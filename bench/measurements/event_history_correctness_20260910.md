# Event-history correctness checkpoint, 2026-09-10

The implementation checkpoint is `32d2b315a`, followed by centring cleanup and
the sixteen-channel sensitivity memory guard. This is not a completed joint
multimodal model or a certified release.

All compilation and test execution took place on MSI using the existing warm
target, with at most four build workers. Local work consisted of source edits,
inspection, and Git operations.

## Executed checks

* Python boundary tests: **21 passed**, most recent runtime **0.33 s**. These
  execute the actual argument conversion with a recording native boundary;
  they are not native model fits.
* Final focused Rust build: **55.53 s**. The three objective regressions then
  **passed in 6.11 s**: total reference-law derivatives, third/fourth directional
  derivatives, and rejection of unresolved added-factor curvature. The first
  regression also checks the centring snapshot's serialization round trip.
* Final focused Rust numerical/contract run: **31 passed, 0 failed in 29.95 s**,
  including backward interpolation, finite differences, risk masks, reference
  endpoints, and finite combined log-rate derivatives. **16 broader fit tests
  were excluded**. Earlier broad attempts reached their runtime bounds; this
  final result is a completed focused run, not a complete-suite pass.
* Independent static-frailty survival check: reference survival **0.164118**,
  continuous-time target **0.164119**. Maximum log-normalizer discrepancy from
  the independently inverted Laplace transform was **5.69e-6**.
* Reference time refinement at 36, 72, and 144 intervals gave absolute survival
  errors **4.1093e-5**, **1.0289e-5**, and **2.5732e-6**, respectively.

## Unresolved results

The high-variance static-frailty example still exposes latent-grid integration
error. At event time 0.1, a 17-point-per-axis calculation gave loading curvature
approximately **-0.45573**, versus **-0.37349** from independent integration.
Automatic differentiation agreed with finite differences of the computed
filter. Increasing the order could lose positivity. The production curvature
check now rejects this unresolved calculation; this does not solve the grid's
representation limit.

Combined `cargo check` for `gam-pyffi` and `gam-cli` reached its **120 s** bound;
a cached retry reached its **90 s** bound. Neither is a successful integration
check. No fresh native Python wheel, complete end-to-end fit suite, external
calibration study, or biobank-scale benchmark was completed.

The directional-profile product and final Laplace rank criterion remain
approximations, including a nonregular boundary. Their present implementation
does not establish exact marginal evidence or globally optimal structure.

The positive-signature decoder, joint modalities, learned entry/ascertainment
law, disease-triggered state dynamics, and structured state inference are
specified in `docs/latent-signatures.md` but are not implemented by this
checkpoint. The serving handle still retains the training cohort; serializing
the centring snapshot is not a complete deployable-model serialization format.
