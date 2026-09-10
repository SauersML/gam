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

## Static-factor correction

A subsequent change represents the actual static boundary by rate zero and
integrates a fully static history on a single posterior-adapted grid. With
33 nodes per axis, added-loading curvature is **-0.3735106145** at event times
0.1, 0.5, and 0.9. The independent integral gives **-0.3734935154**. The event-time
spread is below **1e-12** and the finite-difference discrepancy below **2e-8**.
The regression completed in **0.08 s**, following a **42.05 s** warm build.

The five objective regressions subsequently **passed in 2.88 s** after a
**35.92 s** warm build. They include mixed static/dynamic derivatives and
continued rejection of unresolved near-static curvature.

After making the slow-plateau proposal an actual static factor, the censored
survival fit and PIT regression **passed in 0.84 s** (400 subjects); its
Kaplan–Meier PIT distance was **0.019**. The final focused run completed with
**33 passed, 0 failed in 27.40 s**, with 16 broader fit tests excluded.

## Unresolved results

The high-variance near-static-frailty example still exposes latent-grid integration
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

A bounded profile of the optimized root build script identified its substring
scanner as the integration-check bottleneck: about **75%** of sampled CPU time
was in `find_banned_code_fragments` and its byte comparisons. This was a
25-second, 49-Hz profile; the repository checks were not disabled.

The directional-profile product and final Laplace rank criterion remain
approximations, including a nonregular boundary. Their present implementation
does not establish exact marginal evidence or globally optimal structure.

The positive-signature decoder, joint modalities, learned entry/ascertainment
law, disease-triggered state dynamics, and structured state inference are
specified in `docs/latent-signatures.md` but are not implemented by this
checkpoint. The serving handle still retains the training cohort; serializing
the centring snapshot is not a complete deployable-model serialization format.
