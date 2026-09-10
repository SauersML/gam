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

## Reference integration refinement

Reference acceptance now checks both time refinement and latent quadrature at
fixed parameters. It refines the larger contribution until their sum meets
the requested tolerance. The additional regression detects latent integration
error with an unchanged time grid and rejects comparisons across different
coefficient states or non-finite reference moments. The six objective checks
passed in **2.19 s**. After the finite static-rate reporting change, the final
focused suite completed with **34 passed, 0 failed in 27.86 s**; **16 broader
fit tests were excluded**.

Rank-path output now names `proposed_rate` in the data's time unit. A static
proposal is zero, avoiding an infinite logarithm in that diagnostic.

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
The replacement overlapping substring search passed **609,336** exhaustive
Unicode/overlap comparisons. On the build-script source microbenchmark it took
**206.47 ms**, compared with **548.24 ms** for the old search (**2.66×**).
The benchmark is `experiments/event_history_scanner.rs`. This is a matcher
benchmark, not a measured whole-build speedup: a complete integration check
still reached its 65-second cap, and a direct scanner run reached 55 seconds
(11.41 seconds user CPU, 0.85 seconds system CPU).

The directional-profile product and final Laplace rank criterion remain
approximations, including a nonregular boundary. Their present implementation
does not establish exact marginal evidence or globally optimal structure.

The serving handle still retains the training cohort; serializing the centring
snapshot is not a complete deployable-model serialization format. The following
joint-model work is subsequent to the older-model checkpoint above.

## Joint signature density and structured integration

Commit `7d35c4af9` added a positive signature decoder, genetic OU drive, conditional
entry regression, constant learned disease jumps, Student-t/probit/ordinal/count
measurement channels, and structured latent Laplace inference. The initial
**11 focused checks passed in 0.04 s**, following a **77 s** warm build.

The subsequent importance-integration change evaluates the complete joint density
under a mixture of its conditional Gaussian path prior and Laplace approximation.
It preserves one sampled objective for values and derivatives, carries supplied
reference sensitivities, and checks finite-variance tails at moved parameter
states. The integral and posterior moments have separate estimated-error limits;
unresolved evaluations return errors.

The final **15 focused checks passed in 1.67 s**, following a **62 s** warm build.
The raw output is `event_history_joint_20260910.txt`. These checks include:

* An eight-signature, 33-node path with two missing genetic scores: the Gaussian
  integral is one, the structured covariance matches its analytic limit, and
  4,096 importance paths recover the same integral to roundoff. This is a
  Gaussian-limit test, not a high-dimensional non-Gaussian inference benchmark.
* The structured sampling map's covariance agrees with inverse precision to
  **1e-13**, including a dense genetic border and nonsymmetric temporal blocks.
* An independent one-dimensional Student-t measurement integral gives log
  marginal **-2.45748432528629**. The 16,384-path importance estimate is
  **-2.46500002955279**, with estimated standard error **0.00423684** and effective
  sample count **12,660.65**. Its final-state mean is **1.26333024**, compared with
  **1.26809223** independently; variance is **0.48957316**, compared with
  **0.48704663**. This is one reproducible Monte Carlo realization, not a claim
  that the estimated standard errors have been externally calibrated.
* The independent reference first failed its 65-versus-129-node convergence
  check. Refining to 129 versus 257 nodes reduced log-integral disagreement to
  **2.58e-10** without relaxing the tolerance. A separate SciPy adaptive integral
  on MSI gave probability **0.08565014808261512**, reported integration error
  **2.12e-14**, consistent with that reference.
* Sampled value, gradient, and curvature agree with finite differences using
  the same bank, including supplied normalizer sensitivities. A parameter move
  violating the sufficient finite-variance tail condition is rejected, as are
  unsatisfied likelihood and posterior-moment error requests.

All builds used the existing MSI cache and four workers. An initial RNG feature
configuration triggered dependent rebuilds and reached the 95-second cap; the
dependencies now disable unused default features. The successful final check
compiled only the event-history crate. No local compilation or tests were run.

The complete reference evolution with disease jumps, learned late-entry
conditioning from that reference law, parameter fitting, automatic structure
selection, Python/CLI serving, and deployable model serialization still need
integration. The importance diagnostics do not establish external calibration,
biobank-scale performance, or exact evidence. The goal is not complete.

## Reference particles with disease histories

The subsequent reference implementation carries the same genetic OU states,
once/terminal risk sets, and nonterminal jumps as the joint density. Its fixed
event proposals are reweighted at every coefficient evaluation; derivatives
include those weights and the risk-set moments. The normalized observation
integral and posterior return the exact coefficient/reference state they used.

The first **18 joint checks passed in 1.63 s** after a **67 s** warm build. After
stabilizing the reference error calculation, the final **20 checks passed in
1.64 s**. Raw final output is `event_history_reference_20260910.txt`. The added
checks cover competing risk-set membership, horizon rejection, refusal of
excessive event steps and unresolved sampling diagnostics, genetic/jump/event-
weight derivatives, and consistency of the normalized observation integral with
its returned reference state. Rare-event log probabilities retain derivatives
at log hazard **-800**, and moment error estimates avoid multiplying underflowed
weights by overflowing squared activities.

The final Cargo build first reached its **95 s** cap. A diagnostic retry then
failed at linking with undefined cached internal Rust symbols; it reported
**69.17 s** elapsed and about **1,045,752 KiB** maximum RSS. A process observation
showed about **31 s** of Cargo work before rustc started. Replaying Cargo's exact
compiler invocation without this crate's incremental-object option, keeping all
warm dependency artifacts and restricting execution to four CPUs, completed in
**36.94 s**. The final 20-test result is from that successfully linked binary.
Neither the timeout nor the link failure is counted as a passing Cargo run.

This is a differentiated finite-step reference engine, not a resolved reference
calculation. Its event update admits at most one event per interval. Within-bank
dispersion estimates do not account for all dependence induced by the shared
normalizer. Adaptive time/particle refinement and independent reference
replicates remain required, along with the fitting, structure-learning,
entry-conditioning, serving, and calibration work listed above.
