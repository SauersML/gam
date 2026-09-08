# September 7 acceptance recovery receipts

The current Rust-library acceptance targets passed on MSI:

| Target | Tests passed | Contract |
| --- | ---: | --- |
| `gam-identifiability::structural_coordinates_2748` | 4 | Priority-directed spanning reduction, preservation of structural coordinates and their coefficients, independent reduction of another block, and rejection of a malformed coordinate declaration. |
| `gam-solve::measured_span_2612` | 4 | The strict one-observation cutoff, empty span for bounded curvature, disagreement with the penalty kernel in both directions, and invariance under nonorthogonal shear with a wrong-metric negative control. |
| `gam-solve::public_library_surface_2829` | 2 | Actual external owned/borrowed generic design fitting and explicit metric selection. |
| `gam-sae::construction::tests_trace_whitening_2333` | 5 | Four complete symmetric-basis deflation-fold checks plus actual row-whitened Trace/dense adjoint parity. |

Eight historically deleted identities were reintroduced through the current
public APIs, and the historical Trace consumer pin was reintroduced through the
current production consumer. No compatibility wrappers or unused production
helpers were added.

The measured-span target built warm in **3.70 seconds** and ran four tests in
**0.022 seconds**. The Trace target ran five tests in **0.05 seconds** and recorded
one genuinely spectrally deflated row. Its joint adjoint gap was
**2.55440113505756e-12** and coordinate gap **3.979039320256561e-13**, at adjoint
magnitude **199.93419830762554**. The comparison remains
`1e-12 * (1 + magnitude)` and repeat output must be bit-identical.

Two fixture failures were useful and were repaired before counting passes:

- The first Trace state had no spectral deflation and failed its branch-activity
  assertion. One exactly saturated softmax row supplies a genuine null logit
  direction, while the remaining rows retain both active atoms and full-rank,
  row-varying whitening. The production factorizer now activates the required
  branch without a fixture search.
- The first external fit expectation omitted the solver's declared fixed
  stabilization ridge. The corrected design is already standardized with
  orthogonal equal-norm columns, and its expected coefficients solve the exact
  equations `(6 + 1e-8) beta = [12,18]`. The assertion was strengthened to
  `1e-12`, rather than relaxed around an unexplained bias.

Logs under `bench/measurements/issue_triage_20260907/` are
`public-api-focused.log`, `public-api-final.log`, `measured-span-focused.log`, and
`trace-cached.log`. The SAE executable was freshly built with 16 code-generation
units on eight assigned CPUs in 79 seconds; subsequent tests reuse the binary.
These receipts apply to the current MSI working source, including preexisting
edits, not an assertion that a whole published commit passes.

The historical recovery issue remains open. A source walk after the structural
and Trace additions found **243 of 303** historical identities still absent
across **106 old source paths**; the four subsequent measured-span restorations
reduce that identified missing set by four. Large exact-A, logdet-adjoint,
co-collapse, and outer-invariance groups still need individual semantic mapping
and execution. Public API reachability and a green source census do not replace
those acceptance proofs. The detailed mapping and unresolved groups are recorded
in `docs/test-census-2818-recovery.md`.
