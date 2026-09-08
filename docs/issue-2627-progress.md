# Issue 2627: verification ledger

The acceptance condition remains zero failing tests across the full Rust
workspace, gam-pyffi, Python (including slow and torch populations), reference
quality tests, and doctests. A passing subset, missing population, timeout,
deleted regression, or build failure does not satisfy that condition.

## September 8: periodic penalty regression

`periodic_bspline_terms_build_with_cyclic_penalty_and_formula_alias` requested
`BSplineIdentifiability::None` but expected the constant-function penalty to
vanish under sum-to-zero centering. The recorded MSI failure was `2 != 1`.
The production builder intentionally retains that penalty when the constant
survives (#2783); discarding it would restore intercept confounding.

The test now checks both cases without changing production behavior:

- Uncentered: two distinct penalty sources, roughness nullity one, constant
  penalty rank one, zero roughness energy on the constant vector, and positive
  constant-function energy.
- Centered: one full-rank penalty in a coefficient chart one dimension smaller.
- Existing periodic wrapping, derivatives, formula aliases, and tensor-margin
  assertions remain active.

MSI verification: **5 passed, 0 failed, 0 ignored, 0 filtered out, 0.10 seconds**.
The exact edited file was compiled with `rustc --test`, linked against the
existing `w5-target/debug/deps/libgam-d2bdd2ba4cf1842d.rlib` and its matching
`libndarray-3e2fc606731d0706.rlib`. The periodic builder's SHA-256 was identical
in the local, y2, and w5 sources:
`f4c93765f226c90803f04a347d6d16d484618d7314cc253c7b1f0f155ed35300`.
This is a focused test receipt against existing compiled dependencies, not a
fresh whole-worktree build. Output is in
`bench/measurements/issue_triage_20260907/periodic-2627-direct.log`.

## Remaining verification and failures

The attempted Cargo/nextest root integration build did not reach test execution.
After removing obsolete inherited OpenBLAS linker flags, the root scanner
reported 26 violations: four undocumented unwraps in the atlas nerve test and
22 unused underscore parameters in newly added derivative APIs. These must be
resolved before the current worktree can receive a full build verdict.

The September 8 w5 logs also contain remaining CLI, survival, model
materialization, and SAE failures and timeouts. Their selections exclude other
tests, so they are not a whole-suite denominator. The active GitHub Rust CI
run discovered during this audit is 34253714873, at 7c3c5b43b0e34823d2fd4c9d135d7cac63694676.
Its terminal results still need inspection. No full-suite pass has been proven;
the issue must remain open.

## September 8: SAE assembly and fixture repairs

The next focused MSI run reports **15 passed, 1 failed, 0 ignored, 1,252
filtered out, 2.67 seconds**. This is a targeted development population, not
a full-suite denominator. The complete output is
`bench/measurements/issue_triage_20260907/sae-2627-final.log`.

Production repairs:

- The amplitude-barrier majorizer is a scalar ridge. Carry it through the
  structured smooth blocks on every layout; classifying it as a general dense
  penalty incorrectly disabled framed device data. The unused dense branch
  and return flag were removed. A new analytic-reference check verifies that
  both CPU and device operands retain the ridge across the assembly threshold.
- Parallel assembly now fills the admitted row buffers directly. Riemannian
  projection overwrites their contents, and frame projection retains the
  output-gradient buffer. The existing two-iteration regression now verifies
  unchanged allocation addresses and changed numerical contents for the row,
  gradient, and device-frame storage.
- An all-zero separation carrier induces a zero majorizer even when its
  overlap-space metric is nonzero. Do not retain that zero operator as a reason
  to refuse device assembly. Exact signed curvature remains independently owned.

Fixture repairs retain their original mathematical checks: the sphere uses
three ambient coordinates; degree-two sphere harmonics have five curved
columns; the quadratic-loss test receives a symmetric positive Gram (and now
also checks rejection of the asymmetric input); and a single softmax atom has
no active entropy-strength coordinate. The dispersion test additionally checks
active entropy, smoothness, and ARD scaling with multiple atoms.

The large-border helper formerly constructed a rank-at-most-two matrix while
requesting rank eight. It now constructs the requested rank and asserts it.
Its routing fixture provides distinct output subspaces. This moves the test
past the incorrect `512 != 2048` border: it now reaches border 2,048 and fails
at `large factored assembly must install device operands`. That remaining
device-admission defect is **not fixed** by this batch.

Validation compiled the current durable MSI `y2` source with `build.sh test
--target-dir /scratch.global/sauer354/y2-target --config
profile.test.package.gam-sae.codegen-units=16 -p gam-sae --lib --no-run`, four
build workers, and two Rayon/test workers. No local builds or tests ran.
The final rebuild took 169 seconds. Source and executable checksums are in
`bench/measurements/issue_triage_20260907/sae-2627-source-sha256.txt`.
The shared development tree contains other uncommitted changes, including a
separate exact-beta test module. Only the reviewed changes above are applied
to main; this receipt does not claim an exact-main or whole-workspace build.

## Fresh complete gam-pyffi population

Rust CI run **34253714873**, commit `7c3c5b43b`, finished its separate
gam-pyffi job **102173061164**: **124 run, 120 passed, 4 failed, 0 skipped,
12.449 seconds**. The four failures are:

- `batch_tests::circle_latent_recovers_circle_not_collapse`: Riemannian solver
  nonconvergence after 200 iterations, residual `7.094570715e-4` vs `1e-8`.
- `inference::inference_instruments::tests::matched_controls_do_not_promote_circle_on_seeded_isotropic_noise_2262`:
  Gaussian-mixture order seven did not certify; its parameter map was not
  contracting at the reported endpoint.
- `inference::inference_instruments::tests::ring_of_clusters_owns_discrete_cyclic_verdict_2262`:
  order nine failed parameter-map certification (`8.622042e-7` vs `1.490e-8`).
- `tests::shared_tangent_fit_is_output_rotation_equivariant`: coefficient
  rotation discrepancy `3.506` vs the `1e-7` assertion.

The workspace archive build succeeded and all ten Rust shards started. The
shards and both Python jobs were still running at this update. These are
additional measured failures, not a zero-failure result; #2627 remains open.
