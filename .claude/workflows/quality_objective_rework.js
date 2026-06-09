export const meta = {
  name: 'quality-objective-rework',
  description: 'Rework every reference-quality test to assert OBJECTIVE quality (truth recovery / predictive accuracy / calibration / constraints) instead of "gam reproduces the reference tool\'s fitted output"; the mature tool becomes a match-or-beat baseline on the objective metric',
  phases: [
    { title: 'Discover', detail: 'list all quality test files' },
    { title: 'Rework', detail: 'one agent per test rewrites its assertions to objective quality' },
  ],
}

const GUIDE = `
PRINCIPLE (non-negotiable): a quality test must assert gam's OBJECTIVE quality, NEVER that gam reproduces a
reference tool's fitted output. "We do the same thing as mgcv/gamlss/lifelines" is NOT a quality claim — matching
another tool's noisy fit proves nothing (it could be wrong, or both overfit alike). Rewrite the test's
pass/fail assertion(s) to an objective metric, and demote the mature tool to a BASELINE TO MATCH-OR-BEAT on
that metric (or drop it from the assertion if a pure objective metric suffices).

Read FIRST: src/test_support/reference.rs (harness) and the test file. The harness API is unchanged
(Column/run_r/run_python/relative_l2/rmse/max_abs_diff/pearson). You may still COMPUTE the reference and print
rel_l2 with eprintln! for context — just don't make "close to reference" the pass criterion.

Choose the right objective assertion for THIS test:
  1. TRUTH RECOVERY (data generated from a known function/parameters — most synthetic tests): assert
     RMSE(gam_fit, truth) <= a principled bar (e.g. <= noise sigma, or a small fraction of signal range).
     If the reference is also fit, additionally assert gam's error <= reference's error * 1.10 (match-or-beat
     on ACCURACY). The PRIMARY claim is gam recovers the truth.
  2. PREDICTIVE ACCURACY (real data, no known truth): make a deterministic train/test split (fixed seed/index),
     fit gam on train, predict test, and assert an absolute held-out metric bar AND gam >= reference - margin:
       Gaussian -> test R^2 >= bar and RMSE <= ref_RMSE*1.1; Binomial -> AUC >= bar and >= ref_AUC - 0.02;
       Poisson/Gamma -> held-out deviance/log-loss <= ref + margin; Survival -> concordance >= bar and/or
       integrated Brier <= ref + margin. Compute the metric on gam's own predictions.
  3. CALIBRATION / UNCERTAINTY: CI empirical coverage within nominal +/- delta (on simulated truth); PIT ~uniform
     (KS statistic small); CRPS(gam) <= CRPS(reference) * 1.05.
  4. STRUCTURE / CONSTRAINTS: assert the property directly — monotone fit is monotone (successive diffs >= -eps);
     simplex closure (rows sum to 1, all >= 0); periodic seam continuity (value+derivative match at the wrap);
     survival S(t) in [0,1] and non-increasing; manifold/geodesic/Frechet axioms.
  5. OPTIMIZATION: gam's REML/penalized-likelihood at its solution >= reference's - margin (as-good-or-better).
  6. EDF / complexity: do NOT assert "edf == reference edf". At most assert gam's edf is in a sane,
     signal-appropriate range (1 < edf < k). Prefer dropping edf-matching entirely.

EXCEPTION — reference IS mathematical GROUND TRUTH: when the reference computes an exact mathematical quantity
(geomstats geodesic distance / exp-log / Frechet mean; scipy link transform / johnsonsu / boxcox; exact
brute-force LOO refits; an analytic conjugate posterior; spherical-harmonic truth), then asserting gam matches it
IS an objective accuracy claim — KEEP it (that is correctness vs ground truth, not "same as a peer tool").

Rules: never weaken a bound to force a pass; a genuine quality shortfall failing is fine. Never edit gam source —
only the test. Keep identical data to gam and any baseline. No banned patterns (no \`let _ =\`,
allow(dead_code/unused), #[cfg(feature)], env::var, _-prefixed fn params, black_box); every import used; the test
MUST still compile (verify APIs by reading src; if you add a metric, implement it with plain Rust/ndarray). One
focused #[test]. Update the module doc-comment to state the OBJECTIVE metric it asserts. Do NOT run
cargo/python/R (no compiling); only Edit the test file.
COMMIT FREQUENTLY: after EVERY Edit, commit the file you just touched — \`git add <that exact file path>\`
then \`git commit -m 'wip: <test name>'\`. Commit even if it does NOT compile yet, even mid-edit / breaking;
we want very frequent small commits (after every single edit; never more than ~5 minutes uncommitted). Stage
ONLY your file by explicit path — NEVER \`git add -A\`/\`-u\`. Retry on \`.git/index.lock\`. Do NOT push.
Reply with: which objective metric you now assert, and whether the reference remained (as baseline / as ground
truth) or was dropped from the assertion.`

phase('Discover')
const listed = await agent(
  `List every reference-quality test file. Run: ls tests/quality_vs_*.rs  (use Bash), and return the file paths.`,
  { phase: 'Discover', label: 'discover', schema: {
    type: 'object', additionalProperties: false, required: ['files'],
    properties: { files: { type: 'array', items: { type: 'string' } } },
  }, agentType: 'Explore' },
)
const files = ((listed && listed.files) || []).filter(f => f && f.endsWith('.rs'))
log(`reworking ${files.length} quality tests to objective-quality assertions`)
if (files.length === 0) return { error: 'no test files discovered' }

const results = await pipeline(
  files,
  (file) => agent(
    `Rework the reference-quality test at ${file} so its assertions measure OBJECTIVE quality, not closeness to a reference tool's output.\n${GUIDE}`,
    { phase: 'Rework', label: `rework:${file.replace(/^tests\/quality_vs_/, '').replace(/\.rs$/, '')}` },
  ).then(text => ({ file, text })),
)

const done = results.filter(Boolean)
log(`reworked ${done.length}/${files.length} tests to objective-quality`)
return { reworked: done.length, files: done.map(d => d.file) }
