export const meta = {
  name: 'quality-fix-failures',
  description: 'Diagnose and fix reference-quality tests that fail at runtime: distinguish a real test bug (wrong reference call / misaligned data / wrong expected layout / miscalibrated bound) from an honest gam-vs-reference divergence, and fix the test bugs in place',
  phases: [{ title: 'FixFailures', detail: 'one agent per failing test: read its failure log, diagnose, fix' }],
}

// Failing tests (local R-backed sweep) + where their full failure log lives.
const FAILS = [
  { t: 'quality_vs_gamlss_gaussian_location_scale',
    hint: 'R body failed: "Error in data[match(names(newdata), names(data))]". The gamlss predict(newdata=) call is malformed. Fix the R body to obtain fitted mu and sigma correctly (e.g. fitted(m,"mu")/fitted(m,"sigma") on the training data, or a correctly-formed predict). Compare gam mu and sigma to gamlss.' },
  { t: 'quality_vs_survival_location_scale_lognormal',
    hint: 'panicked at line 108 — diagnose whether it is a harness/R error or a real assertion (gam vs survival::survreg lognormal AFT). Fix any test-side issue; if gam genuinely diverges with a correct comparison, keep the assertion but document it as an honest divergence.' },
  { t: 'quality_vs_scam_monotone_baseline',
    hint: 'assertion at line 335 — gam monotone smooth vs scam bs="mpi". Check the comparison is correct (same data, same monotone target, fitted values aligned) and the bound is principled. Fix test bugs; keep an honest divergence if the comparison is sound.' },
  { t: 'quality_vs_vgam_multinomial_softmax',
    hint: 'assertion at line 264 — gam multinomial softmax vs VGAM vglm(multinomial). Verify class-probability alignment (reference category, column order) and the bound. Fix misalignment; keep honest divergence if sound.' },
  { t: 'quality_vs_mass_ordinal_polr',
    hint: 'assertion at line 354 — gam continuation-ratio vs VGAM sratio (despite the filename, it uses VGAM). Verify the ordinal parameterization matches (cumulative vs continuation-ratio vs stopping-ratio) and category/threshold alignment. Fix mismatches; keep honest divergence if sound.' },
  { t: 'quality_vs_betareg_beta_logit',
    hint: 'panicked at line 98 col 77 — a panic in TEST code (likely a missing column lookup `col["..."]`, an unwrap, or a parse), NOT an assertion. Read line 98 and fix the test-code bug so it runs, then ensure it compares gam beta regression to betareg correctly.' },
  { t: 'quality_vs_flexsurv_weibull_aft',
    hint: 'assertion at line 142: "Weibull beta = [time0, time1, trt] expected length 3, got 4". The test wrongly assumes gam exposes 3 coefficients; gam\'s Weibull AFT design actually has 4 (e.g. intercept + 2 time-basis cols + trt). Fix the test to locate the trt (covariate) coefficient robustly by its design column rather than assuming a fixed length, then compare to survival::survreg/flexsurv weibull. Read src to confirm the coefficient layout.' },
]

const GUIDE = `
Read FIRST: the FULL failure log at /tmp/vr_<TEST>.log (use the Read tool on that exact path — it has the
panic message, stderr from R, and any emitted metrics), then the test file tests/<TEST>.rs, then
src/test_support/reference.rs (harness) and any src/ you need to confirm gam's API / coefficient layout.

Decide the root cause:
  (A) TEST BUG — wrong/malformed reference call (R/Python), data not fed identically, wrong expected
      shape/layout, a panic in test code (bad index/unwrap/parse), reference category/threshold misalignment,
      or a tolerance so wrong it is meaningless. FIX it in place with Edit so the test is correct and runs.
  (B) HONEST DIVERGENCE — the comparison is correct (identical data, right mature reference call, right
      quantity aligned, principled bound) and gam genuinely differs from the reference. Then DO NOT weaken the
      bound and DO NOT modify gam source: leave the assertion failing, but add a one-line comment documenting
      that this is a measured, expected divergence and why.

Rules: never weaken a bound to force a pass; never edit gam source (only the test); identical data to both
engines; correct mature comparator call; principled bound with a one-line justification; no banned patterns
(no \`let _ =\`, no allow(dead_code/unused), no #[cfg(feature)], no env::var, no _-prefixed fn params, no
black_box); every import used; the test MUST compile. Do NOT run cargo/python/R (no compiling). Only Edit the
test file.
COMMIT FREQUENTLY: after EVERY Edit, commit the file you just touched — \`git add <this exact file path>\` then
\`git commit -m 'wip: fix failure'\`. Commit even if it does NOT compile yet, even mid-edit / breaking; we want
very frequent small commits (after every single edit; never more than ~5 minutes uncommitted). Stage ONLY this
file by explicit path — NEVER \`git add -A\`/\`-u\`. Retry on \`.git/index.lock\`. Do NOT push.
Reply with: root cause (A or B), what you changed, and whether it should now pass or honestly diverge.`

const results = await pipeline(
  FAILS,
  (f) => agent(
    `Diagnose and fix the failing reference-quality test: ${f.t}\nKnown symptom: ${f.hint}\n${GUIDE}`,
    { phase: 'FixFailures', label: `fix:${f.t.replace(/^quality_vs_/, '')}` },
  ).then(text => ({ test: f.t, text })),
)

const done = results.filter(Boolean)
log(`diagnosed+fixed ${done.length}/${FAILS.length} failing tests`)
return { handled: done.length, tests: done.map(d => d.test) }
