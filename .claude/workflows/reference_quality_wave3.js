export const meta = {
  name: 'reference-quality-wave3',
  description: 'Wave 3: e2e quality tests for the remaining niche gam axes (exotic links SAS/blended/latent-cloglog, exotic manifolds Poincare/Stiefel/Grassmann/SPD, frailty survival, custom family, remaining model modes) vs the best comparator',
  whenToUse: 'Close the last capability gaps after waves 1-2.',
  phases: [
    { title: 'Plan', detail: 'one planner per niche class' },
    { title: 'Author', detail: 'one agent per spec writes the test (no schema)' },
    { title: 'Review', detail: 'adversarial review + fix' },
  ],
}

const CANON = `
Read FIRST and mirror: src/test_support/reference.rs (harness) and tests/quality_vs_mgcv_gaussian_smooth.rs.
Harness: gam::test_support::reference::{Column, run_r, run_python, relative_l2, rmse, max_abs_diff, pearson,
ReferenceResult}; run_r/run_python(&[Column::new("name",&vec)], body) where the body emits via emit("key",vec);
missing tool/package => the test FAILS (no skip). gam: fit_from_formula(formula,&dataset,&FitConfig)->FitResult
(match the variant in src/solver/workflow.rs); fit.fit.beta/.edf_total(); build_term_collection_design for predict.
VERIFY every API by reading src (esp. src/types.rs for link variants, src/geometry/* for manifolds,
src/families/* for frailty/custom). Datasets under bench/datasets/ via concat!(env!("CARGO_MANIFEST_DIR"),...).

Best comparators for the niche axes:
  gamlss (SHASHo = sinh-arcsinh -> THE comparator for gam's SAS link; also many link families),
  VGAM (alternative links / multivariate), scipy.stats (johnsonsu = sinh-arcsinh; exact ground truth for link math),
  geomstats (Python: geodesics, Frechet mean, exp/log maps on Poincare/Stiefel/Grassmann/SPD -> ground truth for
  gam's manifold primitives), survival::coxph(frailty=) and coxme (frailty random effects),
  a hand-coded reference likelihood in R/Python for custom-family (re-implement the same density and compare).
For axes with NO external tool (e.g. GPU-vs-CPU equivalence, learned-topology), assert an INTRINSIC property
(CPU==GPU within tol; recovery of known truth; equivariance) and document that no head-to-head exists.

RULES: ideal test vs ideal comparator, measuring gam HONESTLY; a failing assertion is acceptable, NEVER weaken a
bound or modify gam source. Identical data to both engines; correct comparator call; principled bound w/ one-line
justification. Do NOT duplicate any existing tests/quality_vs_*.rs file. Banned patterns apply (no let _ =, no
allow(dead_code/unused), no #[cfg(feature)], no env::var, no _-prefixed fn params, no black_box silencer); each test
is its own crate (every import used, warnings=deny); MUST compile (verify APIs by reading src). panic!/println! ok in
tests/. One focused #[test] per file with a module doc-comment. Do NOT run cargo/python/R (no compiling — the
coordinator does that); only Write/Edit the test file.
COMMIT FREQUENTLY: immediately after EVERY Write/Edit, commit the file you just touched — \`git add <that exact
file path>\` then \`git commit -m 'wip: <spec-id>'\`. Commit even if it does NOT compile yet, even mid-edit /
breaking; we want very frequent small commits (after every single edit; never more than ~5 minutes uncommitted).
Stage ONLY your file by explicit path — NEVER \`git add -A\`/\`-u\`. Retry on \`.git/index.lock\`. Do NOT push.
`

const CLASSES = [
  { key: 'exotic_links', title: 'Exotic / learnable links vs gamlss SHASHo / scipy johnsonsu / VGAM',
    guidance: `gam's SAS (sinh-arcsinh) link, blended/mixture inverse-link, beta-logistic, latent-cloglog. Comparator: R gamlss family=SHASHo (sinh-arcsinh) and scipy.stats.johnsonsu for the SAS transform math (exact ground truth); VGAM for alternative links. Assert the fitted mean / link-transform agrees with the reference parameterization. Verify link names/params in src/types.rs and src/mixture_link.rs.` },
  { key: 'exotic_manifolds', title: 'Hyperbolic / Stiefel / Grassmann / SPD primitives vs geomstats',
    guidance: `gam's geometry primitives (src/geometry/poincare.rs, stiefel.rs, grassmann.rs, spd.rs): geodesic distance, exp/log maps, Frechet mean. Comparator Python geomstats (Hyperbolic/PoincareBall, Stiefel, Grassmannian, SPDMatrices) as ground truth. Assert geodesic distance / Frechet mean / exp-log roundtrip agree to tight tolerance on fixed-seed points. Document where gam exposes these (read the geometry modules for the public API).` },
  { key: 'frailty_survival', title: 'Shared-frailty survival vs survival::coxph(frailty) / coxme',
    guidance: `gam's log-normal / probit frailty (src/families/lognormal_kernel.rs) — shared group random effect in survival. Comparator R survival::coxph(Surv(t,e) ~ x + frailty(g)) or coxme. Assert frailty variance component and/or fixed-effect coefficients agree on clustered censored data (fixed-seed or real).` },
  { key: 'custom_family', title: 'User-defined custom family vs a hand-coded reference likelihood',
    guidance: `gam's custom-family path (src/families/custom_family.rs). Implement a KNOWN family (e.g. a specific GLM density) as a gam custom family AND re-implement the same negative-log-likelihood fit in R/Python (or compare to that family's built-in in mgcv/statsmodels). Assert coefficients/fitted values match the reference for the same density on identical data. Verify the custom-family API by reading the module.` },
  { key: 'remaining_modes', title: 'Remaining model modes / response geometries vs best comparator',
    guidance: `Any gam model mode not yet covered by waves 1-2: e.g. multinomial location-scale, simplex/compositional RESPONSE regression (vs compositions/DirichletReg), transformation-survival combos, marginal-slope inference. Pick the closest mature comparator; if none, assert an intrinsic correctness property + document the gap. Read src/solver/workflow.rs FitResult variants and the formula DSL to find genuinely-uncovered modes; do NOT duplicate existing files.` },
]

const SPECS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['tests'],
  properties: { tests: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['id','file','capability','reference_tool','gam_formula','data','metric','rationale'],
    properties: { id:{type:'string'}, file:{type:'string'}, capability:{type:'string'}, reference_tool:{type:'string'},
      gam_formula:{type:'string'}, data:{type:'string'}, metric:{type:'string'}, rationale:{type:'string'} },
  } } },
}

phase('Plan')
const planned = await parallel(CLASSES.map(c => () =>
  agent(`Plan e2e QUALITY tests for the gam engine, niche class: "${c.title}".\n${c.guidance}\n${CANON}\n
Read src/ to confirm the EXACT gam API for this class and the comparator's matching call. Output 2-4 concrete,
buildable specs, each with the best comparator + a principled metric/bound. Skip a spec entirely if gam does not
expose the capability (do not fabricate). Do NOT propose a file matching any existing tests/quality_vs_*.rs.`,
    { phase:'Plan', label:`plan:${c.key}`, schema: SPECS_SCHEMA, agentType:'Explore' })
    .then(r => ({ class:c.key, tests:(r&&r.tests)||[] }))
)).then(rs => rs.filter(Boolean))

const specs = planned.flatMap(p => p.tests.map(t => ({...t, class:p.class}))).filter(t => t&&t.file&&t.id)
log(`planned ${specs.length} wave-3 specs`)
if (specs.length === 0) return { error: 'no specs' }
const seen = new Set(); const uniqueSpecs = []
for (const s of specs) { let f = s.file.endsWith('.rs')?s.file:`${s.file}.rs`; if(!f.startsWith('tests/')) f=`tests/${f}`
  if (seen.has(f)) { const b=f.replace(/\.rs$/,''); let k=2; while(seen.has(`${b}_${k}.rs`))k++; f=`${b}_${k}.rs` }
  seen.add(f); uniqueSpecs.push({...s, file:f}) }

const results = await pipeline(uniqueSpecs,
  (spec) => agent(
    `Author ONE Rust integration test benchmarking a niche gam capability vs a mature comparator.
SPEC: id=${spec.id} file=${spec.file}; capability=${spec.capability}; comparator=${spec.reference_tool};
gam_formula=${spec.gam_formula}; data=${spec.data}; metric=${spec.metric}; rationale=${spec.rationale}
${CANON}
Read the harness + canonical example + the src you need, then Write ${spec.file}. Ensure it compiles (real APIs,
every import used, no banned patterns), identical data to both engines, principled un-weakened bound (one-line
justification), eprintln! the metrics. If gam lacks the capability, write nothing and say "skipped: <reason>".
Reply with the file path (or skip reason) + one-sentence summary.`,
    { phase:'Author', label:`author:${spec.id}` }),
  (authored, spec) => agent(
    `Adversarially review + fix (Edit) the test at ${spec.file} (capability ${spec.capability}; comparator
${spec.reference_tool}). If no file, reply "no file". Else verify: identical data, correct mature comparator call,
right quantity grid-aligned, principled un-weakened bound, compiles (real APIs/every import used), no banned
patterns, single #[test]. Fix in place. Reply ok/needs-work + issues fixed.`,
    { phase:'Review', label:`review:${spec.id}` }).then(rev => ({ spec:spec.id, file:spec.file, authored, review:rev })),
)
const done = results.filter(Boolean)
log(`wave-3: ${done.length} specs authored+reviewed`)
return { count: done.length, files: done.map(d => d.file) }
