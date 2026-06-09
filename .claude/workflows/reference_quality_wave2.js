export const meta = {
  name: 'reference-quality-wave2',
  description: 'Wave 2: e2e quality tests for the remaining gam axes + research-flagged comparators (rstpm2, R-INLA, PyMC/numpyro+arviz, loo, properscoring, mgcv sos/spheresmooth/Directional, compositions, GpGp/GPyTorch, InterpretML-EBM/NAM, PySINDy, gamlss2/bamlss/XGBoostLSS, mgcv se.fit CI)',
  whenToUse: 'Extend the reference-quality suite to the capabilities and mature comparators not yet covered by wave 1.',
  phases: [
    { title: 'Plan', detail: 'one planner per new capability class expands concrete specs vs the best comparator' },
    { title: 'Author', detail: 'one agent per spec writes a Rust integration test (no schema; just writes the file)' },
    { title: 'Review', detail: 'adversarial check that each test calls the right comparator and asserts a principled bound' },
  ],
}

const EXISTING = `quality_vs_betareg_beta_logit, quality_vs_flexsurv_rp_baseline, quality_vs_flexsurv_rp_spline,
quality_vs_flexsurv_weibull_aft, quality_vs_gam_competing_risks_integral_identity, quality_vs_gamlss_* (location-scale,
binomial, cyclic, by_group, multi_smooth, survival_ls), quality_vs_lifelines_* (competing_risks_cif, cox_like, *_aft,
smooth_tensor_baseline), quality_vs_lme4_* (random_intercept, random_slope, random_intercept_by_smooth),
quality_vs_mass_ordinal_polr, quality_vs_mgcv_* (cyclic_cubic, duchon, factor_smooth_fs/sz, gaulss_gaussian/tensor,
gaussian_smooth, matern, poisson_tensor, pspline, tensor_*, thin_plate_*), quality_vs_pygam_pspline,
quality_vs_scam_monotone_baseline, quality_vs_sklearn_binomial_logit/poisson_log, quality_vs_statsmodels_* (binomial_probit,
gam_additive, gamma_log, multinomial, negbin, ordinal_mnlogit, transformation_survival, tweedie),
quality_vs_survival_location_scale_lognormal, quality_vs_vgam_multinomial_*`

const CANON = `
CANONICAL FILES — read these FIRST and mirror them exactly:
  * src/test_support/reference.rs — the harness. API:
    gam::test_support::reference::{Column, run_r, run_python, relative_l2, rmse, max_abs_diff, pearson, ReferenceResult}
    run_r(&[Column::new("name",&vec)], r#"R body; data.frame df; emit("key", vec)"#) -> ReferenceResult
    run_python(&[Column::new(...)], r#"python body; pandas df (or dict of np arrays); emit("key", iterable)"#)
    result.scalar("key") / result.vector("key"). A missing tool/package => the test FAILS (no skipping).
  * tests/quality_vs_mgcv_gaussian_smooth.rs — the canonical example (load -> fit_from_formula ->
    build_term_collection_design at eval points -> apply beta -> run_r/run_python -> assert agreement).

GAM FIT API: use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
  encode_recordswith_inferred_schema}; gam::smooth::build_term_collection_design; gam::matrix::LinearOperator.
  Match the right FitResult variant (read src/solver/workflow.rs); fit.fit.edf_total(), fit.fit.beta.
  Verify formula DSL + survmodel(...) syntax in src/inference/formula_dsl.rs. For NUTS/posterior, ALO, manifold,
  SINDy, transformation: READ the relevant src module (inference/hmc.rs, inference/alo.rs, geometry/*, sindy if
  present, families/transformation_normal.rs) to get the exact API before writing.

DATASETS under bench/datasets/ via concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/<f>"): lidar, bone,
  cirrhosis, heart_failure_clinical_records_dataset, icu_survival_death/_los, haberman, horse, prostate, wine,
  hgdp_1kg_pc_data.tsv (PCs/geo for sphere/manifold). Prefer real data; else fixed-seed synthetic fed IDENTICALLY.

MATURE COMPARATORS for wave 2 (pick the best-in-class; assume CI provisions; missing => hard fail, acceptable):
  R: rstpm2 (PENALIZED generalized survival models, thin-plate on log-time — closest analog to gam's penalized
       flexible baseline; reimplements Stata stpm2/pstpm2), flexsurv::flexsurvspline (Royston-Parmar),
     INLA (R-INLA / inlabru / fmesher — scalable approx-Bayesian latent-Gaussian, SPDE; the scalability comparator),
     brms / rstanarm (Stan PPL Bayesian GAM), bamlss + gamlss2 + gamboostLSS (distributional regression),
     loo (PSIS-LOO/WAIC), mgcv (bs="sos" spline-on-sphere; te(bs=c("cc","cc")) torus; predict(se.fit=TRUE) for CIs;
       cox.ph), spheresmooth / Directional / nprotreg / rcosmo (sphere/directional — fragmented, document the gap),
     compositions / robCompositions (Aitchison geometry for compositional responses — document the gap),
     scam (shape constraints), splines2 (I/M-splines), fields/GpGp/laGP/mlegp (GP/kriging), mlt/tram (transformation
       models), scoringrules (CRPS/energy/log score), cmprsk (CIF).
  Python: PyMC + numpyro + arviz (NUTS reference + R-hat/ESS — for gam's NUTS/HMC posterior & diagnostics),
     properscoring / scoringrules (CRPS), interpret (InterpretML EBM — the ML-world GAM), pygam
       (LinearGAM/LogisticGAM/PoissonGAM/te — a 2nd direct GAM reference), GPyTorch / GPflow / sklearn
       GaussianProcessRegressor (GP), pysindy (the reference for gam's SINDy module), xgboostlss / lightgbmlss /
       ngboost (distributional boosting), scikit-survival, scipy (exact ground truth: boxcox, distributions).

DISTINCTIVE-AXIS RULE: for sphere/torus/manifold and compositional responses there is often NO integrated
  head-to-head (the field is fragmented single-purpose tools). For those, the test should (a) compare against the
  closest available tool where one exists (e.g. mgcv bs="sos" for sphere, compositions for Aitchison geometry,
  scipy/geomstats for geodesic/Frechet ground truth), AND (b) assert an INTRINSIC correctness property gam must
  satisfy (seam/periodic continuity, geodesic consistency, simplex closure, rotation equivariance, recovery of a
  known smooth truth). Document in a comment that the fragmentation is itself the finding.

HARD RULES:
  - GOAL: ideal test vs the ideal mature comparator, measuring gam HONESTLY. A failing assertion because gam
    diverges is acceptable — NEVER weaken the bound, NEVER modify gam source. Tests must still be high quality.
  - Identical data to both engines; correct comparator call; element-wise/grid-aligned comparison of the quantity
    that matters; a PRINCIPLED bound with a one-line justification (not so loose it asserts nothing). No skipping.
  - Do NOT duplicate an existing file. Existing wave-1 files (do NOT re-create): ${EXISTING}
  - Banned patterns (apply to tests EXCEPT panic!/println! allowed in tests/): no \`let _ =\`, no allow(dead_code/unused),
    no #[cfg(feature=...)], no new Cargo features, no std::env::var, no black_box silencer. Each test is its own crate:
    every import MUST be used (warnings=deny). The test MUST COMPILE — verify APIs by reading source.
  - One focused file per spec: tests/quality_vs_<tool>_<capability>.rs, single #[test] fn, module doc-comment naming
    the mature tool it benchmarks against and why.
  - Do NOT run cargo/python/R, and do NOT compile — the coordinator does that. Only Write (author) /
    Edit (review) the test file.
  - COMMIT FREQUENTLY: immediately after EVERY Write/Edit, commit the file you just touched —
    \`git add <that exact file path>\` then \`git commit -m 'wip: <spec-id>'\`. Commit even if it does NOT
    compile yet, even mid-edit / breaking; we want very frequent small commits (after every single edit;
    never more than ~5 minutes of uncommitted work). Stage ONLY your file by explicit path — NEVER
    \`git add -A\`/\`-u\`. Retry on \`.git/index.lock\`. Do NOT push.
`

const CLASSES = [
  { key: 'survival_penalized', title: 'Penalized flexible-parametric survival vs rstpm2 / flexsurv',
    guidance: `gam's penalized Royston-Parmar / flexible baseline hazard. PRIMARY comparator rstpm2::pstpm2/stpm2 (penalized generalized survival models, thin-plate on log-time, log-log link — the closest analog), plus flexsurv::flexsurvspline. Assert predicted survival/hazard S(t|x) on a time grid + linear-predictor coefficients on real censored data (cirrhosis/bone) or fixed-seed synthetic. Include survival + smooth covariate.` },
  { key: 'scalable_bayes_inla', title: 'Scalable approximate-Bayes vs R-INLA',
    guidance: `gam's REML/Laplace smoothing as latent-Gaussian inference. Comparator R-INLA (inla(..., family=...) with rw2/spde smooth) — the dominant scalable approx-Bayesian latent-Gaussian tool. Assert posterior mean of the smooth and posterior sd / credible-interval width agree on the same data. Note INLA's non-commercial license in a comment.` },
  { key: 'nuts_posterior', title: 'NUTS / HMC posterior vs PyMC / numpyro + arviz',
    guidance: `gam's NUTS/HMC sampler (model.sample) and Polya-Gamma binomial. Rebuild the SAME penalized model in PyMC or numpyro, sample, and compare posterior mean & sd of the linear predictor / coefficients; use arviz for R-hat/ESS sanity (gam's own diagnostics should agree). Include a conjugate-gaussian case compared to the ANALYTIC posterior (scipy) as exact ground truth. READ src/inference/hmc.rs + sample.rs for the API.` },
  { key: 'loo_alo', title: 'ALO / leave-one-out vs brute-force CV / loo / arviz',
    guidance: `gam's ALO (approximate LOO). Ground truth = brute-force exact LOO (refit gam leaving each point out) — assert ALO predictions/residuals match exact LOO tightly. Where a pointwise log-likelihood is available, also compare to loo (R PSIS-LOO) / arviz. READ src/inference/alo.rs.` },
  { key: 'calibration_crps', title: 'Distributional calibration (CRPS / PIT) vs properscoring / scoringrules',
    guidance: `for gam's location-scale and survival predictive distributions, assert calibration: CRPS computed independently by properscoring/scoringrules matches gam's predictive distribution, and the PIT histogram is ~uniform on held-out data. Compare gam's CRPS to a reference distributional model's (gamlss/lifelines) CRPS — gam should match or beat it.` },
  { key: 'manifold_sphere', title: 'Spherical / toroidal smooths vs mgcv sos / spheresmooth / Directional',
    guidance: `gam's intrinsic S2 sphere(lon,lat) vs mgcv s(...,bs="sos") (spline-on-sphere) on lat/long data; torus vs mgcv te(bs=c("cc","cc")); cylinder vs te(bs=c("cc","ps")). Where no integrated tool exists, also assert an intrinsic property (seam/periodic continuity, recovery of a known spherical-harmonic truth, rotation behavior). Use hgdp_1kg geo PCs or fixed-seed synthetic on the sphere. Apply the DISTINCTIVE-AXIS RULE.` },
  { key: 'compositional_response', title: 'Compositional / simplex responses vs compositions (Aitchison)',
    guidance: `gam's simplex/compositional response geometry (ALR/CLR/closure). Comparator R compositions/robCompositions for the Aitchison-geometry transforms and Frechet mean; assert gam's closure/log-ratio transforms and fitted compositions agree, and satisfy simplex closure (sum to 1, nonneg). READ src/geometry/simplex.rs. Apply the DISTINCTIVE-AXIS RULE (document the gap).` },
  { key: 'gp_smooths', title: 'Gaussian-process / Matern smooths vs GpGp / fields / GPyTorch / sklearn GP',
    guidance: `gam's matern()/gp smooths as GP regression. Comparators sklearn GaussianProcessRegressor (Matern kernel), GPyTorch/GPflow, R fields::mKrig / GpGp. Assert fitted-function agreement and length-scale/EDF sanity on a fixed-seed GP-drawn surface where the truth is known.` },
  { key: 'transformation_mlt', title: 'Transformation models vs mlt / tram / scipy boxcox',
    guidance: `gam's transformation_normal (learnable Box-Cox-like lambda + normal errors). Comparator R mlt/tram (most likely transformation models) and scipy.stats.boxcox/yeojohnson for the transformation parameter. Assert the estimated transformation parameter and fitted conditional mean agree. READ src/families/transformation_normal.rs.` },
  { key: 'uncertainty_ci', title: 'Confidence/credible intervals vs mgcv predict(se.fit=TRUE)',
    guidance: `pointwise CI half-width and empirical coverage for a gaussian smooth. Comparator mgcv predict(se.fit=TRUE). Assert CI half-width agreement across the covariate range and, on simulated truth, empirical coverage near nominal (e.g. 0.90 +/- 0.05). READ src/inference/predict for the interval API.` },
  { key: 'interpretable_ml_gam', title: 'Interpretable-ML GAMs vs InterpretML-EBM / pyGAM (logistic/poisson)',
    guidance: `position gam against ML-world GAMs: InterpretML EBM (interpret.glassbox ExplainableBoostingRegressor/Classifier) and pyGAM LogisticGAM/PoissonGAM/te. Assert predictive parity (R2/AUC/deviance) AND, vs pyGAM, shape-function agreement on the same data. Use real datasets.` },
  { key: 'sindy_dynamics', title: 'SINDy sparse dynamics vs PySINDy',
    guidance: `IF gam exposes a SINDy module (search src for sindy / sparse identification), benchmark its recovered coefficients on a known ODE system (e.g. Lorenz / linear oscillator, fixed seed) against PySINDy (STLSQ). Assert the identified nonzero terms and coefficients match. If no Rust SINDy entry point exists, SKIP this spec (do not fabricate).` },
]

const SPECS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['tests'],
  properties: {
    tests: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['id','file','capability','reference_tool','gam_formula','data','metric','rationale'],
      properties: {
        id: { type: 'string' }, file: { type: 'string', description: 'tests/quality_vs_<tool>_<capability>.rs (must NOT match an existing file)' },
        capability: { type: 'string' }, reference_tool: { type: 'string' }, gam_formula: { type: 'string' },
        data: { type: 'string' }, metric: { type: 'string' }, rationale: { type: 'string' },
      },
    } },
  },
}

phase('Plan')
const planned = await parallel(CLASSES.map(c => () =>
  agent(
    `Plan end-to-end QUALITY tests for the gam engine, class: "${c.title}".
${c.guidance}

${CANON}

Read enough of src/ to confirm the EXACT gam API/DSL for this class and the comparator's matching call.
Output concrete, buildable specs (typically 2-5) covering this class, each with the best comparator and a
principled metric+bound. Do NOT propose a file that matches an existing wave-1 file.`,
    { phase: 'Plan', label: `plan:${c.key}`, schema: SPECS_SCHEMA, agentType: 'Explore' },
  ).then(r => ({ class: c.key, tests: (r && r.tests) || [] }))
)).then(rs => rs.filter(Boolean))

const specs = planned.flatMap(p => p.tests.map(t => ({ ...t, class: p.class }))).filter(t => t && t.file && t.id)
log(`planned ${specs.length} wave-2 specs across ${CLASSES.length} classes`)
if (specs.length === 0) return { error: 'no specs planned' }

const seen = new Set()
const uniqueSpecs = []
for (const s of specs) {
  let f = s.file.endsWith('.rs') ? s.file : `${s.file}.rs`
  if (!f.startsWith('tests/')) f = `tests/${f}`
  if (seen.has(f)) { const base = f.replace(/\.rs$/, ''); let k = 2; while (seen.has(`${base}_${k}.rs`)) k++; f = `${base}_${k}.rs` }
  seen.add(f); uniqueSpecs.push({ ...s, file: f })
}
log(`authoring ${uniqueSpecs.length} wave-2 files`)

// Author (NO schema — just write the file and return a short text summary) then adversarial review.
const results = await pipeline(
  uniqueSpecs,
  (spec) => agent(
    `Author ONE Rust integration test benchmarking a gam capability against a mature comparator.
SPEC: id=${spec.id} file=${spec.file}
  capability: ${spec.capability}
  comparator: ${spec.reference_tool}
  gam formula: ${spec.gam_formula}
  data: ${spec.data}
  metric/bound: ${spec.metric}
  rationale: ${spec.rationale}

${CANON}

Read src/test_support/reference.rs and tests/quality_vs_mgcv_gaussian_smooth.rs, then read whatever src/ you need
to get the API exactly right. Write the test to ${spec.file} with the Write tool. Ensure it compiles (real APIs,
every import used, no banned patterns), feeds identical data to both engines, and asserts a principled un-weakened
bound (one-line justification). Print key metrics with eprintln!. Do NOT run cargo/python/R, but DO commit
after every Write/Edit per the COMMIT FREQUENTLY rule above (\`git add ${spec.file}\` + \`git commit\`, even
if it does not compile; never \`git add -A\`; do NOT push). If the capability
genuinely does not exist in gam, write NOTHING and say so.
Reply with: the file path you wrote (or "skipped: <reason>") and a one-sentence summary.`,
    { phase: 'Author', label: `author:${spec.id}` },
  ),
  (authored, spec) => agent(
    `Adversarially review the reference-quality test at ${spec.file} (capability ${spec.capability}; comparator
${spec.reference_tool}). If the file was not written, reply "no file". Otherwise read it + src/test_support/reference.rs
and FIX in place with Edit when wrong: identical data to both engines? correct mature comparator call? compares the
quantity that matters, grid-aligned? bound principled and NOT weakened (nor so loose it asserts nothing)? compiles
(real APIs, every import used)? no skipping, no banned patterns, single #[test]? Apply fixes with Edit.
COMMIT FREQUENTLY: after EVERY Edit, \`git add ${spec.file}\` + \`git commit -m 'wip: review ${spec.id}'\`
— even if it does not compile, even mid-fix; stage only ${spec.file} (never \`git add -A\`); retry on
\`.git/index.lock\`; do NOT run cargo/python/R; do NOT push.
Reply with: ok/needs-work and the issues you found and fixed.`,
    { phase: 'Review', label: `review:${spec.id}` },
  ).then(rev => ({ spec: spec.id, file: spec.file, authored, review: rev })),
)

const done = results.filter(Boolean)
log(`wave-2: authored+reviewed ${done.length} specs`)
return { count: done.length, files: done.map(d => d.file) }
