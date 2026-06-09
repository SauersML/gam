export const meta = {
  name: 'reference-quality-suite',
  description: 'Author e2e quality tests benchmarking gam against the MATURE standard tool for every capability and combination (mgcv, gamlss, survival, flexsurv, VGAM, scam, lme4, cmprsk, mlt, scikit-learn, statsmodels, pyGAM, lifelines, scikit-survival, GP libs, PyMC/numpyro, arviz/loo, geomstats, ...)',
  whenToUse: 'Build out end-to-end, real-data quality tests that benchmark every gam capability and feature-combination against the best mature reference tool for it.',
  phases: [
    { title: 'Plan', detail: 'one planner per capability class expands concrete test specs against the best reference' },
    { title: 'Author', detail: 'one agent per spec writes a Rust integration test using the reference harness' },
    { title: 'Review', detail: 'adversarial check that each test calls the right reference and asserts a principled, un-weakened bound' },
  ],
}

// ────────────────────────────────────────────────────────────────────────────
// Shared context. Agents READ the two canonical files and mirror them, so
// authored tests stay correct by construction rather than from prose that drifts.
// ────────────────────────────────────────────────────────────────────────────
const CANON = `
CANONICAL FILES — read these FIRST and mirror them exactly:
  * src/test_support/reference.rs — the reference-comparison harness. Public API:
      gam::test_support::reference::{Column, run_r, run_python, relative_l2, rmse, max_abs_diff, pearson, ReferenceResult}
      - run_r(&[Column::new("name", &vec_f64)], r#"...R body..."#) -> ReferenceResult
        The R body sees the data as data.frame \`df\`; it calls emit("key", numeric_vector)
        to return results, read back via result.scalar("key") / result.vector("key").
        A missing R package (library(x) error) => the test FAILS (there is no skipping).
      - run_python(...) mirrors run_r; body sees pandas \`df\` (or a dict of np arrays),
        calls emit("key", iterable). Use for scikit-learn / scipy / statsmodels / pyGAM /
        lifelines / scikit-survival / PyMC / numpyro / arviz / geomstats / GPy / properscoring.
  * tests/quality_vs_mgcv_gaussian_smooth.rs — the CANONICAL EXAMPLE. Copy its structure:
    load/build data -> fit gam via fit_from_formula -> rebuild design with
    build_term_collection_design at evaluation points -> apply beta for fitted values ->
    fit the SAME model with the reference tool via run_r/run_python -> assert agreement on
    the quantity that matters. It currently achieves rel_l2 ~0.005 vs mgcv on lidar.

GAM FIT API (in-Rust, no Python needed for gam itself):
  use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
            encode_recordswith_inferred_schema};
  use gam::smooth::build_term_collection_design;
  use gam::matrix::LinearOperator;
  - fit_from_formula("y ~ s(x)", &encoded_dataset, &FitConfig{family:Some("gaussian".into()), ..Default::default()})
    returns FitResult; match the right variant (Standard, GaussianLocationScale,
    BinomialLocationScale, SurvivalLocationScale, SurvivalTransformation, BernoulliMarginalSlope,
    SurvivalMarginalSlope, ...). READ src/solver/workflow.rs for the variant fields, and
    src/inference/formula_dsl.rs for the exact DSL syntax (s/te/ti/cc/sphere/duchon/matern/
    by=/bs="re"/fs/sz/linear constraints/linkwiggle/timewiggle/survmodel(...)/etc.).
  - fit.fit.edf_total() -> Option<f64>; fit.fit.beta is the coefficient vector.
  - Synthesize data with encode_recordswith_inferred_schema(headers, rows) (see
    tests/duchon_sin8_quality.rs) or load a real CSV with load_csvwith_inferred_schema.

REAL DATASETS under bench/datasets/ (use concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/<f>")):
  lidar.csv (range->logratio; 1-D smoothing), bone.csv (t,d,trt; survival), prostate.csv,
  wine.csv, haberman.csv (binary survival counts), horse.csv, cirrhosis.csv (survival+status),
  heart_failure_clinical_records_dataset.csv, icu_survival_death.csv / icu_survival_los.csv,
  five_day.csv / 31_day.csv, hgdp_1kg_pc_data.tsv (PCs/geo, for sphere/manifold).
  Prefer real data; otherwise fixed-seed synthetic fed IDENTICALLY to gam and the reference.

MATURE REFERENCE TOOLS — pick the BEST-IN-CLASS standard for each capability (do NOT default
to mgcv for everything). Assume CI provisions them; a missing one is a hard failure, which is
acceptable. Map capability -> reference:
  R packages:
    mgcv        : GAM smooths (bs=tp/cc/ds/gp/sos/re), te/ti tensor, families
                  (poisson/Gamma/binomial(probit/cloglog)/nb/tw/betar/ocat/multinom), gaulss
                  location-scale, predict(se.fit=TRUE) for CIs, smoothing-param selection (REML).
    gamlss      : distributional regression mu+sigma(+nu,tau); families NO/BI/GA/BCT/...;
                  gamlss.cens for censored; the standard for GAMLSS-style location-scale.
    survival    : survreg (parametric AFT weibull/lognormal/loglogistic), coxph, survfit, frailty.
    flexsurv    : flexsurvspline (Royston-Parmar flexible parametric survival) — the direct
                  reference for gam's RP/spline survival baseline; flexsurvreg for AFT.
    VGAM        : vglm multinomial / proportional-odds / tobit / zero-inflated / many families.
    nnet/ordinal/MASS : multinom, clm (ordinal), glm.nb (negbin), polr.
    betareg     : beta regression.
    scam        : SHAPE-CONSTRAINED additive models (monotone mpi/mpd, convex) — reference for
                  gam's monotonic/shape constraints (linear(...,constraint=...)).
    splines2    : I-spline / M-spline / monotone bases — reference for gam's I-spline basis.
    cmprsk      : competing-risks cumulative incidence (cuminc) — reference for gam competing risks.
    lme4/nlme   : mixed models / variance components — reference for random effects re()/group().
    fields/GpGp/laGP/mlegp : Gaussian-process / kriging — reference for Matern/GP smooths.
    gss         : smoothing-spline ANOVA — alt reference for additive/Duchon.
    quantreg    : quantile regression.
    mlt/tram    : transformation models — reference for gam's transformation_normal.
    loo         : PSIS-LOO / WAIC — reference for gam's ALO/LOO diagnostics.
    brms/rstanarm/bamlss/INLA : Bayesian additive/regression — reference for posterior means/intervals.
    circular/CircStats : circular/directional statistics — for periodic/sphere sanity.
  Python modules:
    scikit-learn : LinearRegression/LogisticRegression/PoissonRegressor/GammaRegressor/TweedieRegressor,
                   GaussianProcessRegressor, SplineTransformer; plain-GLM and GP baselines.
    statsmodels  : GLM (all families/links), GAM (BSplines/CyclicCubicSplines), GEE, QuantReg,
                   MixedLM, discrete (NegativeBinomial, Poisson, ZeroInflated, ordinal).
    pyGAM        : LinearGAM/LogisticGAM/PoissonGAM/GammaGAM with s()/te()/factor() — a second
                   direct GAM reference for cross-checking smooths and lambda selection.
    lifelines    : KaplanMeier, CoxPH, WeibullAFT/LogNormalAFT/LogLogisticAFT, AalenJohansen (CIF).
    scikit-survival : CoxPHSurvivalAnalysis, RandomSurvivalForest, concordance.
    PyMC / numpyro + arviz : NUTS sampling reference; arviz for R-hat / ESS — reference for gam's
                   NUTS/HMC posterior and convergence diagnostics.
    properscoring / scoringrules : CRPS / energy score — reference for distributional calibration.
    geomstats    : Frechet mean, geodesics, sphere/SPD/Stiefel/Grassmann — reference for gam's
                   manifold geometry primitives and intrinsic smooths.
    GPy / gpytorch : Gaussian-process regression — alt reference for Matern/GP smooths.
    scipy        : distributions, special functions, optimize, interpolate — exact ground truth.

HARD RULES (CLAUDE.md + repo memory — violating these FAILS review):
  - GOAL: author the IDEAL test against the IDEAL mature reference and let it MEASURE gam's
    quality honestly. A failing assertion because gam genuinely diverges from the reference is
    an ACCEPTABLE, useful outcome — DO NOT weaken the bound to make it pass, and DO NOT modify
    gam's source. We test; we do not fix code here. Breaking main / failing tests is fine.
  - But the TEST ITSELF must be high quality: identical data to both engines; the correct,
    mature reference call (right package, family, bs=, dist=, link); element-wise/grid-aligned
    comparison of the quantity that matters (fitted function / smooth shape / EDF / coefficients
    / survival curve / CIF / CI width / posterior mean+sd / CRPS / variance component); and a
    PRINCIPLED bound justified by a one-line comment. A bound so loose it asserts nothing is a
    review failure just as much as a weakened one.
  - NO skipping for missing software deps (only hardware/GPU may skip, which is not relevant here).
  - COMBINATIONS matter: every class must include cross-feature combinations (e.g. family x basis,
    location-scale x cyclic, survival x smooth covariate, by-factor smooth x random intercept,
    multi-smooth additive, tensor x non-gaussian family). Bugs hide in combinations.
  - Banned patterns (build.rs-enforced, apply to test files too EXCEPT panic!/println! which ARE
    allowed in tests/): NO \`let _ =\`, NO \`allow(dead_code)\`/\`allow(unused...)\`, NO
    \`#[cfg(feature=...)]\`, NO new Cargo features, NO \`std::env::var\`, NO \`black_box\` silencer.
    Each integration test is its OWN crate: every import/helper you add MUST be used (warnings=deny).
  - The test MUST COMPILE: use only real APIs you verified by reading the source; if unsure of a
    signature, READ it. Prefer fewer, correct assertions over many guessed ones.
  - One focused test file per spec: tests/quality_vs_<tool>_<capability>.rs, single #[test] fn,
    a module doc-comment stating what mature tool it benchmarks against and why.
  - Do NOT run cargo / python / R, and do NOT compile — the coordinator compiles, runs, and triages.
    Only Write (author) / Edit (review) the test file.
  - COMMIT FREQUENTLY as you go. Immediately after EVERY Write/Edit, commit the file you just touched:
    \`git add <that exact file path>\` then \`git commit -m 'wip: <spec-id>'\`. Commit even if the code
    does NOT compile yet, even mid-edit / breaking — we want very frequent small commits (after every
    single edit; never leave more than ~5 minutes of work uncommitted). Stage ONLY the file(s) you
    authored, by explicit path — NEVER \`git add -A\`/\`-u\` (it sweeps other agents' in-flight files).
    If a commit hits a \`.git/index.lock\`, wait a moment and retry. Do NOT push.
`

// ────────────────────────────────────────────────────────────────────────────
// Capability classes spanning the full inventory, each mapped to the best
// mature reference(s). Planners expand each into concrete specs + combinations.
// ────────────────────────────────────────────────────────────────────────────
const CLASSES = [
  { key: 'smooths_1d', title: '1-D smooth bases vs mgcv / pyGAM / statsmodels',
    guidance: `s(x) p-spline, thin-plate s(x,bs="tp"), Duchon, Matern/GP, cyclic cc(x). Fit gam and the reference with the matching basis; assert fitted-function relative L2 + EDF. Cross-check ONE against pyGAM and ONE against statsmodels GAM as independent second references. Include a low-noise truth-recovery case both must nail.` },
  { key: 'tensor', title: 'Tensor & multivariate smooths vs mgcv',
    guidance: `s(x,z,bs="tp") thin-plate 2-D, te(x,z) tensor product, ti() interaction, and 3-D te. Compare vs mgcv te()/ti()/s(,bs="tp") on a fixed-seed surface; assert fitted-surface relative L2 on a grid + EDF. Include a tensor x non-gaussian-family combination.` },
  { key: 'glm_families', title: 'GLM/GAM families vs mgcv / sklearn / statsmodels / VGAM',
    guidance: `poisson(log), Gamma(log), binomial logit/probit/cloglog, negative-binomial, tweedie, beta. Reference mgcv family= AND a second tool: sklearn PoissonRegressor/GammaRegressor/TweedieRegressor/LogisticRegression, statsmodels GLM, betareg for beta, MASS glm.nb for negbin, VGAM for tweedie/zip. Assert fitted-mean / coefficient agreement + predictive parity. Use real data where suitable.` },
  { key: 'multinomial_ordinal', title: 'Multinomial & ordinal vs VGAM / nnet / statsmodels',
    guidance: `multinomial (softmax) and ordinal responses. Reference: nnet::multinom / VGAM::vglm(multinomial) / statsmodels MNLogit; ordinal via MASS::polr / ordinal::clm. Assert class-probability agreement and coefficient agreement.` },
  { key: 'location_scale', title: 'Location-scale / GAMLSS vs gamlss / mgcv gaulss / statsmodels',
    guidance: `gaussian location-scale (mu + log-sigma) and binomial location-scale, smooth in BOTH. Reference R gamlss (NO/BI) and mgcv gaulss. Assert agreement of fitted mu AND fitted sigma across the covariate range. Include a location-scale x cyclic-smooth combination.` },
  { key: 'survival_parametric', title: 'Parametric / flexible survival vs survreg / flexsurv / lifelines',
    guidance: `Weibull/lognormal/loglogistic AFT and Royston-Parmar/spline baselines. Reference survival::survreg, flexsurv::flexsurvspline (RP — direct match for gam's spline baseline), lifelines *AFT. Assert predicted survival curve S(t|x) on a time grid + AFT coefficients/scale. Use bone.csv/cirrhosis.csv or fixed-seed synthetic with known params. Include survival x smooth-covariate combination.` },
  { key: 'survival_cox_cif', title: 'Cox & competing-risks vs survival / cmprsk / lifelines',
    guidance: `Cox PH and competing-risks CIF. Reference survival::coxph/survfit, cmprsk::cuminc, lifelines AalenJohansen. Assert partial-likelihood coefficients and/or predicted cumulative incidence per cause on a time grid.` },
  { key: 'random_effects', title: 'Random effects & factor smooths vs lme4 / mgcv',
    guidance: `random intercepts re(g)/s(g,bs="re"), random slopes, factor smooths fs/sz, by-variable s(x,by=f). Reference lme4::lmer/glmer (variance components) and mgcv s(,bs="re")/s(x,by=f). Assert variance-component and predicted group-effect / per-level smooth agreement. Include by-factor-smooth x random-intercept combination.` },
  { key: 'geometric_manifold', title: 'Geometric / manifold smooths vs mgcv sos/cc & geomstats',
    guidance: `periodic 1-D vs mgcv bs="cc"; sphere S2 vs mgcv bs="sos" (use lon/lat); cyclic tensor (torus) vs mgcv te(bs=c("cc","cc")); cylinder vs te(bs=c("cc","ps")). Also validate manifold primitives (sphere Frechet mean / geodesic) vs geomstats. Assert fitted-function relative L2 on a grid honoring periodicity (seam continuity). Closes a class with NO current reference coverage.` },
  { key: 'shape_constraints', title: 'Monotone / shape-constrained smooths vs scam / splines2',
    guidance: `gam's linear(x, constraint="increasing"/"decreasing"/...) and I-spline monotone bases. Reference scam::scam (bs="mpi"/"mpd" monotone, convex) and splines2 iSpline. Assert the fitted monotone function matches scam and is genuinely monotone; compare fitted values relative L2.` },
  { key: 'gp_smooths', title: 'Gaussian-process / Matern smooths vs GpGp / fields / GPy / sklearn GP',
    guidance: `gam's matern()/gp smooths. Reference fields::Krig/mKrig, GpGp, sklearn GaussianProcessRegressor, GPy. Assert fitted-function agreement and length-scale/EDF sanity on a fixed-seed GP-drawn surface.` },
  { key: 'transformation', title: 'Transformation models vs mlt / tram / scipy boxcox',
    guidance: `gam's transformation_normal (learnable Box-Cox-like lambda + normal errors). Reference R mlt/tram transformation models and scipy.stats.boxcox / yeojohnson for the lambda. Assert agreement of the estimated transformation parameter and fitted conditional mean.` },
  { key: 'uncertainty_ci', title: 'Uncertainty / confidence intervals vs mgcv predict(se.fit)',
    guidance: `pointwise credible/confidence interval width and coverage for a gaussian smooth. Reference mgcv predict(se.fit=TRUE). Assert CI half-width agreement across the covariate range and, on simulated truth, empirical coverage near nominal (e.g. 0.90+/-0.05).` },
  { key: 'bayes_sampling', title: 'NUTS / posterior sampling vs PyMC / numpyro / brms + arviz',
    guidance: `gam's NUTS/HMC posterior (model.sample) and Polya-Gamma binomial. Reference: rebuild the SAME penalized model in PyMC/numpyro and sample; compare posterior mean and sd of the linear predictor / coefficients. Use arviz for R-hat/ESS sanity. For a conjugate gaussian case, compare to the analytic posterior (scipy) as exact ground truth.` },
  { key: 'loo_alo', title: 'ALO / leave-one-out vs loo / arviz / brute-force CV',
    guidance: `gam's ALO (approximate LOO). Reference: brute-force exact LOO refits (gam itself, leave-one-out loop) as ground truth, and loo (R) / arviz PSIS-LOO where a likelihood is available. Assert ALO predictions/residuals match exact LOO within a tight tolerance.` },
  { key: 'distributional_calibration', title: 'Distributional calibration (CRPS/PIT) vs properscoring / scoringrules',
    guidance: `for location-scale and survival predictive distributions, assert calibration: CRPS computed vs properscoring/scoringrules matches gam's predictive distribution, and PIT histogram is ~uniform on held-out data. Compare gam's CRPS to the reference distributional model's CRPS (gamlss / lifelines) — gam should match or beat it.` },
  { key: 'combinations', title: 'Hard feature combinations vs the right reference',
    guidance: `Combinations where bugs hide, each fed identical data to gam and the best reference: poisson + thin-plate 2-D vs mgcv; binomial + cyclic vs mgcv; location-scale + smooth-in-both vs gamlss; survival AFT + smooth covariate vs flexsurv/gamlss.cens; additive multi-smooth y~s(x1)+s(x2)+s(x3) vs mgcv; by-factor smooth + random intercept vs mgcv/lme4; tensor + Gamma family vs mgcv; monotone constraint + second smooth vs scam. Assert the fitted quantity agrees.` },
]

const SPECS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['tests'],
  properties: {
    tests: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['id', 'file', 'capability', 'reference_tool', 'gam_formula', 'data', 'metric', 'rationale'],
        properties: {
          id: { type: 'string', description: 'short kebab id, unique within class' },
          file: { type: 'string', description: 'tests/quality_vs_<tool>_<capability>.rs' },
          capability: { type: 'string', description: 'the gam capability/combination under test' },
          reference_tool: { type: 'string', description: 'best mature reference + which call (e.g. "flexsurv::flexsurvspline", "statsmodels GLM Poisson", "scam bs=mpi", "PyMC NUTS")' },
          gam_formula: { type: 'string', description: 'gam formula + family/model config' },
          data: { type: 'string', description: 'real dataset filename OR synthetic recipe (truth fn, n, seed, noise/params)' },
          metric: { type: 'string', description: 'comparison metric + the agreement bound to assert and why it is principled' },
          rationale: { type: 'string', description: 'why this reference is THE mature standard for this capability' },
        },
      },
    },
  },
}

const AUTHORED_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['file', 'wrote', 'summary'],
  properties: {
    file: { type: 'string' },
    wrote: { type: 'boolean' },
    summary: { type: 'string', description: 'what it fits, which reference it calls, the asserted bound' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['file', 'ok', 'issues'],
  properties: {
    file: { type: 'string' },
    ok: { type: 'boolean', description: 'true if correct reference, identical data, principled un-weakened bound, compiles' },
    issues: { type: 'array', items: { type: 'string' } },
  },
}

// ── Phase 1: plan the matrix (one planner per class, in parallel) ──
phase('Plan')
const planned = await parallel(CLASSES.map(c => () =>
  agent(
    `You are planning end-to-end QUALITY tests for the gam statistical engine, class: "${c.title}".
${c.guidance}

${CANON}

Read enough of src/ (families, terms, inference/formula_dsl.rs, solver/workflow.rs) and existing
tests/ to know the EXACT formula-DSL syntax + FitResult variant for each capability in this class,
and to confirm the reference tool's matching call. Output concrete, buildable test specs covering
this class AND its important feature-combinations, each naming the BEST mature reference and a
principled comparison metric + bound. Favor real datasets; give exact synthetic recipes otherwise.
Aim for thorough coverage (typically 4-7 specs, including >=1 combination).`,
    { phase: 'Plan', label: `plan:${c.key}`, schema: SPECS_SCHEMA, agentType: 'Explore' },
  ).then(r => ({ class: c.key, tests: (r && r.tests) || [] }))
)).then(rs => rs.filter(Boolean))

const specs = planned.flatMap(p => p.tests.map(t => ({ ...t, class: p.class })))
  .filter(t => t && t.file && t.id)
log(`planned ${specs.length} test specs across ${CLASSES.length} capability classes`)
if (specs.length === 0) return { error: 'no specs planned' }

// de-dup filenames so parallel authors never collide
const seen = new Set()
const uniqueSpecs = []
for (const s of specs) {
  let f = s.file.endsWith('.rs') ? s.file : `${s.file}.rs`
  if (!f.startsWith('tests/')) f = `tests/${f}`
  if (seen.has(f)) {
    const base = f.replace(/\.rs$/, '')
    let k = 2
    while (seen.has(`${base}_${k}.rs`)) k++
    f = `${base}_${k}.rs`
  }
  seen.add(f)
  uniqueSpecs.push({ ...s, file: f })
}
log(`authoring ${uniqueSpecs.length} unique test files`)

// ── Phase 2+3: author then adversarially review, pipelined per spec ──
const results = await pipeline(
  uniqueSpecs,
  (spec) => agent(
    `Author ONE Rust integration test that benchmarks a gam capability against a mature reference tool.

SPEC:
  id:            ${spec.id}
  file:          ${spec.file}
  capability:    ${spec.capability}
  reference:     ${spec.reference_tool}
  gam formula:   ${spec.gam_formula}
  data:          ${spec.data}
  metric/bound:  ${spec.metric}
  rationale:     ${spec.rationale}

${CANON}

Steps:
  1. Read src/test_support/reference.rs and tests/quality_vs_mgcv_gaussian_smooth.rs in full.
  2. Read whatever src/ files you need to get the formula syntax + FitResult variant + prediction
     path EXACTLY right for THIS capability (verify against the code; do not guess signatures).
  3. Write the test to ${spec.file}: build/load identical data, fit gam, fit the reference via
     run_r/run_python, assert the principled agreement bound. Justify the bound in a one-line
     comment grounded in the math. Print the key metrics with eprintln!. NEVER weaken the bound
     to pass and NEVER modify gam source — a genuine divergence failing the test is fine.
  4. Use the Write tool to create the file. Ensure it COMPILES (real APIs only; every import used;
     no banned patterns). Do NOT run cargo/python/R. COMMIT FREQUENTLY: the instant you finish a
     Write/Edit, \`git add ${spec.file}\` then \`git commit -m 'wip: ${spec.id}'\` — even if it does
     not compile yet, even mid-edit. Stage only ${spec.file} by path (never \`git add -A\`); retry on
     \`.git/index.lock\`; do NOT push.

Return the file path, whether you wrote it, and a one-line summary.`,
    { phase: 'Author', label: `author:${spec.id}`, schema: AUTHORED_SCHEMA },
  ),
  (authored, spec) => agent(
    `Adversarially review the reference-quality test at ${spec.file} (capability: ${spec.capability};
reference: ${spec.reference_tool}). Read the file and src/test_support/reference.rs, and FIX in
place with Edit when wrong:
  - Identical data fed to gam and the reference?
  - Correct, mature reference call (right package, family/bs=/dist=/link, right API)?
  - Compares the quantity that matters, aligned element-wise / on the same grid?
  - Bound is PRINCIPLED and NOT weakened (and not so loose it asserts nothing)? gam diverging and
    failing is acceptable; a dishonest bound is not.
  - Compiles (real APIs, every import used), no skipping, no banned patterns, single #[test].
Apply fixes directly with Edit. COMMIT FREQUENTLY: after EVERY Edit, \`git add ${spec.file}\` then
\`git commit -m 'wip: review ${spec.id}'\` — even if it does not compile, even mid-fix. Stage only
${spec.file} by path (never \`git add -A\`); retry on \`.git/index.lock\`; do NOT run cargo/python/R; do
NOT push. Return ok + the issues you found/fixed.`,
    { phase: 'Review', label: `review:${spec.id}`, schema: REVIEW_SCHEMA },
  ).then(rev => ({ spec: spec.id, file: spec.file, authored, review: rev })),
)

const done = results.filter(Boolean)
log(`authored+reviewed ${done.length} reference-quality tests`)
return {
  count: done.length,
  files: done.map(d => d.file),
  needs_attention: done.filter(d => d.review && d.review.ok === false)
    .map(d => ({ file: d.file, issues: d.review.issues })),
}
