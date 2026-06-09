export const meta = {
  name: 'quality-strengthen-with-real-data',
  description: 'Strengthen synthetic-only reference-quality tests by ADDING a real-data arm (held-out predictive quality on a fitting real dataset + match-or-beat the reference), keeping the synthetic truth-recovery arm. Skip pure-math-ground-truth and known-dynamics tests.',
  phases: [
    { title: 'Match', detail: 'map each synthetic-only test to a fitting real dataset (or skip)' },
    { title: 'Strengthen', detail: 'add a real-data #[test] arm to each matched test' },
  ],
}

const DATASETS = `
REAL DATASETS in bench/datasets/ (load via concat!(env!("CARGO_MANIFEST_DIR"),"/bench/datasets/<f>")):
  lidar.csv            1-D smooth: range -> logratio (Gaussian)            -> any 1-D smooth/basis, GP, CI
  quakes.csv           spatial: long,lat,depth -> mag (Gaussian)           -> thin-plate/tensor 2-D, GP/matern 2-D
  global_major_city_temp.csv  sphere: lat,lon -> temp                      -> sphere / spatial smooths
  badhealth.csv        count: age,badh -> numvisit (Poisson/NB)            -> poisson, negbin, count LOO/EBM
  nottem_monthly_temp.csv  cyclic: month(1-12) -> temp                     -> cyclic smooths
  bike_sharing_torus.csv   torus: season(0-365),hour(0-24) -> log1p(cnt)   -> torus / cyclic tensor
  solar_zenith_angle.csv   cylinder: month,tst_hours -> sza                -> cylinder / cyclic-linear tensor
  gagurine.csv         heteroscedastic: Age -> GAG                         -> location-scale, distributional, CRPS, CI
  sleepstudy.csv       longitudinal: Subject,Days -> Reaction              -> random effects (re/fs), factor smooth
  penguins.csv         multinomial: bill/flipper/mass -> species(3)        -> multinomial, ordinal, 2-class binomial
  prostate.csv         binary: pc1,pc2,... -> y                            -> binomial logit/probit/cloglog, links
  haberman.csv         binary: age,year,nodes -> survival status           -> binomial, links
  heart_failure_clinical_records_dataset.csv  binary/survival              -> binomial, survival, ALO/LOO
  veteran_lung.csv / bone.csv / cirrhosis.csv / icu_survival_death.csv  right-censored survival -> AFT/Cox/CIF/frailty/transformation
  wine.csv             ordinal/regression: chem -> quality                 -> ordinal, regression
  crabs.csv            multivariate: 5 measurements, 4 groups              -> SPD, GP multivariate, by-group/factor, multinomial(4)
  olive_oils.csv       8 fatty acids + region(9)/area                      -> multinomial(region), regression, Grassmann
  skye_afm_lavas.csv   compositional: A,F,M (sum 100)                      -> compositional / simplex
`

const MATCH_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['matches'],
  properties: { matches: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    required: ['file', 'dataset', 'plan'],
    properties: {
      file: { type: 'string', description: 'tests/quality_vs_*.rs (existing synthetic-only test)' },
      dataset: { type: 'string', description: 'bench/datasets/<f> that fits the test capability, or "SKIP"' },
      plan: { type: 'string', description: 'if a dataset: response col, covariate cols, gam formula, reference tool, objective held-out metric. if SKIP: one-line reason (pure-math ground truth / known-dynamics / no fitting real data).' },
    },
  } } },
}

phase('Match')
const matched = await agent(
  `List every synthetic-only reference-quality test and match each to a fitting REAL dataset (or SKIP).
Run: grep -rL 'bench/datasets/' tests/quality_vs_*.rs   (these are the synthetic-only tests).
For EACH, read its capability (from the filename + a quick skim) and pick the single best-fitting real dataset
from the inventory below whose modeling type matches the test's capability, OR "SKIP".
SKIP when: the reference is pure MATHEMATICAL GROUND TRUTH (geomstats geodesics/Frechet, scipy transforms,
analytic conjugate posterior, exact brute-force LOO is fine to strengthen with real data actually) ; or the test
validates KNOWN DYNAMICS (pysindy ODE) ; or NO real dataset fits the capability. When in doubt and a real dataset
plausibly fits, MATCH it (we want to strengthen broadly).
${DATASETS}
Output the full match list (every synthetic-only test, with dataset or SKIP + a concrete plan).`,
  { phase: 'Match', label: 'match-datasets', schema: MATCH_SCHEMA, agentType: 'Explore' },
)
const all = (matched && matched.matches) || []
const todo = all.filter(m => m && m.file && m.dataset && m.dataset.toUpperCase() !== 'SKIP')
log(`matched ${todo.length} tests to real datasets to strengthen; ${all.length - todo.length} kept synthetic-only`)
if (todo.length === 0) return { strengthened: 0, note: 'nothing matched' }

const GUIDE = `
Strengthen ONE existing synthetic-only quality test by ADDING a real-data arm. DO NOT remove or weaken the
existing synthetic #[test] (its known-truth RMSE-recovery is the accuracy proof) — ADD a SECOND #[test] fn named
<existing>_on_real_data that exercises the SAME gam capability on the assigned real dataset.

Read FIRST: src/test_support/reference.rs, tests/quality_vs_mgcv_gaussian_smooth.rs (the canonical real-data
template), and the test file you are editing. Use gam::test_support::reference (Column/run_r/run_python);
gam::fit_from_formula; gam::smooth::build_term_collection_design; load real data with
gam::load_csvwith_inferred_schema(Path::new(concat!(env!("CARGO_MANIFEST_DIR"),"/bench/datasets/<f>"))).

The new real-data arm must assert OBJECTIVE quality (real data => truth unknown): deterministic train/test split
(fixed index), fit gam on train, predict test, and assert an ABSOLUTE held-out metric bar AND gam >= reference -
margin on that SAME metric (Gaussian R^2/RMSE, binomial AUC/log-loss, count deviance, survival concordance,
distributional NLL/CRPS, multiclass accuracy/log-loss). The mature tool is a match-or-beat BASELINE, never the
output to replicate. Compute metrics in plain Rust. Put the dataset SOURCE in a comment.

CRITICAL: identical train/test rows in the SAME order to gam and the reference; within ONE run_r/run_python call
every Column MUST be equal length (don't mix train-length and test-length columns — pass an is_train mask or
separate calls). No banned patterns (no \`let _ =\`, allow(dead_code/unused), #[cfg(feature)], std::env::var,
underscore-prefixed fn params, black_box); EVERY import used (add any new imports you need, e.g.
load_csvwith_inferred_schema, std::path::Path); the WHOLE FILE must still compile — verify APIs by reading source.
panic!/println! ok in tests/. Do NOT run cargo or python/R (no compiling). Only Edit the file (append the new
fn + any imports).
COMMIT FREQUENTLY: after EVERY Edit, commit the file you just touched — \`git add <this exact file path>\` then
\`git commit -m 'wip: real-data arm'\`. Commit even if it does NOT compile yet, even mid-edit / breaking; we want
very frequent small commits (after every single edit; never more than ~5 minutes uncommitted). Stage ONLY this
file by explicit path — NEVER \`git add -A\`/\`-u\`. Retry on \`.git/index.lock\`. Do NOT push.
Reply with: the real dataset used, the gam formula, and the objective metric the new arm asserts.`

const results = await pipeline(
  todo,
  (m) => agent(
    `Add a real-data arm to ${m.file} using dataset ${m.dataset}. Plan: ${m.plan}\n${GUIDE}`,
    { phase: 'Strengthen', label: `strengthen:${m.file.replace(/^tests\/quality_vs_/, '').replace(/\.rs$/, '')}` },
  ).then(text => ({ file: m.file, dataset: m.dataset, text })),
)
const done = results.filter(Boolean)
log(`strengthened ${done.length} tests with a real-data arm`)
return { strengthened: done.length, files: done.map(d => d.file) }
