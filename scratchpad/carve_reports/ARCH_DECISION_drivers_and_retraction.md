# #1521 Architecture decisions (locked 2026-06-26)

## 1. GAM fit-orchestration drivers → gam-models (NOT gam-solve)

`smooth/design_construction.rs` (8331L) + `smooth/spatial_optimization.rs` (9185L) are
**not solver-pure**. They embed three upward dependencies that sit AT/ABOVE gam-models:
- `crate::families::family_runtime::{FamilyStrategy, strategy_for_spec}` (link-jet dispatch) — in **gam-models**
- `crate::families::block_layout::block_count::validate_block_count` — in **gam-models**
- `crate::inference::lawley::{lawley_lr_bartlett_factor, RhoPenaltyComponent}` + `inference::higher_order::bartlett_factor_from_mean` — inference tier

Therefore they are TOP-LEVEL fit orchestration (build design ← families, run REML ← gam-solve,
LR inference ← inference tier). Their true home is **gam-models** (which already deps
gam-solve+gam-terms+gam-model-api+gam-problem). Moving them INTO gam-solve would create three
gam-solve→gam-models 2-cycles; moving them UP into gam-models dissolves all of them:
- family_runtime / block_count → `crate::`
- gam_solve::{estimate,reml,rho_optimizer,pirls,model_types,latent_cache,...} → legal downward
- inference::lawley/higher_order → `gam_terms::inference::*` (where SAE-inference agent lands them, lower still)
- gam_terms::{basis,construction,analytic_penalties,latent,smooth::term_specs symbols} → legal downward

`gam-solve` stays the PURE numerical optimizer. `gam-models` = "concrete models + how to fit them."
No new crate, no over-shatter. term_specs.rs STAYS in gam-terms (gam-terms own code consumes its
symbols; drivers consume them downward as gam_terms::smooth::*). No gam-terms→driver back-edge exists.

Pub-surface work required in gam-terms::smooth: make currently-private sibling mods
(coefficient_transforms, error, input_standardization, shape_constraints, penalty_priors) + the
term_specs symbols the drivers reference `pub`, since drivers leave the crate.

## 2. LatentRetractionRegistry / RetractionKind → gam-problem (resolves the lone terms→solve field edge)

`term_specs.rs:2447` (stays in gam-terms): `StandardLatentCoordConfig.retraction_registry:
crate::solver::latent_cache::LatentRetractionRegistry` — a gam-solve type used as a struct FIELD in
a gam-terms struct = real terms→solve edge, independent of the driver move.

- `RetractionKind` (riemannian_retraction.rs) deps ONLY ndarray → pure leaf, descends to gam-problem.
- `LatentRetractionRegistry` = `{ block: Option<RetractionKind> }` → descends with it.
Then gam-terms field type = `gam_problem::LatentRetractionRegistry` (downward, legal); gam-solve
re-exports / uses gam_problem:: for its latent machinery. Same contract-down pattern as Phase A.

## 3. Fitted-model SCC → gam-models (families + fitted-model + prediction + generative are ONE crate)

`src/inference/{model.rs(5159L), predict_io.rs(2158L), generative.rs(803L),
model_payload_builders.rs(1051L), full_conformal.rs(3899L)}` are mutually recursive with the
gam-models families (generative→family_runtime; model/predict_io types consumed by
survival/predict.rs, transformation_normal.rs, marginal_slope_orthogonal.rs, gamlss.rs,
family_runtime.rs). A true SCC — cannot split across crates. RESOLUTION: move all five INTO
gam-models at `crates/gam-models/src/inference/`. The cycle becomes intra-crate (legal).
gam-models = "concrete statistical models: families + fitted representation + prediction +
generative." gam-inference (hmc/posterior_bands/conformal.rs — the top tier) is UNCHANGED and keeps
consuming the fitted model. Recompile goal still met: stable foundation (gam-terms/gam-solve/
gam-problem) stays separate from the churny model layer.

ONLY `formula_dsl.rs` (3063L; deps only crate::smooth + crate::types) peels off → gam-terms
(at gam-models/.. no: at crates/gam-terms/src/inference/formula_dsl.rs alongside lawley etc.).

Path rewrites for the 5 moved-into-gam-models files: estimate/pirls/model_types/mixture_link/
solver::*/custom_family → gam_solve::*; basis/smooth → gam_terms::*; inference::formula_dsl →
gam_terms::inference::formula_dsl; matrix → gam_linalg; probability → gam_math; util::span →
gam_runtime; types → gam_problem/gam_spec; families::* stays crate::; inference::{model,predict_io,
generative,...} stays crate:: (now intra gam-models). Root src/inference/mod.rs re-exports →
gam_models::inference::* (+ formula_dsl → gam_terms::inference::formula_dsl) to keep gam::inference::*
facade stable for gam-cli/gam-pyffi. MUST run AFTER the driver-relocation agent (both edit gam-models).
