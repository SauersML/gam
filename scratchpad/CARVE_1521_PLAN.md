# #1521 Root-Crate Carve — Full Plan (codex-executable)

## Goal / end state
Root `gam` becomes a THIN FACADE: `lib.rs` (re-exports + `init_parallelism` + macros) + the already-thin re-export shims (`types.rs`, `model_types.rs`, `outer_subsample.rs` — each is 1 line `pub use gam_*::…`, LEAVE them) + `util/` (16 lines). All heavy code lives in sub-crates so an inference/report/config edit recompiles only that small crate + the facade, not the 653-file engine.

## What moves (only 4 things have real code)
| Source (root src/) | Lines | Destination | Notes |
|---|---|---|---|
| `inference/` (18 files) | ~17k | **NEW crate `gam-inference`** | top engine tier; the big recompile win |
| `report/` (mod.rs+sparkline.rs) | ~1.2k | **NEW crate `gam-report`** | clean leaf — uses NO other root module |
| `config_resolve.rs` | ~1.2k | **NEW crate `gam-config`** | depends on `inference` (2 refs) + fit_orchestration |
| `terms/analytic_penalties/manifest.rs` | 180 | **into `gam-terms`** | belongs with the penalty registry |

Glue that STAYS in root (cheap, no recompile cost): `lib.rs`, `macros.rs` (#[macro_export]), `util/` (or fold into gam-inference if it's the only user), and the 3 one-line shim modules.

## Dependency DAG (verified no cycle)
- `gam-report` → gam-models, gam-solve, gam-linalg (+ comfy-table/serde for formatting). LEAF.
- `gam-inference` → gam-solve, gam-problem, gam-models, gam-linalg, gam-terms, gam-sae, gam-gpu, gam-runtime, gam-data + externals: general-mcmc, statrs, rand_distr, faer, ndarray, rayon, serde, serde_json, libm, log.
- `gam-config` → gam-inference, gam-models, gam-solve, serde_json, serde.
- facade `gam` deps += gam-inference, gam-report, gam-config.
- **No back-edge**: inference's down-registration stays in root `init_parallelism` (it calls `gam_inference::hmc_io::…` and registers into `gam_problem` registries; gam-solve still calls THROUGH registries, never imports gam-inference). Clean.

## Path-rewrite table for moved inference/ files (codex applies LITERALLY, NO COMPILE)
Inside `crates/gam-inference/src/**` after the move:
- `crate::inference::`  → `crate::`            (inference is now the crate root)
- `gam::inference::`    → `crate::`
- `crate::types::`      → `gam_problem::types::`
- `gam::types::`        → `gam_problem::types::`
- `crate::model_types::`→ `gam_solve::model_types::`
- `gam::model_types::`  → `gam_solve::model_types::`
- `crate::gpu`          → `gam_gpu`
- `gam::gpu`            → `gam_gpu`
- `crate::psis`         → `gam_solve::psis`
- `crate::faer_ndarray` / `gam::faer_ndarray` → `gam_linalg::faer_ndarray`
- `crate::matrix` / `gam::matrix` → `gam_linalg::matrix`
- `crate::families::` / `gam::families::` → `gam_models::`
- `crate::estimate::` / `gam::estimate::` → `gam_solve::estimate::`
- `gam::solver::`       → `gam_solve::`
- `gam::construction::` → `gam_terms::construction::`
- `crate::smooth::` / `gam::smooth::` → `gam_terms::smooth::`  (EXCEPT the two joint builders `build_term_collection_designs_and_freeze_joint`/`build_term_collection_designs_joint` → `gam_models::fit_orchestration::drivers::`)
- `crate::util::`       → `crate::util::` IF util is moved into gam-inference; else `gam::util::`
- `super::` (within inference subtree) → unchanged
- `gam_terms::inference::` / `gam_sae::inference::` / `gam_solve::inference::` → UNCHANGED (these are OTHER crates' own inference submodules, not us — do NOT rewrite)
- the `inference::hmc_io as hmc` self-alias inside the crate → `crate::hmc_io as hmc` if present

## Parallel decomposition (massively parallel via `codex exec`, NO-COMPILE each)
Lead (me) does the COUPLED scaffolding FIRST (deterministic, fast): create the 3 crate dirs + `Cargo.toml`s + add to workspace `members`. Then dispatch in parallel:
- **codex A** → fill `gam-inference`: `git mv src/inference/* crates/gam-inference/src/`, make `mod.rs`→`lib.rs` root, apply the rewrite table above to every file. NO COMPILE.
- **codex B** → fill `gam-report`: move `src/report/{mod.rs→lib.rs,sparkline.rs}`, rewrite `crate::report::`→`crate::`, other `crate::X`/`gam::X`→owning crate. NO COMPILE.
- **codex C** → fill `gam-config`: move `src/config_resolve.rs`→`crates/gam-config/src/lib.rs`, rewrite (esp. `gam::inference::`→`gam_inference::`, `crate::`→owning crates). NO COMPILE.
- **codex D** → move `manifest.rs` into gam-terms + wire its `mod` decl. NO COMPILE.

## Lead integration (after codex, I COMPILE — user authorized me to compile)
1. root `lib.rs`: `pub mod inference`→`pub use gam_inference as inference`; `pub mod report`→`pub use gam_report as report`; `pub mod config_resolve`→`pub use gam_config as config_resolve`. KEEP every downstream re-export (`pub use inference::{alo,generative,…}`, `pub use inference::hmc_io as hmc`, etc.) — they now resolve through gam_inference. Fix `init_parallelism` body `crate::inference::hmc_io::…`→`gam_inference::hmc_io::…`.
2. root `Cargo.toml`: add gam-inference/gam-report/gam-config path-deps; move inference-only externals (general-mcmc, statrs, rand_distr) OUT of root into gam-inference if root no longer uses them directly.
3. `git rm` the moved root files (src/inference, src/report, src/config_resolve.rs).
4. gam-cli / gam-pyffi: they consume `gam::inference`/`gam::report`/`gam::config_resolve` via the facade → still resolve, NO change needed (facade re-exports). Verify by compile.
5. **COMPILE GATE**: `cargo check --workspace` then `cargo test --no-run` (the 999 root tests MUST compile — the issue's hard requirement). Fix-forward misses (codex will miss edge cases; I patch). Audit codex output for reward-hacking (stubs/fakes) per the codex rule.
6. Push incrementally. When root is a thin facade and everything compiles, **close #1521**.

## Verification of payoff
After: editing `inference/` recompiles `gam-inference` + facade only — not the monolith. Confirms the #1521 thesis.
