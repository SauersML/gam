# gam-inference path-rewrite spec (#1521) — NO COMPILE

Files in `crates/gam-inference/src/` were moved verbatim out of root `gam` `src/inference/`.
Inside the new crate, `inference` IS the crate root. Rewrite ONLY module-path prefixes; change NO logic, add/remove NOTHING, stub NOTHING.

## Rewrite rules (apply as whole-segment path replacements)
- `crate::inference::`  → `crate::`
- `gam::inference::`    → `crate::`
- `gam::hmc::`          → `crate::hmc_io::`
- `crate::bail_invalid_estim` / `gam::bail_invalid_estim` → `gam_problem::bail_invalid_estim`
- `crate::types` / `gam::types`               → `gam_problem::types`
- `crate::model_types` / `gam::model_types`   → `gam_solve::model_types`
- `crate::estimate` / `gam::estimate`         → `gam_solve::estimate`
- `crate::solver` / `gam::solver`             → `gam_solve`
- `crate::pirls` / `gam::pirls`               → `gam_solve::pirls`
- `crate::mixture_link` / `gam::mixture_link` → `gam_solve::mixture_link`
- `crate::psis`                               → `gam_solve::psis`
- `crate::families` / `gam::families`         → `gam_models`
- `crate::basis`                              → `gam_terms::basis`
- `crate::term_builder`                       → `gam_terms::term_builder`
- `crate::terms`                              → `gam_terms`
- `crate::smooth` / `gam::smooth`             → `gam_terms::smooth`
- `gam::construction`                         → `gam_terms::construction`
- `crate::data` / `gam::data`                 → `gam_data`
- `crate::faer_ndarray` / `gam::faer_ndarray` → `gam_linalg::faer_ndarray`
- `crate::matrix` / `gam::matrix`             → `gam_linalg::matrix`
- `crate::linalg` / `gam::linalg`             → `gam_linalg`

## KEEP UNCHANGED (do NOT rewrite — these are correct already)
- `crate::probability`, `crate::quadrature`, `crate::generative`, `crate::sample`, `crate::posterior`, `crate::hmc_io`, `crate::model_comparison`, `crate::rho_posterior`, `crate::functionals`, `crate::skovgaard`, `crate::util`, and any `crate::<X>` where `<X>.rs` is a sibling file in `crates/gam-inference/src/` — these are this crate's OWN submodules.
- `gam_solve::quadrature`, `gam_solve::inference::`, `gam_terms::inference::`, `gam_sae::inference::` — already correct (other crates' modules); do NOT touch.
- All already-correct `gam_problem::`, `gam_solve::`, `gam_models::`, `gam_linalg::`, `gam_gpu::`, `gam_math::`, `gam_data::`, `gam_sae::`, `super::`, and all third-party crates (ndarray, statrs, rand, faer, rayon, libm, general_mcmc, serde, log, wide).

## Ambiguity rule
For a bare `gam::X` not listed above: if `X.rs` is a sibling file in this crate's src/, rewrite to `crate::X`; otherwise map to the owning sub-crate (`gam_solve`/`gam_terms`/`gam_models`/`gam_linalg`/`gam_problem`).
