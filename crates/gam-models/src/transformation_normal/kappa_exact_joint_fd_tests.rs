//! gam#979 — the exact-joint outer gradient of the transformation-normal
//! criterion, differenced against the criterion itself.
//!
//! `fit_transformation_normal` hands the spatial κ-optimizer a closure that
//! evaluates the profiled LAML criterion `V(ρ, ψ)` and its analytic gradient.
//! Every ψ-derivative ingredient of that gradient carries its own
//! finite-difference gate — the CTN ψ terms at fixed β, the `G_x(κ)` penalty
//! channel, the Duchon radial jets — but the ASSEMBLED gradient the line search
//! actually follows had none. The large-scale CTN preprocessor
//! (`duchon(pc1..pc16, centers=24, order=0, power=9, length_scale=1)`) then
//! spent every strong-Wolfe search walking a direction whose analytic slope
//! was `−74` while the criterion rose along it, fell back to Armijo
//! backtracking at ~55 evaluations per iteration, and never finished inside
//! its budget. That is what this gate measures: at the production chart's
//! orders, the ψ component of `∇V` against a Ridders-extrapolated central
//! difference of `V`, through the SAME geometry constructor the optimizer's
//! cache uses (`build_transformation_exact_geometry`).
//!
//! Every ρ component is differenced in the same pass, so a ρ↔ψ cross-term
//! defect cannot masquerade as a ψ-only one.

#![cfg(test)]

