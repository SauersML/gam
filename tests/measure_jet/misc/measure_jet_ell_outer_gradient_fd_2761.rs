//! #2761: the FULL joint-REML gradient w.r.t. the measure-jet representer range
//! `ψ = ln ℓ` must match a central finite difference of the COMPLETE outer
//! criterion — not just of the penalty block.
//!
//! `learn_length_scale` is on by default (#2761), so every `mjs` fit now enrolls
//! `ln ℓ` as a real outer coordinate. Five fixtures in this target refuse with
//!
//! ```text
//!   line_search=StepSizeTooSmall after 50 attempt(s)
//!   [the direction descended but no step improved the objective]
//! ```
//!
//! at a checkpoint that reproduces to nine digits across unrelated changes. That
//! message has exactly two causes — a gradient that disagrees with its
//! objective, or an objective that is not smooth in the coordinate — and the
//! ℓ-profile probe already excluded one variant of the second (the design
//! refused to BUILD past `ℓ ≈ 2.8`; fixed by pulling the energy back through its
//! factor). This test settles the first, on the same fixture, through the
//! generic outer runner's structured analytic-vs-FD audit: the same instrument
//! `matern_2d_iso_kappa_outer_gradient_fd` uses for the Matérn `log κ`.
//!
//! `ln ℓ` is the exact analogue of that `log κ` — the module header calls it
//! "matérn's `log_kappa` analog" — so it is held to the same standard.

// `measure_jet_perf_parity`'s fixture: a 1-D curve in 3-D, the shape whose
// outer search refuses.

