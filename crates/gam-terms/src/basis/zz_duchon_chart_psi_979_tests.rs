//! gam#979 — the hybrid Duchon design ψ-derivative under the kernel chart.
//!
//! In high dimension at a large spectral power the raw hybrid Duchon–Matérn
//! kernel underflows (its spectral normalization is ~1e-15 at `d = 16`,
//! `s = 9`), and the forward basis ships the kernel block multiplied by the
//! chart amplitude `α(ψ) = 1/max|K|` (`duchon_kernel_chart`). The design the
//! REML criterion is built on is therefore `α(ψ)·K(ψ)`, and its ψ-derivative
//! is `α(K_ψ + (ln α)_ψ K)`, not `K_ψ`. Before this gate existed, the
//! derivative operator formed `K_ψ` alone: at the large-scale benchmark's
//! `duchon(pc1..pc16, order=0, power=9, length_scale=1)` that is ~1e-15 of
//! the true derivative, the analytic outer gradient silently dropped every
//! κ-dependence that enters through the design, and the κ line search walked
//! uphill on every trial (the `gam fit --transformation-normal` timeout).
//!
//! The gate differences the FORWARD design the basis actually ships, through
//! its own frozen chart, against the operator's materialized first and second
//! ψ-derivatives. The low-dimensional control has `α = 1` and pins that the
//! chart is inert there; the 16-D fixture asserts `α ≠ 1` so it cannot pass
//! vacuously.

#![cfg(test)]

use ndarray::{Array2, ArrayView2};

use super::*;

