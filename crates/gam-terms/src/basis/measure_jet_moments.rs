//! Measure-jet frame data interface: per-cell frozen-weight polynomial
//! moment tables with a binomial-shift merge monoid
//! (`docs/measure_jet_frame.md`, §2 "Data interface: moments or
//! nothing").
//!
//! This module aggregates caller-computed weights into order-0..2 coordinate
//! moments. Those tables exactly determine polynomial couplings under the
//! same frozen weights, including the local affine sufficient statistics used
//! by `measure_jet_smooth.rs`. They do NOT exactly determine Gaussian
//! transforms at moved kernel centers: support curves, Gaussian Gram entries,
//! and Gaussian `XᵀWX` products need their own kernel pass or a separately
//! controlled approximation. Truncation does NOT live here either: the caller
//! computes the Gaussian weights `w_i` (mass × kernel profile, with whatever
//! cutoff its explicit `e^{−ρ²/2}` tolerance budget licenses) and this module
//! only aggregates what it is handed.
//!
//! # The monoid
//!
//! A table holds, per response channel `g` and per coordinate multi-index
//! `α` with `|α| ≤ 2`, the centered moment `μ_α = Σ_i w_i g_i (x_i − c)^α`
//! about the cell reference point `c`. The binomial shift
//!
//! ```text
//!   μ′_α = Σ_{β ≤ α}  C(α, β) (c − c′)^{α−β} μ_β
//! ```
//!
//! re-expresses the same frozen-weight polynomial table about any other
//! center `c′` exactly as a finite polynomial identity. It does not move the
//! Gaussian kernel center or recompute weights. Merging two tables with
//! already-compatible frozen weights is therefore "recenter to a common
//! reference, add componentwise":
//! an associative, commutative monoid whose identity is the empty (all-zero)
//! table at any center. Exact distributed fitting, exact online updates, and
//! bit-reproducibility under sorted reduction are corollaries of that one
//! algebraic fact ([`merge_moment_tables`] is a monoid homomorphism from
//! disjoint row sets under union to tables under ⊕).
//!
//! # Determinism / bit-exactness convention (sorted reduction)
//!
//! Floating-point addition is commutative but not associative, so the monoid
//! laws hold algebraically while bit-patterns depend on reduction ORDER.
//! This module pins one order everywhere:
//!
//! - [`accumulate_moment_table`] splits rows into fixed-size chunks
//!   ([`MEASURE_JET_MOMENT_CHUNK_ROWS`], never derived from thread count),
//!   accumulates each chunk sequentially in row order, and folds the chunk
//!   partials sequentially in chunk-index order — the sorted reduction. The
//!   result is bit-identical across runs, machines, and rayon pool sizes.
//! - [`recenter_moment_table`] evaluates the shift in ONE fixed expression
//!   order (documented at the site).
//! - [`merge_moment_tables`] canonically orients its operands by the
//!   lexicographic total order on centers (`f64::total_cmp` per coordinate),
//!   so `a ⊕ b` and `b ⊕ a` execute the SAME instruction stream and are
//!   bit-identical for arbitrary inputs.
//!
//! Cross-GROUPING bit-identity — `(A⊕B)⊕C` vs `A⊕(B⊕C)` — additionally
//! requires the moment arithmetic itself to be exact; the in-module tests
//! pin it on dyadic lattices (integer coordinates/channels, dyadic weights),
//! where every product and sum is exactly representable, and callers
//! reducing many chunks get run-to-run determinism by folding in chunk-index
//! order exactly as the accumulator does.
//!
//! # 1:1 contract with `assemble_weighted_forms`
//!
//! [`jet_sufficient_stats`] reproduces, in closed form from a stored table
//! whose weights were computed for the same center and scale, exactly the
//! local-fit quantities the current workhorse
//! (`measure_jet_smooth.rs::assemble_weighted_forms`) computes from raw
//! points per (center, scale) block: the kernel mass `q`, the dimensionless
//! weighted feature mean `a_mean`, the dimensionless slope Gram
//! `G = Φ̃ᵀWΦ̃/q`, the weighted channel mean `uᵀv`, and the exact-projection
//! right-hand side `Bᵀv/q` — so the substrate can later replace that
//! same-center point loop without changing a single number.

use ndarray::{Array1, Array2};

/// The local jet-fit sufficient statistics read off one table — exactly the
/// per-block quantities `assemble_weighted_forms` (measure_jet_smooth.rs)
/// computes from raw points when the table weights are frozen at the same
/// center and scale, reproduced in closed form from stored moments.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureJetJetStats {
    /// Kernel mass `q = Σ w_i` (unit-channel zeroth moment).
    pub q: f64,
    /// Weighted mean of the requested value channel: `uᵀv = m0[ch]/q`.
    pub mean: f64,
    /// Dimensionless slope Gram `G = Φ̃ᵀWΦ̃/q = m2[0]/(qε²) − ā·āᵀ` with
    /// `ā = m1[0]/(qε)` (`Φ` rows are `(x_i − c)/ε`).
    pub gram: Array2<f64>,
    /// Local-fit right-hand side `Bᵀv/q = m1[ch]/(qε) − ā·(m0[ch]/q)` — the
    /// vector the exact weighted affine projection consumes.
    pub cross: Array1<f64>,
}

