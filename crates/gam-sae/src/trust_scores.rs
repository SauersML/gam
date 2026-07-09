//! Per-row / per-atom trust scoring for the SAE-manifold research facade.
//!
//! Motivation
//! ----------
//! Each fitted atom carries a scalar `trust_score` (from the fit diagnostics:
//! tangent conditioning, coverage, activation frequency, …). Downstream
//! research consumers want a *per-row* trust: how trustworthy is the set of
//! atoms a given input row was actually routed to. The Python facade used to
//! assemble this from the raw assignment matrix; the SPEC keeps all numeric
//! policy in Rust, so the assembly lives here.
//!
//! Math
//! ----
//! Given an assignment matrix `A ∈ R^{N×K}` (row `i`'s soft membership in each
//! of the `K` atoms) and the per-atom trust vector `τ ∈ R^K`, only the
//! non-negative part of an assignment routes trust:
//!
//! ```text
//!     w_{ik}   = max(A_{ik}, 0)                        (routed weight)
//!     s_i      = Σ_k w_{ik}                            (row weight)
//!     p_{ik}   = w_{ik} / s_i     if s_i > 0, else 0   (row-normalized routing)
//!     C_{ik}   = p_{ik} · τ_k                          (per-atom trust credit)
//!     r_i      = Σ_k C_{ik}                            (row trust)
//! ```
//!
//! `r_i` is thus the routing-weighted average of the trust of the atoms row
//! `i` uses. A row with no positive assignment (dead row) contributes zero
//! credit and gets `r_i = 0` rather than a divide-by-zero.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Per-atom trust credit `C ∈ R^{N×K}` and per-row trust `r ∈ R^N` from an
/// assignment matrix and the per-atom trust vector. See the module docs for
/// the exact definition.
///
/// Errors when the assignment column count does not match the trust length.
pub fn row_trust_scores(
    assignments: ArrayView2<f64>,
    atom_trust: ArrayView1<f64>,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let (n_rows, n_atoms) = assignments.dim();
    if n_atoms != atom_trust.len() {
        return Err(format!(
            "trust score assignments/atom_trust mismatch: {} assignment columns vs {} trust entries",
            n_atoms,
            atom_trust.len()
        ));
    }

    let mut per_atom = Array2::<f64>::zeros((n_rows, n_atoms));
    let mut row = Array1::<f64>::zeros(n_rows);
    for i in 0..n_rows {
        // Row weight over the non-negative (routed) part of the assignment.
        let mut denom = 0.0_f64;
        for k in 0..n_atoms {
            let w = assignments[[i, k]].max(0.0);
            denom += w;
        }
        if denom > 0.0 {
            let mut r = 0.0_f64;
            for k in 0..n_atoms {
                let w = assignments[[i, k]].max(0.0);
                let credit = (w / denom) * atom_trust[k];
                per_atom[[i, k]] = credit;
                r += credit;
            }
            row[i] = r;
        }
        // else: dead row — per_atom stays zero, row stays zero.
    }
    Ok((row, per_atom))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn routing_weighted_average_of_atom_trust() {
        // Two atoms, trust [0.2, 0.8]. Row 0 routes 3:1 to atoms (0,1).
        let assignments = array![[3.0, 1.0], [0.0, 2.0]];
        let atom_trust = array![0.2, 0.8];
        let (row, per_atom) = row_trust_scores(assignments.view(), atom_trust.view()).unwrap();
        // Row 0: p = [0.75, 0.25]; credit = [0.15, 0.20]; row = 0.35.
        assert!((per_atom[[0, 0]] - 0.15).abs() < 1e-12);
        assert!((per_atom[[0, 1]] - 0.20).abs() < 1e-12);
        assert!((row[0] - 0.35).abs() < 1e-12);
        // Row 1: all weight on atom 1; credit = [0, 0.8]; row = 0.8.
        assert!((per_atom[[1, 0]]).abs() < 1e-12);
        assert!((per_atom[[1, 1]] - 0.8).abs() < 1e-12);
        assert!((row[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn negative_assignments_are_clipped_out() {
        // Negative membership does not route trust; only atom 1's +1 counts.
        let assignments = array![[-5.0, 1.0]];
        let atom_trust = array![0.3, 0.9];
        let (row, per_atom) = row_trust_scores(assignments.view(), atom_trust.view()).unwrap();
        assert!((per_atom[[0, 0]]).abs() < 1e-12);
        assert!((per_atom[[0, 1]] - 0.9).abs() < 1e-12);
        assert!((row[0] - 0.9).abs() < 1e-12);
    }

    #[test]
    fn dead_row_yields_zero_not_nan() {
        let assignments = array![[0.0, -1.0]];
        let atom_trust = array![0.3, 0.9];
        let (row, per_atom) = row_trust_scores(assignments.view(), atom_trust.view()).unwrap();
        assert_eq!(row[0], 0.0);
        assert_eq!(per_atom[[0, 0]], 0.0);
        assert_eq!(per_atom[[0, 1]], 0.0);
    }

    #[test]
    fn column_mismatch_errors() {
        let assignments = array![[1.0, 2.0, 3.0]];
        let atom_trust = array![0.3, 0.9];
        assert!(row_trust_scores(assignments.view(), atom_trust.view()).is_err());
    }
}
