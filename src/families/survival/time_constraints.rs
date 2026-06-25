//! Shared builder for survival-time derivative-guard monotonicity constraints.
//!
//! Both `survival_location_scale` and `survival_marginal_slope` need to turn a
//! derivative design block `D` and its row offsets `o` into the row-wise linear
//! inequality system
//!
//!   `D β + o ≥ guard`   ⇔   `D β ≥ guard − o`
//!
//! that enforces `q'(t) ≥ guard` (survival-time monotonicity / derivative
//! safety) inside the inner active-set / KKT machinery. The two families used to
//! carry byte-for-byte duplicates of this construction with two policy
//! differences buried in the copies:
//!
//! * the admissible guard range (`survival_location_scale` accepts a zero guard,
//!   `survival_marginal_slope` requires a strictly positive guard), and
//! * the feasibility slack used when deciding whether a row with no usable time
//!   coefficients can satisfy the guard from its offset alone.
//!
//! This module hosts the single implementation and makes both differences
//! explicit, configurable inputs ([`GuardPolicy`] and [`FeasibilityTolerance`]).
//! The builder is error-type agnostic: it reports structured failures via
//! [`GuardConstraintFailure`], and each family renders that into its own error
//! enum/wording through a small adapter. No family keeps a second copy.

use crate::matrix::DesignMatrix;
use crate::pirls::LinearInequalityConstraints;
use ndarray::{Array1, Array2};

/// Admissible range for the derivative guard, made explicit per family.
///
/// The guard is the lower bound the time derivative `q'(t)` must clear. A
/// `survival_location_scale` block can ride a degenerate `guard == 0` (a bare
/// non-negativity request), whereas `survival_marginal_slope` deliberately
/// rejects it: its row-wise representation is the *only* place the guard is
/// allowed to live, and a zero guard there would silently collapse the
/// monotonicity barrier it exists to enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardPolicy {
    /// Accept any finite `guard ≥ 0` (used by `survival_location_scale`).
    NonNegative,
    /// Require a finite `guard > 0`; reject zero (used by
    /// `survival_marginal_slope`).
    Positive,
}

impl GuardPolicy {
    /// True when `guard` satisfies this policy. Non-finite guards never satisfy
    /// any policy.
    #[inline]
    fn admits(self, guard: f64) -> bool {
        if !guard.is_finite() {
            return false;
        }
        match self {
            GuardPolicy::NonNegative => guard >= 0.0,
            GuardPolicy::Positive => guard > 0.0,
        }
    }

    /// Human-readable description of the admissible range, used in the
    /// structured guard-range failure so each family can render an accurate
    /// message without re-encoding the policy.
    #[inline]
    pub fn range_description(self) -> &'static str {
        match self {
            GuardPolicy::NonNegative => ">= 0",
            GuardPolicy::Positive => "> 0",
        }
    }
}

/// Feasibility slack used when a row carries no usable time coefficients and
/// must clear the guard from its offset alone.
///
/// Both families compare `offset + tol(offset, guard) ≥ guard` (equivalently
/// `guard − offset ≤ tol`). They use different slack factors of the same shape
/// `factor · (1 + max(|offset|, |guard|))`; the variants below name the active
/// policy directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeasibilityTolerance {
    /// `1e-12 · (1 + max(|offset|, |guard|))`.
    AbsoluteScaled,
    /// `256 · f64::EPSILON · (1 + max(|offset|, |guard|))` —
    /// `survival_marginal_slope`'s epsilon-scaled slack.
    EpsilonScaled,
}

impl FeasibilityTolerance {
    #[inline]
    fn slack(self, offset: f64, guard: f64) -> f64 {
        let scale = 1.0 + offset.abs().max(guard.abs());
        match self {
            FeasibilityTolerance::AbsoluteScaled => 1e-12 * scale,
            FeasibilityTolerance::EpsilonScaled => 256.0 * f64::EPSILON * scale,
        }
    }
}

/// Explicit policy bundle a family hands the shared builder.
#[derive(Clone, Copy, Debug)]
pub struct GuardConstraintPolicy {
    /// Admissible guard range for this family.
    pub guard_policy: GuardPolicy,
    /// Feasibility slack used for coefficient-free rows.
    pub feasibility: FeasibilityTolerance,
}

/// Structured failure produced by the shared builder. Each family maps this onto
/// its own error enum and wording in a thin adapter; the builder never commits
/// to a family's error vocabulary.
#[derive(Clone, Debug)]
pub enum GuardConstraintFailure {
    /// `design.nrows() != offsets.len()`.
    RowOffsetMismatch { rows: usize, offsets: usize },
    /// The guard value is outside the family's admissible range.
    GuardOutOfRange { guard: f64, range: &'static str },
    /// A derivative offset is non-finite at the given row.
    NonFiniteOffset { row: usize, offset: f64 },
    /// A derivative design entry is non-finite at the given cell.
    NonFiniteDesign { row: usize, col: usize },
    /// A row that cannot move the derivative (no time coefficients, or a zero
    /// derivative-design row) cannot clear the guard from its offset alone.
    InfeasibleRow {
        row: usize,
        offset: f64,
        guard: f64,
        /// True when the whole block has zero columns (no time coefficients at
        /// all); false when this individual derivative-design row is zero.
        no_time_coefficients: bool,
    },
}

/// Build the row-wise `D β ≥ guard − o` derivative-guard constraints shared by
/// the survival families.
///
/// Returns `Ok(None)` when no row needs an explicit constraint (every row is
/// already satisfied by its offset, or the block has no movable rows), and
/// `Ok(Some(_))` with the normalized constraint system otherwise. Each emitted
/// row is scaled by `max(‖row‖, |rhs|, 1)` so the downstream active-set
/// feasibility tolerance applies uniformly across rows of disparate magnitude.
///
/// Policy differences between families are supplied through `policy`; failures
/// are reported structurally so each family can render its own wording.
pub fn build_time_derivative_guard_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
    policy: GuardConstraintPolicy,
) -> Result<Option<LinearInequalityConstraints>, GuardConstraintFailure> {
    if design_derivative_exit.nrows() != derivative_offset_exit.len() {
        return Err(GuardConstraintFailure::RowOffsetMismatch {
            rows: design_derivative_exit.nrows(),
            offsets: derivative_offset_exit.len(),
        });
    }
    if !policy.guard_policy.admits(derivative_guard) {
        return Err(GuardConstraintFailure::GuardOutOfRange {
            guard: derivative_guard,
            range: policy.guard_policy.range_description(),
        });
    }

    let p = design_derivative_exit.ncols();
    if p == 0 {
        // No time coefficients at all: every row must clear the guard from its
        // offset alone, otherwise the guard is structurally infeasible.
        for (row, &offset) in derivative_offset_exit.iter().enumerate() {
            if !offset.is_finite() {
                return Err(GuardConstraintFailure::NonFiniteOffset { row, offset });
            }
            if offset + policy.feasibility.slack(offset, derivative_guard) < derivative_guard {
                return Err(GuardConstraintFailure::InfeasibleRow {
                    row,
                    offset,
                    guard: derivative_guard,
                    no_time_coefficients: true,
                });
            }
        }
        return Ok(None);
    }

    let dense = design_derivative_exit.to_dense();
    let mut active_rows: Vec<usize> = Vec::new();
    for row in 0..dense.nrows() {
        let offset = derivative_offset_exit[row];
        if !offset.is_finite() {
            return Err(GuardConstraintFailure::NonFiniteOffset { row, offset });
        }
        let mut row_norm_sq = 0.0_f64;
        for col in 0..p {
            let value = dense[[row, col]];
            if !value.is_finite() {
                return Err(GuardConstraintFailure::NonFiniteDesign { row, col });
            }
            row_norm_sq += value * value;
        }
        let required = derivative_guard - offset;
        if row_norm_sq <= 1e-24 {
            // A zero derivative-design row cannot be moved by any β; it must
            // already satisfy the guard from its offset alone.
            if required > policy.feasibility.slack(offset, derivative_guard) {
                return Err(GuardConstraintFailure::InfeasibleRow {
                    row,
                    offset,
                    guard: derivative_guard,
                    no_time_coefficients: false,
                });
            }
            continue;
        }
        active_rows.push(row);
    }

    if active_rows.is_empty() {
        return Ok(None);
    }

    let mut a = Array2::<f64>::zeros((active_rows.len(), p));
    let mut b = Array1::<f64>::zeros(active_rows.len());
    for (out_row, &src_row) in active_rows.iter().enumerate() {
        let row = dense.row(src_row);
        let rhs = derivative_guard - derivative_offset_exit[src_row];
        let row_norm = row.dot(&row).sqrt();
        let scale = row_norm.max(rhs.abs()).max(1.0);
        for col in 0..p {
            a[[out_row, col]] = dense[[src_row, col]] / scale;
        }
        b[out_row] = rhs / scale;
    }
    Ok(Some(
        LinearInequalityConstraints::new(a, b)
            .expect("derivative-guard constraint shape invariant"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const LS_POLICY: GuardConstraintPolicy = GuardConstraintPolicy {
        guard_policy: GuardPolicy::NonNegative,
        feasibility: FeasibilityTolerance::AbsoluteScaled,
    };
    const MS_POLICY: GuardConstraintPolicy = GuardConstraintPolicy {
        guard_policy: GuardPolicy::Positive,
        feasibility: FeasibilityTolerance::EpsilonScaled,
    };

    fn dense(rows: usize, cols: usize, data: &[f64]) -> DesignMatrix {
        DesignMatrix::from(Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap())
    }

    /// Representative non-degenerate input: both family policies share the same
    /// admissible-guard region (`guard > 0`) and the same constraint geometry,
    /// so the produced constraint matrices must be bit-for-bit identical. This
    /// is the core "one builder" parity assertion the issue asks for.
    #[test]
    fn parity_of_constraint_matrices_across_family_policies() {
        let design = dense(3, 2, &[2.0, 0.5, 1.0, -0.25, 4.0, 1.5]);
        let offsets = array![0.10, -0.20, 0.40];
        let guard = 0.05;

        let ls = build_time_derivative_guard_constraints(&design, &offsets, guard, LS_POLICY)
            .expect("location-scale policy must build")
            .expect("expected active rows");
        let ms = build_time_derivative_guard_constraints(&design, &offsets, guard, MS_POLICY)
            .expect("marginal-slope policy must build")
            .expect("expected active rows");

        assert_eq!(ls.a.shape(), ms.a.shape());
        assert_eq!(ls.b.len(), ms.b.len());
        for (x, y) in ls.a.iter().zip(ms.a.iter()) {
            assert_eq!(
                x, y,
                "constraint A entries must match exactly across policies"
            );
        }
        for (x, y) in ls.b.iter().zip(ms.b.iter()) {
            assert_eq!(
                x, y,
                "constraint b entries must match exactly across policies"
            );
        }

        // Row normalization invariant: A β ≥ b with each row scaled by
        // max(‖row‖, |rhs|, 1). Verify the scale on the first active row.
        let raw_row = design.to_dense().row(0).to_owned();
        let raw_rhs = guard - offsets[0];
        let scale = raw_row.dot(&raw_row).sqrt().max(raw_rhs.abs()).max(1.0);
        assert!((ls.a[[0, 0]] - raw_row[0] / scale).abs() < 1e-15);
        assert!((ls.a[[0, 1]] - raw_row[1] / scale).abs() < 1e-15);
        assert!((ls.b[0] - raw_rhs / scale).abs() < 1e-15);
    }

    /// The explicit guard-policy difference: a zero guard is admissible for the
    /// location-scale policy but rejected by the marginal-slope policy.
    #[test]
    fn guard_policy_difference_is_explicit() {
        let design = dense(2, 1, &[1.0, 1.0]);
        let offsets = array![0.0, 0.0];

        // Zero guard: location-scale accepts, marginal-slope rejects.
        let ls_zero = build_time_derivative_guard_constraints(&design, &offsets, 0.0, LS_POLICY);
        assert!(ls_zero.is_ok(), "non-negative policy must admit guard == 0");

        let ms_zero = build_time_derivative_guard_constraints(&design, &offsets, 0.0, MS_POLICY);
        match ms_zero {
            Err(GuardConstraintFailure::GuardOutOfRange { guard, range }) => {
                assert_eq!(guard, 0.0);
                assert_eq!(range, "> 0");
            }
            other => panic!("positive policy must reject guard == 0, got {other:?}"),
        }

        // Negative guard: rejected by both.
        for policy in [LS_POLICY, MS_POLICY] {
            match build_time_derivative_guard_constraints(&design, &offsets, -1.0, policy) {
                Err(GuardConstraintFailure::GuardOutOfRange { .. }) => {}
                other => panic!("negative guard must be rejected, got {other:?}"),
            }
        }
    }

    /// Coefficient-free block: rows must clear the guard from offsets alone, and
    /// the structured infeasibility failure carries the offending row.
    #[test]
    fn coefficient_free_feasibility_uses_policy_slack() {
        let design = dense(2, 0, &[]);

        // Offsets comfortably above the guard: feasible, no constraints.
        let ok =
            build_time_derivative_guard_constraints(&design, &array![1.0, 2.0], 0.5, MS_POLICY)
                .expect("feasible offsets must not error");
        assert!(
            ok.is_none(),
            "coefficient-free feasible block emits no rows"
        );

        // An offset below the guard by more than the slack: infeasible.
        match build_time_derivative_guard_constraints(&design, &array![1.0, 0.1], 0.5, MS_POLICY) {
            Err(GuardConstraintFailure::InfeasibleRow {
                row,
                no_time_coefficients,
                ..
            }) => {
                assert_eq!(row, 1);
                assert!(no_time_coefficients);
            }
            other => panic!("expected infeasible coefficient-free row, got {other:?}"),
        }
    }

    /// A zero derivative-design row (block has columns, but this row is all
    /// zeros) is reported as infeasible with `no_time_coefficients == false`.
    #[test]
    fn zero_design_row_reports_individual_infeasibility() {
        let design = dense(2, 1, &[0.0, 1.0]);
        let offsets = array![0.0, 0.0];
        match build_time_derivative_guard_constraints(&design, &offsets, 0.5, MS_POLICY) {
            Err(GuardConstraintFailure::InfeasibleRow {
                row,
                no_time_coefficients,
                ..
            }) => {
                assert_eq!(row, 0);
                assert!(!no_time_coefficients);
            }
            other => panic!("expected zero-row infeasibility, got {other:?}"),
        }
    }
}
