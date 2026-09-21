//! Shared builder for survival-time derivative-guard monotonicity constraints.
//!
//! Both `survival_location_scale` and `survival_marginal_slope` need to turn a
//! derivative design block `D` and its row offsets `o` into the row-wise linear
//! inequality system
//!
//!   `D β + o ≥ guard`   ⇔   `D β ≥ guard − o`
//!
//! that enforces `q'(t) ≥ guard` (survival-time monotonicity / derivative
//! safety) inside the inner active-set / KKT machinery. This module hosts the
//! single implementation. The one policy difference between families, the
//! admissible guard range, is an explicit [`GuardPolicy`] input. Everything
//! else is shared:
//!
//! * [`derivative_row_is_immovable`] decides which rows no `β` can move, and
//!   the marginal-slope feasibility validator reads the same predicate;
//! * [`derivative_guard_feasibility_band`] is the one slack for "does `q'`
//!   clear the guard", used here for rows that must clear it from their
//!   offset alone and by the marginal-slope likelihood-domain predicate.
//!
//! The builder is error-type agnostic: it reports structured failures via
//! [`GuardConstraintFailure`], and each family renders that into its own error
//! enum/wording through a small adapter. No family keeps a second copy.

use gam_linalg::matrix::DesignMatrix;
use gam_solve::pirls::LinearInequalityConstraints;
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
    pub(crate) fn range_description(self) -> &'static str {
        match self {
            GuardPolicy::NonNegative => ">= 0",
            GuardPolicy::Positive => "> 0",
        }
    }
}

/// Width of the band below `guard` in which a derivative value `q_prime` still
/// counts as clearing the guard: `q_prime + band ≥ guard`.
///
/// A movable row reaches the solver as `a·β ≥ rhs` scaled by
/// `max(‖row‖, |rhs|, 1)`, and the active-set solver certifies that scaled
/// row to [`gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL`]. So a
/// converged active constraint legitimately sits up to that tolerance times
/// the row scale on the infeasible side of the exact bound. The band is that
/// contract with the row scale replaced by the value-side magnitude
/// `1 + max(|q'|, |guard|)`, which does not depend on a design row, so it
/// applies equally to a row with no movable coefficients. The factor 4
/// absorbs the re-evaluation of `q'` from `β` against the scaled residual the
/// solver reported (#788).
///
/// Every "does `q'` clear the guard" test in the survival families reads this
/// band: the offset-only test for immovable rows here, and the marginal-slope
/// likelihood-domain predicate `survival_derivative_guard_violated`. So a row
/// the builder admits from its offset is never refused by the row kernel for
/// the same `q' = offset`, and vice versa.
#[inline]
pub(crate) fn derivative_guard_feasibility_band(q_prime: f64, guard: f64) -> f64 {
    4.0 * gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        * (1.0 + q_prime.abs().max(guard.abs()))
}

/// True when a row that no `β` can move clears the guard from its offset.
#[inline]
fn offset_clears_guard(offset: f64, guard: f64) -> bool {
    offset + derivative_guard_feasibility_band(offset, guard) >= guard
}

/// True when no coefficient vector can move this derivative-design row: every
/// entry is exactly zero.
///
/// This is the only unit-free "cannot move" test. A change of time unit
/// rescales `D`, so any positive cutoff on `‖row‖` would classify the same row
/// as immovable in one unit system and movable in another. A row with any
/// nonzero entry is moved by `β` and goes to the solver as a constraint.
#[inline]
pub(crate) fn derivative_row_is_immovable<'a>(row: impl IntoIterator<Item = &'a f64>) -> bool {
    row.into_iter().all(|&value| value == 0.0)
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
/// The family's admissible guard range is supplied through `guard_policy`;
/// failures are reported structurally so each family can render its own wording.
pub fn build_time_derivative_guard_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
    guard_policy: GuardPolicy,
) -> Result<Option<LinearInequalityConstraints>, GuardConstraintFailure> {
    if design_derivative_exit.nrows() != derivative_offset_exit.len() {
        return Err(GuardConstraintFailure::RowOffsetMismatch {
            rows: design_derivative_exit.nrows(),
            offsets: derivative_offset_exit.len(),
        });
    }
    if !guard_policy.admits(derivative_guard) {
        return Err(GuardConstraintFailure::GuardOutOfRange {
            guard: derivative_guard,
            range: guard_policy.range_description(),
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
            if !offset_clears_guard(offset, derivative_guard) {
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
        if let Some(col) = (0..p).find(|&col| !dense[[row, col]].is_finite()) {
            return Err(GuardConstraintFailure::NonFiniteDesign { row, col });
        }
        if derivative_row_is_immovable(dense.row(row)) {
            // No β moves this row; it must already satisfy the guard from its
            // offset alone.
            if !offset_clears_guard(offset, derivative_guard) {
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

    const LS_POLICY: GuardPolicy = GuardPolicy::NonNegative;
    const MS_POLICY: GuardPolicy = GuardPolicy::Positive;

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
    fn coefficient_free_feasibility_uses_offset_band() {
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

    /// A change of time unit rescales `D`, the offsets and the guard by the
    /// same `c`, so the accept/refuse verdict and the set of constrained rows
    /// must not depend on `c` (#3766). Row 0 is tiny but nonzero (the old
    /// `‖row‖² ≤ 1e-24` cutoff called it immovable at `c = 1` and refused the
    /// block, yet called it movable at `c = 1e3`); row 1 is exactly zero with
    /// an offset above the guard; row 2 is an ordinary movable row.
    #[test]
    fn verdict_is_invariant_to_a_change_of_time_unit_3766() {
        let base_design = [1e-13, 0.0, 0.0, 0.0, 0.5, 2.0];
        let base_offsets = [0.0, 0.3, -0.1];
        let base_guard = 0.2;
        for c in [1e-3, 1.0, 1e3] {
            let design = dense(
                3,
                2,
                &base_design.iter().map(|v| v * c).collect::<Vec<_>>(),
            );
            let offsets = Array1::from_iter(base_offsets.iter().map(|v| v * c));
            for policy in [LS_POLICY, MS_POLICY] {
                let built = build_time_derivative_guard_constraints(
                    &design,
                    &offsets,
                    base_guard * c,
                    policy,
                )
                .unwrap_or_else(|failure| panic!("c={c}: must build, got {failure:?}"))
                .unwrap_or_else(|| panic!("c={c}: rows 0 and 2 are movable"));
                assert_eq!(
                    built.a.nrows(),
                    2,
                    "c={c}: exactly the two movable rows are constrained"
                );
            }

            // The exactly-zero row with its offset below the guard is refused
            // at every scale.
            let short = Array1::from_iter(
                [0.0, 0.1, -0.1].iter().map(|v: &f64| v * c),
            );
            match build_time_derivative_guard_constraints(&design, &short, base_guard * c, MS_POLICY)
            {
                Err(GuardConstraintFailure::InfeasibleRow {
                    row,
                    no_time_coefficients,
                    ..
                }) => {
                    assert_eq!(row, 1, "c={c}");
                    assert!(!no_time_coefficients, "c={c}");
                }
                other => panic!("c={c}: zero row below the guard must be refused, got {other:?}"),
            }
        }
    }

    /// Both family policies admit an offset-only row by the same band, and it
    /// is the band the marginal-slope likelihood-domain predicate applies to
    /// `q' = offset`: the build-time and evaluation-time verdicts coincide on
    /// either side of the band edge.
    #[test]
    fn offset_band_matches_the_marginal_slope_domain_predicate_3766() {
        let design = dense(1, 0, &[]);
        for guard in [1e-6, 0.5, 40.0] {
            let edge = guard - derivative_guard_feasibility_band(guard, guard);
            for offset in [
                guard,
                edge + 1e-3 * (guard - edge),
                edge - 1e-3 * (guard - edge),
                guard - 1e-3,
            ] {
                let domain_ok =
                    !crate::survival::marginal_slope::survival_derivative_guard_violated(
                        offset, guard,
                    );
                for policy in [LS_POLICY, MS_POLICY] {
                    let built = build_time_derivative_guard_constraints(
                        &design,
                        &array![offset],
                        guard,
                        policy,
                    );
                    assert_eq!(
                        built.is_ok(),
                        domain_ok,
                        "guard={guard:e}, offset={offset:e}: builder and domain predicate disagree"
                    );
                }
            }
        }
    }
}
