//! Cyclic (periodic) B-spline basis and finite-difference penalty.
//!
//! For a periodic covariate — angle, time-of-day, day-of-year — December must
//! meet January without a seam. We bend the coefficient axis into a ring so
//! the smooth has no endpoints to privilege: the last coefficient is the first
//! one's neighbour, full stop.
//!
//! This module owns the self-contained periodic-domain math for 1-D cyclic
//! smooths: the wrap-around finite-difference penalty `S = D'D`, the half-open
//! period folding used by every periodic evaluator, the period-aware great-arc
//! distance, the uniform cyclic knot vector, and the dense cyclic B-spline
//! design assembly. Callers in `basis` handle term specs, streaming chunking,
//! and downstream matrix wiring.

use super::{BasisError, PeriodicBSplineBasisSpec, build_periodic_bspline_basis_1d};
use gam_linalg::faer_ndarray::fast_ata;
use ndarray::{Array1, Array2, ArrayView1};

/// Creates a cyclic finite-difference penalty `S = D' D`.
///
/// Unlike the ordinary P-spline Reinsch penalty, every difference stencil wraps
/// around the coefficient ring. For `order = 2`, each row is
/// `β_i - 2β_{i+1} + β_{i+2}` modulo the number of coefficients, so the
/// constant vector is the only null direction and no endpoint is privileged.
pub fn create_cyclic_difference_penalty_matrix(
    num_basis_functions: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_basis_functions {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_basis_functions,
        });
    }

    let mut d = Array2::<f64>::eye(num_basis_functions);
    for _ in 0..order {
        let previous = d;
        d = Array2::<f64>::zeros((num_basis_functions, num_basis_functions));
        for i in 0..num_basis_functions {
            let next = (i + 1) % num_basis_functions;
            for j in 0..num_basis_functions {
                d[[i, j]] = previous[[next, j]] - previous[[i, j]];
            }
        }
    }
    Ok(fast_ata(&d))
}

/// Creates the OPEN (non-wrapping) finite-difference penalty `S = D'D` over the
/// same coefficient ring, but with the difference stencils truncated at the
/// endpoints instead of closed across the seam.
///
/// This is the `γ = 0` (interval) end of the closure family: the constant *and*
/// the low-order polynomial tails are unpenalised at the boundary, so the two
/// ends are free to drift apart. `S_circle = S_open + S_wrap`, where `S_wrap`
/// is the closing-edge contribution the cyclic penalty adds on top.
pub fn create_open_difference_penalty_matrix(
    num_basis_functions: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_basis_functions {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_basis_functions,
        });
    }
    // Banded forward-difference operator with `num_basis - order` rows: the
    // ordinary open P-spline Reinsch difference, no wrap.
    let rows = num_basis_functions - order;
    let mut d = Array2::<f64>::zeros((rows, num_basis_functions));
    for i in 0..rows {
        // Row i is the order-th forward difference centred at i:
        // coefficients are the signed binomials (-1)^{order-j} C(order, j).
        for j in 0..=order {
            let sign = if (order - j) % 2 == 0 { 1.0 } else { -1.0 };
            d[[i, i + j]] = sign * binomial(order, j);
        }
    }
    Ok(fast_ata(&d))
}

/// The closure-aware difference penalty `S(γ) = S_open + c(γ)·S_wrap` and its
/// first/second γ-derivatives, where `S_wrap = S_circle − S_open` is the
/// seam-closing contribution and `c(γ)` is the boundary-conductance interpolant
/// (`c(0)=0`, `c(1)=1`). At `γ = 1` this is exactly the cyclic penalty; at
/// `γ = 0` it is exactly the open penalty. Returns `(S, ∂S/∂γ, ∂²S/∂γ²)`.
///
/// This is the penalty-moving MVP of the closure family (#1015); the
/// support-moving period-extension basis lives in
/// [`gam_geometry::closure_family`]. Both ride the same ψ-channel pattern as
/// `M_κ` (#944).
pub fn create_closure_difference_penalty_jet(
    num_basis_functions: usize,
    order: usize,
    gamma: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), BasisError> {
    let s_open = create_open_difference_penalty_matrix(num_basis_functions, order)?;
    let s_circle = create_cyclic_difference_penalty_matrix(num_basis_functions, order)?;
    let s_wrap = &s_circle - &s_open;
    Ok(gam_geometry::conductance_penalty_jet(
        &s_open, &s_wrap, gamma,
    ))
}

/// Binomial coefficient `C(n, k)` for the small penalty orders used here.
pub(crate) fn binomial(n: usize, k: usize) -> f64 {
    let mut acc = 1.0_f64;
    for i in 0..k {
        acc = acc * (n - i) as f64 / (i + 1) as f64;
    }
    acc
}

#[inline]
pub(crate) fn wrap_to_period(x: f64, start: f64, period: f64) -> f64 {
    // Keep wrapped values numerically inside the half-open period
    // `[start, start + period)`. `rem_euclid` can return exactly `period`
    // after extreme-roundoff cancellation (e.g. `(start + period - eps - start)`
    // rounding up to `period`), which lifts the result to `start + period`
    // and pushes basis evaluators that assume `x < start + period` over the
    // right knot. Fold that boundary back to `start` so every caller — the
    // dense cyclic B-spline evaluator, the periodic Duchon kernel matrix,
    // and the cylinder/torus tensor margins — agrees with the
    // `wrap_periodic_phase` convention used by the derivative path.
    let offset = (x - start).rem_euclid(period);
    if offset >= period {
        start
    } else {
        start + offset
    }
}

#[inline]
pub(crate) fn cyclic_distance_1d(x: f64, c: f64, period: f64) -> f64 {
    let delta = (x - c).abs().rem_euclid(period);
    delta.min(period - delta)
}

/// Knot anchor for the uniform cyclic grid: the supplied domain origin.
///
/// Two symmetries compete here and are PROVABLY incompatible for any anchor
/// that is a pure function of `(start, period)`:
///
/// * translation equivariance — shifting the data AND the declared domain by
///   `c` must reproduce the identical design (the model cannot depend on the
///   coordinate origin of the covariate, e.g. day-of-year vs days-since-epoch);
/// * sub-knot seam invariance (#1593) — re-declaring `period_start` for FIXED
///   data should leave the spanned function space unchanged.
///
/// Both transformations present to this function as the same input change
/// (`start → start + c`), differing only in whether the data moved with it, so
/// no data-independent anchor can satisfy both. The previous #1593 cure snapped
/// the knots to the absolute lattice `{ m·h : m ∈ ℤ }` through the global zero
/// — sub-knot seam invariant, but translation NON-equivariant: translating data
/// and domain by a non-multiple of `h` (e.g. `0.37h`) left the knots pinned to
/// the old lattice and changed the fitted function space (span residual ~12% of
/// design energy). Translation equivariance is the property a regression basis
/// must have, so the knots anchor at the DECLARED domain origin: `period_start`
/// is an explicit gauge choice of the model specification, exactly like knot
/// placement for an open spline. A whole-knot seam shift (`start → start + m·h`)
/// remains an exact circulant relabel of the same basis.
#[inline]
pub(crate) fn cyclic_knot_anchor(start: f64, period: f64, num_basis: usize) -> (f64, f64) {
    let h = period / num_basis as f64;
    (start, h)
}

pub(crate) fn cyclic_uniform_knot_vector(
    start: f64,
    end: f64,
    degree: usize,
    num_basis: usize,
) -> Array1<f64> {
    let period = end - start;
    let (anchor, h) = cyclic_knot_anchor(start, period, num_basis);
    let total_knots = num_basis + 2 * degree + 1;
    Array1::from_iter((0..total_knots).map(|i| anchor + (i as f64 - degree as f64) * h))
}

pub(crate) fn create_cyclic_bspline_basis_dense(
    data: ArrayView1<'_, f64>,
    start: f64,
    end: f64,
    degree: usize,
    num_basis: usize,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    if end <= start {
        return Err(BasisError::InvalidRange(start, end));
    }
    if num_basis <= degree {
        crate::bail_invalid_basis!(
            "cyclic B-spline basis requires more basis functions ({num_basis}) than degree ({degree})"
        );
    }
    let period = end - start;
    let knots = cyclic_uniform_knot_vector(start, end, degree, num_basis);

    // Anchor the cardinal-basis phase to the SAME grid the knot vector uses.
    // `cyclic_uniform_knot_vector` places the knots via `cyclic_knot_anchor`;
    // the periodic evaluator centers its cardinal functions at `origin + m·h`,
    // so both must resolve the anchor through the one shared helper or the
    // evaluated basis phase-shifts relative to the knot grid it is supposed to
    // realize (the internal-inconsistency half of #1593).
    let (anchor, _h) = cyclic_knot_anchor(start, period, num_basis);

    // Evaluate the cyclic cardinal basis directly on the circle rather than
    // folding an open B-spline design and summing columns modulo `num_basis`.
    // The open-knot construction is sensitive to its half-open endpoint
    // convention: at `x == start + period` the wrapped value lands exactly on
    // the left boundary of an open spline and the endpoint row can differ from
    // the row at `start` (the tensor-margin seam bug in #1800). The cardinal
    // evaluator owns the periodic phase arithmetic and normalizes the wrapped
    // translates of each cardinal B-spline, so `start` and `end` are identical
    // by construction and prediction uses the same runtime basis as fitting.
    let cyclic = build_periodic_bspline_basis_1d(
        data,
        &PeriodicBSplineBasisSpec {
            degree,
            num_basis,
            period,
            origin: anchor,
            // The value evaluator does not use the penalty order; pass the
            // standard cyclic smooth order so spec validation remains shared.
            penalty_order: 2.min(num_basis - 1),
        },
    )?;
    Ok((cyclic, knots))
}

#[cfg(test)]
mod closure_tests {
    use super::*;

    /// DIAGNOSTIC (#1593): under a whole-knot seam shift `start' = start + h`,
    /// the cyclic basis evaluated on the SAME physical angles must be a rigid
    /// cyclic permutation of the un-shifted basis: `B'(θ) = B(θ)·P`. If this
    /// fails, the per-anchor designs are not gauge-related and no constraint can
    /// restore invariance.
    #[test]
    fn cyclic_basis_rigid_rotation_under_whole_knot_shift() {
        let degree = 3usize;
        let num_basis = 8usize;
        let start = 0.0_f64;
        let period = std::f64::consts::TAU;
        let h = period / num_basis as f64;
        let thetas = Array1::from_iter((0..40).map(|i| (i as f64 + 0.3) / 40.0 * period));
        let (b0, _) = create_cyclic_bspline_basis_dense(
            thetas.view(),
            start,
            start + period,
            degree,
            num_basis,
        )
        .unwrap();
        let (b1, _) = create_cyclic_bspline_basis_dense(
            thetas.view(),
            start + h,
            start + h + period,
            degree,
            num_basis,
        )
        .unwrap();
        // Try every cyclic shift; the best should match to round-off if the
        // basis rotates rigidly.
        let mut best = f64::INFINITY;
        let mut best_shift = 0usize;
        for shift in 0..num_basis {
            let mut maxerr = 0.0_f64;
            for r in 0..b0.nrows() {
                for j in 0..num_basis {
                    let permuted = b0[[r, (j + shift) % num_basis]];
                    maxerr = maxerr.max((b1[[r, j]] - permuted).abs());
                }
            }
            if maxerr < best {
                best = maxerr;
                best_shift = shift;
            }
        }
        eprintln!("[cyclic-rigid] best cyclic-shift match err={best:.3e} at shift={best_shift}");
        assert!(
            best < 1e-10,
            "cyclic basis is NOT a rigid cyclic permutation under a whole-knot seam shift: \
             best max|ΔB| over all {num_basis} shifts = {best:.3e} (shift {best_shift})"
        );
    }

    /// GUARD: the cyclic basis must be translation EQUIVARIANT — shifting the
    /// data AND the declared domain by the same offset `c` (including sub-knot
    /// offsets like `0.37h`) must reproduce the identical design row-for-row,
    /// because the model cannot depend on the coordinate origin of the
    /// covariate. This is the symmetry the old absolute-lattice anchor
    /// (`start − start.rem_euclid(h)`) broke: unless `c` was a multiple of the
    /// knot spacing the knots stayed pinned to the old lattice while the data
    /// moved, changing the fitted function space (~12% span residual at
    /// `c = 0.37h`). It is mathematically incompatible with sub-knot
    /// seam-shift invariance for a data-independent anchor (both present as
    /// `start → start + c`), so `period_start` is an explicit gauge of the
    /// model spec and the knots anchor at the declared domain origin.
    #[test]
    fn cyclic_basis_translation_equivariant() {
        let degree = 3usize;
        let num_basis = 12usize;
        let period = std::f64::consts::TAU;
        let h = period / num_basis as f64;
        let thetas = Array1::from_iter((0..200).map(|i| (i as f64 + 0.123) / 200.0 * period));
        let (reference, ref_knots) =
            create_cyclic_bspline_basis_dense(thetas.view(), 0.0, period, degree, num_basis)
                .unwrap();
        for frac in [0.37_f64, 0.5, 0.81, 1.0, 1.5, 2.8, 1.0e6 + 0.37] {
            let c = frac * h;
            let shifted = thetas.mapv(|t| t + c);
            let (bs, knots) = create_cyclic_bspline_basis_dense(
                shifted.view(),
                c,
                c + period,
                degree,
                num_basis,
            )
            .unwrap();
            let mut maxerr = 0.0_f64;
            for r in 0..reference.nrows() {
                for j in 0..num_basis {
                    maxerr = maxerr.max((bs[[r, j]] - reference[[r, j]]).abs());
                }
            }
            let mut knot_err = 0.0_f64;
            for (k_ref, k_new) in ref_knots.iter().zip(knots.iter()) {
                knot_err = knot_err.max((k_new - (k_ref + c)).abs());
            }
            eprintln!(
                "[cyclic-translate] c={c:.4} (frac {frac}) design err={maxerr:.3e} knot err={knot_err:.3e}"
            );
            // Round-off of `t + c` / `start + c` grows with |c|; allow only
            // that representation error beyond the exact-arithmetic bound.
            let ulp = 1e-9 * c.abs().max(1.0);
            assert!(
                maxerr < 1e-9 + ulp,
                "cyclic basis is NOT translation equivariant at offset frac={frac}: \
                 max row error {maxerr:.3e} (knots must anchor to the declared domain origin)"
            );
            assert!(
                knot_err < 1e-9 + ulp,
                "cyclic knots did not translate with the domain at offset frac={frac}: \
                 max knot drift {knot_err:.3e}"
            );
        }
    }

    /// At `γ = 0` the closure penalty is the open penalty; at `γ = 1` it is the
    /// cyclic penalty. The wrap piece carries the seam.
    #[test]
    pub(crate) fn closure_penalty_interpolates_open_to_cyclic() {
        let n = 8;
        let order = 2;
        let s_open = create_open_difference_penalty_matrix(n, order).unwrap();
        let s_circle = create_cyclic_difference_penalty_matrix(n, order).unwrap();

        let (s0, _, _) = create_closure_difference_penalty_jet(n, order, 0.0).unwrap();
        let (s1, _, _) = create_closure_difference_penalty_jet(n, order, 1.0).unwrap();
        assert!((&s0 - &s_open).iter().all(|v| v.abs() < 1e-12));
        assert!((&s1 - &s_circle).iter().all(|v| v.abs() < 1e-12));
    }

    /// The closure penalty's γ-derivative matches a finite difference of the
    /// penalty matrix.
    #[test]
    pub(crate) fn closure_penalty_gamma_derivative_matches_fd() {
        let n = 6;
        let order = 2;
        let g = 0.45;
        let (_, ds, _) = create_closure_difference_penalty_jet(n, order, g).unwrap();
        let h = 1e-6;
        let (sp, _, _) = create_closure_difference_penalty_jet(n, order, g + h).unwrap();
        let (sm, _, _) = create_closure_difference_penalty_jet(n, order, g - h).unwrap();
        let fd = (&sp - &sm).mapv(|v| v / (2.0 * h));
        assert!((&ds - &fd).iter().all(|v| v.abs() < 1e-6));
    }

    /// The open penalty leaves the constant vector and the linear ramp in its
    /// null space (order-2 difference), while the cyclic penalty only leaves the
    /// constant — the wrap piece is what penalises the boundary ramp.
    #[test]
    pub(crate) fn open_penalty_null_space_is_larger_than_cyclic() {
        let n = 7;
        let s_open = create_open_difference_penalty_matrix(n, 2).unwrap();
        let ones = ndarray::Array1::<f64>::ones(n);
        let ramp = ndarray::Array1::from_iter((0..n).map(|i| i as f64));
        // Constant is in both null spaces; ramp is in the open null space.
        let open_const = ones.dot(&s_open.dot(&ones));
        let open_ramp = ramp.dot(&s_open.dot(&ramp));
        assert!(open_const.abs() < 1e-10, "open S·1 ≠ 0: {open_const}");
        assert!(open_ramp.abs() < 1e-8, "open S·ramp ≠ 0: {open_ramp}");
    }
}
