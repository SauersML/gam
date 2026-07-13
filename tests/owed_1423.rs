//! Owed-work regression for #1423 — the mixed-periodicity (cylinder / torus)
//! Duchon null space must be the polynomials of degree `< m` in the NON-periodic
//! coordinates only (constants on the periodic axes).
//!
//! ## The defect (now fixed)
//!
//! The chord-polyharmonic mixed-periodicity kernel forced a constants-only
//! constraint (`Z = null([1])`), so the unpenalised directions were just the
//! global constant — the genuinely-unpenalised low-order POLYNOMIAL modes in the
//! non-periodic coordinate (e.g. the linear trend `y` on a cylinder with `m = 2`)
//! were NOT in the null space. A linear trend along the non-periodic axis was
//! therefore wrongly penalised, biasing the fit toward flat.
//!
//! The additive (ANOVA) reproducing-kernel fix builds the null space from the
//! polynomials of degree `< m` in the non-periodic coordinates (`{1, y}` for the
//! `m = 2` cylinder), appends those exact monomial columns to the design as
//! explicit UNPENALISED columns, and forms the penalty `Ω = Zᵀ K_CC Z` only on
//! the kernel columns — so the polynomial columns carry an exactly-zero penalty
//! block.
//!
//! ## What this guards
//!
//! With identifiability left unconstrained (`SpatialIdentifiability::None`, so no
//! transform mixes the blocks), the realized mixed-periodicity Duchon design ends
//! in exactly `n_poly` polynomial columns whose penalty rows and columns are
//! exactly zero, where `n_poly` is the number of monomials of total degree `< m`
//! over the non-periodic axes (2 = `{1, y}` for the `m = 2` cylinder; 3 =
//! `{1, y, y²}` for an `m = 3` cylinder). Equivalently the penalty nullity is at
//! least `n_poly`: the unpenalised polynomial modes the fix restores.
//!
//! Reference-as-truth: the null-space structure is an intrinsic property of the
//! product-RKHS Duchon penalty, asserted on gam's own realized design / penalty —
//! never against another tool's output.

use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, PenaltySource, SpatialIdentifiability,
    build_duchon_basis,
};
use ndarray::Array2;

fn cylinder_data() -> Array2<f64> {
    // Axis 0 periodic (period 1), axis 1 the non-periodic coordinate `y`.
    Array2::from_shape_vec(
        (9, 2),
        vec![
            0.05, -0.6, 0.20, 0.1, 0.37, 0.9, 0.51, -0.3, 0.66, 0.5, 0.78, -0.1, 0.90, 0.8, 0.97,
            -0.45, 0.12, 0.33,
        ],
    )
    .unwrap()
}

/// Number of monomials of total degree `< m` over `n_axes` non-periodic axes.
/// (Degree `0..=m-1`, multiset coefficient.) For 1 non-periodic axis this is
/// simply `m`.
fn nonperiodic_poly_count(n_axes: usize, m: usize) -> usize {
    // Sum over degree d in 0..m of C(d + n_axes - 1, n_axes - 1).
    let mut total = 0usize;
    for d in 0..m {
        // multiset C(d + n_axes - 1, n_axes - 1)
        let top = d + n_axes - 1;
        let k = n_axes - 1;
        total += binom(top, k);
    }
    total
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num = 1usize;
    let mut den = 1usize;
    for i in 0..k {
        num *= n - i;
        den *= i + 1;
    }
    num / den
}

fn cylinder_spec(data: &Array2<f64>, nullspace_order: DuchonNullspaceOrder) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: Some(vec![Some(1.0), None]),
        length_scale: None,
        power: 0.0,
        nullspace_order,
        // Unconstrained so the penalty is exactly Ω padded with zero polynomial
        // columns (no identifiability transform mixing the two blocks).
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: Default::default(),
    }
}

/// MERGE GATE (#1423): the mixed-periodicity Duchon design ends in exactly
/// `n_poly` explicit polynomial columns whose penalty block is exactly zero
/// (the unpenalised non-periodic polynomials of degree `< m`), for both the
/// `m = 2` cylinder (`{1, y}`) and the `m = 3` cylinder (`{1, y, y²}`).
#[test]
fn mixed_periodicity_nonperiodic_polynomials_are_unpenalised_1423() {
    // (nullspace_order, m): Linear → m = 2, Degree(2) → m = 3.
    for (order, m) in [
        (DuchonNullspaceOrder::Linear, 2usize),
        (DuchonNullspaceOrder::Degree(2), 3usize),
    ] {
        let data = cylinder_data();
        let spec = cylinder_spec(&data, order);
        let built = build_duchon_basis(data.view(), &spec)
            .unwrap_or_else(|e| panic!("m={m} cylinder mixed-periodicity build failed: {e:?}"));
        let primary = built
            .active_penalties
            .iter()
            .find(|penalty| matches!(&penalty.info.source, PenaltySource::Primary))
            .unwrap_or_else(|| panic!("m={m}: cylinder Duchon build emitted no primary penalty"));
        let s = &primary.matrix;
        let p = s.nrows();
        let n_poly = nonperiodic_poly_count(1, m); // 1 non-periodic axis
        assert_eq!(n_poly, m, "1-axis poly count must equal m");
        assert!(
            p > n_poly,
            "m={m}: penalty must have kernel columns beyond the {n_poly} polynomial columns; \
             got p={p}"
        );

        // The trailing `n_poly` columns are the explicit unpenalised polynomials
        // (non-periodic coords, degree < m). Their penalty rows AND columns must
        // be exactly zero (modulo rounding).
        let scale = s.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
        for col in (p - n_poly)..p {
            for row in 0..p {
                assert!(
                    s[[row, col]].abs() < 1e-12 * scale,
                    "m={m}: polynomial column {col} must be UNPENALISED (#1423); \
                     S[{row},{col}] = {:.3e} (scale {scale:.3e})",
                    s[[row, col]]
                );
                assert!(
                    s[[col, row]].abs() < 1e-12 * scale,
                    "m={m}: polynomial row {col} must be UNPENALISED (#1423); \
                     S[{col},{row}] = {:.3e} (scale {scale:.3e})",
                    s[[col, row]]
                );
            }
        }

        // The kernel block must itself be non-trivial (a real roughness penalty),
        // so the unpenalised polynomial null space is a genuine SUBSPACE, not the
        // whole penalty collapsing to zero.
        let kernel_block_scale = (0..(p - n_poly))
            .flat_map(|i| (0..(p - n_poly)).map(move |j| (i, j)))
            .map(|(i, j)| s[[i, j]].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            kernel_block_scale > 1e-8,
            "m={m}: the kernel-column penalty block must be a non-trivial roughness penalty; \
             got max |S_kernel| = {kernel_block_scale:.3e}"
        );
    }
}
