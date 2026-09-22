//! The curvature half of the smoothing-corrected covariance (gam#3229).
//!
//! # The target
//!
//! The quantity a smoothing-corrected covariance reports is the θ-mixture variance of the
//! coefficients, and to first order in `V_θ = Var(θ̂)` that is
//!
//! ```text
//!   Var(β | y) = E_θ[V(θ)] + Var_θ(m(θ))
//!              ≈ V(θ̂) + J_m V_θ J_mᵀ + ½ Σ_jk V_θ[j,k] ∂²V/∂θ_j∂θ_k,
//! ```
//!
//! with `m(θ)` the reported posterior mean and `V(θ)` the reported conditional covariance.
//! Only the middle term was ever assembled. The third is the SAME order in `V_θ` as the
//! second, not a higher-order remainder: on the issue's one-dimensional witness the two are
//! `2.77e-5` and `−1.99e-5` at `V_θ = 0.005`, and their sum `7.8e-6` is the exact mixture's
//! `7.84e-6`, while the carried term alone is off by a factor of three and a half.
//!
//! # What this module computes, and where the identity holds
//!
//! At an INTERIOR mode the reported conditional covariance is the inverse of the penalized
//! precision, `V(ρ) = M(ρ)⁻¹` with `M(ρ) = XᵀWX + Σ_k λ_k S̃_k` and `λ_k = e^{ρ_k}`. The
//! penalties `S̃_k` carry no ρ, so along the ρ block
//!
//! ```text
//!   Ṁ_k = λ_k S̃_k = D_k,        M̈_jk = δ_jk D_k,
//! ```
//!
//! both EXACT, not truncations. Differentiating `V = M⁻¹` twice,
//!
//! ```text
//!   ∂V/∂ρ_k    = −V D_k V,
//!   ∂²V/∂ρ_j∂ρ_k = V (D_j V D_k + D_k V D_j − δ_jk D_k) V,
//! ```
//!
//! so, using the symmetry of `V_ρ` to merge the two cross terms,
//!
//! ```text
//!   ½ Σ_jk V_ρ[j,k] ∂²V/∂ρ_j∂ρ_k
//!     = V [ Σ_jk V_ρ[j,k] D_j V D_k ] V  −  ½ V [ Σ_k V_ρ[k,k] D_k ] V.
//! ```
//!
//! The first bracket is a double sum over the ρ grid, which would cost `k²` products of the
//! coefficient dimension. It is not computed that way: with any factorization
//! `V_ρ = L Lᵀ` — here the eigen-factorization, since `V_ρ` is positive SEMI-definite and a
//! Cholesky is not available on a singular one —
//!
//! ```text
//!   Σ_jk V_ρ[j,k] D_j V D_k = Σ_m E_m V E_m,     E_m = Σ_j L[j,m] D_j,
//! ```
//!
//! which is `k` products, the same order as the first-order term's own `A = V U` chain.
//! Each `E_m` is symmetric because every `D_j` is, so `E_m V E_m` is symmetric, and the whole
//! term is symmetric — but it is NOT positive semi-definite, and it must not be: the witness
//! measures it negative. That is why it is returned as a matrix and not as a factor.
//!
//! # Where the identity does NOT hold
//!
//! At a CONSTRAINED mode the reported `V(θ)` is the truncated-Gaussian covariance on the
//! cone, not `M⁻¹`, and its second θ-derivative is a third- and fourth-cumulant object of
//! the truncated law rather than the matrix identity above. A ψ coordinate's `M̈` is the
//! family's own second design derivative and is not published anywhere. This module refuses
//! both rather than pricing them with the ambient identity, because an ambient `V''` at a
//! truncated `V` would be a third convention on top of the two gam#3229 already reports.

use ndarray::{Array2, ArrayView2};

/// Why the curvature term is not available at this mode (gam#3229).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CovarianceCurvatureAbsence {
    /// The reported conditional covariance is a truncated-Gaussian covariance on the
    /// constraint cone, not `M⁻¹`, so the matrix identity this module derives does not
    /// describe its second derivative.
    TruncatedConditionalCovariance,
    /// The outer coordinates include design (ψ) axes, whose `M̈` is the family's own second
    /// design derivative and is published by no family.
    DesignAxesPresent { psi_dimension: usize },
}

impl std::fmt::Display for CovarianceCurvatureAbsence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedConditionalCovariance => write!(
                formatter,
                "the conditional covariance at a constrained mode is a truncated-Gaussian \
                 covariance, whose second derivative is a cumulant object rather than the \
                 penalized precision's inverse (gam#3229)"
            ),
            Self::DesignAxesPresent { psi_dimension } => write!(
                formatter,
                "{psi_dimension} design (psi) outer axis/axes carry a second design derivative \
                 no family publishes, so the covariance curvature is not defined over them \
                 (gam#3229)"
            ),
        }
    }
}

/// `½ Σ_jk V_ρ[j,k] ∂²V/∂ρ_j∂ρ_k` for `V(ρ) = M(ρ)⁻¹`, from the precision's own ρ-drifts.
///
/// `v` is `V(ρ̂)`, `drifts[k]` is `D_k = λ_k S̃_k` in the SAME coefficient frame as `v`, and
/// `rho_covariance` is `V_ρ` over the same coordinates, in the same order. The module doc
/// carries the derivation; the identity it rests on is that `S̃_k` is ρ-free, so
/// `M̈_jk = δ_jk D_k` exactly.
///
/// The result is symmetric and generally INDEFINITE. It is added to a covariance as a sum,
/// where the negative-diagonal judgement belongs to `gam_problem::se_from_covariance`.
pub fn laplace_covariance_curvature_term(
    v: ArrayView2<'_, f64>,
    drifts: &[Array2<f64>],
    rho_covariance: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let p = v.nrows();
    if v.ncols() != p {
        return Err(format!(
            "covariance curvature: V is {}x{}, not square",
            v.nrows(),
            v.ncols()
        ));
    }
    let k = drifts.len();
    if rho_covariance.dim() != (k, k) {
        return Err(format!(
            "covariance curvature: V_rho is {:?} for {k} drift(s)",
            rho_covariance.dim()
        ));
    }
    for (index, drift) in drifts.iter().enumerate() {
        if drift.dim() != (p, p) {
            return Err(format!(
                "covariance curvature: drift {index} is {:?}, not {p}x{p}",
                drift.dim()
            ));
        }
    }
    if k == 0 {
        return Ok(Array2::<f64>::zeros((p, p)));
    }

    // `V_rho = L Lᵀ` by its own eigen-factorization. It is positive SEMI-definite — the
    // identified-subspace inverse returns exact zeros on the directions it dropped — so a
    // Cholesky is not available and an eigenvalue at or below zero is a direction that
    // carries no rho variance and contributes nothing.
    let mut symmetric = rho_covariance.to_owned();
    gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
    let (eigenvalues, eigenvectors) = {
        use gam_linalg::faer_ndarray::FaerEigh;
        symmetric
            .eigh(faer::Side::Lower)
            .map_err(|error| format!("covariance curvature: V_rho eigendecomposition: {error}"))?
    };

    let mut cross = Array2::<f64>::zeros((p, p));
    for (column, &eigenvalue) in eigenvalues.iter().enumerate() {
        if !(eigenvalue > 0.0) {
            continue;
        }
        let weight = eigenvalue.sqrt();
        // `E_m = Σ_j L[j,m] D_j`, symmetric because every `D_j` is.
        let mut e_m = Array2::<f64>::zeros((p, p));
        for (j, drift) in drifts.iter().enumerate() {
            let coefficient = weight * eigenvectors[[j, column]];
            if coefficient == 0.0 {
                continue;
            }
            e_m.scaled_add(coefficient, drift);
        }
        // `E_m V E_m`, in that order: the middle factor is the covariance, not a repeated
        // drift.
        let right = v.dot(&e_m);
        cross += &e_m.dot(&right);
    }

    // `−½ Σ_k V_rho[k,k] D_k`, the `M̈` half.
    let mut second = Array2::<f64>::zeros((p, p));
    for (k_index, drift) in drifts.iter().enumerate() {
        let weight = rho_covariance[[k_index, k_index]];
        if weight == 0.0 {
            continue;
        }
        second.scaled_add(-0.5 * weight, drift);
    }
    cross += &second;

    let mut term = v.dot(&cross).dot(&v);
    gam_linalg::matrix::symmetrize_in_place(&mut term);
    if term.iter().any(|value| !value.is_finite()) {
        return Err("covariance curvature: the assembled term is not finite".to_string());
    }
    Ok(term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// One coordinate, where the whole object has a closed form that is not this module's:
    /// `V(ρ) = 1/(1 + e^ρ)`, so `V'' = −e^ρ(1+e^ρ)⁻² + 2e^{2ρ}(1+e^ρ)⁻³`. At `ρ = ln 2` that
    /// is `−2/9 + 8/27 = 2/27`, and the term is `½·V_ρ·2/27`.
    ///
    /// The comparison is against a value written from the scalar derivative of the inverse,
    /// which shares no line of code with the matrix assembly. Both sides are sums and
    /// quotients of exactly representable rationals through `exp`, so the band is the counted
    /// roundings: five in the assembly (one `sqrt`, two products, one scaled add, one
    /// symmetrize) and three in the reference, at half an ulp of the result's own magnitude.
    #[test]
    fn one_coordinate_matches_the_scalar_second_derivative_of_the_inverse() {
        let lambda = 2.0_f64;
        let m = 1.0 + lambda;
        let v = array![[1.0 / m]];
        let drifts = vec![array![[lambda]]];
        let rho_variance = 0.005_f64;
        let term = laplace_covariance_curvature_term(
            v.view(),
            &drifts,
            array![[rho_variance]].view(),
        )
        .expect("one-coordinate curvature term");

        let second_derivative = -lambda / (m * m) + 2.0 * lambda * lambda / (m * m * m);
        let reference = 0.5 * rho_variance * second_derivative;
        let band = 8.0 * f64::EPSILON * reference.abs().max(f64::MIN_POSITIVE);
        assert!(
            (term[[0, 0]] - reference).abs() <= band,
            "curvature term {:e} against {reference:e}, off by more than {band:e}",
            term[[0, 0]]
        );
        // The witness's sign: the omitted term is NEGATIVE where the carried one is positive.
        assert!(second_derivative > 0.0, "this fixture's V'' is positive");
    }

    /// Two coordinates, graded against a symmetric second difference of the literal inverse
    /// along each eigen-direction of `V_ρ`, which is what `tr(V'' V_ρ)` decomposes into and
    /// which shares no line with the assembly.
    ///
    /// The step is derived, not chosen: a symmetric second difference of a smooth `f` carries
    /// truncation `h²|f⁗|/12` and roundoff `4ε|f|/h²`, which are equal at
    /// `h = (48 ε |f| / |f⁗|)^{1/4}`, and with `|f⁗|` and `|f|` of the same order on this
    /// fixture that is `h = (48 ε)^{1/4}`.
    ///
    /// The BAND is the difference's own two error terms, and both are read here rather than
    /// assumed. Writing `D(h)` for the second difference at step `h`, `D(h) = f'' + c·h² +
    /// O(h⁴)`, so `(D(2h) − D(h))/3` IS `c·h²`, the truncation at `h`, measured. The roundoff
    /// is counted: three evaluations enter each difference (`f(+h)`, `f(−h)` and twice
    /// `f(0)`), each to within half an ulp of `|f|`, and the sum is divided by `h²`, so it is
    /// `4ε·max|f|/h²` per direction. Denominating the band in `|f''|` instead — the size of
    /// the answer — is what made this assertion fail at a correct value: the roundoff is
    /// carried by `|f|`, which on this fixture is about four hundred times larger, so a band
    /// proportional to the answer is thirty times too tight.
    #[test]
    fn two_coordinates_match_a_second_difference_of_the_inverse() {
        let base = array![[3.0_f64, 0.4], [0.4, 2.0]];
        let s1 = array![[1.0_f64, 0.2], [0.2, 0.5]];
        let s2 = array![[0.3_f64, -0.1], [-0.1, 1.4]];
        let rho = [0.3_f64, -0.2];
        let precision = |offset: [f64; 2]| -> Array2<f64> {
            let mut m = base.clone();
            m.scaled_add((rho[0] + offset[0]).exp(), &s1);
            m.scaled_add((rho[1] + offset[1]).exp(), &s2);
            m
        };
        let inverse = |offset: [f64; 2]| -> Array2<f64> {
            let m = precision(offset);
            let determinant = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
            array![
                [m[[1, 1]] / determinant, -m[[0, 1]] / determinant],
                [-m[[1, 0]] / determinant, m[[0, 0]] / determinant]
            ]
        };

        let v = inverse([0.0, 0.0]);
        let drifts = vec![
            s1.mapv(|value| value * rho[0].exp()),
            s2.mapv(|value| value * rho[1].exp()),
        ];
        let rho_covariance = array![[0.05_f64, 0.01], [0.01, 0.03]];
        let term = laplace_covariance_curvature_term(v.view(), &drifts, rho_covariance.view())
            .expect("two-coordinate curvature term");

        // `½ tr(V'' V_ρ) = ½ Σ_m d²V/dt²` along `d_m = √μ_m q_m`, the eigen-factorization's
        // own columns: exactly the decomposition the assembly uses, evaluated on the literal
        // inverse instead.
        let mut symmetric = rho_covariance.clone();
        gam_linalg::matrix::symmetrize_in_place(&mut symmetric);
        let (eigenvalues, eigenvectors) = {
            use gam_linalg::faer_ndarray::FaerEigh;
            symmetric.eigh(faer::Side::Lower).expect("V_rho eigh")
        };
        let step = (48.0 * f64::EPSILON).powf(0.25);
        // `½ Σ_m D_m(h)` at `h` and at `2h`, and the largest magnitude the differenced
        // function reaches over every point either one evaluated.
        let difference_at = |h: f64| -> (Array2<f64>, f64) {
            let mut total = Array2::<f64>::zeros((2, 2));
            let mut scale_of_f = v.iter().fold(0.0_f64, |worst, value| worst.max(value.abs()));
            for (column, &eigenvalue) in eigenvalues.iter().enumerate() {
                if !(eigenvalue > 0.0) {
                    continue;
                }
                let scale = eigenvalue.sqrt();
                let direction = [
                    scale * eigenvectors[[0, column]],
                    scale * eigenvectors[[1, column]],
                ];
                let forward = inverse([h * direction[0], h * direction[1]]);
                let backward = inverse([-h * direction[0], -h * direction[1]]);
                for value in forward.iter().chain(backward.iter()) {
                    scale_of_f = scale_of_f.max(value.abs());
                }
                let second = (&forward + &backward - &v.mapv(|value| 2.0 * value))
                    .mapv(|value| value / (h * h));
                total.scaled_add(0.5, &second);
            }
            (total, scale_of_f)
        };
        let (reference, scale_of_f) = difference_at(step);
        let (coarse, _) = difference_at(2.0 * step);

        // Four roundings of `|f|` per direction, divided by `h²`, over the two directions the
        // sum runs; each direction's `½` is already in the difference above.
        let directions = eigenvalues.iter().filter(|&&value| value > 0.0).count() as f64;
        let roundoff = directions * 0.5 * 4.0 * f64::EPSILON * scale_of_f / (step * step);
        for (index, ((assembled, expected), coarse_value)) in term
            .iter()
            .zip(reference.iter())
            .zip(coarse.iter())
            .enumerate()
        {
            let truncation = (expected - coarse_value).abs() / 3.0;
            let band = truncation + roundoff;
            assert!(
                (assembled - expected).abs() <= band,
                "entry {index}: assembled {assembled:e} against second difference {expected:e}, \
                 off by more than its measured truncation {truncation:e} plus its counted \
                 roundoff {roundoff:e}"
            );
        }
    }

    /// No coordinate is no term, exactly: a fit with no smoothing parameter has no rho
    /// variance to propagate and the curvature is the zero matrix, not an absence.
    #[test]
    fn no_coordinate_is_an_exact_zero() {
        let v = array![[0.5_f64, 0.1], [0.1, 0.25]];
        let term = laplace_covariance_curvature_term(
            v.view(),
            &[],
            Array2::<f64>::zeros((0, 0)).view(),
        )
        .expect("empty curvature term");
        assert_eq!(term, Array2::<f64>::zeros((2, 2)));
    }

    /// A direction the identified-subspace inverse dropped carries exactly zero rho variance
    /// and contributes exactly nothing, so a singular `V_rho` is an ordinary input.
    #[test]
    fn a_dropped_rho_direction_contributes_exactly_zero() {
        let v = array![[0.4_f64, 0.05], [0.05, 0.3]];
        let drifts = vec![
            array![[1.0_f64, 0.0], [0.0, 0.2]],
            array![[0.1_f64, 0.0], [0.0, 0.9]],
        ];
        let full = array![[0.02_f64, 0.0], [0.0, 0.0]];
        let only_first = laplace_covariance_curvature_term(v.view(), &drifts, full.view())
            .expect("rank-one curvature term");
        let single = laplace_covariance_curvature_term(
            v.view(),
            std::slice::from_ref(&drifts[0]),
            array![[0.02_f64]].view(),
        )
        .expect("one-drift curvature term");
        for (index, (both, one)) in only_first.iter().zip(single.iter()).enumerate() {
            assert!(
                (both - one).abs() <= 8.0 * f64::EPSILON * one.abs().max(f64::MIN_POSITIVE),
                "entry {index}: {both:e} against {one:e}"
            );
        }
    }
}
