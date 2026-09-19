//! The boundary-mode posterior approximation (#979, ruling (c)).
//!
//! At a converged constrained mode `β̂` the quadratic Laplace model can be IMPROPER on
//! the feasible cone (the #979 600×10 survival fit: `In(H) = (32, 0, 1)`, copositive
//! minimum `−5.33`). There the cone-truncated Gaussian has no moments, but the local
//! posterior still does. Its constraint-normal coordinates are held by the KKT
//! multipliers, not by the curvature.
//!
//! Write the active rows (unit-scaled) as `A`, `Z` an orthonormal basis of `null(A)`,
//! `N = Aᵀ(AAᵀ)⁻¹`, and `δ = β − β̂ = Zt + Nw` with `w = Aδ ≥ 0`. With
//! `∇F(β̂) = Aᵀμ` and `M = ∇²F(β̂)` the second-order model is
//!
//! ```text
//! F(β̂ + δ) − F(β̂) ≈ μᵀw + ½ δᵀMδ,
//! ```
//!
//! and integrating `t` out exactly leaves `μᵀw + ½wᵀSw` on `w ≥ 0`, with `G = ZᵀMZ`,
//! `C = ZᵀMN` and `S = NᵀMN − CᵀG⁻¹C`. The approximation keeps the exact conditional
//! law of `t` given `w`, `N(−G⁻¹Cw, G⁻¹)`, and drops `½wᵀSw`. The constraint-normal
//! coordinates are then independent exponentials at the multipliers, so
//!
//! ```text
//! δ = P w + Z ξ,   w_i ~ Exp(μ_i),   ξ ~ N(0, G⁻¹),   P = N − Z G⁻¹ C,
//! E[β] = β̂ + P μ⁻¹,   Cov[β] = Z G⁻¹ Zᵀ + P diag(μ⁻²) Pᵀ.
//! ```
//!
//! It is published only under a certificate, and every entry of the certificate is held
//! to the accuracy the module reports its moments to
//! ([`super::ORTHANT_MOMENT_RELATIVE_TOLERANCE`]). With `e = diag(μ)·w`, `e_i ~ Exp(1)`
//! independently and the dropped term is `½eᵀKe`, `K = diag(μ)⁻¹ S diag(μ)⁻¹`.
//!
//! * **Overturn tail.** The dropped curvature overturns the multiplier term only where
//!   `−½eᵀKe ≥ 1ᵀe`. Every `e ≥ 0` is `(1ᵀe)·v` with `v` on the simplex, so that set lies
//!   inside `{1ᵀe ≥ 2/c₋}`, `c₋ = −min_v vᵀKv`. `1ᵀe ~ Gamma(q, 1)`, so its mass there
//!   is exact: `e^{−x} Σ_{k<q} x^k/k!` at `x = 2/c₋`, and zero when `K` is copositive.
//! * **First-order moment bound.** A published moment `E[φ]` moves at first order by
//!   `−½ Cov(φ, eᵀKe)`, which Cauchy–Schwarz bounds by `½·sd(φ)·‖K‖_F·√E‖e‖⁴`.
//!   `E‖e‖⁴ = 4q² + 20q`, and `sd(φ)/E[φ]` is at most `√5` over the first and second
//!   moments (it is `√5` for `e_i²`).
//! * **Tangent shift.** The multiplier fit leaves a residual `r = ∇F − Aᵀμ` the model
//!   does not carry. It moves the face mean by `Z G⁻¹ Zᵀ r`, measured here in posterior
//!   standard deviations.
//! * **Inactive walls.** The approximation models no inactive row, so each one must lie
//!   beyond the law's reach at double precision. That is the horizon
//!   `constraint_face_candidates` reads off `f64::EPSILON`, union-bounded over the
//!   Gaussian part and each wall-ward exponential part.
//!
//! Nothing here bounds the likelihood's third-order terms. The proper-cone Laplace law
//! ignores those too, so the approximation is no weaker in that respect.

use gam_linalg::utils::certified_spd_factorize;
use gam_math::probability::standard_normal_quantile;
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use super::ORTHANT_MOMENT_RELATIVE_TOLERANCE;

/// The measured evidence a boundary-mode approximation is published under.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryModeCertificate {
    /// `μ`: the nonnegative multipliers of the active rows at unit row scale, which are
    /// the exponential rates of their constraint-normal coordinates.
    pub rates: Array1<f64>,
    /// `min_v vᵀKv` over the simplex, with `K = diag(μ)⁻¹ S diag(μ)⁻¹`.
    pub scaled_copositive_minimum: f64,
    /// Exponential-law mass where the dropped curvature overturns the multiplier term.
    pub overturn_tail_mass: f64,
    /// Bound on the leading relative change of every published first and second moment.
    pub first_order_moment_bound: f64,
    /// Largest face-mean shift the multiplier fit's residual implies, in posterior
    /// standard deviations.
    pub tangent_shift: f64,
    /// The accuracy each entry above is held to.
    pub tolerance: f64,
}

impl BoundaryModeCertificate {
    fn measured(&self) -> [(&'static str, f64); 3] {
        [
            ("overturn tail mass", self.overturn_tail_mass),
            ("first-order moment bound", self.first_order_moment_bound),
            ("tangent shift", self.tangent_shift),
        ]
    }
}

/// The certified boundary-mode law at a constrained mode whose quadratic Laplace model is
/// improper on the cone (see the module documentation).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryModeApproximation {
    /// Constraint rows the mode binds, in `certificate.rates` order.
    pub active_rows: Vec<usize>,
    /// `E[β] = β̂ + P μ⁻¹`.
    pub mean: Array1<f64>,
    /// `P = N − Z G⁻¹ C`, `p × q`.
    pub lift: Array2<f64>,
    /// `Z G⁻¹ Zᵀ`, `p × p`: the Gaussian part of the law.
    pub face_covariance: Array2<f64>,
    pub certificate: BoundaryModeCertificate,
}

/// Why no boundary-mode approximation was published at a converged constrained mode.
///
/// The caller keeps the mode under its moment decline either way, and the decline records
/// this refusal, so a saved model says why its moments are unavailable at the boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryModeRefusal {
    pub reason: String,
    /// The measured certificate, once the approximation got as far as measuring it. A
    /// refusal by the certificate names its failing entry, such as the overturn tail mass,
    /// and keeps every measured value.
    #[serde(default)]
    pub certificate: Option<BoundaryModeCertificate>,
}

impl BoundaryModeRefusal {
    fn measured(reason: String, certificate: &BoundaryModeCertificate) -> Self {
        Self {
            reason,
            certificate: Some(certificate.clone()),
        }
    }
}

impl From<String> for BoundaryModeRefusal {
    fn from(reason: String) -> Self {
        Self {
            reason,
            certificate: None,
        }
    }
}

impl std::fmt::Display for BoundaryModeRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.reason)
    }
}

impl BoundaryModeApproximation {
    /// The boundary-mode law at `mode`, or why it may not be published.
    ///
    /// `precision` is `∇²F(β̂)` and `penalized_gradient` is `∇F(β̂)`, for `F` the negative
    /// log posterior the fit minimized. Every refusal names its reason, and a refusal made
    /// once the certificate is measured also carries the certificate.
    pub fn at_converged_mode(
        precision: ArrayView2<'_, f64>,
        constraints: &LinearInequalityConstraints,
        mode: &Array1<f64>,
        penalized_gradient: &Array1<f64>,
    ) -> Result<Self, BoundaryModeRefusal> {
        let p = precision.nrows();
        if precision.ncols() != p
            || constraints.a.ncols() != p
            || constraints.a.nrows() != constraints.b.len()
            || mode.len() != p
            || penalized_gradient.len() != p
        {
            return Err(format!(
                "boundary-mode approximation: shapes disagree (precision {:?}, constraints \
                 {}x{} with {} bounds, mode {}, gradient {})",
                precision.dim(),
                constraints.a.nrows(),
                constraints.a.ncols(),
                constraints.b.len(),
                mode.len(),
                penalized_gradient.len()
            )
            .into());
        }
        let face = crate::active_set::active_face(mode, constraints).ok_or_else(|| {
            "boundary-mode approximation: the mode does not match the constraint width".to_string()
        })?;
        let q = face.active_idx.len();
        if q == 0 {
            return Err(
                "boundary-mode approximation: the mode binds no constraint row, so no \
                 coordinate is held by a multiplier"
                    .to_string()
                    .into(),
            );
        }
        let rows = face.a_active.clone();

        let gram = rows.dot(&rows.t());
        let gram_factor = certified_spd_factorize(&gram, "boundary-mode active-row Gram AAᵀ")
            .map_err(|error| {
                format!(
                    "boundary-mode approximation: the {q} active row(s) are not independent: \
                     {error}"
                )
            })?;
        let (gram_solved_rows, _) = gram_factor.solve_matrix(&rows).map_err(|error| {
            format!("boundary-mode approximation: the active-row Gram solve failed: {error}")
        })?;
        let right_inverse = gram_solved_rows.t().to_owned();
        let (rank, tangent) = crate::active_set::null_space_of_rows(&rows).ok_or_else(|| {
            "boundary-mode approximation: the active rows' null space could not be computed"
                .to_string()
        })?;
        if rank != q {
            return Err(format!(
                "boundary-mode approximation: {q} active row(s) have numerical rank {rank}"
            )
            .into());
        }

        let precision = precision.to_owned();
        let precision_times_right_inverse = precision.dot(&right_inverse);
        let tangent_dimension = tangent.ncols();
        let (lift, mut schur, face_covariance) = if tangent_dimension == 0 {
            (
                right_inverse.clone(),
                right_inverse.t().dot(&precision_times_right_inverse),
                Array2::<f64>::zeros((p, p)),
            )
        } else {
            let mut face_precision = tangent.t().dot(&precision).dot(&tangent);
            gam_linalg::matrix::symmetrize_in_place(&mut face_precision);
            let face_factor =
                certified_spd_factorize(&face_precision, "boundary-mode face precision ZᵀMZ")
                    .map_err(|error| {
                        format!(
                            "boundary-mode approximation: the precision is not positive \
                             definite on the {tangent_dimension}-dimensional face tangent: \
                             {error}"
                        )
                    })?;
            let coupling = tangent.t().dot(&precision_times_right_inverse);
            let (face_solved_coupling, _) =
                face_factor.solve_matrix(&coupling).map_err(|error| {
                    format!("boundary-mode approximation: the face coupling solve failed: {error}")
                })?;
            let face_inverse = face_factor
                .inverse()
                .map_err(|error| {
                    format!("boundary-mode approximation: the face inverse failed: {error}")
                })?
                .into_inverse();
            (
                &right_inverse - &tangent.dot(&face_solved_coupling),
                right_inverse.t().dot(&precision_times_right_inverse)
                    - coupling.t().dot(&face_solved_coupling),
                tangent.dot(&face_inverse).dot(&tangent.t()),
            )
        };
        gam_linalg::matrix::symmetrize_in_place(&mut schur);

        let (rates, residual) = crate::active_set::nonnegative_cone_multipliers(
            &rows,
            penalized_gradient,
        )
        .ok_or_else(|| {
            "boundary-mode approximation: the nonnegative multiplier fit refused".to_string()
        })?;
        if let Some(position) = (0..q).find(|&position| !(rates[position] > 0.0)) {
            return Err(format!(
                "boundary-mode approximation: active constraint row {} carries multiplier \
                 {:.3e}; without strict complementarity its coordinate has no exponential scale",
                face.active_idx[position], rates[position]
            )
            .into());
        }

        let mut rate_scaled_lift = lift.clone();
        for (mut column, &rate) in rate_scaled_lift.columns_mut().into_iter().zip(rates.iter()) {
            column.mapv_inplace(|value| value / rate);
        }
        let covariance = &face_covariance + &rate_scaled_lift.dot(&rate_scaled_lift.t());
        let face_shift = face_covariance.dot(&residual);
        let tangent_shift = (0..p)
            .map(|coordinate| {
                let spread = covariance[[coordinate, coordinate]];
                if spread > 0.0 {
                    face_shift[coordinate].abs() / spread.sqrt()
                } else if face_shift[coordinate] == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0_f64, f64::max);

        let mut scaled_schur = schur;
        for i in 0..q {
            for j in 0..q {
                scaled_schur[[i, j]] /= rates[i] * rates[j];
            }
        }
        let (scaled_copositive_minimum, _) =
            crate::cone_reduction::copositive_simplex_minimum(scaled_schur.view()).map_err(
                |error| format!("boundary-mode approximation: scaled normal curvature: {error}"),
            )?;
        let overturn_tail_mass = if scaled_copositive_minimum < 0.0 {
            gamma_upper_tail(q, 2.0 / -scaled_copositive_minimum)
        } else {
            0.0
        };
        let frobenius = scaled_schur.iter().map(|value| value * value).sum::<f64>().sqrt();
        let face_width = q as f64;
        let first_order_moment_bound =
            0.5 * 5.0_f64.sqrt() * frobenius * (4.0 * face_width * face_width + 20.0 * face_width).sqrt();

        let certificate = BoundaryModeCertificate {
            rates,
            scaled_copositive_minimum,
            overturn_tail_mass,
            first_order_moment_bound,
            tangent_shift,
            tolerance: ORTHANT_MOMENT_RELATIVE_TOLERANCE,
        };
        if let Some((name, value)) = certificate
            .measured()
            .into_iter()
            .find(|&(_, value)| !(value <= certificate.tolerance))
        {
            return Err(BoundaryModeRefusal::measured(
                format!(
                    "boundary-mode approximation not certified: {name} {value:.3e} exceeds the \
                     moment accuracy {:.1e} (multipliers {}, min vᵀKv over the simplex {:.3e})",
                    certificate.tolerance,
                    render_rates(&certificate.rates),
                    certificate.scaled_copositive_minimum
                ),
                &certificate,
            ));
        }

        for row in 0..constraints.a.nrows() {
            if face.active_idx.contains(&row) {
                continue;
            }
            let normal = constraints.a.row(row);
            let norm = normal.dot(&normal).sqrt();
            if !(norm > 0.0) {
                continue;
            }
            let unit = normal.mapv(|value| value / norm);
            let slack = unit.dot(mode) - constraints.b[row] / norm;
            let gaussian_spread = unit.dot(&face_covariance.dot(&unit)).max(0.0).sqrt();
            let reach = unit.dot(&lift);
            let wallward: Vec<usize> = (0..q).filter(|&position| reach[position] < 0.0).collect();
            let share = f64::EPSILON / (wallward.len() as f64 + 1.0);
            let gaussian_horizon = -standard_normal_quantile(share).map_err(|error| {
                format!("boundary-mode approximation: inactive-wall horizon: {error}")
            })?;
            let horizon = gaussian_spread * gaussian_horizon
                + wallward
                    .iter()
                    .map(|&position| -reach[position] / certificate.rates[position])
                    .sum::<f64>()
                    * (1.0 / share).ln();
            if !(slack > horizon) {
                return Err(BoundaryModeRefusal::measured(
                    format!(
                        "boundary-mode approximation: inactive constraint row {row} sits \
                         {slack:.3e} from the mode, inside the approximate law's double-precision \
                         reach {horizon:.3e}, and the approximation does not model that wall"
                    ),
                    &certificate,
                ));
            }
        }

        let mean = mode + &lift.dot(&certificate.rates.mapv(|rate| 1.0 / rate));
        Ok(Self {
            active_rows: face.active_idx,
            mean,
            lift,
            face_covariance,
            certificate,
        })
    }

    /// `Cov[β] = Z G⁻¹ Zᵀ + P diag(μ⁻²) Pᵀ`.
    pub fn covariance(&self) -> Array2<f64> {
        let mut rate_scaled_lift = self.lift.clone();
        for (mut column, &rate) in rate_scaled_lift
            .columns_mut()
            .into_iter()
            .zip(self.certificate.rates.iter())
        {
            column.mapv_inplace(|value| value / rate);
        }
        &self.face_covariance + &rate_scaled_lift.dot(&rate_scaled_lift.t())
    }

    pub fn summary(&self) -> String {
        format!(
            "boundary-mode approximation on constraint row(s) {:?}: exponential rates {}, min \
             vᵀKv over the simplex {:.3e}, overturn tail mass {:.3e}, first-order moment bound \
             {:.3e}, tangent shift {:.3e}, each within {:.1e}",
            self.active_rows,
            render_rates(&self.certificate.rates),
            self.certificate.scaled_copositive_minimum,
            self.certificate.overturn_tail_mass,
            self.certificate.first_order_moment_bound,
            self.certificate.tangent_shift,
            self.certificate.tolerance
        )
    }

    pub(super) fn validate(&self, dimension: usize, constraint_count: usize) -> Result<(), String> {
        let q = self.active_rows.len();
        if self.mean.len() != dimension
            || self.lift.dim() != (dimension, q)
            || self.face_covariance.dim() != (dimension, dimension)
            || self.certificate.rates.len() != q
        {
            return Err(format!(
                "boundary-mode approximation shapes disagree with p={dimension}, q={q}: mean {}, \
                 lift {:?}, face covariance {:?}, rates {}",
                self.mean.len(),
                self.lift.dim(),
                self.face_covariance.dim(),
                self.certificate.rates.len()
            ));
        }
        let mut unique_rows = self.active_rows.clone();
        unique_rows.sort_unstable();
        unique_rows.dedup();
        if q == 0 || unique_rows.len() != q || unique_rows.iter().any(|&row| row >= constraint_count) {
            return Err(format!(
                "boundary-mode approximation names active rows {:?} that are not unique valid \
                 indices for {constraint_count} inequalities",
                self.active_rows
            ));
        }
        if self
            .mean
            .iter()
            .chain(self.lift.iter())
            .chain(self.face_covariance.iter())
            .chain(self.certificate.rates.iter())
            .any(|value| !value.is_finite())
            || !self.certificate.scaled_copositive_minimum.is_finite()
        {
            return Err("boundary-mode approximation contains a non-finite value".to_string());
        }
        if self.certificate.rates.iter().any(|&rate| !(rate > 0.0)) {
            return Err("boundary-mode approximation has a non-positive exponential rate".to_string());
        }
        if let Some((name, value)) = self
            .certificate
            .measured()
            .into_iter()
            .find(|&(_, value)| !(value <= self.certificate.tolerance))
        {
            return Err(format!(
                "boundary-mode approximation carries {name} {value:.3e} above its tolerance {:.1e}",
                self.certificate.tolerance
            ));
        }
        Ok(())
    }
}

/// `P(Gamma(shape, 1) ≥ x) = e^{−x} Σ_{k<shape} x^k / k!`, summed in the log domain so a
/// large `x` underflows to zero instead of overflowing its terms.
fn gamma_upper_tail(shape: usize, x: f64) -> f64 {
    let log_x = x.ln();
    let mut log_term = 0.0_f64;
    let mut largest = 0.0_f64;
    let mut log_terms = Vec::with_capacity(shape);
    log_terms.push(0.0_f64);
    for k in 1..shape {
        log_term += log_x - (k as f64).ln();
        largest = largest.max(log_term);
        log_terms.push(log_term);
    }
    let log_sum = largest
        + log_terms
            .iter()
            .map(|value| (value - largest).exp())
            .sum::<f64>()
            .ln();
    (log_sum - x).exp()
}

fn render_rates(rates: &Array1<f64>) -> String {
    let rendered: Vec<String> = rates.iter().map(|rate| format!("{rate:.3e}")).collect();
    format!("[{}]", rendered.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn lower_bound_on_the_first_coordinate() -> LinearInequalityConstraints {
        LinearInequalityConstraints {
            a: array![[1.0, 0.0]],
            b: array![0.0],
        }
    }

    /// `M = diag(−1, 2)` under `β₀ ≥ 0` at the mode `(0, 0.5)`. The quadratic is improper
    /// along the bound (the normal Schur complement is `−1`) and proper on the face. A
    /// multiplier of `100` puts the exponential scale at `0.01`, where the dropped
    /// curvature is `K = −1e-4`: overturn mass `e^{−20000}`, first-order bound
    /// `½·√5·1e-4·√24 = 5.48e-4`, both inside `1e-3`.
    #[test]
    fn a_strong_multiplier_certifies_the_exponential_boundary_law() {
        let precision = array![[-1.0, 0.0], [0.0, 2.0]];
        let mode = array![0.0, 0.5];
        let gradient = array![100.0, 0.0];
        let approximation = BoundaryModeApproximation::at_converged_mode(
            precision.view(),
            &lower_bound_on_the_first_coordinate(),
            &mode,
            &gradient,
        )
        .expect("a multiplier of 100 against a normal curvature of -1 certifies");
        assert_eq!(approximation.active_rows, vec![0]);
        let rate = approximation.certificate.rates[0];
        assert!((rate - 100.0).abs() <= 1e-10, "rate {rate}");
        assert!(
            (approximation.mean[0] - 0.01).abs() <= 1e-14
                && (approximation.mean[1] - 0.5).abs() <= 1e-14,
            "E[β] = β̂ + P/μ = (0.01, 0.5), got {:?}",
            approximation.mean
        );
        let covariance = approximation.covariance();
        assert!(
            (covariance[[0, 0]] - 1e-4).abs() <= 1e-16
                && (covariance[[1, 1]] - 0.5).abs() <= 1e-14
                && covariance[[0, 1]].abs() <= 1e-16,
            "Cov[β] = diag(1/μ², 1/2), got {covariance:?}"
        );
        assert!(
            (approximation.certificate.scaled_copositive_minimum + 1e-4).abs() <= 1e-16,
            "K = S/μ² = -1e-4, got {}",
            approximation.certificate.scaled_copositive_minimum
        );
        approximation
            .validate(2, 1)
            .expect("the published approximation validates at its own dimension");
    }

    /// Same fixture, multiplier `1`: `K = −1`, so the dropped curvature overturns the
    /// multiplier term at `1ᵀe ≥ 2`, which carries `e^{−2}` of the law.
    #[test]
    fn a_weak_multiplier_is_refused_by_its_certificate() {
        let precision = array![[-1.0, 0.0], [0.0, 2.0]];
        let mode = array![0.0, 0.5];
        let gradient = array![1.0, 0.0];
        let refusal = BoundaryModeApproximation::at_converged_mode(
            precision.view(),
            &lower_bound_on_the_first_coordinate(),
            &mode,
            &gradient,
        )
        .expect_err("a multiplier of 1 against a normal curvature of -1 is not certified");
        assert!(
            refusal.reason.contains("not certified") && refusal.reason.contains("overturn tail mass 1.353e-1"),
            "the refusal must name the failing entry and its value, got: {refusal}"
        );
        let certificate = refusal
            .certificate
            .as_ref()
            .expect("a refusal by the certificate keeps the measured certificate");
        assert!(
            (certificate.overturn_tail_mass - (-2.0_f64).exp()).abs() <= 1e-10,
            "the recorded overturn tail mass must be e^-2, got {:e}",
            certificate.overturn_tail_mass
        );
    }

    #[test]
    fn a_zero_multiplier_has_no_exponential_scale() {
        let precision = array![[-1.0, 0.0], [0.0, 2.0]];
        let mode = array![0.0, 0.5];
        let gradient = array![0.0, 0.0];
        let refusal = BoundaryModeApproximation::at_converged_mode(
            precision.view(),
            &lower_bound_on_the_first_coordinate(),
            &mode,
            &gradient,
        )
        .expect_err("a zero multiplier holds no coordinate");
        assert!(refusal.reason.contains("strict complementarity"), "got: {refusal}");
    }

    /// The fixture `a_cone_improper_posterior_keeps_the_mode_under_a_named_decline` uses:
    /// `M = diag(1, −2)` is indefinite ON the face, so no Gaussian face law exists.
    #[test]
    fn an_indefinite_face_is_refused_before_any_multiplier_is_read() {
        let precision = array![[1.0, 0.0], [0.0, -2.0]];
        let mode = array![0.0, 0.5];
        let gradient = array![100.0, 0.0];
        let refusal = BoundaryModeApproximation::at_converged_mode(
            precision.view(),
            &lower_bound_on_the_first_coordinate(),
            &mode,
            &gradient,
        )
        .expect_err("an indefinite face has no Gaussian law");
        assert!(refusal.reason.contains("not positive definite on the 1-dimensional face tangent"), "got: {refusal}");
    }

    /// An inactive wall at `β₁ ≤ 0.6` sits `0.1` from the mode against a face standard
    /// deviation of `√0.5`, so the law reaches it. The control wall at `β₁ ≤ 100` is far
    /// beyond the horizon and certifies.
    #[test]
    fn an_inactive_wall_inside_the_laws_reach_is_refused_and_a_far_one_is_not() {
        let precision = array![[-1.0, 0.0], [0.0, 2.0]];
        let mode = array![0.0, 0.5];
        let gradient = array![100.0, 0.0];
        let near = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, -1.0]],
            b: array![0.0, -0.6],
        };
        let refusal =
            BoundaryModeApproximation::at_converged_mode(precision.view(), &near, &mode, &gradient)
                .expect_err("a wall 0.1 away inside a spread of 0.71 is within reach");
        assert!(refusal.reason.contains("inactive constraint row 1"), "got: {refusal}");
        let far = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, -1.0]],
            b: array![0.0, -100.0],
        };
        let approximation =
            BoundaryModeApproximation::at_converged_mode(precision.view(), &far, &mode, &gradient)
                .expect("a wall 99.5 away is beyond the double-precision horizon");
        assert_eq!(approximation.active_rows, vec![0]);
    }

    #[test]
    fn the_gamma_upper_tail_matches_its_closed_forms() {
        assert!((gamma_upper_tail(1, 2.0) - (-2.0_f64).exp()).abs() <= 1e-15);
        assert!((gamma_upper_tail(2, 3.0) - 4.0 * (-3.0_f64).exp()).abs() <= 1e-15);
        assert_eq!(gamma_upper_tail(3, 1.0e6), 0.0);
    }
}
