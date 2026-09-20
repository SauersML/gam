//! Test support for the full-conformal tests: an independent closed-form
//! Gaussian-REML criterion, the oracle the Layer-3 tests of [`super::honest`]
//! re-select `ρ̂(z)` against.

use faer::Side;
use ndarray::{Array1, Array2};

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};

/// `Σ_i r_i²` over the residuals `r_i = y_i − x_iᵀβ`.
fn residual_sum_of_squares(x: &Array2<f64>, y: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    let mut total = 0.0;
    for (row, &response) in x.rows().into_iter().zip(y.iter()) {
        let mut fitted = 0.0;
        for (&entry, &coefficient) in row.iter().zip(beta.iter()) {
            fitted += entry * coefficient;
        }
        let residual = response - fitted;
        total += residual * residual;
    }
    total
}

/// Closed-form Gaussian-REML criterion for the single-penalty model
/// `Sλ = λ S` (`ρ = log λ`), with or without the augmented test row.
///
/// # Why this object exists
///
/// Layer 1 ([`super::ExactGaussianFullConformal`]) holds ρ fixed, but the
/// Gaussian fitting map re-selects ρ̂ by REML on whatever data it sees,
/// including the augmented row `(x_*, z)`. [`super::honest`] computes that
/// map's set without ever evaluating `ρ̂(z)` pointwise; this object evaluates
/// the criterion pointwise from its own dense Cholesky, which is what an
/// independent oracle for [`super::honest`] needs.
///
/// # The closed forms (single penalty `Sλ = λ S`)
///
/// Augmented penalized least squares with the test row included:
///
/// ```text
///   A(λ)   = XᵀX + x_* x_*ᵀ + λ S         (independent of z)
///   c(z)   = Xᵀy + x_* z ,  β̂ = A(λ)⁻¹ c(z) = a + b z   (affine in z)
///   D(ρ,z) = ‖y_aug‖² − c(z)ᵀ A(λ)⁻¹ c(z)              (penalized RSS)
/// ```
///
/// The Gaussian REML criterion to MINIMIZE over ρ (σ² profiled out, additive
/// constants dropped; `M₀ = nullity(S)`, `r = rank(S)`, `n_eff = n (+1` if the
/// test row is present`)`):
///
/// ```text
///   Ṽ(ρ,z) = (n_eff − M₀) · log D(ρ,z) + log|A(λ)| − r ρ
/// ```
pub(super) struct GaussianRemlRhoResponse<'a> {
    x: &'a Array2<f64>,
    y: &'a Array1<f64>,
    s: &'a Array2<f64>,
    x_star: &'a Array1<f64>,
    n: usize,
    p: usize,
    rank_s: usize,
    pub(super) xtx: Array2<f64>,
    pub(super) xty: Array1<f64>,
    pub(super) rho_domain: (f64, f64),
    pub(super) augmented_rho_domain: (f64, f64),
}

/// One closed-form evaluation of the (possibly augmented) Gaussian REML
/// criterion at `ρ = log λ`.
pub(super) struct RemlEval {
    /// `Ṽ(ρ,z)` (additive constants dropped — only differences in ρ matter).
    pub(super) value: f64,
    /// The penalized RSS `D` the criterion's `log D` term is taken of.
    penalized_rss: f64,
}

impl<'a> GaussianRemlRhoResponse<'a> {
    /// Build the response object. Computes `rank(S)` once by symmetric
    /// eigendecomposition, counting the eigenvalues above the REML engine's
    /// `positive_eigenvalue_threshold`: the positive-eigenspace decision the
    /// fit's own penalty pseudo-logdet makes.
    pub(super) fn new(
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        s: &'a Array2<f64>,
        x_star: &'a Array1<f64>,
    ) -> Result<Self, String> {
        let n = x.nrows();
        let p = x.ncols();
        if y.len() != n {
            return Err("gaussian reml response: row-count mismatch".to_string());
        }
        if s.nrows() != p || s.ncols() != p || x_star.len() != p {
            return Err("gaussian reml response: column-count mismatch".to_string());
        }
        let (evals, _) = s.eigh(Side::Lower).map_err(|e| {
            format!("gaussian reml response: penalty eigendecomposition failed: {e:?}")
        })?;
        let threshold = gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold(
            evals.as_slice().ok_or_else(|| {
                "gaussian reml response: penalty eigenvalues are not contiguous".to_string()
            })?,
        );
        let rank_s = evals.iter().filter(|&&e| e > threshold).count();
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);
        // #2902 row 8: ρ is searched in the #2812 resolvability domain of the Gram
        // against S. The test row adds `x_* x_*ᵀ` to the Gram whatever z is, so
        // every ρ̂(z) shares one augmented domain.
        let domain_of = |gram: &Array2<f64>| {
            gam_solve::estimate::rho_domain::coordinate_domain(
                gam_solve::estimate::rho_domain::penalty_range_gammas_from_gram(gram, s)
                    .as_deref()
                    .and_then(gam_solve::estimate::rho_domain::resolvability_interval),
                None,
            )
        };
        let rho_domain = domain_of(&xtx);
        let mut augmented_xtx = xtx.clone();
        for i in 0..p {
            for j in 0..p {
                augmented_xtx[[i, j]] += x_star[i] * x_star[j];
            }
        }
        let augmented_rho_domain = domain_of(&augmented_xtx);
        Ok(Self {
            x,
            y,
            s,
            x_star,
            n,
            p,
            rank_s,
            xtx,
            xty,
            rho_domain,
            augmented_rho_domain,
        })
    }

    /// Closed-form REML evaluation at `ρ`. `z = Some(_)` augments with the
    /// test row; `z = None` is the original-data criterion.
    pub(super) fn eval(&self, rho: f64, z: Option<f64>) -> Result<RemlEval, String> {
        let p = self.p;
        let n_eff = self.n + usize::from(z.is_some());
        let m0 = p - self.rank_s;
        if n_eff <= m0 {
            return Err(format!(
                "gaussian reml response: degrees of freedom n_eff−M₀ = {n_eff}−{m0} ≤ 0; \
                 REML criterion undefined"
            ));
        }
        let coef = (n_eff - m0) as f64;
        let r = self.rank_s as f64;
        let lambda = gam_problem::checked_exp_log_strength(rho)
            .map_err(|error| format!("gaussian REML conformal response: {error}"))?;

        // A(λ) = XᵀX + λ S [+ x_* x_*ᵀ].
        let mut a = self.xtx.clone();
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] += lambda * self.s[[i, j]];
            }
        }
        if z.is_some() {
            for i in 0..p {
                for j in 0..p {
                    a[[i, j]] += self.x_star[i] * self.x_star[j];
                }
            }
        }
        let chol = a
            .cholesky(Side::Lower)
            .map_err(|e| format!("gaussian reml response: A(λ) not SPD: {e:?}"))?;

        // c(z) = Xᵀy [+ x_* z].
        let mut c = self.xty.clone();
        if let Some(zv) = z {
            for j in 0..p {
                c[j] += self.x_star[j] * zv;
            }
        }
        let beta = chol.solvevec(&c);
        let pen = lambda * beta.dot(&self.s.dot(&beta));

        // #2280: `D` is the penalized sum of squares at β̂, formed as one:
        // `‖y − Xβ̂‖² [+ (z − x_*ᵀβ̂)²] + pen`. The closed form it equals at the exact
        // solve, `yᵀy [+ z²] − cᵀβ̂`, is a difference that rounds at `u·yᵀy` and takes
        // the solve's backward error `E` at first order, as `β̂ᵀEβ̂`, so a `D` below
        // that scale came back as roundoff, or as a non-positive "degenerate fit".
        // Formed from residuals, the solve's error enters `D` only at second order.
        let training_rss = residual_sum_of_squares(self.x, self.y, &beta);
        let test_rss = match z {
            Some(zv) => {
                let mut fitted = 0.0;
                for (&entry, &coefficient) in self.x_star.iter().zip(beta.iter()) {
                    fitted += entry * coefficient;
                }
                let residual = zv - fitted;
                residual * residual
            }
            None => 0.0,
        };
        let d = training_rss + test_rss + pen;
        if !(d > 0.0) {
            return Err(format!(
                "gaussian reml response: non-positive penalized RSS D = {d}; degenerate fit"
            ));
        }

        let logdet: f64 = 2.0 * chol.diag().iter().map(|d| d.ln()).sum::<f64>();
        Ok(RemlEval {
            value: coef * d.ln() + logdet - r * rho,
            penalized_rss: d,
        })
    }

    /// The penalized RSS `D(ρ, z)` of the augmented criterion (or the training
    /// criterion for `z = None`): the quantity whose `log` the REML criterion
    /// takes.
    pub(super) fn penalized_rss(&self, rho: f64, z: Option<f64>) -> Result<f64, String> {
        Ok(self.eval(rho, z)?.penalized_rss)
    }
}
