//! Shared #771 gate for a Tweedie `y ~ x` (log link) fit: the fitted dispersion
//! `φ̂` must sit within a derived band of the true `φ` (#4131).
//!
//! The engine's `φ̂` is the Pearson statistic at the fitted mean,
//! `φ̂ = mean tᵢ(μ̂ᵢ)` with `tᵢ(μ) = (yᵢ − μ)²/μ^p` (`estimate_tweedie_phi_from_eta`).
//! Its sampling SD follows from the Tweedie cumulants
//! `κ₂ = φμ^p`, `κ₃ = pφ²μ^{2p−1}`, `κ₄ = p(2p−1)φ³μ^{3p−2}` together with the
//! influence of the fitted mean. To first order,
//!
//! `φ̂ − φ ≈ mean(tᵢ − φ) − pφ·x̄ᵀ(β̂ − β)`, `β̂ − β ≈ H⁻¹·mean(xᵢ sᵢ)`,
//!
//! where `sᵢ = (yᵢ − μᵢ)μᵢ^{1−p}` is the log-link score, `H = mean μ^{2−p}xxᵀ`,
//! and `−pφ` is `E ∂tᵢ/∂ηᵢ`. With `Var tᵢ = p(2p−1)φ³μ^{p−2} + 2φ²`,
//! `Cov(tᵢ, sᵢ) = pφ²` and `Var sᵢ = φμ^{2−p}`, this gives
//!
//! `n·Var φ̂ = mean[p(2p−1)φ³μ^{p−2} + 2φ²] − p²φ³·x̄ᵀH⁻¹x̄`.
//!
//! The plug-in mean also biases the statistic by `−(#coefficients)·φ/n` (the `n`
//! versus `n − p` divisor). The gate therefore reads
//! `|φ̂ − φ| ≤ Z·SD(φ̂) + (#coefficients)·φ/n`. `Z = 4` is a two-sided normal
//! tail of 6.3e-5 per check. In the #4131 simulation (2000 replicates of each
//! fixture), z had SD 0.96 to 1.03 and max |z| = 3.9. A naive i.i.d. SE, without the mean-influence
//! term, gives z SD 0.56 at high φ.

/// Two-sided normal quantile of the φ̂ gate (tail 6.3e-5 per check).
const PHI_GATE_Z: f64 = 4.0;

/// A Tweedie(p) log-link `y ~ x` fixture with unit prior weights and true mean
/// `μᵢ = exp(b0 + bx·xᵢ)`.
pub(super) struct TweedieLogFixture<'a> {
    pub x: &'a [f64],
    pub b0: f64,
    pub bx: f64,
    pub p: f64,
}

impl TweedieLogFixture<'_> {
    /// First-order SD of the Pearson `φ̂` at the true mean and true `φ`.
    pub(super) fn pearson_phi_sd(&self, phi: f64) -> f64 {
        let p = self.p;
        let n = self.x.len() as f64;
        let mut var_t = 0.0;
        // H = mean μ^{2−p} (1, x)(1, x)ᵀ and x̄ = mean (1, x).
        let (mut h00, mut h01, mut h11, mut xbar1) = (0.0, 0.0, 0.0, 0.0);
        for &xi in self.x {
            let mu = (self.b0 + self.bx * xi).exp();
            var_t += p * (2.0 * p - 1.0) * phi.powi(3) * mu.powf(p - 2.0) + 2.0 * phi * phi;
            let w = mu.powf(2.0 - p);
            h00 += w;
            h01 += w * xi;
            h11 += w * xi * xi;
            xbar1 += xi;
        }
        var_t /= n;
        h00 /= n;
        h01 /= n;
        h11 /= n;
        xbar1 /= n;
        // x̄ᵀH⁻¹x̄ for x̄ = (1, x̄₁) with the 2×2 inverse written out.
        let det = h00 * h11 - h01 * h01;
        let quad = (h11 - 2.0 * h01 * xbar1 + h00 * xbar1 * xbar1) / det;
        let n_var = var_t - p * p * phi.powi(3) * quad;
        assert!(
            n_var.is_finite() && n_var > 0.0,
            "φ̂ variance is not positive: {n_var:?}"
        );
        (n_var / n).sqrt()
    }

    /// Assert that a fitted `φ̂` recovers the true `φ` within the derived band.
    /// `n_coefficients` is the fit's coefficient count (the divisor bias).
    pub(super) fn assert_phi_hat_recovers(
        &self,
        tag: &str,
        phi_hat: f64,
        phi: f64,
        n_coefficients: usize,
    ) {
        let n = self.x.len() as f64;
        let sd = self.pearson_phi_sd(phi);
        let bias = n_coefficients as f64 * phi / n;
        let bar = PHI_GATE_Z * sd + bias;
        let z = (phi_hat - phi) / sd;
        eprintln!(
            "[tweedie-φ̂ {tag}] true φ={phi}; φ̂={phi_hat:.5}; SD(φ̂)={sd:.5}; z={z:.3}; \
             bar |φ̂−φ| ≤ {bar:.5} ({PHI_GATE_Z}·SD + divisor bias {bias:.5})"
        );
        assert!(
            (phi_hat - phi).abs() <= bar,
            "Tweedie φ̂ ({tag}) is off the true dispersion: φ̂={phi_hat:.5} vs φ={phi} \
             (|φ̂−φ| = {:.5} > {bar:.5} = {PHI_GATE_Z}·SD {sd:.5} + divisor bias {bias:.5}; \
             z = {z:.3})",
            (phi_hat - phi).abs()
        );
    }
}
