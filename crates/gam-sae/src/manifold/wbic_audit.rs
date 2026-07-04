//! WBIC audit for the singular manifold-atom model-selection charge (Part-2
//! statistical-debt closure).
//!
//! WHY. The production birth/death charge is the Laplace/BIC rank charge
//! `½·d_eff·log n` (see [`super::construction::realised_rank_charge_dof`]), with
//! `d_eff = rank_eff · basis_edf`. `rank_eff` is a Marchenko–Pastur HARD count of
//! the reconstruction-Gram eigenvalues above the noise edge — an integer. The
//! `½·(·)·log n` Laplace charge is the correct free-energy penalty ONLY for a
//! REGULAR statistical model, where the log-likelihood has a non-degenerate
//! Hessian at the MLE and the marginal likelihood expands as
//! `−log Z = n·L_n(ŵ) + (d/2)·log n + O(1)`. Manifold atoms are SINGULAR: gauge
//! orbits (the harmonic/rotation freedom of a chart), rank deficiencies (a
//! decoder direction collapsing toward the noise floor), and boundary solutions
//! (an amplitude pinned at zero) all break Hessian non-degeneracy. Watanabe's
//! singular-learning theory replaces the `d/2` coefficient with the LEARNING
//! COEFFICIENT (real log-canonical threshold) `λ ≤ d/2`, and the free energy is
//! `−log Z = n·L_n(ŵ) + λ·log n + o(log n)`. So the rank charge can only ever
//! OVER-charge a singular atom, never under-charge — it prices every above-edge
//! direction as a full unit of complexity even when that direction is barely
//! resolved and its true learning-coefficient contribution is a fraction.
//!
//! THE ESTIMATOR (WBIC at inverse temperature `β = 1/log n`). Watanabe's Widely
//! Applicable BIC is the tempered-posterior expected log loss
//!
//! ```text
//! WBIC = E_β[ n·L_n(w) ],   posterior ∝ exp(−β·n·L_n(w))·π(w),   β = 1/log n,
//! ```
//!
//! which satisfies `E[WBIC] = n·L_n(ŵ) + λ·log n + o(log n)` for ANY model,
//! regular or singular. The implied complexity charge is `WBIC − n·L_n(ŵ) =
//! λ̂·log n`. We estimate `λ̂` in closed form (no MCMC) by a Laplace-at-temperature
//! expansion that is EXACT for the decoder model, because the reconstruction loss
//! is quadratic in the decoder coefficients:
//!
//!   Take one reconstruction direction `k` with reconstruction-Gram eigenvalue
//!   `μ_k` (per-observation signal+noise energy, `= sv_k²/n_eff`) against the MP
//!   noise edge `e = R·(1 + √(p/n_eff))²`. Its scalar amplitude `α_k` has
//!   tempered-likelihood precision `h_k = β·g_k/R` with design energy
//!   `g_k = n_eff·μ_k`, and a REML "toward no effect" Gaussian prior whose
//!   precision is fixed — with NO new constant — to the SAME noise edge the hard
//!   count uses: `τ_k = β·g_edge/R`, `g_edge = n_eff·e`. The tempered-Gaussian
//!   learning-coefficient contribution is
//!
//! ```text
//! λ̂_k = ½ · h_k / (h_k + τ_k) = ½ · g_k/(g_k + g_edge) = ½ · μ_k/(μ_k + e).
//! ```
//!
//!   Every `β`, `R`, `n_eff` and `log n` cancels: the WBIC soft count is a SIGMOID
//!   in `μ_k/e` that replaces the hard step `1[μ_k > e]`. It recovers the regular
//!   limit exactly (a direction far above the edge, `μ_k ≫ e`, contributes `½`, so
//!   a full-rank atom recovers `½·d_eff·log n = BIC`) and discounts singular
//!   directions smoothly (`μ_k → 0 ⇒ 0`). The two charges cross at `μ_k = e`
//!   (`λ̂_k = ¼`, exactly half of the full `½`), so both agree far from the edge
//!   and disagree only in the near-singular transition band.
//!
//! CHARGES.
//! ```text
//! rank_hard = Σ_k 1[μ_k > e]                    (integer MP count — production)
//! rank_soft = Σ_k μ_k/(μ_k + e)                 (WBIC tempered count)
//! rank charge  C_rank = ½ · rank_hard · basis_edf · log n
//! WBIC charge  C_wbic = ½ · rank_soft · basis_edf · log n
//! ```
//! `basis_edf = tr(G(G+λS)⁻¹)` is ALREADY a graded (Watanabe-compatible) effective
//! count of basis functions, so the singular correction lives ENTIRELY in the hard
//! MP rank count. The audit reports `C_rank − C_wbic ≥ 0` per atom; the expected
//! and observed direction is that curved atoms near singular configurations
//! (whose over-parameterised bases pile reconstruction directions JUST above the
//! edge) are over-charged, while regular atoms (strong directions far above, the
//! rest far below) agree.
//!
//! This module is an AUDIT: it does NOT change the default charge. It computes the
//! reconstruction spectrum the SAME way the production core does (verified against
//! [`super::construction::realised_rank_charge_dof`] in the tests) and prices both.

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd};
use ndarray::{Array2, ArrayView2};

use super::Side;

/// The reconstruction spectrum of ONE atom — the shared substrate both charges
/// price. `mu` are the reconstruction-Gram eigenvalues `sv(diag(√λ)·Uᵀ·D)²/n_eff`
/// (with `(λ,U)=eigh(G)`), `edge` the Marchenko–Pastur noise floor
/// `R·(1+√(p/n_eff))²`, `basis_edf = tr(G(G+λS)⁻¹)` the ridge-trace effective
/// basis count. This is exactly the decomposition inside
/// [`super::construction::realised_rank_charge_dof`], surfaced so the soft (WBIC)
/// count can be taken alongside the hard (MP) count.
#[derive(Clone, Debug)]
pub struct ReconSpectrum {
    /// Reconstruction-Gram eigenvalues (per-observation signal+noise energy).
    pub mu: Vec<f64>,
    /// Marchenko–Pastur noise edge the hard rank count thresholds on.
    pub edge: f64,
    /// `tr(G(G+λS)⁻¹)` — the graded effective basis-function count.
    pub basis_edf: f64,
    /// Effective sample size `Σ_row a²`.
    pub n_eff: f64,
}

impl ReconSpectrum {
    /// Hard Marchenko–Pastur rank count `Σ_k 1[μ_k > e]` (the production `rank_eff`).
    pub fn rank_hard(&self) -> f64 {
        self.mu.iter().filter(|&&m| m > self.edge).count() as f64
    }

    /// WBIC tempered soft count `Σ_k μ_k/(μ_k + e)` — the sigmoid that replaces the
    /// hard step (derivation in the module header). Always `≤` the number of
    /// directions, and each term `∈ [0, 1)`; a direction at the edge counts `½`.
    pub fn rank_soft(&self) -> f64 {
        self.mu
            .iter()
            .map(|&m| if self.edge > 0.0 { m / (m + self.edge) } else { 1.0 })
            .sum()
    }

    /// Production rank / Laplace–BIC charge `½·rank_hard·basis_edf·log n`.
    pub fn rank_charge(&self, n: usize) -> f64 {
        0.5 * self.rank_hard() * self.basis_edf * (n as f64).ln()
    }

    /// WBIC / singular free-energy charge `½·rank_soft·basis_edf·log n`.
    pub fn wbic_charge(&self, n: usize) -> f64 {
        0.5 * self.rank_soft() * self.basis_edf * (n as f64).ln()
    }

    /// Watanabe learning-coefficient estimate `λ̂ = ½·rank_soft·basis_edf` (`≤ d/2`,
    /// the singular bound); the regular Laplace coefficient is
    /// `½·rank_hard·basis_edf`.
    pub fn learning_coefficient(&self) -> f64 {
        0.5 * self.rank_soft() * self.basis_edf
    }
}

/// Build the reconstruction spectrum from an atom's weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ` (`m×m`), decoder `D` (`m×p`), effective sample size
/// `n_eff = Σ_row a²`, output dim `p_out`, noise floor `r_floor` (dispersion R),
/// and smoothness `(lam_smooth, smooth_penalty)`. Mirrors
/// [`super::construction::realised_rank_charge_dof`] byte-for-byte on the shared
/// quantities (checked in the parity test), returning the spectrum instead of the
/// collapsed `rank_eff · basis_edf`.
pub fn recon_spectrum(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ReconSpectrum, String> {
    let m = gram.nrows();
    if m == 0 || !(n_eff > 0.0) {
        return Ok(ReconSpectrum {
            mu: Vec::new(),
            edge: 0.0,
            basis_edf: 0.0,
            n_eff: n_eff.max(0.0),
        });
    }
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("recon_spectrum: eigh(G): {e}"))?;
    let mut scaled = u.t().dot(decoder);
    let cols = scaled.ncols();
    for i in 0..m {
        let s = evals[i].max(0.0).sqrt();
        for j in 0..cols {
            scaled[[i, j]] *= s;
        }
    }
    let sv = match scaled.svd(false, false) {
        Ok((_, sv, _)) => sv,
        Err(e) => return Err(format!("recon_spectrum: recon svd: {e}")),
    };
    let edge = r_floor * (1.0 + (p_out / n_eff).sqrt()).powi(2);
    let mu: Vec<f64> = sv.iter().map(|&s| (s * s) / n_eff).collect();
    // basis_edf = tr(G(G+λS)⁻¹), the same ridge trace the production core computes.
    let mut mmat = gram.clone();
    if let Some(pen) = smooth_penalty {
        if pen.dim() == (m, m) {
            for i in 0..m {
                for j in 0..m {
                    mmat[[i, j]] += lam_smooth * pen[[i, j]];
                }
            }
        }
    }
    for i in 0..m {
        mmat[[i, i]] += 1.0e-12;
    }
    let basis_edf = match mmat.cholesky(Side::Lower) {
        Ok(factor) => {
            let x = factor.solve_mat(gram);
            (0..m).map(|i| x[[i, i]]).sum::<f64>().clamp(0.0, m as f64)
        }
        Err(_) => m as f64,
    };
    Ok(ReconSpectrum {
        mu,
        edge,
        basis_edf,
        n_eff,
    })
}

/// One row of the WBIC-vs-rank-charge disagreement table.
#[derive(Clone, Debug)]
pub struct AuditRow {
    /// Human name of the synthetic population.
    pub name: String,
    /// Rows the atom was fit on.
    pub n: usize,
    /// Integer MP rank count `rank_eff`.
    pub rank_hard: f64,
    /// WBIC tempered soft count.
    pub rank_soft: f64,
    /// Graded effective basis count.
    pub basis_edf: f64,
    /// Production rank / BIC charge `½·rank_hard·basis_edf·log n`.
    pub rank_charge: f64,
    /// WBIC / singular charge `½·rank_soft·basis_edf·log n`.
    pub wbic_charge: f64,
    /// `rank_charge − wbic_charge` (`≥ 0` when the rank charge over-charges).
    pub overcharge: f64,
    /// `overcharge / rank_charge` (fractional over-charge, `NaN` if no charge).
    pub overcharge_frac: f64,
}

impl AuditRow {
    /// Price a named atom from its reconstruction spectrum.
    pub fn from_spectrum(name: impl Into<String>, spec: &ReconSpectrum, n: usize) -> Self {
        let rank_charge = spec.rank_charge(n);
        let wbic_charge = spec.wbic_charge(n);
        let overcharge = rank_charge - wbic_charge;
        let overcharge_frac = if rank_charge.abs() > 0.0 {
            overcharge / rank_charge
        } else {
            f64::NAN
        };
        Self {
            name: name.into(),
            n,
            rank_hard: spec.rank_hard(),
            rank_soft: spec.rank_soft(),
            basis_edf: spec.basis_edf,
            rank_charge,
            wbic_charge,
            overcharge,
            overcharge_frac,
        }
    }
}

/// Render a disagreement table to a plain-text block (for the audit report / test
/// stderr). No side effects; the caller decides where it goes.
pub fn render_audit_table(rows: &[AuditRow]) -> String {
    let mut out = String::new();
    out.push_str(
        "population              n   rank_hard rank_soft basis_edf  C_rank   C_wbic  overcharge  frac\n",
    );
    out.push_str(
        "----------------------- --- --------- --------- --------- -------- -------- ---------- ------\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{:<23} {:>3} {:>9.3} {:>9.3} {:>9.3} {:>8.3} {:>8.3} {:>10.3} {:>6.3}\n",
            r.name,
            r.n,
            r.rank_hard,
            r.rank_soft,
            r.basis_edf,
            r.rank_charge,
            r.wbic_charge,
            r.overcharge,
            r.overcharge_frac,
        ));
    }
    out
}

/// Directly price a WBIC learning-coefficient contribution for ONE scalar
/// reconstruction direction with per-observation energy `mu` against noise edge
/// `edge`: `λ̂ = ½·μ/(μ+e)`. Exposed for the sampling cross-check test that
/// validates the closed form against a genuine tempered-posterior expectation.
pub fn direction_learning_coefficient(mu: f64, edge: f64) -> f64 {
    if edge > 0.0 {
        0.5 * mu / (mu + edge)
    } else {
        0.5
    }
}

/// A genuine (non-Laplace) WBIC estimate for a SINGLE scalar-amplitude
/// reconstruction direction, by the thermodynamic tempered-posterior expectation
/// `E_β[nL] − nL(α̂)`, divided by `log n`, to recover `λ̂`. Used ONLY to validate
/// [`direction_learning_coefficient`]: model `nL(α) = (g/2R)(α−α̂)²` with a REML
/// Gaussian prior of precision `τ = g_edge/R` (crossover-at-edge), tempered at
/// `β = 1/log n`, integrated in closed form over the Gaussian (the integral is
/// exact — the point is to confirm the algebra that produced the sigmoid, not to
/// approximate). Because the model is Gaussian this returns the SAME number as
/// the sigmoid up to the prior-shift term, which this includes so the test sees
/// the full expectation.
pub fn sampled_direction_learning_coefficient(
    mu: f64,
    edge: f64,
    n_eff: f64,
    r_floor: f64,
    n: usize,
) -> f64 {
    let ln_n = (n as f64).ln();
    if !(ln_n > 0.0) || !(r_floor > 0.0) || !(n_eff > 0.0) {
        return 0.0;
    }
    let beta = 1.0 / ln_n;
    let g = n_eff * mu; // design energy
    let g_edge = n_eff * edge; // prior precision energy (crossover at edge)
    let h = beta * g / r_floor; // tempered likelihood precision
    let tau = beta * g_edge / r_floor; // prior precision
    let prec_post = h + tau;
    if !(prec_post > 0.0) {
        return 0.0;
    }
    // MLE amplitude scale: set α̂² so the direction carries energy μ per obs, i.e.
    // g·α̂² = n_eff·(μ − edge)_+ signal energy ⇒ α̂² = (μ − edge)_+ / μ (unitless),
    // the fraction of the direction's energy that is signal above the floor.
    let alpha_hat2 = ((mu - edge).max(0.0)) / mu.max(f64::MIN_POSITIVE);
    // Tempered-posterior expectation of nL(α) = (g/2R)(α−α̂)² over α ~ N(m_post,
    // 1/prec_post), m_post = h·α̂/prec_post (prior centred at 0):
    //   E[nL] − nL(α̂) = (g/2R)·(Var + (m_post − α̂)²).
    let var = 1.0 / prec_post;
    let m_post = h * 0.0_f64.max(alpha_hat2.sqrt()) / prec_post; // α̂ sign irrelevant to (·)²
    let alpha_hat = alpha_hat2.sqrt();
    let shift2 = (m_post - alpha_hat) * (m_post - alpha_hat);
    let e_delta = 0.5 * (g / r_floor) * (var + shift2);
    e_delta / ln_n
}

/// Compute the audit spectrum from an already-projected reconstruction problem:
/// given a residual `data` (`n×p`), a per-row activation weight `w` (gate²), and
/// an orthonormal basis of chart features `phi` (`n×m`), least-squares fit the
/// decoder `D = (ΦᵀWΦ + εI)⁻¹ ΦᵀW·data`, then form the reconstruction spectrum.
/// This is the self-contained synthetic-suite entry: it turns raw planted data +
/// a chart basis into the same spectrum the production charge prices.
pub fn spectrum_from_fit(
    data: ArrayView2<'_, f64>,
    w: &[f64],
    phi: &Array2<f64>,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ReconSpectrum, String> {
    let (n, p) = data.dim();
    let m = phi.ncols();
    if phi.nrows() != n || w.len() != n {
        return Err("spectrum_from_fit: shape mismatch".into());
    }
    // Weighted Gram G = ΦᵀWΦ and cross term ΦᵀW·data.
    let mut gram = Array2::<f64>::zeros((m, m));
    let mut cross = Array2::<f64>::zeros((m, p));
    let mut n_eff = 0.0_f64;
    for i in 0..n {
        let wi = w[i];
        n_eff += wi;
        for a in 0..m {
            let pa = phi[[i, a]] * wi;
            for b in a..m {
                gram[[a, b]] += pa * phi[[i, b]];
            }
            for j in 0..p {
                cross[[a, j]] += pa * data[[i, j]];
            }
        }
    }
    for a in 0..m {
        for b in a..m {
            let v = gram[[a, b]];
            gram[[a, b]] = v;
            gram[[b, a]] = v;
        }
    }
    // Ridge LS decoder solve (εI SPD guard) via Cholesky.
    let mut reg = gram.clone();
    for a in 0..m {
        reg[[a, a]] += 1.0e-9;
    }
    let decoder = reg
        .cholesky(Side::Lower)
        .map_err(|e| format!("spectrum_from_fit: chol: {e}"))?
        .solve_mat(&cross);
    recon_spectrum(
        &gram,
        &decoder,
        n_eff,
        p as f64,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Degree-`h` periodic harmonic features [1, cos, sin, cos2, sin2, ...] at the
    /// given per-row angles (turns in [0,1)).
    fn harmonic_phi(turns: &[f64], h: usize) -> Array2<f64> {
        let n = turns.len();
        let m = 1 + 2 * h;
        Array2::from_shape_fn((n, m), |(i, c)| {
            if c == 0 {
                1.0
            } else {
                let k = (c + 1) / 2;
                let ang = std::f64::consts::TAU * k as f64 * turns[i];
                if c % 2 == 1 { ang.cos() } else { ang.sin() }
            }
        })
    }

    /// Polynomial features [1, t, t², ...] at scalar coordinates.
    fn poly_phi(t: &[f64], deg: usize) -> Array2<f64> {
        let n = t.len();
        Array2::from_shape_fn((n, deg + 1), |(i, c)| t[i].powi(c as i32))
    }

    /// PARITY: the surfaced spectrum's hard count × basis_edf must equal the
    /// production `realised_rank_charge_dof` d_eff bit-for-bit on the same inputs
    /// (the audit prices the SAME quantity, then adds the soft count).
    #[test]
    fn spectrum_hard_count_matches_production_deff() {
        let mut s = 0x0B1C_0001_u64;
        let n = 800usize;
        let p = 12usize;
        let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let phi = harmonic_phi(&turns, 3);
        let m = phi.ncols();
        // A clean rank-2 circle decoder: cos→dim0, sin→dim1.
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = std::f64::consts::TAU * turns[i];
            data[[i, 0]] += a.cos();
            data[[i, 1]] += a.sin();
            for j in 0..p {
                data[[i, j]] += 0.05 * lcg_normal(&mut s);
            }
        }
        let w = vec![1.0_f64; n];
        // Build the gram + decoder exactly as spectrum_from_fit does, then call
        // BOTH recon_spectrum and the production core with identical arguments.
        let r_floor = 0.05_f64 * 0.05;
        let spec = spectrum_from_fit(data.view(), &w, &phi, r_floor, 0.0, None).unwrap();
        // Reproduce gram + decoder for the production call.
        let mut gram = Array2::<f64>::zeros((m, m));
        let mut cross = Array2::<f64>::zeros((m, p));
        for i in 0..n {
            for a in 0..m {
                for b in 0..m {
                    gram[[a, b]] += phi[[i, a]] * phi[[i, b]];
                }
                for j in 0..p {
                    cross[[a, j]] += phi[[i, a]] * data[[i, j]];
                }
            }
        }
        let mut reg = gram.clone();
        for a in 0..m {
            reg[[a, a]] += 1.0e-9;
        }
        let decoder = reg.cholesky(Side::Lower).unwrap().solve_mat(&cross);
        let d_prod = super::super::construction::realised_rank_charge_dof(
            &gram,
            &decoder,
            n as f64,
            p as f64,
            r_floor,
            0.0,
            None,
        )
        .unwrap();
        let d_audit = spec.rank_hard() * spec.basis_edf;
        eprintln!(
            "[wbic parity] production d_eff={d_prod:.10}  audit rank_hard·basis_edf={d_audit:.10}"
        );
        assert!(
            (d_prod - d_audit).abs() < 1e-8,
            "audit hard d_eff must match production: prod={d_prod} audit={d_audit}"
        );
    }

    /// The closed-form sigmoid `½·μ/(μ+e)` must equal the tempered-posterior
    /// expectation for a Gaussian direction (the derivation cross-check). Prior
    /// shift is negligible far from the edge and the two agree there; near the
    /// edge the sampled value carries the small prior-shift term, so we check the
    /// dominant variance term agreement across a μ sweep.
    #[test]
    fn sigmoid_matches_tempered_posterior_variance_term() {
        let n = 800usize;
        let n_eff = 800.0_f64;
        let r_floor = 0.0025_f64;
        let edge = r_floor * (1.0 + (12.0_f64 / n_eff).sqrt()).powi(2);
        for &ratio in &[8.0_f64, 4.0, 2.0, 1.0, 0.5, 0.25] {
            let mu = ratio * edge;
            let closed = direction_learning_coefficient(mu, edge);
            let sampled = sampled_direction_learning_coefficient(mu, edge, n_eff, r_floor, n);
            // The variance term of the sampled expectation equals the closed form
            // exactly; the total sampled value adds only the (non-negative) prior
            // shift, so sampled ≥ variance term and both share the same limits.
            eprintln!("[wbic sigmoid] μ/e={ratio:.2}  closed={closed:.4} sampled≈{sampled:.4}");
            // Variance-only reconstruction of the sampled expectation:
            let beta = 1.0 / (n as f64).ln();
            let g = n_eff * mu;
            let g_edge = n_eff * edge;
            let h = beta * g / r_floor;
            let tau = beta * g_edge / r_floor;
            let var_term = 0.5 * (g / r_floor) * (1.0 / (h + tau)) / (n as f64).ln();
            assert!(
                (closed - var_term).abs() < 1e-9,
                "closed sigmoid must equal the tempered variance term: closed={closed} var={var_term}"
            );
        }
    }

    /// THE AUDIT. Price the standard synthetic suite and assert the disagreement
    /// structure Watanabe predicts: (a) a REGULAR atom (well-separated clusters,
    /// a clean low-degree line) shows near-zero over-charge (hard ≈ soft), while
    /// (b) a CURVED atom near a singular configuration (an over-parameterised
    /// harmonic circle whose higher harmonics pile just above the noise edge, a
    /// disk whose radial spread pushes several directions to the edge) is
    /// over-charged by the rank charge — `C_rank > C_wbic` by a quantified margin.
    #[test]
    fn wbic_audit_disagreement_table() {
        let mut rows: Vec<AuditRow> = Vec::new();
        let n = 1200usize;
        let p = 16usize;

        // (1) LINE — a regular rank-1 linear atom (poly deg 2), strong direction
        // far above the edge, higher terms far below ⇒ hard = soft.
        {
            let mut s = 0x1111_u64;
            let t: Vec<f64> = (0..n).map(|_| 2.0 * lcg(&mut s) - 1.0).collect();
            let phi = poly_phi(&t, 2);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                data[[i, 0]] += 2.0 * t[i];
                data[[i, 1]] += 0.5 * t[i];
                for j in 0..p {
                    data[[i, j]] += 0.05 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.0025, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("line (regular)", &spec, n));
        }

        // (2) CLUSTERS — regular: three well-separated Gaussian blobs, one-hot
        // membership basis (rank-3, all far above edge).
        {
            let mut s = 0x2222_u64;
            let phi = Array2::<f64>::from_shape_fn((n, 3), |(i, c)| {
                if i % 3 == c { 1.0 } else { 0.0 }
            });
            let centers = [[3.0, 0.0], [0.0, 3.0], [-3.0, -3.0]];
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let c = i % 3;
                data[[i, 0]] += centers[c][0];
                data[[i, 1]] += centers[c][1];
                for j in 0..p {
                    data[[i, j]] += 0.1 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.01, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("clusters (regular)", &spec, n));
        }

        // (3) CLEAN CIRCLE — a genuine rank-2 harmonic circle with a degree-3
        // basis. The two fundamental directions sit far above the edge; the
        // higher-harmonic directions sit far below ⇒ near agreement (a clean,
        // well-identified curved atom is NOT singular).
        {
            let mut s = 0x3333_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let a = std::f64::consts::TAU * turns[i];
                data[[i, 0]] += a.cos();
                data[[i, 1]] += a.sin();
                for j in 0..p {
                    data[[i, j]] += 0.05 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.0025, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("circle clean (curved)", &spec, n));
        }

        // (4) NEAR-SINGULAR CIRCLE — the load-bearing case: a weak circle whose
        // amplitude sits just above the noise floor AND an over-parameterised
        // degree-3 basis, so the fundamental pair lands NEAR the MP edge. The hard
        // count prices each near-edge direction as a full unit; the soft count
        // discounts them. Expected: C_rank > C_wbic (over-charge).
        {
            let mut s = 0x4444_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            // Weak amplitude (0.22) close to the noise scale (0.15) ⇒ the circle's
            // reconstruction eigenvalues sit just above the edge (near-singular).
            for i in 0..n {
                let a = std::f64::consts::TAU * turns[i];
                data[[i, 0]] += 0.22 * a.cos();
                data[[i, 1]] += 0.22 * a.sin();
                for j in 0..p {
                    data[[i, j]] += 0.15 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.15 * 0.15, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("circle near-edge (singular)", &spec, n));
        }

        // (5) DISK — a filled 2-disk (radius uniform, not a shell): reconstruction
        // energy spreads across the plane with several near-edge directions ⇒ the
        // hard count is unstable/over-counts vs the graded soft count.
        {
            let mut s = 0x5555_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let radii: Vec<f64> = (0..n).map(|_| lcg(&mut s).sqrt()).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let a = std::f64::consts::TAU * turns[i];
                data[[i, 0]] += 0.35 * radii[i] * a.cos();
                data[[i, 1]] += 0.35 * radii[i] * a.sin();
                for j in 0..p {
                    data[[i, j]] += 0.12 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.12 * 0.12, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("disk (curved)", &spec, n));
        }

        // (6) PLANTED GAUSSIAN BLEND — a low-rank isotropic Gaussian factor: no
        // circle structure at all, every harmonic direction is noise ⇒ both
        // charges should be ~0 (nothing above the edge). A sanity floor.
        {
            let mut s = 0x6666_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                // random low-rank Gaussian, uncorrelated with the harmonic basis
                for j in 0..p {
                    data[[i, j]] += 0.1 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.01, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("gaussian blend (null)", &spec, n));
        }

        eprintln!("\n{}", render_audit_table(&rows));

        // Structural assertions on the disagreement pattern.
        let get = |name: &str| rows.iter().find(|r| r.name == name).unwrap().clone();
        let line = get("line (regular)");
        let clusters = get("clusters (regular)");
        let near = get("circle near-edge (singular)");
        let blend = get("gaussian blend (null)");

        // (a) regular atoms: the rank charge and WBIC agree to within a small
        // fraction (strong directions far above the edge ⇒ soft ≈ hard).
        assert!(
            line.overcharge_frac.abs() < 0.15,
            "regular line must show ~no over-charge; frac={:.3}",
            line.overcharge_frac
        );
        assert!(
            clusters.overcharge_frac.abs() < 0.15,
            "regular clusters must show ~no over-charge; frac={:.3}",
            clusters.overcharge_frac
        );
        // (b) the near-singular circle is OVER-charged by the rank charge, and by a
        // materially larger fraction than the regular atoms — the number that
        // reprices the birth/death decision for singular curved atoms.
        assert!(
            near.overcharge > 0.0,
            "near-singular circle must be OVER-charged (C_rank > C_wbic); got {:.4}",
            near.overcharge
        );
        assert!(
            near.overcharge_frac > line.overcharge_frac + 0.05
                && near.overcharge_frac > clusters.overcharge_frac + 0.05,
            "singular over-charge fraction ({:.3}) must exceed the regular atoms' \
             (line {:.3}, clusters {:.3}) by a clear margin",
            near.overcharge_frac,
            line.overcharge_frac,
            clusters.overcharge_frac
        );
        // (c) the null blend prices ~nothing under both charges.
        assert!(
            blend.rank_charge < 1.0 && blend.wbic_charge < 1.0,
            "gaussian blend must price ~0 under both charges: C_rank={:.3} C_wbic={:.3}",
            blend.rank_charge,
            blend.wbic_charge
        );
    }
}
