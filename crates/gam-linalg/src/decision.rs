//! Decision-currency primitives (theory master design §9-step-6, issue #2337).
//!
//! A "decision" here is a discrete claim extracted from floating-point data —
//! the numerical rank of a design, whether a Newton step is stationary, whether
//! a reduction is real or roundoff. The theme of this module is that such a
//! claim is trustworthy only when it is expressed in a *currency* that is
//! invariant under the symmetries of the problem and is decided with a margin
//! wider than the backward error committed forming the quantity we decide on.
//! Each item below carries its derivation as a doc comment.
//!
//! The four currencies:
//!   * [`equilibrate_gram`] — a Gram/rank decision must be gauge-invariant under
//!     positive per-column rescaling; equilibration puts it in that gauge.
//!   * [`certified_rank`] — a rank claim with a two-sided multiplicative gap is
//!     stable (bitwise-reproducible) under perturbations below the band width.
//!   * [`newton_decrement_enclosure`] — the Newton decrement λ_N² is the
//!     affine-invariant stationarity currency; an inexact solve still yields a
//!     rigorous two-sided enclosure of it.
//!   * [`ShadowSum`] — a reduction carries its own rounding floor, so "is this
//!     decrement real?" is decided against a certified error bar.

use ndarray::{Array1, Array2};

/// Diagonally equilibrate a symmetric Gram matrix into its column-scale gauge.
///
/// Returns `(C, s)` with `C = D^{-1/2} · G · D^{-1/2}`, `D = diag(G)`, and the
/// per-column scale vector `s_j = sqrt(G_jj)`. A column with `G_jj ≤ 0` (a null
/// or numerically empty direction) is given unit scale `s_j = 1`, leaving its
/// row/column of `C` unchanged.
///
/// # Why this is the right currency for a rank decision
///
/// **Congruence preserves the decision (Sylvester's law of inertia).** With
/// `Δ = D^{-1/2} = diag(1/s_j) ≻ 0`, `C = ΔGΔ` is a *congruence* of `G`. By
/// Sylvester's law of inertia a congruence `C = MᵀGM` with `M` nonsingular
/// preserves the inertia `(n₊, n₀, n₋)` of `G`, hence its rank and the sign of
/// every eigenvalue. Deciding the rank of `C` therefore decides the rank of
/// `G` exactly — equilibration changes the conditioning, never the answer.
///
/// **Gauge invariance.** For any positive diagonal `Λ = diag(λ_j) ≻ 0`, the map
/// `G ↦ ΛGΛ` sends `D ↦ Λ D Λ`, so `s_j ↦ λ_j s_j` and
/// `(ΛGΛ)_{ij} / (λ_i s_i · λ_j s_j) = G_{ij}/(s_i s_j) = C_{ij}`. Thus `C` is
/// *invariant* under the column-scale gauge `G ↦ ΛGΛ`: it is the canonical
/// representative of `G`'s congruence orbit under positive diagonal scaling.
/// A rank test on `C` cannot be fooled by a single stiff column.
///
/// **Near-optimal conditioning (van der Sluis).** Among all positive diagonal
/// scalings `Δ`, equilibrating so that `diag(ΔGΔ)` is constant (here, unit) is
/// within a factor of `p` of the minimum achievable spectral condition number:
/// `κ(D^{-1/2}GD^{-1/2}) ≤ p · min_{Δ≻0 diagonal} κ(ΔGΔ)` for a `p×p` SPD `G`
/// (van der Sluis, 1969). So this cheap choice is provably close to the best a
/// diagonal preconditioner can do, and the residual decision then reads the
/// true correlation structure rather than the column magnitudes.
pub fn equilibrate_gram(g: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let p = g.nrows();
    let scale: Array1<f64> = Array1::from_shape_fn(p, |j| {
        let d = g[[j, j]];
        if d > 0.0 { d.sqrt() } else { 1.0 }
    });
    let mut c = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            c[[i, j]] = g[[i, j]] / (scale[i] * scale[j]);
        }
    }
    (c, scale)
}

/// Outcome of a certified numerical-rank decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankDecision {
    /// The rank is certified: every kept singular value clears the upper band
    /// edge and every dropped one falls below the lower band edge.
    Certified {
        /// Certified numerical rank `r`.
        rank: usize,
        /// Smallest kept singular value `σ_r` (`+∞` when `rank == 0`).
        sigma_r: f64,
        /// Largest dropped singular value `σ_{r+1}` (`0` when `rank == n`).
        sigma_next: f64,
        /// Multiplicative slack of the dropped side below the lower edge,
        /// `low / σ_{r+1}` (`+∞` when `σ_{r+1} = 0`); `≥ 1` by construction.
        margin_low: f64,
        /// Multiplicative slack of the kept side above the upper edge,
        /// `σ_r / high` (`+∞` when `rank == 0`); `≥ 1` by construction.
        margin_high: f64,
    },
    /// The rank is undecidable at this tolerance: a singular value lands inside
    /// the open guard band `(tol/(1+gap), tol·(1+gap))`.
    Ambiguous {
        /// Rank if the in-band value is treated as dropped: `#{σ ≥ high}`.
        rank_floor: usize,
        /// Rank if the in-band value is treated as kept: `#{σ > low}`.
        rank_ceil: usize,
        /// The offending singular value sitting inside the band.
        sigma_in_band: f64,
        /// Tolerance the decision was posed at.
        tol: f64,
        /// Multiplicative half-gap the decision was posed with.
        gap: f64,
    },
}

/// Certify the numerical rank of a spectrum against a two-sided guard band.
///
/// The rank `r` is [`Certified`](RankDecision::Certified) iff
/// `σ_r ≥ tol·(1+gap)` **and** `σ_{r+1} ≤ tol/(1+gap)` (with `σ_{n+1} := 0`);
/// otherwise the outcome is [`Ambiguous`](RankDecision::Ambiguous), naming the
/// value that fell inside the band. Inputs need not be sorted; they are ordered
/// descending internally.
///
/// # Why the two-sided gap is the decision's currency
///
/// **Perturbation invariance ⇒ host stability.** Write the band edges as
/// `high = tol·(1+gap)`, `low = tol/(1+gap)`. A Certified decision keeps every
/// `σ ≥ high` and drops every `σ ≤ low`; the open interval `(low, high)` is
/// empty of data. Any perturbation `|Δσ_i|` strictly smaller than the distance
/// from each `σ_i` to the nearer band edge leaves the partition — and hence the
/// integer `r` — unchanged. The decision is therefore a locally constant
/// function of the spectrum: identical (bitwise) integer outputs for any inputs
/// agreeing to within the margins. Given reproducible inputs it is
/// host-stable. An `Ambiguous` outcome is the honest report that no such
/// margin exists, so the integer would be host-dependent.
///
/// **Decide in the design's currency, not its square.** Forming the raw Gram
/// `G = XᵀX` and deciding on its eigenvalues squares the condition number:
/// `κ(G) = κ(X)²`, and the eigenvalues are computed with backward error
/// `O(u · σ_max(X)²)` (u the unit roundoff) — the decision inherits an error
/// bar quadratic in the design's scale. Deciding on the *equilibrated* design
/// or its equilibrated Gram (see [`equilibrate_gram`]) commits only
/// `O(u · σ_max)` in the decision's own linear currency, so a gap of a few `u`
/// suffices to certify. This is why the caller equilibrates first, then
/// certifies.
pub fn certified_rank(singular_values: &[f64], tol: f64, gap: f64) -> RankDecision {
    let mut sv: Vec<f64> = singular_values.to_vec();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = sv.len();
    let high = tol * (1.0 + gap);
    let low = tol / (1.0 + gap);

    // A singular value strictly inside the open band makes `r` undecidable.
    if let Some(&sigma_in_band) = sv.iter().find(|&&s| s > low && s < high) {
        let rank_floor = sv.iter().filter(|&&s| s >= high).count();
        let rank_ceil = sv.iter().filter(|&&s| s > low).count();
        return RankDecision::Ambiguous {
            rank_floor,
            rank_ceil,
            sigma_in_band,
            tol,
            gap,
        };
    }

    // Clean split: everything is either `≥ high` (kept) or `≤ low` (dropped).
    let rank = sv.iter().filter(|&&s| s >= high).count();
    let sigma_r = if rank == 0 { f64::INFINITY } else { sv[rank - 1] };
    let sigma_next = if rank < n { sv[rank] } else { 0.0 };
    let margin_high = if rank == 0 { f64::INFINITY } else { sigma_r / high };
    let margin_low = if sigma_next == 0.0 {
        f64::INFINITY
    } else {
        low / sigma_next
    };
    RankDecision::Certified {
        rank,
        sigma_r,
        sigma_next,
        margin_low,
        margin_high,
    }
}

/// A rigorous two-sided enclosure `[lower, upper]` of a scalar quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecrementEnclosure {
    /// Certified lower bound.
    pub lower: f64,
    /// Certified upper bound.
    pub upper: f64,
}

/// Enclose the squared Newton decrement `λ_N² = gᵀH⁻¹g` from an inexact solve.
///
/// Given an approximate solve `z ≈ H⁻¹g` with residual `r = g − Hz` and a
/// positive lower bound `ℓ ≤ λ_min(H)` on the Hessian's smallest eigenvalue,
/// returns `[gᵀz + rᵀz, gᵀz + rᵀz + ‖r‖²/ℓ]`, which contains `λ_N²`. Returns
/// `None` when `ℓ ≤ 0` (no positive-definite certificate available).
///
/// The arguments are `g_dot_z = gᵀz`, `r_dot_z = rᵀz`, `r_norm_sq = ‖r‖²`,
/// and `lambda_min_lower = ℓ`.
///
/// # Derivation
///
/// Substitute `g = Hz + r` (the definition of the residual) into `λ_N²`:
///
/// ```text
///   λ_N² = gᵀH⁻¹g = (Hz + r)ᵀ H⁻¹ (Hz + r)
///        = zᵀHz + 2 rᵀz + rᵀH⁻¹r         [ (Hz)ᵀH⁻¹(Hz) = zᵀHz, symmetry ]
///        = zᵀ(Hz + r) + rᵀz + rᵀH⁻¹r     [ regroup: zᵀHz + rᵀz = zᵀ(Hz)+rᵀz ]
///        = zᵀg + rᵀz + rᵀH⁻¹r
///        = gᵀz + rᵀz + rᵀH⁻¹r.
/// ```
///
/// For `H ⪰ ℓI ≻ 0` we have `0 ⪯ H⁻¹ ⪯ (1/ℓ)I`, hence
/// `0 ≤ rᵀH⁻¹r ≤ ‖r‖²/ℓ`. Adding the constant `gᵀz + rᵀz` to this two-sided
/// bound on the only unknown term gives the enclosure. When `r = 0` (exact
/// solve) the enclosure collapses to the exact `λ_N² = gᵀz`.
///
/// # Why `λ_N` is *the* stationarity currency
///
/// The decrement is affine-invariant: under a coordinate change `θ ↦ Tθ` the
/// gradient and Hessian transform as `g ↦ T^{-T}g`, `H ↦ T^{-T}HT^{-1}`, so
/// `gᵀH⁻¹g ↦ gᵀT⁻¹ (T H⁻¹ Tᵀ) T^{-T} g = gᵀH⁻¹g` is unchanged. Unlike `‖g‖`,
/// which depends on the arbitrary parameterization, `λ_N²` measures proximity
/// to the stationary point in the metric the problem itself supplies — so a
/// stopping test posed in this currency is invariant to how the model is
/// coordinatized.
pub fn newton_decrement_enclosure(
    g_dot_z: f64,
    r_dot_z: f64,
    r_norm_sq: f64,
    lambda_min_lower: f64,
) -> Option<DecrementEnclosure> {
    if lambda_min_lower <= 0.0 {
        return None;
    }
    let lower = g_dot_z + r_dot_z;
    let upper = lower + r_norm_sq / lambda_min_lower;
    Some(DecrementEnclosure { lower, upper })
}

/// A running sum that also carries the data needed to certify its own rounding
/// floor: the accumulated value, the sum of magnitudes, and the term count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShadowSum {
    /// Accumulated (finite-precision) sum.
    pub sum: f64,
    /// Sum of magnitudes `Σ|x_i|`, the scale of the rounding floor.
    pub abs_sum: f64,
    /// Number of terms pushed.
    pub count: usize,
}

impl ShadowSum {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one term into the running sum.
    pub fn push(&mut self, x: f64) {
        self.sum += x;
        self.abs_sum += x.abs();
        self.count += 1;
    }

    /// Combine two independently accumulated sums (associative, for reductions).
    pub fn merge(&mut self, other: &ShadowSum) {
        self.sum += other.sum;
        self.abs_sum += other.abs_sum;
        self.count += other.count;
    }

    /// Certified forward-error floor for **sequential** summation:
    /// `γ_{n−1} · Σ|x_i|`.
    ///
    /// # Derivation
    ///
    /// Let `γ_k = k·u / (1 − k·u)` (Higham's constant, `u` the unit roundoff).
    /// The standard forward-error bound for recursive summation of `n` terms is
    /// `|fl(S) − S| ≤ γ_{n−1} · Σ_{i} |x_i|`: each of the `n − 1` additions
    /// commits a relative error `≤ u`, the errors compound multiplicatively as
    /// `∏(1 + δ_i)` with `|δ_i| ≤ u`, and `∏(1+δ_i) − 1` is bounded in modulus
    /// by `γ_{n−1}` (Higham, *Accuracy and Stability of Numerical Algorithms*,
    /// Lemma 3.1 and §4.2). The bound is returned in the operands' own units so
    /// a candidate decrement can be compared directly against it.
    ///
    /// If `(n−1)·u ≥ 1` the bound `γ_{n−1}` is not defined (the geometric
    /// factor diverges); we saturate to `+∞`, the honest statement that at this
    /// term count and precision no nontrivial floor can be certified.
    pub fn rounding_floor(&self, unit_roundoff: f64) -> f64 {
        let depth = self.count.saturating_sub(1);
        gamma(depth, unit_roundoff) * self.abs_sum
    }

    /// Certified forward-error floor for a reduction of a given `depth`:
    /// `γ_depth · Σ|x_i|`.
    ///
    /// Sequential summation has depth `n − 1`; pairwise/tree reduction lowers
    /// the number of additions on any accumulation path to `⌈log₂ n⌉`,
    /// improving the constant from `γ_{n−1}` to `γ_{⌈log₂ n⌉}`. A caller that
    /// reduces with a tree (see [`crate::pairwise_reduce`]) passes that
    /// effective depth here to obtain the tighter, still-rigorous floor.
    pub fn rounding_floor_with_depth(&self, unit_roundoff: f64, depth: usize) -> f64 {
        gamma(depth, unit_roundoff) * self.abs_sum
    }
}

/// Higham's `γ_k = k·u / (1 − k·u)`, saturating to `+∞` once `k·u ≥ 1`.
fn gamma(k: usize, unit_roundoff: f64) -> f64 {
    let ku = (k as f64) * unit_roundoff;
    if ku >= 1.0 {
        f64::INFINITY
    } else {
        ku / (1.0 - ku)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_ndarray::{FaerEigh, fast_ata};
    use faer::Side;
    use ndarray::{Array1, Array2};

    // Machine unit roundoff for f64 (2^-53).
    const U: f64 = f64::EPSILON / 2.0;

    fn eigenvalues(m: &Array2<f64>) -> Vec<f64> {
        let (evals, _) = m.eigh(Side::Lower).expect("eigh");
        evals.to_vec()
    }

    #[test]
    fn equilibration_certifies_full_rank_where_raw_gram_would_kill_eleven_columns() {
        // A 12-column design whose first column is stiffened by 2.4e6 so the
        // Gram anisotropy is (2.4e6)² ≈ 5.76e12. Columns are orthonormal (rows
        // 0..11 form an identity block; extra rows are zero) so the raw Gram is
        // diagonal: eigenvalues [ (2.4e6)², 1, …, 1 ].
        let n = 50usize;
        let p = 12usize;
        let stiff = 2.4e6_f64;
        let mut x = Array2::<f64>::zeros((n, p));
        for j in 0..p {
            x[[j, j]] = if j == 0 { stiff } else { 1.0 };
        }
        let g = fast_ata(&x);

        // Raw-Gram decision with the codebase-style size-scaled machine-epsilon
        // cutoff τ = λ_max · 64 · n · ε (the schematic "u·p·λ_max" of the
        // design; here the size factor is n): the eleven unit eigenvalues sit
        // below τ and are killed, leaving rank 1.
        let raw_evals = eigenvalues(&g);
        let raw_lambda_max = raw_evals.iter().cloned().fold(0.0_f64, f64::max);
        let raw_tol = raw_lambda_max * 64.0 * (n as f64) * f64::EPSILON;
        let raw_rank = raw_evals.iter().filter(|&&e| e > raw_tol).count();
        assert_eq!(raw_rank, 1, "raw size-scaled cutoff must kill 11 columns");

        // Equilibrated decision: D^{-1/2} G D^{-1/2} = I_12, every eigenvalue is
        // 1, and certified_rank returns the full rank 12 with a huge margin.
        let (g_eq, _) = equilibrate_gram(&g);
        let eq_evals = eigenvalues(&g_eq);
        for &e in &eq_evals {
            assert!((e - 1.0).abs() < 1e-9, "equilibrated spectrum must be ~1");
        }
        let eq_lambda_max = eq_evals.iter().cloned().fold(0.0_f64, f64::max);
        let nk = (n.max(p)) as f64;
        let eq_tol = eq_lambda_max * 64.0 * nk * f64::EPSILON;
        match certified_rank(&eq_evals, eq_tol, 1.0) {
            RankDecision::Certified {
                rank, margin_high, ..
            } => {
                assert_eq!(rank, 12, "equilibrated Gram is full rank");
                assert!(
                    margin_high > 1e10,
                    "kept side must clear the band by a huge factor, got {margin_high}"
                );
            }
            other => panic!("expected Certified full rank, got {other:?}"),
        }
    }

    #[test]
    fn spectrum_inside_two_sided_band_is_ambiguous() {
        // tol = 1, gap = 1 ⇒ band (0.5, 2). The value 1.0 lands inside it.
        let sv = [10.0_f64, 3.0, 1.0, 0.2];
        match certified_rank(&sv, 1.0, 1.0) {
            RankDecision::Ambiguous {
                rank_floor,
                rank_ceil,
                sigma_in_band,
                ..
            } => {
                assert_eq!(rank_floor, 2, "#{{σ ≥ 2}} = 2");
                assert_eq!(rank_ceil, 3, "#{{σ > 0.5}} = 3");
                assert_eq!(sigma_in_band, 1.0);
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn decrement_enclosure_is_exact_when_residual_zero_and_contains_truth_when_perturbed() {
        // Small SPD H, exact g, exact z = H⁻¹g.
        let h = Array2::from_shape_vec(
            (3, 3),
            vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        )
        .unwrap();
        let g = Array1::from_vec(vec![1.0, -2.0, 0.5]);

        // Exact solve via the spectral inverse H⁻¹ = V diag(1/λ) Vᵀ.
        let (evals, evecs) = h.eigh(Side::Lower).expect("eigh");
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(lambda_min > 0.0, "H must be SPD");
        let vt_g = evecs.t().dot(&g);
        let scaled: Array1<f64> =
            Array1::from_shape_fn(3, |i| vt_g[i] / evals[i]);
        let z = evecs.dot(&scaled);
        let true_lambda_n_sq = g.dot(&z);

        // r = g − Hz = 0 (up to roundoff): enclosure collapses to the truth.
        let hz = h.dot(&z);
        let r = &g - &hz;
        let g_dot_z = g.dot(&z);
        let r_dot_z = r.dot(&z);
        let r_norm_sq = r.dot(&r);
        let ell = lambda_min * 0.999; // valid lower bound ℓ ≤ λ_min(H)
        let exact = newton_decrement_enclosure(g_dot_z, r_dot_z, r_norm_sq, ell)
            .expect("positive definite");
        assert!(
            (exact.upper - exact.lower).abs() < 1e-10,
            "width must be ~0 when r=0"
        );
        assert!((exact.lower - true_lambda_n_sq).abs() < 1e-9);

        // Perturb z; the enclosure must still contain the true λ_N².
        let z_bad = &z + &Array1::from_vec(vec![0.05, -0.03, 0.02]);
        let hz_bad = h.dot(&z_bad);
        let r_bad = &g - &hz_bad;
        let encl = newton_decrement_enclosure(
            g.dot(&z_bad),
            r_bad.dot(&z_bad),
            r_bad.dot(&r_bad),
            ell,
        )
        .expect("positive definite");
        assert!(
            encl.lower <= true_lambda_n_sq + 1e-9 && true_lambda_n_sq <= encl.upper + 1e-9,
            "enclosure [{}, {}] must contain λ_N² = {true_lambda_n_sq}",
            encl.lower,
            encl.upper
        );
        assert!(encl.upper - encl.lower > 0.0, "inexact solve widens the band");

        // A non-positive lower bound yields no certificate.
        assert!(newton_decrement_enclosure(g_dot_z, r_dot_z, r_norm_sq, 0.0).is_none());
    }

    #[test]
    fn shadow_sum_error_stays_within_certified_rounding_floor() {
        let mut acc = ShadowSum::new();
        for _ in 0..1_000_000 {
            acc.push(0.1);
        }
        assert_eq!(acc.count, 1_000_000);
        let exact = 100_000.0_f64;
        let error = (acc.sum - exact).abs();
        let floor = acc.rounding_floor(U);
        assert!(
            error <= floor,
            "summation error {error} must not exceed rounding floor {floor}"
        );
        // The tree-depth floor is tighter but must still bound the sequential
        // error only if the caller actually reduced with a tree; here we merely
        // check the constant shrinks with depth.
        assert!(acc.rounding_floor_with_depth(U, 20) < floor);
    }

    #[test]
    fn shadow_sum_merge_is_additive() {
        let mut a = ShadowSum::new();
        let mut b = ShadowSum::new();
        a.push(1.0);
        a.push(-2.0);
        b.push(3.0);
        a.merge(&b);
        assert_eq!(a.count, 3);
        assert_eq!(a.sum, 2.0);
        assert_eq!(a.abs_sum, 6.0);
    }

    #[test]
    fn gamma_saturates_when_ku_exceeds_one() {
        assert!(gamma(usize::MAX, U).is_infinite());
        assert_eq!(gamma(0, U), 0.0);
    }
}
