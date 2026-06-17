//! Certified Chebyshev-in-ψ Gram tensor: n-independent design-moving trials
//! (#1033 item b).
//!
//! ## Why
//!
//! When a design-moving hyperparameter ψ (= log κ for the radial families) is
//! searched by the outer loop, every trial today rebuilds the n×k design and
//! re-forms XᵀWX — an O(n·k) + O(n·k²) pass per trial. But along the trial
//! window every design entry `X(ψ)[i, j]` is an ANALYTIC function of ψ on a
//! compact interval (Matérn channels depend on (r, ℓ) only through κr and
//! κ-power prefactors; Duchon power blocks are ψ-free; partial-fraction
//! coefficients are analytic scalars), so the whole design admits a
//! geometrically-convergent Chebyshev expansion
//!
//! ```text
//!   X(ψ) = Σ_{d=0}^{D} X_d · T_d(ψ̃),     ψ̃ = affine map of ψ to [−1, 1],
//! ```
//!
//! with n×k coefficient slabs `X_d` computed ONCE from D+1 exact design
//! evaluations at Chebyshev nodes (a first-kind DCT). Precomputing the
//! expanded Gram `G̃[d][e] = X_dᵀ W X_e` and cross-products `c̃[d] = X_dᵀ W z`
//! in ONE pass over the data makes every subsequent trial n-free:
//!
//! ```text
//!   XᵀWX(ψ) = Σ_{d,e} T_d(ψ̃) T_e(ψ̃) G̃[d][e]          O(D²k²)
//!   XᵀWz(ψ) = Σ_d T_d(ψ̃) c̃[d]                          O(D k)
//!   ∂/∂ψ (XᵀWX) = Σ_{d,e} (T_d′T_e + T_dT_e′) G̃[d][e]   O(D²k²)
//! ```
//!
//! The ψ-gradient comes from the SAME representation as the value — one
//! source of truth, structurally immune to the objective↔gradient desync
//! class. `T_d′(ψ̃) = d·U_{d−1}(ψ̃) · dψ̃/dψ` is closed-form.
//!
//! ## Certification, not approximation-by-fiat
//!
//! Same discipline as [`crate::basis::radial_profile`]: [`PsiGramTensor::build`]
//! returns `None` (callers keep the exact per-trial path) unless BOTH
//! 1. the Chebyshev coefficient tail of the per-ψ AMPLITUDE-NORMALIZED design
//!    `X(ψ)/a(ψ)` decays below [`PSI_GRAM_CERT_RTOL`] of the design scale
//!    (geometric-decay certificate for analytic interpolands, with node-count
//!    escalation; the scalar amplitude `a(ψ)` is factored out before the
//!    certificate so it holds regardless of input geometry — #1216 — and
//!    carried analytically back through the assembled Gram), and
//! 2. deterministic off-node spot checks of the ASSEMBLED Gram against an
//!    exactly rebuilt Gram agree to [`PSI_GRAM_SPOT_RTOL`].
//!
//! Trials outside `[psi_lo, psi_hi]` are the caller's signal to fall back to
//! the exact path ([`PsiGramTensor::contains`]).

use ndarray::{Array1, Array2, ArrayView1};

/// Relative ceiling on the per-column Chebyshev coefficient tail.
pub const PSI_GRAM_CERT_RTOL: f64 = 1.0e-12;

/// Relative agreement required at the off-node Gram spot checks.
pub const PSI_GRAM_SPOT_RTOL: f64 = 1.0e-10;

/// Relative agreement required of the analytic ψ-DERIVATIVE `dgram_dpsi`
/// against a high-order finite difference of the exactly rebuilt Gram, used to
/// certify the interior gradient sub-window. The downstream outer REML gradient
/// contracts `∂G/∂ψ` through `H⁻¹` and `β̂`, amplifying the absolute derivative
/// error by `‖∂G/∂ψ‖·‖H⁻¹‖`; this rtol is set deep enough (≈4 orders below the
/// 1e-7 outer-gradient bar, relative to the Gram-derivative scale) that even
/// the amplified error stays bit-tight in the gradient lane.
pub const PSI_GRAM_GRAD_SPOT_RTOL: f64 = 1.0e-11;

/// Number of equispaced scan points (per side) used to locate the interior
/// gradient sub-window where `dgram_dpsi` certifies.
pub const PSI_GRAM_GRAD_SCAN_POINTS: usize = 64;

/// Node-count escalation ladder for the expansion build (degree = nodes − 1).
///
/// The top rung sizes to WIDE trial windows: Chebyshev coefficients of the
/// Matérn-type channels decay like Bessel `I_d(σ)` with `σ ≈ s_max·halfwidth`
/// (s = κr), which only drops below the 1e-12 tail tolerance for `d ≳ 2σ` —
/// e.g. σ ≈ 9 (s_max ≈ 8, ±1.1 window) needs degree ≳ 40, so 33 nodes refuse
/// and 65 certify. Node counts stay trivially cheap (one design eval each).
///
/// The per-ψ amplitude normalization (#1216, [`PsiGramTensor::amp`]) is what
/// keeps this ladder sufficient on the WIDE STANDARDIZED geometry default 1-D
/// fits use (#1215): without it the design's many-decade ψ-dynamic-range floors
/// the relative coefficient tail above 1e-12 (a rounding floor, not analytic
/// decay) at every rung; factoring out the scalar amplitude `a(ψ)` first lets
/// the normalized design's true geometric tail certify inside the 65-node ladder
/// regardless of input geometry.
pub const PSI_GRAM_NODE_LADDER: [usize; 4] = [9, 17, 33, 65];

/// Number of deterministic off-node spot-check ψ values.
pub const PSI_GRAM_SPOT_POINTS: usize = 3;

/// Certified Chebyshev-in-ψ expansion of a design-moving Gram (#1033b).
///
/// Holds the one-time n-pass products; every per-trial accessor is O(D²k²)
/// or cheaper and never touches n rows again.
pub struct PsiGramTensor {
    psi_lo: f64,
    psi_hi: f64,
    /// Interior sub-window `[grad_psi_lo, grad_psi_hi] ⊆ [psi_lo, psi_hi]` over
    /// which the ANALYTIC ψ-derivative `dgram_dpsi` reproduces the exact design
    /// derivative to [`PSI_GRAM_GRAD_SPOT_RTOL`] (#1033b gradient lane).
    ///
    /// The value reconstruction `gram_at` is certified over the FULL window
    /// (`T_d ≤ 1` everywhere), but the derivative reconstruction amplifies the
    /// coefficient-tail error by `T_d′ ∼ d²`, which blows up toward the window
    /// endpoints (the classic Chebyshev endpoint phenomenon). The gradient lane
    /// therefore only fires on this certified interior sub-window; near-edge
    /// trials keep the exact per-trial slab gradient. `contains` (value lane)
    /// still spans the full window.
    grad_psi_lo: f64,
    grad_psi_hi: f64,
    /// Number of Chebyshev coefficients (degree + 1).
    n_coeff: usize,
    k: usize,
    /// `gram[d * n_coeff + e]` = `Y_dᵀ W Y_e` (k×k) of the **per-ψ NORMALIZED**
    /// design `Y(ψ) = X(ψ)/a(ψ)`; symmetric in (d, e) up to transpose:
    /// `gram[d][e] == gram[e][d]ᵀ`. The physical Gram is recovered analytically
    /// as `XᵀWX(ψ) = a(ψ)²·Σ T_d T_e gram[d][e]` (see [`Self::gram_at`]).
    gram: Vec<Array2<f64>>,
    /// `rhs[d]` = `Y_dᵀ W z` of the normalized design; the physical RHS is
    /// `XᵀWz(ψ) = a(ψ)·Σ T_d rhs[d]`.
    rhs: Vec<Array1<f64>>,
    /// `zᵀWz` — ψ-free, captured at build so the Gaussian sufficient-statistic
    /// triple can be assembled per trial without any row access.
    zt_w_z: f64,
    /// Certified 1-D Chebyshev expansion of the per-ψ design AMPLITUDE
    /// `a(ψ) = √(‖X(ψ)‖²_F / (n·k))` (root-mean-square design entry), in the
    /// mapped coordinate `ψ̃ ∈ [−1,1]`.
    ///
    /// ## Why per-ψ normalization (#1216)
    ///
    /// The raw radial design column is `X(ψ)[i,j] = kernel(r_{ij}·e^{ψ})`. Over
    /// a WIDE ψ-window (production standardizes the covariate axis to unit spread
    /// since #1215, so `r·e^ψ` ranges over orders of magnitude) the design's
    /// amplitude varies by many decades across the window. The first-kind DCT
    /// coefficient slabs `X_d` are then dominated by the large-amplitude end of
    /// the window, and the per-column tail certificate — which is RELATIVE to the
    /// window-global column scale — never sees the small-amplitude end's true
    /// geometric decay: it floors at the rounding level of the large evaluations
    /// (~2e-11 of col-scale, NOT 1e-12), so the certificate refuses at every
    /// rung and the n-free fast path never attaches on standardized geometry.
    ///
    /// Factoring out the scalar amplitude `a(ψ)` BEFORE the certificate collapses
    /// the design's ψ-dynamic-range to ~O(1) across the whole window, so the
    /// NORMALIZED design's Chebyshev tail decays to true `1e-12` within the
    /// existing 65-node ladder REGARDLESS of input geometry. `a(ψ)` is itself a
    /// single positive analytic-in-ψ scalar (root-mean-square of analytic kernel
    /// entries), so it Chebyshev-expands and certifies trivially, and is carried
    /// back through the Gram/rhs and their ψ-derivatives ANALYTICALLY — value,
    /// gradient and curvature stay one source of truth.
    amp: Cheb1d,
}

/// A certified 1-D Chebyshev expansion of a smooth scalar `f(ψ̃)` on the mapped
/// window `ψ̃ ∈ [−1, 1]`, holding its coefficients so `value`, `deriv` (in `ψ̃`)
/// and `deriv2` (in `ψ̃`) are exact O(D) reconstructions. The caller applies the
/// `dψ̃/dψ` chain-rule factors.
struct Cheb1d {
    /// First-kind Chebyshev coefficients `c_0..c_{m−1}` so
    /// `f(ψ̃) = Σ_d c_d T_d(ψ̃)`.
    coeff: Vec<f64>,
}

impl Cheb1d {
    /// Build from `m` exact node samples at the first-kind nodes
    /// `x_i = cos(π(2i+1)/(2m))` (the SAME nodes `build_at` uses), via the
    /// first-kind discrete orthogonality `c_d = (γ_d/m) Σ_i f(x_i) T_d(x_i)`.
    fn from_node_samples(samples: &[f64], t_at_nodes: &[Vec<f64>], m: usize) -> Self {
        let mut coeff = vec![0.0_f64; m];
        for (d, c) in coeff.iter_mut().enumerate() {
            let gamma = if d == 0 { 1.0 } else { 2.0 };
            let mut acc = 0.0_f64;
            for (i, &f) in samples.iter().enumerate() {
                acc += f * t_at_nodes[i][d];
            }
            *c = gamma / m as f64 * acc;
        }
        Self { coeff }
    }

    /// Relative tail-decay certificate, mirroring the design tail certificate:
    /// the trailing quarter of the coefficients must fall below
    /// `PSI_GRAM_CERT_RTOL × scale`, where `scale = max_d |c_d|`.
    fn tail_certifies(&self) -> bool {
        let m = self.coeff.len();
        let scale = self
            .coeff
            .iter()
            .fold(0.0_f64, |a, &c| a.max(c.abs()))
            .max(1e-300);
        let tail_start = m - (m / 4).max(1);
        self.coeff
            .iter()
            .skip(tail_start)
            .all(|&c| c.abs() <= PSI_GRAM_CERT_RTOL * scale)
    }

    fn value(&self, x: f64) -> f64 {
        let t = cheb_t(x, self.coeff.len());
        self.coeff.iter().zip(t.iter()).map(|(&c, &td)| c * td).sum()
    }

    /// `df/dψ̃` (mapped-coordinate derivative).
    fn deriv(&self, x: f64) -> f64 {
        let tp = cheb_t_prime(x, self.coeff.len());
        self.coeff
            .iter()
            .zip(tp.iter())
            .map(|(&c, &tpd)| c * tpd)
            .sum()
    }

    /// `d²f/dψ̃²` (mapped-coordinate second derivative).
    fn deriv2(&self, x: f64) -> f64 {
        let tpp = cheb_t_double_prime(x, self.coeff.len());
        self.coeff
            .iter()
            .zip(tpp.iter())
            .map(|(&c, &tppd)| c * tppd)
            .sum()
    }
}

/// One ladder rung's outcome: a hard evaluation failure aborts the whole
/// build (no larger rung can fix a non-finite design), an uncertified tail
/// escalates to the next rung, and a candidate proceeds to the spot check.
enum BuildOutcome {
    EvalFailed,
    TailNotCertified,
    Candidate(PsiGramTensor),
}

/// Chebyshev values `T_0..T_{n−1}` at `x ∈ [−1, 1]`.
fn cheb_t(x: f64, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n];
    if n > 0 {
        t[0] = 1.0;
    }
    if n > 1 {
        t[1] = x;
    }
    for d in 2..n {
        t[d] = 2.0 * x * t[d - 1] - t[d - 2];
    }
    t
}

/// Chebyshev derivative values `T_0′..T_{n−1}′` at `x ∈ [−1, 1]` in the
/// MAPPED coordinate (multiply by `dx/dψ` for the ψ-derivative):
/// `T_d′ = d · U_{d−1}` with the Chebyshev-U recurrence.
fn cheb_t_prime(x: f64, n: usize) -> Vec<f64> {
    let mut u = vec![0.0; n.max(1)];
    // U_0 = 1, U_1 = 2x, U_d = 2x U_{d−1} − U_{d−2}.
    if !u.is_empty() {
        u[0] = 1.0;
    }
    if n > 1 {
        u[1] = 2.0 * x;
    }
    for d in 2..n {
        u[d] = 2.0 * x * u[d - 1] - u[d - 2];
    }
    let mut tp = vec![0.0; n];
    for d in 1..n {
        tp[d] = d as f64 * u[d - 1];
    }
    tp
}

/// Chebyshev SECOND-derivative values `T_0″..T_{n−1}″` at `x ∈ [−1, 1]` in the
/// MAPPED coordinate (multiply by `(dx/dψ)²` for the ψ-second-derivative).
///
/// Differentiating the value recurrence `T_d = 2x T_{d−1} − T_{d−2}` twice in
/// `x` gives a singularity-free three-term recurrence in lock-step with `cheb_t`
/// / `cheb_t_prime`:
///   `T_d′  = 2 T_{d−1} + 2x T_{d−1}′ − T_{d−2}′`,
///   `T_d″  = 4 T_{d−1}′ + 2x T_{d−1}″ − T_{d−2}″`,
/// with `T_0 = T_0′ = T_0″ = 0`-seeds as below. Unlike the closed form
/// `T_n″ = n((n+1)T_n − U_n)/(x²−1)` this never divides by `x²−1`, so it stays
/// exact at the window edges `x = ±1`.
fn cheb_t_double_prime(x: f64, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n];
    let mut tp = vec![0.0; n];
    let mut tpp = vec![0.0; n];
    if n > 0 {
        t[0] = 1.0; // T_0 = 1, T_0′ = T_0″ = 0
    }
    if n > 1 {
        t[1] = x; // T_1 = x, T_1′ = 1, T_1″ = 0
        tp[1] = 1.0;
    }
    for d in 2..n {
        t[d] = 2.0 * x * t[d - 1] - t[d - 2];
        tp[d] = 2.0 * t[d - 1] + 2.0 * x * tp[d - 1] - tp[d - 2];
        tpp[d] = 4.0 * tp[d - 1] + 2.0 * x * tpp[d - 1] - tpp[d - 2];
    }
    tpp
}

impl PsiGramTensor {
    /// Build and certify the tensor over `psi ∈ [psi_lo, psi_hi]`.
    ///
    /// `eval_design(psi)` must return the EXACT n×k design at `psi` (the same
    /// builder the per-trial path uses — exactness of the expansion is judged
    /// against it). `weights` are the fixed observation weights, `z` the fixed
    /// weighted-response target (e.g. `y − offset`). Returns `None` when the
    /// window is degenerate, any evaluation fails/has non-finite entries, or
    /// no ladder rung certifies — callers then keep the exact per-trial path.
    pub fn build(
        mut eval_design: impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> Option<Self> {
        if !(psi_lo.is_finite() && psi_hi.is_finite()) || psi_hi <= psi_lo {
            return None;
        }
        for &m in PSI_GRAM_NODE_LADDER.iter() {
            match Self::build_at(&mut eval_design, weights, z, psi_lo, psi_hi, m) {
                // An exact evaluation failed or was non-finite somewhere in
                // the window — no larger rung can fix that.
                BuildOutcome::EvalFailed => return None,
                // Tail not yet below the certificate at this rung: escalate.
                // (Conflating this with EvalFailed would kill the ladder at
                // its first — intentionally coarse — rung.)
                BuildOutcome::TailNotCertified => continue,
                BuildOutcome::Candidate(mut candidate) => {
                    if candidate.spot_check(&mut eval_design, weights) {
                        // Narrow the gradient sub-window to the certified
                        // interior (the value lane keeps the full window).
                        candidate.certify_gradient_window(&mut eval_design, weights);
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    fn build_at(
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
        m: usize,
    ) -> BuildOutcome {
        // First-kind Chebyshev nodes (no endpoints) and exact design slabs.
        let mut nodes_x = vec![0.0_f64; m];
        let mut designs: Vec<Array2<f64>> = Vec::with_capacity(m);
        for (i, x_slot) in nodes_x.iter_mut().enumerate() {
            let x = (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * m) as f64).cos();
            *x_slot = x;
            let psi = 0.5 * (psi_lo + psi_hi) + 0.5 * (psi_hi - psi_lo) * x;
            let Ok(design) = eval_design(psi) else {
                return BuildOutcome::EvalFailed;
            };
            if design.iter().any(|v| !v.is_finite()) {
                return BuildOutcome::EvalFailed;
            }
            designs.push(design);
        }
        let (n, k) = designs[0].dim();
        if designs.iter().any(|d| d.dim() != (n, k))
            || weights.len() != n
            || z.len() != n
            || n == 0
            || k == 0
        {
            return BuildOutcome::EvalFailed;
        }
        let t_at_nodes: Vec<Vec<f64>> = nodes_x.iter().map(|&x| cheb_t(x, m)).collect();

        // ── Per-ψ design normalization (#1216) ──────────────────────────────
        // Probe the per-node design amplitude `a(ψ_i) = √(‖X(ψ_i)‖²_F / (n·k))`
        // (root-mean-square design entry). On WIDE standardized geometry the
        // raw design varies by many decades across the window, flooring the
        // RELATIVE coefficient tail above the certificate; normalizing by `a(ψ)`
        // removes that ψ-dynamic-range so the tail decays to true 1e-12 within
        // the existing ladder. `a(ψ)` is a single positive analytic-in-ψ scalar
        // (RMS of analytic kernel entries), Chebyshev-expanded and certified
        // here, then carried analytically back through the Gram/rhs.
        let nk = (n * k) as f64;
        let amp_samples: Vec<f64> = designs
            .iter()
            .map(|d| (d.iter().map(|&v| v * v).sum::<f64>() / nk).sqrt())
            .collect();
        // A degenerate (all-zero) design at any node has no amplitude to factor
        // out — refuse rather than divide by zero (no larger rung repairs it).
        if amp_samples.iter().any(|&a| !a.is_finite() || a <= 0.0) {
            return BuildOutcome::EvalFailed;
        }
        let amp = Cheb1d::from_node_samples(&amp_samples, &t_at_nodes, m);
        // The amplitude scalar must itself certify (it is analytic, so this is
        // far inside the ladder); a non-certifying amplitude means the design is
        // not analytically normalizable on this window — fall back.
        if !amp.tail_certifies() {
            return BuildOutcome::TailNotCertified;
        }

        // First-kind discrete orthogonality on the NORMALIZED design slabs
        //   Y_d = (γ_d / m) Σ_i (X(ψ_i)/a(ψ_i)) T_d(x_i),  γ_0 = 1, γ_d = 2.
        let mut coeff_slabs: Vec<Array2<f64>> = Vec::with_capacity(m);
        for d in 0..m {
            let gamma = if d == 0 { 1.0 } else { 2.0 };
            let mut slab = Array2::<f64>::zeros((n, k));
            for (i, design) in designs.iter().enumerate() {
                let wgt = gamma / m as f64 * t_at_nodes[i][d] / amp_samples[i];
                slab.scaled_add(wgt, design);
            }
            coeff_slabs.push(slab);
        }
        // Tail-decay certificate per design column: the trailing quarter of
        // the coefficient slabs must fall below rtol × column scale.
        let mut col_scale = vec![0.0_f64; k];
        for slab in &coeff_slabs {
            for (j, scale) in col_scale.iter_mut().enumerate() {
                for i in 0..n {
                    *scale = scale.max(slab[[i, j]].abs());
                }
            }
        }
        let tail_start = m - (m / 4).max(1);
        for slab in coeff_slabs.iter().skip(tail_start) {
            for (j, &scale) in col_scale.iter().enumerate() {
                let bound = PSI_GRAM_CERT_RTOL * scale.max(1e-300);
                for i in 0..n {
                    if slab[[i, j]].abs() > bound {
                        return BuildOutcome::TailNotCertified;
                    }
                }
            }
        }
        // One-time n-pass products on the NORMALIZED slabs:
        //   G̃[d][e] = Y_dᵀ W Y_e, c̃[d] = Y_dᵀ W z (the amplitude `a(ψ)` is
        //   reattached analytically at assembly, #1216).
        let mut weighted: Vec<Array2<f64>> = Vec::with_capacity(m);
        for slab in &coeff_slabs {
            let mut ws = slab.clone();
            for (mut row, &w) in ws.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|v| v * w);
            }
            weighted.push(ws);
        }
        let mut wz = Array1::<f64>::zeros(z.len());
        let mut zt_w_z = 0.0_f64;
        for ((slot, &w), &zv) in wz.iter_mut().zip(weights.iter()).zip(z.iter()) {
            *slot = w * zv;
            zt_w_z += w * zv * zv;
        }
        let mut gram: Vec<Array2<f64>> = Vec::with_capacity(m * m);
        let mut rhs = Vec::with_capacity(m);
        for d in 0..m {
            for e in 0..m {
                if e < d {
                    // Symmetry: G̃[d][e] = G̃[e][d]ᵀ — reuse, don't recompute.
                    let g: Array2<f64> = gram[e * m + d].t().to_owned();
                    gram.push(g);
                } else {
                    gram.push(coeff_slabs[d].t().dot(&weighted[e]));
                }
            }
            rhs.push(coeff_slabs[d].t().dot(&wz));
        }
        BuildOutcome::Candidate(Self {
            psi_lo,
            psi_hi,
            // Provisional: `build` narrows these to the certified interior after
            // the value spot-check passes (`certify_gradient_window`).
            grad_psi_lo: psi_lo,
            grad_psi_hi: psi_hi,
            n_coeff: m,
            k,
            gram,
            rhs,
            zt_w_z,
            amp,
        })
    }

    /// Off-node certification: the ASSEMBLED Gram must reproduce the exactly
    /// rebuilt Gram at deterministic interior ψ values.
    fn spot_check(
        &self,
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
    ) -> bool {
        for s in 0..PSI_GRAM_SPOT_POINTS {
            // Golden-ratio low-discrepancy interior points — never the nodes.
            let frac = ((s as f64 + 1.0) * 0.618_033_988_749_894_9).fract();
            let psi = self.psi_lo + frac * (self.psi_hi - self.psi_lo);
            let Ok(design) = eval_design(psi) else {
                return false;
            };
            let mut wd = design.clone();
            for (mut row, &w) in wd.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|v| v * w);
            }
            let exact = design.t().dot(&wd);
            let assembled = self.gram_at(psi);
            let scale = exact
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1e-300);
            for (a, b) in assembled.iter().zip(exact.iter()) {
                if (a - b).abs() > PSI_GRAM_SPOT_RTOL * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Locate the largest centered interior interval where the analytic
    /// derivative `dgram_dpsi` reproduces a central finite difference of the
    /// exactly rebuilt Gram to [`PSI_GRAM_GRAD_SPOT_RTOL`], and store it as the
    /// gradient sub-window. Scans inward symmetrically from both endpoints; the
    /// value lane (`gram_at`) is unaffected. One-time cost: a handful of extra
    /// exact design evals (each cheap under the radial profile).
    fn certify_gradient_window(
        &mut self,
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
    ) {
        let span = self.psi_hi - self.psi_lo;
        // 4th-order central stencil for the exact-derivative reference
        //   G'(ψ) ≈ [G(ψ−2h) − 8G(ψ−h) + 8G(ψ+h) − G(ψ+2h)] / (12h)
        // so the reference truncation is O(h⁴) — far below the tight rtol at a
        // moderate step, letting the certificate measure the reconstruction
        // error rather than the reference's own FD error.
        let h = (span * 1e-3).max(1e-6);
        let exact_dgram = |psi: f64,
                           eval: &mut dyn FnMut(f64) -> Result<Array2<f64>, String>|
         -> Option<Array2<f64>> {
            let weighted_gram = |p: f64,
                                 eval: &mut dyn FnMut(f64) -> Result<Array2<f64>, String>|
             -> Option<Array2<f64>> {
                let design = eval(p).ok()?;
                let mut wd = design.clone();
                for (mut row, &w) in wd.outer_iter_mut().zip(weights.iter()) {
                    row.mapv_inplace(|v| v * w);
                }
                Some(design.t().dot(&wd))
            };
            let g_m2 = weighted_gram(psi - 2.0 * h, eval)?;
            let g_m1 = weighted_gram(psi - h, eval)?;
            let g_p1 = weighted_gram(psi + h, eval)?;
            let g_p2 = weighted_gram(psi + 2.0 * h, eval)?;
            Some((g_m2 - 8.0 * &g_m1 + 8.0 * &g_p1 - g_p2) / (12.0 * h))
        };
        // True when the analytic derivative matches the exact FD at `psi`.
        let certifies = |me: &Self,
                         psi: f64,
                         eval: &mut dyn FnMut(f64) -> Result<Array2<f64>, String>|
         -> bool {
            // Keep the 4th-order FD stencil (ψ ± 2h) strictly inside the window.
            if psi - 2.0 * h <= me.psi_lo || psi + 2.0 * h >= me.psi_hi {
                return false;
            }
            let Some(exact) = exact_dgram(psi, eval) else {
                return false;
            };
            let analytic = me.dgram_dpsi(psi);
            let scale = exact
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1e-300);
            analytic
                .iter()
                .zip(exact.iter())
                .all(|(a, b)| (a - b).abs() <= PSI_GRAM_GRAD_SPOT_RTOL * scale)
        };
        // Scan inward from each endpoint to the first certified point.
        let n = PSI_GRAM_GRAD_SCAN_POINTS;
        let mut lo = self.psi_hi;
        let mut hi = self.psi_lo;
        let mut found = false;
        for i in 0..=n {
            let psi = self.psi_lo + span * (i as f64) / (n as f64);
            if certifies(self, psi, eval_design) {
                lo = psi;
                found = true;
                break;
            }
        }
        for i in (0..=n).rev() {
            let psi = self.psi_lo + span * (i as f64) / (n as f64);
            if certifies(self, psi, eval_design) {
                hi = psi;
                break;
            }
        }
        if found && hi > lo {
            self.grad_psi_lo = lo;
            self.grad_psi_hi = hi;
        } else {
            // No certified interior: disable the gradient lane entirely
            // (empty sub-window) — callers keep the exact slab gradient.
            self.grad_psi_lo = f64::NAN;
            self.grad_psi_hi = f64::NAN;
        }
    }

    /// True when `psi` lies inside the certified window.
    pub fn contains(&self, psi: f64) -> bool {
        psi.is_finite() && psi >= self.psi_lo && psi <= self.psi_hi
    }

    /// True when `psi` lies inside the certified gradient sub-window — the
    /// region where the analytic ψ-derivative is bit-tight against the exact
    /// design derivative (#1033b). Outside it (near the window edges) callers
    /// must keep the exact slab gradient.
    pub fn contains_for_gradient(&self, psi: f64) -> bool {
        psi.is_finite()
            && self.grad_psi_lo.is_finite()
            && self.grad_psi_hi.is_finite()
            && psi >= self.grad_psi_lo
            && psi <= self.grad_psi_hi
    }

    fn mapped(&self, psi: f64) -> f64 {
        (2.0 * psi - (self.psi_lo + self.psi_hi)) / (self.psi_hi - self.psi_lo)
    }

    /// Normalized-design Gram `N(ψ) = Σ T_d T_e (Y_dᵀWY_e)` and its first two
    /// mapped-coordinate `ψ̃`-derivatives, assembled n-free in O(D²k²).
    /// Returns `(N, dN/dψ̃, d²N/dψ̃²)`. The physical Gram and its ψ-derivatives
    /// reattach the amplitude `a(ψ)` via the product rule in [`Self::gram_at`]
    /// & friends.
    fn norm_gram_jets(&self, x: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let t = cheb_t(x, self.n_coeff);
        let tp = cheb_t_prime(x, self.n_coeff);
        let tpp = cheb_t_double_prime(x, self.n_coeff);
        let mut n0 = Array2::<f64>::zeros((self.k, self.k));
        let mut n1 = Array2::<f64>::zeros((self.k, self.k));
        let mut n2 = Array2::<f64>::zeros((self.k, self.k));
        for d in 0..self.n_coeff {
            for e in 0..self.n_coeff {
                let g = &self.gram[d * self.n_coeff + e];
                n0.scaled_add(t[d] * t[e], g);
                n1.scaled_add(tp[d] * t[e] + t[d] * tp[e], g);
                n2.scaled_add(tpp[d] * t[e] + 2.0 * tp[d] * tp[e] + t[d] * tpp[e], g);
            }
        }
        (n0, n1, n2)
    }

    /// Normalized-design RHS `M(ψ) = Σ T_d (Y_dᵀWz)` and its first two mapped
    /// `ψ̃`-derivatives, n-free in O(Dk). Returns `(M, dM/dψ̃, d²M/dψ̃²)`.
    fn norm_rhs_jets(&self, x: f64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let t = cheb_t(x, self.n_coeff);
        let tp = cheb_t_prime(x, self.n_coeff);
        let tpp = cheb_t_double_prime(x, self.n_coeff);
        let mut m0 = Array1::<f64>::zeros(self.k);
        let mut m1 = Array1::<f64>::zeros(self.k);
        let mut m2 = Array1::<f64>::zeros(self.k);
        for d in 0..self.n_coeff {
            m0.scaled_add(t[d], &self.rhs[d]);
            m1.scaled_add(tp[d], &self.rhs[d]);
            m2.scaled_add(tpp[d], &self.rhs[d]);
        }
        (m0, m1, m2)
    }

    /// `XᵀWX(ψ) = a(ψ)²·N(ψ)` assembled n-free in O(D²k²).
    pub fn gram_at(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let a = self.amp.value(x);
        let (mut out, _, _) = self.norm_gram_jets(x);
        out.mapv_inplace(|v| v * a * a);
        out
    }

    /// `XᵀWz(ψ) = a(ψ)·M(ψ)` assembled n-free in O(Dk).
    pub fn rhs_at(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let a = self.amp.value(x);
        let (mut out, _, _) = self.norm_rhs_jets(x);
        out.mapv_inplace(|v| v * a);
        out
    }

    /// Exact `∂(XᵀWX)/∂ψ` from the SAME representation as the value — the
    /// structural cure for the objective↔gradient desync class on this
    /// channel. n-free, O(D²k²).
    ///
    /// `G(ψ) = a²·N`, so by the product rule (all in the mapped `ψ̃`, then chain
    /// rule `dψ̃/dψ`):  `dG/dψ̃ = 2 a a' N + a² N'`.
    pub fn dgram_dpsi(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let a = self.amp.value(x);
        let ap = self.amp.deriv(x);
        let (n0, n1, _) = self.norm_gram_jets(x);
        let mut out = n0;
        out.mapv_inplace(|v| v * (2.0 * a * ap));
        out.scaled_add(a * a, &n1);
        out.mapv_inplace(|v| v * dx_dpsi);
        out
    }

    /// Exact `∂(XᵀWz)/∂ψ`, n-free. `r(ψ) = a·M`, `dr/dψ̃ = a' M + a M'`.
    pub fn drhs_dpsi(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let a = self.amp.value(x);
        let ap = self.amp.deriv(x);
        let (m0, m1, _) = self.norm_rhs_jets(x);
        let mut out = m0;
        out.mapv_inplace(|v| v * ap);
        out.scaled_add(a, &m1);
        out.mapv_inplace(|v| v * dx_dpsi);
        out
    }

    /// Exact `∂²(XᵀWX)/∂ψ²` from the SAME representation as the value/gradient —
    /// the n-free curvature that lets the outer Newton/ARC step read the τ-τ
    /// Hessian's design-moving block without re-streaming an O(n) slab Gram
    /// (#1033, Gaussian-identity single-ψ Hessian channel). O(D²k²).
    ///
    /// `XᵀWX(ψ) = Σ_{d,e} T_d(x) T_e(x) G_{de}` with `x = mapped(ψ)`, so by the
    /// product rule in `x` (then chain rule `(dx/dψ)²`):
    ///   `∂²/∂x² = T_d″ T_e + 2 T_d′ T_e′ + T_d T_e″`.
    pub fn d2gram_dpsi2(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let dx_dpsi_sq = dx_dpsi * dx_dpsi;
        let a = self.amp.value(x);
        let ap = self.amp.deriv(x);
        let app = self.amp.deriv2(x);
        let (n0, n1, n2) = self.norm_gram_jets(x);
        // d²/dψ̃² [a²N] = (2a'² + 2a a'')N + 4 a a' N' + a² N''.
        let mut out = n0;
        out.mapv_inplace(|v| v * (2.0 * ap * ap + 2.0 * a * app));
        out.scaled_add(4.0 * a * ap, &n1);
        out.scaled_add(a * a, &n2);
        out.mapv_inplace(|v| v * dx_dpsi_sq);
        out
    }

    /// Exact `∂²(XᵀWz)/∂ψ²`, n-free. `r = a·M`, so
    /// `d²/dψ̃² [aM] = a''M + 2a'M' + aM''`, times `(dψ̃/dψ)²`.
    pub fn d2rhs_dpsi2(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let dx_dpsi_sq = dx_dpsi * dx_dpsi;
        let a = self.amp.value(x);
        let ap = self.amp.deriv(x);
        let app = self.amp.deriv2(x);
        let (m0, m1, m2) = self.norm_rhs_jets(x);
        let mut out = m0;
        out.mapv_inplace(|v| v * app);
        out.scaled_add(2.0 * ap, &m1);
        out.scaled_add(a, &m2);
        out.mapv_inplace(|v| v * dx_dpsi_sq);
        out
    }

    /// Assemble the Gaussian-identity sufficient-statistic cache at `psi`
    /// without touching a single data row — the bridge from this tensor into
    /// the inner PLS solver's fast path (#1033b → `GaussianFixedCache`).
    ///
    /// `(XᵀWX, XᵀWz, zᵀWz)` is everything the Gaussian penalized solve needs
    /// at any λ, so a ψ-trial that holds a certified tensor can hand the
    /// inner solver this cache instead of realizing the n×k design. The
    /// caller is responsible for `contains(psi)` (off-window trials fall back
    /// to the exact realizer path). Dense-path bridge only: the sparse
    /// scatter cache stays `None`.
    pub fn gaussian_fixed_cache_at(&self, psi: f64) -> crate::pirls::GaussianFixedCache {
        crate::pirls::GaussianFixedCache {
            xtwx_orig: self.gram_at(psi),
            xtwy_orig: self.rhs_at(psi),
            centered_weighted_y_sq: self.zt_w_z,
            xtwx_sparse_orig: None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic Matérn-shaped synthetic design: entries g(e^{u_ij + ψ}) with
    /// g(s) = (1 + s)·exp(−s) (the ν = 3/2 Matérn shape) plus a ψ-free power
    /// column — the exact structural mix of the production radial designs.
    fn synth_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let r = 0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (n as f64 * k as f64) * 3.0;
                if j == k - 1 {
                    // ψ-free polynomial block column.
                    x[[i, j]] = r * r * r;
                } else {
                    let s = r * psi.exp();
                    x[[i, j]] = (1.0 + s) * (-s).exp();
                }
            }
        }
        Ok(x)
    }

    fn exact_gram(psi: f64, n: usize, k: usize, w: &Array1<f64>) -> Array2<f64> {
        let design = synth_design(psi, n, k).unwrap();
        let mut wd = design.clone();
        for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
            row.mapv_inplace(|v| v * wi);
        }
        design.t().dot(&wd)
    }

    /// #1033b primitive gate: the certified tensor must reproduce the exact
    /// Gram/rhs at arbitrary in-window ψ to certification accuracy, and its
    /// analytic ψ-derivative must match central finite differences of the
    /// exact Gram — value and gradient from one representation.
    #[test]
    fn psi_gram_tensor_matches_exact_gram_and_fd_gradient() {
        let (n, k) = (160usize, 7usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
        let (psi_lo, psi_hi) = (-1.2_f64, 1.0_f64);

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        for &psi in &[-1.1, -0.63, 0.0, 0.41, 0.97] {
            assert!(tensor.contains(psi));
            let exact = exact_gram(psi, n, k, &w);
            let fast = tensor.gram_at(psi);
            let scale = exact.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            for (a, b) in fast.iter().zip(exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "gram mismatch at psi={psi}: fast={a}, exact={b}"
                );
            }
            // rhs check against the exact weighted cross-product.
            let design = synth_design(psi, n, k).unwrap();
            let mut wz = Array1::<f64>::zeros(n);
            for ((slot, &wi), &zi) in wz.iter_mut().zip(w.iter()).zip(z.iter()) {
                *slot = wi * zi;
            }
            let exact_rhs = design.t().dot(&wz);
            let fast_rhs = tensor.rhs_at(psi);
            let rscale = exact_rhs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            for (a, b) in fast_rhs.iter().zip(exact_rhs.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * rscale,
                    "rhs mismatch at psi={psi}: fast={a}, exact={b}"
                );
            }
            // Analytic ψ-gradient vs central FD of the EXACT gram.
            let h = 1e-5;
            let g_plus = exact_gram(psi + h, n, k, &w);
            let g_minus = exact_gram(psi - h, n, k, &w);
            let dg = tensor.dgram_dpsi(psi);
            let dscale = dg.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-12);
            for ((a, p), m_) in dg.iter().zip(g_plus.iter()).zip(g_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-5 * dscale,
                    "dgram/dpsi mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
        }
        // Outside the window the caller must fall back to the exact path.
        assert!(!tensor.contains(psi_hi + 0.5));
        assert!(!tensor.contains(psi_lo - 0.5));

        // Bridge gate (#1033b → GaussianFixedCache): the n-free cache must
        // reproduce the exactly streamed sufficient statistics, and the
        // ridge-penalized solves through both must agree — the inner PLS
        // consumes nothing else, so this certifies the trial-loop handoff.
        for &psi in &[-0.9, 0.2, 0.8] {
            let cache = tensor.gaussian_fixed_cache_at(psi);
            let design = synth_design(psi, n, k).unwrap();
            let mut wd = design.clone();
            for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            let exact_gram = design.t().dot(&wd);
            let exact_rhs = wd.t().dot(&z);
            let exact_ztwz: f64 = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi * zi).sum();
            assert!(
                (cache.centered_weighted_y_sq - exact_ztwz).abs()
                    <= 1e-12 * exact_ztwz.abs().max(1e-300),
                "zᵀWz drift: cache={}, exact={exact_ztwz}",
                cache.centered_weighted_y_sq
            );
            // Ridge-penalized solve agreement: (G + I)β = r on both sides.
            let solve = |g: &Array2<f64>, r: &Array1<f64>| -> Array1<f64> {
                let mut a = g.clone();
                for i in 0..k {
                    a[[i, i]] += 1.0;
                }
                // Small dense Gauss elimination (k = 7 in this test).
                let mut aug = Array2::<f64>::zeros((k, k + 1));
                aug.slice_mut(ndarray::s![.., ..k]).assign(&a);
                aug.slice_mut(ndarray::s![.., k]).assign(r);
                for col in 0..k {
                    let piv = (col..k)
                        .max_by(|&p, &q| aug[[p, col]].abs().total_cmp(&aug[[q, col]].abs()))
                        .unwrap();
                    if piv != col {
                        for j in 0..=k {
                            let tmp = aug[[col, j]];
                            aug[[col, j]] = aug[[piv, j]];
                            aug[[piv, j]] = tmp;
                        }
                    }
                    let p = aug[[col, col]];
                    for row in 0..k {
                        if row == col {
                            continue;
                        }
                        let f = aug[[row, col]] / p;
                        for j in col..=k {
                            aug[[row, j]] -= f * aug[[col, j]];
                        }
                    }
                }
                Array1::from_iter((0..k).map(|i| aug[[i, k]] / aug[[i, i]]))
            };
            let beta_fast = solve(&cache.xtwx_orig, &cache.xtwy_orig);
            let beta_exact = solve(&exact_gram, &exact_rhs);
            let bscale = beta_exact
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()))
                .max(1e-300);
            for (a, b) in beta_fast.iter().zip(beta_exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-8 * bscale,
                    "penalized solve drift at psi={psi}: fast={a}, exact={b}"
                );
            }
        }
    }

    /// #1033 n-independence invariant (structural, build-free, bit-tight):
    /// after the one-time `build` n-pass, EVERY per-trial accessor the certified
    /// κ/ψ outer-loop hot path consumes — the value `(gram_at, rhs_at)`, the
    /// gradient `(dgram_dpsi, drhs_dpsi)`, the Hessian-channel curvature
    /// `(d2gram_dpsi2, d2rhs_dpsi2)`, and the inner-solver bridge
    /// `gaussian_fixed_cache_at` — must touch ZERO data rows. We prove this by
    /// instrumenting the `eval_design` closure with an invocation counter (the
    /// closure is the ONLY route to the n×k design): the counter advances during
    /// `build` (the certified node ladder + spot/gradient-window checks) and must
    /// then stay FROZEN across an entire ψ-trial sweep. This is the
    /// "no surface rebuild / no n×k re-realization on a cache-hit trial"
    /// invariant the outer-loop seam (`SpatialJointContext::eval_full`,
    /// `skip_design_realization`) relies on — asserted here at the tensor source
    /// of truth, independent of whether the full κ-fit converges or of any
    /// wall-clock measurement.
    #[test]
    fn psi_gram_tensor_trial_accessors_touch_no_data_rows() {
        use std::cell::Cell;

        let (n, k) = (256usize, 6usize);
        let w = Array1::from_iter((0..n).map(|i| 0.8 + 0.4 * ((i % 4) as f64) / 3.0));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.21).sin() + 0.2));
        let (psi_lo, psi_hi) = (-1.1_f64, 0.95_f64);

        // The closure is the SOLE path to the n×k design; count every call.
        let calls = Cell::new(0usize);
        let tensor = PsiGramTensor::build(
            |psi| {
                calls.set(calls.get() + 1);
                synth_design(psi, n, k)
            },
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        // The one-time build necessarily streamed the design at the Chebyshev
        // nodes (plus off-node spot / gradient-window checks). Freeze the count.
        let build_calls = calls.get();
        assert!(
            build_calls > 0,
            "build must have streamed the design at least once (sanity)"
        );

        // A dense ψ-trial sweep strictly inside the certified window. Every
        // accessor below is what a per-trial outer-loop eval consumes.
        let m = 64usize;
        let lo = psi_lo + 0.05;
        let hi = psi_hi - 0.05;
        for i in 0..m {
            let psi = lo + (hi - lo) * (i as f64) / (m as f64 - 1.0);
            assert!(tensor.contains(psi));
            // Value lane.
            let _g = tensor.gram_at(psi);
            let _r = tensor.rhs_at(psi);
            // Gradient lane.
            let _dg = tensor.dgram_dpsi(psi);
            let _dr = tensor.drhs_dpsi(psi);
            // Hessian-channel curvature.
            let _d2g = tensor.d2gram_dpsi2(psi);
            let _d2r = tensor.d2rhs_dpsi2(psi);
            // Inner-solver bridge (the GaussianFixedCache the PLS fast path reads).
            let _cache = tensor.gaussian_fixed_cache_at(psi);
        }

        assert_eq!(
            calls.get(),
            build_calls,
            "n-independence VIOLATED: a per-trial accessor re-streamed the n×k \
             design ({} extra eval_design calls across {m} ψ-trials). The certified \
             κ/ψ outer loop must serve value + gradient + Hessian curvature + the \
             inner-solver cache from k-space sufficient statistics only, with NO \
             per-trial n-row work.",
            calls.get() - build_calls
        );
    }

    /// #1033 Hessian-channel primitive gate: the n-free second ψ-derivatives
    /// `d2gram_dpsi2` / `d2rhs_dpsi2` must match central FD of the analytic FIRST
    /// derivatives (`dgram_dpsi` / `drhs_dpsi`) — the curvature the outer Newton
    /// /ARC step reads when the Gaussian Hessian channel is served from the
    /// tensor instead of a re-streamed O(n) slab. Differencing the analytic first
    /// derivative (not the exact gram) keeps this a pure check of the
    /// second-derivative recurrence, isolated from the build's value-lane tol.
    #[test]
    fn psi_gram_tensor_second_derivative_matches_fd_of_gradient() {
        let (n, k) = (160usize, 7usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
        let (psi_lo, psi_hi) = (-1.2_f64, 1.0_f64);

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        let h = 1e-5;
        for &psi in &[-1.0, -0.5, 0.0, 0.4, 0.9] {
            // ∂²G/∂ψ² vs central FD of the analytic ∂G/∂ψ.
            let dg_plus = tensor.dgram_dpsi(psi + h);
            let dg_minus = tensor.dgram_dpsi(psi - h);
            let d2g = tensor.d2gram_dpsi2(psi);
            let gscale = d2g.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-9);
            for ((a, p), m_) in d2g.iter().zip(dg_plus.iter()).zip(dg_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-4 * gscale,
                    "d2gram/dpsi2 mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
            // ∂²(XᵀWz)/∂ψ² vs central FD of the analytic ∂(XᵀWz)/∂ψ.
            let dr_plus = tensor.drhs_dpsi(psi + h);
            let dr_minus = tensor.drhs_dpsi(psi - h);
            let d2r = tensor.d2rhs_dpsi2(psi);
            let rscale = d2r.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-9);
            for ((a, p), m_) in d2r.iter().zip(dr_plus.iter()).zip(dr_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-4 * rscale,
                    "d2rhs/dpsi2 mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
        }
    }

    /// The penalized Gaussian profile deviance at a fixed ridge λ, assembled
    /// PURELY from the sufficient-statistic triple `(G, r, c) = (XᵀWX, XᵀWz, zᵀWz)`:
    ///
    /// ```text
    ///   β(λ) = (G + λS)⁻¹ r,   D(ψ;λ) = c − 2 βᵀr + βᵀ(G + λS)β = c − βᵀr
    /// ```
    ///
    /// (the second equality uses the normal equations `(G + λS)β = r`). This is
    /// EXACTLY the object the inner Gaussian PLS minimizes over β, and it is a
    /// pure function of `(G, r, c)` — n-free. Returns `(D, β)` so the caller can
    /// also probe the coefficient lane. `s_ridge` is the ridge penalty matrix.
    fn profile_deviance(
        g: &Array2<f64>,
        r: &Array1<f64>,
        c: f64,
        s_ridge: &Array2<f64>,
        lambda: f64,
        k: usize,
    ) -> (f64, Array1<f64>) {
        // Dense (G + λS) β = r via partial-pivot Gauss elimination (small k).
        let mut a = g.clone();
        a.scaled_add(lambda, s_ridge);
        let mut aug = Array2::<f64>::zeros((k, k + 1));
        aug.slice_mut(ndarray::s![.., ..k]).assign(&a);
        aug.slice_mut(ndarray::s![.., k]).assign(r);
        for col in 0..k {
            let piv = (col..k)
                .max_by(|&p, &q| aug[[p, col]].abs().total_cmp(&aug[[q, col]].abs()))
                .unwrap();
            if piv != col {
                for j in 0..=k {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[piv, j]];
                    aug[[piv, j]] = tmp;
                }
            }
            let p = aug[[col, col]];
            for row in 0..k {
                if row == col {
                    continue;
                }
                let f = aug[[row, col]] / p;
                for j in col..=k {
                    aug[[row, j]] -= f * aug[[col, j]];
                }
            }
        }
        let beta = Array1::from_iter((0..k).map(|i| aug[[i, k]] / aug[[i, i]]));
        let deviance = c - beta.dot(r);
        (deviance, beta)
    }

    /// #1033 bit-tight Hessian + κ-optimum gate. The fast path's promise is not
    /// merely that the Gram VALUE matches at sampled ψ — it is that the WHOLE
    /// outer κ search (objective, its ψ-curvature, and therefore the located
    /// optimum) is reproduced by the n-free sufficient-statistic representation
    /// to machine precision. This harness certifies exactly that:
    ///
    ///   1. **Objective**: the penalized profile deviance `D(ψ)` assembled from
    ///      the tensor's `(gram_at, rhs_at, zᵀWz)` matches the exactly streamed
    ///      `XᵀWX/XᵀWz/zᵀWz` deviance bit-tight at every ψ on a fine grid.
    ///   2. **Curvature (Hessian)**: the second ψ-derivative `D''(ψ)` of the
    ///      fast-path objective matches the second ψ-derivative of the EXACT
    ///      objective (central FD of the streamed deviance) — the curvature the
    ///      outer Newton step reads must be the true curvature, not an
    ///      approximation that drifts off the value (the objective↔gradient
    ///      desync class, now extended to the second order).
    ///   3. **κ-optimum**: the argmin of `D(ψ)` over the grid is IDENTICAL
    ///      between the two assemblies — the fast path lands on the same κ as the
    ///      exact streamed search, to the grid resolution AND bit-tight in the
    ///      objective value at that node.
    #[test]
    fn psi_gram_tensor_bit_tight_hessian_and_kappa_optimum() {
        let (n, k) = (200usize, 6usize);
        // Heterogeneous weights + a response with genuine ψ-dependent curvature
        // so the deviance has a non-degenerate interior minimum in ψ.
        let w = Array1::from_iter((0..n).map(|i| 0.7 + 0.6 * (((i * 7) % 5) as f64) / 4.0));
        let z = Array1::from_iter((0..n).map(|i| {
            let t = (i as f64) / (n as f64 - 1.0);
            (3.0 * t).sin() + 0.3 * (7.0 * t).cos()
        }));
        let (psi_lo, psi_hi) = (-1.0_f64, 0.9_f64);
        // Fixed ridge λ over the search — the κ optimizer profiles ψ at fixed
        // smoothing here; identity-S ridge keeps the profile well-posed.
        let s_ridge = Array2::<f64>::eye(k);
        let lambda = 0.5_f64;

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        let exact_ztwz: f64 = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi * zi).sum();

        // Exact streamed deviance at arbitrary ψ — the ground truth the n-free
        // path must reproduce.
        let exact_deviance = |psi: f64| -> f64 {
            let design = synth_design(psi, n, k).unwrap();
            let mut wd = design.clone();
            for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            let g = design.t().dot(&wd);
            let r = wd.t().dot(&z);
            profile_deviance(&g, &r, exact_ztwz, &s_ridge, lambda, k).0
        };

        // Fast n-free deviance from the certified tensor.
        let fast_deviance = |psi: f64| -> f64 {
            let g = tensor.gram_at(psi);
            let r = tensor.rhs_at(psi);
            profile_deviance(&g, &r, exact_ztwz, &s_ridge, lambda, k).0
        };

        // Dense grid strictly inside the certified window (away from the edges,
        // where the build's value lane is still certified but we want a clean
        // central-FD second derivative to exist on both sides).
        let m = 81usize;
        let lo = psi_lo + 0.06;
        let hi = psi_hi - 0.06;
        let grid: Vec<f64> = (0..m)
            .map(|i| lo + (hi - lo) * (i as f64) / (m as f64 - 1.0))
            .collect();

        // (1) Objective bit-tight across the whole grid; track argmin on both.
        let mut worst_value_rel = 0.0_f64;
        let (mut fast_argmin, mut fast_min) = (f64::NAN, f64::INFINITY);
        let (mut exact_argmin, mut exact_min) = (f64::NAN, f64::INFINITY);
        for &psi in &grid {
            let de = exact_deviance(psi);
            let df = fast_deviance(psi);
            let rel = (de - df).abs() / de.abs().max(1e-300);
            worst_value_rel = worst_value_rel.max(rel);
            if df < fast_min {
                fast_min = df;
                fast_argmin = psi;
            }
            if de < exact_min {
                exact_min = de;
                exact_argmin = psi;
            }
        }
        assert!(
            worst_value_rel <= 1e-9,
            "penalized profile deviance: fast n-free assembly diverged from exact \
             streamed by rel {worst_value_rel:.3e} (> 1e-9) somewhere on the ψ grid"
        );

        // (3) κ-optimum: identical grid node AND bit-tight value there. The
        // argmin must be a true interior minimum (not a window edge) for this to
        // certify the OUTER search rather than a boundary artifact.
        assert_eq!(
            fast_argmin.to_bits(),
            exact_argmin.to_bits(),
            "κ-optimum mismatch: fast argmin ψ={fast_argmin}, exact argmin ψ={exact_argmin} \
             — the n-free objective located a different optimum"
        );
        assert!(
            fast_argmin > lo + 1e-9 && fast_argmin < hi - 1e-9,
            "κ-optimum landed on the grid edge ψ={fast_argmin}; the fixture must have \
             an INTERIOR minimum for this to test the outer search, not a boundary"
        );
        let opt_rel = (exact_min - fast_min).abs() / exact_min.abs().max(1e-300);
        assert!(
            opt_rel <= 1e-9,
            "κ-optimum objective value drift at ψ={fast_argmin}: fast={fast_min}, \
             exact={exact_min}, rel={opt_rel:.3e}"
        );

        // (2) Gradient + curvature from the tensor's ANALYTIC ψ-derivatives.
        //
        // Differencing two objectives that agree only to ~1e-9 in VALUE cannot
        // certify their curvature: the central second difference divides by h²,
        // so the ~1e-9 value gap (which is NOT common-mode — they are different
        // assemblies) is amplified by 1/h² and swamps any real curvature signal.
        // The principled bit-tight curvature check uses the tensor's OWN analytic
        // ψ-derivatives `dgram_dpsi`/`drhs_dpsi`: the envelope gradient of the
        // profile deviance `D(ψ) = c − rᵀA⁻¹r`, `A = G + λS`, is
        //
        //   D'(ψ) = −2 βᵀ(∂r/∂ψ) + βᵀ(∂G/∂ψ)β,   β = A⁻¹r,
        //
        // assembled n-free from `(dgram_dpsi, drhs_dpsi)`. We certify this
        // analytic gradient against a central FD of the EXACT streamed objective
        // (first order ⇒ only 1/h amplification, so the ~1e-9 value agreement is
        // not destroyed), and certify the curvature by central-differencing the
        // ANALYTIC gradient (again 1/h, not 1/h²). This is the same one-
        // representation value↔gradient↔curvature consistency the production fast
        // path relies on for the outer Newton step.
        let solve_a = |g: &Array2<f64>, r: &Array1<f64>| -> Array1<f64> {
            profile_deviance(g, r, exact_ztwz, &s_ridge, lambda, k).1
        };
        // Analytic n-free ψ-gradient of the penalized profile deviance, valid on
        // the certified gradient sub-window where `dgram_dpsi` is bit-tight.
        let analytic_grad = |psi: f64| -> f64 {
            let g = tensor.gram_at(psi);
            let r = tensor.rhs_at(psi);
            let beta = solve_a(&g, &r);
            let dg = tensor.dgram_dpsi(psi);
            let dr = tensor.drhs_dpsi(psi);
            -2.0 * beta.dot(&dr) + beta.dot(&dg.dot(&beta))
        };

        // Two finite-difference steps, each near the optimum of its own
        // truncation/rounding trade-off:
        //   * `h_grad = 1e-6` for the FIRST derivative (central FD ⇒ O(h²)
        //     truncation, O(ε/h) rounding ⇒ optimum near 1e-5..1e-6);
        //   * `h_curv = 2e-4` for the curvature. A SECOND difference divides by
        //     h², so its rounding floor is O(ε·|D|/h²): at h=1e-6 that is
        //     ~1e-16/1e-12 = 1e-4 of |D|, comparable to the curvature itself —
        //     useless. h≈2e-4 puts the rounding floor at ~1e-16/4e-8 ≈ 2.5e-9·|D|
        //     and the O(h²·D⁗) truncation around the same scale, so the second
        //     difference is meaningful. The analytic-gradient curvature is
        //     differenced at the SAME h_curv so the two carry the same
        //     truncation order and the comparison is apples-to-apples.
        let h_grad = 1e-6_f64;
        let h_curv = 2e-4_f64;
        let mut worst_grad_rel = 0.0_f64;
        let mut worst_hess_rel = 0.0_f64;
        let mut tested = 0usize;
        for &psi in &grid {
            // The exact-objective curvature stencil reaches ±2·h_curv; require the
            // whole stencil to stay inside the certified gradient sub-window so the
            // analytic-gradient differences are all bit-tight.
            if !tensor.contains_for_gradient(psi - 2.0 * h_curv)
                || !tensor.contains_for_gradient(psi + 2.0 * h_curv)
            {
                continue;
            }
            tested += 1;
            // Analytic gradient vs central FD of the EXACT streamed objective.
            let exact_g1 =
                (exact_deviance(psi + h_grad) - exact_deviance(psi - h_grad)) / (2.0 * h_grad);
            let ag = analytic_grad(psi);
            let gscale = exact_g1.abs().max(1e-6);
            worst_grad_rel = worst_grad_rel.max((exact_g1 - ag).abs() / gscale);
            // Curvature: central FD of the ANALYTIC gradient (n-free) vs central
            // second difference of the EXACT objective, both at h_curv.
            let analytic_h2 =
                (analytic_grad(psi + h_curv) - analytic_grad(psi - h_curv)) / (2.0 * h_curv);
            let exact_h2 = (exact_deviance(psi + h_curv) - 2.0 * exact_deviance(psi)
                + exact_deviance(psi - h_curv))
                / (h_curv * h_curv);
            let hscale = exact_h2.abs().max(1e-3);
            worst_hess_rel = worst_hess_rel.max((analytic_h2 - exact_h2).abs() / hscale);
        }
        assert!(
            tested > 0,
            "no ψ on the grid lay inside the certified gradient sub-window"
        );
        assert!(
            worst_grad_rel <= 1e-5,
            "ψ-gradient mismatch: the tensor's analytic n-free objective gradient diverged \
             from the exact streamed objective by rel {worst_grad_rel:.3e} (> 1e-5)"
        );
        // The curvature compares an analytic-gradient central difference against
        // an exact-objective second difference; the residual O(h²) truncation +
        // O(ε/h²) rounding floor at h_curv=2e-4 sets a realistic bit-tight bar of
        // ~1e-3 relative (any larger gap is a genuine curvature divergence, not FD
        // noise — the value/gradient lanes already certify the objective itself to
        // ~1e-9/1e-5).
        assert!(
            worst_hess_rel <= 1e-3,
            "ψ-curvature (Hessian) mismatch: fast n-free objective curvature diverged \
             from the exact streamed objective by rel {worst_hess_rel:.3e} (> 1e-3) — \
             the outer Newton step would read a different curvature than the truth"
        );

        eprintln!(
            "[psi-gram-bittight] n={n} k={k} grid={m} grad-tested={tested}  \
             worst |ΔD|/D={worst_value_rel:.2e}  worst |ΔD'|/D'={worst_grad_rel:.2e}  \
             worst |ΔD''|/D''={worst_hess_rel:.2e}  κ-opt ψ={fast_argmin:.6} (interior, bit-identical)"
        );
    }

    /// Certification negative: a NON-analytic (kinked) design must refuse to
    /// certify rather than silently approximate.
    #[test]
    fn psi_gram_tensor_refuses_non_analytic_design() {
        let (n, k) = (40usize, 3usize);
        let w = Array1::from_elem(n, 1.0);
        let z = Array1::from_elem(n, 0.5);
        let tensor = PsiGramTensor::build(
            |psi| {
                let mut x = Array2::<f64>::zeros((n, k));
                for i in 0..n {
                    for j in 0..k {
                        // |ψ| kink at 0 inside the window: not analytic.
                        x[[i, j]] = psi.abs() + (i + j) as f64 / (n + k) as f64;
                    }
                }
                Ok(x)
            },
            w.view(),
            z.view(),
            -1.0,
            1.0,
        );
        assert!(
            tensor.is_none(),
            "kinked design must fail the tail-decay/spot-check certificates"
        );
    }

}
