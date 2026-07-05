//! Cross-atom parse conditioning — the BETWEEN-atom identifiability certificate.
//!
//! [`crate::identifiability`]'s residual-gauge certificate is the *within*-atom
//! half: for each fitted atom it asks which gauge subgroup the data + isometry
//! penalty pin. This module is the missing *between*-atom half — the certificate
//! the evidence provably CANNOT provide.
//!
//! # Why the evidence is blind to this (the sign proposition)
//!
//! At a parse `z = Σ_k a_k g_k(t_k)`, the coordinate/amplitude block of the
//! Hessian contains `J_SᵀJ_S/σ²` with `J_S = [K_{k₁} … K_{k_s}]`,
//! `K_k = [g_k | a_k ∂g_k]` (value column ‖ tangent columns). Block-decomposing
//! the log-det,
//!
//! ```text
//! log det(J_SᵀJ_S) = Σ_k log det(K_kᵀK_k) + log det(B_SᵀB_S),
//! ```
//!
//! where `B_k = K_k (K_kᵀK_k)^{-1/2}` is the per-atom-whitened block and the
//! cross term `log det(B_SᵀB_S) = Σ log sin²`(inter-atom principal angles) `≤ 0`,
//! `→ −∞` as two atoms' tangent cones collide. A collision therefore *lowers*
//! `½log|H|`, hence *lowers* the outer criterion `V = loss + ½log|H| − occam`:
//! at fixed fit, Bayesian evidence strictly **prefers the unidentifiable parse**.
//! Correct Bayes, wrong interpretability. So identifiability needs a *certificate*
//! channel — measured, reported, never silently folded into the race — exactly
//! parallel to the engine's "topology measured, not latched" doctrine.
//!
//! # The certificate
//!
//! Whiten each block (removing the per-atom conditioning the evidence already
//! charges), stack, and take `μ_S = σ_min(B_S) ∈ [0, 1]`.
//!
//! **Theorem (parse stability).** If `μ_S ≥ μ₀` near the parse, a data
//! perturbation `Δz` moves the whitened parse by `‖Δθ‖_white ≤ ‖Δz‖/μ₀`
//! (pseudoinverse + inverse-function remainder). `1/μ_S` **is** the
//! superposition-interference amplification factor, now a number per row.
//!
//! Companion outputs: `whitened_excess = tr((B_SᵀB_S)^{-1}) − m` (scale-free
//! interference part of the parse variance, `0` iff the atoms are orthogonal)
//! and `attribution_risk = σ²·tr((J_SᵀJ_S)^{-1})` (exact expected squared
//! attribution error — per-feature confidence intervals fall out for free).
//!
//! # Closed forms (the validated anchors)
//!
//! Two atoms whose whitened tangents meet at principal angle θ give
//! `margin = √(1 − cos θ)` and `whitened_excess = 2cos²θ/(1 − cos²θ)`;
//! orthogonal ⇒ margin 1, excess 0; collision ⇒ margin → 0, risks diverge. The
//! overcomplete case `Σ(d_k+1) > p` is refused with the Terracini bound named
//! (local identifiability of the sparse manifold decomposition requires
//! `rank(J_S) = Σ(d_k+1) ≤ p`).

use std::collections::BTreeMap;

use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2};

use faer::Side;

/// One atom's contribution to a parse: its value `g_k(t)` and its
/// amplitude-scaled tangent block `a_k · ∂g_k`.
///
/// `value` is the length-`p` reconstruction the atom contributes on this row
/// (from the reconstruction cache); `tangent` is `(p, d_k)`, one column per
/// latent axis, each column `a_k · ∂g_k/∂t_axis` (from
/// `a · fill_decoded_derivative_row`). The stacked block is
/// `K_k = [value | tangent]`, of shape `(p, d_k + 1)`.
#[derive(Debug, Clone)]
pub struct ParseBlock {
    /// Which accepted atom this block belongs to.
    pub atom: usize,
    /// `g_k(t)` — the atom's reconstruction contribution, length `p`.
    pub value: Array1<f64>,
    /// `a_k · ∂g_k` — amplitude-scaled tangent columns, `(p, d_k)`.
    pub tangent: Array2<f64>,
}

impl ParseBlock {
    /// `d_k + 1`, the tangent-space dimension this atom contributes to the join.
    fn block_cols(&self) -> usize {
        self.tangent.ncols() + 1
    }

    /// Assemble `K_k = [value | tangent]`, shape `(p, d_k + 1)`.
    fn stacked(&self) -> Result<Array2<f64>, String> {
        let p = self.value.len();
        if self.tangent.nrows() != p && self.tangent.ncols() != 0 {
            return Err(format!(
                "ParseBlock(atom {}): tangent has {} rows but value has length {p}",
                self.atom,
                self.tangent.nrows()
            ));
        }
        let cols = self.block_cols();
        let mut k = Array2::<f64>::zeros((p, cols));
        for i in 0..p {
            k[[i, 0]] = self.value[i];
        }
        for c in 0..self.tangent.ncols() {
            for i in 0..p {
                k[[i, c + 1]] = self.tangent[[i, c]];
            }
        }
        Ok(k)
    }
}

/// The between-atom identifiability certificate for one sampled parse / clique.
#[derive(Debug, Clone)]
pub struct TerraciniCertificate {
    /// The atoms co-firing in this parse, in the order supplied.
    pub atoms: Vec<usize>,
    /// Ambient output dimension `p`.
    pub p: usize,
    /// Total whitened tangent dimension `m = Σ_k (d_k + 1)`.
    pub m: usize,
    /// `μ_S = σ_min(B_S) ∈ [0, 1]` — the whitened Terracini margin.
    pub margin: f64,
    /// `1/μ_S` — the superposition-interference amplification factor.
    pub amplification: f64,
    /// `log det(B_SᵀB_S) = Σ log sin²`(inter-atom principal angles) `≤ 0`. This
    /// is the cross term the outer evidence adds with the WRONG sign.
    pub cross_gram_logdet: f64,
    /// `tr((B_SᵀB_S)^{-1}) − m` — scale-free interference part of parse
    /// variance, `0` iff the whitened blocks are mutually orthogonal.
    pub whitened_excess: f64,
    /// `σ²·tr((J_SᵀJ_S)^{-1})` — exact expected squared attribution error.
    pub attribution_risk: f64,
}

/// Symmetric inverse square root `G^{-1/2}` of a small SPD Gram (ridge-lifted).
fn inverse_sqrt(gram: &Array2<f64>, ridge: f64) -> Result<Array2<f64>, String> {
    let n = gram.nrows();
    let mut g = gram.clone();
    if ridge > 0.0 {
        for i in 0..n {
            g[[i, i]] += ridge;
        }
    }
    let (w, v) = g
        .eigh(Side::Lower)
        .map_err(|e| format!("terracini: eigh for whitening failed: {e}"))?;
    // G^{-1/2} = V diag(1/√w) Vᵀ; guard non-positive eigenvalues (ridge should
    // make them all positive, but stay defensive against a zero column).
    let mut scaled = v.clone();
    for c in 0..n {
        let wc = w[c];
        if !(wc.is_finite() && wc > 0.0) {
            return Err(format!(
                "terracini: whitening Gram not positive-definite (eigenvalue {wc:.3e}); \
                 a per-atom block is rank-deficient — raise ridge"
            ));
        }
        let inv = 1.0 / wc.sqrt();
        for r in 0..n {
            scaled[[r, c]] *= inv;
        }
    }
    Ok(scaled.dot(&v.t()))
}

/// Build the between-atom parse certificate.
///
/// `noise_var = σ²` scales `attribution_risk`; `ridge` lifts every Gram
/// inversion for numerical stability (pass `0.0` for the exact closed-form
/// anchors). Returns `Err` when the parse is overcomplete (`m > p`), naming the
/// Terracini bound: a sparse manifold decomposition is locally identifiable only
/// when `rank(J_S) = Σ(d_k + 1) ≤ p`.
pub fn parse_certificate(
    blocks: &[ParseBlock],
    noise_var: f64,
    ridge: f64,
) -> Result<TerraciniCertificate, String> {
    if blocks.is_empty() {
        return Err("terracini: parse has no co-firing atoms".to_string());
    }
    let p = blocks[0].value.len();
    if p == 0 {
        return Err("terracini: ambient dimension p = 0".to_string());
    }
    for b in blocks {
        if b.value.len() != p {
            return Err(format!(
                "terracini: atom {} value length {} != p = {p}",
                b.atom,
                b.value.len()
            ));
        }
    }
    let m: usize = blocks.iter().map(ParseBlock::block_cols).sum();
    let atoms: Vec<usize> = blocks.iter().map(|b| b.atom).collect();
    if m > p {
        return Err(format!(
            "terracini: overcomplete parse — Σ(d_k+1) = {m} > p = {p}; the join of the \
             {} atoms exceeds the ambient dimension, so by the Terracini bound the sparse \
             manifold decomposition is NOT locally identifiable at this parse (rank(J_S) \
             cannot reach {m})",
            atoms.len()
        ));
    }

    // Assemble the un-whitened stack J_S and the per-atom-whitened stack B_S.
    let mut j_s = Array2::<f64>::zeros((p, m));
    let mut b_s = Array2::<f64>::zeros((p, m));
    let mut col = 0usize;
    for b in blocks {
        let k = b.stacked()?;
        let cols = k.ncols();
        let gram = k.t().dot(&k);
        let g_inv_sqrt = inverse_sqrt(&gram, ridge)?;
        let bk = k.dot(&g_inv_sqrt);
        for c in 0..cols {
            for i in 0..p {
                j_s[[i, col + c]] = k[[i, c]];
                b_s[[i, col + c]] = bk[[i, c]];
            }
        }
        col += cols;
    }

    // Whitened cross-Gram B_SᵀB_S (m×m): its spectrum is the whole certificate.
    let btb = b_s.t().dot(&b_s);
    let w_b = btb
        .eigh(Side::Lower)
        .map_err(|e| format!("terracini: eigh of whitened cross-Gram failed: {e}"))?
        .0;
    let mut min_eig = f64::INFINITY;
    let mut logdet = 0.0_f64;
    let mut trace_inv = 0.0_f64;
    for &lam in w_b.iter() {
        if lam < min_eig {
            min_eig = lam;
        }
        // Guard the log/inverse against a collapsed direction (collision).
        let lam_c = lam.max(0.0);
        logdet += lam_c.max(1.0e-300).ln();
        trace_inv += if lam_c > 1.0e-300 {
            1.0 / lam_c
        } else {
            f64::INFINITY
        };
    }
    let margin = min_eig.max(0.0).sqrt();
    let amplification = if margin > 0.0 {
        1.0 / margin
    } else {
        f64::INFINITY
    };
    let whitened_excess = trace_inv - m as f64;

    // Attribution risk σ²·tr((J_SᵀJ_S)^{-1}) on the un-whitened stack.
    let jtj = j_s.t().dot(&j_s);
    let w_j = jtj
        .eigh(Side::Lower)
        .map_err(|e| format!("terracini: eigh of J_SᵀJ_S failed: {e}"))?
        .0;
    let mut trace_inv_j = 0.0_f64;
    for &lam in w_j.iter() {
        let lam_c = lam + ridge;
        trace_inv_j += if lam_c > 1.0e-300 {
            1.0 / lam_c
        } else {
            f64::INFINITY
        };
    }
    let attribution_risk = noise_var * trace_inv_j;

    Ok(TerraciniCertificate {
        atoms,
        p,
        m,
        margin,
        amplification,
        cross_gram_logdet: logdet,
        whitened_excess,
        attribution_risk,
    })
}

/// Worst-case aggregate of one co-firing clique's sampled certificates.
#[derive(Debug, Clone)]
pub struct CliqueStat {
    /// The clique (sorted atom set).
    pub atoms: Vec<usize>,
    /// How many parses were recorded for this clique.
    pub n_samples: usize,
    /// Worst (smallest) sampled margin — the identifiability of this clique is
    /// only as good as its most-collided parse.
    pub min_margin: f64,
    /// Mean sampled margin.
    pub mean_margin: f64,
    /// Largest sampled amplification `1/μ_S`.
    pub max_amplification: f64,
    /// Largest sampled attribution risk.
    pub max_attribution_risk: f64,
    /// Most-negative sampled cross-Gram log-det (the deepest evidence pull).
    pub min_cross_gram_logdet: f64,
}

/// Accumulates parse certificates per co-firing clique (the sorted set of
/// atoms). `finish` returns the cliques worst-margin-first, so a birth-time
/// consumer sees the tangent-collision failures at the top.
#[derive(Debug, Clone, Default)]
pub struct CliqueAccumulator {
    map: BTreeMap<Vec<usize>, CliqueStat>,
}

impl CliqueAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one sampled certificate into its clique's running worst-case stats.
    pub fn record(&mut self, cert: &TerraciniCertificate) {
        let mut key = cert.atoms.clone();
        key.sort_unstable();
        let entry = self.map.entry(key.clone()).or_insert_with(|| CliqueStat {
            atoms: key,
            n_samples: 0,
            min_margin: f64::INFINITY,
            mean_margin: 0.0,
            max_amplification: 0.0,
            max_attribution_risk: 0.0,
            min_cross_gram_logdet: f64::INFINITY,
        });
        // `mean_margin` holds the running sum until `finish`.
        entry.mean_margin += cert.margin;
        entry.n_samples += 1;
        entry.min_margin = entry.min_margin.min(cert.margin);
        entry.max_amplification = entry.max_amplification.max(cert.amplification);
        entry.max_attribution_risk = entry.max_attribution_risk.max(cert.attribution_risk);
        entry.min_cross_gram_logdet = entry.min_cross_gram_logdet.min(cert.cross_gram_logdet);
    }

    /// Finalize: divide the margin sums into means and return worst-margin-first.
    pub fn finish(self) -> Vec<CliqueStat> {
        let mut out: Vec<CliqueStat> = self
            .map
            .into_values()
            .map(|mut s| {
                if s.n_samples > 0 {
                    s.mean_margin /= s.n_samples as f64;
                }
                s
            })
            .collect();
        out.sort_by(|a, b| {
            a.min_margin
                .partial_cmp(&b.min_margin)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// A block that is a single unit `value` vector with no tangent columns:
    /// `d_k + 1 = 1`, whitening reduces `K_k` to that unit vector, so `B_S`'s
    /// columns are exactly the supplied unit vectors and the certificate is the
    /// principal-angle closed form.
    fn unit_block(atom: usize, v: &[f64]) -> ParseBlock {
        let value = Array1::from_vec(v.to_vec());
        ParseBlock {
            atom,
            value,
            tangent: Array2::<f64>::zeros((v.len(), 0)),
        }
    }

    #[test]
    fn closed_form_angle_margin_and_excess() {
        // Two unit tangents in the (e0, e1) plane at angle θ. margin = √(1−cosθ),
        // excess = 2cos²θ/(1−cos²θ), to 1e-9.
        for &theta in &[0.1_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let c = theta.cos();
            let s = theta.sin();
            let blocks = vec![
                unit_block(0, &[1.0, 0.0, 0.0, 0.0]),
                unit_block(1, &[c, s, 0.0, 0.0]),
            ];
            let cert = parse_certificate(&blocks, 1.0, 0.0).unwrap();
            let want_margin = (1.0 - c).sqrt();
            let want_excess = 2.0 * c * c / (1.0 - c * c);
            assert!(
                (cert.margin - want_margin).abs() < 1e-9,
                "θ={theta}: margin {} vs {want_margin}",
                cert.margin
            );
            assert!(
                (cert.whitened_excess - want_excess).abs() < 1e-9,
                "θ={theta}: excess {} vs {want_excess}",
                cert.whitened_excess
            );
            // cross-Gram log-det = log((1+c)(1−c)) = log(1−c²) = Σ log sin² angle.
            let want_logdet = (1.0 - c * c).ln();
            assert!(
                (cert.cross_gram_logdet - want_logdet).abs() < 1e-9,
                "θ={theta}: logdet {} vs {want_logdet}",
                cert.cross_gram_logdet
            );
            // amplification = 1/margin.
            assert!((cert.amplification - 1.0 / want_margin).abs() < 1e-7);
        }
    }

    #[test]
    fn orthogonal_is_perfectly_conditioned() {
        let blocks = vec![
            unit_block(0, &[1.0, 0.0, 0.0]),
            unit_block(1, &[0.0, 1.0, 0.0]),
        ];
        let cert = parse_certificate(&blocks, 2.0, 0.0).unwrap();
        assert!((cert.margin - 1.0).abs() < 1e-12, "margin {}", cert.margin);
        assert!(cert.whitened_excess.abs() < 1e-12, "excess {}", cert.whitened_excess);
        assert!(cert.cross_gram_logdet.abs() < 1e-12);
        // J_S orthonormal ⇒ risk = σ²·tr(I) = σ²·m = 2·2 = 4.
        assert!((cert.attribution_risk - 4.0).abs() < 1e-9, "risk {}", cert.attribution_risk);
    }

    #[test]
    fn collision_diverges() {
        // Near-collinear ⇒ margin → 0, amplification and risks blow up.
        let eps: f64 = 1e-6;
        let blocks = vec![
            unit_block(0, &[1.0, 0.0]),
            unit_block(1, &[(eps).cos(), (eps).sin()]),
        ];
        let cert = parse_certificate(&blocks, 1.0, 0.0).unwrap();
        assert!(cert.margin < 1e-3, "margin should collapse, got {}", cert.margin);
        assert!(cert.amplification > 1e2, "amplification {}", cert.amplification);
        assert!(cert.whitened_excess > 1e5, "excess {}", cert.whitened_excess);
    }

    #[test]
    fn overcomplete_is_refused() {
        // Three unit blocks in p = 2 ⇒ m = 3 > 2: Terracini bound violated.
        let blocks = vec![
            unit_block(0, &[1.0, 0.0]),
            unit_block(1, &[0.0, 1.0]),
            unit_block(2, &[1.0, 1.0]),
        ];
        let err = parse_certificate(&blocks, 1.0, 0.0).unwrap_err();
        assert!(err.contains("overcomplete"), "expected Terracini refusal, got: {err}");
    }

    #[test]
    fn tangent_block_whitening_removes_per_atom_conditioning() {
        // A block with a value + a badly-scaled tangent: after whitening the
        // per-atom conditioning is gone, so a second orthogonal-plane atom still
        // sees margin 1.
        let mut tan = Array2::<f64>::zeros((4, 1));
        tan[[2, 0]] = 100.0; // huge tangent scale, orthogonal to everything else
        let a = ParseBlock {
            atom: 0,
            value: Array1::from_vec(vec![3.0, 0.0, 0.0, 0.0]), // unnormalized value
            tangent: tan,
        };
        let b = unit_block(1, &[0.0, 1.0, 0.0, 0.0]);
        let cert = parse_certificate(&[a, b], 1.0, 0.0).unwrap();
        assert_eq!(cert.m, 3);
        assert!(
            (cert.margin - 1.0).abs() < 1e-9,
            "orthogonal blocks after whitening must give margin 1, got {}",
            cert.margin
        );
    }

    #[test]
    fn clique_accumulator_orders_worst_first() {
        let mut acc = CliqueAccumulator::new();
        // clique {0,1}: two parses, one benign, one collided.
        let benign = parse_certificate(
            &[unit_block(0, &[1.0, 0.0]), unit_block(1, &[0.0, 1.0])],
            1.0,
            0.0,
        )
        .unwrap();
        let collided = parse_certificate(
            &[
                unit_block(0, &[1.0, 0.0]),
                unit_block(1, &[(0.05_f64).cos(), (0.05_f64).sin()]),
            ],
            1.0,
            0.0,
        )
        .unwrap();
        // clique {2,3}: well conditioned.
        let other = parse_certificate(
            &[unit_block(2, &[1.0, 0.0, 0.0]), unit_block(3, &[0.0, 1.0, 0.0])],
            1.0,
            0.0,
        )
        .unwrap();
        acc.record(&benign);
        acc.record(&collided);
        acc.record(&other);
        let stats = acc.finish();
        assert_eq!(stats.len(), 2);
        // {0,1} took the collided min-margin, so it must sort first.
        assert_eq!(stats[0].atoms, vec![0, 1]);
        assert_eq!(stats[0].n_samples, 2);
        assert!(stats[0].min_margin < stats[1].min_margin);
        assert!(stats[0].min_margin <= collided.margin + 1e-12);
    }
}
