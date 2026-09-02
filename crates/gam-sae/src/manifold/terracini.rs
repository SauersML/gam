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
//! `K_k = [g_k | a_k ∂g_k]` (value column ‖ tangent columns). Its log-det splits
//! EXACTLY into a per-atom channel and a cross channel,
//!
//! ```text
//! log det(J_SᵀJ_S) = Σ_k log det(K_kᵀK_k) + log det(B_SᵀB_S),
//!                    └── per-atom volumes ─┘  └── cross term ≤ 0, → −∞ at collision ──┘
//! ```
//!
//! where `B_k = K_k (K_kᵀK_k)^{-1/2}` is the per-atom-whitened block. A collision
//! drives the cross term to `−∞`, so it *lowers* `½log|H|`, hence *lowers* the
//! outer criterion `V = loss + ½log|H| − occam`: at fixed fit, Bayesian evidence
//! strictly **prefers the unidentifiable parse**. Correct Bayes, wrong
//! interpretability. So identifiability needs a *certificate* channel — measured,
//! reported, and (per the integration plan §1.2) **never folded into `V`**, which
//! would double-charge the correct evidence. Default mode is report-only; a
//! harvest-level birth veto is a flag ([`TerraciniMode`]).
//!
//! # The certificate
//!
//! Whiten each block, stack, and take `μ_S = σ_min(B_S) ∈ [0, 1]`. `1/μ_S` is the
//! superposition-interference amplification (Theorem: `‖Δθ‖_white ≤ ‖Δz‖/μ_S`);
//! `whitened_excess = tr((B_SᵀB_S)⁻¹) − m` is the scale-free interference (0 iff
//! orthogonal); `attribution_risk = σ²·tr((J_SᵀJ_S)⁻¹)` is the exact expected
//! squared attribution error. The per-atom channel is reported as
//! [`TerraciniCertificate::per_atom_logdet`] so the split reconciles against the
//! evidence's own per-atom blocks.
//!
//! # Scale (integration plan §3c/§4)
//!
//! The certificate is computed on a **stratified row sample** at harvest cadence
//! (never per-row in the inner solve — exact-per-row is ~10¹⁵ FLOP at n=10⁸), and
//! aggregated in **bounded** state: per-atom worst/quantile margin (`O(K)`),
//! per-co-occurring-pair mean whitened margin (`O(active-pairs)`), and exact
//! clique margins retained ONLY for flagged rows — replacing the per-exact-pattern
//! map (`O(#patterns) ≈ O(rows)` at high `K`).
//!
//! # Closed forms (validated anchors)
//!
//! Two atoms whose whitened tangents meet at acute principal angle θ give
//! `margin = √(1 − cos θ)` and `whitened_excess = 2cos²θ/(1 − cos²θ)`; orthogonal
//! ⇒ margin 1, excess 0; collision ⇒ margin → 0, risks diverge; overcomplete
//! `Σ(d_k+1) > p` is refused with the Terracini bound named.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};

/// Designed certification target per atom. This is the reviewer's requested
/// `~10^4` scale, expressed as a named certifier policy rather than an
/// allocation-side threshold.
pub const TERRACINI_CERTIFIER_ROWS_PER_ATOM: usize = 10_000;

/// One atom's contribution to a parse: its value `g_k(t)` and its
/// amplitude-scaled tangent block `a_k · ∂g_k`.
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

}

/// The between-atom identifiability certificate for one sampled parse.
#[derive(Debug, Clone)]
pub struct TerraciniCertificate {
    /// The co-firing atoms, sorted (the pattern).
    pub pattern: Vec<usize>,
    /// Ambient output dimension `p`.
    pub p: usize,
    /// Total whitened tangent dimension `m = Σ_k (d_k + 1)`.
    pub m: usize,
    /// `μ_S = σ_min(B_S) ∈ [0, 1]` — the whitened Terracini margin.
    pub margin: f64,
    /// `1/μ_S` — the superposition-interference amplification factor.
    pub amplification: f64,
    /// `log det(B_SᵀB_S)` — the cross channel of the log-det split (`≤ 0`,
    /// `−∞` at collision); the term the evidence rewards with the wrong sign.
    pub cross_gram_logdet: f64,
    /// `tr((B_SᵀB_S)⁻¹) − m` — scale-free interference excess, `0` iff orthogonal.
    pub whitened_excess: f64,
    /// `σ²·tr((J_SᵀJ_S)⁻¹)` — exact expected squared attribution error.
    pub attribution_risk: f64,
    /// Per-atom `log det(K_kᵀK_k)` — the per-atom channel of the log-det split,
    /// so `Σ per_atom_logdet + cross_gram_logdet = log det(J_SᵀJ_S)`.
    pub per_atom_logdet: Vec<f64>,
}

// ============================================================================
// Aggregation (bounded, scale-safe) — replaces the O(#patterns) accumulator
// ============================================================================

/// How the terracini certificate is allowed to affect the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerraciniMode {
    /// Do not compute the certificate at all.
    Off,
    /// Compute and report; NEVER touch the criterion `V` or gate any move.
    #[default]
    Report,
    /// Report AND allow the harvest birth veto (a tripwire; still off `V`).
    Veto,
}

/// Configuration for a terracini scan.
#[derive(Debug, Clone)]
pub struct TerraciniConfig {
    pub mode: TerraciniMode,
    /// Ambient noise variance `σ²` for `attribution_risk`.
    pub noise_var: f64,
    /// Relative Tikhonov floor on each per-atom Gram.
    pub ridge: f64,
    /// A sampled clique whose margin falls below this is retained exactly in the
    /// report and, in `Veto` mode, blocks births that would collapse it.
    pub flag_margin: f64,
    /// Skip exact clique certificates whose active-set size exceeds this (the
    /// pairwise pass still covers them). Keeps per-row cost bounded.
    pub max_clique_atoms: usize,
    /// Run the cheap pairwise pass over co-occurring pairs.
    pub pair_pass: bool,
    /// Per-atom margin reservoir size for the quantile estimate.
    pub reservoir_cap: usize,
    /// Per-atom designed rows gathered before certification. The selected row
    /// set is sparse/reservoir state, never a dense `N×K` materialization.
    pub reservoir_rows_per_atom: usize,
}

impl Default for TerraciniConfig {
    fn default() -> Self {
        Self {
            mode: TerraciniMode::Report,
            noise_var: 1.0,
            ridge: 1.0e-12,
            flag_margin: 1.0e-2,
            max_clique_atoms: 16,
            pair_pass: true,
            reservoir_cap: 256,
            reservoir_rows_per_atom: TERRACINI_CERTIFIER_ROWS_PER_ATOM,
        }
    }
}

/// Per-atom margin summary: the worst and a quantile margin over every sampled
/// pair/clique the atom participates in. `O(1)` state per atom (bounded
/// reservoir).
#[derive(Debug, Clone)]
pub struct AtomMarginStat {
    pub atom: usize,
    pub n: usize,
    pub min_margin: f64,
    pub mean_margin: f64,
    /// Lower-tail quantile (5%) of the atom's sampled margins.
    pub q05_margin: f64,
    pub max_amplification: f64,
}

/// Per-co-occurring-pair summary: the mean whitened principal margin (the cheap
/// pass; pairwise margins upper-bound clique margins).
#[derive(Debug, Clone)]
pub struct PairMarginStat {
    pub a: usize,
    pub b: usize,
    pub n: usize,
    pub min_margin: f64,
    pub mean_margin: f64,
}

/// An exact clique margin retained because it fell below `flag_margin`.
#[derive(Debug, Clone)]
pub struct FlaggedClique {
    pub row: usize,
    pub pattern: Vec<usize>,
    pub margin: f64,
    pub cross_gram_logdet: f64,
    pub attribution_risk: f64,
}

#[derive(Debug, Clone)]
struct AtomAcc {
    n: usize,
    min_margin: f64,
    sum_margin: f64,
    max_amplification: f64,
    reservoir: Vec<f64>,
}

impl AtomAcc {
    fn quantile(&self, q: f64) -> f64 {
        if self.reservoir.is_empty() {
            return f64::NAN;
        }
        let mut v = self.reservoir.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((q * (v.len() as f64 - 1.0)).round() as usize).min(v.len() - 1);
        v[idx]
    }
}

#[derive(Debug, Clone)]
struct PairAcc {
    n: usize,
    min_margin: f64,
    sum_margin: f64,
}

/// The report a scan produces: worst atoms first, worst pairs first, and the
/// exact flagged cliques. Report-only by construction — nothing here changes `V`.
#[derive(Debug, Clone)]
pub struct TerraciniReport {
    pub mode: TerraciniMode,
    pub n_rows_scanned: usize,
    pub atoms: Vec<AtomMarginStat>,
    pub pairs: Vec<PairMarginStat>,
    pub flagged_cliques: Vec<FlaggedClique>,
    pub flag_margin: f64,
}

/// Bounded-state aggregator over sampled parse certificates.
#[derive(Debug, Clone)]
pub struct TerraciniAggregator {
    per_atom: BTreeMap<usize, AtomAcc>,
    per_pair: BTreeMap<(usize, usize), PairAcc>,
    flagged: Vec<FlaggedClique>,
    flag_margin: f64,
    n_rows: usize,
    mode: TerraciniMode,
}

impl TerraciniAggregator {

    /// Finalize into a worst-first report.
    pub fn finish(mut self) -> TerraciniReport {
        let mut atoms: Vec<AtomMarginStat> = self
            .per_atom
            .iter()
            .map(|(&atom, acc)| AtomMarginStat {
                atom,
                n: acc.n,
                min_margin: acc.min_margin,
                mean_margin: if acc.n > 0 {
                    acc.sum_margin / acc.n as f64
                } else {
                    f64::NAN
                },
                q05_margin: acc.quantile(0.05),
                max_amplification: acc.max_amplification,
            })
            .collect();
        atoms.sort_by(|x, y| {
            x.min_margin
                .partial_cmp(&y.min_margin)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut pairs: Vec<PairMarginStat> = self
            .per_pair
            .iter()
            .map(|(&(a, b), acc)| PairMarginStat {
                a,
                b,
                n: acc.n,
                min_margin: acc.min_margin,
                mean_margin: if acc.n > 0 {
                    acc.sum_margin / acc.n as f64
                } else {
                    f64::NAN
                },
            })
            .collect();
        pairs.sort_by(|x, y| {
            x.mean_margin
                .partial_cmp(&y.mean_margin)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.flagged.sort_by(|x, y| {
            x.margin
                .partial_cmp(&y.margin)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        TerraciniReport {
            mode: self.mode,
            n_rows_scanned: self.n_rows,
            atoms,
            pairs,
            flagged_cliques: self.flagged,
            flag_margin: self.flag_margin,
        }
    }
}

// ============================================================================
// Wiring — building the certificate from a fitted term, at harvest cadence
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// A block that is a single unit `value` vector with no tangent columns.
    fn unit_block(atom: usize, v: &[f64]) -> ParseBlock {
        ParseBlock {
            atom,
            value: Array1::from_vec(v.to_vec()),
            tangent: Array2::<f64>::zeros((v.len(), 0)),
        }
    }

    #[test]
    fn closed_form_angle_margin_and_excess() {
        // Acute principal angle θ ⇒ margin = √(1−cosθ), excess = 2cos²θ/(1−cos²θ).
        for &theta in &[0.1_f64, 0.5, 1.0, 1.4, 1.5] {
            let c = theta.cos();
            let s = theta.sin();
            let cert = parse_certificate(
                &[
                    unit_block(0, &[1.0, 0.0, 0.0, 0.0]),
                    unit_block(1, &[c, s, 0.0, 0.0]),
                ],
                1.0,
                0.0,
            )
            .unwrap();
            assert!(
                (cert.margin - (1.0 - c).sqrt()).abs() < 1e-9,
                "θ={theta} margin {}",
                cert.margin
            );
            let want_excess = 2.0 * c * c / (1.0 - c * c);
            assert!(
                (cert.whitened_excess - want_excess).abs() < 1e-9,
                "θ={theta} excess"
            );
            let want_logdet = (1.0 - c * c).ln();
            assert!(
                (cert.cross_gram_logdet - want_logdet).abs() < 1e-9,
                "θ={theta} logdet"
            );
            assert_eq!(cert.pattern, vec![0, 1]);
        }
    }

    #[test]
    fn logdet_channel_split_is_exact() {
        // Σ per_atom_logdet + cross_gram_logdet == log det(J_SᵀJ_S) on a
        // non-trivial 2-atom parse with tangents.
        let mut ta = Array2::<f64>::zeros((4, 1));
        ta[[1, 0]] = 2.0;
        let a = ParseBlock {
            atom: 0,
            value: Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            tangent: ta,
        };
        let mut tb = Array2::<f64>::zeros((4, 1));
        tb[[1, 0]] = 0.5;
        tb[[3, 0]] = 1.0;
        let b = ParseBlock {
            atom: 1,
            value: Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]),
            tangent: tb,
        };
        let cert = parse_certificate(&[a.clone(), b.clone()], 1.0, 0.0).unwrap();
        let ka = a.stacked().unwrap();
        let kb = b.stacked().unwrap();
        let mut j = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for c in 0..2 {
                j[[i, c]] = ka[[i, c]];
                j[[i, 2 + c]] = kb[[i, c]];
            }
        }
        let jtj = j.t().dot(&j);
        let evals = jtj.eigh(Side::Lower).unwrap().0;
        let ref_logdet: f64 = evals.iter().map(|&x| x.ln()).sum();
        let split = cert.per_atom_logdet.iter().sum::<f64>() + cert.cross_gram_logdet;
        assert!(
            (split - ref_logdet).abs() < 1e-9,
            "split {split} vs {ref_logdet}"
        );
    }

    #[test]
    fn orthogonal_is_perfectly_conditioned() {
        let cert = parse_certificate(
            &[
                unit_block(0, &[1.0, 0.0, 0.0]),
                unit_block(1, &[0.0, 1.0, 0.0]),
            ],
            2.0,
            0.0,
        )
        .unwrap();
        assert!((cert.margin - 1.0).abs() < 1e-12);
        assert!(cert.whitened_excess.abs() < 1e-12);
        assert!((cert.attribution_risk - 4.0).abs() < 1e-9);
    }

    #[test]
    fn collision_diverges() {
        let eps: f64 = 1e-6;
        let cert = parse_certificate(
            &[
                unit_block(0, &[1.0, 0.0]),
                unit_block(1, &[(eps).cos(), (eps).sin()]),
            ],
            1.0,
            0.0,
        )
        .unwrap();
        assert!(cert.margin < 1e-3, "margin {}", cert.margin);
        assert!(cert.amplification > 1e2);
        assert!(cert.whitened_excess > 1e5);
    }

    #[test]
    fn singular_joint_jacobian_has_infinite_attribution_risk_even_with_whitening_ridge() {
        let mut tangent = Array2::<f64>::zeros((2, 1));
        tangent[[0, 0]] = 1.0;
        let cert = parse_certificate(
            &[ParseBlock {
                atom: 0,
                value: Array1::from_vec(vec![1.0, 0.0]),
                tangent,
            }],
            1.0,
            1.0e-6,
        )
        .expect("the whitening diagnostic may regularize its own Gram");
        assert!(
            cert.attribution_risk.is_infinite(),
            "an exact risk cannot hide a singular physical Jacobian behind a ridge"
        );
    }

    #[test]
    fn overcomplete_is_refused() {
        let err = parse_certificate(
            &[
                unit_block(0, &[1.0, 0.0]),
                unit_block(1, &[0.0, 1.0]),
                unit_block(2, &[1.0, 1.0]),
            ],
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(err.contains("overcomplete"), "got: {err}");
    }

    #[test]
    fn aggregator_is_bounded_and_ranks_worst_first() {
        // Two cliques: {0,1} benign, {2,3} collided. Per-atom + per-pair state is
        // O(atoms)+O(pairs), independent of how many rows we push.
        let benign = parse_certificate(
            &[unit_block(0, &[1.0, 0.0]), unit_block(1, &[0.0, 1.0])],
            1.0,
            0.0,
        )
        .unwrap();
        let collided = parse_certificate(
            &[
                unit_block(2, &[1.0, 0.0]),
                unit_block(3, &[(0.008_f64).cos(), (0.008_f64).sin()]),
            ],
            1.0,
            0.0,
        )
        .unwrap();
        let mut agg = TerraciniAggregator::new(1e-2, 64, TerraciniMode::Veto);
        for _ in 0..10_000 {
            agg.record_clique(0, &benign);
            agg.record_clique(1, &collided);
            agg.record_pair(2, 3, collided.margin, collided.amplification);
        }
        let report = agg.finish();
        // Bounded: 4 atoms, 1 recorded pair, the flagged {2,3} clique.
        assert_eq!(report.atoms.len(), 4);
        assert_eq!(report.pairs.len(), 1);
        assert!(report.atoms[0].atom == 2 || report.atoms[0].atom == 3);
        assert!(report.atoms[0].min_margin < report.atoms[3].min_margin);
        assert!(!report.flagged_cliques.is_empty());
        assert_eq!(report.flagged_cliques[0].pattern, vec![2, 3]);
        // Veto fires for a birth joining the collided clique, not the benign one.
        assert!(report.vetoes_birth_into(&[2, 3]));
        assert!(!report.vetoes_birth_into(&[0, 1]));
    }

    #[test]
    fn stratified_coverage_hits_every_atom() {
        let rows = vec![
            (0usize, vec![0usize, 1]),
            (1, vec![0, 2]),
            (2, vec![1, 2]),
            (3, vec![3, 4]),
            (4, vec![0, 1]),
        ];
        let sel = stratified_rows_for_coverage(&rows, 1, 100);
        let mut seen = std::collections::BTreeSet::new();
        for r in &sel {
            for &a in &rows.iter().find(|(rr, _)| rr == r).unwrap().1 {
                seen.insert(a);
            }
        }
        for a in 0..=4 {
            assert!(seen.contains(&a), "atom {a} not covered");
        }
    }
}
