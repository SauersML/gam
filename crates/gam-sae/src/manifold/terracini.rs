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

use std::collections::{BTreeMap, BTreeSet};

use gam_linalg::faer_ndarray::FaerEigh;
use gam_solve::row_sampling_measure::RowSamplingMeasure;
use ndarray::{Array1, Array2, ArrayView2};

use faer::Side;

use super::{SaeManifoldAtom, SaeManifoldTerm};

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
    /// `d_k + 1`, the tangent-space dimension this atom contributes to the join.
    fn block_cols(&self) -> usize {
        self.tangent.ncols() + 1
    }

    /// Assemble `K_k = [value | tangent]`, shape `(p, d_k + 1)`.
    fn stacked(&self) -> Result<Array2<f64>, String> {
        let p = self.value.len();
        if self.tangent.ncols() != 0 && self.tangent.nrows() != p {
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

/// Symmetric inverse square root `G^{-1/2}` and `log det G` of a small SPD Gram.
fn inverse_sqrt_and_logdet(gram: &Array2<f64>, ridge: f64) -> Result<(Array2<f64>, f64), String> {
    let n = gram.nrows();
    let mut g = gram.clone();
    if ridge > 0.0 {
        // Relative Tikhonov floor so an internally rank-deficient block surfaces
        // as a per-atom problem, not a poisoned cross certificate.
        let tr = (0..n).map(|i| gram[[i, i]]).sum::<f64>().max(1.0e-300);
        let floor = ridge * tr / (n as f64);
        for i in 0..n {
            g[[i, i]] += floor;
        }
    }
    let (w, v) = g
        .eigh(Side::Lower)
        .map_err(|e| format!("terracini: eigh for whitening failed: {e}"))?;
    let mut scaled = v.clone();
    let mut logdet = 0.0_f64;
    for c in 0..n {
        let wc = w[c];
        if !(wc.is_finite() && wc > 0.0) {
            return Err(format!(
                "terracini: per-atom block not positive-definite (eigenvalue {wc:.3e}); \
                 charge it as a per-atom rank failure, not a cross collision"
            ));
        }
        logdet += wc.ln();
        let inv = 1.0 / wc.sqrt();
        for r in 0..n {
            scaled[[r, c]] *= inv;
        }
    }
    Ok((scaled.dot(&v.t()), logdet))
}

/// Build the between-atom parse certificate for one row's active atoms.
///
/// `noise_var = σ²` scales `attribution_risk`; `ridge` is a relative Tikhonov
/// floor on each per-atom Gram (pass `0.0` for the exact closed-form anchors).
/// Returns `Err` when the parse is overcomplete (`m > p`), naming the Terracini
/// bound.
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
    let mut pattern: Vec<usize> = blocks.iter().map(|b| b.atom).collect();
    pattern.sort_unstable();
    if m > p {
        return Err(format!(
            "terracini: overcomplete parse — Σ(d_k+1) = {m} > p = {p}; by the Terracini \
             bound the sparse manifold decomposition is NOT locally identifiable at this \
             parse (rank(J_S) cannot reach {m})"
        ));
    }

    let mut j_s = Array2::<f64>::zeros((p, m));
    let mut b_s = Array2::<f64>::zeros((p, m));
    let mut per_atom_logdet = Vec::with_capacity(blocks.len());
    let mut col = 0usize;
    for b in blocks {
        let k = b.stacked()?;
        let cols = k.ncols();
        let gram = k.t().dot(&k);
        let (g_inv_sqrt, logdet) = inverse_sqrt_and_logdet(&gram, ridge)?;
        per_atom_logdet.push(logdet);
        let bk = k.dot(&g_inv_sqrt);
        for c in 0..cols {
            for i in 0..p {
                j_s[[i, col + c]] = k[[i, c]];
                b_s[[i, col + c]] = bk[[i, c]];
            }
        }
        col += cols;
    }

    let btb = b_s.t().dot(&b_s);
    let w_b = btb
        .eigh(Side::Lower)
        .map_err(|e| format!("terracini: eigh of whitened cross-Gram failed: {e}"))?
        .0;
    let mut min_eig = f64::INFINITY;
    let mut logdet = 0.0_f64;
    let mut trace_inv = 0.0_f64;
    let mut singular = false;
    for &lam in w_b.iter() {
        if lam < min_eig {
            min_eig = lam;
        }
        let lam_c = lam.max(0.0);
        if lam_c > 1.0e-300 {
            logdet += lam_c.ln();
            trace_inv += 1.0 / lam_c;
        } else {
            singular = true;
        }
    }
    let margin = min_eig.max(0.0).sqrt();
    let amplification = if margin > 0.0 {
        1.0 / margin
    } else {
        f64::INFINITY
    };
    let (cross_gram_logdet, whitened_excess) = if singular {
        (f64::NEG_INFINITY, f64::INFINITY)
    } else {
        (logdet, trace_inv - m as f64)
    };

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
        pattern,
        p,
        m,
        margin,
        amplification,
        cross_gram_logdet,
        whitened_excess: whitened_excess.max(0.0),
        attribution_risk,
        per_atom_logdet,
    })
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
    cap: usize,
}

impl AtomAcc {
    fn new(cap: usize) -> Self {
        Self {
            n: 0,
            min_margin: f64::INFINITY,
            sum_margin: 0.0,
            max_amplification: 0.0,
            reservoir: Vec::new(),
            cap: cap.max(1),
        }
    }
    fn push(&mut self, margin: f64, amplification: f64) {
        self.n += 1;
        self.min_margin = self.min_margin.min(margin);
        self.sum_margin += margin;
        self.max_amplification = self.max_amplification.max(amplification);
        // Deterministic reservoir: keep the `cap` smallest margins (the tail the
        // quantile and any veto care about). Bounded memory, independent of rows.
        if self.reservoir.len() < self.cap {
            self.reservoir.push(margin);
        } else if let Some((idx, &worst)) = self
            .reservoir
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if margin < worst {
                self.reservoir[idx] = margin;
            }
        }
    }
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

impl TerraciniReport {
    /// Veto consult: is there a flagged clique that is a subset of `pattern`
    /// (i.e. a birth joining `pattern` would sit on a collided clique)? Only
    /// meaningful in `Veto` mode; in any other mode this always returns `false`.
    pub fn vetoes_birth_into(&self, pattern: &[usize]) -> bool {
        if self.mode != TerraciniMode::Veto {
            return false;
        }
        let set: std::collections::BTreeSet<usize> = pattern.iter().copied().collect();
        self.flagged_cliques
            .iter()
            .any(|fc| fc.pattern.iter().all(|a| set.contains(a)))
    }
}

/// Bounded-state aggregator over sampled parse certificates.
#[derive(Debug, Clone)]
pub struct TerraciniAggregator {
    per_atom: BTreeMap<usize, AtomAcc>,
    per_pair: BTreeMap<(usize, usize), PairAcc>,
    flagged: Vec<FlaggedClique>,
    flag_margin: f64,
    reservoir_cap: usize,
    n_rows: usize,
    mode: TerraciniMode,
}

impl TerraciniAggregator {
    pub fn new(flag_margin: f64, reservoir_cap: usize, mode: TerraciniMode) -> Self {
        Self {
            per_atom: BTreeMap::new(),
            per_pair: BTreeMap::new(),
            flagged: Vec::new(),
            flag_margin,
            reservoir_cap,
            n_rows: 0,
            mode,
        }
    }

    fn atom_push(&mut self, atom: usize, margin: f64, amplification: f64) {
        let cap = self.reservoir_cap;
        self.per_atom
            .entry(atom)
            .or_insert_with(|| AtomAcc::new(cap))
            .push(margin, amplification);
    }

    /// Record the cheap pairwise certificate for co-occurring atoms `(a, b)`.
    pub fn record_pair(&mut self, a: usize, b: usize, margin: f64, amplification: f64) {
        let key = if a <= b { (a, b) } else { (b, a) };
        let e = self.per_pair.entry(key).or_insert_with(|| PairAcc {
            n: 0,
            min_margin: f64::INFINITY,
            sum_margin: 0.0,
        });
        e.n += 1;
        e.min_margin = e.min_margin.min(margin);
        e.sum_margin += margin;
        self.atom_push(a, margin, amplification);
        self.atom_push(b, margin, amplification);
    }

    /// Record an exact clique certificate; retains it only if it flags.
    pub fn record_clique(&mut self, row: usize, cert: &TerraciniCertificate) {
        for &atom in &cert.pattern {
            self.atom_push(atom, cert.margin, cert.amplification);
        }
        if cert.margin < self.flag_margin {
            self.flagged.push(FlaggedClique {
                row,
                pattern: cert.pattern.clone(),
                margin: cert.margin,
                cross_gram_logdet: cert.cross_gram_logdet,
                attribution_risk: cert.attribution_risk,
            });
        }
    }

    fn note_row(&mut self) {
        self.n_rows += 1;
    }

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

/// One atom's intrinsic latent dimension `d_k`, from its cached basis Jacobian.
fn atom_latent_dim(atom: &SaeManifoldAtom) -> usize {
    atom.basis_jacobian.shape()[2]
}

/// Build the [`ParseBlock`] for atom `k` on a given row from the fitted term.
///
/// The value column is the amplitude-Jacobian `g_k(t)` ([`SaeManifoldAtom::decoded_row`]);
/// the tangent columns are `a_k · ∂g_k/∂t_axis` ([`SaeManifoldAtom::decoded_derivative_row`]),
/// with `a_k` this row's assignment mass — the SAME numbers `row_jets_for_logdet`
/// already builds, so this is a channel split, not a new evaluation pass.
pub fn parse_block_from_term(
    term: &SaeManifoldTerm,
    atom: usize,
    row: usize,
    amplitude: f64,
) -> ParseBlock {
    let a = &term.atoms[atom];
    let value = a.decoded_row(row);
    let d = atom_latent_dim(a);
    let p = value.len();
    let mut tangent = Array2::<f64>::zeros((p, d));
    for axis in 0..d {
        let deriv = a.decoded_derivative_row(row, axis);
        for i in 0..p {
            tangent[[i, axis]] = amplitude * deriv[i];
        }
    }
    ParseBlock {
        atom,
        value,
        tangent,
    }
}

/// Stratified row selection ensuring each atom is covered by ≥ `q` sampled rows
/// (up to `cap` total rows). `rows_with_patterns` is the full per-row active-atom
/// list (e.g. from the harvest sparse codes); returns the selected row indices.
/// Deterministic (round-robin over each atom's occurrence list), so low-occupancy
/// atoms — exactly the ones the certificate exists to protect — are never starved.
pub fn stratified_rows_for_coverage(
    rows_with_patterns: &[(usize, Vec<usize>)],
    q: usize,
    cap: usize,
) -> Vec<usize> {
    // Per-atom occurrence lists (indices into rows_with_patterns).
    let mut occ: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, (_row, pattern)) in rows_with_patterns.iter().enumerate() {
        for &atom in pattern {
            occ.entry(atom).or_default().push(i);
        }
    }
    let mut chosen = vec![false; rows_with_patterns.len()];
    let mut count_for_atom: BTreeMap<usize, usize> = BTreeMap::new();
    let mut n_selected = 0usize;
    // Round-robin: give each still-uncovered atom one of its rows per pass.
    let mut progress = true;
    while progress && n_selected < cap {
        progress = false;
        let atoms: Vec<usize> = occ.keys().copied().collect();
        for atom in atoms {
            if *count_for_atom.get(&atom).unwrap_or(&0) >= q {
                continue;
            }
            let next = occ
                .get(&atom)
                .and_then(|rows| rows.iter().copied().find(|&idx| !chosen[idx]));
            if let Some(idx) = next {
                chosen[idx] = true;
                n_selected += 1;
                for &a in &rows_with_patterns[idx].1 {
                    *count_for_atom.entry(a).or_insert(0) += 1;
                }
                progress = true;
                if n_selected >= cap {
                    break;
                }
            }
        }
    }
    rows_with_patterns
        .iter()
        .enumerate()
        .filter(|(i, _)| chosen[*i])
        .map(|(_, (row, _))| *row)
        .collect()
}

/// Designed row reservoir for Terracini certification.
///
/// The first pass draws a measure-designed sample carrying the same row-design
/// object used by the honesty-weighted fit path; the second pass actively tops
/// up rare atoms by round-robin coverage so low-occupancy atoms are not audited
/// on weaker evidence merely because their rows were scarce under the global
/// measure. Returned rows are sorted and unique.
pub fn designed_reservoir_rows_for_coverage(
    rows_with_patterns: &[(usize, Vec<usize>)],
    measure: Option<&RowSamplingMeasure>,
    q: usize,
    cap: usize,
    seed: u64,
) -> Result<Vec<usize>, String> {
    if rows_with_patterns.is_empty() || q == 0 || cap == 0 {
        return Ok(Vec::new());
    }
    let mut selected = BTreeSet::new();
    if let Some(measure) = measure {
        let n_rows = rows_with_patterns
            .iter()
            .map(|(row, _)| *row)
            .max()
            .map_or(0usize, |row| row + 1);
        if measure.n_rows() != n_rows {
            return Err(format!(
                "designed_reservoir_rows_for_coverage: measure covers {} rows but patterns cover {n_rows}",
                measure.n_rows()
            ));
        }
        let sample = measure.designed_subsample(cap, seed);
        let eligible: BTreeSet<usize> = rows_with_patterns.iter().map(|(row, _)| *row).collect();
        for row in sample.rows {
            if eligible.contains(&row) {
                selected.insert(row);
            }
        }
    }

    let topup = stratified_rows_for_coverage(rows_with_patterns, q, cap);
    for row in topup {
        if selected.len() >= cap {
            break;
        }
        selected.insert(row);
    }
    Ok(selected.into_iter().collect())
}

fn sparse_rows_with_patterns(
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
    k_atoms: usize,
) -> Result<Vec<(usize, Vec<usize>)>, String> {
    if indices.dim() != codes.dim() {
        return Err(format!(
            "sparse_rows_with_patterns: indices shape {:?} != codes shape {:?}",
            indices.dim(),
            codes.dim()
        ));
    }
    let mut rows = Vec::new();
    for row in 0..indices.nrows() {
        let mut atoms = BTreeSet::new();
        for slot in 0..indices.ncols() {
            let code = codes[[row, slot]];
            if code == 0.0 {
                continue;
            }
            let atom = indices[[row, slot]] as usize;
            if atom >= k_atoms {
                return Err(format!(
                    "sparse_rows_with_patterns: atom index {atom} out of range 0..{k_atoms}"
                ));
            }
            atoms.insert(atom);
        }
        if atoms.len() >= 2 {
            rows.push((row, atoms.into_iter().collect()));
        }
    }
    Ok(rows)
}

fn sparse_code_amplitude(
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
    row: usize,
    atom: usize,
) -> f64 {
    let mut amplitude = 0.0_f64;
    for slot in 0..indices.ncols() {
        if indices[[row, slot]] as usize == atom {
            amplitude += codes[[row, slot]] as f64;
        }
    }
    amplitude
}

/// Scan a set of sampled rows and produce the terracini report.
///
/// `rows_with_patterns` are the sampled `(row, active-atoms)`; `amplitudes` is
/// `(N, K)` (e.g. [`SaeManifoldTerm::fitted_assignment_amplitudes`]). The cheap
/// pairwise pass runs over every co-occurring pair; exact clique certificates run
/// for patterns up to `cfg.max_clique_atoms`. In `Off` mode returns an empty
/// report without touching the term.
pub fn terracini_scan(
    term: &SaeManifoldTerm,
    rows_with_patterns: &[(usize, Vec<usize>)],
    amplitudes: ArrayView2<f64>,
    cfg: &TerraciniConfig,
) -> TerraciniReport {
    let mut agg = TerraciniAggregator::new(cfg.flag_margin, cfg.reservoir_cap, cfg.mode);
    if cfg.mode == TerraciniMode::Off {
        return agg.finish();
    }
    for (row, pattern) in rows_with_patterns {
        if pattern.len() < 2 {
            continue;
        }
        agg.note_row();
        let blocks: Vec<ParseBlock> = pattern
            .iter()
            .map(|&k| parse_block_from_term(term, k, *row, amplitudes[[*row, k]]))
            .collect();
        // Cheap pairwise pass (necessary screen; upper-bounds clique margins).
        if cfg.pair_pass {
            for i in 0..blocks.len() {
                for j in (i + 1)..blocks.len() {
                    if let Ok(cert) = parse_certificate(
                        &[blocks[i].clone(), blocks[j].clone()],
                        cfg.noise_var,
                        cfg.ridge,
                    ) {
                        agg.record_pair(
                            blocks[i].atom,
                            blocks[j].atom,
                            cert.margin,
                            cert.amplification,
                        );
                    }
                }
            }
        }
        // Exact clique certificate for bounded-size patterns.
        if pattern.len() <= cfg.max_clique_atoms {
            if let Ok(cert) = parse_certificate(&blocks, cfg.noise_var, cfg.ridge) {
                agg.record_clique(*row, &cert);
            }
        }
    }
    agg.finish()
}

/// Reservoir-sized Terracini scan from the canonical sparse-code state.
///
/// `indices/codes` are the sparse `N×s` routing state. Certification first
/// gathers a designed, per-atom-covered row reservoir and then evaluates only
/// those rows, with amplitudes read from the sparse code slots. No dense `N×K`
/// assignment matrix or `N×P` curved-prediction cache is constructed.
pub fn terracini_scan_sparse_codes(
    term: &SaeManifoldTerm,
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
    measure: Option<&RowSamplingMeasure>,
    seed: u64,
    cfg: &TerraciniConfig,
) -> Result<TerraciniReport, String> {
    if cfg.mode == TerraciniMode::Off {
        return Ok(TerraciniAggregator::new(cfg.flag_margin, cfg.reservoir_cap, cfg.mode).finish());
    }
    if indices.nrows() != term.n_obs() {
        return Err(format!(
            "terracini_scan_sparse_codes: sparse codes have {} rows but term has {}",
            indices.nrows(),
            term.n_obs()
        ));
    }
    let rows_with_patterns = sparse_rows_with_patterns(indices, codes, term.k_atoms())?;
    let cap = cfg
        .reservoir_rows_per_atom
        .saturating_mul(term.k_atoms())
        .min(rows_with_patterns.len());
    let rows = designed_reservoir_rows_for_coverage(
        &rows_with_patterns,
        measure,
        cfg.reservoir_rows_per_atom,
        cap,
        seed,
    )?;
    let selected: BTreeSet<usize> = rows.into_iter().collect();
    let mut agg = TerraciniAggregator::new(cfg.flag_margin, cfg.reservoir_cap, cfg.mode);
    for (row, pattern) in rows_with_patterns {
        if !selected.contains(&row) {
            continue;
        }
        agg.note_row();
        let blocks: Vec<ParseBlock> = pattern
            .iter()
            .map(|&k| {
                parse_block_from_term(
                    term,
                    k,
                    row,
                    sparse_code_amplitude(indices, codes, row, k),
                )
            })
            .collect();
        if cfg.pair_pass {
            for i in 0..blocks.len() {
                for j in (i + 1)..blocks.len() {
                    if let Ok(cert) = parse_certificate(
                        &[blocks[i].clone(), blocks[j].clone()],
                        cfg.noise_var,
                        cfg.ridge,
                    ) {
                        agg.record_pair(
                            blocks[i].atom,
                            blocks[j].atom,
                            cert.margin,
                            cert.amplification,
                        );
                    }
                }
            }
        }
        if pattern.len() <= cfg.max_clique_atoms {
            if let Ok(cert) = parse_certificate(&blocks, cfg.noise_var, cfg.ridge) {
                agg.record_clique(row, &cert);
            }
        }
    }
    Ok(agg.finish())
}

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
            assert!((cert.margin - (1.0 - c).sqrt()).abs() < 1e-9, "θ={theta} margin {}", cert.margin);
            let want_excess = 2.0 * c * c / (1.0 - c * c);
            assert!((cert.whitened_excess - want_excess).abs() < 1e-9, "θ={theta} excess");
            let want_logdet = (1.0 - c * c).ln();
            assert!((cert.cross_gram_logdet - want_logdet).abs() < 1e-9, "θ={theta} logdet");
            assert_eq!(cert.pattern, vec![0, 1]);
        }
    }

    #[test]
    fn logdet_channel_split_is_exact() {
        // Σ per_atom_logdet + cross_gram_logdet == log det(J_SᵀJ_S) on a
        // non-trivial 2-atom parse with tangents.
        let mut ta = Array2::<f64>::zeros((4, 1));
        ta[[1, 0]] = 2.0;
        let a = ParseBlock { atom: 0, value: Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), tangent: ta };
        let mut tb = Array2::<f64>::zeros((4, 1));
        tb[[1, 0]] = 0.5;
        tb[[3, 0]] = 1.0;
        let b = ParseBlock { atom: 1, value: Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0]), tangent: tb };
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
        assert!((split - ref_logdet).abs() < 1e-9, "split {split} vs {ref_logdet}");
    }

    #[test]
    fn orthogonal_is_perfectly_conditioned() {
        let cert = parse_certificate(
            &[unit_block(0, &[1.0, 0.0, 0.0]), unit_block(1, &[0.0, 1.0, 0.0])],
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
            &[unit_block(0, &[1.0, 0.0]), unit_block(1, &[(eps).cos(), (eps).sin()])],
            1.0,
            0.0,
        )
        .unwrap();
        assert!(cert.margin < 1e-3, "margin {}", cert.margin);
        assert!(cert.amplification > 1e2);
        assert!(cert.whitened_excess > 1e5);
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
            &[unit_block(2, &[1.0, 0.0]), unit_block(3, &[(0.008_f64).cos(), (0.008_f64).sin()])],
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
