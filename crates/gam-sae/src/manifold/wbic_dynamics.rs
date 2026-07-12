//! Cross-checkpoint WBIC learning-coefficient dynamics (M2): the per-atom
//! running-complexity time series `λ_k(step)` with an anytime-valid birth test.
//!
//! WHY. [`super::wbic_audit`] prices ONE fit's singular-learning learning
//! coefficient `λ_k = ½·rank_soft_k·basis_edf_k` — the finite-`n` running
//! complexity `λ(N_eff) = d(−log Z)/d(log N_eff)` of a single atom
//! ([`super::wbic_audit::ReconSpectrum::learning_coefficient`], Theorem K). That
//! is a snapshot. During training an atom is BORN: its decoder goes from noise
//! (every reconstruction direction below the Marchenko–Pastur edge, `rank_soft ≈
//! 0`, `λ ≈ 0`) to a resolved low-rank structure (`rank_soft ≥ 1`, `λ ≥ ½`). So a
//! FEATURE BIRTH IS A `λ_k` JUMP. This module sweeps the audit's `λ_k` across the
//! OLMo intermediate-training checkpoint axis to recover, per atom, the
//! trajectory `λ_k(step)`, and wraps the step-to-step jumps in the same
//! anytime-valid e-process the cross-checkpoint change instrument uses
//! ([`super::super::inference::checkpoint_dynamics`]), so the multiple-testing
//! over (atom × step) is FDR-controlled at any data-dependent stopping time.
//!
//! It is pure ASSEMBLY of three landed instruments — none is reimplemented:
//!
//! * [`super::wbic_audit::recon_spectrum`] — the per-checkpoint reconstruction
//!   spectrum of an atom's decoder curve, from which `λ_k` is read. The decoder
//!   grid carries the fitted curve VALUES on a shared latent grid but no chart
//!   design, so — exactly as [`super::super::inference::checkpoint_dynamics`]
//!   documents for its Riesz inputs — the honest basis here is the identity
//!   interpolation basis (one observation per grid node), i.e. `basis_edf = 1`
//!   per reconstruction direction. We reuse the audit's SVD spectrum + MP edge
//!   computation verbatim and price its directions with that graded count, so
//!   `λ_k = ½·rank_soft_k` is the audit's learning coefficient specialized to the
//!   grid-interpolation basis.
//! * [`super::super::inference::layer_transport`] — the checkpoint axis is reused
//!   as the transport "layer" axis, exactly as `checkpoint_dynamics` does, to
//!   ALIGN the atom's latent chart across consecutive checkpoints (so a `λ_k`
//!   jump is scored on the SAME atom, not a chart relabelling). Best-effort: a
//!   degenerate chart simply carries no transport report for that step.
//! * [`gam_terms::inference::structure_evidence`] — each consecutive-step
//!   `λ_k` jump `Δλ = λ^{(c+1)} − λ^{(c)}`, studentized by a grid-node jackknife
//!   SE, feeds one calibrated anytime-valid e-value into a per-atom
//!   [`StructureLedger`] under the null "the atom's complexity did not jump at
//!   this step". The per-atom ledger is FDR-safe over its steps; the top-level
//!   [`e_benjamini_hochberg`] over the atoms' peak birth e-values is the
//!   cross-atom multiple-testing wrapper. A confirmed positive jump is a BIRTH.

use gam_terms::inference::structure_evidence::{
    ClaimKind, StructureLedger, e_benjamini_hochberg, log_e_from_p_calibrator,
};
use ndarray::{Array2, ArrayView2, ArrayView4};
use statrs::distribution::{ContinuousCDF, Normal};

use super::wbic_audit::{ReconSpectrum, recon_spectrum};
use crate::inference::layer_transport::{ChartTopology, LayerTransportReport, fit_layer_transport};

/// Absolute floor on the studentizing SE of a `λ_k` jump, plus the relative floor
/// as a fraction of the local `λ` scale. A perfectly stable atom has a jackknife
/// SE at the rounding floor; without a floor a machine-eps `Δλ` over a machine-eps
/// SE would fabricate a spurious `z`. A real birth (`Δλ ~ 1`) dwarfs the floor, so
/// the floor only suppresses numerical non-jumps, never a genuine one.
const LAMBDA_JUMP_SE_ABS_FLOOR: f64 = 1.0e-9;
const LAMBDA_JUMP_SE_REL_FLOOR: f64 = 1.0e-6;

/// Inputs for one cross-checkpoint `λ_k` dynamics run.
///
/// `decoder_grid` is `[n_checkpoints, n_atoms, n_grid, ambient_dim]` — the decoder
/// curve of every atom sampled on the shared latent grid at every checkpoint, the
/// SAME layout [`super::super::inference::checkpoint_dynamics::CheckpointDynamicsInput`]
/// consumes (so the two instruments run off one harvested tensor).
pub struct WbicDynamicsInput<'a> {
    pub decoder_grid: ArrayView4<'a, f64>,
    pub checkpoint_ids: &'a [String],
    pub atom_names: &'a [String],
    /// Noise dispersion `R` setting the Marchenko–Pastur edge `R·(1+√(p/n))²` the
    /// hard/soft rank counts threshold on (the audit's `r_floor`).
    pub r_floor: f64,
    /// Target FDR level for the per-atom step ledger and the cross-atom e-BH.
    pub birth_alpha: f64,
}

/// One consecutive-checkpoint `λ_k` jump and its anytime-valid evidence.
#[derive(Clone, Debug)]
pub struct LambdaJump {
    /// Step index `c → c+1` (0-based over the checkpoint axis).
    pub step: usize,
    /// Earlier / later checkpoint labels.
    pub from_ckpt: String,
    pub to_ckpt: String,
    /// `Δλ = λ^{(c+1)} − λ^{(c)}` — positive for a birth (complexity rising),
    /// negative for a death.
    pub delta_lambda: f64,
    /// Jackknife SE of the jump, `√(se_c0² + se_c1²)` (grid-node jackknife per
    /// checkpoint), floored (see [`LAMBDA_JUMP_SE_ABS_FLOOR`]).
    pub se: f64,
    /// Studentized jump `Δλ / se`.
    pub z: f64,
    /// Calibrated anytime-valid log-e-value for the no-jump null at this step.
    pub log_e: f64,
    /// Confirmed as a BIRTH by the per-atom e-BH certificate at `birth_alpha`
    /// AND `Δλ > 0`. A confirmed negative jump is a death, not a birth.
    pub born: bool,
}

/// The `λ_k(step)` trajectory of one atom across the checkpoint axis.
#[derive(Clone, Debug)]
pub struct AtomLambdaTrajectory {
    pub atom_name: String,
    /// `λ_k^{(c)} = ½·rank_soft^{(c)}` at every checkpoint (the time series).
    pub lambda: Vec<f64>,
    /// WBIC tempered soft rank count at every checkpoint.
    pub rank_soft: Vec<f64>,
    /// Hard Marchenko–Pastur detection count at every checkpoint.
    pub mp_reconstruction_rank: Vec<usize>,
    /// Rank the production criterion charges at every checkpoint, including the
    /// #2258 promotion of an alive decoder below the MP detection edge.
    pub production_chargeable_rank: Vec<usize>,
    /// Consecutive-step jumps with their birth evidence.
    pub jumps: Vec<LambdaJump>,
    /// Per-atom anytime-valid ledger: one calibrated e-value per step under the
    /// no-jump null. FDR-safe over the atom's steps at any stopping time.
    pub birth_evidence: StructureLedger,
    /// Best-effort chart correspondence per step (checkpoint axis as the layer
    /// axis); `None` for a step whose chart was too degenerate to align.
    pub transports: Vec<Option<LayerTransportReport>>,
}

impl AtomLambdaTrajectory {
    /// The atom's birth log-e-value: the log of the MEAN of its positive-jump
    /// e-values, averaged over the atom's fixed step count. `−∞` if it never rose.
    /// Feeds the cross-atom e-BH.
    ///
    /// The natural summary "did this atom ever rise?" is a UNION over steps. Its
    /// tempting instrument — `max_step e_step` — is NOT an e-value: for `S` steps
    /// `E_{H0}[max_s e_s] ≤ S`, not `≤ 1`, so feeding it to e-BH inflates the
    /// cross-atom FDR by up to a factor of the step count. The valid union
    /// instrument is the MIXTURE (average) e-value `ē = (1/S)·Σ_s e_s·1[Δλ_s>0]`:
    /// each summand `e_s·1[Δλ_s>0] ≤ e_s` has `E_{H0}[e_s] ≤ 1` (the per-step
    /// calibrated e-value), the sign indicator lies in `{0,1}` so it can only
    /// shrink the summand, and the divisor is the FIXED step count `S` (not the
    /// data-dependent number of positive steps), hence `E_{H0}[ē] ≤ (1/S)·Σ_s 1 =
    /// 1` — a genuine e-value. We prefer the mixture to a compounded product
    /// martingale here because a birth is a SINGLE step-jump (a union claim), not
    /// evidence accumulating across every step; the product would test "jumped at
    /// every step" and vanish for a lone birth. A large single-step birth e-value
    /// survives the `1/S` dilution and still clears e-BH; a never-born atom sums
    /// to 0 (`log −∞`) and cannot be selected. Computed by log-sum-exp for
    /// overflow safety.
    pub fn peak_birth_log_e(&self) -> f64 {
        let n_steps = self.jumps.len();
        if n_steps == 0 {
            return f64::NEG_INFINITY;
        }
        let positive: Vec<f64> = self
            .jumps
            .iter()
            .filter(|j| j.delta_lambda > 0.0)
            .map(|j| j.log_e)
            .collect();
        let max = positive.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            // No positive jump ⇒ the mixture is exactly 0 ⇒ log −∞.
            return f64::NEG_INFINITY;
        }
        // log( (1/S)·Σ_{Δλ>0} exp(log_e) ) = logsumexp − ln S, overflow-safe.
        let sumexp: f64 = positive.iter().map(|&l| (l - max).exp()).sum();
        max + sumexp.ln() - (n_steps as f64).ln()
    }
}

/// The full run: every atom's `λ_k(step)` trajectory plus the cross-atom birth
/// certificate (which atoms are born somewhere in training, FDR-controlled).
#[derive(Clone, Debug)]
pub struct WbicDynamicsReport {
    pub atoms: Vec<AtomLambdaTrajectory>,
    /// Indices into `atoms` the cross-atom e-BH confirms as born (over the atoms'
    /// peak positive-jump e-values, at `birth_alpha`).
    pub cross_atom_born: Vec<usize>,
}

/// Sweep the WBIC learning coefficient `λ_k` across the checkpoint axis for every
/// atom, returning per-atom `λ_k(step)` trajectories and the anytime-valid birth
/// certificate.
///
/// Per atom, per consecutive checkpoint step `c → c+1`:
/// 1. builds the reconstruction spectrum of each checkpoint's decoder curve and
///    reads `λ_k^{(c)}` ([`atom_learning_coefficient`]);
/// 2. aligns the two charts (best-effort transport, checkpoint axis as layer axis);
/// 3. studentizes the jump `Δλ` by a grid-node jackknife SE and absorbs the
///    calibrated anytime-valid e-value into the atom's no-jump ledger.
/// Finally runs e-BH over the atoms' peak positive jumps for the cross-atom
/// birth certificate.
pub fn wbic_lambda_dynamics(input: &WbicDynamicsInput<'_>) -> Result<WbicDynamicsReport, String> {
    let shape = input.decoder_grid.shape();
    let (n_checkpoints, n_atoms, n_grid, ambient_dim) = (shape[0], shape[1], shape[2], shape[3]);
    if n_checkpoints < 2 {
        return Err(format!(
            "wbic lambda dynamics needs at least two checkpoints, got {n_checkpoints}"
        ));
    }
    if input.checkpoint_ids.len() != n_checkpoints {
        return Err(format!(
            "checkpoint_ids length {} disagrees with decoder grid checkpoint axis {n_checkpoints}",
            input.checkpoint_ids.len()
        ));
    }
    if input.atom_names.len() != n_atoms {
        return Err(format!(
            "atom_names length {} disagrees with decoder grid atom axis {n_atoms}",
            input.atom_names.len()
        ));
    }
    if n_grid < 2 || ambient_dim == 0 {
        return Err(format!(
            "wbic lambda dynamics needs a non-trivial grid ({n_grid}) and ambient dim ({ambient_dim})"
        ));
    }
    if !(input.r_floor > 0.0) {
        return Err(format!(
            "wbic lambda dynamics needs a positive noise floor r_floor, got {}",
            input.r_floor
        ));
    }
    if !(input.birth_alpha > 0.0 && input.birth_alpha < 1.0) {
        return Err(format!(
            "wbic lambda dynamics needs birth_alpha in (0,1), got {}",
            input.birth_alpha
        ));
    }
    if input.decoder_grid.iter().any(|v| !v.is_finite()) {
        return Err("wbic lambda dynamics decoder grid must be finite".to_string());
    }

    let mut atoms = Vec::with_capacity(n_atoms);
    for atom in 0..n_atoms {
        let atom_name = input.atom_names[atom].clone();

        // --- λ_k(step) time series: the WBIC learning coefficient per checkpoint.
        let mut lambda = Vec::with_capacity(n_checkpoints);
        let mut rank_soft = Vec::with_capacity(n_checkpoints);
        let mut mp_reconstruction_rank = Vec::with_capacity(n_checkpoints);
        let mut production_chargeable_rank = Vec::with_capacity(n_checkpoints);
        let mut lambda_se = Vec::with_capacity(n_checkpoints);
        for c in 0..n_checkpoints {
            let curve = input.decoder_grid.slice(ndarray::s![c, atom, .., ..]);
            let spec = atom_learning_spectrum(curve, input.r_floor).map_err(|e| {
                format!(
                    "wbic spectrum for atom '{atom_name}' checkpoint {} failed: {e}",
                    input.checkpoint_ids[c]
                )
            })?;
            lambda.push(spec.learning_coefficient());
            rank_soft.push(spec.rank_soft());
            mp_reconstruction_rank.push(spec.mp_reconstruction_rank());
            production_chargeable_rank.push(spec.production_chargeable_rank());
            lambda_se.push(lambda_jackknife_se(curve, input.r_floor));
        }

        // --- per-step jumps + anytime-valid no-jump evidence.
        let mut jumps = Vec::with_capacity(n_checkpoints - 1);
        let mut transports = Vec::with_capacity(n_checkpoints - 1);
        let mut birth_evidence = StructureLedger::new();
        for step in 0..n_checkpoints - 1 {
            let c0 = step;
            let c1 = step + 1;
            let delta_lambda = lambda[c1] - lambda[c0];
            // Independent-checkpoint jump SE, floored so a numerical non-jump
            // cannot masquerade as a studentized signal.
            let raw_se = (lambda_se[c0] * lambda_se[c0] + lambda_se[c1] * lambda_se[c1]).sqrt();
            let scale = 0.5 * (lambda[c0].abs() + lambda[c1].abs());
            let se = raw_se
                .max(LAMBDA_JUMP_SE_ABS_FLOOR)
                .max(LAMBDA_JUMP_SE_REL_FLOOR * scale);
            let z = delta_lambda / se;
            let log_e = no_jump_log_e_value(z)?;

            let claim = birth_evidence.register(ClaimKind::Custom {
                label: format!(
                    "atom '{atom_name}' λ jumped from checkpoint {} to {}",
                    input.checkpoint_ids[c0], input.checkpoint_ids[c1]
                ),
            });
            birth_evidence.absorb_log(claim, log_e)?;

            transports.push(best_effort_transport(input.decoder_grid, atom, c0, c1));

            jumps.push(LambdaJump {
                step,
                from_ckpt: input.checkpoint_ids[c0].clone(),
                to_ckpt: input.checkpoint_ids[c1].clone(),
                delta_lambda,
                se,
                z,
                log_e,
                born: false, // set from the per-atom certificate below
            });
        }

        // Per-atom e-BH over the steps: a confirmed POSITIVE jump is a birth.
        let cert = birth_evidence.certify(input.birth_alpha);
        for (jump, entry) in jumps.iter_mut().zip(cert.entries.iter()) {
            jump.born = entry.confirmed && jump.delta_lambda > 0.0;
        }

        atoms.push(AtomLambdaTrajectory {
            atom_name,
            lambda,
            rank_soft,
            mp_reconstruction_rank,
            production_chargeable_rank,
            jumps,
            birth_evidence,
            transports,
        });
    }

    // Cross-atom multiple-testing wrapper: e-BH over the atoms' peak positive-jump
    // e-values. Atoms that never rose contribute a non-positive log-e and cannot
    // be selected.
    let peak_log_e: Vec<f64> = atoms
        .iter()
        .map(|a| {
            let p = a.peak_birth_log_e();
            if p.is_finite() { p } else { f64::NEG_INFINITY }
        })
        .collect();
    // e-BH expects finite inputs; map a never-born −∞ to a large negative so its
    // e-value is ~0 (never confirmed) without polluting the ordering.
    let finite_log_e: Vec<f64> = peak_log_e
        .iter()
        .map(|&v| if v.is_finite() { v } else { -1.0e300 })
        .collect();
    let cross_atom_born = e_benjamini_hochberg(&finite_log_e, input.birth_alpha);

    Ok(WbicDynamicsReport {
        atoms,
        cross_atom_born,
    })
}

/// The reconstruction spectrum of ONE decoder curve `g: t ↦ ℝ^ambient` sampled on
/// a grid, priced with the grid-interpolation basis (`basis_edf = 1`).
///
/// Reuses [`recon_spectrum`] for the SVD reconstruction spectrum `μ` and the
/// Marchenko–Pastur edge (with the identity chart Gram `G = I_{n_grid}`, its `μ`
/// is exactly `sv(curve)²/n_eff`, the decoder curve's per-observation singular
/// energies). The audit's `basis_edf = tr(G(G+λS)⁻¹)` collapses to `n_grid` under
/// the identity Gram — that is the interpolation design's own dof, not the atom's
/// graded chart-basis count, so we OVERRIDE it to the honest per-direction value
/// `1` (the decoder curve carries no chart design; each reconstruction direction
/// is one graded unit). The learning coefficient is then `λ = ½·rank_soft`, the
/// audit's `λ̂` specialized to this basis.
fn atom_learning_spectrum(
    curve: ArrayView2<'_, f64>,
    r_floor: f64,
) -> Result<ReconSpectrum, String> {
    let (n_grid, ambient) = curve.dim();
    let gram = Array2::<f64>::eye(n_grid);
    let decoder = curve.to_owned();
    recon_spectrum(
        &gram,
        &decoder,
        n_grid as f64,
        ambient as f64,
        r_floor,
        0.0,
        None,
    )?
    // Grid-interpolation basis: one graded unit per reconstruction direction.
    .with_audit_basis_edf(1.0)
}

/// Grid-node delete-one jackknife SE of `λ = ½·rank_soft` at one checkpoint. Each
/// grid node is one observation of the decoder curve (the honest interpolation
/// design); leaving it out and recomputing `λ` gives the jackknife dispersion
/// `se² = (m−1)/m · Σ_i (λ_{(i)} − λ̄)²`. Deterministic (no RNG), and it reflects
/// genuine node-level variability of the complexity summary, not a placeholder.
/// Degenerate cases (too few nodes) return `0.0`, which the jump floor lifts.
fn lambda_jackknife_se(curve: ArrayView2<'_, f64>, r_floor: f64) -> f64 {
    let (n_grid, ambient) = curve.dim();
    if n_grid < 3 {
        return 0.0;
    }
    let mut leave_one = Vec::with_capacity(n_grid);
    for drop in 0..n_grid {
        let mut sub = Array2::<f64>::zeros((n_grid - 1, ambient));
        let mut r = 0usize;
        for i in 0..n_grid {
            if i == drop {
                continue;
            }
            for j in 0..ambient {
                sub[[r, j]] = curve[[i, j]];
            }
            r += 1;
        }
        match atom_learning_spectrum(sub.view(), r_floor) {
            Ok(spec) => leave_one.push(spec.learning_coefficient()),
            Err(_) => return 0.0,
        }
    }
    let m = leave_one.len() as f64;
    let mean = leave_one.iter().sum::<f64>() / m;
    let ss = leave_one
        .iter()
        .map(|&l| (l - mean) * (l - mean))
        .sum::<f64>();
    ((m - 1.0) / m * ss).max(0.0).sqrt()
}

/// Best-effort chart transport across one checkpoint step, reusing the layer
/// transport instrument (checkpoint axis as layer axis), exactly as
/// [`super::super::inference::checkpoint_dynamics`] does. `None` when the chart is
/// too short or degenerate to align — `λ_k` is chart-invariant, so this is a
/// diagnostic companion, never a hard dependency of the jump test.
fn best_effort_transport(
    grid: ArrayView4<'_, f64>,
    atom: usize,
    c0: usize,
    c1: usize,
) -> Option<LayerTransportReport> {
    let coords_from = grid.slice(ndarray::s![c0, atom, .., 0]).to_owned();
    let coords_to = grid.slice(ndarray::s![c1, atom, .., 0]).to_owned();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in coords_from.iter().chain(coords_to.iter()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let (lo, hi) = if hi <= lo {
        (lo - 0.5, lo + 0.5)
    } else {
        let pad = (hi - lo) * 1e-6;
        (lo - pad, hi + pad)
    };
    let topology = ChartTopology::Interval { lo, hi };
    fit_layer_transport(
        c0,
        c1,
        coords_from.view(),
        coords_to.view(),
        topology,
        topology,
    )
    .ok()
}

/// Anytime-valid log-e-value for the no-jump null `Δλ = 0` from the studentized
/// jump `z = Δλ / se` (local Gaussian `Δλ ~ N(Δλ_true, se²)`).
///
/// Mirrors [`super::super::inference::checkpoint_dynamics`]: the in-sample ratio
/// `exp(½ z²)` is NOT an e-value (its H0 expectation diverges), so we map the jump
/// to its two-sided normal p-value `p = 2(1 − Φ(|z|))` and route it through the
/// frozen κ = ½ p→e calibrator [`log_e_from_p_calibrator`], which satisfies
/// `E_{H0}[e(P)] ≤ 1` for any superuniform p — a genuine e-value that compounds
/// validly into the atom's birth e-process.
fn no_jump_log_e_value(z: f64) -> Result<f64, String> {
    if !z.is_finite() {
        return Ok(0.0);
    }
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| format!("standard normal construction failed: {e}"))?;
    let p: f64 = (2.0 * (1.0 - normal.cdf(z.abs()))).clamp(f64::MIN_POSITIVE, 1.0);
    log_e_from_p_calibrator(p)
}

/// Render the `λ_k(step)` trajectories to a plain-text block (for the demo / test
/// stderr). One row per atom: its `λ` at every checkpoint, then the confirmed
/// birth step (if any). No side effects; the caller decides where it goes.
pub fn render_lambda_dynamics(report: &WbicDynamicsReport, checkpoint_ids: &[String]) -> String {
    let mut out = String::new();
    out.push_str("atom                    ");
    for id in checkpoint_ids {
        out.push_str(&format!("{id:>8}"));
    }
    out.push_str("   birth@step\n");
    for (idx, atom) in report.atoms.iter().enumerate() {
        out.push_str(&format!("{:<23}", atom.atom_name));
        for l in &atom.lambda {
            out.push_str(&format!("{l:>8.3}"));
        }
        let birth = atom
            .jumps
            .iter()
            .find(|j| j.born)
            .map(|j| format!("  {}→{}", j.from_ckpt, j.to_ckpt))
            .unwrap_or_else(|| "  —".to_string());
        let cross = if report.cross_atom_born.contains(&idx) {
            " [born]"
        } else {
            ""
        };
        out.push_str(&format!("{birth}{cross}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

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

    /// Build an OLMo-style checkpoint fixture `[n_ckpt, n_atoms, n_grid, ambient]`
    /// with three archetypes of training dynamics:
    ///   * atom 0 STABLE-STRONG: a resolved rank-1 decoder present from the first
    ///     checkpoint (λ ≈ ½ throughout, no birth);
    ///   * atom 1 BORN: pure noise for the first `birth_ckpt` checkpoints, then a
    ///     strong rank-1 structure switches on (λ jumps ≈0 → ≈½ at `birth_ckpt`);
    ///   * atom 2 NULL: pure noise at every checkpoint (λ ≈ 0 throughout).
    fn olmo_birth_fixture(
        n_ckpt: usize,
        n_grid: usize,
        ambient: usize,
        birth_ckpt: usize,
    ) -> Array4<f64> {
        let mut s = 0xB1D_7000_u64;
        let mut grid = Array4::<f64>::zeros((n_ckpt, 3, n_grid, ambient));
        let noise = 0.02_f64;
        let signal = 1.0_f64;
        for c in 0..n_ckpt {
            for g in 0..n_grid {
                let t = g as f64 / (n_grid - 1) as f64;
                let a = std::f64::consts::TAU * t;
                for comp in 0..ambient {
                    // atom 0: a clean rank-1 cosine feature, every checkpoint.
                    grid[[c, 0, g, comp]] = if comp == 0 {
                        signal * a.cos()
                    } else {
                        noise * lcg_normal(&mut s)
                    };
                    // atom 1: noise until birth_ckpt, then the same clean feature.
                    grid[[c, 1, g, comp]] = if c >= birth_ckpt && comp == 0 {
                        signal * a.cos()
                    } else {
                        noise * lcg_normal(&mut s)
                    };
                    // atom 2: pure noise forever.
                    grid[[c, 2, g, comp]] = noise * lcg_normal(&mut s);
                }
            }
        }
        grid
    }

    fn run(grid: &Array4<f64>) -> (WbicDynamicsReport, Vec<String>) {
        let (n_ckpt, _n_atoms, _n_grid, _amb) = grid.dim();
        let ckpt_ids: Vec<String> = (0..n_ckpt).map(|c| format!("step{c}")).collect();
        let atom_names = vec![
            "stable-strong".to_string(),
            "born".to_string(),
            "null".to_string(),
        ];
        // Honest Marchenko–Pastur noise dispersion R = σ² of the decoder-value
        // noise. λ = ½·rank_soft has an inherent noise floor (the soft sigmoid
        // sums sub-edge directions), so the ROBUST birth signal is the JUMP Δλ
        // and the relative ordering, not an absolute-small λ — that is what the
        // assertions and the anytime-valid ledger key on.
        let input = WbicDynamicsInput {
            decoder_grid: grid.view(),
            checkpoint_ids: &ckpt_ids,
            atom_names: &atom_names,
            r_floor: 0.02_f64 * 0.02,
            birth_alpha: 0.05,
        };
        (wbic_lambda_dynamics(&input).expect("dynamics"), ckpt_ids)
    }

    /// THE DEMO. On the OLMo-style birth fixture the driver must (a) produce a
    /// `λ_k(step)` trajectory per atom, (b) detect the born atom's λ jump at the
    /// exact birth checkpoint via the anytime-valid ledger, and (c) NOT flag the
    /// stable-strong or null atoms — the whole point of the multiple-testing
    /// wrapper.
    #[test]
    fn detects_feature_birth_as_lambda_jump() {
        let n_ckpt = 6;
        let n_grid = 33; // ≥ MIN_TRANSPORT_OBS so the transport companion fits
        let ambient = 8;
        let birth_ckpt = 3;
        let grid = olmo_birth_fixture(n_ckpt, n_grid, ambient, birth_ckpt);
        let (report, ckpt_ids) = run(&grid);

        eprintln!("\n{}", render_lambda_dynamics(&report, &ckpt_ids));

        assert_eq!(report.atoms.len(), 3);
        let strong = &report.atoms[0];
        let born = &report.atoms[1];
        let null = &report.atoms[2];

        // (a) every atom carries a full λ_k(step) time series.
        for a in &report.atoms {
            assert_eq!(a.lambda.len(), n_ckpt);
            assert_eq!(a.jumps.len(), n_ckpt - 1);
        }

        // The born atom's λ rises sharply at the birth transition (the signal
        // direction crosses far above the MP edge). The birth-step jump must be
        // both large and the atom's dominant jump — the noise floor cancels in
        // the difference, so Δλ isolates the birth.
        let birth_jump = born.jumps[birth_ckpt - 1].delta_lambda;
        assert!(
            birth_jump > 0.25,
            "born atom must show a large positive λ jump at birth, got {birth_jump}"
        );
        let max_other_jump = born
            .jumps
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != birth_ckpt - 1)
            .map(|(_, j)| j.delta_lambda.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            birth_jump > max_other_jump + 0.1,
            "the birth jump ({birth_jump}) must dominate every other step jump ({max_other_jump})"
        );

        // (b) exactly one confirmed birth step, at exactly birth_ckpt-1→birth_ckpt.
        let born_steps: Vec<&LambdaJump> = born.jumps.iter().filter(|j| j.born).collect();
        assert_eq!(
            born_steps.len(),
            1,
            "born atom must have exactly one confirmed birth step, got {}",
            born_steps.len()
        );
        assert_eq!(
            born_steps[0].step,
            birth_ckpt - 1,
            "birth must be detected at the birth checkpoint transition"
        );
        assert!(
            born_steps[0].delta_lambda > 0.0,
            "a birth is a positive λ jump"
        );

        // (c) the stable-strong and null atoms are NOT flagged as born (no jump).
        assert!(
            strong.jumps.iter().all(|j| !j.born),
            "stable-strong atom must never be flagged born (λ flat): {:?}",
            strong.lambda
        );
        assert!(
            null.jumps.iter().all(|j| !j.born),
            "null atom must never be flagged born (λ ~ 0): {:?}",
            null.lambda
        );

        // The cross-atom e-BH selects the born atom and only the born atom.
        assert!(
            report.cross_atom_born.contains(&1),
            "cross-atom certificate must confirm the born atom, got {:?}",
            report.cross_atom_born
        );
        assert!(
            !report.cross_atom_born.contains(&0) && !report.cross_atom_born.contains(&2),
            "cross-atom certificate must not confirm the stable or null atoms, got {:?}",
            report.cross_atom_born
        );
    }

    /// The stable-strong atom must carry MORE complexity than the null atom at
    /// every checkpoint — λ_k is a genuine complexity readout, not a constant —
    /// and the born atom must cross from the null regime to the strong regime.
    #[test]
    fn lambda_orders_strong_above_null_and_born_crosses() {
        let grid = olmo_birth_fixture(6, 33, 8, 3);
        let (report, _ids) = run(&grid);
        let strong = &report.atoms[0];
        let born = &report.atoms[1];
        let null = &report.atoms[2];
        for c in 0..strong.lambda.len() {
            assert!(
                strong.lambda[c] > null.lambda[c] + 0.15,
                "strong λ ({}) must exceed null λ ({}) at checkpoint {c}",
                strong.lambda[c],
                null.lambda[c]
            );
        }
        // born crosses from the null band to the strong band.
        assert!(born.lambda[0] < strong.lambda[0]);
        assert!(*born.lambda.last().unwrap() > *null.lambda.last().unwrap() + 0.15);
    }

    /// The null atom accumulates NO change evidence (its e-process log-evidence
    /// stays ≤ 0), while the born atom accumulates strictly positive evidence —
    /// the anytime-valid discriminator is real, not a constant.
    #[test]
    fn born_out_evidences_null() {
        let grid = olmo_birth_fixture(6, 33, 8, 3);
        let (report, _ids) = run(&grid);
        let total_log_e = |a: &AtomLambdaTrajectory| -> f64 {
            a.birth_evidence
                .certify(0.05)
                .entries
                .iter()
                .map(|e| e.log_e)
                .sum()
        };
        let born_log_e = total_log_e(&report.atoms[1]);
        let null_log_e = total_log_e(&report.atoms[2]);
        assert!(
            born_log_e > null_log_e,
            "born change-evidence {born_log_e} must exceed null {null_log_e}"
        );
        assert!(
            born_log_e > 0.0,
            "a real birth must accumulate positive log-evidence, got {born_log_e}"
        );
    }

    #[test]
    fn rejects_single_checkpoint_and_bad_alpha() {
        let grid = Array4::<f64>::zeros((1, 2, 8, 3));
        let ids = vec!["only".to_string()];
        let names = vec!["a".to_string(), "b".to_string()];
        let bad = WbicDynamicsInput {
            decoder_grid: grid.view(),
            checkpoint_ids: &ids,
            atom_names: &names,
            r_floor: 0.01,
            birth_alpha: 0.05,
        };
        assert!(wbic_lambda_dynamics(&bad).is_err());

        let grid2 = Array4::<f64>::zeros((3, 2, 8, 3));
        let ids2 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let bad_alpha = WbicDynamicsInput {
            decoder_grid: grid2.view(),
            checkpoint_ids: &ids2,
            atom_names: &names,
            r_floor: 0.01,
            birth_alpha: 1.5,
        };
        assert!(wbic_lambda_dynamics(&bad_alpha).is_err());
    }
}
