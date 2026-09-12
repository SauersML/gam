//! The shattered-circle CENSUS: run [`super::curl`]'s witness statistics over an
//! arbitrary linear dictionary, including one this engine did not fit.
//!
//! # Why this seam has to exist
//!
//! [`super::curl`] proves that a mean-zero circle is *invisible* to any
//! residual-driven producer: its cone **is** its 2-plane, so two linear atoms
//! reconstruct it exactly and no residual is left to drive a birth. That claim is
//! an indictment of every dictionary trained by minimising reconstruction error —
//! which is to say, of the entire public SAE inventory, not only of this engine's
//! own intermediate fits. Testing it therefore requires running the witness
//! statistics against a FOREIGN dictionary: someone else's decoder matrix and
//! someone else's encoder's coefficients.
//!
//! The driver in [`crate::structure_harvest`] could not do that. It reads a fitted
//! [`crate::manifold::SaeManifoldTerm`] — this engine's atoms, this engine's
//! assignment logits — so the only way to census a foreign dictionary was to
//! transcribe the statistics somewhere else. A transcription is exactly where the
//! screen silently loses its calibration: the antipodal coalescing (without which
//! the move is a documented no-op on every nonnegative-gate dictionary), the
//! influence-function SE that makes κ a 2σ screen rather than a hand-picked
//! cutoff, the `2σ²` noise debiasing of the radius, and the `R̂ > σ·π/√3`
//! rate–distortion crossover are all easy to drop and impossible to miss the
//! absence of, because a screen with looser gates still prints a number.
//!
//! So the census lives here, dictionary-agnostic, and the fitted-term driver
//! calls it. One implementation of the acceptance rule, two callers.
//!
//! # What a caller supplies
//!
//! A list of [`AtomFrame`]s: per atom, a unit ambient direction, a per-row gate
//! mask, and the atom's per-row ambient IMAGE. The image is lazy
//! ([`AtomImage`]) because the foreign case is rank-one — atom `a` contributes
//! `coef[r]·dir` to row `r` — and materialising `K` dense `n×p` images for a
//! `K = 16384` dictionary is not a memory budget anyone has. This engine's own
//! atoms carry a genuinely dense image, so both spellings are first class.
//!
//! # What comes back
//!
//! Every screened pair's [`CurlVerdict`], not only the accepted ones. A census is
//! a distribution — the κ histogram against its matched null is the measurement;
//! the accepted set is a consequence. Plane geometry (the parse `α, β` and the
//! orthonormal frame) is retained only for accepted pairs, since that is the only
//! place it is consumed (seed construction) and keeping it for every screened pair
//! would make the census quadratic in memory as well as in pairs.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use rayon::prelude::*;

use super::curl::{
    CurlVerdict, coalesce_antipodal, cooccurrence_pairs_sparse, curl_verdict,
    ring_permutation_evidence, orthonormal_pair_coords,
};
use super::pair_phase::ebh_reject;

/// One atom's per-row ambient image, supplied lazily.
///
/// `Dense` is a materialised `n×p` block (this engine's fitted atoms, whose image
/// is not rank-one). `RankOne` is `coef[r] · dir` — the image of a linear
/// dictionary atom, which is what a foreign SAE provides and which must never be
/// expanded to `n×p` per atom.
pub enum AtomImage<'a> {
    /// A materialised `n×p` per-row image.
    Dense(ArrayView2<'a, f64>),
    /// The rank-one image `coef[r]·dir` of a linear dictionary atom.
    RankOne {
        /// Per-row coefficient (length `n`); zero means the atom did not fire.
        coef: ArrayView1<'a, f64>,
        /// Ambient decoder direction (length `p`); need not be unit.
        dir: ArrayView1<'a, f64>,
    },
}

impl AtomImage<'_> {
    /// Add this atom's contribution at ambient row `row` into `out` (length `p`).
    fn accumulate_row(&self, row: usize, out: &mut [f64]) {
        match self {
            AtomImage::Dense(img) => {
                for (j, slot) in out.iter_mut().enumerate() {
                    *slot += img[[row, j]];
                }
            }
            AtomImage::RankOne { coef, dir } => {
                let c = coef[row];
                if c == 0.0 {
                    return;
                }
                for (j, slot) in out.iter_mut().enumerate() {
                    *slot += c * dir[j];
                }
            }
        }
    }
}

/// One atom as the census reads it.
pub struct AtomFrame<'a> {
    /// The caller's atom index, carried through into the verdict rows.
    pub id: usize,
    /// Ambient direction (need not be unit; coalescing normalises).
    pub dir: Array1<f64>,
    /// Per-row gate mask (length `n`).
    pub active: Vec<bool>,
    /// Per-row ambient image.
    pub image: AtomImage<'a>,
}

/// Candidate-generation knobs. These bound the SEARCH (which pairs are looked at),
/// never the ACCEPTANCE — the accept/refuse rule is [`curl_verdict`]'s derived
/// conjunction and takes no configuration.
#[derive(Clone, Copy, Debug)]
pub struct CurlCensusConfig {
    /// Harmonic order the circle chart would be charged at.
    pub harmonics: usize,
    /// Decoder cosine at or below which two rectified halves coalesce.
    pub coalesce_cos_threshold: f64,
    /// Gate overlap at or below which two rectified halves coalesce.
    pub coalesce_max_overlap: f64,
    /// Minimum co-firing rows for a pair to be screened at all, counted over EVERY
    /// row (see [`cooccurrence_pairs_sparse`]). Set it by what the κ standard
    /// error needs, not by what a subsample can reach.
    pub min_cooccurrence: usize,
    /// Cap on the rows the plane parse is formed on. Co-firing rows above this
    /// count are strided down; the κ SE is already saturated well below it.
    pub subsample_rows: usize,
    /// Permutation surrogates per candidate plane, or `0` to DERIVE the only
    /// defensible value (which is what every caller should do; the override exists
    /// so two statistics can be compared at a fixed budget).
    ///
    /// The budget is not a precision knob to be set as large as patience allows.
    /// It is two-sided, and both sides are sharp:
    ///
    ///   * **Too small and no rejection is possible.** The indicator e-value tops
    ///     out at `B + 1`, and e-BH rejects at rank `k` only when the `k`-th
    ///     largest e-value clears `m/(α·k)`. Below `B + 1 = m/α` even rank 1 is out
    ///     of reach, and the census prints the same zero it would print for an
    ///     absence of structure.
    ///   * **Too large and real signal is destroyed.** The indicator pays `B + 1`
    ///     only when NO surrogate reaches the observation, and a plane whose true
    ///     p-value is `p` holds that with probability `≈ (1 − p)^B`. Pushing `B`
    ///     from `2·10⁵` to `10⁶` takes a `p = 10⁻⁵` plane from a 13% chance of
    ///     scoring at all to a 0.005% chance. More draws is not more evidence; past
    ///     a point it is less.
    ///
    /// `B + 1 = m/α` is exactly where those meet: the smallest budget at which
    /// EVERY rank is reachable (`B + 1 = m/α ≥ m/(α·k)` for all `k ≥ 1`), and
    /// therefore the largest one that buys anything. No knob, no taste — the size
    /// of the search fixes it.
    pub null_replicates: usize,
    /// Target false discovery rate for the e-BH ledger over all screened pairs.
    pub fdr_alpha: f64,
}

/// The plane geometry retained for an ACCEPTED pair, so a caller can build the
/// race-ready seed without re-projecting.
pub struct AcceptedPlane {
    /// Ambient row indices the parse was formed on.
    pub rows: Vec<usize>,
    /// First in-plane coordinate, one per row of `rows`.
    pub alpha: Array1<f64>,
    /// Second in-plane coordinate, one per row of `rows`.
    pub beta: Array1<f64>,
    /// Orthonormal plane frame, first axis.
    pub e1: Array1<f64>,
    /// Orthonormal plane frame, second axis.
    pub e2: Array1<f64>,
    /// Plane centre the parse was taken about.
    pub center: Array1<f64>,
}

/// One screened pair of signed directions.
pub struct CensusPair {
    /// Caller atom indices coalesced into the first signed direction.
    pub members_a: Vec<usize>,
    /// Caller atom indices coalesced into the second signed direction.
    pub members_b: Vec<usize>,
    /// Rows both signed directions fired on (after the subsample cap).
    pub n_co_fire: usize,
    /// The derived witness verdict.
    pub verdict: CurlVerdict,
    /// Exact lower-tail Monte-Carlo p-value of κ against the per-pair permutation
    /// null; `1.0` for pairs the deterministic screens already refused (no
    /// surrogates are drawn for those, and an e-value of `0` is what they carry
    /// into the ledger).
    pub p_value: f64,
    /// Indicator permutation e-value, `null_replicates + 1` when no surrogate
    /// reached the observed κ and `0` otherwise. Null mean `≤ 1` with no
    /// dependence assumption, which is what makes the e-BH ledger valid here.
    pub e_value: f64,
    /// Spearman `ρ` between `α²` and `β²` — `−1` for a ring, `0` under
    /// independence. The statistic the p-value and e-value are computed on.
    pub rank_rho: f64,
    /// Mean of `ρ` under that pair's own permutation null.
    pub null_rho_mean: f64,
    /// Standard deviation of `ρ` under that pair's own permutation null.
    pub null_rho_sd: f64,
    /// True ⇒ this pair is an e-BH discovery at [`CurlCensusConfig::fdr_alpha`]
    /// over the WHOLE screened family. This, not the per-pair screen, is what a
    /// census is entitled to call a finding.
    pub fdr_discovery: bool,
    /// Plane geometry, retained whenever the robust geometry gate passed.
    pub accepted_geometry: Option<AcceptedPlane>,
}

/// The census over one dictionary.
pub struct CurlCensus {
    /// The ambient noise scale the rate–distortion screen was run against.
    pub sigma: f64,
    /// Atoms supplied.
    pub n_frames: usize,
    /// Signed directions after antipodal coalescing.
    pub n_signed: usize,
    /// How many of those were MERGES of two rectified halves. A large count on a
    /// nonnegative-gate dictionary is the direct evidence that pairing raw atoms
    /// — the transcription this seam exists to retire — screens the wrong planes.
    pub n_coalesced: usize,
    /// Every screened pair, in candidate-generation order (co-firing count desc).
    pub pairs: Vec<CensusPair>,
    /// The e-BH level the ledger ran at.
    pub fdr_alpha: f64,
    /// Smallest e-value that would have been rejected — the ledger's own report of
    /// whether [`CurlCensusConfig::null_replicates`] was resolute enough to matter.
    pub ebh_threshold: f64,
    /// Pairs that beat EVERY surrogate, i.e. that attained the maximum e-value the
    /// replicate budget allows.
    pub n_max_e: usize,
    /// Surrogates actually drawn per candidate plane.
    pub replicates_drawn: usize,
    /// Replicates that would have been needed for those `n_max_e` pairs to be
    /// discoveries: `m/(α·n_max_e) − 1`.
    ///
    /// Without this a budget shortfall is indistinguishable from an absence of
    /// structure — both print zero discoveries. Compare it against
    /// [`CurlCensusConfig::null_replicates`]: if it is larger, the census did not
    /// measure anything about the data, it measured the budget.
    pub replicates_required: f64,
}

impl CurlCensus {
    /// e-BH discoveries: the pairs the census reports as shattered circles.
    pub fn accepted(&self) -> usize {
        self.pairs.iter().filter(|p| p.fdr_discovery).count()
    }
}

/// The MDL charge a circle chart pays over the two flat directions it would
/// replace, in nats: the BIC parameter charge `½·m·ln n_eff` for the chart's
/// `m = 2H+1` basis rows. Shared by every caller so the census and the fitted-term
/// harvest cannot price the same move differently.
fn circle_delta_charge_nats(harmonics: usize, n_eff: f64) -> f64 {
    let m_circle = (2 * harmonics + 1) as f64;
    0.5 * m_circle * n_eff.max(2.0).ln()
}

/// Census the shattered circles a linear dictionary is hiding.
///
/// Stages, in the order `super::curl` documents them: coalesce rectified
/// antipodal halves into signed directions, generate co-firing candidate planes
/// over a row subsample, project each candidate's joint parse, and adjudicate with
/// [`curl_verdict`] at the supplied ambient noise scale `sigma`.
///
/// `sigma` is the per-coordinate RMS of the dictionary's own reconstruction
/// residual — the noise floor the rate–distortion screen measures the ring radius
/// against. It is the caller's to compute because only the caller knows what its
/// dictionary reconstructs; both callers compute the same quantity the same way.
pub fn census_shattered_circles(
    frames: &[AtomFrame<'_>],
    n_rows: usize,
    ambient_p: usize,
    sigma: f64,
    cfg: &CurlCensusConfig,
) -> Result<CurlCensus, String> {
    if !(sigma > 0.0 && sigma.is_finite()) {
        return Err(format!(
            "curl census: sigma must be finite and > 0, got {sigma}"
        ));
    }
    if frames.len() < 2 {
        return Ok(CurlCensus {
            sigma,
            n_frames: frames.len(),
            n_signed: 0,
            n_coalesced: 0,
            pairs: Vec::new(),
            fdr_alpha: cfg.fdr_alpha,
            ebh_threshold: f64::INFINITY,
            n_max_e: 0,
            replicates_drawn: 0,
            replicates_required: f64::NAN,
        });
    }

    let dirs: Vec<ArrayView1<f64>> = frames.iter().map(|f| f.dir.view()).collect();
    let actives: Vec<Vec<bool>> = frames.iter().map(|f| f.active.clone()).collect();
    let ids: Vec<usize> = frames.iter().map(|f| f.id).collect();
    let signed = coalesce_antipodal(
        &dirs,
        &actives,
        &ids,
        cfg.coalesce_cos_threshold,
        cfg.coalesce_max_overlap,
    );
    let n_coalesced = signed.iter().filter(|s| s.members.len() > 1).count();
    if signed.len() < 2 {
        return Ok(CurlCensus {
            sigma,
            n_frames: frames.len(),
            n_signed: signed.len(),
            n_coalesced,
            pairs: Vec::new(),
            fdr_alpha: cfg.fdr_alpha,
            ebh_threshold: f64::INFINITY,
            n_max_e: 0,
            replicates_drawn: 0,
            replicates_required: f64::NAN,
        });
    }

    let frame_of: std::collections::HashMap<usize, usize> =
        ids.iter().enumerate().map(|(i, a)| (*a, i)).collect();
    let signed_active: Vec<Vec<bool>> = signed.iter().map(|s| s.active.clone()).collect();
    let candidate_pairs = cooccurrence_pairs_sparse(&signed_active, cfg.min_cooccurrence);
    // The derived budget: B + 1 = m/α, the smallest at which every e-BH rank is
    // reachable and the largest that buys anything. See `null_replicates`.
    let replicates = if cfg.null_replicates > 0 {
        cfg.null_replicates
    } else {
        ((candidate_pairs.len() as f64 / cfg.fdr_alpha).ceil() as usize)
            .saturating_sub(1)
            .max(2)
    };

    let out: Vec<CensusPair> = candidate_pairs
        .par_iter()
        .filter_map(|&(si, sj, _count)| {
            let di = &signed[si];
            let dj = &signed[sj];
            let mut co_fire: Vec<usize> = (0..n_rows)
                .filter(|&r| {
                    di.active.get(r).copied().unwrap_or(false)
                        && dj.active.get(r).copied().unwrap_or(false)
                })
                .collect();
            if co_fire.len() < cfg.min_cooccurrence.max(2) {
                return None;
            }
            if co_fire.len() > cfg.subsample_rows {
                let stride = (co_fire.len() / cfg.subsample_rows).max(1);
                co_fire = co_fire.iter().copied().step_by(stride).collect();
            }

            // The candidate plane's image is the SUM of both signed axes' member
            // atom images: the two directions' joint parse, isolated from the rest
            // of the dictionary's reconstruction.
            let members: Vec<usize> =
                di.members.iter().chain(dj.members.iter()).copied().collect();
            let mut x = Array2::<f64>::zeros((co_fire.len(), ambient_p));
            let mut acc = vec![0.0_f64; ambient_p];
            for (row_out, &r) in co_fire.iter().enumerate() {
                acc.iter_mut().for_each(|v| *v = 0.0);
                for &atom in &members {
                    if let Some(&fi) = frame_of.get(&atom) {
                        frames[fi].image.accumulate_row(r, &mut acc);
                    }
                }
                for (j, v) in acc.iter().enumerate() {
                    x[[row_out, j]] = *v;
                }
            }
            let mut center = Array1::<f64>::zeros(ambient_p);
            for row_out in 0..co_fire.len() {
                for j in 0..ambient_p {
                    center[j] += x[[row_out, j]];
                }
            }
            center.mapv_inplace(|v| v / co_fire.len() as f64);

            let (alpha, beta, e1, e2) = orthonormal_pair_coords(
                x.view(),
                di.dir.view(),
                dj.dir.view(),
                center.view(),
            )
            .ok()?;
            let n_eff = co_fire.len() as f64;
            let verdict = curl_verdict(
                alpha.view(),
                beta.view(),
                sigma,
                n_eff,
                circle_delta_charge_nats(cfg.harmonics, n_eff),
            )
            .ok()?;
            // Surrogates are drawn only for planes the deterministic screens
            // already accept. A refused plane carries `e = 0` into the ledger,
            // which is a valid e-value and keeps the multiplicity burden of the
            // WHOLE search on the books rather than quietly shrinking the family
            // to the candidates that happened to look good.
            let (p_value, e_value, rank_rho, null_rho_mean, null_rho_sd) = if verdict.geometry_ok
            {
                // Seed from the atom identities so the null is reproducible and
                // independent of how the pairs happened to be ordered.
                let seed = (di.members[0] as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (dj.members[0] as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                ring_permutation_evidence(alpha.view(), beta.view(), replicates, seed | 1)
                    .unwrap_or((1.0, 0.0, f64::NAN, f64::NAN, f64::NAN))
            } else {
                (1.0, 0.0, f64::NAN, f64::NAN, f64::NAN)
            };
            let accepted_geometry = if verdict.geometry_ok {
                Some(AcceptedPlane {
                    rows: co_fire,
                    alpha,
                    beta,
                    e1,
                    e2,
                    center,
                })
            } else {
                None
            };
            Some(CensusPair {
                members_a: di.members.clone(),
                members_b: dj.members.clone(),
                n_co_fire: n_eff as usize,
                verdict,
                p_value,
                e_value,
                rank_rho,
                null_rho_mean,
                null_rho_sd,
                fdr_discovery: false,
                accepted_geometry,
            })
        })
        .collect();

    // One e-BH ledger over the whole screened family. e-BH is valid under
    // ARBITRARY dependence between the pairs' e-values, which is the property this
    // census needs: candidate planes share atoms, share rows, and are anything but
    // independent.
    let mut out = out;
    let e_values: Vec<f64> = out.iter().map(|p| p.e_value).collect();
    let rejected = ebh_reject(&e_values, cfg.fdr_alpha);
    let m = e_values.len() as f64;
    let ebh_threshold = if rejected.is_empty() {
        f64::INFINITY
    } else {
        m / (cfg.fdr_alpha * rejected.len() as f64)
    };
    for i in rejected {
        out[i].fdr_discovery = true;
    }
    let n_max_e = out.iter().filter(|p| p.e_value > 0.0).count();
    let replicates_required = if n_max_e == 0 {
        f64::INFINITY
    } else {
        m / (cfg.fdr_alpha * n_max_e as f64) - 1.0
    };

    Ok(CurlCensus {
        sigma,
        n_frames: frames.len(),
        n_signed: signed.len(),
        n_coalesced,
        pairs: out,
        fdr_alpha: cfg.fdr_alpha,
        ebh_threshold,
        n_max_e,
        replicates_drawn: replicates,
        replicates_required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    /// Build a rank-one linear dictionary that has SHATTERED a planted circle the
    /// way a nonnegative-gate SAE does: four rectified half-atoms `±u, ±v`, each
    /// firing on the rows where its own half of the plane is positive.
    fn shattered_circle_frames(
        n: usize,
        p: usize,
        radius: f64,
        noise: f64,
    ) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut u = Array1::<f64>::zeros(p);
        let mut v = Array1::<f64>::zeros(p);
        u[0] = 1.0;
        v[1] = 1.0;
        // Deterministic low-discrepancy angles + a deterministic jitter, so the
        // test carries no RNG dependence.
        let mut coefs: Vec<Array1<f64>> = (0..4).map(|_| Array1::<f64>::zeros(n)).collect();
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let theta = TAU * t;
            let jitter = noise * ((i as f64 * 12.9898).sin() * 43758.5453).fract();
            let a = radius * theta.cos() + jitter;
            let b = radius * theta.sin() - jitter;
            // Rectified halves: +u, -u, +v, -v.
            coefs[0][i] = a.max(0.0);
            coefs[1][i] = (-a).max(0.0);
            coefs[2][i] = b.max(0.0);
            coefs[3][i] = (-b).max(0.0);
        }
        let dirs = vec![u.clone(), u.mapv(|z| -z), v.clone(), v.mapv(|z| -z)];
        (dirs, coefs)
    }

    fn frames_from<'a>(
        dirs: &'a [Array1<f64>],
        coefs: &'a [Array1<f64>],
    ) -> Vec<AtomFrame<'a>> {
        dirs.iter()
            .zip(coefs.iter())
            .enumerate()
            .map(|(id, (dir, coef))| AtomFrame {
                id,
                dir: dir.clone(),
                active: coef.iter().map(|&c| c > 0.0).collect(),
                image: AtomImage::RankOne {
                    coef: coef.view(),
                    dir: dir.view(),
                },
            })
            .collect()
    }

    fn cfg() -> CurlCensusConfig {
        CurlCensusConfig {
            harmonics: 1,
            coalesce_cos_threshold: -0.9,
            coalesce_max_overlap: 0.1,
            min_cooccurrence: 40,
            subsample_rows: 4000,
            // Two planted-ring pairs against a family of a handful: e-BH needs
            // `e >= m/(alpha*rank)`, so a few hundred draws is already resolute.
            null_replicates: 2000,
            fdr_alpha: 0.05,
        }
    }

    /// The launch-blocker claim, made executable: on a nonnegative-gate
    /// dictionary the four rectified halves coalesce into two signed axes, and
    /// only then does the plane the circle actually lives in get screened.
    #[test]
    fn coalescing_recovers_the_signed_plane_of_a_shattered_circle() {
        let (dirs, coefs) = shattered_circle_frames(2000, 8, 3.0, 0.05);
        let frames = frames_from(&dirs, &coefs);
        let census = census_shattered_circles(&frames, 2000, 8, 0.05, &cfg())
            .expect("census must run on a well-formed shattered dictionary");
        assert_eq!(
            census.n_coalesced, 2,
            "the four rectified halves must coalesce into exactly two signed axes, got {} \
             (n_signed = {})",
            census.n_coalesced, census.n_signed
        );
        assert_eq!(
            census.n_signed, 2,
            "no half-atom should ride unpaired here, got {}",
            census.n_signed
        );
        assert!(
            census.accepted() >= 1,
            "the coalesced plane of a planted ring must be accepted; verdicts: {:?}",
            census
                .pairs
                .iter()
                .map(|p| (p.verdict.kappa, p.verdict.z_below_gaussian, p.verdict.radius))
                .collect::<Vec<_>>()
        );
    }

    /// The census must REFUSE an isotropic Gaussian plane: κ sits at the
    /// Gaussian-fill value 2, so the 2σ screen cannot clear it. This is the arm
    /// that makes an acceptance mean something.
    #[test]
    fn gaussian_fill_is_refused() {
        let n = 2000;
        let p = 8;
        let mut coefs: Vec<Array1<f64>> = (0..4).map(|_| Array1::<f64>::zeros(n)).collect();
        // A deterministic Box–Muller pair from a low-discrepancy sequence: an
        // isotropic 2-D Gaussian, whose radius law is κ = 2 by construction.
        for i in 0..n {
            let u1 = ((i as f64 + 0.5) / n as f64).max(1e-12);
            let u2 = ((i as f64 * 0.6180339887).fract() + 0.5 / n as f64).min(1.0 - 1e-12);
            let r = (-2.0 * u1.ln()).sqrt();
            let a = r * (TAU * u2).cos();
            let b = r * (TAU * u2).sin();
            coefs[0][i] = a.max(0.0);
            coefs[1][i] = (-a).max(0.0);
            coefs[2][i] = b.max(0.0);
            coefs[3][i] = (-b).max(0.0);
        }
        let mut u = Array1::<f64>::zeros(p);
        let mut v = Array1::<f64>::zeros(p);
        u[0] = 1.0;
        v[1] = 1.0;
        let dirs = vec![u.clone(), u.mapv(|z| -z), v.clone(), v.mapv(|z| -z)];
        let frames = frames_from(&dirs, &coefs);
        let census = census_shattered_circles(&frames, n, p, 0.05, &cfg())
            .expect("census must run on the Gaussian-fill arm");
        assert_eq!(
            census.accepted(),
            0,
            "an isotropic Gaussian plane must not be read as a ring; κ values: {:?}",
            census
                .pairs
                .iter()
                .map(|p| (p.verdict.kappa, p.verdict.z_below_gaussian))
                .collect::<Vec<_>>()
        );
    }

    /// Without coalescing, the SAME shattered dictionary yields no accepted plane:
    /// pairing raw rectified halves screens a quarter-turn wedge, whose angles
    /// cannot cover the circle. This is the defect a transcription of the screen
    /// silently ships.
    #[test]
    fn raw_half_atom_pairing_misses_the_circle_entirely() {
        let (dirs, coefs) = shattered_circle_frames(2000, 8, 3.0, 0.05);
        let frames = frames_from(&dirs, &coefs);
        // Disable coalescing by demanding an impossible cosine: every half-atom
        // then rides as its own "signed" direction, exactly as a raw-atom census
        // would pair them.
        let mut no_coalesce = cfg();
        no_coalesce.coalesce_cos_threshold = -1.5;
        let census = census_shattered_circles(&frames, 2000, 8, 0.05, &no_coalesce)
            .expect("census must run with coalescing disabled");
        assert_eq!(
            census.n_coalesced, 0,
            "the control arm must not coalesce anything"
        );
        assert_eq!(
            census.accepted(),
            0,
            "raw half-atom pairs must not recover the circle; that they cannot is why \
             coalescing is a launch blocker, not a refinement"
        );
    }
}
