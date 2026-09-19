//! Certified compression of a declared latent law with many atoms (gam#2928).
//!
//! A caller that declares its score's law one atom per training row — gnomon's
//! calibrate fixtures do — hands the anchor a law of `K = n` nodes, and every
//! residual evaluation and every Taylor table of [`crate::latent_anchor`] is
//! `O(K)`, so an iterate costs `O(n·K)`. A law of `n` atoms resolves the anchor
//! only to its own sampling error, so this module replaces the sorted atoms by
//! a quadrature whose anchor provably sits within a small fraction of that
//! resolution of the exact one.
//!
//! **Rule.** Contiguous bins of the sorted atoms, each replaced by the two-point
//! Gauss rule of its own discrete measure: the roots of the bin's second
//! orthogonal polynomial `p₂(y) = y² − (t/s²)·y − s²` (`y = u − μ`; `s²`, `t` the
//! bin's variance and third central moment), with positive weights, nodes
//! inside the bin, exact through degree three. A bin of at most two atoms is its
//! own rule and keeps its atoms. The compressed law is again a law with positive
//! weights summing to one, so the anchoring residual stays strictly monotone
//! and its root unique.
//!
//! **Bound.** With `f(u) = F(α + b·u)` and `F = Φ(∓·)`, Gauss's error formula
//! for a positive measure gives, per bin,
//!
//! ```text
//!     ∫ f dν_j − Σ_G f = f⁗(ξ)/4! · ∫ p₂² dν_j,        ξ ∈ [lo_j, hi_j],
//! ```
//!
//! and `f⁗(u) = b⁴·F⁗(η)` with `|F⁗(x)| = |x³ − 3x|·φ(x)`, so the marginal tails
//! of the two laws differ by at most
//!
//! ```text
//!     E(α, b) = Σ_j W_j·m_j·b⁴·max_{x ∈ α + b·[lo_j, hi_j]} |x³ − 3x|·φ(x) / 24,     m_j = ∫ p₂² dν_j / W_j.
//! ```
//!
//! Each term is formed in log space over its own interval, so the bound stays
//! relative where the marginal tail is tiny.
//!
//! **Certificate.** The exact tail is monotone in `α`, so where `T_K ∓ E`
//! straddles the target at `α_K ∓ Δ` the exact root lies in `[α_K − Δ, α_K + Δ]`.
//! `Δ = 2E/D` plus the solve's own residual slack (`D = Σ_k w_k φ(η_k)`) usually
//! straddles, and is doubled until it does ([`CompressedLaw::certify`]).
//!
//! **Target.** `Δ ≤ 10⁻³·SE_α`, `SE_α = sd_w(Φ(∓η))/(√n_eff·D)` with
//! `n_eff = 1/Σ w²` of the atoms: a thousandth of what resampling the atoms
//! would move the anchor by.
//!
//! **K.** Bins are refined greedily against design points `(q, b)`: at the point
//! furthest over its target, the bin contributing most to its bound is split at
//! the midpoint of its range, until every point meets half its target — the
//! other half pays for one doubling of `Δ` in the certificate. Dense regions end
//! in near-equal widths; sparse tails end in exact atoms where they matter.

use gam_math::probability::{normal_logcdf, signed_probit_logcdf_and_mills_ratio};
use rayon::prelude::*;

use crate::latent_anchor::{AnchorGrid, AnchorGridOwned, solve_anchor};

/// The certified anchor error a compression must meet, as a fraction of the
/// anchor's sampling standard error on the declared atoms.
pub(crate) const ANCHOR_ERROR_FRACTION_OF_SE: f64 = 1e-3;

/// Doublings of `Δ` a certificate tries before it reports the anchor
/// uncertified.
const CERTIFICATE_DOUBLINGS: usize = 40;

/// Missed anchors a refinement adds as design points, worst first: every
/// refinement at least resolves the anchors furthest over their target.
const REFINEMENT_POINTS: usize = 64;

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// `√(3 − √6)` and `√(3 + √6)`: the critical points of `|x³ − 3x|·φ(x)`.
const HERMITE3_DENSITY_CRITICAL: [f64; 4] = [
    -2.334_414_218_338_977_3,
    -0.741_963_784_302_725_8,
    0.741_963_784_302_725_8,
    2.334_414_218_338_977_3,
];

/// A `(q, b)` at which a compression is designed to meet its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DesignPoint {
    pub(crate) q: f64,
    pub(crate) slope: f64,
}

/// The operating range a compression is designed over before any fit has run:
/// marginal indices through `|q| = 8` (tail probabilities to about 6e-16) and
/// observed slopes through `|b| = 4`, both signs. A fit whose rows leave it is
/// certified at its own rows and refined there.
pub(crate) fn default_design() -> Vec<DesignPoint> {
    const Q: [f64; 13] = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0];
    const B: [f64; 8] = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0];
    Q.iter()
        .flat_map(|&q| B.iter().map(move |&slope| DesignPoint { q, slope }))
        .collect()
}

/// One contiguous bin of the sorted atoms and what its bound reads.
#[derive(Clone, Debug)]
struct LawBin {
    start: usize,
    end: usize,
    weight: f64,
    mean: f64,
    variance: f64,
    third: f64,
    /// `log(∫ p₂² dν / W)`; `−∞` for a bin its rule reproduces exactly.
    log_gauss_moment: f64,
    lo: f64,
    hi: f64,
}

impl LawBin {
    fn from_atoms(nodes: &[f64], weights: &[f64], start: usize, end: usize) -> Self {
        let atoms = &nodes[start..end];
        let masses = &weights[start..end];
        let weight: f64 = masses.iter().sum();
        let mean = atoms.iter().zip(masses).map(|(u, w)| u * w).sum::<f64>() / weight;
        let mut variance = 0.0;
        let mut third = 0.0;
        for (&u, &w) in atoms.iter().zip(masses) {
            let y = u - mean;
            variance += w * y * y;
            third += w * y * y * y;
        }
        variance /= weight;
        third /= weight;
        let lo = atoms[0];
        let hi = atoms[atoms.len() - 1];
        let mut bin = Self {
            start,
            end,
            weight,
            mean,
            variance,
            third,
            log_gauss_moment: f64::NEG_INFINITY,
            lo,
            hi,
        };
        if bin.two_point_rule().is_some() {
            let tilt = third / variance;
            let moment = atoms
                .iter()
                .zip(masses)
                .map(|(&u, &w)| {
                    let y = u - mean;
                    let p = y * y - tilt * y - variance;
                    w * p * p
                })
                .sum::<f64>()
                / weight;
            if moment > 0.0 {
                bin.log_gauss_moment = moment.ln();
            }
        }
        bin
    }

    fn atoms(&self) -> usize {
        self.end - self.start
    }

    fn splittable(&self) -> bool {
        self.atoms() > 2 && self.hi > self.lo
    }

    /// The bin's two-point Gauss rule `[(node, weight); 2]`, or `None` where the
    /// bin keeps its atoms: at most two of them, or a rule rounding cannot place
    /// strictly inside the bin in increasing order.
    fn two_point_rule(&self) -> Option<[(f64, f64); 2]> {
        if self.atoms() <= 2 || !(self.variance > 0.0) || !(self.hi > self.lo) {
            return None;
        }
        let tilt = self.third / self.variance;
        let spread = (tilt * tilt + 4.0 * self.variance).sqrt();
        let upper = 0.5 * (tilt + spread);
        let lower = 0.5 * (tilt - spread);
        let (low_node, high_node) = (self.mean + lower, self.mean + upper);
        let low_weight = self.weight * upper / (upper - lower);
        let high_weight = self.weight * (-lower) / (upper - lower);
        let inside = self.lo <= low_node && low_node < high_node && high_node <= self.hi;
        (inside && low_weight > 0.0 && high_weight > 0.0 && low_weight.is_finite() && high_weight.is_finite())
            .then_some([(low_node, low_weight), (high_node, high_weight)])
    }

    /// `log` of this bin's term of `E(α, b)`.
    fn log_error(&self, alpha: f64, slope: f64) -> f64 {
        if self.log_gauss_moment == f64::NEG_INFINITY || slope == 0.0 {
            return f64::NEG_INFINITY;
        }
        let a = alpha + slope * self.lo;
        let b = alpha + slope * self.hi;
        self.weight.ln() + self.log_gauss_moment + 4.0 * slope.abs().ln() - 24.0_f64.ln()
            + log_max_hermite3_density(a.min(b), a.max(b))
    }
}

/// `log max_{x ∈ [lo, hi]} |x³ − 3x|·φ(x)`.
fn log_max_hermite3_density(lo: f64, hi: f64) -> f64 {
    let at = |x: f64| (x * x * x - 3.0 * x).abs().ln() - 0.5 * x * x - LN_SQRT_2PI;
    let mut best = at(lo).max(at(hi));
    for &x in &HERMITE3_DENSITY_CRITICAL {
        if lo < x && x < hi {
            best = best.max(at(x));
        }
    }
    best
}

fn log_sum_exp(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let max = values.clone().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values.map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// `log(1 + e^x)`.
fn log1p_exp(x: f64) -> f64 {
    if x > 36.0 { x } else { x.exp().ln_1p() }
}

/// A law's marginal tail at `(α, b)` on the smaller side, in log space.
struct TailState {
    log_tail: f64,
    /// `D/T = Σ_k w_k φ(η_k) / T`.
    density_over_tail: f64,
    /// `Σ_k w_k (Φ(∓η_k)/T − 1)²`.
    relative_variance: f64,
}

fn tail_state(alpha: f64, slope: f64, grid: AnchorGrid<'_>, survival_side: bool) -> Option<TailState> {
    let m = grid.len();
    let mut log_terms = Vec::with_capacity(m);
    let mut mills = Vec::with_capacity(m);
    let mut log_max = f64::NEG_INFINITY;
    for k in 0..m {
        let eta = alpha + slope * grid.nodes[k];
        let (log_cdf, ratio) = signed_probit_logcdf_and_mills_ratio(if survival_side { -eta } else { eta });
        let term = grid.log_weights[k] + log_cdf;
        log_max = log_max.max(term);
        log_terms.push(term);
        mills.push(ratio);
    }
    if !log_max.is_finite() {
        return None;
    }
    let sum: f64 = log_terms.iter().map(|t| (t - log_max).exp()).sum();
    let log_tail = log_max + sum.ln();
    let mut density_over_tail = 0.0;
    let mut second = 0.0;
    for k in 0..m {
        let share = (log_terms[k] - log_tail).exp();
        density_over_tail += share * mills[k];
        second += share * (log_terms[k] - grid.log_weights[k] - log_tail).exp();
    }
    Some(TailState {
        log_tail,
        density_over_tail,
        relative_variance: (second - 1.0).max(0.0),
    })
}

fn log_target(q: f64) -> f64 {
    if q >= 0.0 { normal_logcdf(-q) } else { normal_logcdf(q) }
}

/// `SE_α = sd_w(Φ(∓η))/(√n_eff·D)` from a tail state.
fn standard_error(state: &TailState, effective_atoms: f64) -> f64 {
    state.relative_variance.sqrt() / (effective_atoms.sqrt() * state.density_over_tail)
}

/// A compressed declared law, with what its certificates read.
#[derive(Clone, Debug)]
pub(crate) struct CompressedLaw {
    grid: AnchorGridOwned,
    /// The declared atoms, which an audit sample of the certificates is
    /// checked against.
    declared: AnchorGridOwned,
    bins: Vec<LawBin>,
    atoms: usize,
    effective_atoms: f64,
}

/// What a certificate says about one anchor on a compressed law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompressionCertificate {
    /// `|α_n − α_K| ≤ delta` when `certified`.
    pub(crate) delta: f64,
    /// The anchor's sampling standard error on the declared atoms.
    pub(crate) standard_error: f64,
    pub(crate) certified: bool,
}

impl CompressionCertificate {
    /// Whether the certified error meets [`ANCHOR_ERROR_FRACTION_OF_SE`].
    pub(crate) fn meets_target(&self) -> bool {
        self.certified && self.delta <= ANCHOR_ERROR_FRACTION_OF_SE * self.standard_error
    }
}

impl CompressedLaw {
    /// Compress a declared law, designed over `design`. `None` where the law is
    /// anchored as declared: where the certified node count is more than half
    /// the atoms. Compression exists to cut the anchor's `O(K)` work, and a
    /// declared law gives up its exactness only where that work at least
    /// halves, so small laws stay exact and bit-identical because their
    /// certified compression cannot halve them, not by a threshold.
    pub(crate) fn compress(
        nodes: &[f64],
        weights: &[f64],
        design: &[DesignPoint],
    ) -> Result<Option<Self>, String> {
        let n = nodes.len();
        if design.is_empty() {
            return Err(format!(
                "declared latent law compression of {n} atoms needs design points: with none, \
                 nothing bounds the anchor error and every atom would merge into one bin"
            ));
        }
        if weights.len() != n || nodes.windows(2).any(|pair| !(pair[0] <= pair[1])) {
            return Err(format!(
                "declared latent law compression needs ascending nodes with one weight each \
                 (got {n} nodes, {} weights)",
                weights.len()
            ));
        }
        let exact = AnchorGridOwned::new(nodes.to_vec(), weights.to_vec());
        let effective_atoms = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        // Each design point's exact anchor and the bound its half target allows:
        // `2E/D ≤ ½·10⁻³·SE_α`, i.e. `E ≤ 10⁻³·SE_α·D/4`, in log space.
        let points = design
            .par_iter()
            .map(|point| -> Result<(f64, f64, f64), String> {
                let alpha = solve_anchor(point.q, point.slope, exact.view())?;
                let state = tail_state(alpha, point.slope, exact.view(), point.q >= 0.0).ok_or_else(|| {
                    format!(
                        "declared latent law compression: non-finite marginal tail at design point \
                         q={}, b={}",
                        point.q, point.slope
                    )
                })?;
                let se = standard_error(&state, effective_atoms);
                let log_allowed = (ANCHOR_ERROR_FRACTION_OF_SE / 4.0).ln()
                    + se.ln()
                    + state.density_over_tail.ln()
                    + state.log_tail;
                Ok((alpha, point.slope, log_allowed))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut bins = vec![LawBin::from_atoms(nodes, weights, 0, n)];
        let contribution = |bin: &LawBin| -> Vec<f64> {
            points.iter().map(|&(alpha, slope, _)| bin.log_error(alpha, slope)).collect()
        };
        let mut contributions = vec![contribution(&bins[0])];
        loop {
            let worst = (0..points.len())
                .map(|p| (p, log_sum_exp(contributions.iter().map(|c| c[p])) - points[p].2))
                .fold(None, |best: Option<(usize, f64)>, (p, excess)| match best {
                    Some((_, top)) if top >= excess => best,
                    _ => Some((p, excess)),
                });
            let Some((point, excess)) = worst else {
                break;
            };
            if excess <= 0.0 {
                break;
            }
            let Some(split) = (0..bins.len())
                .filter(|&j| bins[j].splittable())
                .max_by(|&a, &b| contributions[a][point].total_cmp(&contributions[b][point]))
            else {
                break;
            };
            let bin = bins[split].clone();
            let midpoint = bin.lo + 0.5 * (bin.hi - bin.lo);
            let cut = bin.start + nodes[bin.start..bin.end].partition_point(|&u| u <= midpoint);
            let left = LawBin::from_atoms(nodes, weights, bin.start, cut);
            let right = LawBin::from_atoms(nodes, weights, cut, bin.end);
            contributions[split] = contribution(&left);
            contributions.insert(split + 1, contribution(&right));
            bins[split] = left;
            bins.insert(split + 1, right);
            // Every bin keeps at least one node, so more than `n/2` bins can
            // no longer halve the law.
            if 2 * bins.len() > n {
                return Ok(None);
            }
        }
        let mut compressed_nodes = Vec::with_capacity(2 * bins.len());
        let mut compressed_weights = Vec::with_capacity(2 * bins.len());
        for bin in &bins {
            match bin.two_point_rule() {
                Some(rule) => {
                    for (node, weight) in rule {
                        compressed_nodes.push(node);
                        compressed_weights.push(weight);
                    }
                }
                None => {
                    compressed_nodes.extend_from_slice(&nodes[bin.start..bin.end]);
                    compressed_weights.extend_from_slice(&weights[bin.start..bin.end]);
                }
            }
        }
        if 2 * compressed_nodes.len() > n {
            return Ok(None);
        }
        Ok(Some(Self {
            grid: AnchorGridOwned::new(compressed_nodes, compressed_weights),
            declared: exact,
            bins,
            atoms: n,
            effective_atoms,
        }))
    }

    /// The compressed law the anchor solves on.
    pub(crate) fn grid(&self) -> &AnchorGridOwned {
        &self.grid
    }

    /// Atoms of the declared law.
    pub(crate) fn atoms(&self) -> usize {
        self.atoms
    }

    /// Bins the atoms were grouped into.
    pub(crate) fn bins(&self) -> usize {
        self.bins.len()
    }

    /// `log E(α, b)`.
    fn log_error_bound(&self, alpha: f64, slope: f64) -> f64 {
        log_sum_exp(self.bins.iter().map(|bin| bin.log_error(alpha, slope)))
    }

    /// Certify the anchor `root` solved on the compressed law at `(q, b)`: the
    /// smallest `Δ = 2^i·(2E/D + |F|/|F′|)` at which the compressed tail, widened
    /// by the bound, straddles the target on both sides — so the exact root on
    /// the declared atoms lies within `Δ` of `root` — and the anchor's sampling
    /// standard error on those atoms.
    pub(crate) fn certify(&self, q: f64, slope: f64, root: f64) -> CompressionCertificate {
        let uncertified = CompressionCertificate {
            delta: f64::INFINITY,
            standard_error: f64::NAN,
            certified: false,
        };
        let survival_side = q >= 0.0;
        let target = log_target(q);
        let Some(state) = tail_state(root, slope, self.grid.view(), survival_side) else {
            return uncertified;
        };
        let standard_error = standard_error(&state, self.effective_atoms);
        let log_bound = self.log_error_bound(root, slope);
        if log_bound == f64::NEG_INFINITY && state.log_tail == target {
            return CompressionCertificate {
                delta: 0.0,
                standard_error,
                certified: true,
            };
        }
        // `|F′| = D/T`; the solve's slack in `α` is `|F|/|F′|`.
        let slack = (state.log_tail - target).abs() / state.density_over_tail;
        let bound_step = 2.0 * (log_bound - state.log_tail).exp() / state.density_over_tail;
        // Where the bound is about zero (exact bins deep in a tail) the first
        // step would sit below `α`'s own float resolution, where `α ± Δ` is `α`
        // and no straddle can show; start at a few ulp of `α` so the doublings
        // reach resolvable steps.
        let mut delta = (bound_step + slack).max(4.0 * f64::EPSILON * (1.0 + root.abs()));
        // Where `α` moves the tail up (the complement side) the root lies above
        // `α` exactly when the widened tail at `α` is below the target.
        let straddles = |alpha: f64, tail_above_target: bool| -> bool {
            let Some(shifted) = tail_state(alpha, slope, self.grid.view(), survival_side) else {
                return false;
            };
            let log_relative_error = self.log_error_bound(alpha, slope) - shifted.log_tail;
            if tail_above_target {
                log_relative_error < 0.0
                    && shifted.log_tail + (-log_relative_error.exp()).ln_1p() > target
            } else {
                shifted.log_tail + log1p_exp(log_relative_error) < target
            }
        };
        for _ in 0..CERTIFICATE_DOUBLINGS {
            let (below, above) = (root - delta, root + delta);
            // Survival side: the tail falls in `α`, so it must sit above the
            // target below the root and under it above; the complement side is
            // the mirror.
            let certified = if survival_side {
                straddles(below, true) && straddles(above, false)
            } else {
                straddles(below, false) && straddles(above, true)
            };
            if certified {
                return CompressionCertificate {
                    delta,
                    standard_error,
                    certified: true,
                };
            }
            delta *= 2.0;
        }
        CompressionCertificate {
            standard_error,
            ..uncertified
        }
    }

    /// Certify every anchor a fit on this law solved, at its converged inputs
    /// `(q, b)`: the record the fit's ledger carries, and the inputs of the
    /// anchors that missed [`ANCHOR_ERROR_FRACTION_OF_SE`] — the design points a
    /// refined compression must add.
    pub(crate) fn certify_anchors(
        &self,
        anchors: &[(f64, f64)],
    ) -> Result<(DeclaredLawCompressionRecord, Vec<DesignPoint>), String> {
        let certificates = anchors
            .par_iter()
            .map(|&(q, slope)| -> Result<(DesignPoint, f64, CompressionCertificate), String> {
                let root = solve_anchor(q, slope, self.grid.view())?;
                Ok((DesignPoint { q, slope }, root, self.certify(q, slope, root)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // The audit sample: anchors evenly spaced through the caller's order,
        // both ends included, solved again on the declared atoms. The caller
        // orders anchors by `q`, so the sample reaches both tails.
        let sample: Vec<usize> = if certificates.len() <= AUDIT_ANCHORS {
            (0..certificates.len()).collect()
        } else {
            (0..AUDIT_ANCHORS)
                .map(|i| i * (certificates.len() - 1) / (AUDIT_ANCHORS - 1))
                .collect()
        };
        // Beside the converged anchors, extreme-tail anchors in both tails at
        // the smallest and largest converged slope: marginal indices through
        // |q| = 30, far past any row, where a compression's error is largest
        // relative to the tail. They audit the certificate's soundness only;
        // no target applies there.
        let slopes = certificates
            .iter()
            .map(|(point, _, _)| point.slope)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), b| (lo.min(b), hi.max(b)));
        let mut audited: Vec<(DesignPoint, f64, CompressionCertificate)> =
            sample.iter().map(|&i| certificates[i]).collect();
        if slopes.0.is_finite() {
            // A time-constant slope makes both ends one slope: audit it once.
            let mut tail_slopes = vec![slopes.0];
            if slopes.1 != slopes.0 {
                tail_slopes.push(slopes.1);
            }
            let tail = tail_slopes
                .into_iter()
                .flat_map(|slope| AUDIT_TAIL_Q.iter().map(move |&q| DesignPoint { q, slope }))
                .collect::<Vec<_>>()
                .par_iter()
                .map(|&point| -> Result<(DesignPoint, f64, CompressionCertificate), String> {
                    let root = solve_anchor(point.q, point.slope, self.grid.view())?;
                    Ok((point, root, self.certify(point.q, point.slope, root)))
                })
                .collect::<Result<Vec<_>, String>>()?;
            audited.extend(tail);
        }
        let mut measured_over_bound = audited
            .par_iter()
            .map(|(point, root, certificate)| -> Result<f64, String> {
                let exact = solve_anchor(point.q, point.slope, self.declared.view())?;
                let measured = (exact - root).abs();
                // An uncertified anchor has no bound to hold, and must show at
                // the top of the sample rather than as a perfect zero.
                Ok(if !certificate.certified {
                    f64::INFINITY
                } else if measured == 0.0 {
                    0.0
                } else {
                    measured / certificate.delta
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        for ((point, root, certificate), ratio) in
            audited.iter().zip(&measured_over_bound).skip(sample.len())
        {
            log::info!(
                "[survival-marginal-slope latent-z] compression audit tail anchor q={} b={:e}: root {:e}, \
                 certified error {:e} ({}), measured / certified {:e} (gam#2928)",
                point.q,
                point.slope,
                root,
                certificate.delta,
                if certificate.certified { "certified" } else { "uncertified" },
                ratio,
            );
        }
        let mut record = DeclaredLawCompressionRecord {
            atoms: self.atoms,
            bins: self.bins.len(),
            nodes: self.grid.nodes.len(),
            anchors_checked: certificates.len(),
            anchors_meeting_target: 0,
            max_delta: 0.0,
            max_delta_over_standard_error: 0.0,
            bound_over_target: [f64::NAN; 3],
            measured_over_bound: min_median_max(&mut measured_over_bound),
            anchors_audited: audited.len(),
        };
        let mut over_target = Vec::with_capacity(certificates.len());
        let mut missed = Vec::new();
        for (point, _, certificate) in certificates {
            // An anchor that could not be certified has no finite ratio.
            let (ratio, target_ratio) = if certificate.certified {
                (
                    certificate.delta / certificate.standard_error,
                    certificate.delta / (ANCHOR_ERROR_FRACTION_OF_SE * certificate.standard_error),
                )
            } else {
                (f64::INFINITY, f64::INFINITY)
            };
            record.max_delta = record.max_delta.max(certificate.delta);
            record.max_delta_over_standard_error = record.max_delta_over_standard_error.max(ratio);
            over_target.push(target_ratio);
            if certificate.meets_target() {
                record.anchors_meeting_target += 1;
            } else {
                missed.push((ratio, point));
            }
        }
        record.bound_over_target = min_median_max(&mut over_target);
        missed.sort_by(|a, b| b.0.total_cmp(&a.0));
        missed.truncate(REFINEMENT_POINTS);
        Ok((record, missed.into_iter().map(|(_, point)| point).collect()))
    }
}

/// Anchors of a converged fit a certificate pass solves again on the declared
/// atoms, so the record carries measured errors beside the certified ones.
const AUDIT_ANCHORS: usize = 64;

/// Marginal indices of the extreme-tail audit anchors, both tails: `|q| = 8`
/// (tail probabilities about 6e-16, the design range's edge) and `|q| = 30`
/// (about 5e-198, far past any row).
const AUDIT_TAIL_Q: [f64; 4] = [-30.0, -8.0, 8.0, 30.0];

/// `[min, median, max]` of `values`; `NaN`s for an empty slice.
fn min_median_max(values: &mut [f64]) -> [f64; 3] {
    if values.is_empty() {
        return [f64::NAN; 3];
    }
    values.sort_by(f64::total_cmp);
    [values[0], values[values.len() / 2], values[values.len() - 1]]
}

/// What a fit on a compressed declared law records about the compression
/// (gam#2928): the law's size before and after, and the certified anchor error
/// at every row's converged inputs against the anchor's sampling error.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeclaredLawCompressionRecord {
    pub(crate) atoms: usize,
    pub(crate) bins: usize,
    pub(crate) nodes: usize,
    pub(crate) anchors_checked: usize,
    pub(crate) anchors_meeting_target: usize,
    /// The largest certified `|α_n − α_K|`; infinite where an anchor could not
    /// be certified.
    pub(crate) max_delta: f64,
    pub(crate) max_delta_over_standard_error: f64,
    /// `[min, median, max]` over every converged anchor of the certified error
    /// against its target `10⁻³·SE_α`; at most one where every anchor meets it.
    pub(crate) bound_over_target: [f64; 3],
    /// `[min, median, max]` over the audit sample of the measured error on the
    /// declared atoms against the certified one; at most one where the bound
    /// holds.
    pub(crate) measured_over_bound: [f64; 3],
    /// Anchors in the audit sample, re-solved on the declared atoms.
    pub(crate) anchors_audited: usize,
}

impl From<&DeclaredLawCompressionRecord> for crate::inference::model::SavedDeclaredLawCompression {
    fn from(record: &DeclaredLawCompressionRecord) -> Self {
        // JSON has no infinity: an uncertified error is saved as `None`.
        let finite = |value: f64| value.is_finite().then_some(value);
        Self {
            atoms: record.atoms,
            bins: record.bins,
            nodes: record.nodes,
            anchors_checked: record.anchors_checked,
            anchors_meeting_target: record.anchors_meeting_target,
            max_certified_error: finite(record.max_delta),
            max_certified_error_over_standard_error: finite(record.max_delta_over_standard_error),
            certified_error_over_target: record.bound_over_target.map(finite),
            measured_error_over_certified: record.measured_over_bound.map(finite),
            anchors_audited: record.anchors_audited,
        }
    }
}

#[cfg(test)]
mod compression_tests {
    use super::*;
    use crate::test_support::{gauss_hermite_probabilists, skewed_grid};
    use gam_math::probability::standard_normal_quantile;

    /// One atom per row of a standardized log-normal with skewness about one,
    /// as gnomon's calibrate fixtures declare their score law (gam#2928).
    fn lognormal_atoms(n: usize) -> (Vec<f64>, Vec<f64>) {
        let raw: Vec<f64> = (0..n)
            .map(|k| {
                let p = (k as f64 + 0.5) / n as f64;
                (0.32 * standard_normal_quantile(p).expect("interior probability")).exp()
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / n as f64;
        let sd = (raw.iter().map(|u| (u - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        (raw.iter().map(|u| (u - mean) / sd).collect(), vec![1.0 / n as f64; n])
    }

    /// Small declared laws stay exact because their certified compression
    /// cannot halve them: the skewed 41-node law certifies about 33 nodes and
    /// Gauss–Hermite 65 about 38 (gam#2928 prototype), both above half.
    #[test]
    fn laws_their_certified_compression_cannot_halve_are_anchored_as_declared() {
        let skewed = skewed_grid();
        assert!(
            CompressedLaw::compress(&skewed.nodes, &skewed.weights, &default_design())
                .expect("compress")
                .is_none()
        );
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("Gauss–Hermite law");
        assert!(
            CompressedLaw::compress(&nodes, &weights, &default_design())
                .expect("compress")
                .is_none()
        );
    }

    #[test]
    fn compressed_law_is_a_declared_grid_with_the_atoms_moments_through_degree_three() {
        let (nodes, weights) = lognormal_atoms(2_000);
        let law = CompressedLaw::compress(&nodes, &weights, &default_design())
            .expect("compress")
            .expect("a 2,000-atom law compresses");
        let grid = law.grid();
        crate::bms::EmpiricalZGrid::new(grid.nodes.clone(), grid.weights.clone(), "compressed law")
            .expect("the compressed law is a valid declared grid");
        assert!(grid.nodes.windows(2).all(|pair| pair[0] < pair[1]));
        let span = nodes.iter().fold(0.0_f64, |m, u| m.max(u.abs()));
        for degree in 0..=3 {
            let moment = |u: &[f64], w: &[f64]| -> f64 {
                u.iter().zip(w).map(|(u, w)| w * u.powi(degree)).sum()
            };
            let exact = moment(&nodes, &weights);
            let compressed = moment(&grid.nodes, &grid.weights);
            // Rounding of the two sums and of each bin's moment passes: every
            // term is at most `span^degree` in magnitude, and at most
            // `n + K` terms and `2·(degree + 3)` operations enter per term.
            let rounding = f64::EPSILON
                * (nodes.len() + grid.nodes.len()) as f64
                * 2.0
                * (degree as f64 + 3.0)
                * span.powi(degree).max(1.0);
            assert!(
                (exact - compressed).abs() <= rounding,
                "degree {degree}: atoms {exact:e} vs compressed {compressed:e} (rounding bound {rounding:e})"
            );
        }
        eprintln!(
            "[2928 compression] {} atoms -> {} bins, {} nodes",
            law.atoms(),
            law.bins(),
            grid.nodes.len()
        );
    }

    #[test]
    fn exact_anchor_lies_inside_the_certified_interval_across_both_tails() {
        let (nodes, weights) = lognormal_atoms(2_000);
        let exact = AnchorGridOwned::new(nodes.clone(), weights.clone());
        let law = CompressedLaw::compress(&nodes, &weights, &default_design())
            .expect("compress")
            .expect("a 2,000-atom law compresses");
        let design = default_design();
        let mut certified = 0usize;
        for &slope in &[0.5, 0.8, 1.6, 4.0, -2.0] {
            for &q in &[-40.0, -10.0, -8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 10.0, 40.0] {
                let exact_root = solve_anchor(q, slope, exact.view()).expect("exact root");
                let root = solve_anchor(q, slope, law.grid().view()).expect("compressed root");
                let certificate = law.certify(q, slope, root);
                let measured = (exact_root - root).abs();
                eprintln!(
                    "[2928 compression] q={q} b={slope} |α_n − α_K|={measured:.3e} Δ={:.3e} \
                     certified={} SE={:.3e} Δ/SE={:.3e} bound/measured={:.2e}",
                    certificate.delta,
                    certificate.certified,
                    certificate.standard_error,
                    certificate.delta / certificate.standard_error,
                    certificate.delta / measured
                );
                if certificate.certified {
                    certified += 1;
                    assert!(
                        measured <= certificate.delta,
                        "q={q} b={slope}: exact root {exact_root} outside the certified \
                         [{root} ± {:e}]",
                        certificate.delta
                    );
                }
                if design.contains(&DesignPoint { q, slope }) {
                    assert!(
                        certificate.meets_target(),
                        "q={q} b={slope}: at a design point the certificate must meet \
                         10⁻³·SE: Δ={:e}, SE={:e}",
                        certificate.delta,
                        certificate.standard_error
                    );
                }
            }
        }
        eprintln!("[2928 compression] {certified} of 55 anchors certified");
    }
}
