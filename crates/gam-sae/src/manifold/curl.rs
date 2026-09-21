//! The missing structure moves: flat pair → circle (`curl`) and circle → flat
//! (`flatten`).
//!
//! # Why these are invisible to every other producer
//!
//! A mean-zero circle's cone **is** its 2-plane (support-invisibility:
//! `ℝ_{>0}·ellipse = plane∖{0}`), so a linear dictionary lawfully parks two
//! directions `u, v` on any centered rotational feature and splits every firing
//! into a co-active `(α, β)`. No residual is left, so the ISA producer never
//! fires; no evidence gap opens, so no race is called. The *joint amplitude law*
//! is the only witness:
//!
//!   * `κ = m₄/m₂²` on `r² = α² + β²` — `1` = ring, `2` = Gaussian fill (the
//!     zero-gain point of the coding law), `1/q` = gated spike.
//!   * first/second circular resultants `R₁, R₂` — coverage of the angle, and
//!     the diameter degeneracy (`R₂ → 1` ⇒ a line, not a circle).
//!   * the rate–distortion pre-screen `n_eff·½·ln(3R̂²/(π²σ²)) − Δcharge`, with
//!     the derived crossover `R̂ > σ·π/√3` below which a circle cannot pay for
//!     itself.
//!
//! Influence-function SEs make the κ gate a 2σ screen; the engine's existing
//! topology race stays the judge — `curl` only submits a race-ready seed.
//!
//! `flatten` is the inverse demotion, so the move pair is falsifiable *inside*
//! the dictionary's life, not just at birth: it is the exact complement of the
//! ring recognition `curl` promotes on, read on the atom's image in its own
//! principal plane and on the rows it parses. A parse that is no longer a
//! recognizable ring (e.g. Gaussian fill) demotes the circle to a rank-2 flat
//! plane; a diameter collapse of either demotes it to rank-1.
//!
//! # The proposer pipeline (Phase 4)
//!
//! The geodict verdict math is only the *judge's pre-screen*. Turning it into a
//! move that fires on real dictionaries needs three stages the delivered
//! detector did not contain (INTEGRATION_PLAN Phase 4, risks #3/#5):
//!
//!   1. [`coalesce_antipodal`] — a nonnegative-gate dictionary shatters a
//!      centered circle into up to FOUR rectified half-atoms (`±u, ±v`); curl
//!      candidates must form over the coalesced signed directions or the move is
//!      a no-op on every such dictionary (launch blocker).
//!   2. [`cooccurrence_pairs_sparse`] — candidate planes come from co-firing
//!      counts over the coalesced directions, read off the transposed firing
//!      lists so EVERY row is counted without an `O(K²)` enumeration.
//!   3. [`CurlCooldownLedger`] — an atom-set-keyed cooldown so
//!      `curl → flatten → curl` cannot oscillate across rounds.
//!
//! The term-level driver that assembles per-atom directions/gates, projects the
//! plane, ranks by `net_evidence_nats`, and submits seeds to the birth/race
//! plumbing lives in [`crate::structure_harvest`]; these stay pure so each is
//! unit-testable in isolation.

use std::f64::consts::TAU;

use gam_math::probability::normal_cdf;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// The rate–distortion crossover radius factor `π/√3`: below `R̂ = σ·π/√3` a
/// centered circle cannot pay its charge, so the screen refuses it regardless
/// of κ. Derived, not tuned: quantizing the arc `s ∈ [0, 2πR̂)` with cell width
/// `Δ` gives positional MSE `Δ²/12`; matching the noise floor (`Δ²/12 = σ²`)
/// makes the per-row angle cost `ln(2πR̂/Δ) = ln(πR̂/(√3σ))` nats against the
/// flat two-coordinate reference `2·ln(R̂/σ)`, so the per-row gain is
/// `½·ln(3R̂²/(π²σ²))`, whose zero is exactly `R̂ = σ·π/√3 ≈ 1.814σ`. Sanity:
/// the crossover must exceed σ — a ring smaller than its own noise annulus is
/// described at least as compactly by the Gaussian fill.
///
/// `Δ²/12` is the small-cell law, so this crossover is a high-resolution
/// approximation, accurate only while `πR̂/(√3σ)` is large; near the crossover
/// the codebook holds a few cells and the finite code
/// (`description_length::circle_phase_code`) differs. It prices neither the
/// support nor the decoder, so it cannot decide whether a ring pays: the atomic
/// replacement ledger ([`crate::manifold::curve_promotion`]) adjudicates on
/// [`ring_recognition`] and its own finite codebook and never reads this screen
/// (#2933 F23). [`CurlVerdict::geometry_ok`] still conjoins it.
///
/// Derived (#2469): `π/√3`, the zero of the per-row gain above, written to
/// f64 precision.
pub const RD_CROSSOVER_FACTOR: f64 = 1.8137993642342178; // π / √3

/// The evidence level the κ / resultant gates fire at (a 2σ screen, matching the
/// pair-κ merge screen's derivation level family).
const CURL_Z: f64 = 2.0;

/// The verdict for a candidate flat-pair → circle promotion.
#[derive(Debug, Clone)]
pub struct CurlVerdict {
    /// `κ = E[r⁴]/E[r²]²` on `r² = α² + β²`. `1` ring, `2` Gaussian fill.
    pub kappa: f64,
    /// Influence-function standard error of `κ`.
    pub kappa_se: f64,
    /// `(2 − κ)/se` — how many σ below the Gaussian-fill value `2` the radius
    /// law sits. Positive and large ⇒ a genuine ring.
    pub z_below_gaussian: f64,
    /// First circular resultant `R₁ = |E[e^{iθ}]|` (coverage; `→ 0` full ring).
    pub resultant1: f64,
    /// Second circular resultant `R₂ = |E[e^{2iθ}]|` (`→ 1` diameter/line).
    pub resultant2: f64,
    /// `R̂ = √(max(E[r²] − 2σ², 0))` — the noise-debiased fitted radius (the raw
    /// √(E\[r²\]) is biased up by the 2-D noise energy `2σ²` on the active support).
    pub radius: f64,
    /// `½·ln(3R̂²/(π²σ²))` — the per-row coding gain in nats, in its small-cell
    /// (high-resolution) approximation: a race screen, not a certificate.
    pub gain_nats_per_row: f64,
    /// `n_eff·½·ln(3R̂²/(π²σ²)) − Δcharge` — the net evidence for the circle.
    pub net_evidence_nats: f64,
    /// True ⇒ recommend submitting the `CircleSeed` to the race.
    pub recommend_curl: bool,
    /// The conjunction WITHOUT the κ gate: the rate–distortion screen pays, the
    /// angle is covered (`R₁` small) and the plane is not a diameter (`R₂` small).
    ///
    /// These three are robust; the κ gate is not, and a census has to say so
    /// separately. `κ = m₄/m₂²` is a FOURTH moment, so its breakdown point is
    /// essentially zero: a planted ring of 3000 rows contaminated by 20 unrelated
    /// co-firings at eight times the ring radius measures `κ = 194` while `R₁` and
    /// `R₂` still read a clean, fully covered, non-degenerate circle. On a real
    /// dictionary every co-firing pair carries that contamination, so a screen
    /// that conjoins the κ gate refuses ground truth. Callers doing their own
    /// calibrated test on a robust statistic should gate on this and let the test
    /// decide, rather than paying for `κ`'s tail twice.
    pub geometry_ok: bool,
}

/// Radius-law moments and their influence-function SEs over the paired coords.
struct RadiusLaw {
    n: usize,
    m2: f64,
    kappa: f64,
    kappa_se: f64,
}

fn radius_law(alpha: ArrayView1<f64>, beta: ArrayView1<f64>) -> Result<RadiusLaw, String> {
    let n = alpha.len();
    if beta.len() != n {
        return Err(format!("curl: α len {n} != β len {}", beta.len()));
    }
    if n < 2 {
        return Err("curl: need at least 2 rows for the radius law".to_string());
    }
    let inv = 1.0 / n as f64;
    let mut m2 = 0.0_f64; // E[r²]
    let mut m4 = 0.0_f64; // E[r⁴]
    let mut m8 = 0.0_f64; // E[r⁸] (for the κ SE)
    for i in 0..n {
        let r2 = alpha[i] * alpha[i] + beta[i] * beta[i];
        let r4 = r2 * r2;
        m2 += r2;
        m4 += r4;
        m8 += r4 * r4;
    }
    m2 *= inv;
    m4 *= inv;
    m8 *= inv;
    if !(m2 > 0.0) {
        return Err("curl: zero in-plane energy (m₂ = 0)".to_string());
    }
    let kappa = m4 / (m2 * m2);
    // Delta-method SE of κ̂ = m̂₄/m̂₂² from the joint variance of (m̂₂, m̂₄):
    // grad = (−2 m₄/m₂³, 1/m₂²), Var(m̂₂) = (E[r⁴]−m₂²)/n, Var(m̂₄) = (E[r⁸]−m₄²)/n,
    // Cov(m̂₂, m̂₄) = (E[r⁶]−m₂·m₄)/n. The E[r⁶] cross moment is accumulated
    // directly (no proxy) — keeping the covariance term is what makes this SE
    // exact to first order.
    let mut m6 = 0.0_f64;
    for i in 0..n {
        let r2 = alpha[i] * alpha[i] + beta[i] * beta[i];
        m6 += r2 * r2 * r2;
    }
    m6 *= inv;
    let v22 = (m4 - m2 * m2).max(0.0) / n as f64;
    let v44 = (m8 - m4 * m4).max(0.0) / n as f64;
    let c24 = (m6 - m2 * m4) / n as f64;
    let g2 = -2.0 * m4 / (m2 * m2 * m2);
    let g4 = 1.0 / (m2 * m2);
    let var_kappa = g2 * g2 * v22 + g4 * g4 * v44 + 2.0 * g2 * g4 * c24;
    let kappa_se = var_kappa.max(0.0).sqrt();
    Ok(RadiusLaw {
        n,
        m2,
        kappa,
        kappa_se,
    })
}

/// Sample circular moments `(E[cos θ], E[sin θ], E[cos 2θ], E[sin 2θ])` of the
/// parse angles `θ_i = atan2(β_i, α_i)`.
fn circular_moments(alpha: ArrayView1<f64>, beta: ArrayView1<f64>) -> [f64; 4] {
    let n = alpha.len();
    let inv = 1.0 / n as f64;
    let (mut c1, mut s1, mut c2, mut s2) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let th = beta[i].atan2(alpha[i]);
        c1 += th.cos();
        s1 += th.sin();
        c2 += (2.0 * th).cos();
        s2 += (2.0 * th).sin();
    }
    [c1 * inv, s1 * inv, c2 * inv, s2 * inv]
}

/// The witnesses a centered circle leaves in its joint amplitude law, read with no
/// coding price and no noise scale (#2933 F23).
#[derive(Debug, Clone)]
pub struct RingRecognition {
    /// `κ = E[r⁴]/E[r²]²` on `r² = α² + β²`. `1` ring, `2` Gaussian fill.
    pub kappa: f64,
    /// Influence-function standard error of `κ`.
    pub kappa_se: f64,
    /// `(2 − κ)/se` — how many σ below the Gaussian-fill value `2` the radius
    /// law sits.
    pub z_below_gaussian: f64,
    /// First circular resultant `R₁ = |E[e^{iθ}]|` (coverage; `→ 0` full ring).
    pub resultant1: f64,
    /// Second circular resultant `R₂ = |E[e^{2iθ}]|` (`→ 1` diameter/line).
    pub resultant2: f64,
    /// `√E[r²]`, the RMS in-plane radius, measured with no noise model.
    pub rms_radius: f64,
    /// The plane has collapsed to a diameter: `R₂` sits nearer the line template
    /// `R₂ = 1` than the isotropic template `R₂ = 0`. A diameter is never
    /// covered, and it is the one diameter screen both directions of the
    /// curl/flatten pair read, so a plane cannot be neither curlable nor
    /// flattenable (#3506).
    pub diameter: bool,
    /// The plane is not a diameter and the first harmonic resolves no departure
    /// from the centrally symmetric null `E[e^{iθ}] = 0`: the studentized
    /// Rayleigh statistic (asymptotically `Exp(1)`, see `recognize`) is below
    /// `−ln Φ(−CURL_Z)`, so a full circle or ellipse is refused with probability
    /// `Φ(−CURL_Z)`, the κ gate's level, at every n.
    pub covered: bool,
    /// κ resolvably below the Gaussian-fill value 2 (2σ) and `covered`.
    pub recognized: bool,
}

fn recognize(law: &RadiusLaw, alpha: ArrayView1<f64>, beta: ArrayView1<f64>) -> RingRecognition {
    let [c1, s1, c2, s2] = circular_moments(alpha, beta);
    let resultant1 = c1.hypot(s1);
    let resultant2 = c2.hypot(s2);
    let z_below_gaussian = if law.kappa_se > 0.0 {
        (2.0 - law.kappa) / law.kappa_se
    } else if law.kappa < 2.0 {
        f64::INFINITY
    } else {
        0.0
    };
    // Diameter degeneracy. A line through the origin has R₂ = 1 and an isotropic
    // plane has R₂ = 0. R₂ is NOT screened against the isotropic null: the
    // periodic atom a curl seeds carries independent sin/cos decoder rows, so an
    // ellipse of aspect b is a ring of that family, and at uniform phase it reads
    // R₂ = (1 − b)/(1 + b) > 0, which a Rayleigh screen on R₂ refuses at every b < 1
    // once n is large (#3827). What R₂ decides is which template the plane sits
    // nearer, the line (R₂ = 1) or the isotropic plane (R₂ = 0); the equidistant
    // boundary is R₂ = ½. A diameter is never covered, and this is the one
    // diameter screen both directions of the curl/flatten pair read (#3506).
    let diameter = resultant2 >= 0.5;
    // Coverage against the centrally symmetric null E[e^{iθ}] = 0 (#3827). Every
    // full closed curve of the periodic family symmetric about its center (a
    // circle or an ellipse) satisfies it; an arc of width w < 2π has
    // R₁ → sin(w/2)/(w/2) > 0. Under the null, by the CLT, the mean
    // m = (E[cos θ], E[sin θ]) is asymptotically N(0, V/n) with
    //     V = ½·[[1 + c₂, s₂], [s₂, 1 − c₂]],   det V = ¼(1 − R₂²),
    // read off the sample second harmonic (the null-restricted score form), so
    //     Q = ½·n·mᵀV⁻¹m = n·[(1 − c₂)c₁² − 2s₂c₁s₁ + (1 + c₂)s₁²] / (1 − R₂²)
    // is asymptotically χ²₂/2 = Exp(1). On an isotropic plane Q = n·R₁², the
    // Rayleigh statistic. Refusing iff Q ≥ t with t = −ln α, α = Φ(−CURL_Z) (the κ
    // gate's level), refuses a full ring with probability α at every n and any
    // fixed arc with probability → 1. There is no absolute floor. The diameter
    // screen runs first, so 1 − R₂² ≥ ¾ wherever Q is read.
    let covered = !diameter && {
        let n = law.n as f64;
        let q = n * ((1.0 - c2) * c1 * c1 - 2.0 * s2 * c1 * s1 + (1.0 + c2) * s1 * s1)
            / (1.0 - resultant2 * resultant2);
        q < -normal_cdf(-CURL_Z).ln()
    };
    RingRecognition {
        kappa: law.kappa,
        kappa_se: law.kappa_se,
        z_below_gaussian,
        resultant1,
        resultant2,
        rms_radius: law.m2.sqrt(),
        diameter,
        covered,
        recognized: z_below_gaussian > CURL_Z && covered,
    }
}

/// Recognize a candidate plane `(α, β)` as a ring: κ resolvably below the
/// Gaussian-fill value 2 (2σ), the angle covered (`R₁` not resolvable from the
/// centrally symmetric null) and no diameter degeneracy (`R₂`; see
/// [`RingRecognition::covered`]). No noise scale enters and nothing is priced; a
/// caller that prices the replacement in bits decides acceptance on this and its
/// own ledger, not on the small-cell screen of [`curl_verdict`] (#2933 F23).
pub fn ring_recognition(
    alpha: ArrayView1<f64>,
    beta: ArrayView1<f64>,
) -> Result<RingRecognition, String> {
    let law = radius_law(alpha, beta)?;
    Ok(recognize(&law, alpha, beta))
}

/// Adjudicate a candidate flat pair `(α, β)` for promotion to a circle.
///
/// `sigma` is the ambient per-coordinate noise scale (the RD reference),
/// `n_eff = Σ a²` the pattern's effective occupancy (NOT the raw row count),
/// `delta_charge` the module-4 charge at that occupancy. The recommendation is a
/// conjunction: [`ring_recognition`] and the small-cell RD screen paying
/// (`R̂ > σ·π/√3` and net evidence positive). The screen is a high-resolution
/// approximation that prices no support dividend (see [`RD_CROSSOVER_FACTOR`]).
pub fn curl_verdict(
    alpha: ArrayView1<f64>,
    beta: ArrayView1<f64>,
    sigma: f64,
    n_eff: f64,
    delta_charge: f64,
) -> Result<CurlVerdict, String> {
    if !(sigma > 0.0 && sigma.is_finite()) {
        return Err(format!("curl: sigma must be finite and > 0, got {sigma}"));
    }
    let law = radius_law(alpha, beta)?;
    let recognition = recognize(&law, alpha, beta);
    // Noise-debiased radius. `(α, β)` are the plane coords on the ACTIVE SUPPORT
    // (the co-firing rows the driver passes), so a noisy ring
    // `x = R(cosθ, sinθ) + ε`, `ε ~ N(0, σ²I₂)`, has `E[r²] = R² + 2σ²`: each of
    // the two in-plane coordinates carries the per-coordinate noise variance σ².
    // `√m₂` therefore estimates `√(R² + 2σ²)`, biased UP by the 2-D noise annulus
    // — enough to push a sub-crossover ring (e.g. `R = 1.5σ` ⇒ `√m₂ = 2.06σ`)
    // above the `σ·π/√3 ≈ 1.814σ` acceptance threshold even though its true coding
    // gain is negative. Subtract the `2σ²` noise energy first (same debiasing as
    // `isa_seed`'s `a² = (m₂ − 2σ²)/q̂`, here with `q̂ = 1` on the active support).
    let radius = (law.m2 - 2.0 * sigma * sigma).max(0.0).sqrt();
    // Per-row circle coding gain ½·ln(3R̂²/(π²σ²)) in nats, in its small-cell
    // approximation (arc cells with noise Δ²/12). Equivalently ln(R̂/σ) − ln(π/√3):
    // the shape constant −0.5954… is what makes the gain vanish at the
    // RD_CROSSOVER_FACTOR radius, not at R̂ = σ. At coarse resolution the finite
    // phase code differs (`description_length::circle_phase_code`); this value only
    // screens and ranks race candidates (#2933 F23).
    let gain_nats_per_row = {
        use std::f64::consts::PI;
        0.5 * (3.0 * radius * radius / (PI * PI * sigma * sigma)).ln()
    };
    let net_evidence_nats = n_eff * gain_nats_per_row - delta_charge;

    let rd_pays = radius > sigma * RD_CROSSOVER_FACTOR && net_evidence_nats > 0.0;
    let geometry_ok = rd_pays && recognition.covered;
    let recommend_curl = recognition.z_below_gaussian > CURL_Z && geometry_ok;

    Ok(CurlVerdict {
        kappa: recognition.kappa,
        kappa_se: recognition.kappa_se,
        z_below_gaussian: recognition.z_below_gaussian,
        resultant1: recognition.resultant1,
        resultant2: recognition.resultant2,
        radius,
        gain_nats_per_row,
        net_evidence_nats,
        recommend_curl,
        geometry_ok,
    })
}

/// A race-ready circle seed in the engine's periodic-harmonic decoder layout.
///
/// The decoder has `1 + 2·harmonics` basis rows: row 0 is the constant
/// (`center`), then for each harmonic `h` a `(sin, cos)` pair. The fundamental
/// carries the fitted radius: `cos₁ = R̂·u`, `sin₁ = R̂·v`; higher harmonics
/// start at zero for the race to refine. Reconstruction of row `i` at
/// `t = theta_turns[i]` is
/// `center + Σ_h [ sin(2π h t)·sin_h + cos(2π h t)·cos_h ]`.
#[derive(Debug, Clone)]
pub struct CircleSeed {
    /// Per-row angle in turns `∈ [0, 1)`, `θ_i/2π`.
    pub theta_turns: Array1<f64>,
    /// Fitted radius `R̂`.
    pub radius: f64,
    /// Decoder in the periodic-harmonic layout, `(1 + 2·harmonics, p)`.
    pub decoder: Array2<f64>,
    /// The center (row 0 of the decoder), length `p`.
    pub center: Array1<f64>,
}

impl CircleSeed {
    /// Reconstruct the ambient point at `t` turns from the harmonic decoder.
    pub fn reconstruct(&self, t: f64) -> Array1<f64> {
        let p = self.decoder.ncols();
        let harmonics = (self.decoder.nrows() - 1) / 2;
        let mut out = self.center.clone();
        let phase = TAU * t;
        for h in 1..=harmonics {
            let sin_row = 2 * h - 1;
            let cos_row = 2 * h;
            let (sh, ch) = ((h as f64 * phase).sin(), (h as f64 * phase).cos());
            for j in 0..p {
                out[j] += sh * self.decoder[[sin_row, j]] + ch * self.decoder[[cos_row, j]];
            }
        }
        out
    }
}

/// Build the periodic-harmonic circle seed from the fitted plane frame `(u, v)`
/// and the paired coords `(α, β)`. `u, v` are the two ambient directions
/// (length `p`) the flat pair lives in — they need NOT be orthonormal; the seed
/// reconstructs the ring exactly in whatever frame it was parked. `harmonics ≥ 1`.
pub fn curl_seed(
    u: ArrayView1<f64>,
    v: ArrayView1<f64>,
    alpha: ArrayView1<f64>,
    beta: ArrayView1<f64>,
    harmonics: usize,
    center: ArrayView1<f64>,
) -> Result<CircleSeed, String> {
    let p = u.len();
    if v.len() != p || center.len() != p {
        return Err(format!(
            "curl_seed: u/v/center length mismatch (p from u = {p}, v = {}, center = {})",
            v.len(),
            center.len()
        ));
    }
    if harmonics == 0 {
        return Err("curl_seed: need at least 1 harmonic".to_string());
    }
    let n = alpha.len();
    if beta.len() != n {
        return Err("curl_seed: α/β length mismatch".to_string());
    }
    // Per-row angle and radius; R̂ is the RMS radius (constant on a clean ring).
    let mut theta_turns = Array1::<f64>::zeros(n);
    let mut m2 = 0.0_f64;
    for i in 0..n {
        theta_turns[i] = beta[i].atan2(alpha[i]) / TAU;
        if theta_turns[i] < 0.0 {
            theta_turns[i] += 1.0;
        }
        m2 += alpha[i] * alpha[i] + beta[i] * beta[i];
    }
    let radius = if n > 0 { (m2 / n as f64).sqrt() } else { 0.0 };

    let mut decoder = Array2::<f64>::zeros((1 + 2 * harmonics, p));
    for j in 0..p {
        decoder[[0, j]] = center[j];
        // Fundamental: cos₁ = R̂·u, sin₁ = R̂·v.
        decoder[[1, j]] = radius * v[j]; // sin₁ row (row 2h-1 with h=1)
        decoder[[2, j]] = radius * u[j]; // cos₁ row (row 2h with h=1)
    }
    Ok(CircleSeed {
        theta_turns,
        radius,
        decoder,
        center: center.to_owned(),
    })
}

/// The verdict for the inverse move — demoting a circle back to flat.
///
/// It is the exact complement of [`ring_recognition`] read on two planes of the
/// same atom: its own image in the principal plane of that image (geometry), and
/// the rows the atom parses projected onto that plane (the data). A circle that
/// would not be recognized as a ring today is not carrying rotational structure
/// and is demoted, so curl and flatten share one threshold set and no plane is
/// both curlable and flattenable, or neither (#3506).
#[derive(Debug, Clone)]
pub struct FlattenVerdict {
    /// Ring witnesses of the atom's own image `Φ·B` in its principal plane.
    pub image: RingRecognition,
    /// Ring witnesses of the parsed rows projected on the same plane.
    pub parse: RingRecognition,
    /// The residual rank the circle should be demoted to: `1` = line (diameter
    /// collapse of the image or of the parse), `2` = flat plane (the parse is not
    /// a recognizable ring, e.g. a Gaussian fill with `κ ≈ 2`). Meaningful only
    /// when `recommend_flatten`.
    pub residual_rank: usize,
    /// True ⇒ this "circle" is not carrying rotational structure and should be
    /// demoted.
    pub recommend_flatten: bool,
}

/// Adjudicate whether a fitted circle has degenerated and should be flattened.
///
/// `(image_alpha, image_beta)` are the atom's own image rows `Φ·B` (centered)
/// in the principal plane of that image; `(parse_alpha, parse_beta)` are the
/// rows the atom parses, centered and projected on the same plane. Both are
/// read with [`ring_recognition`]:
///
///   * a diameter in either plane (`R₂` past the screen `covered` refuses) is a
///     rank-1 line — the decoder traces a segment, or the data it explains lie
///     on one;
///   * otherwise a parse that is not recognized as a ring (κ not resolvably below
///     the Gaussian-fill value 2, or the angle not covered) is a rank-2 plane;
///   * otherwise the ring stands.
///
/// No threshold is introduced here: every cut is the one `curl` promotes on.
pub fn flatten_verdict(
    image_alpha: ArrayView1<f64>,
    image_beta: ArrayView1<f64>,
    parse_alpha: ArrayView1<f64>,
    parse_beta: ArrayView1<f64>,
) -> Result<FlattenVerdict, String> {
    let n = image_alpha.len();
    if image_beta.len() != n || parse_alpha.len() != n || parse_beta.len() != n {
        return Err(format!(
            "flatten_verdict: plane length mismatch (image α {n}, image β {}, parse α {}, parse β {})",
            image_beta.len(),
            parse_alpha.len(),
            parse_beta.len()
        ));
    }
    let image = ring_recognition(image_alpha, image_beta)?;
    let parse = ring_recognition(parse_alpha, parse_beta)?;
    let (recommend_flatten, residual_rank) = if image.diameter || parse.diameter {
        (true, 1)
    } else if !parse.recognized {
        (true, 2)
    } else {
        (false, 2)
    };
    Ok(FlattenVerdict {
        image,
        parse,
        residual_rank,
        recommend_flatten,
    })
}

/// Gram–Schmidt orthonormalize a candidate plane frame `(u, v)` and project the
/// ambient rows `x` (`n×p`) onto it, returning the paired coords `(α, β)`. This
/// is the projection the pair-κ co-firing screen's plane feeds `curl_verdict`.
pub fn orthonormal_pair_coords(
    x: ArrayView2<f64>,
    u: ArrayView1<f64>,
    v: ArrayView1<f64>,
    mean: ArrayView1<f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let (n, p) = x.dim();
    if u.len() != p || v.len() != p || mean.len() != p {
        return Err("orthonormal_pair_coords: frame/mean dim mismatch with x".to_string());
    }
    let un = u.dot(&u).sqrt();
    if !(un > 0.0) {
        return Err("orthonormal_pair_coords: u is zero".to_string());
    }
    let e1: Array1<f64> = u.mapv(|z| z / un);
    let vproj = v.dot(&e1);
    let mut e2: Array1<f64> = (0..p).map(|j| v[j] - vproj * e1[j]).collect();
    let e2n = e2.dot(&e2).sqrt();
    if !(e2n > 0.0) {
        return Err("orthonormal_pair_coords: u,v are collinear".to_string());
    }
    e2.mapv_inplace(|z| z / e2n);

    let mut alpha = Array1::<f64>::zeros(n);
    let mut beta = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (mut a, mut b) = (0.0, 0.0);
        for j in 0..p {
            let xi = x[[i, j]] - mean[j];
            a += xi * e1[j];
            b += xi * e2[j];
        }
        alpha[i] = a;
        beta[i] = b;
    }
    Ok((alpha, beta, e1, e2))
}

// ---------------------------------------------------------------------------
// Proposer pipeline: antipodal coalescing → co-occurrence candidate gen →
// hysteresis cooldown. These are the stages the delivered geodict detector did
// NOT contain (INTEGRATION_PLAN Phase 4 items 1–2 and risk #3/#5); the verdict /
// seed / orthonormal-coords math above is the geodict core they feed.
//
// They are kept pure (ndarray only) so the term-level driver in
// `structure_harvest.rs` can assemble the per-atom directions/gates and hand
// them here, and so each stage is unit-testable in isolation.
// ---------------------------------------------------------------------------

/// A signed ambient direction recovered by antipodal coalescing: either a single
/// signed linear atom, or the merge of two rectified half-atoms (`±d`) a
/// nonnegative-gate dictionary produced for one signed direction.
#[derive(Debug, Clone)]
pub struct SignedDirection {
    /// Unit ambient direction of the coalesced signed axis (length `p`).
    pub dir: Array1<f64>,
    /// The atom indices coalesced into this signed direction (one or two).
    pub members: Vec<usize>,
    /// Per-row activity mask (length `n`): the UNION of the members' gates — a
    /// row is active on the signed direction when either rectified half fired.
    pub active: Vec<bool>,
}

/// Cosine of two vectors; `0` if either is (near-)zero.
fn cosine(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let na = a.dot(&a).sqrt();
    let nb = b.dot(&b).sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    a.dot(&b) / (na * nb)
}

/// Overlap (Jaccard) of two boolean row masks; `0` when both are empty.
fn mask_overlap(a: &[bool], b: &[bool]) -> f64 {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        if *x || *y {
            union += 1;
        }
        if *x && *y {
            inter += 1;
        }
    }
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Coalesce rectified antipodal half-atoms into signed directions
/// (INTEGRATION_PLAN Phase 4.1; launch blocker risk #3).
///
/// A nonnegative-gate dictionary parses a centered signed direction `d` as two
/// rectified half-atoms `d⁺, d⁻ = −d` whose decoder cosine is `≈ −1` and whose
/// gates are near-disjoint (a row fires one half or the other, rarely both). A
/// centered circle in a 2-plane shatters into up to FOUR such halves (`±u, ±v`).
/// Curl candidates must be formed over the COALESCED signed directions, or the
/// move is a no-op on every nonnegative-gate dictionary.
///
/// `dirs[i]` is atom `atom_ids[i]`'s ambient direction (need not be unit),
/// `active[i]` its per-row gate mask. A pair `(i, j)` coalesces when
/// `cos(dir_i, dir_j) ≤ cos_threshold` (opposite) AND
/// `overlap(active_i, active_j) ≤ max_overlap` (disjoint). Each atom coalesces
/// with at most one partner (greedy, most-antipodal first); an unpaired atom
/// rides as its own already-signed direction so signed dictionaries (no
/// rectification) still yield candidates.
pub fn coalesce_antipodal(
    dirs: &[ArrayView1<f64>],
    active: &[Vec<bool>],
    atom_ids: &[usize],
    cos_threshold: f64,
    max_overlap: f64,
) -> Vec<SignedDirection> {
    let k = dirs.len();
    assert_eq!(active.len(), k, "coalesce: dirs/active length mismatch");
    assert_eq!(atom_ids.len(), k, "coalesce: dirs/atom_ids length mismatch");

    // Enumerate antipodal + disjoint candidate merges, most-antipodal first. The
    // enumeration is `O(K²·p)` in the cosine and `O(K²·n)` in the overlap, which
    // at dictionary scale (`K` in the thousands) is the whole cost of the census;
    // the outer index is therefore fanned out, and the result re-sorted so the
    // greedy binding below stays bit-deterministic in `(dirs, active)`.
    let mut merges: Vec<(f64, usize, usize)> = (0..k)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut local: Vec<(f64, usize, usize)> = Vec::new();
            for j in (i + 1)..k {
                let c = cosine(dirs[i], dirs[j]);
                if c <= cos_threshold && mask_overlap(&active[i], &active[j]) <= max_overlap {
                    local.push((c, i, j));
                }
            }
            local
        })
        .collect();
    // Most antipodal (smallest cosine) binds first; deterministic tiebreak.
    merges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    let mut used = vec![false; k];
    let mut out: Vec<SignedDirection> = Vec::new();
    for (_c, i, j) in merges {
        if used[i] || used[j] {
            continue;
        }
        // A zero direction has no unit vector and cannot be merged.
        let ni = dirs[i].dot(&dirs[i]).sqrt();
        let nj = dirs[j].dot(&dirs[j]).sqrt();
        if ni == 0.0 || nj == 0.0 {
            continue;
        }
        used[i] = true;
        used[j] = true;
        // Orient both halves to a common sign and average: e ∝ d̂_i − d̂_j (since
        // d_j ≈ −d_i, this is the mean signed axis, robust to unequal norms).
        let mut e: Array1<f64> = dirs[i].mapv(|x| x / ni);
        for (idx, val) in dirs[j].iter().enumerate() {
            e[idx] -= val / nj;
        }
        let en = e.dot(&e).sqrt();
        if en <= 0.0 {
            // Degenerate (exactly opposite unit vectors that cancelled): fall
            // back to member i's own direction.
            e = dirs[i].mapv(|x| x / ni);
        } else {
            e.mapv_inplace(|x| x / en);
        }
        let union: Vec<bool> = active[i]
            .iter()
            .zip(active[j].iter())
            .map(|(a, b)| *a || *b)
            .collect();
        out.push(SignedDirection {
            dir: e,
            members: vec![atom_ids[i], atom_ids[j]],
            active: union,
        });
    }
    // Unpaired atoms ride as already-signed directions.
    for i in 0..k {
        if used[i] {
            continue;
        }
        let ni = dirs[i].dot(&dirs[i]).sqrt();
        if ni == 0.0 {
            continue;
        }
        out.push(SignedDirection {
            dir: dirs[i].mapv(|x| x / ni),
            members: vec![atom_ids[i]],
            active: active[i].clone(),
        });
    }
    out
}

/// Exact permutation evidence for the ring reading of one candidate plane.
///
/// The κ screen is a 2σ gate on a single pair. A census asks it of tens of
/// thousands of pairs, so "κ resolvably below 2" stops being a claim anyone can
/// act on without a multiplicity account, and the influence-function SE — a
/// first-order delta-method quantity on a fourth moment — is exactly the kind of
/// asymptotic approximation whose tail is least trustworthy.
///
/// # The null
///
/// "These two in-plane coordinates are independent given their own marginals",
/// realised by permuting `beta` against `alpha`. It is the null a ring is a claim
/// against, and it is much sharper than a Gaussian reference: it strips out
/// everything about the ring signature that lives in the MARGINALS and asks only
/// whether the PAIRING carries structure. It also controls, for free, the confound
/// a gated dictionary manufactures — a JumpReLU threshold carves the low-amplitude
/// corner out of every co-firing cloud, hollowing it near the origin, and the
/// surrogates carry the identical hole.
///
/// # The statistic is a RANK correlation, because κ has no breakdown point
///
/// A ring is the statement `α² + β² = R²`: the two squared coordinates are
/// perfectly, monotonically anti-dependent. That is a rank fact, and reading it as
/// one is what makes the test survive contact with a real dictionary.
///
/// The moment version does not survive it. `κ = m₄/m₂²` is a FOURTH moment, so a
/// handful of unrelated co-firings dominate it. Measured, on planted ground truth:
/// a ring of 3000 rows planted in a public SAE's own code plane, contaminated by
/// the ~20 rows where the two atoms co-fire naturally at several times the ring
/// radius, reports `κ = 194` at a planted radius of 1σ and `κ = 4.0` at 4σ —
/// while `R₁ = 0.017` and `R₂ = 0.028` on the same rows read a clean, fully
/// covered, non-degenerate circle. The κ gate refused ground truth at every
/// radius tested. Every co-firing pair in a real dictionary carries that
/// contamination; it is the normal case, not a pathology.
///
/// So the statistic is Spearman's `ρ` between `a = α²` and `b = β²`:
///
/// ```text
///   ρ = 1 − 6·Σᵢ (rank(aᵢ) − rank(b_{π(i)}))² / (n(n²−1)) ,
/// ```
///
/// which is `−1` for a ring, `0` under independence, and bounded by construction —
/// a dynamic range of 1 where κ's is 0.25, and immune to the tail that destroys κ.
/// The lower tail is the ring tail.
///
/// An earlier attempt added the fourth circular harmonic `R₄ = |E[e^{4iθ}]|` to κ,
/// on the argument that a uniform angle kills every harmonic while the product of
/// two ring marginals piles mass at four corners. The spike-in arm refuted it: at
/// a matched budget κ alone returned 2 e-BH discoveries and κ + R₄ returned 0, at
/// every planted radius. A nonnegative gate cannot carry a CENTRED ring, so every
/// co-firing circle a real dictionary holds is offset into the positive quadrant
/// and keeps 4-fold corner structure after the parse re-centres it; `R₄` sits above
/// its null mean and vetoes the circles it was added to find. A statistic argued
/// for on paper and never measured is how a screen acquires a veto nobody notices.
///
/// # Cost, and why it is not what the budget suggests
///
/// Ranks are computed once. Each surrogate is then one shuffle of the rank vector
/// and one dot product — Spearman's `ρ` is Pearson's on ranks, and the rank
/// marginals are permutation-invariant, so only `Σᵢ rank(aᵢ)·rank(b_{π(i)})` moves.
///
/// Naively that is `B` draws per candidate and `B` grows with the family size, so a
/// census would be quadratic in the number of pairs it screens. It is not, because
/// **the e-value depends only on the EVENT that no surrogate reaches the
/// observation**, and once one does the indicator is zero for good. Stopping at the
/// first exceedance leaves `e` exactly the same random variable and costs a
/// candidate with true p-value `p` about `1/p` draws instead of `B`. The full
/// budget is spent only on the handful of planes that are going to clear, which is
/// what makes an exhaustive exact test over tens of thousands of pairs affordable
/// at all.
///
/// The p-value pays for that. It is exact (`1/(B+1)`) for a plane that clears, and
/// for one that does not it is the inverse-binomial estimate `2/(t+1)` at the
/// stopping draw `t` — enough to rank the refusals, not a calibrated quantity. The
/// e-value, which is what the ledger consumes, is unaffected.
///
/// Returns `(p_value, e_value, rho, null_rho_mean, null_rho_sd)`. The p-value is the
/// exact lower-tail Monte-Carlo value; the e-value is the indicator
/// `(B+1)·1{no surrogate reaches the observation}`, whose null mean is 1 under
/// exchangeability with NO dependence assumption, and which is what an e-BH ledger
/// consumes.
pub fn ring_permutation_evidence(
    alpha: ArrayView1<f64>,
    beta: ArrayView1<f64>,
    replicates: usize,
    seed: u64,
) -> Result<(f64, f64, f64, f64, f64), String> {
    let n = alpha.len();
    if beta.len() != n {
        return Err(format!(
            "ring_permutation_evidence: α len {n} != β len {}",
            beta.len()
        ));
    }
    if n < 2 {
        return Err("ring_permutation_evidence: need at least 2 rows".to_string());
    }
    if replicates < 2 {
        return Err("ring_permutation_evidence: need at least 2 replicates".to_string());
    }
    // Midrank of each squared coordinate; ties share their average rank so the
    // statistic is well defined on the exactly-zero amplitudes a gate produces.
    let midranks = |v: &[f64]| -> Vec<f64> {
        let mut order: Vec<usize> = (0..v.len()).collect();
        order.sort_by(|&i, &j| v[i].total_cmp(&v[j]).then(i.cmp(&j)));
        let mut out = vec![0.0_f64; v.len()];
        let mut i = 0usize;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() && v[order[j]] == v[order[i]] {
                j += 1;
            }
            let mid = 0.5 * ((i + 1) as f64 + j as f64);
            for slot in &order[i..j] {
                out[*slot] = mid;
            }
            i = j;
        }
        out
    };
    let a: Vec<f64> = alpha.iter().map(|&v| v * v).collect();
    let b: Vec<f64> = beta.iter().map(|&v| v * v).collect();
    let ra = midranks(&a);
    let rb = midranks(&b);
    let nf = n as f64;
    let mean_r = 0.5 * (nf + 1.0);
    let centred_a: Vec<f64> = ra.iter().map(|r| r - mean_r).collect();
    let mut centred_b: Vec<f64> = rb.iter().map(|r| r - mean_r).collect();
    let ss_a: f64 = centred_a.iter().map(|r| r * r).sum();
    let ss_b: f64 = centred_b.iter().map(|r| r * r).sum();
    if !(ss_a > 0.0 && ss_b > 0.0) {
        return Err("ring_permutation_evidence: a coordinate is constant".to_string());
    }
    // Both rank sums-of-squares are permutation-invariant, so only the cross term
    // moves and ρ is an affine function of it.
    let denom = (ss_a * ss_b).sqrt();
    let rho_of = |cross: f64| cross / denom;
    let cross_obs: f64 = centred_a
        .iter()
        .zip(centred_b.iter())
        .map(|(x, y)| x * y)
        .sum();
    let rho_obs = rho_of(cross_obs);

    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut exceed = 0usize;
    let mut drawn = 0usize;
    // Null moments accumulated CENTRED (Welford): each increment
    // `(r − m_old)(r − m_new)` is a product of two same-signed factors because
    // the rounded running mean stays between `m_old` and `r`, so the spread is
    // `≥ 0` by construction and needs no clamp (#4086 sibling).
    let (mut mk, mut centred_r2) = (0.0_f64, 0.0_f64);
    for _draw in 0..replicates {
        for i in (1..n).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            centred_b.swap(i, j);
        }
        let cross: f64 = centred_a
            .iter()
            .zip(centred_b.iter())
            .map(|(x, y)| x * y)
            .sum();
        let r = rho_of(cross);
        drawn += 1;
        let before = r - mk;
        mk += before / drawn as f64;
        centred_r2 += before * (r - mk);
        if cross <= cross_obs {
            exceed = 1;
            break;
        }
    }
    let df = drawn as f64;
    let sk = (centred_r2 / df).sqrt();
    let e_value = if exceed == 0 {
        replicates as f64 + 1.0
    } else {
        0.0
    };
    let p_value = (1.0 + exceed as f64) / (df + 1.0);
    Ok((p_value, e_value, rho_obs, mk, sk))
}

/// Co-occurring signed-direction pairs counted over EVERY row/// Co-occurring signed-direction pairs counted over EVERY row, from the SPARSE
/// firing pattern rather than by scanning dense masks.
///
/// The obvious spelling is `O(K²·|rows|)`: ask, for each of the `K(K−1)/2` pairs,
/// how often both masks are set. That cost forces the caller to pass a row
/// SUBSAMPLE — and a subsample imposes a detection floor. A direction
/// that fires on a `f` fraction of rows co-fires with a partner on at most `f`
/// of a `|rows|`-row subsample, so at `f = 5·10⁻⁴` and `|rows| = 4000` the
/// expected count is `2`. Every pair carrying a rare concept is below the floor
/// before any statistic is computed. Which is to say: the screen for circles was
/// blind to exactly the population circles are found in — weekday, month,
/// small-integer and other low-frequency features.
///
/// Counting from the transposed (per-row) firing lists costs `O(Σ_r L_r²)`, where
/// `L_r` is the number of directions active on row `r` — for a sparse dictionary a
/// few hundred operations per row, independent of `K`. So the subsample can be
/// dropped entirely and the floor becomes what it should have been all along: the
/// number of co-firings the κ standard error needs.
///
/// Returns `(i, j, count)` with `count ≥ min_cooccur`, sorted by count descending.
pub fn cooccurrence_pairs_sparse(
    active: &[Vec<bool>],
    min_cooccur: usize,
) -> Vec<(usize, usize, usize)> {
    let k = active.len();
    if k < 2 {
        return Vec::new();
    }
    let n = active.iter().map(|m| m.len()).max().unwrap_or(0);
    // Transpose to per-row active lists.
    let mut per_row: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, mask) in active.iter().enumerate() {
        for (r, &on) in mask.iter().enumerate() {
            if on {
                per_row[r].push(i as u32);
            }
        }
    }
    let mut counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for row in &per_row {
        for a in 0..row.len() {
            for b in (a + 1)..row.len() {
                let key = (row[a] as u64) << 32 | row[b] as u64;
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    let mut out: Vec<(usize, usize, usize)> = counts
        .into_iter()
        .filter(|&(_, c)| c as usize >= min_cooccur)
        .map(|(key, c)| ((key >> 32) as usize, (key & 0xffff_ffff) as usize, c as usize))
        .collect();
    out.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    out
}

/// A cooldown ledger that keys on the atom-set involved in a curl / flatten move
/// so the pair cannot oscillate `curl → flatten → curl` across rounds
/// (INTEGRATION_PLAN Phase 4.5; risk #5). A move on a given atom-set is blocked
/// while its hash sits in cooldown; [`Self::tick`] decrements every entry by one
/// round, so a cooldown of `c` rounds silences that atom-set for `c` rounds
/// after either direction of the move fired on it.
#[derive(Debug, Clone, Default)]
pub struct CurlCooldownLedger {
    /// atom-set hash → rounds remaining before the move is allowed again.
    entries: std::collections::HashMap<u64, usize>,
}

/// Order-independent hash of an atom set (the cooldown key).
pub fn atom_set_hash(atoms: &[usize]) -> u64 {
    let mut sorted: Vec<usize> = atoms.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    // FNV-1a over the sorted indices — order-independent, stable across runs.
    let mut h = 0xcbf29ce484222325u64;
    for a in sorted {
        for b in (a as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

impl CurlCooldownLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a move on `atoms` is currently blocked (its atom-set is cooling).
    pub fn blocked(&self, atoms: &[usize]) -> bool {
        self.entries
            .get(&atom_set_hash(atoms))
            .is_some_and(|&r| r > 0)
    }

    /// Record that a move fired on `atoms`; silence that atom-set for `cooldown`
    /// rounds (both curl and flatten read the same ledger, so the two directions
    /// cannot chase each other).
    pub fn record(&mut self, atoms: &[usize], cooldown: usize) {
        if cooldown == 0 {
            return;
        }
        self.entries.insert(atom_set_hash(atoms), cooldown);
    }

    /// Advance one round: every cooling atom-set loses one round of cooldown.
    pub fn tick(&mut self) {
        self.entries.retain(|_, r| {
            *r = r.saturating_sub(1);
            *r > 0
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use std::f64::consts::PI;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    #[test]
    fn ring_is_recommended_kappa_near_one() {
        let mut s = 0x51A9_u64;
        let n = 4000usize;
        let radius = 3.0_f64;
        let sigma = 0.05_f64;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let th = TAU * lcg(&mut s);
            alpha[i] = radius * th.cos() + sigma * lcg_normal(&mut s);
            beta[i] = radius * th.sin() + sigma * lcg_normal(&mut s);
        }
        // occupancy high, charge modest so the RD screen pays.
        let v = curl_verdict(alpha.view(), beta.view(), sigma, n as f64, 50.0).unwrap();
        assert!((v.kappa - 1.0).abs() < 0.1, "ring κ≈1, got {}", v.kappa);
        assert!(v.recommend_curl, "clean ring must be recommended");
    }

    #[test]
    fn gaussian_fill_is_rejected_kappa_near_two() {
        let mut s = 0x6A2_u64;
        let n = 4000usize;
        let sigma = 1.0_f64;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            alpha[i] = 2.0 * lcg_normal(&mut s);
            beta[i] = 2.0 * lcg_normal(&mut s);
        }
        let v = curl_verdict(alpha.view(), beta.view(), sigma, n as f64, 50.0).unwrap();
        assert!(
            (v.kappa - 2.0).abs() < 0.15,
            "Gaussian fill κ≈2, got {}",
            v.kappa
        );
        assert!(!v.recommend_curl, "Gaussian fill must be rejected");
    }

    /// #3827: a 330° arc is not a ring. Its radius law is a perfect shell (κ = 1),
    /// so only the coverage screen can refuse it, and at n = 4000 its first
    /// resultant `R₁ = sin(w/2)/(w/2) ≈ 0.090` puts the studentized Rayleigh
    /// statistic `Q ≈ n·R₁² ≈ 32` far past the cut `−ln Φ(−2) ≈ 3.78`. The former `R₁ < 2/√n + 0.15` gate read `0.090 <
    /// 0.182` and certified this arc as a full ring at every n.
    #[test]
    fn a_330_degree_arc_is_not_covered_3827() {
        let n = 4000usize;
        let width = 330.0_f64.to_radians();
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let th = width * (i as f64 + 0.5) / n as f64;
            alpha[i] = 3.0 * th.cos();
            beta[i] = 3.0 * th.sin();
        }
        let rec = ring_recognition(alpha.view(), beta.view()).unwrap();
        let expected_r1 = (0.5 * width).sin() / (0.5 * width);
        assert!(
            (rec.resultant1 - expected_r1).abs() < 1e-6,
            "arc R₁ = sin(w/2)/(w/2) = {expected_r1}, got {}",
            rec.resultant1
        );
        assert!(
            rec.z_below_gaussian > CURL_Z,
            "the arc's radius law is a shell; κ = {}",
            rec.kappa
        );
        assert!(
            !rec.covered && !rec.recognized,
            "a 330° arc must not be certified as a covered ring (R₁ = {}, n·R₁² = {})",
            rec.resultant1,
            n as f64 * rec.resultant1 * rec.resultant1
        );
    }

    /// #3827: an ellipse is a ring of the periodic family (independent sin/cos
    /// decoder rows), not a diameter. At aspect 0.8 and uniform phase it reads
    /// `R₂ = (1 − b)/(1 + b) ≈ 0.11`, so `n·R₂² ≈ 40` at n = 4000: a Rayleigh screen
    /// on `R₂` refuses it, and reading that screen as the diameter flag demotes
    /// it to a rank-1 line. It must stay covered and recognized, and flatten must
    /// leave it standing.
    #[test]
    fn an_ellipse_is_a_covered_ring_not_a_diameter_3827() {
        let n = 4000usize;
        let mut s = 0xE11_u64;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let th = TAU * lcg(&mut s);
            alpha[i] = 3.0 * th.cos();
            beta[i] = 2.4 * th.sin();
        }
        let rec = ring_recognition(alpha.view(), beta.view()).unwrap();
        assert!(
            n as f64 * rec.resultant2 * rec.resultant2 > 20.0,
            "premise: the ellipse's R₂ is resolved (R₂ = {})",
            rec.resultant2
        );
        assert!(!rec.diameter, "an ellipse is not a diameter (R₂ = {})", rec.resultant2);
        assert!(
            rec.covered && rec.recognized,
            "a full ellipse is a covered ring (R₁ = {}, R₂ = {}, κ = {})",
            rec.resultant1,
            rec.resultant2,
            rec.kappa
        );
        let v = flatten_verdict(alpha.view(), beta.view(), alpha.view(), beta.view()).unwrap();
        assert!(!v.recommend_flatten, "a full ellipse must not be flattened");
    }

    /// #3827: the geometry screens are a calibrated test of full coverage. Over `B` independent uniform rings the refusal rate of `covered` is
    /// Binomial(B, α)/B with `α = Φ(−CURL_Z)`, so it must land within four
    /// binomial standard errors `4·√(α(1−α)/B)` of α. The former gate, with its
    /// absolute `+0.15` floor and fixed `R₂ < 0.5` cut, refused essentially no
    /// uniform ring at this n: a rate of 0, miscalibrated conservative.
    #[test]
    fn uniform_ring_refusal_rate_is_the_kappa_gate_level_3827() {
        let (b, n) = (4000usize, 100usize);
        let mut s = 0x3827_u64;
        let mut refused = 0usize;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for _ in 0..b {
            for i in 0..n {
                let th = TAU * lcg(&mut s);
                alpha[i] = th.cos();
                beta[i] = th.sin();
            }
            if !ring_recognition(alpha.view(), beta.view()).unwrap().covered {
                refused += 1;
            }
        }
        let level = normal_cdf(-CURL_Z);
        let rate = refused as f64 / b as f64;
        let band = 4.0 * (level * (1.0 - level) / b as f64).sqrt();
        assert!(
            (rate - level).abs() < band,
            "uniform-ring refusal rate {rate} must match α = {level} within {band}"
        );
    }

    /// #3827: the coverage screen is studentized by the second harmonic, so it
    /// keeps its level on an ellipse, whose angle law is not uniform. Over `B`
    /// independent aspect-½ ellipses at uniform phase (`R₂ = ⅓`) the refusal rate
    /// of `covered` must land within four binomial standard errors of α.
    #[test]
    fn ellipse_refusal_rate_is_the_kappa_gate_level_3827() {
        let (b, n) = (4000usize, 200usize);
        let mut s = 0xE111_u64;
        let mut refused = 0usize;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for _ in 0..b {
            for i in 0..n {
                let th = TAU * lcg(&mut s);
                alpha[i] = th.cos();
                beta[i] = 0.5 * th.sin();
            }
            if !ring_recognition(alpha.view(), beta.view()).unwrap().covered {
                refused += 1;
            }
        }
        let level = normal_cdf(-CURL_Z);
        let rate = refused as f64 / b as f64;
        let band = 4.0 * (level * (1.0 - level) / b as f64).sqrt();
        assert!(
            (rate - level).abs() < band,
            "ellipse refusal rate {rate} must match α = {level} within {band}"
        );
    }

    #[test]
    fn seed_reconstructs_nonorthogonal_frame_ring_to_1e10() {
        // A ring parked in a NON-orthogonal frame (u,v not perpendicular, not
        // unit). curl_seed must reconstruct the ambient points exactly.
        let mut s = 0x1234_u64;
        let n = 200usize;
        let p = 5usize;
        let u = Array1::from_vec(vec![1.0, 0.5, 0.0, -0.2, 0.0]);
        let v = Array1::from_vec(vec![0.3, 1.0, 0.4, 0.0, 0.1]);
        let center = Array1::from_vec(vec![0.7, -0.3, 0.2, 1.1, 0.0]);
        let radius = 2.0_f64;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        let mut pts = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let th = TAU * lcg(&mut s);
            let (a, b) = (radius * th.cos(), radius * th.sin());
            alpha[i] = a;
            beta[i] = b;
            for j in 0..p {
                pts[[i, j]] = center[j] + a * u[j] + b * v[j];
            }
        }
        let seed = curl_seed(
            u.view(),
            v.view(),
            alpha.view(),
            beta.view(),
            3,
            center.view(),
        )
        .unwrap();
        assert!(
            (seed.radius - radius).abs() < 1e-9,
            "radius {}",
            seed.radius
        );
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let rec = seed.reconstruct(seed.theta_turns[i]);
            for j in 0..p {
                max_err = max_err.max((rec[j] - pts[[i, j]]).abs());
            }
        }
        assert!(max_err < 1e-10, "reconstruction max err {max_err:.3e}");
    }

    /// A healthy ring image `(α, β) = r(cos θ, sin θ)`, `r ≈ 2`, θ uniform.
    fn noisy_ring(s: &mut u64, n: usize) -> (Array1<f64>, Array1<f64>) {
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = 2.0 + 0.02 * lcg_normal(s);
            let th = TAU * lcg(s);
            alpha[i] = r * th.cos();
            beta[i] = r * th.sin();
        }
        (alpha, beta)
    }

    #[test]
    fn flatten_demotes_gaussian_fill_to_rank2() {
        // A ring decoder whose rows parse an isotropic Gaussian blob: the image
        // is a ring, the data are not.
        let mut s = 0x9F1_u64;
        let n = 3000usize;
        let (image_alpha, image_beta) = noisy_ring(&mut s, n);
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            alpha[i] = 2.0 * lcg_normal(&mut s);
            beta[i] = 2.0 * lcg_normal(&mut s);
        }
        let v = flatten_verdict(image_alpha.view(), image_beta.view(), alpha.view(), beta.view())
            .unwrap();
        assert!(v.image.recognized, "the image is a ring (κ={:.3})", v.image.kappa);
        assert!(!v.parse.recognized, "a Gaussian parse is no ring (κ={:.3})", v.parse.kappa);
        assert!(v.recommend_flatten);
        assert_eq!(v.residual_rank, 2);
    }

    #[test]
    fn flatten_demotes_diameter_to_rank1() {
        let mut s = 0x33A_u64;
        let n = 3000usize;
        // amplitude along a single line θ ∈ {0, π}: a diameter.
        let alpha = Array1::from_shape_fn(n, |_| 3.0 * lcg_normal(&mut s));
        let beta = Array1::<f64>::zeros(n);
        let v = flatten_verdict(alpha.view(), beta.view(), alpha.view(), beta.view()).unwrap();
        assert!(v.image.diameter && v.parse.diameter);
        assert!(v.recommend_flatten);
        assert_eq!(v.residual_rank, 1);
    }

    /// #3506 — the segment image `α = R cos φ, β = 0` under evenly spread phases
    /// `φ`. Paired with the PHASE angle its radius law reads `κ = E[cos⁴]/E[cos²]²
    /// = 3/2` exactly, with `R₂ = 0`: a covered "ring" sitting on the old magic
    /// `κ > 1.5` cut and never flattened. Read geometrically it is a diameter.
    #[test]
    fn flatten_demotes_segment_under_uniform_phases_to_rank1_3506() {
        let n = 400usize;
        let alpha = Array1::from_shape_fn(n, |i| 3.0 * (TAU * i as f64 / n as f64).cos());
        let beta = Array1::<f64>::zeros(n);
        let v = flatten_verdict(alpha.view(), beta.view(), alpha.view(), beta.view()).unwrap();
        assert!(
            v.image.diameter,
            "a segment is a diameter (R₂ = {:.4})",
            v.image.resultant2
        );
        assert!(v.recommend_flatten);
        assert_eq!(v.residual_rank, 1);
    }

    #[test]
    fn healthy_ring_not_flattened() {
        let mut s = 0x77C_u64;
        let n = 3000usize;
        let (alpha, beta) = noisy_ring(&mut s, n);
        let v = flatten_verdict(alpha.view(), beta.view(), alpha.view(), beta.view()).unwrap();
        assert!(
            !v.recommend_flatten,
            "healthy ring must not flatten (κ={:.3} R2={:.3})",
            v.parse.kappa, v.parse.resultant2
        );
    }

    /// The flatten verdict is the exact complement of the ring recognition curl
    /// promotes on: whenever image and parse coincide, a plane flattens iff it
    /// is not recognized as a ring.
    #[test]
    fn flatten_is_the_complement_of_ring_recognition() {
        let mut s = 0xC0DE_u64;
        let n = 800usize;
        let (ring_a, ring_b) = noisy_ring(&mut s, n);
        let gauss_a = Array1::from_shape_fn(n, |_| lcg_normal(&mut s));
        let gauss_b = Array1::from_shape_fn(n, |_| lcg_normal(&mut s));
        let half_a = Array1::from_shape_fn(n, |i| (PI * i as f64 / n as f64).cos());
        let half_b = Array1::from_shape_fn(n, |i| (PI * i as f64 / n as f64).sin());
        for (alpha, beta) in [(&ring_a, &ring_b), (&gauss_a, &gauss_b), (&half_a, &half_b)] {
            let rec = ring_recognition(alpha.view(), beta.view()).unwrap();
            let v = flatten_verdict(alpha.view(), beta.view(), alpha.view(), beta.view()).unwrap();
            assert_eq!(v.recommend_flatten, !rec.recognized);
        }
    }

    #[test]
    fn coalesce_merges_four_rectified_halves_into_two_signed_axes() {
        // A centered circle in the (e0,e1) plane, shattered by a nonneg gate into
        // four rectified halves ±u, ±v with DISJOINT gates.
        let p = 6usize;
        let n = 400usize;
        let mut up = Array1::<f64>::zeros(p);
        up[0] = 1.0;
        let un = up.mapv(|x| -x); // −u
        let mut vp = Array1::<f64>::zeros(p);
        vp[1] = 1.0;
        let vn = vp.mapv(|x| -x); // −v
        // Disjoint quarter-arc gates: rows 0..100 fire +u, 100..200 +v, etc.
        let mask =
            |lo: usize, hi: usize| -> Vec<bool> { (0..n).map(|r| r >= lo && r < hi).collect() };
        let dirs = [up.view(), un.view(), vp.view(), vn.view()];
        let active = vec![mask(0, 100), mask(200, 300), mask(100, 200), mask(300, 400)];
        let ids = [10usize, 11, 12, 13];
        let signed = coalesce_antipodal(&dirs, &active, &ids, -0.9, 0.1);
        assert_eq!(
            signed.len(),
            2,
            "four halves must coalesce into two signed axes"
        );
        for sd in &signed {
            assert_eq!(sd.members.len(), 2, "each signed axis merges a ± pair");
            // Union gate covers both halves' rows (200 active).
            let active_count = sd.active.iter().filter(|b| **b).count();
            assert_eq!(active_count, 200);
        }
    }

    #[test]
    fn coalesce_leaves_a_lone_signed_atom_unmerged() {
        let p = 3usize;
        let n = 10usize;
        let mut a = Array1::<f64>::zeros(p);
        a[0] = 1.0;
        let dirs = [a.view()];
        let active = vec![vec![true; n]];
        let ids = [7usize];
        let signed = coalesce_antipodal(&dirs, &active, &ids, -0.9, 0.1);
        assert_eq!(signed.len(), 1);
        assert_eq!(signed[0].members, vec![7]);
    }

    #[test]
    fn coalesce_refuses_overlapping_antipodal_gates() {
        // Opposite directions but the SAME (fully-overlapping) gate — not a
        // rectified split (that would be a genuine two-sided line), so no merge.
        let p = 3usize;
        let n = 10usize;
        let mut a = Array1::<f64>::zeros(p);
        a[0] = 1.0;
        let b = a.mapv(|x| -x);
        let dirs = [a.view(), b.view()];
        let active = vec![vec![true; n], vec![true; n]];
        let ids = [1usize, 2];
        let signed = coalesce_antipodal(&dirs, &active, &ids, -0.9, 0.1);
        assert_eq!(signed.len(), 2, "overlapping gates must not coalesce");
    }

    /// The permutation test's two ends: a clean ring beats every surrogate, and a
    /// product of two ring marginals — same marginals, corner-clumped rather than
    /// a shell — does not. The middle assertion pins κ's whole range against this
    /// null at under 0.3, which is the arithmetic that motivated trying an angular
    /// channel and that the spike-in power arm then measured against.
    #[test]
    fn permutation_test_resolves_a_ring_its_marginals_cannot_fake() {
        let n = 600;
        let mut alpha = Array1::<f64>::zeros(n);
        let mut beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            let th = TAU * (i as f64 + 0.5) / n as f64;
            alpha[i] = th.cos();
            beta[i] = th.sin();
        }
        let (p_ring, e_ring, rho_ring, _m, _s) =
            ring_permutation_evidence(alpha.view(), beta.view(), 400, 12345)
                .expect("a clean ring must admit a permutation test");
        assert!(
            e_ring > 0.0,
            "a clean ring must beat every surrogate; p = {p_ring}, ρ = {rho_ring}"
        );
        assert!(
            rho_ring < -0.9,
            "a ring is α² + β² = R², i.e. ρ ≈ −1; got {rho_ring}"
        );

        // κ alone: the ring sits at 1.0 and its own permuted marginals at 1.25,
        // so the separation the angular channel supplies is what carries the test.
        let a: Vec<f64> = alpha.iter().map(|v| v * v).collect();
        let b: Vec<f64> = beta.iter().map(|v| v * v).collect();
        let inv = 1.0 / n as f64;
        let m2 = (a.iter().sum::<f64>() + b.iter().sum::<f64>()) * inv;
        let kappa_obs = (a.iter().map(|v| v * v).sum::<f64>() * inv
            + b.iter().map(|v| v * v).sum::<f64>() * inv
            + 2.0 * a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>() * inv)
            / (m2 * m2);
        let kappa_indep = (a.iter().map(|v| v * v).sum::<f64>() * inv
            + b.iter().map(|v| v * v).sum::<f64>() * inv
            + 2.0 * (a.iter().sum::<f64>() * inv) * (b.iter().sum::<f64>() * inv))
            / (m2 * m2);
        assert!(
            (kappa_indep - kappa_obs) < 0.3,
            "κ's whole range against this null is under 0.3 (ring {kappa_obs:.3} vs \
             permuted marginals {kappa_indep:.3}) — which is why partial structure \
             needs the angular channel"
        );

        // The matched refusal: independent draws with the SAME arcsine marginals
        // are a corner-clumped product, not a ring, and must not be accepted.
        let mut a_sh = Array1::<f64>::zeros(n);
        let mut b_sh = Array1::<f64>::zeros(n);
        for i in 0..n {
            let th = TAU * (i as f64 + 0.5) / n as f64;
            let ph = TAU * ((i as f64 * 0.6180339887).fract());
            a_sh[i] = th.cos();
            b_sh[i] = ph.sin();
        }
        let (_p_prod, e_prod, rho_prod, _mm, _ss) =
            ring_permutation_evidence(a_sh.view(), b_sh.view(), 400, 999)
                .expect("the product arm must admit a permutation test");
        assert_eq!(
            e_prod, 0.0,
            "a product of two ring marginals is not a ring and must not clear the test \
             (ρ = {rho_prod})"
        );
    }

    #[test]
    fn cooccurrence_counts_and_ranks() {
        // dir 0 and 1 co-fire on rows 0..6; dir 2 fires elsewhere.
        let active = vec![
            (0..10).map(|r| r < 6).collect::<Vec<_>>(),
            (0..10).map(|r| r < 6).collect::<Vec<_>>(),
            (0..10).map(|r| r >= 6).collect::<Vec<_>>(),
        ];
        let pairs = cooccurrence_pairs_sparse(&active, 3);
        assert_eq!(pairs.first().map(|p| (p.0, p.1, p.2)), Some((0, 1, 6)));
        // (0,2) and (1,2) never co-fire → excluded by min_cooccur.
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn cooldown_blocks_then_expires() {
        let mut led = CurlCooldownLedger::new();
        assert!(!led.blocked(&[3, 1, 2]));
        led.record(&[1, 2, 3], 2);
        // Order-independent key.
        assert!(led.blocked(&[3, 2, 1]));
        led.tick();
        assert!(led.blocked(&[1, 2, 3]));
        led.tick();
        assert!(
            !led.blocked(&[1, 2, 3]),
            "cooldown must expire after c ticks"
        );
    }
}
