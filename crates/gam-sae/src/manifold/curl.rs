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
//!   * the rate–distortion pre-screen `n_eff·ln(R̂/σ) − Δcharge`, with the
//!     derived crossover `R̂ > σ·√3/π` below which a circle cannot pay for
//!     itself.
//!
//! Influence-function SEs make the κ gate a 2σ screen; the engine's existing
//! topology race stays the judge — `curl` only submits a race-ready seed.
//!
//! `flatten` is the inverse demotion, so the move pair is falsifiable *inside*
//! the dictionary's life, not just at birth: a Gaussian-fill radius law demotes
//! a circle to a rank-2 flat plane, a diameter collapse demotes it to rank-1.

use std::f64::consts::{PI, TAU};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// The rate–distortion crossover radius factor `√3/π`: below `R̂ = σ·√3/π` a
/// centered circle cannot pay its charge, so the screen refuses it regardless of
/// κ. Derived (memo 1), not tuned.
pub const RD_CROSSOVER_FACTOR: f64 = 0.5513288954217920; // √3 / π

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
    /// `R̂ = √(E[r²])` — the fitted radius.
    pub radius: f64,
    /// `ln(R̂/σ)` — the per-row coding gain.
    pub gain_nats_per_row: f64,
    /// `n_eff·ln(R̂/σ) − Δcharge` — the net evidence for the circle.
    pub net_evidence_nats: f64,
    /// True ⇒ recommend submitting the `CircleSeed` to the race.
    pub recommend_curl: bool,
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
        return Err(format!(
            "curl: α len {n} != β len {}",
            beta.len()
        ));
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
    // grad = (−2 m₄/m₂³, 1/m₂²). Var(m̂₂)=(E[r⁴]−m₂²)/n, Var(m̂₄)=(E[r⁸]−m₄²)/n,
    // Cov=(E[r⁶]−m₂m₄)/n. We fold E[r⁶] into the two we have via Cauchy proxy is
    // unnecessary — accumulate it directly.
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

/// Circular resultants `R₁ = |E[e^{iθ}]|`, `R₂ = |E[e^{2iθ}]|` of the parse
/// angles `θ_i = atan2(β_i, α_i)`.
fn circular_resultants(alpha: ArrayView1<f64>, beta: ArrayView1<f64>) -> (f64, f64) {
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
    let r1 = ((c1 * inv).powi(2) + (s1 * inv).powi(2)).sqrt();
    let r2 = ((c2 * inv).powi(2) + (s2 * inv).powi(2)).sqrt();
    (r1, r2)
}

/// Adjudicate a candidate flat pair `(α, β)` for promotion to a circle.
///
/// `sigma` is the ambient per-coordinate noise scale (the RD reference),
/// `n_eff = Σ a²` the pattern's effective occupancy (NOT the raw row count),
/// `delta_charge` the module-4 charge at that occupancy. The recommendation is a
/// conjunction: κ resolvably below the Gaussian-fill value 2 (2σ), the RD screen
/// paying (`R̂ > σ·√3/π` and net evidence positive), full angular coverage
/// (`R₁` small), and no diameter degeneracy (`R₂` not saturated).
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
    let (resultant1, resultant2) = circular_resultants(alpha, beta);
    let radius = law.m2.sqrt();
    let gain_nats_per_row = (radius / sigma).ln();
    let net_evidence_nats = n_eff * gain_nats_per_row - delta_charge;
    let z_below_gaussian = if law.kappa_se > 0.0 {
        (2.0 - law.kappa) / law.kappa_se
    } else if law.kappa < 2.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Coverage / degeneracy screens: a full ring has R₁ ≈ 0 and R₂ ≈ 0; a
    // diameter (line through the origin) has R₂ ≈ 1. Screen R₁, R₂ at the same
    // 2σ level using the uniform-null SE 1/√n for each resultant.
    let res_se = 1.0 / (law.n as f64).sqrt();
    let coverage_ok = resultant1 < CURL_Z * res_se + 0.15; // lenient absolute floor
    let not_diameter = resultant2 < 0.5;

    let rd_pays = radius > sigma * RD_CROSSOVER_FACTOR && net_evidence_nats > 0.0;
    let recommend_curl =
        z_below_gaussian > CURL_Z && rd_pays && coverage_ok && not_diameter;

    Ok(CurlVerdict {
        kappa: law.kappa,
        kappa_se: law.kappa_se,
        z_below_gaussian,
        resultant1,
        resultant2,
        radius,
        gain_nats_per_row,
        net_evidence_nats,
        recommend_curl,
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
#[derive(Debug, Clone)]
pub struct FlattenVerdict {
    /// `κ` of the radius law (`≈ 2` ⇒ Gaussian fill ⇒ rank-2 plane).
    pub kappa: f64,
    /// Second resultant (`≈ 1` ⇒ diameter collapse ⇒ rank-1 line).
    pub resultant2: f64,
    /// The residual rank the circle should be demoted to: `2` = flat plane
    /// (Gaussian fill), `1` = line (diameter collapse). Meaningful only when
    /// `recommend_flatten`.
    pub residual_rank: usize,
    /// True ⇒ this "circle" is not carrying rotational structure and should be
    /// demoted.
    pub recommend_flatten: bool,
}

/// Adjudicate whether a fitted circle has degenerated and should be flattened.
///
/// `radii` and `angles` are the per-row fitted polar coordinates of the atom.
/// A Gaussian-fill radius law (κ ≈ 2) means the "circle" is really a flat 2-D
/// Gaussian blob — demote to rank 2. A diameter collapse (second resultant ≈ 1,
/// the angle mass on one line) means it is rank 1. A healthy ring (κ ≈ 1, angles
/// covering the circle) is left alone.
pub fn flatten_verdict(
    radii: ArrayView1<f64>,
    angles: ArrayView1<f64>,
) -> Result<FlattenVerdict, String> {
    let n = radii.len();
    if angles.len() != n {
        return Err("flatten_verdict: radii/angles length mismatch".to_string());
    }
    if n < 2 {
        return Err("flatten_verdict: need at least 2 rows".to_string());
    }
    let alpha: Array1<f64> = (0..n).map(|i| radii[i] * angles[i].cos()).collect();
    let beta: Array1<f64> = (0..n).map(|i| radii[i] * angles[i].sin()).collect();
    let law = radius_law(alpha.view(), beta.view())?;
    let (_r1, resultant2) = circular_resultants(alpha.view(), beta.view());

    // Diameter collapse takes precedence: even a Gaussian-looking radius law on a
    // single line is a rank-1 structure.
    let diameter = resultant2 > 0.7;
    let gaussian_fill = law.kappa > 1.5;
    let (recommend_flatten, residual_rank) = if diameter {
        (true, 1)
    } else if gaussian_fill {
        (true, 2)
    } else {
        (false, 2)
    };
    Ok(FlattenVerdict {
        kappa: law.kappa,
        resultant2,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

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
        eprintln!(
            "[curl ring] κ={:.4}±{:.4} z={:.2} R1={:.3} R2={:.3} R̂={:.3} net={:.1} rec={}",
            v.kappa, v.kappa_se, v.z_below_gaussian, v.resultant1, v.resultant2, v.radius,
            v.net_evidence_nats, v.recommend_curl
        );
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
        eprintln!(
            "[curl gauss] κ={:.4}±{:.4} z={:.2} rec={}",
            v.kappa, v.kappa_se, v.z_below_gaussian, v.recommend_curl
        );
        assert!((v.kappa - 2.0).abs() < 0.15, "Gaussian fill κ≈2, got {}", v.kappa);
        assert!(!v.recommend_curl, "Gaussian fill must be rejected");
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
        let seed = curl_seed(u.view(), v.view(), alpha.view(), beta.view(), 3, center.view())
            .unwrap();
        assert!((seed.radius - radius).abs() < 1e-9, "radius {}", seed.radius);
        let mut max_err = 0.0_f64;
        for i in 0..n {
            let rec = seed.reconstruct(seed.theta_turns[i]);
            for j in 0..p {
                max_err = max_err.max((rec[j] - pts[[i, j]]).abs());
            }
        }
        assert!(max_err < 1e-10, "reconstruction max err {max_err:.3e}");
    }

    #[test]
    fn flatten_demotes_gaussian_fill_to_rank2() {
        let mut s = 0x9F1_u64;
        let n = 3000usize;
        let mut radii = Array1::<f64>::zeros(n);
        let mut angles = Array1::<f64>::zeros(n);
        for i in 0..n {
            let a = 2.0 * lcg_normal(&mut s);
            let b = 2.0 * lcg_normal(&mut s);
            radii[i] = (a * a + b * b).sqrt();
            angles[i] = b.atan2(a);
        }
        let v = flatten_verdict(radii.view(), angles.view()).unwrap();
        eprintln!("[flatten gauss] κ={:.3} R2={:.3} rank={} rec={}", v.kappa, v.resultant2, v.residual_rank, v.recommend_flatten);
        assert!(v.recommend_flatten);
        assert_eq!(v.residual_rank, 2);
    }

    #[test]
    fn flatten_demotes_diameter_to_rank1() {
        let mut s = 0x33A_u64;
        let n = 3000usize;
        let mut radii = Array1::<f64>::zeros(n);
        let mut angles = Array1::<f64>::zeros(n);
        for i in 0..n {
            // amplitude along a single line θ ≈ 0 (or π): a diameter.
            let g = 3.0 * lcg_normal(&mut s);
            radii[i] = g.abs();
            angles[i] = if g >= 0.0 { 0.0 } else { PI };
        }
        let v = flatten_verdict(radii.view(), angles.view()).unwrap();
        eprintln!("[flatten diam] κ={:.3} R2={:.3} rank={}", v.kappa, v.resultant2, v.residual_rank);
        assert!(v.recommend_flatten);
        assert_eq!(v.residual_rank, 1);
    }

    #[test]
    fn healthy_ring_not_flattened() {
        let mut s = 0x77C_u64;
        let n = 3000usize;
        let mut radii = Array1::<f64>::zeros(n);
        let mut angles = Array1::<f64>::zeros(n);
        for i in 0..n {
            radii[i] = 2.0 + 0.02 * lcg_normal(&mut s);
            angles[i] = TAU * lcg(&mut s);
        }
        let v = flatten_verdict(radii.view(), angles.view()).unwrap();
        assert!(!v.recommend_flatten, "healthy ring must not flatten (κ={:.3} R2={:.3})", v.kappa, v.resultant2);
    }
}
