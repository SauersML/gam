//! Do two INDEPENDENTLY TRAINED dictionaries find the same circle?
//!
//! `curl_census_foreign` reports each accepted plane's orthonormal ambient frame.
//! Atom indices do not survive retraining, but a 2-plane in the residual stream
//! does, so the question "is this the same circle" is the question "are these the
//! same subspace" — answered by the principal angles between the two frames.
//!
//! For frames `A, B` (both `p×2`, orthonormal columns) the singular values of
//! `AᵀB` are the cosines of the principal angles. The LARGER angle is the honest
//! summary: two planes agree only if they agree in both directions, and a pair
//! that shares one direction and nothing else is a shared atom, not a shared
//! circle.
//!
//! The scale is calibrated against the only null that matters here — random
//! 2-planes in the same ambient dimension. In `p = 2560` a random pair sits within
//! a degree or two of orthogonal, so the null is not a formality: it is what turns
//! "the angle is 12°" into a number with a probability attached.
//!
//! Usage: `plane_replication <a.planes.json> <b.planes.json> <null_draws> <out.json>`

use std::fs;
use std::path::Path;

fn frame_of(plane: &serde_json::Value, key: &str) -> Result<Vec<f64>, String> {
    plane
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("plane is missing the `{key}` frame axis"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("`{key}` holds a non-numeric entry"))
        })
        .collect()
}

/// Principal angles `(θ_max, θ_min)` in radians between two 2-planes given
/// orthonormal frames `(a1, a2)` and `(b1, b2)`. The `2×2` cross-Gram's singular
/// values are the COSINES, and a `2×2` SVD is closed form: the singular values of
/// `M` are `√((t ± √(t² − 4d²))/2)` with `t = ‖M‖_F²` and `d = det M`.
///
/// Note the inversion: `acos` is decreasing, so the LARGER singular value gives
/// the SMALLER angle. `θ_max = acos(s_min)` is the summary that matters — two
/// planes agree only if they agree in both directions, and a pair that shares one
/// direction and is orthogonal in the other is a shared atom, not a shared
/// circle.
fn principal_angles(a1: &[f64], a2: &[f64], b1: &[f64], b2: &[f64]) -> (f64, f64) {
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y.iter()).map(|(p, q)| p * q).sum::<f64>();
    let m = [dot(a1, b1), dot(a1, b2), dot(a2, b1), dot(a2, b2)];
    let t = m.iter().map(|v| v * v).sum::<f64>();
    let d = m[0] * m[3] - m[1] * m[2];
    let disc = (t * t - 4.0 * d * d).max(0.0).sqrt();
    let s_max = ((t + disc) * 0.5).max(0.0).sqrt().min(1.0);
    let s_min = ((t - disc) * 0.5).max(0.0).sqrt().min(1.0);
    (s_min.acos(), s_max.acos())
}

/// A deterministic standard normal stream (splitmix64 + Box–Muller), so the null
/// is reproducible without a dependency on any RNG's version-to-version stream.
struct Normals(u64);

impl Normals {
    fn bits(&mut self) -> u64 {
        gam_linalg::utils::splitmix64(&mut self.0)
    }

    fn unit(&mut self) -> f64 {
        (self.bits() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A uniformly random orthonormal 2-frame in `p` dimensions.
    fn frame(&mut self, p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut u: Vec<f64> = (0..p).map(|_| self.normal()).collect();
        let nu = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        u.iter_mut().for_each(|v| *v /= nu);
        let mut w: Vec<f64> = (0..p).map(|_| self.normal()).collect();
        let proj: f64 = w.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
        for (wi, ui) in w.iter_mut().zip(u.iter()) {
            *wi -= proj * ui;
        }
        let nw = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        w.iter_mut().for_each(|v| *v /= nw);
        (u, w)
    }
}

fn planes(path: &Path) -> Result<Vec<serde_json::Value>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    Ok(doc
        .get("planes")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: plane_replication <a.planes.json> <b.planes.json> <null_draws> <out.json>".to_string());
    }
    let a = planes(Path::new(&args[1]))?;
    let b = planes(Path::new(&args[2]))?;
    let draws: usize = args[3]
        .parse()
        .map_err(|e| format!("null_draws must be a positive integer: {e}"))?;
    if a.is_empty() || b.is_empty() {
        return Err(format!(
            "nothing to match: {} planes on the left, {} on the right",
            a.len(),
            b.len()
        ));
    }

    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut best_angles: Vec<f64> = Vec::new();
    let mut ambient = 0usize;
    for pa in &a {
        let a1 = frame_of(pa, "e1")?;
        let a2 = frame_of(pa, "e2")?;
        ambient = a1.len();
        let mut best = (std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2, 0usize);
        for (j, pb) in b.iter().enumerate() {
            let b1 = frame_of(pb, "e1")?;
            let b2 = frame_of(pb, "e2")?;
            if b1.len() != a1.len() {
                return Err("the two censuses have different ambient dimensions".to_string());
            }
            let (theta_max, theta_min) = principal_angles(&a1, &a2, &b1, &b2);
            if theta_max < best.0 {
                best = (theta_max, theta_min, j);
            }
        }
        best_angles.push(best.0);
        rows.push(serde_json::json!({
            "left_members": [pa.get("members_a"), pa.get("members_b")],
            "right_members": [b[best.2].get("members_a"), b[best.2].get("members_b")],
            "principal_angle_deg_max": best.0.to_degrees(),
            "principal_angle_deg_min": best.1.to_degrees(),
            "left_kappa": pa.get("kappa"),
            "right_kappa": b[best.2].get("kappa"),
        }));
    }

    // Null: the best-of-`|b|` larger principal angle a RANDOM plane achieves
    // against the same right-hand set, so the null carries the same maximisation
    // the observed statistic does. Matching without that is a free win.
    let mut rng = Normals(0x2502_C13C_1E05_u64 | 1);
    let mut null_best: Vec<f64> = Vec::with_capacity(draws);
    let right: Vec<(Vec<f64>, Vec<f64>)> = b
        .iter()
        .map(|pb| Ok((frame_of(pb, "e1")?, frame_of(pb, "e2")?)))
        .collect::<Result<_, String>>()?;
    for _draw in 0..draws {
        let (r1, r2) = rng.frame(ambient);
        let mut best = std::f64::consts::FRAC_PI_2;
        for (b1, b2) in &right {
            let (theta_max, _theta_min) = principal_angles(&r1, &r2, b1, b2);
            if theta_max < best {
                best = theta_max;
            }
        }
        null_best.push(best);
    }
    null_best.sort_by(f64::total_cmp);
    let null_p05 = null_best[((0.05 * draws as f64) as usize).min(draws.saturating_sub(1))];
    let matched = best_angles.iter().filter(|&&t| t < null_p05).count();

    let doc = serde_json::json!({
        "ambient_p": ambient,
        "left_planes": a.len(),
        "right_planes": b.len(),
        "null_draws": draws,
        "null_best_angle_deg_p05": null_p05.to_degrees(),
        "null_best_angle_deg_median": null_best[draws / 2].to_degrees(),
        "matched_below_null_p05": matched,
        "matches": rows,
    });
    fs::write(
        Path::new(&args[4]),
        serde_json::to_string(&doc).map_err(|e| format!("serialise: {e}"))?,
    )
    .map_err(|e| format!("write {}: {e}", args[4]))?;
    eprintln!(
        "[replication] {} of {} left planes match a right plane below the random-plane \
         5% point ({:.1}°); random best-of-{} median {:.1}°",
        matched,
        a.len(),
        null_p05.to_degrees(),
        b.len(),
        null_best[draws / 2].to_degrees()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    // `std::process::exit` is banned by the root build script, and one hit in
    // ANY crate aborts that script -- which means no root-package target builds
    // at all: the whole integration-test suite and the wheel. Returning the
    // error is the sanctioned idiom and keeps the same nonzero exit status.
    run().map_err(|error| format!("plane_replication: {error}"))
}
