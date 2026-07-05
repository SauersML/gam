//! O(2) classification of circle transports — the Fourier-rigidity classifier.
//!
//! # Theory (Fourier rigidity)
//!
//! Any linear map carrying an elliptical atom bijectively onto an elliptical
//! atom induces an angle map `h(θ) = ±θ + φ`: rotary structure is *forced* by the
//! geometry. (Sketch: a linear bijection sends the source ellipse's parametric
//! circle to the target's; expanding in Fourier modes, only the `±1` modes can
//! survive a linear-in-coordinates map, so `h` is a pure ±winding plus a phase.)
//!
//! The classifier inverts this. From matched `(θ_in, θ_out)` samples,
//!
//! ```text
//! S₊ = |Σ e^{i(θ_out − θ_in)}|,   S₋ = |Σ e^{i(θ_out + θ_in)}|.
//! ```
//!
//! The larger of `S₊/n`, `S₋/n` selects the winding (`+1` rotation, `−1`
//! reflection); the winner's argument is the phase `φ`; `defect = 1 − max/n` is
//! the circular variance about the fitted rigid map — the **O(2)-departure**.
//! `defect ≈ 0` certifies the rigid class; a large defect on a pair whose own
//! composition defect is small localizes harmonic mixing.

use std::f64::consts::PI;

use ndarray::ArrayView1;

/// The O(2) class a transport falls into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportClass {
    /// `h(θ) = +θ + φ` — a rotation (winding `+1`).
    Rotation,
    /// `h(θ) = −θ + φ` — a reflection (winding `−1`).
    Reflection,
    /// Neither rigid class fits (large O(2) defect): harmonic mixing.
    Mixing,
}

/// The Fourier-rigidity report for one transport.
#[derive(Debug, Clone)]
pub struct TransportReport {
    /// The winning O(2) class.
    pub class: TransportClass,
    /// The winding degree: `+1` rotation, `−1` reflection, `0` mixing.
    pub degree: i32,
    /// The fitted phase `φ` in radians, in `(−π, π]`. Meaningful for the two
    /// rigid classes; for `Mixing` it is the argument of the stronger (still
    /// weak) resultant.
    pub phase: f64,
    /// `defect = 1 − max(S₊, S₋)/n` — the O(2) departure (circular variance
    /// about the fitted rigid map). `0` = perfectly rigid.
    pub defect: f64,
    /// `S₊/n` — the normalized rotation resultant.
    pub s_plus: f64,
    /// `S₋/n` — the normalized reflection resultant.
    pub s_minus: f64,
}

/// The defect below which a transport is certified as one of the rigid O(2)
/// classes. Above it the map is reported as `Mixing`.
pub const RIGID_DEFECT_TOL: f64 = 0.15;

/// Classify a transport from matched input/output angles (radians).
///
/// Returns the winding class, phase, and O(2) defect. `theta_in` and
/// `theta_out` must be the same length (≥ 1) and are treated as matched pairs
/// (same latent point transported through the map).
pub fn classify_transport(
    theta_in: ArrayView1<f64>,
    theta_out: ArrayView1<f64>,
) -> Result<TransportReport, String> {
    let n = theta_in.len();
    if theta_out.len() != n {
        return Err(format!(
            "classify_transport: θ_in len {n} != θ_out len {}",
            theta_out.len()
        ));
    }
    if n == 0 {
        return Err("classify_transport: no angle samples".to_string());
    }
    let inv = 1.0 / n as f64;
    // S₊ tracks θ_out − θ_in (rotation), S₋ tracks θ_out + θ_in (reflection).
    let (mut cp, mut sp, mut cm, mut sm) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let d = theta_out[i] - theta_in[i];
        let s = theta_out[i] + theta_in[i];
        cp += d.cos();
        sp += d.sin();
        cm += s.cos();
        sm += s.sin();
    }
    let s_plus = ((cp * inv).powi(2) + (sp * inv).powi(2)).sqrt();
    let s_minus = ((cm * inv).powi(2) + (sm * inv).powi(2)).sqrt();

    let (class, degree, phase, best) = if s_plus >= s_minus {
        // Rotation: φ = arg Σ e^{i(θ_out − θ_in)}.
        (TransportClass::Rotation, 1, sp.atan2(cp), s_plus)
    } else {
        // Reflection: φ = arg Σ e^{i(θ_out + θ_in)}.
        (TransportClass::Reflection, -1, sm.atan2(cm), s_minus)
    };
    let defect = 1.0 - best;
    let (class, degree) = if defect > RIGID_DEFECT_TOL {
        (TransportClass::Mixing, 0)
    } else {
        (class, degree)
    };
    Ok(TransportReport {
        class,
        degree,
        phase,
        defect,
        s_plus,
        s_minus,
    })
}

impl TransportReport {
    /// The phase in degrees, wrapped to `(−180, 180]` — the report line the
    /// transport ladder prints ("rotates the phase by 51.4°, defect 0.03").
    pub fn phase_degrees(&self) -> f64 {
        self.phase * 180.0 / PI
    }
}

/// Convenience: classify a transport sampled on a grid of source angles, given a
/// closure that maps a source angle to its transported target angle (e.g. a
/// `FittedTransport::eval` wrapped to return the output circle's angle). `grid`
/// is the number of evenly-spaced source samples on `[0, 2π)`.
pub fn classify_from_angle_map<F>(grid: usize, map: F) -> Result<TransportReport, String>
where
    F: Fn(f64) -> f64,
{
    if grid == 0 {
        return Err("classify_from_angle_map: grid must be > 0".to_string());
    }
    let mut tin = Vec::with_capacity(grid);
    let mut tout = Vec::with_capacity(grid);
    for k in 0..grid {
        let th = std::f64::consts::TAU * (k as f64) / (grid as f64);
        tin.push(th);
        tout.push(map(th));
    }
    let tin = ndarray::Array1::from_vec(tin);
    let tout = ndarray::Array1::from_vec(tout);
    classify_transport(tin.view(), tout.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::f64::consts::TAU;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn recovers_rotation_shift_and_winding() {
        let phi = 51.4_f64.to_radians();
        let n = 400usize;
        let tin: Array1<f64> = (0..n).map(|k| TAU * k as f64 / n as f64).collect();
        let tout: Array1<f64> = tin.mapv(|t| t + phi);
        let r = classify_transport(tin.view(), tout.view()).unwrap();
        eprintln!(
            "[xport rot] class={:?} deg={} φ={:.3}° defect={:.2e}",
            r.class, r.degree, r.phase_degrees(), r.defect
        );
        assert_eq!(r.class, TransportClass::Rotation);
        assert_eq!(r.degree, 1);
        assert!((r.phase_degrees() - 51.4).abs() < 1e-2, "phase {}", r.phase_degrees());
        assert!(r.defect < 1e-9, "defect {}", r.defect);
    }

    #[test]
    fn recovers_reflection() {
        let phi = 0.9_f64;
        let n = 400usize;
        let tin: Array1<f64> = (0..n).map(|k| TAU * k as f64 / n as f64).collect();
        // θ_out = −θ_in + φ ⇒ reflection.
        let tout: Array1<f64> = tin.mapv(|t| -t + phi);
        let r = classify_transport(tin.view(), tout.view()).unwrap();
        eprintln!(
            "[xport refl] class={:?} deg={} φ={:.4} defect={:.2e}",
            r.class, r.degree, r.phase, r.defect
        );
        assert_eq!(r.class, TransportClass::Reflection);
        assert_eq!(r.degree, -1);
        // For reflection, the recovered phase is φ (the argument of Σe^{i(θout+θin)}).
        let wrapped = ((r.phase - phi + PI).rem_euclid(TAU)) - PI;
        assert!(wrapped.abs() < 1e-9, "reflection phase off by {wrapped}");
        assert!(r.defect < 1e-9);
    }

    #[test]
    fn scrambled_map_reports_mixing() {
        let mut s = 0xBADF00D_u64;
        let n = 400usize;
        let tin: Array1<f64> = (0..n).map(|k| TAU * k as f64 / n as f64).collect();
        // Random target angles: no rigid structure.
        let tout: Array1<f64> = (0..n).map(|_| TAU * lcg(&mut s)).collect();
        let r = classify_transport(tin.view(), tout.view()).unwrap();
        eprintln!(
            "[xport scram] class={:?} S₊={:.3} S₋={:.3} defect={:.3}",
            r.class, r.s_plus, r.s_minus, r.defect
        );
        assert_eq!(r.class, TransportClass::Mixing);
        assert_eq!(r.degree, 0);
        assert!(r.defect > 0.8, "scrambled defect should be large, got {}", r.defect);
    }

    #[test]
    fn angle_map_convenience_matches() {
        let phi = 0.3_f64;
        let r = classify_from_angle_map(256, |t| t + phi).unwrap();
        assert_eq!(r.class, TransportClass::Rotation);
        assert!((r.phase - phi).abs() < 1e-9);
    }
}
