//! O(2) classification of circle transports — the Fourier-rigidity classifier.
//!
//! **Theorem (Fourier rigidity).** Any linear map carrying an elliptical atom
//! bijectively onto an elliptical atom induces an angle map `h(θ) = ±θ + φ`:
//! writing `e(h(θ)) = u + M e(θ)` with `M = A′⁺ W A`, the identity `‖e(h)‖ ≡ 1`
//! forces (frequency-2 part) `MᵀM = λI` and (frequency-1 part) `Mᵀu = 0`, hence
//! `u = 0`, `λ = 1`, `M ∈ O(2)`. Rotary "clock arithmetic" is forced by the
//! geometry, not discovered by training.
//!
//! The classifier inverts this: from matched `(θ_in, θ_out)` samples (e.g.
//! [`FittedTransport::eval`] on a grid), `S₊ = |Σ e^{i(θ_out − θ_in)}|`,
//! `S₋ = |Σ e^{i(θ_out + θ_in)}|`; the larger resultant selects the winding, its
//! argument is `φ`, and `defect = 1 − max(S₊, S₋)/n` is the circular variance
//! about the fitted rigid map — the O(2) departure. A large defect on a pair
//! whose composition defect is small localizes harmonic mixing. All angles in
//! radians.

use ndarray::Array1;

use crate::inference::layer_transport::{ChartTopology, FittedTransport};

/// Discrete class of a fitted circle transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircleTransportClass {
    /// `θ ↦ θ + φ` (winding +1).
    Shift,
    /// `θ ↦ −θ + φ` (winding −1).
    Reflect,
    /// Neither resultant is dominant: harmonic mixing / not O(2).
    Mixing,
}

/// The Fourier-rigidity report for one circle transport.
#[derive(Debug, Clone)]
pub struct CircleTransportReport {
    /// Layers this map connects (for legibility in a ladder report).
    pub layer_from: usize,
    pub layer_to: usize,
    pub n_samples: usize,
    /// `+1` shift, `−1` reflection (winding of the recovered map).
    pub winding: i8,
    /// Phase `φ` in radians, in `(−π, π]`.
    pub phase: f64,
    /// `1 − max(S₊, S₋)/n ∈ [0, 1]`: `0` = exact O(2) element, `≈ 1` = no rigid
    /// structure.
    pub defect: f64,
    /// Resultants for both hypotheses (diagnostics).
    pub resultant_shift: f64,
    pub resultant_reflect: f64,
    pub class: CircleTransportClass,
}

impl CircleTransportReport {
    /// Phase in degrees, wrapped to `(−180, 180]` — the ladder's report line.
    pub fn phase_degrees(&self) -> f64 {
        self.phase * 180.0 / std::f64::consts::PI
    }
}

/// Classify a circle transport from paired angle samples (radians).
///
/// `Mixing` is declared when the winner's resultant fails to beat the loser by
/// more than the natural `2/√n` resultant scale — i.e. the data cannot
/// distinguish shift from reflection, which for genuine O(2) elements happens
/// only at `n` too small to matter.
pub fn classify_circle_transport(
    theta_in: &[f64],
    theta_out: &[f64],
    layer_from: usize,
    layer_to: usize,
) -> Result<CircleTransportReport, String> {
    if theta_in.len() != theta_out.len() {
        return Err("classify_circle_transport: length mismatch".to_string());
    }
    let n = theta_in.len();
    if n < 4 {
        return Err(format!(
            "classify_circle_transport: need at least 4 samples, got {n}"
        ));
    }
    let nf = n as f64;
    let (mut cp, mut sp, mut cm, mut sm) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &b) in theta_in.iter().zip(theta_out.iter()) {
        let d = b - a;
        cp += d.cos();
        sp += d.sin();
        let s = b + a;
        cm += s.cos();
        sm += s.sin();
    }
    let r_shift = (cp * cp + sp * sp).sqrt() / nf;
    let r_reflect = (cm * cm + sm * sm).sqrt() / nf;
    let (winding, phase, best, other) = if r_shift >= r_reflect {
        (1i8, sp.atan2(cp), r_shift, r_reflect)
    } else {
        (-1i8, sm.atan2(cm), r_reflect, r_shift)
    };
    let defect = 1.0 - best;
    let sep = 2.0 / nf.sqrt();
    let class = if best - other <= sep {
        CircleTransportClass::Mixing
    } else if winding == 1 {
        CircleTransportClass::Shift
    } else {
        CircleTransportClass::Reflect
    };
    Ok(CircleTransportReport {
        layer_from,
        layer_to,
        n_samples: n,
        winding,
        phase,
        defect,
        resultant_shift: r_shift,
        resultant_reflect: r_reflect,
        class,
    })
}

/// Classify a fitted transport between two CIRCLE charts by grid-sampling its
/// angle map. Returns `None` when either endpoint is not a circle (winding is an
/// O(2) notion; on intervals there is no phase). `grid ≥ 4`.
pub fn classify_circle_transport_fit(
    fit: &FittedTransport,
    from: ChartTopology,
    to: ChartTopology,
    layer_from: usize,
    layer_to: usize,
    grid: usize,
) -> Option<CircleTransportReport> {
    if !matches!(from, ChartTopology::Circle) || !matches!(to, ChartTopology::Circle) {
        return None;
    }
    let g = grid.max(4);
    let theta_in: Vec<f64> = (0..g)
        .map(|k| std::f64::consts::TAU * (k as f64) / (g as f64))
        .collect();
    let out = fit.eval(Array1::from_vec(theta_in.clone()).view()).ok()?;
    let theta_out: Vec<f64> = out.to_vec();
    classify_circle_transport(&theta_in, &theta_out, layer_from, layer_to).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn recovers_shift_phase_and_low_defect() {
        let mut s = 3u64;
        let phi = 0.9_f64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..512 {
            let th = std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI;
            a.push(th);
            b.push(th + phi + 0.01 * (lcg(&mut s) - 0.5));
        }
        let r = classify_circle_transport(&a, &b, 17, 18).unwrap();
        assert_eq!(r.class, CircleTransportClass::Shift);
        assert_eq!(r.winding, 1);
        assert!((r.phase - phi).abs() < 0.01);
        assert!(r.defect < 1e-3);
    }

    #[test]
    fn recovers_reflection() {
        let mut s = 5u64;
        let phi = -1.3_f64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..512 {
            let th = std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI;
            a.push(th);
            b.push(-th + phi);
        }
        let r = classify_circle_transport(&a, &b, 0, 1).unwrap();
        assert_eq!(r.class, CircleTransportClass::Reflect);
        assert_eq!(r.winding, -1);
        let mut dphi = r.phase - phi;
        while dphi > std::f64::consts::PI {
            dphi -= std::f64::consts::TAU;
        }
        while dphi < -std::f64::consts::PI {
            dphi += std::f64::consts::TAU;
        }
        assert!(dphi.abs() < 1e-9);
        assert!(r.defect < 1e-12);
    }

    #[test]
    fn scrambled_map_reports_mixing_defect() {
        let mut s = 9u64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for _ in 0..512 {
            a.push(std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI);
            b.push(std::f64::consts::TAU * lcg(&mut s) - std::f64::consts::PI);
        }
        let r = classify_circle_transport(&a, &b, 0, 2).unwrap();
        assert!(r.defect > 0.8);
        assert_eq!(r.class, CircleTransportClass::Mixing);
    }
}
