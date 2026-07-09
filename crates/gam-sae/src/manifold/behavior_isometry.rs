//! #2015 — the per-atom **representation–behavior isometry defect**: the reported
//! statistic that turns the steering paper's isometry *assumption* into a
//! *measured* per-atom quantity of the two-block fit.
//!
//! # What it measures
//!
//! A Rung-2 two-block atom decodes ONE shared latent coordinate `t` into both an
//! activation image `x(t) = Φ_k(t) B_k` and a nats-unit behavior image
//! `y(t) = Φ_k(t) C_k` (the `√λ_y` un-done, [`BehaviorBlock::split_decoder`]). The
//! steering paper's headline claim is that the representation manifold and the
//! behavior manifold are **scaled-isometric** on natural data: moving along the
//! shared coordinate changes activation and behavior in lock-step, up to one
//! global scale. In the induced 1-D geometry that is exactly the statement
//!
//! ```text
//!   s_x(t) / s_y(t) = const across t,     s_x = ‖dx/dt‖,  s_y = ‖dy/dt‖ .
//! ```
//!
//! The two induced metrics are proportional iff their speed ratio is constant.
//! This module reports, per atom, the **speed ratio** `r(t) = s_x(t)/s_y(t)`:
//! its (support-weighted) mean is the isometry **scale**, and its coefficient of
//! variation is the isometry **defect** — `0` ⟺ an exact scaled isometry, large
//! ⟺ the correspondence between representation and behavior bends along the atom.
//! An atom with a high defect is *flagged*, not silently distorted (issue #2015,
//! point 1).
//!
//! # Why the defect is gauge-invariant (no chart canonicalization needed)
//!
//! Both speeds are taken with respect to the *same* latent `t`. Under any
//! reparameterization `t ↦ u(t)` both `s_x` and `s_y` are multiplied by the same
//! Jacobian `|dt/du|`, so their **ratio is invariant**. The defect therefore does
//! not depend on the residual `Diff(S¹)` / `Diff([0,1])` chart gauge the fit
//! happened to land in — unlike the within-chart arc-length defect
//! ([`crate::manifold::atom_coordinate_fidelity`]), which measures a property of
//! the parameterization itself. The two are complementary: arc-length defect asks
//! "is the chart honest?"; isometry defect asks "does the honest chart carry the
//! SAME geometry in activation and in behavior?".
//!
//! # Calibration readout
//!
//! `s_y(t)²` is the behavioral dose in nats per unit `t²`
//! ([`SphereTangentEmbedding::predicted_nats`]): a latent step `Δt` costs
//! `s_y(t)²·Δt²` nats. Its mean is reported as [`AtomBehaviorIsometry::nats_per_unit_t`]
//! — the raw calibration quantity the unit-speed re-gauge (issue #2015, point 3;
//! the `= 2` kill test on #1942) drives to a constant. This module reports it;
//! pinning it to `2` is the follow-up arc-length re-gauge of the behavior block.

use ndarray::ArrayView1;

use crate::chart_canonicalization::curve_speeds;

use super::{SaeManifoldTerm, SupportMeasure};

/// The per-atom representation–behavior isometry certificate of a fitted Rung-2
/// two-block atom. Produced by [`atom_behavior_isometry`]; `None` for atoms
/// without a `d = 1` chart or when no behavior block is installed.
#[derive(Debug, Clone)]
pub struct AtomBehaviorIsometry {
    /// The atom this certificate is for.
    pub atom_idx: usize,
    /// Number of positive-support rows the speeds were read at.
    pub n_rows: usize,
    /// Total occupancy mass `Σ_i w_i` from the shared atom support measure.
    pub support_mass: f64,
    /// `true` iff the behavior image actually moves along the latent (behavior
    /// speed RMS above the numerical floor). A behaviorally inert atom (`C_k ≈ 0`,
    /// constant behavior) has `false` here and a `NaN` defect/scale — there is no
    /// correspondence to certify, reported honestly rather than as a defect.
    pub behavior_engaged: bool,
    /// Support-weighted RMS of the activation induced speed `s_x = ‖dx/dt‖`.
    pub activation_speed_rms: f64,
    /// Support-weighted RMS of the nats-unit behavior induced speed `s_y = ‖dy/dt‖`.
    pub behavior_speed_rms: f64,
    /// The isometry **scale**: support-weighted mean of `r = s_x/s_y` over the
    /// rows where behavior moves. Activation length per unit behavior length.
    /// `NaN` when `behavior_engaged == false`.
    pub scale: f64,
    /// The isometry **defect**: support-weighted coefficient of variation of
    /// `r = s_x/s_y`, `std(r)/mean(r)`. `0` ⟺ an exact scaled isometry; grows as
    /// the representation–behavior correspondence bends along the atom. `NaN` when
    /// `behavior_engaged == false`.
    pub defect_cv: f64,
    /// `min r / scale` over the rows — how much slower (relative to its mean) the
    /// activation moves per unit behavior at its most compressed row. `NaN` when
    /// not engaged.
    pub min_ratio_over_scale: f64,
    /// `max r / scale` over the rows — the most stretched row. `NaN` when not
    /// engaged.
    pub max_ratio_over_scale: f64,
    /// Support-weighted mean of `s_y²` — the behavioral dose in **nats per unit
    /// `t²`** ([`super::SphereTangentEmbedding::predicted_nats`]). The unit-speed
    /// re-gauge drives this to a constant (`≈ 2` in the calibrated gauge); this is
    /// the raw readout. `NaN` when not engaged.
    pub nats_per_unit_t: f64,
}

/// Numerical floor below which a behavior induced speed is treated as zero (the
/// atom does not move behavior at that row). Relative to the atom's own maximum
/// behavior speed so it is scale-free; a small absolute companion floor guards
/// the all-zero (inert) atom.
const BEHAVIOR_SPEED_REL_FLOOR: f64 = 1.0e-9;

/// Build the representation–behavior isometry certificate for one fitted atom, or
/// `None` when the atom has no `d = 1` chart (the induced-speed construction is
/// 1-D) or when no behavior block is installed on the term (there is no y-block to
/// compare the activation geometry against).
///
/// The row set mirrors the other per-atom diagnostics: all of the atom's fitted
/// coordinate rows, support-weighted by the shared atom support measure so
/// low-occupancy rows do not dominate the ratio statistics.
pub fn atom_behavior_isometry(
    term: &SaeManifoldTerm,
    atom_idx: usize,
) -> Result<Option<AtomBehaviorIsometry>, String> {
    // A behavior block is required: without a y-block there is no second geometry.
    let Some(block) = term.behavior_block().cloned() else {
        return Ok(None);
    };
    // The induced-speed construction (curve_speeds) is a 1-D latent property; use
    // the same d = 1 gate the in-loop unit-speed retraction uses.
    if term.d1_unit_speed_topology(atom_idx).is_none() {
        return Ok(None);
    }
    let atom = &term.atoms[atom_idx];
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!("atom_behavior_isometry: atom {atom_idx} has no basis evaluator")
    })?;

    // Split the fitted augmented decoder [B_k | √λ_y C_k] into the activation
    // decoder B_k and the nats-unit behavior decoder C_k (the √λ_y un-done).
    let (mut b_k, mut c_k) = block.split_decoder(atom.decoder_coefficients.view())?;
    // Scale quotient (#2099): the atom's physical decoder is `exp(s_k)·[B_k | C_k]`
    // — under the unit-Frobenius peel the magnitude lives in the explicit
    // log-amplitude `s_k`, and `decoder_coefficients` carries only the shape. Both
    // split blocks share the SINGLE per-atom amplitude (the whole augmented
    // decoder is scaled by `exp(s_k)` in the contribution `exp(s_k)·Φ·[B|√λ_y C]`),
    // so both induced speeds `s_x = ‖Φ'·exp(s_k)B_k‖` and `s_y = ‖Φ'·exp(s_k)C_k‖`
    // pick up the SAME factor. The ratio statistics (`scale = mean(s_x/s_y)`,
    // `defect_cv`, `min/max_ratio_over_scale`) are therefore bit-identical either
    // way — `exp(s_k)` cancels in every ratio — but the ABSOLUTE readouts
    // (`activation_speed_rms`, `behavior_speed_rms`, and the calibration dose
    // `nats_per_unit_t = mean(s_y²)`, which is a physical nats-per-unit-t² quantity)
    // must reflect the PHYSICAL scale, so they consume `exp(s_k)·B_k`/`exp(s_k)·C_k`
    // here. Guarded on `!= 0.0` exactly as `SaeManifoldAtom`'s forward paths are, so
    // at the default (quotient off, `s_k = 0`) this is a bit-for-bit no-op.
    if atom.log_amplitude != 0.0 {
        let amp = atom.log_amplitude.exp();
        b_k.mapv_inplace(|v| v * amp);
        c_k.mapv_inplace(|v| v * amp);
    }

    let coords = term.assignment.coords[atom_idx].as_matrix().to_owned();
    if coords.ncols() != 1 {
        return Ok(None);
    }

    // One jet evaluation at the fitted rows, contracted with each decoder block:
    // s_x = ‖Φ'(t) B_k‖, s_y = ‖Φ'(t) C_k‖ (nats-unit).
    let (_phi, jet) = evaluator.evaluate(coords.view())?;
    let s_x = curve_speeds(&jet, b_k.view())?;
    let s_y = curve_speeds(&jet, c_k.view())?;
    if s_x.len() != coords.nrows() || s_y.len() != coords.nrows() {
        return Err(format!(
            "atom_behavior_isometry: speed profiles have lengths {}/{} but atom has {} rows",
            s_x.len(),
            s_y.len(),
            coords.nrows()
        ));
    }

    let support = SupportMeasure::from_assignment(&term.assignment, atom_idx)?;
    let weights = support.weights();
    if weights.len() != s_x.len() {
        return Err(format!(
            "atom_behavior_isometry: support has {} rows but atom has {}",
            weights.len(),
            s_x.len()
        ));
    }

    Ok(Some(assemble(atom_idx, &s_x, &s_y, weights, support.mass())))
}

/// The representation–behavior isometry certificate for every atom of the term
/// (in atom order); each entry is `None` when [`atom_behavior_isometry`] returns
/// `None` for that atom (non-`d = 1`, or no behavior block installed).
pub fn behavior_isometry_report(
    term: &SaeManifoldTerm,
) -> Result<Vec<Option<AtomBehaviorIsometry>>, String> {
    (0..term.atoms.len())
        .map(|k| atom_behavior_isometry(term, k))
        .collect()
}

/// Assemble the certificate from the two speed profiles and the support weights.
/// Split out so the arithmetic is unit-testable without a fitted term.
fn assemble(
    atom_idx: usize,
    s_x: &[f64],
    s_y: &[f64],
    weights: ArrayView1<'_, f64>,
    support_mass: f64,
) -> AtomBehaviorIsometry {
    // Weighted RMS of each speed over positive-support rows.
    let mut mass = 0.0_f64;
    let mut sx_sq = 0.0_f64;
    let mut sy_sq = 0.0_f64;
    let mut sy_max = 0.0_f64;
    let mut n_rows = 0usize;
    for i in 0..s_x.len() {
        let w = weights[i];
        if !(w > 0.0) {
            continue;
        }
        n_rows += 1;
        mass += w;
        sx_sq += w * s_x[i] * s_x[i];
        sy_sq += w * s_y[i] * s_y[i];
        sy_max = sy_max.max(s_y[i]);
    }
    let activation_speed_rms = if mass > 0.0 { (sx_sq / mass).sqrt() } else { f64::NAN };
    let behavior_speed_rms = if mass > 0.0 { (sy_sq / mass).sqrt() } else { f64::NAN };
    let nats_per_unit_t = if mass > 0.0 { sy_sq / mass } else { f64::NAN };

    // A behaviorally inert atom (C_k ≈ 0) has no behavior geometry to match.
    let floor = BEHAVIOR_SPEED_REL_FLOOR * sy_max;
    let behavior_engaged = mass > 0.0 && sy_max > 0.0;
    if !behavior_engaged {
        return AtomBehaviorIsometry {
            atom_idx,
            n_rows,
            support_mass,
            behavior_engaged: false,
            activation_speed_rms,
            behavior_speed_rms,
            scale: f64::NAN,
            defect_cv: f64::NAN,
            min_ratio_over_scale: f64::NAN,
            max_ratio_over_scale: f64::NAN,
            nats_per_unit_t,
        };
    }

    // Weighted mean and variance of r = s_x/s_y over rows where behavior moves.
    let mut r_mass = 0.0_f64;
    let mut r_mean = 0.0_f64;
    let mut r_min = f64::INFINITY;
    let mut r_max = f64::NEG_INFINITY;
    let mut ratios: Vec<(f64, f64)> = Vec::with_capacity(s_x.len());
    for i in 0..s_x.len() {
        let w = weights[i];
        if !(w > 0.0) || s_y[i] <= floor {
            continue;
        }
        let r = s_x[i] / s_y[i];
        r_mass += w;
        r_mean += w * r;
        r_min = r_min.min(r);
        r_max = r_max.max(r);
        ratios.push((r, w));
    }
    if !(r_mass > 0.0) {
        // Every moving row was floored out — treat as not engaged.
        return AtomBehaviorIsometry {
            atom_idx,
            n_rows,
            support_mass,
            behavior_engaged: false,
            activation_speed_rms,
            behavior_speed_rms,
            scale: f64::NAN,
            defect_cv: f64::NAN,
            min_ratio_over_scale: f64::NAN,
            max_ratio_over_scale: f64::NAN,
            nats_per_unit_t,
        };
    }
    r_mean /= r_mass;
    let mut r_var = 0.0_f64;
    for (r, w) in &ratios {
        let d = r - r_mean;
        r_var += w * d * d;
    }
    r_var /= r_mass;
    let defect_cv = if r_mean != 0.0 {
        r_var.sqrt() / r_mean.abs()
    } else {
        f64::NAN
    };

    AtomBehaviorIsometry {
        atom_idx,
        n_rows,
        support_mass,
        behavior_engaged: true,
        activation_speed_rms,
        behavior_speed_rms,
        scale: r_mean,
        defect_cv,
        min_ratio_over_scale: r_min / r_mean,
        max_ratio_over_scale: r_max / r_mean,
        nats_per_unit_t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// A constant speed ratio (scaled isometry) reports zero defect and a scale
    /// equal to the constant ratio, and the calibration reads the behavior speed².
    #[test]
    fn constant_ratio_is_zero_defect() {
        let n = 32usize;
        // s_y varies across rows; s_x = 3·s_y exactly ⇒ r ≡ 3, defect 0.
        let s_y: Vec<f64> = (0..n).map(|i| 0.5 + 0.4 * (i as f64 / n as f64)).collect();
        let s_x: Vec<f64> = s_y.iter().map(|&v| 3.0 * v).collect();
        let weights = Array1::<f64>::ones(n);
        let cert = assemble(0, &s_x, &s_y, weights.view(), n as f64);
        assert!(cert.behavior_engaged);
        assert!((cert.scale - 3.0).abs() < 1e-12, "scale {}", cert.scale);
        assert!(cert.defect_cv < 1e-12, "defect {}", cert.defect_cv);
        // nats/unit t = weighted mean of s_y².
        let want: f64 = s_y.iter().map(|v| v * v).sum::<f64>() / n as f64;
        assert!((cert.nats_per_unit_t - want).abs() < 1e-12);
    }

    /// A varying speed ratio (broken isometry) reports a strictly positive defect
    /// whose value matches the CV of the planted ratio.
    #[test]
    fn varying_ratio_reports_positive_defect() {
        let n = 40usize;
        let s_y: Vec<f64> = vec![1.0; n];
        // r_i = 1 + 0.5·cos(2π i/n): mean 1, so CV = std = 0.5/√2.
        let s_x: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos())
            .collect();
        let weights = Array1::<f64>::ones(n);
        let cert = assemble(0, &s_x, &s_y, weights.view(), n as f64);
        let want_cv = (0.5_f64 * 0.5 / 2.0).sqrt(); // std of 0.5·cos over a full period
        assert!(
            (cert.defect_cv - want_cv).abs() < 1e-2,
            "defect {} vs expected {want_cv}",
            cert.defect_cv
        );
        assert!(cert.defect_cv > 0.2);
    }

    /// An inert behavior block (all behavior speeds zero) is reported as not
    /// engaged with a NaN defect — no correspondence to certify, not a defect.
    #[test]
    fn inert_behavior_is_not_engaged() {
        let n = 16usize;
        let s_x: Vec<f64> = vec![1.0; n];
        let s_y: Vec<f64> = vec![0.0; n];
        let weights = Array1::<f64>::ones(n);
        let cert = assemble(0, &s_x, &s_y, weights.view(), n as f64);
        assert!(!cert.behavior_engaged);
        assert!(cert.defect_cv.is_nan());
        assert!(cert.scale.is_nan());
    }
}
