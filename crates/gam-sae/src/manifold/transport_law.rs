//! Geometric correspondence between two layer images of a circle atom, and its
//! phase-shift law.
//!
//! # The thesis this module tests geometrically
//!
//! "**Binding is transport.** Layers act through a transport groupoid; LINEAR
//! transport of an elliptical (circle) atom is forced to be a phase shift
//! `t ↦ ±t + φ`; the residual gauge obstruction is the atom's linear stabilizer."
//! (module header of [`crate::manifold`]). This module turns that claim into a
//! *geometric* measurement on a fitted 2-layer crosscoder — a shared chart
//! coordinate `t` decoded through per-layer honest decoders `B^(ℓ)` and `B^(ℓ+1)`
//! (the [`CrosscoderLayout`]/[`SaeManifoldTerm::layer_decoder`] bookkeeping).
//!
//! # Operational definition: the nearest-point correspondence
//!
//! Both layers of a crosscoder share the SAME ambient residual-stream dimension,
//! so the atom image at layer `ℓ` (`C^(ℓ) = {Φ_k(t) B^(ℓ)_k}`) and at layer `ℓ+1`
//! (`C^(ℓ+1) = {Φ_k(t) B^(ℓ+1)_k}`) are two curves in one `ℝ^p`. With no network in
//! hand, this module relates the two images by NEAREST POINT. For a source
//! coordinate `t`, decode the SOURCE (layer `ℓ`) image `x(t) = Φ_k(t) B^(ℓ)_k`, then
//! PROJECT `x(t)` onto the CONTINUOUS TARGET (layer `ℓ+1`) atom image to read off
//! the chart coordinate that best reproduces it,
//! `t'(t) = argmin_{t'} ‖x(t) − Φ_k(t') B^(ℓ+1)_k‖²`. The correspondence is
//! `t ↦ t'(t)`. The target projection enumerates every stationary point of this
//! trigonometric polynomial through its companion-matrix roots, so `t'` is not
//! quantized by any sampling lattice.
//!
//! This is a correspondence between two decoder images, not the network's
//! transport, which runs the block between the layers. Two layers that decode the
//! same circle correspond by the identity whatever that block does, even when it
//! rotates the circle. [`crate::response::executed_transport`] measures the
//! executed transport `τ(t; c) = E_{ℓ+1}(T_ℓ(c + D_ℓ(t)) − T_ℓ(c))` next to this
//! correspondence, and pins that planted rotation as its positive control.
//!
//! (The mission brief phrased the grid step as "decode at layer ℓ+1 … project
//! back onto the layer-(ℓ+1) atom image", which is the identity map; the
//! thesis-relevant measurement is decode at ONE layer and project onto the OTHER,
//! implemented here as source = anchor layer `ℓ`, target = output block `ℓ+1`.
//! The projection target IS the layer-(ℓ+1) image, as the brief's clause reads.)
//!
//! # The law test (period-aware circular regression)
//!
//! On the unit circle (chart period `1`, [`LatentManifold::Circle`]) the natural
//! squared error between two coordinates `a, b` is the chordal
//! `c(a,b) = 1 − cos(2π(a − b))` (half the squared chord; `0` iff `a ≡ b`,
//! period-correct with no unwrapping). The report carries the circular
//! coefficient of determination `R² = 1 − SS_res/SS_tot` of the **phase-shift
//! model** `t' = s·t + φ`, `s ∈ {+1, −1}` — the LAW — against the circular-mean
//! baseline, both taken over the uniform measure on the source chart:
//! `SS_tot = 1 − |ρ_tot|` with `ρ_tot = ∫₀¹ e^{i 2π t'(t)} dt`. The optimal `φ` at
//! fixed `s` is the circular mean of `u(t) = t'(t) − s·t`, which maximizes
//! `∫₀¹ cos(2π(u − φ)) dt = |ρ_s|` with `ρ_s = ∫₀¹ e^{i 2π u(t)} dt`; `s` is chosen
//! for the larger resultant, and `SS_res = 1 − |ρ_s|`.
//!
//! `phase_r2 = 1` exactly when the correspondence is a phase shift everywhere on the
//! chart (`NaN` when every corresponding coordinate coincides, a degenerate baseline); its
//! shortfall is the chordal residual the law leaves unexplained, and
//! [`AtomTransportReport::deviation_locus`] reports the chart location where the
//! phase model deviates most (the interesting locus). No smooth-map alternative
//! is fitted and no verdict threshold is applied: the alternative's Fourier order
//! and a gap tolerance would both be tuning constants that no measured transport
//! identifies.
//!
//! # The resultants are integrals, not grid sums
//!
//! There is no sampling resolution to choose. `t'(t)` is analytic except where two
//! local minimizers of the projection objective tie and the nearest point jumps.
//! The three resultants `ρ_tot`, `ρ_+` and `ρ_−` are integrated cell by cell with
//! the `m`-point Gauss–Legendre rule, `m = 2H + 1` the atom's harmonic width,
//! starting from `m` equal cells of `[0, 1)`. A cell is accepted when its estimate
//! and the sum of its two halves' estimates agree within their rounding, and is
//! halved otherwise. Every node is a unit phasor whose angle carries the rounding
//! of `t` and `t'`, at most `2π·2ε`, and the weights sum to the cell width, so an
//! estimate over a cell of width `w` is resolved to `(4π + 1)·ε·w` and two
//! estimates can differ by twice that. A jump confines the halving to the one cell
//! that holds it, and halving ends where a cell's midpoint is no longer
//! representable, so every integral ends. The published transport samples are the
//! nodes of the accepted halves.
//!
//! # Drift statistics (gam#2231 §3)
//!
//! Alongside the law, the report carries the honest-units decoder drift
//! `δ_k = ‖B^(ℓ+1) − B^(ℓ)‖_F / √(‖B^(ℓ)‖_F · ‖B^(ℓ+1)‖_F)` and the principal
//! angles between the two layer images (the row spaces of the two honest
//! decoders in `ℝ^p`).

use super::*;
use crate::chart_coordinate_solve::{ChartBasisKind, PeriodicCurveExtrema};
use gam_math::special::gauss_legendre;
use ndarray::ArrayView1;

/// A reference to one column block of a crosscoder target: the implicit anchor
/// layer `[0, p_x)`, or an explicit output block `ℓ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrosscoderLayer {
    /// The anchor block `[0, p_x)` — the first layer, decoded in honest units
    /// directly (no `√λ` scaling).
    Anchor,
    /// Output block `ℓ` (`0`-based over the `L−1` non-anchor blocks); decoded in
    /// honest units by dividing its column slice by `√λ_ℓ`.
    Block(usize),
}

/// The nearest-point correspondence of one circle atom between two layer images,
/// the phase-shift law fit, and the drift statistics.
#[derive(Clone, Debug)]
pub struct AtomTransportReport {
    /// The atom's harmonic order `H = (M − 1)/2`.
    pub n_harmonics: usize,
    /// The best phase-shift model `t' = s·t + φ`: `(s, φ)` with `s ∈ {+1, −1}`
    /// and `φ` in chart units, wrapped to `[−½, ½)`.
    pub phase_shift: (f64, f64),
    /// Circular `R²` of the phase-shift fit (`1 − SS_res/SS_tot`). The LAW's
    /// goodness of fit; `≈ 1` when the correspondence is a pure phase shift.
    pub phase_r2: f64,
    /// Honest-units decoder drift `δ_k = ‖B_tgt − B_src‖_F /
    /// √(‖B_src‖_F · ‖B_tgt‖_F)` (gam#2231 §3). `NaN` if either decoder is
    /// numerically dead (Frobenius norm at or below `max(M, p)·ε` of the larger
    /// layer's norm), so a shrunk-out layer cannot manufacture a divergent drift
    /// ratio.
    pub drift: f64,
    /// Principal angles (radians, ascending) between the two layer images — the
    /// row spaces of the honest decoders in `ℝ^p`. Length `max(rank_src,
    /// rank_tgt)` with `|rank_src − rank_tgt|` trailing `π/2` entries when the
    /// ranks differ; empty only if BOTH images are numerically rank-0.
    pub principal_angles: Vec<f64>,
    /// The correspondence samples `(t, t')` in chart units that the integrals'
    /// accepted cells evaluated, sorted by `t`, for plotting / downstream analysis.
    pub transport_grid: Vec<(f64, f64)>,
}

impl AtomTransportReport {
    /// The chart location `t` where the phase-shift model deviates most from the
    /// nearest-point correspondence (the largest chordal residual) among the
    /// correspondence samples. `None` for an empty grid. This is the "interesting
    /// locus" where the correspondence departs from a phase shift.
    pub fn deviation_locus(&self) -> Option<f64> {
        let (s, phi) = self.phase_shift;
        let two_pi = std::f64::consts::TAU;
        self.transport_grid
            .iter()
            .map(|&(t, tp)| {
                let resid = 1.0 - (two_pi * (tp - s * t - phi)).cos();
                (t, resid)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| t)
    }
}

/// Whether the phase-shift law measured one atom between two layers.
#[derive(Clone, Debug)]
pub enum AtomTransportStatus {
    Measured(AtomTransportReport),
    /// The law does not describe this atom or layer pair: not a 1-D periodic
    /// atom at the fitted homotopy endpoint, or layer images in different
    /// ambient spaces.
    Undefined { reason: String },
}

/// Measure the nearest-point correspondence of one circle atom between two
/// explicit crosscoder layers (source image projected onto the target image).
pub fn measure_atom_transport_between(
    term: &SaeManifoldTerm,
    layout: &CrosscoderLayout,
    atom: usize,
    source: CrosscoderLayer,
    target: CrosscoderLayer,
) -> Result<AtomTransportStatus, String> {
    if atom >= term.atoms.len() {
        return Err(format!(
            "measure_atom_transport_between: atom index {atom} out of range (K = {})",
            term.atoms.len()
        ));
    }
    if layout.total_dim() != term.output_dim() {
        return Err(format!(
            "measure_atom_transport_between: layout total width {} != term output_dim {} (the layout \
             must describe this term's augmented columns)",
            layout.total_dim(),
            term.output_dim()
        ));
    }
    let atom_ref = &term.atoms[atom];
    if atom_ref.latent_dim() != 1 {
        return Ok(AtomTransportStatus::Undefined {
            reason: format!(
                "the phase-shift law is defined for a 1-D circle atom; atom {atom} has latent_dim {}",
                atom_ref.latent_dim()
            ),
        });
    }
    if atom_ref.basis_kind() != &SaeAtomBasisKind::Periodic {
        return Ok(AtomTransportStatus::Undefined {
            reason: format!(
                "the phase-shift law needs the standard periodic harmonic basis; atom {atom} uses {:?}",
                atom_ref.basis_kind()
            ),
        });
    }
    if atom_ref.homotopy_eta != 1.0 {
        return Ok(AtomTransportStatus::Undefined {
            reason: format!(
                "atom {atom} is at homotopy eta {}, not the fitted eta = 1 endpoint",
                atom_ref.homotopy_eta
            ),
        });
    }

    // Honest-units source and target decoders, both `M × p` in the SAME ambient.
    // #2015 — undo any Tier-0 column-equilibration scale first (a no-op on the
    // historical unequilibrated path).
    let physical_decoder = term.tier0_unscaled_full_width_decoder(atom);
    let b_src = honest_layer_decoder(&physical_decoder, layout, source)?;
    let b_tgt = honest_layer_decoder(&physical_decoder, layout, target)?;
    if b_src.ncols() != b_tgt.ncols() {
        return Ok(AtomTransportStatus::Undefined {
            reason: format!(
                "source ambient width {} != target ambient width {}: the nearest-point transport \
                 needs both layer images in one ambient space",
                b_src.ncols(),
                b_tgt.ncols()
            ),
        });
    }

    let m = physical_decoder.nrows();
    let n_harmonics = m.saturating_sub(1) / 2;
    let basis = ChartBasisKind::Periodic { n_harmonics };
    if basis.width() != m {
        return Err(format!(
            "measure_atom_transport_between: periodic basis width {} != physical decoder width {m}",
            basis.width()
        ));
    }
    let target_gram = b_tgt.dot(&b_tgt.t());
    let target_extrema = PeriodicCurveExtrema::from_gram(target_gram.view())?;

    // Project the decoded source point at `t` onto the continuous target image.
    let mut phi = vec![0.0; m];
    let transport_at = |t: f64| -> Result<f64, String> {
        basis.eval_into(t, &mut phi);
        let source_point = b_src.t().dot(&ArrayView1::from(phi.as_slice()));
        let linear = b_tgt.dot(&source_point);
        let linear = linear.as_slice().ok_or_else(|| {
            "measure_atom_transport_between: target linear coefficients are not contiguous"
                .to_string()
        })?;
        target_extrema
            .minimize_squared_distance(linear)
            .map(|projection| projection.coordinate)
            .map_err(|error| {
                format!("measure_atom_transport_between: source coordinate {t} target projection: {error}")
            })
    };
    let (resultants, transport_grid) = integrate_transport_resultants(m, transport_at)?;

    let (phase_shift, phase_r2) = fit_transport_law(&resultants);
    let drift = decoder_drift(&b_src, &b_tgt);
    let principal_angles = principal_angles_between_images(&b_src, &b_tgt)?;

    Ok(AtomTransportStatus::Measured(AtomTransportReport {
        n_harmonics,
        phase_shift,
        phase_r2,
        drift,
        principal_angles,
        transport_grid,
    }))
}

/// The transport map's three resultants over `t ∈ [0, 1)`, as `(re, im)` pairs
/// `[ρ_tot, ρ_+, ρ_−]` with `ρ_tot = ∫ e^{i2πt'}`, `ρ_± = ∫ e^{i2π(t' ∓ t)}`, and
/// the `(t, t')` nodes of the accepted cells sorted by `t`. The module
/// documentation derives the cell rule and its rounding bound.
fn integrate_transport_resultants(
    width: usize,
    mut transport_at: impl FnMut(f64) -> Result<f64, String>,
) -> Result<([f64; 6], Vec<(f64, f64)>), String> {
    let (nodes, weights) = gauss_legendre(width);
    let tau = std::f64::consts::TAU;
    let mut estimate = |a: f64, b: f64| -> Result<([f64; 6], Vec<(f64, f64)>), String> {
        let half = 0.5 * (b - a);
        let middle = a + half;
        let mut sum = [0.0_f64; 6];
        let mut samples = Vec::with_capacity(nodes.len());
        for (&node, &weight) in nodes.iter().zip(weights.iter()) {
            let t = middle + half * node;
            let t_prime = transport_at(t)?;
            let scaled = half * weight;
            for (slot, angle) in [t_prime, t_prime - t, t_prime + t].into_iter().enumerate() {
                sum[2 * slot] += scaled * (tau * angle).cos();
                sum[2 * slot + 1] += scaled * (tau * angle).sin();
            }
            samples.push((t, t_prime));
        }
        Ok((sum, samples))
    };

    let cells = width as f64;
    let mut pending = Vec::with_capacity(width);
    for cell in 0..width {
        let a = cell as f64 / cells;
        let b = (cell + 1) as f64 / cells;
        let (sum, samples) = estimate(a, b)?;
        pending.push((a, b, sum, samples));
    }
    let mut totals = [0.0_f64; 6];
    let mut transport_grid = Vec::new();
    while let Some((a, b, coarse, coarse_samples)) = pending.pop() {
        let middle = a + 0.5 * (b - a);
        if !(middle > a && middle < b) {
            // The cell's midpoint is not representable: this is the resolution
            // limit of the coordinate itself.
            for slot in 0..6 {
                totals[slot] += coarse[slot];
            }
            transport_grid.extend(coarse_samples);
            continue;
        }
        let (left, left_samples) = estimate(a, middle)?;
        let (right, right_samples) = estimate(middle, b)?;
        let rounding = 2.0 * (2.0 * tau + 1.0) * f64::EPSILON * (b - a);
        let settled = (0..6).all(|slot| (left[slot] + right[slot] - coarse[slot]).abs() <= rounding);
        if settled {
            for slot in 0..6 {
                totals[slot] += left[slot] + right[slot];
            }
            transport_grid.extend(left_samples);
            transport_grid.extend(right_samples);
        } else {
            pending.push((a, middle, left, left_samples));
            pending.push((middle, b, right, right_samples));
        }
    }
    transport_grid.sort_by(|left, right| left.0.total_cmp(&right.0));
    Ok((totals, transport_grid))
}

/// The honest-units decoder of one crosscoder layer carved from the atom's
/// augmented decoder: the anchor slice `[0, p_x)` verbatim, or block `ℓ`'s slice
/// divided by `√λ_ℓ` (exactly [`SaeManifoldTerm::layer_decoder`]'s arithmetic,
/// but keyed off the passed `layout` so this needs no installed layout).
pub(crate) fn honest_layer_decoder(
    decoder: &Array2<f64>,
    layout: &CrosscoderLayout,
    layer: CrosscoderLayer,
) -> Result<Array2<f64>, String> {
    match layer {
        CrosscoderLayer::Anchor => Ok(decoder.slice(s![.., 0..layout.anchor_dim()]).to_owned()),
        CrosscoderLayer::Block(l) => {
            if l >= layout.num_blocks() {
                return Err(format!(
                    "measure_atom_transport_between: block index ℓ={l} out of range (L−1 = {})",
                    layout.num_blocks()
                ));
            }
            let inv = 1.0 / layout.sqrt_lambda(l);
            Ok(decoder
                .slice(s![.., layout.block_range(l)])
                .mapv(|v| inv * v))
        }
    }
}

/// Fit the phase-shift law from the transport map's resultants
/// `[ρ_tot, ρ_+, ρ_−]` over the uniform measure on `[0, 1)`. Returns
/// `((s, φ), phase_r2)`.
///
/// `R²` uses the circular-mean baseline `SS_tot = 1 − |ρ_tot|`. The optimal `φ`
/// at fixed `s` is `arg ρ_s / 2π`; `s` is chosen for the larger resultant, and
/// `SS_res = 1 − |ρ_s|`.
fn fit_transport_law(resultants: &[f64; 6]) -> ((f64, f64), f64) {
    let two_pi = std::f64::consts::TAU;
    let ss_tot = 1.0 - resultants[0].hypot(resultants[1]);
    let mut best_s = 1.0_f64;
    let mut best_phi = 0.0_f64;
    let mut best_ss_res = f64::INFINITY;
    for (s, slot) in [(1.0_f64, 2_usize), (-1.0_f64, 4_usize)] {
        let (cos_u, sin_u) = (resultants[slot], resultants[slot + 1]);
        let ss_res = 1.0 - cos_u.hypot(sin_u);
        if ss_res < best_ss_res {
            best_ss_res = ss_res;
            best_s = s;
            // Circular mean of u in turns, wrapped.
            best_phi = wrap_half(sin_u.atan2(cos_u) / two_pi);
        }
    }
    ((best_s, best_phi), circular_r2(ss_tot, best_ss_res))
}

/// `1 − SS_res/SS_tot`, guarding a degenerate (all-equal responses) baseline.
fn circular_r2(ss_tot: f64, ss_res: f64) -> f64 {
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
    }
}

/// Wrap a turn value to `[−½, ½)` (period 1).
fn wrap_half(x: f64) -> f64 {
    let r = x.rem_euclid(1.0);
    if r >= 0.5 { r - 1.0 } else { r }
}

/// Honest-units decoder drift `δ = ‖B_tgt − B_src‖_F / √(‖B_src‖_F · ‖B_tgt‖_F)`
/// (gam#2231 §3). `NaN` if either decoder is numerically dead — Frobenius norm
/// at or below `max(M, p)·ε` of the LARGER layer's norm, the same
/// `σ_max·max(M,p)·ε` numerical-rank convention `orthonormal_row_basis` uses,
/// applied at the joint scale. The exact-zero guard alone let a
/// shrunk-out-but-not-bitwise-zero layer (‖B‖ ~ 1e−30) blow the geometric-mean
/// denominator up to δ ~ 1e13 and hijack `most_drifting_atom`.
pub(crate) fn decoder_drift(b_src: &Array2<f64>, b_tgt: &Array2<f64>) -> f64 {
    let fro = |a: &Array2<f64>| a.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let ns = fro(b_src);
    let nt = fro(b_tgt);
    let dead = ns.max(nt) * (b_src.nrows().max(b_src.ncols()) as f64) * f64::EPSILON;
    if ns > dead && nt > dead {
        let diff: f64 = b_src
            .iter()
            .zip(b_tgt.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        diff / (ns * nt).sqrt()
    } else {
        f64::NAN
    }
}

/// Principal angles (radians, ascending) between the two layer IMAGES — the row
/// spaces of the honest decoders in `ℝ^p`. Each row space's orthonormal basis is
/// the right-singular vectors of the `M × p` decoder above a numerical-rank
/// threshold; the singular values of `Q_srcᵀ Q_tgt` are the cosines of the
/// angles. Unequal ranks append one `π/2` angle for every unmatched image
/// direction, so nested spans surface the rank change instead of reporting a
/// zero-distance match. A rank-zero image likewise contributes `π/2` for every
/// live direction in the other image.
pub(crate) fn principal_angles_between_images(
    b_src: &Array2<f64>,
    b_tgt: &Array2<f64>,
) -> Result<Vec<f64>, String> {
    let q_src = orthonormal_row_basis(b_src)?; // r_src × p
    let q_tgt = orthonormal_row_basis(b_tgt)?; // r_tgt × p
    let r_src = q_src.nrows();
    let r_tgt = q_tgt.nrows();
    if r_src == 0 || r_tgt == 0 {
        return Ok(vec![std::f64::consts::FRAC_PI_2; r_src.max(r_tgt)]);
    }
    let cross = q_src.dot(&q_tgt.t()); // r_src × r_tgt
    let (_u, svals, _vt) = cross
        .svd(false, false)
        .map_err(|e| format!("principal_angles_between_images: SVD failed: {e}"))?;
    let mut angles = svals
        .iter()
        .map(|&sv| sv.clamp(0.0, 1.0).acos())
        .collect::<Vec<f64>>();
    angles.extend(std::iter::repeat(std::f64::consts::FRAC_PI_2).take(r_src.abs_diff(r_tgt)));
    Ok(angles)
}

/// Orthonormal basis (rows) of the row space of an `M × p` decoder, as an
/// `r × p` matrix, `r` its numerical rank. The right-singular vectors of `B`
/// above the standard `σ_max · max(M,p) · ε` threshold span the row space.
fn orthonormal_row_basis(b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let (_u, svals, vt) = b
        .svd(false, true)
        .map_err(|e| format!("orthonormal_row_basis: SVD failed: {e}"))?;
    let vt = vt.ok_or_else(|| "orthonormal_row_basis: SVD returned no right factor".to_string())?;
    let smax = svals.iter().cloned().fold(0.0_f64, f64::max);
    let tol = smax * (b.nrows().max(b.ncols()) as f64) * f64::EPSILON;
    let rank = svals.iter().filter(|&&s| s > tol).count();
    Ok(vt.slice(s![0..rank, ..]).to_owned())
}
