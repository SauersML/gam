//! Transport-law measurement: is layer-to-layer transport of a circle atom a
//! phase shift?
//!
//! # The thesis this measures
//!
//! "**Binding is transport.** Layers act through a transport groupoid; LINEAR
//! transport of an elliptical (circle) atom is forced to be a phase shift
//! `t ↦ ±t + φ`; the residual gauge obstruction is the atom's linear stabilizer."
//! (module header of [`crate::manifold`]). This module turns that claim into a
//! *measurement* on a fitted 2-layer crosscoder — a shared chart coordinate `t`
//! decoded through per-layer honest decoders `B^(ℓ)` and `B^(ℓ+1)` (the
//! [`CrosscoderLayout`]/[`SaeManifoldTerm::layer_decoder`] bookkeeping).
//!
//! # Operational definition of the transport map
//!
//! Both layers of a crosscoder share the SAME ambient residual-stream dimension,
//! so the atom image at layer `ℓ` (`C^(ℓ) = {Φ_k(t) B^(ℓ)_k}`) and at layer `ℓ+1`
//! (`C^(ℓ+1) = {Φ_k(t) B^(ℓ+1)_k}`) are two curves in one `ℝ^p`. The network's
//! transport carries a layer-`ℓ` feature to layer `ℓ+1`; with no network in hand
//! we approximate that correspondence by NEAREST POINT: for each reported source
//! sample `t_g`, decode the SOURCE (layer `ℓ`) image
//! `x_g = Φ_k(t_g) B^(ℓ)_k`, then PROJECT `x_g` onto the CONTINUOUS TARGET
//! (layer `ℓ+1`) atom image to read off the chart coordinate that best reproduces it,
//! `t'_g = argmin_{t'} ‖x_g − Φ_k(t') B^(ℓ+1)_k‖²`. The empirical transport map is
//! `t_g ↦ t'_g`.  The target projection enumerates every stationary point of
//! this trigonometric polynomial through its companion-matrix roots, so `t'_g`
//! is not quantized by the source-report sampling density.
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
//! baseline `SS_tot = Σ_g c(t'_g, t̄')`. The optimal `φ` at fixed `s` is the
//! circular mean of `u_g = t'_g − s·t_g`, which maximizes
//! `Σ_g cos(2π(u_g − φ)) = |Σ_g e^{i 2π u_g}|`; `s` is chosen for the larger
//! resultant, and `SS_res = G − |Σ_g e^{i 2π u_g}|`.
//!
//! `phase_r2 = 1` exactly when transport is a phase shift at every reported
//! sample (`NaN` when every transported coordinate coincides, a degenerate
//! baseline); its shortfall is the chordal residual the law leaves unexplained,
//! and [`AtomTransportReport::deviation_locus`] reports the chart location where
//! the phase model deviates most (the interesting locus). No smooth-map
//! alternative is fitted and no verdict threshold is applied: the alternative's
//! Fourier order and a gap tolerance would both be tuning constants that no
//! measured transport identifies.
//!
//! # Drift statistics (gam#2231 §3)
//!
//! Alongside the law, the report carries the honest-units decoder drift
//! `δ_k = ‖B^(ℓ+1) − B^(ℓ)‖_F / √(‖B^(ℓ)‖_F · ‖B^(ℓ+1)‖_F)` and the principal
//! angles between the two layer images (the row spaces of the two honest
//! decoders in `ℝ^p`).

use super::*;
use crate::chart_coordinate_solve::{ChartBasisKind, PeriodicCurveExtrema};

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

/// The empirical transport map of one circle atom, the phase-shift law fit, and
/// the drift statistics.
#[derive(Clone, Debug)]
pub struct AtomTransportReport {
    /// The atom index this report is for.
    pub atom: usize,
    /// The source and target layers the transport was measured between.
    pub source: CrosscoderLayer,
    /// The target layer (its image is the projection target).
    pub target: CrosscoderLayer,
    /// Number of source chart samples reported over `[0, 1)`.  Target
    /// coordinates are solved continuously and do not inherit this resolution.
    pub grid_resolution: usize,
    /// The atom's harmonic order `H = (M − 1)/2`.
    pub n_harmonics: usize,
    /// The best phase-shift model `t' = s·t + φ`: `(s, φ)` with `s ∈ {+1, −1}`
    /// and `φ` in chart units, wrapped to `[−½, ½)`.
    pub phase_shift: (f64, f64),
    /// Circular `R²` of the phase-shift fit (`1 − SS_res/SS_tot`). The LAW's
    /// goodness of fit; `≈ 1` when transport is a pure phase shift.
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
    /// The empirical transport samples `(t_g, t'_g)` in chart units, one per grid
    /// point, for plotting / downstream analysis.
    pub transport_grid: Vec<(f64, f64)>,
}

impl AtomTransportReport {
    /// The chart location `t_g` where the phase-shift model deviates most from
    /// the empirical transport (the largest chordal residual). `None` for an
    /// empty grid. This is the "interesting locus" where linear transport breaks.
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

/// Measure the empirical transport of one circle atom between two explicit
/// crosscoder layers (source image projected onto the target image).
pub fn measure_atom_transport_between(
    term: &SaeManifoldTerm,
    layout: &CrosscoderLayout,
    atom: usize,
    source: CrosscoderLayer,
    target: CrosscoderLayer,
    grid_resolution: usize,
) -> Result<AtomTransportReport, String> {
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
        return Err(format!(
            "measure_atom_transport_between: the phase-shift law is defined for a 1-D circle atom; atom \
             {atom} has latent_dim {}",
            atom_ref.latent_dim()
        ));
    }
    if atom_ref.basis_kind() != &SaeAtomBasisKind::Periodic {
        return Err(format!(
            "measure_atom_transport_between: atom {atom} must use the standard periodic harmonic basis, got {:?}",
            atom_ref.basis_kind()
        ));
    }
    if atom_ref.homotopy_eta != 1.0 {
        return Err(format!(
            "measure_atom_transport_between: atom {atom} is at homotopy eta {}, not the fitted eta = 1 endpoint",
            atom_ref.homotopy_eta
        ));
    }

    // Honest-units source and target decoders, both `M × p` in the SAME ambient.
    // #2015 — undo any Tier-0 column-equilibration scale first (a no-op on the
    // historical unequilibrated path).
    let physical_decoder = term.tier0_unscaled_full_width_decoder(atom);
    let b_src = honest_layer_decoder(&physical_decoder, layout, source)?;
    let b_tgt = honest_layer_decoder(&physical_decoder, layout, target)?;
    if b_src.ncols() != b_tgt.ncols() {
        return Err(format!(
            "measure_atom_transport_between: source ambient width {} != target ambient width {} — the \
             nearest-point transport needs both layer images in one ambient space (a crosscoder \
             shares the residual-stream dimension across layers)",
            b_src.ncols(),
            b_tgt.ncols()
        ));
    }

    let m = physical_decoder.nrows();
    let n_harmonics = m.saturating_sub(1) / 2;
    if grid_resolution == 0 {
        return Err("measure_atom_transport_between: grid_resolution must be positive".to_string());
    }

    // Evaluate the standard full-width harmonic basis on the SOURCE grid.
    // These samples do not serve as target candidates.
    let basis = ChartBasisKind::Periodic { n_harmonics };
    let grid = Array2::<f64>::from_shape_fn((grid_resolution, 1), |(g, _)| {
        g as f64 / grid_resolution as f64
    });
    if basis.width() != m {
        return Err(format!(
            "measure_atom_transport_between: periodic basis width {} != physical decoder width {m}",
            basis.width()
        ));
    }
    let mut phi_grid = Array2::<f64>::zeros((grid_resolution, m));
    let mut phi = vec![0.0; m];
    for g in 0..grid_resolution {
        basis.eval_into(grid[[g, 0]], &mut phi);
        for column in 0..m {
            phi_grid[[g, column]] = phi[column];
        }
    }
    let source_image = phi_grid.dot(&b_src); // G × p, decoded source points
    let target_gram = b_tgt.dot(&b_tgt.t());
    let target_extrema = PeriodicCurveExtrema::from_gram(target_gram.view())?;

    // Empirical transport: project each source point onto the continuous target
    // image by comparing every companion-enumerated stationary point. The
    // per-point linear coefficients `B_tgt·x_g` are ONE `G×p · p×M` GEMM (the
    // former per-point gemv was the loop's memory-bound half), and the
    // companion-eigenvalue projections are embarrassingly parallel.
    let linear_all = source_image.dot(&b_tgt.t()); // G × M
    use rayon::prelude::*;
    let tprime: Vec<f64> = (0..grid_resolution)
        .into_par_iter()
        .map(|g| {
            let linear = linear_all.row(g);
            let projection = target_extrema
                .minimize_squared_distance(linear.as_slice().ok_or_else(|| {
                    "measure_atom_transport_between: target linear coefficients are not contiguous"
                        .to_string()
                })?)
                .map_err(|error| {
                    format!("measure_atom_transport_between: source sample {g} target projection: {error}")
                })?;
            Ok(projection.coordinate)
        })
        .collect::<Result<Vec<f64>, String>>()?;
    let t_arr: Vec<f64> = (0..grid_resolution)
        .map(|g| g as f64 / grid_resolution as f64)
        .collect();
    let transport_grid: Vec<(f64, f64)> =
        t_arr.iter().copied().zip(tprime.iter().copied()).collect();

    let (phase_shift, phase_r2) = fit_transport_law(&t_arr, &tprime);
    let drift = decoder_drift(&b_src, &b_tgt);
    let principal_angles = principal_angles_between_images(&b_src, &b_tgt)?;

    Ok(AtomTransportReport {
        atom,
        source,
        target,
        grid_resolution,
        n_harmonics,
        phase_shift,
        phase_r2,
        drift,
        principal_angles,
        transport_grid,
    })
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

/// Fit the phase-shift law to a period-1 circular transport `t ↦ t'`. Returns
/// `((s, φ), phase_r2)`.
///
/// `R²` uses the circular-mean baseline `SS_tot = G − |Σ e^{i2π t'_g}|`. The
/// optimal `φ` at fixed `s` is `circmean(t'_g − s·t_g)`; `s` is chosen for the
/// larger resultant.
fn fit_transport_law(t: &[f64], tprime: &[f64]) -> ((f64, f64), f64) {
    let two_pi = std::f64::consts::TAU;
    let g = t.len();
    let gf = g as f64;

    // Circular-mean baseline SS_tot over the responses t'_g.
    let (sum_sin, sum_cos) = tprime.iter().fold((0.0, 0.0), |(s, c), &v| {
        (s + (two_pi * v).sin(), c + (two_pi * v).cos())
    });
    let r_tot = (sum_sin * sum_sin + sum_cos * sum_cos).sqrt();
    let ss_tot = gf - r_tot;

    // Phase model: for each s ∈ {+1,-1}, best φ is circmean(u), residual SS is
    // G − |Σ e^{i2π u}|. Pick the s with the smaller residual (larger resultant).
    let mut best_s = 1.0_f64;
    let mut best_phi = 0.0_f64;
    let mut best_ss_res = f64::INFINITY;
    for &s in &[1.0_f64, -1.0_f64] {
        let (su, cu) = t
            .iter()
            .zip(tprime.iter())
            .fold((0.0, 0.0), |(a, b), (&ti, &tpi)| {
                let u = tpi - s * ti;
                (a + (two_pi * u).sin(), b + (two_pi * u).cos())
            });
        let r_u = (su * su + cu * cu).sqrt();
        let ss_res = gf - r_u;
        if ss_res < best_ss_res {
            best_ss_res = ss_res;
            best_s = s;
            // Circular mean of u: mean angle atan2(Σsin, Σcos) in turns, wrapped.
            best_phi = wrap_half(su.atan2(cu) / two_pi);
        }
    }
    let phase_r2 = circular_r2(ss_tot, best_ss_res);
    ((best_s, best_phi), phase_r2)
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
