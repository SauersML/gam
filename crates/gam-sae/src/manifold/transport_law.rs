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
//! decoded through per-layer honest decoders `B^(ℓ)` and `B^(ℓ+1)` (the landed M1
//! [`SaeManifoldTerm::run_multiblock_reml_fit`] and its
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
//! period-correct with no unwrapping). We report two nested circular
//! coefficients of determination `R² = 1 − SS_res/SS_tot`, both against the same
//! circular-mean baseline `SS_tot = Σ_g c(t'_g, t̄')`:
//!
//! * **phase-shift model** `t' = s·t + φ`, `s ∈ {+1, −1}` — the LAW. The optimal
//!   `φ` at fixed `s` is the circular mean of `u_g = t'_g − s·t_g`, which
//!   maximizes `Σ_g cos(2π(u_g − φ)) = |Σ_g e^{i 2π u_g}|`; `s` is chosen for the
//!   larger resultant. `SS_res = G − |Σ_g e^{i 2π u_g}|`.
//! * **smooth-map model** `t' = s·t + f(t)` — the alternative hypothesis, with
//!   `f` a Fourier series in `t` to the atom's OWN harmonic order (constant plus
//!   `H = (M−1)/2` harmonics). The constant term subsumes the phase model, so the
//!   extra harmonics are exactly the nonlinear content the phase law forbids.
//!
//! **Verdict.** `phase_r2 ≈ smooth_r2` (small [`AtomTransportReport::law_gap`])
//! ⇒ transport IS a phase shift, the law holds, the extra harmonics buy nothing.
//! A significant gap ⇒ nonlinear transport;
//! [`AtomTransportReport::deviation_locus`] reports the chart location where the
//! phase model deviates most (the interesting locus).
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

/// Outcome of [`measure_atom_transport`]: the empirical layer-to-layer transport
/// map of one circle atom, the phase-shift law test, and the drift statistics.
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
    /// The atom's harmonic order `H = (M − 1)/2`. The smooth-map ALTERNATIVE
    /// fits at order `2H + 1` (the empirical transport between order-H curves
    /// generically carries harmonics above `H`; see the call-site note).
    pub n_harmonics: usize,
    /// The best phase-shift model `t' = s·t + φ`: `(s, φ)` with `s ∈ {+1, −1}`
    /// and `φ` in chart units, wrapped to `[−½, ½)`.
    pub phase_shift: (f64, f64),
    /// Circular `R²` of the phase-shift fit (`1 − SS_res/SS_tot`). The LAW's
    /// goodness of fit; `≈ 1` when transport is a pure phase shift.
    pub phase_r2: f64,
    /// Circular `R²` of the smooth-map alternative (phase shift plus `H`
    /// harmonics of φ̂-centered drift). Nests the law by construction (its
    /// constant term rides on the phase model's circular mean), so it sits
    /// `≥ phase_r2` up to the least-squares/chordal metric mismatch on the
    /// residual harmonics; a (small) negative `law_gap` is honest numerics,
    /// not a clamp. Note the alternative's Fourier order is capped at the
    /// atom's own `H`: the empirical map between two order-`H` curves can
    /// carry harmonics above `H`, so this alternative is CONSERVATIVE — it can
    /// under-detect nonlinearity, never over-detect it.
    pub smooth_r2: f64,
    /// Honest-units decoder drift `δ_k = ‖B_tgt − B_src‖_F /
    /// √(‖B_src‖_F · ‖B_tgt‖_F)` (gam#2231 §3). `NaN` if either decoder is
    /// numerically dead (Frobenius norm ≤ 1e−12 of the larger layer's norm),
    /// so a shrunk-out layer cannot manufacture a divergent drift ratio.
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
    /// `smooth_r2 − phase_r2`: how much circular fit the nonlinear harmonics buy
    /// over the phase-shift law. Small ⇒ the law holds; large ⇒ nonlinear
    /// transport.
    pub fn law_gap(&self) -> f64 {
        self.smooth_r2 - self.phase_r2
    }

    /// The law verdict at a caller-chosen gap tolerance: `true` when the extra
    /// harmonics buy less than `gap_tol` of circular `R²` (transport is a phase
    /// shift) and the phase fit is finite.
    pub fn law_holds(&self, gap_tol: f64) -> bool {
        self.phase_r2.is_finite() && self.smooth_r2.is_finite() && self.law_gap() <= gap_tol
    }

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

/// Measure the empirical anchor→first-block transport of one circle atom and
/// test the phase-shift law. See the module header for the full definition.
///
/// `term` must hold the fitted atom (with an installed periodic basis evaluator);
/// `layout` supplies the anchor width, the block column ranges, and the per-block
/// `√λ_ℓ` unscaling (pass the layout the fit installed, or one built with
/// [`CrosscoderLayout::from_blocks`]). `grid_resolution` is the number of source
/// samples reported over `[0, 1)`; it controls diagnostic sampling, not the
/// continuous target-coordinate solve. Requires `layout.num_blocks() ≥ 1`.
pub fn measure_atom_transport(
    term: &SaeManifoldTerm,
    layout: &CrosscoderLayout,
    atom: usize,
    grid_resolution: usize,
) -> Result<AtomTransportReport, String> {
    measure_atom_transport_between(
        term,
        layout,
        atom,
        CrosscoderLayer::Anchor,
        CrosscoderLayer::Block(0),
        grid_resolution,
    )
}

/// Measure the empirical transport of one circle atom between two explicit
/// crosscoder layers (source image projected onto the target image). The
/// two-layer entry point [`measure_atom_transport`] is the anchor→`Block(0)` case.
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
            "measure_atom_transport: atom index {atom} out of range (K = {})",
            term.atoms.len()
        ));
    }
    if layout.total_dim() != term.output_dim() {
        return Err(format!(
            "measure_atom_transport: layout total width {} != term output_dim {} (the layout \
             must describe this term's augmented columns)",
            layout.total_dim(),
            term.output_dim()
        ));
    }
    let atom_ref = &term.atoms[atom];
    if atom_ref.latent_dim() != 1 {
        return Err(format!(
            "measure_atom_transport: the phase-shift law is defined for a 1-D circle atom; atom \
             {atom} has latent_dim {}",
            atom_ref.latent_dim()
        ));
    }
    if atom_ref.basis_kind() != &SaeAtomBasisKind::Periodic {
        return Err(format!(
            "measure_atom_transport: atom {atom} must use the standard periodic harmonic basis, got {:?}",
            atom_ref.basis_kind()
        ));
    }
    if atom_ref.homotopy_eta != 1.0 {
        return Err(format!(
            "measure_atom_transport: atom {atom} is at homotopy eta {}, not the fitted eta = 1 endpoint",
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
            "measure_atom_transport: source ambient width {} != target ambient width {} — the \
             nearest-point transport needs both layer images in one ambient space (a crosscoder \
             shares the residual-stream dimension across layers)",
            b_src.ncols(),
            b_tgt.ncols()
        ));
    }

    let m = physical_decoder.nrows();
    let n_harmonics = m.saturating_sub(1) / 2;
    if grid_resolution == 0 {
        return Err("measure_atom_transport: grid_resolution must be positive".to_string());
    }

    // Reporting density and diagnostic-fit density are independent.  The
    // caller may deliberately request a very coarse report, while the smooth
    // alternative still needs more source evaluations than coefficients.  Use
    // the smallest multiple of the requested density that identifies the fit;
    // this keeps every requested report point exactly on the diagnostic grid.
    let alt_harmonics = (2 * n_harmonics + 1).max(grid_resolution / 16);
    let smooth_coefficient_count = 2 * alt_harmonics + 1;
    let required_fit_samples = smooth_coefficient_count + 1;
    let diagnostic_multiplier = required_fit_samples.div_ceil(grid_resolution).max(1);
    let diagnostic_resolution = grid_resolution
        .checked_mul(diagnostic_multiplier)
        .ok_or_else(|| "measure_atom_transport: diagnostic grid size overflow".to_string())?;

    // Evaluate the standard full-width harmonic basis on the diagnostic SOURCE
    // grid.  These samples do not serve as target candidates.
    let basis = ChartBasisKind::Periodic { n_harmonics };
    let grid = Array2::<f64>::from_shape_fn((diagnostic_resolution, 1), |(g, _)| {
        g as f64 / diagnostic_resolution as f64
    });
    if basis.width() != m {
        return Err(format!(
            "measure_atom_transport: periodic basis width {} != physical decoder width {m}",
            basis.width()
        ));
    }
    let mut phi_grid = Array2::<f64>::zeros((diagnostic_resolution, m));
    let mut phi = vec![0.0; m];
    for g in 0..diagnostic_resolution {
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
    let tprime: Vec<f64> = (0..diagnostic_resolution)
        .into_par_iter()
        .map(|g| {
            let linear = linear_all.row(g);
            let projection = target_extrema
                .minimize_squared_distance(linear.as_slice().ok_or_else(|| {
                    "measure_atom_transport: target linear coefficients are not contiguous"
                        .to_string()
                })?)
                .map_err(|error| {
                    format!("measure_atom_transport: source sample {g} target projection: {error}")
                })?;
            Ok(projection.coordinate)
        })
        .collect::<Result<Vec<f64>, String>>()?;
    let t_arr: Vec<f64> = (0..diagnostic_resolution)
        .map(|g| g as f64 / diagnostic_resolution as f64)
        .collect();
    let transport_grid: Vec<(f64, f64)> = (0..grid_resolution)
        .map(|g| {
            let diagnostic_index = g * diagnostic_multiplier;
            (t_arr[diagnostic_index], tprime[diagnostic_index])
        })
        .collect();

    // The smooth ALTERNATIVE's Fourier order is deliberately HIGHER than the
    // atom's own H: the empirical transport t ↦ t'(t) between two order-H
    // curves is a projection argmin and generically carries harmonics ABOVE H
    // (composition/inversion of trigonometric polynomials is not order-H), so
    // capping the alternative at H under-detects nonlinearity — measured
    // 2026-07-10 on the planted θ' = θ + a·sinθ arm: the recovered drift needs
    // ~2H harmonics for R² 0.86 vs 0.72 at H. `2H + 1` keeps the detector a
    // low-order smooth model (K = 4H + 3 coefficients ≪ grid_resolution, which
    // the guard above already enforces at the STRICTER alternative order) while
    // removing the conservative bias the 2026-07-10 audit flagged.
    // Grid-proportional detector order: the drift spectrum of a composed /
    // inverted trigonometric map decays slowly (measured on the planted
    // θ+0.8·sinθ arm: R² 0.86 at H=11, 0.90 at H=16, 0.95 at H=31), so the
    // alternative uses the larger of the curve-derived 2H+1 and grid/16 —
    // still ≥8× oversampled (K = 2·alt+1 coefficients vs grid points).
    let (phase_shift, phase_r2, smooth_r2) = fit_transport_law(&t_arr, &tprime, alt_harmonics);
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
        smooth_r2,
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
                    "measure_atom_transport: block index ℓ={l} out of range (L−1 = {})",
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

/// Fit the phase-shift law and the smooth-map alternative to a period-1 circular
/// transport `t ↦ t'`. Returns `((s, φ), phase_r2, smooth_r2)`.
///
/// Both `R²` share the circular-mean baseline `SS_tot = G − |Σ e^{i2π t'_g}|`.
/// The phase model's optimal `φ` at fixed `s` is `circmean(t'_g − s·t_g)`; `s` is
/// chosen for the larger resultant. The smooth model regresses the wrapped drift
/// `t'_g − s·t_g` on a constant plus `n_harmonics` Fourier harmonics of `t_g`.
fn fit_transport_law(t: &[f64], tprime: &[f64], n_harmonics: usize) -> ((f64, f64), f64, f64) {
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

    // Smooth-map alternative: t' = s·t + φ̂ + f(t), f a Fourier series (constant
    // + n_harmonics harmonics) of t fit by least squares to the φ̂-CENTERED
    // wrapped drift d_g = wrap(t'_g − s·t_g − φ̂). Centering by the phase
    // model's circular mean is load-bearing: when the true offset sits near the
    // wrap boundary ±½, the UNcentered wrapped drift is a ±½ square-wave-like
    // discontinuous signal whose Fourier LS fit is garbage — the smooth R²
    // then craters below the phase R² and a genuinely nonlinear transport
    // would read as "law holds". Centered, the drift is small and continuous,
    // and the zero-function f reproduces the phase model exactly, so the
    // alternative nests the law by construction (no clamp needed — a small
    // negative gap is honest LS/chordal metric mismatch, and a smooth model
    // that fails to beat the phase model IS the law holding).
    let smooth_r2 = fit_smooth_alternative(t, tprime, best_s, best_phi, n_harmonics, ss_tot)
        .unwrap_or(phase_r2);

    ((best_s, best_phi), phase_r2, smooth_r2)
}

/// Least-squares Fourier fit of the φ-centered wrapped transport drift; returns
/// the smooth model's circular `R²`, or `None` if the normal equations are
/// singular (then the caller falls back to the phase `R²`).
fn fit_smooth_alternative(
    t: &[f64],
    tprime: &[f64],
    s: f64,
    phi: f64,
    n_harmonics: usize,
    ss_tot: f64,
) -> Option<f64> {
    let two_pi = std::f64::consts::TAU;
    let k = 2 * n_harmonics + 1;

    // Design matrix D (G × K): [1, sin2πt, cos2πt, …, sin2πHt, cos2πHt].
    let design = |ti: f64| -> Vec<f64> {
        let mut row = Vec::with_capacity(k);
        row.push(1.0);
        for h in 1..=n_harmonics {
            let a = two_pi * h as f64 * ti;
            row.push(a.sin());
            row.push(a.cos());
        }
        row
    };

    // Normal equations DᵀD c = Dᵀ d with d_g the wrapped drift.
    let mut dtd = Array2::<f64>::zeros((k, k));
    let mut dtb = Array1::<f64>::zeros(k);
    for (&ti, &tpi) in t.iter().zip(tprime.iter()) {
        let row = design(ti);
        // φ-centered drift: small and continuous when the law nearly holds,
        // even for offsets at the wrap boundary ±½.
        let d = wrap_half(tpi - s * ti - phi);
        for i in 0..k {
            dtb[i] += row[i] * d;
            for j in 0..k {
                dtd[[i, j]] += row[i] * row[j];
            }
        }
    }
    let coeffs = solve_spd(&dtd, &dtb)?;

    // Residual SS under the fitted smooth map, chordal on the circle.
    let mut ss_res = 0.0_f64;
    for (&ti, &tpi) in t.iter().zip(tprime.iter()) {
        let row = design(ti);
        let f: f64 = row.iter().zip(coeffs.iter()).map(|(&r, &c)| r * c).sum();
        let pred = s * ti + phi + f;
        ss_res += 1.0 - (two_pi * (tpi - pred)).cos();
    }
    Some(circular_r2(ss_tot, ss_res))
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
/// ≤ 1e−12 of the LARGER layer's norm (a relative gate, matching the rank
/// threshold used for principal angles). The exact-zero guard alone let a
/// shrunk-out-but-not-bitwise-zero layer (‖B‖ ~ 1e−30) blow the geometric-mean
/// denominator up to δ ~ 1e13 and hijack `most_drifting_atom`.
pub(crate) fn decoder_drift(b_src: &Array2<f64>, b_tgt: &Array2<f64>) -> f64 {
    let fro = |a: &Array2<f64>| a.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let ns = fro(b_src);
    let nt = fro(b_tgt);
    let dead = 1e-12 * ns.max(nt);
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

/// Solve the small SPD system `A c = b` (`A` `K × K`) by dense Cholesky. Returns
/// `None` when `A` is not numerically positive definite (a singular design).
fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let k = a.nrows();
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for p in 0..j {
                sum -= l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if !(sum > 0.0) {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Forward solve L y = b.
    let mut y = Array1::<f64>::zeros(k);
    for i in 0..k {
        let mut sum = b[i];
        for p in 0..i {
            sum -= l[[i, p]] * y[p];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ c = y.
    let mut c = Array1::<f64>::zeros(k);
    for i in (0..k).rev() {
        let mut sum = y[i];
        for p in (i + 1)..k {
            sum -= l[[p, i]] * c[p];
        }
        c[i] = sum / l[[i, i]];
    }
    Some(c)
}
