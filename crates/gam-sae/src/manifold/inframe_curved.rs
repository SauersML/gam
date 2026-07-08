//! In-frame curved fit: low-rank ambient frames as the PRIMARY β parameterization
//! for the curved manifold-SAE stage (reviewer priority #2).
//!
//! # Why this exists
//!
//! The curved engine's arrow-Schur border carries `Σ_k M_k · p` coefficients and
//! each atom's posterior covariance is `(M_k · p)²`. At K=32k, M=8, p=4096 that is
//! a ~10⁹-coefficient border and 8.6 GB *per atom* of covariance — impossible at
//! LLM width. The `p` factor is the whole problem: a curved atom's decoded image
//! lives in a handful of ambient directions, so the curved fit never needs the
//! full width.
//!
//! # The three-stage cascade (see `docs/inframe-curved-frames.md`)
//!
//! 1. **Full-width linear / block lane** on the whole corpus owns the full-`p`
//!    projection; its residual `R = X − X̂_linear` feeds this module.
//! 2. **Frames own full-`p`.** Per region we learn a Grassmann frame `U_g`
//!    (`p × r`, `r ≈ 8–32`, `St(r, p)`) from the region's residual span — a
//!    thin-SVD seed that is a closed-form MAP point, refreshable by one polar step
//!    (`crate::frames::GrassmannFrame`). Charged ONCE, globally, against `N`.
//! 3. **Curved REML, purely in-frame.** The curved chart is fit in the `r`-dim
//!    in-frame coordinates `Z_g = R_g U_g`; the border is `Σ M_k·r`, the posterior
//!    covariance is `(M·r)²`, and both are projected back to ambient ON DEMAND.
//!    Each region pays a held-out `½·d_eff·log n_eff` deviance charge and the
//!    accepted set is chosen by e-BH.
//!
//! The single evidence currency matches the joint REML gate
//! (`crate::manifold::rank_charge_dof`); the frame's `r(p−r)` Grassmann degrees of
//! freedom are charged once against the whole corpus (amortized), never per block
//! — charging them per region would reject everything at `p = 4096`.
//!
//! Every accept/reject is an evidence comparison; the frame rank `r` is the
//! numerical rank of the residual span at [`crate::frames::SAE_FRAME_RANK_CUTOFF`]
//! clamped to a configured band (no magic constant). Nothing here touches the
//! CERT-owned atom/gauge algebra or the PRECOND-owned arrow-Schur solver: it
//! builds only on the public `GrassmannFrame` frame primitive.

use ndarray::{Array1, Array2, ArrayView2};

use crate::frames::{
    GrassmannCrossMoment, GrassmannFrame, SAE_FRAME_ACTIVATION_MARGIN,
    SAE_FRAME_MIN_AUTO_OUTPUT_DIM, SAE_FRAME_RANK_CUTOFF,
};
use super::weight_frame_catalog::{WeightFrameCatalog, WeightFrameSource};
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd, fast_ab, fast_abt};
use faer::Side;

/// Bytes per `f64`, matching the streaming-plan ledger.
const BYTES_PER_F64: usize = 8;

/// Configuration for the in-frame curved cascade. Defaults are principled, not
/// magic: the rank band brackets the reviewer's `r ≈ 8–32`, the spectral cutoff
/// is the crate-wide frame cutoff, and every accept is an evidence decision.
#[derive(Clone, Debug)]
pub struct InFrameCurvedConfig {
    /// Lower clamp on the learned frame rank `r`.
    pub frame_rank_min: usize,
    /// Upper clamp on the learned frame rank `r`.
    pub frame_rank_max: usize,
    /// Relative spectral cutoff selecting `r` from the residual singular values.
    pub rank_cutoff: f64,
    /// Cross-fit folds for the held-out deviance estimate (`>= 2`).
    pub crossfit_folds: usize,
    /// e-BH false-discovery level over regions, in `(0, 1]`.
    pub alpha: f64,
    /// Minimum accepted held-out margin (deviance gain minus charge).
    pub min_effect: f64,
    /// Tikhonov ridge on the in-frame whitening covariance.
    pub whitening_ridge: f64,
    /// Minimum region firings before a curved fit is attempted.
    pub min_rows: usize,
    /// Slow-timescale Grassmann refresh: one polar step from the fitted
    /// cross-moment, KEPT only if the held-out in-frame EV improves.
    pub frame_refresh: bool,
}

impl Default for InFrameCurvedConfig {
    fn default() -> Self {
        Self {
            frame_rank_min: 2,
            frame_rank_max: 32,
            rank_cutoff: SAE_FRAME_RANK_CUTOFF,
            crossfit_folds: 4,
            alpha: 0.10,
            min_effect: 0.0,
            whitening_ridge: 1.0e-8,
            min_rows: 32,
            frame_refresh: false,
        }
    }
}

/// One curved region handed to the cascade: the corpus rows that route into an
/// atom (or block of atoms) and the atom's basis size `M` (border width per
/// ambient direction). The residual matrix rows are addressed by `rows`.
#[derive(Clone, Debug)]
pub struct CurvedRegion {
    /// Row indices into the Stage-1 residual matrix.
    pub rows: Vec<usize>,
    /// Atom basis size `M_k` — the border carries `M_k · r` (in-frame) or
    /// `M_k · p` (dense) coefficients for this region.
    pub basis_size: usize,
}

/// Held-out evidence for a single region's curved refinement, in the same
/// currency the joint REML gate uses (deviance gain vs `½·d_eff·log n`).
#[derive(Clone, Debug)]
pub struct RegionEvidence {
    pub n_rows: usize,
    pub n_effective: f64,
    pub frame_rank: usize,
    pub linear_loss: f64,
    pub chart_loss: f64,
    pub deviance_gain: f64,
    pub mean_delta: f64,
    pub se: f64,
    pub ci_low: f64,
    pub charge: f64,
    pub margin: f64,
    pub log_e_value: f64,
    pub accepted_pre_ebh: bool,
    pub accepted: bool,
}

/// Whether a chartable frame was occupied by this corpus slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChartOccupancyStatus {
    /// Data rows occupied the frame and a chart fit was scored.
    Occupied,
    /// The weight-sourced frame exists, but no corpus rows landed in it here.
    ChartableUnoccupied,
    /// The frame has rows, but not enough to score the curved chart honestly.
    InsufficientOccupancy,
}

/// Per-region record: its learned frame, evidence, and in-frame footprint.
#[derive(Clone, Debug)]
pub struct RegionRecord {
    pub region: usize,
    /// Source component for weight-sourced frames; `None` for data-sourced
    /// residual-span frames.
    pub frame_source: Option<WeightFrameSource>,
    /// Index into the source frame catalog when this record came from weights.
    pub frame_catalog_index: Option<usize>,
    pub occupancy_status: ChartOccupancyStatus,
    pub basis_size: usize,
    pub frame_rank: usize,
    /// `r(p − r)` Grassmann DOF this region's frame profiled out — charged once
    /// against the whole corpus, not against this region.
    pub frame_manifold_dim: usize,
    pub evidence: RegionEvidence,
    /// In-frame border coefficients for this region's atom: `M · r`.
    pub inframe_border_coeffs: usize,
    /// Dense border coefficients that the full-`p` path would carry: `M · p`.
    pub dense_border_coeffs: usize,
}

/// Exact border / covariance byte accounting for both parameterizations, so the
/// orders-of-magnitude claim is asserted from measured numbers.
#[derive(Clone, Debug)]
pub struct CascadeMemoryLedger {
    pub p: usize,
    pub n_regions_accepted: usize,
    pub dense_border_coeffs: usize,
    pub inframe_border_coeffs: usize,
    pub dense_cov_bytes: usize,
    pub inframe_cov_bytes: usize,
    /// `½ · Σ_g r_g(p − r_g) · log N` — the Stage-2 frame charge, paid ONCE
    /// against the whole corpus and amortized across every region.
    pub global_frame_charge: f64,
}

impl CascadeMemoryLedger {
    /// Ratio by which the in-frame border is smaller than the dense border.
    pub fn border_shrink(&self) -> f64 {
        if self.inframe_border_coeffs == 0 {
            return f64::INFINITY;
        }
        self.dense_border_coeffs as f64 / self.inframe_border_coeffs as f64
    }

    /// Ratio by which the in-frame covariance footprint is smaller than dense.
    pub fn cov_shrink(&self) -> f64 {
        if self.inframe_cov_bytes == 0 {
            return f64::INFINITY;
        }
        self.dense_cov_bytes as f64 / self.inframe_cov_bytes as f64
    }
}

/// Result of the in-frame curved cascade over a set of regions.
#[derive(Clone, Debug)]
pub struct InFrameCurvedResult {
    /// Accepted curved atom images, stored in their `r`-dimensional frames.
    /// Ambient `N × p` materialization is intentionally lazy.
    pub curved_prediction: InFrameCurvedPrediction,
    pub records: Vec<RegionRecord>,
    pub accepted_regions: Vec<usize>,
    pub ledger: CascadeMemoryLedger,
}

/// Data occupancy for one weight-sourced catalog frame.
#[derive(Clone, Debug)]
pub struct WeightFrameOccupancy {
    /// Index into [`WeightFrameCatalog::entries`].
    pub frame_index: usize,
    /// Corpus rows whose residual coordinates occupy this weight frame.
    pub rows: Vec<usize>,
    /// Atom basis size `M_k` for the chart to fit in this frame.
    pub basis_size: usize,
}

/// Accepted in-frame prediction for one curved region.
#[derive(Clone, Debug)]
pub struct InFrameCurvedRegionPrediction {
    pub rows: Vec<usize>,
    pub frame: GrassmannFrame,
    pub frame_source: Option<WeightFrameSource>,
    /// Fitted chart image in frame coordinates (`rows.len() × frame.rank()`).
    pub fitted_coords: Array2<f64>,
}

impl InFrameCurvedRegionPrediction {
    pub fn frame_rank(&self) -> usize {
        self.frame.rank()
    }

    pub fn frame_source(&self) -> Option<&WeightFrameSource> {
        self.frame_source.as_ref()
    }

    pub fn inframe_entries(&self) -> usize {
        self.fitted_coords.len()
    }

    pub fn ambient_entries_if_materialized(&self) -> usize {
        self.rows
            .len()
            .saturating_mul(self.frame.output_dim())
    }

    pub fn materialize_ambient(&self) -> Array2<f64> {
        fast_abt(&self.fitted_coords, &self.frame.frame().to_owned())
    }

    fn fill_row_into(&self, local_row: usize, out: &mut [f64]) {
        let u = self.frame.frame();
        let r = self.frame.rank();
        for c in 0..self.frame.output_dim() {
            let mut acc = 0.0;
            for axis in 0..r {
                acc += self.fitted_coords[[local_row, axis]] * u[[c, axis]];
            }
            out[c] = acc;
        }
    }
}

/// Lazy curved prediction over the full corpus. The hot fit path stores only
/// accepted `N_g × r_g` images plus frames; callers must ask explicitly for an
/// ambient slice or full materialization.
#[derive(Clone, Debug)]
pub struct InFrameCurvedPrediction {
    n_rows: usize,
    output_dim: usize,
    regions: Vec<InFrameCurvedRegionPrediction>,
}

impl InFrameCurvedPrediction {
    pub fn new(
        n_rows: usize,
        output_dim: usize,
        regions: Vec<InFrameCurvedRegionPrediction>,
    ) -> Self {
        Self {
            n_rows,
            output_dim,
            regions,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn regions(&self) -> &[InFrameCurvedRegionPrediction] {
        &self.regions
    }

    pub fn inframe_entries(&self) -> usize {
        self.regions
            .iter()
            .map(InFrameCurvedRegionPrediction::inframe_entries)
            .sum()
    }

    pub fn ambient_entries_if_materialized(&self) -> usize {
        self.n_rows.saturating_mul(self.output_dim)
    }

    pub fn accepted_ambient_entries_if_eager(&self) -> usize {
        self.regions
            .iter()
            .map(InFrameCurvedRegionPrediction::ambient_entries_if_materialized)
            .sum()
    }

    pub fn materialize_rows(&self, rows: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows.len(), self.output_dim));
        let mut row_buf = vec![0.0_f64; self.output_dim];
        for (out_row, &global_row) in rows.iter().enumerate() {
            for region in &self.regions {
                if let Some(local_row) = region.rows.iter().position(|&row| row == global_row) {
                    region.fill_row_into(local_row, &mut row_buf);
                    for c in 0..self.output_dim {
                        out[[out_row, c]] = row_buf[c];
                    }
                    break;
                }
            }
        }
        out
    }

    pub fn materialize_ambient(&self) -> Array2<f64> {
        let rows: Vec<usize> = (0..self.n_rows).collect();
        self.materialize_rows(&rows)
    }
}

/// Fit the curved stage in learned low-rank ambient frames.
///
/// `residual` is the Stage-1 residual `R = X − X̂_linear` (`N × p`), `regions`
/// the curved candidates, `n_tokens_total` the corpus token count `N` used to
/// amortize the global frame charge.
pub fn fit_inframe_curved_regions(
    residual: ArrayView2<'_, f64>,
    regions: &[CurvedRegion],
    n_tokens_total: usize,
    config: &InFrameCurvedConfig,
) -> Result<InFrameCurvedResult, String> {
    let p = residual.ncols();
    if p == 0 {
        return Err("fit_inframe_curved_regions: residual must have p >= 1".to_string());
    }
    if !(config.alpha > 0.0 && config.alpha <= 1.0) {
        return Err("fit_inframe_curved_regions: alpha must be in (0, 1]".to_string());
    }
    if config.frame_rank_min == 0 || config.frame_rank_max < config.frame_rank_min {
        return Err("fit_inframe_curved_regions: require 1 <= frame_rank_min <= frame_rank_max"
            .to_string());
    }

    let mut fits: Vec<Option<RegionFit>> = Vec::with_capacity(regions.len());
    for region in regions {
        fits.push(fit_one_region(residual, region, config)?);
    }

    // e-BH selection over regions with a valid pre-gate acceptance.
    let mut logs = Vec::new();
    let mut refs = Vec::new();
    for (i, fit) in fits.iter().enumerate() {
        if let Some(f) = fit {
            if f.evidence.accepted_pre_ebh && f.evidence.margin >= config.min_effect {
                logs.push(f.evidence.log_e_value);
                refs.push(i);
            }
        }
    }
    let keep = ebh(&logs, config.alpha);
    for (&ok, &i) in keep.iter().zip(refs.iter()) {
        if let Some(f) = fits[i].as_mut() {
            f.evidence.accepted = ok;
        }
    }

    // Assemble lazy prediction records and the measured memory ledger. No `N × p`
    // curved image is formed here; accepted atom images remain `N_g × r_g`.
    let mut prediction_regions = Vec::new();
    let mut records = Vec::with_capacity(regions.len());
    let mut accepted_regions = Vec::new();
    let mut dense_border = 0usize;
    let mut inframe_border = 0usize;
    let mut dense_cov = 0usize;
    let mut inframe_cov = 0usize;
    let mut frame_charge = 0.0f64;
    let ln_n = (n_tokens_total.max(2) as f64).ln();

    for (i, (region, fit)) in regions.iter().zip(fits.iter()).enumerate() {
        let Some(fit) = fit else {
            continue;
        };
        let m = region.basis_size;
        let r = fit.frame_rank;
        let inframe_border_coeffs = m.saturating_mul(r);
        let dense_border_coeffs = m.saturating_mul(p);
        let manifold_dim = r.saturating_mul(p.saturating_sub(r));
        records.push(RegionRecord {
            region: i,
            frame_source: None,
            frame_catalog_index: None,
            occupancy_status: ChartOccupancyStatus::Occupied,
            basis_size: m,
            frame_rank: r,
            frame_manifold_dim: manifold_dim,
            evidence: fit.evidence.clone(),
            inframe_border_coeffs,
            dense_border_coeffs,
        });
        if !fit.evidence.accepted {
            continue;
        }
        accepted_regions.push(i);
        dense_border += dense_border_coeffs;
        inframe_border += inframe_border_coeffs;
        dense_cov += dense_border_coeffs
            .saturating_mul(dense_border_coeffs)
            .saturating_mul(BYTES_PER_F64);
        inframe_cov += inframe_border_coeffs
            .saturating_mul(inframe_border_coeffs)
            .saturating_mul(BYTES_PER_F64);
        frame_charge += 0.5 * manifold_dim as f64 * ln_n;
        prediction_regions.push(InFrameCurvedRegionPrediction {
            rows: region.rows.clone(),
            frame: fit.frame.clone(),
            frame_source: None,
            fitted_coords: fit.fitted_coords.clone(),
        });
    }

    let ledger = CascadeMemoryLedger {
        p,
        n_regions_accepted: accepted_regions.len(),
        dense_border_coeffs: dense_border,
        inframe_border_coeffs: inframe_border,
        dense_cov_bytes: dense_cov,
        inframe_cov_bytes: inframe_cov,
        global_frame_charge: frame_charge,
    };

    Ok(InFrameCurvedResult {
        curved_prediction: InFrameCurvedPrediction::new(residual.nrows(), p, prediction_regions),
        records,
        accepted_regions,
        ledger,
    })
}

/// Fit curved charts inside a weight-sourced frame catalog.
///
/// The catalog determines chartability and mechanism attribution. The supplied
/// occupancies are the only data-dependent part: rows decide whether a frame is
/// occupied here and provide the coordinates for evidence scoring. A catalog
/// frame with zero rows still emits a [`RegionRecord`] with
/// [`ChartOccupancyStatus::ChartableUnoccupied`].
pub fn fit_inframe_curved_weight_frame_catalog(
    residual: ArrayView2<'_, f64>,
    catalog: &WeightFrameCatalog,
    occupancies: &[WeightFrameOccupancy],
    n_tokens_total: usize,
    config: &InFrameCurvedConfig,
) -> Result<InFrameCurvedResult, String> {
    let p = residual.ncols();
    if p == 0 {
        return Err("fit_inframe_curved_weight_frame_catalog: residual must have p >= 1".to_string());
    }
    if catalog.output_dim() != p {
        return Err(format!(
            "fit_inframe_curved_weight_frame_catalog: catalog output dim {} != residual dim {p}",
            catalog.output_dim()
        ));
    }
    if !(config.alpha > 0.0 && config.alpha <= 1.0) {
        return Err(
            "fit_inframe_curved_weight_frame_catalog: alpha must be in (0, 1]".to_string(),
        );
    }

    let mut fits: Vec<Option<RegionFit>> = Vec::with_capacity(occupancies.len());
    let mut statuses = Vec::with_capacity(occupancies.len());
    for occupancy in occupancies {
        let entry = catalog.entry(occupancy.frame_index).ok_or_else(|| {
            format!(
                "fit_inframe_curved_weight_frame_catalog: frame index {} out of range",
                occupancy.frame_index
            )
        })?;
        if occupancy.rows.is_empty() {
            fits.push(None);
            statuses.push(ChartOccupancyStatus::ChartableUnoccupied);
            continue;
        }
        let region = CurvedRegion {
            rows: occupancy.rows.clone(),
            basis_size: occupancy.basis_size,
        };
        let fit = fit_one_region_in_frame(residual, &region, &entry.frame, config)?;
        let status = if fit.is_some() {
            ChartOccupancyStatus::Occupied
        } else {
            ChartOccupancyStatus::InsufficientOccupancy
        };
        fits.push(fit);
        statuses.push(status);
    }

    let mut logs = Vec::new();
    let mut refs = Vec::new();
    for (i, fit) in fits.iter().enumerate() {
        if let Some(f) = fit {
            if f.evidence.accepted_pre_ebh && f.evidence.margin >= config.min_effect {
                logs.push(f.evidence.log_e_value);
                refs.push(i);
            }
        }
    }
    let keep = ebh(&logs, config.alpha);
    for (&ok, &i) in keep.iter().zip(refs.iter()) {
        if let Some(f) = fits[i].as_mut() {
            f.evidence.accepted = ok;
        }
    }

    let mut prediction_regions = Vec::new();
    let mut records = Vec::with_capacity(occupancies.len());
    let mut accepted_regions = Vec::new();
    let mut dense_border = 0usize;
    let mut inframe_border = 0usize;
    let mut dense_cov = 0usize;
    let mut inframe_cov = 0usize;
    let mut frame_charge = 0.0f64;
    let ln_n = (n_tokens_total.max(2) as f64).ln();

    for (i, occupancy) in occupancies.iter().enumerate() {
        let entry = catalog.entry(occupancy.frame_index).expect("validated catalog index");
        let r = entry.frame.rank();
        let m = occupancy.basis_size;
        let inframe_border_coeffs = m.saturating_mul(r);
        let dense_border_coeffs = m.saturating_mul(p);
        let manifold_dim = r.saturating_mul(p.saturating_sub(r));
        let evidence = fits[i]
            .as_ref()
            .map(|fit| fit.evidence.clone())
            .unwrap_or_else(|| empty_evidence(occupancy.rows.len(), r));
        records.push(RegionRecord {
            region: i,
            frame_source: Some(entry.source.clone()),
            frame_catalog_index: Some(occupancy.frame_index),
            occupancy_status: statuses[i].clone(),
            basis_size: m,
            frame_rank: r,
            frame_manifold_dim: manifold_dim,
            evidence,
            inframe_border_coeffs,
            dense_border_coeffs,
        });
        let Some(fit) = fits[i].as_ref() else {
            continue;
        };
        if !fit.evidence.accepted {
            continue;
        }
        accepted_regions.push(i);
        dense_border += dense_border_coeffs;
        inframe_border += inframe_border_coeffs;
        dense_cov += dense_border_coeffs
            .saturating_mul(dense_border_coeffs)
            .saturating_mul(BYTES_PER_F64);
        inframe_cov += inframe_border_coeffs
            .saturating_mul(inframe_border_coeffs)
            .saturating_mul(BYTES_PER_F64);
        frame_charge += 0.5 * manifold_dim as f64 * ln_n;
        prediction_regions.push(InFrameCurvedRegionPrediction {
            rows: occupancy.rows.clone(),
            frame: fit.frame.clone(),
            frame_source: Some(entry.source.clone()),
            fitted_coords: fit.fitted_coords.clone(),
        });
    }

    let n_regions_accepted = accepted_regions.len();
    Ok(InFrameCurvedResult {
        curved_prediction: InFrameCurvedPrediction::new(residual.nrows(), p, prediction_regions),
        records,
        accepted_regions,
        ledger: CascadeMemoryLedger {
            p,
            n_regions_accepted,
            dense_border_coeffs: dense_border,
            inframe_border_coeffs: inframe_border,
            dense_cov_bytes: dense_cov,
            inframe_cov_bytes: inframe_cov,
            global_frame_charge: frame_charge,
        },
    })
}

/// Fit ONE region's curved chart purely in its learned ambient frame and return
/// the learned frame rank together with the ambient curved prediction
/// (`n_g × p`, the in-frame chart lifted through the frame). This is the
/// gate-free single-region path — the arithmetic is `r`-dimensional throughout
/// and `p` only reappears in the final lift `Ẑ Uᵀ`. Returns `None` when the
/// region admits no beneficial low-rank frame (it belongs on the certified
/// full-`p` path). Exposed for drivers and parity checks; the gated
/// [`fit_inframe_curved_regions`] shares the same internals.
pub fn inframe_curved_region_prediction(
    residual: ArrayView2<'_, f64>,
    rows: &[usize],
    config: &InFrameCurvedConfig,
) -> Result<Option<InFrameCurvedRegionPrediction>, String> {
    if rows.len() < 2 {
        return Ok(None);
    }
    let r_g = take_rows(residual, rows);
    let Some(frame) = learn_frame(&r_g, config)? else {
        return Ok(None);
    };
    let z = fast_ab(&r_g, &frame.frame().to_owned());
    let fitted = fit_radial_all(&z, config.whitening_ridge)?;
    Ok(Some(InFrameCurvedRegionPrediction {
        rows: rows.to_vec(),
        frame,
        frame_source: None,
        fitted_coords: fitted,
    }))
}

/// PRODUCTION SEAM. Learn the low-rank ambient frame of a curved region's
/// residual span *before* any dense fit, returned as the exact
/// [`GrassmannFrame`] the arrow-Schur assembly consumes through
/// `SaeManifoldAtom::decoder_frame` / `SaeManifoldTerm::any_frame_active`.
///
/// Assigning this to a curved atom's `decoder_frame` flips the term onto its
/// `frames_engaged` factored path (border `M·r`, posterior covariance `(M·r)²`)
/// so the composed / terminal joint fit never materializes the dense
/// `O((M·p)²)` joint Hessian that OOMs at LLM width. Unlike
/// `SaeManifoldTerm::auto_activate_decoder_frames` — which SVDs an already-fitted
/// full-`p` decoder and therefore only helps *after* the dense fit has already
/// paid the memory — this learns the frame straight from the region residual, so
/// the dense fit is never run.
///
/// Returns `None` when the region admits no beneficial low-rank frame (its span
/// fills the ambient width): that region belongs on the certified full-`p` path
/// and the caller must leave `decoder_frame` unset (dense) for it.
///
/// NOTE for the wiring lane: the existing arrow-Schur gates
/// `frames_engaged = any_frame_active() && !whitens_likelihood` — frames + a
/// `WhitenedStructured` metric are out of scope (#974). Seed frames on the
/// isotropic / OutputFisher / no-metric composed fit; under active structured
/// whitening the assembly correctly falls back to the dense path.
pub fn residual_span_frame(
    residual: ArrayView2<'_, f64>,
    rows: &[usize],
    config: &InFrameCurvedConfig,
) -> Result<Option<GrassmannFrame>, String> {
    if rows.len() < 2 {
        return Ok(None);
    }
    let r_g = take_rows(residual, rows);
    learn_frame(&r_g, config)
}

/// PRODUCTION WIRING HOOK — the single call the composed / curved fit makes to
/// route a curved atom through the in-frame path instead of the dense ambient
/// Hessian. Learns the atom's low-rank frame from the residual it will explain
/// (`residual`, rows `rows`) and installs it EXACTLY the way
/// `SaeManifoldAtom::maybe_activate_decoder_frame` installs a decoder-derived
/// frame: sets `decoder_frame` AND projects the decoder `B ← (B U) Uᵀ` so the
/// factored `B = C Uᵀ` holds from the first assembly (without the projection the
/// factored C-solve moves only within `range(U)` and the fit never converges).
///
/// The difference from `maybe_activate_decoder_frame` is WHERE the frame comes
/// from: there it is the SVD of an already-fitted full-`p` decoder (so the dense
/// O((M·p)²) fit has already paid the memory); here it is the residual span,
/// learned BEFORE any dense fit, so the wall is never hit. After this returns
/// `Some(r)`, `term.any_frame_active()` is true and the arrow-Schur assembly
/// takes its `frames_engaged` factored path (border `M·r`, covariance `(M·r)²`).
///
/// Returns the installed frame rank, or `None` when the residual admits no
/// beneficial low-rank frame (the atom stays on the certified full-`p` path).
///
/// Wiring note: engages only when the term is NOT whitened
/// (`frames_engaged = any_frame_active() && !whitens_likelihood`, #974). Call it
/// on the isotropic / OutputFisher composed fit.
pub fn activate_residual_frame(
    atom: &mut crate::manifold::SaeManifoldAtom,
    residual: ArrayView2<'_, f64>,
    rows: &[usize],
    config: &InFrameCurvedConfig,
) -> Result<Option<usize>, String> {
    // Same benefit policy as `SaeManifoldAtom::decoder_frame_activation_rank`:
    // never engage a frame below the minimum output width, and only when the
    // learned rank materially shrinks the border (`r ≤ p·(1 − margin)`). Below
    // that the dense full-`p` path is both cheaper and the certified reference,
    // so the atom stays on it — this also keeps the small-`p` synthetic fits
    // bit-for-bit unchanged (frames were never beneficial there).
    let p = residual.ncols();
    if p < SAE_FRAME_MIN_AUTO_OUTPUT_DIM {
        atom.decoder_frame = None;
        return Ok(None);
    }
    let Some(frame) = residual_span_frame(residual, rows, config)? else {
        atom.decoder_frame = None;
        return Ok(None);
    };
    let r = frame.rank();
    if (r as f64) > (p as f64) * (1.0 - SAE_FRAME_ACTIVATION_MARGIN) {
        atom.decoder_frame = None;
        return Ok(None);
    }
    let u = frame.frame().to_owned(); // p × r
    // Project the decoder onto the frame so B = C·Uᵀ holds exactly (mirrors
    // maybe_activate_decoder_frame's B ← (B U) Uᵀ convergence guard).
    let c_proj = fast_ab(&atom.decoder_coefficients, &u); // M × r
    atom.decoder_coefficients = fast_abt(&c_proj, &u); // M × p
    atom.decoder_frame = Some(frame);
    Ok(Some(r))
}

struct RegionFit {
    frame: GrassmannFrame,
    frame_rank: usize,
    /// Fitted in-frame coordinate prediction (`n_g × r`) — the radial chart
    /// evaluated on the region's rows, ready to lift through the frame.
    fitted_coords: Array2<f64>,
    evidence: RegionEvidence,
}

fn fit_one_region(
    residual: ArrayView2<'_, f64>,
    region: &CurvedRegion,
    config: &InFrameCurvedConfig,
) -> Result<Option<RegionFit>, String> {
    let n_g = region.rows.len();
    if n_g < config.min_rows || n_g < 2 * config.crossfit_folds.max(2) {
        return Ok(None);
    }
    let r_g = take_rows(residual, &region.rows);

    let frame = match learn_frame(&r_g, config)? {
        Some(f) => f,
        None => return Ok(None),
    };
    let r = frame.rank();
    // In-frame coordinates Z = R_g · U  (n_g × r). The curved engine only ever
    // sees these r columns; p never reappears until the ambient lift.
    let mut z = fast_ab(&r_g, &frame.frame().to_owned());

    let evidence = crossfit_evidence(&z, config, n_g)?;

    // Optional slow-timescale frame refresh: re-polar from the fitted
    // cross-moment and keep it ONLY if held-out in-frame EV improves.
    let mut frame = frame;
    if config.frame_refresh {
        if let Some((refreshed, refreshed_z, refreshed_ev)) =
            try_frame_refresh(&r_g, &z, &frame, config, n_g)?
        {
            if refreshed_ev.deviance_gain > evidence.deviance_gain {
                frame = refreshed;
                z = refreshed_z;
                let fitted_coords = fit_radial_all(&z, config.whitening_ridge)?;
                return Ok(Some(RegionFit {
                    frame,
                    frame_rank: r,
                    fitted_coords,
                    evidence: refreshed_ev,
                }));
            }
        }
    }

    let fitted_coords = fit_radial_all(&z, config.whitening_ridge)?;
    Ok(Some(RegionFit {
        frame,
        frame_rank: r,
        fitted_coords,
        evidence,
    }))
}

fn fit_one_region_in_frame(
    residual: ArrayView2<'_, f64>,
    region: &CurvedRegion,
    frame: &GrassmannFrame,
    config: &InFrameCurvedConfig,
) -> Result<Option<RegionFit>, String> {
    let n_g = region.rows.len();
    if n_g < config.min_rows || n_g < 2 * config.crossfit_folds.max(2) {
        return Ok(None);
    }
    if frame.output_dim() != residual.ncols() {
        return Err(format!(
            "fit_one_region_in_frame: frame output dim {} != residual dim {}",
            frame.output_dim(),
            residual.ncols()
        ));
    }
    let r_g = take_rows(residual, &region.rows);
    let z = fast_ab(&r_g, &frame.frame().to_owned());
    let evidence = crossfit_evidence(&z, config, n_g)?;
    let fitted_coords = fit_radial_all(&z, config.whitening_ridge)?;
    Ok(Some(RegionFit {
        frame: frame.clone(),
        frame_rank: frame.rank(),
        fitted_coords,
        evidence,
    }))
}

fn empty_evidence(n_rows: usize, frame_rank: usize) -> RegionEvidence {
    RegionEvidence {
        n_rows,
        n_effective: n_rows as f64,
        frame_rank,
        linear_loss: 0.0,
        chart_loss: 0.0,
        deviance_gain: 0.0,
        mean_delta: 0.0,
        se: f64::INFINITY,
        ci_low: f64::NEG_INFINITY,
        charge: 0.0,
        margin: 0.0,
        log_e_value: 0.0,
        accepted_pre_ebh: false,
        accepted: false,
    }
}

/// Learn the region's ambient frame from its residual span: the top-`r` right
/// singular subspace of `R_g` (`n_g × p`), with `r` the numerical rank at the
/// relative spectral cutoff clamped to the configured band. Returns `None` when
/// no beneficial low-rank frame exists (rank fills the ambient width, so the
/// region stays on the certified full-`p` path).
fn learn_frame(r_g: &Array2<f64>, config: &InFrameCurvedConfig) -> Result<Option<GrassmannFrame>, String> {
    let (n_g, p) = r_g.dim();
    let (_u, sv, vt_opt) = r_g
        .svd(false, true)
        .map_err(|e| format!("inframe learn_frame: SVD failed: {e}"))?;
    let vt = vt_opt.ok_or_else(|| "inframe learn_frame: SVD returned no right factor".to_string())?;
    let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
    if !(max_sv > 0.0) {
        return Ok(None);
    }
    let tol = config.rank_cutoff * max_sv;
    let numerical_rank = sv.iter().filter(|&&v| v > tol).count();
    // Clamp to the band, to the available singular directions, and keep a
    // positive Grassmann dimension (r <= p - 1).
    let available = vt.nrows().min(n_g).min(p.saturating_sub(1));
    let r = numerical_rank
        .max(config.frame_rank_min)
        .min(config.frame_rank_max)
        .min(available);
    if r == 0 || p.saturating_sub(r) == 0 {
        return Ok(None);
    }
    // Frame columns = top-r rows of Vᵀ, transposed → V[:, 0..r]  (p × r).
    let mut frame = Array2::<f64>::zeros((p, r));
    for col in 0..r {
        for row in 0..p {
            frame[[row, col]] = vt[[col, row]];
        }
    }
    let mut gauge = Array1::<f64>::zeros(r);
    for i in 0..r {
        gauge[i] = sv.get(i).copied().unwrap_or(0.0);
    }
    Ok(Some(GrassmannFrame::from_oriented(frame, gauge)))
}

/// One closed-form polar refresh: accumulate the decoder-target cross-moment of
/// the region's residual against its current in-frame coordinates and re-polar.
/// Returns the refreshed frame, refreshed coordinates, and re-scored evidence.
fn try_frame_refresh(
    r_g: &Array2<f64>,
    z: &Array2<f64>,
    frame: &GrassmannFrame,
    config: &InFrameCurvedConfig,
    n_g: usize,
) -> Result<Option<(GrassmannFrame, Array2<f64>, RegionEvidence)>, String> {
    let (p, r) = (frame.output_dim(), frame.rank());
    let mut cross = GrassmannCrossMoment::new(p, r);
    cross.accumulate(r_g.view(), z.view())?;
    let refreshed = cross.polar_frame()?;
    let z_new = fast_ab(r_g, &refreshed.frame().to_owned());
    let ev = crossfit_evidence(&z_new, config, n_g)?;
    Ok(Some((refreshed, z_new, ev)))
}

/// Cross-fit held-out deviance of the in-frame radial curved chart over the
/// in-frame rank-1 linear (PCA) baseline. All arithmetic is `r`-dimensional.
fn crossfit_evidence(
    z: &Array2<f64>,
    config: &InFrameCurvedConfig,
    n_g: usize,
) -> Result<RegionEvidence, String> {
    let n = z.nrows();
    let r = z.ncols();
    let folds = config.crossfit_folds.max(2).min(n);
    let mut linear_loss = vec![0.0f64; n];
    let mut chart_loss = vec![0.0f64; n];
    for fold in 0..folds {
        let mut train = Vec::new();
        let mut eval = Vec::new();
        for i in 0..n {
            if i % folds == fold {
                eval.push(i);
            } else {
                train.push(i);
            }
        }
        if train.len() < 2 || eval.is_empty() {
            continue;
        }
        let train_z = take_index(z, &train);
        let eval_z = take_index(z, &eval);
        let whitening = Whitening::fit(&train_z, config.whitening_ridge)?;
        let train_w = whitening.transform(&train_z);
        let eval_w = whitening.transform(&eval_z);
        let linear_pred_w = pca_rank1_reconstruct(&train_w, &eval_w)?;
        let chart_pred_w = radial_predict(&train_w, &eval_w);
        let linear_pred = whitening.inverse(&linear_pred_w);
        let chart_pred = whitening.inverse(&chart_pred_w);
        for (pos, &row) in eval.iter().enumerate() {
            linear_loss[row] = row_sse(&eval_z, &linear_pred, pos);
            chart_loss[row] = row_sse(&eval_z, &chart_pred, pos);
        }
    }
    let delta: Vec<f64> = linear_loss
        .iter()
        .zip(chart_loss.iter())
        .map(|(l, c)| l - c)
        .collect();
    let n_eff = autocorr_ess(&delta);
    let mean_delta = delta.iter().sum::<f64>() / n as f64;
    let se = newey_west_se(&delta);
    let ci_low = mean_delta - 1.959963984540054 * se;
    let linear_total: f64 = linear_loss.iter().sum();
    let chart_total: f64 = chart_loss.iter().sum();
    // Profiled Gaussian held-out deviance gain, in NATS. The held-out residual
    // lives in the `r`-dimensional frame subspace, so it is an r-DIMENSIONAL
    // isotropic Gaussian: its profiled per-row deviance carries the log-det
    // factor `r/2`, giving the total `½·r·n_eff·ln(SSE_lin/SSE_chart)` — NOT the
    // scalar `½·n_eff·ln(...)`. Omitting `r` made the gain `r ×` too small
    // against the `r·ln(n_eff)` charge below (d_eff = 2r), structurally
    // suppressing higher-rank in-frame charts. The ratio is scale-invariant, so
    // `gain − charge` stays unit-free. Totals floored so a perfect chart fit
    // yields a large finite gain, not ln(∞).
    let sse_floor = 1.0e-12 * (linear_total.max(chart_total)).max(1.0e-300);
    let gain = 0.5
        * (r.max(1) as f64)
        * n_eff.max(2.0)
        * (linear_total.max(sse_floor) / chart_total.max(sse_floor)).ln();
    // In-frame curved DOF: the radial chart carries a center + scale per frame
    // direction (2r), matching the block-lane chart convention.
    let d_eff = (2 * r).max(1) as f64;
    let charge = 0.5 * d_eff * n_eff.max(2.0).ln();
    let margin = gain - charge;
    // Both terms are nats now; the e-value exponent needs no per-unit rescale.
    let log_e_value = margin.max(0.0);
    Ok(RegionEvidence {
        n_rows: n_g,
        n_effective: n_eff,
        frame_rank: r,
        linear_loss: linear_total,
        chart_loss: chart_total,
        deviance_gain: gain,
        mean_delta,
        se,
        ci_low,
        charge,
        margin,
        log_e_value,
        accepted_pre_ebh: margin >= config.min_effect && ci_low > 0.0,
        accepted: false,
    })
}

/// Fit the radial chart on all rows (the prediction that gets lifted to ambient).
fn fit_radial_all(z: &Array2<f64>, ridge: f64) -> Result<Array2<f64>, String> {
    let whitening = Whitening::fit(z, ridge)?;
    let w = whitening.transform(z);
    let pred_w = radial_predict(&w, &w);
    Ok(whitening.inverse(&pred_w))
}

/// Dense full-`p` radial-chart reference: the SAME curved chart fit in the full
/// ambient width with NO frame. Used by the parity test (must agree with the
/// in-frame fit when the residual lies in `range(U)`) and to demonstrate the
/// `O(p³)` whitening the in-frame path avoids.
pub fn dense_ambient_radial_reference(
    r_g: ArrayView2<'_, f64>,
    ridge: f64,
) -> Result<Array2<f64>, String> {
    let owned = r_g.to_owned();
    fit_radial_all(&owned, ridge)
}

// ----- compact numeric helpers (r-dimensional; self-contained) -----

struct Whitening {
    mean: Vec<f64>,
    eigvec: Array2<f64>,
    scale: Vec<f64>,
    dim: usize,
}

impl Whitening {
    fn fit(x: &Array2<f64>, ridge: f64) -> Result<Self, String> {
        let n = x.nrows();
        let d = x.ncols();
        let mut mean = vec![0.0; d];
        for j in 0..d {
            for i in 0..n {
                mean[j] += x[[i, j]];
            }
            mean[j] /= n.max(1) as f64;
        }
        let mut cov = Array2::<f64>::zeros((d, d));
        for i in 0..n {
            for a in 0..d {
                let va = x[[i, a]] - mean[a];
                for b in 0..d {
                    cov[[a, b]] += va * (x[[i, b]] - mean[b]);
                }
            }
        }
        let denom = (n.saturating_sub(1)).max(1) as f64;
        cov.mapv_inplace(|v| v / denom);
        let (vals, eigvec) = cov
            .eigh(Side::Lower)
            .map_err(|e| format!("inframe whitening eigh failed: {e}"))?;
        let max_eval = vals.iter().copied().fold(0.0, f64::max).max(1.0);
        let floor = ridge.max(f64::EPSILON * max_eval);
        let scale = vals.iter().map(|&v| (v.max(0.0) + floor).sqrt()).collect();
        Ok(Self {
            mean,
            eigvec,
            scale,
            dim: d,
        })
    }

    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.dim));
        for i in 0..x.nrows() {
            for k in 0..self.dim {
                let mut v = 0.0;
                for j in 0..self.dim {
                    v += (x[[i, j]] - self.mean[j]) * self.eigvec[[j, k]];
                }
                out[[i, k]] = v / self.scale[k];
            }
        }
        out
    }

    fn inverse(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.dim));
        for i in 0..x.nrows() {
            for j in 0..self.dim {
                let mut v = self.mean[j];
                for k in 0..self.dim {
                    v += x[[i, k]] * self.scale[k] * self.eigvec[[j, k]];
                }
                out[[i, j]] = v;
            }
        }
        out
    }
}

/// Rank-1 PCA reconstruction of `eval` from the leading eigenvector of `train`.
fn pca_rank1_reconstruct(train: &Array2<f64>, eval: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = train.nrows();
    let d = train.ncols();
    let mut mean = vec![0.0; d];
    for j in 0..d {
        for i in 0..n {
            mean[j] += train[[i, j]];
        }
        mean[j] /= n.max(1) as f64;
    }
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..n {
        for a in 0..d {
            let va = train[[i, a]] - mean[a];
            for b in 0..d {
                cov[[a, b]] += va * (train[[i, b]] - mean[b]);
            }
        }
    }
    let denom = n.saturating_sub(1).max(1) as f64;
    cov.mapv_inplace(|v| v / denom);
    let (vals, eigvec) = cov
        .eigh(Side::Lower)
        .map_err(|e| format!("inframe pca eigh failed: {e}"))?;
    let mut top = 0usize;
    let mut top_val = f64::NEG_INFINITY;
    for (k, &v) in vals.iter().enumerate() {
        if v > top_val {
            top_val = v;
            top = k;
        }
    }
    let mut out = Array2::<f64>::zeros((eval.nrows(), d));
    for i in 0..eval.nrows() {
        let mut score = 0.0;
        for j in 0..d {
            score += (eval[[i, j]] - mean[j]) * eigvec[[j, top]];
        }
        for j in 0..d {
            out[[i, j]] = mean[j] + score * eigvec[[j, top]];
        }
    }
    Ok(out)
}

/// Radial-shell chart: project each whitened row to the train mean radius shell.
fn radial_predict(train: &Array2<f64>, eval: &Array2<f64>) -> Array2<f64> {
    let d = train.ncols();
    let mut radius = 0.0;
    for i in 0..train.nrows() {
        let mut ss = 0.0;
        for j in 0..d {
            ss += train[[i, j]] * train[[i, j]];
        }
        radius += ss.sqrt();
    }
    radius /= train.nrows().max(1) as f64;
    let mut out = Array2::<f64>::zeros(eval.dim());
    for i in 0..eval.nrows() {
        let mut norm = 0.0;
        for j in 0..d {
            norm += eval[[i, j]] * eval[[i, j]];
        }
        norm = norm.sqrt().max(1.0e-12);
        for j in 0..d {
            out[[i, j]] = radius * eval[[i, j]] / norm;
        }
    }
    out
}

fn ebh(logs: &[f64], alpha: f64) -> Vec<bool> {
    let m = logs.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        logs[b]
            .partial_cmp(&logs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut k_star = 0usize;
    for (rank0, &idx) in order.iter().enumerate() {
        let rank = rank0 + 1;
        let threshold = (m as f64 / (alpha * rank as f64)).ln();
        if logs[idx] >= threshold {
            k_star = rank;
        }
    }
    let mut keep = vec![false; m];
    for &idx in order.iter().take(k_star) {
        keep[idx] = true;
    }
    keep
}

fn take_rows(x: ArrayView2<'_, f64>, rows: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.len(), x.ncols()));
    for (i, &row) in rows.iter().enumerate() {
        for j in 0..x.ncols() {
            out[[i, j]] = x[[row, j]];
        }
    }
    out
}

fn take_index(x: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.len(), x.ncols()));
    for (i, &row) in rows.iter().enumerate() {
        for j in 0..x.ncols() {
            out[[i, j]] = x[[row, j]];
        }
    }
    out
}

fn row_sse(a: &Array2<f64>, b: &Array2<f64>, row: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        let d = a[[row, j]] - b[[row, j]];
        s += d * d;
    }
    s
}

fn autocorr_ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return n as f64;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return n as f64;
    }
    let lag_cap = (n as f64).sqrt() as usize;
    let mut rho_sum = 0.0;
    for lag in 1..=lag_cap.max(1).min(n - 1) {
        let mut cov = 0.0;
        for i in lag..n {
            cov += (x[i] - mean) * (x[i - lag] - mean);
        }
        cov /= (n - lag) as f64;
        let rho = cov / var;
        if rho <= 0.0 || !rho.is_finite() {
            break;
        }
        rho_sum += rho;
    }
    (n as f64 / (1.0 + 2.0 * rho_sum)).max(1.0)
}

fn newey_west_se(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return f64::INFINITY;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let lag_cap = (n as f64).sqrt() as usize;
    let mut gamma0 = 0.0;
    for v in x {
        gamma0 += (v - mean) * (v - mean);
    }
    gamma0 /= n as f64;
    let mut var = gamma0;
    for lag in 1..=lag_cap.max(1).min(n - 1) {
        let mut gamma = 0.0;
        for i in lag..n {
            gamma += (x[i] - mean) * (x[i - lag] - mean);
        }
        gamma /= n as f64;
        let w = 1.0 - lag as f64 / (lag_cap as f64 + 1.0);
        var += 2.0 * w * gamma;
    }
    (var.max(0.0) / n as f64).sqrt()
}
