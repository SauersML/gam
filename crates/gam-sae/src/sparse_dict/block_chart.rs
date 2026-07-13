use ndarray::{Array2, ArrayView2, ArrayView3};

use super::block::{
    block_sparse_dictionary_block_coords, block_sparse_dictionary_project_residual,
    block_sparse_dictionary_project_residual_with_codes, reconstruct_block_sparse_rows,
};

#[derive(Clone, Debug)]
pub struct BlockChartComposeConfig {
    pub block_size: usize,
    pub block_topk: usize,
    pub gamma: f32,
    pub residual_target: bool,
    pub min_firings: usize,
    pub max_blocks: usize,
    pub crossfit_folds: usize,
    pub min_effect: f64,
    /// Dimensionless ridge fraction of the largest covariance eigenvalue.
    pub whitening_ridge: f64,
    pub pair_screen: bool,
    pub pair_top_blocks: usize,
    pub max_pairs: usize,
    pub pair_min_cofirings: usize,
    pub pair_min_score: f64,
    /// Router tile width for block-coordinate projection. This is a memory-budget
    /// parameter owned by the caller, not a hidden constant inside composition.
    pub block_tile: usize,
}

/// Declared target FDR level `α` for the genuine universal-inference split-LR
/// e-value + full-family e-BH discovery certificate (#2246). Distinct from the
/// descriptive `selected_by_bic` gate: the screened family FEEDS e-BH at this
/// level, and [`BlockChartComposeResult::fdr_selected_chart_blocks`] /
/// `fdr_selected_chart_pairs` are the candidates confirmed at FDR ≤ `α`. A
/// module constant (not a `BlockChartComposeConfig` field) so the honest
/// discovery certificate is always emitted without churning every caller's
/// explicit config literal.
pub const CHART_FDR_ALPHA: f64 = 0.1;

impl Default for BlockChartComposeConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            block_topk: 32,
            gamma: 1.0,
            residual_target: true,
            min_firings: 64,
            max_blocks: 256,
            crossfit_folds: 2,
            min_effect: 0.0,
            whitening_ridge: 1.0e-8,
            pair_screen: true,
            pair_top_blocks: 64,
            max_pairs: 128,
            pair_min_cofirings: 64,
            pair_min_score: 0.20,
            block_tile: 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChartEvidence {
    pub n_rows: usize,
    pub n_effective: f64,
    /// Total held-out SSE of the linear (rank-1 PCA) comparator (squared-coord).
    pub linear_loss: f64,
    /// Total held-out SSE of the curved (radial chart) predictor (squared-coord).
    pub chart_loss: f64,
    /// Profiled-Gaussian held-out deviance gain in NATS,
    /// `(n_eff/2)·ln(SSE_lin/SSE_chart)` — scale-invariant, same currency as the
    /// complexity charge (the S4 fix; formerly the scale-dependent SSE difference).
    pub deviance_gain: f64,
    /// Mean per-row profiled deviance contribution in NATS (the half-log-ratio
    /// deviance); `ci_low/ci_high` are its ±1.96·se interval, also in nats.
    pub mean_delta: f64,
    pub se: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// Complexity charge `½·d_eff·ln(n_eff)` in NATS.
    pub charge: f64,
    /// `deviance_gain − charge` in NATS — scale-invariant.
    pub margin: f64,
    /// Descriptive model-selection verdict: positive held-out BIC margin above
    /// `min_effect`, with a positive lower confidence bound on mean deviance.
    /// This is not a p-value, e-value, or FDR-controlled discovery.
    pub selected_by_bic: bool,
    /// Genuine universal-inference split-LR log-e-value (#2246): `ln Ē` for the
    /// cross-fit e-value `Ē = p_alt(D1; θ̂_{D0}) / sup_{H0} p_null(D1)` with the
    /// radial-ring alternative fit on a disjoint split and the linear-shell null
    /// re-maximized on the held-out fold. Satisfies `E_{H0}[E] ≤ 1` (unlike the
    /// BIC `margin`), so it — not `margin` — is the quantity fed to e-BH. `−∞`
    /// for a candidate too degenerate to support a curved claim. See
    /// [`super::shell_vs_ring_log_evalue`].
    pub log_e: f64,
    /// Whether the full-family e-BH over every screened candidate's `log_e`
    /// confirmed THIS candidate as genuine curved structure at the configured
    /// `fdr_alpha`. Set by [`compose_block_coordinate_charts`] after the family
    /// is assembled; `false` in a standalone [`ChartEvidence`].
    pub fdr_selected: bool,
}

#[derive(Clone, Debug)]
pub struct BlockChartRecord {
    pub block0: usize,
    pub block1: Option<usize>,
    pub screen_score: f64,
    pub evidence: ChartEvidence,
}

#[derive(Clone, Debug)]
pub struct BlockChartComposeResult {
    pub reconstructed: Array2<f32>,
    pub block_records: Vec<BlockChartRecord>,
    pub pair_records: Vec<BlockChartRecord>,
    pub selected_blocks: Vec<usize>,
    pub selected_chart_blocks: Vec<usize>,
    pub selected_chart_pairs: Vec<(usize, usize)>,
    /// Declared FDR level `α` the discovery certificate below controls at.
    pub fdr_alpha: f64,
    /// Single blocks whose curved chart the full-family e-BH confirms as genuine
    /// structure at FDR ≤ `fdr_alpha` (#2246). This is the honest discovery list;
    /// `selected_chart_blocks` above is the descriptive BIC gate for comparison.
    pub fdr_selected_chart_blocks: Vec<usize>,
    /// Block pairs whose curved chart the full-family e-BH confirms at FDR ≤
    /// `fdr_alpha`.
    pub fdr_selected_chart_pairs: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct BlockSeedManifestConfig {
    pub block_size: usize,
    pub block_topk: usize,
    pub gamma: f32,
    pub residual_target: bool,
    pub n_basis_chart: usize,
    pub include_bases: bool,
    pub name_prefix: String,
    /// Router tile width used when residual-target seed coordinates are emitted.
    pub block_tile: usize,
}

#[derive(Clone, Debug)]
pub struct MdlFeaturizerRow {
    pub name: String,
    pub kind: String,
    pub total_var: f64,
    pub n_tokens: usize,
    pub n_firings: usize,
    pub n_params: usize,
    pub coded_var: Vec<f64>,
    pub g_dict: usize,
    pub k_active: usize,
    pub block_name: Option<String>,
    pub chart_name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BlockSeedRecord {
    pub block: usize,
    pub block_dim: usize,
    pub n_firings: usize,
    pub utilization: f32,
    pub stable_rank: f32,
    pub coded_var: Vec<f64>,
    pub total_var: f64,
    pub block_linear_ev: f64,
    pub basis: Option<Vec<Vec<f32>>>,
    pub mdl_block: MdlFeaturizerRow,
    pub mdl_chart: MdlFeaturizerRow,
    /// Matched description length (bits) of the FLAT / linear block comparator:
    /// `block_size` dictionary columns plus the per-firing coordinate coding bits at
    /// the achieved coordinate-SE resolution (see [`crate::description_length`]).
    pub matched_dl_flat: crate::description_length::MatchedDl,
    /// Matched description length (bits) of the CURVED circle chart: `n_basis_chart`
    /// dictionary columns plus the SAME per-firing coordinate coding bits. The extra
    /// columns are the price curvature pays; the coding bits are matched.
    pub matched_dl_chart: crate::description_length::MatchedDl,
    /// Matched-DL delta `flat − chart` (bits): positive ⇒ the curved chart is the
    /// shorter code (curvature pays), negative ⇒ the flat block is cheaper here. The
    /// honest curved-vs-flat headline, read in bits rather than raw EV.
    pub matched_dl_delta_bits: f64,
}

#[derive(Clone, Debug)]
pub struct BlockSeedManifest {
    pub n_blocks: usize,
    pub block_size: usize,
    pub block_topk: usize,
    pub ambient_p: usize,
    pub gamma: f32,
    pub explained_variance: f64,
    pub residual_target: bool,
    pub n_basis_chart: usize,
    pub blocks: Vec<BlockSeedRecord>,
}

#[derive(Clone, Debug)]
struct Whitening {
    mean: Vec<f64>,
    eigvec: Vec<f64>,
    scale: Vec<f64>,
    dim: usize,
}

#[derive(Clone, Debug)]
struct CandidateFit {
    rows: Vec<usize>,
    blocks: Vec<usize>,
    fitted_coords: Array2<f64>,
    screen_score: f64,
    evidence: ChartEvidence,
}

pub fn compose_block_coordinate_charts(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    config: &BlockChartComposeConfig,
) -> Result<BlockChartComposeResult, String> {
    validate_inputs(x, decoder, blocks, codes, config)?;
    let base = reconstruct_block_sparse_rows(decoder, blocks, codes, config.block_size)?;
    let selected = select_blocks(x, decoder, blocks, codes, config)?;
    let mut singles = Vec::new();
    for &g in &selected {
        let rows = rows_for_block(blocks, g);
        let coords_all = block_coords_for_config(x, decoder, blocks, codes, config, g)?;
        let coords = take_rows(&coords_all, &rows);
        let evidence = crossfit_evidence(&coords, config)?;
        let fitted_coords = fit_radial_chart_all(&coords, config.whitening_ridge)?;
        singles.push(CandidateFit {
            rows,
            blocks: vec![g],
            fitted_coords,
            screen_score: 1.0,
            evidence,
        });
    }

    let mut pairs = Vec::new();
    if config.pair_screen {
        let screens = screen_pairs(x, decoder, blocks, codes, config, &selected)?;
        for (g0, g1, score) in screens {
            if score < config.pair_min_score {
                continue;
            }
            let rows = rows_for_pair(blocks, g0, g1);
            let coords0_all = block_coords_for_config(x, decoder, blocks, codes, config, g0)?;
            let coords1_all = block_coords_for_config(x, decoder, blocks, codes, config, g1)?;
            let coords0 = take_rows(&coords0_all, &rows);
            let coords1 = take_rows(&coords1_all, &rows);
            let coords = hstack(&coords0, &coords1);
            let evidence = crossfit_evidence(&coords, config)?;
            let fitted_coords = fit_radial_chart_all(&coords, config.whitening_ridge)?;
            pairs.push(CandidateFit {
                rows,
                blocks: vec![g0, g1],
                fitted_coords,
                screen_score: score,
                evidence,
            });
        }
    }

    let mut reconstructed = base;
    let mut replaced = vec![vec![false; decoder.nrows() / config.block_size]; x.nrows()];

    for pair in &pairs {
        if !pair.evidence.selected_by_bic {
            continue;
        }
        let b = config.block_size;
        for (local_row, &row) in pair.rows.iter().enumerate() {
            for (slot, &g) in pair.blocks.iter().enumerate() {
                subtract_block_contribution(
                    &mut reconstructed,
                    decoder,
                    blocks,
                    codes,
                    config.block_size,
                    row,
                    g,
                );
                add_lifted_coords(
                    &mut reconstructed,
                    decoder,
                    config.block_size,
                    row,
                    g,
                    pair.fitted_coords
                        .row(local_row)
                        .as_slice()
                        .ok_or("non-contiguous pair row")?,
                    slot * b,
                );
                replaced[row][g] = true;
            }
        }
    }

    for single in &singles {
        if !single.evidence.selected_by_bic {
            continue;
        }
        let g = single.blocks[0];
        for (local_row, &row) in single.rows.iter().enumerate() {
            if replaced[row][g] {
                continue;
            }
            subtract_block_contribution(
                &mut reconstructed,
                decoder,
                blocks,
                codes,
                config.block_size,
                row,
                g,
            );
            add_lifted_coords(
                &mut reconstructed,
                decoder,
                config.block_size,
                row,
                g,
                single
                    .fitted_coords
                    .row(local_row)
                    .as_slice()
                    .ok_or("non-contiguous single row")?,
                0,
            );
        }
    }

    // Genuine FDR-controlled discovery (#2246): the screened family — EVERY
    // single and pair the energy/score screen surfaced — feeds one full-family
    // e-BH over the universal-inference split-LR log-e-values. The screen feeds
    // the family, it does NOT gate it: running e-BH over only the BIC-selected
    // subset would redefine the family post hoc and void the guarantee. Order is
    // singles first, then pairs, so a rejected index < singles.len() is a single.
    let mut family_log_e: Vec<f64> = Vec::with_capacity(singles.len() + pairs.len());
    family_log_e.extend(singles.iter().map(|c| c.evidence.log_e));
    family_log_e.extend(pairs.iter().map(|c| c.evidence.log_e));
    let fdr = super::split_lr_fdr::family_fdr_certificate(family_log_e, CHART_FDR_ALPHA)
        .map_err(|error| format!("block chart e-BH certificate: {error}"))?;
    for &idx in &fdr.rejected {
        if idx < singles.len() {
            singles[idx].evidence.fdr_selected = true;
        } else {
            pairs[idx - singles.len()].evidence.fdr_selected = true;
        }
    }
    let fdr_selected_chart_blocks = singles
        .iter()
        .filter(|c| c.evidence.fdr_selected)
        .map(|c| c.blocks[0])
        .collect::<Vec<_>>();
    let fdr_selected_chart_pairs = pairs
        .iter()
        .filter(|c| c.evidence.fdr_selected)
        .map(|c| (c.blocks[0], c.blocks[1]))
        .collect::<Vec<_>>();

    let block_records = singles
        .iter()
        .map(|c| BlockChartRecord {
            block0: c.blocks[0],
            block1: None,
            screen_score: c.screen_score,
            evidence: c.evidence.clone(),
        })
        .collect::<Vec<_>>();
    let pair_records = pairs
        .iter()
        .map(|c| BlockChartRecord {
            block0: c.blocks[0],
            block1: Some(c.blocks[1]),
            screen_score: c.screen_score,
            evidence: c.evidence.clone(),
        })
        .collect::<Vec<_>>();
    let selected_chart_blocks = singles
        .iter()
        .filter(|c| c.evidence.selected_by_bic)
        .map(|c| c.blocks[0])
        .collect::<Vec<_>>();
    let selected_chart_pairs = pairs
        .iter()
        .filter(|c| c.evidence.selected_by_bic)
        .map(|c| (c.blocks[0], c.blocks[1]))
        .collect::<Vec<_>>();
    Ok(BlockChartComposeResult {
        reconstructed,
        block_records,
        pair_records,
        selected_blocks: selected,
        selected_chart_blocks,
        selected_chart_pairs,
        fdr_alpha: CHART_FDR_ALPHA,
        fdr_selected_chart_blocks,
        fdr_selected_chart_pairs,
    })
}

pub fn block_sparse_dictionary_firings(
    blocks: ArrayView2<'_, u32>,
    n_blocks: usize,
) -> Result<Vec<usize>, String> {
    let mut counts = vec![0usize; n_blocks];
    for i in 0..blocks.nrows() {
        for j in 0..blocks.ncols() {
            let g = blocks[[i, j]] as usize;
            if g >= n_blocks {
                return Err(format!(
                    "block firings: block index {g} out of range 0..{n_blocks}"
                ));
            }
            counts[g] += 1;
        }
    }
    Ok(counts)
}

pub fn block_sparse_dictionary_seed_manifest(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    block_utilization: &[f32],
    block_stable_rank: &[f32],
    explained_variance: f64,
    config: &BlockSeedManifestConfig,
) -> Result<BlockSeedManifest, String> {
    if config.block_size == 0 {
        return Err("block seed manifest: block_size must be >= 1".to_string());
    }
    if decoder.nrows() == 0 || decoder.nrows() % config.block_size != 0 {
        return Err(
            "block seed manifest: decoder rows must be a positive multiple of block_size"
                .to_string(),
        );
    }
    if x.ncols() != decoder.ncols() {
        return Err(format!(
            "block seed manifest: X has P={} but decoder has P={}",
            x.ncols(),
            decoder.ncols()
        ));
    }
    if blocks.nrows() != x.nrows() {
        return Err(format!(
            "block seed manifest: blocks has {} rows but X has {}",
            blocks.nrows(),
            x.nrows()
        ));
    }
    let n_blocks = decoder.nrows() / config.block_size;
    if block_utilization.len() != n_blocks || block_stable_rank.len() != n_blocks {
        return Err(format!(
            "block seed manifest: block reports must have length {n_blocks}, got {} and {}",
            block_utilization.len(),
            block_stable_rank.len()
        ));
    }
    let firings = block_sparse_dictionary_firings(blocks, n_blocks)?;
    let ambient_var = centered_energy_view(x).max(f64::MIN_POSITIVE);
    let mut records = Vec::with_capacity(n_blocks);
    for g in 0..n_blocks {
        let coords = block_coords_for_seed_config(x, decoder, config, g)?;
        let coded_var = coordinate_spectrum(&coords)?;
        let total_var = coded_var.iter().sum::<f64>();
        let total_var_report = total_var.max(f64::MIN_POSITIVE);
        let block_ev = centered_energy(&coords) / ambient_var;
        let base = format!("{}{}", config.name_prefix, g);
        let n_firings = firings[g].max(1);
        let mdl_block = MdlFeaturizerRow {
            name: format!("{base}-linear-{}d", config.block_size),
            kind: "block".to_string(),
            total_var: total_var_report,
            n_tokens: x.nrows(),
            n_firings,
            n_params: config.block_size * decoder.ncols(),
            coded_var: coded_var.clone(),
            g_dict: n_blocks,
            k_active: config.block_topk,
            block_name: None,
            chart_name: None,
        };
        let chart_name = format!("{base}-circle-chart");
        let block_name = mdl_block.name.clone();
        let mdl_chart = MdlFeaturizerRow {
            name: chart_name.clone(),
            kind: "chart".to_string(),
            total_var: total_var_report,
            n_tokens: x.nrows(),
            n_firings,
            n_params: config.n_basis_chart * decoder.ncols(),
            coded_var: vec![total_var_report],
            g_dict: n_blocks,
            k_active: config.block_topk,
            block_name: Some(block_name),
            chart_name: Some(chart_name),
        };
        // #P3 matched-DL report column: per-firing coordinate SEs from the block's
        // firing radii, coded at the uniform-quantization rule, plus the column
        // charge. Curved (n_basis_chart columns) vs flat (block_size columns) read
        // in bits. Per-firing SE = σ̂/(2π‖z‖) (the b=2 circle rule; a report column,
        // so the general-b block uses it as the phase-resolution proxy).
        let firing_rows = rows_for_block(blocks, g);
        let firing_coords = take_rows(&coords, &firing_rows);
        let (matched_dl_flat, matched_dl_chart) =
            matched_dl_for_block(&firing_coords, config, decoder.ncols(), block_ev);
        let matched_dl_delta_bits =
            crate::description_length::matched_dl_delta(&matched_dl_flat, &matched_dl_chart);
        records.push(BlockSeedRecord {
            block: g,
            block_dim: config.block_size,
            n_firings: firings[g],
            utilization: block_utilization[g],
            stable_rank: block_stable_rank[g],
            coded_var,
            total_var,
            block_linear_ev: block_ev,
            basis: config
                .include_bases
                .then(|| block_basis(decoder, config.block_size, g)),
            mdl_block,
            mdl_chart,
            matched_dl_flat,
            matched_dl_chart,
            matched_dl_delta_bits,
        });
    }
    Ok(BlockSeedManifest {
        n_blocks,
        block_size: config.block_size,
        block_topk: config.block_topk,
        ambient_p: decoder.ncols(),
        gamma: config.gamma,
        explained_variance,
        residual_target: config.residual_target,
        n_basis_chart: config.n_basis_chart,
        blocks: records,
    })
}

/// Matched description length (bits) of the flat/linear block vs the curved circle
/// chart for one block, from its per-firing coordinates.
///
/// Per firing, the radial-scatter noise `σ̂` (unbiased SD of the firing radii `‖z‖`)
/// sets each coordinate's standard error, but the two arms transmit DIFFERENT kinds
/// of coordinate and are coded at their OWN resolution (the S5 fix). The circle
/// chart transmits one PHASE `t = θ/(2π)`, whose arc position ranges over the
/// circumference `2π·â` at the bias-corrected amplitude `â = √max(‖z‖² − 2σ̂², 0)`,
/// so `SE_phase = σ̂/(2π·â)` ([`super::coordinate::phase_coordinate_se`], which
/// evaluates the delta method at the true amplitude rather than the noise-inflated
/// observed radius and saturates at the uniform-phase ceiling). The flat block
/// transmits `block_size` AMPLITUDE coefficients, each ranging over the radius
/// `‖z‖`, so `SE_amp = σ̂/‖z‖`; at the high-SNR regime where `â ≈ ‖z‖` this is
/// `SE_amp ≈ 2π·SE_phase` — a factor `2π` LARGER, hence `log₂(2π)` bits
/// CHEAPER per coordinate. Pricing the flat block's amplitudes at the finer phase
/// rate (as a shared SE did) was an ≈ `block_size·log₂(2π)`-bit-per-firing
/// pro-chart bias; each scalar is now coded at
/// [`crate::description_length::se_resolution_bits`] of its OWN SE. The arms differ
/// in BOTH ledgers: per firing the flat block transmits `block_size` amplitudes
/// where the chart transmits one phase (the code-economy axis), and per dictionary
/// the column charge is `block_size` vs `n_basis_chart` columns of `p` ambient
/// scalars at each arm's own distortion-matched per-scalar precision.
fn matched_dl_for_block(
    firing_coords: &Array2<f32>,
    config: &BlockSeedManifestConfig,
    ambient_p: usize,
    ev: f64,
) -> (
    crate::description_length::MatchedDl,
    crate::description_length::MatchedDl,
) {
    use crate::description_length::{matched_dl, se_resolution_bits};
    // Per-firing radii ‖z‖ and their unbiased radial-scatter noise σ̂.
    let n_fire = firing_coords.nrows();
    let mut norms: Vec<f64> = Vec::with_capacity(n_fire);
    for row in firing_coords.rows() {
        norms.push(
            row.iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                .sqrt(),
        );
    }
    let sigma_hat = if n_fire >= 2 {
        let mean = norms.iter().sum::<f64>() / n_fire as f64;
        let ss: f64 = norms.iter().map(|&r| (r - mean) * (r - mean)).sum();
        (ss / (n_fire - 1) as f64).sqrt()
    } else {
        0.0
    };
    // PER-ARM per-scalar resolution. Both arms derive from the SAME σ̂ and radii,
    // but transmit different coordinates: the chart a phase over the circumference
    // `2π·â` at the bias-corrected amplitude `â = √max(‖z‖²−2σ̂²,0)`
    // (SE_phase = σ̂/(2π·â), saturating at the uniform-phase ceiling), the flat
    // block amplitudes over the radius `‖z‖` (SE_amp = σ̂/‖z‖). se_resolution_bits
    // floors a zero-norm / unidentified firing to 0 bits, so a `+∞` amplitude SE
    // needs no clamp.
    let phase_ses: Vec<f64> = norms
        .iter()
        .map(|&nz| super::coordinate::phase_coordinate_se(sigma_hat, nz))
        .collect();
    let amp_ses: Vec<f64> = norms
        .iter()
        .map(|&nz| {
            if nz > 0.0 {
                sigma_hat / nz
            } else {
                f64::INFINITY
            }
        })
        .collect();
    // Distortion-matched per-scalar decoder precision, PER ARM: the mean per-firing
    // coding rate of THAT arm's coordinate (the `description_length::score`
    // convention — a decoder weight quantised to match its coded coordinate). The
    // flat arm's columns store amplitude directions (amplitude rate); the chart's
    // store harmonic rows read out by the phase (phase rate). Empty ⇒ zero precision.
    let mean_rate = |ses: &[f64]| -> f64 {
        if ses.is_empty() {
            0.0
        } else {
            ses.iter().map(|&se| se_resolution_bits(se)).sum::<f64>() / ses.len() as f64
        }
    };
    let l_param_flat = mean_rate(&amp_ses);
    let l_param_chart = mean_rate(&phase_ses);
    // Per-firing multiplicity is where the code economy lives: the flat block
    // transmits every one of its `block_size` amplitude coefficients per firing, the
    // circle chart transmits ONE phase coordinate — now each at ITS OWN resolution.
    let flat = matched_dl(
        config.block_size as i64,
        config.block_size as i64,
        ambient_p as i64,
        l_param_flat,
        &amp_ses,
        ev,
    );
    let chart = matched_dl(
        config.n_basis_chart as i64,
        1,
        ambient_p as i64,
        l_param_chart,
        &phase_ses,
        ev,
    );
    (flat, chart)
}

fn validate_inputs(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    config: &BlockChartComposeConfig,
) -> Result<(), String> {
    if config.block_size == 0 {
        return Err("block chart compose: block_size must be >= 1".to_string());
    }
    if decoder.nrows() == 0 || decoder.nrows() % config.block_size != 0 {
        return Err(
            "block chart compose: decoder rows must be a positive multiple of block_size"
                .to_string(),
        );
    }
    if x.ncols() != decoder.ncols() {
        return Err(format!(
            "block chart compose: X has P={} but decoder has P={}",
            x.ncols(),
            decoder.ncols()
        ));
    }
    let (n, k) = blocks.dim();
    if n != x.nrows() || codes.shape() != [n, k, config.block_size] {
        return Err(format!(
            "block chart compose: blocks/codes shapes must be ({}, k) and ({}, k, {}), got {:?} and {:?}",
            x.nrows(),
            x.nrows(),
            config.block_size,
            blocks.dim(),
            codes.shape()
        ));
    }
    if config.block_tile == 0 {
        return Err("block chart compose: block_tile must be >= 1".to_string());
    }
    Ok(())
}

fn block_coords_for_config(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    config: &BlockChartComposeConfig,
    block: usize,
) -> Result<Array2<f32>, String> {
    if config.residual_target {
        // Leave-one-block-out residual from the CALLER's codes — the same
        // linear tier `base` reconstructs from — so chart subproblems and the
        // composed objective price one linear state (the co-fit alternation
        // refits codes between passes; a fresh tied re-transform here would
        // target a stale tier).
        block_sparse_dictionary_project_residual_with_codes(
            x,
            decoder,
            blocks,
            codes,
            config.block_size,
            block,
        )
    } else {
        block_sparse_dictionary_block_coords(x, decoder, config.block_size, block)
    }
}

fn block_coords_for_seed_config(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    config: &BlockSeedManifestConfig,
    block: usize,
) -> Result<Array2<f32>, String> {
    if config.residual_target {
        block_sparse_dictionary_project_residual(
            x,
            decoder,
            config.gamma,
            config.block_size,
            config.block_topk,
            config.block_tile,
            block,
        )
    } else {
        block_sparse_dictionary_block_coords(x, decoder, config.block_size, block)
    }
}

fn select_blocks(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    config: &BlockChartComposeConfig,
) -> Result<Vec<usize>, String> {
    let g_total = decoder.nrows() / config.block_size;
    let mut scored = Vec::<(f64, usize, usize)>::new();
    for g in 0..g_total {
        let rows = rows_for_block(blocks, g);
        if rows.len() < config.min_firings {
            continue;
        }
        let coords_all = block_coords_for_config(x, decoder, blocks, codes, config, g)?;
        let coords = take_rows(&coords_all, &rows);
        let energy = centered_energy(&coords);
        scored.push((energy, rows.len(), g));
    }
    scored.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(config.max_blocks.min(scored.len()));
    Ok(scored.into_iter().map(|(_, _, g)| g).collect())
}

fn screen_pairs(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    config: &BlockChartComposeConfig,
    selected: &[usize],
) -> Result<Vec<(usize, usize, f64)>, String> {
    let top = selected.len().min(config.pair_top_blocks);
    let mut out = Vec::new();
    for i in 0..top {
        let g0 = selected[i];
        let z0_all = block_coords_for_config(x, decoder, blocks, codes, config, g0)?;
        for &g1 in selected.iter().take(top).skip(i + 1) {
            let rows = rows_for_pair(blocks, g0, g1);
            if rows.len() < config.pair_min_cofirings {
                continue;
            }
            let z0 = take_rows(&z0_all, &rows);
            let z1_all = block_coords_for_config(x, decoder, blocks, codes, config, g1)?;
            let z1 = take_rows(&z1_all, &rows);
            out.push((g0, g1, pair_score(&z0, &z1)?));
        }
    }
    out.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(config.max_pairs.min(out.len()));
    Ok(out)
}

fn crossfit_evidence(
    coords: &Array2<f32>,
    config: &BlockChartComposeConfig,
) -> Result<ChartEvidence, String> {
    let n = coords.nrows();
    if n < 2 {
        return Err("block chart evidence requires at least two rows".to_string());
    }
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
        let train_coords = take_rows(coords, &train);
        let eval_coords = take_rows(coords, &eval);
        let whitening = fit_whitening(&train_coords, config.whitening_ridge)?;
        let train_w = whitening.transform(&to_f64(&train_coords));
        let eval_w = whitening.transform(&to_f64(&eval_coords));
        let linear_pred_w = pca_reconstruct(&train_w, &eval_w, 1)?;
        let chart_pred_w = radial_predict(&train_w, &eval_w);
        let linear_pred = whitening.inverse(&linear_pred_w);
        let chart_pred = whitening.inverse(&chart_pred_w);
        let eval_f = to_f64(&eval_coords);
        for (pos, &row) in eval.iter().enumerate() {
            linear_loss[row] = row_sse(&eval_f, &linear_pred, pos);
            chart_loss[row] = row_sse(&eval_f, &chart_pred, pos);
        }
    }
    // Held-out losses are SSE (squared coordinate units). Under a coordinate
    // rescale z → c·z every SSE scales by c² while the ½·d_eff·ln(n_eff) complexity
    // charge is dimensionless, so a raw `margin = Σ(SSE_lin − SSE_chart) − charge`
    // MIXES CURRENCIES and lets the measurement units decide acceptance — rescale
    // the coordinates, flip the verdict. The fix scores in the DEVIANCE currency
    // (nats), where the gain lives in the same units as the charge and is
    // scale-invariant.
    //
    // Profiled-Gaussian per-row deviance. With profiled variances
    // ŝ²_lin = SSE_lin/n and ŝ²_chart = SSE_chart/n, row i's Gaussian negative
    // log-likelihood under each model is ½·ln(2π ŝ²) + eᵢ/(2 ŝ²) (eᵢ the row SSE),
    // so the per-row log-loss difference (linear − chart) is
    //
    //   dᵢ = ½·ln(ŝ²_lin / ŝ²_chart) + e_lin,ᵢ/(2 ŝ²_lin) − e_chart,ᵢ/(2 ŝ²_chart).
    //
    // dᵢ is dimensionless (each eᵢ/ŝ² is scale-free; the log is of a ratio), so the
    // whole evidence — mean, se, CI, n_eff and gain — is scale-invariant. It also
    // telescopes: Σ e_lin,ᵢ/(2 ŝ²_lin) = Σ e_chart,ᵢ/(2 ŝ²_chart) = n/2, hence
    // Σ dᵢ = (n/2)·ln(SSE_lin/SSE_chart) and the per-sample mean is exactly the
    // profiled half-log-ratio deviance. Variances are floored RELATIVELY (a
    // fraction of the larger SSE) so a perfect chart fit yields a large finite gain,
    // not ln(∞), while every quantity stays scale-invariant.
    let linear_total = linear_loss.iter().sum::<f64>();
    let chart_total = chart_loss.iter().sum::<f64>();
    let s2_floor = 1.0e-12 * (linear_total.max(chart_total) / n as f64).max(1.0e-300);
    let s2_lin = (linear_total / n as f64).max(s2_floor);
    let s2_chart = (chart_total / n as f64).max(s2_floor);
    let half_log_ratio = 0.5 * (s2_lin / s2_chart).ln();
    // Each row residual `eᵢ = linear_loss[i]` is an SSE over the `q =
    // coords.ncols()` coordinate channels, so the reconstruction is a
    // q-DIMENSIONAL isotropic Gaussian, not a scalar one: its per-component MLE
    // variance is `SSE/(n·q)` and the per-row NLL log-det term carries `q/2`, so
    // the profiled per-row deviance is `q ×` the scalar half-log-ratio form. The
    // ratio `s2_lin/s2_chart` is q-invariant, and `Σ eᵢ/(2 s2)` telescopes the
    // same way, so the whole per-row deviance is exactly `q ×` the scalar
    // expression → `Σ dᵢ = (n·q/2)·ln(SSE_lin/SSE_chart)`. Omitting `q` made
    // curved evidence `q ×` too small against the q-scaled BIC charge below,
    // structurally suppressing multi-coordinate charts.
    let q = coords.ncols().max(1) as f64;
    let deviance: Vec<f64> = (0..n)
        .map(|i| {
            q * (half_log_ratio + linear_loss[i] / (2.0 * s2_lin)
                - chart_loss[i] / (2.0 * s2_chart))
        })
        .collect();
    let n_eff = autocorr_ess(&deviance);
    let mean_delta = deviance.iter().sum::<f64>() / n as f64;
    let se = newey_west_se(&deviance);
    let ci_low = mean_delta - 1.959963984540054 * se;
    let ci_high = mean_delta + 1.959963984540054 * se;
    // Profiled held-out deviance gain in NATS: (n_eff·q/2)·ln(SSE_lin/SSE_chart)
    // = n_eff · mean_delta (the q factor now lives in each dᵢ). Uses the
    // effective (autocorrelation-deflated) sample count — scale- AND q-invariant
    // for the ESS — so cross-fit fold correlation cannot inflate the evidence.
    let d_eff = (2 * coords.ncols()).max(1) as f64;
    let gain = n_eff.max(2.0) * mean_delta;
    let charge = 0.5 * d_eff * n_eff.max(2.0).ln();
    let margin = gain - charge;
    // Genuine universal-inference split-LR log-e-value on the SAME coordinate
    // rows (#2246): a valid `E_{H0}[E] ≤ 1` instrument, unlike the descriptive
    // BIC `margin` above. Fed to the full-family e-BH in
    // `compose_block_coordinate_charts`; `fdr_selected` is set there.
    let coords64 = to_f64(coords);
    let log_e =
        super::split_lr_fdr::shell_vs_ring_log_evalue(&coords64, folds, config.whitening_ridge)?;
    Ok(ChartEvidence {
        n_rows: n,
        n_effective: n_eff,
        linear_loss: linear_total,
        chart_loss: chart_total,
        deviance_gain: gain,
        mean_delta,
        se,
        ci_low,
        ci_high,
        charge,
        margin,
        selected_by_bic: margin >= config.min_effect && ci_low > 0.0,
        log_e,
        fdr_selected: false,
    })
}

fn fit_radial_chart_all(coords: &Array2<f32>, ridge: f64) -> Result<Array2<f64>, String> {
    let whitening = fit_whitening(coords, ridge)?;
    let z = whitening.transform(&to_f64(coords));
    let pred = radial_predict(&z, &z);
    Ok(whitening.inverse(&pred))
}

fn fit_whitening(coords: &Array2<f32>, ridge: f64) -> Result<Whitening, String> {
    let x = to_f64(coords);
    let n = x.nrows();
    let d = x.ncols();
    let mut mean = vec![0.0; d];
    for j in 0..d {
        for i in 0..n {
            mean[j] += x[[i, j]];
        }
        mean[j] /= n.max(1) as f64;
    }
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        for a in 0..d {
            let va = x[[i, a]] - mean[a];
            for b in 0..d {
                cov[a * d + b] += va * (x[[i, b]] - mean[b]);
            }
        }
    }
    let denom = (n.saturating_sub(1)).max(1) as f64;
    for v in &mut cov {
        *v /= denom;
    }
    let (vals, eigvec) = jacobi_eigh(cov, d)?;
    let max_eval = vals.iter().copied().fold(0.0, f64::max);
    if !(ridge.is_finite() && ridge >= 0.0) {
        return Err(format!(
            "whitening ridge fraction must be finite and non-negative; got {ridge}"
        ));
    }
    let spectral_scale = max_eval.max(f64::MIN_POSITIVE);
    let floor = (ridge * spectral_scale).max(f64::EPSILON * spectral_scale);
    let scale = vals
        .into_iter()
        .map(|v| (v.max(0.0) + floor).sqrt())
        .collect();
    Ok(Whitening {
        mean,
        eigvec,
        scale,
        dim: d,
    })
}

impl Whitening {
    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), self.dim));
        for i in 0..x.nrows() {
            for k in 0..self.dim {
                let mut v = 0.0;
                for j in 0..self.dim {
                    v += (x[[i, j]] - self.mean[j]) * self.eigvec[j * self.dim + k];
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
                    v += x[[i, k]] * self.scale[k] * self.eigvec[j * self.dim + k];
                }
                out[[i, j]] = v;
            }
        }
        out
    }
}

fn pca_reconstruct(
    train: &Array2<f64>,
    eval: &Array2<f64>,
    rank: usize,
) -> Result<Array2<f64>, String> {
    let n = train.nrows();
    let d = train.ncols();
    let mut mean = vec![0.0; d];
    for j in 0..d {
        for i in 0..n {
            mean[j] += train[[i, j]];
        }
        mean[j] /= n.max(1) as f64;
    }
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        for a in 0..d {
            let va = train[[i, a]] - mean[a];
            for b in 0..d {
                cov[a * d + b] += va * (train[[i, b]] - mean[b]);
            }
        }
    }
    let denom = n.saturating_sub(1).max(1) as f64;
    for v in &mut cov {
        *v /= denom;
    }
    let (vals, eigvec) = jacobi_eigh(cov, d)?;
    let mut order = (0..d).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        vals[b]
            .partial_cmp(&vals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let r = rank.min(d);
    let mut out = Array2::<f64>::zeros((eval.nrows(), d));
    for i in 0..eval.nrows() {
        for j in 0..d {
            out[[i, j]] = mean[j];
        }
        for &k in order.iter().take(r) {
            let mut score = 0.0;
            for j in 0..d {
                score += (eval[[i, j]] - mean[j]) * eigvec[j * d + k];
            }
            for j in 0..d {
                out[[i, j]] += score * eigvec[j * d + k];
            }
        }
    }
    Ok(out)
}

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

pub(crate) fn jacobi_eigh(mut a: Vec<f64>, n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if a.len() != n * n {
        return Err("jacobi_eigh: matrix length mismatch".to_string());
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..(64 * n.max(1) * n.max(1)) {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off <= 1.0e-12 {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            let akp = a[k * n + p];
            let akq = a[k * n + q];
            a[k * n + p] = c * akp - s * akq;
            a[k * n + q] = s * akp + c * akq;
        }
        for k in 0..n {
            let apk = a[p * n + k];
            let aqk = a[q * n + k];
            a[p * n + k] = c * apk - s * aqk;
            a[q * n + k] = s * apk + c * aqk;
        }
        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }
    let vals = (0..n).map(|i| a[i * n + i]).collect::<Vec<_>>();
    Ok((vals, v))
}

fn pair_score(z0: &Array2<f32>, z1: &Array2<f32>) -> Result<f64, String> {
    let joint = hstack(z0, z1);
    let whitening = fit_whitening(&joint, 1.0e-8)?;
    let z = whitening.transform(&to_f64(&joint));
    let mut radii = Vec::with_capacity(z.nrows());
    for i in 0..z.nrows() {
        let mut ss = 0.0;
        for j in 0..z.ncols() {
            ss += z[[i, j]] * z[[i, j]];
        }
        radii.push(ss.sqrt());
    }
    let mean = radii.iter().sum::<f64>() / radii.len().max(1) as f64;
    if mean <= 0.0 {
        return Ok(0.0);
    }
    let var =
        radii.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / radii.len().max(1) as f64;
    Ok((1.0 - var.sqrt() / mean).clamp(0.0, 1.0))
}

fn rows_for_block(blocks: ArrayView2<'_, u32>, block: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..blocks.nrows() {
        if (0..blocks.ncols()).any(|j| blocks[[i, j]] as usize == block) {
            out.push(i);
        }
    }
    out
}

fn rows_for_pair(blocks: ArrayView2<'_, u32>, g0: usize, g1: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..blocks.nrows() {
        let has0 = (0..blocks.ncols()).any(|j| blocks[[i, j]] as usize == g0);
        let has1 = (0..blocks.ncols()).any(|j| blocks[[i, j]] as usize == g1);
        if has0 && has1 {
            out.push(i);
        }
    }
    out
}

fn take_rows(a: &Array2<f32>, rows: &[usize]) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((rows.len(), a.ncols()));
    for (i, &row) in rows.iter().enumerate() {
        for j in 0..a.ncols() {
            out[[i, j]] = a[[row, j]];
        }
    }
    out
}

fn hstack(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((a.nrows(), a.ncols() + b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[[i, j]] = a[[i, j]];
        }
        for j in 0..b.ncols() {
            out[[i, a.ncols() + j]] = b[[i, j]];
        }
    }
    out
}

fn centered_energy(a: &Array2<f32>) -> f64 {
    let mut means = vec![0.0; a.ncols()];
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            means[j] += a[[i, j]] as f64;
        }
        means[j] /= a.nrows().max(1) as f64;
    }
    let mut e = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let v = a[[i, j]] as f64 - means[j];
            e += v * v;
        }
    }
    e
}

fn centered_energy_view(a: ArrayView2<'_, f32>) -> f64 {
    let mut means = vec![0.0; a.ncols()];
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            means[j] += a[[i, j]] as f64;
        }
        means[j] /= a.nrows().max(1) as f64;
    }
    let mut e = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let v = a[[i, j]] as f64 - means[j];
            e += v * v;
        }
    }
    e
}

fn coordinate_spectrum(coords: &Array2<f32>) -> Result<Vec<f64>, String> {
    let n = coords.nrows();
    let d = coords.ncols();
    let mut means = vec![0.0; d];
    for j in 0..d {
        for i in 0..n {
            means[j] += coords[[i, j]] as f64;
        }
        means[j] /= n.max(1) as f64;
    }
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        for a in 0..d {
            let va = coords[[i, a]] as f64 - means[a];
            for b in 0..d {
                cov[a * d + b] += va * (coords[[i, b]] as f64 - means[b]);
            }
        }
    }
    let denom = n.max(1) as f64;
    for v in &mut cov {
        *v /= denom;
    }
    let (vals, _) = jacobi_eigh(cov, d)?;
    let mut spectrum = vals.into_iter().map(|v| v.max(0.0)).collect::<Vec<_>>();
    spectrum.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(spectrum)
}

fn block_basis(decoder: ArrayView2<'_, f32>, block_size: usize, block: usize) -> Vec<Vec<f32>> {
    let mut basis = vec![vec![0.0; block_size]; decoder.ncols()];
    for p in 0..decoder.ncols() {
        for r in 0..block_size {
            basis[p][r] = decoder[[block * block_size + r, p]];
        }
    }
    basis
}

fn to_f64(a: &Array2<f32>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(a.dim());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            out[[i, j]] = a[[i, j]] as f64;
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

fn subtract_block_contribution(
    out: &mut Array2<f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    b: usize,
    row: usize,
    block: usize,
) {
    for j in 0..blocks.ncols() {
        if blocks[[row, j]] as usize != block {
            continue;
        }
        for r in 0..b {
            let code = codes[[row, j, r]];
            let atom = decoder.row(block * b + r);
            for c in 0..out.ncols() {
                out[[row, c]] -= code * atom[c];
            }
        }
    }
}

fn add_lifted_coords(
    out: &mut Array2<f32>,
    decoder: ArrayView2<'_, f32>,
    b: usize,
    row: usize,
    block: usize,
    coords: &[f64],
    offset: usize,
) {
    for r in 0..b {
        let code = coords[offset + r] as f32;
        let atom = decoder.row(block * b + r);
        for c in 0..out.ncols() {
            out[[row, c]] += code * atom[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-uniform in `[0, 1)` from a counter (no RNG dependency).
    fn unif(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// A 2-D annulus (ring): points near radius 1 with a full spread of angles and
    /// mild radial jitter. The radial (curved) chart predicts each point onto the
    /// mean-radius circle and fits it well, while a rank-1 linear PCA cannot capture
    /// a ring — so the chart genuinely beats the linear comparator (a real ACCEPT).
    fn annulus(n: usize) -> Array2<f32> {
        let mut st = 0x1234_5678u64;
        let mut z = Array2::<f32>::zeros((n, 2));
        for i in 0..n {
            let theta = std::f64::consts::TAU * (i as f64 / n as f64) + 0.01 * unif(&mut st);
            let r = 1.0 + 0.05 * (unif(&mut st) - 0.5);
            z[[i, 0]] = (r * theta.cos()) as f32;
            z[[i, 1]] = (r * theta.sin()) as f32;
        }
        z
    }

    #[test]
    fn crossfit_bic_selection_is_scale_invariant() {
        // The descriptive BIC margin is scored in the profiled-deviance currency
        // (nats), so rescaling the coordinates cannot change the verdict. This
        // exercises the SHIPPED default whitening_ridge (a dimensionless fraction
        // of the largest covariance eigenvalue): the eigenvalue floor is
        // `ridge·max_eval`, which scales with the data like every other term, so
        // the whole pipeline is exactly scale-equivariant and the invariance is
        // pinned to f64 rounding — not an artefact of setting ridge to zero.
        let coords = annulus(400);
        let config = BlockChartComposeConfig::default();
        assert!(
            config.whitening_ridge > 0.0,
            "the default ridge must be exercised (a nonzero absolute ridge would break \
             scale-equivariance; the relative-fraction ridge does not)"
        );
        let base = crossfit_evidence(&coords, &config).expect("evidence");

        // The ring is a genuine curved structure: the chart must beat the linear
        // comparator (a meaningful ACCEPT, not a trivial tie).
        assert!(base.deviance_gain > 0.0, "chart must beat linear on a ring");
        assert!(
            base.selected_by_bic,
            "the ring must be selected by held-out BIC"
        );

        // Rescale every coordinate by 10 (SSE ×100). Deviance-scale scoring must
        // leave the verdict and margin unchanged.
        let mut scaled = coords.clone();
        scaled.mapv_inplace(|v| v * 10.0);
        let big = crossfit_evidence(&scaled, &config).expect("evidence scaled");

        assert_eq!(
            base.selected_by_bic, big.selected_by_bic,
            "accept/reject verdict must be scale-invariant"
        );
        // The deviance currency is scale-invariant in EXACT arithmetic — every
        // dᵢ depends only on the ratio SSE_lin/SSE_chart and the telescoping
        // eᵢ/(2 ŝ²) terms, all of which are unit-free. The residual divergence is
        // pure floating-point: the upstream fit (whitening + relative-ridge
        // regression on a ×10-rescaled Gram matrix) is only scale-EQUIVARIANT up
        // to roundoff, and that ~ε accumulates through SSE → deviance → gain. On
        // the ~3e3-magnitude margin/gain the gap is ~1e-4 absolute ≈ 4e-8
        // relative — a few ×√ε, no summation reordering here makes it bit-exact.
        // Assert a principled RELATIVE tolerance well above the roundoff floor
        // (1e-6) yet far below any physically meaningful scale-dependence.
        let scale_invariant = |a: f64, b: f64| (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0);
        assert!(
            scale_invariant(base.margin, big.margin),
            "margin must be scale-invariant: {} vs {}",
            base.margin,
            big.margin
        );
        assert!(
            scale_invariant(base.deviance_gain, big.deviance_gain),
            "deviance gain must be scale-invariant: {} vs {}",
            base.deviance_gain,
            big.deviance_gain
        );
        assert!(
            scale_invariant(base.ci_low, big.ci_low) && scale_invariant(base.ci_high, big.ci_high),
            "the deviance-scale CI must be scale-invariant"
        );

        // Contrast: the OLD raw-SSE gain (linear_total − chart_total) scales by 100
        // under ×10 — exactly the scale-dependence the deviance currency removes.
        let old_gain_base = base.linear_loss - base.chart_loss;
        let old_gain_big = big.linear_loss - big.chart_loss;
        assert!(
            (old_gain_big - 100.0 * old_gain_base).abs() < 1e-3 * old_gain_big.abs().max(1.0),
            "raw SSE gain is scale-dependent (×100) — the currency the fix replaces"
        );
    }
}
