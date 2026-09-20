//! The estimated latent law the marginal-slope default anchors on (gam#2926).
//!
//! The anchoring equation `E_p[Φ(α + b·z) | a] = π(a)` has a unique solution on
//! every finite law `p`, and the closed-form Gaussian lowering is its `N(0, 1)`
//! case. So the default fit does not make the score look Gaussian; it estimates
//! the law the score has, on the score's own axis, and anchors on that:
//!
//! * [`conditional_law_evidence`] tests whether that law moves on the
//!   marginal-index span — robust Rao score tests of the first three conditional
//!   moments, the span the conditional gate has always used;
//! * where it does not, one finite law ([`build_empirical_law_on_own_axis`]);
//! * where it does, local finite laws by context
//!   ([`local_law_parts`]). The contexts are the covariates
//!   the marginal formula reads, so a saved model replays the law from the
//!   prediction table exactly as `LatentMeasureKind::LocalEmpirical` always has:
//!   scale the columns, mix the `top_k` nearest centres and a fixed share of the
//!   pooled law, combine their grids. Every training row's mixture comes from the
//!   same [`local_empirical_mixture_for_point`] prediction calls, so the law a
//!   row is fitted under and the law it is predicted under are one object.

use super::*;

/// Rows per grid node a context is sized for: a context compresses about
/// `LOCAL_LAW_ROWS_PER_NODE · grid_size` training rows into its grid.
const LOCAL_LAW_ROWS_PER_NODE: usize = 6;
/// Fewest contexts that are local at all.
const LOCAL_LAW_MIN_CONTEXTS: usize = 2;
/// Most contexts; keeps the k-means passes and the per-row mixtures linear in
/// `n` with a small constant.
const LOCAL_LAW_MAX_CONTEXTS: usize = 64;
/// Lloyd passes stop at the first pass that moves no row, or here.
const LOCAL_LAW_MAX_LLOYD_PASSES: usize = 50;
/// A context with fewer rows than this carries no law and is dissolved into its
/// neighbours.
const LOCAL_LAW_MIN_ROWS_PER_CONTEXT: usize = 3;
/// A row's law mixes its four nearest contexts, the count the removed fit-time
/// builder used (bd1c5ac5c5); it bounds a row's law at `4·grid_size` context
/// nodes plus the pooled floor whatever the covariate dimension.
const LOCAL_LAW_TOP_K: usize = 4;
/// The kernel bandwidth in the scaled covariates, one training standard
/// deviation per column: the value the removed fit-time builder used
/// (bd1c5ac5c5).
const LOCAL_LAW_BANDWIDTH: f64 = 1.0;
/// The fixed share of the pooled law in every row's mixture, in units of the
/// kernel's peak `K(0) = 1`: `law(x) = (Σ_c w_c(x)·F_c + ε·F_pooled)/(Σ_c w_c(x) + ε)`.
/// It keeps the normaliser positive where the `top_k + 1` nearest contexts tie
/// and every truncated weight is zero, so the law is continuous everywhere.
const LOCAL_LAW_POOLED_FLOOR: f64 = 1.0e-3;
/// Fixed k-means++ seed, so a fit is a deterministic function of its data.
const LOCAL_LAW_SEED: u64 = 0x2926_0CA1_1A77_0001;

/// The covariates a local law is estimated by context over.
pub(crate) struct LocalLawContext<'a> {
    /// `n × d`: the raw data columns the marginal formula reads, one row per
    /// training row, in `feature_cols` order.
    pub(crate) features: ArrayView2<'a, f64>,
    /// Their indices in the fit's data matrix. Prediction resolves them by
    /// training header, so they must be the fit's own column indices.
    pub(crate) feature_cols: Vec<usize>,
}

/// The data columns a marginal formula reads, sorted and deduplicated: the
/// covariates its marginal-index span is built from, and so the contexts a local
/// law is estimated over. `remap_feature_columns` visits every index-bearing
/// field of the spec; the map collects each column and checks it is a column of
/// the `data_columns`-wide table the context covariates are read from.
pub(crate) fn marginal_formula_context_columns(
    spec: &TermCollectionSpec,
    data_columns: usize,
) -> Result<Vec<usize>, String> {
    let mut columns = Vec::new();
    spec.remap_feature_columns(|column| {
        if column >= data_columns {
            return Err(format!(
                "estimated local latent law: the marginal formula reads data column {column}, \
                 beyond the {data_columns} columns supplied"
            ));
        }
        columns.push(column);
        Ok(column)
    })?;
    columns.sort_unstable();
    columns.dedup();
    Ok(columns)
}

/// Robust Rao score tests of whether the conditional law of `z` moves on the
/// marginal-index span `a(C)`: the conditional mean `z − z̄`, the conditional
/// variance `(z − z̄)² − σ̂²`, and the third standardised moment
/// `((z − z̄)/σ̂)³ − μ̂₃`, each against the weighted-centred span.
///
/// Centring the columns makes each test about structure beyond the global
/// level; a constant column collapses and is dropped by the pseudo-inverse rank.
pub(crate) fn conditional_law_evidence(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    conditioning: Option<ArrayView2<'_, f64>>,
) -> Result<ConditionalLawEvidence, String> {
    let untestable = ConditionalLawEvidence {
        mean_p_value: None,
        variance_p_value: None,
        skewness_p_value: None,
        alpha: AUTO_Z_CONDITIONAL_RAO_ALPHA,
    };
    let Some(a_block) = conditioning else {
        return Ok(untestable);
    };
    let n = z.len();
    if weights.len() != n || a_block.nrows() != n {
        return Err(format!(
            "conditional latent-law test length mismatch: z={n}, weights={}, span rows={}",
            weights.len(),
            a_block.nrows()
        ));
    }
    if a_block.ncols() == 0 {
        return Ok(untestable);
    }
    if z.iter().any(|v| !v.is_finite()) || a_block.iter().any(|v| !v.is_finite()) {
        return Err(
            "conditional latent-law test requires a finite score and a finite marginal-index span"
                .to_string(),
        );
    }
    let total_weight = weights.iter().copied().sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Err(
            "conditional latent-law test requires positive finite total weight".to_string(),
        );
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / total_weight;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / total_weight;
    if !(var.is_finite() && var > 0.0) {
        // A constant score has no conditional law to move; whatever consumes it
        // refuses it for what it is.
        return Ok(untestable);
    }
    let sd = var.sqrt();
    let mut a_centered = a_block.to_owned();
    for j in 0..a_block.ncols() {
        let col_mean = a_block
            .column(j)
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| w * v)
            .sum::<f64>()
            / total_weight;
        a_centered.column_mut(j).mapv_inplace(|v| v - col_mean);
    }
    let centred: Vec<f64> = z.iter().map(|&zi| zi - mean).collect();
    let squared: Vec<f64> = centred.iter().map(|&e| e * e - var).collect();
    let third_moment = centred
        .iter()
        .zip(weights.iter())
        .map(|(&e, &w)| w * (e / sd).powi(3))
        .sum::<f64>()
        / total_weight;
    let cubed: Vec<f64> = centred
        .iter()
        .map(|&e| (e / sd).powi(3) - third_moment)
        .collect();
    Ok(ConditionalLawEvidence {
        mean_p_value: robust_conditional_score_pvalue(a_centered.view(), &centred, weights.view())?,
        variance_p_value: robust_conditional_score_pvalue(
            a_centered.view(),
            &squared,
            weights.view(),
        )?,
        skewness_p_value: robust_conditional_score_pvalue(
            a_centered.view(),
            &cubed,
            weights.view(),
        )?,
        alpha: AUTO_Z_CONDITIONAL_RAO_ALPHA,
    })
}

/// Equal-mass compression of `(z, weights)` into an at-most-`grid_size`-node
/// law ON THE SCORE'S OWN AXIS.
///
/// The shared builder returns its nodes standardised, which is the right object
/// for a residual `ζ` that is unit-variance by construction and the wrong one
/// for a score whose law is the thing being anchored on: a shifted or rescaled
/// law would reach the kernel recentred. The standardised nodes are mapped back
/// through the sample's weighted mean and standard deviation, so the law keeps
/// the score's location and scale, and the compression's within-bin variance
/// loss is repaired rather than inherited.
pub(crate) fn build_empirical_law_on_own_axis(
    z: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    grid_size: usize,
    context: &str,
) -> Result<EmpiricalZGrid, String> {
    let build = empirical_measure_sensitivity::build_empirical_z_grid_with_alpha(
        z, weights, grid_size, context,
    )?;
    if build.standardization_sd.is_none() {
        return Err(format!(
            "{context}: the score is constant on these rows, so there is no law to anchor on"
        ));
    }
    let total_weight = weights.iter().copied().sum::<f64>();
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / total_weight;
    let sd = (z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / total_weight)
        .sqrt();
    let nodes = build.grid.nodes.iter().map(|&u| mean + sd * u).collect();
    EmpiricalZGrid::new(nodes, build.grid.weights, context)
}

/// The Gaussian closed form's certificate anchor at one survival anchor under a
/// finite law (gam#2926): the residual `r = Σ_k w_k Φ(−(α_cf + b·u_k)) − Φ(−q)` at
/// `α_cf = q·√(1+b²)`, through [`survival_certificate_anchor`].
pub(crate) fn closed_form_survival_certificate_anchor(
    q: f64,
    observed_slope: f64,
    law: &EmpiricalZGrid,
) -> Result<super::CertificateAnchor, String> {
    let alpha = q * (1.0 + observed_slope * observed_slope).sqrt();
    let probabilities: Vec<f64> = law
        .nodes
        .iter()
        .map(|&u| survival_tail_probability(q, alpha + observed_slope * u))
        .collect();
    survival_certificate_anchor(q, &law.weights, &probabilities)
}

/// The moving-law certificate's `(ln S, ln(1 − S))` of one rigid survival anchor
/// under `law` (gam#2926): `S = Σ_k w_k Φ(−(α + b·u_k))` at the anchor's intercept `α`,
/// the one the fit solved on its own law, through
/// [`moving_law_rule::log_grid_anchor_probabilities`].
pub(crate) fn survival_anchor_log_probabilities(
    alpha: f64,
    observed_slope: f64,
    law: &EmpiricalZGrid,
) -> Result<(f64, f64), moving_law_rule::AnchorProbabilityFailure> {
    moving_law_rule::log_grid_anchor_probabilities(law, |u| Ok(-(alpha + observed_slope * u)))
}

/// A survival anchor's probability at de-nested index `η` on the smaller tail of
/// the anchor at marginal index `q`: `Φ(−η)` when `q ≥ 0`, `Φ(η)` otherwise.
pub(crate) fn survival_tail_probability(q: f64, eta: f64) -> f64 {
    if q < 0.0 { normal_cdf(eta) } else { normal_cdf(-eta) }
}

/// One survival anchor of the closed-form certificate from a finite law's node
/// weights and per-node tail probabilities ([`survival_tail_probability`]):
/// `r = Σ_k w_k Φ(−η_k) − Φ(−q)` and `π(1−π) = Φ(q)Φ(−q)`. The sums run on the
/// smaller tail, so a survival probability near one keeps its precision: with
/// `q < 0` the upper tail is summed, and because the weights sum to one,
/// `Σ_k w_k Φ(−η_k) − Φ(−q) = −(Σ_k w_k Φ(η_k) − Φ(q))`, the upper tail read with
/// the opposite sign.
pub(crate) fn survival_certificate_anchor(
    q: f64,
    weights: &[f64],
    probabilities: &[f64],
) -> Result<super::CertificateAnchor, String> {
    let (target, sign) = if q < 0.0 {
        (normal_cdf(q), -1.0)
    } else {
        (normal_cdf(-q), 1.0)
    };
    super::CertificateAnchor::on_law(
        weights,
        probabilities,
        target,
        sign,
        normal_cdf(q) * normal_cdf(-q),
    )
    .map_err(|reason| format!("survival anchor at q={q}: {reason}"))
}

fn squared_distance(point: ArrayView1<'_, f64>, center: &[f64]) -> f64 {
    point
        .iter()
        .zip(center.iter())
        .map(|(&x, &c)| (x - c) * (x - c))
        .sum()
}

fn nearest_center(point: ArrayView1<'_, f64>, centers: &[Vec<f64>]) -> usize {
    let mut best = 0;
    let mut best_d2 = f64::INFINITY;
    for (idx, center) in centers.iter().enumerate() {
        let d2 = squared_distance(point, center);
        if d2 < best_d2 {
            best = idx;
            best_d2 = d2;
        }
    }
    best
}

/// Weighted k-means++ seeding over the positive-weight rows, from a fixed seed.
fn seed_centers(
    points: &Array2<f64>,
    weights: &Array1<f64>,
    active: &[usize],
    contexts: usize,
) -> Vec<Vec<f64>> {
    let mut state = LOCAL_LAW_SEED;
    let mut unit = || (gam_linalg::utils::splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
    fn pick(mass: &[f64], target: f64) -> usize {
        let mut cumulative = 0.0;
        for (idx, &m) in mass.iter().enumerate() {
            cumulative += m;
            if cumulative > target {
                return idx;
            }
        }
        mass.len() - 1
    }
    let first_mass: Vec<f64> = active.iter().map(|&row| weights[row]).collect();
    let first_total = first_mass.iter().sum::<f64>();
    let first = pick(&first_mass, unit() * first_total);
    let mut centers = vec![points.row(active[first]).to_vec()];
    let mut nearest_d2: Vec<f64> = active
        .iter()
        .map(|&row| squared_distance(points.row(row), &centers[0]))
        .collect();
    while centers.len() < contexts {
        let mass: Vec<f64> = active
            .iter()
            .zip(nearest_d2.iter())
            .map(|(&row, &d2)| weights[row] * d2)
            .collect();
        let total = mass.iter().sum::<f64>();
        if !(total.is_finite() && total > 0.0) {
            // Every remaining row coincides with a centre: there are no more
            // distinct contexts to seed.
            break;
        }
        let chosen = pick(&mass, unit() * total);
        let center = points.row(active[chosen]).to_vec();
        for (slot, &row) in nearest_d2.iter_mut().zip(active.iter()) {
            *slot = slot.min(squared_distance(points.row(row), &center));
        }
        centers.push(center);
    }
    centers
}

/// Weighted centroid of each context under `assignment`; a context with no mass
/// keeps its previous centre.
fn recompute_centers(
    points: &Array2<f64>,
    weights: &Array1<f64>,
    active: &[usize],
    assignment: &[usize],
    centers: &mut [Vec<f64>],
) {
    let dim = points.ncols();
    let mut sums = vec![vec![0.0; dim]; centers.len()];
    let mut mass = vec![0.0; centers.len()];
    for &row in active {
        let context = assignment[row];
        let w = weights[row];
        mass[context] += w;
        for (slot, &x) in sums[context].iter_mut().zip(points.row(row).iter()) {
            *slot += w * x;
        }
    }
    for (context, center) in centers.iter_mut().enumerate() {
        if mass[context] > 0.0 {
            for (slot, &sum) in center.iter_mut().zip(sums[context].iter()) {
                *slot = sum / mass[context];
            }
        }
    }
}

/// A local law before any row's mixture (gam#2926): the kept context columns
/// and their scales, every row's scaled point, and the contexts' centres and
/// laws with the pooled floor last. [`Self::into_kind`] mixes every training row;
/// the moving-law certificate mixes only the rows a held-out law scores.
pub(crate) struct LocalLawParts {
    feature_cols: Vec<usize>,
    input_scales: Vec<f64>,
    points: Array2<f64>,
    centers: Vec<Vec<f64>>,
    grids: Vec<EmpiricalZGrid>,
    top_k: usize,
    bandwidth: f64,
    mixture: LocalLawMixture,
}

impl LocalLawParts {
    /// The number of contexts, after any too-small context was dissolved.
    pub(crate) fn contexts(&self) -> usize {
        self.centers.len()
    }

    /// The `(grid, weight)` pairs of row `row`'s law: its nearest centres and the
    /// pooled floor, from the one function prediction uses.
    pub(crate) fn row_mixture(&self, row: usize) -> Result<Vec<(usize, f64)>, String> {
        let point = self.points.row(row).to_vec();
        local_empirical_mixture_for_point(
            &point,
            &self.centers,
            self.top_k,
            self.bandwidth,
            self.mixture,
        )
    }

    /// The law, keeping only the mixtures of `rows`, in that order: a held-out law
    /// of the moving-law certificate is read at its own fold's rows alone, so it
    /// stores `(top_k + 1)` pairs per such row and nothing for the rest.
    pub(crate) fn held_out(self, rows: &[usize]) -> Result<HeldOutLocalLaw, String> {
        let mixtures = rows
            .par_iter()
            .map(|&row| self.row_mixture(row))
            .collect::<Result<Vec<_>, String>>()?;
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let mut entries = Vec::with_capacity(mixtures.iter().map(Vec::len).sum());
        for mixture in mixtures {
            entries.extend(mixture);
            offsets.push(entries.len());
        }
        Ok(HeldOutLocalLaw {
            grids: self.grids,
            offsets,
            entries,
        })
    }

    /// The local latent measure, with the mixture of every row of the table the
    /// law was built over: the rows it was estimated from and the zero-weight
    /// rows alike, since a row's mixture reads only its covariates. A held-out
    /// law of the moving-law certificate keeps its fold's rows alone
    /// ([`Self::held_out`]); the full-data law is the one copy with all `n`.
    pub(crate) fn into_kind(self) -> Result<LatentMeasureKind, String> {
        let train_row_mixtures = (0..self.points.nrows())
            .into_par_iter()
            .map(|row| self.row_mixture(row))
            .collect::<Result<Vec<_>, String>>()?;
        let kind = LatentMeasureKind::LocalEmpirical {
            feature_cols: self.feature_cols,
            input_scales: Some(self.input_scales),
            centers: self.centers,
            grids: self.grids,
            top_k: self.top_k,
            bandwidth: self.bandwidth,
            mixture: self.mixture,
            train_row_mixtures: Arc::new(train_row_mixtures),
        };
        kind.validate("estimated local latent law")?;
        Ok(kind)
    }
}

/// A local law's context laws and the mixtures of the rows it is read at
/// ([`LocalLawParts::held_out`]).
pub(crate) struct HeldOutLocalLaw {
    grids: Vec<EmpiricalZGrid>,
    offsets: Vec<usize>,
    entries: Vec<(usize, f64)>,
}

impl HeldOutLocalLaw {
    /// The law of the `position`-th of the rows the law was kept for.
    pub(crate) fn grid(&self, position: usize) -> Result<EmpiricalZGrid, String> {
        let next = position.checked_add(1).and_then(|next| self.offsets.get(next));
        let (start, end) = match (self.offsets.get(position), next) {
            (Some(&start), Some(&end)) => (start, end),
            _ => {
                return Err(format!(
                    "held-out local latent law has no row at position {position} of {}",
                    self.offsets.len().saturating_sub(1)
                ));
            }
        };
        combine_empirical_grids(&self.grids, &self.entries[start..end])
    }
}

/// Estimate local finite laws of `z` by context over `context.features`; the
/// per-row mixtures follow from [`LocalLawParts::into_kind`] (every row) or
/// [`LocalLawParts::held_out`] (the rows a held-out law is read at).
///
/// The columns are scaled by their weighted standard deviation (no centring:
/// prediction divides by the persisted scale and nothing else), constant columns
/// are dropped, and the positive-weight rows are partitioned into weighted
/// k-means contexts. With `contexts = None` there are about
/// `LOCAL_LAW_ROWS_PER_NODE · grid_size` rows per context, the default's sizing;
/// `Some(k)` seeds exactly `k`, the configuration a held-out law of the moving-law
/// certificate reuses from its full-data law. Each context's grid is the
/// equal-mass law of its own rows on the score's own axis, and the pooled law of
/// every row follows the context grids. A row's law mixes its four nearest
/// centres, with Gaussian-kernel weights of bandwidth `LOCAL_LAW_BANDWIDTH` that
/// vanish where a centre leaves the four, and the pooled law at the fixed share
/// `LOCAL_LAW_POOLED_FLOOR`. Only the (context, weight) pairs are kept per row.
pub(crate) fn local_law_parts(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &LocalLawContext<'_>,
    grid_size: usize,
    contexts: Option<usize>,
) -> Result<LocalLawParts, String> {
    let requested_contexts = contexts;
    let n = z.len();
    let features = context.features;
    if weights.len() != n || features.nrows() != n {
        return Err(format!(
            "estimated local latent law length mismatch: z={n}, weights={}, context rows={}",
            weights.len(),
            features.nrows()
        ));
    }
    if features.ncols() != context.feature_cols.len() {
        return Err(format!(
            "estimated local latent law context has {} columns but names {} feature columns",
            features.ncols(),
            context.feature_cols.len()
        ));
    }
    if features.iter().any(|v| !v.is_finite()) {
        return Err(
            "estimated local latent law requires finite context covariates".to_string(),
        );
    }
    if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
        return Err(
            "estimated local latent law requires finite non-negative weights".to_string(),
        );
    }
    let active: Vec<usize> = (0..n).filter(|&row| weights[row] > 0.0).collect();
    let total_weight = active.iter().map(|&row| weights[row]).sum::<f64>();
    let weight_sq = active.iter().map(|&row| weights[row] * weights[row]).sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0 && weight_sq > 0.0) {
        return Err(
            "estimated local latent law requires positive finite total weight".to_string(),
        );
    }

    let mut kept = Vec::new();
    let mut scales = Vec::new();
    for j in 0..features.ncols() {
        let column = features.column(j);
        let mean = active.iter().map(|&row| weights[row] * column[row]).sum::<f64>() / total_weight;
        let var = active
            .iter()
            .map(|&row| weights[row] * (column[row] - mean) * (column[row] - mean))
            .sum::<f64>()
            / total_weight;
        let magnitude = active
            .iter()
            .fold(0.0_f64, |acc, &row| acc.max(column[row].abs()));
        let sd = var.sqrt();
        if sd.is_finite() && sd > gam_linalg::roundoff::accumulation_growth(n + 1) * magnitude {
            kept.push(j);
            scales.push(sd);
        }
    }
    if kept.is_empty() {
        return Err(
            "estimated local latent law: every covariate the marginal formula reads is constant \
             on the training rows, so there is no context to estimate a local law over"
                .to_string(),
        );
    }
    let dim = kept.len();
    let mut points = Array2::<f64>::zeros((n, dim));
    for (local, (&j, &scale)) in kept.iter().zip(scales.iter()).enumerate() {
        points
            .column_mut(local)
            .assign(&features.column(j).mapv(|value| value / scale));
    }

    let effective_n = total_weight * total_weight / weight_sq;
    let sized = (effective_n / (LOCAL_LAW_ROWS_PER_NODE * grid_size) as f64).floor() as usize;
    let contexts = requested_contexts
        .unwrap_or_else(|| sized.clamp(LOCAL_LAW_MIN_CONTEXTS, LOCAL_LAW_MAX_CONTEXTS))
        .min(active.len() / LOCAL_LAW_MIN_ROWS_PER_CONTEXT);
    if contexts < LOCAL_LAW_MIN_CONTEXTS {
        return Err(format!(
            "estimated local latent law: {} positive-weight rows cannot fill {LOCAL_LAW_MIN_CONTEXTS} \
             contexts of at least {LOCAL_LAW_MIN_ROWS_PER_CONTEXT} rows",
            active.len()
        ));
    }

    let mut centers = seed_centers(&points, weights, &active, contexts);
    let mut assignment = vec![usize::MAX; n];
    for _ in 0..LOCAL_LAW_MAX_LLOYD_PASSES {
        let next: Vec<usize> = active
            .par_iter()
            .map(|&row| nearest_center(points.row(row), &centers))
            .collect();
        let mut moved = false;
        for (&context_idx, &row) in next.iter().zip(active.iter()) {
            moved |= assignment[row] != context_idx;
            assignment[row] = context_idx;
        }
        recompute_centers(&points, weights, &active, &assignment, &mut centers);
        if !moved {
            break;
        }
    }

    // Dissolve contexts too small to carry a law. A row of a surviving context
    // was already nearest its own centre, so reassignment only moves the rows of
    // dissolved contexts, and every survivor keeps at least its own rows.
    let mut counts = vec![0usize; centers.len()];
    for &row in &active {
        counts[assignment[row]] += 1;
    }
    let survivors: Vec<Vec<f64>> = centers
        .iter()
        .zip(counts.iter())
        .filter(|(_, count)| **count >= LOCAL_LAW_MIN_ROWS_PER_CONTEXT)
        .map(|(center, _)| center.clone())
        .collect();
    if survivors.len() < LOCAL_LAW_MIN_CONTEXTS {
        return Err(format!(
            "estimated local latent law: the training rows fill only {} context(s) of at least \
             {LOCAL_LAW_MIN_ROWS_PER_CONTEXT} rows",
            survivors.len()
        ));
    }
    let mut centers = survivors;
    let reassigned: Vec<usize> = active
        .par_iter()
        .map(|&row| nearest_center(points.row(row), &centers))
        .collect();
    for (&context_idx, &row) in reassigned.iter().zip(active.iter()) {
        assignment[row] = context_idx;
    }
    recompute_centers(&points, weights, &active, &assignment, &mut centers);

    let mut grids = Vec::with_capacity(centers.len() + 1);
    for context_idx in 0..centers.len() {
        let rows: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&row| assignment[row] == context_idx)
            .collect();
        let local_z = Array1::from_iter(rows.iter().map(|&row| z[row]));
        let local_w = Array1::from_iter(rows.iter().map(|&row| weights[row]));
        grids.push(build_empirical_law_on_own_axis(
            local_z.view(),
            local_w.view(),
            grid_size,
            &format!("estimated local latent law, context {context_idx}"),
        )?);
    }
    grids.push(build_empirical_law_on_own_axis(
        z.view(),
        weights.view(),
        grid_size,
        "estimated local latent law, pooled floor",
    )?);

    let top_k = centers.len().min(LOCAL_LAW_TOP_K);
    Ok(LocalLawParts {
        feature_cols: kept.iter().map(|&j| context.feature_cols[j]).collect(),
        input_scales: scales,
        points,
        centers,
        grids,
        top_k,
        bandwidth: LOCAL_LAW_BANDWIDTH,
        mixture: LocalLawMixture::VanishingAtTruncation {
            floor: LOCAL_LAW_POOLED_FLOOR,
        },
    })
}

/// The kernel mixture over the `top_k` nearest context centres of one scaled
/// covariate point, as `(grid index, weight)` pairs. Shared by the fit, which
/// computes every training row's mixture with it, and by prediction, which
/// computes a new row's.
///
/// Under [`LocalLawMixture::NearestNormalized`] the weights are kernel values
/// relative to the nearest centre (so the nearest is exactly 1 and the mixture
/// never underflows as a whole), renormalised over the `top_k`. Under
/// [`LocalLawMixture::VanishingAtTruncation`] a kept centre's weight is its
/// kernel value `K(d) = exp(−d²/2h²)` less the `(top_k + 1)`-th centre's, which
/// is zero exactly where the centre enters or leaves the top `top_k`, and the
/// pooled law — the grid after the context grids — enters at the fixed weight
/// `floor`, so the normaliser never vanishes and every weight is continuous in
/// the point.
pub(crate) fn local_empirical_mixture_for_point(
    point: &[f64],
    centers: &[Vec<f64>],
    top_k: usize,
    bandwidth: f64,
    mixture: LocalLawMixture,
) -> Result<Vec<(usize, f64)>, String> {
    if centers.is_empty() {
        return Err("local empirical latent law has no centers".to_string());
    }
    if top_k == 0 {
        return Err("local empirical latent law top_k must be positive".to_string());
    }
    if !(bandwidth.is_finite() && bandwidth > 0.0) {
        return Err(format!(
            "local empirical latent law bandwidth must be finite and positive, got {bandwidth}"
        ));
    }
    let bw2 = bandwidth * bandwidth;
    let mut distances = Vec::<(usize, f64)>::with_capacity(centers.len());
    for (idx, center) in centers.iter().enumerate() {
        if center.len() != point.len() {
            return Err(format!(
                "local empirical latent law center {idx} dimension mismatch: center={}, point={}",
                center.len(),
                point.len()
            ));
        }
        let d2 = center
            .iter()
            .zip(point.iter())
            .map(|(&c, &x)| {
                let delta = x - c;
                delta * delta
            })
            .sum::<f64>();
        if !d2.is_finite() {
            return Err("local empirical latent law distance is non-finite".to_string());
        }
        distances.push((idx, d2));
    }
    distances.sort_by(|left, right| left.1.total_cmp(&right.1));
    let k = top_k.min(distances.len());
    let mut weights = Vec::with_capacity(k + 1);
    let mut total = 0.0;
    match mixture {
        LocalLawMixture::NearestNormalized => {
            let d2_nearest = distances.first().map_or(0.0, |&(_, d2)| d2);
            for &(idx, d2) in distances.iter().take(k) {
                let weight = (-0.5 * (d2 - d2_nearest) / bw2).exp();
                weights.push((idx, weight));
                total += weight;
            }
        }
        LocalLawMixture::VanishingAtTruncation { floor } => {
            if !(floor.is_finite() && floor > 0.0) {
                return Err(format!(
                    "local empirical latent law pooled floor must be finite and positive, got {floor}"
                ));
            }
            let kernel = |d2: f64| (-0.5 * d2 / bw2).exp();
            let truncation = distances.get(k).map_or(0.0, |&(_, d2)| kernel(d2));
            for &(idx, d2) in distances.iter().take(k) {
                let weight = kernel(d2) - truncation;
                if weight > 0.0 {
                    weights.push((idx, weight));
                    total += weight;
                }
            }
            weights.push((centers.len(), floor));
            total += floor;
        }
    }
    if !(total.is_finite() && total > 0.0) {
        return Err(
            "local empirical latent law mixture has non-positive total weight".to_string(),
        );
    }
    for (_, weight) in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The anchor `α` with `Σ_k w_k Φ(α + b·u_k) = Φ(q)`, by bisection.
    fn anchor(grid: &EmpiricalZGrid, q: f64, slope: f64) -> f64 {
        let target = normal_cdf(q);
        let (mut low, mut high) = (-40.0_f64, 40.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            let value: f64 = grid.pairs().map(|(u, w)| w * normal_cdf(mid + slope * u)).sum();
            if value < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }

    /// Context centres, their grids followed by the pooled grid, and the mixture
    /// size.
    struct Layout {
        centers: Vec<Vec<f64>>,
        grids: Vec<EmpiricalZGrid>,
        top_k: usize,
    }

    /// `count` contexts with visibly different three-node laws, plus a pooled law.
    fn grids(count: usize) -> Vec<EmpiricalZGrid> {
        let mut out: Vec<EmpiricalZGrid> = (0..count)
            .map(|c| {
                let shift = 0.45 * c as f64;
                let tilt = 0.1 + 0.08 * c as f64;
                EmpiricalZGrid::new(
                    vec![-1.5 + shift, -0.2 + shift, 0.9 + 2.0 * shift],
                    vec![tilt, 0.6 - tilt, 0.4],
                    "context",
                )
                .expect("grid")
            })
            .collect();
        out.push(
            EmpiricalZGrid::new(vec![-1.0, 0.0, 1.0], vec![0.25, 0.5, 0.25], "pooled")
                .expect("grid"),
        );
        out
    }

    fn triangle() -> Layout {
        Layout {
            centers: vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            grids: grids(3),
            top_k: 2,
        }
    }

    fn pentagon() -> Layout {
        Layout {
            centers: (0..5)
                .map(|j| {
                    let angle = std::f64::consts::FRAC_PI_2 + std::f64::consts::TAU * j as f64 / 5.0;
                    vec![angle.cos(), angle.sin()]
                })
                .collect(),
            grids: grids(5),
            top_k: 4,
        }
    }

    /// `(α, p̂)` of a row at `point` under `rule`.
    fn row_anchor(layout: &Layout, point: &[f64], rule: LocalLawMixture) -> (f64, f64) {
        let (q, slope, z) = (-0.3, 1.2, 0.4);
        let weights =
            local_empirical_mixture_for_point(point, &layout.centers, layout.top_k, 0.5, rule)
                .expect("mixture");
        let law = combine_empirical_grids(&layout.grids, &weights).expect("row law");
        let alpha = anchor(&law, q, slope);
        (alpha, normal_cdf(alpha + slope * z))
    }

    /// The largest symmetric difference quotients of `α` and `p̂` at step `h`
    /// over 201 points of the segment `origin + s·direction`, `|s| ≤ 0.02`, whose
    /// middle point is `origin` itself.
    fn path_quotients(
        layout: &Layout,
        origin: [f64; 2],
        direction: [f64; 2],
        h: f64,
        rule: LocalLawMixture,
    ) -> (f64, f64) {
        let norm = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        let at = |s: f64| {
            [
                origin[0] + s * direction[0] / norm,
                origin[1] + s * direction[1] / norm,
            ]
        };
        let mut worst = (0.0_f64, 0.0_f64);
        for step in 0..=200 {
            let s = -0.02 + 0.0002 * step as f64;
            let (alpha_left, p_left) = row_anchor(layout, &at(s - 0.5 * h), rule);
            let (alpha_right, p_right) = row_anchor(layout, &at(s + 0.5 * h), rule);
            worst.0 = worst.0.max((alpha_right - alpha_left).abs() / h);
            worst.1 = worst.1.max((p_right - p_left).abs() / h);
        }
        worst
    }

    /// gam#2926: a row's declared law is a continuous function of its covariates,
    /// everywhere.
    ///
    /// Three paths, each through a point where the nearest-centre ranking is
    /// degenerate: on `(0.2, 0.2 + s)` the second- and third-nearest of three
    /// centres swap at `s = 0`, which is where a centre enters or leaves a
    /// `top_k = 2` mixture; the circumcentre `(0.5, 0.5)` is equidistant from all
    /// three, so every truncated weight there is zero; and the centre of a
    /// regular pentagon is equidistant from all five, the same tie for
    /// `top_k = 4`. Under the floored vanishing rule the largest difference
    /// quotient of `α` and `p̂` on each path must not grow as the step shrinks
    /// tenfold. Under the legacy rule it grows like `1/h` at the swap, which is
    /// what shows the paths do cross one.
    #[test]
    fn local_law_is_continuous_across_swaps_and_ties_2926() {
        let floored = LocalLawMixture::VanishingAtTruncation {
            floor: LOCAL_LAW_POOLED_FLOOR,
        };
        let cases = [
            ("rank 2/3 swap, top_k=2", triangle(), [0.2, 0.2], [0.0, 1.0]),
            ("three-way tie, top_k=2", triangle(), [0.5, 0.5], [1.0, 0.37]),
            ("five-way tie, top_k=4", pentagon(), [0.0, 0.0], [1.0, 0.0]),
        ];
        for (label, layout, origin, direction) in &cases {
            let (alpha_coarse, p_coarse) = path_quotients(layout, *origin, *direction, 1e-4, floored);
            let (alpha_fine, p_fine) = path_quotients(layout, *origin, *direction, 1e-5, floored);
            eprintln!(
                "[2926 continuity] {label}: max |Δα|/h h=1e-4 {alpha_coarse:.4} h=1e-5 \
                 {alpha_fine:.4} | max |Δp̂|/h h=1e-4 {p_coarse:.4} h=1e-5 {p_fine:.4}"
            );
            assert!(
                alpha_fine <= 1.1 * alpha_coarse + 1e-6 && p_fine <= 1.1 * p_coarse + 1e-6,
                "{label}: the floored rule's difference quotients must not grow as the step \
                 shrinks: |Δα|/h {alpha_coarse} -> {alpha_fine}, |Δp̂|/h {p_coarse} -> {p_fine}"
            );
        }
        let swap = triangle();
        let (legacy_coarse, _) = path_quotients(
            &swap,
            [0.2, 0.2],
            [0.0, 1.0],
            1e-4,
            LocalLawMixture::NearestNormalized,
        );
        let (legacy_fine, _) = path_quotients(
            &swap,
            [0.2, 0.2],
            [0.0, 1.0],
            1e-5,
            LocalLawMixture::NearestNormalized,
        );
        eprintln!(
            "[2926 continuity] legacy rule at the swap: max |Δα|/h h=1e-4 {legacy_coarse:.1} \
             h=1e-5 {legacy_fine:.1}"
        );
        assert!(
            legacy_fine > 5.0 * legacy_coarse,
            "control: the legacy rule must jump at the swap, or this path does not cross one; \
             |Δα|/h {legacy_coarse} -> {legacy_fine}"
        );
    }

    /// gam#2926: the equal-mass compression's nodes ascend exactly on a tie shared
    /// by several bins. A bin's rounded mean `Σ take·z / Σ take` can land either
    /// side of the tied value, and on the skewed survival fixture that put node 10
    /// one ulp below node 9, which the grid refuses.
    #[test]
    fn a_tie_shared_by_several_bins_gives_ascending_nodes_2926() {
        let tied = -0.709_314_748_167_235_8;
        let mut state = 0x2926_0071_E000_0001_u64;
        let rows = 4000;
        let z: Vec<f64> = (0..rows)
            .map(|row| match row {
                0..400 => -2.0 + 1e-3 * row as f64,
                400..3600 => tied,
                _ => 1.0 + 1e-3 * row as f64,
            })
            .collect();
        let weights: Vec<f64> = (0..rows)
            .map(|_| {
                0.25 + (gam_linalg::utils::splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let grid = build_empirical_law_on_own_axis(
            ArrayView1::from(z.as_slice()),
            ArrayView1::from(weights.as_slice()),
            65,
            "tie test",
        )
        .expect("a tied score has a law");
        for pair in grid.nodes.windows(2) {
            assert!(pair[0] <= pair[1], "nodes out of order: {} then {}", pair[0], pair[1]);
        }
    }

    /// gam#2926: a row's combined law carries every node once, strictly
    /// ascending, with the weights of the grids that share it summed, and a
    /// centre whose kernel weight underflowed to zero contributes nothing.
    #[test]
    fn combined_grids_coalesce_shared_nodes_2926() {
        let grids = [
            EmpiricalZGrid::new(vec![-1.0, 0.0, 2.0], vec![0.25, 0.5, 0.25], "first").expect("grid"),
            EmpiricalZGrid::new(vec![-1.0, 1.0, 2.0], vec![0.5, 0.25, 0.25], "second").expect("grid"),
            EmpiricalZGrid::new(vec![5.0, 6.0], vec![0.5, 0.5], "underflowed").expect("grid"),
        ];
        let law = combine_empirical_grids(&grids, &[(1, 0.25), (0, 0.75), (2, 0.0)])
            .expect("combined law");
        assert_eq!(law.nodes, vec![-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(law.weights, vec![0.3125, 0.375, 0.0625, 0.25]);
    }

    /// gam#2968's certificate on a fit-free cell: one Bernoulli closed-form anchor
    /// per row at marginal index `q_i` and slope `b_i` spread over the rows, read
    /// under the law estimated from `z`.
    fn closed_form_certificate_on(z: &[f64], second_order: bool) -> ClosedFormAnchorResidual {
        let weights = vec![1.0; z.len()];
        let law = build_empirical_law_on_own_axis(
            ArrayView1::from(z),
            ArrayView1::from(weights.as_slice()),
            super::super::DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
            "certificate cell",
        )
        .expect("estimated law");
        let rows = 400;
        let mut accumulator = ClosedFormAnchorAccumulator::new(&law.weights, second_order);
        for row in 0..rows {
            let t = row as f64 / (rows - 1) as f64;
            let q = -1.5 + 3.0 * t;
            let slope = 0.3 + 0.9 * ((7 * row) % rows) as f64 / rows as f64;
            let intercept = q * (1.0 + slope * slope).sqrt();
            let probabilities: Vec<f64> =
                law.nodes.iter().map(|&u| normal_cdf(intercept + slope * u)).collect();
            let mu = normal_cdf(q);
            let anchor =
                CertificateAnchor::on_law(&law.weights, &probabilities, mu, 1.0, mu * (1.0 - mu))
                    .expect("anchor");
            accumulator.add(&anchor, 1.0).expect("add");
        }
        let sampling =
            ScoreSampling::from_weights(ArrayView1::from(weights.as_slice())).expect("sampling");
        accumulator.finish(sampling).expect("certificate")
    }

    /// A standardised gamma score of skewness `skew` (standard normal at 0).
    fn standardised_scores(rng: &mut rand::rngs::StdRng, n: usize, skew: f64) -> Vec<f64> {
        use rand::RngExt as _;
        if skew == 0.0 {
            return (0..n).map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal)).collect();
        }
        let shape = 4.0 / (skew * skew);
        let gamma = rand_distr::Gamma::new(shape, 1.0).expect("gamma");
        (0..n).map(|_| (rng.sample(gamma) - shape) / shape.sqrt()).collect()
    }

    /// gam#2968: `SE(D̂)` is the sampling error of `D̂` over the score sample the
    /// law is estimated from. On a skewed score, where `D̂` is a clear bias and not
    /// noise, it matches the score bootstrap that rebuilds the law per resample.
    #[test]
    fn closed_form_certificate_standard_error_matches_the_score_bootstrap_2968() {
        use rand::{RngExt as _, SeedableRng as _};
        let mut rng = rand::rngs::StdRng::seed_from_u64(2968);
        let n = 3000;
        let z = standardised_scores(&mut rng, n, 1.0);
        let certificate = closed_form_certificate_on(&z, true);
        let standard_error = certificate.standard_error.expect("second-order certificate");
        assert!(
            certificate.excess_kl > 3.0 * standard_error,
            "fixture invariant: the skewed cell must carry a visible bias, {}",
            certificate.summary()
        );
        assert!(
            closed_form_certificate_on(&z, false).standard_error.is_none(),
            "a first-order fold measures no standard error"
        );
        let resamples = 400;
        let mut draws = Vec::with_capacity(resamples);
        let mut resample = vec![0.0; n];
        for _ in 0..resamples {
            for value in resample.iter_mut() {
                *value = z[rng.random_range(0..n)];
            }
            draws.push(closed_form_certificate_on(&resample, false).excess_kl);
        }
        let mean = draws.iter().sum::<f64>() / resamples as f64;
        let bootstrap_sd = (draws.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>()
            / (resamples - 1) as f64)
            .sqrt();
        let ratio = standard_error / bootstrap_sd;
        // The ratio's Monte-Carlo error is about 1/√(2·400) = 0.035.
        assert!(
            (0.85..=1.15).contains(&ratio),
            "SE(D̂) = {standard_error:.4e} against the bootstrap SD {bootstrap_sd:.4e}: ratio \
             {ratio:.3} ({})",
            certificate.summary()
        );
    }

    /// gam#2968: at the null the refusal holds its level. On exactly Gaussian
    /// scores, a fresh sample per replicate and the law rebuilt from it, no
    /// replicate's `D̂` passes `z_{1−α}·SE(D̂)`; on a materially skewed score at a
    /// size where the declaration's bias dominates the law's noise, every one does.
    #[test]
    fn declared_gaussian_refusal_holds_its_level_and_refuses_material_skew_2968() {
        use rand::SeedableRng as _;
        let mut rng = rand::rngs::StdRng::seed_from_u64(29680);
        let mut largest = f64::NEG_INFINITY;
        for n in [1000, 30_000] {
            for _ in 0..if n == 1000 { 400 } else { 40 } {
                let z = standardised_scores(&mut rng, n, 0.0);
                let certificate = closed_form_certificate_on(&z, true);
                let standard_error = certificate.standard_error.expect("standard error");
                largest = largest.max(certificate.excess_kl / standard_error);
                assert_eq!(
                    certificate.refuses_declaration().expect("decision"),
                    None,
                    "an exactly Gaussian score at n = {n} was refused: {}",
                    certificate.summary()
                );
            }
        }
        for _ in 0..10 {
            let z = standardised_scores(&mut rng, 30_000, 1.0);
            let certificate = closed_form_certificate_on(&z, true);
            assert!(
                certificate.refuses_declaration().expect("decision").is_some(),
                "a skew-1 score at n = 30000 was kept: {}",
                certificate.summary()
            );
        }
        eprintln!("largest D̂/SE on exact Gaussian scores: {largest:.3}");
    }
}
