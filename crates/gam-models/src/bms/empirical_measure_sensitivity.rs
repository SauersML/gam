//! Sensitivity of the global-empirical latent measure to the calibrated score
//! it is built from — the `D = ∂node/∂ζ` half of the gam#2484 Murphy–Topel
//! generalization.
//!
//! # Why this exists
//!
//! On the BMS conditional location-scale branch the second-stage latent measure
//! is **built from the first-stage output**: `ζ = (z − m̂(C))/√v̂(C)` is
//! compressed into an equal-mass grid by
//! [`build_empirical_z_grid_with_alpha`], and the row kernel then integrates
//! against that grid. So a row's log-likelihood depends on `ζ` twice — directly
//! through its own `ζ_i`, and through a grid that every other `ζ_j` helped
//! build. The Murphy–Topel chain
//!
//! ```text
//!   G = ∂(score_β)/∂θ₁ = Σ_j (d score_β / d ζ_j) · (∂ζ_j/∂θ₁)
//! ```
//!
//! therefore needs the TOTAL `d score_β / d ζ_j`, not the per-row mixed partial
//! the standard-normal kernel supplies. The cross-row half factors as
//! `Σ_b u_b · D_{bj}` with `u_b = ∂(score_β)/∂node_b` (the row side, from
//! [`rigid_empirical_score_zeta_channels`]) and `D = ∂node/∂ζ` (the measure
//! side, from [`EmpiricalZGridBuild::node_zeta_vjp`]), which is what makes the
//! correction assemble as one modified sensitivity matrix
//! `S_eff = S + (U_Q·D)ᵀ` and leaves the whole downstream congruence untouched.
//!
//! # Why `D` is a closed form and not a differentiated sort
//!
//! [`build_empirical_z_grid_with_alpha`] sorts the positive-weight rows by `ζ`,
//! then walks bins of equal **weight**. Bin boundaries are therefore set by
//! cumulative weight and are exactly independent of the `ζ` values; for a fixed
//! sort order the allocation matrix `α` (how much of row `i`'s weight landed in
//! bin `c`) is *exactly constant* in `ζ`, and the grid weights carry no
//! `ζ`-sensitivity at all. Only the node VALUES move:
//!
//! ```text
//!   raw node   n_c = Σ_i α_{ci} ζ_i / W_c ,   W_c = Σ_i α_{ci}
//!   standardized  x_b = (n_b − μ)/sd ,  μ = Σ_c w_c n_c , sd² = Σ_c w_c (n_c − μ)²
//!
//!   ∂n_c/∂ζ_i = α_{ci}/W_c                        =: A_{ci}
//!   ∂x_b/∂n_c = (1/sd)·[(δ_{bc} − w_c) − x_b·w_c·x_c] =: (1/sd)·M_{bc}
//!   D = (1/sd)·M·A
//! ```
//!
//! (`∂μ/∂n_c = w_c` and, using `Σ_c w_c (n_c − μ) = 0`, `∂sd/∂n_c = w_c·x_c`.)
//! On the branch where the spread sits inside its rounding band the standardization is skipped, so
//! `M = I` and the `1/sd` factor is absent.
//!
//! `M` is a PROJECTION, and that bounds the whole channel: `M·1 = 0` and
//! `M·x = 0`, because the standardized grid carries weighted mean 0 and
//! weighted sd 1 by construction. Shifting every `ζ_i` by the same amount, or
//! rescaling them all, moves every raw node and no standardized one. So the
//! cross-row channel only ever transmits the SHAPE of a node perturbation —
//! which is why it is materially smaller in the published standard error than
//! its own matrix norm suggests, and why what it scales with is the slope
//! (the thing that lets a row see the latent axis at all) rather than the grid
//! size. Both identities are gated in `empirical_measure_2484_tests`, and the
//! standard-error sweep behind the claim is
//! `scripts/probe_2484_empirical_measure_sensitivity.py`.
//!
//! The one thing that is NOT differentiable is the sort order, and it fails in
//! exactly one place: a tied group of rows whose cumulative-mass span a bin
//! boundary cuts. There the left and right derivatives genuinely differ, so the
//! build records a typed refusal rather than a number — see
//! [`EmpiricalGridTieStraddle`].
//!
//! # What is recorded, and where it lives
//!
//! `α`, `W`, and the standardization `sd` are **fit-time provenance**, not part
//! of the measure. [`EmpiricalZGrid`] is on the persistence wire (it is inside
//! `LatentMeasureKind`, which `predict_io` reads back), and its `PartialEq` is
//! the measure's identity — two grids equal by nodes and weights must not
//! compare unequal because they were built from different rows. So the record
//! rides on a separate build product, [`EmpiricalZGridBuild`], returned only by
//! [`build_empirical_z_grid_with_alpha`]; [`build_empirical_z_grid`] stays a thin
//! wrapper over it and every existing caller and the wire are untouched.

use super::{
    EmpiricalZGrid, LatentMeasureKind,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A tied `ζ` group whose mass a bin boundary cuts, which is the only way the
/// equal-mass compression stops being differentiable in `ζ`.
///
/// Rows that share a `ζ` value have no defined order, and the fill loop hands
/// whichever comes first the earlier bin. When the whole tied group lands in one
/// bin that is harmless — every member contributes its full weight to that bin
/// regardless of the order, so `α` is invariant to the permutation. When a
/// boundary cuts the group, the split depends on the order, and moving one
/// member's `ζ` up through the tie swaps the allocation: the left and right
/// derivatives of `n_c` differ and `D` does not exist.
///
/// This is a property of the DATA, computable from the sorted `(ζ, w)` pairs
/// alone, and it is a yes/no rather than a tolerance.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EmpiricalGridTieStraddle {
    /// The tied `ζ` value.
    pub(crate) value: f64,
    /// How many positive-weight rows share it.
    pub(crate) rows: usize,
    /// Index of the first bin boundary strictly inside the group's mass span.
    pub(crate) boundary: usize,
}

/// The equal-mass grid plus the fit-time provenance the Murphy–Topel correction
/// needs to differentiate it in `ζ`.
///
/// Fit-time only, by construction: nothing here is on the persistence wire and
/// nothing here is needed to APPLY the measure, only to differentiate it.
#[derive(Clone, Debug)]
pub(crate) struct EmpiricalZGridBuild {
    /// The measure itself. There is exactly one builder, so the recorded
    /// allocation below is the one the fit's own compression performed rather
    /// than a reconstruction of it.
    pub(crate) grid: EmpiricalZGrid,
    /// The allocation `α` as `(emitted node index, ORIGINAL row index, mass)`.
    ///
    /// Indexed by EMITTED node, not by loop iteration: a bin whose weight came
    /// out at zero is never emitted, so `nodes.len()` can be below the requested
    /// `m` and the two indices are not the same counter.
    ///
    /// The original row index can only be RECORDED, never recovered: the builder
    /// filters zero-weight rows out and sorts what is left, so the row identity
    /// is destroyed before the fill loop runs.
    ///
    /// There is no fixed bound on entries per row. A row whose weight exceeds
    /// the per-bin target fills several consecutive bins and appears once per
    /// bin it touches; `nnz(α) ≤ n + m − 1`.
    pub(crate) alpha: Vec<(usize, usize, f64)>,
    /// `W_c = Σ_i α_{ci}`, the UNNORMALIZED mass of each emitted bin. Length
    /// equals `grid.nodes.len()`.
    pub(crate) bin_mass: Vec<f64>,
    /// The `sd` the standardization divided by, or `None` when
    /// [`recenter_rescale_empirical_grid`] skipped it (degenerate spread). The
    /// stored nodes have weighted sd 1 by construction, so this factor is not
    /// recoverable from the grid.
    pub(crate) standardization_sd: Option<f64>,
    /// Number of rows in the `ζ` vector the grid was built from — the width of
    /// `D`, and the length of every vector [`Self::node_zeta_vjp`] returns.
    /// Includes the zero-weight rows the builder filtered out; they simply have
    /// an all-zero column.
    pub(crate) n_rows: usize,
    /// `Some` when the equal-mass compression is not differentiable in `ζ`. The
    /// grid and the point estimates are unaffected; only `D` is refused.
    pub(crate) tie_straddle: Option<EmpiricalGridTieStraddle>,
}

impl EmpiricalZGridBuild {
    /// `Dᵀ·V` for an `m × p` right-hand side `V`, i.e. the `n × p` matrix whose
    /// row `i` is `Σ_b V_{b·}·D_{bi}`.
    ///
    /// This is the direction the Murphy–Topel seam needs: `U_Q` is `p_β × m`, and
    /// the cross-row contribution to the per-row sensitivity matrix is
    /// `(U_Q·D)ᵀ = Dᵀ·U_Qᵀ`.
    ///
    /// Evaluated as `Aᵀ·(Mᵀ·V)/sd`. `Mᵀ` never has to be formed: two `m`-length
    /// reductions give
    /// `(Mᵀv)_c = (1/sd)·[v_c − w_c·Σ_b v_b − w_c·x_c·Σ_b x_b·v_b]` in `O(m·p)`,
    /// after which `Aᵀ` is a sparse scatter through `α` in `O(nnz(α)·p)`. The
    /// whole apply is therefore linear in the data, not quadratic in the grid.
    ///
    /// Returns `Err` when the build recorded a tie straddle: there is no
    /// derivative to apply.
    pub(crate) fn node_zeta_vjp(&self, v: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if let Some(tie) = self.tie_straddle.as_ref() {
            return Err(format!(
                "the empirical latent measure is not differentiable in the calibrated score: {} \
                 rows are tied at ζ = {:.6} and equal-mass bin boundary {} cuts the tied group, so \
                 the allocation depends on an order the data does not define and the left and \
                 right derivatives of the grid nodes differ",
                tie.rows, tie.value, tie.boundary
            ));
        }
        let m = self.grid.nodes.len();
        if v.nrows() != m {
            return Err(format!(
                "empirical grid node-sensitivity VJP expects {m} node rows, got {}",
                v.nrows()
            ));
        }
        let p = v.ncols();
        // Mᵀ·V. With M_{bc} = (δ_{bc} − w_c) − x_b·w_c·x_c,
        //   (Mᵀ V)_{c·} = V_{c·} − w_c·Σ_b V_{b·} − w_c·x_c·Σ_b x_b·V_{b·}.
        // On the skipped-standardization branch M = I and this is the identity.
        let mut mt_v = v.to_owned();
        if let Some(sd) = self.standardization_sd {
            let mut sum_v = vec![0.0_f64; p];
            let mut sum_xv = vec![0.0_f64; p];
            for b in 0..m {
                let x_b = self.grid.nodes[b];
                for j in 0..p {
                    sum_v[j] += v[[b, j]];
                    sum_xv[j] += x_b * v[[b, j]];
                }
            }
            let inv_sd = 1.0 / sd;
            for c in 0..m {
                let w_c = self.grid.weights[c];
                let x_c = self.grid.nodes[c];
                for j in 0..p {
                    mt_v[[c, j]] = inv_sd * (v[[c, j]] - w_c * sum_v[j] - w_c * x_c * sum_xv[j]);
                }
            }
        }
        // Aᵀ·(Mᵀ V), scattered through the recorded allocation:
        // row i accumulates (α_{ci}/W_c)·(Mᵀ V)_{c·} for every bin it touched.
        let mut out = Array2::<f64>::zeros((self.n_rows, p));
        for &(node, row, mass) in &self.alpha {
            let w_bin = self.bin_mass[node];
            if !(w_bin.is_finite() && w_bin > 0.0) {
                return Err(format!(
                    "empirical grid node-sensitivity VJP: bin {node} has non-positive mass {w_bin}"
                ));
            }
            let scale = mass / w_bin;
            if scale == 0.0 {
                continue;
            }
            for j in 0..p {
                out[[row, j]] += scale * mt_v[[node, j]];
            }
        }
        Ok(out)
    }
}

/// Equal-mass compression of `(z, weights)` into an at-most-`grid_size`-node
/// discrete measure, WITH the allocation record that makes it differentiable.
///
/// This is the ONLY builder — `build_global_empirical_latent_measure` returns
/// its `grid` and its record together — so the measure a fit integrates against
/// and the measure the correction differentiates cannot drift apart.
pub(crate) fn build_empirical_z_grid_with_alpha(
    z: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    grid_size: usize,
    context: &str,
) -> Result<EmpiricalZGridBuild, String> {
    if grid_size < 3 {
        return Err(format!(
            "empirical latent measure grid_size must be at least 3, got {grid_size}"
        ));
    }
    if z.len() != weights.len() {
        return Err(format!(
            "{context} length mismatch: z={}, weights={}",
            z.len(),
            weights.len()
        ));
    }
    // `(ζ, weight, ORIGINAL row)`. The row index is threaded through the filter
    // and the sort because it cannot be reconstructed afterwards; the comparator
    // still reads `.0` alone, so the stable sort's tie order is exactly what it
    // was before the index existed.
    let mut pairs = Vec::<(f64, f64, usize)>::with_capacity(z.len());
    for (idx, (&zi, &wi)) in z.iter().zip(weights.iter()).enumerate() {
        if !zi.is_finite() {
            return Err(format!(
                "{context} z value at row {idx} is non-finite ({zi})"
            ));
        }
        if !(wi.is_finite() && wi >= 0.0) {
            return Err(format!(
                "{context} weight at row {idx} must be finite and non-negative, got {wi}"
            ));
        }
        if wi > 0.0 {
            pairs.push((zi, wi, idx));
        }
    }
    if pairs.len() < 2 {
        return Err(format!(
            "{context} requires at least two positive-weight rows"
        ));
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .expect("validated empirical latent z values are finite")
    });
    let total_weight = pairs.iter().map(|(_, weight, _)| *weight).sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Err(format!("{context} requires positive finite total weight"));
    }

    let m = grid_size.min(pairs.len());
    let mut nodes = Vec::with_capacity(m);
    let mut out_weights = Vec::with_capacity(m);
    let mut bin_mass = Vec::with_capacity(m);
    let mut alpha = Vec::<(usize, usize, f64)>::with_capacity(pairs.len() + m);
    let bin_weight_target = total_weight / (m as f64);
    let mut cursor = 0usize;
    let mut remaining = pairs[0].1;
    for _ in 0..m {
        let mut need = bin_weight_target;
        let mut bin_weight = 0.0;
        let mut bin_sum = 0.0;
        // Allocation entries for the bin under construction. Held aside because
        // the emitted node index is only known once the bin is known to emit.
        let mut bin_alpha = Vec::<(usize, f64)>::new();
        // `need` loses one rounded subtraction per pair it takes and a pair's
        // `remaining` one per bin it feeds: a residual inside that accumulation's
        // band `γ·weight` is filled to working precision, so the cursor advances
        // instead of spinning on round-off.
        let need_band = gam_linalg::roundoff::accumulation_growth(pairs.len() + 1) * bin_weight_target;
        while need > need_band && cursor < pairs.len() {
            let take = remaining.min(need);
            bin_sum += take * pairs[cursor].0;
            bin_weight += take;
            bin_alpha.push((pairs[cursor].2, take));
            need -= take;
            remaining -= take;
            if remaining <= gam_linalg::roundoff::accumulation_growth(m + 1) * pairs[cursor].1 {
                cursor += 1;
                if cursor < pairs.len() {
                    remaining = pairs[cursor].1;
                }
            }
        }
        if bin_weight > 0.0 {
            let node_index = nodes.len();
            nodes.push(bin_sum / bin_weight);
            out_weights.push(bin_weight / total_weight);
            bin_mass.push(bin_weight);
            for (row, take) in bin_alpha {
                alpha.push((node_index, row, take));
            }
        }
    }
    if nodes.len() < 2 {
        return Err(format!(
            "{context} compression produced fewer than two nodes"
        ));
    }
    let standardization = recenter_rescale_empirical_grid(&mut nodes, &out_weights);
    let total = out_weights.iter().sum::<f64>();
    if total.is_finite() && total > 0.0 {
        for weight in &mut out_weights {
            *weight /= total;
        }
    }
    let tie_straddle = detect_tie_straddle(&pairs, total_weight, m);
    Ok(EmpiricalZGridBuild {
        grid: EmpiricalZGrid::new(nodes, out_weights, context)?,
        alpha,
        bin_mass,
        standardization_sd: standardization.map(|(_, sd)| sd),
        n_rows: z.len(),
        tie_straddle,
    })
}

/// Center and scale the grid nodes to weighted mean 0 / weighted sd 1, and
/// return the `(mean, sd)` the map used — `None` when the spread sat inside the
/// weighted mean's rounding band `γ_{n+1}·max|node|` and the map was therefore
/// skipped.
///
/// Returning the pair rather than `()` is what lets the sensitivity apply the
/// `1/sd` factor without a second copy of the mean/sd formula that could drift
/// from this one.
pub(crate) fn recenter_rescale_empirical_grid(
    nodes: &mut [f64],
    weights: &[f64],
) -> Option<(f64, f64)> {
    let total = weights.iter().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let mean = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * node)
        .sum::<f64>()
        / total;
    let var = nodes
        .iter()
        .zip(weights.iter())
        .map(|(&node, &weight)| weight * (node - mean).powi(2))
        .sum::<f64>()
        / total;
    let sd = var.sqrt();
    let magnitude = nodes.iter().fold(0.0_f64, |acc, node| acc.max(node.abs()));
    if sd.is_finite()
        && sd > gam_linalg::roundoff::accumulation_growth(nodes.len() + 1) * magnitude
    {
        for node in nodes {
            *node = (*node - mean) / sd;
        }
        Some((mean, sd))
    } else {
        None
    }
}

/// The differentiability certificate: find the first tied `ζ` group whose
/// cumulative-mass span an equal-mass bin boundary cuts.
///
/// A tied group that lies entirely inside one bin is harmless — every member
/// contributes its whole weight to that bin whatever order the sort chose — so
/// the check is exactly "does a boundary land strictly inside a tie", not "are
/// there ties". Boundaries are at `k·total/m` for `k = 1..m−1`; "strictly
/// inside" is measured with the same rounding band the fill loop uses to decide
/// a bin is exhausted, so a boundary that coincides with a row edge to working
/// precision counts as outside (which is what the fill loop does with it).
fn detect_tie_straddle(
    pairs: &[(f64, f64, usize)],
    total_weight: f64,
    m: usize,
) -> Option<EmpiricalGridTieStraddle> {
    let bin_weight_target = total_weight / (m as f64);
    let edge_tol = gam_linalg::roundoff::accumulation_growth(pairs.len() + 1) * bin_weight_target;
    let mut cumulative = 0.0_f64;
    let mut index = 0usize;
    while index < pairs.len() {
        let value = pairs[index].0;
        let start = cumulative;
        let mut end = cumulative;
        let mut run = 0usize;
        while index < pairs.len() && pairs[index].0 == value {
            end += pairs[index].1;
            run += 1;
            index += 1;
        }
        cumulative = end;
        if run < 2 {
            continue;
        }
        // Boundaries strictly inside (start, end).
        let first = ((start + edge_tol) / bin_weight_target).floor() as i64 + 1;
        let last = ((end - edge_tol) / bin_weight_target).ceil() as i64 - 1;
        for boundary in first.max(1)..=last.min(m as i64 - 1) {
            let position = (boundary as f64) * bin_weight_target;
            if position > start + edge_tol && position < end - edge_tol {
                return Some(EmpiricalGridTieStraddle {
                    value,
                    rows: run,
                    boundary: boundary as usize,
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// The row side: `S` (direct) and `U_Q` (through the grid)
// ---------------------------------------------------------------------------

/// The two `ζ`-sensitivity channels of the rigid empirical-grid BMS kernel.
///
/// Together with `D` these are everything the Murphy–Topel chain needs:
///
/// ```text
///   d score_β / d ζ_j  =  s_j  +  Σ_b u_b · D_{bj}
///   S_eff              =  S    +  Dᵀ·U_Qᵀ
/// ```
pub(crate) struct EmpiricalRigidZetaChannels {
    /// `S` (`n × p_β`): the DIRECT sensitivity `∂score_β,i/∂ζ_i` with the grid
    /// held fixed — the analogue of what
    /// [`super::gradient_paths::rigid_standard_normal_score_zeta_sensitivity`]
    /// returns for the closed-form kernel, and NOT equal to it: the empirical
    /// row's observed index is `a(m, g) + s·g·ζ_i` around an implicitly solved
    /// intercept, not `q·√(1+(s·g)²) + s·g·ζ_i`.
    pub(crate) direct: Array2<f64>,
    /// `U_Qᵀ` (`m × p_β`): `∂score_β/∂node_b` summed over ALL rows. This is the
    /// channel that has no counterpart in the standard-normal kernel, because
    /// there the measure is a fixed law rather than a statistic of the sample.
    pub(crate) node: Array2<f64>,
}

/// Both `ζ`-sensitivity channels of the rigid empirical-grid kernel, in one
/// pass over the rows (they share the per-row intercept solve, which is the
/// expensive part).
///
/// # The derivation
///
/// Write the row's implicitly-solved intercept as `a`, the grid nodes as `x_b`
/// with weights `π_b`, `s` for the probit frailty scale, and
/// `η_b = a + s·g·x_b`. The calibration that defines `a` is
///
/// ```text
///   F(a; m, g, x) = Σ_b π_b Φ(η_b) − μ(m) = 0
/// ```
///
/// so, with `Ψ_p = Σ_b π_b Φ^{(p)}(η_b)` and `Ξ_p = Σ_b π_b Φ^{(p)}(η_b)·(s x_b)`,
///
/// ```text
///   a_m   =  μ'(m)/Ψ_1              a_g = −Ξ_1/Ψ_1
///   a_x_b = −s·g·π_b Φ'(η_b)/Ψ_1
/// ```
///
/// The last line has the two properties that say the sign and scale are right:
/// `Σ_b a_x_b = −s·g` (shifting every node by δ shifts every `η_b` by `s·g·δ`,
/// which the intercept must absorb exactly), and it is exactly zero at `g = 0`
/// (a fit with no slope cannot see the latent axis).
///
/// Differentiating `a_m` and `a_g` once more, through both `a` and the explicit
/// `x_b`:
///
/// ```text
///   dΨ_1/dx_b = Ψ_2·a_x_b + s·g·π_b Φ''(η_b)
///   dΞ_1/dx_b = Ξ_2·a_x_b + s·g·π_b Φ''(η_b)·(s x_b) + s·π_b Φ'(η_b)
///   a_{m,x_b} = −a_m·(dΨ_1/dx_b)/Ψ_1
///   a_{g,x_b} = −[dΞ_1/dx_b + a_g·(dΨ_1/dx_b)]/Ψ_1
/// ```
///
/// The row's observed index is `e = a(m, g) + s·g·ζ_i`, so `e_m = a_m`,
/// `e_g = a_g + s·ζ_i`, `e_{x_b} = a_{x_b}`, `e_ζ = s·g`, `e_{m,ζ} = 0`,
/// `e_{g,ζ} = s`, and the second-order `e_{θ,x_b}` are the intercept's own. With
/// `ℓ = −w·log Φ(σ·e)` and `k = [ℓ, ℓ', ℓ'']` the shared signed-probit stack
/// ([`super::gradient_paths::signed_probit_neglog_unary_stack`], already carrying
/// the row weight and the NEGATION), the LOG-LIKELIHOOD mixed partials are
///
/// ```text
///   ∂²(log L)/∂θ∂u = −k₂·e_θ·e_u − σ·k₁·e_{θ,u}
/// ```
///
/// for `u ∈ {ζ_i, x_b}`, which is what this function evaluates and contracts
/// through the block design rows.
///
/// # Cost
///
/// One intercept root-solve and one `O(m)` node pass per row, then two
/// `n × m` by `n × p` GEMMs — so `O(n·(solve + m) + n·m·p_β)`, once, after the
/// fit. The row solve is the same one the fit performs at every inner
/// iteration, so this pass is a small fraction of a fit that already ran
/// thousands of them. Memory is two `n × m` coefficient blocks alongside the
/// `n × p_β` sensitivity the standard-normal branch already builds.
///
/// # Sign convention
///
/// LOG-LIKELIHOOD, matching
/// [`super::gradient_paths::rigid_standard_normal_mixed_z_sensitivity`] (#1131):
/// the returned quantities are `∂²(log L)/∂(primary)∂u`, so the downstream
/// `Vb·G` is `+∂β̂/∂θ₁` rather than its negative.
pub(crate) fn rigid_empirical_score_zeta_channels(
    base_link: &gam_problem::InverseLink,
    marginal_eta: &Array1<f64>,
    slope_eta: &Array1<f64>,
    zeta: &Array1<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    probit_scale: f64,
    grid: &EmpiricalZGrid,
    marginal_design: ArrayView2<'_, f64>,
    slope_design: ArrayView2<'_, f64>,
    p_beta: usize,
) -> Result<EmpiricalRigidZetaChannels, String> {
    let n = marginal_eta.len();
    let p_m = marginal_design.ncols();
    let r = slope_design.ncols();
    if slope_eta.len() != n
        || zeta.len() != n
        || y.len() != n
        || weights.len() != n
        || marginal_design.nrows() != n
        || slope_design.nrows() != n
    {
        return Err(format!(
            "empirical score_zeta channels row mismatch: marginal_eta={n}, slope_eta={}, zeta={}, \
             y={}, weights={}, marginal_design rows={}, slope_design rows={}",
            slope_eta.len(),
            zeta.len(),
            y.len(),
            weights.len(),
            marginal_design.nrows(),
            slope_design.nrows()
        ));
    }
    if p_m + r != p_beta {
        return Err(format!(
            "empirical score_zeta channels width mismatch: marginal({p_m}) + slope({r}) != \
             p_beta({p_beta})"
        ));
    }
    let m = grid.nodes.len();
    let s = probit_scale;

    let mut direct = Array2::<f64>::zeros((n, p_beta));
    // Per-row node coefficients, kept as two `n × m` blocks so the contraction
    // into `U_Qᵀ` is two GEMMs rather than an `O(n·m·p_β)` scalar scatter.
    let mut node_coeff_marginal = Array2::<f64>::zeros((n, m));
    let mut node_coeff_slope = Array2::<f64>::zeros((n, m));

    let mut phi1 = vec![0.0_f64; m];
    let mut phi2 = vec![0.0_f64; m];
    for i in 0..n {
        let marginal = super::family::bernoulli_marginal_link_map(base_link, marginal_eta[i])?;
        let g = slope_eta[i];
        let a = super::gradient_paths::empirical_intercept_from_marginal(
            marginal.mu,
            marginal.q,
            g,
            s,
            &grid.nodes,
            &grid.weights,
            None,
        )?;
        let observed_slope = s * g;

        // Ψ_1, Ψ_2, Ξ_1, Ξ_2 and the per-node CDF derivatives they are built
        // from. `Φ' = φ`, `Φ'' = −η·φ`, taken from the same stack the row jet
        // uses so the two paths cannot drift.
        let (mut psi1, mut psi2, mut xi1, mut xi2) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for b in 0..m {
            let node = grid.nodes[b];
            let pi = grid.weights[b];
            let eta_b = a + observed_slope * node;
            let stack = super::gradient_paths::unary_derivatives_normal_cdf(eta_b);
            let d1 = pi * stack[1];
            let d2 = pi * stack[2];
            phi1[b] = d1;
            phi2[b] = d2;
            psi1 += d1;
            psi2 += d2;
            xi1 += d1 * (s * node);
            xi2 += d2 * (s * node);
        }
        if !(psi1.is_finite() && psi1 > 0.0) {
            return Err(format!(
                "empirical score_zeta channels: non-positive calibration Jacobian Ψ₁={psi1} at \
                 row {i}"
            ));
        }
        let a_m = marginal.mu1 / psi1;
        let a_g = -xi1 / psi1;

        // The row's own observed index and the shared signed-probit stack. `k`
        // already carries the row weight and the NLL negation, so
        // `w·σ·L' = −σ·k₁` and `w·L'' = −k₂`.
        let sigma = 2.0 * y[i] - 1.0;
        let e = a + observed_slope * zeta[i];
        let k = super::gradient_paths::signed_probit_neglog_unary_stack(sigma * e, weights[i]);
        if !(k[1].is_finite() && k[2].is_finite()) {
            return Err(format!(
                "empirical score_zeta channels: non-finite signed-probit stack at row {i}"
            ));
        }
        let e_m = a_m;
        let e_g = a_g + s * zeta[i];

        // Direct channel: e_ζ = s·g, e_{m,ζ} = 0, e_{g,ζ} = s.
        let direct_m = -k[2] * e_m * observed_slope;
        let direct_g = -k[2] * e_g * observed_slope - sigma * k[1] * s;
        if direct_m != 0.0 {
            for (j, &x) in marginal_design.row(i).iter().enumerate() {
                direct[[i, j]] = direct_m * x;
            }
        }
        if direct_g != 0.0 {
            for (j, &x) in slope_design.row(i).iter().enumerate() {
                direct[[i, p_m + j]] = direct_g * x;
            }
        }

        // Node channel.
        for b in 0..m {
            let a_xb = -observed_slope * phi1[b] / psi1;
            let d_psi1 = psi2 * a_xb + observed_slope * phi2[b];
            let d_xi1 = xi2 * a_xb + observed_slope * phi2[b] * (s * grid.nodes[b]) + s * phi1[b];
            let a_m_xb = -a_m * d_psi1 / psi1;
            let a_g_xb = -(d_xi1 + a_g * d_psi1) / psi1;
            node_coeff_marginal[[i, b]] = -k[2] * e_m * a_xb - sigma * k[1] * a_m_xb;
            node_coeff_slope[[i, b]] = -k[2] * e_g * a_xb - sigma * k[1] * a_g_xb;
        }
    }

    // `U_Qᵀ[b, ·] = Σ_i coeff[i, b]·design.row(i)`, i.e. `Cᵀ·X` per block.
    let node_marginal =
        gam_linalg::faer_ndarray::fast_atb(&node_coeff_marginal.view(), &marginal_design);
    let node_slope =
        gam_linalg::faer_ndarray::fast_atb(&node_coeff_slope.view(), &slope_design);
    let mut node = Array2::<f64>::zeros((m, p_beta));
    node.slice_mut(ndarray::s![.., ..p_m])
        .assign(&node_marginal);
    node.slice_mut(ndarray::s![.., p_m..])
        .assign(&node_slope);

    if !direct.iter().all(|v| v.is_finite()) || !node.iter().all(|v| v.is_finite()) {
        return Err("empirical score_zeta channels produced a non-finite sensitivity".to_string());
    }
    Ok(EmpiricalRigidZetaChannels { direct, node })
}

// ---------------------------------------------------------------------------
// Which generated-regressor channel this fit's measure has
// ---------------------------------------------------------------------------

/// The seam's decision about the Murphy–Topel channel, as a value rather than a
/// branch buried in the fit driver.
///
/// gam#2484 originally read "a non-StandardNormal second-stage measure has no
/// generated-regressor correction". That is no longer the class: the ordinary
/// rigid `GlobalEmpirical` fit HAS one. What is left without a channel is a
/// short, enumerable list, and this type is where the list lives — separately
/// testable, so each arm has a witness that does not need a full fit.
pub(crate) enum EmpiricalGeneratedRegressorChannel<'a> {
    /// `StandardNormal`: the closed-form kernel, whose mixed derivative is local
    /// in the row. Nothing extra to add.
    ClosedForm,
    /// A rigid global-empirical measure with a differentiable build record: the
    /// direct channel is the empirical kernel's own and the cross-row channel is
    /// pulled back through this record.
    Empirical(&'a EmpiricalZGridBuild),
    /// No channel. The covariance is withheld and the reason says which shape or
    /// which property of the data removed it.
    Unavailable {
        latent_measure: String,
        unavailable_channel: String,
    },
}

/// Classify the measure the fit ended up with.
///
/// `flex_active` is the score-warp / link-deviation flag: those blocks evaluate
/// a basis AT the latent score, so the row's dependence on the grid is not the
/// rigid intercept's and the node derivative derived here does not describe it.
pub(crate) fn classify_empirical_generated_regressor_channel<'a>(
    latent_measure: &LatentMeasureKind,
    build: Option<&'a EmpiricalZGridBuild>,
    flex_active: bool,
) -> EmpiricalGeneratedRegressorChannel<'a> {
    match latent_measure {
        LatentMeasureKind::StandardNormal => EmpiricalGeneratedRegressorChannel::ClosedForm,
        LatentMeasureKind::LocalEmpirical { .. } => {
            EmpiricalGeneratedRegressorChannel::Unavailable {
                latent_measure: "local-empirical".to_string(),
                unavailable_channel:
                    "a per-row local-empirical measure carries no fit-time build record: it is \
                     only produced by deserializing a saved model, never by the latent-measure \
                     gate, so the equal-mass allocation behind its nodes is not known"
                        .to_string(),
            }
        }
        LatentMeasureKind::GlobalEmpirical { .. } if flex_active => {
            EmpiricalGeneratedRegressorChannel::Unavailable {
                latent_measure: "global-empirical".to_string(),
                unavailable_channel:
                    "a score-warp / link-deviation block evaluates a basis AT the latent score, \
                     so the row depends on the grid through that basis as well as through the \
                     calibration intercept, and the rigid kernel's node derivative is not the \
                     row's"
                        .to_string(),
            }
        }
        LatentMeasureKind::GlobalEmpirical { .. } => match build {
            None => EmpiricalGeneratedRegressorChannel::Unavailable {
                latent_measure: "global-empirical".to_string(),
                unavailable_channel:
                    "the global-empirical measure reached the covariance seam with no build \
                     record, so the equal-mass allocation it was compressed with is unknown"
                        .to_string(),
            },
            Some(build) => match build.tie_straddle.as_ref() {
                Some(tie) => EmpiricalGeneratedRegressorChannel::Unavailable {
                    latent_measure: "global-empirical".to_string(),
                    unavailable_channel: format!(
                        "the equal-mass compression is not differentiable in the calibrated \
                         score on this data: {} rows are tied at ζ = {:.6} and bin boundary {} \
                         cuts the tied group, so which member lands in which bin is an order the \
                         data does not define and the left and right derivatives of the grid \
                         nodes genuinely differ",
                        tie.rows, tie.value, tie.boundary
                    ),
                },
                None => EmpiricalGeneratedRegressorChannel::Empirical(build),
            },
        },
    }
}
