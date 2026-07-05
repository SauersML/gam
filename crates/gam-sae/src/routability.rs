//! Routability / interference-floor diagnostic — the routing-side account of SAE
//! "dark matter".
//!
//! Both sparse lanes route a row by a **gate**: the linear lane
//! ([`crate::sparse_dict`]) scores atom `k` by `|⟨x, d_k⟩|` (unit-norm `d_k`);
//! the block lane ([`crate::sparse_dict::block`]) scores block `g` by the group
//! ℓ₂ energy `‖x D_gᵀ‖₂` of an orthonormal `b_g`-frame `D_g`. A feature is
//! *routable* only if its own block wins that gate against every other block. The
//! question this module answers is quantitative: **below what target-to-clutter
//! ratio is a feature invisible to any gate at width `p`, no matter how the
//! dictionary is fitted?** That threshold is a hard floor on which features a
//! width-`p` router can ever separate — dark matter seen from the routing side.
//!
//! # Model and derivation
//!
//! Consider `K` blocks whose subspaces are in **generic position**: model each
//! frame `D_g` as an independent uniformly-random `b_g`-frame (its row-span
//! uniform on the Grassmannian `Gr(b_g, p)`), independent of the row being
//! routed. A row carries a **target firing** of amplitude `a` inside its own
//! block plus **interference + noise** of total ℓ₂ norm `ν`, orthogonal (in
//! generic position, asymptotically) to the target direction.
//!
//! **Target gate.** The row places energy `a` in its own block's subspace, so the
//! true block's gate is `≈ a` (the interference adds `O(ν·√(b/p))` in expectation,
//! lower order once `a/ν` is in the regime of interest).
//!
//! **Off-block gate (one block).** For an off-block `g`, the gate is the norm of
//! the row's projection onto `D_g`'s subspace. Because `D_g` is independent of the
//! row, the *target* component projects like any other vector and the
//! gate is governed by the residual: `‖P_g r‖` for `r` the unit residual direction
//! scaled by `ν`. Projecting a fixed unit vector onto a uniformly-random `b`-frame
//! is, in distribution, reading `b` coordinates of a uniformly-random unit vector,
//! so
//!
//! ```text
//!     E‖P_g r‖² = b/p,   hence   E‖P_g r‖ ≤ √(E‖P_g r‖²) = √(b/p)      (Jensen).
//! ```
//!
//! Thus the *mean* off-block gate is at most `ν·√(b/p)` — the **subspace term**.
//!
//! **Union over `K` blocks (Gaussian–Lipschitz).** The map `r ↦ ‖P_g r‖` is
//! 1-Lipschitz on the unit sphere `S^{p-1}`, so by Lévy concentration a single
//! off-block gate obeys
//!
//! ```text
//!     P( ‖P_g r‖ ≥ E‖P_g r‖ + t )  ≤  exp(−p t² / 2).
//! ```
//!
//! To bound the **maximum** off-block gate over all `K` blocks at confidence
//! `1 − δ`, allocate `δ/K` failure mass to each block: set
//! `exp(−p t²/2) = δ/K`, giving the deviation `t = √(2·ln(K/δ) / p)` — the
//! **union-bound term**. By the union bound, with probability `≥ 1 − δ`,
//!
//! ```text
//!     max_g ‖P_g r‖  ≤  ν·( √(b_max/p) + √( 2·ln(K/δ) / p ) ).
//! ```
//!
//! **Routability floor.** The true block wins whenever its gate `a` exceeds this
//! maximum off-block gate. Dividing by `ν` gives a *dimensionless* target-to-
//! clutter threshold on `a/ν`, the quantity [`RoutabilityFloor::floor`] holds:
//!
//! ```text
//!     a/ν  ≳  floor(δ)  =  √(b_max/p)  +  √( 2·ln(K/δ) / p ).
//! ```
//!
//! Setting `δ = 1` recovers the confidence-free form `√(b_max/p) + √(2 ln K / p)`;
//! smaller `δ` widens the floor by `√(2·ln(1/δ)/p)`. Everything is computed in
//! `f64`. For the linear lane `b_max = 1`; for the block lane `b_max` is the
//! largest block size.
//!
//! **Minimum routable energy.** If the target fires at amplitude `a` with
//! interference `ν` (orthogonal), the total energy at the firing site is
//! `a² + ν²`, of which the fraction the *target* carries is `a²/(a²+ν²)`. At the
//! floor `a = floor·ν` this is [`minimum_routable_energy`]:
//! `floor²/(1 + floor²)` — the fraction of residual energy a feature must place in
//! its own subspace at its firing sites to clear the router.
//!
//! # Generic position vs. trained dictionaries
//!
//! The floor above is the *random-model* floor. A trained dictionary's atoms are
//! correlated with the data (and with each other), so real residuals routed
//! against real atoms can exhibit larger cross-gates than the random model
//! predicts. [`routability_audit`] measures the empirical max-cross-gate
//! distribution on a fitted dictionary and reports its ratio to the closed-form
//! floor as [`RoutabilityAudit::coherence_excess`] — a direct audit of how far the
//! generic-position assumption is violated in practice.

use ndarray::ArrayView2;

/// Closed-form routability / interference floor for a width-`p` router over `K`
/// blocks of maximum subspace dimension `b_max`, at confidence `1 − δ`.
///
/// [`Self::floor`] is the **dimensionless** target-to-clutter threshold on `a/ν`:
/// a feature firing at amplitude `a` against interference of norm `ν` routes
/// correctly with probability `≥ 1 − δ` (under the generic-position model)
/// whenever `a/ν ≳ floor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutabilityFloor {
    /// Router width (ambient / activation dimension) `p`.
    pub p: usize,
    /// Number of blocks `K` the gate ranges over (for the linear lane, the number
    /// of atoms; for the block lane, the number of blocks `G`).
    pub n_blocks: usize,
    /// Largest block subspace dimension `b_max` (1 for the linear atom lane).
    pub b_max: usize,
    /// Confidence parameter `δ ∈ (0, 1]`: the floor holds with probability
    /// `1 − δ` over the random-frame model.
    pub delta: f64,
    /// The dimensionless floor `√(b_max/p) + √(2·ln(K/δ)/p)` on `a/ν`.
    pub floor: f64,
}

/// Closed-form [`RoutabilityFloor`]: `floor = √(b_max/p) + √(2·ln(K/δ)/p)`.
///
/// Panics on a degenerate configuration (`p == 0`, `n_blocks == 0`,
/// `b_max == 0`, `b_max > p`, or a non-finite / non-positive `δ`) — these are
/// programming errors, not data conditions, so failing closed is correct.
pub fn routability_floor(p: usize, n_blocks: usize, b_max: usize, delta: f64) -> RoutabilityFloor {
    assert!(p > 0, "routability_floor requires p >= 1");
    assert!(n_blocks > 0, "routability_floor requires n_blocks >= 1");
    assert!(b_max > 0, "routability_floor requires b_max >= 1");
    assert!(
        b_max <= p,
        "routability_floor requires b_max <= p (a b-frame must fit in R^p)"
    );
    assert!(
        delta.is_finite() && delta > 0.0,
        "routability_floor requires a finite delta > 0"
    );
    let pf = p as f64;
    // Subspace term: the mean off-block gate bound ν·√(b_max/p).
    let subspace = (b_max as f64 / pf).sqrt();
    // Union-bound term: √(2·ln(K/δ)/p). With n_blocks >= 1 and δ ≤ 1 the log is
    // non-negative; the max(0) guards the δ > 1 case a caller may still pass.
    let log_arg = (n_blocks as f64) / delta;
    let union = (2.0 * log_arg.ln().max(0.0) / pf).sqrt();
    let floor = subspace + union;
    RoutabilityFloor {
        p,
        n_blocks,
        b_max,
        delta,
        floor,
    }
}

/// Fraction of residual energy a feature must place in its own subspace at its
/// firing sites to be routable at all: `floor² / (1 + floor²)`.
///
/// Derived in the module doc: at the floor `a = floor·ν` with target ⟂
/// interference, the target carries `a²/(a²+ν²) = floor²/(1+floor²)` of the
/// firing-site energy. A feature below this energy fraction is invisible to any
/// width-`p` gate under the generic-position model.
pub fn minimum_routable_energy(floor: &RoutabilityFloor) -> f64 {
    let f2 = floor.floor * floor.floor;
    f2 / (1.0 + f2)
}

/// Empirical companion to [`routability_floor`]: the measured max-cross-gate
/// distribution of a *fitted* dictionary against real residuals, compared to the
/// closed-form floor.
#[derive(Clone, Debug)]
pub struct RoutabilityAudit {
    /// Number of residual rows audited (rows with (near-)zero norm are skipped).
    pub n_rows: usize,
    /// The closed-form floor this audit is measured against.
    pub floor: RoutabilityFloor,
    /// Requested `(level, value)` quantiles of the per-row max cross-gate
    /// (max over blocks of the group ℓ₂ gate divided by the residual norm),
    /// in the order the caller passed the levels.
    pub quantiles: Vec<(f64, f64)>,
    /// Mean per-row max cross-gate.
    pub empirical_mean: f64,
    /// Largest per-row max cross-gate observed.
    pub empirical_max: f64,
    /// The empirical `(1 − δ)`-quantile of the per-row max cross-gate — the
    /// empirical analogue of [`RoutabilityFloor::floor`] at the same confidence.
    pub confidence_quantile: f64,
    /// Coherence excess: `confidence_quantile / floor`. `1.0` means the fitted
    /// dictionary sits exactly on the generic-position floor; `> 1.0` means the
    /// trained atoms are more mutually/data coherent than random, so the *real*
    /// routing floor is higher than the random-model floor by this factor.
    pub coherence_excess: f64,
    /// Fraction of audited rows whose max cross-gate is at or below the floor.
    pub fraction_below_floor: f64,
}

/// Audit a fitted dictionary's routing floor against real residual rows.
///
/// `decoder` is `K×P` (linear lane: `K` unit-norm atom rows; block lane: `K =
/// G·b` frame rows, block `g` occupying rows `[g·b, g·b+b)`). `block_size` is `1`
/// for the linear atom lane, `b` for the block lane; it must divide `K`. For each
/// residual row `r` the per-block gate is the group ℓ₂ `‖r D_gᵀ‖₂` (which reduces
/// to `|⟨r, d_k⟩|` at `block_size == 1`); the row's **max cross-gate** is the
/// largest such gate over all blocks, divided by `‖r‖₂` so it is directly
/// comparable to the dimensionless [`RoutabilityFloor::floor`]. All gate
/// accumulation is `f64`.
///
/// `quantile_levels` are the quantile levels (each in `[0, 1]`) to report;
/// `delta` sets the confidence at which the closed-form floor and the empirical
/// `(1 − δ)`-quantile are compared.
pub fn routability_audit(
    decoder: ArrayView2<'_, f32>,
    residuals: ArrayView2<'_, f32>,
    block_size: usize,
    delta: f64,
    quantile_levels: &[f64],
) -> Result<RoutabilityAudit, String> {
    let k_rows = decoder.nrows();
    let p = decoder.ncols();
    if k_rows == 0 || p == 0 {
        return Err("routability_audit: decoder must be a non-empty K×P matrix".to_string());
    }
    if block_size == 0 {
        return Err("routability_audit: block_size must be >= 1".to_string());
    }
    if k_rows % block_size != 0 {
        return Err(format!(
            "routability_audit: decoder has K={k_rows} rows, not a multiple of block_size {block_size}"
        ));
    }
    if residuals.ncols() != p {
        return Err(format!(
            "routability_audit: residuals have P={} columns but the decoder has P={p}",
            residuals.ncols()
        ));
    }
    if !quantile_levels.iter().all(|&q| (0.0..=1.0).contains(&q)) {
        return Err("routability_audit: quantile levels must lie in [0, 1]".to_string());
    }
    let n_blocks = k_rows / block_size;
    let floor = routability_floor(p, n_blocks, block_size, delta);

    // Per-row max cross-gate = (max over blocks of ‖r D_gᵀ‖₂) / ‖r‖₂, f64.
    let mut per_row: Vec<f64> = Vec::with_capacity(residuals.nrows());
    for r in residuals.outer_iter() {
        let mut norm2 = 0.0f64;
        for &v in r.iter() {
            norm2 += v as f64 * v as f64;
        }
        let norm = norm2.sqrt();
        if norm <= 1.0e-12 {
            continue; // a zero residual routes nowhere; it carries no cross-gate
        }
        let mut best = 0.0f64;
        for g in 0..n_blocks {
            let mut energy = 0.0f64;
            for row_off in 0..block_size {
                let atom = decoder.row(g * block_size + row_off);
                let mut dot = 0.0f64;
                for (rv, av) in r.iter().zip(atom.iter()) {
                    dot += *rv as f64 * *av as f64;
                }
                energy += dot * dot;
            }
            let gate = energy.sqrt();
            if gate > best {
                best = gate;
            }
        }
        per_row.push(best / norm);
    }
    let n_rows = per_row.len();
    if n_rows == 0 {
        return Err("routability_audit: no residual rows with positive norm".to_string());
    }

    let mut sorted = per_row.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile_at = |level: f64| -> f64 {
        let idx = (level * (n_rows - 1) as f64).round() as usize;
        sorted[idx.min(n_rows - 1)]
    };

    let quantiles: Vec<(f64, f64)> = quantile_levels
        .iter()
        .map(|&level| (level, quantile_at(level)))
        .collect();
    let empirical_mean = per_row.iter().sum::<f64>() / n_rows as f64;
    let empirical_max = sorted[n_rows - 1];
    let confidence_quantile = quantile_at((1.0 - delta).clamp(0.0, 1.0));
    let coherence_excess = if floor.floor > 0.0 {
        confidence_quantile / floor.floor
    } else {
        f64::INFINITY
    };
    let fraction_below_floor =
        per_row.iter().filter(|&&v| v <= floor.floor).count() as f64 / n_rows as f64;

    Ok(RoutabilityAudit {
        n_rows,
        floor,
        quantiles,
        empirical_mean,
        empirical_max,
        confidence_quantile,
        coherence_excess,
        fraction_below_floor,
    })
}

#[cfg(test)]
#[path = "routability_tests.rs"]
mod routability_tests;
