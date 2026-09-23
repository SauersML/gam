//! #2933 F45 — the partition function of the softmax entropy gate prior.
//!
//! [`SoftmaxAssignmentSparsityPenalty`] scores `λ·H(a)` per row with `a = softmax(ℓ/τ)` on the
//! `(K − 1)`-simplex. As a density over the relaxed gates that energy needs
//!
//! ```text
//! Z_K(λ) = ∫_Δ exp(−λ·H(a)) da = ∫_Δ ∏_k exp(λ·a_k ln a_k) da,
//! ```
//!
//! Lebesgue measure on `(a_1, …, a_{K−1})`. The softmax change of variables over the `K − 1` free
//! logits supplies `|da/dℓ|`, not this mass, so the normalized negative log prior of a row is
//! `λ·H(a) + J + ln Z_K(λ)`.
//!
//! [`SoftmaxAssignmentSparsityPenalty`]: super::SoftmaxAssignmentSparsityPenalty

use gam_math::special::gauss_legendre;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

/// `ln Z_K(λ)` and its log-strength derivative; see [`softmax_entropy_log_partition`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftmaxEntropyLogPartition {
    /// `ln Z_K(λ)`.
    pub value: f64,
    /// `∂ ln Z_K/∂ ln λ = λ·E[Σ_k a_k ln a_k] = −λ·E[H]` under the normalized prior.
    pub log_strength_derivative: f64,
    /// Accepted table pieces summed over every group table.
    pub table_pieces: usize,
    /// Integrand evaluations spent.
    pub integrand_evaluations: usize,
}

/// Declared accuracy of [`softmax_entropy_log_partition`], relative to `1 + |·|` for the value and
/// for the log-strength derivative. Each group table is certified to
/// `PARTITION_LOCAL_TOLERANCE`; a table error enters the next split's integral at most twice, so
/// the `log₂K` levels of balanced splitting keep the result within `2·log₂K` local tolerances.
pub const SOFTMAX_ENTROPY_PARTITION_RELATIVE_TOLERANCE: f64 = 1.0e-9;

/// Relative interpolation error each table piece is certified to; `2·log₂(32768)·1e-11 = 3e-10`
/// is below [`SOFTMAX_ENTROPY_PARTITION_RELATIVE_TOLERANCE`].
const PARTITION_LOCAL_TOLERANCE: f64 = 1.0e-11;

/// Chebyshev--Lobatto intervals of the coarse interpolant a piece is checked with; the piece
/// stores the twice-finer rule whose new nodes do the checking.
const PARTITION_PIECE_ORDER: usize = 8;

/// Error each group integral is refined to, in the units the table check reads a node in: `ln Z_j`
/// against `1 + |ln Z_j|`, and `λ'·∂_λ ln Z_j` against `1 + |λ'·∂_λ ln Z_j|`. A node error `δ` in
/// those units reaches the check through the coarse interpolant as at most `(1 + Λ)·δ`, where
/// `Λ ≤ 1 + (2/π)·ln n` is the Lebesgue constant of the `n`-interval Chebyshev--Lobatto rule, so
/// holding it below half of `PARTITION_LOCAL_TOLERANCE` keeps the integrals' own error from being
/// read as interpolation error, which no bisection removes. The integrand's logarithm sums terms as
/// large as `|ln Z_j|`, so the mass and the moment each carry a rounding error of order
/// `ε·|ln Z_j|` relative to themselves; `ln Z_j` measured against `1 + |ln Z_j|`, and the slope
/// measured through `moment − slope·mass`, in which an error common to both integrals cancels, do
/// not inherit it.
fn partition_integral_tolerance() -> f64 {
    let lebesgue = 1.0 + std::f64::consts::FRAC_2_PI * (PARTITION_PIECE_ORDER as f64).ln();
    0.5 * PARTITION_LOCAL_TOLERANCE / (1.0 + lebesgue)
}

/// Pieces per group table at which [`softmax_entropy_log_partition`] refuses.
const PARTITION_MAX_PIECES: usize = 1 << 12;

/// Gauss--Legendre order of each panel rule.
const PARTITION_PANEL_ORDER: usize = 16;

/// Panel count at which one group integral refuses.
const PARTITION_MAX_PANELS: usize = 1 << 12;

/// Interior sample points per half interval used to locate the group integrand's mode.
const PARTITION_SCAN_POINTS: usize = 128;

/// Results kept for repeated `(K, λ)`: a fit evaluates its criterion at one strength many times.
const PARTITION_MEMO_CAPACITY: usize = 64;

/// One accepted interval `[u_left, u_right]` of a group table in `u = ln(1 + λ')`, with
/// `(ln Z_j, ∂_λ ln Z_j)` at its `2·PARTITION_PIECE_ORDER + 1` Chebyshev--Lobatto nodes.
struct TablePiece {
    u_left: f64,
    u_right: f64,
    log_partition: Vec<f64>,
    slope: Vec<f64>,
}

/// Every group table one `K` needs, each a sorted cover of `[0, ln(1 + λ)]` by accepted pieces.
struct GroupTables {
    span: f64,
    /// Lobatto nodes of the fine rule on `[0, 1]`; the even-indexed ones form the coarse rule.
    unit_nodes: Vec<f64>,
    fine_weights: Vec<f64>,
    coarse_weights: Vec<f64>,
    tables: HashMap<usize, Vec<TablePiece>>,
}

/// Barycentric weights of the `intervals`-interval Chebyshev--Lobatto rule.
fn lobatto_weights(intervals: usize) -> Vec<f64> {
    (0..=intervals)
        .map(|m| {
            let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
            if m == 0 || m == intervals { 0.5 * sign } else { sign }
        })
        .collect()
}

/// Barycentric interpolation of `(values, slopes)` at `x` on the given nodes and weights.
fn barycentric(nodes: &[f64], weights: &[f64], values: &[f64], slopes: &[f64], x: f64) -> (f64, f64) {
    let mut numerator_value = 0.0;
    let mut numerator_slope = 0.0;
    let mut denominator = 0.0;
    for (m, &node) in nodes.iter().enumerate() {
        let gap = x - node;
        if gap == 0.0 {
            return (values[m], slopes[m]);
        }
        let weight = weights[m] / gap;
        numerator_value += weight * values[m];
        numerator_slope += weight * slopes[m];
        denominator += weight;
    }
    (numerator_value / denominator, numerator_slope / denominator)
}

impl GroupTables {
    fn new(span: f64) -> Self {
        let fine = 2 * PARTITION_PIECE_ORDER;
        let unit_nodes = (0..=fine)
            .map(|m| 0.5 * (1.0 - (std::f64::consts::PI * m as f64 / fine as f64).cos()))
            .collect();
        Self {
            span,
            unit_nodes,
            fine_weights: lobatto_weights(fine),
            coarse_weights: lobatto_weights(PARTITION_PIECE_ORDER),
            tables: HashMap::new(),
        }
    }

    /// `(ln Z_j(λ'), ∂_λ ln Z_j(λ'))`. A single gate is the point simplex, `Z_1 ≡ 1`.
    fn group(&self, size: usize, strength: f64) -> Result<(f64, f64), String> {
        if size == 1 {
            return Ok((0.0, 0.0));
        }
        let pieces = self.tables.get(&size).ok_or_else(|| {
            format!("softmax entropy log partition: group table of size {size} was not built")
        })?;
        let u = strength.ln_1p().clamp(0.0, self.span);
        let index = pieces
            .partition_point(|piece| piece.u_right < u)
            .min(pieces.len().saturating_sub(1));
        let piece = pieces.get(index).ok_or_else(|| {
            format!("softmax entropy log partition: group table of size {size} is empty")
        })?;
        let width = piece.u_right - piece.u_left;
        let x = if width > 0.0 { (u - piece.u_left) / width } else { 0.0 };
        Ok(barycentric(
            &self.unit_nodes,
            &self.fine_weights,
            &piece.log_partition,
            &piece.slope,
            x,
        ))
    }

    /// Build the table of `size` from the tables of its halves. A piece is accepted when the coarse
    /// interpolant predicts the fine rule's new nodes to `PARTITION_LOCAL_TOLERANCE`, in the value
    /// and in `λ'·∂_λ ln Z_j`; otherwise it is bisected in `u`.
    fn build(&mut self, size: usize, evaluations: &mut usize) -> Result<(), String> {
        let first = size / 2;
        let coarse_nodes: Vec<f64> = self.unit_nodes.iter().step_by(2).copied().collect();
        let mut accepted = Vec::new();
        let mut pending = vec![(0.0_f64, self.span)];
        while let Some((u_left, u_right)) = pending.pop() {
            if accepted.len() + pending.len() >= PARTITION_MAX_PIECES {
                return Err(format!(
                    "softmax entropy log partition: group table of size {size} needs more than \
                     {PARTITION_MAX_PIECES} pieces over u ∈ [0, {}]",
                    self.span
                ));
            }
            let width = u_right - u_left;
            let mut log_partition = Vec::with_capacity(self.unit_nodes.len());
            let mut slope = Vec::with_capacity(self.unit_nodes.len());
            for &x in &self.unit_nodes {
                let node_strength = (u_left + x * width).exp_m1();
                let (value, derivative) =
                    combine_groups(self, first, size - first, node_strength, evaluations)?;
                log_partition.push(value);
                slope.push(derivative);
            }
            let coarse_values: Vec<f64> = log_partition.iter().step_by(2).copied().collect();
            let coarse_slopes: Vec<f64> = slope.iter().step_by(2).copied().collect();
            let mut worst = 0.0_f64;
            for m in (1..self.unit_nodes.len()).step_by(2) {
                let x = self.unit_nodes[m];
                let node_strength = (u_left + x * width).exp_m1();
                let (value, derivative) = barycentric(
                    &coarse_nodes,
                    &self.coarse_weights,
                    &coarse_values,
                    &coarse_slopes,
                    x,
                );
                let value_gap = (value - log_partition[m]).abs() / (1.0 + log_partition[m].abs());
                let derivative_gap = node_strength * (derivative - slope[m]).abs()
                    / (1.0 + (node_strength * slope[m]).abs());
                worst = worst.max(value_gap).max(derivative_gap);
            }
            if !worst.is_finite() {
                return Err(format!(
                    "softmax entropy log partition: non-finite table check for size {size} on u ∈ \
                     [{u_left}, {u_right}]"
                ));
            }
            if worst <= PARTITION_LOCAL_TOLERANCE || width == 0.0 {
                accepted.push(TablePiece {
                    u_left,
                    u_right,
                    log_partition,
                    slope,
                });
                continue;
            }
            let middle = 0.5 * (u_left + u_right);
            if !(middle > u_left && middle < u_right) {
                return Err(format!(
                    "softmax entropy log partition: group table of size {size} did not reach \
                     relative error {PARTITION_LOCAL_TOLERANCE:e} on u ∈ [{u_left}, {u_right}] \
                     (check gap {worst:.3e})"
                ));
            }
            pending.push((u_left, middle));
            pending.push((middle, u_right));
        }
        accepted.sort_by(|a, b| a.u_left.total_cmp(&b.u_left));
        self.tables.insert(size, accepted);
        Ok(())
    }
}

/// `t ln t`, with its limit `0` at `t = 0`.
fn entropy_term(t: f64) -> f64 {
    if t > 0.0 { t * t.ln() } else { 0.0 }
}

/// One panel of a group integral on the half-interval variable `v ∈ [0, 1/2]`.
struct PartitionPanel {
    left: f64,
    right: f64,
    mass: f64,
    moment: f64,
    /// Whole-panel rule minus the rule on its halves, for the mass and for the moment.
    mass_error: f64,
    moment_error: f64,
}

/// `(ln Z_{j1+j2}(λ), ∂_λ ln Z_{j1+j2}(λ))` from the two group tables.
///
/// Splitting the gates into groups of sizes `j1` and `j2` with group masses `t` and `1 − t`, and
/// rescaling each group to its own simplex, gives
///
/// ```text
/// Z_{j1+j2}(λ) = ∫₀¹ t^{j1−1} (1−t)^{j2−1} e^{λ[t ln t + (1−t) ln(1−t)]} Z_{j1}(λt) Z_{j2}(λ(1−t)) dt,
/// ∂_λ ln Z_{j1+j2}(λ) = E[t ln t + (1−t) ln(1−t) + t·∂ ln Z_{j1}(λt) + (1−t)·∂ ln Z_{j2}(λ(1−t))],
/// ```
///
/// because `Σ_{k ∈ group} a_k ln a_k = t ln t + t·Σ b_k ln b_k` for `a = t·b`. Each half of `[0, 1]`
/// is integrated in the distance `v` from its own endpoint, so `t` and `1 − t` are both exact
/// near either end. Panels are cut at the scanned mode at widths `2^i/(2√(j1 + j2))`, the scale of
/// the Beta factor, and geometrically toward both endpoints down to the scale `1/(λ ln λ)` at which
/// a large strength concentrates the mass near a vertex. Each panel carries the 16-point rule on
/// its halves, with the difference from the whole-panel rule as its error indicator. The panel with
/// the largest indicator relative to its budget is bisected until the indicators sum within
/// [`partition_integral_tolerance`] in the units of the table check: the mass errors over the mass
/// against `1 + |ln Z|`, and the errors of `moment − slope·mass` over the mass against
/// `1/max(1, λ) + |slope|`, which bounds `λ·∂_λ ln Z` against `1 + |λ·∂_λ ln Z|` for every `λ`.
/// Exhausting [`PARTITION_MAX_PANELS`] is a refusal.
fn combine_groups(
    tables: &GroupTables,
    first: usize,
    second: usize,
    strength: f64,
    evaluations: &mut usize,
) -> Result<(f64, f64), String> {
    // `(log integrand, moment weight)` at the group masses `(t, s)`, `t + s = 1`.
    let integrand = |t: f64, s: f64| -> Result<(f64, f64), String> {
        let (log_first, slope_first) = tables.group(first, strength * t)?;
        let (log_second, slope_second) = tables.group(second, strength * s)?;
        let mixing = entropy_term(t) + entropy_term(s);
        let log_value = (first - 1) as f64 * t.ln()
            + (second - 1) as f64 * s.ln()
            + strength * mixing
            + log_first
            + log_second;
        Ok((log_value, mixing + t * slope_first + s * slope_second))
    };
    // Half 0 measures `v` from `t = 0`, half 1 from `t = 1`.
    let masses = |half: usize, v: f64| if half == 0 { (v, 1.0 - v) } else { (1.0 - v, v) };

    let width = 0.5 / ((first + second) as f64).sqrt();
    let mut peak = f64::NEG_INFINITY;
    let mut scan_modes = [0.5_f64; 2];
    for (half, scan_mode) in scan_modes.iter_mut().enumerate() {
        let mut best = f64::NEG_INFINITY;
        for i in 1..=PARTITION_SCAN_POINTS {
            let v = 0.5 * i as f64 / PARTITION_SCAN_POINTS as f64;
            let (t, s) = masses(half, v);
            let (log_value, _) = integrand(t, s)?;
            *evaluations += 1;
            if log_value > best {
                best = log_value;
                *scan_mode = v;
            }
        }
        peak = peak.max(best);
    }
    if !peak.is_finite() {
        return Err(format!(
            "softmax entropy log partition: group integrand has no finite mode for sizes \
             ({first}, {second}) at λ={strength}"
        ));
    }

    // Endpoint cuts reach the vertex scale `1/(λ ln λ)` of a large strength, and at least `2^−4`.
    let vertex_scale = strength * strength.ln_1p();
    let depth = (16.0 * (1.0 + vertex_scale)).log2().ceil().clamp(4.0, 60.0) as i32;
    let mut half_cuts = Vec::with_capacity(2);
    for (half, &scan_mode) in scan_modes.iter().enumerate() {
        let mut cuts = vec![0.0, 0.5, scan_mode];
        for i in 2..=depth {
            cuts.push(0.5_f64.powi(i));
        }
        let mut step = width;
        while scan_mode - step > 0.0 {
            cuts.push(scan_mode - step);
            step *= 2.0;
        }
        step = width;
        while scan_mode + step < 0.5 {
            cuts.push(scan_mode + step);
            step *= 2.0;
        }
        cuts.retain(|cut| (0.0..=0.5).contains(cut));
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        // A large strength puts the integrand's maximum at a vertex, below the scan's
        // resolution; the geometric cuts sample it, so the normalizing peak cannot be
        // exceeded by orders of magnitude inside a panel.
        for &cut in cuts.iter().filter(|&&cut| cut > 0.0) {
            let (t, s) = masses(half, cut);
            let (log_value, _) = integrand(t, s)?;
            *evaluations += 1;
            peak = peak.max(log_value);
        }
        half_cuts.push((half, cuts));
    }

    let (nodes, weights) = gauss_legendre(PARTITION_PANEL_ORDER);
    let mut rule = |half: usize, left: f64, right: f64| -> Result<(f64, f64), String> {
        let centre = 0.5 * (left + right);
        let radius = 0.5 * (right - left);
        let mut mass = 0.0;
        let mut moment = 0.0;
        for (node, weight) in nodes.iter().zip(&weights) {
            let (t, s) = masses(half, centre + radius * node);
            let (log_value, psi) = integrand(t, s)?;
            let f = (log_value - peak).exp();
            mass += weight * f;
            moment += weight * psi * f;
        }
        *evaluations += nodes.len();
        Ok((radius * mass, radius * moment))
    };
    let mut priced = |half: usize, left: f64, right: f64| -> Result<(usize, PartitionPanel), String> {
        let middle = 0.5 * (left + right);
        if !(middle > left && middle < right) {
            return Err(format!(
                "softmax entropy log partition: panel [{left}, {right}] cannot be halved for \
                 sizes ({first}, {second}) at λ={strength}"
            ));
        }
        let (coarse_mass, coarse_moment) = rule(half, left, right)?;
        let (left_mass, left_moment) = rule(half, left, middle)?;
        let (right_mass, right_moment) = rule(half, middle, right)?;
        let mass = left_mass + right_mass;
        let moment = left_moment + right_moment;
        Ok((
            half,
            PartitionPanel {
                left,
                right,
                mass,
                moment,
                mass_error: coarse_mass - mass,
                moment_error: coarse_moment - moment,
            },
        ))
    };

    let mut panels = Vec::new();
    for (half, cuts) in &half_cuts {
        for pair in cuts.windows(2) {
            panels.push(priced(*half, pair[0], pair[1])?);
        }
    }
    let tolerance = partition_integral_tolerance();
    loop {
        let mass: f64 = panels.iter().map(|(_, panel)| panel.mass).sum();
        let moment: f64 = panels.iter().map(|(_, panel)| panel.moment).sum();
        if !(mass > 0.0 && mass.is_finite() && moment.is_finite()) {
            return Err(format!(
                "softmax entropy log partition: non-positive or non-finite mass {mass} / moment \
                 {moment} for sizes ({first}, {second}) at λ={strength}"
            ));
        }
        let log_partition = mass.ln() + peak;
        let slope = moment / mass;
        let value_budget = tolerance * (1.0 + log_partition.abs());
        let slope_budget = tolerance * (1.0 / strength.max(1.0) + slope.abs());
        let value_error = |panel: &PartitionPanel| panel.mass_error.abs() / mass;
        let slope_error =
            |panel: &PartitionPanel| (panel.moment_error - slope * panel.mass_error).abs() / mass;
        let value_indicator: f64 = panels.iter().map(|(_, panel)| value_error(panel)).sum();
        let slope_indicator: f64 = panels.iter().map(|(_, panel)| slope_error(panel)).sum();
        if value_indicator <= value_budget && slope_indicator <= slope_budget {
            return Ok((log_partition, slope));
        }
        if panels.len() >= PARTITION_MAX_PANELS {
            return Err(format!(
                "softmax entropy log partition did not converge within {PARTITION_MAX_PANELS} \
                 panels for sizes ({first}, {second}) at λ={strength}: indicators \
                 {value_indicator:.3e} against budget {value_budget:.3e} (ln Z), \
                 {slope_indicator:.3e} against budget {slope_budget:.3e} (∂_λ ln Z)"
            ));
        }
        let indicator = |panel: &PartitionPanel| {
            value_error(panel) / value_budget + slope_error(panel) / slope_budget
        };
        let mut worst = 0;
        for (index, (_, panel)) in panels.iter().enumerate() {
            if indicator(panel) > indicator(&panels[worst].1) {
                worst = index;
            }
        }
        let (half, panel) = panels.swap_remove(worst);
        let middle = 0.5 * (panel.left + panel.right);
        panels.push(priced(half, panel.left, middle)?);
        panels.push(priced(half, middle, panel.right)?);
    }
}

/// Every group size below `k_atoms` that balanced halving reaches, smallest first.
fn group_sizes(k_atoms: usize) -> BTreeSet<usize> {
    fn collect(size: usize, sizes: &mut BTreeSet<usize>) {
        let first = size / 2;
        for part in [first, size - first] {
            if part >= 2 && sizes.insert(part) {
                collect(part, sizes);
            }
        }
    }
    let mut sizes = BTreeSet::new();
    collect(k_atoms, &mut sizes);
    sizes
}

/// Every group table, then the top split integrated at `λ` itself. Returns
/// `(ln Z_K, ∂_λ ln Z_K, accepted pieces)`.
fn log_partition_from_tables(
    k_atoms: usize,
    strength: f64,
    evaluations: &mut usize,
) -> Result<(f64, f64, usize), String> {
    let mut tables = GroupTables::new(strength.ln_1p());
    for size in group_sizes(k_atoms) {
        tables.build(size, evaluations)?;
    }
    let pieces = tables.tables.values().map(Vec::len).sum();
    let first = k_atoms / 2;
    let (value, slope) = combine_groups(&tables, first, k_atoms - first, strength, evaluations)?;
    Ok((value, slope, pieces))
}

fn partition_memo() -> &'static Mutex<Vec<(usize, u64, SoftmaxEntropyLogPartition)>> {
    static MEMO: OnceLock<Mutex<Vec<(usize, u64, SoftmaxEntropyLogPartition)>>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Vec::with_capacity(PARTITION_MEMO_CAPACITY)))
}

/// `ln Z_K(λ)` and `∂ ln Z_K/∂ ln λ` for the softmax entropy prior `exp(−λ·H(a))` on the
/// `(K − 1)`-simplex (#2933 F45).
///
/// The simplex integral of a product over coordinates is computed by balanced group splitting
/// (`combine_groups`): a group of `j` gates needs `ln Z_j` and its slope at every strength in
/// `[0, λ]`, tabulated on Chebyshev--Lobatto nodes in `u = ln(1 + λ')` and read back by barycentric
/// interpolation, and the top split is integrated at `λ` itself. Group sizes halve, so a `K`-gate
/// partition needs at most `2·log₂K` tables and its cost does not grow with `K` beyond that.
///
/// Accuracy. Each table covers `[0, ln(1 + λ)]` by pieces. A piece is accepted when its 8-interval
/// interpolant predicts the 16-interval rule's new nodes to `PARTITION_LOCAL_TOLERANCE`, and is
/// bisected otherwise, so the pieces concentrate where `ln Z_j` bends: near `λ' ≈ j` a large group
/// passes from near-uniform routing to near-vertex routing, and a single global rule cannot resolve
/// that bend. The declared result error is [`SOFTMAX_ENTROPY_PARTITION_RELATIVE_TOLERANCE`]. This is
/// refinement of a numerical integral of one fixed function, not a search over parameters
/// (SPEC 18). Exhausting `PARTITION_MAX_PIECES`, a piece that cannot be halved, or a group
/// integral that does not converge, is a refusal.
///
/// Limits: `Z_K(0) = 1/(K − 1)!` with slope `∂_λ ln Z_K(0) = 1 − H_K` (`H_K` the harmonic number),
/// and `ln Z_K` is convex in `λ` (its second derivative is `Var[Σ a ln a]`), so
/// `−ln (K−1)! + λ(1 − H_K) ≤ ln Z_K(λ) ≤ −ln (K−1)!`. `K = 1` is the point simplex, `Z_1 ≡ 1`.
pub fn softmax_entropy_log_partition(
    k_atoms: usize,
    strength: f64,
) -> Result<SoftmaxEntropyLogPartition, String> {
    if k_atoms == 0 || !(strength.is_finite() && strength >= 0.0) {
        return Err(format!(
            "softmax entropy log partition needs K >= 1 and a finite non-negative strength; got \
             K={k_atoms}, λ={strength}"
        ));
    }
    if k_atoms == 1 {
        return Ok(SoftmaxEntropyLogPartition {
            value: 0.0,
            log_strength_derivative: 0.0,
            table_pieces: 0,
            integrand_evaluations: 0,
        });
    }
    if let Ok(memo) = partition_memo().lock()
        && let Some((_, _, result)) = memo
            .iter()
            .find(|(k, bits, _)| *k == k_atoms && *bits == strength.to_bits())
    {
        return Ok(*result);
    }
    let mut evaluations = 0;
    let (value, slope, pieces) = log_partition_from_tables(k_atoms, strength, &mut evaluations)?;
    let result = SoftmaxEntropyLogPartition {
        value,
        log_strength_derivative: strength * slope,
        table_pieces: pieces,
        integrand_evaluations: evaluations,
    };
    if let Ok(mut memo) = partition_memo().lock() {
        if memo.len() >= PARTITION_MEMO_CAPACITY {
            memo.remove(0);
        }
        memo.push((k_atoms, strength.to_bits(), result));
    }
    Ok(result)
}

#[cfg(test)]
mod softmax_entropy_partition_tests {
    //! Every reference here is an independent quadrature of the simplex integral itself
    //! (tanh--sinh in each coordinate), not the group recursion.
    use super::*;
    use statrs::function::gamma::ln_gamma;

    /// Double-exponential quadrature of `f` on `[a, b]`: halve the step until two estimates
    /// agree to `1e-12` relative, from `h = 1/2` down to `h = 1/256`. `f` receives
    /// `(x, x − a, b − x)` so endpoint distances stay exact.
    fn tanh_sinh(a: f64, b: f64, f: &dyn Fn(f64, f64, f64) -> f64) -> f64 {
        let half = 0.5 * (b - a);
        let estimate = |h: f64| {
            let mut sum = 0.0;
            let steps = (6.0 / h).ceil() as i64;
            for i in -steps..=steps {
                let x = i as f64 * h;
                let s = std::f64::consts::FRAC_PI_2 * x.sinh();
                let weight = std::f64::consts::FRAC_PI_2 * x.cosh() / s.cosh().powi(2);
                // Distance from the nearer endpoint: `1 − tanh s = 2/(1 + e^{2s})`.
                let to_left = half * 2.0 / (1.0 + (2.0 * s).exp());
                let to_right = half * 2.0 / (1.0 + (-2.0 * s).exp());
                if to_left <= 0.0 || to_right <= 0.0 {
                    continue;
                }
                sum += weight * f(a + to_left, to_left, to_right);
            }
            half * h * sum
        };
        let mut h = 0.5;
        let mut previous = estimate(h);
        loop {
            h *= 0.5;
            let current = estimate(h);
            if (current - previous).abs() <= 1.0e-12 * current.abs() || h <= 1.0 / 256.0 {
                return current;
            }
            previous = current;
        }
    }

    /// `(ln Z_2(λ), ∂_λ ln Z_2(λ))` by direct quadrature over `a_1 ∈ (0, 1)`.
    fn reference_two(strength: f64) -> (f64, f64) {
        let log_f = |t: f64, s: f64| strength * (entropy_term(t) + entropy_term(s));
        let mass = tanh_sinh(0.0, 1.0, &|_, t, s| log_f(t, s).exp());
        let moment = tanh_sinh(0.0, 1.0, &|_, t, s| {
            (entropy_term(t) + entropy_term(s)) * log_f(t, s).exp()
        });
        (mass.ln(), moment / mass)
    }

    /// `(ln Z_3(λ), ∂_λ ln Z_3(λ))` by nested quadrature over the triangle.
    fn reference_three(strength: f64) -> (f64, f64) {
        let inner = |x: f64, rest: f64, weighted: bool| {
            tanh_sinh(0.0, rest, &|_, y, z| {
                let s = entropy_term(x) + entropy_term(y) + entropy_term(z);
                let f = (strength * s).exp();
                if weighted { s * f } else { f }
            })
        };
        let mass = tanh_sinh(0.0, 1.0, &|x, _, rest| inner(x, rest, false));
        let moment = tanh_sinh(0.0, 1.0, &|x, _, rest| inner(x, rest, true));
        (mass.ln(), moment / mass)
    }

    fn partition(k: usize, strength: f64) -> SoftmaxEntropyLogPartition {
        softmax_entropy_log_partition(k, strength)
            .unwrap_or_else(|error| panic!("K={k}, λ={strength}: {error}"))
    }

    #[test]
    fn two_and_three_gate_partitions_match_direct_quadrature_2933() {
        for strength in [1.0e-6, 0.3, 2.0, 9.0, 40.0, 300.0] {
            for (k, (log_mass, slope)) in
                [(2usize, reference_two(strength)), (3, reference_three(strength))]
            {
                let computed = partition(k, strength);
                assert!(
                    (computed.value - log_mass).abs() <= 1.0e-8 * (1.0 + log_mass.abs()),
                    "K={k}, λ={strength}: ln Z {:.15e} vs direct quadrature {log_mass:.15e}",
                    computed.value
                );
                let derivative = strength * slope;
                assert!(
                    (computed.log_strength_derivative - derivative).abs()
                        <= 1.0e-8 * (1.0 + derivative.abs()),
                    "K={k}, λ={strength}: ∂ ln Z/∂ ln λ {:.15e} vs −λ·E[H] by direct \
                     quadrature {derivative:.15e}",
                    computed.log_strength_derivative
                );
            }
        }
    }

    /// The derivative is the prior expectation of `Σ a ln a` (checked against direct quadrature
    /// above); here it must also be a central difference of the value in `ln λ`.
    #[test]
    fn log_strength_derivative_is_a_central_difference_of_the_value_2933() {
        for k in [2usize, 3, 7, 64] {
            for strength in [0.4_f64, 5.0, 80.0] {
                let h = 1.0e-4_f64;
                let difference = (partition(k, strength * h.exp()).value
                    - partition(k, strength * (-h).exp()).value)
                    / (2.0 * h);
                let derivative = partition(k, strength).log_strength_derivative;
                assert!(
                    (derivative - difference).abs() <= 1.0e-6 * (1.0 + derivative.abs()),
                    "K={k}, λ={strength}: derivative {derivative:.12e} vs central difference \
                     {difference:.12e}"
                );
            }
        }
    }

    #[test]
    fn vanishing_strength_recovers_the_simplex_volume_and_uniform_slope_2933() {
        for k in [2usize, 5, 64, 1000] {
            let strength = 1.0e-10;
            let computed = partition(k, strength);
            let log_volume = -ln_gamma(k as f64);
            assert!(
                (computed.value - log_volume).abs() <= 1.0e-8 * (1.0 + log_volume.abs()),
                "K={k}: ln Z at λ→0 is {:.15e}, the simplex volume gives {log_volume:.15e}",
                computed.value
            );
            let harmonic: f64 = (1..=k).map(|i| 1.0 / i as f64).sum();
            let slope = computed.log_strength_derivative / strength;
            assert!(
                (slope - (1.0 - harmonic)).abs() <= 1.0e-6 * harmonic,
                "K={k}: ∂_λ ln Z at λ→0 is {slope:.12e}, the uniform simplex gives \
                 {:.12e}",
                1.0 - harmonic
            );
        }
    }

    /// At `K = 32768` the value and slope sit inside the convexity bounds at small, moderate and
    /// large strength, and the cost stays bounded by the halving recursion.
    #[test]
    fn large_dictionary_partition_is_bounded_and_affordable_2933() {
        let k = 32_768usize;
        let log_volume = -ln_gamma(k as f64);
        let harmonic: f64 = (1..=k).map(|i| 1.0 / i as f64).sum();
        for strength in [1.0_f64, 32_768.0, 1.0e6] {
            let started = std::time::Instant::now();
            let computed = partition(k, strength);
            let elapsed = started.elapsed();
            eprintln!(
                "[F45] K={k} λ={strength}: ln Z={:.12e}, ∂lnZ/∂lnλ={:.12e}, table pieces {}, \
                 evaluations {}, wall {elapsed:.2?}",
                computed.value,
                computed.log_strength_derivative,
                computed.table_pieces,
                computed.integrand_evaluations
            );
            let value_slack = SOFTMAX_ENTROPY_PARTITION_RELATIVE_TOLERANCE * (1.0 + log_volume.abs());
            assert!(
                computed.value <= log_volume + value_slack
                    && computed.value
                        >= log_volume + strength * (1.0 - harmonic) - value_slack,
                "K={k}, λ={strength}: ln Z {:.12e} outside [{:.12e}, {log_volume:.12e}]",
                computed.value,
                log_volume + strength * (1.0 - harmonic)
            );
            let slope = computed.log_strength_derivative / strength;
            let slope_slack =
                SOFTMAX_ENTROPY_PARTITION_RELATIVE_TOLERANCE * (1.0 / strength + (harmonic - 1.0));
            assert!(
                slope <= 0.0 && slope >= (1.0 - harmonic) - slope_slack,
                "K={k}, λ={strength}: slope {slope:.12e} outside [{:.12e}, 0]",
                1.0 - harmonic
            );
            assert!(
                computed.integrand_evaluations <= 200_000_000,
                "K={k}, λ={strength}: {} integrand evaluations",
                computed.integrand_evaluations
            );
        }
    }

    #[test]
    fn invalid_inputs_are_refused_2933() {
        for (k, strength) in [(0usize, 1.0), (3, -1.0), (3, f64::NAN), (3, f64::INFINITY)] {
            assert!(
                softmax_entropy_log_partition(k, strength).is_err(),
                "K={k}, λ={strength} must be refused"
            );
        }
        assert_eq!(partition(1, 7.0).value, 0.0, "the point simplex has unit mass");
    }
}
