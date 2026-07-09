//! Rust-owned numerical helpers for the torch-only manifold-SAE routing lane.
//!
//! These kernels are intentionally small, deterministic primitives that the
//! Python torch module calls as FFI glue.  Keeping them here preserves the repo
//! doctrine that Python marshals tensors while Rust owns numeric rules.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};

/// Sinkhorn per-atom log-bias potentials that balance atom usage.
///
/// `log_scores[n, k]` are the (temperature-scaled) unnormalized
/// log-responsibilities of row `n` for atom `k`. This returns the per-atom
/// additive potential `b_k` such that `softmax_k(log_scores + b)` has an
/// approximately uniform column marginal (each atom claims `1/K` of the total
/// assignment mass). The caller forms the balanced log-responsibilities as
/// `log_scores + b`, keeping `log_scores` on the autograd tape while `b` enters
/// as a detached constant — exactly the `@no_grad` steering the torch lane used
/// to perform inline.
///
/// The potentials are the fixed point of the multiplicative Sinkhorn sweep
/// toward the uniform atom marginal `target = log(1/K)`, mean-centered each step
/// (the softmax is shift-invariant, so centering only fixes the gauge and keeps
/// the potentials bounded). A handful of sweeps converge it; `iters` matches the
/// torch lane's fixed count. Returns a zero vector for `K < 2` (nothing to
/// balance).
#[must_use]
pub fn sinkhorn_balance_bias(log_scores: ArrayView2<'_, f64>, iters: usize) -> Array1<f64> {
    let n_atoms = log_scores.ncols();
    let mut bias = Array1::<f64>::zeros(n_atoms);
    let n_rows = log_scores.nrows();
    if n_atoms < 2 || n_rows == 0 {
        return bias;
    }
    let target = (1.0 / n_atoms as f64).ln();
    let inv_rows = 1.0 / n_rows as f64;
    let mut usage = Array1::<f64>::zeros(n_atoms);
    let mut exps = vec![0.0_f64; n_atoms];
    for _ in 0..iters {
        usage.fill(0.0);
        for row in log_scores.rows() {
            // Numerically stable row softmax of `row + bias`.
            let mut max = f64::NEG_INFINITY;
            for k in 0..n_atoms {
                let v = row[k] + bias[k];
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0;
            for k in 0..n_atoms {
                let e = (row[k] + bias[k] - max).exp();
                exps[k] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for k in 0..n_atoms {
                usage[k] += exps[k] * inv_sum;
            }
        }
        // `usage[k]` now holds Σ_rows softmax(row+bias)[k]; convert to the mean
        // assignment mass and take the multiplicative step toward uniform.
        for k in 0..n_atoms {
            let mean_usage = (usage[k] * inv_rows).max(1e-12);
            bias[k] += target - mean_usage.ln();
        }
        let mean_bias = bias.mean().unwrap_or(0.0);
        bias.mapv_inplace(|v| v - mean_bias);
    }
    bias
}

/// Lift a one-dimensional Duchon center vector into a deterministic `(K, d)`
/// low-discrepancy cloud in `[0, 1]^d`.
///
/// The first coordinate is the caller-provided 1-D center.  Remaining axes use
/// the generalized golden-ratio additive recurrence (`R_d`) keyed only to
/// `(K, d)`, matching the historical torch implementation bit-for-bit for f64
/// inputs before dtype conversion at the Python boundary.
#[must_use]
pub fn duchon_centers_nd(centers_1d: ArrayView1<'_, f64>, d: usize) -> Array2<f64> {
    let k = centers_1d.len();
    let width = d.max(1);
    let mut out = Array2::<f64>::zeros((k, width));
    for (row, center) in centers_1d.iter().enumerate() {
        out[(row, 0)] = *center;
    }
    if d <= 1 || k == 0 {
        return out;
    }

    // Historical R_d generalized golden-ratio fixed point: x^d = x + 1.
    // Thirty-two fixed-point refinements are retained deliberately for
    // bit-level continuity with the previous torch lane.  The count is not a
    // tuning knob: for the smallest routed multi-axis case (d=2) each step
    // roughly doubles correct digits near the fixed point, so 32 iterations is
    // beyond f64 mantissa resolution; larger d is contractive more quickly.
    let mut phi = 2.0_f64;
    for _ in 0..32 {
        phi = (1.0 + phi).powf(1.0 / d as f64);
    }
    for axis in 1..d {
        let alpha = (1.0 / phi).powi(axis as i32).rem_euclid(1.0);
        for row in 0..k {
            let idx = (row + 1) as f64;
            out[(row, axis)] = (idx * alpha + 0.5).rem_euclid(1.0);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Deterministic `@no_grad` routing anchors for the torch `softmax_topk` lane.
//
// The gate that trains `softmax_topk` (`ManifoldSAE.reconstruction_topk_gate`)
// leans on three seed-free clustering primitives to break the near-symmetric
// init on energy-degenerate data (issue #1282): a line-clustering anchor, a
// quadratic union-of-subspaces anchor (plus its transferable decision rule),
// and a residual-PC matching-pursuit commitment. Each is a pure numeric rule
// over detached inputs (the torch caller keeps the tape-coupled straight-through
// composition in Python and marshals f64 arrays across the FFI); the rules
// key only on the reconstruction residual / input direction covariance, never
// on hardcoded geometry. They all produce a hard one-hot assignment whose atom
// labels are arbitrary (the downstream routing metric is label-invariant), so
// singular-vector sign and eigenvalue-ordering conventions do not affect the
// contract — only the partition and the confidence gate do.
// ---------------------------------------------------------------------------

/// L2-normalize each row of `x` (norm floored at `1e-12`, matching the torch
/// `clamp_min(1e-12)`).
fn normalize_rows(x: ArrayView2<'_, f64>) -> Array2<f64> {
    let (n, d) = x.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let mut norm = 0.0;
        for j in 0..d {
            norm += x[(i, j)] * x[(i, j)];
        }
        let inv = 1.0 / norm.sqrt().max(1e-12);
        for j in 0..d {
            out[(i, j)] = x[(i, j)] * inv;
        }
    }
    out
}

/// Gram matrix `mᵀm` (`d × d`) of a row-major `(n, d)` block.
fn gram(m: ArrayView2<'_, f64>) -> Array2<f64> {
    let (n, d) = m.dim();
    let mut g = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in a..d {
            let mut s = 0.0;
            for i in 0..n {
                s += m[(i, a)] * m[(i, b)];
            }
            g[(a, b)] = s;
            g[(b, a)] = s;
        }
    }
    g
}

/// Eigenvector of the largest eigenvalue of a symmetric `d × d` matrix — the top
/// right singular vector of the block whose Gram matrix is `g`. Sign is
/// arbitrary (the callers use it only through `|·|` or a label-invariant sign
/// split). Returns `None` if the symmetric eigensolver does not converge,
/// mirroring the torch `except LinAlgError: return None` fallback.
fn top_eigvec(g: &Array2<f64>) -> Option<Array1<f64>> {
    let (evals, evecs) = crate::manifold::jacobi_symmetric(g).ok()?;
    let mut best = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in evals.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    Some(evecs.column(best).to_owned())
}

/// Lower median of `vals` — the element at index `(len - 1) / 2` of the sorted
/// values, matching `torch.median` (which returns the lower of the two middle
/// values for an even count).
fn lower_median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[(sorted.len() - 1) / 2]
}

/// Build an `(n, atoms)` one-hot from a per-row atom index.
fn onehot_from_assign(assign: &[usize], atoms: usize) -> Array2<f64> {
    let mut onehot = Array2::<f64>::zeros((assign.len(), atoms));
    for (i, &k) in assign.iter().enumerate() {
        onehot[(i, k)] = 1.0;
    }
    onehot
}

/// Deterministic line-clustering of the input row directions (issue #1282).
///
/// Returns `Some((onehot (N, atoms), confident))` where each row is assigned to
/// the cluster whose principal *line* (sign-invariant top right singular vector
/// of the cluster's unit rows) it aligns with most — a seed-free k-lines
/// clustering. `confident` requires balanced clusters (smallest holds at least
/// `0.6·N/atoms` rows) and a clear per-row line margin (mean gap between the
/// best and second-best alignment `≥ 0.25`). Returns `None` when there are too
/// few rows (`N < 2·atoms`), fewer than two atoms, or the eigensolver fails —
/// exactly the torch `(None, False)` cases.
#[must_use]
pub fn direction_cluster_anchor(
    x: ArrayView2<'_, f64>,
    n_atoms: usize,
    iters: usize,
) -> Option<(Array2<f64>, bool)> {
    let (n, d) = x.dim();
    if n_atoms < 2 || n < 2 * n_atoms {
        return None;
    }
    let xn = normalize_rows(x);
    // Deterministic farthest-line init: first line = top PC of the centered unit
    // rows; each next line = the row least aligned with the chosen lines.
    let mut colmean = vec![0.0; d];
    for i in 0..n {
        for c in 0..d {
            colmean[c] += xn[(i, c)];
        }
    }
    for c in colmean.iter_mut() {
        *c /= n as f64;
    }
    let mut centered = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for c in 0..d {
            centered[(i, c)] = xn[(i, c)] - colmean[c];
        }
    }
    let first = top_eigvec(&gram(centered.view()))?;
    let mut centers: Vec<Array1<f64>> = vec![first];
    for _ in 1..n_atoms {
        let mut best = 0usize;
        let mut best_align = f64::INFINITY;
        for i in 0..n {
            let mut aligned = f64::NEG_INFINITY;
            for c in &centers {
                let mut dot = 0.0;
                for k in 0..d {
                    dot += xn[(i, k)] * c[k];
                }
                aligned = aligned.max(dot.abs());
            }
            if aligned < best_align {
                best_align = aligned;
                best = i;
            }
        }
        centers.push(xn.row(best).to_owned());
    }
    let mut cmat = Array2::<f64>::zeros((n_atoms, d));
    for (k, c) in centers.iter().enumerate() {
        for j in 0..d {
            cmat[(k, j)] = c[j];
        }
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for i in 0..n {
            let mut best = 0usize;
            let mut best_align = f64::NEG_INFINITY;
            for k in 0..n_atoms {
                let mut dot = 0.0;
                for j in 0..d {
                    dot += xn[(i, j)] * cmat[(k, j)];
                }
                let a = dot.abs();
                if a > best_align {
                    best_align = a;
                    best = k;
                }
            }
            assign[i] = best;
        }
        for k in 0..n_atoms {
            let idx: Vec<usize> = (0..n).filter(|&i| assign[i] == k).collect();
            if idx.is_empty() {
                continue;
            }
            let mut gm = Array2::<f64>::zeros((d, d));
            for a in 0..d {
                for b in a..d {
                    let mut s = 0.0;
                    for &i in &idx {
                        s += xn[(i, a)] * xn[(i, b)];
                    }
                    gm[(a, b)] = s;
                    gm[(b, a)] = s;
                }
            }
            let vk = top_eigvec(&gm)?;
            for j in 0..d {
                cmat[(k, j)] = vk[j];
            }
        }
    }
    // `assign` is the argmax from the start of the final sweep (before that
    // sweep's center update), matching the torch lane's out-of-sync read of
    // `assign` (pre-update) against the final centers used for the margin.
    let mut counts = vec![0.0_f64; n_atoms];
    for &k in &assign {
        counts[k] += 1.0;
    }
    let min_count = counts.iter().cloned().fold(f64::INFINITY, f64::min);
    let balance = min_count / (n as f64 / n_atoms as f64);
    let mut margin_sum = 0.0;
    for i in 0..n {
        let mut top1 = f64::NEG_INFINITY;
        let mut top2 = f64::NEG_INFINITY;
        for k in 0..n_atoms {
            let mut dot = 0.0;
            for j in 0..d {
                dot += xn[(i, j)] * cmat[(k, j)];
            }
            let a = dot.abs();
            if a > top1 {
                top2 = top1;
                top1 = a;
            } else if a > top2 {
                top2 = a;
            }
        }
        margin_sum += top1 - top2;
    }
    let margin = margin_sum / n as f64;
    let confident = balance >= 0.6 && margin >= 0.25;
    Some((onehot_from_assign(&assign, n_atoms), confident))
}

/// Balanced quadratic split whose two clusters each form a low-rank subspace
/// (issue #1282). Searches deterministic `(i, j, threshold)` splits of the
/// signed cross-term features `x_i·x_j` (thresholds: the batch lower-median and
/// exactly zero, the sign of the cross product) and accepts the split whose two
/// clusters have a sharply smaller PCA tail-residual than every competing
/// feature. Returns `Some((onehot (N, 2), i, j, threshold))` only when the
/// winner is both absolutely low-residual (`≤ 0.05`) and uniquely better than
/// the cross-feature runner-up (`second ≥ max(3·best, best + 0.02)`); otherwise
/// `None`. The returned `(i, j, threshold)` is the transferable decision rule
/// (see [`apply_anchor_rule`]).
#[must_use]
pub fn quadratic_subspace_anchor(
    x: ArrayView2<'_, f64>,
    subspace_dim: usize,
) -> Option<(Array2<f64>, usize, usize, f64)> {
    let (n, d) = x.dim();
    if n < 4 || d < 2 {
        return None;
    }
    let xn = normalize_rows(x);
    let rank = subspace_dim.min(d);
    let min_count = (2.0_f64).max((0.3 * n as f64).ceil()) as usize;

    // Summed per-cluster PCA tail-residual (energy beyond `rank` principal
    // directions), normalized by `n`. `None` if either cluster is too small.
    let split_residual = |assign: &[usize]| -> Option<f64> {
        let mut total = 0.0;
        for k in 0..2 {
            let idx: Vec<usize> = (0..n).filter(|&i| assign[i] == k).collect();
            let count = idx.len();
            if count < min_count {
                return None;
            }
            let mut mean = vec![0.0; d];
            for &i in &idx {
                for c in 0..d {
                    mean[c] += xn[(i, c)];
                }
            }
            for c in mean.iter_mut() {
                *c /= count as f64;
            }
            let mut g = Array2::<f64>::zeros((d, d));
            for a in 0..d {
                for b in a..d {
                    let mut s = 0.0;
                    for &i in &idx {
                        s += (xn[(i, a)] - mean[a]) * (xn[(i, b)] - mean[b]);
                    }
                    g[(a, b)] = s;
                    g[(b, a)] = s;
                }
            }
            // `singular` has length `min(count, d)`; the torch tail sums the
            // squared singular values ranked `rank..`, i.e. the eigenvalues of
            // the Gram beyond the top `rank` (the remainder are structurally
            // zero). Only taken when `min(count, d) > rank`, else the tail is 0.
            let tail = if count.min(d) > rank {
                let (evals, _) = crate::manifold::jacobi_symmetric(&g).ok()?;
                let mut vals: Vec<f64> = evals.iter().map(|&e| e.max(0.0)).collect();
                vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                vals[rank..].iter().sum::<f64>()
            } else {
                0.0
            };
            total += tail / n as f64;
        }
        Some(total)
    };

    let mut best_assign: Option<Vec<usize>> = None;
    let mut best_resid: Option<f64> = None;
    let mut best_rule: Option<(usize, usize, f64)> = None;
    let mut second_resid: Option<f64> = None;
    for i in 0..d {
        for j in (i + 1)..d {
            let feature: Vec<f64> = (0..n).map(|r| xn[(r, i)] * xn[(r, j)]).collect();
            let median = lower_median(&feature);
            let mut feature_best: Option<f64> = None;
            let mut feature_best_assign: Option<Vec<usize>> = None;
            let mut feature_best_threshold = 0.0;
            for &threshold in &[median, 0.0] {
                let assign: Vec<usize> = feature
                    .iter()
                    .map(|&f| usize::from(f > threshold))
                    .collect();
                let count1 = assign.iter().filter(|&&a| a == 1).count();
                if count1.min(n - count1) < min_count {
                    continue;
                }
                let resid = match split_residual(&assign) {
                    Some(r) => r,
                    None => continue,
                };
                if feature_best.is_none_or(|fb| resid < fb) {
                    feature_best = Some(resid);
                    feature_best_assign = Some(assign);
                    feature_best_threshold = threshold;
                }
            }
            let (fb, fba) = match (feature_best, feature_best_assign) {
                (Some(fb), Some(fba)) => (fb, fba),
                _ => continue,
            };
            if best_resid.is_none_or(|br| fb < br) {
                if let Some(br) = best_resid {
                    if second_resid.is_none_or(|sr| br < sr) {
                        second_resid = Some(br);
                    }
                }
                best_resid = Some(fb);
                best_assign = Some(fba);
                best_rule = Some((i, j, feature_best_threshold));
            } else if second_resid.is_none_or(|sr| fb < sr) {
                second_resid = Some(fb);
            }
        }
    }

    let (best_assign, best, second) = match (best_assign, best_resid, second_resid) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return None,
    };
    let confident = best <= 0.05 && second >= (3.0 * best).max(best + 0.02);
    if !confident {
        return None;
    }
    let (i, j, threshold) = best_rule.unwrap();
    Some((onehot_from_assign(&best_assign, 2), i, j, threshold))
}

/// Route an arbitrary batch by a cached quadratic-subspace decision rule.
///
/// Applies the persisted `(i, j, threshold)` split of `x_i·x_j` (on L2-normalized
/// rows) to produce an `(N, 2)` one-hot, so held-out rows are routed by the same
/// high-margin union-of-subspaces discriminant that anchored training. `i` and
/// `j` are assumed in-bounds (the caller guards `i, j < d`).
#[must_use]
pub fn apply_anchor_rule(x: ArrayView2<'_, f64>, i: usize, j: usize, threshold: f64) -> Array2<f64> {
    let n = x.nrows();
    let xn = normalize_rows(x);
    let mut onehot = Array2::<f64>::zeros((n, 2));
    for r in 0..n {
        let feature = xn[(r, i)] * xn[(r, j)];
        onehot[(r, usize::from(feature > threshold))] = 1.0;
    }
    onehot
}

/// Residual-PC commitment one-hot for the early training window (issue #1282):
/// a seed-free port of the closed-form lane's
/// `reseed_atoms_onto_distinct_residual_pcs`.
///
/// Phase 1 (`step < commit_steps/2`) commits all rows to atom 0 (the
/// decoder-harmonic penalty then confines it to the manifolds' shared averaged
/// plane, so its residual carries their difference). Phase 2 splits rows: during
/// a short seed window by the sign of the top residual principal component of
/// atom 0's fit (with a lower-median fallback when the sign split is degenerate),
/// then by which atom currently reconstructs each row with the smaller residual
/// (with a residual-gap median balance guard). Keys only on the reconstruction
/// residual. Returns `None` for the non-committing configuration (`step ≥
/// commit_steps`) or an eigensolver failure; atom labels are arbitrary.
#[must_use]
pub fn matching_pursuit_commit(
    x: ArrayView2<'_, f64>,
    per_atom_recon: ArrayView3<'_, f64>,
    code: ArrayView2<'_, f64>,
    step: usize,
    commit_steps: usize,
    n_atoms: usize,
) -> Option<Array2<f64>> {
    let (n, d) = x.dim();
    let f = n_atoms;
    if step >= commit_steps {
        return None;
    }
    let phase1_end = (commit_steps / 2).max(1);
    if step < phase1_end {
        // Phase 1: everything to atom 0.
        let assign = vec![0usize; n];
        return Some(onehot_from_assign(&assign, f));
    }
    // Per-row, per-atom best non-negative scalar fit residual.
    let mut resid = Array2::<f64>::zeros((n, f));
    for i in 0..n {
        for k in 0..f {
            let mut s = 0.0;
            for c in 0..d {
                let diff = code[(i, k)] * per_atom_recon[(i, k, c)] - x[(i, c)];
                s += diff * diff;
            }
            resid[(i, k)] = s;
        }
    }
    let seed_steps = ((commit_steps - phase1_end) / 4).max(2);
    let assign: Vec<usize> = if step < phase1_end + seed_steps {
        // Seed window: split by the sign of atom 0's top residual PC.
        let mut resid0 = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for c in 0..d {
                resid0[(i, c)] = code[(i, 0)] * per_atom_recon[(i, 0, c)] - x[(i, c)];
            }
        }
        let mut mean = vec![0.0; d];
        for i in 0..n {
            for c in 0..d {
                mean[c] += resid0[(i, c)];
            }
        }
        for c in mean.iter_mut() {
            *c /= n as f64;
        }
        let mut rd = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for c in 0..d {
                rd[(i, c)] = resid0[(i, c)] - mean[c];
            }
        }
        let vh0 = top_eigvec(&gram(rd.view()))?;
        let proj: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = 0.0;
                for c in 0..d {
                    s += rd[(i, c)] * vh0[c];
                }
                s
            })
            .collect();
        let mut a: Vec<usize> = proj.iter().map(|&p| usize::from(p > 0.0)).collect();
        let sum1: usize = a.iter().sum();
        if sum1 == 0 || sum1 == n {
            let med = lower_median(&proj);
            a = proj.iter().map(|&p| usize::from(p > med)).collect();
        }
        a
    } else {
        // Residual-energy assignment with a residual-gap median balance guard.
        let gap: Vec<f64> = (0..n).map(|i| resid[(i, 0)] - resid[(i, 1)]).collect();
        let mut a: Vec<usize> = gap.iter().map(|&g| usize::from(g > 0.0)).collect();
        let count1: usize = a.iter().sum();
        if count1.min(n - count1) < n / 4 {
            let med = lower_median(&gap);
            a = gap.iter().map(|&g| usize::from(g > med)).collect();
        }
        a
    };
    Some(onehot_from_assign(&assign, f))
}

/// Exponential-moving-average update of the per-row assignment accumulator.
///
/// `prev[n, k]` is the current accumulator and `signal[n, k]` the freshly
/// observed per-row assignment distribution (a commitment one-hot or the soft
/// responsibilities); both are detached, non-negative `(N, F)` matrices of the
/// same shape. Returns the blended accumulator `beta * prev + (1 - beta) *
/// signal`. This is the sole arithmetic the torch `_update_assign_ema` used to
/// perform inline; the Python side keeps the stateful orchestration (lazy
/// sizing, reset on a row-count change, the training-only guard) and delegates
/// only this recurrence so the EMA math lives in exactly one place.
#[must_use]
pub fn assign_ema_update(
    prev: ArrayView2<'_, f64>,
    signal: ArrayView2<'_, f64>,
    beta: f64,
) -> Array2<f64> {
    let one_minus = 1.0 - beta;
    let mut out = prev.to_owned();
    out.zip_mut_with(&signal, |p, &s| *p = beta * *p + one_minus * s);
    out
}

#[cfg(test)]
mod tests {
    use super::{
        apply_anchor_rule, assign_ema_update, direction_cluster_anchor, duchon_centers_nd,
        matching_pursuit_commit, quadratic_subspace_anchor, sinkhorn_balance_bias,
    };
    use ndarray::{array, Array2, Array3};

    #[test]
    fn duchon_centers_nd_preserves_first_axis_and_shape() {
        let centers = array![0.0, 0.5, 1.0];
        let lifted = duchon_centers_nd(centers.view(), 3);
        assert_eq!(lifted.shape(), &[3, 3]);
        assert_eq!(lifted.column(0), centers);
    }

    #[test]
    fn assign_ema_update_blends_prev_and_signal() {
        // beta*prev + (1-beta)*signal, elementwise, shape-preserving.
        let prev = array![[1.0, 0.0], [0.0, 1.0]];
        let signal = array![[0.0, 1.0], [1.0, 0.0]];
        let out = assign_ema_update(prev.view(), signal.view(), 0.75);
        assert_eq!(out.shape(), &[2, 2]);
        assert!((out[(0, 0)] - 0.75).abs() < 1e-12);
        assert!((out[(0, 1)] - 0.25).abs() < 1e-12);
        assert!((out[(1, 0)] - 0.25).abs() < 1e-12);
        assert!((out[(1, 1)] - 0.75).abs() < 1e-12);
        // beta = 1 keeps the accumulator; beta = 0 replaces it with the signal.
        let keep = assign_ema_update(prev.view(), signal.view(), 1.0);
        assert_eq!(keep, prev);
        let replace = assign_ema_update(prev.view(), signal.view(), 0.0);
        assert_eq!(replace, signal);
    }

    #[test]
    fn sinkhorn_balance_bias_equalizes_atom_usage() {
        // Two atoms with a strong per-row preference for atom 0 (column 0 much
        // larger). Unbalanced usage would concentrate on atom 0; the Sinkhorn
        // potentials must push the mean column mass toward 1/K = 0.5.
        let mut scores = Array2::<f64>::zeros((64, 2));
        for i in 0..64 {
            scores[(i, 0)] = 2.0;
            scores[(i, 1)] = -2.0;
        }
        let bias = sinkhorn_balance_bias(scores.view(), 12);
        // Recompute the balanced column marginal.
        let mut usage = [0.0_f64; 2];
        for i in 0..64 {
            let a = scores[(i, 0)] + bias[0];
            let b = scores[(i, 1)] + bias[1];
            let m = a.max(b);
            let (ea, eb) = ((a - m).exp(), (b - m).exp());
            let s = ea + eb;
            usage[0] += ea / s;
            usage[1] += eb / s;
        }
        usage[0] /= 64.0;
        usage[1] /= 64.0;
        assert!(
            (usage[0] - 0.5).abs() < 0.05 && (usage[1] - 0.5).abs() < 0.05,
            "sinkhorn usage not balanced: {usage:?}"
        );
        // Gauge fixed: potentials are mean-centered.
        assert!(bias.mean().unwrap().abs() < 1e-9);
    }

    #[test]
    fn sinkhorn_balance_bias_trivial_for_single_atom() {
        let scores = Array2::<f64>::ones((8, 1));
        let bias = sinkhorn_balance_bias(scores.view(), 12);
        assert_eq!(bias.len(), 1);
        assert_eq!(bias[0], 0.0);
    }

    #[test]
    fn duchon_centers_nd_handles_empty_and_one_dimensional_inputs() {
        let empty = array![];
        assert_eq!(duchon_centers_nd(empty.view(), 4).shape(), &[0, 4]);
        let centers = array![0.25, 0.75];
        assert_eq!(
            duchon_centers_nd(centers.view(), 1),
            centers.into_shape_clone((2, 1)).unwrap()
        );
    }

    // Two disjoint circles living in orthogonal 2-planes of R^4: rows 0..m on the
    // (0,1)-plane, rows m..2m on the (2,3)-plane. The two ambient direction
    // subspaces are orthogonal, so line clustering must recover the partition and
    // flag it confident.
    fn two_orthogonal_circles(m: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((2 * m, 4));
        for i in 0..m {
            let t = std::f64::consts::TAU * (i as f64) / (m as f64);
            x[(i, 0)] = t.cos();
            x[(i, 1)] = t.sin();
            x[(m + i, 2)] = t.cos();
            x[(m + i, 3)] = t.sin();
        }
        x
    }

    #[test]
    fn direction_cluster_anchor_returns_one_hot_partition() {
        // Structural invariants any faithful port must satisfy: a valid batch
        // yields an `(N, atoms)` one-hot with every row assigned to exactly one
        // atom. The exact clustering outcome and the balance/margin confidence
        // verdict on the two-circle fixtures are the behavioral contract of the
        // #1282 pytest suite (which builds the wheel), not asserted here.
        let x = two_orthogonal_circles(48);
        let (onehot, _confident) = direction_cluster_anchor(x.view(), 2, 25)
            .expect("valid batch must return an assignment");
        assert_eq!(onehot.shape(), &[96, 2]);
        for i in 0..96 {
            assert!(
                (onehot[(i, 0)] + onehot[(i, 1)] - 1.0).abs() < 1e-12,
                "row {i} is not one-hot",
            );
        }
    }

    #[test]
    fn direction_cluster_anchor_rejects_too_few_rows() {
        // Fewer than 2·atoms rows → None (torch `(None, False)`).
        let x = two_orthogonal_circles(1);
        assert!(direction_cluster_anchor(x.view(), 2, 25).is_none());
    }

    // Energy-degenerate signed circles: x = (cosθ, sinθ, cosθ, ±sinθ). Both
    // manifolds have identical per-coordinate energy; only the sign of the
    // channel-1×channel-3 cross term (sinθ·(±sinθ) = ±sin²θ) separates them.
    fn signed_circles(m: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((2 * m, 4));
        for i in 0..m {
            let t = std::f64::consts::TAU * (i as f64 + 0.37) / (m as f64);
            let (c, s) = (t.cos(), t.sin());
            x[(i, 0)] = c;
            x[(i, 1)] = s;
            x[(i, 2)] = c;
            x[(i, 3)] = s;
            x[(m + i, 0)] = c;
            x[(m + i, 1)] = s;
            x[(m + i, 2)] = c;
            x[(m + i, 3)] = -s;
        }
        x
    }

    #[test]
    fn quadratic_subspace_anchor_splits_signed_circles() {
        let x = signed_circles(64);
        let (onehot, i, j, _threshold) = quadratic_subspace_anchor(x.view(), 2)
            .expect("energy-degenerate circles must yield a confident quadratic split");
        assert_eq!(onehot.shape(), &[128, 2]);
        // The discriminating cross term couples channels 1 and 3.
        assert!(
            (i, j) == (1, 3) || (i, j) == (3, 1),
            "unexpected discriminating pair ({i}, {j})"
        );
        let first_atom = if onehot[(0, 0)] > 0.5 { 0 } else { 1 };
        for r in 0..64 {
            assert!(onehot[(r, first_atom)] > 0.5, "+s branch row {r} misrouted");
        }
        for r in 64..128 {
            assert!(onehot[(r, 1 - first_atom)] > 0.5, "-s branch row {r} misrouted");
        }
    }

    #[test]
    fn apply_anchor_rule_matches_a_quadratic_split() {
        let x = signed_circles(64);
        let (onehot, i, j, threshold) =
            quadratic_subspace_anchor(x.view(), 2).expect("confident split");
        let applied = apply_anchor_rule(x.view(), i, j, threshold);
        assert_eq!(applied, onehot, "applied rule must reproduce the anchor split");
    }

    #[test]
    fn matching_pursuit_commit_phase1_routes_all_to_atom0() {
        let n = 20;
        let x = signed_circles(n / 2);
        let recon = Array3::<f64>::zeros((n, 2, 4));
        let code = Array2::<f64>::zeros((n, 2));
        // step 0 with commit_steps 8 → phase 1 (step < 4): everything to atom 0.
        let onehot = matching_pursuit_commit(x.view(), recon.view(), code.view(), 0, 8, 2)
            .expect("phase 1 always commits");
        assert_eq!(onehot.shape(), &[n, 2]);
        for r in 0..n {
            assert!(onehot[(r, 0)] > 0.5 && onehot[(r, 1)] < 0.5);
        }
    }

    #[test]
    fn matching_pursuit_commit_returns_none_after_window() {
        let n = 20;
        let x = signed_circles(n / 2);
        let recon = Array3::<f64>::zeros((n, 2, 4));
        let code = Array2::<f64>::zeros((n, 2));
        assert!(matching_pursuit_commit(x.view(), recon.view(), code.view(), 8, 8, 2).is_none());
    }
}
