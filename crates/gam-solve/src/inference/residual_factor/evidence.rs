//! Laplace evidence of one factor rank of the structured residual model.
//!
//! The rows of bin `b` (of `B'` occupied activity bins) are modelled as
//! `r_n ~ N(0, Σ_b)` with `Σ_b = c_b·ΛΛᵀ + D`, `D = diag(e^η)`. The evidence of
//! rank `r` is the marginal likelihood
//!
//! ```text
//!     Z_r = ∫ Π_b N(R_b | 0, Σ_b) · π(Λ | D) · π(c) dΛ dη dc
//! ```
//!
//! under the priors
//!
//! * `η` flat (the same improper factor in every rank, so it cancels in the
//!   rank comparison);
//! * `Λ | D` matrix-variate Cauchy (matrix-t, one degree of freedom) centred
//!   at `Λ = 0` with row scale `D` and column scale `I_r`:
//!   `π(Λ|D) = C_r · |D|^{-r/2} · |I_r + ΛᵀD⁻¹Λ|^{-(p+r)/2}` — proper, centred
//!   on "no factor", invariant under `Λ → ΛQ`, and scale-equivariant with the
//!   noise it is measured against;
//! * the activity law `w_b = n_b c_b / n` (the row-mean-one constraint puts
//!   `w` on the simplex) Dirichlet with concentrations `α_b = B'·n_b/n`: the
//!   flat Dirichlet's total concentration `B'`, with its mode at the null law
//!   `c ≡ 1`.
//!
//! The integral runs over the factor's orbit space. `Λ` and `ΛQ` (`Q ∈ O(r)`)
//! carry the same likelihood and prior, so with the rows permuted so that the
//! leading `r × r` block is lower triangular with positive diagonal,
//! `Λ = L·Q` and `dΛ = Π_k L_kk^{r−k-1} dL dQ` (0-indexed `k`), and
//! `∫ dQ = Vol(O(r)) = 2^r π^{r²/2} / Γ_r(r/2)`. The chart stores the
//! diagonal as `t_k = ln L_kk` and the activity law as `w = softmax([κ, 0])`,
//! so the integrand over `θ = (η, L, κ)` is smooth and unconstrained, and the
//! Laplace approximation is taken at its certified mode with the exact
//! analytic Hessian.
//!
//! The data are standardized channel-wise before the solve (`s_j² = S_jj/n`).
//! Every prior above is scale-equivariant, so the raw-data evidence is exactly
//! the standardized evidence less `n·Σ_j ln s_j`.

use ndarray::{Array1, Array2, ArrayView2};

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use gam_linalg::pairwise_reduce::par_deterministic_block_fold;
use gam_linalg::roundoff::accumulation_growth;
use opt::{
    Arc, DecrementBands, FallbackPolicy, FirstOrderObjective, FirstOrderSample,
    ObjectiveEvalError, SecondOrderObjective, SecondOrderSample, TerminationReason,
    ZerothOrderObjective,
};
use statrs::function::gamma::ln_gamma;

/// Per-bin second moments of the standardized residuals, over the occupied
/// activity bins (slots) in ascending bin order.
pub(super) struct BinnedMoments {
    /// Residual width `p`.
    pub(super) p: usize,
    /// Row count `n`.
    pub(super) n: f64,
    /// Row count `n_b` of each slot.
    pub(super) counts: Vec<f64>,
    /// Standardized scatter `S̃_b = diag(s)⁻¹ (Σ_{n∈b} r_n r_nᵀ) diag(s)⁻¹`.
    pub(super) scatter: Vec<Array2<f64>>,
    /// Slot of every row.
    pub(super) row_slot: Vec<usize>,
    /// Channel scales `s_j = √(S_jj / n)`.
    pub(super) scale: Array1<f64>,
}

/// Accumulate the per-bin scatter of `r` and standardize it. A channel whose
/// second moment is exactly zero carries no noise scale for the model to
/// measure against, and is refused.
pub(super) fn binned_moments(
    r: ArrayView2<'_, f64>,
    row_bin: &[usize],
    bins: usize,
) -> Result<BinnedMoments, String> {
    let n = r.nrows();
    let p = r.ncols();
    let (mut scatter, counts) = par_deterministic_block_fold(
        n,
        |range| {
            let mut acc = vec![Array2::<f64>::zeros((p, p)); bins];
            let mut cnt = vec![0.0_f64; bins];
            for i in range {
                let b = row_bin[i];
                cnt[b] += 1.0;
                let s = &mut acc[b];
                for a in 0..p {
                    let ra = r[[i, a]];
                    for c in 0..=a {
                        s[[a, c]] += ra * r[[i, c]];
                    }
                }
            }
            (acc, cnt)
        },
        |(mut acc, mut cnt), (part, part_cnt)| {
            for b in 0..bins {
                acc[b] += &part[b];
                cnt[b] += part_cnt[b];
            }
            (acc, cnt)
        },
    )
    .ok_or_else(|| "structured residual evidence: no rows".to_string())?;
    for s in scatter.iter_mut() {
        for a in 0..p {
            for c in 0..a {
                s[[c, a]] = s[[a, c]];
            }
        }
    }
    let mut slot_of_bin = vec![usize::MAX; bins];
    let mut slot_counts = Vec::new();
    let mut slot_scatter = Vec::new();
    for b in 0..bins {
        if counts[b] > 0.0 {
            slot_of_bin[b] = slot_counts.len();
            slot_counts.push(counts[b]);
            slot_scatter.push(std::mem::replace(&mut scatter[b], Array2::zeros((0, 0))));
        }
    }
    let nf = n as f64;
    let mut scale = Array1::<f64>::zeros(p);
    for j in 0..p {
        let total: f64 = slot_scatter.iter().map(|s| s[[j, j]]).sum();
        if !(total > 0.0) {
            return Err(format!(
                "structured residual evidence: residual channel {j} is identically zero, so \
                 it has no noise scale to model"
            ));
        }
        scale[j] = (total / nf).sqrt();
    }
    for s in slot_scatter.iter_mut() {
        for a in 0..p {
            for c in 0..p {
                s[[a, c]] /= scale[a] * scale[c];
            }
        }
    }
    let row_slot = row_bin.iter().map(|&b| slot_of_bin[b]).collect();
    Ok(BinnedMoments {
        p,
        n: nf,
        counts: slot_counts,
        scatter: slot_scatter,
        row_slot,
        scale,
    })
}

/// The largest factor rank a `p`-channel factor model identifies: the
/// Ledermann bound, the largest `r` whose free parameter count
/// `pr + p − r(r−1)/2` does not exceed the `p(p+1)/2` covariance entries it is
/// fitted to, i.e. `(p − r)² ≥ p + r`. Above it the orbit space has
/// directions the likelihood does not see, and a Laplace evidence there is not
/// defined.
pub(super) fn ledermann_bound(p: usize) -> usize {
    let mut r = 0usize;
    while r < p && (p - r - 1) * (p - r - 1) >= p + r + 1 {
        r += 1;
    }
    r
}

/// `ln Γ_r(a) = r(r−1)/4 · ln π + Σ_{j=0}^{r−1} ln Γ(a − j/2)`.
pub(super) fn ln_multivariate_gamma(r: usize, a: f64) -> f64 {
    let rf = r as f64;
    let mut out = rf * (rf - 1.0) / 4.0 * std::f64::consts::PI.ln();
    for j in 0..r {
        out += ln_gamma(a - j as f64 / 2.0);
    }
    out
}

/// `ln Vol(O(r)) = r ln 2 + (r²/2) ln π − ln Γ_r(r/2)`, the Haar volume the
/// `Λ = L·Q` chart integrates out.
pub(super) fn log_orthogonal_group_volume(r: usize) -> f64 {
    let rf = r as f64;
    rf * std::f64::consts::LN_2 + rf * rf / 2.0 * std::f64::consts::PI.ln()
        - ln_multivariate_gamma(r, rf / 2.0)
}

/// `ln C_r` of the `p × r` matrix-Cauchy prior:
/// `ln Γ_r((p+r)/2) − (pr/2) ln π − ln Γ_r(r/2)`.
pub(super) fn log_matrix_cauchy_normalizer(p: usize, r: usize) -> f64 {
    let (pf, rf) = (p as f64, r as f64);
    ln_multivariate_gamma(r, (pf + rf) / 2.0)
        - pf * rf / 2.0 * std::f64::consts::PI.ln()
        - ln_multivariate_gamma(r, rf / 2.0)
}

/// One free entry of the factor chart: chart row `i` is data row `row`, and a
/// diagonal entry (`i == col`) is stored as its logarithm.
#[derive(Clone, Copy, Debug)]
struct ChartEntry {
    row: usize,
    col: usize,
    diagonal: bool,
}

/// The log posterior of one rank over the chart `θ = [η (p), L entries, κ]`.
pub(super) struct EvidenceProblem<'a> {
    moments: &'a BinnedMoments,
    rank: usize,
    entries: Vec<ChartEntry>,
    has_kappa: bool,
    dirichlet_alpha: Vec<f64>,
    dirichlet_log_norm: f64,
    log_prior_norm: f64,
    growth: f64,
    start: Array1<f64>,
}

/// Value, gradient, Hessian and rounding bands of the log posterior (the
/// quantity maximized, not the solver's cost).
pub(super) struct EvidenceEval {
    pub(super) value: f64,
    pub(super) gradient: Array1<f64>,
    pub(super) hessian: Option<Array2<f64>>,
    bands: DecrementBands,
}

/// The unpacked model at a chart point, in standardized units.
struct Unpacked {
    lambda: Array2<f64>,
    diagonal: Array1<f64>,
    weights: Vec<f64>,
    log_weights: Vec<f64>,
    scale: Vec<f64>,
}

impl<'a> EvidenceProblem<'a> {
    pub(super) fn new(moments: &'a BinnedMoments, rank: usize) -> Result<Self, String> {
        let p = moments.p;
        let slots = moments.counts.len();
        if rank > ledermann_bound(p) {
            return Err(format!(
                "structured residual evidence: rank {rank} exceeds the identifiable bound {} \
                 for p = {p}",
                ledermann_bound(p)
            ));
        }
        let has_kappa = rank > 0 && slots > 1;

        // Warm start: probabilistic PCA of the pooled standardized moment.
        let mut pooled = Array2::<f64>::zeros((p, p));
        for s in &moments.scatter {
            pooled += s;
        }
        pooled /= moments.n;
        let (evals, evecs) = pooled
            .eigh(Side::Lower)
            .map_err(|e| format!("structured residual evidence: warm-start eigh: {e:?}"))?;
        let mut eta0 = Array1::<f64>::zeros(p);
        let mut lambda0 = Array2::<f64>::zeros((p, rank));
        if rank == 0 {
            for j in 0..p {
                eta0[j] = pooled[[j, j]].ln();
            }
        } else {
            let sigma2 = evals.iter().take(p - rank).sum::<f64>() / (p - rank) as f64;
            if !(sigma2 > 0.0) {
                return Err(format!(
                    "structured residual evidence: the pooled moment's trailing spectrum has \
                     mean {sigma2}, so no rank-{rank} warm start exists"
                ));
            }
            for k in 0..rank {
                let col = p - 1 - k;
                let excess = evals[col] - sigma2;
                if !(excess > 0.0) {
                    return Err(format!(
                        "structured residual evidence: eigenvalue {} of the pooled moment does \
                         not exceed the trailing mean {sigma2}, so no rank-{rank} warm start \
                         exists",
                        evals[col]
                    ));
                }
                let amp = excess.sqrt();
                for i in 0..p {
                    lambda0[[i, k]] = evecs[[i, col]] * amp;
                }
            }
            eta0.fill(sigma2.ln());
        }

        // Row order: pivoted Gram–Schmidt on the rows of Λ0 (largest residual
        // norm first, ties to the lower index), so the leading block of the
        // chart is lower triangular with the largest attainable diagonal.
        let mut order: Vec<usize> = Vec::with_capacity(p);
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(rank);
        let mut residual_rows: Vec<Array1<f64>> =
            (0..p).map(|i| lambda0.row(i).to_owned()).collect();
        for _ in 0..rank {
            let mut best: Option<(usize, f64)> = None;
            for i in 0..p {
                if order.contains(&i) {
                    continue;
                }
                let norm2 = residual_rows[i].dot(&residual_rows[i]);
                if best.map_or(true, |(_, b)| norm2 > b) {
                    best = Some((i, norm2));
                }
            }
            let Some((pick, norm2)) = best.filter(|&(_, norm2)| norm2 > 0.0) else {
                return Err(format!(
                    "structured residual evidence: the rank-{rank} warm start has linearly \
                     dependent columns"
                ));
            };
            let q = &residual_rows[pick] / norm2.sqrt();
            for i in 0..p {
                let proj = residual_rows[i].dot(&q);
                residual_rows[i] = &residual_rows[i] - &(&q * proj);
            }
            order.push(pick);
            basis.push(q);
        }
        for i in 0..p {
            if !order.contains(&i) {
                order.push(i);
            }
        }
        let mut entries = Vec::new();
        let mut start_entries = Vec::new();
        for (i, &row) in order.iter().enumerate() {
            for k in 0..(i + 1).min(rank) {
                let value = lambda0.row(row).dot(&basis[k]);
                let diagonal = i == k;
                entries.push(ChartEntry { row, col: k, diagonal });
                start_entries.push(if diagonal { value.ln() } else { value });
            }
        }

        let q = if has_kappa { slots - 1 } else { 0 };
        let mut start = Array1::<f64>::zeros(p + entries.len() + q);
        for j in 0..p {
            start[j] = eta0[j];
        }
        for (e, v) in start_entries.iter().enumerate() {
            start[p + e] = *v;
        }
        let (dirichlet_alpha, dirichlet_log_norm) = if has_kappa {
            let total = slots as f64;
            let alpha: Vec<f64> = moments
                .counts
                .iter()
                .map(|&nb| total * nb / moments.n)
                .collect();
            let norm = ln_gamma(total) - alpha.iter().map(|&a| ln_gamma(a)).sum::<f64>();
            // The Dirichlet mode `w_b = α_b / B' = n_b / n` is the null law
            // `c ≡ 1`, which is where the chart starts.
            for b in 0..q {
                start[p + entries.len() + b] = (alpha[b] / alpha[slots - 1]).ln();
            }
            (alpha, norm)
        } else {
            (Vec::new(), 0.0)
        };
        // One evaluation accumulates, per slot, the `p × p` moment products
        // and the `p × r`, `r × r` factor contractions, then the prior's `p`
        // and `r` sums.
        let growth =
            accumulation_growth(slots * (p * p + p * rank + rank * rank) + p + rank);
        Ok(Self {
            moments,
            rank,
            entries,
            has_kappa,
            dirichlet_alpha,
            dirichlet_log_norm,
            log_prior_norm: log_matrix_cauchy_normalizer(p, rank),
            growth,
            start,
        })
    }

    /// Chart dimension.
    pub(super) fn dim(&self) -> usize {
        self.start.len()
    }

    /// The warm start.
    pub(super) fn start(&self) -> &Array1<f64> {
        &self.start
    }

    fn unpack(&self, theta: &Array1<f64>) -> Unpacked {
        let p = self.moments.p;
        let slots = self.moments.counts.len();
        let mut lambda = Array2::<f64>::zeros((p, self.rank));
        for (e, entry) in self.entries.iter().enumerate() {
            let v = theta[p + e];
            lambda[[entry.row, entry.col]] = if entry.diagonal { v.exp() } else { v };
        }
        let diagonal = Array1::from_iter((0..p).map(|j| theta[j].exp()));
        let (weights, log_weights) = if self.has_kappa {
            let base = p + self.entries.len();
            let logits: Vec<f64> = (0..slots)
                .map(|b| if b + 1 < slots { theta[base + b] } else { 0.0 })
                .collect();
            let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_sum = top + logits.iter().map(|&l| (l - top).exp()).sum::<f64>().ln();
            let log_w: Vec<f64> = logits.iter().map(|&l| l - log_sum).collect();
            (log_w.iter().map(|&l| l.exp()).collect(), log_w)
        } else {
            let w: Vec<f64> = self.moments.counts.iter().map(|&nb| nb / self.moments.n).collect();
            let lw = w.iter().map(|&x| x.ln()).collect();
            (w, lw)
        };
        // Without a free activity law the scale is the null law `c ≡ 1`
        // exactly, not its rounded reconstruction from the weights.
        let scale = if self.has_kappa {
            (0..slots)
                .map(|b| self.moments.n / self.moments.counts[b] * weights[b])
                .collect()
        } else {
            vec![1.0; slots]
        };
        Unpacked {
            lambda,
            diagonal,
            weights,
            log_weights,
            scale,
        }
    }

    /// The log posterior over the chart (likelihood, both priors, the chart's
    /// log-Jacobian and every normalizer), its exact gradient, and, when asked,
    /// its exact Hessian, each with its rounding band.
    pub(super) fn evaluate(
        &self,
        theta: &Array1<f64>,
        want_hessian: bool,
    ) -> Result<EvidenceEval, String> {
        let p = self.moments.p;
        let r = self.rank;
        let slots = self.moments.counts.len();
        let pr = p * r;
        let du = pr + p + slots;
        let lam = |i: usize, k: usize| i * r + k;
        let eta = |j: usize| pr + j;
        let cvar = |b: usize| pr + p + b;
        let u = self.unpack(theta);
        let lambda = &u.lambda;
        let d = &u.diagonal;
        let ln_two_pi = (2.0 * std::f64::consts::PI).ln();

        let mut value = 0.0_f64;
        let mut magnitude = 0.0_f64;
        let mut gu = Array1::<f64>::zeros(du);
        let mut gmag = Array1::<f64>::zeros(du);
        let mut hu = want_hessian.then(|| Array2::<f64>::zeros((du, du)));
        let eye = Array2::<f64>::eye(p);
        let abs_lambda = lambda.mapv(f64::abs);

        // Likelihood, one slot at a time.
        for b in 0..slots {
            let nb = self.moments.counts[b];
            let cb = u.scale[b];
            let s = &self.moments.scatter[b];
            let mut sigma = lambda.dot(&lambda.t()) * cb;
            for j in 0..p {
                sigma[[j, j]] += d[j];
            }
            let chol = sigma.cholesky(Side::Lower).map_err(|e| {
                format!("structured residual evidence: Σ_{b} is not positive definite: {e:?}")
            })?;
            let log_det: f64 = chol.diag().iter().map(|&l| 2.0 * l.ln()).sum();
            let w = chol.solve_mat(&eye);
            let ws = w.dot(s);
            let tr_ws: f64 = (0..p).map(|j| ws[[j, j]]).sum();
            let m = ws.dot(&w);
            value += -0.5 * nb * log_det - 0.5 * tr_ws - 0.5 * nb * p as f64 * ln_two_pi;
            magnitude += 0.5 * nb * log_det.abs() + 0.5 * tr_ws.abs() + 0.5 * nb * p as f64 * ln_two_pi;
            let g = &m - &(&w * nb);
            let gl = g.dot(lambda);
            let abs_mg = &m.mapv(f64::abs) + &(w.mapv(f64::abs) * nb);
            let abs_mg_l = abs_mg.dot(&abs_lambda);
            let mut gc = 0.0_f64;
            let mut gc_mag = 0.0_f64;
            for i in 0..p {
                for k in 0..r {
                    gu[lam(i, k)] += cb * gl[[i, k]];
                    gmag[lam(i, k)] += cb * abs_mg_l[[i, k]];
                    gc += lambda[[i, k]] * gl[[i, k]];
                    gc_mag += abs_lambda[[i, k]] * abs_mg_l[[i, k]];
                }
            }
            for j in 0..p {
                gu[eta(j)] += 0.5 * d[j] * g[[j, j]];
                gmag[eta(j)] += 0.5 * d[j] * abs_mg[[j, j]];
            }
            gu[cvar(b)] = 0.5 * gc;
            gmag[cvar(b)] = 0.5 * gc_mag;

            if let Some(h) = hu.as_mut() {
                let y = &(&w * (0.5 * nb)) - &m;
                let wl = w.dot(lambda);
                let yl = y.dot(lambda);
                let ltwl = lambda.t().dot(&wl);
                let ltyl = lambda.t().dot(&yl);
                let wl_ltyl = wl.dot(&ltyl);
                let yl_ltwl = yl.dot(&ltwl);
                let c2 = cb * cb;
                for i in 0..p {
                    for k in 0..r {
                        let x = lam(i, k);
                        for j in 0..p {
                            for l in 0..r {
                                let mut v = c2
                                    * (wl[[i, l]] * yl[[j, k]]
                                        + wl[[j, k]] * yl[[i, l]]
                                        + w[[i, j]] * ltyl[[k, l]]
                                        + y[[i, j]] * ltwl[[k, l]]);
                                if k == l {
                                    v += cb * g[[i, j]];
                                }
                                h[[x, lam(j, l)]] += v;
                            }
                            let v = cb * d[j] * (w[[i, j]] * yl[[j, k]] + y[[i, j]] * wl[[j, k]]);
                            h[[x, eta(j)]] += v;
                            h[[eta(j), x]] += v;
                        }
                        let v = gl[[i, k]] + cb * (wl_ltyl[[i, k]] + yl_ltwl[[i, k]]);
                        h[[x, cvar(b)]] += v;
                        h[[cvar(b), x]] += v;
                    }
                }
                for i in 0..p {
                    for j in 0..p {
                        let mut v = d[i] * d[j] * w[[i, j]] * y[[i, j]];
                        if i == j {
                            v += 0.5 * d[j] * g[[j, j]];
                        }
                        h[[eta(i), eta(j)]] += v;
                    }
                    let v: f64 = d[i] * (0..r).map(|k| wl[[i, k]] * yl[[i, k]]).sum::<f64>();
                    h[[eta(i), cvar(b)]] += v;
                    h[[cvar(b), eta(i)]] += v;
                }
                let mut cc = 0.0_f64;
                for k in 0..r {
                    for l in 0..r {
                        cc += ltwl[[k, l]] * ltyl[[l, k]];
                    }
                }
                h[[cvar(b), cvar(b)]] += cc;
            }
        }

        // Matrix-Cauchy prior on Λ | D.
        value += self.log_prior_norm;
        magnitude += self.log_prior_norm.abs();
        let log_d_sum: f64 = (0..p).map(|j| theta[j]).sum();
        value -= r as f64 / 2.0 * log_d_sum;
        magnitude += r as f64 / 2.0 * (0..p).map(|j| theta[j].abs()).sum::<f64>();
        for j in 0..p {
            gu[eta(j)] -= r as f64 / 2.0;
            gmag[eta(j)] += r as f64 / 2.0;
        }
        if r > 0 {
            let coef = (p + r) as f64 / 2.0;
            let mut kmat = Array2::<f64>::eye(r);
            for k in 0..r {
                for l in 0..r {
                    kmat[[k, l]] += (0..p).map(|i| lambda[[i, k]] * lambda[[i, l]] / d[i]).sum::<f64>();
                }
            }
            let kchol = kmat.cholesky(Side::Lower).map_err(|e| {
                format!("structured residual evidence: prior capacitance not positive definite: {e:?}")
            })?;
            let log_det_k: f64 = kchol.diag().iter().map(|&l| 2.0 * l.ln()).sum();
            value -= coef * log_det_k;
            magnitude += coef * log_det_k.abs();
            let pk = kchol.solve_mat(&Array2::<f64>::eye(r));
            let a = lambda.dot(&pk);
            let rr = a.dot(&lambda.t());
            for i in 0..p {
                for k in 0..r {
                    gu[lam(i, k)] -= coef * 2.0 * a[[i, k]] / d[i];
                    gmag[lam(i, k)] += coef * 2.0 * a[[i, k]].abs() / d[i];
                }
                gu[eta(i)] += coef * rr[[i, i]] / d[i];
                gmag[eta(i)] += coef * rr[[i, i]].abs() / d[i];
            }
            if let Some(h) = hu.as_mut() {
                for i in 0..p {
                    for k in 0..r {
                        let x = lam(i, k);
                        for j in 0..p {
                            for l in 0..r {
                                let mut v = -(a[[i, l]] * a[[j, k]] + rr[[i, j]] * pk[[k, l]]) / d[j];
                                if i == j {
                                    v += pk[[k, l]];
                                }
                                h[[x, lam(j, l)]] -= coef * 2.0 / d[i] * v;
                            }
                            let mut v = 2.0 * rr[[i, j]] * a[[j, k]] / (d[i] * d[j]);
                            if i == j {
                                v -= 2.0 * a[[i, k]] / d[i];
                            }
                            h[[x, eta(j)]] -= coef * v;
                            h[[eta(j), x]] -= coef * v;
                        }
                    }
                    for j in 0..p {
                        let mut v = -rr[[i, j]] * rr[[i, j]] / (d[i] * d[j]);
                        if i == j {
                            v += rr[[i, i]] / d[i];
                        }
                        h[[eta(i), eta(j)]] -= coef * v;
                    }
                }
            }
        }

        // Chain to the chart θ = [η, L entries, κ].
        let m_entries = self.entries.len();
        let q = if self.has_kappa { slots - 1 } else { 0 };
        let dim = p + m_entries + q;
        let kbase = p + m_entries;
        let mut grad = Array1::<f64>::zeros(dim);
        let mut grad_mag = Array1::<f64>::zeros(dim);
        let uidx: Vec<usize> = self.entries.iter().map(|e| lam(e.row, e.col)).collect();
        let sfac: Vec<f64> = self
            .entries
            .iter()
            .map(|e| if e.diagonal { lambda[[e.row, e.col]] } else { 1.0 })
            .collect();
        for j in 0..p {
            grad[j] = gu[eta(j)];
            grad_mag[j] = gmag[eta(j)];
        }
        for (e, entry) in self.entries.iter().enumerate() {
            grad[p + e] = sfac[e] * gu[uidx[e]];
            grad_mag[p + e] = sfac[e].abs() * gmag[uidx[e]];
            if entry.diagonal {
                // Chart log-Jacobian Π_k L_kk^{r−k−1} · Π_k dL_kk/dt_k.
                let power = (r - entry.col) as f64;
                let t = theta[p + e];
                value += power * t;
                magnitude += (power * t).abs();
                grad[p + e] += power;
                grad_mag[p + e] += power;
            }
        }
        // Jacobian of c_b = (n/n_b) w_b in κ, and the Dirichlet prior.
        let jc = |b: usize, a: usize| -> f64 {
            let delta = if a == b { 1.0 } else { 0.0 };
            self.moments.n / self.moments.counts[b] * u.weights[b] * (delta - u.weights[a])
        };
        if self.has_kappa {
            let total = slots as f64;
            value += self.dirichlet_log_norm;
            magnitude += self.dirichlet_log_norm.abs();
            for b in 0..slots {
                value += self.dirichlet_alpha[b] * u.log_weights[b];
                magnitude += (self.dirichlet_alpha[b] * u.log_weights[b]).abs();
            }
            for a in 0..q {
                let mut g = self.dirichlet_alpha[a] - total * u.weights[a];
                let mut gm = self.dirichlet_alpha[a] + total * u.weights[a];
                for b in 0..slots {
                    g += gu[cvar(b)] * jc(b, a);
                    gm += gmag[cvar(b)] * jc(b, a).abs();
                }
                grad[kbase + a] = g;
                grad_mag[kbase + a] = gm;
            }
        }
        if !value.is_finite() || grad.iter().any(|v| !v.is_finite()) {
            return Err("structured residual evidence: non-finite log posterior".to_string());
        }

        let hessian = match hu {
            None => None,
            Some(h) => {
                let mut ht = Array2::<f64>::zeros((dim, dim));
                for i in 0..p {
                    for j in 0..p {
                        ht[[i, j]] = h[[eta(i), eta(j)]];
                    }
                }
                for (e, _) in self.entries.iter().enumerate() {
                    for j in 0..p {
                        let v = sfac[e] * h[[uidx[e], eta(j)]];
                        ht[[p + e, j]] = v;
                        ht[[j, p + e]] = v;
                    }
                    for (f, _) in self.entries.iter().enumerate() {
                        ht[[p + e, p + f]] = sfac[e] * sfac[f] * h[[uidx[e], uidx[f]]];
                    }
                    if self.entries[e].diagonal {
                        ht[[p + e, p + e]] += gu[uidx[e]] * sfac[e];
                    }
                }
                if self.has_kappa {
                    let total = slots as f64;
                    for a in 0..q {
                        for j in 0..p {
                            let v: f64 = (0..slots).map(|b| h[[eta(j), cvar(b)]] * jc(b, a)).sum();
                            ht[[kbase + a, j]] = v;
                            ht[[j, kbase + a]] = v;
                        }
                        for (e, _) in self.entries.iter().enumerate() {
                            let v: f64 = sfac[e]
                                * (0..slots).map(|b| h[[uidx[e], cvar(b)]] * jc(b, a)).sum::<f64>();
                            ht[[kbase + a, p + e]] = v;
                            ht[[p + e, kbase + a]] = v;
                        }
                        for e in 0..q {
                            let delta_ae = if a == e { 1.0 } else { 0.0 };
                            let mut v = -total * u.weights[a] * (delta_ae - u.weights[e]);
                            for b in 0..slots {
                                v += jc(b, a) * h[[cvar(b), cvar(b)]] * jc(b, e);
                                let wb = u.weights[b];
                                let (dab, deb) =
                                    (if a == b { 1.0 } else { 0.0 }, if e == b { 1.0 } else { 0.0 });
                                let second = self.moments.n / self.moments.counts[b]
                                    * (wb * (deb - u.weights[e]) * (dab - u.weights[a])
                                        - wb * u.weights[a] * (delta_ae - u.weights[e]));
                                v += gu[cvar(b)] * second;
                            }
                            ht[[kbase + a, kbase + e]] = v;
                        }
                    }
                }
                if ht.iter().any(|v| !v.is_finite()) {
                    return Err("structured residual evidence: non-finite Hessian".to_string());
                }
                Some(ht)
            }
        };
        let hessian_band = hessian
            .as_ref()
            .map_or(0.0, |h| self.growth * h.iter().map(|v| v * v).sum::<f64>().sqrt());
        Ok(EvidenceEval {
            value,
            gradient: grad,
            hessian,
            bands: DecrementBands {
                objective: self.growth * magnitude,
                tolerance: self.growth * magnitude,
                gradient: grad_mag * self.growth,
                hessian: hessian_band,
            },
        })
    }
}

/// The solver's view: it minimizes the negated log posterior.
struct EvidenceObjective<'p, 'a> {
    problem: &'p EvidenceProblem<'a>,
}

impl EvidenceEval {
    fn into_cost_sample(self) -> SecondOrderSample {
        SecondOrderSample {
            value: -self.value,
            gradient: -self.gradient,
            hessian: self.hessian.map(|h| -h),
            decrement_bands: Some(self.bands),
        }
    }
}

impl ZerothOrderObjective for EvidenceObjective<'_, '_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.problem
            .evaluate(x, false)
            .map(|e| -e.value)
            .map_err(ObjectiveEvalError::recoverable)
    }
}

impl FirstOrderObjective for EvidenceObjective<'_, '_> {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.problem
            .evaluate(x, false)
            .map(|e| FirstOrderSample {
                value: -e.value,
                gradient: -e.gradient,
            })
            .map_err(ObjectiveEvalError::recoverable)
    }
}

impl SecondOrderObjective for EvidenceObjective<'_, '_> {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError> {
        self.problem
            .evaluate(x, true)
            .map(EvidenceEval::into_cost_sample)
            .map_err(ObjectiveEvalError::recoverable)
    }
}

/// The certified fit of one rank, in raw residual units.
pub(super) struct RankFit {
    pub(super) log_evidence: f64,
    /// `Λ` with `D^{-1/2}Λ` column-orthogonal, columns in descending energy,
    /// each column's largest-magnitude entry positive.
    pub(super) lambda: Array2<f64>,
    pub(super) diagonal: Array1<f64>,
    /// Activity scale `c_b` of every slot.
    pub(super) slot_scale: Vec<f64>,
}

/// Fit one rank to its certified posterior mode and return its Laplace
/// evidence. A mode the solver cannot certify, or one whose negated Hessian is
/// not positive definite, is refused: there is no evidence to report for it.
pub(super) fn fit_rank(moments: &BinnedMoments, rank: usize) -> Result<RankFit, String> {
    let p = moments.p;
    let problem = EvidenceProblem::new(moments, rank)?;
    let x0 = problem.start().clone();
    let seed = problem.evaluate(&x0, true)?.into_cost_sample();
    // Cubic regularization, not a trust region: the probabilistic-PCA start is
    // a stationary point of the factor likelihood whenever the planted
    // loadings are channel-symmetric, and there it is a saddle (one direction
    // of positive curvature in the log posterior). ARC's exact hard-case step
    // leaves along that direction; a Steihaug trust region sees a vanishing
    // gradient and cannot.
    let solution = Arc::new(x0.clone(), EvidenceObjective { problem: &problem })
        .with_fallback_policy(FallbackPolicy::Never)
        .with_initial_sample(x0, seed)
        .run()
        .map_err(|e| format!("structured residual evidence: rank {rank} mode search: {e}"))?;
    if !matches!(
        solution.termination,
        TerminationReason::NewtonDecrementCertified { .. }
    ) {
        return Err(format!(
            "structured residual evidence: rank {rank} mode search stopped without a \
             certified mode ({:?})",
            solution.termination
        ));
    }
    let theta = solution.final_point;
    let at = problem.evaluate(&theta, true)?;
    let hessian = at
        .hessian
        .as_ref()
        .ok_or_else(|| "structured residual evidence: Hessian missing".to_string())?;
    let neg = hessian.mapv(|v| -v);
    let chol = neg.cholesky(Side::Lower).map_err(|e| {
        format!(
            "structured residual evidence: rank {rank} mode is not a strict maximum (negated \
             Hessian not positive definite: {e:?})"
        )
    })?;
    let log_det: f64 = chol.diag().iter().map(|&l| 2.0 * l.ln()).sum();
    let dim = problem.dim() as f64;
    let log_scale_sum: f64 = moments.scale.iter().map(|s| s.ln()).sum();
    let log_evidence = at.value
        + log_orthogonal_group_volume(rank)
        + 0.5 * dim * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * log_det
        - moments.n * log_scale_sum;

    let u = problem.unpack(&theta);
    let mut lambda = u.lambda;
    let mut diagonal = u.diagonal;
    for i in 0..p {
        let s = moments.scale[i];
        for k in 0..rank {
            lambda[[i, k]] *= s;
        }
        diagonal[i] *= s * s;
    }
    if rank > 0 {
        let mut b = lambda.clone();
        for i in 0..p {
            let w = diagonal[i].sqrt();
            for k in 0..rank {
                b[[i, k]] /= w;
            }
        }
        let (_, vecs) = b
            .t()
            .dot(&b)
            .eigh(Side::Lower)
            .map_err(|e| format!("structured residual evidence: factor rotation: {e:?}"))?;
        let mut rot = Array2::<f64>::zeros((rank, rank));
        for k in 0..rank {
            for l in 0..rank {
                rot[[l, k]] = vecs[[l, rank - 1 - k]];
            }
        }
        lambda = lambda.dot(&rot);
        for k in 0..rank {
            let mut pivot = 0usize;
            for i in 1..p {
                if lambda[[i, k]].abs() > lambda[[pivot, k]].abs() {
                    pivot = i;
                }
            }
            if lambda[[pivot, k]] < 0.0 {
                lambda.column_mut(k).mapv_inplace(|v| -v);
            }
        }
    }
    Ok(RankFit {
        log_evidence,
        lambda,
        diagonal,
        slot_scale: u.scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_normal(state: &mut u64) -> f64 {
        let mut next = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let u1 = 1.0 - next();
        let u2 = next();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn planted(n: usize, lambda0: &Array2<f64>, sigma: f64, seed: u64) -> Array2<f64> {
        let (p, r) = lambda0.dim();
        let mut state = seed;
        let mut out = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let f: Vec<f64> = (0..r).map(|_| lcg_normal(&mut state)).collect();
            for i in 0..p {
                let mut v = sigma * lcg_normal(&mut state);
                for k in 0..r {
                    v += lambda0[[i, k]] * f[k];
                }
                out[[row, i]] = v;
            }
        }
        out
    }

    #[test]
    fn normalizers_match_closed_forms() {
        use std::f64::consts::PI;
        assert!((log_matrix_cauchy_normalizer(2, 1) - (1.0 / (2.0 * PI)).ln()).abs() < 1e-12);
        assert!((log_matrix_cauchy_normalizer(1, 1) - (1.0 / PI).ln()).abs() < 1e-12);
        assert_eq!(log_matrix_cauchy_normalizer(5, 0), 0.0);
        assert!((log_orthogonal_group_volume(1) - 2.0_f64.ln()).abs() < 1e-12);
        assert!((log_orthogonal_group_volume(2) - (4.0 * PI).ln()).abs() < 1e-12);
        assert_eq!(log_orthogonal_group_volume(0), 0.0);
        let bounds: Vec<usize> = (1..=6).map(ledermann_bound).collect();
        assert_eq!(bounds, vec![0, 0, 1, 1, 2, 3]);
    }

    /// The analytic gradient and Hessian of the log posterior against central
    /// differences of the value and of the analytic gradient, at a point off
    /// the mode, on a three-bin rank-2 problem (every block: η, the chart's
    /// diagonal and off-diagonal entries, κ).
    #[test]
    fn analytic_derivatives_match_central_differences() {
        let p = 5usize;
        let lambda0 = ndarray::array![[1.0, 0.2], [0.8, -0.5], [-0.3, 0.9], [0.4, 0.4], [0.1, -0.7]];
        let n = 900usize;
        let resid = planted(n, &lambda0, 0.5, 0x243F6A8885A308D3);
        let row_bin: Vec<usize> = (0..n).map(|i| i * 3 / n).collect();
        let moments = binned_moments(resid.view(), &row_bin, 3).expect("moments");
        let problem = EvidenceProblem::new(&moments, 2).expect("problem");
        let dim = problem.dim();
        assert_eq!(dim, p + (p * 2 - 1) + 2);
        let mut theta = problem.start().clone();
        for (i, v) in theta.iter_mut().enumerate() {
            *v += 0.05 * ((i as f64) * 1.7).sin();
        }
        let at = problem.evaluate(&theta, true).expect("eval");
        let h = at.hessian.as_ref().expect("hessian");
        let step = 1e-5;
        for a in 0..dim {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[a] += step;
            minus[a] -= step;
            let ep = problem.evaluate(&plus, false).expect("plus");
            let em = problem.evaluate(&minus, false).expect("minus");
            let fd_grad = (ep.value - em.value) / (2.0 * step);
            let scale = 1.0 + at.gradient[a].abs();
            assert!(
                (fd_grad - at.gradient[a]).abs() <= 1e-5 * scale.max(at.value.abs() * 1e-4),
                "gradient[{a}]: analytic {} vs central {fd_grad}",
                at.gradient[a]
            );
            for c in 0..dim {
                let fd = (ep.gradient[c] - em.gradient[c]) / (2.0 * step);
                let tol = 1e-5 * (1.0 + h[[a, c]].abs() + h[[c, c]].abs().sqrt() * h[[a, a]].abs().sqrt());
                assert!(
                    (fd - h[[a, c]]).abs() <= tol,
                    "hessian[{a},{c}]: analytic {} vs central {fd}",
                    h[[a, c]]
                );
            }
        }
    }

    /// Rank 0 has the closed-form evidence
    /// `Σ_j [ln Γ(n/2) + (n/2) ln(2/S_jj)] − (np/2) ln 2π` (the η integral is a
    /// gamma integral). The Laplace value differs from it by Stirling's
    /// `1/(12·n/2)` per channel, `p/(6n)` in total.
    #[test]
    fn rank_zero_laplace_matches_the_exact_gamma_integral() {
        let p = 4usize;
        let n = 400usize;
        let lambda0 = Array2::<f64>::zeros((p, 0));
        let mut resid = planted(n, &lambda0, 1.0, 0x13198A2E03707344);
        for i in 0..n {
            for j in 0..p {
                resid[[i, j]] *= 1.0 + j as f64;
            }
        }
        let row_bin = vec![0usize; n];
        let moments = binned_moments(resid.view(), &row_bin, 1).expect("moments");
        let fit = fit_rank(&moments, 0).expect("rank 0");
        let nf = n as f64;
        let mut exact = -0.5 * nf * p as f64 * (2.0 * std::f64::consts::PI).ln();
        for j in 0..p {
            let sjj: f64 = (0..n).map(|i| resid[[i, j]] * resid[[i, j]]).sum();
            exact += ln_gamma(nf / 2.0) + nf / 2.0 * (2.0 / sjj).ln();
        }
        let gap = exact - fit.log_evidence;
        let stirling = p as f64 / (6.0 * nf);
        assert!(
            (gap - stirling).abs() <= 0.1 * stirling,
            "exact − Laplace = {gap:e}, Stirling predicts {stirling:e}"
        );
    }
}

