//! The exact invariance a penalized criterion has in its own smoothing
//! parameters, and the subspace a curvature certificate must therefore refuse
//! to judge (#2676).
//!
//! # The invariance
//!
//! Every criterion in this crate — REML, LAML, the profiled Gaussian score,
//! their Firth/Jeffreys variants — depends on the smoothing parameters
//! `lambda` ONLY through the assembled penalty
//!
//! ```text
//!     P_lambda(beta) = sum_i lambda_i (beta - mu_i)' S_i (beta - mu_i).
//! ```
//!
//! So if a vector `w` satisfies `sum_i w_i S_i = 0` **and** the two companion
//! conditions its prior means impose (see [`PenaltyMapInvariance`]), then
//! `P_{lambda + s w} = P_lambda` identically in `s`, and the criterion is
//! EXACTLY constant along that line of `lambda`. Nothing about the fit, the
//! data, or the family enters: it is a property of the penalty map alone.
//!
//! Such a `w` is precisely a null vector of the Gram matrix
//! `G_ij = <A_i, A_j>_F` of the penalty operators, because
//! `w' G w = ||sum_i w_i A_i||_F^2`.
//!
//! That identity is the definition and NOT the algorithm. It carries the
//! defect `||sum_i w_i A_i||_F` squared, so asking an `f64` eigensolver for
//! `null(G)` decides at `sqrt(eps)`: every map within `1.5e-8` of dependent
//! reads as exactly dependent. `DoubleDouble` carries the measurement that
//! makes this concrete and the arithmetic that removes it.
//!
//! # Why the certificate has to know
//!
//! `rho = log lambda` is a nonlinear reparameterisation, so for any smooth `V`
//!
//! ```text
//!     H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)          (*)
//! ```
//!
//! holds exactly — the second term is pure chain rule and carries no curvature.
//! Lift `w` to rho by `t = diag(lambda)^{-1} w`. Then `H_lambda w = 0` gives
//!
//! ```text
//!     t' H_rho t = sum_k (g_rho)_k t_k^2        EXACTLY, at every point,
//! ```
//!
//! whose magnitude is bounded by `sum_k |(g_rho)_k| t_k^2` — which is
//! *verbatim* the per-direction gradient floor
//! `crate::estimate::smoothing_correction::invert_identified_rho_hessian`
//! and the `H + diag(|g|)` test in `crate::rho_optimizer::run` compare
//! against. A direction of this subspace does not sit NEAR the decision
//! boundary of those gates; it sits ON it, by identity, and which side it
//! lands on is decided by the disagreement between the gradient evaluation and
//! the Hessian evaluation.
//!
//! Measured on `geo_disease_matern` (the #2676 repro):
//! `sigma = 2.0930992e-5`, `sum_k g_k v_k^2 = 2.0946774e-5`, intrinsic
//! `-1.578e-8` — the identity holding to `7.5e-4` relative, with the gate's
//! whole verdict riding on the sign of that residual.
//!
//! # ⚠ And this does NOT apply to a near-invariance
//!
//! Everything above is an identity, and it holds only where `sum_i w_i A_i` is
//! EXACTLY zero. Where it is merely small — `||sum_i w_i A_i||_F = delta` — the
//! criterion carries genuine curvature of order `delta^2` along the lift, the
//! residual of `t'H_rho t - sum_k g_k t_k^2` is that curvature and NOT the
//! assembly's error, and deflating the direction hides a measurement instead of
//! a rounding. Measured (#2676 penalty-map probe): the
//! `geo_disease_*_matern` cells' redundancy is a small-length-scale limit —
//! `delta = 2.079e-15` below `4e-2`, `1.874e-5` at the cold `Auto` geometry,
//! `3.396e-1` at the geometry the fit settles on — so on those cells there is
//! nothing here to deflate, and a certification that said otherwise was reading
//! `delta^2` against `eps`.
//!
//! The repair is therefore NOT a wider floor. The comparison is degenerate, not
//! under-resolved: **deflate the subspace, then apply the existing, unchanged
//! rule to its complement**, so that no direction is judged by a test whose
//! boundary it occupies identically.
//!
//! # What deflating cannot hide
//!
//! * *Genuine curvature.* On the deflated subspace the ρ-curvature is
//!   `sum_k g_k t_k^2` — a pure function of the gradient, which the outer loop
//!   has already certified against its own stationarity bound. There is no
//!   second-order information there to lose. #2665's `lambda_min = -1.6e3`
//!   saddle is not in it and still refuses.
//! * *A rho-prior's curvature.* When a prior on `rho` is present the criterion
//!   is no longer exactly flat along the lift, and (*) picks up the prior's own
//!   second derivative. Every prior this crate offers — `Normal`,
//!   `GammaPrecision` (`rate·e^rho`), and the PC prior
//!   (`rho/2 + theta·e^{-rho/2}`) — is CONVEX in rho, so that addition is
//!   positive semidefinite. Deflating can only discard a direction whose
//!   curvature was `artifact + (something >= 0)`; the only way it could have
//!   refused is if the artifact dominated, which is the very round-off verdict
//!   this module exists to remove.
//!
//! # What it does NOT claim
//!
//! Nothing here says the deflated directions are minima. It says they are not
//! MEASUREMENTS — the sibling distinction `gam_math::score_opt` already draws
//! when it reports that a request is unsatisfiable rather than that a bound was
//! violated (#2614).

use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2};

/// A double-double (unevaluated two-term) real, carrying `hi + lo` with
/// `|lo| <= ulp(hi)/2` — roughly 32 decimal digits.
///
/// # Why this site needs more than `f64`
///
/// The penalty map's rank is `rank({A_i})`, and the quantity that decides it is
/// the DEFECT `delta = min_w ||sum_i w_i A_i||_F / ||w||` — a linear measure of
/// how far the operators are from dependent. The Gram `G_ij = <A_i, A_j>_F`
/// carries it SQUARED: `lambda_min(G) = delta^2`. So a rank test taken on `G` in
/// working precision, at `G`'s own `eps`, is a defect test at `sqrt(eps)`:
///
/// ```text
///     delta = 1.5e-8   =>   lambda_min(G) = 2.2e-16 = eps
/// ```
///
/// and every penalty map within `1.5e-8` of a linear dependency reads as
/// EXACTLY dependent. Measured on `geo_disease_matern` (centers=24, n=4000,
/// #2676 penalty-map probe): pair defect `1.238e-8`, certified
/// nullity 1, the direction deflated, and the invariance residual the deflation
/// then feeds into the curvature resolution read `1.170e-8` — a number that was
/// reported as the ASSEMBLY'S ERROR and is in fact the criterion's own genuine
/// curvature along a direction it is not flat along.
///
/// The classical repair is to never form the normal equations: factor the
/// operator stack instead, whose singular values ARE the defects. Materializing
/// that stack costs `k * block^2` doubles, which at a shared block of a few
/// thousand columns is hundreds of megabytes. Doing the same arithmetic in
/// double-double instead costs a constant factor of TIME and no memory at all,
/// and it restores exactly what the squaring took: with `G` accurate to
/// `~eps^2` and its Cholesky taken in the same precision, the pivot
/// `sqrt(d_j)` is the defect, accurate to `~eps` RELATIVE, at any magnitude
/// down to `eps` itself.
///
/// This is the same lesson as this module's other compensation (#2748) one
/// level up: compute the quantity to the accuracy the decision needs, rather
/// than widening the bar.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    fn new(value: f64) -> Self {
        Self {
            hi: value,
            lo: 0.0,
        }
    }

    /// The nearest `f64`. Correct to a rounding of the double-double value.
    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Knuth's TwoSum: exact for any two finite `f64`, no ordering assumption.
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let sum = a + b;
        let b_virtual = sum - a;
        let a_virtual = sum - b_virtual;
        ((sum), ((a - a_virtual) + (b - b_virtual)))
    }

    /// Dekker's QuickTwoSum: exact when `|a| >= |b|`, which the renormalisation
    /// step below guarantees.
    fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
        let sum = a + b;
        (sum, b - (sum - a))
    }

    /// Exact product, via one fused multiply-add. `mul_add` rounds ONCE, so
    /// `a*b - p` is representable and is the product's exact residual.
    fn two_product(a: f64, b: f64) -> (f64, f64) {
        let product = a * b;
        (product, a.mul_add(b, -product))
    }

    fn add(self, other: Self) -> Self {
        let (sum, mut error) = Self::two_sum(self.hi, other.hi);
        error += self.lo + other.lo;
        let (hi, lo) = Self::quick_two_sum(sum, error);
        Self { hi, lo }
    }

    fn sub(self, other: Self) -> Self {
        self.add(Self {
            hi: -other.hi,
            lo: -other.lo,
        })
    }

    fn mul(self, other: Self) -> Self {
        let (product, mut error) = Self::two_product(self.hi, other.hi);
        error += self.hi * other.lo + self.lo * other.hi;
        let (hi, lo) = Self::quick_two_sum(product, error);
        Self { hi, lo }
    }

    /// Exact product of two `f64`, as a double-double. This is how every term
    /// of the Gram enters: rounding the product first would put an `eps`-sized
    /// error on a quantity the decision reads at `eps^2`.
    fn from_product(a: f64, b: f64) -> Self {
        let (hi, lo) = Self::two_product(a, b);
        Self { hi, lo }
    }

    /// One Newton correction on the `f64` quotient, which doubles its digits.
    fn div(self, other: Self) -> Self {
        let approximate = self.hi / other.hi;
        if !approximate.is_finite() {
            return Self::new(approximate);
        }
        let remainder = self.sub(other.mul(Self::new(approximate)));
        let correction = remainder.hi / other.hi;
        let (hi, lo) = Self::quick_two_sum(approximate, correction);
        Self { hi, lo }
    }

    /// One Newton correction on the `f64` square root, same doubling.
    fn sqrt(self) -> Self {
        if self.hi <= 0.0 {
            return Self::ZERO;
        }
        let approximate = self.hi.sqrt();
        let remainder = self.sub(Self::new(approximate).mul(Self::new(approximate)));
        let correction = remainder.to_f64() / (2.0 * approximate);
        let (hi, lo) = Self::quick_two_sum(approximate, correction);
        Self { hi, lo }
    }

    fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
}

/// The exact null space of the penalty map, in `lambda` coordinates.
///
/// Built from the canonical penalties alone. No tolerance is chosen: the rank
/// boundary is decided in the DEFECT's own units — the norm of what is left of
/// an operator after projecting onto the ones already accepted — against the
/// construction error operators of that size carry. See `DoubleDouble` for
/// why the Gram's eigenvalues cannot answer this and what it cost when they
/// were asked to (#2676).
#[derive(Debug, Clone)]
pub struct PenaltyMapInvariance {
    /// Orthonormal columns spanning `null({A_i})`, shape `k x d`.
    basis: Array2<f64>,
    /// The defect floor the rank boundary was decided at.
    resolution: f64,
}

impl PenaltyMapInvariance {
    /// Dimension of the invariance, i.e. the certified structural nullity of
    /// the penalty map. This is the number `k - rank(G)` the smoothing
    /// correction's nullity identity is denominated in.
    pub fn dimension(&self) -> usize {
        self.basis.ncols()
    }

    /// The DEFECT floor the rank boundary was decided at, i.e. the largest
    /// `min_c ||A_j - sum c_i A_i||_F` this constructor is willing to call
    /// zero. Denominated in the operators' own norm, NOT in the Gram's — a
    /// defect of `r` shows up in the Gram as `r^2`, and reporting the square
    /// is what let a `1.2e-8` near-dependency read as exact (#2676).
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// Build from the canonical penalty bundle.
    ///
    /// The Gram is taken over the AUGMENTED operators
    ///
    /// ```text
    ///     A_i = [[ S_i,        -S_i mu_i        ],
    ///            [ -mu_i' S_i,  mu_i' S_i mu_i  ]]
    /// ```
    ///
    /// so that `sum_i w_i A_i = 0` is equivalent to the FULL centered quadratic
    /// `sum_i lambda_i (beta - mu_i)' S_i (beta - mu_i)` being invariant along
    /// `w` — the quadratic, linear and constant parts all at once. With zero
    /// prior means (the overwhelmingly common case) `A_i` is `S_i` bordered by
    /// zeros and the Gram is bit-identical to the plain `tr(S_i S_j)` one, so
    /// this generalisation cannot move any existing verdict; with nonzero
    /// means it can only ADD conditions, i.e. shrink the invariance, which is
    /// the conservative direction.
    pub fn from_canonical_penalties(
        canonical: &[CanonicalPenalty],
        coefficient_dimension: usize,
    ) -> Result<Self, String> {
        let k = canonical.len();
        if k == 0 {
            return Ok(Self {
                basis: Array2::zeros((0, 0)),
                resolution: 0.0,
            });
        }
        for (index, penalty) in canonical.iter().enumerate() {
            let block_dimension = penalty.col_range.end.saturating_sub(penalty.col_range.start);
            if penalty.col_range.end > coefficient_dimension
                || penalty.local.dim() != (block_dimension, block_dimension)
                || penalty.prior_mean.len() != block_dimension
            {
                return Err(format!(
                    "canonical penalty {index} has range {:?}, local shape {:?}, prior mean length \
                     {}, coefficient dimension {coefficient_dimension}",
                    penalty.col_range,
                    penalty.local.dim(),
                    penalty.prior_mean.len(),
                ));
            }
        }

        // ── Disjoint supports carry no invariance, and that is a theorem ──
        //
        // If no two penalties share a coefficient column then `tr(S_i S_j) = 0`
        // and `c_i . c_j = 0` for every `i != j`, so
        //
        //     G = diag(||S_i||_F^2 + 2||c_i||^2) + q q',    q_i = mu_i' S_i mu_i,
        //
        // which is a positive-definite diagonal plus a rank-1 PSD term: strictly
        // positive definite, hence `null(G) = {0}`. This is the ordinary
        // additive model — one penalty per smooth, on its own coefficient block
        // — i.e. very nearly every fit, and it exits here in `O(k^2)` range
        // comparisons instead of the `O(k^2 * block^2)` Gram. That matters
        // because this runs at every certification and `block` can be large.
        //
        // Guarded on every `||S_i||_F > 0`: a rank-zero penalty would make the
        // diagonal singular, and `canonicalize_penalty_specs` drops those, but
        // this constructor is public and must not assume it.
        let norms: Vec<f64> = canonical
            .iter()
            .map(|penalty| penalty.local.iter().map(|value| value * value).sum::<f64>())
            .collect();
        let shares_support = (0..k).any(|i| {
            ((i + 1)..k).any(|j| {
                canonical[i].col_range.start < canonical[j].col_range.end
                    && canonical[j].col_range.start < canonical[i].col_range.end
            })
        });
        if !shares_support && norms.iter().all(|value| *value > 0.0) {
            return Ok(Self {
                basis: Array2::zeros((k, 0)),
                resolution: 0.0,
            });
        }

        // The centering vectors c_i = S_i mu_i, in GLOBAL coefficient
        // coordinates so that overlapping blocks add correctly, and the
        // scalars q_i = mu_i' S_i mu_i. Allocated only when some prior mean is
        // nonzero: `k x p` is 13 MB at `k = 50, p = 4096`, and the overwhelming
        // majority of penalty maps are centered at zero, where `A_i` is `S_i`
        // bordered by exact zeros and the whole border term vanishes.
        let centered = canonical
            .iter()
            .any(|penalty| penalty.prior_mean.iter().any(|value| *value != 0.0));
        let mut centering =
            Array2::<f64>::zeros((k, if centered { coefficient_dimension } else { 0 }));
        let mut quadratic = Array1::<f64>::zeros(k);
        for (index, penalty) in canonical.iter().enumerate() {
            if !centered || penalty.prior_mean.iter().all(|value| *value == 0.0) {
                continue;
            }
            let start = penalty.col_range.start;
            let block = penalty.col_range.end - start;
            for row in 0..block {
                let mut accumulated = 0.0_f64;
                for col in 0..block {
                    accumulated += penalty.local[[row, col]] * penalty.prior_mean[col];
                }
                centering[[index, start + row]] = accumulated;
                quadratic[index] += penalty.prior_mean[row] * accumulated;
            }
        }

        // Gram of the unscaled augmented maps, in DOUBLE-DOUBLE. Positive
        // lambdas only rescale the columns of the map and therefore cannot
        // change this rank.
        //
        // Every term enters as an EXACT product (`from_product`) and every
        // addition is exact to the second word, so a Gram entry of size `O(1)`
        // carries `O(m * eps^2)` of error rather than `O(eps)`. That is what
        // makes the rank boundary below a statement about the OPERATORS rather
        // than about `sqrt(eps)`; see `DoubleDouble` for the measurement that
        // forced it.
        let mut gram = vec![DoubleDouble::ZERO; k * k];
        for i in 0..k {
            for j in i..k {
                let start = canonical[i].col_range.start.max(canonical[j].col_range.start);
                let end = canonical[i].col_range.end.min(canonical[j].col_range.end);
                let mut accumulator = DoubleDouble::ZERO;
                for global_row in start..end {
                    for global_col in start..end {
                        accumulator = accumulator.add(DoubleDouble::from_product(
                            canonical[i].local[[
                                global_row - canonical[i].col_range.start,
                                global_col - canonical[i].col_range.start,
                            ]],
                            canonical[j].local[[
                                global_row - canonical[j].col_range.start,
                                global_col - canonical[j].col_range.start,
                            ]],
                        ));
                    }
                }
                // The two border blocks of A_i contribute 2 c_i . c_j, the
                // corner contributes q_i q_j. Both enter the same accumulator:
                // they are terms of the same inner product, and splitting them
                // off would reintroduce a rounding at exactly the scale the
                // decision is taken at.
                let mut border = DoubleDouble::ZERO;
                for column in 0..centering.ncols() {
                    border = border.add(DoubleDouble::from_product(
                        centering[[i, column]],
                        centering[[j, column]],
                    ));
                }
                accumulator = accumulator.add(border.mul(DoubleDouble::new(2.0)));
                accumulator =
                    accumulator.add(DoubleDouble::from_product(quadratic[i], quadratic[j]));
                if !accumulator.is_finite() {
                    return Err(format!(
                        "penalty-map Gram entry ({i},{j}) is not finite"
                    ));
                }
                gram[i * k + j] = accumulator;
                gram[j * k + i] = accumulator;
            }
        }

        // ── The rank boundary, in the DEFECT's own units ──
        //
        // `floor(r)` is the size of a residual that `r + 1` operators of this
        // size could produce out of their own construction arithmetic alone.
        // Each operator is an accumulation over its `m = block^2` entries, so
        // its own error is `sqrt(m) * eps * ||A||_F`; a residual against `r`
        // accepted columns combines `r + 1` such independent errors, which add
        // in quadrature. Nothing is chosen: the model is the arithmetic, and it
        // is calibrated by the two populations it has to separate — measured on
        // `geo_disease_matern`, a pair known equal sits at `2.079e-15` against
        // a `sqrt(m) * eps` of `2.220e-15`, and the nearest pair known distinct
        // sits at `8.75e-9`, six orders the other side.
        let entries = canonical
            .iter()
            .map(|penalty| penalty.local.len())
            .max()
            .unwrap_or(0) as f64;
        // The largest operator norm in the group, NOT `max(1)`: the residual
        // lives in the operators' own absolute units, and clamping the scale up
        // would make the floor loose in exact proportion to how small the
        // operators are. An all-zero map has scale zero, floor zero, and every
        // pivot fails `> 0` — so every direction is null, which is what an
        // all-zero penalty map means.
        let scale = norms
            .iter()
            .fold(0.0_f64, |worst, value| worst.max(value.sqrt()));
        let operator_error = entries.sqrt() * f64::EPSILON * scale;
        let defect_floor = |accepted: usize| operator_error * ((accepted + 1) as f64).sqrt();

        // ── Pivoted Cholesky of the double-double Gram ──
        //
        // `G = L L'` with column pivoting. At step `s` the pivot `d[j]` is the
        // SQUARED norm of `A_j`'s residual against the span already accepted,
        // so `sqrt(d[j])` IS the defect of column `j` against that span —
        // exactly the quantity the boundary is denominated in, and (unlike
        // `lambda_min(G)` read off an `f64` eigensolver) known to full relative
        // accuracy because the whole recurrence runs in double-double.
        let mut lower = vec![DoubleDouble::ZERO; k * k];
        let mut diagonal: Vec<DoubleDouble> = (0..k).map(|i| gram[i * k + i]).collect();
        let mut order: Vec<usize> = (0..k).collect();
        let mut rank = k;
        for step in 0..k {
            let pivot = (step..k).fold(step, |best, index| {
                if diagonal[index].to_f64() > diagonal[best].to_f64() {
                    index
                } else {
                    best
                }
            });
            let residual = diagonal[pivot];
            // A true Gram is PSD, so a pivot below `-floor^2` is not a rank
            // statement — it is a malformed input, and saying so beats
            // silently truncating the rank.
            let floor = defect_floor(step);
            if residual.to_f64() < -(floor * floor) {
                return Err(format!(
                    "penalty-map Gram is not positive semidefinite: pivot {:.3e} at step {step} \
                     is below the negated squared defect floor {:.3e}",
                    residual.to_f64(),
                    -(floor * floor),
                ));
            }
            if !(residual.sqrt().to_f64() > floor) {
                rank = step;
                break;
            }
            order.swap(step, pivot);
            diagonal.swap(step, pivot);
            for column in 0..step {
                lower.swap(step * k + column, pivot * k + column);
            }
            let pivot_root = diagonal[step].sqrt();
            lower[step * k + step] = pivot_root;
            for row in (step + 1)..k {
                let mut accumulated = gram[order[row] * k + order[step]];
                for column in 0..step {
                    accumulated =
                        accumulated.sub(lower[row * k + column].mul(lower[step * k + column]));
                }
                let entry = accumulated.div(pivot_root);
                lower[row * k + step] = entry;
                diagonal[row] = diagonal[row].sub(entry.mul(entry));
            }
        }

        // ── The null space of the MAP, from the factor rather than the Gram ──
        //
        // Column `j` of the permuted stack is `sum_t L[j][t] q_t` in the
        // orthonormal frame the factorisation produced, so
        // `sum_j w_j A_j = 0` reads `L[:, 0..rank]' w = 0`. With `L1` the
        // leading `rank x rank` triangle (nonsingular by the pivot test above)
        // and `L2` the rows below it, the solutions are
        // `w1 = -(L1')^{-1} L2' w2` for free `w2`, which is a basis of exactly
        // `k - rank` columns.
        let mut basis = Array2::<f64>::zeros((k, k - rank));
        for (column, free) in (rank..k).enumerate() {
            let mut w = Array1::<f64>::zeros(k);
            w[free] = 1.0;
            // Back-substitute `L1' w1 = -L2' w2` from the bottom row up.
            for row in (0..rank).rev() {
                let mut accumulated = 0.0_f64;
                for index in (row + 1)..k {
                    accumulated += lower[index * k + row].to_f64() * w[index];
                }
                let pivot_root = lower[row * k + row].to_f64();
                w[row] = if pivot_root != 0.0 {
                    -accumulated / pivot_root
                } else {
                    0.0
                };
            }
            for (position, &source) in order.iter().enumerate() {
                basis[[source, column]] = w[position];
            }
        }
        let basis = if basis.ncols() == 0 {
            basis
        } else {
            match orthonormalize_columns(&basis) {
                Some(orthonormal) => orthonormal,
                // Every column of a null basis is independent by construction,
                // so losing them all means the back-substitution produced
                // nothing usable. Certify nothing rather than a wrong subspace.
                None => Array2::<f64>::zeros((k, 0)),
            }
        };

        // The one number that decides whether the curvature certificate gets to
        // deflate anything, printed in the DEFECT's units and with the bar it
        // was decided against. Without this line a redundancy warning and a
        // `structural_zero = 0` classification sit in the same log with nothing
        // connecting them, which is exactly the state #2748 was found in.
        log::debug!(
            "[PENALTY-INVARIANCE] k={k} rank={rank} certified_nullity={} \
             defect_floor={operator_error:.6e} pivot_defects={:?} gram_diagonal={:?} \
             (a column is certified dependent when its residual defect sqrt(d_j) — the norm of \
             what is left of it after projecting onto the columns already accepted — is at or \
             under the floor; the floor is one operator-construction error per operator involved, \
             in quadrature)",
            basis.ncols(),
            (0..k)
                .map(|i| diagonal[i].sqrt().to_f64())
                .collect::<Vec<_>>(),
            (0..k).map(|i| gram[i * k + i].to_f64()).collect::<Vec<_>>(),
        );
        Ok(Self {
            basis,
            resolution: operator_error,
        })
    }

    /// Lift the invariance to the outer coordinate vector.
    ///
    /// `t = diag(lambda)^{-1} w` is the tangent, at `lambda`, of the curve in
    /// `rho` that the straight line `lambda + s w` traces. `theta_dimension`
    /// and `rho_offset` embed it into a wider outer vector (the exact-joint
    /// spatial route optimises `theta = (rho, psi)`; a mixture/SAS route
    /// appends link coordinates), leaving every non-`rho` coordinate zero:
    /// those coordinates are not part of the invariance, and the identity above
    /// holds for the embedded vector because its non-`rho` components vanish.
    ///
    /// Returns `None` when there is nothing to deflate, so callers stay on the
    /// bit-identical legacy path.
    pub fn theta_directions(
        &self,
        lambdas: &Array1<f64>,
        theta_dimension: usize,
        rho_offset: usize,
    ) -> Option<Array2<f64>> {
        let k = self.basis.nrows();
        if self.basis.ncols() == 0 || k == 0 {
            return None;
        }
        if lambdas.len() != k || rho_offset + k > theta_dimension {
            return None;
        }
        if !lambdas.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        let mut lifted = Array2::<f64>::zeros((theta_dimension, self.basis.ncols()));
        for column in 0..self.basis.ncols() {
            for row in 0..k {
                lifted[[rho_offset + row, column]] = self.basis[[row, column]] / lambdas[row];
            }
        }
        orthonormalize_columns(&lifted)
    }
}

/// Modified Gram-Schmidt with a relative drop tolerance, returning `None` when
/// nothing survives.
///
/// The drop tolerance is the classical loss-of-orthogonality scale
/// `64 * n * EPSILON` relative to the incoming column norm — the same
/// arithmetic-floor coefficient the sibling
/// `crate::estimate::smoothing_correction::eigenpair_backward_error_bound`
/// uses, kept identical so the two places that decide "this is round-off" agree.
/// A column whose residual against the accepted basis has fallen that far is
/// numerically dependent, and keeping it would admit a direction determined
/// entirely by round-off.
pub fn orthonormalize_columns(columns: &Array2<f64>) -> Option<Array2<f64>> {
    let rows = columns.nrows();
    let mut accepted: Vec<Array1<f64>> = Vec::with_capacity(columns.ncols());
    let drop_relative = 64.0 * (rows.max(1) as f64) * f64::EPSILON;
    for index in 0..columns.ncols() {
        let mut vector = columns.column(index).to_owned();
        let incoming = vector.dot(&vector).sqrt();
        if !(incoming > 0.0) || !incoming.is_finite() {
            continue;
        }
        // Twice is enough (Kahan-Parlett): one pass can lose orthogonality on a
        // nearly dependent column, two passes recover it or reveal dependence.
        for _ in 0..2 {
            for basis_vector in &accepted {
                let projection = vector.dot(basis_vector);
                vector.scaled_add(-projection, basis_vector);
            }
        }
        let residual = vector.dot(&vector).sqrt();
        if !residual.is_finite() || residual <= drop_relative * incoming {
            continue;
        }
        vector.mapv_inplace(|value| value / residual);
        accepted.push(vector);
    }
    if accepted.is_empty() {
        return None;
    }
    let mut basis = Array2::<f64>::zeros((rows, accepted.len()));
    for (column, vector) in accepted.iter().enumerate() {
        basis.column_mut(column).assign(vector);
    }
    Some(basis)
}

/// Orthonormal basis of the subspace a curvature certificate is entitled to
/// judge: the orthogonal complement of `span({e_k : k in excluded} U deflate)`.
///
/// Returns `None` when the complement is empty (nothing left to judge) and
/// `Some(Z)` with `Z' Z = I` otherwise. With `deflate = None` this is exactly
/// the indicator basis of the un-excluded coordinates, so `Z' H Z` is the
/// interior sub-block the certificate has always taken — bit for bit.
pub fn judged_subspace_basis(
    dimension: usize,
    excluded: &[usize],
    deflate: Option<&Array2<f64>>,
) -> Option<Array2<f64>> {
    if dimension == 0 {
        return None;
    }
    let excluded_set: std::collections::BTreeSet<usize> =
        excluded.iter().copied().filter(|k| *k < dimension).collect();
    let interior: Vec<usize> = (0..dimension)
        .filter(|k| !excluded_set.contains(k))
        .collect();
    if interior.is_empty() {
        return None;
    }
    let deflate = match deflate {
        Some(matrix) if matrix.nrows() == dimension && matrix.ncols() > 0 => matrix,
        // Nothing to deflate: the interior indicator basis, which reproduces
        // the historical sub-block extraction exactly.
        _ => {
            let mut basis = Array2::<f64>::zeros((dimension, interior.len()));
            for (column, &row) in interior.iter().enumerate() {
                basis[[row, column]] = 1.0;
            }
            return Some(basis);
        }
    };
    use gam_linalg::faer_ndarray::FaerEigh;
    // Deflate only the part of the invariance that lives INSIDE the judged
    // face.
    //
    // The identity that licenses deflation — `t' H_rho t = sum_k g_k t_k^2` —
    // is a statement about the FULL direction `t`. Simply dropping `t`'s
    // excluded components and deflating what is left would break it by
    // `O(||t_excluded|| * ||H||)`, and in the extreme (an invariance direction
    // that is mostly ON an excluded coordinate) it would deflate a direction
    // carrying REAL curvature — hiding exactly what the certificate exists to
    // find. So take the maximal subspace of `span(deflate)` whose excluded
    // components vanish: `deflate * null(R)` with `R` the excluded rows.
    //
    // "Vanish" is judged at orthonormality round-off (`64 * n * EPSILON`,
    // squared because `R'R` carries squared norms) — the columns of `deflate`
    // are orthonormal, so that is the scale at which a component is
    // indistinguishable from an exact zero. Anything above it is kept in the
    // judged block, i.e. the pre-#2676 verdict, which is the conservative
    // direction and costs nothing on a fit with no railed coordinate (the case
    // every #2676 refusal reports).
    let deflate_owned;
    let deflate = if excluded_set.is_empty() {
        deflate
    } else {
        let mut excluded_rows =
            Array2::<f64>::zeros((excluded_set.len(), deflate.ncols()));
        for (target, &row) in excluded_set.iter().enumerate() {
            for column in 0..deflate.ncols() {
                excluded_rows[[target, column]] = deflate[[row, column]];
            }
        }
        let mut gram = excluded_rows.t().dot(&excluded_rows);
        gam_linalg::matrix::symmetrize_in_place(&mut gram);
        let (eigenvalues, eigenvectors) = gram.eigh(faer::Side::Lower).ok()?;
        let round_off = 64.0 * (dimension as f64) * f64::EPSILON;
        let free: Vec<usize> = (0..deflate.ncols())
            .filter(|&index| eigenvalues[index] <= round_off * round_off)
            .collect();
        if free.is_empty() {
            let mut basis = Array2::<f64>::zeros((dimension, interior.len()));
            for (column, &row) in interior.iter().enumerate() {
                basis[[row, column]] = 1.0;
            }
            return Some(basis);
        }
        let mut selector = Array2::<f64>::zeros((deflate.ncols(), free.len()));
        for (column, &source) in free.iter().enumerate() {
            selector
                .column_mut(column)
                .assign(&eigenvectors.column(source));
        }
        deflate_owned = deflate.dot(&selector);
        &deflate_owned
    };
    // Restrict to the interior coordinates. Exact by the selection above: every
    // surviving column's excluded components are zero to round-off.
    let mut restricted = Array2::<f64>::zeros((interior.len(), deflate.ncols()));
    for column in 0..deflate.ncols() {
        for (target, &row) in interior.iter().enumerate() {
            restricted[[target, column]] = deflate[[row, column]];
        }
    }
    let interior_dimension = interior.len();
    let orthonormal = match orthonormalize_columns(&restricted) {
        Some(basis) => basis,
        None => {
            let mut basis = Array2::<f64>::zeros((dimension, interior_dimension));
            for (column, &row) in interior.iter().enumerate() {
                basis[[row, column]] = 1.0;
            }
            return Some(basis);
        }
    };
    // Complement inside the interior block, taken as the range of the
    // orthogonal projector `P = I - Q Q'`. `P` is symmetric with spectrum
    // exactly {0, 1}, so selecting eigenvalues above 1/2 is a decision with an
    // O(1) margin rather than one taken at round-off scale — the whole point of
    // this module is to stop deciding things at round-off scale.
    let mut projector = Array2::<f64>::eye(interior_dimension);
    projector -= &orthonormal.dot(&orthonormal.t());
    gam_linalg::matrix::symmetrize_in_place(&mut projector);
    let (eigenvalues, eigenvectors) = projector.eigh(faer::Side::Lower).ok()?;
    let kept: Vec<usize> = (0..interior_dimension)
        .filter(|&index| eigenvalues[index] > 0.5)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let mut basis = Array2::<f64>::zeros((dimension, kept.len()));
    for (column, &source) in kept.iter().enumerate() {
        for (target, &row) in interior.iter().enumerate() {
            basis[[row, column]] = eigenvectors[[target, source]];
        }
    }
    Some(basis)
}

/// Orthonormal basis of the orthogonal complement of `basis` in `R^dimension`
/// — i.e. exactly the directions [`judged_subspace_basis`] removed.
///
/// Derived from `basis` rather than from whatever was passed in as the
/// deflation: the judged basis is the authority on what was actually deflated
/// (the face restriction and the dependence drop can both shrink it), and a
/// caller reconstructing that set from its own input would be reading a
/// different subspace than the one the verdict was taken on.
///
/// Same `I - Z Z'` projector construction, with the same O(1) selection margin.
pub fn deflated_directions(dimension: usize, basis: &Array2<f64>) -> Option<Array2<f64>> {
    use gam_linalg::faer_ndarray::FaerEigh;
    if basis.nrows() != dimension || dimension == 0 || basis.ncols() >= dimension {
        return None;
    }
    let mut projector = Array2::<f64>::eye(dimension);
    projector -= &basis.dot(&basis.t());
    gam_linalg::matrix::symmetrize_in_place(&mut projector);
    let (eigenvalues, eigenvectors) = projector.eigh(faer::Side::Lower).ok()?;
    let kept: Vec<usize> = (0..dimension)
        .filter(|&index| eigenvalues[index] > 0.5)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let mut removed = Array2::<f64>::zeros((dimension, kept.len()));
    for (column, &source) in kept.iter().enumerate() {
        removed.column_mut(column).assign(&eigenvectors.column(source));
    }
    Some(removed)
}

/// The **measured** `‖δH_ρ‖₂` this site is entitled to, read off the one
/// identity the penalty map guarantees is exactly zero (#2748).
///
/// # The identity
///
/// Let `T` have orthonormal columns lying inside the criterion's certified
/// invariance, lifted to `ρ` (the subspace [`PenaltyMapInvariance`] certifies
/// and [`judged_subspace_basis`] deflates). Along any `t` in it the criterion
/// is exactly constant in `λ`, so with `w = diag(λ)t` and `ρ(s) = log(λ + sw)`
///
/// ```text
///     0 = d²/ds² V(ρ(s)) = t' H_ρ t + g_ρ · ρ''(0),   ρ''(0)_k = -t_k²
/// ```
///
/// and the same argument on a mixed second difference gives the full block
/// statement
///
/// ```text
///     T' H_ρ T  -  T' diag(g_ρ) T  =  0      EXACTLY, at every ρ.
/// ```
///
/// # What its residual therefore is
///
/// Whatever comes back is error, and only error: no property of the data, the
/// family or the fit can move it off zero. It is a **certified lower bound on
/// `‖δH_ρ‖₂` for the (Hessian, gradient) pair as evaluated here**, which is
/// exactly the currency the ρ-curvature gate spends — that gate compares a
/// Hessian eigenvalue against a gradient-built floor, so an inconsistency
/// between the two evaluations is precisely its resolution limit.
///
/// Measured on `geo_disease_eas_matern_k6` (#2748): `9.872e-8`, against an
/// eigensolver backward error of `8.342e-19` on the same matrix at the same ρ
/// — eleven orders apart, because the two answer different questions. The
/// curvature the gate refused there was `-2.010e-8`, i.e. **inside** the
/// assembly's own demonstrated inconsistency.
///
/// Returns `None` when there is nothing to measure (no directions, a shape
/// mismatch, a gradient of the wrong length) or when the residual is not
/// finite: an absent measurement must stay absent rather than become a zero,
/// which would silently assert that the assembly is exact.
pub fn invariance_residual_2norm(
    hessian_rho: &Array2<f64>,
    outer_gradient: &Array1<f64>,
    directions: &Array2<f64>,
) -> Option<f64> {
    use gam_linalg::faer_ndarray::FaerEigh;

    let n = hessian_rho.nrows();
    if n == 0 || hessian_rho.ncols() != n || directions.nrows() != n {
        return None;
    }
    let d = directions.ncols();
    if d == 0 || outer_gradient.len() != n {
        return None;
    }
    // `T'(H - diag(g))T`, formed as `T'HT - T'diag(g)T` so the large `H` product
    // is a single GEMM pair and the diagonal term costs `O(n·d²)`.
    let mut residual = directions.t().dot(hessian_rho).dot(directions);
    for row in 0..d {
        for col in 0..d {
            let mut chain_rule = 0.0_f64;
            for k in 0..n {
                chain_rule += outer_gradient[k] * directions[[k, row]] * directions[[k, col]];
            }
            residual[[row, col]] -= chain_rule;
        }
    }
    gam_linalg::matrix::symmetrize_in_place(&mut residual);
    if residual.iter().any(|value| !value.is_finite()) {
        return None;
    }
    // Spectral norm of a symmetric block: the largest eigenvalue magnitude.
    // `d` is the certified nullity of the penalty map — 1 or 2 in practice — so
    // this eigendecomposition is free.
    let (eigenvalues, _) = residual.eigh(faer::Side::Lower).ok()?;
    let norm = eigenvalues
        .iter()
        .fold(0.0_f64, |accumulated, value| accumulated.max(value.abs()));
    norm.is_finite().then_some(norm)
}

/// Compress a symmetric matrix onto the judged subspace: `Z' H Z`.
pub fn compress_to_judged_subspace(matrix: &Array2<f64>, basis: &Array2<f64>) -> Array2<f64> {
    let mut compressed = basis.t().dot(matrix).dot(basis);
    gam_linalg::matrix::symmetrize_in_place(&mut compressed);
    compressed
}

#[cfg(test)]
#[path = "penalty_invariance_tests.rs"]
mod penalty_invariance_tests;
