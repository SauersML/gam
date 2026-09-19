//! The λ̂-selection replay: the null law of a smooth-term statistic when the
//! term's smoothing parameters are CHOSEN by the outer REML criterion from the
//! same data, rather than given (#2672).
//!
//! It lives here, below the fit orchestration, because two instruments read
//! against it: the whole-term likelihood-ratio test in `gam-models`, and the
//! summary Wald test in [`crate::inference::smooth_test`].

use ndarray::{Array1, Array2};

/// The null law of `W(λ̂)` when `λ̂` is CHOSEN by the outer criterion rather than
/// given — the reference the whole-term LR statistic actually needs (#2672).
///
/// # The defect this exists for
///
/// `SmoothLrReferenceDf`'s spectrum is the exact null law of `W` *at a fixed*
/// `λ`. `λ̂` is not fixed: REML picks it from a continuum, on the same data that
/// produced `W`. Measured on a Gaussian null with `σ` known — so no Lawley term
/// is in play and the reference is the only thing being tested — the two move
/// together (`corr(W, Σw) = 0.94–0.96`) but not by enough, and the conditional
/// reference over-rejects:
///
/// ```text
///                    α = .20    .10     .05     .01
///   conditional      .2060   .1320   .0840   .0180     n = 30,  k = 12
///                    .2160   .1100   .0580   .0140     n = 100, k = 12
///                    .1850   .1025   .0650   .0150     n = 200, k = 12
/// ```
///
/// It is not a mean problem, and it must not be fixed as one: on those same runs
/// `E[W]/E[Σw] ≈ 2.4–2.5`, and dividing `W` by that ratio takes the size at
/// `α = 0.05` from `.087` to `.0000`. The reference is not mean-matched to `W`
/// and is not supposed to be.
///
/// # The replay, and why it needs no refit
///
/// Diagonalize the term's fitted penalty `S_jj` against the Schur-complemented
/// information `Ĩ_jj` — the pair is symmetric-definite, so a single basis
/// diagonalizes both, with generalized eigenvalues `ν_k = p_k/(1 − p_k)` read
/// straight off the penalty shares `lr_tested_block` already computes. In
/// that basis the tested block is `q` independent standard normals `u_k`, and
/// BOTH the statistic and the criterion that selects `λ` are closed forms in
/// them and in the scale `t = λ/λ̂`:
///
/// ```text
/// W(t)  = Σ_k (2f_k − f_k²) u_k² ,          f_k = 1/(1 + t·ν_k)
/// V(t)  = ½ Σ_k u_k² ·t·ν_k/(1 + t·ν_k)
///       + ½ Σ_{k: ν_k > 0} log((1 + t·ν_k)/(t·ν_k))     (+ terms free of t)
/// ```
///
/// So the whole selection — draw data, choose `λ̂`, read `W` — is a function of
/// `q` numbers, and the null law of `W(λ̂)` can be generated exactly (within the
/// same quadratic expansion the conditional law already assumes) with no design,
/// no response and no refit. `t = 1` reproduces the conditional law, which is
/// what makes this a strict generalization rather than a different reference.
///
/// # What it buys, measured
///
/// Same runs, same replicates, `20 000` draws per fit:
///
/// ```text
///                    α = .20    .10     .05     .01
///   selection-aware  .1940   .1160   .0560   .0120     n = 30,  k = 12
///                    .2020   .0840   .0440   .0080     n = 100, k = 12
///                    .1775   .0925   .0425   .0075     n = 200, k = 12
/// ```
///
/// Closer to nominal at every level in every cell — twelve of twelve — and the
/// `α = 0.05` column goes from a mean of `.069` to `.047` against a per-cell
/// Monte-Carlo standard error of `.0097`.
///
/// # The Monte-Carlo error is removed where it would matter
///
/// The replay is a simulation, so its tail is an estimate. The conditional tail
/// is NOT — `gam_math::probability::signed_weighted_chi_square_sf` evaluates it by
/// inversion. The two are strongly dependent (the same draws, differing only in
/// whether `t` is selected or held at one), so the replay reports the
/// DIFFERENCE and adds it to the exact conditional value:
///
/// ```text
/// p_selection = p_conditional(w) + [ P̂(W_sel ≥ w_sel) − P̂(W_cond ≥ w) ]
/// ```
///
/// where `w_sel` is the observation under the replay's own selection (see
/// [`Self::observed`]; `w_sel = w` when no score was supplied).
///
/// a textbook control variate. The bracket is a difference of two indicators
/// that agree on most draws, so its variance is a fraction of either term's, and
/// the standard error of the pair is measured per query and published in the
/// report's own accuracy bound rather than assumed.
#[derive(Clone)]
pub struct SmoothLrSelectionReplay {
    /// `ν_k = eig(Ĩ_jj⁻¹ S_jj(λ̂))`, the term's generalized penalty spectrum at
    /// the fitted scale, ascending. `t = 1` is the fit.
    ///
    /// It is published on EVERY lane. The multi-scale lane used to leave it
    /// empty — its draws select points whose basis moves with `t`, so it has no
    /// single diagonalizing spectrum to report *per selected point* — but the
    /// FITTED point always has one, it is the object the whole replay is built
    /// on, and a consumer asking what was replayed is asking about that. An
    /// empty vector there was an accident of which lane ran, not a statement
    /// about the term.
    pub generalized: Vec<f64>,
    /// `W(λ̂(u))` over the draws, IN DRAW ORDER.
    selection_sample: Vec<f64>,
    /// `W(1)` over the SAME draws, in the same order — the control variate.
    ///
    /// The order is the pairing, and the pairing is the whole point: sorting
    /// either sample would leave the two counts correct and destroy the paired
    /// difference whose variance is what makes this a control variate rather
    /// than two independent estimates.
    conditional_sample: Vec<f64>,
    /// The OBSERVED data scored the way every draw is: its whitened block
    /// score run through the same selection, read at the selected `t` and at
    /// `t = 1`. `None` when the caller supplied no score.
    ///
    /// # Why the observed statistic has to be scored by the replay's rule
    ///
    /// The sample above is `W(λ̂(u))` under the REPLAY's selection — the
    /// certified global minimum of its criterion over the window. The observed
    /// `W` is the statistic at the FIT's `λ̂`, which the outer REML search chose
    /// on the real surface. Comparing the two is only a p-value when they are
    /// the same functional of the data, and for a null term they are not: on a
    /// flat surface the fit stops at an interior `λ̂` with `W ≈ 1e-5`, while the
    /// replay's rule sends the same data to the wall with `W ≈ 1e-9`. Every
    /// draw that rails then sits BELOW the observed value, and the p-value is
    /// the non-railing fraction on almost every such replicate — a mass near
    /// `0.5` on the Gaussian cells and a conservative one near `1` on the
    /// binomial and Poisson ones.
    ///
    /// Scoring the observation by the replay's own rule makes the comparison
    /// one functional against its own law. The quadratic model the replay runs
    /// is `W_q(t; z) = Σ z²(1 − f(t)²)`, and the observed `W` is that model's
    /// value at `t = 1` to first order, so the observed statistic under the
    /// replay's selection is `W · W_q(t*; z)/W_q(1; z)`: the fit's own exact
    /// `W`, carried to the replay's selected point by the quadratic model's own
    /// ratio. At `t* = 1` the ratio is one and nothing moves.
    observed: Option<ObservedSelection>,
}

/// `W_q` of the observed whitened score, at `t = 1` and at the replay's
/// selected `t*` — see [`SmoothLrSelectionReplay::observed`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct ObservedSelection {
    conditional: f64,
    selected: f64,
}

impl std::fmt::Debug for SmoothLrSelectionReplay {
    /// The two samples are thousands of draws each and are never what a reader
    /// of a failure message wants; the spectrum that generated them is.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmoothLrSelectionReplay")
            .field("generalized", &self.generalized)
            .field("draws", &self.selection_sample.len())
            .field("observed", &self.observed)
            .finish()
    }
}

impl PartialEq for SmoothLrSelectionReplay {
    fn eq(&self, other: &Self) -> bool {
        self.generalized == other.generalized
            && self.selection_sample == other.selection_sample
            && self.conditional_sample == other.conditional_sample
            && self.observed == other.observed
    }
}

/// Draws used to generate the selection replay.
///
/// The replay's only error is the Monte-Carlo error of the control-variate
/// DIFFERENCE, not of a tail — the tail itself is inverted exactly. That
/// difference is a mean of indicators that agree on all but the draws whose
/// selected `t` moves them across the threshold, so its standard error is a
/// fraction of `√(p(1−p)/N)`. At this budget that raw bound is `3.4e-3` at
/// `p = 0.05`, and the measured control-variate standard error on the fixtures
/// is between one and two orders below it. It is measured and published per
/// query rather than assumed, so a caller never has to take this number on
/// trust — and the cost is one certified 1-D global search per draw on the
/// common-scale lane.
const SMOOTH_LR_SELECTION_DRAWS: usize = 4096;

/// Draws for the multi-scale replay.
///
/// Fewer than the one-dimensional path's, and the reason is arithmetic rather
/// than a different accuracy target: a common-scale draw is ONE certified 1-D
/// search over a criterion diagonalized once per term, and a multi-scale draw is
/// a joint trust-region Newton descent that refactors an `r × r` pair of
/// factorizations at every step it tries. The standard error it leaves is
/// measured and published per query — a coarser replay that says how coarse it
/// is beats a finer one nobody can afford to run.
const SMOOTH_LR_MULTISCALE_DRAWS: usize = 2048;

/// The term's penalty geometry in the basis the replay's criterion lives in:
/// whitened by the Schur-complemented information and factored into ROOTS.
///
/// # Why roots, and why this is not an implementation detail
///
/// The replayed criterion is
///
/// ```text
/// V(t) = ½ Σ_j c_j² e_j/(1 + e_j)  +  ½ [ log|I + T(t)| − log|T(t)|₊ ],
/// T(t) = Σ_i t_i λ̂_i · Wᵀ S_i W,   e = eig T(t),
/// ```
///
/// and the bracket is its whole Occam half — the only term that stops the
/// selection running to `t → 0`. The first summand of the bracket is benign: a
/// mode `e` carries `log(1 + e)`, so an error of `ε‖T‖` in a mode near zero
/// costs `ε‖T‖`. The SECOND is not, and this repo has already written down why,
/// in `penalty_logdet.rs`'s `SpectrumScale` (#2644):
///
/// > `S_λ = Σ_k λ_k S_k` is a SUM OF SQUARES, so forming it squares the
/// > conditioning of the objects it is built from … Every backward-stable
/// > factorization of the ASSEMBLED matrix therefore prices `log|S_λ|₊` to
/// > `O(ε·κ(S_λ))`, while the same quantity taken from the stacked scaled ROOTS
/// > costs `O(ε·√κ(S_λ))`. The outer smoothing search routinely drives
/// > `κ(S_λ)` past `1e14` (one λ at its ceiling beside a null-space shrinkage λ
/// > near zero is enough) …
///
/// The parenthesis is this fixture. A default `s(z)` is a DOUBLE-penalty smooth
/// (wiggliness plus a null-space ridge), and a null-true smooth is exactly the
/// fit that rails the first λ up and the second down. Measured on a whitened
/// `q = 9` pair at the separations the box allows, the assembled route against
/// the root route:
///
/// ```text
/// ρ̂ = (0, 0)      offset  20.564 vs  20.564     error   0.000
/// ρ̂ = (12, −12)   offset  29.813 vs  29.813     error   0.000
/// ρ̂ = (18, −24)   offset  19.189 vs  53.811     error −34.623
/// ρ̂ = (29, −29)   offset   8.103 vs  63.811     error −55.709
/// ```
///
/// and the error is not a perturbation of the selection, it REPLACES it: the
/// dropped modes are the ones carrying `−ln t_i`, i.e. the coercivity that makes
/// the criterion blow up as `t_i → 0`, so their loss makes the criterion
/// monotone in `ln t_i` and the replay picks a wall. That is the same mechanism
/// `from_components` documents under #1237, reached from the replay's side.
///
/// So the geometry is carried as `R_i` with `R_iᵀ R_i = Wᵀ S_i W` for the
/// term's λ-FREE components, plus their `ρ̂_i` and the structural rank of their
/// SUM. Everything the replay needs at a point — the eigenbasis, the
/// shares, the statistic's weights and the criterion's log-determinant — is then
/// one thin SVD of the stacked scaled roots.
///
/// # Why the rank has to be structural
///
/// `log|T|₊` is a sum over `range(T)`, and `range(T)` does not depend on `t`
/// (every `t_i > 0`). Deciding membership by `e_j > 0` — which is what both
/// replay lanes did — asks a floating-point comparison to separate a structural
/// zero from a mode `1e18` below the largest one, and it answers at random:
/// a noise-negative mode silently drops a `log(1 + 1/e) ≈ 24` contribution, and
/// a noise-positive one invents `log(1 + 1e16) ≈ 37`. The rank is taken once,
/// from the λ-free stacked roots, where the spectrum is well scaled.
struct SelectionGeometry {
    /// `R_i`, `rank_i × q`, with `R_iᵀR_i = Wᵀ S_i W` for the term's λ-free
    /// penalty component `i`. Rows are `√σ · uᵀ` over that COMPONENT's own
    /// modes above its OWN relative floor, so the truncation never sees the λ
    /// dynamic range.
    roots: Vec<Array2<f64>>,
    /// `ρ̂_i = ln λ̂_i`, the fitted scale of component `i`. `ln t_i = 0` is the
    /// fit.
    log_lambda: Vec<f64>,
    /// `q`, the tested block's identified dimension after whitening.
    dimension: usize,
    /// `rank(Σ_i Wᵀ S_i W)`, `t`-independent — the index set `log|T|₊` runs
    /// over.
    rank: usize,
    /// Rows of the stacked root matrix, at least `dimension` so the thin SVD's
    /// right factor is a full orthonormal basis of the block (the padding rows
    /// are zero and change nothing else).
    stacked_rows: usize,
    /// `U`, `q × rank`: an orthonormal basis of `range(Σ_i Wᵀ S_i W)`, the
    /// `t`-free subspace `log|T|₊` runs over.
    ///
    /// Taken from the same UNIT stacked-roots decomposition the rank is, so the
    /// two cannot disagree about which directions are structural.
    range_basis: Array2<f64>,
    /// `R_i U`, `rank_i × rank`: each component's root already expressed in the
    /// range basis, so a scaled stack of them is `M(t)` with
    /// `M(t)ᵀM(t) = Uᵀ T(t) U` — full rank by construction, which is what makes
    /// its triangular factor a pseudo-determinant rather than an approximation
    /// to one.
    range_roots: Vec<Array2<f64>>,
}

/// The criterion and the statistic at one `t`, WITHOUT an eigenbasis.
///
/// # Why a second evaluator exists beside [`SelectionGeometry::at`]
///
/// `at` returns the full eigensystem, which is the right object when every draw
/// is asked about the same point: one `O(q³)` decomposition is amortized over
/// thousands of `O(q²)` projections. The multi-scale lane inverts that ratio —
/// each draw ends at its own selected point — and an eigendecomposition per draw
/// is twenty times the arithmetic the answer needs, in allocations as much as in
/// flops.
///
/// Everything the replay reads at a point is available from two triangular
/// factorizations of `r × r` objects, `r = rank(T)`:
///
/// ```text
/// C(t) = Uᵀ T(t) U = RᵀR,     R = qr(M(t)),   M(t) = [√(t_iλ̂_i)·R_iU ; …]
/// I + C = R̃ᵀR̃,               R̃ = qr([M(t); I])
/// D    = (I + C)⁻¹ C,         v = Uᵀu
/// criterion = vᵀDv + log|I + C| − log|C|
/// statistic = ‖u‖² − ‖Dv‖²
/// ```
///
/// The identities are exact, not approximations of the eigen route: `D`'s
/// eigenvalues are the shares `f_j = e_j/(1 + e_j)`, so `vᵀDv` is
/// `Σ_j c_j² f_j`; `‖u‖² − ‖Dv‖²` is `Σ_j c_j²(1 − f_j²) = Σ_j w_j c_j²`
/// because `w_j = 2f̄_j − f̄_j² = 1 − f_j²`; and a direction outside `range(T)`
/// has `f = 0`, so it drops out of the first and carries its full `c²` in the
/// second, exactly as the eigen route's `log(1 + 0) = 0` and `w = 1` do.
///
/// # Where the conditioning goes
///
/// Neither log-determinant is read off an assembled sum (#2644, #2902):
///
/// * `log|C|` is taken from the TRIANGULAR FACTOR of the scaled roots. `κ(C)`
///   reaches `e^{60}` on a null-true double-penalty smooth, where an assembled
///   Cholesky has no small pivots left to speak of.
/// * `log|I + C|` and `D` are taken from the triangular factor of the roots
///   bordered by `I`. An assembled `I + C` carries an ABSOLUTE error of `ε‖C‖`
///   into every mode, and `‖C‖` follows the largest scale: on a dense pair at a
///   scale separation of 40 it moved the criterion by `4.2e-6` against an exact
///   axis slice, where the bordered factor agrees to `1e-14`.
///
/// Both reductions stack their blocks in decreasing scale. A Householder
/// reduction of rows graded by `e^{30}` keeps each small row accurate to its own
/// scale only when every row it is eliminated against precedes it; stacked the
/// other way, the same bordered factor misses by `4.7e-5` on that pair.
struct SelectionFactor {
    /// `r`, the structural rank.
    rank: usize,
    /// `M(t)`, overwritten in place by the Householder reduction that reads its
    /// triangular factor.
    stacked: Array2<f64>,
    /// `R`'s diagonal, which the reduction overwrites in `stacked`.
    diagonal: Vec<f64>,
    /// `[M(t); I]`, overwritten in place by its Householder reduction.
    bordered: Array2<f64>,
    /// `R̃`'s diagonal, which the reduction overwrites in `bordered`.
    bordered_diagonal: Vec<f64>,
    /// `R̃ᵀ`, the lower factor of `I + C`.
    factor: Array2<f64>,
    /// The blocks in decreasing scale, the identity as the last index.
    order: Vec<usize>,
    /// `log|I + T| − log|T|₊`, the criterion's `t`-dependent Occam term.
    offset: f64,
}

/// One grid point of the replay: the data operator's eigenvalues, the statistic's
/// weights, and the basis both are diagonal in.
struct SelectionPoint {
    /// `e_j = eig T(t)`, descending.
    eigenvalues: Vec<f64>,
    /// `w_j = 2f̄_j − f̄_j²` with `f̄ = 1 − f`, the statistic's null weights.
    weights: Vec<f64>,
    /// Columns are the eigenvectors of `T(t)`, in the same order.
    basis: Array2<f64>,
}

impl SelectionGeometry {
    /// Whiten the term's λ-free penalty components by the Schur-complemented
    /// information and factor each into its own root.
    ///
    /// Declines with `NoInformation` when the information has no identified
    /// direction and `NoPenaltyComponents` when the whitened components leave
    /// nothing penalized — in both cases there is nothing to replay. A
    /// decomposition that refuses, or a factor that is non-finite or
    /// inconsistent in shape, is `GeometryRefused`.
    fn whiten(
        whitener: &Array2<f64>,
        unit_penalties: &[Array2<f64>],
        log_lambda: &[f64],
    ) -> Result<Self, SmoothLrSelectionDecline> {
        use SmoothLrSelectionDecline::{GeometryRefused, NoInformation, NoPenaltyComponents};
        if unit_penalties.is_empty() || unit_penalties.len() != log_lambda.len() {
            return Err(NoPenaltyComponents);
        }
        let dimension = whitener.ncols();
        if dimension == 0 || whitener.nrows() == 0 {
            return Err(NoInformation);
        }

        let mut roots = Vec::with_capacity(unit_penalties.len());
        for penalty in unit_penalties {
            if penalty.nrows() != whitener.nrows() || penalty.ncols() != whitener.nrows() {
                return Err(GeometryRefused);
            }
            // `Wᵀ S W` is symmetric as a mathematical object and its two
            // triangles differ only by summation order — but
            // `strict_symmetric_eigh` REFUSES the input rather than symmetrizing
            // it for the caller, which is the right contract and the reason this
            // is explicit here. Dropping it is a silent `GeometryRefused` on
            // every real fit.
            let whitened = symmetrized(whitener.t().dot(penalty).dot(whitener));
            if whitened.iter().any(|value| !value.is_finite()) {
                return Err(GeometryRefused);
            }
            roots.push(psd_root(&whitened).ok_or(GeometryRefused)?);
        }
        if roots.iter().all(|root| root.nrows() == 0) {
            return Err(NoPenaltyComponents);
        }
        let stacked_rows = roots.iter().map(|root| root.nrows()).sum::<usize>();
        // `range(Σ_i S̃_i)` is what `log|T|₊` runs over, and it is `t`-free. Taken
        // from the UNIT stacked roots, whose singular values span only the
        // components' own conditioning — the λ ratio that makes the assembled
        // sum unreadable is not present here at all.
        let unit = stack_roots(&roots, &vec![0.0; roots.len()], stacked_rows.max(dimension));
        let (_, unit_singular, unit_right) =
            gam_linalg::faer_ndarray::FaerSvd::svd(&unit, false, true)
                .map_err(|_| GeometryRefused)?;
        let unit_largest = unit_singular.iter().copied().fold(0.0_f64, f64::max);
        let bar = unit_largest * (dimension as f64) * f64::EPSILON * 100.0;
        let rank = unit_singular.iter().filter(|&&value| value > bar).count();
        if rank == 0 {
            return Err(NoPenaltyComponents);
        }
        // The same decomposition that decided the rank also names the subspace:
        // the leading `rank` right singular vectors of the UNIT stack span
        // `range(Σ_i S̃_i)`. Deciding the two from one object is what stops them
        // disagreeing about which directions are structural.
        let unit_right = unit_right.ok_or(GeometryRefused)?;
        if unit_right.nrows() < rank || unit_right.ncols() != dimension {
            return Err(GeometryRefused);
        }
        let mut range_basis = Array2::<f64>::zeros((dimension, rank));
        for column in 0..rank {
            for row in 0..dimension {
                range_basis[[row, column]] = unit_right[[column, row]];
            }
        }
        let range_roots: Vec<Array2<f64>> =
            roots.iter().map(|root| root.dot(&range_basis)).collect();
        Ok(Self {
            roots,
            log_lambda: log_lambda.to_vec(),
            dimension,
            rank,
            stacked_rows: stacked_rows.max(dimension),
            range_basis,
            range_roots,
        })
    }

    /// Evaluate the geometry at one grid point `ln t`.
    ///
    /// One thin SVD of the stacked scaled roots supplies all four quantities:
    /// `eig T = σ²`, the eigenbasis is the right singular factor, and the
    /// log-determinants are `Σ log(1 + σ²)` and `2 Σ_{j < rank} log σ` — the
    /// second over the structural rank rather than over a sign test.
    fn at(&self, log_t: &[f64]) -> Option<SelectionPoint> {
        if log_t.len() != self.roots.len() {
            return None;
        }
        let scaled: Vec<f64> = self
            .log_lambda
            .iter()
            .zip(log_t.iter())
            .map(|(rho, shift)| rho + shift)
            .collect();
        let stacked = stack_roots(&self.roots, &scaled, self.stacked_rows);
        let (_, singular, right) =
            gam_linalg::faer_ndarray::FaerSvd::svd(&stacked, false, true).ok()?;
        // Thin SVD on a matrix with at least `dimension` rows: the right factor
        // is `dimension × dimension`, so every direction of the block — including
        // the ones the penalty never reaches — has a basis vector.
        let right = right?;
        if right.nrows() != self.dimension || singular.len() < self.dimension {
            return None;
        }
        let mut eigenvalues = Vec::with_capacity(self.dimension);
        let mut weights = Vec::with_capacity(self.dimension);
        let mut log_det_hessian = 0.0_f64;
        let mut log_det_penalty = 0.0_f64;
        for index in 0..self.dimension {
            let sigma = singular[index];
            if !sigma.is_finite() || sigma < 0.0 {
                return None;
            }
            let eigenvalue = sigma * sigma;
            log_det_hessian += eigenvalue.ln_1p();
            if index < self.rank {
                if !(sigma > 0.0) {
                    return None;
                }
                log_det_penalty += 2.0 * sigma.ln();
            }
            let fraction = if eigenvalue.is_finite() {
                eigenvalue / (1.0 + eigenvalue)
            } else {
                1.0
            };
            let shrinkage = 1.0 - fraction;
            eigenvalues.push(eigenvalue);
            weights.push(2.0 * shrinkage - shrinkage * shrinkage);
        }
        let offset = log_det_hessian - log_det_penalty;
        if !offset.is_finite() {
            return None;
        }
        // Columns are eigenvectors: `right` is `Vᵀ` from the thin SVD, so its
        // ROWS are the right singular vectors.
        let mut basis = Array2::<f64>::zeros((self.dimension, self.dimension));
        for column in 0..self.dimension {
            for row in 0..self.dimension {
                basis[[row, column]] = right[[column, row]];
            }
        }
        Some(SelectionPoint {
            eigenvalues,
            weights,
            basis,
        })
    }
}

impl SelectionFactor {
    /// Buffers sized for one geometry. Allocated once per replay, never inside
    /// the loop the refinement spends its time in.
    fn new(geometry: &SelectionGeometry) -> Self {
        let rank = geometry.rank;
        let rows = geometry
            .range_roots
            .iter()
            .map(|root| root.nrows())
            .sum::<usize>();
        Self {
            rank,
            stacked: Array2::zeros((rows.max(rank), rank)),
            diagonal: vec![0.0; rank],
            bordered: Array2::zeros((rows + rank, rank)),
            bordered_diagonal: vec![0.0; rank],
            factor: Array2::zeros((rank, rank)),
            order: Vec::with_capacity(geometry.range_roots.len() + 1),
            offset: 0.0,
        }
    }

    /// Factor the geometry at `ln t`. `false` means the point is unusable and
    /// the caller must not read the scores.
    fn refactor(&mut self, geometry: &SelectionGeometry, log_t: &[f64]) -> bool {
        let components = geometry.range_roots.len();
        if log_t.len() != components {
            return false;
        }
        // Block `components` is the identity of `[M; I]`, at scale one.
        let log_scale = |block: usize| {
            if block == components {
                0.0
            } else {
                0.5 * (geometry.log_lambda[block] + log_t[block])
            }
        };
        for block in 0..components {
            // `exp(s/2)` rather than `sqrt(exp(s))`, so a `λ̂` at the box wall
            // never round-trips through an intermediate that overflows.
            if !log_scale(block).exp().is_finite() {
                return false;
            }
        }
        // Rows in decreasing scale (see the type's doc): each block goes in
        // before every block it outweighs.
        self.order.clear();
        self.order.extend(0..=components);
        self.order
            .sort_by(|&left, &right| log_scale(right).total_cmp(&log_scale(left)));
        self.stacked.fill(0.0);
        self.bordered.fill(0.0);
        let (mut stacked_row, mut bordered_row) = (0usize, 0usize);
        for &block in &self.order {
            if block == components {
                for index in 0..self.rank {
                    self.bordered[[bordered_row + index, index]] = 1.0;
                }
                bordered_row += self.rank;
                continue;
            }
            let root = &geometry.range_roots[block];
            let scale = log_scale(block).exp();
            for row in 0..root.nrows() {
                for column in 0..self.rank {
                    let value = scale * root[[row, column]];
                    self.stacked[[stacked_row + row, column]] = value;
                    self.bordered[[bordered_row + row, column]] = value;
                }
            }
            stacked_row += root.nrows();
            bordered_row += root.nrows();
        }
        let Some(log_determinant) =
            householder_triangularize(&mut self.stacked, &mut self.diagonal)
        else {
            return false;
        };
        let Some(log_hessian) =
            householder_triangularize(&mut self.bordered, &mut self.bordered_diagonal)
        else {
            return false;
        };
        // `R̃ᵀR̃ = [M; I]ᵀ[M; I] = I + C`, kept as the lower factor `R̃ᵀ`.
        for row in 0..self.rank {
            for column in 0..self.rank {
                self.factor[[row, column]] = match row.cmp(&column) {
                    std::cmp::Ordering::Equal => self.bordered_diagonal[row],
                    std::cmp::Ordering::Greater => self.bordered[[column, row]],
                    std::cmp::Ordering::Less => 0.0,
                };
            }
        }
        self.offset = 2.0 * (log_hessian - log_determinant);
        self.offset.is_finite()
    }

    /// `(criterion, statistic)` for one draw, from its coordinates in the range
    /// basis and the squared norm of the WHOLE draw.
    ///
    /// The norm carries the directions the penalty never reaches: they are
    /// absent from `projected` (which lives in `range(T)`) and they contribute
    /// nothing to the criterion and their full square to the statistic.
    fn score(&mut self, projected: &[f64], norm_squared: f64) -> (f64, f64) {
        // `D = (I + C)⁻¹C = I − (I + C)⁻¹`, so `vᵀDv = ‖v‖² − ‖R̃⁻ᵀv‖²` and
        // `D v = v − (I + C)⁻¹v`, without ever forming `C v`.
        let whitened =
            gam_linalg::triangular::forward_substitution_lower_vector(&self.factor, projected);
        let solved = gam_linalg::triangular::cholesky_solve_vector(&self.factor, projected);
        let mut data = 0.0_f64;
        let mut mapped_norm = 0.0_f64;
        for row in 0..self.rank {
            data += projected[row] * projected[row] - whitened[row] * whitened[row];
            let mapped = projected[row] - solved[row];
            mapped_norm += mapped * mapped;
        }
        (self.offset + data, norm_squared - mapped_norm)
    }

    /// The Wald statistic `uᵀ(I + T)⁻¹u` for one draw, from the same inputs as
    /// [`Self::score`].
    ///
    /// On `range(T)` the form is `vᵀ(I + C)⁻¹v = ‖R̃⁻ᵀv‖²`, and off it the draw
    /// keeps its full square, so the statistic is `‖u‖² − (‖v‖² − ‖R̃⁻ᵀv‖²)`.
    fn wald(&self, projected: &[f64], norm_squared: f64) -> f64 {
        let whitened =
            gam_linalg::triangular::forward_substitution_lower_vector(&self.factor, projected);
        let mut data = 0.0_f64;
        for row in 0..self.rank {
            data += projected[row] * projected[row] - whitened[row] * whitened[row];
        }
        norm_squared - data
    }

    /// The draw's criterion at the point last [`Self::refactor`]ed, with its exact
    /// gradient and Hessian in `ρ = ln t` over EVERY scale, and the rounding bands
    /// a Newton-decrement certificate is decided against.
    ///
    /// # The derivatives are read off the two factorizations already formed
    ///
    /// With `C_j = e^{ρ̂_j + ρ_j} Uᵀ S̃_j U` the part of `C` scale `j` owns,
    /// `∂C/∂ρ_j = C_j` and `∂C_j/∂ρ_k = δ_jk C_j`. Write `M = QR` and
    /// `[M; I] = Q̃R̃` for the two reductions, `Q_j` and `Q̃_j` for the rows of each
    /// thin factor that block `j` occupies, and
    ///
    /// ```text
    /// G_j = Q_jᵀQ_j,   G̃_j = Q̃_jᵀQ̃_j,   w = R̃⁻ᵀv,   y_j = G̃_j w.
    /// ```
    ///
    /// Then `C_j = RᵀG_jR = R̃ᵀG̃_jR̃`, so `C⁻¹C_j = R⁻¹G_jR` and
    /// `(I + C)⁻¹C_j = R̃⁻¹G̃_jR̃`, and the criterion
    /// `log|I + C| − log|C| + ‖v‖² − vᵀ(I + C)⁻¹v` differentiates to
    ///
    /// ```text
    /// g_j  = tr G̃_j − tr G_j + w·y_j,
    /// H_jk = δ_jk g_j − ⟨G̃_j, G̃_k⟩ + ⟨G_j, G_k⟩ − 2 y_j·y_k.
    /// ```
    ///
    /// Every quantity is an inner product of rows of an orthonormal factor, so it
    /// inherits the row-graded accuracy the reductions were ordered for (see the
    /// type's doc) and never touches an assembled `C` or an inverse of `R`.
    ///
    /// # Bands
    ///
    /// Each entry is an accumulation over the reductions' `rows · r` rounded
    /// operations (the Householder backward-error bound, Higham Thm 19.4), so every
    /// band is [`gam_linalg::roundoff::accumulation_band`] at that depth over the
    /// absolute sum of the entry's own summands. The Hessian's band is charged to
    /// the Frobenius norm of the matrix of those sums, which bounds its spectral
    /// norm.
    fn jet(
        &self,
        geometry: &SelectionGeometry,
        projected: &[f64],
    ) -> Option<opt::SecondOrderSample> {
        let rank = self.rank;
        let components = geometry.range_roots.len();
        if projected.len() != rank {
            return None;
        }
        let stacked_q = householder_thin_q(&self.stacked);
        let bordered_q = householder_thin_q(&self.bordered);
        let whitened =
            gam_linalg::triangular::forward_substitution_lower_vector(&self.factor, projected);
        // Each block's rows sit where `refactor` put them.
        let mut stacked_start = vec![0usize; components];
        let mut bordered_start = vec![0usize; components];
        let (mut stacked_row, mut bordered_row) = (0usize, 0usize);
        for &block in &self.order {
            if block == components {
                bordered_row += rank;
                continue;
            }
            stacked_start[block] = stacked_row;
            bordered_start[block] = bordered_row;
            stacked_row += geometry.range_roots[block].nrows();
            bordered_row += geometry.range_roots[block].nrows();
        }
        let gram = |orthonormal: &Array2<f64>, start: usize, rows: usize| {
            let block = orthonormal.slice(ndarray::s![start..start + rows, ..]);
            block.t().dot(&block)
        };
        let mut plain = Vec::with_capacity(components);
        let mut bordered = Vec::with_capacity(components);
        let mut mapped = Vec::with_capacity(components);
        for component in 0..components {
            let rows = geometry.range_roots[component].nrows();
            plain.push(gram(&stacked_q, stacked_start[component], rows));
            let own = gram(&bordered_q, bordered_start[component], rows);
            mapped.push(own.dot(&whitened));
            bordered.push(own);
        }

        let mut value = self.offset;
        let mut magnitude = 0.0_f64;
        for index in 0..rank {
            value += projected[index] * projected[index] - whitened[index] * whitened[index];
            magnitude += 2.0
                * (self.diagonal[index].abs().ln().abs()
                    + self.bordered_diagonal[index].abs().ln().abs())
                + projected[index] * projected[index]
                + whitened[index] * whitened[index];
        }
        let depth = self.bordered.nrows() * rank;
        let mut gradient = Array1::<f64>::zeros(components);
        let mut gradient_band = Array1::<f64>::zeros(components);
        let mut diagonal_sum = vec![0.0_f64; components];
        for component in 0..components {
            let own_trace = bordered[component].diag().sum();
            let plain_trace = plain[component].diag().sum();
            let quadratic = whitened.dot(&mapped[component]);
            gradient[component] = own_trace - plain_trace + quadratic;
            diagonal_sum[component] = own_trace + plain_trace + quadratic.abs();
            gradient_band[component] =
                gam_linalg::roundoff::accumulation_band(depth, diagonal_sum[component]);
        }
        let frobenius = |left: &Array2<f64>, right: &Array2<f64>| {
            left.iter()
                .zip(right.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let mut hessian = Array2::<f64>::zeros((components, components));
        let mut hessian_sums = 0.0_f64;
        for row in 0..components {
            for column in row..components {
                let own = frobenius(&bordered[row], &bordered[column]);
                let base = frobenius(&plain[row], &plain[column]);
                let coupling = mapped[row].dot(&mapped[column]);
                let mut entry = -own + base - 2.0 * coupling;
                let mut sum = own.abs() + base.abs() + 2.0 * coupling.abs();
                if row == column {
                    entry += gradient[row];
                    sum += diagonal_sum[row];
                }
                hessian[[row, column]] = entry;
                hessian[[column, row]] = entry;
                let copies = if row == column { 1.0 } else { 2.0 };
                hessian_sums += copies * sum * sum;
            }
        }
        let finite = value.is_finite()
            && gradient.iter().all(|entry| entry.is_finite())
            && hessian.iter().all(|entry| entry.is_finite());
        finite.then(|| opt::SecondOrderSample {
            value,
            gradient,
            hessian: Some(hessian),
            decrement_bands: Some(opt::DecrementBands {
                objective: gam_linalg::roundoff::accumulation_band(depth, magnitude),
                gradient: gradient_band,
                hessian: gam_linalg::roundoff::accumulation_band(depth, hessian_sums.sqrt()),
            }),
        })
    }
}

/// One draw's multi-scale criterion as a function of its OPEN scales, for
/// [`opt::NewtonTrustRegion`]. A scale whose window is empty stays at the fitted
/// point, `ln t = 0`.
struct MultiscaleCriterion<'a> {
    geometry: &'a SelectionGeometry,
    factor: &'a mut SelectionFactor,
    coordinates: &'a [f64],
    open: &'a [usize],
    point: Vec<f64>,
}

impl MultiscaleCriterion<'_> {
    /// Factor the geometry at the full point `x` names. A point the reductions
    /// refuse — a scale that overflows, a column that collapses — is outside the
    /// criterion's domain, which the trust region answers by shrinking.
    fn factor_at(&mut self, x: &Array1<f64>) -> Result<(), opt::ObjectiveEvalError> {
        for (slot, &axis) in self.open.iter().enumerate() {
            self.point[axis] = x[slot];
        }
        if self.factor.refactor(self.geometry, &self.point) {
            Ok(())
        } else {
            Err(opt::ObjectiveEvalError::recoverable(
                "the selection criterion cannot be factored at this point",
            ))
        }
    }
}

impl opt::ZerothOrderObjective for MultiscaleCriterion<'_> {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, opt::ObjectiveEvalError> {
        self.factor_at(x)?;
        let norm_squared = self.coordinates.iter().map(|value| value * value).sum();
        Ok(self.factor.score(self.coordinates, norm_squared).0)
    }
}

impl opt::FirstOrderObjective for MultiscaleCriterion<'_> {
    fn eval_grad(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<opt::FirstOrderSample, opt::ObjectiveEvalError> {
        let sample = opt::SecondOrderObjective::eval_hessian(self, x)?;
        Ok(opt::FirstOrderSample {
            value: sample.value,
            gradient: sample.gradient,
        })
    }
}

impl opt::SecondOrderObjective for MultiscaleCriterion<'_> {
    fn eval_hessian(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<opt::SecondOrderSample, opt::ObjectiveEvalError> {
        self.factor_at(x)?;
        let full = self
            .factor
            .jet(self.geometry, self.coordinates)
            .ok_or_else(|| {
                opt::ObjectiveEvalError::recoverable(
                    "the selection criterion's derivatives are not finite at this point",
                )
            })?;
        let open = self.open;
        let restrict = |full: &Array1<f64>| Array1::from_iter(open.iter().map(|&axis| full[axis]));
        let hessian = full.hessian.as_ref().map(|hessian| {
            Array2::from_shape_fn((open.len(), open.len()), |(row, column)| {
                hessian[[open[row], open[column]]]
            })
        });
        let bands = full
            .decrement_bands
            .as_ref()
            .map(|bands| opt::DecrementBands {
                objective: bands.objective,
                gradient: restrict(&bands.gradient),
                hessian: bands.hessian,
            });
        Ok(opt::SecondOrderSample {
            value: full.value,
            gradient: restrict(&full.gradient),
            hessian,
            decrement_bands: bands,
        })
    }
}

/// Overwrite `matrix` (`n × r`, `n ≥ r`) with the Householder reduction whose
/// triangular factor `R` satisfies `RᵀR = matrixᵀmatrix`, writing `R`'s diagonal
/// to `diagonal` and returning `Σ_j ln|R_jj| = ½ log det(MᵀM)`.
///
/// The strictly-upper triangle of the leading `r × r` block holds `R`'s
/// off-diagonal entries on return; the diagonal cells are left holding the
/// reflector vectors and must be read from `diagonal`.
///
/// `None` when a column collapses — for a matrix of full column rank by
/// construction that is a statement about the input, not a tolerance, so the
/// caller refuses the point rather than continuing with a determinant it cannot
/// price.
fn householder_triangularize(matrix: &mut Array2<f64>, diagonal: &mut [f64]) -> Option<f64> {
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    if rows < columns || diagonal.len() != columns {
        return None;
    }
    let mut log_determinant = 0.0_f64;
    for pivot in 0..columns {
        let mut norm_squared = 0.0_f64;
        for row in pivot..rows {
            norm_squared += matrix[[row, pivot]] * matrix[[row, pivot]];
        }
        let norm = norm_squared.sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return None;
        }
        log_determinant += norm.ln();
        // Reflect onto `−sign(x_pivot)·‖x‖ e₁`, the sign that avoids
        // cancellation in `x_pivot − α`.
        let alpha = if matrix[[pivot, pivot]] > 0.0 {
            -norm
        } else {
            norm
        };
        diagonal[pivot] = alpha;
        matrix[[pivot, pivot]] -= alpha;
        let mut reflector_squared = 0.0_f64;
        for row in pivot..rows {
            reflector_squared += matrix[[row, pivot]] * matrix[[row, pivot]];
        }
        if reflector_squared <= 0.0 {
            continue;
        }
        for column in (pivot + 1)..columns {
            let mut inner = 0.0_f64;
            for row in pivot..rows {
                inner += matrix[[row, pivot]] * matrix[[row, column]];
            }
            let scale = 2.0 * inner / reflector_squared;
            for row in pivot..rows {
                matrix[[row, column]] -= scale * matrix[[row, pivot]];
            }
        }
    }
    log_determinant.is_finite().then_some(log_determinant)
}

/// The thin orthonormal factor `Q` (`n × r`) of the reduction
/// [`householder_triangularize`] left in `reduced`, so that `M = QR`.
///
/// The reflectors are still in place — column `j` from its diagonal cell down,
/// the cell holding `x_j − α_j` — and `Q = H_0 ⋯ H_{r−1} [I; 0]` applies them in
/// reverse. Forming `Q` this way keeps it orthonormal to working precision,
/// which `M R⁻¹` would not once `R` carries the λ ratio.
fn householder_thin_q(reduced: &Array2<f64>) -> Array2<f64> {
    let (rows, columns) = reduced.dim();
    let mut orthonormal = Array2::<f64>::zeros((rows, columns));
    for index in 0..columns.min(rows) {
        orthonormal[[index, index]] = 1.0;
    }
    for pivot in (0..columns).rev() {
        let mut reflector_squared = 0.0_f64;
        for row in pivot..rows {
            reflector_squared += reduced[[row, pivot]] * reduced[[row, pivot]];
        }
        if reflector_squared <= 0.0 {
            continue;
        }
        for column in 0..columns {
            let mut inner = 0.0_f64;
            for row in pivot..rows {
                inner += reduced[[row, pivot]] * orthonormal[[row, column]];
            }
            let scale = 2.0 * inner / reflector_squared;
            for row in pivot..rows {
                orthonormal[[row, column]] -= scale * reduced[[row, pivot]];
            }
        }
    }
    orthonormal
}

/// `[√(e^{s_0}) R_0; √(e^{s_1}) R_1; …]`, zero-padded to `rows`.
///
/// The scale is applied as `exp(s/2)` rather than as `sqrt(exp(s))` so a `λ̂` at
/// the box wall never round-trips through an intermediate that overflows.
fn stack_roots(roots: &[Array2<f64>], log_scale: &[f64], rows: usize) -> Array2<f64> {
    let columns = roots
        .iter()
        .map(|root| root.ncols())
        .max()
        .unwrap_or_default();
    let mut stacked = Array2::<f64>::zeros((rows, columns));
    let mut offset = 0usize;
    for (root, &scale) in roots.iter().zip(log_scale.iter()) {
        let factor = (0.5 * scale).exp();
        for row in 0..root.nrows() {
            for column in 0..root.ncols() {
                stacked[[offset + row, column]] = factor * root[[row, column]];
            }
        }
        offset += root.nrows();
    }
    stacked
}

/// A root `R` of a symmetric PSD `S`, `RᵀR = S`, taken from `S`'s OWN
/// eigensystem and truncated at `S`'s own relative noise floor.
///
/// `S` here is always a λ-free whitened penalty component, so its spectrum is
/// well scaled and this is a benign `O(ε)` operation — the dynamic range that
/// makes the weighted sum hard lives in the λ's, not here. This mirrors
/// `penalty_logdet::psd_component_root`, which is private to `gam-solve`;
/// the contract is the same and so is the threshold.
fn psd_root(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let dimension = matrix.nrows();
    if dimension == 0 {
        return Some(Array2::zeros((0, 0)));
    }
    let (values, vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(matrix, faer::Side::Lower).ok()?;
    let largest = values.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    let threshold = 100.0 * (dimension as f64) * f64::EPSILON * largest;
    let kept: Vec<usize> = (0..dimension)
        .filter(|&index| values[index] > threshold)
        .collect();
    let mut root = Array2::<f64>::zeros((kept.len(), dimension));
    for (row, &index) in kept.iter().enumerate() {
        let scale = values[index].sqrt();
        for column in 0..dimension {
            root[[row, column]] = scale * vectors[[column, index]];
        }
    }
    Some(root)
}

/// The symmetric part of a matrix that is symmetric as a mathematical object.
///
/// A congruence `WᵀSW` and an assembled Gram are symmetric by construction and
/// asymmetric by summation order. `strict_symmetric_eigh` validates its input
/// rather than symmetrizing it — a deliberate contract, since a caller handing
/// it a genuinely non-symmetric matrix has a defect — so every congruence on
/// this path passes through here first.
pub fn symmetrized(mut matrix: Array2<f64>) -> Array2<f64> {
    let dimension = matrix.nrows();
    for row in 0..dimension {
        for column in 0..row {
            let mean = 0.5 * (matrix[[row, column]] + matrix[[column, row]]);
            matrix[[row, column]] = mean;
            matrix[[column, row]] = mean;
        }
    }
    matrix
}

/// The tested block's penalty shares AND the whitener the replay needs, from
/// ONE self-adjoint decomposition and with no matrix cancellation anywhere.
pub struct LrTestedBlock {
    /// `p = eig(B^{1/2} S_jj B^{1/2}) ∈ [0, 1]`, ascending.
    pub shares: Vec<f64>,
    /// `W` (`q × dimension`) with `W Wᵀ = Ĩ_jj⁻¹` on the directions the
    /// Schur-complemented information can see, i.e. those with `1 − p > 0`.
    ///
    /// # Why this is not `Ĩ^{-1/2}` computed from `Ĩ`
    ///
    /// `Ĩ_jj = ([H⁻¹]_jj)⁻¹ − S_jj` is the object the derivation is stated
    /// against, and forming it that way is a CANCELLATION of two matrices whose
    /// ratio is `1/(1 − p)`: at `p = 1 − 1e-12` — an ordinary heavily-shrunk
    /// direction of a null-true smooth — the difference is roundoff amplified
    /// twelve orders, and the explicit inverse that produces the first term has
    /// already amplified it once more. Measured on this issue's own `n = 60`
    /// fixture, that route handed the whitening a spectrum whose largest
    /// eigenvalue was spurious, and the relative floor then discarded EVERY
    /// direction the data could see: `q` went `11 → 9` and the replayed law's
    /// mean went `0.96 → 0.0000` while the reference's stayed at `0.96`. The
    /// control variate was then a difference of two different laws.
    ///
    /// The same object is available with no cancellation at all. With
    /// `A = B^{1/2} S B^{1/2} = QΛQᵀ` and `Λ = diag(p)`,
    ///
    /// ```text
    /// B^{1/2} Ĩ B^{1/2} = B^{1/2}(B⁻¹ − S)B^{1/2} = I − A = Q(I − Λ)Qᵀ,
    /// ```
    ///
    /// so `Ĩ⁻¹ = B^{1/2}Q(I − Λ)⁻¹QᵀB^{1/2}` and `W = B^{1/2}Q(I − Λ)^{-1/2}`.
    /// The only subtraction left is the SCALAR `1 − p`, which loses digits
    /// exactly when the direction is genuinely unidentified — and that is then a
    /// statement about the fit rather than an artifact. As a check on the whole
    /// construction, the whitened total penalty at `λ̂` comes out
    /// `(I − Λ)^{-1/2}Λ(I − Λ)^{-1/2} = diag(p/(1 − p))`, i.e. exactly the
    /// generalized spectrum, diagonal, for free.
    pub whitener: Array2<f64>,
    /// `D = B⁻¹W = B^{-1/2}Q(I − Λ)^{-1/2}` (`q × dimension`), the DUAL of the
    /// whitener, over the same kept directions. `Dᵀβ̂_jj` is the tested block's
    /// Schur-profiled SCORE in the whitened basis.
    ///
    /// # Why the score is `B⁻¹β̂`
    ///
    /// With the other blocks profiled out, the penalized estimate of the tested
    /// block is `β̂ = (Ĩ + S)⁻¹ g` for the profiled score `g`, and
    /// `(Ĩ + S)⁻¹ = [H⁻¹]_jj = B` is exactly the Schur complement identity. So
    /// `g = B⁻¹β̂`, and `u = Wᵀg/σ = Dᵀβ̂/σ` is `N(0, WᵀĨW) = N(0, I)` under
    /// the null — the same standard normal the replay draws, read off the fit
    /// rather than simulated. It is formed from `B^{-1/2}` rather than as
    /// `(B⁻¹ − S)W` for the reason [`Self::whitener`] is: that difference is the
    /// cancellation this whole decomposition exists to avoid.
    ///
    /// `None` when `B` has a non-positive eigenvalue, where `B^{-1/2}` does not
    /// exist.
    pub dual: Option<Array2<f64>>,
}

/// Directions the Schur-complemented information cannot separate from the
/// penalty at all.
///
/// `1 − p` is the share of a direction the DATA holds, on a scale where one is
/// unpenalized and zero is fully absorbed by the penalty. It is an eigenvalue of
/// `I − A` with `‖A‖ ≤ 1`, so its own noise floor is absolute rather than
/// relative: `p·ε` for the decomposition, with the same safety factor
/// `positive_eigenvalue_threshold` uses.
const SMOOTH_LR_IDENTIFIED_SHARE_FLOOR_FACTOR: f64 = 100.0;

/// The term's PENALTY SHARES `p = eig([H⁻¹]_jj · S_jj) ∈ [0, 1]`, sorted
/// ascending — the one object every reference on this path is a function of.
///
/// The null weights are `w_j = 1 − p_j²` (see below), and the generalized
/// eigenvalues that drive the selection replay are `ν_j = p_j/(1 − p_j)`, so a
/// single self-adjoint decomposition yields both.
///
/// # Why this is the same spectrum as `eig(2·F_jj − F_jj²)`
///
/// The penalty is block-diagonal by term, so `S_kj = 0` for `k ≠ j` and the
/// tested block of the GLOBAL shrinkage map factors exactly:
///
/// ```text
/// (I − F)_jj = [H⁻¹S]_jj = Σ_k [H⁻¹]_jk S_kj = [H⁻¹]_jj S_jj  =:  P.
/// ```
///
/// Therefore `F_jj = I − P` and `2F_jj − F_jj² = I − (I − F_jj)² = I − P²`, so
/// `w = 1 − eig(P)²` with no approximation anywhere — the same object the trace
/// identities in `lr_null_spectral_moments` summarise, arrived at without
/// forming a non-symmetric matrix.
///
/// # Why it is reachable with a self-adjoint eigensolver
///
/// `P = B S` with `B = [H⁻¹]_jj` symmetric PSD (a principal submatrix of the
/// inverse of a PD Hessian) and `S = S_jj` symmetric PSD. A product of two
/// symmetric PSD matrices is not symmetric, but it is similar to one:
///
/// ```text
/// B^{-1/2} (B S) B^{1/2} = B^{1/2} S B^{1/2},
/// ```
///
/// which is symmetric PSD and is what this computes — via `B = UΛUᵀ` and
/// `B^{1/2} = UΛ^{1/2}Uᵀ` rather than a Cholesky, so a `B` that is singular in
/// some direction (an exactly-unpenalized fit, a rank-deficient block) is a
/// zero eigenvalue rather than a factorization failure. The eigenvalues are real
/// and lie in `[0, 1]` because `F_jj = (Ĩ_jj + S_jj)⁻¹Ĩ_jj` has eigenvalues
/// `c/(c + s)`; they are clamped to that interval against roundoff, and the
/// clamp is the ONLY place a value is altered.
///
/// Returns `None` when either matrix is absent, the block does not fit inside
/// them, or the self-adjoint decomposition refuses — the caller then drops to
/// the two-moment rung rather than scoring against a spectrum it could not
/// compute.
pub fn lr_tested_block(
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &std::ops::Range<usize>,
) -> Option<LrTestedBlock> {
    let (h_inv, s_lambda) = (hessian_inverse?, penalty?);
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end
        || end > h_inv.nrows()
        || end > h_inv.ncols()
        || end > s_lambda.nrows()
        || end > s_lambda.ncols()
    {
        return None;
    }
    // Both blocks are symmetric as mathematical objects; the halves of an
    // assembled Gram/inverse differ only by summation order. Symmetrize
    // explicitly so the self-adjoint entry point receives the matrix it is being
    // asked about rather than one triangle's rounding of it.
    let b = symmetrized(h_inv.slice(ndarray::s![start..end, start..end]).to_owned());
    let s = symmetrized(s_lambda.slice(ndarray::s![start..end, start..end]).to_owned());
    if b.iter().chain(s.iter()).any(|value| !value.is_finite()) {
        return None;
    }
    let dimension = end - start;

    let (b_eigenvalues, b_vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&b, faer::Side::Lower).ok()?;
    // `B^{1/2} = U Λ^{1/2} Uᵀ`. A tiny negative eigenvalue is roundoff on a PSD
    // matrix, so its square root is zero rather than an error.
    let mut root_scaled = b_vectors.clone();
    for (mut column, &eigenvalue) in root_scaled.columns_mut().into_iter().zip(b_eigenvalues.iter())
    {
        let root = eigenvalue.max(0.0).sqrt();
        column.mapv_inplace(|value| value * root);
    }
    let b_root = root_scaled.dot(&b_vectors.t());
    let similar = symmetrized(b_root.dot(&s).dot(&b_root));
    let (shrinkage, shrinkage_vectors) =
        gam_linalg::faer_ndarray::strict_symmetric_eigh(&similar, faer::Side::Lower).ok()?;

    let shares: Vec<f64> = shrinkage.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
    if shares.iter().any(|p| !p.is_finite()) {
        return None;
    }
    // `W = B^{1/2} Q (I − Λ)^{-1/2}` over the identified directions. `A`'s
    // spectrum lives in `[0, 1]`, so the floor on `1 − p` is absolute.
    let floor = SMOOTH_LR_IDENTIFIED_SHARE_FLOOR_FACTOR * (dimension as f64) * f64::EPSILON;
    let kept: Vec<usize> = (0..shares.len())
        .filter(|&index| 1.0 - shares[index] > floor)
        .collect();
    let mut whitener = Array2::<f64>::zeros((dimension, kept.len()));
    for (column, &index) in kept.iter().enumerate() {
        let scale = (1.0 - shares[index]).sqrt();
        for row in 0..dimension {
            whitener[[row, column]] = shrinkage_vectors[[row, index]] / scale;
        }
    }
    let dual = b_eigenvalues
        .iter()
        .all(|&eigenvalue| eigenvalue > 0.0)
        .then(|| {
            let mut inverse_root = b_vectors.clone();
            for (mut column, &eigenvalue) in
                inverse_root.columns_mut().into_iter().zip(b_eigenvalues.iter())
            {
                let root = eigenvalue.sqrt();
                column.mapv_inplace(|value| value / root);
            }
            inverse_root.dot(&b_vectors.t()).dot(&whitener)
        })
        .filter(|dual| dual.iter().all(|value| value.is_finite()));
    let whitener = b_root.dot(&whitener);
    if whitener.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let mut shares = shares;
    shares.sort_by(|a, b| a.partial_cmp(b).expect("finite shrinkage"));
    Some(LrTestedBlock {
        shares,
        whitener,
        dual,
    })
}

/// The published order of a generalized spectrum: ascending, as
/// [`SmoothLrSelectionReplay::generalized`] documents.
fn ascending(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite generalized spectrum"));
    values
}

/// Why a term's reference carries no selection replay, when it carries none.
///
/// A missing replay is not neutral — it is the difference between pricing `λ̂`
/// as CHOSEN and pricing it as given, which this issue measured at
/// `size@.05 = 0.0962` against nominal `0.05`. So the reference says which step
/// declined and why, rather than publishing a `None` a reader has to attribute
/// by elimination. Every one of these is a statement about the FIT, not about
/// the arithmetic: a term with nothing to select legitimately has no replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrSelectionDecline {
    /// No penalty component reached the driver for this term: it is unpenalized,
    /// or every component's `λ̂` was zero or non-finite, or the components sit
    /// outside the tested coefficient block. Nothing was selected, so the
    /// conditional law IS the selection law.
    NoPenaltyComponents,
    /// The Schur-complemented information `Ĩ_jj` has no identified direction:
    /// every direction of the tested block has a data share `1 − p` at the
    /// decomposition's own noise floor, so there is no basis in which the block
    /// is standard normal. A term the penalty has absorbed entirely.
    NoInformation,
    /// The whitening or a component's root refused: a decomposition failed or
    /// produced a non-finite or inconsistent factor. A REFUSAL — see
    /// [`Self::is_refusal`].
    GeometryRefused,
    /// Every scale's window is closed — the fit is railed against both walls of
    /// the solver's `ρ` box at once, so there was no `λ` it could have chosen
    /// instead.
    WindowClosed,
    /// The fitted point `ln t = 0` could not be evaluated, so there is no
    /// conditional arm to pair the selection with. Refused whole rather than
    /// sampled partial. A REFUSAL — see [`Self::is_refusal`].
    GridRefused,
    /// A draw's certified global minimization of its criterion could not resolve
    /// the criterion's stationary structure, the criterion was not evaluable
    /// inside the window, or an axis slice of it could not be priced, so that
    /// draw has no selection. Refused whole rather than sampled partial. A
    /// REFUSAL — see [`Self::is_refusal`].
    SelectionUnresolved,
    /// The observation's own block score was supplied but is not a finite
    /// vector on the tested block, so the observed statistic cannot be scored
    /// by the replay's selection rule — and a replay whose observation is
    /// selected by a different rule than its draws is not a reference for it.
    ObservedScoreUnusable,
}

impl SmoothLrSelectionDecline {
    /// The serialized label surfaced in reports and failure messages.
    pub fn label(self) -> &'static str {
        match self {
            SmoothLrSelectionDecline::NoPenaltyComponents => "no_penalty_components",
            SmoothLrSelectionDecline::NoInformation => "no_information",
            SmoothLrSelectionDecline::GeometryRefused => "geometry_refused",
            SmoothLrSelectionDecline::WindowClosed => "window_closed",
            SmoothLrSelectionDecline::GridRefused => "grid_refused",
            SmoothLrSelectionDecline::SelectionUnresolved => "selection_unresolved",
            SmoothLrSelectionDecline::ObservedScoreUnusable => "observed_score_unusable",
        }
    }

    /// Whether this decline is a failure of the replay rather than a fact about
    /// the fit.
    ///
    /// `NoPenaltyComponents`, `NoInformation` and `WindowClosed` say there was
    /// no `λ` the fit could have chosen differently, so the conditional law IS
    /// the selection law and the reference is complete without a replay. The
    /// other four say a `λ̂` WAS chosen and the replay that prices the choice
    /// could not be computed. Reading the conditional tail there is the
    /// fixed-`λ` p-value this driver exists not to publish, and it is not a
    /// harmless approximation: on a null term the penalty has absorbed, it
    /// produced `p = 0.0005`. So a refused replay publishes no p-value at all,
    /// and the label says why.
    pub fn is_refusal(self) -> bool {
        match self {
            SmoothLrSelectionDecline::NoPenaltyComponents
            | SmoothLrSelectionDecline::NoInformation
            | SmoothLrSelectionDecline::WindowClosed => false,
            SmoothLrSelectionDecline::GeometryRefused
            | SmoothLrSelectionDecline::GridRefused
            | SmoothLrSelectionDecline::SelectionUnresolved
            | SmoothLrSelectionDecline::ObservedScoreUnusable => true,
        }
    }
}

/// One draw's criterion along ONE log-scale coordinate `u`, diagonal in a basis
/// that does not move with `u`:
///
/// ```text
/// C(u) = Σ_j c_j² s_j(u) + Σ_j ln(1 + e^u ν_j) − r·u − Σ_k ln(1 + e^u μ_k) − const,
/// s_j(u) = e^u ν_j / (1 + e^u ν_j),
/// ```
///
/// Both replay lanes select through it. On the COMMON-SCALE lane every scale
/// moves together, `T(t) = t·T(1)`, so `ν_j = eig T(1)`, `c_j` are the draw's
/// coordinates in that eigenbasis, `r` is the structural rank, `const` is
/// `Σ_{j<r} ln ν_j` and there is no `μ` (the log-determinant runs over EVERY
/// direction: an unpenalized one has `ν = 0` and carries `ln 1 = 0`). A
/// criterion whose pseudo-determinant is itself a moving spectrum carries it as
/// `μ`, subtracted share by share. With `g = s(1 − s)` for either
/// spectrum every derivative is closed form,
///
/// ```text
/// C′ = Σ_j c_j² g_j + Σ_j s_j − r − Σ_k s_k(μ),
/// C″ = Σ_j c_j² g_j(1 − 2s_j) + Σ_j g_j − Σ_k g_k(μ),
/// C‴ = Σ_j c_j² g_j(1 − 6s_j + 6s_j²) + Σ_j g_j(1 − 2s_j) − Σ_k g_k(μ)(1 − 2s_k(μ)),
/// ```
///
/// and every share increases with `u`, so a cell's derivative and curvature ranges
/// follow from its two endpoint shares and the stationary points of the share
/// polynomials (`g` peaks at `s = ½`, `g(1 − 2s)` is extremal at
/// `s = (3 ∓ √3)/6`). That is what lets the replay take each selection as the
/// certified global minimum of its criterion over the window, through
/// [`gam_math::score_opt::maximize_score_1d`].
struct DiagonalCriterion<'a> {
    squares: &'a [f64],
    generalized: &'a [f64],
    rank: usize,
    occam: &'a [f64],
    constant: f64,
}

/// `(s, 1 − s)` for the share `s = x / (1 + x)` of `x = e^u ν`, each to a few
/// ulps RELATIVE.
///
/// The complement is `1 / (1 + x)`, never `1 − s`: near `s = 1` the subtraction
/// keeps only an absolute `ε`, and every share polynomial carries the factor
/// `1 − s`. The forward-error bands the certified search reads are Higham
/// accumulation bounds over summands taken as relatively accurate, so a summand
/// that is not breaks them. On a Bernoulli null the slice `C″` at the two ends of
/// a `1.4e-8`-wide cell came out `2.4e-16` apart — the `ε` of a share near one —
/// against a certified `|C‴|·width` of `1.1e-17` and a band of `8e-19`, so the
/// second-order enclosure was empty and the draw's selection was refused
/// (`selection_unresolved`, one null draw in ten at `n = 100`).
fn share_pair(scaled: f64) -> (f64, f64) {
    let denominator = 1.0 + scaled;
    (scaled / denominator, 1.0 / denominator)
}

/// Range of `f(s, 1 − s)` over the share interval from `lo` to `hi` (each a
/// [`share_pair`]), from its endpoints and the listed stationary shares that
/// fall inside it.
fn share_polynomial_range(
    lo: (f64, f64),
    hi: (f64, f64),
    f: impl Fn(f64, f64) -> f64,
    stationary: &[f64],
) -> (f64, f64) {
    let (at_lo, at_hi) = (f(lo.0, lo.1), f(hi.0, hi.1));
    let mut range = (at_lo.min(at_hi), at_lo.max(at_hi));
    for &point in stationary {
        if lo.0 <= point && point <= hi.0 {
            let value = f(point, 1.0 - point);
            range = (range.0.min(value), range.1.max(value));
        }
    }
    range
}

/// The share polynomials of [`DiagonalCriterion`] in `(s, c = 1 − s)`:
/// `g = s(1 − s) = sc`, `g(1 − 2s) = sc(c − s)` and
/// `g(1 − 6s + 6s²) = sc(1 − 6sc)`. Written in the pair they are relatively
/// accurate wherever `sc` is, and each is bounded in magnitude by its
/// absolute-value evaluation — `sc` for the first two (`|c − s| ≤ c + s = 1`),
/// `sc(1 + 6sc)` for the third — which is what the bands are charged with.
fn share_spread(s: f64, c: f64) -> f64 {
    s * c
}

fn share_skew(s: f64, c: f64) -> f64 {
    s * c * (c - s)
}

fn share_torsion(s: f64, c: f64) -> f64 {
    s * c * (1.0 - 6.0 * s * c)
}

impl DiagonalCriterion<'_> {
    /// `[C, C′, C″, C‴]` at `u`, and the forward-error band of `C` (Higham's
    /// accumulation bound over its summands). `None` when `e^u ν_j` overflows.
    fn jet(&self, log_t: f64) -> Option<([f64; 4], f64)> {
        let t = log_t.exp();
        if !t.is_finite() {
            return None;
        }
        let rank = self.rank as f64;
        let mut value = -rank * log_t - self.constant;
        let mut magnitude = (rank * log_t).abs() + self.constant.abs();
        let (mut first, mut second, mut third) = (-rank, 0.0_f64, 0.0_f64);
        for (&square, &nu) in self.squares.iter().zip(self.generalized.iter()) {
            let scaled = t * nu;
            if !scaled.is_finite() {
                return None;
            }
            let (share, complement) = share_pair(scaled);
            let spread = share_spread(share, complement);
            let skew = share_skew(share, complement);
            let log_term = scaled.ln_1p();
            value += square * share + log_term;
            magnitude += (square * share).abs() + log_term.abs();
            first += square * spread + share;
            second += square * skew + spread;
            third += square * share_torsion(share, complement) + skew;
        }
        for &mu in self.occam {
            let scaled = t * mu;
            if !scaled.is_finite() {
                return None;
            }
            let (share, complement) = share_pair(scaled);
            let log_term = scaled.ln_1p();
            value -= log_term;
            magnitude += log_term.abs();
            first -= share;
            second -= share_spread(share, complement);
            third -= share_skew(share, complement);
        }
        let band = gam_linalg::roundoff::accumulation_band(
            4 * self.squares.len() + 3 * self.occam.len() + 2,
            magnitude,
        );
        Some(([value, first, second, third], band))
    }

    /// Outer ranges of `C′` and `C″` over `[a, b]`, each widened by its own
    /// forward-error band so rounding of the endpoint shares cannot shrink them.
    ///
    /// The cell range from [`Self::natural_ranges`] is intersected with `C′` in
    /// centred form from each endpoint, `C′(a) + (u − a)·C″([a, b])` and
    /// `C′(b) − (b − u)·C″([a, b])`. Both hold for every `u` in the cell by the
    /// fundamental theorem of calculus, so the intersection still contains `C′`.
    /// Along a tail plateau `C′` is small but has one sign, and the centred form
    /// lets such a cell retire by that sign instead of by subdivision.
    ///
    /// Where the data term and the log-determinant pairs cancel, the cell range of
    /// `C″` is itself crosswise and the first-order form is as wide as it. So `C′`
    /// is also intersected with the second-order forms
    /// `C′(a) + (u − a)·C″(a) + ½(u − a)²·C‴([a, b])` and the same from `b`, and
    /// `C″` with `C″(a) + (u − a)·C‴([a, b])` and the same from `b` (Taylor with
    /// the Lagrange remainder). On draw 357 of the split-penalty replay `C′` is
    /// −1.914e-10 across the cell `[−0.7324, −0.7310]`, the first-order form left
    /// its upper end at 2.1e-10, and the search ran out of subdivisions on that
    /// cell. The second-order form excludes zero there (#2902 row 4).
    fn derivative_ranges(
        &self,
        a: f64,
        b: f64,
    ) -> Option<(gam_math::score_opt::ClosedInterval, gam_math::score_opt::ClosedInterval)> {
        let (first, second, third) = self.natural_ranges(a, b)?;
        if !(b > a) {
            return Some((first, second));
        }
        let (at_a, curvature_a, _) = self.natural_ranges(a, a)?;
        let (at_b, curvature_b, _) = self.natural_ranges(b, b)?;
        let width = b - a;
        // Over the cell, `u − a` and `b − u` both run over `[0, width]`, so a term
        // `(u − a)·I` spans `width·[min(I.lo, 0), max(I.hi, 0)]`.
        let span = |lo: f64, hi: f64, scale: f64| (scale * lo.min(0.0), scale * hi.max(0.0));
        let (rise_lo, rise_hi) = span(second.lo, second.hi, width);
        let (tilt_a_lo, tilt_a_hi) = span(curvature_a.lo, curvature_a.hi, width);
        let (tilt_b_lo, tilt_b_hi) = span(-curvature_b.hi, -curvature_b.lo, width);
        let (bend_lo, bend_hi) = span(third.lo, third.hi, 0.5 * width * width);
        let (turn_lo, turn_hi) = span(third.lo, third.hi, width);
        let magnitude = [
            at_a.lo, at_a.hi, at_b.lo, at_b.hi, curvature_a.lo, curvature_a.hi, curvature_b.lo,
            curvature_b.hi,
        ]
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
            + [rise_lo, rise_hi, tilt_a_lo, tilt_a_hi, tilt_b_lo, tilt_b_hi, turn_lo, turn_hi]
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
            + bend_lo.abs().max(bend_hi.abs());
        let band = gam_linalg::roundoff::accumulation_band(8, magnitude);
        let lo = first
            .lo
            .max(at_a.lo + rise_lo - band)
            .max(at_b.lo - rise_hi - band)
            .max(at_a.lo + tilt_a_lo + bend_lo - band)
            .max(at_b.lo + tilt_b_lo + bend_lo - band);
        let hi = first
            .hi
            .min(at_a.hi + rise_hi + band)
            .min(at_b.hi - rise_lo + band)
            .min(at_a.hi + tilt_a_hi + bend_hi + band)
            .min(at_b.hi + tilt_b_hi + bend_hi + band);
        let curvature_lo = second
            .lo
            .max(curvature_a.lo + turn_lo - band)
            .max(curvature_b.lo - turn_hi - band);
        let curvature_hi = second
            .hi
            .min(curvature_a.hi + turn_hi + band)
            .min(curvature_b.hi - turn_lo + band);
        // Every enclosure intersected here contains `C′` or `C″`; an empty
        // intersection breaks that contract and is refused rather than reconciled.
        if !(lo <= hi && curvature_lo <= curvature_hi) {
            return None;
        }
        Some((
            gam_math::score_opt::ClosedInterval::new(lo, hi),
            gam_math::score_opt::ClosedInterval::new(curvature_lo, curvature_hi),
        ))
    }

    /// Outer ranges of `C′` and `C″` over `[a, b]` from the share ranges of each
    /// spectrum.
    ///
    /// The log-determinant spectra enter `C′` as
    /// `Σ_j s(u + ln ν_j) − Σ_k s(u + ln μ_k)`, and `C″` the same way with
    /// `g = s(1 − s)`. Enclosing the two sums separately subtracts their ranges
    /// crosswise. On a slice whose other scales dominate this one, the two
    /// spectra come in near-equal pairs, so the crosswise width is the whole of
    /// both ranges while each pair barely moves. On the coupled dense pair,
    /// `ν = 5.433397857875894e-13` against `μ = 5.433407919278328e-13` left `C′`
    /// enclosed in `±2.6e-4` over cells whose exact slope was about `1e-7`, and
    /// the certified search ran out of subdivisions (#2902). So both spectra are
    /// sorted and paired from the smallest up: a direction only this scale
    /// reaches has no Occam partner and sits at the top of `ν`. Any pairing is an
    /// exact regrouping of the sums. With `d = ln ν − ln μ`,
    /// `s(u + ln ν) − s(u + ln μ) = ∫ g` over a shift of length `|d|`, so it has
    /// the sign of `d` and magnitude at most `|d|·max g`. Likewise
    /// `g(u + ln ν) − g(u + ln μ)` has magnitude at most `|d|·max|g′|`, where
    /// `g′ = s(1 − s)(1 − 2s)`. Each maximum is taken over the shares the pair
    /// spans on the cell. Each pair keeps the intersection of that bound with its
    /// crosswise range. `C‴` pairs the same way, with `g(1 − 2s)` differences
    /// bounded by `|d|·max|g(1 − 6s + 6s²)|`.
    fn natural_ranges(
        &self,
        a: f64,
        b: f64,
    ) -> Option<(
        gam_math::score_opt::ClosedInterval,
        gam_math::score_opt::ClosedInterval,
        gam_math::score_opt::ClosedInterval,
    )> {
        let (t_a, t_b) = (a.exp(), b.exp());
        if !(t_a.is_finite() && t_b.is_finite()) {
            return None;
        }
        let root_three = 3.0_f64.sqrt();
        let root_six = 6.0_f64.sqrt();
        let spread_stationary = [0.5_f64];
        let skew_stationary = [(3.0 - root_three) / 6.0, (3.0 + root_three) / 6.0];
        let torsion_stationary = [0.5_f64, (3.0 - root_six) / 6.0, (3.0 + root_six) / 6.0];
        let rank = self.rank as f64;
        let (mut first_lo, mut first_hi) = (-rank, -rank);
        let (mut second_lo, mut second_hi) = (0.0_f64, 0.0_f64);
        let (mut third_lo, mut third_hi) = (0.0_f64, 0.0_f64);
        let (mut first_magnitude, mut second_magnitude, mut third_magnitude) =
            (rank, 0.0_f64, 0.0_f64);
        let (spread, skew, torsion) = (share_spread, share_skew, share_torsion);
        // The absolute-value evaluations of the three share polynomials over a
        // share range whose largest `sc` is `spread_max` (see [`share_spread`]).
        let modulus = |spread_max: f64| (spread_max, spread_max * (1.0 + 6.0 * spread_max));
        let mut nu_order: Vec<usize> = (0..self.generalized.len())
            .filter(|&index| self.generalized[index] > 0.0)
            .collect();
        nu_order.sort_by(|&left, &right| {
            self.generalized[left].total_cmp(&self.generalized[right])
        });
        let mut mu_order: Vec<usize> = (0..self.occam.len())
            .filter(|&index| self.occam[index] > 0.0)
            .collect();
        mu_order.sort_by(|&left, &right| self.occam[left].total_cmp(&self.occam[right]));
        let pairs = nu_order.len().min(mu_order.len());
        let mut nu_paired = vec![false; self.generalized.len()];
        let mut mu_paired = vec![false; self.occam.len()];
        for index in 0..pairs {
            nu_paired[nu_order[index]] = true;
            mu_paired[mu_order[index]] = true;
        }
        for (index, (&square, &nu)) in self.squares.iter().zip(self.generalized.iter()).enumerate() {
            let (scaled_a, scaled_b) = (t_a * nu, t_b * nu);
            if !(scaled_a.is_finite() && scaled_b.is_finite()) {
                return None;
            }
            let (pair_a, pair_b) = (share_pair(scaled_a), share_pair(scaled_b));
            let (share_lo, share_hi) = (pair_a.0, pair_b.0);
            let (spread_lo, spread_hi) =
                share_polynomial_range(pair_a, pair_b, spread, &spread_stationary);
            let (skew_lo, skew_hi) =
                share_polynomial_range(pair_a, pair_b, skew, &skew_stationary);
            let (torsion_lo, torsion_hi) =
                share_polynomial_range(pair_a, pair_b, torsion, &torsion_stationary);
            let (skew_modulus, torsion_modulus) = modulus(spread_hi);
            first_lo += square * spread_lo;
            first_hi += square * spread_hi;
            second_lo += square * skew_lo;
            second_hi += square * skew_hi;
            third_lo += square * torsion_lo;
            third_hi += square * torsion_hi;
            if !nu_paired[index] {
                first_lo += share_lo;
                first_hi += share_hi;
                second_lo += spread_lo;
                second_hi += spread_hi;
                third_lo += skew_lo;
                third_hi += skew_hi;
            }
            first_magnitude += square * spread_hi + share_hi;
            second_magnitude += square * skew_modulus + spread_hi;
            third_magnitude += square * torsion_modulus + skew_modulus;
        }
        // The Occam spectrum enters with a MINUS sign, so an unpaired share range
        // subtracts crosswise: its largest share lowers the derivative's floor.
        for (index, &mu) in self.occam.iter().enumerate() {
            let (scaled_a, scaled_b) = (t_a * mu, t_b * mu);
            if !(scaled_a.is_finite() && scaled_b.is_finite()) {
                return None;
            }
            let (pair_a, pair_b) = (share_pair(scaled_a), share_pair(scaled_b));
            let (share_lo, share_hi) = (pair_a.0, pair_b.0);
            let (spread_lo, spread_hi) =
                share_polynomial_range(pair_a, pair_b, spread, &spread_stationary);
            let (skew_lo, skew_hi) =
                share_polynomial_range(pair_a, pair_b, skew, &skew_stationary);
            if !mu_paired[index] {
                first_lo -= share_hi;
                first_hi -= share_lo;
                second_lo -= spread_hi;
                second_hi -= spread_lo;
                third_lo -= skew_hi;
                third_hi -= skew_lo;
            }
            first_magnitude += share_hi;
            second_magnitude += spread_hi;
            third_magnitude += modulus(spread_hi).0;
        }
        for index in 0..pairs {
            let (nu, mu) = (self.generalized[nu_order[index]], self.occam[mu_order[index]]);
            let (nu_a, nu_b) = (share_pair(t_a * nu), share_pair(t_b * nu));
            let (mu_a, mu_b) = (share_pair(t_a * mu), share_pair(t_b * mu));
            let (nu_lo, nu_hi) = (nu_a.0, nu_b.0);
            let (mu_lo, mu_hi) = (mu_a.0, mu_b.0);
            let (nu_spread_lo, nu_spread_hi) =
                share_polynomial_range(nu_a, nu_b, spread, &spread_stationary);
            let (mu_spread_lo, mu_spread_hi) =
                share_polynomial_range(mu_a, mu_b, spread, &spread_stationary);
            let (nu_skew_lo, nu_skew_hi) = share_polynomial_range(nu_a, nu_b, skew, &skew_stationary);
            let (mu_skew_lo, mu_skew_hi) = share_polynomial_range(mu_a, mu_b, skew, &skew_stationary);
            let gap = nu.ln() - mu.ln();
            let (span_lo, span_hi) = (share_pair(t_a * nu.min(mu)), share_pair(t_b * nu.max(mu)));
            let (_, spread_max) =
                share_polynomial_range(span_lo, span_hi, spread, &spread_stationary);
            let (skew_min, skew_max) =
                share_polynomial_range(span_lo, span_hi, skew, &skew_stationary);
            let (torsion_min, torsion_max) =
                share_polynomial_range(span_lo, span_hi, torsion, &torsion_stationary);
            let slope_bound = gap.abs() * spread_max;
            let curvature_bound = gap.abs() * skew_min.abs().max(skew_max.abs());
            let torsion_bound = gap.abs() * torsion_min.abs().max(torsion_max.abs());
            if !(slope_bound.is_finite() && curvature_bound.is_finite() && torsion_bound.is_finite()) {
                return None;
            }
            let (signed_lo, signed_hi) = if gap >= 0.0 {
                (0.0, slope_bound)
            } else {
                (-slope_bound, 0.0)
            };
            // Rounding can leave the two bounds a hair apart at a point cell; the
            // band added below covers that hair, so the pair keeps its hull.
            let (pair_lo, pair_hi) = ((nu_lo - mu_hi).max(signed_lo), (nu_hi - mu_lo).min(signed_hi));
            first_lo += pair_lo.min(pair_hi);
            first_hi += pair_lo.max(pair_hi);
            let (bent_lo, bent_hi) = (
                (nu_spread_lo - mu_spread_hi).max(-curvature_bound),
                (nu_spread_hi - mu_spread_lo).min(curvature_bound),
            );
            second_lo += bent_lo.min(bent_hi);
            second_hi += bent_lo.max(bent_hi);
            let (twist_lo, twist_hi) = (
                (nu_skew_lo - mu_skew_hi).max(-torsion_bound),
                (nu_skew_hi - mu_skew_lo).min(torsion_bound),
            );
            third_lo += twist_lo.min(twist_hi);
            third_hi += twist_lo.max(twist_hi);
        }
        let terms = 5 * self.squares.len() + 4 * self.occam.len() + 6 * pairs + 1;
        let first_band = gam_linalg::roundoff::accumulation_band(terms, first_magnitude);
        let second_band = gam_linalg::roundoff::accumulation_band(terms, second_magnitude);
        let third_band = gam_linalg::roundoff::accumulation_band(terms, third_magnitude);
        Some((
            gam_math::score_opt::ClosedInterval::new(first_lo - first_band, first_hi + first_band),
            gam_math::score_opt::ClosedInterval::new(
                second_lo - second_band,
                second_hi + second_band,
            ),
            gam_math::score_opt::ClosedInterval::new(third_lo - third_band, third_hi + third_band),
        ))
    }

    /// The statistic `W(u) = Σ_j c_j² w_j`, `w_j = 2f̄_j − f̄_j²` with `f̄ = 1 − s`.
    fn statistic(&self, log_t: f64) -> f64 {
        let t = log_t.exp();
        self.squares
            .iter()
            .zip(self.generalized.iter())
            .map(|(&square, &nu)| {
                let scaled = t * nu;
                let share = if scaled.is_finite() {
                    scaled / (1.0 + scaled)
                } else {
                    1.0
                };
                let shrinkage = 1.0 - share;
                square * (2.0 * shrinkage - shrinkage * shrinkage)
            })
            .sum()
    }

    /// The Wald statistic `uᵀ(I + T(t))⁻¹u = Σ_j c_j² /(1 + tν_j)` at `u`.
    ///
    /// In the whitened basis the posterior mean at `t` is `(I + T)⁻¹u` and its
    /// covariance is `(I + T)⁻¹`, so the quadratic form of one against the
    /// other is this sum.
    fn wald_statistic(&self, log_t: f64) -> f64 {
        let t = log_t.exp();
        self.squares
            .iter()
            .zip(self.generalized.iter())
            .map(|(&square, &nu)| {
                let scaled = t * nu;
                if scaled.is_finite() {
                    square * share_pair(scaled).1
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// The certified global minimizer of `C` over `[low, high]`: every cell of the
    /// window is derivative-excluded, stationary-isolated to `√ε` in `ln t`, or
    /// proved dominated, by the workspace's certified 1-D search on `−C`.
    fn select(&self, low: f64, high: f64) -> Result<f64, String> {
        use gam_math::score_opt::{
            ClosedInterval, DerivativeEnclosure, ScoreJet, ScoreSample, ScoreValueEnclosure,
        };
        let search = gam_math::score_opt::maximize_score_1d(
            low,
            high,
            f64::EPSILON.sqrt(),
            |log_t| {
                self.jet(log_t)
                    .map(|([value, first, second, third], _)| ScoreJet {
                        value: -value,
                        derivative: -first,
                        curvature: -second,
                        third: -third,
                    })
                    .ok_or_else(|| format!("criterion not evaluable at ln t = {log_t}"))
            },
            |left: ScoreSample, right: ScoreSample| {
                let (first, second) = self
                    .derivative_ranges(left.x, right.x)
                    .ok_or_else(|| format!("criterion not evaluable on [{}, {}]", left.x, right.x))?;
                let band_left = self.jet(left.x).map(|(_, band)| band);
                let band_right = self.jet(right.x).map(|(_, band)| band);
                let (Some(band_left), Some(band_right)) = (band_left, band_right) else {
                    return Err(format!("criterion not evaluable on [{}, {}]", left.x, right.x));
                };
                let evaluation_error = band_left.max(band_right);
                // Mean-value bound from either endpoint: the exact score over the
                // cell stays within `max|C′|·width` of both endpoint values.
                let slope = first.lo.abs().max(first.hi.abs());
                let reach = slope * (right.x - left.x) + evaluation_error;
                Ok(DerivativeEnclosure {
                    score: ScoreValueEnclosure {
                        value: ClosedInterval::new(
                            left.value.min(right.value) - reach,
                            left.value.max(right.value) + reach,
                        ),
                        evaluation_error,
                    },
                    derivative: ClosedInterval::new(-first.hi, -first.lo),
                    curvature: ClosedInterval::new(-second.hi, -second.lo),
                })
            },
        )
        .map_err(|error| format!("{error:?}"))?;
        Ok(search.optimum.x)
    }
}

/// The λ̂-selection replay, or the named reason there is none.
///
/// This is an enum rather than an `Option` so that a consumer cannot read
/// "no replay" without reading why — the two branches are different statements
/// about the fit and the driver has always known which one it made.
#[derive(Clone, Debug, PartialEq)]
pub enum SmoothLrSelection {
    /// `λ̂` was replayed over the box the solver left it.
    Replayed(SmoothLrSelectionReplay),
    /// It was not, for this reason.
    Declined(SmoothLrSelectionDecline),
}

impl SmoothLrSelection {
    /// The replay, when there is one.
    pub fn replay(&self) -> Option<&SmoothLrSelectionReplay> {
        match self {
            SmoothLrSelection::Replayed(replay) => Some(replay),
            SmoothLrSelection::Declined(_) => None,
        }
    }

    /// The decline reason, when there is no replay.
    pub fn decline(&self) -> Option<SmoothLrSelectionDecline> {
        match self {
            SmoothLrSelection::Replayed(_) => None,
            SmoothLrSelection::Declined(reason) => Some(*reason),
        }
    }
}

impl SmoothLrSelectionReplay {
    /// Generate the replay for one term from its whitened penalty geometry and
    /// the window of `ln t` the fit's own `ρ` box leaves open around the fitted
    /// point — ONE window per scale, because the outer search moved each `ρ_i`
    /// independently inside that box.
    ///
    /// Declines — with a reason — when the term has no penalized direction
    /// (nothing to select) or every window is empty (the fit is railed against
    /// both walls), in which case the conditional law IS the selection law and
    /// the caller should use it unmodified; and when the geometry, a draw's
    /// selection or the observation's own score could not be used, which is a
    /// REFUSAL ([`SmoothLrSelectionDecline::is_refusal`]) and leaves the term
    /// with no p-value.
    ///
    /// `observed_score` is the tested block's score at the nested null fit, in
    /// the units of the unscaled information the whitener was built from (see
    /// [`ObservedSelection`]); its whitened image `Wᵀg` is the observation's
    /// own draw.
    pub fn generate(
        whitener: &Array2<f64>,
        unit_penalties: &[Array2<f64>],
        log_lambda: &[f64],
        log_scale_windows: &[(f64, f64)],
        observed_score: Option<&Array1<f64>>,
    ) -> SmoothLrSelection {
        if unit_penalties.is_empty() || unit_penalties.len() != log_lambda.len() {
            return SmoothLrSelection::Declined(
                SmoothLrSelectionDecline::NoPenaltyComponents,
            );
        }
        if whitener.ncols() == 0 {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::NoInformation);
        }
        let geometry = match SelectionGeometry::whiten(whitener, unit_penalties, log_lambda) {
            Ok(geometry) => geometry,
            Err(reason) => return SmoothLrSelection::Declined(reason),
        };
        let observed = match observed_score {
            None => None,
            Some(score) if score.len() == whitener.nrows() => {
                let whitened = whitener.t().dot(score);
                if whitened.iter().any(|value| !value.is_finite()) {
                    return SmoothLrSelection::Declined(
                        SmoothLrSelectionDecline::ObservedScoreUnusable,
                    );
                }
                Some(whitened.to_vec())
            }
            Some(_) => {
                return SmoothLrSelection::Declined(
                    SmoothLrSelectionDecline::ObservedScoreUnusable,
                );
            }
        };
        Self::from_geometry(
            &geometry,
            log_scale_windows,
            SMOOTH_LR_SELECTION_DRAWS,
            SMOOTH_LR_MULTISCALE_DRAWS,
            observed.as_deref(),
        )
    }

    /// Dispatch: a term selecting `m ≥ 2` scales gets the `m`-dimensional
    /// replay; a single scale gets the common-scale lane.
    ///
    /// `observed` is the observation's whitened draw, scored by whichever lane
    /// runs so that it is selected by exactly the rule its reference is.
    fn from_geometry(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        diagonal_draws: usize,
        multiscale_draws: usize,
        observed: Option<&[f64]>,
    ) -> SmoothLrSelection {
        if log_scale_windows.len() != geometry.roots.len() {
            return SmoothLrSelection::Declined(
                SmoothLrSelectionDecline::NoPenaltyComponents,
            );
        }
        if observed.is_some_and(|draw| draw.len() != geometry.dimension) {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::ObservedScoreUnusable);
        }
        if geometry.roots.len() >= 2 {
            return match Self::generate_multiscale(
                geometry,
                log_scale_windows,
                multiscale_draws,
                observed,
            ) {
                Ok(replay) => SmoothLrSelection::Replayed(replay),
                // A closed multi-scale window is not the end of the story: the
                // common-scale slice intersects the same windows and declines
                // for ITSELF if there is genuinely nothing to move. Any other
                // refusal is about the geometry, which the slice shares, so it
                // stands rather than being retried.
                Err(SmoothLrSelectionDecline::WindowClosed) => Self::generate_common_scale(
                    geometry,
                    log_scale_windows,
                    diagonal_draws,
                    observed,
                ),
                Err(reason) => SmoothLrSelection::Declined(reason),
            };
        }
        Self::generate_common_scale(geometry, log_scale_windows, diagonal_draws, observed)
    }

    /// The COMMON-SCALE replay: every scale moved together, `t_i ≡ t`.
    ///
    /// This is the whole selection when the term has one penalty, and the
    /// dispatcher's fallback when a multi-scale term has no open window of its
    /// own to replay.
    ///
    /// Under a common scale `T(t) = t·T(1)`, so the eigenBASIS does not move and
    /// each draw's criterion is diagonal in one decomposition, with closed-form
    /// derivatives of every order ([`DiagonalCriterion`]). The
    /// log-determinant is exact in closed form for the same reason —
    /// `log|T(t)|₊ = rank·ln t + log|T(1)|₊` — so the only quantity that has to
    /// be priced carefully is the `t`-free constant, and it is, through the
    /// stacked roots at the fitted point.
    ///
    /// Each draw's selection is the certified GLOBAL minimum of its criterion
    /// over the window (SPEC rule 18: never the best node of a grid). A draw whose
    /// stationary structure the certified search cannot resolve declines the
    /// whole replay rather than being sampled partially.
    fn generate_common_scale(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        draws: usize,
        observed: Option<&[f64]>,
    ) -> SmoothLrSelection {
        // Moving every scale together, the reachable set is the INTERSECTION of
        // the per-scale windows: a common shift has to keep every `ρ̂_i + ln t`
        // inside the solver's box at once.
        let (mut low, mut high) = (f64::NEG_INFINITY, f64::INFINITY);
        for &(window_low, window_high) in log_scale_windows {
            if !(window_low.is_finite() && window_high.is_finite()) {
                return SmoothLrSelection::Declined(SmoothLrSelectionDecline::WindowClosed);
            }
            low = low.max(window_low);
            high = high.min(window_high);
        }
        if !(low.is_finite() && high.is_finite()) || high <= low {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::WindowClosed);
        }
        let zero = vec![0.0_f64; geometry.roots.len()];
        let Some(fitted) = geometry.at(&zero) else {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::GridRefused);
        };
        // `ν_j = eig T(1)`, descending, with the structural rank leading. Only
        // the leading `rank` of them are in `range(T)`; the rest are the term's
        // unpenalized directions and carry no log-determinant term at any `t`.
        let generalized = fitted.eigenvalues.clone();
        if !generalized.iter().take(geometry.rank).any(|&nu| nu > 0.0) {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::GeometryRefused);
        }
        let constant: f64 = generalized
            .iter()
            .take(geometry.rank)
            .map(|nu| nu.ln())
            .sum();

        // `(W(t*), W(1))` for one draw's squared coordinates in the fitted
        // eigenbasis. The observation goes through this same function.
        let score = |squares: &[f64]| {
            let criterion = DiagonalCriterion {
                squares,
                generalized: &generalized,
                rank: geometry.rank,
                occam: &[],
                constant,
            };
            let selected = criterion.select(low, high).ok()?;
            // The control variate's conditional arm is read AT the fitted scale,
            // `ln t = 0`, on the same draw.
            Some((criterion.statistic(selected), criterion.statistic(0.0)))
        };

        let dimension = geometry.dimension;
        let mut squares = vec![0.0_f64; dimension];
        let mut stream = SelectionDrawStream::new(dimension, draws);
        let mut selection_sample = vec![0.0_f64; draws];
        let mut conditional_sample = vec![0.0_f64; draws];
        for draw in 0..draws {
            stream.fill_chi_square_ones(&mut squares);
            let Some((selected, held)) = score(&squares) else {
                return SmoothLrSelection::Declined(SmoothLrSelectionDecline::SelectionUnresolved);
            };
            selection_sample[draw] = selected;
            conditional_sample[draw] = held;
        }
        let observed = match observed {
            None => None,
            Some(draw) => {
                // The draws are squared coordinates in the fitted eigenbasis;
                // the observation is put in the same coordinates.
                for (column, square) in squares.iter_mut().enumerate() {
                    let projection: f64 = (0..dimension)
                        .map(|row| draw[row] * fitted.basis[[row, column]])
                        .sum();
                    *square = projection * projection;
                }
                let Some((selected, conditional)) = score(&squares) else {
                    return SmoothLrSelection::Declined(
                        SmoothLrSelectionDecline::SelectionUnresolved,
                    );
                };
                Some(ObservedSelection {
                    conditional,
                    selected,
                })
            }
        };
        SmoothLrSelection::Replayed(Self {
            generalized: ascending(generalized),
            selection_sample,
            conditional_sample,
            observed,
        })
    }

    /// Replay a term whose `λ̂` is a VECTOR, over each of its scales separately.
    ///
    /// A single-penalty term's selection is one-dimensional and diagonalizes
    /// (that is [`Self::generate_common_scale`]). A term with `m` penalties — a
    /// double-penalty smooth is `m = 2`, a tensor product more — selects `m`
    /// scales, and no single basis diagonalizes `m` penalties against the
    /// information at once. Scaling all of them together is a one-dimensional
    /// SLICE of that selection, and the slice is not enough: on a two-penalty
    /// Gaussian null,
    ///
    /// ```text
    ///                             α = .20    .10     .05     .01
    ///   conditional (no replay)     .4080   .3400   .2800   .1800
    ///   replay, common scale only   .2840   .1440   .0720   .0040
    ///   replay, both scales         .0440   .0120   .0080   .0040
    /// ```
    ///
    /// so the term's own `m` scales are each selected here. The absolute
    /// numbers in that table are from a harness whose own outer optimizer is a
    /// Nelder–Mead on a flat two-dimensional REML surface and are not to be read
    /// as calibration figures; the ORDERING is what it establishes, and the
    /// ordering is that the missing dimensions matter more than anything else
    /// measured on this issue.
    ///
    /// Each axis carries its OWN window. The reachable set for scale `i` is its
    /// #2812 resolvability interval translated to the fitted point,
    /// `ln t_i ∈ [lo_i − ρ̂_i, hi_i − ρ̂_i]`, and those `m` intervals are only equal
    /// when the `m` components' spectra and fitted scales are. Handing the
    /// selection the COMMON-shift intersection — which is what it used to receive —
    /// truncates every axis to the narrowest one and empties the whole replay as
    /// soon as one `λ̂` rails, which for a null-true double-penalty smooth is the
    /// normal state and not a corner case.
    ///
    /// # Each draw's selection is a certified joint Newton minimum
    ///
    /// The criterion is smooth in `ρ = ln t` and its gradient and Hessian are
    /// exact inner products of the two factorizations that already price it
    /// ([`SelectionFactor::jet`]). Each draw's selection is therefore the
    /// box-constrained trust-region Newton descent of [`opt::NewtonTrustRegion`]
    /// over the open scales jointly, stopped by the Newton-decrement certificate
    /// on those derivatives' rounding bands. See [`Self::select_draw`].
    ///
    /// This replaces, first, a `441`-point bracket grid followed by a compass
    /// descent capped at `96` criterion evaluations per draw (#2902: SPEC rules
    /// 18, 19 and 22), and then a coordinatewise sweep of certified 1-D global
    /// searches. The sweep moved one scale at a time along a valley that couples
    /// them, and on a tensor-product `te(x, z)` beside a factor it spent more than
    /// sixty sweeps of `~140 ms` each on a single draw. A draw whose selection
    /// cannot be priced or certified declines the whole replay rather than being
    /// sampled partially.
    fn generate_multiscale(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        draws: usize,
        observed: Option<&[f64]>,
    ) -> Result<Self, SmoothLrSelectionDecline> {
        let scales = geometry.roots.len();
        if scales < 2 {
            return Err(SmoothLrSelectionDecline::GeometryRefused);
        }
        if log_scale_windows.len() != scales {
            return Err(SmoothLrSelectionDecline::NoPenaltyComponents);
        }
        // A scale whose own window is empty — its `λ̂` railed against both walls
        // at once — stays at the fitted point rather than sinking the replay.
        let mut movable = 0usize;
        for &(low, high) in log_scale_windows {
            if !(low.is_finite() && high.is_finite()) {
                return Err(SmoothLrSelectionDecline::WindowClosed);
            }
            if high > low {
                movable += 1;
            }
        }
        if movable == 0 {
            return Err(SmoothLrSelectionDecline::WindowClosed);
        }
        let zero = vec![0.0_f64; scales];
        // The control variate's conditional arm is read AT the fitted `λ̂`,
        // `ln t = 0` on every axis, through the fitted eigensystem.
        let fitted = geometry
            .at(&zero)
            .ok_or(SmoothLrSelectionDecline::GridRefused)?;
        let dimension = geometry.dimension;
        let mut factor = SelectionFactor::new(geometry);
        let mut stream = SelectionDrawStream::new(dimension, draws);
        let mut draw = vec![0.0_f64; dimension];
        let mut coordinates = vec![0.0_f64; geometry.rank];
        let mut selected = vec![0.0_f64; scales];
        // `(W(t*), W(1))` for one whitened draw. The observation goes through
        // this same function.
        let mut score = |draw: &[f64]| -> Result<(f64, f64), SmoothLrSelectionDecline> {
            let mut norm_squared = 0.0_f64;
            let mut conditional = 0.0_f64;
            for column in 0..dimension {
                norm_squared += draw[column] * draw[column];
                let mut projection = 0.0_f64;
                for row in 0..dimension {
                    projection += draw[row] * fitted.basis[[row, column]];
                }
                conditional += projection * projection * fitted.weights[column];
            }
            for column in 0..geometry.rank {
                let mut projection = 0.0_f64;
                for row in 0..dimension {
                    projection += draw[row] * geometry.range_basis[[row, column]];
                }
                coordinates[column] = projection;
            }
            Self::select_draw(
                geometry,
                log_scale_windows,
                &coordinates,
                &mut factor,
                &mut selected,
            )?;
            Ok((factor.score(&coordinates, norm_squared).1, conditional))
        };
        let mut selection_sample = vec![0.0_f64; draws];
        let mut conditional_sample = vec![0.0_f64; draws];
        for index in 0..draws {
            stream.fill_normals(&mut draw);
            (selection_sample[index], conditional_sample[index]) = score(&draw)?;
        }
        let observed = match observed {
            None => None,
            Some(draw) => {
                let (selected, conditional) = score(draw)?;
                Some(ObservedSelection {
                    conditional,
                    selected,
                })
            }
        };
        Ok(Self {
            generalized: ascending(fitted.eigenvalues),
            selection_sample,
            conditional_sample,
            observed,
        })
    }

    /// One draw's multi-scale selection, written into `selected`, with `factor`
    /// left factored there.
    ///
    /// The draw starts at the fitted point, clamped into each open window, and
    /// [`opt::NewtonTrustRegion`] descends the criterion over the open scales
    /// jointly, inside their windows, on its exact gradient and Hessian. It stops
    /// only when the Newton decrement over the free scales is certified below the
    /// criterion's own rounding band: the point is then a stationary point of the
    /// box-constrained criterion to working precision, a scale at its wall counted
    /// free only while the gradient points back into the window. There is no
    /// iteration budget. The trust region accepts only steps that lower the
    /// criterion, which is bounded below on the box, so it cannot cycle.
    ///
    /// Every other end of the descent — a radius that collapses without a
    /// certificate, a point the factorizations cannot price — is
    /// `SelectionUnresolved`, never the best point reached.
    ///
    /// The selection is a deterministic function of the draw, and the observation
    /// goes through the same function, so the replay's law is the law of the
    /// statistic the row publishes whichever stationary point the descent ends at.
    fn select_draw(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        coordinates: &[f64],
        factor: &mut SelectionFactor,
        selected: &mut [f64],
    ) -> Result<(), SmoothLrSelectionDecline> {
        use SmoothLrSelectionDecline::{NoPenaltyComponents, SelectionUnresolved};
        if selected.len() != log_scale_windows.len() {
            return Err(NoPenaltyComponents);
        }
        let mut open = Vec::with_capacity(selected.len());
        for (axis, (slot, &(low, high))) in selected.iter_mut().zip(log_scale_windows).enumerate() {
            *slot = if high > low {
                open.push(axis);
                0.0_f64.clamp(low, high)
            } else {
                0.0
            };
        }
        if !open.is_empty() {
            let lower = Array1::from_iter(open.iter().map(|&axis| log_scale_windows[axis].0));
            let upper = Array1::from_iter(open.iter().map(|&axis| log_scale_windows[axis].1));
            let start = Array1::from_iter(open.iter().map(|&axis| selected[axis]));
            let bounds = opt::Bounds::new(lower, upper, 0.0).map_err(|_| SelectionUnresolved)?;
            let unbounded = opt::MaxIterations::new(usize::MAX).map_err(|_| SelectionUnresolved)?;
            let criterion = MultiscaleCriterion {
                geometry,
                factor: &mut *factor,
                coordinates,
                open: &open,
                point: selected.to_vec(),
            };
            let solution = opt::NewtonTrustRegion::new(start, criterion)
                .with_bounds(bounds)
                .with_profile(opt::Profile::Deterministic)
                .with_max_iterations(unbounded)
                .run()
                .map_err(|_| SelectionUnresolved)?;
            for (slot, &axis) in open.iter().enumerate() {
                selected[axis] = solution.final_point[slot];
            }
        }
        // The descent's last evaluation may have been a rejected trial.
        if factor.refactor(geometry, selected) {
            Ok(())
        } else {
            Err(SelectionUnresolved)
        }
    }

    /// The observation's statistic under the replay's own selection, given
    /// its statistic `conditional` at the fitted `λ̂`: `conditional` carried by
    /// the quadratic model's ratio `W_q(t*; z)/W_q(1; z)` of the observed
    /// whitened score (see [`Self::observed`]). Unchanged when no observation
    /// was scored, or when its `W_q(1; z)` is not a positive number to divide
    /// by — a score of exactly zero, where every `t` gives the same `W_q = 0`
    /// and the selection moves nothing.
    pub fn observed_selection_threshold(&self, conditional: f64) -> f64 {
        match self.observed {
            Some(observed)
                if observed.conditional.is_finite()
                    && observed.conditional > 0.0
                    && observed.selected.is_finite() =>
            {
                conditional * (observed.selected / observed.conditional)
            }
            _ => conditional,
        }
    }

    /// `(shift, standard_error)` of `P̂(W_sel ≥ x_sel) − P̂(W_cond ≥ x)` on
    /// shared draws, each arm asked about ITS OWN functional of the
    /// observation: `x` is the observed statistic at the fitted `λ̂`, which the
    /// exact conditional tail and its control variate are both read at, and
    /// `x_sel` the same observation under the replay's selection.
    ///
    /// `E[P̂(W_cond ≥ x)]` is the exact conditional tail at `x` for EVERY `x`,
    /// so the sum `p_cond(x) + shift` is an unbiased estimate of
    /// `P(W_sel ≥ x_sel)` whatever the two thresholds are; what their
    /// agreement buys is only the variance, and that is measured, not assumed.
    /// That variance is the paired indicator DIFFERENCE `d_i ∈ {−1, 0, +1}`'s,
    /// zero on every draw whose selection did not carry it across the
    /// thresholds — the control variate, and why the standard error is a
    /// fraction of the naive `√(p(1−p)/N)`.
    pub fn tail_shift_at(&self, conditional_threshold: f64, selection_threshold: f64) -> (f64, f64) {
        let draws = self.selection_sample.len();
        if draws == 0 {
            return (0.0, 0.0);
        }
        let mut sum = 0.0_f64;
        let mut sum_squares = 0.0_f64;
        for (&selected, &held) in self
            .selection_sample
            .iter()
            .zip(self.conditional_sample.iter())
        {
            let difference = f64::from(selected >= selection_threshold)
                - f64::from(held >= conditional_threshold);
            sum += difference;
            sum_squares += difference * difference;
        }
        let count = draws as f64;
        let shift = sum / count;
        // `d_i ∈ {−1, 0, +1}` and is zero on every draw whose selected `t` left
        // it on the same side of `statistic` — which is most of them. That is
        // the control variate, and this is its own sample variance rather than
        // the `√(p(1−p)/N)` of either term alone.
        let variance = (sum_squares / count - shift * shift).max(0.0);
        (shift, (variance / count).sqrt())
    }
}

/// The summary Wald test of one smooth term, read against the null law of its
/// statistic when `λ̂` is SELECTED by the criterion the replay replays.
///
/// # The statistic
///
/// In the whitened basis of [`LrTestedBlock`], the tested block's profiled
/// score is `u = Dᵀβ̂/σ̂ ~ N(0, I)` under the null, the posterior mean at `t` is
/// `(I + T(t))⁻¹u` and its covariance is `(I + T(t))⁻¹`. The Wald form of one
/// against the other is therefore
///
/// ```text
/// W(t) = uᵀ(I + T(t))⁻¹u,          W(0) = β̂ᵀ V_β⁻¹ β̂ / σ̂²  at the fitted λ̂.
/// ```
///
/// # Why the observation goes through the replay's own rule
///
/// A fixed-`λ` reference for `W(0)` is wrong on a selected `λ̂` in the same way
/// the LR's is, and a replay of the selection is not enough on its own either:
/// the observed `t̂` comes out of the solver's search, while each draw's comes
/// out of the replay's certified [`DiagonalCriterion::select`]. On a term the
/// solver shrank against its window, the two land on different points and the
/// p-value is not uniform even when it is sized.
///
/// So the OBSERVED `u` is put through the same lane and the same selection as
/// every draw, and `W` is read at the point that selection returns. The
/// statistic is then one deterministic function `u ↦ W(t̂(u))` applied to the
/// observation and to `N` draws of its null law, and its p-value is uniform to
/// the Monte-Carlo error of the draws, whatever the function is. That law is
/// continuous: the only atom it can have is `W ≡ 0`, a term with no
/// identified direction, which publishes `p = 1`.
///
/// # Estimated scale
///
/// With `σ̂² = σ²χ²_ν/ν`, the observed `u` is `z/s` with `s² = χ²_ν/ν`. Each
/// draw's radius is mapped to that law exactly and comonotonically — see
/// [`studentized_radius_factor`] — so the replay needs no second stream. Taking
/// `σ̂` independent of `u` is exact for an unpenalized Gaussian fit and an
/// approximation for a penalized one, whose residual degrees of freedom are
/// themselves an effective count.
///
/// # Monte-Carlo error
///
/// The p-value is the exact conditional tail of `W(0)` —
/// [`gam_math::probability::signed_weighted_chi_square_sf`] over the weights
/// `1/(1 + e_k)` — plus the paired selection shift of
/// [`SmoothLrSelectionReplay::tail_shift_at`], whose standard error is published.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothWaldSelectionTest {
    /// `W(t̂(u))` at the observation's own selected point.
    pub statistic: f64,
    /// `Σ_k 1/(1 + e_k)` over the block's identified directions at the fitted
    /// `λ̂`: the null mean of `W(0)` at known scale.
    pub ref_df: f64,
    /// `P(W(t̂(u)) ≥ statistic)` under the null.
    pub p_value: f64,
    /// Monte-Carlo standard error of `p_value`; zero where no replay ran.
    pub standard_error: f64,
    /// The replay, or why the conditional law already is the selection law.
    pub selection: SmoothLrSelection,
}

/// Which replay reads a term, from its geometry and windows. It mirrors
/// [`SmoothLrSelectionReplay::from_geometry`]'s dispatch, decided once so the
/// observation and the draws cannot take different lanes.
enum WaldLane {
    /// Every scale moves together over the windows' intersection.
    CommonScale { low: f64, high: f64 },
    /// Each scale moves over its own window.
    Multiscale,
    /// No `λ` the fit could have chosen instead.
    Closed,
}

fn wald_lane(
    geometry: &SelectionGeometry,
    log_scale_windows: &[(f64, f64)],
) -> Result<WaldLane, SmoothLrSelectionDecline> {
    if log_scale_windows.len() != geometry.roots.len() {
        return Err(SmoothLrSelectionDecline::GeometryRefused);
    }
    if !log_scale_windows
        .iter()
        .all(|&(low, high)| low.is_finite() && high.is_finite())
    {
        return Ok(WaldLane::Closed);
    }
    if geometry.roots.len() >= 2 && log_scale_windows.iter().any(|&(low, high)| high > low) {
        return Ok(WaldLane::Multiscale);
    }
    let (mut low, mut high) = (f64::NEG_INFINITY, f64::INFINITY);
    for &(window_low, window_high) in log_scale_windows {
        low = low.max(window_low);
        high = high.min(window_high);
    }
    Ok(if high > low {
        WaldLane::CommonScale { low, high }
    } else {
        WaldLane::Closed
    })
}

/// `f` such that `√f·z` has the law of `z/s`, `s² = χ²_ν/ν`, for a draw `z` of
/// `q` standard normals with `‖z‖² = radius_squared`.
///
/// `z/s` keeps `z`'s direction, which is uniform and independent of both radii,
/// and has squared radius `ν·χ²_q/χ²_ν = ν·B/(1 − B)` with
/// `B ~ Beta(q/2, ν/2)`. Mapping `‖z‖²` to that law at the same quantile level
/// is therefore exact for the joint law. The level is carried by whichever tail
/// is below one half, so neither quantile is asked for a probability that has
/// lost its digits to `1 − p`.
fn studentized_radius_factor(radius_squared: f64, dimension: usize, residual_df: f64) -> Option<f64> {
    use statrs::function::gamma::{gamma_lr, gamma_ur};
    if !(radius_squared > 0.0) {
        return (radius_squared == 0.0).then_some(1.0);
    }
    let (a, b) = (0.5 * dimension as f64, 0.5 * residual_df);
    let upper = gamma_ur(a, 0.5 * radius_squared);
    let ratio = if upper < 0.5 {
        // `1 − B ~ Beta(b, a)` has the same LOWER tail as `B`'s upper one.
        let complement = gam_math::probability::beta_quantile(upper, b, a);
        (1.0 - complement) / complement
    } else {
        let lower = gamma_lr(a, 0.5 * radius_squared);
        let quantile = gam_math::probability::beta_quantile(lower, a, b);
        quantile / (1.0 - quantile)
    };
    let factor = residual_df * ratio / radius_squared;
    (factor.is_finite() && factor >= 0.0).then_some(factor)
}

/// `P(Σ_k w_k χ²_1 > x)` at known scale, or `P(Σ_k w_k χ²_1 / (χ²_ν/ν) > x)` at
/// estimated scale — the second as the signed sum `Σ_k w_k χ²_1 − (x/ν)χ²_ν`
/// read at zero.
fn wald_conditional_tail(
    weights: &[f64],
    statistic: f64,
    residual_df: Option<f64>,
) -> Result<f64, SmoothLrSelectionDecline> {
    use gam_math::probability::{WeightedChiSquareTerm, signed_weighted_chi_square_sf};
    let mut terms: Vec<WeightedChiSquareTerm> = weights
        .iter()
        .map(|&weight| WeightedChiSquareTerm {
            weight,
            degrees_of_freedom: 1.0,
        })
        .collect();
    let tail = match residual_df {
        None => signed_weighted_chi_square_sf(&terms, statistic),
        Some(nu) => {
            terms.push(WeightedChiSquareTerm {
                weight: -statistic / nu,
                degrees_of_freedom: nu,
            });
            signed_weighted_chi_square_sf(&terms, 0.0)
        }
    };
    if tail.probability.is_finite() {
        Ok(tail.probability.clamp(0.0, 1.0))
    } else {
        Err(SmoothLrSelectionDecline::GeometryRefused)
    }
}

/// The summary smooth Wald test ([`SmoothWaldSelectionTest`]).
///
/// `whitener` is [`LrTestedBlock::whitener`], `observed` is `Dᵀβ̂/σ̂` with `D`
/// its [`LrTestedBlock::dual`], and the penalties, fitted scales and windows are
/// what [`SmoothLrSelectionReplay::generate`] takes. `residual_df` is `ν` when
/// the scale was estimated and `None` when it is known.
///
/// Declines that are facts about the fit (`NoInformation`,
/// `NoPenaltyComponents`, `WindowClosed`) return a test on the conditional law;
/// a refusal returns `Err` and the term publishes no p-value.
pub fn smooth_wald_selection_test(
    whitener: &Array2<f64>,
    observed: &[f64],
    unit_penalties: &[Array2<f64>],
    log_lambda: &[f64],
    log_scale_windows: &[(f64, f64)],
    residual_df: Option<f64>,
) -> Result<SmoothWaldSelectionTest, SmoothLrSelectionDecline> {
    wald_selection_test(
        whitener,
        observed,
        unit_penalties,
        log_lambda,
        log_scale_windows,
        residual_df,
        WaldDraws {
            diagonal: SMOOTH_LR_SELECTION_DRAWS,
            multiscale: SMOOTH_LR_MULTISCALE_DRAWS,
        },
    )
}

/// How many null draws each replay lane takes.
#[derive(Clone, Copy)]
struct WaldDraws {
    /// Draws of the common-scale (diagonal) lane.
    diagonal: usize,
    /// Draws of the multiscale lane, each of which refactors the penalty sum.
    multiscale: usize,
}

fn wald_selection_test(
    whitener: &Array2<f64>,
    observed: &[f64],
    unit_penalties: &[Array2<f64>],
    log_lambda: &[f64],
    log_scale_windows: &[(f64, f64)],
    residual_df: Option<f64>,
    draws: WaldDraws,
) -> Result<SmoothWaldSelectionTest, SmoothLrSelectionDecline> {
    use SmoothLrSelectionDecline::{
        GeometryRefused, GridRefused, NoInformation, NoPenaltyComponents, SelectionUnresolved,
        WindowClosed,
    };
    let WaldDraws {
        diagonal: diagonal_draws,
        multiscale: multiscale_draws,
    } = draws;
    let dimension = whitener.ncols();
    if observed.len() != dimension
        || unit_penalties.len() != log_lambda.len()
        || observed.iter().any(|value| !value.is_finite())
        || residual_df.is_some_and(|nu| !(nu.is_finite() && nu > 0.0))
    {
        return Err(GeometryRefused);
    }
    let norm_squared: f64 = observed.iter().map(|value| value * value).sum();
    let declined = |reason, statistic: f64, weights: &[f64]| {
        Ok(SmoothWaldSelectionTest {
            statistic,
            ref_df: weights.iter().sum(),
            p_value: wald_conditional_tail(weights, statistic, residual_df)?,
            standard_error: 0.0,
            selection: SmoothLrSelection::Declined(reason),
        })
    };
    if dimension == 0 {
        return declined(NoInformation, 0.0, &[]);
    }
    let geometry = match SelectionGeometry::whiten(whitener, unit_penalties, log_lambda) {
        Ok(geometry) => geometry,
        Err(NoPenaltyComponents) => {
            return declined(NoPenaltyComponents, norm_squared, &vec![1.0; dimension]);
        }
        Err(NoInformation) => return declined(NoInformation, 0.0, &[]),
        Err(reason) => return Err(reason),
    };
    let zero = vec![0.0_f64; geometry.roots.len()];
    let fitted = geometry.at(&zero).ok_or(GridRefused)?;
    let weights: Vec<f64> = fitted
        .eigenvalues
        .iter()
        .map(|&eigenvalue| share_pair(eigenvalue).1)
        .collect();
    let coordinates_in = |basis: &Array2<f64>, vector: &[f64], out: &mut [f64]| {
        for (column, slot) in out.iter_mut().enumerate() {
            *slot = (0..dimension).map(|row| vector[row] * basis[[row, column]]).sum();
        }
    };

    let (statistic, replay) = match wald_lane(&geometry, log_scale_windows)? {
        WaldLane::Closed => {
            let mut eigen = vec![0.0_f64; dimension];
            coordinates_in(&fitted.basis, observed, &mut eigen);
            let statistic = eigen
                .iter()
                .zip(weights.iter())
                .map(|(coordinate, weight)| coordinate * coordinate * weight)
                .sum();
            return declined(WindowClosed, statistic, &weights);
        }
        WaldLane::CommonScale { low, high } => {
            let generalized = fitted.eigenvalues.clone();
            if !generalized.iter().take(geometry.rank).any(|&nu| nu > 0.0) {
                return Err(GeometryRefused);
            }
            let constant: f64 = generalized.iter().take(geometry.rank).map(|nu| nu.ln()).sum();
            let mut squares = vec![0.0_f64; dimension];
            coordinates_in(&fitted.basis, observed, &mut squares);
            squares.iter_mut().for_each(|value| *value *= *value);
            let observed_criterion = DiagonalCriterion {
                squares: &squares,
                generalized: &generalized,
                rank: geometry.rank,
                occam: &[],
                constant,
            };
            let selected = observed_criterion
                .select(low, high)
                .map_err(|_| SelectionUnresolved)?;
            let statistic = observed_criterion.wald_statistic(selected);

            let mut stream = SelectionDrawStream::new(dimension, diagonal_draws);
            let mut selection_sample = vec![0.0_f64; diagonal_draws];
            let mut conditional_sample = vec![0.0_f64; diagonal_draws];
            for draw in 0..diagonal_draws {
                stream.fill_chi_square_ones(&mut squares);
                if let Some(nu) = residual_df {
                    let factor = studentized_radius_factor(squares.iter().sum(), dimension, nu)
                        .ok_or(GeometryRefused)?;
                    squares.iter_mut().for_each(|value| *value *= factor);
                }
                let draw_criterion = DiagonalCriterion {
                    squares: &squares,
                    generalized: &generalized,
                    rank: geometry.rank,
                    occam: &[],
                    constant,
                };
                let selected = draw_criterion
                    .select(low, high)
                    .map_err(|_| SelectionUnresolved)?;
                selection_sample[draw] = draw_criterion.wald_statistic(selected);
                conditional_sample[draw] = draw_criterion.wald_statistic(0.0);
            }
            (
                statistic,
                SmoothLrSelectionReplay {
                    generalized: ascending(generalized),
                    selection_sample,
                    conditional_sample,
                    observed: None,
                },
            )
        }
        WaldLane::Multiscale => {
            let scales = geometry.roots.len();
            let mut factor = SelectionFactor::new(&geometry);
            let mut coordinates = vec![0.0_f64; geometry.rank];
            let mut selected = vec![0.0_f64; scales];
            let mut select = |vector: &[f64], norm_squared: f64| {
                coordinates_in(&geometry.range_basis, vector, &mut coordinates);
                SmoothLrSelectionReplay::select_draw(
                    &geometry,
                    log_scale_windows,
                    &coordinates,
                    &mut factor,
                    &mut selected,
                )?;
                Ok(factor.wald(&coordinates, norm_squared))
            };
            let statistic = select(observed, norm_squared)?;

            let mut stream = SelectionDrawStream::new(dimension, multiscale_draws);
            let mut draw = vec![0.0_f64; dimension];
            let mut eigen = vec![0.0_f64; dimension];
            let mut selection_sample = vec![0.0_f64; multiscale_draws];
            let mut conditional_sample = vec![0.0_f64; multiscale_draws];
            for index in 0..multiscale_draws {
                stream.fill_normals(&mut draw);
                let mut draw_norm: f64 = draw.iter().map(|value| value * value).sum();
                if let Some(nu) = residual_df {
                    let factor =
                        studentized_radius_factor(draw_norm, dimension, nu).ok_or(GeometryRefused)?;
                    let root = factor.sqrt();
                    draw.iter_mut().for_each(|value| *value *= root);
                    draw_norm *= factor;
                }
                coordinates_in(&fitted.basis, &draw, &mut eigen);
                conditional_sample[index] = eigen
                    .iter()
                    .zip(weights.iter())
                    .map(|(coordinate, weight)| coordinate * coordinate * weight)
                    .sum();
                selection_sample[index] = select(&draw, draw_norm)?;
            }
            (
                statistic,
                SmoothLrSelectionReplay {
                    generalized: ascending(fitted.eigenvalues.clone()),
                    selection_sample,
                    conditional_sample,
                    observed: None,
                },
            )
        }
    };
    let (shift, standard_error) = replay.tail_shift_at(statistic, statistic);
    let conditional = wald_conditional_tail(&weights, statistic, residual_df)?;
    Ok(SmoothWaldSelectionTest {
        statistic,
        ref_df: weights.iter().sum(),
        p_value: (conditional + shift).clamp(0.0, 1.0),
        standard_error,
        selection: SmoothLrSelection::Replayed(replay),
    })
}

/// Deterministic `χ²_1` draws for the selection replay.
///
/// A p-value must not depend on a thread count, a machine or a run (#1017), so
/// the replay cannot take draws from a shared or seeded-at-startup generator. It
/// uses a counter-based stream instead: SplitMix64 on an index, mapped through
/// [`gam_math::probability::standard_normal_quantile`] and squared. Same
/// spectrum, same window, same numbers, everywhere, forever.
///
/// The stream is STRATIFIED per coordinate: draw `i` of coordinate `k` takes its
/// uniform from the `i`-th of `N` equal bins, in an order permuted per
/// coordinate. That is a Latin hypercube, and for a functional that is nearly a
/// sum over coordinates — which `W = Σ_k w_k u_k²` is exactly — it removes the
/// part of the Monte-Carlo error the bins already account for.
struct SelectionDrawStream {
    /// The `N` stratum midpoints, mapped through the normal quantile and
    /// squared. Every coordinate draws from THIS set — only the order differs —
    /// so the quantile is evaluated `N` times per term rather than `N × q`.
    values: Vec<f64>,
    /// The same strata as SIGNED normal quantiles.
    signed: Vec<f64>,
    /// One permutation of `0..N` per coordinate.
    permutations: Vec<Vec<u32>>,
    index: usize,
}

impl SelectionDrawStream {
    fn new(dimension: usize, draws: usize) -> Self {
        let signed: Vec<f64> = (0..draws)
            .map(|bin| {
                // Bin midpoint: never `0` or `1`, so the quantile is finite.
                let uniform = (bin as f64 + 0.5) / draws as f64;
                gam_math::probability::standard_normal_quantile(uniform)
                    .expect("a bin midpoint is strictly inside (0, 1)")
            })
            .collect();
        let values: Vec<f64> = signed.iter().map(|normal| normal * normal).collect();
        let mut permutations = Vec::with_capacity(dimension);
        for coordinate in 0..dimension {
            let mut order: Vec<u32> = (0..draws as u32).collect();
            // Fisher–Yates with a counter-based stream keyed by the coordinate.
            let mut state = 0x9E37_79B9_7F4A_7C15_u64
                ^ (coordinate as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
            for position in (1..order.len()).rev() {
                state = split_mix64(state);
                let pick = (state % (position as u64 + 1)) as usize;
                order.swap(position, pick);
            }
            permutations.push(order);
        }
        Self {
            values,
            signed,
            permutations,
            index: 0,
        }
    }

    /// Fill one draw. The `zip` is the length contract: the stream writes one
    /// value per coordinate it was built for and nothing beyond, so a
    /// mis-sized buffer is a short write rather than an assertion.
    fn fill_chi_square_ones(&mut self, out: &mut [f64]) {
        for (slot, permutation) in out.iter_mut().zip(self.permutations.iter()) {
            *slot = self.values[permutation[self.index] as usize];
        }
        self.index += 1;
    }

    /// The same stratified draw as SIGNED normals, for the multi-scale replay:
    /// there the tested block is not diagonal at every grid point, so the
    /// quadratic form needs the vector and not its coordinatewise squares. The
    /// sign is taken from the stratum's own side of the median, which is what
    /// makes it the normal quantile rather than its absolute value.
    fn fill_normals(&mut self, out: &mut [f64]) {
        for (slot, permutation) in out.iter_mut().zip(self.permutations.iter()) {
            *slot = self.signed[permutation[self.index] as usize];
        }
        self.index += 1;
    }
}

/// SplitMix64, used only to permute the strata. Any full-period mixer would do;
/// what matters is that it is a pure function of an index.
#[inline]
fn split_mix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod selection_replay_tests {
    use super::{
        DiagonalCriterion, SMOOTH_LR_SELECTION_DRAWS, SelectionDrawStream, SelectionFactor,
        SelectionGeometry, SmoothLrSelection, SmoothLrSelectionDecline, SmoothLrSelectionReplay,
        split_mix64, wald_conditional_tail,
    };
    use ndarray::Array2;

    /// A shrunk-smooth generalized spectrum: one direction the data can still
    /// see and a geometric tail the penalty has taken.
    fn spectrum() -> Vec<f64> {
        vec![0.3_f64, 1.0, 4.0, 20.0, 120.0, 900.0]
    }

    /// The diagonal criterion's closed-form jet against central differences of
    /// itself, its cell ranges against interior point derivatives, and the
    /// certified selection a value-local minimum no sampled point undercuts. It
    /// carries an Occam spectrum, as an axis slice does, so shares of both signs
    /// are exercised.
    #[test]
    fn common_scale_criterion_jet_enclosure_and_selection_agree_2902() {
        let generalized = spectrum();
        let squares = [1.7_f64, 0.2, 3.1, 0.05, 2.2, 0.9];
        let occam = [0.08_f64, 35.0];
        let constant: f64 = generalized.iter().map(|nu| nu.ln()).sum();
        let criterion = DiagonalCriterion {
            squares: &squares,
            generalized: &generalized,
            rank: generalized.len() - occam.len(),
            occam: &occam,
            constant,
        };
        let jet = |u: f64| criterion.jet(u).expect("evaluable jet").0;
        let step = 1.0e-5;
        for u in [-3.0_f64, 0.0, 2.5] {
            let [_, first, second, third] = jet(u);
            let (up, down) = (jet(u + step), jet(u - step));
            for (order, analytic, difference) in [
                (1, first, (up[0] - down[0]) / (2.0 * step)),
                (2, second, (up[1] - down[1]) / (2.0 * step)),
                (3, third, (up[2] - down[2]) / (2.0 * step)),
            ] {
                assert!(
                    (analytic - difference).abs() <= 1.0e-5 * (1.0 + analytic.abs()),
                    "u={u}: derivative of order {order} {analytic} vs central difference \
                     {difference}"
                );
            }
        }
        let (cell_lo, cell_hi) = (-1.0_f64, 1.5_f64);
        let (first_range, second_range) = criterion
            .derivative_ranges(cell_lo, cell_hi)
            .expect("evaluable cell");
        for index in 0..=8 {
            let u = cell_lo + (cell_hi - cell_lo) * index as f64 / 8.0;
            let [_, first, second, _] = jet(u);
            assert!(
                first_range.lo <= first && first <= first_range.hi,
                "C' {first} at u={u} escapes its cell range {first_range:?}"
            );
            assert!(
                second_range.lo <= second && second <= second_range.hi,
                "C'' {second} at u={u} escapes its cell range {second_range:?}"
            );
        }
        let (low, high) = (-8.0_f64, 8.0_f64);
        let selected = criterion.select(low, high).expect("certified selection");
        assert!((low..=high).contains(&selected), "selection {selected} left the window");
        let value = |u: f64| jet(u)[0];
        let selected_value = value(selected);
        for probe in [low, high, 0.0, selected - 1.0e-2, selected + 1.0e-2] {
            if (low..=high).contains(&probe) {
                assert!(
                    selected_value <= value(probe) + 1.0e-9 * (1.0 + selected_value.abs()),
                    "the certified selection {selected} (C={selected_value}) is undercut at \
                     u={probe} (C={})",
                    value(probe)
                );
            }
        }
    }

    /// #2902: on a tail plateau the two log-determinant spectra come in near-equal
    /// pairs. The paired, centred enclosure must still contain `C′` and `C″`, and
    /// the certified search must finish over the whole window instead of
    /// exhausting its subdivisions. The spectra are the coupled dense pair's
    /// axis-1 slice at `ln t = (18, 0)`, where the crosswise enclosure refused.
    #[test]
    fn paired_spectra_certify_a_tail_plateau_2902() {
        let generalized = [
            2.48379138847655e-7,
            5.433397857875894e-13,
            3.215731391695153e-13,
            4.727515454438352e-15,
            8.324201263235987e-16,
            3.0588533320371987e-16,
            6.792526470844082e-17,
        ];
        let occam = [
            5.433407919278328e-13,
            3.2157340977465174e-13,
            4.7275155803531896e-15,
            8.324201266050822e-16,
            3.0588533360082377e-16,
            6.792526477702502e-17,
        ];
        let squares = [1.7_f64, 0.4, 2.2, 0.05, 0.9, 1.1, 0.3];
        let criterion = DiagonalCriterion {
            squares: &squares,
            generalized: &generalized,
            rank: 1,
            occam: &occam,
            constant: 0.0,
        };
        let jet = |u: f64| criterion.jet(u).expect("evaluable jet").0;
        for (cell_lo, cell_hi) in [(30.91_f64, 30.9111_f64), (30.0, 33.0), (-18.0, 42.0)] {
            let (first_range, second_range) = criterion
                .derivative_ranges(cell_lo, cell_hi)
                .expect("evaluable cell");
            for index in 0..=16 {
                let u = cell_lo + (cell_hi - cell_lo) * index as f64 / 16.0;
                let [_, first, second, _] = jet(u);
                assert!(
                    first_range.lo <= first && first <= first_range.hi,
                    "C' {first} at u={u} escapes {first_range:?} on [{cell_lo}, {cell_hi}]"
                );
                assert!(
                    second_range.lo <= second && second <= second_range.hi,
                    "C'' {second} at u={u} escapes {second_range:?} on [{cell_lo}, {cell_hi}]"
                );
            }
        }
        criterion
            .select(-18.0, 42.0)
            .expect("the paired enclosure certifies the plateau");
    }

    /// #2902: where the data term and the log-determinant pairs cancel, the cell
    /// range of `C″` is crosswise and the first-order centred form cannot give `C′`
    /// a sign. The spectra are draw 357's axis-1 slice from the split-penalty
    /// replay (`a_split_penalty_reproduces_the_single_scale_law`), which ran out of
    /// subdivisions on the cell `[−0.732421875, −0.73095703125]`. The second-order
    /// enclosure must still contain `C′` and `C″`, must give that cell its sign,
    /// and the certified search must finish over the whole window.
    #[test]
    fn a_cancelling_slice_takes_its_sign_from_the_second_order_form_2902() {
        let squares = [
            0.0019305094414728218,
            0.01003959414067558,
            0.015890890388347078,
            0.0002931438173635976,
        ];
        let generalized = [
            0.00337861862107217,
            0.0033748178802278823,
            0.0033672419945676687,
            0.003344717051138586,
        ];
        let occam = [
            0.0033900723827513172,
            0.003390072382751316,
            0.003390072382751315,
            0.003390072382751314,
        ];
        let criterion = DiagonalCriterion {
            squares: &squares,
            generalized: &generalized,
            rank: 0,
            occam: &occam,
            constant: 0.0,
        };
        let jet = |u: f64| criterion.jet(u).expect("evaluable jet").0;
        for (cell_lo, cell_hi) in [
            (-0.732421875_f64, -0.73095703125_f64),
            (-1.0, 0.5),
            (-6.0, 6.0),
            (2.9, 3.1),
        ] {
            let (first_range, second_range) = criterion
                .derivative_ranges(cell_lo, cell_hi)
                .expect("evaluable cell");
            for index in 0..=16 {
                let u = cell_lo + (cell_hi - cell_lo) * index as f64 / 16.0;
                let [_, first, second, _] = jet(u);
                assert!(
                    first_range.lo <= first && first <= first_range.hi,
                    "C' {first} at u={u} escapes {first_range:?} on [{cell_lo}, {cell_hi}]"
                );
                assert!(
                    second_range.lo <= second && second <= second_range.hi,
                    "C'' {second} at u={u} escapes {second_range:?} on [{cell_lo}, {cell_hi}]"
                );
            }
        }
        let (first_range, _) = criterion
            .derivative_ranges(-0.732421875, -0.73095703125)
            .expect("evaluable cell");
        assert!(
            first_range.hi < 0.0,
            "C' is negative across the cell, but its enclosure {first_range:?} does not exclude zero"
        );
        criterion
            .select(-6.0, 6.0)
            .expect("the second-order enclosure certifies the cancelling slice");
    }

    /// A slice whose shares all sit on a plateau — within `~1e-9` of one, or at
    /// `~1e-3` — the shape of a Bernoulli null draw's axis slice near its
    /// minimum: `C′` and `C″` are `~4e-3`, so the enclosure's forward-error band
    /// is `~1e-18`. Across a `2⁻²⁴`-wide cell a share near one moves by about one
    /// ulp, and every share polynomial carries `1 − s`: taken as the
    /// subtraction, that ulp moves `C″` between the cell's ends by `~ε`, far
    /// more than `|C‴|·width` plus the band, so the second-order enclosure came
    /// out empty on about half of these cells and the draw's selection was
    /// refused. Every narrow cell must stay evaluable and must contain the jet.
    #[test]
    fn a_saturated_share_keeps_the_curvature_enclosure_consistent() {
        let generalized = [1.0e9_f64, 3.0e9, 1.0e-3];
        let occam = [2.0e9_f64];
        let squares = [2.0_f64, 1.5, 0.7];
        // Two saturated generalized shares less one saturated Occam share: the
        // rank cancels them, and `C′` is the unsaturated share's `~4e-3`.
        let criterion = DiagonalCriterion {
            squares: &squares,
            generalized: &generalized,
            rank: 1,
            occam: &occam,
            constant: 0.0,
        };
        let jet = |u: f64| criterion.jet(u).expect("evaluable jet").0;
        let width = 2.0_f64.powi(-24);
        for index in 0..64 {
            let cell_lo = -1.0 + 2.0 * index as f64 / 64.0;
            let cell_hi = cell_lo + width;
            let (first_range, second_range) =
                criterion.derivative_ranges(cell_lo, cell_hi).unwrap_or_else(|| {
                    panic!("the narrow cell [{cell_lo}, {cell_hi}] has an empty enclosure")
                });
            for u in [cell_lo, 0.5 * (cell_lo + cell_hi), cell_hi] {
                let [_, first, second, _] = jet(u);
                assert!(
                    first_range.lo <= first && first <= first_range.hi,
                    "C' {first} at u={u} escapes {first_range:?} on [{cell_lo}, {cell_hi}]"
                );
                assert!(
                    second_range.lo <= second && second <= second_range.hi,
                    "C'' {second} at u={u} escapes {second_range:?} on [{cell_lo}, {cell_hi}]"
                );
            }
        }
    }

    /// The geometry a bare generalized spectrum corresponds to: unit
    /// information, one penalty whose whitened form is `diag(ν)`, fitted at
    /// `λ̂ = 1`. `eig(Ĩ⁻¹S) = ν` exactly, so this is the identity map from the
    /// spectrum these tests are written in terms of onto the object the replay
    /// is built from.
    fn diagonal(spectrum: &[f64]) -> SelectionGeometry {
        let q = spectrum.len();
        let mut penalty = Array2::<f64>::zeros((q, q));
        for (index, &value) in spectrum.iter().enumerate() {
            penalty[[index, index]] = value;
        }
        SelectionGeometry::whiten(&Array2::eye(q), std::slice::from_ref(&penalty), &[0.0])
            .expect("diagonal geometry")
    }

    fn replay_from(spectrum: &[f64], window: (f64, f64), draws: usize) -> SmoothLrSelectionReplay {
        match SmoothLrSelectionReplay::from_geometry(&diagonal(spectrum), &[window], draws, draws, None) {
            SmoothLrSelection::Replayed(replay) => replay,
            SmoothLrSelection::Declined(reason) => {
                panic!("expected a replay, declined: {}", reason.label())
            }
        }
    }

    /// `N(0, 1)` draws for the calibration study below, from a counter stream
    /// the replay's own strata do not share (Box–Muller on SplitMix64 words).
    fn observation(rep: u64, dimension: usize) -> Vec<f64> {
        let unit = |counter: u64| {
            ((split_mix64(0xC0FF_EE00_0000_0000 ^ counter) >> 11) as f64 + 0.5)
                * (-53.0_f64).exp2()
        };
        (0..dimension as u64)
            .map(|j| {
                let base = 2 * (rep * dimension as u64 + j);
                let (u, v) = (unit(base), unit(base + 1));
                (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
            })
            .collect()
    }

    /// Two-sided calibration of the selection-corrected tail, seeded.
    ///
    /// In the replay's own world the observation is a whitened score
    /// `z ~ N(0, I)` and its statistic at the fitted scale is
    /// `x = W_q(0; z) = Σ_j w_j z_j²`. The p-value the corrected reference
    /// assigns it must be `U(0, 1)` over the WHOLE range — not merely sized at
    /// one α — because a conservative p is as wrong as an anti-conservative one.
    ///
    /// Reading the selection arm at the observed `x` itself asks the wrong
    /// question: `x` is the observation under the fitted `λ̂`, and the replay's
    /// selected law is a law of the statistic under the replay's selection. On
    /// the same observations that version fails this test (asserted below), so
    /// the test pins the rescoring and not just the arithmetic.
    ///
    /// Two Monte-Carlo errors are in play and both are carried: the `R`
    /// observations' (`√(α(1−α)/R)`) and the replay's own finite `N` draws
    /// (`√(α(1−α)/N)`), which fix one empirical selected law for every
    /// observation. The tolerances are three combined standard errors for the
    /// sizes and the Kolmogorov `0.999` quantile `1.949·√(1/R + 1/N)` for `D`.
    #[test]
    fn the_rescored_selection_tail_is_uniform_on_both_sides() {
        let generalized = spectrum();
        let dimension = generalized.len();
        let window = (-30.0_f64, 30.0);
        let draws = SMOOTH_LR_SELECTION_DRAWS;
        let weights: Vec<f64> = generalized
            .iter()
            .map(|&nu| {
                let share = nu / (1.0 + nu);
                1.0 - share * share
            })
            .collect();
        let base = replay_from(&generalized, window, draws);
        let replications = 600_u64;
        let mut rescored = Vec::new();
        let mut unscored = Vec::new();
        for rep in 0..replications {
            let z = observation(rep, dimension);
            let statistic: f64 = z
                .iter()
                .zip(weights.iter())
                .map(|(value, weight)| weight * value * value)
                .sum();
            let replay = match SmoothLrSelectionReplay::from_geometry(
                &diagonal(&generalized),
                &[window],
                draws,
                draws,
                Some(&z),
            ) {
                SmoothLrSelection::Replayed(replay) => replay,
                SmoothLrSelection::Declined(reason) => {
                    panic!("rep {rep}: declined: {}", reason.label())
                }
            };
            // The observation does not move the draws, and its conditional
            // statistic is the one the fit reports.
            assert_eq!(replay.selection_sample, base.selection_sample);
            let observed = replay.observed.expect("an observed selection");
            assert!(
                (observed.conditional - statistic).abs() <= 1e-12 * statistic.max(1.0),
                "rep {rep}: W_q(0; z) = {} but Σ w z² = {statistic}",
                observed.conditional
            );
            // The LR reference's own read at a known scale: the exact
            // conditional tail at `W`, moved by the replay's shift with the
            // observation scored under the replay's selection.
            let conditional =
                wald_conditional_tail(&weights, statistic, None).expect("a finite tail");
            let p_value = |replay: &SmoothLrSelectionReplay| {
                let (shift, _) = replay
                    .tail_shift_at(statistic, replay.observed_selection_threshold(statistic));
                (conditional + shift).clamp(0.0, 1.0)
            };
            rescored.push(p_value(&replay));
            let mut blind = replay;
            blind.observed = None;
            unscored.push(p_value(&blind));
        }
        let r = replications as f64;
        let n = draws as f64;
        let kolmogorov = |sample: &[f64]| {
            let mut sorted = sample.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted
                .iter()
                .enumerate()
                .map(|(i, &p)| (p - i as f64 / r).max((i as f64 + 1.0) / r - p))
                .fold(0.0_f64, f64::max)
        };
        let ks_limit = 1.949 * (1.0 / r + 1.0 / n).sqrt();
        let d = kolmogorov(&rescored);
        assert!(d <= ks_limit, "two-sided KS D = {d:.4} > {ks_limit:.4}");
        for alpha in [0.10_f64, 0.05, 0.01] {
            let size = rescored.iter().filter(|&&p| p <= alpha).count() as f64 / r;
            let tolerance = 3.0 * (alpha * (1.0 - alpha) * (1.0 / r + 1.0 / n)).sqrt();
            assert!(
                (size - alpha).abs() <= tolerance,
                "size at {alpha}: {size:.4}, outside {alpha} ± {tolerance:.4}"
            );
        }
        let blind_d = kolmogorov(&unscored);
        assert!(
            blind_d > ks_limit,
            "the unrescored tail must fail this test (D = {blind_d:.4})"
        );
    }

    /// The replay is a p-value input, so it must not depend on a thread, a
    /// machine or a run (#1017). It is a counter-based stratified stream, and
    /// this pins that: two independent generations are bit-identical.
    #[test]
    fn the_replay_is_bit_identical_across_generations() {
        let first = replay_from(&spectrum(), (-8.0, 8.0), SMOOTH_LR_SELECTION_DRAWS);
        let second = replay_from(&spectrum(), (-8.0, 8.0), SMOOTH_LR_SELECTION_DRAWS);
        assert_eq!(first, second);
        for statistic in [0.05_f64, 0.5, 1.5, 4.0] {
            assert_eq!(first.tail_shift_at(statistic, statistic), second.tail_shift_at(statistic, statistic));
        }
    }

    /// Every coordinate draws from the SAME stratum midpoints, in a different
    /// order — a Latin hypercube. Sorting one coordinate's draws must reproduce
    /// the strata exactly, or the stratification is not what the doc claims and
    /// the variance argument behind the draw budget does not hold.
    #[test]
    fn every_coordinate_is_a_permutation_of_the_same_strata() {
        let replay = replay_from(&[1.0], (-6.0, 6.0), SMOOTH_LR_SELECTION_DRAWS);
        let mut conditional = replay.conditional_sample.clone();
        conditional.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        // The conditional weight at `t = 1` for `ν = 1` is `2·½ − ¼ = 0.75`.
        let mut expected: Vec<f64> = (0..SMOOTH_LR_SELECTION_DRAWS)
            .map(|bin| {
                let uniform = (bin as f64 + 0.5) / SMOOTH_LR_SELECTION_DRAWS as f64;
                let normal =
                    gam_math::probability::standard_normal_quantile(uniform).expect("interior");
                0.75 * normal * normal
            })
            .collect();
        expected.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        for (got, want) in conditional.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "the conditional sample is not the strata: {got} vs {want}"
            );
        }
    }

    /// The replay is not inert, and what it does is DISPERSE the statistic
    /// rather than shift it.
    ///
    /// I assumed twice over that selecting `λ` could only inflate `W`, and the
    /// measurement says otherwise both times: on this spectrum `E[W(λ̂)] = 1.13`
    /// against `E[W(1)] = 2.17`, and at `W = E[W(1)]` the replay *shrinks* the
    /// upper tail by `0.19`. Under a fresh null draw the criterion usually
    /// prefers MORE shrinkage than the fitted point, so the mean falls — while
    /// a draw that happens to look wiggly buys itself a smaller `λ` and a much
    /// larger `W`, so the spread rises. Dispersion is the invariant; the sign of
    /// the tail shift is a property of where the fitted `λ̂` sits relative to the
    /// null's typical choice, i.e. of the fit, not of the construction. It is
    /// asserted on real fits in the integration suite, not here.
    #[test]
    fn selection_disperses_the_statistic_and_is_not_inert() {
        let replay = replay_from(&spectrum(), (-10.0, 10.0), SMOOTH_LR_SELECTION_DRAWS);
        let draws = replay.selection_sample.len() as f64;
        let mean = |sample: &[f64]| sample.iter().sum::<f64>() / draws;
        let variance = |sample: &[f64]| {
            let m = mean(sample);
            sample.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / draws
        };
        assert!(
            variance(&replay.selection_sample) > variance(&replay.conditional_sample),
            "selection did not disperse the statistic: {} vs {}",
            variance(&replay.selection_sample),
            variance(&replay.conditional_sample)
        );
        let conditional_mean = mean(&replay.conditional_sample);
        let mut any_move = false;
        for multiple in [0.25_f64, 1.0, 4.0, 16.0] {
            let statistic = multiple * conditional_mean;
            let (shift, standard_error) = replay.tail_shift_at(statistic, statistic);
            assert!(
                shift.is_finite() && (-1.0..=1.0).contains(&shift) && standard_error >= 0.0,
                "at W={statistic} the shift {shift} is not a probability difference"
            );
            if shift.abs() > 4.0 * standard_error {
                any_move = true;
            }
        }
        assert!(
            any_move,
            "the replay never moved the tail by more than its own noise — it is inert"
        );
        // Dispersion at the far end, stated where `N` draws can still resolve
        // it: the most extreme value a selected scale reaches has to exceed the
        // most extreme the fitted scale reaches, on the SAME draws.
        let extreme = |sample: &[f64]| sample.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            extreme(&replay.selection_sample) > extreme(&replay.conditional_sample),
            "the selected law never reached past the conditional one's most extreme \
             draw ({} vs {}), so it is not dispersing the upper tail at all",
            extreme(&replay.selection_sample),
            extreme(&replay.conditional_sample)
        );
    }

    /// The published standard error has to be HONEST, because it is what the
    /// report's accuracy bound is built from. Quadrupling the draws must move
    /// the shift by no more than the two reported standard errors allow — a
    /// self-consistency check that fails if the error is understated, which is
    /// the failure mode that matters (an overstated one is merely pessimistic).
    #[test]
    fn the_published_standard_error_covers_a_four_fold_draw_increase() {
        let coarse = replay_from(&spectrum(), (-10.0, 10.0), 4096);
        let fine = replay_from(&spectrum(), (-10.0, 10.0), 16384);
        let conditional_mean = coarse.conditional_sample.iter().sum::<f64>()
            / coarse.conditional_sample.len() as f64;
        for multiple in [0.5_f64, 1.0, 2.0, 4.0, 8.0] {
            let statistic = multiple * conditional_mean;
            let (coarse_shift, coarse_error) = coarse.tail_shift_at(statistic, statistic);
            let (fine_shift, fine_error) = fine.tail_shift_at(statistic, statistic);
            let allowance = 3.0 * (coarse_error + fine_error) + 1e-12;
            assert!(
                (coarse_shift - fine_shift).abs() <= allowance,
                "at W={statistic} the shift moved {coarse_shift} -> {fine_shift} under a \
                 four-fold draw increase, outside the {allowance} the two published \
                 standard errors ({coarse_error:.3e}, {fine_error:.3e}) allow"
            );
        }
        // And the finer run's own error must actually be smaller — a standard
        // error that does not fall with the budget is not a standard error.
        let statistic = 2.0 * conditional_mean;
        assert!(
            fine.tail_shift_at(statistic, statistic).1 < coarse.tail_shift_at(statistic, statistic).1,
            "the reported standard error did not fall when the draws quadrupled"
        );
    }

    /// The multi-scale replay must agree with the one-dimensional one when
    /// there is only one scale to select — that is the seam between the two
    /// paths, and a seam nobody checks is a seam that drifts.
    ///
    /// `generate_multiscale` refuses a single penalty by construction (there is
    /// nothing it can do that the diagonal path cannot do faster), so the
    /// agreement is checked by handing it the SAME penalty split in two halves:
    /// `S = ½S + ½S` selects two scales whose sum is the one scale, so the
    /// two-dimensional box contains the one-dimensional family along its
    /// diagonal and the two references must land on the same law.
    #[test]
    fn a_split_penalty_reproduces_the_single_scale_law() {
        let q = 4;
        let information = Array2::<f64>::eye(q);
        let mut penalty = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            penalty[[index, index]] = 0.5 * (index as f64 + 1.0);
        }
        let half = penalty.clone() * 0.5;
        let split_geometry =
            SelectionGeometry::whiten(&information, &[half.clone(), half], &[0.0, 0.0])
                .expect("split geometry");
        let split = SmoothLrSelectionReplay::generate_multiscale(
            &split_geometry,
            &[(-6.0, 6.0), (-6.0, 6.0)],
            2048,
            None,
        )
        .expect("multiscale replay");
        // With `information = I` the generalized eigenvalues ARE the penalty's
        // diagonal, so the one-dimensional replay is directly constructible.
        let single = replay_from(
            &(0..q)
                .map(|index| 0.5 * (index as f64 + 1.0))
                .collect::<Vec<f64>>(),
            (-6.0, 6.0),
            2048,
        );
        let mean = |sample: &[f64]| sample.iter().sum::<f64>() / sample.len() as f64;
        let split_mean = mean(&split.conditional_sample);
        let single_mean = mean(&single.conditional_sample);
        assert!(
            (split_mean - single_mean).abs() <= 0.05 * single_mean.abs().max(1.0),
            "the two paths disagree on the CONDITIONAL law: {split_mean} vs {single_mean}"
        );
        let split_selected = mean(&split.selection_sample);
        let single_selected = mean(&single.selection_sample);
        eprintln!(
            "[2672 seam] conditional {split_mean:.6} vs {single_mean:.6}; \
             selected {split_selected:.6} vs {single_selected:.6}"
        );
        assert!(
            (split_selected - single_selected).abs() <= 0.25 * single_selected.abs().max(1.0),
            "the two paths disagree on the SELECTED law by more than Monte-Carlo \
             error can explain: {split_selected} vs {single_selected}"
        );
        // Both lanes publish the same generalized spectrum, because it is a
        // property of the term and not of which grid was affordable.
        for (from_split, from_single) in split
            .generalized
            .iter()
            .zip(single.generalized.iter())
        {
            assert!(
                (from_split - from_single).abs() <= 1e-9 * from_single.abs().max(1.0),
                "the two lanes publish different generalized spectra: {:?} vs {:?}",
                split.generalized,
                single.generalized
            );
        }
    }

    /// The multi-scale path is for terms that actually select several scales,
    /// and it says so rather than pretending on the ones it cannot serve.
    #[test]
    fn the_multiscale_path_declines_what_it_cannot_serve() {
        let information = Array2::<f64>::eye(3);
        let penalty = Array2::<f64>::eye(3);
        // One scale: the diagonal path is strictly better, so this declines.
        let single = SelectionGeometry::whiten(
            &information,
            std::slice::from_ref(&penalty),
            &[0.0],
        )
        .expect("single geometry");
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(
                &single,
                &[(-6.0, 6.0)],
                256,
                None,
            )
            .is_err()
        );
        // Five scales: no bracket budget caps the axes any more, so the
        // multi-scale lane serves every one of them.
        let many = vec![penalty.clone(); 5];
        let windows = vec![(-6.0, 6.0); 5];
        let crowded = SelectionGeometry::whiten(&information, &many, &[0.0; 5])
            .expect("crowded geometry");
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(&crowded, &windows, 256, None).is_ok(),
            "a term with five scales is replayed over all five"
        );
        assert!(
            SmoothLrSelectionReplay::from_geometry(&crowded, &windows, 256, 256, None)
                .replay()
                .is_some(),
            "a term with five scales still gets a replay"
        );
        // Every window closed: nothing to select on any axis.
        let pair = SelectionGeometry::whiten(
            &information,
            &[penalty.clone(), penalty.clone()],
            &[0.0, 0.0],
        )
        .expect("pair geometry");
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(
                &pair,
                &[(1.0, 1.0), (2.0, 2.0)],
                256,
                None,
            ) == Err(SmoothLrSelectionDecline::WindowClosed)
        );
        // But ONE open axis is still a selection, and used to be discarded with
        // the closed one — the intersection of the two windows is empty.
        assert!(
            SmoothLrSelectionReplay::generate_multiscale(
                &pair,
                &[(1.0, 1.0), (-6.0, 6.0)],
                256,
                None,
            )
            .is_ok(),
            "a scale whose own window is open must still be replayed when a \
             SIBLING scale's window is closed"
        );
        // Information with no identified direction cannot be whitened at all.
        assert!(
            SelectionGeometry::whiten(
                &Array2::<f64>::zeros((3, 3)),
                &[penalty.clone(), penalty],
                &[0.0, 0.0],
            )
            .is_err()
        );
    }

    /// A term with nothing to select — no penalized direction, or a window the
    /// solver's box has closed — has no replay, and the conditional law is the
    /// selection law. This is the branch that keeps an unpenalized block exactly
    /// the textbook chi-square.
    #[test]
    fn nothing_to_select_means_no_replay() {
        assert_eq!(
            SelectionGeometry::whiten(&Array2::eye(3), &[], &[]).err(),
            Some(SmoothLrSelectionDecline::NoPenaltyComponents)
        );
        assert_eq!(
            SelectionGeometry::whiten(&Array2::eye(2), &[Array2::zeros((2, 2))], &[0.0]).err(),
            Some(SmoothLrSelectionDecline::NoPenaltyComponents),
            "a component that whitens to nothing penalizes nothing: a fact about the fit, \
             not a refusal"
        );
        let geometry = diagonal(&spectrum());
        for window in [(4.0_f64, -4.0_f64), (f64::NAN, 1.0)] {
            assert_eq!(
                SmoothLrSelectionReplay::from_geometry(&geometry, &[window], 256, 256, None).decline(),
                Some(SmoothLrSelectionDecline::WindowClosed),
                "a closed window must decline with a NAMED reason"
            );
        }
    }

    /// The MULTI-SCALE replay is a p-value input too, and its per-draw descent
    /// is the part of it that could most easily stop being a pure function of
    /// the geometry (#1017).
    ///
    /// `the_replay_is_bit_identical_across_generations` pins the diagonal lane,
    /// where the whole computation is a fixed grid. This pins the lane that
    /// searches: two generations must agree bit for bit on both samples and on
    /// every tail shift read off them.
    #[test]
    fn the_multiscale_replay_is_bit_identical_across_generations() {
        let q = 5;
        let mut bending = Array2::<f64>::zeros((q, q));
        let mut ridge = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            if index < 3 {
                bending[[index, index]] = 1.0 + index as f64;
            } else {
                ridge[[index, index]] = 0.5 + index as f64;
            }
        }
        let generate = || {
            let geometry = SelectionGeometry::whiten(
                &Array2::eye(q),
                &[bending.clone(), ridge.clone()],
                &[6.0, -9.0],
            )
            .expect("geometry");
            SmoothLrSelectionReplay::generate_multiscale(
                &geometry,
                &[(-36.0, 24.0), (-21.0, 39.0)],
                512,
                None,
            )
            .expect("multiscale replay")
        };
        let first = generate();
        let second = generate();
        assert_eq!(first, second);
        for statistic in [0.05_f64, 0.5, 1.5, 4.0] {
            assert_eq!(first.tail_shift_at(statistic, statistic), second.tail_shift_at(statistic, statistic));
        }
    }

    /// #2672: the eigen route and the factor route are the SAME function.
    ///
    /// The multi-scale lane prices its conditional arm through the eigensystem of
    /// `T(t)` and each draw's selected statistic through two triangular
    /// factorizations, because one amortizes over draws and the other cannot.
    /// They are two routes to one number, and the control variate subtracts one
    /// from the other — so a discrepancy between them is not a rounding
    /// difference, it is a shift read between two different functions.
    ///
    /// Checked on a DENSE information with two dense components at separations
    /// up to the box's own width, which is where the two routes' conditioning
    /// differs most, and at `t` off the fitted point in both directions.
    #[test]
    fn the_two_evaluators_price_a_point_identically_2672() {
        let q = 7;
        let mixing = Array2::from_shape_fn((q, q), |(row, column)| {
            let a = row as f64 + 1.0;
            let b = column as f64 + 1.0;
            ((a * 0.7 + b * 1.3).sin() + 0.25 * (a * b).cos()) / (1.0 + 0.1 * a * b)
        });
        let information = mixing.dot(&mixing.t()) + Array2::<f64>::eye(q) * 0.5;
        let mut bending = Array2::<f64>::zeros((q, q));
        for index in 0..q - 2 {
            bending[[index, index]] = 1.0;
            bending[[index, index + 1]] = -0.5;
            bending[[index + 1, index]] = -0.5;
        }
        let bending = mixing.dot(&bending.dot(&mixing.t()));
        let bending = bending.dot(&bending.t());
        let ridge = mixing.dot(&mixing.t());
        let draw: Vec<f64> = (0..q)
            .map(|index| ((index as f64 + 1.0) * 0.9).sin() + 0.3)
            .collect();
        let norm_squared: f64 = draw.iter().map(|value| value * value).sum();
        for separation in [0.0_f64, 18.0, 40.0] {
            let geometry = SelectionGeometry::whiten(
                &information,
                &[bending.clone(), ridge.clone()],
                &[0.5 * separation, -0.5 * separation],
            )
            .expect("geometry");
            let mut factor = SelectionFactor::new(&geometry);
            let mut coordinates = vec![0.0_f64; geometry.rank];
            for log_t in [[0.0_f64, 0.0], [-2.5, 1.75], [3.0, -4.0], [-8.0, -8.0]] {
                let evaluated = geometry.at(&log_t).expect("eigen route");
                // The eigen route's criterion from its own spectrum: the Occam term
                // `log|I + T| − log|T|₊` over the structural rank, then the data
                // operator `f_j = e_j/(1 + e_j)` on each squared coordinate.
                let log_det_hessian: f64 =
                    evaluated.eigenvalues.iter().map(|value| value.ln_1p()).sum();
                let log_det_penalty: f64 = evaluated.eigenvalues[..geometry.rank]
                    .iter()
                    .map(|value| value.ln())
                    .sum();
                let mut criterion = log_det_hessian - log_det_penalty;
                let mut statistic = 0.0_f64;
                for column in 0..geometry.dimension {
                    let mut coordinate = 0.0_f64;
                    for row in 0..geometry.dimension {
                        coordinate += draw[row] * evaluated.basis[[row, column]];
                    }
                    let square = coordinate * coordinate;
                    let eigenvalue = evaluated.eigenvalues[column];
                    criterion += square * (eigenvalue / (1.0 + eigenvalue));
                    statistic += square * evaluated.weights[column];
                }
                assert!(
                    factor.refactor(&geometry, &log_t),
                    "separation {separation}, ln t = {log_t:?}: the factor route refused a \
                     point the eigen route priced"
                );
                for column in 0..geometry.rank {
                    coordinates[column] = (0..geometry.dimension)
                        .map(|row| draw[row] * geometry.range_basis[[row, column]])
                        .sum();
                }
                let (fast_criterion, fast_statistic) =
                    factor.score(&coordinates, norm_squared);
                assert!(
                    (fast_criterion - criterion).abs() <= 1e-8 * criterion.abs().max(1.0),
                    "separation {separation}, ln t = {log_t:?}: criterion {fast_criterion} \
                     (factor) vs {criterion} (eigen)"
                );
                assert!(
                    (fast_statistic - statistic).abs() <= 1e-8 * statistic.abs().max(1.0),
                    "separation {separation}, ln t = {log_t:?}: statistic {fast_statistic} \
                     (factor) vs {statistic} (eigen)"
                );
            }
        }
    }

    /// A dense information with two dense components, a bending-style one and a
    /// ridge, and one fixed draw: the fixture on which the multi-scale criterion's
    /// scales couple.
    fn dense_pair() -> (Array2<f64>, Array2<f64>, Array2<f64>, Vec<f64>) {
        let q = 7;
        let mixing = Array2::from_shape_fn((q, q), |(row, column)| {
            let a = row as f64 + 1.0;
            let b = column as f64 + 1.0;
            ((a * 0.7 + b * 1.3).sin() + 0.25 * (a * b).cos()) / (1.0 + 0.1 * a * b)
        });
        let information = mixing.dot(&mixing.t()) + Array2::<f64>::eye(q) * 0.5;
        let mut bending = Array2::<f64>::zeros((q, q));
        for index in 0..q - 2 {
            bending[[index, index]] = 1.0;
            bending[[index, index + 1]] = -0.5;
            bending[[index + 1, index]] = -0.5;
        }
        let bending = mixing.dot(&bending.dot(&mixing.t()));
        let bending = bending.dot(&bending.t());
        let ridge = mixing.dot(&mixing.t());
        let draw = (0..q)
            .map(|index| ((index as f64 + 1.0) * 0.9).sin() + 0.3)
            .collect();
        (information, bending, ridge, draw)
    }

    /// A draw's coordinates in the geometry's range basis.
    fn range_coordinates(geometry: &SelectionGeometry, draw: &[f64]) -> Vec<f64> {
        (0..geometry.rank)
            .map(|column| {
                (0..geometry.dimension)
                    .map(|row| draw[row] * geometry.range_basis[[row, column]])
                    .sum()
            })
            .collect()
    }

    /// The separable pair — unit information, a bending penalty on four
    /// directions and a ridge on the other two, at the separation a null-true
    /// smooth reaches — and the coupled dense pair: the two fixtures the
    /// multi-scale selection is checked on.
    fn separable_and_coupled() -> (SelectionGeometry, SelectionGeometry) {
        let q = 6;
        let mut bending = Array2::<f64>::zeros((q, q));
        let mut ridge = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            if index < 4 {
                bending[[index, index]] = 1.0 + index as f64;
            } else {
                ridge[[index, index]] = 1.0;
            }
        }
        let separable =
            SelectionGeometry::whiten(&Array2::eye(q), &[bending, ridge], &[12.0, -12.0])
                .expect("separable geometry");
        let (information, dense_bending, dense_ridge, _) = dense_pair();
        let coupled =
            SelectionGeometry::whiten(&information, &[dense_bending, dense_ridge], &[9.0, -9.0])
                .expect("coupled geometry");
        (separable, coupled)
    }

    /// #2902: the factor route's gradient and Hessian are the criterion's.
    ///
    /// Checked against the eigen route's spectrum in closed form, not against
    /// differences of the value. Under a COMMON shift `ρ + u·1` every eigenvalue
    /// of `T` scales by `e^u`, so with `s_j = e_j/(1 + e_j)` and `c_j` the draw's
    /// eigen-coordinates
    ///
    /// ```text
    /// Σ_j g_j   = Σ_j c_j² s_j(1 − s_j) + Σ_j s_j − rank,
    /// Σ_jk H_jk = Σ_j c_j² s_j(1 − s_j)(1 − 2s_j) + Σ_j s_j(1 − s_j),
    /// ```
    ///
    /// on either fixture. On the SEPARABLE pair every eigen-direction belongs to
    /// one scale, so the same sums restricted to a scale's own directions (and
    /// its own rank) are that scale's `g_j` and `H_jj`, and `H_01 = 0`.
    #[test]
    fn the_criterion_jet_is_the_closed_form_derivative_2902() {
        let (separable, coupled) = separable_and_coupled();
        let (_, _, _, dense_draw) = dense_pair();
        let separable_draw: Vec<f64> = (0..6)
            .map(|index| ((index as f64 + 1.0) * 1.3).cos() + 0.4)
            .collect();
        for (label, geometry, draw, per_axis) in [
            ("separable", &separable, &separable_draw, true),
            ("coupled", &coupled, &dense_draw, false),
        ] {
            let coordinates = range_coordinates(geometry, draw);
            let mut factor = SelectionFactor::new(geometry);
            for log_t in [[0.0_f64, 0.0], [-2.5, 1.75], [3.0, -4.0], [-8.0, 6.0]] {
                assert!(
                    factor.refactor(geometry, &log_t),
                    "{label} {log_t:?}: refused"
                );
                let jet = factor.jet(geometry, &coordinates).expect("finite jet");
                let hessian = jet.hessian.clone().expect("a Hessian");
                let evaluated = geometry.at(&log_t).expect("eigen route");
                // Per scale (index 0, 1) and over both (index 2):
                // `(Σ c²g + Σ s, Σ c²g(1 − 2s) + Σ g, owned rank)`.
                let mut first = [0.0_f64; 3];
                let mut second = [0.0_f64; 3];
                let mut owned = [0usize; 3];
                for column in 0..geometry.dimension {
                    let eigenvalue = evaluated.eigenvalues[column];
                    let share = eigenvalue / (1.0 + eigenvalue);
                    let spread = share * (1.0 - share);
                    let coordinate: f64 = (0..geometry.dimension)
                        .map(|row| draw[row] * evaluated.basis[[row, column]])
                        .sum();
                    let square = coordinate * coordinate;
                    let reach: Vec<f64> = geometry
                        .roots
                        .iter()
                        .map(|root| {
                            root.dot(&evaluated.basis.column(column))
                                .iter()
                                .map(|value| value * value)
                                .sum()
                        })
                        .collect();
                    let axis = usize::from(reach[1] > reach[0]);
                    let structural = column < geometry.rank;
                    for slot in [axis, 2] {
                        first[slot] += square * spread + share;
                        second[slot] += square * spread * (1.0 - 2.0 * share) + spread;
                        owned[slot] += usize::from(structural);
                    }
                }
                let close = |analytic: f64, closed: f64| {
                    (analytic - closed).abs() <= 1e-8 * (1.0 + closed.abs())
                };
                let total_gradient = jet.gradient.sum();
                let total_hessian = hessian.sum();
                let common_first = first[2] - owned[2] as f64;
                assert!(
                    close(total_gradient, common_first),
                    "{label} {log_t:?}: Σg = {total_gradient} vs closed form {common_first}"
                );
                assert!(
                    close(total_hessian, second[2]),
                    "{label} {log_t:?}: ΣH = {total_hessian} vs closed form {}",
                    second[2]
                );
                if per_axis {
                    for axis in 0..2 {
                        let closed = first[axis] - owned[axis] as f64;
                        assert!(
                            close(jet.gradient[axis], closed),
                            "{label} {log_t:?}: g_{axis} = {} vs closed form {closed}",
                            jet.gradient[axis]
                        );
                        assert!(
                            close(hessian[[axis, axis]], second[axis]),
                            "{label} {log_t:?}: H_{axis}{axis} = {} vs closed form {}",
                            hessian[[axis, axis]],
                            second[axis]
                        );
                    }
                    assert!(
                        hessian[[0, 1]].abs() <= 1e-8 * (1.0 + second[2].abs()),
                        "{label} {log_t:?}: separable scales couple, H_01 = {}",
                        hessian[[0, 1]]
                    );
                }
            }
        }
    }

    /// #2902: each draw's multi-scale selection is a certified stationary point
    /// of the box-constrained criterion, and a local minimum of it.
    ///
    /// This is the claim [`SmoothLrSelectionReplay::select_draw`] makes, checked
    /// at the point it returns through the jet and the canonical evaluator. With a
    /// scale at its wall held only while its gradient points out of the window,
    /// the Newton decrement over the free scales must certify against the
    /// criterion's rounding bands, and no probe in a small box around the
    /// selection may undercut it.
    #[test]
    fn multiscale_selection_is_a_certified_box_stationary_point_2902() {
        let (separable, coupled) = separable_and_coupled();
        let windows = [(-42.0_f64, 18.0_f64), (-18.0, 42.0)];
        for (label, geometry) in [("separable", &separable), ("coupled", &coupled)] {
            let mut factor = SelectionFactor::new(geometry);
            let mut stream = SelectionDrawStream::new(geometry.dimension, 48);
            let mut draw = vec![0.0_f64; geometry.dimension];
            let mut selected = vec![0.0_f64; 2];
            for index in 0..48 {
                stream.fill_normals(&mut draw);
                let norm_squared: f64 = draw.iter().map(|value| value * value).sum();
                let coordinates = range_coordinates(geometry, &draw);
                SmoothLrSelectionReplay::select_draw(
                    geometry,
                    &windows,
                    &coordinates,
                    &mut factor,
                    &mut selected,
                )
                .unwrap_or_else(|decline| {
                    panic!("{label} draw {index}: no certified selection ({decline:?})")
                });
                let jet = factor.jet(geometry, &coordinates).expect("finite jet");
                let hessian = jet.hessian.clone().expect("a Hessian");
                let bands = jet.decrement_bands.clone().expect("bands");
                // A scale is held by its wall exactly when the descent direction
                // `−g` leaves the window there.
                let active: Vec<bool> = (0..2)
                    .map(|axis| {
                        let (low, high) = windows[axis];
                        let slope = jet.gradient[axis];
                        (selected[axis] == low && slope > 0.0)
                            || (selected[axis] == high && slope < 0.0)
                    })
                    .collect();
                let verdict =
                    opt::newton_decrement_verdict(&hessian, &jet.gradient, Some(&active), &bands);
                assert!(
                    verdict.is_certified(),
                    "{label} draw {index}: the selection {selected:?} (held {active:?}) is not a \
                     certified box-stationary point: {verdict:?}"
                );
                let at_selected = factor.score(&coordinates, norm_squared).0;
                let tolerance = 1e-10 * (1.0 + at_selected.abs());
                for step in [1e-3_f64, 1e-1] {
                    for (d0, d1) in [
                        (1.0, 0.0),
                        (-1.0, 0.0),
                        (0.0, 1.0),
                        (0.0, -1.0),
                        (1.0, 1.0),
                        (1.0, -1.0),
                        (-1.0, 1.0),
                        (-1.0, -1.0),
                    ] {
                        let probe = [
                            (selected[0] + step * d0).clamp(windows[0].0, windows[0].1),
                            (selected[1] + step * d1).clamp(windows[1].0, windows[1].1),
                        ];
                        assert!(
                            factor.refactor(geometry, &probe),
                            "{label}: the evaluator refused the probe {probe:?}"
                        );
                        let at_probe = factor.score(&coordinates, norm_squared).0;
                        assert!(
                            at_selected <= at_probe + tolerance,
                            "{label} draw {index}: the selection {selected:?} (V={at_selected}) \
                             is undercut at {probe:?} (V={at_probe})"
                        );
                    }
                }
            }
        }
    }

    /// #2672: the criterion's log-determinant is priced from the stacked scaled
    /// ROOTS, so a term whose scales separate keeps the coercivity that decides
    /// the selection.
    ///
    /// The replay used to read `Σ_{e > 0} log(1 + 1/e)` off the eigenvalues of
    /// the ASSEMBLED whitened sum. That is the route `penalty_logdet.rs`'s
    /// `SpectrumScale` documents as `O(ε·κ(S_λ))` — and `κ` here is
    /// `exp(ρ̂₁ − ρ̂₂)`, which the box allows to reach `e⁶⁰`. Past `κ ≈ 1e16` the
    /// smaller scale's genuine modes are below the eigendecomposition's own
    /// noise floor, so their `log(1 + 1/e)` is dropped when the noise lands
    /// negative and invented when it lands positive.
    ///
    /// The identity that makes this checkable without a second implementation:
    /// under a COMMON shift the criterion's offset is exactly
    /// `Σ_j log(1 + t·ν_j) − rank·ln t − Σ_{j<rank} ln ν_j`, so scaling every
    /// scale by `t` must move the offset by exactly `−rank·ln t` once the
    /// `log(1 + tν)` part is subtracted. That is a statement about the ANSWER,
    /// not about the arithmetic, and the assembled route violates it by tens of
    /// nats at the separations this fixture uses.
    #[test]
    fn the_criterion_keeps_its_coercivity_when_the_scales_separate() {
        let q = 6;
        // A bending-style penalty on the first four directions and a
        // null-space ridge on the last two: the default double penalty's shape.
        let mut bending = Array2::<f64>::zeros((q, q));
        let mut ridge = Array2::<f64>::zeros((q, q));
        for index in 0..q {
            if index < 4 {
                bending[[index, index]] = 1.0 + index as f64;
            } else {
                ridge[[index, index]] = 1.0;
            }
        }
        for separation in [0.0_f64, 12.0, 24.0, 42.0, 58.0] {
            let geometry = SelectionGeometry::whiten(
                &Array2::eye(q),
                &[bending.clone(), ridge.clone()],
                &[0.5 * separation, -0.5 * separation],
            )
            .expect("geometry");
            assert_eq!(
                geometry.rank, q,
                "the two components span the block, so the structural rank is q \
                 whatever the separation"
            );
            let base = geometry.at(&[0.0, 0.0]).expect("fitted point");
            for shift in [-3.0_f64, 1.5] {
                let moved = geometry.at(&[shift, shift]).expect("shifted point");
                let predicted: f64 = base
                    .eigenvalues
                    .iter()
                    .map(|&nu| (nu * shift.exp()).ln_1p() - nu.ln_1p())
                    .sum::<f64>()
                    - geometry.rank as f64 * shift;
                let change =
                    occam_offset(&moved, geometry.rank) - occam_offset(&base, geometry.rank);
                assert!(
                    (change - predicted).abs() <= 1e-8 * (1.0 + predicted.abs()),
                    "at separation {separation} a common shift of {shift} moved the \
                     criterion's offset by {change} where the closed form says {predicted} \
                     — the log-determinant has lost the scales it cannot see"
                );
            }
        }
    }

    /// The geometry has to survive a DENSE information and a DENSE penalty,
    /// which is the only shape a real fit ever presents.
    ///
    /// Every other fixture in this module hands `SelectionGeometry::whiten` an
    /// identity information and a diagonal penalty, so the congruence
    /// `Wᵀ S W` comes out EXACTLY symmetric and the self-adjoint entry point's
    /// input validation never fires. On a real fit it is a product of three
    /// dense matrices, its two triangles differ by summation order, and
    /// `strict_symmetric_eigh` — correctly — refuses rather than symmetrizing
    /// for the caller. That refusal is silent: it becomes
    /// `SmoothLrSelectionDecline::GeometryRefused`, i.e. no replay at all, on
    /// EVERY fit. Measured that way before the symmetrization was restored:
    /// `y ~ s(z) [poisson]` declined with `geometry_refused` on the first cell
    /// of the integration sweep.
    #[test]
    fn the_geometry_survives_a_dense_information_and_a_dense_penalty() {
        let q = 7;
        // A deterministic dense SPD information and two dense PSD penalties
        // with complementary-ish ranges, all built by congruence so nothing is
        // diagonal and nothing is exactly symmetric in floating point.
        let mixing = Array2::from_shape_fn((q, q), |(row, column)| {
            let a = row as f64 + 1.0;
            let b = column as f64 + 1.0;
            ((a * 0.7 + b * 1.3).sin() + 0.25 * (a * b).cos()) / (1.0 + 0.1 * a * b)
        });
        let information = mixing.dot(&mixing.t()) + Array2::<f64>::eye(q) * 0.5;
        let mut bending = Array2::<f64>::zeros((q, q));
        for index in 0..q - 2 {
            bending[[index, index]] = 1.0;
            bending[[index, index + 1]] = -0.5;
            bending[[index + 1, index]] = -0.5;
        }
        let bending = mixing.dot(&bending.dot(&mixing.t()));
        let bending = bending.dot(&bending.t());
        let ridge = mixing.dot(&mixing.t());
        for separation in [0.0_f64, 30.0, 55.0] {
            let geometry = SelectionGeometry::whiten(
                &information,
                &[bending.clone(), ridge.clone()],
                &[0.5 * separation, -0.5 * separation],
            )
            .unwrap_or_else(|reason| {
                panic!(
                    "the geometry refused a dense fit at separation {separation} ({}) — \
                     that is a silent `no replay` on every real model",
                    reason.label()
                )
            });
            let replay = SmoothLrSelectionReplay::from_geometry(
                &geometry,
                &[(-6.0, 6.0), (-6.0, 6.0)],
                512,
                512,
                None,
            );
            let replay = replay.replay().unwrap_or_else(|| {
                panic!(
                    "declined at separation {separation}: {:?}",
                    SmoothLrSelectionReplay::from_geometry(
                        &geometry,
                        &[(-6.0, 6.0), (-6.0, 6.0)],
                        512,
                        512,
                        None,
                    )
                    .decline()
                )
            });
            assert_eq!(
                replay.generalized.len(),
                geometry.dimension,
                "the published spectrum must cover the block"
            );
            assert!(
                replay
                    .generalized
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
            );
        }
    }

    /// The rank is STRUCTURAL, so an unpenalized direction contributes no
    /// log-determinant term at any scale — and a `1e-17` of roundoff on it
    /// cannot invent one.
    #[test]
    fn an_unpenalized_direction_carries_no_log_determinant_term() {
        let q = 4;
        let mut penalty = Array2::<f64>::zeros((q, q));
        for index in 0..q - 1 {
            penalty[[index, index]] = 1.0 + index as f64;
        }
        let geometry =
            SelectionGeometry::whiten(&Array2::eye(q), std::slice::from_ref(&penalty), &[0.0])
                .expect("geometry");
        assert_eq!(geometry.rank, q - 1);
        let base = geometry.at(&[0.0]).expect("fitted point");
        assert_eq!(base.eigenvalues.len(), q);
        // The last direction is unpenalized: weight one, share zero.
        let last = base.eigenvalues[q - 1];
        assert!(last.abs() < 1e-12, "expected a structural zero, got {last}");
        assert!((base.weights[q - 1] - 1.0).abs() < 1e-12);
        // And the offset moves by exactly `−rank·ln t` under a common shift, so
        // the unpenalized direction is not being counted.
        for shift in [-5.0_f64, 4.0] {
            let moved = geometry.at(&[shift]).expect("shifted point");
            let predicted: f64 = base
                .eigenvalues
                .iter()
                .map(|&nu| (nu * shift.exp()).ln_1p() - nu.ln_1p())
                .sum::<f64>()
                - geometry.rank as f64 * shift;
            let change = occam_offset(&moved, geometry.rank) - occam_offset(&base, geometry.rank);
            assert!(
                (change - predicted).abs() <= 1e-9 * (1.0 + predicted.abs()),
                "the unpenalized direction leaked a log-determinant term: {change} vs {predicted}"
            );
        }
    }

    /// The criterion's Occam offset `log|I + T| − log|T|₊` at an evaluated point,
    /// read off that point's own spectrum over the structural rank — the quantity
    /// `SelectionGeometry::at` prices from the stacked roots and refuses on when
    /// it is not finite.
    fn occam_offset(point: &super::SelectionPoint, rank: usize) -> f64 {
        let log_det_hessian: f64 = point.eigenvalues.iter().map(|value| value.ln_1p()).sum();
        let log_det_penalty: f64 = point.eigenvalues[..rank]
            .iter()
            .map(|value| value.ln())
            .sum();
        log_det_hessian - log_det_penalty
    }
}

#[cfg(test)]
mod wald_selection_tests {
    use super::{
        SmoothLrSelection, SmoothLrSelectionDecline, lr_tested_block, split_mix64,
        WaldDraws, studentized_radius_factor, wald_selection_test,
    };
    use ndarray::{Array1, Array2};

    /// A counter-based standard normal stream, independent of the replay's own.
    struct Normals(u64);

    impl Normals {
        fn next(&mut self) -> f64 {
            self.0 = split_mix64(self.0);
            gam_math::probability::standard_normal_from_uniform_bits(self.0)
                .expect("a 52-bit uniform is inside (0, 1)")
        }
    }

    /// Kolmogorov–Smirnov distance of `p` from `U(0, 1)`.
    fn ks_distance(p: &mut [f64]) -> f64 {
        p.sort_by(f64::total_cmp);
        let count = p.len() as f64;
        p.iter()
            .enumerate()
            .map(|(index, &value)| {
                (value - index as f64 / count).max((index as f64 + 1.0) / count - value)
            })
            .fold(0.0, f64::max)
    }

    /// The published p-value is `U(0, 1)` over replicates of the null: the
    /// Kolmogorov–Smirnov distance is inside its 0.1% critical value and the
    /// size at each level is within three Monte-Carlo standard errors of
    /// nominal, on BOTH sides, so a conservative law fails as surely as an
    /// anti-conservative one.
    fn assert_uniform(label: &str, p_values: &mut [f64]) {
        let replicates = p_values.len() as f64;
        for alpha in [0.01_f64, 0.05, 0.1] {
            let size = p_values.iter().filter(|&&p| p <= alpha).count() as f64 / replicates;
            let band = 3.0 * (alpha * (1.0 - alpha) / replicates).sqrt();
            assert!(
                (size - alpha).abs() <= band,
                "{label}: size {size} at alpha {alpha} is outside nominal ± {band}"
            );
        }
        let distance = ks_distance(p_values);
        let critical = 1.95 / replicates.sqrt();
        assert!(
            distance <= critical,
            "{label}: KS distance {distance} from U(0, 1) exceeds {critical}"
        );
    }

    /// One null replicate of `u`: `z` at known scale, `z/s` with
    /// `s² = χ²_ν/ν` at estimated scale.
    fn null_draw(normals: &mut Normals, dimension: usize, residual_df: Option<usize>) -> Vec<f64> {
        let draw: Vec<f64> = (0..dimension).map(|_| normals.next()).collect();
        let Some(nu) = residual_df else {
            return draw;
        };
        let chi_square: f64 = (0..nu).map(|_| normals.next().powi(2)).sum();
        let scale = (chi_square / nu as f64).sqrt();
        draw.into_iter().map(|value| value / scale).collect()
    }

    fn null_p_values(
        penalties: &[Array2<f64>],
        log_lambda: &[f64],
        windows: &[(f64, f64)],
        residual_df: Option<usize>,
        replicates: usize,
        seed: u64,
    ) -> Vec<f64> {
        let dimension = penalties[0].nrows();
        let whitener = Array2::<f64>::eye(dimension);
        let mut normals = Normals(seed);
        (0..replicates)
            .map(|_| {
                let observed = null_draw(&mut normals, dimension, residual_df);
                let test = wald_selection_test(
                    &whitener,
                    &observed,
                    penalties,
                    log_lambda,
                    windows,
                    residual_df.map(|nu| nu as f64),
                    WaldDraws {
                        diagonal: 1024,
                        multiscale: 256,
                    },
                )
                .expect("the null replicate is priced");
                assert!(
                    matches!(test.selection, SmoothLrSelection::Replayed(_)),
                    "the replicate was not replayed: {:?}",
                    test.selection.decline()
                );
                test.p_value
            })
            .collect()
    }

    /// A single-penalty smooth with a two-dimensional null space.
    fn wiggliness(dimension: usize) -> Array2<f64> {
        Array2::from_diag(&Array1::from_iter(
            (0..dimension).map(|index| if index < 2 { 0.0 } else { (index * index) as f64 }),
        ))
    }

    /// The same smooth's null-space ridge, the second penalty of a
    /// double-penalty term.
    fn ridge(dimension: usize) -> Array2<f64> {
        Array2::from_diag(&Array1::from_iter(
            (0..dimension).map(|index| if index < 2 { 1.0 } else { 0.0 }),
        ))
    }

    #[test]
    fn single_penalty_wald_p_value_is_uniform_at_known_scale() {
        let mut p = null_p_values(&[wiggliness(8)], &[0.0], &[(-12.0, 12.0)], None, 400, 11);
        assert_uniform("single penalty, known scale", &mut p);
    }

    /// A term REML shrank onto the upper face of its window: the fitted
    /// `λ̂` is ten e-folds up and one e-fold short of the wall.
    #[test]
    fn shrunk_single_penalty_wald_p_value_is_uniform_at_known_scale() {
        let mut p = null_p_values(&[wiggliness(8)], &[10.0], &[(-20.0, 1.0)], None, 400, 12);
        assert_uniform("shrunk single penalty, known scale", &mut p);
    }

    #[test]
    fn single_penalty_wald_p_value_is_uniform_at_estimated_scale() {
        let mut p = null_p_values(&[wiggliness(8)], &[0.0], &[(-12.0, 12.0)], Some(6), 400, 13);
        assert_uniform("single penalty, estimated scale", &mut p);
    }

    #[test]
    fn double_penalty_wald_p_value_is_uniform_at_known_and_estimated_scale() {
        let penalties = [wiggliness(6), ridge(6)];
        let windows = [(-10.0, 10.0), (-10.0, 6.0)];
        let mut known = null_p_values(&penalties, &[0.0, 4.0], &windows, None, 300, 14);
        assert_uniform("double penalty, known scale", &mut known);
        let mut estimated = null_p_values(&penalties, &[0.0, 4.0], &windows, Some(8), 300, 15);
        assert_uniform("double penalty, estimated scale", &mut estimated);
    }

    /// With the windows closed the statistic is the fitted Wald form
    /// `β̂ᵀB⁻¹β̂/σ̂²`, read through the tested block's own dual, and the p-value
    /// is its exact conditional tail.
    #[test]
    fn closed_window_wald_is_the_quadratic_form_of_the_estimate() {
        let dimension = 5;
        let mut normals = Normals(21);
        let mut factor = Array2::<f64>::zeros((dimension, dimension));
        factor.iter_mut().for_each(|value| *value = normals.next());
        let penalty = wiggliness(dimension);
        let lambda = 3.0_f64;
        // `H = X'X + λS`, `B = H⁻¹`.
        let hessian = factor.t().dot(&factor)
            + Array2::<f64>::eye(dimension) * 0.5
            + penalty.mapv(|value| lambda * value);
        let inverse = gam_linalg::faer_ndarray::FaerCholesky::cholesky(&hessian, faer::Side::Lower)
            .expect("positive definite")
            .solve_mat(&Array2::<f64>::eye(dimension));
        let block = lr_tested_block(
            Some(&inverse),
            Some(&penalty.mapv(|value| lambda * value)),
            &(0..dimension),
        )
        .expect("tested block");
        let dual = block.dual.expect("B is positive definite");
        let beta = Array1::from_iter((0..dimension).map(|_| normals.next()));
        let sigma = 1.3_f64;
        let observed = dual.t().dot(&beta).mapv(|value| value / sigma);
        let test = wald_selection_test(
            &block.whitener,
            observed.as_slice().expect("contiguous"),
            std::slice::from_ref(&penalty),
            &[lambda.ln()],
            &[(0.0, 0.0)],
            None,
            WaldDraws {
                diagonal: 64,
                multiscale: 64,
            },
        )
        .expect("closed window is a fact about the fit");
        assert_eq!(
            test.selection.decline(),
            Some(SmoothLrSelectionDecline::WindowClosed)
        );
        let expected = beta.dot(&hessian.dot(&beta)) / (sigma * sigma);
        assert!(
            (test.statistic - expected).abs() <= 1e-9 * expected,
            "Wald {} against β̂ᵀB⁻¹β̂/σ² {expected}",
            test.statistic
        );
        assert!(test.p_value > 0.0 && test.p_value < 1.0 && test.standard_error == 0.0);
    }

    /// The radius map sends `χ²_q` to `ν·χ²_q/χ²_ν` quantile for quantile:
    /// through the CDF of `F(q, ν)` the image of a `χ²_q` level is that level.
    #[test]
    fn studentized_radius_factor_maps_chi_square_levels_to_f_levels() {
        use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};
        let (q, nu) = (7_usize, 5.0_f64);
        let chi = ChiSquared::new(q as f64).expect("chi-square");
        let fisher = FisherSnedecor::new(q as f64, nu).expect("F");
        for level in [1e-6_f64, 0.01, 0.3, 0.5, 0.7, 0.99, 1.0 - 1e-6] {
            let radius = chi.inverse_cdf(level);
            let factor = studentized_radius_factor(radius, q, nu).expect("factor");
            let mapped = fisher.cdf(factor * radius / q as f64);
            assert!(
                (mapped - level).abs() <= 1e-8 * level.min(1.0 - level).max(1e-3),
                "level {level} mapped to {mapped}"
            );
        }
    }
}
