// The #1063 per-term smooth significance test: a genuine likelihood-ratio
// statistic from a constrained refit, its Lawley Bartlett correction, and the
// reference distribution it is scored against (#2672).
//
// Split out of `spatial_optimization.rs` under the #780 line-count gate. It is
// `include!`d into `drivers/mod.rs` alongside the driver it came from, so it
// keeps the same flat namespace and the same import surface — nothing here
// changed except which file it lives in.

/// Provenance tag for the smooth-term significance correction (#1063): which
/// statistic the reported p-value is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrCorrection {
    /// A per-term likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)` that has
    /// been Bartlett-corrected with the fixed-λ Lawley factor `c = E[W|λ]/d`
    /// (`W* = W/c`, scored against the reference law). The λ̂-sampling
    /// variation is not a second Lawley term: the reference's selection replay
    /// integrates the law over the λ̂ the fit could have chosen, and `c`
    /// rescales that integrated law.
    LawleyLrFixedLambda,
    /// No second-order correction was applied — either the family has no
    /// closed-form Lawley cumulant jets or the null refit did not converge — so
    /// the uncorrected `χ²_d` of the raw LR statistic stands.
    None,
}

impl SmoothLrCorrection {
    /// The serialized provenance label surfaced in the summary table.
    pub fn label(self) -> &'static str {
        match self {
            SmoothLrCorrection::LawleyLrFixedLambda => "lawley_lr_fixed_lambda",
            SmoothLrCorrection::None => "none",
        }
    }
}

/// Which lane supplied a [`SmoothLrReferenceDf`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrReferenceSource {
    /// The statistic's own null spectrum `w`, in full, scored by inverting its
    /// moment generating function along a saddle-point contour. This is the
    /// exact lane: the reference IS the null law, not a distribution fitted to
    /// some of its moments.
    ///
    /// The spectrum is assembled from `[H⁻¹]_jj` and the term's own λ-weighted
    /// penalty block through the symmetric similarity
    /// `w_j = 1 − eig(B^{1/2} S_jj B^{1/2})²` — see
    /// `lr_tested_block` for why that is the same spectrum as
    /// `eig(2·F_jj − F_jj²)` and why it is the better-conditioned way to reach
    /// it.
    NullSpectrum,
    /// `[H⁻¹]_jj` or the penalty block was unavailable, but the
    /// coefficient-influence matrix was, so only the first two *moments* of the
    /// spectrum are recoverable (`tr A` and `tr A²` for `A = 2F_jj − F_jj²`,
    /// both traces of powers of one block). The reference is then the
    /// two-moment match `g·χ²_ν`.
    ///
    /// It is EXACT at both ends of the shrinkage range and wrong in between,
    /// which is worth stating precisely because the ends are where the intuition
    /// goes. An unpenalized term has `w ≡ 1` and the match is the textbook
    /// `χ²_q`; a term REML has shrunk to its null space has one weight of order
    /// one over a tail of dust — measured on a null-true `k = 12` fit, `w =
    /// (0.322, 5.9e-7, 7.1e-8, …)` — and a single distinct weight is a scaled
    /// chi-square exactly. The gap opens at moderate shrinkage, where several
    /// weights are comparable and unequal: on `f_j = 1/(1 + λγ_j)` for a
    /// second-difference penalty at `λ = 0.01`, `k = 20`, the size delivered at
    /// a nominal `α` is `1.02×` at `0.05`, `1.11×` at `0.01`, `1.31×` at `10⁻³`
    /// and `1.61×` at `10⁻⁴` — one-signed, anti-conservative, and worse the
    /// deeper the tail.
    ///
    /// It is a surrogate for the lane above, not a different claim about the
    /// statistic.
    SpectralMomentMatch,
    /// Neither the spectrum nor its moments were recoverable, so the reference
    /// falls back to the classical unit-weight shape
    /// `χ²_{max(edf, null_dim, 1)}` — every retained direction counted as if it
    /// were unpenalized. It is the only reference recoverable from a scalar
    /// EDF, and it is conservative for the same reason the whole pre-#2672
    /// assembly was: unit weights over-state the statistic's spread.
    UnitWeightFallback,
}

/// The null law of `W(λ̂)` when `λ̂` is CHOSEN by the outer criterion rather than
/// given — the reference the whole-term LR statistic actually needs (#2672).
///
/// # The defect this exists for
///
/// [`SmoothLrReferenceDf`]'s spectrum is the exact null law of `W` *at a fixed*
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
    /// On a PROFILED-scale replay, each draw's weight-one residual `χ²_h`, in
    /// draw order: the same variate drove that draw's selection (through the
    /// profiled deviance) and is the weight-one block of its `V`, so the tail
    /// shift reads it back rather than integrating over a fresh one.
    /// `None` when the selection did not profile a scale.
    residual_unit_sample: Option<Vec<f64>>,
    /// Stratified coordinates the replay consumed, so draws that are to be
    /// independent of it start after them.
    consumed_coordinates: usize,
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
    ///
    /// On a profiled replay the observation is selected by the profiled
    /// criterion too, over its OWN residual deviance ([`ObservedDraw`]), so
    /// that it is chosen by exactly the rule each draw is.
    observed: Option<ObservedSelection>,
}

/// `W_q` of the observed whitened score, at `t = 1` and at the replay's
/// selected `t*` — see [`SmoothLrSelectionReplay::observed`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct ObservedSelection {
    conditional: f64,
    selected: f64,
}

/// The observation as the replay scores it: its whitened block score, and on a
/// profiled replay the full fit's penalized deviance `D_p(λ̂)` in the same
/// units as the score's squares.
///
/// A profiled draw selects over `m·ln(R + D(t))`, with `R` the part of the
/// penalized deviance the tested scale does not reach. The observation's `R`
/// is not drawn — it is the fit's own: along the tested scale the penalized
/// deviance is `R + Σ_j c_j² s_j(t)` in the fitted eigenbasis, so
/// `R = D_p(λ̂) − Σ_j c_j² s_j(1)`, read off the fit at `t = 1`.
#[derive(Clone, Copy, Debug)]
struct ObservedDraw<'a> {
    whitened: &'a [f64],
    penalized_deviance: Option<f64>,
}

impl std::fmt::Debug for SmoothLrSelectionReplay {
    /// The two samples are thousands of draws each and are never what a reader
    /// of a failure message wants; the spectrum that generated them is.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SmoothLrSelectionReplay")
            .field("generalized", &self.generalized)
            .field("draws", &self.selection_sample.len())
            .field("profiled", &self.residual_unit_sample.is_some())
            .field("observed", &self.observed)
            .finish()
    }
}

impl PartialEq for SmoothLrSelectionReplay {
    fn eq(&self, other: &Self) -> bool {
        self.generalized == other.generalized
            && self.selection_sample == other.selection_sample
            && self.conditional_sample == other.conditional_sample
            && self.residual_unit_sample == other.residual_unit_sample
            && self.consumed_coordinates == other.consumed_coordinates
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
/// a sweep of them, each over an [`AxisSlice`] whose diagonalizing basis has to
/// be re-derived at the draw's current point. The standard error it leaves is
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
    /// `rank(Σ_{j≠i} Wᵀ S_j W)` for each component `i`, from the same UNIT
    /// stacked roots and the same bar as `rank`. The other `rank −
    /// complement_rank[i]` directions of `range(T)` are reached by component `i`
    /// alone, and an [`AxisSlice`] prices them as exact linear terms rather than
    /// asking a floating-point cosine whether it is zero.
    complement_rank: Vec<usize>,
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

impl SelectionPoint {
    /// The observation's profiled floor at this point: its penalized deviance
    /// less the tested block's part of it, `R = D_p − Σ_j c_j²·e_j/(1 + e_j)`,
    /// with `c` the whitened score in this point's eigenbasis (see
    /// [`ObservedDraw`]). `None` unless that leaves a positive deviance — `R`
    /// holds the residual sum of squares, so anything else is a score and a
    /// deviance that do not describe one fit.
    fn observed_floor(&self, whitened: &[f64], penalized_deviance: f64) -> Option<f64> {
        let mut reached = 0.0_f64;
        for (column, &eigenvalue) in self.eigenvalues.iter().enumerate() {
            let projection: f64 = whitened
                .iter()
                .enumerate()
                .map(|(row, value)| value * self.basis[[row, column]])
                .sum();
            reached += projection * projection * eigenvalue / (1.0 + eigenvalue);
        }
        let floor = penalized_deviance - reached;
        (floor.is_finite() && floor > 0.0).then_some(floor)
    }
}

impl SelectionGeometry {
    /// Whiten the term's λ-free penalty components by the Schur-complemented
    /// information and factor each into its own root.
    ///
    /// Returns `None` when the information has no identified direction, a
    /// decomposition refuses, or the components leave nothing penalized —
    /// in every case the caller has nothing to replay.
    fn whiten(
        whitener: &Array2<f64>,
        unit_penalties: &[Array2<f64>],
        log_lambda: &[f64],
    ) -> Option<Self> {
        if unit_penalties.is_empty() || unit_penalties.len() != log_lambda.len() {
            return None;
        }
        let dimension = whitener.ncols();
        if dimension == 0 || whitener.nrows() == 0 {
            return None;
        }

        let mut roots = Vec::with_capacity(unit_penalties.len());
        for penalty in unit_penalties {
            if penalty.nrows() != whitener.nrows() || penalty.ncols() != whitener.nrows() {
                return None;
            }
            // `Wᵀ S W` is symmetric as a mathematical object and its two
            // triangles differ only by summation order — but
            // `strict_symmetric_eigh` REFUSES the input rather than symmetrizing
            // it for the caller, which is the right contract and the reason this
            // is explicit here. Dropping it is a silent `GeometryRefused` on
            // every real fit.
            let whitened = symmetrized(whitener.t().dot(penalty).dot(whitener));
            if whitened.iter().any(|value| !value.is_finite()) {
                return None;
            }
            roots.push(psd_root(&whitened)?);
        }
        if roots.iter().all(|root| root.nrows() == 0) {
            return None;
        }
        let stacked_rows = roots.iter().map(|root| root.nrows()).sum::<usize>();
        // `range(Σ_i S̃_i)` is what `log|T|₊` runs over, and it is `t`-free. Taken
        // from the UNIT stacked roots, whose singular values span only the
        // components' own conditioning — the λ ratio that makes the assembled
        // sum unreadable is not present here at all.
        let unit = stack_roots(&roots, &vec![0.0; roots.len()], stacked_rows.max(dimension));
        let (_, unit_singular, unit_right) =
            gam_linalg::faer_ndarray::FaerSvd::svd(&unit, false, true).ok()?;
        let unit_largest = unit_singular.iter().copied().fold(0.0_f64, f64::max);
        let bar = unit_largest * (dimension as f64) * f64::EPSILON * 100.0;
        let rank = unit_singular.iter().filter(|&&value| value > bar).count();
        if rank == 0 {
            return None;
        }
        // The same decomposition that decided the rank also names the subspace:
        // the leading `rank` right singular vectors of the UNIT stack span
        // `range(Σ_i S̃_i)`. Deciding the two from one object is what stops them
        // disagreeing about which directions are structural.
        let unit_right = unit_right?;
        if unit_right.nrows() < rank || unit_right.ncols() != dimension {
            return None;
        }
        let mut range_basis = Array2::<f64>::zeros((dimension, rank));
        for column in 0..rank {
            for row in 0..dimension {
                range_basis[[row, column]] = unit_right[[column, row]];
            }
        }
        let range_roots: Vec<Array2<f64>> =
            roots.iter().map(|root| root.dot(&range_basis)).collect();
        // Which directions a component reaches alone is structural, so it is read
        // off the λ-free roots once, at the bar that decided `rank`.
        let mut complement_rank = Vec::with_capacity(roots.len());
        for axis in 0..roots.len() {
            let others: Vec<Array2<f64>> = roots
                .iter()
                .enumerate()
                .filter(|&(component, _)| component != axis)
                .map(|(_, root)| root.clone())
                .collect();
            let rows = others.iter().map(|root| root.nrows()).sum::<usize>();
            if rows == 0 {
                complement_rank.push(0);
                continue;
            }
            let complement = stack_roots(&others, &vec![0.0; others.len()], rows.max(dimension));
            let (_, singular, _) =
                gam_linalg::faer_ndarray::FaerSvd::svd(&complement, false, false).ok()?;
            complement_rank.push(singular.iter().filter(|&&value| value > bar).count());
        }
        Some(Self {
            roots,
            log_lambda: log_lambda.to_vec(),
            dimension,
            rank,
            stacked_rows: stacked_rows.max(dimension),
            range_basis,
            range_roots,
            complement_rank,
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
fn symmetrized(mut matrix: Array2<f64>) -> Array2<f64> {
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
    /// The whitening or a component's root refused: `Ĩ_jj` has no identified
    /// direction, a decomposition failed, or the components span nothing.
    GeometryRefused,
    /// Every scale's window is closed — the fit is railed against both walls of
    /// the solver's `ρ` box at once, so there was no `λ` it could have chosen
    /// instead.
    WindowClosed,
    /// The fitted point `ln t = 0` could not be evaluated, so there is no
    /// conditional arm to pair the selection with. Refused whole rather than
    /// sampled partial.
    GridRefused,
    /// A draw's certified global minimization of its criterion could not resolve
    /// the criterion's stationary structure, the criterion was not evaluable
    /// inside the window, or an axis slice of it could not be priced, so that
    /// draw has no selection. Refused whole rather than sampled partial.
    SelectionUnresolved,
    /// The observation's own block score was supplied but is not a finite
    /// vector on the tested block, so the observed statistic cannot be scored
    /// by the replay's selection rule — and a replay whose observation is
    /// selected by a different rule than its draws is not a reference for it.
    ObservedScoreUnusable,
    /// The family profiles its scale, but the model's penalized rank is smaller
    /// than the tested term's own, so the residual deviance the profiled
    /// selection is driven by has no consistent law to draw from.
    ProfileInconsistent,
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
            SmoothLrSelectionDecline::ProfileInconsistent => "profile_inconsistent",
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
/// direction: an unpenalized one has `ν = 0` and carries `ln 1 = 0`). On an
/// [`AxisSlice`] only one scale moves, and `μ` carries the directions that scale
/// shares with the others through `log|T(u)|₊`. With `g = s(1 − s)` for either
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
///
/// # The profiled-scale form
///
/// A family whose scale is PROFILED does not select with `C`. Its REML cost is
/// `½[m·ln D_p(u) + log|I + T(u)| − log|T(u)|₊]`, with `D_p` the penalized
/// residual deviance and `m = n − M_p` the positive-weight rows less the
/// balanced penalty's structural null space (the solver's own profiled
/// residual degrees of freedom). Along the replay `D_p(u) = F + D(u)`,
/// `D = Σ_j c_j² s_j`, where the floor `F` is every part of the deviance the
/// moving scale does not reach. So with [`ProfiledFloor`] set the data term is
/// `m·ln(F + D)` in place of `D`, and with `a = F + D`
///
/// ```text
/// (m ln a)′ = m·D′/a,
/// (m ln a)″ = m·(D″/a − (D′/a)²),
/// (m ln a)‴ = m·(D‴/a − 3·D′D″/a² + 2·(D′/a)³),
/// ```
///
/// where `D′, D″, D‴` are the data sums above. `D` increases with `u`, so over a
/// cell `a` is enclosed by its endpoint values and every ratio by interval
/// division by that positive range.
struct DiagonalCriterion<'a> {
    squares: &'a [f64],
    generalized: &'a [f64],
    rank: usize,
    occam: &'a [f64],
    constant: f64,
    profile: Option<ProfiledFloor>,
}

/// The profiled-scale data term `m·ln(floor + Σ_j c_j² s_j)` of a
/// [`DiagonalCriterion`]: `multiplier` is `m`, `floor` the deviance the moving
/// scale does not reach, in the same units as the squares.
#[derive(Clone, Copy, Debug)]
struct ProfiledFloor {
    multiplier: f64,
    floor: f64,
}

/// Range of `f` over the share interval `[lo, hi]`, from its endpoints and the
/// listed stationary points that fall inside it.
fn share_polynomial_range(lo: f64, hi: f64, f: impl Fn(f64) -> f64, stationary: &[f64]) -> (f64, f64) {
    let (at_lo, at_hi) = (f(lo), f(hi));
    let mut range = (at_lo.min(at_hi), at_lo.max(at_hi));
    for &point in stationary {
        if lo <= point && point <= hi {
            let value = f(point);
            range = (range.0.min(value), range.1.max(value));
        }
    }
    range
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
        // The data sums `D, D′, D″, D‴`, kept apart from the log-determinant
        // terms so the profiled form can take their logarithm.
        let (mut data, mut data_first, mut data_second, mut data_third) =
            (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut data_magnitude = 0.0_f64;
        for (&square, &nu) in self.squares.iter().zip(self.generalized.iter()) {
            let scaled = t * nu;
            if !scaled.is_finite() {
                return None;
            }
            let share = scaled / (1.0 + scaled);
            let spread = share * (1.0 - share);
            let log_term = scaled.ln_1p();
            value += log_term;
            magnitude += log_term.abs();
            first += share;
            second += spread;
            third += spread * (1.0 - 2.0 * share);
            data += square * share;
            data_magnitude += (square * share).abs();
            data_first += square * spread;
            data_second += square * spread * (1.0 - 2.0 * share);
            data_third += square * spread * (1.0 - 6.0 * share + 6.0 * share * share);
        }
        for &mu in self.occam {
            let scaled = t * mu;
            if !scaled.is_finite() {
                return None;
            }
            let share = scaled / (1.0 + scaled);
            let spread = share * (1.0 - share);
            let log_term = scaled.ln_1p();
            value -= log_term;
            magnitude += log_term.abs();
            first -= share;
            second -= spread;
            third -= spread * (1.0 - 2.0 * share);
        }
        let terms = 4 * self.squares.len() + 3 * self.occam.len() + 2;
        let Some(profile) = self.profile else {
            value += data;
            magnitude += data_magnitude;
            first += data_first;
            second += data_second;
            third += data_third;
            let band = gam_linalg::roundoff::accumulation_band(terms, magnitude);
            return Some(([value, first, second, third], band));
        };
        let level = profile.floor + data;
        if !(level > 0.0 && level.is_finite()) {
            return None;
        }
        let multiplier = profile.multiplier;
        let (slope, bend, twist) = (data_first / level, data_second / level, data_third / level);
        let log_level = level.ln();
        value += multiplier * log_level;
        first += multiplier * slope;
        second += multiplier * (bend - slope * slope);
        third += multiplier * (twist - 3.0 * slope * bend + 2.0 * slope * slope * slope);
        // `level` carries the accumulation error of its `len + 1` summands, which
        // the logarithm turns into an absolute error of that over `level`; the
        // product with `m` and the final addition carry one rounding each.
        let level_band = gam_linalg::roundoff::accumulation_band(
            self.squares.len() + 1,
            profile.floor.abs() + data_magnitude,
        );
        let band = gam_linalg::roundoff::accumulation_band(terms, magnitude)
            + multiplier * level_band / level
            + gam_linalg::roundoff::accumulation_band(2, (multiplier * log_level).abs());
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
        let spread = |s: f64| s * (1.0 - s);
        let skew = |s: f64| s * (1.0 - s) * (1.0 - 2.0 * s);
        let torsion = |s: f64| s * (1.0 - s) * (1.0 - 6.0 * s + 6.0 * s * s);
        // The data sums `D` and `D′, D″, D‴`, enclosed apart from the
        // log-determinant terms so the profiled form can divide them by `F + D`.
        let (mut data_lo, mut data_hi) = (0.0_f64, 0.0_f64);
        let (mut data_first_lo, mut data_first_hi) = (0.0_f64, 0.0_f64);
        let (mut data_second_lo, mut data_second_hi) = (0.0_f64, 0.0_f64);
        let (mut data_third_lo, mut data_third_hi) = (0.0_f64, 0.0_f64);
        let (mut data_first_magnitude, mut data_second_magnitude, mut data_third_magnitude) =
            (0.0_f64, 0.0_f64, 0.0_f64);
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
            let share_lo = scaled_a / (1.0 + scaled_a);
            let share_hi = scaled_b / (1.0 + scaled_b);
            let (spread_lo, spread_hi) =
                share_polynomial_range(share_lo, share_hi, spread, &spread_stationary);
            let (skew_lo, skew_hi) =
                share_polynomial_range(share_lo, share_hi, skew, &skew_stationary);
            let (torsion_lo, torsion_hi) =
                share_polynomial_range(share_lo, share_hi, torsion, &torsion_stationary);
            data_lo += square * share_lo;
            data_hi += square * share_hi;
            data_first_lo += square * spread_lo;
            data_first_hi += square * spread_hi;
            data_second_lo += square * skew_lo;
            data_second_hi += square * skew_hi;
            data_third_lo += square * torsion_lo;
            data_third_hi += square * torsion_hi;
            if !nu_paired[index] {
                first_lo += share_lo;
                first_hi += share_hi;
                second_lo += spread_lo;
                second_hi += spread_hi;
                third_lo += skew_lo;
                third_hi += skew_hi;
            }
            data_first_magnitude += square * spread_hi.abs().max(spread_lo.abs());
            data_second_magnitude += square * skew_hi.abs().max(skew_lo.abs());
            data_third_magnitude += square * torsion_hi.abs().max(torsion_lo.abs());
            first_magnitude += share_hi.abs();
            second_magnitude += spread_hi.abs();
            third_magnitude += skew_hi.abs().max(skew_lo.abs());
        }
        // The Occam spectrum enters with a MINUS sign, so an unpaired share range
        // subtracts crosswise: its largest share lowers the derivative's floor.
        for (index, &mu) in self.occam.iter().enumerate() {
            let (scaled_a, scaled_b) = (t_a * mu, t_b * mu);
            if !(scaled_a.is_finite() && scaled_b.is_finite()) {
                return None;
            }
            let share_lo = scaled_a / (1.0 + scaled_a);
            let share_hi = scaled_b / (1.0 + scaled_b);
            let (spread_lo, spread_hi) =
                share_polynomial_range(share_lo, share_hi, spread, &spread_stationary);
            let (skew_lo, skew_hi) =
                share_polynomial_range(share_lo, share_hi, skew, &skew_stationary);
            if !mu_paired[index] {
                first_lo -= share_hi;
                first_hi -= share_lo;
                second_lo -= spread_hi;
                second_hi -= spread_lo;
                third_lo -= skew_hi;
                third_hi -= skew_lo;
            }
            first_magnitude += share_hi.abs();
            second_magnitude += spread_hi.abs().max(spread_lo.abs());
            third_magnitude += skew_hi.abs().max(skew_lo.abs());
        }
        let share = |scaled: f64| scaled / (1.0 + scaled);
        for index in 0..pairs {
            let (nu, mu) = (self.generalized[nu_order[index]], self.occam[mu_order[index]]);
            let (nu_lo, nu_hi) = (share(t_a * nu), share(t_b * nu));
            let (mu_lo, mu_hi) = (share(t_a * mu), share(t_b * mu));
            let (nu_spread_lo, nu_spread_hi) =
                share_polynomial_range(nu_lo, nu_hi, spread, &spread_stationary);
            let (mu_spread_lo, mu_spread_hi) =
                share_polynomial_range(mu_lo, mu_hi, spread, &spread_stationary);
            let (nu_skew_lo, nu_skew_hi) = share_polynomial_range(nu_lo, nu_hi, skew, &skew_stationary);
            let (mu_skew_lo, mu_skew_hi) = share_polynomial_range(mu_lo, mu_hi, skew, &skew_stationary);
            let gap = nu.ln() - mu.ln();
            let (span_lo, span_hi) = (share(t_a * nu.min(mu)), share(t_b * nu.max(mu)));
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
        let mut terms = 5 * self.squares.len() + 4 * self.occam.len() + 6 * pairs + 1;
        match self.profile {
            None => {
                first_lo += data_first_lo;
                first_hi += data_first_hi;
                second_lo += data_second_lo;
                second_hi += data_second_hi;
                third_lo += data_third_lo;
                third_hi += data_third_hi;
                first_magnitude += data_first_magnitude;
                second_magnitude += data_second_magnitude;
                third_magnitude += data_third_magnitude;
            }
            Some(profile) => {
                // `a = F + D` over the cell, and the ratios `x = D′/a ≥ 0`,
                // `y = D″/a`, `z = D‴/a` by division by that positive range.
                let (level_lo, level_hi) = (profile.floor + data_lo, profile.floor + data_hi);
                if !(level_lo > 0.0 && level_hi.is_finite()) {
                    return None;
                }
                let divide = |lo: f64, hi: f64| {
                    (
                        (lo / level_lo).min(lo / level_hi),
                        (hi / level_lo).max(hi / level_hi),
                    )
                };
                let (slope_lo, slope_hi) = (
                    data_first_lo.max(0.0) / level_hi,
                    data_first_hi.max(0.0) / level_lo,
                );
                let (bend_lo, bend_hi) = divide(data_second_lo, data_second_hi);
                let (twist_lo, twist_hi) = divide(data_third_lo, data_third_hi);
                // `x·y` with `x ≥ 0` is extremal at the ends of `x`.
                let cross_lo = (slope_lo * bend_lo).min(slope_hi * bend_lo);
                let cross_hi = (slope_lo * bend_hi).max(slope_hi * bend_hi);
                let multiplier = profile.multiplier;
                first_lo += multiplier * slope_lo;
                first_hi += multiplier * slope_hi;
                second_lo += multiplier * (bend_lo - slope_hi * slope_hi);
                second_hi += multiplier * (bend_hi - slope_lo * slope_lo);
                third_lo += multiplier
                    * (twist_lo - 3.0 * cross_hi + 2.0 * slope_lo * slope_lo * slope_lo);
                third_hi += multiplier
                    * (twist_hi - 3.0 * cross_lo + 2.0 * slope_hi * slope_hi * slope_hi);
                // Relative to the magnitudes they are built from, the ratios
                // carry the errors of both the sums and `a`, `len + 1` terms each,
                // and the products and combination a handful more.
                let bend_magnitude = bend_lo.abs().max(bend_hi.abs());
                let twist_magnitude = twist_lo.abs().max(twist_hi.abs());
                let cross_magnitude = cross_lo.abs().max(cross_hi.abs());
                first_magnitude += multiplier * slope_hi;
                second_magnitude += multiplier * (bend_magnitude + slope_hi * slope_hi);
                third_magnitude += multiplier
                    * (twist_magnitude + 3.0 * cross_magnitude + 2.0 * slope_hi * slope_hi * slope_hi);
                terms += 2 * self.squares.len() + 8;
            }
        }
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

/// One draw's multi-scale criterion along ONE scale's coordinate, with every
/// other scale held at the draw's current point, in the diagonal form
/// [`DiagonalCriterion`] certifies.
///
/// # The slice is exact, not a local model
///
/// In the range basis, with `u` the moving coordinate,
/// `C(u) = B + e^u A`, `A = λ̂_i Uᵀ S̃_i U` and `B` the other scales at their
/// current `t_j`. The criterion is `vᵀDv + log|I + C| − log|C|` with
/// `D = (I + C)⁻¹C` (see [`SelectionFactor`]), and each half diagonalizes in `u`:
///
/// * **Data and `log|I + C|`.** `I + B = LLᵀ` and `L⁻¹AL⁻ᵀ = V diag(ν) Vᵀ`, so
///   `I + C(u) = LV(I + e^u ν)VᵀLᵀ`. With `h = VᵀL⁻¹v`,
///   `vᵀDv = const + Σ_j h_j² s_j(u)` and
///   `log|I + C(u)| = log|I + B| + Σ_j ln(1 + e^u ν_j)`. `B` does not move along
///   the slice, so the absolute error of an assembled `I + B` is common to every
///   value the slice returns and cancels from each change it is asked about. So
///   `I + B` is assembled and `ν` is read off the singular values of
///   `L⁻¹(A's root)ᵀ`.
/// * **`log|C|₊`.** Here the conditioning is the whole problem (#2644), so it is
///   taken from the stacked scaled ROOTS at the current point `x = u_i`,
///   `M = [B's roots; e^{x/2}·A's root] = QR`, never from an assembled sum. With
///   `Q = [Q_B; Q_A]` orthonormal, `C(u) = Rᵀ(Q_BᵀQ_B + e^{u−x} Q_AᵀQ_A)R`, and
///   `Q_BᵀQ_B + Q_AᵀQ_A = I` gives the two blocks common eigenvectors with
///   eigenvalues `c_k²` and `s_k² = 1 − c_k²` — the cosine–sine decomposition of
///   `Q`. So `log|C(u)| = log|C(x)| + Σ_k ln(c_k² + e^{u−x} s_k²)`: a direction
///   with `c = 0` only `A` reaches and contributes `u` exactly (it is counted into
///   `r`), one with `s = 0` contributes nothing, and every other one is
///   `ln c_k² + ln(1 + e^u μ_k)` with `μ_k = e^{−x} s_k²/c_k²`.
///
/// `c_k` and `s_k` are the singular values of their own blocks, paired by order
/// (cosines ascending against sines descending), and of each pair the SMALLER
/// member is the one read from its block: a singular value carries an absolute
/// error of `ε‖Q‖ = ε`, so the small member keeps digits that `1 − (large)²` would
/// not. Which directions have `c = 0` is not decided by that arithmetic: it is
/// `rank − complement_rank[i]`, read once per term off the λ-free roots.
struct AxisSlice {
    /// `h_j²`, the draw's squared coordinates in `A`'s basis against `I + B`.
    squares: Vec<f64>,
    /// `ν_j`, the spectrum of `A` against `I + B`.
    generalized: Vec<f64>,
    /// Directions of `range(C)` only `A` reaches.
    rank: usize,
    /// `μ_k`, the directions `A` shares with `B`.
    occam: Vec<f64>,
    /// `‖v‖² − ‖L⁻¹v‖²`, the part of the draw's data term `vᵀDv` this scale
    /// does not move: `vᵀDv = ‖v‖² − vᵀ(I + C)⁻¹v = ‖v‖² − ‖L⁻¹v‖² + Σ_j h_j² s_j`.
    /// Only the profiled criterion reads it, as part of its floor.
    unreached: f64,
}

impl AxisSlice {
    /// The slice of the criterion along scale `axis` through `log_t`, for the draw
    /// whose coordinates in the range basis are `coordinates`. `None` when a
    /// factorization refuses or a scale overflows: the point cannot be priced.
    fn new(
        geometry: &SelectionGeometry,
        log_t: &[f64],
        axis: usize,
        coordinates: &[f64],
    ) -> Option<Self> {
        let rank = geometry.rank;
        if log_t.len() != geometry.range_roots.len()
            || axis >= log_t.len()
            || coordinates.len() != rank
        {
            return None;
        }
        let mut hessian = Array2::<f64>::eye(rank);
        let mut other_rows = 0usize;
        for (component, root) in geometry.range_roots.iter().enumerate() {
            if component == axis {
                continue;
            }
            let scale = (geometry.log_lambda[component] + log_t[component]).exp();
            if !scale.is_finite() {
                return None;
            }
            hessian.scaled_add(scale, &root.t().dot(root));
            other_rows += root.nrows();
        }
        let lower = gam_linalg::triangular::cholesky_factor_in_place(
            hessian.view(),
            gam_linalg::triangular::CholeskyGuard::FiniteStrict,
        )?;
        let own_scale = (0.5 * geometry.log_lambda[axis]).exp();
        if !own_scale.is_finite() {
            return None;
        }
        let own = geometry.range_roots[axis].mapv(|value| own_scale * value);
        let own_rows = own.nrows();
        let mut squares = Vec::with_capacity(own_rows);
        let mut generalized = Vec::with_capacity(own_rows);
        let mapped = gam_linalg::triangular::forward_substitution_lower_vector(&lower, coordinates);
        // `(I + B)⁻¹ ≼ I`, so the exact difference is nonnegative; only its
        // rounding can take it below zero.
        let unreached = (coordinates.iter().map(|value| value * value).sum::<f64>()
            - mapped.iter().map(|value| value * value).sum::<f64>())
        .max(0.0);
        if own_rows > 0 {
            let whitened =
                gam_linalg::triangular::forward_substitution_lower_matrix(&lower, own.t());
            let (left, singular, _) =
                gam_linalg::faer_ndarray::FaerSvd::svd(&whitened, true, false).ok()?;
            let left = left?;
            for (mode, &sigma) in singular.iter().enumerate() {
                if !(sigma.is_finite() && sigma >= 0.0) {
                    return None;
                }
                let projection = left.column(mode).dot(&mapped);
                squares.push(projection * projection);
                generalized.push(sigma * sigma);
            }
        }

        let stacked_rows = other_rows + own_rows;
        if stacked_rows < rank {
            return None;
        }
        let mut stacked = Array2::<f64>::zeros((stacked_rows, rank));
        let mut offset = 0usize;
        for (component, root) in geometry.range_roots.iter().enumerate() {
            if component == axis {
                continue;
            }
            let scale = (0.5 * (geometry.log_lambda[component] + log_t[component])).exp();
            stacked
                .slice_mut(ndarray::s![offset..offset + root.nrows(), ..])
                .assign(&root.mapv(|value| scale * value));
            offset += root.nrows();
        }
        let current = (0.5 * log_t[axis]).exp();
        if !current.is_finite() {
            return None;
        }
        stacked
            .slice_mut(ndarray::s![offset.., ..])
            .assign(&own.mapv(|value| current * value));
        let mut diagonal = vec![0.0_f64; rank];
        householder_triangularize(&mut stacked, &mut diagonal)?;
        let orthonormal = householder_thin_q(&stacked);
        let mut sines = if own_rows > 0 {
            gam_linalg::faer_ndarray::FaerSvd::svd(
                &orthonormal.slice(ndarray::s![other_rows.., ..]),
                false,
                false,
            )
            .ok()?
            .1
            .to_vec()
        } else {
            Vec::new()
        };
        let mut cosines = if other_rows > 0 {
            gam_linalg::faer_ndarray::FaerSvd::svd(
                &orthonormal.slice(ndarray::s![..other_rows, ..]),
                false,
                false,
            )
            .ok()?
            .1
            .to_vec()
        } else {
            Vec::new()
        };
        sines.sort_by(|a, b| b.total_cmp(a));
        sines.resize(rank, 0.0);
        cosines.sort_by(|a, b| a.total_cmp(b));
        let mut paired_cosines = vec![0.0_f64; rank.saturating_sub(cosines.len())];
        paired_cosines.extend(cosines);

        let reached_alone = rank.checked_sub(geometry.complement_rank[axis])?;
        let mut linear = reached_alone;
        let mut occam = Vec::with_capacity(rank - reached_alone);
        let inverse_current = (-log_t[axis]).exp();
        for mode in reached_alone..rank {
            let (sine, cosine) = (sines[mode], paired_cosines[mode]);
            let (sine_squared, cosine_squared) = if sine <= cosine {
                (sine * sine, 1.0 - sine * sine)
            } else {
                (1.0 - cosine * cosine, cosine * cosine)
            };
            if !(sine_squared > 0.0) {
                continue;
            }
            if !(cosine_squared > 0.0) {
                linear += 1;
                continue;
            }
            let scaled = inverse_current * sine_squared / cosine_squared;
            if !scaled.is_finite() {
                return None;
            }
            occam.push(scaled);
        }
        Some(Self {
            squares,
            generalized,
            rank: linear,
            occam,
            unreached,
        })
    }

    /// The slice's criterion; `residual` is the draw's profiled residual
    /// `(m, R)` when the family profiles its scale, and the slice's floor is then
    /// `R` plus the part of the data term this scale does not reach.
    fn criterion(&self, residual: Option<(f64, f64)>) -> DiagonalCriterion<'_> {
        DiagonalCriterion {
            squares: &self.squares,
            generalized: &self.generalized,
            rank: self.rank,
            occam: &self.occam,
            constant: 0.0,
            profile: residual.map(|(multiplier, residual)| ProfiledFloor {
                multiplier,
                floor: residual + self.unreached,
            }),
        }
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

/// What a PROFILED-scale family's selection is driven by beyond the tested
/// block: the dimensions of the residual deviance its REML criterion takes the
/// logarithm of.
///
/// A family whose scale is profiled selects `λ` by minimizing
/// `½[m·ln D_p + log|I + T| − log|T|₊]` ([`DiagonalCriterion`]), so a draw's
/// selection depends on the WHOLE penalized deviance `D_p`, not only on the
/// tested block's part of it. Under the null the rest of `D_p` is, in units of
/// the scale,
///
/// ```text
///   R = χ²_h + χ²_{r − r_j},     m = h + r,
/// ```
///
/// with `h` the residual directions no column reaches, `r` the model's
/// penalized rank and `r_j` the tested term's. The `χ²_h` is the residual sum
/// of squares' weight-one block — the SAME variate that is the weight-one block
/// of the statistic's `V` ([`SmoothLrProfiledScale`]) — and `χ²_{r − r_j}` is
/// the other penalized directions' share: a penalized direction's
/// `y²·s = χ²_1` exactly under the Gaussian law REML's own fixed point
/// `E[D_p] = m·φ` is the mean of. With a known scale the selection sees none of
/// this and the replay needs no profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SmoothLrSelectionProfile {
    /// `h`, the residual directions no column reaches.
    pub(crate) residual_unit_dimension: usize,
    /// `r`, the structural rank of the model's balanced penalty.
    pub(crate) penalized_rank: usize,
}

/// A profiled replay's per-draw residual deviance, in draw order.
struct ProfiledResidualDraws {
    /// `m = h + r`.
    multiplier: f64,
    /// The weight-one `χ²_h`, shared with the statistic's `V`.
    unit: Vec<f64>,
    /// `R = χ²_h + χ²_{r − r_j}`, the floor of the draw's profiled deviance.
    residual: Vec<f64>,
}

impl SmoothLrSelectionProfile {
    /// Stratified draws of `R`, on the two coordinates after the replay's
    /// `dimension`: `χ²_h` on coordinate `dimension`, `χ²_{r − r_j}` on
    /// `dimension + 1`. Declines when the tested term's rank exceeds the
    /// model's, which leaves `χ²_{r − r_j}` undefined.
    fn residual_draws(
        self,
        geometry: &SelectionGeometry,
        draws: usize,
    ) -> Result<ProfiledResidualDraws, SmoothLrSelectionDecline> {
        let other = self
            .penalized_rank
            .checked_sub(geometry.rank)
            .ok_or(SmoothLrSelectionDecline::ProfileInconsistent)?;
        let unit = stratified_chi_square(self.residual_unit_dimension, geometry.dimension, draws);
        let other = stratified_chi_square(other, geometry.dimension + 1, draws);
        let residual = unit.iter().zip(other.iter()).map(|(unit, other)| unit + other).collect();
        Ok(ProfiledResidualDraws {
            multiplier: (self.residual_unit_dimension + self.penalized_rank) as f64,
            unit,
            residual,
        })
    }
}

impl SmoothLrSelectionReplay {
    /// Generate the replay for one term from its whitened penalty geometry and
    /// the window of `ln t` the fit's own `ρ` box leaves open around the fitted
    /// point — ONE window per scale, because the outer search moved each `ρ_i`
    /// independently inside that box.
    ///
    /// Declines — with a reason — when the term has no penalized direction
    /// (nothing to select), the geometry could not be whitened, or every window
    /// is empty (the fit is railed against both walls), in which case the
    /// conditional law IS the selection law and the caller should use it
    /// unmodified.
    ///
    /// `observed_score` is the tested block's score at the nested null fit, in
    /// the units of the unscaled information the whitener was built from (see
    /// [`ObservedSelection`]); its whitened image `Wᵀg` is the observation's
    /// own draw.
    ///
    /// `profile` is `Some` exactly when the family profiles its scale: each
    /// draw then selects with the profiled criterion over its own residual
    /// deviance ([`SmoothLrSelectionProfile`]) rather than the known-scale one.
    fn generate(
        whitener: &Array2<f64>,
        unit_penalties: &[Array2<f64>],
        log_lambda: &[f64],
        log_scale_windows: &[(f64, f64)],
        profile: Option<SmoothLrSelectionProfile>,
        observed_score: Option<&Array1<f64>>,
        observed_penalized_deviance: Option<f64>,
    ) -> SmoothLrSelection {
        if unit_penalties.is_empty() || unit_penalties.len() != log_lambda.len() {
            return SmoothLrSelection::Declined(
                SmoothLrSelectionDecline::NoPenaltyComponents,
            );
        }
        if whitener.ncols() == 0 {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::NoInformation);
        }
        let Some(geometry) = SelectionGeometry::whiten(whitener, unit_penalties, log_lambda)
        else {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::GeometryRefused);
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
            profile,
            SMOOTH_LR_SELECTION_DRAWS,
            SMOOTH_LR_MULTISCALE_DRAWS,
            observed.as_deref().map(|whitened| ObservedDraw {
                whitened,
                penalized_deviance: observed_penalized_deviance,
            }),
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
        profile: Option<SmoothLrSelectionProfile>,
        diagonal_draws: usize,
        multiscale_draws: usize,
        observed: Option<ObservedDraw<'_>>,
    ) -> SmoothLrSelection {
        if log_scale_windows.len() != geometry.roots.len() {
            return SmoothLrSelection::Declined(
                SmoothLrSelectionDecline::NoPenaltyComponents,
            );
        }
        // A profiled selection needs the observation's own deviance to select
        // it by the profiled rule; without it the observation cannot be scored
        // the way its reference is.
        if observed.is_some_and(|draw| {
            draw.whitened.len() != geometry.dimension
                || (profile.is_some() && draw.penalized_deviance.is_none())
        }) {
            return SmoothLrSelection::Declined(SmoothLrSelectionDecline::ObservedScoreUnusable);
        }
        if geometry.roots.len() >= 2 {
            return match Self::generate_multiscale(
                geometry,
                log_scale_windows,
                profile,
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
                    profile,
                    diagonal_draws,
                    observed,
                ),
                Err(reason) => SmoothLrSelection::Declined(reason),
            };
        }
        Self::generate_common_scale(geometry, log_scale_windows, profile, diagonal_draws, observed)
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
        profile: Option<SmoothLrSelectionProfile>,
        draws: usize,
        observed: Option<ObservedDraw<'_>>,
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
        // eigenbasis, and on a profiled replay its floor. Under a common scale
        // the eigenbasis reaches every direction of the block, so the floor is
        // the draw's residual deviance alone. The observation goes through
        // this same function.
        let score = |squares: &[f64], profile: Option<ProfiledFloor>| {
            let criterion = DiagonalCriterion {
                squares,
                generalized: &generalized,
                rank: geometry.rank,
                occam: &[],
                constant,
                profile,
            };
            let selected = criterion.select(low, high).ok()?;
            // The control variate's conditional arm is read AT the fitted scale,
            // `ln t = 0`, on the same draw.
            Some((criterion.statistic(selected), criterion.statistic(0.0)))
        };

        let profiled = match profile
            .map(|profile| profile.residual_draws(geometry, draws))
            .transpose()
        {
            Ok(profiled) => profiled,
            Err(reason) => return SmoothLrSelection::Declined(reason),
        };

        let dimension = geometry.dimension;
        let mut squares = vec![0.0_f64; dimension];
        let mut stream = SelectionDrawStream::new(dimension, draws);
        let mut selection_sample = vec![0.0_f64; draws];
        let mut conditional_sample = vec![0.0_f64; draws];
        for draw in 0..draws {
            stream.fill_chi_square_ones(&mut squares);
            let floor = profiled.as_ref().map(|profiled| ProfiledFloor {
                multiplier: profiled.multiplier,
                floor: profiled.residual[draw],
            });
            let Some((selected, held)) = score(&squares, floor) else {
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
                        .map(|row| draw.whitened[row] * fitted.basis[[row, column]])
                        .sum();
                    *square = projection * projection;
                }
                let floor = match (profiled.as_ref(), draw.penalized_deviance) {
                    (None, _) => None,
                    (Some(profiled), Some(deviance)) => {
                        let Some(floor) = fitted.observed_floor(draw.whitened, deviance) else {
                            return SmoothLrSelection::Declined(
                                SmoothLrSelectionDecline::ObservedScoreUnusable,
                            );
                        };
                        Some(ProfiledFloor {
                            multiplier: profiled.multiplier,
                            floor,
                        })
                    }
                    (Some(_), None) => {
                        return SmoothLrSelection::Declined(
                            SmoothLrSelectionDecline::ObservedScoreUnusable,
                        );
                    }
                };
                let Some((selected, conditional)) = score(&squares, floor) else {
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
            consumed_coordinates: consumed_coordinates(dimension, profiled.is_some()),
            residual_unit_sample: profiled.map(|profiled| profiled.unit),
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
    /// # Each draw's selection is a coordinatewise CERTIFIED descent
    ///
    /// No basis diagonalizes `m` penalties against the information at once, but
    /// one scale at a time does: with every other scale held at the draw's current
    /// point, the criterion along scale `i` is exactly an [`AxisSlice`], and its
    /// certified global minimum over that axis's window comes from the same
    /// [`DiagonalCriterion`] search the common-scale lane uses. See
    /// [`Self::select_draw`] for the sweep and why it terminates without a budget.
    ///
    /// This replaces a `441`-point bracket grid followed by a compass descent
    /// capped at `96` criterion evaluations per draw, which kept the best point a
    /// capped draw had reached (#2902: SPEC rules 18, 19 and 22). A draw whose
    /// slice cannot be priced or certified declines the whole replay rather than
    /// being sampled partially.
    fn generate_multiscale(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        profile: Option<SmoothLrSelectionProfile>,
        draws: usize,
        observed: Option<ObservedDraw<'_>>,
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
        let profiled = profile
            .map(|profile| profile.residual_draws(geometry, draws))
            .transpose()?;
        let dimension = geometry.dimension;
        let mut factor = SelectionFactor::new(geometry);
        let mut stream = SelectionDrawStream::new(dimension, draws);
        let mut draw = vec![0.0_f64; dimension];
        let mut coordinates = vec![0.0_f64; geometry.rank];
        let mut selected = vec![0.0_f64; scales];
        // The observation's profiled `(m, R)`, read off the fit in the fitted
        // eigensystem before that is handed to the replay.
        let observed_residual = match (observed, profiled.as_ref()) {
            (Some(draw), Some(profiled)) => {
                let deviance = draw
                    .penalized_deviance
                    .ok_or(SmoothLrSelectionDecline::ObservedScoreUnusable)?;
                let floor = fitted
                    .observed_floor(draw.whitened, deviance)
                    .ok_or(SmoothLrSelectionDecline::ObservedScoreUnusable)?;
                Some((profiled.multiplier, floor))
            }
            _ => None,
        };
        // `(W(t*), W(1))` for one whitened draw and, on a profiled replay, its
        // `(m, R)`. The observation goes through this same function.
        let mut score = |draw: &[f64],
                         residual: Option<(f64, f64)>|
         -> Result<(f64, f64), SmoothLrSelectionDecline> {
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
            Self::select_draw(geometry, log_scale_windows, &coordinates, residual, &mut selected)?;
            if !factor.refactor(geometry, &selected) {
                return Err(SmoothLrSelectionDecline::SelectionUnresolved);
            }
            Ok((factor.score(&coordinates, norm_squared).1, conditional))
        };
        let mut selection_sample = vec![0.0_f64; draws];
        let mut conditional_sample = vec![0.0_f64; draws];
        for index in 0..draws {
            stream.fill_normals(&mut draw);
            let residual = profiled
                .as_ref()
                .map(|profiled| (profiled.multiplier, profiled.residual[index]));
            (selection_sample[index], conditional_sample[index]) = score(&draw, residual)?;
        }
        let observed = match observed {
            None => None,
            Some(draw) => {
                let (selected, conditional) = score(draw.whitened, observed_residual)?;
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
            consumed_coordinates: consumed_coordinates(dimension, profiled.is_some()),
            residual_unit_sample: profiled.map(|profiled| profiled.unit),
            observed,
        })
    }

    /// One draw's multi-scale selection, written into `selected`.
    ///
    /// The draw starts at the fitted point, clamped into each open window, and
    /// sweeps the open axes in order. An axis moves to the certified global
    /// minimum of its [`AxisSlice`] only when that lowers the slice's criterion by
    /// more than both points' forward-error bands, so a move is never a rounding
    /// artefact. The descent stops at the first sweep that moves nothing: every
    /// axis is then at its own global minimum through the point, which is a
    /// stationary point of the box-constrained criterion. Every accepted move
    /// lowers a criterion that is bounded below on the box by more than its own
    /// rounding, so the sweep terminates without an iteration budget.
    ///
    /// `residual` is the draw's profiled `(m, R)` when the family profiles its
    /// scale ([`AxisSlice::criterion`]).
    fn select_draw(
        geometry: &SelectionGeometry,
        log_scale_windows: &[(f64, f64)],
        coordinates: &[f64],
        residual: Option<(f64, f64)>,
        selected: &mut [f64],
    ) -> Result<(), SmoothLrSelectionDecline> {
        if selected.len() != log_scale_windows.len() {
            return Err(SmoothLrSelectionDecline::NoPenaltyComponents);
        }
        for (slot, &(low, high)) in selected.iter_mut().zip(log_scale_windows) {
            *slot = if high > low { 0.0_f64.clamp(low, high) } else { 0.0 };
        }
        loop {
            let mut moved = false;
            for (axis, &(low, high)) in log_scale_windows.iter().enumerate() {
                if !(high > low) {
                    continue;
                }
                let slice = AxisSlice::new(geometry, selected, axis, coordinates)
                    .ok_or(SmoothLrSelectionDecline::SelectionUnresolved)?;
                let criterion = slice.criterion(residual);
                let candidate = criterion
                    .select(low, high)
                    .map_err(|_| SmoothLrSelectionDecline::SelectionUnresolved)?;
                let (
                    Some(([at_candidate, ..], candidate_band)),
                    Some(([at_incumbent, ..], incumbent_band)),
                ) = (criterion.jet(candidate), criterion.jet(selected[axis]))
                else {
                    return Err(SmoothLrSelectionDecline::SelectionUnresolved);
                };
                if at_candidate < at_incumbent - (candidate_band + incumbent_band) {
                    selected[axis] = candidate;
                    moved = true;
                }
            }
            if !moved {
                return Ok(());
            }
        }
    }
    /// The factor that carries the observation from the fitted `λ̂` to the
    /// replay's own selection: the quadratic model's ratio `W_q(t*; z)/W_q(1; z)`
    /// of the observed whitened score (see [`Self::observed`]). One when no
    /// observation was scored, or when its `W_q(1; z)` is not a positive number
    /// to divide by — a score of exactly zero, where every `t` gives the same
    /// `W_q = 0` and the selection moves nothing.
    fn observed_ratio(&self) -> f64 {
        match self.observed {
            Some(observed)
                if observed.conditional.is_finite()
                    && observed.conditional > 0.0
                    && observed.selected.is_finite() =>
            {
                observed.selected / observed.conditional
            }
            _ => 1.0,
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
    fn tail_shift_at(&self, conditional_threshold: f64, selection_threshold: f64) -> (f64, f64) {
        paired_mean_with_error(
            self.selection_sample
                .iter()
                .zip(self.conditional_sample.iter())
                .map(|(&selected, &held)| {
                    f64::from(selected >= selection_threshold)
                        - f64::from(held >= conditional_threshold)
                }),
        )
    }

    /// The same shift for a statistic whose scale was PROFILED, integrated over
    /// the residual law rather than read at one point of it.
    ///
    /// With an estimated scale the event `W ≥ w` is `Q ≥ c·V`
    /// ([`SmoothLrProfiledScale`]), and under the replay's selection the
    /// observation's `Q` is carried by [`Self::observed_ratio`] `ρ` with its `V`
    /// unchanged, so its event is `Q_sel ≥ ρc·V`. The shift the replay has to
    /// supply is therefore
    ///
    /// ```text
    ///   E[ 1{Q_sel ≥ ρc·V} − 1{Q_cond ≥ c·V} ],
    ///   V = Σ_r v_r·χ²_1 + χ²_h,
    /// ```
    ///
    /// over the joint law of the draw and `V`. Conditional plus shift is then
    /// `P(Q_sel ≥ ρc·V)` up to the Monte-Carlo error of the paired difference,
    /// because the analytic conditional tail `P(Q_cond − c·V ≥ 0)` integrates
    /// over the same `V`. What this replaced read the shift at the single
    /// threshold `c·E[V]`: the shift is a difference of two tails, which is not
    /// linear in the threshold, so its value at the mean threshold is not its
    /// mean over the threshold's law.
    ///
    /// The non-unit part `V' = Σ_r v_r·χ²_1` is drawn from the same stratified
    /// stream as the replay, on coordinates numbered after every one the replay
    /// consumed, so it is independent of the draws it pairs with and the whole
    /// construction is still a pure function of its inputs. The weight-one block
    /// `χ²_h` — the dominant term, `h = n − p` — depends on how the replay
    /// selected:
    ///
    /// - A PROFILED selection drove each draw with its own `χ²_h`
    ///   ([`SmoothLrSelectionProfile`]): a large residual flattens the profiled
    ///   criterion's data term and so moves `λ̂`, and the same residual is the
    ///   weight-one block of that draw's `V`. The two are one variate, so the
    ///   event is read on the draw's own. Integrating a fresh `χ²_h` instead
    ///   would pair a selection with a residual it was not made under.
    /// - A known-scale selection never saw the residual, so `χ²_h` is
    ///   independent of the draw and is integrated in CLOSED FORM:
    ///   `P(Q ≥ r(V' + χ²_h)) = 1 − P(χ²_h > Q/r − V')`, one regularized
    ///   incomplete gamma per draw and arm.
    fn profiled_tail_shift(
        &self,
        conditional_ratio: f64,
        selection_ratio: f64,
        scale: &SmoothLrProfiledScale,
    ) -> (f64, f64) {
        let draws = self.selection_sample.len();
        let non_unit = stratified_weighted_chi_square_sum(
            &scale.residual_weights,
            self.consumed_coordinates,
            draws,
        );
        let arms = self
            .selection_sample
            .iter()
            .zip(self.conditional_sample.iter())
            .zip(non_unit.iter());
        if let Some(unit_sample) = self.residual_unit_sample.as_ref() {
            return paired_mean_with_error(arms.zip(unit_sample.iter()).map(
                |(((&selected, &held), &non_unit), &unit)| {
                    let residual = non_unit + unit;
                    f64::from(selected >= selection_ratio * residual)
                        - f64::from(held >= conditional_ratio * residual)
                },
            ));
        }
        let unit = scale.residual_unit_dimension;
        // `P(Q ≥ r·V)` for one draw of `Q` and of `V'`.
        let exceedance = |quadratic: f64, ratio: f64, non_unit: f64| {
            let room = quadratic / ratio - non_unit;
            if unit > 0.0 {
                if room > 0.0 {
                    1.0 - gam_math::probability::chi_square_sf(room, unit)
                } else {
                    0.0
                }
            } else {
                f64::from(room >= 0.0)
            }
        };
        paired_mean_with_error(arms.map(|((&selected, &held), &non_unit)| {
            exceedance(selected, selection_ratio, non_unit)
                - exceedance(held, conditional_ratio, non_unit)
        }))
    }
}

/// Stratified coordinates a replay of `dimension` consumes: its own block, and
/// on a profiled replay the two residual coordinates after it
/// ([`SmoothLrSelectionProfile::residual_draws`]).
fn consumed_coordinates(dimension: usize, profiled: bool) -> usize {
    if profiled { dimension + 2 } else { dimension }
}

/// `(mean, standard error)` of a paired-difference sample. `d_i ∈ {−1, 0, +1}`
/// (or its closed-form average over `χ²_h`) is zero on every draw whose
/// selection left it on the same side of both thresholds — which is most of
/// them. That is the control variate, and this is its own sample variance
/// rather than the `√(p(1−p)/N)` of either term alone. Empty is `(0, 0)`: no
/// replay, no shift.
///
/// The spread is accumulated in two passes about the sample's own mean:
/// `M₂ = Σ (dᵢ − d̄)²` is a sum of squares, so it is `≥ 0` in floating point
/// with no clamp. The mean is read pivoted on the first draw,
/// `d̄ = d₁ + Σ (dᵢ − d₁)/N`, so a constant sample has every pivoted
/// difference exactly zero and reports `d̄ = d₁` and `M₂ = 0` exactly. Any
/// rounding left in `d̄` enters `M₂` only at second order, since
/// `Σ (dᵢ − m)² = Σ (dᵢ − d̄)² + N (d̄ − m)²`. The one-pass `Σd²/N − d̄²`
/// it replaces differences two quantities of size `d̄²`, so it resolves the
/// variance only in steps of `ulp(d̄²)` and reports a positive error on a
/// draw set that is constant (#4086). A running (Welford) mean does not
/// fix this either: on an ordered sample it drifts by `ulp(d̄)` per step,
/// which is not small beside a spread far below `|d̄|`.
fn paired_mean_with_error(differences: impl Iterator<Item = f64>) -> (f64, f64) {
    let sample: Vec<f64> = differences.collect();
    let Some(&pivot) = sample.first() else {
        return (0.0, 0.0);
    };
    let count = sample.len() as f64;
    let shift = pivot + sample.iter().map(|d| d - pivot).sum::<f64>() / count;
    let centred_squares: f64 = sample.iter().map(|d| (d - shift) * (d - shift)).sum();
    (shift, (centred_squares / count / count).sqrt())
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
        let signed = stratum_normals(draws);
        let values: Vec<f64> = signed.iter().map(|normal| normal * normal).collect();
        let permutations = (0..dimension)
            .map(|coordinate| stratum_permutation(coordinate, draws))
            .collect();
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

/// The `N` stratum midpoints of the standard normal, in stratum order.
fn stratum_normals(draws: usize) -> Vec<f64> {
    (0..draws)
        .map(|bin| {
            // Bin midpoint: never `0` or `1`, so the quantile is finite.
            let uniform = (bin as f64 + 0.5) / draws as f64;
            gam_math::probability::standard_normal_quantile(uniform)
                .expect("a bin midpoint is strictly inside (0, 1)")
        })
        .collect()
}

/// The order in which coordinate `coordinate` visits the `N` strata: a
/// Fisher–Yates shuffle driven by a counter-based stream keyed by the
/// coordinate, so the same coordinate index always gets the same order.
fn stratum_permutation(coordinate: usize, draws: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..draws as u32).collect();
    let mut state =
        0x9E37_79B9_7F4A_7C15_u64 ^ (coordinate as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    for position in (1..order.len()).rev() {
        state = split_mix64(state);
        let pick = (state % (position as u64 + 1)) as usize;
        order.swap(position, pick);
    }
    order
}

/// `Σ_r a_r·χ²_{1,r}` over `draws` stratified draws, one coordinate per weight,
/// the coordinates numbered from `first_coordinate`.
///
/// The offset is what keeps these draws INDEPENDENT of a replay's: a replay of
/// dimension `q` consumes coordinates `0..q`, so a sum that is to be independent
/// of it starts at `q`. Built one coordinate at a time, so the memory is one
/// permutation rather than one per weight.
fn stratified_weighted_chi_square_sum(
    weights: &[f64],
    first_coordinate: usize,
    draws: usize,
) -> Vec<f64> {
    let squares: Vec<f64> = stratum_normals(draws)
        .iter()
        .map(|normal| normal * normal)
        .collect();
    let mut sum = vec![0.0_f64; draws];
    for (offset, &weight) in weights.iter().enumerate() {
        let order = stratum_permutation(first_coordinate + offset, draws);
        for (slot, &stratum) in sum.iter_mut().zip(order.iter()) {
            *slot += weight * squares[stratum as usize];
        }
    }
    sum
}

/// `χ²_k` over `draws` stratified draws on ONE coordinate: draw `i` is the
/// `χ²_k` quantile at the midpoint of the stratum [`stratum_permutation`] gives
/// it, so the sample is the same Latin-hypercube construction as the replay's
/// own `χ²_1` coordinates at the cost of one quantile per draw rather than `k`
/// normals. `k = 0` is the point mass at zero.
fn stratified_chi_square(degrees_of_freedom: usize, coordinate: usize, draws: usize) -> Vec<f64> {
    if degrees_of_freedom == 0 {
        return vec![0.0; draws];
    }
    let degrees_of_freedom = degrees_of_freedom as f64;
    stratum_permutation(coordinate, draws)
        .into_iter()
        .map(|stratum| {
            let uniform = (f64::from(stratum) + 0.5) / draws as f64;
            gam_math::probability::chi_square_quantile(uniform, degrees_of_freedom)
        })
        .collect()
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

/// The reference distribution [`SmoothTermLrInference`] scores its statistic
/// against, reported as the two spectral moments it is built from (#2672).
///
/// # What the statistic's null law actually is
///
/// Expand the log-likelihood quadratically about the unpenalized MLE `β̃` and
/// write `I = X'WX`, `S` for the penalty, `H = I + S`, `j` for the tested block
/// and `n` for the retained one. The penalized fit is `β̂ = Fβ̃` with
/// `F = H⁻¹I`, and the null fit is the retained block's own projection, so
///
/// ```text
/// W = β̃_j' (Ĩ_jj − N) β̃_j ,   β̃_j ~ N(0, Ĩ_jj⁻¹)
/// ```
///
/// with `Ĩ_jj = I_jj − I_jn I_nn⁻¹ I_nj` the Schur complement and
/// `N = [S H⁻¹ I H⁻¹ S]_jj`. Setting `H̃ = Ĩ_jj + S_jj` and `P = H̃⁻¹S_jj`, that
/// collapses to `(Ĩ_jj − N)Ĩ_jj⁻¹ = H̃(I − P²)H̃⁻¹`, and — because the block of
/// the GLOBAL influence matrix equals the Schur-complement influence,
/// `F_jj = H̃⁻¹Ĩ_jj = I − P` — the eigenvalues are exactly `2F_jj − F_jj²`.
/// So
///
/// ```text
/// W = Σ_j w_j χ²_1 ,   w = eig(2·F_jj − F_jj²) ∈ (0, 1]^q.
/// ```
///
/// # Consequences, and what this replaced
///
/// `Σ w_j = 2 tr(F_jj) − tr(F_jj²)` is Wood's `edf1`. So `edf1` is not a
/// citation here, it is the statistic's first-order null MEAN, derived. What it
/// is not is a chi-square degrees of freedom: `Var(W) = 2 Σ w_j²` against the
/// mean-matched `χ²_{Σw}`'s `2 Σ w_j`, and `w_j ≤ 1`, so a mean-matched chi-square
/// is over-dispersed for every penalized term and the test is conservative by
/// construction.
///
/// # Why the reference is the spectrum and not two of its moments
///
/// Matching the second moment as well — `W ≈ g·χ²_ν` with `ν = (Σw)²/Σw²`,
/// `g = Σw²/Σw` — fixes the *shape* with no free constant, and is EXACT whenever
/// the weights are equal (which includes the classical unpenalized case
/// `w ≡ 1 ⇒ ν = q, g = 1 ⇒ χ²_q`). It is not exact otherwise, and the error is
/// one-signed and grows with the depth of the tail, which is the half of the
/// p-value range that decides anything. Measured against the exact law on the
/// shrinkage spectrum `f_j = 1/(1 + λγ_j)` of a second-difference penalty, over
/// six decades of `λ` and `k ∈ {6, 12, 20}`, the size a two-moment reference
/// actually delivers at a nominal `α` is
///
/// ```text
/// α = 0.05   0.99 – 1.02 ×      α = 1e-3   1.01 – 1.31 ×
/// α = 0.01   1.00 – 1.11 ×      α = 1e-4   1.14 – 1.61 ×
/// ```
///
/// i.e. it is fine where the test is least discriminating and up to 61%
/// anti-conservative where it is most.
///
/// Where the gap lives matters as much as its size, and it is not where the
/// intuition puts it. The surrogate is exact at BOTH ends of the shrinkage
/// range — `w ≡ 1` unpenalized, and a single distinct weight once REML has
/// shrunk a term to its null space (measured on a null-true `k = 12` fit:
/// `w = (0.322, 5.9e-7, 7.1e-8, …)`, where the two references agree to eight
/// figures). It opens in the middle, at moderate shrinkage, which is exactly
/// where a smooth term carrying real signal sits. Nothing about the statistic requires that
/// trade: the weights are the parameters of an exactly invertible
/// characteristic function, and
/// `gam_math::probability::signed_weighted_chi_square_sf`
/// inverts it with a *returned* error bound, relative to the tail itself, so
/// the smallest tail any of the numbers above resolves is resolved to near
/// full precision. So the
/// reference is `P(Σ_j w_j χ²_1 > W)` itself, and the `(ν, g)` pair survives only
/// as a two-number summary of the spectrum's shape, published for continuity and
/// no longer consulted when the spectrum is known.
///
/// The one-moment reference this replaced sits at the far end of the same axis,
/// with the same sign: on a spectrum shaped like a shrunk smooth
/// (`0.08, 0.02, 0.005, 0.001, 2e-4`), at `x = 8·Σw` the exact tail is `1.4e-3`
/// and the mean-matched chi-square reports `3.6e-2` — 26× conservative.
///
/// # What went away with the mean-only reference
///
/// Three things, and none of them needed a replacement:
///
/// * `+ tr(X'WX · J Var(ρ̂) Jᵀ)/φ`, the Wood–Pya–Säfken smoothing-parameter
///   inflation added under #1872. That is a *coefficient-covariance* correction
///   for AIC; it is not a term in this statistic's null law, and it is largest
///   exactly where the outer criterion is flattest — i.e. where the term has the
///   LEAST effective d.f. Measured on the #2672 fixture: a replicate with
///   `edf = 0.070` was handed `rho_uncertainty = 1.79`, twenty-five times the
///   term's own effective d.f. It was holding the size up by an unrelated
///   mechanism. λ̂'s sampling variation enters the reference through its
///   selection replay, which integrates the law over the λ̂ the fit could have
///   chosen.
/// * `.max(edf)` and `.max(null_dim)`. Both are automatic: `w_j = 1` exactly on
///   an unpenalized direction, so `Σ w_j ≥ null_dim` by construction, and `Σ w_j`
///   dominates `tr(F_jj) = edf` because `w_j = 2f_j − f_j² ≥ f_j` for `f_j ∈ [0,1]`.
/// * `.max(1.0)`, the #1766 degeneracy floor. It existed because `χ²_d` with
///   `d → 0` reports any positive `W` as maximally significant. The scaled
///   reference cannot degenerate that way: as REML shrinks a term the weights and
///   the statistic collapse *together*, `W/g` stays `O(1)`, and `ν → q`. The floor
///   was a patch on the wrong shape, not on a missing quantity.
///
/// # And what all of it is the reference FOR
///
/// Everything above describes the law of `Q`, the statistic a KNOWN-scale
/// likelihood ratio is. A profiled Gaussian's `W` is not `Q` — it is
/// `n·ln(1 + Q/V) + B`, with `V` the residual sum of squares the same fit
/// estimated its `σ̂` from. Scoring `W` against `Q`'s law is anti-conservative
/// at `O(1/ν)`: measured `size@.05 = 0.0792` pooled over 480 replicates at
/// `n ∈ {30, 50}` against a nominal `0.05`, with the Lawley factor inert
/// throughout, so the whole miss belonged here. [`Self::profiled_scale`] is
/// that channel and [`SmoothLrProfiledScale`] is the derivation; it is `None`
/// on every family whose dispersion is already inside the IRLS weight.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothLrReferenceDf {
    /// The null spectrum itself, `w_j ∈ [0, 1]`, sorted descending — the whole
    /// reference on the [`SmoothLrReferenceSource::NullSpectrum`] lane. Empty on
    /// the two lanes that could not reach it, which is exactly the condition
    /// under which `Self::tail_probability_with_bound` falls back to the `(ν, g)` pair.
    pub weights: Vec<f64>,
    /// First spectral moment `Σ_j w_j = 2·tr(F_jj) − tr(F_jj²)` — Wood's `edf1`,
    /// and exactly the statistic's first-order null mean `E[W|λ]`. This is the
    /// `d` the Lawley Bartlett factor `c = 1 + Δε/d` is denominated in.
    pub mean: f64,
    /// Second spectral moment `Σ_j w_j² = tr((2F_jj − F_jj²)²)`, i.e. `Var(W)/2`.
    pub second_moment: f64,
    /// Shape of the two-moment SUMMARY `ν = mean²/second_moment`. It is what the
    /// reference used to be, and it is still what the reference is on the
    /// [`SmoothLrReferenceSource::SpectralMomentMatch`] and
    /// [`SmoothLrReferenceSource::UnitWeightFallback`] lanes; on the exact lane
    /// it is a published descriptor of the spectrum's shape and is not consulted.
    pub chi_square_df: f64,
    /// Scale of that summary, `g = second_moment/mean`. Same status as
    /// [`Self::chi_square_df`].
    pub scale: f64,
    /// The agreement between the two independently-assembled routes to this
    /// spectrum, when the fit supplied the inputs for both: the larger of the
    /// two relative residuals between `(Σw, Σw²)` read off `[H⁻¹]_jj S_jj` and
    /// `(tr A, tr A²)` read off the influence block, `A = 2F_jj − F_jj²`.
    ///
    /// The two are the same object by an algebraic identity that depends on the
    /// penalty being block-diagonal by term AND on `Vb`, `F` and `S` being
    /// published in one coefficient basis. Neither is checkable by inspection,
    /// and both have been wrong here before (`#2672`'s similarity-map drop, its
    /// internal-basis first-order correction, and its block-local
    /// `coeff_range`). So the driver measures the identity on every fit that can
    /// support it and publishes the number rather than assuming it.
    ///
    /// `None` when only one route was available — which is a statement about the
    /// fit, not a failure.
    pub moment_residual: Option<f64>,
    /// The term's conditional effective degrees of freedom `tr(F_jj)`
    /// (`per_term_edf`), reported for continuity with the summary table and used
    /// as the fallback base when neither the spectrum nor its moments are
    /// available.
    pub edf: f64,
    /// The term's joint unpenalized null-space dimension `dim(∩_k null(S_k))`,
    /// reported because it is the analytic lower bound on `mean` and therefore
    /// the cheapest check that the spectrum was assembled on the right block.
    pub null_dim: usize,
    /// Which lane supplied the reference.
    pub source: SmoothLrReferenceSource,
    /// The λ̂-selection replay, or the NAMED reason there is none (#2672).
    ///
    /// A decline means the conditional law IS the selection law here — nothing
    /// was selected — and the tail is read from [`Self::weights`] alone. It is
    /// an enum rather than an `Option` because a missing replay is a statement
    /// about the fit, and a reader who does not have to look at which statement
    /// will not.
    pub selection: SmoothLrSelection,
    /// The ESTIMATED-SCALE channel, present exactly when the fit profiled its
    /// own Gaussian dispersion out of a residual sum of squares (#2672).
    ///
    /// `None` is a statement about the family, not a missing measurement: every
    /// other likelihood on this path carries its dispersion in the IRLS weight
    /// and its `W` is not a function of a second, independently-estimated
    /// scalar. See [`SmoothLrProfiledScale`].
    pub profiled_scale: Option<SmoothLrProfiledScale>,
}

/// What the reference needs in order to score a statistic whose SCALE was
/// estimated from the same residuals (#2672).
///
/// # The statistic is a ratio, exactly
///
/// gam's profiled Gaussian log-likelihood is `ℓ = −½[n·ln 2π + n·ln(D/ν) −
/// Σ ln w_i + ν]` with `D` the weighted residual sum of squares and `ν` the
/// residual degrees of freedom it divides by — so the whole-term LR statistic
/// is, with no expansion anywhere,
///
/// ```text
///   W = 2(ℓ_full − ℓ_null) = n·ln(D_0/D_f) + n·ln(ν_f/ν_0) + (ν_0 − ν_f)
///     = n·ln(1 + Q/V) + B,
///   Q = (D_0 − D_f)/σ²,   V = D_f/σ²,   B = n·ln(ν_f/ν_0) + (ν_0 − ν_f).
/// ```
///
/// `Q` is what [`SmoothLrReferenceDf::weights`] is the null spectrum OF, and
/// the known-scale reference scores it directly. The shipped reference
/// therefore answers the question "how extreme is `Q`" when the question asked
/// was "how extreme is `n·ln(1 + Q/V) + B`" — and `V` is a random variable of
/// the same data, with mean `ν` and spread `√(2ν)`. Both the mean shift
/// `ν/(ν−2)` and the extra spread push the test anti-conservative, and both are
/// `O(1/ν)`: invisible at `n = 1000`, worth `0.03` in size at `n = 30`. It is
/// the same reason mgcv's smooth-term p-values take an `F` reference when the
/// scale is estimated and a `χ²` when it is known.
///
/// # Inverting it costs nothing, because the map is monotone
///
/// `W > w ⟺ Q/V > exp((w − B)/n) − 1`, so
///
/// ```text
///   P(W > w) = P( Q − c(w)·V > 0 ),   c(w) = expm1((w − B)/n),
/// ```
///
/// a linear combination of independent chi-squares with a NEGATIVE weight,
/// evaluated at zero — which is exactly
/// [`gam_math::probability::signed_weighted_chi_square_sf`]. There
/// is no expansion, no `κ`-convention to pick, and no separate `F`-family
/// approximation: `n` and `ν` appear where the log-likelihood actually put
/// them.
///
/// # Where the residual law comes from
///
/// The same spectral object the numerator uses, taken over the whole model
/// instead of over the tested block. The profiled Gaussian's hat matrix
/// `A = X H⁻¹X'W` is symmetric in the whitened coordinates the weighted RSS is
/// a sum of squares in (`X̃ = W^{1/2}X`, where `ε̃ = W^{1/2}ε` has covariance
/// `σ²I` because `Var(y_i) = σ²/w_i`), with eigenvalues `f_i = 1 − p_i` — the
/// `p_i` being the penalty shares `lr_tested_block` returns, which are
/// unchanged by the whitening since `H⁻¹S` is — and `n − p` further zeros. The
/// true mean is annihilated because it lies in the penalty's null space. So
///
/// ```text
///   V = ε̃'(I − Ã)²ε̃ ~ Σ_i p_i²·χ²_1  +  χ²_{n−p},
/// ```
///
/// exact at fixed `λ`, with `n` the POSITIVE-WEIGHT row count in both places.
/// The `n − p` unit directions are folded into ONE term with `n − p` degrees of
/// freedom, which is what keeps an `n`-sized reference the same cost as a
/// `p`-sized one.
///
/// # What is approximated, stated plainly
///
/// Three things, all inherited rather than introduced, and none of them the
/// `O(1/ν)` term this channel exists to remove.
///
/// * `Q`'s spectrum is the reference's own claim, unchanged.
/// * `Q` and `V` are taken INDEPENDENT — exact for the unpenalized linear model
///   by Cochran, approximate under penalization, and the same independence
///   every `F` reference for a penalized smooth rests on.
/// * `V` is taken CENTRAL. It is exactly central when the mean lies in the
///   penalty's null space, which is what the tested term being null gives on
///   its own block; a DIFFERENT term in the model carrying real signal that the
///   penalty shrinks adds a non-centrality. That inflates `V`, which inflates
///   the p-value — so the residual runs conservative, in the direction a test
///   is allowed to be wrong.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothLrProfiledScale {
    /// `n` — the multiplier the profiled `ln σ̂²` carries in the log-likelihood.
    pub observations: f64,
    /// `B = n·ln(ν_f/ν_0) + (ν_0 − ν_f)`, the part of `W` that is a function of
    /// the two fits' residual degrees of freedom and of nothing random.
    ///
    /// It is NOT negligible and it is not the same sign as the rest: on the
    /// `n = 30` Gaussian cells it runs `−0.13` to `−0.61`.
    pub deterministic_offset: f64,
    /// `p_i²` over the whole model's penalty shares — the non-trivial half of
    /// the residual quadratic form's spectrum.
    pub residual_weights: Vec<f64>,
    /// `n − p`, the residual directions no design column reaches. Each carries
    /// weight exactly one, so they are one term with this many degrees of
    /// freedom rather than this many terms.
    pub residual_unit_dimension: f64,
}

impl SmoothLrReferenceDf {
    /// The CONDITIONAL tail — the fixed-`λ` law alone, with the λ̂-selection
    /// replay held out. This is the tail `Self::tail_probability_with_bound`
    /// reports when nothing was selected, and it is the reference the replay
    /// corrects.
    pub fn conditional_tail_probability(&self, statistic: f64) -> f64 {
        self.conditional_tail_with_bound(statistic).0
    }

    /// `P(W > statistic)` under this reference, with an absolute bound on its
    /// error.
    ///
    /// On the exact lane this is `P(Σ_j w_j χ²_1 > W)` by inversion of its moment
    /// generating function; on the two surrogate lanes it is the two-moment
    /// `P(χ²_ν > W/g)`. Both are scale-equivariant in the same way, which is what
    /// lets the Bartlett correction be applied as `W/c` on either.
    ///
    /// A non-finite statistic propagates as `NaN` rather than being scored: the
    /// LR statistic is `NaN` exactly when the null refit did not produce a finite
    /// log-likelihood, and there is no p-value for a test that was not run.
    ///
    /// The bound is the inversion's own, derived from the arithmetic that
    /// produced the value (see `gam_math::probability::signed_weighted_chi_square_sf`),
    /// plus twice the selection replay's Monte-Carlo standard error when a
    /// replay corrects it.
    pub fn tail_probability_with_bound(&self, statistic: f64) -> (f64, f64) {
        let (conditional, bound) = self.conditional_tail_with_bound(statistic);
        let Some(replay) = self.selection.replay() else {
            return (conditional, bound);
        };
        if !conditional.is_finite() {
            return (conditional, bound);
        }
        let observed_ratio = replay.observed_ratio();
        let (shift, standard_error) = match self.profiled_scale.as_ref() {
            None => replay.tail_shift_at(statistic, observed_ratio * statistic),
            Some(scale) => {
                let ratio = ((statistic - scale.deterministic_offset) / scale.observations).exp_m1();
                if !(ratio > 0.0) {
                    // `conditional` is the certain `1` here (see
                    // `conditional_tail_with_bound`), and every draw of either
                    // arm clears a non-positive threshold, so nothing moves.
                    return (conditional, bound);
                }
                replay.profiled_tail_shift(ratio, observed_ratio * ratio, scale)
            }
        };
        (
            (conditional + shift).clamp(0.0, 1.0),
            bound + 2.0 * standard_error,
        )
    }

    /// The statistic's OWN null law, as a list of `λ_j·χ²_{h_j}` terms: the
    /// exact spectrum where the fit supplied one, and the two-moment summary
    /// `(g, ν)` where it did not.
    ///
    /// The two lanes are one object here rather than two branches because the
    /// summary is not an approximation of a different shape — it is the same
    /// linear combination with one term. Reading it this way is what lets the
    /// profiled-scale route below apply on every lane instead of only on the
    /// lane that reached the spectrum.
    fn null_law_terms(&self) -> Vec<gam_math::probability::WeightedChiSquareTerm> {
        use gam_math::probability::WeightedChiSquareTerm;
        if self.weights.is_empty() {
            return vec![WeightedChiSquareTerm {
                weight: self.scale,
                degrees_of_freedom: self.chi_square_df,
            }];
        }
        self.weights
            .iter()
            .map(|&weight| WeightedChiSquareTerm {
                weight,
                degrees_of_freedom: 1.0,
            })
            .collect()
    }

    fn conditional_tail_with_bound(&self, statistic: f64) -> (f64, f64) {
        if !statistic.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let summary =
            |w: f64| gam_math::probability::chi_square_sf(w / self.scale, self.chi_square_df);
        if self.weights.is_empty() && self.profiled_scale.is_none() {
            // The summary IS the reference on the two degraded lanes, and it is
            // a closed form: no truncation, so no bound to report.
            return (summary(statistic), 0.0);
        }
        let mut terms = self.null_law_terms();
        let Some(scale) = self.profiled_scale.as_ref() else {
            return tail_with_bound(&terms, statistic);
        };
        // `W > w  ⟺  Q/V > expm1((w − B)/n)`, so the tail is the SIGNED
        // combination `Q − c·V` at zero. See [`SmoothLrProfiledScale`].
        let ratio = ((statistic - scale.deterministic_offset) / scale.observations).exp_m1();
        if !ratio.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        if ratio <= 0.0 {
            // `W` is at or under the value the two fits' degrees of freedom
            // alone produce. `Q ≥ 0` and `V > 0`, so `Q − cV ≥ 0` with
            // certainty and the statistic is not evidence of anything.
            return (1.0, 0.0);
        }
        terms.extend(scale.residual_weights.iter().map(|&weight| {
            gam_math::probability::WeightedChiSquareTerm {
                weight: -ratio * weight,
                degrees_of_freedom: 1.0,
            }
        }));
        if scale.residual_unit_dimension > 0.0 {
            terms.push(gam_math::probability::WeightedChiSquareTerm {
                weight: -ratio,
                degrees_of_freedom: scale.residual_unit_dimension,
            });
        }
        tail_with_bound(&terms, 0.0)
    }

    /// The typed p-value: a point value when its published accuracy resolves
    /// it away from zero, and an explicit ceiling when it does not.
    ///
    /// `(value, accuracy)` is what [`Self::tail_probability_with_bound`] returned
    /// at the statistic. When `value ≤ accuracy` the interval the reference
    /// certifies, `[value − accuracy, value + accuracy]`, contains zero: the
    /// digits of `value` are not the tail, and publishing them as a probability
    /// would be publishing noise. What IS known there is the interval's top, so
    /// that is what is reported.
    ///
    /// `None` when the value or its accuracy is not a number, which the driver
    /// reports as [`SmoothLrUnavailable::TailNotComputable`].
    pub fn typed_p_value(value: f64, accuracy: f64) -> Option<SmoothLrPValue> {
        if !(value.is_finite() && accuracy.is_finite()) {
            return None;
        }
        if value > accuracy {
            return Some(SmoothLrPValue::Resolved(value));
        }
        // "p ≤ 0" is false for every continuous law; a ceiling that rounds to
        // zero is lifted to the smallest positive double, the least
        // representable true statement.
        Some(SmoothLrPValue::UpperBound(
            (value.max(0.0) + accuracy).clamp(f64::from_bits(1), 1.0),
        ))
    }
}

/// A smooth term's LR p-value as the reference can actually certify it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SmoothLrPValue {
    /// A point value, resolved away from zero by the published accuracy
    /// [`SmoothTermLrInference::p_value_bound`].
    Resolved(f64),
    /// `p ≤` this ceiling. The published accuracy does not separate the value
    /// from zero, so a point value would be noise; see
    /// [`SmoothLrReferenceDf::typed_p_value`].
    UpperBound(f64),
}

/// Why a tested smooth term carries no LR p-value. Every smooth term the LR test
/// applies to gets a row: a p-value ([`SmoothLrPValue`]) or one of these, never a
/// silent gap and never an error for the whole call because one term's test
/// could not run. (A shape-constrained term, which the test does not apply to,
/// is named by [`smooth_term_lr_unavailable_forspec`].)
#[derive(Clone, Debug, PartialEq)]
pub enum SmoothLrUnavailable {
    /// The term spans no coefficient columns in the fitted design, so there is
    /// nothing to drop and no hypothesis to test.
    EmptyCoefficientBlock,
    /// The term's null law has no positive mean or no positive two-moment
    /// shape/scale — the tested block carries no degree of freedom the data can
    /// move, so no tail can be read from it.
    DegenerateReference,
    /// The full model could not be refitted from the supplied data, so no term
    /// has an `ℓ_full` to be compared against.
    FullRefitFailed(String),
    /// The reduced model (this term's block fixed at zero, every other
    /// smoothing parameter at the full fit's `λ̂`) did not reach a converged
    /// optimum. A fit object is only ever the product of a converged
    /// optimization, so there is no `ℓ_null` and no statistic.
    NullFitNotConverged(String),
    /// The reduced model is not the full model with this term's block
    /// constrained to zero at the full fit's `λ̂` — a penalty spans the tested
    /// block and a surviving one, a constraint needs the tested block, the
    /// link shape was estimated jointly, or the fit uses the bounded-linear
    /// route — so no nested likelihood ratio is defined for it here.
    NullFitUnsupported(String),
    /// The reduced model converged but its log-likelihood is not finite.
    NullLogLikelihoodNotFinite,
    /// The statistic was formed but the reference returned no finite tail or
    /// accuracy for it.
    TailNotComputable,
}

impl SmoothLrUnavailable {
    /// Stable machine-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::EmptyCoefficientBlock => "empty_coefficient_block",
            Self::DegenerateReference => "degenerate_reference",
            Self::FullRefitFailed(_) => "full_refit_failed",
            Self::NullFitNotConverged(_) => "null_fit_not_converged",
            Self::NullFitUnsupported(_) => "null_fit_unsupported",
            Self::NullLogLikelihoodNotFinite => "null_log_likelihood_not_finite",
            Self::TailNotComputable => "tail_not_computable",
        }
    }
}

impl std::fmt::Display for SmoothLrUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCoefficientBlock => f.write_str("the term spans no coefficient columns"),
            Self::DegenerateReference => {
                f.write_str("the term's null law has no positive mean, shape or scale")
            }
            Self::FullRefitFailed(message) => write!(f, "full-model refit failed: {message}"),
            Self::NullFitNotConverged(message) => {
                write!(f, "the reduced model (term fixed at zero) did not converge: {message}")
            }
            Self::NullFitUnsupported(message) => {
                write!(f, "no nested reduced model for this term: {message}")
            }
            Self::NullLogLikelihoodNotFinite => {
                f.write_str("the reduced model's log-likelihood is not finite")
            }
            Self::TailNotComputable => {
                f.write_str("the reference produced no finite tail at this statistic")
            }
        }
    }
}

/// One smooth term's LR significance outcome: the report, or why there is none.
#[derive(Clone, Debug)]
pub struct SmoothTermLrReport {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    pub outcome: Result<SmoothTermLrInference, SmoothLrUnavailable>,
}

impl SmoothTermLrReport {
    /// The report, when the test ran.
    pub fn inference(&self) -> Option<&SmoothTermLrInference> {
        self.outcome.as_ref().ok()
    }
}

impl SmoothLrPValue {
    /// The point value, when there is one.
    pub fn value(self) -> Option<f64> {
        match self {
            Self::Resolved(value) => Some(value),
            Self::UpperBound(_) => None,
        }
    }

    /// The ceiling, when the tail was reported as one.
    pub fn upper_bound(self) -> Option<f64> {
        match self {
            Self::UpperBound(bound) => Some(bound),
            Self::Resolved(_) => None,
        }
    }
}

/// The weighted chi-square tail with its bound stated absolutely, the form the
/// replay's shift is added to.
fn tail_with_bound(terms: &[gam_math::probability::WeightedChiSquareTerm], statistic: f64) -> (f64, f64) {
    let tail = gam_math::probability::signed_weighted_chi_square_sf(terms, statistic);
    (tail.probability, tail.absolute_error())
}

/// A smooth term the per-term LR test does not report, with the typed reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmoothTermLrUnavailable {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// Why no p-value exists for this term.
    pub reason: gam_solve::estimate::SmoothPValueUnavailable,
}

/// The smooth terms of `resolvedspec` that [`smooth_term_lr_inference_forspec`]
/// cannot test, each with its typed reason. A shape-constrained term's null
/// `f = 0` is the apex of its constraint cone, so the LR statistic has no
/// calibrated reference law (see [`gam_solve::estimate::SmoothPValueUnavailable`]).
pub fn smooth_term_lr_unavailable_forspec(
    resolvedspec: &TermCollectionSpec,
) -> Vec<SmoothTermLrUnavailable> {
    resolvedspec
        .smooth_terms
        .iter()
        .enumerate()
        .filter_map(|(term_idx, term)| {
            gam_solve::estimate::smooth_pvalue_unavailable(&term.shape).map(|reason| {
                SmoothTermLrUnavailable {
                    name: term.name.clone(),
                    term_idx,
                    reason,
                }
            })
        })
        .collect()
}

/// The Bartlett-corrected per-term significance report for one penalized smooth
/// term (#1063). Unlike the summary table's Wood rank-truncated **Wald**
/// statistic, this is a genuine **likelihood-ratio** statistic from a
/// constrained refit (the smooth dropped), so the exact Lawley LR Bartlett
/// factor corrects the right quantity.
#[derive(Clone, Debug)]
pub struct SmoothTermLrInference {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// The uncorrected likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)`,
    /// floored at zero (a non-negative LR by construction).
    pub statistic_lr: f64,
    /// The statistic's first-order null mean `d = E[W|λ] = Σ_j w_j`, which is
    /// Wood's `edf1 = 2·tr(F_bb) − tr(F_bb²)` exactly (see
    /// [`SmoothLrReferenceDf`] for why that is a derivation and not a citation).
    /// This is the `d` the Lawley Bartlett factor `c = 1 + Δε/d` is denominated
    /// in. It is **not** a chi-square degrees of freedom — the reference the
    /// p-values are read from is [`Self::ref_df_provenance`]'s
    /// `chi_square_df`/`scale` pair, which coincides with `ref_df` only when the
    /// tested block is unpenalized.
    pub ref_df: f64,
    /// The reference distribution itself: both spectral moments of the null law,
    /// the `(ν, g)` pair resolved from them, and which lane supplied it (#2672).
    pub ref_df_provenance: SmoothLrReferenceDf,
    /// The fixed-λ Lawley LR Bartlett factor `c = 1 + Δε(ρ̂)/d` — the scale the
    /// corrected reference applies to every spectral weight,
    /// `w_j → w_j + Δε(ρ̂)·w_j/d` — when computable, else `1.0` (no correction).
    /// `Δε(ρ̂)` is taken at the λ the reference is built at and vanishes with
    /// the tested block, so `c` stays bounded as `d → 0`.
    pub bartlett_factor: f64,
    /// Corrected statistic `W* = W/c`.
    pub statistic_corrected: f64,
    /// Uncorrected tail probability `P(χ²_ν > W/g)` under the null law's own
    /// two-moment reference.
    pub p_value_uncorrected: f64,
    /// Corrected tail probability `P(χ²_ν > W*/g)`; equals the uncorrected value
    /// when no correction was applied. Scoring `W*` against the reference `L` is
    /// scoring `W` against `c·L`, so the correction composes without a second
    /// convention.
    pub p_value_corrected: f64,
    /// Whether the second-order correction is **material** (#939 deliverable 4):
    /// the per-test diagnostic "is `n` too small for first-order inference
    /// *here*?". `true` when a correction was applied and it moves the result by
    /// more than [`SMOOTH_LR_MATERIAL_THRESHOLD`] — measured as the larger of the
    /// relative Bartlett-factor distance from one `|c − 1|` and the relative
    /// p-value change `|p* − p| / max(p, p*, ε)`. `false` when `correction` is
    /// [`SmoothLrCorrection::None`] (no correction was applied).
    pub material: bool,
    /// Which statistic the corrected p-value is built from.
    pub correction: SmoothLrCorrection,
    /// The CONDITIONAL tail of the corrected statistic — the p-value the
    /// fixed-`λ` law alone would report, before the λ̂-selection replay moves it
    /// (#2672).
    ///
    /// Published so the correction is visible rather than folded in:
    /// `p_value_corrected − p_value_conditional` is exactly what treating `λ̂` as
    /// chosen rather than given is worth on this fit, and it is the quantity a
    /// reader should be shown if they are going to be asked to accept it. Equal
    /// to `p_value_corrected` when no selection was possible.
    pub p_value_conditional: f64,
    /// Certified absolute accuracy of the two published p-values — the larger of
    /// the two bounds [`SmoothLrReferenceDf::tail_probability_with_bound`]
    /// returned (#2672).
    ///
    /// `0.0` on the closed-form lane (a degraded reference without a profiled
    /// scale), where the tail is a chi-square survival function. On the exact
    /// lane it is the inversion's own derived error bound, so a consumer reads
    /// the accuracy rather than inheriting it.
    pub p_value_bound: f64,
    /// [`Self::p_value_corrected`] as the reference can certify it: the point
    /// value when [`Self::p_value_bound`] resolves it away from zero, else an
    /// explicit ceiling `p ≤ bound` — a strong effect's tail sits below the
    /// quadrature's absolute accuracy, and its digits there are roundoff.
    pub p_value: SmoothLrPValue,
}

/// The materiality threshold for [`SmoothTermLrInference::material`] (#939
/// deliverable 4): a correction is flagged material when it changes the result
/// by more than 10%.
pub const SMOOTH_LR_MATERIAL_THRESHOLD: f64 = 0.10;

/// The end-to-end per-term likelihood-ratio significance report for every
/// penalized (shape-unconstrained) smooth term in a fitted model, magically
/// Bartlett-corrected when the family carries closed-form Lawley cumulant jets
/// (#1063, follow-up to #939).
///
/// # Why an LR statistic (not the summary Wald)
///
/// The summary table's `wood_smooth_test` is Wood's rank-truncated **Wald**
/// statistic `T = β̂'Σ̂⁻β̂`. Lawley's ε corrects the **likelihood-ratio**
/// statistic, and under penalization the Wald form is already a weighted χ²
/// whose second-order mean is *not* `d + Δε` — dividing `T` by the LR factor
/// would correct the wrong statistic. The principled route (#1063 Option 1) is
/// to compute a real per-term LR statistic by a constrained refit and correct
/// *that*:
///
/// ```text
/// W = 2(ℓ_full − ℓ_null),   W* = W / c,   c = 1 + Δε/d,   p = P(χ²_d > W*).
/// ```
///
/// # Method
///
/// 1. Fit the full model and read `ℓ_full` and the per-term coefficient ranges /
///    EDF / influence block. The full design's column layout fixes the tested
///    block for the Lawley factor.
/// 2. For each penalized smooth term, fit the nested null model: the full
///    design and likelihood with that term's coefficient block fixed at zero and
///    every surviving smoothing parameter held at the full fit's `λ̂`
///    ([`gam_solve::estimate::fit_nested_at_fitted_log_lambdas`]). Re-selecting
///    `λ` for the reduced model would make `W` the difference of two REML
///    optima, which at a null-railed term is the outer search's tolerance and
///    not a likelihood ratio. `W = 2(ℓ_full − ℓ_null)`.
/// 3. The reference d.f. `d` is the Wood truncation `tr(F)²/tr(F²)` on the
///    term's influence block (the same `ref_df` the summary Wald row reports),
///    floored at `max(edf, null_dim, 1)`: this LR test drops the whole term, so
///    `d` is at least the dimension the term spans when present (its null-space
///    dimension, never below 1). The non-symmetric `tr(F²)` can collapse toward
///    0 at a shrunk-to-null fit and violate that bound — see the inline note at
///    the `ref_df` binding.
/// 4. When the family has closed-form cumulant jets, evaluate Lawley's ε at the
///    **null** linear predictor (an expectation evaluated at the null fit), fold
///    the full λ-scaled penalty `S_λ` into the information, and Bartlett-correct
///    `W` with [`gam_terms::inference::lawley::lawley_lr_bartlett_factor`]. The
///    null annihilates the tested block's penalty (`S_λ β₀ = 0` on that block),
///    so the penalized Lawley expansion applies verbatim.
/// 5. Otherwise (no closed-form jets) the uncorrected `χ²_d` stands with
///    provenance `none` — never weakened.
///
/// # Output
///
/// Exactly one [`SmoothTermLrReport`] per tested smooth term, in term order.
/// Its outcome is either the inference — whose [`SmoothTermLrInference::p_value`]
/// is a resolved value or an explicit ceiling — or the typed
/// [`SmoothLrUnavailable`] reason the term has none: a reduced model whose fit
/// did not converge, a degenerate reference, and so on. A failure of one term's
/// refit is that term's reason and never the call's error; `Err` is reserved
/// for inputs that break the driver's own invariants.
///
/// Shape-constrained smooths have no calibrated LR reference, and
/// [`smooth_term_lr_unavailable_forspec`] names them with the typed reason,
/// matching the summary table's policy, so they get no report here.
pub fn smooth_term_lr_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<Vec<SmoothTermLrReport>, EstimationError> {
    use gam_terms::inference::lawley::{
        LAWLEY_PAIR_MATRIX_MAX_ROWS, known_scale_expected_jets_with_dispersion,
        lawley_lr_bartlett_factor,
    };

    let n = data.nrows();
    // Full fit: ℓ_full, the per-term coefficient ranges/EDF/influence, and the
    // full design whose column layout fixes each tested block for Lawley.
    //
    // A failure here is every term's reason, reported per term like any other:
    // the call's contract is one row per smooth, each a p-value or a reason.
    let full = match fit_term_collection_forspec(
        data,
        y,
        weights,
        offset,
        resolvedspec,
        family.clone(),
        options,
    ) {
        Ok(full) => full,
        Err(error) => {
            let message = error.to_string();
            return Ok(resolvedspec
                .smooth_terms
                .iter()
                .enumerate()
                .map(|(term_idx, term)| SmoothTermLrReport {
                    name: term.name.clone(),
                    term_idx,
                    outcome: Err(SmoothLrUnavailable::FullRefitFailed(message.clone())),
                })
                .collect());
        }
    };
    let ll_full = full.fit.log_likelihood;
    let p_total = full.design.design.ncols();
    let lambdas = full.fit.lambdas.as_slice().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth_term_lr_inference: non-contiguous lambda vector".to_string(),
        )
    })?;
    let s_lambda = weighted_blockwise_penalty_sum(&full.design.penalties, lambdas, p_total);
    // Full design as a dense n×p array for the Lawley pair-matrix reduction.
    let full_design_dense = full.design.design.to_dense();
    let influence = full.fit.coefficient_influence();
    // `H⁻¹`, unscaled: `beta_covariance()` publishes `Vb = H⁻¹·scale`, and the
    // scale is the family's own documented coefficient-covariance multiplier
    // (`σ̂²` for the profiled Gaussian, `1` for every family whose IRLS weight
    // already carries the dispersion). The null spectrum is `1 − eig(H⁻¹_jj
    // S_jj)²`, a product of two matrices in reciprocal units, so the multiplier
    // has to come off exactly here or every weight is wrong by that factor.
    // A family with no scalar multiplier (custom/GAMLSS) yields `None` and the
    // reference drops to the two-moment rung, which needs no scale at all.
    let hessian_inverse = full
        .fit
        .coefficient_covariance_scale()
        .ok()
        .filter(|scale| scale.is_finite() && *scale > 0.0)
        .zip(full.fit.beta_covariance())
        .map(|(scale, covariance)| covariance.mapv(|value| value / scale));
    // `SmoothTerm::coeff_range` is BLOCK-LOCAL — 0-based within the smooth block
    // — while the global coefficient layout is `[intercept | linear | random |
    // smooth]`. Every consumer that indexes a global object with it has to shift
    // by `smooth_start` first (`smooth_term_summary.rs`, the constraint audit and
    // the anisotropic provider all do). This driver did not, and it indexes FOUR
    // global objects with it: the influence matrix `F` (both the per-term EDF
    // trace and Wood's `edf1`), the weighted Gram and correction inside the WPS
    // trace, and — worst — the `tested` column set handed to Lawley, which
    // decides WHICH HYPOTHESIS the mean shift is computed for.
    //
    // This is the #1360 defect in a fourth place: the window slides one column
    // per preceding parametric column, folding the intercept and the linear
    // terms into the smooth's block and dropping as many real smooth columns off
    // the end. It is never zero — the intercept alone makes `smooth_start ≥ 1`.
    //
    // It was invisible because three of the four consumers were degraded to
    // index-free fallbacks: `coefficient_influence` was `None` on every model
    // with a conditioned parametric column (fixed alongside this, #2672), so
    // `per_term_edf` fell through to the penalty-block-trace channel — which is
    // indexed by PENALTY block, not by coefficient, and is therefore correct —
    // and `wood_reference_df` returned `None` outright. Restoring `F` is what
    // made the offset observable: on this issue's `y ~ x + s(z)` fixture the
    // per-term EDF of a null smooth jumped from `0.054` (penalty-trace channel,
    // correct) to `2.040` (influence trace over columns `0..9`), which is the
    // unpenalized intercept's `1` plus the parametric `x`'s `1` plus the smooth's
    // own `0.04` — the offset read off the arithmetic.
    let smooth_start = p_total.saturating_sub(full.design.smooth.total_smooth_cols());
    let fitted_likelihood = resolved_likelihood_for_fit(&full.fit)?;
    let family_disp = lawley_dispersion_for_family(&fitted_likelihood, &full.fit)?;
    // The estimated-scale channel (#2672), assembled once because every part of
    // it except the deterministic offset is a property of the FULL fit and not
    // of which term is being tested.
    //
    // The observation count is the POSITIVE-WEIGHT row count, not the raw one.
    // That is the count the optimizer's own `φ̂ = weighted_rss/(n − edf)` uses
    // (#584: a zero-weight row is exactly an absent row, and counting it in the
    // denominator while the numerator excludes it biases `φ̂` low), and it is
    // also the count that multiplies `ln σ̂²` in the log-likelihood, since a
    // zero-weight row's `−½(… − ln w_i …)` term is not summable at all. The two
    // have to be the same number or `W = n·ln(D_0/D_f) + B` is not an identity.
    let profiled_observations = weights.iter().filter(|weight| **weight > 0.0).count();
    let profiled_residual_shares = profiled_scale_residual_shares(
        &fitted_likelihood,
        hessian_inverse.as_ref(),
        &s_lambda,
        p_total,
    )?;
    let full_residual_df = profiled_residual_shares
        .as_ref()
        .and(profiled_residual_degrees_of_freedom(
            &full.fit,
            profiled_observations,
        ));
    // `(v, h)`: the non-trivial residual weights and the degrees of freedom of
    // the weight-one block. On the exact rung that block is the `n − p`
    // directions no column reaches; on the summary rung the whole residual law
    // is folded into it at the fit's own `ν`.
    // A profiled family's REML criterion selects every term's `λ` through
    // `m·ln D_p` with `m = n − M_p`, `M_p = p − r` the balanced penalty's
    // structural null space, so the replay has to select through the same
    // residual (#2672). Only the exact rung carries a replay: the summary rung
    // has no Hessian inverse to whiten a term by.
    let selection_profile = match profiled_residual_shares.as_ref() {
        Some(Some(_)) => {
            let penalized_rank = gam_terms::construction::balanced_penalty_structural_rank(
                full.design
                    .penalties
                    .iter()
                    .map(|block| (block.local.view(), block.col_range.clone())),
                p_total,
            )?;
            Some(SmoothLrSelectionProfile {
                residual_unit_dimension: profiled_observations.saturating_sub(p_total),
                penalized_rank,
            })
        }
        _ => None,
    };
    // `D_p(λ̂) = D + β̂ᵀS_λβ̂`, the deviance the profiled REML criterion takes
    // the logarithm of.
    let full_penalized_deviance = selection_profile.and_then(|_| {
        let beta = full.fit.beta_flat();
        (beta.len() == s_lambda.nrows()).then(|| full.fit.deviance + beta.dot(&s_lambda.dot(&beta)))
    });
    let profiled_residual = profiled_residual_shares.zip(full_residual_df).map(
        |(shares, residual_df)| match shares {
            Some(spectrum) => (spectrum, profiled_observations.saturating_sub(p_total) as f64),
            None => (Vec::new(), residual_df),
        },
    );

    // The nested null of every term is the full problem with one block fixed at
    // zero, solved at the full fit's `ρ̂` on the offset the full fit was solved
    // with. The bounded-linear route solves a different problem (a box on the
    // bounded coefficients) that this nested fit does not reproduce.
    let null_offset = full
        .design
        .compose_offset(offset, "smooth likelihood-ratio null model")
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let bounded_linear = resolvedspec.has_bounded_linear_terms();
    let nested_inputs = gam_solve::estimate::NestedFixedLambdaInputs {
        design: &full.design.design,
        y,
        weights,
        offset: null_offset.view(),
        penalties: &full.design.penalties,
        nullspace_dims: &full.design.nullspace_dims,
        linear_constraints: full.design.linear_constraints.as_ref(),
        fit: &full.fit,
        tol: options.tol,
        link_shape_estimated: options.optimize_sas || options.optimize_mixture,
    };

    let mut out = Vec::<SmoothTermLrReport>::new();
    for (term_idx, design_term) in full.design.smooth.terms.iter().enumerate() {
        let report = |outcome| SmoothTermLrReport {
            name: design_term.name.clone(),
            term_idx,
            outcome,
        };
        let penalty_range = full
            .design
            .smooth_term_penalty_range(term_idx)
            .map_err(EstimationError::InvalidInput)?;
        let (block_start, k) = penalty_range
            .map(|range| (range.start, range.len()))
            .unwrap_or((0, 0));
        // Shape-constrained smooths have no calibrated LR reference; they are
        // reported by `smooth_term_lr_unavailable_forspec` instead.
        if gam_solve::estimate::smooth_pvalue_unavailable(&design_term.shape).is_some() {
            continue;
        }
        // Shifted into the GLOBAL coefficient layout — see `smooth_start` above.
        let coeff_range = (smooth_start + design_term.coeff_range.start)
            ..(smooth_start + design_term.coeff_range.end);
        if coeff_range.start >= coeff_range.end || coeff_range.end > p_total {
            out.push(report(Err(SmoothLrUnavailable::EmptyCoefficientBlock)));
            continue;
        }
        // Per-term EDF for the χ² reference df FALLBACK (used only when the
        // influence matrix `F` is unavailable). Route through `per_term_edf`,
        // which uses the ADDITIVE per-block trace channel
        // (`|coeff_range| − Σ_{kk∈term} tr_kk`) and caps at the model total,
        // rather than the raw `edf_by_block` block-sum `Σ_{kk}(rank_kk − tr_kk)`.
        // For a multi-penalty term (te/ti/double-penalty) the penalties share one
        // coefficient range, so the rank-based block-sum OVER-COUNTS the term EDF
        // (Σ rank_kk > |coeff_range|) and would inflate the LR reference df,
        // biasing the smooth-term test conservative on large/sparse fits where `F`
        // is not materialised. (Same per-block over-count class as the multinomial
        // `edf_per_class` fix.)
        let edf = full.fit.per_term_edf(coeff_range.clone(), block_start, k);
        // The term's **joint** unpenalized null-space dimension: the coefficient
        // directions penalized by *no* active penalty — the polynomial part a
        // penalized smooth always carries when present, which no penalty can
        // shrink. This is `dim(∩_k null(S_k)) = p_local − rank(Σ_k S_k)`, the
        // INTERSECTION of the per-penalty null spaces, computed by
        // `wald_unpenalized_dim()` — the very same scalar the summary Wald test
        // (`wood_smooth_test`) floors its reference d.f. at, so the LR and Wald
        // tests reference a consistent d.f.
        //
        // It must NOT be `nullspace_dims.iter().sum()`: that *unions* the null
        // spaces (the #1360 defect — see `joint_unpenalized_dim`'s docs). A
        // double-penalty smooth carries a bending penalty (null space = its
        // polynomial part) plus a complementary null-space ridge (which penalizes
        // exactly that polynomial part), so the two null spaces are disjoint and
        // the joint null space is EMPTY (dim 0) — yet the per-penalty dims sum to
        // ~`p_local`. Flooring `ref_df` at that sum pins it to the full basis
        // dimension for every fit (e.g. 11 for a k=12 s(x)), making the LR test
        // badly conservative for genuine moderate signals while only accidentally
        // masking the collapse.
        let null_dim = design_term.wald_unpenalized_dim();
        // The reference the whole-term LR statistic is scored against: the first
        // two moments of its OWN null law, not a chi-square fitted to its mean.
        // See `lr_null_reference` for the derivation and for what this replaced.
        // The term's own penalties restricted to the tested coefficient block,
        // carried λ-FREE with their `ρ̂_i = ln λ̂_i` alongside. The separation is
        // not bookkeeping: the replay's criterion needs `log|Σ_i λ_i S_i|₊`, and
        // that quantity is only computable from the components and their scales
        // — an assembled sum has already lost it (#2644). See
        // [`SelectionGeometry`].
        let mut term_penalties = Vec::<Array2<f64>>::new();
        let mut term_log_lambda = Vec::<f64>::new();
        for (blockwise, &lambda) in full
            .design
            .penalties
            .get(block_start..block_start + k)
            .into_iter()
            .flatten()
            .zip(lambdas[block_start..(block_start + k).min(lambdas.len())].iter())
        {
            let range = &blockwise.col_range;
            if range.start < coeff_range.start || range.end > coeff_range.end {
                continue;
            }
            if !(lambda.is_finite() && lambda > 0.0) {
                continue;
            }
            let mut local = Array2::<f64>::zeros((coeff_range.len(), coeff_range.len()));
            let offset = range.start - coeff_range.start;
            let width = range.end - range.start;
            for row in 0..width {
                for column in 0..width {
                    local[[offset + row, offset + column]] = blockwise.local[[row, column]];
                }
            }
            term_penalties.push(local);
            term_log_lambda.push(lambda.ln());
        }
        // ONE window per scale. The outer search moved each `ρ_i` independently
        // inside its box, so scale `i` could reach `ln t_i ∈ [−B − ρ̂_i, B − ρ̂_i]`
        // and no further. The single COMMON-shift window this used to compute is
        // the intersection of those intervals: correct for a slice that moves
        // every scale together (which is what `generate_common_scale` still
        // derives from these), and wrong for a grid that moves them separately.
        // On a null-true double-penalty smooth at `ρ̂ = (18, −24)` each axis has
        // ~50 of room and the intersection leaves 18; when one `λ̂` rails the
        // intersection is EMPTY and the replay was declined outright.
        // #2812: each term's log-scale window is its own resolvability
        // interval, `[ln(ε γ_min), ln(γ_max / ε)]` over the design-relative
        // penalty spectrum on the tested block, expressed relative to `ρ̂`.
        // Below it the term is unpenalized to working precision, above it the
        // term sits on its null space; the replay has nothing to explore past
        // either edge. A term whose geometry cannot be projected keeps the
        // precision box around unit strength.
        let block_gram = {
            let dense = full.design.design.to_dense();
            let block = dense.slice(ndarray::s![.., coeff_range.start..coeff_range.end]);
            block.t().dot(&block)
        };
        let log_scale_windows: Vec<(f64, f64)> = term_penalties
            .iter()
            .zip(term_log_lambda.iter())
            .map(|(local, &rho)| {
                let interval =
                    gam_solve::estimate::rho_domain::penalty_range_gammas_from_gram(
                        &block_gram,
                        local,
                    )
                    .as_deref()
                    .and_then(gam_solve::estimate::rho_domain::resolvability_interval);
                let (lo, hi) = gam_solve::estimate::rho_domain::coordinate_domain(interval, None);
                (lo - rho, hi - rho)
            })
            .collect();
        // Null model: this term's block fixed at zero, at the full fit's `ρ̂`.
        // It is solved before the reference because the selection replay
        // scores the OBSERVED data too: the tested block's score at the nested
        // null, `g_j = X_jᵀ ∂ℓ/∂η`, is the draw the replay's own `z` stands in
        // for (see `SmoothLrSelectionReplay::observed`). The reasons keep their
        // precedence — a degenerate reference is reported before the null fit's.
        let null_outcome = if bounded_linear {
            None
        } else {
            Some(gam_solve::estimate::fit_nested_at_fitted_log_lambdas(
                &nested_inputs,
                coeff_range.clone(),
            )?)
        };
        // The score is on the replay's unit-dispersion scale. Every family
        // whose working weight already carries the dispersion reads it off the
        // reporting likelihood directly; the profiled Gaussian's score is
        // `X_jᵀ W (y − μ̂₀)` (unit `φ`), and dividing by `√(D_f / E[V])` puts
        // it on the unit scale the replay's draws are on. The profiled
        // criterion's argmin is invariant to that common scale, so all it has
        // to do is put the score and the observed penalized deviance in the
        // same units.
        let mut observed_penalized_deviance = None;
        let observed_score = match null_outcome.as_ref() {
            Some(gam_solve::estimate::NestedFixedLambdaOutcome::Converged(null)) => {
                let block_score = full_design_dense
                    .slice(ndarray::s![.., coeff_range.start..coeff_range.end])
                    .t()
                    .dot(&null.eta_score);
                let unit_scale = match profiled_residual.as_ref() {
                    Some((residual_weights, residual_unit_dimension)) => {
                        let expected_residual =
                            residual_weights.iter().sum::<f64>() + residual_unit_dimension;
                        full.fit.deviance / expected_residual
                    }
                    None => 1.0,
                };
                // A profiled selection is driven by the whole penalized
                // deviance, so the observation carries its own, in the same
                // units as its score's squares.
                if selection_profile.is_some() && unit_scale.is_finite() && unit_scale > 0.0 {
                    observed_penalized_deviance = full_penalized_deviance.map(|deviance| deviance / unit_scale);
                }
                (unit_scale.is_finite() && unit_scale > 0.0)
                    .then(|| block_score.mapv(|value| value / unit_scale.sqrt()))
            }
            _ => None,
        };
        let reference = lr_null_reference(
            influence,
            hessian_inverse.as_ref(),
            Some(&s_lambda),
            &coeff_range,
            edf,
            null_dim,
            &log_scale_windows,
            &term_penalties,
            &term_log_lambda,
            selection_profile,
            observed_score.as_ref(),
            observed_penalized_deviance,
        );
        let mut reference = reference;
        let ref_df = reference.mean;
        if !(ref_df.is_finite()
            && ref_df > 0.0
            && reference.chi_square_df.is_finite()
            && reference.chi_square_df > 0.0
            && reference.scale.is_finite()
            && reference.scale > 0.0)
        {
            out.push(report(Err(SmoothLrUnavailable::DegenerateReference)));
            continue;
        }

        let Some(null_outcome) = null_outcome else {
            out.push(report(Err(SmoothLrUnavailable::NullFitUnsupported(
                "the model has bounded linear terms, whose box-constrained fit the \
                 nested fixed-lambda null does not reproduce"
                    .to_string(),
            ))));
            continue;
        };
        let null = match null_outcome {
            gam_solve::estimate::NestedFixedLambdaOutcome::Converged(null) => null,
            gam_solve::estimate::NestedFixedLambdaOutcome::NotConverged(message) => {
                out.push(report(Err(SmoothLrUnavailable::NullFitNotConverged(message))));
                continue;
            }
            gam_solve::estimate::NestedFixedLambdaOutcome::Unsupported(message) => {
                out.push(report(Err(SmoothLrUnavailable::NullFitUnsupported(message))));
                continue;
            }
        };
        if !null.log_likelihood.is_finite() {
            out.push(report(Err(SmoothLrUnavailable::NullLogLikelihoodNotFinite)));
            continue;
        }
        let log_likelihood_ratio = 2.0 * (ll_full - null.log_likelihood);
        // η at the null fit, offset included: Lawley reads it on the full
        // design's rows.
        let eta_null = Some(null.eta);
        let null_residual_df = null.profiled_residual_df;

        // The estimated-scale channel needs BOTH fits' residual degrees of
        // freedom, so it is completed here rather than where the rest of the
        // reference was built: `B = n·ln(ν_f/ν_0) + (ν_0 − ν_f)` is the only
        // part of `W` that is a function of the null refit and of nothing
        // random (#2672).
        if let Some((residual_weights, residual_unit_dimension)) = profiled_residual.as_ref()
            && let (Some(full_df), Some(null_df)) = (full_residual_df, null_residual_df)
            && full_df > 0.0
            && null_df > 0.0
        {
            let observations = profiled_observations as f64;
            reference.profiled_scale = Some(SmoothLrProfiledScale {
                observations,
                deterministic_offset: observations * (full_df / null_df).ln()
                    + (null_df - full_df),
                residual_weights: residual_weights.clone(),
                residual_unit_dimension: *residual_unit_dimension,
            });
        }
        // The statistic's support is the reference's. A known-scale `W` is a
        // non-negative combination of chi-squares, so a negative value is the
        // two fits' optimizer noise and zero is the same event. A profiled `W`
        // is `n·ln(1 + Q/V) + B`, whose support starts at `B`, and `B < 0`
        // whenever the full fit spends any residual degree of freedom the null
        // does not (`n·ln x < n(x − 1) ≤ ν_0(x − 1)` for `x = ν_f/ν_0 < 1`,
        // since `ν_0 ≤ n`): a `W` in `(B, 0)` is an ordinary null draw, and moving
        // it to zero scored it as `P(W > 0)` — an atom near 0.6 carrying the
        // half of the null replicates whose term REML shrinks away.
        let statistic_lr = if reference.profiled_scale.is_some() {
            log_likelihood_ratio
        } else {
            log_likelihood_ratio.max(0.0)
        };
        let ref_df_provenance = reference.clone();

        let (p_uncorrected, mut p_bound) = reference.tail_probability_with_bound(statistic_lr);
        let mut p_conditional = reference.conditional_tail_probability(statistic_lr);

        // Magic Bartlett correction: only when the LR statistic is finite, the
        // family has closed-form jets, n is in the resolvable regime, and the
        // factor is computable. Otherwise the uncorrected χ² stands.
        let mut bartlett_factor = 1.0;
        let mut statistic_corrected = statistic_lr;
        let mut p_corrected = p_uncorrected;
        let mut correction = SmoothLrCorrection::None;
        if let (Some(eta), true, true) = (
            eta_null.as_ref(),
            statistic_lr.is_finite(),
            n <= LAWLEY_PAIR_MATRIX_MAX_ROWS,
        ) {
            let kappas: Option<Vec<_>> = (0..n)
                .map(|i| {
                    known_scale_expected_jets_with_dispersion(
                        &fitted_likelihood.spec,
                        eta[i],
                        family_disp,
                    )
                    .and_then(|jets| jets.kappas().ok())
                })
                .collect();
            if let Some(kappas) = kappas {
                let fixed_factor = lawley_lr_bartlett_factor(
                    full_design_dense.view(),
                    &kappas,
                    Some(s_lambda.view()),
                    coeff_range.clone(),
                    ref_df,
                );
                if let Ok(c_cond) = fixed_factor
                    && c_cond.is_finite()
                    && c_cond > 0.0
                {
                    correction = SmoothLrCorrection::LawleyLrFixedLambda;
                    bartlett_factor = c_cond;
                    statistic_corrected = statistic_lr / c_cond;
                    // Scoring `W/c` against the reference is scoring `W`
                    // against `c·L`: the factor rescales every spectral weight
                    // (the law is exactly scale-equivariant), and the
                    // selection replay inside `L` already carries the λ̂ the
                    // fit could have chosen, so the correction composes with
                    // it without a second convention. `c = 1 + Δε/d` is a ratio
                    // of two quantities taken at the same λ, so it stays
                    // bounded as a shrunk term's `d → 0`.
                    let (corrected, corrected_bound) =
                        reference.tail_probability_with_bound(statistic_corrected);
                    p_corrected = corrected;
                    p_conditional = reference.conditional_tail_probability(statistic_corrected);
                    p_bound = p_bound.max(corrected_bound);
                }
            }
        }

        // Materiality (#939 deliverable 4): only when a correction was actually
        // applied, flagged when it moves the result by more than the 10%
        // threshold — by the Bartlett factor's distance from one OR the relative
        // p-value shift, whichever is larger (a factor near one can still flip a
        // p-value sitting on the α boundary, and vice versa).
        let material = match correction {
            SmoothLrCorrection::LawleyLrFixedLambda => {
                let factor_move = (bartlett_factor - 1.0).abs();
                let p_denom = p_uncorrected.max(p_corrected);
                // Two zero p-values have not moved.
                let p_move = if p_uncorrected.is_finite() && p_corrected.is_finite() && p_denom > 0.0
                {
                    (p_corrected - p_uncorrected).abs() / p_denom
                } else {
                    0.0
                };
                factor_move > SMOOTH_LR_MATERIAL_THRESHOLD || p_move > SMOOTH_LR_MATERIAL_THRESHOLD
            }
            SmoothLrCorrection::None => false,
        };

        let Some(p_value) = SmoothLrReferenceDf::typed_p_value(p_corrected, p_bound) else {
            out.push(report(Err(SmoothLrUnavailable::TailNotComputable)));
            continue;
        };

        out.push(report(Ok(SmoothTermLrInference {
            name: design_term.name.clone(),
            term_idx,
            statistic_lr,
            ref_df,
            ref_df_provenance,
            bartlett_factor,
            statistic_corrected,
            p_value_uncorrected: p_uncorrected,
            p_value_corrected: p_corrected,
            material,
            correction,
            p_value_conditional: p_conditional,
            p_value_bound: p_bound,
            p_value,
        })));
    }
    Ok(out)
}

/// The residual degrees of freedom the fit's profiled `σ̂` ACTUALLY divided by,
/// read back as `D/σ̂²` from two published fields.
///
/// Not recomputed as `n − edf_total`. That is what the optimizer uses on one of
/// its two branches — the other, taken when inference is off, divides by `n` —
/// and a reference that assumed the branch would be silently wrong on the other
/// one. `D` is the weighted residual sum of squares (the Gaussian deviance) and
/// `σ̂² = D/ν` by construction, so `ν = D/σ̂²` inverts the fit's own convention
/// whatever it was.
///
/// The inversion is exact rather than approximate, and the reason is a
/// type-level one: `ProfiledGaussian` implies the IDENTITY link, because
/// `gam_spec`'s `legal_cell_kind` admits no other Gaussian cell — a
/// `(Gaussian, log)` model is not constructible rather than merely unusual. The
/// optimizer's weighted-RSS channel is wired for the identity link, so on every
/// fit that reaches this function `D` IS `Σ w_i(y_i − μ_i)²` and `σ̂² = D/ν`
/// holds by construction.
///
/// `None` when the two fields cannot produce a residual degrees of freedom that
/// is finite, positive, and no larger than the sample size — a statement that
/// this fit's `σ̂` is not the profiled residual one the identity above assumes,
/// which is exactly when the estimated-scale channel must stay switched off.
fn profiled_residual_degrees_of_freedom(fit: &UnifiedFitResult, n: usize) -> Option<f64> {
    let variance = fit.standard_deviation * fit.standard_deviation;
    if !(variance.is_finite() && variance > 0.0 && fit.deviance.is_finite() && fit.deviance > 0.0) {
        return None;
    }
    let residual_df = fit.deviance / variance;
    // The `n` ceiling carries a tolerance because `ν = D/σ̂²` is a ratio of two
    // separately-rounded published numbers, and the branch that divides by `n`
    // exactly lands on the ceiling.
    let ceiling = n as f64 * (1.0 + 8.0 * f64::EPSILON);
    (residual_df.is_finite() && residual_df > 0.0 && residual_df <= ceiling).then_some(residual_df)
}

/// The residual quadratic form's spectrum, `v_i = p_i²` over the WHOLE model's
/// penalty shares — or `None` when the family does not profile a Gaussian scale
/// out of a residual sum of squares.
///
/// This is `lr_tested_block` over the full coefficient range rather than over
/// a term's: the same self-adjoint decomposition and the same `[0, 1]` shares,
/// deliberately not a second route to the same object. See
/// [`SmoothLrProfiledScale`] for why `V ~ Σ_i p_i²·χ²_1 + χ²_{n−p}`.
///
/// `Some(None)` — the family profiles a scale but the shares are unreachable —
/// is a rung, not a refusal, on the same ladder the numerator's reference
/// already has: the fit that cannot publish `H⁻¹` is the fit whose numerator is
/// also a two-moment summary. What the caller does with it is fold the whole
/// residual law into `V ~ χ²_{ν}` at the fit's own `ν`, which is mgcv's `F`
/// reference exactly. Two things are then inexact rather than exact, both
/// bounded by `Σ_i f_i(1 − f_i)` — the count of PARTIALLY shrunk directions,
/// zero at both ends of the shrinkage range: `E[V] = tr((I−A)²) = ν −
/// Σ f(1−f)` rather than `ν`, and `Var(V)/2 = Σv² ≤ Σv` so `χ²_ν`
/// over-disperses. That is strictly better than the known-scale reference,
/// which is what the alternative — dropping the channel — would silently
/// restore.
fn profiled_scale_residual_shares(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    hessian_inverse: Option<&Array2<f64>>,
    penalty: &Array2<f64>,
    p_total: usize,
) -> Result<Option<Option<Vec<f64>>>, EstimationError> {
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    if !matches!(resolved, gam_spec::ResolvedLikelihoodScale::ProfiledGaussian) {
        return Ok(None);
    }
    Ok(Some(
        lr_tested_block(hessian_inverse, Some(penalty), &(0..p_total))
            .map(|block| block.shares.iter().map(|share| share * share).collect()),
    ))
}

fn resolved_likelihood_for_fit(
    fit: &UnifiedFitResult,
) -> Result<gam_spec::GlmLikelihoodSpec, EstimationError> {
    let spec = fit.likelihood_family.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth-term LR inference requires an engine-level GLM likelihood".to_string(),
        )
    })?;
    gam_spec::GlmLikelihoodSpec::try_new(spec.clone(), fit.likelihood_scale.clone())
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))
}

/// The response dispersion `phi` Lawley needs for cumulant scaling. This is
/// deliberately distinct from the coefficient-covariance multiplier used by
/// the WPS trace below: Gamma Lawley uses `1 / shape`, while its PIRLS Hessian
/// already carries `shape` and therefore has covariance multiplier one.
fn lawley_dispersion_for_family(
    likelihood: &gam_spec::GlmLikelihoodSpec,
    fit: &UnifiedFitResult,
) -> Result<f64, EstimationError> {
    let profiled_standard_deviation = matches!(
        likelihood
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        gam_spec::ResolvedLikelihoodScale::ProfiledGaussian
    )
    .then_some(fit.standard_deviation);
    gam_solve::estimate::dispersion_from_likelihood(likelihood, profiled_standard_deviation)
        .map(|dispersion| dispersion.phi())
}

/// The reference distribution for the whole-term LR statistic: its own null
/// spectrum `w` when that is recoverable, and the two-moment summary of it when
/// only the moments are.
///
/// The derivation, and why the spectrum rather than two of its moments, is on
/// [`SmoothLrReferenceDf`]. What is worth stating at the code is the ladder, and
/// that each rung is a strictly weaker instrument on the SAME quantity rather
/// than a different claim:
///
/// 1. **The spectrum** (`lr_tested_block`) — needs `[H⁻¹]_jj`
///    and the term's λ-weighted penalty block. Exact.
/// 2. **Its first two moments** ([`lr_null_spectral_moments`]) — needs only the
///    coefficient-influence block, because with `A = 2F − F²`
///
///    ```text
///    Σ w   = tr A  = 2·tr F − tr F²
///    Σ w²  = tr A² = 4·tr F² − 4·tr F³ + tr F⁴
///    ```
///
///    are traces of powers of one `q × q` block. Reading the weights THEMSELVES
///    off `F_jj` is what rung 1 avoids: `F_jj = H̃⁻¹Ĩ_jj` is not symmetric, so it
///    would need a general eigensolver, while rung 1 reaches the same spectrum
///    through a self-adjoint one.
/// 3. **A scalar EDF** — `χ²_{max(edf, null_dim, 1)}`, the unit-weight shape.
///
/// The lane taken is tagged in the returned provenance, so a consumer can tell
/// an exact reference from a summary of one instead of inferring it from the
/// numbers.
///
/// `term_penalties` are the term's λ-FREE penalty components on the tested
/// block and `term_log_lambda` their fitted `ρ̂_i`, carried separately rather
/// than pre-multiplied. That is not bookkeeping: the selection replay's
/// criterion needs `log|Σ_i λ_i S_i|₊`, and an assembled sum has already lost
/// it whenever the `λ_i` separate (#2644 — see [`SelectionGeometry`]).
/// `log_scale_windows` carries ONE window per component, because the outer
/// search moved each `ρ_i` independently inside its own box.
/// `selection_profile` is the residual the replay's selection is driven by when
/// the family profiles its scale ([`SmoothLrSelectionProfile`]), `None` when it
/// does not; `observed_penalized_deviance` is then the full fit's `D_p(λ̂)` in
/// the observed score's units, which selects the observation by the same
/// profiled rule ([`ObservedDraw`]).
fn lr_null_reference(
    influence: Option<&Array2<f64>>,
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
    edf: f64,
    null_dim: usize,
    log_scale_windows: &[(f64, f64)],
    term_penalties: &[Array2<f64>],
    term_log_lambda: &[f64],
    selection_profile: Option<SmoothLrSelectionProfile>,
    observed_score: Option<&Array1<f64>>,
    observed_penalized_deviance: Option<f64>,
) -> SmoothLrReferenceDf {
    let from_moments = |mean: f64, second_moment: f64, source| SmoothLrReferenceDf {
        weights: Vec::new(),
        mean,
        second_moment,
        chi_square_df: mean * mean / second_moment,
        scale: second_moment / mean,
        moment_residual: None,
        edf,
        null_dim,
        source,
        // The degraded lanes do not have the spectrum, so they cannot have the
        // geometry the replay is built from either.
        selection: SmoothLrSelection::Declined(SmoothLrSelectionDecline::GeometryRefused),
        // Completed by the caller once the null refit has produced the second
        // residual degrees of freedom `B` needs (#2672).
        profiled_scale: None,
    };
    let unit_weight = || {
        let df = edf.max(null_dim as f64).max(1.0);
        from_moments(df, df, SmoothLrReferenceSource::UnitWeightFallback)
    };
    let influence_moments = lr_null_spectral_moments(influence, coeff_range);

    // Rung 1 — the spectrum itself.
    if let Some(block) = lr_tested_block(hessian_inverse, penalty, coeff_range) {
        let mut weights: Vec<f64> = block.shares.iter().map(|&p| 1.0 - p * p).collect();
        weights.sort_by(|a, b| b.partial_cmp(a).expect("finite weights"));
        let mean: f64 = weights.iter().sum();
        let second_moment: f64 = weights.iter().map(|w| w * w).sum();
        if mean.is_finite() && mean > 0.0 && second_moment.is_finite() && second_moment > 0.0 {
            // The identity check, measured rather than assumed. Denominated
            // relatively and floored at one so a term shrunk to nothing does not
            // report a huge residual for a difference of `1e-16`.
            let moment_residual = influence_moments.map(|[trace_mean, trace_second]| {
                let first = (mean - trace_mean).abs() / mean.abs().max(1.0);
                let second = (second_moment - trace_second).abs() / second_moment.abs().max(1.0);
                first.max(second)
            });
            return SmoothLrReferenceDf {
                weights,
                mean,
                second_moment,
                chi_square_df: mean * mean / second_moment,
                scale: second_moment / mean,
                moment_residual,
                edf,
                null_dim,
                source: SmoothLrReferenceSource::NullSpectrum,
                // One entry point for both lanes. `generate` whitens the term's
                // λ-free components by the Schur-complemented information,
                // factors them into roots, and dispatches on how many scales the
                // term actually selects. The generalized spectrum it reports is
                // read off that geometry rather than reconstructed as
                // `p_k/(1 − p_k)` from the penalty shares — a share is a number
                // in `[0, 1]`, so a structural zero and a `1e-17` of roundoff are
                // one machine epsilon apart there, and the criterion's
                // log-determinant is the one place that difference is worth
                // `log(1 + 1e17)`.
                selection: SmoothLrSelectionReplay::generate(
                    &block.whitener,
                    term_penalties,
                    term_log_lambda,
                    log_scale_windows,
                    selection_profile,
                    observed_score,
                    observed_penalized_deviance,
                ),
                profiled_scale: None,
            };
        }
    }

    // Rung 2 — two moments of it, off the influence block.
    let Some([mean, second_moment]) = influence_moments else {
        return unit_weight();
    };
    if !(mean.is_finite() && mean > 0.0 && second_moment.is_finite() && second_moment > 0.0) {
        return unit_weight();
    }
    from_moments(
        mean,
        second_moment,
        SmoothLrReferenceSource::SpectralMomentMatch,
    )
}

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
/// identities in [`lr_null_spectral_moments`] summarise, arrived at without
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
/// The tested block's penalty shares AND the whitener the replay needs, from
/// ONE self-adjoint decomposition and with no matrix cancellation anywhere.
pub(crate) struct LrTestedBlock {
    /// `p = eig(B^{1/2} S_jj B^{1/2}) ∈ [0, 1]`, ascending.
    shares: Vec<f64>,
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
    whitener: Array2<f64>,
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

fn lr_tested_block(
    hessian_inverse: Option<&Array2<f64>>,
    penalty: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
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
    let b = symmetrized(h_inv.slice(s![start..end, start..end]).to_owned());
    let s = symmetrized(s_lambda.slice(s![start..end, start..end]).to_owned());
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
    let whitener = b_root.dot(&whitener);
    if whitener.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let mut shares = shares;
    shares.sort_by(|a, b| a.partial_cmp(b).expect("finite shrinkage"));
    Some(LrTestedBlock { shares, whitener })
}

/// `[tr A, tr A²]` for `A = 2·F_jj − F_jj²` on the tested coefficient block.
///
/// Returns `None` when the influence matrix is absent, the block is outside it,
/// or either trace is non-finite — the caller then falls back to the unit-weight
/// shape rather than scoring against a spectrum it could not compute.
fn lr_null_spectral_moments(
    influence: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
) -> Option<[f64; 2]> {
    let f = influence?;
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end || end > f.nrows() || end > f.ncols() {
        return None;
    }
    let block = f.slice(s![start..end, start..end]).to_owned();
    let squared = block.dot(&block);
    let cubed = squared.dot(&block);
    let quartic = squared.dot(&squared);
    let trace = |m: &Array2<f64>| (0..m.nrows()).map(|i| m[[i, i]]).sum::<f64>();
    let (t1, t2, t3, t4) = (
        trace(&block),
        trace(&squared),
        trace(&cubed),
        trace(&quartic),
    );
    let mean = 2.0 * t1 - t2;
    let second_moment = 4.0 * t2 - 4.0 * t3 + t4;
    (mean.is_finite() && second_moment.is_finite()).then_some([mean, second_moment])
}

#[cfg(test)]
mod lr_null_reference_tests {
    use super::{
        SmoothLrReferenceSource, lr_null_reference, lr_null_spectral_moments,
        lr_tested_block,
    };
    use ndarray::Array2;

    /// No selection window: these unit tests are about the CONDITIONAL law, so
    /// they hold `λ` fixed and the replay is inert. The replay's own behaviour
    /// is pinned separately.
    const WINDOW: &[(f64, f64)] = &[];

    /// `M⁻¹` for a symmetric PD `M`, through the same self-adjoint entry point
    /// the production path uses. The tests need an inverse only to BUILD the two
    /// inputs (`H⁻¹` and `F = H⁻¹(H − S)`) from one `H`; nothing under test reads
    /// it.
    fn symmetric_inverse(matrix: &Array2<f64>) -> Array2<f64> {
        let (values, vectors) =
            gam_linalg::faer_ndarray::strict_symmetric_eigh(matrix, faer::Side::Lower)
                .expect("symmetric PD inverse");
        let mut scaled = vectors.clone();
        for (mut column, &value) in scaled.columns_mut().into_iter().zip(values.iter()) {
            column.mapv_inplace(|entry| entry / value);
        }
        scaled.dot(&vectors.t())
    }

    /// A diagonal influence block has `F_jj` eigenvalues on the diagonal, so the
    /// spectrum is `2f − f²` term by term and both moments are hand-computable.
    /// This is the identity the whole reference rests on; it is checked against
    /// the definition rather than against another implementation of itself.
    #[test]
    fn the_spectral_moments_are_the_weights_of_the_null_law() {
        let f_diag = [0.9_f64, 0.5, 0.2, 0.05];
        let mut influence = Array2::<f64>::zeros((6, 6));
        // Deliberately offset: the block is columns 2..6, and rows/columns
        // outside it carry values that must not leak into either trace.
        influence[[0, 0]] = 7.0;
        influence[[1, 1]] = -3.0;
        influence[[0, 3]] = 11.0;
        influence[[5, 1]] = -2.0;
        for (i, &f) in f_diag.iter().enumerate() {
            influence[[2 + i, 2 + i]] = f;
        }
        let [mean, second] =
            lr_null_spectral_moments(Some(&influence), &(2..6)).expect("moments available");
        let weights: Vec<f64> = f_diag.iter().map(|f| 2.0 * f - f * f).collect();
        let want_mean: f64 = weights.iter().sum();
        let want_second: f64 = weights.iter().map(|w| w * w).sum();
        assert!(
            (mean - want_mean).abs() < 1e-12 && (second - want_second).abs() < 1e-12,
            "moments ({mean}, {second}) vs weights {weights:?} -> ({want_mean}, {want_second})"
        );
    }

    /// THE IDENTITY THE EXACT LANE RESTS ON, on a design where every block is
    /// coupled to every other: the spectrum read off `[H⁻¹]_jj` and `S_jj` has
    /// the same two moments as the spectrum read off the influence block, which
    /// are computed by completely different arithmetic (two self-adjoint
    /// decompositions versus four traces of powers of a non-symmetric matrix).
    ///
    /// `(I − F)_jj = [H⁻¹]_jj S_jj` is only true because the penalty is
    /// block-diagonal by term, so the fixture puts a SEPARATE penalty on the
    /// retained block as well: the identity must survive other terms being
    /// penalized (that is the difference between the Schur complement of the
    /// penalized retained block and of the unpenalized one), and it must fail if
    /// anyone ever lets a penalty couple two terms.
    #[test]
    fn the_penalty_spectrum_and_the_influence_moments_are_the_same_object() {
        let (retained, tested) = (3usize, 5usize);
        let p = retained + tested;
        // A dense SPD Gram with real cross-block coupling.
        let mut gram = Array2::<f64>::zeros((p, p));
        for row in 0..p {
            for col in 0..p {
                gram[[row, col]] = 1.0 / (1.0 + (row as f64 - col as f64).abs())
                    + if row == col { 0.75 } else { 0.0 };
            }
        }
        for lambda in [0.0_f64, 1e-3, 1.0, 25.0, 1e4, 1e7] {
            // Block-diagonal penalty: a second-difference block on the tested
            // term and an unrelated ridge on the retained one.
            let mut penalty = Array2::<f64>::zeros((p, p));
            for row in 0..retained {
                penalty[[row, row]] = 0.3;
            }
            for row in 0..tested.saturating_sub(2) {
                for (offset_a, coefficient_a) in [(0usize, 1.0_f64), (1, -2.0), (2, 1.0)] {
                    for (offset_b, coefficient_b) in [(0usize, 1.0_f64), (1, -2.0), (2, 1.0)] {
                        penalty[[retained + row + offset_a, retained + row + offset_b]] +=
                            lambda * coefficient_a * coefficient_b;
                    }
                }
            }
            let hessian = &gram + &penalty;
            let hessian_inverse = symmetric_inverse(&hessian);
            let influence = hessian_inverse.dot(&gram);

            let weights =
                lr_tested_block(Some(&hessian_inverse), Some(&penalty), &(retained..p))
                    .map(|block| block.shares)
                    .map(|shares| shares.iter().map(|&q| 1.0 - q * q).collect::<Vec<f64>>())
                    .expect("spectrum available");
            let [mean, second] = lr_null_spectral_moments(Some(&influence), &(retained..p))
                .expect("moments available");
            let spectrum_mean: f64 = weights.iter().sum();
            let spectrum_second: f64 = weights.iter().map(|w| w * w).sum();
            // `1e-7` relative, not roundoff: at `λ = 1e7` the INFLUENCE route
            // is what loses the digits — `2·trF − trF²` differences two nearly
            // equal quantities while `trF → 0` — and it comes in at `2.5e-9`
            // relative there against `<1e-15` at every smaller `λ`. That
            // asymmetry is one of the reasons the penalty route is the primary
            // one; the bar is set where the WEAKER of the two routes lives.
            assert!(
                (spectrum_mean - mean).abs() < 1e-7 * mean.abs().max(1.0)
                    && (spectrum_second - second).abs() < 1e-7 * second.abs().max(1.0),
                "lambda={lambda}: spectrum moments ({spectrum_mean}, {spectrum_second}) \
                 disagree with influence-trace moments ({mean}, {second})"
            );
            assert!(
                weights.iter().all(|w| (0.0..=1.0).contains(w)),
                "lambda={lambda}: weights escaped [0,1]: {weights:?}"
            );
            assert!(
                weights.windows(2).all(|pair| pair[0] >= pair[1]),
                "lambda={lambda}: weights are not sorted descending: {weights:?}"
            );
        }
    }

    /// Each rung of the ladder degrades to the next and SAYS so. A consumer that
    /// cannot tell an exact reference from a summary of one, or a summary from a
    /// scalar-EDF fallback, cannot reason about the number it was handed.
    #[test]
    fn each_missing_input_degrades_exactly_one_rung_and_visibly() {
        let q = 4;
        let influence = Array2::<f64>::eye(q) * 0.5;
        let hessian_inverse = Array2::<f64>::eye(q);
        let penalty = Array2::<f64>::eye(q) * 0.5;

        // Everything present: the exact lane, carrying weights.
        let exact = lr_null_reference(
            Some(&influence),
            Some(&hessian_inverse),
            Some(&penalty),
            &(0..q),
            2.0,
            1,
            WINDOW,
            &[],
            &[],
            None,
            None,
            None,
        );
        assert_eq!(exact.source, SmoothLrReferenceSource::NullSpectrum);
        assert_eq!(exact.weights.len(), q);

        // No `H⁻¹` (or no penalty): the moments off `F`, and NO weights — which
        // is exactly the condition `tail_probability_with_bound` switches on.
        for degraded in [
            lr_null_reference(Some(&influence), None, Some(&penalty), &(0..q), 2.0, 1, WINDOW, &[], &[], None, None, None),
            lr_null_reference(
                Some(&influence),
                Some(&hessian_inverse),
                None,
                &(0..q),
                2.0,
                1,
                WINDOW,
                &[],
                &[],
                None,
                None,
                None,
            ),
        ] {
            assert_eq!(degraded.source, SmoothLrReferenceSource::SpectralMomentMatch);
            assert!(degraded.weights.is_empty());
            assert!((degraded.mean - exact.mean).abs() < 1e-12);
        }

        // Nothing at all: the unit-weight shape with its `max(edf, null_dim, 1)`.
        let fallback = lr_null_reference(None, None, None, &(0..q), 2.5, 1, WINDOW, &[], &[], None, None, None);
        assert_eq!(fallback.source, SmoothLrReferenceSource::UnitWeightFallback);
        assert!(fallback.weights.is_empty());
        assert_eq!(fallback.chi_square_df, 2.5);
        assert_eq!(fallback.scale, 1.0);
        // The `max(edf, null_dim, 1)` shape is retained only on this lane.
        assert_eq!(
            lr_null_reference(None, None, None, &(0..4), 0.01, 3, WINDOW, &[], &[], None, None, None).chi_square_df,
            3.0
        );
        assert_eq!(
            lr_null_reference(None, None, None, &(0..4), 0.01, 0, WINDOW, &[], &[], None, None, None).chi_square_df,
            1.0
        );
    }
}

/// The ESTIMATED-SCALE channel (#2672): the reference a profiled Gaussian's
/// `W = n·ln(1 + Q/V) + B` actually has.
#[cfg(test)]
mod profiled_scale_reference_tests {
    use super::{
        SmoothLrProfiledScale, SmoothLrReferenceDf, SmoothLrReferenceSource, SmoothLrSelection,
        SmoothLrSelectionDecline,
    };

    /// A reference carrying an explicit spectrum, no selection replay, and the
    /// strictest tail accuracy the clamp allows.
    fn reference(weights: Vec<f64>, profiled_scale: Option<SmoothLrProfiledScale>) -> SmoothLrReferenceDf {
        let mean: f64 = weights.iter().sum();
        let second_moment: f64 = weights.iter().map(|w| w * w).sum();
        SmoothLrReferenceDf {
            weights,
            mean,
            second_moment,
            chi_square_df: mean * mean / second_moment,
            scale: second_moment / mean,
            moment_residual: None,
            edf: mean,
            null_dim: 0,
            source: SmoothLrReferenceSource::NullSpectrum,
            selection: SmoothLrSelection::Declined(SmoothLrSelectionDecline::GeometryRefused),
            profiled_scale,
        }
    }

    /// When the tested block's spectrum is FLAT the ratio law is a Fisher-
    /// Snedecor tail in closed form, so the whole channel — the `expm1`
    /// inversion, the residual spectrum, the multiplicity fold — is checkable
    /// against `fisher_snedecor_sf` with nothing shared but the arithmetic.
    ///
    /// `Q = g·χ²_q` and `V = χ²_ν`, so
    /// `P(W > w) = P(g·χ²_q > c·χ²_ν) = P(F_{q,ν} > c·ν/(g·q))`.
    #[test]
    fn a_flat_spectrum_makes_the_profiled_reference_an_f_tail() {
        for &(q, scale) in &[(1usize, 1.0_f64), (4, 1.0), (4, 0.37), (9, 2.5)] {
            for &nu in &[8.0_f64, 26.0, 191.0] {
                let observations = nu + q as f64;
                let subject = reference(
                    vec![scale; q],
                    Some(SmoothLrProfiledScale {
                        observations,
                        deterministic_offset: 0.0,
                        // `V ~ χ²_ν` exactly: no partially-shrunk direction, so
                        // the whole residual law is the unit-multiplicity term.
                        residual_weights: Vec::new(),
                        residual_unit_dimension: nu,
                    }),
                );
                for &statistic in &[0.05_f64, 0.8, 3.0, 12.0, 30.0] {
                    let (got, bound) = subject.tail_probability_with_bound(statistic);
                    let ratio = (statistic / observations).exp_m1();
                    let want = gam_math::probability::fisher_snedecor_sf(
                        ratio * nu / (scale * q as f64),
                        q as f64,
                        nu,
                    );
                    assert!(
                        (got - want).abs() <= bound + 1e-9,
                        "q={q} g={scale} ν={nu} W={statistic}: {got} vs F-tail {want} \
                         (certified {bound:.3e})"
                    );
                }
            }
        }
    }

    /// The deterministic offset is part of the statistic, not a nuisance: a `W`
    /// at or below it corresponds to `Q/V ≤ 0`, which cannot happen, so the
    /// p-value is exactly one rather than something the quadrature invents.
    #[test]
    fn a_statistic_under_the_deterministic_offset_is_not_evidence() {
        let subject = reference(
            vec![1.0_f64, 0.5],
            Some(SmoothLrProfiledScale {
                observations: 30.0,
                deterministic_offset: -0.61,
                residual_weights: vec![0.25],
                residual_unit_dimension: 24.0,
            }),
        );
        for &statistic in &[-5.0_f64, -0.61, -0.7] {
            let (got, bound) = subject.tail_probability_with_bound(statistic);
            assert_eq!(got, 1.0, "W={statistic}");
            assert_eq!(bound, 0.0);
        }
        // And it is a strict boundary: just above the offset the tail is still
        // essentially one, but it is no longer the exact branch.
        let (just_above, _) = subject.tail_probability_with_bound(-0.6099);
        assert!(just_above < 1.0 && just_above > 0.999, "{just_above}");
    }

    /// A profiled `W` between the offset and zero is an ordinary draw from the
    /// reference: its tail is resolved, strictly between the tail at zero and
    /// one, and falls as `W` rises. Scoring it as `W = 0` put an atom at the
    /// tail at zero under half of the null replicates.
    #[test]
    fn a_statistic_between_the_offset_and_zero_is_scored_where_it_is() {
        use super::SmoothLrPValue;
        let subject = reference(
            vec![1.0_f64, 0.5],
            Some(SmoothLrProfiledScale {
                observations: 30.0,
                deterministic_offset: -0.61,
                residual_weights: vec![0.25],
                residual_unit_dimension: 24.0,
            }),
        );
        let (at_zero, _) = subject.tail_probability_with_bound(0.0);
        let mut previous = 1.0;
        for &statistic in &[-0.5_f64, -0.3, -0.1, -1e-3] {
            let (value, accuracy) = subject.tail_probability_with_bound(statistic);
            assert!(value.is_finite() && accuracy.is_finite(), "W={statistic}: {value} ± {accuracy}");
            assert!(value > at_zero && value < previous, "W={statistic}: {value} vs [{at_zero}, {previous}]");
            assert_eq!(
                SmoothLrReferenceDf::typed_p_value(value, accuracy),
                Some(SmoothLrPValue::Resolved(value))
            );
            previous = value;
        }
    }

    /// A huge effect is reported as a finite tiny p-value that is the tail.
    ///
    /// On the profiled reference the tail is `P(F_{q,ν} > c·ν/(g·q))` in closed
    /// form (see the flat-spectrum test above), which `fisher_snedecor_sf`
    /// evaluates in log space. The reference's inversion is accurate relative
    /// to the tail, so a tail of `10⁻⁹⁰` is resolved as a point value within
    /// its own published accuracy of the exact one — not rounded to zero, and
    /// not floored at an absolute accuracy.
    #[test]
    fn a_huge_effect_is_a_finite_tiny_p_value_at_the_exact_tail() {
        use super::SmoothLrPValue;
        let (q, scale, nu) = (4usize, 0.37_f64, 26.0_f64);
        let observations = nu + q as f64;
        let subject = reference(
            vec![scale; q],
            Some(SmoothLrProfiledScale {
                observations,
                deterministic_offset: 0.0,
                residual_weights: Vec::new(),
                residual_unit_dimension: nu,
            }),
        );
        for &statistic in &[90.0_f64, 150.0, 400.0] {
            let (value, accuracy) = subject.tail_probability_with_bound(statistic);
            let ratio = (statistic / observations).exp_m1();
            let exact =
                gam_math::probability::fisher_snedecor_sf(ratio * nu / (scale * q as f64), q as f64, nu);
            assert!(exact > 0.0 && exact < 1e-16, "W={statistic}: {exact:.3e}");
            assert_eq!(
                SmoothLrReferenceDf::typed_p_value(value, accuracy),
                Some(SmoothLrPValue::Resolved(value)),
                "W={statistic}: {value:.3e} ± {accuracy:.3e}"
            );
            assert!(
                (value - exact).abs() <= accuracy,
                "W={statistic}: {value:.6e} ± {accuracy:.3e} against the exact tail {exact:.6e}"
            );
        }
    }

    /// A value its own accuracy does not separate from zero is published as the
    /// top of the certified interval, never as a point value, and never as a
    /// ceiling of zero.
    #[test]
    fn an_unresolved_tail_is_the_top_of_its_certified_interval() {
        use super::SmoothLrPValue;
        assert_eq!(
            SmoothLrReferenceDf::typed_p_value(4e-17, 1e-13),
            Some(SmoothLrPValue::UpperBound(1e-13 + 4e-17))
        );
        // A residue below zero is not part of the ceiling.
        assert_eq!(
            SmoothLrReferenceDf::typed_p_value(-3e-15, 1e-13),
            Some(SmoothLrPValue::UpperBound(1e-13))
        );
        assert_eq!(
            SmoothLrReferenceDf::typed_p_value(0.0, 0.0),
            Some(SmoothLrPValue::UpperBound(f64::from_bits(1)))
        );
        assert_eq!(SmoothLrReferenceDf::typed_p_value(0.2, 1e-13), Some(SmoothLrPValue::Resolved(0.2)));
        assert_eq!(SmoothLrReferenceDf::typed_p_value(f64::NAN, 0.0), None);
        assert_eq!(SmoothLrReferenceDf::typed_p_value(0.2, f64::NAN), None);
    }

    /// On a flat known-scale spectrum the law is a chi-square with a closed-form
    /// tail, so the inversion can be checked against it: even a tail of `10⁻⁴⁰`
    /// is a resolved finite value within its own accuracy of the exact one — no
    /// floor at `10⁻¹⁶`, no `1 − cdf` cancellation.
    #[test]
    fn a_closed_form_tail_resolves_arbitrarily_deep() {
        use super::SmoothLrPValue;
        let subject = reference(vec![1.0; 3], None);
        for &statistic in &[80.0_f64, 200.0, 600.0] {
            let (value, accuracy) = subject.tail_probability_with_bound(statistic);
            let exact = gam_math::probability::chi_square_sf(statistic, 3.0);
            assert!(exact > 0.0 && exact < 1e-16, "W={statistic}: {exact:.3e}");
            assert_eq!(
                SmoothLrReferenceDf::typed_p_value(value, accuracy),
                Some(SmoothLrPValue::Resolved(value)),
                "W={statistic}: {value:.3e} ± {accuracy:.3e}"
            );
            assert!(
                (value - exact).abs() <= accuracy,
                "W={statistic}: {value:.6e} ± {accuracy:.3e} against the exact tail {exact:.6e}"
            );
        }
    }
}

#[cfg(test)]
mod selection_replay_tests {
    use super::{
        AxisSlice, DiagonalCriterion, ObservedDraw, SMOOTH_LR_SELECTION_DRAWS,
        SelectionDrawStream, SelectionFactor, SelectionGeometry, SmoothLrSelection,
        SmoothLrSelectionDecline, SmoothLrReferenceDf, SmoothLrReferenceSource,
        SmoothLrSelectionProfile, SmoothLrSelectionReplay, paired_mean_with_error, split_mix64,
        stratified_chi_square,
    };
    use ndarray::Array2;

    /// #4086 sibling: the paired-difference spread is two-pass centred, so it
    /// cannot cancel. Half the draws at `c + s`, half at `c − s` (`c = 0.75`, `s` a
    /// power of two) have mean exactly `c` and population variance exactly
    /// `s²`, so the reference is exact with no second implementation. The
    /// retired one-pass `Σd²/N − d̄²` is evaluated alongside to show this is
    /// the regime where it loses the variance (it resolves only `ulp(c²)`),
    /// and a constant sample must report an error of exactly zero.
    #[test]
    fn paired_standard_error_is_centred_and_exact_under_a_large_mean_4086() {
        let c = 0.75_f64;
        let mut one_pass_lost = 0usize;
        for k in 20..32_i32 {
            let s = 2.0_f64.powi(-k);
            let mut sample = vec![c + s; 128];
            sample.extend(std::iter::repeat_n(c - s, 128));
            let n = sample.len() as f64;
            let exact_error = (s * s / n).sqrt();
            let (shift, error) = paired_mean_with_error(sample.iter().copied());
            assert_eq!(shift, c, "the construction gives the mean exactly");
            assert!(
                (error - exact_error).abs() <= 4.0 * f64::EPSILON * exact_error,
                "s = 2^-{k}: centred error {error:.17e} vs exact {exact_error:.17e}"
            );
            let sum: f64 = sample.iter().sum();
            let sum_squares: f64 = sample.iter().map(|d| d * d).sum();
            let one_pass = sum_squares / n - (sum / n) * (sum / n);
            if (one_pass - s * s).abs() > 0.5 * s * s {
                one_pass_lost += 1;
            }
        }
        assert!(
            one_pass_lost > 0,
            "the retired one-pass form must be seen to lose the variance, or this \
             test is not exercising the cancelling regime"
        );
        for value in [0.1_f64, 1.0 / 3.0, -0.7, 1.0] {
            let (shift, error) = paired_mean_with_error(std::iter::repeat_n(value, 10));
            assert_eq!(error, 0.0, "a constant sample of {value} has no spread");
            assert!((shift - value).abs() <= 2.0 * f64::EPSILON * value.abs());
        }
    }

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
            profile: None,
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
            profile: None,
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
            profile: None,
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
        match SmoothLrSelectionReplay::from_geometry(&diagonal(spectrum), &[window], None, draws, draws, None) {
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
                None,
                draws,
                draws,
                Some(ObservedDraw {
                    whitened: &z,
                    penalized_deviance: None,
                }),
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
            let p_value = |selection| {
                let mut reference = SmoothLrReferenceDf {
                    weights: weights.clone(),
                    mean: weights.iter().sum(),
                    second_moment: weights.iter().map(|w| w * w).sum(),
                    chi_square_df: 1.0,
                    scale: 1.0,
                    moment_residual: None,
                    edf: 0.0,
                    null_dim: 0,
                    source: SmoothLrReferenceSource::NullSpectrum,
                    selection,
                    profiled_scale: None,
                };
                reference.chi_square_df = reference.mean * reference.mean / reference.second_moment;
                reference.scale = reference.second_moment / reference.mean;
                reference.tail_probability_with_bound(statistic).0
            };
            rescored.push(p_value(SmoothLrSelection::Replayed(replay.clone())));
            let mut blind = replay;
            blind.observed = None;
            unscored.push(p_value(SmoothLrSelection::Replayed(blind)));
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

    /// Stratified χ² draws are the exact quantiles of the stratum midpoints, so
    /// their mean is `k` up to the midpoint rule's error, and a zero-dimensional
    /// variate is identically zero rather than a failed quantile.
    #[test]
    fn stratified_chi_square_draws_have_the_chi_square_mean() {
        for &k in &[1usize, 7, 40, 900] {
            let draws = stratified_chi_square(k, 3, 4096);
            let mean = draws.iter().sum::<f64>() / draws.len() as f64;
            assert!((mean - k as f64).abs() < 1.0e-2 * k as f64, "k={k}: mean {mean}");
            assert!(draws.iter().all(|value| value.is_finite() && *value > 0.0));
        }
        assert!(stratified_chi_square(0, 3, 64).iter().all(|value| *value == 0.0));
    }

    /// An estimated-scale family selects on the profiled criterion, so its
    /// replay draws the residual deviance too: the weight-one `χ²_h` is kept
    /// for the tail shift to share with `V`, and the two coordinates it and the
    /// other penalized directions occupy are consumed. A tested term whose
    /// rank exceeds the model's leaves `χ²_{r − r_j}` undefined and declines.
    /// As `h → ∞` the profiled criterion's `m·ln(R + D)` tends to `m·ln R +
    /// D·m/R` with `m/R → 1`, i.e. the known-scale criterion, so the selected
    /// statistics converge to the unprofiled replay's.
    #[test]
    fn a_profiled_replay_draws_the_residual_law_and_tends_to_the_known_scale_one() {
        let geometry = diagonal(&spectrum());
        let draws = 512;
        let replay = |profile| match SmoothLrSelectionReplay::from_geometry(
            &geometry,
            &[(-8.0, 8.0)],
            profile,
            draws,
            draws,
            None,
        ) {
            SmoothLrSelection::Replayed(replay) => replay,
            SmoothLrSelection::Declined(reason) => panic!("declined: {}", reason.label()),
        };
        let known = replay(None);
        assert!(known.residual_unit_sample.is_none());
        assert_eq!(known.consumed_coordinates, geometry.dimension);

        let profiled = replay(Some(SmoothLrSelectionProfile {
            residual_unit_dimension: 20,
            penalized_rank: geometry.rank + 3,
        }));
        let unit = profiled.residual_unit_sample.as_ref().expect("profiled draws");
        assert_eq!(unit.len(), profiled.selection_sample.len());
        let mean = unit.iter().sum::<f64>() / unit.len() as f64;
        assert!((mean - 20.0).abs() < 0.5, "χ²_20 draws average {mean}");
        assert_eq!(profiled.consumed_coordinates, geometry.dimension + 2);
        assert_eq!(profiled.conditional_sample, known.conditional_sample);
        assert_ne!(profiled.selection_sample, known.selection_sample);

        let asymptotic = replay(Some(SmoothLrSelectionProfile {
            residual_unit_dimension: 100_000_000,
            penalized_rank: geometry.rank,
        }));
        for (index, (profiled, known)) in asymptotic
            .selection_sample
            .iter()
            .zip(known.selection_sample.iter())
            .enumerate()
        {
            assert!(
                (profiled - known).abs() <= 1.0e-2 * (1.0 + known.abs()),
                "draw {index}: profiled {profiled} vs known-scale {known}"
            );
        }

        assert_eq!(
            SmoothLrSelectionReplay::from_geometry(
                &geometry,
                &[(-8.0, 8.0)],
                Some(SmoothLrSelectionProfile {
                    residual_unit_dimension: 20,
                    penalized_rank: geometry.rank - 1,
                }),
                draws,
                draws,
                None,
            )
            .decline(),
            Some(SmoothLrSelectionDecline::ProfileInconsistent)
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
            None,
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
                None,
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
            SmoothLrSelectionReplay::generate_multiscale(&crowded, &windows, None, 256, None).is_ok(),
            "a term with five scales is replayed over all five"
        );
        assert!(
            SmoothLrSelectionReplay::from_geometry(&crowded, &windows, None, 256, 256, None)
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
                None,
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
                None,
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
            .is_none()
        );
    }

    /// A term with nothing to select — no penalized direction, or a window the
    /// solver's box has closed — has no replay, and the conditional law is the
    /// selection law. This is the branch that keeps an unpenalized block exactly
    /// the textbook chi-square.
    #[test]
    fn nothing_to_select_means_no_replay() {
        assert!(SelectionGeometry::whiten(&Array2::eye(3), &[], &[]).is_none());
        assert!(
            SelectionGeometry::whiten(&Array2::eye(2), &[Array2::zeros((2, 2))], &[0.0]).is_none()
        );
        let geometry = diagonal(&spectrum());
        for window in [(4.0_f64, -4.0_f64), (f64::NAN, 1.0)] {
            assert_eq!(
                SmoothLrSelectionReplay::from_geometry(&geometry, &[window], None, 256, 256, None).decline(),
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
                None,
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

    /// #2902: an axis slice is the multi-scale criterion along its axis, exactly.
    ///
    /// The slice prices `log|I + C|` from an assembled `I + B` and `log|C|₊` from
    /// the cosine–sine decomposition of the stacked roots, so it is checked
    /// against [`SelectionFactor`]'s route on a dense information with two dense
    /// components, at separations up to the box's width: every change of the
    /// slice's value along its axis must be the same change of the criterion.
    #[test]
    fn an_axis_slice_is_the_criterion_along_its_axis_2902() {
        let (information, bending, ridge, draw) = dense_pair();
        let norm_squared: f64 = draw.iter().map(|value| value * value).sum();
        for separation in [0.0_f64, 18.0, 40.0] {
            let geometry = SelectionGeometry::whiten(
                &information,
                &[bending.clone(), ridge.clone()],
                &[0.5 * separation, -0.5 * separation],
            )
            .expect("geometry");
            let coordinates = range_coordinates(&geometry, &draw);
            let mut factor = SelectionFactor::new(&geometry);
            for point in [[0.0_f64, 0.0], [-2.5, 1.75], [3.0, -4.0]] {
                for axis in 0..2 {
                    let slice = AxisSlice::new(&geometry, &point[..], axis, &coordinates)
                        .expect("an axis slice at a priced point");
                    let criterion = slice.criterion(None);
                    let slice_anchor = criterion.jet(point[axis]).expect("slice jet").0[0];
                    assert!(factor.refactor(&geometry, &point[..]));
                    let anchor = factor.score(&coordinates, norm_squared).0;
                    for offset in [-9.0_f64, -1.5, 2.0, 6.0] {
                        let mut moved = point;
                        moved[axis] += offset;
                        let slice_value = criterion.jet(moved[axis]).expect("slice jet").0[0];
                        assert!(factor.refactor(&geometry, &moved[..]));
                        let value = factor.score(&coordinates, norm_squared).0;
                        let slice_change = slice_value - slice_anchor;
                        let change = value - anchor;
                        assert!(
                            (slice_change - change).abs()
                                <= 1e-7 * (1.0 + value.abs().max(anchor.abs())),
                            "separation {separation}, point {point:?}, axis {axis}, offset \
                             {offset}: the slice moves by {slice_change}, the criterion by {change}"
                        );
                    }
                }
            }
        }
    }

    /// #2902: each draw's multi-scale selection is the criterion's minimum along
    /// every axis through it, and on a separable pair over the whole box.
    ///
    /// This is the claim [`SmoothLrSelectionReplay::select_draw`] makes, checked
    /// through the canonical evaluator at points the descent never visited. On the
    /// SEPARABLE pair — unit information, a bending penalty on four directions and
    /// a ridge on the other two, at the separation a null-true smooth reaches — the
    /// criterion is a sum of one function per scale, so a coordinatewise minimum is
    /// the global one and no probe anywhere in the box may undercut it. On the
    /// COUPLED dense pair only probes along each axis through the selection are
    /// held to it, which is what a coordinatewise minimum is.
    #[test]
    fn multiscale_selection_is_a_coordinatewise_certified_minimum_2902() {
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
        let coupled = SelectionGeometry::whiten(
            &information,
            &[dense_bending, dense_ridge],
            &[9.0, -9.0],
        )
        .expect("coupled geometry");
        let windows = [(-42.0_f64, 18.0_f64), (-18.0, 42.0)];
        let mut state = 0x2902_0004_u64;
        let mut uniform = move || {
            state = split_mix64(state);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for (label, geometry, whole_box) in [
            ("separable", &separable, true),
            ("coupled", &coupled, false),
        ] {
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
                    None,
                    &mut selected,
                )
                .expect("a certified multi-scale selection");
                let mut value_at = |point: &[f64]| {
                    assert!(
                        factor.refactor(geometry, point),
                        "{label}: the evaluator refused the probe {point:?}"
                    );
                    factor.score(&coordinates, norm_squared).0
                };
                let at_selected = value_at(selected.as_slice());
                let tolerance = 1e-8 * (1.0 + at_selected.abs());
                let mut probes: Vec<[f64; 2]> = vec![[0.0, 0.0]];
                for _ in 0..24 {
                    let along = [
                        windows[0].0 + (windows[0].1 - windows[0].0) * uniform(),
                        windows[1].0 + (windows[1].1 - windows[1].0) * uniform(),
                    ];
                    probes.push([along[0], selected[1]]);
                    probes.push([selected[0], along[1]]);
                    if whole_box {
                        probes.push(along);
                    }
                }
                for probe in probes {
                    let at_probe = value_at(&probe[..]);
                    assert!(
                        at_selected <= at_probe + tolerance,
                        "{label} draw {index}: the selection {selected:?} (V={at_selected}) is \
                         undercut at {probe:?} (V={at_probe})"
                    );
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
            .unwrap_or_else(|| {
                panic!(
                    "the geometry refused a dense fit at separation {separation} —                      that is a silent `no replay` on every real model"
                )
            });
            let replay = SmoothLrSelectionReplay::from_geometry(
                &geometry,
                &[(-6.0, 6.0), (-6.0, 6.0)],
                None,
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
                        None,
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

    /// Independent standard normals for the brute-force checks: Box–Muller on
    /// SplitMix64, a stream disjoint from the replay's stratified one.
    fn brute_force_normals(seed: u64, count: usize) -> Vec<f64> {
        let mut state = seed;
        let mut uniform = || {
            state = split_mix64(state);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..count)
            .map(|_| {
                let (a, b) = (uniform(), uniform());
                (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
            })
            .collect()
    }

    /// An estimated-scale selection shift is `E_V[P(Q_sel ≥ cV) − P(Q_cond ≥ cV)]`
    /// over the residual law of `V`, not the same difference at `c·E[V]`. The
    /// replay's closed-form-in-`χ²_h` integral must agree with a brute-force
    /// Monte-Carlo over `V` drawn independently of it, on the replay's own
    /// paired samples, and the single-threshold reading it replaced must not.
    ///
    /// The residual law is the `n = 60` Gaussian cell's shape: a handful of
    /// partially-shrunk directions and `h = 44` unit ones.
    #[test]
    fn the_profiled_selection_shift_integrates_over_the_residual_law() {
        use super::SmoothLrProfiledScale;
        let replay = replay_from(&spectrum(), (-8.0, 8.0), SMOOTH_LR_SELECTION_DRAWS);
        let scale = SmoothLrProfiledScale {
            observations: 60.0,
            deterministic_offset: 0.0,
            residual_weights: vec![0.81, 0.36, 0.09, 0.01],
            residual_unit_dimension: 44.0,
        };
        let mean_v: f64 =
            scale.residual_weights.iter().sum::<f64>() + scale.residual_unit_dimension;
        let unit = scale.residual_unit_dimension as usize;
        let per_draw = scale.residual_weights.len() + unit;
        let outer = 4000usize;
        let normals = brute_force_normals(0xB4_2672, outer * per_draw);
        let residual_draws: Vec<f64> = normals
            .chunks(per_draw)
            .map(|z| {
                let (weighted, unit_part) = z.split_at(scale.residual_weights.len());
                weighted
                    .iter()
                    .zip(scale.residual_weights.iter())
                    .map(|(zi, w)| w * zi * zi)
                    .sum::<f64>()
                    + unit_part.iter().map(|zi| zi * zi).sum::<f64>()
            })
            .collect();
        let paired_shift_at = |threshold: f64| {
            replay
                .selection_sample
                .iter()
                .zip(replay.conditional_sample.iter())
                .map(|(&selected, &held)| {
                    f64::from(selected >= threshold) - f64::from(held >= threshold)
                })
                .sum::<f64>()
                / replay.selection_sample.len() as f64
        };
        for ratio in [0.01_f64, 0.03, 0.06, 0.1] {
            let per_v: Vec<f64> = residual_draws
                .iter()
                .map(|&v| paired_shift_at(ratio * v))
                .collect();
            let brute = per_v.iter().sum::<f64>() / outer as f64;
            let spread = (per_v.iter().map(|s| (s - brute).powi(2)).sum::<f64>()
                / (outer - 1) as f64)
                .sqrt();
            let brute_error = spread / (outer as f64).sqrt();
            let (integrated, _) = replay.profiled_tail_shift(ratio, ratio, &scale);
            let at_mean = paired_shift_at(ratio * mean_v);
            eprintln!(
                "c={ratio}: integrated {integrated:.5} brute {brute:.5} ± {brute_error:.5} \
                 at E[V] {at_mean:.5}"
            );
            assert!(
                (integrated - brute).abs() <= 4.0 * brute_error + 5.0e-4,
                "c={ratio}: integrated shift {integrated} vs brute-force E_V {brute} \
                 (± {brute_error})"
            );
        }
    }
}

#[cfg(test)]
mod lr_null_spectrum_moment_tests {
    use super::*;

    /// No penalty components and therefore no selection window: these tests are
    /// about the CONDITIONAL law, so the replay is inert.
    const WINDOW: &[(f64, f64)] = &[];

    // The whole-term LR reference (#1766, #1872, #2672). The first spectral
    // moment `tr(2F − F²)` IS Wood's `edf1`; what changed under #2672 is that it
    // is now the MEAN of a reference whose SHAPE comes from the second moment,
    // instead of being handed to a chi-square as a degrees of freedom.

    #[test]
    fn the_first_moment_is_wood_edf1() {
        // A symmetric smoother block with eigenvalues {0.9, 0.4}: a partially
        // shrunk penalized term. edf = tr = 1.3; tr(F²) = 0.81 + 0.16 = 0.97;
        // edf1 = 2·1.3 − 0.97 = 1.63. (Diagonal ⇒ block F² trace = Σ λ².)
        let f = ndarray::array![[0.9_f64, 0.0], [0.0, 0.4]];
        let [mean, second] = lr_null_spectral_moments(Some(&f), &(0..2)).unwrap();
        assert!(
            (mean - 1.63).abs() < 1e-12,
            "the first spectral moment is Wood's edf1 = 2*tr - tr(F^2) = 1.63, got {mean}"
        );
        // And it dominates the raw edf, analytically: w_j = 2f_j − f_j² ≥ f_j on
        // [0, 1], so no `.max(edf)` guard is needed to make it hold.
        assert!(mean >= 1.3 - 1e-12, "edf1 {mean} must be >= edf 1.3");
        // Second moment: w = {0.99, 0.64} ⇒ Σw² = 0.9801 + 0.4096.
        assert!((second - 1.3897).abs() < 1e-12, "second moment {second}");
    }

    #[test]
    fn a_corrupted_block_degrades_to_the_fallback_rather_than_being_floored() {
        // A real influence block has eigenvalues in [0, 1], so `tr(F²)` cannot
        // run away. Numerical corruption can still produce one that does, and
        // the pre-#2672 code floored `edf1` back at `tr` — silently returning a
        // reference derived from a block it had just decided was unusable. The
        // spectral reference does not paper over it: the first moment goes
        // negative and the assembly degrades to the unit-weight lane, VISIBLY.
        let f = ndarray::array![[0.5_f64, 40.0], [40.0, 0.5]];
        let [mean, _] = lr_null_spectral_moments(Some(&f), &(0..2)).unwrap();
        assert!(mean < 0.0, "the corrupted block's first moment is {mean}");
        let reference = lr_null_reference(Some(&f), None, None, &(0..2), 1.0, 1, WINDOW, &[], &[], None, None, None);
        assert_eq!(reference.source, SmoothLrReferenceSource::UnitWeightFallback);
        assert_eq!(reference.chi_square_df, 1.0);
        assert_eq!(reference.scale, 1.0);
    }

    #[test]
    fn returns_none_on_a_missing_or_out_of_range_block() {
        // No influence matrix at all → None (caller falls back to the
        // unit-weight `max(edf, null_dim, 1)` shape).
        assert!(lr_null_spectral_moments(None, &(0..2)).is_none());
        // An out-of-bounds range → None, never a panic.
        let f = ndarray::array![[0.5_f64, 0.0], [0.0, 0.5]];
        assert!(lr_null_spectral_moments(Some(&f), &(0..5)).is_none());
        // A fully-shrunk block has ZERO moments, which is not a usable
        // reference either — the caller must see the fallback, not a divide by
        // zero.
        let zero = ndarray::array![[0.0_f64, 0.0], [0.0, 0.0]];
        assert_eq!(
            lr_null_spectral_moments(Some(&zero), &(0..2)).unwrap(),
            [0.0, 0.0]
        );
        assert_eq!(
            lr_null_reference(Some(&zero), None, None, &(0..2), 0.0, 0, WINDOW, &[], &[], None, None, None).source,
            SmoothLrReferenceSource::UnitWeightFallback
        );
    }
}
