//! `RowMetric` — the single provenance-carrying per-row inner product shared by
//! the SAE-manifold **likelihood** (residual whitening) and the **gauge**
//! ([`crate::terms::analytic_penalties::IsometryPenalty`] pullback weight).
//!
//! # Why this exists
//!
//! The SAE-manifold machine historically carried *two* independent inner
//! products:
//!
//! * the **likelihood** measured reconstruction residuals isotropically — a
//!   single scalar dispersion `φ̂ = RSS / residual-dof`, the data-fit loop
//!   summing the bare `½ rᵀr`; there was no per-row metric at all; and
//! * the **gauge** carried its own per-row metric in
//!   `IsometryPenalty.weight: WeightField` — a low-rank `W_n = U_n U_nᵀ`
//!   pullback `g_n = J_nᵀ W_n J_n`, settable independently of anything the
//!   likelihood saw.
//!
//! Nothing structurally forced "the metric the likelihood whitens by" to equal
//! "the metric the gauge pulls back through". That is exactly the
//! objective↔gradient-desync bug class wearing geometry clothing: a
//! likelihood-metric ≠ gauge-metric state was *representable*.
//!
//! `RowMetric` collapses the two into one object. The likelihood whitens
//! through it; the gauge `WeightField` is *constructed from* it. A
//! divergent-metric state is therefore unrepresentable — there is only one
//! per-row factor stack `U_n`, with one [`MetricProvenance`] tag.
//!
//! # Magic-by-default selector
//!
//! There is no flag. The provenance is chosen by whether per-row Fisher factors
//! exist:
//!
//! * no factors supplied ⇒ [`MetricProvenance::Euclidean`]; `W_n = I_p`;
//!   whitening is the identity, so `φ̂` and the data-fit loop are
//!   **bit-for-bit** the prior isotropic path; and
//! * per-row Fisher factors supplied ⇒ [`MetricProvenance::OutputFisher`]; the
//!   residual is whitened by `U_nᵀ` and the gauge pulls back through the same
//!   `U_n`.
//!
//! # Validation
//!
//! Every metric block is constructed **through**
//! [`crate::inference::fisher_rao::normalize_fisher_rao_blocks`], which
//! broadcasts and eigenvalue-validates PSD-ness. `RowMetric` does not
//! reimplement that validation; it materializes `W_n = U_n U_nᵀ` (which is PSD
//! by construction) and runs it through the shared normalizer as the
//! single point of truth for "is this a valid precision metric".
//!
//! Any rank floor used to make a block invertible for an internal solve is
//! **solver-only** (mirroring `RidgePolicy::solver_only`, #747): it never enters
//! the residual the objective sums, so `δ` cannot bias the criterion.

use ndarray::{Array2, Array3, ArrayView1};
use std::sync::Arc;

use crate::inference::fisher_rao::normalize_fisher_rao_blocks;

/// Where the per-row metric came from — the provenance that makes
/// "likelihood-metric ≠ gauge-metric" diagnosable instead of silent.
///
/// Object 4 (the gauge object) reads this to certify which inner product the
/// fit actually used; #974 fills [`MetricProvenance::WhitenedStructured`] with a
/// factor-analytic residual-covariance whitening.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetricProvenance {
    /// `M_n = I_p` for every row. The likelihood is isotropic and the gauge
    /// pullback reduces to the bare `J_nᵀ J_n`. This is the default and is
    /// bit-for-bit the historical isotropic-`φ̂` path.
    Euclidean,
    /// `M_n = U_n U_nᵀ (+ solver-only δI)` from supplied per-row output-Fisher
    /// factors `U_n ∈ ℝ^{p × rank}`. The canonical "one unit of latent motion ↦
    /// one unit of behavioral change" metric: residuals are whitened in the
    /// output-Fisher inner product and the gauge pulls back through the same
    /// factors. The `rank` is carried in the provenance so a consumer (Object 4)
    /// can certify the factor rank that produced the inner product.
    OutputFisher { rank: usize },
    /// `M_n = U_n U_nᵀ` from per-row output-Fisher factors that aggregate the
    /// **downstream** influence of position `n` over future positions through
    /// the KV path, rather than the same-position logits of
    /// [`MetricProvenance::OutputFisher`] (#980, mechanism 2).
    ///
    /// The same-position pullback `∂logits_t/∂x_t` can be ≈ 0 for a feature
    /// whose entire causal effect lands many tokens later (information carried
    /// forward through attention); a gauge built on it is blind to exactly that
    /// content. This provenance is the forward-looking alternative: each row's
    /// factor `U_n` is the top-`rank` factorization of the aggregated output
    /// Fisher `Σ_{t ≥ n} (∂logits_t/∂x_n)ᵀ F_t (∂logits_t/∂x_n)` over future
    /// positions the residual stream at `n` reaches. It is provenance-generic:
    /// it whitens nothing ([`Self::whitens_likelihood`] is `false`, like
    /// [`MetricProvenance::OutputFisher`]) and drives the gauge / lens /
    /// enrichment unchanged ([`Self::is_output_fisher_like`]). The lens/gauge
    /// machinery consumes it identically; only the *scientific* reading
    /// changes — dormant-feature detection becomes forward-looking (a feature
    /// driving far-future tokens now registers behavioral coupling that the
    /// same-position metric reported as ≈ 0).
    OutputFisherDownstream { rank: usize },
    /// Structured-residual whitening: `M_n = Σ_n^{-1}` from the **estimated**
    /// factor-analytic residual covariance `Σ_n = Λ c(z_n) Λᵀ + D` (#974), with
    /// `factor_rank` the selected factor count. Produced by
    /// [`crate::inference::residual_factor::StructuredResidualModel::row_metric`];
    /// the only provenance for which
    /// [`whitens_likelihood`](RowMetric::whitens_likelihood) is `true`. It
    /// carries the same low-rank factor layout as
    /// [`MetricProvenance::OutputFisher`].
    WhitenedStructured { factor_rank: usize },
}

/// The single per-row metric object. Holds one low-rank factor stack `U_n` (or
/// none, for Euclidean) plus the validated PSD blocks, tagged with its
/// [`MetricProvenance`].
///
/// `p` is the output dimensionality (residual / Jacobian-column dimension); the
/// per-row factor `U_n ∈ ℝ^{p × rank}` so `W_n = U_n U_nᵀ ∈ ℝ^{p × p}` without
/// ever being materialized as `p × p` in any hot path.
#[derive(Clone, Debug)]
pub struct RowMetric {
    provenance: MetricProvenance,
    n_rows: usize,
    p: usize,
    rank: usize,
    /// `(n_rows, p * rank)` row-major: `U_n[i, k] = u[n, i * rank + k]`. `None`
    /// for [`MetricProvenance::Euclidean`] (the identity factor is implicit).
    factors: Option<Arc<Array2<f64>>>,
    /// **Solver-only** Tikhonov floor `δ` added as `δ I_p` to make a
    /// rank-deficient `U_n U_nᵀ` invertible for an *internal solve only*.
    ///
    /// Invariant (mirrors `RidgePolicy::solver_only`, #747): `δ` **never** enters
    /// any quantity that feeds the evidence criterion. The criterion-facing
    /// quad-form / whitening / fisher-mass methods all use the *un-floored*
    /// `U_n U_nᵀ`; only [`Self::solve_floor`]-tagged solver helpers see `δ`. A
    /// nonzero floor therefore cannot bias the objective the optimizer reports.
    solver_delta: f64,
    /// Per-row traces `tr(M_n)` of the criterion-facing (un-floored) metric.
    ///
    /// This is the only dense-block reduction any consumer reads (the #980
    /// Fisher-mass row measure); the `(n_rows, p, p)` block stack itself is
    /// validated **streamingly** at construction through
    /// [`normalize_fisher_rao_blocks`] one row at a time and then dropped.
    /// Retaining it was `n·p²·8` bytes — 13 GiB at `(n=2000, p=896)` and an
    /// OOM at LLM-scale `p` — for a record nothing ever re-read. The solver
    /// `δ` is deliberately *not* baked in here, so this is the
    /// criterion-facing trace.
    traces: ndarray::Array1<f64>,
}

impl RowMetric {
    /// Euclidean metric: `W_n = I_p` for all `n`. Whitening is the identity, so
    /// the likelihood residual path is bit-for-bit the prior isotropic `φ̂`.
    ///
    /// Constructed directly: the identity stack is PSD axiomatically, so
    /// routing it through the dense normalizer would materialize and
    /// spectrum-check `n` identity blocks (`n·p²` memory, `n·p³` flops) to
    /// validate a tautology. `tr(I_p) = p` per row.
    pub fn euclidean(n_rows: usize, p: usize) -> Result<Self, String> {
        Ok(Self {
            provenance: MetricProvenance::Euclidean,
            n_rows,
            p,
            rank: p,
            factors: None,
            solver_delta: 0.0,
            traces: ndarray::Array1::<f64>::from_elem(n_rows, p as f64),
        })
    }

    /// Output-Fisher metric: per-row low-rank factors `U_n ∈ ℝ^{p × rank}`
    /// supplied as a `(n_rows, p * rank)` row-major matrix (`U_n[i, k] =
    /// u[n, i * rank + k]`). The induced `M_n = U_n U_nᵀ` is PSD by
    /// construction; it is validated through [`normalize_fisher_rao_blocks`] so
    /// the validation path is shared. No solver floor (`δ = 0`).
    pub fn output_fisher(u: Arc<Array2<f64>>, p: usize, rank: usize) -> Result<Self, String> {
        Self::from_factors(MetricProvenance::OutputFisher { rank }, u, p, rank, 0.0)
    }

    /// Downstream-influence output-Fisher metric: per-row factors `U_n ∈
    /// ℝ^{p × rank}` whose `M_n = U_n U_nᵀ` is the aggregated output Fisher of
    /// position `n` over the **future** positions it reaches through the KV path
    /// ([`MetricProvenance::OutputFisherDownstream`], #980 mechanism 2). The
    /// factor layout is identical to [`Self::output_fisher`]; only the
    /// provenance tag (and hence the scientific reading) differs. Whitens
    /// nothing, drives the gauge / lens / enrichment exactly as the
    /// same-position metric does — the consuming machinery is provenance-generic
    /// (see [`Self::is_output_fisher_like`]).
    pub fn output_fisher_downstream(
        u: Arc<Array2<f64>>,
        p: usize,
        rank: usize,
    ) -> Result<Self, String> {
        Self::from_factors(
            MetricProvenance::OutputFisherDownstream { rank },
            u,
            p,
            rank,
            0.0,
        )
    }

    /// Like [`Self::output_fisher`] but with a **solver-only** Tikhonov floor
    /// `δ ≥ 0`. The floor is recorded for solver helpers only; every
    /// criterion-facing method (`quad_form`, `whiten_residual`, `fisher_mass`)
    /// ignores it (#747 discipline), so the evidence criterion is `δ`-free.
    pub fn output_fisher_with_solver_floor(
        u: Arc<Array2<f64>>,
        p: usize,
        rank: usize,
        solver_delta: f64,
    ) -> Result<Self, String> {
        if !(solver_delta.is_finite() && solver_delta >= 0.0) {
            return Err(format!(
                "RowMetric::output_fisher_with_solver_floor: solver_delta must be finite and \
                 non-negative; got {solver_delta}"
            ));
        }
        Self::from_factors(
            MetricProvenance::OutputFisher { rank },
            u,
            p,
            rank,
            solver_delta,
        )
    }

    /// Structured-residual whitening from supplied per-row precision factors.
    ///
    /// `u` carries the per-row factor stack `U_n ∈ ℝ^{p × rank}` (row-major flat)
    /// with `U_n U_nᵀ = M_n = Σ_n^{-1}` — the precision of the **estimated**
    /// residual-covariance noise model. This is the low-level constructor; the
    /// #974 producer that *fits* `Σ_n` (a low-rank factor + diagonal + smooth
    /// activity-scale) and assembles these factors is
    /// [`crate::inference::residual_factor::StructuredResidualModel`], which calls
    /// through here. Because the provenance is
    /// [`MetricProvenance::WhitenedStructured`], [`Self::whitens_likelihood`] is
    /// `true`: a metric built this way is the first that whitens the likelihood.
    pub fn whitened_structured(u: Arc<Array2<f64>>, p: usize, rank: usize) -> Result<Self, String> {
        Self::from_factors(
            MetricProvenance::WhitenedStructured { factor_rank: rank },
            u,
            p,
            rank,
            0.0,
        )
    }

    /// Fit the #974 structured residual-covariance model to the residual matrix
    /// `R ∈ ℝ^{n×p}` against the smooth activity coordinate `z ∈ ℝ^n`, and return
    /// the resulting likelihood-whitening [`RowMetric`].
    ///
    /// This is the convenience entry point that ties the estimator to its
    /// consumer in one call: it runs
    /// [`StructuredResidualModel::fit`](crate::inference::residual_factor::StructuredResidualModel::fit)
    /// (deterministic fixed-iteration alternation + evidence-ladder rank
    /// selection) and lowers the fitted per-row precisions into a
    /// [`MetricProvenance::WhitenedStructured`] metric. `max_factor_rank` caps the
    /// factor count the ladder may select.
    pub fn from_estimated_residual_covariance(
        residuals: ndarray::ArrayView2<'_, f64>,
        activity: ArrayView1<'_, f64>,
        max_factor_rank: usize,
    ) -> Result<Self, String> {
        use crate::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
        let n = residuals.nrows();
        let model = StructuredResidualModel::fit(ResidualFactorInput {
            residuals,
            activity,
            max_factor_rank,
        })?;
        model.row_metric(n)
    }

    fn from_factors(
        provenance: MetricProvenance,
        u: Arc<Array2<f64>>,
        p: usize,
        rank: usize,
        solver_delta: f64,
    ) -> Result<Self, String> {
        let n_rows = u.nrows();
        if u.ncols() != p * rank {
            return Err(format!(
                "RowMetric::from_factors: factor matrix has {} cols; expected p*rank = {}*{} = {}",
                u.ncols(),
                p,
                rank,
                p * rank
            ));
        }
        if !u.iter().all(|v| v.is_finite()) {
            return Err("RowMetric::from_factors: factors must be finite".to_string());
        }
        // Materialize W_n = U_n U_nᵀ one row at a time (PSD by construction),
        // validate each through the single shared normalizer rather than
        // reimplementing the PSD check, record its trace, and drop the block.
        // Streaming keeps construction O(p²) memory; the former whole-stack
        // materialization retained `n·p²` doubles nothing ever re-read.
        let mut traces = ndarray::Array1::<f64>::zeros(n_rows);
        let mut full = Array3::<f64>::zeros((1, p, p));
        for row in 0..n_rows {
            for i in 0..p {
                for j in 0..p {
                    let mut acc = 0.0;
                    for k in 0..rank {
                        acc += u[[row, i * rank + k]] * u[[row, j * rank + k]];
                    }
                    full[[0, i, j]] = acc;
                }
            }
            normalize_fisher_rao_blocks(full.view().into_dyn(), 1, p)
                .map_err(|e| format!("RowMetric::from_factors: row {row}: {e}"))?;
            let mut tr = 0.0_f64;
            for i in 0..p {
                tr += full[[0, i, i]];
            }
            traces[row] = tr;
        }
        Ok(Self {
            provenance,
            n_rows,
            p,
            rank,
            factors: Some(u),
            solver_delta,
            traces,
        })
    }

    /// The provenance tag (consumed by Object 4 to certify the inner product).
    pub fn provenance(&self) -> MetricProvenance {
        self.provenance
    }

    /// Whether this metric is allowed to **whiten the likelihood** (i.e. replace
    /// the isotropic reconstruction data-fit `½ rᵀr` with the whitened
    /// `½ rᵀ M_n r`).
    ///
    /// This is TRUE **only** for [`MetricProvenance::WhitenedStructured`] — a
    /// genuinely *estimated noise model* (a factor-analytic residual covariance,
    /// #974), for which whitening the likelihood is the statistically correct
    /// thing to do. For [`MetricProvenance::Euclidean`] there is nothing to
    /// whiten by, and for [`MetricProvenance::OutputFisher`] the inner product is
    /// an **output-geometry gauge**, not an estimated noise model — whitening the
    /// likelihood by it would silently replace the reconstruction loss with a
    /// Fisher pullback (the #980 failure mode). So both leave the likelihood
    /// untouched and only the gauge sees the metric (see [`Self::drives_gauge`]).
    pub fn whitens_likelihood(&self) -> bool {
        matches!(self.provenance, MetricProvenance::WhitenedStructured { .. })
    }

    /// Whether this metric **drives the gauge** — i.e. the isometry-penalty
    /// pullback weight is taken from it rather than the identity.
    ///
    /// TRUE for any non-[`MetricProvenance::Euclidean`] provenance: both
    /// [`MetricProvenance::OutputFisher`] and
    /// [`MetricProvenance::WhitenedStructured`] supply a non-identity per-row
    /// inner product the gauge pulls back through. Euclidean reduces the gauge
    /// pullback to the bare `J_nᵀ J_n`, so it does not drive the gauge.
    pub fn drives_gauge(&self) -> bool {
        !matches!(self.provenance, MetricProvenance::Euclidean)
    }

    /// Whether this metric is an **output-Fisher gauge** — either the
    /// same-position [`MetricProvenance::OutputFisher`] or the downstream
    /// [`MetricProvenance::OutputFisherDownstream`] (#980). The two share every
    /// consumer behavior (Sym(F) separation under the gauge, two-lens coupling,
    /// steering geometry, enrichment); they differ only in the *scientific*
    /// reading of what behavioral coupling means (same-position vs
    /// forward-looking). Consumers that gate on "is this an output-Fisher
    /// pullback" should use this predicate rather than matching one variant, so
    /// the downstream metric rides the identical path.
    pub fn is_output_fisher_like(&self) -> bool {
        matches!(
            self.provenance,
            MetricProvenance::OutputFisher { .. } | MetricProvenance::OutputFisherDownstream { .. }
        )
    }

    /// Number of rows the metric is defined over.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Output dimensionality `p` (residual / Jacobian-column dimension).
    pub fn p_out(&self) -> usize {
        self.p
    }

    /// The factor rank: the dimension of the whitened residual
    /// [`Self::whiten_residual_row`] returns (and the column count of the per-row
    /// factor `U_n ∈ ℝ^{p × rank}`). For [`MetricProvenance::Euclidean`] this is
    /// `p` (the implicit identity factor), so a consumer that sizes a whitened
    /// buffer by `metric_rank()` gets the right length in every provenance.
    pub fn metric_rank(&self) -> usize {
        self.rank
    }

    /// Per-row traces `tr(M_n)` of the criterion-facing (un-floored) metric —
    /// the Fisher-mass reduction the #980 row measure consumes. The dense
    /// `(n_rows, p, p)` stack is validated streamingly at construction and
    /// never retained; consumers wanting an explicit `W_n` rebuild it from
    /// [`Self::metric_rank`]-sized factors.
    pub fn row_traces(&self) -> ndarray::ArrayView1<'_, f64> {
        self.traces.view()
    }

    /// Whiten a single `p`-dimensional residual row `r` into the coordinates
    /// whose squared Euclidean norm equals `rᵀ W_n r`.
    ///
    /// * Euclidean: returns `r` unchanged (`‖r‖² = rᵀ I r`), so the likelihood
    ///   reproduces the isotropic `½ rᵀr` data-fit bit-for-bit.
    /// * Factored: returns `U_nᵀ r ∈ ℝ^{rank}`, with
    ///   `‖U_nᵀ r‖² = rᵀ U_n U_nᵀ r = rᵀ W_n r`.
    ///
    /// This is the load-bearing identity that lets the data-fit loop sum
    /// `0.5 * Σ whitened²` and recover exactly `rᵀ W_n r` whatever the
    /// provenance.
    pub fn whiten_residual_row(&self, row: usize, r: ArrayView1<'_, f64>) -> Vec<f64> {
        match &self.factors {
            None => r.iter().copied().collect(),
            Some(u) => {
                let mut out = vec![0.0_f64; self.rank];
                for k in 0..self.rank {
                    let mut acc = 0.0;
                    for i in 0..self.p {
                        acc += u[[row, i * self.rank + k]] * r[i];
                    }
                    out[k] = acc;
                }
                out
            }
        }
    }

    /// The factor entry `U_n[i, k]` for one row (`i ∈ [0, p)`, `k ∈ [0, rank)`).
    /// For [`MetricProvenance::Euclidean`] the implicit factor is `I_p`, so this
    /// returns `1.0` when `i == k` and `0.0` otherwise — letting a consumer that
    /// whitens a Jacobian via `factor_entry` produce the identity whitening
    /// without a provenance branch. Reads the **un-floored** factors (criterion
    /// face, #747).
    pub fn factor_entry(&self, row: usize, i: usize, k: usize) -> f64 {
        match &self.factors {
            None => {
                if i == k {
                    1.0
                } else {
                    0.0
                }
            }
            Some(u) => u[[row, i * self.rank + k]],
        }
    }

    /// Apply the full per-row metric `M_n x = U_n (U_nᵀ x) ∈ ℝ^p` for one
    /// `p`-vector `x`, formed factored (`rank` flops in, `p` flops out) — never
    /// materializing `M_n` as `p × p`. Euclidean returns `x` unchanged
    /// (`M_n = I_p`). This is the p-space metric-applied vector the SAE β-tier
    /// data-fit gradient contracts (β lives in p-output space, so its gradient
    /// needs `M_n r_n`, not the rank-space whitened residual `U_nᵀ r_n`). Uses the
    /// **un-floored** factors (criterion face, `δ`-free, #747 invariant).
    pub fn apply_metric_row(&self, row: usize, x: ArrayView1<'_, f64>) -> Vec<f64> {
        match &self.factors {
            None => x.iter().copied().collect(),
            Some(u) => {
                // w = U_nᵀ x ∈ ℝ^{rank}.
                let mut w = vec![0.0_f64; self.rank];
                for k in 0..self.rank {
                    let mut acc = 0.0;
                    for i in 0..self.p {
                        acc += u[[row, i * self.rank + k]] * x[i];
                    }
                    w[k] = acc;
                }
                // out = U_n w ∈ ℝ^p.
                let mut out = vec![0.0_f64; self.p];
                for i in 0..self.p {
                    let mut acc = 0.0;
                    for k in 0..self.rank {
                        acc += u[[row, i * self.rank + k]] * w[k];
                    }
                    out[i] = acc;
                }
                out
            }
        }
    }

    /// Pullback metric `g_n = J_nᵀ W_n J_n` for one row, formed as
    /// `(J_nᵀ U_n)(U_nᵀ J_n)` — never materializing the `p × p` `W_n`.
    ///
    /// `j_row` is the row's Jacobian `J_n ∈ ℝ^{p × d}` flattened row-major
    /// (`J_n[i, a] = j_row[i * d + a]`). Returns the `d × d` `g_n`.
    pub fn pullback(&self, row: usize, j_row: &[f64], d: usize) -> Array2<f64> {
        match &self.factors {
            None => {
                // W_n = I_p ⇒ g_n = J_nᵀ J_n.
                let mut g = Array2::<f64>::zeros((d, d));
                for a in 0..d {
                    for b in a..d {
                        let mut acc = 0.0;
                        for i in 0..self.p {
                            acc += j_row[i * d + a] * j_row[i * d + b];
                        }
                        g[[a, b]] = acc;
                        g[[b, a]] = acc;
                    }
                }
                g
            }
            Some(u) => {
                // M_n = U_nᵀ J_n ∈ ℝ^{rank × d}; g_n = M_nᵀ M_n.
                let mut m = Array2::<f64>::zeros((self.rank, d));
                for k in 0..self.rank {
                    for a in 0..d {
                        let mut acc = 0.0;
                        for i in 0..self.p {
                            acc += u[[row, i * self.rank + k]] * j_row[i * d + a];
                        }
                        m[[k, a]] = acc;
                    }
                }
                let mut g = Array2::<f64>::zeros((d, d));
                for a in 0..d {
                    for b in a..d {
                        let mut acc = 0.0;
                        for k in 0..self.rank {
                            acc += m[[k, a]] * m[[k, b]];
                        }
                        g[[a, b]] = acc;
                        g[[b, a]] = acc;
                    }
                }
                g
            }
        }
    }

    /// Quadratic form `r_nᵀ M_n r_n` for one row's residual `r_n ∈ ℝ^p`, formed
    /// **factored** as `‖U_nᵀ r_n‖²` — never materializing the `p × p` `M_n`.
    ///
    /// This is the criterion-facing squared residual the likelihood sums; it uses
    /// the **un-floored** `U_n U_nᵀ`, so the solver `δ` does not enter it
    /// (#747 invariant). Euclidean provenance returns the bit-identical `‖r_n‖²`.
    pub fn quad_form(&self, row: usize, r: ArrayView1<'_, f64>) -> f64 {
        match &self.factors {
            None => r.iter().map(|&v| v * v).sum(),
            Some(_) => self
                .whiten_residual_row(row, r)
                .iter()
                .map(|&w| w * w)
                .sum(),
        }
    }

    /// Whiten a per-row Jacobian `J_n ∈ ℝ^{p × d}` (row-major flat,
    /// `J_n[i, a] = j_row[i * d + a]`) into `M_n = U_nᵀ J_n ∈ ℝ^{rank × d}` so
    /// that `M_nᵀ M_n = J_nᵀ (U_n U_nᵀ) J_n = J_nᵀ W_n J_n` is the pullback
    /// **without** any `p × p` intermediate. Euclidean returns `J_n` reshaped to
    /// `(p, d)` (the identity whitening). Solver `δ` is not applied (criterion
    /// face).
    pub fn whiten_jacobian(&self, row: usize, j_row: &[f64], d: usize) -> Array2<f64> {
        match &self.factors {
            None => {
                let mut out = Array2::<f64>::zeros((self.p, d));
                for i in 0..self.p {
                    for a in 0..d {
                        out[[i, a]] = j_row[i * d + a];
                    }
                }
                out
            }
            Some(u) => {
                let mut m = Array2::<f64>::zeros((self.rank, d));
                for k in 0..self.rank {
                    for a in 0..d {
                        let mut acc = 0.0;
                        for i in 0..self.p {
                            acc += u[[row, i * self.rank + k]] * j_row[i * d + a];
                        }
                        m[[k, a]] = acc;
                    }
                }
                m
            }
        }
    }

    /// Fisher mass of a per-row output vector `x_n ∈ ℝ^p`: the scalar
    /// `x_nᵀ M_n x_n` (alias of [`Self::quad_form`] read as an information mass
    /// rather than a residual square). Factored, never `p × p`, `δ`-free.
    pub fn fisher_mass(&self, row: usize, x: ArrayView1<'_, f64>) -> f64 {
        self.quad_form(row, x)
    }

    /// The **solver-only** Tikhonov floor `δ` (#747). Returned for internal
    /// solver helpers that need `U_n U_nᵀ + δ I` to be invertible; by contract
    /// no caller may fold this into a criterion-facing quantity. Always `0` for
    /// Euclidean and for factored metrics built without an explicit floor.
    pub fn solver_floor(&self) -> f64 {
        self.solver_delta
    }

    /// The gauge view of this metric: the
    /// [`crate::terms::analytic_penalties::WeightField`] the
    /// [`crate::terms::analytic_penalties::IsometryPenalty`] pulls back through.
    ///
    /// This is the **single** way an `IsometryPenalty` acquires a non-identity
    /// gauge metric — the independent `WeightField` setter has been removed — so
    /// the gauge metric is, by construction, the same object the likelihood
    /// whitens with.
    pub fn to_weight_field(&self) -> crate::terms::analytic_penalties::WeightField {
        use crate::terms::analytic_penalties::WeightField;
        match &self.factors {
            None => WeightField::Identity,
            Some(u) => WeightField::Factored {
                u: Arc::clone(u),
                rank: self.rank,
                p_out: self.p,
            },
        }
    }
}
