//! `RowMetric` — the single provenance-carrying per-row inner product shared by
//! the SAE-manifold **likelihood** (residual whitening) and the **gauge**
//! (isometry pullback weight).
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
//! [`crate::normalize_fisher_rao_blocks`], which
//! broadcasts and eigenvalue-validates PSD-ness. `RowMetric` does not
//! reimplement that validation; it materializes `W_n = U_n U_nᵀ` (which is PSD
//! by construction) and runs it through the shared normalizer as the
//! single point of truth for "is this a valid precision metric".
//!
//! Any rank floor used to make a block invertible for an internal solve is
//! **solver-only** (mirroring `RidgePolicy::solver_only`, #747): it never enters
//! the residual the objective sums, so `δ` cannot bias the criterion.
//!
//! # Rung 1 — the behavioral metric *in the reconstruction loss* (nats currency)
//!
//! [`MetricProvenance::OutputFisher`] installs the output-Fisher inner product
//! as a **gauge** metric only: it whitens *nothing* (`whitens_likelihood()` is
//! `false`), by deliberate #980 contract, so reconstruction stays the isotropic
//! `½‖r‖²`. That answers "what coordinate is canonical", not "what does a
//! reconstruction error *cost*".
//!
//! [`MetricProvenance::BehavioralFisher`] is the opposite deliberate choice:
//! the **same** low-rank output-Fisher factors, but installed as the
//! reconstruction *likelihood weight*. Plain MSE prices a reconstruction error
//! `e = x − x̂` by its Euclidean size; the model, however, reads the activation
//! only through the rest of the network, so the behavioral cost of `e` is the
//! KL between the clean and corrupted next-token distributions,
//! `KL ≈ ½ eᵀ G(x) e` with `G = JᵀFJ` the network-Jacobian pullback of the
//! output Fisher `F` (units: **nats**). Minimizing `(x−x̂)ᵀ G (x−x̂)` instead of
//! `‖x−x̂‖²` is **generalized least squares**: for a *fixed* per-row `G` it is
//! still a linear Gaussian model in the coefficients, so the entire
//! REML/evidence/EDF/certificate stack survives verbatim — this is why the
//! metric rides the identical `whitens_likelihood()` plumbing the
//! [`MetricProvenance::WhitenedStructured`] noise model uses, and why the G=I
//! limit reproduces the plain-MSE fit bit-for-bit (see the module tests).
//!
//! This is the principled form of Braun's end-to-end **KL + MSE** objective.
//! Anchoring to the activation keeps it *reconstruction* (it does not collapse
//! to "match the logits by any means" — the decoder still has to reproduce `x`),
//! while pricing the residual in nats through `G`. The payoff is automatic
//! selection for *mattering*: `G`'s null directions — activation structure the
//! rest of the network cannot read — are penalized nothing, because
//! `eᵀ G e = 0` there. MSE in a behaviorally-inert direction goes free, which is
//! the correct behavior, not a bug: nothing downstream changes, so nothing
//! should be paid.
//!
//! **The d×d `G` is never materialized.** `G` is sketched by `s` random probes,
//! `vᵢ = Jᵀ F^{1/2} uᵢ` (`uᵢ` iid, `s ≈ 4…16`), computed by `s` backward passes
//! per token at *harvest* time (the model-interaction boundary) and stored as
//! the columns of the per-row factor `U_n = [v₁ … v_s] ∈ ℝ^{p×s}`. Then
//! `G ≈ Σᵢ vᵢ vᵢᵀ = U_n U_nᵀ` and the criterion-facing
//! `eᵀ G e ≈ Σᵢ (vᵢᵀ e)² = ‖U_nᵀ e‖²` is exactly what
//! [`RowMetric::quad_form`] / [`RowMetric::whiten_residual_row`] already
//! compute — zero train-time model cost, `O(p·s)` per row. See
//! [`RowMetric::behavioral_fisher`] and the probe-packing helper
//! [`pack_probe_factors`].

use ndarray::{Array2, Array3, ArrayView1};
use std::sync::Arc;

use crate::normalize_fisher_rao_blocks;

/// Per-observation behavioral-metric field `W_n ∈ ℝ^{p × p}`, stored in
/// **low-rank factored form** `W_n = U_n U_n^T` with `U_n ∈ ℝ^{p × r_n}`.
///
/// The canonical coordinate is the one where one unit of motion in `t` is one
/// unit of behavioral change in the output space, so the `W_n` weighting is
/// load-bearing: the pullback metric is `g_n = J_n^T W_n J_n`. Storing as
/// `U_n` lets every contraction in this module run in
/// `(J^T U_n)(U_n^T J)` order, which is `O(p · r · d + r · d²)` per row — we
/// **never** materialize the `p × p` `W_n`, which is essential when `p`
/// (number of observation channels) is large but rank is small (e.g. one or
/// two behavioral dimensions per latent observation).
///
/// `Identity` is the gauge-fix default and corresponds to `U_n = I_p` so the
/// pullback reduces to the standard `J_n^T J_n`. `Factored` stores the
/// per-row `U_n` blocks contiguously: every row's factor is `p × rank`, and
/// rows may share the same rank (uniform-rank case) or vary if the field is
/// data-driven. For the uniform-rank case the storage is
/// `(n_obs, p * rank)` row-major.
#[derive(Clone)]
pub enum WeightField {
    /// `W_n = I_p` for every `n`. Reduces to the bare pullback `J^T J`.
    Identity,
    /// Per-row low-rank factor `U_n ∈ ℝ^{p × rank}`. Storage layout: a
    /// `(n_obs, p * rank)` row-major matrix where row `n` packs `U_n` in
    /// column-major-within-row order `U_n[i, k] = u[n, i * rank + k]`.
    Factored {
        u: Arc<Array2<f64>>,
        rank: usize,
        p_out: usize,
    },
}

impl std::fmt::Debug for WeightField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightField::Identity => f.write_str("Identity"),
            WeightField::Factored { u, rank, p_out } => f
                .debug_struct("Factored")
                .field("shape", &format_args!("{}×{}", u.nrows(), u.ncols()))
                .field("rank", rank)
                .field("p_out", p_out)
                .finish(),
        }
    }
}

impl WeightField {
    /// Apply `U_n^T J_n` for a specific row, given both the row's `J_n` flat
    /// `(p * d)` slice and the row's `U_n` flat `(p * rank)` slice. Returns
    /// the `(rank × d)` matrix and its row count.
    pub fn project_jac_row_with_u(
        u_row: &[f64],
        jac_row: &[f64],
        p: usize,
        rank: usize,
        d: usize,
    ) -> Array2<f64> {
        // M[k, a] = Σ_i U[i, k] · J[i, a].
        let mut m = Array2::<f64>::zeros((rank, d));
        for k in 0..rank {
            for a in 0..d {
                let mut s = 0.0;
                for i in 0..p {
                    s += u_row[i * rank + k] * jac_row[i * d + a];
                }
                m[[k, a]] = s;
            }
        }
        m
    }
}

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
    /// **Rung 1** — the output-Fisher metric installed as the reconstruction
    /// **likelihood weight** (generalized least squares in nats), not merely as
    /// a gauge. `M_n = U_n U_nᵀ ≈ G_n = J_nᵀ F_n J_n` is the `s`-probe sketch of
    /// the pulled-back output Fisher, with `U_n = [v₁ … v_s]`,
    /// `vᵢ = J_nᵀ F_n^{1/2} uᵢ`, and `probes = s` the number of random probes
    /// (the factor rank).
    ///
    /// This is the *only* [`RowMetric::is_output_fisher_like`]-adjacent
    /// provenance for which [`RowMetric::whitens_likelihood`] is `true`: the
    /// data-fit sums `½ eᵀ G_n e = ½ ‖U_nᵀ e‖²` (nats) instead of `½‖e‖²`. It is
    /// distinct from [`Self::OutputFisher`] precisely because the choice to let
    /// the metric enter the *loss* (rather than only the gauge) is deliberate and
    /// must not be silently inherited by the #980 gauge / two-tier-harvest
    /// contract — that contract relies on [`Self::OutputFisher`] whitening
    /// nothing. Because `G_n` is a *fixed* per-row metric, the whitened problem
    /// is again linear-Gaussian in the coefficients, so REML/evidence/EDF are
    /// unchanged (the GLS-preserves-REML property, verified in the module tests
    /// against the `G=I` plain-MSE limit).
    BehavioralFisher { probes: usize },
    /// Structured-residual whitening: `M_n = Σ_n^{-1}` from the **estimated**
    /// factor-analytic residual covariance `Σ_n = Λ c(z_n) Λᵀ + D` (#974), with
    /// `factor_rank` the selected factor count. Produced by
    /// Structured-residual producers materialize this provenance when they fit
    /// a residual-covariance whitening model;
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

    /// **Rung 1** — the output-Fisher metric as a reconstruction *likelihood
    /// weight* (GLS in nats): per-row `s`-probe factors `U_n ∈ ℝ^{p × probes}`
    /// supplied as a `(n_rows, p * probes)` row-major matrix
    /// (`U_n[i, k] = u[n, i * probes + k]`), so that column `k` is the probe
    /// vector `v_k = J_nᵀ F_n^{1/2} u_k` and `M_n = U_n U_nᵀ ≈ G_n`. Unlike
    /// [`Self::output_fisher`], the resulting metric returns
    /// `whitens_likelihood() == true`: the data-fit prices reconstruction error
    /// as `½ eᵀ G_n e`. Validated through [`normalize_fisher_rao_blocks`] like
    /// every factored metric; no solver floor (`δ = 0`).
    ///
    /// See [`pack_probe_factors`] to build `u` from a natural `(n, p, s)` probe
    /// stack emitted at harvest time.
    pub fn behavioral_fisher(u: Arc<Array2<f64>>, p: usize, probes: usize) -> Result<Self, String> {
        Self::from_factors(
            MetricProvenance::BehavioralFisher { probes },
            u,
            p,
            probes,
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
    /// residual-covariance noise model. This is the low-level constructor; #974
    /// producers that *fit* `Σ_n` (a low-rank factor + diagonal + smooth
    /// activity-scale) assemble these factors and call through here. Because the
    /// provenance is
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
    /// This is TRUE for two provenances, for two distinct reasons:
    ///
    /// * [`MetricProvenance::WhitenedStructured`] — a genuinely *estimated noise
    ///   model* (a factor-analytic residual covariance, #974), for which
    ///   whitening the likelihood is the statistically correct thing to do; and
    /// * [`MetricProvenance::BehavioralFisher`] — the **Rung 1** deliberate
    ///   choice to price reconstruction error in nats: the output-Fisher metric
    ///   `G_n` installed *as the loss weight* (`½ eᵀ G_n e`), a generalized
    ///   least-squares reconstruction. Because `G_n` is a fixed per-row metric
    ///   the problem stays linear-Gaussian, so REML/evidence/EDF are preserved.
    ///
    /// It is FALSE for [`MetricProvenance::Euclidean`] (nothing to whiten by) and
    /// for the *gauge-only* [`MetricProvenance::OutputFisher`] /
    /// [`MetricProvenance::OutputFisherDownstream`]: there the output-Fisher
    /// inner product is an **output-geometry gauge**, and whitening the
    /// likelihood by it *implicitly* (without the caller electing GLS) would
    /// silently replace the reconstruction loss with a Fisher pullback — the #980
    /// failure mode, and the reason the two-tier harvest can withhold factors
    /// from a row without changing its loss. `BehavioralFisher` is the *explicit*
    /// election of that same arithmetic as the intended objective.
    pub fn whitens_likelihood(&self) -> bool {
        matches!(
            self.provenance,
            MetricProvenance::WhitenedStructured { .. } | MetricProvenance::BehavioralFisher { .. }
        )
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
    #[inline]
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
    #[inline]
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
    #[inline]
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
    /// [`crate::WeightField`] the isometry penalty pulls back through.
    ///
    /// This is the **single** way an `IsometryPenalty` acquires a non-identity
    /// gauge metric — the independent `WeightField` setter has been removed — so
    /// the gauge metric is, by construction, the same object the likelihood
    /// whitens with.
    pub fn to_weight_field(&self) -> crate::WeightField {
        use crate::WeightField;
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

/// Pack a harvest-emitted probe stack into the row-major factor layout
/// [`RowMetric::behavioral_fisher`] expects.
///
/// The harvest boundary (the model-interaction side) emits, per token, `s`
/// probe vectors `vₖ = J_nᵀ F_n^{1/2} uₖ ∈ ℝ^p` — the natural shape is
/// `probes[n, i, k] = (vₖ)ᵢ`, an `(n_rows, p, probes)` stack. This assembles the
/// `(n_rows, p · probes)` row-major matrix `u[n, i·probes + k] = probes[n, i, k]`
/// that the constructor consumes so that column `k` of the per-row factor `U_n`
/// is exactly probe `vₖ` and `M_n = U_n U_nᵀ = Σₖ vₖ vₖᵀ ≈ G_n`.
///
/// This is a pure repack of the standard C-order flattening; it exists so the
/// harvest → metric seam is a single named, validated Rust surface rather than
/// an ad-hoc reshape at each call site. Errors on non-finite entries so the
/// failure is caught here rather than deep in [`normalize_fisher_rao_blocks`].
pub fn pack_probe_factors(probes: ndarray::ArrayView3<'_, f64>) -> Result<Array2<f64>, String> {
    let (n_rows, p, s) = probes.dim();
    if s == 0 {
        return Err("pack_probe_factors: need at least one probe (s == 0)".to_string());
    }
    if !probes.iter().all(|v| v.is_finite()) {
        return Err("pack_probe_factors: probe entries must be finite".to_string());
    }
    let mut u = Array2::<f64>::zeros((n_rows, p * s));
    for n in 0..n_rows {
        for i in 0..p {
            for k in 0..s {
                u[[n, i * s + k]] = probes[[n, i, k]];
            }
        }
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ── RowMetric::euclidean ──────────────────────────────────────────────────

    #[test]
    fn euclidean_metric_has_correct_dimensions() {
        let m = RowMetric::euclidean(5, 3).unwrap();
        assert_eq!(m.n_rows(), 5);
        assert_eq!(m.p_out(), 3);
        assert_eq!(m.metric_rank(), 3);
    }

    #[test]
    fn euclidean_metric_traces_equal_p() {
        let p = 4_usize;
        let m = RowMetric::euclidean(3, p).unwrap();
        for tr in m.row_traces().iter() {
            assert!((*tr - p as f64).abs() < 1e-14, "trace {tr} != p={p}");
        }
    }

    #[test]
    fn euclidean_provenance_is_euclidean() {
        let m = RowMetric::euclidean(1, 2).unwrap();
        assert_eq!(m.provenance(), MetricProvenance::Euclidean);
    }

    #[test]
    fn euclidean_does_not_whiten_likelihood() {
        let m = RowMetric::euclidean(1, 2).unwrap();
        assert!(!m.whitens_likelihood());
    }

    #[test]
    fn euclidean_does_not_drive_gauge() {
        let m = RowMetric::euclidean(1, 2).unwrap();
        assert!(!m.drives_gauge());
    }

    #[test]
    fn euclidean_is_not_output_fisher_like() {
        let m = RowMetric::euclidean(1, 2).unwrap();
        assert!(!m.is_output_fisher_like());
    }

    #[test]
    fn euclidean_solver_floor_is_zero() {
        let m = RowMetric::euclidean(1, 2).unwrap();
        assert_eq!(m.solver_floor(), 0.0);
    }

    #[test]
    fn euclidean_to_weight_field_is_identity() {
        let m = RowMetric::euclidean(1, 2).unwrap();
        assert!(matches!(m.to_weight_field(), WeightField::Identity));
    }

    #[test]
    fn euclidean_whiten_residual_is_passthrough() {
        let m = RowMetric::euclidean(1, 3).unwrap();
        let r = array![1.0_f64, 2.0, 3.0];
        let w = m.whiten_residual_row(0, r.view());
        assert_eq!(w, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn euclidean_factor_entry_is_identity() {
        let m = RowMetric::euclidean(1, 3).unwrap();
        assert_eq!(m.factor_entry(0, 0, 0), 1.0);
        assert_eq!(m.factor_entry(0, 1, 1), 1.0);
        assert_eq!(m.factor_entry(0, 2, 2), 1.0);
        assert_eq!(m.factor_entry(0, 0, 1), 0.0);
        assert_eq!(m.factor_entry(0, 1, 0), 0.0);
    }

    #[test]
    fn euclidean_quad_form_is_squared_norm() {
        let m = RowMetric::euclidean(1, 3).unwrap();
        let r = array![1.0_f64, 2.0, 2.0];
        assert!((m.quad_form(0, r.view()) - 9.0).abs() < 1e-14);
    }

    // ── MetricProvenance predicates ───────────────────────────────────────────

    #[test]
    fn output_fisher_drives_gauge_but_not_likelihood() {
        let u = Arc::new(array![[1.0_f64]]);
        let m = RowMetric::output_fisher(u, 1, 1).unwrap();
        assert!(m.drives_gauge());
        assert!(!m.whitens_likelihood());
        assert!(m.is_output_fisher_like());
    }

    #[test]
    fn whitened_structured_whitens_likelihood_and_drives_gauge() {
        let u = Arc::new(array![[1.0_f64]]);
        let m = RowMetric::whitened_structured(u, 1, 1).unwrap();
        assert!(m.whitens_likelihood());
        assert!(m.drives_gauge());
        assert!(!m.is_output_fisher_like());
    }

    #[test]
    fn behavioral_fisher_whitens_likelihood_and_drives_gauge() {
        // The Rung-1 deliberate GLS metric: unlike the gauge-only OutputFisher,
        // it whitens the reconstruction likelihood.
        let u = Arc::new(array![[1.0_f64, 0.5]]); // p=1, probes=2
        let m = RowMetric::behavioral_fisher(u, 1, 2).unwrap();
        assert!(m.whitens_likelihood());
        assert!(m.drives_gauge());
        assert_eq!(m.provenance(), MetricProvenance::BehavioralFisher { probes: 2 });
        assert_eq!(m.metric_rank(), 2);
    }

    #[test]
    fn behavioral_fisher_quad_form_is_probe_sum() {
        // p=2, s=2 probes v1=(1,0), v2=(0,2) → G = diag(1,4);
        // e=(3,1) → eᵀGe = 9·1 + 1·4 = 13 = Σ (vᵢᵀe)² = 3² + 2² = 13.
        // Column-major-within-row layout U[i,k]=u[i*probes+k]:
        //   U[0,0]=1 U[0,1]=0  U[1,0]=0 U[1,1]=2
        let u = Arc::new(array![[1.0_f64, 0.0, 0.0, 2.0]]);
        let m = RowMetric::behavioral_fisher(u, 2, 2).unwrap();
        let e = array![3.0_f64, 1.0];
        assert!((m.quad_form(0, e.view()) - 13.0).abs() < 1e-12);
    }

    #[test]
    fn behavioral_fisher_g_identity_reproduces_euclidean_quad_form() {
        // GLS with G=I must reduce to plain MSE. Identity probes (s=p, U=I_p)
        // ⇒ M_n = I ⇒ quad_form == ‖e‖², matching Euclidean bit-for-bit, and
        // metric_rank == p so the whitened residual-dof accounting is unchanged.
        let p = 3;
        let mut u = Array2::<f64>::zeros((1, p * p));
        for i in 0..p {
            u[[0, i * p + i]] = 1.0;
        }
        let bf = RowMetric::behavioral_fisher(Arc::new(u), p, p).unwrap();
        let euc = RowMetric::euclidean(1, p).unwrap();
        let e = array![1.5_f64, -2.0, 0.25];
        assert_eq!(bf.metric_rank(), euc.metric_rank());
        assert!((bf.quad_form(0, e.view()) - euc.quad_form(0, e.view())).abs() < 1e-14);
        // and whitened residual is the residual itself (identity whitening)
        assert_eq!(bf.whiten_residual_row(0, e.view()), vec![1.5, -2.0, 0.25]);
    }

    #[test]
    fn pack_probe_factors_matches_manual_layout() {
        use ndarray::Array3;
        // n=1, p=2, s=2: probes[0,i,k] = v_k[i]; v0=(1,3), v1=(2,4)
        let mut probes = Array3::<f64>::zeros((1, 2, 2));
        probes[[0, 0, 0]] = 1.0; // v0[0]
        probes[[0, 1, 0]] = 3.0; // v0[1]
        probes[[0, 0, 1]] = 2.0; // v1[0]
        probes[[0, 1, 1]] = 4.0; // v1[1]
        let u = pack_probe_factors(probes.view()).unwrap();
        // Layout U[i,k] = u[i*s + k]: [v0[0],v1[0], v0[1],v1[1]] = [1,2,3,4]
        assert_eq!(u.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        // Round-trips into a valid metric whose G = v0 v0ᵀ + v1 v1ᵀ.
        let m = RowMetric::behavioral_fisher(Arc::new(u), 2, 2).unwrap();
        // e=(1,0): eᵀGe = v0[0]²+v1[0]² = 1+4 = 5.
        let e = array![1.0_f64, 0.0];
        assert!((m.quad_form(0, e.view()) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn pack_probe_factors_rejects_zero_probes() {
        use ndarray::Array3;
        let probes = Array3::<f64>::zeros((2, 3, 0));
        assert!(pack_probe_factors(probes.view()).is_err());
    }

    #[test]
    fn output_fisher_downstream_is_output_fisher_like() {
        let u = Arc::new(array![[1.0_f64]]);
        let m = RowMetric::output_fisher_downstream(u, 1, 1).unwrap();
        assert!(m.is_output_fisher_like());
        assert!(m.drives_gauge());
    }

    // ── WeightField::project_jac_row_with_u ──────────────────────────────────

    #[test]
    fn project_jac_with_identity_returns_jac() {
        // p=2, rank=2, d=2; U=I_2, J=[[1,2],[3,4]] → M = U^T J = J
        let u_row = [1.0_f64, 0.0, 0.0, 1.0]; // U[i,k]=u[i*rank+k], I_2
        let j_row = [1.0_f64, 2.0, 3.0, 4.0]; // J[i,a]=j[i*d+a]
        let m = WeightField::project_jac_row_with_u(&u_row, &j_row, 2, 2, 2);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-14);
        assert!((m[[0, 1]] - 2.0).abs() < 1e-14);
        assert!((m[[1, 0]] - 3.0).abs() < 1e-14);
        assert!((m[[1, 1]] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn project_jac_with_zeros_returns_zero_matrix() {
        let u_row = [0.0_f64, 0.0];
        let j_row = [1.0_f64, 2.0];
        let m = WeightField::project_jac_row_with_u(&u_row, &j_row, 2, 1, 1);
        assert_eq!(m[[0, 0]], 0.0);
    }
}
