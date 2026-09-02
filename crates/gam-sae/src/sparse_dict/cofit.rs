//! Joint co-fitting of the linear block tier and the curved chart tier
//! (residual-orthogonality trap closure).
//!
//! # The trap this closes
//!
//! The block-chart compose lane ([`super::block_chart`]) fits curved charts to
//! the **least-squares residual** of a frozen linear dictionary. But an LS
//! residual is orthogonal to the fitted span: the linear tiling has already
//! absorbed the local tangent *and* the curvature into where it placed its
//! atoms, so what is left in the residual is high-frequency sawtooth
//! quantisation noise between atoms — exactly the thing a *smooth* chart cannot
//! represent. The one-shot fit-curved-on-linear-residual protocol therefore
//! hands the curved lane a target from which the very structure it is meant to
//! find has been removed by construction.
//!
//! # The fix: monotone two-block coordinate descent
//!
//! Model the reconstruction as two **additive** tiers,
//! `x̂ = L(codes) + C(charts)`, and alternate two block solves that both descend
//! the SAME penalised joint objective
//!
//! ```text
//!   J(codes, charts) = ‖target − L − C‖²_F  +  λ_lin · ‖codes‖²_F
//! ```
//!
//! (the linear tier's ridge is explicit in `J`; the curved tier's complexity
//! penalty is realised as the compose lane's cross-fit BIC acceptance charge,
//! which admits a chart only when its cross-validated deviance gain exceeds its
//! `½·d_eff·log n_eff` information charge — a descriptive per-chart BIC gate, not
//! an FDR-controlled e-BH discovery — surfaced per round as
//! [`CofitRound::curved_charge`]).
//!
//! * **Block A — linear tier refit.** With the charts (hence `C`) held fixed and
//!   the block routing frozen, re-solve the per-row active-set ridge
//!   least-squares codes against the *chart-adjusted* target `target − C`. This
//!   is an exact block minimisation of `J` over the linear codes (the previous
//!   codes are always feasible), so it is **provably monotone**: `J` cannot
//!   increase. It is precisely the step the one-shot protocol never takes — the
//!   linear tier stops chasing the curvature that the chart already explains, so
//!   its atoms are freed to model the genuinely linear part.
//! * **Block B — curved joint fit.** With the linear codes held fixed, re-fit the
//!   charts against the *linear-adjusted* target `target − L` through the
//!   existing curved surface ([`compose_block_coordinate_charts`]). The compose
//!   lane's acceptance is cross-fit gated rather than a pure held-in minimiser,
//!   so this step is **guarded**: the candidate is committed only when it does
//!   not increase `J`. The previous chart set is always available as the
//!   fallback, so the round is monotone by construction either way.
//!
//! Each committed round therefore has `J[r] ≤ J[r-1]` up to numerical
//! tolerance. Convergence requires an entire deterministic A/B replay to leave
//! codes, chart ownership, and both reconstruction components bit-identical; an
//! objective stall alone never mints a fit. The curved solver's internals are
//! **untouched** — it is called through its existing public surface with an
//! adjusted target.

use ndarray::{Array2, Array3};

use super::block_chart::{BlockChartComposeConfig, BlockChartComposeResult};

/// Configuration for [`cofit_block_and_curved`].
#[derive(Clone, Debug)]
pub struct CofitConfig {
    /// Maximum number of complete deterministic A/B replays. Exhaustion is a
    /// non-convergence error; a [`CofitReport`] is created only after one replay
    /// leaves the complete fitted state bit-identical.
    pub max_rounds: usize,
    /// Linear-tier ridge `λ_lin` on the per-row active-set least-squares codes.
    pub code_ridge: f32,
    /// Relative slack for the monotone-non-increase invariant. A round whose
    /// objective exceeds the previous by more than
    /// `monotone_slack · (|J_prev| + 1)` is a bug and aborts the fit.
    pub monotone_slack: f64,
    /// Curved-tier compose configuration. Its `block_size`, `block_topk` and
    /// `gamma` are overwritten from the passed routing/frames each round so the
    /// tiers always agree on geometry; `residual_target` is forced on (the
    /// co-fit *is* the principled residual protocol).
    pub chart: BlockChartComposeConfig,
}

impl Default for CofitConfig {
    fn default() -> Self {
        Self {
            max_rounds: 256,
            code_ridge: 1.0e-6,
            monotone_slack: 1.0e-6,
            chart: BlockChartComposeConfig::default(),
        }
    }
}

/// Per-round telemetry of the co-fit alternation.
#[derive(Clone, Debug)]
pub struct CofitRound {
    /// Round index (`0` is the one-shot fit-curved-on-linear-residual baseline;
    /// `≥1` are A/B alternation rounds).
    pub round: usize,
    /// Joint objective `J = ‖target − (L+C)‖²_F + λ_lin‖codes‖²_F` at round end.
    pub objective: f64,
    /// Reconstruction term `‖target − (L+C)‖²_F` (Frobenius SSE).
    pub recon_sse: f64,
    /// Linear-tier ridge energy `λ_lin · ‖codes‖²_F`.
    pub linear_ridge: f64,
    /// Total BIC complexity charge (`Σ ½·d_eff·log n_eff`) of the descriptively
    /// accepted charts this round — the curved tier's information penalty,
    /// enforced as a per-chart BIC acceptance gate (not an FDR-controlled e-BH
    /// discovery).
    pub curved_charge: f64,
    /// Composed explained variance (`1 − RSS/TSS`, mean baseline).
    pub explained_variance: f64,
    /// Number of accepted curved charts (single blocks + pairs) this round.
    pub n_accepted_charts: usize,
    /// Whether the linear block A step strictly reduced the objective this round
    /// (always a non-increase; `false` when it was already at the block optimum).
    pub linear_improved: bool,
    /// Whether the curved block B candidate was committed (`false` = the guard
    /// kept the previous chart set because the candidate did not reduce `J`).
    pub curved_committed: bool,
}

/// Result of a co-fit run.
#[derive(Clone, Debug)]
pub struct CofitReport {
    /// Composed reconstruction `L + C`, `N×P`.
    pub reconstructed: Array2<f32>,
    /// Linear-tier reconstruction `L` over the chart-*unowned* blocks, `N×P`
    /// (the blocks a chart replaced are excluded — they live in `C`).
    pub linear_reconstruction: Array2<f32>,
    /// Additive curved correction `C = composed − L`, `N×P` (the lifted chart
    /// coordinates of the accepted, chart-owned blocks).
    pub curved_correction: Array2<f32>,
    /// Refit linear-tier codes, `N×k×b`, at the frozen routing.
    pub codes: Array3<f32>,
    /// Final composed explained variance.
    pub explained_variance: f64,
    /// Per-round telemetry (index 0 is the one-shot baseline).
    pub rounds: Vec<CofitRound>,
    /// Final curved compose result (chart records, acceptances, screens).
    pub compose: BlockChartComposeResult,
}

