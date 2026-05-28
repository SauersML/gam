//! Family-level primary Jacobian + row-Hessian metric for identifiability work.
//!
//! Identifiability for cross-block flex anchors lives on the *linearized*
//! likelihood row map (the family Jacobian), not on raw block columns under a
//! scalar IRLS-W metric. Survival_marginal_slope's row map has four to five
//! coupled primary channels (entry survival index `q0`, exit survival /
//! event-density index `q1`, exit-time derivative `qd1`, raw log-slope `g`,
//! and an additive scalar "deviation block" perturbation that score-warp and
//! link-dev attach to the exit-time index). Bernoulli_marginal_slope has a
//! single scalar channel that recovers the old IRLS-W path exactly.
//!
//! T1 (this module) only defines the abstraction and the data structures used
//! to evaluate it. Wiring into the cross-block residualizer happens in T2.
//!
//! # BlockEffectiveJacobian — IFT-derived Jacobians for flex blocks
//!
//! Score-warp and link-deviation blocks parameterise the survival family via
//! the implicit function theorem (IFT).  For a constraint
//!
//!   Γ_i(a_ri, g_i, h, w) − q_ri = 0,   where a(β) is implicitly defined,
//!
//! the effective Jacobian (∂eta_r/∂β_k) has the form
//!
//!   ∂eta_r/∂β_k = (E_a/Γ_a)·∂q_r/∂β_k
//!                 + (E_g − E_a·Γ_g/Γ_a)·∂g/∂β_k
//!                 + (E_h − E_a·Γ_h/Γ_a)·∂h/∂β_k   [score-warp]
//!                 + (E_w − E_a·Γ_w/Γ_a)·∂w/∂β_k   [link-dev]
//!
//! At rigid initialisation (a=0, g=0, h=0, w=0) the correction terms
//! simplify and `∂h/∂β_score` reduces to the local-cubic basis evaluated
//! at z (score-warp) and `∂w/∂β_link` reduces to the local-cubic basis
//! evaluated at `u = a + g·z` (link-dev).
//!
//! [`BlockEffectiveJacobian`] is a marker super-trait over
//! [`crate::families::identifiability_compiler::RowJacobianOperator`] that
//! documents this IFT provenance and provides a human-readable block label.
//! Concrete impls for SMGS are [`ScoreWarpEffectiveJacobian`] and
//! [`LinkDevEffectiveJacobian`]; the Bernoulli analogues follow the same
//! `QChannelBlockOperator` shape with K=1.

use ndarray::Array3;

use crate::linalg::matrix::DesignMatrix;

/// Identifier for a block contributing to a family's primary-Jacobian map.
/// The wrapper is intentionally opaque; the caller (T2) chooses the numbering
/// scheme it needs (per-block id, anchor id, etc.).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BlockTag(pub u32);

/// Per-row PSD metric on a family's primary channels plus the block-level
/// channel-Jacobian operator. Concrete implementations carry only the data the
/// trait needs to evaluate — they do not own the family struct.
pub trait BlockPrimaryJacobian {
    /// Number of primary channels in this family's row map.
    fn n_channels(&self) -> usize;

    /// Number of training rows.
    fn n_rows(&self) -> usize;

    /// For block `block`: return one entry per primary channel; entry `c`
    /// is `None` iff the block contributes nothing to channel `c`; otherwise
    /// it is the `(n_rows × p_block)` design whose rows are the linearized
    /// contribution from that block to that channel. The returned designs
    /// share storage with the family's owned designs via `DesignMatrix`'s
    /// internal `Arc`/operator references where possible.
    fn channel_contributions(&self, block: BlockTag) -> Vec<Option<DesignMatrix>>;

    /// Per-row symmetric PSD Hessian of the family's negative log-likelihood
    /// in the primary-channel coordinates, evaluated at the fit-time pilot
    /// state. Shape `(n_rows, n_channels, n_channels)`, symmetric in the last
    /// two axes.
    fn row_channel_metric(&self) -> Array3<f64>;
}

// ── Survival marginal-slope (5 channels) ──────────────────────────────

/// Survival_marginal_slope primary channels.
///
/// `q0`, `q1`, `qd1`, `g` come from the rigid four-primary row calculus
/// (`row_primary_closed_form` in `survival_marginal_slope.rs`):
///
///   eta0 = q0 · c(g) + s_f · g · z       (entry probit index)
///   eta1 = q1 · c(g) + s_f · g · z       (exit probit index)
///   ad1  = qd1 · c(g)                    (positive time derivative at exit)
///
/// `eta_scalar` is an additive scalar perturbation that score-warp /
/// link-dev attach to the exit index `eta1`. It does not enter `eta0` or
/// `ad1`, mirroring how those flex blocks attach in the bernoulli analog
/// and how their per-row designs are concatenated into the joint exit-time
/// linear predictor.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SurvivalPrimaryChannel {
    EntryLocation = 0,
    ExitLocation = 1,
    ExitDerivative = 2,
    Logslope = 3,
    EtaScalar = 4,
}

impl SurvivalPrimaryChannel {
    pub const COUNT: usize = 5;
    pub const ALL: [SurvivalPrimaryChannel; Self::COUNT] = [
        Self::EntryLocation,
        Self::ExitLocation,
        Self::ExitDerivative,
        Self::Logslope,
        Self::EtaScalar,
    ];
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Tags survival blocks the cross-block compiler residualizes against. The
/// numerical `BlockTag(u32)` wraps these so T2 can extend the enumeration
/// without touching the trait signature.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SurvivalBlock {
    Time = 0,
    Marginal = 1,
    Logslope = 2,
    ScoreWarp = 3,
    LinkDev = 4,
}

impl SurvivalBlock {
    #[inline]
    pub const fn tag(self) -> BlockTag {
        BlockTag(self as u32)
    }
}

/// Survival_marginal_slope primary-Jacobian operator.
///
/// Holds the per-block training-row designs and the per-row pilot Hessian
/// (built once by the family fit driver via
/// `survival_marginal_slope::row_state_hessian_5ch`). Channel mappings:
///
///   - Time block contributes to EntryLocation (`design_entry`),
///     ExitLocation (`design_exit`), ExitDerivative
///     (`design_derivative_exit`).
///   - Marginal block contributes to EntryLocation and ExitLocation
///     (same design — additive to both q0 and q1).
///   - Logslope block contributes to Logslope only (the log-slope basis).
///   - ScoreWarp and LinkDev each contribute to EtaScalar at their
///     training-row designs.
#[derive(Clone, Debug)]
pub struct SurvivalPrimaryJacobian {
    pub n_rows: usize,
    pub time_design_entry: DesignMatrix,
    pub time_design_exit: DesignMatrix,
    pub time_design_derivative_exit: DesignMatrix,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    pub score_warp_design: Option<DesignMatrix>,
    pub link_dev_design: Option<DesignMatrix>,
    /// Shape `(n_rows, 5, 5)`. Pilot row-Hessian of `-ℓ` in the channel
    /// ordering of `SurvivalPrimaryChannel`.
    pub row_metric: Array3<f64>,
}

impl BlockPrimaryJacobian for SurvivalPrimaryJacobian {
    #[inline]
    fn n_channels(&self) -> usize {
        SurvivalPrimaryChannel::COUNT
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn channel_contributions(&self, block: BlockTag) -> Vec<Option<DesignMatrix>> {
        let mut out: Vec<Option<DesignMatrix>> =
            (0..SurvivalPrimaryChannel::COUNT).map(|_| None).collect();
        match block {
            t if t == SurvivalBlock::Time.tag() => {
                out[SurvivalPrimaryChannel::EntryLocation.index()] =
                    Some(self.time_design_entry.clone());
                out[SurvivalPrimaryChannel::ExitLocation.index()] =
                    Some(self.time_design_exit.clone());
                out[SurvivalPrimaryChannel::ExitDerivative.index()] =
                    Some(self.time_design_derivative_exit.clone());
            }
            t if t == SurvivalBlock::Marginal.tag() => {
                out[SurvivalPrimaryChannel::EntryLocation.index()] =
                    Some(self.marginal_design.clone());
                out[SurvivalPrimaryChannel::ExitLocation.index()] =
                    Some(self.marginal_design.clone());
            }
            t if t == SurvivalBlock::Logslope.tag() => {
                out[SurvivalPrimaryChannel::Logslope.index()] = Some(self.logslope_design.clone());
            }
            t if t == SurvivalBlock::ScoreWarp.tag() => {
                if let Some(d) = self.score_warp_design.as_ref() {
                    out[SurvivalPrimaryChannel::EtaScalar.index()] = Some(d.clone());
                }
            }
            t if t == SurvivalBlock::LinkDev.tag() => {
                if let Some(d) = self.link_dev_design.as_ref() {
                    out[SurvivalPrimaryChannel::EtaScalar.index()] = Some(d.clone());
                }
            }
            _ => {}
        }
        out
    }

    #[inline]
    fn row_channel_metric(&self) -> Array3<f64> {
        self.row_metric.clone()
    }
}

// ── Bernoulli marginal-slope (1 channel — EtaScalar) ──────────────────

/// Bernoulli_marginal_slope primary channels. The single channel is the
/// exit-time probit index `η`; the row metric is the existing IRLS W
/// vector `spec.weights[i] · φ(η_i)² / (Φ(η_i)·(1−Φ(η_i)))`. This trait
/// impl is a faithful re-expression of the existing scalar W path — T2 can
/// switch the bernoulli cross-block residualizer onto the trait without
/// changing behavior.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BernoulliPrimaryChannel {
    EtaScalar = 0,
}

impl BernoulliPrimaryChannel {
    pub const COUNT: usize = 1;
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Tags bernoulli blocks the cross-block compiler residualizes against.
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BernoulliBlock {
    Marginal = 0,
    Logslope = 1,
    ScoreWarp = 2,
    LinkDev = 3,
}

impl BernoulliBlock {
    #[inline]
    pub const fn tag(self) -> BlockTag {
        BlockTag(self as u32)
    }
}

/// Bernoulli_marginal_slope primary-Jacobian operator. Single EtaScalar
/// channel; row metric = pilot IRLS W from the existing W-metric pilot.
#[derive(Clone, Debug)]
pub struct BernoulliPrimaryJacobian {
    pub n_rows: usize,
    pub marginal_design: DesignMatrix,
    pub logslope_design: DesignMatrix,
    pub score_warp_design: Option<DesignMatrix>,
    pub link_dev_design: Option<DesignMatrix>,
    /// Per-row pilot IRLS W weights; shape `(n_rows,)`. The trait exposes
    /// this as the diagonal of a `(n_rows, 1, 1)` row Hessian, reducing
    /// exactly to the existing scalar W metric.
    pub pilot_irls_w: ndarray::Array1<f64>,
}

impl BlockPrimaryJacobian for BernoulliPrimaryJacobian {
    #[inline]
    fn n_channels(&self) -> usize {
        BernoulliPrimaryChannel::COUNT
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn channel_contributions(&self, block: BlockTag) -> Vec<Option<DesignMatrix>> {
        let mut out: Vec<Option<DesignMatrix>> =
            (0..BernoulliPrimaryChannel::COUNT).map(|_| None).collect();
        match block {
            t if t == BernoulliBlock::Marginal.tag() => {
                out[BernoulliPrimaryChannel::EtaScalar.index()] =
                    Some(self.marginal_design.clone());
            }
            t if t == BernoulliBlock::Logslope.tag() => {
                out[BernoulliPrimaryChannel::EtaScalar.index()] =
                    Some(self.logslope_design.clone());
            }
            t if t == BernoulliBlock::ScoreWarp.tag() => {
                if let Some(d) = self.score_warp_design.as_ref() {
                    out[BernoulliPrimaryChannel::EtaScalar.index()] = Some(d.clone());
                }
            }
            t if t == BernoulliBlock::LinkDev.tag() => {
                if let Some(d) = self.link_dev_design.as_ref() {
                    out[BernoulliPrimaryChannel::EtaScalar.index()] = Some(d.clone());
                }
            }
            _ => {}
        }
        out
    }

    fn row_channel_metric(&self) -> Array3<f64> {
        let n = self.n_rows;
        assert_eq!(
            self.pilot_irls_w.len(),
            n,
            "BernoulliPrimaryJacobian: pilot_irls_w length {} does not match n_rows {}",
            self.pilot_irls_w.len(),
            n,
        );
        let mut metric = Array3::<f64>::zeros((n, 1, 1));
        for i in 0..n {
            metric[[i, 0, 0]] = self.pilot_irls_w[i];
        }
        metric
    }
}

// ── BlockEffectiveJacobian — IFT-derived marker trait ─────────────────

/// Marker super-trait for row-Jacobian operators whose column layout
/// is derived from the implicit function theorem (IFT) at the family's
/// current linearisation state.
///
/// Implementors carry the same interface as
/// [`crate::families::identifiability_compiler::RowJacobianOperator`]
/// and additionally expose a human-readable `block_label` so the
/// unified audit can attribute any dropped column to the correct
/// flex-block name in its `DroppedColumn` records.
///
/// At rigid initialisation (a=0, g=0, h=0, w=0) the IFT correction
/// terms vanish and the effective Jacobian reduces to the local-cubic
/// basis evaluated at the block's linearisation argument (z for
/// score-warp, `u = a + g·z` for link-dev).  The concrete impls
/// below capture this via the `dq` / `dqd1` matrices passed to
/// [`crate::families::survival_marginal_slope_identifiability::QChannelBlockOperator`].
pub trait BlockEffectiveJacobian:
    crate::families::identifiability_compiler::RowJacobianOperator
{
    /// Human-readable label identifying this flex block.
    /// Used by the unified audit to populate [`crate::solver::identifiability_audit::DroppedColumn::block`].
    fn block_label(&self) -> &str;
}

/// IFT-derived effective Jacobian for the survival score-warp-deviation
/// block.
///
/// At rigid initialisation the channel contributions reduce to:
///
/// - q0 channel: `dq[i, j]`  (local-cubic basis row)
/// - q1 channel: `dq[i, j]`  (same, enter and exit indices share the block)
/// - qd1 channel: `dqd1[i, j]` (derivative of the cubic basis wrt t, evaluated at exit t)
/// - g channel: 0
///
/// This matches the [`crate::families::survival_marginal_slope_identifiability::QChannelBlockOperator`]
/// channel routing exactly.
pub struct ScoreWarpEffectiveJacobian {
    inner: crate::families::survival_marginal_slope_identifiability::QChannelBlockOperator,
}

impl ScoreWarpEffectiveJacobian {
    /// Construct from the (n × p) local-cubic-basis design and its
    /// time derivative evaluated at training rows.
    pub fn new(dq: ndarray::Array2<f64>, dqd1: ndarray::Array2<f64>) -> Self {
        Self {
            inner: crate::families::survival_marginal_slope_identifiability::QChannelBlockOperator::new(
                dq, dqd1,
            ),
        }
    }
}

impl crate::families::identifiability_compiler::RowJacobianOperator
    for ScoreWarpEffectiveJacobian
{
    fn k(&self) -> usize {
        self.inner.k()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        self.inner.apply_row(row, delta_beta, out);
    }
    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.inner.evaluate_full()
    }
}

impl BlockEffectiveJacobian for ScoreWarpEffectiveJacobian {
    fn block_label(&self) -> &str {
        "score_warp_dev"
    }
}

/// IFT-derived effective Jacobian for the survival link-deviation block.
///
/// At rigid initialisation `u = a + g·z = 0` and the channel contributions
/// reduce to:
///
/// - q0 channel: `dq[i, j]`  (local-cubic basis in u, evaluated at u≈0=z)
/// - q1 channel: `dq[i, j]`
/// - qd1 channel: `dqd1[i, j]`
/// - g channel: 0
///
/// Same channel routing as [`ScoreWarpEffectiveJacobian`]; the distinction
/// is purely semantic (the linearisation argument is u, not z).
pub struct LinkDevEffectiveJacobian {
    inner: crate::families::survival_marginal_slope_identifiability::QChannelBlockOperator,
}

impl LinkDevEffectiveJacobian {
    /// Construct from the (n × p) local-cubic-basis-in-u design and its
    /// derivative, evaluated at the pilot `u` (= exit-location index) at
    /// training rows.
    pub fn new(dq: ndarray::Array2<f64>, dqd1: ndarray::Array2<f64>) -> Self {
        Self {
            inner: crate::families::survival_marginal_slope_identifiability::QChannelBlockOperator::new(
                dq, dqd1,
            ),
        }
    }
}

impl crate::families::identifiability_compiler::RowJacobianOperator
    for LinkDevEffectiveJacobian
{
    fn k(&self) -> usize {
        self.inner.k()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        self.inner.apply_row(row, delta_beta, out);
    }
    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.inner.evaluate_full()
    }
}

impl BlockEffectiveJacobian for LinkDevEffectiveJacobian {
    fn block_label(&self) -> &str {
        "link_dev"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn bernoulli_metric_reduces_to_irls_w() {
        let w = Array1::from(vec![0.25_f64, 0.1875, 0.0625]);
        let p = 2usize;
        let m = ndarray::Array2::<f64>::zeros((3, p));
        let dm = DesignMatrix::from(m.clone());
        let bj = BernoulliPrimaryJacobian {
            n_rows: 3,
            marginal_design: dm.clone(),
            logslope_design: dm.clone(),
            score_warp_design: None,
            link_dev_design: None,
            pilot_irls_w: w.clone(),
        };
        let metric = bj.row_channel_metric();
        assert_eq!(metric.shape(), &[3, 1, 1]);
        for i in 0..3 {
            assert!((metric[[i, 0, 0]] - w[i]).abs() < 1e-15);
        }
        // No contribution from absent flex blocks.
        let contrib = bj.channel_contributions(BernoulliBlock::ScoreWarp.tag());
        assert!(contrib[0].is_none());
        // Marginal contributes to the single channel.
        let contrib = bj.channel_contributions(BernoulliBlock::Marginal.tag());
        assert!(contrib[0].is_some());
    }

    #[test]
    fn survival_channel_routing() {
        let n = 4;
        let pt = 3;
        let pm = 2;
        let pg = 2;
        let mk = |p: usize| DesignMatrix::from(ndarray::Array2::<f64>::zeros((n, p)));
        let row_metric = Array3::<f64>::zeros((
            n,
            SurvivalPrimaryChannel::COUNT,
            SurvivalPrimaryChannel::COUNT,
        ));
        let sj = SurvivalPrimaryJacobian {
            n_rows: n,
            time_design_entry: mk(pt),
            time_design_exit: mk(pt),
            time_design_derivative_exit: mk(pt),
            marginal_design: mk(pm),
            logslope_design: mk(pg),
            score_warp_design: None,
            link_dev_design: None,
            row_metric,
        };
        assert_eq!(sj.n_channels(), 5);
        let time = sj.channel_contributions(SurvivalBlock::Time.tag());
        assert!(time[SurvivalPrimaryChannel::EntryLocation.index()].is_some());
        assert!(time[SurvivalPrimaryChannel::ExitLocation.index()].is_some());
        assert!(time[SurvivalPrimaryChannel::ExitDerivative.index()].is_some());
        assert!(time[SurvivalPrimaryChannel::Logslope.index()].is_none());
        assert!(time[SurvivalPrimaryChannel::EtaScalar.index()].is_none());
        let marg = sj.channel_contributions(SurvivalBlock::Marginal.tag());
        assert!(marg[SurvivalPrimaryChannel::EntryLocation.index()].is_some());
        assert!(marg[SurvivalPrimaryChannel::ExitLocation.index()].is_some());
        assert!(marg[SurvivalPrimaryChannel::ExitDerivative.index()].is_none());
    }
}
