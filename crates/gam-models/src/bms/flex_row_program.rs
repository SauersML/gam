//! Canonical semantic program for one BMS FLEX row.
//!
//! The program is deliberately independent of a derivative representation.
//! It freezes the scalar root, spline-span stacks, and unary derivative stacks
//! once, then interprets the same typed expression in any
//! [`RuntimeJetScalar`]. Optimized moment lowerings consume
//! [`BmsFlexProgramPoint`] instead of restating the row coordinates, so the
//! empirical `O(G + cells * r^2)` schedule and the generic jet evaluator share
//! one semantic input contract.

use super::hessian_paths::PrimarySlices;
use gam_math::jet_scalar::{RuntimeJetScalar, filtered_implicit_solve_runtime_scalar};
use ndarray::Array1;

/// Declarative second-order lowering of the calibration-CDF semantic node.
///
/// Backends provide only the primitive `D(first)` and
/// `D(explicit_second) - Q(first, first)` contractions. This schedule owns
/// which contractions exist and their index traversal, so empirical moments,
/// standard-normal moments, and CUDA emission cannot maintain separate loop
/// formulas.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BmsFlexCalibrationOrder2Node {
    InterceptFirst,
    InterceptSecond,
    PrimaryFirst { primary: usize },
    InterceptPrimarySecond { primary: usize },
    PrimaryPairSecond { left: usize, right: usize },
}

/// Backend lowering phases for the Order2 calibration schedule.
///
/// CPU moment consumers expand these phases into indexed nodes. Generated
/// backends may preserve a phase as one runtime loop, avoiding max-width
/// source unrolling while retaining the same semantic phase order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BmsFlexCalibrationOrder2Phase {
    InterceptFirst,
    InterceptSecond,
    PrimaryFirstAndInterceptSecond,
    PrimaryPairSecond,
}

/// Declarative directional derivative of the Order2 calibration schedule.
///
/// `DirectionStart` lets an optimized backend compile the sparse coefficient
/// direction once, then consume every derivative node for that direction.
/// The node order is part of the semantic program: CPU moments and generated
/// device code do not own independent primary/direction loops.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BmsFlexCalibrationOrder3Node {
    DirectionStart {
        direction: usize,
    },
    InterceptDirectionalSecond {
        direction: usize,
    },
    InterceptDirectionalThird {
        direction: usize,
    },
    InterceptPrimaryDirectionalThird {
        direction: usize,
        primary: usize,
    },
    PrimaryPairDirectionalThird {
        direction: usize,
        left: usize,
        right: usize,
    },
}

/// Declarative mixed directional derivative of the Order2 calibration
/// schedule. Each `pair` indexes one backend-owned `(left, right)` direction
/// pair; the program owns every scalar, primary, and primary-pair visit within
/// it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BmsFlexCalibrationOrder4Node {
    DirectionPairStart {
        pair: usize,
    },
    InterceptMixedThird {
        pair: usize,
    },
    InterceptMixedFourth {
        pair: usize,
    },
    InterceptPrimaryMixedFourth {
        pair: usize,
        primary: usize,
    },
    PrimaryPairMixedFourth {
        pair: usize,
        left: usize,
        right: usize,
    },
}

/// Dependency-ordered row finalizer shared by optimized CPU and generated
/// device backends after calibration moments have been lowered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BmsFlexRowOrder2FinalizerNode {
    ImplicitFirst { primary: usize },
    ImplicitFirstComplete,
    ImplicitSecond { left: usize, right: usize },
    ObservedFirst { primary: usize },
    ObservedScoreSensitivity { primary: usize },
    ObservedSecond { left: usize, right: usize },
    NegLogFirst { primary: usize },
}

/// Backend lowering phases for the dependency-ordered Order2 row finalizer.
/// Indexed CPU visits and compact generated device loops are expansions of
/// this one phase stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BmsFlexRowOrder2FinalizerPhase {
    ImplicitFirst,
    ImplicitFirstComplete,
    ImplicitSecond,
    ObservedFirst,
    ObservedScoreSensitivity,
    ObservedSecond,
    NegLogFirst,
}

/// Borrowed scalar coordinates of the canonical BMS FLEX row program.
///
/// This is the zero-allocation input to optimized lowerings. The dense generic
/// interpreter compiles the same coordinates into [`BmsFlexRowProgram`], while
/// the empirical Order2 lowering keeps the grid-node loop scalar and factors
/// primary-space work outside it.
#[derive(Clone, Copy)]
pub(super) struct BmsFlexProgramPoint<'a> {
    primary: &'a PrimarySlices,
    slope: f64,
    beta_h: Option<&'a Array1<f64>>,
    beta_w: Option<&'a Array1<f64>>,
    intercept_root: f64,
    inv_f_a: f64,
    scale: f64,
    mu_stack: [f64; 5],
}

impl<'a> BmsFlexProgramPoint<'a> {
    pub(super) fn new(
        primary: &'a PrimarySlices,
        slope: f64,
        beta_h: Option<&'a Array1<f64>>,
        beta_w: Option<&'a Array1<f64>>,
        intercept_root: f64,
        inv_f_a: f64,
        scale: f64,
        mu_stack: [f64; 5],
    ) -> Result<Self, String> {
        if primary.total < 2 || primary.q >= primary.total || primary.logslope >= primary.total {
            return Err("BMS FLEX row program has an invalid primary layout".to_string());
        }
        match (primary.h.as_ref(), beta_h) {
            (Some(range), Some(beta)) if range.len() == beta.len() => {}
            (None, None) => {}
            (Some(range), Some(beta)) => {
                return Err(format!(
                    "BMS FLEX score coefficients {} != primary range {}",
                    beta.len(),
                    range.len()
                ));
            }
            _ => return Err("BMS FLEX score primary/beta presence mismatch".to_string()),
        }
        match (primary.w.as_ref(), beta_w) {
            (Some(range), Some(beta)) if range.len() == beta.len() => {}
            (None, None) => {}
            (Some(range), Some(beta)) => {
                return Err(format!(
                    "BMS FLEX link coefficients {} != primary range {}",
                    beta.len(),
                    range.len()
                ));
            }
            _ => return Err("BMS FLEX link primary/beta presence mismatch".to_string()),
        }
        if !intercept_root.is_finite() {
            return Err("BMS FLEX row program has a non-finite intercept root".to_string());
        }
        if !(inv_f_a.is_finite() && inv_f_a > 0.0) {
            return Err(format!(
                "BMS FLEX row program has invalid inverse calibration Jacobian {inv_f_a}"
            ));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!("BMS FLEX row program has invalid scale {scale}"));
        }
        Ok(Self {
            primary,
            slope,
            beta_h,
            beta_w,
            intercept_root,
            inv_f_a,
            scale,
            mu_stack,
        })
    }

    #[inline]
    pub(super) fn primary(self) -> &'a PrimarySlices {
        self.primary
    }

    #[inline]
    pub(super) fn slope(self) -> f64 {
        self.slope
    }

    #[inline]
    pub(super) fn beta_h(self) -> Option<&'a Array1<f64>> {
        self.beta_h
    }

    #[inline]
    pub(super) fn beta_w(self) -> Option<&'a Array1<f64>> {
        self.beta_w
    }

    #[inline]
    pub(super) fn intercept_root(self) -> f64 {
        self.intercept_root
    }

    #[inline]
    pub(super) fn inv_f_a(self) -> f64 {
        self.inv_f_a
    }

    #[inline]
    pub(super) fn scale(self) -> f64 {
        self.scale
    }

    #[inline]
    pub(super) fn mu_stack(self) -> [f64; 5] {
        self.mu_stack
    }
}

/// One compiled index expression
/// `scale * (a + b*z + b*score_warp + link_warp)`.
#[derive(Clone)]
pub(super) struct BmsFlexIndexProgram {
    pub(super) z: f64,
    pub(super) score_values: Vec<f64>,
    pub(super) link_stacks: Vec<[f64; 5]>,
}

/// One weighted calibration-CDF node in the row program.
#[derive(Clone)]
pub(super) struct BmsFlexCalibrationProgramNode {
    pub(super) index: BmsFlexIndexProgram,
    pub(super) weight: f64,
    pub(super) cdf_stack: [f64; 5],
}

/// The single typed BMS FLEX row expression.
///
/// Rigid BMS is the `h = w = None`, dimension-two specialization. FLEX adds
/// score-warp and link-deviation primaries. V/G/H, one-seed third, and two-seed
/// fourth channels are interpretations of [`Self::evaluate`]; optimized
/// lowerings may change storage and loop order, but consume the same
/// [`BmsFlexProgramPoint`] contract.
#[derive(Clone)]
pub(super) struct BmsFlexRowProgram {
    primary: PrimarySlices,
    pub(super) intercept_root: f64,
    pub(super) inv_f_a: f64,
    scale: f64,
    mu_stack: [f64; 5],
    calibration: Vec<BmsFlexCalibrationProgramNode>,
    observed: BmsFlexIndexProgram,
    pub(super) observed_sign: f64,
    pub(super) observed_neglog_stack: [f64; 5],
}

impl BmsFlexRowProgram {
    pub(super) fn for_each_calibration_order2(
        active_primaries: &[usize],
        need_hessian: bool,
        mut visit: impl FnMut(BmsFlexCalibrationOrder2Node),
    ) {
        let result = Self::try_for_each_calibration_order2(
            active_primaries,
            need_hessian,
            |node| -> Result<(), std::convert::Infallible> {
                visit(node);
                Ok(())
            },
        );
        match result {
            Ok(()) => {}
            Err(never) => match never {},
        }
    }

    /// Interpret the canonical sparse Order2 schedule for one calibration
    /// cell. `active_primaries` excludes the marginal target coordinate, whose
    /// `-mu(q)` derivatives are emitted by the row finalizer.
    pub(super) fn try_for_each_calibration_order2<E>(
        active_primaries: &[usize],
        need_hessian: bool,
        visit: impl FnMut(BmsFlexCalibrationOrder2Node) -> Result<(), E>,
    ) -> Result<(), E> {
        Self::try_for_each_calibration_order2_indexed(
            active_primaries.len(),
            |position| active_primaries[position],
            need_hessian,
            visit,
        )
    }

    pub(super) fn try_for_each_calibration_order2_contiguous<E>(
        active_primaries: std::ops::Range<usize>,
        need_hessian: bool,
        visit: impl FnMut(BmsFlexCalibrationOrder2Node) -> Result<(), E>,
    ) -> Result<(), E> {
        let start = active_primaries.start;
        Self::try_for_each_calibration_order2_indexed(
            active_primaries.len(),
            |position| start + position,
            need_hessian,
            visit,
        )
    }

    fn try_for_each_calibration_order2_indexed<E>(
        active_count: usize,
        active_at: impl Fn(usize) -> usize,
        need_hessian: bool,
        mut visit: impl FnMut(BmsFlexCalibrationOrder2Node) -> Result<(), E>,
    ) -> Result<(), E> {
        Self::try_for_each_calibration_order2_phase(need_hessian, |phase| {
            match phase {
                BmsFlexCalibrationOrder2Phase::InterceptFirst => {
                    visit(BmsFlexCalibrationOrder2Node::InterceptFirst)?;
                }
                BmsFlexCalibrationOrder2Phase::InterceptSecond => {
                    visit(BmsFlexCalibrationOrder2Node::InterceptSecond)?;
                }
                BmsFlexCalibrationOrder2Phase::PrimaryFirstAndInterceptSecond => {
                    for position in 0..active_count {
                        let primary = active_at(position);
                        visit(BmsFlexCalibrationOrder2Node::PrimaryFirst { primary })?;
                        if need_hessian {
                            visit(BmsFlexCalibrationOrder2Node::InterceptPrimarySecond {
                                primary,
                            })?;
                        }
                    }
                }
                BmsFlexCalibrationOrder2Phase::PrimaryPairSecond => {
                    for left_position in 0..active_count {
                        let left = active_at(left_position);
                        for right_position in left_position..active_count {
                            visit(BmsFlexCalibrationOrder2Node::PrimaryPairSecond {
                                left,
                                right: active_at(right_position),
                            })?;
                        }
                    }
                }
            }
            Ok(())
        })
    }

    /// Visit the canonical Order2 calibration phases without choosing a
    /// backend index representation.
    pub(super) fn try_for_each_calibration_order2_phase<E>(
        need_hessian: bool,
        mut visit: impl FnMut(BmsFlexCalibrationOrder2Phase) -> Result<(), E>,
    ) -> Result<(), E> {
        visit(BmsFlexCalibrationOrder2Phase::InterceptFirst)?;
        if need_hessian {
            visit(BmsFlexCalibrationOrder2Phase::InterceptSecond)?;
        }
        visit(BmsFlexCalibrationOrder2Phase::PrimaryFirstAndInterceptSecond)?;
        if need_hessian {
            visit(BmsFlexCalibrationOrder2Phase::PrimaryPairSecond)?;
        }
        Ok(())
    }

    /// Interpret the canonical directional derivative of the Order2 schedule
    /// for a contiguous range of active primary coordinates.
    pub(super) fn try_for_each_calibration_order3_contiguous<E>(
        active_primaries: std::ops::Range<usize>,
        direction_count: usize,
        visit: impl FnMut(BmsFlexCalibrationOrder3Node) -> Result<(), E>,
    ) -> Result<(), E> {
        let start = active_primaries.start;
        Self::try_for_each_calibration_order3_indexed(
            active_primaries.len(),
            |position| start + position,
            direction_count,
            visit,
        )
    }

    fn try_for_each_calibration_order3_indexed<E>(
        active_count: usize,
        active_at: impl Fn(usize) -> usize,
        direction_count: usize,
        mut visit: impl FnMut(BmsFlexCalibrationOrder3Node) -> Result<(), E>,
    ) -> Result<(), E> {
        for direction in 0..direction_count {
            visit(BmsFlexCalibrationOrder3Node::DirectionStart { direction })?;
            visit(BmsFlexCalibrationOrder3Node::InterceptDirectionalSecond { direction })?;
            visit(BmsFlexCalibrationOrder3Node::InterceptDirectionalThird { direction })?;
            for position in 0..active_count {
                let primary = active_at(position);
                visit(
                    BmsFlexCalibrationOrder3Node::InterceptPrimaryDirectionalThird {
                        direction,
                        primary,
                    },
                )?;
            }
            for left_position in 0..active_count {
                let left = active_at(left_position);
                for right_position in left_position..active_count {
                    let right = active_at(right_position);
                    visit(BmsFlexCalibrationOrder3Node::PrimaryPairDirectionalThird {
                        direction,
                        left,
                        right,
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Interpret the canonical mixed directional derivative of the Order2
    /// schedule for a contiguous primary range and
    /// `direction_pair_count` backend-owned direction pairs.
    pub(super) fn try_for_each_calibration_order4_contiguous<E>(
        active_primaries: std::ops::Range<usize>,
        direction_pair_count: usize,
        visit: impl FnMut(BmsFlexCalibrationOrder4Node) -> Result<(), E>,
    ) -> Result<(), E> {
        let start = active_primaries.start;
        Self::try_for_each_calibration_order4_indexed(
            active_primaries.len(),
            |position| start + position,
            direction_pair_count,
            visit,
        )
    }

    fn try_for_each_calibration_order4_indexed<E>(
        active_count: usize,
        active_at: impl Fn(usize) -> usize,
        direction_pair_count: usize,
        mut visit: impl FnMut(BmsFlexCalibrationOrder4Node) -> Result<(), E>,
    ) -> Result<(), E> {
        for pair in 0..direction_pair_count {
            visit(BmsFlexCalibrationOrder4Node::DirectionPairStart { pair })?;
            visit(BmsFlexCalibrationOrder4Node::InterceptMixedThird { pair })?;
            visit(BmsFlexCalibrationOrder4Node::InterceptMixedFourth { pair })?;
            for position in 0..active_count {
                let primary = active_at(position);
                visit(BmsFlexCalibrationOrder4Node::InterceptPrimaryMixedFourth { pair, primary })?;
            }
            for left_position in 0..active_count {
                let left = active_at(left_position);
                for right_position in left_position..active_count {
                    let right = active_at(right_position);
                    visit(BmsFlexCalibrationOrder4Node::PrimaryPairMixedFourth {
                        pair,
                        left,
                        right,
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Interpret the dependency-ordered implicit/observed Order2 finalizer.
    pub(super) fn try_for_each_order2_finalizer<E>(
        primary_count: usize,
        need_hessian: bool,
        mut visit: impl FnMut(BmsFlexRowOrder2FinalizerNode) -> Result<(), E>,
    ) -> Result<(), E> {
        Self::try_for_each_order2_finalizer_phase(need_hessian, |phase| {
            match phase {
                BmsFlexRowOrder2FinalizerPhase::ImplicitFirst => {
                    for primary in 0..primary_count {
                        visit(BmsFlexRowOrder2FinalizerNode::ImplicitFirst { primary })?;
                    }
                }
                BmsFlexRowOrder2FinalizerPhase::ImplicitFirstComplete => {
                    visit(BmsFlexRowOrder2FinalizerNode::ImplicitFirstComplete)?;
                }
                BmsFlexRowOrder2FinalizerPhase::ImplicitSecond => {
                    for left in 0..primary_count {
                        for right in left..primary_count {
                            visit(BmsFlexRowOrder2FinalizerNode::ImplicitSecond { left, right })?;
                        }
                    }
                }
                BmsFlexRowOrder2FinalizerPhase::ObservedFirst => {
                    for primary in 0..primary_count {
                        visit(BmsFlexRowOrder2FinalizerNode::ObservedFirst { primary })?;
                    }
                }
                BmsFlexRowOrder2FinalizerPhase::ObservedScoreSensitivity => {
                    for primary in 0..primary_count {
                        visit(BmsFlexRowOrder2FinalizerNode::ObservedScoreSensitivity { primary })?;
                    }
                }
                BmsFlexRowOrder2FinalizerPhase::ObservedSecond => {
                    for left in 0..primary_count {
                        for right in left..primary_count {
                            visit(BmsFlexRowOrder2FinalizerNode::ObservedSecond { left, right })?;
                        }
                    }
                }
                BmsFlexRowOrder2FinalizerPhase::NegLogFirst => {
                    for primary in 0..primary_count {
                        visit(BmsFlexRowOrder2FinalizerNode::NegLogFirst { primary })?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Visit the canonical dependency phases without selecting an index
    /// representation for a backend.
    pub(super) fn try_for_each_order2_finalizer_phase<E>(
        need_hessian: bool,
        mut visit: impl FnMut(BmsFlexRowOrder2FinalizerPhase) -> Result<(), E>,
    ) -> Result<(), E> {
        visit(BmsFlexRowOrder2FinalizerPhase::ImplicitFirst)?;
        visit(BmsFlexRowOrder2FinalizerPhase::ImplicitFirstComplete)?;
        if need_hessian {
            visit(BmsFlexRowOrder2FinalizerPhase::ImplicitSecond)?;
        }
        visit(BmsFlexRowOrder2FinalizerPhase::ObservedFirst)?;
        visit(BmsFlexRowOrder2FinalizerPhase::ObservedScoreSensitivity)?;
        if need_hessian {
            visit(BmsFlexRowOrder2FinalizerPhase::ObservedSecond)?;
        }
        visit(BmsFlexRowOrder2FinalizerPhase::NegLogFirst)?;
        Ok(())
    }

    pub(super) fn from_parts(
        point: BmsFlexProgramPoint<'_>,
        calibration: Vec<BmsFlexCalibrationProgramNode>,
        observed: BmsFlexIndexProgram,
        observed_sign: f64,
        observed_neglog_stack: [f64; 5],
    ) -> Result<Self, String> {
        Ok(Self {
            primary: point.primary().clone(),
            intercept_root: point.intercept_root(),
            inv_f_a: point.inv_f_a(),
            scale: point.scale(),
            mu_stack: point.mu_stack(),
            calibration,
            observed,
            observed_sign,
            observed_neglog_stack,
        })
    }

    #[inline]
    fn evaluate_index<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        a: &S,
        vars: &[S],
        index: &BmsFlexIndexProgram,
        workspace: &'arena S::Workspace,
    ) -> S {
        let dimension = self.primary.total;
        let b = &vars[self.primary.logslope];
        let u = a.add(&b.scale(index.z));
        let mut inside = u.clone();

        if let Some(range) = self.primary.h.as_ref() {
            let mut score = S::constant(0.0, dimension, workspace);
            for (local, idx) in range.clone().enumerate() {
                score = score.add(&vars[idx].scale(index.score_values[local]));
            }
            inside = inside.add(&b.mul(&score));
        }

        if let Some(range) = self.primary.w.as_ref() {
            let mut warp = S::constant(0.0, dimension, workspace);
            for (local, idx) in range.clone().enumerate() {
                let basis = u.compose_unary(index.link_stacks[local]);
                warp = warp.add(&vars[idx].mul(&basis));
            }
            inside = inside.add(&warp);
        }
        inside.scale(self.scale)
    }

    /// Interpret the canonical program in a runtime-sized derivative algebra.
    /// `lift_iters` is two for Order2, three for OneSeed, and four for TwoSeed.
    pub(super) fn evaluate<'arena, S: RuntimeJetScalar<'arena>>(
        &self,
        vars: &[S],
        lift_iters: usize,
        workspace: &'arena S::Workspace,
    ) -> Result<S, String> {
        let dimension = self.primary.total;
        if vars.len() != dimension {
            return Err(format!(
                "BMS FLEX row program received {} primaries, expected {dimension}",
                vars.len()
            ));
        }
        if vars.iter().any(|var| var.dimension() != dimension) {
            return Err("BMS FLEX row program received a mismatched jet dimension".to_string());
        }

        let neg_mu = vars[self.primary.q].compose_unary(self.mu_stack).neg();
        let constraint = |a: &S| -> S {
            let mut residual = neg_mu.clone();
            for node in &self.calibration {
                let eta = self.evaluate_index(a, vars, &node.index, workspace);
                residual = residual.add(&eta.compose_unary(node.cdf_stack).scale(node.weight));
            }
            residual
        };
        let intercept = filtered_implicit_solve_runtime_scalar(
            self.intercept_root,
            self.inv_f_a,
            lift_iters,
            dimension,
            workspace,
            constraint,
        );
        let signed = self
            .evaluate_index(&intercept, vars, &self.observed, workspace)
            .scale(self.observed_sign);
        Ok(signed.compose_unary(self.observed_neglog_stack))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_node_stream_owns_sparse_primary_and_direction_order() {
        let active = [1, 4];
        let mut order2 = Vec::new();
        BmsFlexRowProgram::for_each_calibration_order2(&active, true, |node| order2.push(node));
        assert_eq!(
            order2,
            vec![
                BmsFlexCalibrationOrder2Node::InterceptFirst,
                BmsFlexCalibrationOrder2Node::InterceptSecond,
                BmsFlexCalibrationOrder2Node::PrimaryFirst { primary: 1 },
                BmsFlexCalibrationOrder2Node::InterceptPrimarySecond { primary: 1 },
                BmsFlexCalibrationOrder2Node::PrimaryFirst { primary: 4 },
                BmsFlexCalibrationOrder2Node::InterceptPrimarySecond { primary: 4 },
                BmsFlexCalibrationOrder2Node::PrimaryPairSecond { left: 1, right: 1 },
                BmsFlexCalibrationOrder2Node::PrimaryPairSecond { left: 1, right: 4 },
                BmsFlexCalibrationOrder2Node::PrimaryPairSecond { left: 4, right: 4 },
            ]
        );

        let mut order3 = Vec::new();
        BmsFlexRowProgram::try_for_each_calibration_order3_indexed(
            active.len(),
            |position| active[position],
            2,
            |node| -> Result<(), std::convert::Infallible> {
                order3.push(node);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(order3.len(), 16);
        assert_eq!(
            &order3[..8],
            &[
                BmsFlexCalibrationOrder3Node::DirectionStart { direction: 0 },
                BmsFlexCalibrationOrder3Node::InterceptDirectionalSecond { direction: 0 },
                BmsFlexCalibrationOrder3Node::InterceptDirectionalThird { direction: 0 },
                BmsFlexCalibrationOrder3Node::InterceptPrimaryDirectionalThird {
                    direction: 0,
                    primary: 1,
                },
                BmsFlexCalibrationOrder3Node::InterceptPrimaryDirectionalThird {
                    direction: 0,
                    primary: 4,
                },
                BmsFlexCalibrationOrder3Node::PrimaryPairDirectionalThird {
                    direction: 0,
                    left: 1,
                    right: 1,
                },
                BmsFlexCalibrationOrder3Node::PrimaryPairDirectionalThird {
                    direction: 0,
                    left: 1,
                    right: 4,
                },
                BmsFlexCalibrationOrder3Node::PrimaryPairDirectionalThird {
                    direction: 0,
                    left: 4,
                    right: 4,
                },
            ]
        );
        assert!(order3[8..].iter().all(|node| match node {
            BmsFlexCalibrationOrder3Node::DirectionStart { direction }
            | BmsFlexCalibrationOrder3Node::InterceptDirectionalSecond { direction }
            | BmsFlexCalibrationOrder3Node::InterceptDirectionalThird { direction }
            | BmsFlexCalibrationOrder3Node::InterceptPrimaryDirectionalThird {
                direction, ..
            }
            | BmsFlexCalibrationOrder3Node::PrimaryPairDirectionalThird { direction, .. } => {
                *direction == 1
            }
        }));

        let mut order4 = Vec::new();
        BmsFlexRowProgram::try_for_each_calibration_order4_indexed(
            active.len(),
            |position| active[position],
            1,
            |node| -> Result<(), std::convert::Infallible> {
                order4.push(node);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            order4,
            vec![
                BmsFlexCalibrationOrder4Node::DirectionPairStart { pair: 0 },
                BmsFlexCalibrationOrder4Node::InterceptMixedThird { pair: 0 },
                BmsFlexCalibrationOrder4Node::InterceptMixedFourth { pair: 0 },
                BmsFlexCalibrationOrder4Node::InterceptPrimaryMixedFourth {
                    pair: 0,
                    primary: 1,
                },
                BmsFlexCalibrationOrder4Node::InterceptPrimaryMixedFourth {
                    pair: 0,
                    primary: 4,
                },
                BmsFlexCalibrationOrder4Node::PrimaryPairMixedFourth {
                    pair: 0,
                    left: 1,
                    right: 1,
                },
                BmsFlexCalibrationOrder4Node::PrimaryPairMixedFourth {
                    pair: 0,
                    left: 1,
                    right: 4,
                },
                BmsFlexCalibrationOrder4Node::PrimaryPairMixedFourth {
                    pair: 0,
                    left: 4,
                    right: 4,
                },
            ]
        );

        let mut finalizer = Vec::new();
        BmsFlexRowProgram::try_for_each_order2_finalizer(
            2,
            true,
            |node| -> Result<(), std::convert::Infallible> {
                finalizer.push(node);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            finalizer,
            vec![
                BmsFlexRowOrder2FinalizerNode::ImplicitFirst { primary: 0 },
                BmsFlexRowOrder2FinalizerNode::ImplicitFirst { primary: 1 },
                BmsFlexRowOrder2FinalizerNode::ImplicitFirstComplete,
                BmsFlexRowOrder2FinalizerNode::ImplicitSecond { left: 0, right: 0 },
                BmsFlexRowOrder2FinalizerNode::ImplicitSecond { left: 0, right: 1 },
                BmsFlexRowOrder2FinalizerNode::ImplicitSecond { left: 1, right: 1 },
                BmsFlexRowOrder2FinalizerNode::ObservedFirst { primary: 0 },
                BmsFlexRowOrder2FinalizerNode::ObservedFirst { primary: 1 },
                BmsFlexRowOrder2FinalizerNode::ObservedScoreSensitivity { primary: 0 },
                BmsFlexRowOrder2FinalizerNode::ObservedScoreSensitivity { primary: 1 },
                BmsFlexRowOrder2FinalizerNode::ObservedSecond { left: 0, right: 0 },
                BmsFlexRowOrder2FinalizerNode::ObservedSecond { left: 0, right: 1 },
                BmsFlexRowOrder2FinalizerNode::ObservedSecond { left: 1, right: 1 },
                BmsFlexRowOrder2FinalizerNode::NegLogFirst { primary: 0 },
                BmsFlexRowOrder2FinalizerNode::NegLogFirst { primary: 1 },
            ]
        );
    }
}
