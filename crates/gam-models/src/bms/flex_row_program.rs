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
    InterceptSecond,
    PrimaryFirst { primary: usize },
    InterceptPrimarySecond { primary: usize },
    PrimaryPairSecond { left: usize, right: usize },
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
        mut visit: impl FnMut(BmsFlexCalibrationOrder2Node) -> Result<(), E>,
    ) -> Result<(), E> {
        if need_hessian {
            visit(BmsFlexCalibrationOrder2Node::InterceptSecond)?;
        }
        for &primary in active_primaries {
            visit(BmsFlexCalibrationOrder2Node::PrimaryFirst { primary })?;
            if need_hessian {
                visit(BmsFlexCalibrationOrder2Node::InterceptPrimarySecond { primary })?;
            }
        }
        if need_hessian {
            for (position, &left) in active_primaries.iter().enumerate() {
                for &right in &active_primaries[position..] {
                    visit(BmsFlexCalibrationOrder2Node::PrimaryPairSecond { left, right })?;
                }
            }
        }
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
