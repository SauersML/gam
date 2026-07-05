//! Contract composition and loop holonomy — the "Atlas Machine" composition
//! layer (issue: end-to-end bounds over per-edge diagnostics).
//!
//! The crate already emits per-edge diagnostics for the atlas of charts and the
//! transports between them: [`layer_transport`](crate::inference::layer_transport)
//! reports isometry defects and composition-law defects, [`steering`](crate::inference::steering)
//! reports validity radii and amortization gaps, and
//! [`transport_class`](crate::inference::transport_class) classifies each circle
//! transport as an `O(2)` element `(winding, phase)`. What was missing is the
//! step that COMPOSES those local certificates into a statement about a whole
//! chain or a whole loop. This module is that step.
//!
//! Two instruments live here, both pure `f64` and both obeying the house
//! **measure-don't-latch** doctrine used across the crate (validity radii are
//! reported, never silently clipped — see [`steering`](crate::inference::steering);
//! the composition-defect floor is reported, never folded in — see
//! [`layer_transport`](crate::inference::layer_transport)):
//!
//! 1. **Contract composition** ([`compose_contracts`], [`compose_with_trace`]):
//!    the shadowing bound that turns a chain of per-component error/expansion
//!    certificates into a single end-to-end error bound, plus a domain-feasibility
//!    trace that reports where the accumulated error would leave a component's
//!    certified domain instead of quietly shrinking the bound to fit.
//! 2. **Loop holonomy** ([`loop_holonomy`], [`holonomy_from_transports`]): the
//!    net `O(2)` element obtained by composing the transports around a closed
//!    loop of charts, with a trivial/nontrivial verdict whose tolerance is
//!    DERIVED from the loop's own defects (never a magic constant).

use crate::inference::transport_class::CircleTransportReport;
use std::f64::consts::{PI, TAU};

/// One component map of a chart-coordinate pipeline, abstracted to exactly the
/// three numbers a composition bound needs.
///
/// The map is a certificate, not the map itself: it says that on a coordinate
/// ball of radius [`domain_radius`](Self::domain_radius) about the nominal
/// trajectory, the realized map stays within [`defect`](Self::defect) of the
/// true map it approximates (both measured in the same chart metric), and it
/// expands that metric by at most a factor [`lipschitz`](Self::lipschitz).
///
/// This is the common shape of every per-edge diagnostic the crate already
/// emits — an amortized decoder (defect = amortization gap, lipschitz from the
/// pullback), a fitted transport (defect = isometry/composition defect,
/// lipschitz from `|h′|`), a steering push (defect = endpoint-vs-path dose gap,
/// domain_radius = validity radius).
#[derive(Debug, Clone)]
pub struct Contract {
    /// Legible name of the component (for trace reporting).
    pub name: String,
    /// Radius of the coordinate ball on which this certificate is valid.
    pub domain_radius: f64,
    /// Sup-norm gap between the realized map and the true map on the domain
    /// ball, in the chart metric.
    pub defect: f64,
    /// Metric expansion (Lipschitz) bound of the map on the domain ball.
    pub lipschitz: f64,
}

impl Contract {
    /// Validate the three certificate numbers: all finite, none negative.
    fn validate(&self, stage: usize) -> Result<(), String> {
        for (label, v) in [
            ("domain_radius", self.domain_radius),
            ("defect", self.defect),
            ("lipschitz", self.lipschitz),
        ] {
            if !v.is_finite() {
                return Err(format!(
                    "contract stage {stage} ({}): {label} = {v} is not finite",
                    self.name
                ));
            }
            if v < 0.0 {
                return Err(format!(
                    "contract stage {stage} ({}): {label} = {v} is negative",
                    self.name
                ));
            }
        }
        Ok(())
    }
}

/// The end-to-end certificate produced by composing a [`Contract`] chain.
#[derive(Debug, Clone)]
pub struct ComposedContract {
    /// Shadowing bound on the sup-norm gap between the composed realized map
    /// and the composed true map over the whole chain.
    pub total_defect: f64,
    /// `per_stage_contribution[j]` is stage `j`'s additive share of
    /// [`total_defect`](Self::total_defect): `defect_j · Π_{i>j} lipschitz_i`.
    /// Sums to `total_defect`. Earlier stages carry more when the later
    /// expansion factors exceed 1 (their error is amplified by more stages).
    pub per_stage_contribution: Vec<f64>,
    /// Whether the accumulated error stays inside every stage's certified
    /// domain along the chain. For [`compose_contracts`] this is the drift-only
    /// check (accumulated error alone must fit each `domain_radius`); for
    /// [`compose_with_trace`] the feasible path always returns `true` and an
    /// infeasible one returns `Err` naming the offending stages.
    pub domain_ok: bool,
}

/// Compose a chain of component contracts into one end-to-end shadowing bound.
///
/// The chain is applied in order: `F = f_{n} ∘ … ∘ f_1`, with `f_j` the realized
/// map of `chain[j-1]` approximating the true map `g_j` (`|f_j − g_j| ≤
/// defect_j` on its domain) and `g_j` metric-Lipschitz with constant
/// `lipschitz_j`.
///
/// # Shadowing bound (two-line induction)
///
/// Let `F_k = f_k ∘ … ∘ f_1`, `G_k = g_k ∘ … ∘ g_1`, and `E_k = sup_x |F_k(x) −
/// G_k(x)|` (`E_0 = 0`). Splitting the `k`-th step through the intermediate
/// point `F_{k-1}(x)`:
///
/// ```text
/// |F_k(x) − G_k(x)|
///   = |f_k(F_{k-1}(x)) − g_k(G_{k-1}(x))|
///   ≤ |f_k(F_{k-1}(x)) − g_k(F_{k-1}(x))|   (approximation error of f_k)
///   + |g_k(F_{k-1}(x)) − g_k(G_{k-1}(x))|   (g_k is lipschitz_k-Lipschitz)
///   ≤ defect_k + lipschitz_k · E_{k-1}.
/// ```
///
/// Taking the sup gives the recurrence `E_k ≤ defect_k + lipschitz_k · E_{k-1}`,
/// and unrolling it from `E_0 = 0` gives the closed form
///
/// ```text
/// E_n ≤ Σ_{j=1}^{n} defect_j · Π_{i=j+1}^{n} lipschitz_i,
/// ```
///
/// which is [`total_defect`](ComposedContract::total_defect); the `j`-th summand
/// is [`per_stage_contribution`](ComposedContract::per_stage_contribution)`[j-1]`.
///
/// [`domain_ok`](ComposedContract::domain_ok) is the drift-only feasibility
/// check available without entry radii: the error accumulated *before* stage `j`
/// (namely `E_{j-1}`) must not already exceed stage `j`'s `domain_radius`, else
/// the realized trajectory has drifted outside where `f_j`'s certificate holds.
/// Use [`compose_with_trace`] to fold in the nominal input spread as well.
///
/// An empty chain is the identity: zero defect, no contributions, feasible.
pub fn compose_contracts(chain: &[Contract]) -> ComposedContract {
    let n = chain.len();
    // Suffix products Π_{i>j} lipschitz_i, computed right-to-left.
    let mut suffix = vec![1.0_f64; n + 1];
    for j in (0..n).rev() {
        suffix[j] = suffix[j + 1] * chain[j].lipschitz;
    }
    let mut per_stage_contribution = vec![0.0_f64; n];
    let mut total_defect = 0.0_f64;
    for j in 0..n {
        let c = chain[j].defect * suffix[j + 1];
        per_stage_contribution[j] = c;
        total_defect += c;
    }

    // Drift-only domain check: accumulated error entering stage j is E_{j-1};
    // require it to fit stage j's domain_radius. Propagate E the same way the
    // bound does.
    let mut domain_ok = true;
    let mut accumulated = 0.0_f64;
    for stage in chain.iter() {
        if accumulated > stage.domain_radius {
            domain_ok = false;
        }
        accumulated = stage.defect + stage.lipschitz * accumulated;
    }

    ComposedContract {
        total_defect,
        per_stage_contribution,
        domain_ok,
    }
}

/// Compose a contract chain and check domain feasibility against the nominal
/// input spread at each stage entry.
///
/// `entry_radii[j]` is the radius of the coordinate ball the pipeline actually
/// feeds to stage `j` (the nominal spread of inputs, before approximation
/// error). The realized inputs to stage `j` sit within `entry_radii[j] + E_{j-1}`
/// of the certified center, where `E_{j-1}` is the error accumulated before
/// stage `j` (propagated exactly as in [`compose_contracts`]). Feasibility
/// requires that displacement to fit the stage's certified domain:
///
/// ```text
/// entry_radii[j] + E_{j-1} ≤ domain_radius_j    for every stage j.
/// ```
///
/// Per the measure-don't-latch doctrine, a violation is **reported, never folded
/// into the bound**: [`total_defect`](ComposedContract::total_defect) is always
/// the honest shadowing bound of the chain (it is never shrunk to make a stage
/// fit), and any stage whose certified domain is overrun is surfaced as an
/// `Err` naming the stage and the exact overflow. A feasible chain returns
/// `Ok` with [`domain_ok`](ComposedContract::domain_ok)` = true`.
///
/// `Err` is also returned for structural problems (length mismatch, non-finite
/// or negative certificate numbers or entry radii).
pub fn compose_with_trace(
    chain: &[Contract],
    entry_radii: &[f64],
) -> Result<ComposedContract, String> {
    if chain.len() != entry_radii.len() {
        return Err(format!(
            "compose_with_trace: {} contracts but {} entry radii",
            chain.len(),
            entry_radii.len()
        ));
    }
    for (j, c) in chain.iter().enumerate() {
        c.validate(j)?;
    }
    for (j, &r) in entry_radii.iter().enumerate() {
        if !r.is_finite() || r < 0.0 {
            return Err(format!("compose_with_trace: entry_radii[{j}] = {r} invalid"));
        }
    }

    let composed = compose_contracts(chain);

    // Feasibility trace: displacement entering stage j is entry_radii[j] + E_{j-1}.
    // Collect every violation rather than bailing on the first — a full report
    // is more actionable than the first offender.
    let mut violations: Vec<String> = Vec::new();
    let mut accumulated = 0.0_f64;
    for (j, stage) in chain.iter().enumerate() {
        let required = entry_radii[j] + accumulated;
        if required > stage.domain_radius {
            let overflow = required - stage.domain_radius;
            violations.push(format!(
                "stage {j} ({}): entry radius {} + accumulated error {} = {} exceeds \
                 domain_radius {} by {}",
                stage.name, entry_radii[j], accumulated, required, stage.domain_radius, overflow
            ));
        }
        accumulated = stage.defect + stage.lipschitz * accumulated;
    }

    if !violations.is_empty() {
        return Err(format!(
            "compose_with_trace: {} domain violation(s) (bound total_defect = {} is unchanged): {}",
            violations.len(),
            composed.total_defect,
            violations.join("; ")
        ));
    }

    Ok(ComposedContract {
        domain_ok: true,
        ..composed
    })
}

/// The net `O(2)` element obtained by composing the transports around a closed
/// loop of charts, with a derived trivial/nontrivial verdict.
#[derive(Debug, Clone)]
pub struct HolonomyReport {
    /// Number of edges (transports) composed around the loop.
    pub loop_len: usize,
    /// Sign of the net element: `+1` net rotation, `−1` net reflection
    /// (product of the per-edge signs).
    pub net_sign: i8,
    /// Net rotation angle in `(−π, π]` (for a net reflection, the phase of the
    /// reflected element).
    pub net_angle: f64,
    /// Whether the net element is within tolerance of the identity: a rotation
    /// (`net_sign = +1`) whose angle is below [`angle_tolerance`](Self::angle_tolerance).
    /// A net reflection is never trivial (the identity is a rotation).
    pub is_trivial: bool,
    /// Tolerance the trivial verdict uses, DERIVED as the loop's composed defect
    /// bound (see [`loop_holonomy`]).
    pub angle_tolerance: f64,
}

/// Wrap an angle into `(−π, π]`.
fn wrap_pi(x: f64) -> f64 {
    let w = (x + PI).rem_euclid(TAU) - PI;
    if w <= -PI { w + TAU } else { w }
}

/// Compose a closed loop of circle isometries and report the net `O(2)` element.
///
/// Each edge is an `O(2)` element `(sign, angle)` in the vocabulary of
/// [`transport_class`](crate::inference::transport_class): `sign = +1` is the
/// rotation `x ↦ x + angle`, `sign = −1` is the reflection `x ↦ −x + angle`
/// (matching [`CircleTransportReport::winding`] and `phase`). The elements are
/// composed exactly in `O(2)`: for `g = (s_g, φ_g)` applied after `f = (s_f,
/// φ_f)`,
///
/// ```text
/// (g ∘ f)(x) = s_g·(s_f·x + φ_f) + φ_g = (s_g·s_f)·x + (s_g·φ_f + φ_g),
/// ```
///
/// so signs multiply and a reflection flips the orientation of every angle it
/// composes over. Folding this over the loop yields the net element; rotation
/// angles add (mod 2π) and the net sign is the product of the edge signs — both
/// tracked exactly, not sampled.
///
/// # Trivial verdict and its derived tolerance
///
/// Nontrivial holonomy is the obstruction to the loop being a single global
/// feature: if transporting a concept all the way around the loop does not
/// return it to itself, the charts cannot be glued into one coordinate for that
/// feature across the loop. The identity is a rotation by `0`, so the net
/// element is trivial only when `net_sign = +1` and `|net_angle|` is within the
/// tolerance.
///
/// The tolerance is not a magic constant: it is the loop's own composed defect
/// bound. Each transport is an isometry, so its Lipschitz constant is exactly
/// `1`; feeding `lipschitz_i ≡ 1` into the shadowing bound of
/// [`compose_contracts`] collapses `Σ_j defect_j · Π_{i>j} lipschitz_i` to
/// `Σ_j defect_j`. So the angle by which the composed transport is uncertain is
/// bounded by the sum of the per-edge defects, and any net angle smaller than
/// that cannot be distinguished from the identity — the honest measure-don't-latch
/// verdict (a cleaner loop, with smaller defects, gets a tighter tolerance and
/// so is easier to certify nontrivial).
pub fn loop_holonomy(edges: &[(i8, f64)], defects: &[f64]) -> HolonomyReport {
    // Fold the O(2) elements in loop order: acc ← edge ∘ acc.
    let mut acc_sign = 1i8;
    let mut acc_angle = 0.0_f64;
    for &(sign, angle) in edges.iter() {
        let s = if sign >= 0 { 1i8 } else { -1i8 };
        acc_angle = (s as f64) * acc_angle + angle;
        acc_sign *= s;
    }
    let net_angle = wrap_pi(acc_angle);

    // Derived tolerance: composed defect bound of the loop = Σ defects, since
    // isometries have Lipschitz 1 (see the doc comment). Defensive against a
    // caller passing a mismatched-length or non-finite defect slice.
    let angle_tolerance = defects
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .sum::<f64>();

    let is_trivial = acc_sign == 1 && net_angle.abs() <= angle_tolerance;

    HolonomyReport {
        loop_len: edges.len(),
        net_sign: acc_sign,
        net_angle,
        is_trivial,
        angle_tolerance,
    }
}

/// Adapter: compute loop holonomy directly from a loop of fitted circle
/// transport classifications.
///
/// [`CircleTransportReport`] already carries the `O(2)` element as
/// [`winding`](CircleTransportReport::winding) (the sign) and
/// [`phase`](CircleTransportReport::phase) (the angle), plus its
/// [`defect`](CircleTransportReport::defect) (the `O(2)` departure that bounds
/// the angle uncertainty). This lifts them straight into [`loop_holonomy`]
/// without the caller restating the `(sign, angle)` interface. The reports must
/// be given in loop order.
pub fn holonomy_from_transports(loop_edges: &[CircleTransportReport]) -> HolonomyReport {
    let edges: Vec<(i8, f64)> = loop_edges.iter().map(|r| (r.winding, r.phase)).collect();
    let defects: Vec<f64> = loop_edges.iter().map(|r| r.defect).collect();
    loop_holonomy(&edges, &defects)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(name: &str, domain_radius: f64, defect: f64, lipschitz: f64) -> Contract {
        Contract {
            name: name.to_string(),
            domain_radius,
            defect,
            lipschitz,
        }
    }

    #[test]
    fn composition_matches_closed_form_sum() {
        // Hand-built 3-stage chain with distinct numbers.
        let chain = [
            c("a", 10.0, 0.1, 2.0),
            c("b", 10.0, 0.2, 3.0),
            c("c", 10.0, 0.4, 5.0),
        ];
        let out = compose_contracts(&chain);
        // total = 0.1·(3·5) + 0.2·(5) + 0.4·(1) = 1.5 + 1.0 + 0.4 = 2.9.
        assert!((out.total_defect - 2.9).abs() < 1e-12);
        assert!((out.per_stage_contribution[0] - 1.5).abs() < 1e-12);
        assert!((out.per_stage_contribution[1] - 1.0).abs() < 1e-12);
        assert!((out.per_stage_contribution[2] - 0.4).abs() < 1e-12);
        // Contributions sum to the total.
        let s: f64 = out.per_stage_contribution.iter().sum();
        assert!((s - out.total_defect).abs() < 1e-12);
        assert!(out.domain_ok);
    }

    #[test]
    fn lipschitz_gt_one_amplifies_early_defects_more() {
        // Equal defects, equal expansion > 1: earlier stages carry more.
        let chain = [
            c("a", 100.0, 0.3, 2.0),
            c("b", 100.0, 0.3, 2.0),
            c("c", 100.0, 0.3, 2.0),
            c("d", 100.0, 0.3, 2.0),
        ];
        let out = compose_contracts(&chain);
        for w in out.per_stage_contribution.windows(2) {
            assert!(w[0] > w[1], "contribution not strictly decreasing: {w:?}");
        }
        // Closed form: 0.3·(8 + 4 + 2 + 1) = 0.3·15 = 4.5.
        assert!((out.total_defect - 4.5).abs() < 1e-12);
    }

    #[test]
    fn empty_chain_is_identity() {
        let out = compose_contracts(&[]);
        assert_eq!(out.total_defect, 0.0);
        assert!(out.per_stage_contribution.is_empty());
        assert!(out.domain_ok);
    }

    #[test]
    fn trace_reports_domain_violation_at_right_stage() {
        // Stage 0 amplifies heavily so the error entering stage 2 blows its
        // tight domain; stages 0 and 1 have generous domains.
        let chain = [
            c("a", 100.0, 1.0, 10.0),
            c("b", 100.0, 1.0, 10.0),
            c("c", 5.0, 1.0, 1.0),
        ];
        // entry spreads are small; the accumulated error is what overruns.
        // E_0 = 0, E_1 = 1 + 10·0 = 1, E_2 = 1 + 10·1 = 11.
        // Stage 2 entry displacement = 0 + 11 = 11 > domain_radius 5.
        let entry = [0.0, 0.0, 0.0];
        let res = compose_with_trace(&chain, &entry);
        let err = res.expect_err("expected a domain violation");
        assert!(err.contains("stage 2"), "wrong stage reported: {err}");
        assert!(err.contains("by 6"), "wrong overflow reported: {err}");
        // The bound itself is still reported, unmodified.
        assert!(err.contains("total_defect"));
    }

    #[test]
    fn trace_feasible_chain_is_ok() {
        let chain = [
            c("a", 100.0, 0.1, 1.0),
            c("b", 100.0, 0.1, 1.0),
            c("c", 100.0, 0.1, 1.0),
        ];
        let entry = [1.0, 1.0, 1.0];
        let out = compose_with_trace(&chain, &entry).expect("feasible");
        assert!(out.domain_ok);
        assert!((out.total_defect - 0.3).abs() < 1e-12);
    }

    #[test]
    fn trace_length_mismatch_errors() {
        let chain = [c("a", 1.0, 0.1, 1.0)];
        assert!(compose_with_trace(&chain, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn rotations_summing_to_zero_are_trivial() {
        // Pure rotations that sum to 2π ≡ 0, tiny defects.
        let edges = [(1i8, 2.0), (1, 2.0), (1, TAU - 4.0)];
        let defects = [1e-6, 1e-6, 1e-6];
        let r = loop_holonomy(&edges, &defects);
        assert_eq!(r.net_sign, 1);
        assert!(r.net_angle.abs() < 1e-9, "net_angle = {}", r.net_angle);
        assert!(r.is_trivial);
    }

    #[test]
    fn small_net_rotation_with_tiny_defects_is_nontrivial() {
        // Net rotation π/7, defects far below it → cannot be excluded as noise.
        let edges = [(1i8, PI / 7.0)];
        let defects = [1e-4];
        let r = loop_holonomy(&edges, &defects);
        assert_eq!(r.net_sign, 1);
        assert!((r.net_angle - PI / 7.0).abs() < 1e-12);
        assert!(!r.is_trivial);
    }

    #[test]
    fn two_reflections_compose_to_a_rotation() {
        let edges = [(-1i8, 0.3), (-1, 0.9)];
        let defects = [1e-6, 1e-6];
        let r = loop_holonomy(&edges, &defects);
        // (-1)·(-1) = +1: net rotation.
        assert_eq!(r.net_sign, 1);
        // acc: start (1,0); edge0 → (-1, 0.3); edge1 = (-1,0.9)∘(-1,0.3):
        // sign +1, angle = -1·0.3 + 0.9 = 0.6.
        assert!((r.net_angle - 0.6).abs() < 1e-12);
    }

    #[test]
    fn single_reflection_stays_a_reflection() {
        let edges = [(1i8, 0.2), (-1, 0.4)];
        let defects = [1e-6, 1e-6];
        let r = loop_holonomy(&edges, &defects);
        assert_eq!(r.net_sign, -1);
        // A reflection is never the identity.
        assert!(!r.is_trivial);
    }

    #[test]
    fn tolerance_above_net_angle_cannot_exclude_identity() {
        // Same π/7 net rotation, but defects summing above π/7: the bound can't
        // distinguish it from the identity, so measure-don't-latch calls it
        // trivial (we do not latch a nontrivial verdict the data can't support).
        let edges = [(1i8, PI / 7.0)];
        let defects = [PI / 7.0 + 0.01];
        let r = loop_holonomy(&edges, &defects);
        assert!(r.angle_tolerance > (PI / 7.0));
        assert!(r.is_trivial);
    }

    #[test]
    fn empty_loop_is_trivial_identity() {
        let r = loop_holonomy(&[], &[]);
        assert_eq!(r.loop_len, 0);
        assert_eq!(r.net_sign, 1);
        assert_eq!(r.net_angle, 0.0);
        assert!(r.is_trivial);
    }

    #[test]
    fn adapter_matches_plain_interface() {
        let reports = [
            CircleTransportReport {
                layer_from: 0,
                layer_to: 1,
                n_samples: 128,
                winding: -1,
                phase: 0.3,
                defect: 1e-6,
                resultant_shift: 0.0,
                resultant_reflect: 1.0,
                class: crate::inference::transport_class::CircleTransportClass::Reflect,
            },
            CircleTransportReport {
                layer_from: 1,
                layer_to: 2,
                n_samples: 128,
                winding: -1,
                phase: 0.9,
                defect: 1e-6,
                resultant_shift: 0.0,
                resultant_reflect: 1.0,
                class: crate::inference::transport_class::CircleTransportClass::Reflect,
            },
        ];
        let via_adapter = holonomy_from_transports(&reports);
        let via_plain = loop_holonomy(&[(-1, 0.3), (-1, 0.9)], &[1e-6, 1e-6]);
        assert_eq!(via_adapter.net_sign, via_plain.net_sign);
        assert!((via_adapter.net_angle - via_plain.net_angle).abs() < 1e-15);
        assert_eq!(via_adapter.is_trivial, via_plain.is_trivial);
    }
}
