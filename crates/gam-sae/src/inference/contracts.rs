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
//! 1. **Contract composition** ([`compose_contracts`]): the shadowing bound that
//!    turns a chain of per-component error/expansion certificates into a single
//!    end-to-end error bound, with a drift-only domain-feasibility check at the
//!    nominal input; [`whole_set_containment`] extends the claim to a declared
//!    ball of inputs.
//! 2. **Loop holonomy** ([`loop_holonomy`]): the
//!    net `O(2)` element obtained by composing the transports around a closed
//!    loop of charts, with a trivial/nontrivial verdict whose tolerance is
//!    DERIVED from the loop's own defects (never a magic constant).

use std::f64::consts::{PI, TAU};

use gam_linalg::roundoff::accumulation_growth;

/// One component map of a chart-coordinate pipeline, abstracted to exactly the
/// three numbers a composition bound needs.
///
/// The map is a certificate, not the map itself: it says that on a coordinate
/// ball of radius [`domain_radius`](Self::domain_radius) about its center (the
/// nominal trajectory, unless [`whole_set_containment`] declares an offset),
/// the realized map stays within [`defect`](Self::defect) of the
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

/// The end-to-end certificate produced by composing a [`Contract`] chain.
#[derive(Debug, Clone)]
pub struct ComposedContract {
    /// Shadowing bound on the sup-norm gap between the composed realized map
    /// and the composed true map over the whole chain, at every input whose
    /// trajectories stay inside each stage's certified ball: the nominal input
    /// when [`domain_ok`](Self::domain_ok), every input of the initial ball when
    /// [`WholeSetContainment::contained`].
    pub total_defect: f64,
    /// `per_stage_contribution[j]` is stage `j`'s additive share of
    /// [`total_defect`](Self::total_defect): `defect_j · Π_{i>j} lipschitz_i`.
    /// Sums to `total_defect`. Earlier stages carry more when the later
    /// expansion factors exceed 1 (their error is amplified by more stages).
    pub per_stage_contribution: Vec<f64>,
    /// Whether the accumulated error stays inside every stage's certified
    /// domain along the chain: the drift-only check (accumulated error alone,
    /// with its rounding band, must fit each `domain_radius`). It licenses
    /// [`total_defect`](Self::total_defect) at the nominal input only, and a
    /// non-finite certificate that reaches a later stage fails it.
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
/// The induction evaluates `f_k`'s defect and `g_k`'s expansion at `F_{k-1}(x)`
/// and `G_{k-1}(x)`, so it holds only where both lie inside stage `k`'s
/// certified ball. [`domain_ok`](ComposedContract::domain_ok) is that
/// requirement at the nominal input, with every ball centered on the nominal
/// trajectory: the error accumulated *before* stage `j` (namely `E_{j-1}`) must
/// not already exceed stage `j`'s `domain_radius`, else the realized trajectory
/// has drifted outside where `f_j`'s certificate holds. A claim over a set of
/// inputs needs [`whole_set_containment`].
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

    // Drift-only domain check: the whole-set requirement at initial radius 0
    // with every certificate centered on the nominal trajectory, so the error
    // entering stage j is E_{j-1} alone.
    let domain_ok = entry_requirements(chain, 0.0, &vec![0.0; n])
        .iter()
        .all(|stage| stage.contained);

    ComposedContract {
        total_defect,
        per_stage_contribution,
        domain_ok,
    }
}

/// Stage `k`'s entry requirement in a whole-set claim (see
/// [`whole_set_containment`]). Every distance is in stage `k`'s input chart
/// metric.
#[derive(Debug, Clone)]
pub struct StageContainment {
    /// `R_{k-1} = initial_radius · Π_{i<k} lipschitz_i`: every TRUE trajectory
    /// from the initial ball enters the stage within this distance of the
    /// nominal trajectory.
    pub nominal_spread: f64,
    /// `E_{k-1}`, the shadowing bound of the stages before this one: every
    /// REALIZED trajectory enters the stage within this distance of its true
    /// trajectory.
    pub drift: f64,
    /// `s_k`, the declared distance from the nominal trajectory's point entering
    /// the stage to the center of the stage's certified ball.
    pub nominal_offset: f64,
    /// `s_k + R_{k-1} + E_{k-1}`: every true and every realized trajectory from
    /// the initial ball enters the stage within this distance of the
    /// certificate's center.
    pub required_radius: f64,
    /// `γ_{2k+3} · required_radius` (Higham, ASNA 2nd ed., Lemma 3.1). The
    /// longest rounded path to `required_radius` is `2k` operations on
    /// non-negative operands (`k − 1` products and `k − 1` additions in `E`, one
    /// addition each for `R` and `s`); one more increment turns the computed
    /// value's relative error into an upper bound on the exact value, and two
    /// more cover forming this band and adding it.
    pub rounding_band: f64,
    /// Whether `required_radius + rounding_band ≤ domain_radius`. A NaN
    /// requirement or radius is never contained.
    pub contained: bool,
}

/// A [`Contract`] chain's containment over a declared ball of inputs.
#[derive(Debug, Clone)]
pub struct WholeSetContainment {
    /// The chain's shadowing bound.
    pub composed: ComposedContract,
    /// Radius of the declared initial ball about the nominal input, in the
    /// first stage's chart metric.
    pub initial_radius: f64,
    /// `stages[k-1]` is stage `k`'s entry requirement.
    pub stages: Vec<StageContainment>,
    /// Whether every stage is contained, so that
    /// [`composed.total_defect`](ComposedContract::total_defect) bounds
    /// `|F_n(x) − G_n(x)|` for every input `x` of the initial ball.
    pub contained: bool,
}

/// Check that a [`Contract`] chain's shadowing bound holds over a whole ball of
/// inputs, not only at the nominal input.
///
/// Let `x̄_0` be the nominal input, `x̄_k = g_k(x̄_{k-1})` the nominal trajectory
/// through the true maps, `c_k` the center of stage `k`'s certified ball, and
/// `s_k = |x̄_{k-1} − c_k|` its declared offset (`nominal_offsets[k-1]`, zero
/// when the certificate was built about the nominal trajectory). If every stage
/// satisfies
///
/// ```text
/// s_k + R_{k-1} + E_{k-1} ≤ domain_radius_k,
/// R_k = initial_radius · Π_{i≤k} lipschitz_i,   E_k = defect_k + lipschitz_k · E_{k-1},   E_0 = 0,
/// ```
///
/// then `|F_n(x) − G_n(x)| ≤ E_n = total_defect` for every `x` with
/// `|x − x̄_0| ≤ initial_radius`.
///
/// # Proof (induction on `k`)
///
/// Assume `|G_{k-1}(x) − x̄_{k-1}| ≤ R_{k-1}` and `|F_{k-1}(x) − G_{k-1}(x)| ≤
/// E_{k-1}`, which hold at `k = 1`. Then `x̄_{k-1}`, `G_{k-1}(x)` and
/// `F_{k-1}(x)` lie within `s_k`, `s_k + R_{k-1}` and `s_k + R_{k-1} + E_{k-1}`
/// of `c_k`, all inside the certified ball, so
///
/// ```text
/// |G_k(x) − x̄_k|    = |g_k(G_{k-1}(x)) − g_k(x̄_{k-1})| ≤ lipschitz_k · R_{k-1} = R_k,
/// |F_k(x) − G_k(x)| ≤ |f_k(F_{k-1}(x)) − g_k(F_{k-1}(x))| + |g_k(F_{k-1}(x)) − g_k(G_{k-1}(x))|
///                   ≤ defect_k + lipschitz_k · E_{k-1} = E_k.
/// ```
///
/// Stage 1's requirement `s_1 + initial_radius ≤ domain_radius_1` puts the
/// initial ball inside the first certificate. The drift-only
/// [`ComposedContract::domain_ok`] drops `R` and `s`, so a stage that expands
/// the metric can carry true trajectories from the initial ball out of a later
/// certificate while the drift still fits, and the bound then says nothing about
/// them; the tests pin such an input.
///
/// Refuses a negative or non-finite initial radius, offset, defect or Lipschitz
/// constant, a negative or NaN domain radius (`+∞` is a global certificate), and
/// an offset count that differs from the chain length.
pub fn whole_set_containment(
    chain: &[Contract],
    initial_radius: f64,
    nominal_offsets: &[f64],
) -> Result<WholeSetContainment, String> {
    if !(initial_radius.is_finite() && initial_radius >= 0.0) {
        return Err(format!(
            "whole-set containment needs a finite non-negative initial radius; got {initial_radius}"
        ));
    }
    if nominal_offsets.len() != chain.len() {
        return Err(format!(
            "the chain has {} stages but {} nominal offsets were declared",
            chain.len(),
            nominal_offsets.len()
        ));
    }
    for (stage, &offset) in chain.iter().zip(nominal_offsets) {
        for (label, value) in [
            ("nominal offset", offset),
            ("defect", stage.defect),
            ("lipschitz constant", stage.lipschitz),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!(
                    "stage {:?} has a {label} of {value}; it must be finite and non-negative",
                    stage.name
                ));
            }
        }
        if !(stage.domain_radius >= 0.0) {
            return Err(format!(
                "stage {:?} has a domain radius of {}; it must be non-negative (+inf for a global \
                 certificate)",
                stage.name, stage.domain_radius
            ));
        }
    }
    let stages = entry_requirements(chain, initial_radius, nominal_offsets);
    let contained = stages.iter().all(|stage| stage.contained);
    Ok(WholeSetContainment {
        composed: compose_contracts(chain),
        initial_radius,
        stages,
        contained,
    })
}

/// Each stage's entry requirement `s_k + R_{k-1} + E_{k-1}` with its rounding
/// band, unvalidated: a NaN defect or Lipschitz constant leaves every later
/// stage uncontained, and a NaN offset or radius its own stage.
fn entry_requirements(
    chain: &[Contract],
    initial_radius: f64,
    nominal_offsets: &[f64],
) -> Vec<StageContainment> {
    let mut stages = Vec::with_capacity(chain.len());
    let mut nominal_spread = initial_radius;
    let mut drift = 0.0_f64;
    for (index, (stage, &nominal_offset)) in chain.iter().zip(nominal_offsets).enumerate() {
        let required_radius = nominal_offset + nominal_spread + drift;
        // Stage k = index + 1: see `StageContainment::rounding_band` for γ_{2k+3}.
        let rounding_band = accumulation_growth(2 * index + 5) * required_radius;
        stages.push(StageContainment {
            nominal_spread,
            drift,
            nominal_offset,
            required_radius,
            rounding_band,
            contained: required_radius + rounding_band <= stage.domain_radius,
        });
        nominal_spread *= stage.lipschitz;
        drift = stage.defect + stage.lipschitz * drift;
    }
    stages
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

/// The inverse of the `O(2)` edge `(sign, angle)`.
///
/// The element acts as `x ↦ sign·x + angle`; its inverse is `y ↦ sign·(y −
/// angle) = sign·y − sign·angle` (using `sign² = 1`), i.e. `(sign, −sign·angle)`.
/// This is what closes a composition triangle into a loop: given the two
/// forward hops `h_ab, h_bc` and the direct map `h_ac`, the loop
/// `h_ab, h_bc, h_ac⁻¹` returns to the start, so its [`loop_holonomy`] measures
/// exactly the failure of the composition law `h_ac = h_bc ∘ h_ab` as an `O(2)`
/// element.
pub fn invert_o2_edge(edge: (i8, f64)) -> (i8, f64) {
    let s = if edge.0 >= 0 { 1i8 } else { -1i8 };
    (s, -(s as f64) * edge.1)
}

/// Compose a closed loop of circle isometries and report the net `O(2)` element.
///
/// Each edge is an `O(2)` element `(sign, angle)` in the vocabulary of
/// [`transport_class`](crate::inference::transport_class): `sign = +1` is the
/// rotation `x ↦ x + angle`, `sign = −1` is the reflection `x ↦ −x + angle`
/// (matching `CircleTransportReport::winding` and `phase`). The elements are
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
///
/// Each `defects[i]` is therefore a sup-norm angular gap in radians between edge
/// `i`'s realized transport and its `O(2)` element, such as
/// [`CircleTransportReport::max_angle_gap`](crate::inference::transport_class::CircleTransportReport::max_angle_gap).
/// The circular variance `CircleTransportReport::defect` (`1 − R`) is not an
/// angle. It is about half the squared gap, so summing it gives a tolerance far
/// below the bound and calls noisy loops nontrivial.
///
/// Every input must be what the tolerance derivation assumes, so a malformed
/// loop is refused rather than repaired:
/// - one defect per edge, since the bound sums over every edge;
/// - every defect finite and non-negative, since dropping one would shrink the
///   tolerance and let an edge of unknown size certify a nontrivial loop;
/// - every sign `+1` or `−1` and every angle finite, since an element outside
///   `O(2)` has no place in the fold.
pub fn loop_holonomy(edges: &[(i8, f64)], defects: &[f64]) -> Result<HolonomyReport, String> {
    if defects.len() != edges.len() {
        return Err(format!(
            "loop_holonomy: the loop has {} edges but {} defects were declared; the tolerance \
             sums one defect per edge",
            edges.len(),
            defects.len()
        ));
    }
    for (index, (&(sign, angle), &defect)) in edges.iter().zip(defects).enumerate() {
        if sign != 1 && sign != -1 {
            return Err(format!(
                "loop_holonomy: edge {index} has sign {sign}; an O(2) element has sign +1 or -1"
            ));
        }
        if !angle.is_finite() {
            return Err(format!("loop_holonomy: edge {index} has a non-finite angle {angle}"));
        }
        if !(defect.is_finite() && defect >= 0.0) {
            return Err(format!(
                "loop_holonomy: edge {index} has defect {defect}; it must be finite and non-negative"
            ));
        }
    }

    // Fold the O(2) elements in loop order: acc ← edge ∘ acc.
    let mut acc_sign = 1i8;
    let mut acc_angle = 0.0_f64;
    for &(sign, angle) in edges {
        acc_angle = f64::from(sign) * acc_angle + angle;
        acc_sign *= sign;
    }
    let net_angle = wrap_pi(acc_angle);

    // Derived tolerance: composed defect bound of the loop = Σ defects, since
    // isometries have Lipschitz 1 (see the doc comment).
    let angle_tolerance = defects.iter().sum::<f64>();

    let is_trivial = acc_sign == 1 && net_angle.abs() <= angle_tolerance;

    Ok(HolonomyReport {
        loop_len: edges.len(),
        net_sign: acc_sign,
        net_angle,
        is_trivial,
        angle_tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::accumulation_band;

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
    fn rotations_summing_to_zero_are_trivial() {
        // Pure rotations that sum to 2π ≡ 0, tiny defects.
        let edges = [(1i8, 2.0), (1, 2.0), (1, TAU - 4.0)];
        let defects = [1e-6, 1e-6, 1e-6];
        let r = loop_holonomy(&edges, &defects).unwrap();
        assert_eq!(r.net_sign, 1);
        assert!(r.net_angle.abs() < 1e-9, "net_angle = {}", r.net_angle);
        assert!(r.is_trivial);
    }

    #[test]
    fn small_net_rotation_with_tiny_defects_is_nontrivial() {
        // Net rotation π/7, defects far below it → cannot be excluded as noise.
        let edges = [(1i8, PI / 7.0)];
        let defects = [1e-4];
        let r = loop_holonomy(&edges, &defects).unwrap();
        assert_eq!(r.net_sign, 1);
        assert!((r.net_angle - PI / 7.0).abs() < 1e-12);
        assert!(!r.is_trivial);
    }

    #[test]
    fn two_reflections_compose_to_a_rotation() {
        let edges = [(-1i8, 0.3), (-1, 0.9)];
        let defects = [1e-6, 1e-6];
        let r = loop_holonomy(&edges, &defects).unwrap();
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
        let r = loop_holonomy(&edges, &defects).unwrap();
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
        let r = loop_holonomy(&edges, &defects).unwrap();
        assert!(r.angle_tolerance > (PI / 7.0));
        assert!(r.is_trivial);
    }

    #[test]
    fn empty_loop_is_trivial_identity() {
        let r = loop_holonomy(&[], &[]).unwrap();
        assert_eq!(r.loop_len, 0);
        assert_eq!(r.net_sign, 1);
        assert_eq!(r.net_angle, 0.0);
        assert!(r.is_trivial);
    }

    #[test]
    fn loop_holonomy_refuses_inputs_the_tolerance_cannot_bound() {
        // A net rotation of π/7 with one tiny defect and one NaN defect used to
        // drop the NaN and certify the loop nontrivial on a tolerance of 1e-4.
        let edges = [(1i8, PI / 7.0), (1, 0.0)];
        for defects in [[1e-4, f64::NAN], [1e-4, f64::INFINITY], [1e-4, -1.0]] {
            let error = loop_holonomy(&edges, &defects).expect_err("invalid defect");
            assert!(error.contains("edge 1 has defect"), "{error}");
        }
        // One defect short: the missing edge's defect is unknown.
        let error = loop_holonomy(&edges, &[1e-4]).expect_err("short defects");
        assert!(error.contains("2 edges but 1 defects"), "{error}");
        // Signs outside {+1, -1} and non-finite angles are not O(2) elements.
        let error = loop_holonomy(&[(0i8, 0.1)], &[0.0]).expect_err("sign 0");
        assert!(error.contains("sign 0"), "{error}");
        let error = loop_holonomy(&[(1i8, f64::NAN)], &[0.0]).expect_err("NaN angle");
        assert!(error.contains("non-finite angle"), "{error}");
    }

    #[test]
    fn invert_o2_edge_round_trips_to_identity() {
        use crate::inference::contracts::invert_o2_edge;
        // A rotation and its inverse compose to the identity.
        let e = (1i8, 0.7);
        let inv = invert_o2_edge(e);
        let r = loop_holonomy(&[e, inv], &[0.0, 0.0]).unwrap();
        assert_eq!(r.net_sign, 1);
        assert!(r.net_angle.abs() < 1e-12);
        assert!(r.is_trivial);
        // A reflection is its own inverse's sign; edge·inv = identity too.
        let f = (-1i8, 1.1);
        let finv = invert_o2_edge(f);
        assert_eq!(finv.0, -1);
        let r2 = loop_holonomy(&[f, finv], &[0.0, 0.0]).unwrap();
        assert_eq!(r2.net_sign, 1);
        assert!(r2.net_angle.abs() < 1e-12);
    }

    #[test]
    fn whole_set_containment_refuses_an_expanding_spread_that_drift_only_accepts() {
        // Stage 1: g₁(x) = 3x, realized f₁(x) = 3x + 0.01, certified on |x| ≤ 1.
        // Stage 2: g₂(y) = y, realized f₂(y) = y + 0.01 on its certified ball
        // |y| ≤ 1 and y + 100 outside it, which the certificate permits. The
        // nominal input is 0, so the nominal trajectory is 0 and both balls are
        // centered on it.
        let chain = [c("expand", 1.0, 0.01, 3.0), c("identity", 1.0, 0.01, 1.0)];
        let realized = |x: f64| {
            let y = 3.0 * x + 0.01;
            if y.abs() <= 1.0 { y + 0.01 } else { y + 100.0 }
        };
        let truth = |x: f64| 3.0 * x;
        let composed = compose_contracts(&chain);
        assert_eq!(composed.total_defect, 0.02);
        assert!(composed.domain_ok, "the drift-only check accepts the nominal input");
        let point = whole_set_containment(&chain, 0.0, &[0.0, 0.0]).expect("valid chain");
        assert_eq!(point.contained, composed.domain_ok);

        // Positive control: x = 0.5 enters stage 2 at 1.51, outside |y| ≤ 1, and
        // breaks the bound by two orders of magnitude.
        let gap = (realized(0.5) - truth(0.5)).abs();
        assert!(gap > 99.0 + composed.total_defect, "gap {gap}");
        let wide = whole_set_containment(&chain, 0.5, &[0.0, 0.0]).expect("valid chain");
        assert!(!wide.contained);
        assert!(wide.stages[0].contained, "the initial ball fits the first certificate");
        assert!(!wide.stages[1].contained);
        assert_eq!(wide.stages[1].nominal_spread, 1.5);
        assert_eq!(wide.stages[1].drift, 0.01);

        // Radius 0.3: stage 2 needs 0.9 + 0.01 ≤ 1, so the bound must hold at
        // every input of the ball, the adversarial branch included.
        let narrow = whole_set_containment(&chain, 0.3, &[0.0, 0.0]).expect("valid chain");
        assert!(narrow.contained);
        let steps = 600;
        for step in 0..=steps {
            let x = -0.3 + 0.6 * step as f64 / steps as f64;
            // `realized − truth` rounds through two additions and a subtraction
            // on operands no larger than |3x| + 0.02, both sides share the
            // product 3x, and `total_defect` is the exact double of 0.01.
            let band = accumulation_band(5, 2.0 * (3.0 * x).abs() + 0.04);
            let gap = (realized(x) - truth(x)).abs();
            assert!(gap <= composed.total_defect + band, "x = {x}: gap {gap}");
        }
    }

    #[test]
    fn whole_set_containment_charges_the_certificate_center_offset() {
        // The chain above, with stage 2's certificate built about c₂ = 0.2
        // instead of the nominal point 0: f₂(y) = y + 0.01 on |y − 0.2| ≤ 1 and
        // y + 100 outside.
        let chain = [c("expand", 1.0, 0.01, 3.0), c("identity", 1.0, 0.01, 1.0)];
        let realized = |x: f64| {
            let y = 3.0 * x + 0.01;
            if (y - 0.2).abs() <= 1.0 { y + 0.01 } else { y + 100.0 }
        };
        let truth = |x: f64| 3.0 * x;
        let centered = whole_set_containment(&chain, 0.3, &[0.0, 0.0]).expect("valid chain");
        assert!(centered.contained, "ignoring the offset, stage 2 needs 0.91 ≤ 1");
        // Positive control: x = −0.3 enters stage 2 at −0.89, outside |y − 0.2| ≤ 1.
        let gap = (realized(-0.3) - truth(-0.3)).abs();
        assert!(gap > 99.0 + centered.composed.total_defect, "gap {gap}");
        let offset = whole_set_containment(&chain, 0.3, &[0.0, 0.2]).expect("valid chain");
        assert!(!offset.contained);
        assert!(offset.stages[0].contained);
        // 0.2 + 0.3·3 + 0.01: four inexact decimal literals (1.11 included),
        // three rounded operations and the comparing subtraction.
        let required = offset.stages[1].required_radius;
        assert!(
            (required - 1.11).abs() <= accumulation_band(8, 2.0 * 1.11),
            "required {required}"
        );
    }

    #[test]
    fn containment_at_the_exact_requirement_is_refused_by_the_rounding_band() {
        // Stage 2's requirement 0.5·3 + 0.25 = 1.75 is exact in binary. The band
        // is positive, so a radius of exactly 1.75 is refused, and a radius two
        // bands above it is accepted.
        let tight = [c("expand", 1.0, 0.25, 3.0), c("tight", 1.75, 0.0, 1.0)];
        let at = whole_set_containment(&tight, 0.5, &[0.0, 0.0]).expect("valid chain");
        assert_eq!(at.stages[1].required_radius, 1.75);
        assert!(at.stages[1].rounding_band > 0.0);
        assert!(!at.stages[1].contained);
        let above = 1.75 + 2.0 * at.stages[1].rounding_band;
        let loose = [c("expand", 1.0, 0.25, 3.0), c("loose", above, 0.0, 1.0)];
        let accepted = whole_set_containment(&loose, 0.5, &[0.0, 0.0]).expect("valid chain");
        assert!(accepted.contained);
        // A global certificate contains any finite requirement.
        let global = [c("expand", 1.0, 0.25, 3.0), c("global", f64::INFINITY, 0.0, 1.0)];
        let everywhere = whole_set_containment(&global, 0.5, &[0.0, 0.0]).expect("valid chain");
        assert!(everywhere.contained);
    }

    #[test]
    fn a_non_finite_certificate_fails_closed() {
        // A NaN Lipschitz constant at stage 1 makes the drift entering stage 2
        // NaN, and the predicate "refuse when the drift exceeds the radius"
        // cannot see it.
        let nan_drift = f64::NAN;
        assert!(!(nan_drift > 1.0), "the exceed-radius predicate accepts a NaN drift");
        let chain = [c("broken", 1.0, 0.01, f64::NAN), c("next", 1.0, 0.01, 1.0)];
        assert!(!compose_contracts(&chain).domain_ok);
        let refused = whole_set_containment(&chain, 0.0, &[0.0, 0.0]).expect_err("NaN lipschitz");
        assert!(refused.contains("lipschitz constant"), "{refused}");
        let miscounted =
            whole_set_containment(&chain[..1], 0.0, &[0.0, 0.0]).expect_err("offset count");
        assert!(miscounted.contains("1 stages but 2 nominal offsets"), "{miscounted}");
    }

}
