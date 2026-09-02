use crate::parameter_block::ParameterBlockInput;
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam_solve::pirls::LinearInequalityConstraints;
use gam_terms::basis::ispline_function_penalties;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct WiggleBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub double_penalty: bool,
}

/// Semantic identity of one canonical I-spline penalty block.
///
/// The order of these values is the smoothing-parameter order. Persisting the
/// topology prevents inference code from guessing a derivative order from a
/// lambda index or inventing a zero block when the guess is invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum WigglePenaltyBlockKind {
    Roughness { derivative_order: usize },
    NullspaceShrinkage { derivative_order: usize },
}

/// Complete semantic description of a realized monotone-wiggle penalty list.
///
/// `derivative_orders` is already canonicalized into the exact roughness-block
/// order used by fitting: primary first, followed by deduplicated additional
/// orders. `blocks` additionally records whether the primary roughness emitted
/// a function-metric nullspace shrinkage coordinate. For example, an order-one
/// anchored I-spline roughness is full rank, so `double_penalty=true` emits no
/// synthetic ridge block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WigglePenaltyMetadata {
    pub derivative_orders: Vec<usize>,
    pub double_penalty: bool,
    pub blocks: Vec<WigglePenaltyBlockKind>,
}

/// Exact matrices and nullities accompanying [`WigglePenaltyMetadata`].
#[derive(Clone, Debug)]
pub struct CanonicalWigglePenaltySet {
    pub metadata: WigglePenaltyMetadata,
    pub matrices: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct SelectedWiggleBasis {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub block: ParameterBlockInput,
    pub penalty_metadata: WigglePenaltyMetadata,
}

// gam#2695: the WARP knot generator. A monotone warp is composed onto a
// β-dependent index, so a boundary knot's multiplicity is a step in the basis's
// own derivative tower and therefore in the inner objective. Warp blocks build
// their knots here; the clamped `gam_terms::basis::initializewiggle_knots_from_seed`
// stays for bases evaluated on FIXED data — a response transform — where the
// evaluation point never moves across that boundary.
pub(crate) use gam_terms::basis::monotone_warp_knots_from_seed;

/// The highest derivative of a composed warp's basis that must be continuous
/// for the optimized objective to be continuously differentiable (gam#2695).
///
/// A composed warp is not evaluated on fixed data. It sits on the model's index,
/// `q = q₀ + Σ_j βw_j·I_j(q₀)` with `q₀ = −η_t·e^{−η_ls}`, so `q₀` moves with β
/// while the knots stay where the seed put them, and the objective differentiates
/// the basis rather than merely evaluating it.
///
/// # Read off the row program, not argued
///
/// `sls_row_nll_wiggle` composes the basis into the row jet at two places, and
/// the second is the one that sets this order:
///
/// ```text
///   q₁ʷ = q₁ + Σ βw_j·I_j(q₁)          stack [I, I′, I″, I‴, I⁗]
///   m₁  = 1  + Σ βw_j·I′_j(q₁)         stack [I′, I″, I‴, I⁗, …]   <- SHIFTED BY ONE
///   g   = η_t′ + m₁·q̇₀,  and  ℓ ∋ −d·log g
/// ```
///
/// `H = ∂²(−ℓ)/∂β²` is the order-2 coefficient of that jet, and `m₁`'s order-2
/// coefficient reads its stack's slot 2 — which, because `m₁` is built from `I′`,
/// is **`I‴`**. `Φ = ½ Σ g(λ(Z_JᵀHZ_J))` is a TERM OF THE OBJECTIVE, not a
/// diagnostic about it — the inner NLL is `−ℓ + ½βᵀSβ − Φ` — so the objective's
/// own value reads `I‴`, and a step in `I‴` is a jump in the function the trust
/// region compares two points on.
///
/// Continuity of the objective value is necessary but not sufficient for the
/// analytic-gradient optimizer. Its Newton right-hand side and convergence gate
/// consume `∇Φ`, so the objective has to be `C¹` wherever an event row's moving
/// index crosses a knot. Which basis order that pins is MEASURED, not derived:
/// `knot_ladder_2695` drives the production `∇Φ` across such a crossing at
/// every degree from 3 to 6 and reads the gap at two straddles 100× apart. The
/// gap shrinks 99.5× at degree 4 and ≈100× at 5 and 6 (a continuous gradient)
/// and 1.02× at degree 3 (a step). So the order that must stay continuous is
/// the third: degree 4 is the first `C¹` degree. The exact outer curvature
/// evaluates the fourth and fifth derivatives, but does not require them to be
/// continuous.
///
/// # The whole ladder, so the floor is read for what it buys
///
/// This constant is the highest order that must remain continuous. Each lowering
/// of the row jet reads one order higher through `m₁`'s shifted stack, and each
/// has a named consumer:
///
/// ```text
///   jet order  lowering                          basis order  consumer
///   0          value                             I′  (m₁)     ℓ in the accept test
///   1          gradient                          I″  (m₁)     ∇ℓ, KKT residual
///   2          row_order2 → H                    I‴  (m₁)     Φ = ½Σ g(λ(Z_JᵀHZ_J))  <- objective VALUE
///   3          row_third_contracted → ∂H/∂β      I⁗  (m₁)     ∇Φ (Newton rhs, KKT residual), Daleckii–Krein H_Φ
///   4          row_fourth_contracted → ∂²H/∂β²   I⁗′ (m₁)     exact Jeffreys completion, outer second directional
/// ```
///
/// A degree-`d` I-spline is `C^{d−1}` at a simple knot. The table would read
/// as "∇Φ needs `I⁗` continuous, hence degree 5", and that reading was shipped
/// once (`793b9d072`): its own non-vacuity arm then refused, because on the
/// production gradient the degree-4 crossing gap scales linearly with the
/// straddle (ratio 99.5 for a 100× shrink) while the degree-3 gap does not
/// (ratio 1.02). Whatever the assembled `∂H/∂β` contracts against at a
/// crossing, the piecewise-constant part of `I⁗` does not survive into `∇Φ`,
/// and the piecewise-constant `I‴` of degree 3 does. The constant is the
/// measured order, `3`, hence degree `4`; the ladder that fixes it prints its
/// table on every run.
pub(crate) const COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER: usize = 3;

/// The smallest public degree at which a composed warp's basis is continuous to
/// [`COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER`], hence the
/// smallest degree at which the inner objective is continuously differentiable.
///
/// A degree-`d` I-spline is `C^{d−1}` at a knot of multiplicity one, so the
/// requirement `d − 1 ≥ order` is what this returns. Nothing is chosen: the
/// order comes from the objective and the `−1` from the spline.
///
/// # Why this RAISES the degree instead of refusing it
///
/// `5139b8977` derived the same floor and REFUSED below it. That was reverted in
/// `3c530dbda` for a reason that still holds: a refusal breaks every working fit
/// that asked for degree 2 and buys none of them a fit. The requirement is not a
/// statement about what the user may ask for — it is a property of the objective
/// the model then has to minimise, in the same class as "a B-spline basis needs
/// at least `degree + 1` knots". So a composed warp is BUILT at this degree, the
/// realised degree is what the block, the knots, the penalties and the saved
/// metadata all carry, and the raise is logged rather than silent.
#[inline]
pub(crate) fn composed_warp_minimum_degree() -> usize {
    COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER + 1
}

#[inline]
pub(crate) fn monotone_wiggle_internal_degree(degree: usize) -> Result<usize, String> {
    // Public monotone-wiggle degree refers to the value basis. The low-level
    // I-spline builder integrates a degree-`internal_degree` specification
    // into a degree-`internal_degree + 1` value basis, so we subtract one here
    // to keep the public degree and the per-span value degree aligned.
    degree
        .checked_sub(1)
        .filter(|&internal_degree| internal_degree >= 1)
        .ok_or_else(|| "monotone wiggle degree must be >= 2".to_string())
}

/// Build the exact ordered function-space penalty set for an anchored
/// I-spline monotone wiggle.
///
/// `derivative_orders` must already be in the fitting order and contain no
/// duplicates. The first order is the primary roughness; only its structural
/// null space is eligible for the separate double-penalty coordinate. Every
/// matrix comes from the canonical `C^T S_B C` function Gram, never a
/// coefficient difference or identity metric.
///
/// # The assembled set always has a trivial joint null space (gam#2647)
///
/// A monotone link warp is not an ordinary smooth. It is composed onto a free
/// index — `q = q₀ + w(q₀)` with `q₀ = −η_t·e^{−η_ls}` (binomial location-scale)
/// or `q = η + w(η)` (binomial mean) — and the index carries its own free
/// scale. That makes the warp's LINEAR direction a gauge rather than a shape:
/// for any `s > 0`,
///
/// ```text
///   (β_index, β_w)  ↦  (β_index / s,  β_w + (s−1)·ℓ),      B·ℓ = (u − left)
/// ```
///
/// reproduces the same `q` on every row inside the knot hull, because
/// `q₀/s + (s−1)(q₀/s − left) + w(q₀/s)` is `q₀` up to the constant the warp can
/// itself absorb. The likelihood is exactly invariant along that orbit. The
/// PENALTY is not: the index block is penalized, so `½βᵀSβ` falls like `1/s²`
/// there. If `ℓ` is unpenalized the penalized objective therefore decreases
/// monotonically in `s` and has **no minimiser** — the inner solve walks the
/// orbit forever, `‖β‖∞` diverging while `½βᵀSβ` falls exactly like `‖β‖⁻²`,
/// which is the measured signature on gam#2647.
///
/// An order-`k` anchored I-spline roughness has structural nullity `k − 1`, so
/// every set whose smallest order exceeds one leaves `ℓ` free unless something
/// closes it. `double_penalty` is a user knob and cannot be the thing that
/// decides whether the criterion is bounded below, so the closure below is
/// unconditional: after assembling the requested roughness blocks, the JOINT
/// null space of the set (at unit smoothing) is computed in the function metric
/// and, when non-trivial, one shrinkage coordinate spanning it is appended.
///
/// This is the same treatment — and the same argument — the binomial
/// location-scale log-σ block already receives unconditionally in
/// `build_binomial_threshold_and_scale_blocks`, where `(β_t, β_ls) ↦ (c·β_t,
/// β_ls + ln c)` is the exactly analogous index-scale gauge and an identity
/// shrinkage penalty is appended that the caller never asked for and cannot
/// switch off. The wiggle block simply never received it.
///
/// It is a **no-op on every already-well-posed configuration**: the shipped
/// default (`orders = [1, 2, 3]`) contains the order-one roughness, which is
/// full rank on the anchored basis, so the joint null space is already trivial
/// and nothing is appended; likewise whenever `double_penalty` already closed a
/// single-order set. Only configurations whose criterion is otherwise unbounded
/// gain a coordinate, and that coordinate's strength is chosen by REML like any
/// other.
pub fn canonical_wiggle_function_penalties(
    knots: &Array1<f64>,
    degree: usize,
    derivative_orders: &[usize],
    double_penalty: bool,
) -> Result<CanonicalWigglePenaltySet, String> {
    if derivative_orders.is_empty() {
        return Err("wiggle penalty metadata requires at least one derivative order".to_string());
    }
    if derivative_orders.contains(&0) {
        return Err("wiggle penalty derivative orders must all be positive".to_string());
    }
    for (index, &order) in derivative_orders.iter().enumerate() {
        if derivative_orders[..index].contains(&order) {
            return Err(format!(
                "wiggle penalty derivative order {order} is duplicated in canonical metadata"
            ));
        }
    }

    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    let mut blocks = Vec::new();
    let mut matrices = Vec::new();
    let mut nullspace_dims = Vec::new();
    for (index, &derivative_order) in derivative_orders.iter().enumerate() {
        let penalties = ispline_function_penalties(
            knots.view(),
            internal_degree,
            derivative_order,
            index == 0 && double_penalty,
        )
        .map_err(|error| error.to_string())?;
        blocks.push(WigglePenaltyBlockKind::Roughness { derivative_order });
        matrices.push(penalties.roughness);
        nullspace_dims.push(penalties.roughness_nullspace_dim);
        if let Some(nullspace_shrinkage) = penalties.nullspace_shrinkage {
            blocks.push(WigglePenaltyBlockKind::NullspaceShrinkage { derivative_order });
            matrices.push(nullspace_shrinkage);
            nullspace_dims.push(0);
        }
    }

    // Gauge closure (gam#2647) — see the type-level note above. The joint null
    // space is read off the SUM of the assembled blocks at unit smoothing, which
    // is `null(Σ S_j) = ⋂_j null(S_j)` exactly because every `S_j` is PSD, so a
    // direction survives only when NO requested block penalizes it. Reading the
    // sum (rather than the primary alone) is what makes this both complete —
    // a multi-order set is judged by what it collectively leaves free — and
    // idempotent: when `double_penalty` already emitted a shrinkage coordinate,
    // that coordinate is inside the sum, the intersection is empty, and nothing
    // is appended. Tagged with the PRIMARY derivative order, so a single-order
    // set with `double_penalty = true` and one with `double_penalty = false`
    // produce the same topology, which is the point.
    //
    // Each block enters the sum divided by its OWN mean diagonal. Without that
    // the test would not be "does any block penalize this direction" but "does
    // any block penalize it comparably to the stiffest block present": an
    // order-3 roughness has eigenvalues orders above an order-1 roughness on the
    // same knots, so on the shipped default (`orders = [1, 2, 3]`) the order-1
    // block — which is the one that closes the gauge — would sit inside the
    // rank tolerance of the order-3 block and the sum would report a null space
    // that does not exist, appending a coordinate to a configuration that never
    // needed one. A per-block scale is the only thing that makes the
    // intersection `⋂_j null(S_j)` the quantity actually computed, and it is
    // derived from the matrices rather than chosen.
    let primary_order = derivative_orders[0];
    let joint_dim = matrices.first().map_or(0, |m| m.nrows());
    if joint_dim > 0 {
        let mut joint = Array2::<f64>::zeros((joint_dim, joint_dim));
        for matrix in &matrices {
            let mean_diagonal =
                (0..joint_dim).map(|i| matrix[[i, i]].abs()).sum::<f64>() / joint_dim as f64;
            if !(mean_diagonal > 0.0) || !mean_diagonal.is_finite() {
                continue;
            }
            joint.scaled_add(1.0 / mean_diagonal, matrix);
        }
        // Failure here is propagated rather than swallowed. Skipping the closure
        // on a set we cannot certify as closed would ship exactly the criterion
        // this exists to prevent, and silently — a refusal naming the reason is
        // the strictly more useful outcome.
        let function_gram = gam_terms::basis::ispline_function_gram(knots.view(), internal_degree)
            .map_err(|error| {
                format!(
                    "wiggle gauge closure needs the I-spline function Gram to decide whether the \
                     assembled penalty set leaves a reparameterization of the index unpenalized, \
                     and it could not be built: {error}"
                )
            })?;
        if let Some(gauge_shrinkage) =
            gam_terms::basis::function_space_nullspace_shrinkage(&joint, &function_gram).map_err(
                |error| {
                    format!(
                        "wiggle gauge closure could not resolve the joint null space of the \
                         assembled penalty set: {error}"
                    )
                },
            )?
        {
            blocks.push(WigglePenaltyBlockKind::NullspaceShrinkage {
                derivative_order: primary_order,
            });
            matrices.push(gauge_shrinkage);
            nullspace_dims.push(0);
        }
    }

    Ok(CanonicalWigglePenaltySet {
        metadata: WigglePenaltyMetadata {
            derivative_orders: derivative_orders.to_vec(),
            double_penalty,
            blocks,
        },
        matrices,
        nullspace_dims,
    })
}

fn buildwiggle_block_input_from_canonical_penalties(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    canonical: &CanonicalWigglePenaltySet,
) -> Result<ParameterBlockInput, String> {
    let design = monotone_wiggle_basis_from_knots(seed, knots, degree)?;
    let p = design.ncols();
    if p == 0 {
        return Err("wiggle basis has no free monotone columns".to_string());
    }
    if canonical.matrices.len() != canonical.nullspace_dims.len()
        || canonical.matrices.len() != canonical.metadata.blocks.len()
    {
        return Err(
            "canonical wiggle penalty matrices, nullities, and topology disagree".to_string(),
        );
    }
    for (index, matrix) in canonical.matrices.iter().enumerate() {
        if matrix.dim() != (p, p) {
            return Err(format!(
                "canonical I-spline penalty block {index} is {}x{} but wiggle design has {p} columns",
                matrix.nrows(),
                matrix.ncols(),
            ));
        }
    }
    Ok(ParameterBlockInput {
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::zeros(seed.len()),
        penalties: canonical
            .matrices
            .iter()
            .cloned()
            .map(crate::model_types::PenaltySpec::Dense)
            .collect(),
        nullspace_dims: canonical.nullspace_dims.clone(),
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(p)),
    })
}

/// Build a monotone I-spline block carrying the COMPLETE requested penalty set.
///
/// Callers that want several derivative orders must come through here rather
/// than building a primary-order block and appending the rest: the gauge closure
/// in [`canonical_wiggle_function_penalties`] is a property of the assembled set
/// (it asks what the set collectively leaves unpenalized), so it can only be
/// decided once, on the final list. Assembling in two stages would judge the
/// primary order alone and could both add a coordinate the later orders made
/// unnecessary and miss one they left open.
///
/// The emitted order is unchanged from the previous two-stage assembly —
/// primary roughness, its optional double-penalty coordinate, then the extra
/// orders in the order given — so persisted penalty topologies are unaffected.
pub fn buildwiggle_block_input_from_orders(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    derivative_orders: &[usize],
    double_penalty: bool,
) -> Result<ParameterBlockInput, String> {
    let canonical =
        canonical_wiggle_function_penalties(knots, degree, derivative_orders, double_penalty)?;
    buildwiggle_block_input_from_canonical_penalties(seed, knots, degree, &canonical)
}

pub(crate) fn monotone_wiggle_basis_from_knots(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    monotone_wiggle_basis_with_derivative_order(seed, knots, degree, 0)
}

/// A monotone warp basis and every derivative order of it, as ONE function on
/// all of `ℝ`.
///
/// # What a warp needs that a shape basis does not (gam#2695)
///
/// A shape basis is evaluated on fixed data: its evaluation point never moves,
/// so how smooth it is at a knot is invisible. A WARP is composed onto the
/// model's own index — `q = q₀ + Σ_j βw_j·I_j(q₀)` with
/// `q₀ = −η_t·e^{−η_ls}` — so `q₀` moves with β while the knots stay where the
/// seed put them, and the inner objective differentiates the composition:
///
/// ```text
///     ℓ            reads  w′        (the event Jacobian q̇ = (1 + w′)·r)
///     ∇ℓ           reads  w″
///     H            reads  w‴        and Φ = ½Σ g(λ(Z_JᵀHZ_J)) reads H
///     ∇Φ = ∂H/∂β   reads  w⁗
/// ```
///
/// `Φ` is part of the objective the trust region accepts on, so a STEP in any
/// of `w′ … w‴` is a step in the objective, and `actual/predicted` cannot
/// approach `1` at any step size. The warp must therefore be `C³`, and each
/// column is `C^{degree−1−m}` at a knot of multiplicity `m`.
///
/// # Why this is one line now
///
/// It used to be the clamped I-spline inside its knot hull plus a hand-written
/// linear tail outside it, because `create_ispline_dense` is constant outside
/// the hull and a constant-extended I-spline has a corner there. The tail moved
/// the corner from order 1 to order 2 and no further, and it could not move it
/// further: at a clamped edge most columns have `I′ = 0` and `I″ ≠ 0`, and a
/// monotone `C²` extension of a column with `I′(e) = 0 > I″(e)` does not exist
/// (`I′` would have to go negative immediately). The corner is the boundary
/// knot's MULTIPLICITY, not the extrapolation rule, and the cure is a knot
/// vector whose ends are simple — [`gam_terms::basis::monotone_warp_knots`] —
/// evaluated by [`gam_terms::basis::ispline_ramp_basis_dense`], which is the
/// same ramp on all of `ℝ` and reproduces the clamped convention bit for bit
/// wherever that convention was right.
pub fn monotone_wiggle_basis_with_derivative_order(
    seed: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    derivative_order: usize,
) -> Result<Array2<f64>, String> {
    let internal_degree = monotone_wiggle_internal_degree(degree)?;
    gam_terms::basis::ispline_ramp_basis_dense(
        seed,
        knots.view(),
        internal_degree,
        derivative_order,
    )
    .map_err(|error| error.to_string())
}

/// The `β ≥ 0` system a monotone-wiggle block is subject to, as an explicit
/// dense system with unit rows.
///
/// This is the ONE definition of that cone. Both the constraint set the
/// blockwise QP enforces ([`monotone_wiggle_nonnegative_constraints`]) and any
/// line-search barrier that clips a step inside it are built from here, so the
/// two cannot be constructed from different systems — a barrier hook that
/// re-derives the cone by hand is how gam#2719's coordinate loop ended up with
/// a `1e-10` tolerance on the iterate and none at all on the step, while the QP
/// enforcing the identical rows worked to `1e-8`.
pub(crate) fn monotone_wiggle_nonnegative_system(
    beta_dim: usize,
) -> Option<LinearInequalityConstraints> {
    if beta_dim == 0 {
        return None;
    }
    let mut a = Array2::<f64>::zeros((beta_dim, beta_dim));
    for i in 0..beta_dim {
        a[[i, i]] = 1.0;
    }
    Some(LinearInequalityConstraints {
        a,
        b: Array1::zeros(beta_dim),
    })
}

pub(crate) fn monotone_wiggle_nonnegative_constraints(
    beta_dim: usize,
) -> Option<gam_solve::pirls::ConstraintSet> {
    monotone_wiggle_nonnegative_system(beta_dim).map(gam_solve::pirls::ConstraintSet::Dense)
}

pub(crate) fn validate_monotone_wiggle_beta_nonnegative<'a>(
    beta: impl IntoIterator<Item = &'a f64>,
    context: &str,
) -> Result<(), String> {
    for (idx, &value) in beta.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{context} coefficient {idx} is non-finite"));
        }
        if value < -1e-12 {
            return Err(format!(
                "{context} coefficient {idx} is negative ({value:.3e}); monotone wiggle coefficients must be non-negative"
            ));
        }
    }
    Ok(())
}

/// Slack tolerance for the `beta >= 0` monotone-wiggle inequality constraints.
///
/// The constrained inner Newton/QP holds a binding coordinate at the boundary
/// only up to its own KKT tolerance, so an accepted step can leave the active
/// coordinate a few ULPs below zero (e.g. `-2e-9`). That is feasibility within
/// the solver tolerance, not a genuine sign violation, so the post-update hook
/// projects such coordinates back onto the non-negative cone (clamps them to
/// exactly `0`) rather than failing the fit. The band matches the constrained
/// blockwise solver's KKT tolerances (`1e-6 * scale + 1e-10`,
/// `1e-10 * (1 + scale)`); anything more negative survives the projection and
/// is rejected by [`validate_monotone_wiggle_beta_nonnegative`].
pub(crate) const MONOTONE_WIGGLE_ACTIVE_SET_TOL: f64 = 1e-6;

/// Project a monotone-wiggle coefficient vector onto the non-negative cone the
/// `beta >= 0` constraints define, clamping coordinates the constrained solve
/// left slightly negative (within [`MONOTONE_WIGGLE_ACTIVE_SET_TOL`]) to exactly
/// `0`. Coordinates more negative than the tolerance are left untouched so the
/// subsequent [`validate_monotone_wiggle_beta_nonnegative`] still rejects
/// genuine sign violations.
pub(crate) fn project_monotone_wiggle_beta_nonnegative(mut beta: Array1<f64>) -> Array1<f64> {
    for value in beta.iter_mut() {
        if *value < 0.0 && *value >= -MONOTONE_WIGGLE_ACTIVE_SET_TOL {
            *value = 0.0;
        }
    }
    beta
}

/// Resolve a requested wiggle penalty-order set into:
///
/// - the primary derivative order used by the monotone I-spline function
///   roughness, and
/// - the remaining function-derivative orders to append on the same basis.
///
/// The primary order is the smallest requested order. If the list is empty,
/// `default_primary` is used. Zero is never silently dropped: it is not a
/// roughness derivative and is therefore a typed configuration error. Extra
/// orders are returned in original order, deduplicated, and exclude primary.
pub fn split_wiggle_penalty_orders(
    default_primary: usize,
    penalty_orders: &[usize],
) -> Result<(usize, Vec<usize>), String> {
    if default_primary == 0 {
        return Err("default wiggle penalty derivative order must be positive".to_string());
    }
    if penalty_orders.contains(&0) {
        return Err("wiggle penalty derivative orders must all be positive".to_string());
    }
    let primary_order = penalty_orders
        .iter()
        .copied()
        .min()
        .unwrap_or(default_primary);
    let mut extras = Vec::new();
    for &order in penalty_orders {
        if order == primary_order || extras.contains(&order) {
            continue;
        }
        extras.push(order);
    }
    Ok((primary_order, extras))
}

pub(crate) fn select_wiggle_basis_from_seed(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
    penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    select_wiggle_basis_from_seed_with_knots(seed, cfg, penalty_orders, WarpKnotEnds::Clamped)
}

/// Which end condition the warp's knot vector carries.
///
/// A boundary knot of multiplicity `degree + 1` makes the ramp `C^{-1}` there:
/// `I'` steps from `0` outside to its interior one-sided value inside, and `I''`
/// steps by `2, 6, 12, 20` at degree `2, 3, 4, 5`. That step reaches the inner
/// objective through `H` (gam#2695), so a warp the objective differentiates
/// needs [`WarpKnotEnds::Simple`].
///
/// [`WarpKnotEnds::Clamped`] is not a second opinion about the mathematics — it
/// is where a subsystem's SAVED-MODEL runtime still reconstructs its deviation
/// on the clamped convention (the BMS anchored-cubic replay and the
/// marginal-slope deviation runtime both do) and has to move to the ramp
/// definition before its knots can. Each one is its own piece of work; doing it
/// half-way would leave a fit and its replay reading different functions, which
/// is the fault this issue is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WarpKnotEnds {
    Clamped,
    Simple,
}

pub(crate) fn select_wiggle_basis_from_seed_with_knots(
    seed: ArrayView1<'_, f64>,
    cfg: &WiggleBlockConfig,
    penalty_orders: &[usize],
    ends: WarpKnotEnds,
) -> Result<SelectedWiggleBasis, String> {
    let (primary_order, extra_orders) =
        split_wiggle_penalty_orders(cfg.penalty_order, penalty_orders)?;
    let mut derivative_orders = Vec::with_capacity(1 + extra_orders.len());
    derivative_orders.push(primary_order);
    derivative_orders.extend(extra_orders);
    // REALISED DEGREE (gam#2695). A simple-ended warp is one the inner objective
    // COMPOSES rather than evaluates, so its basis has to be continuous to
    // `COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER` for that
    // objective to be C¹ in β. The floor is tied to `Simple` ends and not applied to
    // `Clamped` ones deliberately: at a boundary knot of multiplicity `degree + 1`
    // the ramp is `C^{-1}` at EVERY degree, so raising the degree there buys
    // nothing — a clamped composed warp becomes admissible by moving its ends
    // first, which is the work `WarpKnotEnds` already names and tracks.
    let degree = match ends {
        WarpKnotEnds::Simple => {
            let minimum = composed_warp_minimum_degree();
            if cfg.degree < minimum {
                log::info!(
                    "[warp-degree] composed monotone warp requested degree {} and is built at \
                     {minimum}: the inner objective reads the basis's derivative of order {} \
                     (H is the order-2 coefficient of the row jet and reaches the basis through \
                     m1 = 1 + sum betaw_j B'_j, i.e. the tower SHIFTED BY ONE; Phi = 1/2 sum \
                     g(lambda(Z_J^T H Z_J)) is a term of the objective, and its gradient reads \
                     the basis's FOURTH derivative), while a degree-d I-spline is only C^(d-1) \
                     at a simple knot — so degree {} leaves the analytic gradient discontinuous \
                     across every knot the index crosses (gam#2695)",
                    cfg.degree,
                    COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER,
                    cfg.degree,
                );
                minimum
            } else {
                cfg.degree
            }
        }
        WarpKnotEnds::Clamped => cfg.degree,
    };
    let knots = match ends {
        WarpKnotEnds::Simple => {
            monotone_warp_knots_from_seed(seed, degree, cfg.num_internal_knots)?
        }
        WarpKnotEnds::Clamped => {
            gam_terms::basis::initializewiggle_knots_from_seed(seed, degree, cfg.num_internal_knots)?
        }
    };
    let canonical =
        canonical_wiggle_function_penalties(&knots, degree, &derivative_orders, cfg.double_penalty)?;
    let block = buildwiggle_block_input_from_canonical_penalties(seed, &knots, degree, &canonical)?;
    Ok(SelectedWiggleBasis {
        knots,
        degree,
        block,
        penalty_metadata: canonical.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_types::PenaltySpec;
    use gam_terms::basis::initializewiggle_knots_from_seed;
    use ndarray::Array1;

    fn dense_penalty(spec: &PenaltySpec) -> &Array2<f64> {
        match spec {
            PenaltySpec::Dense(m) => m,
            other => panic!("expected Dense penalty, got {other:?}"),
        }
    }

    fn is_symmetric(m: &Array2<f64>) -> bool {
        let n = m.nrows();
        if m.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                if (m[[i, j]] - m[[j, i]]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    // ---- monotone_wiggle_internal_degree ----

    #[test]
    fn internal_degree_rejects_degree_below_two() {
        // degree 0 and 1 yield internal_degree < 1 -> Err with the documented message.
        for d in [0usize, 1] {
            let err = monotone_wiggle_internal_degree(d).unwrap_err();
            assert_eq!(err, "monotone wiggle degree must be >= 2");
        }
    }

    #[test]
    fn internal_degree_is_degree_minus_one_for_valid_degrees() {
        // degree >= 2 -> Ok(degree - 1): the per-span value degree aligned to the
        // public value-basis degree.
        assert_eq!(monotone_wiggle_internal_degree(2).unwrap(), 1);
        assert_eq!(monotone_wiggle_internal_degree(3).unwrap(), 2);
        assert_eq!(monotone_wiggle_internal_degree(10).unwrap(), 9);
    }

    // ---- buildwiggle_block_input_from_knots (driven via seed for valid knots) ----

    fn build(double_penalty: bool, penalty_order: usize) -> (ParameterBlockInput, usize) {
        // A spread-out seed so knot generation yields several monotone columns.
        let seed = Array1::linspace(0.0, 1.0, 40);
        let cfg = WiggleBlockConfig {
            degree: 3,
            num_internal_knots: 5,
            penalty_order,
            double_penalty,
        };
        let knots =
            initializewiggle_knots_from_seed(seed.view(), cfg.degree, cfg.num_internal_knots)
                .expect("knot init");
        let block = buildwiggle_block_input_from_knots(
            seed.view(),
            &knots,
            cfg.degree,
            cfg.penalty_order,
            cfg.double_penalty,
        )
        .expect("build block");
        let p = block.design.ncols();
        (block, p)
    }

    #[test]
    fn single_penalty_block_shapes_and_invariants() {
        let (block, p) = build(false, 2);
        assert!(p >= 2, "expected multiple monotone columns, got p={p}");
        // Offset is zeros with length = seed length.
        assert_eq!(block.offset.len(), 40);
        assert!(block.offset.iter().all(|&v| v == 0.0));
        // initial_beta is Some(zeros(p)).
        let beta = block.initial_beta.as_ref().expect("initial_beta");
        assert_eq!(beta.len(), p);
        assert!(beta.iter().all(|&v| v == 0.0));
        // One ROUGHNESS penalty, plus the unconditional gauge-closure
        // coordinate (gam#2647): an order-two roughness leaves the linear warp
        // free, and the linear warp is the index scale, not a shape. This
        // assertion used to read `== 1`, which is exactly the shape of the
        // defect — a warp block shipped with an unpenalized reparameterization
        // direction, so the penalized criterion had no minimiser.
        assert_eq!(block.penalties.len(), 2);
        assert_eq!(block.nullspace_dims.len(), 2);
        // The exact function-derivative Gram is p x p and symmetric.
        let s = dense_penalty(&block.penalties[0]);
        assert_eq!(s.dim(), (p, p));
        assert!(is_symmetric(s));
        // The anchored I-spline excludes the constant polynomial, so the
        // order-two derivative null space contains only the linear direction.
        assert_eq!(block.nullspace_dims[0], 1);
        // The closure coordinate penalizes a null space of its own dimension 0.
        assert_eq!(block.nullspace_dims[1], 0);
    }

    /// Smallest generalized eigenvalue of `Σ_j S_j` against the I-spline
    /// function Gram, relative to the largest — i.e. how close the assembled
    /// penalty set comes to leaving a whole function direction free.
    fn relative_joint_nullity_margin(
        knots: &Array1<f64>,
        degree: usize,
        orders: &[usize],
        double_penalty: bool,
    ) -> f64 {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let canonical = canonical_wiggle_function_penalties(knots, degree, orders, double_penalty)
            .expect("canonical wiggle penalties");
        let dim = canonical.matrices[0].nrows();
        // Per-block normalization, matching the closure. The fitted penalty is
        // `Σ_j λ_j S_j` with each `λ_j > 0` chosen independently by REML, so the
        // set leaves a direction free iff EVERY block does — `⋂_j null(S_j)`,
        // which is weight-independent. Summing the raw matrices asks a different
        // and wrong question: an order-4 roughness dominates an order-1 one by
        // five orders on these knots, so the raw sum reports as "free" every
        // direction that is merely penalized much more weakly than the stiffest
        // block, and the shrinkage coordinate — whose scale is the function
        // Gram, not a high derivative — looks like nothing next to it.
        let mut joint = Array2::<f64>::zeros((dim, dim));
        for matrix in &canonical.matrices {
            let mean_diagonal =
                (0..dim).map(|i| matrix[[i, i]].abs()).sum::<f64>() / dim as f64;
            if !(mean_diagonal > 0.0) || !mean_diagonal.is_finite() {
                continue;
            }
            joint.scaled_add(1.0 / mean_diagonal, matrix);
        }
        let internal_degree = monotone_wiggle_internal_degree(degree).expect("internal degree");
        let gram = gam_terms::basis::ispline_function_gram(knots.view(), internal_degree)
            .expect("I-spline function Gram");
        // Whiten by the function metric so the ratio is a statement about
        // FUNCTIONS, not about the coefficient chart: `G^{-1/2} S G^{-1/2}`.
        let (gvals, gvecs) = gram.eigh(Side::Lower).expect("Gram eigh");
        let gmax = gvals.iter().copied().fold(0.0_f64, f64::max);
        let mut g_inv_sqrt = Array2::<f64>::zeros((dim, dim));
        for k in 0..dim {
            let lam = gvals[k].max(1e-14 * gmax);
            let scale = 1.0 / lam.sqrt();
            let vk = gvecs.column(k);
            for i in 0..dim {
                for j in 0..dim {
                    g_inv_sqrt[[i, j]] += scale * vk[i] * vk[j];
                }
            }
        }
        let whitened = g_inv_sqrt.dot(&joint).dot(&g_inv_sqrt);
        let (svals, _) = whitened.eigh(Side::Lower).expect("whitened penalty eigh");
        let hi = svals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lo = svals.iter().copied().fold(f64::INFINITY, f64::min);
        lo / hi.max(f64::MIN_POSITIVE)
    }

    /// gam#2647, stated as the invariant rather than as one fixture.
    ///
    /// A monotone link warp is composed onto a free index, so any function
    /// direction the assembled penalty set leaves unpenalized is a
    /// reparameterization the index can absorb for free — the penalized
    /// criterion then has no minimiser and the inner solve diverges along the
    /// orbit. The invariant is therefore: **for every configuration this crate
    /// can emit, the assembled set has a trivial joint null space.** Checked in
    /// the function metric so the verdict cannot be manufactured by a coefficient
    /// rescale.
    ///
    /// Before the fix, every `double_penalty = false` row here (and every
    /// `[2]`/`[2,3]` row regardless) had a joint null space of dimension ≥ 1,
    /// i.e. a margin at machine zero.
    #[test]
    fn wiggle_penalty_set_always_closes_its_own_null_space_2647() {
        let seed = Array1::linspace(0.0, 1.0, 60);
        for degree in [2usize, 3, 4] {
            let knots = initializewiggle_knots_from_seed(seed.view(), degree, 5)
                .expect("knot init for the invariant sweep");
            let max_order = degree; // value degree = degree, so order <= degree is represented
            let mut order_sets: Vec<Vec<usize>> = Vec::new();
            for primary in 1..=max_order {
                order_sets.push(vec![primary]);
            }
            if max_order >= 2 {
                order_sets.push((1..=max_order).collect());
                order_sets.push((2..=max_order).collect());
            }
            for orders in &order_sets {
                for double_penalty in [false, true] {
                    let margin =
                        relative_joint_nullity_margin(&knots, degree, orders, double_penalty);
                    assert!(
                        margin > 1e-10,
                        "degree {degree}, orders {orders:?}, double_penalty={double_penalty}: \
                         the assembled wiggle penalty set leaves a function direction free \
                         (smallest/largest whitened penalty eigenvalue = {margin:.6e}). That \
                         direction is a reparameterization of the index the warp is composed \
                         onto, so the penalized criterion is unbounded below along it (gam#2647)."
                    );
                }
            }
        }
    }

    /// The gauge closure must be **invisible** to every configuration that was
    /// already well posed — most of all the shipped default.
    ///
    /// `WigglePenaltyConfig::cubic_triple_operator_default` is `degree = 3`,
    /// `orders = [1, 2, 3]`, `double_penalty = true`. Order one is full rank on
    /// the anchored basis (`roughness_nullspace_dim = order − 1 = 0`), so that
    /// set already leaves nothing free and the closure must append nothing: the
    /// emitted topology has to stay exactly three roughness blocks, in order,
    /// with no shrinkage coordinate anywhere. Asserted on the topology rather
    /// than on a count so a coordinate appearing in the middle is caught too.
    ///
    /// This is the assertion that would fail if the joint-null test were done
    /// on the RAW sum instead of on the per-block-normalized one: the order-3
    /// roughness dominates the order-1 roughness by orders of magnitude on these
    /// knots, and an unnormalized sum reports a null space the set does not have.
    #[test]
    fn shipped_default_wiggle_topology_is_untouched_by_the_gauge_closure_2647() {
        let seed = Array1::linspace(0.0, 1.0, 60);
        let cfg = gam_spec::WigglePenaltyConfig::cubic_triple_operator_default();
        let knots = initializewiggle_knots_from_seed(seed.view(), cfg.degree, cfg.num_internal_knots)
            .expect("knot init for the shipped default");
        let canonical = canonical_wiggle_function_penalties(
            &knots,
            cfg.degree,
            &cfg.penalty_orders,
            cfg.double_penalty,
        )
        .expect("shipped-default canonical penalties");
        assert_eq!(
            canonical.metadata.blocks,
            vec![
                WigglePenaltyBlockKind::Roughness {
                    derivative_order: 1
                },
                WigglePenaltyBlockKind::Roughness {
                    derivative_order: 2
                },
                WigglePenaltyBlockKind::Roughness {
                    derivative_order: 3
                },
            ],
            "the gauge closure changed the shipped-default wiggle penalty topology"
        );
        assert_eq!(canonical.matrices.len(), 3);
        assert_eq!(canonical.nullspace_dims, vec![0, 1, 2]);
    }

    /// The concrete gauge, named: an order-two roughness leaves the LINEAR warp
    /// free, and the linear warp is exactly the index rescale
    /// `(β_index, β_w) ↦ (β_index/s, β_w + (s−1)ℓ)`. The closure must charge for
    /// it while the roughness alone does not.
    #[test]
    fn linear_warp_direction_is_free_under_roughness_and_charged_after_closure_2647() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        let seed = Array1::linspace(0.0, 1.0, 60);
        let degree = 3usize;
        let knots = initializewiggle_knots_from_seed(seed.view(), degree, 5).expect("knot init");
        let canonical = canonical_wiggle_function_penalties(&knots, degree, &[2], false)
            .expect("order-two canonical set");
        assert_eq!(
            canonical.matrices.len(),
            2,
            "order-two roughness must be accompanied by its gauge closure"
        );
        let roughness = &canonical.matrices[0];
        let closure = &canonical.matrices[1];
        let dim = roughness.nrows();

        // ℓ: the coefficient vector of the linear warp, recovered as the
        // roughness null direction (the anchored basis excludes constants, so
        // the order-two null space is exactly the linear ramp).
        let (rvals, rvecs) = roughness.eigh(Side::Lower).expect("roughness eigh");
        let rmax = rvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut k_min = 0usize;
        for k in 0..dim {
            if rvals[k] < rvals[k_min] {
                k_min = k;
            }
        }
        let ell = rvecs.column(k_min).to_owned();
        let rough_energy = ell.dot(&roughness.dot(&ell));
        let closure_energy = ell.dot(&closure.dot(&ell));
        assert!(
            rough_energy <= 1e-10 * rmax,
            "the linear warp must be free under an order-two roughness: ℓᵀSℓ = {rough_energy:.6e} \
             against λ_max = {rmax:.6e}"
        );
        assert!(
            closure_energy > 1e-8 * rmax.max(1.0),
            "the gauge closure must charge for the linear warp: ℓᵀRℓ = {closure_energy:.6e}"
        );
        // And the free direction must be the one it charges MOST. Exact
        // complementarity is deliberately not asserted: the shrinkage is
        // `(G Z)(G Z)ᵀ` with `Z` spanning `null(S)` in the FUNCTION metric, so
        // it annihilates the metric-generalized eigenvectors of `(S, G)` — not
        // the ordinary coefficient-space eigenvectors of `S` used here, which
        // are not `G`-orthogonal to `Z`. Measured on this fixture the null
        // direction carries 2.35 against a worst range direction of 0.449, and
        // demanding zero there would be asserting a property this construction
        // (the one the `double_penalty` path has always used) does not have.
        let range_energy = (0..dim)
            .filter(|&k| rvals[k] > 1e-8 * rmax)
            .map(|k| {
                let v = rvecs.column(k);
                v.dot(&closure.dot(&v)).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            closure_energy > range_energy,
            "the closure must charge the FREE direction more than any direction the roughness \
             already penalizes: null energy {closure_energy:.6e} against max range energy \
             {range_energy:.6e}"
        );
    }

    #[test]
    fn double_penalty_appends_nullspace_only_function_ridge() {
        let (block, p) = build(true, 2);
        assert!(p >= 2);
        // Order two has one structural null direction, so double penalty emits
        // one separate function-space shrinkage block.
        assert_eq!(block.penalties.len(), 2);
        assert_eq!(block.nullspace_dims.len(), 2);
        let ridge = dense_penalty(&block.penalties[1]);
        assert_eq!(ridge.dim(), (p, p));
        assert!(is_symmetric(ridge));
        assert!(
            (0..p).any(|i| (0..p).any(|j| i != j && ridge[[i, j]].abs() > 1e-12)),
            "function-metric null shrinkage must not collapse to eye(p)"
        );
        assert_eq!(block.nullspace_dims[1], 0);
    }

    #[test]
    fn order_one_has_no_nullspace_ridge() {
        let (block, _) = build(true, 1);
        assert_eq!(block.penalties.len(), 1);
        assert_eq!(block.nullspace_dims, vec![0]);
    }

    #[test]
    fn unsupported_derivative_order_is_rejected_not_clamped() {
        let seed = Array1::linspace(0.0, 1.0, 40);
        let knots = initializewiggle_knots_from_seed(seed.view(), 3, 5).expect("knot init");
        let error = match buildwiggle_block_input_from_knots(seed.view(), &knots, 3, 4, false) {
            Ok(_) => panic!("order above represented value degree must be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("derivative"), "unexpected error: {error}");
    }

    #[test]
    fn explicit_zero_penalty_order_is_rejected() {
        let error = split_wiggle_penalty_orders(2, &[0, 2]).unwrap_err();
        assert_eq!(
            error,
            "wiggle penalty derivative orders must all be positive"
        );
    }
}

/// gam#2695 — the monotone warp basis is ONE `C^{degree−1}` function on `ℝ`.
///
/// The composed warp `q = q₀ + Σ_j βw_j·I_j(q₀)` is differentiated three times
/// by the joint-Newton machinery — `ℓ` reads `w′`, `∇ℓ` reads `w″`, and `H`
/// (which the Firth value `Φ = ½Σ g(λ(Z_JᵀHZ_J))` is a function of, inside the
/// accept test) reads `w‴` — in a state where `q₀` moves with β while the knots
/// stay where the seed put them. A basis that STEPS at any of those orders puts
/// that step into the objective, and `actual/predicted` then cannot approach
/// `1` at any step size.
///
/// These pins are stated on the basis itself, where the contract lives, rather
/// than on the fit that exposed it.
#[cfg(test)]
mod warp_basis_smoothness_2695_tests {
    use super::*;
    use gam_terms::basis::monotone_warp_knots;
    use ndarray::Array1;

    const DEGREE: usize = 2;

    /// The shipped `linkwiggle(internal_knots=2)` shape over `[-1, 2]`, built
    /// by the WARP generator: simple knots throughout, the grid continued by
    /// `degree` spans at each end. Degree 2 is deliberate here — this module
    /// pins the BASIS, whose contract is `C^{degree−1}`; whether a COMPOSED
    /// warp may be built at that degree is a separate question, and the answer
    /// is measured on the FIT rather than asserted here — see
    /// `a_degree_two_composed_warp_steps_even_on_the_warp_knot_vector_2695`.
    fn knots() -> Array1<f64> {
        monotone_warp_knots(-1.0, 2.0, DEGREE, 2).expect("warp knots")
    }

    fn basis_at(x: f64, order: usize) -> Array1<f64> {
        let seed = Array1::from_elem(1, x);
        monotone_wiggle_basis_with_derivative_order(seed.view(), &knots(), DEGREE, order)
            .expect("warp basis")
            .row(0)
            .to_owned()
    }

    /// The contract, at every knot and every order the objective reads.
    ///
    /// Scale-free: the one-sided gap must FALL with the step. A step leaves it
    /// flat, which is what the clamped vector does at order 2 — see
    /// `gam_terms::basis::ispline_ramp::tests::the_clamped_vector_steps_at_order_two_at_every_degree`
    /// for that negative control.
    #[test]
    fn every_derivative_below_the_degree_is_continuous_at_every_knot() {
        let knots = knots();
        for order in 0..DEGREE {
            for &knot in knots.iter() {
                let gap = |h: f64| -> f64 {
                    let lo = basis_at(knot - h, order);
                    let hi = basis_at(knot + h, order);
                    lo.iter()
                        .zip(hi.iter())
                        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
                };
                let coarse = gap(1.0e-3);
                let fine = gap(1.0e-6);
                assert!(
                    fine <= coarse / 100.0 + 1.0e-12,
                    "order {order} steps at knot {knot}: gap {coarse:.3e} at h=1e-3 and \
                     {fine:.3e} at h=1e-6, a ratio of {:.2e} against the 1000x a continuous \
                     derivative must give",
                    coarse / fine.max(f64::MIN_POSITIVE),
                );
            }
        }
    }

    /// Non-vacuity for the pin above: the basis MOVES across those knots, so
    /// "continuous" is not being satisfied by a table of zeros.
    #[test]
    fn the_basis_moves_materially_across_its_own_knots() {
        let knots = knots();
        let low = basis_at(knots[0] - 1.0, 0);
        let high = basis_at(knots[knots.len() - 1] + 1.0, 0);
        for j in 0..low.len() {
            assert!(
                (high[j] - low[j]).abs() > 0.5,
                "column {j} must traverse its whole ramp across the knot vector; got \
                 {:.3e} -> {:.3e}",
                low[j],
                high[j],
            );
        }
    }

    /// Value and derivative are one function: a central difference of the
    /// VALUE reproduces the reported DERIVATIVE, inside the knots and outside.
    #[test]
    fn the_value_and_its_reported_derivative_are_one_function() {
        let cbrt_eps = f64::EPSILON.cbrt();
        for &x in &[-6.0_f64, -2.5, -1.0, -0.25, 0.5, 1.25, 2.0, 3.0, 7.0] {
            let h = cbrt_eps * (1.0 + x.abs());
            let analytic = basis_at(x, 1);
            let plus = basis_at(x + h, 0);
            let minus = basis_at(x - h, 0);
            for j in 0..analytic.len() {
                // A degree-2 warp is `C¹` but not `C²`, so a difference
                // straddling a knot is only first-order accurate; keep the
                // bound at the straddling accuracy rather than the interior one.
                let fd = (plus[j] - minus[j]) / (2.0 * h);
                let tol = 1.0e-4 * (1.0 + analytic[j].abs());
                assert!(
                    (fd - analytic[j]).abs() <= tol,
                    "at x={x} column {j}: analytic I' = {:.9e} but a central difference of \
                     the SAME basis's value is {fd:.9e}",
                    analytic[j],
                );
            }
        }
    }

    /// The warp stays a warp: every column is non-decreasing on all of `ℝ`, so
    /// `w = Σ βw_j I_j` with `βw ≥ 0` is monotone everywhere.
    #[test]
    fn every_column_is_non_decreasing_on_the_whole_line() {
        let grid = Array1::linspace(-8.0, 9.0, 341);
        let values = monotone_wiggle_basis_with_derivative_order(grid.view(), &knots(), DEGREE, 0)
            .expect("warp values");
        let slopes = monotone_wiggle_basis_with_derivative_order(grid.view(), &knots(), DEGREE, 1)
            .expect("warp slopes");
        for j in 0..values.ncols() {
            for i in 1..values.nrows() {
                assert!(
                    values[[i, j]] >= values[[i - 1, j]] - 1.0e-12,
                    "column {j} decreases between x={} and x={}: {} -> {}",
                    grid[i - 1],
                    grid[i],
                    values[[i - 1, j]],
                    values[[i, j]],
                );
            }
            for i in 0..slopes.nrows() {
                assert!(
                    slopes[[i, j]] >= -1.0e-12,
                    "column {j} has a negative slope {} at x={}",
                    slopes[[i, j]],
                    grid[i],
                );
            }
        }
    }

    /// Outside its own support a ramp is EXACTLY `0` or `1`, with every
    /// derivative exactly zero — the property that lets the warp be extended to
    /// all of `ℝ` with no tail convention at all.
    #[test]
    fn each_column_is_flat_outside_its_own_support() {
        let knots = knots();
        for far in [knots[0] - 3.0, knots[knots.len() - 1] + 3.0] {
            let value = basis_at(far, 0);
            for order in 1..=3usize {
                let derivative = basis_at(far, order);
                for j in 0..derivative.len() {
                    assert_eq!(
                        derivative[j], 0.0,
                        "column {j} order {order} at x={far} must be exactly zero"
                    );
                }
            }
            for j in 0..value.len() {
                assert!(
                    value[j] == 0.0 || value[j] == 1.0,
                    "column {j} at x={far} must be exactly 0 or 1, got {}",
                    value[j]
                );
            }
        }
    }
}

/// gam#2695 — a COMPOSED warp is built at the degree its own objective requires.
///
/// The basis module above pins `C^{degree−1}`, a property of the basis. This
/// module pins the other half: which degree a warp the objective COMPOSES is
/// allowed to be built at, and that the answer is applied by construction rather
/// than by refusal.
#[cfg(test)]
mod composed_warp_degree_2695_tests {
    use super::*;
    use ndarray::Array1;

    fn seed() -> Array1<f64> {
        Array1::from_shape_fn(24, |i| -1.5 + 3.0 * (i as f64) / 23.0)
    }

    fn cfg(degree: usize) -> WiggleBlockConfig {
        WiggleBlockConfig {
            degree,
            num_internal_knots: 2,
            penalty_order: 2,
            double_penalty: false,
        }
    }

    /// The floor is derived from two stated facts, so it must equal their
    /// composition rather than a literal that happens to agree today.
    #[test]
    fn the_minimum_degree_is_the_required_continuity_order_plus_the_spline_loss() {
        assert_eq!(
            composed_warp_minimum_degree(),
            COMPOSED_WARP_REQUIRED_CONTINUOUS_BASIS_DERIVATIVE_ORDER + 1,
            "a degree-d I-spline is C^(d-1) at a simple knot, so continuity to order k \
             needs d >= k + 1"
        );
        // Non-vacuity: the floor has to bind on the shipped witness, which asks
        // for `linkwiggle(degree=2, internal_knots=2)`.
        assert!(
            composed_warp_minimum_degree() > 2,
            "the floor must exclude degree 2 or the witness this issue is named for is \
             unaffected by it"
        );
    }

    /// A simple-ended (composed) warp asked for a degree below the floor is
    /// BUILT at the floor, and every object it returns carries the realised
    /// degree — the knot vector's span count, the design's column count, and the
    /// degree the block reports to the saved model.
    #[test]
    fn a_composed_warp_below_the_floor_is_built_at_the_floor() {
        let minimum = composed_warp_minimum_degree();
        let seed = seed();
        for requested in 2..=minimum {
            let selected = select_wiggle_basis_from_seed_with_knots(
                seed.view(),
                &cfg(requested),
                &[2],
                WarpKnotEnds::Simple,
            )
            .expect("composed warp basis");
            assert_eq!(
                selected.degree,
                requested.max(minimum),
                "requested degree {requested}"
            );
            // `monotone_warp_knots` emits `num_internal_knots + 1` spans across
            // the range and continues the grid by `degree` spans at each end,
            // all simple — `num_internal_knots + 2 + 2·degree` knots. So the
            // knot count is a direct readout of the degree the basis was
            // actually built at.
            assert_eq!(
                selected.knots.len(),
                2 + 2 + 2 * selected.degree,
                "knot count must follow the REALISED degree, not the requested one"
            );
            // Column count is `num_internal_knots + degree` for this generator.
            assert_eq!(
                selected.block.design.ncols(),
                2 + selected.degree,
                "the design must be the realised degree's basis"
            );
        }
    }

    /// Above the floor nothing moves: the realised degree is the requested one.
    #[test]
    fn a_composed_warp_at_or_above_the_floor_is_untouched() {
        let seed = seed();
        for requested in composed_warp_minimum_degree()..=composed_warp_minimum_degree() + 2 {
            let selected = select_wiggle_basis_from_seed_with_knots(
                seed.view(),
                &cfg(requested),
                &[2],
                WarpKnotEnds::Simple,
            )
            .expect("composed warp basis");
            assert_eq!(selected.degree, requested);
        }
    }

    /// The floor is tied to SIMPLE ends and must not reach the clamped
    /// subsystems: at a boundary knot of multiplicity `degree + 1` the ramp is
    /// `C^{-1}` at every degree, so raising the degree there would change those
    /// fits while fixing nothing. Their route to admissibility is moving their
    /// ends, which `WarpKnotEnds` already names.
    #[test]
    fn a_clamped_warp_keeps_the_degree_it_asked_for() {
        let seed = seed();
        let selected =
            select_wiggle_basis_from_seed_with_knots(seed.view(), &cfg(2), &[2], WarpKnotEnds::Clamped)
                .expect("clamped wiggle basis");
        assert_eq!(
            selected.degree, 2,
            "the composed-warp floor must not silently re-shape a clamped block"
        );
    }
}
