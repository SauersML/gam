#![cfg(test)]
//! #4077 — finite-difference receipts for the half-line mass's DERIVATIVE channels.
//!
//! `pinned_half_line_channels` hands every exact-`A` route three legs: the θ-adjoint
//! (σ-leg through the `PinnedSlotWeight` adjoint + gradient-leg rows
//! `2κ₁·sign(g)·[A_raw]_u`), the ρ curvature trace, and — when crosscoder pricing
//! spans are installed — the #2231 span probe. The maintainer's suite FDs the ARD
//! ρ-partial end to end (`pinned_half_line_mass_enters_the_outer_gradient_4077`),
//! which contracts the θ-adjoint along ONE response direction. What it never
//! measures, and these tests do:
//!
//! 1. the θ-adjoint AS A VECTOR: `Σ_u c_u` displaced along an arbitrary inner
//!    direction `d` (zero on the pinned slots, where production responses are
//!    zero) must differentiate to `⟨gamma, d⟩` in FULL `log|A|` units — the
//!    module's own units convention, verified rather than trusted;
//! 2. the per-slot Jacobian claim `d g_u = ⟨row_curvature, d_t⟩ +
//!    ⟨border_curvature, d_β⟩` at GENUINELY pinned slots (the operands test
//!    checks the rows against dense `A` at an interior slot named as pinned —
//!    equality with `A`, not the derivative of `g`);
//! 3. the #2231 crosscoder span leg: the probe scales the block's target columns
//!    by `1.5` and reads `σ(1.5) − σ(1)` as `∂σ/∂log λ`. Since `c = √λ` and
//!    `A_raw` is affine in the target, `[σ(1+δ/2) − σ(1−δ/2)]/(2δ)` must
//!    reproduce it to rounding — an independent measurement with different
//!    probe points. The block's implicit right-hand side entries at the pinned
//!    slots are checked the same way against `∂g_u/∂log λ`;
//! 4. the ASSEMBLY: the complete block-coordinate gradient (data channel +
//!    Jacobian `−n·p_ℓ/2` + `½tr(A⁺∂A/∂log λ_ℓ)` trace + implicit `−½Γᵀθ̂`
//!    adjoint, the last two carrying the half-line legs) against a central
//!    difference of the value `fit_at_fixed_rho` itself re-prices at
//!    `log λ = ±δ` — each arm through its own authoritative inner solve on the
//!    objective's own `√λ`-scaled target. The factor-wise #2231 receipt runs on
//!    a fixture WITHOUT pinned slots; this one runs where the half-line
//!    channels are live.

use super::*;
use crate::basis::EuclideanPatchEvaluator;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2, array, s};
use std::sync::Arc;

const LO: f64 = -1.0;
const HI: f64 = 1.0;

fn interval_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    interval_fixture_with_gram(|width| Array2::<f64>::eye(width))
}

/// #2822 — the same interval fixture at a caller-provided reference gram, so a
/// rank-deficient roughness can be installed without forking the generator.
fn interval_fixture_with_gram(
    gram: impl FnOnce(usize) -> Array2<f64>,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let p = 2usize;
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch basis"));
    // Generating latent: three rows sit beyond the interval's upper end, so the
    // fitted coordinates of those rows are pushed onto `hi` with `g < 0` there.
    let truth: [f64; 8] = [-0.7, -0.3, 0.1, 0.4, 1.6, 1.8, 2.0, 0.8];
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth[row].clamp(-0.9, 0.9));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("coords evaluate");
    let width = phi.ncols();
    let decoder = Array2::<f64>::from_shape_fn((width, p), |(b, o)| {
        [[0.1, -0.2], [1.0, 0.6], [0.2, -0.3]][b][o]
    });
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let s = truth[row];
        for o in 0..p {
            target[[row, o]] = decoder[[0, o]]
                + decoder[[1, o]] * s
                + decoder[[2, o]] * s * s
                + 0.03 * (1.7 * row as f64 + 0.9 * o as f64).sin();
        }
    }
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "interval".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        gram(width),
    )
    .expect("atom shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Interval { lo: LO, hi: HI }],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, 1.0, vec![array![-3.0]]);
    (term, target, rho)
}

/// The fixture converged at its seed `ρ`, the state every #4077 half-line check
/// reads (mirrors `converged_interval_state` in `tests_interval_active_bound_3438`).
fn converged_interval_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (mut term, target, rho) = interval_fixture();
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .expect("the criterion prices the interval fixture");
    (term, target, rho)
}

/// #2822 — the interval fixture with a rank-deficient reference gram whose
/// null space SURVIVES production's entry stages:
///
/// `prepare_entry_stages` (construction.rs) runs the #1117 data-supported
/// basis reduction before every criterion built over the supplied state; it
/// screens the weighted data Gram `G_k = Φ_kᵀ diag(a_k²) Φ_k` and
/// reparametrizes any atom whose Gram is rank-deficient onto the span above
/// `SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF`, contracting the penalty to `QᵀSQ`.
/// A planted null survives iff it lies in that span — and on this fixture the
/// Gram is numerically rank 2/3, with the data-null direction mixing every
/// basis axis, so an AXIS-ALIGNED planted null (span{1}, span{s}, span{s²}
/// were each tried) is generically absorbed and `QᵀSQ` comes back full rank:
/// the hand-derived `M_null` assert caught exactly that (the fold-in then
/// correctly prices `M_null = 0` for the reduced model — the receipt doing
/// its job).
///
/// So the null is planted on the TOP eigenvector of the production Gram
/// itself — the maximally data-supported direction, provably retained by any
/// cutoff-relative reduction — computed here through the SAME accumulator the
/// reduction reads (`empty_decoder_gram_accumulator` +
/// `accumulate_decoder_gram`), then `S = I − vvᵀ`. Exactly one null survives
/// whether the reduction fires (`2·(2−1)`) or not (`2·(3−2)`): the
/// hand-derived `M_null = r_k·1` is reduction-invariant.
fn rank_deficient_interval_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (probe, _, _) = interval_fixture_with_gram(|width| Array2::<f64>::eye(width));
    let mut grams = probe.empty_decoder_gram_accumulator();
    probe
        .accumulate_decoder_gram(&mut grams)
        .expect("the probe term accumulates its decoder gram");
    let mut gram = grams
        .first()
        .expect("the probe term carries the fixture atom")
        .clone();
    for i in 0..gram.nrows() {
        for j in 0..i {
            let sym = 0.5 * (gram[[i, j]] + gram[[j, i]]);
            gram[[i, j]] = sym;
            gram[[j, i]] = sym;
        }
    }
    let (evals, evecs) = gram
        .eigh(Side::Lower)
        .expect("the fixture's data gram eigendecomposes");
    let top = evals
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).expect("finite eigenvalues"))
        .expect("the data gram has a spectrum")
        .0;
    let v = evecs.column(top).to_owned();
    interval_fixture_with_gram(move |width| {
        let mut planted = Array2::<f64>::eye(width);
        for i in 0..width {
            for j in 0..width {
                planted[[i, j]] -= v[i] * v[j];
            }
        }
        planted
    })
}

/// The rank-deficient fixture converged at its seed `ρ` — the state the
/// assembly receipt below re-prices. The structural asserts pin the
/// hand-derivation: exactly the one planted null survives the entry stages,
/// and no decoder frame re-profiles the border (the derived `M_null = 2`
/// assumes `r_k = output_dim = 2`).
fn converged_rank_deficient_penalty_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (mut term, target, rho) = rank_deficient_interval_fixture();
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .expect("the criterion prices the rank-deficient interval fixture");
    let rank_s = SaeManifoldTerm::symmetric_rank(term.atoms[0].smooth_penalty())
        .expect("the fixture's penalty has a symmetric rank");
    let m_k = term.atoms[0].smooth_penalty().nrows();
    assert_eq!(
        rank_s + 1,
        m_k,
        "exactly the one planted null (the top data-Gram direction) may survive the entry stages"
    );
    assert_eq!(
        term.atoms[0].border_frame_rank(),
        2,
        "the decoder frame re-profiled the border; the hand-derived M_null = 2·1 assumes r_k = 2"
    );
    (term, target, rho)
}

/// #2822 — a TWO-ATOM interval fixture at caller-provided per-atom reference
/// grams, so the fold-in's per-atom SUM can be priced at distinct per-atom
/// null counts. Atom A carries the single-atom fixture's truth (rows 4–6
/// beyond `hi`); atom B mirrors it (rows 1–3 beyond), so BOTH atoms pin bound
/// slots at the converged state. Zero logits keep the softmax weights at the
/// canonicalizer's exact ½/½ (`canonicalize_softmax_logits` pins the last
/// column to 0), and the target is the atoms' exact additive mixture
/// `Σ_k a_k·Φ_k B_k` plus the same small sine perturbation.
///
/// The decoders are CHANNEL-SEPARATED (atom A's response lives on output 0,
/// atom B's on output 1) — a cosmetic-looking choice that is load-bearing:
/// with both decoders spread over both outputs, the rank-1-penalized atom B
/// chased the ½/½ mixture target, its coordinates collapsed mid-fit, and a
/// stage re-entering `prepare_entry_stages` snapshotted a numerically rank-2
/// data Gram — the #1117 reduction fired and PERMANENTLY contracted the atom
/// (wrapped evaluator: the contraction survives every basis refresh), taking
/// one planted null with it. Channel separation keeps each atom's projection
/// of the mixture on its own signal, the coordinates stay spread, the data
/// Gram stays rank 3, and no reduction fires.
fn two_atom_interval_fixture_with_grams(
    gram_a: impl FnOnce(usize) -> Array2<f64>,
    gram_b: impl FnOnce(usize) -> Array2<f64>,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 8usize;
    let p = 2usize;
    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("patch basis"));
    let truth_a: [f64; 8] = [-0.7, -0.3, 0.1, 0.4, 1.6, 1.8, 2.0, 0.8];
    let truth_b: [f64; 8] = [0.8, 2.0, 1.8, 1.6, 0.4, 0.1, -0.3, -0.7];
    let coords_a =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth_a[row].clamp(-0.9, 0.9));
    let coords_b =
        Array2::<f64>::from_shape_fn((n, 1), |(row, _)| 0.8 * truth_b[row].clamp(-0.9, 0.9));
    let (phi_a, jet_a) = evaluator
        .evaluate(coords_a.view())
        .expect("coords evaluate");
    let (phi_b, jet_b) = evaluator
        .evaluate(coords_b.view())
        .expect("coords evaluate");
    let width_a = phi_a.ncols();
    let width_b = phi_b.ncols();
    let decoder_a = Array2::<f64>::from_shape_fn((width_a, p), |(b, o)| {
        [[0.12, 0.01], [1.0, 0.03], [0.25, -0.02]][b][o]
    });
    let decoder_b = Array2::<f64>::from_shape_fn((width_b, p), |(b, o)| {
        [[0.02, -0.15], [-0.03, 0.7], [0.01, -0.3]][b][o]
    });
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let (s_a, s_b) = (truth_a[row], truth_b[row]);
        for o in 0..p {
            let decode = |d: &Array2<f64>, s: f64| d[[0, o]] + d[[1, o]] * s + d[[2, o]] * s * s;
            target[[row, o]] = 0.5 * decode(&decoder_a, s_a)
                + 0.5 * decode(&decoder_b, s_b)
                + 0.03 * (1.7 * row as f64 + 0.9 * o as f64).sin();
        }
    }
    let atom_a = SaeManifoldAtom::new_with_provided_function_gram(
        "interval_a".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi_a,
        jet_a,
        decoder_a,
        gram_a(width_a),
    )
    .expect("atom A shapes agree")
    .with_basis_second_jet(evaluator.clone());
    let atom_b = SaeManifoldAtom::new_with_provided_function_gram(
        "interval_b".to_string(),
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi_b,
        jet_b,
        decoder_b,
        gram_b(width_b),
    )
    .expect("atom B shapes agree")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 2)),
        vec![coords_a, coords_b],
        vec![
            LatentManifold::Interval { lo: LO, hi: HI },
            LatentManifold::Interval { lo: LO, hi: HI },
        ],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom_a, atom_b], assignment).expect("term");
    // One ARD entry per atom (`log_ard.len()` is the atom count the ρ
    // constructor broadcasts over).
    let rho = SaeManifoldRho::new(0.0, 1.0, vec![array![-3.0], array![-3.0]]);
    (term, target, rho)
}

/// #2822 — the two-atom fixture with rank-deficient penalties planted on each
/// atom's OWN data-supported directions: atom A nulls the TOP eigenvector of
/// its weighted Gram `G_A`, atom B the top TWO of `G_B`, each computed
/// through the same `accumulate_decoder_gram` accumulator the #1117
/// reduction reads (so the planted nulls provably survive it). The per-atom
/// fold-in then prices `M_null = r_A·1 + r_B·2 = 2 + 4 = 6` — a SUM no
/// single-atom install can produce, which is the point: with one atom,
/// `null_dims += …` inside the fold-in loop is indistinguishable from `=`,
/// and a dropped or double-counted atom passes. Distinct per-atom counts
/// (1 vs 2) also catch a swapped accumulation order.
fn two_atom_rank_deficient_interval_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let eye = |width: usize| Array2::<f64>::eye(width);
    // Converge an identity-gram probe FIRST: `prepare_entry_stages` seeds the
    // coords by decoder projection BEFORE the #1117 reduction screens each
    // atom's weighted Gram, so the data-supported span that decides a null's
    // survival lives at the FITTED state, not at the seed state the fixture
    // hands over — planting atom B's two-dim null plane on the SEED Gram lost
    // one null to a reduction firing at the moved state (the single-atom
    // fixture never hit this: its one planted null rides the top eigenvector,
    // the most rotation-robust direction). Planting on the converged probe's
    // Grams aligns each null space with the span the final reduction sees:
    // whether the reduction fires or not, exactly the planted count survives.
    let (mut probe, probe_target, probe_rho) = two_atom_interval_fixture_with_grams(eye, eye);
    probe
        .penalized_quasi_laplace_criterion_with_cache(
            probe_target.view(),
            &probe_rho,
            None,
            60,
            0.4,
            1.0e-8,
            1.0e-8,
        )
        .expect("the identity-gram probe converges before planting");
    let mut grams = probe.empty_decoder_gram_accumulator();
    probe
        .accumulate_decoder_gram(&mut grams)
        .expect("the probe term accumulates both atoms' decoder grams");
    assert_eq!(
        grams.len(),
        2,
        "the two-atom probe must carry one Gram accumulator per atom"
    );
    let symmetrized_eig = |gram: &Array2<f64>| -> (Array1<f64>, Array2<f64>) {
        let mut sym = gram.clone();
        for i in 0..sym.nrows() {
            for j in 0..i {
                let value = 0.5 * (sym[[i, j]] + sym[[j, i]]);
                sym[[i, j]] = value;
                sym[[j, i]] = value;
            }
        }
        sym.eigh(Side::Lower)
            .expect("the fixture's data gram eigendecomposes")
    };
    // Atom A: null the TOP eigenvector of its own Gram (one null).
    let (evals_a, evecs_a) = symmetrized_eig(&grams[0]);
    let top_a = evals_a
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).expect("finite eigenvalues"))
        .expect("the data gram has a spectrum")
        .0;
    let v_a = evecs_a.column(top_a).to_owned();
    // Atom B: null the top TWO eigenvectors of its own Gram (two nulls — a
    // count distinct from atom A's, so the SUM is unambiguous about which
    // atom contributed which term).
    let (evals_b, evecs_b) = symmetrized_eig(&grams[1]);
    let mut order_b: Vec<usize> = (0..evals_b.len()).collect();
    order_b.sort_by(|&x, &y| {
        evals_b[y]
            .partial_cmp(&evals_b[x])
            .expect("finite eigenvalues")
    });
    let v_b1 = evecs_b.column(order_b[0]).to_owned();
    let v_b2 = evecs_b.column(order_b[1]).to_owned();
    two_atom_interval_fixture_with_grams(
        move |width| {
            let mut planted = Array2::<f64>::eye(width);
            for i in 0..width {
                for j in 0..width {
                    planted[[i, j]] -= v_a[i] * v_a[j];
                }
            }
            planted
        },
        move |width| {
            let mut planted = Array2::<f64>::eye(width);
            for i in 0..width {
                for j in 0..width {
                    planted[[i, j]] -= v_b1[i] * v_b1[j] + v_b2[i] * v_b2[j];
                }
            }
            planted
        },
    )
}

/// The two-atom rank-deficient fixture converged at its seed ρ, with the
/// hand-derivation pinned PER ATOM: atom A retains exactly its one planted
/// null, atom B exactly its two, both at the unre-profiled border frame
/// `r_k = 2` — so the hand-derived sum is `M_null = 2·1 + 2·2 = 6`.
fn converged_two_atom_rank_deficient_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (mut term, target, rho) = two_atom_rank_deficient_interval_fixture();
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .expect("the criterion prices the two-atom rank-deficient interval fixture");
    for (k, planted) in [1usize, 2usize].iter().enumerate() {
        let rank_s = SaeManifoldTerm::symmetric_rank(term.atoms[k].smooth_penalty())
            .expect("the fixture's penalty has a symmetric rank");
        let m_k = term.atoms[k].smooth_penalty().nrows();
        assert_eq!(
            rank_s + planted,
            m_k,
            "atom {k}: exactly its {planted} planted null(s) may survive the entry stages"
        );
        assert_eq!(
            term.atoms[k].border_frame_rank(),
            2,
            "atom {k}: the decoder frame re-profiled the border; the hand-derived M_null = 2·1 + 2·2 assumes r_k = 2"
        );
    }
    (term, target, rho)
}

/// The fitted state's `B` system and its evidence factor.
fn fitted_cache(
    term: &mut SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let system = term
        .assemble_arrow_schur(target.view(), rho, None)
        .expect("the inner system assembles at the fitted state");
    let options = term.evidence_factor_options();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&system, 0.0, 0.0, &options)
        .expect("the evidence factor holds at the fitted state");
    cache
}

/// The term moved by `scale·d` over the inner coordinates `(t, β)` — one
/// coordinate per row for `t`, the basis-major decoder for `β` — with the basis
/// refreshed at the displaced coordinates, so every operand the half-line mass
/// reads moves with the state.
fn displaced(term: &SaeManifoldTerm, d: &SaeArrowVector, scale: f64) -> SaeManifoldTerm {
    let mut moved = term.clone();
    let coords = moved.assignment.coords[0].as_matrix();
    let coordinate_step =
        Array2::from_shape_vec(coords.dim(), d.t.to_vec()).expect("one coordinate per row");
    moved.assignment.coords[0] = LatentCoordValues::from_matrix_with_manifold(
        (&coords + &(coordinate_step * scale)).view(),
        LatentIdMode::None,
        LatentManifold::Interval { lo: LO, hi: HI },
    );
    let decoder = moved.atoms[0].decoder_coefficients().clone();
    let decoder_step =
        Array2::from_shape_vec(decoder.dim(), d.beta.to_vec()).expect("basis-major decoder");
    moved.atoms[0]
        .set_decoder_coefficients(&decoder + &(decoder_step * scale))
        .expect("the decoder keeps its shape");
    moved
        .refresh_basis_from_current_coords()
        .expect("the basis refreshes at the displaced coordinates");
    moved
}

/// The pinned slots' `Σ_u c_u` in full `log|A|` units, at the term's state.
fn half_line_mass(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    target: ArrayView2<'_, f64>,
    cache: &ArrowFactorCache,
) -> f64 {
    term.pinned_half_line_slots(rho, target, cache)
        .expect("half-line operands")
        .iter()
        .map(|slot| slot.log_det_correction())
        .sum()
}

/// #4077 — the θ-adjoint channel is the inner-coordinate derivative of the
/// half-line mass, in FULL `log|A|` units, with the gradient leg the raw
/// curvature rows and the σ-leg the weight-adjoint.
#[test]
fn half_line_theta_adjoint_is_the_inner_coordinate_derivative_4077() {
    let (mut term, target, rho) = converged_interval_state();
    let cache = fitted_cache(&mut term, &target, &rho);
    let slots = term
        .pinned_half_line_slots(&rho, target.view(), &cache)
        .expect("half-line operands");
    assert!(!slots.is_empty(), "no slot is pinned at the fitted state");
    let Some((_, gamma)) = term
        .pinned_half_line_channels(&rho, target.view(), &cache)
        .expect("the channels price at the fitted state")
    else {
        panic!("the fitted state pins slots, so the channels must be Some");
    };

    // A deterministic direction over `(t, β)`, ZERO on the pinned slots: the
    // production IFT response is projected to zero there, so the adjoint's
    // components on those coordinates are never contracted against motion.
    let pinned_rows: Vec<usize> = term
        .last_pinned_bound_slots
        .iter()
        .map(|&(row, _)| row)
        .collect();
    let mut d = SaeArrowVector {
        t: Array1::from_shape_fn(term.n_obs(), |row| {
            (pinned_rows.contains(&row))
                .then(|| 0.0)
                .unwrap_or(0.7 - 0.11 * row as f64)
        }),
        beta: Array1::from_shape_fn(cache.k, |j| 0.31 - 0.07 * j as f64),
    };
    // Normalise so `‖d‖_∞ = 1`: the step sizes below are in state units.
    let direction_scale =
        d.t.iter()
            .chain(d.beta.iter())
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    d.t.mapv_inplace(|value| value / direction_scale);
    d.beta.mapv_inplace(|value| value / direction_scale);

    let mass =
        |scale: f64| half_line_mass(&displaced(&term, &d, scale), &rho, target.view(), &cache);
    let operand = |scale: f64| {
        displaced(&term, &d, scale)
            .pinned_half_line_slots(&rho, target.view(), &cache)
            .expect("half-line operands at the displaced state")
    };
    let richardson = |coarse: f64, fine: f64| ((4.0 * fine - coarse) / 3.0, (coarse - fine).abs());
    let fd = |eps: f64, probe: &dyn Fn(f64) -> f64| (probe(eps) - probe(-eps)) / (2.0 * eps);
    let step = 1.0e-4_f64;
    let (mass_fd, mass_spread) = richardson(fd(step, &mass), fd(0.5 * step, &mass));
    let gamma_dot_d = gamma.t.dot(&d.t) + gamma.beta.dot(&d.beta);

    // 1. The vector claim: the FULL-units derivative of `Σ_u c_u` along `d`.
    println!(
        "[#4077 θ-adjoint] ⟨gamma, d⟩ = {gamma_dot_d:.12e}, FD of Σc_u = {mass_fd:.12e} \
         (spread {mass_spread:.3e})"
    );
    let mass_tolerance = 10.0 * mass_spread + 1.0e-6 * mass_fd.abs().max(1.0);
    assert!(
        (gamma_dot_d - mass_fd).abs() <= mass_tolerance,
        "the θ-adjoint {gamma_dot_d} is not the full-units inner derivative of the \
         half-line mass {mass_fd} (|Δ| = {:.3e}, tolerance {mass_tolerance:.3e})",
        (gamma_dot_d - mass_fd).abs()
    );

    // 2. The per-slot gradient-leg Jacobian at the GENUINELY pinned slots, and
    //    the σ-leg by subtraction (leg 3).
    let plus = operand(step);
    let minus = operand(-step);
    let mut gradient_leg = 0.0_f64;
    let mut sigma_leg_fd = 0.0_f64;
    for (position, slot) in slots.iter().enumerate() {
        let dg_fd = (plus[position].gradient - minus[position].gradient) / (2.0 * step);
        let d_sigma_fd = (plus[position].curvature - minus[position].curvature) / (2.0 * step);
        let border_dot: f64 = slot
            .border_curvature
            .iter()
            .zip(d.beta.iter())
            .map(|(&(_, value), &step_value)| value * step_value)
            .sum();
        // `row_curvature` is ROW-LOCAL (`q` entries); the joint offset is the
        // row's block start, exactly where the channels scatter it
        // (`theta.t[base + other]`).
        let base = cache.row_offsets[slot.row];
        let q = slot.row_curvature.len();
        let dg_analytic = slot.row_curvature.dot(&d.t.slice(s![base..base + q])) + border_dot;
        println!(
            "[#4077 θ-adjoint] slot {}: dg FD {dg_fd:.12e} vs rows {dg_analytic:.12e}; \
             dσ FD {d_sigma_fd:.12e} (κ₂+κ₁² = {:.12e})",
            slot.index,
            slot.curvature_coefficient()
        );
        let gradient_tolerance = 1.0e-6 * dg_fd.abs().max(1.0e-3);
        assert!(
            (dg_fd - dg_analytic).abs() <= gradient_tolerance,
            "slot {}: the raw curvature rows are not the Jacobian of g_u: FD {dg_fd:.12e} \
             vs analytic {dg_analytic:.12e}",
            slot.index
        );
        gradient_leg += slot.gradient_coefficient() * dg_fd;
        sigma_leg_fd += slot.curvature_coefficient() * d_sigma_fd;
    }

    // 3. The σ-leg by subtraction: what the weight-adjoint must have contributed.
    let sigma_leg_analytic = gamma_dot_d - gradient_leg;
    println!(
        "[#4077 θ-adjoint] σ-leg by subtraction {sigma_leg_analytic:.12e} vs FD \
         {sigma_leg_fd:.12e}; gradient leg {gradient_leg:.12e}"
    );
    let sigma_tolerance = 10.0 * mass_spread + 1.0e-6 * sigma_leg_fd.abs().max(1.0e-3);
    assert!(
        (sigma_leg_analytic - sigma_leg_fd).abs() <= sigma_tolerance,
        "the weight-adjoint σ-leg {sigma_leg_analytic} is not Σ(κ₂+κ₁²)·dσ = {sigma_leg_fd} \
         (|Δ| = {:.3e})",
        (sigma_leg_analytic - sigma_leg_fd).abs()
    );
}

/// #2231/#4077 — with crosscoder pricing spans installed, the span leg's `1.5`
/// probe reads `∂σ_u/∂log λ` exactly (affine target, `c = √λ`), and the block's
/// implicit right-hand side carries `∂g_u/∂log λ` on the pinned slots.
#[test]
fn crosscoder_span_leg_prices_exact_sigma_and_gradient_derivatives_4077() {
    let (mut term, target, mut rho) = converged_interval_state();
    let p = target.ncols();
    let cache = fitted_cache(&mut term, &target, &rho);
    let base = term
        .pinned_half_line_slots(&rho, target.view(), &cache)
        .expect("half-line operands");
    assert!(!base.is_empty(), "no slot is pinned at the fitted state");

    // One crosscoder block over every output column, weight `λ = 1`
    // (`log λ = 0`), exactly as the outer objective installs pricing spans.
    rho.log_lambda_block = vec![0.0];
    term.crosscoder_pricing_spans = Some((0, vec![p]));
    let block = rho.block_flat_range().start;
    let Some((trace, _)) = term
        .pinned_half_line_channels(&rho, target.view(), &cache)
        .expect("the channels price with spans installed")
    else {
        panic!("the fitted state pins slots, so the channels must be Some");
    };

    // The block's target columns scaled by `factor`, everything else fixed.
    let scaled = |factor: f64| {
        let mut shifted = target.clone();
        shifted
            .slice_mut(s![.., 0..p])
            .mapv_inplace(|value| factor * value);
        term.pinned_half_line_slots(&rho, shifted.view(), &cache)
            .expect("half-line operands at the scaled target")
    };
    // `c = 1 ± δ/2` moves `log λ` by `±δ` (c = √λ), so the central difference
    // over `2δ` IS `∂/∂log λ` — with the affineness making it exact to rounding.
    let delta = 1.0e-6_f64;
    let plus = scaled(1.0 + delta / 2.0);
    let minus = scaled(1.0 - delta / 2.0);
    let moved = scaled(1.5);

    let mut probe_gap = 0.0_f64;
    let mut sigma_leg = 0.0_f64;
    let mut gradient_leg_fd = 0.0_f64;
    for (position, slot) in base.iter().enumerate() {
        let d_sigma_fd = (plus[position].curvature - minus[position].curvature) / (2.0 * delta);
        let d_sigma_probe = moved[position].curvature - slot.curvature;
        println!(
            "[#4077 span] slot {}: ∂σ/∂log λ probe {d_sigma_probe:.12e} vs FD {d_sigma_fd:.12e}",
            slot.index
        );
        let sigma_scale = d_sigma_fd.abs().max(1.0e-6);
        assert!(
            (d_sigma_probe - d_sigma_fd).abs() <= 1.0e-7 * sigma_scale,
            "slot {}: the 1.5 probe {d_sigma_probe} is not ∂σ/∂log λ = {d_sigma_fd}",
            slot.index
        );
        probe_gap = probe_gap.max((d_sigma_probe - d_sigma_fd).abs());
        sigma_leg += 0.5 * slot.curvature_coefficient() * d_sigma_fd;

        let d_gradient_fd = (plus[position].gradient - minus[position].gradient) / (2.0 * delta);
        gradient_leg_fd += 0.5 * slot.gradient_coefficient() * d_gradient_fd;
    }

    // The trace's block entry is the criterion-unit σ-leg (and nothing else:
    // no prior operator owns a block coordinate).
    println!(
        "[#4077 span] trace[block] = {:.12e}, σ-leg FD {sigma_leg:.12e} (max probe gap \
         {probe_gap:.3e}); gradient leg FD {gradient_leg_fd:.12e}",
        trace[block]
    );
    let trace_tolerance = 1.0e-7 * sigma_leg.abs().max(1.0e-6);
    assert!(
        (trace[block] - sigma_leg).abs() <= trace_tolerance,
        "trace[block] {} is not the span σ-leg {} (|Δ| = {:.3e})",
        trace[block],
        sigma_leg,
        (trace[block] - sigma_leg).abs()
    );

    // The block's implicit right-hand side carries `∂g_u/∂log λ` on the pinned
    // slots — the entries the outer gradient's half-line leg multiplies.
    let rhs = term
        .crosscoder_block_ift_rhs(&cache, target.view(), 0..p)
        .expect("the block implicit right-hand side");
    for (position, slot) in base.iter().enumerate() {
        let d_gradient_fd = (plus[position].gradient - minus[position].gradient) / (2.0 * delta);
        let rhs_entry = rhs.t[slot.index];
        println!(
            "[#4077 span] slot {}: rhs ∂g/∂log λ {rhs_entry:.12e} vs FD {d_gradient_fd:.12e}",
            slot.index
        );
        let rhs_tolerance = 1.0e-6 * d_gradient_fd.abs().max(1.0e-3);
        assert!(
            (rhs_entry - d_gradient_fd).abs() <= rhs_tolerance,
            "slot {}: the block rhs entry {rhs_entry} is not ∂g_u/∂log λ = {d_gradient_fd}",
            slot.index
        );
    }
}

/// #4077/#2231 — the COMPLETE block-coordinate gradient is the derivative of
/// the value the objective itself re-prices.
///
/// `log λ` enters the reported cost through the target scaling `√λ·Y` (applied
/// by `fit_at_fixed_rho`'s authoritative lane), the change-of-variables
/// Jacobian `−(n·p_ℓ/2)·log λ_ℓ`, and the inner re-solve — and the analytic
/// entry assembles four channels ([`outer_objective`] documents them at the
/// `block_log_lambda_gradient` doc): the envelope data derivative `D_ℓ`, the
/// Jacobian `−n·p_ℓ/2`, the `½tr(A⁺∂A/∂log λ_ℓ)` trace, and the implicit
/// `−½·Γᵀθ̂_ρ` adjoint — the last two carrying the half-line legs this module's
/// other two receipts verified in isolation (the span σ-probe and the block
/// rhs entries). `tests_crosscoder_block_gradient_2231` checks the channels
/// factor-wise at a frozen state WITHOUT pinned slots; here the state pins
/// slots (the half-line channels are live), each perturbed arm runs its own
/// authoritative inner solve, and the central difference of the reported costs
/// meets the complete analytic entry in one shot.
///
/// [`outer_objective`]: super::outer_objective
#[test]
fn block_coordinate_complete_gradient_is_the_repriced_value_derivative_4077() {
    use super::tests_kappa_outer_gradient_2935::richardson;

    let (state, target, rho_seed) = converged_interval_state();
    let n = target.nrows() as f64;
    let p = target.ncols();
    let mut template = rho_seed.clone();
    template.log_lambda_block = vec![0.0];

    // One anchor column and one output relevance block, as #2231 installs.
    let mut objective = SaeManifoldOuterObjective::new(
        state.clone(),
        target.clone(),
        None,
        template.clone(),
        60,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .with_crosscoder_blocks(1, vec![p - 1])
    .expect("one anchor column and one output block");
    let baseline = objective.baseline_rho.clone();
    let block = baseline.block_flat_range().start;
    let flat = baseline
        .to_flat(&objective.term.assignment)
        .expect("the typed flat layout");

    // Base arm: re-price at log λ = 0 through the production lane.
    objective
        .fit_at_fixed_rho(flat.view())
        .expect("the base arm re-prices");
    let base_cost = objective
        .terminal_penalized_quasi_laplace_criterion
        .expect("the base arm prices a finite value");

    // The dense route re-prices the installed state; the Jacobian vanishes at
    // log λ = 0, so its cost must be the fixed-rho cost.
    let evaluation = objective
        .evaluate_outer_criterion_route(&baseline, true, false)
        .expect("the dense route prices the fitted state");
    assert!(
        (evaluation.cost - base_cost).abs() <= 1.0e-9 * base_cost.abs().max(1.0),
        "the dense route's cost {} and the fixed-rho cost {} disagree at log λ = 0",
        evaluation.cost,
        base_cost
    );
    // The state the gradient differentiates pins slots — the half-line
    // channels contribute to its block entry, they are not dormant here.
    let mut fitted = objective.term.clone();
    let cache = fitted_cache(&mut fitted, &target, &baseline);
    let pinned = fitted
        .pinned_half_line_slots(&baseline, target.view(), &cache)
        .expect("half-line operands at the fitted state");
    assert!(!pinned.is_empty(), "the re-priced state pins no slot");
    let analytic = objective
        .analytic_gradient_for_outer_evaluation(&baseline, &evaluation)
        .expect("the dense route differentiates the fitted state")[block];

    // FD arms: fresh objectives re-pricing at log λ = ±δ, each through its own
    // authoritative lane (own target scaling, own inner solve, own Jacobian).
    let repriced = |delta: f64| -> f64 {
        let mut arm = SaeManifoldOuterObjective::new(
            state.clone(),
            target.clone(),
            None,
            template.clone(),
            60,
            0.4,
            1.0e-8,
            1.0e-8,
        )
        .with_crosscoder_blocks(1, vec![p - 1])
        .expect("one anchor column and one output block");
        let mut shifted = flat.clone();
        shifted[block] += delta;
        arm.fit_at_fixed_rho(shifted.view())
            .unwrap_or_else(|err| panic!("the arm at log λ = {delta} re-prices: {err}"));
        arm.terminal_penalized_quasi_laplace_criterion
            .unwrap_or_else(|| panic!("the arm at log λ = {delta} prices a finite value"))
    };
    let central = |delta: f64| (repriced(delta) - repriced(-delta)) / (2.0 * delta);
    let coarse = central(1.25e-2);
    let fine = central(6.25e-3);
    let (derivative, spread) = richardson(coarse, fine);
    println!(
        "[#4077 assembly] complete dV/dlog λ = {analytic:.12e}; re-priced FD {derivative:.12e} \
         (coarse {coarse:.12e}, fine {fine:.12e}, spread {spread:.3e}); base cost {base_cost:.9e}; \
         Jacobian channel −n·p_ℓ/2 = {}",
        -0.5 * n * (p - 1) as f64
    );
    let tolerance = 10.0 * spread + 1.0e-8 * derivative.abs().max(1.0);
    assert!(
        (analytic - derivative).abs() <= tolerance,
        "the complete block gradient {analytic} is not the re-priced value's log λ derivative \
         {derivative} (|Δ| = {}, tolerance {tolerance})",
        (analytic - derivative).abs()
    );
}

/// #2822/#4077 — the global-dispersion coordinate's COMPLETE gradient is
/// the derivative of the value the objective itself re-prices — the second
/// coordinate family of the assembly ladder.
///
/// `with_global_dispersion` installs the SAME four-channel assembly over a
/// different coordinate: one `log λ` over every output column (`p_x = 0`, no
/// anchor column), and a Jacobian that prices the flat-prior decoder
/// directions the change of variables does not normalize
/// (`M_null = extra + Σ_k r_k·(M_k − rank S_k)`, folded in at install from the
/// atoms — `with_crosscoder_blocks` prices null dims as 0).
/// `tests_crosscoder_block_gradient_2231` checks this family's dense/streaming
/// route parity at one frozen state; the crosscoder family's assembly receipt
/// is the test above. Here the #2822 coordinate is verified at the assembly
/// level at TWO installs: this fixture's atoms price `Σ_k r_k·(M_k − rank S_k)
/// = 0` (a full-rank penalty), so the second pass adds `extra_null_dims = 1`
/// — the peeled-per-column-mean semantic the builder's doc names — to run the
/// Jacobian's null-dims channel at a nonzero value (a hardcoded 0 there
/// passes the crosscoder family, whose null dims are all 0, but not this
/// arm). Both passes pin slots (the half-line channels ride the global
/// coordinate too, not only a per-column block), each perturbed arm re-scales
/// the whole target and re-runs its own authoritative inner solve, and the
/// central difference of the reported costs meets the complete analytic entry
/// in one shot.
#[test]
fn global_dispersion_complete_gradient_is_the_repriced_value_derivative_2822() {
    assert_global_dispersion_complete_gradient_is_repriced_derivative(
        converged_interval_state(),
        60,
        0,
        0.0,
        "M_null = 0",
    );
    assert_global_dispersion_complete_gradient_is_repriced_derivative(
        converged_interval_state(),
        60,
        1,
        1.0,
        "one peeled-mean null dim",
    );
}

/// #2822 — the per-atom fold-in `r_k·(M_k − rank S_k)` at a GENUINELY
/// rank-deficient penalty: the fixture's gram `I − vvᵀ`, with `v` the TOP
/// eigenvector of the production data Gram, carries a one-dimensional null
/// space planted on the maximally data-supported direction — so it survives
/// the #1117 data-supported basis reduction (see the fixture's doc for the
/// axis-aligned attempts that did not) — and `with_global_dispersion` prices
/// `M_null = 2·1 = 2` from the atoms alone, no `extra_null_dims`. The FD
/// receipt verifies value≡gradient at that install, the exact `M_null == 2`
/// (then `3` with one peeled-mean dim beside the geometric null) checks the
/// fold-in against hand-derived geometry, and the two analytic entries over
/// identical states differ by exactly the one null-dim channel's `+0.5` —
/// the same isolation #4608 observed by print, now asserted on top of a
/// geometric null space rather than a constant.
#[test]
fn global_dispersion_gradient_prices_the_penalty_null_space_2822() {
    let geometric = assert_global_dispersion_complete_gradient_is_repriced_derivative(
        converged_rank_deficient_penalty_state(),
        60,
        0,
        2.0,
        "geometric penalty null space",
    );
    let beside_geometric = assert_global_dispersion_complete_gradient_is_repriced_derivative(
        converged_rank_deficient_penalty_state(),
        60,
        1,
        3.0,
        "one peeled-mean null dim beside the geometric null space",
    );
    assert!(
        (beside_geometric - geometric - 0.5).abs() <= 1.0e-12,
        "one extra null dim shifts dV/dlog λ by {:+e}, not the Jacobian channel's exact +0.5 \
         ({geometric:.12e} → {beside_geometric:.12e} over identical states)",
        beside_geometric - geometric - 0.5
    );
}

/// #2822 — the per-atom fold-in SUM, at the VALUE level, on the envelope lane
/// alone. Two atoms with DISTINCT planted null counts (one and two) at the
/// unre-profiled frame `r_k = 2` price `M_null = 2·1 + 2·2 = 6` from the
/// atoms — the fixture's structural asserts pin each atom's surviving null
/// count individually, so a fold-in that overwrote instead of accumulated,
/// dropped an atom, or swapped the per-atom counts fails them.
///
/// The value receipt isolates the null-dims channel exactly: installs at
/// `extra_null_dims = e` and `e' > e` have BIT-IDENTICAL inner fits (`e`
/// only enters the Jacobian fold-in, never the criterion the fit optimizes),
/// so the central difference of each install's re-priced value differs by
/// exactly `+0.5·(e' − e)` — the Jacobian channel `−(n·p − M_null)/2`'s null
/// slope, measured through the production value lane (`fit_at_fixed_rho`,
/// each arm its own authoritative inner solve at `√λ`-scaled target).
///
/// Why not the complete analytic-gradient receipt (#4609's shape): on this
/// fixture the DENSE route (`evaluate_outer_criterion_route`) re-solves from
/// the restored state and lands ~5.6 above the envelope's discovered basin at
/// inner budget 60 (~1.8 at 120, and at 400 the deeper walk stalls on the
/// shallow valley the rank-1 penalty leaves — the engine's fail-closed
/// refusal, KKT ‖g‖ ≈ 4× over tolerance). Two-atom mixtures genuinely expose
/// dense-vs-envelope basin divergence here; that is recorded as a finding for
/// the #3355 dead-basin lane, not papered over in this fixture.
#[test]
fn global_dispersion_value_prices_the_two_atom_null_sum_2822() {
    // One central difference per step size: the envelope lane's basin choice
    // can flip between FD arms on this fixture (the coarse and fine slopes
    // below need not agree — that is the recorded basin-divergence finding),
    // but the ISOLATION between installs at the SAME step size is exact,
    // because the installs' inner fits are bit-identical.
    let repriced_slopes = |extra_null_dims: usize| -> (f64, f64) {
        let (state, target, rho_seed) = converged_two_atom_rank_deficient_state();
        let mut template = rho_seed.clone();
        template.log_lambda_block = vec![0.0];
        let build = || {
            SaeManifoldOuterObjective::new(
                state.clone(),
                target.clone(),
                None,
                template.clone(),
                60,
                0.4,
                1.0e-8,
                1.0e-8,
            )
            .with_global_dispersion(extra_null_dims)
            .expect("one global dispersion coordinate over every column")
        };
        let objective = build();
        let baseline = objective.baseline_rho.clone();
        let block = baseline.block_flat_range().start;
        let flat = baseline
            .to_flat(&objective.term.assignment)
            .expect("the typed flat layout");
        let value_at = |delta: f64| -> f64 {
            let mut arm = build();
            let mut shifted = flat.clone();
            shifted[block] += delta;
            arm.fit_at_fixed_rho(shifted.view())
                .unwrap_or_else(|err| panic!("the arm at log λ = {delta} re-prices: {err}"));
            arm.terminal_penalized_quasi_laplace_criterion
                .unwrap_or_else(|| panic!("the arm at log λ = {delta} prices a finite value"))
        };
        let central = |delta: f64| (value_at(delta) - value_at(-delta)) / (2.0 * delta);
        let coarse = central(1.25e-2);
        let fine = central(6.25e-3);
        println!(
            "[#2822 two-atom value, extra = {extra_null_dims}] FD slopes: coarse {coarse:.12e}, \
             fine {fine:.12e} (basin flips between step sizes are the recorded divergence, \
             not a receipt input)"
        );
        (coarse, fine)
    };
    let (base_coarse, base_fine) = repriced_slopes(0);
    let (one_coarse, one_fine) = repriced_slopes(1);
    assert!(
        (one_coarse - base_coarse - 0.5).abs() <= 1.0e-9
            && (one_fine - base_fine - 0.5).abs() <= 1.0e-9,
        "one extra null dim shifts the value's log λ central difference by (coarse {:+e}, fine \
         {:+e}), not the Jacobian channel's exact +0.5 at BOTH step sizes over identical \
         two-atom fits ({base_coarse:.12e}/{base_fine:.12e} → {one_coarse:.12e}/{one_fine:.12e})",
        one_coarse - base_coarse - 0.5,
        one_fine - base_fine - 0.5
    );
}

fn assert_global_dispersion_complete_gradient_is_repriced_derivative(
    fixture: (SaeManifoldTerm, Array2<f64>, SaeManifoldRho),
    inner_budget: usize,
    extra_null_dims: usize,
    expected_m_null: f64,
    label: &str,
) -> f64 {
    use super::tests_kappa_outer_gradient_2935::richardson;

    let (state, target, rho_seed) = fixture;
    let n = target.nrows() as f64;
    let p = target.ncols();
    let mut template = rho_seed.clone();
    template.log_lambda_block = vec![0.0];

    // One log-noise-precision coordinate over every output column, as #2822
    // installs (p_x = 0; the whole target is the one block).
    let mut objective = SaeManifoldOuterObjective::new(
        state.clone(),
        target.clone(),
        None,
        template.clone(),
        inner_budget,
        0.4,
        1.0e-8,
        1.0e-8,
    )
    .with_global_dispersion(extra_null_dims)
    .expect("one global dispersion coordinate over every column");
    let baseline = objective.baseline_rho.clone();
    let block = baseline.block_flat_range().start;
    let flat = baseline
        .to_flat(&objective.term.assignment)
        .expect("the typed flat layout");

    // Base arm: re-price at log λ = 0 through the production lane.
    objective
        .fit_at_fixed_rho(flat.view())
        .expect("the base arm re-prices");
    let base_cost = objective
        .terminal_penalized_quasi_laplace_criterion
        .expect("the base arm prices a finite value");

    // The dense route re-prices the installed state; the Jacobian vanishes at
    // log λ = 0, so its cost must be the fixed-rho cost.
    let evaluation = objective
        .evaluate_outer_criterion_route(&baseline, true, false)
        .expect("the dense route prices the fitted state");
    assert!(
        (evaluation.cost - base_cost).abs() <= 1.0e-9 * base_cost.abs().max(1.0),
        "the dense route's cost {} and the fixed-rho cost {} disagree at log λ = 0",
        evaluation.cost,
        base_cost
    );
    // The state the gradient differentiates pins slots — the half-line
    // channels contribute to the global coordinate's entry, they are not
    // dormant here either.
    let mut fitted = objective.term.clone();
    let cache = fitted_cache(&mut fitted, &target, &baseline);
    let pinned = fitted
        .pinned_half_line_slots(&baseline, target.view(), &cache)
        .expect("half-line operands at the fitted state");
    assert!(!pinned.is_empty(), "the re-priced state pins no slot");
    let analytic = objective
        .analytic_gradient_for_outer_evaluation(&baseline, &evaluation)
        .expect("the dense route differentiates the fitted state")[block];

    // The M_null the install priced from the atoms (the same sum
    // with_global_dispersion folds into the Jacobian channel); read from the
    // pre-fit clone the install itself saw, so the print cannot drift.
    let m_null: f64 = extra_null_dims as f64
        + state
            .atoms
            .iter()
            .map(|atom| {
                let rank_s = SaeManifoldTerm::symmetric_rank(atom.smooth_penalty())
                    .expect("the fixture's penalty has a symmetric rank");
                let m_k = atom.smooth_penalty().nrows();
                (atom.border_frame_rank() * m_k.saturating_sub(rank_s)) as f64
            })
            .sum::<f64>();

    // The fold-in's output, checked against the HAND-DERIVED null count of the
    // fixture's construction — not the same sum re-run. `block_jacobian` feeds
    // this `null_dims` to the value every FD arm re-prices AND to the analytic
    // gradient, so the FD receipt below is blind to a wrong fold-in (a
    // misranked penalty, a wrong frame multiplicity, a dropped atom); this
    // exact assertion is that independent receipt.
    assert_eq!(
        m_null, expected_m_null,
        "the install's per-atom fold-in M_null ({label}) disagrees with the hand-derived null count"
    );

    // FD arms: fresh objectives re-pricing at log λ = ±δ, each through its own
    // authoritative lane (own whole-target scaling, own inner solve, own
    // Jacobian with the M_null correction).
    let repriced = |delta: f64| -> f64 {
        let mut arm = SaeManifoldOuterObjective::new(
            state.clone(),
            target.clone(),
            None,
            template.clone(),
            inner_budget,
            0.4,
            1.0e-8,
            1.0e-8,
        )
        .with_global_dispersion(extra_null_dims)
        .expect("one global dispersion coordinate over every column");
        let mut shifted = flat.clone();
        shifted[block] += delta;
        arm.fit_at_fixed_rho(shifted.view())
            .unwrap_or_else(|err| panic!("the arm at log λ = {delta} re-prices: {err}"));
        arm.terminal_penalized_quasi_laplace_criterion
            .unwrap_or_else(|| panic!("the arm at log λ = {delta} prices a finite value"))
    };
    let central = |delta: f64| (repriced(delta) - repriced(-delta)) / (2.0 * delta);
    let coarse = central(1.25e-2);
    let fine = central(6.25e-3);
    let (derivative, spread) = richardson(coarse, fine);
    println!(
        "[#2822 assembly, {label}] complete dV/dlog λ = {analytic:.12e}; re-priced FD {derivative:.12e} \
         (coarse {coarse:.12e}, fine {fine:.12e}, spread {spread:.3e}); base cost {base_cost:.9e}; \
         Jacobian channel −(n·p − M_null)/2 = {} (M_null = {m_null} over p = {p} columns)",
        -0.5 * (n * p as f64 - m_null)
    );
    let tolerance = 10.0 * spread + 1.0e-8 * derivative.abs().max(1.0);
    assert!(
        (analytic - derivative).abs() <= tolerance,
        "the complete global-dispersion gradient {analytic} ({label}) is not the re-priced value's \
         log λ derivative {derivative} (|Δ| = {}, tolerance {tolerance})",
        (analytic - derivative).abs()
    );
    analytic
}
