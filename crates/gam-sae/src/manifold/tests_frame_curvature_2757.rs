//! #2757 — the residual-gauge curvature is held in the structure the
//! decoder-frame parameterization gives it, and the certificate it produces is
//! unchanged by that.
//!
//! The defect: `fit_diagnostics_report` materialized `H = RᵀR` as a dense
//! `param_dim × param_dim = (p·D)²` matrix and took its dense symmetric
//! eigendecomposition, at `(p·D)³` flops and `(p·D)²` memory — 45.97 GiB and
//! 60.5% of the whole fit at `p = 4096`. The per-row pinning Jacobian is
//! output-coordinate diagonal, so with a metric that does not couple output
//! coordinates `H` is exactly block diagonal: `p` blocks of `D × D`.
//!
//! These are the gates on that claim, from four independent angles:
//!
//! 1. the structured curvature equals the dense Gram **entry by entry**, so
//!    the cheap object is the same object;
//! 2. the off-block entries are structurally zero, on a fixture whose decoder
//!    is dense on every output coordinate (so this cannot be read as sparsity);
//! 3. the certificate — verdicts, pinning rank, per-generator energy fractions
//!    — is identical whether reduced from the blocks or from the dense Gram;
//! 4. the representation actually holds `p·D²` scalars rather than `(p·D)²`,
//!    which is the cost claim itself and is load-immune.
//!
//! Plus the gauge-driving arm, where the metric *does* couple output
//! coordinates: there the curvature falls back to its root, whose dual Gram
//! carries the same spectrum, and the certificate must again be unchanged.

use crate::identifiability::{FrameColumnLayout, ResidualGaugeCurvature};
use crate::manifold::construction::ResidualGaugeCurvatureSource;
use crate::manifold::{AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldTerm};
use gam_terms::latent::LatentManifold;
use ndarray::Array2;
use std::sync::Arc;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// `k_atoms` planted circles in `p`-dimensional output space, assembled (not
/// fitted) so these gates measure the certificate rather than the optimizer.
///
/// `dense_tail` puts a nonzero decoder weight on **every** output coordinate of
/// every atom. Without it the frames would be axis-sparse and the
/// block-diagonality claim would be indistinguishable from "the fixture's
/// decoder happens to be sparse".
fn planted_term(n: usize, p: usize, k_atoms: usize, dense_tail: bool) -> SaeManifoldTerm {
    planted_term_with_gate(n, p, k_atoms, dense_tail, 3.0)
}

/// The materialized curvature a [`ResidualGaugeCurvatureSource`] carries, or a
/// panic naming why one was expected.
///
/// Deliberately a gate-side helper rather than an accessor on the enum: a caller
/// that wants the stored representation is asserting something about the fit's
/// metric and its row count, and production never asks — it matches on the arm
/// and takes the route that arm names.
pub(crate) fn expect_stored(
    source: ResidualGaugeCurvatureSource,
    context: &str,
) -> ResidualGaugeCurvature {
    match source {
        ResidualGaugeCurvatureSource::Stored(curvature) => curvature,
        // SAFETY: gate-side only. Every caller is a `#[test]` that has just
        // asserted the fit's metric and row count put it on the materializing
        // arm; reaching here means that assertion was wrong, which is the
        // failure the gate exists to report.
        ResidualGaugeCurvatureSource::Streamed { layout, .. } => panic!(
            "{context}: expected a materialized curvature, but this fit's curvature is \
             streamed (param_dim = {})",
            layout.param_dim()
        ),
    }
}

/// The same fixture, shared with the #2757 cost probe
/// ([`crate::manifold::probe_report_cost_2757_tests`]) so the stopwatch and the gates
/// measure the identical object.
pub(crate) fn planted_term_for_probe(
    n: usize,
    p: usize,
    k_atoms: usize,
    dense_tail: bool,
) -> SaeManifoldTerm {
    planted_term(n, p, k_atoms, dense_tail)
}

/// As [`planted_term`], with an explicit gate logit so a caller can plant a term
/// that claims no rows at all.
fn planted_term_with_gate(
    n: usize,
    p: usize,
    k_atoms: usize,
    dense_tail: bool,
    gate_logit: f64,
) -> SaeManifoldTerm {
    let mut s = 0x2757_0000_0000_0001u64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("harmonic order 3"));
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let theta: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r]);
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .expect("periodic evaluate");
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, (2 * k) % p]] = 1.0;
        decoder[[2, (2 * k + 1) % p]] = 1.0;
        if dense_tail {
            for c in 0..p {
                decoder[[0, c]] = 0.05 * (lcg(&mut s) - 0.5);
                decoder[[1, c]] += 0.05 * (lcg(&mut s) - 0.5);
                decoder[[2, c]] += 0.05 * (lcg(&mut s) - 0.5);
            }
        }
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("circle{k}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .expect("atom blocks agree")
            .with_basis_second_jet(evaluator.clone()),
        );
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, k_atoms), gate_logit);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("assignment blocks agree");
    let mut term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    term.set_guards_enabled(false);
    term
}

/// `H = Σ_n J_nᵀ M_n J_n` written out from the definition, with the per-row
/// pinning Jacobian materialized as the dense `p × param_dim` block the
/// certificate's doc comment describes. Deliberately naive: this is the
/// reference the structured builder is judged against, so it must not share
/// any of its reasoning.
pub(crate) fn reference_dense_gram(
    term: &SaeManifoldTerm,
    metric: &gam_problem::RowMetric,
    layout: &FrameColumnLayout,
) -> Array2<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let param_dim = layout.param_dim();
    let assignments = term.assignment.assignments();
    let mut gram = Array2::<f64>::zeros((param_dim, param_dim));
    let mut tangent = vec![0.0_f64; p];
    let rank = metric.metric_rank();
    for row in 0..n {
        let mut j = Array2::<f64>::zeros((p, param_dim));
        let mut base = 0usize;
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let d = atom.latent_dim();
            let a_nk = assignments[[row, atom_idx]];
            if a_nk > 0.0 {
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        j[[i, base + i * d + axis]] += a_nk * tangent[i];
                    }
                }
            }
            base += p * d;
        }
        // `M_n = U_n U_nᵀ`, so `JᵀM J = (U_nᵀJ)ᵀ(U_nᵀJ)`.
        let mut whitened = Array2::<f64>::zeros((rank, param_dim));
        for r in 0..rank {
            for c in 0..param_dim {
                let mut acc = 0.0_f64;
                for i in 0..p {
                    acc += metric.factor_entry(row, i, r) * j[[i, c]];
                }
                whitened[[r, c]] = acc;
            }
        }
        gram = gram + whitened.t().dot(&whitened);
    }
    gram
}

/// The stacked metric-whitened root `R` itself (`n·rank × param_dim`), built the
/// same independent way [`reference_dense_gram`] is — no
/// `fill_row_frame_jacobian`, no accumulator, no layout arithmetic beyond
/// `offset_k + i·d_k + a` written out inline.
///
/// This is the object the streamed operator claims to be an operator OVER, so
/// checking the operator against it is checking it against an independent
/// derivation rather than against the code it shares. `RᵀR` is
/// [`reference_dense_gram`]'s output by construction.
pub(crate) fn reference_dense_root(
    term: &SaeManifoldTerm,
    metric: &gam_problem::RowMetric,
    layout: &FrameColumnLayout,
) -> Array2<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let param_dim = layout.param_dim();
    let rank = metric.metric_rank();
    let assignments = term.assignment.assignments();
    let mut root = Array2::<f64>::zeros((n * rank, param_dim));
    let mut tangent = vec![0.0_f64; p];
    for row in 0..n {
        let mut j = Array2::<f64>::zeros((p, param_dim));
        let mut base = 0usize;
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let d = atom.latent_dim();
            let a_nk = assignments[[row, atom_idx]];
            if a_nk > 0.0 {
                for axis in 0..d {
                    atom.fill_decoded_derivative_row(row, axis, &mut tangent);
                    for i in 0..p {
                        j[[i, base + i * d + axis]] += a_nk * tangent[i];
                    }
                }
            }
            base += p * d;
        }
        for r in 0..rank {
            for c in 0..param_dim {
                let mut acc = 0.0_f64;
                for i in 0..p {
                    acc += metric.factor_entry(row, i, r) * j[[i, c]];
                }
                root[[row * rank + r, c]] = acc;
            }
        }
    }
    root
}

/// A curvature built in a DIFFERENT parameterization must be refused even when
/// its `param_dim` matches.
///
/// `param_dim = Σ_k p·d_k` is not injective in the atom shapes: two atoms of
/// `d = 1` and one atom of `d = 2` give the same total over the same `p`, and
/// the same block dimension `D = 2` — but every `(i, l) ↦ c` differs, so a
/// curvature from one silently reindexes the other's generators. The layout is
/// carried on the representation precisely so this is checkable.
#[test]
fn a_curvature_from_a_different_frame_layout_is_refused() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (16usize, 8usize, 2usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let curvature = expect_stored(streamed, "unpinned path streams its curvature");
    let root_rows = curvature.root_rows();
    let mine = FrameColumnLayout::new(p, &[1usize, 1]);
    // One atom of d = 2 rather than two of d = 1: same param_dim, same D.
    let impostor = FrameColumnLayout::new(p, &[2usize]);
    assert_eq!(impostor.param_dim(), mine.param_dim());
    assert_eq!(impostor.block_dim(), mine.block_dim());
    assert_ne!(
        impostor.column(1, 1),
        mine.column(1, 1),
        "the two layouts must disagree somewhere for this gate to bite"
    );

    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let relabelled = ResidualGaugeCurvature::OutputBlockRoots {
        roots: ndarray::Array3::<f64>::zeros((p, 2, 2)),
        dense_rows: Array2::<f64>::zeros((0, impostor.param_dim())),
        layout: impostor,
        root_rows,
    };
    let message = residual_gauge_exact_from_curvature(&model, &views, &ops, relabelled)
        .err()
        .expect("a curvature from another parameterization must be refused");
    assert!(
        message.contains("frame-column layout"),
        "refusal must name the cause, got {message:?}"
    );
}

