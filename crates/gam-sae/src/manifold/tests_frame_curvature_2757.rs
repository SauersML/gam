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
use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::time::Instant;

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

/// A stable tag for the route a source names: the stored representation's own
/// structure tag, or `streamed_operator`.
pub(crate) fn source_structure_tag(source: &ResidualGaugeCurvatureSource) -> &'static str {
    match source {
        ResidualGaugeCurvatureSource::Stored(curvature) => curvature.structure_tag(),
        ResidualGaugeCurvatureSource::Streamed { .. } => "streamed_operator",
    }
}

/// How many `f64` the route holds for the curvature itself.
///
/// Zero for the streamed route, and that zero is the whole point: it is the
/// load-immune, exact regression gate on #2757's memory claim, in the same
/// currency [`ResidualGaugeCurvature::stored_scalars`] already reports.
pub(crate) fn source_stored_scalars(source: &ResidualGaugeCurvatureSource) -> usize {
    match source {
        ResidualGaugeCurvatureSource::Stored(curvature) => curvature.stored_scalars(),
        ResidualGaugeCurvatureSource::Streamed { .. } => 0,
    }
}

/// The row count of the root `R` the source describes — the same number in both
/// arms, since both describe the same `R`.
pub(crate) fn source_root_rows(source: &ResidualGaugeCurvatureSource) -> usize {
    match source {
        ResidualGaugeCurvatureSource::Stored(curvature) => curvature.root_rows(),
        ResidualGaugeCurvatureSource::Streamed { root_rows, .. } => *root_rows,
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

/// The unit smoothing state the gates fit at, shared with the cost probe.
pub(crate) fn unit_rho_for_probe(k_atoms: usize) -> SaeManifoldRho {
    unit_rho(k_atoms)
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

/// A term whose decoder carries only the constant harmonic, so every decoded
/// tangent — and therefore the whole residual-gauge curvature — is exactly zero.
pub(crate) fn planted_constant_decoder_term(n: usize, p: usize, k_atoms: usize) -> SaeManifoldTerm {
    let mut s = 0x2757_0000_0000_0009u64;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).expect("harmonic order 3"));
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut manifolds = Vec::with_capacity(k_atoms);
    for k in 0..k_atoms {
        let coords = Array2::<f64>::from_shape_fn((n, 1), |_| lcg(&mut s));
        let (phi, jet) = evaluator
            .evaluate(coords.view())
            .expect("periodic evaluate");
        let mut decoder = Array2::<f64>::zeros((3, p));
        for c in 0..p {
            decoder[[0, c]] = 0.3 + 0.01 * c as f64;
        }
        atoms.push(
            SaeManifoldAtom::new_with_provided_function_gram(
                format!("flat{k}"),
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
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::from_elem((n, k_atoms), 3.0),
        coord_blocks,
        manifolds,
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .expect("assignment blocks agree");
    let mut term = SaeManifoldTerm::new(atoms, assignment).expect("term");
    term.set_guards_enabled(false);
    term
}

fn unit_rho(k_atoms: usize) -> SaeManifoldRho {
    SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k_atoms])
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

/// Gate 1 + 2 — the structured curvature IS the dense Gram, and its off-block
/// entries are structurally zero on a decoder that is dense everywhere.
#[test]
fn output_block_curvature_equals_the_dense_gram_and_has_no_off_block_mass() {
    let (n, p, k_atoms) = (48usize, 24usize, 3usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    assert!(
        !metric.drives_gauge(),
        "the diagnostic fallback must be the Euclidean (non-gauge-driving) metric"
    );
    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(
            &metric,
            &layout,
            Array2::<f64>::zeros((0, layout.param_dim())),
        )
        .expect("streamed curvature");
    assert_eq!(curvature.structure_tag(), "output_block_roots");
    assert_eq!(curvature.root_rows(), n * metric.metric_rank());

    let reference = reference_dense_gram(&term, &metric, &layout);
    let structured = curvature.to_dense_gram();
    assert_eq!(structured.dim(), reference.dim());

    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "the fixture must produce a nonzero curvature");
    let mut worst = 0.0_f64;
    let mut off_block = 0.0_f64;
    let mut off_block_reference = 0.0_f64;
    for a in 0..reference.nrows() {
        let ia = layout.output_of(a).expect("column in range");
        for b in 0..reference.ncols() {
            let ib = layout.output_of(b).expect("column in range");
            worst = worst.max((structured[[a, b]] - reference[[a, b]]).abs());
            if ia != ib {
                off_block = off_block.max(structured[[a, b]].abs());
                off_block_reference = off_block_reference.max(reference[[a, b]].abs());
            }
        }
    }
    assert!(
        worst <= 1.0e-12 * scale,
        "structured curvature must reproduce the dense Gram: worst |Δ| {worst:.3e} \
         against scale {scale:.3e}"
    );
    assert_eq!(
        off_block, 0.0,
        "the structured curvature must carry no mass between two output coordinates"
    );
    assert_eq!(
        off_block_reference, 0.0,
        "and neither does the dense reference — the block structure is the operator's, \
         not the representation's"
    );

    // Diagonal mass is spread over every output coordinate, so the zero
    // off-block is not a statement about a decoder that touches only a few.
    let touched = (0..p)
        .filter(|&i| {
            (0..k_atoms).any(|l| {
                let c = layout.column(i, l);
                reference[[c, c]].abs() > 0.0
            })
        })
        .count();
    assert_eq!(
        touched, p,
        "every output coordinate must carry curvature for this gate to bite"
    );
}

/// Gate 4 — the cost claim, as an exact count rather than a wall-clock
/// threshold: the curvature holds `p·D²` scalars where the dense Gram holds
/// `(p·D)²`.
#[test]
fn output_block_curvature_stores_p_times_fewer_scalars_than_the_dense_gram() {
    let (n, p, k_atoms) = (24usize, 64usize, 4usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(
            &metric,
            &layout,
            Array2::<f64>::zeros((0, layout.param_dim())),
        )
        .expect("streamed curvature");
    let param_dim = layout.param_dim();
    let d = layout.block_dim();
    assert_eq!(curvature.stored_scalars(), p * d * d);
    assert_eq!(
        curvature.stored_scalars() * p,
        param_dim * param_dim,
        "the saving is exactly the factor p the dense layout was padding by"
    );
}

/// Gate 3 — the certificate is identical whichever representation it is
/// reduced from. This is the one that makes the change a cost change and not a
/// behaviour change.
#[test]
fn certificate_is_identical_under_the_structured_and_dense_reductions() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (40usize, 20usize, 3usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let structured = expect_stored(streamed, "unpinned path streams its curvature");
    assert_eq!(structured.structure_tag(), "output_block_roots");
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows: structured.root_rows(),
    };

    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        vec![None; model.atoms.len()];
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();

    let from_blocks = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("structured certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");

    assert_eq!(
        from_blocks.pinning_rank, from_dense.pinning_rank,
        "pinning rank must not depend on the representation"
    );
    assert_eq!(from_blocks.generators.len(), from_dense.generators.len());
    assert!(
        !from_blocks.generators.is_empty(),
        "the fixture must enumerate generators for this gate to bite"
    );
    for (b, dsn) in from_blocks
        .generators
        .iter()
        .zip(from_dense.generators.iter())
    {
        assert_eq!(b.description, dsn.description);
        assert_eq!(b.family, dsn.family);
        assert_eq!(
            b.unpinned, dsn.unpinned,
            "generator '{}' verdict must not depend on the representation",
            b.description
        );
        let gap = (b.pinned_energy_fraction - dsn.pinned_energy_fraction).abs();
        assert!(
            gap <= 1.0e-12,
            "generator '{}' energy fraction differs by {gap:.3e} between representations",
            b.description
        );
    }
    assert_eq!(from_blocks.group_signature(), from_dense.group_signature());
    assert_eq!(
        from_blocks.residual_gauge_dim,
        from_dense.residual_gauge_dim
    );
}

/// A curvature with rank-deficient blocks: half the output coordinates carry no
/// decoder mass at all, so their blocks are exactly zero and must contribute
/// nothing to the pinning rank — the same answer the dense spectrum gives.
#[test]
fn rank_deficient_blocks_agree_with_the_dense_spectrum() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (32usize, 16usize, 2usize);
    // `dense_tail = false` leaves each atom's decoder supported on two output
    // coordinates, so most blocks are identically zero.
    let term = planted_term(n, p, k_atoms, false);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let structured = expect_stored(streamed, "unpinned path streams its curvature");
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows: structured.root_rows(),
    };
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        vec![None; model.atoms.len()];
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let from_blocks = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("structured certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");
    assert!(
        from_blocks.pinning_rank < p * k_atoms,
        "the fixture must be rank deficient for this gate to bite (rank {} of {})",
        from_blocks.pinning_rank,
        p * k_atoms
    );
    assert_eq!(from_blocks.pinning_rank, from_dense.pinning_rank);
}

/// The gauge-driving arm: an output-Fisher metric couples output coordinates,
/// so the curvature is NOT block diagonal and the builder must say so — and
/// what it returns must still reproduce the dense Gram exactly.
///
/// This test used to assert `structure_tag() == "dense_gram"` here, i.e. it
/// CODIFIED the surviving half of #2757's own defect: with more root rows than
/// columns the builder assembled the `param_dim × param_dim` Gram and the
/// certificate read its spectrum through a symmetric eigendecomposition. The
/// rows are now folded into the `param_dim`-square upper-triangular factor `T`
/// with `TᵀT = RᵀR` instead, so the tag is `dual_root` on both sides of the
/// fork and the production builder emits no `DenseGram` at all.
///
/// The Gram-equality assertion below is unchanged and is what makes that a cost
/// and resolution change rather than a different answer: `to_dense_gram()` on
/// the folded factor is `TᵀT`, which must still match the naive
/// `Σ_n J_nᵀ M_n J_n` reference entry by entry.
#[test]
fn gauge_driving_metric_folds_into_a_root_that_reproduces_the_dense_gram() {
    let (n, p, k_atoms, rank) = (12usize, 10usize, 2usize, 3usize);
    let mut term = planted_term(n, p, k_atoms, true);
    let mut s = 0x2757_FEED_0000_0001u64;
    let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg(&mut s) - 0.5);
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable with the term");
    assert!(metric.drives_gauge());

    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(
            &metric,
            &layout,
            Array2::<f64>::zeros((0, layout.param_dim())),
        )
        .expect("streamed curvature");
    // n·rank = 36 root rows against param_dim = 20 columns, so the root is the
    // larger object and its rows are folded into the triangular factor.
    assert_eq!(curvature.structure_tag(), "dual_root");
    assert_eq!(
        curvature.stored_scalars(),
        layout.param_dim() * layout.param_dim(),
        "the folded factor is param_dim-square, which is the point: it is the \
         SMALLER of the two exact sides once the root outgrows its columns"
    );

    let reference = reference_dense_gram(&term, &metric, &layout);
    let built = curvature.to_dense_gram();
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0);
    let worst = built
        .iter()
        .zip(reference.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1.0e-12 * scale,
        "gauge-driving curvature must reproduce the dense Gram: worst |Δ| {worst:.3e}"
    );
    // A gauge-driving metric genuinely couples output coordinates, so the block
    // structure must NOT be claimed here.
    let mut off_block = 0.0_f64;
    for a in 0..reference.nrows() {
        let ia = layout.output_of(a).expect("in range");
        for b in 0..reference.ncols() {
            if ia != layout.output_of(b).expect("in range") {
                off_block = off_block.max(reference[[a, b]].abs());
            }
        }
    }
    assert!(
        off_block > 0.0,
        "an output-Fisher metric must couple output coordinates, else this arm is vacuous"
    );
}

/// The folded factor certifies the SAME model as the dense Gram it replaces.
///
/// This is the equivalence the previous test's tag change rests on. Both
/// representations are handed to the identical certificate entry point and
/// every per-generator verdict must agree — the pinned/unpinned flag exactly,
/// the relative curvature fraction to the last digits either instrument can
/// claim. A cost change that moved a verdict would be a different certificate
/// wearing the same name.
///
/// The pinning RANK is deliberately not asserted equal, and that is the second
/// half of why the fold is the right object rather than merely a cheaper one.
/// `gram_spectral_rank` has to take the decision on `λ = σ²` and is therefore
/// floored at the eigensolver's own resolution `ε·param_dim·λ_max`; the root
/// decision is `σ > τ` with `τ = α·ε·N·σ_max`, which resolves `τ² ≈ 1e-16·λ_max`
/// — five orders finer at this shape and more at production width. So the Gram
/// can only ever count a SUBSET of what the root counts, which is what is
/// asserted, alongside the bound neither may exceed.
#[test]
fn the_folded_root_and_the_dense_gram_certify_the_same_model() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms, rank) = (24usize, 12usize, 3usize, 3usize);
    let mut term = planted_term(n, p, k_atoms, true);
    let mut s = 0x2757_F01D_0000_0001u64;
    let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg(&mut s) - 0.5);
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable");

    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let (model, source) = term
        .to_residual_gauge_model(metric.clone(), None, false)
        .expect("certificate model");
    // n·rank = 72 root rows against param_dim = 36 columns, so production streams
    // this fit rather than materializing it (#2757). That route is gated in
    // `tests_streamed_curvature_2757`; THIS gate is about the two MATERIALIZED
    // reductions agreeing, so it asks the builder for the folded factor
    // explicitly — the witness the streamed route is judged against.
    assert_eq!(source_structure_tag(&source), "streamed_operator");
    let folded = term
        .residual_gauge_streamed_data_curvature(
            &metric,
            &layout,
            Array2::<f64>::zeros((0, layout.param_dim())),
        )
        .expect("materialized curvature");
    assert_eq!(folded.structure_tag(), "dual_root");
    assert_eq!(folded.root_rows(), n * rank);

    let dense = ResidualGaugeCurvature::DenseGram {
        gram: folded.to_dense_gram(),
        root_rows: folded.root_rows(),
    };
    let views = term.atom_parameter_views();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..k_atoms).map(|_| None).collect();
    let from_fold = residual_gauge_exact_from_curvature(&model, &views, &ops, folded)
        .expect("folded certificate");
    let from_gram = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense-Gram certificate");

    assert_eq!(from_fold.generators.len(), from_gram.generators.len());
    assert!(
        !from_fold.generators.is_empty(),
        "a certificate with no generators cannot separate the two reductions"
    );
    for (a, b) in from_fold.generators.iter().zip(from_gram.generators.iter()) {
        assert_eq!(a.description, b.description);
        assert_eq!(
            a.unpinned, b.unpinned,
            "generator `{}` is {} from the folded root and {} from the dense Gram",
            a.description,
            if a.unpinned { "unpinned" } else { "pinned" },
            if b.unpinned { "unpinned" } else { "pinned" }
        );
        let gap = (a.pinned_energy_fraction - b.pinned_energy_fraction).abs();
        assert!(
            gap <= 1.0e-9,
            "generator `{}` scores {:.17e} from the folded root against {:.17e} from \
             the dense Gram",
            a.description,
            a.pinned_energy_fraction,
            b.pinned_energy_fraction
        );
    }

    let bound = (n * rank).min(model.param_dim());
    assert!(
        from_fold.pinning_rank <= bound && from_gram.pinning_rank <= bound,
        "no reduction may report a rank above min(root rows, param_dim) = {bound}; \
         folded {} gram {}",
        from_fold.pinning_rank,
        from_gram.pinning_rank
    );
    assert!(
        from_gram.pinning_rank <= from_fold.pinning_rank,
        "the Gram decision is floored at the eigensolver's resolution, so it can \
         only count a subset of what the root counts; gram {} > folded {}",
        from_gram.pinning_rank,
        from_fold.pinning_rank
    );
}

/// A curvature that is exactly zero must certify at rank zero on the FOLD arm
/// too, not merely on the block arm.
///
/// The fold is the one representation whose stored object (a `param_dim`-square
/// factor) is the same shape whether the curvature is full rank or identically
/// zero, so "the factor is big" must not be read as "the rank is big".
#[test]
fn a_zero_curvature_folds_to_rank_zero_under_a_gauge_driving_metric() {
    let (n, p, k_atoms, rank) = (20usize, 8usize, 2usize, 3usize);
    let mut term = planted_constant_decoder_term(n, p, k_atoms);
    let mut s = 0x2757_F01D_0000_0002u64;
    let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg(&mut s) - 0.5);
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable");

    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(
            &metric,
            &layout,
            Array2::<f64>::zeros((0, layout.param_dim())),
        )
        .expect("streamed curvature");
    // n·rank = 60 root rows against param_dim = 16: the fold arm.
    assert_eq!(curvature.structure_tag(), "dual_root");
    let worst = curvature
        .to_dense_gram()
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        worst == 0.0,
        "a constant decoder has an identically zero curvature; the folded factor \
         reconstructs {worst:.3e}"
    );
}

/// The same arm at a shape where the root has fewer rows than columns.
///
/// This is also the gate on the *rank* half of #2757. `H = RᵀR` with 12 root
/// rows in 80 parameters has rank at most 12 — a mathematical bound, not an
/// estimate. Before the fix the dense-Gram path reported **45**, because the
/// tolerance `τ = 100·ε·N·σ_max` (deliberately 100x above an SVD's backward
/// error) was being squared into `τ²`, which lands ~10⁹x BELOW a symmetric
/// eigensolver's own resolution, so 33 roundoff eigenvalues cleared it. Both
/// representations must now agree, and both must respect the bound.
#[test]
fn dual_root_and_dense_gram_agree_on_a_rank_neither_may_exceed() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms, rank) = (6usize, 40usize, 2usize, 2usize);
    let mut term = planted_term(n, p, k_atoms, true);
    let mut s = 0x2757_FEED_0000_0002u64;
    let factors = Array2::<f64>::from_shape_fn((n, p * rank), |_| lcg(&mut s) - 0.5);
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable");

    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let curvature = term
        .residual_gauge_streamed_data_curvature(
            &metric,
            &layout,
            Array2::<f64>::zeros((0, layout.param_dim())),
        )
        .expect("streamed curvature");
    // n*rank = 12 root rows against param_dim = 80 columns.
    assert_eq!(curvature.structure_tag(), "dual_root");
    assert_eq!(curvature.stored_scalars(), n * rank * layout.param_dim());

    let reference = reference_dense_gram(&term, &metric, &layout);
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = curvature
        .to_dense_gram()
        .iter()
        .zip(reference.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1.0e-12 * scale,
        "dual-root curvature must reproduce the dense Gram: worst |Δ| {worst:.3e}"
    );

    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let structured = expect_stored(streamed, "unpinned path streams its curvature");
    let root_rows = structured.root_rows();
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows,
    };
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let from_root = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("dual-root certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");

    let bound = root_rows.min(layout.param_dim());
    assert!(
        from_root.pinning_rank <= bound,
        "rank(RᵀR) <= rows(R): root path reported {} against the bound {bound}",
        from_root.pinning_rank
    );
    assert!(
        from_dense.pinning_rank <= bound,
        "rank(RᵀR) <= rows(R): dense-Gram path reported {} against the bound {bound} —          the Gram is counting eigenvalues below its own resolution",
        from_dense.pinning_rank
    );
    assert_eq!(
        from_root.pinning_rank, from_dense.pinning_rank,
        "the two representations must decide one rank"
    );
    for (r, d) in from_root
        .generators
        .iter()
        .zip(from_dense.generators.iter())
    {
        assert_eq!(r.unpinned, d.unpinned, "generator '{}'", r.description);
        assert!(
            (r.pinned_energy_fraction - d.pinned_energy_fraction).abs() <= 1.0e-10,
            "generator '{}' energy fraction",
            r.description
        );
    }
}

/// A non-finite curvature must be REFUSED, not summarised.
///
/// The dense path got this for free: `FaerEigh::eigh` validates its input and
/// returns a typed error. `FaerSvd::svd` does not, so a NaN singular value
/// would silently fail every `σ > τ` test and be reported as a rank-zero,
/// fully-unpinned model — the most permissive verdict the certificate can
/// issue, from the least trustworthy input.
#[test]
fn a_non_finite_curvature_is_refused_in_every_representation() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (16usize, 8usize, 2usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let clean = expect_stored(streamed, "unpinned path streams its curvature");
    let root_rows = clean.root_rows();
    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();

    // The clean curvature certifies.
    assert!(
        residual_gauge_exact_from_curvature(&model, &views, &ops, clean).is_ok(),
        "the fixture must certify before the poisoned arms mean anything"
    );

    let mut poisoned = ndarray::Array3::<f64>::zeros((p, k_atoms, k_atoms));
    poisoned[[p - 1, k_atoms - 1, k_atoms - 1]] = f64::NAN;
    let arms = [
        ResidualGaugeCurvature::OutputBlockRoots {
            roots: poisoned,
            dense_rows: Array2::<f64>::zeros((0, layout.param_dim())),
            layout: layout.clone(),
            root_rows,
        },
        ResidualGaugeCurvature::DualRoot {
            root: Array2::<f64>::from_elem((2, layout.param_dim()), f64::NAN),
            root_rows,
        },
        ResidualGaugeCurvature::DenseGram {
            gram: Array2::<f64>::from_elem((layout.param_dim(), layout.param_dim()), f64::NAN),
            root_rows,
        },
    ];
    for arm in arms {
        let tag = arm.structure_tag();
        let refusal = residual_gauge_exact_from_curvature(&model, &views, &ops, arm);
        let message = refusal
            .err()
            .unwrap_or_else(|| panic!("{tag}: a non-finite curvature must be refused"));
        assert!(
            message.contains("non-finite"),
            "{tag}: refusal must name the cause, got {message:?}"
        );
    }
}

/// A term carrying no assignment mass anywhere: the curvature is exactly zero,
/// which is a certificate (rank 0, everything unpinned), not an error and not a
/// panic. The old dense path reached this through an eigendecomposition of a
/// zero matrix; the block path never decomposes anything.
#[test]
fn an_unassigned_term_certifies_at_rank_zero() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (12usize, 6usize, 2usize);
    // A decoder with no harmonic content at all: every decoded tangent is
    // identically zero, so the curvature is the zero operator.
    let term = planted_constant_decoder_term(n, p, k_atoms);
    for row in 0..n {
        for atom in &term.atoms {
            let tangent = atom.decoded_derivative_row(row, 0);
            assert!(
                tangent.iter().all(|v| *v == 0.0),
                "the fixture must have no decoded tangent for this gate to bite"
            );
        }
    }
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    let curvature = expect_stored(streamed, "unpinned path streams its curvature");
    assert_eq!(curvature.structure_tag(), "output_block_roots");
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let report = residual_gauge_exact_from_curvature(&model, &views, &ops, curvature)
        .expect("a zero curvature is a certificate, not an error");
    assert_eq!(report.pinning_rank, 0);
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

/// The pin-active branch: same function, same root cause, worse blowup.
///
/// With the isometry pin ACTIVE `to_residual_gauge_model` used to materialize
/// each per-row pinning Jacobian as a dense `p x param_dim` block and retain
/// all `n` of them -- `8*n*p^2*D` bytes, which is 2.55 GiB PER OBSERVATION at
/// `p = 4096, D = 19`. It now streams the same block roots the unpinned branch
/// does and carries the pin's `sum_k d_k` rows as a symmetric update, so the
/// retained Jacobian is gone entirely.
#[test]
fn the_pin_active_branch_streams_instead_of_retaining_a_dense_jacobian() {
    let (n, k_atoms) = (12usize, 2usize);
    for &p in &[16usize, 32, 64] {
        let term = planted_term(n, p, k_atoms, true);
        let metric = term.diagnostic_metric().expect("metric");
        let (model, streamed) = term
            .to_residual_gauge_model(metric, None, true)
            .expect("pin-active certificate model");
        assert!(
            model.jacobian_rows.is_empty(),
            "the pin-active branch must not retain a dense per-row Jacobian"
        );
        let curvature = expect_stored(streamed, "both branches stream their curvature");
        assert_eq!(curvature.structure_tag(), "output_block_roots");
        assert!(
            model.isometry_penalty_root.nrows() > 0,
            "the pin must actually be installed for this gate to bite"
        );
        // `p*D^2` block roots plus the pin's `D'*(p*D)` rows -- linear in `p`,
        // against the `n*p^2*D` the dense layout retained.
        let expected =
            p * k_atoms * k_atoms + model.isometry_penalty_root.nrows() * model.param_dim();
        assert_eq!(curvature.stored_scalars(), expected);
        assert!(
            curvature.stored_scalars() < n * p * model.param_dim(),
            "the structured curvature must be smaller than one dense Jacobian stack"
        );
    }
}

/// The pin-active certificate must equal the dense one it replaces.
///
/// `H = (+)_i R_i^T R_i + L^T L` is block diagonal plus a rank-`sum_k d_k`
/// update, and that sum's spectrum is genuinely global -- the pin's rows cannot
/// be folded into the blocks. It is still exactly computable by inertia, so the
/// rank, the stiffness scale and every generator verdict must match the dense
/// Gram's to the last decision.
#[test]
fn the_pin_active_certificate_matches_the_dense_gram_exactly() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (24usize, 12usize, 3usize);
    let term = planted_term(n, p, k_atoms, true);
    let metric = term.diagnostic_metric().expect("metric");
    let (model, streamed) = term
        .to_residual_gauge_model(metric, None, true)
        .expect("pin-active certificate model");
    let structured = expect_stored(streamed, "pin-active branch streams its curvature");
    assert!(model.isometry_penalty_root.nrows() > 0);
    let dense = ResidualGaugeCurvature::DenseGram {
        gram: structured.to_dense_gram(),
        root_rows: structured.root_rows(),
    };
    let views: Vec<Option<crate::identifiability::AtomParameterView>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..model.atoms.len()).map(|_| None).collect();
    let from_blocks = residual_gauge_exact_from_curvature(&model, &views, &ops, structured)
        .expect("structured pin-active certificate");
    let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
        .expect("dense certificate");
    assert_eq!(
        from_blocks.pinning_rank, from_dense.pinning_rank,
        "the inertia count must agree with the dense spectrum"
    );
    assert!(!from_blocks.generators.is_empty());
    for (b, d) in from_blocks
        .generators
        .iter()
        .zip(from_dense.generators.iter())
    {
        assert_eq!(b.description, d.description);
        assert_eq!(
            b.unpinned, d.unpinned,
            "generator '{}' verdict must not depend on the representation",
            b.description
        );
        let gap = (b.pinned_energy_fraction - d.pinned_energy_fraction).abs();
        assert!(
            gap <= 1.0e-10,
            "generator '{}' energy fraction differs by {gap:.3e}",
            b.description
        );
    }
    assert_eq!(from_blocks.group_signature(), from_dense.group_signature());
}

/// The whole production entry point, through BOTH branches, against the dense
/// reduction of the very same curvature.
///
/// The gates above exercise `residual_gauge_exact_from_curvature` directly.
/// This one goes through `fit_diagnostics_report`, which is where a branch that
/// was left pointing at the old entry point would show up — and did: routing
/// the pin-active branch through `residual_gauge_exact` after
/// `jacobian_rows` stopped being populated would have reported a zero
/// curvature, every generator unpinned, and no error at all.
#[test]
fn fit_diagnostics_report_certifies_the_same_thing_on_both_branches() {
    use crate::identifiability::residual_gauge_exact_from_curvature;

    let (n, p, k_atoms) = (24usize, 12usize, 3usize);
    for pin in [false, true] {
        let term = planted_term(n, p, k_atoms, true);
        let rho = unit_rho(k_atoms);
        let fitted = term
            .try_fitted_target_aware(Array2::<f64>::zeros((n, p)).view(), Some(&rho))
            .expect("fitted");
        let report = term
            .fit_diagnostics_report(None, pin, None, fitted.view(), None)
            .expect("diagnostics report")
            .residual_gauge;
        assert!(
            report.pinning_rank > 0,
            "pin={pin}: the certificate must see a nonzero curvature"
        );

        // The same model, reduced from the dense Gram of the same curvature.
        let metric = term.diagnostic_metric().expect("metric");
        let (model, streamed) = term
            .to_residual_gauge_model(metric, None, pin)
            .expect("certificate model");
        let structured = expect_stored(streamed, "both branches stream their curvature");
        assert_eq!(
            (model.isometry_penalty_root.nrows() > 0),
            pin,
            "pin={pin}: the pin must be installed exactly when requested"
        );
        let dense = ResidualGaugeCurvature::DenseGram {
            gram: structured.to_dense_gram(),
            root_rows: structured.root_rows(),
        };
        let views: Vec<Option<crate::identifiability::AtomParameterView>> =
            (0..model.atoms.len()).map(|_| None).collect();
        let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> = if pin {
            model
                .atoms
                .iter()
                .map(|_| None)
                .collect::<Vec<Option<crate::identifiability::OrbitPenaltyOperator>>>()
        } else {
            (0..model.atoms.len()).map(|_| None).collect()
        };
        let from_dense = residual_gauge_exact_from_curvature(&model, &views, &ops, dense)
            .expect("dense certificate");
        assert_eq!(
            report.pinning_rank, from_dense.pinning_rank,
            "pin={pin}: the shipped report must agree with the dense reduction"
        );
        assert_eq!(
            report.diffeomorphism_unpinned, !pin,
            "pin={pin}: the escalation must track the installed pin"
        );
    }
}

/// The inertia counter itself, against a dense eigendecomposition, on a
/// deliberately awkward operator: a structurally empty output coordinate, a
/// rank-deficient block, and an update that reaches into the blocks' null
/// spaces.
#[test]
fn block_plus_rows_inertia_matches_a_dense_eigendecomposition() {
    use crate::identifiability::frame_curvature::BlockPlusRowsSpectrum;
    use gam_linalg::faer_ndarray::FaerEigh;

    let layout = FrameColumnLayout::new(5, &[2usize, 1]);
    let (p, d) = (layout.output_dim(), layout.block_dim());
    let mut seed = 0x2757_11E4_71A0_0001u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let mut roots = ndarray::Array3::<f64>::zeros((p, d, d));
    for i in 0..p {
        if i == 2 {
            continue; // a structurally empty output coordinate
        }
        for a in 0..d {
            for b in a..d {
                // Block 3 keeps its last row zero: rank deficient without being
                // the zero block.
                if i == 3 && a == d - 1 {
                    continue;
                }
                roots[[i, a, b]] = next();
            }
        }
    }
    let mut dense_rows = Array2::<f64>::zeros((2, layout.param_dim()));
    for j in 0..2 {
        for c in 0..layout.param_dim() {
            dense_rows[[j, c]] = next();
        }
    }
    let curvature = ResidualGaugeCurvature::OutputBlockRoots {
        roots: roots.clone(),
        dense_rows: dense_rows.clone(),
        layout: layout.clone(),
        root_rows: p * d + 2,
    };
    let gram = curvature.to_dense_gram();
    let (evals, _) = gram.eigh(faer::Side::Lower).expect("dense reference");
    let spectrum =
        BlockPlusRowsSpectrum::new(&roots, &dense_rows, &layout).expect("inertia machinery");

    let lambda_max = spectrum.lambda_max().expect("lambda_max");
    let reference_max = evals.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        (lambda_max - reference_max).abs() <= 1.0e-10 * reference_max.max(1.0),
        "lambda_max {lambda_max:.12e} against the dense {reference_max:.12e}"
    );

    // The eigenvalue COUNT above a shift must match at every scale, including
    // shifts inside the spectrum where the update has moved eigenvalues across.
    for exponent in -13i32..=1 {
        let shift = reference_max * 10.0_f64.powi(exponent);
        let counted = spectrum.count_above(shift).expect("inertia count");
        let reference = evals.iter().filter(|v| **v > shift).count();
        assert_eq!(
            counted, reference,
            "shift {shift:.3e}: inertia says {counted}, the dense spectrum says {reference}"
        );
    }
}

/// The wall #2757 was filed on, measured on the fixed path. The dense path's
/// own numbers on this fixture (committed in `854ed7caa`) were
/// `eigh_s` 0.157 / 0.970 / 7.013 at `p` = 256 / 512 / 1024 — a clean cubic and
/// 88.1% of the whole report at the widest cell.
#[test]
fn diagnostics_report_no_longer_grows_cubically_in_the_output_dimension() {
    let (n, k_atoms) = (48usize, 4usize);
    let mut timings: Vec<(usize, f64)> = Vec::new();
    println!("\n#2757: fit_diagnostics_report on the structured curvature (n={n}, K={k_atoms})");
    for &p in &[256usize, 512, 1024] {
        let term = planted_term(n, p, k_atoms, true);
        let rho = unit_rho(k_atoms);
        let fitted = term
            .try_fitted_target_aware(Array2::<f64>::zeros((n, p)).view(), Some(&rho))
            .expect("fitted");
        let metric = term.diagnostic_metric().expect("metric");
        let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
        let curvature = term
            .residual_gauge_streamed_data_curvature(
                &metric,
                &layout,
                Array2::<f64>::zeros((0, layout.param_dim())),
            )
            .expect("streamed curvature");
        assert_eq!(curvature.structure_tag(), "output_block_roots");
        assert_eq!(curvature.stored_scalars(), p * k_atoms * k_atoms);

        let started = Instant::now();
        let report = term
            .fit_diagnostics_report(None, false, None, fitted.view(), None)
            .expect("diagnostics report");
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "  p={p:>5} param_dim={:>6} report {seconds:>8.3}s  pinning_rank={}",
            p * k_atoms,
            report.residual_gauge.pinning_rank
        );
        timings.push((p, seconds));
    }
    // Cubic in `param_dim ∝ p` is 8x per doubling; the dense path measured 6.2x
    // and 7.2x here. The structured path is linear in `p` at fixed `D`, so a
    // generous 4x ceiling separates the two regimes without turning host noise
    // into a red test.
    for pair in timings.windows(2) {
        let (p_lo, t_lo) = pair[0];
        let (p_hi, t_hi) = pair[1];
        let growth = t_hi / t_lo.max(1.0e-6);
        assert!(
            growth <= 4.0,
            "doubling p from {p_lo} to {p_hi} multiplied the report by {growth:.2}x; \
             the certification cost must not be cubic in the output dimension"
        );
    }
}
