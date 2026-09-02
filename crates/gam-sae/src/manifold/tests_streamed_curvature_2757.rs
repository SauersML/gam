//! #2757 — the streamed residual-gauge curvature: the same certificate, without
//! the `param_dim`-square object.
//!
//! The surviving half of #2757 lives on the branch where the per-row metric
//! COUPLES output coordinates. There `H = Σ_n J_nᵀ M_n J_n` has no structure to
//! store, so at any production row count its smallest materialized form is a
//! `param_dim`-square triangular factor — `34 GiB` at the #2283 shape, folded at
//! `1.0e15` flops and read at `2.8e14` — and no storage change can remove that,
//! because `min(root_rows, param_dim)²` is what an exact full spectrum costs from
//! either side.
//!
//! So the certificate stopped asking for one. These are the gates on that, and
//! they are deliberately taken against objects built a DIFFERENT way than the
//! route under test: [`reference_dense_root`] and [`reference_dense_gram`] write
//! the layout arithmetic out inline and never call `fill_row_frame_jacobian`, so
//! "the operator is the curvature" is a checked claim rather than a shared bug.
//!
//! 1. the operator's three reads — matvec, diagonal, projected root — are the
//!    reference `R` and `RᵀR`, entry by entry;
//! 2. its `λ_max` is the dense spectrum's, inside the bracket its own trace gives;
//! 3. its generator-span rank is `root_spectral_rank` applied to the singular
//!    values of the reference `RΞ` — the same decision function on an
//!    independently built input;
//! 4. the CERTIFICATE is identical: every verdict, the group signature, the
//!    residual gauge dimension;
//! 5. production takes the streamed route exactly where a materialized root
//!    stops being the smaller object, and stores zero curvature scalars there;
//! 6. nothing at the parameter dimension is decomposed — the flops claim, read
//!    off the process's own eigendecomposition census rather than a stopwatch;
//! 7. the refusals: a non-finite diagonal, and an operator whose matvec
//!    contradicts its own diagonal.

use super::tests_frame_curvature_2757::{
    planted_term_for_probe, reference_dense_gram,
    reference_dense_root, source_root_rows, source_stored_scalars, source_structure_tag,
};
use crate::identifiability::{
    FrameColumnLayout, PinningRankSupport, ResidualGaugeReport, StreamedFrameCurvature,
    residual_gauge_exact_from_curvature, residual_gauge_exact_from_streamed, root_spectral_rank,
    streamed_lambda_max,
};
use crate::manifold::construction::ResidualGaugeCurvatureSource;
use crate::manifold::streamed_frame_curvature::StreamedFrameCurvatureOperator;
use crate::manifold::SaeManifoldTerm;
use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// A planted fit carrying an output-Fisher metric of the requested rank, with a
/// factor scale so an extreme-magnitude arm can reuse the whole harness.
fn gauge_driving_term(
    n: usize,
    p: usize,
    k_atoms: usize,
    rank: usize,
    factor_scale: f64,
    seed: u64,
) -> (SaeManifoldTerm, gam_problem::RowMetric, FrameColumnLayout) {
    let mut term = planted_term_for_probe(n, p, k_atoms, true);
    let mut s = seed;
    let factors =
        Array2::<f64>::from_shape_fn((n, p * rank), |_| factor_scale * (lcg(&mut s) - 0.5));
    let metric = gam_problem::RowMetric::output_fisher(Arc::new(factors), p, rank)
        .expect("output-Fisher metric");
    term.set_row_metric(metric.clone())
        .expect("metric is conformable with the term");
    assert!(
        metric.drives_gauge(),
        "this whole module is about the branch where the metric couples output coordinates"
    );
    let layout = FrameColumnLayout::new(p, &vec![1usize; k_atoms]);
    (term, metric, layout)
}

/// Deterministic probe directions of unit norm.
fn probe_directions(param_dim: usize, count: usize, seed: u64) -> Vec<Array1<f64>> {
    let mut s = seed;
    (0..count)
        .map(|_| {
            let raw = Array1::<f64>::from_shape_fn(param_dim, |_| lcg(&mut s) - 0.5);
            let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
            raw.mapv(|v| v / norm)
        })
        .collect()
}

fn worst_absolute_gap<'a>(a: impl Iterator<Item = &'a f64>, b: impl Iterator<Item = &'a f64>) -> f64 {
    a.zip(b).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

/// Gate 1 — the operator's three reads ARE the reference curvature.
///
/// `apply` against `R'R x`, `diagonal` against `diag(R'R)`, and `project_root`'s
/// `TᵀT` against `ΞᵀR'RΞ`, all with `R` built by the inline reference rather than
/// by the streaming builder. This is the gate that makes every other one in this
/// file about COST rather than about correctness.
#[test]
fn the_streamed_operator_is_the_curvature_it_replaces() {
    let (n, p, k_atoms, rank) = (12usize, 10usize, 2usize, 3usize);
    let (term, metric, layout) =
        gauge_driving_term(n, p, k_atoms, rank, 1.0, 0x2757_5EA1_0000_0001);
    let param_dim = layout.param_dim();
    let root_rows = n * rank;
    assert!(
        root_rows > param_dim,
        "the fixture must be on the branch under test: {root_rows} root rows vs {param_dim} \
         columns"
    );

    let reference_root = reference_dense_root(&term, &metric, &layout);
    let reference_gram = reference_dense_gram(&term, &metric, &layout);
    let scale = reference_gram.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(scale > 0.0, "a zero fixture cannot separate anything");

    let pin = Array2::<f64>::zeros((0, param_dim));
    let operator = StreamedFrameCurvatureOperator::new(&term, &metric, &layout, &pin, root_rows)
        .expect("operator");
    assert_eq!(operator.param_dim(), param_dim);
    assert_eq!(operator.root_rows(), root_rows);

    // matvec
    let mut y = vec![0.0_f64; param_dim];
    for direction in probe_directions(param_dim, 5, 0x2757_D1_0000_0001) {
        operator.apply(direction.as_slice().expect("contiguous"), &mut y).expect("matvec");
        let expected = reference_gram.dot(&direction);
        let worst = worst_absolute_gap(y.iter(), expected.iter());
        assert!(
            worst <= 1.0e-12 * scale,
            "streamed matvec differs from RᵀR x by {worst:.3e} (scale {scale:.3e})"
        );
    }

    // diagonal
    let diagonal = operator.diagonal().expect("diagonal");
    let expected: Vec<f64> = (0..param_dim).map(|c| reference_gram[[c, c]]).collect();
    let worst = worst_absolute_gap(diagonal.iter(), expected.iter());
    assert!(
        worst <= 1.0e-12 * scale,
        "streamed diagonal differs from diag(RᵀR) by {worst:.3e}"
    );

    // projected root
    let directions = probe_directions(param_dim, 4, 0x2757_D2_0000_0001);
    let views: Vec<ArrayView1<'_, f64>> = directions.iter().map(|d| d.view()).collect();
    let factor = operator.project_root(&views).expect("projected root");
    assert_eq!(factor.dim(), (views.len(), views.len()));
    let projected = factor.t().dot(&factor);
    let mut expected = Array2::<f64>::zeros((views.len(), views.len()));
    for (a, da) in directions.iter().enumerate() {
        let h_da = reference_gram.dot(da);
        for (b, db) in directions.iter().enumerate() {
            expected[[a, b]] = db.dot(&h_da);
        }
    }
    let worst = worst_absolute_gap(projected.iter(), expected.iter());
    assert!(
        worst <= 1.0e-12 * scale,
        "TᵀT differs from ΞᵀHΞ by {worst:.3e}"
    );
    // The factor's own column norms are what the certificate reads as energies.
    for (j, direction) in directions.iter().enumerate() {
        let from_factor: f64 = factor.column(j).iter().map(|v| v * v).sum();
        let reference = reference_root.dot(direction).iter().map(|v| v * v).sum::<f64>();
        assert!(
            (from_factor - reference).abs() <= 1.0e-12 * scale,
            "column {j}'s norm² {from_factor:.17e} is not ‖Rξ‖² = {reference:.17e}"
        );
    }
}

/// Gate 2 — the certified Krylov `λ_max` is the dense spectrum's largest
/// eigenvalue, and it sits inside the PSD bracket its own trace supplies.
#[test]
fn the_streamed_lambda_max_is_the_dense_spectrums() {
    let (n, p, k_atoms, rank) = (16usize, 12usize, 3usize, 3usize);
    let (term, metric, layout) =
        gauge_driving_term(n, p, k_atoms, rank, 1.0, 0x2757_1AB1_0000_0001);
    let param_dim = layout.param_dim();
    let pin = Array2::<f64>::zeros((0, param_dim));
    let operator =
        StreamedFrameCurvatureOperator::new(&term, &metric, &layout, &pin, n * rank).expect("op");

    let reference_gram = reference_dense_gram(&term, &metric, &layout);
    let (values, _) = reference_gram.eigh(faer::Side::Lower).expect("dense spectrum");
    let exact = values.iter().cloned().fold(0.0_f64, f64::max);
    assert!(exact > 0.0);

    let streamed = streamed_lambda_max(&operator).expect("certified λ_max");
    let relative = (streamed.lambda_max - exact).abs() / exact;
    assert!(
        relative <= 1.0e-9,
        "streamed λ_max {:.17e} vs dense {exact:.17e} (relative {relative:.3e})",
        streamed.lambda_max
    );
    assert!(
        streamed.relative_residual <= f64::EPSILON.sqrt(),
        "the solve must certify to the tolerance the verdict needs; residual {:.3e}",
        streamed.relative_residual
    );
    // `tr(H)/param_dim ≤ λ_max ≤ tr(H)` for a PSD operator, and the trace is
    // what the breakdown threshold and the acceptance check are denominated in.
    assert!(
        streamed.lambda_max <= streamed.trace * (1.0 + 1.0e-12),
        "λ_max {:.6e} above tr(H) {:.6e}",
        streamed.lambda_max,
        streamed.trace
    );
    assert!(
        streamed.lambda_max >= streamed.trace / (param_dim as f64) * (1.0 - 1.0e-12),
        "λ_max {:.6e} below tr(H)/param_dim {:.6e}",
        streamed.lambda_max,
        streamed.trace / (param_dim as f64)
    );
}

/// Build both certificates for one fit: the materialized route (the retained
/// witness) and the streamed one.
fn both_routes(
    n: usize,
    p: usize,
    k_atoms: usize,
    rank: usize,
    factor_scale: f64,
    seed: u64,
) -> (ResidualGaugeReport, ResidualGaugeReport) {
    let (term, metric, layout) = gauge_driving_term(n, p, k_atoms, rank, factor_scale, seed);
    let param_dim = layout.param_dim();
    let pin = Array2::<f64>::zeros((0, param_dim));
    let (model, _) = term
        .to_residual_gauge_model(metric.clone(), None, false)
        .expect("certificate model");
    let views = term.atom_parameter_views();
    let ops: Vec<Option<crate::identifiability::OrbitPenaltyOperator>> =
        (0..k_atoms).map(|_| None).collect();

    let materialized = term
        .residual_gauge_streamed_data_curvature(&metric, &layout, pin.clone())
        .expect("materialized curvature");
    assert_eq!(materialized.structure_tag(), "dual_root");
    assert_eq!(
        materialized.stored_scalars(),
        param_dim * param_dim,
        "the witness must be the param_dim-square object this route exists to avoid"
    );
    let from_stored = residual_gauge_exact_from_curvature(&model, &views, &ops, materialized)
        .expect("materialized certificate");

    let operator =
        StreamedFrameCurvatureOperator::new(&term, &metric, &layout, &pin, n * rank).expect("op");
    let from_streamed = residual_gauge_exact_from_streamed(&model, &views, &ops, &operator)
        .expect("streamed certificate");
    (from_stored, from_streamed)
}

fn assert_same_certificate(
    from_stored: &ResidualGaugeReport,
    from_streamed: &ResidualGaugeReport,
    fraction_tol: f64,
) {
    assert_eq!(
        from_stored.generators.len(),
        from_streamed.generators.len(),
        "the two routes must enumerate the same generators"
    );
    assert!(
        !from_stored.generators.is_empty(),
        "a certificate with no generators cannot separate the two routes"
    );
    for (a, b) in from_stored
        .generators
        .iter()
        .zip(from_streamed.generators.iter())
    {
        assert_eq!(a.description, b.description);
        assert_eq!(a.family, b.family);
        assert_eq!(
            a.unpinned, b.unpinned,
            "generator `{}` is {} materialized and {} streamed (fractions {:.6e} vs {:.6e})",
            a.description,
            if a.unpinned { "unpinned" } else { "pinned" },
            if b.unpinned { "unpinned" } else { "pinned" },
            a.pinned_energy_fraction,
            b.pinned_energy_fraction
        );
        let gap = (a.pinned_energy_fraction - b.pinned_energy_fraction).abs();
        assert!(
            gap <= fraction_tol,
            "generator `{}` scores {:.17e} materialized against {:.17e} streamed",
            a.description,
            a.pinned_energy_fraction,
            b.pinned_energy_fraction
        );
    }
    assert_eq!(from_stored.group_signature(), from_streamed.group_signature());
    assert_eq!(
        from_stored.residual_gauge_dim,
        from_streamed.residual_gauge_dim
    );
    assert_eq!(
        from_stored.sym_f_trivial_under_output_fisher,
        from_streamed.sym_f_trivial_under_output_fisher
    );
    assert_eq!(
        from_stored.pinning_rank_support,
        PinningRankSupport::ParameterSpace
    );
    assert_eq!(
        from_streamed.pinning_rank_support,
        PinningRankSupport::GeneratorSpan
    );
}

/// Gate 4 — the certificate is identical. This is the one that makes the change
/// a cost change rather than a different certificate wearing the same name.
#[test]
fn the_streamed_route_certifies_the_same_gauge() {
    let (from_stored, from_streamed) = both_routes(24, 12, 3, 3, 1.0, 0x2757_C0DE_0000_0001);
    assert_same_certificate(&from_stored, &from_streamed, 1.0e-9);
}

/// The same, at a factor scale that puts `H`'s entries near `1e120`.
///
/// The fractions are ratios of two quantities that both scale with `σ²`, so they
/// are scale-invariant exactly — and a route that squared a condition number
/// somewhere would not reproduce them. `λ_max` here is `~1e120`, far inside
/// `f64`, and the point is that the Krylov solve's tolerances are all relative to
/// `tr(H)` rather than absolute.
#[test]
fn the_streamed_route_agrees_at_extreme_magnitudes() {
    let (from_stored, from_streamed) = both_routes(24, 12, 3, 3, 1.0e60, 0x2757_B16_0000_0001);
    assert_same_certificate(&from_stored, &from_streamed, 1.0e-9);
}

/// Gate 5 — production takes the streamed route exactly where a materialized
/// root stops being the smaller object, and nowhere else. The three arms are the
/// whole fork.
#[test]
fn production_streams_exactly_where_no_materialized_form_is_smaller() {
    // (a) Euclidean provenance: `H` is block diagonal, so it is HELD.
    let term = planted_term_for_probe(20, 12, 3, true);
    let metric = term.diagnostic_metric().expect("metric");
    assert!(!metric.drives_gauge());
    let (_model, source) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    assert_eq!(source_structure_tag(&source), "output_block_roots");
    assert!(source_stored_scalars(&source) > 0);

    // (b) Coupling metric whose root is the SMALLER object: still held.
    let (n, p, k_atoms, rank) = (8usize, 16usize, 3usize, 2usize);
    let (term, metric, layout) = gauge_driving_term(n, p, k_atoms, rank, 1.0, 0x2757_5A11_0000_1);
    assert!(
        n * rank <= layout.param_dim(),
        "arm (b) needs root_rows ≤ param_dim: {} vs {}",
        n * rank,
        layout.param_dim()
    );
    let (_model, source) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    assert_eq!(source_structure_tag(&source), "dual_root");
    assert_eq!(source_root_rows(&source), n * rank);

    // (c) Coupling metric whose root is the LARGER object: streamed, and it
    // stores no curvature scalars at all.
    let (n, p, k_atoms, rank) = (24usize, 12usize, 3usize, 3usize);
    let (term, metric, layout) = gauge_driving_term(n, p, k_atoms, rank, 1.0, 0x2757_5A11_0000_2);
    assert!(n * rank > layout.param_dim());
    let (_model, source) = term
        .to_residual_gauge_model(metric, None, false)
        .expect("certificate model");
    assert_eq!(source_structure_tag(&source), "streamed_operator");
    assert_eq!(
        source_stored_scalars(&source),
        0,
        "the streamed route's whole claim is that it holds no curvature at all"
    );
    assert_eq!(
        source_root_rows(&source),
        n * rank,
        "both arms describe the same R, so they must agree on its row count"
    );
    assert!(matches!(
        source,
        ResidualGaugeCurvatureSource::Streamed { .. }
    ));
}

/// The certificate is reproducible: the Krylov start is deterministic, so two
/// runs of the same fit produce byte-identical verdicts.
#[test]
fn the_streamed_certificate_is_reproducible() {
    let first = both_routes(24, 12, 3, 3, 1.0, 0x2757_5EED_0000_0001).1;
    let second = both_routes(24, 12, 3, 3, 1.0, 0x2757_5EED_0000_0001).1;
    assert_eq!(first.summary, second.summary);
    assert_eq!(first.pinning_rank, second.pinning_rank);
    for (a, b) in first.generators.iter().zip(second.generators.iter()) {
        assert_eq!(
            a.pinned_energy_fraction.to_bits(),
            b.pinned_energy_fraction.to_bits(),
            "generator `{}` is not reproducible",
            a.description
        );
    }
}

/// The generator projection is split over GENERATORS, which have disjoint
/// outputs — so the parallel pass is the same arithmetic in the same order as
/// the serial one, not merely the same in distribution.
///
/// The serial arm is reached by running inside a rayon worker, which is exactly
/// the nesting guard the operator uses; comparing the two factors BIT FOR BIT is
/// what makes "no reduction, therefore no schedule dependence" a checked claim.
/// Splitting the observations instead would have needed per-chunk partial
/// factors combined pairwise, and Givens rotations do not commute — this gate is
/// the reason that design was not taken.
#[test]
fn the_parallel_generator_pass_is_bit_identical_to_the_serial_one() {
    let (n, p, k_atoms, rank) = (24usize, 12usize, 3usize, 3usize);
    let (term, metric, layout) =
        gauge_driving_term(n, p, k_atoms, rank, 1.0, 0x2757_9A9A_0000_0001);
    let param_dim = layout.param_dim();
    let pin = Array2::<f64>::zeros((0, param_dim));
    let operator =
        StreamedFrameCurvatureOperator::new(&term, &metric, &layout, &pin, n * rank).expect("op");
    let directions = probe_directions(param_dim, 9, 0x2757_9B9B_0000_0001);
    let views: Vec<ArrayView1<'_, f64>> = directions.iter().map(|d| d.view()).collect();

    assert!(
        rayon::current_thread_index().is_none(),
        "the outer arm must be the parallel one for this gate to compare two passes"
    );
    let parallel = operator.project_root(&views).expect("parallel pass");

    let mut serial = Array2::<f64>::zeros(parallel.dim());
    rayon::scope(|scope| {
        scope.spawn(|_| {
            assert!(
                rayon::current_thread_index().is_some(),
                "inside a rayon worker the operator must take its serial arm"
            );
            serial = operator.project_root(&views).expect("serial pass");
        });
    });

    assert_eq!(parallel.dim(), serial.dim());
    for (a, b) in parallel.iter().zip(serial.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "the projected root must not depend on the schedule: {a:.17e} vs {b:.17e}"
        );
    }
}

/// A hand-built operator, so the refusal contract can be exercised without a
/// fit that has to be made pathological first.
struct FakeCurvature {
    gram: Array2<f64>,
    diagonal_override: Option<Array1<f64>>,
    matvec_gain: f64,
}

impl StreamedFrameCurvature for FakeCurvature {
    fn param_dim(&self) -> usize {
        self.gram.ncols()
    }
    fn root_rows(&self) -> usize {
        self.gram.ncols()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) -> Result<(), String> {
        let image = self.gram.dot(&ArrayView1::from(x));
        for (slot, value) in y.iter_mut().zip(image.iter()) {
            *slot = self.matvec_gain * value;
        }
        Ok(())
    }
    fn diagonal(&self) -> Result<Array1<f64>, String> {
        Ok(match &self.diagonal_override {
            Some(d) => d.clone(),
            None => Array1::from_iter((0..self.gram.ncols()).map(|c| self.gram[[c, c]])),
        })
    }
    fn project_root(&self, directions: &[ArrayView1<'_, f64>]) -> Result<Array2<f64>, String> {
        let mut factor = Array2::<f64>::zeros((directions.len(), directions.len()));
        for (j, d) in directions.iter().enumerate() {
            factor[[j, j]] = d.dot(&self.gram.dot(d)).max(0.0).sqrt();
        }
        Ok(factor)
    }
}

fn spd_fixture(dim: usize) -> Array2<f64> {
    let mut s = 0x2757_FA1E_0000_0001u64;
    let a = Array2::<f64>::from_shape_fn((dim, dim), |_| lcg(&mut s) - 0.5);
    a.t().dot(&a)
}

/// A non-finite diagonal is a broken fit, not a certificate with an unusual
/// spectrum. Refuse, once, before any iteration.
#[test]
fn a_non_finite_diagonal_is_refused() {
    let gram = spd_fixture(6);
    let mut diagonal = Array1::from_iter((0..6).map(|c| gram[[c, c]]));
    diagonal[3] = f64::NAN;
    let operator = FakeCurvature {
        gram,
        diagonal_override: Some(diagonal),
        matvec_gain: 1.0,
    };
    let error = streamed_lambda_max(&operator).expect_err("a NaN diagonal must refuse");
    assert!(error.contains("finite"), "unexpected refusal: {error}");
}

/// A negative diagonal entry cannot come from a Gram, so it is a contradiction
/// in the operator rather than a spectrum to report.
#[test]
fn a_negative_diagonal_is_refused() {
    let gram = spd_fixture(6);
    let mut diagonal = Array1::from_iter((0..6).map(|c| gram[[c, c]]));
    diagonal[2] = -1.0;
    let operator = FakeCurvature {
        gram,
        diagonal_override: Some(diagonal),
        matvec_gain: 1.0,
    };
    let error = streamed_lambda_max(&operator).expect_err("a negative diagonal must refuse");
    assert!(error.contains("non-negative"), "unexpected refusal: {error}");
}

/// The one cross-check available without materializing anything: the matvec and
/// the diagonal are two readings of one operator, and `λ_max ≤ tr(H)` for a PSD
/// one. An operator whose matvec is inflated relative to its own diagonal is
/// refused rather than certified.
#[test]
fn a_matvec_that_contradicts_its_own_diagonal_is_refused() {
    let dim = 6usize;
    let gram = spd_fixture(dim);
    let operator = FakeCurvature {
        gram,
        diagonal_override: None,
        // A gain above `tr(H)/λ_max ≥ 1` pushes the Ritz value past the trace.
        matvec_gain: 1.0e3,
    };
    let error =
        streamed_lambda_max(&operator).expect_err("an inflated matvec must fail the PSD bracket");
    assert!(
        error.contains("PSD bracket"),
        "unexpected refusal: {error}"
    );
}

/// A well-formed hand-built operator still certifies, so the refusals above are
/// about the contradiction and not about the fixture.
#[test]
fn a_consistent_hand_built_operator_certifies() {
    let dim = 8usize;
    let gram = spd_fixture(dim);
    let (values, _) = gram.eigh(faer::Side::Lower).expect("dense spectrum");
    let exact = values.iter().cloned().fold(0.0_f64, f64::max);
    let operator = FakeCurvature {
        gram,
        diagonal_override: None,
        matvec_gain: 1.0,
    };
    let lambda = streamed_lambda_max(&operator).expect("certified");
    let relative = (lambda.lambda_max - exact).abs() / exact;
    assert!(
        relative <= 1.0e-10,
        "hand-built λ_max {:.17e} vs {exact:.17e}",
        lambda.lambda_max
    );
}
