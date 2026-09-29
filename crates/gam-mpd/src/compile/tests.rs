#![cfg(test)]
//! Known-answer tests of the native edit compiler.

use ndarray::{Array2, ArrayView1, ArrayView2, array, s};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::super::lift::{TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::super::test_support::test_governor;
use super::linear::{
    ConstraintSide, Coverage, EditMetric, LinearSiteProblem, LinearWitness, Requirement, ResponseClass,
    compile_linear_site,
};
use super::ties::{BlockRef, TieConstraint, check_ties};
use super::{CompiledParameterEdit, ControlRealization, DescriptiveReason, NativeEditPlan};

fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
}

fn dense(plan: &NativeEditPlan, storage: &str, shape: (usize, usize)) -> Array2<f64> {
    plan.edits()
        .iter()
        .find(|edit| edit.storage.0 == storage)
        .map_or_else(|| Array2::zeros(shape), |edit| edit.delta.left().dot(&edit.delta.right().t()))
}

fn max_abs(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// A registry with one `rows × cols` storage read by one identity linear use.
fn single_use(rows: usize, cols: usize, seed: u64) -> (TensorRegistry, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let weight = uniform(&mut rng, rows, cols);
    let mut registry = TensorRegistry::default();
    registry
        .register_storage(TensorId("w".into()), weight.view().into_dyn())
        .expect("storage");
    registry
        .register_use_site(UseSiteId("w#0".into()), TensorId("w".into()), UseMap::Linear(TieOrientation::Identity))
        .expect("use");
    (registry, weight)
}

fn linear<'a>(
    inputs: ArrayView2<'a, f64>,
    targets: ArrayView2<'a, f64>,
    class: ResponseClass<'a>,
) -> Requirement<'a> {
    Requirement::Linear {
        site: UseSiteId("w#0".into()),
        inputs,
        targets,
        target_radius: 0.0,
        class,
    }
}

fn problem<'a>(
    registry: &'a TensorRegistry,
    native: ArrayView2<'a, f64>,
    requirements: Vec<Requirement<'a>>,
) -> LinearSiteProblem<'a> {
    LinearSiteProblem {
        registry,
        storage: TensorId("w".into()),
        native,
        requirements,
        metric: EditMetric::frobenius(),
        ties: Vec::new(),
        off_target: Vec::new(),
    }
}

/// The set-type target that asks for `changes` on top of the native response.
fn set_targets(inputs: &Array2<f64>, weight: &Array2<f64>, changes: &Array2<f64>) -> Array2<f64> {
    inputs.dot(&weight.t()) + changes
}

#[test]
fn spanning_inputs_recover_the_target_map_exactly() {
    let (registry, weight) = single_use(3, 4, 1);
    let mut rng = StdRng::seed_from_u64(2);
    let target = uniform(&mut rng, 3, 4);
    // Six inputs in R^4: they span, and the kernel of X is two-dimensional but consistent.
    let inputs = uniform(&mut rng, 6, 4);
    let changes = inputs.dot(&target.t());
    let targets = set_targets(&inputs, &weight, &changes);
    let report = compile_linear_site(
        &problem(&registry, weight.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::AllInputs)]),
        "gain",
        test_governor(),
    )
    .expect("compiles");
    assert_eq!(report.coverage, vec![Coverage::Covered]);
    assert_eq!(report.right_rank, 4);
    let ControlRealization::ExactlyRealized { residual, .. } = &report.compiled.realization else {
        panic!("a spanning consistent sample is exactly realized: {:?}", report.compiled.realization);
    };
    let EvidenceStatus::Exact { value, numerical_error, basis, .. } = residual else {
        panic!("exact residual");
    };
    assert_eq!(*basis, ExactBasis::Algebraic);
    assert!(*value <= numerical_error + report.allowance[0], "{value} {numerical_error}");
    let plan = report.compiled.plan.as_ref().expect("plan");
    let edit = dense(plan, "w", (3, 4));
    assert!(max_abs((&edit - &target).view()) < 1e-12, "the unique solution is the target map");
    assert!(report.undeclared_uses.is_empty());
    let changes_record = plan.intervention_changes();
    assert_eq!(changes_record.len(), 1);
}

#[test]
fn a_duplicated_input_with_two_demands_is_infeasible_with_its_kernel_witness() {
    let (registry, weight) = single_use(2, 3, 3);
    let inputs = array![[1.0, 2.0, -1.0], [0.5, 0.0, 1.0], [1.0, 2.0, -1.0]];
    let changes = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
    let targets = set_targets(&inputs, &weight, &changes);
    let report = compile_linear_site(
        &problem(&registry, weight.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::Sample)]),
        "gain",
        test_governor(),
    )
    .expect("compiles");
    assert!(report.compiled.plan.is_none());
    let ControlRealization::Descriptive {
        reason: DescriptiveReason::KernelViolation,
        witness: Some(EvidenceStatus::Counterexample { witness, value, .. }),
        ..
    } = &report.compiled.realization
    else {
        panic!("kernel violation expected: {:?}", report.compiled.realization);
    };
    let LinearWitness::Kernel {
        side,
        direction,
        input_norm_upper,
        edit_norm_lower_bound,
    } = witness
    else {
        panic!("kernel witness");
    };
    assert_eq!(*side, ConstraintSide::Right);
    // v ∝ e0 − e2 up to sign: the two copies of one input.
    assert!((direction[0] + direction[2]).abs() < 1e-12 && direction[1].abs() < 1e-12);
    assert!((value - 2.0_f64.sqrt()).abs() < 1e-12, "‖Y v‖ = |1 − (−1)|/√2");
    assert!(*input_norm_upper < 1e-14);
    assert!(*edit_norm_lower_bound > 1e12);
}

#[test]
fn an_undercovered_class_is_only_empirical_and_names_a_missing_direction() {
    let (registry, weight) = single_use(3, 4, 4);
    let mut rng = StdRng::seed_from_u64(5);
    let target = uniform(&mut rng, 3, 4);
    let inputs = uniform(&mut rng, 2, 4);
    let changes = inputs.dot(&target.t());
    let targets = set_targets(&inputs, &weight, &changes);
    let report = compile_linear_site(
        &problem(&registry, weight.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::AllInputs)]),
        "gain",
        test_governor(),
    )
    .expect("compiles");
    let Coverage::Uncovered { dimension, direction } = &report.coverage[0] else {
        panic!("uncovered expected");
    };
    assert_eq!(*dimension, 2);
    let reach: f64 = inputs.rows().into_iter().map(|row| row.iter().zip(direction).map(|(a, b)| a * b).sum::<f64>().abs()).fold(0.0, f64::max);
    assert!(reach < 1e-12, "the named direction is orthogonal to every sampled input");
    assert!(matches!(report.compiled.realization, ControlRealization::EmpiricallyValidated { .. }));
    // Minimum norm: the edit vanishes on the complement of the sampled span.
    let edit = dense(report.compiled.plan.as_ref().expect("plan"), "w", (3, 4));
    let off_span = edit.dot(&ndarray::Array1::from(direction.clone()));
    assert!(off_span.iter().all(|v| v.abs() < 1e-12));
    // A declared span inside the sample is covered.
    let inside = inputs.slice(s![..1, ..]).to_owned();
    let spanned = compile_linear_site(
        &problem(&registry, weight.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::Span(inside.view()))]),
        "gain",
        test_governor(),
    )
    .expect("compiles");
    assert_eq!(spanned.coverage, vec![Coverage::Covered]);
    assert!(matches!(spanned.compiled.realization, ControlRealization::ExactlyRealized { .. }));
}

#[test]
fn the_weighted_metric_gives_its_closed_form_minimum() {
    let (registry, weight) = single_use(2, 3, 6);
    let inputs = array![[1.0, 1.0, 0.0]];
    let changes = array![[1.0, -2.0]];
    let targets = set_targets(&inputs, &weight, &changes);
    let col_scale = vec![1.0, 3.0, 2.0];
    let mut declared = problem(&registry, weight.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::Sample)]);
    declared.metric = EditMetric::weighted(None, Some(col_scale.clone())).expect("metric");
    let report = compile_linear_site(&declared, "gain", test_governor()).expect("compiles");
    let edit = dense(report.compiled.plan.as_ref().expect("plan"), "w", (2, 3));
    // min ‖ΔW D_c‖_F s.t. ΔW x = y: ΔW = y (D_c⁻² x)ᵀ / (xᵀ D_c⁻² x).
    let weights = [1.0, 1.0 / 9.0, 0.0];
    let denominator = 1.0 + 1.0 / 9.0;
    for row in 0..2 {
        for col in 0..3 {
            let expected = changes[[0, row]] * weights[col] / denominator;
            assert!((edit[[row, col]] - expected).abs() < 1e-14, "{row} {col}");
        }
    }
    let (norm, band) = report.metric_norm.expect("norm");
    let direct: f64 = (0..2)
        .flat_map(|r| (0..3).map(move |c| (r, c)))
        .map(|(r, c)| (edit[[r, c]] * col_scale[c]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - direct).abs() <= band + 1e-15);
}

#[test]
fn zero_requests_compile_to_the_native_plan() {
    let (registry, weight) = single_use(2, 2, 7);
    let inputs = array![[1.0, 0.0], [0.0, 1.0]];
    let changes = Array2::<f64>::zeros((2, 2));
    let targets = set_targets(&inputs, &weight, &changes);
    let report = compile_linear_site(
        &problem(&registry, weight.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::AllInputs)]),
        "gain",
        test_governor(),
    )
    .expect("compiles");
    assert!(report.compiled.plan.as_ref().expect("plan").is_native(), "ρ(0) = θ");
}

#[test]
fn renaming_rescaling_and_duplicating_observations_leave_the_edit_unchanged() {
    let (registry, weight) = single_use(3, 5, 8);
    let mut rng = StdRng::seed_from_u64(9);
    let target = uniform(&mut rng, 3, 5);
    let inputs = uniform(&mut rng, 3, 5);
    let changes = inputs.dot(&target.t());
    let compile = |x: &Array2<f64>, y: &Array2<f64>| {
        let targets = set_targets(x, &weight, y);
        let report = compile_linear_site(
            &problem(&registry, weight.view(), vec![linear(x.view(), targets.view(), ResponseClass::Sample)]),
            "gain",
            test_governor(),
        )
        .expect("compiles");
        dense(report.compiled.plan.as_ref().expect("plan"), "w", (3, 5))
    };
    let base = compile(&inputs, &changes);
    let order = [2, 0, 1];
    let renamed = compile(&inputs.select(ndarray::Axis(0), &order), &changes.select(ndarray::Axis(0), &order));
    let mut scaled_x = inputs.clone();
    let mut scaled_y = changes.clone();
    for (row, factor) in [3.0, -0.5, 7.0].into_iter().enumerate() {
        scaled_x.row_mut(row).mapv_inplace(|v| v * factor);
        scaled_y.row_mut(row).mapv_inplace(|v| v * factor);
    }
    let rescaled = compile(&scaled_x, &scaled_y);
    let doubled = compile(
        &ndarray::concatenate![ndarray::Axis(0), inputs, inputs.slice(s![..1, ..])],
        &ndarray::concatenate![ndarray::Axis(0), changes, changes.slice(s![..1, ..])],
    );
    for other in [renamed, rescaled, doubled] {
        assert!(max_abs((&other - &base).view()) < 1e-12);
    }
}

/// A tied embedding: storage `emb` (`vocab × width`), read as rows by the lookup and as a
/// linear map by the output head through its alias.
fn tied_embedding() -> (TensorRegistry, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(10);
    let embedding = uniform(&mut rng, 4, 3);
    let mut registry = TensorRegistry::default();
    registry.register_storage(TensorId("emb".into()), embedding.view().into_dyn()).expect("storage");
    registry.register_alias(TensorId("head".into()), TensorId("emb".into())).expect("alias");
    registry
        .register_use_site(UseSiteId("emb#0".into()), TensorId("emb".into()), UseMap::Stored)
        .expect("lookup");
    registry
        .register_use_site(UseSiteId("emb#1".into()), TensorId("head".into()), UseMap::Linear(TieOrientation::Identity))
        .expect("head");
    (registry, embedding)
}

fn tied_problem<'a>(
    registry: &'a TensorRegistry,
    native: ArrayView2<'a, f64>,
    hidden: ArrayView2<'a, f64>,
    logits: ArrayView2<'a, f64>,
    held: ArrayView2<'a, f64>,
) -> LinearSiteProblem<'a> {
    LinearSiteProblem {
        registry,
        storage: TensorId("emb".into()),
        native,
        requirements: vec![
            Requirement::Linear {
                site: UseSiteId("emb#1".into()),
                inputs: hidden,
                targets: logits,
                target_radius: 0.0,
                class: ResponseClass::Sample,
            },
            Requirement::StoredRows {
                rows: vec![1],
                targets: held,
                target_radius: 0.0,
            },
        ],
        metric: EditMetric::frobenius(),
        ties: Vec::new(),
        off_target: Vec::new(),
    }
}

#[test]
fn a_tied_head_edit_moves_the_embedding_it_shares_and_refuses_to_untie() {
    let (registry, embedding) = tied_embedding();
    let hidden = array![[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]];
    // The head's logit of token 1 must not move while its embedding row is held at its
    // native value: consistent, so the compiled edit keeps row 1.
    let logits = array![[0.5, 0.0, -1.0, 0.25], [0.0, 0.0, 2.0, 1.0]];
    let logit_targets = set_targets(&hidden, &embedding, &logits);
    let held = embedding.slice(s![1..2, ..]).to_owned();
    let declared = |targets| tied_problem(&registry, embedding.view(), hidden.view(), targets, held.view());
    let report = compile_linear_site(&declared(logit_targets.view()), "head", test_governor()).expect("compiles");
    let plan = report.compiled.plan.as_ref().expect("feasible");
    let edit = dense(plan, "emb", (4, 3));
    assert!(edit.row(1).iter().all(|v| v.abs() < 1e-14), "the held embedding row stays");
    assert!(max_abs((edit.dot(&hidden.t()) - logits.t()).view()) < 1e-12);
    assert_eq!(plan.edits()[0].storage.0, "emb", "the edit acts on storage, so the alias moves with it");
    // Asking token 1's logit to move while its embedding row is held breaks the tie: the
    // compiler reports the incompatibility instead of editing the head alone.
    let moved = array![[0.5, 0.3, -1.0, 0.25], [0.0, 0.0, 2.0, 1.0]];
    let moved_targets = set_targets(&hidden, &embedding, &moved);
    let report = compile_linear_site(&declared(moved_targets.view()), "head", test_governor()).expect("compiles");
    assert!(report.compiled.plan.is_none());
    let ControlRealization::Descriptive {
        reason: DescriptiveReason::IncompatibleSides,
        witness: Some(EvidenceStatus::Counterexample { witness, .. }),
        ..
    } = &report.compiled.realization
    else {
        panic!("incompatible sides expected: {:?}", report.compiled.realization);
    };
    assert_eq!(*witness, LinearWitness::Compatibility { left: 0, right: 0 });
}

#[test]
fn an_edit_of_one_side_of_a_stored_twice_tie_is_reported_broken() {
    let mut rng = StdRng::seed_from_u64(11);
    let shared = uniform(&mut rng, 2, 3);
    let mut registry = TensorRegistry::default();
    registry.register_storage(TensorId("w".into()), shared.view().into_dyn()).expect("w");
    registry
        .register_storage(TensorId("w_copy".into()), shared.t().to_owned().view().into_dyn())
        .expect("copy");
    registry
        .register_use_site(UseSiteId("w#0".into()), TensorId("w".into()), UseMap::Linear(TieOrientation::Identity))
        .expect("use");
    let tie = TieConstraint {
        first: BlockRef { storage: TensorId("w".into()), rows: 0..2, cols: 0..3 },
        second: BlockRef { storage: TensorId("w_copy".into()), rows: 0..3, cols: 0..2 },
        transposed: true,
        scale: 1.0,
    };
    let inputs = array![[1.0, 0.0, 0.0]];
    let changes = array![[1.0, 1.0]];
    let targets = set_targets(&inputs, &shared, &changes);
    let mut declared = problem(&registry, shared.view(), vec![linear(inputs.view(), targets.view(), ResponseClass::Sample)]);
    declared.ties = vec![tie.clone()];
    let report = compile_linear_site(&declared, "gain", test_governor()).expect("compiles");
    assert!(matches!(
        report.compiled.realization,
        ControlRealization::Descriptive { reason: DescriptiveReason::TieBroken, .. }
    ));
    // Editing both sides together keeps the tie.
    let left = array![[1.0], [1.0]];
    let right = array![[1.0], [0.0], [0.0]];
    let plan = NativeEditPlan::new(
        &registry,
        vec![
            CompiledParameterEdit {
                storage: TensorId("w".into()),
                delta: super::super::apply::FactoredEdit::new(left.clone(), right.clone()).expect("edit"),
            },
            CompiledParameterEdit {
                storage: TensorId("w_copy".into()),
                delta: super::super::apply::FactoredEdit::new(right, left).expect("edit"),
            },
        ],
    )
    .expect("plan");
    assert!(check_ties(&plan, &[tie], test_governor()).expect("checked").is_empty());
}

#[test]
fn statuses_refuse_stronger_claims_than_their_evidence() {
    let sampled: EvidenceStatus<(), ()> =
        EvidenceStatus::exact(0.0, 0.0, ExactBasis::Exhaustive { cardinality: 3 }, None, ()).expect("status");
    assert!(ControlRealization::exactly_realized(sampled.clone()).is_err());
    assert!(ControlRealization::empirically_validated(sampled.clone()).is_ok());
    assert!(ControlRealization::descriptive(DescriptiveReason::CoupledControls, Some(sampled)).is_err());
    let estimate: EvidenceStatus<(), ()> = EvidenceStatus::statistical_estimate(0.1, 0.01, 10, ()).expect("status");
    assert!(ControlRealization::empirically_validated(estimate).is_err());
}

mod coupled_controls {
    use super::super::controls::{
        CoupledControlProblem, GainAxis, GainSite, ReaderHomogeneity, ReaderSite, SiteChoice, SupportNode,
        compile_coupled_controls,
    };
    use super::*;

    /// Units 0 and 1 carry control 0; unit 2 carries controls 1 and 2; unit 3 carries 3.
    fn writer() -> Array2<f64> {
        array![
            [0.7, 0.0, 0.0, 0.0],
            [-1.3, 0.0, 0.0, 0.0],
            [0.0, 0.4, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.9],
        ]
    }

    /// A down projection (`width × units`) and an up projection (`controls × width`).
    fn registry() -> (TensorRegistry, Array2<f64>, Array2<f64>) {
        let mut rng = StdRng::seed_from_u64(20);
        let down = uniform(&mut rng, 3, 4);
        let up = uniform(&mut rng, 4, 3);
        let mut registry = TensorRegistry::default();
        registry.register_storage(TensorId("down".into()), down.view().into_dyn()).expect("down");
        registry.register_storage(TensorId("up".into()), up.view().into_dyn()).expect("up");
        (registry, down, up)
    }

    fn down_site(down: &Array2<f64>) -> GainSite<'_> {
        GainSite {
            storage: TensorId("down".into()),
            weight: down.view(),
            axis: GainAxis::Columns,
        }
    }

    #[test]
    fn writer_gains_realize_a_rescaling_constant_on_coupling_classes() {
        let (registry, down, _) = registry();
        let a = writer();
        let alpha = [2.0, 3.0, 3.0, 0.5];
        let report = compile_coupled_controls(
            &CoupledControlProblem {
                registry: &registry,
                writer: a.view(),
                alpha: &alpha,
                writer_site: Some(down_site(&down)),
                reader_site: None,
            },
            "scale",
        )
        .expect("compiles");
        assert_eq!(report.writer_classes.len(), 3);
        assert_eq!(report.choice, Some(SiteChoice::Writer));
        assert_eq!(report.unit_gains.as_deref(), Some(&[2.0, 2.0, 3.0, 0.5][..]));
        let ControlRealization::ExactlyRealized {
            residual: EvidenceStatus::Exact { value, .. },
            ..
        } = &report.compiled.realization
        else {
            panic!("exact");
        };
        assert_eq!(*value, 0.0);
        let edit = dense(report.compiled.plan.as_ref().expect("plan"), "down", (3, 4));
        let expected = &down * &array![[1.0, 1.0, 2.0, -0.5]];
        assert!(max_abs((&edit - &expected).view()) < 1e-15, "ΔW = W diag(s − 1)");
    }

    #[test]
    fn a_split_request_inside_a_class_is_coupled_with_its_path() {
        let (registry, down, up) = registry();
        let a = writer();
        let alpha = [2.0, 3.0, 4.0, 1.0];
        let report = compile_coupled_controls(
            &CoupledControlProblem {
                registry: &registry,
                writer: a.view(),
                alpha: &alpha,
                writer_site: Some(down_site(&down)),
                reader_site: None,
            },
            "scale",
        )
        .expect("compiles");
        let ControlRealization::Descriptive {
            reason: DescriptiveReason::CoupledControls,
            witness: Some(EvidenceStatus::Counterexample { witness, .. }),
            ..
        } = &report.compiled.realization
        else {
            panic!("coupled: {:?}", report.compiled.realization);
        };
        assert_eq!(witness.path, vec![SupportNode::Control(1), SupportNode::Unit(2), SupportNode::Control(2)]);
        assert!(report.writer_residual.0 > report.writer_residual.1);
        // A linear reader (SwiGLU's up rows) realizes it alone.
        let reader = ReaderSite {
            site: GainSite {
                storage: TensorId("up".into()),
                weight: up.view(),
                axis: GainAxis::Rows,
            },
            homogeneity: ReaderHomogeneity::Linear,
        };
        let report = compile_coupled_controls(
            &CoupledControlProblem {
                registry: &registry,
                writer: a.view(),
                alpha: &alpha,
                writer_site: Some(down_site(&down)),
                reader_site: Some(reader),
            },
            "scale",
        )
        .expect("compiles");
        assert_eq!(report.choice, Some(SiteChoice::Reader));
        assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
        let edit = dense(report.compiled.plan.as_ref().expect("plan"), "up", (4, 3));
        let mut expected = up.clone();
        for (row, gain) in alpha.iter().enumerate() {
            expected.row_mut(row).mapv_inplace(|v| v * (gain - 1.0));
        }
        assert!(max_abs((&edit - &expected).view()) < 1e-15);
    }

    #[test]
    fn a_relu_reader_takes_magnitudes_and_the_writer_takes_signs() {
        let (registry, down, up) = registry();
        let a = writer();
        let sites = |alpha: &'static [f64]| CoupledControlProblem {
            registry: &registry,
            writer: a.view(),
            alpha,
            writer_site: Some(down_site(&down)),
            reader_site: Some(ReaderSite {
                site: GainSite {
                    storage: TensorId("up".into()),
                    weight: up.view(),
                    axis: GainAxis::Rows,
                },
                homogeneity: ReaderHomogeneity::PositivelyHomogeneous,
            }),
        };
        let report = compile_coupled_controls(&sites(&[-1.0, 2.0, 3.0, 1.0]), "scale").expect("compiles");
        assert_eq!(report.choice, Some(SiteChoice::WriterAndReader));
        assert_eq!(report.unit_gains.as_deref(), Some(&[-1.0, -1.0, 1.0, 1.0][..]));
        assert_eq!(report.control_gains.as_deref(), Some(&[1.0, 2.0, 3.0, 1.0][..]));
        assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
        let report = compile_coupled_controls(&sites(&[1.0, 2.0, -3.0, 1.0]), "scale").expect("compiles");
        let ControlRealization::Descriptive {
            witness: Some(EvidenceStatus::Counterexample { witness, .. }),
            ..
        } = &report.compiled.realization
        else {
            panic!("a sign change inside a class is coupled");
        };
        assert_eq!(witness.path, vec![SupportNode::Control(1), SupportNode::Unit(2), SupportNode::Control(2)]);
    }

    #[test]
    fn classes_are_invariant_to_renaming_rescaling_and_duplicating_controls() {
        let (registry, _, _) = registry();
        let a = writer();
        let classes = |a: &Array2<f64>| {
            let alpha = vec![1.0; a.ncols()];
            let report = compile_coupled_controls(
                &CoupledControlProblem {
                    registry: &registry,
                    writer: a.view(),
                    alpha: &alpha,
                    writer_site: None,
                    reader_site: None,
                },
                "scale",
            )
            .expect("compiles");
            let mut sets: Vec<Vec<usize>> = report.writer_classes.into_iter().map(|class| class.controls).collect();
            sets.sort();
            sets
        };
        let base = classes(&a);
        assert_eq!(base, vec![vec![0], vec![1, 2], vec![3]]);
        let order = [3, 2, 0, 1];
        let renamed = classes(&a.select(ndarray::Axis(1), &order));
        assert_eq!(renamed, vec![vec![0], vec![1, 3], vec![2]], "renamed controls land in the renamed classes");
        let rescaled = classes(&(&a * &array![[5.0, -0.25, 3.0, 1e-3]]));
        assert_eq!(rescaled, base);
        let duplicated = classes(&ndarray::concatenate![ndarray::Axis(1), a, a.slice(s![.., 1..2])]);
        assert_eq!(duplicated, vec![vec![0], vec![1, 2, 4], vec![3]], "a duplicate joins its original's class");
    }

    #[test]
    fn a_dense_fitted_writer_couples_everything_but_a_uniform_rescaling() {
        let (registry, down, _) = registry();
        let mut rng = StdRng::seed_from_u64(21);
        let a = uniform(&mut rng, 4, 3);
        let compile = |alpha: &[f64]| {
            compile_coupled_controls(
                &CoupledControlProblem {
                    registry: &registry,
                    writer: a.view(),
                    alpha,
                    writer_site: Some(down_site(&down)),
                    reader_site: None,
                },
                "scale",
            )
            .expect("compiles")
        };
        let report = compile(&[1.0, 1.5, 1.0]);
        assert_eq!(report.writer_classes.len(), 1);
        assert!(!report.compiled.realization.is_native_control());
        assert!(compile(&[1.5, 1.5, 1.5]).compiled.realization.is_native_control());
    }
}

mod fixed_rank_chart {
    use super::super::super::dense::solve;
    use super::super::chart::{ChartSetting, FactorBinding, FixedRankChart, compile_chart_edit};
    use super::super::linear::frobenius;
    use super::*;
    use ndarray::Axis;

    fn factors() -> (Array2<f64>, Array2<f64>) {
        let mut rng = StdRng::seed_from_u64(30);
        (uniform(&mut rng, 6, 2), uniform(&mut rng, 2, 5))
    }

    #[test]
    fn the_dependent_block_is_recovered_from_the_free_coordinates() {
        let (write, read) = factors();
        let chart = FixedRankChart::from_factors(write.view(), read.view()).expect("chart");
        let dependent = chart.dependent().expect("dependent");
        let (native, native_band) = chart.native_dependent();
        let distance = frobenius((&dependent.values - &native).view());
        assert!(distance <= dependent.frobenius_band + native_band, "{distance} {}", dependent.frobenius_band);
        assert!(dependent.frobenius_band < 1e-12);
        let full = write.dot(&read);
        let block = full.select(Axis(0), chart.free_rows()).select(Axis(1), chart.free_cols());
        assert!(max_abs((&block - &dependent.values).view()) < 1e-13);
    }

    #[test]
    fn a_chart_edit_lifts_to_native_factors_with_exact_pivot_rows() {
        let (write, read) = factors();
        let mut registry = TensorRegistry::default();
        registry.register_storage(TensorId("o".into()), write.view().into_dyn()).expect("o");
        // The read factor is stored transposed, as a value projection holding Vᵀ.
        registry
            .register_storage(TensorId("v".into()), read.t().to_owned().view().into_dyn())
            .expect("v");
        let chart = FixedRankChart::from_factors(write.view(), read.view()).expect("chart");
        let (a, b, c) = chart.coordinates();
        let a_new = &a + &array![[0.1, -0.2], [0.05, 0.3]];
        let b_new = &b * 1.5;
        let c_new = &c - 0.25;
        let setting = ChartSetting {
            a: Some(a_new.view()),
            b: Some(b_new.view()),
            c: Some(c_new.view()),
        };
        let write_binding = FactorBinding {
            storage: TensorId("o".into()),
            stored_transposed: false,
        };
        let read_binding = FactorBinding {
            storage: TensorId("v".into()),
            stored_transposed: true,
        };
        let report =
            compile_chart_edit(&registry, &chart, &setting, &write_binding, &read_binding, "ov").expect("lifts");
        assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
        for &row in chart.pivot_rows() {
            assert_eq!(report.lifted_write.row(row), write.row(row), "pivot rows keep the native values");
        }
        let product = report.lifted_write.dot(&report.lifted_read);
        let pick = |rows: &[usize], cols: &[usize]| product.select(Axis(0), rows).select(Axis(1), cols);
        let (pr, fr, pc, fc) = (chart.pivot_rows(), chart.free_rows(), chart.pivot_cols(), chart.free_cols());
        assert!(max_abs((&pick(pr, pc) - &a_new).view()) < 1e-13);
        assert!(max_abs((&pick(pr, fc) - &b_new).view()) < 1e-13);
        assert!(max_abs((&pick(fr, pc) - &c_new).view()) < 1e-13);
        let dependent = c_new.dot(&solve(a_new.view(), b_new.view()).expect("solve"));
        assert!(max_abs((&pick(fr, fc) - &dependent).view()) < 1e-12, "the dependent block follows");
        assert!(max_abs((&report.dependent.values - &dependent).view()) < 1e-13);
        let plan = report.compiled.plan.as_ref().expect("plan");
        let o_edit = dense(plan, "o", (6, 2));
        let v_edit = dense(plan, "v", (5, 2));
        assert!(max_abs((&(&write + &o_edit) - &report.lifted_write).view()) < 1e-15);
        assert!(max_abs((&(&read.t() + &v_edit) - &report.lifted_read.t()).view()) < 1e-15);
        let unchanged = compile_chart_edit(
            &registry,
            &chart,
            &ChartSetting::default(),
            &write_binding,
            &read_binding,
            "ov",
        )
        .expect("lifts");
        assert!(unchanged.compiled.plan.as_ref().expect("plan").is_native(), "ρ(0) = θ");
    }

    /// Factor pairs of inner dimension 4 whose product has rank 2: the deficiency sits in the
    /// read factor (reduced through `R₁`) or in the write factor (through `R₂`).
    fn wide_pairs() -> Vec<(Array2<f64>, Array2<f64>)> {
        let mut rng = StdRng::seed_from_u64(31);
        let full_write = uniform(&mut rng, 6, 4);
        let low_read = uniform(&mut rng, 4, 2).dot(&uniform(&mut rng, 2, 5));
        let low_write = uniform(&mut rng, 6, 2).dot(&uniform(&mut rng, 2, 4));
        let full_read = uniform(&mut rng, 4, 5);
        vec![(full_write, low_read), (low_write, full_read)]
    }

    #[test]
    fn a_wide_factor_pair_is_reduced_to_its_chart_rank_and_lifts_back() {
        for (write, read) in wide_pairs() {
            let mut registry = TensorRegistry::default();
            registry.register_storage(TensorId("o".into()), write.view().into_dyn()).expect("o");
            registry.register_storage(TensorId("v".into()), read.view().into_dyn()).expect("v");
            let chart = FixedRankChart::from_factors(write.view(), read.view()).expect("chart");
            assert_eq!((chart.rank(), chart.inner_dimension()), (2, 4));
            assert!(chart.reduction_band() < 1e-12, "{}", chart.reduction_band());
            let dependent = chart.dependent().expect("dependent");
            let (native, native_band) = chart.native_dependent();
            let distance = frobenius((&dependent.values - &native).view());
            assert!(distance <= dependent.frobenius_band + native_band, "{distance}");
            let (a, b, c) = chart.coordinates();
            let a_new = &a + &array![[0.2, 0.0], [-0.1, 0.15]];
            let b_new = &b * 0.5;
            let c_new = &c + 0.1;
            let setting = ChartSetting { a: Some(a_new.view()), b: Some(b_new.view()), c: Some(c_new.view()) };
            let binding = |name: &str| FactorBinding { storage: TensorId(name.into()), stored_transposed: false };
            let report = compile_chart_edit(&registry, &chart, &setting, &binding("o"), &binding("v"), "ov").expect("lifts");
            assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
            assert_eq!(report.lifted_write.dim(), (6, 4));
            let product = report.lifted_write.dot(&report.lifted_read);
            let pick = |rows: &[usize], cols: &[usize]| product.select(Axis(0), rows).select(Axis(1), cols);
            let (pr, fr, pc, fc) = (chart.pivot_rows(), chart.free_rows(), chart.pivot_cols(), chart.free_cols());
            assert!(max_abs((&pick(pr, pc) - &a_new).view()) < 1e-12);
            assert!(max_abs((&pick(pr, fc) - &b_new).view()) < 1e-12);
            assert!(max_abs((&pick(fr, pc) - &c_new).view()) < 1e-12);
            let expected = c_new.dot(&solve(a_new.view(), b_new.view()).expect("solve"));
            assert!(max_abs((&pick(fr, fc) - &expected).view()) < 1e-11, "the dependent block follows");
            let plan = report.compiled.plan.as_ref().expect("plan");
            assert!(max_abs((&(&write + &dense(plan, "o", (6, 4))) - &report.lifted_write).view()) < 1e-15);
            assert!(max_abs((&(&read + &dense(plan, "v", (4, 5))) - &report.lifted_read).view()) < 1e-15);
            let unchanged =
                compile_chart_edit(&registry, &chart, &ChartSetting::default(), &binding("o"), &binding("v"), "ov")
                    .expect("lifts");
            assert!(unchanged.compiled.plan.as_ref().expect("plan").is_native());
        }
    }
}

mod query_key {
    use super::super::super::attention::{RotaryEmbedding, RotaryPairing};
    use super::super::bilinear::{HeadRows, QueryKeyEditProblem, ScoreClaim, compile_query_key_edit};
    use super::*;

    const HEAD: usize = 6;
    const WIDTH: usize = 5;

    fn rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.1],
            attention_scaling: 1.1,
        }
    }

    /// `α² Rot(Δ)` on the rotary planes and the identity on pass-through coordinates.
    fn relative(rotary: &RotaryEmbedding, delta: i64) -> Array2<f64> {
        let mut r = Array2::<f64>::eye(HEAD);
        let scale = rotary.attention_scaling * rotary.attention_scaling;
        for (plane, &frequency) in rotary.inverse_frequencies.iter().enumerate() {
            let (a, b) = rotary.plane(plane);
            let (sin, cos) = (delta as f64 * frequency).sin_cos();
            r[[a, a]] = scale * cos;
            r[[a, b]] = -scale * sin;
            r[[b, a]] = scale * sin;
            r[[b, b]] = scale * cos;
        }
        r
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }

    struct Fixture {
        registry: TensorRegistry,
        q: Array2<f64>,
        k: Array2<f64>,
        dq: Array2<f64>,
        dk: Array2<f64>,
        xq: Array2<f64>,
        xk: Array2<f64>,
    }

    fn fixture() -> Fixture {
        let mut rng = StdRng::seed_from_u64(40);
        let q = uniform(&mut rng, HEAD, WIDTH);
        let k = uniform(&mut rng, HEAD, WIDTH);
        let dq = uniform(&mut rng, HEAD, WIDTH) * 0.5;
        let dk = uniform(&mut rng, HEAD, WIDTH) * 0.5;
        let xq = uniform(&mut rng, 3, WIDTH);
        let xk = uniform(&mut rng, 4, WIDTH);
        let mut fused = Array2::<f64>::zeros((2 * HEAD, WIDTH));
        fused.slice_mut(s![..HEAD, ..]).assign(&q);
        fused.slice_mut(s![HEAD.., ..]).assign(&k);
        let mut registry = TensorRegistry::default();
        registry.register_storage(TensorId("qk".into()), fused.view().into_dyn()).expect("qk");
        Fixture {
            registry,
            q,
            k,
            dq,
            dk,
            xq,
            xk,
        }
    }

    fn rows(offset: usize) -> HeadRows {
        HeadRows {
            storage: TensorId("qk".into()),
            row_offset: offset,
        }
    }

    #[test]
    fn the_finite_change_keeps_the_cross_term_and_matches_brute_force() {
        let f = fixture();
        let (q_set, k_set) = (&f.q + &f.dq, &f.k + &f.dk);
        let rotary = rotary();
        let query_positions = [1_i64, 2, 3];
        let key_positions = [0_i64, 1, 2, 3];
        let sigma = 1.0 / (HEAD as f64).sqrt();
        let problem = QueryKeyEditProblem {
            registry: &f.registry,
            query: f.q.view(),
            key: f.k.view(),
            query_setting: q_set.view(),
            key_setting: k_set.view(),
            query_rows: rows(0),
            key_rows: rows(HEAD),
            rotary: Some(&rotary),
            score_scale: sigma,
            queries: f.xq.view(),
            query_positions: &query_positions,
            keys: f.xk.view(),
            key_positions: &key_positions,
            causal: true,
            claim: ScoreClaim::FirstOrder,
        };
        let report = compile_query_key_edit(&problem, "route").expect("certifies");
        let (q2, k2) = (q_set.clone(), k_set.clone());
        for t in 0..3 {
            let mut base = Vec::new();
            let mut edited = Vec::new();
            let mut first = Vec::new();
            for s in 0..4 {
                let r = relative(&rotary, key_positions[s] - query_positions[t]);
                let score = |qw: &Array2<f64>, kw: &Array2<f64>| {
                    sigma * qw.dot(&f.xq.row(t)).dot(&r.dot(&kw.dot(&f.xk.row(s))))
                };
                let cross = score(&f.dq, &f.dk);
                assert!((report.cross[[t, s]] - cross).abs() < 1e-13, "cross term Δqᵀ R Δk");
                if key_positions[s] > query_positions[t] {
                    assert_eq!(report.exact_change.values[[t, s]], 0.0);
                    continue;
                }
                let change = score(&q2, &k2) - score(&f.q, &f.k);
                let band = report.exact_change.bands[[t, s]];
                assert!((report.exact_change.values[[t, s]] - change).abs() <= band + 1e-14, "{t} {s}");
                assert!((report.first_order[[t, s]] + cross - change).abs() < 1e-12, "first order + cross = finite");
                assert!(cross.abs() > 1e-6, "the fixture's cross term is not negligible");
                base.push(score(&f.q, &f.k));
                edited.push(score(&q2, &k2));
                first.push(score(&f.q, &f.k) + report.first_order[[t, s]]);
            }
            let (p, p2, p_first) = (softmax(&base), softmax(&edited), softmax(&first));
            let row = &report.rows[t];
            for (index, (&before, &after)) in p.iter().zip(&p2).enumerate() {
                assert!((row.change.values[index] - (after - before)).abs() <= row.change.bands[index] + 1e-14);
            }
            let tv: f64 = p2.iter().zip(&p_first).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
            assert!(tv > 0.0 && tv <= row.claim_total_variation, "TV {tv} ≤ tanh bound {}", row.claim_total_variation);
        }
        assert!(matches!(report.compiled.realization, ControlRealization::EmpiricallyValidated { .. }));
        // One fused storage carries both blocks.
        let plan = report.compiled.plan.as_ref().expect("plan");
        assert_eq!(plan.edits().len(), 1);
        let edit = dense(plan, "qk", (2 * HEAD, WIDTH));
        assert!(max_abs((&edit.slice(s![..HEAD, ..]) - &f.dq).view()) < 1e-15);
        assert!(max_abs((&edit.slice(s![HEAD.., ..]) - &f.dk).view()) < 1e-15);
    }

    #[test]
    fn a_one_sided_edit_is_exactly_first_order() {
        let f = fixture();
        let q_set = &f.q + &f.dq;
        let positions_q = [0_i64, 1, 2];
        let positions_k = [0_i64, 1, 2, 3];
        let problem = QueryKeyEditProblem {
            registry: &f.registry,
            query: f.q.view(),
            key: f.k.view(),
            query_setting: q_set.view(),
            key_setting: f.k.view(),
            query_rows: rows(0),
            key_rows: rows(HEAD),
            rotary: None,
            score_scale: 1.0,
            queries: f.xq.view(),
            query_positions: &positions_q,
            keys: f.xk.view(),
            key_positions: &positions_k,
            causal: false,
            claim: ScoreClaim::FirstOrder,
        };
        let report = compile_query_key_edit(&problem, "route").expect("certifies");
        assert!(report.cross.iter().all(|v| *v == 0.0));
        assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
    }

    #[test]
    fn a_score_setting_is_solved_on_both_sides_and_certified() {
        use super::super::bilinear::{QueryKeySolveProblem, ScoreRequirement, solve_query_key_setting};
        // A one-dimensional head: scores are q_t k_s, so a rank-one target needs both sides to
        // move (neither side alone can fit a·bᵀ from the native q, k).
        let mut registry = TensorRegistry::default();
        let fused = array![[0.8, -0.3], [0.2, 0.9]];
        registry.register_storage(TensorId("qk".into()), fused.view().into_dyn()).expect("qk");
        let q = fused.slice(s![..1, ..]).to_owned();
        let k = fused.slice(s![1.., ..]).to_owned();
        let xq = array![[1.0, 0.0], [0.3, 1.0]];
        let xk = array![[0.5, 1.0], [1.0, -0.4]];
        let (a, b) = ([1.5, -0.5], [0.7, 2.0]);
        let requirements: Vec<ScoreRequirement> = (0..2)
            .flat_map(|t| (0..2).map(move |s| ScoreRequirement { query: t, key: s, target: a[t] * b[s] }))
            .collect();
        let positions = [0_i64, 0];
        let problem = QueryKeySolveProblem {
            registry: &registry,
            query: q.view(),
            key: k.view(),
            query_rows: rows(0),
            key_rows: rows(1),
            rotary: None,
            score_scale: 1.0,
            queries: xq.view(),
            query_positions: &positions,
            keys: xk.view(),
            key_positions: &positions,
            causal: false,
            requirements: &requirements,
            target_radius: 0.0,
            max_rounds: 500,
        };
        let report = solve_query_key_setting(&problem, "route").expect("solves");
        assert!(report.rounds > 1, "one side alone cannot fit the rank-one target");
        let ControlRealization::ExactlyRealized { residual: EvidenceStatus::Exact { basis, .. }, .. } =
            &report.compiled.realization
        else {
            panic!("certified: {:?} after {} rounds", report.residuals, report.rounds);
        };
        assert_eq!(*basis, ExactBasis::Exhaustive { cardinality: 4 });
        for requirement in &requirements {
            let score = report.query_setting.dot(&xq.row(requirement.query))[0]
                * report.key_setting.dot(&xk.row(requirement.key))[0];
            assert!((score - requirement.target).abs() < 1e-12);
            // The exact finite change agrees with the target's change from the native score.
            let native = q.dot(&xq.row(requirement.query))[0] * k.dot(&xk.row(requirement.key))[0];
            let change = report.certification.exact_change.values[[requirement.query, requirement.key]];
            let band = report.certification.exact_change.bands[[requirement.query, requirement.key]];
            assert!((change - (requirement.target - native)).abs() <= band + 1e-12);
        }
        let plan = report.compiled.plan.as_ref().expect("plan");
        let edit = dense(plan, "qk", (2, 2));
        assert!(max_abs((&edit.slice(s![..1, ..]) - &(&report.query_setting - &q)).view()) < 1e-15);
        assert!(max_abs((&edit.slice(s![1.., ..]) - &(&report.key_setting - &k)).view()) < 1e-15);
    }

    #[test]
    fn a_score_setting_one_side_can_reach_is_certified_in_one_round() {
        use super::super::bilinear::{QueryKeySolveProblem, ScoreRequirement, solve_query_key_setting};
        let f = fixture();
        let rotary = rotary();
        let query_positions = [1_i64, 2, 3];
        let key_positions = [0_i64, 1, 2, 3];
        let requirements = [
            ScoreRequirement { query: 0, key: 1, target: 2.0 },
            ScoreRequirement { query: 2, key: 0, target: -1.5 },
            ScoreRequirement { query: 2, key: 3, target: 0.25 },
        ];
        let problem = QueryKeySolveProblem {
            registry: &f.registry,
            query: f.q.view(),
            key: f.k.view(),
            query_rows: rows(0),
            key_rows: rows(HEAD),
            rotary: Some(&rotary),
            score_scale: 0.5,
            queries: f.xq.view(),
            query_positions: &query_positions,
            keys: f.xk.view(),
            key_positions: &key_positions,
            causal: true,
            requirements: &requirements,
            target_radius: 0.0,
            max_rounds: 3,
        };
        let report = solve_query_key_setting(&problem, "route").expect("solves");
        assert_eq!(report.rounds, 1);
        assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
        for (requirement, (residual, band)) in requirements.iter().zip(&report.residuals) {
            assert!(residual.abs() <= *band, "{requirement:?}");
        }
    }
}

mod null_edits {
    use super::super::null::{NullEditOutcome, physically_null_supremum};
    use super::*;

    #[test]
    fn the_supremum_is_the_generalized_eigenvalue_on_the_range() {
        let g = array![[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let k = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]];
        let NullEditOutcome::Bounded(EvidenceStatus::Exact {
            value,
            numerical_error,
            witness: Some(witness),
            ..
        }) = physically_null_supremum(g.view(), k.view()).expect("bounded")
        else {
            panic!("bounded exact");
        };
        assert!((value - 3.0).abs() <= numerical_error + 1e-15);
        assert!((witness.edit_size - 1.0).abs() < 1e-14 && (witness.response_error - 3.0).abs() < 1e-14);
        // A change of edit coordinates u = T w leaves the supremum unchanged.
        let t = array![[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]];
        let g2 = t.t().dot(&g).dot(&t);
        let k2 = t.t().dot(&k).dot(&t);
        let NullEditOutcome::Bounded(EvidenceStatus::Exact {
            value: moved,
            numerical_error: moved_error,
            ..
        }) = physically_null_supremum(g2.view(), k2.view()).expect("bounded")
        else {
            panic!("bounded exact");
        };
        assert!((moved - 3.0).abs() <= moved_error + 1e-13);
    }

    #[test]
    fn a_response_on_the_edit_kernel_is_refused_with_its_direction() {
        let g = array![[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let k = array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.5]];
        let NullEditOutcome::Refused(EvidenceStatus::Counterexample { witness, value, .. }) =
            physically_null_supremum(g.view(), k.view()).expect("decided")
        else {
            panic!("refused");
        };
        assert!((witness.direction[2].abs() - 1.0).abs() < 1e-14);
        assert!((value - 0.5).abs() < 1e-14);
        assert!(witness.edit_size.abs() < 1e-15);
    }
}

mod set_type_settings {
    use super::super::linear::OffTargetInputs;
    use super::super::{SetAtom, Setting};
    use super::*;

    fn atom(control: &str, value: f64) -> SetAtom<f64> {
        SetAtom {
            control: control.to_string(),
            value,
        }
    }

    #[test]
    fn histories_reduce_to_one_normal_form() {
        // Left-annihilativity: a later set of a control annihilates the earlier one.
        let overwritten = Setting::from_history([atom("gate", 2.0), atom("gate", 0.5)]);
        assert_eq!(overwritten, Setting::from_history([atom("gate", 0.5)]));
        // Sets of distinct controls commute.
        let one = Setting::from_history([atom("gate", 2.0), atom("route", 3.0)]);
        let other = Setting::from_history([atom("route", 3.0), atom("gate", 2.0)]);
        assert_eq!(one, other);
        // Composition is the history's concatenation; the native setting is its identity.
        let composed = one.clone().then(Setting::from_history([atom("gate", 7.0)]));
        assert_eq!(composed, Setting::from_history([atom("gate", 2.0), atom("route", 3.0), atom("gate", 7.0)]));
        assert_eq!(Setting::native().then(one.clone()), one);
        assert_eq!(one.clone().then(Setting::native()), one);
        assert!(Setting::<f64>::native().is_native());
        assert_eq!(composed.get("gate"), Some(&7.0));
    }

    #[test]
    fn declaring_off_target_inputs_as_set_to_native_removes_their_damage() {
        let (registry, weight) = single_use(3, 4, 50);
        let mut rng = StdRng::seed_from_u64(51);
        let on_target = uniform(&mut rng, 1, 4);
        let off_target = uniform(&mut rng, 2, 4);
        let changes = array![[1.0, -0.5, 0.25]];
        let targets = set_targets(&on_target, &weight, &changes);
        let off_sets = || {
            vec![OffTargetInputs {
                site: UseSiteId("w#0".into()),
                inputs: off_target.view(),
            }]
        };
        let mut rank_one = problem(&registry, weight.view(), vec![linear(on_target.view(), targets.view(), ResponseClass::Sample)]);
        rank_one.off_target = off_sets();
        let report = compile_linear_site(&rank_one, "edit", test_governor()).expect("compiles");
        let Some(EvidenceStatus::Exact { value: damage, numerical_error, .. }) = report.off_target_damage else {
            panic!("damage reported");
        };
        assert!(damage - numerical_error > 0.0, "the unconstrained minimum-norm edit moves off-target inputs");
        // The same edit with the off-target responses set to their native values.
        let held = off_target.dot(&weight.t());
        let mut guarded = problem(
            &registry,
            weight.view(),
            vec![
                linear(on_target.view(), targets.view(), ResponseClass::Sample),
                linear(off_target.view(), held.view(), ResponseClass::Sample),
            ],
        );
        guarded.off_target = off_sets();
        let report = compile_linear_site(&guarded, "edit", test_governor()).expect("compiles");
        let Some(EvidenceStatus::Exact { value: guarded_damage, numerical_error: guarded_band, .. }) =
            report.off_target_damage
        else {
            panic!("damage reported");
        };
        assert!(guarded_damage <= guarded_band + report.allowance[0], "{guarded_damage} {guarded_band}");
        let edit = dense(report.compiled.plan.as_ref().expect("plan"), "w", (3, 4));
        assert!(max_abs((edit.dot(&on_target.t()) - changes.t()).view()) < 1e-12);
    }
}

mod through_nonlinearities {
    use super::super::path::{PathLayer, PathProblem, PathSite, compile_path};
    use super::*;
    use ndarray::Array1;

    const WIDTH: usize = 4;
    const HIDDEN: usize = 6;

    struct Block {
        registry: TensorRegistry,
        gain: Vec<f64>,
        gate: Array2<f64>,
        up: Array2<f64>,
        down: Array2<f64>,
    }

    fn block() -> Block {
        let mut rng = StdRng::seed_from_u64(60);
        let gate = uniform(&mut rng, HIDDEN, WIDTH);
        let up = uniform(&mut rng, HIDDEN, WIDTH);
        let down = uniform(&mut rng, WIDTH, HIDDEN);
        let mut registry = TensorRegistry::default();
        for (name, weight) in [("gate", &gate), ("up", &up), ("down", &down)] {
            registry.register_storage(TensorId(name.into()), weight.view().into_dyn()).expect("storage");
        }
        Block { registry, gain: vec![1.2, 0.8, 1.0, 0.9], gate, up, down }
    }

    /// `x + W_d [silu(W_g n(x)) ⊙ W_u n(x)]`, `n` the gained RMSNorm, written out directly.
    fn reference(b: &Block, gate: &Array2<f64>, up: &Array2<f64>, down: &Array2<f64>, x: ArrayView1<'_, f64>) -> Array1<f64> {
        let r = (1e-6 + x.dot(&x) / WIDTH as f64).sqrt();
        let normed = Array1::from_shape_fn(WIDTH, |i| b.gain[i] * x[i] / r);
        let a = gate.dot(&normed);
        let u = up.dot(&normed);
        let hidden = Array1::from_shape_fn(HIDDEN, |i| a[i] / (1.0 + (-a[i]).exp()) * u[i]);
        &x + &down.dot(&hidden)
    }

    fn layers(b: &Block) -> Vec<PathLayer> {
        vec![PathLayer::Residual(vec![
            PathLayer::RmsNorm { gain: Some(b.gain.clone()), epsilon: 1e-6 },
            PathLayer::Swiglu { gate: 0, up: 1 },
            PathLayer::Linear { site: 2 },
        ])]
    }

    fn sites<'a>(b: &'a Block, editable: [bool; 3]) -> Vec<PathSite<'a>> {
        [("gate", &b.gate), ("up", &b.up), ("down", &b.down)]
            .into_iter()
            .zip(editable)
            .map(|((name, weight), editable)| PathSite { storage: TensorId(name.into()), weight: weight.view(), bias: None, editable })
            .collect()
    }

    fn applied(plan: &NativeEditPlan, name: &str, weight: &Array2<f64>) -> Array2<f64> {
        weight + &dense(plan, name, weight.dim())
    }

    #[test]
    fn a_requirement_behind_a_norm_and_a_gate_is_solved_at_the_gate_and_certified() {
        let b = block();
        let mut rng = StdRng::seed_from_u64(61);
        let inputs = uniform(&mut rng, 3, WIDTH);
        // Targets reachable by moving the gate (nonlinear in it, through silu).
        let moved_gate = &b.gate + &(uniform(&mut rng, HIDDEN, WIDTH) * 0.1);
        let targets = Array2::from_shape_fn((3, WIDTH), |(r, c)| reference(&b, &moved_gate, &b.up, &b.down, inputs.row(r))[c]);
        let off = uniform(&mut rng, 2, WIDTH);
        let problem = PathProblem {
            registry: &b.registry,
            sites: sites(&b, [true, false, false]),
            layers: layers(&b),
            inputs: inputs.view(),
            targets: targets.view(),
            target_radius: 0.0,
            off_target: Some(off.view()),
            max_iterations: 60,
        };
        let report = compile_path(&problem, "gate").expect("solves");
        assert!(report.iterations > 1, "a nonlinear site needs more than one Gauss–Newton step");
        assert!(
            matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }),
            "{:?} after {}",
            report.compiled.realization,
            report.iterations
        );
        // Execute the stored plan through an independent forward.
        let plan = report.compiled.plan.as_ref().expect("plan");
        assert_eq!(plan.edits().len(), 1);
        let gate = applied(plan, "gate", &b.gate);
        for row in 0..3 {
            let out = reference(&b, &gate, &b.up, &b.down, inputs.row(row));
            for c in 0..WIDTH {
                assert!((out[c] - targets[[row, c]]).abs() <= report.residual_band[[row, c]] + 1e-13);
            }
        }
        assert!(report.off_target_damage.is_some());
    }

    #[test]
    fn a_requirement_linear_in_its_site_is_certified_at_once() {
        let b = block();
        let mut rng = StdRng::seed_from_u64(62);
        let inputs = uniform(&mut rng, 3, WIDTH);
        let native = Array2::from_shape_fn((3, WIDTH), |(r, c)| reference(&b, &b.gate, &b.up, &b.down, inputs.row(r))[c]);
        let targets = &native + &(uniform(&mut rng, 3, WIDTH) * 0.5);
        let problem = PathProblem {
            registry: &b.registry,
            sites: sites(&b, [false, false, true]),
            layers: layers(&b),
            inputs: inputs.view(),
            targets: targets.view(),
            target_radius: 0.0,
            off_target: None,
            max_iterations: 5,
        };
        let report = compile_path(&problem, "down").expect("solves");
        // Linear in its site: the first step is exact up to where LSQR stopped, and at most
        // one more step finishes it.
        assert!(report.iterations <= 2, "{}", report.iterations);
        assert!(matches!(report.compiled.realization, ControlRealization::ExactlyRealized { .. }));
        let down = applied(report.compiled.plan.as_ref().expect("plan"), "down", &b.down);
        for row in 0..3 {
            let out = reference(&b, &b.gate, &b.up, &down, inputs.row(row));
            assert!(out.iter().zip(targets.row(row)).all(|(a, t)| (a - t).abs() < 1e-12));
        }
    }

    #[test]
    fn an_overdetermined_requirement_is_only_empirically_validated() {
        let b = block();
        let mut rng = StdRng::seed_from_u64(63);
        // 8 inputs × 4 outputs = 32 conditions on the 24 entries of the down projection.
        let inputs = uniform(&mut rng, 8, WIDTH);
        let targets = uniform(&mut rng, 8, WIDTH);
        let problem = PathProblem {
            registry: &b.registry,
            sites: sites(&b, [false, false, true]),
            layers: layers(&b),
            inputs: inputs.view(),
            targets: targets.view(),
            target_radius: 0.0,
            off_target: None,
            max_iterations: 5,
        };
        let report = compile_path(&problem, "down").expect("solves");
        let ControlRealization::EmpiricallyValidated { residual: EvidenceStatus::Exact { value, numerical_error, .. }, .. } =
            &report.compiled.realization
        else {
            panic!("an infeasible requirement is not exactly realized");
        };
        assert!(value > numerical_error);
    }
}
