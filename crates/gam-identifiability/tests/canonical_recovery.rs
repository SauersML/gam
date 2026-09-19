//! #2818 recovery through the supported Rust canonicalization API.

use std::ops::Range;
use std::sync::Arc;

use gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_problem::test_support::{spec_from_dense, spec_from_dense_with_priority};
use gam_problem::{
    AdditiveBlockJacobian, BlockEffectiveJacobian, CoefficientCoordinate, FamilyLinearizationState,
    PenaltyMatrix,
};
use ndarray::{Array1, Array2, s};

#[test]
fn canonical_dead_column_callback_block_is_not_reduced_1590() {
    let n = 128;
    let p = 4;
    let x = Array1::linspace(-1.0_f64, 1.0, n);
    let design = Array2::from_shape_fn((n, p), |(row, column)| match column {
        0 => 0.0,
        1 => 1.0 + x[row],
        2 => 1.0,
        _ => (2.0 * x[row]).sin(),
    });
    let specs: Vec<_> = (0..2)
        .map(|cause| {
            let mut spec = spec_from_dense_with_priority(
                &format!("time_cause_{}", cause + 1),
                design.clone(),
                102 - cause as u8,
            );
            spec.jacobian_callback = Some(Arc::new(AdditiveBlockJacobian {
                design: Arc::new(design.clone()),
                own_output: cause,
                n_family_outputs: 2,
            }));
            spec
        })
        .collect();
    let canonical = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[CoefficientCoordinate::Spanning; 2],
        None,
    )
    .expect("the callback's raw coefficient width remains synchronized");
    assert!(canonical.used_channel_aware_audit);
    assert!(
        canonical
            .audit
            .dropped_columns
            .iter()
            .any(|drop| drop.column == 0),
        "the audit must actually detect the zero column"
    );
    // Nonzero coefficients make this an actual identity-lift check: a broken
    // lift that merely returned zeros could pass the old all-zero fixture.
    let coefficients: Vec<_> = (0..2)
        .map(|cause| {
            assert_eq!(canonical.reduced_specs[cause].design.ncols(), p);
            Array1::from_shape_fn(p, |column| 0.3 + cause as f64 - 0.2 * column as f64)
        })
        .collect();
    let lifted = canonical.gauge.lift_block_betas(&coefficients);
    assert_eq!(lifted, coefficients);
}

struct FixedTwoChannelJacobian {
    full: Array2<f64>,
    n: usize,
}

impl BlockEffectiveJacobian for FixedTwoChannelJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, String> {
        // The canonical row operator requests coefficient-independent Jacobians
        // with an empty beta; operating-point audit calls supply actual beta.
        if !state.beta.is_empty() && state.beta.len() != self.full.ncols() {
            return Err(format!(
                "fixture coefficient width {} differs from Jacobian width {}",
                state.beta.len(),
                self.full.ncols()
            ));
        }
        let end = rows.end.min(self.n);
        if rows.start > end {
            return Err("fixture row range is reversed or out of bounds".into());
        }
        let width = end - rows.start;
        let mut output = Array2::zeros((2 * width, self.full.ncols()));
        for channel in 0..2 {
            output
                .slice_mut(s![channel * width..(channel + 1) * width, ..])
                .assign(&self.full.slice(s![
                    channel * self.n + rows.start..channel * self.n + end,
                    ..
                ]));
        }
        Ok(output)
    }

    fn n_outputs(&self) -> usize {
        2
    }
}

/// The operating point a converged-state drift check linearizes at, for the fixtures
/// below: per-block column scales, and a switch that gives every block with a converged
/// recipe that recipe's columns.
struct FixtureOperatingPoint {
    column_scales: Vec<Vec<f64>>,
    converged_shape: bool,
}

/// A two-channel callback whose effective columns are the fixture columns scaled by the
/// operating point's per-block column scales, switching to `converged_full` when the
/// operating point asks for the converged shape.
struct ScaledTwoChannelJacobian {
    full: Array2<f64>,
    converged_full: Option<Array2<f64>>,
    block: usize,
    n: usize,
}

impl BlockEffectiveJacobian for ScaledTwoChannelJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let point = state
            .family_scalars
            .as_ref()
            .and_then(|scalars| scalars.downcast_ref::<FixtureOperatingPoint>());
        let source = match (point, self.converged_full.as_ref()) {
            (Some(point), Some(converged)) if point.converged_shape => converged,
            _ => &self.full,
        };
        let end = rows.end.min(self.n);
        if rows.start > end {
            return Err("fixture row range is reversed or out of bounds".into());
        }
        let width = end - rows.start;
        let p = source.ncols();
        let mut output = Array2::zeros((2 * width, p));
        for channel in 0..2 {
            output
                .slice_mut(s![channel * width..(channel + 1) * width, ..])
                .assign(&source.slice(s![
                    channel * self.n + rows.start..channel * self.n + end,
                    ..
                ]));
        }
        if let Some(point) = point {
            for column in 0..p {
                let scale = point.column_scales[self.block][column];
                output.column_mut(column).mapv_inplace(|value| value * scale);
            }
        }
        Ok(output)
    }

    fn n_outputs(&self) -> usize {
        2
    }
}

/// The first `count` Legendre polynomials on a uniform grid of `n` points in [-1, 1].
fn legendre_columns(n: usize, count: usize) -> Array2<f64> {
    let x = Array1::linspace(-1.0_f64, 1.0, n);
    let mut legendre = Array2::zeros((n, count));
    for row in 0..n {
        legendre[[row, 0]] = 1.0;
        if count > 1 {
            legendre[[row, 1]] = x[row];
        }
        for degree in 2..count {
            legendre[[row, degree]] = ((2 * degree - 1) as f64 * x[row]
                * legendre[[row, degree - 1]]
                - (degree - 1) as f64 * legendre[[row, degree - 2]])
                / degree as f64;
        }
    }
    legendre
}

/// Indicators of disjoint row segments of the given lengths, in order, on `n` rows.
fn segment_indicator_columns(n: usize, lengths: &[usize]) -> Array2<f64> {
    let mut columns = Array2::zeros((n, lengths.len()));
    let mut start = 0;
    for (column, &length) in lengths.iter().enumerate() {
        columns.slice_mut(s![start..start + length, column]).fill(1.0);
        start += length;
    }
    columns
}

/// Two-channel fixture blocks from per-block column recipes over the columns of `basis`,
/// with the same columns on both channels. `converged[block]`, where present, is that
/// block's recipe at an operating point that asks for the converged shape.
fn two_channel_specs(
    basis: &Array2<f64>,
    recipes: &[Vec<Vec<(usize, f64)>>],
    converged: &[Option<Vec<Vec<(usize, f64)>>>],
    penalized: bool,
) -> Vec<gam_problem::ParameterBlockSpec> {
    let n = basis.nrows();
    let build = |recipe: &Vec<Vec<(usize, f64)>>| {
        Array2::from_shape_fn((2 * n, recipe.len()), |(row, column)| {
            recipe[column]
                .iter()
                .map(|&(index, weight)| weight * basis[[row % n, index]])
                .sum::<f64>()
        })
    };
    recipes
        .iter()
        .enumerate()
        .map(|(block, recipe)| {
            let full = build(recipe);
            let p = full.ncols();
            let mut spec = spec_from_dense(
                &format!("surface_{}", block + 1),
                full.slice(s![..n, ..]).to_owned(),
            );
            if penalized {
                spec.penalties = vec![PenaltyMatrix::Dense(Array2::eye(p))];
                spec.initial_log_lambdas = Array1::zeros(1);
                spec.nullspace_dims = vec![0];
            }
            let converged_full = converged.get(block).and_then(Option::as_ref).map(build);
            spec.jacobian_callback = Some(Arc::new(ScaledTwoChannelJacobian {
                full,
                converged_full,
                block,
                n,
            }));
            spec
        })
        .collect()
}

fn operating_point(
    column_scales: Vec<Vec<f64>>,
    converged_shape: bool,
) -> Arc<dyn std::any::Any + Send + Sync> {
    Arc::new(FixtureOperatingPoint {
        column_scales,
        converged_shape,
    })
}

/// #2627 finding 9: the converged drift check must rank with the audit that produced
/// the pilot verdict. Each block carries a penalty-covered internal alias, so the
/// channel-aware pilot drops a design-structural column that the penalty identifies.
/// The penalty-augmented flat re-audit reports it recovered and refuses an unchanged
/// model. The like-for-like channel-aware re-audit reports no change.
#[test]
fn converged_drift_ranks_penalty_covered_aliases_with_the_pilot_audit_2627() {
    use gam_identifiability::audit::{audit_beta_relative_change, maybe_log_audit_drift};
    use gam_identifiability::canonical::converged_channel_aware_verdict;
    let n = 64;
    let specs = two_channel_specs(
        &legendre_columns(n, 8),
        &[
            vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)], vec![(0, 1.0), (1, 1.0)]],
            vec![vec![(3, 1.0)], vec![(4, 1.0)], vec![(5, 1.0)], vec![(3, 1.0), (4, 1.0)]],
        ],
        &[],
        true,
    );
    let pilot = operating_point(vec![vec![1.0; 4], vec![1.0; 4]], false);
    let converged = operating_point(vec![vec![2.0; 4], vec![0.5; 4]], false);
    let canonical = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[CoefficientCoordinate::Spanning; 2],
        Some(Arc::clone(&pilot)),
    )
    .expect("penalty-covered aliases canonicalize");
    assert!(canonical.used_channel_aware_audit);
    assert!(
        !canonical.audit.dropped_columns.is_empty(),
        "the channel-aware pilot must drop the design-structural alias"
    );
    assert!(!canonical.audit.fatal, "the penalty covers each alias");
    let beta_pilot = vec![0.0; 8];
    let beta_current: Vec<f64> = (0..8).map(|index| 0.25 + 0.1 * index as f64).collect();
    let flat = maybe_log_audit_drift(
        &specs,
        &canonical.audit,
        &beta_pilot,
        &beta_current,
        Some(&converged),
        0,
        1,
        1.0,
    )
    .expect("flat drift audit runs")
    .expect("a period-one drift audit always runs");
    assert!(
        flat.verdict_changed() && !flat.recovered.is_empty(),
        "positive control: the penalty-augmented flat re-audit reports the penalty-covered \
         aliases as recovered, the refusal on today's code (pilot_rank={} current_rank={})",
        flat.pilot_rank,
        flat.current_rank
    );
    let like_for_like = converged_channel_aware_verdict(
        &specs,
        &canonical,
        converged,
        audit_beta_relative_change(&beta_pilot, &beta_current),
        0,
    )
    .expect("channel-aware converged verdict runs");
    assert!(
        !like_for_like.refuses() && !like_for_like.representative_swap(),
        "the same rank definition at both operating points leaves the verdict unchanged \
         (pilot_rank={} current_rank={} pilot_gauge_rank={} newly_dropped={} recovered={})",
        like_for_like.drift.pilot_rank,
        like_for_like.drift.current_rank,
        like_for_like.pilot_gauge_rank(),
        like_for_like.drift.newly_dropped.len(),
        like_for_like.drift.recovered.len()
    );
}

/// #2627 finding 9: the like-for-like drift check still refuses a direction the fit
/// loses at convergence. An identified column whose effective channel weight vanishes
/// at the converged operating point is newly dropped.
#[test]
fn converged_drift_refuses_a_direction_lost_at_convergence_2627() {
    use gam_identifiability::audit::audit_beta_relative_change;
    use gam_identifiability::canonical::converged_channel_aware_verdict;
    let n = 64;
    let specs = two_channel_specs(
        &legendre_columns(n, 8),
        &[
            vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)], vec![(3, 1.0)]],
            vec![vec![(4, 1.0)], vec![(5, 1.0)]],
        ],
        &[],
        true,
    );
    let pilot = operating_point(vec![vec![1.0; 4], vec![1.0; 2]], false);
    let converged = operating_point(vec![vec![1.0; 4], vec![1.0, 0.0]], false);
    let canonical = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[CoefficientCoordinate::Spanning; 2],
        Some(Arc::clone(&pilot)),
    )
    .expect("independent columns canonicalize");
    assert!(canonical.used_channel_aware_audit);
    assert!(
        canonical.audit.dropped_columns.is_empty(),
        "the pilot identifies every column"
    );
    let beta_pilot = vec![0.0; 6];
    let beta_current = vec![0.5; 6];
    let verdict = converged_channel_aware_verdict(
        &specs,
        &canonical,
        converged,
        audit_beta_relative_change(&beta_pilot, &beta_current),
        0,
    )
    .expect("channel-aware converged verdict runs");
    assert!(
        !verdict.drift.newly_dropped.is_empty() && verdict.refuses(),
        "a column lost at the converged operating point must be newly dropped and refuse \
         (newly_dropped={} current_rank={} pilot_gauge_rank={})",
        verdict.drift.newly_dropped.len(),
        verdict.drift.current_rank,
        verdict.pilot_gauge_rank()
    );
}

/// #2627 finding 9: the like-for-like drift check still refuses a fatal flip. Two
/// unpenalized blocks are independent at the pilot, and at the converged operating
/// point the second block's first column becomes a copy of the first block's, an alias
/// no penalty covers.
#[test]
fn converged_drift_refuses_a_fatal_alias_at_convergence_2627() {
    use gam_identifiability::audit::audit_beta_relative_change;
    use gam_identifiability::canonical::converged_channel_aware_verdict;
    let n = 64;
    let specs = two_channel_specs(
        &legendre_columns(n, 8),
        &[
            vec![vec![(0, 1.0)], vec![(1, 1.0)]],
            vec![vec![(2, 1.0)], vec![(3, 1.0)]],
        ],
        &[None, Some(vec![vec![(0, 1.0)], vec![(3, 1.0)]])],
        false,
    );
    let pilot = operating_point(vec![vec![1.0; 2], vec![1.0; 2]], false);
    let converged = operating_point(vec![vec![1.0; 2], vec![1.0; 2]], true);
    let canonical = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[CoefficientCoordinate::Spanning; 2],
        Some(Arc::clone(&pilot)),
    )
    .expect("independent unpenalized columns canonicalize");
    assert!(canonical.used_channel_aware_audit);
    assert!(!canonical.audit.fatal, "the pilot has no alias");
    let beta_pilot = vec![0.0; 4];
    let beta_current = vec![0.5; 4];
    let verdict = converged_channel_aware_verdict(
        &specs,
        &canonical,
        converged,
        audit_beta_relative_change(&beta_pilot, &beta_current),
        0,
    )
    .expect("channel-aware converged verdict runs");
    assert!(
        verdict.drift.current_fatal && !verdict.drift.pilot_fatal && verdict.refuses(),
        "an unpenalized exact alias appearing at convergence must flip the verdict to fatal \
         and refuse (pilot_fatal={} current_fatal={} newly_dropped={})",
        verdict.drift.pilot_fatal,
        verdict.drift.current_fatal,
        verdict.drift.newly_dropped.len()
    );
}

/// #2627 finding 9: a converged drop set is judged through the pilot's gauge, not by
/// column label. The channel-aware compile residualizes each block against the blocks
/// before it, so an alias is absorbed by the last block it touches; gauge priority does
/// not enter that attribution. When the operating point moves an alias class onto a
/// later block, the drop moves with it at an unchanged rank.
///
/// Blocks A = [e0, e1], B = [b, e2] and C = [c, e4] are built on disjoint row-segment
/// indicators e0..e5.
/// - Pilot b = e0, c = e5: B absorbs the alias b = e0, and the reducing gauge removes b.
/// - Converged b = e3, c = e0 + e3: C absorbs the alias c = e0 + b. The joint rank is
///   unchanged, and the pilot's kept columns are still full rank, because c carries e3
///   once b is gone. Comparing drop labels refuses this model (the positive control),
///   and the verdict accepts it as a representative swap.
/// - Converged b = e3, c = e0: the raw rank still agrees, because b carries e3. The
///   problem the fit ran lost e3 along with b, and the verdict refuses.
///
/// Why indicators: the per-block pivot equilibrates each candidate column by its own
/// cross-block residual, so it ranks a roundoff residual like a real one. With segment
/// lengths that are powers of two, every Gram entry is exact and the other blocks'
/// Gram is diagonal, so a copied column's residual is exactly zero and the pivot
/// demotes it without depending on roundoff.
#[test]
fn converged_verdict_accepts_a_representative_swap_and_refuses_a_gauge_loss_2627() {
    use gam_identifiability::audit::{
        IdentifiabilityAudit, audit_beta_relative_change, audit_verdict_drift,
    };
    use gam_identifiability::canonical::{
        channel_aware_audit_at_operating_scalars, converged_channel_aware_verdict,
    };
    let basis = segment_indicator_columns(64, &[16, 8, 4, 2, 1, 32]);
    let pilot_recipes = vec![
        vec![vec![(0, 1.0)], vec![(1, 1.0)]],
        vec![vec![(0, 1.0)], vec![(2, 1.0)]],
        vec![vec![(5, 1.0)], vec![(4, 1.0)]],
    ];
    let fixture = |converged_c: Vec<(usize, f64)>| {
        two_channel_specs(
            &basis,
            &pilot_recipes,
            &[
                None,
                Some(vec![vec![(3, 1.0)], vec![(2, 1.0)]]),
                Some(vec![converged_c, vec![(4, 1.0)]]),
            ],
            true,
        )
    };
    let scales = vec![vec![1.0; 2]; 3];
    let pilot = operating_point(scales.clone(), false);
    let converged = operating_point(scales, true);
    let beta_pilot = vec![0.0; 6];
    let beta_current = vec![0.5; 6];
    let change = audit_beta_relative_change(&beta_pilot, &beta_current);
    let drops = |audit: &IdentifiabilityAudit| {
        audit
            .dropped_columns
            .iter()
            .map(|dropped| (dropped.block.clone(), dropped.column))
            .collect::<Vec<(String, usize)>>()
    };

    let swap_specs = fixture(vec![(0, 1.0), (3, 1.0)]);
    let canonical = canonicalize_for_identifiability_with_operating_scalars(
        &swap_specs,
        &[CoefficientCoordinate::Spanning; 3],
        Some(Arc::clone(&pilot)),
    )
    .expect("a penalty-covered cross-block alias canonicalizes");
    assert!(canonical.used_channel_aware_audit);
    assert_eq!(
        drops(&canonical.audit),
        vec![("surface_2".to_string(), 0)],
        "B absorbs the pilot alias b = e0"
    );
    assert!(!canonical.audit.fatal, "the penalty covers the alias");
    assert_eq!(canonical.reduced_specs[1].design.ncols(), 1, "the reducing gauge removes b");
    let current =
        channel_aware_audit_at_operating_scalars(&swap_specs, Some(Arc::clone(&converged)))
            .expect("channel-aware converged audit runs");
    assert_eq!(
        drops(&current),
        vec![("surface_3".to_string(), 0)],
        "C absorbs the converged alias c = e0 + b"
    );
    let label_drift = audit_verdict_drift(&canonical.audit, &current, change, 0);
    assert!(
        label_drift.verdict_changed()
            && label_drift.pilot_rank == label_drift.current_rank
            && label_drift.newly_dropped.len() == 1
            && label_drift.recovered.len() == 1,
        "positive control: comparing drop labels refuses a swap of representatives at an \
         unchanged rank (pilot_rank={} current_rank={} newly_dropped={} recovered={})",
        label_drift.pilot_rank,
        label_drift.current_rank,
        label_drift.newly_dropped.len(),
        label_drift.recovered.len()
    );
    let swap = converged_channel_aware_verdict(
        &swap_specs,
        &canonical,
        Arc::clone(&converged),
        change,
        0,
    )
    .expect("channel-aware converged verdict runs");
    assert!(
        swap.pilot_gauge_reaudit.is_some(),
        "a reducing gauge is re-audited at convergence"
    );
    assert!(
        !swap.refuses() && swap.representative_swap(),
        "the pilot's kept columns are still full rank at convergence, so a swap of \
         representatives must be accepted (pilot_rank={} current_rank={} pilot_gauge_rank={})",
        swap.drift.pilot_rank,
        swap.drift.current_rank,
        swap.pilot_gauge_rank()
    );

    let loss_specs = fixture(vec![(0, 1.0)]);
    let loss_canonical = canonicalize_for_identifiability_with_operating_scalars(
        &loss_specs,
        &[CoefficientCoordinate::Spanning; 3],
        Some(Arc::clone(&pilot)),
    )
    .expect("the same pilot canonicalizes");
    assert_eq!(
        drops(&loss_canonical.audit),
        vec![("surface_2".to_string(), 0)],
        "the loss fixture shares the pilot's drop"
    );
    let loss = converged_channel_aware_verdict(&loss_specs, &loss_canonical, converged, change, 0)
        .expect("channel-aware converged verdict runs");
    assert_eq!(
        loss.drift.pilot_rank, loss.drift.current_rank,
        "b carries e3, so the raw rank agrees"
    );
    assert!(
        loss.refuses() && loss.pilot_gauge_rank() < loss.drift.pilot_rank,
        "the pilot's gauge removed b, the only column carrying e3 at convergence, so the \
         problem the fit ran lost rank and must refuse (pilot_rank={} pilot_gauge_rank={})",
        loss.drift.pilot_rank,
        loss.pilot_gauge_rank()
    );
}

#[test]
fn penalty_covered_competing_risks_redundancy_canonicalises_cleanly_1590() {
    let n = 64;
    let x = Array1::linspace(-1.0_f64, 1.0, n);
    let mut legendre = Array2::zeros((n, 6));
    for row in 0..n {
        legendre[[row, 0]] = 1.0;
        legendre[[row, 1]] = x[row];
        for degree in 2..6 {
            legendre[[row, degree]] =
                ((2 * degree - 1) as f64 * x[row] * legendre[[row, degree - 1]]
                    - (degree - 1) as f64 * legendre[[row, degree - 2]])
                    / degree as f64;
        }
    }
    // Eight coefficient directions, six independent likelihood directions:
    // block one repeats P0+P1 internally; block two shares 50*(P0+P1+P2)
    // with block one and also carries the three genuinely new P3/P4/P5 axes.
    let first = Array2::from_shape_fn((2 * n, 4), |(row, column)| {
        let row = row % n;
        if column < 3 {
            legendre[[row, column]]
        } else {
            legendre[[row, 0]] + legendre[[row, 1]]
        }
    });
    let second = Array2::from_shape_fn((2 * n, 4), |(row, column)| {
        let row = row % n;
        if column < 3 {
            legendre[[row, column + 3]]
        } else {
            50.0 * (legendre[[row, 0]] + legendre[[row, 1]] + legendre[[row, 2]])
        }
    });
    let specs: Vec<_> = [first.clone(), second.clone()]
        .into_iter()
        .enumerate()
        .map(|(block, full)| {
            let mut spec = spec_from_dense(
                &format!("time_cause_{}", block + 1),
                full.slice(s![..n, ..]).to_owned(),
            );
            spec.penalties = vec![PenaltyMatrix::Dense(Array2::eye(4))];
            spec.initial_log_lambdas = Array1::zeros(1);
            spec.nullspace_dims = vec![0];
            spec.jacobian_callback = Some(Arc::new(FixedTwoChannelJacobian { full, n }));
            spec
        })
        .collect();
    let rank = |matrix: &Array2<f64>| {
        let gram = matrix.t().dot(matrix);
        let (values, _) = gram
            .eigh(faer::Side::Lower)
            .expect("independent small-Gram rank oracle");
        let largest = values.iter().copied().fold(0.0_f64, f64::max);
        assert!(largest > 1.0 && values.iter().all(|value| value.is_finite()));
        let tolerance = 128.0 * f64::EPSILON * largest;
        values.iter().filter(|&&value| value > tolerance).count()
    };
    let mut joint = Array2::zeros((2 * n, 8));
    joint.slice_mut(s![.., ..4]).assign(&first);
    joint.slice_mut(s![.., 4..]).assign(&second);
    assert_eq!(
        rank(&joint),
        6,
        "the fixture must expose exactly two redundancies"
    );
    let canonical = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[CoefficientCoordinate::Spanning; 2],
        None,
    )
    .expect("penalty-covered multi-channel redundancy canonicalizes");
    assert!(canonical.used_channel_aware_audit);
    let reduced_width: usize = canonical
        .reduced_specs
        .iter()
        .map(|spec| spec.design.ncols())
        .sum();
    assert_eq!(
        reduced_width, 7,
        "one retained coefficient is identified by its penalty"
    );
    let mut reduced_joint = Array2::zeros((2 * n, reduced_width));
    let mut offset = 0;
    for spec in &canonical.reduced_specs {
        let width = spec.design.ncols();
        let beta = vec![0.0; width];
        let state = FamilyLinearizationState {
            beta: &beta,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        };
        let jacobian = spec
            .effective_jacobian_at("#1590 recovery", &state)
            .unwrap();
        assert_eq!(jacobian.dim(), (2 * n, width));
        reduced_joint
            .slice_mut(s![.., offset..offset + width])
            .assign(&jacobian);
        offset += width;
        assert_eq!(spec.penalties.len(), 1);
        assert_eq!(
            spec.penalties[0].as_dense_cow().as_ref(),
            &Array2::<f64>::eye(width),
            "the retained data-null direction must remain penalty-identified"
        );
    }
    assert_eq!(
        rank(&reduced_joint),
        6,
        "no real likelihood direction may be discarded"
    );
    eprintln!(
        "#1590 multi-channel canonicalization: raw_width=8 raw_rank=6 retained_width={reduced_width} retained_rank=6"
    );
}
