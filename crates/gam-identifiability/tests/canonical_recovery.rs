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
                design: design.clone(),
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
