//! Local error accounting for a gated linear library along an actual execution.
//!
//! These are measured row-wise transfers at one use site. The algebraic closure is
//! not a bound for unobserved inputs or a certificate for the composed network.

use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::{Array2, ArrayView2};

/// Signed error terms in the native read and write coordinates of one site.
#[derive(Debug)]
pub struct SiteError {
    /// Actual input minus the clean input.
    pub incoming: Array2<f64>,
    /// Incoming error propagated through the source map.
    pub propagated: Array2<f64>,
    /// Omitted library output evaluated on the clean input.
    pub clean_omission: Array2<f64>,
    /// Omitted library output evaluated on the incoming error.
    pub incoming_omission: Array2<f64>,
    /// All-on library output minus source output, on the actual input.
    pub reconstruction: Array2<f64>,
    /// Gated actual output minus clean source output.
    pub observed: Array2<f64>,
    /// Floating-point residual of the signed transfer identity.
    pub closure: Array2<f64>,
}

/// Account for `observed = propagated - clean_omission - incoming_omission
/// + reconstruction`, without constructing a gated dense map for each row.
///
/// `w` is written × read; `v` is components × read; `u` is components ×
/// written. Inputs are rows × read and `mask` is rows × components. Gates
/// may be any finite number in [0, 1]. Terms retain native coordinates;
/// their Euclidean magnitudes need not be comparable between different sites.
pub fn linear_site(
    w: ArrayView2<'_, f64>,
    v: ArrayView2<'_, f64>,
    u: ArrayView2<'_, f64>,
    clean: ArrayView2<'_, f64>,
    actual: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, f64>,
) -> Result<SiteError, String> {
    let (written, read) = w.dim();
    let components = v.nrows();
    let rows = clean.nrows();
    if v.ncols() != read
        || u.dim() != (components, written)
        || clean.ncols() != read
        || actual.dim() != clean.dim()
        || mask.dim() != (rows, components)
    {
        return Err("linear site: incompatible map, library, input, or mask shapes".into());
    }
    for (name, values) in [
        ("w", w),
        ("v", v),
        ("u", u),
        ("clean", clean),
        ("actual", actual),
        ("mask", mask),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("linear site: {name} contains a nonfinite value"));
        }
    }
    if mask.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err("linear site: gates must lie in [0, 1]".into());
    }
    let incoming = &actual - &clean;
    let propagated = fast_abt(&incoming, &w);
    let omitted = mask.mapv(|gate| 1.0 - gate);
    let clean_reads = fast_abt(&clean, &v);
    let incoming_reads = fast_abt(&incoming, &v);
    let actual_reads = fast_abt(&actual, &v);
    let clean_omission = fast_ab(&(&clean_reads * &omitted), &u);
    let incoming_omission = fast_ab(&(&incoming_reads * &omitted), &u);
    let reconstruction = fast_ab(&actual_reads, &u) - fast_abt(&actual, &w);
    let observed = fast_ab(&(&actual_reads * &mask), &u) - fast_abt(&clean, &w);
    let closure =
        &observed - &(&propagated - &clean_omission - &incoming_omission + &reconstruction);
    let result = SiteError {
        incoming,
        propagated,
        clean_omission,
        incoming_omission,
        reconstruction,
        observed,
        closure,
    };
    if [
        &result.incoming,
        &result.propagated,
        &result.clean_omission,
        &result.incoming_omission,
        &result.reconstruction,
        &result.observed,
        &result.closure,
    ]
    .iter()
    .any(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err("linear site: error accounting overflowed to a nonfinite value".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::linear_site;
    use ndarray::{Array2, array};

    fn close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.dim(), expected.dim());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn two_shears_require_omission_on_incoming_error() {
        let first_w = array![[1., 2.], [0., 1.]];
        let first_v = array![[1., 0.], [0., 1.], [0., 1.]];
        let first_u = array![[1., 0.], [0., 1.], [2., 0.]];
        let input = array![[0., 1.]];
        let gate = array![[1., 1., 0.]];
        let first = linear_site(
            first_w.view(),
            first_v.view(),
            first_u.view(),
            input.view(),
            input.view(),
            gate.view(),
        )
        .unwrap();
        close(&first.observed, &array![[-2., 0.]]);
        let second_w = array![[1., 0.], [3., 1.]];
        let second_v = array![[1., 0.], [0., 1.], [1., 0.]];
        let second_u = array![[1., 0.], [0., 1.], [0., 3.]];
        let clean = input.dot(&first_w.t());
        let independent_second = linear_site(
            second_w.view(),
            second_v.view(),
            second_u.view(),
            clean.view(),
            clean.view(),
            gate.view(),
        )
        .unwrap();
        close(&independent_second.observed, &array![[0., -6.]]);
        close(&independent_second.incoming_omission, &array![[0., 0.]]);
        let actual = &clean + &first.observed;
        let second = linear_site(
            second_w.view(),
            second_v.view(),
            second_u.view(),
            clean.view(),
            actual.view(),
            gate.view(),
        )
        .unwrap();
        close(&second.propagated, &array![[-2., -6.]]);
        close(&second.clean_omission, &array![[0., 6.]]);
        close(&second.incoming_omission, &array![[0., -6.]]);
        close(&second.observed, &array![[-2., -6.]]);
        close(&second.closure, &array![[0., 0.]]);
        // Independent clean omissions alone would incorrectly predict -12.
        close(
            &(&second.propagated - &second.clean_omission),
            &array![[-2., -12.]],
        );
        close(
            &(first.observed.dot(&second_w.t()) + independent_second.observed),
            &array![[-2., -12.]],
        );
    }

    #[test]
    fn identity_native_shears_have_zero_individual_readout_error_and_unit_joint_error() {
        let w = Array2::eye(2);
        let first_v = array![[1., 0.], [0., 1.], [1., 0.], [1., 0.]];
        let first_u = array![[1., 0.], [0., 1.], [0., 0.01], [0., -0.01]];
        let second_v = array![[1., 0.], [0., 1.], [0., 1.], [0., 1.]];
        let second_u = array![[1., 0.], [0., 1.], [100., 0.], [-100., 0.]];
        let clean = array![[1., 0.]];
        let gates = array![[1., 1., 1., 0.]];
        let first = linear_site(
            w.view(),
            first_v.view(),
            first_u.view(),
            clean.view(),
            clean.view(),
            gates.view(),
        )
        .unwrap();
        let second_alone = linear_site(
            w.view(),
            second_v.view(),
            second_u.view(),
            clean.view(),
            clean.view(),
            gates.view(),
        )
        .unwrap();
        assert_eq!(first.observed[[0, 0]], 0.);
        assert_eq!(second_alone.observed[[0, 0]], 0.);
        let actual = &clean + &first.observed;
        let joint = linear_site(
            w.view(),
            second_v.view(),
            second_u.view(),
            clean.view(),
            actual.view(),
            gates.view(),
        )
        .unwrap();
        close(&joint.clean_omission, &array![[0., 0.]]);
        close(&joint.incoming_omission, &array![[-1., 0.]]);
        close(&joint.reconstruction, &array![[0., 0.]]);
        close(&joint.observed, &array![[1., 0.01]]);
        close(&joint.closure, &array![[0., 0.]]);
    }

    #[test]
    fn imperfect_all_on_library_reports_reconstruction() {
        let w = array![[2.]];
        let v = array![[1.]];
        let u = array![[3.]];
        let clean = array![[4.]];
        let actual = array![[5.]];
        let mask = array![[1.]];
        let result = linear_site(
            w.view(),
            v.view(),
            u.view(),
            clean.view(),
            actual.view(),
            mask.view(),
        )
        .unwrap();
        close(&result.propagated, &array![[2.]]);
        close(&result.reconstruction, &array![[5.]]);
        close(&result.clean_omission, &array![[0.]]);
        close(&result.incoming_omission, &array![[0.]]);
        close(&result.observed, &array![[7.]]);
        close(&result.closure, &array![[0.]]);
    }

    #[test]
    fn exact_library_closes_for_fractional_gates_and_multiple_rows() {
        let w = array![[2., -1.], [1., 3.]];
        let v = Array2::eye(2);
        let u = w.t().to_owned();
        let clean = array![[1., 2.], [-3., 4.]];
        let actual = array![[2., 1.], [5., -2.]];
        let mask = array![[0.25, 0.75], [0., 1.]];
        let result = linear_site(
            w.view(),
            v.view(),
            u.view(),
            clean.view(),
            actual.view(),
            mask.view(),
        )
        .unwrap();
        close(&result.reconstruction, &Array2::zeros((2, 2)));
        close(&result.closure, &Array2::zeros((2, 2)));
        close(&result.observed, &array![[0.25, -4.25], [12., -15.]]);
    }

    #[test]
    fn rejects_bad_shapes_nonfinite_values_and_invalid_gates() {
        let good = array![[1.]];
        let bad_shape = array![[1., 2.]];
        let nan = array![[f64::NAN]];
        let infinity = array![[f64::INFINITY]];
        let negative = array![[-0.1]];
        let above_one = array![[1.1]];
        assert!(
            linear_site(
                good.view(),
                bad_shape.view(),
                good.view(),
                good.view(),
                good.view(),
                good.view()
            )
            .is_err()
        );
        assert!(
            linear_site(
                good.view(),
                good.view(),
                good.view(),
                good.view(),
                bad_shape.view(),
                good.view()
            )
            .is_err()
        );
        for bad in [&nan, &infinity] {
            for position in 0..6 {
                let mut inputs = [good.view(); 6];
                inputs[position] = bad.view();
                assert!(
                    linear_site(
                        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
                    )
                    .is_err()
                );
            }
        }
        for bad in [&negative, &above_one, &bad_shape] {
            assert!(
                linear_site(
                    good.view(),
                    good.view(),
                    good.view(),
                    good.view(),
                    good.view(),
                    bad.view()
                )
                .is_err()
            );
        }
    }
}
