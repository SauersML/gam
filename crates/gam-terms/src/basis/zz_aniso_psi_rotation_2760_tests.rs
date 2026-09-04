//! gam#2760 — a per-axis ψ-derivative bundle carried through the stage-2
//! joint-null absorption rotation `Q`. The realized design is `X·Q` and each
//! penalty block is `Qᵀ S Q`; since `Q` does not depend on ψ, every stored
//! derivative transforms the same way, and this pins that the bundle's rotation
//! is exactly those two products on every channel it carries — including the
//! streamed cross-penalty provider, whose blocks are produced after the fact.

#[cfg(test)]
mod zz_aniso_psi_rotation_2760_tests {
    use crate::basis::{AnisoBasisPsiDerivatives, AnisoPenaltyCrossProvider, JointNullRotation};
    use ndarray::{Array2, array};

    /// An exact 2×2 rotation (3-4-5 triangle), so `QᵀQ = I` to the bit.
    fn rotation() -> JointNullRotation {
        JointNullRotation {
            rotation: array![[0.6, -0.8], [0.8, 0.6]],
            joint_nullity: 1,
        }
    }

    fn design_block(seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((3, 2), |(i, j)| seed + i as f64 - 0.5 * j as f64)
    }

    fn penalty_block(seed: f64) -> Array2<f64> {
        // Symmetric, as every penalty block is.
        let a = Array2::from_shape_fn((2, 2), |(i, j)| seed + 0.25 * (i * 2 + j) as f64);
        &a + &a.t()
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        assert_eq!(a.dim(), b.dim(), "shapes must match to difference");
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn the_bundle_rotates_as_x_q_and_qt_s_q_on_every_channel_2760() {
        let q = rotation();
        let cross_seed = 7.0_f64;
        let bundle = AnisoBasisPsiDerivatives {
            design_first: vec![design_block(1.0), design_block(2.0)],
            design_second_diag: vec![design_block(3.0), design_block(4.0)],
            design_second_cross: vec![design_block(5.0)],
            design_second_cross_pairs: vec![(0, 1)],
            penalties_first: vec![vec![penalty_block(1.0)], vec![penalty_block(2.0)]],
            penalties_second_diag: vec![vec![penalty_block(3.0)], vec![penalty_block(4.0)]],
            penalties_cross_pairs: vec![(0, 1)],
            penalties_cross_provider: Some(AnisoPenaltyCrossProvider::new(move |a, b| {
                Ok(vec![penalty_block(cross_seed + a as f64 + b as f64)])
            })),
            implicit_operator: None,
        };

        let rotated = bundle
            .clone()
            .rotated_by_joint_null(&q)
            .expect("a square rotation matching the block width is admissible")
            .expect("every block admits the rotation");

        let expect_design = |source: &Array2<f64>| source.dot(&q.rotation);
        let expect_penalty = |source: &Array2<f64>| q.rotation.t().dot(source).dot(&q.rotation);

        for (axis, (got, want)) in rotated
            .design_first
            .iter()
            .zip(bundle.design_first.iter())
            .enumerate()
        {
            assert!(
                max_abs_diff(got, &expect_design(want)) <= 1e-12,
                "axis {axis}: dX/dψ must rotate as X·Q"
            );
        }
        for (axis, (got, want)) in rotated
            .design_second_diag
            .iter()
            .zip(bundle.design_second_diag.iter())
            .enumerate()
        {
            assert!(
                max_abs_diff(got, &expect_design(want)) <= 1e-12,
                "axis {axis}: d²X/dψ² must rotate as X·Q"
            );
        }
        assert!(
            max_abs_diff(
                &rotated.design_second_cross[0],
                &expect_design(&bundle.design_second_cross[0])
            ) <= 1e-12,
            "the cross second derivative must rotate as X·Q"
        );
        for (axis, (got, want)) in rotated
            .penalties_first
            .iter()
            .zip(bundle.penalties_first.iter())
            .enumerate()
        {
            assert!(
                max_abs_diff(&got[0], &expect_penalty(&want[0])) <= 1e-12,
                "axis {axis}: dS/dψ must rotate as QᵀSQ"
            );
        }
        for (axis, (got, want)) in rotated
            .penalties_second_diag
            .iter()
            .zip(bundle.penalties_second_diag.iter())
            .enumerate()
        {
            assert!(
                max_abs_diff(&got[0], &expect_penalty(&want[0])) <= 1e-12,
                "axis {axis}: d²S/dψ² must rotate as QᵀSQ"
            );
        }
        // The streamed cross-penalty provider produces its blocks on demand,
        // so its rotation has to travel with the closure.
        let streamed = rotated
            .penalties_cross_provider
            .as_ref()
            .expect("the provider survives the rotation")
            .evaluate(0, 1)
            .expect("the provider answers for the pair it advertises");
        assert!(
            max_abs_diff(&streamed[0], &expect_penalty(&penalty_block(cross_seed + 1.0)))
                <= 1e-12,
            "a streamed cross-penalty block must rotate as QᵀSQ"
        );
        // Non-vacuity: the rotation is not the identity on any of these blocks.
        assert!(
            max_abs_diff(&rotated.design_first[0], &bundle.design_first[0]) > 1e-3,
            "the fixture's rotation must actually move the blocks"
        );
    }

    #[test]
    fn a_block_the_rotation_cannot_act_on_is_declined_not_asserted_2760() {
        let q = rotation();
        let bundle = AnisoBasisPsiDerivatives {
            // Three coefficients against a two-coefficient rotation.
            design_first: vec![Array2::<f64>::zeros((3, 3))],
            design_second_diag: vec![Array2::<f64>::zeros((3, 3))],
            design_second_cross: Vec::new(),
            design_second_cross_pairs: Vec::new(),
            penalties_first: vec![vec![Array2::<f64>::zeros((3, 3))]],
            penalties_second_diag: vec![vec![Array2::<f64>::zeros((3, 3))]],
            penalties_cross_pairs: Vec::new(),
            penalties_cross_provider: None,
            implicit_operator: None,
        };
        assert!(
            bundle
                .rotated_by_joint_null(&q)
                .expect("a shape mismatch is a decline, never an error")
                .is_none(),
            "the caller must be able to fall back rather than assert"
        );
    }
}
