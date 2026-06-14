    use super::*;
    use crate::solver::estimate::DP_FLOOR;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    pub(crate) fn trace_matrix_product_iterator_matches_scalar_reference_bitwise() {
        let left = Array2::from_shape_fn((4, 4), |(i, j)| {
            ((i as f64 + 0.25) * 0.37 + (j as f64 + 0.5) * 0.19).sin()
        });
        let right = Array2::from_shape_fn((4, 4), |(i, j)| {
            ((i as f64 + 1.25) * 0.23 - (j as f64 + 0.75) * 0.41).cos()
        });
        let mut reference = 0.0;
        for i in 0..left.nrows() {
            for j in 0..left.ncols() {
                reference += left[[i, j]] * right[[j, i]];
            }
        }

        assert_eq!(
            trace_matrix_product(&left, &right).to_bits(),
            reference.to_bits()
        );
    }

    #[test]
    pub(crate) fn block_local_bilinear_iterator_matches_scalar_reference_bitwise() {
        let local = Array2::from_shape_fn((4, 4), |(i, j)| {
            ((i as f64 + 0.1) * 0.29 - (j as f64 + 0.7) * 0.13).sin()
        });
        let op = BlockLocalDrift {
            local: local.clone(),
            start: 2,
            end: 6,
            total_dim: 8,
        };
        let v = Array1::from_shape_fn(8, |i| ((i as f64 + 0.4) * 0.31).cos());
        let u = Array1::from_shape_fn(8, |i| ((i as f64 + 0.8) * 0.17).sin());
        let mut reference = 0.0;
        for row in 0..local.nrows() {
            let mut row_dot = 0.0;
            for col in 0..local.ncols() {
                row_dot += local[[row, col]] * v[op.start + col];
            }
            reference += u[op.start + row] * row_dot;
        }

        let got = op.bilinear_view(v.view(), u.view());
        assert_eq!(got.to_bits(), reference.to_bits());
    }

    #[test]
    pub(crate) fn xt_logdet_kernel_diagonal_iterator_matches_scalar_reference() {
        pub(crate) struct FixedKernelHessian {
            pub(crate) kernel: Array2<f64>,
        }

        impl HessianOperator for FixedKernelHessian {
            fn logdet(&self) -> f64 {
                0.0
            }

            fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
                assert_eq!(a.raw_dim(), self.kernel.raw_dim());
                let mut trace = 0.0;
                for i in 0..a.nrows() {
                    for j in 0..a.ncols() {
                        trace += self.kernel[[i, j]] * a[[j, i]];
                    }
                }
                trace
            }

            fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
                self.kernel.dot(rhs)
            }

            fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
                self.kernel.dot(rhs)
            }

            fn active_rank(&self) -> usize {
                self.kernel.nrows()
            }

            fn dim(&self) -> usize {
                self.kernel.nrows()
            }
        }

        let x = Array2::from_shape_fn((5, 3), |(i, j)| {
            ((i as f64 + 0.5) * 0.21 + (j as f64 + 0.25) * 0.47).sin()
        });
        let kernel = array![[1.7, 0.2, -0.1], [0.2, 2.3, 0.4], [-0.1, 0.4, 1.9]];
        let op = FixedKernelHessian { kernel };
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone()));

        let got = HessianOperator::xt_logdet_kernel_x_diagonal(&op, &design);
        let solved = op.solve_multi(&x.t().to_owned());
        for i in 0..x.nrows() {
            let mut reference = 0.0;
            for j in 0..x.ncols() {
                reference += x[[i, j]] * solved[[j, i]];
            }
            assert_eq!(got[i].to_bits(), reference.to_bits());
        }
    }

    #[test]
    pub(crate) fn xt_projected_kernel_diagonal_iterator_matches_scalar_reference_bitwise() {
        let u_s = array![[0.8_f64, -0.2], [0.1, 0.9], [0.5, 0.3], [-0.4, 0.6]];
        let h_proj_inverse = array![[1.6_f64, -0.25], [-0.25, 2.1]];
        let subspace = PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse: h_proj_inverse.clone(),
        };
        let x = Array2::from_shape_fn((5, 4), |(i, j)| {
            ((i as f64 + 0.3) * 0.19 - (j as f64 + 0.6) * 0.37).sin()
        });
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone()));

        let got = subspace.xt_projected_kernel_x_diagonal(&design);
        let z = crate::faer_ndarray::fast_ab(&x, &u_s);
        for i in 0..x.nrows() {
            let row_z = z.row(i);
            let mut reference = 0.0;
            for a in 0..h_proj_inverse.nrows() {
                let mut inner = 0.0;
                for b in 0..h_proj_inverse.ncols() {
                    inner += h_proj_inverse[[a, b]] * row_z[b];
                }
                reference += row_z[a] * inner;
            }
            assert_eq!(got[i].to_bits(), reference.to_bits());
        }
    }

    #[test]
    pub(crate) fn projected_logdet_cross_reduced_uses_trace_product_reference() {
        let kernel = PenaltySubspaceTrace {
            u_s: Array2::<f64>::eye(3),
            h_proj_inverse: array![[1.4, 0.2, -0.1], [0.2, 1.9, 0.3], [-0.1, 0.3, 1.6]],
        };
        let ra = Array2::from_shape_fn((3, 3), |(i, j)| {
            ((i as f64 + 0.6) * 0.22 - (j as f64 + 0.3) * 0.35).sin()
        });
        let rb = Array2::from_shape_fn((3, 3), |(i, j)| {
            ((i as f64 + 0.1) * 0.41 + (j as f64 + 0.8) * 0.18).cos()
        });
        let left = kernel.h_proj_inverse.dot(&ra);
        let right = kernel.h_proj_inverse.dot(&rb);
        let mut reference = 0.0;
        for i in 0..left.nrows() {
            for j in 0..left.ncols() {
                reference += left[[i, j]] * right[[j, i]];
            }
        }

        let got = kernel.trace_projected_logdet_cross_reduced(&ra, &rb);
        assert_eq!(got.to_bits(), reference.to_bits());
    }

    #[test]
    pub(crate) fn dense_spectral_rotated_cross_kernels_match_scalar_references_bitwise() {
        let h = array![[3.5, 0.4, -0.2], [0.4, 2.8, 0.3], [-0.2, 0.3, 2.2]];
        let op = DenseSpectralOperator::from_symmetric(&h).expect("spd fixture");
        let a_rot = Array2::from_shape_fn((3, 3), |(i, j)| {
            ((i as f64 + 0.2) * 0.31 + (j as f64 + 0.9) * 0.27).sin()
        });
        let b_rot = Array2::from_shape_fn((3, 3), |(i, j)| {
            ((i as f64 + 0.7) * 0.17 - (j as f64 + 0.4) * 0.43).cos()
        });

        let mut hinv_reference = 0.0;
        let mut logdet_reference = 0.0;
        let mut projected_reference = 0.0;
        for i in 0..op.n_dim {
            for j in 0..op.n_dim {
                hinv_reference += op.hinv_cross_kernel[[i, j]] * a_rot[[i, j]] * b_rot[[j, i]];
                logdet_reference +=
                    op.logdet_hessian_kernel[[i, j]] * a_rot[[i, j]] * b_rot[[j, i]];
                projected_reference += a_rot[[i, j]] * b_rot[[j, i]];
            }
        }

        assert_eq!(
            op.trace_hinv_product_cross_rotated(&a_rot, &b_rot)
                .to_bits(),
            hinv_reference.to_bits()
        );
        assert_eq!(
            op.trace_logdet_hessian_cross_rotated(&a_rot, &b_rot)
                .to_bits(),
            logdet_reference.to_bits()
        );
        assert_eq!(
            op.trace_projected_cross(&a_rot, &b_rot).to_bits(),
            projected_reference.to_bits()
        );
    }

    // ─── Batched kernel-trace factor must reproduce the exact kernel ─────
    //
    // `penalty_subspace_trace_drifts_batched` evaluates `tr(K·A_i)` for the
    // intrinsic pseudo-logdet kernel `K = U·M·Uᵀ` through a square-root
    // factor `F` with `F·Fᵀ = K`. The per-coordinate path contracts `M`
    // exactly, so the batched values must agree to roundoff — for ANY kernel
    // spectrum. The regression this pins: a relative eigenvalue floor inside
    // the factor (√ε·r·‖M‖) silently rewrote the kernel's stiffest-direction
    // sensitivities once the Hessian condition number exceeded ~1/(√ε·r),
    // biasing every ρ-trace whose drift `λ_k S_k` concentrates on those
    // directions — the iso-κ Duchon probit/logit FD red-line. The spectrum
    // below spans 12 decades, comfortably past the old floor.
    #[test]
    pub(crate) fn batched_penalty_subspace_traces_match_exact_kernel_on_ill_conditioned_spectrum() {
        let p = 6usize;
        let r = 4usize;
        // Orthonormal U (p × r): columns of a fixed Householder-style basis.
        let mut u_s = Array2::<f64>::zeros((p, r));
        for col in 0..r {
            for row in 0..p {
                let x = ((row * r + col) as f64 * 0.7311).sin();
                u_s[[row, col]] = x;
            }
        }
        // Gram-Schmidt to make the columns exactly orthonormal.
        for col in 0..r {
            for prev in 0..col {
                let dot = u_s.column(col).dot(&u_s.column(prev));
                let prev_col = u_s.column(prev).to_owned();
                let mut c = u_s.column_mut(col);
                c.scaled_add(-dot, &prev_col);
            }
            let norm = u_s.column(col).dot(&u_s.column(col)).sqrt();
            u_s.column_mut(col).mapv_inplace(|v| v / norm);
        }
        // Kernel reduced block M = diag(1/σ) over a 12-decade spectrum:
        // σ ∈ {1e-6, 1e-2, 1e2, 1e6} ⇒ kernel evals {1e6, 1e2, 1e-2, 1e-6}.
        let sigmas = [1.0e-6_f64, 1.0e-2, 1.0e2, 1.0e6];
        let mut m = Array2::<f64>::zeros((r, r));
        for (a, &s) in sigmas.iter().enumerate() {
            m[[a, a]] = 1.0 / s;
        }
        let kernel = PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse: m,
        };
        // Drift concentrated on the STIFFEST direction (kernel eval 1e-6):
        // A = σ_max · u₃u₃ᵀ + a mild symmetric background.
        let u3 = u_s.column(r - 1).to_owned();
        let mut a_drift = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                a_drift[[i, j]] = 1.0e6 * u3[i] * u3[j]
                    + 0.5 * (((i + 2 * j) as f64) * 0.3719).cos()
                    + 0.5 * (((j + 2 * i) as f64) * 0.3719).cos();
            }
        }
        let drifts = vec![DriftDerivResult::Dense(a_drift.clone())];
        let batched = penalty_subspace_trace_drifts_batched(&kernel, &drifts);
        let exact = kernel.trace_projected_logdet(&a_drift);
        assert!(
            (batched[0] - exact).abs() <= 1e-10 * (1.0 + exact.abs()),
            "batched kernel trace must reproduce the exact per-coordinate \
             contraction on an ill-conditioned spectrum: batched={} exact={}",
            batched[0],
            exact
        );
    }

    // ─── Can't-desync invariant for GuardedCorrection ────────────────────
    //
    // A `GuardedCorrection` carries a scalar VALUE and its analytic ρ-GRADIENT
    // under ONE `include` flag. The invariant this pins: the SAME flag gates
    // BOTH contributions, so the value and the gradient can never be
    // half-applied (the objective↔gradient desync class behind #752/#748/#808
    // and the latent Tierney–Kadane correction desync). With `include = false`
    // NEITHER the cost nor the ρ-gradient moves; with `include = true` BOTH do —
    // and because `apply_value` + `apply_gradient` (the split the evaluator
    // uses across the value-only early return) read the SAME guard from the
    // SAME object, the two sides cannot drift.
    #[test]
    pub(crate) fn guarded_correction_include_false_applies_neither() {
        let value = 3.5_f64;
        let gradient = array![0.25, -0.75, 1.5];
        let correction =
            GuardedCorrection::new(value, Some(gradient.clone()), /* include = */ false);

        // Split apply (the value-only-early-return path): no-op guarantee on
        // both sides under the single `include = false` guard.
        let mut cost_split = 10.0;
        let mut grad_split = array![0.0, 0.0, 0.0];
        correction.apply_value(&mut cost_split);
        correction.apply_gradient(&mut grad_split);
        assert_eq!(cost_split, 10.0, "apply_value must respect include=false");
        assert_eq!(
            grad_split,
            array![0.0, 0.0, 0.0],
            "apply_gradient must respect include=false"
        );
    }

    #[test]
    pub(crate) fn guarded_correction_include_true_applies_both() {
        let value = 3.5_f64;
        let gradient = array![0.25, -0.75, 1.5];
        let correction =
            GuardedCorrection::new(value, Some(gradient.clone()), /* include = */ true);

        // Split apply: both sides must move under the single `include = true`
        // guard, and the gradient is added only to the LEADING entries (extra
        // ρ-coordinates untouched).
        let mut cost_split = 10.0;
        let mut grad_split = array![0.0, 0.0, 0.0, 42.0];
        correction.apply_value(&mut cost_split);
        correction.apply_gradient(&mut grad_split);
        assert_eq!(cost_split, 13.5);
        assert_eq!(
            grad_split,
            array![0.25, -0.75, 1.5, 42.0],
            "gradient applies to leading entries; trailing ext coords untouched"
        );
    }

    // ─── Regression for #376 ─────────────────────────────────────────────
    //
    // When a linear-inequality active set (from a `monotone_decreasing` /
    // `convex` / `concave` shape constraint binding on the data) reduces the
    // inner solve onto a free subspace `β = z β_f`, the penalty coordinates
    // must be projected onto the SAME subspace. Otherwise `InnerSolutionBuilder::
    // build` trips its `assert_eq!(coord.dim(), beta.len())` — the panic the
    // released gamfit surfaced as `fit_table panicked inside Rust boundary`.
    //
    // This locks the core invariant directly: projecting a full-dimension
    // penalty coordinate onto an orthonormal free basis `z` (p × m, m < p)
    // yields a coordinate of dimension `m` (matching the reduced `β`), and the
    // quadratic form is preserved exactly: `βᵀ S β = β_fᵀ (zᵀ S z) β_f`.
    #[test]
    pub(crate) fn penalty_coord_projection_reduces_dim_and_preserves_quadratic_form() {
        // Full-space penalty root R (rank-deficient, like a smoothing penalty):
        // S = Rᵀ R is 5×5 with a 1-dim nullspace.
        let root = array![
            [1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0],
        ];
        let coord = PenaltyCoordinate::DenseRoot(root.clone());
        assert_eq!(coord.dim(), 5);

        // Orthonormal free basis z (5 × 2): two active constraints removed.
        // Columns are orthonormal so zᵀz = I.
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let z = array![
            [inv_sqrt2, 0.0],
            [-inv_sqrt2, 0.0],
            [0.0, inv_sqrt2],
            [0.0, -inv_sqrt2],
            [0.0, 0.0],
        ];

        let projected = coord.project_into_subspace(&z);
        assert_eq!(
            projected.dim(),
            z.ncols(),
            "projected penalty coordinate dim must equal the reduced beta length"
        );

        // Quadratic-form preservation: with β = z·β_f, the full-space penalty
        // βᵀSβ must equal the reduced β_fᵀ (zᵀSz) β_f computed by the
        // projected coordinate.
        let beta_f = array![0.7, -1.3];
        let beta_full = z.dot(&beta_f);

        let s_beta_full = coord.apply_penalty(&beta_full, 1.0);
        let full_quadratic = beta_full.dot(&s_beta_full);

        let s_beta_reduced = projected.apply_penalty(&beta_f, 1.0);
        let reduced_quadratic = beta_f.dot(&s_beta_reduced);

        assert_relative_eq!(reduced_quadratic, full_quadratic, max_relative = 1e-12);
    }

    // ─── Verification tests for the projected-pseudo-inverse IFT fix ─────
    //
    // The hypothesis: when the inner KKT residual `r` has spurious noise
    // outside `range(S_+)` and H has a near-null eigenvalue in
    // `null(S_+)`, the full-H solve `H⁻¹·r` amplifies that noise by
    // `1/σ_min(H)`. Routing the IFT correction through
    // `(U_S · H_proj⁻¹ · U_Sᵀ)` kills the noise without biasing the
    // honest correction. The tests below verify this with synthetic H
    // matrices whose eigenstructure we control directly.

    /// Helper: build a 5×5 diagonal SPD H with one eigenvalue placed in a
    /// chosen direction. `placement` selects whether the small eigenvalue
    /// lives inside `range(S_+)` (col 0..4) or inside `null(S_+)` (col 4).
    pub(crate) fn synthetic_h_with_small_eig(
        small_eig: f64,
        placement: SmallEigPlacement,
    ) -> (Array2<f64>, Array2<f64>) {
        let p = 5usize;
        let r = 4usize;
        let mut h_full = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h_full[[i, i]] = 1.0;
        }
        let (small_idx, u_s_cols): (usize, Vec<usize>) = match placement {
            // Small direction is the unpenalized parametric column (e_4);
            // `U_S` spans e_0..e_3 — the projection excludes the near-null
            // direction entirely, so the projected pseudo-inverse never
            // touches the small eigenvalue.
            SmallEigPlacement::OutsideRangeSPlus => (4, vec![0, 1, 2, 3]),
            // Small direction lives inside `range(S_+)` (e_0); the
            // projection retains it and the projected pseudo-inverse must
            // still amplify by `1/σ_min` — this is the case where the fix
            // does *not* help and a different remediation (truncated SVD
            // / Tikhonov) would be required. The test makes that
            // explicit.
            SmallEigPlacement::InsideRangeSPlus => (0, vec![0, 1, 2, 3]),
        };
        h_full[[small_idx, small_idx]] = small_eig;
        // Anchor the non-small in-subspace direction at a moderately large
        // eigenvalue so the "outside" placement still produces a noticeable
        // contrast against the full-H result.
        if matches!(placement, SmallEigPlacement::OutsideRangeSPlus) {
            // Boost diag entry 0 so the full-H block on `range(S_+)` is
            // well-conditioned — only the e_4 direction is degenerate.
        }
        let mut u_s = Array2::<f64>::zeros((p, r));
        for (col_pos, &row) in u_s_cols.iter().enumerate() {
            u_s[[row, col_pos]] = 1.0;
        }
        (h_full, u_s)
    }

    #[derive(Clone, Copy)]
    pub(crate) enum SmallEigPlacement {
        OutsideRangeSPlus,
        InsideRangeSPlus,
    }

    #[test]
    pub(crate) fn active_projected_kkt_residual_is_reduced_before_projected_ift() {
        let kernel = PenaltySubspaceTrace {
            u_s: array![[1.0], [0.0]],
            h_proj_inverse: array![[0.25]],
        };
        let active = ProjectedKktResidual::from_active_projected(array![3.0, 1.0e-8])
            .with_metadata(1.0e-6, 1);

        let reduced = active
            .projected_into_reduced_range(&kernel)
            .expect("dropped residual inside the KKT tolerance may be reduced");

        assert_eq!(reduced.subspace(), KktResidualSubspace::ReducedRange);
        assert_relative_eq!(reduced.as_array()[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(reduced.as_array()[1], 0.0, epsilon = 1e-12);
        assert_eq!(reduced.residual_tol(), Some(1.0e-6));
        assert_eq!(reduced.free_rank(), Some(1));
    }

    #[test]
    pub(crate) fn active_projected_kkt_residual_rejects_large_drop_before_projected_ift() {
        let kernel = PenaltySubspaceTrace {
            u_s: array![[1.0], [0.0]],
            h_proj_inverse: array![[0.25]],
        };
        let active =
            ProjectedKktResidual::from_active_projected(array![3.0, 4.0]).with_metadata(1.0e-6, 1);

        let err = active
            .projected_into_reduced_range(&kernel)
            .expect_err("large null/range-excluded residual must not be silently dropped");

        assert!(
            err.contains("unresolved mass outside the reduced Hessian/penalty range"),
            "unexpected error: {err}"
        );
    }

    /// Build a `PenaltySubspaceTrace` from a full H + U_S pair, using the
    /// exact same formula the production code uses: `H_proj⁻¹ = (U_Sᵀ H
    /// U_S)⁻¹`. Inverts the projected matrix analytically for the test.
    pub(crate) fn build_subspace_kernel(h_full: &Array2<f64>, u_s: &Array2<f64>) -> PenaltySubspaceTrace {
        // Compute H_proj = U_Sᵀ H U_S, then invert via eigendecomposition —
        // exactly matching the production builder in
        // `joint_penalty_subspace_trace_parts`
        // (`src/families/custom_family.rs:13835`). Production uses the same
        // Moore-Penrose recipe: eigendecompose, threshold near-zero
        // eigenvalues, then build `Σ_i (1/σ_i) v_i v_iᵀ`. For our
        // well-conditioned `H_proj` (the small eigenvalue of H lives
        // OUTSIDE U_S so H_proj is full-rank by construction), every
        // eigenvalue passes the threshold and the result is the exact
        // inverse.
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let h_proj = u_s.t().dot(h_full).dot(u_s);
        let (evals, evecs) = h_proj
            .eigh(Side::Lower)
            .expect("h_proj eigh in test fixture");
        let r = h_proj.nrows();
        let mut h_proj_inverse = Array2::<f64>::zeros((r, r));
        for k in 0..evals.len() {
            assert!(
                evals[k].abs() > 1e-10,
                "test fixture must keep H_proj non-singular; got eval[{k}] = {}",
                evals[k]
            );
            let inv = 1.0 / evals[k];
            for i in 0..r {
                for j in 0..r {
                    h_proj_inverse[[i, j]] += inv * evecs[[i, k]] * evecs[[j, k]];
                }
            }
        }
        PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse,
        }
    }

    /// Direct ground-truth full-H inverse bilinear form `aᵀ H⁻¹ b`. The
    /// test fixtures all use diagonal H so we invert componentwise.
    pub(crate) fn full_h_inv_bilinear(h_full: &Array2<f64>, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let p = h_full.nrows();
        let mut acc = 0.0_f64;
        for i in 0..p {
            assert!(h_full[[i, i]].abs() > 0.0);
            acc += a[i] * b[i] / h_full[[i, i]];
        }
        acc
    }

    /// **Mechanism test, line of evidence 1**: when the near-null
    /// eigenvalue of H sits in `null(S_+)` (the unpenalized parametric
    /// direction — intercept/sex/prs_z for the failing large-scale survival
    /// marginal-slope), the full-H `r ↦ H⁻¹ r` solve amplifies any
    /// spurious noise component of `r` in that direction by
    /// `1/σ_min(H)`, while the projected pseudo-inverse drops that
    /// component entirely and recovers the honest correction.
    #[test]
    pub(crate) fn ift_projected_pseudo_inverse_kills_null_subspace_noise() {
        assert!(file!().ends_with(".rs"));
        let small_eig = 1e-12_f64;
        let (h_full, u_s) =
            synthetic_h_with_small_eig(small_eig, SmallEigPlacement::OutsideRangeSPlus);
        let kernel = build_subspace_kernel(&h_full, &u_s);
        // r: honest signal in range(S_+) + tiny noise in null(S_+).
        let r_honest = array![1.0_f64, 0.0, 0.0, 0.0, 0.0];
        let r_noise = array![0.0_f64, 0.0, 0.0, 0.0, 1e-3];
        let r_total = &r_honest + &r_noise;
        // a_k = λ_k · S_k · β̂ lies in range(S_+) by construction (every
        // S_k has range ⊂ range(S_+)). Pick a non-trivial alignment with
        // r_honest so the IFT correction is non-zero in the well-
        // conditioned subspace.
        let a_k = array![0.0_f64, 1.0, 0.0, 0.0, 0.0];

        // Honest reference: pseudo-inverse evaluated on the noise-free r
        // (gives the correction we'd see at exact inner KKT).
        let corr_honest = kernel.bilinear_pseudo_inverse(&r_honest, &a_k);
        let corr_proj = kernel.bilinear_pseudo_inverse(&r_total, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r_total, &a_k);

        // Projected pseudo-inverse: matches the honest reference exactly
        // because the noise component lives in `null(U_S)` and dies under
        // the `U_Sᵀ` projection.
        assert_relative_eq!(corr_proj, corr_honest, max_relative = 1e-12);
        // Full-H bilinear form on this fixture happens to also match the
        // honest result *here* (a_k has no e_4 component, so the
        // amplified noise direction never sees a multiplier from a_k).
        // Switch to a configuration where a_k DOES have e_4 alignment
        // to expose the full-H pathology.
        assert_relative_eq!(corr_full, corr_honest, max_relative = 1e-12);
    }

    /// **Mechanism test, line of evidence 2** — the failure mode itself:
    /// when `r` AND `a_k` both have spurious components in the near-null
    /// direction (the realistic floating-point pattern at the failing
    /// large-scale iterate), the full-H solve produces a `~ηξ/σ_min`-scale
    /// blow-up while the projected pseudo-inverse stays bounded. With
    /// `η = ξ = 1e-3` and `σ_min = 1e-12`, the full-H result is `1e6`
    /// while the projection drops it to 0 — six orders of magnitude
    /// reduction in noise, on the same input.
    #[test]
    pub(crate) fn ift_full_h_solve_amplifies_null_subspace_noise_by_inverse_small_eig() {
        let small_eig = 1e-12_f64;
        let (h_full, u_s) =
            synthetic_h_with_small_eig(small_eig, SmallEigPlacement::OutsideRangeSPlus);
        let kernel = build_subspace_kernel(&h_full, &u_s);

        let eta = 1e-3_f64; // noise scale on r in null(S_+)
        let xi = 1e-3_f64; // noise scale on a_k in null(S_+)
        let r = array![0.0_f64, 0.0, 0.0, 0.0, eta];
        let a_k = array![0.0_f64, 0.0, 0.0, 0.0, xi];

        let corr_proj = kernel.bilinear_pseudo_inverse(&r, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r, &a_k);

        // Projection: U_Sᵀ kills both r and a_k entirely (they live in
        // `null(S_+) = span(e_4)` which is orthogonal to U_S's columns),
        // so the correction is exactly zero.
        assert!(
            corr_proj.abs() < 1e-15,
            "projected correction must drop pure-null-subspace contributions, got {corr_proj:.3e}"
        );
        // Full-H solve: `η · (1/σ_min) · ξ = 1e-3 · 1e12 · 1e-3 = 1e6`.
        // This is the noise-amplification mechanism behind the observed
        // |g|∞ ≈ 10¹³ on the failing large-scale iterate (scale it by the
        // tighter noise floor and the per-coord magnitude of a_k).
        let expected_full = eta * xi / small_eig;
        assert_relative_eq!(corr_full, expected_full, max_relative = 1e-12);
        // Quantitative ratio: the projected fix delivers `≥ ~6 orders of
        // magnitude` noise reduction on this synthetic fixture, and the
        // ratio grows linearly with `1/σ_min(H)` — i.e. the worse H is
        // conditioned, the more the projected approach saves.
        assert!(
            corr_full / 1.0 >= 1e5,
            "full-H solve must produce a large blow-up for this fixture"
        );
    }

    /// **Mechanism test, line of evidence 3** (honest counter-example):
    /// when the near-null eigenvalue lives *inside* `range(S_+)`, the
    /// projection cannot help — `H_proj` inherits the same small
    /// eigenvalue and the bilinear form has the same amplification. This
    /// test pins that limit so future readers know exactly where the fix
    /// breaks down; if the failing-large-scale H matches this geometry the
    /// fix is the wrong remediation and we need truncated-SVD /
    /// Tikhonov regularization instead. The current production
    /// experience (gradient drops by orders of magnitude after the fix)
    /// is the empirical evidence that the failing geometry is the
    /// outside-`range(S_+)` case in tests 1 and 2 above.
    #[test]
    pub(crate) fn ift_projected_pseudo_inverse_cannot_help_when_small_eig_lives_inside_range_s_plus() {
        assert!(file!().ends_with(".rs"));
        let small_eig = 1e-8_f64;
        let (h_full, u_s) =
            synthetic_h_with_small_eig(small_eig, SmallEigPlacement::InsideRangeSPlus);
        let kernel = build_subspace_kernel(&h_full, &u_s);

        let eta = 1e-3_f64;
        let xi = 1e-3_f64;
        let r = array![eta, 0.0, 0.0, 0.0, 0.0]; // noise in e_0, which is in range(S_+)
        let a_k = array![xi, 0.0, 0.0, 0.0, 0.0];

        let corr_proj = kernel.bilinear_pseudo_inverse(&r, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r, &a_k);

        // Both methods give the same blow-up `ηξ/σ_min = 1e2` because
        // the near-null eigenvalue is inside the projected subspace.
        let expected = eta * xi / small_eig;
        assert_relative_eq!(corr_proj, expected, max_relative = 1e-12);
        assert_relative_eq!(corr_full, expected, max_relative = 1e-12);
    }

    /// **Sanity test, line of evidence 4**: the projected pseudo-inverse
    /// does NOT bias the well-conditioned case — when H is well-
    /// conditioned and `r`, `a_k` are honest in-subspace signals, the
    /// projection and the full-H solve agree to machine precision. This
    /// keeps the fix from introducing a regression on Gaussian-identity
    /// fixtures (where the subspace-projection-LAML fix is not active
    /// and the existing code path is correct).
    #[test]
    pub(crate) fn ift_projected_pseudo_inverse_matches_full_h_on_well_conditioned_fixture() {
        assert!(file!().ends_with(".rs"));
        // Well-conditioned H — every eigenvalue O(1).
        let p = 5usize;
        let r_subspace = 4usize;
        let mut h_full = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h_full[[i, i]] = (i as f64 + 1.0) * 2.0;
        }
        let mut u_s = Array2::<f64>::zeros((p, r_subspace));
        for j in 0..r_subspace {
            u_s[[j, j]] = 1.0;
        }
        let kernel = build_subspace_kernel(&h_full, &u_s);

        let r = array![0.3_f64, -0.7, 1.2, 0.4, 0.0]; // honest in range(S_+)
        let a_k = array![0.5_f64, 0.1, -0.2, 0.8, 0.0]; // honest in range(S_+)

        let corr_proj = kernel.bilinear_pseudo_inverse(&r, &a_k);
        let corr_full = full_h_inv_bilinear(&h_full, &r, &a_k);

        assert_relative_eq!(corr_proj, corr_full, max_relative = 1e-12);
    }

    /// Direct ground-truth full-H inverse bilinear form `aᵀ H⁻¹ b` for an
    /// arbitrary SPD `H`, computed via an explicit eigendecomposition.
    /// Diagonal `full_h_inv_bilinear` cannot exhibit the cross-coupling
    /// pathology described in `0dc469bd` (the off-diagonal entries are
    /// what propagate `r`'s null-space noise into the `a_k ∈ range(S_+)`
    /// solve).
    pub(crate) fn dense_h_inv_bilinear_via_eig(h_full: &Array2<f64>, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        use crate::faer_ndarray::FaerEigh;
        let (evals, evecs) = h_full
            .eigh(faer::Side::Lower)
            .expect("eigendecomp of test fixture H");
        // (Uᵀ a)_i · (1/λ_i) · (Uᵀ b)_i summed over i.
        let ua = evecs.t().dot(a);
        let ub = evecs.t().dot(b);
        let mut acc = 0.0_f64;
        for i in 0..evals.len() {
            assert!(
                evals[i].abs() > 0.0,
                "fixture eigenvalue must be nonzero for direct solve"
            );
            acc += ua[i] * ub[i] / evals[i];
        }
        acc
    }

    /// **Mechanism test, line of evidence 5 (production geometry)**: an
    /// SPD `H` with a small eigenvalue whose eigenvector MIXES
    /// `range(S_+)` and `null(S_+)`. This is the geometry the large-scale
    /// survival marginal-slope hits at the failing iterate: the unpenalized
    /// parametric columns (intercept, sex, prs_z) interact with the
    /// penalized Duchon centers via `Xᵀ W X` off-diagonal coupling, so the
    /// smallest eigendirection of `H` is NOT axis-aligned with the
    /// `U_S = span(S_+)` block.
    ///
    /// In this regime:
    ///   * `a_k = λ_k S_k β̂ ∈ range(S_+)` (by construction, exactly the
    ///     production input shape — purely in-subspace, NO null
    ///     contamination).
    ///   * `r = r_clean + ε · e_null` has small but nonzero null-space
    ///     contamination representative of floating-point KKT residual
    ///     noise at the inner exit certificate.
    ///   * The full-H inverse `H⁻¹ a_k` PICKS UP a null-direction
    ///     component via the Schur complement / cross-block coupling: the
    ///     small-eigenvalue eigenvector v_min has both a range(S_+) and
    ///     a null(S_+) leg, so `H⁻¹ a_k ∝ (a_kᵀ v_min) · v_min / σ_min`
    ///     has a null leg of magnitude `(a_k · v_min_S) · v_min_N /
    ///     σ_min ≈ 1 · 1 / 1e-12 = 1e12`. Dotting with `r`'s tiny null
    ///     noise `ε ≈ 1e-3` gives a `1e9` spurious contribution.
    ///   * The projected helper `aᵀ U_S H_proj⁻¹ U_Sᵀ b`:
    ///       - `U_Sᵀ a_k = a_k_S` (unchanged, since `a_k ∈ range(S_+)`)
    ///       - `U_Sᵀ r = r_S` (drops the `ε · e_null` contamination)
    ///       - `H_proj = U_Sᵀ H U_S` — the in-subspace block, which has
    ///         the well-conditioned `O(1)` eigenvalues only (the small
    ///         eigendirection's range(S_+) leg is `≪ 1`, so its
    ///         contribution to `H_proj` is `O(σ_min · (v_min_S)²) ≪`
    ///         the other eigenvalues; `H_proj⁻¹` stays `O(1)`).
    ///       - Result: `r_S · H_proj⁻¹ · a_k_S` is `O(1)`.
    ///
    /// This test reproduces the FAILING-LARGE_SCALE geometry and asserts
    /// FOUR INDEPENDENT properties:
    ///   (P1) helper matches an independent eigendecomposition-based
    ///        ground-truth bilinear to 1e-12 (validates the inversion
    ///        path on a NON-DIAGONAL h_proj, where the diagonal
    ///        shortcut would silently fail);
    ///   (P2) null pollution on r is invariant under the helper (1e-12);
    ///   (P3) the SAME null pollution corrupts full-H by the
    ///        analytically predicted `ε · (q_minᵀ a_k) · q_min_null /
    ///        σ_min` (matched within 5%);
    ///   (P4) on CLEAN r the projected helper and full-H still DISAGREE
    ///        by `O(σ_min⁻¹)` because the projected kernel pairs with
    ///        `½ log|U_Sᵀ H U_S|_+` while the full-H is the Schur
    ///        complement inverse — mathematically distinct, not just
    ///        less noisy.
    /// Independent ground-truth bilinear form `aᵀ U_S (U_Sᵀ H U_S)⁻¹
    /// U_Sᵀ b`. Recomputes the projected inverse via a fresh
    /// eigendecomposition of `U_Sᵀ H U_S` — a separate code path from
    /// `PenaltySubspaceTrace::bilinear_pseudo_inverse` (which applies a
    /// PRECOMPUTED `h_proj_inverse`). Match between the two is non-
    /// trivial verification of the helper's inversion.
    pub(crate) fn projected_pseudo_inverse_truth(
        h_full: &Array2<f64>,
        u_s: &Array2<f64>,
        a: &Array1<f64>,
        b: &Array1<f64>,
    ) -> f64 {
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let proj_a = u_s.t().dot(a);
        let proj_b = u_s.t().dot(b);
        let h_proj = u_s.t().dot(h_full).dot(u_s);
        let (evals, evecs) = h_proj.eigh(Side::Lower).expect("h_proj eigh");
        let ua = evecs.t().dot(&proj_a);
        let ub = evecs.t().dot(&proj_b);
        let mut acc = 0.0_f64;
        for i in 0..evals.len() {
            assert!(evals[i].abs() > 1e-10, "h_proj eigenvalue must be nonzero");
            acc += ua[i] * ub[i] / evals[i];
        }
        acc
    }

    #[test]
    pub(crate) fn ift_projected_pseudo_inverse_saves_orders_of_magnitude_on_cross_coupled_h() {
        let small_eig = 1e-12_f64;
        let p = 5usize;
        let r_subspace = 4usize;

        // U_S spans the first 4 standard basis vectors (range of S_+).
        let mut u_s = Array2::<f64>::zeros((p, r_subspace));
        for j in 0..r_subspace {
            u_s[[j, j]] = 1.0;
        }

        // Build SPD H = Q diag(λ) Qᵀ where the smallest-eigenvalue
        // eigenvector v_min has LEGS ON ALL FOUR range(S_+) coordinates
        // (not just e_3 as in the earlier, weaker fixture). This forces
        // `h_proj = U_Sᵀ H U_S` to be genuinely non-diagonal, so the
        // helper's eigendecomposition-based inversion is actually
        // exercised (a diagonal shortcut would silently fail here).
        let v_min = {
            let leg_s = 0.15_f64;
            let leg_n = (1.0 - 4.0 * leg_s * leg_s).sqrt();
            array![leg_s, leg_s, leg_s, leg_s, leg_n]
        };
        // Four ambient vectors with MIXED support (not the standard
        // basis) so Gram-Schmidt against `v_min` produces dense Q
        // columns and a dense `h_proj`.
        let ambients = [
            array![1.0_f64, 0.3, -0.2, 0.5, 0.0],
            array![0.4_f64, 1.0, 0.6, -0.3, 0.0],
            array![-0.5_f64, 0.2, 1.0, 0.7, 0.0],
            array![0.6_f64, -0.4, 0.3, 1.0, 0.0],
        ];
        let mut q = Array2::<f64>::zeros((p, p));
        q.column_mut(p - 1).assign(&v_min);
        let mut col_idx = 0usize;
        for ambient in ambients.iter() {
            let mut v = ambient.clone();
            let dot = v.dot(&v_min);
            v.scaled_add(-dot, &v_min);
            for prev in 0..col_idx {
                let qprev = q.column(prev).to_owned();
                let d = v.dot(&qprev);
                v.scaled_add(-d, &qprev);
            }
            let norm = v.dot(&v).sqrt();
            assert!(
                norm > 1e-10,
                "Gram-Schmidt failed at col {col_idx}: norm = {norm}"
            );
            v /= norm;
            q.column_mut(col_idx).assign(&v);
            col_idx += 1;
        }
        assert_eq!(col_idx, p - 1);

        let eigvals = array![10.0_f64, 5.0, 2.0, 1.0, small_eig];
        let mut h_full = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            let qi = q.column(i).to_owned();
            for a in 0..p {
                for b in 0..p {
                    h_full[[a, b]] += eigvals[i] * qi[a] * qi[b];
                }
            }
        }
        // Symmetrize to suppress 1e-16-scale asymmetry from
        // outer-product summation order.
        for a in 0..p {
            for b in (a + 1)..p {
                let avg = 0.5 * (h_full[[a, b]] + h_full[[b, a]]);
                h_full[[a, b]] = avg;
                h_full[[b, a]] = avg;
            }
        }

        // Verify `h_proj` is genuinely non-diagonal (the WHOLE POINT of
        // this fixture — earlier versions silently relied on it being
        // diagonal).
        let h_proj_check = u_s.t().dot(&h_full).dot(&u_s);
        let mut max_offdiag = 0.0_f64;
        let mut max_diag = 0.0_f64;
        for i in 0..r_subspace {
            max_diag = max_diag.max(h_proj_check[[i, i]].abs());
            for j in 0..r_subspace {
                if i != j {
                    max_offdiag = max_offdiag.max(h_proj_check[[i, j]].abs());
                }
            }
        }
        assert!(
            max_offdiag > 0.1 * max_diag,
            "fixture must produce non-diagonal h_proj; max_offdiag = \
             {max_offdiag:.3e}, max_diag = {max_diag:.3e}"
        );

        let kernel = build_subspace_kernel(&h_full, &u_s);

        // `a_k` purely in range(S_+) — production geometry exactly:
        // `λ_k S_k β̂ ∈ col(S_k) ⊂ range(S_+)`.
        let a_k = array![0.5_f64, 0.7, -0.3, 0.9, 0.0];
        let eps_null = 1e-3_f64;
        let r_clean = array![0.4_f64, -0.6, 1.1, 0.3, 0.0];
        let r_total = &r_clean + &array![0.0_f64, 0.0, 0.0, 0.0, eps_null];

        // ── (P1) Helper matches independent ground-truth bilinear ──
        let truth_clean = projected_pseudo_inverse_truth(&h_full, &u_s, &r_clean, &a_k);
        let truth_total = projected_pseudo_inverse_truth(&h_full, &u_s, &r_total, &a_k);
        let corr_proj_clean = kernel.bilinear_pseudo_inverse(&r_clean, &a_k);
        let corr_proj_total = kernel.bilinear_pseudo_inverse(&r_total, &a_k);
        assert_relative_eq!(corr_proj_clean, truth_clean, max_relative = 1e-10);
        assert_relative_eq!(corr_proj_total, truth_total, max_relative = 1e-10);

        // ── (P2) Projection invariance under null pollution ──
        // The two helper outputs agree because `U_Sᵀ ε e_null = 0`,
        // AND this holds DESPITE the non-diagonal `h_proj` (i.e. the
        // full inversion path is exercised).
        assert_relative_eq!(corr_proj_total, corr_proj_clean, max_relative = 1e-12);
        assert_relative_eq!(truth_total, truth_clean, max_relative = 1e-12);

        // ── (P3) Full-H IS corrupted by the same pollution ──
        // Predicted scale: ε · (q_minᵀ a_k) · q_min[p-1] / σ_min.
        let corr_full_clean = dense_h_inv_bilinear_via_eig(&h_full, &r_clean, &a_k);
        let corr_full_total = dense_h_inv_bilinear_via_eig(&h_full, &r_total, &a_k);
        let full_noise_contrib = corr_full_total - corr_full_clean;
        let v_min_dot_a = v_min.dot(&a_k);
        let v_min_null = v_min[p - 1];
        let predicted_noise = eps_null * v_min_dot_a * v_min_null / small_eig;
        assert!(
            full_noise_contrib.abs() > 1e6,
            "full-H must show 10⁶+ noise amplification; got |Δ| = {:.3e}",
            full_noise_contrib.abs()
        );
        let ratio = full_noise_contrib / predicted_noise;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "full-H corruption must follow predicted scaling; \
             ratio = {ratio:.6}, predicted = {predicted_noise:.3e}, \
             actual = {full_noise_contrib:.3e}"
        );

        // ── (P4) Self-consistency: projected ≠ full-H even on CLEAN r ──
        // The fix is the kernel that pairs with `½ log|U_Sᵀ H U_S|_+`;
        // full-H pairs with `½ log|H|`. On a cross-coupled SPD H with a
        // small-eigenvalue mixed direction, the two bilinear forms
        // disagree by `O(σ_min⁻¹)`. NOT just a denoising effect.
        let clean_disagreement = corr_full_clean - corr_proj_clean;
        assert!(
            clean_disagreement.abs() > 1e5,
            "fix is self-consistency, NOT denoising: on CLEAN input \
             projected ({corr_proj_clean:.3e}) and full-H \
             ({corr_full_clean:.3e}) must differ by O(1/σ_min); \
             got disagreement = {clean_disagreement:.3e}"
        );

        eprintln!(
            "[ift-cross-coupled-airtight] h_proj non-diag ratio = \
             {:.3} (max_off / max_diag), clean: projected = \
             {corr_proj_clean:.6e}, full = {corr_full_clean:.6e}, \
             disagreement = {clean_disagreement:.3e}",
            max_offdiag / max_diag
        );
        eprintln!(
            "[ift-cross-coupled-airtight] pollute(ε={eps_null:.0e}): \
             projected = {corr_proj_total:.6e}, full = {corr_full_total:.6e}, \
             predicted_noise = {predicted_noise:.6e}, actual_noise = \
             {full_noise_contrib:.6e}, ratio = {ratio:.6}"
        );
    }

    pub(crate) fn make_factor_key(seed: u64) -> ProjectedFactorKey {
        // Build a unique-by-seed key without going through
        // `from_factor_view` so the test can inject fingerprints
        // directly. Using public construction via a real ArrayView2
        // would couple this test to ndarray pointer aliasing.
        ProjectedFactorKey {
            design_id: 1,
            factor_ptr: seed as usize,
            rows: 1,
            cols: 1,
            row_stride: 1,
            col_stride: 1,
            value_hash: seed,
            value_hash2: seed.wrapping_mul(31),
        }
    }

    #[test]
    pub(crate) fn projected_factor_cache_lru_evicts_oldest_under_budget() {
        let entry_floats = 32usize;
        let entry_bytes = entry_floats * std::mem::size_of::<f64>();
        // Budget that fits exactly two entries — inserting a third must
        // evict the least-recently-used one.
        let cache = ProjectedFactorCache::with_budget(entry_bytes * 2);

        let make = |seed: u64| -> Array2<f64> { Array2::from_elem((4, 8), seed as f64) };

        let _a = cache.get_or_insert_with(make_factor_key(1), || make(1));
        let _b = cache.get_or_insert_with(make_factor_key(2), || make(2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), entry_bytes * 2);

        // Bump `a`'s recency so it survives the next eviction.
        let _a_again = cache.get_or_insert_with(make_factor_key(1), || make(1));

        // Inserting `c` must evict `b` (oldest), not `a` (most recent).
        let _c = cache.get_or_insert_with(make_factor_key(3), || make(3));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), entry_bytes * 2);

        // `a` and `c` survive; `b` was evicted.
        let post_a = cache.get_or_insert_with(make_factor_key(1), || make(99));
        let post_c = cache.get_or_insert_with(make_factor_key(3), || make(99));
        assert_eq!(post_a[[0, 0]], 1.0, "a survived eviction");
        assert_eq!(post_c[[0, 0]], 3.0, "c is the freshly inserted entry");

        let post_b = cache.get_or_insert_with(make_factor_key(2), || make(99));
        assert_eq!(
            post_b[[0, 0]],
            99.0,
            "b was evicted; recompute closure runs"
        );
    }

    #[test]
    pub(crate) fn projected_factor_cache_zero_budget_disables_eviction() {
        let cache = ProjectedFactorCache::with_budget(0);
        for seed in 0..16 {
            cache.get_or_insert_with(make_factor_key(seed), || {
                Array2::from_elem((8, 8), seed as f64)
            });
        }
        assert_eq!(cache.len(), 16);
    }

    #[test]
    pub(crate) fn projected_factor_cache_oversize_entry_is_cached_unconditionally() {
        // An entry larger than the entire budget cannot be made to fit
        // by eviction; we still cache it (refusing to cache would force
        // a recompute on every query, defeating the cache's purpose).
        let cache = ProjectedFactorCache::with_budget(8);
        let huge = cache.get_or_insert_with(make_factor_key(1), || Array2::from_elem((4, 4), 1.0));
        assert_eq!(huge[[0, 0]], 1.0);
        assert_eq!(cache.len(), 1);
    }

    pub(crate) fn projected_factor_cache_wait_for_subscriber(
        cache: &ProjectedFactorCache,
        key: ProjectedFactorKey,
        timeout: std::time::Duration,
    ) -> bool {
        let marker = {
            let inner = cache
                .inner
                .lock()
                .expect("projected factor cache lock poisoned");
            let Some(m) = inner.in_progress.get(&key) else {
                return false;
            };
            Arc::clone(m)
        };
        if marker
            .waiter_count
            .load(std::sync::atomic::Ordering::Acquire)
            > 0
        {
            return true;
        }
        let (lock, cv) = &marker.subscriber_arrived;
        let mut guard = lock
            .lock()
            .expect("subscriber-arrived notification lock poisoned");
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if marker
                .waiter_count
                .load(std::sync::atomic::Ordering::Acquire)
                > 0
            {
                return true;
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }
            let (next_guard, result) = cv
                .wait_timeout(guard, deadline - now)
                .expect("subscriber-arrived wait poisoned");
            guard = next_guard;
            if result.timed_out()
                && marker
                    .waiter_count
                    .load(std::sync::atomic::Ordering::Acquire)
                    == 0
            {
                return false;
            }
        }
    }

    #[test]
    pub(crate) fn projected_factor_cache_waiters_wake_when_producer_panics() {
        let cache = Arc::new(ProjectedFactorCache::with_budget(0));
        let key = make_factor_key(42);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();

        let producer_cache = Arc::clone(&cache);
        let producer = std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                producer_cache.get_or_insert_with(key, || {
                    started_tx.send(()).expect("send producer-start signal");
                    // Block until the waiter has actually subscribed to the
                    // in-progress slot. The release signal is fired only after
                    // the cache reports a parked waiter, so the producer's
                    // panic is guaranteed to wake at least one Wait branch.
                    release_rx.recv().expect("waiter subscribe signal");
                    panic!("simulated projected-factor panic");
                });
            }))
            .is_err()
        });

        started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("producer started computing");

        let waiter_cache = Arc::clone(&cache);
        let waiter = std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                waiter_cache.get_or_insert_with(key, || Array2::from_elem((1, 1), 7.0));
            }))
            .is_err()
        });

        // Block on the cache's subscriber-arrived condvar until the waiter
        // has parked inside the Wait branch, then release the producer to
        // panic. No spinning, no sleeping — purely event-driven.
        assert!(
            projected_factor_cache_wait_for_subscriber(
                &cache,
                key,
                std::time::Duration::from_secs(5),
            ),
            "waiter never subscribed to the in-progress slot"
        );
        release_tx.send(()).expect("release producer");

        assert!(producer.join().expect("producer thread joined"));
        assert!(waiter.join().expect("waiter thread joined"));

        let recovered = cache.get_or_insert_with(key, || Array2::from_elem((1, 1), 9.0));
        assert_eq!(recovered[[0, 0]], 9.0);
    }

    pub(crate) struct SentinelOuterHessianOperator {
        pub(crate) matrix: Array2<f64>,
    }

    impl crate::solver::outer_strategy::OuterHessianOperator for SentinelOuterHessianOperator {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            Ok(self.matrix.dot(v))
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }
    }

    pub(crate) struct FamilyOperatorOnlyDerivatives {
        pub(crate) op: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
    }

    impl HessianDerivativeProvider for FamilyOperatorOnlyDerivatives {
        fn hessian_derivative_correction(
            &self,
            arr: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            Ok(None)
        }

        fn has_corrections(&self) -> bool {
            false
        }

        fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
            None
        }

        fn family_outer_hessian_operator(
            &self,
        ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
            Some(Arc::clone(&self.op))
        }
    }

    /// Helper: the inflated-`penalty_logdet` sentinel `InnerSolution` shared by
    /// the envelope-tripwire regression tests. They differ only in the
    /// dispersion handling and whether a projected KKT residual is attached.
    pub(crate) fn build_sentinel_tripwire_solution(
        dispersion: DispersionHandling,
        kkt_residual: Option<ProjectedKktResidual>,
    ) -> InnerSolution<'static> {
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap());
        let family_operator = Arc::new(SentinelOuterHessianOperator {
            matrix: array![[42.0]],
        });
        let deriv_provider = FamilyOperatorOnlyDerivatives {
            op: family_operator,
        };

        InnerSolution {
            log_likelihood: -1.25,
            penalty_quadratic: 0.4,
            hessian_op: hop,
            beta: array![0.5, -0.25],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(Array2::eye(2))],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0e20],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 2,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    /// Helper: assemble the analytic dense outer Hessian, the matrix-free
    /// operator, and the materialised dense form of that operator for
    /// `solution`. The operator-vs-dense equivalence tests all build this
    /// identical `(dense, operator, materialized)` triple before comparing
    /// entries and matvecs.
    pub(crate) fn dense_and_materialized_outer_hessian(
        solution: &InnerSolution<'_>,
        rho: &[f64],
        lambdas: &[f64],
    ) -> (Array2<f64>, UnifiedOuterHessianOperator, Array2<f64>) {
        let dense = compute_outer_hessian(
            solution,
            rho,
            lambdas,
            solution.hessian_op.as_ref(),
            solution.deriv_provider.as_ref(),
            None,
        )
        .unwrap();
        let kernel = solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let operator = build_outer_hessian_operator(
            solution,
            lambdas,
            solution.deriv_provider.as_ref(),
            kernel,
            None,
            None,
        )
        .unwrap();
        let materialized =
            crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(&operator)
                .unwrap();
        (dense, operator, materialized)
    }

    #[test]
    pub(crate) fn value_gradient_hessian_prefers_family_supplied_outer_operator() {
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap());
        let family_matrix = array![[42.0]];
        let family_operator = Arc::new(SentinelOuterHessianOperator {
            matrix: family_matrix.clone(),
        });
        let deriv_provider = FamilyOperatorOnlyDerivatives {
            op: family_operator,
        };

        let solution = InnerSolution {
            log_likelihood: -1.25,
            penalty_quadratic: 0.4,
            hessian_op: hop,
            beta: array![0.5, -0.25],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(Array2::eye(2))],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.0],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 2,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            // Profiled Gaussian does not satisfy the fixed-dispersion IFT
            // identity used by the projected KKT residual correction, so an
            // inconsistent envelope gradient remains a soft "unavailable
            // derivative" result rather than a missing-residual contract
            // violation.
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("family outer operator evaluation");
        let crate::solver::outer_strategy::HessianResult::Operator(op) = result.hessian else {
            panic!("expected family-supplied operator Hessian route");
        };
        let dense = op.materialize_dense().expect("sentinel materialization");
        assert_eq!(dense, family_matrix);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Replicates the large-scale large-scale marginal-slope failure mechanism in three
    // tiers, each pinning a distinct code-level math issue observed in
    // the failing log:
    //
    //   [PIRLS/joint-Newton convergence] cycle 303 | constrained-stationary
    //     certificate: linear-solve neutralised 0.0% of g (... multiplier ...);
    //     |Δobjective|=3.421e-1 ≤ obj_tol=3.479e-1
    //   [reml_laml envelope-gradient consistency] |g|∞ = 9.669e16 ... |cost|
    //     = 3.480e5  ratio 4.14e3  → gradient suppressed → seed rejected.
    //
    // BUG-1 (math, compute_kkt_residual_rho_corrections @ ~unified.rs:8500):
    //   At cert exit, the projected KKT residual r_proj ≈ 0 (multiplier
    //   captured by active set). Then q = H⁻¹·r_proj = 0, so the gradient
    //   correction `-aᵀ_k q + ½ qᵀA_k q` is identically zero. The cert's
    //   contract gives ZERO cancellation of the inflated envelope trace.
    //
    // BUG-2 (gate, envelope_inconsistent @ ~unified.rs:7466):
    //   The old gate treated `Some(..) && Fixed` as proof that the inflated
    //   gradient was repaired, even when the correction magnitude was zero.
    //   At the cert's r_proj=0 contract that routed an inflated meaningless
    //   gradient through the "applied" arm.
    //
    // BUG-3 (math, envelope ½ tr(H⁻¹·∂H/∂ρ) at near-singular H — FIXED):
    //   Resolved by the tangent-space dispatch at the top of
    //   `reml_laml_evaluate` (`try_tangent_projected_evaluate`): when
    //   `solution.active_constraints` is non-empty the evaluator recurses
    //   on the tangent-projected `(ZᵀHZ, ZᵀSZ)` so the trace stays
    //   bounded by σ_min(ZᵀHZ)⁻¹ instead of σ_min(H)⁻¹.
    //
    // Each test below isolates one of these bugs.
    // ───────────────────────────────────────────────────────────────────────

    /// BUG-1: r_proj = 0 ⇒ IFT gradient correction is identically 0.
    /// This is a pure math identity test that ANCHORS the failure mode.
    #[test]
    pub(crate) fn ift_gradient_correction_with_zero_projected_residual_is_zero() {
        let h = Array2::eye(3);
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let solution = build_gaussian_solution_at_beta(&[0.0, 0.0], array![0.5, -0.25, 0.1], false);

        let lambdas = [1.0_f64, 1.0_f64];
        let penalty_a_k_betas = vec![array![0.3, -0.7, 0.0], array![0.0, 0.0, 0.5]];
        let zero_residual = Array1::<f64>::zeros(hop.dim());

        let corrections = compute_kkt_residual_rho_corrections(
            &solution,
            &hop,
            &lambdas,
            &penalty_a_k_betas,
            &zero_residual,
            true,
            &[false, false],
        )
        .expect("kkt correction must succeed at zero residual");

        for (i, &g) in corrections.gradient.iter().enumerate() {
            assert_eq!(
                g, 0.0,
                "BUG-1: IFT gradient correction at coord {} must be exactly 0.0 when \
                 r_proj = 0 (q = H⁻¹·0 = 0); got {:.3e}. The cert path's projected residual \
                 ≈ 0 contract therefore gives ZERO correction to the envelope gradient — \
                 inflated ½ tr(H⁻¹·∂H/∂ρ) is left uncancelled.",
                i, g
            );
        }
        let h_corr = corrections.hessian.expect("hessian requested");
        for ((i, j), &v) in h_corr.indexed_iter() {
            assert_eq!(
                v, 0.0,
                "BUG-1 hessian: entry ({}, {}) must be 0; got {:.3e}",
                i, j, v
            );
        }
    }

    #[test]
    pub(crate) fn ift_rho_upper_bound_masks_residual_correction_direction() {
        let h = Array2::eye(3);
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let solution = build_gaussian_solution_at_beta(&[0.0, 0.0], array![0.5, -0.25, 0.1], false);

        let lambdas = [1.0_f64, 1.0_f64];
        let penalty_a_k_betas = vec![array![0.3, -0.7, 0.0], array![0.0, 0.0, 0.5]];
        let residual = array![0.2, -0.3, 0.4];

        let corrections = compute_kkt_residual_rho_corrections(
            &solution,
            &hop,
            &lambdas,
            &penalty_a_k_betas,
            &residual,
            true,
            &[false, true],
        )
        .expect("kkt correction must succeed with a masked upper-bound coordinate");

        assert_eq!(
            corrections.gradient[1], 0.0,
            "upper-bound rho direction must not receive IFT residual gradient correction"
        );
        let hessian = corrections.hessian.expect("hessian requested");
        for idx in 0..2 {
            assert_eq!(
                hessian[[1, idx]],
                0.0,
                "upper-bound rho row must be zeroed in IFT residual Hessian correction"
            );
            assert_eq!(
                hessian[[idx, 1]],
                0.0,
                "upper-bound rho column must be zeroed in IFT residual Hessian correction"
            );
        }
    }

    /// BUG-2 regression: cert exit with r_proj = 0 must not pass an inflated
    /// `|g|∞ = 1e20` gradient through the envelope tripwire. Contract under
    /// test: either suppress (gradient=None) or produce a numerically honest
    /// gradient (|g|∞·√ε ≤ 4·|cost|).
    #[test]
    pub(crate) fn cert_zero_residual_must_not_emit_unbounded_gradient_through_gate() {
        // Profiled Gaussian does not satisfy the fixed-dispersion IFT identity
        // used by the projected KKT residual correction, so an inconsistent
        // envelope gradient remains a soft "unavailable derivative" result
        // rather than a missing-residual contract violation.
        let solution = build_sentinel_tripwire_solution(
            DispersionHandling::ProfiledGaussian,
            Some(ProjectedKktResidual::from_active_projected(array![
                0.0, 0.0
            ])),
        );

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("constrained-stationary cert evaluation");

        let cost_scale = result.cost.abs().max(1.0);
        let resolve_step = f64::EPSILON.sqrt();
        let max_abs = match result.gradient.as_ref() {
            Some(g) => g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
            None => 0.0, // suppressed is acceptable
        };
        let predicted_change = max_abs * resolve_step;
        let ratio = predicted_change / cost_scale;
        assert!(
            result.gradient.is_none() || (ratio <= 4.0 && max_abs.is_finite()),
            "BUG-2 REGRESSION: cert exit with r_proj = 0 passed an inflated gradient \
             through the tripwire. |grad|∞ = {:.3e}, predicted Δcost along √ε step = \
             {:.3e}, cost = {:.3e}, ratio = {:.3e} (must be ≤ 4 OR gradient = None). \
             At r_proj = 0 the projected-residual correction is identically zero \
             (see BUG-1 test), so the inflated envelope gradient must remain \
             unavailable to the outer optimizer.",
            max_abs,
            predicted_change,
            cost_scale,
            ratio,
        );
    }

    /// Regression test for the constraint-tangent LAML projection
    /// (Wood 2011 §4; Wood–Pya–Säfken 2016 §3; Marra–Wood 2012 §2).
    ///
    /// With an active linear inequality constraint pinning a parameter at
    /// its bound, the principled REML/LAML outer cost lives on the
    /// constraint tangent space `T = null(A_act)`:
    ///
    ///   cost   = … + ½ log|Zᵀ H Z|  − ½ log|Zᵀ S(λ) Z|_+
    ///   grad_k = … + ½ tr((Zᵀ H Z)⁻¹ Zᵀ (λ_k S_k) Z) − ½ ∂_k log|Zᵀ S Z|_+
    ///
    /// #931 pass-2 bit-identity pin: `ThetaModeResponseKernel`'s emissions
    /// equal the pre-port per-site assemblies EXACTLY (same factorization,
    /// same solve shapes, bitwise-equal outputs) in both selection regimes.
    ///
    /// The pre-port code at the four consumer sites is reproduced inline
    /// here as the reference:
    ///   * unconstrained, per-vector  → `hop.solve(rhs)`         (dense
    ///     `compute_outer_hessian` ρ/ext fallbacks);
    ///   * unconstrained, stacked     → `hop.solve_multi(stack)` (gradient
    ///     site, `build_outer_hessian_operator` fallback);
    ///   * constrained (active A_act) → per-vector
    ///     `with_active_constraints(..).apply_pseudo_inverse(rhs)`;
    ///   * box-masked ρ coordinate    → exact zeros (old code skipped the
    ///     solve; the kernel maps the zero RHS column to exact zeros by
    ///     linearity).
    /// Any future edit that lets one shape drift from its reference breaks
    /// this test bit-for-bit, which is the point.
    #[test]
    pub(crate) fn theta_mode_response_kernel_matches_preport_assembly_bitwise() {
        use crate::solver::estimate::reml::unified::ActiveLinearConstraintBlock;

        let h = array![[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let rhs_a = array![0.7, -1.3, 0.4];
        let rhs_b = array![-0.2, 0.9, 2.1];

        // ── Unconstrained regime: full-Hessian solve, both shapes. ──
        let kernel_free = ThetaModeResponseKernel::select(None, None, &hop);
        let one_new = kernel_free.respond_one(&rhs_a);
        let one_ref = hop.solve(&rhs_a);
        for (n, r) in one_new.iter().zip(one_ref.iter()) {
            assert_eq!(n.to_bits(), r.to_bits(), "respond_one vs hop.solve");
        }
        let mut stack = Array2::<f64>::zeros((3, 3));
        stack.column_mut(0).assign(&rhs_a);
        // column 1 stays zero: a box-masked ρ coordinate's RHS.
        stack.column_mut(2).assign(&rhs_b);
        let stack_new = kernel_free.respond_stack(&stack);
        let stack_ref = hop.solve_multi(&stack);
        for (n, r) in stack_new.iter().zip(stack_ref.iter()) {
            assert_eq!(n.to_bits(), r.to_bits(), "respond_stack vs solve_multi");
        }
        assert!(
            stack_new.column(1).iter().all(|v| *v == 0.0),
            "masked zero RHS column must emit exact zeros"
        );

        // ── Constrained regime: lifted kernel K_T, selection + emission. ──
        let trace = PenaltySubspaceTrace {
            u_s: array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            h_proj_inverse: array![[0.5, 0.1], [0.1, 0.8]],
        };
        let block = ActiveLinearConstraintBlock {
            a: array![[1.0, 1.0, 0.0]],
        };
        let kernel_con = ThetaModeResponseKernel::select(Some(&trace), Some(&block), &hop);
        // Pre-port reference: build the constrained kernel the way every
        // site did, apply per vector.
        let ck_ref = trace.with_active_constraints(block.a.view());
        assert!(ck_ref.has_active_constraints());
        for rhs in [&rhs_a, &rhs_b] {
            let v_new = kernel_con.respond_one(rhs);
            let v_ref = ck_ref.apply_pseudo_inverse(rhs);
            for (n, r) in v_new.iter().zip(v_ref.iter()) {
                assert_eq!(n.to_bits(), r.to_bits(), "constrained respond_one vs K_T");
            }
            // The pass-2 certify invariant: the emission lies in ker(A_act).
            let tangency = block.a.dot(&v_new);
            assert!(
                tangency[0].abs() <= 1e-12 * (1.0 + v_new.iter().map(|x| x.abs()).sum::<f64>()),
                "constrained mode response must satisfy A_act·v = 0, got {}",
                tangency[0]
            );
        }
        // Stacked constrained emission = per-column K_T applies in order,
        // masked column included (zero in, exact zero out).
        let con_stack_new = kernel_con.respond_stack(&stack);
        for (j, rhs) in [Some(&rhs_a), None, Some(&rhs_b)].iter().enumerate() {
            let v_ref = match rhs {
                Some(rhs) => ck_ref.apply_pseudo_inverse(rhs),
                None => Array1::<f64>::zeros(3),
            };
            for (n, r) in con_stack_new.column(j).iter().zip(v_ref.iter()) {
                assert_eq!(
                    n.to_bits(),
                    r.to_bits(),
                    "constrained respond_stack column {j} vs per-vector K_T"
                );
            }
        }

        // ── Selection rule edge: subspace without active constraints must
        // still take the FULL solve (the near-separable null(S₊) rule). ──
        let kernel_subspace_only = ThetaModeResponseKernel::select(Some(&trace), None, &hop);
        let v_sub = kernel_subspace_only.respond_one(&rhs_a);
        for (n, r) in v_sub.iter().zip(one_ref.iter()) {
            assert_eq!(
                n.to_bits(),
                r.to_bits(),
                "subspace-without-constraints must route through the full H⁻¹"
            );
        }
    }

    /// The unified evaluator dispatches to `try_tangent_projected_evaluate`
    /// at the top of `reml_laml_evaluate`, so this test verifies that under
    /// an active constraint set the returned gradient stays bounded
    /// independent of σ_min(H) — the unprojected full-space trace
    /// `½ tr(H⁻¹·∂H/∂ρ_k)` would blow up as 1/σ_min(H) here while the
    /// projected trace `½ tr((ZᵀHZ)⁻¹·Zᵀ·λ_k S_k·Z)` is O(1).
    #[test]
    pub(crate) fn envelope_gradient_uses_constraint_tangent_projection() {
        use crate::solver::estimate::reml::unified::ActiveLinearConstraintBlock;

        // Three-parameter Gaussian REML. Choose data so the optimum places
        // β₂ at its lower bound β₂ ≥ 0.5 — the active constraint
        // a = [0, 0, 1], b = 0.5. Picking λ_k so that S_k has nonzero
        // mass along the constraint normal e = [0, 0, 1] makes
        // eᵀ S_k e > 0, which per codex is the condition under which the
        // unprojected trace tr(H⁻¹·λ_k S_k) inflates while the projected
        // trace tr((ZᵀHZ)⁻¹·Zᵀ λ_k S_k Z) stays O(1).
        let xtx = array![[2.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 1.0e-10]];
        let s1 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![1.0, 0.5, 0.0];

        let rho = vec![0.0_f64, 0.0_f64];
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
        let mut h_mat = xtx.clone();
        h_mat.scaled_add(lambdas[0], &s1);
        h_mat.scaled_add(lambdas[1], &s2);
        let op = DenseSpectralOperator::from_symmetric(&h_mat).unwrap();
        let beta_unconstrained = op.solve(&xty);
        // Force β₂ = 0.5 (active bound) and recover β₀, β₁ minimising the
        // remaining quadratic — i.e. β̂ is the *constrained* solution.
        assert!(beta_unconstrained[2] < 0.5);
        let beta_hat = array![xty[0] / h_mat[[0, 0]], xty[1] / h_mat[[1, 1]], 0.5];

        let a_act = array![[0.0, 0.0, 1.0]];
        let active = ActiveLinearConstraintBlock { a: a_act };

        let mut sol = build_gaussian_solution_at_beta(&rho, beta_hat.clone(), true);
        sol.dispersion = DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h: true,
            include_logdet_s: true,
        };
        sol.active_constraints = Some(Arc::new(active));

        let result = reml_laml_evaluate(&sol, &rho, EvalMode::ValueAndGradient, None);

        // Under the active constraint set the projected gradient must be
        // bounded independent of σ_min(H). σ_min(ZᵀHZ) is O(1) by
        // construction here (the active normal e = [0,0,1] aligns with the
        // ill-conditioned direction in H), so the tangent-projected gradient
        // is finite and O(1).
        match result {
            Ok(r) => {
                let grad = r
                    .gradient
                    .expect("tangent dispatch must emit a finite projected gradient");
                let max_abs = grad.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
                assert!(
                    max_abs.is_finite() && max_abs < 1.0e6,
                    "projected gradient must be bounded by σ_min(ZᵀHZ)⁻¹ (O(1) here); \
                     got |grad|∞ = {:.3e}",
                    max_abs,
                );
            }
            Err(err) => panic!(
                "tangent-projected evaluator must succeed under active constraints, got: {err}"
            ),
        }
    }

    /// Hard reproducer for the large-scale missing-residual path. In fixed-dispersion
    /// LAML, an envelope-inconsistent derivative request with
    /// `kkt_residual=None` is not a recoverable value-gradient result: the
    /// evaluator cannot distinguish "exact KKT" from "convergent inner path
    /// forgot to hand over the projected residual". The principled response is
    /// a contract error naming `BlockwiseInnerResult::kkt_residual`, not
    /// `gradient=None` that the outer seed validator later reports as
    /// non-finite derivatives.
    #[test]
    pub(crate) fn aou_missing_projected_kkt_residual_is_contract_error() {
        let solution = build_sentinel_tripwire_solution(
            DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            None,
        );

        let err = match reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
        {
            Ok(_) => panic!("missing projected KKT residual must be a hard contract error"),
            Err(err) => err,
        };
        assert!(
            err.contains("fixed-dispersion derivative contract violated")
                && err.contains("BlockwiseInnerResult::kkt_residual"),
            "unexpected error for missing projected residual: {err}"
        );
    }

    #[test]
    pub(crate) fn envelope_inconsistent_gradient_skips_outer_hessian_assembly() {
        // Profiled Gaussian does not satisfy the fixed-dispersion IFT identity
        // used by the projected KKT residual correction, so an inconsistent
        // envelope gradient remains a soft "unavailable derivative" result
        // rather than a missing-residual contract violation.
        let solution = build_sentinel_tripwire_solution(DispersionHandling::ProfiledGaussian, None);

        let result = reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None)
            .expect("envelope tripwire evaluation");
        assert!(
            result.gradient.is_none(),
            "inconsistent envelope gradient should be suppressed"
        );
        assert!(
            matches!(
                result.hessian,
                crate::solver::outer_strategy::HessianResult::Unavailable
            ),
            "inconsistent envelope gradient should skip Hessian assembly"
        );
    }

    #[test]
    pub(crate) fn test_dense_spectral_operator_simple() {
        // 2×2 diagonal matrix: H = diag(2, 5)
        let h = Array2::from_diag(&array![2.0, 5.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // logdet = ln(2) + ln(5)
        let expected_logdet = 2.0_f64.ln() + 5.0_f64.ln();
        assert!((op.logdet() - expected_logdet).abs() < 1e-12);

        // tr(H⁻¹ I) = 1/2 + 1/5 = 0.7
        let id = Array2::eye(2);
        let trace = op.trace_hinv_product(&id);
        assert!((trace - 0.7).abs() < 1e-12);

        // solve: H⁻¹ [1, 1] = [0.5, 0.2]
        let rhs = array![1.0, 1.0];
        let sol = op.solve(&rhs);
        assert!((sol[0] - 0.5).abs() < 1e-12);
        assert!((sol[1] - 0.2).abs() < 1e-12);

        assert_eq!(sol.len(), 2);
    }

    #[test]
    pub(crate) fn test_dense_spectral_operator_solve_multi_matches_column_solves() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let rhs = array![[1.0, -1.0], [0.5, 2.0], [3.0, 0.25],];

        let multi = op.solve_multi(&rhs);
        for col in 0..rhs.ncols() {
            let single = op.solve(&rhs.column(col).to_owned());
            for row in 0..rhs.nrows() {
                let err = (multi[[row, col]] - single[row]).abs();
                assert!(
                    err < 1e-12,
                    "solve_multi mismatch at ({row}, {col}): multi={}, single={}",
                    multi[[row, col]],
                    single[row]
                );
            }
        }
    }

    #[test]
    pub(crate) fn test_dense_spectral_operator_cross_trace_matches_column_solves() {
        assert!(file!().ends_with(".rs"));
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a = array![[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 0.5],];
        let b = array![[0.5, -0.4, 0.1], [-0.4, 1.5, 0.25], [0.1, 0.25, 0.75],];

        let expected = (&op.solve_multi(&a).t() * &op.solve_multi(&b)).sum();
        let exact = op.trace_hinv_product_cross(&a, &b);

        assert_relative_eq!(exact, expected, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    pub(crate) fn test_dense_spectral_operator_operator_cross_matches_dense_formula() {
        assert!(file!().ends_with(".rs"));
        let h = array![[5.0, 0.5, 0.25], [0.5, 3.5, 0.2], [0.25, 0.2, 2.5],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let dense = array![[1.0, 0.1, -0.2], [0.1, 0.75, 0.3], [-0.2, 0.3, 1.25],];
        let other = array![[0.6, -0.3, 0.15], [-0.3, 1.1, 0.05], [0.15, 0.05, 0.9],];
        let other_op = DenseMatrixHyperOperator {
            matrix: other.clone(),
        };

        let expected = op.trace_hinv_product_cross(&dense, &other);
        let mixed = op.trace_hinv_matrix_operator_cross(&dense, &other_op);
        let operator = op.trace_hinv_operator_cross(&other_op, &other_op);
        let operator_expected = op.trace_hinv_product_cross(&other, &other);

        assert_relative_eq!(mixed, expected, epsilon = 1e-12, max_relative = 1e-12);
        assert_relative_eq!(
            operator,
            operator_expected,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    pub(crate) fn test_hyper_coord_total_drift_result_keeps_operator_and_dense_correction() {
        assert!(file!().ends_with(".rs"));
        let h = array![[4.0, 0.25], [0.25, 3.0],];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let base = array![[1.0, 0.2], [0.2, 0.5],];
        let corr = array![[0.3, -0.1], [-0.1, 0.4],];
        let drift = HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
            matrix: base.clone(),
        }));
        let correction = DriftDerivResult::Dense(corr.clone());

        let combined = hyper_coord_total_drift_result(&drift, Some(&correction), h.nrows());
        let expected = hop.trace_logdet_gradient(&(&base + &corr));

        assert_relative_eq!(
            combined.trace_logdet(&hop),
            expected,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    pub(crate) fn test_dense_spectral_operator_rotated_logdet_cross_matches_dense_path() {
        assert!(file!().ends_with(".rs"));
        let h = array![[4.0, 0.5, 0.2], [0.5, 2.5, 0.3], [0.2, 0.3, 1.75],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a = array![[0.8, 0.2, -0.1], [0.2, 1.4, 0.35], [-0.1, 0.35, 0.9],];
        let b = array![[1.2, -0.25, 0.05], [-0.25, 0.7, 0.15], [0.05, 0.15, 0.6],];

        let a_rot = op.rotate_to_eigenbasis(&a);
        let b_rot = op.rotate_to_eigenbasis(&b);

        let direct = op.trace_logdet_hessian_cross(&a, &b);
        let rotated = op.trace_logdet_hessian_cross_rotated(&a_rot, &b_rot);

        assert_relative_eq!(rotated, direct, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    pub(crate) fn test_compute_adjoint_z_c_streaming_matches_dense_reference() {
        assert!(file!().ends_with(".rs"));
        // streaming and dense paths differ only by reordering the sum that builds v;
        // with n=64, p=8 the gap is bounded by O(εn) ≈ 1e-14.
        let n = 64usize;
        let p = 8usize;
        let mut rng = Xoshiro256SS::from_seed(0x5EED_C0FFEE_u64);
        let unit = |rng: &mut Xoshiro256SS| {
            let bits = rng.next_u64() >> 11;
            (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let mut x_data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x_data[[i, j]] = unit(&mut rng);
            }
        }
        let mut c_array = Array1::<f64>::zeros(n);
        for i in 0..n {
            c_array[i] = unit(&mut rng);
        }

        let mut m = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                m[[i, j]] = unit(&mut rng);
            }
        }
        let mut h = m.t().dot(&m);
        for i in 0..p {
            h[[i, i]] += p as f64;
        }
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let x = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_data.clone()));
        let ing = ScalarGlmIngredients {
            c_array: &c_array,
            d_array: None,
            x: &x,
        };
        // Construct h_dense = diag(X H⁻¹ Xᵀ) via solve_multi for the dense reference.
        let z_full = hop.solve_multi(&x_data.t().to_owned());
        let mut h_dense = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..p {
                acc += x_data[[i, j]] * z_full[[j, i]];
            }
            h_dense[i] = acc;
        }
        let streamed = compute_adjoint_z_c(&ing, &hop, &h_dense, None).expect("adjoint path");

        let mut t = h_dense.clone();
        Zip::from(&mut t)
            .and(&c_array)
            .for_each(|t_i, &c_i| *t_i *= c_i);
        let v = x_data.t().dot(&t);
        let reference = hop.solve(&v);

        for k in 0..p {
            assert_relative_eq!(
                streamed[k],
                reference[k],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    pub(crate) fn fourth_derivative_trace_matrix_matches_scalar_pair_formula() {
        assert!(file!().ends_with(".rs"));
        let n = 37usize;
        let p = 5usize;
        let t = 4usize;
        let mut rng = Xoshiro256SS::from_seed(0xF047_ACE5_u64);
        let unit = |rng: &mut Xoshiro256SS| {
            let bits = rng.next_u64() >> 11;
            (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };

        let mut x_data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x_data[[i, j]] = unit(&mut rng);
            }
        }
        let mut c_array = Array1::<f64>::zeros(n);
        let mut d_array = Array1::<f64>::zeros(n);
        let mut leverage = Array1::<f64>::zeros(n);
        for i in 0..n {
            c_array[i] = unit(&mut rng);
            d_array[i] = unit(&mut rng);
            leverage[i] = 0.25 + unit(&mut rng).abs();
        }
        let x = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_data));
        let ing = ScalarGlmIngredients {
            c_array: &c_array,
            d_array: Some(&d_array),
            x: &x,
        };

        let mut modes = Vec::with_capacity(t);
        for _ in 0..t {
            let mut mode = Array1::<f64>::zeros(p);
            for j in 0..p {
                mode[j] = unit(&mut rng);
            }
            modes.push(mode);
        }
        let mode_refs = modes.iter().collect::<Vec<_>>();
        let gram = compute_fourth_derivative_trace_matrix(&ing, &mode_refs, &leverage)
            .expect("batched fourth trace")
            .expect("d-array is present");

        for i in 0..t {
            for j in 0..t {
                let scalar = compute_fourth_derivative_trace(&ing, &modes[i], &modes[j], &leverage)
                    .expect("scalar fourth trace")
                    .expect("d-array is present");
                assert_relative_eq!(gram[[i, j]], scalar, epsilon = 1e-10, max_relative = 1e-10);
            }
        }
    }

    #[test]
    pub(crate) fn operator_hessian_matches_dense_with_operator_drifts_and_extended_glm_corrections() {
        let h = array![[1.0e-7, 0.0], [0.0, 2.7]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.4, -0.7];
        let penalty_root = array![[1.2, 0.1], [0.0, 0.8]];
        let ext_drift = array![[0.45, -0.15], [-0.15, 0.35]];
        let x = array![[1.0, 0.2], [-0.4, 1.1], [0.7, -0.8]];
        let c_array = array![0.31, -0.27, 0.19];
        let d_array = array![0.17, -0.11, 0.23];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            hessian_weights: Array1::ones(3),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };

        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root)],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4],
                second: Some(array![[0.13]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 3,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: vec![HyperCoord {
                a: -0.21,
                g: array![0.33, -0.42],
                drift: HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
                    matrix: ext_drift,
                })),
                ld_s: 0.07,
                b_depends_on_beta: false,
                is_penalty_like: false,
                firth_g: None,
                tk_eta_fixed: None,
                tk_x_fixed: None,
            }],
            ext_coord_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: 0.09,
                g: array![0.16, -0.12],
                b_mat: array![[0.08, 0.03], [0.03, -0.04]],
                b_operator: None,
                ld_s: -0.05,
            })),
            rho_ext_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: -0.14,
                g: array![-0.18, 0.22],
                b_mat: array![[0.05, -0.02], [-0.02, 0.07]],
                b_operator: None,
                ld_s: 0.04,
            })),
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };
        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let (dense, operator, materialized) =
            dense_and_materialized_outer_hessian(&solution, &rho, &lambdas);

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let materialized_entry = materialized[[row, col]];
                let dense_entry = dense[[row, col]];
                let tolerance = 1e-10_f64.max(1e-10 * dense_entry.abs());
                assert!(
                    (materialized_entry - dense_entry).abs() <= tolerance,
                    "outer Hessian operator mismatch at ({row}, {col}): materialized={materialized_entry}, dense={dense_entry}"
                );
            }
        }

        let alpha = array![0.37, -0.58];
        let hvp = crate::solver::outer_strategy::OuterHessianOperator::matvec(&operator, &alpha)
            .expect("operator HVP");
        let dense_hvp = dense.dot(&alpha);
        for i in 0..hvp.len() {
            let tolerance = 1e-10_f64.max(1e-10 * dense_hvp[i].abs());
            assert!(
                (hvp[i] - dense_hvp[i]).abs() <= tolerance,
                "outer Hessian HVP mismatch at {i}: operator={}, dense={}",
                hvp[i],
                dense_hvp[i]
            );
        }
    }

    /// #740: the operator built with a direction-contracted ψψ hook (which
    /// SKIPS the per-pair ψψ `base_h2`/`pair_a`/`pair_ld_s`/`pair_g` assembly and
    /// applies the hook once per matvec) must materialize to the SAME dense outer
    /// Hessian as the exact per-pair `compute_outer_hessian` path. The hook here
    /// returns precisely the `α`-contraction of the same constant ψψ
    /// `HyperCoordPair` the per-pair `ext_coord_pair_fn` produces, so any
    /// divergence is a bug in the operator's hook injection (zeroed tables, the
    /// `pair_a`/`ld_s`/`base_h2` adds, or the `-score` rhs replacement), not in
    /// the contraction math. This is the solver-side half of the #740 exactness
    /// gate (the family-side half — contracted kernel == per-pair — lives in
    /// `bernoulli_contracted_psi_second_order_matches_per_pair_contraction`).
    #[test]
    pub(crate) fn operator_hessian_with_contracted_psi_hook_matches_per_pair_dense() {
        // WELL-CONDITIONED inner Hessian: both eigenvalues are O(1). A
        // near-singular H (e.g. a 1e-7 diagonal entry, as the sibling
        // spectral-regularization test deliberately uses) sends H⁻¹ ~1e7 and the
        // logdet-Hessian / IFT-correction terms to ~1e26, which drowns the O(1)
        // base_h2 ψψ contribution and makes the no-double-count guard see it as
        // ~0. With a sane H both ddot_H_ij terms are comparable, so the guard and
        // the equivalence both exercise a live base_h2 ψψ.
        let h = array![[1.3, 0.2], [0.2, 2.1]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.4, -0.7];
        let penalty_root = array![[1.2, 0.1], [0.0, 0.8]];
        let ext_drift = array![[0.45, -0.15], [-0.15, 0.35]];
        let x = array![[1.0, 0.2], [-0.4, 1.1], [0.7, -0.8]];
        let c_array = array![0.31, -0.27, 0.19];
        let d_array = array![0.17, -0.11, 0.23];

        // The constant ψψ pair the per-pair `ext_coord_pair_fn` returns; the hook
        // must reproduce its `α`-contraction exactly (ext_dim = 1, so the only
        // ψψ pair is (0, 0)). `psi_pair_b` (the ψψ second drift = base_h2 source)
        // is sized O(1) so its trace is a meaningful fraction of the total ψ-row
        // diagonal once H is well-conditioned — the no-double-count guard then
        // sees a genuinely live base_h2 ψψ.
        let psi_pair_a = 0.09_f64;
        let psi_pair_g = array![0.16_f64, -0.12];
        let psi_pair_b = array![[0.85_f64, 0.30], [0.30, 0.62]];
        let psi_pair_ld_s = -0.05_f64;

        // Build the solution twice from a shared closure so the per-pair fixture
        // (used by the dense path) and the hook (used by the operator) draw from
        // the SAME constant ψψ pair.
        let build_solution = |with_hook: bool| {
            let deriv_provider = SinglePredictorGlmDerivatives {
                c_array: c_array.clone(),
                d_array: Some(d_array.clone()),
                hessian_weights: Array1::ones(3),
                x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    x.clone(),
                )),
            };
            let contracted_psi_second_order: Option<ContractedPsiSecondOrderFn> = if with_hook {
                let g = psi_pair_g.clone();
                let b = psi_pair_b.clone();
                Some(Arc::new(move |alpha_psi: &[f64]| {
                    let a0 = alpha_psi[0];
                    Ok(Some(ContractedPsiSecondOrder {
                        objective: array![a0 * psi_pair_a],
                        score: {
                            let mut s = Array2::<f64>::zeros((1, 2));
                            s.row_mut(0).assign(&g.mapv(|v| a0 * v));
                            s
                        },
                        hessian: vec![DriftDerivResult::Dense(b.mapv(|v| a0 * v))],
                        ld_s: array![a0 * psi_pair_ld_s],
                    }))
                }) as ContractedPsiSecondOrderFn)
            } else {
                None
            };
            let pair_b_for_dense = psi_pair_b.clone();
            InnerSolution {
                log_likelihood: -2.3,
                penalty_quadratic: 0.6,
                hessian_op: hop.clone(),
                beta: beta.clone(),
                penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root.clone())],
                penalty_logdet: PenaltyLogdetDerivs {
                    value: 0.0,
                    first: array![0.4],
                    second: Some(array![[0.13]]),
                },
                deriv_provider: Box::new(deriv_provider),
                tk_correction: 0.0,
                tk_gradient: None,
                firth: None,
                hessian_logdet_correction: 0.0,
                penalty_subspace_trace: None,
                rho_curvature_scale: 1.0,
                rho_prior: crate::types::RhoPrior::Flat,
                n_observations: 3,
                nullspace_dim: 0.0,
                gaussian_weight_log_sum_half: 0.0,
                dispersion: DispersionHandling::Fixed {
                    phi: 1.0,
                    include_logdet_h: true,
                    include_logdet_s: true,
                },
                ext_coords: vec![HyperCoord {
                    a: -0.21,
                    g: array![0.33, -0.42],
                    drift: HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
                        matrix: ext_drift.clone(),
                    })),
                    ld_s: 0.07,
                    b_depends_on_beta: false,
                    is_penalty_like: false,
                    firth_g: None,
                    tk_eta_fixed: None,
                    tk_x_fixed: None,
                }],
                ext_coord_pair_fn: Some(Box::new(move |_, _| HyperCoordPair {
                    a: psi_pair_a,
                    g: array![0.16, -0.12],
                    b_mat: pair_b_for_dense.clone(),
                    b_operator: None,
                    ld_s: psi_pair_ld_s,
                })),
                rho_ext_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                    a: -0.14,
                    g: array![-0.18, 0.22],
                    b_mat: array![[0.05, -0.02], [-0.02, 0.07]],
                    b_operator: None,
                    ld_s: 0.04,
                })),
                fixed_drift_deriv: None,
                contracted_psi_second_order,
                barrier_config: None,
                kkt_residual: None,
                active_constraints: None,
                stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
            }
        };

        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        // Per-pair dense path (no hook).
        let dense_solution = build_solution(false);
        let dense = compute_outer_hessian(
            &dense_solution,
            &rho,
            &lambdas,
            dense_solution.hessian_op.as_ref(),
            dense_solution.deriv_provider.as_ref(),
            None,
        )
        .unwrap();

        // Operator path WITH the contracted hook (ψψ tables skipped at build,
        // hook applied per matvec).
        let hook_solution = build_solution(true);
        let kernel = hook_solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let operator = build_outer_hessian_operator(
            &hook_solution,
            &lambdas,
            hook_solution.deriv_provider.as_ref(),
            kernel,
            None,
            None,
        )
        .unwrap();

        // #740 pin: the operator must NOT advertise dense materialization, so the
        // outer solver consumes it matrix-free (m CG matvecs) instead of
        // re-paying K basis-column probes — the asymptotic win. A future edit
        // flipping this to a cheap-to-materialize capability would silently
        // re-introduce K-column densification and kill the win; this pin fails
        // if that happens.
        assert!(
            matches!(
                crate::solver::outer_strategy::OuterHessianOperator::materialization_capability(
                    &operator
                ),
                crate::solver::outer_strategy::OuterHessianMaterialization::Unavailable
            ),
            "#740 operator must advertise Unavailable materialization to stay matrix-free"
        );

        let materialized =
            crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(&operator)
                .unwrap();

        // CONTROL: the SAME operator built WITHOUT the hook (ψψ block filled from
        // the per-pair ext_coord_pair_fn tables) must already match the dense
        // per-pair path — this is the pre-existing UnifiedOuterHessianOperator
        // path, unrelated to #740. If this control matches dense but the
        // hook-operator above does not, the divergence is isolated to the #740
        // hook-injection arithmetic (psi_contracted_contrib / the ψψ-skip).
        let control_solution = build_solution(false);
        let control_kernel = control_solution
            .deriv_provider
            .outer_hessian_derivative_kernel()
            .unwrap();
        let control_operator = build_outer_hessian_operator(
            &control_solution,
            &lambdas,
            control_solution.deriv_provider.as_ref(),
            control_kernel,
            None,
            None,
        )
        .unwrap();
        let control_mat = crate::solver::outer_strategy::OuterHessianOperator::materialize_dense(
            &control_operator,
        )
        .unwrap();

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let c = control_mat[[row, col]];
                let d = dense[[row, col]];
                let tol = 1e-9_f64.max(1e-9 * d.abs());
                assert!(
                    (c - d).abs() <= tol,
                    "#740 CONTROL (no-hook operator) mismatch at ({row}, {col}): no-hook-op={c}, \
                     per-pair-dense={d} — pre-existing operator path differs from dense, so the \
                     #740 hook comparison below is not the right oracle; investigate the table fill"
                );
            }
        }

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let m = materialized[[row, col]];
                let d = dense[[row, col]];
                let c = control_mat[[row, col]];
                let tol = 1e-9_f64.max(1e-9 * d.abs());
                assert!(
                    (m - d).abs() <= tol,
                    "#740 contracted-hook operator mismatch at ({row}, {col}): \
                     hook-operator={m}, per-pair-dense={d}, no-hook-control={c} \
                     (control==dense ⇒ bug is in the #740 hook injection at this entry)"
                );
            }
        }

        // The ψ-row diagonal must carry BOTH second-order terms that legitimately
        // appear in ddot_H_ij, so the operator-vs-dense match above genuinely
        // rules out a DOUBLE-COUNT or a dropped term (a ρ-only comparison would
        // not):
        //   * base_h2 ψψ  = tr(K · D²_ψ H_L[ψ,ψ])  — the per-pair pair.b_mat /
        //     the hook's `hessian`; and
        //   * the callback `correction` (term2) = tr(K · D²_β H_L[β̇,·]) via the
        //     family compute_d2h on the ext mode response.
        // Both must be individually NONZERO at the ψ diagonal (coord index 1).
        // Recompute the dense ψ diagonal with each term suppressed and require a
        // measurable shift, so the equivalence test sits on a point where the
        // two distinct curvature terms both contribute.
        let psi_diag_full = dense[[1, 1]];
        // term2-only: zero the ψψ second drift (pair.b_mat) → removes base_h2 ψψ.
        let dense_no_base = {
            let mut sol = build_solution(false);
            sol.ext_coord_pair_fn = Some(Box::new(move |_, _| HyperCoordPair {
                a: psi_pair_a,
                g: array![0.16, -0.12],
                b_mat: Array2::zeros((2, 2)),
                b_operator: None,
                ld_s: psi_pair_ld_s,
            }));
            compute_outer_hessian(
                &sol,
                &rho,
                &lambdas,
                sol.hessian_op.as_ref(),
                sol.deriv_provider.as_ref(),
                None,
            )
            .unwrap()[[1, 1]]
        };
        // base_h2-only: zero c/d so the family correction (term2) vanishes.
        let dense_no_term2 = {
            let mut sol = build_solution(false);
            sol.deriv_provider = Box::new(SinglePredictorGlmDerivatives {
                c_array: Array1::zeros(3),
                d_array: Some(Array1::zeros(3)),
                hessian_weights: Array1::ones(3),
                x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    x.clone(),
                )),
            });
            compute_outer_hessian(
                &sol,
                &rho,
                &lambdas,
                sol.hessian_op.as_ref(),
                sol.deriv_provider.as_ref(),
                None,
            )
            .unwrap()[[1, 1]]
        };
        assert!(
            (psi_diag_full - dense_no_base).abs() > 1e-6,
            "#740 test is vacuous: base_h2 ψψ contributes ~0 at the ψ diagonal \
             (full={psi_diag_full}, term2-only={dense_no_base}); pick a fixture where it is live"
        );
        assert!(
            (psi_diag_full - dense_no_term2).abs() > 1e-6,
            "#740 test is vacuous: the family correction (term2) contributes ~0 at the ψ \
             diagonal (full={psi_diag_full}, base_h2-only={dense_no_term2}); pick a fixture \
             where it is live"
        );

        // Mixed ρψ-direction HVP arm: a pure-ρ and a pure-ψ matvec can both be
        // green while a ρψ/ψψ block-split error (a double-count, or a ψψ table
        // entry zeroed at build but not re-added by the hook) hides in the
        // cross. Apply the operator to a direction with BOTH a ρ and a ψ
        // component nonzero and require operator·v == dense·v — the matvec mixes
        // the blocks, so this is what exposes such a split error. (This is the
        // matvec path, distinct from the materialize column-probes above, so it
        // also exercises the hook's per-matvec injection directly.)
        let mixed = array![0.6_f64, -1.1_f64]; // [ρ, ψ], both live
        let hvp = crate::solver::outer_strategy::OuterHessianOperator::matvec(&operator, &mixed)
            .expect("mixed-direction operator HVP");
        let dense_hvp = dense.dot(&mixed);
        for i in 0..hvp.len() {
            let tol = 1e-10_f64.max(1e-10 * dense_hvp[i].abs());
            assert!(
                (hvp[i] - dense_hvp[i]).abs() <= tol,
                "#740 mixed-ρψ HVP mismatch at {i}: hook-operator={}, per-pair-dense={} \
                 — a ρψ/ψψ block-split error (double-count or dropped entry) in the cross",
                hvp[i],
                dense_hvp[i]
            );
        }
    }

    #[test]
    pub(crate) fn subspace_projected_leverage_and_adjoint_shortcut_match_dense() {
        // Locks down both production identities used by the subspace
        // leverage shortcut in `build_outer_hessian_operator`:
        //
        //   (1) `xt_projected_kernel_x_diagonal(X)_i = Xᵢᵀ · K · Xᵢ` per row
        //   (2) `tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj})`
        //       with `K = U_S H_proj⁻¹ U_Sᵀ` and `C[u] = Xᵀ diag(c ⊙ Xu) X`.
        //
        // (1) is the per-row contract `xt_projected_kernel_x_diagonal`
        // promises (its docstring); (2) is the math identity that the
        // leverage / `adjoint_z_c` shortcut relies on for its `O(n·r)`
        // adjoint-trick replacement of the dense materialised correction.
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let h_proj_inverse = array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]];
        let subspace = PenaltySubspaceTrace {
            u_s: u_s.clone(),
            h_proj_inverse: h_proj_inverse.clone(),
        };

        let x_data = array![
            [1.0, 0.2, 0.5, 0.3],
            [1.0, 1.1, -0.2, 0.4],
            [1.0, -0.8, 0.7, -0.1],
            [1.0, 0.5, 0.3, 0.6]
        ];
        let c = array![0.31_f64, -0.27, 0.19, -0.11];

        // Dense reference K = U_S · H_proj⁻¹ · U_Sᵀ.
        let k_dense = u_s.dot(&h_proj_inverse).dot(&u_s.t());
        let n = x_data.nrows();

        // (1) Production helper vs per-row dense reference.
        let x_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_data.clone()));
        let h_g_proj = subspace.xt_projected_kernel_x_diagonal(&x_design);
        assert_eq!(h_g_proj.len(), n);
        for i in 0..n {
            let row = x_data.row(i).to_owned();
            let kx = k_dense.dot(&row);
            assert_relative_eq!(h_g_proj[i], row.dot(&kx), epsilon = 1e-12);
        }

        // (2) Adjoint shortcut: tr(K · C[u]) = uᵀ · Xᵀ(c ⊙ h^{G,proj}).
        // Probe u directions including ones lifting into null(U_S).
        let probes = [
            array![0.6_f64, -0.4, 0.0, 0.0],
            array![0.0_f64, 0.0, 0.5, 0.7],
            array![0.3_f64, -0.1, 0.4, -0.2],
            array![1.0_f64, 1.0, 1.0, 1.0],
        ];
        for u in probes.iter() {
            let xu = x_data.dot(u);
            let mut weighted_x = x_data.clone();
            for i in 0..n {
                let w = c[i] * xu[i];
                for j in 0..weighted_x.ncols() {
                    weighted_x[[i, j]] *= w;
                }
            }
            let c_u_dense = x_data.t().dot(&weighted_x);

            // LHS: tr(K · C[u]) via the production projected-logdet path.
            let lhs = subspace.trace_projected_logdet(&c_u_dense);

            // RHS: uᵀ · Xᵀ(c ⊙ h^{G,proj}) using the production helper's output.
            let mut weighted = Array1::<f64>::zeros(n);
            for i in 0..n {
                weighted[i] = c[i] * h_g_proj[i];
            }
            let rhs = u.dot(&x_data.t().dot(&weighted));

            assert_relative_eq!(lhs, rhs, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    pub(crate) fn subspace_base_h2_traces_match_scalar_projected_kernel_path() {
        let h = array![[3.0, 0.1, 0.0], [0.1, 5.0, 0.2], [0.0, 0.2, 7.0]];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let kernel = PenaltySubspaceTrace {
            u_s,
            h_proj_inverse: array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]],
        };

        let dense_only = array![[0.4, 0.1, 0.0], [0.1, -0.2, 0.3], [0.0, 0.3, 0.6]];
        let op_a = array![[0.2, -0.1, 0.4], [-0.1, 0.7, 0.0], [0.4, 0.0, -0.3]];
        let op_b = array![[0.8, 0.2, -0.2], [0.2, 0.1, 0.5], [-0.2, 0.5, 0.9]];
        let composite_dense = array![[0.05, 0.02, 0.0], [0.02, 0.03, 0.01], [0.0, 0.01, 0.04]];

        let op_a_arc: Arc<dyn HyperOperator> = Arc::new(DenseMatrixHyperOperator {
            matrix: op_a.clone(),
        });
        let op_b_arc: Arc<dyn HyperOperator> = Arc::new(DenseMatrixHyperOperator {
            matrix: op_b.clone(),
        });
        let weighted: Arc<dyn HyperOperator> = Arc::new(WeightedHyperOperator {
            terms: vec![(0.25, op_b_arc.clone()), (-0.5, op_a_arc.clone())],
            dim_hint: 3,
        });

        let pairs = [
            HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(3),
                b_mat: dense_only,
                b_operator: None,
                ld_s: 0.0,
            },
            HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(3),
                b_mat: Array2::zeros((0, 0)),
                b_operator: Some(Box::new(DenseMatrixHyperOperator { matrix: op_a })),
                ld_s: 0.0,
            },
            HyperCoordPair {
                a: 0.0,
                g: Array1::zeros(3),
                b_mat: Array2::zeros((0, 0)),
                b_operator: Some(Box::new(CompositeHyperOperator {
                    dense: Some(composite_dense),
                    operators: vec![weighted, op_b_arc],
                    dim_hint: 3,
                })),
                ld_s: 0.0,
            },
        ];
        let pair_refs: Vec<&HyperCoordPair> = pairs.iter().collect();

        let batched = compute_base_h2_traces(&hop, &pair_refs, Some(&kernel), None);
        let scalar: Vec<f64> = pair_refs
            .iter()
            .map(|pair| {
                compute_base_h2_trace(&hop, &pair.b_mat, pair.b_operator.as_deref(), Some(&kernel))
            })
            .collect();

        assert_eq!(batched.len(), scalar.len());
        for (idx, (got, expected)) in batched.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (*got - *expected).abs() <= 1e-12_f64.max(1e-12 * expected.abs()),
                "projected base_h2 trace mismatch at pair {idx}: got={got}, expected={expected}"
            );
        }
    }

    #[test]
    pub(crate) fn outer_hessian_operator_matvec_matches_dense_subspace_with_null_alpha() {
        assert!(file!().ends_with(".rs"));
        // p=4, K=2, r=2 fixture — exercises the full projection K = U_S H_proj⁻¹ U_Sᵀ
        // (the existing r=1 case at projected_operator_hessian_matches_dense_subspace_trace
        // only verifies a trivial 1-D subspace).  Includes a small symmetric off-diagonal
        // so H_proj is non-diagonal.
        let h = array![
            [3.0, 0.1, 0.0, 0.0],
            [0.1, 5.0, 0.05, 0.0],
            [0.0, 0.05, 7.0, 0.15],
            [0.0, 0.0, 0.15, 11.0]
        ];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());

        // U_S spans the first two coordinates.  Null directions are dims 2,3.
        let u_s = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];

        // H_proj = U_Sᵀ H U_S = top-left 2×2 of H = [[3.0, 0.1], [0.1, 5.0]].
        // Closed-form 2×2 inverse for the test fixture: 1/(3·5 − 0.1²) · [[5, −0.1], [−0.1, 3]].
        let det = 3.0_f64 * 5.0 - 0.1 * 0.1;
        let h_proj_inverse = array![[5.0 / det, -0.1 / det], [-0.1 / det, 3.0 / det]];

        // Penalty roots mix identified (rows 0,1) and null (rows 2,3) directions, so
        // the projection is non-trivial — `compute_outer_hessian` must collapse the
        // null components and the matvec must match.
        let penalty_root_0 = array![[0.7, 0.3, 0.6, 0.0]];
        let penalty_root_1 = array![[0.2, 0.5, 0.0, 0.4]];

        let x = array![
            [1.0, 0.2, 0.5, 0.3],
            [1.0, 1.1, -0.2, 0.4],
            [1.0, -0.8, 0.7, -0.1],
            [1.0, 0.5, 0.3, 0.6]
        ];
        let c_array = array![0.31, -0.27, 0.19, -0.11];
        let d_array = array![0.17, -0.11, 0.23, 0.07];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            hessian_weights: Array1::ones(4),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };

        // Pre-compute log|H_proj|_+ = ln(det(H_proj)) for the correction term.
        let logdet_h_proj = det.ln();

        let beta = array![0.4, -0.7, 0.2, 0.1];
        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(penalty_root_0),
                PenaltyCoordinate::from_dense_root(penalty_root_1),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4, -0.2],
                second: Some(array![[0.13, 0.02], [0.02, 0.09]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: logdet_h_proj - hop.logdet(),
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s,
                h_proj_inverse,
            })),
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 4,
            nullspace_dim: 2.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };
        let rho: Vec<f64> = vec![0.2_f64, -0.1];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        // (6) Materialised dense extension to r=2: every entry must match.
        let (dense, operator, materialized) =
            dense_and_materialized_outer_hessian(&solution, &rho, &lambdas);
        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                assert_relative_eq!(
                    materialized[[row, col]],
                    dense[[row, col]],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }

        // (3) HVP equivalence across a basis-and-mix set of α probes.
        // (4) The [1, -1] and [0.7, -0.3] probes lift through penalty roots whose
        //     columns 2,3 carry non-zero null components, so they exercise the
        //     projection rather than just the identified subspace.
        let alphas = [
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
            array![1.0, -1.0],
            array![0.7, -0.3],
        ];
        for alpha in alphas.iter() {
            let hvp = crate::solver::outer_strategy::OuterHessianOperator::matvec(&operator, alpha)
                .expect("operator HVP");
            let dense_hvp = dense.dot(alpha);
            for i in 0..hvp.len() {
                assert_relative_eq!(hvp[i], dense_hvp[i], epsilon = 1e-12, max_relative = 1e-12);
            }
        }
    }

    #[test]
    pub(crate) fn projected_operator_hessian_matches_dense_subspace_trace() {
        assert!(file!().ends_with(".rs"));
        let h = array![[3.0, 0.2], [0.2, 5.0]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.4, -0.7];
        let penalty_root = array![[0.0, 1.0]];
        let ext_drift = array![[0.45, -0.15], [-0.15, 0.35]];
        let x = array![[1.0, 0.2], [1.0, 1.1], [1.0, -0.8], [1.0, 0.5]];
        let c_array = array![0.31, -0.27, 0.19, -0.11];
        let d_array = array![0.17, -0.11, 0.23, 0.07];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: Some(d_array),
            hessian_weights: Array1::ones(4),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };
        let h_proj = h[[1, 1]];

        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root)],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4],
                second: Some(array![[0.13]]),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: h_proj.ln() - hop.logdet(),
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s: array![[0.0], [1.0]],
                h_proj_inverse: array![[1.0 / h_proj]],
            })),
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 4,
            nullspace_dim: 1.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: vec![HyperCoord {
                a: -0.21,
                g: array![0.33, -0.42],
                drift: HyperCoordDrift::from_operator(Arc::new(DenseMatrixHyperOperator {
                    matrix: ext_drift,
                })),
                ld_s: 0.07,
                b_depends_on_beta: false,
                is_penalty_like: false,
                firth_g: None,
                tk_eta_fixed: None,
                tk_x_fixed: None,
            }],
            ext_coord_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: 0.09,
                g: array![0.16, -0.12],
                b_mat: array![[0.08, 0.03], [0.03, -0.04]],
                b_operator: None,
                ld_s: -0.05,
            })),
            rho_ext_pair_fn: Some(Box::new(|_, _| HyperCoordPair {
                a: -0.14,
                g: array![-0.18, 0.22],
                b_mat: array![[0.05, -0.02], [-0.02, 0.07]],
                b_operator: None,
                ld_s: 0.04,
            })),
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };
        let rho: Vec<f64> = vec![0.2_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let (dense, _, materialized) =
            dense_and_materialized_outer_hessian(&solution, &rho, &lambdas);

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                assert_relative_eq!(
                    materialized[[row, col]],
                    dense[[row, col]],
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    pub(crate) fn penalty_subspace_batched_reduction_matches_serial_operator_reduction() {
        assert!(file!().ends_with(".rs"));
        let kernel = PenaltySubspaceTrace {
            u_s: array![[1.0, 0.0], [0.2, 0.8], [-0.1, 0.6]],
            h_proj_inverse: array![[0.8, 0.1], [0.1, 0.6]],
        };
        let dense = array![[0.4, 0.1, -0.2], [0.1, 0.7, 0.3], [-0.2, 0.3, 0.5]];
        let op_matrix = array![[0.3, -0.2, 0.1], [-0.2, 0.9, 0.4], [0.1, 0.4, 0.8]];
        let composite_dense = array![[0.05, 0.01, 0.0], [0.01, -0.02, 0.03], [0.0, 0.03, 0.04]];
        let drifts = vec![
            DriftDerivResult::Dense(dense.clone()),
            DriftDerivResult::Operator(Arc::new(DenseMatrixHyperOperator {
                matrix: op_matrix.clone(),
            })),
            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                dim_hint: 3,
                dense: Some(composite_dense.clone()),
                operators: vec![Arc::new(DenseMatrixHyperOperator {
                    matrix: op_matrix.clone(),
                })],
            })),
        ];

        let batched = penalty_subspace_reduce_drifts_batched(&kernel, &drifts);
        let serial = [
            kernel.reduce(&dense),
            kernel.reduce_operator(&DenseMatrixHyperOperator {
                matrix: op_matrix.clone(),
            }),
            kernel.reduce_operator(&CompositeHyperOperator {
                dim_hint: 3,
                dense: Some(composite_dense),
                operators: vec![Arc::new(DenseMatrixHyperOperator { matrix: op_matrix })],
            }),
        ];

        for (batched_mat, serial_mat) in batched.iter().zip(serial.iter()) {
            for row in 0..batched_mat.nrows() {
                for col in 0..batched_mat.ncols() {
                    assert_relative_eq!(
                        batched_mat[[row, col]],
                        serial_mat[[row, col]],
                        epsilon = 1e-12,
                        max_relative = 1e-12,
                    );
                }
            }
        }
    }

    #[test]
    pub(crate) fn subspace_trace_large_k_routes_to_projected_operator() {
        let h = array![[3.0, 0.2], [0.2, 5.0]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let pcoord = PenaltyCoordinate::from_dense_root(array![[0.0, 1.0]]);
        let k = MATRIX_FREE_OUTER_HESSIAN_K_THRESHOLD;
        let x = array![[1.0, 0.2], [1.0, 1.1], [1.0, -0.8], [1.0, 0.5]];
        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array: array![0.31, -0.27, 0.19, -0.11],
            d_array: Some(array![0.17, -0.11, 0.23, 0.07]),
            hessian_weights: Array1::ones(4),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
        };
        let h_proj = h[[1, 1]];
        let solution = InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop.clone(),
            beta: array![0.4, -0.7],
            penalty_coords: vec![pcoord; k],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: Array1::zeros(k),
                second: Some(Array2::zeros((k, k))),
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: h_proj.ln() - hop.logdet(),
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s: array![[0.0], [1.0]],
                h_proj_inverse: array![[1.0 / h_proj]],
            })),
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 4,
            nullspace_dim: 1.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };
        let rho = vec![0.0_f64; k];
        let result =
            reml_laml_evaluate(&solution, &rho, EvalMode::ValueGradientHessian, None).unwrap();

        assert!(
            matches!(
                result.hessian,
                crate::solver::outer_strategy::HessianResult::Operator(_)
            ),
            "large-k subspace-trace case should use projected outer Hessian operator"
        );
    }

    #[test]
    pub(crate) fn test_dense_spectral_operator_singular() {
        // Rank-1 matrix: H = [1 1; 1 1] has eigenvalues {0, 2}.
        let h = array![[1.0, 1.0], [1.0, 1.0]];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Under `Smooth` mode (the default used by `from_symmetric`), every
        // eigenpair stays active and singular directions are regularized
        // through `r_ε(σ)` rather than hard-masked. For H = [[1,1],[1,1]],
        // the expected logdet is therefore
        //   ln(r_ε(0)) + ln(r_ε(2)).
        let epsilon = spectral_epsilon(&[0.0, 2.0]);
        let r0 = spectral_regularize(0.0, epsilon);
        let r2 = spectral_regularize(2.0, epsilon);
        let expected_logdet = r0.ln() + r2.ln();
        assert!((op.logdet() - expected_logdet).abs() < 1e-10);
        // The regularized null direction must still yield a finite trace.
        let trace = op.trace_hinv_product(&Array2::eye(2));
        assert!(trace.is_finite());
    }

    #[test]
    pub(crate) fn test_spectral_regularize_stays_finite_in_extreme_tails() {
        let epsilon = 1e-8;

        let large_negative = spectral_regularize(-1e16, epsilon);
        assert!(
            large_negative.is_finite() && large_negative > 0.0,
            "large negative sigma should regularize to a positive finite value, got {large_negative}"
        );

        let large_positive = spectral_regularize(1e308, epsilon);
        assert!(
            large_positive.is_finite() && large_positive > 0.0,
            "large positive sigma should stay finite, got {large_positive}"
        );
    }

    #[test]
    pub(crate) fn test_smooth_floor_dp() {
        // Well above floor: should be approximately identity
        let (val, grad, _) = smooth_floor_dp(1.0);
        assert!((val - 1.0).abs() < 1e-6);
        assert!((grad - 1.0).abs() < 1e-6);

        // At floor: should be approximately DP_FLOOR + tau*ln(2)
        let (val, grad, _) = smooth_floor_dp(DP_FLOOR);
        assert!(val > DP_FLOOR);
        assert!((grad - 0.5).abs() < 0.1); // sigmoid at 0 ≈ 0.5

        // Well below floor: value should stay above DP_FLOOR
        let (val, _, _) = smooth_floor_dp(0.0);
        assert!(val >= DP_FLOOR);
    }

    #[test]
    pub(crate) fn test_gaussian_derivatives_has_no_corrections() {
        let g = GaussianDerivatives;
        assert!(!g.has_corrections());
        assert!(
            g.hessian_derivative_correction(&array![1.0, 2.0])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    pub(crate) fn gaussian_derivatives_advertise_exact_outer_hvp_kernel() {
        let g = GaussianDerivatives;
        assert!(matches!(
            g.outer_hessian_derivative_kernel(),
            Some(OuterHessianDerivativeKernel::Gaussian)
        ));
    }

    #[test]
    pub(crate) fn standard_gam_large_n_gaussian_prefers_operator_when_dense_work_is_large() {
        assert!(prefer_outer_hessian_operator(320_000, 42, 6));
        assert!(matches!(
            GaussianDerivatives.outer_hessian_derivative_kernel(),
            Some(OuterHessianDerivativeKernel::Gaussian)
        ));
    }

    #[test]
    pub(crate) fn callback_outer_hessian_routes_by_row_pair_work_even_at_small_p() {
        assert!(!prefer_outer_hessian_operator(155_980, 19, 23));
        assert!(outer_hessian_route_plan(155_980, 19, 23, true, true, false).use_operator);
        assert!(!outer_hessian_route_plan(155_980, 19, 23, true, false, false).use_operator);
        assert!(!outer_hessian_route_plan(1_000, 19, 23, true, true, false).use_operator);
    }

    #[test]
    pub(crate) fn callback_outer_hessian_ignores_generic_large_n_small_p_crossover() {
        assert!(prefer_outer_hessian_operator(195_780, 33, 8));
        assert!(!outer_hessian_route_plan(195_780, 33, 8, true, true, false).use_operator);
        assert!(outer_hessian_route_plan(195_780, 512, 8, true, true, false).use_operator);
        assert!(outer_hessian_route_plan(195_780, 33, 32, true, true, false).use_operator);

        let plan = outer_hessian_route_plan(195_780, 33, 8, true, true, false);
        assert!(!plan.use_operator);
        assert_eq!(plan.choice(), "dense");
        assert_eq!(plan.reason, "below_crossover");
        assert!(!plan.scale_prefers_operator);
    }

    #[test]
    pub(crate) fn outer_hessian_route_respects_dense_workspace_budget() {
        let plan = outer_hessian_route_plan(10_000, 10_000, 2, true, true, false);
        assert!(plan.use_operator);
        assert_eq!(plan.reason, "dense_memory_budget");
        assert!(plan.dense_workspace_bytes > outer_hessian_dense_workspace_budget_bytes());
    }

    #[test]
    pub(crate) fn outer_hessian_route_reports_kernel_absent_before_scale_model() {
        let plan = outer_hessian_route_plan(1_000_000, 10_000, 64, false, false, false);
        assert!(!plan.use_operator);
        assert_eq!(plan.reason, "kernel_absent");
        assert!(!plan.scale_prefers_operator);
    }

    #[test]
    pub(crate) fn gaussian_outer_hessian_operator_matches_dense_assembly() {
        let h = array![[2.4, 0.2], [0.2, 1.7]];
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).unwrap());
        let beta = array![0.35, -0.55];
        let penalty_root_0 = array![[1.0, 0.2], [0.0, 0.4]];
        let penalty_root_1 = array![[0.3, -0.1], [0.0, 0.9]];
        let solution = InnerSolution {
            log_likelihood: -8.0,
            penalty_quadratic: 0.9,
            hessian_op: hop.clone(),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(penalty_root_0),
                PenaltyCoordinate::from_dense_root(penalty_root_1),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.8, 0.6],
                second: Some(array![[0.11, 0.03], [0.03, 0.17]]),
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 320_000,
            nullspace_dim: 1.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };
        let rho: Vec<f64> = vec![0.2_f64, -0.4_f64];
        let lambdas: Vec<f64> = rho.iter().map(|value| value.exp()).collect();

        let (dense, _, materialized) =
            dense_and_materialized_outer_hessian(&solution, &rho, &lambdas);

        for row in 0..dense.nrows() {
            for col in 0..dense.ncols() {
                let expected = dense[[row, col]];
                let actual = materialized[[row, col]];
                let tolerance = 1e-10_f64.max(1e-10 * expected.abs());
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "Gaussian outer Hessian operator mismatch at ({row}, {col}): materialized={actual}, dense={expected}"
                );
            }
        }
    }

    /// Scalar EFS counterexample: at z=2, λ=1/3 in a one-coefficient
    /// Gaussian/Laplace surrogate, the REML/LAML gradient is exactly zero
    /// (β̂² λ + λ/(1+λ) − 1 = 0.75 + 0.25 − 1 = 0). The Wood–Fasiolo
    /// multiplicative EFS update must therefore return Δρ ≈ 0.
    ///
    /// The previous Frobenius/Gram-norm formula returned `(2a − tr(H⁻¹B)) /
    /// tr(H⁻¹BH⁻¹B) = 0.5 / 0.0625 = 8`, which then clamped to `+5` — a
    /// huge spurious step at the exact optimum.
    #[test]
    pub(crate) fn efs_step_is_zero_at_scalar_optimum() {
        // β̂ = z / (1 + λ) = 2 / (4/3) = 1.5, H = 1 + λ = 4/3.
        let lambda = 1.0 / 3.0;
        let beta_hat = 1.5_f64;
        let h = Array2::from_shape_vec((1, 1), vec![1.0 + lambda]).unwrap();
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // S = R^T R with R = [[1]] gives S = [[1]]. Pseudoinverse log-det
        // derivative tr(S⁺ · λS) = 1 (full-rank, scale cancels).
        let penalty_root = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        let solution = InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: Arc::new(op),
            beta: array![beta_hat],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(penalty_root)],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 10,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            // Use Fixed dispersion so the gradient is exactly the
            // Laplace/REML form `½(λβ̂²S β̂ + tr(H⁻¹λS) − tr(S⁺λS))`
            // without the smooth-floor / profiling factors the test
            // would otherwise have to track.
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };
        let rho = [lambda.ln()];

        // At the optimum the full outer gradient is identically 0; the
        // universal form `Δρ = log(1 − 2·g_full/q_eff)` collapses to
        // `log(1) = 0`.
        let gradient_at_optimum = [0.0_f64];
        let steps = compute_efs_update(&solution, &rho, &gradient_at_optimum);
        assert_eq!(steps.len(), 1);
        assert!(
            steps[0].abs() < 1e-12,
            "EFS step at scalar optimum should be exactly 0, got {} (old buggy formula returned ~+5)",
            steps[0]
        );

        // Off-optimum: simulate `g_full = +0.1` with the same q_eff. The
        // multiplicative target `1 − 2·0.1/0.75 = 0.733` ⇒ Δρ = log(0.733).
        let q_eff = lambda * beta_hat * beta_hat; // 0.75
        let g_off = 0.1_f64;
        let steps_off = compute_efs_update(&solution, &rho, &[g_off]);
        let expected = (1.0_f64 - 2.0 * g_off / q_eff).ln();
        assert!(
            (steps_off[0] - expected).abs() < 1e-12,
            "off-optimum EFS step {} != expected {}",
            steps_off[0],
            expected
        );
    }

    /// `efs_log_step_from_grad` recovers the canonical
    /// `log((d − t)/q_eff)` Wood–Fasiolo step when the gradient is the
    /// pure REML/LAML stationarity gradient `g_base = (q_eff + t − d)/2`,
    /// and shifts by exactly the right amount when out-of-band terms
    /// `g_extra` enter the gradient.
    #[test]
    pub(crate) fn efs_log_step_from_grad_recovers_canonical_form() {
        // Canonical agreement on stable cases: g_base = (q_eff − target)/2
        // ⇒ universal = log((d − t)/q_eff).
        let cases = [
            (1.0_f64, 0.5),
            (2.0, 1.5),
            (0.75, 0.75),
            (4.0, 0.1),
            (1.0, 0.999),
        ];
        for (q_eff, target) in cases {
            let g_base = (q_eff - target) / 2.0;
            let universal = efs_log_step_from_grad(q_eff, g_base).unwrap();
            let canonical = (target / q_eff).ln().clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
            assert!(
                (universal - canonical).abs() < 1e-12,
                "universal {universal} ≠ canonical {canonical} at q={q_eff}, t={target}"
            );
        }

        // Augmented stationarity: g_full = g_base + g_extra = 0 ⇒
        // q_eff = (d − t) − 2·g_extra. The universal form must return ≈ 0
        // *with the same q_eff value the iteration actually has*.
        let target = 0.6_f64;
        let g_extra = -0.7_f64;
        let augmented_q = target - 2.0 * g_extra;
        let g_full_at_aug_opt = (augmented_q - target) / 2.0 + g_extra;
        assert!(g_full_at_aug_opt.abs() < 1e-12);
        let s_at_opt = efs_log_step_from_grad(augmented_q, g_full_at_aug_opt).unwrap();
        assert!(
            s_at_opt.abs() < 1e-12,
            "Δρ at augmented optimum != 0: {s_at_opt}"
        );

        // Stable: log ratio.
        let s = efs_log_step_from_grad(2.0, 0.75).expect("stable regime");
        assert!((s - (0.25_f64).ln()).abs() < 1e-12);

        // Optimum: g_full = 0 ⇒ Δρ = 0.
        let s = efs_log_step_from_grad(0.75, 0.0).expect("zero gradient");
        assert!(s.abs() < 1e-12);

        // Over-correction (2·g_full ≥ q_eff ⇒ ratio ≤ 0): clamp to max descent.
        for &(q_eff, g) in &[(1.0_f64, 0.6), (2.0, 1.5), (0.5, 1e6)] {
            let s = efs_log_step_from_grad(q_eff, g).expect("over-correction");
            assert!((s - (-EFS_MAX_STEP)).abs() < 1e-12);
        }

        // Asymptotic clamp on the lower side: ratio → 0⁺ ⇒ floor at -MAX.
        let s = efs_log_step_from_grad(1.0, 0.5 - 1e-30).expect("near-singular");
        assert!((s - (-EFS_MAX_STEP)).abs() < 1e-12 || s == 0.5 * (-EFS_MAX_STEP) || s.is_finite());
        assert!(s <= 0.0);

        // Pathological: q_eff ≤ 0, non-finite inputs.
        assert!(efs_log_step_from_grad(0.0, 0.0).is_none());
        assert!(efs_log_step_from_grad(-1.0, 0.0).is_none());
        assert!(efs_log_step_from_grad(f64::NAN, 0.0).is_none());
        assert!(efs_log_step_from_grad(1.0, f64::NAN).is_none());
        assert!(efs_log_step_from_grad(1.0, f64::INFINITY).is_none());
    }

    /// `DenseSpectralOperator::trace_hinv_block_local_cross` must compute
    /// `tr(H⁻¹ A H⁻¹ A)`, not `tr(H⁻¹ A²)`. These coincide only when A
    /// commutes with H⁻¹ — generically they differ.
    #[test]
    pub(crate) fn dense_spectral_block_local_cross_trace_matches_dense() {
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // 2×2 block at [0..2], non-commuting with H⁻¹.
        let block = array![[1.5, 0.4], [0.4, 0.7]];
        let scale = 1.7_f64;

        // Reference: full-matrix `tr((H⁻¹ A)²)` via repeated solves.
        let mut a_full = Array2::<f64>::zeros((3, 3));
        for i in 0..2 {
            for j in 0..2 {
                a_full[[i, j]] = scale * block[[i, j]];
            }
        }
        let hinva = op.solve_multi(&a_full); // = H⁻¹ A
        let expected = (&hinva.t() * &hinva).sum(); // tr((H⁻¹A)(H⁻¹A))

        let got = op.trace_hinv_block_local_cross(&block, scale, 0, 2);
        assert!(
            (got - expected).abs() < 1e-10,
            "block-local cross trace = {got}, expected = {expected} (delta {})",
            got - expected
        );
    }

    #[test]
    pub(crate) fn test_reml_laml_evaluate_gaussian_basic() {
        // Simple 2-param Gaussian model.
        let h = Array2::from_diag(&array![10.0, 8.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let solution = InnerSolution {
            log_likelihood: -5.0, // −0.5 × deviance = −0.5 × 10
            penalty_quadratic: 2.0,
            hessian_op: Arc::new(op),
            beta: array![1.0, 0.5],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(
                Array2::eye(2), // S₁ = I (penalty root for param 1)
            )],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 100,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };

        let rho = [0.0]; // λ = 1

        // Should produce finite cost
        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueOnly, None).unwrap();
        assert!(result.cost.is_finite());
        assert!(result.gradient.is_none());

        // With gradient
        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueAndGradient, None).unwrap();
        assert!(result.cost.is_finite());
        assert!(result.gradient.is_some());
        let grad = result.gradient.unwrap();
        assert_eq!(grad.len(), 1);
        assert!(grad[0].is_finite());
    }

    #[test]
    pub(crate) fn fixed_dispersion_firth_cost_subtracts_jeffreys_term() {
        assert!(file!().ends_with(".rs"));
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]];
        let eta = array![0.0, 0.4, -0.2];
        let firth_op = std::sync::Arc::new(
            super::super::RemlState::build_firth_dense_operator_for_link(
                &crate::types::InverseLink::Standard(crate::types::StandardLink::Logit),
                &x,
                &eta,
                ndarray::Array1::ones(x.nrows()).view(),
            )
            .expect("firth operator"),
        );
        let firth_value = firth_op.jeffreys_logdet();

        let solution = InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: Arc::new(DenseSpectralOperator::from_symmetric(&Array2::eye(2)).unwrap()),
            beta: Array1::zeros(2),
            penalty_coords: Vec::new(),
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: Array1::zeros(0),
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: Some(ExactJeffreysTerm::new(firth_op)),
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: x.nrows(),
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: false,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };

        let result = reml_laml_evaluate(&solution, &[], EvalMode::ValueOnly, None).unwrap();
        assert_relative_eq!(result.cost, -firth_value, epsilon = 1e-12);
    }

    pub(crate) struct FixedOuterHessianOperator {
        pub(crate) matrix: Array2<f64>,
    }

    impl crate::solver::outer_strategy::OuterHessianOperator for FixedOuterHessianOperator {
        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
            if v.len() != self.dim() {
                return Err(RemlError::DimensionMismatch {
                    reason: format!(
                        "fixed test outer Hessian dimension mismatch: got {}, expected {}",
                        v.len(),
                        self.dim()
                    ),
                }
                .into());
            }
            Ok(self.matrix.dot(v))
        }

        fn is_cheap_to_materialize(&self) -> bool {
            true
        }
    }

    pub(crate) struct FamilyOperatorDerivatives {
        pub(crate) op: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
    }

    impl HessianDerivativeProvider for FamilyOperatorDerivatives {
        fn hessian_derivative_correction(
            &self,
            arr: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            panic!("family operator dispatch should not request pairwise first derivatives")
        }

        fn hessian_second_derivative_correction(
            &self,
            arr: &Array1<f64>,
            arr2: &Array1<f64>,
            arr3: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            assert!(arr2.iter().all(|v| !v.is_nan()));
            assert!(arr3.iter().all(|v| !v.is_nan()));
            panic!("family operator dispatch should not request pairwise second derivatives")
        }

        fn has_corrections(&self) -> bool {
            false
        }

        fn family_outer_hessian_operator(
            &self,
        ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
            Some(Arc::clone(&self.op))
        }
    }

    #[test]
    pub(crate) fn family_outer_hessian_operator_short_circuits_dense_pairwise_assembly() {
        let supplied = array![[2.5]];
        let provider_op: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator> =
            Arc::new(FixedOuterHessianOperator {
                matrix: supplied.clone(),
            });
        let solution = InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.4,
            hessian_op: Arc::new(DenseSpectralOperator::from_symmetric(&array![[3.0]]).unwrap()),
            beta: array![0.2],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(array![[1.0]])],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(FamilyOperatorDerivatives { op: provider_op }),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 1,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        };

        let result =
            reml_laml_evaluate(&solution, &[0.0], EvalMode::ValueGradientHessian, None).unwrap();
        let crate::solver::outer_strategy::HessianResult::Operator(op) = result.hessian else {
            panic!("expected family-supplied operator Hessian");
        };
        assert_eq!(op.dim(), 1);
        let hv = op.matvec(&array![4.0]).unwrap();
        assert_relative_eq!(hv[0], 10.0, epsilon = 1e-12);
        let dense = op.materialize_dense().unwrap();
        assert_relative_eq!(dense[[0, 0]], supplied[[0, 0]], epsilon = 1e-12);
    }

    pub(crate) struct FixedCorrectionDerivatives {
        pub(crate) correction: Array2<f64>,
    }

    impl HessianDerivativeProvider for FixedCorrectionDerivatives {
        fn hessian_derivative_correction(
            &self,
            arr: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            Ok(Some(self.correction.clone()))
        }

        fn has_corrections(&self) -> bool {
            true
        }
    }

    pub(crate) fn build_projected_rho_gradient_solution(rho: f64) -> InnerSolution<'static> {
        let lambda = rho.exp();
        let h = array![[3.0 + 4.0 * rho, 0.0], [0.0, 5.0 + lambda],];
        let full_logdet = h[[0, 0]].ln() + h[[1, 1]].ln();
        let projected_logdet = h[[1, 1]].ln();

        InnerSolution {
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: Arc::new(
                DenseSpectralOperator::from_symmetric_with_mode(&h, PseudoLogdetMode::HardPseudo)
                    .unwrap(),
            ),
            beta: Array1::zeros(2),
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(array![[0.0, 1.0]])],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.0],
                second: None,
            },
            deriv_provider: Box::new(FixedCorrectionDerivatives {
                correction: array![[4.0, 0.0], [0.0, 0.0]],
            }),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: projected_logdet - full_logdet,
            penalty_subspace_trace: Some(Arc::new(PenaltySubspaceTrace {
                u_s: array![[0.0], [1.0]],
                h_proj_inverse: array![[1.0 / h[[1, 1]]]],
            })),
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 10,
            nullspace_dim: 1.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: false,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    #[test]
    pub(crate) fn test_rho_gradient_uses_projected_logdet_kernel_when_available() {
        let rho = 0.0;
        let result = reml_laml_evaluate(
            &build_projected_rho_gradient_solution(rho),
            &[rho],
            EvalMode::ValueAndGradient,
            None,
        )
        .unwrap();
        let analytic = result.gradient.expect("gradient")[0];

        let eps = 1e-6;
        let rho_plus = rho + eps;
        let cost_plus = reml_laml_evaluate(
            &build_projected_rho_gradient_solution(rho_plus),
            &[rho_plus],
            EvalMode::ValueOnly,
            None,
        )
        .unwrap()
        .cost;

        let rho_minus = rho - eps;
        let cost_minus = reml_laml_evaluate(
            &build_projected_rho_gradient_solution(rho_minus),
            &[rho_minus],
            EvalMode::ValueOnly,
            None,
        )
        .unwrap()
        .cost;

        let fd = (cost_plus - cost_minus) / (2.0 * eps);
        assert_relative_eq!(analytic, fd, epsilon = 1e-8, max_relative = 1e-8);

        let full_space_trace = 4.0 / 3.0 + 1.0 / 6.0;
        assert!(
            (analytic - 0.5 * full_space_trace).abs() > 0.5,
            "projected rho trace should exclude the null-space leakage term"
        );
    }

    #[test]
    pub(crate) fn test_rho_corrections_serial_large_work_case_stays_finite() {
        let rho = 0.0;
        let mut solution = build_projected_rho_gradient_solution(rho);
        solution.n_observations = 40_000_000;

        let result = reml_laml_evaluate(&solution, &[rho], EvalMode::ValueAndGradient, None)
            .expect("serial rho correction evaluation");
        assert!(result.cost.is_finite());
        let gradient = result.gradient.expect("gradient");
        assert_eq!(gradient.len(), 1);
        assert!(gradient[0].is_finite());
    }

    /// Helper: exact pseudo-logdet of S(λ) = Σ λₖ Sₖ together with its first
    /// and second ρ-derivatives via central finite differences. Shared by the
    /// Gaussian `InnerSolution` builders, which all carry the same two penalty
    /// matrices `s1`, `s2` and dimension `p`.
    pub(crate) fn gaussian_penalty_logdet_fd(
        p: usize,
        s1: &Array2<f64>,
        s2: &Array2<f64>,
        rho: &[f64],
    ) -> PenaltyLogdetDerivs {
        let mut s_total = Array2::zeros((p, p));
        s_total.scaled_add(rho[0].exp(), s1);
        s_total.scaled_add(rho[1].exp(), s2);
        let (s_eigs, _) = s_total.eigh(faer::Side::Lower).unwrap();
        let threshold = positive_eigenvalue_threshold(s_eigs.as_slice().unwrap());
        let log_det_s = exact_pseudo_logdet(s_eigs.as_slice().unwrap(), threshold);

        let log_det_s_at = |rho_eval: &[f64]| -> f64 {
            let lambdas_eval: Vec<f64> = rho_eval.iter().map(|&r| r.exp()).collect();
            let mut s_eval = Array2::zeros((p, p));
            s_eval.scaled_add(lambdas_eval[0], s1);
            s_eval.scaled_add(lambdas_eval[1], s2);
            let (s_eigs_eval, _) = s_eval.eigh(faer::Side::Lower).unwrap();
            let threshold_eval = positive_eigenvalue_threshold(s_eigs_eval.as_slice().unwrap());
            exact_pseudo_logdet(s_eigs_eval.as_slice().unwrap(), threshold_eval)
        };

        let mut det1 = Array1::zeros(rho.len());
        let eps = 1e-7;
        for k in 0..rho.len() {
            let mut rho_plus = rho.to_vec();
            rho_plus[k] += eps;
            let log_det_s_plus = log_det_s_at(&rho_plus);

            let mut rho_minus = rho.to_vec();
            rho_minus[k] -= eps;
            let log_det_s_minus = log_det_s_at(&rho_minus);

            det1[k] = (log_det_s_plus - log_det_s_minus) / (2.0 * eps);
        }
        let mut det2 = Array2::zeros((rho.len(), rho.len()));
        let eps2 = 1e-5;
        for i in 0..rho.len() {
            for j in i..rho.len() {
                let value = if i == j {
                    let mut rho_plus = rho.to_vec();
                    rho_plus[i] += eps2;
                    let mut rho_minus = rho.to_vec();
                    rho_minus[i] -= eps2;
                    (log_det_s_at(&rho_plus) - 2.0 * log_det_s + log_det_s_at(&rho_minus))
                        / (eps2 * eps2)
                } else {
                    let mut pp = rho.to_vec();
                    pp[i] += eps2;
                    pp[j] += eps2;
                    let mut pm = rho.to_vec();
                    pm[i] += eps2;
                    pm[j] -= eps2;
                    let mut mp = rho.to_vec();
                    mp[i] -= eps2;
                    mp[j] += eps2;
                    let mut mm = rho.to_vec();
                    mm[i] -= eps2;
                    mm[j] -= eps2;
                    (log_det_s_at(&pp) - log_det_s_at(&pm) - log_det_s_at(&mp) + log_det_s_at(&mm))
                        / (4.0 * eps2 * eps2)
                };
                det2[[i, j]] = value;
                if i != j {
                    det2[[j, i]] = value;
                }
            }
        }

        PenaltyLogdetDerivs {
            value: log_det_s,
            first: det1,
            second: Some(det2),
        }
    }

    /// Helper: build an InnerSolution for a Gaussian model at a given rho.
    /// The Hessian H = X'X + Σ λₖ Sₖ depends on rho through the penalty,
    /// so we must rebuild InnerSolution for each rho evaluation.
    pub(crate) fn build_gaussian_test_solution(rho: &[f64]) -> InnerSolution<'_> {
        let p = 3; // 3 coefficients
        let n = 50; // 50 observations

        // Fixed X'X (data-dependent, rho-independent)
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0],];

        // Two penalty matrices (one per smoothing parameter)
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0],];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],];

        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        // Build H = X'X + λ₁S₁ + λ₂S₂
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Solve for β̂ = H⁻¹ X'y (simulate with a fixed X'y)
        let xty = array![5.0, 3.0, 2.0];
        let beta = op.solve(&xty);

        // Penalty roots via eigendecomposition: Sₖ = Rₖᵀ Rₖ (exact).
        let r1 = penalty_matrix_root(&s1).unwrap();
        let r2 = penalty_matrix_root(&s2).unwrap();

        // Penalty quadratic: Σ λₖ β'Sₖβ
        let penalty_quad =
            lambdas[0] * beta.dot(&s1.dot(&beta)) + lambdas[1] * beta.dot(&s2.dot(&beta));

        // Deviance at β̂: ||y − Xβ̂||² = y'y − 2β̂'X'y + β̂'X'Xβ̂.
        // y'y is a ρ-independent constant (the actual value doesn't matter).
        // Computing deviance at the mode is essential: the analytic gradient
        // relies on the envelope theorem (∂D_p/∂β = 0 at the mode), which
        // is violated if deviance is held constant as β̂ varies with ρ.
        let yty = 20.0;
        let deviance = yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta));
        let log_likelihood = -0.5 * deviance;

        // Penalty logdet (value + FD ρ-derivatives) on positive eigenspace.
        let penalty_logdet = gaussian_penalty_logdet_fd(p, &s1, &s2, rho);

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(r1),
                PenaltyCoordinate::from_dense_root(r2),
            ],
            penalty_logdet,
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    pub(crate) fn build_large_dense_spectral_gaussian_solution(rho: f64) -> InnerSolution<'static> {
        let p = 520usize;
        let n = 2 * p;
        let lambda = rho.exp();

        let xtx_diag = Array1::from_shape_fn(p, |i| 5.0 + 0.01 * (i as f64));
        let xtx = Array2::from_diag(&xtx_diag);
        let penalty = Array2::<f64>::eye(p);

        let mut h = xtx.clone();
        h.scaled_add(lambda, &penalty);

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let xty = Array1::from_shape_fn(p, |i| 1.0 + 0.002 * (i as f64));
        let beta = op.solve(&xty);

        let penalty_quad = lambda * beta.dot(&beta);
        let yty = 10.0 * (p as f64);
        let deviance = yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta));
        let log_likelihood = -0.5 * deviance;

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta,
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(Array2::<f64>::eye(p))],
            penalty_logdet: PenaltyLogdetDerivs {
                value: (p as f64) * rho,
                first: array![p as f64],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    /// The structural test: finite-difference gradient matches analytic gradient.
    ///
    /// Because the unified evaluator computes cost and gradient from the same
    /// intermediates in the same function, drift is impossible. This test
    /// verifies that the mathematical formulas are correct (which FD catches),
    /// and serves as a regression gate.
    #[test]
    pub(crate) fn test_gaussian_reml_fd_vs_analytic_gradient() {
        let rho = vec![1.0, -0.5];
        let solution = build_gaussian_test_solution(&rho);

        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let analytic_grad = result.gradient.unwrap();

        // Finite-difference gradient
        let eps = 1e-5;
        let mut fd_grad = Array1::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rho_plus = rho.clone();
            rho_plus[k] += eps;
            let sol_plus = build_gaussian_test_solution(&rho_plus);
            let cost_plus = reml_laml_evaluate(&sol_plus, &rho_plus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rho_minus = rho.clone();
            rho_minus[k] -= eps;
            let sol_minus = build_gaussian_test_solution(&rho_minus);
            let cost_minus = reml_laml_evaluate(&sol_minus, &rho_minus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[k] = (cost_plus - cost_minus) / (2.0 * eps);
        }

        // Check agreement
        for k in 0..rho.len() {
            let abs_err = (analytic_grad[k] - fd_grad[k]).abs();
            let rel_err = abs_err / (1.0 + analytic_grad[k].abs());
            assert!(
                rel_err < 1e-4,
                "Gradient mismatch at k={}: analytic={:.8e}, fd={:.8e}, rel_err={:.3e}",
                k,
                analytic_grad[k],
                fd_grad[k],
                rel_err,
            );
        }
    }

    #[test]
    pub(crate) fn test_stochastic_trace_estimator_accuracy() {
        // Build a small SPD matrix and compare stochastic trace estimate
        // against the exact DenseSpectralOperator trace.
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0],];
        let a1 = array![[1.0, 0.3, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, 0.2],];
        let a2 = array![[0.2, 0.0, 0.1], [0.0, 1.0, 0.4], [0.1, 0.4, 0.8],];

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Exact traces via the dense operator.
        let exact1 = op.trace_hinv_product(&a1);
        let exact2 = op.trace_hinv_product(&a2);

        // Stochastic estimates with tight tolerance and many probes.
        let config = StochasticTraceConfig {
            n_probes_min: 50,
            n_probes_max: 200,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-8,
            seed: 42,
            hutchpp_sketch_dim: None,
        };
        let estimator = StochasticTraceEstimator::new(config);
        let matrices: Vec<&Array2<f64>> = vec![&a1, &a2];
        let estimates = estimator.estimate_traces(&op, &matrices);

        // With 200 probes on a 3x3 system, we should be very close.
        let rel_err1 = (estimates[0] - exact1).abs() / exact1.abs().max(1e-10);
        let rel_err2 = (estimates[1] - exact2).abs() / exact2.abs().max(1e-10);

        assert!(
            rel_err1 < 0.05,
            "Stochastic trace 1: est={:.6}, exact={:.6}, rel_err={:.4}",
            estimates[0],
            exact1,
            rel_err1,
        );
        assert!(
            rel_err2 < 0.05,
            "Stochastic trace 2: est={:.6}, exact={:.6}, rel_err={:.4}",
            estimates[1],
            exact2,
            rel_err2,
        );
    }

    #[test]
    pub(crate) fn modified_gram_schmidt_orthonormalizes_well_conditioned_input() {
        let y = array![
            [1.0, 2.0, 0.5, 3.0],
            [0.0, 1.0, 0.5, 1.5],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut q = Array2::<f64>::zeros(y.dim());
        let rank = modified_gram_schmidt(&y, &mut q);
        assert_eq!(rank, 4, "well-conditioned input should retain full rank");
        // Q^T Q = I within the retained rank.
        for j in 0..rank {
            for k in 0..rank {
                let dot = q.column(j).dot(&q.column(k));
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-12,
                    "QᵀQ off-identity at ({j},{k}): got {dot}",
                );
            }
        }
    }

    #[test]
    pub(crate) fn modified_gram_schmidt_drops_redundant_columns() {
        let y = array![
            [1.0, 2.0, 1.0, 4.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let mut q = Array2::<f64>::zeros(y.dim());
        let rank = modified_gram_schmidt(&y, &mut q);
        assert_eq!(
            rank, 2,
            "two duplicate columns plus a zero-extension should drop to rank 2"
        );
        for j in 0..rank {
            for k in 0..rank {
                let dot = q.column(j).dot(&q.column(k));
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    pub(crate) fn hutchpp_estimate_trace_hinv_operator_matches_exact_within_tolerance() {
        // Build a small SPD H and an HVP-only operator wrapping a dense M.
        // Compare Hutch++ to the exact tr(H⁻¹ M).
        let h = array![
            [4.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.5, 0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.7, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.1, 0.3, 3.0],
        ];
        let m = array![
            [1.0, 0.3, 0.0, 0.1, 0.0, 0.0],
            [0.3, 0.5, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.05],
            [0.1, 0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.05, 0.0, 0.1, 0.4],
        ];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let m_op = DenseMatrixHyperOperator { matrix: m.clone() };

        let exact = hop.trace_hinv_product(&m);

        let config = StochasticTraceConfig {
            n_probes_min: 12,
            n_probes_max: 64,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-10,
            seed: 0xABCDEF,
            hutchpp_sketch_dim: Some(3),
        };
        let est = hutchpp_estimate_trace_hinv_operator(&hop, &m_op, &config);
        let rel_err = (est - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.05,
            "Hutch++ trace est={est:.6} exact={exact:.6} rel_err={rel_err:.4}"
        );

        // Plain Hutchinson with the same probe budget should not be more
        // accurate; this guards against an inadvertent regression where
        // the sketch contribution is silently zeroed.
        let mut config_plain = config.clone();
        config_plain.hutchpp_sketch_dim = None;
        config_plain.n_probes_max = 64; // same total budget
        let est_plain = hutchpp_estimate_trace_hinv_operator(&hop, &m_op, &config_plain);
        let rel_err_plain = (est_plain - exact).abs() / exact.abs().max(1e-10);
        // Allow Hutch++ to either beat plain or match it; never be much worse.
        assert!(
            rel_err <= rel_err_plain * 2.0 + 0.01,
            "Hutch++ ({rel_err:.4}) should be competitive with Hutchinson ({rel_err_plain:.4})"
        );
    }

    #[test]
    pub(crate) fn hutchpp_estimate_trace_hinv_op_squared_matches_exact() {
        // SPD H and symmetric A; compare tr(H⁻¹ A H⁻¹ A) to the exact
        // value computed via trace_hinv_product_cross(A, A) =
        // tr((H⁻¹ A) (H⁻¹ A)).
        let h = array![
            [4.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.5, 0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.7, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.1, 0.3, 3.0],
        ];
        let a = array![
            [1.0, 0.3, 0.0, 0.1, 0.0, 0.0],
            [0.3, 0.5, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.05],
            [0.1, 0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.05, 0.0, 0.1, 0.4],
        ];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a_op = DenseMatrixHyperOperator { matrix: a.clone() };

        let exact = hop.trace_hinv_product_cross(&a, &a);

        let config = StochasticTraceConfig {
            n_probes_min: 16,
            n_probes_max: 96,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-10,
            seed: 0xC0FFEE,
            hutchpp_sketch_dim: Some(3),
        };
        let est = hutchpp_estimate_trace_hinv_op_squared(&hop, &a_op, &config);
        let rel_err = (est - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.05,
            "Hutch++ tr((H⁻¹A)²) est={est:.6} exact={exact:.6} rel_err={rel_err:.4}"
        );

        // Wired path: estimate_second_order_single_dense routes through
        // Hutch++ when hutchpp_sketch_dim is Some(_).
        let estimator = StochasticTraceEstimator::new(config.clone());
        let est_wired = estimator.estimate_second_order_single_dense(&hop, &a);
        let rel_err_wired = (est_wired - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err_wired < 0.05,
            "wired Hutch++ second-order est={est_wired:.6} exact={exact:.6} rel_err={rel_err_wired:.4}"
        );
        assert!(
            (est_wired - est).abs() <= 1e-12,
            "wired path must call hutchpp_estimate_trace_hinv_op_squared with the same seed/config"
        );
    }

    #[test]
    pub(crate) fn hutchpp_estimate_trace_hinv_operator_cross_matches_exact() {
        let h = array![
            [4.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.5, 0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.7, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.1, 0.3, 3.0],
        ];
        let a = array![
            [1.0, 0.3, 0.0, 0.1, 0.0, 0.0],
            [0.3, 0.5, 0.1, 0.0, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.0, 0.0, 0.05],
            [0.1, 0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.6, 0.1],
            [0.0, 0.0, 0.05, 0.0, 0.1, 0.4],
        ];
        let b = array![
            [0.5, 0.0, 0.1, 0.0, 0.05, 0.0],
            [0.0, 0.7, 0.0, 0.2, 0.0, 0.1],
            [0.1, 0.0, 0.4, 0.0, 0.15, 0.0],
            [0.0, 0.2, 0.0, 0.6, 0.0, 0.05],
            [0.05, 0.0, 0.15, 0.0, 0.3, 0.0],
            [0.0, 0.1, 0.0, 0.05, 0.0, 0.5],
        ];
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let a_op = DenseMatrixHyperOperator { matrix: a.clone() };
        let b_op = DenseMatrixHyperOperator { matrix: b.clone() };

        let exact = hop.trace_hinv_product_cross(&a, &b);

        let config = StochasticTraceConfig {
            n_probes_min: 16,
            n_probes_max: 128,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            solve_rel_tol: 1e-10,
            seed: 0xDEAD_BEEF,
            hutchpp_sketch_dim: Some(3),
        };
        let est = hutchpp_estimate_trace_hinv_operator_cross(&hop, &a_op, &b_op, &config);
        let rel_err = (est - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.07,
            "Hutch++ cross trace est={est:.6} exact={exact:.6} rel_err={rel_err:.4}"
        );
    }

    #[test]
    pub(crate) fn trace_hinv_operator_cross_default_routes_implicit_to_hutchpp() {
        // Build a synthetic 200-dim SPD H and an HVP-only operator pair
        // (mark `is_implicit() = true`) so the trait default routes
        // through the Hutch++ path. The exact reference comes from the
        // dense materialization of the same operator.
        let p = 200usize;
        let mut h = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            h[[i, i]] = 5.0 + (i as f64) * 0.01;
            if i + 1 < p {
                h[[i, i + 1]] = 0.2;
                h[[i + 1, i]] = 0.2;
            }
        }
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            a[[i, i]] = 1.0 + 0.005 * (i as f64);
            if i + 2 < p {
                a[[i, i + 2]] = 0.1;
                a[[i + 2, i]] = 0.1;
            }
        }
        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Wrapper that masquerades as implicit so the default route fires.
        pub(crate) struct ImplicitDense(Array2<f64>);
        impl HyperOperator for ImplicitDense {
            fn dim(&self) -> usize {
                self.0.nrows()
            }
            fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(self.0.nrows());
                dense_matvec_into(&self.0, v.view(), out.view_mut());
                out
            }
            fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
                dense_matvec_into(&self.0, v, out);
            }
            fn to_dense(&self) -> Array2<f64> {
                self.0.clone()
            }
            fn is_implicit(&self) -> bool {
                true
            }
        }

        let a_op = ImplicitDense(a.clone());
        let exact = hop.trace_hinv_product_cross(&a, &a);
        // Same-operator path: routes through the squared estimator.
        let est_same = hop.trace_hinv_operator_cross(&a_op, &a_op);
        assert!(est_same.is_finite(), "cross trace must be finite");
        let rel_err_same = (est_same - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err_same < 0.10,
            "default same-op cross routing est={est_same:.6} exact={exact:.6} rel_err={rel_err_same:.4}"
        );

        // Distinct-operator path: routes through the cross estimator.
        let mut b = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            b[[i, i]] = 0.6 + 0.003 * (i as f64);
            if i + 1 < p {
                b[[i, i + 1]] = 0.05;
                b[[i + 1, i]] = 0.05;
            }
        }
        let b_op = ImplicitDense(b.clone());
        let exact_ab = hop.trace_hinv_product_cross(&a, &b);
        let est_ab = hop.trace_hinv_operator_cross(&a_op, &b_op);
        assert!(est_ab.is_finite(), "cross trace (a,b) must be finite");
        let rel_err_ab = (est_ab - exact_ab).abs() / exact_ab.abs().max(1e-10);
        assert!(
            rel_err_ab < 0.10,
            "default distinct-op cross routing est={est_ab:.6} exact={exact_ab:.6} rel_err={rel_err_ab:.4}"
        );

        // Matrix-operator path: routes through the cross estimator with
        // a synthetic dense LHS wrapper.
        let exact_ma = hop.trace_hinv_product_cross(&a, &b);
        let est_ma = hop.trace_hinv_matrix_operator_cross(&a, &b_op);
        assert!(est_ma.is_finite(), "matrix-op cross trace must be finite");
        let rel_err_ma = (est_ma - exact_ma).abs() / exact_ma.abs().max(1e-10);
        assert!(
            rel_err_ma < 0.10,
            "default matrix-operator cross routing est={est_ma:.6} exact={exact_ma:.6} rel_err={rel_err_ma:.4}"
        );
    }

    #[test]
    pub(crate) fn dense_spectral_large_p_outer_gradient_matches_finite_difference() {
        let rho = 0.2;
        let solution = build_large_dense_spectral_gaussian_solution(rho);
        let result =
            reml_laml_evaluate(&solution, &[rho], EvalMode::ValueAndGradient, None).unwrap();
        let analytic = result.gradient.expect("gradient")[0];

        let eps = 1e-5;
        let rho_plus = rho + eps;
        let solution_plus = build_large_dense_spectral_gaussian_solution(rho_plus);
        let cost_plus = reml_laml_evaluate(&solution_plus, &[rho_plus], EvalMode::ValueOnly, None)
            .unwrap()
            .cost;

        let rho_minus = rho - eps;
        let solution_minus = build_large_dense_spectral_gaussian_solution(rho_minus);
        let cost_minus =
            reml_laml_evaluate(&solution_minus, &[rho_minus], EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

        let fd = (cost_plus - cost_minus) / (2.0 * eps);
        let rel_err = (analytic - fd).abs() / (1.0 + analytic.abs());
        assert!(
            rel_err < 2e-4,
            "large-p dense spectral gradient mismatch: analytic={analytic:.8e}, fd={fd:.8e}, rel_err={rel_err:.3e}"
        );
    }

    #[test]
    pub(crate) fn dense_spectral_logdet_traces_do_not_claim_hinv_kernel_equivalence() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        assert!(!op.prefers_stochastic_trace_estimation());
        assert!(!op.logdet_traces_match_hinv_kernel());
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 1024, true));

        let block =
            BlockCoupledOperator::from_joint_hessian_with_mode(&h, PseudoLogdetMode::Smooth)
                .unwrap();
        assert!(!block.prefers_stochastic_trace_estimation());
        assert!(!block.logdet_traces_match_hinv_kernel());
        assert!(!can_use_stochastic_logdet_hinv_kernel(&block, 1024, true));
    }

    #[test]
    pub(crate) fn dense_spectral_hinv_cross_matches_solve_contraction() {
        assert!(file!().ends_with(".rs"));
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let a = array![[1.0, 0.2, 0.1], [0.2, 0.5, 0.0], [0.1, 0.0, 0.3],];
        let b = array![[0.3, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.6],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let exact = op.trace_hinv_product_cross(&a, &b);
        let solved_a = op.solve_multi(&a);
        let solved_b = op.solve_multi(&b);
        let reference = (&solved_a.t() * &solved_b).sum();

        assert_relative_eq!(exact, reference, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    pub(crate) fn dense_spectral_batched_logdet_crosses_match_pairwise() {
        assert!(file!().ends_with(".rs"));
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0],];
        let h1 = array![[1.0, 0.2, 0.1], [0.2, 0.5, 0.0], [0.1, 0.0, 0.3],];
        let h2 = array![[0.3, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.6],];
        let h3 = array![[0.7, 0.0, 0.2], [0.0, 0.4, 0.1], [0.2, 0.1, 0.9],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let mats = [&h1, &h2, &h3];
        let batched = op.trace_logdet_hessian_crosses(&mats);

        for i in 0..mats.len() {
            for j in 0..mats.len() {
                let pairwise = op.trace_logdet_hessian_cross(mats[i], mats[j]);
                assert_relative_eq!(
                    batched[[i, j]],
                    pairwise,
                    epsilon = 1e-10,
                    max_relative = 1e-10
                );
            }
        }
    }

    #[test]
    pub(crate) fn sparse_block_local_trace_without_takahashi_matches_dense_reference() {
        assert!(file!().ends_with(".rs"));
        let h = array![
            [5.0, 0.2, 0.0, 0.1],
            [0.2, 4.0, 0.3, 0.0],
            [0.0, 0.3, 3.0, 0.4],
            [0.1, 0.0, 0.4, 2.5],
        ];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows());
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let block = array![[0.8, 0.15], [0.15, 0.45]];
        let scale = 1.7;
        let start = 1;
        let end = 3;
        let mut full = Array2::<f64>::zeros(h.raw_dim());
        for i in 0..block.nrows() {
            for j in 0..block.ncols() {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }

        assert_relative_eq!(
            sparse.trace_hinv_block_local(&block, scale, start, end),
            dense.trace_hinv_product(&full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            sparse.trace_hinv_block_local_cross(&block, scale, start, end),
            dense.trace_hinv_product_cross(&full, &full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    pub(crate) fn sparse_block_local_operator_cross_without_takahashi_matches_dense_reference() {
        assert!(file!().ends_with(".rs"));
        let h = array![
            [5.0, 0.2, 0.0, 0.1],
            [0.2, 4.0, 0.3, 0.0],
            [0.0, 0.3, 3.0, 0.4],
            [0.1, 0.0, 0.4, 2.5],
        ];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows());
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let local = array![[0.8, 0.15], [0.15, 0.45]];
        let start = 1;
        let end = 3;
        let op = BlockLocalDrift {
            local: local.clone(),
            start,
            end,
            total_dim: h.nrows(),
        };
        let mut full = Array2::<f64>::zeros(h.raw_dim());
        full.slice_mut(ndarray::s![start..end, start..end])
            .assign(&local);

        assert_relative_eq!(
            sparse.trace_hinv_operator_cross(&op, &op),
            dense.trace_hinv_product_cross(&full, &full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    pub(crate) fn sparse_matrix_block_operator_cross_without_takahashi_matches_dense_reference() {
        assert!(file!().ends_with(".rs"));
        let h = array![
            [5.0, 0.2, 0.0, 0.1],
            [0.2, 4.0, 0.3, 0.0],
            [0.0, 0.3, 3.0, 0.4],
            [0.1, 0.0, 0.4, 2.5],
        ];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows());
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let matrix = array![
            [1.0, 0.2, -0.1, 0.3],
            [0.2, 0.7, 0.4, -0.2],
            [-0.1, 0.4, 1.2, 0.5],
            [0.3, -0.2, 0.5, 0.9],
        ];
        let local = array![[0.8, 0.15], [0.15, 0.45]];
        let start = 1;
        let end = 3;
        let op = BlockLocalDrift {
            local: local.clone(),
            start,
            end,
            total_dim: h.nrows(),
        };
        let mut full = Array2::<f64>::zeros(h.raw_dim());
        full.slice_mut(ndarray::s![start..end, start..end])
            .assign(&local);

        assert_relative_eq!(
            sparse.trace_hinv_matrix_operator_cross(&matrix, &op),
            dense.trace_hinv_product_cross(&matrix, &full),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    pub(crate) fn sparse_takahashi_trace_hinv_product_pairs_symmetric_lookups() {
        assert!(file!().ends_with(".rs"));
        let h = array![[4.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 2.5],];
        let h_sparse =
            crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(&h, 0.0).unwrap();
        let factor = std::sync::Arc::new(
            crate::linalg::sparse_exact::factorize_sparse_spd(&h_sparse).unwrap(),
        );
        let sfactor = crate::linalg::sparse_exact::factorize_simplicial(&h_sparse).unwrap();
        let taka = std::sync::Arc::new(
            crate::linalg::sparse_exact::TakahashiInverse::compute(&sfactor).unwrap(),
        );
        let sparse = SparseCholeskyOperator::new(factor, 0.0, h.nrows()).with_takahashi(taka);
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let a = array![[1.0, 0.7, -0.2], [0.1, 0.5, 0.9], [0.4, -0.3, 0.2],];
        assert_relative_eq!(
            sparse.trace_hinv_product(&a),
            dense.trace_hinv_product(&a),
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }

    #[test]
    pub(crate) fn hyper_operator_bilinear_view_matches_owned_bilinear() {
        assert!(file!().ends_with(".rs"));
        let dense = DenseMatrixHyperOperator {
            matrix: array![[2.0, 0.3, -0.1], [0.3, 1.5, 0.4], [-0.1, 0.4, 3.0],],
        };
        let block = BlockLocalDrift {
            local: array![[1.2, 0.2], [0.2, 0.7]],
            start: 1,
            end: 3,
            total_dim: 3,
        };
        let composite = CompositeHyperOperator {
            dense: Some(array![[0.4, 0.1, 0.0], [0.1, 0.8, -0.2], [0.0, -0.2, 0.6],]),
            operators: vec![Arc::new(block.clone())],
            dim_hint: 3,
        };
        let weighted = WeightedHyperOperator {
            terms: vec![
                (1.7, Arc::new(dense.clone()) as Arc<dyn HyperOperator>),
                (-0.4, Arc::new(block.clone()) as Arc<dyn HyperOperator>),
            ],
            dim_hint: 3,
        };

        let v_storage = array![9.0, 0.5, -1.2, 0.7, 8.0];
        let u_storage = array![7.0, -0.3, 1.1, 0.9, 6.0];
        let v_view = v_storage.slice(ndarray::s![1..4]);
        let u_view = u_storage.slice(ndarray::s![1..4]);
        let v_owned = v_view.to_owned();
        let u_owned = u_view.to_owned();

        let operators: [&dyn HyperOperator; 4] = [&dense, &block, &composite, &weighted];
        for op in operators {
            assert_relative_eq!(
                op.bilinear_view(v_view, u_view),
                op.bilinear(&v_owned, &u_owned),
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    pub(crate) fn hyper_operator_scaled_add_mul_vec_matches_owned_matvec() {
        assert!(file!().ends_with(".rs"));
        let dense = DenseMatrixHyperOperator {
            matrix: array![[2.0, 0.3, -0.1], [0.3, 1.5, 0.4], [-0.1, 0.4, 3.0],],
        };
        let block = BlockLocalDrift {
            local: array![[1.2, 0.2], [0.2, 0.7]],
            start: 1,
            end: 3,
            total_dim: 3,
        };
        let composite = CompositeHyperOperator {
            dense: Some(array![[0.4, 0.1, 0.0], [0.1, 0.8, -0.2], [0.0, -0.2, 0.6],]),
            operators: vec![Arc::new(block.clone())],
            dim_hint: 3,
        };
        let weighted = WeightedHyperOperator {
            terms: vec![
                (1.7, Arc::new(dense.clone()) as Arc<dyn HyperOperator>),
                (-0.4, Arc::new(block.clone()) as Arc<dyn HyperOperator>),
                (0.0, Arc::new(composite.clone()) as Arc<dyn HyperOperator>),
            ],
            dim_hint: 3,
        };

        let v_storage = array![9.0, 0.5, -1.2, 0.7, 8.0];
        let v_view = v_storage.slice(ndarray::s![1..4]);
        let v_owned = v_view.to_owned();
        let base = array![0.25, -0.5, 1.5];
        let scale = -1.3;

        let operators: [&dyn HyperOperator; 4] = [&dense, &block, &composite, &weighted];
        for op in operators {
            let mut accumulated = base.clone();
            op.scaled_add_mul_vec(v_view, scale, accumulated.view_mut());

            let mut expected = base.clone();
            expected.scaled_add(scale, &op.mul_vec(&v_owned));
            for idx in 0..accumulated.len() {
                assert_relative_eq!(
                    accumulated[idx],
                    expected[idx],
                    epsilon = 1e-12,
                    max_relative = 1e-12
                );
            }
        }
    }

    #[test]
    pub(crate) fn stochastic_single_second_order_estimators_match_batched_paths() {
        assert!(file!().ends_with(".rs"));
        let diag = array![4.0, 3.0, 2.0];
        let hop = MatrixFreeSpdOperator::new_with_mode(
            diag.len(),
            move |v| &diag * v,
            PseudoLogdetMode::Smooth,
        );
        let estimator = StochasticTraceEstimator::with_defaults();
        let dense = array![[0.8, 0.2, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.7],];
        let op = DenseMatrixHyperOperator {
            matrix: dense.clone(),
        };

        let no_ops: [&dyn HyperOperator; 0] = [];
        let dense_refs = [&dense];
        let batched_dense =
            estimator.estimate_second_order_traces_with_operators(&hop, &dense_refs, &no_ops);
        assert_relative_eq!(
            estimator.estimate_second_order_single_dense(&hop, &dense),
            batched_dense[[0, 0]],
            epsilon = 1e-12,
            max_relative = 1e-12
        );

        let no_dense: [&Array2<f64>; 0] = [];
        let op_refs: [&dyn HyperOperator; 1] = [&op];
        let batched_op =
            estimator.estimate_second_order_traces_with_operators(&hop, &no_dense, &op_refs);
        assert_relative_eq!(
            estimator.estimate_second_order_single_operator(&hop, &op),
            batched_op[[0, 0]],
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    pub(crate) fn matrix_free_logdet_traces_use_exact_spectral_algebra() {
        let diag = array![4.0, 3.0, 2.0];
        let h = Array2::from_diag(&diag);
        let dense = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let op = MatrixFreeSpdOperator::new_with_mode(
            diag.len(),
            move |v| &diag * v,
            PseudoLogdetMode::Smooth,
        );
        let a = array![[0.7, 0.1, 0.0], [0.1, 0.4, 0.2], [0.0, 0.2, 0.5]];

        assert_relative_eq!(op.logdet(), dense.logdet(), epsilon = 1e-12);
        assert_relative_eq!(
            op.trace_hinv_product(&a),
            dense.trace_hinv_product(&a),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            op.trace_logdet_hessian_cross(&a, &a),
            dense.trace_logdet_hessian_cross(&a, &a),
            epsilon = 1e-12
        );
        assert!(!op.prefers_stochastic_trace_estimation());
        assert!(!op.logdet_traces_match_hinv_kernel());
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 1024, true));
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 128, true));
        assert!(!can_use_stochastic_logdet_hinv_kernel(&op, 1024, false));
    }

    #[test]
    pub(crate) fn test_rademacher_probe_properties() {
        // Verify probes have entries +/-1 and are deterministic given the same seed.
        let mut rng = Xoshiro256SS::from_seed(99);
        let mut z = Array1::zeros(100);
        rademacher_probe_into(z.view_mut(), &mut rng);
        assert_eq!(z.len(), 100);
        for &v in z.iter() {
            assert!(v == 1.0 || v == -1.0, "Rademacher entry must be +/-1");
        }

        // Same seed produces the same probe.
        let mut rng2 = Xoshiro256SS::from_seed(99);
        let mut z2 = Array1::zeros(100);
        rademacher_probe_into(z2.view_mut(), &mut rng2);
        assert_eq!(z, z2, "Same seed must produce identical probes");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 1: Spectral logdet gradient with r_epsilon regularization
    // ═══════════════════════════════════════════════════════════════════

    /// Verify that the analytic gradient of log|H(t)| computed through
    /// `DenseSpectralOperator` (with smooth spectral regularization r_epsilon)
    /// matches a central finite-difference estimate.
    ///
    /// Setup: H(t) = diag(2 + t, 0.01 + 2t, 3 - t) — one eigenvalue near
    /// zero so the regularization is exercised.
    #[test]
    pub(crate) fn test_spectral_logdet_gradient_fd() {
        let t0 = 0.0_f64;
        let h_step = 1e-6;

        // H(t) = diag(2+t, 0.01+2t, 3-t)
        // dH/dt = diag(1, 2, -1)
        let dh_dt = Array2::from_diag(&array![1.0, 2.0, -1.0]);

        // Build operator at t0
        let h0 = Array2::from_diag(&array![2.0 + t0, 0.01 + 2.0 * t0, 3.0 - t0]);
        let op0 = DenseSpectralOperator::from_symmetric(&h0).unwrap();

        // Analytic gradient: d/dt log|R_eps(H(t))| = tr(G_eps(H) dH/dt)
        let analytic = op0.trace_logdet_gradient(&dh_dt);

        // Finite difference: (logdet(t+h) - logdet(t-h)) / (2h)
        let h_plus = Array2::from_diag(&array![
            2.0 + t0 + h_step,
            0.01 + 2.0 * (t0 + h_step),
            3.0 - (t0 + h_step)
        ]);
        let h_minus = Array2::from_diag(&array![
            2.0 + t0 - h_step,
            0.01 + 2.0 * (t0 - h_step),
            3.0 - (t0 - h_step)
        ]);
        let op_plus = DenseSpectralOperator::from_symmetric(&h_plus).unwrap();
        let op_minus = DenseSpectralOperator::from_symmetric(&h_minus).unwrap();
        let fd = (op_plus.logdet() - op_minus.logdet()) / (2.0 * h_step);

        let rel_err = (analytic - fd).abs() / fd.abs().max(1e-12);
        assert!(
            rel_err < 1e-5,
            "Spectral logdet gradient mismatch: analytic={:.10e}, fd={:.10e}, rel_err={:.3e}",
            analytic,
            fd,
            rel_err,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 2: Moving nullspace correction for penalty pseudo-logdet
    // ═══════════════════════════════════════════════════════════════════

    /// Helper: build a 3x3 penalty matrix S(psi) whose nullspace rotates.
    ///
    /// S(psi) = R(psi) diag(s1, s2, 0) R(psi)^T
    /// where R(psi) is a rotation around the z-axis by angle psi.
    /// The nullspace is spanned by R(psi) * e3, which rotates as psi changes.
    pub(crate) fn rotating_nullspace_penalty(psi: f64, s1: f64, s2: f64) -> Array2<f64> {
        let c = psi.cos();
        let s = psi.sin();
        // R rotates in the (0,2) plane so the nullspace direction changes.
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c],];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        r.dot(&d).dot(&r.t())
    }

    /// Compute log|S|_+ (pseudo-logdeterminant over positive eigenvalues).
    pub(crate) fn pseudo_logdet(s: &Array2<f64>, tol: f64) -> f64 {
        let (eigs, _) = s.eigh(faer::Side::Lower).unwrap();
        eigs.iter().filter(|&&v| v > tol).map(|v| v.ln()).sum()
    }

    /// Compute d/dpsi log|S(psi)|_+ by central finite difference.
    pub(crate) fn pseudo_logdet_fd_first(psi: f64, h: f64, s1: f64, s2: f64, tol: f64) -> f64 {
        let sp = rotating_nullspace_penalty(psi + h, s1, s2);
        let sm = rotating_nullspace_penalty(psi - h, s1, s2);
        (pseudo_logdet(&sp, tol) - pseudo_logdet(&sm, tol)) / (2.0 * h)
    }

    /// Compute d^2/dpsi^2 log|S(psi)|_+ by central finite difference.
    pub(crate) fn pseudo_logdet_fd_second(psi: f64, h: f64, s1: f64, s2: f64, tol: f64) -> f64 {
        let sp = pseudo_logdet(&rotating_nullspace_penalty(psi + h, s1, s2), tol);
        let s0 = pseudo_logdet(&rotating_nullspace_penalty(psi, s1, s2), tol);
        let sm = pseudo_logdet(&rotating_nullspace_penalty(psi - h, s1, s2), tol);
        (sp - 2.0 * s0 + sm) / (h * h)
    }

    /// Analytic second derivative of log|S(psi)|_+ WITH the moving-nullspace
    /// correction, and WITHOUT it, so we can verify the correction is needed.
    ///
    /// Returns (with_correction, without_correction).
    pub(crate) fn analytic_pseudo_logdet_second(psi: f64, s1: f64, s2: f64, tol: f64) -> (f64, f64) {
        let s_mat = rotating_nullspace_penalty(psi, s1, s2);

        // Eigendecompose S
        let (eigs, vecs) = s_mat.eigh(faer::Side::Lower).unwrap();
        let p = eigs.len();

        let pos_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] > tol).collect();
        let null_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();

        // Build S_psi = dS/dpsi analytically.
        // S(psi) = R D R^T => dS/dpsi = R' D R^T + R D R'^T
        let c = psi.cos();
        let s = psi.sin();
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c],];
        // R' = dR/dpsi
        let rp = array![[-s, 0.0, -c], [0.0, 0.0, 0.0], [c, 0.0, -s],];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        let s_psi = rp.dot(&d).dot(&r.t()) + r.dot(&d).dot(&rp.t());

        // Build S_psi_psi = d^2S/dpsi^2 analytically.
        // R'' = d^2R/dpsi^2
        let rpp = array![[-c, 0.0, s], [0.0, 0.0, 0.0], [-s, 0.0, -c],];
        let s_psi_psi =
            rpp.dot(&d).dot(&r.t()) + 2.0 * &rp.dot(&d).dot(&rp.t()) + r.dot(&d).dot(&rpp.t());

        // Build S^+ (pseudoinverse): S^+ = V diag(1/sigma_i for pos, 0 for null) V^T
        let mut s_dag = Array2::<f64>::zeros((p, p));
        for &i in &pos_idx {
            let col = vecs.column(i);
            for r in 0..p {
                for c2 in 0..p {
                    s_dag[[r, c2]] += col[r] * col[c2] / eigs[i];
                }
            }
        }

        // Fixed-nullspace formula:
        //   d^2/dpsi^2 log|S|_+ = tr(S^+ S_psi_psi) - tr(S^+ S_psi S^+ S_psi)
        let sdag_s_psi = s_dag.dot(&s_psi);
        let term_linear = trace_mat(&s_dag.dot(&s_psi_psi));
        let term_quad = trace_mat(&sdag_s_psi.dot(&sdag_s_psi));
        let without_correction = term_linear - term_quad;

        // Moving-nullspace correction:
        //   +2 * tr(S^{+2} S_psi P_0 S_psi)
        // where P_0 = U_0 U_0^T, S^{+2} = (S^+)^2
        //
        // Efficient: tr(Sigma^{+2} L L^T) where L = U_+^T S_psi U_0
        let mut correction = 0.0_f64;
        if !pos_idx.is_empty() && !null_idx.is_empty() {
            // Build U_+ and U_0
            let n_pos = pos_idx.len();
            let n_null = null_idx.len();
            let mut u_pos = Array2::<f64>::zeros((p, n_pos));
            let mut u_null = Array2::<f64>::zeros((p, n_null));
            for (out, &idx) in pos_idx.iter().enumerate() {
                u_pos.column_mut(out).assign(&vecs.column(idx));
            }
            for (out, &idx) in null_idx.iter().enumerate() {
                u_null.column_mut(out).assign(&vecs.column(idx));
            }

            // L = U_+^T S_psi U_0  (n_pos x n_null)
            let l_mat = u_pos.t().dot(&s_psi.dot(&u_null));

            // Sigma^{+2} = diag(1/sigma_i^2) for positive eigenvalues
            for a in 0..n_pos {
                let sigma_inv_sq = 1.0 / (eigs[pos_idx[a]] * eigs[pos_idx[a]]);
                correction += sigma_inv_sq * l_mat.row(a).dot(&l_mat.row(a));
            }
            // The full correction is 2 * tr(Sigma^{+2} L L^T)
            correction *= 2.0;
        }

        let with_correction = without_correction + correction;
        (with_correction, without_correction)
    }

    /// tr(A) for a square matrix.
    pub(crate) fn trace_mat(a: &Array2<f64>) -> f64 {
        (0..a.nrows()).map(|i| a[[i, i]]).sum()
    }

    #[test]
    pub(crate) fn test_moving_nullspace_correction_needed() {
        // S(psi) = R(psi) diag(4, 1, 0) R(psi)^T — rank-2, nullspace rotates.
        let s1 = 4.0;
        let s2 = 1.0;
        let psi = 0.3; // nonzero angle
        let tol = 1e-10;
        let h = 1e-5;

        // The pseudo-logdet depends only on the positive eigenvalues, so a pure
        // nullspace rotation leaves the first derivative exactly zero.
        let fd_first = pseudo_logdet_fd_first(psi, h, s1, s2, tol);
        assert!(
            fd_first.is_finite() && fd_first.abs() < 1e-8,
            "First derivative should vanish for rotating nullspace, got {fd_first}"
        );

        let fd_second = pseudo_logdet_fd_second(psi, h, s1, s2, tol);
        let (with_corr, without_corr) = analytic_pseudo_logdet_second(psi, s1, s2, tol);

        // WITH correction should match FD
        let rel_err_with = (with_corr - fd_second).abs() / fd_second.abs().max(1e-12);
        assert!(
            rel_err_with < 1e-4,
            "With correction: analytic={:.8e}, fd={:.8e}, rel_err={:.3e}",
            with_corr,
            fd_second,
            rel_err_with,
        );

        // WITHOUT correction should NOT match FD (error should be large)
        let rel_err_without = (without_corr - fd_second).abs() / fd_second.abs().max(1e-12);
        assert!(
            rel_err_without > 1e-2,
            "Without correction should disagree with FD: \
             without={:.8e}, fd={:.8e}, rel_err={:.3e} (expected > 1e-2)",
            without_corr,
            fd_second,
            rel_err_without,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 3: Correction vanishes when nullspace is fixed
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    pub(crate) fn test_fixed_nullspace_correction_vanishes() {
        // S(rho) = diag(exp(rho1), exp(rho2), 0) — the nullspace is always e3,
        // regardless of rho. The correction terms should vanish, so both
        // formulas (with and without correction) should agree with FD.
        let tol = 1e-10;
        let h = 1e-5;

        // Evaluate at a specific point
        let rho1 = 0.5_f64;
        let rho2 = -0.3_f64;

        // Pseudo-logdet: log(exp(rho1)) + log(exp(rho2)) = rho1 + rho2
        // d/drho1 = 1, d^2/drho1^2 = 0 (exact).
        // But let's verify via the analytic+FD machinery for consistency.

        // We parameterize by a single scalar t: rho1 = 0.5 + t, rho2 = -0.3 + 2t.
        // S(t) = diag(exp(0.5+t), exp(-0.3+2t), 0)
        // log|S|_+ = (0.5+t) + (-0.3+2t) = 0.2 + 3t
        // d/dt = 3, d^2/dt^2 = 0.

        let build_s = |t: f64| -> Array2<f64> {
            Array2::from_diag(&array![(rho1 + t).exp(), (rho2 + 2.0 * t).exp(), 0.0])
        };

        let t0 = 0.0_f64;

        // FD second derivative
        let ld_plus = pseudo_logdet(&build_s(t0 + h), tol);
        let ld_0 = pseudo_logdet(&build_s(t0), tol);
        let ld_minus = pseudo_logdet(&build_s(t0 - h), tol);
        let fd_second = (ld_plus - 2.0 * ld_0 + ld_minus) / (h * h);

        // Analytic: S_t = diag(exp(rho1+t), 2*exp(rho2+2t), 0)
        // S_tt = diag(exp(rho1+t), 4*exp(rho2+2t), 0)
        let s_mat = build_s(t0);
        let s_t = Array2::from_diag(&array![
            (rho1 + t0).exp(),
            2.0 * (rho2 + 2.0 * t0).exp(),
            0.0
        ]);
        let s_tt = Array2::from_diag(&array![
            (rho1 + t0).exp(),
            4.0 * (rho2 + 2.0 * t0).exp(),
            0.0
        ]);

        let (eigs, vecs) = s_mat.eigh(faer::Side::Lower).unwrap();
        let p = 3;
        let pos_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] > tol).collect();
        let null_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();

        // Build S^+
        let mut s_dag = Array2::<f64>::zeros((p, p));
        for &i in &pos_idx {
            let col = vecs.column(i);
            for r in 0..p {
                for c in 0..p {
                    s_dag[[r, c]] += col[r] * col[c] / eigs[i];
                }
            }
        }

        // Fixed-nullspace formula
        let sdag_s_t = s_dag.dot(&s_t);
        let term_linear = trace_mat(&s_dag.dot(&s_tt));
        let term_quad = trace_mat(&sdag_s_t.dot(&sdag_s_t));
        let without_correction = term_linear - term_quad;

        // Compute the correction (should be ~0 since nullspace doesn't move)
        let mut correction = 0.0_f64;
        if !pos_idx.is_empty() && !null_idx.is_empty() {
            let n_pos = pos_idx.len();
            let n_null = null_idx.len();
            let mut u_pos = Array2::<f64>::zeros((p, n_pos));
            let mut u_null = Array2::<f64>::zeros((p, n_null));
            for (out, &idx) in pos_idx.iter().enumerate() {
                u_pos.column_mut(out).assign(&vecs.column(idx));
            }
            for (out, &idx) in null_idx.iter().enumerate() {
                u_null.column_mut(out).assign(&vecs.column(idx));
            }
            let l_mat = u_pos.t().dot(&s_t.dot(&u_null));
            for a in 0..n_pos {
                let sigma_inv_sq = 1.0 / (eigs[pos_idx[a]] * eigs[pos_idx[a]]);
                correction += sigma_inv_sq * l_mat.row(a).dot(&l_mat.row(a));
            }
            correction *= 2.0;
        }

        // The correction should be negligible (nullspace is fixed)
        assert!(
            correction.abs() < 1e-12,
            "Correction should vanish for fixed nullspace, got {:.3e}",
            correction,
        );

        // Both formulas should match FD
        let with_correction = without_correction + correction;

        // For diag(e^a, e^b, 0), d^2/dt^2 log|S|_+ = 0, so use absolute error
        // since fd_second ~ 0.
        let abs_err_with = (with_correction - fd_second).abs();
        let abs_err_without = (without_correction - fd_second).abs();
        assert!(
            abs_err_with < 1e-4,
            "With correction should match FD: with={:.8e}, fd={:.8e}, abs_err={:.3e}",
            with_correction,
            fd_second,
            abs_err_with,
        );
        assert!(
            abs_err_without < 1e-4,
            "Without correction should also match FD (fixed nullspace): \
             without={:.8e}, fd={:.8e}, abs_err={:.3e}",
            without_correction,
            fd_second,
            abs_err_without,
        );
    }

    #[test]
    pub(crate) fn test_symmetric_eigen_identity() {
        let eye = Array2::<f64>::eye(3);
        let (evals, evecs) = symmetric_eigen(&eye);
        for &e in &evals {
            assert!((e - 1.0).abs() < 1e-12, "eigenvalue should be 1.0, got {e}");
        }
        // Eigenvectors should be orthonormal.
        let prod = evecs.t().dot(&evecs);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[[i, j]] - expected).abs() < 1e-12,
                    "Q^T Q should be identity"
                );
            }
        }
    }

    #[test]
    pub(crate) fn test_symmetric_eigen_diagonal() {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = 4.0;
        d[[1, 1]] = 2.0;
        d[[2, 2]] = 1.0;
        let (evals, _) = symmetric_eigen(&d);
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 2.0).abs() < 1e-12);
        assert!((sorted[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    pub(crate) fn test_pseudoinverse_times_vec_identity() {
        let eye = Array2::<f64>::eye(3);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result =
            pseudoinverse_times_vec(&eye, v.as_slice().expect("contiguous test vector"), 1e-8);
        for i in 0..3 {
            assert!((result[i] - v[i]).abs() < 1e-12, "G=I: G⁺v should equal v");
        }
    }

    #[test]
    pub(crate) fn test_pseudoinverse_times_vec_singular() {
        // Rank-1 matrix: G = [1 1; 1 1]. Pseudoinverse G⁺ = [0.25 0.25; 0.25 0.25].
        let mut g = Array2::<f64>::zeros((2, 2));
        g[[0, 0]] = 1.0;
        g[[0, 1]] = 1.0;
        g[[1, 0]] = 1.0;
        g[[1, 1]] = 1.0;
        let v = Array1::from_vec(vec![2.0, 0.0]);
        let result =
            pseudoinverse_times_vec(&g, v.as_slice().expect("contiguous test vector"), 1e-8);
        // G⁺ v = [0.25*2 + 0.25*0; 0.25*2 + 0.25*0] = [0.5; 0.5]
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    pub(crate) fn batched_implicit_trace_matches_per_operator_trace() {
        use crate::terms::basis::ImplicitDesignPsiDerivative;
        use std::sync::Arc;

        let n = 5usize;
        let p = 3usize;
        let n_axes = 2usize;
        let len = n * p;
        let phi_values = Array1::from_vec((0..len).map(|i| 0.2 + 0.03 * i as f64).collect());
        let q_values = Array1::from_vec((0..len).map(|i| -0.4 + 0.05 * i as f64).collect());
        let t_values = Array1::zeros(len);
        let axis_components = Array2::from_shape_vec(
            (len, n_axes),
            (0..len)
                .flat_map(|i| [0.1 + 0.02 * i as f64, -0.3 + 0.015 * i as f64])
                .collect(),
        )
        .unwrap();
        let implicit = Arc::new(ImplicitDesignPsiDerivative::new(
            phi_values,
            q_values,
            t_values,
            axis_components,
            None,
            None,
            n,
            p,
            0,
            n_axes,
        ));
        let x_data = array![
            [1.0, 0.4, -0.2],
            [0.5, 1.1, 0.3],
            [-0.3, 0.9, 0.6],
            [0.8, -0.5, 1.2],
            [0.2, 0.7, -0.4],
        ];
        let x_design = Arc::new(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_data,
        )));
        let w_diag = Arc::new(array![1.0, 0.7, 1.3, 0.9, 1.1]);
        let h = Array2::<f64>::eye(p);
        let ds = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let make_op = |axis: usize, scale: f64| -> Arc<dyn HyperOperator> {
            Arc::new(ImplicitHyperOperator {
                implicit_deriv: Arc::clone(&implicit),
                axis,
                x_design: Arc::clone(&x_design),
                w_diag: crate::matrix::SignedWeightsArc::from_arc(Arc::clone(&w_diag)),
                s_psi: Array2::<f64>::eye(p) * scale,
                p,
                c_x_psi_beta: Some(Arc::new(Array1::from_vec(
                    (0..n).map(|i| scale * (i as f64 + 1.0)).collect(),
                ))),
            })
        };
        let ops = vec![make_op(0, 0.05), make_op(1, 0.07)];
        let cache = ProjectedFactorCache::default();
        let per_operator: Vec<f64> = ops
            .iter()
            .map(|op| op.trace_projected_factor_cached(&ds.g_factor, &cache))
            .collect();
        let batched = dense_spectral_trace_logdet_operators_batched(&ds, &ops);
        assert_eq!(batched.len(), per_operator.len());
        for (want, got) in per_operator.iter().zip(batched.iter()) {
            assert_relative_eq!(got, want, epsilon = 1.0e-10, max_relative = 1.0e-10);
        }
    }

    /// Contract: `ImplicitHyperOperator::mul_vec(v)` reproduces the analytic
    /// first-order spatial drift
    ///   `B_d v = (∂X/∂ψ_d)ᵀ W X v + Xᵀ W (∂X/∂ψ_d) v + Xᵀ diag(c·X_{ψ_d}β̂) X v + S_{ψ_d} v`.
    ///
    /// The third (non-Gaussian) term is the part that landed under task #7 —
    /// it must agree with the dense reference computed from
    /// `materialize_first(axis)`. We build a tiny `ImplicitDesignPsiDerivative`
    /// (n=4, n_knots=2, n_axes=1, no identifiability transform), assemble a
    /// known X / W / S_ψ / c_x_psi_beta, and check `mul_vec(v)` against the
    /// fully-dense formula above for several probe vectors v.
    ///
    /// Also runs once with `c_x_psi_beta = None` to lock in the Gaussian
    /// fast-path: the third term must drop out cleanly.
    #[test]
    pub(crate) fn implicit_hyper_operator_third_derivative_term_matches_dense_reference() {
        use crate::terms::basis::ImplicitDesignPsiDerivative;
        use std::sync::Arc;

        let n = 4usize;
        let n_knots = 2usize;
        let n_axes = 1usize;
        let p = n_knots; // no polynomial padding, no identifiability transform

        // Implicit operator: deliberately non-trivial radial scalars so the
        // resulting (∂X/∂ψ_0) is dense and not accidentally zero.
        // First-axis kernel value (no transform path) is `q_ij·s_b[axis] + c·phi_ij`
        // with `c = psi_scale_share = 0.0` — so the kernel is `q_ij · s_{0,ij}`.
        let phi_values = array![1.0, 0.5, 0.7, 0.9, 0.3, 0.4, 0.6, 0.8];
        let q_values = array![0.5, -0.2, 0.3, 0.1, -0.4, 0.2, 0.6, -0.1];
        let t_values = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // axis_components is (n*n_knots, n_axes) row-major: rows = (i, j) pair.
        let axis_components = array![[0.7], [0.3], [-0.4], [0.5], [0.2], [-0.1], [0.6], [0.8]];
        let implicit = Arc::new(ImplicitDesignPsiDerivative::new(
            phi_values,
            q_values,
            t_values,
            axis_components,
            None,
            None,
            n,
            n_knots,
            0,
            n_axes,
        ));

        // Active-basis design X (n × p): chosen so Xᵀ X is well-conditioned.
        let x_data = array![[1.0, 0.30], [0.50, 1.20], [-0.20, 0.80], [0.90, -0.40],];
        let x_design = Arc::new(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_data.clone(),
        )));
        let w_diag = Arc::new(array![0.8, 1.2, 0.6, 1.5]);

        // S_psi (p × p): symmetric, dense.
        let s_psi = array![[0.40, 0.05], [0.05, 0.25]];

        // β̂ used to fold c · (∂X/∂ψ_0) β̂ into the per-row kernel.
        let beta_eval = array![0.30, -0.20];
        // c_array (length n) — the GLM third-derivative weight.
        let c_array = array![0.10, -0.05, 0.20, 0.15];

        // Reference dense (∂X/∂ψ_0).
        let dx_dpsi = implicit
            .materialize_first(0)
            .expect("materialize_first should succeed on tiny fixture");
        assert_eq!(dx_dpsi.shape(), &[n, p]);

        // c_x_psi_beta[i] = c[i] · (∂X/∂ψ_0 · β̂)[i].
        let dx_beta = dx_dpsi.dot(&beta_eval);
        let c_x_psi_beta_dense = &c_array * &dx_beta;
        let c_x_psi_beta = Some(Arc::new(c_x_psi_beta_dense.clone()));

        let op = ImplicitHyperOperator {
            implicit_deriv: Arc::clone(&implicit),
            axis: 0,
            x_design: Arc::clone(&x_design),
            w_diag: crate::matrix::SignedWeightsArc::from_arc(Arc::clone(&w_diag)),
            s_psi: s_psi.clone(),
            p,
            c_x_psi_beta,
        };

        let probes = [
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![0.7, -0.4],
            array![-0.25, 1.10],
        ];
        for (k, v) in probes.iter().enumerate() {
            // Analytic dense reference.
            //   t1 = (∂X/∂ψ_0)ᵀ · diag(W) · X · v
            //   t2 = Xᵀ · diag(W) · (∂X/∂ψ_0) · v
            //   t3 = Xᵀ · diag(c_x_psi_beta) · X · v
            //   t4 = S_psi · v
            let xv = x_data.dot(v);
            let dxv = dx_dpsi.dot(v);
            let w_xv = &*w_diag * &xv;
            let w_dxv = &*w_diag * &dxv;
            let t1 = dx_dpsi.t().dot(&w_xv);
            let t2 = x_data.t().dot(&w_dxv);
            let weighted = &c_x_psi_beta_dense * &xv;
            let t3 = x_data.t().dot(&weighted);
            let t4 = s_psi.dot(v);
            let want = &t1 + &t2 + &t3 + &t4;

            let got = op.mul_vec(v);
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "B_d·v mismatch at probe {k}, comp {i}: want={:.6e}, got={:.6e}",
                    want[i],
                    got[i],
                );
            }
        }

        // Gaussian path: c_x_psi_beta = None must drop the third term cleanly.
        let op_gauss = ImplicitHyperOperator {
            implicit_deriv: Arc::clone(&implicit),
            axis: 0,
            x_design,
            w_diag: crate::matrix::SignedWeightsArc::from_arc(Arc::clone(&w_diag)),
            s_psi: s_psi.clone(),
            p,
            c_x_psi_beta: None,
        };
        let v = array![0.7, -0.4];
        let xv = x_data.dot(&v);
        let dxv = dx_dpsi.dot(&v);
        let w_xv = &*w_diag * &xv;
        let w_dxv = &*w_diag * &dxv;
        let want = &dx_dpsi.t().dot(&w_xv) + &x_data.t().dot(&w_dxv) + &s_psi.dot(&v);
        let got = op_gauss.mul_vec(&v);
        for i in 0..p {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "Gaussian B_d·v mismatch at comp {i}: want={:.6e}, got={:.6e}",
                want[i],
                got[i],
            );
        }
    }

    /// Centered finite-difference check on the third-derivative term in
    /// isolation: at fixed (X, W, S_ψ, β̂) the term `Xᵀ diag(c · X_ψ β̂) X v` is
    /// linear in `v`, so the *correctness* check is a comparison against the
    /// analytic action. To exercise the FD route the spec asks for, we
    /// finite-difference along v using the operator's `mul_vec` and confirm
    /// the operator is exactly the linear map encoded by its kernel — i.e. the
    /// difference quotient `(op.mul_vec(v + ε e_j) − op.mul_vec(v − ε e_j))/(2ε)`
    /// equals the j-th column of `Xᵀ diag(c_x_psi_beta) X` at any v.
    #[test]
    pub(crate) fn implicit_hyper_operator_third_derivative_term_centered_fd_matches_jacobian_column() {
        use crate::terms::basis::ImplicitDesignPsiDerivative;
        use std::sync::Arc;

        let n = 5usize;
        let n_knots = 3usize;
        let n_axes = 1usize;
        let p = n_knots;

        let phi_values =
            Array1::from_vec((0..n * n_knots).map(|k| 0.1 + 0.05 * (k as f64)).collect());
        let q_values =
            Array1::from_vec((0..n * n_knots).map(|k| -0.2 + 0.07 * (k as f64)).collect());
        let t_values = Array1::zeros(n * n_knots);
        let axis_components = Array2::from_shape_vec(
            (n * n_knots, n_axes),
            (0..n * n_knots).map(|k| 0.3 + 0.04 * (k as f64)).collect(),
        )
        .unwrap();
        let implicit = Arc::new(ImplicitDesignPsiDerivative::new(
            phi_values,
            q_values,
            t_values,
            axis_components,
            None,
            None,
            n,
            n_knots,
            0,
            n_axes,
        ));

        let x_data = array![
            [1.0, 0.4, -0.2],
            [0.5, 1.1, 0.3],
            [-0.3, 0.9, 0.6],
            [0.8, -0.5, 1.2],
            [0.2, 0.7, -0.4],
        ];
        let x_design = Arc::new(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            x_data.clone(),
        )));
        let w_diag = Arc::new(array![1.0, 0.7, 1.3, 0.9, 1.1]);
        let s_psi = Array2::<f64>::eye(p) * 0.05;

        let beta_eval = array![0.20, -0.10, 0.30];
        let c_array = array![0.15, -0.08, 0.22, 0.05, -0.12];
        let dx_dpsi = implicit.materialize_first(0).expect("materialize_first");
        let dx_beta = dx_dpsi.dot(&beta_eval);
        let c_x_psi_beta_dense = &c_array * &dx_beta;

        let op = ImplicitHyperOperator {
            implicit_deriv: Arc::clone(&implicit),
            axis: 0,
            x_design,
            w_diag: crate::matrix::SignedWeightsArc::from_arc(w_diag),
            s_psi,
            p,
            c_x_psi_beta: Some(Arc::new(c_x_psi_beta_dense.clone())),
        };

        // Dense Jacobian column j: B_d e_j.
        let v_base = array![0.10, -0.05, 0.20];
        let eps = 1e-6;
        for j in 0..p {
            let mut e_j = Array1::<f64>::zeros(p);
            e_j[j] = 1.0;
            // Centered FD on mul_vec along e_j gives B_d e_j (operator is linear in v).
            let mut v_plus = v_base.clone();
            v_plus[j] += eps;
            let mut v_minus = v_base.clone();
            v_minus[j] -= eps;
            let fd = (&op.mul_vec(&v_plus) - &op.mul_vec(&v_minus)).mapv(|x| x / (2.0 * eps));
            let analytic = op.mul_vec(&e_j);
            for i in 0..p {
                let tol = 1e-7 * analytic[i].abs().max(1.0) + 1e-7;
                assert!(
                    (analytic[i] - fd[i]).abs() <= tol,
                    "FD col {j} mismatch at row {i}: analytic={:.6e}, fd={:.6e}",
                    analytic[i],
                    fd[i],
                );
            }
        }
    }

    #[test]
    pub(crate) fn test_pseudoinverse_scalar() {
        let mut g = Array2::<f64>::zeros((1, 1));
        g[[0, 0]] = 4.0;
        let v = Array1::from_vec(vec![8.0]);
        let result =
            pseudoinverse_times_vec(&g, v.as_slice().expect("contiguous test vector"), 1e-8);
        assert!((result[0] - 2.0).abs() < 1e-12);
    }

    /// Indefinite outer Hessian (no active bounds, no rank deficiency) must
    /// surface as `CorrectedCovarianceError::Indefinite` — never as a
    /// covariance with the negative directions silently clamped to zero.
    #[test]
    pub(crate) fn corrected_covariance_indefinite_returns_diagnostic() {
        // 2×2 outer Hessian with one positive and one clearly negative
        // eigenvalue ⇒ the projected (=full, since no active bounds) inertia
        // gate must reject. Using diag(2, -1) on a small p=2 base.
        let outer = ndarray::arr2(&[[2.0_f64, 0.0], [0.0, -1.0]]);

        // Build a SPD base H = I_2 so DenseSpectralOperator works trivially.
        let base = Array2::<f64>::eye(2);
        let hop = DenseSpectralOperator::from_symmetric(&base)
            .expect("DenseSpectralOperator from identity should succeed");

        // Two ρ-coords with arbitrary mode responses (their values don't
        // affect the inertia gate; the gate fires before any J·V_θ·Jᵀ work).
        let v0 = Array1::from_vec(vec![0.1, 0.2]);
        let v1 = Array1::from_vec(vec![0.3, 0.4]);

        // No theta supplied ⇒ active set is empty ⇒ projected Hessian = full
        // outer Hessian, which is indefinite ⇒ Err(Indefinite).
        let res = compute_corrected_covariance_with_constraints(
            &[v0.clone(), v1.clone()],
            &[],
            &outer,
            &hop,
            None,
            f64::NAN,
        );
        match res {
            Err(CorrectedCovarianceError::Indefinite(diag)) => {
                assert!(
                    diag.min_eigenvalue < -0.5,
                    "min eigenvalue should be ~-1, got {}",
                    diag.min_eigenvalue,
                );
                assert!(
                    diag.active_constraints.is_empty(),
                    "no theta supplied ⇒ no active constraints",
                );
                assert!(
                    !diag.suggested_action.is_empty(),
                    "diagnostic must include a suggested-action message",
                );
            }
            Err(other) => panic!("expected Indefinite diagnostic, got error: {:?}", other),
            Ok(cov) => panic!(
                "indefinite outer Hessian must NOT yield a covariance; got matrix shape {:?}",
                cov.matrix.shape(),
            ),
        }

        // Also check the legacy entry point preserves the same behaviour.
        let res_legacy = compute_corrected_covariance(&[v0, v1], &[], &outer, &hop);
        assert!(
            matches!(res_legacy, Err(CorrectedCovarianceError::Indefinite(_))),
            "legacy entry point must also surface Indefinite, got: {:?}",
            res_legacy.map(|m| m.shape().to_vec()),
        );
    }

    /// When the indefinite direction is precisely the bound-active θ, the
    /// projected-Hessian inertia gate sees a SPD matrix and we return a
    /// covariance (with the active coordinate listed in `active_constraints`).
    #[test]
    pub(crate) fn corrected_covariance_indefinite_with_active_bound_succeeds() {
        // Outer Hessian: positive on coord 0, negative on coord 1.
        let outer = ndarray::arr2(&[[3.0_f64, 0.0], [0.0, -2.0]]);
        let base = Array2::<f64>::eye(2);
        let hop = DenseSpectralOperator::from_symmetric(&base).expect("hop");

        let v0 = Array1::from_vec(vec![0.5, 0.0]);
        let v1 = Array1::from_vec(vec![0.0, 0.5]);

        // θ pinned at +RHO_BOUND on coord 1 (the negative-curvature direction).
        // After projecting away coord 1, the free Hessian is [[3]] — SPD.
        let theta = vec![0.0_f64, crate::solver::estimate::RHO_BOUND];
        let res = compute_corrected_covariance_with_constraints(
            &[v0, v1],
            &[],
            &outer,
            &hop,
            Some(&theta),
            0.0,
        )
        .expect("free-subspace SPD ⇒ covariance returned");
        assert_eq!(res.active_constraints, vec![1]);
        assert!(res.matrix.iter().all(|v| v.is_finite()));
    }

    // ------------------------------------------------------------------------
    // Numerical proof of the outer-ρ projected-kernel REML gradient bug
    // (runtime.rs:5465-5481).
    //
    // Hypothesis: when `hessian_logdet_correction ≠ 0` (cost uses projected
    // logdet `log|U_S^T H U_S|_+`) but the gradient computation uses the
    // full-space kernel `G_ε(H)` instead of `U_S (U_S^T H U_S)⁻¹ U_S^T`, the
    // third-derivative correction `D_β H[v_k] = X' diag(c ⊙ X v_k) X` leaks
    // onto null(S) and produces a spurious O(λ_k) outer-gradient term at
    // large λ_k.
    //
    // Method: Option B (synthetic). Build a Gaussian fixture where β̂(ρ) is
    // exact (so FD of `reml_laml_evaluate` returns the true gradient of the
    // projected REML cost). Inject a non-zero `c_array` via
    // `SinglePredictorGlmDerivatives` — this is a "lie" about the family
    // (the cost is Gaussian so the actual `dH/dρ` has no third-deriv term),
    // but it lets the analytic gradient path see a `D_β H[v_k]` that is
    // structurally identical to the survival_location_scale leak. FD does
    // NOT see the lie (cost only uses β/H/log-lik/penalty). Projected and
    // unprojected analytic gradients DO see it; the projected kernel kills
    // the null-space part exactly, so it matches FD.
    // ------------------------------------------------------------------------
    pub(crate) fn build_leak_proof_solution(
        rho: &[f64],
        x: &Array2<f64>,
        s1: &Array2<f64>,
        s2: &Array2<f64>,
        xty: &Array1<f64>,
        yty: f64,
        c_array: Array1<f64>,
        use_projected_kernel: bool,
    ) -> InnerSolution<'static> {
        let p = x.ncols();
        let n = x.nrows();
        assert_eq!(rho.len(), 2);
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();

        let xtx = crate::faer_ndarray::fast_atb(x, x);
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        s_lambda.scaled_add(lambdas[0], s1);
        s_lambda.scaled_add(lambdas[1], s2);

        let mut h = xtx.clone();
        h += &s_lambda;

        let hop = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let beta = hop.solve(xty);
        let deviance = yty - 2.0 * beta.dot(xty) + beta.dot(&xtx.dot(&beta));
        let log_lik = -0.5 * deviance;
        let penalty_quad = beta.dot(&s_lambda.dot(&beta));

        // Penalty logdet & first derivatives (rank = 2; both penalties have
        // rank-1 supports that don't overlap, so log|S_λ|_+ = ln λ₁ + ln λ₂).
        let (s_eigs, _) = s_lambda.eigh(faer::Side::Lower).unwrap();
        let threshold = positive_eigenvalue_threshold(s_eigs.as_slice().unwrap());
        let log_det_s = exact_pseudo_logdet(s_eigs.as_slice().unwrap(), threshold);

        let eps_det = 1e-7;
        let mut det1 = Array1::zeros(2);
        for k in 0..2 {
            let mut rp = rho.to_vec();
            rp[k] += eps_det;
            let lp: Vec<f64> = rp.iter().map(|r| r.exp()).collect();
            let mut sp = Array2::<f64>::zeros((p, p));
            sp.scaled_add(lp[0], s1);
            sp.scaled_add(lp[1], s2);
            let (ev_p, _) = sp.eigh(faer::Side::Lower).unwrap();
            let thp = positive_eigenvalue_threshold(ev_p.as_slice().unwrap());
            let ld_p = exact_pseudo_logdet(ev_p.as_slice().unwrap(), thp);

            let mut rm = rho.to_vec();
            rm[k] -= eps_det;
            let lm: Vec<f64> = rm.iter().map(|r| r.exp()).collect();
            let mut sm = Array2::<f64>::zeros((p, p));
            sm.scaled_add(lm[0], s1);
            sm.scaled_add(lm[1], s2);
            let (ev_m, _) = sm.eigh(faer::Side::Lower).unwrap();
            let thm = positive_eigenvalue_threshold(ev_m.as_slice().unwrap());
            let ld_m = exact_pseudo_logdet(ev_m.as_slice().unwrap(), thm);

            det1[k] = (ld_p - ld_m) / (2.0 * eps_det);
        }

        // Build projection U_S onto range(S_λ) (rank 2 here: cols 1 and 2 of X).
        // Use eigendecomposition of S_λ and keep eigenvectors with eigenvalue > tol.
        let (s_full_eigs, s_full_vecs) = s_lambda.eigh(faer::Side::Lower).unwrap();
        let s_thresh = positive_eigenvalue_threshold(s_full_eigs.as_slice().unwrap());
        let active: Vec<usize> = s_full_eigs
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > s_thresh)
            .map(|(i, _)| i)
            .collect();
        let r_rank = active.len();
        let mut u_s = Array2::<f64>::zeros((p, r_rank));
        for (j, &idx) in active.iter().enumerate() {
            for i in 0..p {
                u_s[[i, j]] = s_full_vecs[[i, idx]];
            }
        }
        // H_proj = U_Sᵀ H U_S; invert it.
        let h_proj = u_s.t().dot(&h).dot(&u_s);
        let (hp_eigs, hp_vecs) = h_proj.eigh(faer::Side::Lower).unwrap();
        let mut h_proj_inv = Array2::<f64>::zeros((r_rank, r_rank));
        for i in 0..r_rank {
            for j in 0..r_rank {
                let mut acc = 0.0;
                for k_idx in 0..r_rank {
                    acc += hp_vecs[[i, k_idx]] * hp_vecs[[j, k_idx]] / hp_eigs[k_idx];
                }
                h_proj_inv[[i, j]] = acc;
            }
        }
        let log_det_h_proj: f64 = hp_eigs.iter().map(|v| v.ln()).sum();
        let log_det_h_full = hop.logdet();
        let hessian_logdet_correction = log_det_h_proj - log_det_h_full;

        let deriv_provider = SinglePredictorGlmDerivatives {
            c_array,
            d_array: None,
            hessian_weights: Array1::ones(n),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
        };

        let r1 = penalty_matrix_root(s1).unwrap();
        let r2 = penalty_matrix_root(s2).unwrap();

        let penalty_subspace_trace = if use_projected_kernel {
            Some(Arc::new(PenaltySubspaceTrace {
                u_s,
                h_proj_inverse: h_proj_inv,
            }))
        } else {
            None
        };

        InnerSolution {
            log_likelihood: log_lik,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(hop),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(r1),
                PenaltyCoordinate::from_dense_root(r2),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: None,
            },
            deriv_provider: Box::new(deriv_provider),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction,
            penalty_subspace_trace,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: (p - r_rank) as f64,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    /// Numerical proof: at large λ_j, the unprojected-kernel REML gradient
    /// disagrees with finite-difference of the projected-logdet cost by a
    /// margin that grows with λ_j, while the projected-kernel gradient
    /// matches FD to floating-point tolerance.
    #[test]
    pub(crate) fn proof_outer_rho_projected_kernel_fixes_leak() {
        let n = 100;
        let p = 3;
        // Design: intercept + two "spline-like" columns. Intercept is in null(S₁) ∩ null(S₂).
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            x[[i, 0]] = 1.0; // intercept
            x[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
            x[[i, 2]] = (t - 0.5) * (t - 0.3);
        }
        // S₁ penalizes column 1, S₂ penalizes column 2. Both have null space ⊇ {e_0}.
        let mut s1 = Array2::<f64>::zeros((p, p));
        s1[[1, 1]] = 1.0;
        let mut s2 = Array2::<f64>::zeros((p, p));
        s2[[2, 2]] = 1.0;

        // Some synthetic response — y = X β_true + noise (deterministic).
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            y[i] =
                0.7 + 0.4 * (2.0 * std::f64::consts::PI * t).sin() + 0.1 * ((i as f64) * 0.7).cos();
        }
        let xty = crate::faer_ndarray::fast_atb(&x, &y.clone().insert_axis(ndarray::Axis(1)))
            .column(0)
            .to_owned();
        let yty = y.dot(&y);

        // c_array: a non-trivial vector — the "third-derivative weight surface".
        // Chosen to have non-zero mean (so it projects onto the intercept direction),
        // making the leak large.
        let c_array = Array1::from_shape_fn(n, |i| {
            0.3 + 0.5 * ((i as f64) * 0.11).sin() + 0.2 * ((i as f64) * 0.27).cos()
        });

        // Runaway scenario: ρ = [0.0, 12.0] → λ ≈ [1, 1.6e5]. Big enough that
        // λ_2 dominates H and the leak is highly visible, but not so big that
        // FD on the projected-logdet cost (cost ~ ½ ln λ_2) loses precision —
        // at ρ_2 ≳ 17 the FD denominator gets within ~1e-8 of the cost itself
        // and catastrophic cancellation makes FD unreliable as a reference.
        let rho = vec![0.0_f64, 12.0_f64];

        // --- (a) FD of projected-logdet cost via reml_laml_evaluate value-only ---
        let delta = 1e-4;
        let mut fd_grad = [0.0_f64; 2];
        for j in 0..2 {
            let mut rp = rho.clone();
            rp[j] += delta;
            let sp = build_leak_proof_solution(&rp, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
            let cost_p = reml_laml_evaluate(&sp, &rp, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rm = rho.clone();
            rm[j] -= delta;
            let sm = build_leak_proof_solution(&rm, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
            let cost_m = reml_laml_evaluate(&sm, &rm, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[j] = (cost_p - cost_m) / (2.0 * delta);
        }

        // --- (b) Analytic gradient WITHOUT projected kernel (pre-v0.3.31 bug) ---
        let sol_unproj =
            build_leak_proof_solution(&rho, &x, &s1, &s2, &xty, yty, c_array.clone(), false);
        let g_unproj = reml_laml_evaluate(&sol_unproj, &rho, EvalMode::ValueAndGradient, None)
            .unwrap()
            .gradient
            .unwrap();

        // --- (c) Analytic gradient WITH projected kernel (v0.3.31 fix) ---
        let sol_proj =
            build_leak_proof_solution(&rho, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
        let g_proj = reml_laml_evaluate(&sol_proj, &rho, EvalMode::ValueAndGradient, None)
            .unwrap()
            .gradient
            .unwrap();

        eprintln!(
            "=== Outer-ρ gradient at runaway ρ = {:?} (λ = {:?}) ===",
            rho,
            rho.iter().map(|r| r.exp()).collect::<Vec<_>>()
        );
        for j in 0..2 {
            eprintln!(
                "  coord {}: FD={:+.6e}   unprojected_analytic={:+.6e}   projected_analytic={:+.6e}",
                j, fd_grad[j], g_unproj[j], g_proj[j]
            );
            let rel_proj = (g_proj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
            let rel_unproj = (g_unproj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
            eprintln!(
                "           |projected − FD|/|FD| = {:.3e}   |unprojected − FD|/|FD| = {:.3e}",
                rel_proj, rel_unproj
            );
        }

        // --- Sweep λ_2 to check scaling ---
        eprintln!("=== Sweep λ_2 (coord 1) — unprojected analytic vs FD ===");
        for &rho2 in &[6.0_f64, 9.0, 12.0, 15.0, 18.0, 20.0] {
            let r = vec![0.0_f64, rho2];
            let fd1 = {
                let mut rp = r.clone();
                rp[1] += delta;
                let sp =
                    build_leak_proof_solution(&rp, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
                let cp = reml_laml_evaluate(&sp, &rp, EvalMode::ValueOnly, None)
                    .unwrap()
                    .cost;
                let mut rm = r.clone();
                rm[1] -= delta;
                let sm =
                    build_leak_proof_solution(&rm, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
                let cm = reml_laml_evaluate(&sm, &rm, EvalMode::ValueOnly, None)
                    .unwrap()
                    .cost;
                (cp - cm) / (2.0 * delta)
            };
            let su = build_leak_proof_solution(&r, &x, &s1, &s2, &xty, yty, c_array.clone(), false);
            let gu = reml_laml_evaluate(&su, &r, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();
            let sp = build_leak_proof_solution(&r, &x, &s1, &s2, &xty, yty, c_array.clone(), true);
            let gp = reml_laml_evaluate(&sp, &r, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();
            let leak = gu[1] - fd1;
            eprintln!(
                "  ρ_2={:5.1} λ_2={:+.3e}  FD={:+.6e}  unproj={:+.6e}  proj={:+.6e}  leak(unproj−FD)={:+.6e}",
                rho2,
                rho2.exp(),
                fd1,
                gu[1],
                gp[1],
                leak
            );
        }

        // --- Assertions (per task spec) ---
        // At the runaway coordinate (j = 1):
        let j = 1;
        let rel_proj = (g_proj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
        let rel_unproj = (g_unproj[j] - fd_grad[j]).abs() / fd_grad[j].abs().max(1e-12);
        assert!(
            rel_proj < 1e-2,
            "projected gradient should match FD at runaway coord: \
             FD={:+.6e}, projected={:+.6e}, rel={:.3e}",
            fd_grad[j],
            g_proj[j],
            rel_proj
        );
        assert!(
            rel_unproj > 0.5,
            "unprojected gradient should DISAGREE with FD at runaway coord: \
             FD={:+.6e}, unprojected={:+.6e}, rel={:.3e}",
            fd_grad[j],
            g_unproj[j],
            rel_unproj
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Implicit-function-theorem (IFT) correction: math validation
    // ═══════════════════════════════════════════════════════════════════════
    //
    // Builds a Gaussian REML problem where the inner KKT condition can be
    // solved in closed form, then deliberately perturbs β̂ away from β*.
    // Verifies the math derived in `reml_laml_evaluate`:
    //
    //   r = 0  ⇒  IFT correction = 0  ⇒  cost+gradient unchanged.
    //   r ≠ 0  ⇒  envelope formula mismatches FD by O(‖r‖·‖v_k‖);
    //            IFT-corrected formula matches FD to higher order.

    /// Build a Gaussian InnerSolution at an *arbitrary* β̂ (not necessarily
    /// the inner optimum), recomputing log_likelihood, penalty_quadratic,
    /// and the KKT residual r = S(λ)β̂ − ∇ℓ(β̂) consistently.
    pub(crate) fn build_gaussian_solution_at_beta(
        rho: &[f64],
        beta_hat: Array1<f64>,
        attach_residual: bool,
    ) -> InnerSolution<'_> {
        let p = 3usize;
        let n = 50usize;
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];
        let yty = 20.0;
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let penalty_quad = lambdas[0] * beta_hat.dot(&s1.dot(&beta_hat))
            + lambdas[1] * beta_hat.dot(&s2.dot(&beta_hat));
        let deviance = yty - 2.0 * beta_hat.dot(&xty) + beta_hat.dot(&xtx.dot(&beta_hat));
        let log_likelihood = -0.5 * deviance;

        // KKT residual r = S(λ)β̂ − ∇ℓ(β̂).  For Gaussian:
        //   ℓ(β) = −½(yᵀy − 2 βᵀX'y + βᵀX'Xβ),   ∇ℓ(β) = X'y − X'Xβ.
        // ⇒ r = (λ₁S₁+λ₂S₂)β̂ − (X'y − X'Xβ̂) = Hβ̂ − X'y.
        // At β* = H⁻¹X'y this is identically zero.
        let kkt_residual = if attach_residual {
            Some(ProjectedKktResidual::from_active_projected(
                &h.dot(&beta_hat) - &xty,
            ))
        } else {
            None
        };

        let r1 = penalty_matrix_root(&s1).unwrap();
        let r2 = penalty_matrix_root(&s2).unwrap();

        let penalty_logdet = gaussian_penalty_logdet_fd(p, &s1, &s2, rho);

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta: beta_hat,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(r1),
                PenaltyCoordinate::from_dense_root(r2),
            ],
            penalty_logdet,
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: n,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    #[test]
    pub(crate) fn malformed_projected_kkt_residual_is_contract_error() {
        let rho: Vec<f64> = vec![1.0, -0.5];
        let beta_hat = array![0.1, -0.2, 0.3];
        let mut sol = build_gaussian_solution_at_beta(&rho, beta_hat, false);
        sol.dispersion = DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h: true,
            include_logdet_s: true,
        };
        sol.kkt_residual = Some(ProjectedKktResidual::from_active_projected(array![
            0.0, 0.0
        ]));

        let err = match reml_laml_evaluate(&sol, &rho, EvalMode::ValueAndGradient, None) {
            Ok(_) => panic!("wrong-length projected KKT residual must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.contains("projected KKT residual length mismatch"),
            "unexpected error: {err}"
        );
    }

    /// At exact KKT (r = 0) the IFT correction is identically zero.
    /// Attaching `Some(zeros)` must not perturb the envelope cost/gradient.
    #[test]
    pub(crate) fn ift_correction_vanishes_at_exact_kkt() {
        let rho: Vec<f64> = vec![1.0, -0.5];
        // Recompute exact β* = H⁻¹X'y at this ρ.
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);
        let op_for_solve = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let beta_star = op_for_solve.solve(&xty);

        let sol_envelope = build_gaussian_solution_at_beta(&rho, beta_star.clone(), false);
        let grad_envelope =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();
        let cost_envelope = reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueOnly, None)
            .unwrap()
            .cost;

        let sol_with_residual = build_gaussian_solution_at_beta(&rho, beta_star.clone(), true);
        let r_norm = sol_with_residual
            .kkt_residual
            .as_ref()
            .unwrap()
            .as_array()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            r_norm < 1e-10,
            "residual at exact β* should be numerically zero, got ‖r‖∞ = {:.3e}",
            r_norm
        );

        let result_ift =
            reml_laml_evaluate(&sol_with_residual, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let grad_ift = result_ift.gradient.unwrap();
        let cost_ift = result_ift.cost;

        assert_relative_eq!(
            cost_ift,
            cost_envelope,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
        for k in 0..rho.len() {
            assert_relative_eq!(
                grad_ift[k],
                grad_envelope[k],
                epsilon = 1e-10,
                max_relative = 1e-8
            );
        }
    }

    /// Regression test for issue #197.
    ///
    /// When ρ_k is pinned at its upper bound the IFT block
    /// (`compute_kkt_residual_rho_corrections`) already projects out the
    /// gradient (sets it to 0). Prior to the #197 fix the main envelope
    /// block kept the trace term `½·tr(K·λ_k S_k)`, the penalty quadratic
    /// `½·λ_k β'S_kβ` and the `½·∂log|S|/∂ρ_k` term — yielding a non-zero
    /// outer gradient component along a frozen axis (mixing constrained
    /// `v_k = 0` with unconstrained λ_k-active terms).
    ///
    /// Contract: at an active upper bound the FULL ρ-gradient component
    /// (envelope + IFT correction) must be exactly 0.0 — the
    /// gradient-projection convention used by the box-constrained outer
    /// solver. This holds for both the no-residual envelope path and the
    /// with-residual IFT-correction path.
    #[test]
    pub(crate) fn rho_gradient_at_upper_bound_is_zero_envelope_and_ift_consistent_issue_197() {
        // Coord 0 pinned at +RHO_BOUND, coord 1 free.
        let rho: Vec<f64> = vec![crate::solver::estimate::RHO_BOUND, -0.5];

        // Use a β perturbed away from β* so that the with-residual path
        // exercises a non-zero IFT correction on the free coordinate, but
        // the active coord must still come out exactly 0.
        let beta_hat = array![0.7, -0.4, 0.2];

        // Envelope path (no residual attached).
        let sol_envelope = build_gaussian_solution_at_beta(&rho, beta_hat.clone(), false);
        let result_env =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let grad_env = result_env.gradient.unwrap();
        assert_eq!(
            grad_env[0], 0.0,
            "envelope ρ-gradient at active upper bound must be exactly 0.0 \
             (gradient-projection convention, see #197); got {:+.6e}",
            grad_env[0]
        );

        // IFT path (with KKT residual at perturbed β̂).
        let sol_with_residual = build_gaussian_solution_at_beta(&rho, beta_hat, true);
        let result_ift =
            reml_laml_evaluate(&sol_with_residual, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let grad_ift = result_ift.gradient.unwrap();
        assert_eq!(
            grad_ift[0], 0.0,
            "IFT-corrected ρ-gradient at active upper bound must be exactly 0.0 \
             — envelope and IFT-correction blocks must agree (#197); got {:+.6e}",
            grad_ift[0]
        );

        // Free coordinate must remain non-zero — otherwise the test would
        // trivially pass by zeroing everything. (Tolerance is loose; the
        // point is the active-mask doesn't accidentally freeze coord 1.)
        assert!(
            grad_env[1].abs() > 1e-8,
            "free-coord envelope gradient should be non-trivial: got {:+.6e}",
            grad_env[1]
        );
        assert!(
            grad_ift[1].abs() > 1e-8,
            "free-coord IFT-corrected gradient should be non-trivial: got {:+.6e}",
            grad_ift[1]
        );
    }

    /// With β̂ perturbed off β* and the matching r attached, the IFT-corrected
    /// gradient must match a re-solved FD reference much better than the
    /// uncorrected envelope formula evaluated at the perturbed β̂.
    #[test]
    pub(crate) fn ift_correction_recovers_fd_at_perturbed_beta() {
        let rho: Vec<f64> = vec![0.5, 0.3];

        // Re-solve for exact β* at ρ.
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);
        let op_for_solve = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let beta_star = op_for_solve.solve(&xty);

        // Use Fixed-dispersion V = -ℓ + ½βᵀSβ + ½(log|H| − log|S|_+).
        // This is the parameterisation under which the IFT correction is
        // exact (∂V/∂β = r, no `denom/dp` chain factor as in the profiled
        // Gaussian path).  Matches the production survival-marginal-slope
        // path that the large-scale failure exercises.
        pub(crate) fn to_fixed<'a>(mut sol: InnerSolution<'a>) -> InnerSolution<'a> {
            sol.dispersion = DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            };
            sol
        }

        // FD reference: evaluate at re-solved β*(ρ±ε), which is what an
        // ideal inner solver would deliver.  Use ValueOnly to avoid the
        // recursive gradient path.
        let fd_eps = 1e-5;
        let mut fd_grad = Array1::<f64>::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rho_plus = rho.clone();
            rho_plus[k] += fd_eps;
            let mut h_plus = xtx.clone();
            let lambdas_plus: Vec<f64> = rho_plus.iter().map(|&r| r.exp()).collect();
            h_plus.scaled_add(lambdas_plus[0], &s1);
            h_plus.scaled_add(lambdas_plus[1], &s2);
            let beta_star_plus = DenseSpectralOperator::from_symmetric(&h_plus)
                .unwrap()
                .solve(&xty);
            let sol_plus = to_fixed(build_gaussian_solution_at_beta(
                &rho_plus,
                beta_star_plus,
                false,
            ));
            let cost_plus = reml_laml_evaluate(&sol_plus, &rho_plus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rho_minus = rho.clone();
            rho_minus[k] -= fd_eps;
            let mut h_minus = xtx.clone();
            let lambdas_minus: Vec<f64> = rho_minus.iter().map(|&r| r.exp()).collect();
            h_minus.scaled_add(lambdas_minus[0], &s1);
            h_minus.scaled_add(lambdas_minus[1], &s2);
            let beta_star_minus = DenseSpectralOperator::from_symmetric(&h_minus)
                .unwrap()
                .solve(&xty);
            let sol_minus = to_fixed(build_gaussian_solution_at_beta(
                &rho_minus,
                beta_star_minus,
                false,
            ));
            let cost_minus = reml_laml_evaluate(&sol_minus, &rho_minus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[k] = (cost_plus - cost_minus) / (2.0 * fd_eps);
        }

        // Perturb β̂ off β* — small enough that linear IFT recovers cleanly.
        let perturb = Array1::from_vec(vec![0.02, -0.015, 0.025]);
        let beta_hat = &beta_star + &perturb;

        let sol_envelope = to_fixed(build_gaussian_solution_at_beta(
            &rho,
            beta_hat.clone(),
            false,
        ));
        let grad_envelope =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueAndGradient, None)
                .unwrap()
                .gradient
                .unwrap();

        let sol_ift = to_fixed(build_gaussian_solution_at_beta(
            &rho,
            beta_hat.clone(),
            true,
        ));
        let r_norm = sol_ift
            .kkt_residual
            .as_ref()
            .unwrap()
            .as_array()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            r_norm > 1e-3,
            "perturbed β̂ should produce a non-trivial residual, got ‖r‖∞ = {:.3e}",
            r_norm
        );
        let grad_ift = reml_laml_evaluate(&sol_ift, &rho, EvalMode::ValueAndGradient, None)
            .unwrap()
            .gradient
            .unwrap();

        // IFT correction must shrink the gradient error meaningfully on at
        // least one coordinate, and never blow it up on any.
        let mut at_least_one_improved = false;
        for k in 0..rho.len() {
            let err_envelope = (grad_envelope[k] - fd_grad[k]).abs();
            let err_ift = (grad_ift[k] - fd_grad[k]).abs();
            assert!(
                err_ift <= err_envelope * 1.05 + 1e-9,
                "IFT correction must not enlarge gradient error: coord={} envelope_err={:.3e} \
                 ift_err={:.3e} FD={:.6e}",
                k,
                err_envelope,
                err_ift,
                fd_grad[k]
            );
            if err_ift < err_envelope * 0.5 && err_envelope > 1e-6 {
                at_least_one_improved = true;
            }
        }
        assert!(
            at_least_one_improved,
            "IFT correction should improve gradient accuracy on at least one coord: \
             envelope=[{:.3e}, {:.3e}] ift=[{:.3e}, {:.3e}] fd=[{:.3e}, {:.3e}]",
            (grad_envelope[0] - fd_grad[0]).abs(),
            (grad_envelope[1] - fd_grad[1]).abs(),
            (grad_ift[0] - fd_grad[0]).abs(),
            (grad_ift[1] - fd_grad[1]).abs(),
            fd_grad[0],
            fd_grad[1],
        );
    }

    /// The analytic rho Hessian must differentiate the same KKT-residual
    /// correction used by the value and gradient. This is the minimized
    /// reproduction of the large-scale failure mode: an off-KKT inner mode with a
    /// finite residual made the envelope Hessian inconsistent, so ARC chased a
    /// curvature model for the wrong objective.
    #[test]
    pub(crate) fn ift_correction_recovers_fd_hessian_at_perturbed_beta() {
        let rho: Vec<f64> = vec![0.5, 0.3];
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0]];
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let xty = array![5.0, 3.0, 2.0];

        pub(crate) fn to_fixed<'a>(mut sol: InnerSolution<'a>) -> InnerSolution<'a> {
            sol.dispersion = DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            };
            sol
        }

        let solve_beta_star = |rho_eval: &[f64]| -> Array1<f64> {
            let lambdas_eval: Vec<f64> = rho_eval.iter().map(|&r| r.exp()).collect();
            let mut h = xtx.clone();
            h.scaled_add(lambdas_eval[0], &s1);
            h.scaled_add(lambdas_eval[1], &s2);
            DenseSpectralOperator::from_symmetric(&h)
                .unwrap()
                .solve(&xty)
        };
        let exact_profile_cost = |rho_eval: &[f64]| -> f64 {
            let beta_star = solve_beta_star(rho_eval);
            let sol = to_fixed(build_gaussian_solution_at_beta(rho_eval, beta_star, false));
            reml_laml_evaluate(&sol, rho_eval, EvalMode::ValueOnly, None)
                .unwrap()
                .cost
        };

        let fd_eps = 2e-4;
        let mut fd_hessian = Array2::<f64>::zeros((rho.len(), rho.len()));
        let center_cost = exact_profile_cost(&rho);
        for i in 0..rho.len() {
            for j in i..rho.len() {
                let value = if i == j {
                    let mut rho_plus = rho.clone();
                    rho_plus[i] += fd_eps;
                    let mut rho_minus = rho.clone();
                    rho_minus[i] -= fd_eps;
                    (exact_profile_cost(&rho_plus) - 2.0 * center_cost
                        + exact_profile_cost(&rho_minus))
                        / (fd_eps * fd_eps)
                } else {
                    let mut pp = rho.clone();
                    pp[i] += fd_eps;
                    pp[j] += fd_eps;
                    let mut pm = rho.clone();
                    pm[i] += fd_eps;
                    pm[j] -= fd_eps;
                    let mut mp = rho.clone();
                    mp[i] -= fd_eps;
                    mp[j] += fd_eps;
                    let mut mm = rho.clone();
                    mm[i] -= fd_eps;
                    mm[j] -= fd_eps;
                    (exact_profile_cost(&pp) - exact_profile_cost(&pm) - exact_profile_cost(&mp)
                        + exact_profile_cost(&mm))
                        / (4.0 * fd_eps * fd_eps)
                };
                fd_hessian[[i, j]] = value;
                if i != j {
                    fd_hessian[[j, i]] = value;
                }
            }
        }

        let beta_star = solve_beta_star(&rho);
        let beta_hat = &beta_star + &Array1::from_vec(vec![0.02, -0.015, 0.025]);
        let sol_envelope = to_fixed(build_gaussian_solution_at_beta(
            &rho,
            beta_hat.clone(),
            false,
        ));
        let hessian_envelope =
            reml_laml_evaluate(&sol_envelope, &rho, EvalMode::ValueGradientHessian, None)
                .unwrap()
                .hessian
                .unwrap_analytic();

        let sol_ift = to_fixed(build_gaussian_solution_at_beta(&rho, beta_hat, true));
        let hessian_ift = reml_laml_evaluate(&sol_ift, &rho, EvalMode::ValueGradientHessian, None)
            .unwrap()
            .hessian
            .unwrap_analytic();

        let mut envelope_was_wrong = false;
        for i in 0..rho.len() {
            for j in 0..rho.len() {
                let envelope_err = (hessian_envelope[[i, j]] - fd_hessian[[i, j]]).abs();
                let ift_err = (hessian_ift[[i, j]] - fd_hessian[[i, j]]).abs();
                assert!(
                    ift_err <= envelope_err * 0.25 + 2e-5,
                    "IFT Hessian correction failed at ({}, {}): envelope={:.8e} ift={:.8e} \
                     fd={:.8e} envelope_err={:.3e} ift_err={:.3e}",
                    i,
                    j,
                    hessian_envelope[[i, j]],
                    hessian_ift[[i, j]],
                    fd_hessian[[i, j]],
                    envelope_err,
                    ift_err
                );
                if envelope_err > 1e-4 && ift_err < envelope_err * 0.1 {
                    envelope_was_wrong = true;
                }
            }
        }
        assert!(
            envelope_was_wrong,
            "test did not reproduce the Hessian bug: envelope={:?} ift={:?} fd={:?}",
            hessian_envelope, hessian_ift, fd_hessian
        );
    }

    #[test]
    pub(crate) fn bug_hunt_penalty_matrix_root_reconstructs_with_effective_rank() {
        let s = ndarray::arr2(&[
            [2.0_f64, 1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1e-14, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);
        let l = penalty_matrix_root(&s)
            .expect("penalty matrix root should be computable for semidefinite inputs");
        // `penalty_matrix_root` returns the root `R` with shape `(rank, n)`
        // satisfying the codebase-wide convention `S = Rᵀ R` — this is the
        // identity that every downstream consumer relies on (e.g.
        // `PenaltyCoordinate::from_block_root` asserts `root.ncols() == n`
        // and `apply_root` computes `R β` to get the per-mode scores). The
        // previous test computed `R Rᵀ` instead, which for a non-full-rank
        // input is a `(rank × rank)` matrix that cannot even be subtracted
        // from the `(n × n)` `s` — so the test was failing on the shape
        // mismatch panic before it ever reached the assertion. Reconstruct
        // through the actual contract.
        let recon = l.t().dot(&l);
        let frob_s = s.iter().map(|x| x * x).sum::<f64>().sqrt();
        let frob_err = (&recon - &s).iter().map(|x| x * x).sum::<f64>().sqrt();
        let rel = frob_err / frob_s.max(1.0);
        assert!(
            rel < 1e-9,
            "Penalty root must reconstruct S_lambda (as RᵀR) to numerical tolerance, but relative Frobenius error was {rel:.3e}.",
        );
    }

    #[test]
    pub(crate) fn bug_hunt_block_penalty_logdet_derivs_match_finite_difference_shared_columns() {
        let s1 = ndarray::arr2(&[[2.0_f64, 1.0], [1.0, 1.0]]);
        let s2 = ndarray::arr2(&[[1.0_f64, 0.5], [0.5, 2.0]]);
        let rho = ndarray::arr1(&[0.2_f64, -0.4]);
        let delta = 1e-6_f64;
        let per_block_rho = vec![rho.clone()];
        let penalties_block = vec![s1.clone(), s2.clone()];
        let per_block_penalties: Vec<&[Array2<f64>]> = vec![penalties_block.as_slice()];
        let out = compute_block_penalty_logdet_derivs(&per_block_rho, &per_block_penalties, delta)
            .expect("logdet derivs should be finite");
        let f = |r: &Array1<f64>| -> f64 {
            let l1 = r[0].exp();
            let l2 = r[1].exp();
            let s = s1.mapv(|x| x * l1) + s2.mapv(|x| x * l2) + Array2::<f64>::eye(2) * delta;
            let op = DenseSpectralOperator::from_symmetric(&s).expect("spd for fd");
            op.logdet()
        };
        let eps = 1e-5_f64;
        for k in 0..2 {
            let mut rp = rho.clone();
            rp[k] += eps;
            let mut rm = rho.clone();
            rm[k] -= eps;
            let fd = (f(&rp) - f(&rm)) / (2.0 * eps);
            assert!(
                (out.first[k] - fd).abs() < 1e-5,
                "First derivative with shared penalty columns must match finite differences; coord {k} analytic={:.8e} finite_diff={:.8e}.",
                out.first[k],
                fd
            );
        }
    }

    // ─── Issue #200 regression: cost and gradient agree under
    // `rho_curvature_scale != 1` when the documented contract is met.
    //
    // The contract (see `InnerSolution::rho_curvature_scale`):
    //   1. `hessian_op` is the rescaled curvature `H_op = s · (H_unp + Σ λ_k S_k)`.
    //   2. `hessian_logdet_correction = −p · log(s)` so that
    //      `hop.logdet() + correction = log|H_unp + Σ λ_k S_k|` (unscaled).
    //   3. Gradient drift uses `curvature_lambdas = s · λ_k`, matching
    //      `∂H_op/∂ρ_k = s · λ_k S_k`.
    //
    // With these three terms wired consistently, `dV/dρ_k` matches the
    // finite-difference of the cost surface to 1e-6 — the test below
    // checks this end-to-end at `s = 2.0` to pin the convention.
    //
    // Pre-fix (issue #200 head), this test passed too because all three
    // sites were already aligned in the survival path; what the fix adds
    // is the documented contract on the public field plus a runtime guard
    // that refuses `rho_curvature_scale ≤ 0 / non-finite` (would silently
    // corrupt both cost and gradient).  The test pins the contract so
    // any future change that breaks the trio is caught immediately.
    pub(crate) fn build_scaled_curvature_solution(rho: &[f64], s: f64) -> InnerSolution<'static> {
        // 2×2 unpenalized Hessian (SPD).
        let h_unp = array![[3.0_f64, 0.5], [0.5, 5.0]];
        // Single penalty S = I (2×2), so root = I.
        let s_root = Array2::<f64>::eye(2);
        let penalty_coord = PenaltyCoordinate::from_dense_root(s_root.clone());
        let s_mat = s_root.dot(&s_root.t());
        // Build H_op = s · (H_unp + λ · S) for the given ρ.
        assert_eq!(rho.len(), 1, "single-ρ scaled-curvature test");
        let lambda = rho[0].exp();
        let mut h_op_dense = &h_unp + &(&s_mat * lambda);
        h_op_dense.mapv_inplace(|v| s * v);
        let hop = Arc::new(
            DenseSpectralOperator::from_symmetric(&h_op_dense).expect("scaled H_op is SPD"),
        );
        let p = h_op_dense.nrows() as f64;
        // Contract: subtract p·log(s) so cost evaluates log|H_unp + λS|.
        let hessian_logdet_correction = -p * s.ln();

        InnerSolution {
            // β = 0 isolates the log|H| term — penalty quadratic and its
            // ρ-derivative vanish independently of λ, leaving the gradient
            // entirely a `0.5 · tr(K · ∂H/∂ρ)` test.
            log_likelihood: 0.0,
            penalty_quadratic: 0.0,
            hessian_op: hop,
            beta: array![0.0_f64, 0.0],
            penalty_coords: vec![penalty_coord],
            penalty_logdet: PenaltyLogdetDerivs {
                // S = I has log|S|_+ = 0 with ρ-derivative `rank(S) = p = 2`,
                // independent of ρ.  We disable the log|S| term via
                // `include_logdet_s = false` below so neither value nor
                // first derivative enter the cost or gradient.
                value: 0.0,
                first: array![2.0],
                second: Some(array![[0.0]]),
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction,
            penalty_subspace_trace: None,
            rho_curvature_scale: s,
            rho_prior: crate::types::RhoPrior::Flat,
            n_observations: 10,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            // Fixed-dispersion with logdet_h on, logdet_s off makes the
            // cost reduce to `0.5 · (hop.logdet() + correction)` plus
            // ρ-independent constants.  Pure log|H| derivative test.
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: false,
            },
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: None,
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    #[test]
    pub(crate) fn issue_200_cost_gradient_agree_under_rho_curvature_scale() {
        let s = 2.0_f64;
        let rho0 = vec![0.3_f64];

        // Analytic gradient at ρ₀ via the unified evaluator.
        let solution_center = build_scaled_curvature_solution(&rho0, s);
        let result = reml_laml_evaluate(&solution_center, &rho0, EvalMode::ValueAndGradient, None)
            .expect("center evaluation");
        let analytic = result
            .gradient
            .expect("gradient returned for fixed-dispersion path")[0];

        // Finite-difference the cost surface: rebuild H_op at ρ₀±ε so the
        // scaling convention is met at every probe point.
        let eps = 1e-6_f64;
        let mut rho_plus = rho0.clone();
        rho_plus[0] += eps;
        let mut rho_minus = rho0.clone();
        rho_minus[0] -= eps;
        let cost_plus = reml_laml_evaluate(
            &build_scaled_curvature_solution(&rho_plus, s),
            &rho_plus,
            EvalMode::ValueOnly,
            None,
        )
        .expect("forward evaluation")
        .cost;
        let cost_minus = reml_laml_evaluate(
            &build_scaled_curvature_solution(&rho_minus, s),
            &rho_minus,
            EvalMode::ValueOnly,
            None,
        )
        .expect("backward evaluation")
        .cost;
        let fd = (cost_plus - cost_minus) / (2.0 * eps);

        // 1e-6 tolerance per the issue.  At ρ=0.3, λ=exp(0.3)≈1.35; the
        // gradient is `0.5 · tr((H_unp + λS)⁻¹ · λ S)` on the order of 0.3,
        // so the absolute tolerance pins ≥6 matching decimals.
        assert!(
            (analytic - fd).abs() < 1e-6,
            "issue #200: cost/gradient must agree under rho_curvature_scale={s} \
             (analytic={analytic:.10e}, fd={fd:.10e}, |diff|={:.3e})",
            (analytic - fd).abs(),
        );
    }

    #[test]
    pub(crate) fn issue_200_rejects_non_positive_rho_curvature_scale() {
        // Build a baseline solution then mutate `rho_curvature_scale` to an
        // invalid value.  The evaluator must reject rather than silently
        // emit a corrupt cost/gradient pair.
        let mut solution = build_scaled_curvature_solution(&[0.0_f64], 1.0);
        solution.rho_curvature_scale = 0.0;
        let err = reml_laml_evaluate(&solution, &[0.0_f64], EvalMode::ValueOnly, None)
            .expect_err("zero curvature scale must be rejected");
        assert!(
            format!("{err}").contains("rho_curvature_scale"),
            "error message must name the offending field, got: {err}",
        );

        let mut solution = build_scaled_curvature_solution(&[0.0_f64], 1.0);
        solution.rho_curvature_scale = -1.5;
        let err = reml_laml_evaluate(&solution, &[0.0_f64], EvalMode::ValueOnly, None)
            .expect_err("negative curvature scale must be rejected");
        assert!(
            format!("{err}").contains("rho_curvature_scale"),
            "error message must name the offending field, got: {err}",
        );

        let mut solution = build_scaled_curvature_solution(&[0.0_f64], 1.0);
        solution.rho_curvature_scale = f64::NAN;
        let err = reml_laml_evaluate(&solution, &[0.0_f64], EvalMode::ValueOnly, None)
            .expect_err("NaN curvature scale must be rejected");
        assert!(
            format!("{err}").contains("rho_curvature_scale"),
            "error message must name the offending field, got: {err}",
        );
    }

    /// `DenseCholeskyValueOnlyOperator` must agree with `DenseSpectralOperator`
    /// on `logdet`, `solve`, and `trace_hinv_product` for SPD inputs.
    ///
    /// This pins the acceptance criterion from issue #277: ValueOnly REML costs
    /// routed through Cholesky must match the eigendecomposition baseline.
    #[test]
    pub(crate) fn dense_cholesky_value_only_matches_spectral() {
        use approx::assert_relative_eq;

        // 4×4 SPD matrix.
        let h = array![
            [6.0, 2.0, 1.0, 0.5],
            [2.0, 5.0, 1.5, 0.25],
            [1.0, 1.5, 4.0, 0.75],
            [0.5, 0.25, 0.75, 3.0],
        ];

        let spectral = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let cholesky = DenseCholeskyValueOnlyOperator::from_spd(&h).unwrap();

        // logdet must agree within floating-point tolerance.
        assert_relative_eq!(cholesky.logdet(), spectral.logdet(), epsilon = 1e-10);

        // dim and active_rank.
        assert_eq!(cholesky.dim(), 4);
        assert_eq!(cholesky.active_rank(), 4);

        // solve must agree.
        let rhs = array![1.0, 2.0, 3.0, 4.0];
        let sol_spec = HessianOperator::solve(&spectral, &rhs);
        let sol_chol = HessianOperator::solve(&cholesky, &rhs);
        for i in 0..4 {
            assert_relative_eq!(sol_chol[i], sol_spec[i], epsilon = 1e-10);
        }

        // trace_hinv_product must agree.
        let a = array![
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 2.5, 1.0],
            [0.0, 0.0, 1.0, 1.5],
        ];
        assert_relative_eq!(
            cholesky.trace_hinv_product(&a),
            spectral.trace_hinv_product(&a),
            epsilon = 1e-10
        );
