#[cfg(test)]
mod amortized_encoder_tests {
    use crate::terms::sae::manifold::tests::small_two_atom_periodic_term;

    /// #1026 ladder item 2/3 — the amortized encoder is reachable end-to-end
    /// from a fitted term and is certificate-honest: it encodes the dictionary's
    /// own fit-time target, returns one result per atom with the right shape, and
    /// every row is either certified or counted in
    /// `encode_uncertified_count` (never silently miscounted), with the exact
    /// fallback strictly reducing the uncertified count it inherits.
    #[test]
    fn amortized_encode_fitted_is_reachable_and_certificate_honest() {
        let (term, target, rho) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();

        let results = term
            .amortized_encode_fitted(target.view(), &rho)
            .expect("amortized encode of the fit-time target runs end-to-end");
        assert_eq!(
            results.len(),
            k,
            "one encode result per atom in dictionary order"
        );

        for (atom_idx, result) in results.iter().enumerate() {
            assert_eq!(
                result.coords.nrows(),
                n,
                "atom {atom_idx} encode must produce one coordinate per row"
            );
            assert_eq!(
                result.coords.ncols(),
                term.atoms[atom_idx].latent_dim,
                "atom {atom_idx} encode coords must match its latent dim"
            );
            // The uncertified count is the honest tally of rows the certificate
            // could not gate — it must equal the false entries of the mask.
            let uncertified = result.certified.iter().filter(|c| !**c).count();
            assert_eq!(
                result.encode_uncertified_count, uncertified,
                "atom {atom_idx} uncertified count must match the certificate mask"
            );
            assert_eq!(
                result.certified.len(),
                n,
                "atom {atom_idx} certificate mask must cover every row"
            );
        }
    }

    /// The fitted amplitudes the encoder derives are exactly the assignment
    /// masses the reconstruction is assembled from — feeding them back is the
    /// self-consistency the distilled map is supervised against.
    #[test]
    fn fitted_assignment_amplitudes_match_the_assignment_masses() {
        let (term, _target, rho) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();
        let amplitudes = term
            .fitted_assignment_amplitudes(&rho)
            .expect("fitted amplitudes derive from the assignment");
        assert_eq!(amplitudes.dim(), (n, k));
        for row in 0..n {
            let a = term
                .assignment
                .try_assignments_row_for_rho(row, &rho)
                .expect("assignment row resolves");
            for atom_idx in 0..k {
                assert_eq!(
                    amplitudes[[row, atom_idx]],
                    a[atom_idx],
                    "amplitude[{row},{atom_idx}] must equal the assignment mass"
                );
            }
        }
    }
}

#[cfg(test)]
mod outer_gradient_error_classification_1451_tests {
    use super::OuterGradientError;

    /// #1451 — the three numerical/linear-algebra failure sites inside the
    /// deflation path (`apply_cached_arrow_hessian`, the projected `h_span.eigh`,
    /// and `DeflatedArrowSolver::from_orthonormal_gauges`) must distinguish a
    /// genuine near-singular conditioning trip (FD-eligible `IllConditioned`)
    /// from an internal-invariant defect — a shape/dimension mismatch or a
    /// non-finite intermediate — which MUST propagate (`InternalInvariant`, NOT
    /// FD-eligible).
    ///
    /// `OuterGradientError::classify_arrow_solver_error` is the helper all three
    /// sites route through. Before the #1451 fix every failure there was
    /// re-labelled `IllConditioned` (the original `conditioning_err`), so the
    /// shape/non-finite cases below would have been FD-eligible — masking an
    /// internal defect behind a plausible-but-wrong FD descent direction, exactly
    /// the regression #1436 set out to eliminate. This test pins that a
    /// shape/non-finite error classifies to `InternalInvariant` (so it
    /// propagates) while a genuine finite, correctly-shaped near-singular failure
    /// stays `IllConditioned` (so it keeps the #1273 FD fallback).
    #[test]
    fn classify_arrow_solver_error_routes_shape_and_nonfinite_to_internal_1451() {
        let conditioning = || OuterGradientError::IllConditioned {
            reason: "near-singular joint Hessian (min/max pivot ratio 5.3e-16)".to_string(),
        };

        // Shape/dimension-mismatch markers emitted by the deflation helpers must
        // classify as InternalInvariant and therefore be NOT FD-eligible.
        let shape_messages = [
            "apply_cached_arrow_hessian: vector shapes (t=3, beta=2) != cache shapes (t=4, beta=2)",
            "DeflatedArrowSolver: gauge length 5 != cache full length 6",
            "DeflatedArrowSolver: solution length 5 != cache full length 6",
        ];
        for msg in shape_messages {
            let classified =
                OuterGradientError::classify_arrow_solver_error(msg, conditioning());
            assert!(
                matches!(classified, OuterGradientError::InternalInvariant { .. }),
                "shape mismatch must classify to InternalInvariant (#1451); got {classified}"
            );
            assert!(
                !classified.is_conditioning_recoverable(),
                "a shape mismatch must NOT be conditioning-recoverable (#1451); got {classified}"
            );
            assert!(
                !classified.admits_plain_solver_fallback(1.0),
                "a shape mismatch must NOT admit the plain-solver fallback even at finite cost (#1451)"
            );
        }

        // Non-finite-intermediate markers must likewise propagate as internal.
        let nonfinite_messages = [
            "DeflatedArrowSolver: gauge stiffness must be finite and positive; got NaN",
            "outer_gradient_arrow_solver: non-finite entry in projected gauge Hessian",
        ];
        for msg in nonfinite_messages {
            let classified =
                OuterGradientError::classify_arrow_solver_error(msg, conditioning());
            assert!(
                matches!(classified, OuterGradientError::InternalInvariant { .. }),
                "non-finite intermediate must classify to InternalInvariant (#1451); \
                 got {classified}"
            );
            assert!(
                !classified.is_conditioning_recoverable(),
                "a non-finite intermediate must NOT be conditioning-recoverable (#1451); got {classified}"
            );
        }

        // A genuine near-singular linear-algebra failure on a finite, correctly
        // shaped input (back-solve / Cholesky/Woodbury factor that tripped on
        // rank-deficiency) is the legitimate #1273 conditioning case: it must
        // KEEP IllConditioned and stay conditioning-recoverable.
        let conditioning_messages = [
            "DeflatedArrowSolver: gauge Woodbury factor failed: matrix is not positive definite",
            "DeflatedArrowSolver: gauge back-solve: singular factor",
        ];
        for msg in conditioning_messages {
            let classified =
                OuterGradientError::classify_arrow_solver_error(msg, conditioning());
            assert!(
                matches!(classified, OuterGradientError::IllConditioned { .. }),
                "a finite, correctly-shaped near-singular failure must KEEP \
                 IllConditioned (#1451 / #1273); got {classified}"
            );
            assert!(
                classified.is_conditioning_recoverable(),
                "a genuine conditioning failure must remain conditioning-recoverable (#1273); got {classified}"
            );
            assert!(
                classified.admits_plain_solver_fallback(1.0),
                "a genuine conditioning failure at finite cost must admit the plain-solver fallback (#1273)"
            );
        }
    }
}

#[cfg(test)]
mod softmax_majorizer_active_entry_1410_tests {
    //! #1410 — the active-only softmax-entropy curvature helpers
    //! ([`super::active_softmax_gershgorin_majorizer_entry`],
    //! [`super::softmax_dense_entropy_hessian_entry`],
    //! [`super::softmax_majorizer_log_mean`]) let the compact assembly /
    //! θ-adjoint / exact-Hessian-correction paths read one `(k)` diagonal or
    //! `(k,j)` matrix entry WITHOUT materialising the full-`K` `d` vector or the
    //! `K×K` dense entropy/majorizer blocks per row — the residual per-worker
    //! `O(K)`/`O(K²)` scratch that defeated the compact `O(top_k·d)`-per-token
    //! contract.
    //!
    //! Correctness is single-sourced: these helpers MUST reproduce the
    //! `SoftmaxAssignmentSparsityPenalty` dense library routines
    //! (`psd_majorizer_abs_row_sums`, `row_psd_majorizer`, `row_dense_hessian`)
    //! BIT-FOR-BIT, because the assembled `B`, the criterion's `log|H|`, and the
    //! #1006 θ-adjoint all differentiate ONE operator. If the dense library
    //! formula ever changes, this oracle fails and forces the helpers back into
    //! sync (preventing the value↔adjoint desync the compact rewrite must not
    //! introduce).

    use crate::terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty;

    /// Deterministic, well-spread softmax logit rows (a long tail plus a few
    /// peaks) so the abs-row-sum / dense-Hessian algebra is exercised across
    /// near-zero and near-one assignment masses.
    fn logit_rows(k: usize) -> Vec<Vec<f64>> {
        let mut rows = Vec::new();
        // Row a: a few sharp peaks spread across K, deep floor elsewhere.
        let mut a = vec![-7.0_f64; k];
        for &peak in &[0usize, k / 3, 2 * k / 3, k - 1] {
            a[peak] = 5.0 + (peak as f64) * 0.001;
        }
        rows.push(a);
        // Row b: smoothly varying logits (no degenerate ties).
        let b: Vec<f64> = (0..k)
            .map(|i| ((i as f64) * 0.37).sin() * 2.0 - (i as f64) / (k as f64))
            .collect();
        rows.push(b);
        // Row c: near-uniform (entropy Hessian indefinite here — the regime the
        // Gershgorin majorizer exists for).
        rows.push(vec![0.01; k]);
        rows
    }

    #[test]
    fn active_softmax_gershgorin_matches_dense_majorizer_1410() {
        let k = 64usize;
        let temperature = 0.8_f64;
        let scale = 1.7_f64;
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        for row in logit_rows(k) {
            // Dense reference: full-K abs-row-sum diagonal `d`.
            let d_dense = penalty.psd_majorizer_abs_row_sums(&row, scale);
            // The helper consumes the softmax row `a`, not raw logits, exactly as
            // the assembly/adjoint feed it `assignments`. Build `a` the same way
            // the penalty does internally.
            let a = crate::terms::sae::assignment::softmax_row(
                ndarray::ArrayView1::from(row.as_slice()),
                temperature,
            );
            let a = a.as_slice().expect("softmax row contiguous");
            let m = super::softmax_majorizer_log_mean(a);
            for kk in 0..k {
                let got = super::active_softmax_gershgorin_majorizer_entry(a, kk, m, scale);
                assert_eq!(
                    got, d_dense[kk],
                    "active Gershgorin majorizer entry must equal the dense \
                     psd_majorizer_abs_row_sums[{kk}] BIT-FOR-BIT (single-source #1410/#1419)"
                );
            }
        }
    }

    #[test]
    fn active_softmax_dense_entropy_hessian_entry_matches_dense_block_1410() {
        let k = 48usize;
        let temperature = 1.3_f64;
        let scale = 0.9_f64;
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        for row in logit_rows(k) {
            let h_dense = penalty.row_dense_hessian(&row, scale);
            let a = crate::terms::sae::assignment::softmax_row(
                ndarray::ArrayView1::from(row.as_slice()),
                temperature,
            );
            let a = a.as_slice().expect("softmax row contiguous");
            let m = super::softmax_majorizer_log_mean(a);
            for kk in 0..k {
                for jj in 0..k {
                    let got = super::softmax_dense_entropy_hessian_entry(a, kk, jj, m, scale);
                    assert_eq!(
                        got, h_dense[[kk, jj]],
                        "active dense entropy-Hessian entry ({kk},{jj}) must equal \
                         row_dense_hessian BIT-FOR-BIT (single-source #1410/#1418)"
                    );
                }
            }
        }
    }

    #[test]
    fn active_softmax_majorizer_logit_derivative_matches_dense_1410() {
        let k = 40usize;
        let temperature = 0.7_f64;
        let scale = 1.1_f64;
        let inv_tau = 1.0 / temperature;
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        for row in logit_rows(k) {
            let a = crate::terms::sae::assignment::softmax_row(
                ndarray::ArrayView1::from(row.as_slice()),
                temperature,
            );
            let a = a.as_slice().expect("softmax row contiguous");
            let m = super::softmax_majorizer_log_mean(a);
            // Pin the active diagonal entry against the dense library derivative
            // matrix (which is diagonal: `out[[kk, kk]]`) for several `w`.
            for w in [0usize, k / 2, k - 1] {
                let dense = penalty.row_psd_majorizer_logit_derivative(&row, scale, w);
                for kk in 0..k {
                    let got = super::active_softmax_majorizer_logit_derivative_entry(
                        a, kk, w, m, scale, inv_tau,
                    );
                    assert_eq!(
                        got, dense[[kk, kk]],
                        "active majorizer logit-derivative ∂D_({kk},{kk})/∂z_{w} must equal \
                         row_psd_majorizer_logit_derivative diagonal BIT-FOR-BIT \
                         (single-source #1410/#1419/#1006)"
                    );
                }
            }
        }
    }
}
