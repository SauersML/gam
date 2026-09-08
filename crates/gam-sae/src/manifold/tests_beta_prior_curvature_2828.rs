// `manifold/mod.rs` declares this module only as
// `#[cfg(test)] mod tests_beta_prior_curvature_2828;`, so every item here is
// test-only. Stating that scope in the file makes it a claim the compiler
// enforces rather than one carried by the filename.
#![cfg(test)]
//! #2828 — the β-tier decoder priors install a PSD MAJORIZER, not their exact
//! curvature, and the exact stationarity Hessian `A = B_raw + ΔC` must undo
//! exactly that substitution and nothing else.
//!
//! Three priors are in scope, all registry-independent and all always assembled:
//! the collinearity-gated decoder repulsion (#1026/#1610/#2343), the interior
//! amplitude barrier (#2343), and the Jeffreys separation barrier
//! (#1522/#1610/#2731). Each is nonconvex, each hands the inner Newton solve a
//! PSD stand-in, and before #2828 `ΔC` had NO β leg at all — so `A_ββ` was the
//! stand-in, and `A` was not the second derivative of anything.
//!
//! `dense_exact_a_matches_finite_difference_of_the_kkt_gradient_2330`
//! (`tests_sparse_curvature_operator_2500`) is the end-to-end statement. The
//! gates here are the per-side ones, and they are deliberately independent of
//! each other:
//!
//! * the EXACT side, against a second difference of the priors' own values —
//!   the only oracle that does not go through any of the code being tested;
//! * the MAJORIZER side, against the dense `sys.hbb` the assembly writes — so a
//!   change to what is installed cannot silently leave the remainder behind;
//! * the two forms of the repulsion majorizer (dense scatter and matrix-free
//!   carrier) against each other, because the assembly picks between them by
//!   lane and `A = B + ΔC` needs `B` to be lane-invariant;
//! * the Daleckii–Krein overlap-space Hessian against a second difference of the
//!   floored spectral value, on a synthetic matrix with no SAE in sight.

use super::*;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use ndarray::{Array1, Array2};

/// The three prior values the β-tier majorizers stand in for, summed. This is
/// the scalar whose second derivative the `exact` side of
/// [`SaeManifoldTerm::decoder_prior_beta_hvp_pair`] claims to be.
fn decoder_prior_value(term: &SaeManifoldTerm) -> f64 {
    term.decoder_repulsion_value(1.0)
        + term.amplitude_barrier_value(1.0)
        + term.separation_barrier_value(1.0)
}

/// A finite-difference endpoint that keeps `anchor`'s three per-assembly frozen
/// gates. `SaeManifoldTerm::clone` drops them and every value seam re-derives
/// them from whatever state it is handed, so an endpoint built with a bare
/// `clone` differentiates a DIFFERENT objective at every step (#2828).
fn frozen_gate_endpoint(anchor: &SaeManifoldTerm) -> SaeManifoldTerm {
    let mut endpoint = anchor.clone();
    endpoint.decoder_repulsion_gate = anchor.decoder_repulsion_gate.clone();
    endpoint.barrier_coactivation_gate = anchor.barrier_coactivation_gate.clone();
    endpoint.amplitude_barrier_gate = anchor.amplitude_barrier_gate;
    endpoint.streaming_gates_frozen = true;
    endpoint
}

/// `anchor` with `step·direction` added to the flat decoder vector, gates kept.
fn beta_step(anchor: &SaeManifoldTerm, direction: &Array1<f64>, step: f64) -> SaeManifoldTerm {
    let mut moved = frozen_gate_endpoint(anchor);
    let coords = Array1::<f64>::zeros(anchor.n_obs() * anchor.assignment.row_block_dim());
    let signed = direction.mapv(|value| value * step.signum());
    moved
        .apply_newton_step(coords.view(), signed.view(), step.abs())
        .expect("beta-only finite-difference endpoint");
    moved
}

/// A term whose gates are frozen at its own state, ready to be differentiated.
fn anchored(term: &SaeManifoldTerm) -> SaeManifoldTerm {
    let mut anchor = term.clone();
    anchor.refresh_decoder_repulsion_gate();
    anchor.refresh_barrier_coactivation_gate();
    anchor.refresh_amplitude_barrier_gate();
    anchor.streaming_gates_frozen = true;
    anchor
}

/// The #2500 ThresholdGate fixture, optionally with the AMPLITUDE barrier's
/// turn-on radius forced onto the smallest atom.
///
/// At the fixture's own state the barrier is INERT — its production radius is
/// `ε² = SAE_BARRIER_ACTIVE_NORM_REL_FLOOR²·max_k‖B_k‖²_F` with
/// `SAE_BARRIER_ACTIVE_NORM_REL_FLOOR = 1e-6`, i.e. `ε² = 2.5e-13` here, so its
/// whole β Hessian measures 7.3e-12 and a gate that only ever ran on this state
/// would not be testing the amplitude leg at all. Shrinking a decoder until the
/// production radius bites is not a usable alternative: it needs
/// `‖B_1‖² ≈ 1e-12‖B_0‖²`, and a finite difference then has to take steps below
/// `1e-7` on entries of order `1e-7`.
///
/// `amplitude_radius` instead sets the frozen radius directly, which is exactly
/// what the assembly does — [`SaeManifoldTerm::amplitude_barrier_gate`] is a
/// per-assembly frozen scalar, and value, gradient, majorizer and remainder all
/// read that one field. Pinning it at the smallest atom's own `u = ‖B_k‖²_F`
/// puts that atom at `u = ε²`, the barrier's most curved point, at an O(1)
/// decoder scale.
fn amplitude_gated_fixture(
    amplitude_radius: bool,
) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let (term, target, rho) = threshold_gate_tiny_fixture(false);
    let mut anchor = anchored(&term);
    if amplitude_radius {
        let smallest = anchor
            .atoms
            .iter()
            .map(|atom| {
                atom.decoder_coefficients()
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            smallest.is_finite() && smallest > 0.0,
            "the fixture must carry a positive smallest decoder energy"
        );
        anchor.amplitude_barrier_gate = Some(smallest);
    }
    (anchor, target, rho)
}

/// A dense probe direction in β, deterministic and dense in every block.
fn probe_direction(len: usize, phase: f64) -> Array1<f64> {
    let mut v = Array1::<f64>::zeros(len);
    for idx in 0..len {
        let angle = phase + 0.37 * (idx as f64);
        v[idx] = angle.sin() + 0.5 * (2.0 * angle).cos();
    }
    let norm = v.dot(&v).sqrt();
    v.mapv_inplace(|value| value / norm);
    v
}

/// GATE 1 — the EXACT side is the second derivative of the priors' own values.
///
/// The oracle is a four-point second difference of `decoder_prior_value`, which
/// shares no code with the operator: the values come from
/// `decoder_repulsion_value` / `amplitude_barrier_value` /
/// `separation_barrier_value`, the operator from `decoder_prior_beta_hvp_pair`.
/// If the two agree column by column, `A_ββ` is the curvature of the objective
/// the line search ranks, which is the whole of #2828 item 1 on the β block.
///
/// Run on two arms. `shrink = 1` is the #2500 fixture, where the repulsion
/// (value 4.3e-1) and the separation barrier (2.7e-3) carry the curvature; at
/// `shrink = 1e-6` atom 1 sits at the amplitude barrier's turn-on radius, so
/// that leg — inert on the first arm — is the one under test.
#[test]
fn decoder_prior_exact_beta_curvature_is_the_second_derivative_of_the_prior_values_2828() {
    for amplitude_radius in [false, true] {
        let (anchor, _target, _rho) = amplitude_gated_fixture(amplitude_radius);
        let beta_dim = anchor.beta_dim();
        assert!(
            anchor.decoder_repulsion_gate.is_some(),
            "amplitude_radius={amplitude_radius}: the fixture must carry a live repulsion, or \
             this gate is vacuous"
        );
        assert_eq!(
            anchor.amplitude_barrier_value(1.0) > 1.0e-3,
            amplitude_radius,
            "amplitude_radius={amplitude_radius}: the two arms exist to differ in whether the \
             AMPLITUDE leg is live; measured value {:.6e}",
            anchor.amplitude_barrier_value(1.0)
        );

        // Four-point second difference. `h = 1e-4` balances the O(h²)
        // truncation against the O(eps/h²) cancellation of a second difference;
        // both land near 1e-8 on values of order 1e-1.
        let h = 1.0e-4_f64;
        let mut fd = Array2::<f64>::zeros((beta_dim, beta_dim));
        for col in 0..beta_dim {
            let mut e_col = Array1::<f64>::zeros(beta_dim);
            e_col[col] = 1.0;
            let plus = beta_step(&anchor, &e_col, h);
            let minus = beta_step(&anchor, &e_col, -h);
            for row in 0..beta_dim {
                let mut e_row = Array1::<f64>::zeros(beta_dim);
                e_row[row] = 1.0;
                let pp = decoder_prior_value(&beta_step(&plus, &e_row, h));
                let pm = decoder_prior_value(&beta_step(&plus, &e_row, -h));
                let mp = decoder_prior_value(&beta_step(&minus, &e_row, h));
                let mm = decoder_prior_value(&beta_step(&minus, &e_row, -h));
                fd[[row, col]] = (pp - pm - mp + mm) / (4.0 * h * h);
            }
        }
        let scale = fd.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            scale > 1.0e-3,
            "amplitude_radius={amplitude_radius}: the priors must carry MATERIAL beta curvature, \
             else this gate is a zero-vs-zero tautology; max|H_fd| = {scale:.6e}"
        );

        let mut worst = 0.0_f64;
        let mut worst_at = (0usize, 0usize);
        for col in 0..beta_dim {
            let mut e_col = Array1::<f64>::zeros(beta_dim);
            e_col[col] = 1.0;
            let (exact, _majorizer) = anchor
                .decoder_prior_beta_hvp_pair(1.0, e_col.view())
                .expect("decoder-prior beta curvature");
            for row in 0..beta_dim {
                let error = (exact[row] - fd[[row, col]]).abs();
                if error > worst {
                    worst = error;
                    worst_at = (row, col);
                }
            }
        }
        assert!(
            worst <= 1.0e-5 * scale.max(1.0),
            "amplitude_radius={amplitude_radius}: the decoder priors' EXACT beta curvature is \
             not the second \
             derivative of their own values. Worst absolute error {worst:.6e} at \
             {worst_at:?} against a scale of {scale:.6e}. `A_ββ = B_ββ + ΔC_ββ` is then \
             not the Hessian of the penalized objective and every consumer of the exact A \
             — the IndefiniteObservedInformation refusal, the +inf probe pricing, \
             ½log|A|, and the IFT adjoint's A⁻¹ — is reading the wrong operator."
        );
    }
}

/// GATE 2 — the MAJORIZER side is exactly what the assembly writes.
///
/// On the dense β-curvature lane `sys.hbb` carries the decoder-prior curvature
/// and nothing else: the data-fit Gram rides `g_blocks`, the smoothness Gram
/// rides `smooth_ops`, and no analytic registry is supplied here. So `sys.hbb`
/// IS the installed majorizer, and comparing the `majorizer` half of
/// `decoder_prior_beta_hvp_pair` against it column by column is the exact
/// statement `ΔC` needs: the remainder subtracts what was added, not a
/// re-derivation of it.
#[test]
fn decoder_prior_installed_beta_majorizer_equals_the_assembled_hbb_2828() {
    for amplitude_radius in [false, true] {
        let (mut anchor, target, rho) = amplitude_gated_fixture(amplitude_radius);
        let beta_dim = anchor.beta_dim();
        drop(
            anchor
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("dense-lane arrow assembly"),
        );
        // `assemble_arrow_schur` hands `sys.hbb` back to the term as the reusable
        // border workspace on its way out (`reclaim_border_hbb_workspace`), so the
        // assembled block is read from there rather than from the returned system.
        let installed = anchor.border_hbb_workspace.clone();
        assert_eq!(
            installed.dim(),
            (beta_dim, beta_dim),
            "amplitude_radius={amplitude_radius}: this gate is stated on the DENSE \
             beta-curvature lane, where `hbb` is materialized; a zero-width block means the \
             assembly took the matrix-free lane and the comparison below would be vacuous"
        );
        let installed_scale = installed
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            installed_scale > 1.0e-3,
            "amplitude_radius={amplitude_radius}: the assembly must install MATERIAL \
             decoder-prior curvature; max|hbb| = {installed_scale:.6e}"
        );
        let mut worst = 0.0_f64;
        let mut worst_at = (0usize, 0usize);
        for col in 0..beta_dim {
            let mut unit = Array1::<f64>::zeros(beta_dim);
            unit[col] = 1.0;
            let (_exact, majorizer) = anchor
                .decoder_prior_beta_hvp_pair(1.0, unit.view())
                .expect("decoder-prior beta majorizer");
            for row in 0..beta_dim {
                let error = (majorizer[row] - installed[[row, col]]).abs();
                if error > worst {
                    worst = error;
                    worst_at = (row, col);
                }
            }
        }
        assert!(
            worst <= 1.0e-12 * installed_scale,
            "amplitude_radius={amplitude_radius}: the majorizer `ΔC` subtracts is not the one \
             the assembly \
             installed. Worst absolute error {worst:.6e} at {worst_at:?} against a scale \
             of {installed_scale:.6e}."
        );
    }
}

/// GATE 3 — the repulsion majorizer is LANE-INVARIANT.
///
/// The assembly installs it two ways: a dense scatter into `sys.hbb`
/// (`accumulate_psd_majorizer_dense`) on the dense lane, and a matrix-free
/// `CoupledCarrierPenaltyOp` on the un-framed lanes that build no dense block
/// (#2828). `A = B_raw + ΔC` is only well posed if those are the same operator,
/// and this asserts it entry by entry.
///
/// It also pins the hole that made the carrier form necessary:
/// `add_sae_decoder_repulsion` returns early on `!dense_beta_curvature` having
/// applied only the GRADIENT, and the `deferred_factored` mark it leaves was
/// consumed on the FRAMED lane alone — so the un-framed matrix-free lane kept
/// the repulsion's force and dropped its curvature.
#[test]
fn decoder_repulsion_majorizer_is_the_same_operator_on_both_lanes_2828() {
    let (anchor, _target, _rho) = amplitude_gated_fixture(false);
    let beta_dim = anchor.beta_dim();
    let per_fit = anchor
        .live_decoder_repulsion_penalty()
        .expect("the fixture carries a live repulsion");
    let target_beta = anchor.flatten_beta();
    let rho_local = Array1::<f64>::zeros(0);
    let mut dense = Array2::<f64>::zeros((beta_dim, beta_dim));
    per_fit.accumulate_psd_majorizer_dense(target_beta.view(), rho_local.view(), 1.0, &mut dense);
    let carrier = anchor
        .decoder_repulsion_majorizer_carrier_op(1.0)
        .expect("the carrier form of the same majorizer");
    let carrier_dense = carrier.to_dense();
    assert_eq!(carrier_dense.dim(), dense.dim());
    let scale = dense.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(
        scale > 1.0e-3,
        "the repulsion must install MATERIAL curvature here; max|H| = {scale:.6e}"
    );
    let worst = dense
        .iter()
        .zip(carrier_dense.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(
        worst <= 1.0e-12 * scale,
        "the dense scatter and the matrix-free carrier are not the same repulsion \
         majorizer: max|dense − carrier| = {worst:.6e} against a scale of {scale:.6e}. The \
         assembly picks between them by lane, so `B` would depend on the lane and \
         `A = B + ΔC` would not be well defined."
    );

    // ... and the carrier really is the penalty's own PSD majorizer, not just
    // some symmetric block that happens to match the scatter.
    let probe = probe_direction(beta_dim, 0.61);
    let expected = per_fit.psd_majorizer_hvp(target_beta.view(), rho_local.view(), probe.view());
    let mut applied = vec![0.0_f64; beta_dim];
    carrier.matvec(
        probe.as_slice().expect("contiguous probe"),
        applied.as_mut_slice(),
    );
    let worst_apply = expected
        .iter()
        .zip(applied.iter())
        .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(
        worst_apply <= 1.0e-12 * scale,
        "the carrier operator does not apply `psd_majorizer_hvp`: max error \
         {worst_apply:.6e}"
    );
}

/// GATE 4 — the overlap's second derivative, against a second difference of the
/// overlap's own gradient.
///
/// `∂²o/∂B²` is the term the separation barrier replaces with the isotropic
/// `lev·I` ridge, and it is the one piece of #2828 that had to be derived by
/// hand rather than read off an existing closed form. The oracle differences the
/// overlap `o = ‖B_jB_kᵀ‖²_F/(‖B_jB_jᵀ‖_F·‖B_kB_kᵀ‖_F)` directly, so it shares
/// nothing with the implementation but the definition.
#[test]
fn rank_aware_overlap_second_derivative_matches_finite_differences_2828() {
    let (term, _target, _rho) = amplitude_gated_fixture(false);
    let bj = term.atoms[0].decoder_coefficients().clone();
    let bk = term.atoms[1].decoder_coefficients().clone();
    let p = bj.ncols();
    let (m_j, m_k) = (bj.nrows(), bk.nrows());
    let overlap = |bj: &Array2<f64>, bk: &Array2<f64>| -> f64 {
        let cross = bj.dot(&bk.t());
        let energy = cross.iter().map(|value| value * value).sum::<f64>();
        let d_j = bj
            .dot(&bj.t())
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let d_k = bk
            .dot(&bk.t())
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        energy / (d_j * d_k)
    };
    let d_j = bj
        .dot(&bj.t())
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let d_k = bk
        .dot(&bk.t())
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let o = overlap(&bj, &bk);
    assert!(
        o > 1.0e-2 && o < 1.0 - 1.0e-6,
        "the fixture pair must sit strictly inside the overlap range for this derivative \
         to be the one the barrier differentiates; o = {o:.6e}"
    );

    // Two directions: one leaning on `B_j`, one on `B_k`, so a leg that is wrong
    // in only one atom block cannot hide.
    for (label, weight_j, weight_k) in [("j-weighted", 1.0_f64, 0.25_f64), ("k-weighted", 0.25, 1.0)]
    {
        let vj = Array2::from_shape_fn((m_j, p), |(a, o_col)| {
            weight_j * (((a * p + o_col) as f64) * 0.37 + 0.11).sin()
        });
        let vk = Array2::from_shape_fn((m_k, p), |(b, o_col)| {
            weight_k * (((b * p + o_col) as f64) * 0.53 + 0.29).cos()
        });
        let (hj, hk) = SaeManifoldTerm::overlap_second_derivative_hvp(
            &bj,
            &bk,
            o,
            d_j,
            d_k,
            vj.view(),
            vk.view(),
        );
        let h = 1.0e-6_f64;
        let gradient = |bj: &Array2<f64>, bk: &Array2<f64>| -> (Array2<f64>, Array2<f64>) {
            let mut gj = Array2::<f64>::zeros((m_j, p));
            let mut gk = Array2::<f64>::zeros((m_k, p));
            for a in 0..m_j {
                for o_col in 0..p {
                    let mut plus = bj.clone();
                    plus[[a, o_col]] += h;
                    let mut minus = bj.clone();
                    minus[[a, o_col]] -= h;
                    gj[[a, o_col]] = (overlap(&plus, bk) - overlap(&minus, bk)) / (2.0 * h);
                }
            }
            for b in 0..m_k {
                for o_col in 0..p {
                    let mut plus = bk.clone();
                    plus[[b, o_col]] += h;
                    let mut minus = bk.clone();
                    minus[[b, o_col]] -= h;
                    gk[[b, o_col]] = (overlap(bj, &plus) - overlap(bj, &minus)) / (2.0 * h);
                }
            }
            (gj, gk)
        };
        let (gj_plus, gk_plus) = gradient(&(&bj + &(&vj * h)), &(&bk + &(&vk * h)));
        let (gj_minus, gk_minus) = gradient(&(&bj - &(&vj * h)), &(&bk - &(&vk * h)));
        let fd_j = (&gj_plus - &gj_minus).mapv(|value| value / (2.0 * h));
        let fd_k = (&gk_plus - &gk_minus).mapv(|value| value / (2.0 * h));
        let scale = fd_j
            .iter()
            .chain(fd_k.iter())
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            scale > 1.0,
            "({label}) the probe must excite real overlap curvature; max|FD| = {scale:.6e}"
        );
        let worst = hj
            .iter()
            .zip(fd_j.iter())
            .chain(hk.iter().zip(fd_k.iter()))
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            worst <= 1.0e-4 * scale,
            "({label}) `∂²o/∂B²·V` disagrees with a central difference of `∂o/∂B` by \
             {worst:.6e} against a scale of {scale:.6e}"
        );
    }
}

/// GATE 5 — the Daleckii–Krein overlap-space Hessian, on a synthetic spectrum.
///
/// The separation barrier's own doc argues "`F` is LINEAR in the overlaps `o_e`,
/// so the overlap-space Hessian is exactly Gauss–Newton". Linearity of `F` does
/// remove the `∂²F/∂o²` term, but the value is `−½ Σ ln m(λ_i(F))` with `m` the
/// SMOOTH spectral floor, and the second derivative of a spectral function is a
/// divided difference, not a product of resolvents. The Gauss–Newton form
/// `q q (GG + GG)` is the special case `m(λ) = λ + ε`.
///
/// This gate states the general form directly against a second difference of the
/// floored value, on a matrix with no SAE anywhere in it, at three `ε` — one
/// where the floor is inactive (`m` affine, so the two forms must AGREE), and
/// two where it bites (where they must not).
#[test]
fn floored_spectral_second_derivative_is_the_divided_difference_2828() {
    let f = ndarray::arr2(&[[1.0_f64, 0.62, 0.11], [0.62, 1.0, 0.41], [0.11, 0.41, 1.0]]);
    let direction = ndarray::arr2(&[[0.0_f64, 0.7, -0.3], [0.7, 0.0, 0.45], [-0.3, 0.45, 0.0]]);
    let value = |eps: f64, f: &Array2<f64>| -> f64 {
        let (lams, _vecs) = f.eigh(Side::Lower).expect("symmetric eigh");
        -0.5 * lams
            .iter()
            .map(|&lam| SaeManifoldTerm::barrier_spectral_m(lam, eps).ln())
            .sum::<f64>()
    };
    // `m` is affine exactly where `(λ+ε)/ε ≥ 30`, i.e. for ε SMALL against the
    // spectrum (this `F` has `λ_min ≈ 0.28`, so `ε = 1e-3` is far above the
    // knee and `m(λ) = λ + ε` there); the divided difference must then reproduce
    // the Gauss-Newton resolvent form exactly. `0.9` and `3.0` sit inside the
    // knee, where it must not.
    for (eps, floor_active) in [(1.0e-3_f64, false), (0.9, true), (3.0, true)] {
        let (lams, vecs) = f.eigh(Side::Lower).expect("symmetric eigh");
        // `E_ij = vᵢᵀ E vⱼ` in the eigenbasis.
        let e_hat = vecs.t().dot(&direction).dot(&vecs);
        let mut analytic = 0.0_f64;
        for i in 0..lams.len() {
            for j in 0..lams.len() {
                analytic += SaeManifoldTerm::barrier_spectral_f_prime_divided(lams[i], lams[j], eps)
                    * e_hat[[i, j]]
                    * e_hat[[i, j]];
            }
        }
        analytic *= -0.5;
        let h = 1.0e-4_f64;
        let fd = (value(eps, &(&f + &(&direction * h))) - 2.0 * value(eps, &f)
            + value(eps, &(&f - &(&direction * h))))
            / (h * h);
        assert!(
            (analytic - fd).abs() <= 1.0e-6 * (1.0 + fd.abs()),
            "eps={eps}: the divided-difference second derivative {analytic:.9e} disagrees \
             with a second difference of the floored value {fd:.9e}"
        );
        // The Gauss-Newton form the assembly's majorizer is built from: `½
        // tr(G E G E)` with `G = Σ (m′/m) v vᵀ`.
        let mut g = Array2::<f64>::zeros(f.dim());
        for (i, &lam) in lams.iter().enumerate() {
            let scale = SaeManifoldTerm::barrier_spectral_m_prime(lam, eps)
                / SaeManifoldTerm::barrier_spectral_m(lam, eps);
            for a in 0..f.nrows() {
                for b in 0..f.nrows() {
                    g[[a, b]] += scale * vecs[[a, i]] * vecs[[b, i]];
                }
            }
        }
        let gauss_newton = 0.5
            * g.dot(&direction)
                .dot(&g)
                .dot(&direction)
                .diag()
                .iter()
                .sum::<f64>();
        let deviation = (gauss_newton - fd).abs();
        if floor_active {
            assert!(
                deviation > 1.0e-6 * (1.0 + fd.abs()),
                "eps={eps}: this arm is supposed to be one where the smooth floor BITES, so \
                 the Gauss-Newton form must differ from the truth; it agreed to \
                 {deviation:.6e}. Without a live discrepancy here the gate would pass on an \
                 implementation that never left the affine branch."
            );
        } else {
            assert!(
                deviation <= 1.0e-6 * (1.0 + fd.abs()),
                "eps={eps}: on the affine branch of `m` the Gauss-Newton and \
                 divided-difference forms are the same operator; they differ by \
                 {deviation:.6e}"
            );
        }
    }
}

/// GATE 6 — the majorized `A_ββ` manufactured an indefiniteness that is not
/// there.
///
/// This is #2828's headline claim, stated on a fixture that was NOT built for
/// it: `obb_patchd_fixture`, the ordered-Beta–Bernoulli mode
/// `tests_logdet_adjoint_780` uses as its Patch-D arbiter, whose doc calls it
/// "positive definite at the converged mode". At its converged state the exact
/// `A` with the β leg is PD, and the SAME operator with the leg removed — which
/// is exactly the pre-#2828 `A = B_raw + ΔC_θ` — carries a negative eigenvalue
/// at −1.8e-4 against a spectral norm of 3.0e1.
///
/// So on this fixture the "converged but indefinite" verdict was an artefact of
/// the operator, not a saddle upstream, and the negative-curvature escape it
/// would have triggered would have been chasing a direction the objective does
/// not have. The gate asserts both halves, because only the pair is evidence:
/// PD alone could be a coincidence of this mode, and indefinite-without-the-leg
/// alone could be an artefact of removing curvature.
#[test]
fn the_majorized_beta_block_manufactures_a_spurious_negative_direction_2828() {
    let (mut term, target, rho) =
        crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture(0.0, -6.0);
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("the Patch-D fixture must converge to its own mode");
    let exact = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("exact A at the converged mode");
    let total_t = cache.delta_t_len();
    let k = cache.k;
    assert!(k > 0, "the fixture must carry a border block");

    // `E = B − A` on the border block IS the leg, negated, so `A + E` is the
    // pre-#2828 operator: `B_raw + ΔC_θ` with the assembly's majorizer left in
    // place on `ββ`.
    let gap = term
        .decoder_prior_majorizer_gap_border(&cache)
        .expect("decoder-prior majorization gap")
        .expect("the fixture must have a live beta-tier prior, or this gate is vacuous");
    let mut majorized = exact.clone();
    for row in 0..k {
        for col in 0..k {
            majorized[[total_t + row, total_t + col]] += gap[[row, col]];
        }
    }

    let (exact_eigs, _) = exact.eigh(Side::Lower).expect("exact spectrum");
    let (majorized_eigs, _) = majorized.eigh(Side::Lower).expect("majorized spectrum");
    let norm = exact_eigs
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    // The same relative band the criterion classifies against: below it an
    // eigenvalue is not resolved as negative at all.
    let floor = (exact.nrows() as f64) * f64::EPSILON * norm;
    let exact_min = exact_eigs.iter().copied().fold(f64::INFINITY, f64::min);
    let majorized_min = majorized_eigs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        exact_min > -floor,
        "the exact A must be positive definite at this mode (this is the fixture \
         `tests_logdet_adjoint_780` selected FOR its definiteness); min eigenvalue \
         {exact_min:.6e} against a resolution floor of {floor:.6e} and a spectral norm of \
         {norm:.6e}"
    );
    assert!(
        majorized_min < -1.0e-6 * norm,
        "the pre-#2828 operator — the same A with the assembly's beta-tier majorizer left \
         in place — must carry the negative direction this fix removes, or this gate is \
         not measuring the defect; its min eigenvalue is {majorized_min:.6e} against a \
         spectral norm of {norm:.6e}"
    );
}
