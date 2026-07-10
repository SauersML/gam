//! Cold-start seed routing for the SAE-manifold fit (issues #174, #629, #630).
//!
//! These are the closed-form seeding-policy primitives the fit entry uses
//! before the joint Arrow-Schur solve: the joint ridge-LSQ decoder seed, the
//! mean-centred residual-logit routing seed, the output-energy clustering that
//! separates periodic seed coordinates, and the EM-style alternation that
//! refines all three together. Moved here from `gam-pyffi` (issue #2236) so the
//! CLI, Rust library users, and the Python binding seed identically; the
//! binding is marshalling only.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_ata, fast_atb};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

use crate::assignment::{ibp_map_row, jumprelu_row, topk_row};

use super::{SaeAtomBasisKind, SaeManifoldTerm};

/// Build a data-driven asymmetric assignment-logit seed for a cold start
/// (issue #629). A uniform logit seed (`Array2::zeros`) is an exact symmetric
/// saddle of the joint objective whenever the atoms are exchangeable under the
/// assignment forward map: every atom carries identical responsibility, the
/// LSQ decoder init projects the same target onto every atom, and the
/// assignment update has no gradient to break the tie, so the fit never routes.
/// The tiny `random_state` jitter is too weak to escape on conditioned data.
///
/// This helper runs one EM-style M-then-E step on the seed geometry: it fits
/// each atom's decoder independently against the *full* response (each atom's
/// own seed coordinates already give it a distinct `Phi_k`), measures the
/// per-row reconstruction residual under that fit, and emits mean-centred logits
/// that prefer the atom which best explains each row. Rows that every atom
/// explains equally well land at exactly zero logits (the residual ties centre
/// to the neutral state), so the existing jitter still breaks those rare ties;
/// rows with a clear best atom get a decisive — but bounded, hence escapable by
/// the Newton refinement — head start. The mean-centring is translation-identity
/// for softmax and keeps the IBP-MAP `sigmoid(logit/τ)` gate neutral (0.5) on
/// ties instead of slamming both gates shut, so the seed is safe for both
/// assignment maps. The result is a proper responsibility seed rather than a
/// saddle.
pub fn sae_residual_seed_logits(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    gain: f64,
) -> Result<Array2<f64>, String> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let mut logits = Array2::<f64>::zeros((n_obs, k_atoms));
    if n_obs == 0 || p_out == 0 || k_atoms <= 1 {
        return Ok(logits);
    }
    if basis_values.shape()[0] != k_atoms || basis_values.shape()[1] != n_obs {
        return Err(format!(
            "sae_residual_seed_logits: basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values.shape()
        ));
    }
    let z_owned = z.to_owned();
    // Per-row residual energy after fitting each atom independently.
    let mut resid = Array2::<f64>::zeros((n_obs, k_atoms));
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        if m_k == 0 {
            // No basis columns: the atom predicts zero, so its residual is the
            // full row energy. Leave `resid` column at that value below.
            for row in 0..n_obs {
                let mut e = 0.0_f64;
                for col in 0..p_out {
                    e += z[[row, col]] * z[[row, col]];
                }
                resid[[row, atom_idx]] = e;
            }
            continue;
        }
        // Phi_k = basis_values[atom_idx, :, :m_k]  (N, m_k).
        let mut phi = Array2::<f64>::zeros((n_obs, m_k));
        for row in 0..n_obs {
            for c in 0..m_k {
                phi[[row, c]] = basis_values[[atom_idx, row, c]];
            }
        }
        let mut gram = fast_ata(&phi);
        let mut trace = 0.0_f64;
        for i in 0..m_k {
            trace += gram[[i, i]];
        }
        let jitter = (trace / m_k as f64).max(1.0).max(1.0e-12) * 1.0e-8;
        for i in 0..m_k {
            gram[[i, i]] += jitter;
        }
        let rhs = fast_atb(&phi, &z_owned);
        let factor = gram
            .cholesky(Side::Lower)
            .map_err(|err| format!("sae_residual_seed_logits: Cholesky failed: {err:?}"))?;
        let b_k = factor.solve_mat(&rhs); // (m_k, p_out)
        if !b_k.iter().all(|v| v.is_finite()) {
            return Err("sae_residual_seed_logits: non-finite LSQ solution".to_string());
        }
        let fitted = phi.dot(&b_k); // (N, p_out)
        for row in 0..n_obs {
            let mut e = 0.0_f64;
            for col in 0..p_out {
                let d = z[[row, col]] - fitted[[row, col]];
                e += d * d;
            }
            resid[[row, atom_idx]] = e;
        }
    }
    // Convert per-row residuals to mean-centred logits relative to each row's
    // own scale so the head start is dimensionless. The best atom (lowest
    // residual) gets a positive logit, the worst a negative one, and a row whose
    // atoms all explain it equally well lands at exactly zero. Normalise by the
    // row's mean residual across atoms (with a floor relative to the dataset to
    // keep near-zero-energy rows well posed) so the spread is O(gain) regardless
    // of output magnitude.
    //
    // The mean-centring is what keeps the seed safe across assignment maps.
    // Softmax is translation-invariant, so subtracting the per-row mean leaves
    // it bit-identical to the raw `-gain·r/m` form. IBP-MAP, by contrast, maps
    // each logit through an unnormalised `sigmoid(logit/τ)`: an *uncentred*
    // negative-only seed would push *every* gate below 0.5 and slam a tied row
    // shut (`sigmoid(-gain/τ)≈0`), which is worse than the neutral 0.5/0.5 state
    // the uniform saddle held. Centring restores `logit=0 ⇒ gate=0.5` on ties
    // and opens the gate (`logit>0`) only for atoms that beat the row mean.
    let mut global_mean = 0.0_f64;
    for row in 0..n_obs {
        for k in 0..k_atoms {
            global_mean += resid[[row, k]];
        }
    }
    global_mean /= (n_obs * k_atoms) as f64;
    let floor = (global_mean * 1.0e-6).max(1.0e-12);
    for row in 0..n_obs {
        let mut row_mean = 0.0_f64;
        for k in 0..k_atoms {
            row_mean += resid[[row, k]];
        }
        row_mean = (row_mean / k_atoms as f64).max(floor);
        for k in 0..k_atoms {
            logits[[row, k]] = -gain * (resid[[row, k]] - row_mean) / row_mean;
        }
    }
    Ok(logits)
}

pub fn sae_output_energy_cluster_labels(z: ArrayView2<'_, f64>, k_atoms: usize) -> Vec<usize> {
    let (n_obs, p_out) = z.dim();
    let mut labels = vec![0usize; n_obs];
    if n_obs == 0 || p_out == 0 || k_atoms <= 1 {
        return labels;
    }
    let mut features = Array2::<f64>::zeros((n_obs, p_out));
    let mut row_energy = vec![0.0_f64; n_obs];
    for row in 0..n_obs {
        let mut energy = 0.0_f64;
        for col in 0..p_out {
            let value = z[[row, col]];
            energy += value * value;
        }
        row_energy[row] = energy;
        let denom = energy.max(1.0e-12);
        for col in 0..p_out {
            let value = z[[row, col]];
            features[[row, col]] = value * value / denom;
        }
    }

    let mut centers = Array2::<f64>::zeros((k_atoms, p_out));
    let first = row_energy
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    centers.row_mut(0).assign(&features.row(first));
    let mut min_dist = vec![0.0_f64; n_obs];
    for row in 0..n_obs {
        let mut dist = 0.0_f64;
        for col in 0..p_out {
            let diff = features[[row, col]] - centers[[0, col]];
            dist += diff * diff;
        }
        min_dist[row] = dist;
    }
    for atom_idx in 1..k_atoms {
        let next = min_dist
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        centers.row_mut(atom_idx).assign(&features.row(next));
        for row in 0..n_obs {
            let mut dist = 0.0_f64;
            for col in 0..p_out {
                let diff = features[[row, col]] - centers[[atom_idx, col]];
                dist += diff * diff;
            }
            if dist < min_dist[row] {
                min_dist[row] = dist;
            }
        }
    }

    for _ in 0..20 {
        let mut changed = false;
        for row in 0..n_obs {
            let mut best_atom = 0usize;
            let mut best_dist = f64::INFINITY;
            for atom_idx in 0..k_atoms {
                let mut dist = 0.0_f64;
                for col in 0..p_out {
                    let diff = features[[row, col]] - centers[[atom_idx, col]];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_atom = atom_idx;
                }
            }
            if labels[row] != best_atom {
                labels[row] = best_atom;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        centers.fill(0.0);
        let mut counts = vec![0usize; k_atoms];
        for row in 0..n_obs {
            let atom_idx = labels[row];
            counts[atom_idx] += 1;
            for col in 0..p_out {
                centers[[atom_idx, col]] += features[[row, col]];
            }
        }
        for atom_idx in 0..k_atoms {
            if counts[atom_idx] == 0 {
                let row = atom_idx % n_obs;
                centers.row_mut(atom_idx).assign(&features.row(row));
                continue;
            }
            let inv = 1.0 / counts[atom_idx] as f64;
            for col in 0..p_out {
                centers[[atom_idx, col]] *= inv;
            }
        }
    }
    labels
}

pub fn sae_refine_periodic_seed_coords_by_cluster(
    z: ArrayView2<'_, f64>,
    atom_kinds: &[SaeAtomBasisKind],
    labels: &[usize],
    seed_coords: &mut Array3<f64>,
) -> Result<(), String> {
    let (n_obs, p_out) = z.dim();
    let k_atoms = atom_kinds.len();
    if labels.len() != n_obs {
        return Err(format!(
            "sae_refine_periodic_seed_coords_by_cluster: labels length {} must equal n_obs={n_obs}",
            labels.len()
        ));
    }
    if n_obs < 2 || p_out < 2 || k_atoms <= 1 {
        return Ok(());
    }
    for (atom_idx, kind) in atom_kinds.iter().enumerate() {
        if !matches!(kind, SaeAtomBasisKind::Periodic) {
            continue;
        }
        let rows: Vec<usize> = (0..n_obs).filter(|&row| labels[row] == atom_idx).collect();
        if rows.len() < 2 {
            continue;
        }
        let mut mean = Array1::<f64>::zeros(p_out);
        for &row in &rows {
            for col in 0..p_out {
                mean[col] += z[[row, col]];
            }
        }
        let inv_count = 1.0 / rows.len() as f64;
        for col in 0..p_out {
            mean[col] *= inv_count;
        }

        let mut local = Array2::<f64>::zeros((rows.len(), p_out));
        for (out_row, &src_row) in rows.iter().enumerate() {
            for col in 0..p_out {
                local[[out_row, col]] = z[[src_row, col]] - mean[col];
            }
        }
        let (_u_opt, _s_vals, vt_opt) = local.svd(false, true).map_err(|err| {
            format!("sae_refine_periodic_seed_coords_by_cluster: SVD failed: {err:?}")
        })?;
        let vt = vt_opt.ok_or_else(|| {
            "sae_refine_periodic_seed_coords_by_cluster: SVD returned no Vt".to_string()
        })?;
        if vt.nrows() < 2 {
            continue;
        }
        let pc1 = vt.row(0);
        let pc2 = vt.row(1);
        let two_pi = std::f64::consts::TAU;
        for row in 0..n_obs {
            let mut a = 0.0_f64;
            let mut b = 0.0_f64;
            for col in 0..p_out {
                let centered = z[[row, col]] - mean[col];
                a += centered * pc1[col];
                b += centered * pc2[col];
            }
            let phase = b.atan2(a) / two_pi;
            seed_coords[[atom_idx, row, 0]] = phase - phase.floor();
        }
    }
    Ok(())
}

/// Seed each atom's decoder coefficient block via a joint ridge-regularized
/// least-squares projection of `Z` onto the atom design `[a_init * Phi_1, ...,
/// a_init * Phi_K]`, where `a_init` is the assignment map that the inner Newton
/// driver will produce at iteration 0 from the supplied `initial_logits`.
/// IBP-MAP uses the base `alpha` because learnable-alpha fits start with
/// `rho0 = 0`, and JumpReLU uses the configured hard threshold.
///
/// Zero-initialised decoder coefficients leave the joint-fit Arrow-Schur
/// system in a degenerate fixed point on multi-atom configurations: the
/// data-fit Jacobian, the assignment-weighted decoder gradient, and the
/// sparsity-prior gradient cannot all be zero simultaneously, but the
/// assignment prior (IBP-MAP stick-breaking or softmax entropy) is the only
/// term with a non-zero gradient at iter 0. The optimizer then collapses the
/// assignments to zero before any data signal has accumulated, even on
/// trivially-separable signals such as the K=2 periodic torus reproducer in
/// issue #174. Seeding with a closed-form LSQ projection eliminates the
/// degeneracy: at iter 0 the residual already carries the data information
/// the atoms need, so the assignment update has both a sparsity-prior pull
/// and a data-fit push to balance against.
///
/// Returns the padded `(K, M_max, p_out)` decoder array directly.
pub fn sae_decoder_lsq_init(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    initial_logits: ArrayView2<'_, f64>,
    assignment_kind: &str,
    alpha: f64,
    tau: f64,
    jumprelu_threshold: f64,
    top_k: Option<usize>,
) -> Result<Array3<f64>, String> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((k_atoms, m_max, p_out));
    if n_obs == 0 || p_out == 0 || k_atoms == 0 {
        return Ok(out);
    }
    if basis_values.shape()[0] != k_atoms || basis_values.shape()[1] != n_obs {
        return Err(format!(
            "sae_decoder_lsq_init: basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values.shape()
        ));
    }
    if initial_logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "sae_decoder_lsq_init: initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
            initial_logits.dim()
        ));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(format!(
            "sae_decoder_lsq_init: tau must be finite and positive; got {tau}"
        ));
    }
    // Compute per-row, per-atom assignment weight a_init that matches the
    // forward map of `assignment_kind` evaluated at `initial_logits`.
    let mut a_init = Array2::<f64>::zeros((n_obs, k_atoms));
    match assignment_kind {
        "softmax" => {
            if let Some(k_top) = top_k {
                if k_top == 0 || k_top > k_atoms {
                    return Err(format!(
                        "sae_decoder_lsq_init: top_k must satisfy 1 <= top_k <= k_atoms={k_atoms}; got {k_top}"
                    ));
                }
            }
            let inv_tau = 1.0 / tau;
            for row in 0..n_obs {
                let mut max_logit = f64::NEG_INFINITY;
                for k in 0..k_atoms {
                    let v = initial_logits[[row, k]];
                    if v > max_logit {
                        max_logit = v;
                    }
                }
                let mut sum = 0.0_f64;
                let mut buf = vec![0.0_f64; k_atoms];
                for k in 0..k_atoms {
                    let v = ((initial_logits[[row, k]] - max_logit) * inv_tau).exp();
                    buf[k] = v;
                    sum += v;
                }
                if sum > 0.0 && sum.is_finite() {
                    for k in 0..k_atoms {
                        a_init[[row, k]] = buf[k] / sum;
                    }
                }
                if let Some(k_top) = top_k {
                    if k_top < k_atoms {
                        let mut paired: Vec<(f64, usize)> =
                            (0..k_atoms).map(|k| (a_init[[row, k]], k)).collect();
                        let cmp = |a: &(f64, usize), b: &(f64, usize)| {
                            b.0.partial_cmp(&a.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(a.1.cmp(&b.1))
                        };
                        paired.select_nth_unstable_by(k_top - 1, cmp);
                        let mut keep = vec![false; k_atoms];
                        for &(_, atom_idx) in paired.iter().take(k_top) {
                            keep[atom_idx] = true;
                        }
                        let kept_sum: f64 = (0..k_atoms)
                            .filter(|&atom_idx| keep[atom_idx])
                            .map(|atom_idx| a_init[[row, atom_idx]])
                            .sum();
                        if !(kept_sum.is_finite() && kept_sum > 0.0) {
                            return Err(format!(
                                "sae_decoder_lsq_init: top_k softmax projection has non-positive kept mass on row {row}"
                            ));
                        }
                        for atom_idx in 0..k_atoms {
                            a_init[[row, atom_idx]] = if keep[atom_idx] {
                                a_init[[row, atom_idx]] / kept_sum
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
        "ibp_map" => {
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(format!(
                    "sae_decoder_lsq_init: alpha must be finite and positive for IBP-MAP; got {alpha}"
                ));
            }
            // Use the base alpha here. In learnable-alpha fits the first rho
            // coordinate starts at zero, so alpha_eff = alpha at initialization.
            for row in 0..n_obs {
                let weights = ibp_map_row(initial_logits.row(row), tau);
                for k in 0..k_atoms {
                    a_init[[row, k]] = weights[k];
                }
            }
        }
        // #1777 canonical token for the hard-sigmoid gate.
        "threshold_gate" => {
            if !jumprelu_threshold.is_finite() {
                return Err(format!(
                    "sae_decoder_lsq_init: jumprelu_threshold must be finite; got {jumprelu_threshold}"
                ));
            }
            for row in 0..n_obs {
                let weights = jumprelu_row(initial_logits.row(row), tau, jumprelu_threshold);
                for k in 0..k_atoms {
                    a_init[[row, k]] = weights[k];
                }
            }
        }
        // #1026 — hard top-`k` support gate. The forward map at `initial_logits`
        // is exactly `topk_row`: gate 1.0 on the `k_top` largest logits (ties
        // toward the lower atom index), 0 elsewhere. Reusing the production
        // helper keeps the LSQ seed bit-consistent with the fit's gate.
        "topk" => {
            let k_top = top_k.ok_or_else(|| {
                "sae_decoder_lsq_init: assignment_kind 'topk' requires the top_k \
                 argument (the fixed per-row support size)"
                    .to_string()
            })?;
            if k_top == 0 || k_top > k_atoms {
                return Err(format!(
                    "sae_decoder_lsq_init: top_k must satisfy 1 <= top_k <= k_atoms={k_atoms}; got {k_top}"
                ));
            }
            for row in 0..n_obs {
                let weights = topk_row(initial_logits.row(row), k_top);
                for k in 0..k_atoms {
                    a_init[[row, k]] = weights[k];
                }
            }
        }
        other => {
            return Err(format!(
                "sae_decoder_lsq_init: unsupported assignment_kind {other:?}"
            ));
        }
    }
    // Build joint design X = [a_init[:,0] * Phi_1 | ... | a_init[:,K-1] * Phi_K]
    // with column count M_total = sum_k basis_sizes[k]. If every atom has zero
    // weight on a row, that row contributes nothing — but with all the
    // supported initial logits we use, a_init has at least one non-zero
    // column per row. Solve (X^T X + ridge I) B = X^T Z, then split.
    let offsets: Vec<usize> = {
        let mut acc = 0usize;
        let mut v = Vec::with_capacity(k_atoms + 1);
        v.push(0);
        for &m in basis_sizes {
            acc += m;
            v.push(acc);
        }
        v
    };
    let m_total = offsets[k_atoms];
    if m_total == 0 {
        return Ok(out);
    }
    let mut x = Array2::<f64>::zeros((n_obs, m_total));
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let off = offsets[atom_idx];
        for row in 0..n_obs {
            let w = a_init[[row, atom_idx]];
            if w == 0.0 {
                continue;
            }
            for basis_col in 0..m_k {
                x[[row, off + basis_col]] = w * basis_values[[atom_idx, row, basis_col]];
            }
        }
    }
    // Symmetric normal-equations matrix and rhs.
    let mut xtx = fast_ata(&x);
    // Diagonal Tikhonov ridge for the seed projection (issue #671 multi-atom
    // conditioning). The cold multi-atom seed places near-identical coordinates
    // on every atom (the periodic seed shares the leading principal component
    // across atoms), so the joint design's per-atom column blocks are nearly
    // collinear and `X^T X` is severely ill-conditioned. A tiny mean-relative
    // ridge (the historical `mean_diag * 1e-8`) leaves the near-null directions
    // unregularized, producing decoder coefficients of order 1e5; the
    // DecoderIncoherence penalty's gradient is cubic in `B`, so those seeds blow
    // the joint solver up by ~1e15. We instead anchor the ridge to the SPECTRAL
    // scale (the maximum diagonal, an upper bound on the largest eigenvalue)
    // with a larger relative floor. This bounds the seed solution norm by
    // roughly `||X^T Z|| / ridge` while leaving well-conditioned designs
    // essentially unchanged (the ridge stays negligible against the signal
    // eigenvalues there). Conditioning the seed is correct here: the inner
    // data-fit Newton step refines `B` from a sane, bounded starting point
    // rather than a pathological one.
    let mut trace = 0.0_f64;
    let mut max_diag = 0.0_f64;
    for i in 0..m_total {
        let d = xtx[[i, i]];
        trace += d;
        if d > max_diag {
            max_diag = d;
        }
    }
    let mean_diag = (trace / m_total as f64).max(0.0);
    // Spectral-scale ridge: tie the floor to the largest diagonal so collinear
    // column blocks (small eigenvalues) are damped relative to the design's
    // dominant scale, not its average. `1e-4` is large enough to keep the seed
    // coefficient norm bounded under near-duplicate atoms yet small enough that
    // a well-conditioned design recovers essentially the unregularized LSQ fit.
    let spectral_scale = max_diag.max(mean_diag).max(1.0e-12);
    let jitter = spectral_scale * 1.0e-4;
    for i in 0..m_total {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, &z.to_owned());
    let factor = xtx
        .cholesky(Side::Lower)
        .map_err(|err| format!("sae_decoder_lsq_init: Cholesky failed: {err:?}"))?;
    let b_joint = factor.solve_mat(&xtz);
    if !b_joint.iter().all(|v| v.is_finite()) {
        return Err("sae_decoder_lsq_init: non-finite LSQ solution".to_string());
    }
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let off = offsets[atom_idx];
        for basis_col in 0..m_k {
            for out_col in 0..p_out {
                out[[atom_idx, basis_col, out_col]] = b_joint[[off + basis_col, out_col]];
            }
        }
    }
    Ok(out)
}

/// EM-style seed refinement that resolves the cold-start routing collapse of
/// the training fit (issues #629, #630) before the joint Arrow-Schur solve.
///
/// The cold residual-logit seed ([`sae_residual_seed_logits`]) is computed at
/// the cold latent coordinates (the per-atom PCA/atan2 seed). Those coordinates
/// are *shared* across atoms (the seed places the same leading component on
/// every atom), so each atom's independent LSQ fit against the full response is
/// equally mediocre on every row: the per-row residual barely separates the
/// atoms and the logit seed stays near the symmetric saddle the random jitter
/// cannot escape. The joint solver then never routes (the planted disjoint
/// atoms collapse to a near-uniform mixture, negative R²).
///
/// This is the exact dual of the frozen-decoder OOS fix (#628): there, each row
/// is placed in the correct latent basin by projecting it onto every atom's
/// *known* decoder through the complete rank-1 Fourier stationary set
/// ([`SaeManifoldTerm::seed_coords_by_decoder_projection`]). Here the decoder is
/// being *learned*, so we alternate the two exact steps the OOS path and the
/// existing seed already provide:
///
/// 1. **Coordinate E-step** — project each row's rank-1 Fourier latent onto the
///    current decoder by enumerating every stationary point. This separates the
///    atoms' geometries: a row generated from atom `k` moves to the coordinate
///    where atom `k` reconstructs it well, while the off-atoms move to wherever
///    their current decoder is least wrong on that row.
/// 2. **Decoder M-step** — refit every atom's decoder by the same weighted joint
///    LSQ used for the cold init ([`sae_decoder_lsq_init`]), now at the
///    *separated* coordinates, so each atom's block specializes toward the rows
///    it actually explains.
/// 3. **Routing seed** — recompute the mean-centred residual logits
///    ([`sae_residual_seed_logits`]) at the refined geometry. With the atoms now
///    geometrically distinct, the per-row residual is decisive and the seed is
///    one-hot for the planted disjoint oracle.
///
/// A handful of rounds converges this alternation for separable atoms while
/// leaving an already-routed warm fit at its fixed point (the projection finds
/// the same global coordinate, the LSQ recovers the same decoder). Unbounded or
/// basis-linear atoms (Duchon / Euclidean patch) are left untouched by step 1.
/// Compact multivariate charts are rejected because this engine does not yet
/// carry the interval extension needed to enumerate their complete stationary
/// sets.
///
/// Only invoked for cold-start multi-atom softmax / IBP-MAP fits; JumpReLU keeps
/// its margin-above-threshold gate seed and warm starts are respected verbatim.
pub fn sae_em_refine_routing_seed(
    term: &mut SaeManifoldTerm,
    z: ArrayView2<'_, f64>,
    basis_sizes: &[usize],
    assignment_kind: &str,
    alpha: f64,
    tau: f64,
    jumprelu_threshold: f64,
    random_state: u64,
    top_k: Option<usize>,
) -> Result<(), String> {
    const SAE_SEED_REFINE_ROUNDS: usize = 4;
    const SAE_RESIDUAL_SEED_GAIN: f64 = 4.0;
    // Same tiny seed-keyed logit jitter the cold-start path applies (issue
    // #178): the refined residual logits are decisive (O(gain)), so this 1e-3
    // perturbation does not change which atom wins, but it keeps distinct
    // `random_state` values on distinct inner Newton trajectories and fixed
    // seeds bit-identical. Without it, the deterministic EM seed would erase
    // the seed-dependence the cold-start jitter installed upstream.
    const SAE_RANDOM_STATE_LOGIT_JITTER: f64 = 1.0e-3;
    let k_atoms = basis_sizes.len();
    let n_obs = z.nrows();
    if k_atoms <= 1 || n_obs == 0 {
        return Ok(());
    }
    let m_max = basis_sizes.iter().copied().max().unwrap_or(0);
    if m_max == 0 {
        return Ok(());
    }
    for _ in 0..SAE_SEED_REFINE_ROUNDS {
        // 1. Coordinate E-step: project each row onto the current decoder.
        term.seed_coords_by_decoder_projection(z)?;
        // Snapshot the refreshed per-atom basis `Φ_k(t_k)` as a padded
        // (K, N, m_max) stack for the closed-form seed helpers.
        let mut basis3 = Array3::<f64>::zeros((k_atoms, n_obs, m_max));
        for atom_idx in 0..k_atoms {
            let phi = &term.atoms[atom_idx].basis_values;
            let m_k = basis_sizes[atom_idx];
            if phi.dim() != (n_obs, m_k) {
                return Err(format!(
                    "sae_em_refine_routing_seed: atom {atom_idx} basis is {:?}, expected ({n_obs}, {m_k})",
                    phi.dim()
                ));
            }
            for row in 0..n_obs {
                for c in 0..m_k {
                    basis3[[atom_idx, row, c]] = phi[[row, c]];
                }
            }
        }
        // 2. Decoder M-step: weighted joint LSQ at the refined coordinates,
        //    using the current routing as the responsibility weights.
        let decoder = sae_decoder_lsq_init(
            basis3.view(),
            basis_sizes,
            z,
            term.assignment.logits.view(),
            assignment_kind,
            alpha,
            tau,
            jumprelu_threshold,
            top_k,
        )?;
        for atom_idx in 0..k_atoms {
            let m_k = basis_sizes[atom_idx];
            let p_out = term.atoms[atom_idx].decoder_coefficients.ncols();
            let dst = &mut term.atoms[atom_idx].decoder_coefficients;
            for c in 0..m_k {
                for out_col in 0..p_out {
                    dst[[c, out_col]] = decoder[[atom_idx, c, out_col]];
                }
            }
        }
        // 3. Routing seed: mean-centred residual logits at the refined geometry.
        let logits =
            sae_residual_seed_logits(basis3.view(), basis_sizes, z, SAE_RESIDUAL_SEED_GAIN)?;
        term.assignment.logits.assign(&logits);
    }
    // Re-apply the seed-keyed jitter the deterministic refinement above erased,
    // so `random_state` keeps perturbing the inner Newton trajectory (#178).
    let mut state = random_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for row in 0..n_obs {
        for atom_idx in 0..k_atoms {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map top 53 bits to a double in [0, 1), then to [-1, 1).
            let u = ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
            let signed = 2.0 * u - 1.0;
            term.assignment.logits[[row, atom_idx]] += SAE_RANDOM_STATE_LOGIT_JITTER * signed;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for issue #174: the joint LSQ seed for K=2 IBP-MAP
    /// must produce a non-zero decoder and a residual smaller than the
    /// trivial zero-decoder baseline. Without this seed the joint Newton
    /// driver collapses A → 0 before any data signal accumulates.
    #[test]
    fn sae_decoder_lsq_init_produces_nontrivial_seed() {
        use ndarray::Array3;
        let n = 50usize;
        let p = 4usize;
        let k = 2usize;
        let m = 3usize;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            // Every output column is a linear combination of {1, sin a, cos a} —
            // exactly the column space the 3-column periodic seed basis spans. A
            // second-harmonic component (sin 2a / cos 2a) is orthogonal to that
            // basis over the full period and so is unreachable by any decoder built
            // on it; planting it would cap the achievable R² at the first-harmonic
            // energy fraction (0.5 here) and the "explain most of the signal"
            // assertion below could never hold. The seed must be judged on signal
            // it can actually represent.
            z[[i, 0]] = a.sin();
            z[[i, 1]] = a.cos();
            z[[i, 2]] = 0.6 * a.sin() - 0.4 * a.cos();
            z[[i, 3]] = 0.25 + 0.5 * a.cos();
        }
        // Build padded basis_values (K, N, M_max=m).
        let mut basis = Array3::<f64>::zeros((k, n, m));
        for atom_idx in 0..k {
            let shift = (atom_idx as f64) * 0.21;
            for i in 0..n {
                let a = 2.0 * std::f64::consts::PI * ((i as f64) / (n as f64) + shift);
                basis[[atom_idx, i, 0]] = 1.0;
                basis[[atom_idx, i, 1]] = a.sin();
                basis[[atom_idx, i, 2]] = a.cos();
            }
        }
        let basis_sizes = vec![m; k];
        let logits = Array2::<f64>::zeros((n, k));
        let decoder = sae_decoder_lsq_init(
            basis.view(),
            &basis_sizes,
            z.view(),
            logits.view(),
            "ibp_map",
            1.0, // alpha (IBP concentration; canonical default)
            0.7, // tau
            0.0, // jumprelu_threshold (unused for ibp_map)
            None,
        )
        .expect("LSQ seed must succeed");
        assert_eq!(decoder.shape(), &[k, m, p]);
        let mut max_abs = 0.0_f64;
        for v in decoder.iter() {
            assert!(v.is_finite());
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
        assert!(
            max_abs > 1.0e-3,
            "LSQ-seeded decoder should be non-trivial; max |B| = {max_abs:.6}"
        );

        // The seeded reconstruction must explain most of Z under the SAME forward
        // map the joint LSQ solved against: fitted[i,:] = Σ_k a_k · Phi_k[i,:] · B_k
        // where a_k is the IBP-MAP activation of the initial (all-zero) logits. For
        // zero logits the posterior-mean Bernoulli gate is σ(0) = 0.5. Ordered
        // stick-breaking shrinkage is scored by the IBP prior, not multiplied
        // into the reconstruction a second time.
        // Reconstructing with the true per-atom weights (rather than an imagined
        // uniform gate) is what makes this a faithful check of the LSQ seed: the
        // solver's design columns are a_k · Phi_k, so the fit it returns is only
        // meaningful when scored back through the same a_k.
        let a_init = ibp_map_row(
            ndarray::Array1::<f64>::zeros(k).view(),
            0.7, // tau (matches the sae_decoder_lsq_init call above)
        );
        let mut fitted = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                let mut acc = 0.0;
                for atom_idx in 0..k {
                    let mut atom_out = 0.0;
                    for col in 0..m {
                        atom_out += basis[[atom_idx, i, col]] * decoder[[atom_idx, col, j]];
                    }
                    acc += a_init[atom_idx] * atom_out;
                }
                fitted[[i, j]] = acc;
            }
        }
        let mut ssr = 0.0;
        let mut sst = 0.0;
        for i in 0..n {
            for j in 0..p {
                let r = z[[i, j]] - fitted[[i, j]];
                ssr += r * r;
                sst += z[[i, j]] * z[[i, j]];
            }
        }
        let r2 = 1.0 - ssr / sst.max(1.0e-12);
        assert!(
            r2 > 0.5,
            "LSQ-seeded iter-0 reconstruction R² = {r2:.4} should explain most of the signal"
        );
    }

    /// Regression test for issue #629: the cold-start residual seed must break
    /// the symmetric saddle of a uniform logit init by preferring, per row, the
    /// atom whose seed geometry best reconstructs that row. Planted: two
    /// periodic atoms with distinct seed frequencies driving disjoint output
    /// blocks with known one-hot routing. The seed logits must (a) not be uniform
    /// and (b) argmax-route most rows to their generating atom.
    #[test]
    fn sae_residual_seed_logits_breaks_symmetry_and_routes() {
        use ndarray::Array3;
        let n = 64usize;
        let p = 4usize;
        let k = 2usize;
        let m = 3usize;
        let two_pi = std::f64::consts::TAU;
        // Distinct seed *frequency* per atom. A phase shift alone leaves the
        // {1, sin, cos} column space invariant — sin/cos of a shifted argument are
        // linear combinations of the unshifted pair — so two phase-shifted periodic
        // atoms would span the identical subspace, the independent per-atom LSQ fits
        // would produce bit-identical residuals, and the residual seed could not
        // tell them apart (every logit collapses to exactly zero). Distinct
        // harmonics give the atoms genuinely different geometries, so a row's
        // generating atom reconstructs it strictly better than the off-atom whose
        // basis cannot represent that frequency at all.
        let harmonic = [1.0_f64, 2.0_f64];
        // Deterministic pseudo-random latent + balanced shuffled routing.
        let mut t = vec![0.0_f64; n];
        let mut assign = vec![0usize; n];
        let mut state = 0x1234_5678_9abc_def0_u64;
        for i in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            t[i] = ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000);
            assign[i] = if i < n / 2 { 0 } else { 1 };
        }
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            assign.swap(i, j);
        }
        // Per-atom seed basis (N, m) padded into (K, N, m).
        let mut basis = Array3::<f64>::zeros((k, n, m));
        for atom_idx in 0..k {
            for i in 0..n {
                let a = two_pi * harmonic[atom_idx] * t[i];
                basis[[atom_idx, i, 0]] = 1.0;
                basis[[atom_idx, i, 1]] = a.sin();
                basis[[atom_idx, i, 2]] = a.cos();
            }
        }
        // Disjoint decoder blocks: atom 0 -> cols [0,1], atom 1 -> cols [2,3].
        let mut blocks = vec![Array2::<f64>::zeros((m, p)); k];
        blocks[0][[1, 0]] = 1.5;
        blocks[0][[2, 1]] = -1.2;
        blocks[1][[1, 2]] = 1.3;
        blocks[1][[2, 3]] = 0.9;
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let kk = assign[i];
            for j in 0..p {
                let mut acc = 0.0;
                for col in 0..m {
                    acc += basis[[kk, i, col]] * blocks[kk][[col, j]];
                }
                z[[i, j]] = acc;
            }
        }
        let basis_sizes = vec![m; k];
        let logits = sae_residual_seed_logits(basis.view(), &basis_sizes, z.view(), 4.0)
            .expect("residual seed must succeed");
        assert_eq!(logits.shape(), &[n, k]);
        assert!(logits.iter().all(|v| v.is_finite()));

        // (a) Symmetry must be broken: at least one row has a non-trivial gap.
        let max_gap = (0..n)
            .map(|i| (logits[[i, 0]] - logits[[i, 1]]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_gap > 0.3,
            "residual seed left a near-symmetric logit field (max gap {max_gap:.4}); \
                 the uniform saddle would not be escaped"
        );

        // (b) The seed must route most rows to their generating atom, up to
        // the trivial atom-label permutation.
        let mut acc_direct = 0usize;
        for i in 0..n {
            let winner = if logits[[i, 0]] >= logits[[i, 1]] {
                0
            } else {
                1
            };
            if winner == assign[i] {
                acc_direct += 1;
            }
        }
        let acc = (acc_direct.max(n - acc_direct)) as f64 / n as f64;
        assert!(
            acc >= 0.9,
            "residual seed routing accuracy {acc:.3} (up to permutation) is too low; \
                 the E-step seed should recover the planted one-hot assignment"
        );
    }
}
