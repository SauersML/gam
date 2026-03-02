use super::*;

impl<'a> RemlState<'a> {
    pub(super) fn dense_projected_exact_eligible(
        n_obs: usize,
        eff_rank: usize,
        k_count: usize,
    ) -> bool {
        if eff_rank == 0 || eff_rank > 1024 {
            return false;
        }
        let work_n_r2_k = (n_obs as u128)
            .saturating_mul(eff_rank as u128)
            .saturating_mul(eff_rank as u128)
            .saturating_mul(k_count as u128);
        let work_r2_k2 = (eff_rank as u128)
            .saturating_mul(eff_rank as u128)
            .saturating_mul(k_count as u128)
            .saturating_mul(k_count as u128);
        work_n_r2_k <= 1_000_000_000 && work_r2_k2 <= 100_000_000
    }

    pub(super) fn dense_projected_tk(
        z_mat: &Array2<f64>,
        w_pos: &Array2<f64>,
        r_k: &Array2<f64>,
        lambda_k: f64,
        c_weighted_u_k: &Array1<f64>,
    ) -> Array2<f64> {
        // T_k = W' H_k W
        //     = lambda_k (R_k W)' (R_k W)
        //       + Z' diag(c ⊙ u_k) Z,
        // where Z = XW.
        let rk_w = fast_ab(r_k, w_pos);
        let mut t_k = fast_ata(&rk_w);
        t_k.mapv_inplace(|v| v * lambda_k);

        let mut z_weighted = z_mat.clone();
        for i in 0..z_weighted.nrows() {
            let weight = c_weighted_u_k[i];
            for j in 0..z_weighted.ncols() {
                z_weighted[[i, j]] *= weight;
            }
        }
        t_k += &fast_atb(z_mat, &z_weighted);
        t_k
    }

    pub(super) fn dense_projected_trace_hinv_hkl(
        z_mat: &Array2<f64>,
        w_pos: &Array2<f64>,
        r_k: Option<&Array2<f64>>,
        lambda_k: f64,
        diag_kl: &Array1<f64>,
    ) -> f64 {
        // tr(H_+^dagger H_{k,l}) = tr(W' H_{k,l} W)
        //                        = tr(T_{k,l}),
        // where
        //   T_{k,l} = delta_{k,l} lambda_k (R_k W)'(R_k W)
        //             + Z' diag(diag_kl) Z.
        let mut trace = 0.0_f64;
        if let Some(r_k) = r_k {
            let rk_w = fast_ab(r_k, w_pos);
            let penalty_part = fast_ata(&rk_w);
            trace += lambda_k * penalty_part.diag().sum();
        }

        for j in 0..z_mat.ncols() {
            let mut quad = 0.0_f64;
            for i in 0..z_mat.nrows() {
                let zij = z_mat[[i, j]];
                quad += diag_kl[i] * zij * zij;
            }
            trace += quad;
        }
        trace
    }

    pub(super) fn dense_projected_trace_quadratic(t_k: &Array2<f64>, t_l: &Array2<f64>) -> f64 {
        // tr(H_+^dagger H_k H_+^dagger H_l) = tr(T_k T_l)
        Self::trace_product(t_k, t_l)
    }

    pub(super) fn select_trace_backend(n_obs: usize, p_dim: usize, k_count: usize) -> TraceBackend {
        // Workload-aware policy driven by (n, p, K):
        // - Exact for moderate total complexity.
        // - Hutchinson/Hutch++ as n·p·K and p²·K² costs grow.
        //
        // Note: this backend switch currently lives inside the dense
        // transformed REML Hessian path below. Replacing the stochastic
        // branch with selected inversion is only meaningful after the trace
        // computation is moved onto a sparse/banded factorization of the
        // original penalized system; it is not effective while "exact"
        // still means forming dense Array2 contractions/inverses.
        //
        // The bottleneck term is the second-order logdet contribution
        //   L_{k,l} = 0.5 [ -tr(H^{-1} H_l H^{-1} H_k) + tr(H^{-1} H_{kl}) ].
        // In the current transformed-coordinate implementation, the "exact"
        // backend forms dense solves/contractions to evaluate these traces,
        // while Hutchinson/Hutch++ estimates them from probe identities
        //   tr(A) = E[z' A z],   z_i in {+1,-1}.
        // Selected inversion would change the cost model only if H and the
        // derivative matrices are represented on a sparse pattern where the
        // needed inverse entries remain local.
        //
        // Proxies:
        //   w_npk   ~ n*p*K   (X/Xᵀ + diagonal contractions)
        //   w_pk2   ~ p*K²    (pairwise rho-Hessian assembly)
        let k = k_count.max(1);
        let w_npk = (n_obs as u128)
            .saturating_mul(p_dim as u128)
            .saturating_mul(k as u128);
        let w_pk2 = (p_dim as u128).saturating_mul((k as u128).saturating_mul(k as u128));

        if p_dim <= 700 && k <= 20 && w_npk <= 220_000_000 && w_pk2 <= 20_000_000 {
            return TraceBackend::Exact;
        }

        let very_large = p_dim >= 1_800 || k >= 28 || w_npk >= 1_100_000_000 || w_pk2 >= 85_000_000;
        if very_large {
            let sketch = if p_dim >= 3_500 || w_npk >= 2_500_000_000 {
                12
            } else {
                8
            };
            let probes = if k >= 36 || w_pk2 >= 150_000_000 {
                28
            } else {
                22
            };
            return TraceBackend::HutchPP { probes, sketch };
        }

        let probes = if w_npk >= 700_000_000 || k >= 24 {
            34
        } else if w_npk >= 350_000_000 {
            28
        } else {
            22
        };
        TraceBackend::Hutchinson { probes }
    }

    #[inline]
    pub(super) fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub(super) fn rademacher_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows, cols));
        for j in 0..cols {
            for i in 0..rows {
                let h = Self::splitmix64(
                    seed ^ ((i as u64).wrapping_mul(0x9E37)) ^ ((j as u64).wrapping_mul(0x85EB)),
                );
                out[[i, j]] = if (h & 1) == 0 { -1.0 } else { 1.0 };
            }
        }
        out
    }

    pub(super) fn orthonormalize_columns(a: &Array2<f64>, tol: f64) -> Array2<f64> {
        let p = a.nrows();
        let c = a.ncols();
        let mut q = Array2::<f64>::zeros((p, c));
        let mut kept = 0usize;
        for j in 0..c {
            let mut v = a.column(j).to_owned();
            for t in 0..kept {
                let qt = q.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > tol {
                q.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
            }
        }
        if kept == c {
            q
        } else {
            q.slice(ndarray::s![.., 0..kept]).to_owned()
        }
    }
}
