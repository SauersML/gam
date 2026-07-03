//! #2023 tiered SAE spine: the orchestration container that composes a shared
//! mean (Tier-0), a large linear sparse-dictionary bulk (Tier-1), and an
//! evidence-K curved tier (Tier-2) fit on the whitened Tier-1 residual.
//!
//! This module owns the **spine-level** types every tier hangs off:
//!   * [`Tier0Mean`] — the single shared mean μ. Moving the DC out of every atom
//!     into ONE Tier-0 mean is the structural kill of the co-collapse-to-mean
//!     class (issue #10 / #1893): on the de-meaned data the all-atoms-equal-to-
//!     mean state reconstructs zero, so it is EV-invisible and gets pruned rather
//!     than rewarded and PC-reseeded.
//!   * [`TieredConfig`] — the composed-fit knobs.
//!   * [`interference_subspace`] — Tier-1's active subspace `Q` (what the linear
//!     dictionary already explains) and its orthogonal complement `Q⊥`. Per the
//!     #2021 coupling the linear dictionary *is* the interference model for the
//!     curved fit: the Tier-2 GLS weight down-weights `Q` (penalizes `Q⊥`), so
//!     curved atoms chase only residual directions. Emitted so Tier-2 can install
//!     a HELD `behavioral_fisher` metric (`structured_whitening=False`) — the
//!     path that both realizes #2021 and avoids the structured-whitening fitter
//!     bug.
//!   * [`WhitenedResidualHandoff`] — the Mode-B (shared whitened residual)
//!     hand-off to Tier-2.
//!   * [`TieredSaeFit`] — the composed artifact. Generic over the Tier-2 artifact
//!     type `T2` so the `tier2-curved` owner defines that struct without a
//!     circular dependency on this module.
//!
//! Term-level composition (concatenating a Tier-1 linear term with a Tier-2
//! curved term into one solve) already lives in [`crate::manifold`]:
//! `SaeManifoldTerm::merge_tiers` + `manifold::stagewise::terminal_joint_assembly`
//! (exact additivity under independent JumpReLU/IBP gates). The Mode-A per-block
//! scale-out (one K=1 curved chart per orthonormal Tier-1 block) consumes the
//! block frames on the block-sparse fit directly; see `sparse_dict::block`.

use std::collections::BTreeMap;

use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::sparse_dict::{SparseDictConfig, SparseDictFit};

/// Tier-0: the single shared mean μ (length `p`). The global DC lives here, not
/// duplicated across `K` per-atom intercepts.
#[derive(Clone, Debug)]
pub struct Tier0Mean {
    /// The shared mean, length `p`.
    pub mean: Array1<f64>,
}

impl Tier0Mean {
    /// Fit Tier-0 as the column mean of `z` (`N×P`). This is the train-split mean;
    /// hold it fixed and reuse it for out-of-sample de-meaning and for the EV
    /// baseline so held-out EV is measured against the same Tier-0 constant.
    pub fn fit(z: ArrayView2<'_, f64>) -> Result<Self, String> {
        if z.nrows() == 0 || z.ncols() == 0 {
            return Err("Tier0Mean::fit requires a non-empty (N, P) matrix".to_string());
        }
        let mean = z
            .mean_axis(Axis(0))
            .ok_or_else(|| "Tier0Mean::fit: mean_axis returned None".to_string())?;
        Ok(Self { mean })
    }

    /// De-mean: `R0 = z − μ` (row-broadcast). The Tier-1 bulk is fit on this.
    pub fn apply(&self, z: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if z.ncols() != self.mean.len() {
            return Err(format!(
                "Tier0Mean::apply: z has P={} but μ has length {}",
                z.ncols(),
                self.mean.len()
            ));
        }
        Ok(&z - &self.mean.view().insert_axis(Axis(0)))
    }

    /// Add μ back to a de-meaned reconstruction (`recon + μ`), row-broadcast.
    pub fn reconstruct(&self, recon: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if recon.ncols() != self.mean.len() {
            return Err(format!(
                "Tier0Mean::reconstruct: recon has P={} but μ has length {}",
                recon.ncols(),
                self.mean.len()
            ));
        }
        Ok(&recon + &self.mean.view().insert_axis(Axis(0)))
    }
}

/// Tier-0 PER-CONTEXT mean: one mean vector per context/template group, with a
/// global fallback for groups unseen at fit time. On real residual streams a
/// per-prompt/per-template DC otherwise leaks into the fit (measured to drive
/// held-out EV negative), so per-template demean is the production Tier-0 for
/// grouped data; [`Tier0Mean`] is the single-group (global) special case. Same
/// structural DC-atom kill as `Tier0Mean` (#10), applied within each context.
#[derive(Clone, Debug)]
pub struct PerContextMean {
    /// Global fallback mean (used for groups unseen at fit time), length `p`.
    pub global: Array1<f64>,
    /// Per-group column means, keyed by context/template id.
    pub group_means: BTreeMap<i64, Array1<f64>>,
}

impl PerContextMean {
    /// Fit per-group column means from `z` (`N×P`) and `group_ids` (length `N`,
    /// one context id per row). Also stores the global mean as the fallback.
    pub fn fit(z: ArrayView2<'_, f64>, group_ids: &[i64]) -> Result<Self, String> {
        let n = z.nrows();
        let p = z.ncols();
        if n == 0 || p == 0 {
            return Err("PerContextMean::fit requires a non-empty (N, P) matrix".to_string());
        }
        if group_ids.len() != n {
            return Err(format!(
                "PerContextMean::fit: group_ids length {} != N {n}",
                group_ids.len()
            ));
        }
        let global = z
            .mean_axis(Axis(0))
            .ok_or_else(|| "PerContextMean::fit: global mean_axis returned None".to_string())?;
        let mut sums: BTreeMap<i64, (Array1<f64>, usize)> = BTreeMap::new();
        for (row, &g) in z.rows().into_iter().zip(group_ids.iter()) {
            let entry = sums.entry(g).or_insert_with(|| (Array1::<f64>::zeros(p), 0usize));
            entry.0 += &row;
            entry.1 += 1;
        }
        let mut group_means = BTreeMap::new();
        for (g, (sum, count)) in sums {
            if count > 0 {
                group_means.insert(g, sum / count as f64);
            }
        }
        Ok(Self {
            global,
            group_means,
        })
    }

    /// The mean for a context: its own if seen at fit time, else the global fallback.
    pub fn row_mean(&self, group: i64) -> &Array1<f64> {
        self.group_means.get(&group).unwrap_or(&self.global)
    }

    /// De-mean each row by its context mean: `R0[i] = z[i] − μ_{group[i]}`.
    pub fn apply(&self, z: ArrayView2<'_, f64>, group_ids: &[i64]) -> Result<Array2<f64>, String> {
        if group_ids.len() != z.nrows() {
            return Err(format!(
                "PerContextMean::apply: group_ids length {} != N {}",
                group_ids.len(),
                z.nrows()
            ));
        }
        let mut out = z.to_owned();
        for (mut row, &g) in out.rows_mut().into_iter().zip(group_ids.iter()) {
            row -= self.row_mean(g);
        }
        Ok(out)
    }

    /// Add each row's context mean back to a de-meaned reconstruction.
    pub fn reconstruct(
        &self,
        recon: ArrayView2<'_, f64>,
        group_ids: &[i64],
    ) -> Result<Array2<f64>, String> {
        if group_ids.len() != recon.nrows() {
            return Err(format!(
                "PerContextMean::reconstruct: group_ids length {} != N {}",
                group_ids.len(),
                recon.nrows()
            ));
        }
        let mut out = recon.to_owned();
        for (mut row, &g) in out.rows_mut().into_iter().zip(group_ids.iter()) {
            row += self.row_mean(g);
        }
        Ok(out)
    }
}

/// Knobs for a composed tiered fit.
#[derive(Clone, Debug)]
pub struct TieredConfig {
    /// Tier-1 collapsed-linear sparse dictionary configuration (carries `K`, the
    /// active budget `s`, epochs, and the GPU score-routing mode).
    pub tier1: SparseDictConfig,
    /// Rank `r` of the interference subspace `Q` handed to Tier-2 (`None` ⇒ pick
    /// by the 99% energy threshold in [`interference_subspace`]).
    pub lambda_seed_rank: Option<usize>,
    /// Whether to run the Tier-2 curved tier at all (`false` ⇒ Tier-0 + Tier-1
    /// only, the linear-bulk baseline).
    pub tier2_enabled: bool,
}

impl TieredConfig {
    /// A Tier-0 + Tier-1 config at dictionary width `k_linear` (Tier-2 disabled).
    pub fn linear_bulk(k_linear: usize) -> Self {
        Self {
            tier1: SparseDictConfig::new(k_linear),
            lambda_seed_rank: None,
            tier2_enabled: false,
        }
    }
}

/// Tier-1's interference subspace: the directions the linear dictionary already
/// explains (`q`), its orthogonal complement (`q_perp`), and the per-direction
/// energy scale (`scale`, the singular values of the usage-weighted decoder).
///
/// `q` is `P×r` with orthonormal columns; `q_perp` is `P×(P−r)` with orthonormal
/// columns; together they are a full orthonormal basis of `ℝ^P` (`q ⟂ q_perp`).
/// Tier-2's GLS weight `G = q_perp q_perpᵀ = I − q qᵀ` down-weights `q` — so the
/// curved fit pursues only what Tier-1 misses. `scale` (length `r`) lets Tier-2
/// set the `Σ = Λ c Λᵀ + D` factor magnitude without refitting.
#[derive(Clone, Debug)]
pub struct InterferenceSubspace {
    /// Active subspace, `P×r`, orthonormal columns (what Tier-1 explains).
    pub q: Array2<f64>,
    /// Orthogonal complement, `P×(P−r)`, orthonormal columns (`Q⊥`).
    pub q_perp: Array2<f64>,
    /// Singular values of the usage-weighted decoder along `q`, length `r`.
    pub scale: Array1<f64>,
}

/// Compute Tier-1's [`InterferenceSubspace`] from a fitted sparse dictionary.
///
/// Forms the usage-weighted decoder Gram `G = Σ_k w_k d_k d_kᵀ` (`P×P`), where
/// `d_k` is atom `k`'s decoder row and `w_k = Σ_i codes[i,k]²` is its total fired
/// energy (so dead atoms contribute nothing and `Q` is genuinely the *active*
/// subspace). The eigenvectors of `G` split into the top-`r` (the active subspace
/// `Q`) and the trailing `P−r` (`Q⊥`); `scale = √eval` along `Q`.
///
/// `rank`: `Some(r)` pins `r = min(r, P)`; `None` keeps the smallest `r` whose
/// eigen-energy reaches 99% of the total (at least 1).
pub fn interference_subspace(
    fit: &SparseDictFit,
    rank: Option<usize>,
) -> Result<InterferenceSubspace, String> {
    let decoder = fit.decoder.view();
    let k = decoder.nrows();
    let p = decoder.ncols();
    if k == 0 || p == 0 {
        return Err("interference_subspace: empty decoder".to_string());
    }

    // Per-atom fired energy w_k = Σ_i codes[i,k]².
    let mut weight = vec![0.0f64; k];
    for (idx_row, code_row) in fit.indices.rows().into_iter().zip(fit.codes.rows()) {
        for (&atom_u32, &code) in idx_row.iter().zip(code_row.iter()) {
            let atom = atom_u32 as usize;
            if atom < k {
                weight[atom] += (code as f64) * (code as f64);
            }
        }
    }

    // Usage-weighted decoder `Dw` (K×P), Dw_k = √w_k · d_k, then G = Dwᵀ Dw (P×P)
    // via a single GEMM rather than K rank-1 updates.
    let mut dw = Array2::<f64>::zeros((k, p));
    for atom in 0..k {
        let sw = weight[atom].max(0.0).sqrt();
        if sw == 0.0 {
            continue;
        }
        let src = decoder.row(atom);
        let mut dst = dw.row_mut(atom);
        for c in 0..p {
            dst[c] = sw * (src[c] as f64);
        }
    }
    let gram = dw.t().dot(&dw);

    // Symmetric eigendecomposition: ascending eigenvalues, columns are the
    // orthonormal eigenvectors (leading direction is the LAST column).
    let (evals, evecs) = gram
        .eigh(faer::Side::Lower)
        .map_err(|err| format!("interference_subspace eigensolve failed: {err}"))?;
    let total: f64 = evals.iter().map(|&e| e.max(0.0)).sum();
    if total <= 0.0 {
        return Err(
            "interference_subspace: Tier-1 decoder carries no fired energy (all atoms dead)"
                .to_string(),
        );
    }

    // Choose r (columns are ascending, so the active subspace is the TAIL).
    let r = match rank {
        Some(r) => r.min(p).max(1),
        None => {
            // Smallest r whose top-r eigen-energy reaches 99% of the total.
            let mut acc = 0.0f64;
            let mut chosen = 1usize;
            for (taken, &e) in evals.iter().rev().enumerate() {
                acc += e.max(0.0);
                chosen = taken + 1;
                if acc >= 0.99 * total {
                    break;
                }
            }
            chosen.min(p).max(1)
        }
    };

    // q = last r columns (largest eigenvalues), scale = √eval along q.
    let mut q = Array2::<f64>::zeros((p, r));
    let mut scale = Array1::<f64>::zeros(r);
    for j in 0..r {
        let col = p - 1 - j; // descending: p-1 is the largest
        q.column_mut(j).assign(&evecs.column(col));
        scale[j] = evals[col].max(0.0).sqrt();
    }
    // q_perp = the leading (p − r) columns (smallest eigenvalues), the complement.
    let pr = p - r;
    let mut q_perp = Array2::<f64>::zeros((p, pr));
    for j in 0..pr {
        q_perp.column_mut(j).assign(&evecs.column(j));
    }

    Ok(InterferenceSubspace { q, q_perp, scale })
}

/// Mode-B hand-off: the whitened *shared* residual and Tier-1's interference
/// model, handed to the Tier-2 curved fit (`tier2-curved` / #17). Tier-2 fits its
/// curved atoms on `residual`, installing a held GLS metric built from
/// `interference` (down-weighting `q`, i.e. what Tier-1 explains).
#[derive(Clone, Debug)]
pub struct WhitenedResidualHandoff {
    /// Post-Tier-1 residual `R = (z − μ) − T1.reconstruct()`, `N×P`, f64.
    pub residual: Array2<f64>,
    /// Tier-1's interference subspace (`q`, `q_perp`, `scale`).
    pub interference: InterferenceSubspace,
    /// The frozen Tier-1 decoder, `K×P` (for out-of-sample residual recompute).
    pub tier1_decoder: Array2<f32>,
    /// The Tier-0 shared mean μ, length `P`.
    pub mean: Array1<f64>,
}

/// The composed tiered artifact. Generic over the Tier-2 artifact `T2` (defined
/// by the `tier2-curved` owner as `Tier2CurvedArtifact`) so this container has no
/// circular dependency on the curved-tier module. `tier2` is `None` when Tier-2
/// is disabled or every curved birth is rejected.
#[derive(Clone, Debug)]
pub struct TieredSaeFit<T2> {
    /// Tier-0 shared mean.
    pub tier0: Tier0Mean,
    /// Tier-1 linear sparse-dictionary bulk.
    pub tier1: SparseDictFit,
    /// Tier-2 curved artifact (owner-defined), if present.
    pub tier2: Option<T2>,
    /// Combined held-in explained variance against the Tier-0 mean baseline.
    pub explained_variance: f64,
}

/// Explained variance `1 − RSS/TSS` of `recon` against `z`, with the total sum of
/// squares taken about the supplied Tier-0 `mean` (the honest tiered baseline: a
/// model must beat "predict the shared mean", not "predict zero").
pub fn explained_variance_vs_mean(
    z: ArrayView2<'_, f64>,
    recon: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
) -> f64 {
    let mut rss = 0.0f64;
    for (zr, rr) in z.rows().into_iter().zip(recon.rows()) {
        for c in 0..z.ncols() {
            let d = zr[c] - rr[c];
            rss += d * d;
        }
    }
    let baseline = &z - &mean.view().insert_axis(Axis(0));
    let tss: f64 = baseline.iter().map(|&v| v * v).sum();
    if tss <= 0.0 {
        return f64::NAN;
    }
    1.0 - rss / tss
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tier0_mean_roundtrips() {
        let z = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let t0 = Tier0Mean::fit(z.view()).expect("fit");
        assert!((t0.mean[0] - 3.0).abs() < 1e-12);
        assert!((t0.mean[1] - 4.0).abs() < 1e-12);
        let demeaned = t0.apply(z.view()).expect("apply");
        // Column means of the de-meaned data are ~0.
        let cm = demeaned.mean_axis(Axis(0)).unwrap();
        assert!(cm[0].abs() < 1e-12 && cm[1].abs() < 1e-12);
        // reconstruct(apply(z)) == z.
        let back = t0.reconstruct(demeaned.view()).expect("reconstruct");
        for (a, b) in back.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn interference_subspace_q_and_qperp_are_orthonormal_complements() {
        // A 2-atom dictionary in p=3 spanning e0 and e1; e2 is unexplained.
        let decoder = array![[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = array![[0u32, 1u32], [0u32, 1u32]];
        let codes = array![[2.0f32, 1.0f32], [2.0f32, 1.0f32]];
        let fit = SparseDictFit {
            decoder,
            indices,
            codes,
            explained_variance: 0.0,
            epochs: 0,
            converged: true,
            active: 2,
            score_route_stats: Default::default(),
        };
        let sub = interference_subspace(&fit, Some(2)).expect("subspace");
        assert_eq!(sub.q.dim(), (3, 2));
        assert_eq!(sub.q_perp.dim(), (3, 1));
        // q columns orthonormal.
        let gq = sub.q.t().dot(&sub.q);
        assert!((gq[[0, 0]] - 1.0).abs() < 1e-9 && (gq[[1, 1]] - 1.0).abs() < 1e-9);
        assert!(gq[[0, 1]].abs() < 1e-9);
        // q ⟂ q_perp.
        let cross = sub.q.t().dot(&sub.q_perp);
        assert!(cross.iter().all(|&v| v.abs() < 1e-9));
        // q_perp must be (±)e2, the unexplained direction.
        assert!(sub.q_perp[[2, 0]].abs() > 0.999);
        assert!(sub.q_perp[[0, 0]].abs() < 1e-6 && sub.q_perp[[1, 0]].abs() < 1e-6);
        // Atom 0 (code 2) carries more energy than atom 1 (code 1) ⇒ larger scale first.
        assert!(sub.scale[0] >= sub.scale[1]);
    }

    #[test]
    fn per_context_mean_zeros_each_group_and_falls_back() {
        // group 0 centered at (10,10), group 1 at (−5,−5).
        let z = array![[11.0, 9.0], [9.0, 11.0], [-4.0, -6.0], [-6.0, -4.0]];
        let groups = [0i64, 0, 1, 1];
        let pcm = PerContextMean::fit(z.view(), &groups).expect("fit");
        assert!((pcm.row_mean(0)[0] - 10.0).abs() < 1e-12);
        assert!((pcm.row_mean(1)[0] + 5.0).abs() < 1e-12);
        // Unseen context falls back to the global mean.
        assert!((pcm.row_mean(999)[0] - pcm.global[0]).abs() < 1e-12);
        // Per-context de-mean zeros each group (⇒ column sums ~0 overall).
        let demeaned = pcm.apply(z.view(), &groups).expect("apply");
        let col_sum = demeaned.sum_axis(Axis(0));
        assert!(col_sum[0].abs() < 1e-12 && col_sum[1].abs() < 1e-12);
        // Roundtrip.
        let back = pcm.reconstruct(demeaned.view(), &groups).expect("reconstruct");
        for (a, b) in back.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}
