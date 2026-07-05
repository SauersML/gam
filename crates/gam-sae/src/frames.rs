use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::SaeManifoldTerm;
use gam_linalg::faer_ndarray::{FaerSvd, fast_ab, fast_abt, fast_atb};

#[derive(Clone, Debug)]
pub(crate) struct FrameProjection {
    pub(crate) p: usize,
    pub(crate) beta_offsets: Vec<usize>,
    pub(crate) border_offsets: Vec<usize>,
    pub(crate) basis_sizes: Vec<usize>,
    pub(crate) ranks: Vec<usize>,
    frames: Vec<Option<Array2<f64>>>,
}

impl FrameProjection {
    pub(crate) fn new(term: &SaeManifoldTerm) -> Self {
        Self {
            p: term.output_dim(),
            beta_offsets: term.beta_offsets(),
            border_offsets: term.factored_border_offsets(),
            basis_sizes: term.atoms.iter().map(|atom| atom.basis_size()).collect(),
            ranks: term
                .atoms
                .iter()
                .map(|atom| atom.border_frame_rank())
                .collect(),
            frames: term
                .atoms
                .iter()
                .map(|atom| {
                    atom.decoder_frame
                        .as_ref()
                        .map(|frame| frame.frame().to_owned())
                })
                .collect(),
        }
    }

    pub(crate) fn beta_dim(&self) -> usize {
        self.basis_sizes.iter().sum::<usize>() * self.p
    }

    /// Owned per-atom output frames (`U_k`, `p × r_k`), `None` for an unframed
    /// atom (`U_k = I_p`). Handed to the #974 whitened factored β-Hessian
    /// operator so it can expand the factored coordinates through each frame.
    pub(crate) fn frames_owned(&self) -> Vec<Option<Array2<f64>>> {
        self.frames.clone()
    }

    pub(crate) fn border_dim(&self) -> usize {
        self.basis_sizes
            .iter()
            .zip(&self.ranks)
            .map(|(m, r)| m * r)
            .sum()
    }

    pub(crate) fn lift_border_vec(&self, border: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.beta_dim());
        for atom in 0..self.basis_sizes.len() {
            self.lift_atom_vec_into(atom, border, out.view_mut());
        }
        out
    }

    pub(crate) fn project_border_vec(&self, beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.border_dim());
        for atom in 0..self.basis_sizes.len() {
            self.project_atom_vec_into(atom, beta, out.view_mut(), 1.0);
        }
        out
    }

    pub(crate) fn lift_block(&self, atom: usize, block: ArrayView2<'_, f64>) -> Array2<f64> {
        let m = self.basis_sizes[atom];
        let r = self.ranks[atom];
        if self.frames[atom].is_none() {
            return block.to_owned();
        }
        let uk = self.frames[atom].as_ref().expect("framed atom has a frame");
        let mut out = Array2::<f64>::zeros((m * self.p, m * self.p));
        for b1 in 0..m {
            for b2 in 0..m {
                for c1 in 0..self.p {
                    for c2 in 0..self.p {
                        let mut acc = 0.0;
                        for j1 in 0..r {
                            for j2 in 0..r {
                                acc +=
                                    uk[[c1, j1]] * block[[b1 * r + j1, b2 * r + j2]] * uk[[c2, j2]];
                            }
                        }
                        out[[b1 * self.p + c1, b2 * self.p + c2]] = acc;
                    }
                }
            }
        }
        out
    }

    pub(crate) fn project_block(&self, hbb: ArrayView2<'_, f64>) -> Array2<f64> {
        let t = self.project_rows(hbb);
        let mut out = Array2::<f64>::zeros((self.border_dim(), self.border_dim()));
        for atom in 0..self.basis_sizes.len() {
            self.project_block_left_atom(atom, t.view(), out.view_mut());
        }
        out
    }

    pub(crate) fn project_rows(&self, block: ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((block.nrows(), self.border_dim()));
        for row in 0..block.nrows() {
            let projected = self.project_border_vec(block.row(row));
            out.row_mut(row).assign(&projected);
        }
        out
    }

    pub(crate) fn atom_border_range(&self, atom: usize) -> std::ops::Range<usize> {
        let start = self.border_offsets[atom];
        start..start + self.basis_sizes[atom] * self.ranks[atom]
    }

    pub(crate) fn lift_axis_into(
        &self,
        out: &mut Array1<f64>,
        atom: usize,
        basis_col: usize,
        frame_col: usize,
    ) {
        let base = self.beta_offsets[atom] + basis_col * self.p;
        match &self.frames[atom] {
            None => out[base + frame_col] = 1.0,
            Some(uk) => {
                for out_col in 0..self.p {
                    out[base + out_col] = uk[[out_col, frame_col]];
                }
            }
        }
    }

    pub(crate) fn lift_local_axis_into(
        &self,
        out: &mut Array1<f64>,
        atom: usize,
        basis_col: usize,
        frame_col: usize,
    ) {
        let base = basis_col * self.p;
        match &self.frames[atom] {
            None => out[base + frame_col] = 1.0,
            Some(uk) => {
                for out_col in 0..self.p {
                    out[base + out_col] = uk[[out_col, frame_col]];
                }
            }
        }
    }

    pub(crate) fn project_atom_vec_into(
        &self,
        atom: usize,
        beta: ArrayView1<'_, f64>,
        mut out: ndarray::ArrayViewMut1<'_, f64>,
        scale: f64,
    ) {
        let m = self.basis_sizes[atom];
        let r = self.ranks[atom];
        let ob = self.beta_offsets[atom];
        let oc = self.border_offsets[atom];
        for basis_col in 0..m {
            let base_b = ob + basis_col * self.p;
            let base_c = oc + basis_col * r;
            match &self.frames[atom] {
                None => {
                    for j in 0..r {
                        out[base_c + j] += scale * beta[base_b + j];
                    }
                }
                Some(uk) => {
                    for j in 0..r {
                        let mut acc = 0.0;
                        for i in 0..self.p {
                            acc += uk[[i, j]] * beta[base_b + i];
                        }
                        out[base_c + j] += scale * acc;
                    }
                }
            }
        }
    }

    pub(crate) fn project_local_atom_vec_into(
        &self,
        atom: usize,
        beta: ArrayView1<'_, f64>,
        out: ndarray::ArrayViewMut1<'_, f64>,
        scale: f64,
    ) {
        self.project_atom_vec_into_with_base(atom, beta, out, scale, 0);
    }

    pub(crate) fn project_atom_vec_into_with_base(
        &self,
        atom: usize,
        beta: ArrayView1<'_, f64>,
        mut out: ndarray::ArrayViewMut1<'_, f64>,
        scale: f64,
        beta_base_offset: usize,
    ) {
        let m = self.basis_sizes[atom];
        let r = self.ranks[atom];
        let oc = self.border_offsets[atom];
        for basis_col in 0..m {
            let base_b = beta_base_offset + basis_col * self.p;
            let base_c = oc + basis_col * r;
            match &self.frames[atom] {
                None => {
                    for j in 0..r {
                        out[base_c + j] += scale * beta[base_b + j];
                    }
                }
                Some(uk) => {
                    for j in 0..r {
                        let mut acc = 0.0;
                        for i in 0..self.p {
                            acc += uk[[i, j]] * beta[base_b + i];
                        }
                        out[base_c + j] += scale * acc;
                    }
                }
            }
        }
    }

    pub(crate) fn lift_atom_vec_into(
        &self,
        atom: usize,
        border: ArrayView1<'_, f64>,
        mut out: ndarray::ArrayViewMut1<'_, f64>,
    ) {
        let m = self.basis_sizes[atom];
        let r = self.ranks[atom];
        let ob = self.beta_offsets[atom];
        let oc = self.border_offsets[atom];
        for basis_col in 0..m {
            let base_b = ob + basis_col * self.p;
            let base_c = oc + basis_col * r;
            match &self.frames[atom] {
                None => {
                    for i in 0..self.p {
                        out[base_b + i] = border[base_c + i];
                    }
                }
                Some(uk) => {
                    for i in 0..self.p {
                        let mut acc = 0.0;
                        for j in 0..r {
                            acc += uk[[i, j]] * border[base_c + j];
                        }
                        out[base_b + i] = acc;
                    }
                }
            }
        }
    }

    pub(crate) fn accumulate_output_project(
        &self,
        atom: usize,
        c_base: usize,
        output: usize,
        value: f64,
        out: &mut [f64],
    ) {
        match &self.frames[atom] {
            None => out[c_base + output] += value,
            Some(uk) => {
                let rank = self.ranks[atom];
                let frame_row = uk.row(output);
                let frame_slice = frame_row.as_slice().expect("frame rows are contiguous");
                let out_slice = &mut out[c_base..c_base + rank];
                for (slot, &u) in out_slice.iter_mut().zip(frame_slice.iter()) {
                    *slot += value * u;
                }
            }
        }
    }

    pub(crate) fn output_variance(
        &self,
        atom: usize,
        cov_c: ArrayView2<'_, f64>,
        basis: ArrayView1<'_, f64>,
        output: usize,
    ) -> f64 {
        let Some(uk) = &self.frames[atom] else {
            return self.full_output_variance(atom, cov_c, basis, output);
        };
        let m = self.basis_sizes[atom];
        let r = self.ranks[atom];
        let mut var = 0.0;
        for b1 in 0..m {
            let phi1 = basis[b1];
            if phi1 == 0.0 {
                continue;
            }
            for b2 in 0..m {
                let phi2 = basis[b2];
                if phi2 == 0.0 {
                    continue;
                }
                for j1 in 0..r {
                    for j2 in 0..r {
                        var += phi1
                            * phi2
                            * uk[[output, j1]]
                            * cov_c[[b1 * r + j1, b2 * r + j2]]
                            * uk[[output, j2]];
                    }
                }
            }
        }
        var
    }

    pub(crate) fn full_output_variance(
        &self,
        atom: usize,
        cov: ArrayView2<'_, f64>,
        basis: ArrayView1<'_, f64>,
        output: usize,
    ) -> f64 {
        let m = self.basis_sizes[atom];
        let mut var = 0.0;
        for b1 in 0..m {
            let phi1 = basis[b1];
            if phi1 == 0.0 {
                continue;
            }
            for b2 in 0..m {
                var += phi1 * basis[b2] * cov[[b1 * self.p + output, b2 * self.p + output]];
            }
        }
        var
    }

    pub(crate) fn project_block_left_atom(
        &self,
        atom: usize,
        t: ArrayView2<'_, f64>,
        mut out: ndarray::ArrayViewMut2<'_, f64>,
    ) {
        let m = self.basis_sizes[atom];
        let r = self.ranks[atom];
        let ob = self.beta_offsets[atom];
        let oc = self.border_offsets[atom];
        for basis_col in 0..m {
            let base_b = ob + basis_col * self.p;
            let base_c = oc + basis_col * r;
            match &self.frames[atom] {
                None => {
                    for j in 0..r {
                        for c in 0..out.ncols() {
                            out[[base_c + j, c]] += t[[base_b + j, c]];
                        }
                    }
                }
                Some(uk) => {
                    for j in 0..r {
                        for c in 0..out.ncols() {
                            let mut acc = 0.0;
                            for i in 0..self.p {
                                acc += uk[[i, j]] * t[[base_b + i, c]];
                            }
                            out[[base_c + j, c]] += acc;
                        }
                    }
                }
            }
        }
    }
}

/// Build the frames-engaged device SAE PCG data (issue #1017/#1026): the
/// factored-border analogue of the full-`B` `DeviceSaePcgData`. The penalty side
/// carries the smooth `λ S_k ⊗ I_{r_k}` blocks (right-width `r_k`, at `off_c[k]`)
/// and the data-fit `G_{ij} ⊗ W_{ij}` blocks; the reduced-Schur side carries the
/// per-row DENSE cross-block `H_tβ^(i)` as a row-major `q_i × border_dim` slab.
///
/// `args.frame_blocks` are the same `(g, w)` blocks fed to
/// `FactoredFrameKroneckerOp`, snapshotted before that op consumed them.
/// `args.smooth_scaled_s[k]` is `λ S_k` (`M_k × M_k`). A row whose `htbeta` is
/// not at the factored width contributes an empty slab (reduced-Schur term zero).
pub(crate) struct FramedDeviceArgs<'a> {
    pub p: usize,
    pub border_dim: usize,
    pub border_offsets: &'a [usize],
    pub ranks: &'a [usize],
    pub basis_sizes: &'a [usize],
    pub smooth_scaled_s: &'a [Array2<f64>],
    pub frame_blocks: Vec<gam_solve::arrow_schur::FactoredFrameGBlock>,
    pub rows: &'a [gam_solve::arrow_schur::ArrowRowBlock],
}

pub(crate) fn build_framed_device_sae_data(
    args: FramedDeviceArgs<'_>,
) -> gam_solve::arrow_schur::DeviceSaePcgData {
    use gam_solve::arrow_schur::{DeviceSaeFrameData, DeviceSaePcgData, DeviceSaeSmoothBlock};
    let FramedDeviceArgs {
        p,
        border_dim,
        border_offsets,
        ranks,
        basis_sizes,
        smooth_scaled_s,
        frame_blocks,
        rows,
    } = args;
    let n_atoms = ranks.len();
    let mut smooth_blocks = Vec::with_capacity(n_atoms);
    let mut smooth_ranks = Vec::with_capacity(n_atoms);
    for k in 0..n_atoms {
        smooth_blocks.push(DeviceSaeSmoothBlock {
            global_offset: border_offsets[k],
            factor_a: smooth_scaled_s[k].clone(),
        });
        smooth_ranks.push(ranks[k]);
    }
    let row_htbeta: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| {
            let (qi, w) = row.htbeta.dim();
            if w != border_dim {
                return Vec::new();
            }
            let mut flat = vec![0.0_f64; qi * w];
            for c in 0..qi {
                for a in 0..w {
                    flat[c * w + a] = row.htbeta[[c, a]];
                }
            }
            flat
        })
        .collect();
    DeviceSaePcgData {
        p,
        beta_dim: border_dim,
        // #1033: empty shared slices — the frames path carries its cross-block
        // through `frame.frame_blocks`, not the full-`B` `a_phi`/`local_jac`.
        a_phi: std::sync::Arc::from(Vec::new().into_boxed_slice()),
        local_jac: std::sync::Arc::from(Vec::new().into_boxed_slice()),
        smooth_blocks,
        sparse_g_blocks: Vec::new(),
        frame: Some(DeviceSaeFrameData {
            ranks: ranks.to_vec(),
            basis_sizes: basis_sizes.to_vec(),
            border_offsets: border_offsets.to_vec(),
            frame_blocks,
            smooth_ranks,
            row_htbeta,
        }),
    }
}

/// Relative spectral cutoff used when the Grassmann-frame factorization decides
/// the effective column rank `r` of an atom's decoder `B_k` (issue #972). A
/// singular value of `B_k` below `cutoff · σ_max` carries `< (σ/σ_max)²` of the
/// decoder energy and is dropped from the profiled frame.
pub(crate) const SAE_FRAME_RANK_CUTOFF: f64 = 1.0e-7;

/// Small ambient decoders stay on the full-`B` path. Below this width the dense
/// decoder border is cheap, while auto-profiling a cold low-rank decoder changes
/// the β coordinates before the inner solve has learned the output span.
pub(crate) const SAE_FRAME_MIN_AUTO_OUTPUT_DIM: usize = 12;

/// Border-saving threshold for auto-activating the low-rank Grassmann
/// factorization (issue #972). The factored border holds `Σ_k M_k · r` instead
/// of `Σ_k M_k · p`, so factorization is beneficial only when the chosen frame
/// rank `r` is materially smaller than the ambient output dimension `p`. We
/// require `r ≤ p · (1 − margin)` (frame must shrink the per-atom border by at
/// least this fraction) AND a positive absolute gap `p − r ≥ 1`, so a full-rank
/// atom (`r == p`) never pays the polar-step / frame-storage cost for zero
/// border saving and stays bit-for-bit on the historical full-`B` path.
pub(crate) const SAE_FRAME_ACTIVATION_MARGIN: f64 = 0.25;

/// A Grassmann point: a `p × r` column-orthonormal FRAME `U` spanning an atom's
/// decoder column space (issue #972).
///
/// The decoder coefficient matrix `B_k` (`M_k × p`) factors as `B_k = C_k · Uᵀ`
/// where `C_k` (`M_k × r`) is the coordinate matrix that lives IN the
/// arrow-Schur border and `U` (`p × r`) is this frame, profiled OUT of the
/// border by closed-form streaming polar steps. The border then carries only
/// `Σ_k M_k · r` coefficients rather than `Σ_k M_k · p` — the reduction that
/// keeps the border Cholesky / evidence log-det tractable at frontier `p`.
///
/// **Canonical inner gauge.** `U` is only defined up to a right `r × r`
/// orthogonal rotation `U → U R` (with the matching `C_k → C_k R`); the column
/// span (the Grassmann point) is invariant. For deterministic serialization we
/// pin a canonical representative: the frame is the left-singular subspace of
/// the cross-moment, ordered by descending singular value, with each column's
/// sign fixed so its largest-magnitude entry is non-negative. The ordering is
/// recorded by the `gauge_singular_values` field so the same span always
/// serializes to the same bytes (no run-to-run rotation drift).
#[derive(Debug, Clone)]
pub struct GrassmannFrame {
    /// Column-orthonormal frame `U`, shape `(p, r)` with `Uᵀ U = I_r`.
    frame: Array2<f64>,
    /// Singular values of the most recent cross-moment used to build `U`,
    /// descending, length `r`. The canonical ordering gauge (issue #972).
    gauge_singular_values: Array1<f64>,
}

impl GrassmannFrame {
    /// Ambient output dimension `p`.
    pub fn output_dim(&self) -> usize {
        self.frame.nrows()
    }

    /// Frame rank `r` (number of profiled column directions).
    pub fn rank(&self) -> usize {
        self.frame.ncols()
    }

    /// Canonical descending singular values of the cross-moment that fixed this
    /// frame's column ordering (issue #972). Exposed so the serialization /
    /// canonicalization path can read the recorded gauge and reproduce the same
    /// span byte-for-byte (no run-to-run rotation drift).
    pub fn gauge_singular_values(&self) -> &Array1<f64> {
        &self.gauge_singular_values
    }

    /// Read-only view of the orthonormal frame `U` (`p × r`).
    pub fn frame(&self) -> ArrayView2<'_, f64> {
        self.frame.view()
    }

    /// Grassmann manifold dimension `r·(p − r)` of this frame — the count of
    /// profiled-out degrees of freedom that must enter the Laplace evidence
    /// dimension accounting (issue #972, evidence honesty). A point on the
    /// Grassmannian `Gr(r, p)` has exactly this many intrinsic coordinates.
    pub fn manifold_dimension(&self) -> usize {
        let r = self.rank();
        let p = self.output_dim();
        r * (p - r)
    }

    /// Build the canonical-gauge frame for a `p × r` orthonormal `U` paired with
    /// its `gauge_singular_values`. Enforces the column-sign convention
    /// (largest-magnitude entry per column non-negative) so the span serializes
    /// deterministically. The caller guarantees `U` is already column-orthonormal
    /// and its columns are ordered by descending singular value.
    pub(crate) fn from_oriented(
        mut frame: Array2<f64>,
        gauge_singular_values: Array1<f64>,
    ) -> Self {
        let (p, r) = frame.dim();
        for col in 0..r {
            // Sign-fix: make the largest-magnitude entry of each column
            // non-negative so `U` and `−U` (same span) serialize identically.
            let mut pivot_abs = 0.0_f64;
            let mut pivot_val = 0.0_f64;
            for row in 0..p {
                let v = frame[[row, col]];
                if v.abs() > pivot_abs {
                    pivot_abs = v.abs();
                    pivot_val = v;
                }
            }
            if pivot_val < 0.0 {
                for row in 0..p {
                    frame[[row, col]] = -frame[[row, col]];
                }
            }
        }
        Self {
            frame,
            gauge_singular_values,
        }
    }

    /// Closed-form streaming POLAR step (issue #972): given an accumulated
    /// `p × r` cross-moment `Mcm` (a sum of decoder-target outer products that
    /// pulls the frame toward the current column-span evidence), return the
    /// orthogonal polar factor `U_new = polar(Mcm)`.
    ///
    /// `polar(M) = W Vᵀ` from the thin SVD `M = W Σ Vᵀ`: the nearest
    /// column-orthonormal matrix to `M` in Frobenius norm, and the closed-form
    /// MAP frame update on the Grassmannian. Runs OUTSIDE the border (an
    /// `O(p r² )` thin SVD), so the border never carries the `p` factor.
    /// `gauge_singular_values = Σ` records the canonical descending-σ ordering.
    pub fn polar_update(cross_moment: ArrayView2<'_, f64>) -> Result<Self, String> {
        let (p, r) = cross_moment.dim();
        if p == 0 || r == 0 {
            return Err("GrassmannFrame::polar_update: cross-moment must be non-empty".into());
        }
        if r > p {
            return Err(format!(
                "GrassmannFrame::polar_update: frame rank r={r} cannot exceed output dim p={p}"
            ));
        }
        let owned = cross_moment.to_owned();
        let (u_opt, sv, vt_opt) = owned
            .svd(true, true)
            .map_err(|e| format!("GrassmannFrame::polar_update: SVD failed: {e}"))?;
        let w = u_opt.ok_or_else(|| {
            "GrassmannFrame::polar_update: thin SVD returned no left factor".to_string()
        })?;
        let vt = vt_opt.ok_or_else(|| {
            "GrassmannFrame::polar_update: thin SVD returned no right factor".to_string()
        })?;
        // `W` is `p × r`, `Vᵀ` is `r × r`. polar(M) = W·Vᵀ is `p × r`,
        // column-orthonormal because both factors have orthonormal columns/rows.
        let polar = fast_ab(&w, &vt);
        Ok(Self::from_oriented(polar, sv))
    }

    /// Project a coordinate matrix `C_k` (`M_k × r`) back to the full decoder
    /// `B_k = C_k · Uᵀ` (`M_k × p`) — the reconstruction used wherever the
    /// full-`B` consumers (assembly, decode, smoothness pullback) read the
    /// decoder. `fast_abt` computes `C_k · Uᵀ` without materializing `Uᵀ`.
    pub fn reconstruct_decoder(&self, coords: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if coords.ncols() != self.rank() {
            return Err(format!(
                "GrassmannFrame::reconstruct_decoder: coord cols {} must equal frame rank {}",
                coords.ncols(),
                self.rank()
            ));
        }
        Ok(fast_abt(&coords.to_owned(), &self.frame))
    }

    /// Project a full decoder `B_k` (`M_k × p`) onto this frame, returning the
    /// coordinate matrix `C_k = B_k · U` (`M_k × r`) that the border stores.
    /// The frame is orthonormal so `U` is its own pseudo-inverse-from-the-right:
    /// `C_k = B_k U` recovers the in-span coordinates exactly and discards the
    /// component of `B_k` orthogonal to the frame (zero when `B_k`'s span lies in
    /// `range(U)`, i.e. when the frame rank matched the decoder rank).
    pub fn project_decoder(&self, decoder: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if decoder.ncols() != self.output_dim() {
            return Err(format!(
                "GrassmannFrame::project_decoder: decoder cols {} must equal output dim {}",
                decoder.ncols(),
                self.output_dim()
            ));
        }
        Ok(fast_ab(&decoder.to_owned(), &self.frame))
    }

    /// Largest principal angle (radians) between this frame's column span and
    /// another `p × r'` orthonormal frame's span — the Grassmann geodesic
    /// distance component used by the planted-atom recovery verifier (issue
    /// #972).
    ///
    /// The naive formula `arccos(min σ_i(UᵀV))` loses half the available
    /// precision for near-parallel spans: when `cos θ = 1 − ε` (the
    /// `ε ~ fp64.eps` regime hit by a polar update of an already-orthonormal
    /// frame), `arccos(1 − ε) ≈ √(2ε)` ≈ `1.49e-8`, so a planted span the
    /// solver actually recovered to machine precision was being reported as
    /// `O(√fp64.eps)` off. The stable form uses BOTH the cosines from
    /// `M = UᵀV` (small-angle limit: `cos θ ≈ 1 − θ²/2`, sensitive to noise)
    /// AND the sines from the orthogonal complement
    /// `V_⊥ = (I − UUᵀ) V` (small-angle limit: `sin θ ≈ θ`, sensitive to the
    /// quantity we actually want), then combines them with `atan2(sin, cos)`.
    /// `atan2` returns a precise angle across the whole `[0, π/2]` interval
    /// regardless of which leg is small — so an exactly-equal-frame test now
    /// reports the genuine ~fp64.eps residual instead of an inflated
    /// `√fp64.eps`. The pairing is exact because the singular values of
    /// `M` and `V_⊥` are matched component-wise to the same principal
    /// angle: `σ_r(M) = cos θ_max` and `σ_1(V_⊥) = sin θ_max`.
    pub fn max_principal_angle(&self, other: ArrayView2<'_, f64>) -> Result<f64, String> {
        if other.nrows() != self.output_dim() {
            return Err(format!(
                "GrassmannFrame::max_principal_angle: other rows {} must equal output dim {}",
                other.nrows(),
                self.output_dim()
            ));
        }
        let other_owned = other.to_owned();
        let overlap = fast_atb(&self.frame, &other_owned);
        let (_u, sv_cos, _vt) = overlap
            .svd(false, false)
            .map_err(|e| format!("GrassmannFrame::max_principal_angle: cos-SVD failed: {e}"))?;
        // V_⊥ = V − U·(UᵀV); its largest singular value is sin(θ_max).
        let u_overlap = fast_ab(&self.frame, &overlap);
        let v_perp = &other_owned - &u_overlap;
        let (_u, sv_sin, _vt) = v_perp
            .svd(false, false)
            .map_err(|e| format!("GrassmannFrame::max_principal_angle: sin-SVD failed: {e}"))?;
        // Smallest cosine and largest sine both correspond to θ_max; combine
        // via atan2 for full precision across [0, π/2]. Clamp the SVD outputs
        // into [0, 1] before pairing — both arise from singular values of
        // matrices whose true norms are ≤ 1, so any drift above 1 or below
        // 0 is pure floating-point noise.
        let min_cos = sv_cos
            .iter()
            .copied()
            .fold(1.0_f64, f64::min)
            .clamp(0.0, 1.0);
        let max_sin = sv_sin
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .clamp(0.0, 1.0);
        Ok(max_sin.atan2(min_cos))
    }
}

/// Streaming `p × r` cross-moment accumulator for the closed-form polar frame
/// update (issue #972). Sums decoder-target outer products `Σ_i t_i c_iᵀ`
/// (ambient target `t_i ∈ ℝ^p` against in-span coordinate `c_i ∈ ℝ^r`) so the
/// frame can be re-polared from accumulated evidence WITHOUT re-touching the
/// border. Accumulation is `O(p r)` per update and never forms a `p × p` matrix.
#[derive(Debug, Clone)]
pub struct GrassmannCrossMoment {
    moment: Array2<f64>,
}

impl GrassmannCrossMoment {
    /// Empty `p × r` accumulator.
    pub fn new(output_dim: usize, rank: usize) -> Self {
        Self {
            moment: Array2::<f64>::zeros((output_dim, rank)),
        }
    }

    /// Accumulate the full-batch cross-moment `Targetᵀ · Coords` where
    /// `targets` is `(N × p)` ambient decoder targets and `coords` is `(N × r)`
    /// in-span coordinates. `fast_atb` forms `Targetᵀ Coords` (`p × r`) directly.
    pub fn accumulate(
        &mut self,
        targets: ArrayView2<'_, f64>,
        coords: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        if targets.ncols() != self.moment.nrows() || coords.ncols() != self.moment.ncols() {
            return Err(format!(
                "GrassmannCrossMoment::accumulate: expected targets (·,{}) and coords (·,{}); \
                 got (·,{}) and (·,{})",
                self.moment.nrows(),
                self.moment.ncols(),
                targets.ncols(),
                coords.ncols()
            ));
        }
        if targets.nrows() != coords.nrows() {
            return Err(format!(
                "GrassmannCrossMoment::accumulate: targets rows {} must equal coords rows {}",
                targets.nrows(),
                coords.nrows()
            ));
        }
        let block = fast_atb(&targets.to_owned(), &coords.to_owned());
        self.moment += &block;
        Ok(())
    }

    /// Read the accumulated `p × r` cross-moment.
    pub fn moment(&self) -> ArrayView2<'_, f64> {
        self.moment.view()
    }

    /// Re-polar the frame from the accumulated cross-moment (the streaming
    /// closed-form step): `U_new = polar(Mcm)`.
    pub fn polar_frame(&self) -> Result<GrassmannFrame, String> {
        GrassmannFrame::polar_update(self.moment.view())
    }
}

/// Verification helper (issue #972): recover the planted low-rank column span of
/// an atom by polaring the decoder-target cross-moment and report the largest
/// principal angle (radians) between the recovered frame and a planted
/// orthonormal frame `planted` (`p × r`).
///
/// `targets` (`N × p`) are the ambient decoder targets and `coords` (`N × r`)
/// the latent coordinates that generated them (`targets ≈ coords · plantedᵀ`).
/// The closed-form polar of `Σ targetsᵀ coords` recovers `range(planted)`; a
/// successful low-rank fit drives the returned angle to `0`. Used by the
/// `planted_low_rank_frame_recovered_by_polar` test, and available to callers
/// that want a runtime span-recovery diagnostic.
pub fn grassmann_recover_planted_span_angle(
    targets: ArrayView2<'_, f64>,
    coords: ArrayView2<'_, f64>,
    planted: ArrayView2<'_, f64>,
) -> Result<f64, String> {
    let p = targets.ncols();
    let r = coords.ncols();
    if planted.dim() != (p, r) {
        return Err(format!(
            "grassmann_recover_planted_span_angle: planted frame must be ({p}, {r}); got {:?}",
            planted.dim()
        ));
    }
    let mut cross = GrassmannCrossMoment::new(p, r);
    cross.accumulate(targets, coords)?;
    let frame = cross.polar_frame()?;
    frame.max_principal_angle(planted)
}

/// Verification helper (issue #972): the factored arrow-Schur border dimension
/// equals `Σ_k M_k · r_k` exactly. Returns `Ok(())` iff the invariant holds for
/// `term`, else an explanatory error. Compiled-in so the border-size contract is
/// checkable at runtime, not only in tests.
pub fn grassmann_assert_border_dim_invariant(term: &SaeManifoldTerm) -> Result<(), String> {
    let expected: usize = term
        .atoms
        .iter()
        .map(|a| a.basis_size() * a.border_frame_rank())
        .sum();
    let got = term.factored_border_dim();
    if got != expected {
        return Err(format!(
            "grassmann border-dim invariant violated: factored_border_dim() = {got}, \
             expected Σ M_k·r_k = {expected}"
        ));
    }
    Ok(())
}
