use super::*;

/// Which radial kernel family is being used. Stored in the streaming operator
/// so that (q, t) scalars can be recomputed on the fly without a closure.
#[derive(Debug, Clone)]
pub enum RadialScalarKind {
    /// Matern kernel: (length_scale, nu).
    Matern { length_scale: f64, nu: MaternNu },
    /// Hybrid Duchon kernel: parameters needed for `duchon_radial_jets`.
    Duchon {
        length_scale: f64,
        p_order: usize,
        s_order: usize,
        dim: usize,
        coeffs: DuchonPartialFractionCoeffs,
    },
    /// Pure Duchon kernel: a single intrinsic polyharmonic block.
    PureDuchon {
        block_order: usize,
        p_order: usize,
        s_order: usize,
        dim: usize,
    },
    /// Thin-Plate Spline kernel: isotropic with a scalar length-scale, used
    /// only with `n_axes = 1` (`ScalarTotal` streaming mode). The chain rule
    /// for ψ = log κ = -log(length_scale) and r̃ = ‖x − c‖ gives
    /// ∂φ/∂ψ   = φ'(z)·z
    /// ∂²φ/∂ψ² = φ''(z)·z² + φ'(z)·z
    /// where z = r̃ / length_scale. With the operator's c=0, s_0 = r̃²,
    /// `shape_0 = q·s_0` and `shape_00 = t·s_0² + 2 q s_0` reproduce both
    /// derivatives exactly when q = φ'(r̃)/r̃ and t = (φ''(r̃) − q)/r̃².
    ThinPlate { length_scale: f64, dim: usize },
}


impl RadialScalarKind {
    /// Evaluate the `(phi, q, t)` radial scalars for a given distance `r`.
    ///
    /// `q = φ'(r)/r` and `t = (φ''(r) - q)/r²` (with the appropriate
    /// finite limits at `r → 0`). This is exactly the scalar pair needed
    /// to assemble the first and second derivatives of `Φ(t) = φ(‖t − c‖)`
    /// with respect to the input location `t`:
    ///
    /// ```text
    /// ∂Φ/∂t_a       = q · (t − c)_a
    /// ∂²Φ/∂t_a∂t_b  = q · δ_ab + t · (t − c)_a (t − c)_b
    /// ```
    ///
    /// Re-pointing the existing ψ-derivative machinery at the first kernel
    /// argument t (see `crate::terms::input_loc_derivatives`).
    ///
    /// Returns `true` iff both `q = φ'(r)/r` and `t = (φ''(r) − q)/r²` have
    /// finite limits as `r → 0+` for this kernel. When this returns `false`
    /// the design-row gradient/Hessian at a center collision (`r = 0`) is not
    /// defined by a single finite value; callers must either move off the
    /// collision or surface a `BasisError::DegenerateAtCollision`.
    ///
    /// Smoothness criteria used here (matching the analytic limits derived
    /// in this file and the comments on `eval_design_triplet`):
    ///   - Matérn ν = 1/2: `q = -s·E/r → -∞`, not smooth.
    ///   - Matérn ν = 3/2: `q` finite but `t = s³E/r → ∞`, not smooth.
    ///   - Matérn ν = 5/2, 7/2, 9/2: both finite, smooth.
    ///   - Duchon hybrid (`Duchon`): finite via the hybrid PFD identity;
    ///     the radial-jets routine produces a finite limit, so smooth.
    ///   - PureDuchon (raw polyharmonic block, exponent α = 2m − d):
    ///       non-log case and α ≥ 4 ⇒ both `q` and `t` vanish (smooth);
    ///       log case at any α, or α < 4 ⇒ at least one derivative diverges.
    ///   - ThinPlate dim = 1: φ = r³, `q = 3r → 0`, but `t = 3/r → ∞`. The
    ///     1-D Hessian formula `q·δ + t·s·s` at r = 0 has the only diagonal
    ///     entry contracted by `s_a = 0`, but the bare scalar limit is still
    ///     not finite, so we report it as non-smooth and let callers in 1-D
    ///     (where `s_a` literally vanishes) opt in by handling the error.
    ///     Dim 2 (log r), Dim 3 (-r) both diverge.
    #[inline]
    pub(crate) fn is_smooth_at_collision(&self) -> bool {
        match self {
            RadialScalarKind::Matern { nu, .. } => matches!(
                nu,
                MaternNu::FiveHalves | MaternNu::SevenHalves | MaternNu::NineHalves
            ),
            RadialScalarKind::Duchon { .. } => true,
            RadialScalarKind::PureDuchon {
                p_order,
                s_order,
                dim,
                ..
            } => {
                let alpha = duchon_scaling_exponent(*p_order, *s_order, *dim);
                let is_log = (*dim) % 2 == 0 && {
                    let half = (alpha / 2.0).round();
                    half >= 0.0 && (half * 2.0 - alpha).abs() < 1e-12
                };
                !is_log && alpha >= 4.0
            }
            RadialScalarKind::ThinPlate { .. } => false,
        }
    }

    pub(crate) fn eval_design_triplet(&self, r: f64) -> Result<(f64, f64, f64), BasisError> {
        match self {
            RadialScalarKind::Matern { length_scale, nu } => {
                let (phi, q, t, _, _) =
                    matern_aniso_extended_radial_scalars(r, *length_scale, *nu)?;
                Ok((phi, q, t))
            }
            RadialScalarKind::Duchon {
                length_scale,
                p_order,
                s_order,
                dim,
                coeffs,
            } => {
                let jets = duchon_radial_jets(r, *length_scale, *p_order, *s_order, *dim, coeffs)?;
                Ok((jets.phi, jets.q, jets.t))
            }
            RadialScalarKind::PureDuchon {
                block_order, dim, ..
            } => {
                let phi = polyharmonic_kernel(r, (*block_order) as f64, *dim);
                if r < 1e-14 {
                    // Collision: q = φ'/r and t = (φ'' − q)/r² generally
                    // diverge here. Only the non-log, α = 2m − d ≥ 4 case
                    // gives finite limits (both 0). Otherwise the design
                    // gradient/Hessian at r = 0 is undefined: surface a
                    // `DegenerateAtCollision` so callers can detect it.
                    if !self.is_smooth_at_collision() {
                        return Err(BasisError::DegenerateAtCollision {
                            kernel: "PureDuchon (polyharmonic)",
                            dim: *dim,
                            m: *block_order as f64,
                            message: "raw polyharmonic block φ(r) = c r^α (log r) is \
                                      not C² at r = 0 for α = 2m − d < 4 or for log \
                                      cases; first/second radial derivatives diverge",
                        });
                    }
                    return Ok((phi, 0.0, 0.0));
                }
                let (q, t, _, _) =
                    duchon_polyharmonic_operator_block_jets(r, *block_order as f64, *dim)?;
                Ok((phi, q, t))
            }
            RadialScalarKind::ThinPlate { length_scale, dim } => {
                // (q, t) individually diverge at r = 0 for ThinPlate
                // (q = 2 log r + 1 → −∞ in dim 2, q = −1/r → −∞ in dim 3,
                // t = 3/r → ∞ in dim 1, …) but the chain-rule coefficient
                // `c = raw_psi_isotropic_share` is 0 for ThinPlate, so every
                // consumer multiplies q by a squared displacement s_a and t
                // by s_a · s_b before use (design row uses φ alone, and
                // φ(0) = 0). The products
                //   q · s_a = (φ'(r) · r) · (s_a / r²),
                //   t · s_a · s_b = (φ''(r) · r² − φ'(r) · r) · (s_a/r²)·(s_b/r²)
                // both vanish as r → 0+, since r · φ'(r) → 0 and r² · φ''(r) → 0
                // for every standard TPS kernel (φ = r³ in dim 1, r² log r in
                // dim 2, −r in dim 3, and the general polyharmonic case for
                // d ≥ 4) and the ratios s_a/r², s_b/r² are bounded. The
                // closed-form ψ-derivative limit at the collision is
                // therefore (0, 0, 0).
                if r < 1e-14 {
                    return Ok((0.0, 0.0, 0.0));
                }
                let scaled_r = r / *length_scale;
                let (phi, phi_kernel_first, phi_kernel_second) =
                    thin_plate_kernel_triplet_from_scaled_distance(scaled_r, *dim)?;
                // The implicit operator uses derivatives w.r.t. the unscaled r
                // (the operator's chain rule will rescale them to ψ-derivatives
                // via s_0 = r²). Convert φ'(z), φ''(z) → φ'(r), φ''(r) by the
                // length-scale chain rule:
                //   φ'(r)  = φ'(z) / length_scale
                //   φ''(r) = φ''(z) / length_scale²
                let phi_r = phi_kernel_first / *length_scale;
                let phi_rr = phi_kernel_second / (*length_scale * *length_scale);
                let q = phi_r / r;
                let t = (phi_rr - q) / (r * r);
                Ok((phi, q, t))
            }
        }
    }

    #[inline]
    fn raw_psi_isotropic_share(&self) -> f64 {
        match self {
            RadialScalarKind::Matern { .. } => 0.0,
            RadialScalarKind::Duchon {
                p_order,
                s_order,
                dim,
                ..
            } => duchon_scaling_exponent(*p_order, *s_order, *dim) / *dim as f64,
            RadialScalarKind::PureDuchon {
                p_order,
                s_order,
                dim,
                ..
            } => duchon_scaling_exponent(*p_order, *s_order, *dim) / *dim as f64,
            // ThinPlate is a pure radial kernel φ(z) with no κ^δ prefactor;
            // the chain rule has no isotropic share term.
            RadialScalarKind::ThinPlate { .. } => 0.0,
        }
    }

    #[inline]
    fn is_duchon_family(&self) -> bool {
        matches!(
            self,
            RadialScalarKind::Duchon { .. } | RadialScalarKind::PureDuchon { .. }
        )
    }

    /// Whether the radial-kind enforces a hard guard against accidental
    /// dense `(n × p)` ψ-derivative materialization. Duchon-family terms
    /// always do (they are streaming-only at any scale). ThinPlate joins
    /// the guard list because the new scalar-streaming routing makes it
    /// genuine to rely on the implicit operator at large scale, and a
    /// downstream consumer that sneaks in a `materialize_dense()` call
    /// would silently re-introduce the same `n × p` allocation we wired
    /// streaming to avoid. The guard panics only when the resource
    /// policy says the materialization would exceed budget — small `n`
    /// problems still get the dense fast path.
    #[inline]
    fn enforces_dense_materialization_budget(&self) -> bool {
        matches!(
            self,
            RadialScalarKind::Duchon { .. }
                | RadialScalarKind::PureDuchon { .. }
                | RadialScalarKind::ThinPlate { .. }
        )
    }
}


/// Shared chunked-operator machinery for the streaming basis evaluators.
///
/// `StreamingMaternEvaluator`, `StreamingSphereEvaluator` and
/// `StreamingBSplineEvaluator` differ only in how a single row chunk of the
/// design is materialized (`for_row_chunk`) and the chunk size policy
/// (`chunk_rows`); every other operator method — the chunked matvec,
/// transpose-matvec, weighted Gram and dense materialization — is identical
/// boilerplate over that one primitive. This trait carries those shared
/// methods as defaults keyed on `for_row_chunk`, so each evaluator implements
/// only the per-basis pieces. The `NAME` const bakes the struct name into the
/// panic/error strings so diagnostics stay per-evaluator.
trait ChunkedDesign {
    /// Struct name used in assertion / error messages.
    const NAME: &'static str;

    /// Number of design rows (observations).
    fn op_nrows(&self) -> usize;

    /// Number of design columns (basis functions after any transform).
    fn op_ncols(&self) -> usize;

    /// Row-block size used to bound the per-chunk working set.
    fn chunk_rows(&self) -> usize;

    /// Materialize the dense design rows `[start, end)` — the only genuinely
    /// per-evaluator computation.
    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64>;

    /// Chunked matvec `output = X · theta`.
    fn chunked_gradient_into(&self, theta: ArrayView1<'_, f64>, output: &mut Array1<f64>) {
        assert_eq!(
            theta.len(),
            self.op_ncols(),
            "{} theta width mismatch",
            Self::NAME
        );
        assert_eq!(
            output.len(),
            self.op_nrows(),
            "{} output length mismatch",
            Self::NAME
        );
        output.fill(0.0);
        let nrows = self.op_nrows();
        for start in (0..nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(nrows);
            let chunk = self.for_row_chunk(start, end);
            let values = chunk.dot(&theta);
            output.slice_mut(s![start..end]).assign(&values);
        }
    }

    /// Chunked matvec returning a fresh vector (`LinearOperator::apply`).
    fn chunked_apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.op_nrows());
        self.chunked_gradient_into(vector.view(), &mut out);
        out
    }

    /// Chunked transpose-matvec `out = Xᵀ · vector`
    /// (`LinearOperator::apply_transpose`).
    fn chunked_apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            vector.len(),
            self.op_nrows(),
            "{} transpose vector length mismatch",
            Self::NAME
        );
        let nrows = self.op_nrows();
        let mut out = Array1::<f64>::zeros(self.op_ncols());
        for start in (0..nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(nrows);
            let chunk = self.for_row_chunk(start, end);
            let partial = chunk.t().dot(&vector.slice(s![start..end]));
            out += &partial;
        }
        out
    }

    /// Chunked weighted Gram `XᵀWX` (`LinearOperator::diag_xtw_x`).
    fn chunked_diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String>
    where
        Self: Sync,
    {
        let nrows = self.op_nrows();
        if weights.len() != nrows {
            return Err(format!(
                "{} diag_xtw_x weight length mismatch: weights={}, nrows={}",
                Self::NAME,
                weights.len(),
                nrows
            ));
        }
        let p = self.op_ncols();
        let chunk_rows = self.chunk_rows();
        let starts = (0..nrows).step_by(chunk_rows).collect::<Vec<_>>();
        Ok(starts
            .into_par_iter()
            .fold(
                || Array2::<f64>::zeros((p, p)),
                |mut acc, start| {
                    let end = (start + chunk_rows).min(nrows);
                    let chunk = self.for_row_chunk(start, end);
                    let mut weighted = chunk.clone();
                    for local in 0..(end - start) {
                        let w = weights[start + local];
                        weighted.row_mut(local).mapv_inplace(|v| v * w);
                    }
                    acc += &chunk.t().dot(&weighted);
                    acc
                },
            )
            .reduce(
                || Array2::<f64>::zeros((p, p)),
                |mut a, b| {
                    a += &b;
                    a
                },
            ))
    }

    /// Chunked dense row fill (`DenseDesignOperator::row_chunk_into`).
    fn chunked_row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if rows.end > self.op_nrows() || rows.start > rows.end {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: Self::ROW_RANGE_OOB,
            });
        }
        if out.nrows() != rows.end - rows.start || out.ncols() != self.op_ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: Self::ROW_CHUNK_SHAPE_MISMATCH,
            });
        }
        out.assign(&self.for_row_chunk(rows.start, rows.end));
        Ok(())
    }

    /// Full dense materialization (`DenseDesignOperator::to_dense`).
    fn chunked_to_dense(&self) -> Array2<f64> {
        self.for_row_chunk(0, self.op_nrows())
    }

    /// Static `&str` context strings for the row-chunk errors — kept as
    /// associated consts because `MatrixMaterializationError::MissingRowChunk`
    /// stores `&'static str`, so a runtime-formatted name cannot be used.
    const ROW_RANGE_OOB: &'static str;
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str;
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingMaternEvaluator {
    data: Arc<Array2<f64>>,
    centers: Arc<Array2<f64>>,
    length_scale: f64,
    nu: MaternNu,
    metric_weights: Arc<[f64]>,
    ident_transform: Option<Arc<Array2<f64>>>,
    include_intercept: bool,
    chunk_size: usize,
    total_cols: usize,
}


impl StreamingMaternEvaluator {
    pub(crate) fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        length_scale: f64,
        nu: MaternNu,
        aniso_log_scales: Option<Vec<f64>>,
        ident_transform: Option<Arc<Array2<f64>>>,
        include_intercept: bool,
        chunk_size: Option<usize>,
    ) -> Result<Self, String> {
        if data.ncols() != centers.ncols() {
            return Err(format!(
                "StreamingMaternEvaluator: data dim {} != centers dim {}",
                data.ncols(),
                centers.ncols()
            ));
        }
        let metric_weights = match aniso_log_scales {
            Some(eta) => {
                if eta.len() != data.ncols() {
                    return Err(format!(
                        "StreamingMaternEvaluator: aniso_log_scales len {} != data dim {}",
                        eta.len(),
                        data.ncols()
                    ));
                }
                eta.into_iter().map(|v| (2.0 * v).exp()).collect::<Vec<_>>()
            }
            None => vec![1.0; data.ncols()],
        };
        if let Some(z) = ident_transform.as_ref()
            && z.nrows() != centers.nrows()
        {
            return Err(format!(
                "StreamingMaternEvaluator: identifiability transform rows {} != centers {}",
                z.nrows(),
                centers.nrows()
            ));
        }
        let kernel_cols = ident_transform
            .as_ref()
            .map_or(centers.nrows(), |z| z.ncols());
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            centers: Arc::new(centers.as_standard_layout().to_owned()),
            length_scale,
            nu,
            metric_weights: Arc::from(metric_weights),
            ident_transform,
            include_intercept,
            chunk_size: chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1),
            total_cols: kernel_cols + usize::from(include_intercept),
        })
    }

    fn raw_kernel_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k_raw = self.centers.nrows();
        let dim = self.data.ncols();
        let data = self
            .data
            .as_slice()
            .expect("StreamingMaternEvaluator stores standard-layout data");
        let centers = self
            .centers
            .as_slice()
            .expect("StreamingMaternEvaluator stores standard-layout centers");
        let mut values = vec![0.0_f64; chunk_n * k_raw];
        values
            .par_chunks_mut(k_raw)
            .enumerate()
            .for_each(|(local, out_row)| {
                let global = rows.start + local;
                let x = &data[global * dim..(global + 1) * dim];
                for j in 0..k_raw {
                    let c = &centers[j * dim..(j + 1) * dim];
                    let mut r2 = 0.0_f64;
                    for axis in 0..dim {
                        let h = x[axis] - c[axis];
                        r2 += self.metric_weights[axis] * h * h;
                    }
                    out_row[j] = matern_kernel_from_distance(r2.sqrt(), self.length_scale, self.nu)
                        .expect("validated Matérn inputs should not fail");
                }
            });
        Array2::from_shape_vec((chunk_n, k_raw), values)
            .expect("StreamingMaternEvaluator chunk shape should match generated values")
    }

    fn for_row_chunk_impl(&self, start: usize, end: usize) -> Array2<f64> {
        let raw = self.raw_kernel_chunk(start..end);
        let kernel = match self.ident_transform.as_ref() {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        };
        if !self.include_intercept {
            return kernel;
        }
        let mut out = Array2::<f64>::ones((end - start, kernel.ncols() + 1));
        out.slice_mut(s![.., ..kernel.ncols()]).assign(&kernel);
        out
    }
}


impl ChunkedDesign for StreamingMaternEvaluator {
    const NAME: &'static str = "StreamingMaternEvaluator";
    const ROW_RANGE_OOB: &'static str = "StreamingMaternEvaluator row range out of bounds";
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str =
        "StreamingMaternEvaluator row_chunk_into shape mismatch";

    fn op_nrows(&self) -> usize {
        self.data.nrows()
    }

    fn op_ncols(&self) -> usize {
        self.total_cols
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.data.nrows().max(1))
    }

    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        assert!(
            start <= end && end <= self.data.nrows(),
            "StreamingMaternEvaluator row chunk out of bounds"
        );
        self.for_row_chunk_impl(start, end)
    }
}


impl LinearOperator for StreamingMaternEvaluator {
    fn nrows(&self) -> usize {
        self.op_nrows()
    }

    fn ncols(&self) -> usize {
        self.op_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply_transpose(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.chunked_diag_xtw_x(weights)
    }
}


impl DenseDesignOperator for StreamingMaternEvaluator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        self.chunked_row_chunk_into(rows, out)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.chunked_to_dense()
    }
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingSphereEvaluator {
    data: Arc<Array2<f64>>,
    centers: Arc<Array2<f64>>,
    penalty_order: usize,
    radians: bool,
    wahba_kernel: SphereWahbaKernel,
    constraint_transform: Option<Arc<Array2<f64>>>,
    sin_lat_c: Arc<[f64]>,
    cos_lat_c: Arc<[f64]>,
    sin_lon_c: Arc<[f64]>,
    cos_lon_c: Arc<[f64]>,
    chunk_size: usize,
    total_cols: usize,
}


impl StreamingSphereEvaluator {
    pub(crate) fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        penalty_order: usize,
        radians: bool,
        wahba_kernel: SphereWahbaKernel,
        constraint_transform: Option<Arc<Array2<f64>>>,
        chunk_size: Option<usize>,
    ) -> Result<Self, String> {
        validate_lat_lon_matrix(data.view(), "StreamingSphereEvaluator data", radians)
            .map_err(|e| e.to_string())?;
        validate_lat_lon_matrix(centers.view(), "StreamingSphereEvaluator centers", radians)
            .map_err(|e| e.to_string())?;
        if !(1..=4).contains(&penalty_order) {
            return Err(format!(
                "StreamingSphereEvaluator: penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
            ));
        }
        if let Some(z) = constraint_transform.as_ref()
            && z.nrows() != centers.nrows()
        {
            return Err(format!(
                "StreamingSphereEvaluator: constraint transform rows {} != centers {}",
                z.nrows(),
                centers.nrows()
            ));
        }
        let deg = if radians {
            1.0
        } else {
            std::f64::consts::PI / 180.0
        };
        let mut sin_lat_c = Vec::<f64>::with_capacity(centers.nrows());
        let mut cos_lat_c = Vec::<f64>::with_capacity(centers.nrows());
        let mut sin_lon_c = Vec::<f64>::with_capacity(centers.nrows());
        let mut cos_lon_c = Vec::<f64>::with_capacity(centers.nrows());
        for c in centers.outer_iter() {
            let (s_lat, c_lat) = (c[0] * deg).sin_cos();
            let (s_lon, c_lon) = (c[1] * deg).sin_cos();
            sin_lat_c.push(s_lat);
            cos_lat_c.push(c_lat);
            sin_lon_c.push(s_lon);
            cos_lon_c.push(c_lon);
        }
        let total_cols = constraint_transform
            .as_ref()
            .map_or(centers.nrows(), |z| z.ncols());
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            centers: Arc::new(centers.as_standard_layout().to_owned()),
            penalty_order,
            radians,
            wahba_kernel,
            constraint_transform,
            sin_lat_c: Arc::from(sin_lat_c),
            cos_lat_c: Arc::from(cos_lat_c),
            sin_lon_c: Arc::from(sin_lon_c),
            cos_lon_c: Arc::from(cos_lon_c),
            chunk_size: chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1),
            total_cols,
        })
    }

    fn raw_kernel_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k = self.centers.nrows();
        let deg = if self.radians {
            1.0
        } else {
            std::f64::consts::PI / 180.0
        };
        let mut values = vec![0.0_f64; chunk_n * k];
        values
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(local, out_row)| {
                use wide::f64x4;
                let row = rows.start + local;
                let lat = self.data[[row, 0]] * deg;
                let lon = self.data[[row, 1]] * deg;
                let (sin_lat, cos_lat) = lat.sin_cos();
                let (sin_lon, cos_lon) = lon.sin_cos();
                let sin_lat_v = f64x4::from(sin_lat);
                let cos_lat_v = f64x4::from(cos_lat);
                let sin_lon_v = f64x4::from(sin_lon);
                let cos_lon_v = f64x4::from(cos_lon);
                for cidx in 0..(k / 4) {
                    let base = cidx * 4;
                    let sl_c = f64x4::from([
                        self.sin_lat_c[base],
                        self.sin_lat_c[base + 1],
                        self.sin_lat_c[base + 2],
                        self.sin_lat_c[base + 3],
                    ]);
                    let cl_c = f64x4::from([
                        self.cos_lat_c[base],
                        self.cos_lat_c[base + 1],
                        self.cos_lat_c[base + 2],
                        self.cos_lat_c[base + 3],
                    ]);
                    let sn_c = f64x4::from([
                        self.sin_lon_c[base],
                        self.sin_lon_c[base + 1],
                        self.sin_lon_c[base + 2],
                        self.sin_lon_c[base + 3],
                    ]);
                    let cn_c = f64x4::from([
                        self.cos_lon_c[base],
                        self.cos_lon_c[base + 1],
                        self.cos_lon_c[base + 2],
                        self.cos_lon_c[base + 3],
                    ]);
                    let dlon_cos = cos_lon_v * cn_c + sin_lon_v * sn_c;
                    let cos_gamma = sin_lat_v * sl_c + cos_lat_v * cl_c * dlon_cos;
                    let arr = wahba_sphere_kernel_from_cos_simd_kind(
                        cos_gamma,
                        self.penalty_order,
                        self.wahba_kernel,
                    )
                    .to_array();
                    out_row[base..base + 4].copy_from_slice(&arr);
                }
                let tail_start = (k / 4) * 4;
                for j in tail_start..k {
                    let dlon_cos = cos_lon * self.cos_lon_c[j] + sin_lon * self.sin_lon_c[j];
                    let cos_gamma =
                        sin_lat * self.sin_lat_c[j] + cos_lat * self.cos_lat_c[j] * dlon_cos;
                    out_row[j] = wahba_sphere_kernel_from_cos_kind(
                        cos_gamma,
                        self.penalty_order,
                        self.wahba_kernel,
                    )
                    .expect("validated sphere kernel inputs should not fail");
                }
            });
        Array2::from_shape_vec((chunk_n, k), values)
            .expect("StreamingSphereEvaluator chunk shape should match generated values")
    }

    fn for_row_chunk_impl(&self, start: usize, end: usize) -> Array2<f64> {
        let raw = self.raw_kernel_chunk(start..end);
        match self.constraint_transform.as_ref() {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        }
    }
}


impl ChunkedDesign for StreamingSphereEvaluator {
    const NAME: &'static str = "StreamingSphereEvaluator";
    const ROW_RANGE_OOB: &'static str = "StreamingSphereEvaluator row range out of bounds";
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str =
        "StreamingSphereEvaluator row_chunk_into shape mismatch";

    fn op_nrows(&self) -> usize {
        self.data.nrows()
    }

    fn op_ncols(&self) -> usize {
        self.total_cols
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.data.nrows().max(1))
    }

    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        assert!(
            start <= end && end <= self.data.nrows(),
            "StreamingSphereEvaluator row chunk out of bounds"
        );
        self.for_row_chunk_impl(start, end)
    }
}


impl LinearOperator for StreamingSphereEvaluator {
    fn nrows(&self) -> usize {
        self.op_nrows()
    }

    fn ncols(&self) -> usize {
        self.op_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply_transpose(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.chunked_diag_xtw_x(weights)
    }
}


impl DenseDesignOperator for StreamingSphereEvaluator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        self.chunked_row_chunk_into(rows, out)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.chunked_to_dense()
    }
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingBSplineEvaluator {
    data: Arc<Array1<f64>>,
    knots: Arc<Array1<f64>>,
    degree: usize,
    periodic: Option<(f64, f64, usize)>,
    transform: Option<Arc<Array2<f64>>>,
    chunk_size: usize,
    total_cols: usize,
}


impl StreamingBSplineEvaluator {
    pub(crate) fn new(
        data: Arc<Array1<f64>>,
        knots: Arc<Array1<f64>>,
        degree: usize,
        periodic: Option<(f64, f64, usize)>,
        transform: Option<Arc<Array2<f64>>>,
        chunk_size: Option<usize>,
    ) -> Result<Self, String> {
        let raw_cols = bspline_raw_column_count(knots.as_ref(), degree, periodic)?;
        if let Some(z) = transform.as_ref()
            && z.nrows() != raw_cols
        {
            return Err(format!(
                "StreamingBSplineEvaluator: transform rows {} != raw basis columns {}",
                z.nrows(),
                raw_cols
            ));
        }
        Ok(Self {
            data: Arc::new(data.as_standard_layout().to_owned()),
            knots: Arc::new(knots.as_standard_layout().to_owned()),
            degree,
            periodic,
            total_cols: transform.as_ref().map_or(raw_cols, |z| z.ncols()),
            transform,
            chunk_size: chunk_size.unwrap_or(DEFAULT_STREAMING_CHUNK_ROWS).max(1),
        })
    }

    fn raw_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        bspline_raw_row_chunk(
            self.data.view(),
            self.knots.view(),
            self.degree,
            self.periodic,
            start,
            end,
        )
        .expect("StreamingBSplineEvaluator validated inputs should build row chunks")
    }

    fn for_row_chunk_impl(&self, start: usize, end: usize) -> Array2<f64> {
        let raw = self.raw_chunk(start, end);
        match self.transform.as_ref() {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        }
    }
}


impl ChunkedDesign for StreamingBSplineEvaluator {
    const NAME: &'static str = "StreamingBSplineEvaluator";
    const ROW_RANGE_OOB: &'static str = "StreamingBSplineEvaluator row range out of bounds";
    const ROW_CHUNK_SHAPE_MISMATCH: &'static str =
        "StreamingBSplineEvaluator row_chunk_into shape mismatch";

    fn op_nrows(&self) -> usize {
        self.data.len()
    }

    fn op_ncols(&self) -> usize {
        self.total_cols
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.data.len().max(1))
    }

    fn for_row_chunk(&self, start: usize, end: usize) -> Array2<f64> {
        assert!(
            start <= end && end <= self.data.len(),
            "StreamingBSplineEvaluator row chunk out of bounds"
        );
        self.for_row_chunk_impl(start, end)
    }
}


impl LinearOperator for StreamingBSplineEvaluator {
    fn nrows(&self) -> usize {
        self.op_nrows()
    }

    fn ncols(&self) -> usize {
        self.op_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.chunked_apply_transpose(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.chunked_diag_xtw_x(weights)
    }
}


impl DenseDesignOperator for StreamingBSplineEvaluator {
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        self.chunked_row_chunk_into(rows, out)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.chunked_to_dense()
    }
}


/// Data stored for streaming (on-the-fly) recomputation of radial jet scalars.
/// Instead of persisting O(n*k*(d+2)) arrays, the operator stores the original
/// data/centers/eta and recomputes q/t/s per chunk during matvec operations.
#[derive(Debug, Clone)]
pub(crate) enum StreamingAxisMode {
    /// Per-axis anisotropic ψ_a derivatives: expose one `s_a` component per axis.
    PerAxis { metric_weights: Arc<[f64]> },
    /// Scalar ψ derivative: expose a single component equal to the total
    /// scaled squared radius r² = Σ_a exp(2η_a) h_a².
    ScalarTotal { metric_weights: Arc<[f64]> },
}


#[derive(Debug, Clone)]
pub(crate) struct StreamingRadialState {
    /// Data matrix, shape (n, d).
    data: Arc<Array2<f64>>,
    /// Center matrix, shape (k, d).
    centers: Arc<Array2<f64>>,
    /// How per-pair axis components are exposed to the derivative operator.
    axis_mode: StreamingAxisMode,
    /// Which radial kernel family to use for recomputation.
    radial_kind: RadialScalarKind,
    /// Lazily materialized radial-scalar cache. (phi, q, t) per (i, j) pair
    /// — independent of axis, identical across every per-axis chunk loop —
    /// so collapses (axes × calls × chunks × n × n_knots) streaming radial
    /// evaluations into a single O(n × n_knots) sweep per operator. The
    /// inner `Option` is `None` when the parallel fill encountered a radial
    /// evaluation error (e.g. a non-finite r); callers fall back to the
    /// streaming path which propagates the error through `compute_pair`.
    triplet_cache: Arc<std::sync::OnceLock<Option<StreamingTripletCache>>>,
}


#[derive(Debug)]
pub(crate) struct StreamingTripletCache {
    phi: Vec<f64>,
    q: Vec<f64>,
    t: Vec<f64>,
}


/// Memory cap (bytes) above which we keep streaming the radial scalars
/// instead of materializing the (phi, q, t) triplet cache. Three `Vec<f64>`
/// arrays of length `n × n_knots` consume `24 × n × n_knots` bytes; the cap
/// keeps the resident footprint bounded for designs that would blow past a
/// few hundred MiB.
pub(crate) const STREAMING_TRIPLET_CACHE_BYTE_BUDGET: usize = 1 << 30;


impl StreamingRadialState {
    fn cache_fits_budget(&self) -> bool {
        let total = self
            .data
            .nrows()
            .saturating_mul(self.centers.nrows())
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_mul(3);
        total <= STREAMING_TRIPLET_CACHE_BYTE_BUDGET
    }

    fn ensure_triplet_cache(&self) -> Option<&StreamingTripletCache> {
        if !self.cache_fits_budget() {
            return None;
        }
        let n = self.data.nrows();
        let n_knots = self.centers.nrows();
        if n == 0 || n_knots == 0 {
            return None;
        }
        // The OnceLock holds `Option<StreamingTripletCache>` so a fill that
        // hits an invalid `eval_design_triplet` (e.g. a non-finite r) does
        // not poison the cache silently — consumers see `None` and fall back
        // to the streaming `compute_pair` path that propagates the error
        // through `Result<…, BasisError>`.
        self.triplet_cache
            .get_or_init(|| self.materialize_triplet_cache())
            .as_ref()
    }

    fn materialize_triplet_cache(&self) -> Option<StreamingTripletCache> {
        let n = self.data.nrows();
        let n_knots = self.centers.nrows();
        let total = n * n_knots;
        let mut phi = vec![0.0_f64; total];
        let mut q = vec![0.0_f64; total];
        let mut t = vec![0.0_f64; total];

        let metric_weights: &[f64] = match &self.axis_mode {
            StreamingAxisMode::PerAxis { metric_weights }
            | StreamingAxisMode::ScalarTotal { metric_weights } => metric_weights,
        };
        let dim = metric_weights.len();
        assert_eq!(dim, self.data.ncols());
        assert_eq!(dim, self.centers.ncols());

        // SERIAL fill: `ensure_triplet_cache` is called from inside outer
        // `into_par_iter` workers (e.g. the per-axis cross-trace sweep at
        // `projected_operator_terms_batched`). A nested `par_chunks_mut`
        // inside this `OnceLock::get_or_init` closure would deadlock the
        // global rayon pool — every outer worker blocks on the OnceLock
        // while the one that won the race tries to schedule child tasks no
        // worker is free to pick up (see `feedback_oncelock_rayon_deadlock`).
        //
        // The serial sweep is only affordable when the per-pair radial
        // evaluation is cheap. For the 16-D power-9 hybrid Duchon kernel a
        // single exact `eval_design_triplet` costs tens of microseconds
        // across its partial-fraction blocks, and at the large-scale
        // conditional-PGS shape (n·k ≈ 480k pairs) this loop was ~15–20 s
        // of single-threaded work per κ-trial — the dominant cost of the
        // whole CTN stage-1 fit (#979; the cost model in the previous
        // version of this comment assumed a cheap kernel). For large sweeps
        // we therefore build a certified 1-D Chebyshev radial profile once
        // (a few hundred exact evaluations, see `radial_profile`) from a
        // distance-only pre-pass over the radius range, and answer per-pair
        // queries with a Clenshaw contraction; out-of-range or uncertified
        // cases fall back to the exact evaluator per pair.
        let pair_radius = |i: usize, j: usize| -> f64 {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                // Streaming constructors set n=data.nrows(), n_knots=centers.nrows(),
                // and require dim=data.ncols()=centers.ncols(); the loop ranges
                // therefore keep both uget reads in-bounds.
                let h = unsafe { self.data.uget((i, a)) - self.centers.uget((j, a)) }; // SAFETY: bounds per the comment immediately above
                r2 += metric_weights[a] * h * h;
            }
            r2.sqrt()
        };
        let profile = if total >= RADIAL_PROFILE_MIN_PAIRS {
            let mut r_lo = f64::INFINITY;
            let mut r_hi = 0.0_f64;
            for i in 0..n {
                for j in 0..n_knots {
                    let r = pair_radius(i, j);
                    if r > 0.0 {
                        r_lo = r_lo.min(r);
                        r_hi = r_hi.max(r);
                    }
                }
            }
            if r_lo.is_finite() && r_hi > r_lo {
                radial_profile::RadialProfile::build(&self.radial_kind, r_lo, r_hi)
            } else {
                None
            }
        } else {
            None
        };
        for i in 0..n {
            let row_off = i * n_knots;
            for j in 0..n_knots {
                let r = pair_radius(i, j);
                let triplet = match profile.as_ref() {
                    Some(profile) => profile.eval_or_exact(&self.radial_kind, r),
                    None => self.radial_kind.eval_design_triplet(r),
                };
                match triplet {
                    Ok((pv, qv, tv)) => {
                        phi[row_off + j] = pv;
                        q[row_off + j] = qv;
                        t[row_off + j] = tv;
                    }
                    Err(_) => return None,
                }
            }
        }
        Some(StreamingTripletCache { phi, q, t })
    }

    #[inline]
    fn fill_s_buf(&self, i: usize, j: usize, s_buf: &mut [f64]) {
        match &self.axis_mode {
            StreamingAxisMode::PerAxis { metric_weights } => {
                let dim = metric_weights.len();
                assert_eq!(s_buf.len(), dim);
                for a in 0..dim {
                    // SAFETY: compute_pair/ensure_triplet_cache callers pass i <
                    // data.nrows() and j < centers.nrows(); streaming constructors
                    // require dim=data.ncols()=centers.ncols(), and this loop has a < dim.
                    let h = unsafe { self.data.uget((i, a)) - self.centers.uget((j, a)) };
                    s_buf[a] = metric_weights[a] * h * h;
                }
            }
            StreamingAxisMode::ScalarTotal { metric_weights } => {
                assert_eq!(s_buf.len(), 1);
                let dim = metric_weights.len();
                let mut r2 = 0.0;
                for a in 0..dim {
                    // SAFETY: compute_pair/ensure_triplet_cache callers pass i <
                    // data.nrows() and j < centers.nrows(); streaming constructors
                    // require dim=data.ncols()=centers.ncols(), and this loop has a < dim.
                    let h = unsafe { self.data.uget((i, a)) - self.centers.uget((j, a)) };
                    r2 += metric_weights[a] * h * h;
                }
                s_buf[0] = r2;
            }
        }
    }

    /// Compute `(phi, q, t, s_a[0..d])` for a single `(data_row i, center j)` pair.
    ///
    /// Returns `(phi, q, t)` and writes per-axis components into `s_buf` (length d).
    #[inline]
    fn compute_pair(
        &self,
        i: usize,
        j: usize,
        s_buf: &mut [f64],
    ) -> Result<(f64, f64, f64), BasisError> {
        assert!(i < self.data.nrows() && j < self.centers.nrows());
        self.fill_s_buf(i, j, s_buf);
        match &self.axis_mode {
            StreamingAxisMode::PerAxis { metric_weights } => {
                let r2: f64 = (0..metric_weights.len()).map(|a| s_buf[a]).sum();
                self.radial_kind.eval_design_triplet(r2.sqrt())
            }
            StreamingAxisMode::ScalarTotal { .. } => {
                let r2 = s_buf[0];
                self.radial_kind.eval_design_triplet(r2.sqrt())
            }
        }
    }
}
