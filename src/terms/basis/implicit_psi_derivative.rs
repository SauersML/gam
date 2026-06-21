use super::*;

/// Implicit representation of ∂X/∂ψ_d that supports matrix-vector products
/// without materializing the full (n x p) derivative matrices.
///
/// For anisotropic Matern / Duchon terms with D axes, the dense path creates
/// D matrices of size (n x p_smooth) for dX/dpsi_d. At n=400K, p=2000, D=16,
/// that is ~100 GB.
///
/// Two storage modes:
///
/// **Materialized** (small-to-medium problems): stores pre-computed arrays
/// - `phi_values[i*n_knots + j]` = phi(r_{ij})
/// - `q_values[i*n_knots + j]` = phi'(r_{ij}) / r_{ij}
/// - `t_values[i*n_knots + j]` = (phi''(r_{ij}) - q_{ij}) / r_{ij}^2
/// - `axis_components[i*n_knots + j, d]` = exp(2 eta_d) * (x_{id} - c_{jd})^2
/// Memory: O(n * k * (D + 2)).
///
/// **Streaming** (large scale): stores only data/centers/eta/kernel params
/// and recomputes (q, t, s_a) on the fly during each matvec.
/// Memory: O(n*d + k*d) -- no per-(data,knot) storage.
///
/// The raw-psi chain rule:
///   shape_a   = q * s_a
///   shape_ab  = t * s_a * s_b + 2 q s_a 1[a=b]
///   dphi/dpsi_a         = shape_a + c * phi
///   d2phi/(dpsi_a dpsi_b) = shape_ab + c (shape_a + shape_b) + c^2 phi
/// where `c = 0` for Matérn and `c = delta / d` for hybrid Duchon.
#[derive(Debug, Clone)]
pub struct ImplicitDesignPsiDerivative {
    /// Pre-computed kernel values (materialized mode).
    /// Shape: (n * n_knots,). Empty in streaming mode.
    pub(crate) phi_values: Array1<f64>,

    /// Pre-computed per (data, knot) pair axis components (materialized mode).
    /// Shape: (n * n_knots, D) stored in row-major order.
    /// Empty (0x0) in streaming mode.
    pub(crate) axis_components: Array2<f64>,

    /// Pre-computed R-operator first scalar (materialized mode).
    /// Shape: (n * n_knots,). Empty in streaming mode.
    pub(crate) q_values: Array1<f64>,

    /// Pre-computed R-operator second scalar (materialized mode).
    /// Shape: (n * n_knots,). Empty in streaming mode.
    pub(crate) t_values: Array1<f64>,

    /// When set, enables streaming recomputation of q/t/s from raw inputs
    /// instead of reading from the pre-computed arrays above.
    pub(crate) streaming: Option<StreamingRadialState>,

    /// Identifiability/constraint transform Z: (n_knots x p_constrained).
    /// Gauge ownership is upstream; the implicit operator stores this frozen
    /// section only so forward/transpose matvecs can apply the already-gauged
    /// chart without materializing derivative matrices. For Duchon this is the
    /// kernel-constraint nullspace Z_kernel; for Matern with identifiability
    /// constraints, it is the corresponding Z. `None` means the identity.
    pub(crate) ident_transform: Option<Array2<f64>>,

    /// Optional full identifiability transform applied after Z_kernel + padding.
    /// This is likewise replay/application metadata for the matrix-free
    /// operator, not a second coefficient-coordinate owner. For Duchon terms
    /// that have an additional global identifiability transform, this is applied
    /// after the kernel constraint and polynomial padding.
    /// Shape: (p_constrained + n_poly, p_final).
    pub(crate) full_ident_transform: Option<Array2<f64>>,

    /// Number of data points.
    pub(crate) n: usize,

    /// Number of knots (raw basis functions before identifiability transform).
    pub(crate) n_knots: usize,

    /// Number of polynomial columns appended after the smooth part.
    /// These have zero derivative with respect to psi_d.
    pub(crate) n_poly: usize,

    /// Number of axes (dimension D).
    pub(crate) n_axes: usize,

    /// Isotropic scaling contribution per raw anisotropic psi axis.
    pub(crate) psi_scale_share: f64,

    /// Optional exposed-axis to raw-axis linear combinations.
    /// When present, axis `a` represents Σ_i coeff_i * raw_axis_i.
    pub(crate) axis_combinations: Option<Vec<Vec<(usize, f64)>>>,
}

/// Streaming design derivative for one per-row latent coordinate `t[n, a]`.
///
/// The operator stores the shared latent matrix plus either radial-kernel
/// ingredients or a precomputed non-radial derivative jet. Individual REML
/// hyper-directions carry only a flat coordinate index and call
/// `forward_mul_axis` / `transpose_mul_axis` to expose the corresponding
/// one-row design derivative on demand.
pub struct LatentCoordDesignDerivative {
    pub(crate) provider: Arc<dyn LocalDesignJacobianProvider>,
}

#[derive(Debug, Clone)]
pub(crate) struct RadialLatentCoordLocalDesignJacobian {
    pub(crate) latent: Arc<crate::terms::latent::LatentCoordValues>,
    pub(crate) centers: Arc<Array2<f64>>,
    pub(crate) radial_kind: RadialScalarKind,
    pub(crate) ident_transform: Option<Array2<f64>>,
    pub(crate) full_ident_transform: Option<Array2<f64>>,
    pub(crate) n_poly: usize,
    pub(crate) polynomial_order: Option<DuchonNullspaceOrder>,
}

#[derive(Debug, Clone)]
pub(crate) struct JetLatentCoordLocalDesignJacobian {
    pub(crate) latent: Arc<crate::terms::latent::LatentCoordValues>,
    pub(crate) jet: Arc<Array3<f64>>,
    pub(crate) ident_transform: Option<Array2<f64>>,
}

impl std::fmt::Debug for LatentCoordDesignDerivative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LatentCoordDesignDerivative")
            .field("n_data", &self.n_data())
            .field("latent_dim", &self.latent_dim())
            .field("n_axes", &self.n_axes())
            .field("p_out", &self.p_out())
            .field("provider", &self.provider)
            .finish()
    }
}

impl Clone for LatentCoordDesignDerivative {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
        }
    }
}

impl RadialLatentCoordLocalDesignJacobian {
    pub(crate) fn p_constrained(&self) -> usize {
        self.ident_transform
            .as_ref()
            .map_or(self.centers.nrows(), Array2::ncols)
    }

    pub(crate) fn p_after_pad(&self) -> usize {
        self.p_constrained() + self.n_poly
    }

    pub(crate) fn p_out(&self) -> usize {
        self.full_ident_transform
            .as_ref()
            .map_or(self.p_after_pad(), Array2::ncols)
    }
}

impl JetLatentCoordLocalDesignJacobian {
    pub(crate) fn p_out(&self) -> usize {
        self.ident_transform
            .as_ref()
            .map_or(self.jet.shape()[1], Array2::ncols)
    }
}

/// The complete contract a per-row latent / novel-manifold coordinate type must
/// supply to participate in the REML design-derivative operator surface.
///
/// Onboarding a new coordinate type (the SAE / novel-manifold frontier) reduces
/// to implementing the small set of *required* methods below — the coordinate
/// geometry (`n_data`, `latent_dim`, `n_axes`) plus the single genuinely-new
/// payload `local_design_jacobian_row` (the local block ∂(design row)/∂(coord)).
/// The streaming operator surface consumed by `LatentCoordDerivativeOp` in
/// `src/solver/reml/mod.rs` — forward matvec, transpose matvec, and dense
/// materialization, together with the flat-axis → (row, axis) decode — is
/// inherited as *default* methods and never re-implemented per coordinate type.
///
/// This is the close condition for #767: a new coordinate type touches zero
/// operator-surface code; it provides only its local Jacobian and geometry.
pub(crate) trait LocalDesignJacobianProvider: Send + Sync + std::fmt::Debug {
    /// Number of data rows `n` the operator spans.
    fn n_data(&self) -> usize;

    /// Latent coordinate dimension `d` (perturbation axes per row).
    fn latent_dim(&self) -> usize;

    /// Number of flat hyper-axes `n · d` (one per (row, coordinate-axis) pair).
    fn n_axes(&self) -> usize;

    /// Number of output-basis columns in each local design-Jacobian row.
    fn p_out(&self) -> usize;

    /// The only per-coordinate payload: the projected local design-Jacobian row
    /// ∂(design row `row`)/∂(coordinate axis `axis`) in output-basis columns.
    fn local_design_jacobian_row(&self, row: usize, axis: usize)
    -> Result<Array1<f64>, BasisError>;

    /// Decode a flat hyper-axis into its `(row, coordinate axis)`. Row-major over
    /// `(row, axis)` with stride `latent_dim`; uniform across coordinate types.
    fn row_axis(&self, flat_axis: usize) -> (usize, usize) {
        let d = self.latent_dim();
        (flat_axis / d, flat_axis % d)
    }

    /// Forward matvec for one flat hyper-axis: place `J_row · u` at `row`.
    fn forward_mul_axis(
        &self,
        flat_axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            flat_axis < self.n_axes(),
            "latent-coordinate derivative flat axis out of bounds in forward_mul_axis: flat_axis={flat_axis}, n_axes={}",
            self.n_axes()
        );
        let (row, axis) = self.row_axis(flat_axis);
        let local_jacobian = self.local_design_jacobian_row(row, axis)?;
        assert_eq!(
            u.len(),
            local_jacobian.len(),
            "latent-coordinate derivative coefficient length mismatch in forward_mul_axis"
        );
        let value = local_jacobian.dot(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        out[row] = value;
        Ok(out)
    }

    /// Transpose matvec for one flat hyper-axis: scatter `v[row] · J_rowᵀ`.
    fn transpose_mul_axis(
        &self,
        flat_axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            flat_axis < self.n_axes(),
            "latent-coordinate derivative flat axis out of bounds in transpose_mul_axis: flat_axis={flat_axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            v.len(),
            self.n_data(),
            "latent-coordinate derivative row-adjoint length mismatch in transpose_mul_axis"
        );
        let (row, axis) = self.row_axis(flat_axis);
        let scale = v[row];
        Ok(self
            .local_design_jacobian_row(row, axis)?
            .mapv(|value| scale * value))
    }

    /// Dense `(n_data × p_out)` materialization of one flat hyper-axis: the local
    /// Jacobian row placed at `row`, all other rows zero.
    fn materialize_axis(&self, flat_axis: usize) -> Result<Array2<f64>, BasisError> {
        assert!(
            flat_axis < self.n_axes(),
            "latent-coordinate derivative flat axis out of bounds in materialize_axis: flat_axis={flat_axis}, n_axes={}",
            self.n_axes()
        );
        let (row, axis) = self.row_axis(flat_axis);
        let projected = self.local_design_jacobian_row(row, axis)?;
        let mut out = Array2::<f64>::zeros((self.n_data(), projected.len()));
        out.row_mut(row).assign(&projected);
        Ok(out)
    }
}

/// The rayon chunk size for parallel implicit matvec operations.
/// Each chunk processes this many data points before reducing.
pub(crate) const IMPLICIT_MATVEC_CHUNK_SIZE: usize = 1000;

/// Minimum data size to activate parallel iteration for implicit matvecs.
pub(crate) const IMPLICIT_MATVEC_PAR_THRESHOLD: usize = 10_000;

/// Number of lower-triangular center rows per tile when assembling dense
/// ThinPlate penalty ψ-derivative kernel blocks.
pub(crate) const THIN_PLATE_PENALTY_PSI_TILE_ROWS: usize = 32;

impl LatentCoordDesignDerivative {
    pub(crate) fn from_local_design_jacobian_provider(
        provider: Arc<dyn LocalDesignJacobianProvider>,
    ) -> Self {
        Self { provider }
    }

    pub(crate) fn new_matern(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        centers: Arc<Array2<f64>>,
        length_scale: f64,
        nu: MaternNu,
        include_intercept: bool,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != centers.ncols() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative Matérn dimension mismatch: latent d={} centers d={}",
                latent.latent_dim(),
                centers.ncols()
            );
        }
        Ok(Self::from_local_design_jacobian_provider(Arc::new(
            RadialLatentCoordLocalDesignJacobian {
                latent,
                centers,
                radial_kind: RadialScalarKind::Matern { length_scale, nu },
                ident_transform,
                full_ident_transform: None,
                n_poly: usize::from(include_intercept),
                polynomial_order: None,
            },
        )))
    }

    pub(crate) fn new_duchon(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        centers: Arc<Array2<f64>>,
        length_scale: Option<f64>,
        power: f64,
        nullspace_order: DuchonNullspaceOrder,
        full_ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != centers.ncols() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative Duchon dimension mismatch: latent d={} centers d={}",
                latent.latent_dim(),
                centers.ncols()
            );
        }
        let effective_order = duchon_effective_nullspace_order(centers.view(), nullspace_order);
        let p_order = duchon_p_from_nullspace_order(effective_order);
        let s_order = power.max(0.0).round() as usize;
        let radial_kind = if let Some(length_scale) = length_scale {
            RadialScalarKind::Duchon {
                length_scale,
                p_order,
                s_order,
                dim: centers.ncols(),
                coeffs: duchon_partial_fraction_coeffs(
                    p_order,
                    s_order,
                    1.0 / length_scale.max(1e-300),
                ),
            }
        } else {
            RadialScalarKind::PureDuchon {
                block_order: pure_duchon_block_order(p_order, power).max(1.0) as usize,
                p_order,
                s_order,
                dim: centers.ncols(),
            }
        };
        let mut workspace = BasisWorkspace::default();
        let ident_transform =
            kernel_constraint_nullspace(centers.view(), effective_order, &mut workspace.cache)?;
        let n_poly = polynomial_block_from_order(centers.view(), effective_order).ncols();
        Ok(Self::from_local_design_jacobian_provider(Arc::new(
            RadialLatentCoordLocalDesignJacobian {
                latent,
                centers,
                radial_kind,
                ident_transform: Some(ident_transform),
                full_ident_transform,
                n_poly,
                polynomial_order: Some(effective_order),
            },
        )))
    }

    pub(crate) fn new_sphere(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        centers: Arc<Array2<f64>>,
        penalty_order: usize,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != centers.ncols() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative sphere dimension mismatch: latent d={} centers d={}",
                latent.latent_dim(),
                centers.ncols()
            );
        }
        let raw_jet = sphere_first_derivative_nd(
            latent.as_matrix().view(),
            centers.view(),
            penalty_order,
            true,
        )?;
        let jet = latent.design_gradient_wrt_t_dispatch(
            crate::terms::latent::InputLocationDerivative::Jet(raw_jet.view()),
        )?;
        Self::from_jet(latent, jet, ident_transform)
    }

    pub(crate) fn new_periodic_bspline(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        data_range: (f64, f64),
        degree: usize,
        num_basis: usize,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        let raw_jet = periodic_bspline_first_derivative_nd(
            latent.as_matrix().view(),
            data_range,
            degree,
            num_basis,
        )?;
        let jet = latent.design_gradient_wrt_t_dispatch(
            crate::terms::latent::InputLocationDerivative::Jet(raw_jet.view()),
        )?;
        Self::from_jet(latent, jet, ident_transform)
    }

    pub(crate) fn new_tensor_bspline(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        knots_per_axis: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        let knot_views = knots_per_axis
            .iter()
            .map(|knots| knots.view())
            .collect::<Vec<_>>();
        let raw_jet =
            bspline_tensor_first_derivative(latent.as_matrix().view(), &knot_views, &degrees)?;
        let jet = latent.design_gradient_wrt_t_dispatch(
            crate::terms::latent::InputLocationDerivative::Jet(raw_jet.view()),
        )?;
        Self::from_jet(latent, jet, ident_transform)
    }

    pub(crate) fn new_pca(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        basis_matrix: Arc<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if latent.latent_dim() != basis_matrix.nrows() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative Pca dimension mismatch: latent d={} basis rows={}",
                latent.latent_dim(),
                basis_matrix.nrows()
            );
        }
        let mut jet =
            Array3::<f64>::zeros((latent.n_obs(), basis_matrix.ncols(), basis_matrix.nrows()));
        for row in 0..latent.n_obs() {
            for axis in 0..basis_matrix.nrows() {
                for col in 0..basis_matrix.ncols() {
                    jet[[row, col, axis]] = basis_matrix[[axis, col]];
                }
            }
        }
        Self::from_jet(latent, jet, None)
    }

    pub(crate) fn from_jet(
        latent: Arc<crate::terms::latent::LatentCoordValues>,
        jet: Array3<f64>,
        ident_transform: Option<Array2<f64>>,
    ) -> Result<Self, BasisError> {
        if jet.shape()[0] != latent.n_obs() || jet.shape()[2] != latent.latent_dim() {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative jet shape {:?} does not match latent shape ({}, {}, {})",
                jet.shape(),
                latent.n_obs(),
                jet.shape()[1],
                latent.latent_dim()
            );
        }
        if let Some(z) = ident_transform.as_ref()
            && z.nrows() != jet.shape()[1]
        {
            crate::bail_dim_basis!(
                "LatentCoordDesignDerivative identifiability transform has {} rows but derivative jet has {} basis columns",
                z.nrows(),
                jet.shape()[1]
            );
        }
        Ok(Self::from_local_design_jacobian_provider(Arc::new(
            JetLatentCoordLocalDesignJacobian {
                latent,
                jet: Arc::new(jet),
                ident_transform,
            },
        )))
    }

    pub(crate) fn n_data(&self) -> usize {
        self.provider.n_data()
    }

    pub(crate) fn latent_dim(&self) -> usize {
        self.provider.latent_dim()
    }

    pub(crate) fn n_axes(&self) -> usize {
        self.provider.n_axes()
    }

    pub(crate) fn p_out(&self) -> usize {
        self.provider.p_out()
    }
}

impl RadialLatentCoordLocalDesignJacobian {
    pub(crate) fn project_and_pad(
        &self,
        raw_knot: &Array1<f64>,
        raw_poly: &Array1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        let constrained = match &self.ident_transform {
            Some(z) => z.t().dot(raw_knot),
            None => raw_knot.clone(),
        };
        let mut padded = Array1::<f64>::zeros(constrained.len() + self.n_poly);
        padded
            .slice_mut(s![..constrained.len()])
            .assign(&constrained);
        if self.n_poly > 0 {
            padded.slice_mut(s![constrained.len()..]).assign(raw_poly);
        }
        Ok(match &self.full_ident_transform {
            Some(zf) => zf.t().dot(&padded),
            None => padded,
        })
    }

    pub(crate) fn kernel_axis_scalar(
        &self,
        row: usize,
        center: usize,
        axis: usize,
    ) -> Result<f64, BasisError> {
        let t_row = self.latent.row(row);
        let mut r2 = 0.0_f64;
        for a in 0..self.latent.latent_dim() {
            let delta = t_row[a] - self.centers[[center, a]];
            r2 += delta * delta;
        }
        let r = r2.sqrt();
        if r == 0.0 {
            // At a center collision the axis component s_axis = (t − c)_axis
            // is exactly zero. The product q · s_axis is therefore 0 for any
            // kernel whose q has a finite limit; for kernels where q diverges
            // the value is genuinely indeterminate (0 · ∞) and we must not
            // pretend it is zero. Defer to the kernel's classification.
            if self.radial_kind.is_smooth_at_collision() {
                return Ok(0.0);
            }
            return Err(BasisError::DegenerateAtCollision {
                kernel: "RadialScalarKind (design axis)",
                dim: self.latent.latent_dim(),
                m: 0.0,
                message: "radial scalar q = φ'/r has no finite limit at r = 0; \
                          the design row axis component is undefined",
            });
        }
        let (_, q, _) = self.radial_kind.eval_design_triplet(r)?;
        Ok(q * (t_row[axis] - self.centers[[center, axis]]))
    }

    pub(crate) fn polynomial_axis_values(&self, row: usize, axis: usize) -> Array1<f64> {
        let Some(order) = self.polynomial_order else {
            return Array1::<f64>::zeros(self.n_poly);
        };
        let max_degree = match order {
            DuchonNullspaceOrder::Zero => 0usize,
            DuchonNullspaceOrder::Linear => 1usize,
            DuchonNullspaceOrder::Degree(k) => k,
        };
        let t_row = self.latent.row(row);
        let exponents = monomial_exponents(self.latent.latent_dim(), max_degree);
        let mut out = Array1::<f64>::zeros(exponents.len());
        for (col, alpha) in exponents.iter().enumerate() {
            let a_axis = alpha[axis];
            if a_axis == 0 {
                continue;
            }
            let mut value = a_axis as f64;
            for a in 0..self.latent.latent_dim() {
                let exp_a = if a == axis { a_axis - 1 } else { alpha[a] };
                if exp_a != 0 {
                    value *= t_row[a].powi(exp_a as i32);
                }
            }
            out[col] = value;
        }
        out
    }
}

impl JetLatentCoordLocalDesignJacobian {
    pub(crate) fn project_jet(&self, raw_knot: &Array1<f64>) -> Result<Array1<f64>, BasisError> {
        Ok(match &self.ident_transform {
            Some(z) => z.t().dot(raw_knot),
            None => raw_knot.clone(),
        })
    }
}

impl LocalDesignJacobianProvider for LatentCoordDesignDerivative {
    fn n_data(&self) -> usize {
        self.provider.n_data()
    }

    fn latent_dim(&self) -> usize {
        self.provider.latent_dim()
    }

    fn n_axes(&self) -> usize {
        self.provider.n_axes()
    }

    fn p_out(&self) -> usize {
        self.provider.p_out()
    }

    fn local_design_jacobian_row(
        &self,
        row: usize,
        axis: usize,
    ) -> Result<Array1<f64>, BasisError> {
        self.provider.local_design_jacobian_row(row, axis)
    }
}

impl LocalDesignJacobianProvider for RadialLatentCoordLocalDesignJacobian {
    fn n_data(&self) -> usize {
        self.latent.n_obs()
    }

    fn latent_dim(&self) -> usize {
        self.latent.latent_dim()
    }

    fn n_axes(&self) -> usize {
        self.latent.len()
    }

    fn p_out(&self) -> usize {
        Self::p_out(self)
    }

    fn local_design_jacobian_row(
        &self,
        row: usize,
        axis: usize,
    ) -> Result<Array1<f64>, BasisError> {
        let mut raw_knot = Array1::<f64>::zeros(self.centers.nrows());
        for center in 0..self.centers.nrows() {
            raw_knot[center] = self.kernel_axis_scalar(row, center, axis)?;
        }
        let raw_poly = self.polynomial_axis_values(row, axis);
        self.project_and_pad(&raw_knot, &raw_poly)
    }
}

impl LocalDesignJacobianProvider for JetLatentCoordLocalDesignJacobian {
    fn n_data(&self) -> usize {
        self.latent.n_obs()
    }

    fn latent_dim(&self) -> usize {
        self.latent.latent_dim()
    }

    fn n_axes(&self) -> usize {
        self.latent.len()
    }

    fn p_out(&self) -> usize {
        Self::p_out(self)
    }

    fn local_design_jacobian_row(
        &self,
        row: usize,
        axis: usize,
    ) -> Result<Array1<f64>, BasisError> {
        let mut raw_knot = Array1::<f64>::zeros(self.jet.shape()[1]);
        for basis_col in 0..self.jet.shape()[1] {
            raw_knot[basis_col] = self.jet[[row, basis_col, axis]];
        }
        self.project_jet(&raw_knot)
    }
}

impl ImplicitDesignPsiDerivative {
    /// Construct from pre-computed radial jet scalars.
    ///
    /// # Arguments
    /// - `q_values`: (n * n_knots,) — φ'(r)/r for each (data, knot) pair.
    /// - `t_values`: (n * n_knots,) — (φ''(r) - q) / r² for each pair.
    /// - `axis_components`: (n * n_knots, D) — s_{d,ij} = exp(2η_d) · h_d² for each pair/axis.
    /// - `ident_transform`: optional (n_knots × p_constrained) constraint projection.
    /// - `full_ident_transform`: optional further projection after padding.
    /// - `n`, `n_knots`, `n_poly`, `n_axes`: dimensions.
    /// Construct from pre-computed (materialized) radial jet scalars.
    /// This is the original path for small-to-medium problems where
    /// O(n*k*(d+2)) storage is acceptable.
    pub fn new(
        phi_values: Array1<f64>,
        q_values: Array1<f64>,
        t_values: Array1<f64>,
        axis_components: Array2<f64>,
        ident_transform: Option<Array2<f64>>,
        full_ident_transform: Option<Array2<f64>>,
        n: usize,
        n_knots: usize,
        n_poly: usize,
        n_axes: usize,
    ) -> Self {
        assert_eq!(
            phi_values.len(),
            n * n_knots,
            "implicit psi derivative phi length mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            phi_values.len()
        );
        assert_eq!(
            q_values.len(),
            n * n_knots,
            "implicit psi derivative q length mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            q_values.len()
        );
        assert_eq!(
            t_values.len(),
            n * n_knots,
            "implicit psi derivative t length mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            t_values.len()
        );
        assert_eq!(
            axis_components.nrows(),
            n * n_knots,
            "implicit psi derivative axis-component row mismatch: expected n*n_knots={}*{}={}, got {}",
            n,
            n_knots,
            n * n_knots,
            axis_components.nrows()
        );
        assert_eq!(
            axis_components.ncols(),
            n_axes,
            "implicit psi derivative axis-component column mismatch: expected n_axes={n_axes}, got {}",
            axis_components.ncols()
        );
        Self {
            phi_values,
            axis_components,
            q_values,
            t_values,
            streaming: None,
            ident_transform,
            full_ident_transform,
            n,
            n_knots,
            n_poly,
            n_axes,
            psi_scale_share: 0.0,
            axis_combinations: None,
        }
    }

    pub(crate) fn with_psi_scale_share(mut self, psi_scale_share: f64) -> Self {
        self.psi_scale_share = psi_scale_share;
        self
    }

    /// Construct a streaming operator that recomputes (q, t, s_a) on the fly
    /// from raw data/centers/eta during each matvec. No O(n*k) arrays are stored.
    /// This is the large-scale path.
    pub(crate) fn new_streaming(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        eta: Vec<f64>,
        radial_kind: RadialScalarKind,
        ident_transform: Option<Array2<f64>>,
        full_ident_transform: Option<Array2<f64>>,
        n_poly: usize,
    ) -> Self {
        let n = data.nrows();
        let n_knots = centers.nrows();
        let n_axes = data.ncols();
        let psi_scale_share = radial_kind.raw_psi_isotropic_share();
        assert_eq!(eta.len(), n_axes);
        assert_eq!(
            centers.ncols(),
            n_axes,
            "streaming radial centers have {} columns but data/eta have {n_axes}",
            centers.ncols()
        );
        let metric_weights: Arc<[f64]> = Arc::from(centered_aniso_metric_weights(&eta));
        Self {
            // Empty arrays -- not used in streaming mode.
            phi_values: Array1::<f64>::zeros(0),
            axis_components: Array2::<f64>::zeros((0, 0)),
            q_values: Array1::<f64>::zeros(0),
            t_values: Array1::<f64>::zeros(0),
            streaming: Some(StreamingRadialState {
                data,
                centers,
                axis_mode: StreamingAxisMode::PerAxis { metric_weights },
                radial_kind,
                triplet_cache: Arc::new(std::sync::OnceLock::new()),
            }),
            ident_transform,
            full_ident_transform,
            n,
            n_knots,
            n_poly,
            n_axes,
            psi_scale_share,
            axis_combinations: None,
        }
    }

    /// Construct a streaming operator for a scalar ψ derivative. The operator
    /// exposes a single axis component equal to the full scaled squared
    /// distance r² under the fixed metric defined by `eta`.
    pub(crate) fn new_streaming_scalar(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        eta: Vec<f64>,
        radial_kind: RadialScalarKind,
        ident_transform: Option<Array2<f64>>,
        full_ident_transform: Option<Array2<f64>>,
        n_poly: usize,
    ) -> Self {
        let n = data.nrows();
        let n_knots = centers.nrows();
        let dim = data.ncols();
        assert_eq!(eta.len(), dim);
        assert_eq!(
            centers.ncols(),
            dim,
            "streaming scalar radial centers have {} columns but data/eta have {dim}",
            centers.ncols()
        );
        let metric_weights: Arc<[f64]> = Arc::from(centered_aniso_metric_weights(&eta));
        Self {
            phi_values: Array1::<f64>::zeros(0),
            axis_components: Array2::<f64>::zeros((0, 0)),
            q_values: Array1::<f64>::zeros(0),
            t_values: Array1::<f64>::zeros(0),
            streaming: Some(StreamingRadialState {
                data,
                centers,
                axis_mode: StreamingAxisMode::ScalarTotal { metric_weights },
                radial_kind,
                triplet_cache: Arc::new(std::sync::OnceLock::new()),
            }),
            ident_transform,
            full_ident_transform,
            n,
            n_knots,
            n_poly,
            n_axes: 1,
            psi_scale_share: 0.0,
            axis_combinations: None,
        }
    }

    /// Whether this operator is in streaming (recompute-on-the-fly) mode.
    #[inline]
    pub(crate) fn is_streaming(&self) -> bool {
        self.streaming.is_some()
    }

    /// Number of data points.
    pub fn n_data(&self) -> usize {
        self.n
    }

    /// Number of axes (D).
    pub fn n_axes(&self) -> usize {
        self.axis_combinations
            .as_ref()
            .map_or(self.n_axes, Vec::len)
    }

    pub(crate) fn is_duchon_family(&self) -> bool {
        self.streaming.as_ref().is_some_and(|state| {
            matches!(
                state.radial_kind,
                RadialScalarKind::Duchon { .. } | RadialScalarKind::PureDuchon { .. }
            )
        }) || self.psi_scale_share != 0.0
    }

    /// Whether this operator is wired up by a basis whose large-scale path
    /// is supposed to stay implicit, so a dense `(n × p)` materialization
    /// here is a regression rather than a normal compute path. Duchon-family
    /// terms qualify because they are streaming-only at any scale; ThinPlate
    /// qualifies because the new scalar-streaming routing relies on the
    /// implicit operator above the policy threshold and a sneaky
    /// `materialize_dense()` would silently re-introduce the n × p
    /// allocation we just removed. The flag is consulted by the
    /// materialize_first / materialize_second_diag / materialize_second_cross
    /// guards to fire `assert_no_dense_derivative_materialization` for these
    /// kinds whenever the resource policy says the materialization would
    /// exceed budget. Small-n problems still pass the assertion and get the
    /// dense fast path.
    pub(crate) fn enforces_dense_materialization_budget(&self) -> bool {
        if self
            .streaming
            .as_ref()
            .is_some_and(|state| state.radial_kind.enforces_dense_materialization_budget())
        {
            return true;
        }
        // The materialized-mode path keeps no `radial_kind` to inspect, but
        // a non-zero psi_scale_share is the unambiguous Duchon-family
        // signature there (Matern uses 0, ThinPlate uses 0). Materialized
        // ThinPlate / Matern terms are in the dense fast path and the
        // guard does not need to fire for them.
        self.psi_scale_share != 0.0
    }

    /// Output dimension: total basis columns in the final space.
    pub fn p_out(&self) -> usize {
        if let Some(ref zf) = self.full_ident_transform {
            zf.ncols()
        } else {
            self.p_after_pad()
        }
    }

    pub(crate) fn append_full_transform(
        mut self,
        transform: &Array2<f64>,
    ) -> Result<Self, BasisError> {
        if transform.nrows() != self.p_out() {
            crate::bail_dim_basis!(
                "implicit psi derivative transform has {} rows but operator has {} output columns",
                transform.nrows(),
                self.p_out()
            );
        }
        self.full_ident_transform = Some(match self.full_ident_transform.take() {
            Some(existing) => fast_ab(&existing, transform),
            None => transform.clone(),
        });
        Ok(self)
    }

    /// Dimension after kernel constraint + polynomial padding (before full ident).
    pub(crate) fn p_after_pad(&self) -> usize {
        let p_constrained = self.p_constrained();
        p_constrained + self.n_poly
    }

    /// Dimension after kernel constraint projection (before poly padding).
    pub(crate) fn p_constrained(&self) -> usize {
        match &self.ident_transform {
            Some(z) => z.ncols(),
            None => self.n_knots,
        }
    }

    /// Accumulate raw knot-space vector from weighted (data, knot) contributions.
    /// Returns a vector of length n_knots: Σ_i w_i · scalar_{ij} for each knot j.
    ///
    /// This is the core primitive: for each data point i, accumulate
    /// `v[i] * per_pair_scalar(i,j)` into knot j.
    pub(crate) fn accumulate_knot_vector<F>(&self, v: &ArrayView1<f64>, per_pair: F) -> Array1<f64>
    where
        F: Fn(usize) -> f64 + Send + Sync,
    {
        let n = self.n;
        let k = self.n_knots;

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            // Parallel path: chunk data points and reduce.
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let partial_sums: Vec<Array1<f64>> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut local = Array1::<f64>::zeros(k);
                    for i in start..end {
                        let vi = v[i];
                        if vi == 0.0 {
                            continue;
                        }
                        let base = i * k;
                        for j in 0..k {
                            local[j] += vi * per_pair(base + j);
                        }
                    }
                    local
                })
                .collect();
            let mut total = Array1::<f64>::zeros(k);
            for p in partial_sums {
                total += &p;
            }
            total
        } else {
            // Sequential path.
            let mut total = Array1::<f64>::zeros(k);
            for i in 0..n {
                let vi = v[i];
                if vi == 0.0 {
                    continue;
                }
                let base = i * k;
                for j in 0..k {
                    total[j] += vi * per_pair(base + j);
                }
            }
            total
        }
    }

    /// Streaming accumulate knot vector from on-the-fly radial scalars.
    pub(crate) fn streaming_accumulate_knot_vector<G>(
        &self,
        v: &ArrayView1<f64>,
        deriv_fn: G,
    ) -> Result<Array1<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync,
    {
        let st = self.streaming.as_ref().unwrap();
        let (n, k, dim) = (self.n, self.n_knots, self.n_axes);
        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let err_flag = std::sync::atomic::AtomicBool::new(false);
            let nc = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let ps: Vec<Array1<f64>> = (0..nc)
                .into_par_iter()
                .map(|ci| {
                    let s = ci * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let e = (s + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut loc = Array1::<f64>::zeros(k);
                    let mut sb = vec![0.0; dim];
                    for i in s..e {
                        let vi = v[i];
                        if vi == 0.0 {
                            continue;
                        }
                        for j in 0..k {
                            match st.compute_pair(i, j, &mut sb) {
                                Ok((phi, q, t)) => {
                                    loc[j] += vi * deriv_fn(phi, q, t, &sb);
                                }
                                Err(_) => {
                                    err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return loc;
                                }
                            }
                        }
                    }
                    loc
                })
                .collect();
            if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                crate::bail_invalid_basis!(
                    "radial scalar evaluation failed during streaming accumulate_knot_vector"
                        .into(),
                );
            }
            let mut tot = Array1::<f64>::zeros(k);
            for p in ps {
                tot += &p;
            }
            Ok(tot)
        } else {
            let mut tot = Array1::<f64>::zeros(k);
            let mut sb = vec![0.0; dim];
            for i in 0..n {
                let vi = v[i];
                if vi == 0.0 {
                    continue;
                }
                for j in 0..k {
                    let (phi, q, t) = st.compute_pair(i,j,&mut sb).map_err(|e| BasisError::InvalidInput(
                        format!("radial scalar evaluation failed during streaming accumulate_knot_vector: {e}"),
                    ))?;
                    tot[j] += vi * deriv_fn(phi, q, t, &sb);
                }
            }
            Ok(tot)
        }
    }
    /// Streaming forward multiply.
    pub(crate) fn streaming_forward_mul<G>(
        &self,
        u_knot: &Array1<f64>,
        deriv_fn: G,
    ) -> Result<Array1<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync,
    {
        let st = self.streaming.as_ref().unwrap();
        let (n, k, dim) = (self.n, self.n_knots, self.n_axes);
        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let err_flag = std::sync::atomic::AtomicBool::new(false);
            let nc = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let cr: Vec<(usize, Vec<f64>)> = (0..nc)
                .into_par_iter()
                .map(|ci| {
                    let s = ci * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let e = (s + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut loc = vec![0.0; e - s];
                    let mut sb = vec![0.0; dim];
                    for i in s..e {
                        let mut val = 0.0;
                        for j in 0..k {
                            match st.compute_pair(i, j, &mut sb) {
                                Ok((phi, q, t)) => {
                                    val += deriv_fn(phi, q, t, &sb) * u_knot[j];
                                }
                                Err(_) => {
                                    err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                    break;
                                }
                            }
                        }
                        loc[i - s] = val;
                    }
                    (s, loc)
                })
                .collect();
            if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
                crate::bail_invalid_basis!(
                    "radial scalar evaluation failed during streaming forward_mul".into(),
                );
            }
            let mut res = Array1::<f64>::zeros(n);
            for (s, vs) in cr {
                for (o, &v) in vs.iter().enumerate() {
                    res[s + o] = v;
                }
            }
            Ok(res)
        } else {
            let mut res = Array1::<f64>::zeros(n);
            let mut sb = vec![0.0; dim];
            for i in 0..n {
                let mut val = 0.0;
                for j in 0..k {
                    let (phi, q, t) = st.compute_pair(i, j, &mut sb).map_err(|e| {
                        BasisError::InvalidInput(format!(
                            "radial scalar evaluation failed during streaming forward_mul: {e}"
                        ))
                    })?;
                    val += deriv_fn(phi, q, t, &sb) * u_knot[j];
                }
                res[i] = val;
            }
            Ok(res)
        }
    }
    /// Streaming materialization: build (n x k) raw matrix then project.
    pub(crate) fn streaming_materialize<G>(&self, deriv_fn: G) -> Result<Array2<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64]) -> f64 + Send + Sync,
    {
        let st = self.streaming.as_ref().unwrap();
        let (n, k, dim) = (self.n, self.n_knots, self.n_axes);
        let mut raw = Array2::<f64>::zeros((n, k));
        let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
        let nc = n.div_ceil(cs);
        let err_flag = std::sync::atomic::AtomicBool::new(false);
        {
            let rp = SendPtr(raw.as_mut_ptr());
            let ef = &err_flag;
            (0..nc).into_par_iter().for_each(move |ci| {
                let s = ci * cs;
                let e = (s + cs).min(n);
                let mut sb = vec![0.0; dim];
                for i in s..e {
                    for j in 0..k {
                        match st.compute_pair(i, j, &mut sb) {
                            // SAFETY: chunk ci owns rows [s..e) of the raw n×k buffer,
                            // so offsets i*k+j for i ∈ [s,e), j ∈ [0,k) are pairwise
                            // disjoint across workers and stay within n*k = raw.len().
                            Ok((phi, q, t)) => unsafe {
                                *rp.add(i * k + j) = deriv_fn(phi, q, t, &sb);
                            },
                            Err(_) => {
                                ef.store(true, std::sync::atomic::Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                }
            });
        }
        if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
            crate::bail_invalid_basis!(
                "radial scalar evaluation failed during streaming materialize".into(),
            );
        }
        Ok(self.project_matrix(raw))
    }

    /// Project a raw knot-space vector through the identifiability transform
    /// and pad with zeros for polynomial columns.
    pub(crate) fn project_and_pad(&self, raw_knot_vec: &Array1<f64>) -> Array1<f64> {
        // Step 1: apply kernel constraint Z (if present).
        let constrained = match &self.ident_transform {
            Some(z) => z.t().dot(raw_knot_vec),
            None => raw_knot_vec.clone(),
        };

        // Step 2: pad with polynomial zeros.
        let p_padded = constrained.len() + self.n_poly;
        let mut padded = Array1::<f64>::zeros(p_padded);
        padded
            .slice_mut(s![..constrained.len()])
            .assign(&constrained);

        // Step 3: apply full identifiability transform (if present).
        match &self.full_ident_transform {
            Some(zf) => zf.t().dot(&padded),
            None => padded,
        }
    }

    /// Expand a coefficient vector from the final space back to raw knot space.
    /// This is the transpose path: p_out → (padded) → (constrained) → n_knots.
    pub(crate) fn unproject(&self, u: &ArrayView1<f64>) -> Array1<f64> {
        // Step 1: undo full identifiability transform.
        let after_full = match &self.full_ident_transform {
            Some(zf) => zf.dot(u),
            None => u.to_owned(),
        };

        // Step 2: extract smooth part (drop polynomial padding).
        let p_constrained = self.p_constrained();
        let smooth_part = after_full.slice(s![..p_constrained]);

        // Step 3: undo kernel constraint Z.
        match &self.ident_transform {
            Some(z) => z.dot(&smooth_part),
            None => smooth_part.to_owned(),
        }
    }

    /// Batched `unproject` for a (p_out × rank) coefficient matrix.
    /// Returns (n_knots × rank) via two BLAS3 matmuls — the same algebra as
    /// `unproject`, but amortized across all rank columns of `u`. Used by
    /// `forward_mul_matrix` so per-axis trace evaluations can be a single
    /// chunked GEMM rather than rank-many `forward_mul` calls.
    pub fn unproject_matrix(&self, u: &ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(u.nrows(), self.p_out());
        // Step 1: undo full identifiability transform → (p_after_pad, rank).
        let after_full = match &self.full_ident_transform {
            Some(zf) => fast_ab(zf, u),
            None => u.to_owned(),
        };
        // Step 2: drop polynomial padding rows → (p_constrained, rank).
        let p_constrained = self.p_constrained();
        let smooth_part = after_full.slice(s![..p_constrained, ..]);
        // Step 3: undo kernel constraint Z → (n_knots, rank).
        match &self.ident_transform {
            Some(z) => fast_ab(z, &smooth_part),
            None => smooth_part.to_owned(),
        }
    }

    /// Compute (∂X/∂ψ_d)^T v for a given axis d and vector v of length n.
    ///
    /// Returns a vector of length p_out (total basis dimension after all transforms).
    ///
    /// Formula in raw knot space:
    ///   [raw]_j = Σ_i v_i · q_{ij} · s_{d,ij}
    /// then project through Z and pad.
    ///
    /// Note: q = φ_r/r and s_d = exp(2ψ_d)·h_d² are UNNORMALIZED axis components.
    /// With this convention, q·s_d = (φ_r/r)·(exp(2ψ_d)·h_d²) = φ_r·(s_d/r),
    /// which equals the correct ∂φ/∂ψ_d = φ_r·∂r/∂ψ_d = φ_r·s_d/r.
    /// No r² correction is needed — that would be required only if s_d were
    /// the fractional quantity s_d/r².
    pub fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first transpose axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            v.len(),
            self.n,
            "implicit psi first transpose row-adjoint length mismatch"
        );
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                let raw = self.streaming_accumulate_knot_vector(v, |phi, q, _, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
                })?;
                return Ok(self.project_and_pad(&raw));
            }
            let c = self.psi_scale_share;
            let raw = self.accumulate_knot_vector(v, |idx| {
                let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                Self::transformed_first_kernel_value(
                    self.phi_values[idx],
                    self.q_values[idx],
                    s_combo,
                    combo_sum,
                    c,
                )
            });
            return Ok(self.project_and_pad(&raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            let raw =
                self.streaming_accumulate_knot_vector(v, |phi, q, _, sb| q * sb[axis] + c * phi)?;
            return Ok(self.project_and_pad(&raw));
        }
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let raw = self.accumulate_knot_vector(v, |idx| qv[idx] * af[[idx, axis]] + c * pv[idx]);
        Ok(self.project_and_pad(&raw))
    }

    /// Compute (∂X/∂ψ_d) u for a given axis d and vector u of length p_out.
    ///
    /// Returns a vector of length n.
    ///
    /// Formula: for each data point i,
    ///   result_i = Σ_j q_{ij} · s_{d,ij} · u_knot_j
    /// where u_knot = Z · u_smooth (unprojected back to knot space).
    pub fn forward_mul(&self, axis: usize, u: &ArrayView1<f64>) -> Result<Array1<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first forward axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            u.len(),
            self.p_out(),
            "implicit psi first forward coefficient length mismatch"
        );
        let u_knot = self.unproject(u);
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_forward_mul(&u_knot, |phi, q, _, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
                let mut result = Array1::<f64>::zeros(n);
                let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
                let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                        let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                        let mut local = vec![0.0; end - start];
                        for i in start..end {
                            let base = i * k;
                            let mut val = 0.0;
                            for j in 0..k {
                                let idx = base + j;
                                let s_combo =
                                    self.transformed_combo_axis_value_materialized(idx, combo);
                                val += Self::transformed_first_kernel_value(
                                    self.phi_values[idx],
                                    self.q_values[idx],
                                    s_combo,
                                    combo_sum,
                                    c,
                                ) * u_knot[j];
                            }
                            local[i - start] = val;
                        }
                        (start, local)
                    })
                    .collect();
                for (start, vals) in chunk_results {
                    for (offset, &v) in vals.iter().enumerate() {
                        result[start + offset] = v;
                    }
                }
                return Ok(result);
            }
            let mut result = Array1::<f64>::zeros(n);
            for i in 0..n {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    val += Self::transformed_first_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        s_combo,
                        combo_sum,
                        c,
                    ) * u_knot[j];
                }
                result[i] = val;
            }
            return Ok(result);
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_forward_mul(&u_knot, |phi, q, _, sb| q * sb[axis] + c * phi);
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let mut result = Array1::<f64>::zeros(n);
            // Parallel over chunks of data points.
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let mut local = vec![0.0; end - start];
                    for i in start..end {
                        let base = i * k;
                        let mut val = 0.0;
                        for j in 0..k {
                            val += (qv[base + j] * af[[base + j, axis]] + c * pv[base + j])
                                * u_knot[j];
                        }
                        local[i - start] = val;
                    }
                    (start, local)
                })
                .collect();
            for (start, vals) in chunk_results {
                for (offset, &v) in vals.iter().enumerate() {
                    result[start + offset] = v;
                }
            }
            Ok(result)
        } else {
            let mut result = Array1::<f64>::zeros(n);
            for i in 0..n {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    val += (qv[base + j] * af[[base + j, axis]] + c * pv[base + j]) * u_knot[j];
                }
                result[i] = val;
            }
            Ok(result)
        }
    }

    /// Compute (∂²X/∂ψ_d²)^T v — diagonal second derivative, same axis.
    ///
    /// Matrix-free variant of `materialize_second_diag`: avoids forming the
    /// full (n × p_out) matrix when only a single adjoint matvec is needed.
    pub fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal transpose axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            v.len(),
            self.n,
            "implicit psi second diagonal transpose row-adjoint length mismatch"
        );
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo, combo, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap_s, c,
                    )
                })?;
                return Ok(self.project_and_pad(&raw));
            }
            let c = self.psi_scale_share;
            let raw = self.accumulate_knot_vector(v, |idx| {
                let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                let overlap_s = self.transformed_combo_overlap_materialized(idx, combo, combo);
                Self::transformed_second_kernel_value(
                    self.phi_values[idx],
                    self.q_values[idx],
                    self.t_values[idx],
                    s_combo,
                    combo_sum,
                    s_combo,
                    combo_sum,
                    overlap_s,
                    c,
                )
            });
            return Ok(self.project_and_pad(&raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                let s = sb[axis];
                2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
            })?;
            return Ok(self.project_and_pad(&raw));
        }
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let raw = self.accumulate_knot_vector(v, |idx| {
            let s = af[[idx, axis]];
            2.0 * qv[idx] * s + tv[idx] * s * s + 2.0 * c * qv[idx] * s + c * c * pv[idx]
        });
        Ok(self.project_and_pad(&raw))
    }

    /// Compute (∂²X/∂ψ_d∂ψ_e)^T v — cross second derivative (d ≠ e).
    pub fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross transpose first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross transpose second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross transpose requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        assert_eq!(
            v.len(),
            self.n,
            "implicit psi second cross transpose row-adjoint length mismatch"
        );
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                    let s_d = combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let s_e = combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_d, sum_d, s_e, sum_e, overlap_s, c,
                    )
                })?;
                return Ok(self.project_and_pad(&raw));
            }
            let c = self.psi_scale_share;
            let raw = self.accumulate_knot_vector(v, |idx| {
                let s_d = self.transformed_combo_axis_value_materialized(idx, combo_d);
                let s_e = self.transformed_combo_axis_value_materialized(idx, combo_e);
                let overlap_s = self.transformed_combo_overlap_materialized(idx, combo_d, combo_e);
                Self::transformed_second_kernel_value(
                    self.phi_values[idx],
                    self.q_values[idx],
                    self.t_values[idx],
                    s_d,
                    sum_d,
                    s_e,
                    sum_e,
                    overlap_s,
                    c,
                )
            });
            return Ok(self.project_and_pad(&raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            let raw = self.streaming_accumulate_knot_vector(v, |phi, q, t, sb| {
                t * sb[axis_d] * sb[axis_e] + c * q * (sb[axis_d] + sb[axis_e]) + c * c * phi
            })?;
            return Ok(self.project_and_pad(&raw));
        }
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let raw = self.accumulate_knot_vector(v, |idx| {
            tv[idx] * af[[idx, axis_d]] * af[[idx, axis_e]]
                + c * qv[idx] * (af[[idx, axis_d]] + af[[idx, axis_e]])
                + c * c * pv[idx]
        });
        Ok(self.project_and_pad(&raw))
    }

    /// Compute (∂²X/∂ψ_d²) u — forward diagonal second derivative.
    pub fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal forward axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        assert_eq!(
            u.len(),
            self.p_out(),
            "implicit psi second diagonal forward coefficient length mismatch"
        );
        let u_knot = self.unproject(u);
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo, combo, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let compute_row = |i: usize| -> f64 {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    let overlap_s = self.transformed_combo_overlap_materialized(idx, combo, combo);
                    val += Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_combo,
                        combo_sum,
                        s_combo,
                        combo_sum,
                        overlap_s,
                        c,
                    ) * u_knot[j];
                }
                val
            };
            if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
                let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
                let mut result = Array1::<f64>::zeros(n);
                let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                        let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                        let local: Vec<f64> = (start..end).map(compute_row).collect();
                        (start, local)
                    })
                    .collect();
                for (start, vals) in chunk_results {
                    for (offset, &value) in vals.iter().enumerate() {
                        result[start + offset] = value;
                    }
                }
                return Ok(result);
            }
            return Ok(Array1::from_vec((0..n).map(compute_row).collect()));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                let s = sb[axis];
                2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let compute_row = |i: usize| -> f64 {
            let base = i * k;
            let mut val = 0.0;
            for j in 0..k {
                let s = af[[base + j, axis]];
                val += (2.0 * qv[base + j] * s
                    + tv[base + j] * s * s
                    + 2.0 * c * qv[base + j] * s
                    + c * c * pv[base + j])
                    * u_knot[j];
            }
            val
        };

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let mut result = Array1::<f64>::zeros(n);
            let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let local: Vec<f64> = (start..end).map(compute_row).collect();
                    (start, local)
                })
                .collect();
            for (start, vals) in chunk_results {
                for (offset, &value) in vals.iter().enumerate() {
                    result[start + offset] = value;
                }
            }
            Ok(result)
        } else {
            Ok(Array1::from_vec((0..n).map(compute_row).collect()))
        }
    }

    /// Compute (∂²X/∂ψ_d∂ψ_e) u — forward cross second derivative.
    pub fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<f64>,
    ) -> Result<Array1<f64>, BasisError> {
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross forward first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross forward second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross forward requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        assert_eq!(
            u.len(),
            self.p_out(),
            "implicit psi second cross forward coefficient length mismatch"
        );
        let u_knot = self.unproject(u);
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                    let s_d = combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let s_e = combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_d, sum_d, s_e, sum_e, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let compute_row = |i: usize| -> f64 {
                let base = i * k;
                let mut val = 0.0;
                for j in 0..k {
                    let idx = base + j;
                    let s_d = self.transformed_combo_axis_value_materialized(idx, combo_d);
                    let s_e = self.transformed_combo_axis_value_materialized(idx, combo_e);
                    let overlap_s =
                        self.transformed_combo_overlap_materialized(idx, combo_d, combo_e);
                    val += Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_d,
                        sum_d,
                        s_e,
                        sum_e,
                        overlap_s,
                        c,
                    ) * u_knot[j];
                }
                val
            };
            if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
                let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
                let mut result = Array1::<f64>::zeros(n);
                let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                    .into_par_iter()
                    .map(|chunk_idx| {
                        let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                        let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                        let local: Vec<f64> = (start..end).map(compute_row).collect();
                        (start, local)
                    })
                    .collect();
                for (start, vals) in chunk_results {
                    for (offset, &value) in vals.iter().enumerate() {
                        result[start + offset] = value;
                    }
                }
                return Ok(result);
            }
            return Ok(Array1::from_vec((0..n).map(compute_row).collect()));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_forward_mul(&u_knot, |phi, q, t, sb| {
                t * sb[axis_d] * sb[axis_e] + c * q * (sb[axis_d] + sb[axis_e]) + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let af = &self.axis_components;
        let pv = &self.phi_values;
        let qv = &self.q_values;
        let tv = &self.t_values;
        let compute_row = |i: usize| -> f64 {
            let base = i * k;
            let mut val = 0.0;
            for j in 0..k {
                val += (tv[base + j] * af[[base + j, axis_d]] * af[[base + j, axis_e]]
                    + c * qv[base + j] * (af[[base + j, axis_d]] + af[[base + j, axis_e]])
                    + c * c * pv[base + j])
                    * u_knot[j];
            }
            val
        };

        if n >= IMPLICIT_MATVEC_PAR_THRESHOLD {
            let n_chunks = n.div_ceil(IMPLICIT_MATVEC_CHUNK_SIZE);
            let mut result = Array1::<f64>::zeros(n);
            let chunk_results: Vec<(usize, Vec<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * IMPLICIT_MATVEC_CHUNK_SIZE;
                    let end = (start + IMPLICIT_MATVEC_CHUNK_SIZE).min(n);
                    let local: Vec<f64> = (start..end).map(compute_row).collect();
                    (start, local)
                })
                .collect();
            for (start, vals) in chunk_results {
                for (offset, &value) in vals.iter().enumerate() {
                    result[start + offset] = value;
                }
            }
            Ok(result)
        } else {
            Ok(Array1::from_vec((0..n).map(compute_row).collect()))
        }
    }

    /// Materialize the full (n × p_out) first-derivative matrix for axis d.
    ///
    /// Efficient O(n * k) construction: builds the raw (n × k) kernel derivative
    /// matrix directly, then projects through identifiability transforms.
    /// This is used when the dense matrix is needed temporarily (e.g., for
    /// HyperCoord construction) while avoiding simultaneous storage of all D axes.
    pub fn materialize_first(&self, axis: usize) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first materialization axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        if self.enforces_dense_materialization_budget() {
            assert_no_dense_derivative_materialization(self.n, self.p_out(), self.n_axes());
        }
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_materialize(|phi, q, _, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let mut raw = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    raw[[i, j]] = Self::transformed_first_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        s_combo,
                        combo_sum,
                        c,
                    );
                }
            }
            return Ok(self.project_matrix(raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_materialize(|phi, q, _, sb| q * sb[axis] + c * phi);
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let mut raw = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let base = i * k;
            for j in 0..k {
                raw[[i, j]] = self.q_values[base + j] * self.axis_components[[base + j, axis]]
                    + c * self.phi_values[base + j];
            }
        }
        Ok(self.project_matrix(raw))
    }

    /// Materialize the full (n × p_out) second diagonal derivative matrix for axis d.
    pub fn materialize_second_diag(&self, axis: usize) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal materialization axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        if self.enforces_dense_materialization_budget() {
            assert_no_dense_derivative_materialization(self.n, self.p_out(), self.n_axes());
        }
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_materialize(|phi, q, t, sb| {
                    let s_combo = combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo, combo, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let mut raw = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let s_combo = self.transformed_combo_axis_value_materialized(idx, combo);
                    let overlap_s = self.transformed_combo_overlap_materialized(idx, combo, combo);
                    raw[[i, j]] = Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_combo,
                        combo_sum,
                        s_combo,
                        combo_sum,
                        overlap_s,
                        c,
                    );
                }
            }
            return Ok(self.project_matrix(raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_materialize(|phi, q, t, sb| {
                let s = sb[axis];
                2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let mut raw = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let base = i * k;
            for j in 0..k {
                let s = self.axis_components[[base + j, axis]];
                raw[[i, j]] = 2.0 * self.q_values[base + j] * s
                    + self.t_values[base + j] * s * s
                    + 2.0 * c * self.q_values[base + j] * s
                    + c * c * self.phi_values[base + j];
            }
        }
        Ok(self.project_matrix(raw))
    }

    /// Materialize the full (n × p_out) cross second derivative matrix for axes (d, e).
    ///
    /// Dense materialization of the t · s_d · s_e cross coupling.
    pub fn materialize_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross materialization first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross materialization second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross materialization requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        if self.enforces_dense_materialization_budget() {
            assert_no_dense_derivative_materialization(self.n, self.p_out(), self.n_axes());
        }
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            if self.is_streaming() {
                let c = self.psi_scale_share;
                return self.streaming_materialize(|phi, q, t, sb| {
                    let s_d = combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let s_e = combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum();
                    let overlap_s = Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb);
                    Self::transformed_second_kernel_value(
                        phi, q, t, s_d, sum_d, s_e, sum_e, overlap_s, c,
                    )
                });
            }
            let n = self.n;
            let k = self.n_knots;
            let c = self.psi_scale_share;
            let mut raw = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let base = i * k;
                for j in 0..k {
                    let idx = base + j;
                    let s_d = self.transformed_combo_axis_value_materialized(idx, combo_d);
                    let s_e = self.transformed_combo_axis_value_materialized(idx, combo_e);
                    let overlap_s =
                        self.transformed_combo_overlap_materialized(idx, combo_d, combo_e);
                    raw[[i, j]] = Self::transformed_second_kernel_value(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        s_d,
                        sum_d,
                        s_e,
                        sum_e,
                        overlap_s,
                        c,
                    );
                }
            }
            return Ok(self.project_matrix(raw));
        }
        if self.is_streaming() {
            let c = self.psi_scale_share;
            return self.streaming_materialize(|phi, q, t, sb| {
                t * sb[axis_d] * sb[axis_e] + c * q * (sb[axis_d] + sb[axis_e]) + c * c * phi
            });
        }
        let n = self.n;
        let k = self.n_knots;
        let c = self.psi_scale_share;
        let mut raw = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let base = i * k;
            for j in 0..k {
                raw[[i, j]] = self.t_values[base + j]
                    * self.axis_components[[base + j, axis_d]]
                    * self.axis_components[[base + j, axis_e]]
                    + c * self.q_values[base + j]
                        * (self.axis_components[[base + j, axis_d]]
                            + self.axis_components[[base + j, axis_e]])
                    + c * c * self.phi_values[base + j];
            }
        }
        Ok(self.project_matrix(raw))
    }

    /// Project a raw (n × k) kernel-space matrix through all transforms to
    /// produce an (n × p_out) matrix: Z_kernel → pad poly → full ident.
    pub(crate) fn project_matrix(&self, raw: Array2<f64>) -> Array2<f64> {
        // Step 1: kernel constraint projection.
        let constrained = match &self.ident_transform {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        };

        // Step 2: polynomial padding.
        let padded = if self.n_poly > 0 {
            let cols = constrained.ncols();
            let mut out = Array2::<f64>::zeros((self.n, cols + self.n_poly));
            out.slice_mut(s![.., ..cols]).assign(&constrained);
            out
        } else {
            constrained
        };

        // Step 3: full identifiability transform.
        match &self.full_ident_transform {
            Some(zf) => fast_ab(&padded, zf),
            None => padded,
        }
    }

    pub(crate) fn project_matrix_rows(&self, raw: Array2<f64>) -> Array2<f64> {
        let nrows = raw.nrows();
        let constrained = match &self.ident_transform {
            Some(z) => fast_ab(&raw, z),
            None => raw,
        };
        let padded = if self.n_poly > 0 {
            let cols = constrained.ncols();
            let mut out = Array2::<f64>::zeros((nrows, cols + self.n_poly));
            out.slice_mut(s![.., ..cols]).assign(&constrained);
            out
        } else {
            constrained
        };
        match &self.full_ident_transform {
            Some(zf) => fast_ab(&padded, zf),
            None => padded,
        }
    }

    pub(crate) fn row_chunk_with_kernel<G>(
        &self,
        rows: std::ops::Range<usize>,
        deriv_fn: G,
    ) -> Result<Array2<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64], usize) -> f64,
    {
        let raw = self.row_chunk_with_kernel_raw(rows, deriv_fn)?;
        Ok(self.project_matrix_rows(raw))
    }

    /// Like `row_chunk_with_kernel` but returns the raw (chunk × n_knots)
    /// kernel scalars without the identifiability/padding projection. Used
    /// by `forward_mul_matrix`, which does the projection on the rank side
    /// instead (`unproject_matrix(F)`) so the (n × p_out) projected
    /// derivative is never materialized for large-scale row counts.
    pub(crate) fn row_chunk_with_kernel_raw<G>(
        &self,
        rows: std::ops::Range<usize>,
        deriv_fn: G,
    ) -> Result<Array2<f64>, BasisError>
    where
        G: Fn(f64, f64, f64, &[f64], usize) -> f64,
    {
        let mut raw = Array2::<f64>::zeros((rows.end - rows.start, self.n_knots));
        if let Some(st) = self.streaming.as_ref() {
            let mut sb = vec![0.0; self.n_axes];
            if let Some(cache) = st.ensure_triplet_cache() {
                for (local, i) in rows.enumerate() {
                    let base = i * self.n_knots;
                    for j in 0..self.n_knots {
                        let idx = base + j;
                        st.fill_s_buf(i, j, &mut sb);
                        raw[[local, j]] =
                            deriv_fn(cache.phi[idx], cache.q[idx], cache.t[idx], &sb, idx);
                    }
                }
            } else {
                for (local, i) in rows.enumerate() {
                    for j in 0..self.n_knots {
                        let (phi, q, t) = st.compute_pair(i, j, &mut sb)?;
                        raw[[local, j]] = deriv_fn(phi, q, t, &sb, i * self.n_knots + j);
                    }
                }
            }
        } else {
            for (local, i) in rows.enumerate() {
                let base = i * self.n_knots;
                for j in 0..self.n_knots {
                    let idx = base + j;
                    raw[[local, j]] = deriv_fn(
                        self.phi_values[idx],
                        self.q_values[idx],
                        self.t_values[idx],
                        &[],
                        idx,
                    );
                }
            }
        }
        Ok(raw)
    }

    pub fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first row chunk axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            return self.row_chunk_with_kernel(rows, |phi, q, _, sb, idx| {
                let s_combo = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo)
                } else {
                    combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
            });
        }
        self.row_chunk_with_kernel(rows, |phi, q, _, sb, idx| {
            let s = if sb.is_empty() {
                self.axis_components[[idx, axis]]
            } else {
                sb[axis]
            };
            q * s + c * phi
        })
    }

    /// Raw (chunk × n_knots) first-order kernel scalars for axis d, without
    /// the identifiability/padding projection. Pairs with `unproject_matrix`
    /// in `forward_mul_matrix`: the kernel scalars stay in raw knot space
    /// while the rank side (F) is unprojected to knot space, so the per-chunk
    /// GEMM is (chunk × n_knots) · (n_knots × rank) rather than (chunk × p_out)
    /// · (p_out × rank). Saves both flops and a (chunk × p_out) intermediate.
    pub fn row_chunk_first_raw(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi first raw row chunk axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            return self.row_chunk_with_kernel_raw(rows, |phi, q, _, sb, idx| {
                let s_combo = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo)
                } else {
                    combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                Self::transformed_first_kernel_value(phi, q, s_combo, combo_sum, c)
            });
        }
        self.row_chunk_with_kernel_raw(rows, |phi, q, _, sb, idx| {
            let s = if sb.is_empty() {
                self.axis_components[[idx, axis]]
            } else {
                sb[axis]
            };
            q * s + c * phi
        })
    }

    pub fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis < self.n_axes(),
            "implicit psi second diagonal row chunk axis out of bounds: axis={axis}, n_axes={}",
            self.n_axes()
        );
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo = self.transformed_axis_combination(axis);
            let combo_sum = Self::transformed_combo_sum(combo);
            return self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
                let s_combo = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo)
                } else {
                    combo
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                let overlap = if sb.is_empty() {
                    self.transformed_combo_overlap_materialized(idx, combo, combo)
                } else {
                    Self::transformed_combo_overlap_streaming(combo, combo, sb)
                };
                Self::transformed_second_kernel_value(
                    phi, q, t, s_combo, combo_sum, s_combo, combo_sum, overlap, c,
                )
            });
        }
        self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
            let s = if sb.is_empty() {
                self.axis_components[[idx, axis]]
            } else {
                sb[axis]
            };
            2.0 * q * s + t * s * s + 2.0 * c * q * s + c * c * phi
        })
    }

    pub fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, BasisError> {
        assert!(
            axis_d < self.n_axes(),
            "implicit psi second cross row chunk first axis out of bounds: axis_d={axis_d}, n_axes={}",
            self.n_axes()
        );
        assert!(
            axis_e < self.n_axes(),
            "implicit psi second cross row chunk second axis out of bounds: axis_e={axis_e}, n_axes={}",
            self.n_axes()
        );
        assert_ne!(
            axis_d, axis_e,
            "implicit psi second cross row chunk requires distinct axes: axis_d={axis_d}, axis_e={axis_e}"
        );
        let c = self.psi_scale_share;
        if self.axis_combinations.is_some() {
            let combo_d = self.transformed_axis_combination(axis_d);
            let combo_e = self.transformed_axis_combination(axis_e);
            let sum_d = Self::transformed_combo_sum(combo_d);
            let sum_e = Self::transformed_combo_sum(combo_e);
            return self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
                let s_d = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo_d)
                } else {
                    combo_d
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                let s_e = if sb.is_empty() {
                    self.transformed_combo_axis_value_materialized(idx, combo_e)
                } else {
                    combo_e
                        .iter()
                        .map(|(raw_axis, coeff)| coeff * sb[*raw_axis])
                        .sum()
                };
                let overlap = if sb.is_empty() {
                    self.transformed_combo_overlap_materialized(idx, combo_d, combo_e)
                } else {
                    Self::transformed_combo_overlap_streaming(combo_d, combo_e, sb)
                };
                Self::transformed_second_kernel_value(phi, q, t, s_d, sum_d, s_e, sum_e, overlap, c)
            });
        }
        self.row_chunk_with_kernel(rows, |phi, q, t, sb, idx| {
            let sd = if sb.is_empty() {
                self.axis_components[[idx, axis_d]]
            } else {
                sb[axis_d]
            };
            let se = if sb.is_empty() {
                self.axis_components[[idx, axis_e]]
            } else {
                sb[axis_e]
            };
            t * sd * se + c * q * (sd + se) + c * c * phi
        })
    }

    /// Single-row specialization of `row_chunk_first(axis, row..row+1)` that
    /// writes the length-`p_out` row directly into the caller-provided buffer.
    ///
    /// This is the row-local API used by `CustomFamilyPsiLinearMapRef::row_vector`
    /// for survival rowwise exact-Hessian paths, which previously applied a
    /// unit-vector `transpose_mul` trick (O(n·K) per row) to recover a single
    /// row. Avoids allocating a temporary (1 × p_out) matrix per row call.
    pub fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), BasisError> {
        assert!(
            row < self.n,
            "implicit psi row-vector request out of bounds: row={row}, n={}",
            self.n
        );
        assert_eq!(
            out.len(),
            self.p_out(),
            "implicit psi row-vector output length mismatch"
        );
        let chunk = self.row_chunk_first(axis, row..row + 1)?;
        out.assign(&chunk.row(0));
        Ok(())
    }

    pub(crate) fn transformed_axis_combination(&self, axis: usize) -> &[(usize, f64)] {
        self.axis_combinations
            .as_ref()
            .expect("transformed axis combinations")
            .get(axis)
            .map(Vec::as_slice)
            .expect("transformed axis index")
    }

    #[inline]
    pub(crate) fn transformed_combo_sum(combo: &[(usize, f64)]) -> f64 {
        combo.iter().map(|(_, coeff)| *coeff).sum()
    }

    #[inline]
    pub(crate) fn transformed_combo_axis_value_materialized(
        &self,
        idx: usize,
        combo: &[(usize, f64)],
    ) -> f64 {
        combo
            .iter()
            .map(|(raw_axis, coeff)| coeff * self.axis_components[[idx, *raw_axis]])
            .sum()
    }

    #[inline]
    pub(crate) fn transformed_combo_overlap_streaming(
        combo_left: &[(usize, f64)],
        combo_right: &[(usize, f64)],
        sb: &[f64],
    ) -> f64 {
        let mut overlap = 0.0;
        for &(left_axis, left_coeff) in combo_left {
            for &(right_axis, right_coeff) in combo_right {
                if left_axis == right_axis {
                    overlap += left_coeff * right_coeff * sb[left_axis];
                }
            }
        }
        overlap
    }

    #[inline]
    pub(crate) fn transformed_combo_overlap_materialized(
        &self,
        idx: usize,
        combo_left: &[(usize, f64)],
        combo_right: &[(usize, f64)],
    ) -> f64 {
        let mut overlap = 0.0;
        for &(left_axis, left_coeff) in combo_left {
            for &(right_axis, right_coeff) in combo_right {
                if left_axis == right_axis {
                    overlap += left_coeff * right_coeff * self.axis_components[[idx, left_axis]];
                }
            }
        }
        overlap
    }

    #[inline]
    pub(crate) fn transformed_first_kernel_value(
        phi: f64,
        q: f64,
        s_combo: f64,
        coeff_sum: f64,
        psi_scale_share: f64,
    ) -> f64 {
        q * s_combo + psi_scale_share * coeff_sum * phi
    }

    #[inline]
    pub(crate) fn transformed_second_kernel_value(
        phi: f64,
        q: f64,
        t: f64,
        s_left: f64,
        left_sum: f64,
        s_right: f64,
        right_sum: f64,
        overlap_s: f64,
        psi_scale_share: f64,
    ) -> f64 {
        t * s_left * s_right
            + 2.0 * q * overlap_s
            + psi_scale_share * q * (right_sum * s_left + left_sum * s_right)
            + psi_scale_share * psi_scale_share * left_sum * right_sum * phi
    }
}

pub(crate) fn build_aniso_design_psi_derivatives_shared(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    eta: &[f64],
    p_final: usize,
    ident_transform: Option<Array2<f64>>,
    full_ident_transform: Option<Array2<f64>>,
    n_poly: usize,
    radial_kind: RadialScalarKind,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    let dim = data.ncols();
    if eta.len() != dim {
        crate::bail_dim_basis!(
            "aniso design derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        );
    }

    let policy = crate::resource::ResourcePolicy::default_library();
    let force_operator = radial_kind.is_duchon_family();
    let dense_derivatives_exceed_budget =
        should_use_implicit_operators_with_policy(n, p_final, dim, &policy);
    let operator_only = force_operator || dense_derivatives_exceed_budget;
    let cache_radial_components = should_cache_implicit_radial_components(n, k, dim, &policy);
    // gam#1376 — the per-axis ψ derivatives this operator produces are ALREADY
    // the derivatives w.r.t. the κ-optimizer's raw coordinate, so NO cross-axis
    // centering projection is installed (for any family). The optimizer's per-
    // axis coordinate `psi_a` is decoded into both the global length scale
    // `ℓ = exp(−mean(psi))` and the centered contrast `eta_a = psi_a − mean(psi)`
    // simultaneously; in the kernel argument `x² = r²/ℓ² = Σ_a exp(2·psi_a)·h_a²`
    // the `mean(psi)` cancels, so the effective per-axis exponent is the raw
    // `psi_a` and `∂φ/∂psi_a = q·s_a` is the native per-axis ψ derivative. The
    // earlier `with_raw_eta_centering` projection annihilated the all-ones
    // (global-scale) direction and broke the analytic↔FD match (rel≈0.85). The
    // dense path (`build_matern_basis_log_kappa_aniso_derivatives`) is corrected
    // identically — it no longer centers downstream.

    // ── Streaming path: large scale ─────────────────────────────────────
    // When even the compact radial cache would exceed the operator-cache
    // budget, store only data/centers/eta/radial_kind and recompute
    // (q, t, s_a) chunkwise during each matvec. Otherwise the operator-only
    // path below caches phi/q/t/s_a without materializing dense derivative
    // matrices.
    if operator_only && !cache_radial_components {
        let op = ImplicitDesignPsiDerivative::new_streaming(
            shared_owned_data_matrix_from_view(data),
            shared_owned_centers_matrix_from_view(centers),
            eta.to_vec(),
            radial_kind,
            ident_transform,
            full_ident_transform,
            n_poly,
        );
        return Ok(AnisoBasisPsiDerivatives {
            design_first: Vec::new(),
            design_second_diag: Vec::new(),
            design_second_cross: Vec::new(),
            design_second_cross_pairs: Vec::new(),
            penalties_first: vec![Vec::new(); dim],
            penalties_second_diag: vec![Vec::new(); dim],
            penalties_cross_pairs: Vec::new(),
            penalties_cross_provider: None,
            implicit_operator: Some(op),
        });
    }

    // ── Materialized radial-cache path ────────────────────────────────────
    // Allocate O(n*k) arrays up front and fill with parallel chunks that
    // write directly into preallocated storage via raw pointers. No
    // intermediate Vec<(i, q_row, t_row, s_row)> collection.
    let nk = n.checked_mul(k).ok_or_else(|| {
        BasisError::InvalidInput("aniso radial cache has too many data-center pairs".to_string())
    })?;
    if nk.checked_mul(dim).is_none() {
        crate::bail_invalid_basis!("aniso radial cache axis component storage is too large");
    }
    let mut phi_values = Array1::<f64>::zeros(nk);
    let mut q_values = Array1::<f64>::zeros(nk);
    let mut t_values = Array1::<f64>::zeros(nk);
    let mut axis_components = Array2::<f64>::zeros((nk, dim));

    let psi_scale_share = radial_kind.raw_psi_isotropic_share();

    let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
    let nc = n.div_ceil(cs);
    // Capture the *first* underlying radial-evaluation error rather than a
    // bare boolean: at an extreme trial hyperparameter the anisotropic
    // distance `r` can push the Duchon/Matérn radial kernel out of its
    // evaluable range, and the caller (the spatial-κ optimizer) needs the
    // real cause to decide whether the trial point is merely infeasible
    // (retreat) versus a genuine invariant violation (abort). Swallowing it
    // as "radial scalar evaluation failed" hid both the cause and the
    // recoverability.
    let first_err: std::sync::Mutex<Option<BasisError>> = std::sync::Mutex::new(None);
    // For large sweeps, replace per-pair exact radial evaluation with a
    // certified 1-D Chebyshev profile built once from a distance-only
    // pre-pass over the radius range (see `radial_profile`): at the 16-D
    // power-9 hybrid Duchon configuration a single exact triplet costs tens
    // of microseconds across its partial-fraction blocks, and this n·k
    // sweep was the dominant per-κ-trial cost of large-scale fits (#979).
    // Out-of-range radii and uncertified builds fall back to the exact
    // evaluator per pair.
    let profile = if nk >= RADIAL_PROFILE_MIN_PAIRS {
        let mut r_lo = f64::INFINITY;
        let mut r_hi = 0.0_f64;
        let mut drb = vec![0.0; dim];
        let mut cb = vec![0.0; dim];
        for i in 0..n {
            for a in 0..dim {
                drb[a] = data[[i, a]];
            }
            for j in 0..k {
                for a in 0..dim {
                    cb[a] = centers[[j, a]];
                }
                let (r, _) = aniso_distance_and_components(&drb, &cb, eta);
                if r > 0.0 {
                    r_lo = r_lo.min(r);
                    r_hi = r_hi.max(r);
                }
            }
        }
        if r_lo.is_finite() && r_hi > r_lo {
            radial_profile::RadialProfile::build(&radial_kind, r_lo, r_hi)
        } else {
            None
        }
    } else {
        None
    };
    {
        let pp = SendPtr(phi_values.as_mut_ptr());
        let qp = SendPtr(q_values.as_mut_ptr());
        let tp = SendPtr(t_values.as_mut_ptr());
        let ap = SendPtr(axis_components.as_mut_ptr());
        let ferr = &first_err;
        let profile_ref = profile.as_ref();
        (0..nc).into_par_iter().for_each(move |ci| {
            let start = ci * cs;
            let end = start.saturating_add(cs).min(n);
            let mut drb = vec![0.0; dim];
            let mut cb = vec![0.0; dim];
            for i in start..end {
                for a in 0..dim {
                    drb[a] = data[[i, a]];
                }
                for j in 0..k {
                    for a in 0..dim {
                        cb[a] = centers[[j, a]];
                    }
                    let (r, sv) = aniso_distance_and_components(&drb, &cb, eta);
                    let triplet = match profile_ref {
                        Some(profile) => profile.eval_or_exact(&radial_kind, r),
                        None => radial_kind.eval_design_triplet(r),
                    };
                    let (phi, q, t) = match triplet {
                        Ok(p) => p,
                        Err(e) => {
                            let mut slot = ferr.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    };
                    let flat = i * k + j;
                    // SAFETY: each Rayon chunk owns a disjoint i-row range,
                    // so flat=i*k+j stays in 0..nk for phi/q/t and
                    // flat*dim+a stays in 0..nk*dim for axis_components.
                    unsafe {
                        *pp.add(flat) = phi;
                        *qp.add(flat) = q;
                        *tp.add(flat) = t;
                        for a in 0..dim {
                            *ap.add(flat * dim + a) = sv[a];
                        }
                    }
                }
            }
        });
    }
    if let Some(cause) = first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(BasisError::InvalidInput(format!(
            "radial scalar evaluation failed during aniso derivative construction \
             (eta={eta:?}): {cause}"
        )));
    }

    let op = ImplicitDesignPsiDerivative::new(
        phi_values,
        q_values,
        t_values,
        axis_components,
        ident_transform,
        full_ident_transform,
        n,
        k,
        n_poly,
        dim,
    )
    .with_psi_scale_share(psi_scale_share);

    // gam#1376 — the operator stays in the NATIVE per-axis ψ frame (no
    // `with_raw_eta_centering`): the κ-optimizer coordinate `psi_a` already maps
    // to the effective per-axis exponent `psi_a` of the kernel argument (the
    // `mean(psi)` it injects into the centered contrast is exactly cancelled by
    // the `ℓ = exp(−mean(psi))` it injects into the length scale), so the native
    // `∂φ/∂psi_a` produced by `materialize_first`/`materialize_second_*` (and by
    // the operator matvecs) is the correct raw-coordinate derivative. The
    // earlier centering broke the analytic↔FD match — see the comment above.

    if operator_only {
        return Ok(AnisoBasisPsiDerivatives {
            design_first: Vec::new(),
            design_second_diag: Vec::new(),
            design_second_cross: Vec::new(),
            design_second_cross_pairs: Vec::new(),
            penalties_first: vec![Vec::new(); dim],
            penalties_second_diag: vec![Vec::new(); dim],
            penalties_cross_pairs: Vec::new(),
            penalties_cross_provider: None,
            implicit_operator: Some(op),
        });
    }

    let design_first = (0..dim)
        .map(|a| op.materialize_first(a))
        .collect::<Result<Vec<_>, _>>()?;
    let design_second_diag = (0..dim)
        .map(|a| op.materialize_second_diag(a))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(AnisoBasisPsiDerivatives {
        design_first,
        design_second_diag,
        design_second_cross: Vec::new(),
        design_second_cross_pairs: Vec::new(),
        penalties_first: vec![Vec::new(); dim],
        penalties_second_diag: vec![Vec::new(); dim],
        penalties_cross_pairs: Vec::new(),
        penalties_cross_provider: None,
        implicit_operator: Some(op),
    })
}

#[derive(Debug, Clone)]
pub(crate) struct ScalarDesignPsiDerivatives {
    pub(crate) design_first: Array2<f64>,
    pub(crate) design_second_diag: Array2<f64>,
    pub(crate) implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

pub(crate) fn build_scalar_design_psi_derivatives_shared(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    fixed_eta: Option<&[f64]>,
    p_final: usize,
    ident_transform: Option<Array2<f64>>,
    full_ident_transform: Option<Array2<f64>>,
    n_poly: usize,
    radial_kind: RadialScalarKind,
    psi_scale_share: f64,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let n = data.nrows();
    let k = centers.nrows();
    let dim = data.ncols();
    if let Some(eta) = fixed_eta
        && eta.len() != dim
    {
        crate::bail_dim_basis!(
            "scalar design derivatives: eta.len()={} != data dimension {dim}",
            eta.len()
        );
    }

    let policy = crate::resource::ResourcePolicy::default_library();
    let force_operator = radial_kind.is_duchon_family();
    let dense_derivatives_exceed_budget =
        should_use_implicit_operators_with_policy(n, p_final, 1, &policy);
    let operator_only = force_operator || dense_derivatives_exceed_budget;
    let cache_radial_components = should_cache_implicit_radial_components(n, k, 1, &policy);
    if operator_only && !cache_radial_components {
        let metric_eta = fixed_eta
            .map(|eta| eta.to_vec())
            .unwrap_or_else(|| vec![0.0; dim]);
        let op = ImplicitDesignPsiDerivative::new_streaming_scalar(
            shared_owned_data_matrix_from_view(data),
            shared_owned_centers_matrix_from_view(centers),
            metric_eta,
            radial_kind,
            ident_transform,
            full_ident_transform,
            n_poly,
        )
        .with_psi_scale_share(psi_scale_share);
        return Ok(ScalarDesignPsiDerivatives {
            design_first: Array2::<f64>::zeros((0, 0)),
            design_second_diag: Array2::<f64>::zeros((0, 0)),
            implicit_operator: Some(op),
        });
    }

    let nk = n.checked_mul(k).ok_or_else(|| {
        BasisError::InvalidInput("scalar radial cache has too many data-center pairs".to_string())
    })?;
    let mut phi_values = Array1::<f64>::zeros(nk);
    let mut q_values = Array1::<f64>::zeros(nk);
    let mut t_values = Array1::<f64>::zeros(nk);
    let mut axis_components = Array2::<f64>::zeros((nk, 1));

    let cs = IMPLICIT_MATVEC_CHUNK_SIZE;
    let nc = n.div_ceil(cs);
    let first_err: std::sync::Mutex<Option<BasisError>> = std::sync::Mutex::new(None);
    // Same certified radial-profile amortization as the per-axis sweep
    // above: one distance-only pre-pass for the radius range, one profile
    // build, Clenshaw per pair, exact fallback out of range (#979).
    let pair_r = |i: usize, j: usize, drb: &mut [f64], cb: &mut [f64]| -> f64 {
        if let Some(eta) = fixed_eta {
            for a in 0..dim {
                drb[a] = data[[i, a]];
                cb[a] = centers[[j, a]];
            }
            aniso_distance_and_components(drb, cb, eta).0
        } else {
            stable_euclidean_norm((0..dim).map(|a| data[[i, a]] - centers[[j, a]]))
        }
    };
    let profile = if nk >= RADIAL_PROFILE_MIN_PAIRS {
        let mut r_lo = f64::INFINITY;
        let mut r_hi = 0.0_f64;
        let mut drb = vec![0.0; dim];
        let mut cb = vec![0.0; dim];
        for i in 0..n {
            for j in 0..k {
                let r = pair_r(i, j, &mut drb, &mut cb);
                if r > 0.0 {
                    r_lo = r_lo.min(r);
                    r_hi = r_hi.max(r);
                }
            }
        }
        if r_lo.is_finite() && r_hi > r_lo {
            radial_profile::RadialProfile::build(&radial_kind, r_lo, r_hi)
        } else {
            None
        }
    } else {
        None
    };
    {
        let pp = SendPtr(phi_values.as_mut_ptr());
        let qp = SendPtr(q_values.as_mut_ptr());
        let tp = SendPtr(t_values.as_mut_ptr());
        let ap = SendPtr(axis_components.as_mut_ptr());
        let ferr = &first_err;
        let profile_ref = profile.as_ref();
        (0..nc).into_par_iter().for_each(move |ci| {
            let start = ci * cs;
            let end = start.saturating_add(cs).min(n);
            let mut data_row_buf = vec![0.0; dim];
            let mut center_buf = vec![0.0; dim];
            for i in start..end {
                for a in 0..dim {
                    data_row_buf[a] = data[[i, a]];
                }
                for j in 0..k {
                    let (r, scalar_component) = if let Some(eta) = fixed_eta {
                        for a in 0..dim {
                            center_buf[a] = centers[[j, a]];
                        }
                        let (r, components) =
                            aniso_distance_and_components(&data_row_buf, &center_buf, eta);
                        (r, components.into_iter().sum::<f64>())
                    } else {
                        let r =
                            stable_euclidean_norm((0..dim).map(|a| data[[i, a]] - centers[[j, a]]));
                        (r, r * r)
                    };
                    let triplet = match profile_ref {
                        Some(profile) => profile.eval_or_exact(&radial_kind, r),
                        None => radial_kind.eval_design_triplet(r),
                    };
                    let (phi, q, t) = match triplet {
                        Ok(p) => p,
                        Err(e) => {
                            let mut slot = ferr.lock().unwrap_or_else(|p| p.into_inner());
                            if slot.is_none() {
                                *slot = Some(e);
                            }
                            return;
                        }
                    };
                    let flat = i * k + j;
                    // SAFETY: each Rayon chunk owns a disjoint i-row range
                    // of the nk-long phi/q/t/axis buffers, so flat=i*k+j is
                    // in-bounds for every write and never aliases another worker.
                    unsafe {
                        *pp.add(flat) = phi;
                        *qp.add(flat) = q;
                        *tp.add(flat) = t;
                        *ap.add(flat) = scalar_component;
                    }
                }
            }
        });
    }
    if let Some(cause) = first_err.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(BasisError::InvalidInput(format!(
            "radial scalar evaluation failed during scalar derivative construction: {cause}"
        )));
    }

    let op = ImplicitDesignPsiDerivative::new(
        phi_values,
        q_values,
        t_values,
        axis_components,
        ident_transform,
        full_ident_transform,
        n,
        k,
        n_poly,
        1,
    )
    .with_psi_scale_share(psi_scale_share);

    if operator_only {
        return Ok(ScalarDesignPsiDerivatives {
            design_first: Array2::<f64>::zeros((0, 0)),
            design_second_diag: Array2::<f64>::zeros((0, 0)),
            implicit_operator: Some(op),
        });
    }

    Ok(ScalarDesignPsiDerivatives {
        design_first: op.materialize_first(0)?,
        design_second_diag: op.materialize_second_diag(0)?,
        implicit_operator: Some(op),
    })
}
