/// Emits the `theta`-keyed memoization accessors shared verbatim by the
/// single-block and n-block exact-joint design caches. Both carry the same
/// `current_theta` / `last_cost` / `last_eval` fields, so the cost/eval
/// lookups and the `store_eval` writer are identical; this macro is the single
/// source so the two inherent impls cannot drift.
macro_rules! impl_exact_joint_theta_memo {
    () => {
        fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
            if self
                .current_theta
                .as_ref()
                .is_some_and(|cached| theta_values_match(cached, theta))
            {
                self.last_eval
                    .as_ref()
                    .map(|cached| cached.0)
                    .or(self.last_cost)
            } else {
                None
            }
        }

        fn memoized_eval(
            &self,
            theta: &Array1<f64>,
        ) -> Option<(f64, Array1<f64>, gam_problem::HessianValue)> {
            if self
                .current_theta
                .as_ref()
                .is_some_and(|cached| theta_values_match(cached, theta))
            {
                self.last_eval.clone()
            } else {
                None
            }
        }

        fn store_eval(&mut self, eval: (f64, Array1<f64>, gam_problem::HessianValue)) {
            self.last_cost = Some(eval.0);
            self.last_eval = Some(eval);
        }
    };
}

struct SingleBlockExactJointDesignCache<'d> {
    realizer: FrozenTermCollectionIncrementalRealizer<'d>,
    current_theta: Option<Array1<f64>>,
    // Memo key for `last_cost`/`last_eval`. Distinct from `current_theta` (which
    // tracks the θ the n×k design is REALIZED at): on the #1033 certified
    // Gaussian path `eval_full` evaluates a trial ψ WITHOUT re-realizing the
    // design (the tensor serves value+gradient n-free), so the eval θ and the
    // realized-design θ diverge. Keying the memo on a dedicated field keeps a
    // ψ-skip from ever mis-associating one ψ's cost/eval with another ψ's key.
    last_eval_theta: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_eval: Option<(f64, Array1<f64>, gam_problem::HessianValue)>,
    // #1033: ψ-invariant hyper-direction slab cache. The κ hyper_dirs (the n×k
    // ∂X/∂ψ design-derivative slabs + their k×k penalty derivatives) are a pure
    // function of (data, frozen spec, REALIZED column layout) — they do NOT
    // depend on the trial ψ once the design is fixed. On the certified Gaussian
    // n-free path `eval_full` evaluates trial ψ WITHOUT re-realizing the design,
    // so the realized layout (and hence the hyper_dirs) is identical across an
    // entire run of skip-path trials. Rebuilding them each trial re-runs the
    // basis ψ-derivative over all n rows + an O(n·k²) `fast_ab` rotation — the
    // last per-trial O(n) pass in the κ loop. Cache them keyed by the realizer
    // `design_revision`: a skip-path trial (revision unchanged) reuses the
    // build; a slow-path trial (revision advanced) rebuilds and re-keys.
    cached_hyper_dirs: Option<(u64, Vec<DirectionalHyperParam>)>,
    spatial_terms: Vec<usize>,
    rho_dim: usize,
    dims_per_term: Vec<usize>,
}

impl<'d> SingleBlockExactJointDesignCache<'d> {
    fn new_with_policy(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
        spatial_terms: Vec<usize>,
        rho_dim: usize,
        dims_per_term: Vec<usize>,
        policy: &gam_runtime::resource::ResourcePolicy,
    ) -> Result<Self, String> {
        Ok(Self {
            realizer: FrozenTermCollectionIncrementalRealizer::new_with_policy(
                data, spec, design, policy,
            )?,
            current_theta: None,
            last_eval_theta: None,
            last_cost: None,
            last_eval: None,
            cached_hyper_dirs: None,
            spatial_terms,
            rho_dim,
            dims_per_term,
        })
    }

    fn design_revision(&self) -> u64 {
        self.realizer.design_revision()
    }

    /// Build the κ hyper-directions for the CURRENT realized design, reusing the
    /// `cached_hyper_dirs` slab when the realizer revision has not advanced since
    /// the last build (#1033). The slab is ψ-invariant at a fixed realized
    /// layout, so a skip-path trial (which does not re-realize the design) gets a
    /// bit-identical clone instead of re-running the per-row basis ψ-derivative +
    /// O(n·k²) rotation. A revision change (slow-path re-realization) rebuilds and
    /// re-keys. The clone is an O(n·k) memcpy — far cheaper than the O(n·k²)
    /// rebuild, and the conditioning pass it feeds is itself skipped on the
    /// certified path (see `prepare_eval_state`'s fast path).
    fn hyper_dirs_for_current_design(
        &mut self,
        data: ArrayView2<'_, f64>,
        kind: SpatialHyperKind,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let revision = self.realizer.design_revision();
        if let Some((cached_rev, dirs)) = self.cached_hyper_dirs.as_ref()
            && *cached_rev == revision
        {
            return Ok(dirs.clone());
        }
        let t_build = std::time::Instant::now();
        let dirs = try_build_spatial_log_kappa_hyper_dirs(
            data,
            self.realizer.spec(),
            self.realizer.design(),
            &self.spatial_terms,
        )?
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "failed to build {} hyper_dirs at current {}",
                kind.adjective(),
                kind.coord_name(),
            ))
        })?;
        // Every accepted step realizes a new design revision, so this rebuild
        // runs once per step inside the gradient call, where nothing clocked it
        // (#2735).
        log::info!(
            "[STAGE] {} psi derivative rebuild (design revision {revision}, {} directions): {:.3}s",
            kind.label(),
            dirs.len(),
            t_build.elapsed().as_secs_f64(),
        );
        self.cached_hyper_dirs = Some((revision, dirs.clone()));
        Ok(dirs)
    }

    fn nfree_tensor_gradient_hyper_dirs(
        &mut self,
        theta: &Array1<f64>,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let psi = &theta.as_slice().ok_or_else(|| {
            EstimationError::InvalidInput(
                "nfree_tensor_gradient_hyper_dirs: theta is not contiguous".to_string(),
            )
        })?[self.rho_dim..];
        let (global_range, p_total, s_psi_components) = self
            .realizer
            .canonical_penalty_derivatives_at_psi(&self.spatial_terms, psi)
            .map_err(EstimationError::InvalidInput)?;
        let zero_x = gam_solve::estimate::reml::HyperDesignDerivative::zero(
            self.realizer.design().design.nrows(),
            p_total,
        );
        let components = s_psi_components
            .into_iter()
            .enumerate()
            .map(|(penalty_index, local)| {
                (
                    penalty_index,
                    gam_solve::estimate::reml::HyperPenaltyDerivative::from_embedded(
                        local,
                        global_range.clone(),
                        p_total,
                    ),
                )
            })
            .collect::<Vec<_>>();
        Ok(DirectionalHyperParam::new_compact(zero_x, components, None, None)?.not_penalty_like())
            .map(|dir| vec![dir])
    }

    /// Realize `theta`'s ψ tail on the cached design.
    ///
    /// Typed (gam#2760): see `apply_log_kappa` — a trial ψ the collection's model
    /// cannot be realized at is a domain wall, not a fatal error, and only the
    /// error VARIANT can carry that.
    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), EstimationError> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }
        let t_ensure = std::time::Instant::now();
        let log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            theta,
            self.rho_dim,
            self.dims_per_term.clone(),
        );
        self.realizer
            .apply_log_kappa(&log_kappa, &self.spatial_terms)?;
        // The ψ this realization is FOR. `ensure_theta` is memoized on θ, so
        // every line here is a distinct point, and the cost of a spatial fit is
        // the number of them: 518 realizations against 189 reported outer
        // evaluations on the 6-D k=100 fit. Which points those are — a line
        // search, a probe ladder, or one evaluation re-entered — cannot be read
        // from a line that prints only how long it took (#2735).
        log::info!(
            "[STAGE] ensure_theta (apply_log_kappa, {} terms) psi={:?}: {:.3}s",
            self.spatial_terms.len(),
            theta
                .iter()
                .skip(self.rho_dim)
                .map(|value| format!("{value:.6}"))
                .collect::<Vec<_>>(),
            t_ensure.elapsed().as_secs_f64(),
        );
        self.current_theta = Some(theta.clone());
        self.last_eval_theta = None;
        self.last_cost = None;
        self.last_eval = None;
        Ok(())
    }

    // Memo methods keyed on `last_eval_theta` (NOT `current_theta`): the #1033
    // certified Gaussian path evaluates a trial ψ without re-realizing the
    // design, so the eval θ and the realized-design θ can differ. Keying the
    // memo on the eval θ keeps a ψ-skip from mis-associating one ψ's result
    // with another ψ's key. The other exact-joint caches still use the shared
    // `impl_exact_joint_theta_memo!` macro (they always realize before eval).
    fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
        if self
            .last_eval_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval
                .as_ref()
                .map(|cached| cached.0)
                .or(self.last_cost)
        } else {
            None
        }
    }

    fn memoized_eval(
        &self,
        theta: &Array1<f64>,
    ) -> Option<(f64, Array1<f64>, gam_problem::HessianValue)> {
        if self
            .last_eval_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval.clone()
        } else {
            None
        }
    }

    /// Drop every memoized criterion value, keeping the realized design.
    ///
    /// The memo is keyed on θ alone, so it cannot tell two evaluations of two
    /// different MEASURES at the same θ apart. `begin_exact_polish` changes the
    /// measure — it retires the #1033b n-free surrogate — so the surrogate's
    /// value at the search checkpoint must not be served to the exact lane that
    /// follows (gam#2760).
    fn forget_eval_memo(&mut self) {
        self.last_eval_theta = None;
        self.last_cost = None;
        self.last_eval = None;
    }

    /// Record an eval result keyed to the θ it was computed at. Used in place of
    /// the macro's `store_eval` so the memo key reflects the EVAL θ even when the
    /// design was not re-realized at that θ (#1033 certified skip).
    fn store_eval_at(
        &mut self,
        theta: &Array1<f64>,
        eval: (f64, Array1<f64>, gam_problem::HessianValue),
    ) {
        self.last_eval_theta = Some(theta.clone());
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
    }

    /// Record a cost-only result keyed to the θ it was computed at, so
    /// `memoized_cost` keys on the EVAL θ (matching `store_eval_at`).
    fn store_cost_at(&mut self, theta: &Array1<f64>, cost: f64) {
        self.last_eval_theta = Some(theta.clone());
        self.last_cost = Some(cost);
        // A cost-only probe carries no gradient/Hessian, so drop any prior
        // full eval: `memoized_cost` prefers `last_eval.0`, and a stale
        // `last_eval` from a different θ must never answer for this θ.
        self.last_eval = None;
    }

    fn spec(&self) -> &TermCollectionSpec {
        self.realizer.spec()
    }

    fn design(&self) -> &TermCollectionDesign {
        self.realizer.design()
    }

    /// True when the single spatial term's frozen geometry admits an EXACT,
    /// n-free penalty re-key at a new length-scale (#1033). The κ-loop fast path
    /// gates its design-realization skip on this (replacing the old certified
    /// `psi_penalty_tensor_covers` gate): the skip leaves `reset_surface`
    /// un-run, so it is sound only when `S(ψ_new)` can be rebuilt n-free.
    fn supports_nfree_penalty_rekey(&self) -> bool {
        self.realizer
            .supports_nfree_penalty_rekey(&self.spatial_terms)
    }

    fn supports_nfree_gradient_only_routing(&self) -> bool {
        self.realizer
            .supports_nfree_gradient_only_routing(&self.spatial_terms)
    }

    /// Build the EXACT canonical penalty surface `S(ψ)` at the length-scale
    /// implied by `theta`'s ψ tail, entirely n-free (#1033). Maps ψ→length-scale
    /// with the IDENTICAL `spatial_term_psi_to_length_scale_and_aniso` the slow
    /// path uses, reuses the frozen basis geometry, and runs the SAME
    /// `canonicalize_penalty_specs` pipeline `reset_surface` runs — so the
    /// returned canonical list is the one the kept reference surface must be
    /// re-keyed with on the design-revision fast path. The caller (which holds
    /// `cache`) computes this and hands the owned result to the evaluator via
    /// `stage_fast_path_penalty`, avoiding a `&mut cache` borrow alias.
    fn canonical_penalties_at(
        &mut self,
        theta: &Array1<f64>,
        frozen_penalty_ranks: &[usize],
    ) -> Result<(Vec<gam_terms::construction::CanonicalPenalty>, Vec<usize>), String> {
        let psi = &theta
            .as_slice()
            .ok_or_else(|| "canonical_penalties_at: theta is not contiguous".to_string())?
            [self.rho_dim..];
        self.realizer
            .canonical_penalties_at_psi(&self.spatial_terms, psi, frozen_penalty_ranks)
    }
}

struct SingleBlockLatentCoordDesignCache {
    data: Array2<f64>,
    spec: TermCollectionSpec,
    design: TermCollectionDesign,
    current_theta: Option<Array1<f64>>,
    current_latent: Option<std::sync::Arc<gam_terms::latent::LatentCoordValues>>,
    current_hyper_dirs: Option<Vec<gam_solve::estimate::reml::DirectionalHyperParam>>,
    current_design_cache_id: Option<u64>,
    latent_design_cache: gam_solve::latent_cache::LatentDesignCache,
    last_cost: Option<f64>,
    last_eval: Option<(f64, Array1<f64>, gam_problem::HessianValue)>,
    term_index: gam_problem::types::SmoothTermIdx,
    feature_cols: Vec<usize>,
    rho_dim: usize,
    n_obs: usize,
    latent_dim: usize,
    id_mode: gam_terms::latent::LatentIdMode,
    manifold: gam_terms::latent::LatentManifold,
    retraction_registry: gam_solve::latent_cache::LatentRetractionRegistry,
    latent_id: u64,
    analytic_penalties: Option<std::sync::Arc<gam_terms::AnalyticPenaltyRegistry>>,
    analytic_rho_count: usize,
    design_revision: u64,
}

impl SingleBlockLatentCoordDesignCache {
    fn new(
        data: Array2<f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
        latent: &StandardLatentCoordConfig,
        rho_dim: usize,
    ) -> Result<Self, String> {
        if latent.term_index.get() >= spec.smooth_terms.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate term index {} out of bounds for {} smooth terms",
                latent.term_index,
                spec.smooth_terms.len()
            ))
            .into());
        }
        if latent.feature_cols.len() != latent.values.latent_dim() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate feature width mismatch: feature_cols={}, latent_dim={}",
                latent.feature_cols.len(),
                latent.values.latent_dim()
            ))
            .into());
        }
        if latent.values.n_obs() != data.nrows() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate row mismatch: latent n={}, data n={}",
                latent.values.n_obs(),
                data.nrows()
            ))
            .into());
        }
        let analytic_rho_count = latent
            .analytic_penalties
            .as_ref()
            .map_or(0, |registry| registry.total_rho_count());
        Ok(Self {
            data,
            spec,
            design,
            current_theta: None,
            current_latent: None,
            current_hyper_dirs: None,
            current_design_cache_id: None,
            latent_design_cache: gam_solve::latent_cache::LatentDesignCache::default(),
            last_cost: None,
            last_eval: None,
            term_index: latent.term_index,
            feature_cols: latent.feature_cols.clone(),
            rho_dim,
            n_obs: latent.values.n_obs(),
            latent_dim: latent.values.latent_dim(),
            id_mode: latent.values.id_mode().clone(),
            manifold: latent.values.manifold().clone(),
            retraction_registry: latent.values.retraction_registry().clone(),
            latent_id: latent.values.latent_id(),
            analytic_penalties: latent.analytic_penalties.clone(),
            analytic_rho_count,
            design_revision: 0,
        })
    }

    fn design_revision(&self) -> u64 {
        self.design_revision
    }

    fn design(&self) -> &TermCollectionDesign {
        &self.design
    }

    fn latent(&self) -> Result<std::sync::Arc<gam_terms::latent::LatentCoordValues>, String> {
        self.current_latent
            .as_ref()
            .cloned()
            .ok_or_else(|| "latent-coordinate cache has not been realized".to_string())
    }

    fn analytic_penalties(&self) -> Option<std::sync::Arc<gam_terms::AnalyticPenaltyRegistry>> {
        self.analytic_penalties.clone()
    }

    fn analytic_penalty_rho_count(&self) -> usize {
        self.analytic_rho_count
    }

    fn hyper_dirs(&self) -> Result<Vec<gam_solve::estimate::reml::DirectionalHyperParam>, String> {
        self.current_hyper_dirs
            .as_ref()
            .cloned()
            .ok_or_else(|| "latent-coordinate hyper_dirs cache has not been realized".to_string())
    }

    fn latent_basis_kind(&self) -> Result<gam_solve::latent_cache::LatentBasisKind, String> {
        let smooth_term = self
            .design
            .smooth
            .terms
            .get(self.term_index.get())
            .ok_or_else(|| {
                SmoothError::dimension_mismatch(format!(
                    "LatentCoord term index {} out of bounds for realized smooth design",
                    self.term_index
                ))
            })?;
        let termspec = self
            .spec
            .smooth_terms
            .get(self.term_index.get())
            .ok_or_else(|| {
                SmoothError::dimension_mismatch(format!(
                    "LatentCoord term index {} out of bounds for resolved smooth spec",
                    self.term_index
                ))
            })?;
        match (&termspec.basis, &smooth_term.metadata) {
            (
                SmoothBasisSpec::Matern { .. },
                BasisMetadata::Matern {
                    centers,
                    length_scale,
                    nu,
                    aniso_log_scales,
                    input_scale,
                    ..
                },
            ) => Ok(gam_solve::latent_cache::LatentBasisKind::Matern {
                centers: centers.clone(),
                // The metadata's frame pair travels together into the cache
                // key and into the radii it builds (#2643).
                input_scale: *input_scale,
                length_scale: *length_scale,
                nu: *nu,
                aniso_log_scales: aniso_log_scales
                    .clone()
                    .unwrap_or_else(|| vec![0.0; centers.ncols()]),
                chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                    self.n_obs,
                    centers.nrows(),
                ),
            }),
            (
                SmoothBasisSpec::Duchon { .. },
                BasisMetadata::Duchon {
                    centers,
                    length_scale,
                    power,
                    nullspace_order,
                    aniso_log_scales,
                    input_scale,
                    ..
                },
            ) => Ok(gam_solve::latent_cache::LatentBasisKind::Duchon {
                centers: centers.clone(),
                // See the Matérn arm (#2643).
                input_scale: *input_scale,
                length_scale: *length_scale,
                power: *power,
                nullspace_order: *nullspace_order,
                aniso_log_scales: aniso_log_scales
                    .clone()
                    .unwrap_or_else(|| vec![0.0; centers.ncols()]),
            }),
            (
                SmoothBasisSpec::Sphere { .. },
                BasisMetadata::Sphere {
                    centers,
                    penalty_order,
                    method,
                    ..
                },
            ) if matches!(*method, gam_terms::basis::SphereMethod::Wahba) => {
                Ok(gam_solve::latent_cache::LatentBasisKind::Sphere {
                    centers: centers.clone(),
                    penalty_order: *penalty_order,
                    chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                        self.n_obs,
                        centers.nrows(),
                    ),
                })
            }
            (
                SmoothBasisSpec::BSpline1D { spec, .. },
                BasisMetadata::BSpline1D {
                    knots,
                    periodic,
                    degree: meta_degree,
                    ..
                },
            ) => {
                // Issue #340: prefer the metadata-recorded effective degree
                // (which reflects fit-time auto-shrink) over the upstream
                // user-requested `spec.degree`.
                let effective_degree = meta_degree.unwrap_or(spec.degree);
                if let Some((domain_start, period, num_basis)) = periodic {
                    Ok(gam_solve::latent_cache::LatentBasisKind::PeriodicBspline {
                        domain_start: *domain_start,
                        period: *period,
                        degree: effective_degree,
                        num_basis: *num_basis,
                        chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                            self.n_obs, *num_basis,
                        ),
                    })
                } else {
                    let num_basis_est = knots.len().saturating_sub(effective_degree + 1);
                    Ok(gam_solve::latent_cache::LatentBasisKind::TensorBspline {
                        knots: vec![knots.clone()],
                        degrees: vec![effective_degree],
                        chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                            self.n_obs,
                            num_basis_est,
                        ),
                    })
                }
            }
            (
                SmoothBasisSpec::TensorBSpline { .. },
                BasisMetadata::TensorBSpline { knots, degrees, .. },
            ) => Ok(gam_solve::latent_cache::LatentBasisKind::TensorBspline {
                knots: knots.clone(),
                degrees: degrees.clone(),
                chunk_size: None,
            }),
            (
                SmoothBasisSpec::Pca { .. },
                BasisMetadata::Pca {
                    basis_matrix,
                    centered,
                    center_mean,
                    pca_basis_path,
                    chunk_size,
                    ..
                },
            ) => {
                let center_mean_fingerprint = if *centered && pca_basis_path.is_none() {
                    let mean = center_mean.as_ref().ok_or_else(|| {
                        SmoothError::invalid_config(
                            "latent-coordinate Pca cache key requires center_mean when centered",
                        )
                    })?;
                    Some(gam_solve::latent_cache::pca_center_mean_fingerprint(mean))
                } else {
                    None
                };
                Ok(gam_solve::latent_cache::LatentBasisKind::Pca {
                    basis_matrix: basis_matrix.clone(),
                    centered: *centered,
                    center_mean_fingerprint,
                    pca_basis_path: pca_basis_path.clone(),
                    chunk_size: *chunk_size,
                })
            }
            _ => Err(SmoothError::invalid_config(
                "latent-coordinate design cache could not key the realized latent smooth basis"
                    .to_string(),
            )
            .into()),
        }
    }

    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), String> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }
        let latent_flat_len = self.n_obs * self.latent_dim;
        let direct_hyper_count = latent_coord_direct_hyper_count(&self.id_mode, self.latent_dim);
        let expected =
            self.rho_dim + latent_flat_len + self.analytic_rho_count + direct_hyper_count;
        if theta.len() != expected {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate theta length mismatch: got {}, expected {} (rho_dim={}, n={}, d={}, analytic_rhos={}, direct_hypers={})",
                theta.len(),
                expected,
                self.rho_dim,
                self.n_obs,
                self.latent_dim,
                self.analytic_rho_count,
                direct_hyper_count
            ))
            .into());
        }
        let flat = theta
            .slice(s![self.rho_dim..self.rho_dim + latent_flat_len])
            .to_owned();
        let latent = std::sync::Arc::new(
            gam_terms::latent::LatentCoordValues::from_flat_with_manifold_and_retraction_and_id(
                flat,
                self.n_obs,
                self.latent_dim,
                self.id_mode.clone(),
                self.manifold.clone(),
                self.retraction_registry.clone(),
                self.latent_id,
            ),
        );
        let latent_values_changed = self
            .current_latent
            .as_ref()
            .map(|cached| !latent_values_match(cached.as_flat(), latent.as_flat()))
            .unwrap_or(true);
        if latent_values_changed {
            self.latent_design_cache.invalidate_all();
            self.current_design_cache_id = None;
            self.design_revision = self.design_revision.wrapping_add(1);
        }
        for n in 0..self.n_obs {
            for axis in 0..self.latent_dim {
                let col = self.feature_cols[axis];
                self.data[[n, col]] = latent.as_flat()[n * self.latent_dim + axis];
            }
        }

        let basis_kind = self.latent_basis_kind()?;
        let rebuilt_width = self.design.design.ncols();
        let spec = self.spec.clone();
        let term_index = self.term_index;
        let analytic_rho_count = self.analytic_rho_count;
        let data = self.data.view();
        let design_context_digest = gam_solve::latent_cache::latent_design_context_cache_digest(
            data,
            &spec,
            term_index,
            analytic_rho_count,
            &self.feature_cols,
        )
        .map_err(|e| e.to_string())?;
        let lookup = self
            .latent_design_cache
            .lookup_or_compute(latent.clone(), basis_kind, design_context_digest, || {
                let rebuilt = build_term_collection_design(data, &spec).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "failed to rebuild latent-coordinate design: {e}"
                    ))
                })?;
                if rebuilt.design.ncols() != rebuilt_width {
                    crate::bail_invalid_estim!(
                        "latent-coordinate design topology changed: rebuilt p={}, cached p={}",
                        rebuilt.design.ncols(),
                        rebuilt_width
                    );
                }
                let hyper_dirs = try_build_latent_coord_hyper_dirs(
                    latent.clone(),
                    &spec,
                    &rebuilt,
                    &[term_index],
                    analytic_rho_count,
                )?
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "failed to build latent-coordinate hyper_dirs".to_string(),
                    )
                })?;
                Ok(gam_solve::latent_cache::ComputedLatentDesign {
                    design: rebuilt,
                    hyper_dirs,
                })
            })
            .map_err(|e| e.to_string())?;
        if lookup.cached.design.design.ncols() != self.design.design.ncols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate design topology changed: rebuilt p={}, cached p={}",
                lookup.cached.design.design.ncols(),
                self.design.design.ncols()
            ))
            .into());
        }
        self.design = lookup.cached.design.clone();
        self.current_hyper_dirs = Some(lookup.cached.hyper_dirs.clone());
        self.current_latent = Some(latent);
        self.current_theta = Some(theta.clone());
        self.last_cost = None;
        self.last_eval = None;
        if !latent_values_changed && self.current_design_cache_id != Some(lookup.entry_id) {
            self.design_revision = self.design_revision.wrapping_add(1);
        }
        self.current_design_cache_id = Some(lookup.entry_id);
        Ok(())
    }

    fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval
                .as_ref()
                .map(|cached| cached.0)
                .or(self.last_cost)
        } else {
            None
        }
    }

    fn memoized_eval(
        &self,
        theta: &Array1<f64>,
    ) -> Option<(f64, Array1<f64>, gam_problem::HessianValue)> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval.clone()
        } else {
            None
        }
    }

    fn store_eval(&mut self, eval: (f64, Array1<f64>, gam_problem::HessianValue)) {
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
    }

    fn store_cost(&mut self, cost: f64) {
        self.last_cost = Some(cost);
    }

    fn reset(&mut self) {
        self.current_theta = None;
        self.current_latent = None;
        self.current_hyper_dirs = None;
        self.current_design_cache_id = None;
        self.latent_design_cache.invalidate();
        self.last_cost = None;
        self.last_eval = None;
    }
}

