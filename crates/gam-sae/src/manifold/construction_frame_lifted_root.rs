// Included from `construction_quasi_laplace.rs` (#3434): the evidence root's
// acceptance in the lifted chart `ξ = (vec δC, vec W)` of the learned frames.
//
// The fixed-frame inner solve moves `(t, C)` with every frame `U` held, and `U`
// moves only at the polar refresh, which re-expresses the decoder in the span it
// already occupies. Its root is therefore stationary in `(t, C)` but not in the
// frame orientations: at that root `Tᵀ·∇_B L` is non-zero along `W`, and the
// lifted operator `A_ξ = [[A_tt, A_tB·T], [Tᵀ·A_Bt, Tᵀ·A_BB·T + E]]` the
// frame-integrated information prices need not be positive there. The root is
// accepted in the lifted chart before it is priced: its pencil `(A_ξ, Φ_ξ)` is
// classified with the same band edges, concave-clamp basin and decrement
// certificate the fixed-frame root reads, and a root that fails is descended
// along the chart curve `B(α) = (C + α·δC)(U + α·U⊥W)ᵀ`, whose slope and
// curvature at `α = 0` are exactly `g_ξᵀd̂` and `d̂ᵀA_ξd̂`.

/// The refusal label of a lifted-chart saddle no committed step descends.
const FRAME_LIFTED_ROOT_BLOCK: &str = "frame-integrated root";

/// What the lifted-chart acceptance did to an evidence root (#3434).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameLiftedRoot {
    /// The root is certified in the lifted chart, or the chart is not the one the
    /// frame-integrated information is priced on at this state.
    Settled,
    /// A lifted step committed a material decrease; the root must converge again.
    Moved,
    /// A resolved negative lifted direction the concave clamp does not explain, which
    /// no committed step descends.
    Saddle,
}

/// The lifted-chart stationarity certificate of an evidence root (#3434).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameLiftedCertificate {
    /// The smallest pencil eigenvalue of `(A_ξ, Φ_ξ)`.
    pub(crate) min_curvature: f64,
    /// The resolved negative lifted directions, below their own band edges.
    pub(crate) resolved_negative: Option<ResolvedNegativeCurvature>,
    /// Whether the concave clamp explains every resolved negative direction (#2333).
    pub(crate) clamp_explained: bool,
    /// `λ² = Σ_{μᵢ > τᵢ} cᵢ²/μᵢ`, `c = Wᵀg_ξ`.
    pub(crate) lambda_sq: f64,
    /// `½λ²/(|f| + 1)`.
    pub(crate) relative: f64,
    /// No unexplained resolved negative direction, and the decrement certifies.
    pub(crate) certified: bool,
}

/// The certificate with what a descent of the lifted chart reads.
struct FrameLiftedVerdict {
    certificate: FrameLiftedCertificate,
    /// `c = Wᵀg_ξ` in the pencil's `Φ_ξ`-orthonormal eigenbasis.
    coefficients: Array1<f64>,
    /// The refused basin directions with their basin curvature `κ`.
    refused: Vec<(Array1<f64>, f64)>,
    /// The gate-frozen penalized objective at the root.
    objective: f64,
}

/// One atom's velocity along the lifted chart.
enum LearnedFrameChartMotion {
    /// `δC` (`M×r`) and the frame's normal velocity `N = U⊥·W` (`p×r`).
    Framed {
        delta_coordinates: Array2<f64>,
        normal_velocity: Array2<f64>,
    },
    /// `δB` (`M×p`).
    Unframed { delta_decoder: Array2<f64> },
}

/// A unit-speed velocity of the lifted chart: the iterate's first-order motion
/// `(δt, T·δξ)` has unit norm in the coordinates [`SaeManifoldTerm::inner_iterate_scale`]
/// measures.
struct LearnedFrameChartStep {
    delta_t: Array1<f64>,
    atoms: Vec<LearnedFrameChartMotion>,
}

impl LearnedFrameTangentMap {
    /// The unit-speed chart velocity along a joint `(t, ξ)` direction, with the norm
    /// `‖(d_t, T·d_ξ)‖` it was divided by, or `None` for a direction that moves nothing.
    fn chart_step(
        &self,
        term: &SaeManifoldTerm,
        direction: ArrayView1<'_, f64>,
        total_t: usize,
    ) -> Result<Option<(LearnedFrameChartStep, f64)>, String> {
        let xi_dim = self.lift.ncols();
        if direction.len() != total_t + xi_dim || self.ranges.len() != term.k_atoms() {
            return Err(format!(
                "lifted chart step: direction length {} for {total_t} coordinates and {xi_dim} \
                 tangent coordinates over {} atom ranges ({} atoms)",
                direction.len(),
                self.ranges.len(),
                term.k_atoms(),
            ));
        }
        let d_t = direction.slice(s![..total_t]);
        let d_beta = self.lift.dot(&direction.slice(s![total_t..]));
        let norm = (d_t.dot(&d_t) + d_beta.dot(&d_beta)).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Ok(None);
        }
        let unit = direction.mapv(|value| value / norm);
        let p = term.output_dim();
        let mut atoms = Vec::with_capacity(term.k_atoms());
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let range = self.ranges[atom_idx].clone();
            let start = total_t + range.start;
            let m = atom.basis_size();
            atoms.push(match atom.decoder_frame.as_ref() {
                Some(frame) => {
                    let u = frame.frame();
                    let r = u.ncols();
                    if range.len() != m * r + r * (p - r) {
                        return Err(format!(
                            "lifted chart step: atom {atom_idx} has {} tangent coordinates for a \
                             {m}×{r} coordinate block on a rank-{r} frame in ℝ^{p}",
                            range.len()
                        ));
                    }
                    let delta_coordinates =
                        Array2::from_shape_fn((m, r), |(b, j)| unit[start + b * r + j]);
                    let w = Array2::from_shape_fn((p - r, r), |(i, j)| {
                        unit[start + m * r + i * r + j]
                    });
                    LearnedFrameChartMotion::Framed {
                        delta_coordinates,
                        normal_velocity: orthonormal_frame_complement(u)?.dot(&w),
                    }
                }
                None => {
                    if range.len() != m * p {
                        return Err(format!(
                            "lifted chart step: unframed atom {atom_idx} has {} coordinates for \
                             an {m}×{p} decoder",
                            range.len()
                        ));
                    }
                    LearnedFrameChartMotion::Unframed {
                        delta_decoder: Array2::from_shape_fn((m, p), |(b, c)| {
                            unit[start + b * p + c]
                        }),
                    }
                }
            });
        }
        Ok(Some((
            LearnedFrameChartStep {
                delta_t: unit.slice(s![..total_t]).to_owned(),
                atoms,
            },
            norm,
        )))
    }
}

impl SaeManifoldTerm {
    /// Whether the frame-integrated information is priced on the dense lifted
    /// chart at this state, the route whose pencil the acceptance classifies.
    fn frame_lifted_chart_admitted(&self, total_t: usize) -> Result<bool, String> {
        Ok(matches!(
            self.fitted_response_frame_conditioning()?,
            SaeFrameConditioning::MarginalOverLearnedFrames
        ) && matches!(
            self.frame_integrated_route(total_t)?,
            FrameIntegratedRoute::Dense
        ))
    }

    /// #3434 — the lifted-chart certificate of the root `cache` was factored at, or
    /// `None` where the frame-integrated information is not priced on the dense
    /// lifted chart.
    pub(crate) fn frame_lifted_certificate(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        cache: &ArrowFactorCache,
    ) -> Result<Option<FrameLiftedCertificate>, String> {
        if !self.frame_lifted_chart_admitted(cache.delta_t_len())? {
            return Ok(None);
        }
        let information = self.frame_marginal_information(rho, target, registry)?;
        Ok(Some(
            self.frame_lifted_verdict(target, rho, registry, &information)?
                .certificate,
        ))
    }

    /// The pencil `(A_ξ, Φ_ξ)` classified as the fixed-frame refined root is
    /// ([`Self::exact_root_classification`]): resolved negative directions go to the
    /// concave-clamp basin, and otherwise the exact decrement decides through
    /// [`Self::inner_decrement_certifies`].
    fn frame_lifted_verdict(
        &self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        information: &FrameMarginalInformation,
    ) -> Result<FrameLiftedVerdict, String> {
        let joint = &information.joint;
        let dim = joint.eigenvalues.len();
        if information.gradient.len() != dim || joint.eigenvectors.dim() != (dim, dim) {
            return Err(format!(
                "frame-lifted root: pencil spectrum {dim} and eigenvectors {:?} against lifted \
                 gradient {}",
                joint.eigenvectors.dim(),
                information.gradient.len()
            ));
        }
        let min_curvature = joint
            .eigenvalues
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let resolved_negative = ResolvedNegativeCurvature::of_directions(
            (0..dim).map(|index| (joint.eigenvalues[index], joint.rank_floor(index))),
        );
        let mut refused = Vec::new();
        let clamp_explained = match resolved_negative {
            None => false,
            Some(_) => {
                let e_diag = information
                    .unframed
                    .materialize_ard_concave_clamp_diagonal(rho, &information.cache)?;
                match Self::classify_exact_hessian_basin(
                    joint,
                    &e_diag,
                    information.gap_border.as_ref(),
                    information.total_t,
                    FRAME_LIFTED_ROOT_BLOCK,
                    Some(&mut refused),
                ) {
                    Ok(_) => true,
                    Err(_) if !refused.is_empty() => false,
                    Err(error) => return Err(error.to_string()),
                }
            }
        };
        let coefficients = joint.eigenvectors.t().dot(&information.gradient);
        let pencil_floor = sae_exact_a_pencil_floor();
        let lambda_sq = (0..dim)
            .filter(|&index| joint.eigenvalues[index] > pencil_floor.max(joint.resolution[index]))
            .map(|index| coefficients[index] * coefficients[index] / joint.eigenvalues[index])
            .sum::<f64>();
        let objective = self.penalized_objective_total(target, rho, registry, 1.0)?;
        let relative = 0.5 * lambda_sq / (objective.abs() + 1.0);
        let certified = (resolved_negative.is_none() || clamp_explained)
            && Self::inner_decrement_certifies(relative);
        Ok(FrameLiftedVerdict {
            certificate: FrameLiftedCertificate {
                min_curvature,
                resolved_negative,
                clamp_explained,
                lambda_sq,
                relative,
                certified,
            },
            coefficients,
            refused,
            objective,
        })
    }

    /// The spectral step of the lifted pencil, `d = W·a`, with its curvature
    /// `dᵀA_ξd = Σ aᵢ²μᵢ`. Where the pencil holds refused saddle directions the step is
    /// saddle-free, `aᵢ = −cᵢ/|μᵢ|` over the directions clear of the band; otherwise it
    /// is the Newton step `aᵢ = −cᵢ/μᵢ` over the directions the decrement sums.
    fn frame_lifted_spectral_step(
        joint: &ExactHessianSpectralBlock,
        verdict: &FrameLiftedVerdict,
    ) -> (Array1<f64>, f64) {
        let dim = joint.eigenvalues.len();
        let saddle_free = !verdict.refused.is_empty();
        let pencil_floor = sae_exact_a_pencil_floor();
        let mut weights = Array1::<f64>::zeros(dim);
        let mut curvature = 0.0_f64;
        for index in 0..dim {
            let mu = joint.eigenvalues[index];
            let scale = if saddle_free {
                if mu.abs() <= joint.rank_floor(index) {
                    continue;
                }
                mu.abs()
            } else {
                if mu <= pencil_floor.max(joint.resolution[index]) {
                    continue;
                }
                mu
            };
            let weight = -verdict.coefficients[index] / scale;
            weights[index] = weight;
            curvature += weight * weight * mu;
        }
        (joint.eigenvectors.dot(&weights), curvature)
    }

    /// #3434 — accept the evidence root in the lifted chart of the learned frames.
    ///
    /// A certified root settles. Otherwise the pencil's spectral step and then each
    /// refused basin direction are descended along the chart curve, and the first to
    /// commit a decrease above the material floor moves the root. Each commit lowers
    /// the gate-frozen objective by more than that floor, so repeated acceptances end.
    /// A saddle none of them descends is refused; a positive root whose decrement no
    /// step realizes is not a converged optimum, and is refused with its decrement.
    fn settle_frame_lifted_root(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        cache: &ArrowFactorCache,
    ) -> Result<FrameLiftedRoot, String> {
        if !self.frame_lifted_chart_admitted(cache.delta_t_len())? {
            return Ok(FrameLiftedRoot::Settled);
        }
        let information = self.frame_marginal_information(rho, target, registry)?;
        let verdict = self.frame_lifted_verdict(target, rho, registry, &information)?;
        let certificate = &verdict.certificate;
        log::debug!(
            "[SAE-FRAME-LIFT] lifted root: min μ {:.6e}, resolved negative {:?}, clamp \
             explained {}, λ² {:.6e}, ½λ²/scale {:.6e}, certified {}",
            certificate.min_curvature,
            certificate.resolved_negative,
            certificate.clamp_explained,
            certificate.lambda_sq,
            certificate.relative,
            certificate.certified,
        );
        if certificate.certified {
            return Ok(FrameLiftedRoot::Settled);
        }
        let mut candidates = vec![Self::frame_lifted_spectral_step(&information.joint, &verdict)];
        candidates.extend(verdict.refused.iter().cloned());
        for (direction, curvature) in &candidates {
            if self.descend_frame_lifted_direction(
                target,
                rho,
                registry,
                &information,
                direction.view(),
                *curvature,
                verdict.objective,
            )? {
                return Ok(FrameLiftedRoot::Moved);
            }
        }
        if !verdict.refused.is_empty() {
            log::debug!(
                "[SAE-FRAME-LIFT] none of {} lifted direction(s) realizes a decrease above the \
                 material floor; the lifted saddle stays refused",
                candidates.len(),
            );
            return Ok(FrameLiftedRoot::Saddle);
        }
        Err(format!(
            "frame-lifted root: ½λ²/scale = {:.6e} (λ² = {:.6e}) exceeds the decrement \
             tolerance in the lifted chart, and no lifted step lowers the objective \
             {:.10e} by more than its material floor",
            certificate.relative, certificate.lambda_sq, verdict.objective,
        ))
    }

    /// One joint `(t, ξ)` direction descended along the chart curve, under the commit
    /// law of the exact-A saddle descent: the line minimum is applied, re-evaluated,
    /// and kept only if it lowers the objective by more than the material floor;
    /// otherwise the state is restored. `curvature` is `dᵀA_ξd` for `d` as given.
    fn descend_frame_lifted_direction(
        &mut self,
        target: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        registry: Option<&AnalyticPenaltyRegistry>,
        information: &FrameMarginalInformation,
        direction: ArrayView1<'_, f64>,
        curvature: f64,
        base_objective: f64,
    ) -> Result<bool, String> {
        if !base_objective.is_finite() {
            return Ok(false);
        }
        let along = information.gradient.dot(&direction);
        let oriented = if along > 0.0 {
            direction.mapv(|value| -value)
        } else {
            direction.to_owned()
        };
        let Some((step, norm)) =
            information
                .tangent
                .chart_step(self, oriented.view(), information.total_t)?
        else {
            return Ok(false);
        };
        let slope = along.abs() / norm;
        let negative_curvature = (-curvature).max(0.0) / (norm * norm);
        let material_floor =
            SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL * (1.0 + base_objective.abs());
        let snapshot = self.snapshot_mutable_state();
        let line = self.minimize_objective_along_curve(
            target,
            rho,
            registry,
            &|term: &mut Self, alpha: f64| term.advance_along_learned_frame_chart(&step, alpha),
            base_objective,
            slope,
            negative_curvature,
            material_floor,
            &snapshot,
        )?;
        if !(line.alpha > 0.0 && base_objective - line.value > material_floor) {
            return Ok(false);
        }
        if let Err(err) = self.advance_along_learned_frame_chart(&step, line.alpha) {
            self.restore_mutable_state(&snapshot).map_err(|restore_err| {
                format!(
                    "frame-lifted root: committed chart step failed ({err}); restoring the \
                     pre-descent state also failed ({restore_err})"
                )
            })?;
            return Err(format!("frame-lifted root: committed chart step: {err}"));
        }
        let committed_objective = match self.penalized_objective_total(target, rho, registry, 1.0)
        {
            Ok(value) => value,
            Err(err) => {
                self.restore_mutable_state(&snapshot).map_err(|restore_err| {
                    format!(
                        "frame-lifted root: committed objective evaluation failed ({err}); \
                         restoring the pre-descent state also failed ({restore_err})"
                    )
                })?;
                return Err(format!("frame-lifted root: committed objective evaluation: {err}"));
            }
        };
        let decrease = base_objective - committed_objective;
        if !(committed_objective.is_finite() && decrease > material_floor) {
            self.restore_mutable_state(&snapshot)
                .map_err(|err| format!("frame-lifted root: {err}"))?;
            return Ok(false);
        }
        log::debug!(
            "[SAE-FRAME-LIFT] descended the lifted chart: curvature {:.6e}, slope {slope:.6e}, \
             α={:.6e}, objective {base_objective:.10e} → {committed_objective:.10e} (decrease \
             {decrease:.6e}, floor {material_floor:.6e}, {} objective evaluations)",
            curvature / (norm * norm),
            line.alpha,
            line.objective_evaluations,
        );
        Ok(true)
    }

    /// Move the state to `α` along a unit-speed chart velocity: the coordinates by
    /// `α·δt`, each framed decoder along `B(α) = (C + α·δC)(U + α·N)ᵀ` with its frame
    /// the polar factor of `U + α·N`, and each unframed decoder by `α·δB`.
    fn advance_along_learned_frame_chart(
        &mut self,
        step: &LearnedFrameChartStep,
        alpha: f64,
    ) -> Result<(), String> {
        if step.atoms.len() != self.k_atoms() {
            return Err(format!(
                "frame-lifted root: chart step carries {} atoms for {}",
                step.atoms.len(),
                self.k_atoms()
            ));
        }
        let border_len = if self.last_frames_active {
            self.factored_border_dim()
        } else {
            self.beta_dim()
        };
        self.apply_newton_step(
            step.delta_t.view(),
            Array1::<f64>::zeros(border_len).view(),
            alpha,
        )?;
        for (atom, motion) in self.atoms.iter_mut().zip(&step.atoms) {
            match motion {
                LearnedFrameChartMotion::Framed {
                    delta_coordinates,
                    normal_velocity,
                } => atom.advance_decoder_frame_along_tangent(
                    delta_coordinates.view(),
                    normal_velocity.view(),
                    alpha,
                )?,
                LearnedFrameChartMotion::Unframed { delta_decoder } => atom
                    .decoder_coefficients_mut()
                    .scaled_add(alpha, delta_decoder),
            }
        }
        Ok(())
    }
}
