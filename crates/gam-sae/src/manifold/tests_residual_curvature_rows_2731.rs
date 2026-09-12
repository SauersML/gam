//! #2731 — the residual-curvature plan against the per-apply form it replaced.
//!
//! `apply_exact_hessian_minus_b_by_row_jets` below is the non-softmax body of
//! `apply_exact_hessian_minus_b_prepared` as it stood before
//! [`PreparedResidualCurvatureRows`] existed: every apply rebuilds the row
//! program's jets, recomputes the row residual, and contracts the two. It is
//! kept only as this pin's independent arm. The plan changes WHEN those two
//! contractions happen and nothing else, so the pin asks for bit-identity, on a
//! direction whose `t` and `β` components are all nonzero, over both
//! ThresholdGate strata and the unweighted and weighted ordered Beta--Bernoulli
//! fixture. A mutation arm perturbs the largest contracted entry of each leg and
//! requires the comparison to see it, so a pin that cannot fail does not pass.
#![cfg(test)]
use super::*;
use crate::assignment::AssignmentMode;
use crate::manifold::tests_logdet_adjoint_780::obb_patchd_fixture;
use crate::manifold::tests_sparse_curvature_operator_2500::threshold_gate_tiny_fixture;
use gam_solve::arrow_schur::{ArrowSolveOptions, solve_arrow_newton_step_with_options};
use ndarray::{Array1, ArrayView2};

impl SaeManifoldTerm {
    /// The per-apply residual-curvature form, verbatim from before #2731, for a
    /// non-softmax gate. Test-only independent arm; production reads the plan.
    fn apply_exact_hessian_minus_b_by_row_jets(
        &self,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        v: &SaeArrowVector,
        prepared: &PreparedDecoderPriorBetaCurvature,
    ) -> Result<SaeArrowVector, String> {
        assert!(
            !matches!(self.assignment.mode, AssignmentMode::Softmax { .. }),
            "the per-apply oracle covers the jet-window gates; softmax has its own resident kernel"
        );
        self.assignment.validate_rho_domain(rho)?;
        let p = self.output_dim();
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let total_t = cache.delta_t_len();
        let second_jets = self.atom_second_jets()?;
        let border = self.border_channels_for_cache(cache)?;
        let row_loss_w = self.row_loss_weights.as_deref();
        let ard_axis_periods: Vec<Vec<Option<f64>>> = self
            .assignment
            .coords
            .iter()
            .map(|coord| coord.effective_axis_periods())
            .collect();
        let ard_precisions = self.validated_ard_precisions(rho)?;
        let mut out = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(cache.k),
        };
        let whitens = self
            .row_metric
            .as_ref()
            .is_some_and(|metric| metric.whitens_likelihood());
        let mut decoded = vec![0.0_f64; p];
        let mut fitted = Array1::<f64>::zeros(p);
        let mut error = Array1::<f64>::zeros(p);
        let mut assignments = Array1::<f64>::zeros(self.k_atoms());
        let mut ordered_logit_direction = matches!(
            self.assignment.mode,
            AssignmentMode::OrderedBetaBernoulli { .. }
        )
        .then(|| Array1::<f64>::zeros(n * k_atoms));
        let threshold_gate_remainder = match self.assignment.mode {
            AssignmentMode::ThresholdGate { .. } => Some(
                crate::assignment::threshold_gate_negative_hessian_remainder_weighted(
                    &self.assignment,
                    rho,
                    row_loss_w,
                )?,
            ),
            _ => None,
        };
        let mut jet_window: std::collections::VecDeque<SaeRowJets> =
            std::collections::VecDeque::new();
        let mut jet_window_next = 0usize;
        for row in 0..n {
            let q = cache.row_dims[row];
            let base = cache.row_offsets[row];
            let a_scratch = assignments.as_slice_mut().expect("contiguous scratch");
            self.assignment.try_assignments_row_into(row, a_scratch)?;
            if jet_window.is_empty() {
                jet_window_next = self.refill_jet_window(
                    jet_window_next,
                    cache,
                    &second_jets,
                    &border,
                    &mut jet_window,
                )?;
            }
            let jets = jet_window
                .pop_front()
                .expect("jet window must be non-empty");
            let sqrt_row_w = row_loss_w.map_or(1.0, |w| w[row].sqrt());
            fitted.fill(0.0);
            let active_atoms = self
                .last_row_layout
                .as_ref()
                .map(|layout| layout.active_atoms[row].as_slice());
            for k in 0..k_atoms {
                if active_atoms.is_some_and(|active| active.binary_search(&k).is_err()) {
                    continue;
                }
                self.atoms[k].fill_decoded_row(row, &mut decoded);
                let a_k = assignments[k];
                for out_col in 0..p {
                    fitted[out_col] += a_k * decoded[out_col];
                }
            }
            for out_col in 0..p {
                error[out_col] = sqrt_row_w * (fitted[out_col] - target[[row, out_col]]);
            }
            let error_metric: Vec<f64> = match self.row_metric.as_ref() {
                Some(metric) if whitens => metric.apply_metric_row(row, error.view()),
                _ => error.to_vec(),
            };
            let v_t: Vec<f64> = (0..q).map(|c| v.t[base + c]).collect();
            if let Some(direction) = ordered_logit_direction.as_mut() {
                for (local, var) in jets.vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        direction[row * k_atoms + atom] = v_t[local];
                    }
                }
            }
            for a in 0..q {
                let mut acc = 0.0_f64;
                for b in 0..q {
                    let r_ab = sae_dot(&error_metric, jets.second(a, b));
                    acc += r_ab * v_t[b];
                }
                out.t[base + a] += acc;
            }
            for a in 0..q {
                for (beta_pos, channel) in border.iter().enumerate() {
                    let r_ab = sae_dot(&error_metric, jets.beta_deriv(a, beta_pos));
                    out.t[base + a] += r_ab * v.beta[channel.index];
                    out.beta[channel.index] += r_ab * v_t[a];
                }
            }
            let w_row = row_loss_w.map_or(1.0, |w| w[row]);
            for (a, va) in jets.vars.iter().enumerate() {
                let SaeLocalRowVar::Coord { atom, axis } = *va else {
                    continue;
                };
                if rho.log_ard[atom].is_empty() {
                    continue;
                }
                let alpha = ard_precisions[atom][axis];
                let t_val = self.assignment.coords[atom].row(row)[axis];
                let prior = ArdAxisPrior::eval(alpha, t_val, ard_axis_periods[atom][axis]);
                let neg = prior.negative_hessian_remainder();
                if neg != 0.0 {
                    out.t[base + a] += w_row * neg * v_t[a];
                }
            }
            if let Some(remainder) = threshold_gate_remainder.as_ref() {
                for (a, va) in jets.vars.iter().enumerate() {
                    let SaeLocalRowVar::Logit { atom } = *va else {
                        continue;
                    };
                    let neg = remainder[row * k_atoms + atom];
                    if neg != 0.0 {
                        out.t[base + a] += neg * v_t[a];
                    }
                }
            }
        }
        if let Some(direction) = ordered_logit_direction {
            let delta = crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
                &self.assignment,
                rho,
                row_loss_w,
                direction.view(),
            )?;
            for row in 0..n {
                let base = cache.row_offsets[row];
                let vars = self.row_vars_for_cache_row(row, cache)?;
                for (local, var) in vars.iter().enumerate() {
                    if let SaeLocalRowVar::Logit { atom } = *var {
                        out.t[base + local] += delta[row * k_atoms + atom];
                    }
                }
            }
        }
        if cache.k > 0 {
            let beta_dim = self.beta_dim();
            let projection = crate::frames::FrameProjection::new(self);
            let framed = self.last_frames_active && cache.k == self.factored_border_dim();
            if framed {
                let lifted = projection.lift_border_vec(v.beta.view());
                let delta = self.decoder_prior_exact_minus_majorizer_beta_hvp_prepared(
                    prepared,
                    lifted.view(),
                )?;
                let projected = projection.project_border_vec(delta.view());
                for (index, &value) in projected.iter().enumerate() {
                    out.beta[index] += value;
                }
            } else if cache.k == beta_dim {
                let delta = self.decoder_prior_exact_minus_majorizer_beta_hvp_prepared(
                    prepared,
                    v.beta.view(),
                )?;
                for (index, &value) in delta.iter().enumerate() {
                    out.beta[index] += value;
                }
            } else {
                return Err(format!(
                    "apply_exact_hessian_minus_b_by_row_jets: border width {} is neither the \
                     full-B beta_dim {beta_dim} nor the factored border dim {}",
                    cache.k,
                    self.factored_border_dim(),
                ));
            }
        }
        Ok(out)
    }
}

/// Every component nonzero: `x/97 − ½` and `x/89 − ½` vanish only at a
/// half-integer `x`, which an integer residue never is. A zero component would
/// let a wrong contraction multiply into nothing and pass.
fn dense_direction(total_t: usize, k: usize) -> SaeArrowVector {
    SaeArrowVector {
        t: Array1::from_shape_fn(total_t, |i| ((i * 37 + 11) % 97) as f64 / 97.0 - 0.5),
        beta: Array1::from_shape_fn(k, |j| ((j * 53 + 7) % 89) as f64 / 89.0 - 0.5),
    }
}

fn differing_entries(left: &SaeArrowVector, right: &SaeArrowVector) -> (usize, f64) {
    assert_eq!(left.t.len(), right.t.len(), "t widths differ");
    assert_eq!(left.beta.len(), right.beta.len(), "β widths differ");
    let mut count = 0usize;
    let mut worst = 0.0_f64;
    for (a, b) in left
        .t
        .iter()
        .chain(left.beta.iter())
        .zip(right.t.iter().chain(right.beta.iter()))
    {
        if a.to_bits() != b.to_bits() {
            count += 1;
            worst = worst.max((a - b).abs());
        }
    }
    (count, worst)
}

/// `(row, index)` of the largest-magnitude nonzero contraction in one leg of a
/// plan — `t–t` when `tt`, else `t–β` — or `None` when that leg is identically
/// zero at this state.
fn largest_contraction(plan: &PreparedResidualCurvatureRows, tt: bool) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for (row, plan_row) in plan.rows.iter().enumerate() {
        let values = if tt {
            &plan_row.residual_tt
        } else {
            &plan_row.residual_tbeta
        };
        for (index, value) in values.iter().enumerate() {
            if *value != 0.0 && best.is_none_or(|(_, _, magnitude)| value.abs() > magnitude) {
                best = Some((row, index, value.abs()));
            }
        }
    }
    best.map(|(row, index, _)| (row, index))
}

/// Compare the plan with the per-apply form at one state. Returns whether the
/// `t–t` and `t–β` contractions were live there, so each test can require that
/// its arms exercised both legs at least once.
fn check_plan_against_per_apply(
    label: &str,
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> (bool, bool) {
    let prepared = term.prepare_decoder_prior_beta_curvature(1.0);
    let residual = term
        .prepare_residual_curvature_rows(target, cache)
        .expect("residual-curvature plan");
    assert_eq!(
        residual.rows.len(),
        term.n_obs(),
        "{label}: a jet-window gate must carry one plan row per observation"
    );
    let v = dense_direction(cache.delta_t_len(), cache.k);
    let oracle = term
        .apply_exact_hessian_minus_b_by_row_jets(rho, target, cache, &v, &prepared)
        .expect("per-apply form");
    for pass in ["first", "reused"] {
        let planned = term
            .apply_exact_hessian_minus_b_prepared(rho, target, cache, &v, &prepared, &residual)
            .expect("planned form");
        let (count, worst) = differing_entries(&planned, &oracle);
        assert_eq!(
            count, 0,
            "{label} ({pass} apply of one plan): {count} entries differ from the per-apply form, \
             worst {worst:.3e}"
        );
    }

    let tt_entry = largest_contraction(&residual, true);
    let tbeta_entry = largest_contraction(&residual, false);
    for (leg, entry) in [("t–t", tt_entry), ("t–β", tbeta_entry)] {
        let Some((row, index)) = entry else {
            continue;
        };
        let mut mutated = term
            .prepare_residual_curvature_rows(target, cache)
            .expect("residual-curvature plan for the mutation arm");
        let values = if leg == "t–t" {
            &mut mutated.rows[row].residual_tt
        } else {
            &mut mutated.rows[row].residual_tbeta
        };
        values[index] *= 1.0 + 1.0e-6;
        let perturbed = term
            .apply_exact_hessian_minus_b_prepared(rho, target, cache, &v, &prepared, &mutated)
            .expect("planned form against a perturbed plan");
        let (count, _) = differing_entries(&perturbed, &oracle);
        assert!(
            count > 0,
            "{label}: perturbing the largest {leg} contraction by a relative 1e-6 left the apply \
             bit-identical to the per-apply form, so this comparison cannot fail"
        );
    }
    (tt_entry.is_some(), tbeta_entry.is_some())
}

#[test]
fn residual_curvature_plan_equals_the_per_apply_form_on_threshold_gate_strata_2731() {
    let mut live = (false, false);
    for straddle in [false, true] {
        let (term, mut target, rho) = threshold_gate_tiny_fixture(straddle);
        // The fixture draws its target from the state it returns, so at this frozen
        // anchor every row residual is round-off, the contractions are round-off, and
        // a relative 1e-6 change to one is absorbed by the O(1) ARD and threshold
        // remainders: job 448182 saw the mutation arm leave the apply bit-identical
        // at straddle=false. A fixed offset the frozen state does not interpolate
        // gives both legs a residual of order 0.05.
        for ((row, col), value) in target.indexed_iter_mut() {
            *value += ((row * 5 + col * 3) % 7) as f64 / 70.0 - 0.04;
        }
        let mut anchor = term.clone();
        let (_value, _loss, cache) = anchor
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("threshold-gate fixed-theta cache");
        anchor.streaming_gates_frozen = true;
        let arm = check_plan_against_per_apply(
            &format!("threshold gate straddle={straddle}"),
            &anchor,
            target.view(),
            &rho,
            &cache,
        );
        live = (live.0 || arm.0, live.1 || arm.1);
    }
    assert!(
        live.0 && live.1,
        "the ThresholdGate arms must exercise both residual-curvature legs (t–t live: {}, t–β \
         live: {})",
        live.0,
        live.1
    );
}

#[test]
fn residual_curvature_plan_equals_the_per_apply_form_on_ordered_beta_bernoulli_2731() {
    let mut live = (false, false);
    for weighted in [false, true] {
        let (mut term, target, rho) = obb_patchd_fixture(0.01, -1.0);
        if weighted {
            term.set_row_loss_weights(
                (0..term.n_obs())
                    .map(|row| 0.5 + (row % 4) as f64 * 0.25)
                    .collect(),
            )
            .expect("positive design weights");
        }
        let system = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("fixed-state OBB assembly");
        let (_, _, cache) = solve_arrow_newton_step_with_options(
            &system,
            1.0e-6,
            1.0e-6,
            &ArrowSolveOptions::direct(),
        )
        .expect("positive Newton metric");
        let arm = check_plan_against_per_apply(
            &format!("ordered Beta-Bernoulli weighted={weighted}"),
            &term,
            target.view(),
            &rho,
            &cache,
        );
        live = (live.0 || arm.0, live.1 || arm.1);
    }
    assert!(
        live.0 && live.1,
        "the ordered Beta-Bernoulli arms must exercise both residual-curvature legs (t–t live: \
         {}, t–β live: {})",
        live.0,
        live.1
    );
}
