//! #1033 — the chart-geometry amortized-routing predictor (kept out of
//! `construction.rs`, which sits at the #780 line-count ceiling). A separate
//! `impl SaeManifoldTerm` block deriving the ρ-invariant frozen routing from the
//! current dictionary's encode-chart geometry; see
//! [`RoutingPredictor::ChartGeometry`].

use super::*;
use crate::amortized_encoder::{
    AmortizationGap, AmortizedCode, ExactRowSolution, LearnedAmortizedEncoder,
};
use crate::encode::joint_encode_fallback_fraction;
use super::outer_objective::reconstruction_explained_variance;

impl SaeManifoldTerm {
    /// #2 (reviewer condition) — fit the DISTILLED / AMORTIZED encoder against
    /// this fitted term's own exact per-row solution. The exact solver's
    /// converged state — gate logits, per-atom coords, and rho-resolved
    /// assignment masses — is the supervision target; the encoder learns the
    /// one-matmul map `x ↦ (logits, coords, amplitudes)` that reproduces it. The
    /// evidence chooses the encoder's capacity (linear vs diagonal-quadratic).
    ///
    /// `targets` is the same `n × p` ambient corpus the dictionary was fit
    /// against (the encoder regresses the exact code on `x`). The returned
    /// encoder is deployed on held-out rows via [`Self::amortized_encode`].
    pub fn fit_amortized_encoder(
        &self,
        targets: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
    ) -> Result<LearnedAmortizedEncoder, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let p = self.output_dim();
        if targets.dim() != (n, p) {
            return Err(format!(
                "fit_amortized_encoder: targets {:?} must be (n={n}, p={p})",
                targets.dim()
            ));
        }
        let logits = self.assignment.logits.clone();
        let mut coords = Vec::with_capacity(k);
        for atom_idx in 0..k {
            coords.push(self.assignment.coords[atom_idx].as_matrix().to_owned());
        }
        let amplitudes = self.fitted_assignment_amplitudes(rho)?;
        // Seam-invariant periodic path: circular axes are regressed through
        // their (cos, sin) embedding instead of the raw coordinate, so a chart
        // seam inside the data cloud no longer pulls predictions to the
        // antipode. Periods are STRUCTURAL (read from each atom's basis kind).
        let periods = self.amortized_axis_periods();
        LearnedAmortizedEncoder::fit_with_axis_periods(
            targets,
            logits.view(),
            &coords,
            amplitudes.view(),
            &periods,
        )
    }

    /// Per-atom, per-axis period descriptors for the amortized encoder's
    /// seam-invariant periodic path: `Some(period)` for a circular axis, `None`
    /// for Euclidean / interval / spherical ones. Structural — derived from the
    /// atom's basis kind, never inferred from data. When a manifold's flattened
    /// axis list does not tile the atom's latent dimension exactly (nested
    /// products with implicit widths), the atom conservatively falls back to
    /// all-`None`, which reproduces the raw (pre-periodic) behavior for that
    /// atom rather than mislabeling an axis.
    fn amortized_axis_periods(&self) -> Vec<crate::amortized_encoder::AxisPeriods> {
        fn push_axes(m: &LatentManifold, out: &mut Vec<Option<f64>>) {
            match m {
                LatentManifold::Circle { period } => out.push(Some(*period)),
                LatentManifold::Euclidean => out.push(None),
                LatentManifold::Interval { .. } => out.push(None),
                LatentManifold::Sphere { dim } => {
                    for _ in 0..*dim {
                        out.push(None);
                    }
                }
                LatentManifold::Product(parts) => {
                    for part in parts {
                        push_axes(part, out);
                    }
                }
                LatentManifold::ProductWithMetric { manifolds, .. } => {
                    for part in manifolds {
                        push_axes(part, out);
                    }
                }
            }
        }
        self.atoms
            .iter()
            .map(|atom| {
                let d = atom.latent_dim;
                let manifold = atom.basis_kind.latent_manifold(d);
                let mut axes = Vec::with_capacity(d);
                push_axes(&manifold, &mut axes);
                if axes.len() == d {
                    axes
                } else if let LatentManifold::Circle { period } = manifold {
                    // A circle manifold on a d-axis atom labels every axis with
                    // the shared period (the periodic evaluator's convention).
                    vec![Some(period); d]
                } else {
                    vec![None; d]
                }
            })
            .collect()
    }

    /// Decode an amortized code into an ambient reconstruction
    /// `Σ_k z_k · Φ_k(t̂_k) · B_k` — the one-matmul-encode reconstruction, decoded
    /// through the SAME frozen dictionary the exact path uses. An atom with no
    /// basis evaluator (non-differentiable image) contributes nothing.
    pub fn decode_amortized_code(&self, code: &AmortizedCode) -> Result<Array2<f64>, String> {
        let p = self.output_dim();
        let k = self.k_atoms();
        if code.coords.len() != k || code.amplitudes.ncols() != k {
            return Err(format!(
                "decode_amortized_code: code carries {} coord blocks / {} amplitude cols but K={k}",
                code.coords.len(),
                code.amplitudes.ncols()
            ));
        }
        let n = code.amplitudes.nrows();
        let mut recon = Array2::<f64>::zeros((n, p));
        for atom_idx in 0..k {
            let atom = &self.atoms[atom_idx];
            let Some(evaluator) = atom.basis_evaluator.as_ref() else {
                continue;
            };
            let block = &code.coords[atom_idx];
            if block.nrows() != n {
                return Err(format!(
                    "decode_amortized_code: coord block {atom_idx} has {} rows, expected {n}",
                    block.nrows()
                ));
            }
            let (phi, _jac) = evaluator.evaluate(block.view())?;
            let decoded = phi.dot(&atom.decoder_coefficients); // (n × p)
            for row in 0..n {
                let z = code.amplitudes[[row, atom_idx]];
                if z == 0.0 {
                    continue;
                }
                for col in 0..p {
                    recon[[row, col]] += z * decoded[[row, col]];
                }
            }
        }
        Ok(recon)
    }

    /// Deploy the distilled encoder on fresh rows: predict the code in one matmul
    /// and decode it to an ambient reconstruction. This is the PRIMARY
    /// out-of-sample path (the SAE-comparable one-matmul encode).
    pub fn amortized_encode(
        &self,
        encoder: &LearnedAmortizedEncoder,
        x_new: ArrayView2<'_, f64>,
    ) -> Result<(AmortizedCode, Array2<f64>), String> {
        let code = encoder.predict(x_new)?;
        let recon = self.decode_amortized_code(&code)?;
        Ok((code, recon))
    }

    /// Assemble the amortization-gap artifact (reviewer condition #2) on held-out
    /// rows `x_new`, given the EXACT solver's solution on the same rows
    /// (`exact_recon`, `exact_logits`, `exact_coords`, `exact_amplitudes` — as
    /// produced by the per-row test-time optimizer). Reports EV(exact) vs
    /// EV(amortized) — the oracle line and the deployed number — the coordinate /
    /// gate / amplitude error, and the joint multi-start-fallback fraction on the
    /// exact solution (the encode-tax cost multiplier). `amplitude_floor` is the
    /// mass above which an atom counts as co-active for the joint diagnostic.
    pub fn amortization_gap(
        &self,
        encoder: &LearnedAmortizedEncoder,
        x_new: ArrayView2<'_, f64>,
        exact: ExactRowSolution<'_>,
        amplitude_floor: f64,
    ) -> Result<AmortizationGap, String> {
        let (code, amortized_recon) = self.amortized_encode(encoder, x_new)?;
        let ev_exact = reconstruction_explained_variance(x_new, exact.recon);
        let ev_amortized = reconstruction_explained_variance(x_new, amortized_recon.view());
        let ev_gap = match (ev_exact, ev_amortized) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        // Wrap-aware scoring: per-axis errors on circular axes are charged on
        // the short arc (min(|Δ|, period − |Δ|)), so a seam-straddling exact/
        // amortized pair no longer registers as a near-full-period miss.
        let score_periods = self.amortized_axis_periods();
        let errors = LearnedAmortizedEncoder::error_stats_wrapped(
            &code,
            exact.logits,
            exact.coords,
            exact.amplitudes,
            &score_periods,
        )?;
        let joint_multistart_fraction = joint_encode_fallback_fraction(
            &self.atoms,
            exact.coords,
            exact.amplitudes,
            amplitude_floor,
        )?;
        Ok(AmortizationGap {
            ev_exact,
            ev_amortized,
            ev_gap,
            errors,
            joint_multistart_fraction,
            used_quadratic_head: encoder.used_quadratic_head,
            encoder_log_evidence: encoder.log_evidence,
            encoder_feature_dim: encoder.feature_dim,
            encoder_effective_dof: encoder.effective_dof,
        })
    }
}

impl SaeManifoldTerm {
    /// #1033 — CHART-GEOMETRY amortized routing predictor: derive a ρ-invariant
    /// per-(row, atom) routing LOGIT from the current dictionary's encode-chart
    /// geometry, with NO learned net. For each atom it runs the amortized encode
    /// of `targets` (the atlas distilled from the current decoder/basis) to a
    /// predicted coord `t̂_ik`, reconstructs the amplitude-1 image
    /// `γ_k(t̂_ik) = Bᵀ φ(t̂_ik)`, and maps the reconstruction ALIGNMENT
    /// `⟨x_i, γ̂_ik⟩` (with `γ̂` the unit-normalized image) to a logit via the
    /// fixed `gate_logit_scale`. Higher alignment ⇒ larger logit ⇒ the gate
    /// prefers atom `k` for row `i` — exactly the routing the dictionary's own
    /// geometry implies, recomputed from the CURRENT decoder so it tracks the
    /// dictionary as it evolves (unlike a static logit snapshot). The overall
    /// `gate_logit_scale` (a temperature) is the single calibratable knob; the
    /// cosine-aligned `⟨x, γ̂⟩` is otherwise on the natural `‖x‖` scale.
    ///
    /// This is the [`RoutingPredictor::ChartGeometry`] arm: install its output via
    /// `assignment.with_frozen_routing(Some(..))` to freeze the routing for an
    /// outer iterate. An atom whose predicted coord is uncertified falls back to
    /// the row's CURRENT logit (the encoder cannot improve a row it cannot
    /// certify, so the existing routing for that row is preserved).
    pub fn chart_geometry_routing_logits(
        &self,
        targets: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        gate_logit_scale: f64,
    ) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k_atoms = self.k_atoms();
        let p = self.output_dim();
        if targets.dim() != (n, p) {
            return Err(format!(
                "chart_geometry_routing_logits: targets {:?} != ({n}, {p})",
                targets.dim()
            ));
        }
        // Per-atom amortized encode (atlas distilled from the current dictionary)
        // → predicted coords t̂ + per-row certificates.
        let encoded = self.amortized_encode_fitted(targets, rho)?;
        // Start from the current logits so an uncertified ROW keeps its existing
        // routing verbatim (the predictor only overrides rows it can fully
        // certify — see the per-row gate below).
        let mut logits = self.assignment.logits.clone();
        // Accumulate the certified alignment logits and a per-(row, atom)
        // validity mask WITHOUT writing them into `logits` yet: an alignment
        // logit is `gate_logit_scale·⟨x, γ̂⟩` on the natural ‖x‖ scale, whereas the
        // legacy `self.assignment.logits` live on the trained-gate scale. Splicing
        // one certified alignment logit into a row while the other atoms in that
        // row keep their legacy logits would build a softmax row that MIXES the
        // two currencies — the gate would compare an alignment score against a
        // trained score and route on the scale mismatch, not the geometry. So we
        // route PER ROW by certification status: a row is rewritten to the
        // alignment scale only when EVERY atom is certified for it (the whole row
        // is single-currency); any row with an uncertified atom stays entirely on
        // the legacy scale.
        let mut aligned = Array2::<f64>::zeros((n, k_atoms));
        let mut valid = vec![true; n]; // row fully certified so far
        for atom_idx in 0..k_atoms {
            let atom = &self.atoms[atom_idx];
            let evaluator = match atom.basis_evaluator.as_ref() {
                Some(e) => e.clone(),
                // No evaluator ⇒ this atom can never be certified at a fresh coord,
                // so no row can be FULLY certified: mark every row legacy.
                None => {
                    valid.iter_mut().for_each(|v| *v = false);
                    continue;
                }
            };
            let result = &encoded[atom_idx];
            let m = atom.basis_size();
            for row in 0..n {
                if !result.certified[row] {
                    valid[row] = false;
                    continue;
                }
                // Reconstruct the amplitude-1 image γ_k(t̂) = Bᵀ φ(t̂).
                let t_hat: Vec<f64> = result.coords.row(row).iter().copied().collect();
                let coord_2d = Array2::from_shape_vec((1, atom.latent_dim), t_hat)
                    .map_err(|e| format!("chart_geometry_routing_logits: coord reshape: {e}"))?;
                let (phi, _jet) = evaluator.evaluate(coord_2d.view())?;
                let mut gamma = vec![0.0_f64; p];
                for basis_col in 0..m {
                    let phi_v = phi[[0, basis_col]];
                    if phi_v == 0.0 {
                        continue;
                    }
                    for out in 0..p {
                        gamma[out] += phi_v * atom.decoder_coefficients[[basis_col, out]];
                    }
                }
                // Unit-normalize the reconstruction (routing is amplitude-free —
                // the gate cares about DIRECTION/alignment, not magnitude) and
                // align with the row.
                let norm = gamma.iter().map(|g| g * g).sum::<f64>().sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    valid[row] = false;
                    continue;
                }
                let mut align = 0.0_f64;
                for out in 0..p {
                    align += targets[[row, out]] * (gamma[out] / norm);
                }
                if align.is_finite() {
                    aligned[[row, atom_idx]] = gate_logit_scale * align;
                } else {
                    valid[row] = false;
                }
            }
        }
        // Per-row gate: rewrite a row onto the alignment scale only if the whole
        // row certified; otherwise it keeps its legacy logits untouched.
        for row in 0..n {
            if valid[row] {
                for atom_idx in 0..k_atoms {
                    logits[[row, atom_idx]] = aligned[[row, atom_idx]];
                }
            }
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod amortized_encoder_glue_tests {
    //! Term-level integration of the distilled encoder: fit against a term's own
    //! exact per-row code, encode in one matmul, and assemble the
    //! amortization-gap artifact. The mission bar — the amortized (one-matmul)
    //! held-out EV reaches a derived fraction of the exact-solve EV.
    use super::*;
    use crate::assignment::{AssignmentMode, SaeAssignment};
    use crate::manifold::{EuclideanPatchEvaluator, SaeAtomBasisKind, SaeManifoldAtom};
    use gam_terms::latent::LatentManifold;
    use ndarray::{Array1, Array2};
    use std::sync::Arc;

    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn signed(&mut self) -> f64 {
            2.0 * self.unit() - 1.0
        }
    }

    /// Decoder directions / offsets shared by every atom instance so a held-out
    /// term uses the SAME dictionary as the trained one.
    const DIRS: [[f64; 6]; 2] = [
        [1.0, 0.3, -0.2, 0.1, 0.0, 0.4],
        [-0.1, 0.9, 0.2, -0.3, 0.5, 0.0],
    ];
    const OFFSETS: [[f64; 6]; 2] = [
        [0.2, -0.1, 0.0, 0.3, 0.1, -0.2],
        [0.0, 0.2, -0.3, 0.1, -0.1, 0.2],
    ];

    /// A planted two-atom flat term with its faithful ambient target and the exact
    /// code it was generated from — the fixture the amortization-gap tests fit and
    /// score against.
    struct PlantedTwoAtomTerm {
        /// The two-atom flat (degree-1) SAE term.
        term: SaeManifoldTerm,
        /// The ambient target `x = Σ_k z_k (b0_k + t_k·b1_k)`, faithful to the term.
        target: Array2<f64>,
        /// The term's ρ hyperparameters.
        rho: SaeManifoldRho,
        /// Planted exact per-(row, atom) gate logits.
        logits: Array2<f64>,
        /// Planted exact per-atom coordinate blocks.
        coords: Vec<Array2<f64>>,
        /// Planted exact per-(row, atom) amplitudes.
        amps: Array2<f64>,
    }

    /// Build a SELF-CONSISTENT [`PlantedTwoAtomTerm`]: the ambient target is
    /// `x = Σ_k z_k (b0_k + t_k·b1_k)` where `z_k` are the term's OWN assignment
    /// masses (softmax of the stored logits), so decoding the term's exact code
    /// reproduces `x` to machine precision (EV ≈ 1).
    fn planted_two_atom_term(n: usize, seed: u64) -> PlantedTwoAtomTerm {
        let p = 6usize;
        let k = 2usize;
        let mut rng = Lcg(seed);
        let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 1).expect("patch"));
        let mut coords_blocks: Vec<Array2<f64>> = Vec::new();
        for atom_idx in 0..k {
            let mut c = Array2::<f64>::zeros((n, 1));
            for row in 0..n {
                c[[row, 0]] = 0.5 * (atom_idx as f64 + 1.0) + 1.5 * rng.signed();
            }
            coords_blocks.push(c);
        }
        // Build atoms with decoder rows [offset; dir] on the degree-1 monomials.
        // Each atom's basis_values / jet must be evaluated at THAT atom's own
        // per-row coordinates (n rows), so the term's per-atom design matches the
        // assignment's `n_obs` (a 1-row probe would make `SaeManifoldTerm::new`
        // reject the shape).
        let mut atoms = Vec::new();
        for atom_idx in 0..k {
            let (phi_k, jet_k) = evaluator
                .evaluate(coords_blocks[atom_idx].view())
                .expect("eval");
            let m = phi_k.ncols();
            let mut dec = Array2::<f64>::zeros((m, p));
            for col in 0..p {
                dec[[0, col]] = OFFSETS[atom_idx][col];
                dec[[1, col]] = DIRS[atom_idx][col];
            }
            let atom = SaeManifoldAtom::new(
                "lin",
                SaeAtomBasisKind::EuclideanPatch,
                1,
                phi_k,
                jet_k,
                dec,
                Array2::<f64>::eye(m),
            )
            .expect("atom")
            .with_basis_second_jet(evaluator.clone());
            atoms.push(atom);
        }
        // Both atoms active (positive logits) so the softmax gives well-defined,
        // strictly-positive masses and the gate call (logit > 0) is unambiguous.
        let logits = Array2::<f64>::from_elem((n, k), 1.0);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            coords_blocks.clone(),
            vec![LatentManifold::Euclidean, LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .expect("assignment");
        let term = SaeManifoldTerm::new(atoms, assignment).expect("term");
        let rho =
            SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::zeros(1), Array1::zeros(1)]);
        // The exact amplitudes are the term's OWN masses; generate x from them so
        // the term's exact code reconstructs x exactly (self-consistent).
        let amps = term.fitted_assignment_amplitudes(&rho).expect("masses");
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for atom_idx in 0..k {
                let t = coords_blocks[atom_idx][[row, 0]];
                let z = amps[[row, atom_idx]];
                for col in 0..p {
                    x[[row, col]] += z * (OFFSETS[atom_idx][col] + t * DIRS[atom_idx][col]);
                }
            }
        }
        PlantedTwoAtomTerm {
            term,
            target: x,
            rho,
            logits,
            coords: coords_blocks,
            amps,
        }
    }

    /// The distilled encoder, fit to the term's exact code, reproduces the
    /// held-out reconstruction to a high fraction of the exact-solve EV — the
    /// amortization gap is small — and the artifact fields are all well-formed.
    #[test]
    fn amortized_encode_reaches_high_fraction_of_exact_ev() {
        let PlantedTwoAtomTerm {
            term,
            target: x_tr,
            rho,
            ..
        } = planted_two_atom_term(400, 7);
        // Fit the encoder on the term's own exact code (logits/coords/masses).
        let encoder = term
            .fit_amortized_encoder(x_tr.view(), &rho)
            .expect("fit encoder");

        // Held-out rows from the SAME dictionary (identical atoms, fresh coords).
        let PlantedTwoAtomTerm {
            term: term_te,
            target: x_te,
            logits: lg_te,
            coords: co_te,
            amps: am_te,
            ..
        } = planted_two_atom_term(200, 99);
        // The oracle line: the exact code decoded through the (identical) frozen
        // dictionary reconstructs the held-out target to machine precision.
        let exact_recon_te = {
            let code = crate::amortized_encoder::AmortizedCode {
                logits: lg_te.clone(),
                coords: co_te.clone(),
                amplitudes: am_te.clone(),
            };
            term_te.decode_amortized_code(&code).expect("decode te")
        };

        let gap = term
            .amortization_gap(
                &encoder,
                x_te.view(),
                ExactRowSolution {
                    recon: exact_recon_te.view(),
                    logits: lg_te.view(),
                    coords: &co_te,
                    amplitudes: am_te.view(),
                },
                1.0e-9,
            )
            .expect("gap");

        let ev_exact = gap.ev_exact.expect("exact EV defined");
        let ev_amortized = gap.ev_amortized.expect("amortized EV defined");
        eprintln!(
            "[ENCODE-GAP] EV_exact={:.4} EV_amortized={:.4} EV_gap={:.4} \
             coord_rmse={:.4} gate_agreement={:.4} amp_rmse={:.4} \
             joint_multistart_frac={:.4} used_quadratic_head={} log_evidence={:.1}",
            ev_exact,
            ev_amortized,
            gap.ev_gap.unwrap_or(f64::NAN),
            gap.errors.coord_rmse,
            gap.errors.gate_agreement,
            gap.errors.amplitude_rmse,
            gap.joint_multistart_fraction,
            gap.used_quadratic_head,
            gap.encoder_log_evidence,
        );
        // The exact reconstruction is faithful by construction (self-consistent).
        assert!(
            ev_exact > 0.99,
            "planted exact reconstruction must be near-perfect, got {ev_exact}"
        );
        // The mission bar: the one-matmul encode recovers a high fraction of the
        // exact-solve EV. The gap is the deployed encode cost.
        assert!(
            ev_amortized >= 0.9 * ev_exact,
            "amortized EV {ev_amortized} must reach >=90% of exact EV {ev_exact} on a \
             linearly-encodable dictionary"
        );
        assert!(
            gap.ev_gap.expect("gap defined") >= -1.0e-6,
            "the exact solve cannot be materially BEATEN by its own amortization"
        );
        assert!(
            (0.0..=1.0).contains(&gap.joint_multistart_fraction),
            "joint fallback fraction must be a probability, got {}",
            gap.joint_multistart_fraction
        );
        // The error-stats half of the artifact is well-formed.
        assert!(gap.errors.coord_rmse.is_finite() && gap.errors.coord_rmse >= 0.0);
        assert!((0.0..=1.0).contains(&gap.errors.gate_agreement));
    }
}
