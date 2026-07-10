//! On-manifold causal steering by chart-coordinate group action (gam#2234).
//!
//! A flat SAE steers by adding a fixed ambient direction `x' = x + α·w`, which
//! walks off the data manifold — the model sees an input it never produces. A
//! manifold SAE has a stronger primitive: each atom `k` is a chart
//! (`t ∈ M_k`, amplitude `a`, decoder `Φ_k(t)·B_k`), so steering moves the CODE
//! instead of the ambient vector:
//!
//! ```text
//!   x' = x + a·(Φ_k(t ⊕ δ) − Φ_k(t))·B_k
//! ```
//!
//! where `⊕` is the manifold group action carried by the atom's
//! [`LatentManifold`] — reusing [`LatentManifold::retract`], the SAME
//! canonicalizing map the fit optimizes against, so no wrapping / modular
//! arithmetic is re-implemented here:
//!
//! * `Circle { period }` — per-axis phase add modulo `period` (the `S¹` group
//!   action); `retract(t, δ) = wrap(t + δ)`.
//! * `Euclidean` — translation `t + δ`.
//! * `Interval { lo, hi }` — clamped translation (the bounded-patch action).
//! * `Sphere` — the embedded normalized retraction `(t + δ)/‖t + δ‖`.
//! * `Product` / `ProductWithMetric` — the group action applied blockwise per
//!   factor.
//!
//! The steered contribution keeps the FITTED amplitude `exp(s_k)` and the fitted
//! per-row gate `a_{ik}` untouched: the intervention changes only WHICH value
//! the feature takes (content) at fixed strength, never the strength itself.
//!
//! Two surfaces:
//! * [`SaeManifoldTerm::steer_rows`] returns the ambient DELTA
//!   `a·(Φ(t⊕δ)−Φ(t))·B_k` for the selected rows — the caller ADDS it to `x`.
//! * [`SaeManifoldTerm::steer_decode`] returns the full steered per-atom
//!   contribution `a·Φ(t⊕δ)·B_k` (absolute, not a delta) for E4 zoo scoring.

use super::*;

impl SaeManifoldAtom {
    /// Ambient decode `exp(s_k)·Φ_k(coords)·B_k` at ARBITRARY coordinates,
    /// returned as an `(coords.nrows(), p)` matrix — the steering counterpart of
    /// [`Self::fill_decoded_row`], which reads the atom's stored on-atom basis.
    ///
    /// Evaluation goes through the atom's OWN installed basis evaluator, so a
    /// #1117 rank-reduced atom (whose evaluator is a `SubspaceReducedEvaluator`
    /// emitting the reduced `Φ̃` matched to the reduced decoder `B̃ = Qᵀ B`) stays
    /// width-consistent by construction — `Φ̃` and `B̃` are read from the same
    /// reduced representation, so no full-width re-expansion is needed (the #2135
    /// width-mismatch hazard is avoided because we never rebuild the inner basis).
    /// Honors the curvature-homotopy dial exactly as [`Self::refresh_basis`] does
    /// (`η = 1` ⇒ the un-dialed basis, bit-for-bit `evaluate`).
    ///
    /// Errors when the atom has no installed evaluator (a caller-managed basis
    /// cannot be re-evaluated at off-grid coordinates) or when `coords` has the
    /// wrong latent width.
    pub fn decode_at_coords(&self, coords: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        let evaluator = self.basis_evaluator.as_ref().ok_or_else(|| {
            "SaeManifoldAtom::decode_at_coords: atom has no installed basis evaluator; a \
             caller-managed basis cannot be re-evaluated at steered coordinates"
                .to_string()
        })?;
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "SaeManifoldAtom::decode_at_coords: coords have {} columns, latent_dim is {}",
                coords.ncols(),
                self.latent_dim
            ));
        }
        let phi = if self.homotopy_eta == 1.0 {
            evaluator.evaluate(coords)?.0
        } else {
            evaluator.evaluate_phi_eta(coords, self.homotopy_eta)?.phi
        };
        let m = self.basis_size();
        if phi.ncols() != m {
            return Err(format!(
                "SaeManifoldAtom::decode_at_coords: evaluator emitted {} basis columns but the \
                 decoder expects {m}",
                phi.ncols()
            ));
        }
        let mut decoded = phi.dot(&self.decoder_coefficients);
        // #2022 — the fitted contribution is exp(s_k)·Φ·B; scale to match the
        // stored decode. Skipped at s_k == 0.0 (default) so the amplitude-free
        // atom decodes bit-for-bit as `Φ·B`.
        if self.log_amplitude != 0.0 {
            let amp = self.log_amplitude.exp();
            decoded.mapv_inplace(|v| v * amp);
        }
        Ok(decoded)
    }
}

impl SaeManifoldTerm {
    fn honest_crosscoder_layer_values(
        &self,
        values: &Array2<f64>,
        layer: CrosscoderLayer,
    ) -> Result<Array2<f64>, String> {
        let layout = self.crosscoder_layout().ok_or_else(|| {
            "SaeManifoldTerm::steer_layer: no crosscoder layout is installed".to_string()
        })?;
        if values.ncols() != layout.total_dim() {
            return Err(format!(
                "SaeManifoldTerm::steer_layer: augmented value width {} != layout width {}",
                values.ncols(),
                layout.total_dim()
            ));
        }
        match layer {
            CrosscoderLayer::Anchor => Ok(values.slice(s![.., 0..layout.anchor_dim()]).to_owned()),
            CrosscoderLayer::Block(index) => {
                if index >= layout.num_blocks() {
                    return Err(format!(
                        "SaeManifoldTerm::steer_layer: block {index} out of range (L-1 = {})",
                        layout.num_blocks()
                    ));
                }
                let inv_scale = 1.0 / layout.sqrt_lambda(index);
                Ok(values
                    .slice(s![.., layout.block_range(index)])
                    .mapv(|value| value * inv_scale))
            }
        }
    }

    /// Apply the manifold group action `t ⊕ δ` to atom `k`'s coordinates on the
    /// selected `rows`, returning the steered coordinates `(rows.len(), d_k)`.
    ///
    /// `δ` is a single length-`d_k` chart-coordinate step applied to every
    /// selected row through [`LatentManifold::retract`] — the group action of the
    /// atom's own manifold (circle phase add, translation, blockwise product).
    /// The FITTED coordinates are read (never mutated) from the assignment state.
    fn steered_coords(
        &self,
        atom: usize,
        rows: &[usize],
        delta: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if atom >= self.k_atoms() {
            return Err(format!(
                "SaeManifoldTerm::steer: atom index {atom} out of range (K = {})",
                self.k_atoms()
            ));
        }
        let coord = &self.assignment.coords[atom];
        let d = coord.latent_dim();
        if delta.len() != d {
            return Err(format!(
                "SaeManifoldTerm::steer: delta length {} != atom {atom} latent_dim {d}",
                delta.len()
            ));
        }
        let manifold = coord.manifold();
        let base = coord.as_matrix();
        let n = coord.n_obs();
        let mut steered = Array2::<f64>::zeros((rows.len(), d));
        for (out_row, &row) in rows.iter().enumerate() {
            if row >= n {
                return Err(format!(
                    "SaeManifoldTerm::steer: row {row} out of range (n = {n})"
                ));
            }
            // t ⊕ δ — the group action via the atom's manifold retraction.
            let moved = manifold.retract(base.row(row), delta);
            for a in 0..d {
                steered[[out_row, a]] = moved[a];
            }
        }
        Ok(steered)
    }

    /// Gather atom `k`'s FITTED coordinates for the selected `rows`, as an
    /// `(rows.len(), d_k)` matrix — the un-steered baseline decoded through the
    /// SAME evaluator path as the steered coordinates, so their difference is
    /// exactly the group-action delta with no on-grid/off-grid discretization
    /// mismatch (and `δ = 0` yields a bit-identical matrix ⇒ an exactly-zero
    /// steer delta, since the coordinates are already manifold-projected and
    /// `retract(t, 0)` is idempotent on them).
    fn base_coords(&self, atom: usize, rows: &[usize]) -> Result<Array2<f64>, String> {
        let coord = &self.assignment.coords[atom];
        let d = coord.latent_dim();
        let base = coord.as_matrix();
        let n = coord.n_obs();
        let mut out = Array2::<f64>::zeros((rows.len(), d));
        for (out_row, &row) in rows.iter().enumerate() {
            if row >= n {
                return Err(format!(
                    "SaeManifoldTerm::steer: row {row} out of range (n = {n})"
                ));
            }
            for a in 0..d {
                out[[out_row, a]] = base[[row, a]];
            }
        }
        Ok(out)
    }

    /// Fitted per-row gates `a_{ik}` for the selected `rows` (one scalar per row),
    /// read from the fitted assignment state (its own logits / mode), untouched
    /// by the steer.
    fn steer_gates(&self, atom: usize, rows: &[usize]) -> Result<Vec<f64>, String> {
        let mut gates = Vec::with_capacity(rows.len());
        for &row in rows {
            let a = self.assignment.try_assignments_row(row)?;
            if atom >= a.len() {
                return Err(format!(
                    "SaeManifoldTerm::steer: atom index {atom} out of range for assignment row \
                     (K = {})",
                    a.len()
                ));
            }
            gates.push(a[atom]);
        }
        Ok(gates)
    }

    /// The ambient steering DELTA `a_{ik}·(Φ_k(t_i ⊕ δ) − Φ_k(t_i))·B_k` for atom
    /// `k` on the selected `rows`, shape `(rows.len(), p)`. The caller adds this
    /// to the ambient activation `x` to move token `i` along atom `k`'s chart by
    /// the intrinsic coordinate step `δ` (radians / fraction-of-period, per the
    /// atom's manifold), staying on the decoded feature image by construction.
    ///
    /// `δ` is a length-`d_k` chart step (the SAME `δ` applied to every selected
    /// row). Amplitude `exp(s_k)` and gate `a_{ik}` are the fitted values,
    /// untouched — the delta changes content at fixed strength. Circle closure is
    /// exact up to floating-point wrap: `δ = 0` is an exactly-zero delta and
    /// `δ = period` returns to the start (`|Δ| ≈ 0`).
    pub fn steer_rows(
        &self,
        atom: usize,
        rows: &[usize],
        delta: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let steered_coords = self.steered_coords(atom, rows, delta)?;
        let base_coords = self.base_coords(atom, rows)?;
        let atom_ref = &self.atoms[atom];
        let steered_decode = atom_ref.decode_at_coords(steered_coords.view())?;
        let base_decode = atom_ref.decode_at_coords(base_coords.view())?;
        let gates = self.steer_gates(atom, rows)?;
        let p = self.output_dim();
        let mut out = Array2::<f64>::zeros((rows.len(), p));
        for out_row in 0..rows.len() {
            let a = gates[out_row];
            if a == 0.0 {
                continue;
            }
            for c in 0..p {
                out[[out_row, c]] = a * (steered_decode[[out_row, c]] - base_decode[[out_row, c]]);
            }
        }
        Ok(out)
    }

    /// Crosscoder-aware steering delta for one layer in honest activation
    /// units. The shared fitted coordinate is moved exactly once by
    /// [`Self::steer_rows`]; this accessor selects the requested decoder column
    /// block and removes its `sqrt(lambda_l)` fit-space scaling.
    pub fn steer_layer_delta(
        &self,
        atom: usize,
        layer: CrosscoderLayer,
        rows: &[usize],
        delta: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let augmented = self.steer_rows(atom, rows, delta)?;
        self.honest_crosscoder_layer_values(&augmented, layer)
    }

    /// The full steered per-atom contribution `a_{ik}·Φ_k(t_i ⊕ δ)·B_k` for atom
    /// `k` on the selected `rows`, shape `(rows.len(), p)` — the ABSOLUTE moved
    /// contribution (not a delta), used by the E4 zoo ground-truth check where
    /// the steered reconstruction is compared against the planted manifold point
    /// at `θ + δ`. Same amplitude/gate handling as [`Self::steer_rows`].
    pub fn steer_decode(
        &self,
        atom: usize,
        rows: &[usize],
        delta: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let steered_coords = self.steered_coords(atom, rows, delta)?;
        let steered_decode = self.atoms[atom].decode_at_coords(steered_coords.view())?;
        let gates = self.steer_gates(atom, rows)?;
        let p = self.output_dim();
        let mut out = Array2::<f64>::zeros((rows.len(), p));
        for out_row in 0..rows.len() {
            let a = gates[out_row];
            if a == 0.0 {
                continue;
            }
            for c in 0..p {
                out[[out_row, c]] = a * steered_decode[[out_row, c]];
            }
        }
        Ok(out)
    }

    /// Absolute steered contribution in one crosscoder layer's honest units.
    /// Anchor and downstream layers read the same moved chart coordinate and
    /// fitted gate; only their decoder column blocks differ.
    pub fn steer_layer_decode(
        &self,
        atom: usize,
        layer: CrosscoderLayer,
        rows: &[usize],
        delta: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let augmented = self.steer_decode(atom, rows, delta)?;
        self.honest_crosscoder_layer_values(&augmented, layer)
    }
}
