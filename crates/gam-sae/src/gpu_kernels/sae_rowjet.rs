//! Complete softmax SAE reconstruction row jets on CPU and CUDA (#2304).
//!
//! The quasi-Laplace consumers differentiate
//!
//! ```text
//! f_c = sum_k z_k D_{k,c}(t_k),       z = softmax(inv_tau * logits),
//! ```
//!
//! in the live row coordinates: free gate logits and every active latent
//! coordinate.  The authoritative contract here is therefore the complete
//! packed schedule consumed by `SaeRowJets`: reconstruction first and second
//! channels, decoder-border values, and decoder-border mixed channels.  There
//! is no gate-only compatibility surface.
//!
//! The CUDA kernels evaluate the same centered-moment identities as
//! [`crate::row_jet_program::execute_softmax_row_program`].  In particular they
//! emit both orientations of every logit-by-coordinate block and the
//! same-atom coordinate Hessian from the live decoded basis jets.  Every output
//! element costs O(1); the old seeded dense-jet recurrence, whose intermediates
//! carried O(q^2) channels and whose logit Hessian cost O(K) per output, is gone.
//!
//! Production calls this module through a bounded tile plan.  The plan accounts
//! for every host and device input/output byte, uses the calibrated row-kernel
//! crossover, and makes the CPU/device choice before allocating or launching.
//! Once a device tile is selected, NVRTC/allocation/launch failures are errors;
//! they are never hidden by retrying the tile on the CPU.

use crate::row_jet_program::{
    SaeRowPrimary, SaeScheduledRowJets, SaeSoftmaxRowProgramSource, execute_softmax_row_program,
};

/// One primary in a complete SAE reconstruction row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeRowJetPrimary {
    Logit { atom: usize },
    Coordinate { atom: usize, axis: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeCoordinateSlot {
    pub atom: usize,
    pub axis: usize,
    pub slot: usize,
}

fn public_primary(value: SaeRowPrimary) -> SaeRowJetPrimary {
    match value {
        SaeRowPrimary::Logit { atom } => SaeRowJetPrimary::Logit { atom },
        SaeRowPrimary::Coord { atom, axis } => SaeRowJetPrimary::Coordinate { atom, axis },
    }
}

fn scheduled_primary(value: SaeRowJetPrimary) -> SaeRowPrimary {
    match value {
        SaeRowJetPrimary::Logit { atom } => SaeRowPrimary::Logit { atom },
        SaeRowJetPrimary::Coordinate { atom, axis } => SaeRowPrimary::Coord { atom, axis },
    }
}

/// `(is_logit, atom)` for one primary — the discriminant the θ-adjoint trace
/// reduction branches on.
fn primary_kind_atom(value: SaeRowJetPrimary) -> (bool, usize) {
    match value {
        SaeRowJetPrimary::Logit { atom } => (true, atom),
        SaeRowJetPrimary::Coordinate { atom, .. } => (false, atom),
    }
}

/// Derivative of the softmax data-weight product `z_{atom_a}·z_{atom_b}` with
/// respect to logit `atom_w`, divided out to a per-pair factor. Mirrors
/// `SaeManifoldTerm::softmax_data_weight_product_logit_factor` so the contracted
/// trace oracle differentiates the exact same operator the dense adjoint does.
fn softmax_data_weight_product_logit_factor(
    assignments: &[f64],
    atom_a: usize,
    atom_b: usize,
    atom_w: usize,
    inv_tau: f64,
) -> f64 {
    let a_w = assignments[atom_w];
    let left = if atom_w == atom_a { 1.0 } else { 0.0 } - a_w;
    let right = if atom_w == atom_b { 1.0 } else { 0.0 } - a_w;
    (left + right) * inv_tau
}

/// Complete semantic inputs for one softmax row.
///
/// `decoded_first` is indexed by the live primary slot (`q * p`); logit slots
/// are exact zero. `decoded_second` is `q * q * p`; only same-atom coordinate
/// pairs can be nonzero. These are contractions of the term's live basis
/// Jacobian/Hessian with its decoder, not derivatives reconstructed from stale
/// coordinates or a second basis implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct SaeSoftmaxRowJetInput {
    pub n_atoms: usize,
    pub out_dim: usize,
    pub primaries: Vec<SaeRowJetPrimary>,
    pub coordinate_slots: Vec<SaeCoordinateSlot>,
    pub gate_values: Vec<f64>,
    pub active_atoms: Vec<bool>,
    pub sqrt_row_weight: f64,
    pub decoded: Vec<f64>,
    pub decoded_first: Vec<f64>,
    pub decoded_second: Vec<f64>,
    pub beta_atoms: std::sync::Arc<[usize]>,
    pub beta_basis_values: Vec<f64>,
    pub beta_basis_first: Vec<f64>,
    pub beta_outputs: std::sync::Arc<[f64]>,
}

impl SaeSoftmaxRowJetInput {
    pub fn coordinate_slots_for(primaries: &[SaeRowJetPrimary]) -> Vec<SaeCoordinateSlot> {
        let mut slots: Vec<SaeCoordinateSlot> = primaries
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(slot, primary)| match primary {
                SaeRowJetPrimary::Coordinate { atom, axis } => {
                    Some(SaeCoordinateSlot { atom, axis, slot })
                }
                SaeRowJetPrimary::Logit { .. } => None,
            })
            .collect();
        slots.sort_unstable_by_key(|entry| (entry.atom, entry.axis));
        slots
    }

    #[inline]
    pub fn n_primaries(&self) -> usize {
        self.primaries.len()
    }

    #[inline]
    pub fn n_beta_borders(&self) -> usize {
        self.beta_atoms.len()
    }

    /// Snapshot the authoritative borrowed production row source into a bounded
    /// tile input. No gate or basis law is re-evaluated here: values and decoded
    /// derivatives come from the exact live source used by the CPU schedule.
    pub(crate) fn from_source<S: SaeSoftmaxRowProgramSource>(
        source: &S,
        sqrt_row_weight: f64,
        shared_beta_layout: Option<(std::sync::Arc<[usize]>, std::sync::Arc<[f64]>)>,
    ) -> Result<Self, String> {
        let n_atoms = source.n_atoms();
        let out_dim = source.out_dim();
        let q = source.n_primaries();
        let n_beta = source.n_beta_borders();
        let primaries: Vec<SaeRowJetPrimary> = (0..q)
            .map(|slot| public_primary(source.primary(slot)))
            .collect();

        let decoded_len = n_atoms
            .checked_mul(out_dim)
            .ok_or_else(|| "SAE row-jet decoded length overflow".to_string())?;
        let first_len = q
            .checked_mul(out_dim)
            .ok_or_else(|| "SAE row-jet decoded-first length overflow".to_string())?;
        let second_len = q
            .checked_mul(q)
            .and_then(|value| value.checked_mul(out_dim))
            .ok_or_else(|| "SAE row-jet decoded-second length overflow".to_string())?;
        let beta_first_len = q
            .checked_mul(n_beta)
            .ok_or_else(|| "SAE row-jet beta-first length overflow".to_string())?;
        let beta_output_len = n_beta
            .checked_mul(out_dim)
            .ok_or_else(|| "SAE row-jet beta-output length overflow".to_string())?;

        let mut decoded = vec![0.0; decoded_len];
        for atom in 0..n_atoms {
            source.fill_decoded(atom, &mut decoded[atom * out_dim..(atom + 1) * out_dim]);
        }

        let mut decoded_first = vec![0.0; first_len];
        for (slot, primary) in primaries.iter().copied().enumerate() {
            if let SaeRowJetPrimary::Coordinate { atom, axis } = primary {
                source.fill_decoded_first(
                    atom,
                    axis,
                    &mut decoded_first[slot * out_dim..(slot + 1) * out_dim],
                );
            }
        }

        let mut decoded_second = vec![0.0; second_len];
        for (slot_a, primary_a) in primaries.iter().copied().enumerate() {
            let SaeRowJetPrimary::Coordinate {
                atom: atom_a,
                axis: axis_a,
            } = primary_a
            else {
                continue;
            };
            for (slot_b, primary_b) in primaries.iter().copied().enumerate() {
                let SaeRowJetPrimary::Coordinate {
                    atom: atom_b,
                    axis: axis_b,
                } = primary_b
                else {
                    continue;
                };
                if atom_a == atom_b {
                    let start = (slot_a * q + slot_b) * out_dim;
                    source.fill_decoded_second(
                        atom_a,
                        axis_a,
                        axis_b,
                        &mut decoded_second[start..start + out_dim],
                    );
                }
            }
        }

        let beta_atoms: std::sync::Arc<[usize]> = shared_beta_layout.as_ref().map_or_else(
            || {
                (0..n_beta)
                    .map(|border| source.beta_border_atom(border))
                    .collect::<Vec<_>>()
                    .into()
            },
            |(atoms, _)| atoms.clone(),
        );
        let beta_basis_values: Vec<f64> = (0..n_beta)
            .map(|border| source.beta_border_basis_value(border))
            .collect();
        let mut beta_basis_first = vec![0.0; beta_first_len];
        for (slot, primary) in primaries.iter().copied().enumerate() {
            let SaeRowJetPrimary::Coordinate { atom, axis } = primary else {
                continue;
            };
            for border in 0..n_beta {
                if beta_atoms[border] == atom {
                    beta_basis_first[slot * n_beta + border] =
                        source.beta_border_basis_first(border, axis);
                }
            }
        }
        let beta_outputs: std::sync::Arc<[f64]> = match shared_beta_layout {
            Some((_, outputs)) => outputs,
            None => {
                let mut outputs = vec![0.0; beta_output_len];
                for border in 0..n_beta {
                    outputs[border * out_dim..(border + 1) * out_dim]
                        .copy_from_slice(source.beta_border_output(border));
                }
                outputs.into()
            }
        };

        let input = Self {
            n_atoms,
            out_dim,
            coordinate_slots: Self::coordinate_slots_for(&primaries),
            primaries,
            gate_values: (0..n_atoms).map(|atom| source.gate_value(atom)).collect(),
            active_atoms: (0..n_atoms)
                .map(|atom| source.atom_is_active(atom))
                .collect(),
            sqrt_row_weight,
            decoded,
            decoded_first,
            decoded_second,
            beta_atoms,
            beta_basis_values,
            beta_basis_first,
            beta_outputs,
        };
        input.validate()?;
        Ok(input)
    }

    pub fn validate(&self) -> Result<(), String> {
        let k = self.n_atoms;
        let p = self.out_dim;
        let q = self.n_primaries();
        let n_beta = self.n_beta_borders();
        if k == 0 || p == 0 {
            return Err(format!(
                "complete SAE row jet requires nonzero atoms/output dimension; got K={k}, p={p}"
            ));
        }
        if !self.sqrt_row_weight.is_finite() || self.sqrt_row_weight < 0.0 {
            return Err(format!(
                "SAE row-jet sqrt_row_weight must be finite and nonnegative; got {}",
                self.sqrt_row_weight
            ));
        }
        let expected = [
            ("gate_values", self.gate_values.len(), k),
            ("active_atoms", self.active_atoms.len(), k),
            ("decoded", self.decoded.len(), checked_product(&[k, p])?),
            (
                "decoded_first",
                self.decoded_first.len(),
                checked_product(&[q, p])?,
            ),
            (
                "decoded_second",
                self.decoded_second.len(),
                checked_product(&[q, q, p])?,
            ),
            ("beta_basis_values", self.beta_basis_values.len(), n_beta),
            (
                "beta_basis_first",
                self.beta_basis_first.len(),
                checked_product(&[q, n_beta])?,
            ),
            (
                "beta_outputs",
                self.beta_outputs.len(),
                checked_product(&[n_beta, p])?,
            ),
        ];
        for (label, got, want) in expected {
            if got != want {
                return Err(format!(
                    "SAE row-jet {label} length {got} != expected {want}"
                ));
            }
        }
        if let Some((atom, value)) = self
            .gate_values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value < 0.0 || *value > 1.0)
        {
            return Err(format!(
                "SAE row-jet gate_values[{atom}] must be a probability; got {value}"
            ));
        }
        let gate_sum: f64 = self.gate_values.iter().sum();
        let rounding_mass = (k as f64) * f64::EPSILON;
        if rounding_mass >= 1.0 {
            return Err(format!(
                "SAE row-jet K={k} is too large to certify a normalized f64 softmax"
            ));
        }
        let summation_bound = rounding_mass / (1.0 - rounding_mass);
        let normalization_bound = f64::EPSILON + summation_bound + f64::EPSILON * summation_bound;
        if (gate_sum - 1.0).abs() > normalization_bound * gate_sum.abs().max(1.0) {
            return Err(format!(
                "SAE row-jet gate values must sum to one within the f64 normalization bound {normalization_bound:e}; got {gate_sum:e}"
            ));
        }
        for (slot, primary) in self.primaries.iter().copied().enumerate() {
            let atom = match primary {
                SaeRowJetPrimary::Logit { atom } | SaeRowJetPrimary::Coordinate { atom, .. } => {
                    atom
                }
            };
            if atom >= k {
                return Err(format!(
                    "SAE row-jet primary {slot} references atom {atom} outside K={k}"
                ));
            }
            if self.primaries[..slot].contains(&primary) {
                return Err(format!(
                    "SAE row-jet primary {primary:?} appears more than once"
                ));
            }
        }
        let expected_coordinate_slots = Self::coordinate_slots_for(&self.primaries);
        if self.coordinate_slots != expected_coordinate_slots {
            return Err(
                "SAE row-jet coordinate slot table is not the canonical primary-derived table"
                    .to_string(),
            );
        }
        for (border, &atom) in self.beta_atoms.iter().enumerate() {
            if atom >= k {
                return Err(format!(
                    "SAE row-jet beta border {border} references atom {atom} outside K={k}"
                ));
            }
        }
        for (label, values) in [
            ("gate_values", self.gate_values.as_slice()),
            ("decoded", self.decoded.as_slice()),
            ("decoded_first", self.decoded_first.as_slice()),
            ("decoded_second", self.decoded_second.as_slice()),
            ("beta_basis_values", self.beta_basis_values.as_slice()),
            ("beta_basis_first", self.beta_basis_first.as_slice()),
            ("beta_outputs", self.beta_outputs.as_ref()),
        ] {
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "SAE row-jet {label}[{index}] must be finite; got {value}"
                ));
            }
        }
        Ok(())
    }
}

fn checked_product(factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| format!("SAE row-jet shape product overflow for {factors:?}"))
    })
}

/// Flattened complete channels for a same-shape row tile.
#[derive(Debug, Clone, PartialEq)]
pub struct SaeRowJetChannels {
    pub n_rows: usize,
    pub n_atoms: usize,
    pub q: usize,
    pub p: usize,
    pub n_beta: usize,
    pub first: Vec<f64>,
    pub second: Vec<f64>,
    pub beta: Vec<f64>,
    /// The mathematically identical `beta_deriv` and `beta_l_deriv` channel is
    /// stored once and expanded only when constructing `SaeScheduledRowJets`.
    pub beta_mixed: Vec<f64>,
}

impl SaeRowJetChannels {
    fn zeros(
        n_rows: usize,
        n_atoms: usize,
        q: usize,
        p: usize,
        n_beta: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            n_rows,
            n_atoms,
            q,
            p,
            n_beta,
            first: vec![0.0; checked_product(&[n_rows, q, p])?],
            second: vec![0.0; checked_product(&[n_rows, q, q, p])?],
            beta: vec![0.0; checked_product(&[n_rows, n_beta, p])?],
            beta_mixed: vec![0.0; checked_product(&[n_rows, q, n_beta, p])?],
        })
    }

    pub(crate) fn into_scheduled_rows(self) -> Vec<SaeScheduledRowJets> {
        let mut rows = Vec::with_capacity(self.n_rows);
        let first_row_len = self.q * self.p;
        let second_row_len = self.q * self.q * self.p;
        let beta_row_len = self.n_beta * self.p;
        let mixed_row_len = self.q * self.n_beta * self.p;
        for row in 0..self.n_rows {
            let mut scheduled = SaeScheduledRowJets::zeros(self.q, self.p, self.n_beta);
            for slot in 0..self.q {
                let source = row * first_row_len + slot * self.p;
                scheduled
                    .first_mut(slot)
                    .copy_from_slice(&self.first[source..source + self.p]);
                for other in 0..self.q {
                    let source = row * second_row_len + (slot * self.q + other) * self.p;
                    scheduled
                        .second_mut(slot, other)
                        .copy_from_slice(&self.second[source..source + self.p]);
                }
            }
            for border in 0..self.n_beta {
                let source = row * beta_row_len + border * self.p;
                scheduled
                    .beta_mut(border)
                    .copy_from_slice(&self.beta[source..source + self.p]);
                for slot in 0..self.q {
                    let source = row * mixed_row_len + (slot * self.n_beta + border) * self.p;
                    let values = &self.beta_mixed[source..source + self.p];
                    scheduled
                        .beta_deriv_mut(slot, border)
                        .copy_from_slice(values);
                    scheduled
                        .beta_l_deriv_mut(slot, border)
                        .copy_from_slice(values);
                }
            }
            rows.push(scheduled);
        }
        rows
    }
}

/// One resident contraction against the complete row-jet tower (#2304).
///
/// The A10 trace on the elementwise tile proved the tower itself is cheap
/// (9.9 ms of kernels) while materializing and downloading the packed
/// channels is not (646 ms of copies, 98.5% of copy+kernel time): the sole
/// production consumers immediately reduce every channel against per-row
/// vectors. These contraction shapes keep the tower on device: the kernels
/// compute the same channel formulas but reduce them in place, so the only
/// downloads are the `n·q` t-outputs and `n·n_beta` β-outputs and the only
/// row-varying upload beyond the semantic inputs is one `n·p` probe vector
/// (plus the small direction coefficients for the bilinear form).
#[derive(Debug, Clone, Copy)]
pub enum SaeRowJetContraction<'a> {
    /// `t[r][a] = ⟨first(r,a,·), probe_r⟩`, `beta[r][c] = ⟨beta(r,c,·), probe_r⟩`.
    ///
    /// This is the IFT right-hand-side shape: the consumer's whitened metric
    /// (when present) folds into the probe as `probe = U(Uᵀv)` because the
    /// consumer dot is `⟨Uᵀ jet, Uᵀ v⟩ = ⟨jet, U Uᵀ v⟩` exactly.
    Linear {
        /// Row-major `n_rows × p`.
        probe: &'a [f64],
    },
    /// The exact-Hessian HVP residual-curvature shape:
    ///
    /// `t[r][a]    = Σ_b ⟨probe_r, second(r,a,b,·)⟩ v_t[r][b]
    ///             + Σ_c ⟨probe_r, mixed(r,a,c,·)⟩ v_beta[r][c]`
    /// `beta[r][c] = Σ_a ⟨probe_r, mixed(r,a,c,·)⟩ v_t[r][a]`
    ///
    /// with `probe` the (metric-applied) per-row residual.
    Bilinear {
        /// Row-major `n_rows × p`.
        probe: &'a [f64],
        /// Row-major `n_rows × q`.
        v_t: &'a [f64],
        /// Row-major `n_rows × n_beta`.
        v_beta: &'a [f64],
    },
    /// The `Γ = tr(H⁻¹ ∂H/∂θ)` log-det θ-adjoint shape (#2304).
    ///
    /// For every t-direction `w` and every β-direction `w_β` the consumer
    /// (`logdet_theta_adjoint_for_block`) reduces the derivative matrix of the
    /// tower against the per-row selected inverse. With
    /// `dh_w[a][b] = ⟨second(a,w),first(b)⟩ + ⟨first(a),second(b,w)⟩` (and the
    /// softmax data-weight-product substitution below for logit `w` over
    /// coordinate pairs) the outputs are
    ///
    /// `t[r][w]    = Σ_{a,b} E_tt[r][a][b] dh_w[a][b]
    ///             + Σ_a Σ_c 2·inv_vβ[r][a][c] (⟨second(a,w),β(c)⟩ + ⟨first(a),βderiv(w,c)⟩)
    ///             + Σ_{i,j} βinv[i][j] (⟨βderiv(w,i),β(j)⟩ + ⟨β(i),βderiv(w,j)⟩)`,
    /// `β[r][c_w] = Σ_{a,b} E_tt[r][a][b] (⟨βderiv(a,c_w),first(b)⟩ + ⟨first(a),βderiv(b,c_w)⟩)
    ///             + Σ_a Σ_c 2·inv_vβ[r][a][c] ⟨βderiv(a,c_w),β(c)⟩`.
    ///
    /// `E_tt` is the row-local t–t weight with the Daleckii–Krein deflation
    /// correction already folded in (`E = U(W⊙F)Uᵀ`, `W = Uᵀ inv_vv U`); it
    /// equals `inv_vv` when the row carries no deflation. The t–β and β–β
    /// blocks are never deflated, so they contract the raw `inv_vβ` / `βinv`.
    /// The purely scalar diagonal channels (softmax majorizer, ARD, ordered
    /// Beta–Bernoulli prior, empirical-mass column pass) are NOT part of this
    /// tower reduction; the caller adds them as an `E_tt`-weighted host
    /// post-fold, which is exact because `tr(E·(dh+dh_scalar))` is linear.
    Trace {
        /// Row-major `n_rows × q × q` deflation-folded t–t weight `E_tt`.
        e_tt: &'a [f64],
        /// Row-major `n_rows × q × n_beta` t–β selected inverse, restricted to
        /// this tile's border channels.
        inv_vbeta: &'a [f64],
        /// Row-major `n_beta × n_beta` β–β selected inverse, shared across the
        /// same-shape tile (the border layout is identical for every row).
        beta_inv: &'a [f64],
    },
}

/// Reduced outputs of one contracted tile: per-row t and β coefficients only.
#[derive(Debug, Clone, PartialEq)]
pub struct SaeRowJetContractedTile {
    pub n_rows: usize,
    pub q: usize,
    pub n_beta: usize,
    /// Row-major `n_rows × q`.
    pub t: Vec<f64>,
    /// Row-major `n_rows × n_beta`.
    pub beta: Vec<f64>,
}

impl<'a> SaeRowJetContraction<'a> {
    fn validate(&self, n: usize, q: usize, p: usize, n_beta: usize) -> Result<(), String> {
        let expect = |label: &str, got: usize, want: usize| -> Result<(), String> {
            if got != want {
                return Err(format!(
                    "SAE row-jet contraction {label} length {got} != expected {want}"
                ));
            }
            Ok(())
        };
        match *self {
            SaeRowJetContraction::Linear { probe } => {
                expect("probe", probe.len(), checked_product(&[n, p])?)?;
                finite_or_err("probe", probe)
            }
            SaeRowJetContraction::Bilinear { probe, v_t, v_beta } => {
                expect("probe", probe.len(), checked_product(&[n, p])?)?;
                expect("v_t", v_t.len(), checked_product(&[n, q])?)?;
                expect("v_beta", v_beta.len(), checked_product(&[n, n_beta])?)?;
                finite_or_err("probe", probe)?;
                finite_or_err("v_t", v_t)?;
                finite_or_err("v_beta", v_beta)
            }
            SaeRowJetContraction::Trace {
                e_tt,
                inv_vbeta,
                beta_inv,
            } => {
                expect("e_tt", e_tt.len(), checked_product(&[n, q, q])?)?;
                expect(
                    "inv_vbeta",
                    inv_vbeta.len(),
                    checked_product(&[n, q, n_beta])?,
                )?;
                expect(
                    "beta_inv",
                    beta_inv.len(),
                    checked_product(&[n_beta, n_beta])?,
                )?;
                finite_or_err("e_tt", e_tt)?;
                finite_or_err("inv_vbeta", inv_vbeta)?;
                finite_or_err("beta_inv", beta_inv)
            }
        }
    }
}

fn finite_or_err(label: &str, values: &[f64]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "SAE row-jet contraction {label}[{index}] must be finite; got {value}"
        ));
    }
    Ok(())
}

/// Evaluate one already-planned bounded tile REDUCED against a contraction,
/// never materializing the packed channel tensors on the host. Device failures
/// propagate; the CPU path evaluates the identical reduction against the
/// authoritative row program one row at a time.
pub fn execute_softmax_row_jet_tile_contracted(
    rows: &[SaeSoftmaxRowJetInput],
    inv_tau: f64,
    path: SaeRowJetPath,
    contraction: SaeRowJetContraction<'_>,
) -> Result<SaeRowJetContractedTile, String> {
    if !inv_tau.is_finite() || inv_tau <= 0.0 {
        return Err(format!(
            "SAE row-jet inverse temperature must be finite and positive; got {inv_tau}"
        ));
    }
    let (_k, q, p, n_beta) = validate_tile(rows)?;
    contraction.validate(rows.len(), q, p, n_beta)?;
    if rows.is_empty() {
        return Ok(SaeRowJetContractedTile {
            n_rows: 0,
            q: 0,
            n_beta: 0,
            t: Vec::new(),
            beta: Vec::new(),
        });
    }
    match path {
        SaeRowJetPath::Cpu => cpu_contracted_tile(rows, inv_tau, q, p, n_beta, contraction),
        SaeRowJetPath::Device => {
            #[cfg(target_os = "linux")]
            {
                device::device_contracted_tile(rows, inv_tau, contraction)
                    .map_err(|error| error.to_string())
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err("contracted SAE row-jet device tile requested on a non-Linux host".to_string())
            }
        }
    }
}

/// CPU reduction: run the authoritative row program per row and apply the SAME
/// dot/accumulation order the production consumers use on scheduled rows. Only
/// one row's channels are live at a time — the tile's packed tensors are never
/// materialized.
fn cpu_contracted_tile(
    rows: &[SaeSoftmaxRowJetInput],
    inv_tau: f64,
    q: usize,
    p: usize,
    n_beta: usize,
    contraction: SaeRowJetContraction<'_>,
) -> Result<SaeRowJetContractedTile, String> {
    let n = rows.len();
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum() };
    let mut t = vec![0.0_f64; checked_product(&[n, q])?];
    let mut beta = vec![0.0_f64; checked_product(&[n, n_beta])?];
    for (row, input) in rows.iter().enumerate() {
        let source = InputSource::new(input);
        let scheduled = execute_softmax_row_program(&source, inv_tau, input.sqrt_row_weight);
        source.finish()?;
        match contraction {
            SaeRowJetContraction::Linear { probe } => {
                let probe_row = &probe[row * p..(row + 1) * p];
                for slot in 0..q {
                    t[row * q + slot] = dot(scheduled.first(slot), probe_row);
                }
                for border in 0..n_beta {
                    beta[row * n_beta + border] = dot(scheduled.beta(border), probe_row);
                }
            }
            SaeRowJetContraction::Bilinear { probe, v_t, v_beta } => {
                let probe_row = &probe[row * p..(row + 1) * p];
                let v_t_row = &v_t[row * q..(row + 1) * q];
                let v_beta_row = &v_beta[row * n_beta..(row + 1) * n_beta];
                for a in 0..q {
                    let mut acc = 0.0_f64;
                    for b in 0..q {
                        acc += dot(probe_row, scheduled.second(a, b)) * v_t_row[b];
                    }
                    for border in 0..n_beta {
                        acc += dot(probe_row, scheduled.beta_deriv(a, border)) * v_beta_row[border];
                    }
                    t[row * q + a] = acc;
                }
                for border in 0..n_beta {
                    let mut acc = 0.0_f64;
                    for a in 0..q {
                        acc += dot(probe_row, scheduled.beta_deriv(a, border)) * v_t_row[a];
                    }
                    beta[row * n_beta + border] = acc;
                }
            }
            SaeRowJetContraction::Trace {
                e_tt,
                inv_vbeta,
                beta_inv,
            } => {
                let e_row = &e_tt[row * q * q..(row + 1) * q * q];
                let vbeta_row = &inv_vbeta[row * q * n_beta..(row + 1) * q * n_beta];
                // One t-adjoint direction `w` (a live logit or coordinate slot).
                for w in 0..q {
                    let (w_is_logit, atom_w) = primary_kind_atom(input.primaries[w]);
                    let mut gamma = 0.0_f64;
                    for a in 0..q {
                        let (a_is_logit, atom_a) = primary_kind_atom(input.primaries[a]);
                        for b in 0..q {
                            let (b_is_logit, atom_b) = primary_kind_atom(input.primaries[b]);
                            // Under softmax a logit `w` differentiates the
                            // coordinate-pair data curvature `⟨J_a,J_b⟩` through
                            // the assignment weights, not through second jets.
                            let dh = if w_is_logit && !a_is_logit && !b_is_logit {
                                dot(scheduled.first(a), scheduled.first(b))
                                    * softmax_data_weight_product_logit_factor(
                                        &input.gate_values,
                                        atom_a,
                                        atom_b,
                                        atom_w,
                                        inv_tau,
                                    )
                            } else {
                                dot(scheduled.second(a, w), scheduled.first(b))
                                    + dot(scheduled.first(a), scheduled.second(b, w))
                            };
                            gamma += e_row[a * q + b] * dh;
                        }
                        for border in 0..n_beta {
                            let dh = dot(scheduled.second(a, w), scheduled.beta(border))
                                + dot(scheduled.first(a), scheduled.beta_deriv(w, border));
                            gamma += 2.0 * vbeta_row[a * n_beta + border] * dh;
                        }
                    }
                    for i in 0..n_beta {
                        for j in 0..n_beta {
                            let dh = dot(scheduled.beta_deriv(w, i), scheduled.beta(j))
                                + dot(scheduled.beta(i), scheduled.beta_deriv(w, j));
                            gamma += beta_inv[i * n_beta + j] * dh;
                        }
                    }
                    t[row * q + w] = gamma;
                }
                // One β-adjoint direction `w_beta`.
                for w_beta in 0..n_beta {
                    let mut gamma = 0.0_f64;
                    for a in 0..q {
                        for b in 0..q {
                            let dh = dot(scheduled.beta_l_deriv(a, w_beta), scheduled.first(b))
                                + dot(scheduled.first(a), scheduled.beta_l_deriv(b, w_beta));
                            gamma += e_row[a * q + b] * dh;
                        }
                        for border in 0..n_beta {
                            let dh = dot(scheduled.beta_l_deriv(a, w_beta), scheduled.beta(border));
                            gamma += 2.0 * vbeta_row[a * n_beta + border] * dh;
                        }
                    }
                    beta[row * n_beta + w_beta] = gamma;
                }
            }
        }
    }
    Ok(SaeRowJetContractedTile {
        n_rows: n,
        q,
        n_beta,
        t,
        beta,
    })
}

/// Exact byte accounting for one same-shape row-jet tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeRowJetMemoryLedger {
    pub fixed_device_bytes: usize,
    pub device_bytes_per_row: usize,
    pub fixed_host_bytes: usize,
    pub cpu_host_bytes_per_row: usize,
    pub host_bytes_per_row: usize,
}

impl SaeRowJetMemoryLedger {
    pub fn for_shape(k: usize, q: usize, p: usize, n_beta: usize) -> Result<Self, String> {
        let f64_bytes = std::mem::size_of::<f64>();
        let i32_bytes = std::mem::size_of::<i32>();
        let f64_count = |shape: &[usize]| -> Result<usize, String> {
            checked_product(shape)?
                .checked_mul(f64_bytes)
                .ok_or_else(|| format!("SAE row-jet f64 byte count overflow for {shape:?}"))
        };
        let i32_count = |shape: &[usize]| -> Result<usize, String> {
            checked_product(shape)?
                .checked_mul(i32_bytes)
                .ok_or_else(|| format!("SAE row-jet i32 byte count overflow for {shape:?}"))
        };

        let mut fixed_device_bytes = f64_count(&[n_beta, p])?
            .checked_add(i32_count(&[n_beta])?)
            .ok_or_else(|| "SAE row-jet fixed device-byte sum overflow".to_string())?;
        let mut empty_handle_bytes = 0usize;
        let mut add_empty_handle = |bytes: usize| -> Result<(), String> {
            empty_handle_bytes = empty_handle_bytes
                .checked_add(bytes)
                .ok_or_else(|| "SAE row-jet empty-handle byte sum overflow".to_string())?;
            Ok(())
        };
        if q == 0 {
            add_empty_handle(i32_count(&[2])?)?; // primary kind + atom
            add_empty_handle(f64_count(&[4])?)?; // d1 + d2 + first + second
        }
        if n_beta == 0 {
            add_empty_handle(i32_count(&[1])?)?; // beta atom
            add_empty_handle(f64_count(&[3])?)?; // beta phi + output + beta result
        }
        if q == 0 || n_beta == 0 {
            add_empty_handle(f64_count(&[2])?)?; // beta first + mixed result
        }
        fixed_device_bytes = fixed_device_bytes
            .checked_add(empty_handle_bytes)
            .ok_or_else(|| "SAE row-jet fixed empty-handle byte sum overflow".to_string())?;
        let input_f64 = [
            f64_count(&[k])?,
            f64_count(&[1])?,
            f64_count(&[k, p])?,
            f64_count(&[q, p])?,
            f64_count(&[q, q, p])?,
            f64_count(&[n_beta])?,
            f64_count(&[q, n_beta])?,
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
        .ok_or_else(|| "SAE row-jet input-byte sum overflow".to_string())?;
        let input_i32 = [i32_count(&[k])?, i32_count(&[q])?, i32_count(&[q])?]
            .into_iter()
            .try_fold(0usize, |sum, value| sum.checked_add(value))
            .ok_or_else(|| "SAE row-jet descriptor-byte sum overflow".to_string())?;
        let output = [
            f64_count(&[q, p])?,
            f64_count(&[q, q, p])?,
            f64_count(&[n_beta, p])?,
            f64_count(&[q, n_beta, p])?,
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
        .ok_or_else(|| "SAE row-jet output-byte sum overflow".to_string())?;
        let device_bytes_per_row = input_f64
            .checked_add(input_i32)
            .and_then(|value| value.checked_add(output))
            .ok_or_else(|| "SAE row-jet device row-byte sum overflow".to_string())?;

        // Host residency has two phases. During a launch, the authoritative
        // semantic row snapshots coexist with contiguous CUDA staging and the
        // allocated download vectors. During production expansion, those row
        // snapshots and flat downloads coexist with `SaeScheduledRowJets` (whose
        // two mixed arrays intentionally duplicate the single wire channel).
        let active_bool_bytes = k
            .checked_mul(std::mem::size_of::<bool>())
            .ok_or_else(|| "SAE row-jet active-mask byte count overflow".to_string())?;
        let primary_bytes = q
            .checked_mul(std::mem::size_of::<SaeRowJetPrimary>())
            .ok_or_else(|| "SAE row-jet primary byte count overflow".to_string())?;
        // The shape planner runs before row snapshots exist, so charge the
        // canonical slot table at its exact worst-case length q. Logit entries
        // make the realized table smaller, never larger than this bound.
        let coordinate_slot_bytes = q
            .checked_mul(std::mem::size_of::<SaeCoordinateSlot>())
            .ok_or_else(|| "SAE row-jet coordinate-slot byte count overflow".to_string())?;
        let beta_atom_bytes = n_beta
            .checked_mul(std::mem::size_of::<usize>())
            .ok_or_else(|| "SAE row-jet beta-atom byte count overflow".to_string())?;
        let semantic_input = [
            input_f64,
            f64_count(&[n_beta, p])?,
            active_bool_bytes,
            primary_bytes,
            coordinate_slot_bytes,
            beta_atom_bytes,
            std::mem::size_of::<SaeSoftmaxRowJetInput>(),
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
        .ok_or_else(|| "SAE row-jet semantic-input byte sum overflow".to_string())?;
        let production_layout = q
            .checked_mul(std::mem::size_of::<SaeRowJetPrimary>())
            .and_then(|value| value.checked_add(std::mem::size_of::<Vec<SaeRowJetPrimary>>()))
            .ok_or_else(|| "SAE row-jet production-layout byte sum overflow".to_string())?;
        let staging_input = input_f64
            .checked_add(input_i32)
            .ok_or_else(|| "SAE row-jet staging-input byte sum overflow".to_string())?;
        let scheduled_output = output
            .checked_add(f64_count(&[q, n_beta, p])?)
            .and_then(|value| value.checked_add(std::mem::size_of::<SaeScheduledRowJets>()))
            .ok_or_else(|| "SAE row-jet scheduled-output byte sum overflow".to_string())?;
        let shared_host = semantic_input
            .checked_add(production_layout)
            .and_then(|value| value.checked_add(output))
            .ok_or_else(|| "SAE row-jet shared host byte sum overflow".to_string())?;
        let cpu_host_bytes_per_row = shared_host
            .checked_add(scheduled_output)
            .ok_or_else(|| "SAE row-jet CPU host row-byte sum overflow".to_string())?;
        let host_bytes_per_row = shared_host
            .checked_add(staging_input.max(scheduled_output))
            .ok_or_else(|| "SAE row-jet host row-byte sum overflow".to_string())?;
        let fixed_host_bytes = [
            f64_count(&[k])?,
            f64_count(&[n_beta, p])?,
            n_beta
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or_else(|| "SAE row-jet shared beta-atom byte overflow".to_string())?,
            i32_count(&[n_beta])?,
            std::mem::size_of::<SaeRowJetChannels>(),
            4 * std::mem::size_of::<Vec<()>>(),
        ]
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
        .ok_or_else(|| "SAE row-jet fixed host byte sum overflow".to_string())?;
        Ok(Self {
            fixed_device_bytes,
            device_bytes_per_row,
            fixed_host_bytes,
            cpu_host_bytes_per_row,
            host_bytes_per_row,
        })
    }

    /// Byte accounting for a REDUCED (contracted) same-shape tile (#2304).
    ///
    /// The contracted seam never materializes the packed channel tower on
    /// either wire. Relative to [`Self::for_shape`] the `q²·p` second channel
    /// and the `q·n_beta·p` mixed channel leave the host download and the
    /// device output entirely; the device instead keeps the phase-one dot
    /// tables (`k + q + n_beta + q²`) and the reduced outputs (`q + n_beta`)
    /// resident, and the only row-varying host↔device traffic is the `p` probe
    /// upload plus the `q + n_beta` reduced scalars downloaded. This models the
    /// `Linear`/`Bilinear` contracted tiles — the live contracted callers.
    pub fn for_contracted_shape(
        k: usize,
        q: usize,
        p: usize,
        n_beta: usize,
    ) -> Result<Self, String> {
        let f64_bytes = std::mem::size_of::<f64>();
        let i32_bytes = std::mem::size_of::<i32>();
        let f64_count = |shape: &[usize]| -> Result<usize, String> {
            checked_product(shape)?
                .checked_mul(f64_bytes)
                .ok_or_else(|| format!("SAE row-jet f64 byte count overflow for {shape:?}"))
        };
        let i32_count = |shape: &[usize]| -> Result<usize, String> {
            checked_product(shape)?
                .checked_mul(i32_bytes)
                .ok_or_else(|| format!("SAE row-jet i32 byte count overflow for {shape:?}"))
        };
        let sum = |terms: &[usize], label: &str| -> Result<usize, String> {
            terms
                .iter()
                .copied()
                .try_fold(0usize, |acc, value| acc.checked_add(value))
                .ok_or_else(|| format!("SAE row-jet contracted {label} byte sum overflow"))
        };

        let elementwise = Self::for_shape(k, q, p, n_beta)?;
        // Semantic device inputs staged per row (same tensors the elementwise
        // tile uploads) plus the reduced phase-one/phase-two device state.
        let input_f64 = sum(
            &[
                f64_count(&[k])?,
                f64_count(&[1])?,
                f64_count(&[k, p])?,
                f64_count(&[q, p])?,
                f64_count(&[q, q, p])?,
                f64_count(&[n_beta])?,
                f64_count(&[q, n_beta])?,
            ],
            "input-f64",
        )?;
        let input_i32 = sum(
            &[i32_count(&[k])?, i32_count(&[q])?, i32_count(&[q])?],
            "input-i32",
        )?;
        let staging_input = input_f64
            .checked_add(input_i32)
            .ok_or_else(|| "SAE row-jet contracted staging-input byte sum overflow".to_string())?;
        let device_reduced = sum(
            &[
                f64_count(&[k])?,      // decoded·probe dots
                f64_count(&[q])?,      // d1·probe dots
                f64_count(&[n_beta])?, // beta_output·probe dots
                f64_count(&[q, q])?,   // d2·probe dots (bilinear)
                f64_count(&[q])?,      // reduced t output
                f64_count(&[n_beta])?, // reduced beta output
                f64_count(&[p])?,      // probe upload
                f64_count(&[q])?,      // v_t upload (bilinear)
                f64_count(&[n_beta])?, // v_beta upload (bilinear)
            ],
            "device-reduced",
        )?;
        let device_bytes_per_row = staging_input
            .checked_add(device_reduced)
            .ok_or_else(|| "SAE row-jet contracted device row-byte sum overflow".to_string())?;

        // Host residency drops the packed-channel download and its
        // `SaeScheduledRowJets` expansion; only the reduced scalars accumulate
        // per row, and one row's probe/direction operands are live at a time.
        // `shared_host` is the semantic snapshot + production layout the
        // contracted tile still holds, WITHOUT the `output` tower.
        let semantic_input = sum(
            &[
                input_f64,
                f64_count(&[n_beta, p])?,
                k.checked_mul(std::mem::size_of::<bool>())
                    .ok_or_else(|| "SAE row-jet contracted active-mask overflow".to_string())?,
                q.checked_mul(std::mem::size_of::<SaeRowJetPrimary>())
                    .ok_or_else(|| "SAE row-jet contracted primary overflow".to_string())?,
                q.checked_mul(std::mem::size_of::<SaeCoordinateSlot>())
                    .ok_or_else(|| "SAE row-jet contracted coordinate-slot overflow".to_string())?,
                n_beta
                    .checked_mul(std::mem::size_of::<usize>())
                    .ok_or_else(|| "SAE row-jet contracted beta-atom overflow".to_string())?,
                std::mem::size_of::<SaeSoftmaxRowJetInput>(),
            ],
            "semantic-input",
        )?;
        let production_layout = q
            .checked_mul(std::mem::size_of::<SaeRowJetPrimary>())
            .and_then(|value| value.checked_add(std::mem::size_of::<Vec<SaeRowJetPrimary>>()))
            .ok_or_else(|| "SAE row-jet contracted production-layout overflow".to_string())?;
        let shared_host = semantic_input
            .checked_add(production_layout)
            .ok_or_else(|| "SAE row-jet contracted shared-host byte sum overflow".to_string())?;
        let reduced_operands = sum(
            &[
                f64_count(&[q])?,
                f64_count(&[n_beta])?,
                f64_count(&[p])?,
                f64_count(&[q])?,
                f64_count(&[n_beta])?,
            ],
            "reduced-operands",
        )?;
        let cpu_host_bytes_per_row = shared_host
            .checked_add(reduced_operands)
            .ok_or_else(|| "SAE row-jet contracted CPU host row-byte sum overflow".to_string())?;
        let host_bytes_per_row = shared_host
            .checked_add(reduced_operands)
            .and_then(|value| value.checked_add(staging_input.saturating_sub(reduced_operands)))
            .ok_or_else(|| "SAE row-jet contracted host row-byte sum overflow".to_string())?;
        // The CPU path materializes exactly one packed row at a time; charge it
        // once as a fixed cost rather than per tile row.
        let one_scheduled_row = sum(
            &[
                f64_count(&[q, p])?,
                f64_count(&[q, q, p])?,
                f64_count(&[n_beta, p])?,
                f64_count(&[q, n_beta, p])?,
                std::mem::size_of::<SaeScheduledRowJets>(),
            ],
            "one-scheduled-row",
        )?;
        let fixed_host_bytes = elementwise
            .fixed_host_bytes
            .checked_add(one_scheduled_row)
            .ok_or_else(|| "SAE row-jet contracted fixed host byte sum overflow".to_string())?;
        Ok(Self {
            fixed_device_bytes: elementwise.fixed_device_bytes,
            device_bytes_per_row,
            fixed_host_bytes,
            cpu_host_bytes_per_row,
            host_bytes_per_row,
        })
    }

    fn maximum_rows(self, device_budget: usize, host_budget: usize) -> usize {
        let device_rows = device_budget
            .checked_sub(self.fixed_device_bytes)
            .map_or(0, |bytes| bytes / self.device_bytes_per_row.max(1));
        let host_rows = host_budget
            .checked_sub(self.fixed_host_bytes)
            .map_or(0, |bytes| bytes / self.host_bytes_per_row.max(1));
        device_rows.min(host_rows)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeRowJetPath {
    Device,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaeRowJetExecutionPlan {
    pub path: SaeRowJetPath,
    pub tile_rows: usize,
    pub ledger: SaeRowJetMemoryLedger,
}

/// Decide the backend and bounded tile width before any CUDA launch.
pub fn plan_softmax_row_jets(
    total_rows: usize,
    k: usize,
    q: usize,
    p: usize,
    n_beta: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<SaeRowJetExecutionPlan, String> {
    if k == 0 || p == 0 {
        return Err(format!(
            "complete SAE row-jet plan requires nonzero K and p; got K={k}, p={p}"
        ));
    }
    let ledger = SaeRowJetMemoryLedger::for_shape(k, q, p, n_beta)?;
    plan_dispatch(total_rows, k, q, p, n_beta, mode, ledger)
}

/// Decide the backend and bounded tile width for a REDUCED (contracted) tile.
///
/// The contracted ledger charges the smaller reduced download plus probe
/// upload instead of the full packed tower, so a same-shape tile fits more
/// rows per host/device budget than the elementwise plan admits. Used by the
/// resident-contraction consumers (`contracted_softmax_linear_rhs`,
/// `contracted_softmax_bilinear_hvp`).
pub fn plan_softmax_row_jets_contracted(
    total_rows: usize,
    k: usize,
    q: usize,
    p: usize,
    n_beta: usize,
    mode: gam_gpu::GpuPolicy,
) -> Result<SaeRowJetExecutionPlan, String> {
    if k == 0 || p == 0 {
        return Err(format!(
            "contracted SAE row-jet plan requires nonzero K and p; got K={k}, p={p}"
        ));
    }
    let ledger = SaeRowJetMemoryLedger::for_contracted_shape(k, q, p, n_beta)?;
    plan_dispatch(total_rows, k, q, p, n_beta, mode, ledger)
}

/// Shared backend/tile-width dispatch for an already-computed shape ledger.
fn plan_dispatch(
    total_rows: usize,
    k: usize,
    q: usize,
    p: usize,
    n_beta: usize,
    mode: gam_gpu::GpuPolicy,
    ledger: SaeRowJetMemoryLedger,
) -> Result<SaeRowJetExecutionPlan, String> {
    if total_rows == 0 {
        return Ok(SaeRowJetExecutionPlan {
            path: SaeRowJetPath::Cpu,
            tile_rows: 0,
            ledger,
        });
    }
    let host_budget = crate::manifold::sae_host_in_core_budget_bytes().0;
    let one_row_host_bytes = ledger
        .fixed_host_bytes
        .checked_add(ledger.cpu_host_bytes_per_row)
        .ok_or_else(|| "complete SAE one-row host-byte sum overflow".to_string())?;
    if one_row_host_bytes > host_budget {
        let second_channel_bytes = checked_product(&[q, q, p])?
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| "complete SAE q^2 p channel-byte sum overflow".to_string())?;
        return Err(format!(
            "complete SAE row jet refuses one K={k}, q={q}, p={p}, beta={n_beta} row: the q^2*p second channel alone is {second_channel_bytes} bytes and simultaneous host residency is {one_row_host_bytes} bytes, exceeding the cgroup-aware budget {host_budget}"
        ));
    }
    if mode == gam_gpu::GpuPolicy::Off {
        return Ok(SaeRowJetExecutionPlan {
            path: SaeRowJetPath::Cpu,
            tile_rows: 1,
            ledger,
        });
    }

    // This lower bound is derived from the smallest runtime calibration point.
    // It lets ordinary CPU-sized fits avoid creating a CUDA context at all.
    if mode == gam_gpu::GpuPolicy::Auto
        && total_rows < gam_gpu::policy::GpuDispatchPolicy::MIN_CALIBRATABLE_ROW_KERNEL_N
    {
        return Ok(SaeRowJetExecutionPlan {
            path: SaeRowJetPath::Cpu,
            tile_rows: 1,
            ledger,
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        if mode == gam_gpu::GpuPolicy::Required {
            return Err(
                "complete SAE row jet requires CUDA, which is unavailable on this platform"
                    .to_string(),
            );
        }
        Ok(SaeRowJetExecutionPlan {
            path: SaeRowJetPath::Cpu,
            tile_rows: 1,
            ledger,
        })
    }

    #[cfg(target_os = "linux")]
    {
        let runtime = if mode == gam_gpu::GpuPolicy::Required {
            gam_gpu::device_runtime::GpuRuntime::require()
                .map_err(|error| format!("complete SAE row jet requires CUDA: {error}"))?
        } else {
            let Some(runtime) = gam_gpu::device_runtime::GpuRuntime::resolve(mode)
                .map_err(|error| format!("complete SAE row-jet CUDA admission failed: {error}"))?
            else {
                return Ok(SaeRowJetExecutionPlan {
                    path: SaeRowJetPath::Cpu,
                    tile_rows: 1,
                    ledger,
                });
            };
            runtime
        };
        if mode == gam_gpu::GpuPolicy::Auto && total_rows < runtime.policy.row_kernel_min_n {
            return Ok(SaeRowJetExecutionPlan {
                path: SaeRowJetPath::Cpu,
                tile_rows: 1,
                ledger,
            });
        }
        let tile_rows = ledger.maximum_rows(runtime.memory_budget_bytes, host_budget);
        if tile_rows == 0 {
            if mode == gam_gpu::GpuPolicy::Required {
                return Err(format!(
                    "complete SAE row jet cannot fit one tile: fixed_device={} device_per_row={} host_per_row={} device_budget={} host_budget={}",
                    ledger.fixed_device_bytes,
                    ledger.device_bytes_per_row,
                    ledger.host_bytes_per_row,
                    runtime.memory_budget_bytes,
                    host_budget
                ));
            }
            return Ok(SaeRowJetExecutionPlan {
                path: SaeRowJetPath::Cpu,
                tile_rows: 1,
                ledger,
            });
        }
        Ok(SaeRowJetExecutionPlan {
            path: SaeRowJetPath::Device,
            tile_rows: tile_rows.min(total_rows),
            ledger,
        })
    }
}

fn validate_tile(rows: &[SaeSoftmaxRowJetInput]) -> Result<(usize, usize, usize, usize), String> {
    let Some(first) = rows.first() else {
        return Ok((0, 0, 0, 0));
    };
    first.validate()?;
    let shape = (
        first.n_atoms,
        first.n_primaries(),
        first.out_dim,
        first.n_beta_borders(),
    );
    for (row, input) in rows.iter().enumerate().skip(1) {
        input.validate()?;
        let candidate = (
            input.n_atoms,
            input.n_primaries(),
            input.out_dim,
            input.n_beta_borders(),
        );
        if candidate != shape {
            return Err(format!(
                "SAE row-jet tile row {row} shape {candidate:?} != first-row shape {shape:?}"
            ));
        }
        if input.beta_outputs != first.beta_outputs {
            return Err(format!(
                "SAE row-jet tile row {row} has a different decoder-border output frame"
            ));
        }
        if input.beta_atoms != first.beta_atoms {
            return Err(format!(
                "SAE row-jet tile row {row} has a different decoder-border atom layout"
            ));
        }
    }
    Ok(shape)
}

/// Evaluate one already-planned bounded tile. Device failures propagate.
pub fn execute_softmax_row_jet_tile(
    rows: &[SaeSoftmaxRowJetInput],
    inv_tau: f64,
    path: SaeRowJetPath,
) -> Result<SaeRowJetChannels, String> {
    if !inv_tau.is_finite() || inv_tau <= 0.0 {
        return Err(format!(
            "SAE row-jet inverse temperature must be finite and positive; got {inv_tau}"
        ));
    }
    let (k, q, p, n_beta) = validate_tile(rows)?;
    if rows.is_empty() {
        return SaeRowJetChannels::zeros(0, 0, 0, 0, 0);
    }
    match path {
        SaeRowJetPath::Cpu => cpu_tile(rows, inv_tau, k, q, p, n_beta),
        SaeRowJetPath::Device => {
            #[cfg(target_os = "linux")]
            {
                device::device_tile(rows, inv_tau, k, q, p, n_beta)
                    .map_err(|error| error.to_string())
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err("complete SAE row-jet device tile requested on a non-Linux host".to_string())
            }
        }
    }
}

struct InputSource<'a> {
    input: &'a SaeSoftmaxRowJetInput,
    structural_error: std::cell::RefCell<Option<String>>,
}

impl<'a> InputSource<'a> {
    fn new(input: &'a SaeSoftmaxRowJetInput) -> Self {
        Self {
            input,
            structural_error: std::cell::RefCell::new(None),
        }
    }

    fn coordinate_slot(&self, atom: usize, axis: usize, context: &'static str) -> Option<usize> {
        let slot = self
            .input
            .coordinate_slots
            .binary_search_by_key(&(atom, axis), |entry| (entry.atom, entry.axis))
            .ok()
            .map(|index| self.input.coordinate_slots[index].slot);
        if slot.is_none() {
            let mut error = self.structural_error.borrow_mut();
            if error.is_none() {
                *error = Some(format!(
                    "{context} requested absent SAE coordinate ({atom}, {axis})"
                ));
            }
        }
        slot
    }

    fn finish(&self) -> Result<(), String> {
        match self.structural_error.borrow_mut().take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl SaeSoftmaxRowProgramSource for InputSource<'_> {
    fn n_atoms(&self) -> usize {
        self.input.n_atoms
    }
    fn out_dim(&self) -> usize {
        self.input.out_dim
    }
    fn n_primaries(&self) -> usize {
        self.input.n_primaries()
    }
    fn primary(&self, slot: usize) -> SaeRowPrimary {
        scheduled_primary(self.input.primaries[slot])
    }
    fn gate_value(&self, atom: usize) -> f64 {
        self.input.gate_values[atom]
    }
    fn atom_is_active(&self, atom: usize) -> bool {
        self.input.active_atoms[atom]
    }
    fn fill_decoded(&self, atom: usize, out: &mut [f64]) {
        let p = self.input.out_dim;
        out.copy_from_slice(&self.input.decoded[atom * p..(atom + 1) * p]);
    }
    fn fill_decoded_first(&self, atom: usize, axis: usize, out: &mut [f64]) {
        let p = self.input.out_dim;
        let Some(slot) = self.coordinate_slot(atom, axis, "decoded first channel") else {
            out.fill(0.0);
            return;
        };
        out.copy_from_slice(&self.input.decoded_first[slot * p..(slot + 1) * p]);
    }
    fn fill_decoded_second(&self, atom: usize, axis_a: usize, axis_b: usize, out: &mut [f64]) {
        let p = self.input.out_dim;
        let q = self.input.n_primaries();
        let Some(slot_a) = self.coordinate_slot(atom, axis_a, "decoded second left channel") else {
            out.fill(0.0);
            return;
        };
        let Some(slot_b) = self.coordinate_slot(atom, axis_b, "decoded second right channel")
        else {
            out.fill(0.0);
            return;
        };
        let start = (slot_a * q + slot_b) * p;
        out.copy_from_slice(&self.input.decoded_second[start..start + p]);
    }
    fn n_beta_borders(&self) -> usize {
        self.input.n_beta_borders()
    }
    fn beta_border_atom(&self, border: usize) -> usize {
        self.input.beta_atoms[border]
    }
    fn beta_border_basis_value(&self, border: usize) -> f64 {
        self.input.beta_basis_values[border]
    }
    fn beta_border_basis_first(&self, border: usize, axis: usize) -> f64 {
        let atom = self.input.beta_atoms[border];
        self.coordinate_slot(atom, axis, "beta-border first channel")
            .map_or(0.0, |slot| {
                self.input.beta_basis_first[slot * self.input.n_beta_borders() + border]
            })
    }
    fn beta_border_output(&self, border: usize) -> &[f64] {
        let p = self.input.out_dim;
        &self.input.beta_outputs[border * p..(border + 1) * p]
    }
}

fn cpu_tile(
    rows: &[SaeSoftmaxRowJetInput],
    inv_tau: f64,
    k: usize,
    q: usize,
    p: usize,
    n_beta: usize,
) -> Result<SaeRowJetChannels, String> {
    let mut out = SaeRowJetChannels::zeros(rows.len(), k, q, p, n_beta)?;
    for (row, input) in rows.iter().enumerate() {
        let source = InputSource::new(input);
        let scheduled = execute_softmax_row_program(&source, inv_tau, input.sqrt_row_weight);
        source.finish()?;
        for slot in 0..q {
            let target = row * q * p + slot * p;
            out.first[target..target + p].copy_from_slice(scheduled.first(slot));
            for other in 0..q {
                let target = row * q * q * p + (slot * q + other) * p;
                out.second[target..target + p].copy_from_slice(scheduled.second(slot, other));
            }
        }
        for border in 0..n_beta {
            let target = row * n_beta * p + border * p;
            out.beta[target..target + p].copy_from_slice(scheduled.beta(border));
            for slot in 0..q {
                let target = row * q * n_beta * p + (slot * n_beta + border) * p;
                let mixed = scheduled.beta_deriv(slot, border);
                if mixed != scheduled.beta_l_deriv(slot, border) {
                    return Err(format!(
                        "SAE CPU row program produced unequal beta mixed channels at row={row}, slot={slot}, border={border}"
                    ));
                }
                out.beta_mixed[target..target + p].copy_from_slice(mixed);
            }
        }
    }
    Ok(out)
}

/// Direct centered-moment CUDA implementation. No per-primary jet arrays are
/// materialized; each thread writes one packed output element.
pub const COMPLETE_SOFTMAX_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void sae_rowjet_first(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* decoded, const double* d1, const double* sqrt_w,
    double inv_tau, int k, int q, int p, unsigned long long total,
    double* first)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int c=(int)(index%(unsigned long long)p);
  unsigned long long rem=index/(unsigned long long)p;
  int slot=(int)(rem%(unsigned long long)q);
  int row=(int)(rem/(unsigned long long)q);
  int a=atom[row*q+slot];
  double root=sqrt_w[row];
  if(kind[row*q+slot]==0){
    double mean=0.0;
    for(int component=0;component<k;++component){
      if(active[row*k+component]) mean += z[row*k+component]*decoded[(row*k+component)*p+c];
    }
    double component=active[row*k+a] ? decoded[(row*k+a)*p+c] : 0.0;
    double centered=component-mean;
    double coefficient=root*(inv_tau*z[row*k+a]);
    first[index]=coefficient*centered;
  }else{
    if(!active[row*k+a]) { first[index]=0.0; return; }
    double coefficient=z[row*k+a]*root;
    first[index]=coefficient*d1[(row*q+slot)*p+c];
  }
}

extern "C" __global__ void sae_rowjet_second(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* decoded, const double* d1, const double* d2,
    const double* sqrt_w, double inv_tau, int k, int q, int p,
    unsigned long long total, double* second)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int c=(int)(index%(unsigned long long)p);
  unsigned long long rem=index/(unsigned long long)p;
  int slot_b=(int)(rem%(unsigned long long)q); rem/=(unsigned long long)q;
  int slot_a=(int)(rem%(unsigned long long)q);
  int row=(int)(rem/(unsigned long long)q);
  int ka=kind[row*q+slot_a], kb=kind[row*q+slot_b];
  int aa=atom[row*q+slot_a], ab=atom[row*q+slot_b];
  double root=sqrt_w[row];
  if(ka==0 && kb==0){
    double mean=0.0;
    for(int component=0;component<k;++component){
      if(active[row*k+component]) mean += z[row*k+component]*decoded[(row*k+component)*p+c];
    }
    double component_a=active[row*k+aa] ? decoded[(row*k+aa)*p+c] : 0.0;
    double component_b=active[row*k+ab] ? decoded[(row*k+ab)*p+c] : 0.0;
    double centered_a=component_a-mean;
    double centered_b=component_b-mean;
    double za=z[row*k+aa], zb=z[row*k+ab];
    double diagonal=aa==ab ? 1.0 : 0.0;
    double common=inv_tau*inv_tau*za;
    double coefficient_a=root*(common*(diagonal-zb));
    double coefficient_b=root*(-common*zb);
    second[index]=coefficient_a*centered_a+coefficient_b*centered_b;
  }else if(ka==0 || kb==0){
    int logit_atom=ka==0 ? aa : ab;
    int coord_atom=ka==0 ? ab : aa;
    int coord_slot=ka==0 ? slot_b : slot_a;
    if(!active[row*k+coord_atom]) { second[index]=0.0; return; }
    double diagonal=coord_atom==logit_atom ? 1.0 : 0.0;
    double coefficient=z[row*k+coord_atom]*(diagonal-z[row*k+logit_atom])*inv_tau;
    coefficient*=root;
    second[index]=coefficient*d1[(row*q+coord_slot)*p+c];
  }else if(aa==ab){
    if(!active[row*k+aa]) { second[index]=0.0; return; }
    double coefficient=z[row*k+aa]*root;
    second[index]=coefficient*d2[((row*q+slot_a)*q+slot_b)*p+c];
  }else{
    second[index]=0.0;
  }
}

extern "C" __global__ void sae_rowjet_beta(
    const double* z, const int* active, const int* beta_atom,
    const double* beta_phi, const double* beta_output, const double* sqrt_w,
    int k, int p, int nb, unsigned long long total, double* beta)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int c=(int)(index%(unsigned long long)p);
  unsigned long long rem=index/(unsigned long long)p;
  int border=(int)(rem%(unsigned long long)nb);
  int row=(int)(rem/(unsigned long long)nb);
  int a=beta_atom[border];
  if(!active[row*k+a]) { beta[index]=0.0; return; }
  double base=z[row*k+a]*beta_phi[row*nb+border];
  base*=sqrt_w[row];
  beta[index]=base*beta_output[border*p+c];
}

extern "C" __global__ void sae_rowjet_beta_mixed(
    const double* z, const int* active, const int* kind, const int* atom,
    const int* beta_atom, const double* beta_phi, const double* beta_first,
    const double* beta_output, const double* sqrt_w, double inv_tau,
    int k, int q, int p, int nb, unsigned long long total, double* mixed)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int c=(int)(index%(unsigned long long)p);
  unsigned long long rem=index/(unsigned long long)p;
  int border=(int)(rem%(unsigned long long)nb); rem/=(unsigned long long)nb;
  int slot=(int)(rem%(unsigned long long)q);
  int row=(int)(rem/(unsigned long long)q);
  int target=beta_atom[border];
  if(!active[row*k+target]) { mixed[index]=0.0; return; }
  int source_atom=atom[row*q+slot];
  double scalar;
  if(kind[row*q+slot]==0){
    double diagonal=target==source_atom ? 1.0 : 0.0;
    scalar=z[row*k+target]*(diagonal-z[row*k+source_atom])*inv_tau;
    scalar*=beta_phi[row*nb+border];
  }else if(source_atom==target){
    scalar=z[row*k+target]*beta_first[(row*q+slot)*nb+border];
  }else{
    scalar=0.0;
  }
  scalar*=sqrt_w[row];
  mixed[index]=scalar*beta_output[border*p+c];
}

// ---- resident contraction kernels (#2304) ----
//
// The tower is reduced on device instead of materialized: phase one computes
// per-row dot tables of the semantic input tensors against one probe vector,
// phase two assembles the contracted t/beta outputs from the SAME scalar
// coefficient formulas the elementwise kernels above use. Both phases are
// gated against the authoritative CPU row program, so the formulas cannot
// drift apart silently.

// out[row*m + j] = sum_c mat[(row*m + j)*p + c] * x[row*p + c]
extern "C" __global__ void sae_rowjet_dot_rowmat(
    const double* mat, const double* x, int m, int p,
    unsigned long long total, double* out)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int row=(int)(index/(unsigned long long)m);
  double acc=0.0;
  for(int c=0;c<p;++c) acc += mat[index*(unsigned long long)p+c]*x[row*p+c];
  out[index]=acc;
}

// out[row*m + j] = sum_c mat[j*p + c] * x[row*p + c]   (row-shared matrix)
extern "C" __global__ void sae_rowjet_dot_shared(
    const double* mat, const double* x, int m, int p,
    unsigned long long total, double* out)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int j=(int)(index%(unsigned long long)m);
  int row=(int)(index/(unsigned long long)m);
  double acc=0.0;
  for(int c=0;c<p;++c) acc += mat[j*p+c]*x[row*p+c];
  out[index]=acc;
}

// t[row*q + slot] = <first(row,slot,.), probe_row> via the dot tables.
extern "C" __global__ void sae_rowjet_contract_linear_t(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* dd, const double* d1dot, const double* sqrt_w,
    double inv_tau, int k, int q, unsigned long long total, double* t_out)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int slot=(int)(index%(unsigned long long)q);
  int row=(int)(index/(unsigned long long)q);
  int a=atom[row*q+slot];
  double root=sqrt_w[row];
  if(kind[row*q+slot]==0){
    double mean_dot=0.0;
    for(int component=0;component<k;++component){
      if(active[row*k+component]) mean_dot += z[row*k+component]*dd[row*k+component];
    }
    double component=active[row*k+a] ? dd[row*k+a] : 0.0;
    t_out[index]=root*(inv_tau*z[row*k+a])*(component-mean_dot);
  }else{
    t_out[index]=active[row*k+a] ? z[row*k+a]*root*d1dot[row*q+slot] : 0.0;
  }
}

// beta[row*nb + border] = <beta(row,border,.), probe_row> via the shared dot.
extern "C" __global__ void sae_rowjet_contract_linear_beta(
    const double* z, const int* active, const int* beta_atom,
    const double* beta_phi, const double* bodot, const double* sqrt_w,
    int k, int nb, unsigned long long total, double* beta_out)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int border=(int)(index%(unsigned long long)nb);
  int row=(int)(index/(unsigned long long)nb);
  int a=beta_atom[border];
  if(!active[row*k+a]) { beta_out[index]=0.0; return; }
  beta_out[index]=sqrt_w[row]*z[row*k+a]*beta_phi[row*nb+border]*bodot[row*nb+border];
}

// t[row*q + a] = sum_b <probe, second(a,b)> v_t[b] + sum_c <probe, mixed(a,c)> v_beta[c]
extern "C" __global__ void sae_rowjet_contract_bilinear_t(
    const double* z, const int* active, const int* kind, const int* atom,
    const int* beta_atom, const double* beta_phi, const double* beta_first,
    const double* dd, const double* d1dot, const double* d2dot,
    const double* bodot, const double* v_t, const double* v_beta,
    const double* sqrt_w, double inv_tau, int k, int q, int nb,
    unsigned long long total, double* t_out)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int slot_a=(int)(index%(unsigned long long)q);
  int row=(int)(index/(unsigned long long)q);
  int ka=kind[row*q+slot_a];
  int aa=atom[row*q+slot_a];
  double root=sqrt_w[row];
  double mean_dot=0.0;
  for(int component=0;component<k;++component){
    if(active[row*k+component]) mean_dot += z[row*k+component]*dd[row*k+component];
  }
  double acc=0.0;
  for(int slot_b=0;slot_b<q;++slot_b){
    int kb=kind[row*q+slot_b];
    int ab=atom[row*q+slot_b];
    double sdot;
    if(ka==0 && kb==0){
      double component_a=active[row*k+aa] ? dd[row*k+aa] : 0.0;
      double component_b=active[row*k+ab] ? dd[row*k+ab] : 0.0;
      double centered_a=component_a-mean_dot;
      double centered_b=component_b-mean_dot;
      double za=z[row*k+aa], zb=z[row*k+ab];
      double diagonal=aa==ab ? 1.0 : 0.0;
      double common=inv_tau*inv_tau*za;
      sdot=root*(common*(diagonal-zb))*centered_a+root*(-common*zb)*centered_b;
    }else if(ka==0 || kb==0){
      int logit_atom=ka==0 ? aa : ab;
      int coord_atom=ka==0 ? ab : aa;
      int coord_slot=ka==0 ? slot_b : slot_a;
      if(active[row*k+coord_atom]){
        double diagonal=coord_atom==logit_atom ? 1.0 : 0.0;
        double coefficient=z[row*k+coord_atom]*(diagonal-z[row*k+logit_atom])*inv_tau;
        sdot=root*coefficient*d1dot[row*q+coord_slot];
      }else{
        sdot=0.0;
      }
    }else if(aa==ab){
      sdot=active[row*k+aa] ? root*z[row*k+aa]*d2dot[(row*q+slot_a)*q+slot_b] : 0.0;
    }else{
      sdot=0.0;
    }
    acc += sdot*v_t[row*q+slot_b];
  }
  for(int border=0;border<nb;++border){
    int target=beta_atom[border];
    double scalar=0.0;
    if(active[row*k+target]){
      if(ka==0){
        double diagonal=target==aa ? 1.0 : 0.0;
        scalar=z[row*k+target]*(diagonal-z[row*k+aa])*inv_tau*beta_phi[row*nb+border];
      }else if(aa==target){
        scalar=z[row*k+target]*beta_first[(row*q+slot_a)*nb+border];
      }
      scalar*=root;
    }
    acc += scalar*bodot[row*nb+border]*v_beta[row*nb+border];
  }
  t_out[index]=acc;
}

// beta[row*nb + border] = sum_a <probe, mixed(a,border)> v_t[a]
extern "C" __global__ void sae_rowjet_contract_bilinear_beta(
    const double* z, const int* active, const int* kind, const int* atom,
    const int* beta_atom, const double* beta_phi, const double* beta_first,
    const double* bodot, const double* v_t, const double* sqrt_w,
    double inv_tau, int k, int q, int nb, unsigned long long total,
    double* beta_out)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int border=(int)(index%(unsigned long long)nb);
  int row=(int)(index/(unsigned long long)nb);
  int target=beta_atom[border];
  if(!active[row*k+target]) { beta_out[index]=0.0; return; }
  double root=sqrt_w[row];
  double acc=0.0;
  for(int slot=0;slot<q;++slot){
    int source_atom=atom[row*q+slot];
    double scalar;
    if(kind[row*q+slot]==0){
      double diagonal=target==source_atom ? 1.0 : 0.0;
      scalar=z[row*k+target]*(diagonal-z[row*k+source_atom])*inv_tau;
      scalar*=beta_phi[row*nb+border];
    }else if(source_atom==target){
      scalar=z[row*k+target]*beta_first[(row*q+slot)*nb+border];
    }else{
      scalar=0.0;
    }
    acc += root*scalar*bodot[row*nb+border]*v_t[row*q+slot];
  }
  beta_out[index]=acc;
}
"#;

#[cfg(target_os = "linux")]
mod device {
    use super::{
        COMPLETE_SOFTMAX_KERNEL_SOURCE, SaeRowJetChannels, SaeRowJetContractedTile,
        SaeRowJetContraction, SaeRowJetPrimary, SaeSoftmaxRowJetInput, checked_product,
    };
    use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use std::borrow::Cow;
    use std::sync::{Arc, OnceLock};

    struct Backend {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("sae_rowjet_complete")?;
                let ptx = gam_gpu::device_cache::compile_ptx_arch(COMPLETE_SOFTMAX_KERNEL_SOURCE)
                    .gpu_ctx("complete SAE row-jet NVRTC compile")?;
                let module = parts
                    .ctx
                    .load_module(ptx)
                    .gpu_ctx("complete SAE row-jet module load")?;
                Ok(Backend {
                    stream: parts.stream,
                    module,
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    fn grid(total: usize) -> Result<LaunchConfig, GpuError> {
        // Eight warps expose enough independent fp64 outputs per block without
        // consuming shared memory or constraining register occupancy.
        const THREADS: u32 = 8 * 32;
        let total_u64 = u64::try_from(total)
            .map_err(|_| gam_gpu::gpu_err!("complete SAE row-jet output length overflow"))?;
        let blocks = total_u64.div_ceil(u64::from(THREADS));
        let blocks = u32::try_from(blocks)
            .map_err(|_| gam_gpu::gpu_err!("complete SAE row-jet grid overflow"))?;
        Ok(LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    fn device_length(shape: &[usize]) -> Result<usize, GpuError> {
        checked_product(shape).map_err(|reason| gam_gpu::gpu_err!("{reason}"))
    }

    fn nonempty_f64(values: &[f64]) -> Cow<'_, [f64]> {
        if values.is_empty() {
            Cow::Owned(vec![0.0])
        } else {
            Cow::Borrowed(values)
        }
    }

    fn nonempty_i32(values: &[i32]) -> Cow<'_, [i32]> {
        if values.is_empty() {
            Cow::Owned(vec![0])
        } else {
            Cow::Borrowed(values)
        }
    }

    /// Uploaded semantic inputs shared by the elementwise and contracted tiles.
    struct Staged {
        z_dev: CudaSlice<f64>,
        active_dev: CudaSlice<i32>,
        kind_dev: CudaSlice<i32>,
        atom_dev: CudaSlice<i32>,
        decoded_dev: CudaSlice<f64>,
        d1_dev: CudaSlice<f64>,
        d2_dev: CudaSlice<f64>,
        sqrt_w_dev: CudaSlice<f64>,
        beta_atom_dev: CudaSlice<i32>,
        beta_phi_dev: CudaSlice<f64>,
        beta_first_dev: CudaSlice<f64>,
        beta_output_dev: CudaSlice<f64>,
        k_i32: i32,
        q_i32: i32,
        p_i32: i32,
        nb_i32: i32,
    }

    fn stage_inputs(
        stream: &Arc<CudaStream>,
        rows: &[SaeSoftmaxRowJetInput],
        k: usize,
        q: usize,
        p: usize,
        n_beta: usize,
    ) -> Result<Staged, GpuError> {
        let n = rows.len();
        let mut z = Vec::with_capacity(device_length(&[n, k])?);
        let mut active = Vec::with_capacity(device_length(&[n, k])?);
        let mut kind = Vec::with_capacity(device_length(&[n, q])?);
        let mut atom = Vec::with_capacity(device_length(&[n, q])?);
        let mut decoded = Vec::with_capacity(device_length(&[n, k, p])?);
        let mut d1 = Vec::with_capacity(device_length(&[n, q, p])?);
        let mut d2 = Vec::with_capacity(device_length(&[n, q, q, p])?);
        let mut sqrt_w = Vec::with_capacity(n);
        let mut beta_phi = Vec::with_capacity(device_length(&[n, n_beta])?);
        let mut beta_first = Vec::with_capacity(device_length(&[n, q, n_beta])?);
        for input in rows {
            z.extend_from_slice(&input.gate_values);
            active.extend(input.active_atoms.iter().map(|&value| i32::from(value)));
            for primary in &input.primaries {
                match *primary {
                    SaeRowJetPrimary::Logit { atom: source } => {
                        kind.push(0_i32);
                        atom.push(i32::try_from(source).map_err(|_| {
                            gam_gpu::gpu_err!("complete SAE row-jet atom index overflows i32")
                        })?);
                    }
                    SaeRowJetPrimary::Coordinate { atom: source, .. } => {
                        kind.push(1_i32);
                        atom.push(i32::try_from(source).map_err(|_| {
                            gam_gpu::gpu_err!("complete SAE row-jet atom index overflows i32")
                        })?);
                    }
                }
            }
            decoded.extend_from_slice(&input.decoded);
            d1.extend_from_slice(&input.decoded_first);
            d2.extend_from_slice(&input.decoded_second);
            sqrt_w.push(input.sqrt_row_weight);
            beta_phi.extend_from_slice(&input.beta_basis_values);
            beta_first.extend_from_slice(&input.beta_basis_first);
        }
        let beta_atom: Vec<i32> = rows[0]
            .beta_atoms
            .iter()
            .map(|&source| {
                i32::try_from(source).map_err(|_| {
                    gam_gpu::gpu_err!("complete SAE row-jet beta atom index overflows i32")
                })
            })
            .collect::<Result<_, _>>()?;

        // cudarc does not guarantee zero-length allocations. Dummy cells are
        // never read because the corresponding kernel has total=0 and is not launched.
        let z_dev = stream.clone_htod(&z).gpu_ctx("SAE row-jet htod gates")?;
        let active_dev = stream
            .clone_htod(&active)
            .gpu_ctx("SAE row-jet htod active mask")?;
        let kind_dev = stream
            .clone_htod(nonempty_i32(&kind).as_ref())
            .gpu_ctx("SAE row-jet htod primary kinds")?;
        let atom_dev = stream
            .clone_htod(nonempty_i32(&atom).as_ref())
            .gpu_ctx("SAE row-jet htod primary atoms")?;
        let decoded_dev = stream
            .clone_htod(&decoded)
            .gpu_ctx("SAE row-jet htod decoded")?;
        let d1_dev = stream
            .clone_htod(nonempty_f64(&d1).as_ref())
            .gpu_ctx("SAE row-jet htod decoded first")?;
        let d2_dev = stream
            .clone_htod(nonempty_f64(&d2).as_ref())
            .gpu_ctx("SAE row-jet htod decoded second")?;
        let sqrt_w_dev = stream
            .clone_htod(&sqrt_w)
            .gpu_ctx("SAE row-jet htod row weights")?;
        let beta_atom_dev = stream
            .clone_htod(nonempty_i32(&beta_atom).as_ref())
            .gpu_ctx("SAE row-jet htod beta atoms")?;
        let beta_phi_dev = stream
            .clone_htod(nonempty_f64(&beta_phi).as_ref())
            .gpu_ctx("SAE row-jet htod beta basis values")?;
        let beta_first_dev = stream
            .clone_htod(nonempty_f64(&beta_first).as_ref())
            .gpu_ctx("SAE row-jet htod beta basis first")?;
        let beta_output_dev = stream
            .clone_htod(nonempty_f64(rows[0].beta_outputs.as_ref()).as_ref())
            .gpu_ctx("SAE row-jet htod beta outputs")?;
        let k_i32 =
            i32::try_from(k).map_err(|_| gam_gpu::gpu_err!("SAE row-jet K overflows i32"))?;
        let q_i32 =
            i32::try_from(q).map_err(|_| gam_gpu::gpu_err!("SAE row-jet q overflows i32"))?;
        let p_i32 =
            i32::try_from(p).map_err(|_| gam_gpu::gpu_err!("SAE row-jet p overflows i32"))?;
        let nb_i32 = i32::try_from(n_beta)
            .map_err(|_| gam_gpu::gpu_err!("SAE row-jet beta count overflows i32"))?;
        Ok(Staged {
            z_dev,
            active_dev,
            kind_dev,
            atom_dev,
            decoded_dev,
            d1_dev,
            d2_dev,
            sqrt_w_dev,
            beta_atom_dev,
            beta_phi_dev,
            beta_first_dev,
            beta_output_dev,
            k_i32,
            q_i32,
            p_i32,
            nb_i32,
        })
    }

    pub(super) fn device_tile(
        rows: &[SaeSoftmaxRowJetInput],
        inv_tau: f64,
        k: usize,
        q: usize,
        p: usize,
        n_beta: usize,
    ) -> Result<SaeRowJetChannels, GpuError> {
        let n = rows.len();
        let b = backend()?;
        let stream = b.stream.clone();
        let Staged {
            z_dev,
            active_dev,
            kind_dev,
            atom_dev,
            decoded_dev,
            d1_dev,
            d2_dev,
            sqrt_w_dev,
            beta_atom_dev,
            beta_phi_dev,
            beta_first_dev,
            beta_output_dev,
            k_i32,
            q_i32,
            p_i32,
            nb_i32,
        } = stage_inputs(&stream, rows, k, q, p, n_beta)?;

        let first_len = device_length(&[n, q, p])?;
        let second_len = device_length(&[n, q, q, p])?;
        let beta_len = device_length(&[n, n_beta, p])?;
        let mixed_len = device_length(&[n, q, n_beta, p])?;
        let mut first_dev = stream
            .alloc_zeros::<f64>(first_len.max(1))
            .gpu_ctx("SAE row-jet alloc first")?;
        let mut second_dev = stream
            .alloc_zeros::<f64>(second_len.max(1))
            .gpu_ctx("SAE row-jet alloc second")?;
        let mut beta_dev = stream
            .alloc_zeros::<f64>(beta_len.max(1))
            .gpu_ctx("SAE row-jet alloc beta")?;
        let mut mixed_dev = stream
            .alloc_zeros::<f64>(mixed_len.max(1))
            .gpu_ctx("SAE row-jet alloc beta mixed")?;

        if first_len != 0 {
            let function = b
                .module
                .load_function("sae_rowjet_first")
                .gpu_ctx("SAE row-jet first load")?;
            let total = u64::try_from(first_len)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet first length overflows u64"))?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&z_dev)
                .arg(&active_dev)
                .arg(&kind_dev)
                .arg(&atom_dev)
                .arg(&decoded_dev)
                .arg(&d1_dev)
                .arg(&sqrt_w_dev)
                .arg(&inv_tau)
                .arg(&k_i32)
                .arg(&q_i32)
                .arg(&p_i32)
                .arg(&total)
                .arg(&mut first_dev);
            // SAFETY: the loaded kernel's argument ABI matches this builder, and
            // `grid(first_len)` covers only the `first_len` allocated outputs.
            unsafe { launch.launch(grid(first_len)?) }.gpu_ctx("SAE row-jet first launch")?;
        }
        if second_len != 0 {
            let function = b
                .module
                .load_function("sae_rowjet_second")
                .gpu_ctx("SAE row-jet second load")?;
            let total = u64::try_from(second_len)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet second length overflows u64"))?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&z_dev)
                .arg(&active_dev)
                .arg(&kind_dev)
                .arg(&atom_dev)
                .arg(&decoded_dev)
                .arg(&d1_dev)
                .arg(&d2_dev)
                .arg(&sqrt_w_dev)
                .arg(&inv_tau)
                .arg(&k_i32)
                .arg(&q_i32)
                .arg(&p_i32)
                .arg(&total)
                .arg(&mut second_dev);
            // SAFETY: the loaded kernel's argument ABI matches this builder, and
            // `grid(second_len)` covers only the `second_len` allocated outputs.
            unsafe { launch.launch(grid(second_len)?) }.gpu_ctx("SAE row-jet second launch")?;
        }
        if beta_len != 0 {
            let function = b
                .module
                .load_function("sae_rowjet_beta")
                .gpu_ctx("SAE row-jet beta load")?;
            let total = u64::try_from(beta_len)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet beta length overflows u64"))?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&z_dev)
                .arg(&active_dev)
                .arg(&beta_atom_dev)
                .arg(&beta_phi_dev)
                .arg(&beta_output_dev)
                .arg(&sqrt_w_dev)
                .arg(&k_i32)
                .arg(&p_i32)
                .arg(&nb_i32)
                .arg(&total)
                .arg(&mut beta_dev);
            // SAFETY: the loaded kernel's argument ABI matches this builder, and
            // `grid(beta_len)` covers only the `beta_len` allocated outputs.
            unsafe { launch.launch(grid(beta_len)?) }.gpu_ctx("SAE row-jet beta launch")?;
        }
        if mixed_len != 0 {
            let function = b
                .module
                .load_function("sae_rowjet_beta_mixed")
                .gpu_ctx("SAE row-jet beta mixed load")?;
            let total = u64::try_from(mixed_len)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet mixed length overflows u64"))?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&z_dev)
                .arg(&active_dev)
                .arg(&kind_dev)
                .arg(&atom_dev)
                .arg(&beta_atom_dev)
                .arg(&beta_phi_dev)
                .arg(&beta_first_dev)
                .arg(&beta_output_dev)
                .arg(&sqrt_w_dev)
                .arg(&inv_tau)
                .arg(&k_i32)
                .arg(&q_i32)
                .arg(&p_i32)
                .arg(&nb_i32)
                .arg(&total)
                .arg(&mut mixed_dev);
            // SAFETY: the loaded kernel's argument ABI matches this builder, and
            // `grid(mixed_len)` covers only the `mixed_len` allocated outputs.
            unsafe { launch.launch(grid(mixed_len)?) }.gpu_ctx("SAE row-jet beta mixed launch")?;
        }

        let mut out = SaeRowJetChannels::zeros(n, k, q, p, n_beta)
            .map_err(|reason| gam_gpu::gpu_err!("{reason}"))?;
        if first_len != 0 {
            stream
                .memcpy_dtoh(&first_dev, &mut out.first)
                .gpu_ctx("SAE row-jet dtoh first")?;
        }
        if second_len != 0 {
            stream
                .memcpy_dtoh(&second_dev, &mut out.second)
                .gpu_ctx("SAE row-jet dtoh second")?;
        }
        if beta_len != 0 {
            stream
                .memcpy_dtoh(&beta_dev, &mut out.beta)
                .gpu_ctx("SAE row-jet dtoh beta")?;
        }
        if mixed_len != 0 {
            stream
                .memcpy_dtoh(&mixed_dev, &mut out.beta_mixed)
                .gpu_ctx("SAE row-jet dtoh beta mixed")?;
        }
        stream.synchronize().gpu_ctx("SAE row-jet synchronize")?;
        Ok(out)
    }

    /// Resident contracted tile (#2304): the tower is reduced on device and
    /// only the `n·q` / `n·n_beta` coefficients come back to the host. The
    /// packed `q²p`-class channel tensors are never allocated on either side.
    pub(super) fn device_contracted_tile(
        rows: &[SaeSoftmaxRowJetInput],
        inv_tau: f64,
        contraction: SaeRowJetContraction<'_>,
    ) -> Result<SaeRowJetContractedTile, GpuError> {
        let n = rows.len();
        let first = &rows[0];
        let (k, q, p, n_beta) = (
            first.n_atoms,
            first.n_primaries(),
            first.out_dim,
            first.n_beta_borders(),
        );
        let b = backend()?;
        let stream = b.stream.clone();
        let staged = stage_inputs(&stream, rows, k, q, p, n_beta)?;

        // Narrow the public contraction to the two shapes that have a device
        // kernel. The θ-adjoint trace reduction has no single-probe device
        // kernel yet; its resident form (per-row `E_tt`/`inv_vβ` weighted
        // second-jet reduction) is the hardware-gated follow-on, and the CPU
        // arm is the authoritative oracle. Refusing it here — and dropping it
        // from the type the assembly match below consumes — keeps that
        // impossible branch unexpressible rather than relying on a later
        // panic/`unreachable!` arm.
        enum DeviceContraction<'a> {
            Linear,
            Bilinear { v_t: &'a [f64], v_beta: &'a [f64] },
        }
        let (probe, device) = match contraction {
            SaeRowJetContraction::Linear { probe } => (probe, DeviceContraction::Linear),
            SaeRowJetContraction::Bilinear { probe, v_t, v_beta } => {
                (probe, DeviceContraction::Bilinear { v_t, v_beta })
            }
            SaeRowJetContraction::Trace { .. } => {
                return Err(gam_gpu::gpu_err!(
                    "SAE row-jet Trace contraction has no device kernel yet; plan the CPU path"
                ));
            }
        };
        let probe_dev = stream
            .clone_htod(nonempty_f64(probe).as_ref())
            .gpu_ctx("SAE row-jet htod contraction probe")?;

        // Phase one: dot tables of the semantic tensors against the probe.
        let dd_len = device_length(&[n, k])?;
        let d1dot_len = device_length(&[n, q])?;
        let bodot_len = device_length(&[n, n_beta])?;
        let mut dd_dev = stream
            .alloc_zeros::<f64>(dd_len.max(1))
            .gpu_ctx("SAE row-jet alloc decoded dots")?;
        let mut d1dot_dev = stream
            .alloc_zeros::<f64>(d1dot_len.max(1))
            .gpu_ctx("SAE row-jet alloc d1 dots")?;
        let mut bodot_dev = stream
            .alloc_zeros::<f64>(bodot_len.max(1))
            .gpu_ctx("SAE row-jet alloc beta-output dots")?;
        let dot_rowmat = b
            .module
            .load_function("sae_rowjet_dot_rowmat")
            .gpu_ctx("SAE row-jet dot rowmat load")?;
        let launch_rowmat = |mat: &CudaSlice<f64>,
                             m: usize,
                             out: &mut CudaSlice<f64>,
                             label: &'static str|
         -> Result<(), GpuError> {
            let total = device_length(&[n, m])?;
            if total == 0 {
                return Ok(());
            }
            let m_i32 = i32::try_from(m)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet dot width overflows i32"))?;
            let total_u64 = u64::try_from(total)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet dot length overflows u64"))?;
            let mut launch = stream.launch_builder(&dot_rowmat);
            launch
                .arg(mat)
                .arg(&probe_dev)
                .arg(&m_i32)
                .arg(&staged.p_i32)
                .arg(&total_u64)
                .arg(out);
            // SAFETY: the loaded kernel's argument ABI matches this builder,
            // and `grid(total)` covers only the `total` allocated outputs.
            unsafe { launch.launch(grid(total)?) }.gpu_ctx(label)?;
            Ok(())
        };
        launch_rowmat(
            &staged.decoded_dev,
            k,
            &mut dd_dev,
            "SAE row-jet decoded dots",
        )?;
        launch_rowmat(&staged.d1_dev, q, &mut d1dot_dev, "SAE row-jet d1 dots")?;
        if bodot_len != 0 {
            let function = b
                .module
                .load_function("sae_rowjet_dot_shared")
                .gpu_ctx("SAE row-jet dot shared load")?;
            let total_u64 = u64::try_from(bodot_len)
                .map_err(|_| gam_gpu::gpu_err!("SAE row-jet dot length overflows u64"))?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&staged.beta_output_dev)
                .arg(&probe_dev)
                .arg(&staged.nb_i32)
                .arg(&staged.p_i32)
                .arg(&total_u64)
                .arg(&mut bodot_dev);
            // SAFETY: the loaded kernel's argument ABI matches this builder,
            // and `grid(bodot_len)` covers only the allocated outputs.
            unsafe { launch.launch(grid(bodot_len)?) }.gpu_ctx("SAE row-jet beta-output dots")?;
        }

        // Phase two: assemble the reduced outputs.
        let t_len = device_length(&[n, q])?;
        let beta_len = device_length(&[n, n_beta])?;
        let mut t_dev = stream
            .alloc_zeros::<f64>(t_len.max(1))
            .gpu_ctx("SAE row-jet alloc contracted t")?;
        let mut beta_dev = stream
            .alloc_zeros::<f64>(beta_len.max(1))
            .gpu_ctx("SAE row-jet alloc contracted beta")?;
        match device {
            DeviceContraction::Linear => {
                if t_len != 0 {
                    let function = b
                        .module
                        .load_function("sae_rowjet_contract_linear_t")
                        .gpu_ctx("SAE row-jet linear t load")?;
                    let total_u64 = u64::try_from(t_len)
                        .map_err(|_| gam_gpu::gpu_err!("SAE row-jet t length overflows u64"))?;
                    let mut launch = stream.launch_builder(&function);
                    launch
                        .arg(&staged.z_dev)
                        .arg(&staged.active_dev)
                        .arg(&staged.kind_dev)
                        .arg(&staged.atom_dev)
                        .arg(&dd_dev)
                        .arg(&d1dot_dev)
                        .arg(&staged.sqrt_w_dev)
                        .arg(&inv_tau)
                        .arg(&staged.k_i32)
                        .arg(&staged.q_i32)
                        .arg(&total_u64)
                        .arg(&mut t_dev);
                    // SAFETY: the loaded kernel's argument ABI matches this
                    // builder, and `grid(t_len)` covers only the outputs.
                    unsafe { launch.launch(grid(t_len)?) }
                        .gpu_ctx("SAE row-jet linear t launch")?;
                }
                if beta_len != 0 {
                    let function = b
                        .module
                        .load_function("sae_rowjet_contract_linear_beta")
                        .gpu_ctx("SAE row-jet linear beta load")?;
                    let total_u64 = u64::try_from(beta_len)
                        .map_err(|_| gam_gpu::gpu_err!("SAE row-jet beta length overflows u64"))?;
                    let mut launch = stream.launch_builder(&function);
                    launch
                        .arg(&staged.z_dev)
                        .arg(&staged.active_dev)
                        .arg(&staged.beta_atom_dev)
                        .arg(&staged.beta_phi_dev)
                        .arg(&bodot_dev)
                        .arg(&staged.sqrt_w_dev)
                        .arg(&staged.k_i32)
                        .arg(&staged.nb_i32)
                        .arg(&total_u64)
                        .arg(&mut beta_dev);
                    // SAFETY: the loaded kernel's argument ABI matches this
                    // builder, and `grid(beta_len)` covers only the outputs.
                    unsafe { launch.launch(grid(beta_len)?) }
                        .gpu_ctx("SAE row-jet linear beta launch")?;
                }
            }
            DeviceContraction::Bilinear { v_t, v_beta } => {
                let d2dot_len = device_length(&[n, q, q])?;
                let mut d2dot_dev = stream
                    .alloc_zeros::<f64>(d2dot_len.max(1))
                    .gpu_ctx("SAE row-jet alloc d2 dots")?;
                launch_rowmat(
                    &staged.d2_dev,
                    checked_product(&[q, q]).map_err(|reason| gam_gpu::gpu_err!("{reason}"))?,
                    &mut d2dot_dev,
                    "SAE row-jet d2 dots",
                )?;
                let v_t_dev = stream
                    .clone_htod(nonempty_f64(v_t).as_ref())
                    .gpu_ctx("SAE row-jet htod v_t")?;
                let v_beta_dev = stream
                    .clone_htod(nonempty_f64(v_beta).as_ref())
                    .gpu_ctx("SAE row-jet htod v_beta")?;
                if t_len != 0 {
                    let function = b
                        .module
                        .load_function("sae_rowjet_contract_bilinear_t")
                        .gpu_ctx("SAE row-jet bilinear t load")?;
                    let total_u64 = u64::try_from(t_len)
                        .map_err(|_| gam_gpu::gpu_err!("SAE row-jet t length overflows u64"))?;
                    let mut launch = stream.launch_builder(&function);
                    launch
                        .arg(&staged.z_dev)
                        .arg(&staged.active_dev)
                        .arg(&staged.kind_dev)
                        .arg(&staged.atom_dev)
                        .arg(&staged.beta_atom_dev)
                        .arg(&staged.beta_phi_dev)
                        .arg(&staged.beta_first_dev)
                        .arg(&dd_dev)
                        .arg(&d1dot_dev)
                        .arg(&d2dot_dev)
                        .arg(&bodot_dev)
                        .arg(&v_t_dev)
                        .arg(&v_beta_dev)
                        .arg(&staged.sqrt_w_dev)
                        .arg(&inv_tau)
                        .arg(&staged.k_i32)
                        .arg(&staged.q_i32)
                        .arg(&staged.nb_i32)
                        .arg(&total_u64)
                        .arg(&mut t_dev);
                    // SAFETY: the loaded kernel's argument ABI matches this
                    // builder, and `grid(t_len)` covers only the outputs.
                    unsafe { launch.launch(grid(t_len)?) }
                        .gpu_ctx("SAE row-jet bilinear t launch")?;
                }
                if beta_len != 0 {
                    let function = b
                        .module
                        .load_function("sae_rowjet_contract_bilinear_beta")
                        .gpu_ctx("SAE row-jet bilinear beta load")?;
                    let total_u64 = u64::try_from(beta_len)
                        .map_err(|_| gam_gpu::gpu_err!("SAE row-jet beta length overflows u64"))?;
                    let mut launch = stream.launch_builder(&function);
                    launch
                        .arg(&staged.z_dev)
                        .arg(&staged.active_dev)
                        .arg(&staged.kind_dev)
                        .arg(&staged.atom_dev)
                        .arg(&staged.beta_atom_dev)
                        .arg(&staged.beta_phi_dev)
                        .arg(&staged.beta_first_dev)
                        .arg(&bodot_dev)
                        .arg(&v_t_dev)
                        .arg(&staged.sqrt_w_dev)
                        .arg(&inv_tau)
                        .arg(&staged.k_i32)
                        .arg(&staged.q_i32)
                        .arg(&staged.nb_i32)
                        .arg(&total_u64)
                        .arg(&mut beta_dev);
                    // SAFETY: the loaded kernel's argument ABI matches this
                    // builder, and `grid(beta_len)` covers only the outputs.
                    unsafe { launch.launch(grid(beta_len)?) }
                        .gpu_ctx("SAE row-jet bilinear beta launch")?;
                }
            }
        }

        let mut t = vec![0.0_f64; t_len];
        let mut beta = vec![0.0_f64; beta_len];
        if t_len != 0 {
            stream
                .memcpy_dtoh(&t_dev, &mut t)
                .gpu_ctx("SAE row-jet dtoh contracted t")?;
        }
        if beta_len != 0 {
            stream
                .memcpy_dtoh(&beta_dev, &mut beta)
                .gpu_ctx("SAE row-jet dtoh contracted beta")?;
        }
        stream
            .synchronize()
            .gpu_ctx("SAE row-jet contracted synchronize")?;
        Ok(SaeRowJetContractedTile {
            n_rows: n,
            q,
            n_beta,
            t,
            beta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_fixture(n: usize) -> Vec<SaeSoftmaxRowJetInput> {
        let k = 3;
        let p = 2;
        let primaries = vec![
            SaeRowJetPrimary::Logit { atom: 0 },
            SaeRowJetPrimary::Logit { atom: 1 },
            SaeRowJetPrimary::Coordinate { atom: 0, axis: 0 },
            SaeRowJetPrimary::Coordinate { atom: 1, axis: 0 },
            SaeRowJetPrimary::Coordinate { atom: 1, axis: 1 },
            SaeRowJetPrimary::Coordinate { atom: 2, axis: 0 },
        ];
        let q = primaries.len();
        (0..n)
            .map(|row| {
                let logits: Vec<f64> = (0..k)
                    .map(|atom| 0.4 * ((row * 17 + atom * 11 + 1) as f64 * 0.07).sin())
                    .collect();
                let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|value| (value - shift).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let gate_values = exps.iter().map(|value| value / sum).collect();
                let decoded: Vec<f64> = (0..k * p)
                    .map(|index| ((row * 13 + index * 7 + 3) as f64 * 0.09).cos())
                    .collect();
                let mut decoded_first = vec![0.0; q * p];
                for slot in 2..q {
                    for c in 0..p {
                        decoded_first[slot * p + c] =
                            ((row * 19 + slot * 5 + c + 2) as f64 * 0.04).sin();
                    }
                }
                let mut decoded_second = vec![0.0; q * q * p];
                for a in 2..q {
                    for b in 2..q {
                        let same_atom = match (primaries[a], primaries[b]) {
                            (
                                SaeRowJetPrimary::Coordinate { atom: left, .. },
                                SaeRowJetPrimary::Coordinate { atom: right, .. },
                            ) => left == right,
                            _ => false,
                        };
                        if same_atom {
                            for c in 0..p {
                                decoded_second[(a * q + b) * p + c] =
                                    ((row * 23 + a * 7 + b * 3 + c + 1) as f64 * 0.03).cos();
                            }
                        }
                    }
                }
                let beta_atoms = vec![0, 1, 2];
                let n_beta = beta_atoms.len();
                let beta_basis_values = vec![0.8, -0.3, 0.5];
                let mut beta_basis_first = vec![0.0; q * n_beta];
                beta_basis_first[2 * n_beta] = 0.2;
                beta_basis_first[3 * n_beta + 1] = -0.4;
                beta_basis_first[4 * n_beta + 1] = 0.7;
                beta_basis_first[5 * n_beta + 2] = -0.1;
                SaeSoftmaxRowJetInput {
                    n_atoms: k,
                    out_dim: p,
                    coordinate_slots: SaeSoftmaxRowJetInput::coordinate_slots_for(&primaries),
                    primaries: primaries.clone(),
                    gate_values,
                    active_atoms: vec![true, true, row % 2 == 0],
                    sqrt_row_weight: (1.0 + row as f64 * 0.1).sqrt(),
                    decoded,
                    decoded_first,
                    decoded_second,
                    beta_atoms: beta_atoms.into(),
                    beta_basis_values,
                    beta_basis_first,
                    beta_outputs: vec![1.0, 0.2, -0.5, 0.8, 0.3, -0.7].into(),
                }
            })
            .collect()
    }

    #[test]
    fn complete_cpu_rowjet_contains_coordinate_mixed_and_beta_channels_2304() {
        let rows = complete_fixture(2);
        let out = execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Cpu)
            .expect("complete CPU row jet");
        assert_eq!(out.q, 6);
        assert!(
            out.first[2 * out.p..3 * out.p]
                .iter()
                .any(|value| *value != 0.0)
        );
        let mixed = (0 * out.q * out.q + 0 * out.q + 2) * out.p;
        assert!(
            out.second[mixed..mixed + out.p]
                .iter()
                .any(|value| *value != 0.0)
        );
        let transpose = (0 * out.q * out.q + 2 * out.q) * out.p;
        assert_eq!(
            &out.second[mixed..mixed + out.p],
            &out.second[transpose..transpose + out.p]
        );
        assert!(out.beta.iter().any(|value| *value != 0.0));
        assert!(out.beta_mixed.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn memory_ledger_counts_coordinate_and_mixed_tensors_2304() {
        fn assert_exact_ledger(
            k: usize,
            q: usize,
            p: usize,
            n_beta: usize,
            staging_is_host_peak: bool,
        ) -> SaeRowJetMemoryLedger {
            let f64_bytes = std::mem::size_of::<f64>();
            let i32_bytes = std::mem::size_of::<i32>();

            // Reconstruct every resident allocation independently of
            // `SaeRowJetMemoryLedger::for_shape`. In particular, the complete
            // coordinate Hessian is resident on both sides of the kernel and
            // the single beta-mixed wire channel becomes two scheduled arrays.
            let gate_bytes = k * f64_bytes;
            let sqrt_weight_bytes = f64_bytes;
            let decoded_bytes = k * p * f64_bytes;
            let decoded_first_bytes = q * p * f64_bytes;
            let decoded_second_bytes = q * q * p * f64_bytes;
            let beta_basis_bytes = n_beta * f64_bytes;
            let beta_basis_first_bytes = q * n_beta * f64_bytes;
            let input_f64_bytes = gate_bytes
                + sqrt_weight_bytes
                + decoded_bytes
                + decoded_first_bytes
                + decoded_second_bytes
                + beta_basis_bytes
                + beta_basis_first_bytes;
            let input_i32_bytes = (k + q + q) * i32_bytes;

            let first_output_bytes = q * p * f64_bytes;
            let second_output_bytes = q * q * p * f64_bytes;
            let beta_output_bytes = n_beta * p * f64_bytes;
            let beta_mixed_wire_bytes = q * n_beta * p * f64_bytes;
            let flat_output_bytes = first_output_bytes
                + second_output_bytes
                + beta_output_bytes
                + beta_mixed_wire_bytes;

            assert_eq!(decoded_second_bytes, second_output_bytes);
            let expected_device_bytes_per_row =
                input_f64_bytes + input_i32_bytes + flat_output_bytes;
            let expected_fixed_device_bytes = beta_output_bytes + n_beta * i32_bytes;

            let semantic_input_bytes = input_f64_bytes
                + beta_output_bytes
                + k * std::mem::size_of::<bool>()
                + q * std::mem::size_of::<SaeRowJetPrimary>()
                + q * std::mem::size_of::<SaeCoordinateSlot>()
                + n_beta * std::mem::size_of::<usize>()
                + std::mem::size_of::<SaeSoftmaxRowJetInput>();
            let production_layout_bytes = q * std::mem::size_of::<SaeRowJetPrimary>()
                + std::mem::size_of::<Vec<SaeRowJetPrimary>>();
            let staging_input_bytes = input_f64_bytes + input_i32_bytes;
            let scheduled_output_bytes = first_output_bytes
                + second_output_bytes
                + beta_output_bytes
                + 2 * beta_mixed_wire_bytes
                + std::mem::size_of::<SaeScheduledRowJets>();
            assert_eq!(
                scheduled_output_bytes,
                flat_output_bytes
                    + beta_mixed_wire_bytes
                    + std::mem::size_of::<SaeScheduledRowJets>()
            );
            let shared_host_bytes =
                semantic_input_bytes + production_layout_bytes + flat_output_bytes;
            let expected_cpu_host_bytes_per_row = shared_host_bytes + scheduled_output_bytes;
            let expected_host_bytes_per_row =
                shared_host_bytes + staging_input_bytes.max(scheduled_output_bytes);
            let expected_fixed_host_bytes = gate_bytes
                + beta_output_bytes
                + n_beta * std::mem::size_of::<usize>()
                + n_beta * i32_bytes
                + std::mem::size_of::<SaeRowJetChannels>()
                + 4 * std::mem::size_of::<Vec<()>>();

            let expected = SaeRowJetMemoryLedger {
                fixed_device_bytes: expected_fixed_device_bytes,
                device_bytes_per_row: expected_device_bytes_per_row,
                fixed_host_bytes: expected_fixed_host_bytes,
                cpu_host_bytes_per_row: expected_cpu_host_bytes_per_row,
                host_bytes_per_row: expected_host_bytes_per_row,
            };
            let actual = SaeRowJetMemoryLedger::for_shape(k, q, p, n_beta).expect("memory ledger");
            assert_eq!(actual, expected, "shape K={k}, q={q}, p={p}, beta={n_beta}");

            if staging_is_host_peak {
                assert!(staging_input_bytes > scheduled_output_bytes);
                assert_eq!(
                    actual.host_bytes_per_row,
                    shared_host_bytes + staging_input_bytes
                );
                assert!(actual.host_bytes_per_row > actual.cpu_host_bytes_per_row);
            } else {
                assert!(scheduled_output_bytes > staging_input_bytes);
                assert_eq!(
                    actual.host_bytes_per_row,
                    shared_host_bytes + scheduled_output_bytes
                );
                assert_eq!(actual.host_bytes_per_row, actual.cpu_host_bytes_per_row);
            }
            actual
        }

        // The complete production fixture has a q^2 coordinate channel and a
        // large scheduled expansion, so production expansion is the host peak.
        let ledger = assert_exact_ledger(3, 6, 2, 3, false);
        // A wide atom input with one primary makes CUDA staging the host peak.
        assert_exact_ledger(64, 1, 2, 1, true);

        assert_eq!(
            ledger.maximum_rows(ledger.fixed_device_bytes, usize::MAX),
            0
        );
        assert_eq!(
            ledger.maximum_rows(
                ledger.fixed_device_bytes + ledger.device_bytes_per_row,
                ledger.fixed_host_bytes + ledger.host_bytes_per_row
            ),
            1
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn complete_device_matches_cpu_every_channel_when_admitted_2304() {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => return,
            Err(error) => panic!("complete row-jet CUDA admission failed: {error}"),
        }
        // A 37-row smoke fixture launches each kernel for too little time to
        // establish that the admitted production path actually occupies the
        // device. Keep a complete, memory-bounded workload large enough for
        // external utilization/profiler sampling, warm the NVRTC/module path
        // once, then require every measured pass to match the one CPU oracle.
        const ROW_COUNT: usize = 1 << 17;
        const MEASURED_PASSES: usize = 8;
        let rows = complete_fixture(ROW_COUNT);
        let cpu = execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Cpu).expect("CPU oracle");

        execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Device)
            .expect("admitted device warm-up must execute without a host retry");

        let mut max_error = 0.0_f64;
        for pass in 0..MEASURED_PASSES {
            let device = execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Device)
                .unwrap_or_else(|error| {
                    panic!("admitted device pass {pass} must execute without a host retry: {error}")
                });
            max_error = cpu
                .first
                .iter()
                .chain(&cpu.second)
                .chain(&cpu.beta)
                .chain(&cpu.beta_mixed)
                .zip(
                    device
                        .first
                        .iter()
                        .chain(&device.second)
                        .chain(&device.beta)
                        .chain(&device.beta_mixed),
                )
                .fold(max_error, |maximum, (left, right)| {
                    maximum.max((left - right).abs())
                });
        }
        let outputs_per_pass =
            cpu.first.len() + cpu.second.len() + cpu.beta.len() + cpu.beta_mixed.len();
        eprintln!(
            "SAE_ROWJET_GPU_ACCEPT rows={ROW_COUNT} measured_passes={MEASURED_PASSES} outputs_per_pass={outputs_per_pass} max_abs_error={max_error:.17e}"
        );
        assert!(
            max_error <= 1.0e-12,
            "complete SAE device/CPU row-jet error {max_error:e} exceeds 1e-12"
        );
    }

    /// Deterministic per-row contraction vectors matched to `complete_fixture`.
    fn contraction_vectors(
        n: usize,
        q: usize,
        p: usize,
        n_beta: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let probe: Vec<f64> = (0..n * p)
            .map(|index| ((index * 29 + 5) as f64 * 0.011).sin())
            .collect();
        let v_t: Vec<f64> = (0..n * q)
            .map(|index| ((index * 31 + 7) as f64 * 0.013).cos())
            .collect();
        let v_beta: Vec<f64> = (0..n * n_beta)
            .map(|index| ((index * 37 + 11) as f64 * 0.017).sin())
            .collect();
        (probe, v_t, v_beta)
    }

    /// The CPU contracted seam must equal reducing the materialized CPU
    /// channels with the production consumers' own dot/accumulation order —
    /// bit for bit, because both run the identical row program and the
    /// identical f64 reduction order.
    #[test]
    fn contracted_cpu_tile_equals_materialized_channel_reduction_2304() {
        let rows = complete_fixture(5);
        let channels = execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Cpu)
            .expect("complete CPU row jet");
        let (n, q, p, n_beta) = (channels.n_rows, channels.q, channels.p, channels.n_beta);
        let (probe, v_t, v_beta) = contraction_vectors(n, q, p, n_beta);
        let scheduled = channels.into_scheduled_rows();
        let dot =
            |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum() };

        let linear = execute_softmax_row_jet_tile_contracted(
            &rows,
            1.0,
            SaeRowJetPath::Cpu,
            SaeRowJetContraction::Linear { probe: &probe },
        )
        .expect("CPU linear contraction");
        assert_eq!((linear.n_rows, linear.q, linear.n_beta), (n, q, n_beta));
        for (row, jets) in scheduled.iter().enumerate() {
            let probe_row = &probe[row * p..(row + 1) * p];
            for slot in 0..q {
                let expected = dot(jets.first(slot), probe_row);
                assert_eq!(
                    linear.t[row * q + slot],
                    expected,
                    "linear t mismatch at row={row}, slot={slot}"
                );
            }
            for border in 0..n_beta {
                let expected = dot(jets.beta(border), probe_row);
                assert_eq!(
                    linear.beta[row * n_beta + border],
                    expected,
                    "linear beta mismatch at row={row}, border={border}"
                );
            }
        }

        let bilinear = execute_softmax_row_jet_tile_contracted(
            &rows,
            1.0,
            SaeRowJetPath::Cpu,
            SaeRowJetContraction::Bilinear {
                probe: &probe,
                v_t: &v_t,
                v_beta: &v_beta,
            },
        )
        .expect("CPU bilinear contraction");
        for (row, jets) in scheduled.iter().enumerate() {
            let probe_row = &probe[row * p..(row + 1) * p];
            let v_t_row = &v_t[row * q..(row + 1) * q];
            let v_beta_row = &v_beta[row * n_beta..(row + 1) * n_beta];
            for a in 0..q {
                let mut expected = 0.0_f64;
                for b in 0..q {
                    expected += dot(probe_row, jets.second(a, b)) * v_t_row[b];
                }
                for border in 0..n_beta {
                    expected += dot(probe_row, jets.beta_deriv(a, border)) * v_beta_row[border];
                }
                assert_eq!(
                    bilinear.t[row * q + a],
                    expected,
                    "bilinear t mismatch at row={row}, slot={a}"
                );
            }
            for border in 0..n_beta {
                let mut expected = 0.0_f64;
                for a in 0..q {
                    expected += dot(probe_row, jets.beta_deriv(a, border)) * v_t_row[a];
                }
                assert_eq!(
                    bilinear.beta[row * n_beta + border],
                    expected,
                    "bilinear beta mismatch at row={row}, border={border}"
                );
            }
        }
    }

    /// Non-trivial-value guard: on the complete fixture the contracted outputs
    /// must be generically nonzero in every block (t logit, t coordinate, and
    /// β), so the parity assertions above cannot pass on an all-zeros bug.
    #[test]
    fn contracted_cpu_tile_outputs_are_generically_nonzero_2304() {
        let rows = complete_fixture(3);
        let (n, q, p, n_beta) = (3, 6, 2, 3);
        let (probe, v_t, v_beta) = contraction_vectors(n, q, p, n_beta);
        let linear = execute_softmax_row_jet_tile_contracted(
            &rows,
            1.0,
            SaeRowJetPath::Cpu,
            SaeRowJetContraction::Linear { probe: &probe },
        )
        .expect("CPU linear contraction");
        // Slots 0-1 are logits, slots 2+ are coordinates (see complete_fixture).
        assert!(linear.t[0] != 0.0, "logit-slot contraction must engage");
        assert!(
            linear.t[2] != 0.0,
            "coordinate-slot contraction must engage"
        );
        assert!(linear.beta.iter().any(|&value| value != 0.0));
        let bilinear = execute_softmax_row_jet_tile_contracted(
            &rows,
            1.0,
            SaeRowJetPath::Cpu,
            SaeRowJetContraction::Bilinear {
                probe: &probe,
                v_t: &v_t,
                v_beta: &v_beta,
            },
        )
        .expect("CPU bilinear contraction");
        assert!(bilinear.t.iter().any(|&value| value != 0.0));
        assert!(bilinear.beta.iter().any(|&value| value != 0.0));
    }

    /// Resident-contraction acceptance (#2304): on an admitted device, both
    /// contraction shapes must reproduce the CPU reduction at the same
    /// workload scale as the elementwise gate, while transferring only the
    /// `n·q`/`n·n_beta` reduced outputs.
    #[cfg(target_os = "linux")]
    #[test]
    fn contracted_device_matches_cpu_reduction_when_admitted_2304() {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => return,
            Err(error) => panic!("contracted row-jet CUDA admission failed: {error}"),
        }
        const ROW_COUNT: usize = 1 << 17;
        const MEASURED_PASSES: usize = 4;
        let rows = complete_fixture(ROW_COUNT);
        let (q, p, n_beta) = (
            rows[0].n_primaries(),
            rows[0].out_dim,
            rows[0].n_beta_borders(),
        );
        let (probe, v_t, v_beta) = contraction_vectors(ROW_COUNT, q, p, n_beta);
        let shapes: [(&str, SaeRowJetContraction<'_>); 2] = [
            ("linear", SaeRowJetContraction::Linear { probe: &probe }),
            (
                "bilinear",
                SaeRowJetContraction::Bilinear {
                    probe: &probe,
                    v_t: &v_t,
                    v_beta: &v_beta,
                },
            ),
        ];
        for (label, contraction) in shapes {
            let cpu = execute_softmax_row_jet_tile_contracted(
                &rows,
                1.0,
                SaeRowJetPath::Cpu,
                contraction,
            )
            .expect("CPU contraction oracle");
            execute_softmax_row_jet_tile_contracted(&rows, 1.0, SaeRowJetPath::Device, contraction)
                .expect("admitted contracted device warm-up must execute without a host retry");
            let mut max_error = 0.0_f64;
            for pass in 0..MEASURED_PASSES {
                let device = execute_softmax_row_jet_tile_contracted(
                    &rows,
                    1.0,
                    SaeRowJetPath::Device,
                    contraction,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "admitted contracted device pass {pass} must execute without a host retry: {error}"
                    )
                });
                max_error = cpu
                    .t
                    .iter()
                    .chain(&cpu.beta)
                    .zip(device.t.iter().chain(&device.beta))
                    .fold(max_error, |maximum, (left, right)| {
                        maximum.max((left - right).abs())
                    });
            }
            let outputs_per_pass = cpu.t.len() + cpu.beta.len();
            eprintln!(
                "SAE_ROWJET_CONTRACT_GPU_ACCEPT shape={label} rows={ROW_COUNT} measured_passes={MEASURED_PASSES} outputs_per_pass={outputs_per_pass} max_abs_error={max_error:.17e}"
            );
            assert!(
                max_error <= 1.0e-12,
                "contracted SAE device/CPU {label} reduction error {max_error:e} exceeds 1e-12"
            );
        }
    }

    /// The `Trace` seam must equal the dense θ-adjoint reduction of the
    /// materialized channels against a per-row selected inverse — bit for bit,
    /// because both run the identical row program and identical f64 reduction.
    /// Uses a symmetric `E_tt` / `beta_inv` (as the true deflation-folded
    /// selected inverse is), an arbitrary `inv_vbeta`, and guards that every
    /// output block engages so an all-zeros bug cannot pass.
    #[test]
    fn contracted_trace_matches_manual_selected_inverse_reduction_2304() {
        let rows = complete_fixture(4);
        let inv_tau = 1.0;
        let channels =
            execute_softmax_row_jet_tile(&rows, inv_tau, SaeRowJetPath::Cpu).expect("CPU row jet");
        let (n, q, n_beta) = (channels.n_rows, channels.q, channels.n_beta);
        let scheduled = channels.into_scheduled_rows();
        let dot =
            |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum() };

        let mut e_tt = vec![0.0_f64; n * q * q];
        for row in 0..n {
            for a in 0..q {
                for b in a..q {
                    let value = ((row * 13 + a * 7 + b * 3 + 1) as f64 * 0.02).cos();
                    e_tt[row * q * q + a * q + b] = value;
                    e_tt[row * q * q + b * q + a] = value;
                }
            }
        }
        let inv_vbeta: Vec<f64> = (0..n * q * n_beta)
            .map(|index| ((index * 17 + 3) as f64 * 0.01).sin())
            .collect();
        let mut beta_inv = vec![0.0_f64; n_beta * n_beta];
        for i in 0..n_beta {
            for j in i..n_beta {
                let value = ((i * 5 + j * 3 + 2) as f64 * 0.03).cos();
                beta_inv[i * n_beta + j] = value;
                beta_inv[j * n_beta + i] = value;
            }
        }

        let trace = execute_softmax_row_jet_tile_contracted(
            &rows,
            inv_tau,
            SaeRowJetPath::Cpu,
            SaeRowJetContraction::Trace {
                e_tt: &e_tt,
                inv_vbeta: &inv_vbeta,
                beta_inv: &beta_inv,
            },
        )
        .expect("CPU trace contraction");
        assert_eq!((trace.n_rows, trace.q, trace.n_beta), (n, q, n_beta));

        let mut any_logit = false;
        let mut any_coord = false;
        let mut any_beta = false;
        for (row, jets) in scheduled.iter().enumerate() {
            let e_row = &e_tt[row * q * q..(row + 1) * q * q];
            let vbeta_row = &inv_vbeta[row * q * n_beta..(row + 1) * q * n_beta];
            for w in 0..q {
                let (w_is_logit, atom_w) = primary_kind_atom(rows[row].primaries[w]);
                let mut expected = 0.0_f64;
                for a in 0..q {
                    let (a_is_logit, atom_a) = primary_kind_atom(rows[row].primaries[a]);
                    for b in 0..q {
                        let (b_is_logit, atom_b) = primary_kind_atom(rows[row].primaries[b]);
                        let dh = if w_is_logit && !a_is_logit && !b_is_logit {
                            dot(jets.first(a), jets.first(b))
                                * softmax_data_weight_product_logit_factor(
                                    &rows[row].gate_values,
                                    atom_a,
                                    atom_b,
                                    atom_w,
                                    inv_tau,
                                )
                        } else {
                            dot(jets.second(a, w), jets.first(b))
                                + dot(jets.first(a), jets.second(b, w))
                        };
                        expected += e_row[a * q + b] * dh;
                    }
                    for border in 0..n_beta {
                        let dh = dot(jets.second(a, w), jets.beta(border))
                            + dot(jets.first(a), jets.beta_deriv(w, border));
                        expected += 2.0 * vbeta_row[a * n_beta + border] * dh;
                    }
                }
                for i in 0..n_beta {
                    for j in 0..n_beta {
                        let dh = dot(jets.beta_deriv(w, i), jets.beta(j))
                            + dot(jets.beta(i), jets.beta_deriv(w, j));
                        expected += beta_inv[i * n_beta + j] * dh;
                    }
                }
                assert_eq!(
                    trace.t[row * q + w],
                    expected,
                    "trace t mismatch row={row}, w={w}"
                );
                if trace.t[row * q + w] != 0.0 {
                    if w_is_logit {
                        any_logit = true;
                    } else {
                        any_coord = true;
                    }
                }
            }
            for w_beta in 0..n_beta {
                let mut expected = 0.0_f64;
                for a in 0..q {
                    for b in 0..q {
                        let dh = dot(jets.beta_l_deriv(a, w_beta), jets.first(b))
                            + dot(jets.first(a), jets.beta_l_deriv(b, w_beta));
                        expected += e_row[a * q + b] * dh;
                    }
                    for border in 0..n_beta {
                        let dh = dot(jets.beta_l_deriv(a, w_beta), jets.beta(border));
                        expected += 2.0 * vbeta_row[a * n_beta + border] * dh;
                    }
                }
                assert_eq!(
                    trace.beta[row * n_beta + w_beta],
                    expected,
                    "trace beta mismatch row={row}, w_beta={w_beta}"
                );
                if trace.beta[row * n_beta + w_beta] != 0.0 {
                    any_beta = true;
                }
            }
        }
        assert!(any_logit, "logit-direction trace must engage");
        assert!(any_coord, "coordinate-direction trace must engage");
        assert!(any_beta, "beta-direction trace must engage");
    }

    /// Independent finite-difference oracle for the θ-adjoint's softmax-logit
    /// substitution — the novel #2304 surface that
    /// `contracted_trace_matches_manual_selected_inverse_reduction_2304` cannot
    /// validate because that test re-implements the production reduction formula
    /// (a change-detector, not a correctness oracle).
    ///
    /// The whole point of the `Trace` shape is `Γ = tr(H⁻¹ ∂H/∂θ)`, where for a
    /// logit direction `w` over a coordinate pair `(a,b)` the row Hessian's data
    /// curvature block is the reconstruction Gram `G[a][b] = ⟨J_a, J_b⟩` and the
    /// code differentiates it through the softmax assignment weights (the
    /// `softmax_data_weight_product_logit_factor` substitution) rather than
    /// through second jets. With the coordinate first jet
    /// `J_a = z_{atom_a}·√w·da` (da = decoded-first for slot a) the Gram is
    /// `G[a][b] = z_{atom_a} z_{atom_b} · w · ⟨da,db⟩`, an explicit function of
    /// the softmax gates alone. Choosing the row-local t–t weight `E_tt` to
    /// couple only always-active coordinate slots (and zeroing the border
    /// inverses) collapses the logit-direction output to
    /// `t[w] = Σ_{a,b} E_tt[a][b]·∂G[a][b]/∂ℓ_w = ∂/∂ℓ_w ⟨E_tt, G(ℓ)⟩`.
    ///
    /// This test recomputes that scalar `Φ(ℓ) = ⟨E_tt, G(ℓ)⟩` directly from the
    /// softmax gates and the decoder inner products — sharing no code with the
    /// production reduction, the substitution factor, or the row program's
    /// jets — and central-differences it. A sign error, a missing/extra `inv_tau`
    /// factor, a wrong assignment index in the substitution, or a
    /// reduction-bookkeeping bug on the logit rows all break agreement. Run at
    /// two temperatures so a dropped `inv_tau` cannot hide.
    #[test]
    fn contracted_trace_logit_adjoint_matches_softmax_finite_difference_2304() {
        for &inv_tau in &[1.0_f64, 1.3_f64] {
            let rows = complete_fixture(4);
            let n = rows.len();
            let q = rows[0].primaries.len();
            let p = rows[0].out_dim;
            let n_beta = rows[0].beta_atoms.len();

            // Atoms 0 and 1 carry `active_atoms = true` on every row of
            // `complete_fixture`; atom 2 does not. Restricting `E_tt` to their
            // coordinate slots keeps every first jet in play nonzero, so the
            // production substitution and the independent Gram agree row by row.
            let coord_slots: Vec<usize> = rows[0]
                .primaries
                .iter()
                .enumerate()
                .filter_map(|(slot, primary)| match primary {
                    SaeRowJetPrimary::Coordinate { atom, .. } if *atom == 0 || *atom == 1 => {
                        Some(slot)
                    }
                    _ => None,
                })
                .collect();
            assert!(
                coord_slots.len() >= 2,
                "fixture must expose at least two always-active coordinate slots"
            );
            let coord_atom = |slot: usize| -> usize {
                match rows[0].primaries[slot] {
                    SaeRowJetPrimary::Coordinate { atom, .. } => atom,
                    SaeRowJetPrimary::Logit { .. } => {
                        panic!("coord_slots must only contain coordinate primaries")
                    }
                }
            };

            // Deterministic symmetric weight on the coupled coordinate slots.
            let raw_weight =
                |a: usize, b: usize| -> f64 { ((a * 7 + b * 3 + 1) as f64 * 0.05).cos() };
            let mut e_tt = vec![0.0_f64; n * q * q];
            for row in 0..n {
                for &a in &coord_slots {
                    for &b in &coord_slots {
                        let value = 0.5 * (raw_weight(a, b) + raw_weight(b, a));
                        e_tt[row * q * q + a * q + b] = value;
                    }
                }
            }
            // Zero borders isolate the coordinate-pair data-curvature term.
            let inv_vbeta = vec![0.0_f64; n * q * n_beta];
            let beta_inv = vec![0.0_f64; n_beta * n_beta];

            let trace = execute_softmax_row_jet_tile_contracted(
                &rows,
                inv_tau,
                SaeRowJetPath::Cpu,
                SaeRowJetContraction::Trace {
                    e_tt: &e_tt,
                    inv_vbeta: &inv_vbeta,
                    beta_inv: &beta_inv,
                },
            )
            .expect("CPU trace contraction");

            let dot = |x: &[f64], y: &[f64]| -> f64 { x.iter().zip(y).map(|(&a, &b)| a * b).sum() };
            // z(ℓ) = softmax(inv_tau·ℓ), stable.
            let softmax = |logits: &[f64]| -> Vec<f64> {
                let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits
                    .iter()
                    .map(|&l| ((l - shift) * inv_tau).exp())
                    .collect();
                let sum: f64 = exps.iter().sum();
                exps.iter().map(|&e| e / sum).collect()
            };

            let h = 1e-6;
            let mut max_err = 0.0_f64;
            let mut scale = 0.0_f64;
            let mut checked = 0usize;
            for row in 0..n {
                let input = &rows[row];
                let row_weight = input.sqrt_row_weight * input.sqrt_row_weight;
                let da = |slot: usize| -> &[f64] { &input.decoded_first[slot * p..(slot + 1) * p] };
                // Logits consistent with the stored gates: softmax(inv_tau·ℓ)=z
                // for ℓ_k = ln(z_k)/inv_tau (softmax is shift-invariant and Σz=1).
                let base_logits: Vec<f64> = input
                    .gate_values
                    .iter()
                    .map(|&z| z.ln() / inv_tau)
                    .collect();
                let phi = |logits: &[f64]| -> f64 {
                    let z = softmax(logits);
                    let mut acc = 0.0_f64;
                    for &a in &coord_slots {
                        for &b in &coord_slots {
                            let weight = e_tt[row * q * q + a * q + b];
                            acc += weight
                                * z[coord_atom(a)]
                                * z[coord_atom(b)]
                                * row_weight
                                * dot(da(a), da(b));
                        }
                    }
                    acc
                };
                for w in 0..q {
                    let SaeRowJetPrimary::Logit { atom: atom_w } = input.primaries[w] else {
                        continue;
                    };
                    let mut plus = base_logits.clone();
                    plus[atom_w] += h;
                    let mut minus = base_logits.clone();
                    minus[atom_w] -= h;
                    let finite_difference = (phi(&plus) - phi(&minus)) / (2.0 * h);
                    let produced = trace.t[row * q + w];
                    max_err = max_err.max((finite_difference - produced).abs());
                    scale = scale.max(produced.abs()).max(finite_difference.abs());
                    checked += 1;
                }
            }
            assert!(checked >= 2, "must exercise at least two logit directions");
            assert!(
                scale > 1e-6,
                "softmax-substitution trace outputs must be non-trivial (inv_tau={inv_tau}, scale={scale})"
            );
            assert!(
                max_err <= 1e-6 * (1.0 + scale),
                "logit θ-adjoint vs softmax finite difference disagree \
                 (inv_tau={inv_tau}, max_err={max_err}, scale={scale})"
            );
        }
    }

    /// The contracted ledger must charge the reduced download + probe upload
    /// and NEVER the `q²·p` second channel or `q·n_beta·p` mixed channel that
    /// dominates the elementwise tile. Pinned by exact byte arithmetic — not
    /// weakened to an inequality.
    #[test]
    fn contracted_ledger_drops_tower_charges_reduced_download_2304() {
        let (k, q, p, n_beta) = (3usize, 6usize, 2usize, 3usize);
        let f = std::mem::size_of::<f64>();
        let i = std::mem::size_of::<i32>();

        let full = SaeRowJetMemoryLedger::for_shape(k, q, p, n_beta).expect("full ledger");
        let contracted = SaeRowJetMemoryLedger::for_contracted_shape(k, q, p, n_beta)
            .expect("contracted ledger");

        let input_f64 = (k + 1 + k * p + q * p + q * q * p + n_beta + q * n_beta) * f;
        let input_i32 = (k + q + q) * i;
        // Reduced device state: phase-one dots (k + q + n_beta + q²), reduced
        // outputs (q + n_beta), probe (p), and v_t/v_beta (q + n_beta).
        let device_reduced = (k + q + n_beta + q * q + q + n_beta + p + q + n_beta) * f;
        assert_eq!(
            contracted.device_bytes_per_row,
            input_f64 + input_i32 + device_reduced,
            "contracted device per-row bytes"
        );

        let semantic_input = input_f64
            + n_beta * p * f
            + k * std::mem::size_of::<bool>()
            + q * std::mem::size_of::<SaeRowJetPrimary>()
            + q * std::mem::size_of::<SaeCoordinateSlot>()
            + n_beta * std::mem::size_of::<usize>()
            + std::mem::size_of::<SaeSoftmaxRowJetInput>();
        let production_layout = q * std::mem::size_of::<SaeRowJetPrimary>()
            + std::mem::size_of::<Vec<SaeRowJetPrimary>>();
        let reduced_operands = (q + n_beta + p + q + n_beta) * f;
        assert_eq!(
            contracted.cpu_host_bytes_per_row,
            semantic_input + production_layout + reduced_operands,
            "contracted CPU host per-row bytes"
        );

        // The contracted device output is exactly the elementwise output tower
        // (`q·p + q²·p + n_beta·p + q·n_beta·p`) swapped for the reduced state,
        // so the `q²·p` second channel and `q·n_beta·p` mixed channel are gone.
        let elementwise_output = (q * p + q * q * p + n_beta * p + q * n_beta * p) * f;
        assert_eq!(
            full.device_bytes_per_row - contracted.device_bytes_per_row,
            elementwise_output - device_reduced,
            "contracted must drop exactly the elementwise output tower"
        );
        assert!(
            device_reduced < q * q * p * f,
            "reduced device state must be smaller than the q^2 p second channel alone"
        );
        assert!(
            contracted.cpu_host_bytes_per_row < full.cpu_host_bytes_per_row,
            "contracted host residency must be smaller than the elementwise tile"
        );
    }
}
