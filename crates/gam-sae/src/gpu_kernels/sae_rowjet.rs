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
        let Some(runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            if mode == gam_gpu::GpuPolicy::Required {
                return Err(
                    "complete SAE row jet requires CUDA, but no runtime was admitted".to_string(),
                );
            }
            return Ok(SaeRowJetExecutionPlan {
                path: SaeRowJetPath::Cpu,
                tile_rows: 1,
                ledger,
            });
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
"#;

#[cfg(target_os = "linux")]
mod device {
    use super::{
        COMPLETE_SOFTMAX_KERNEL_SOURCE, SaeRowJetChannels, SaeRowJetPrimary, SaeSoftmaxRowJetInput,
        checked_product,
    };
    use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
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
        let k_i32 =
            i32::try_from(k).map_err(|_| gam_gpu::gpu_err!("SAE row-jet K overflows i32"))?;
        let q_i32 =
            i32::try_from(q).map_err(|_| gam_gpu::gpu_err!("SAE row-jet q overflows i32"))?;
        let p_i32 =
            i32::try_from(p).map_err(|_| gam_gpu::gpu_err!("SAE row-jet p overflows i32"))?;
        let nb_i32 = i32::try_from(n_beta)
            .map_err(|_| gam_gpu::gpu_err!("SAE row-jet beta count overflows i32"))?;

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
        let ledger = SaeRowJetMemoryLedger::for_shape(3, 6, 2, 3).expect("ledger");
        assert!(ledger.fixed_device_bytes > 0);
        assert!(ledger.device_bytes_per_row > ledger.fixed_device_bytes);
        assert!(ledger.host_bytes_per_row > ledger.cpu_host_bytes_per_row);
        assert!(ledger.host_bytes_per_row > ledger.device_bytes_per_row);
        assert_eq!(
            ledger.maximum_rows(ledger.fixed_device_bytes, usize::MAX),
            0
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn complete_device_matches_cpu_every_channel_when_admitted_2304() {
        if gam_gpu::device_runtime::GpuRuntime::global().is_none() {
            return;
        }
        let rows = complete_fixture(37);
        let cpu = execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Cpu).expect("CPU oracle");
        let device = execute_softmax_row_jet_tile(&rows, 1.0, SaeRowJetPath::Device)
            .expect("admitted device must execute without a host retry");
        let max_error = cpu
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
            .fold(0.0_f64, |maximum, (left, right)| {
                maximum.max((left - right).abs())
            });
        assert!(
            max_error <= 1.0e-12,
            "complete SAE device/CPU row-jet error {max_error:e} exceeds 1e-12"
        );
    }
}
