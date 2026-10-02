//! Residual-stream observability of a decoder as an exact linear system (#2951).
//!
//! Readouts are pulled back through declared steps (linear letters, the rank-one unit
//! letters of an MLP normal form, and the routing-law letters of an attention block) into
//! one weighted observability Gramian, kept factored ([`WeightedObservability`]). Its
//! spectrum ranks the directions of the state by how strongly the readouts see them, and a
//! declared subspace's capture is the fraction of that energy it holds.
//!
//! Every number carries a derived roundoff bound. A rank counts the singular values above
//! [`gam_linalg::roundoff::factor_singular_band`] plus the formation bound, a certified
//! lower bound on the exact rank, exact only where it reaches its ceiling; spectral norms
//! are reported as [`SpectralNormBounds`]. Every reported quantity converts to the shared
//! [`EvidenceStatus`].

use std::fmt;

use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerArrayView, FaerLinalgError, FaerSvd, fast_ata, self_adjoint_eigenvalues,
};
use gam_linalg::roundoff::{
    UNIT_ROUNDOFF, accumulation_band, accumulation_growth, factor_singular_band,
    symmetric_spectrum_rounding_band_at_dim,
};
use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Data, Ix2, s};

use super::joint_operators::RoutingLawLetters;
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};

/// What a reported state bound ranges over, as an evidence-status domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateDomain {
    /// Every state of `ℝᵈ` through a spectral norm, so a bound scales with the
    /// state's norm.
    UnitStates { dimension: usize },
}

/// Bounds on the spectral norm of an exact matrix, read off its computed value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralNormBounds {
    /// `max(0, σ̂₁ − band)`. A positive value certifies a nonzero exact matrix.
    pub lower: f64,
    /// `σ̂₁ + band`. The exact norm is at most this.
    pub upper: f64,
}

impl SpectralNormBounds {
    /// The bounds as the evidence status of `sup_{‖h‖₂ = 1} ‖R h‖₂` for the
    /// measured residual `R` on `ℝᵈ`.
    pub fn evidence(
        &self,
        dimension: usize,
    ) -> Result<EvidenceStatus<Array1<f64>, StateDomain>, EvidenceStatusError> {
        EvidenceStatus::unresolved(
            self.lower,
            self.upper,
            Extremum::Supremum,
            None,
            StateDomain::UnitStates { dimension },
        )
    }
}

/// One forward step of a weighted observability pull-back ([`WeightedObservability`]).
///
/// The step maps the state `h_l ∈ ℝ^{n_l}` to `h_{l+1} ∈ ℝ^{n_{l+1}}` by one of
/// its declared letters, and its readouts read `h_{l+1}`.
#[derive(Clone, Debug)]
pub struct ObservabilityStep<'a> {
    /// The declared letters `T_{l,1..H}`, each `n_{l+1} × n_l`. The alphabet is
    /// exactly what is declared: a residual stream is the identity letter, passed
    /// like any other. An attention block enters only as
    /// [`ObservabilityLetter::RoutingLaws`], whose letters the joint-operator owner
    /// derives.
    pub letters: Vec<ObservabilityLetter<'a>>,
    /// Readouts `R_l` of `h_{l+1}`, each `p × n_{l+1}`. May be empty.
    pub readouts: Vec<ArrayView2<'a, f64>>,
}

/// Letters of an [`ObservabilityStep`].
#[derive(Clone, Debug)]
pub enum ObservabilityLetter<'a> {
    /// One linear letter `T`, `n_{l+1} × n_l`.
    Linear(ArrayView2<'a, f64>),
    /// The rank-one letters `C_j = u_j a_jᵀ` of the units of an MLP normal form
    /// `F(x) = Ax + b + Σ_j u_j ψ(a_jᵀx + β_j)`, one letter per unit: `reads`
    /// holds `a_j` as rows (`n × n_l`), `writes` holds `u_j` as rows
    /// (`n × n_{l+1}`). Pulled back together, `S C_j` has the Gramian of the one
    /// row `‖S u_j‖ a_jᵀ`, so an MLP step is `[Linear(A), Units{..}]` and adds
    /// `Aᵀ G A + Σ_j (u_jᵀ G u_j) a_j a_jᵀ`: each read weighted by how strongly
    /// its write is read. The units must be the merged normal form
    /// (`module_split::MlpNormalForm`): a pair of units with opposite affine forms
    /// is one term of the function, and left unmerged its cancelling writes
    /// each add energy that the function does not carry.
    Units {
        reads: ArrayView2<'a, f64>,
        writes: ArrayView2<'a, f64>,
    },
    /// An attention block's letters, one per routing law: the summed value/output
    /// transport of the heads that share one attention pattern, derived by
    /// `joint_operators::attention_letters` from the block's own query/key operators,
    /// the only constructor of [`RoutingLawLetters`]. One letter per head is not
    /// invariant under the cross-head `GL` gauge of heads with one pattern (it moves
    /// each head's transport and not their sum), and it over-counts, so it cannot be
    /// declared here. Each transport carries its formation band, which the pull-back
    /// propagates as `‖S‖_F ‖E‖_F`.
    RoutingLaws(&'a RoutingLawLetters),
}

impl ObservabilityLetter<'_> {
    /// `(n_{l+1}, n_l)`.
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Linear(letter) => letter.dim(),
            Self::Units { reads, writes } => (writes.ncols(), reads.ncols()),
            Self::RoutingLaws(letters) => letters.transports().first().map_or((0, 0), |transport| transport.dim()),
        }
    }

    /// How many words of length one the letter stands for.
    fn count(&self) -> usize {
        match self {
            Self::Linear(_) => 1,
            Self::Units { reads, .. } => reads.nrows(),
            Self::RoutingLaws(letters) => letters.transports().len(),
        }
    }

    fn validate(&self, output: usize, input: usize) -> Result<(), StateError> {
        let context = "weighted observability: letter";
        require_dimension(self.shape().0, output, context)?;
        require_dimension(self.shape().1, input, context)?;
        match self {
            Self::Linear(letter) => require_finite(letter.view(), context),
            Self::Units { reads, writes } => {
                require_dimension(writes.nrows(), reads.nrows(), "weighted observability: units")?;
                require_finite(reads.view(), context)?;
                require_finite(writes.view(), context)
            }
            Self::RoutingLaws(letters) => {
                for transport in letters.transports() {
                    require_dimension(transport.nrows(), output, context)?;
                    require_dimension(transport.ncols(), input, context)?;
                    require_finite(transport.view(), context)?;
                }
                Ok(())
            }
        }
    }

    /// The pulled-back rows `S T` (or `‖S u_j‖ a_jᵀ`) and a Frobenius bound on
    /// their distance from those of the exact source, given the source's own
    /// Frobenius bound `formation`.
    ///
    /// A source error `E` moves `S T` by `E T`, and `‖E T‖_F ≤ ‖E‖_F·‖T‖₂`, so the
    /// propagated part is `formation` times a certified upper bound on the
    /// letter's spectral norm ([`spectral_norm_bounds`]). The Frobenius norm is
    /// no bound to propagate by: it exceeds `‖T‖₂` by up to `√n` (the identity
    /// letter's `√n` exactly), so over `L` steps the band grew as `(Σ‖T‖_F)^L`
    /// while `σ₁` grows as `‖T‖₂`, and a deep stack's band passed `σ₁`. For units
    /// the weight `‖S u_j‖` moves by at most `‖E u_j‖`, and
    /// `(Σ_j ‖E u_j‖²‖a_j‖²)^{1/2} ≤ ‖E‖_F·min(‖U‖₂·max_j ‖a_j‖, (Σ_j ‖u_j‖²‖a_j‖²)^{1/2})`,
    /// `U` the writes as rows: `‖E Uᵀ‖_F ≤ ‖E‖_F‖U‖₂` gives the first,
    /// `‖E u_j‖ ≤ ‖E‖_F‖u_j‖` the second. The rest of each unit's error, the
    /// rounding of `S Uᵀ`, of the weight and of the row scaling, is its own and is
    /// added by Minkowski. A routing law's transport `T` carries its own error
    /// `Δ` (its band), and `(S + E)(T + Δ) − S T` is bounded by
    /// `‖E‖_F(‖T‖₂ + ‖Δ‖₂) + ‖S‖_F‖Δ‖₂`, `‖T‖₂` again certified from above.
    fn pull(
        &self,
        governor: &MemoryGovernor,
        source: &Array2<f64>,
        absolute_source: &Array2<f64>,
        formation: f64,
    ) -> Result<(Array2<f64>, f64), StateError> {
        let width = source.ncols();
        // An exact letter's spectral norm, needed only when there is an error to carry.
        let norm = |matrix: ArrayView2<'_, f64>, context| -> Result<f64, StateError> {
            if formation > 0.0 {
                Ok(spectral_norm_bounds(governor, &matrix, 0.0, context)?.upper)
            } else {
                Ok(0.0)
            }
        };
        match self {
            Self::Linear(letter) => {
                let moved = source.dot(letter);
                let product_band = entrywise_band_norm(
                    absolute_source
                        .dot(&letter.mapv(f64::abs))
                        .mapv(|magnitude| accumulation_band(width, magnitude)),
                );
                let letter_norm = norm(letter.view(), "weighted observability: letter norm")?;
                Ok((moved, formation * letter_norm + product_band))
            }
            Self::Units { reads, writes } => {
                let read = source.dot(&writes.t());
                let read_band = absolute_source
                    .dot(&writes.t().mapv(f64::abs))
                    .mapv(|magnitude| accumulation_band(width, magnitude));
                let rows = source.nrows();
                let mut moved = reads.to_owned();
                let mut local_squared = 0.0_f64;
                let mut spread_squared = 0.0_f64;
                let mut widest_read = 0.0_f64;
                for (unit, mut row) in moved.rows_mut().into_iter().enumerate() {
                    let weight = read.column(unit).iter().map(|value| value * value).sum::<f64>().sqrt();
                    let write_norm = writes.row(unit).iter().map(|value| value * value).sum::<f64>().sqrt();
                    let read_norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
                    let local_error = read_band.column(unit).iter().map(|value| value * value).sum::<f64>().sqrt()
                        + accumulation_growth(rows + 1) * weight
                        + accumulation_growth(1) * weight;
                    row.mapv_inplace(|value| value * weight);
                    local_squared += (local_error * read_norm).powi(2);
                    spread_squared += (write_norm * read_norm).powi(2);
                    widest_read = widest_read.max(read_norm);
                }
                let writes_norm = norm(writes.view(), "weighted observability: write norm")?;
                let propagated = formation * (writes_norm * widest_read).min(spread_squared.sqrt());
                Ok((moved, local_squared.sqrt() + propagated))
            }
            Self::RoutingLaws(letters) => {
                let rows = source.nrows();
                let mut moved = Array2::<f64>::zeros((rows * letters.transports().len(), width));
                let mut squared = 0.0_f64;
                let source_norm = frobenius(source);
                for (index, (transport, &band)) in letters.transports().iter().zip(letters.bands()).enumerate() {
                    moved.slice_mut(s![index * rows..(index + 1) * rows, ..]).assign(&source.dot(transport));
                    let product_band = entrywise_band_norm(
                        absolute_source
                            .dot(&transport.mapv(f64::abs))
                            .mapv(|magnitude| accumulation_band(width, magnitude)),
                    );
                    let transport_norm = norm(transport.view(), "weighted observability: law transport norm")?;
                    let error = formation * (transport_norm + band) + product_band + source_norm * band;
                    squared += error * error;
                }
                Ok((moved, squared.sqrt() * (1.0 + accumulation_growth(2 * letters.transports().len() + 1))))
            }
        }
    }
}

/// The spectrum of an observability Gramian `G = Fᵀ F`, read off its factor `F`.
#[derive(Clone, Debug)]
pub struct ObservabilitySpectrum {
    /// `σ₁ ≥ σ₂ ≥ …` of `F`, so the eigenvalues of `G` are `σᵢ²`. The Gramian
    /// itself is never formed.
    pub singular_values: Vec<f64>,
    /// Every `σᵢ` is within this of a singular value of a factor of the exact
    /// Gramian: [`factor_singular_band`] plus the pull-back's formation bound.
    pub band: f64,
    /// Singular values above `band`: a certified lower bound on the exact rank.
    pub resolved_rank: usize,
    /// `min(stacked rows, n)`: the exact rank cannot exceed it.
    pub rank_ceiling: usize,
    /// `tr G = Σ σᵢ²`.
    pub energy: f64,
    /// `(Σ σᵢ²)² / Σ σᵢ⁴`, the effective number of directions under the
    /// Gramian's own weighting. A summary of the spectrum with no cutoff; zero
    /// for a zero Gramian.
    pub participation_ratio: f64,
}

impl ObservabilitySpectrum {
    /// The exact rank of the Gramian: exact at its ceiling, otherwise the
    /// resolved lower bound and the ceiling.
    pub fn rank_evidence(&self) -> Result<EvidenceStatus<(), StateDomain>, EvidenceStatusError> {
        let domain = StateDomain::UnitStates {
            dimension: self.rank_ceiling,
        };
        if self.resolved_rank == self.rank_ceiling {
            EvidenceStatus::exact(
                self.resolved_rank as f64,
                0.0,
                ExactBasis::Algebraic,
                None,
                domain,
            )
        } else {
            EvidenceStatus::unresolved(
                self.resolved_rank as f64,
                self.rank_ceiling as f64,
                Extremum::Supremum,
                None,
                domain,
            )
        }
    }

    fn of(singular_values: Vec<f64>, rows: usize, cols: usize, formation: f64, stacked_rows: usize) -> Self {
        let sigma_max = singular_values.first().copied().unwrap_or(0.0);
        let band = factor_singular_band(rows.max(1), cols, sigma_max) + formation;
        let resolved_rank = singular_values.iter().filter(|&&value| value > band).count();
        let energy: f64 = singular_values.iter().map(|value| value * value).sum();
        let quartic: f64 = singular_values.iter().map(|value| value.powi(4)).sum();
        let participation_ratio = if quartic > 0.0 { energy * energy / quartic } else { 0.0 };
        Self {
            singular_values,
            band,
            resolved_rank,
            rank_ceiling: stacked_rows.min(cols),
            energy,
            participation_ratio,
        }
    }
}

/// The weighted observability Gramian of a readout family pulled back through
/// declared forward steps, held as a factor.
///
/// # Definition
///
/// Steps `l = 0..L−1` run forward, `h_{l+1} = T_{l,a} h_l` for a letter `a` of
/// step `l`, and step `l`'s readouts `R_l` read `h_{l+1}`. A word
/// `w = (a_0, …, a_l)` picks one letter per step up to `l`, with
/// `T_w = T_{l,a_l} ⋯ T_{0,a_0}`. The Gramian at the input `h_0` is
///
/// ```text
/// G = Σ_{l=0}^{L−1} Σ_{w = (a_0..a_l)} (R_l T_w)ᵀ (R_l T_w)        (n_0 × n_0),
/// ```
///
/// every readout pulled back along every declared word with unit weight. It is
/// formed as a per-step stacked factor: from the last step backward, `S = [F_{l+1}; R_l]` and `F_l = [S T_{l,1}; …; S T_{l,H}]`, so
/// `F_lᵀ F_l = Σ_a T_{l,a}ᵀ (F_{l+1}ᵀ F_{l+1} + Σ R_lᵀ R_l) T_{l,a}` and
/// `G = F_0ᵀ F_0`. A time-invariant family pulled back to depth `k` is one step
/// repeated `k` times. The range of `G` is the exact observable subspace of the
/// same readouts and letters; what the weighting adds is how strongly each
/// direction is read.
/// Exact rank questions are generically saturated on trained weights, so the
/// spectrum, not the rank, carries the information.
///
/// After every letter the stack is compressed to `diag(σ) Vᵀ` by a thin SVD,
/// which keeps `Fᵀ F`, so `G` is never formed and memory is `O(n²)` per letter.
///
/// # Error
///
/// `formation` bounds the Frobenius distance from `factor` to a factor of the
/// exact Gramian of the supplied matrices, up to a left orthogonal factor. Each
/// product `S T` adds the propagated bound times a certified upper bound on
/// `‖T‖₂` (never `‖T‖_F`, which compounds by up to `√n` a step) and the
/// [`accumulation_band`] of `|S||T|`; each compression adds the SVD's backward
/// error ([`factor_singular_band`], times `√rank` for Frobenius), `σ₁` times the
/// measured orthogonality defect of `V`, and the rounding of `σᵢ Vᵢⱼ`.
#[derive(Clone, Debug)]
pub struct WeightedObservability {
    /// `F = diag(σ) Vᵀ` at `h_0`, `min(rows, n_0) × n_0`, with `Fᵀ F = G`.
    pub factor: Array2<f64>,
    /// `V`'s rows: the Gramian's eigenvectors in decreasing `σ`.
    pub directions: Array2<f64>,
    /// Frobenius bound on the distance of `factor` from an exact factor.
    pub formation: f64,
    /// `step_spectra[l]` is the spectrum at `h_l`, the input of step `l`, of the
    /// pull-back of steps `l..L`. `step_spectra[0]` is the spectrum of `G`.
    pub step_spectra: Vec<ObservabilitySpectrum>,
}

/// How much of the weighted observability a declared candidate subspace holds.
#[derive(Clone, Debug)]
pub struct SubspaceCapture {
    /// The candidate's dimension `k`.
    pub dimension: usize,
    /// `tr(Π G) / tr G = ‖F Wᵀ‖²_F / ‖F‖²_F` for the orthogonal projector
    /// `Π = Wᵀ W` onto the candidate, with its numerical error. A uniformly
    /// random `k`-dimensional subspace captures `k/n` in expectation.
    pub energy_fraction: EvidenceStatus<(), StateDomain>,
    /// Cosines of the principal angles between the top `min(k, rows)` Gramian
    /// directions and the candidate, in decreasing order.
    pub principal_cosines: Vec<f64>,
    /// `σ_k² − σ_{k+1}²`, the eigengap that separates the top `k` directions
    /// (`σ_{k+1} = 0` past the factor's rows).
    pub eigengap: f64,
    /// A Davis–Kahan bound on the sine of the largest angle between the computed
    /// and the exact top-`k` space: `‖ΔG‖₂ / (gap − ‖ΔG‖₂)` with
    /// `‖ΔG‖₂ ≤ 2σ₁e + e²`, `e` the factor's error, capped at `1`, and `1` when
    /// the gap does not exceed `‖ΔG‖₂`.
    pub angle_perturbation: f64,
}

impl WeightedObservability {
    /// Pull the readouts of `steps`, in forward order, back to `h_0`.
    pub fn pull_back(governor: &MemoryGovernor, steps: &[ObservabilityStep<'_>]) -> Result<Self, StateError> {
        let last = steps.last().ok_or(StateError::EmptyFamily {
            context: "weighted observability: steps",
        })?;
        if steps.iter().all(|step| step.readouts.is_empty()) {
            return Err(StateError::EmptyFamily {
                context: "weighted observability: readouts",
            });
        }
        let mut width = last
            .letters
            .first()
            .ok_or(StateError::EmptyFamily {
                context: "weighted observability: letters",
            })?
            .shape()
            .0;
        let mut factor = Array2::<f64>::zeros((0, width));
        let mut directions = Array2::<f64>::zeros((0, width));
        let mut formation = 0.0_f64;
        let mut stacked_rows = 0_usize;
        let mut step_spectra = Vec::with_capacity(steps.len());
        for step in steps.iter().rev() {
            let input = step
                .letters
                .first()
                .ok_or(StateError::EmptyFamily {
                    context: "weighted observability: letters",
                })?
                .shape()
                .1;
            for letter in &step.letters {
                letter.validate(width, input)?;
            }
            for readout in &step.readouts {
                require_sample(readout.view(), width, "weighted observability: readout")?;
            }
            let readout_rows: usize = step.readouts.iter().map(|readout| readout.nrows()).sum();
            let source_rows = factor.nrows() + readout_rows;
            // The source stack and its absolute value.
            let source_reservation =
                reserve(governor, source_rows.max(1), width, 2, "weighted observability: step source")?;
            let mut source = Array2::<f64>::zeros((source_rows, width));
            source.slice_mut(s![..factor.nrows(), ..]).assign(&factor);
            let mut offset = factor.nrows();
            for readout in &step.readouts {
                source
                    .slice_mut(s![offset..offset + readout.nrows(), ..])
                    .assign(readout);
                offset += readout.nrows();
            }
            let absolute_source = source.mapv(f64::abs);
            let mut pulled = Array2::<f64>::zeros((0, input));
            let mut pulled_directions = Array2::<f64>::zeros((0, input));
            let mut pulled_sigma = Vec::new();
            let mut pulled_formation = 0.0_f64;
            for letter in &step.letters {
                // The moved source, its absolute bound, and the stack beside the running factor.
                let moved_rows = match letter {
                    ObservabilityLetter::Linear(_) => source_rows,
                    ObservabilityLetter::Units { reads, .. } => reads.nrows() + source_rows,
                    ObservabilityLetter::RoutingLaws(letters) => letters.transports().len().max(1) * source_rows,
                };
                let letter_reservation = reserve(
                    governor,
                    moved_rows.max(1) + input,
                    input.max(width),
                    3,
                    "weighted observability: letter",
                )?;
                let (moved, moved_formation) = letter.pull(governor, &source, &absolute_source, formation)?;
                let mut stack = Array2::<f64>::zeros((pulled.nrows() + moved.nrows(), input));
                stack.slice_mut(s![..pulled.nrows(), ..]).assign(&pulled);
                stack.slice_mut(s![pulled.nrows().., ..]).assign(&moved);
                let stack_formation = pulled_formation.hypot(moved_formation);
                let compressed = compress_factor(governor, &stack, "weighted observability: compression")?;
                pulled = compressed.factor;
                pulled_directions = compressed.directions;
                pulled_sigma = compressed.singular_values;
                pulled_formation = stack_formation + compressed.formation;
                drop(letter_reservation);
            }
            drop(source_reservation);
            stacked_rows = stacked_rows
                .saturating_add(readout_rows)
                .saturating_mul(step.letters.iter().map(ObservabilityLetter::count).sum());
            step_spectra.push(ObservabilitySpectrum::of(
                pulled_sigma,
                pulled.nrows(),
                input,
                pulled_formation,
                stacked_rows,
            ));
            factor = pulled;
            directions = pulled_directions;
            formation = pulled_formation;
            width = input;
        }
        step_spectra.reverse();
        Ok(Self {
            factor,
            directions,
            formation,
            step_spectra,
        })
    }

    /// The spectrum of `G` at `h_0`.
    pub fn spectrum(&self) -> &ObservabilitySpectrum {
        &self.step_spectra[0]
    }

    /// How much of `G` a declared candidate subspace holds (the row span of
    /// `candidate`, `k × n_0`, of full row rank `k`), and its principal angles
    /// to the top `k` Gramian directions. The candidate is the caller's
    /// declaration; nothing here chooses one.
    pub fn capture(
        &self,
        governor: &MemoryGovernor,
        candidate: ArrayView2<'_, f64>,
    ) -> Result<SubspaceCapture, StateError> {
        let width = self.factor.ncols();
        require_sample(candidate, width, "weighted observability: candidate")?;
        let declared = candidate.nrows();
        let basis = resolved_row_space(
            governor,
            &candidate.to_owned(),
            0.0,
            "weighted observability: candidate",
        )?;
        if basis.rank != declared {
            return Err(StateError::CandidateRankDeficient {
                declared,
                resolved: basis.rank,
            });
        }
        let orthonormal = basis.rows;
        // The projection, its absolute bound, and the directions' overlap.
        let working = reserve(
            governor,
            declared.max(self.factor.nrows()).max(1),
            width.max(declared),
            3,
            "weighted observability: capture",
        )?;
        let orthonormality = orthonormality_defect(&orthonormal);
        let projected = self.factor.dot(&orthonormal.t());
        let projected_band = entrywise_band_norm(
            self.factor
                .mapv(f64::abs)
                .dot(&orthonormal.t().mapv(f64::abs))
                .mapv(|magnitude| accumulation_band(width, magnitude)),
        );
        let total = frobenius(&self.factor);
        let held = frobenius(&projected);
        let held_error = self.formation * (1.0 + orthonormality)
            + projected_band
            + total * orthonormality
            + accumulation_growth(projected.len().max(1)) * held;
        let total_error = self.formation + accumulation_growth(self.factor.len().max(1)) * total;
        let fraction = if total > 0.0 { (held / total).powi(2) } else { 0.0 };
        let lower = if total > 0.0 {
            ((held - held_error).max(0.0) / (total + total_error)).powi(2)
        } else {
            0.0
        };
        let upper = if total > total_error {
            ((held + held_error) / (total - total_error)).powi(2).min(1.0)
        } else {
            1.0
        };
        let numerical_error = (fraction - lower).max(upper - fraction).max(0.0);
        let energy_fraction = EvidenceStatus::exact(
            fraction,
            numerical_error,
            ExactBasis::Algebraic,
            None,
            StateDomain::UnitStates { dimension: width },
        )
        .map_err(|source| StateError::Evidence { source })?;

        let top = declared.min(self.directions.nrows());
        let mut principal_cosines = if top == 0 {
            Vec::new()
        } else {
            let overlap = self.directions.slice(s![..top, ..]).dot(&orthonormal.t());
            let (_, cosines, _) = overlap.svd(false, false).map_err(|source| StateError::Svd {
                context: "weighted observability: principal angles",
                source,
            })?;
            cosines.to_vec()
        };
        principal_cosines.sort_by(|left, right| right.total_cmp(left));
        drop(working);

        let sigma = &self.spectrum().singular_values;
        let at = |index: usize| sigma.get(index).copied().unwrap_or(0.0);
        let eigengap = at(declared - 1).powi(2) - at(declared).powi(2);
        let error = self.formation + factor_singular_band(self.factor.nrows().max(1), width, at(0));
        let perturbation = 2.0 * at(0) * error + error * error;
        let angle_perturbation = if eigengap > perturbation {
            (perturbation / (eigengap - perturbation)).min(1.0)
        } else {
            1.0
        };
        Ok(SubspaceCapture {
            dimension: declared,
            energy_fraction,
            principal_cosines,
            eigengap,
            angle_perturbation,
        })
    }
}

/// A stack compressed to `diag(σ) Vᵀ`, with `σ` in decreasing order, `V`'s rows,
/// and the compression's own Frobenius formation bound.
struct CompressedFactor {
    factor: Array2<f64>,
    directions: Array2<f64>,
    singular_values: Vec<f64>,
    formation: f64,
}

fn compress_factor(
    governor: &MemoryGovernor,
    stack: &Array2<f64>,
    context: &'static str,
) -> Result<CompressedFactor, StateError> {
    let (rows, cols) = stack.dim();
    if rows == 0 {
        return Ok(CompressedFactor {
            factor: Array2::zeros((0, cols)),
            directions: Array2::zeros((0, cols)),
            singular_values: Vec::new(),
            formation: 0.0,
        });
    }
    // The decomposition's working copy, then `Vᵀ`, the directions and the factor.
    let working = reserve(governor, rows, cols, 1, context)?;
    let factors = reserve(governor, cols, cols, 3, context)?;
    let (_, sigma, vt) = stack
        .svd(false, true)
        .map_err(|source| StateError::Svd { context, source })?;
    let vt = vt.ok_or(StateError::Svd {
        context,
        source: FaerLinalgError::SvdNoConvergence { context },
    })?;
    let mut order: Vec<usize> = (0..sigma.len()).collect();
    order.sort_by(|&left, &right| sigma[right].total_cmp(&sigma[left]));
    let kept = order.len();
    let mut directions = Array2::<f64>::zeros((kept, cols));
    let mut factor = Array2::<f64>::zeros((kept, cols));
    let mut singular_values = Vec::with_capacity(kept);
    for (row, &index) in order.iter().enumerate() {
        directions.row_mut(row).assign(&vt.row(index));
        factor
            .row_mut(row)
            .assign(&vt.row(index).mapv(|value| value * sigma[index]));
        singular_values.push(sigma[index]);
    }
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let sigma_norm = singular_values.iter().map(|value| value * value).sum::<f64>().sqrt();
    let formation = (kept as f64).sqrt() * factor_singular_band(rows, cols, sigma_max)
        + sigma_max * orthonormality_defect(&directions)
        + accumulation_growth(1) * sigma_norm;
    drop(working);
    drop(factors);
    Ok(CompressedFactor {
        factor,
        directions,
        singular_values,
        formation,
    })
}

/// `‖W Wᵀ − I‖_F` plus the rounding of the product: a bound on the distance of
/// the rows of `W` from an exactly orthonormal set.
pub(super) fn orthonormality_defect(rows: &Array2<f64>) -> f64 {
    let count = rows.nrows();
    let width = rows.ncols();
    let gram = rows.dot(&rows.t()) - Array2::<f64>::eye(count);
    let absolute = rows.mapv(f64::abs);
    let band = entrywise_band_norm(
        absolute
            .dot(&absolute.t())
            .mapv(|magnitude| accumulation_band(width + 1, magnitude + 1.0)),
    );
    frobenius(&gram) + band
}

pub(super) fn frobenius<S: ndarray::Data<Elem = f64>>(matrix: &ndarray::ArrayBase<S, ndarray::Ix2>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// The resolved row space of a matrix: its singular values, the band, the rank
/// and the leading right singular vectors as rows.
pub(super) struct ResolvedRowSpace {
    pub(super) rank: usize,
    pub(super) rows: Array2<f64>,
}

pub(super) fn resolved_row_space(
    governor: &MemoryGovernor,
    matrix: &Array2<f64>,
    formation: f64,
    context: &'static str,
) -> Result<ResolvedRowSpace, StateError> {
    let (rows, cols) = matrix.dim();
    // The decomposition's working copy, then `Vᵀ` and the resolved rows beside it.
    let working = reserve(governor, rows, cols, 1, context)?;
    let factors = reserve(governor, cols, cols, 2, context)?;
    let (_, sigma, vt) = matrix
        .svd(false, true)
        .map_err(|source| StateError::Svd { context, source })?;
    let vt = vt.ok_or(StateError::Svd {
        context,
        source: FaerLinalgError::SvdNoConvergence { context },
    })?;
    let mut order: Vec<usize> = (0..sigma.len()).collect();
    order.sort_by(|&left, &right| sigma[right].total_cmp(&sigma[left]));
    let singular_values: Vec<f64> = order.iter().map(|&index| sigma[index]).collect();
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);
    let band = factor_singular_band(rows, cols, sigma_max) + formation;
    let rank = singular_values.iter().filter(|&&value| value > band).count();
    let mut resolved = Array2::<f64>::zeros((rank, cols));
    for (row, &index) in order.iter().take(rank).enumerate() {
        resolved.row_mut(row).assign(&vt.row(index));
    }
    drop(working);
    drop(factors);
    Ok(ResolvedRowSpace { rank, rows: resolved })
}

/// Bounds on `‖R‖₂` for an exact matrix `R` whose computed value `R̂` (`m × n`)
/// is within `formation` of it in spectral norm, read off the largest
/// eigenvalue of the Gram of `R̂`'s smaller side. No singular value
/// decomposition is formed.
///
/// With `k = min(m, n)`, `t = max(m, n)`, `u` the unit roundoff,
/// `γ_j = j·u/(1 − j·u)` and `η = 2⁻¹⁰⁷⁴`:
///
/// 1. **Scale.** `B = 2^{−e}·R̂`, with `e` taking `max|r̂ᵢⱼ|` to about `[½, 1)`,
///    so the Gram below neither underflows nor overflows. A power of two is
///    exact except where an entry lands subnormal, which moves it by at most
///    `η`, so by Weyl `|σ_max(B) − 2^{−e}‖R̂‖₂| ≤ β = √(mn)·η`.
/// 2. **Gram.** `Ĝ = fl(BᵀB)` (or `fl(BBᵀ)`), `k × k`, with one triangle
///    mirrored, so `Ĝ` is exactly symmetric. Each entry is an inner product of
///    length `t`, so `|Ĝ − G| ≤ γ_t·|B|ᵀ|B|` entrywise, in any summation order and
///    with or without fused multiply-adds (Higham, *ASNA* 2nd ed., §3.5). The
///    majorant is entrywise non-negative and positive semidefinite, so
///    `‖Ĝ − G‖₂ ≤ γ_t·‖|B|ᵀ|B|‖₂ ≤ γ_t·tr(|B|ᵀ|B|) = γ_t·‖B‖²_F`. A diagonal entry
///    sums squares, so `Ĝᵢᵢ ≥ (1 − γ_t)·Gᵢᵢ`, and the trace sums `k`
///    non-negative terms, so `‖B‖²_F ≤ fl(tr Ĝ)/((1 − γ_t)(1 − γ_k))` and
///    `δ = γ_t·fl(tr Ĝ)/((1 − γ_t)(1 − γ_k))` bounds `‖Ĝ − G‖₂`.
/// 3. **Spectrum.** The self-adjoint eigensolver is backward stable: its
///    computed `λ̂` are the exact eigenvalues of `Ĝ + E` with
///    `‖E‖₂ ≤ ρ = k·(ε·max|λ̂| + η)`
///    ([`symmetric_spectrum_rounding_band_at_dim`]; the same convention as the
///    [`factor_singular_band`] a full SVD reads). By Weyl,
///    `σ_max(B)² = λ_max(G) ∈ [λ̂_max − ρ − δ, λ̂_max + ρ + δ]`.
/// 4. **Root.** `√·` is monotone. The endpoints are widened for the handful of
///    rounded operations that form them, each root is taken one ulp outward,
///    `β` is added on each side, and the result is multiplied back by `2^e`
///    (exact, one more ulp outward for a subnormal landing).
/// 5. **Exact matrix.** `|‖R‖₂ − ‖R̂‖₂| ≤ formation`, so
///    `lower = max(0, σ_lo − formation)` and `upper = σ_hi + formation` bracket
///    `‖R‖₂`.
///
/// Since `‖B‖²_F ≤ k·σ_max(B)²`, `δ ≤ γ_t·k·λ_max`, so the bracket on `‖R̂‖₂` is
/// within about `(t + 1)·k·u/2` of it relatively (`2.3e-10` at `t = k = 2048`),
/// where a full SVD's band is `t·ε`. Every consumer reads only `lower` and
/// `upper` as a certified interval (the evidence status, the wire report, the
/// verdicts built on them), and both remain bounds on the exact norm. The Gram
/// is one `k × k × t` product at pool parallelism and the spectrum an
/// eigenvalue-only decomposition at the fixed EVD degree, so the result is
/// identical at every pool width.
pub(super) fn spectral_norm_bounds<S: Data<Elem = f64>>(
    governor: &MemoryGovernor,
    matrix: &ArrayBase<S, Ix2>,
    formation: f64,
    context: &'static str,
) -> Result<SpectralNormBounds, StateError> {
    if matrix.is_empty() {
        return Ok(SpectralNormBounds {
            lower: 0.0,
            upper: formation,
        });
    }
    let (rows, cols) = matrix.dim();
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(StateError::NonFinite { context });
    }
    let largest = matrix.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if largest == 0.0 {
        return Ok(SpectralNormBounds {
            lower: 0.0,
            upper: formation,
        });
    }
    let short = rows.min(cols);
    let long = rows.max(cols);
    // The scaled copy, then the Gram and the eigensolver's working copy of it.
    let working = reserve(governor, rows, cols, 1, context)?;
    let gram_reservation = reserve(governor, short, short, 2, context)?;
    // `2^e` overflows for `e > 1023` and `2^{−e}` for a subnormal `e`, so each
    // power is applied as two representable halves.
    let exponent = largest.log2().floor() as i32 + 1;
    let (shrink, shrink_tail) = (
        2.0_f64.powi(-(exponent / 2)),
        2.0_f64.powi(-(exponent - exponent / 2)),
    );
    let scaled = matrix.mapv(|value| value * shrink * shrink_tail);
    let gram = if cols <= rows {
        fast_ata(&scaled)
    } else {
        fast_ata(&scaled.t())
    };
    drop(scaled);
    drop(working);
    let gram_view = FaerArrayView::new(&gram);
    let spectrum = self_adjoint_eigenvalues(gram_view.as_ref(), Side::Lower).map_err(|source| {
        StateError::Eigen {
            context,
            source: FaerLinalgError::SelfAdjointEigen(source),
        }
    })?;
    drop(gram_view);
    let spectrum = spectrum.as_ref().column_vector();
    let eigenvalues: Vec<f64> = (0..short).map(|index| spectrum[index]).collect();
    let trace: f64 = (0..short).map(|index| gram[[index, index]]).sum();
    drop(gram);
    drop(gram_reservation);
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(StateError::NonFinite { context });
    }
    let largest_eigenvalue = eigenvalues
        .iter()
        .fold(f64::NEG_INFINITY, |largest, &value| largest.max(value));
    let gram_formation = accumulation_growth(long) * trace
        / ((1.0 - accumulation_growth(long)) * (1.0 - accumulation_growth(short)));
    let spectrum_band = symmetric_spectrum_rounding_band_at_dim(short, &eigenvalues);
    // The slack and each endpoint take a handful of rounded operations.
    let slack = (spectrum_band + gram_formation) * (1.0 + accumulation_growth(8));
    let widen = 4.0 * UNIT_ROUNDOFF;
    let squared_upper = (largest_eigenvalue + slack) * (1.0 + widen);
    let squared_lower = ((largest_eigenvalue - slack) * (1.0 - widen)).max(0.0);
    let subnormal_shift = ((rows as f64).sqrt() * (cols as f64).sqrt() * f64::from_bits(1)).next_up();
    let scaled_upper = (squared_upper.sqrt().next_up() + subnormal_shift).next_up();
    let scaled_lower = (squared_lower.sqrt().next_down() - subnormal_shift).next_down().max(0.0);
    let (grow, grow_tail) = (
        2.0_f64.powi(exponent / 2),
        2.0_f64.powi(exponent - exponent / 2),
    );
    let sigma_upper = (scaled_upper * grow * grow_tail).next_up();
    let sigma_lower = (scaled_lower * grow * grow_tail).next_down().max(0.0);
    Ok(SpectralNormBounds {
        lower: (sigma_lower - formation).max(0.0),
        upper: sigma_upper + formation,
    })
}

/// Reserves `copies` dense `rows × cols` matrices on `governor` before they are formed.
pub(super) fn reserve(
    governor: &MemoryGovernor,
    rows: usize,
    cols: usize,
    copies: usize,
    context: &'static str,
) -> Result<MemoryReservation, StateError> {
    governor
        .try_reserve_dense_f64_copies(rows, cols, copies, context)
        .map_err(|source| StateError::Memory { context, source })
}

/// `‖B‖_F` of an entrywise error bound `B`, which bounds the spectral norm of the
/// error.
pub(super) fn entrywise_band_norm(band: Array2<f64>) -> f64 {
    band.iter().map(|entry| entry * entry).sum::<f64>().sqrt()
}

/// Errors and refusals of the sufficient-state checks.
#[derive(Debug)]
pub enum StateError {
    /// No futures, responses, states or codes were stated.
    EmptyFamily { context: &'static str },
    /// Two dimensions that must agree do not.
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        found: usize,
    },
    /// A stated sample, value, Jacobian or roundoff bound is not finite.
    NonFinite { context: &'static str },
    /// A singular value decomposition failed.
    Svd {
        context: &'static str,
        source: FaerLinalgError,
    },
    /// A self-adjoint eigendecomposition failed.
    Eigen {
        context: &'static str,
        source: FaerLinalgError,
    },
    /// An evaluated defect could not be expressed as an evidence status.
    Evidence { source: EvidenceStatusError },
    /// A dense matrix the check forms does not fit the memory budget.
    Memory {
        context: &'static str,
        source: MemoryReservationError,
    },
    /// A declared candidate subspace's rows do not resolve to full row rank.
    CandidateRankDeficient { declared: usize, resolved: usize },
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFamily { context } => write!(formatter, "{context}: nothing was stated"),
            Self::DimensionMismatch {
                context,
                expected,
                found,
            } => write!(formatter, "{context}: expected dimension {expected}, found {found}"),
            Self::NonFinite { context } => write!(formatter, "{context}: non-finite value"),
            Self::Svd { context, source } => {
                write!(formatter, "{context}: singular value decomposition failed: {source}")
            }
            Self::Eigen { context, source } => {
                write!(formatter, "{context}: self-adjoint eigendecomposition failed: {source}")
            }
            Self::Evidence { source } => {
                write!(formatter, "evidence status refused an evaluated defect: {source}")
            }
            Self::Memory { context, source } => write!(formatter, "{context}: {source}"),
            Self::CandidateRankDeficient { declared, resolved } => write!(
                formatter,
                "declared candidate subspace has {declared} rows but resolves to rank {resolved}"
            ),
        }
    }
}

impl std::error::Error for StateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Svd { source, .. } => Some(source),
            Self::Eigen { source, .. } => Some(source),
            Self::Evidence { source } => Some(source),
            Self::Memory { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn require_dimension(found: usize, expected: usize, context: &'static str) -> Result<(), StateError> {
    if found == expected {
        Ok(())
    } else {
        Err(StateError::DimensionMismatch {
            context,
            expected,
            found,
        })
    }
}

fn require_finite(matrix: ArrayView2<'_, f64>, context: &'static str) -> Result<(), StateError> {
    if matrix.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StateError::NonFinite { context })
    }
}

fn require_sample(
    sample: ArrayView2<'_, f64>,
    dimension: usize,
    context: &'static str,
) -> Result<(), StateError> {
    if sample.nrows() == 0 {
        return Err(StateError::EmptyFamily { context });
    }
    require_dimension(sample.ncols(), dimension, context)?;
    if sample.iter().any(|value| !value.is_finite()) {
        return Err(StateError::NonFinite { context });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;
    use ndarray::s;

    /// The Sylvester Hadamard matrix of order `2^power`: entries `±1`, orthogonal
    /// columns of squared norm `2^power`.
    fn hadamard(power: u32) -> Array2<f64> {
        let order = 1_usize << power;
        Array2::from_shape_fn((order, order), |(row, col)| {
            if (row & col).count_ones() % 2 == 0 { 1.0 } else { -1.0 }
        })
    }

    /// `H_m[:, :k] diag(s) H_n[:, :k]ᵀ`, every entry an exact dyadic sum, so its
    /// exact singular values are `|sᵢ|·√(mn)` and the matrix as stored is exact.
    fn exact_spectrum(rows_power: u32, cols_power: u32, weights: &[f64]) -> Array2<f64> {
        let left = hadamard(rows_power);
        let right = hadamard(cols_power);
        let mut matrix = Array2::<f64>::zeros((left.nrows(), right.nrows()));
        for (index, &weight) in weights.iter().enumerate() {
            for row in 0..left.nrows() {
                for col in 0..right.nrows() {
                    matrix[[row, col]] += weight * left[[row, index]] * right[[col, index]];
                }
            }
        }
        matrix
    }

    /// The Gram-spectrum bounds bracket the exact `σ_max` of exactly representable
    /// matrices with a known spectrum: full rank, rank deficient, wide and tall,
    /// with a clustered top, and at both ends of the exponent range; the bracket is
    /// as tight as its derivation says, and `formation` widens it by exactly itself.
    #[test]
    fn gram_spectrum_bounds_bracket_the_exact_largest_singular_value() {
        let cluster = 1.0 - (-40.0_f64).exp2();
        let cases: Vec<(&str, u32, u32, Vec<f64>)> = vec![
            ("full rank", 6, 6, (0..64).map(|index| 1.0 - index as f64 / 128.0).collect()),
            ("clustered top", 6, 6, vec![1.0, cluster, cluster, cluster, 0.5, 0.25]),
            ("rank deficient tall", 7, 5, vec![0.75, 0.5, 0.0, 0.125]),
            ("rank deficient wide", 5, 7, vec![0.75, 0.5, 0.0, 0.125]),
            ("rank one", 6, 4, vec![0.5]),
            ("negative weights", 5, 5, vec![-1.0, 0.875, -0.875]),
        ];
        for (name, rows_power, cols_power, weights) in cases {
            let matrix = exact_spectrum(rows_power, cols_power, &weights);
            let scale = ((matrix.nrows() * matrix.ncols()) as f64).sqrt();
            let exact = weights.iter().fold(0.0_f64, |largest, weight| largest.max(weight.abs())) * scale;
            for exponent in [-1000_i32, 0, 1000] {
                let power = f64::from(exponent).exp2();
                let scaled = matrix.mapv(|value| value * power);
                let truth = exact * power;
                let bounds = spectral_norm_bounds(test_governor(), &scaled, 0.0, "test").expect("bounds");
                assert!(
                    bounds.lower <= truth && truth <= bounds.upper,
                    "{name} at 2^{exponent}: {bounds:?} misses {truth:e}"
                );
                let long = scaled.nrows().max(scaled.ncols()) as f64;
                let short = scaled.nrows().min(scaled.ncols()) as f64;
                let width = (bounds.upper - bounds.lower) / truth;
                assert!(
                    width <= 4.0 * (long + 1.0) * short * UNIT_ROUNDOFF,
                    "{name} at 2^{exponent}: relative width {width:e}"
                );
            }
            let formation = 0.5 * exact;
            let widened = spectral_norm_bounds(test_governor(), &matrix, formation, "test").expect("bounds");
            let tight = spectral_norm_bounds(test_governor(), &matrix, 0.0, "test").expect("bounds");
            assert_eq!(widened.upper, tight.upper + formation, "{name}");
            assert_eq!(widened.lower, (tight.lower - formation).max(0.0), "{name}");
        }
        let zero = Array2::<f64>::zeros((3, 5));
        let bounds = spectral_norm_bounds(test_governor(), &zero, 0.25, "test").expect("bounds");
        assert_eq!((bounds.lower, bounds.upper), (0.0, 0.25));
        let mut infinite = Array2::<f64>::eye(3);
        infinite[[1, 2]] = f64::INFINITY;
        assert!(matches!(
            spectral_norm_bounds(test_governor(), &infinite, 0.0, "test"),
            Err(StateError::NonFinite { .. })
        ));
    }

    /// On a seeded dense matrix the bracket agrees with the full SVD it replaces:
    /// each interval contains the other's centre, both being certified.
    #[test]
    fn gram_spectrum_bounds_agree_with_the_singular_value_decomposition() {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(2951);
        for (rows, cols) in [(40, 40), (17, 90), (90, 17), (1, 30)] {
            let matrix = Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0));
            let (_, sigma, _) = matrix.svd(false, false).expect("svd");
            let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
            let band = factor_singular_band(rows, cols, sigma_max);
            let bounds = spectral_norm_bounds(test_governor(), &matrix, 0.0, "test").expect("bounds");
            assert!(
                bounds.lower <= sigma_max + band && sigma_max - band <= bounds.upper,
                "{rows}x{cols}: {bounds:?} against σ̂ {sigma_max:e} ± {band:e}"
            );
        }
    }

    /// Weighted observability: a modular-addition-like readout of `p` answers
    /// through two frequency planes of a hidden basis, and attention letters that
    /// act inside those planes and inside their complement separately.
    mod weighted_observability {
        use super::*;
        use crate::test_support::hidden_basis;
        use gam_linalg::faer_ndarray::FaerEigh;
        use gam_linalg::roundoff::symmetric_spectrum_rounding_band;
        use faer::Side;

        const WIDTH: usize = 12;
        const ANSWERS: usize = 13;
        const PLANES: [(usize, f64); 2] = [(2, 1.0), (5, 0.7)];

        /// `U`'s columns as rows: `frame.row(i)` is the `i`-th hidden direction,
        /// the first `2 · PLANES.len()` spanning the key planes.
        fn frame() -> Array2<f64> {
            hidden_basis(WIDTH, 2951).t().to_owned()
        }

        /// Answer `c` is read by `Σ_k α_k (cos(ω_k c) e_{2k} + sin(ω_k c) e_{2k+1})`
        /// in the hidden frame, plus `leak` times a fixed complement row.
        fn readout(leak: f64) -> Array2<f64> {
            let frame = frame();
            let mut rows = Array2::<f64>::zeros((ANSWERS, WIDTH));
            for answer in 0..ANSWERS {
                let mut row = Array1::<f64>::zeros(WIDTH);
                for (plane, &(frequency, weight)) in PLANES.iter().enumerate() {
                    let angle = std::f64::consts::TAU * (frequency * answer) as f64 / ANSWERS as f64;
                    row.scaled_add(weight * angle.cos(), &frame.row(2 * plane));
                    row.scaled_add(weight * angle.sin(), &frame.row(2 * plane + 1));
                }
                for extra in 2 * PLANES.len()..WIDTH {
                    row.scaled_add(leak * ((answer + extra) as f64).sin(), &frame.row(extra));
                }
                rows.row_mut(answer).assign(&row);
            }
            rows
        }

        /// `Uᵀ B U` with `B` a scaled rotation in each key plane and a fixed
        /// contraction on the complement: every letter keeps both subspaces.
        fn letter(head: usize) -> Array2<f64> {
            let frame = frame();
            let mut block = Array2::<f64>::zeros((WIDTH, WIDTH));
            for plane in 0..PLANES.len() {
                let angle = 0.4 + 0.9 * (head + plane) as f64;
                let scale = 0.8 + 0.1 * head as f64;
                let (sine, cosine) = angle.sin_cos();
                let offset = 2 * plane;
                block[[offset, offset]] = scale * cosine;
                block[[offset, offset + 1]] = -scale * sine;
                block[[offset + 1, offset]] = scale * sine;
                block[[offset + 1, offset + 1]] = scale * cosine;
            }
            for row in 2 * PLANES.len()..WIDTH {
                for col in 2 * PLANES.len()..WIDTH {
                    block[[row, col]] = 0.1 * (((row * 7 + col * 3 + head) % 5) as f64 - 2.0) / 2.0;
                }
            }
            frame.t().dot(&block).dot(&frame)
        }

        fn key_planes() -> Array2<f64> {
            frame().slice(s![..2 * PLANES.len(), ..]).to_owned()
        }

        /// Two steps of `{I, A_1, A_2}` with the answer readout after the last
        /// step and a second readout `D` after the first.
        fn steps<'a>(
            identity: &'a Array2<f64>,
            letters: &'a [Array2<f64>],
            answers: &'a Array2<f64>,
            early: &'a Array2<f64>,
        ) -> Vec<ObservabilityStep<'a>> {
            let alphabet: Vec<ObservabilityLetter<'a>> = std::iter::once(identity)
                .chain(letters.iter())
                .map(|letter| ObservabilityLetter::Linear(letter.view()))
                .collect();
            vec![
                ObservabilityStep {
                    letters: alphabet.clone(),
                    readouts: vec![early.view()],
                },
                ObservabilityStep {
                    letters: alphabet,
                    readouts: vec![answers.view()],
                },
            ]
        }

        /// The Gramian formed by enumerating every word, with an entrywise bound on
        /// its formation.
        fn enumerated_gramian(
            identity: &Array2<f64>,
            letters: &[Array2<f64>],
            answers: &Array2<f64>,
            early: &Array2<f64>,
        ) -> (Array2<f64>, f64) {
            let alphabet: Vec<&Array2<f64>> = std::iter::once(identity).chain(letters.iter()).collect();
            let mut gram = Array2::<f64>::zeros((WIDTH, WIDTH));
            let mut absolute = Array2::<f64>::zeros((WIDTH, WIDTH));
            let mut add = |rows: Array2<f64>, depth: usize| {
                gram += &rows.t().dot(&rows);
                let magnitude = rows.mapv(f64::abs);
                absolute += &magnitude
                    .t()
                    .dot(&magnitude)
                    .mapv(|entry| accumulation_growth(depth * WIDTH + rows.nrows() + 64) * entry);
            };
            for first in &alphabet {
                add(early.dot(*first), 2);
                for second in &alphabet {
                    add(answers.dot(*second).dot(*first), 3);
                }
            }
            let band = absolute.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
            (gram, band)
        }

        /// The factor's spectrum is the enumerated Gramian's, within the factor's
        /// derived error and the enumeration's own bound (Weyl).
        #[test]
        fn factor_spectrum_matches_the_enumerated_gramian() {
            let identity = Array2::<f64>::eye(WIDTH);
            let letters = [letter(0), letter(1)];
            let answers = readout(0.05);
            let early = frame().slice(s![WIDTH - 3.., ..]).to_owned();
            let report = WeightedObservability::pull_back(
                test_governor(),
                &steps(&identity, &letters, &answers, &early),
            )
            .expect("pull back");
            let (gram, band) = enumerated_gramian(&identity, &letters, &answers, &early);
            let (eigenvalues, _) = gram.eigh(Side::Lower).expect("eigh");
            let mut expected: Vec<f64> = eigenvalues.to_vec();
            expected.sort_by(|left, right| right.total_cmp(left));
            let sigma = &report.spectrum().singular_values;
            let error = report.formation + report.spectrum().band;
            let tolerance = 2.0 * sigma[0] * error
                + error * error
                + band
                + symmetric_spectrum_rounding_band(&expected);
            for (index, &value) in expected.iter().enumerate() {
                let computed = sigma.get(index).copied().unwrap_or(0.0).powi(2);
                assert!(
                    (computed - value).abs() <= tolerance,
                    "eigenvalue {index}: factor {computed:e}, enumerated {value:e}, tolerance {tolerance:e}"
                );
            }
            assert_eq!(report.step_spectra.len(), 2);
            let trace: f64 = expected.iter().sum();
            assert!((report.spectrum().energy - trace).abs() <= WIDTH as f64 * tolerance);
        }

        /// With no leak every word keeps the key planes, so the Gramian lives in
        /// them: the planes capture all of its energy up to the derived error, the
        /// top-4 directions are the planes, the resolved rank is 4, and the exact
        /// closure's chart holds the same energy.
        #[test]
        fn planted_planes_capture_the_energy() {
            let identity = Array2::<f64>::eye(WIDTH);
            let letters = [letter(0), letter(1)];
            let answers = readout(0.0);
            let early = answers.slice(s![..2, ..]).to_owned();
            let report = WeightedObservability::pull_back(
                test_governor(),
                &steps(&identity, &letters, &answers, &early),
            )
            .expect("pull back");
            let planes = key_planes();
            let capture = report.capture(test_governor(), planes.view()).expect("capture");
            let EvidenceStatus::Exact {
                value,
                numerical_error,
                ..
            } = capture.energy_fraction
            else {
                panic!("an energy fraction is an exact algebraic value");
            };
            assert!(1.0 - value <= numerical_error, "planes hold {value} ± {numerical_error:e}");            assert!(capture.angle_perturbation < 1.0);
            let floor = (1.0 - capture.angle_perturbation.powi(2)).sqrt();
            // The computed directions themselves carry the compression's formation.
            let slack = report.formation / capture.eigengap.sqrt();
            for &cosine in &capture.principal_cosines {
                assert!(cosine >= floor - slack, "principal cosine {cosine} below {floor}");
            }
            assert_eq!(report.spectrum().resolved_rank, 2 * PLANES.len());
            assert!(matches!(
                report.spectrum().rank_evidence().expect("rank evidence"),
                EvidenceStatus::Unresolved { lower, .. } if lower == (2 * PLANES.len()) as f64
            ));
            assert!(report.spectrum().participation_ratio <= (2 * PLANES.len()) as f64);
        }

        /// Positive control: a uniformly random 4-dimensional subspace captures
        /// `4/12` of the energy in expectation, so the mean over draws lands on
        /// `1/3` within its standard error, far from the planted planes' `1`.
        #[test]
        fn random_subspaces_capture_only_their_share() {
            let identity = Array2::<f64>::eye(WIDTH);
            let letters = [letter(0), letter(1)];
            let answers = readout(0.3);
            let early = answers.slice(s![..2, ..]).to_owned();
            let report = WeightedObservability::pull_back(
                test_governor(),
                &steps(&identity, &letters, &answers, &early),
            )
            .expect("pull back");
            let dimension = 2 * PLANES.len();
            let draws = 400;
            let fractions: Vec<f64> = (0..draws)
                .map(|draw| {
                    let basis = hidden_basis(WIDTH, 10_000 + draw as u64);
                    let candidate = basis.slice(s![..dimension, ..]).to_owned();
                    match report
                        .capture(test_governor(), candidate.view())
                        .expect("capture")
                        .energy_fraction
                    {
                        EvidenceStatus::Exact { value, .. } => value,
                        other => panic!("{other:?}"),
                    }
                })
                .collect();
            let mean = fractions.iter().sum::<f64>() / draws as f64;
            let variance =
                fractions.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (draws - 1) as f64;
            let standard_error = (variance / draws as f64).sqrt();
            let share = dimension as f64 / WIDTH as f64;
            assert!(
                (mean - share).abs() <= 4.0 * standard_error,
                "random mean {mean} vs share {share} (se {standard_error:e})"
            );
            let planted = report.capture(test_governor(), key_planes().view()).expect("capture");
            let EvidenceStatus::Exact { value, .. } = planted.energy_fraction else {
                panic!("exact");
            };
            assert!(value > mean + 4.0 * variance.sqrt());
        }

        #[test]
        fn refuses_rank_deficient_candidates_and_empty_families() {
            let identity = Array2::<f64>::eye(WIDTH);
            let letters = [letter(0)];
            let answers = readout(0.0);
            let early = answers.slice(s![..1, ..]).to_owned();
            let report = WeightedObservability::pull_back(
                test_governor(),
                &steps(&identity, &letters, &answers, &early),
            )
            .expect("pull back");
            let row = key_planes().slice(s![..1, ..]).to_owned();
            let doubled = ndarray::concatenate![ndarray::Axis(0), row, row];
            assert!(matches!(
                report.capture(test_governor(), doubled.view()),
                Err(StateError::CandidateRankDeficient { declared: 2, resolved: 1 })
            ));
            let silent = ObservabilityStep {
                letters: vec![ObservabilityLetter::Linear(identity.view())],
                readouts: vec![],
            };
            assert!(matches!(
                WeightedObservability::pull_back(test_governor(), &[silent]),
                Err(StateError::EmptyFamily { .. })
            ));
        }

        /// A deep stack of near-identity steps, each a residual letter
        /// `I + small` beside a few MLP units, read at the top. The exact error of
        /// the pull-back grows by a constant multiple of `ε·σ₁` per step, so the
        /// certified band has to stay `O(L·ε·σ₁)`. Propagating the formation bound by
        /// `‖T‖_F` multiplied it by about `√n` a step (`24^{15} ≈ 5e20` here) and
        /// buried every singular value under the band.
        #[test]
        fn deep_near_identity_stack_keeps_the_band_linear_in_depth() {
            use rand::rngs::StdRng;
            use rand::{RngExt, SeedableRng};
            let (width, depth, units) = (24_usize, 30_usize, 4_usize);
            let mut rng = StdRng::seed_from_u64(0x2951_dee9);
            let mut draw = |rows: usize, scale: f64| {
                Array2::from_shape_fn((rows, width), |_| scale * rng.random_range(-1.0..1.0))
            };
            let residuals: Vec<Array2<f64>> = (0..depth)
                .map(|_| Array2::<f64>::eye(width) + draw(width, 0.02 / (width as f64).sqrt()))
                .collect();
            let reads: Vec<Array2<f64>> = (0..depth).map(|_| draw(units, 0.1)).collect();
            let writes: Vec<Array2<f64>> = (0..depth).map(|_| draw(units, 0.1)).collect();
            let readout = draw(3, 1.0);
            let steps: Vec<ObservabilityStep<'_>> = (0..depth)
                .map(|layer| ObservabilityStep {
                    letters: vec![
                        ObservabilityLetter::Linear(residuals[layer].view()),
                        ObservabilityLetter::Units {
                            reads: reads[layer].view(),
                            writes: writes[layer].view(),
                        },
                    ],
                    readouts: if layer + 1 == depth { vec![readout.view()] } else { vec![] },
                })
                .collect();
            let report = WeightedObservability::pull_back(test_governor(), &steps).expect("pull back");
            for (layer, spectrum) in report.step_spectra.iter().enumerate() {
                let sigma_max = spectrum.singular_values[0];
                let steps_below = (depth - layer) as f64;
                // Per letter, two a step: a length-`n` product band of about
                // `n·u·‖S‖_F ≤ n^{3/2}·u·σ₁` and a compression band of the SVD's
                // `max(m, n)·ε·σ₁·√rank` (rows `m ≤ 2n`) plus `σ₁` times the
                // measured orthogonality defect of `V`, about `n²·u`: each within
                // `2n²·ε·σ₁`, so `8n²·ε·σ₁` a step with room.
                let per_step = 8.0 * (width * width) as f64 * f64::EPSILON * sigma_max;
                assert!(
                    spectrum.band <= steps_below * per_step,
                    "layer {layer}: band {:e} against σ₁ {sigma_max:e} after {steps_below} steps",
                    spectrum.band
                );
                assert!(spectrum.resolved_rank >= 3, "layer {layer}: {spectrum:?}");
            }
        }
    }
}
