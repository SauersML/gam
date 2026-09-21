//! The support term's exact symmetries, its periodic phase orbits, and the
//! Newton–Kantorovich certificates taken on the slice transverse to them (#2576,
//! #3258), moved out of `support_term.rs` to keep that tracked file under the #780
//! 10k-line gate.
//!
//! This is a child module of `support_term`, not a sibling: the certificate reads that
//! module's private pencil helpers (`support_orthonormal_span`, `support_project_off`,
//! `certify_shifted_identity_pd`) and the term's private methods, which a sibling could
//! not see. `use super::*` therefore carries the parent's whole namespace in, and the four
//! entry points the parent and its tests call back into are `pub(super)`.

use super::*;

/// One periodic atom's phase orbit at an installed state
/// ([`SaeSupportSparseTerm::support_phase_orbits`], #3258).
#[derive(Debug, Clone)]
pub(super) struct SupportPhaseOrbit {
    atom: usize,
    /// The atom's ARD precision `α` and the prior's `κ = 2π/P`.
    alpha: f64,
    kappa: f64,
    /// `ρ = |Σᵢ e^{iκtᵢ}|`, its rounding band, and its lower bound over the certificate ball.
    resultant: f64,
    resultant_band: f64,
    resultant_lower: f64,
    /// `φ = arg Σᵢ e^{iκtᵢ}` in `(−π, π]`, and the resultant's components `ρcos φ`, `ρsin φ`.
    phase: f64,
    cosine: f64,
    sine: f64,
    /// `(pencil index, cos κtᵢ, sin κtᵢ)` per routed slot.
    slots: Vec<(usize, f64, f64)>,
    /// `ξ` over the full pencil.
    generator: Array1<f64>,
    /// A bound on `‖ξ‖` over the certificate ball and the orbits through it.
    generator_reach: f64,
}

/// The phase-profiled certificate's outcome for one profiled set (#3258): its verdict, or
/// the candidate orbits (by index) whose Rayleigh quotient on the profiled slice is at or
/// below the shift that set needs, so the certificate at that set cannot hold.
enum SupportPhaseProfile {
    Verdict(SupportKantorovichVerdict),
    Widen(Vec<usize>),
}

/// The unprojected pencil one profiled set re-slices: the exact Hessian, the gradient and
/// the Newton step over the coordinates the pencil keeps, and the exact symmetry
/// generators projected off before the phase generators are appended. The four always
/// travel together and are always rebuilt together, so the certificate takes one pencil
/// rather than four positional arrays.
struct SupportPhasePencil {
    exact: Array2<f64>,
    gradient: Array1<f64>,
    step: Array1<f64>,
    generators: Vec<Array1<f64>>,
}

/// The bands the profiled certificate is measured against. All four are fixed before the
/// widening loop starts and are the same at every profiled set, so they are read once at
/// the state that produced them rather than threaded one scalar at a time.
#[derive(Debug, Clone, Copy)]
struct SupportPhaseBands {
    /// The gradient's rounding band at the installed state.
    gradient_band: f64,
    /// The exact Hessian's Lipschitz constant on the certificate ball.
    lipschitz: f64,
    /// The radius the shift is aimed inside, `bound` less the rounding of the formulas.
    aim: f64,
    /// The ball the verdict is claimed on.
    bound: f64,
}

/// The Newton–Kantorovich slice transverse to `generators` (#2576): the orthonormal basis
/// `V` of their span, and the Hessian `(I − P)A(I − P)`, gradient `(I − P)g` and step
/// `(I − P)Δ` with `P = VVᵀ`.
fn support_slice(
    mut exact: Array2<f64>,
    gradient: &Array1<f64>,
    step: &Array1<f64>,
    generators: &[Array1<f64>],
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>), String> {
    let symmetry = support_orthonormal_span(exact.nrows(), generators)?;
    let gradient = support_project_off(&symmetry, gradient.view());
    let step = support_project_off(&symmetry, step.view());
    if symmetry.ncols() > 0 {
        // `(I − P)A(I − P) = A − V·W − Wᵀ·Vᵀ + V·(W·V)·Vᵀ` with `W = VᵀA`, formed in place.
        let projected_rows = symmetry.t().dot(&exact);
        let corner = projected_rows.dot(&symmetry);
        ndarray::linalg::general_mat_mul(-1.0, &symmetry, &projected_rows, 1.0, &mut exact);
        ndarray::linalg::general_mat_mul(-1.0, &projected_rows.t(), &symmetry.t(), 1.0, &mut exact);
        let lifted = symmetry.dot(&corner);
        ndarray::linalg::general_mat_mul(1.0, &lifted, &symmetry.t(), 1.0, &mut exact);
    }
    Ok((exact, gradient, step, symmetry))
}

/// The Kantorovich solve error `e = ‖g − AΔ‖ + γ_{n+1}·‖|A||Δ| + |g|‖ + ε_g` of the
/// sliced step (the middle term bounds the rounding of forming the residual densely), and
/// `‖Δ‖`.
fn support_slice_error(
    exact: &Array2<f64>,
    gradient: &Array1<f64>,
    step: &Array1<f64>,
    gradient_band: f64,
) -> Result<(f64, f64), String> {
    let applied = exact.dot(step);
    let magnitude = exact.mapv(f64::abs).dot(&step.mapv(f64::abs)) + gradient.mapv(f64::abs);
    let residual = gradient - &applied;
    let error = residual.dot(&residual).sqrt()
        + gam_linalg::roundoff::accumulation_growth(exact.nrows() + 1)
            * magnitude.dot(&magnitude).sqrt()
        + gradient_band;
    let step_norm = step.dot(step).sqrt();
    if !(error.is_finite() && step_norm.is_finite()) {
        return Err(format!(
            "support Newton-Kantorovich certificate: non-finite step {step_norm:e} or solve \
             error {error:e}"
        ));
    }
    Ok((error, step_norm))
}

/// The smallest shift `μ*` whose certified `A − μ*·I ≻ 0` gives a Kantorovich radius at
/// most `aim` (`‖Δ‖ < aim`), raised by `γ₈`, the rounding of its own formula
/// ([`SaeSupportSparseTerm::support_kantorovich_certificate`] derives it).
fn support_kantorovich_shift(lipschitz: f64, step_norm: f64, error: f64, aim: f64) -> f64 {
    let computed = if lipschitz > 0.0 {
        let half_curvature = lipschitz * step_norm
            + (lipschitz * lipschitz * step_norm * step_norm + 2.0 * lipschitz * error).sqrt();
        if half_curvature <= lipschitz * aim {
            half_curvature
        } else {
            (lipschitz * aim * aim + 2.0 * error) / (2.0 * (aim - step_norm))
        }
    } else {
        error / (aim - step_norm)
    };
    computed * (1.0 + gam_linalg::roundoff::accumulation_growth(8))
}

/// `(η, t*)` of a certified shift: `η = ‖Δ‖ + e/μ` and `t* = (1 − √(1 − 2h))·μ/L`,
/// `h = Lη/μ`, rationalized so a small `h` does not cancel. With no solve or rounding
/// error at all (an exactly zero gradient and step) the shift is zero, `A ≻ 0` was
/// certified outright, and `η = ‖Δ‖`.
fn support_kantorovich_radius(
    lipschitz: f64,
    shift: f64,
    step_norm: f64,
    error: f64,
) -> (f64, f64) {
    let eta = if error > 0.0 { step_norm + error / shift } else { step_norm };
    let radius = if lipschitz > 0.0 && eta > 0.0 {
        let h = lipschitz / shift * eta;
        2.0 * eta / (1.0 + (1.0 - 2.0 * h).max(0.0).sqrt())
    } else {
        eta
    };
    (eta, radius)
}

/// `A + σ·VVᵀ` with `σ = μ + max Aᵢᵢ`: any stiffness above `μ` on the span of `V` leaves
/// `A − μ·I ≻ 0` on the slice as the question, and `μ` plus the largest diagonal entry
/// keeps the factored matrix at `A`'s own scale.
fn support_add_orbit_stiffness(exact: &mut Array2<f64>, symmetry: &Array2<f64>, shift: f64) {
    if symmetry.ncols() == 0 {
        return;
    }
    let stiffness = shift + exact.diag().iter().copied().fold(0.0_f64, f64::max);
    ndarray::linalg::general_mat_mul(stiffness, symmetry, &symmetry.t(), 1.0, exact);
}

/// Whether `generator`'s component `v` off the span of `symmetry` has
/// `vᵀ·sliced·v ≤ shift·‖v‖²` (#3258): then `sliced − shift·I ≻ 0` on that slice is false,
/// whatever stiffness is added on the span, so no certificate at `shift` can hold there.
fn support_rayleigh_at_or_below(
    sliced: &Array2<f64>,
    symmetry: &Array2<f64>,
    generator: &Array1<f64>,
    shift: f64,
) -> bool {
    let direction = support_project_off(symmetry, generator.view());
    let norm_squared = direction.dot(&direction);
    norm_squared > 0.0 && sliced.dot(&direction).dot(&direction) <= shift * norm_squared
}

impl SaeSupportSparseTerm {
    /// The exact continuous symmetries of the penalized support objective at the
    /// installed state, as tangent vectors in the pencil layout (the compact coordinates,
    /// then every decoder block in `beta_layout` order), #2576.
    ///
    /// A direction along which the objective is exactly invariant makes `A` singular at
    /// every stationary point, so no `A − μ·I ≻ 0` exists there and the minimum is an
    /// orbit, not a point. The Newton–Kantorovich certificate is then taken on a slice
    /// transverse to those orbits ([`Self::support_kantorovich_certificate`]). The families
    /// below are admitted only where their invariance holds exactly, by bitwise equalities
    /// on the state; a symmetry they do not name leaves `A` singular, and the certificate
    /// refuses there instead of certifying.
    ///
    /// A `Linear` atom of dimension `d ≥ 2` decodes `c₀ + uᵀC` through the basis
    /// `[1, u₁, …, u_d]`. Rotating its rows' coordinates and its slope rows together,
    /// `u ↦ Ru`, `C ↦ RC` with `R ∈ O(d)`, leaves every decode unchanged. The coordinate ARD
    /// energy `½Σ α_a u_a²` is unchanged when every axis has the same `α`, and the smoothing
    /// energy `½λ tr(BᵀSB)` when `S` has no constant–slope coupling and a scalar slope
    /// block. Each generator `E_jk = e_k e_jᵀ − e_j e_kᵀ`, `j < k`, moves `u` by `E_jk u` on
    /// every row of the atom and `C` by `E_jk C`. The basis is checked against `[1, u]` and
    /// its unit Jacobian at the atom's own rows.
    ///
    /// Atoms routed to rows whose basis carries an unpenalized constant column (exactly `1`,
    /// zero jet, a zero row and column of `S`) can trade constant among themselves: moving
    /// `v_a` into every such atom leaves each decode unchanged when the routed atoms of
    /// every row sum to zero, and neither the penalty nor the priors see it. Each null
    /// vector of the integer co-selection Gram `CᵀC` (`C` the row-by-atom incidence), in
    /// each output channel, is such a translation. The noisy-ring pair chart
    /// (`pair_chart_fit_is_certified_reml_on_a_noisy_ring`, five periodic atoms routed two
    /// to a row) has one, and `A` there has two eigenvalues of order `1e-15`, one per
    /// channel.
    ///
    /// An atom no row selects enters the objective only through `½λ βᵀSβ`. Where the null
    /// space of `S` lies on basis axes, those decoder coordinates are exactly zero rows of
    /// the pencil, which the certificate drops; elsewhere `A` stays singular there and the
    /// certificate refuses, as the exact Newton solve does.
    pub(super) fn support_exact_symmetry_generators(
        &self,
        ard_precisions: &[Vec<f64>],
        beta_offsets: &[usize],
        beta_dim: usize,
    ) -> Result<Vec<Array1<f64>>, String> {
        let t_len = self.coordinate_state_len();
        let width = self.output_dim;
        let full_dim = t_len + beta_dim;
        let mut row_starts = Vec::with_capacity(self.n_obs());
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            row_starts.push(cursor);
            cursor += self.assignment.coords_row(row).len();
        }
        if cursor != t_len {
            return Err(format!(
                "support symmetry generators: compact coordinates span {cursor}, expected {t_len}"
            ));
        }
        let slot_start = |row: usize, slot: usize| -> usize {
            row_starts[row]
                + self.assignment.support_indices(row)[..slot]
                    .iter()
                    .map(|&atom| self.assignment.atom_coord_dim(atom as usize))
                    .sum::<usize>()
        };
        let mut generators = Vec::new();
        for atom in 0..self.k_atoms() {
            let penalty = self.atoms[atom].smooth_penalty();
            if self.atom_rows[atom].is_empty() {
                continue;
            }
            let d = self.assignment.atom_coord_dim(atom);
            if self.atoms[atom].basis_kind() != &SaeAtomBasisKind::Linear
                || d < 2
                || self.atoms[atom].basis_size() != d + 1
                || penalty.dim() != (d + 1, d + 1)
            {
                continue;
            }
            let alpha = &ard_precisions[atom];
            if alpha.len() != d
                || alpha
                    .iter()
                    .any(|value| value.to_bits() != alpha[0].to_bits())
                || self.atom_ard_axis_periods(atom).iter().any(Option::is_some)
            {
                continue;
            }
            let slope_ridge = penalty[[1, 1]];
            let rotation_invariant_penalty = (1..=d).all(|j| {
                penalty[[0, j]] == 0.0
                    && penalty[[j, 0]] == 0.0
                    && (1..=d).all(|k| penalty[[j, k]] == if j == k { slope_ridge } else { 0.0 })
            });
            if !rotation_invariant_penalty {
                continue;
            }
            let Some(evaluator) = self.atoms[atom].basis_second_jet.as_ref() else {
                continue;
            };
            let rows = &self.atom_rows[atom];
            let mut coords = Array2::<f64>::zeros((rows.len(), d));
            for (index, &(row, slot)) in rows.iter().enumerate() {
                coords.row_mut(index).assign(&ndarray::ArrayView1::from(
                    self.assignment.coords_for_slot(row, slot),
                ));
            }
            let (phi, jet) = evaluator.evaluate(coords.view())?;
            let affine = phi.dim() == (rows.len(), d + 1)
                && jet.dim() == (rows.len(), d + 1, d)
                && (0..rows.len()).all(|index| {
                    phi[[index, 0]] == 1.0
                        && (0..d).all(|axis| {
                            jet[[index, 0, axis]] == 0.0
                                && phi[[index, 1 + axis]] == coords[[index, axis]]
                                && (0..d).all(|other| {
                                    jet[[index, 1 + axis, other]]
                                        == if axis == other { 1.0 } else { 0.0 }
                                })
                        })
                });
            if !affine {
                continue;
            }
            let decoder = self.atoms[atom].decoder_coefficients();
            for j in 0..d {
                for k in (j + 1)..d {
                    let mut generator = Array1::<f64>::zeros(full_dim);
                    for &(row, slot) in rows {
                        let start = slot_start(row, slot);
                        let u = self.assignment.coords_for_slot(row, slot);
                        generator[start + j] = -u[k];
                        generator[start + k] = u[j];
                    }
                    for channel in 0..width {
                        generator[t_len + beta_offsets[atom] + (1 + j) * width + channel] =
                            -decoder[[1 + k, channel]];
                        generator[t_len + beta_offsets[atom] + (1 + k) * width + channel] =
                            decoder[[1 + j, channel]];
                    }
                    generators.push(generator);
                }
            }
        }
        // Constant redistribution. An atom whose basis has a column that is exactly `1`
        // with zero jet at its rows, and whose penalty leaves that column unpenalized,
        // contributes its constant `c_a` to every row it is routed to. Moving `v_a` of
        // constant between such atoms leaves every decode unchanged exactly when every
        // row's routed members sum to zero, `Cv = 0` for the row-by-member incidence `C`,
        // and the penalty and priors do not see it: an exact translation symmetry, one per
        // null vector of the integer co-selection Gram `CᵀC` and output channel.
        let mut members: Vec<(usize, usize)> = Vec::new();
        for atom in 0..self.k_atoms() {
            let rows = &self.atom_rows[atom];
            let Some(evaluator) = self.atoms[atom].basis_evaluator.as_ref() else {
                continue;
            };
            if rows.is_empty() {
                continue;
            }
            let d = self.assignment.atom_coord_dim(atom);
            let mut coords = Array2::<f64>::zeros((rows.len(), d));
            for (index, &(row, slot)) in rows.iter().enumerate() {
                coords.row_mut(index).assign(&ndarray::ArrayView1::from(
                    self.assignment.coords_for_slot(row, slot),
                ));
            }
            let (phi, jet) = evaluator.evaluate(coords.view())?;
            let penalty = self.atoms[atom].smooth_penalty();
            let constant = (0..phi.ncols()).find(|&column| {
                column < penalty.nrows()
                    && phi.column(column).iter().all(|value| *value == 1.0)
                    && jet
                        .slice(ndarray::s![.., column, ..])
                        .iter()
                        .all(|value| *value == 0.0)
                    && penalty.row(column).iter().all(|value| *value == 0.0)
                    && penalty.column(column).iter().all(|value| *value == 0.0)
            });
            if let Some(column) = constant {
                members.push((atom, column));
            }
        }
        if !members.is_empty() {
            let position: std::collections::HashMap<usize, usize> = members
                .iter()
                .enumerate()
                .map(|(index, &(atom, _))| (atom, index))
                .collect();
            let mut coselection = Array2::<f64>::zeros((members.len(), members.len()));
            for row in 0..self.n_obs() {
                let routed: Vec<usize> = self
                    .assignment
                    .support_indices(row)
                    .iter()
                    .filter_map(|&atom| position.get(&(atom as usize)).copied())
                    .collect();
                for &first in &routed {
                    for &second in &routed {
                        coselection[[first, second]] += 1.0;
                    }
                }
            }
            let (values, vectors) = coselection.eigh(Side::Lower).map_err(|error| {
                format!("support symmetry generators: co-selection spectrum: {error}")
            })?;
            let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&values.to_vec());
            for (mode, &value) in values.iter().enumerate() {
                if value > band {
                    continue;
                }
                for channel in 0..width {
                    let mut generator = Array1::<f64>::zeros(full_dim);
                    for (index, &(atom, column)) in members.iter().enumerate() {
                        generator[t_len + beta_offsets[atom] + column * width + channel] =
                            vectors[[index, mode]];
                    }
                    generators.push(generator);
                }
            }
        }
        Ok(generators)
    }

    /// The phase orbit of every periodic atom whose phase only the periodic ARD prior
    /// selects (#3258).
    ///
    /// A one-dimensional periodic atom's harmonic basis
    /// `[1, sin(ω₁t), cos(ω₁t), …, sin(ω_H t), cos(ω_H t)]`, `ω_h = 2πh/P`, `P` its period,
    /// turns a common shift `t ↦ t + δ` of every routed coordinate into a rotation of each
    /// harmonic pair of decoder rows, and the counter-rotation `B_s ↦ B_s + δω_h B_c`,
    /// `B_c ↦ B_c − δω_h B_s` (to first order) leaves every decode unchanged. A roughness
    /// Gram that commutes with those rotations exactly (diagonal, with bitwise equal entries
    /// on each pair) does not see them either, so the whole objective moves
    /// along the orbit only through the atom's ARD energy
    /// `E(t) = (α/κ²) Σᵢ (1 − cos κtᵢ)`, `κ = 2π/P`. The orbit's generator `ξ` is `1` on
    /// every routed coordinate and `(ω_h B_c, −ω_h B_s)` on the harmonic decoder rows.
    ///
    /// Along the orbit `E(t + δ) = (α/κ²)(n − ρ cos(κδ + φ))`, `ρe^{iφ} = Σᵢ e^{iκtᵢ}`, so the
    /// orbit's curvature is `αρ`: a weak prior on a well-spread ring pins the phase no
    /// harder than `αρ`, however well the data resolve everything else. An atom is returned
    /// only where that minimum is isolated over the whole certificate ball: `ρ` is above
    /// its rounding band and its largest change over `B(θ, bound)`, `κ√n·bound`, since
    /// `|∂ρ/∂tᵢ| ≤ κ`.
    pub(super) fn support_phase_orbits(
        &self,
        ard_precisions: &[Vec<f64>],
        beta_offsets: &[usize],
        beta_dim: usize,
        bound: f64,
    ) -> Result<Vec<SupportPhaseOrbit>, String> {
        let t_len = self.coordinate_state_len();
        let width = self.output_dim;
        let full_dim = t_len + beta_dim;
        let mut row_starts = Vec::with_capacity(self.n_obs());
        let mut cursor = 0usize;
        for row in 0..self.n_obs() {
            row_starts.push(cursor);
            cursor += self.assignment.coords_row(row).len();
        }
        let mut orbits = Vec::new();
        for atom in 0..self.k_atoms() {
            let rows = &self.atom_rows[atom];
            let basis_size = self.atoms[atom].basis_size();
            if self.atoms[atom].basis_kind() != &SaeAtomBasisKind::Periodic
                || self.assignment.atom_coord_dim(atom) != 1
                || rows.is_empty()
                || basis_size < 3
                || basis_size % 2 == 0
            {
                continue;
            }
            let alpha = ard_precisions[atom][0];
            let (Some(period), Some(prior_period)) =
                (self.atom_axis_periods(atom)[0], self.atom_ard_axis_periods(atom)[0])
            else {
                continue;
            };
            // The shift that holds every decode closes the prior's circle only when the
            // chart, the prior and the harmonic basis share one period.
            if !(alpha > 0.0
                && period.to_bits() == prior_period.to_bits()
                && period.to_bits()
                    == crate::manifold::compact_orbit::PERIODIC_HARMONIC_BASIS_PERIOD.to_bits())
            {
                continue;
            }
            if !support_penalty_commutes_with_harmonic_rotations(
                self.atoms[atom].smooth_penalty(),
                basis_size,
            ) {
                continue;
            }
            let kappa = std::f64::consts::TAU / prior_period;
            let mut slots = Vec::with_capacity(rows.len());
            let (mut cosine, mut sine, mut cosine_absolute, mut sine_absolute) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let mut generator = Array1::<f64>::zeros(full_dim);
            for &(row, slot) in rows {
                let index = row_starts[row]
                    + self.assignment.support_indices(row)[..slot]
                        .iter()
                        .map(|&other| self.assignment.atom_coord_dim(other as usize))
                        .sum::<usize>();
                // The prior's own phase formula, so `α·cos κt` cancels its Hessian entry.
                let (sin, cos) = (kappa * self.assignment.coords_for_slot(row, slot)[0]).sin_cos();
                cosine += cos;
                sine += sin;
                cosine_absolute += cos.abs();
                sine_absolute += sin.abs();
                slots.push((index, cos, sin));
                generator[index] = 1.0;
            }
            let count = rows.len();
            let resultant = sine.hypot(cosine);
            // As in `profile_periodic_phase_origins`: `γ_{n+2}` of the components' sums.
            let resultant_band = gam_linalg::roundoff::accumulation_growth(count + 2)
                * sine_absolute.hypot(cosine_absolute);
            let resultant_lower = resultant - resultant_band - kappa * (count as f64).sqrt() * bound;
            if !(resultant_lower > 0.0) {
                continue;
            }
            let decoder = self.atoms[atom].decoder_coefficients();
            let harmonics = (basis_size - 1) / 2;
            for harmonic in 1..=harmonics {
                let frequency = kappa * harmonic as f64;
                for channel in 0..width {
                    generator[t_len + beta_offsets[atom] + (2 * harmonic - 1) * width + channel] =
                        frequency * decoder[[2 * harmonic, channel]];
                    generator[t_len + beta_offsets[atom] + 2 * harmonic * width + channel] =
                        -frequency * decoder[[2 * harmonic - 1, channel]];
                }
            }
            // `ξ` is linear in the decoder with norm `max_h ω_h = Hκ`, and a rotation keeps
            // its norm along the orbit, so `‖ξ‖ ≤ ‖ξ(θ)‖ + Hκ·bound` on the ball and its orbits.
            let generator_reach = generator.dot(&generator).sqrt()
                * (1.0 + gam_linalg::roundoff::accumulation_growth(full_dim + 2))
                + kappa * harmonics as f64 * bound;
            orbits.push(SupportPhaseOrbit {
                atom,
                alpha,
                kappa,
                resultant,
                resultant_band,
                resultant_lower,
                phase: sine.atan2(cosine),
                cosine,
                sine,
                slots,
                generator,
                generator_reach,
            });
        }
        Ok(orbits)
    }

    /// The inner certificate (#2576, #2933 F08): the Newton–Kantorovich theorem, with
    /// each hypothesis established at the installed state `θ`, on a slice transverse to
    /// the objective's exact symmetry orbits.
    ///
    /// Kantorovich (Ortega–Rheinboldt 12.6.2): let `‖A(x) − A(y)‖₂ ≤ L‖x − y‖₂` on
    /// `B(θ, r)`, `‖A(θ)⁻¹‖₂ ≤ β`, `‖A(θ)⁻¹g(θ)‖₂ ≤ η` and `h = βLη ≤ ½`. Then `g` has a
    /// zero `θ*` with `‖θ − θ*‖₂ ≤ t* = (1 − √(1 − 2h))/(βL)` whenever `t* ≤ r`, unique in
    /// that ball. With `A(θ) ⪰ μ·I`, every `x` in `B(θ, t*)` has
    /// `A(x) ⪰ (μ − L·t*)·I ⪰ 0` because `t* ≤ 1/(βL) ≤ μ/L`, so the objective is convex
    /// on that ball and `θ*` minimizes it there. Every parameter is then within `t*` of
    /// the minimum, so the certificate asks `t* ≤ bound`, the iterate scale times the
    /// tolerance.
    ///
    /// Where the objective is exactly invariant along the flows of the generators
    /// [`Self::support_exact_symmetry_generators`] returns, `A` is singular at every
    /// stationary point and the minimum is an orbit. The theorem is then applied to the
    /// objective restricted to the slice `θ + N⊥`, `N` the generators' span and `P` its
    /// orthogonal projector: Hessian `(I − P)A(I − P)`, gradient `(I − P)g`, and `L` still
    /// bounds its Hessian's variation because the projection has norm one. It gives the
    /// slice's unique minimum `θ*` within `t*`. There `(I − P)g(θ*) = 0`, and invariance
    /// makes `g(θ*)` orthogonal to the generators at `θ*`, which span a complement of
    /// `N⊥` that close to `θ`, so `g(θ*) = 0`. Every nearby state lies on the orbit of a
    /// slice state and the objective is constant on orbits (the slice theorem), so the
    /// orbit of `θ*` is a local minimum and every parameter is within `t*` of it.
    /// `(I − P)A(I − P) ≻ μ` on `N⊥` is certified as `(I − P)A(I − P) + σ·P − μ·I ≻ 0` with
    /// `σ` above `μ`, whose eigenvalues on `N` are `σ − μ > 0`. Without generators `P = 0`
    /// and this is the theorem as stated.
    ///
    /// The hypotheses, at `θ`, over the pencil's non-null coordinates (a coordinate whose
    /// rows of both `A` and the majorizer are exactly zero is one the objective does not
    /// depend on, so its gradient and step must be exactly zero too):
    /// - `L` on `B(θ, bound)` ([`Self::support_hessian_lipschitz_bound`]).
    /// - `η`: the computed step `Δ` solves `AΔ = g − ρ`, and the computed `g` is within
    ///   its rounding band `ε_g` of the exact gradient
    ///   ([`Self::gradient_rounding_band`]), so `η ≤ ‖Δ‖ + β·e` with
    ///   `e = ‖ρ‖ + γ_{n+1}·‖|A||Δ| + |g|‖ + ε_g`, where the middle term bounds the
    ///   rounding of forming `ρ = g − AΔ` densely; on the slice every one of these is
    ///   projected, and the projected gradient is within `ε_g` too.
    /// - `β ≤ 1/μ` from a certified `A − μ·I ≻ 0` ([`certify_shifted_identity_pd`]).
    ///
    /// Take `β = 1/μ`, so `u = βL = L/μ` and `η = ‖Δ‖ + e/μ`, both falling as `μ` grows;
    /// the smallest `μ` meeting both conditions is the one to certify. `h ≤ ½` is
    /// `μ² ≥ 2L(‖Δ‖μ + e)`, i.e. `μ ≥ μ₀ = L‖Δ‖ + √(L²‖Δ‖² + 2Le)`, where `t* = 1/u =
    /// μ₀/L`. So `μ* = μ₀` when `μ₀ ≤ L·bound`. Otherwise `u·bound < 1`, and there
    /// `t* ≤ bound` holds exactly when `u ≤ 2(bound − η)/bound²`. That gives
    /// `μ* = μ₁ = (L·bound² + 2e)/(2(bound − ‖Δ‖))`, which is at least `μ₀` because
    /// `bound² ≥ 4η(bound − η)`. Without curvature variation (`L = 0`), `μ* = e/(bound −
    /// ‖Δ‖)` makes `η ≤ bound`. Every case needs `‖Δ‖ < bound`, which is read first, before
    /// any dense work. `μ*` is computed against `bound·(1 − γ₁₆)` and raised by `γ₈`, the
    /// rounding of its own formula, so it is at least the exact threshold and the radius
    /// computed from it is at most `bound`. A single dense Cholesky then decides the
    /// certificate.
    /// It also certifies that `A` has no negative curvature off the orbits. A failed
    /// factorization is a refusal to certify, not a claim that `A` is indefinite: the
    /// caller audits the curvature and steps. The assembled `A` and the computed
    /// projector are taken as the Hessian and the orthogonal projector, as the curvature
    /// audit and the Newton solve take `A`.
    pub(super) fn support_kantorovich_certificate(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        displacement: &SaeSupportNewtonDisplacement,
        bound: f64,
    ) -> Result<SupportKantorovichVerdict, String> {
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let generators =
            self.support_exact_symmetry_generators(ard_precisions, &beta_offsets, beta_dim)?;
        let phase_orbits =
            self.support_phase_orbits(ard_precisions, &beta_offsets, beta_dim, bound)?;
        self.support_kantorovich_certificate_on_slice(
            target,
            lambda_smooth,
            ard_precisions,
            displacement,
            bound,
            &generators,
            &phase_orbits,
        )
    }

    /// [`Self::support_kantorovich_certificate`] on the slice transverse to `generators`,
    /// the exact symmetry directions it projects off (#2576). A symmetry of the objective
    /// missing from `generators` leaves `A` singular on that slice, so no
    /// `A − μ·I ≻ 0` is certified there and the verdict is a refusal: a symmetry this
    /// certificate is not given is never certified over.
    pub(super) fn support_kantorovich_certificate_on_slice(
        &self,
        target: ArrayView2<'_, f64>,
        lambda_smooth: &[f64],
        ard_precisions: &[Vec<f64>],
        displacement: &SaeSupportNewtonDisplacement,
        bound: f64,
        generators: &[Array1<f64>],
        phase_orbits: &[SupportPhaseOrbit],
    ) -> Result<SupportKantorovichVerdict, String> {
        if !(bound.is_finite() && bound > 0.0) {
            return Err(format!(
                "support Newton-Kantorovich certificate: the bound must be finite and positive, \
                 got {bound}"
            ));
        }
        let full_dim = self.admitted_dense_pencil_dim()?;
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let t_len = self.coordinate_state_len();
        if t_len + beta_dim != full_dim
            || displacement.coordinates.len() != t_len
            || displacement.decoder.len() != beta_dim
        {
            return Err(format!(
                "support Newton-Kantorovich certificate: displacement ({}, {}) does not match \
                 the pencil ({t_len}, {beta_dim})",
                displacement.coordinates.len(),
                displacement.decoder.len(),
            ));
        }
        let mut step = Array1::<f64>::zeros(full_dim);
        step.slice_mut(ndarray::s![..t_len])
            .assign(&displacement.coordinates);
        step.slice_mut(ndarray::s![t_len..])
            .assign(&displacement.decoder);
        // `‖Δ‖ < bound` on the slice, read before any dense work. No generator carries
        // weight on a coordinate the pencil drops, so this is the norm checked below.
        let screen = support_orthonormal_span(full_dim, generators)?;
        let screened = support_project_off(&screen, step.view());
        let projected_norm = screened.dot(&screened).sqrt();
        // Each shift formula rounds at most eight times and the radius formula as often, so
        // the shift is raised by `γ₈` over its computed value ([`support_kantorovich_shift`]),
        // which then bounds the exact threshold, and aimed at `bound·(1 − γ₁₆)`, so the
        // radius computed from it stays at or below `bound`.
        let aim = bound * (1.0 - gam_linalg::roundoff::accumulation_growth(16));
        if !(projected_norm < aim) {
            return Ok(SupportKantorovichVerdict::NotCertified(format!(
                "Newton displacement off {} exact symmetry directions ‖Δ‖₂ = \
                 {projected_norm:.6e} is not below the bound {bound:.6e}",
                screen.ncols()
            )));
        }
        let system = self.assemble_arrow_schur(target, lambda_smooth, ard_precisions)?;
        if *system.row_offsets.last().unwrap_or(&0) != t_len {
            return Err(format!(
                "support Newton-Kantorovich certificate: the arrow system's coordinates span {}, \
                 the state's {t_len}",
                system.row_offsets.last().unwrap_or(&0)
            ));
        }
        let mut gradient = Array1::<f64>::zeros(full_dim);
        for (row, block) in system.rows.iter().enumerate() {
            gradient
                .slice_mut(ndarray::s![
                    system.row_offsets[row]..system.row_offsets[row + 1]
                ])
                .assign(&block.gt);
        }
        gradient.slice_mut(ndarray::s![t_len..]).assign(&system.gb);
        let rows = self.support_outer_differential_rows(target, ard_precisions, &beta_offsets)?;
        let (exact, majorizer) =
            self.support_outer_dense_hessian_matrices(&system, &rows, t_len, beta_dim)?;
        let kept: Vec<usize> = (0..full_dim)
            .filter(|&index| {
                !(exact.row(index).iter().all(|value| *value == 0.0)
                    && majorizer.row(index).iter().all(|value| *value == 0.0))
            })
            .collect();
        drop(majorizer);
        if let Some(index) = (0..full_dim)
            .filter(|index| kept.binary_search(index).is_err())
            .find(|&index| gradient[index] != 0.0 || step[index] != 0.0)
        {
            return Ok(SupportKantorovichVerdict::NotCertified(format!(
                "coordinate {index} has exactly zero curvature rows but gradient {:.3e} and step \
                 {:.3e}",
                gradient[index], step[index]
            )));
        }
        let (exact, gradient, step, generators) = if kept.len() == full_dim {
            (exact, gradient, step, generators.to_vec())
        } else {
            (
                exact
                    .select(ndarray::Axis(0), &kept)
                    .select(ndarray::Axis(1), &kept),
                gradient.select(ndarray::Axis(0), &kept),
                step.select(ndarray::Axis(0), &kept),
                generators
                    .iter()
                    .map(|generator| generator.select(ndarray::Axis(0), &kept))
                    .collect(),
            )
        };
        // A phase orbit is profiled only over coordinates the pencil keeps: its generator's
        // support and its slots, re-indexed into the kept pencil. `kept` is the ascending
        // list of kept coordinates, so a coordinate's rank in it IS its index in the kept
        // pencil, and the coordinate is kept exactly when that rank names it. Asking
        // `binary_search` and reading only its `Ok` arm throws away the one thing its
        // `Err` arm carries — an insertion point no orbit may reference — so the rank is
        // taken directly and the membership test is the comparison that follows it.
        let kept_index = |index: usize| {
            let rank = kept.partition_point(|&coordinate| coordinate < index);
            (kept.get(rank) == Some(&index)).then_some(rank)
        };
        let candidates: Vec<SupportPhaseOrbit> = phase_orbits
            .iter()
            .filter_map(|orbit| {
                let slots = orbit
                    .slots
                    .iter()
                    .map(|&(index, cos, sin)| kept_index(index).map(|kept| (kept, cos, sin)))
                    .collect::<Option<Vec<_>>>()?;
                let lost = orbit
                    .generator
                    .iter()
                    .enumerate()
                    .any(|(index, value)| *value != 0.0 && kept_index(index).is_none());
                (!lost).then(|| SupportPhaseOrbit {
                    slots,
                    generator: orbit.generator.select(ndarray::Axis(0), &kept),
                    ..orbit.clone()
                })
            })
            .collect();
        let gradient_band = self.gradient_rounding_band(target, lambda_smooth, ard_precisions)?;
        // The unprojected pencil is kept only while an orbit may still be profiled.
        let unprojected = (!candidates.is_empty())
            .then(|| (exact.clone(), gradient.clone(), step.clone(), generators.clone()));
        let (sliced, sliced_gradient, sliced_step, symmetry) =
            support_slice(exact, &gradient, &step, &generators)?;
        let (error, step_norm) =
            support_slice_error(&sliced, &sliced_gradient, &sliced_step, gradient_band)?;
        if !(step_norm < aim) {
            return Ok(SupportKantorovichVerdict::NotCertified(format!(
                "Newton displacement off {} exact symmetry directions ‖Δ‖₂ = {step_norm:.6e} is \
                 not below the bound {bound:.6e}",
                symmetry.ncols()
            )));
        }
        let lipschitz = match self.support_hessian_lipschitz_bound(target, ard_precisions, bound)? {
            SupportHessianLipschitz::Bounded(lipschitz) => lipschitz,
            SupportHessianLipschitz::Unavailable(reason) => {
                return Ok(SupportKantorovichVerdict::NotCertified(format!(
                    "the exact Hessian has no Lipschitz bound on the ball of radius {bound:.6e}: \
                     {reason}"
                )));
            }
        };
        let shift = support_kantorovich_shift(lipschitz, step_norm, error, aim);
        // #3258: a phase orbit whose unit direction `v̂` off the slice has
        // `v̂ᵀ(I − P)A(I − P)v̂ ≤ μ` makes `A − μ·I ≻ 0` on the slice false, so the
        // certificate above cannot hold at this iterate; those orbits, and only those, are
        // profiled out.
        let mut chosen: Vec<bool> = candidates
            .iter()
            .map(|orbit| support_rayleigh_at_or_below(&sliced, &symmetry, &orbit.generator, shift))
            .collect();
        if !chosen.contains(&true) {
            let mut stiffened = sliced;
            support_add_orbit_stiffness(&mut stiffened, &symmetry, shift);
            return match certify_shifted_identity_pd(stiffened, shift) {
                Ok(_) => {
                    let (eta, radius) =
                        support_kantorovich_radius(lipschitz, shift, step_norm, error);
                    if !(radius <= bound) {
                        return Ok(SupportKantorovichVerdict::NotCertified(format!(
                            "the certified radius {radius:.17e} exceeds the bound {bound:.17e}"
                        )));
                    }
                    Ok(SupportKantorovichVerdict::Certified {
                        radius,
                        shift,
                        lipschitz,
                        eta,
                        symmetry_directions: symmetry.ncols(),
                        profiled_phases: 0,
                    })
                }
                Err(refusal) => Ok(SupportKantorovichVerdict::NotCertified(format!(
                    "A − μ·I ≻ 0 not certified at μ = {shift:.6e} off {} exact symmetry \
                     directions (L = {lipschitz:.6e}, ‖Δ‖₂ = {step_norm:.6e}, solve and \
                     rounding error {error:.6e}, bound {bound:.6e}): {refusal:?}",
                    symmetry.ncols()
                ))),
            };
        }
        drop(sliced);
        let Some((exact, gradient, step, generators)) = unprojected else {
            return Err(
                "support Newton-Kantorovich certificate: phase orbits profiled without \
                 candidates"
                    .to_string(),
            );
        };
        let bands = SupportPhaseBands {
            gradient_band,
            lipschitz,
            aim,
            bound,
        };
        // Profiling raises the shift the certificate needs (`L_Φ ≥ L`, and the stronger
        // claim aims inside `bound`), so an orbit the direct shift passed can fall at or
        // below the profiled one, where it defeats `A_Φ − μ·I ≻ 0` just as the first ones
        // defeated `A − μ·I ≻ 0`. The profiled set is therefore closed under that test: it
        // only grows, each widening is read off the Rayleigh quotients before any
        // factorization, and it stops within the candidates' count.
        loop {
            let profiled: Vec<SupportPhaseOrbit> = candidates
                .iter()
                .zip(&chosen)
                .filter(|(_, chosen)| **chosen)
                .map(|(orbit, _)| orbit.clone())
                .collect();
            let remaining: Vec<(usize, &SupportPhaseOrbit)> = candidates
                .iter()
                .enumerate()
                .filter(|(index, _)| !chosen[*index])
                .collect();
            match self.support_phase_profiled_certificate(
                SupportPhasePencil {
                    exact: exact.clone(),
                    gradient: gradient.clone(),
                    step: step.clone(),
                    generators: generators.clone(),
                },
                bands,
                &profiled,
                &remaining,
            )? {
                SupportPhaseProfile::Verdict(verdict) => return Ok(verdict),
                SupportPhaseProfile::Widen(indices) => {
                    for index in indices {
                        chosen[index] = true;
                    }
                }
            }
        }
    }

    /// The certificate on the phase-profiled objective (#3258), for the orbits `profiled`
    /// [`Self::support_kantorovich_certificate_on_slice`] found the direct certificate
    /// cannot resolve. `exact`, `gradient` and `step` are the kept pencil's, before any
    /// projection, and `generators` the exact symmetry directions on it.
    ///
    /// Profiling. Only the ARD energy `E` of an orbit's atom moves along the orbit, so
    /// `Φ(y) = min_δ F(g_δ y) = F(y) − E(t) + Ẽ(t)` with `Ẽ = (α/κ²)(n − ρ)`, attained at
    /// `δ* = −φ/κ`. `Ẽ` is invariant along the orbit, so `ξ` joins the exact symmetry
    /// generators of `Φ` and the slice is taken transverse to them all. With
    /// `cᵢ = cos(κtᵢ − φ)`, `sᵢ = sin(κtᵢ − φ)`, `∂ρ/∂tᵢ = −κsᵢ` and `∂φ/∂tᵢ = κcᵢ/ρ`, so
    /// - `∇Φ = g + (α/κ)(s − sin κt)` on the atom's slots,
    /// - `∇²Φ = A + α·diag(c − cos κt) − (α/ρ)ccᵀ` on the atom's slot block.
    ///
    /// The Kantorovich hypotheses for `Φ` on `B(θ, bound)`:
    /// - `L_Φ = L + max_a (α_aκ_a + L_Ẽ,a)`: `α·diag(cos κt)` varies by at most `ακ`, and
    ///   `‖d∇²Ẽ‖ ≤ ακ[1 + 3√n/ρ⁻ + 3n^{3/2}/ρ⁻²]‖dy‖` from `|dcᵢ| ≤ κ(|dtᵢ| + |dφ|)`,
    ///   `|dφ| ≤ κ√n‖dy‖/ρ⁻`, `|dρ| ≤ κ√n‖dy‖` and `‖ccᵀ‖ ≤ n`, with `ρ⁻` the resultant's
    ///   lower bound on the ball; the atoms' slot blocks are disjoint, so the maximum.
    /// - `e`: the residual `∇Φ − ∇²Φ·Δ` of the same displacement, with the gradient band
    ///   raised by the rounding of `sᵢ` (its direction error `band/ρ⁻` times `|cᵢ|` and
    ///   `γ₈` of the formula), and the computed `∇²Φ` within `ε_A` of the exact one in norm,
    ///   `ε_A = γ₃(max|Aᵢᵢ| + αn/ρ⁻) + α[(band/ρ⁻ + γ₈)(1 + 3n/ρ⁻) + n·band/ρ⁻²]`, so
    ///   `A − (μ + ε_A)·I ≻ 0` is what is factored.
    ///
    /// It gives a local minimum orbit `y*` of `Φ` within `t*` of `θ`. Then
    /// `x* = g_{δ*(y*)} y*` minimizes `F` locally, since `F(g_δ y) ≥ Φ(y) ≥ Φ(y*) = F(x*)`
    /// for every `y` near `y*`, and every output invariant under the phases is the same at
    /// `x*` as at `y*`. For the minimum itself, `‖y − g_δ y‖ ≤ |δ|·‖ξ‖` along an orbit,
    /// with `‖ξ‖ ≤ X_a` on the ball ([`Self::support_phase_orbits`]), and
    /// `|δ*_a(y*)| ≤ (|φ_a| + band_a/ρ⁻_a)/κ_a + √n_a·t*/ρ⁻_a`. The atoms move disjoint
    /// coordinates, so
    /// `‖θ − x*‖ ≤ t* + D + t*·G`, `D = ‖((|φ_a| + band_a/ρ⁻_a)X_a/κ_a)_a‖`,
    /// `G = ‖(√n_a X_a/ρ⁻_a)_a‖`.
    ///
    /// The verdict. `A_Φ − μ·I ≻ 0` at the `μ` aimed at `bound` certifies the invariant
    /// outputs; without it the verdict is a refusal. The minimum itself is certified when
    /// `A_Φ − μ·I ≻ 0` also holds at the `μ` aimed at `(bound − D)/(1 + G)`, so that
    /// `t* + D + t*·G ≤ bound`. Otherwise the phases are reported unresolved.
    ///
    /// Closure. `remaining` are the candidate orbits (with their indices) not profiled. Its
    /// atom's block of `A_Φ` is `A`'s, so one whose direction off this slice has Rayleigh
    /// quotient at or below `μ + ε_A` makes `A_Φ − μ·I ≻ 0` false; those indices are
    /// returned as [`SupportPhaseProfile::Widen`] before anything is factored, and the
    /// caller profiles them too.
    fn support_phase_profiled_certificate(
        &self,
        pencil: SupportPhasePencil,
        bands: SupportPhaseBands,
        profiled: &[SupportPhaseOrbit],
        remaining: &[(usize, &SupportPhaseOrbit)],
    ) -> Result<SupportPhaseProfile, String> {
        let SupportPhasePencil {
            mut exact,
            mut gradient,
            step,
            mut generators,
        } = pencil;
        let SupportPhaseBands {
            gradient_band,
            lipschitz,
            aim,
            bound,
        } = bands;
        let formula = gam_linalg::roundoff::accumulation_growth(8);
        let mut profile_band_squared = 0.0_f64;
        let mut hessian_band = 0.0_f64;
        let mut lipschitz_extra = 0.0_f64;
        let mut offset_squared = 0.0_f64;
        let mut growth_squared = 0.0_f64;
        for orbit in profiled {
            let (alpha, kappa, resultant) = (orbit.alpha, orbit.kappa, orbit.resultant);
            let lower = orbit.resultant_lower;
            let count = orbit.slots.len() as f64;
            let angle_band = orbit.resultant_band / lower;
            let centered: Vec<(usize, f64, f64, f64, f64)> = orbit
                .slots
                .iter()
                .map(|&(index, cos, sin)| {
                    (
                        index,
                        cos,
                        sin,
                        (cos * orbit.cosine + sin * orbit.sine) / resultant,
                        (sin * orbit.cosine - cos * orbit.sine) / resultant,
                    )
                })
                .collect();
            let mut diagonal = 0.0_f64;
            for &(index, cos, sin, c, s) in &centered {
                exact[[index, index]] += alpha * (c - cos);
                gradient[index] += (alpha / kappa) * (s - sin);
                let rounding =
                    (alpha / kappa) * (c.abs() * angle_band + formula * (s.abs() + sin.abs()));
                profile_band_squared += rounding * rounding;
            }
            for &(row, _, _, c_row, _) in &centered {
                for &(column, _, _, c_column, _) in &centered {
                    exact[[row, column]] -= (alpha / resultant) * c_row * c_column;
                }
                diagonal = diagonal.max(exact[[row, row]].abs());
            }
            hessian_band = hessian_band.max(
                gam_linalg::roundoff::accumulation_growth(3) * (diagonal + alpha * count / lower)
                    + alpha
                        * ((angle_band + formula) * (1.0 + 3.0 * count / lower)
                            + count * orbit.resultant_band / (lower * lower)),
            );
            lipschitz_extra = lipschitz_extra.max(
                alpha
                    * kappa
                    * (2.0
                        + 3.0 * count.sqrt() / lower
                        + 3.0 * count * count.sqrt() / (lower * lower)),
            );
            let offset = (orbit.phase.abs() + angle_band) / kappa * orbit.generator_reach;
            let growth = count.sqrt() * orbit.generator_reach / lower;
            offset_squared += offset * offset;
            growth_squared += growth * growth;
            generators.push(orbit.generator.clone());
        }
        // Each norm is a sum of `|Π|` squares and a root; the bound rounds a few more times.
        let norm_rounding = 1.0 + gam_linalg::roundoff::accumulation_growth(2 * profiled.len() + 8);
        let phase_offset = offset_squared.sqrt() * norm_rounding;
        let phase_growth = growth_squared.sqrt() * norm_rounding;
        let phase_atoms: Vec<usize> = profiled.iter().map(|orbit| orbit.atom).collect();
        let (sliced, sliced_gradient, sliced_step, symmetry) =
            support_slice(exact, &gradient, &step, &generators)?;
        let (error, step_norm) = support_slice_error(
            &sliced,
            &sliced_gradient,
            &sliced_step,
            gradient_band + profile_band_squared.sqrt(),
        )?;
        let lipschitz = (lipschitz + lipschitz_extra) * (1.0 + formula);
        let refuse = |reason: String| -> Result<SupportPhaseProfile, String> {
            Ok(SupportPhaseProfile::Verdict(SupportKantorovichVerdict::NotCertified(reason)))
        };
        if !(step_norm < aim) {
            return refuse(format!(
                "Newton displacement off {} exact symmetry and phase directions ‖Δ‖₂ = \
                 {step_norm:.6e} is not below the bound {bound:.6e}",
                symmetry.ncols()
            ));
        }
        let orbit_shift = support_kantorovich_shift(lipschitz, step_norm, error, aim);
        // An unprofiled phase orbit at or below the orbit shift on this slice defeats
        // `A_Φ − μ·I ≻ 0` there (its atom's block of `A_Φ` is `A`'s, the slot blocks being
        // disjoint), so the set is widened before anything is factored.
        let widen: Vec<usize> = remaining
            .iter()
            .filter(|(_, orbit)| {
                support_rayleigh_at_or_below(
                    &sliced,
                    &symmetry,
                    &orbit.generator,
                    orbit_shift + hessian_band,
                )
            })
            .map(|(index, _)| *index)
            .collect();
        if !widen.is_empty() {
            return Ok(SupportPhaseProfile::Widen(widen));
        }
        let full_aim = (aim - phase_offset) / (1.0 + phase_growth) * (1.0 - formula);
        let full_shift = (full_aim > step_norm)
            .then(|| support_kantorovich_shift(lipschitz, step_norm, error, full_aim));
        // `μ*` falls as its aim grows, so the full shift is at least the orbit shift and a
        // certificate at it certifies the orbit shift too.
        let full_shift = full_shift.map(|full| full.max(orbit_shift));
        let mut stiffened = sliced;
        support_add_orbit_stiffness(
            &mut stiffened,
            &symmetry,
            full_shift.unwrap_or(orbit_shift) + hessian_band,
        );
        let full_certified = match full_shift {
            Some(full) => {
                certify_shifted_identity_pd(stiffened.clone(), full + hessian_band).is_ok()
            }
            None => false,
        };
        if !full_certified {
            if let Err(refusal) = certify_shifted_identity_pd(stiffened, orbit_shift + hessian_band)
            {
                return refuse(format!(
                    "A_Φ − μ·I ≻ 0 not certified at μ = {orbit_shift:.6e} (+ {hessian_band:.3e}) \
                     off {} exact symmetry and phase directions of atoms {phase_atoms:?} (L_Φ = \
                     {lipschitz:.6e}, ‖Δ‖₂ = {step_norm:.6e}, solve and rounding error \
                     {error:.6e}, bound {bound:.6e}): {refusal:?}",
                    symmetry.ncols()
                ));
            }
        }
        let (orbit_eta, orbit_radius) =
            support_kantorovich_radius(lipschitz, orbit_shift, step_norm, error);
        if !(orbit_radius <= bound) {
            return refuse(format!(
                "the certified phase-profiled radius {orbit_radius:.17e} exceeds the bound \
                 {bound:.17e}"
            ));
        }
        // The minimum itself: the stronger claim, at the shift aimed at `(bound − D)/(1 + G)`.
        if let (true, Some(full)) = (full_certified, full_shift) {
            let (eta, radius) = support_kantorovich_radius(lipschitz, full, step_norm, error);
            let radius = (radius + phase_offset + radius * phase_growth) * norm_rounding;
            if radius <= bound {
                return Ok(SupportPhaseProfile::Verdict(SupportKantorovichVerdict::Certified {
                    radius,
                    shift: full,
                    lipschitz,
                    eta,
                    symmetry_directions: symmetry.ncols(),
                    profiled_phases: profiled.len(),
                }));
            }
        }
        Ok(SupportPhaseProfile::Verdict(SupportKantorovichVerdict::PhaseUnresolved {
            radius: orbit_radius,
            shift: orbit_shift,
            lipschitz,
            eta: orbit_eta,
            symmetry_directions: symmetry.ncols(),
            phase_atoms,
            phase_bound: (phase_offset + orbit_radius * phase_growth) * norm_rounding,
        }))
    }
}
