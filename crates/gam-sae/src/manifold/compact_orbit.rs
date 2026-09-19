//! Exact evidence along a declared compact chart orbit (#2234).
//!
//! A periodic atom whose harmonic decoder closes under differentiation, `∂Φ/∂t = ΦK`, carries
//! a one-parameter orbit `γ_s`: shift every coordinate on the atom's axis by `s` and rotate
//! the decoder, `B → exp(sK)ᵀ`-style, so that every reconstruction is unchanged. The data fit
//! and the per-pair isotropic smoothing are exactly invariant along it. Only the periodic ARD
//! energy `η(1 − cos κt)`, `η = α/κ²`, moves.
//!
//! Laplace prices the orbit direction with the chord curvature `τᵀAτ` and drops `∇f·γ″`. At an
//! inner root accepted at tolerance the dropped connection term dominates the intrinsic orbit
//! curvature, so the chord form reads negative and the evidence refuses a state whose exact
//! orbit sits at its ARD minimum (#2234 stall pin; outer2080 sd1 and acc13). This module
//! integrates the orbit coordinate exactly instead:
//!
//! ```text
//!   ∫ exp(−f) dθ = e^{−f̂}·(2π)^{(d−1)/2}·|N/Φ|^{½}·Π μ_⊥^{−½}·∫₀ᴾ exp h(κs) ds
//!   h(x) = −η[V(1 − cos x) + U sin x] + ½η²κ²[a(cos x − 1)² + 2b(cos x − 1) sin x + d sin² x]
//! ```
//!
//! with `V = Σ wᵢ cos κtᵢ`, `U = Σ wᵢ sin κtᵢ` over the rows the ARD prior covers, `N = τᵀΦτ`,
//! `μ_⊥` the pencil of the complement, and `(a, b, d)` the complement pseudo-inverse's quadratic
//! forms of `uᵢ = wᵢ sin κtᵢ`, `vᵢ = wᵢ cos κtᵢ`. The coupling term is the complement's response
//! to the ARD gradient the shift moves; it makes the large-`c` limit reproduce Laplace exactly.

use super::*;
use gam_linalg::roundoff::accumulation_growth;
use gam_math::categorical::log_sum_exp;
use gam_math::special::bessel_i0_centered_terms_from_log_abs;

/// The latent period of the analytic periodic harmonic basis: it emits `sin(2π·h·t)` and
/// `cos(2π·h·t)`, so a shift of `t` by one leaves every column unchanged
/// ([`crate::basis::PeriodicHarmonicEvaluator`], `atom_build.rs`'s `PeriodicHarmonics` plan on
/// `t ∈ [0, 1)`).
const PERIODIC_HARMONIC_BASIS_PERIOD: f64 = 1.0;

/// Why an atom's chart orbit keeps the Laplace pricing of its tangent.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CompactOrbitLaplaceReason {
    /// The atom is not a one-dimensional periodic harmonic chart. Sphere, projective-plane,
    /// torus, cylinder, Klein and every non-compact family keep the Laplace path.
    NotAPeriodicChart,
    /// The compensation design `a·Φ` loses a column, so the decoder compensation is the
    /// minimum-norm one and the smoothing reads the dropped component along the shift.
    RankDeficientCompensation,
    /// `∂Φ/∂t` is not `Φ·K̃` for the stored basis: a subspace-reduced basis that the phase rotation
    /// leaves (#1117), so the shift does not stay in the decoder's span.
    ClosureResidual { residual: f64, band: f64 },
    /// The coordinate prior's period is not the harmonic basis period, so the shift that leaves
    /// the reconstruction invariant does not close the prior's circle.
    PeriodMismatch { prior: f64, basis: f64 },
    /// The shift moves no reconstruction (the atom decodes a constant), so there is no orbit to
    /// integrate.
    StationaryReconstruction,
    /// The atom's orbit shares a connected block of the exact Hessian or the evidence metric with
    /// the closure-certified orbits of `atoms`, so the complement couples their orbit coordinates
    /// and the orbit integral is not a product of per-atom circle integrals. A torus quadrature's
    /// node count multiplies across the group (the demo-like coupling alone needs 1743 nodes per
    /// orbit), so every orbit of the group keeps Laplace.
    CoupledCompactOrbits { atoms: Vec<usize> },
}

/// One atom's pricing of its declared compact chart orbit.
#[derive(Debug, Clone)]
pub(crate) enum CompactOrbitPricing {
    ExactCircle(CircleOrbitGenerator),
    Laplace {
        atom: usize,
        reason: CompactOrbitLaplaceReason,
    },
}

/// A closure-certified circle orbit of one periodic atom at a priced state.
#[derive(Debug, Clone)]
pub(crate) struct CircleOrbitGenerator {
    pub(crate) atom: usize,
    /// `P`, the prior's and the basis's shared period.
    pub(crate) period: f64,
    /// `κ = 2π/P`.
    pub(crate) kappa: f64,
    /// `η = αP²/(2π)²`; zero when the axis carries no ARD prior, where the orbit is flat.
    pub(crate) eta: f64,
    /// `τ` in the joint `(t, β)` layout: a unit shift on every coordinate slot of the atom and
    /// the decoder compensation that holds every reconstruction.
    pub(crate) tangent: Array1<f64>,
    /// `(coordinate slot, row weight wᵢ, coordinate tᵢ)` over the rows the ARD prior covers.
    pub(crate) prior_rows: Vec<(usize, f64, f64)>,
    /// `‖∂Φ − ΦK̃‖_F` and the band it was certified against.
    pub(crate) closure_residual: f64,
    pub(crate) closure_band: f64,
    /// `K̃`, `∂Φ/∂t = ΦK̃`: the decoder compensation is `δC = −K̃C`, so the tangent's border block
    /// moves with the decoder coordinates as `∂τ_β/∂C = −K̃`.
    pub(crate) closure: Array2<f64>,
    /// The atom's border block: its first slot in the joint layout, its basis width `M` and its
    /// border rank (the decoder frame's rank, or the output width without a frame).
    pub(crate) border_start: usize,
    pub(crate) basis_size: usize,
    pub(crate) border_rank: usize,
}

impl CircleOrbitGenerator {
    /// `(ηV, ηU)` with `V = Σ wᵢ cos κtᵢ`, `U = Σ wᵢ sin κtᵢ`: the resultant `c·e^{iφ̄}`.
    pub(crate) fn resultant(&self) -> (f64, f64) {
        let mut cos_sum = 0.0_f64;
        let mut sin_sum = 0.0_f64;
        for &(_, weight, t) in &self.prior_rows {
            let (sin, cos) = (self.kappa * t).sin_cos();
            cos_sum += weight * cos;
            sin_sum += weight * sin;
        }
        (self.eta * cos_sum, self.eta * sin_sum)
    }

    /// `(u, v)` in the joint layout, `uᵢ = wᵢ sin κtᵢ`, `vᵢ = wᵢ cos κtᵢ` on the prior's slots:
    /// the shift moves the ARD gradient by `Δg(x) = ηκ[(cos x − 1)·u + sin x·v]`.
    pub(crate) fn trigonometric_images(&self, dim: usize) -> (Array1<f64>, Array1<f64>) {
        let mut u = Array1::<f64>::zeros(dim);
        let mut v = Array1::<f64>::zeros(dim);
        for &(slot, weight, t) in &self.prior_rows {
            let (sin, cos) = (self.kappa * t).sin_cos();
            u[slot] += weight * sin;
            v[slot] += weight * cos;
        }
        (u, v)
    }

    /// The orbit integrand for complement quadratic forms `a = uᵀGu`, `b = uᵀGv`, `d = vᵀGv`.
    pub(crate) fn integrand(&self, a: f64, b: f64, d: f64) -> CircleOrbitIntegrand {
        let (resultant_cos, resultant_sin) = self.resultant();
        let scale = 0.5 * self.eta * self.eta * self.kappa * self.kappa;
        CircleOrbitIntegrand {
            resultant_cos,
            resultant_sin,
            coupling: [scale * a, scale * b, scale * d],
            period: self.period,
        }
    }
}

/// `(‖∂Φ − ΦK̃‖_F, band, K̃)` for the atom's stored basis and axis-0 Jacobian, with `K̃` the
/// least-squares closure matrix. A harmonic basis closes exactly, `∂Φ = ΦK`, so its residual is
/// the solve's rounding: `γ_{n·m}·(‖Φ‖_F‖K̃‖_F + ‖∂Φ‖_F)` from the solve's operation count.
fn harmonic_closure_certificate(atom: &SaeManifoldAtom) -> Result<(f64, f64, Array2<f64>), String> {
    let design = atom.basis_values.view();
    let jacobian = atom.basis_jacobian.index_axis(ndarray::Axis(2), 0);
    let (rows, cols) = design.dim();
    let closure = solve_design_least_squares(design, jacobian)?;
    let residual = &jacobian - &design.dot(&closure);
    let frobenius = |x: ArrayView2<'_, f64>| x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let band = accumulation_growth(rows * cols)
        * (frobenius(design) * frobenius(closure.view()) + frobenius(jacobian));
    Ok((frobenius(residual.view()), band, closure))
}

impl SaeManifoldTerm {
    /// Every atom's orbit pricing at the state `cache` factors, in atom order.
    pub(crate) fn compact_orbit_pricing(
        &self,
        rho: &SaeManifoldRho,
        cache: &ArrowFactorCache,
    ) -> Result<Vec<CompactOrbitPricing>, String> {
        let n = self.n_obs();
        let q = self.assignment.row_block_dim();
        let coord_offsets = self.assignment.coord_offsets();
        let beta_offsets = self.factored_border_offsets();
        let dense_len = n * q + self.factored_border_dim();
        let precisions = self.validated_ard_precisions(rho)?;
        let row_weights = self.row_loss_weights.as_deref();
        let mut out = Vec::with_capacity(self.k_atoms());
        for atom_idx in 0..self.k_atoms() {
            let atom = &self.atoms[atom_idx];
            let laplace = |reason| CompactOrbitPricing::Laplace {
                atom: atom_idx,
                reason,
            };
            if !matches!(atom.basis_kind(), SaeAtomBasisKind::Periodic)
                || self.assignment.coords[atom_idx].latent_dim() != 1
            {
                out.push(laplace(CompactOrbitLaplaceReason::NotAPeriodicChart));
                continue;
            }
            let Some(prior_period) = self.ard_axis_periods(atom_idx).first().copied().flatten() else {
                out.push(laplace(CompactOrbitLaplaceReason::NotAPeriodicChart));
                continue;
            };
            if prior_period != PERIODIC_HARMONIC_BASIS_PERIOD {
                out.push(laplace(CompactOrbitLaplaceReason::PeriodMismatch {
                    prior: prior_period,
                    basis: PERIODIC_HARMONIC_BASIS_PERIOD,
                }));
                continue;
            }
            if !self.atom_compensation_has_full_column_rank(atom_idx)? {
                out.push(laplace(CompactOrbitLaplaceReason::RankDeficientCompensation));
                continue;
            }
            let (closure_residual, closure_band, closure) = harmonic_closure_certificate(atom)?;
            if !(closure_residual <= closure_band) {
                out.push(laplace(CompactOrbitLaplaceReason::ClosureResidual {
                    residual: closure_residual,
                    band: closure_band,
                }));
                continue;
            }
            // The generator is global in the dense product chart: a unit shift on every row's
            // coordinate and the decoder compensation. A hard-TopK cache holds only the selected
            // atoms' coordinates, so the tangent crosses the one dense-to-arrow seam every chart
            // gauge crosses, which drops the inactive rows' slots and keeps the border.
            let field = Array2::<f64>::ones((n, 1));
            let Some(dense_tangent) = self.dense_step_gauge_vector_from_field(
                atom_idx,
                field.view(),
                &coord_offsets,
                &beta_offsets,
                dense_len,
            )?
            else {
                out.push(laplace(CompactOrbitLaplaceReason::StationaryReconstruction));
                continue;
            };
            let tangent = self.dense_joint_vector_in_arrow_layout(
                dense_tangent.view(),
                &cache.row_offsets,
                cache.k,
                "compact_orbit_pricing",
            )?;
            let alpha = precisions[atom_idx].get(0).copied().unwrap_or(0.0);
            let kappa = std::f64::consts::TAU / prior_period;
            let coords = self.assignment.coords[atom_idx].as_matrix();
            // The ARD prior covers the slots the cache's layout holds for this atom: every row in
            // the dense layout, the selecting rows in a hard-TopK one (#2933 F27), at the same
            // slot [`Self::ard_inverse_traces`] reads.
            let mut prior_rows = Vec::with_capacity(n);
            for row in 0..n {
                let start = match self.last_row_layout.as_ref() {
                    Some(layout) => {
                        let Some(position) = layout.active_atoms[row]
                            .iter()
                            .position(|&active| active == atom_idx)
                        else {
                            continue;
                        };
                        layout.coord_starts[row][position]
                    }
                    None => coord_offsets[atom_idx],
                };
                prior_rows.push((
                    cache.row_offsets[row] + start,
                    row_weights.map_or(1.0, |weights| weights[row]),
                    coords[[row, 0]],
                ));
            }
            out.push(CompactOrbitPricing::ExactCircle(CircleOrbitGenerator {
                atom: atom_idx,
                period: prior_period,
                kappa,
                eta: alpha / (kappa * kappa),
                tangent,
                prior_rows,
                closure_residual,
                closure_band,
                closure,
                border_start: cache.delta_t_len() + beta_offsets[atom_idx],
                basis_size: atom.basis_size(),
                border_rank: atom
                    .decoder_frame
                    .as_ref()
                    .map_or(self.output_dim(), |frame| frame.frame().ncols()),
            }));
        }
        Ok(out)
    }

    /// Keep an exact circle only for an orbit alone in its connected block of `A` and `Φ` (#2234).
    ///
    /// The orbit integrals multiply exactly when the complement pseudo-inverse `G_⊥` couples no two
    /// orbits' trigonometric images, and it couples none when their atoms' joint indices fall in
    /// distinct connected blocks of the operator and the metric: `G_⊥`, the stiffening and `N`
    /// then split over the blocks. The block structure is read off entries both evaluation routes
    /// build the same way. A coordinate row block couples only its own row's slots, so each row is
    /// joined whole; the ordered Beta–Bernoulli mass carriers join their slots across rows; and the
    /// border columns of `A` and `Φ` carry every coordinate–border and border–border entry. Hard
    /// TopK rows that each select one atom, over a border with no cross-atom block, leave every
    /// orbit its own block. Orbits whose blocks meet, directly or through a third orbit, keep
    /// Laplace by name.
    ///
    /// Without `border_columns` (`(A, Φ)` border columns) only the row blocks and the carriers
    /// join. Joins only merge blocks, so every orbit that partial structure couples stays coupled
    /// in the full one, and a partial pass that leaves no orbit exact is already the answer.
    pub(crate) fn separate_coupled_compact_orbits(
        pricings: Vec<CompactOrbitPricing>,
        row_offsets: &[usize],
        border_dim: usize,
        mass_carriers: &[(f64, Vec<(usize, f64)>)],
        border_columns: Option<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)>,
    ) -> Result<Vec<CompactOrbitPricing>, String> {
        let orbits: Vec<usize> = (0..pricings.len())
            .filter(|&index| matches!(pricings[index], CompactOrbitPricing::ExactCircle(_)))
            .collect();
        if orbits.len() < 2 {
            return Ok(pricings);
        }
        let Some(&total_t) = row_offsets.last() else {
            return Err("separate_coupled_compact_orbits: the row offsets are empty".to_string());
        };
        let dim = total_t + border_dim;
        let mut blocks = JointBlocks::new(dim);
        for window in row_offsets.windows(2) {
            let (start, end) = (window[0], window[1]);
            for slot in (start + 1)..end {
                blocks.join(start, slot);
            }
        }
        for (_, carrier) in mass_carriers {
            if let Some(&(first, _)) = carrier.first() {
                for &(slot, _) in carrier {
                    blocks.join(first, slot);
                }
            }
        }
        if let Some((operator_columns, metric_columns)) = border_columns {
            if operator_columns.dim() != (dim, border_dim) || metric_columns.dim() != (dim, border_dim) {
                return Err(format!(
                    "separate_coupled_compact_orbits: border columns {:?} and {:?} for a joint dimension \
                     {dim} with {border_dim} border columns",
                    operator_columns.dim(),
                    metric_columns.dim(),
                ));
            }
            for column in 0..border_dim {
                for index in 0..dim {
                    if operator_columns[[index, column]] != 0.0 || metric_columns[[index, column]] != 0.0 {
                        blocks.join(index, total_t + column);
                    }
                }
            }
        }
        // Each orbit's blocks are the roots its tangent touches; orbits whose blocks meet are one
        // group, closed transitively because a shared block couples them through `G_⊥`.
        let mut touched: Vec<Vec<usize>> = Vec::with_capacity(orbits.len());
        for &index in &orbits {
            let CompactOrbitPricing::ExactCircle(generator) = &pricings[index] else {
                return Err("separate_coupled_compact_orbits: an orbit index names no exact circle".to_string());
            };
            let mut roots: Vec<usize> = generator
                .tangent
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .map(|(slot, _)| blocks.find(slot))
                .collect();
            roots.sort_unstable();
            roots.dedup();
            touched.push(roots);
        }
        let mut groups = JointBlocks::new(orbits.len());
        for left in 0..orbits.len() {
            for right in (left + 1)..orbits.len() {
                if touched[left].iter().any(|root| touched[right].binary_search(root).is_ok()) {
                    groups.join(left, right);
                }
            }
        }
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); orbits.len()];
        for position in 0..orbits.len() {
            let root = groups.find(position);
            members[root].push(position);
        }
        let atoms: Vec<usize> = orbits
            .iter()
            .map(|&index| match &pricings[index] {
                CompactOrbitPricing::ExactCircle(generator) => generator.atom,
                CompactOrbitPricing::Laplace { atom, .. } => *atom,
            })
            .collect();
        let mut out = pricings;
        for group in members.iter().filter(|group| group.len() > 1) {
            for &position in group {
                let partners = group
                    .iter()
                    .filter(|&&other| other != position)
                    .map(|&other| atoms[other])
                    .collect();
                out[orbits[position]] = CompactOrbitPricing::Laplace {
                    atom: atoms[position],
                    reason: CompactOrbitLaplaceReason::CoupledCompactOrbits { atoms: partners },
                };
            }
        }
        Ok(out)
    }
}

/// Disjoint sets over joint indices, rooted at each set's smallest index.
struct JointBlocks {
    parent: Vec<usize>,
}

impl JointBlocks {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            self.parent[index] = self.parent[self.parent[index]];
            index = self.parent[index];
        }
        index
    }

    fn join(&mut self, left: usize, right: usize) {
        let (left, right) = (self.find(left), self.find(right));
        if left != right {
            self.parent[left.max(right)] = left.min(right);
        }
    }
}

/// One circle orbit's integrand at a priced state, in the orbit angle `x = κs ∈ [0, 2π)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CircleOrbitIntegrand {
    /// `ηV = c·cos φ̄`.
    pub(crate) resultant_cos: f64,
    /// `ηU = c·sin φ̄`.
    pub(crate) resultant_sin: f64,
    /// `½η²κ²·a`, `½η²κ²·b`, `½η²κ²·d`: the coupling's trigonometric coefficients.
    pub(crate) coupling: [f64; 3],
    /// The orbit period `P = 2π/κ` in chart coordinate units.
    pub(crate) period: f64,
}

/// The trapezoid quadrature of one orbit integral: `log ∫₀ᴾ exp h(κs) ds` and the normalized
/// node weights `p_j ∝ exp h(x_j)` every expectation leg reads.
#[derive(Debug, Clone)]
pub(crate) struct CircleOrbitIntegral {
    pub(crate) log_integral: f64,
    /// Orbit angles `x_j = 2πj/N`.
    pub(crate) angles: Vec<f64>,
    /// `p_j = exp(h(x_j) − lse_k h(x_k))`, summing to one.
    pub(crate) weights: Vec<f64>,
}

impl CircleOrbitIntegrand {
    /// `c = η·|Σ wᵢ e^{iκtᵢ}|`.
    pub(crate) fn concentration(&self) -> f64 {
        self.resultant_cos.hypot(self.resultant_sin)
    }

    /// `h(x)`.
    pub(crate) fn exponent(&self, x: f64) -> f64 {
        let sin = x.sin();
        // `1 − cos x` through the half angle, so a small shift keeps its quadratic digits.
        let half = (0.5 * x).sin();
        let one_minus_cos = 2.0 * half * half;
        let [qa, qb, qd] = self.coupling;
        -(self.resultant_cos * one_minus_cos + self.resultant_sin * sin)
            + qa * one_minus_cos * one_minus_cos
            - 2.0 * qb * one_minus_cos * sin
            + qd * sin * sin
    }

    /// Upper bound on `log sup_{|Im z| ≤ σ} |exp h(z)|`. For `z = y + iσ`,
    /// `Re Σ wᵢ cos(κtᵢ + z) ≤ (c/η)·cosh σ`, `|cos z − 1| ≤ cosh σ + 1` and `|sin z| ≤ cosh σ`.
    fn strip_log_bound(&self, sigma: f64) -> f64 {
        let c = self.concentration();
        let [qa, qb, qd] = self.coupling;
        let cosh = sigma.cosh();
        c * cosh - self.resultant_cos + (cosh + 1.0) * (cosh + 1.0) * (qa.abs() + 2.0 * qb.abs() + qd.abs())
    }

    /// Relative truncation bound of the `N`-node trapezoid rule (Trefethen–Weideman, SIAM Rev. 56,
    /// 2014, Thm 3.2). `exp h` is entire, so the bound holds at every strip half-width `σ > 0`:
    ///
    /// ```text
    ///   |I_N − I|/I ≤ 2·exp(H(σ) + c cos φ̄ − log I₀(c)) / (e^{Nσ} − 1)
    /// ```
    ///
    /// It is taken at the `σ` where `H(σ) − Nσ` is least, with `H(σ) = c·cosh σ − c cos φ̄ +
    /// Q·(cosh σ + 1)²` and `Q = |qa| + 2|qb| + |qd|` ([`Self::strip_log_bound`]). Balancing only the
    /// resultant's `c·cosh σ` against `Nσ` lets the coupling's `Q·cosh²σ` outgrow the decay, so the
    /// bound would rise with `N` and no node count would meet it.
    ///
    /// The lower bound `I ≥ 2π·e^{−c cos φ̄}·I₀(c)` holds because the coupling is a positive
    /// semidefinite quadratic form.
    pub(crate) fn relative_truncation_bound(&self, nodes: usize) -> f64 {
        let c = self.concentration();
        let [qa, qb, qd] = self.coupling;
        let coupling = qa.abs() + 2.0 * qb.abs() + qd.abs();
        if c + coupling == 0.0 {
            // A constant integrand: the one-node rule is exact.
            return 0.0;
        }
        let sigma = Self::balanced_strip_width(nodes as f64, c, coupling);
        self.log_relative_truncation_bound_at(nodes, sigma).exp()
    }

    /// `log` of the `N`-node truncation bound at strip half-width `sigma`, which stays finite where
    /// the bound itself overflows.
    pub(crate) fn log_relative_truncation_bound_at(&self, nodes: usize, sigma: f64) -> f64 {
        let c = self.concentration();
        // `log I₀(c) = c + (log I₀(c) − c)`, with the centered part from its stable owner.
        let centered_log_i0 = if c > 0.0 {
            bessel_i0_centered_terms_from_log_abs(c.ln()).0
        } else {
            0.0
        };
        let exponent = self.strip_log_bound(sigma) + self.resultant_cos - c - centered_log_i0;
        let decay = nodes as f64 * sigma;
        // `2·e^{exponent}/(e^{decay} − 1)`, evaluated in log space so neither factor overflows.
        std::f64::consts::LN_2 + exponent - decay - (-(-decay).exp()).ln_1p()
    }

    /// The strip half-width minimizing `H(σ) − Nσ`: the root of `sinh σ·(c + 2Q(cosh σ + 1)) = N` on
    /// `σ > 0`. The left side increases from zero and is at least `N` at `asinh(N/(c + 4Q))`, so
    /// bisection on that bracket converges; it stops when the midpoint no longer splits the bracket.
    fn balanced_strip_width(nodes: f64, concentration: f64, coupling: f64) -> f64 {
        let slope = |sigma: f64| sigma.sinh() * (concentration + 2.0 * coupling * (sigma.cosh() + 1.0));
        let mut low = 0.0_f64;
        let mut high = (nodes / (concentration + 4.0 * coupling)).asinh();
        loop {
            let middle = 0.5 * (low + high);
            if middle <= low || middle >= high {
                return high;
            }
            if slope(middle) < nodes {
                low = middle;
            } else {
                high = middle;
            }
        }
    }

    /// The smallest node count whose truncation bound is at or below the rounding band of the
    /// node sum it produces, `γ_N`: past that count further nodes change the computed integral by
    /// less than its own accumulated rounding. The bound tends to zero as `N` grows, so the count
    /// is finite; it is read off the bound, not searched against the integral.
    pub(crate) fn node_count(&self) -> usize {
        let mut nodes = 1usize;
        loop {
            if self.relative_truncation_bound(nodes) <= accumulation_growth(nodes) {
                return nodes;
            }
            nodes += 1;
        }
    }

    /// `log ∫₀ᴾ exp h(κs) ds` by the trapezoid rule at [`Self::node_count`] nodes, with the node
    /// weights.
    pub(crate) fn integrate(&self) -> Result<CircleOrbitIntegral, String> {
        if !(self.period.is_finite() && self.period > 0.0) {
            return Err(format!(
                "compact orbit integral: period must be finite and positive, got {}",
                self.period
            ));
        }
        let nodes = self.node_count();
        let angles: Vec<f64> = (0..nodes)
            .map(|j| std::f64::consts::TAU * j as f64 / nodes as f64)
            .collect();
        let exponents: Vec<f64> = angles.iter().map(|&x| self.exponent(x)).collect();
        let normalizer = log_sum_exp(&exponents)
            .map_err(|error| format!("compact orbit integral: log-sum-exp: {error:?}"))?;
        let weights = exponents.iter().map(|&h| (h - normalizer).exp()).collect();
        Ok(CircleOrbitIntegral {
            log_integral: (self.period / nodes as f64).ln() + normalizer,
            angles,
            weights,
        })
    }
}
