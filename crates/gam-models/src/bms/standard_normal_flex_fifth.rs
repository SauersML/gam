//! The standard-normal FLEX row's link-knot crossing terms.
//!
//! Under the standard-normal latent measure a FLEX row calibrates its intercept
//! through the cell integral `M(a, θ) = Σ_cells ∫ φ(z)·Φ(η(z; a, θ)) dz − μ(q)`.
//! A link-knot crossing is a cell edge that moves with the row scalars, so the
//! fixed-domain cell moments miss its moving-boundary terms from order two on.
//! The link deviation is `C²` at an interior knot but only `C⁰` at a support
//! edge, where its tails turn constant, so an edge crossing carries jumps in `L′`
//! and `L″` as well as `L‴`. This module supplies those terms to the canonical
//! lowerings: the order-two crossing partials of the V/G/H lowering and the
//! order-three calibration crossings of the third-order lowerings.

use super::family::*;
use super::hessian_paths::{CachedDenestedCellMoments, PrimarySlices};
use super::*;

/// Every ordering of up to three coordinate slots, for filling a symmetric tensor.
const ORDERINGS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// One derivative slot of an explicit partial in `(a, θ)`.
#[derive(Clone, Copy)]
pub(super) enum ExplicitSlot {
    U,
    V,
    Intercept,
    Coordinate(usize),
}

/// Visit every set partition of the labels in `mask` exactly once, as the label
/// masks of its blocks. The empty set has one partition, with no blocks.
fn for_each_partition(mask: usize, mut visit: impl FnMut(&[usize])) {
    let mut elements = [0_usize; 5];
    let mut count = 0;
    for label in 0..5 {
        if mask & (1 << label) != 0 {
            elements[count] = label;
            count += 1;
        }
    }
    if count == 0 {
        visit(&[]);
        return;
    }
    let mut block = [0_usize; 5];
    loop {
        let mut masks = [0_usize; 5];
        let mut blocks = 0;
        for (position, &index) in block[..count].iter().enumerate() {
            masks[index] |= 1 << elements[position];
            blocks = blocks.max(index + 1);
        }
        visit(&masks[..blocks]);
        let mut position = count;
        loop {
            if position <= 1 {
                return;
            }
            position -= 1;
            let ceiling = 1 + block[..position].iter().copied().max().unwrap_or(0);
            if block[position] < ceiling {
                block[position] += 1;
                for later in block[position + 1..count].iter_mut() {
                    *later = 0;
                }
                break;
            }
        }
    }
}

/// The z-polynomials every mixed partial of one de-nested index is built from.
///
/// The index is `scale·(a + b·z + b·H(z) + L(a + b·z))` and is linear in each
/// score-warp and link-deviation coefficient, so a partial over two deviation
/// coefficients vanishes and every other partial is a cubic:
/// `base[n_a][n_b] = ∂_a^{n_a} ∂_b^{n_b} η`, and for a deviation coordinate `p`
/// `deviation[p][n_a][n_b] = ∂_a^{n_a} ∂_b^{n_b} ∂_p η`. Both vanish beyond third
/// order in `(a, b)`. `directional[e]` contracts the deviation columns with the
/// deviation components of direction `e`.
#[derive(Clone)]
struct IndexAtoms {
    q: usize,
    slope: usize,
    base: [[[f64; 4]; 4]; 4],
    deviation: Vec<[[[f64; 4]; 4]; 4]>,
    directional: [[[[f64; 4]; 4]; 4]; 2],
    slope_components: [f64; 2],
    /// Raw leading coefficient of each link-deviation column's local cubic.
    link_leading: Vec<f64>,
    /// `∂^S η` for every block content: a direction code (`u` is 2, `v` is 1),
    /// the intercept and slope orders, and the deviation coordinate plus one
    /// (zero for none). Filled once per cell, so a subset is a lookup.
    content: Vec<[f64; 4]>,
}

impl IndexAtoms {
    fn new(
        family: &BernoulliMarginalSlopeFamily,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        base: [[[f64; 4]; 4]; 4],
        score_point: f64,
        link_point: f64,
        directions: [&Array1<f64>; 2],
    ) -> Result<Self, String> {
        let r = primary.total;
        let scale = family.probit_frailty_scale();
        let mut deviation = vec![[[[0.0; 4]; 4]; 4]; r];
        let mut link_leading = vec![0.0; r];
        if let (Some(range), Some(runtime)) = (primary.h.as_ref(), family.score_warp.as_ref()) {
            BernoulliMarginalSlopeFamily::for_each_deviation_basis_cubic_at(
                runtime,
                range,
                score_point,
                "score-warp fifth-order atoms",
                |_, index, span| {
                    deviation[index][0][0] =
                        scale_coeff4(exact_kernel::score_basis_cell_coefficients(span, b), scale);
                    deviation[index][0][1] =
                        scale_coeff4(exact_kernel::score_basis_cell_coefficients(span, 1.0), scale);
                    Ok(())
                },
            )?;
        }
        if let (Some(range), Some(runtime)) = (primary.w.as_ref(), family.link_dev.as_ref()) {
            BernoulliMarginalSlopeFamily::for_each_deviation_basis_cubic_at(
                runtime,
                range,
                link_point,
                "link-deviation fifth-order atoms",
                |_, index, span| {
                    let column = &mut deviation[index];
                    column[0][0] =
                        scale_coeff4(exact_kernel::link_basis_cell_coefficients(span, a, b), scale);
                    let (da, db) = exact_kernel::link_basis_cell_coefficient_partials(span, a, b);
                    column[1][0] = scale_coeff4(da, scale);
                    column[0][1] = scale_coeff4(db, scale);
                    let (daa, dab, dbb) = exact_kernel::link_basis_cell_second_partials(span, a, b);
                    column[2][0] = scale_coeff4(daa, scale);
                    column[1][1] = scale_coeff4(dab, scale);
                    column[0][2] = scale_coeff4(dbb, scale);
                    let (daaa, daab, dabb, dbbb) = exact_kernel::link_basis_cell_third_partials(span);
                    column[3][0] = scale_coeff4(daaa, scale);
                    column[2][1] = scale_coeff4(daab, scale);
                    column[1][2] = scale_coeff4(dabb, scale);
                    column[0][3] = scale_coeff4(dbbb, scale);
                    link_leading[index] = span.c3;
                    Ok(())
                },
            )?;
        }
        let mut directional = [[[[0.0; 4]; 4]; 4]; 2];
        for (target, direction) in directional.iter_mut().zip(directions) {
            for (p, column) in deviation.iter().enumerate() {
                let weight = direction[p];
                if weight == 0.0 || p == primary.q || p == primary.slope {
                    continue;
                }
                for (target_row, column_row) in target.iter_mut().zip(column) {
                    for (target_cell, column_cell) in target_row.iter_mut().zip(column_row) {
                        for (slot, &value) in target_cell.iter_mut().zip(column_cell) {
                            *slot += weight * value;
                        }
                    }
                }
            }
        }
        let mut atoms = Self {
            q: primary.q,
            slope: primary.slope,
            base,
            deviation,
            directional,
            slope_components: [directions[0][primary.slope], directions[1][primary.slope]],
            link_leading,
            content: Vec::new(),
        };
        let mut content = vec![[0.0; 4]; 4 * 4 * 4 * (r + 1)];
        for direction_code in 0..4 {
            for n_a in 0..4 {
                for n_b in 0..4 - n_a {
                    for deviation in 0..=r {
                        content[Self::content_index(direction_code, n_a, n_b, deviation, r)] = atoms
                            .content_polynomial(direction_code, n_a, n_b, deviation.checked_sub(1));
                    }
                }
            }
        }
        atoms.content = content;
        Ok(atoms)
    }

    /// The same atoms with the content table re-expanded about `z`, so a subset
    /// lookup returns `∂^S η(z + t)` as a cubic in `t`.
    fn about(&self, z: f64) -> Self {
        let mut shifted = self.clone();
        for polynomial in &mut shifted.content {
            *polynomial = cubic_about(polynomial, z);
        }
        shifted
    }

    #[inline]
    fn content_index(
        direction_code: usize,
        n_a: usize,
        n_b: usize,
        deviation: usize,
        r: usize,
    ) -> usize {
        ((direction_code * 4 + n_a) * 4 + n_b) * (r + 1) + deviation
    }

    /// `∂^S η` over the slots of `slots` selected by `mask`, read from the
    /// content table.
    fn subset_polynomial(&self, slots: &[ExplicitSlot], mask: usize) -> [f64; 4] {
        let mut direction_code = 0;
        let mut n_a = 0;
        let mut n_b = 0;
        let mut deviation = 0;
        let mut deviations = 0;
        for (position, &slot) in slots.iter().enumerate() {
            if mask & (1 << position) == 0 {
                continue;
            }
            match slot {
                ExplicitSlot::U => direction_code |= 2,
                ExplicitSlot::V => direction_code |= 1,
                ExplicitSlot::Intercept => n_a += 1,
                ExplicitSlot::Coordinate(p) => {
                    if p == self.q {
                        return [0.0; 4];
                    }
                    if p == self.slope {
                        n_b += 1;
                    } else {
                        deviations += 1;
                        deviation = p + 1;
                    }
                }
            }
        }
        if deviations > 1 || n_a + n_b > 3 {
            return [0.0; 4];
        }
        self.content[Self::content_index(direction_code, n_a, n_b, deviation, self.deviation.len())]
    }

    /// `∂^S η` for one block content. Each direction differentiates through its
    /// slope component or through its deviation components, and at most one
    /// deviation derivative survives.
    fn content_polynomial(
        &self,
        direction_code: usize,
        n_a: usize,
        n_b: usize,
        coordinate_deviation: Option<usize>,
    ) -> [f64; 4] {
        let mut directions = [0_usize; 2];
        let mut direction_count = 0;
        if direction_code & 2 != 0 {
            directions[direction_count] = 0;
            direction_count += 1;
        }
        if direction_code & 1 != 0 {
            directions[direction_count] = 1;
            direction_count += 1;
        }
        let deviations = usize::from(coordinate_deviation.is_some());
        let mut out = [0.0; 4];
        for on_slope in 0..(1_usize << direction_count) {
            let mut weight = 1.0;
            let mut b_order = n_b;
            let mut direction_deviation = None;
            let mut total_deviations = deviations;
            for (position, &e) in directions[..direction_count].iter().enumerate() {
                if on_slope & (1 << position) != 0 {
                    weight *= self.slope_components[e];
                    b_order += 1;
                } else {
                    total_deviations += 1;
                    direction_deviation = Some(e);
                }
            }
            if weight == 0.0 || total_deviations > 1 || n_a + b_order > 3 {
                continue;
            }
            let polynomial = if let Some(p) = coordinate_deviation {
                &self.deviation[p][n_a][b_order]
            } else if let Some(e) = direction_deviation {
                &self.directional[e][n_a][b_order]
            } else {
                &self.base[n_a][b_order]
            };
            for (slot, &value) in out.iter_mut().zip(polynomial) {
                *slot += weight * value;
            }
        }
        out
    }
}

/// The `(a, b)` partials of a partition cell's index, scaled.
fn cell_base_partials(
    partition_cell: &exact_kernel::DenestedPartitionCell,
    a: f64,
    b: f64,
    scale: f64,
) -> [[[f64; 4]; 4]; 4] {
    let mut base = [[[0.0; 4]; 4]; 4];
    let (da, db) = exact_kernel::denested_cell_coefficient_partials(
        partition_cell.score_span,
        partition_cell.link_span,
        a,
        b,
    );
    let (daa, dab, dbb) =
        exact_kernel::link_basis_cell_second_partials(partition_cell.link_span, a, b);
    let (daaa, daab, dabb, dbbb) = exact_kernel::denested_cell_third_partials(partition_cell.link_span);
    base[1][0] = scale_coeff4(da, scale);
    base[0][1] = scale_coeff4(db, scale);
    base[2][0] = scale_coeff4(daa, scale);
    base[1][1] = scale_coeff4(dab, scale);
    base[0][2] = scale_coeff4(dbb, scale);
    base[3][0] = scale_coeff4(daaa, scale);
    base[2][1] = scale_coeff4(daab, scale);
    base[1][2] = scale_coeff4(dabb, scale);
    base[0][3] = scale_coeff4(dbbb, scale);
    base
}

/// Taylor coefficients in `t = z − z*`. A moving-boundary term of order five
/// reads at most the fourth `z`-derivative of the integrand jump.
type Taylor = [f64; 5];

const FACTORIALS: [f64; 5] = [1.0, 1.0, 2.0, 6.0, 24.0];

/// The product of two truncated series, through `len` coefficients.
fn taylor_product(left: &Taylor, right: &Taylor, len: usize) -> Taylor {
    let mut out = [0.0; 5];
    for (i, &value) in left.iter().enumerate().take(len) {
        if value == 0.0 {
            continue;
        }
        for (j, &other) in right.iter().enumerate().take(len - i) {
            out[i + j] += value * other;
        }
    }
    out
}

/// `p(z + t)` for a cubic `p`, as a cubic in `t`.
fn cubic_about(polynomial: &[f64; 4], z: f64) -> [f64; 4] {
    [
        polynomial[0] + z * (polynomial[1] + z * (polynomial[2] + z * polynomial[3])),
        polynomial[1] + z * (2.0 * polynomial[2] + 3.0 * z * polynomial[3]),
        polynomial[2] + 3.0 * z * polynomial[3],
        polynomial[3],
    ]
}

/// `φ⁽ʲ⁾(x) = (−1)ʲ·He_j(x)·φ(x)` for `j ≤ 7`, from the probabilists' Hermite
/// recurrence `He_{j+1} = x·He_j − j·He_{j−1}`.
fn density_derivatives(x: f64) -> [f64; 8] {
    let mut hermite = [0.0; 8];
    hermite[0] = 1.0;
    hermite[1] = x;
    for j in 1..7 {
        hermite[j + 1] = x * hermite[j] - j as f64 * hermite[j - 1];
    }
    let density = (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt();
    std::array::from_fn(|j| {
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        sign * hermite[j] * density
    })
}

/// One cell at a link-knot crossing, re-expanded about `z*`: its atoms,
/// `φ⁽ᵏ⁾(η(z* + t))` for `k ≤ 3`, and `Φ(η(z* + t)) − Φ(η*)`.
struct CrossingSide {
    atoms: IndexAtoms,
    density: [Taylor; 4],
    probability: Taylor,
}

impl CrossingSide {
    fn new(
        cell: exact_kernel::DenestedCubicCell,
        atoms: &IndexAtoms,
        z_star: f64,
        eta_density: &[f64; 8],
    ) -> Self {
        let index = cubic_about(&[cell.c0, cell.c1, cell.c2, cell.c3], z_star);
        // Both cells take the same index value at the crossing, so only the
        // motion away from it enters the expansion.
        let motion: Taylor = [0.0, index[1], index[2], index[3], 0.0];
        let mut powers = [[0.0; 5]; 5];
        powers[0][0] = 1.0;
        for order in 1..5 {
            powers[order] = taylor_product(&powers[order - 1], &motion, 5);
        }
        let mut density = [[0.0; 5]; 4];
        for (k, series) in density.iter_mut().enumerate() {
            for (order, power) in powers.iter().enumerate() {
                let weight = eta_density[k + order] / FACTORIALS[order];
                for (slot, &value) in series.iter_mut().zip(power) {
                    *slot += weight * value;
                }
            }
        }
        let mut probability = [0.0; 5];
        for (order, power) in powers.iter().enumerate().skip(1) {
            let weight = eta_density[order - 1] / FACTORIALS[order];
            for (slot, &value) in probability.iter_mut().zip(power) {
                *slot += weight * value;
            }
        }
        Self {
            atoms: atoms.about(z_star),
            density,
            probability,
        }
    }
}

/// The moving-boundary terms of one link-knot crossing `z*(θ) = (τ − a)/b`.
///
/// Across the crossing the calibration integrand `φ(z)·Φ(η)` switches from the
/// left cell's index to the right cell's. With the crossing's base location
/// `z₀`, `Σ_cells ∫` carries the extra `E(θ) = ∫_{z₀}^{z*(θ)} J dz` of the jump
/// `J = φ(z)·(Φ(η_L) − Φ(η_R))`, and Faà di Bruno gives
///   `D^S E = Σ_{R ⊊ S} Σ_{π ∈ Π(S∖R)} (∂^R ∂_z^{|π|−1} J)(z*)·Π_{B ∈ π} D^B z*`.
/// The index is continuous at every crossing, so `J(z*) = 0` and order one has
/// no boundary term. At an interior knot the index is `C²` and the terms start
/// at order four. At a support edge it is only `C⁰`, and they start at order two.
/// Because `b·z* = τ − a` with `a` and `b` linear in the slots,
/// `D_s z* = −(∂_s a + ∂_s b·z*)/b` and `D^B z* = −(1/b)·Σ_{t ∈ B} ∂_t b·D^{B∖t} z*`
/// for `|B| ≥ 2`. Only the intercept and slope-bearing slots move the crossing,
/// so every slot outside `R` is one of them.
struct LinkCrossing {
    z_star: f64,
    slope: f64,
    slope_index: usize,
    slope_components: [f64; 2],
    /// `φ(z* + t)`.
    density: Taylor,
    left: CrossingSide,
    right: CrossingSide,
}

impl LinkCrossing {
    fn new(
        left_cell: exact_kernel::DenestedCubicCell,
        left_atoms: &IndexAtoms,
        right_cell: exact_kernel::DenestedCubicCell,
        right_atoms: &IndexAtoms,
        b: f64,
    ) -> Self {
        let z_star = left_cell.right;
        let z_density = density_derivatives(z_star);
        let eta_density = density_derivatives(left_cell.eta(z_star));
        Self {
            z_star,
            slope: b,
            slope_index: left_atoms.slope,
            slope_components: left_atoms.slope_components,
            density: std::array::from_fn(|order| z_density[order] / FACTORIALS[order]),
            left: CrossingSide::new(left_cell, left_atoms, z_star, &eta_density),
            right: CrossingSide::new(right_cell, right_atoms, z_star, &eta_density),
        }
    }

    /// `(∂^R ∂_z^j J)(z*)/j!` for `j < len`, over the slots selected by `mask`.
    fn jump(&self, polynomials: &[[[f64; 4]; 32]; 2], mask: usize, len: usize) -> Taylor {
        let mut difference = [0.0; 5];
        if mask == 0 {
            for (slot, (left, right)) in difference
                .iter_mut()
                .zip(self.left.probability.iter().zip(&self.right.probability))
                .take(len)
            {
                *slot = left - right;
            }
        } else {
            for_each_partition(mask, |blocks| {
                for (side, cubics, sign) in [
                    (&self.left, &polynomials[0], 1.0),
                    (&self.right, &polynomials[1], -1.0),
                ] {
                    let mut term = side.density[blocks.len() - 1];
                    for &block in blocks {
                        let cubic = cubics[block];
                        term = taylor_product(
                            &term,
                            &[cubic[0], cubic[1], cubic[2], cubic[3], 0.0],
                            len,
                        );
                    }
                    for (slot, &value) in difference.iter_mut().zip(&term).take(len) {
                        *slot += sign * value;
                    }
                }
            });
        }
        taylor_product(&self.density, &difference, len)
    }

    /// `D^S E` over `slots`.
    fn partial(&self, slots: &[ExplicitSlot]) -> f64 {
        let n = slots.len();
        let mut velocity = [0.0; 5];
        let mut rate = [0.0; 5];
        let mut moving = 0_usize;
        for (position, &slot) in slots.iter().enumerate() {
            let (intercept, slope) = match slot {
                ExplicitSlot::U => (0.0, self.slope_components[0]),
                ExplicitSlot::V => (0.0, self.slope_components[1]),
                ExplicitSlot::Intercept => (1.0, 0.0),
                ExplicitSlot::Coordinate(p) => (0.0, if p == self.slope_index { 1.0 } else { 0.0 }),
            };
            velocity[position] = intercept + slope * self.z_star;
            rate[position] = slope;
            if intercept != 0.0 || slope != 0.0 {
                moving |= 1 << position;
            }
        }
        if n < 2 || moving == 0 {
            return 0.0;
        }
        let full = (1_usize << n) - 1;
        let mut polynomials = [[[0.0; 4]; 32]; 2];
        for mask in 1..full {
            polynomials[0][mask] = self.left.atoms.subset_polynomial(slots, mask);
            polynomials[1][mask] = self.right.atoms.subset_polynomial(slots, mask);
        }
        let mut motion = [0.0; 32];
        for mask in 1..=full {
            if mask & !moving != 0 {
                continue;
            }
            motion[mask] = if mask.count_ones() == 1 {
                -velocity[mask.trailing_zeros() as usize] / self.slope
            } else {
                let mut sum = 0.0;
                for (position, &value) in rate.iter().enumerate().take(n) {
                    if mask & (1 << position) != 0 {
                        sum += value * motion[mask & !(1 << position)];
                    }
                }
                -sum / self.slope
            };
        }
        let mut total = 0.0;
        let mut rest = moving;
        while rest != 0 {
            let jump = self.jump(&polynomials, full & !rest, rest.count_ones() as usize);
            for_each_partition(rest, |blocks| {
                let product: f64 = blocks.iter().map(|&block| motion[block]).product();
                total += FACTORIALS[blocks.len() - 1] * jump[blocks.len() - 1] * product;
            });
            rest = (rest - 1) & moving;
        }
        total
    }
}

#[derive(Clone, Copy)]
struct FluxSlot {
    /// `∂u/∂s` at the crossing, `u = a + b·z`.
    crossing: f64,
    /// `∂b/∂s`.
    slope: f64,
    /// `∂η/∂s` at the crossing.
    eta: f64,
    /// `∂Δc₃/∂s`.
    jump: f64,
}

impl FluxSlot {
    const ZERO: Self = Self {
        crossing: 0.0,
        slope: 0.0,
        eta: 0.0,
        jump: 0.0,
    };
}

/// Moving-boundary terms of one interior link-knot crossing `z* = (τ − a)/b`.
///
/// At an interior knot the link deviation is `C²`: across the crossing the index
/// changes by `scale·Δc₃·(u − τ)³`, so the calibration integrand is `C²` and
/// orders through three need no boundary term. Differentiating `Σ_cells ∫`
/// further moves `z*` (`∂z*/∂s = −u_s/b`):
///   order 4: `K·(−Δc₃/b)·Π u_s`;
///   order 5: `K·[−(1/b)·Σ_t (∂_tΔc₃ − η*·∂_tη·Δc₃)·Π_{r≠t} u_r
///            − (Δc₃/b²)·(z* + η*·η_z)·Π u_s + (Δc₃/b²)·Σ_t ∂_t b·Π_{r≠t} u_r]`,
/// with `K = 6·scale·e^{−q(z*)}/2π` and per slot `u_s = ∂u/∂s`, `∂_t b`, `∂_tη`
/// and `∂_tΔc₃` at the crossing. This is `LinkCrossing`'s general term with
/// only `L‴` jumping, in closed form.
struct KnotFlux {
    prefactor: f64,
    jump: f64,
    slope: f64,
    eta_star: f64,
    q_z: f64,
    intercept: FluxSlot,
    coordinates: Vec<FluxSlot>,
    directions: [FluxSlot; 2],
}

impl KnotFlux {
    fn new(
        cell: exact_kernel::DenestedCubicCell,
        atoms: &IndexAtoms,
        column_jumps: &[f64],
        jump: f64,
        b: f64,
        scale: f64,
        primary: &PrimarySlices,
        directions: [&Array1<f64>; 2],
    ) -> Self {
        let z_star = cell.right;
        let eta_star = cell.eta(z_star);
        let eta_z = cell.c1 + z_star * (2.0 * cell.c2 + 3.0 * cell.c3 * z_star);
        let mut coordinates = vec![FluxSlot::ZERO; primary.total];
        for (p, slot) in coordinates.iter_mut().enumerate() {
            if p == primary.slope {
                *slot = FluxSlot {
                    crossing: z_star,
                    slope: 1.0,
                    eta: eval_coeff4_at(&atoms.base[0][1], z_star),
                    jump: 0.0,
                };
            } else if p != primary.q {
                *slot = FluxSlot {
                    crossing: 0.0,
                    slope: 0.0,
                    eta: eval_coeff4_at(&atoms.deviation[p][0][0], z_star),
                    jump: column_jumps[p],
                };
            }
        }
        let direction_slot = |direction: &Array1<f64>| {
            let mut slot = FluxSlot {
                crossing: direction[primary.slope] * z_star,
                slope: direction[primary.slope],
                eta: 0.0,
                jump: 0.0,
            };
            for (p, coordinate) in coordinates.iter().enumerate() {
                slot.eta += direction[p] * coordinate.eta;
                slot.jump += direction[p] * coordinate.jump;
            }
            slot
        };
        let direction_slots = [direction_slot(directions[0]), direction_slot(directions[1])];
        Self {
            prefactor: 6.0 * scale * (-cell.q(z_star)).exp() / std::f64::consts::TAU,
            jump,
            slope: b,
            eta_star,
            q_z: z_star + eta_star * eta_z,
            intercept: FluxSlot {
                crossing: 1.0,
                slope: 0.0,
                eta: eval_coeff4_at(&atoms.base[1][0], z_star),
                jump: 0.0,
            },
            coordinates,
            directions: direction_slots,
        }
    }

    fn slot(&self, slot: ExplicitSlot) -> FluxSlot {
        match slot {
            ExplicitSlot::U => self.directions[0],
            ExplicitSlot::V => self.directions[1],
            ExplicitSlot::Intercept => self.intercept,
            ExplicitSlot::Coordinate(p) => self.coordinates[p],
        }
    }

    fn flux(&self, slots: &[ExplicitSlot]) -> f64 {
        let mut values = [FluxSlot::ZERO; 5];
        for (value, &slot) in values.iter_mut().zip(slots) {
            *value = self.slot(slot);
        }
        let values = &values[..slots.len()];
        let crossing_product_without = |skip: usize| {
            let mut product = 1.0;
            for (position, value) in values.iter().enumerate() {
                if position != skip {
                    product *= value.crossing;
                }
            }
            product
        };
        let b = self.slope;
        let all = crossing_product_without(values.len());
        match values.len() {
            4 => self.prefactor * (-self.jump / b) * all,
            5 => {
                let mut total = -(self.jump / (b * b)) * self.q_z * all;
                for (position, value) in values.iter().enumerate() {
                    let others = crossing_product_without(position);
                    total -= (value.jump - self.eta_star * value.eta * self.jump) / b * others;
                    total += self.jump / (b * b) * value.slope * others;
                }
                self.prefactor * total
            }
            _ => 0.0,
        }
    }
}

/// A link-knot crossing's moving-boundary terms. An interior knot jumps only in
/// `L‴`, where the closed-form flux is exact and cheap. A support edge is only
/// `C⁰`, and the general term runs there.
enum Crossing {
    General(LinkCrossing),
    InteriorKnot(KnotFlux),
}

impl Crossing {
    fn partial(&self, slots: &[ExplicitSlot]) -> f64 {
        match self {
            Self::General(crossing) => crossing.partial(slots),
            Self::InteriorKnot(knot) => knot.flux(slots),
        }
    }
}

impl BernoulliMarginalSlopeFamily {
    /// The moving-boundary terms of the crossing between the adjacent cells
    /// `left` and `right`: the closed-form flux where `τ` is an interior link
    /// knot, the general term where it is a support endpoint.
    fn link_crossing(
        &self,
        left: &exact_kernel::DenestedPartitionCell,
        left_atoms: &IndexAtoms,
        right: &exact_kernel::DenestedPartitionCell,
        right_atoms: &IndexAtoms,
        b: f64,
        primary: &PrimarySlices,
        directions: [&Array1<f64>; 2],
    ) -> Crossing {
        let interior_knot = match (left.right_edge, self.link_dev.as_ref()) {
            (exact_kernel::PartitionEdge::Crossing { tau }, Some(runtime)) => {
                let breakpoints = runtime.breakpoints();
                breakpoints.first() != Some(&tau) && breakpoints.last() != Some(&tau)
            }
            _ => false,
        };
        if !interior_knot {
            return Crossing::General(LinkCrossing::new(
                left.cell,
                left_atoms,
                right.cell,
                right_atoms,
                b,
            ));
        }
        let column_jumps: Vec<f64> = left_atoms
            .link_leading
            .iter()
            .zip(&right_atoms.link_leading)
            .map(|(left_c3, right_c3)| left_c3 - right_c3)
            .collect();
        Crossing::InteriorKnot(KnotFlux::new(
            left.cell,
            left_atoms,
            &column_jumps,
            left.link_span.c3 - right.link_span.c3,
            b,
            self.probit_frailty_scale(),
            primary,
            directions,
        ))
    }
}

/// The link-knot crossings' moving-boundary terms of order two,
/// `[B[a,a], B[a,b], B[b,b]]`, over the intercept and the slope, the only
/// coordinates that move a crossing. With `Δ` the jump of an index partial
/// across a crossing at `z*`,
///   `B[s,t] = φ(z*)·φ(η*)·[Δη_z·U_s·U_t/b² − (Δη_s·U_t + Δη_t·U_s)/b]`,
/// with `U_a = 1` and `U_b = z*`. At an interior knot the index is `C²` and every
/// `Δ` vanishes. At a support edge the link deviation is only `C⁰`.
pub(super) fn standard_normal_flex_crossing_second_partials(
    cells: &[exact_kernel::DenestedPartitionCell],
    a: f64,
    b: f64,
    scale: f64,
) -> [f64; 3] {
    let mut out = [0.0; 3];
    for window in cells.windows(2) {
        let (left, right) = (&window[0], &window[1]);
        if !matches!(left.right_edge, exact_kernel::PartitionEdge::Crossing { .. })
            || right.cell.left != left.cell.right
        {
            continue;
        }
        let z_star = left.cell.right;
        let density = (-left.cell.q(z_star)).exp() / std::f64::consts::TAU;
        let index_slope = |cell: exact_kernel::DenestedCubicCell| {
            cell.c1 + z_star * (2.0 * cell.c2 + 3.0 * z_star * cell.c3)
        };
        let index_partials = |partition_cell: &exact_kernel::DenestedPartitionCell| {
            let (da, db) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            (
                scale * eval_coeff4_at(&da, z_star),
                scale * eval_coeff4_at(&db, z_star),
            )
        };
        let z_jump = index_slope(left.cell) - index_slope(right.cell);
        let (left_a, left_b) = index_partials(left);
        let (right_a, right_b) = index_partials(right);
        let (a_jump, b_jump) = (left_a - right_a, left_b - right_b);
        out[0] += density * (z_jump / (b * b) - 2.0 * a_jump / b);
        out[1] += density * (z_jump * z_star / (b * b) - (a_jump * z_star + b_jump) / b);
        out[2] += density * (z_jump * z_star * z_star / (b * b) - 2.0 * b_jump * z_star / b);
    }
    out
}

/// The link-knot crossings of one row's partition, for adding their
/// moving-boundary terms to explicit calibration partials a lowering
/// accumulates cell by cell.
pub(super) struct CalibrationCrossings {
    crossings: Vec<Crossing>,
}

impl CalibrationCrossings {
    /// `Σ_crossings D^S E` over `slots`.
    pub(super) fn partial(&self, slots: &[ExplicitSlot]) -> f64 {
        self.crossings
            .iter()
            .map(|crossing| crossing.partial(slots))
            .sum()
    }
}

impl BernoulliMarginalSlopeFamily {
    /// Every link-knot crossing of the partition `cells` at intercept `a` and
    /// slope `b`. The direction slots read `directions`.
    pub(super) fn standard_normal_flex_calibration_crossings(
        &self,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        cells: &[exact_kernel::DenestedPartitionCell],
        directions: [&Array1<f64>; 2],
    ) -> Result<CalibrationCrossings, String> {
        let scale = self.probit_frailty_scale();
        let atoms = |partition_cell: &exact_kernel::DenestedPartitionCell| {
            let cell = partition_cell.cell;
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            IndexAtoms::new(
                self,
                primary,
                a,
                b,
                cell_base_partials(partition_cell, a, b, scale),
                z_mid,
                a + b * z_mid,
                directions,
            )
        };
        let mut crossings = Vec::new();
        for window in cells.windows(2) {
            let (left, right) = (&window[0], &window[1]);
            if !matches!(left.right_edge, exact_kernel::PartitionEdge::Crossing { .. })
                || right.cell.left != left.cell.right
            {
                continue;
            }
            crossings.push(self.link_crossing(
                left,
                &atoms(left)?,
                right,
                &atoms(right)?,
                b,
                primary,
                directions,
            ));
        }
        Ok(CalibrationCrossings { crossings })
    }

    /// Every link-knot crossing's moving-boundary terms of orders two and three
    /// over the intercept and the primary coordinates: the terms
    /// `accumulate_primary_third_cell_moments` misses when it sums cell by cell.
    pub(super) fn standard_normal_flex_third_calibration_crossings(
        &self,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        cells: &[CachedDenestedCellMoments],
    ) -> Result<ThirdCalibrationCrossings, String> {
        use ExplicitSlot::{Coordinate, Intercept};
        let r = primary.total;
        let still = Array1::<f64>::zeros(r);
        let partition: Vec<exact_kernel::DenestedPartitionCell> =
            cells.iter().map(|entry| entry.partition_cell).collect();
        let crossings = self.standard_normal_flex_calibration_crossings(
            primary,
            a,
            b,
            &partition,
            [&still, &still],
        )?;
        let mut terms = ThirdCalibrationCrossings {
            r,
            aa: crossings.partial(&[Intercept, Intercept]),
            aaa: crossings.partial(&[Intercept, Intercept, Intercept]),
            au: vec![0.0; r],
            aau: vec![0.0; r],
            uv: vec![0.0; r * r],
            auv: vec![0.0; r * r],
            uvw: vec![0.0; r * r * r],
        };
        for p in 1..r {
            let pc = Coordinate(p);
            terms.au[p] = crossings.partial(&[Intercept, pc]);
            terms.aau[p] = crossings.partial(&[Intercept, Intercept, pc]);
            for q in p..r {
                let qc = Coordinate(q);
                let second = crossings.partial(&[pc, qc]);
                let third = crossings.partial(&[Intercept, pc, qc]);
                for (k, l) in [(p, q), (q, p)] {
                    terms.uv[k * r + l] = second;
                    terms.auv[k * r + l] = third;
                }
                for s in q..r {
                    let value = crossings.partial(&[pc, qc, Coordinate(s)]);
                    let labels = [p, q, s];
                    for ordering in &ORDERINGS {
                        let flat = (labels[ordering[0]] * r + labels[ordering[1]]) * r
                            + labels[ordering[2]];
                        terms.uvw[flat] = value;
                    }
                }
            }
        }
        Ok(terms)
    }

    /// Adds every link-knot crossing's moving-boundary terms to the explicit
    /// calibration partials of orders two and three that
    /// `accumulate_primary_third_cell_moments` sums cell by cell. A directional
    /// partial is linear in its direction, so it contracts the coordinate terms.
    pub(super) fn add_standard_normal_flex_third_calibration_crossings(
        &self,
        primary: &PrimarySlices,
        a: f64,
        b: f64,
        cells: &[CachedDenestedCellMoments],
        row_dirs: &[Array1<f64>],
        f_aa: &mut f64,
        f_au: &mut Array1<f64>,
        f_uv: &mut Array2<f64>,
        f_a_dir: &mut [f64],
        f_aa_dir: &mut [f64],
        f_au_dir: &mut [f64],
        f_uv_dir: &mut [f64],
        f_aaa: &mut f64,
        f_aau: &mut Array1<f64>,
        f_auv: &mut Array2<f64>,
    ) -> Result<(), String> {
        let terms = self.standard_normal_flex_third_calibration_crossings(primary, a, b, cells)?;
        terms.add_base(f_aa, f_au, f_uv, f_aaa, f_aau, f_auv);
        let r = terms.r;
        for (direction, dir) in row_dirs.iter().enumerate() {
            for s in 1..r {
                let weight = dir[s];
                if weight == 0.0 {
                    continue;
                }
                f_a_dir[direction] += weight * terms.au[s];
                f_aa_dir[direction] += weight * terms.aau[s];
                for p in 1..r {
                    f_au_dir[direction * r + p] += weight * terms.auv[p * r + s];
                    for q in 1..r {
                        f_uv_dir[(direction * r + p) * r + q] +=
                            weight * terms.uvw[(p * r + q) * r + s];
                    }
                }
            }
        }
        Ok(())
    }
}

/// The link-knot crossings' moving-boundary terms of orders two and three over
/// the intercept `a` and the primary coordinates, dense and symmetric:
/// `aa = B[a,a]`, `au[p] = B[a,p]`, `uv[p·r + q] = B[p,q]`, `aaa = B[a,a,a]`,
/// `aau[p] = B[a,a,p]`, `auv[p·r + q] = B[a,p,q]` and `uvw[(p·r + q)·r + s] = B[p,q,s]`.
/// The marginal coordinate moves no crossing, so its entries stay zero.
pub(super) struct ThirdCalibrationCrossings {
    r: usize,
    aa: f64,
    aaa: f64,
    au: Vec<f64>,
    aau: Vec<f64>,
    uv: Vec<f64>,
    auv: Vec<f64>,
    uvw: Vec<f64>,
}

impl ThirdCalibrationCrossings {
    /// Adds the direction-free terms to the accumulated partials.
    pub(super) fn add_base(
        &self,
        f_aa: &mut f64,
        f_au: &mut Array1<f64>,
        f_uv: &mut Array2<f64>,
        f_aaa: &mut f64,
        f_aau: &mut Array1<f64>,
        f_auv: &mut Array2<f64>,
    ) {
        let r = self.r;
        *f_aa += self.aa;
        *f_aaa += self.aaa;
        for p in 1..r {
            f_au[p] += self.au[p];
            f_aau[p] += self.aau[p];
            for q in 1..r {
                f_uv[[p, q]] += self.uv[p * r + q];
                f_auv[[p, q]] += self.auv[p * r + q];
            }
        }
    }

    /// The reverse of the directional terms. A directional partial is
    /// `Σ_s dir_s·B[…, s]`, so its adjoint adds `adjoint·B[…, s]` to
    /// `direction_adjoint[s]`. `adj_f_uv_dir` holds the upper triangle the
    /// forward pass reads.
    pub(super) fn add_direction_adjoint(
        &self,
        adj_f_a_dir: f64,
        adj_f_aa_dir: f64,
        adj_f_au_dir: &[f64],
        adj_f_uv_dir: &Array2<f64>,
        direction_adjoint: &mut [f64],
    ) {
        let r = self.r;
        for s in 1..r {
            let mut total = adj_f_a_dir * self.au[s] + adj_f_aa_dir * self.aau[s];
            for p in 1..r {
                total += adj_f_au_dir[p] * self.auv[p * r + s];
                for q in p..r {
                    total += adj_f_uv_dir[[p, q]] * self.uvw[(p * r + q) * r + s];
                }
            }
            direction_adjoint[s] += total;
        }
    }
}
