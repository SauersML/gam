//! The calibration cells a BMS FLEX row's third and fourth contractions are
//! summed over, on either latent law (gam#3290).
//!
//! The calibration `F(a, θ) = ∫ Φ(η(z; a, θ)) dμ(z) − μ(q)` is split at the
//! row's partition cells. On each cell `η` is one cubic in `z`, so every
//! derivative of `F` is a sum over cells of `∫ W(η)·P(z) dμ`: a Hermite weight
//! of `η` times a product of the cell's coefficient polynomials, which is a
//! cubic per primary drawn from the score span at `z` and the link span at
//! `u = a + b·z`. A link column therefore reaches the row only through its
//! span's column table and the scalar moments of the cell, not as a dense jet
//! axis, and a cell touches only the slope and the score and link columns whose
//! spans cover it.
//!
//! Under the standard-normal law `dμ = φ(z) dz` and the cell's moments are the
//! kernel's integrals ([`CachedDenestedCellMoments`]). Under an empirical law
//! `dμ` is the grid's point masses, and the same contractions read the discrete
//! moments of the nodes the cell holds ([`EmpiricalCalibrationCell`]), so both
//! laws share one lowering and the empirical rows no longer carry the dense
//! `r`-wide row-program jets.

use super::family::BernoulliMarginalSlopeFamily;
use super::hessian_paths::CachedDenestedCellMoments;
use super::*;
use crate::latent_anchor::CalibrationUnit;

/// Degree through which an empirical cell's moments are formed: the fourth
/// contraction's `(3η − η³) ⊗ r ⊗ s ⊗ t ⊗ u` term, a degree-nine Hermite weight
/// of the cubic `η` times four cubic coefficient polynomials, reads moment 21.
/// Every lower-order contraction reads a prefix.
pub(super) const EMPIRICAL_CELL_MOMENT_DEGREE: usize = 9 + 4 * 3;

/// The polynomial `p(center + t)` in `t`, for `p` of degree at most three,
/// by repeated synthetic division.
fn shift_cubic(coefficients: &[f64], center: f64) -> Result<[f64; 4], String> {
    if coefficients.len() > 4 {
        return Err(format!(
            "empirical calibration cell received a degree-{} coefficient polynomial; cells are cubic",
            coefficients.len() - 1
        ));
    }
    let mut shifted = [0.0; 4];
    shifted[..coefficients.len()].copy_from_slice(coefficients);
    for start in 0..3 {
        for degree in (start..3).rev() {
            shifted[degree] += center * shifted[degree + 1];
        }
    }
    Ok(shifted)
}

/// Pull an adjoint on the coefficients of `q(t) = p(center + t)` back onto
/// `p`'s: `q_j = Σ_{i≥j} C(i, j)·center^{i−j}·p_i`, so `p̄_i` gains
/// `Σ_{j≤i} C(i, j)·center^{i−j}·q̄_j`.
fn add_shift_adjoint(target: &mut [f64; 4], shifted_adjoint: &[f64; 4], center: f64) {
    const BINOMIAL: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 0.0],
        [1.0, 3.0, 3.0, 1.0],
    ];
    for (degree, slot) in target.iter_mut().enumerate() {
        let mut power = 1.0;
        let mut total = 0.0;
        for lower in (0..=degree).rev() {
            total += BINOMIAL[degree][lower] * power * shifted_adjoint[lower];
            power *= center;
        }
        *slot += total;
    }
}

/// One partition cell of an empirical-law row, holding the nodes of the
/// row's grid it contains as moments about their own midpoint `c`:
/// `m_k = τ·Σ_j w_j·(z_j − c)^k·φ(η(z_j))/N`, in the row's calibration unit
/// `N = Φ(−|q|)` (gam#3639).
///
/// These are the kernel's moments `∫ z^k e^{−(z² + η²)/2} dz = 2π∫ z^k φ(z)φ(η) dz`
/// with `φ(z) dz` replaced by the point masses and the origin moved to `c`, so
/// the kernel's contractions (`cell_*_derivative_from_moments`) apply
/// unchanged once the cell's cubic and every coefficient polynomial are
/// re-expanded about `c`. Centring keeps the moments and the re-expanded
/// polynomials at the scale of the cell's half-width: expanded about zero, a
/// degree-21 product read at a node near `|z| = 2.6` cancels terms up to
/// `(1 + 2.6 + 2.6² + 2.6³)^7` times its value.
pub(super) struct EmpiricalCalibrationCell {
    pub(super) partition_cell: exact_kernel::DenestedPartitionCell,
    center: f64,
    moments: Vec<f64>,
}

impl EmpiricalCalibrationCell {
    fn shifted(&self, coefficients: &[f64]) -> Result<[f64; 4], String> {
        shift_cubic(coefficients, self.center)
    }

    fn shifted_cell(
        &self,
        cell: exact_kernel::DenestedCubicCell,
    ) -> Result<exact_kernel::DenestedCubicCell, String> {
        let [c0, c1, c2, c3] = self.shifted(&[cell.c0, cell.c1, cell.c2, cell.c3])?;
        Ok(exact_kernel::DenestedCubicCell {
            left: cell.left - self.center,
            right: cell.right - self.center,
            c0,
            c1,
            c2,
            c3,
        })
    }
}

/// One calibration cell of a row on either law, with the kernel's
/// first-through-fourth derivative contractions over it.
#[derive(Clone, Copy)]
pub(super) enum CalibrationCell<'a> {
    Moments(&'a CachedDenestedCellMoments),
    Nodes(&'a EmpiricalCalibrationCell),
}

impl CalibrationCell<'_> {
    #[inline]
    pub(super) fn partition_cell(self) -> exact_kernel::DenestedPartitionCell {
        match self {
            Self::Moments(entry) => entry.partition_cell,
            Self::Nodes(entry) => entry.partition_cell,
        }
    }

    pub(super) fn first(self, r: &[f64]) -> Result<f64, String> {
        match self {
            Self::Moments(entry) => {
                exact_kernel::cell_first_derivative_from_moments(r, &entry.state.moments)
            }
            Self::Nodes(entry) => {
                exact_kernel::cell_first_derivative_from_moments(&entry.shifted(r)?, &entry.moments)
            }
        }
    }

    pub(super) fn second(
        self,
        cell: exact_kernel::DenestedCubicCell,
        r: &[f64],
        s: &[f64],
        rs: &[f64],
    ) -> Result<f64, String> {
        match self {
            Self::Moments(entry) => exact_kernel::cell_second_derivative_from_moments(
                cell,
                r,
                s,
                rs,
                &entry.state.moments,
            ),
            Self::Nodes(entry) => exact_kernel::cell_second_derivative_from_moments(
                entry.shifted_cell(cell)?,
                &entry.shifted(r)?,
                &entry.shifted(s)?,
                &entry.shifted(rs)?,
                &entry.moments,
            ),
        }
    }

    pub(super) fn third(
        self,
        cell: exact_kernel::DenestedCubicCell,
        r: &[f64],
        s: &[f64],
        t: &[f64],
        rs: &[f64],
        rt: &[f64],
        st: &[f64],
        rst: &[f64],
    ) -> Result<f64, String> {
        match self {
            Self::Moments(entry) => exact_kernel::cell_third_derivative_from_moments(
                cell,
                r,
                s,
                t,
                rs,
                rt,
                st,
                rst,
                &entry.state.moments,
            ),
            Self::Nodes(entry) => exact_kernel::cell_third_derivative_from_moments(
                entry.shifted_cell(cell)?,
                &entry.shifted(r)?,
                &entry.shifted(s)?,
                &entry.shifted(t)?,
                &entry.shifted(rs)?,
                &entry.shifted(rt)?,
                &entry.shifted(st)?,
                &entry.shifted(rst)?,
                &entry.moments,
            ),
        }
    }

    pub(super) fn fourth(
        self,
        cell: exact_kernel::DenestedCubicCell,
        r: &[f64],
        s: &[f64],
        t: &[f64],
        u: &[f64],
        rs: &[f64],
        rt: &[f64],
        ru: &[f64],
        st: &[f64],
        su: &[f64],
        tu: &[f64],
        rst: &[f64],
        rsu: &[f64],
        rtu: &[f64],
        stu: &[f64],
        rstu: &[f64],
    ) -> Result<f64, String> {
        match self {
            Self::Moments(entry) => exact_kernel::cell_fourth_derivative_from_moments(
                cell,
                r,
                s,
                t,
                u,
                rs,
                rt,
                ru,
                st,
                su,
                tu,
                rst,
                rsu,
                rtu,
                stu,
                rstu,
                &entry.state.moments,
            ),
            Self::Nodes(entry) => exact_kernel::cell_fourth_derivative_from_moments(
                entry.shifted_cell(cell)?,
                &entry.shifted(r)?,
                &entry.shifted(s)?,
                &entry.shifted(t)?,
                &entry.shifted(u)?,
                &entry.shifted(rs)?,
                &entry.shifted(rt)?,
                &entry.shifted(ru)?,
                &entry.shifted(st)?,
                &entry.shifted(su)?,
                &entry.shifted(tu)?,
                &entry.shifted(rst)?,
                &entry.shifted(rsu)?,
                &entry.shifted(rtu)?,
                &entry.shifted(stu)?,
                &entry.shifted(rstu)?,
                &entry.moments,
            ),
        }
    }
}

impl CalibrationCell<'_> {
    /// [`BernoulliMarginalSlopeFamily::add_cell_second_direction_adjoint`]
    /// over this cell: the adjoints of a second contraction on its free first
    /// slot and its second-order slot, in the global-`z` coefficient basis on
    /// both laws.
    pub(super) fn second_direction_adjoint(
        self,
        cell: exact_kernel::DenestedCubicCell,
        first_r: &[f64; 4],
        scalar_adjoint: f64,
        first_s_adjoint: &mut [f64; 4],
        second_adjoint: &mut [f64; 4],
    ) -> Result<(), String> {
        match self {
            Self::Moments(entry) => BernoulliMarginalSlopeFamily::add_cell_second_direction_adjoint(
                cell,
                first_r,
                &entry.state.moments,
                scalar_adjoint,
                first_s_adjoint,
                second_adjoint,
            ),
            Self::Nodes(entry) => {
                let mut shifted_first = [0.0; 4];
                let mut shifted_second = [0.0; 4];
                BernoulliMarginalSlopeFamily::add_cell_second_direction_adjoint(
                    entry.shifted_cell(cell)?,
                    &entry.shifted(first_r)?,
                    &entry.moments,
                    scalar_adjoint,
                    &mut shifted_first,
                    &mut shifted_second,
                )?;
                add_shift_adjoint(first_s_adjoint, &shifted_first, entry.center);
                add_shift_adjoint(second_adjoint, &shifted_second, entry.center);
                Ok(())
            }
        }
    }

    /// [`BernoulliMarginalSlopeFamily::add_cell_third_direction_adjoint`]
    /// over this cell, in the global-`z` coefficient basis on both laws.
    pub(super) fn third_direction_adjoint(
        self,
        cell: exact_kernel::DenestedCubicCell,
        first_r: &[f64; 4],
        first_s: &[f64; 4],
        second_rs: &[f64; 4],
        scalar_adjoint: f64,
        first_t_adjoint: &mut [f64; 4],
        second_rt_adjoint: &mut [f64; 4],
        second_st_adjoint: &mut [f64; 4],
        third_rst_adjoint: &mut [f64; 4],
    ) -> Result<(), String> {
        match self {
            Self::Moments(entry) => BernoulliMarginalSlopeFamily::add_cell_third_direction_adjoint(
                cell,
                first_r,
                first_s,
                second_rs,
                &entry.state.moments,
                scalar_adjoint,
                first_t_adjoint,
                second_rt_adjoint,
                second_st_adjoint,
                third_rst_adjoint,
            ),
            Self::Nodes(entry) => {
                let mut shifted = [[0.0; 4]; 4];
                let [first_t, second_rt, second_st, third_rst] = &mut shifted;
                BernoulliMarginalSlopeFamily::add_cell_third_direction_adjoint(
                    entry.shifted_cell(cell)?,
                    &entry.shifted(first_r)?,
                    &entry.shifted(first_s)?,
                    &entry.shifted(second_rs)?,
                    &entry.moments,
                    scalar_adjoint,
                    first_t,
                    second_rt,
                    second_st,
                    third_rst,
                )?;
                add_shift_adjoint(first_t_adjoint, &shifted[0], entry.center);
                add_shift_adjoint(second_rt_adjoint, &shifted[1], entry.center);
                add_shift_adjoint(second_st_adjoint, &shifted[2], entry.center);
                add_shift_adjoint(third_rst_adjoint, &shifted[3], entry.center);
                Ok(())
            }
        }
    }
}

/// The marginal's `q`-derivatives `μ′..μ⁗` in the unit the row's calibration
/// is written in: `μ = Φ(q)`'s own on the standard-normal cells route,
/// `Φ(q)/N`'s on an empirical law (gam#3639), as the order-two lowering reads
/// them.
#[derive(Clone, Copy, Debug)]
pub(super) struct CalibrationMarginal {
    pub(super) mu1: f64,
    pub(super) mu2: f64,
    pub(super) mu3: f64,
    pub(super) mu4: f64,
}

/// What a FLEX row's third and fourth contractions sum over: the marginal's
/// derivatives in the calibration's unit, and, on an empirical law, the row's
/// node cells. `None` is the standard-normal cells route, whose cell moments
/// the caller draws from its own bundle.
pub(super) struct RowCalibrationRoute {
    pub(super) marginal: CalibrationMarginal,
    pub(super) empirical_cells: Option<Vec<EmpiricalCalibrationCell>>,
}

impl BernoulliMarginalSlopeFamily {
    /// The calibration route of `row` at the intercept `a`: the row context's
    /// root, which the order-two lowering reads on both laws.
    pub(super) fn row_calibration_route(
        &self,
        row: usize,
        q: f64,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<RowCalibrationRoute, String> {
        let marginal = self.marginal_link_map(q)?;
        let Some(grid) = self.training_row_grid(row)? else {
            return Ok(RowCalibrationRoute {
                marginal: CalibrationMarginal {
                    mu1: marginal.mu1,
                    mu2: marginal.mu2,
                    mu3: marginal.mu3,
                    mu4: marginal.mu4,
                },
                empirical_cells: None,
            });
        };
        let unit = CalibrationUnit::new(marginal.q);
        let stack = unit.marginal_stack();
        Ok(RowCalibrationRoute {
            marginal: CalibrationMarginal {
                mu1: stack[1],
                mu2: stack[2],
                mu3: stack[3],
                mu4: stack[4],
            },
            empirical_cells: Some(self.empirical_calibration_cells(
                a, b, beta_h, beta_w, &grid, unit,
            )?),
        })
    }

    /// The empirical grid's nodes bucketed into the row's partition cells at
    /// `(a, b, β)`, each as [`EmpiricalCalibrationCell`] moments. The grid is
    /// sorted (its constructor's invariant), so every node falls in exactly
    /// one cell by one forward pass, the bucketing the order-two lowering
    /// uses; a cell holding no node contributes nothing and is dropped.
    pub(super) fn empirical_calibration_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        grid: &EmpiricalZGrid,
        unit: CalibrationUnit,
    ) -> Result<Vec<EmpiricalCalibrationCell>, String> {
        let partition = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        let mut cells = Vec::with_capacity(partition.len());
        let mut cursor = 0usize;
        for partition_cell in partition {
            let left = partition_cell.cell.left;
            let right = partition_cell.cell.right;
            let begin = cursor;
            if begin < grid.nodes.len() && grid.nodes[begin] < left {
                return Err(format!(
                    "empirical calibration cells found grid node {} below the next cell's left edge {left}",
                    grid.nodes[begin]
                ));
            }
            let end = begin + grid.nodes[begin..].partition_point(|&node| node < right);
            cursor = end;
            if begin == end {
                continue;
            }
            let center = 0.5 * (grid.nodes[begin] + grid.nodes[end - 1]);
            let mut moments = vec![0.0; EMPIRICAL_CELL_MOMENT_DEGREE + 1];
            for (&node, &weight) in grid.nodes[begin..end].iter().zip(&grid.weights[begin..end]) {
                let offset = node - center;
                let mut term = std::f64::consts::TAU
                    * weight
                    * unit.density(partition_cell.cell.eta(node));
                for moment in moments.iter_mut() {
                    *moment += term;
                    term *= offset;
                }
            }
            cells.push(EmpiricalCalibrationCell {
                partition_cell,
                center,
                moments,
            });
        }
        if cursor != grid.nodes.len() {
            return Err(format!(
                "empirical calibration cells consumed {cursor} of {} sorted grid nodes",
                grid.nodes.len()
            ));
        }
        Ok(cells)
    }
}

/// The primaries a calibration cell's coefficient jet can be nonzero on: the
/// slope, always, then every score and link column whose span the cell's
/// probe point lands in, ascending. Every other primary's coefficient
/// polynomials are identically zero on the cell, so each contraction with it
/// in a slot is an exact zero, and a cell's work is quadratic in this set
/// rather than in the row's primary width.
pub(super) fn sorted_active_primaries(active: &mut Vec<usize>) {
    active.sort_unstable();
    active.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `shift_cubic` re-expands `p(z)` about `c`: `q(t) = p(c + t)` at every
    /// `t`, to the rounding of the two Horner evaluations.
    #[test]
    fn shifted_cubic_is_the_same_polynomial_about_the_new_origin_3290() {
        let p = [0.7, -1.3, 0.45, -0.2];
        for center in [-2.4, -0.3, 0.0, 1.1, 2.6] {
            let q = shift_cubic(&p, center).expect("cubic");
            for t in [-0.4, -0.05, 0.0, 0.2, 0.37] {
                let z: f64 = center + t;
                let direct = ((p[3] * z + p[2]) * z + p[1]) * z + p[0];
                let shifted = ((q[3] * t + q[2]) * t + q[1]) * t + q[0];
                let scale = p[0].abs()
                    + p[1].abs() * z.abs()
                    + p[2].abs() * z * z
                    + p[3].abs() * z.abs().powi(3);
                let band = gam_math::roundoff::accumulation_growth(24) * scale;
                assert!(
                    (direct - shifted).abs() <= band,
                    "center={center} t={t}: {direct:+.17e} vs {shifted:+.17e} (band {band:e})"
                );
            }
        }
        assert!(shift_cubic(&[0.0; 5], 1.0).is_err());
    }

    /// `add_shift_adjoint` is the transpose of `shift_cubic`:
    /// `⟨q̄, shift(p)⟩ = ⟨shiftᵀ(q̄), p⟩` for every `p` and `q̄`, to rounding.
    #[test]
    fn shift_adjoint_is_the_transpose_of_the_shift_3290() {
        let p = [0.7, -1.3, 0.45, -0.2];
        let adjoint = [0.3, 1.1, -0.6, 0.25];
        for center in [-2.4, -0.3, 0.0, 1.1, 2.6] {
            let q = shift_cubic(&p, center).expect("cubic");
            let mut pulled = [0.0; 4];
            add_shift_adjoint(&mut pulled, &adjoint, center);
            let forward: f64 = adjoint.iter().zip(&q).map(|(a, b)| a * b).sum();
            let backward: f64 = pulled.iter().zip(&p).map(|(a, b)| a * b).sum();
            let scale: f64 = (0..4)
                .map(|j| {
                    (j..4)
                        .map(|i| (adjoint[j] * p[i]).abs() * 3.0 * center.abs().max(1.0).powi(3))
                        .sum::<f64>()
                })
                .sum();
            let band = gam_math::roundoff::accumulation_growth(24) * scale;
            assert!(
                (forward - backward).abs() <= band,
                "center={center}: {forward:+.17e} vs {backward:+.17e} (band {band:e})"
            );
        }
    }
}
