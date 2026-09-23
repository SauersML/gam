//! The calibration cells a BMS FLEX row's third and fourth contractions are
//! summed over, on either latent law (gam#3290).
//!
//! The calibration `F(a, θ) = ∫ Φ(η(z; a, θ)) dμ(z) − μ(q)` is split at the
//! row's partition cells. On each cell `η` is one cubic in `z`, so every
//! derivative of `F` is a sum over cells of `∫ W(η)·P(z) dμ`: a Hermite weight
//! of `η` times a product of the cell's coefficient polynomials, which is a
//! cubic per primary drawn from the score span at `z` and the link span at
//! `u = a + b·z`. A link column therefore reaches the row only through its
//! span's column table and the cell's measure, not as a dense jet
//! axis, and a cell touches only the slope and the score and link columns whose
//! spans cover it.
//!
//! Under the standard-normal law `dμ = φ(z) dz` and the cell's moments are the
//! kernel's integrals ([`CachedDenestedCellMoments`]). Under an empirical law
//! `dμ` is the grid's point masses, and the same contractions are node sums over
//! the few nodes the cell holds ([`EmpiricalCalibrationCell`]), so both
//! laws share one lowering and the empirical rows no longer carry the dense
//! `r`-wide row-program jets.

use super::family::BernoulliMarginalSlopeFamily;
use super::hessian_paths::CachedDenestedCellMoments;
use super::*;
use crate::latent_anchor::CalibrationUnit;

/// `p(z)` by Horner's rule, for a coefficient polynomial in the global `z`.
///
/// A separate multiply and add, not `f64::mul_add`: on the baseline x86-64
/// target `mul_add` lowers to a call into the soft `fma` routine, which was
/// over a third of this route's profile (gam#3290).
#[inline]
fn horner(coefficients: &[f64], z: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |acc, &coefficient| acc * z + coefficient)
}

/// Add `weight·(1, z, z², z³)` to a cubic's coefficient adjoint.
#[inline]
fn add_power_adjoint(target: &mut [f64; 4], weight: f64, z: f64) {
    let mut power = weight;
    for slot in target.iter_mut() {
        *slot += power;
        power *= z;
    }
}

/// One partition cell of an empirical-law row: the nodes of the row's grid it
/// holds, each with its weighted density `w_j·φ(η(z_j))/N` in the row's
/// calibration unit `N = Φ(−|q|)` (gam#3639).
///
/// Under a point-mass law a cell's integral `∫ W(η)·P(z) dμ` is the node sum
/// `Σ_j w_j·W(η(z_j))·P(z_j)`, so each contraction evaluates its coefficient
/// polynomials and the Hermite weights of `η` at the cell's few nodes directly
/// (the grid's `G` nodes spread over the row's cells). That replaces the
/// kernel's moment contraction, whose products of up to five cubics read
/// moments through degree 21 however few nodes the cell holds, and it keeps
/// every term at the conditioning of a pointwise evaluation. The sums equal
/// the kernel's contractions over the nodes' moments term by term: the kernel
/// returns `⟨P, m⟩/2π` with `m_k = 2π·Σ_j w_j·z_j^k·φ(η_j)`.
pub(super) struct EmpiricalCalibrationCell {
    pub(super) partition_cell: exact_kernel::DenestedPartitionCell,
    /// `(z_j, w_j·φ(η(z_j))/N)` for each node in the cell.
    nodes: Vec<[f64; 2]>,
}

impl EmpiricalCalibrationCell {
    #[inline]
    fn eta(cell: exact_kernel::DenestedCubicCell, z: f64) -> f64 {
        ((cell.c3 * z + cell.c2) * z + cell.c1) * z + cell.c0
    }

    fn first(&self, r: &[f64]) -> f64 {
        self.nodes
            .iter()
            .map(|&[z, density]| density * horner(r, z))
            .sum()
    }

    fn second(&self, cell: exact_kernel::DenestedCubicCell, r: &[f64], s: &[f64], rs: &[f64]) -> f64 {
        self.nodes
            .iter()
            .map(|&[z, density]| {
                let eta = Self::eta(cell, z);
                density * (horner(rs, z) - eta * horner(r, z) * horner(s, z))
            })
            .sum()
    }

    fn third(
        &self,
        cell: exact_kernel::DenestedCubicCell,
        r: &[f64],
        s: &[f64],
        t: &[f64],
        rs: &[f64],
        rt: &[f64],
        st: &[f64],
        rst: &[f64],
    ) -> f64 {
        self.nodes
            .iter()
            .map(|&[z, density]| {
                let eta = Self::eta(cell, z);
                let (pr, ps, pt) = (horner(r, z), horner(s, z), horner(t, z));
                let linear = horner(rs, z) * pt + horner(rt, z) * ps + horner(st, z) * pr;
                density * (horner(rst, z) - eta * linear + (eta * eta - 1.0) * pr * ps * pt)
            })
            .sum()
    }

    fn fourth(
        &self,
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
    ) -> f64 {
        self.nodes
            .iter()
            .map(|&[z, density]| {
                let eta = Self::eta(cell, z);
                let (pr, ps, pt, pu) = (horner(r, z), horner(s, z), horner(t, z), horner(u, z));
                let (prs, prt, pru) = (horner(rs, z), horner(rt, z), horner(ru, z));
                let (pst, psu, ptu) = (horner(st, z), horner(su, z), horner(tu, z));
                let linear = horner(rst, z) * pu
                    + horner(rsu, z) * pt
                    + horner(rtu, z) * ps
                    + horner(stu, z) * pr
                    + prs * ptu
                    + prt * psu
                    + pru * pst;
                let quadratic = prs * pt * pu
                    + prt * ps * pu
                    + pru * ps * pt
                    + pst * pr * pu
                    + psu * pr * pt
                    + ptu * pr * ps;
                let eta_squared = eta * eta;
                density
                    * (horner(rstu, z) - eta * linear
                        + (eta_squared - 1.0) * quadratic
                        + (3.0 - eta_squared) * eta * pr * ps * pt * pu)
            })
            .sum()
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
            Self::Nodes(entry) => Ok(entry.first(r)),
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
            Self::Nodes(entry) => Ok(entry.second(cell, r, s, rs)),
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
            Self::Nodes(entry) => Ok(entry.third(cell, r, s, t, rs, rt, st, rst)),
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
            Self::Nodes(entry) => Ok(entry.fourth(
                cell, r, s, t, u, rs, rt, ru, st, su, tu, rst, rsu, rtu, stu, rstu,
            )),
        }
    }

    /// [`BernoulliMarginalSlopeFamily::add_cell_second_direction_adjoint`]
    /// over this cell: the adjoints of a second contraction on its free first
    /// slot and its second-order slot, in the global-`z` coefficient basis.
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
                // `Σ_j d_j·(RS − η·R·S)`: the second-order slot enters
                // linearly, the free first slot through `−η·R`.
                for &[z, density] in &entry.nodes {
                    let weight = scalar_adjoint * density;
                    let eta = EmpiricalCalibrationCell::eta(cell, z);
                    add_power_adjoint(second_adjoint, weight, z);
                    add_power_adjoint(first_s_adjoint, -weight * eta * horner(first_r, z), z);
                }
                Ok(())
            }
        }
    }

    /// [`BernoulliMarginalSlopeFamily::add_cell_third_direction_adjoint`]
    /// over this cell, in the global-`z` coefficient basis.
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
                // `Σ_j d_j·(RST − η(RS·T + RT·S + ST·R) + (η² − 1)·R·S·T)`,
                // differentiated in the coefficients of `T`, `RT`, `ST` and
                // `RST`.
                for &[z, density] in &entry.nodes {
                    let weight = scalar_adjoint * density;
                    let eta = EmpiricalCalibrationCell::eta(cell, z);
                    let (pr, ps) = (horner(first_r, z), horner(first_s, z));
                    add_power_adjoint(third_rst_adjoint, weight, z);
                    add_power_adjoint(second_rt_adjoint, -weight * eta * ps, z);
                    add_power_adjoint(second_st_adjoint, -weight * eta * pr, z);
                    add_power_adjoint(
                        first_t_adjoint,
                        weight * ((eta * eta - 1.0) * pr * ps - eta * horner(second_rs, z)),
                        z,
                    );
                }
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
    /// `(a, b, β)`, each an [`EmpiricalCalibrationCell`]. The grid is
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
            let nodes = grid.nodes[begin..end]
                .iter()
                .zip(&grid.weights[begin..end])
                .map(|(&node, &weight)| {
                    [node, weight * unit.density(partition_cell.cell.eta(node))]
                })
                .collect();
            cells.push(EmpiricalCalibrationCell {
                partition_cell,
                nodes,
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
    use crate::bms::factored_link_block_3290_tests::EMPIRICAL_NODE_TERM_OPERATIONS;

    /// gam#3290: an empirical cell's node sums are the kernel's contractions
    /// over the same nodes' moments `m_k = 2π·Σ_j d_j·z_j^k`, term by term, for
    /// every order the row contractions read. The kernel evaluates the
    /// products as polynomial convolutions against the moments, the node route
    /// pointwise, so they agree to the rounding of the wider of the two
    /// accumulations on the sum of the terms' magnitudes.
    #[test]
    fn node_sums_are_the_kernel_contractions_over_the_node_moments_3290() {
        let cell = exact_kernel::DenestedCubicCell {
            left: 0.2,
            right: 0.9,
            c0: -0.4,
            c1: 0.7,
            c2: -0.15,
            c3: 0.05,
        };
        let partition_cell = exact_kernel::DenestedPartitionCell {
            cell,
            score_span: exact_kernel::LocalSpanCubic {
                left: 0.0,
                right: 1.0,
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            },
            link_span: exact_kernel::LocalSpanCubic {
                left: 0.0,
                right: 1.0,
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            },
            left_edge: exact_kernel::PartitionEdge::Fixed(0.2),
            right_edge: exact_kernel::PartitionEdge::Fixed(0.9),
        };
        let nodes = vec![[0.25, 0.11], [0.4, 0.23], [0.61, 0.19], [0.85, 0.07]];
        let empirical = EmpiricalCalibrationCell {
            partition_cell,
            nodes: nodes.clone(),
        };
        let degree = 9 + 4 * 3;
        let moments: Vec<f64> = (0..=degree)
            .map(|k| {
                std::f64::consts::TAU
                    * nodes.iter().map(|&[z, d]| d * z.powi(k as i32)).sum::<f64>()
            })
            .collect();
        let poly = |seed: f64| [0.3 * seed, -0.2 + 0.1 * seed, 0.15 * seed, -0.05 * seed];
        let [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15] =
            std::array::from_fn(|index| poly(1.0 + index as f64));
        // Every term of every order is bounded by the fourth-order formula with
        // each factor replaced by its absolute-coefficient polynomial at `|z|`
        // and each Hermite weight by its absolute-coefficient one at `|η|`
        // (the slots of the lower orders are among the fifteen). The kernel's
        // convolutions round each such term, and add them, no worse; its
        // quartic chain is five accumulation stages over `degree + 1` moments.
        let abs_poly = |p: &[f64; 4], z: f64| -> f64 {
            p.iter().rev().fold(0.0, |acc, c| acc * z.abs() + c.abs())
        };
        let all = [&p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13, &p14, &p15];
        let magnitude: f64 = nodes
            .iter()
            .map(|&[z, d]| {
                let eta = abs_poly(&[cell.c0, cell.c1, cell.c2, cell.c3], z);
                let widest = all.iter().fold(0.0_f64, |acc, p| acc.max(abs_poly(p, z)));
                d * (widest
                    + 7.0 * eta * widest * widest
                    + 6.0 * (eta * eta + 1.0) * widest.powi(3)
                    + (3.0 * eta + eta.powi(3)) * widest.powi(4))
            })
            .sum();
        let band = gam_math::roundoff::accumulation_growth(
            nodes.len() * EMPIRICAL_NODE_TERM_OPERATIONS + 5 * (degree + 1),
        ) * magnitude;
        let checks = [
            (
                empirical.first(&p1),
                exact_kernel::cell_first_derivative_from_moments(&p1, &moments).expect("first"),
            ),
            (
                empirical.second(cell, &p1, &p2, &p3),
                exact_kernel::cell_second_derivative_from_moments(cell, &p1, &p2, &p3, &moments)
                    .expect("second"),
            ),
            (
                empirical.third(cell, &p1, &p2, &p3, &p4, &p5, &p6, &p7),
                exact_kernel::cell_third_derivative_from_moments(
                    cell, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &moments,
                )
                .expect("third"),
            ),
            (
                empirical.fourth(
                    cell, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13,
                    &p14, &p15,
                ),
                exact_kernel::cell_fourth_derivative_from_moments(
                    cell, &p1, &p2, &p3, &p4, &p5, &p6, &p7, &p8, &p9, &p10, &p11, &p12, &p13,
                    &p14, &p15, &moments,
                )
                .expect("fourth"),
            ),
        ];
        for (order, (nodes_value, kernel_value)) in checks.into_iter().enumerate() {
            assert!(
                nodes_value != 0.0 && (nodes_value - kernel_value).abs() <= band,
                "order {}: node sum {nodes_value:+.17e} vs kernel {kernel_value:+.17e} (band {band:e})",
                order + 1
            );
        }

        // The adjoints are the gradients of the node sums in the coefficients:
        // contracting them with any coefficient vector reproduces the sum
        // with that slot replaced, since each slot enters linearly.
        let mut first_s = [0.0; 4];
        let mut second = [0.0; 4];
        CalibrationCell::Nodes(&empirical)
            .second_direction_adjoint(cell, &p1, 1.0, &mut first_s, &mut second)
            .expect("second adjoint");
        let dot = |adjoint: &[f64; 4], poly: &[f64; 4]| -> f64 {
            adjoint.iter().zip(poly).map(|(a, p)| a * p).sum()
        };
        let second_value = empirical.second(cell, &p1, &p2, &p3);
        let adjoint_value = dot(&first_s, &p2) + dot(&second, &p3);
        assert!(
            (second_value - adjoint_value).abs() <= band,
            "second adjoint {adjoint_value:+.17e} vs {second_value:+.17e}"
        );
        let mut t_adjoint = [0.0; 4];
        let mut rt_adjoint = [0.0; 4];
        let mut st_adjoint = [0.0; 4];
        let mut rst_adjoint = [0.0; 4];
        CalibrationCell::Nodes(&empirical)
            .third_direction_adjoint(
                cell,
                &p1,
                &p2,
                &p4,
                1.0,
                &mut t_adjoint,
                &mut rt_adjoint,
                &mut st_adjoint,
                &mut rst_adjoint,
            )
            .expect("third adjoint");
        let third_value = empirical.third(cell, &p1, &p2, &p3, &p4, &p5, &p6, &p7);
        let adjoint_value = dot(&t_adjoint, &p3)
            + dot(&rt_adjoint, &p5)
            + dot(&st_adjoint, &p6)
            + dot(&rst_adjoint, &p7);
        assert!(
            (third_value - adjoint_value).abs() <= band,
            "third adjoint {adjoint_value:+.17e} vs {third_value:+.17e}"
        );
    }
}
