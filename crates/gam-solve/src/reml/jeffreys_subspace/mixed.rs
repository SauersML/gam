//! Mixed derivatives in a fixed spectral frame. Higher divided differences
//! carry eigenvector motion, including repeated interior eigenvalues, without
//! differentiating a particular choice of eigenvectors.
use super::*;

/// Divided differences of the capped inverse and its first two floor partials.
/// Same-piece rational identities avoid subtraction at repeated/nearby nodes.
pub(super) fn inverse_difference(nodes: &[f64], floor: f64, floor_order: usize) -> f64 {
    let cap = floor.max(CONDITIONING_GATE_ABSOLUTE_CLEAR);
    let piece = |x: f64| inverse_kernel_piece(x, floor, cap);
    let branch = piece(nodes[0]);
    if nodes.iter().all(|&x| piece(x) == branch) {
        let sign = if nodes.len() % 2 == 1 { 1.0 } else { -1.0 };
        return match branch {
            3 => {
                let coefficient = match floor_order {
                    0 => cap,
                    1 if floor > CONDITIONING_GATE_ABSOLUTE_CLEAR => 1.0,
                    _ => 0.0,
                };
                sign * coefficient
                    * nodes.iter().map(|x| x.recip()).product::<f64>()
                    * nodes.iter().map(|x| x.recip()).sum::<f64>()
            }
            2 => {
                if floor_order == 0 {
                    sign * nodes.iter().map(|x| x.recip()).product::<f64>()
                } else {
                    0.0
                }
            }
            1 => {
                if nodes.len() == 1 {
                    match floor_order {
                        0 => floor.recip(),
                        1 => -floor.recip().powi(2),
                        _ => 2.0 * floor.recip().powi(3),
                    }
                } else {
                    0.0
                }
            }
            _ => {
                let mut product = 1.0;
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for &x in nodes {
                    let z = (floor - x).recip();
                    product *= z;
                    s1 += z;
                    s2 += z * z;
                    s3 += z * z * z;
                }
                match floor_order {
                    0 => floor * product * s1,
                    1 => product * (s1 - floor * (s1 * s1 + s2)),
                    _ => {
                        product
                            * (-2.0 * (s1 * s1 + s2)
                                + floor * (s1 * s1 * s1 + 3.0 * s1 * s2 + 2.0 * s3))
                    }
                }
            }
        };
    }
    let mut storage = [0.0; 4];
    let sorted = &mut storage[..nodes.len()];
    sorted.copy_from_slice(nodes);
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    (inverse_difference(&sorted[1..], floor, floor_order)
        - inverse_difference(&sorted[..last], floor, floor_order))
        / (sorted[last] - sorted[0])
}

/// Which smooth piece of the capped inverse holds `x`: saturated below zero,
/// the floor plateau, the floored inverse, or the capped inverse.
fn inverse_kernel_piece(x: f64, floor: f64, cap: f64) -> u8 {
    if x >= cap {
        3
    } else if x >= floor {
        2
    } else if x >= 0.0 {
        1
    } else {
        0
    }
}

/// Divided differences of the capped inverse on one reduced spectrum, for every
/// node set the Fréchet rows read.
///
/// Each `inverse_frechet_rows` call evaluates the same `[λ_i, λ_k, λ_j]` triples,
/// and the second-order map reads `[λ_i, λ_k, λ_l, λ_j]` for all `m⁴` index tuples
/// of a pair. Recomputing every one by sort-and-recurse was the largest self time
/// of the survival marginal-slope outer Hessian (#979). The spectrum is fixed for a
/// drift base, so pairs and triples are tabulated once in exactly the node order the
/// rows ask for, and a four-node value is formed either from its own same-piece
/// closed form or from the two sorted triples `inverse_difference` would recurse
/// into — the same operations in the same order, so every value is unchanged.
pub(super) struct InverseDividedDifferences {
    m: usize,
    floor: f64,
    cap: f64,
    evals: Vec<f64>,
    pieces: Vec<u8>,
    /// `pairs[order][i·m + j] = inverse_difference(&[λ_i, λ_j], floor, order)`.
    pairs: [Vec<f64>; 3],
    /// `triples[order][(i·m + k)·m + j] = inverse_difference(&[λ_i, λ_k, λ_j], floor, order)`.
    triples: [Vec<f64>; 3],
}

impl InverseDividedDifferences {
    pub(super) fn new(evals: &Array1<f64>, floor: f64) -> Self {
        let values: Vec<f64> = evals.iter().copied().collect();
        let m = values.len();
        let cap = floor.max(CONDITIONING_GATE_ABSOLUTE_CLEAR);
        let pieces = values
            .iter()
            .map(|&x| inverse_kernel_piece(x, floor, cap))
            .collect();
        let pairs = std::array::from_fn(|order| {
            let mut table = Vec::with_capacity(m * m);
            for &left in &values {
                for &right in &values {
                    table.push(inverse_difference(&[left, right], floor, order));
                }
            }
            table
        });
        let triples = std::array::from_fn(|order| {
            let mut table = Vec::with_capacity(m * m * m);
            for &first in &values {
                for &middle in &values {
                    for &last in &values {
                        table.push(inverse_difference(&[first, middle, last], floor, order));
                    }
                }
            }
            table
        });
        Self {
            m,
            floor,
            cap,
            evals: values,
            pieces,
            pairs,
            triples,
        }
    }

    fn triple(&self, order: usize, first: usize, middle: usize, last: usize) -> f64 {
        self.triples[order][(first * self.m + middle) * self.m + last]
    }

    /// `inverse_difference(&[λ_a, λ_b, λ_c, λ_d], floor, 0)`.
    fn quadruple(&self, nodes: [usize; 4]) -> f64 {
        let branch = self.pieces[nodes[0]];
        if nodes.iter().all(|&index| self.pieces[index] == branch) {
            let [a, b, c, d] = nodes.map(|index| self.evals[index]);
            let sign = -1.0;
            return match branch {
                3 => {
                    let product = a.recip() * b.recip() * c.recip() * d.recip();
                    let sum = a.recip() + b.recip() + c.recip() + d.recip();
                    sign * self.cap * product * sum
                }
                2 => sign * (a.recip() * b.recip() * c.recip() * d.recip()),
                1 => 0.0,
                _ => {
                    let mut product = 1.0;
                    let mut s1 = 0.0;
                    for x in [a, b, c, d] {
                        let z = (self.floor - x).recip();
                        product *= z;
                        s1 += z;
                    }
                    self.floor * product * s1
                }
            };
        }
        let mut sorted = nodes;
        for position in 1..4 {
            let mut cursor = position;
            while cursor > 0
                && self.evals[sorted[cursor - 1]].total_cmp(&self.evals[sorted[cursor]])
                    == std::cmp::Ordering::Greater
            {
                sorted.swap(cursor - 1, cursor);
                cursor -= 1;
            }
        }
        (self.triple(0, sorted[1], sorted[2], sorted[3]) - self.triple(0, sorted[0], sorted[1], sorted[2]))
            / (self.evals[sorted[3]] - self.evals[sorted[0]])
    }
}

impl JeffreysHphiDriftBase {
    fn divided_differences(&self) -> &InverseDividedDifferences {
        self.divided_differences
            .get_or_init(|| InverseDividedDifferences::new(&self.evals, self.floor))
    }

    /// Apply the derivative of the omitted true-Hessian completion to a
    /// coefficient direction. `axes` contains `H[v,e_a]` and `moving_axes`
    /// its derivative under `pert_h`. This contracts the fifth likelihood
    /// derivative directly, without assembling a third coefficient tensor.
    pub fn completion_drift_action(
        &self,
        pert_h: &Array2<f64>,
        axes: &[Array2<f64>],
        moving_axes: &[Array2<f64>],
    ) -> Result<Array1<f64>, String> {
        if pert_h.dim() != (self.p, self.p) {
            return Err("Jeffreys completion drift information dimension mismatch".into());
        }
        let e = symmetric_basis_contraction(pert_h.view(), self.ambient_eigenbasis.view());
        let a = self.rotate_axis_rows(axes)?;
        let da = self.rotate_axis_rows(moving_axes)?;
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let dg = g_min * e[[self.idx_min, self.idx_min]]
            + g_max * e[[self.idx_max, self.idx_max]];
        let dfloor = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR * e[[self.idx_max, self.idx_max]]
        } else {
            0.0
        };
        let mut result = Array1::<f64>::zeros(self.p);
        for i in 0..self.m {
            let kernel = inverse_difference(&[self.evals[i]], self.floor, 0);
            let floor_motion = dfloor * inverse_difference(&[self.evals[i]], self.floor, 1);
            for axis in 0..self.p {
                result[axis] -= 0.5 * (
                    self.gate_weight * (kernel * da[[axis, i * self.m + i]]
                        + floor_motion * a[[axis, i * self.m + i]])
                    + dg * kernel * a[[axis, i * self.m + i]]);
            }
            for j in 0..self.m {
                let dk = inverse_difference(&[self.evals[i], self.evals[j]], self.floor, 0)
                    * e[[i, j]];
                for axis in 0..self.p {
                    result[axis] -= 0.5 * self.gate_weight * dk * a[[axis, i * self.m + j]];
                }
            }
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Jeffreys completion drift produced a nonfinite response".into());
        }
        Ok(result)
    }

    /// Differentiate the spectral matrix function in the fixed base frame.
    /// Divided differences include eigenvector motion without dividing by
    /// eigenvalue gaps, so repeated interior eigenvalues need no special case.
    pub(super) fn perturbation_derivative_from_axis_matrices(
        &self,
        pert_h: &Array2<f64>,
        pert_hdots: &[Array2<f64>],
    ) -> Result<Array2<f64>, String> {
        if pert_h.dim() != (self.p, self.p) {
            return Err("Jeffreys drift information dimension mismatch".into());
        }
        let e = symmetric_basis_contraction(pert_h.view(), self.ambient_eigenbasis.view());
        let da = self.rotate_axis_rows(pert_hdots)?;
        let mut dw = self.inverse_frechet_rows(&self.a_rows, &[&e], 0);
        if self.floor_in_relative_regime {
            let dfloor = REDUCED_INFO_RELATIVE_FLOOR * e[[self.idx_max, self.idx_max]];
            if dfloor != 0.0 {
                dw.scaled_add(dfloor, &self.inverse_frechet_rows(&self.a_rows, &[], 1));
            }
        }
        dw += &self.inverse_frechet_rows(&da, &[], 0);
        let mut result = (dw.dot(&self.a_rows.t()) + self.aw_rows.dot(&da.t()))
            * (-0.5 * self.gate_weight);
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let dg = g_min * e[[self.idx_min, self.idx_min]]
            + g_max * e[[self.idx_max, self.idx_max]];
        if dg != 0.0 {
            result.scaled_add(-0.5 * dg, &self.aw_rows.dot(&self.a_rows.t()));
        }
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Jeffreys drift produced nonfinite curvature".into());
        }
        Ok(result)
    }

    pub(super) fn rotate_axis_rows(&self, axes: &[Array2<f64>]) -> Result<Array2<f64>, String> {
        if axes.len() != self.p || axes.iter().any(|a| a.dim() != (self.p, self.p)) {
            return Err("Jeffreys mixed drift requires one full information derivative per coefficient axis".into());
        }
        let mut rows = Array2::zeros((self.p, self.m * self.m));
        for (a, matrix) in axes.iter().enumerate() {
            let rotated =
                symmetric_basis_contraction(matrix.view(), self.ambient_eigenbasis.view());
            for i in 0..self.m {
                for j in 0..self.m {
                    rows[[a, i * self.m + j]] = rotated[[i, j]];
                }
            }
        }
        Ok(rows)
    }

    /// Apply Df, D²f[E,.], or D³f[E,F,.] to every axis matrix.
    /// Storage stays O(p m²); the fourth-order Loewner tensor is never stored.
    fn inverse_frechet_rows(
        &self,
        rows: &Array2<f64>,
        directions: &[&Array2<f64>],
        floor_order: usize,
    ) -> Array2<f64> {
        let m = self.m;
        let squared = m * m;
        let divided = self.divided_differences();
        let mut out = Array2::zeros(rows.raw_dim());
        if directions.len() == 2 {
            // D³f[E,F,A] is linear in A. Assemble that linear map for one
            // output row at a time, sharing its spectral coefficients across
            // every coefficient axis, then contract with BLAS-3. Scattering
            // all p axes inside the four spectral loops costs O(p m^4)
            // strided scalar updates. Here those operations form one dense
            // contraction and the spectral assembly costs only O(m^4).
            // A full m²-by-m² Loewner map is never allocated: the scratch
            // occupies m³ entries and is reused for each output row.
            let e = directions[0].as_standard_layout();
            let f = directions[1].as_standard_layout();
            let e = e.as_slice().expect("standard-layout spectral direction");
            let f = f.as_slice().expect("standard-layout spectral direction");
            let mut weights = Array2::<f64>::zeros((m, squared));
            for i in 0..m {
                weights.fill(0.0);
                let w = weights
                    .as_slice_mut()
                    .expect("freshly allocated weights are contiguous");
                for j in 0..m {
                    for k in 0..m {
                        for l in 0..m {
                            let c = if floor_order == 0 {
                                divided.quadruple([i, k, l, j])
                            } else {
                                inverse_difference(
                                    &[self.evals[i], self.evals[k], self.evals[l], self.evals[j]],
                                    self.floor,
                                    floor_order,
                                )
                            };
                            let (ik, kl, lj) = (i * m + k, k * m + l, l * m + j);
                            w[j * squared + l * m + j] += c * (e[ik] * f[kl] + f[ik] * e[kl]);
                            w[j * squared + kl] += c * (e[ik] * f[lj] + f[ik] * e[lj]);
                            w[j * squared + ik] += c * (e[kl] * f[lj] + f[kl] * e[lj]);
                        }
                    }
                }
                out.slice_mut(ndarray::s![.., i * m..(i + 1) * m])
                    .assign(&rows.dot(&weights.t()));
            }
            return out;
        }
        let input = rows.as_standard_layout();
        let input = input.as_slice().expect("standard-layout axis rows");
        let output = out.as_slice_mut().expect("freshly allocated rows are contiguous");
        if directions.is_empty() {
            let table = &divided.pairs[floor_order];
            for (target, source) in output.chunks_exact_mut(squared).zip(input.chunks_exact(squared)) {
                for ((value, &coefficient), &entry) in
                    target.iter_mut().zip(table.iter()).zip(source.iter())
                {
                    *value = coefficient * entry;
                }
            }
            return out;
        }
        let e = directions[0].as_standard_layout();
        let e = e.as_slice().expect("standard-layout spectral direction");
        let table = &divided.triples[floor_order];
        for (target, source) in output.chunks_exact_mut(squared).zip(input.chunks_exact(squared)) {
            for i in 0..m {
                for j in 0..m {
                    let mut value = 0.0;
                    for k in 0..m {
                        value += table[(i * m + k) * m + j]
                            * (e[i * m + k] * source[k * m + j] + source[i * m + k] * e[k * m + j]);
                    }
                    target[i * m + j] = value;
                }
            }
        }
        out
    }

    /// Exact mixed derivative `D² H_Φ[u,v]` on the current spectral stratum.
    /// The inputs are H_u, H_v, H_uv and their coefficient-axis derivatives.
    /// No mode second response is included; the caller adds `D H_Φ[β_uv]`.
    pub fn mixed_perturbation_derivative_batched_axes(
        &self,
        pert_u: &Array2<f64>,
        pert_v: &Array2<f64>,
        pert_uv: &Array2<f64>,
        axes_u: &[Array2<f64>],
        axes_v: &[Array2<f64>],
        axes_uv: &[Array2<f64>],
    ) -> Result<Array2<f64>, String> {
        if [pert_u, pert_v, pert_uv]
            .iter()
            .any(|h| h.dim() != (self.p, self.p))
        {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        let u = self.direction_frame(pert_u, axes_u)?;
        let v = self.direction_frame(pert_v, axes_v)?;
        self.mixed_perturbation_derivative_from_frames(&u, &v, pert_uv, axes_uv)
    }

    /// Everything the mixed derivative reads from ONE direction: its rotated
    /// information derivative, its rotated coefficient-axis derivatives, their
    /// first Fréchet rows, and the gate and floor channels they drive. An outer
    /// Hessian over `k` coordinates requests `k(k+1)/2` pairs drawn from `k`
    /// mode responses, so a caller batching pairs builds one frame per distinct
    /// direction and closes every pair from two frames.
    pub fn direction_frame(
        &self,
        pert: &Array2<f64>,
        axes: &[Array2<f64>],
    ) -> Result<JeffreysDirectionFrame, String> {
        if pert.dim() != (self.p, self.p) {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let e = symmetric_basis_contraction(pert.view(), self.ambient_eigenbasis.view());
        let rows = self.rotate_axis_rows(axes)?;
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let min = e[[self.idx_min, self.idx_min]];
        let max = e[[self.idx_max, self.idx_max]];
        let floor_scale = if self.moving_relative_floor() {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let floor_motion = floor_scale * max;
        let gate = g_min * min + g_max * max;
        let mut first = self.first_frechet_rows(&self.a_rows, &e, floor_motion);
        first += &self.inverse_frechet_rows(&rows, &[], 0);
        let raw = (first.dot(&self.a_rows.t()) + self.aw_rows.dot(&rows.t())) * -0.5;
        Ok(JeffreysDirectionFrame {
            e,
            rows,
            first,
            raw,
            min,
            max,
            floor_motion,
            gate,
        })
    }

    /// `D² H_Φ[u,v]` closed from the two directions' frames. Only the pair's own
    /// objects — `H_uv`, its coefficient-axis derivatives and the second-order
    /// spectral rows — are formed here.
    pub fn mixed_perturbation_derivative_from_frames(
        &self,
        u: &JeffreysDirectionFrame,
        v: &JeffreysDirectionFrame,
        pert_uv: &Array2<f64>,
        axes_uv: &[Array2<f64>],
    ) -> Result<Array2<f64>, String> {
        if pert_uv.dim() != (self.p, self.p) {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let e = &u.e;
        let f = &v.e;
        let ef = symmetric_basis_contraction(pert_uv.view(), self.ambient_eigenbasis.view());
        let auv = self.rotate_axis_rows(axes_uv)?;
        let a = &self.a_rows;
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let (g_mm, g_mx, g_xx) =
            conditioning_gate_weight_hess(self.evals[self.idx_min], self.evals[self.idx_max]);
        let eigen_mixed = |idx: usize, needed: bool| -> Result<f64, String> {
            if !needed {
                return Ok(0.0);
            }
            let mut result = ef[[idx, idx]];
            for j in 0..self.m {
                if j == idx {
                    continue;
                }
                let gap = self.evals[idx] - self.evals[j];
                if gap.abs() <= f64::EPSILON * self.evals[idx].abs().max(self.evals[j].abs()) * 16.0
                {
                    return Err(
                        "Jeffreys mixed drift is undefined at a repeated active extreme eigenvalue"
                            .into(),
                    );
                }
                result += (e[[idx, j]] * f[[j, idx]] + f[[idx, j]] * e[[j, idx]]) / gap;
            }
            Ok(result)
        };
        let min_uv = eigen_mixed(self.idx_min, g_min != 0.0)?;
        let moving_floor = self.moving_relative_floor();
        let max_uv = eigen_mixed(self.idx_max, g_max != 0.0 || moving_floor)?;
        let floor_scale = if moving_floor {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let floor_uv = floor_scale * max_uv;
        let guv = g_min * min_uv
            + g_max * max_uv
            + g_mm * u.min * v.min
            + g_mx * (u.min * v.max + u.max * v.min)
            + g_xx * u.max * v.max;
        let mut wuv = self.inverse_frechet_rows(a, &[e, f], 0);
        wuv += &self.inverse_frechet_rows(a, &[&ef], 0);
        if v.floor_motion != 0.0 {
            wuv.scaled_add(v.floor_motion, &self.inverse_frechet_rows(a, &[e], 1));
        }
        if u.floor_motion != 0.0 {
            wuv.scaled_add(u.floor_motion, &self.inverse_frechet_rows(a, &[f], 1));
        }
        if u.floor_motion * v.floor_motion != 0.0 {
            wuv.scaled_add(
                u.floor_motion * v.floor_motion,
                &self.inverse_frechet_rows(a, &[], 2),
            );
        }
        if floor_uv != 0.0 {
            wuv.scaled_add(floor_uv, &self.inverse_frechet_rows(a, &[], 1));
        }
        wuv += &self.first_frechet_rows(&v.rows, e, u.floor_motion);
        wuv += &self.first_frechet_rows(&u.rows, f, v.floor_motion);
        wuv += &self.inverse_frechet_rows(&auv, &[], 0);
        let w = &self.aw_rows;
        let raw = w.dot(&a.t()) * -0.5;
        let mut result = (wuv.dot(&a.t())
            + u.first.dot(&v.rows.t())
            + v.first.dot(&u.rows.t())
            + w.dot(&auv.t()))
            * (-0.5 * self.gate_weight);
        result.scaled_add(u.gate, &v.raw);
        result.scaled_add(v.gate, &u.raw);
        result.scaled_add(guv, &raw);
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|value| !value.is_finite()) {
            return Err("Jeffreys mixed drift produced nonfinite curvature".into());
        }
        Ok(result)
    }

    /// `Df[A]` plus the relative-floor motion `dfloor · ∂_floor f[A]` along one
    /// direction, for every axis row.
    fn first_frechet_rows(
        &self,
        rows: &Array2<f64>,
        direction: &Array2<f64>,
        dfloor: f64,
    ) -> Array2<f64> {
        let mut out = self.inverse_frechet_rows(rows, &[direction], 0);
        if dfloor != 0.0 {
            out.scaled_add(dfloor, &self.inverse_frechet_rows(rows, &[], 1));
        }
        out
    }

    /// The relative spectral floor moves with the information along a direction.
    fn moving_relative_floor(&self) -> bool {
        self.floor_in_relative_regime
            && (self.evals.iter().any(|&x| x < self.floor)
                || self.floor > CONDITIONING_GATE_ABSOLUTE_CLEAR)
    }

    fn refuse_inverse_kernel_branch_boundary(&self) -> Result<(), String> {
        let cap = self.floor.max(CONDITIONING_GATE_ABSOLUTE_CLEAR);
        for &value in &self.evals {
            for knot in [0.0, self.floor, cap] {
                let resolution = 16.0 * f64::EPSILON * value.abs().max(knot.abs());
                if (value - knot).abs() <= resolution {
                    return Err(
                        "Jeffreys mixed drift is undefined at an inverse-kernel branch boundary"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// The per-direction half of `D² H_Φ[u,v]`; see
/// [`JeffreysHphiDriftBase::direction_frame`].
pub struct JeffreysDirectionFrame {
    /// Rotated information derivative `Uᵀ H[u] U`.
    e: Array2<f64>,
    /// Rotated coefficient-axis derivatives `vec(Uᵀ H²[u,e_a] U)`.
    rows: Array2<f64>,
    /// First Fréchet rows of the capped inverse along `u`, floor motion included.
    first: Array2<f64>,
    /// `D H_Φ_raw[u]` before the gate weight.
    raw: Array2<f64>,
    min: f64,
    max: f64,
    floor_motion: f64,
    gate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// #979: the tabulated divided differences, including the four-node values
    /// closed from sorted triples, are bit-identical to `inverse_difference` on the
    /// same nodes, across all four pieces of the capped inverse and a repeated
    /// eigenvalue.
    #[test]
    fn tabulated_divided_differences_match_inverse_difference_bitwise_979() {
        let floor = 1e-3;
        let evals = array![-0.3, 2e-4, 0.5, 0.5, 7.0, 40.0];
        let table = InverseDividedDifferences::new(&evals, floor);
        let m = evals.len();
        let same = |left: f64, right: f64| left.to_bits() == right.to_bits();
        for order in 0..3 {
            for i in 0..m {
                for j in 0..m {
                    let direct = inverse_difference(&[evals[i], evals[j]], floor, order);
                    assert!(same(table.pairs[order][i * m + j], direct), "pair ({i},{j}) order {order}");
                    for k in 0..m {
                        let direct = inverse_difference(&[evals[i], evals[k], evals[j]], floor, order);
                        assert!(same(table.triple(order, i, k, j), direct), "triple ({i},{k},{j}) order {order}");
                    }
                }
            }
        }
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for d in 0..m {
                        let direct =
                            inverse_difference(&[evals[a], evals[b], evals[c], evals[d]], floor, 0);
                        let tabulated = table.quadruple([a, b, c, d]);
                        assert!(
                            same(tabulated, direct),
                            "quadruple ({a},{b},{c},{d}): tabulated {tabulated} vs direct {direct}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_jeffreys_drift_matches_first_drift_difference_979() {
        // Gate transition, repeated interior spectrum, moving relative floor,
        // and the signed continuation all exercise different spectral channels.
        for diagonal in [
            [3.0, 7.0, 30.0],
            [0.4, 2.0, 2.0],
            [1e-4, 4e8, 5e8],
            [-0.2, 0.4, 3.0],
        ] {
            let h = Array2::from_diag(&Array1::from_vec(diagonal.to_vec()));
            let z = Array2::eye(3);
            let e = array![[0.2, 0.03, -0.04], [0.03, -0.1, 0.02], [-0.04, 0.02, 0.15]];
            let f = array![[0.1, -0.02, 0.01], [-0.02, 0.2, 0.03], [0.01, 0.03, -0.1]];
            let ef = &e * 0.13 + &f * 0.07;
            let axes = vec![e.clone(), f.clone(), &e + &f];
            let au: Vec<_> = axes.iter().map(|a| a * 0.11).collect();
            let av: Vec<_> = axes.iter().map(|a| a * -0.08).collect();
            let auv: Vec<_> = axes.iter().map(|a| a * 0.03).collect();
            let base = JeffreysHphiDriftBase::prepare_with_axes(h.view(), z.view(), axes.clone())
                .unwrap()
                .unwrap();
            let completion_actual = base.completion_drift_action(&e, &axes, &au).unwrap();
            let score = base.explicit_score_pair(&e, &f, &ef, &au, &av, &auv).unwrap();
            for axis in 0..3 {
                let at = |t: f64| joint_jeffreys_phi_explicit_param_second_derivative(
                    (&h + &axes[axis]*t).view(), z.view(),
                    &(&e + &au[axis]*t), &(&f + &av[axis]*t), &(&ef + &auv[axis]*t)).unwrap();
                let step = 1e-5;
                let fd = (at(step)-at(-step))/(2.0*step);
                let error = (score[axis]-fd).abs()/(1.0+score[axis].abs().max(fd.abs()));
                assert!(error < 2e-5, "scalar third spectrum={diagonal:?}, axis={axis}, analytic={}, fd={fd}, error={error}", score[axis]);
            }
            let completion_at = |t: f64| {
                let ht = &h + &e * t;
                let at: Vec<_> = axes.iter().zip(&au).map(|(a, d)| a + &(d * t)).collect();
                let point = JeffreysHphiDriftBase::prepare_with_axes(ht.view(), z.view(), at.clone())
                    .unwrap().unwrap();
                let rows = point.rotate_axis_rows(&at).unwrap();
                Array1::from_shape_fn(3, |a| {
                    -0.5 * point.gate_weight * (0..3).map(|i| {
                        floored_inverse(point.evals[i], point.floor) * rows[[a, i * 3 + i]]
                    }).sum::<f64>()
                })
            };
            let completion_step = 1e-5;
            let completion_fd = (completion_at(completion_step) - completion_at(-completion_step)) / (2.0 * completion_step);
            for a in 0..3 {
                let scale = 1.0 + completion_actual[a].abs().max(completion_fd[a].abs());
                assert!((completion_actual[a] - completion_fd[a]).abs() < 2e-6 * scale,
                    "completion spectrum={diagonal:?}, axis={a}, analytic={}, fd={}", completion_actual[a], completion_fd[a]);
            }
            let actual = base
                .mixed_perturbation_derivative_batched_axes(
                    &e,
                    &f,
                    &ef,
                    &au,
                    &av,
                    &auv,
                )
                .unwrap();
            let first_at = |t: f64| {
                let ht = &h + &e * t;
                let at = axes.iter().zip(&au).map(|(a, d)| a + &(d * t)).collect();
                let avt = av.iter().zip(&auv).map(|(a, d)| a + &(d * t)).collect();
                JeffreysHphiDriftBase::prepare_with_axes(ht.view(), z.view(), at)
                    .unwrap()
                    .unwrap()
                    .perturbation_derivative_batched_axes(&(&f + &ef * t), Some(avt))
                    .unwrap()
            };
            let step = 1e-5;
            let expected = (first_at(step) - first_at(-step)) / (2.0 * step);
            let scale = expected
                .iter()
                .chain(actual.iter())
                .fold(1.0_f64, |m, x| m.max(x.abs()));
            let error = (&actual - &expected)
                .iter()
                .fold(0.0_f64, |m, x| m.max(x.abs()));
            assert!(
                error < 2e-4 * scale,
                "spectrum={diagonal:?}, relative error={} actual={actual:?} expected={expected:?}",
                error / scale
            );
            let swapped = base
                .mixed_perturbation_derivative_batched_axes(&f, &e, &ef, &av, &au, &auv)
                .unwrap();
            assert!((&actual - &swapped).iter().all(|x| x.abs() < 1e-12 * scale));
        }
    }
}
