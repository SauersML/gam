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
        self.completion_drift_from_rows(&e, &a, &da)
    }

    /// [`Self::completion_drift_action`] on already-rotated objects: `e = Uᵀ H[u] U`,
    /// and `a`, `da` the rotated rows of `{H[v, e_a]}` and `{H[u, v, e_a]}`.
    pub(super) fn completion_drift_from_rows(
        &self,
        e: &Array2<f64>,
        a: &Array2<f64>,
        da: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
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

    /// `D_w` of [`Self::completion_drift_action`] along a second coefficient direction,
    /// with `u` and `v` held fixed (gam#2894): the frozen-policy half of
    /// `D² completion[u, w]·v`. The frozen completion is `−½·g·⟨K, H²[v, e_a]⟩`, so
    ///
    /// ```text
    /// −½[ g_uw⟨K,A⟩ + g_u⟨K_w,A⟩ + g_w⟨K_u,A⟩ + g⟨K_uw,A⟩
    ///     + g_u⟨K,T_vw⟩ + g_w⟨K,T_uv⟩ + g⟨K_u,T_vw⟩ + g⟨K_w,T_uv⟩ + g⟨K,Q⟩ ]
    /// ```
    ///
    /// with `A = H²[v,e_a]`, `T_uv = H³[u,v,e_a]`, `T_vw = H³[v,w,e_a]`,
    /// `Q = H⁴[u,v,w,e_a]` in the base eigenbasis, `⟨K,X⟩ = Σ_i f(λ_i)·X_ii`, and `K_u`,
    /// `K_uw` the first and second Fréchet derivatives of the capped inverse, floor
    /// motion included. The gate/floor motion drift is not part of this half.
    pub fn completion_second_drift_frozen(
        &self,
        pert_u: &Array2<f64>,
        pert_w: &Array2<f64>,
        pert_uw: &Array2<f64>,
        axes_v: &[Array2<f64>],
        moving_uv: &[Array2<f64>],
        moving_vw: &[Array2<f64>],
        fourth_uvw: &[Array2<f64>],
    ) -> Result<Array1<f64>, String> {
        if [pert_u, pert_w, pert_uw].iter().any(|h| h.dim() != (self.p, self.p)) {
            return Err("Jeffreys completion second drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let basis = self.ambient_eigenbasis.view();
        let e_u = symmetric_basis_contraction(pert_u.view(), basis);
        let e_w = symmetric_basis_contraction(pert_w.view(), basis);
        let e_uw = symmetric_basis_contraction(pert_uw.view(), basis);
        let a = self.rotate_axis_rows(axes_v)?;
        let t_uv = self.rotate_axis_rows(moving_uv)?;
        let t_vw = self.rotate_axis_rows(moving_vw)?;
        let q = self.rotate_axis_rows(fourth_uvw)?;
        let (m, p) = (self.m, self.p);
        let FrozenSecondDriftWeights {
            gate_u: g_u,
            gate_w: g_w,
            gate_uw: g_uw,
            kernel,
            weight_u,
            weight_w,
            weight_uw,
        } = self.frozen_second_drift_weights(&e_u, &e_w, &e_uw);
        let g = self.gate_weight;
        let contract_diagonal = |rows: &Array2<f64>, axis: usize| {
            (0..m).map(|i| kernel[i] * rows[[axis, i * m + i]]).sum::<f64>()
        };
        let contract_full = |weight: &Array2<f64>, rows: &Array2<f64>, axis: usize| {
            let mut sum = 0.0_f64;
            for i in 0..m {
                for j in 0..m {
                    sum += weight[[i, j]] * rows[[axis, i * m + j]];
                }
            }
            sum
        };
        let mut result = Array1::<f64>::zeros(p);
        for axis in 0..p {
            let k_a = contract_diagonal(&a, axis);
            let k_t_uv = contract_diagonal(&t_uv, axis);
            let k_t_vw = contract_diagonal(&t_vw, axis);
            let k_q = contract_diagonal(&q, axis);
            let ku_a = contract_full(&weight_u, &a, axis);
            let kw_a = contract_full(&weight_w, &a, axis);
            let kuw_a = contract_full(&weight_uw, &a, axis);
            let ku_t_vw = contract_full(&weight_u, &t_vw, axis);
            let kw_t_uv = contract_full(&weight_w, &t_uv, axis);
            result[axis] = -0.5
                * (g_uw * k_a
                    + g_u * kw_a
                    + g_w * ku_a
                    + g * kuw_a
                    + g_u * k_t_vw
                    + g_w * k_t_uv
                    + g * ku_t_vw
                    + g * kw_t_uv
                    + g * k_q);
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Jeffreys completion second drift produced a nonfinite response".into());
        }
        Ok(result)
    }

    /// The gate and floor channels along `u`, `w` and `(u, w)`, and the first and second
    /// Fréchet weights of the capped inverse in the base eigenbasis: everything the frozen
    /// second completion drift reads from two directions (gam#2894).
    fn frozen_second_drift_weights(
        &self,
        e_u: &Array2<f64>,
        e_w: &Array2<f64>,
        e_uw: &Array2<f64>,
    ) -> FrozenSecondDriftWeights {
        let m = self.m;
        let (imin, imax) = (self.idx_min, self.idx_max);
        let spectral_scale = self.evals.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tie_tolerance = 64.0 * f64::EPSILON * spectral_scale;
        let second_extreme = |e: usize| {
            e_uw[[e, e]] + simple_eigenvalue_second_form(&self.evals, tie_tolerance, e, e_u, e_w)
        };
        let (lmin_u, lmax_u) = (e_u[[imin, imin]], e_u[[imax, imax]]);
        let (lmin_w, lmax_w) = (e_w[[imin, imin]], e_w[[imax, imax]]);
        let (lmin_uw, lmax_uw) = (second_extreme(imin), second_extreme(imax));
        let (g1, g2) = conditioning_gate_weight_grad(self.evals[imin], self.evals[imax]);
        let (g11, g12, g22) = conditioning_gate_weight_hess(self.evals[imin], self.evals[imax]);
        let gate_u = g1 * lmin_u + g2 * lmax_u;
        let gate_w = g1 * lmin_w + g2 * lmax_w;
        let gate_uw = g11 * lmin_u * lmin_w
            + g12 * (lmin_u * lmax_w + lmax_u * lmin_w)
            + g22 * lmax_u * lmax_w
            + g1 * lmin_uw
            + g2 * lmax_uw;
        let rate = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let (floor_u, floor_w, floor_uw) = (rate * lmax_u, rate * lmax_w, rate * lmax_uw);
        let divided = self.divided_differences();
        let kernel: Vec<f64> = (0..m).map(|i| inverse_difference(&[self.evals[i]], self.floor, 0)).collect();
        let kernel_floor: Vec<f64> =
            (0..m).map(|i| inverse_difference(&[self.evals[i]], self.floor, 1)).collect();
        let kernel_floor_floor: Vec<f64> =
            (0..m).map(|i| inverse_difference(&[self.evals[i]], self.floor, 2)).collect();
        // `D²f[E_u, E_w]_ij + Df[E_uw]_ij + floor channels`: the spectral weights the second
        // Fréchet contraction reads, formed once for every axis.
        let mut weight_uw = Array2::<f64>::zeros((m, m));
        let mut weight_u = Array2::<f64>::zeros((m, m));
        let mut weight_w = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let pair = divided.pairs[0][i * m + j];
                let pair_floor = divided.pairs[1][i * m + j];
                let mut second = pair * e_uw[[i, j]]
                    + pair_floor * (floor_u * e_w[[i, j]] + floor_w * e_u[[i, j]]);
                for k in 0..m {
                    second += divided.triple(0, i, k, j)
                        * (e_u[[i, k]] * e_w[[k, j]] + e_w[[i, k]] * e_u[[k, j]]);
                }
                weight_uw[[i, j]] = second;
                weight_u[[i, j]] = pair * e_u[[i, j]];
                weight_w[[i, j]] = pair * e_w[[i, j]];
            }
            weight_uw[[i, i]] += floor_u * floor_w * kernel_floor_floor[i] + floor_uw * kernel_floor[i];
            weight_u[[i, i]] += floor_u * kernel_floor[i];
            weight_w[[i, i]] += floor_w * kernel_floor[i];
        }
        FrozenSecondDriftWeights {
            gate_u,
            gate_w,
            gate_uw,
            kernel,
            weight_u,
            weight_w,
            weight_uw,
        }
    }

    /// The frozen-policy half of `D² completion[u, w]` as a matrix (gam#2894):
    ///
    /// ```text
    /// CTH(W₀₀) + CTH_w(W₀ᵤ) + CTH_u(W₁w) + CTH_uw(W₁₁),
    /// W₀₀ = −½(G_uw K + G_u K_w + G_w K_u + G K_uw),   W₀ᵤ = −½(G_u K + G K_u),
    /// W₁w = −½(G_w K + G K_w),                          W₁₁ = −½ G K,
    /// ```
    ///
    /// `CTH_x(W)_ab = ⟨W, D_x H''[e_a, e_b]⟩`, contractions supplied by the caller. Every
    /// column is [`Self::completion_second_drift_frozen`] along that axis.
    pub fn frozen_completion_second_drift_matrix(
        &self,
        pert_u: &Array2<f64>,
        pert_w: &Array2<f64>,
        pert_uw: &Array2<f64>,
        contracted: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_u: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_w: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_uw: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<Array2<f64>, String> {
        if [pert_u, pert_w, pert_uw].iter().any(|h| h.dim() != (self.p, self.p)) {
            return Err("Jeffreys frozen completion second drift matrix dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let basis = &self.ambient_eigenbasis;
        let e_u = symmetric_basis_contraction(pert_u.view(), basis.view());
        let e_w = symmetric_basis_contraction(pert_w.view(), basis.view());
        let e_uw = symmetric_basis_contraction(pert_uw.view(), basis.view());
        let FrozenSecondDriftWeights {
            gate_u,
            gate_w,
            gate_uw,
            kernel,
            weight_u,
            weight_w,
            weight_uw,
        } = self.frozen_second_drift_weights(&e_u, &e_w, &e_uw);
        let g = self.gate_weight;
        let ambient = |reduced: &Array2<f64>| basis.dot(reduced).dot(&basis.t());
        let kernel = ambient(&Array2::from_diag(&Array1::from_vec(kernel)));
        let kernel_u = ambient(&weight_u);
        let kernel_w = ambient(&weight_w);
        let kernel_uw = ambient(&weight_uw);
        let now = (&kernel * gate_uw + &kernel_w * gate_u + &kernel_u * gate_w + &kernel_uw * g) * -0.5;
        let along_w = (&kernel * gate_u + &kernel_u * g) * -0.5;
        let along_u = (&kernel * gate_w + &kernel_w * g) * -0.5;
        let along_uw = &kernel * (-0.5 * g);
        let mut result = contracted(&now)?
            + contracted_along_w(&along_w)?
            + contracted_along_u(&along_u)?
            + contracted_along_uw(&along_uw)?;
        symmetrize_contiguous(&mut result);
        if result.iter().any(|value| !value.is_finite()) {
            return Err("Jeffreys frozen completion second drift matrix produced nonfinite curvature".into());
        }
        Ok(result)
    }

    /// The first β-drift of the complete second-order completion as a matrix,
    /// `D_u completion` (gam#2894). With `completion = −½·G·CTH(K) − CTH(E) − R`, where
    /// `CTH(W)_ab = ⟨W, H''[e_a, e_b]⟩`, `K` the capped inverse on the Jeffreys span and
    /// `(E, R)` the gate/floor motion of [`JointJeffreysHessianMotion`],
    ///
    /// ```text
    /// D_u completion = CTH(W₀) + CTH_u(W₁) − D_u R,
    /// W₀ = −½·G_u·K − ½·G·K_u − D_u E,    W₁ = −½·G·K − E,
    /// ```
    ///
    /// `CTH_u(W)_ab = ⟨W, H'''[u, e_a, e_b]⟩`. The two contractions are the caller's (a
    /// family's contracted-trace hooks); everything spectral is formed here from
    /// `H[u]` and the rotated `{H''[u, e_a]}`. Every column is the motion-completed
    /// [`Self::completion_drift_action_from_rotated`] along that axis.
    pub fn completion_drift_matrix(
        &self,
        pert_u: &Array2<f64>,
        second_u: &JeffreysRotatedAxes,
        contracted: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_u: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<Array2<f64>, String> {
        let (p, m) = (self.p, self.m);
        if pert_u.dim() != (p, p) || second_u.rows.dim() != (p, m * m) {
            return Err("Jeffreys completion drift matrix dimension mismatch".into());
        }
        let basis = &self.ambient_eigenbasis;
        let e_u = symmetric_basis_contraction(pert_u.view(), basis.view());
        let b_u = &second_u.rows;
        let a_rows = &self.a_rows;
        let (imin, imax) = (self.idx_min, self.idx_max);
        let evals = &self.evals;
        let floor = self.floor;
        let gate = self.gate_weight;
        let spectral_scale = evals.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tie_tolerance = 64.0 * f64::EPSILON * spectral_scale;
        let rate = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let (lambda_min, lambda_max) = (evals[imin], evals[imax]);
        let (g1, g2) = conditioning_gate_weight_grad(lambda_min, lambda_max);
        let divided = self.divided_differences();
        let inverse: Vec<f64> = (0..m).map(|i| floored_inverse(evals[i], floor)).collect();
        let inverse_floor: Vec<f64> =
            (0..m).map(|i| floored_inverse_floor_sensitivity(evals[i], floor)).collect();
        let lambda_min_u = e_u[[imin, imin]];
        let lambda_max_u = e_u[[imax, imax]];
        let gate_u = g1 * lambda_min_u + g2 * lambda_max_u;
        let floor_u = rate * lambda_max_u;
        let ambient = |reduced: &Array2<f64>| basis.dot(reduced).dot(&basis.t());
        let kernel = ambient(&Array2::from_diag(&Array1::from_vec(inverse.clone())));
        let mut kernel_u_reduced = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                kernel_u_reduced[[i, j]] = divided.pairs[0][i * m + j] * e_u[[i, j]];
            }
            kernel_u_reduced[[i, i]] += floor_u * inverse_floor[i];
        }
        let kernel_u = ambient(&kernel_u_reduced);
        let mut weight_now = &kernel * (-0.5 * gate_u) + &kernel_u * (-0.5 * gate);
        let mut weight_along_u = &kernel * (-0.5 * gate);
        let motion = self.hessian_motion_active();
        let mut remainder_drift = Array2::<f64>::zeros((p, p));
        if motion {
            let (g11, g12, g22) = conditioning_gate_weight_hess(lambda_min, lambda_max);
            let (g111, g112, g122, g222) = conditioning_gate_weight_third(lambda_min, lambda_max);
            let mut ungated = 0.0_f64;
            let (mut s_f, mut s_ff, mut s_fff) = (0.0_f64, 0.0_f64, 0.0_f64);
            let mut inverse_lambda_floor = vec![0.0_f64; m];
            let mut inverse_floor_floor = vec![0.0_f64; m];
            for i in 0..m {
                let lambda = evals[i];
                ungated += jeffreys_antiderivative(lambda, floor);
                s_f += jeffreys_antiderivative_floor_sensitivity(lambda, floor);
                s_ff += jeffreys_antiderivative_floor_second_sensitivity(lambda, floor);
                s_fff += jeffreys_antiderivative_floor_third_sensitivity(lambda, floor);
                inverse_lambda_floor[i] = floored_inverse_lambda_floor_sensitivity(lambda, floor);
                inverse_floor_floor[i] = floored_inverse_floor_second_sensitivity(lambda, floor);
            }
            ungated *= 0.5;
            let value_u = 0.5 * (0..m).map(|i| inverse[i] * e_u[[i, i]]).sum::<f64>()
                + 0.5 * s_f * rate * lambda_max_u;
            let floor_trace_u = (0..m).map(|i| inverse_floor[i] * e_u[[i, i]]).sum::<f64>();
            let s_f_u = floor_trace_u + s_ff * rate * lambda_max_u;
            let s_ff_u = s_fff * rate * lambda_max_u
                + (0..m).map(|i| inverse_floor_floor[i] * e_u[[i, i]]).sum::<f64>();
            // Extreme weights of `E` and their drift.
            let omega_min = ungated * g1;
            let omega_max = ungated * g2 + 0.5 * gate * s_f * rate;
            let omega_min_u = value_u * g1 + ungated * (g11 * lambda_min_u + g12 * lambda_max_u);
            let omega_max_u = value_u * g2
                + ungated * (g12 * lambda_min_u + g22 * lambda_max_u)
                + 0.5 * rate * (gate_u * s_f + gate * s_f_u);
            let gap = |e: usize, j: usize| simple_eigenvalue_gap_inverse(evals, tie_tolerance, e, j);
            let eigenvector_drift = |e: usize| {
                let reduced = Array1::from_shape_fn(m, |j| e_u[[j, e]] * gap(e, j));
                basis.dot(&reduced)
            };
            let outer = |x: &Array1<f64>, y: &Array1<f64>| {
                Array2::from_shape_fn((x.len(), y.len()), |(i, j)| x[i] * y[j])
            };
            let mut extreme_weight = Array2::<f64>::zeros((p, p));
            let mut extreme_weight_u = Array2::<f64>::zeros((p, p));
            for (e, omega, omega_u) in [(imin, omega_min, omega_min_u), (imax, omega_max, omega_max_u)] {
                let z = basis.column(e).to_owned();
                let z_u = eigenvector_drift(e);
                extreme_weight.scaled_add(omega, &outer(&z, &z));
                extreme_weight_u.scaled_add(omega_u, &outer(&z, &z));
                extreme_weight_u.scaled_add(omega, &(outer(&z_u, &z) + outer(&z, &z_u)));
            }
            weight_now -= &extreme_weight_u;
            weight_along_u -= &extreme_weight;
            // Axis objects of `R` and their drift along u.
            let row_entry = |rows: &Array2<f64>, a: usize, i: usize, j: usize| rows[[a, i * m + j]];
            let q_min = Array1::from_shape_fn(p, |a| row_entry(a_rows, a, imin, imin));
            let q_max = Array1::from_shape_fn(p, |a| row_entry(a_rows, a, imax, imax));
            let floor_trace =
                Array1::from_shape_fn(p, |a| (0..m).map(|i| inverse_floor[i] * row_entry(a_rows, a, i, i)).sum::<f64>());
            let grad_u = Array1::from_shape_fn(p, |a| {
                0.5 * (0..m).map(|i| inverse[i] * row_entry(a_rows, a, i, i)).sum::<f64>()
                    + 0.5 * s_f * rate * q_max[a]
            });
            let grad_g = &q_min * g1 + &q_max * g2;
            // `λ_i,au = B̃_au[i,i] + D²λ_i[P̃_a, e_u]` for every eigenvalue.
            let lambda_au = |i: usize, a: usize| {
                let mut sum = row_entry(b_u, a, i, i);
                for j in 0..m {
                    let inv = gap(i, j);
                    if inv != 0.0 {
                        sum += 2.0 * row_entry(a_rows, a, i, j) * e_u[[i, j]] * inv;
                    }
                }
                sum
            };
            let lambda_min_au = Array1::from_shape_fn(p, |a| lambda_au(imin, a));
            let lambda_max_au = Array1::from_shape_fn(p, |a| lambda_au(imax, a));
            let divided_u = self.aw_rows.dot(&Array1::from_iter(e_u.iter().copied()));
            let value_au = Array1::from_shape_fn(p, |a| {
                let mut kernel_trace = 0.0_f64;
                let mut floor_cross = 0.0_f64;
                for i in 0..m {
                    kernel_trace += inverse[i] * row_entry(b_u, a, i, i);
                    floor_cross += inverse_floor[i]
                        * (row_entry(a_rows, a, i, i) * lambda_max_u + e_u[[i, i]] * q_max[a]);
                }
                0.5 * divided_u[a]
                    + 0.5 * kernel_trace
                    + 0.5 * rate * floor_cross
                    + 0.5 * s_ff * rate * rate * q_max[a] * lambda_max_u
                    + 0.5 * s_f * rate * lambda_max_au[a]
            });
            let gate_au = Array1::from_shape_fn(p, |a| {
                g11 * q_min[a] * lambda_min_u
                    + g12 * (q_min[a] * lambda_max_u + q_max[a] * lambda_min_u)
                    + g22 * q_max[a] * lambda_max_u
                    + g1 * lambda_min_au[a]
                    + g2 * lambda_max_au[a]
            });
            remainder_drift += &(outer(&gate_au, &grad_u)
                + outer(&grad_g, &value_au)
                + outer(&value_au, &grad_g)
                + outer(&grad_u, &gate_au));
            let quadratic = |x_min: &Array1<f64>, x_max: &Array1<f64>, y_min: &Array1<f64>, y_max: &Array1<f64>| {
                outer(x_min, y_min) * g11
                    + (outer(x_min, y_max) + outer(x_max, y_min)) * g12
                    + outer(x_max, y_max) * g22
            };
            let q2 = quadratic(&q_min, &q_max, &q_min, &q_max);
            let q2_u = outer(&q_min, &q_min) * (g111 * lambda_min_u + g112 * lambda_max_u)
                + (outer(&q_min, &q_max) + outer(&q_max, &q_min)) * (g112 * lambda_min_u + g122 * lambda_max_u)
                + outer(&q_max, &q_max) * (g122 * lambda_min_u + g222 * lambda_max_u)
                + quadratic(&lambda_min_au, &lambda_max_au, &q_min, &q_max)
                + quadratic(&q_min, &q_max, &lambda_min_au, &lambda_max_au);
            remainder_drift.scaled_add(value_u, &q2);
            remainder_drift.scaled_add(ungated, &q2_u);
            if rate != 0.0 {
                let floor_trace_au = Array1::from_shape_fn(p, |a| {
                    (0..m)
                        .map(|i| {
                            (inverse_lambda_floor[i] * e_u[[i, i]] + inverse_floor_floor[i] * floor_u)
                                * row_entry(a_rows, a, i, i)
                                + inverse_floor[i] * lambda_au(i, a)
                        })
                        .sum::<f64>()
                });
                let floor_part = (outer(&floor_trace, &q_max) + outer(&q_max, &floor_trace)) * (0.5 * rate)
                    + outer(&q_max, &q_max) * (0.5 * s_ff * rate * rate);
                let floor_part_u = (outer(&floor_trace_au, &q_max)
                    + outer(&floor_trace, &lambda_max_au)
                    + outer(&lambda_max_au, &floor_trace)
                    + outer(&q_max, &floor_trace_au))
                    * (0.5 * rate)
                    + outer(&q_max, &q_max) * (0.5 * s_ff_u * rate * rate)
                    + (outer(&lambda_max_au, &q_max) + outer(&q_max, &lambda_max_au)) * (0.5 * s_ff * rate * rate);
                remainder_drift.scaled_add(gate_u, &floor_part);
                remainder_drift.scaled_add(gate, &floor_part_u);
            }
            // `ω_e·E_e` with `E_e[a,b] = D²λ_e[P̃_a, P̃_b]`, and its drift
            // `D²λ_e[B̃_au, P̃_b] + D²λ_e[P̃_a, B̃_bu] + D³λ_e[P̃_a, P̃_b, e_u]`.
            for (e, omega, omega_u) in [(imin, omega_min, omega_min_u), (imax, omega_max, omega_max_u)] {
                let gi = Array1::from_shape_fn(m, |j| gap(e, j));
                let x = Array2::from_shape_fn((p, m), |(a, j)| row_entry(a_rows, a, e, j));
                let y = Array2::from_shape_fn((p, m), |(a, j)| row_entry(b_u, a, e, j));
                let doubled = Array2::from_diag(&gi.mapv(|value| 2.0 * value));
                let extreme_second = x.dot(&doubled).dot(&x.t());
                let x_hat = Array2::from_shape_fn((p, m), |(a, j)| x[[a, j]] * gi[j]);
                let c_hat = Array1::from_shape_fn(m, |j| e_u[[e, j]] * gi[j]);
                let v = Array2::from_shape_fn((p, m), |(a, i)| {
                    (0..m).map(|k| row_entry(a_rows, a, i, k) * c_hat[k]).sum::<f64>()
                });
                let s = x_hat.dot(&c_hat);
                let q_e = Array1::from_shape_fn(p, |a| row_entry(a_rows, a, e, e));
                let third = (x_hat.dot(&v.t()) + v.dot(&x_hat.t())) * 2.0
                    + x_hat.dot(&e_u).dot(&x_hat.t()) * 2.0
                    - (outer(&q_e, &s) + outer(&s, &q_e)) * 2.0
                    - x_hat.dot(&x_hat.t()) * (2.0 * e_u[[e, e]]);
                let extreme_second_u = y.dot(&doubled).dot(&x.t()) + x.dot(&doubled).dot(&y.t()) + third;
                remainder_drift.scaled_add(omega_u, &extreme_second);
                remainder_drift.scaled_add(omega, &extreme_second_u);
            }
        }
        let mut result = contracted(&weight_now)? + contracted_along_u(&weight_along_u)?;
        if motion {
            result -= &remainder_drift;
        }
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|value| !value.is_finite()) {
            return Err("Jeffreys completion drift matrix produced nonfinite curvature".into());
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
        gam_model_api::jeffreys_rotated_axis_rows(axes, self.ambient_eigenbasis.view())
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
        // `D²f[E, A]_ij = Σ_k T_ikj (E_ik A_kj + A_ik E_kj)` for every axis row `A`.
        // Each row writes only its own output chunk, so the rows fan over rayon
        // with no cross-row reduction. Inside a row the loop order is (i, k, j):
        // for a fixed (i, k) the j sweep reads three contiguous rows and
        // vectorizes. Every output entry still starts at 0.0 and adds its k terms
        // in increasing k with the same operands, so each value is bit-identical
        // to the strided (i, j, k) scalar loop this replaces, the largest
        // drift-base self time on the rigid marginal-slope ψ gradient (#979).
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::slice::{ParallelSlice, ParallelSliceMut};
        let rows_per_task = (1usize << 15).div_ceil(squared.saturating_mul(m)).max(1);
        output
            .par_chunks_exact_mut(squared)
            .zip(input.par_chunks_exact(squared))
            .with_min_len(rows_per_task)
            .for_each(|(target, source)| {
                for i in 0..m {
                    let table_i = &table[i * squared..(i + 1) * squared];
                    let target_i = &mut target[i * m..(i + 1) * m];
                    for k in 0..m {
                        let (e_ik, source_ik) = (e[i * m + k], source[i * m + k]);
                        for (((value, &coefficient), &source_kj), &e_kj) in target_i
                            .iter_mut()
                            .zip(&table_i[k * m..(k + 1) * m])
                            .zip(&source[k * m..(k + 1) * m])
                            .zip(&e[k * m..(k + 1) * m])
                        {
                            *value += coefficient * (e_ik * source_kj + source_ik * e_kj);
                        }
                    }
                }
            });
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
        self.mixed_perturbation_derivative_from_frames(&u, &v, pert_uv, &self.rotate_axes(axes_uv)?)
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
    /// objects — `H_uv`, its rotated coefficient-axis derivatives and the second-order
    /// spectral rows — are formed here.
    pub fn mixed_perturbation_derivative_from_frames(
        &self,
        u: &JeffreysDirectionFrame,
        v: &JeffreysDirectionFrame,
        pert_uv: &Array2<f64>,
        axes_uv: &JeffreysRotatedAxes,
    ) -> Result<Array2<f64>, String> {
        if pert_uv.dim() != (self.p, self.p) {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let e = &u.e;
        let f = &v.e;
        let ef = symmetric_basis_contraction(pert_uv.view(), self.ambient_eigenbasis.view());
        let auv = &axes_uv.rows;
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
        wuv += &self.inverse_frechet_rows(auv, &[], 0);
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

/// Gate and floor channels along two directions and the capped inverse's first and second
/// Fréchet weights in the base eigenbasis; see
/// [`JeffreysHphiDriftBase::frozen_second_drift_weights`].
struct FrozenSecondDriftWeights {
    gate_u: f64,
    gate_w: f64,
    gate_uw: f64,
    /// `f(λ_i)` of the capped inverse.
    kernel: Vec<f64>,
    /// `Df[E_u]` plus the floor channel along `u`.
    weight_u: Array2<f64>,
    /// `Df[E_w]` plus the floor channel along `w`.
    weight_w: Array2<f64>,
    /// `D²f[E_u, E_w] + Df[E_uw]` plus the floor channels along `(u, w)`.
    weight_uw: Array2<f64>,
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
