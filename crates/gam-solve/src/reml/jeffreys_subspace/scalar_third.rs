//! Third derivatives of the complete scalar Jeffreys term. These are
//! hand-derived spectral chain rules; no eigenvector differentiation is used
//! for interior eigenvalues, including repeated ones.
use super::*;
use super::mixed::inverse_difference;

fn floor_third(x: f64, floor: f64) -> f64 {
    if x >= jeffreys_cap(floor) {
        if floor > CONDITIONING_GATE_ABSOLUTE_CLEAR { 2.0 / floor.powi(3) } else { 0.0 }
    } else if x >= floor {
        0.0
    } else if x >= 0.0 {
        2.0 / floor.powi(3) - 6.0 * x / floor.powi(4)
    } else {
        2.0 / floor.powi(3) - 6.0 * x / (floor - x).powi(4)
    }
}

fn gate_third(lo: f64, hi: f64) -> [f64; 4] {
    let (gm, gx) = conditioning_gate_weight_grad(lo, hi);
    if gm == 0.0 && gx == 0.0 { return [0.0; 4]; }
    if gx == 0.0 {
        return [12.0 / (CONDITIONING_GATE_ABSOLUTE_CLEAR - CONDITIONING_GATE_ABSOLUTE).powi(3), 0.0, 0.0, 0.0];
    }
    let span = CONDITIONING_GATE_RELATIVE_CLEAR.log10() - CONDITIONING_GATE_RELATIVE.log10();
    let t = ((lo / hi).log10() - CONDITIONING_GATE_RELATIVE.log10()) / span;
    let w1 = -6.0 * t * (1.0 - t) / span;
    let w2 = -6.0 * (1.0 - 2.0 * t) / span.powi(2);
    let w3 = 12.0 / span.powi(3);
    let ln = std::f64::consts::LN_10;
    let (a, b) = (1.0 / (lo * ln), -1.0 / (hi * ln));
    let (aa, bb) = (-1.0 / (lo * lo * ln), 1.0 / (hi * hi * ln));
    [w3*a*a*a + 3.0*w2*a*aa + 2.0*w1/(lo.powi(3)*ln),
     w3*a*a*b + w2*aa*b, w3*a*b*b + w2*a*bb,
     w3*b*b*b + 3.0*w2*b*bb - 2.0*w1/(hi.powi(3)*ln)]
}

impl JeffreysHphiDriftBase {
    /// Pure matrix-direction eigenvalue derivatives, ordered as E,F,A,EF,EA,FA,EFA.
    fn extreme_third(&self, i: usize, d: [&Array2<f64>; 3], needed: bool) -> Result<[f64; 7], String> {
        if !needed { return Ok([0.0; 7]); }
        let mut out = [d[0][[i,i]], d[1][[i,i]], d[2][[i,i]], 0.0, 0.0, 0.0, 0.0];
        let mut gaps = vec![0.0; self.m];
        for j in 0..self.m {
            if j == i { continue; }
            let gap = self.evals[i] - self.evals[j];
            if gap.abs() <= 16.0 * f64::EPSILON * self.evals[i].abs().max(self.evals[j].abs()) {
                return Err("third Jeffreys derivative is undefined at a repeated active extreme eigenvalue".into());
            }
            gaps[j] = gap.recip();
            for (s, (u,v)) in [(0,1), (0,2), (1,2)].into_iter().enumerate() {
                out[3+s] += (d[u][[i,j]]*d[v][[j,i]] + d[v][[i,j]]*d[u][[j,i]]) * gaps[j];
            }
        }
        for [u,v,w] in [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]] {
            for j in 0..self.m {
                if j == i { continue; }
                out[6] -= d[u][[i,i]] * d[v][[i,j]] * d[w][[j,i]] * gaps[j].powi(2);
                for k in 0..self.m {
                    if k != i {
                        out[6] += d[u][[i,j]] * d[v][[j,k]] * d[w][[k,i]] * gaps[j] * gaps[k];
                    }
                }
            }
        }
        Ok(out)
    }

    fn scalar_third(&self, e: &Array2<f64>, f: &Array2<f64>, a: &Array2<f64>) -> Result<f64, String> {
        let directions = [e,f,a];
        let first = |q, x: &Array2<f64>| (0..self.m).map(|i|
            0.5 * inverse_difference(&[self.evals[i]], self.floor, q) * x[[i,i]]).sum::<f64>();
        let second = |q, x: &Array2<f64>, y: &Array2<f64>| {
            let mut value = 0.0;
            for i in 0..self.m { for j in 0..self.m {
                value += 0.5 * inverse_difference(&[self.evals[i],self.evals[j]], self.floor,q) * x[[i,j]] * y[[j,i]];
            }}
            value
        };
        let u0 = 0.5 * self.evals.iter().map(|&x| jeffreys_antiderivative(x,self.floor)).sum::<f64>();
        let uf = 0.5 * self.evals.iter().map(|&x| jeffreys_antiderivative_floor_sensitivity(x,self.floor)).sum::<f64>();
        let uff = 0.5 * self.evals.iter().map(|&x| jeffreys_antiderivative_floor_second_sensitivity(x,self.floor)).sum::<f64>();
        let ufff = 0.5 * self.evals.iter().map(|&x| floor_third(x,self.floor)).sum::<f64>();
        let (gm,gx) = conditioning_gate_weight_grad(self.evals[self.idx_min],self.evals[self.idx_max]);
        let (gmm,gmx,gxx) = conditioning_gate_weight_hess(self.evals[self.idx_min],self.evals[self.idx_max]);
        let gt = gate_third(self.evals[self.idx_min], self.evals[self.idx_max]);
        let floor_moves = self.floor_in_relative_regime && (uf != 0.0 || uff != 0.0 || ufff != 0.0);
        let mn = self.extreme_third(self.idx_min,directions,gm != 0.0 || gmm != 0.0 || gmx != 0.0)?;
        let mx = self.extreme_third(self.idx_max,directions,gx != 0.0 || gxx != 0.0 || gmx != 0.0 || floor_moves)?;
        let q = mx.map(|v| if floor_moves { REDUCED_INFO_RELATIVE_FLOOR*v } else {0.0});
        let mut g = [0.0;7];
        for i in 0..7 { g[i] = gm*mn[i] + gx*mx[i]; }
        let gh = |i,j| gmm*mn[i]*mn[j] + gmx*(mn[i]*mx[j]+mx[i]*mn[j]) + gxx*mx[i]*mx[j];
        for (s,(i,j)) in [(0,1),(0,2),(1,2)].into_iter().enumerate() { g[3+s] += gh(i,j); }
        g[6] += gh(3,2)+gh(4,1)+gh(5,0)
            + gt[0]*mn[0]*mn[1]*mn[2]
            + gt[1]*(mx[0]*mn[1]*mn[2]+mn[0]*mx[1]*mn[2]+mn[0]*mn[1]*mx[2])
            + gt[2]*(mn[0]*mx[1]*mx[2]+mx[0]*mn[1]*mx[2]+mx[0]*mx[1]*mn[2])
            + gt[3]*mx[0]*mx[1]*mx[2];
        let mut l = [0.0;7];
        for i in 0..3 { l[i] = first(0,directions[i])+uf*q[i]; }
        for (s,(i,j)) in [(0,1),(0,2),(1,2)].into_iter().enumerate() {
            l[3+s] = second(0,directions[i],directions[j]) + first(1,directions[i])*q[j]
                + first(1,directions[j])*q[i]+uff*q[i]*q[j]+uf*q[3+s];
        }
        for i in 0..self.m { for j in 0..self.m { for k in 0..self.m {
            l[6] += 0.5*inverse_difference(&[self.evals[i],self.evals[j],self.evals[k]],self.floor,0)
                *(e[[i,j]]*f[[j,k]]+f[[i,j]]*e[[j,k]])*a[[k,i]];
        }}}
        for (s,(i,j,k)) in [(0,1,2),(0,2,1),(1,2,0)].into_iter().enumerate() {
            l[6] += second(1,directions[i],directions[j])*q[k]
                + first(2,directions[k])*q[i]*q[j]
                + first(1,directions[k])*q[3+s] + uff*q[3+s]*q[k];
        }
        l[6] += ufff*q[0]*q[1]*q[2]+uf*q[6];
        Ok(self.gate_weight*l[6]+g[0]*l[5]+g[1]*l[4]+g[2]*l[3]
            +g[3]*l[2]+g[4]*l[1]+g[5]*l[0]+g[6]*u0)
    }

    /// ∂_beta ∂_u ∂_v Phi, including explicit motion of every information
    /// derivative, the conditioning gate, and the relative spectral floor.
    pub fn explicit_score_pair(&self, u: &Array2<f64>, v: &Array2<f64>, uv: &Array2<f64>,
        axes_u: Vec<Array2<f64>>, axes_v: Vec<Array2<f64>>, axes_uv: Vec<Array2<f64>>) -> Result<Array1<f64>,String> {
        if [u,v,uv].iter().any(|x| x.dim() != (self.p,self.p)) {
            return Err("Jeffreys third scalar information dimension mismatch".into());
        }
        let rotate = |h: &Array2<f64>| symmetric_basis_contraction(h.view(),self.ambient_eigenbasis.view());
        let (e,f,ef) = (rotate(u),rotate(v),rotate(uv));
        let (au,av,auv) = (self.rotate_axis_rows(axes_u)?,self.rotate_axis_rows(axes_v)?,self.rotate_axis_rows(axes_uv)?);
        // Every matrix below is already in the prepared eigenbasis. Reuse
        // that spectrum exactly: decomposing its diagonal again costs cubic
        // work and needlessly chooses a second basis for repeated eigenvalues.
        let plan = JointJeffreysPlan {
            z_j: Array2::eye(self.m),
            reduced_dim: self.m,
            evals: self.evals.clone(),
            evecs: Array2::eye(self.m),
            lambda_min: self.evals[self.idx_min],
            lambda_max: self.evals[self.idx_max].max(0.0),
            gate_weight: self.gate_weight,
            floor: self.floor,
            floor_in_relative_regime: self.floor_in_relative_regime,
            idx_min: self.idx_min,
            idx_max: self.idx_max,
        };
        let zero = Array2::zeros((self.m,self.m));
        let (we,wf,wef,w0) = (plan.explicit_param_mixed_trace_weights(&e)?,plan.explicit_param_mixed_trace_weights(&f)?,
            plan.explicit_param_mixed_trace_weights(&ef)?,plan.explicit_param_mixed_trace_weights(&zero)?);
        let row = |rows: &Array2<f64>, i| Array2::from_shape_fn((self.m,self.m), |(j,k)| rows[[i,j*self.m+k]]);
        let mut out = Array1::<f64>::zeros(self.p);
        for i in 0..self.p {
            let a = row(&self.a_rows,i);
            out[i] = self.scalar_third(&e,&f,&a)? + wef.contract(&a,&zero)?
                + we.contract(&row(&av,i),&zero)? + wf.contract(&row(&au,i),&zero)?
                + w0.contract(&zero,&row(&auv,i))?;
        }
        if out.iter().any(|v| !v.is_finite()) { return Err("Jeffreys third scalar derivative is nonfinite".into()); }
        Ok(out)
    }
}
