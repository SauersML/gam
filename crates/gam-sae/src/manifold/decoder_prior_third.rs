use gam_terms::analytic_penalties::{
    PreparedDecoderIncoherence,
    normalized_gram::{GramNormalization, NormalizedCrossGram},
};

/// Contracted third derivatives of the same frozen-gate priors that supply
/// the exact beta Hessian. Directions are in full decoder coordinates.
pub(crate) struct PreparedExactDecoderPriorThird {
    repulsion: Option<PreparedDecoderIncoherence>,
    amplitude: Vec<(std::ops::Range<usize>, Array1<f64>, f64, f64)>,
    separation: Vec<PreparedSeparationThird>,
}

struct SeparationThirdEdge {
    j: usize,
    k: usize,
    q: f64,
    indices: Vec<usize>,
    geometry: NormalizedCrossGram,
    gradient: Array1<f64>,
    overlap: f64,
    left_width: usize,
    norms: [f64; 2],
    norm_gradients: Array1<f64>,
}

struct PreparedSeparationThird {
    eps: f64,
    values: Array1<f64>,
    vectors: Array2<f64>,
    first_divided: Array2<f64>,
    gradient: Array2<f64>,
    edges: Vec<SeparationThirdEdge>,
}

impl PreparedSeparationThird {
    fn majorizer_bilinear_add(&self, left: &Array1<f64>, right: &Array1<f64>, out: &mut Array1<f64>) {
        let n = self.values.len();
        let mut g_adjoint = Array2::<f64>::zeros((n, n));
        let mut local = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            let l = Array1::from_iter(edge.indices.iter().map(|&i| left[i]));
            let r = Array1::from_iter(edge.indices.iter().map(|&i| right[i]));
            let hl = edge.geometry.hessian_action(l.view());
            let hr = edge.geometry.hessian_action(r.view());
            local.push((edge.gradient.dot(&l), edge.gradient.dot(&r), hl, hr));
            let alpha = -edge.q * self.gradient[[edge.j, edge.k]];
            let mut radial = 0.0;
            for (part, range) in [0..edge.left_width, edge.left_width..edge.indices.len()].into_iter().enumerate() {
                let trace: f64 = range.clone().map(|i| l[i] * r[i]).sum();
                radial += trace / edge.norms[part];
                let weight = -2.0 * alpha.abs() * edge.overlap * trace / edge.norms[part].powi(2);
                for i in range {
                    out[edge.indices[i]] += weight * edge.norm_gradients[i];
                }
            }
            for (i, &destination) in edge.indices.iter().enumerate() {
                out[destination] += 2.0 * alpha.abs() * radial * edge.gradient[i];
            }
            let weight = -edge.q * alpha.signum() * edge.overlap * radial;
            g_adjoint[[edge.j, edge.k]] += weight;
            g_adjoint[[edge.k, edge.j]] += weight;
        }
        for (a, ea) in self.edges.iter().enumerate() {
            for (b, eb) in self.edges.iter().enumerate() {
                let (la, ra, hl, hr) = &local[a];
                let (lb, rb, _, _) = &local[b];
                let g = &self.gradient;
                let m = ea.q * eb.q * (g[[ea.j, eb.k]] * g[[ea.k, eb.j]]
                    + g[[ea.j, eb.j]] * g[[ea.k, eb.k]]);
                for (i, &destination) in ea.indices.iter().enumerate() {
                    out[destination] += m * (rb * hl[i] + lb * hr[i]);
                }
                let weight = 0.5 * ea.q * eb.q * (la * rb + ra * lb);
                for ((i, j), value) in [
                    ((ea.j, eb.k), g[[ea.k, eb.j]]),
                    ((ea.k, eb.j), g[[ea.j, eb.k]]),
                    ((ea.j, eb.j), g[[ea.k, eb.k]]),
                    ((ea.k, eb.k), g[[ea.j, eb.j]]),
                ] {
                    g_adjoint[[i, j]] += weight * value;
                }
            }
        }
        let f_adjoint = self.first(&g_adjoint);
        for edge in &self.edges {
            let weight = edge.q * (f_adjoint[[edge.j, edge.k]] + f_adjoint[[edge.k, edge.j]]);
            for (i, &destination) in edge.indices.iter().enumerate() {
                out[destination] += weight * edge.gradient[i];
            }
        }
    }

    fn first(&self, direction: &Array2<f64>) -> Array2<f64> {
        let rotated = self.vectors.t().dot(direction).dot(&self.vectors);
        self.vectors.dot(&(&self.first_divided * &rotated)).dot(&self.vectors.t())
    }

    fn second(&self, left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
        let l = self.vectors.t().dot(left).dot(&self.vectors);
        let r = self.vectors.t().dot(right).dot(&self.vectors);
        let n = self.values.len();
        let mut out = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let mut indices = [i, j, k];
                    indices.sort_unstable();
                    let [a, b, c] = indices;
                    let gap = self.values[c] - self.values[a];
                    let scale = self.values[a].abs().max(self.values[c].abs()).max(self.eps);
                    // A second divided difference loses O(eps/gap^2) to
                    // cancellation; its confluent expansion loses O(gap^2).
                    let dd = if gap <= f64::EPSILON.sqrt().sqrt() * scale {
                        let mean = self.values[a]
                            + ((self.values[b] - self.values[a]) + gap) / 3.0;
                        0.5 * SaeManifoldTerm::barrier_log_floor_third(mean, self.eps)
                    } else {
                        (self.first_divided[[b, c]] - self.first_divided[[a, b]]) / gap
                    };
                    out[[i, j]] += dd * (l[[i, k]] * r[[k, j]] + r[[i, k]] * l[[k, j]]);
                }
            }
        }
        self.vectors.dot(&out).dot(&self.vectors.t())
    }

    fn bilinear_add(&self, left: &Array1<f64>, right: &Array1<f64>, out: &mut Array1<f64>) {
        if self.edges.iter().flat_map(|edge| &edge.indices).all(|&i| left[i] == 0.0)
            || self.edges.iter().flat_map(|edge| &edge.indices).all(|&i| right[i] == 0.0)
        {
            return;
        }
        let n = self.values.len();
        let mut fl = Array2::zeros((n, n));
        let mut fr = Array2::zeros((n, n));
        let mut flr = Array2::zeros((n, n));
        let mut local = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            let l = Array1::from_iter(edge.indices.iter().map(|&i| left[i]));
            let r = Array1::from_iter(edge.indices.iter().map(|&i| right[i]));
            let hl = edge.geometry.hessian_action(l.view());
            let hr = edge.geometry.hessian_action(r.view());
            for (matrix, value) in [
                (&mut fl, edge.gradient.dot(&l)),
                (&mut fr, edge.gradient.dot(&r)),
                (&mut flr, l.dot(&hr)),
            ] {
                matrix[[edge.j, edge.k]] = edge.q * value;
                matrix[[edge.k, edge.j]] = edge.q * value;
            }
            local.push((l, r, hl, hr));
        }
        let gl = self.first(&fl);
        let gr = self.first(&fr);
        let glr = self.second(&fl, &fr) + self.first(&flr);
        for (edge, (l, r, hl, hr)) in self.edges.iter().zip(local) {
            let third = edge.geometry.third_bilinear(l.view(), r.view());
            // P=-1/2 tr log(m(F)); each overlap moves both F[j,k] and
            // F[k,j]. The factors 2 and -1/2 cancel at every chain-rule leg.
            for (index, &destination) in edge.indices.iter().enumerate() {
                out[destination] -= edge.q * (
                    glr[[edge.j, edge.k]] * edge.gradient[index]
                    + gr[[edge.j, edge.k]] * hl[index]
                    + gl[[edge.j, edge.k]] * hr[index]
                    + self.gradient[[edge.j, edge.k]] * third[index]
                );
            }
        }
    }
}

impl PreparedExactDecoderPriorThird {
    pub(crate) fn bilinear(&self, exact: bool, left: &Array1<f64>, right: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(left.len());
        if let Some(repulsion) = self.repulsion.as_ref() {
            repulsion.theta_bilinear_add(
                exact,
                left.as_slice().expect("contiguous decoder direction"),
                right.as_slice().expect("contiguous decoder direction"),
                out.as_slice_mut().expect("contiguous decoder adjoint"),
            );
        }
        for (range, beta, phi_prime, phi_second) in &self.amplitude {
            let l = left.slice(s![range.clone()]);
            let r = right.slice(s![range.clone()]);
            let bl = beta.dot(&l);
            let br = beta.dot(&r);
            let lr = l.dot(&r);
            for (i, destination) in range.clone().enumerate() {
                out[destination] += if exact {
                    4.0 * phi_prime * (lr * beta[i] + bl * r[i] + br * l[i])
                        + 8.0 * phi_second * bl * br * beta[i]
                } else {
                    (12.0 * phi_prime + 8.0 * beta.dot(beta) * phi_second) * lr * beta[i]
                };
            }
        }
        for separation in &self.separation {
            if exact {
                separation.bilinear_add(left, right, &mut out);
            } else {
                separation.majorizer_bilinear_add(left, right, &mut out);
            }
        }
        out
    }
}

impl SaeManifoldTerm {
    /// Third derivative of log(m(lambda)), including its exact affine and
    /// exponential tails. Used at confluent spectral divided differences.
    fn barrier_log_floor_third(lam: f64, eps: f64) -> f64 {
        let x = (lam + eps) / eps;
        if x >= 30.0 {
            return 2.0 / (lam + eps).powi(3);
        }
        if x <= -30.0 {
            return 0.0;
        }
        let m = Self::barrier_spectral_m(lam, eps);
        let sigma = Self::barrier_spectral_m_prime(lam, eps);
        let m2 = Self::barrier_spectral_m_second(lam, eps);
        let m3 = sigma * (1.0 - sigma) * (1.0 - 2.0 * sigma) / (eps * eps);
        m3 / m - 3.0 * (sigma / m) * (m2 / m) + 2.0 * (sigma / m).powi(3)
    }

    pub(crate) fn prepare_exact_decoder_prior_third(&self) -> Result<PreparedExactDecoderPriorThird, String> {
        let flat = self.flatten_beta();
        let offsets = self.beta_offsets();
        let repulsion = self.live_decoder_repulsion_penalty().map(|penalty| {
            penalty.prepare_curvature(flat.view(), Array1::zeros(0).view())
        });
        let mut amplitude = Vec::new();
        let energies: Vec<f64> = self.atoms.iter().map(|atom| {
            atom.decoder_coefficients().iter().map(|x| x * x).sum()
        }).collect();
        if let Some(floor) = self.amplitude_barrier_gate {
            for (i, atom) in self.atoms.iter().enumerate() {
                let u = energies[i];
                if let Some((_, g, radial)) = Self::amplitude_barrier_scalars(u, floor, SAE_AMPLITUDE_BARRIER_STRENGTH) {
                    let phi_prime = (radial - g) / (4.0 * u);
                    let phi_second = phi_prime * (2.0 / (2.0 * u + floor) - 2.0 / u - 2.0 / (u + floor));
                    let range = offsets[i]..offsets[i] + atom.decoder_coefficients().len();
                    amplitude.push((range.clone(), flat.slice(s![range]).to_owned(), phi_prime, phi_second));
                }
            }
        }
        let mut separation = Vec::new();
        for component in self.barrier_components(&energies, Self::barrier_norm_floor_sq(&energies)) {
            if !component.eps.is_finite() || component.edges.is_empty() {
                continue;
            }
            let n = component.atoms.len();
            let mut f = Array2::eye(n);
            let mut edges = Vec::with_capacity(component.edges.len());
            for edge in &component.edges {
                f[[edge.jl, edge.kl]] = edge.q * edge.o;
                f[[edge.kl, edge.jl]] = edge.q * edge.o;
                let left = self.atoms[edge.j].decoder_coefficients();
                let right = self.atoms[edge.k].decoder_coefficients();
                let geometry = NormalizedCrossGram::new(left.view(), right.view(), GramNormalization::SelfGramNorm)
                    .ok_or_else(|| "live separation edge has a zero Gram normalizer".to_string())?;
                let indices = (offsets[edge.j]..offsets[edge.j] + left.len())
                    .chain(offsets[edge.k]..offsets[edge.k] + right.len()).collect();
                let gradient = geometry.gradient();
                let norms = [Self::decoder_self_gram_frobenius_norm(left), Self::decoder_self_gram_frobenius_norm(right)];
                let norm_gradients = Array1::from_iter(
                    left.dot(&left.t()).dot(left).iter().map(|v| 2.0 * v / norms[0])
                        .chain(right.dot(&right.t()).dot(right).iter().map(|v| 2.0 * v / norms[1]))
                );
                edges.push(SeparationThirdEdge { j: edge.jl, k: edge.kl, q: edge.q, indices, geometry, gradient,
                    overlap: edge.o, left_width: left.len(), norms, norm_gradients });
            }
            let (values, vectors) = f.eigh(faer::Side::Lower)
                .map_err(|error| format!("separation third-derivative eigensystem: {error:?}"))?;
            let first_divided = Array2::from_shape_fn((n, n), |(i, j)| {
                Self::barrier_spectral_f_prime_divided(values[i], values[j], component.eps)
            });
            let g = values.mapv(|value| Self::barrier_spectral_m_prime(value, component.eps)
                / Self::barrier_spectral_m(value, component.eps));
            let gradient = (&vectors * &g).dot(&vectors.t());
            separation.push(PreparedSeparationThird { eps: component.eps, values, vectors, first_divided, gradient, edges });
        }
        Ok(PreparedExactDecoderPriorThird { repulsion, amplitude, separation })
    }

    pub(crate) fn exact_decoder_prior_theta_pair_add(
        &self, cache: &ArrowFactorCache, prepared: &PreparedExactDecoderPriorThird,
        left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>, weight: f64,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let projection = crate::frames::FrameProjection::new(self);
        let framed = self.last_frames_active && cache.k == self.factored_border_dim();
        if framed {
            let result = prepared.bilinear(true, &projection.lift_border_vec(left), &projection.lift_border_vec(right));
            out.scaled_add(weight, &projection.project_border_vec(result.view()));
        } else if cache.k == self.beta_dim() {
            out.scaled_add(weight, &prepared.bilinear(true, &left.to_owned(), &right.to_owned()));
        } else {
            return Err("exact decoder-prior theta adjoint has an inconsistent border frame".to_string());
        }
        Ok(())
    }

    pub(crate) fn exact_decoder_prior_theta_trace(
        &self, cache: &ArrowFactorCache, inverse: ArrayView2<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        let prepared = self.prepare_exact_decoder_prior_third()?;
        let mut out = Array1::zeros(cache.k);
        let mut unit = Array1::zeros(cache.k);
        for col in 0..cache.k {
            unit[col] = 1.0;
            self.exact_decoder_prior_theta_pair_add(cache, &prepared, inverse.column(col), unit.view(), 1.0, &mut out)?;
            unit[col] = 0.0;
        }
        Ok(out)
    }

    pub(crate) fn decoder_prior_gap_theta_trace(&self, cache: &ArrowFactorCache, inverse: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
        let prepared = self.prepare_exact_decoder_prior_third()?;
        let projection = crate::frames::FrameProjection::new(self);
        let framed = self.last_frames_active && cache.k == self.factored_border_dim();
        if !framed && cache.k != self.beta_dim() {
            return Err("decoder-prior gap adjoint has an inconsistent border frame".to_string());
        }
        let mut out = Array1::zeros(self.beta_dim());
        let mut unit = Array1::zeros(cache.k);
        for col in 0..cache.k {
            unit[col] = 1.0;
            let left = if framed { projection.lift_border_vec(inverse.column(col)) } else { inverse.column(col).to_owned() };
            let right = if framed { projection.lift_border_vec(unit.view()) } else { unit.clone() };
            out += &(prepared.bilinear(false, &left, &right) - prepared.bilinear(true, &left, &right));
            unit[col] = 0.0;
        }
        Ok(if framed { projection.project_border_vec(out.view()) } else { out })
    }
}
