//! Signed decoder-prior curvature captured at the majorizer assembly seam.

#[cfg(test)]
mod prepared_coherence_tests {
    use super::*;

    #[test]
    fn normalized_coherence_preparation_matches_live_hessian_and_third_derivatives_2820() {
        let p = DecoderIncoherencePenalty::new_sparse(
            PsiSlice {
                range: 0..15,
                latent_dim: Some(5),
            },
            vec![2, 3],
            3,
            vec![(0, 1, 0.7)],
            1.3,
            false,
        )
        .unwrap();
        let x = Array1::from_shape_fn(15, |i| (0.17 * (i + 1) as f64).sin() + 0.03 * i as f64);
        let l = Array1::from_shape_fn(15, |i| (0.31 * (i + 2) as f64).cos());
        let r = Array1::from_shape_fn(15, |i| (0.23 * (i + 4) as f64).sin());
        let rho = Array1::zeros(0);
        let prepared = p.prepare_curvature(x.view(), rho.view());
        let mut delta = vec![0.0; 15];
        prepared.remainder_action_add(l.as_slice().unwrap(), &mut delta);
        let reference = p.hvp(x.view(), rho.view(), l.view())
            - p.psd_majorizer_hvp(x.view(), rho.view(), l.view());
        for (actual, &expected) in delta.iter().zip(reference.iter()) {
            assert!((actual - expected).abs() < 1.0e-12 * (1.0 + expected.abs()));
        }
        let mut diagonal = vec![0.0; 15];
        prepared.remainder_diagonal_add(&mut diagonal);
        for col in 0..15 {
            let mut e = Array1::zeros(15);
            e[col] = 1.0;
            let exact = p.hvp(x.view(), rho.view(), e.view())[col];
            let majorizer = p.psd_majorizer_hvp(x.view(), rho.view(), e.view())[col];
            assert!((diagonal[col] - (exact - majorizer)).abs() < 1.0e-12);
        }
        for exact in [false, true] {
            let mut third = vec![0.0; 15];
            prepared.theta_bilinear_add(
                exact,
                l.as_slice().unwrap(),
                r.as_slice().unwrap(),
                &mut third,
            );
            let mut signal = 0.0_f64;
            for coordinate in 0..15 {
                let h = 2.0e-5;
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus[coordinate] += h;
                minus[coordinate] -= h;
                let contract = |state: ArrayView1<'_, f64>| {
                    let action = if exact {
                        p.hvp(state, rho.view(), r.view())
                    } else {
                        p.psd_majorizer_hvp(state, rho.view(), r.view())
                    };
                    l.dot(&action)
                };
                let fd = (contract(plus.view()) - contract(minus.view())) / (2.0 * h);
                signal = signal.max(fd.abs());
                assert!(
                    (third[coordinate] - fd).abs() < 1.0e-8 * (1.0 + fd.abs()),
                    "exact={exact}, coordinate={coordinate}: third={} FD={fd}",
                    third[coordinate]
                );
            }
            assert!(signal > 1.0e-3);
        }
    }
}

use super::*;
use gam_solve::arrow_schur::ExactBetaRemainder;
use gam_terms::analytic_penalties::normalized_gram::{GramNormalization, NormalizedCrossGram};
use ndarray::Array3;

struct OverlapAtom {
    beta: Array2<f64>,
    norm: f64,
    norm_gradient: Array1<f64>,
    offset: usize,
}

pub(super) fn separation_overlap_metric(
    edges: &[BarrierEdge],
    g: &Array2<f64>,
    scale: f64,
) -> Array2<f64> {
    let mut out = Array2::zeros((edges.len(), edges.len()));
    for a in 0..edges.len() {
        for b in 0..=a {
            let ea = &edges[a];
            let eb = &edges[b];
            let value = scale
                * ea.q
                * eb.q
                * (g[[ea.jl, eb.kl]] * g[[ea.kl, eb.jl]] + g[[ea.jl, eb.jl]] * g[[ea.kl, eb.kl]]);
            out[[a, b]] = value;
            out[[b, a]] = value;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_solve::arrow_schur::BetaPenaltyOp;

    #[test]
    fn separation_overlap_metric_is_a_positive_gram_on_frustrated_spectra_2820() {
        let edges: Vec<_> = [(0, 1), (0, 2)]
            .into_iter()
            .map(|(j, k)| BarrierEdge {
                j,
                k,
                jl: j,
                kl: k,
                q: 1.0,
                o: 1.0,
            })
            .collect();
        let mut f = Array2::eye(3);
        for edge in &edges {
            f[[edge.jl, edge.kl]] = 1.0;
            f[[edge.kl, edge.jl]] = 1.0;
        }
        let (eigenvalues, vectors) = f.eigh(faer::Side::Lower).unwrap();
        assert!(
            eigenvalues[0] < -0.4,
            "the witness must have a real negative mode"
        );
        let h = eigenvalues
            .mapv(|lambda| SaeManifoldTerm::barrier_spectral_log_derivatives(lambda, 0.2).1);
        assert!(h.iter().all(|&value| value > 0.0));
        let g = vectors.dot(&Array2::from_diag(&h)).dot(&vectors.t());
        let g_sqrt = vectors
            .dot(&Array2::from_diag(&h.mapv(f64::sqrt)))
            .dot(&vectors.t());
        let scale = 0.37;
        let actual = separation_overlap_metric(&edges, &g, scale);
        let (metric_values, _) = actual.eigh(faer::Side::Lower).unwrap();
        assert!(
            metric_values[0] > 0.0,
            "independent edge directions give a positive definite metric"
        );
        let gram_vectors: Vec<_> = edges
            .iter()
            .map(|edge| {
                let mut derivative = Array2::zeros((3, 3));
                derivative[[edge.jl, edge.kl]] = edge.q;
                derivative[[edge.kl, edge.jl]] = edge.q;
                g_sqrt.dot(&derivative).dot(&g_sqrt)
            })
            .collect();
        for a in 0..edges.len() {
            for b in 0..edges.len() {
                let expected = 0.5
                    * scale
                    * gram_vectors[a]
                        .iter()
                        .zip(gram_vectors[b].iter())
                        .map(|(&x, &y)| x * y)
                        .sum::<f64>();
                assert!((actual[[a, b]] - expected).abs() < 2.0e-13 * (1.0 + expected.abs()));
            }
        }
    }

    fn prior_system(term: &SaeManifoldTerm, kind: usize) -> ArrowSchurSystem {
        let mut system = ArrowSchurSystem::new(0, 0, term.beta_dim());
        let mut ridge = vec![0.0; term.k_atoms()];
        let mut carriers = Vec::new();
        let wrote = match kind {
            0 => term.add_sae_decoder_repulsion(&mut system, 0.37, true),
            1 => {
                term.add_sae_amplitude_barrier(&mut system, 0.37, &mut ridge);
                for (atom_idx, offset) in term.beta_offsets().into_iter().enumerate() {
                    for index in 0..term.atoms[atom_idx].decoder_coefficients().len() {
                        system.hbb[[offset + index, offset + index]] += ridge[atom_idx];
                    }
                }
                ridge.iter().any(|&value| value > 0.0)
            }
            2 => {
                term.add_sae_separation_barrier(&mut system, 0.37, true, &mut ridge, &mut carriers)
            }
            _ => unreachable!(),
        };
        assert!(wrote && !system.exact_beta_remainders.is_empty());
        system
    }

    fn perturbed(term: &SaeManifoldTerm, direction: &Array1<f64>, step: f64) -> SaeManifoldTerm {
        let mut changed = term.clone();
        changed.decoder_repulsion_gate = term.decoder_repulsion_gate.clone();
        changed.barrier_coactivation_gate = term.barrier_coactivation_gate.clone();
        changed.amplitude_barrier_gate = term.amplitude_barrier_gate;
        let offsets = term.beta_offsets();
        for (atom, &offset) in changed.atoms.iter_mut().zip(&offsets) {
            for (i, coefficient) in atom.decoder_coefficients_mut().iter_mut().enumerate() {
                *coefficient += step * direction[offset + i];
            }
        }
        changed
    }

    #[test]
    fn each_decoder_prior_exact_beta_action_matches_its_frozen_gradient_2820() {
        let (base, _, _) = crate::manifold::tests::small_two_atom_periodic_term();
        // A one-output decoder has identically unit normalized overlap, so it
        // cannot exercise the separation gradient's moving-shape terms.
        let atoms = base
            .atoms
            .iter()
            .enumerate()
            .map(|(index, atom)| {
                let decoder = Array2::from_shape_fn((atom.basis_size(), 2), |(row, col)| {
                    (0.31 * (1 + row + 3 * col + 2 * index) as f64).sin()
                        + 0.17 * (row == col) as u8 as f64
                });
                SaeManifoldAtom::new_with_provided_function_gram(
                    &format!("exact-beta-{index}"),
                    SaeAtomBasisKind::Periodic,
                    1,
                    atom.basis_values.clone(),
                    atom.basis_jacobian.clone(),
                    decoder,
                    Array2::eye(atom.basis_size()),
                )
                .unwrap()
            })
            .collect();
        let mut term = SaeManifoldTerm::new(atoms, base.assignment.clone()).unwrap();
        term.decoder_repulsion_gate = Some(vec![(0, 1, 0.7)]);
        term.amplitude_barrier_gate = Some(0.3);
        term.barrier_coactivation_gate = Some(BarrierCoactivationGate {
            pairs: vec![(0, 1, 0.8)],
            atom_neff: vec![80.0, 80.0],
        });
        let direction = Array1::from_shape_fn(term.beta_dim(), |i| ((i + 1) as f64 * 0.37).sin());
        for kind in 0..3 {
            let system = prior_system(&term, kind);
            let mut delta = vec![0.0; term.beta_dim()];
            for remainder in &system.exact_beta_remainders {
                remainder.matvec(direction.as_slice().unwrap(), &mut delta);
            }
            let action = system.hbb.dot(&direction) + &Array1::from_vec(delta.clone());
            let step = 2.0e-5;
            let plus = prior_system(&perturbed(&term, &direction, step), kind).gb;
            let minus = prior_system(&perturbed(&term, &direction, -step), kind).gb;
            let finite_difference = (plus - minus) / (2.0 * step);
            let scale = finite_difference
                .iter()
                .fold(1.0e-8_f64, |a, b| a.max(b.abs()));
            assert!(
                scale > 1.0e-6,
                "prior {kind} must have a live gradient derivative"
            );
            let error = action
                .iter()
                .zip(finite_difference.iter())
                .fold(0.0_f64, |a, (&x, &y)| a.max((x - y).abs()));
            let signal = delta.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
            eprintln!("beta prior {kind}: error={error:e}, scale={scale:e}, remainder={signal:e}");
            assert!(
                signal > 1.0e-8 * scale,
                "prior {kind} must exercise a nonzero omitted curvature"
            );
            assert!(
                error < 2.0e-6 * scale,
                "prior {kind}: exact beta action error {error:e} / {scale:e}"
            );
            let right = Array1::from_shape_fn(term.beta_dim(), |i| ((i + 3) as f64 * 0.19).cos());
            for exact in [false, true] {
                let mut third = vec![0.0; term.beta_dim()];
                for owner in &system.exact_beta_remainders {
                    owner
                        .theta_bilinear(
                            exact,
                            direction.as_slice().unwrap(),
                            right.as_slice().unwrap(),
                            &mut third,
                        )
                        .unwrap();
                }
                let mut max_error = 0.0_f64;
                let mut max_signal = 0.0_f64;
                for coordinate in 0..term.beta_dim() {
                    let mut axis = Array1::zeros(term.beta_dim());
                    axis[coordinate] = 1.0;
                    let contract = |changed: &SaeManifoldTerm| {
                        let assembled = prior_system(changed, kind);
                        let mut action = assembled.hbb.dot(&right);
                        if exact {
                            for owner in &assembled.exact_beta_remainders {
                                owner.matvec(
                                    right.as_slice().unwrap(),
                                    action.as_slice_mut().unwrap(),
                                );
                            }
                        }
                        direction.dot(&action)
                    };
                    let fd = (contract(&perturbed(&term, &axis, step))
                        - contract(&perturbed(&term, &axis, -step)))
                        / (2.0 * step);
                    let error = (third[coordinate] - fd).abs();
                    max_error = max_error.max(error);
                    max_signal = max_signal.max(fd.abs());
                    assert!(
                        error < 1.0e-8 * (1.0 + fd.abs()),
                        "prior={kind}, exact={exact}, coordinate={coordinate}: third={} FD={fd}",
                        third[coordinate]
                    );
                }
                assert!(
                    max_signal > 1.0e-6,
                    "prior={kind}, exact={exact} must exercise a live third contraction"
                );
                eprintln!(
                    "beta prior {kind}, exact={exact}: third_error={max_error:e}, third_signal={max_signal:e}"
                );
            }
        }
    }

    #[test]
    fn spectral_floor_log_derivatives_are_finite_in_the_underflow_tail_2820() {
        let eps = 0.01;
        let (value, first, second) = SaeManifoldTerm::barrier_spectral_log_derivatives(-20.0, eps);
        assert!(value.is_finite());
        assert_eq!(first, 1.0 / eps);
        assert_eq!(second, 0.0);
        for x in [-20.0, -3.0, 0.0, 2.0, 20.0, 40.0] {
            let lambda = eps * (x - 1.0);
            let step = eps * 1.0e-3;
            let analytic = SaeManifoldTerm::barrier_spectral_log_derivatives(lambda, eps).2;
            let hp = SaeManifoldTerm::barrier_spectral_log_derivatives(lambda + step, eps).1;
            let hm = SaeManifoldTerm::barrier_spectral_log_derivatives(lambda - step, eps).1;
            let fd = (hp - hm) / (2.0 * step);
            assert!(
                (analytic - fd).abs() < 2.0e-6 * (1.0 + analytic.abs()),
                "spectral x={x}: h'={analytic:e}, FD={fd:e}"
            );
        }
    }
}

/// Capture the exact Hessian of -1/2 tr log m(F(B)), minus the actual
/// coupled-carrier/Levenberg matrix just installed in the Newton system.
/// The spectral response is a Frechet derivative of h=m'/m, not -G dF G:
/// the latter is only correct on the linear spectral-floor branch.
pub(super) fn separation_remainder(
    term: &SaeManifoldTerm,
    component: &BarrierComponent,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    g: &Array2<f64>,
    majorizer_coupling: &Array2<f64>,
    carriers: &[Vec<(usize, Vec<f64>)>],
    scale: f64,
) -> ExactBetaRemainder {
    let offsets = term.beta_offsets();
    let atoms: Vec<OverlapAtom> = component
        .atoms
        .iter()
        .map(|&atom| {
            let beta = term.atoms[atom].decoder_coefficients().to_owned();
            let gram = beta.dot(&beta.t());
            let norm = gram.iter().map(|v| v * v).sum::<f64>().sqrt();
            let norm_gradient = if norm > 0.0 {
                Array1::from_iter(gram.dot(&beta).iter().map(|&value| 2.0 * value / norm))
            } else {
                Array1::zeros(beta.len())
            };
            OverlapAtom {
                beta,
                norm,
                norm_gradient,
                offset: offsets[atom],
            }
        })
        .collect();
    let overlap_geometry: Vec<_> = component
        .edges
        .iter()
        .map(|edge| {
            NormalizedCrossGram::new(
                atoms[edge.jl].beta.view(),
                atoms[edge.kl].beta.view(),
                GramNormalization::SelfGramNorm,
            )
        })
        .collect();
    let edges: Vec<_> = component
        .edges
        .iter()
        .map(|edge| {
            (
                edge.jl,
                edge.kl,
                edge.q,
                edge.o,
                -scale * edge.q * g[[edge.jl, edge.kl]],
            )
        })
        .collect();
    let global_carriers: Vec<Vec<(usize, Vec<f64>)>> = carriers
        .iter()
        .map(|runs| {
            runs.iter()
                .map(|(atom, values)| (offsets[*atom], values.clone()))
                .collect()
        })
        .collect();
    let coupling = majorizer_coupling.clone();
    let vectors = eigenvectors.clone();
    let eps = component.eps;
    let dd = Array2::from_shape_fn((eigenvalues.len(), eigenvalues.len()), |(i, j)| {
        spectral::h_first_difference(eigenvalues[i], eigenvalues[j], eps)
    });
    let mut fingerprint = gam_runtime::warm_start::Fingerprinter::new();
    fingerprint.write_str("sae-separation-exact-remainder-v1");
    fingerprint.write_f64(scale);
    for atom in &atoms {
        fingerprint.write_usize(atom.offset);
        fingerprint.write_f64_array2(&atom.beta);
    }
    for &(j, k, q, overlap, alpha) in &edges {
        fingerprint.write_usize(j);
        fingerprint.write_usize(k);
        fingerprint.write_f64(q);
        fingerprint.write_f64(overlap);
        fingerprint.write_f64(alpha);
    }
    fingerprint.write_f64_array2(&coupling);
    fingerprint.write_f64_array2(&vectors);
    fingerprint.write_f64_array2(&dd);
    let state = Arc::new(PreparedSeparation {
        atoms,
        edges,
        global_carriers,
        overlap_geometry,
        coupling,
        vectors,
        dd,
        eigenvalues: eigenvalues.clone(),
        g: g.clone(),
        scale,
        eps,
        second_dd: std::sync::OnceLock::new(),
    });
    let theta_owner = Arc::clone(&state);
    ExactBetaRemainder::new(
        term.beta_dim(),
        fingerprint.finish_u64(),
        move |direction, out| state.remainder_action(direction, out),
    )
    .with_theta_bilinear(move |exact, left, right, out| {
        theta_owner.theta_bilinear(exact, left, right, out)
    })
}

struct PreparedSeparation {
    atoms: Vec<OverlapAtom>,
    edges: Vec<(usize, usize, f64, f64, f64)>,
    global_carriers: Vec<Vec<(usize, Vec<f64>)>>,
    overlap_geometry: Vec<Option<NormalizedCrossGram>>,
    coupling: Array2<f64>,
    vectors: Array2<f64>,
    dd: Array2<f64>,
    eigenvalues: Array1<f64>,
    g: Array2<f64>,
    scale: f64,
    eps: f64,
    second_dd: std::sync::OnceLock<Array3<f64>>,
}

impl PreparedSeparation {
    fn overlap_direction(&self, direction: &[f64]) -> Array1<f64> {
        Array1::from_iter(self.global_carriers.iter().map(|runs| {
            runs.iter()
                .map(|(offset, values)| {
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, &value)| value * direction[offset + i])
                        .sum::<f64>()
                })
                .sum::<f64>()
        }))
    }

    fn f_direction(&self, overlap: &Array1<f64>) -> Array2<f64> {
        let mut out = Array2::zeros((self.atoms.len(), self.atoms.len()));
        for (e, &(j, k, q, _, _)) in self.edges.iter().enumerate() {
            out[[j, k]] = q * overlap[e];
            out[[k, j]] = out[[j, k]];
        }
        out
    }

    fn frechet(&self, direction: &Array2<f64>) -> Array2<f64> {
        let spectral = self.vectors.t().dot(&direction.dot(&self.vectors));
        self.vectors
            .dot(&(&self.dd * &spectral))
            .dot(&self.vectors.t())
    }

    fn pair_direction(&self, edge: usize, direction: &[f64]) -> Array1<f64> {
        let (j, k, _, _, _) = self.edges[edge];
        Array1::from_iter([j, k].into_iter().flat_map(|a| {
            let atom = &self.atoms[a];
            direction[atom.offset..atom.offset + atom.beta.len()]
                .iter()
                .copied()
        }))
    }

    fn scatter_pair(&self, edge: usize, values: &Array1<f64>, coefficient: f64, out: &mut [f64]) {
        let (j, k, _, _, _) = self.edges[edge];
        let mut local = 0;
        for a in [j, k] {
            let atom = &self.atoms[a];
            for i in 0..atom.beta.len() {
                out[atom.offset + i] += coefficient * values[local + i];
            }
            local += atom.beta.len();
        }
    }

    fn scatter_carrier(&self, edge: usize, coefficient: f64, out: &mut [f64]) {
        for (offset, values) in &self.global_carriers[edge] {
            for (i, &value) in values.iter().enumerate() {
                out[offset + i] += coefficient * value;
            }
        }
    }

    fn theta_bilinear(
        &self,
        exact: bool,
        left: &[f64],
        right: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let ol = self.overlap_direction(left);
        let or = self.overlap_direction(right);
        let fl = self.f_direction(&ol);
        let fr = self.f_direction(&or);
        if exact {
            let sl = self.vectors.t().dot(&fl.dot(&self.vectors));
            let sr = self.vectors.t().dot(&fr.dot(&self.vectors));
            let gl = self.vectors.dot(&(&self.dd * &sl)).dot(&self.vectors.t());
            let gr = self.vectors.dot(&(&self.dd * &sr)).dot(&self.vectors.t());
            let s = self.atoms.len();
            let second_dd = self.second_dd.get_or_init(|| {
                Array3::from_shape_fn((s, s, s), |(i, k, j)| {
                    spectral::h_second_difference(
                        self.eigenvalues[i],
                        self.eigenvalues[k],
                        self.eigenvalues[j],
                        self.eps,
                    )
                })
            });
            let second_response = Array2::from_shape_fn((s, s), |(i, j)| {
                (0..s)
                    .map(|k| {
                        second_dd[[i, k, j]] * (sl[[i, k]] * sr[[k, j]] + sr[[i, k]] * sl[[k, j]])
                    })
                    .sum::<f64>()
            });
            let mut g_second = self.vectors.dot(&second_response).dot(&self.vectors.t());
            let mut overlap_mixed = Array1::zeros(self.edges.len());
            for (edge, &(j, k, q, _, alpha)) in self.edges.iter().enumerate() {
                let Some(geometry) = &self.overlap_geometry[edge] else {
                    continue;
                };
                let l = self.pair_direction(edge, left);
                let r = self.pair_direction(edge, right);
                let hl = geometry.hessian_action(l.view());
                let hr = geometry.hessian_action(r.view());
                overlap_mixed[edge] = l.dot(&hr);
                self.scatter_pair(edge, &hl, -self.scale * q * gr[[j, k]], out);
                self.scatter_pair(edge, &hr, -self.scale * q * gl[[j, k]], out);
                self.scatter_pair(
                    edge,
                    &geometry.third_bilinear(l.view(), r.view()),
                    alpha,
                    out,
                );
            }
            // The scalar composition also differentiates alpha against the
            // mixed overlap Hessian. Omitting Dh[F_lr] loses a full third-order
            // chain-rule channel even when D²h itself is exact.
            g_second += &self.frechet(&self.f_direction(&overlap_mixed));
            for (edge, &(j, k, q, _, _)) in self.edges.iter().enumerate() {
                self.scatter_carrier(edge, -self.scale * q * g_second[[j, k]], out);
            }
        } else {
            let coupling_l = self.coupling.dot(&ol);
            let coupling_r = self.coupling.dot(&or);
            // Gradient with respect to G of 1/2 scale tr(G F_l G F_r).
            let mut g_weight =
                (fl.dot(&self.g.dot(&fr)) + fr.dot(&self.g.dot(&fl))) * (0.5 * self.scale);
            for (edge, &(j, k, q, overlap, alpha)) in self.edges.iter().enumerate() {
                let Some(geometry) = &self.overlap_geometry[edge] else {
                    continue;
                };
                if alpha == 0.0 && overlap > 0.0 {
                    return Err("separation majorizer theta derivative is undefined at an active absolute-force kink".into());
                }
                let l = self.pair_direction(edge, left);
                let r = self.pair_direction(edge, right);
                self.scatter_pair(
                    edge,
                    &geometry.hessian_action(l.view()),
                    coupling_r[edge],
                    out,
                );
                self.scatter_pair(
                    edge,
                    &geometry.hessian_action(r.view()),
                    coupling_l[edge],
                    out,
                );
                let mut inverse_norm_contraction = 0.0;
                for a in [j, k] {
                    let atom = &self.atoms[a];
                    let dot_lr = (0..atom.beta.len())
                        .map(|i| left[atom.offset + i] * right[atom.offset + i])
                        .sum::<f64>();
                    inverse_norm_contraction += dot_lr / atom.norm;
                    let coefficient =
                        -2.0 * alpha.abs() * overlap * dot_lr / (atom.norm * atom.norm);
                    for (i, &value) in atom.norm_gradient.iter().enumerate() {
                        out[atom.offset + i] += coefficient * value;
                    }
                }
                self.scatter_carrier(edge, 2.0 * alpha.abs() * inverse_norm_contraction, out);
                let force_weight =
                    -self.scale * q * alpha.signum() * overlap * inverse_norm_contraction;
                g_weight[[j, k]] += force_weight;
                g_weight[[k, j]] += force_weight;
            }
            let response = self.frechet(&g_weight);
            for (edge, &(j, k, q, _, _)) in self.edges.iter().enumerate() {
                self.scatter_carrier(edge, 2.0 * q * response[[j, k]], out);
            }
        }
        Ok(())
    }

    fn remainder_action(&self, direction: &[f64], out: &mut [f64]) {
        let overlap_velocities = Array1::from_iter(self.global_carriers.iter().map(|runs| {
            runs.iter()
                .map(|(offset, values)| {
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, &value)| value * direction[offset + i])
                        .sum::<f64>()
                })
                .sum::<f64>()
        }));
        let mut df = Array2::zeros((self.atoms.len(), self.atoms.len()));
        for (edge, &(j, k, q, _, _)) in self.edges.iter().enumerate() {
            df[[j, k]] = q * overlap_velocities[edge];
            df[[k, j]] = df[[j, k]];
        }
        let spectral_df = self.vectors.t().dot(&df.dot(&self.vectors));
        let dg = self
            .vectors
            .dot(&(&self.dd * &spectral_df))
            .dot(&self.vectors.t());
        let majorizer_action = self.coupling.dot(&overlap_velocities);
        for (edge, &(j, k, q, overlap, alpha)) in self.edges.iter().enumerate() {
            let coefficient = -self.scale * q * dg[[j, k]] - majorizer_action[edge];
            for (offset, values) in &self.global_carriers[edge] {
                for (i, &value) in values.iter().enumerate() {
                    out[offset + i] += coefficient * value;
                }
            }
            let Some(geometry) = &self.overlap_geometry[edge] else {
                continue;
            };
            let local_direction = Array1::from_iter([j, k].into_iter().flat_map(|a| {
                let atom = &self.atoms[a];
                direction[atom.offset..atom.offset + atom.beta.len()]
                    .iter()
                    .copied()
            }));
            let hessian_action = geometry.hessian_action(local_direction.view());
            let mut local_offset = 0;
            for a in [j, k] {
                let atom = &self.atoms[a];
                let levenberg = 2.0 * alpha.abs() * overlap / atom.norm;
                for i in 0..atom.beta.len() {
                    let h = hessian_action[local_offset + i];
                    let v = direction[atom.offset + i];
                    out[atom.offset + i] += alpha * h - levenberg * v;
                }
                local_offset += atom.beta.len();
            }
        }
    }
}
