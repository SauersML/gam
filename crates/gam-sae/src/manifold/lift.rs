//! Lifted linear solvers for curved SAE atoms — *curvature is linear structure
//! one polynomial degree up*.
//!
//! # The pattern
//!
//! A curved atom fit by the dense per-row Newton over latent coordinates `t` is
//! nonconvex and basin-plagued. But every curved atom in this crate is, by
//! construction, a *linear* map applied to a fixed nonlinear feature map `Φ(t)`
//! (harmonic phasors, degree-2 monomials, …). Fitting the linear block is a
//! *convex* problem — the **lifted fit** — and recovering the underlying spike
//! parameters `{(a_j, t_j)}` from the fitted linear block is a closed-form
//! algebraic descent. [`crate::super_resolution`] is exactly this pattern for the
//! **circle**: a degree-`H` harmonic circle is a linear map on
//! `(cos t, …, cos Ht, sin t, …, sin Ht)`, and the matrix-pencil / Prony descent
//! un-superposes the fitted Fourier block into point masses. This module
//! generalises the descent to the two remaining curved topologies the term
//! carries ([`crate::manifold::SaeAtomBasisKind::Sphere`],
//! [`crate::manifold::SaeAtomBasisKind::Torus`]).
//!
//! * **Sphere — the Veronese lift.** A mixture of `m` points `v_1..v_m ∈ S^{d-1}`
//!   with amplitudes `a_j > 0` lifts to the PSD matrix `M = Σ_j a_j v_j v_jᵀ`
//!   (the degree-2 Veronese / symmetric-outer-product feature block). The descent
//!   is a symmetric eigendecomposition: [`recover_sphere_spikes`].
//! * **Torus — the Kronecker pencil.** A spike at `(θ, φ) ∈ T²` with per-axis
//!   harmonic degrees `(H₁, H₂)` lifts to the Kronecker product of the two
//!   harmonic phasor vectors; `m` spikes give a sum of `m` Kronecker-rank-1 terms
//!   sampled on the `H₁ × H₂` grid. The descent is 2-D harmonic retrieval by an
//!   enhanced matrix pencil with *auto-paired* axes: [`recover_torus_spikes`].
//!
//! Both descents are exact only in the noiseless limit; [`polish_spikes`] runs a
//! few damped Gauss–Newton steps on the *original* nonconvex objective
//! `‖z − Σ_j a_j Φ(t_j)‖²` given a caller-supplied basis evaluation, and reports
//! the final residual so a caller can gate acceptance.

// ============================================================================
// Shared order selection (mirrors `super_resolution`'s derived thresholds)
// ============================================================================

// ============================================================================
// Sphere — the Veronese lift
// ============================================================================

/// A single recovered point mass on the sphere `S^{d-1}`.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereSpike {
    /// Canonical unit direction `v ∈ S^{d-1}` (length `d`). The lift `v vᵀ` is
    /// invariant under the antipodal flip `v ↦ −v`, so the reported vector is the
    /// canonical representative of the `{v, −v}` gauge orbit: its
    /// largest-magnitude component is non-negative (see [`canonicalize_direction`]).
    pub direction: Vec<f64>,
    /// Amplitude `a > 0` of the spike (the corresponding eigenvalue of the lift).
    pub amplitude: f64,
}

/// The full result of a Veronese-lift recovery.
#[derive(Clone, Debug)]
pub struct SphereRecovery {
    /// Recovered spikes, sorted by amplitude descending.
    pub spikes: Vec<SphereSpike>,
    /// Selected model order `m` (number of point masses), from the count of
    /// eigenvalues above the noise-derived floor.
    pub model_order: usize,
    /// Frobenius norm of `M̂ − Σ_j a_j v_j v_jᵀ` for the recovered model.
    pub residual: f64,
    /// Eigenvalues of the symmetrised lift, descending — the spectrum the order
    /// selection thresholded.
    pub eigenvalues: Vec<f64>,
}

// ============================================================================
// Torus — the Kronecker pencil (2-D harmonic retrieval, auto-paired)
// ============================================================================

/// A single recovered point mass on the torus `T² = S¹ × S¹`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorusSpike {
    /// Axis-1 position `θ ∈ [0, 1)` (fraction of a full turn); the phasor is
    /// `e^{2πi θ}`.
    pub theta: f64,
    /// Axis-2 position `φ ∈ [0, 1)`; the phasor is `e^{2πi φ}`.
    pub phi: f64,
    /// Amplitude `a > 0` (real part of the least-squares Kronecker-Vandermonde
    /// coefficient).
    pub amplitude: f64,
}

/// The full result of a Kronecker-pencil recovery.
#[derive(Clone, Debug)]
pub struct TorusRecovery {
    /// Recovered spikes, sorted by `(θ, φ)` ascending.
    pub spikes: Vec<TorusSpike>,
    /// Selected model order `m`.
    pub model_order: usize,
    /// Frobenius norm of `Ŷ − Σ_j a_j (z_j^{h₁} w_j^{h₂})` for the recovered
    /// model over the `H₁ × H₂` grid.
    pub residual: f64,
    /// Singular values of the enhanced (block-Hankel) matrix, descending.
    pub enhanced_singular_values: Vec<f64>,
}

// ============================================================================
// Polish — damped Gauss–Newton on the original nonconvex objective
// ============================================================================

/// Tuning for [`polish_spikes`].
#[derive(Clone, Debug)]
pub struct PolishOptions {
    /// Maximum outer Gauss–Newton iterations.
    pub max_iters: usize,
    /// Initial Levenberg–Marquardt damping `μ` (added as `μ‖δ‖²`). Small so the
    /// first step is nearly a pure Gauss–Newton step from a good pencil seed.
    pub initial_damping: f64,
    /// Stop when the residual norm improves by less than this between outer
    /// iterations (a local minimum has been reached to working precision).
    pub residual_tol: f64,
}

impl Default for PolishOptions {
    fn default() -> Self {
        Self {
            max_iters: 64,
            initial_damping: 1e-6,
            residual_tol: 1e-12,
        }
    }
}

/// Spike parameters in the original (un-lifted) coordinates: per-spike amplitude
/// and latent coordinate `t_j ∈ ℝ^d`.
#[derive(Clone, Debug)]
pub struct PolishState {
    /// Amplitudes `a_j`, one per spike.
    pub amplitudes: Vec<f64>,
    /// Latent coordinates `t_j`, `coords[j]` of length `d`.
    pub coords: Vec<Vec<f64>>,
}

/// Outcome of [`polish_spikes`].
#[derive(Clone, Debug)]
pub struct PolishResult {
    /// Polished parameters.
    pub state: PolishState,
    /// Final residual `‖z − Σ_j a_j Φ(t_j)‖₂`.
    pub residual: f64,
    /// Outer iterations actually taken.
    pub iterations: usize,
    /// `true` if the loop stopped on the residual-improvement tolerance or a
    /// damped step could no longer improve the residual (local optimum), `false`
    /// if it exhausted `max_iters`.
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt as _;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn gaussian(rng: &mut StdRng) -> f64 {
        let u1 = rng.random::<f64>().max(1e-16);
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    // ---- Sphere ----------------------------------------------------------

    /// Standard-basis direction `e_i` in dimension `d`.
    fn basis_dir(i: usize, d: usize) -> Vec<f64> {
        let mut v = vec![0.0; d];
        v[i] = 1.0;
        v
    }

    /// Chordal distance between two unit vectors, antipodal-aware.
    fn dir_dist(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        (1.0 - dot.abs()).max(0.0)
    }

    #[test]
    fn sphere_noiseless_roundtrip() {
        // m = 3 orthogonal directions in d = 4, distinct amplitudes.
        let d = 4;
        let planted = vec![
            SphereSpike {
                direction: basis_dir(0, d),
                amplitude: 2.0,
            },
            SphereSpike {
                direction: basis_dir(1, d),
                amplitude: 1.3,
            },
            SphereSpike {
                direction: basis_dir(2, d),
                amplitude: 0.7,
            },
        ];
        let m = sphere_lift(&planted, d).expect("lift");
        let rec = recover_sphere_spikes(m.view(), 0.0).expect("recover");
        assert_eq!(rec.model_order, 3, "order from clean spectrum");
        // Spikes come back sorted by amplitude descending == planted order.
        for (r, p) in rec.spikes.iter().zip(planted.iter()) {
            assert!(dir_dist(&r.direction, &p.direction) < 1e-8, "direction");
            assert!((r.amplitude - p.amplitude).abs() < 1e-8, "amplitude");
        }
        assert!(rec.residual < 1e-8, "residual {:.3e}", rec.residual);
    }

    #[test]
    fn sphere_antipodal_gauge_deterministic() {
        // v and −v produce the SAME lift and MUST canonicalise identically.
        let d = 3;
        let raw = vec![0.6, -0.8, 0.0];
        let neg = vec![-0.6, 0.8, 0.0];
        let m_pos = sphere_lift(
            &[SphereSpike {
                direction: raw.clone(),
                amplitude: 1.5,
            }],
            d,
        )
        .unwrap();
        let m_neg = sphere_lift(
            &[SphereSpike {
                direction: neg,
                amplitude: 1.5,
            }],
            d,
        )
        .unwrap();
        let rec_pos = recover_sphere_spikes(m_pos.view(), 0.0).unwrap();
        let rec_neg = recover_sphere_spikes(m_neg.view(), 0.0).unwrap();
        assert_eq!(rec_pos.spikes.len(), 1);
        assert_eq!(rec_neg.spikes.len(), 1);
        let mut canon = raw.clone();
        canonicalize_direction(&mut canon);
        for k in 0..d {
            assert!((rec_pos.spikes[0].direction[k] - canon[k]).abs() < 1e-10);
            assert!((rec_neg.spikes[0].direction[k] - canon[k]).abs() < 1e-10);
        }
    }

    #[test]
    fn sphere_multiplicity_detection() {
        // Exact m ∈ {1, 2, 3} on orthogonal, well-separated amplitudes.
        let d = 5;
        for m in 1..=3 {
            let amps = [2.5, 1.7, 1.0];
            let planted: Vec<SphereSpike> = (0..m)
                .map(|i| SphereSpike {
                    direction: basis_dir(i, d),
                    amplitude: amps[i],
                })
                .collect();
            let mut lift = sphere_lift(&planted, d).unwrap();
            // Tiny noise well below the amplitude scale.
            let mut rng = StdRng::seed_from_u64(100 + m as u64);
            let sigma = 1e-3;
            for i in 0..d {
                for j in i..d {
                    let e = sigma * gaussian(&mut rng);
                    lift[[i, j]] += e;
                    if i != j {
                        lift[[j, i]] += e;
                    }
                }
            }
            let rec = recover_sphere_spikes(lift.view(), sigma).unwrap();
            assert_eq!(rec.model_order, m, "multiplicity m={m}");
        }
    }

    #[test]
    fn sphere_noise_recovery() {
        // m = 2 orthogonal directions, d = 3, sigma = 0.05. Tolerances are
        // perturbation-theory order: Weyl bounds an eigenvalue shift by the noise
        // operator norm ‖E‖ ≈ 2σ√d ≈ 0.17, and a well-separated eigenvector's
        // first-order tilt is ‖E‖/gap ≈ 0.17/1.5 ≈ 0.11 rad, so the chordal
        // distance 1−|⟨v̂,v⟩| ≈ ‖tilt‖²/2 ≈ 0.006. The asserted bounds sit a few×
        // above these with a safety margin.
        let d = 3;
        let sigma = 0.05;
        let planted = vec![
            SphereSpike {
                direction: basis_dir(0, d),
                amplitude: 3.0,
            },
            SphereSpike {
                direction: basis_dir(1, d),
                amplitude: 1.5,
            },
        ];
        let mut lift = sphere_lift(&planted, d).unwrap();
        let mut rng = StdRng::seed_from_u64(2024);
        for i in 0..d {
            for j in i..d {
                let e = sigma * gaussian(&mut rng);
                lift[[i, j]] += e;
                if i != j {
                    lift[[j, i]] += e;
                }
            }
        }
        let rec = recover_sphere_spikes(lift.view(), sigma).unwrap();
        assert_eq!(rec.model_order, 2, "order under moderate noise");
        for (r, p) in rec.spikes.iter().zip(planted.iter()) {
            assert!(
                dir_dist(&r.direction, &p.direction) < 0.05,
                "direction {:.3e}",
                dir_dist(&r.direction, &p.direction)
            );
            assert!(
                (r.amplitude - p.amplitude).abs() < 0.5,
                "amplitude {:.3e}",
                (r.amplitude - p.amplitude).abs()
            );
        }
    }

    // ---- Torus -----------------------------------------------------------

    fn torus_dist(a: f64, b: f64) -> f64 {
        let d = (a - b).abs();
        d.min(1.0 - d)
    }

    /// Greedy match recovered→planted by nearest (θ,φ); returns max position and
    /// amplitude error. Equal counts required.
    fn torus_match_error(rec: &[TorusSpike], planted: &[TorusSpike]) -> (f64, f64) {
        assert_eq!(rec.len(), planted.len(), "spike-count mismatch");
        let mut max_pos = 0.0_f64;
        let mut max_amp = 0.0_f64;
        let mut used = vec![false; planted.len()];
        for r in rec {
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for (j, p) in planted.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let dd = torus_dist(r.theta, p.theta).max(torus_dist(r.phi, p.phi));
                if dd < best_d {
                    best_d = dd;
                    best = j;
                }
            }
            used[best] = true;
            max_pos = max_pos.max(best_d);
            max_amp = max_amp.max((r.amplitude - planted[best].amplitude).abs());
        }
        (max_pos, max_amp)
    }

    #[test]
    fn torus_noiseless_roundtrip_m1_m2() {
        let (h1, h2) = (6, 6);
        for planted in [
            vec![TorusSpike {
                theta: 0.23,
                phi: 0.61,
                amplitude: 1.4,
            }],
            vec![
                TorusSpike {
                    theta: 0.15,
                    phi: 0.72,
                    amplitude: 1.0,
                },
                TorusSpike {
                    theta: 0.63,
                    phi: 0.28,
                    amplitude: 0.8,
                },
            ],
        ] {
            let grid = torus_lift(&planted, h1, h2).unwrap();
            let rec = recover_torus_spikes(&grid, 0.0).unwrap();
            assert_eq!(rec.model_order, planted.len(), "order");
            let (pos_err, amp_err) = torus_match_error(&rec.spikes, &planted);
            assert!(pos_err < 1e-8, "position error {pos_err:.3e}");
            assert!(amp_err < 1e-8, "amplitude error {amp_err:.3e}");
            assert!(rec.residual < 1e-8, "residual {:.3e}", rec.residual);
        }
    }

    #[test]
    fn torus_multiplicity_detection() {
        let (h1, h2) = (6, 6);
        let candidates = [
            vec![TorusSpike {
                theta: 0.30,
                phi: 0.40,
                amplitude: 1.2,
            }],
            vec![
                TorusSpike {
                    theta: 0.12,
                    phi: 0.70,
                    amplitude: 1.1,
                },
                TorusSpike {
                    theta: 0.55,
                    phi: 0.20,
                    amplitude: 0.9,
                },
            ],
            vec![
                TorusSpike {
                    theta: 0.10,
                    phi: 0.15,
                    amplitude: 1.3,
                },
                TorusSpike {
                    theta: 0.45,
                    phi: 0.62,
                    amplitude: 1.0,
                },
                TorusSpike {
                    theta: 0.80,
                    phi: 0.35,
                    amplitude: 0.8,
                },
            ],
        ];
        for planted in candidates {
            let grid = torus_lift(&planted, h1, h2).unwrap();
            let rec = recover_torus_spikes(&grid, 0.0).unwrap();
            assert_eq!(rec.model_order, planted.len(), "multiplicity");
            let (pos_err, _) = torus_match_error(&rec.spikes, &planted);
            assert!(pos_err < 1e-7, "position error {pos_err:.3e}");
        }
    }

    #[test]
    fn torus_adversarial_pairing() {
        // Adversarial fixture: θ₁ < θ₂ but φ₁ > φ₂. Estimating each axis
        // independently and re-pairing by sorting (θ ascending with φ ascending)
        // yields the GHOST spikes (θ₁,φ₂) and (θ₂,φ₁). The joint-diagonalisation
        // pencil must instead return the TRUE pairing (θ₁,φ₁), (θ₂,φ₂).
        let (h1, h2) = (7, 7);
        let planted = vec![
            TorusSpike {
                theta: 0.20,
                phi: 0.75,
                amplitude: 1.0,
            },
            TorusSpike {
                theta: 0.60,
                phi: 0.25,
                amplitude: 0.9,
            },
        ];
        let grid = torus_lift(&planted, h1, h2).unwrap();
        let rec = recover_torus_spikes(&grid, 0.0).unwrap();
        assert_eq!(rec.model_order, 2, "order");
        let (pos_err, _) = torus_match_error(&rec.spikes, &planted);
        assert!(
            pos_err < 1e-7,
            "correct-pairing position error {pos_err:.3e}"
        );
        // Assert the GHOST pairing is NOT what came back: the ghost set has a
        // spike near (0.20, 0.25), which no true spike is close to.
        let ghost = [
            TorusSpike {
                theta: 0.20,
                phi: 0.25,
                amplitude: 1.0,
            },
            TorusSpike {
                theta: 0.60,
                phi: 0.75,
                amplitude: 0.9,
            },
        ];
        let (ghost_err, _) = torus_match_error(&rec.spikes, &ghost);
        assert!(
            ghost_err > 0.3,
            "recovery must not be the ghost pairing (err {ghost_err:.3e})"
        );
    }

    #[test]
    fn torus_noise_recovery() {
        // Well-separated m = 2, sigma = 0.05. The 2-D pencil frequency error is
        // CRB-order: for an H×H grid the single-tone frequency CRB scales as
        // σ/(a·H^{3/2}) rad, i.e. ≈ 0.05/6^{1.5} ≈ 3e-3 rad ⇒ ≈ 5e-4 of a period;
        // the (non-efficient) pencil sits a small constant above this, so 0.03 of
        // a period is a safe several-× bound.
        let (h1, h2) = (6, 6);
        let sigma = 0.05;
        let planted = vec![
            TorusSpike {
                theta: 0.18,
                phi: 0.66,
                amplitude: 1.2,
            },
            TorusSpike {
                theta: 0.62,
                phi: 0.24,
                amplitude: 1.0,
            },
        ];
        let mut grid = torus_lift(&planted, h1, h2).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for a in 0..h1 {
            for b in 0..h2 {
                let re = grid[(a, b)].re + sigma * gaussian(&mut rng);
                let im = grid[(a, b)].im + sigma * gaussian(&mut rng);
                grid[(a, b)] = c64::new(re, im);
            }
        }
        let rec = recover_torus_spikes(&grid, sigma).unwrap();
        assert_eq!(rec.model_order, 2, "order under noise");
        let (pos_err, amp_err) = torus_match_error(&rec.spikes, &planted);
        assert!(pos_err < 0.03, "position error {pos_err:.3e}");
        assert!(amp_err < 0.3, "amplitude error {amp_err:.3e}");
    }

    // ---- Polish ----------------------------------------------------------

    /// Harmonic-circle basis Φ(t) ∈ ℝ^{2H}: for h=1..H the pair
    /// (cos 2πht, sin 2πht), with Jacobian in the single latent coordinate. This
    /// is exactly the original per-atom objective the circle lift is convex-for.
    fn circle_phi(h_max: usize) -> impl Fn(&[f64]) -> (Array1<f64>, Array2<f64>) {
        move |t: &[f64]| {
            let t0 = t[0];
            let d = 2 * h_max;
            let mut phi = Array1::<f64>::zeros(d);
            let mut jac = Array2::<f64>::zeros((d, 1));
            for h in 1..=h_max {
                let w = TAU * h as f64;
                let (s, c) = (w * t0).sin_cos();
                phi[2 * (h - 1)] = c;
                phi[2 * (h - 1) + 1] = s;
                jac[[2 * (h - 1), 0]] = -w * s;
                jac[[2 * (h - 1) + 1, 0]] = w * c;
            }
            (phi, jac)
        }
    }

    fn circle_code(spikes: &[(f64, f64)], h_max: usize) -> Vec<f64> {
        let d = 2 * h_max;
        let mut z = vec![0.0; d];
        for &(t, a) in spikes {
            for h in 1..=h_max {
                let w = TAU * h as f64;
                z[2 * (h - 1)] += a * (w * t).cos();
                z[2 * (h - 1) + 1] += a * (w * t).sin();
            }
        }
        z
    }

    #[test]
    fn polish_converges_from_perturbed_seed() {
        let h_max = 6;
        let true_spikes = [(0.22, 1.1), (0.61, 0.8)];
        let z = circle_code(&true_spikes, h_max);
        // Seed the polish off the truth (as a noisy descent would land).
        let init = PolishState {
            amplitudes: vec![1.1 + 0.12, 0.8 - 0.1],
            coords: vec![vec![0.22 + 0.02], vec![0.61 - 0.025]],
        };
        let (_, seed_res, _) = {
            // Residual at the seed for a sanity floor.
            let phi = circle_phi(h_max);
            let mut fit = vec![0.0; z.len()];
            for j in 0..init.amplitudes.len() {
                let (p, _) = phi(&init.coords[j]);
                for i in 0..z.len() {
                    fit[i] += init.amplitudes[j] * p[i];
                }
            }
            let nsq: f64 = z.iter().zip(&fit).map(|(a, b)| (a - b) * (a - b)).sum();
            ((), nsq.sqrt(), ())
        };
        let res =
            polish_spikes(&z, circle_phi(h_max), init, &PolishOptions::default()).expect("polish");
        assert!(res.converged, "should converge");
        assert!(res.residual < 1e-6, "residual {:.3e}", res.residual);
        assert!(res.residual < seed_res, "polish must reduce the residual");
    }

    #[test]
    fn polish_rejects_shape_mismatch() {
        let h_max = 4;
        let z = circle_code(&[(0.3, 1.0)], h_max);
        // phi returns the wrong length ⇒ must error, not panic.
        let bad = |_: &[f64]| (Array1::<f64>::zeros(3), Array2::<f64>::zeros((3, 1)));
        let init = PolishState {
            amplitudes: vec![1.0],
            coords: vec![vec![0.3]],
        };
        assert!(polish_spikes(&z, bad, init, &PolishOptions::default()).is_err());
    }
}
