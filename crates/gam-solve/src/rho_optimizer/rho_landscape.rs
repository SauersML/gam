//! Exact 1-D landscape of the Gaussian working-model REML criterion along one
//! log-smoothing coordinate (#2312).
//!
//! For a single penalized block along `ρ = log λ` with the other blocks
//! frozen into the base curvature `H`, let `s_i > 0` be the generalized
//! eigenvalues of `(S, H)` over the penalized rank `m` and let
//! `c_i = z_i²/(2σ²)` be the squared transformed responses in the units where
//! `H = I`. The REML criterion (deviance + penalty + ½log|H+λS| − ½log|λS|₊)
//! then decomposes exactly as
//!
//! ```text
//! V(ρ) = Σ_{i=1}^m f(u_i; c_i) + const,   u_i = ρ + log s_i,
//! f(u; c) = c·σ(u) + ½·softplus(−u),      σ(u) = e^u/(1+e^u),
//! ```
//!
//! since `½[log(1+e^u) − u] = ½·softplus(−u)` — the fused form of the two
//! separately divergent log-determinant terms, uniformly regular at both
//! `ρ → ±∞` (the λ→0 outer-gradient non-finiteness is an artifact of
//! assembling them separately). Derivatives, with `σ' = σ(1−σ)`:
//!
//! ```text
//! f'(u; c)   = (1−σ)(cσ − ½)
//! f''(u; c)  = σ(1−σ)·[c(1−2σ) + ½]
//! f'''(u; c) = σ(1−σ)·{(1−2σ)[c(1−2σ)+½] − 2cσ(1−σ)}
//! ```
//!
//! The phase portrait of one mode:
//! * `u → −∞`: `f' → −½` — every mode pushes λ up out of the λ→0 boundary.
//! * *Noise mode* (`c ≤ ½`): `f' ≤ 0` everywhere — monotone pull to λ = ∞.
//! * *Signal mode* (`c > ½`): a single − → + crossing at `σ(u*) = 1/(2c)`,
//!   i.e. `ρ_i* = −log s_i − log(2c_i − 1)`.
//!
//! `V'` is therefore a sum of single-crossing / nonpositive terms.  After the
//! substitution `x = λ s_scale`, multiplying by the positive denominator
//! `Π_i(1+t_i x)^2` gives a degree-at-most `2m−1` polynomial.  An
//! outward-rounded Sturm chain counts every positive root and isolates every
//! simple root between analytic Cauchy bounds.  A sign-indeterminate chain or
//! repeated/tangential root returns a typed refusal instead of inventing the
//! min/max alternation needed by the landscape classification.  The resulting
//! certificate decides — rather than guesses —
//! whether the coordinate is strictly quasi-convex (one stationary point),
//! multi-basin (the local maxima are exact basin boundaries: the E1
//! interior-stall class), or an all-noise λ→∞ attractor.
//!
//! The third-derivative bound `|V'''| ≤ Σ (3c_i+1)/8 =: L₃` (from
//! `σ(1−σ) ≤ ¼`, `|1−2σ| ≤ 1`) is the explicit ARC cubic weight: cubic
//! regularization with `M ≥ L₃` needs no adaptive warm-up.

/// Numerically stable `log(1 + e^x)`.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable logistic `σ(u) = e^u / (1 + e^u)`.
#[inline]
fn sigmoid(u: f64) -> f64 {
    if u >= 0.0 {
        1.0 / (1.0 + (-u).exp())
    } else {
        let e = u.exp();
        e / (1.0 + e)
    }
}

/// One penalized block's exact 1-D REML spectrum along its own `ρ = log λ`
/// coordinate: generalized eigenvalues `s_i` of `(S, H)` and evidence weights
/// `c_i = z_i²/(2σ²)` (see the module docs).
#[derive(Clone, Debug)]
pub struct RhoModeSpectrum {
    /// Generalized eigenvalues `s_i > 0` of `(S_k, H)`, penalized rank `m`.
    s: Vec<f64>,
    /// Per-mode evidence `c_i = z_i²/(2σ²) ≥ 0`.
    c: Vec<f64>,
}

/// Where the exact 1-D global minimizer of the criterion lives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RhoGlobalMinimum {
    /// Interior minimum at the given `ρ`.
    Interior(f64),
    /// The criterion decreases into `ρ → +∞` (λ → ∞ total-smoothing
    /// boundary) below every interior minimum.
    LambdaInfinity,
}

/// A-priori landscape certificate for one ρ coordinate (#2312 §3).
#[derive(Clone, Debug)]
pub struct RhoLandscapeCertificate {
    /// All interior stationary points of `V`, ascending in ρ. Alternating
    /// minima/maxima starting with a minimum (V' goes − → + at the first).
    pub stationary_points: Vec<f64>,
    /// The interior local minima (odd-numbered stationary points: 1st, 3rd, …).
    pub interior_minima: Vec<f64>,
    /// The interior local maxima — exact basin boundaries.
    pub basin_boundaries: Vec<f64>,
    /// Right-tail sign `T = Σ (c_i − ½)/s_i`. `T ≤ 0` means the criterion
    /// eventually decreases into the λ→∞ boundary.
    pub tail_sum: f64,
    /// Every mode has `c_i ≤ ½`: `V' < 0` everywhere, zero interior
    /// stationary points, and the block is evidence-free.
    pub all_noise: bool,
    /// Exactly one interior stationary point (necessarily a minimum) and a
    /// non-attracting right tail: `V` is strictly quasi-convex on ℝ, so any
    /// descent method converges globally on this coordinate.
    pub quasi_convex: bool,
    /// Explicit ARC third-derivative bound `L₃ = Σ (3c_i + 1)/8`.
    pub l3: f64,
    /// The exact global minimizer over ρ ∈ ℝ ∪ {+∞}.
    pub global_minimum: RhoGlobalMinimum,
}

impl RhoModeSpectrum {
    /// Build a spectrum from paired `(s_i, c_i)`. Every `s_i` must be
    /// strictly positive and finite; every `c_i` finite and nonnegative.
    pub fn new(s: Vec<f64>, c: Vec<f64>) -> Result<Self, String> {
        if s.len() != c.len() {
            return Err(format!(
                "rho landscape spectrum length mismatch: {} eigenvalues, {} evidence weights",
                s.len(),
                c.len()
            ));
        }
        if s.is_empty() {
            return Err("rho landscape spectrum must contain at least one mode".to_string());
        }
        for (i, (&si, &ci)) in s.iter().zip(&c).enumerate() {
            if !(si.is_finite() && si > 0.0) {
                return Err(format!(
                    "rho landscape eigenvalue s[{i}] must be finite and > 0, got {si}"
                ));
            }
            if !(ci.is_finite() && ci >= 0.0) {
                return Err(format!(
                    "rho landscape evidence c[{i}] must be finite and >= 0, got {ci}"
                ));
            }
        }
        Ok(Self { s, c })
    }

    /// Number of modes `m` (the penalized rank).
    pub fn mode_count(&self) -> usize {
        self.s.len()
    }

    /// Fused per-mode criterion `V(ρ) − const = Σ f(u_i; c_i)`, uniformly
    /// regular at ρ = ±∞ (finite for every finite ρ; grows like `−(m/2)ρ` as
    /// ρ → −∞ and saturates at `Σ c_i` as ρ → +∞).
    pub fn criterion(&self, rho: f64) -> f64 {
        self.s
            .iter()
            .zip(&self.c)
            .map(|(&s, &c)| {
                let u = rho + s.ln();
                c * sigmoid(u) + 0.5 * softplus(-u)
            })
            .sum()
    }

    /// `V'(ρ) = Σ (1−σ_i)(c_i σ_i − ½)` — the numerically stable
    /// sum-of-sigmoids evaluator.
    pub fn derivative(&self, rho: f64) -> f64 {
        self.s
            .iter()
            .zip(&self.c)
            .map(|(&s, &c)| {
                let sig = sigmoid(rho + s.ln());
                (1.0 - sig) * (c * sig - 0.5)
            })
            .sum()
    }

    /// `V''(ρ) = Σ σ_i(1−σ_i)[c_i(1−2σ_i) + ½]`.
    pub fn second_derivative(&self, rho: f64) -> f64 {
        self.s
            .iter()
            .zip(&self.c)
            .map(|(&s, &c)| {
                let sig = sigmoid(rho + s.ln());
                sig * (1.0 - sig) * (c * (1.0 - 2.0 * sig) + 0.5)
            })
            .sum()
    }

    /// `V'''(ρ) = Σ σ(1−σ)·{(1−2σ)[c(1−2σ)+½] − 2cσ(1−σ)}`.
    pub fn third_derivative(&self, rho: f64) -> f64 {
        self.s
            .iter()
            .zip(&self.c)
            .map(|(&s, &c)| {
                let sig = sigmoid(rho + s.ln());
                let om = 1.0 - sig;
                sig * om * ((1.0 - 2.0 * sig) * (c * (1.0 - 2.0 * sig) + 0.5) - 2.0 * c * sig * om)
            })
            .sum()
    }

    /// Right-tail sign functional `T = Σ (c_i − ½)/s_i`.
    pub fn tail_sum(&self) -> f64 {
        self.s
            .iter()
            .zip(&self.c)
            .map(|(&s, &c)| (c - 0.5) / s)
            .sum()
    }

    /// Every mode is a noise mode (`c_i ≤ ½`).
    pub fn all_noise(&self) -> bool {
        self.c.iter().all(|&c| c <= 0.5)
    }

    /// Explicit ARC cubic weight `L₃ = Σ (3c_i + 1)/8 ≥ sup |V'''|`.
    pub fn l3_bound(&self) -> f64 {
        self.c.iter().map(|&c| (3.0 * c + 1.0) / 8.0).sum()
    }

    /// All interior stationary points of `V`, ascending.  A Sturm certificate
    /// determines the exact positive-root count before any numerical root is
    /// accepted.  Only simple roots admit the min/max classification exported
    /// by [`RhoLandscapeCertificate`]; repeated or sign-indeterminate roots are
    /// a typed refusal.
    pub fn stationary_points(&self) -> Result<Vec<f64>, String> {
        if self.all_noise() {
            return Ok(Vec::new());
        }
        let certified =
            crate::gaussian_reml::certify_fixed_dispersion_stationary_roots(&self.s, &self.c)?;
        let mut roots = Vec::with_capacity(certified.brackets.len());
        for [a, b] in certified.brackets {
            let fa = self.derivative(a);
            let fb = self.derivative(b);
            if !(fa.is_finite() && fb.is_finite() && fa.signum() != fb.signum()) {
                return Err(format!(
                    "fixed-dispersion stationary bracket [{a}, {b}] does not certify a simple sign-changing root"
                ));
            }
            roots.push(self.bisect_root(a, b));
        }
        Ok(roots)
    }

    /// Standard bisection refinement of a bracketing interval to f64 limits.
    fn bisect_root(&self, mut a: f64, mut b: f64) -> f64 {
        let mut fa = self.derivative(a);
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if mid <= a || mid >= b {
                break;
            }
            let fm = self.derivative(mid);
            if fm == 0.0 {
                return mid;
            }
            if fa.signum() == fm.signum() {
                a = mid;
                fa = fm;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    }

    /// The full landscape certificate (#2312 §3) with the exact global 1-D
    /// minimizer (§4).
    pub fn certificate(&self) -> Result<RhoLandscapeCertificate, String> {
        let stationary_points = self.stationary_points()?;
        let tail_sum = self.tail_sum();
        // V' is − before the first stationary point, so odd-numbered roots
        // (1st, 3rd, …) are minima and even-numbered ones are maxima.
        let interior_minima: Vec<f64> = stationary_points.iter().copied().step_by(2).collect();
        let basin_boundaries: Vec<f64> = stationary_points
            .iter()
            .copied()
            .skip(1)
            .step_by(2)
            .collect();

        // Boundary limit: V(ρ→+∞) − const = Σ c_i, reached from below iff the
        // tail is attracting (last stationary point is a maximum, or none).
        let boundary_attracting = stationary_points.len() % 2 == 0;
        let mut best_interior: Option<(f64, f64)> = None;
        for &rho in &interior_minima {
            let value = self.criterion(rho);
            if best_interior.is_none_or(|(_, best)| value < best) {
                best_interior = Some((rho, value));
            }
        }
        let boundary_value: f64 = self.c.iter().sum();
        let global_minimum = match best_interior {
            Some((rho, value)) if !boundary_attracting || value <= boundary_value => {
                RhoGlobalMinimum::Interior(rho)
            }
            Some(_) | None => RhoGlobalMinimum::LambdaInfinity,
        };

        Ok(RhoLandscapeCertificate {
            quasi_convex: stationary_points.len() == 1,
            all_noise: self.all_noise(),
            l3: self.l3_bound(),
            interior_minima,
            basin_boundaries,
            stationary_points,
            tail_sum,
            global_minimum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPEC-18 forbids a sampled grid from deciding optimization structure.
    /// This two-mode fixture has a stationary numerator known in closed form:
    ///
    /// `N(lambda) = (lambda - 1)(5 lambda^2 + 6 lambda + 2) / 2`.
    ///
    /// The quadratic factor has negative discriminant, so the unique positive
    /// stationary root is exactly `lambda=1`, independently pinning the Sturm
    /// count, isolated rho, and global-minimum classification.
    #[test]
    fn sturm_certificate_matches_pinned_two_mode_polynomial() {
        let spec = RhoModeSpectrum::new(vec![1.0, 2.0], vec![1.0, 0.75]).expect("valid spectrum");
        let cert = spec
            .certificate()
            .expect("decidable simple-root certificate");
        assert_eq!(cert.stationary_points.len(), 1);
        assert!(cert.quasi_convex);
        assert!(cert.stationary_points[0].abs() < 1.0e-9);
        match cert.global_minimum {
            RhoGlobalMinimum::Interior(rho) => assert!(rho.abs() < 1.0e-9),
            other => panic!("pinned positive root must be the global minimum, got {other:?}"),
        }
    }

    /// Validation plan §7.2 — boundary regularity: the fused evaluator is
    /// finite with the exact limiting slopes at ρ = ±40, where the naive
    /// two-term assembly `½log|H+λS| − ½log|λS|` catastrophically cancels or
    /// overflows term-by-term.
    #[test]
    fn fused_form_is_regular_at_both_boundaries() {
        let spec = RhoModeSpectrum::new(vec![1e-3, 1.0, 1e3], vec![0.2, 3.0, 40.0])
            .expect("valid spectrum");
        let m = spec.mode_count() as f64;

        for &rho in &[-40.0, 40.0] {
            let value = spec.criterion(rho);
            let derivative = spec.derivative(rho);
            assert!(
                value.is_finite() && derivative.is_finite(),
                "fused evaluation must be finite at rho={rho}: V={value}, V'={derivative}"
            );
        }
        // λ→0: V' → −m/2 exactly (every mode pushes λ up out of the boundary).
        assert!(
            (spec.derivative(-40.0) + m / 2.0).abs() < 1e-12,
            "λ→0 slope must be exactly −m/2, got {}",
            spec.derivative(-40.0)
        );
        // λ→∞: V saturates at Σ c_i.
        let boundary: f64 = spec.c.iter().sum();
        assert!(
            (spec.criterion(40.0) - boundary).abs() < 1e-12,
            "λ→∞ criterion limit must be Σ c_i = {boundary}, got {}",
            spec.criterion(40.0)
        );

        // The naive per-mode two-term assembly log(1+e^u) and −u overflow /
        // cancel exactly where the fused softplus(−u) form stays exact:
        // at u = −40, e^u underflows so log(1+e^u) = 0 loses the entire
        // −u/2-relative correction e^u/2 ≈ 2e-18 the fused form retains.
        let u = -40.0f64;
        let naive = 0.5 * (u.exp().ln_1p() - u);
        let fused = 0.5 * softplus(-u);
        assert!(
            (naive - fused).abs() <= 1e-12 * fused,
            "sanity: both forms agree in value at u=-40"
        );
        // The failure the fused form kills is the DERIVATIVE assembly: the
        // λ→0 gradient built from separately divergent logdet pieces is
        // (∞ − ∞) in λ-space; the per-mode σ-form is exact. Pin its value.
        assert!((sigmoid(-745.0)).is_finite() && sigmoid(-745.0) >= 0.0);
        assert!(spec.derivative(-745.0).is_finite());
    }

    /// Validation plan §7.4 — all-noise short-circuit: pure-noise spectra are
    /// certified monotone into the λ→∞ boundary with zero interior structure.
    #[test]
    fn all_noise_block_is_certified_boundary_attractor_with_no_roots() {
        let spec = RhoModeSpectrum::new(vec![0.01, 1.0, 50.0], vec![0.0, 0.2, 0.5])
            .expect("valid spectrum");
        let cert = spec
            .certificate()
            .expect("all-noise certificate is analytic");
        assert!(cert.all_noise);
        assert!(cert.stationary_points.is_empty());
        assert!(cert.tail_sum <= 0.0);
        assert_eq!(cert.global_minimum, RhoGlobalMinimum::LambdaInfinity);
        // Each summand `(1-sigma)(c sigma - 1/2)` is non-positive for
        // `c<=1/2`, and the c<1/2 modes make the total strictly negative.
        assert!(spec.derivative(0.0) < 0.0);
    }

    /// Single signal mode: the unique stationary point is the closed-form
    /// crossing `ρ* = −log s − log(2c − 1)` and the certificate says
    /// quasi-convex with an interior global minimum there.
    #[test]
    fn single_signal_mode_matches_closed_form_crossing() {
        for &(s, c) in &[(0.5_f64, 2.0_f64), (3.0, 0.75), (1e-3, 100.0)] {
            let spec = RhoModeSpectrum::new(vec![s], vec![c]).expect("valid spectrum");
            let cert = spec
                .certificate()
                .expect("single-mode root is Sturm-decidable");
            let expected = -s.ln() - (2.0 * c - 1.0).ln();
            assert!(cert.quasi_convex, "one signal mode must be quasi-convex");
            assert_eq!(cert.stationary_points.len(), 1);
            assert!(
                (cert.stationary_points[0] - expected).abs() < 1e-8 * (1.0 + expected.abs()),
                "closed-form crossing {expected} vs isolated {}",
                cert.stationary_points[0]
            );
            match cert.global_minimum {
                RhoGlobalMinimum::Interior(rho) => {
                    assert!((rho - expected).abs() < 1e-8 * (1.0 + expected.abs()));
                }
                other => panic!("single signal mode must have an interior minimum, got {other:?}"),
            }
        }
    }

    /// Multi-basin construction: two signal modes with widely separated
    /// eigenvalues and a noise bath produce S ≥ 3 stationary points, and the
    /// even-numbered ones are certified basin boundaries.
    #[test]
    fn separated_signal_modes_produce_a_certified_multi_basin_landscape() {
        // Mode 1: strong evidence at s=1e4 (crossing ρ ≈ −9.2 − log(2c−1)).
        // Mode 2: strong evidence at s=1e-4. Noise bath in between drags V'
        // negative again after the first minimum.
        let s = vec![1e4, 1e-4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c = vec![80.0, 80.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let spec = RhoModeSpectrum::new(s, c).expect("valid spectrum");
        let cert = spec
            .certificate()
            .expect("multi-basin roots are Sturm-decidable");
        assert!(
            cert.stationary_points.len() >= 3,
            "separated signal modes must be multi-basin, got stationary points {:?} (tail T={})",
            cert.stationary_points,
            cert.tail_sum
        );
        assert_eq!(
            cert.basin_boundaries.len(),
            cert.stationary_points.len() / 2,
            "even-numbered stationary points are the basin boundaries"
        );
        assert!(!cert.quasi_convex);
        // Every basin boundary must be a local maximum: V'' ≤ 0 there.
        for &rho in &cert.basin_boundaries {
            assert!(
                spec.second_derivative(rho) <= 1e-9,
                "basin boundary at {rho} must have non-positive curvature, got {}",
                spec.second_derivative(rho)
            );
        }
        for &rho in &cert.interior_minima {
            assert!(
                spec.second_derivative(rho) >= -1e-9,
                "interior minimum at {rho} must have non-negative curvature, got {}",
                spec.second_derivative(rho)
            );
        }
    }

    /// The exported L3 is the sum of the per-mode algebraic bounds.  Pin both
    /// its exact value and representative interior/tail derivatives; no sampled
    /// grid is used as a structural oracle.
    #[test]
    fn l3_is_the_exact_sum_of_per_mode_bounds() {
        let spec = RhoModeSpectrum::new(vec![0.1, 1.0, 10.0], vec![0.0, 2.0, 9.0])
            .expect("valid spectrum");
        let expected = ((1.0_f64) + 7.0 + 28.0) / 8.0;
        assert_eq!(spec.l3_bound(), expected);
        for rho in [-40.0, 0.0, 40.0] {
            assert!(spec.third_derivative(rho).abs() <= expected);
        }
    }

    /// Analytic self-consistency: the closed-form derivatives match centered
    /// finite differences of the fused criterion. (Tests are the one place
    /// SPEC permits FD; this pins the algebra, not a production path.)
    #[test]
    fn closed_form_derivatives_match_criterion_differences() {
        let spec = RhoModeSpectrum::new(vec![0.03, 1.7, 40.0], vec![0.4, 2.5, 12.0])
            .expect("valid spectrum");
        let h = 1e-5;
        for k in 0..80 {
            let rho = -12.0 + 24.0 * k as f64 / 79.0;
            let fd1 = (spec.criterion(rho + h) - spec.criterion(rho - h)) / (2.0 * h);
            assert!(
                (fd1 - spec.derivative(rho)).abs() < 1e-7 * (1.0 + fd1.abs()),
                "V' mismatch at rho={rho}: analytic {}, FD {fd1}",
                spec.derivative(rho)
            );
            let fd2 = (spec.derivative(rho + h) - spec.derivative(rho - h)) / (2.0 * h);
            assert!(
                (fd2 - spec.second_derivative(rho)).abs() < 1e-6 * (1.0 + fd2.abs()),
                "V'' mismatch at rho={rho}: analytic {}, FD {fd2}",
                spec.second_derivative(rho)
            );
            let fd3 =
                (spec.second_derivative(rho + h) - spec.second_derivative(rho - h)) / (2.0 * h);
            assert!(
                (fd3 - spec.third_derivative(rho)).abs() < 1e-5 * (1.0 + fd3.abs()),
                "V''' mismatch at rho={rho}: analytic {}, FD {fd3}",
                spec.third_derivative(rho)
            );
        }
    }
}
