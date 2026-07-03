//! Encoder-half PROOF (#2021, wheel-independent, proof-only): the co-atom-decoder
//! precision `Σ⁻¹ = (σ²·Σ_k a_k B_k B_kᵀ + λ I)⁻¹` makes the ONE-SHOT GLS read of an
//! overlapping curved factor's angle COHERENCE-INDEPENDENT, whereas the naive
//! (identity-metric) read degrades as the frames overlap. This is the Rust mirror
//! of `direct_mu_test.py`, and it is EXACTLY where the `(BᵀΣ⁻¹B)⁻¹` normalization
//! is load-bearing (the single-shot linear read, not the iterated fitter fixed
//! point — see the `gam-whitened-encode-detangles-coherence` note).
//!
//! Scope discipline (team-lead guardrails): PROOF-ONLY. Planted decoders, no
//! fitter, and the reusable [`coatom_precision_apply`] constructor is NOT wired
//! into any fit install path here — the end-to-end install stays held behind the
//! healthy multi-atom fit. The constructor is built in the SAME Woodbury form the
//! eventual install will consume (`Σ⁻¹ = (1/λ)(I − C(λI + CᵀC)⁻¹Cᵀ)`, C stacking
//! the gate-weighted co-atom frames), so proof and install share one path.
//!
//! KILL SIGNAL: if the whitened read is NOT coherence-independent even with ideal
//! planted decoders, the co-atom-Σ form is wrong and the assertion says so loudly
//! — a loud negative here is as valuable as a pass.

use ndarray::{Array1, Array2};

// ---- deterministic RNG (Box–Muller over an LCG), reproducible bit-for-bit ----
fn lcg_uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg_uniform(s).max(1e-12);
    let u2 = lcg_uniform(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Partial-pivot Gaussian-elimination solve `A y = b` for small dense SPD `A`
/// (the D×D Woodbury capacitance and the 2×2 GLS normal matrix). Robust and
/// dependency-free at the sizes this proof uses.
fn solve_dense(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = a.nrows();
    let mut m = a.clone();
    let mut x = b.clone();
    for col in 0..n {
        // Pivot.
        let mut piv = col;
        let mut best = m[[col, col]].abs();
        for r in (col + 1)..n {
            let v = m[[r, col]].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != col {
            for c in 0..n {
                m.swap([col, c], [piv, c]);
            }
            x.swap(col, piv);
        }
        let d = m[[col, col]];
        for r in (col + 1)..n {
            let f = m[[r, col]] / d;
            if f != 0.0 {
                for c in col..n {
                    m[[r, c]] -= f * m[[col, c]];
                }
                x[r] -= f * x[col];
            }
        }
    }
    // Back-substitution.
    let mut y = Array1::<f64>::zeros(n);
    for row in (0..n).rev() {
        let mut acc = x[row];
        for c in (row + 1)..n {
            acc -= m[[row, c]] * y[c];
        }
        y[row] = acc / m[[row, row]];
    }
    y
}

/// REUSABLE co-atom precision apply — `Σ⁻¹ v` for `Σ = C Cᵀ + λ I` via Woodbury
/// (`Σ⁻¹ = (1/λ)(I − C (λ I_D + CᵀC)⁻¹ Cᵀ)`), where `C ∈ ℝ^{p×D}` stacks the
/// gate-weighted co-atom frames (column block `√(σ² a_k) F_k`). Never forms a p×p
/// inverse; the only solve is the D×D capacitance. Always PD for λ>0 (this is the
/// #2080/#1784-safe FULL inverse, NOT the linear `I − s·P` that goes non-PD for
/// K>2 co-active frames).
fn coatom_precision_apply(c: &Array2<f64>, lambda: f64, v: &Array1<f64>) -> Array1<f64> {
    let p = c.nrows();
    let d = c.ncols();
    // w = Cᵀ v  (D).
    let mut w = Array1::<f64>::zeros(d);
    for k in 0..d {
        let mut acc = 0.0;
        for i in 0..p {
            acc += c[[i, k]] * v[i];
        }
        w[k] = acc;
    }
    // cap = λ I_D + Cᵀ C  (D×D), SPD.
    let mut cap = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in 0..d {
            let mut acc = 0.0;
            for i in 0..p {
                acc += c[[i, a]] * c[[i, b]];
            }
            cap[[a, b]] = acc;
        }
        cap[[a, a]] += lambda;
    }
    // y = cap⁻¹ w.
    let y = solve_dense(&cap, &w);
    // Σ⁻¹ v = (1/λ)(v − C y).
    let mut out = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut cy = 0.0;
        for k in 0..d {
            cy += c[[i, k]] * y[k];
        }
        out[i] = (v[i] - cy) / lambda;
    }
    out
}

/// One-shot read of a curved factor's 2D coordinate under a metric `apply`
/// (`apply(v) = M v`): `z = (Bᵀ M B)⁻¹ Bᵀ M x`. `frame` is `p×2` (the two harmonic
/// output directions). Returns the recovered angle `atan2(z1, z0)`.
fn one_shot_angle<F: Fn(&Array1<f64>) -> Array1<f64>>(
    frame: &Array2<f64>,
    x: &Array1<f64>,
    apply: &F,
) -> f64 {
    let p = frame.nrows();
    // M·(each column of B) and M·x.
    let col0: Array1<f64> = frame.column(0).to_owned();
    let col1: Array1<f64> = frame.column(1).to_owned();
    let mb0 = apply(&col0);
    let mb1 = apply(&col1);
    let mx = apply(x);
    // G = Bᵀ M B (2×2), h = Bᵀ M x (2).
    let dot = |u: &Array1<f64>, w: &Array1<f64>| -> f64 {
        (0..p).map(|i| u[i] * w[i]).sum::<f64>()
    };
    let mut g = Array2::<f64>::zeros((2, 2));
    g[[0, 0]] = dot(&col0, &mb0);
    g[[0, 1]] = dot(&col0, &mb1);
    g[[1, 0]] = dot(&col1, &mb0);
    g[[1, 1]] = dot(&col1, &mb1);
    let h = Array1::from_vec(vec![dot(&col0, &mx), dot(&col1, &mx)]);
    let z = solve_dense(&g, &h);
    z[1].atan2(z[0])
}

fn wrap_pi(a: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut x = a.rem_euclid(two_pi);
    if x > std::f64::consts::PI {
        x -= two_pi;
    }
    x
}

/// Gauge-aligned circular RMSE (radians) between estimated and planted angles:
/// the decoder frame fixes the angle only up to a reflection (±1) and a global
/// phase, so score the best-aligned residual over both reflections.
fn gauge_aligned_circular_rmse(est: &[f64], truth: &[f64]) -> f64 {
    let mut best = f64::INFINITY;
    for &sign in &[1.0_f64, -1.0] {
        // Best global phase = circular mean of (est − sign·truth).
        let (mut cs, mut sn) = (0.0, 0.0);
        for i in 0..est.len() {
            let r = est[i] - sign * truth[i];
            cs += r.cos();
            sn += r.sin();
        }
        let phase = sn.atan2(cs);
        let mut sse = 0.0;
        for i in 0..est.len() {
            let e = wrap_pi(est[i] - sign * truth[i] - phase);
            sse += e * e;
        }
        let rmse = (sse / est.len() as f64).sqrt();
        if rmse < best {
            best = rmse;
        }
    }
    best
}

/// LOAD-BEARING encoder-half proof: sweep the frame-coherence dial μ; the whitened
/// (co-atom Σ⁻¹) one-shot read of circle B's angle must stay coherence-INDEPENDENT
/// while the naive read degrades. Mirrors direct_mu_test.py.
#[test]
fn coatom_precision_read_is_coherence_independent_2021() {
    let p = 48usize;
    let n = 320usize;
    let sigma2 = 1.0_f64; // circle A signal variance
    let lambda = 0.03_f64 * 0.03; // idiosyncratic noise variance ⇒ strong A-subspace down-weight
    let noise = 0.03_f64;

    // Circle A occupies output dirs (e0,e1). Circle B's plane overlaps A's by the
    // principal cosine μ: b0 = μ·e0 + √(1−μ²)·e2, b1 = e3.
    let unit = |dirs: &[(usize, f64)]| -> Array1<f64> {
        let mut v = Array1::<f64>::zeros(p);
        for &(i, w) in dirs {
            v[i] = w;
        }
        v
    };
    let a0 = unit(&[(0, 1.0)]);
    let a1 = unit(&[(1, 1.0)]);
    let mut b_a = Array2::<f64>::zeros((p, 2));
    b_a.column_mut(0).assign(&a0);
    b_a.column_mut(1).assign(&a1);

    let mus = [0.0_f64, 0.5, 0.8, 0.95];
    let mut whitened_err = Vec::new();
    let mut naive_err = Vec::new();

    for &mu in &mus {
        // B's frame at coherence μ.
        let b0 = unit(&[(0, mu), (2, (1.0 - mu * mu).sqrt())]);
        let b1 = unit(&[(3, 1.0)]);
        let mut b_b = Array2::<f64>::zeros((p, 2));
        b_b.column_mut(0).assign(&b0);
        b_b.column_mut(1).assign(&b1);

        // Co-atom precision Σ⁻¹ = (σ²·B_A B_Aᵀ + λ I)⁻¹ via the reusable constructor:
        // C = √(σ²·a)·B_A with A fully active (a=1).
        let c = b_a.mapv(|v| v * (sigma2 * 1.0).sqrt());
        let whitened_apply = |v: &Array1<f64>| coatom_precision_apply(&c, lambda, v);
        let naive_apply = |v: &Array1<f64>| v.clone();

        // Data: x = A-image(θ_A) + B-image(θ_B) + noise, read B's angle both ways.
        let mut seed = 0x2021_C0A7_5164_0000u64 ^ ((mu * 1e6) as u64);
        let (mut est_w, mut est_n, mut truth) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let th_a = std::f64::consts::TAU * lcg_uniform(&mut seed);
            let th_b = std::f64::consts::TAU * lcg_uniform(&mut seed);
            let mut x = Array1::<f64>::zeros(p);
            for i in 0..p {
                x[i] = th_a.cos() * a0[i]
                    + th_a.sin() * a1[i]
                    + th_b.cos() * b0[i]
                    + th_b.sin() * b1[i]
                    + noise * lcg_normal(&mut seed);
            }
            est_w.push(one_shot_angle(&b_b, &x, &whitened_apply));
            est_n.push(one_shot_angle(&b_b, &x, &naive_apply));
            truth.push(th_b);
        }
        whitened_err.push(gauge_aligned_circular_rmse(&est_w, &truth));
        naive_err.push(gauge_aligned_circular_rmse(&est_n, &truth));
    }

    let report = format!(
        "μ={:?}\n  whitened circ-rmse = {:?}\n  naive    circ-rmse = {:?}",
        mus, whitened_err, naive_err
    );

    eprintln!("COATOM_REPORT\n{report}");

    // (1) KILL SIGNAL — the whitened read MUST stay coherence-independent even at
    // extreme overlap. If this fails, the co-atom-Σ form does NOT de-tangle and the
    // whole encoder-half approach is wrong — surface it loudly.
    let w_hi = *whitened_err.last().unwrap();
    let w_lo = whitened_err[0];
    assert!(
        w_hi < 0.15,
        "KILL SIGNAL: co-atom Σ⁻¹ read is NOT coherence-independent at μ=0.95 \
         (circ-rmse {w_hi:.3} rad ≥ 0.15) — the co-atom-Σ form fails to de-tangle.\n{report}"
    );
    assert!(
        (w_hi - w_lo).abs() < 0.12,
        "KILL SIGNAL: whitened error is not FLAT across coherence \
         (Δ={:.3} from μ=0 to μ=0.95) — de-tangling is coherence-dependent.\n{report}",
        (w_hi - w_lo).abs()
    );

    // (2) The naive read MUST degrade with coherence (else the fixture doesn't
    // exercise the de-tangling and the proof is vacuous).
    let n_hi = *naive_err.last().unwrap();
    assert!(
        n_hi > 0.3,
        "fixture sanity: the NAIVE read must degrade under overlap (μ=0.95 \
         circ-rmse {n_hi:.3} rad ≤ 0.3), else the test cannot demonstrate de-tangling.\n{report}"
    );

    // (3) At high coherence the whitened read must beat naive by a wide margin —
    // the actual de-tangling win.
    assert!(
        w_hi < n_hi * 0.5,
        "at μ=0.95 the whitened read ({w_hi:.3}) must be far better than naive \
         ({n_hi:.3}) — the (BᵀΣ⁻¹B)⁻¹ normalization un-clips the attenuated direction.\n{report}"
    );
}
