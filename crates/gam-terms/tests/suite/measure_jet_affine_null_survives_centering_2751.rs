//! #2751 regression gate: every ambient-linear direction must stay in the
//! measure-jet energy's null space **after** the term collection's parametric
//! orthogonalization.
//!
//! The defect this pins: the measure-jet head carried only the LINEAR part of
//! the energy's affine null space. The collection's chokepoint reparameterizes
//! to `Z = null(1ᵀX)`, removing exactly one coefficient direction, and the
//! constrained null space is `{γ : Zγ ∈ null(S)}`. With no constant in
//! `null(S)` the removal is charged to a LINEAR direction, so a `d`-dimensional
//! ambient-linear span collapses to `d − 1`. On the `mjs`-backed BMS fixture
//! that showed up as a fitted plane tilted 45° off the planted one (Pearson
//! 0.705 = |cos 45°|) whenever REML selected a large energy λ — which it does
//! for any near-affine truth.
//!
//! Three layers are asserted, because the bug lived in exactly one of them and
//! the other two are the controls that localize a future regression:
//!
//!   1. **energy** — `AᵀQA ≈ 0` in center-value space (the upstream theorem);
//!   2. **basis** — the emitted Primary's nullity is `1 + head_rank` and every
//!      declared null direction realizes an exactly affine surface;
//!   3. **collection** — after `build_term_collection_design`, the ridge limit
//!      of the constrained Primary still reproduces an arbitrary planted plane.
//!      This is the layer that was broken; 1 and 2 were already correct.
//!
//! Layer 3 is the operational statement: at `λ → ∞` the penalized least-squares
//! fit is the least-squares fit restricted to `null(S)`, so "the plane comes
//! back out" is exactly "the plane is still free".

use gam_data::{ColumnKindTag, DataSchema, EncodedDataset as Dataset, SchemaColumn};
use gam_terms::basis::{
    CenterStrategy, MeasureJetBasisSpec, MeasureJetIdentifiability, build_measure_jet_basis,
    measure_jet_band, measure_jet_energy_form,
};
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::build_term_collection_design;
use gam_terms::term_builder::build_termspec;
use ndarray::{Array1, Array2};

fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Symmetric solve by Cholesky with a relative jitter fallback. Widths here are
/// tens of columns, so a textbook factorization is both exact enough and free
/// of any dependency the basis crate does not already carry.
fn spd_solve(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let p = b.len();
    let scale = (0..p).map(|i| a[[i, i]].abs()).fold(0.0, f64::max).max(1.0);
    let mut jitter = 0.0_f64;
    loop {
        let mut l = vec![vec![0.0f64; p]; p];
        let mut ok = true;
        'fact: for i in 0..p {
            for j in 0..=i {
                let mut s = a[[i, j]] + if i == j { jitter * scale } else { 0.0 };
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 0.0 {
                        ok = false;
                        break 'fact;
                    }
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        if !ok {
            jitter = if jitter == 0.0 { 1e-14 } else { jitter * 100.0 };
            assert!(jitter < 1.0, "penalized normal equations are not solvable");
            continue;
        }
        let mut y = vec![0.0f64; p];
        for i in 0..p {
            let mut s = b[i];
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        let mut x = vec![0.0f64; p];
        for i in (0..p).rev() {
            let mut s = y[i];
            for k in (i + 1)..p {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        return x;
    }
}

/// Ridge-limit fit of `y` on `x` with `lambda * s` added on `s`'s column range:
/// as `lambda → ∞` this is the least-squares fit restricted to `null(s)`.
fn ridge_limit_fit(
    x: &Array2<f64>,
    y: &[f64],
    s: &Array2<f64>,
    range: std::ops::Range<usize>,
    lambda: f64,
) -> Vec<f64> {
    let (n, p) = x.dim();
    let mut xtx = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for j in 0..p {
            let xij = x[[i, j]];
            if xij == 0.0 {
                continue;
            }
            for k in j..p {
                xtx[[j, k]] += xij * x[[i, k]];
            }
        }
    }
    for j in 0..p {
        for k in 0..j {
            xtx[[j, k]] = xtx[[k, j]];
        }
    }
    for (a, ca) in range.clone().enumerate() {
        for (b, cb) in range.clone().enumerate() {
            xtx[[ca, cb]] += lambda * s[[a, b]];
        }
    }
    let xty: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[[i, j]] * y[i]).sum())
        .collect();
    spd_solve(&xtx, &xty)
}

// ---------------------------------------------------------------------------
// Layer 1 — the energy annihilates the affine span in center-value space.
// ---------------------------------------------------------------------------

/// Largest relative entry of `AᵀQA`, `A = [1 | centers]`, normalized by
/// `‖Q‖_F` and the columns' own norms.
fn relative_affine_energy(centers: &Array2<f64>, masses: &Array1<f64>, scales: usize) -> f64 {
    let band = measure_jet_band(centers.view(), scales).expect("band");
    let q = measure_jet_energy_form(centers.view(), masses.view(), &band, 1.0, 1.0)
        .expect("energy form");
    let qnorm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let m = centers.nrows();
    let d = centers.ncols();
    let mut affine = Array2::<f64>::ones((m, d + 1));
    affine.slice_mut(ndarray::s![.., 1..]).assign(centers);
    let gram = affine.t().dot(&q).dot(&affine);
    let scale: Vec<f64> = (0..=d)
        .map(|k| affine.column(k).mapv(|v| v * v).sum().sqrt())
        .collect();
    let mut worst = 0.0_f64;
    for a in 0..=d {
        for b in 0..=d {
            worst = worst.max(gram[[a, b]].abs() / (qnorm * scale[a] * scale[b]));
        }
    }
    worst
}

#[test]
fn measure_jet_energy_annihilates_the_affine_span_exactly() {
    // Four geometries: the cleanest possible layout, a scattered one, a
    // 10x-anisotropic one (where a local affine projection could plausibly lose
    // the weak direction), and a single-scale band.
    let mut grid = Array2::<f64>::zeros((16, 2));
    for i in 0..4 {
        for j in 0..4 {
            grid[[i * 4 + j, 0]] = -1.5 + i as f64;
            grid[[i * 4 + j, 1]] = -1.5 + j as f64;
        }
    }
    let mut scattered = Array2::<f64>::zeros((16, 2));
    for i in 0..16 {
        scattered[[i, 0]] = 3.4 * (hashed_unit(2 * i as u64 + 1) - 0.5);
        scattered[[i, 1]] = 3.4 * (hashed_unit(2 * i as u64 + 2) - 0.5);
    }
    let mut squashed = scattered.clone();
    for i in 0..16 {
        squashed[[i, 1]] *= 0.1;
    }
    let masses = Array1::from_elem(16, 1.0 / 16.0);
    // The energy is a sum of PSD local residual forms whose affine annihilation
    // is exact in exact arithmetic; the realized floor is the roundoff of the
    // per-cell pseudo-inverse and the weighted centering, measured at ~1e-17.
    const AFFINE_ENERGY_TOL: f64 = 1e-12;
    for (label, centers, scales) in [
        ("grid4x4", &grid, 3usize),
        ("scattered16", &scattered, 3),
        ("squashed16", &squashed, 3),
        ("scattered16-1scale", &scattered, 1),
    ] {
        let worst = relative_affine_energy(centers, &masses, scales);
        println!("[2751 energy] {label}: worst relative AᵀQA = {worst:.3e}");
        assert!(
            worst <= AFFINE_ENERGY_TOL,
            "{label}: the measure-jet energy must annihilate the affine span exactly; \
             worst relative AᵀQA entry {worst:.3e} > {AFFINE_ENERGY_TOL:.0e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — the emitted Primary's null space is the WHOLE affine span.
// ---------------------------------------------------------------------------

fn uniform_square(n: usize, d: usize) -> Array2<f64> {
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for k in 0..d {
            data[[i, k]] = 3.46 * (hashed_unit((i * d + k) as u64 + 7) - 0.5);
        }
    }
    data
}

#[test]
fn measure_jet_primary_null_space_is_the_whole_affine_span() {
    for d in [1usize, 2, 3] {
        let data = uniform_square(1200, d);
        let spec = MeasureJetBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 16 },
            double_penalty: false,
            multiscale: false,
            identifiability: MeasureJetIdentifiability::CenterSumToZero,
            ..MeasureJetBasisSpec::default()
        };
        let built = build_measure_jet_basis(data.view(), &spec).expect("build basis");
        let design = built.design.to_dense();
        let primary = &built.active_penalties[0];
        // The head is `[1 | x·T]`, so the declared null frame must be exactly
        // one wider than the supported linear rank.
        let frame = primary
            .info
            .structural_null_frame
            .as_ref()
            .expect("the single-scale Primary declares its structural null frame");
        println!(
            "[2751 basis] d={d} p={} nullity={} frame={} ",
            design.ncols(),
            primary.nullity,
            frame.ncols()
        );
        assert_eq!(
            frame.ncols(),
            d + 1,
            "d={d}: the declared null frame must span the whole affine space \
             (constant + {d} linear directions), not just its linear part"
        );
        assert_eq!(
            primary.nullity,
            d + 1,
            "d={d}: the measured nullity must agree with the declaration"
        );
        // Every declared null direction must realize an exactly affine surface,
        // and together they must span the constant plus every coordinate.
        let mut planted = Array2::<f64>::zeros((data.nrows(), d + 1));
        planted.column_mut(0).fill(1.0);
        planted.slice_mut(ndarray::s![.., 1..]).assign(&data);
        for c in 0..frame.ncols() {
            let f = design.dot(&frame.column(c).to_owned());
            let (_, resid) = affine_residual(&planted, &f);
            assert!(
                resid <= 1e-10,
                "d={d}: declared null direction {c} is not an affine surface \
                 (relative non-affine residual {resid:.3e})"
            );
        }
        // The reverse inclusion: each coordinate plane must be reachable inside
        // the declared frame, i.e. the frame's realized surfaces span `{1, x_k}`.
        let realized = design.dot(frame);
        for k in 0..d {
            let target: Vec<f64> = (0..data.nrows()).map(|i| data[[i, k]]).collect();
            let (_, resid) = affine_residual(&realized, &Array1::from(target));
            assert!(
                resid <= 1e-10,
                "d={d}: coordinate {k}'s own plane is not inside the Primary's null space \
                 (relative residual {resid:.3e})"
            );
        }
    }
}

/// Relative least-squares residual of `f` on the columns of `basis`.
fn affine_residual(basis: &Array2<f64>, f: &Array1<f64>) -> (Vec<f64>, f64) {
    let (n, p) = basis.dim();
    let mut gram = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        for k in j..p {
            let v: f64 = (0..n).map(|i| basis[[i, j]] * basis[[i, k]]).sum();
            gram[[j, k]] = v;
            gram[[k, j]] = v;
        }
    }
    // Scale-free jitter keeps a rank-deficient frame solvable; it is far below
    // the 1e-10 residual bound the callers assert.
    let trace = (0..p).map(|j| gram[[j, j]]).sum::<f64>().max(1.0);
    for j in 0..p {
        gram[[j, j]] += 1e-14 * trace;
    }
    let rhs: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| basis[[i, j]] * f[i]).sum())
        .collect();
    let coef = spd_solve(&gram, &rhs);
    let mut sse = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        let pred: f64 = (0..p).map(|j| basis[[i, j]] * coef[j]).sum();
        sse += (f[i] - pred) * (f[i] - pred);
        sst += f[i] * f[i];
    }
    (coef, (sse / sst.max(f64::MIN_POSITIVE)).sqrt())
}

// ---------------------------------------------------------------------------
// Layer 3 — the null space survives the collection's centering. THE gate.
// ---------------------------------------------------------------------------

fn dataset(n: usize, d: usize) -> Dataset {
    let mut headers = vec!["y".to_string()];
    for k in 0..d {
        headers.push(format!("x{k}"));
    }
    let mut values = Array2::<f64>::zeros((n, d + 1));
    for i in 0..n {
        for k in 0..d {
            values[[i, k + 1]] = hashed_unit((i * (d + 1) + k) as u64 + 11);
        }
        values[[i, 0]] = values[[i, 1]];
    }
    Dataset {
        headers: headers.clone(),
        values,
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.clone(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; d + 1],
    }
}

/// At `λ → ∞` the fit is least squares restricted to the constrained Primary's
/// null space. A planted plane along ANY ambient coordinate must come back out
/// of it — that is what "the linear span is still free" means operationally,
/// and it is the property the fixture's Pearson bar reads at second hand.
#[test]
fn collection_centering_leaves_every_linear_direction_free() {
    for d in [2usize, 3] {
        let n = 800;
        let ds = dataset(n, d);
        let col_map = ds.column_map();
        let cols: Vec<String> = (0..d).map(|k| format!("x{k}")).collect();
        let formula = format!("y ~ mjs({}, centers=16)", cols.join(", "));
        let parsed = parse_formula(&formula).expect("parse");
        let spec = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut Vec::new(),
        )
        .expect("term spec");
        let feature = ds.values.clone();
        let design = build_term_collection_design(feature.view(), &spec).expect("design");
        let dense = design.design.to_dense();
        let primary = design
            .penalties
            .iter()
            .zip(&design.penaltyinfo)
            .find(|(_, info)| {
                matches!(
                    info.penalty.source,
                    gam_terms::basis::PenaltySource::Primary
                )
            })
            .map(|(pen, _)| pen)
            .expect("the collection keeps a Primary energy for the mjs term");

        // Plant a plane along each coordinate in turn. Each must be recovered
        // in the ridge limit; before #2751 only the single accidental direction
        // with zero data-mean survived, and a planted `x_0` plane came back
        // tilted 45 degrees into `x_1`.
        for k in 0..d {
            let truth: Vec<f64> = (0..n).map(|i| 0.2 + 0.9 * feature[[i, k + 1]]).collect();
            let beta = ridge_limit_fit(
                &dense,
                &truth,
                &primary.local,
                primary.col_range.clone(),
                1.0e12,
            );
            let fitted: Vec<f64> = (0..n)
                .map(|i| (0..dense.ncols()).map(|j| dense[[i, j]] * beta[j]).sum())
                .collect();
            // Read the recovered plane back as slopes on the raw coordinates.
            let mut planted = Array2::<f64>::ones((n, d + 1));
            for kk in 0..d {
                for i in 0..n {
                    planted[[i, kk + 1]] = feature[[i, kk + 1]];
                }
            }
            let (coef, resid) = affine_residual(&planted, &Array1::from(fitted));
            println!(
                "[2751 collection] d={d} planted x{k}: slopes={:?} nonaffine_resid={resid:.3e}",
                coef[1..].iter().map(|c| (c * 1e4).round() / 1e4).collect::<Vec<_>>()
            );
            // Numerically the ridge limit is a 1e12-weighted projection, so a
            // few 1e-3 of slack is the arithmetic, not the geometry; the defect
            // this pins moved the slopes by 0.5 (0.9 -> 0.43) and put 0.45 of
            // slope on a coordinate whose true slope is zero.
            assert!(
                (coef[k + 1] - 0.9).abs() <= 2.0e-3,
                "d={d}: the planted x{k} slope must survive the collection's centering \
                 inside the Primary's null space; recovered {:.4} (want 0.9). \
                 all slopes {:?}",
                coef[k + 1],
                coef[1..].to_vec()
            );
            for kk in 0..d {
                if kk == k {
                    continue;
                }
                assert!(
                    coef[kk + 1].abs() <= 2.0e-3,
                    "d={d}: planting an x{k} plane must not tilt the recovered surface along \
                     x{kk}; got slope {:.4}. This is the #2751 signature: the centering \
                     deleted a linear direction from the energy's null space and the fit \
                     projected onto what was left",
                    coef[kk + 1]
                );
            }
        }
    }
}
