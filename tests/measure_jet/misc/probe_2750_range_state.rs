//! #2750 probe — the state of the measure-jet representer range at current
//! `main`, on the byte-identical rows this issue's two fixtures use.
//!
//! Diagnostic-only: it prints a table and asserts nothing about the LEVEL of
//! any accuracy number. The only assertions are non-vacuity ones (every arm
//! produced a finite number), because a probe that can print `-` and still go
//! green is indistinguishable from a probe that never ran.
//!
//! ## What it separates
//!
//! `ℓ` is the ONE design-moving coordinate of a measure-jet term: it decides
//! WHICH span the representers occupy, and `λ` shrinks inside a span and can
//! never move one. So for each fixture the table reports, side by side:
//!
//! * **`ell`** — the realized representer range the fit ended at, in the
//!   standardized frame the basis is built in, next to the auto GEOMETRY
//!   heuristic (`median nearest-node spacing`) the pre-#2750 seed used. The
//!   gap between them is how far the response screen moved the seed.
//! * **`span`** — the least-squares projection residual of the NOISELESS truth
//!   onto the realized design's own column span on the held-out grid. This is
//!   the floor no `λ` can beat, so it adjudicates "the design cannot represent
//!   this" against "the criterion mis-allocated the smoothing".
//! * **`rmse`** — the fixture's own held-out reconstruction error.
//!
//! and repeats `span`/`rmse` for `s(x)` on the identical rows, which is the
//! comparator that made this issue's central measurement (`54×` at comparable
//! `edf`) decisive.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::{SmoothBasisSpec, build_term_collection_design};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// SplitMix64 finalizer mapped to `[0, 1)`, byte-identical to the one both
/// fixtures use, so the datasets below are the same rows they fit.
fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// One fixture row: the generator knobs of a `#2750` case, in the exact shape
/// its owning fixture writes them.
struct Fixture {
    name: &'static str,
    seed: u64,
    n: usize,
    freq: f64,
    phase: f64,
    noise: f64,
    jitter: bool,
    /// `true` for the non-convergence fixture, whose jitter key and noise index
    /// are seedless (`i ^ 0xABCD`, `i·2654435761`) rather than seed-mixed.
    seedless_keys: bool,
}

impl Fixture {
    fn signal(&self, x: f64) -> f64 {
        (std::f64::consts::TAU * self.freq * x + self.phase).sin()
    }

    fn xs(&self) -> Vec<f64> {
        let mut xs: Vec<f64> = (0..self.n)
            .map(|i| {
                let base = i as f64 / (self.n as f64 - 1.0);
                if self.jitter {
                    let key = if self.seedless_keys {
                        i as u64 ^ 0xABCD
                    } else {
                        i as u64 ^ self.seed.wrapping_mul(0x1234_5)
                    };
                    (base + (hashed_unit(key) - 0.5) / (self.n as f64 - 1.0)).clamp(0.0, 1.0)
                } else {
                    base
                }
            })
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).expect("finite grid"));
        xs
    }

    fn dataset(&self) -> gam::data::EncodedDataset {
        let headers = ["x", "y"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let rows = self
            .xs()
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let key = if self.seedless_keys {
                    (i as u64).wrapping_mul(2_654_435_761)
                } else {
                    (i as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(self.seed.wrapping_mul(0x9E37_79B9))
                };
                let y = self.signal(x) + self.noise * (2.0 * hashed_unit(key) - 1.0);
                StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
            })
            .collect::<Vec<_>>();
        encode_recordswith_inferred_schema(headers, rows).expect("encode probe dataset")
    }

    /// The fixture's own held-out readout grid.
    fn grid(&self) -> Vec<f64> {
        let count = if self.seedless_keys { 400 } else { 300 };
        let span = if self.seedless_keys { 0.996 } else { 0.994 };
        let start = if self.seedless_keys { 0.002 } else { 0.003 };
        (0..count)
            .map(|i| start + span * i as f64 / (count as f64 - 1.0))
            .collect()
    }
}

fn frame(points: &[f64]) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((points.len(), 2));
    for (i, &t) in points.iter().enumerate() {
        m[[i, 0]] = t;
    }
    m
}

/// Materialize the rebuilt design at `points` as a dense matrix by applying the
/// operator to each unit coefficient vector — the same route the fixtures use
/// to score their predictions, so the columns are the realized ones.
fn dense_design(fit: &gam::StandardFitResult, points: &[f64]) -> Array2<f64> {
    let built = build_term_collection_design(frame(points).view(), &fit.resolvedspec)
        .expect("rebuild design on the probe grid");
    let op = &built.design;
    let (n, p) = (op.nrows(), op.ncols());
    let mut dense = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let mut e = Array1::<f64>::zeros(p);
        e[j] = 1.0;
        dense.column_mut(j).assign(&op.apply(&e));
    }
    dense
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let sse: f64 = a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum();
    (sse / a.len() as f64).sqrt()
}

/// Residual RMSE of the least-squares projection of `y` onto the column span of
/// `x`, by twice-reorthogonalized modified Gram-Schmidt with a relative rank
/// floor. This is the span FLOOR: no penalty can reach below it, because a
/// penalty shrinks within a span and never moves one.
fn span_floor(x: &Array2<f64>, y: &[f64]) -> (f64, usize) {
    let p = x.ncols();
    let scale = (0..p)
        .map(|j| x.column(j).dot(&x.column(j)).sqrt())
        .fold(0.0_f64, f64::max);
    let floor = 1.0e-10 * scale.max(1.0);
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for j in 0..p {
        let mut v = x.column(j).to_owned();
        for _ in 0..2 {
            for q in basis.iter() {
                let c = q.dot(&v);
                v.scaled_add(-c, q);
            }
        }
        let norm = v.dot(&v).sqrt();
        if norm > floor {
            v.mapv_inplace(|z| z / norm);
            basis.push(v);
        }
    }
    let yv = Array1::from_vec(y.to_vec());
    let mut resid = yv.clone();
    for q in basis.iter() {
        let c = q.dot(&yv);
        resid.scaled_add(-c, q);
    }
    let zeros = vec![0.0; resid.len()];
    (
        rmse(resid.as_slice().expect("contiguous residual"), &zeros),
        basis.len(),
    )
}

/// What one arm measured on one fixture.
struct Arm {
    edf: f64,
    columns: usize,
    rank: usize,
    rmse: f64,
    span: f64,
    /// The realized measure-jet representer range in the frame the basis is
    /// built in, or `None` for a comparator basis that has no such coordinate.
    ell: Option<f64>,
}

fn fit_arm(body: &str, fixture: &Fixture, data: &gam::data::EncodedDataset) -> Option<Arm> {
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let fitted = match fit_from_formula(&format!("y ~ {body}"), data, &config) {
        Ok(FitResult::Standard(fit)) => fit,
        Ok(_) => return None,
        Err(error) => {
            println!("[2750-state] {} {body}: REFUSED: {error}", fixture.name);
            return None;
        }
    };
    let grid = fixture.grid();
    let truth: Vec<f64> = grid.iter().map(|&t| fixture.signal(t)).collect();
    let design = dense_design(&fitted, &grid);
    let predicted: Vec<f64> = design.dot(&fitted.fit.beta).to_vec();
    let (span, rank) = span_floor(&design, &truth);
    let ell = fitted.resolvedspec.smooth_terms.first().and_then(|term| {
        if let SmoothBasisSpec::MeasureJet { spec, .. } = &term.basis {
            Some(spec.length_scale)
        } else {
            None
        }
    });
    Some(Arm {
        edf: fitted.fit.edf_total().unwrap_or(f64::NAN),
        columns: design.ncols(),
        rank,
        rmse: rmse(&predicted, &truth),
        span,
        ell,
    })
}

#[test]
fn measure_jet_range_state_on_the_2750_fixtures() {
    init_parallelism();
    let fixtures = [
        Fixture {
            name: "sweep/1",
            seed: 1,
            n: 200,
            freq: 1.0,
            phase: 0.0,
            noise: 0.10,
            jitter: false,
            seedless_keys: false,
        },
        Fixture {
            name: "sweep/2",
            seed: 2,
            n: 200,
            freq: 1.0,
            phase: 1.3,
            noise: 0.05,
            jitter: true,
            seedless_keys: false,
        },
        Fixture {
            name: "sweep/3",
            seed: 3,
            n: 240,
            freq: 1.5,
            phase: 0.7,
            noise: 0.08,
            jitter: false,
            seedless_keys: false,
        },
        Fixture {
            name: "sweep/4",
            seed: 4,
            n: 180,
            freq: 2.0,
            phase: 2.1,
            noise: 0.10,
            jitter: true,
            seedless_keys: false,
        },
        Fixture {
            name: "nonconv",
            seed: 0,
            n: 220,
            freq: 1.0,
            phase: 0.7,
            noise: 0.08,
            jitter: true,
            seedless_keys: true,
        },
    ];

    println!(
        "[2750-state] {:>8} {:>18} {:>10} {:>7} {:>5} {:>5} {:>11} {:>11}",
        "fixture", "arm", "ell", "edf", "p", "rank", "span_floor", "rmse"
    );
    let mut measured = 0usize;
    for fixture in fixtures.iter() {
        let data = fixture.dataset();
        for body in ["s(x, bs=\"mjs\")", "s(x)", "s(x, bs=\"tps\")"] {
            let Some(arm) = fit_arm(body, fixture, &data) else {
                continue;
            };
            let ell = arm
                .ell
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "-".to_string());
            println!(
                "[2750-state] {:>8} {:>18} {:>10} {:>7.3} {:>5} {:>5} {:>11.6} {:>11.6}",
                fixture.name, body, ell, arm.edf, arm.columns, arm.rank, arm.span, arm.rmse
            );
            assert!(
                arm.rmse.is_finite() && arm.span.is_finite() && arm.edf.is_finite(),
                "{}: {body} produced a non-finite measurement",
                fixture.name
            );
            measured += 1;
        }
    }
    assert_eq!(
        measured,
        3 * fixtures.len(),
        "every arm on every fixture must produce a measurement; a probe that can \
         silently drop arms cannot be read on the run that matters"
    );
}
