//! #3684 probe: the K=1 Euclidean-patch arc's outer criterion near the cliff.

use super::*;
use crate::assignment::{AssignmentMode, SaeAssignment};
use gam_solve::rho_optimizer::OuterObjective;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

struct PricedLineLogger;

impl log::Log for PricedLineLogger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        let line = record.args().to_string();
        if line.starts_with("[SAE-EXACT-DENSE] priced")
            || line.contains("saddle")
            || line.contains("refus")
        {
            eprintln!("PROBE3684-LOG {line}");
        }
    }
    fn flush(&self) {}
}

static LOGGER: PricedLineLogger = PricedLineLogger;

fn planted_arc_seed_term(n: usize, p: usize, sigma: f64) -> (SaeManifoldTerm, Array2<f64>) {
    use super::tests::deterministic_circle_noise;
    use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| {
        -1.0 + 2.0 * (row as f64) / (n as f64 - 1.0)
    });
    let target = Array2::<f64>::from_shape_fn((n, p), |(row, col)| {
        let t = coords[[row, 0]];
        deterministic_circle_noise(col, 0) * t
            + deterministic_circle_noise(col, 1) * t * t
            + sigma * deterministic_circle_noise(row, col)
    });
    let evaluator = Arc::new(
        crate::basis::EuclideanPatchEvaluator::new(1, 2)
            .expect("dim 1, degree 2 is a valid Euclidean patch"),
    );
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the planted coordinates lie in the Euclidean patch domain");
    let decoder = fast_ata(&phi)
        .cholesky(faer::Side::Lower)
        .expect("the planted arc's patch Gram is positive definite")
        .solve_mat(&fast_atb(&phi, &target));
    let m = phi.ncols();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "arc",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("phi, jet, decoder and gram were built with matching shapes")
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("one logit column, coordinate block and manifold");
    (
        SaeManifoldTerm::new(vec![atom], assignment).expect("one atom over the same rows"),
        target,
    )
}

fn objective() -> SaeManifoldOuterObjective {
    let (mut term, target) = planted_arc_seed_term(256, 4, 0.02);
    term.gpu_policy = gam_gpu::GpuPolicy::Off;
    let seed_rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, seed_rho, 40, 1.0, 1.0e-6, 1.0e-6)
}

fn report(tag: &str, obj: &mut SaeManifoldOuterObjective, rho: &Array1<f64>) {
    eprintln!("PROBE3684 {tag} rho=[{:.7}, {:.7}] BEGIN", rho[0], rho[1]);
    match obj.eval(rho) {
        Ok(eval) => {
            let g: Vec<String> = eval.gradient.iter().map(|v| format!("{v:.6e}")).collect();
            eprintln!(
                "PROBE3684 {tag} rho=[{:.7}, {:.7}] cost={:.10e} g=[{}]",
                rho[0],
                rho[1],
                eval.cost,
                g.join(", ")
            );
        }
        Err(err) => eprintln!("PROBE3684 {tag} rho=[{:.7}, {:.7}] ERR {err}", rho[0], rho[1]),
    }
}

#[test]
fn probe_3684_arc_outer_cliff_directions() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Debug);
    let layout = objective().baseline_rho.flat_coordinates();
    eprintln!("PROBE3684 flat layout len={} seed={layout:?}", layout.len());
    let centre = Array1::from(vec![2.1697, 2.8027]);
    let offsets = [
        -3.0e-2, -1.0e-2, -3.0e-3, -1.0e-3, -3.0e-4, -1.0e-4, 0.0, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3,
        1.0e-2, 3.0e-2,
    ];
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    for (name, direction) in [("diag", [inv_sqrt2, inv_sqrt2]), ("anti", [inv_sqrt2, -inv_sqrt2])] {
        for &s in &offsets {
            let rho = Array1::from(vec![centre[0] + s * direction[0], centre[1] + s * direction[1]]);
            let mut cold = objective();
            report(&format!("cold-{name} s={s:+.1e}"), &mut cold, &rho);
        }
    }
    let mut warm = objective();
    for &s in &offsets {
        let rho = Array1::from(vec![centre[0] + s * inv_sqrt2, centre[1] + s * inv_sqrt2]);
        report(&format!("warm-diag s={s:+.1e}"), &mut warm, &rho);
    }
}
