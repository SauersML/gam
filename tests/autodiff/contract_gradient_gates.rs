use std::sync::{Arc, RwLock};

use gam::ResourcePolicy;
use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix,
};
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradient,
};
use gam::families::custom_family::{
    CustomFamilyBlockPsiDerivative, EvalMode, evaluate_custom_family_joint_hyper,
};
use gam::families::gamlss::GaussianLocationScaleFamily;
use gam::families::survival::{
    PenaltyBlock, PenaltyBlocks, SurvivalEngineInputs, SurvivalMonotonicityPenalty, SurvivalSpec,
    WorkingModelSurvival,
};
use gam::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam::smooth::BlockwisePenalty;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

const FD_STEP: f64 = 5.0e-6;
const OUTER_FD_STEP: f64 = 5.0e-6;
const REL_TOL: f64 = 1.0e-5;
const REL_FLOOR: f64 = 1.0e-8;
const ABS_TOL: f64 = 1.0e-9;

struct GradientChannel {
    name: &'static str,
    analytic: Vec<f64>,
    fd: Vec<f64>,
}

struct ContractRow {
    name: &'static str,
    run: fn() -> Vec<GradientChannel>,
}

fn assert_channel(row: &str, channel: &GradientChannel) {
    assert_eq!(
        channel.analytic.len(),
        channel.fd.len(),
        "{row}/{} length mismatch",
        channel.name
    );
    for idx in 0..channel.analytic.len() {
        let analytic = channel.analytic[idx];
        let fd = channel.fd[idx];
        assert!(
            analytic.is_finite() && fd.is_finite(),
            "{row}/{}[{idx}] non-finite: analytic={analytic:.12e} fd={fd:.12e}",
            channel.name
        );
        let rel = (analytic - fd).abs() / analytic.abs().max(fd.abs()).max(REL_FLOOR);
        let abs = (analytic - fd).abs();
        assert!(
            abs < ABS_TOL || rel < REL_TOL,
            "{row}/{}[{idx}] gradient is not the differential of the value: \
             analytic={analytic:.12e} fd={fd:.12e} abs={abs:.3e} rel={rel:.3e}",
            channel.name
        );
    }
}

#[test]
fn gradient_is_differential_contract_gate() {
    let rows = [
        ContractRow {
            name: "sae/euclidean-line",
            run: sae_euclidean_line_row,
        },
        ContractRow {
            name: "sae/k2-periodic-overlap",
            run: sae_k2_periodic_overlap_row,
        },
        ContractRow {
            name: "glm-reml/duchon-901-rank-deficient",
            run: glm_reml_outer_row,
        },
        ContractRow {
            name: "glm-reml/binomial-noncanonical-outer",
            run: glm_reml_binomial_noncanonical_outer_row,
        },
        ContractRow {
            name: "survival/laml-net-single-block",
            run: survival_laml_net_single_block_row,
        },
        ContractRow {
            name: "custom-family/joint-laml-penalized-quadratic",
            run: custom_family_joint_laml_penalized_quadratic_row,
        },
        ContractRow {
            name: "gamlss/gaussian-dispersion",
            run: gamlss_gaussian_dispersion_row,
        },
    ];

    for row in rows {
        let channels = (row.run)();
        assert!(
            !channels.is_empty(),
            "{} must report at least one gradient channel",
            row.name
        );
        for channel in channels {
            assert_channel(row.name, &channel);
        }
    }
}

fn warm_sae(term: &mut SaeManifoldTerm, z: &Array2<f64>, rho: &mut SaeManifoldRho) {
    for step in 0..4 {
        let loss = term
            .run_joint_fit_arrow_schur(z.view(), rho, None, 1, 1.0, 1.0e-6, 1.0e-6)
            .unwrap_or_else(|err| panic!("SAE warm step {step} failed: {err}"));
        assert!(
            loss.total().is_finite(),
            "SAE warm step {step} returned non-finite loss"
        );
    }
}

fn sae_value(term: &SaeManifoldTerm, z: &Array2<f64>, rho: &SaeManifoldRho) -> f64 {
    term.penalized_objective_total(z.view(), rho, None, 1.0)
        .expect("SAE penalized objective")
}

fn set_coord(term: &mut SaeManifoldTerm, atom: usize, row: usize, axis: usize, value: f64) {
    let mut coords = term.assignment.coords[atom].as_matrix();
    coords[[row, axis]] = value;
    term.assignment.coords[atom].set_flat(Array1::from_iter(coords.iter().copied()).view());
    term.atoms[atom]
        .refresh_basis(coords.view())
        .expect("refresh perturbed SAE coordinates");
}

fn sae_euclidean_line_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 72usize;
    let p = 6usize;
    let mut coords = Array2::<f64>::zeros((n, 1));
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let u = -1.0 + 2.0 * row as f64 / (n as f64 - 1.0);
        coords[[row, 0]] = 2.5 + 3.0 * u;
        for col in 0..p {
            let phase = (row * (col + 3)) as f64;
            z[[row, col]] = 0.08 * ((col % 3) as f64 - 1.0)
                + (0.35 + 0.07 * col as f64) * u
                + 0.04 * (phase.sin() + 0.5 * (0.37 * phase).cos());
        }
    }

    let evaluator = Arc::new(EuclideanPatchEvaluator::new(1, 2).expect("evaluator"));
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("basis");
    let m = phi.ncols();
    let smooth_penalty =
        gam::basis::create_difference_penalty_matrix(m, 2, None).expect("roughness penalty");
    let atom = SaeManifoldAtom::new(
        "contract-line",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .expect("atom")
    .with_basis_second_jet(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords],
        vec![LatentManifold::Euclidean],
        AssignmentMode::softmax(1.0),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
    let rho = SaeManifoldRho::new(0.0, (0.01_f64).ln(), vec![Array1::<f64>::zeros(1)]);
    (term, z, rho)
}

fn sae_euclidean_line_row() -> Vec<GradientChannel> {
    let (mut term, z, mut rho) = sae_euclidean_line_fixture();
    warm_sae(&mut term, &z, &mut rho);
    let sys = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("SAE line assemble");
    assert_eq!(
        sys.k,
        term.beta_dim(),
        "line row must stay in full beta coordinates"
    );

    let coord_probes = [3usize, 35, 68];
    let mut coord_an = Vec::new();
    let mut coord_fd = Vec::new();
    for row in coord_probes {
        let base = term.assignment.coords[0].as_matrix()[[row, 0]];
        coord_an.push(sys.rows[row].gt[0]);
        let mut plus = term.clone();
        set_coord(&mut plus, 0, row, 0, base + FD_STEP);
        let mut minus = term.clone();
        set_coord(&mut minus, 0, row, 0, base - FD_STEP);
        coord_fd.push((sae_value(&plus, &z, &rho) - sae_value(&minus, &z, &rho)) / (2.0 * FD_STEP));
    }

    let decoder_probes = [(0usize, 0usize), (0, 2), (0, 5)];
    let mut decoder_an = Vec::new();
    let mut decoder_fd = Vec::new();
    let p = term.output_dim();
    for (basis_col, out_col) in decoder_probes {
        let beta_idx = basis_col * p + out_col;
        let base = term.atoms[0].decoder_coefficients[[basis_col, out_col]];
        decoder_an.push(sys.gb[beta_idx]);
        let mut plus = term.clone();
        plus.atoms[0].decoder_coefficients[[basis_col, out_col]] = base + FD_STEP;
        let mut minus = term.clone();
        minus.atoms[0].decoder_coefficients[[basis_col, out_col]] = base - FD_STEP;
        decoder_fd
            .push((sae_value(&plus, &z, &rho) - sae_value(&minus, &z, &rho)) / (2.0 * FD_STEP));
    }

    vec![
        GradientChannel {
            name: "coords",
            analytic: coord_an,
            fd: coord_fd,
        },
        GradientChannel {
            name: "decoder",
            analytic: decoder_an,
            fd: decoder_fd,
        },
    ]
}

fn softmax2(logit0: f64, logit1: f64, temperature: f64) -> [f64; 2] {
    let m = logit0.max(logit1);
    let e0 = ((logit0 - m) / temperature).exp();
    let e1 = ((logit1 - m) / temperature).exp();
    let denom = e0 + e1;
    [e0 / denom, e1 / denom]
}

fn sae_k2_fixture() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
    let n = 36usize;
    let p = 3usize;
    let m = 3usize;
    let temperature = 0.9;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).expect("periodic evaluator"));
    let weights = [
        [
            [0.10, -0.05, 0.03],
            [0.35, -0.20, 0.12],
            [-0.16, 0.18, 0.08],
        ],
        [
            [-0.08, 0.04, 0.06],
            [0.22, 0.10, -0.18],
            [0.11, -0.24, 0.15],
        ],
    ];
    let mut logits = Array2::<f64>::zeros((n, 2));
    let mut coords = vec![Array2::<f64>::zeros((n, 1)), Array2::<f64>::zeros((n, 1))];
    let mut target = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let phase = (row as f64 + 0.35) / n as f64;
        coords[0][[row, 0]] = phase;
        coords[1][[row, 0]] = (phase + 0.21).fract();
        logits[[row, 0]] = if row % 2 == 0 { 0.8 } else { -0.6 };
        let assignments = softmax2(logits[[row, 0]], logits[[row, 1]], temperature);
        for atom in 0..2 {
            let theta = std::f64::consts::TAU * coords[atom][[row, 0]];
            let basis = [1.0, theta.sin(), theta.cos()];
            for out_col in 0..p {
                for basis_col in 0..m {
                    target[[row, out_col]] +=
                        assignments[atom] * basis[basis_col] * weights[atom][basis_col][out_col];
                }
            }
        }
    }

    let mut atoms = Vec::new();
    for atom_idx in 0..2 {
        let (phi, jet) = evaluator
            .evaluate(coords[atom_idx].view())
            .expect("periodic basis");
        let decoder = Array2::from_shape_fn((m, p), |(basis_col, out_col)| {
            weights[atom_idx][basis_col][out_col]
        });
        atoms.push(
            SaeManifoldAtom::new(
                format!("k2_{atom_idx}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(m),
            )
            .expect("periodic atom")
            .with_basis_second_jet(evaluator.clone()),
        );
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords,
        vec![LatentManifold::Circle { period: 1.0 }; 2],
        AssignmentMode::softmax(temperature),
    )
    .expect("K2 assignment");
    let term = SaeManifoldTerm::new(atoms, assignment).expect("K2 term");
    let rho = SaeManifoldRho::new(-6.0, -6.0, vec![array![-6.0], array![-6.0]]);
    (term, target, rho)
}

fn sae_k2_periodic_overlap_row() -> Vec<GradientChannel> {
    let (mut term, z, mut rho) = sae_k2_fixture();
    warm_sae(&mut term, &z, &mut rho);
    let sys = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("SAE K2 assemble");

    let mut logit_an = Vec::new();
    let mut logit_fd = Vec::new();
    for row in [0usize, 11, 28] {
        logit_an.push(sys.rows[row].gt[0]);
        let mut plus = term.clone();
        plus.assignment.logits[[row, 0]] += FD_STEP;
        let mut minus = term.clone();
        minus.assignment.logits[[row, 0]] -= FD_STEP;
        logit_fd.push((sae_value(&plus, &z, &rho) - sae_value(&minus, &z, &rho)) / (2.0 * FD_STEP));
    }

    let mut coord_an = Vec::new();
    let mut coord_fd = Vec::new();
    for (row, atom, local_pos) in [(4usize, 0usize, 1usize), (17, 1, 2), (31, 0, 1)] {
        let base = term.assignment.coords[atom].as_matrix()[[row, 0]];
        coord_an.push(sys.rows[row].gt[local_pos]);
        let mut plus = term.clone();
        set_coord(&mut plus, atom, row, 0, base + FD_STEP);
        let mut minus = term.clone();
        set_coord(&mut minus, atom, row, 0, base - FD_STEP);
        coord_fd.push((sae_value(&plus, &z, &rho) - sae_value(&minus, &z, &rho)) / (2.0 * FD_STEP));
    }

    let mut decoder_an = Vec::new();
    let mut decoder_fd = Vec::new();
    let p = term.output_dim();
    let per_atom_beta = term.atoms[0].decoder_coefficients.len();
    for (atom, basis_col, out_col) in [(0usize, 1usize, 1usize), (1, 2, 2), (1, 0, 0)] {
        let beta_idx = atom * per_atom_beta + basis_col * p + out_col;
        let base = term.atoms[atom].decoder_coefficients[[basis_col, out_col]];
        decoder_an.push(sys.gb[beta_idx]);
        let mut plus = term.clone();
        plus.atoms[atom].decoder_coefficients[[basis_col, out_col]] = base + FD_STEP;
        let mut minus = term.clone();
        minus.atoms[atom].decoder_coefficients[[basis_col, out_col]] = base - FD_STEP;
        decoder_fd
            .push((sae_value(&plus, &z, &rho) - sae_value(&minus, &z, &rho)) / (2.0 * FD_STEP));
    }

    vec![
        GradientChannel {
            name: "logits",
            analytic: logit_an,
            fd: logit_fd,
        },
        GradientChannel {
            name: "coords",
            analytic: coord_an,
            fd: coord_fd,
        },
        GradientChannel {
            name: "decoder",
            analytic: decoder_an,
            fd: decoder_fd,
        },
    ]
}

fn second_difference_penalty(k: usize) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for i in 0..(k - 2) {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    d.t().dot(&d)
}

fn glm_reml_outer_row() -> Vec<GradientChannel> {
    let n = 96usize;
    let k = 6usize;
    let p = 1 + 2 * k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        let mut acc = 1.0;
        for j in 0..k {
            acc *= z;
            x[[i, 1 + j]] = acc;
            x[[i, 1 + k + j]] = acc + 1.0e-3 * ((i + j) as f64).sin();
        }
        y[i] = 0.4 + (std::f64::consts::PI * z).sin() + 0.05 * (7.0 * z).cos();
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let penalties = vec![
        BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k)),
        BlockwisePenalty::new((1 + k)..p, second_difference_penalty(k)),
    ];
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 1000,
        tol: 1.0e-9,
        nullspace_dims: vec![2, 2],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let rho = array![0.15, 0.1501];
    let analytic = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("GLM REML gradient");
    let mut fd = Vec::new();
    for j in 0..rho.len() {
        let mut plus = rho.clone();
        plus[j] += OUTER_FD_STEP;
        let mut minus = rho.clone();
        minus[j] -= OUTER_FD_STEP;
        let fp = evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            x.clone(),
            offset.view(),
            &penalties,
            &opts,
            &plus,
        )
        .expect("GLM REML f+")
        .0;
        let fm = evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            x.clone(),
            offset.view(),
            &penalties,
            &opts,
            &minus,
        )
        .expect("GLM REML f-")
        .0;
        fd.push((fp - fm) / (2.0 * OUTER_FD_STEP));
    }
    vec![GradientChannel {
        name: "rho",
        analytic: analytic.to_vec(),
        fd,
    }]
}

fn glm_reml_binomial_noncanonical_outer_row() -> Vec<GradientChannel> {
    let n = 96usize;
    let k = 6usize;
    let p = 1 + 2 * k;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        x[[i, 0]] = 1.0;
        let z = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        let mut acc = 1.0;
        for j in 0..k {
            acc *= z;
            x[[i, 1 + j]] = acc;
            x[[i, 1 + k + j]] = acc + 1.0e-3 * ((i + j) as f64).sin();
        }
        y[i] = if (std::f64::consts::PI * z).sin() + 0.3 * (3.0 * z).cos() > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let penalties = vec![
        BlockwisePenalty::new(1..(1 + k), second_difference_penalty(k)),
        BlockwisePenalty::new((1 + k)..p, second_difference_penalty(k)),
    ];
    let opts = ExternalOptimOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
        ),
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![2, 2],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let rho = array![0.2, 0.25];
    let analytic = evaluate_externalgradient(
        y.view(),
        weights.view(),
        x.clone(),
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("binomial GLM REML gradient");
    let mut fd = Vec::new();
    for j in 0..rho.len() {
        let mut plus = rho.clone();
        plus[j] += OUTER_FD_STEP;
        let mut minus = rho.clone();
        minus[j] -= OUTER_FD_STEP;
        let fp = evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            x.clone(),
            offset.view(),
            &penalties,
            &opts,
            &plus,
        )
        .expect("binomial GLM REML f+")
        .0;
        let fm = evaluate_externalcost_andridge(
            y.view(),
            weights.view(),
            x.clone(),
            offset.view(),
            &penalties,
            &opts,
            &minus,
        )
        .expect("binomial GLM REML f-")
        .0;
        fd.push((fp - fm) / (2.0 * OUTER_FD_STEP));
    }
    vec![GradientChannel {
        name: "rho",
        analytic: analytic.to_vec(),
        fd,
    }]
}

/// 20-subject net-survival fixture: intercept + a single penalized,
/// mean-centred log-age time covariate (positive exit derivative). Mirrors
/// the in-crate `laml_fd_test_model` fixture: large enough that the
/// observed-information Hessian is well-conditioned at the mode, so the
/// inner PIRLS reaches the tight shim tolerance and V(ρ) is FD-smooth.
/// The first block (λ = 0) is an inactive prefix; only block 1 is active,
/// so the active-block ρ vector has length 1.
fn survival_single_block_model(active_lambda: f64) -> WorkingModelSurvival {
    let age_entry: Array1<f64> = Array1::from(vec![
        30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 62.0, 34.0,
        39.0, 44.0, 49.0, 54.0, 59.0,
    ]);
    let age_exit: Array1<f64> = Array1::from(vec![
        45.0, 48.0, 55.0, 58.0, 62.0, 66.0, 68.0, 47.0, 52.0, 53.0, 55.0, 60.0, 63.0, 70.0, 48.0,
        51.0, 58.0, 62.0, 66.0, 69.0,
    ]);
    let event_target = Array1::from(vec![
        1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    ]);
    let n = age_entry.len();
    let event_competing = Array1::<u8>::zeros(n);
    let sampleweight = Array1::from_elem(n, 1.0_f64);
    let ln_age_mean: f64 = {
        let mut sum = 0.0;
        for i in 0..n {
            sum += age_entry[i].ln() + age_exit[i].ln();
        }
        sum / (2.0 * n as f64)
    };
    let mut x_entry = Array2::<f64>::zeros((n, 2));
    let mut x_exit = Array2::<f64>::zeros((n, 2));
    let mut x_derivative = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_entry[[i, 0]] = 1.0;
        x_exit[[i, 0]] = 1.0;
        x_entry[[i, 1]] = age_entry[i].ln() - ln_age_mean;
        x_exit[[i, 1]] = age_exit[i].ln() - ln_age_mean;
        x_derivative[[i, 0]] = 0.0;
        x_derivative[[i, 1]] = 1.0 / age_exit[i];
    }
    let penalties = PenaltyBlocks::new(vec![
        PenaltyBlock {
            matrix: array![[3.0]],
            lambda: 0.0,
            range: 0..1,
            nullspace_dim: 0,
        },
        PenaltyBlock {
            matrix: array![[2.5]],
            lambda: active_lambda,
            range: 1..2,
            nullspace_dim: 0,
        },
    ]);
    WorkingModelSurvival::from_engine_inputs(
        SurvivalEngineInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            sampleweight: sampleweight.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
        },
        penalties,
        SurvivalMonotonicityPenalty { tolerance: 1e-8 },
        SurvivalSpec::Net,
    )
    .expect("construct single-block survival LAML FD model")
}

fn survival_laml_net_single_block_row() -> Vec<GradientChannel> {
    let model = survival_single_block_model(1.0);
    let beta0 = array![-2.5_f64, 1.0];
    let rho = array![0.3];
    let analytic = model
        .evaluate_survival_lamlcost_and_gradient(rho.as_slice().expect("contiguous rho"), &beta0)
        .expect("survival LAML analytic gradient evaluation should succeed")
        .1;
    let mut fd = Vec::new();
    for j in 0..rho.len() {
        let mut plus = rho.clone();
        plus[j] += OUTER_FD_STEP;
        let mut minus = rho.clone();
        minus[j] -= OUTER_FD_STEP;
        let fp = model
            .evaluate_survival_lamlcost_and_gradient(
                plus.as_slice().expect("contiguous rho"),
                &beta0,
            )
            .expect("survival LAML f+")
            .0;
        let fm = model
            .evaluate_survival_lamlcost_and_gradient(
                minus.as_slice().expect("contiguous rho"),
                &beta0,
            )
            .expect("survival LAML f-")
            .0;
        fd.push((fp - fm) / (2.0 * OUTER_FD_STEP));
    }
    vec![GradientChannel {
        name: "rho",
        analytic: analytic.to_vec(),
        fd,
    }]
}

/// A penalized multi-coefficient quadratic family. With a single penalty
/// block of dimension `m`, the LAML objective is
///     V(ρ) = ½‖β̂−c‖² + ½λ β̂ᵀSβ̂ + ½ log|H| − ½ log|λ S|₊ − ½(m−ν)ρ,
/// where H = I + λS and `c` is a per-coefficient target. The `½ log|H|`
/// term is genuinely ρ-dependent and non-separable across the eigenbasis
/// of S, so it exercises the same matrix-function-of-ρ machinery whose
/// value↔gradient drift is the bug class under test.
#[derive(Clone)]
struct PenalizedQuadraticFamily {
    target: Array1<f64>,
}

impl CustomFamily for PenalizedQuadraticFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = &block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta;
        if beta.len() != self.target.len() {
            return Err("beta/target dimension mismatch".to_string());
        }
        let resid = beta - &self.target;
        let nll = 0.5 * resid.iter().map(|r| r * r).sum::<f64>();
        let m = beta.len();
        Ok(FamilyEvaluation {
            log_likelihood: -nll,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: resid.mapv(|r| -r),
                hessian: SymmetricMatrix::Dense(Array2::<f64>::eye(m)),
            }],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let m = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        Ok(Some(Array2::<f64>::eye(m)))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_idx != 0 {
            return Ok(None);
        }
        let m = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        if direction.len() != m {
            return Err("direction dimension mismatch".to_string());
        }
        // The Hessian is the constant identity ⇒ its directional
        // derivative w.r.t. β is the zero matrix.
        Ok(Some(Array2::<f64>::zeros((m, m))))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let m = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .len();
        if direction.len() != m {
            return Err("direction dimension mismatch".to_string());
        }
        Ok(Some(Array2::<f64>::zeros((m, m))))
    }
}

fn penalized_quadratic_specs(
    target: &Array1<f64>,
    penalty: Array2<f64>,
) -> Vec<ParameterBlockSpec> {
    let m = target.len();
    vec![ParameterBlockSpec {
        name: "penalized-quadratic".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::eye(m))),
        offset: Array1::<f64>::zeros(m),
        penalties: vec![PenaltyMatrix::Dense(penalty)],
        nullspace_dims: vec![0],
        initial_log_lambdas: Array1::<f64>::zeros(1),
        initial_beta: Some(Array1::<f64>::zeros(m)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }]
}

fn custom_family_opts() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ridge_floor: 1e-12,
        inner_tol: 1e-10,
        ..BlockwiseFitOptions::default()
    }
}

fn distinct_diag_penalty(m: usize) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((m, m));
    for j in 0..m {
        s[[j, j]] = 1.0 + j as f64;
    }
    s
}

fn custom_family_joint_laml_penalized_quadratic_row() -> Vec<GradientChannel> {
    let target = array![0.5_f64, -0.3, 0.8, 0.1];
    let family = PenalizedQuadraticFamily {
        target: target.clone(),
    };
    let specs = penalized_quadratic_specs(&target, distinct_diag_penalty(target.len()));
    let opts = custom_family_opts();
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new()];
    let rho = array![0.4];
    let analytic = evaluate_custom_family_joint_hyper(
        &family,
        &specs,
        &opts,
        &rho,
        &derivative_blocks,
        None,
        EvalMode::ValueAndGradient,
    )
    .expect("custom-family joint LAML gradient eval")
    .gradient;
    let mut fd = Vec::new();
    for j in 0..rho.len() {
        let mut plus = rho.clone();
        plus[j] += OUTER_FD_STEP;
        let mut minus = rho.clone();
        minus[j] -= OUTER_FD_STEP;
        let fp = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &opts,
            &plus,
            &derivative_blocks,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("custom-family joint LAML f+")
        .objective;
        let fm = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &opts,
            &minus,
            &derivative_blocks,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("custom-family joint LAML f-")
        .objective;
        fd.push((fp - fm) / (2.0 * OUTER_FD_STEP));
    }
    vec![GradientChannel {
        name: "rho",
        analytic: analytic.to_vec(),
        fd,
    }]
}

fn spec_without_penalty(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::from(design),
        offset: Array1::zeros(n),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn gamlss_gaussian_dispersion_row() -> Vec<GradientChannel> {
    let n = 32usize;
    let mut x_mu = Array2::<f64>::zeros((n, 2));
    let mut x_ls = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -1.0 + 2.0 * i as f64 / (n as f64 - 1.0);
        x_mu[[i, 0]] = 1.0;
        x_mu[[i, 1]] = t;
        x_ls[[i, 0]] = 1.0;
        x_ls[[i, 1]] = t * t - 1.0 / 3.0;
        let mu = 0.3 + 0.7 * t;
        let log_sigma = -0.25_f64 + 0.2 * (t * t - 1.0 / 3.0);
        y[i] = mu + log_sigma.exp() * (0.35 * (5.0 * t).sin() + 0.12 * (11.0 * t).cos());
    }
    let specs = vec![
        spec_without_penalty("mu", x_mu.clone()),
        spec_without_penalty("log_sigma", x_ls.clone()),
    ];
    let family = GaussianLocationScaleFamily {
        y,
        weights: Array1::ones(n),
        mu_design: Some(specs[GaussianLocationScaleFamily::BLOCK_MU].design.clone()),
        log_sigma_design: Some(
            specs[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
                .design
                .clone(),
        ),
        policy: ResourcePolicy::default_library(),
        cached_row_scalars: RwLock::new(None),
    };
    let beta_mu = array![0.25, 0.55];
    let beta_ls = array![-0.35, 0.18];
    let states = vec![
        ParameterBlockState {
            beta: beta_mu.clone(),
            eta: x_mu.dot(&beta_mu),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let eval = family.evaluate(&states).expect("GAMLSS fixed-state eval");
    let analytic_mu = match &eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_MU] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => x_mu
            .t()
            .dot(&(working_weights * &(working_response - &states[0].eta))),
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
    };
    let analytic_ls = match &eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => x_ls
            .t()
            .dot(&(working_weights * &(working_response - &states[1].eta))),
        BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
    };
    let value_at = |flat: &Array1<f64>| -> f64 {
        let mu_beta = flat.slice(ndarray::s![0..2]).to_owned();
        let ls_beta = flat.slice(ndarray::s![2..4]).to_owned();
        let trial_states = vec![
            ParameterBlockState {
                beta: mu_beta.clone(),
                eta: x_mu.dot(&mu_beta),
            },
            ParameterBlockState {
                beta: ls_beta.clone(),
                eta: x_ls.dot(&ls_beta),
            },
        ];
        family
            .log_likelihood_only(&trial_states)
            .expect("GAMLSS fixed-state log-likelihood")
    };
    let beta = array![beta_mu[0], beta_mu[1], beta_ls[0], beta_ls[1]];
    let mut fd = Vec::new();
    for j in 0..beta.len() {
        let mut plus = beta.clone();
        plus[j] += FD_STEP;
        let mut minus = beta.clone();
        minus[j] -= FD_STEP;
        fd.push((value_at(&plus) - value_at(&minus)) / (2.0 * FD_STEP));
    }
    vec![
        GradientChannel {
            name: "mu-beta",
            analytic: analytic_mu.to_vec(),
            fd: fd[0..2].to_vec(),
        },
        GradientChannel {
            name: "log-sigma-beta",
            analytic: analytic_ls.to_vec(),
            fd: fd[2..4].to_vec(),
        },
    ]
}
