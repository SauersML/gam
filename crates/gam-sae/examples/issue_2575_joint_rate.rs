//! #2575/#2517 — the support fixed point's convergence RATE, measured on the
//! eight-arm shape sweep #2517 declared the fast fixture for this work.
//!
//! Each arm plants in-class data (rows are exact `[1, cos τt, sin τt]·B_k`
//! mixtures over `top_k` circle atoms, plus a declared residual), seeds through
//! the production front door, and runs `solve_fixed_point` at the lane's own
//! tolerance. It prints, per arm, whether the certificate was reached, the
//! cycle it was reached on, and the CERTIFIED quantity (the parameter-space
//! Newton step, #2517) split by block — so an A/B across a solver change is a
//! diff of two tables rather than a wall-clock impression.
//!
//! ```text
//! cargo run -p gam-sae --release --example issue_2575_joint_rate
//! ```

use gam_sae::front_door::{SaeFitLane, admit_topk_manifold};
use gam_sae::manifold::{
    SaeSupportSeedRequest, SaeSupportTermSeedRequest, build_sae_support_seed,
    build_sae_support_term_seed, sae_support_effective_atom_dims,
};
use ndarray::{Array2, Axis};

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit(seed: u64) -> f64 {
    ((splitmix64(seed) >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

/// In-class data: every row is an exact mixture of `support_k` circle atoms
/// drawn from a planted dictionary of `k_atoms`, plus `residual` of noise. The
/// generating model is inside the fitted class, so a converged fit exists and
/// the only question the arm asks is whether the solver reaches it.
fn planted_circles(
    n_obs: usize,
    p_out: usize,
    k_atoms: usize,
    support_k: usize,
    residual: f64,
    seed: u64,
) -> Array2<f64> {
    let mut dictionary = vec![0.0_f64; k_atoms * 3 * p_out];
    for (index, value) in dictionary.iter_mut().enumerate() {
        *value = 2.0 * unit(seed ^ 0x1d ^ index as u64) - 1.0;
    }
    let mut target = Array2::<f64>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        for slot in 0..support_k {
            let atom = (splitmix64(seed ^ 0x2b ^ (row as u64) << 12 ^ slot as u64) as usize)
                % k_atoms;
            let t = unit(seed ^ 0x3c ^ (row as u64) << 20 ^ (slot as u64) << 4);
            let phi = [
                1.0,
                (std::f64::consts::TAU * t).cos(),
                (std::f64::consts::TAU * t).sin(),
            ];
            for out in 0..p_out {
                let mut value = 0.0;
                for (basis, weight) in phi.iter().enumerate() {
                    value += weight * dictionary[(atom * 3 + basis) * p_out + out];
                }
                target[[row, out]] += value;
            }
        }
        for out in 0..p_out {
            target[[row, out]] +=
                residual * (2.0 * unit(seed ^ 0x4d ^ (row * p_out + out) as u64) - 1.0);
        }
    }
    target
}

struct Arm {
    n_obs: usize,
    p_out: usize,
    k_atoms: usize,
    support_k: usize,
    residual: f64,
}

fn run_arm(arm: &Arm, tolerance: f64) -> Result<String, String> {
    let atom_basis: Vec<String> = vec!["periodic".to_string(); arm.k_atoms];
    let atom_dim: Vec<usize> = vec![1usize; arm.k_atoms];
    let effective = sae_support_effective_atom_dims(&atom_basis, &atom_dim)?;
    let d_max = effective.iter().copied().max().unwrap_or(1);
    let admission = admit_topk_manifold(arm.n_obs, arm.p_out, arm.k_atoms, d_max, arm.support_k)?;
    if admission.lane != SaeFitLane::CurvedStreaming {
        return Ok(format!("skipped: lane {:?}", admission.lane));
    }
    let target = planted_circles(
        arm.n_obs,
        arm.p_out,
        arm.k_atoms,
        arm.support_k,
        arm.residual,
        0x2575,
    );
    let mean = target.mean_axis(Axis(0)).ok_or("empty target")?;
    let centered = &target - &mean;

    let seed = build_sae_support_seed(SaeSupportSeedRequest {
        target: centered.view(),
        atom_basis: &atom_basis,
        atom_dim: &atom_dim,
        support_k: arm.support_k,
        random_state: 0,
        admission,
    })?;
    let retained = seed.retained_atom_indices.clone();
    let mut term = build_sae_support_term_seed(SaeSupportTermSeedRequest {
        assignment: seed.assignment,
        atom_basis: retained.iter().map(|&a| atom_basis[a].clone()).collect(),
        atom_dim: retained.iter().map(|&a| atom_dim[a]).collect(),
        output_dim: arm.p_out,
        random_state: 0,
    })?
    .term;
    let k_ret = term.k_atoms();
    let ard: Vec<Vec<f64>> = (0..k_ret)
        .map(|atom| vec![1.0; term.assignment.atom_coord_dim(atom)])
        .collect();
    let lambda = vec![1.0e-3_f64; k_ret];

    let outcome = term.solve_fixed_point(centered.view(), &lambda, &ard, tolerance, 1.0);
    let stationarity = term.raw_stationarity(centered.view(), &lambda, &ard)?;
    let objective = term.penalized_objective(centered.view(), &lambda, &ard)?;
    let verdict = match &outcome {
        Ok(report) => format!("RECURRED at cycle {}", report.iterations),
        Err(error) => {
            let accepted = error
                .split("joint Newton steps accepted=")
                .nth(1)
                .and_then(|tail| tail.split(',').next())
                .unwrap_or("n/a")
                .to_string();
            // The refusal's own words, truncated. A stalled arm whose certified
            // quantity is INSIDE the tolerance is refusing for a different
            // reason than one that is outside it, and reading the number without
            // the reason is how a limb gets blamed for another limb's refusal.
            let reason: String = error.chars().take(96).collect();
            format!("stalled (joint {accepted}) [{reason}]")
        }
    };
    Ok(format!(
        "K={k_ret:>3} {verdict:<34} diagonal-scaled={:.3e} (dec {:.3e}, coord {:.3e}) raw={:.3e} obj={objective:.6e}",
        stationarity.scaled_max_abs(),
        stationarity.decoder_scaled_max_abs,
        stationarity.coordinate_scaled_max_abs,
        stationarity.max_abs(),
    ))
}

fn main() {
    let tolerance = 1.0e-6_f64;
    let arms = [
        Arm { n_obs: 120, p_out: 4, k_atoms: 9, support_k: 1, residual: 1.0e-4 },
        Arm { n_obs: 120, p_out: 4, k_atoms: 9, support_k: 1, residual: 1.0e-2 },
        Arm { n_obs: 240, p_out: 8, k_atoms: 24, support_k: 3, residual: 1.0e-4 },
        Arm { n_obs: 240, p_out: 8, k_atoms: 24, support_k: 3, residual: 1.0e-2 },
        Arm { n_obs: 240, p_out: 8, k_atoms: 24, support_k: 1, residual: 1.0e-4 },
        Arm { n_obs: 480, p_out: 8, k_atoms: 24, support_k: 2, residual: 1.0e-4 },
        Arm { n_obs: 480, p_out: 8, k_atoms: 24, support_k: 3, residual: 1.0e-2 },
        Arm { n_obs: 120, p_out: 8, k_atoms: 12, support_k: 2, residual: 1.0e-4 },
    ];
    println!("#2575 support fixed-point rate sweep: tolerance {tolerance:.1e}");
    let mut recurred = 0usize;
    for arm in &arms {
        let label = format!(
            "n={:<4} P={:<3} K={:<3} s={} res={:.0e}",
            arm.n_obs, arm.p_out, arm.k_atoms, arm.support_k, arm.residual
        );
        let started = std::time::Instant::now();
        match run_arm(arm, tolerance) {
            Ok(line) => {
                if line.contains("RECURRED") {
                    recurred += 1;
                }
                println!("{label} | {line} | {:.1}s", started.elapsed().as_secs_f64());
            }
            Err(error) => println!("{label} | ERROR {error}"),
        }
    }
    println!("recurred {recurred}/{}", arms.len());
}
