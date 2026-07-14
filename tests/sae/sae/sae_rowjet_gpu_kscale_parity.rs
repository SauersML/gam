//! #2304 — the production-shaped SAE GPU row jet contains every coordinate
//! channel and matches the CPU row program on every packed output.

use gam::terms::sae::gpu_kernels::sae_rowjet::{
    SaeRowJetPath, SaeRowJetPrimary, SaeSoftmaxRowJetInput, execute_softmax_row_jet_tile,
};

fn fixture(n: usize) -> Vec<SaeSoftmaxRowJetInput> {
    let primaries = vec![
        SaeRowJetPrimary::Logit { atom: 0 },
        SaeRowJetPrimary::Coordinate { atom: 0, axis: 0 },
        SaeRowJetPrimary::Coordinate { atom: 1, axis: 0 },
        SaeRowJetPrimary::Coordinate { atom: 1, axis: 1 },
    ];
    let (k, q, p, n_beta) = (2, primaries.len(), 3, 2);
    (0..n)
        .map(|row| {
            let z0 = 0.25 + 0.5 * ((row + 1) as f64 * 0.017).sin().abs();
            let mut decoded_first = vec![0.0; q * p];
            for slot in 1..q {
                for c in 0..p {
                    decoded_first[slot * p + c] =
                        ((row * 13 + slot * 7 + c + 1) as f64 * 0.031).sin();
                }
            }
            let mut decoded_second = vec![0.0; q * q * p];
            for &(left, right) in &[(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)] {
                for c in 0..p {
                    decoded_second[(left * q + right) * p + c] =
                        ((row * 19 + left * 5 + right * 3 + c + 2) as f64 * 0.023).cos();
                }
            }
            let mut beta_basis_first = vec![0.0; q * n_beta];
            beta_basis_first[n_beta] = 0.3;
            beta_basis_first[2 * n_beta + 1] = -0.2;
            beta_basis_first[3 * n_beta + 1] = 0.6;
            SaeSoftmaxRowJetInput {
                n_atoms: k,
                out_dim: p,
                coordinate_slots: SaeSoftmaxRowJetInput::coordinate_slots_for(&primaries),
                primaries: primaries.clone(),
                gate_values: vec![z0, 1.0 - z0],
                active_atoms: vec![true, true],
                sqrt_row_weight: (1.0 + row as f64 / n as f64).sqrt(),
                decoded: (0..k * p)
                    .map(|index| ((row * 11 + index * 3 + 1) as f64 * 0.041).cos())
                    .collect(),
                decoded_first,
                decoded_second,
                beta_atoms: vec![0, 1].into(),
                beta_basis_values: vec![0.7, -0.4],
                beta_basis_first,
                beta_outputs: vec![1.0, 0.2, -0.1, -0.3, 0.8, 0.5].into(),
            }
        })
        .collect()
}

fn maximum_channel_error(
    left: &gam::terms::sae::gpu_kernels::sae_rowjet::SaeRowJetChannels,
    right: &gam::terms::sae::gpu_kernels::sae_rowjet::SaeRowJetChannels,
) -> f64 {
    left.first
        .iter()
        .chain(&left.second)
        .chain(&left.beta)
        .chain(&left.beta_mixed)
        .zip(
            right
                .first
                .iter()
                .chain(&right.second)
                .chain(&right.beta)
                .chain(&right.beta_mixed),
        )
        .fold(0.0_f64, |maximum, (a, b)| maximum.max((a - b).abs()))
}

#[test]
fn complete_sae_rowjet_gpu_parity_2304() {
    let rows = fixture(41);
    let cpu = execute_softmax_row_jet_tile(&rows, 1.0 / 0.7, SaeRowJetPath::Cpu)
        .expect("complete CPU row-program oracle");

    // The fixture has three coordinate primaries. Pin both orientations of a
    // logit×coordinate block and a same-atom off-diagonal coordinate Hessian.
    let (q, p) = (cpu.q, cpu.p);
    let logit_coord = (0 * q + 2) * p;
    let coord_logit = (2 * q) * p;
    assert_eq!(
        &cpu.second[logit_coord..logit_coord + p],
        &cpu.second[coord_logit..coord_logit + p]
    );
    assert!(
        cpu.second[logit_coord..logit_coord + p]
            .iter()
            .any(|value| *value != 0.0)
    );
    let coord_cross = (2 * q + 3) * p;
    assert!(
        cpu.second[coord_cross..coord_cross + p]
            .iter()
            .any(|value| *value != 0.0)
    );
    assert!(cpu.beta_mixed.iter().any(|value| *value != 0.0));

    #[cfg(target_os = "linux")]
    if gam::gpu::GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in row-jet parity test: {error}"))
        .is_some()
    {
        let device = execute_softmax_row_jet_tile(&rows, 1.0 / 0.7, SaeRowJetPath::Device)
            .expect("admitted CUDA device must run; no host retry is permitted");
        let maximum = maximum_channel_error(&cpu, &device);
        assert!(
            maximum <= 1.0e-12,
            "complete device/CPU SAE row-jet error {maximum:e} exceeds 1e-12"
        );
    }
}
