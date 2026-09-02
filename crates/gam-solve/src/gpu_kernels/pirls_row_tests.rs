// `pirls_row.rs` declares this file as `#[cfg(test)] mod pirls_row_tests;`;
// declaring the test scope in-file makes that a claim the compiler enforces.
#![cfg(test)]

use super::*;

#[cfg(target_os = "linux")]
fn close(got: f64, expected: f64, rel: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (got - expected).abs() <= rel * scale,
        "got={got:?}, expected={expected:?}, rel={rel:?}"
    );
}

#[cfg(target_os = "linux")]
/// Resolve the PIRLS-row device backend, or assert the device-free contract.
///
/// The availability question goes through [`gam_gpu::test_gate::gpu_for_test`]
/// so an absent device is COUNTED and a `GpuPolicy::Required` lane turns it into
/// a failure (#2422). The device-free assertions below are this module's own and
/// are strictly stronger than the shared gate: they additionally require the
/// backend to decline in agreement with the runtime, and to say why. Those are
/// kept — routing through the gate adds the counting, it does not replace the
/// contract.
fn backend_or_assert_runtime_decline() -> Option<&'static PirlsRowBackend> {
    let skips_before = gam_gpu::test_gate::skipped_for_absent_device();
    match gam_gpu::test_gate::gpu_for_test("PIRLS-row device parity") {
        gam_gpu::test_gate::GpuTestGate::Ready(_) => Some(
            PirlsRowBackend::probe()
                .unwrap_or_else(|error| panic!("PIRLS-row CUDA backend probe failed: {error}")),
        ),
        gam_gpu::test_gate::GpuTestGate::AbsentDevice => {
            gam_gpu::test_gate::assert_absent_device_was_counted(skips_before);
            match PirlsRowBackend::probe() {
                Err(GpuError::DriverLibraryUnavailable { reason }) => {
                    assert!(
                        !reason.trim().is_empty(),
                        "a device-free PIRLS-row backend refusal must explain why CUDA is \
                         unavailable"
                    );
                    None
                }
                Ok(_) => panic!(
                    "the process-wide CUDA runtime declined, but the PIRLS-row backend admitted \
                     a device"
                ),
                Err(error) => panic!(
                    "the process-wide CUDA runtime declined cleanly, but the PIRLS-row backend \
                     faulted: {error}"
                ),
            }
        }
    }
}

#[test]
fn refusal_replay_selects_the_smallest_bad_row_atomically() {
    let eta = [0.0, 0.0, 0.0];
    let y = [0.0, 2.0, -1.0];
    let prior = [1.0; 3];
    let status = [
        status_codes::OK,
        status_codes::RESPONSE,
        status_codes::RESPONSE,
    ];
    assert!(matches!(
        replay_first_refusal(
            PirlsRowFamily::BernoulliLogit,
            CurvatureMode::Fisher,
            1.0,
            &eta,
            &y,
            &prior,
            &status,
        ),
        Err(EstimationError::PirlsRowGeometryUnrepresentable { row: 1, .. })
    ));
}

#[cfg(target_os = "linux")]
#[test]
fn generated_sources_have_one_exact_unprojected_contract() {
    let forbidden = [
        "clamp_eta",
        "ETA_CLAMP",
        "MU_FLOOR",
        "W_SOLVER_FLOOR",
        "fmax(",
        "fmin(",
        "flags",
        "1e-12",
        "1e-10",
    ];
    for family in PirlsRowFamily::ALL {
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            for source in [
                cuda_source_for(family, curvature),
                solve_row_source_for(family, curvature),
                ladder_source_for(family, curvature),
            ] {
                for token in forbidden {
                    assert!(!source.contains(token), "{family:?}/{curvature:?}: {token}");
                }
                assert!(source.contains("w_solver = w_hessian"));
                assert!(source.contains("status == PIRLS_OK"));
            }
        }
    }
    let ladder = ladder_source_for(PirlsRowFamily::PoissonLog, CurvatureMode::Fisher);
    assert!(ladder.contains("status_out[k * n + i] = status"));
}

#[cfg(target_os = "linux")]
#[test]
fn nvrtc_declines_without_cuda_else_compiles_every_exact_builtin_mode() -> Result<(), GpuError> {
    let Some(backend) = backend_or_assert_runtime_decline() else {
        return Ok(());
    };
    for family in PirlsRowFamily::ALL {
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            backend.module_for(family, curvature)?;
            backend.module_for_solve(family, curvature)?;
            backend.module_for_ladder(family, curvature)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn device_rows_decline_without_cuda_else_match_cpu_at_log_endpoints_tails_and_tiny_weights() {
    let Some(backend) = backend_or_assert_runtime_decline() else {
        return;
    };
    let stream = backend.inner.ctx.default_stream();
    for family in PirlsRowFamily::ALL {
        let eta: Vec<f64> = match family {
            PirlsRowFamily::PoissonLog | PirlsRowFamily::GammaLog => {
                vec![-700.0, -2.0, 0.0, 2.0, 700.0]
            }
            PirlsRowFamily::BernoulliLogit => vec![-700.0, -2.0, 0.0, 2.0, 700.0],
            PirlsRowFamily::BernoulliProbit => vec![-6.0, -2.0, 0.0, 2.0, 6.0],
            PirlsRowFamily::BernoulliCLogLog => vec![-20.0, -2.0, 0.0, 1.0, 3.0],
            PirlsRowFamily::GaussianIdentity => vec![-700.0, -2.0, 0.0, 2.0, 700.0],
        };
        let y: Vec<f64> = eta
            .iter()
            .map(|&e| match family {
                PirlsRowFamily::PoissonLog | PirlsRowFamily::GammaLog => e.exp(),
                PirlsRowFamily::GaussianIdentity => e + 0.25,
                _ => {
                    if e >= 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let prior = vec![f64::MIN_POSITIVE, 0.25, 1.0, 2.0, f64::MIN_POSITIVE];
        let eta_dev = stream.clone_htod(&eta).unwrap();
        let y_dev = stream.clone_htod(&y).unwrap();
        let prior_dev = stream.clone_htod(&prior).unwrap();
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            let mut out = RowOutputDevBuffers::allocate(&stream, eta.len()).unwrap();
            launch_row_reweight_on_stream(
                backend,
                family,
                curvature,
                2.0,
                &stream,
                eta.len(),
                &eta_dev,
                &y_dev,
                &prior_dev,
                &mut out,
            )
            .unwrap();
            let status = stream.clone_dtoh(&out.status).unwrap();
            let mu = stream.clone_dtoh(&out.mu).unwrap();
            let grad = stream.clone_dtoh(&out.grad_eta).unwrap();
            let wh = stream.clone_dtoh(&out.w_hessian).unwrap();
            let ws = stream.clone_dtoh(&out.w_solver).unwrap();
            let dev = stream.clone_dtoh(&out.deviance).unwrap();
            for i in 0..eta.len() {
                let cpu = row_reweight_cpu_at(
                    i,
                    family,
                    curvature,
                    RowInput {
                        eta: eta[i],
                        y: y[i],
                        prior_weight: prior[i],
                    },
                    2.0,
                );
                match cpu {
                    Ok(cpu) => {
                        assert_eq!(status[i], status_codes::OK);
                        close(mu[i], cpu.mu, 2.0e-12);
                        close(grad[i], cpu.grad_eta, 2.0e-11);
                        close(wh[i], cpu.w_hessian, 2.0e-11);
                        close(ws[i], cpu.w_solver, 2.0e-11);
                        close(dev[i], cpu.deviance, 2.0e-11);
                    }
                    Err(_) => assert_ne!(status[i], status_codes::OK),
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn failed_device_row_declines_without_cuda_else_writes_only_its_status() {
    let Some(backend) = backend_or_assert_runtime_decline() else {
        return;
    };
    let stream = backend.inner.ctx.default_stream();
    let y_dev = stream.clone_htod(&[1.0, 1.0]).unwrap();
    let prior_dev = stream.clone_htod(&[1.0, 1.0]).unwrap();
    let mut out = RowOutputDevBuffers::allocate(&stream, 2).unwrap();

    let valid_eta = stream.clone_htod(&[0.0, 0.0]).unwrap();
    launch_row_reweight_on_stream(
        backend,
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        1.0,
        &stream,
        2,
        &valid_eta,
        &y_dev,
        &prior_dev,
        &mut out,
    )
    .unwrap();
    assert_eq!(stream.clone_dtoh(&out.mu).unwrap(), vec![1.0, 1.0]);

    let invalid_eta = stream.clone_htod(&[701.0, 0.0]).unwrap();
    launch_row_reweight_on_stream(
        backend,
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        1.0,
        &stream,
        2,
        &invalid_eta,
        &y_dev,
        &prior_dev,
        &mut out,
    )
    .unwrap();
    let status = stream.clone_dtoh(&out.status).unwrap();
    let mu = stream.clone_dtoh(&out.mu).unwrap();
    assert_eq!(status, vec![status_codes::ETA_DOMAIN, status_codes::OK]);
    assert_eq!(mu, vec![1.0, 1.0]);
}
