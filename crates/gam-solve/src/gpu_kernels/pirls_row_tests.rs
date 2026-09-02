// `pirls_row.rs` declares this file as `#[cfg(test)] mod pirls_row_tests;`;
// declaring the test scope in-file makes that a claim the compiler enforces.
#![cfg(test)]

use super::*;

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

