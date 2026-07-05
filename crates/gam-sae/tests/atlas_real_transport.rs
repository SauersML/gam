//! Atlas-Machine composition on REAL Qwen3 transport artifacts.
//!
//! Turns the abstract contract composer + loop-holonomy instrument into a
//! measurement on actual model internals. The fixture
//! `tests/data/qwen3_l11_l17_l23_theta.json` holds the per-row circular chart
//! angle `θ` — the angle in each layer's top-2 principal plane of the real
//! centered activations — at residual-stream layers L11 / L17 / L23 of
//! Qwen3.5-35B-A3B, from the MSI activation cache. Each layer's `θ` carries its
//! own arbitrary plane gauge/orientation; the composition triangle below is a
//! CLOSED loop, so every per-layer gauge cancels and the holonomy verdict is
//! gauge-invariant (no cross-layer alignment needed). Full provenance — and why
//! the top-2 plane angle rather than the K=1 SAE atom (its line search
//! live-locks at the MSI gamfit) — is in `tests/data/MSI_PROVENANCE.md`.
//!
//! From those angles this test fits the three inter-layer transports with the
//! crate's own REML machinery — the two forward hops `h: L11→L17`, `L17→L23`
//! and the direct map `L11→L23` — and then:
//!
//!  1. composes the two forward hops into an END-TO-END error bound
//!     ([`compose_contracts`]): each hop's target-space residual is the stage
//!     defect and its `sup|h′|` the metric expansion, so `total_defect` is the
//!     shadowing bound on the composed L11→L23 map built stage-by-stage;
//!  2. measures the LOOP HOLONOMY of the composition triangle
//!     `h_ab, h_bc, h_ac⁻¹` ([`loop_holonomy`]): each transport is classified as
//!     an `O(2)` element and the loop returns to L11, so a nontrivial net
//!     element is the obstruction to "the weekday circle is one global feature
//!     carried consistently around the loop" — equivalently, the measured
//!     failure of the composition law `h_ac = h_bc ∘ h_ab`, judged against the
//!     loop's own summed `O(2)` defects (measure-don't-latch).
//!
//! The printed numbers (visible with `--nocapture`) are the deliverable; the
//! assertions pin only data-agnostic structural invariants, never a fitted
//! value.

use gam_sae::inference::contracts::{Contract, compose_contracts, invert_o2_edge, loop_holonomy};
use gam_sae::inference::layer_transport::{ChartTopology, FittedTransport, fit_transport_map};
use gam_sae::inference::transport_class::classify_circle_transport_fit;
use ndarray::Array1;

const GRID: usize = 512;

fn load_theta() -> (String, Vec<String>, Vec<Array1<f64>>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/qwen3_l11_l17_l23_theta.json"
    );
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("missing real-transport fixture {path}: {e}"));
    let v: serde_json::Value = serde_json::from_str(&raw).expect("fixture is valid JSON");
    let model = v["model"].as_str().unwrap_or("unknown").to_string();
    let keys: Vec<String> = v["layer_keys"]
        .as_array()
        .expect("layer_keys array")
        .iter()
        .map(|k| k.as_str().unwrap().to_string())
        .collect();
    let theta: Vec<Array1<f64>> = keys
        .iter()
        .map(|k| {
            let col = v["theta"][k]
                .as_array()
                .unwrap_or_else(|| panic!("theta[{k}] array"));
            Array1::from_vec(col.iter().map(|x| x.as_f64().unwrap()).collect())
        })
        .collect();
    (model, keys, theta)
}

fn fit(from: &Array1<f64>, to: &Array1<f64>) -> FittedTransport {
    fit_transport_map(
        from.view(),
        to.view(),
        ChartTopology::Circle,
        ChartTopology::Circle,
    )
    .expect("real transport fit")
}

#[test]
fn atlas_composed_bound_and_holonomy_on_real_qwen3_transport() {
    let (model, keys, theta) = load_theta();
    assert_eq!(keys.len(), 3, "need exactly three layers for the triangle");
    let n = theta[0].len();
    for (k, col) in keys.iter().zip(theta.iter()) {
        assert_eq!(col.len(), n, "layer {k} row count disagrees");
    }
    assert!(n >= 64, "too few tokens for a stable circle fit: {n}");

    // Three inter-layer transports: two forward hops and the direct map.
    let h_ab = fit(&theta[0], &theta[1]);
    let h_bc = fit(&theta[1], &theta[2]);
    let h_ac = fit(&theta[0], &theta[2]);

    // (1) End-to-end composed error bound over the forward path L11→L17→L23.
    let c_ab = Contract::from_transport(&h_ab, format!("{}->{}", keys[0], keys[1]), GRID)
        .expect("contract ab");
    let c_bc = Contract::from_transport(&h_bc, format!("{}->{}", keys[1], keys[2]), GRID)
        .expect("contract bc");
    let composed = compose_contracts(&[c_ab.clone(), c_bc.clone()]);

    // (2) Loop holonomy of the composition triangle h_ab, h_bc, h_ac⁻¹.
    let cls_ab = classify_circle_transport_fit(
        &h_ab,
        ChartTopology::Circle,
        ChartTopology::Circle,
        11,
        17,
        GRID,
    )
    .expect("classify ab");
    let cls_bc = classify_circle_transport_fit(
        &h_bc,
        ChartTopology::Circle,
        ChartTopology::Circle,
        17,
        23,
        GRID,
    )
    .expect("classify bc");
    let cls_ac = classify_circle_transport_fit(
        &h_ac,
        ChartTopology::Circle,
        ChartTopology::Circle,
        11,
        23,
        GRID,
    )
    .expect("classify ac");

    let edges = [
        (cls_ab.winding, cls_ab.phase),
        (cls_bc.winding, cls_bc.phase),
        invert_o2_edge((cls_ac.winding, cls_ac.phase)),
    ];
    let defects = [cls_ab.defect, cls_bc.defect, cls_ac.defect];
    let holo = loop_holonomy(&edges, &defects);

    println!("=== Atlas Machine on REAL {model} transport (n={n} tokens) ===");
    println!(
        "hop {}: winding {:+} phase {:+.4} rad  isometry_defect {:.3e}  |h'|max {:.4}  resid_rms {:.3e}",
        c_ab.name, cls_ab.winding, cls_ab.phase, h_ab.isometry_defect, c_ab.lipschitz, c_ab.defect
    );
    println!(
        "hop {}: winding {:+} phase {:+.4} rad  isometry_defect {:.3e}  |h'|max {:.4}  resid_rms {:.3e}",
        c_bc.name, cls_bc.winding, cls_bc.phase, h_bc.isometry_defect, c_bc.lipschitz, c_bc.defect
    );
    println!(
        "direct L11->L23: winding {:+} phase {:+.4} rad  isometry_defect {:.3e}  resid_rms {:.3e}",
        cls_ac.winding, cls_ac.phase, h_ac.isometry_defect, h_ac.residual_rms
    );
    println!(
        "COMPOSED ε (L11→L17→L23 shadowing bound) = {:.6}  [per-stage {:.6}, {:.6}; domain_ok {}]",
        composed.total_defect,
        composed.per_stage_contribution[0],
        composed.per_stage_contribution[1],
        composed.domain_ok
    );
    println!(
        "LOOP HOLONOMY (triangle h_ab·h_bc·h_ac⁻¹): net_sign {:+}  net_angle {:+.6} rad  \
         tolerance(Σdefect) {:.3e}  is_trivial {}",
        holo.net_sign, holo.net_angle, holo.angle_tolerance, holo.is_trivial
    );
    println!(
        "  → composition law {} at the loop's own defect scale",
        if holo.is_trivial {
            "HOLDS"
        } else {
            "is VIOLATED"
        }
    );

    // Structural invariants only — never a fitted value.
    assert!(composed.total_defect.is_finite() && composed.total_defect >= 0.0);
    let stage_sum: f64 = composed.per_stage_contribution.iter().sum();
    assert!((stage_sum - composed.total_defect).abs() <= 1e-12 * composed.total_defect.max(1.0));
    assert_eq!(composed.per_stage_contribution.len(), 2);
    assert!(c_ab.lipschitz > 0.0 && c_bc.lipschitz > 0.0);
    assert_eq!(holo.loop_len, 3);
    assert!(holo.net_sign == 1 || holo.net_sign == -1);
    assert!(holo.net_angle.is_finite());
    assert!(holo.angle_tolerance.is_finite() && holo.angle_tolerance >= 0.0);
}
