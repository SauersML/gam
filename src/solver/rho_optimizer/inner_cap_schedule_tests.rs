use super::{InnerProgressSnapshot, first_order_inner_cap_schedule};

fn snap(last_iters: usize, last_converged: bool) -> Option<InnerProgressSnapshot> {
    Some(InnerProgressSnapshot {
        last_iters,
        last_converged,
        last_ift_residual: None,
        last_accept_rho: None,
    })
}

fn snap_with_accept_rho(
    last_iters: usize,
    last_converged: bool,
    accept_rho: f64,
) -> Option<InnerProgressSnapshot> {
    Some(InnerProgressSnapshot {
        last_iters,
        last_converged,
        last_ift_residual: None,
        last_accept_rho: Some(accept_rho),
    })
}

fn snap_with_residual(
    last_iters: usize,
    last_converged: bool,
    residual: f64,
) -> Option<InnerProgressSnapshot> {
    Some(InnerProgressSnapshot {
        last_iters,
        last_converged,
        last_ift_residual: Some(residual),
        last_accept_rho: None,
    })
}

/// The bridge's snapshot reader must distinguish "no signal yet"
/// (NaN sentinel, encoded as `IFT_RESIDUAL_NO_SIGNAL_BITS`) from
/// "residual was 0.0" (a real signal). Previously the bridge used
/// `bits == 0` to detect no-signal, which collided with
/// `f64::to_bits(0.0) == 0`. This test pins down the new
/// NaN-sentinel discipline at the bridge layer.
#[test]
fn snapshot_distinguishes_zero_residual_from_no_signal() {
    use super::InnerProgressFeedback;
    use crate::solver::estimate::reml::outer_eval::IFT_RESIDUAL_NO_SIGNAL_BITS;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

    // Helper to build a feedback channel with concrete values.
    let make_feedback = |iters: usize, converged: bool, residual_bits: u64| InnerProgressFeedback {
        cap: Arc::new(AtomicUsize::new(0)),
        accepted_iter: Arc::new(AtomicUsize::new(0)),
        last_iters: Arc::new(AtomicUsize::new(iters)),
        last_converged: Arc::new(AtomicBool::new(converged)),
        ift_residual: Arc::new(AtomicU64::new(residual_bits)),
        accept_rho: Arc::new(AtomicU64::new(
            crate::solver::estimate::reml::outer_eval::IFT_RESIDUAL_NO_SIGNAL_BITS,
        )),
    };

    // Sentinel → no IFT signal (last_ift_residual = None).
    let fb = make_feedback(5, true, IFT_RESIDUAL_NO_SIGNAL_BITS);
    let snap = fb.snapshot().expect("iters > 0, snapshot present");
    assert!(
        snap.last_ift_residual.is_none(),
        "sentinel must decode to None"
    );

    // 0.0 residual → genuine signal (last_ift_residual = Some(0.0)).
    // This is the bug: previously the reader treated `bits == 0` as
    // no-signal, dropping the genuine 0.0 residual.
    let fb = make_feedback(5, true, 0.0_f64.to_bits());
    let snap = fb.snapshot().expect("iters > 0, snapshot present");
    assert_eq!(
        snap.last_ift_residual,
        Some(0.0),
        "residual of exactly 0.0 must round-trip as a real signal, \
             not be confused with the no-signal sentinel",
    );

    // Modest finite residual round-trips.
    let fb = make_feedback(5, true, 0.05_f64.to_bits());
    let snap = fb.snapshot().expect("snapshot present");
    assert_eq!(snap.last_ift_residual, Some(0.05));

    // last_iters == 0 → entire snapshot is None (no inner-Newton
    // signal yet at all). Sentinel residual irrelevant.
    let fb = make_feedback(0, false, IFT_RESIDUAL_NO_SIGNAL_BITS);
    assert!(fb.snapshot().is_none());
}

#[test]
fn schedule_falls_back_to_iter_tier_without_feedback() {
    // No inner-progress history yet → coarse iter-count fallback so
    // the cold-start cap is shallow even before the adaptive signal
    // arrives.
    assert_eq!(first_order_inner_cap_schedule(0, None, None), 3);
    assert_eq!(first_order_inner_cap_schedule(1, None, None), 5);
    assert_eq!(first_order_inner_cap_schedule(2, None, None), 10);
    assert_eq!(first_order_inner_cap_schedule(20, None, None), 10);
}

#[test]
fn schedule_uses_last_iters_plus_margin_when_converged() {
    // Inner converged in 4 iters last time → cap = 4+2 = 6.
    assert_eq!(first_order_inner_cap_schedule(2, None, snap(4, true)), 6);
    // Inner converged in 12 → cap = 14.
    assert_eq!(first_order_inner_cap_schedule(5, None, snap(12, true)), 14);
}

#[test]
fn schedule_geometric_backoff_when_last_hit_cap() {
    // Last hit cap at 5 → 2*5=10, max(10, 5+4=9) = 10.
    assert_eq!(first_order_inner_cap_schedule(2, None, snap(5, false)), 10);
    // Last hit cap at 1 → 2*1=2, max(2, 1+4=5) = 5.
    assert_eq!(first_order_inner_cap_schedule(2, None, snap(1, false)), 5);
    // Last hit cap at 30 → would be 60 but ceiling is 64, so 60.
    assert_eq!(first_order_inner_cap_schedule(2, None, snap(30, false)), 60);
}

#[test]
fn schedule_clamps_floor_and_ceiling() {
    // Last converged in 0 (degenerate; should never happen because
    // the producer only writes nonzero, but defensively check the
    // floor of 3).
    assert_eq!(first_order_inner_cap_schedule(2, None, snap(0, true)), 3);
    // Last converged in 100 → ceiling 64.
    assert_eq!(first_order_inner_cap_schedule(2, None, snap(100, true)), 64);
}

#[test]
fn schedule_uncaps_when_outer_converged() {
    // g_ratio < 1% trumps everything: cached β must be at full
    // inner tolerance for the convergence guard.
    assert_eq!(first_order_inner_cap_schedule(0, Some(0.0001), None), 0);
    assert_eq!(
        first_order_inner_cap_schedule(0, Some(0.005), snap(4, true)),
        0
    );
    assert_eq!(
        first_order_inner_cap_schedule(20, Some(0.001), snap(50, false)),
        0
    );
}

#[test]
fn schedule_ignores_modest_g_ratio_decay() {
    // Old schedule had tiered ratio caps at 0.50/0.20/0.05; the new
    // schedule only special-cases the deep-convergence threshold
    // (<1%). Modest decay no longer overrides the adaptive cap.
    assert_eq!(
        first_order_inner_cap_schedule(2, Some(0.30), snap(4, true)),
        6
    );
    assert_eq!(
        first_order_inner_cap_schedule(2, Some(0.05), snap(4, true)),
        6
    );
}

#[test]
fn schedule_uses_ift_residual_to_pick_margin() {
    // Excellent IFT prediction (residual < 0.01): warm-start lands
    // essentially AT the KKT β, so +1 of margin suffices.
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.005)),
        5
    );
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.0001)),
        5
    );
    // Default zone (0.01 ≤ residual < 0.10): +2, current behavior.
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.05)),
        6
    );
    // Poor IFT prediction (residual ≥ 0.10): +4, the inner Newton
    // has more recovery work after a near-flat warm-start.
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.20)),
        8
    );
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, 0.80)),
        8
    );
    // Margin policy is monotone non-decreasing in residual: a worse
    // predictor never produces a tighter cap than a better one.
    let residuals = [0.001, 0.05, 0.30];
    let caps: Vec<usize> = residuals
        .iter()
        .map(|&r| first_order_inner_cap_schedule(2, None, snap_with_residual(4, true, r)))
        .collect();
    for w in caps.windows(2) {
        assert!(
            w[0] <= w[1],
            "ift-residual margin policy regressed monotonicity: {caps:?}"
        );
    }
}

#[test]
fn schedule_bumps_margin_on_poor_lm_accept_rho() {
    // Healthy LM model fidelity (accept_rho ≥ 0.5): margin
    // unchanged from the no-accept-rho baseline (+2 default).
    // last_iters=4, default margin=2 → cap=6.
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.95)),
        6
    );
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.5)),
        6
    );
    // Poor LM model fidelity (accept_rho < 0.5): +2 margin bump
    // beyond the IFT-residual base. last_iters=4, default base=2,
    // accept_rho<0.5 bump=+2 → margin=4 → cap=8.
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.4)),
        8
    );
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.1)),
        8
    );
    // accept_rho saturation guard: `r < 0.5` is the strict
    // textbook "good agreement" cutoff for trust-region gain
    // ratios. Boundary at 0.5 admits, just below 0.5 bumps.
    assert_eq!(
        first_order_inner_cap_schedule(2, None, snap_with_accept_rho(4, true, 0.49)),
        8
    );
}

#[test]
fn schedule_escalates_geometric_backoff_on_very_poor_accept_rho() {
    // Cap-hit (last_converged=false) with VERY poor LM model
    // (accept_rho < 0.3): triple instead of double the cap, so the
    // next solve has materially more iter budget when the model is
    // both insufficient (cap-hit) AND mis-calibrated (poor rho).
    // last_iters=4 → 4*3 = 12, vs 4*2=8 with doubling.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: false,
        last_ift_residual: None,
        last_accept_rho: Some(0.15),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 12);
    // Cap-hit with moderately-poor accept_rho (0.3 ≤ r < 0.5):
    // standard doubling. The threshold for escalation is 0.3, not
    // 0.5, because the +2-margin path (commit 04b30163) already
    // covers the 0.3-0.5 case for the converged branch.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: false,
        last_ift_residual: None,
        last_accept_rho: Some(0.4),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
    // Cap-hit with healthy accept_rho ≥ 0.5: standard doubling.
    // The previous solve hit the cap because it needed more iters,
    // not because the LM was mis-calibrated.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: false,
        last_ift_residual: None,
        last_accept_rho: Some(0.9),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
    // Cap-hit with no accept_rho signal: standard doubling. No
    // escalation when we don't have evidence of LM trouble.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: false,
        last_ift_residual: None,
        last_accept_rho: None,
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
    // Boundary at exactly 0.3: NOT escalated (`< 0.3` is strict).
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: false,
        last_ift_residual: None,
        last_accept_rho: Some(0.3),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 8);
}

#[test]
fn schedule_skips_lm_accept_rho_bump_when_signal_absent() {
    // None for last_accept_rho means "no signal yet" — the schedule
    // must NOT bump the margin in that case (otherwise a fresh
    // surface with the NaN sentinel would get penalty cap inflation
    // for no reason). last_iters=4, last_ift_residual=None →
    // default base margin=2 → cap=6, regardless of accept_rho being
    // unset.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: true,
        last_ift_residual: None,
        last_accept_rho: None,
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 6);
    // Regression-lock the boundary: accept_rho exactly at 0.5
    // (textbook good-agreement cutoff) does NOT bump (`< 0.5` is
    // strict). cap = 4 + 2 = 6.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: true,
        last_ift_residual: None,
        last_accept_rho: Some(0.5),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 6);
    // accept_rho = 1.0 is the textbook "perfect agreement" — never
    // bumps. cap = 4 + 2 = 6.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: true,
        last_ift_residual: None,
        last_accept_rho: Some(1.0),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 6);
}

#[test]
fn schedule_combines_ift_residual_and_lm_accept_rho() {
    // When BOTH signals fire (poor IFT prediction AND poor LM
    // accept_rho), the bumps compose: IFT base = 4, accept_rho
    // bump = +2 → total margin = 6, cap = last_iters + 6.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: true,
        last_ift_residual: Some(0.30),
        last_accept_rho: Some(0.20),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 10);
    // When only LM accept_rho is poor (IFT residual is excellent),
    // the bumps still compose: IFT base = 1 (excellent), accept_rho
    // bump = +2 → margin = 3, cap = 4 + 3 = 7.
    let snap = Some(InnerProgressSnapshot {
        last_iters: 4,
        last_converged: true,
        last_ift_residual: Some(0.005),
        last_accept_rho: Some(0.30),
    });
    assert_eq!(first_order_inner_cap_schedule(2, None, snap), 7);
}
