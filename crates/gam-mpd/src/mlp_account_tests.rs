use super::gates::Unit;
use super::mlp_account::{Account, Rule, canonical, native, prune, refine, refit_writes, share};
use ndarray::{Array1, Array2};

/// A deterministic pseudo-random matrix with entries in `[-scale, scale]`.
fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed | 1;
    Array2::from_shape_fn((rows, cols), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        scale * ((state % 2_000_001) as f64 / 1_000_000.0 - 1.0)
    })
}

fn small_account() -> Account {
    let rules = vec![
        Rule { inputs: vec![0, 2], beta: 0.1, linear: vec![0.3, -0.2], units: vec![Unit { w: vec![1.1, -0.4], d: 0.2, c: 0.7 }] },
        Rule { inputs: vec![1], beta: -0.3, linear: vec![0.5], units: vec![] },
        Rule { inputs: vec![3], beta: 0.0, linear: vec![0.0], units: vec![Unit { w: vec![0.9], d: -0.1, c: 1.3 }, Unit { w: vec![-0.6], d: 0.4, c: -0.5 }] },
    ];
    Account { reads: matrix(4, 5, 7, 0.6), rules, writes: matrix(3, 6, 11, 0.5), offset: Array1::from(vec![0.1, 0.0, -0.2, 0.3, 0.0, 0.05]) }
}

/// The dense map as an account computes the MLP exactly.
#[test]
fn neurons_account_is_the_mlp() {
    let (w_in, w_out) = (matrix(7, 5, 3, 0.8), matrix(6, 7, 5, 0.8));
    let h = matrix(20, 5, 9, 1.5);
    let account = Account::neurons(&w_in, &w_out);
    let gap = (&account.apply(h.view()) - &native(&w_in, &w_out, h.view())).mapv(f64::abs).fold(0.0f64, |m, v| m.max(*v));
    assert!(gap < 1e-12, "{gap}");
}

/// The gauge moves leave the account's output unchanged.
#[test]
fn canonical_keeps_the_output() {
    let mut account = small_account();
    let h = matrix(30, 5, 13, 1.2);
    let before = account.apply(h.view());
    canonical(&mut account);
    let gap = (&account.apply(h.view()) - &before).mapv(f64::abs).fold(0.0f64, |m, v| m.max(*v));
    assert!(gap < 1e-12, "{gap}");
    assert!(account.rules.iter().all(|r| r.beta == 0.0));
}

/// Refitting the writes recovers them when the targets are the account's own output.
#[test]
fn refit_writes_recovers_the_writes() {
    let mut account = small_account();
    let h = matrix(200, 5, 17, 1.2);
    let y = account.apply(h.view());
    account.writes = matrix(3, 6, 19, 0.5);
    account.offset = Array1::zeros(6);
    refit_writes(&mut account, h.view(), y.view()).expect("refit");
    let gap = (&account.apply(h.view()) - &y).mapv(f64::abs).fold(0.0f64, |m, v| m.max(*v));
    assert!(gap < 1e-9, "{gap}");
}

/// Refinement's gradient is the remainder's: from a perturbed start it lowers the remainder.
#[test]
fn refine_lowers_the_remainder() {
    let truth = small_account();
    let h = matrix(300, 5, 23, 1.2);
    let y = truth.apply(h.view());
    let hv = matrix(100, 5, 29, 1.2);
    let yv = truth.apply(hv.view());
    let fisher = {
        let a = matrix(6, 6, 31, 1.0);
        a.t().dot(&a) + Array2::<f64>::eye(6) * 0.1
    };
    let mut account = truth.clone();
    account.reads = &account.reads + &matrix(4, 5, 37, 0.05);
    account.writes = &account.writes + &matrix(3, 6, 41, 0.05);
    let start = account.remainder(h.view(), y.view(), &fisher);
    let refined = refine(&mut account, (h.view(), y.view()), (hv.view(), yv.view()), &fisher, 60, 60);
    assert!(refined.train < 0.05 * start, "{start} -> {}", refined.train);
}

/// Pruning drops an outgoing amplitude that writes nothing, and the read only it read.
#[test]
fn prune_drops_what_pays_nothing() {
    let mut account = small_account();
    account.writes.row_mut(1).fill(0.0);
    let h = matrix(200, 5, 43, 1.2);
    let y = account.apply(h.view());
    let fisher = Array2::<f64>::eye(6);
    let pruned = prune(&mut account, (h.view(), y.view()), (h.view(), y.view()), &fisher, 1e9).expect("prune");
    assert_eq!(pruned.outgoing, 1);
    assert_eq!(pruned.incoming, 1);
    assert!(account.remainder(h.view(), y.view(), &fisher) < 1e-18);
}

/// Neurons share one body.
#[test]
fn neurons_share_one_body() {
    let (w_in, w_out) = (matrix(9, 5, 47, 0.8), matrix(6, 9, 53, 0.8));
    let h = matrix(50, 5, 59, 1.5);
    let mut account = Account::neurons(&w_in, &w_out);
    let y = account.apply(h.view());
    canonical(&mut account);
    let bodies = share(&mut account, h.view(), y.view(), &Array2::eye(6), 1e6);
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0].instances.len(), 9);
}
