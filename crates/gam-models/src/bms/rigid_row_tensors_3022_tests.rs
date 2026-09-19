//! gam#3022: the rigid row kernel's third and fourth tensors are per-row lazy
//! tables shared through the same-β store. Readers that sweep every row
//! concurrently build each row once, and the table's rows are bit-identical to
//! a direct per-row build.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod rigid_row_tensors_3022_tests;` in
//! `bms/mod.rs` with the allowed `*_tests` name.

use super::exact_eval_cache::RigidRowTensors;
use super::multistart_member_2359_tests::{member, rigid_fixture};
use super::row_kernel::BernoulliRigidRowKernel;
use crate::row_kernel::RowKernel;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Sixteen tasks sweep all rows of one cold table in the same order, the way
/// the ψ-axis fan-out reads it. Every row is built exactly once.
#[test]
fn concurrent_row_sweeps_build_each_row_once_3022() {
    let n_rows = 64usize;
    let table = RigidRowTensors::<f64>::new(n_rows);
    let builds: Vec<AtomicUsize> = (0..n_rows).map(|_| AtomicUsize::new(0)).collect();
    let build = |row: usize| -> Result<f64, String> {
        builds[row].fetch_add(1, Ordering::Relaxed);
        // Widen the window in which another sweep reaches this row mid-build.
        std::thread::sleep(std::time::Duration::from_micros(200));
        Ok(row as f64 * 0.5)
    };
    (0..16usize).into_par_iter().for_each(|_| {
        for row in 0..n_rows {
            let value = *table.row(row, || build(row)).expect("row build");
            assert_eq!(value, row as f64 * 0.5);
        }
    });
    let all = table.all_rows(build).expect("all rows");
    assert_eq!(all.len(), n_rows);
    for (row, count) in builds.iter().enumerate() {
        assert_eq!(count.load(Ordering::Relaxed), 1, "row {row} built more than once");
    }
}

/// A failed row build reaches every reader of that row, and only that row.
#[test]
fn a_failed_row_build_reaches_its_readers_3022() {
    let table = RigidRowTensors::<f64>::new(3);
    let build = |row: usize| {
        if row == 1 {
            Err("row 1 is non-finite".to_string())
        } else {
            Ok(1.0)
        }
    };
    assert_eq!(table.all_rows(build).unwrap_err(), "row 1 is non-finite");
    assert_eq!(table.row(1, || Ok(2.0)).unwrap_err(), "row 1 is non-finite");
    assert_eq!(*table.row(0, || Ok(2.0)).unwrap(), 1.0);
    assert!(table.row(3, || Ok(2.0)).is_err());
}

/// Two kernels at one β read one table; its rows, and the contractions served
/// from them, are bit-identical to a direct per-row build.
#[test]
fn rigid_kernel_tables_match_direct_row_builds_3022() {
    let (family, states) = rigid_fixture();
    let family = member(&family);
    let first = BernoulliRigidRowKernel::new(family.clone(), states.clone());
    let second = BernoulliRigidRowKernel::new(family, states);
    let third = first.all_third_full().expect("third rows");
    let fourth = first.all_fourth_full().expect("fourth rows");
    assert!(std::ptr::eq(first.third_rows(), second.third_rows()));
    assert!(std::ptr::eq(first.fourth_rows(), second.fourth_rows()));
    let (u, v) = ([0.7, -1.3], [-0.4, 0.9]);
    for row in 0..third.len() {
        let t3 = second.row_third_full(row).expect("direct third");
        let t4 = second.row_fourth_full(row).expect("direct fourth");
        assert_eq!(*third[row], t3, "row {row} third tensor");
        assert_eq!(*fourth[row], t4, "row {row} fourth tensor");
        assert_eq!(
            second.row_third_contracted(row, &u).expect("third contraction"),
            super::gradient_paths::contract_third_full(&t3, u[0], u[1]),
        );
        assert_eq!(
            second
                .row_fourth_contracted(row, &u, &v)
                .expect("fourth contraction"),
            super::gradient_paths::contract_fourth_full(&t4, u[0], u[1], v[0], v[1]),
        );
    }
}
