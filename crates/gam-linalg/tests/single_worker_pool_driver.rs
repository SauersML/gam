//! `run_on_single_worker_pool` moves a driver onto a one-thread global pool's
//! worker, so the parallel loops it starts run in place instead of crossing
//! threads. This is its own test binary because the property needs the global
//! pool built with one thread, and a process builds its global pool once.

use gam_linalg::parallel::run_on_single_worker_pool;
use rayon::prelude::*;

#[test]
fn one_thread_pool_drives_on_its_worker_and_leaves_workers_in_place() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .expect("this binary builds the global pool first");
    assert_eq!(
        rayon::current_thread_index(),
        None,
        "the test thread is outside the pool"
    );

    let caller = std::thread::current().id();
    let values: Vec<f64> = (0..257).map(|i| f64::from(i) * 0.5).collect();
    let serial: f64 = values.iter().map(|x| x * x).sum();

    let (index, driver, parallel, nested) = run_on_single_worker_pool(|| {
        let index = rayon::current_thread_index();
        let driver = std::thread::current().id();
        let parallel: f64 = values.par_iter().map(|x| x * x).sum();
        // A driver already on a worker runs a nested call where it stands.
        let nested = run_on_single_worker_pool(|| std::thread::current().id());
        (index, driver, parallel, nested)
    });
    assert_eq!(index, Some(0), "the driver runs on the pool's only worker");
    assert_ne!(driver, caller, "the driver left the calling thread");
    assert_eq!(nested, driver, "a nested call does not hop threads again");
    assert_eq!(
        parallel.to_bits(),
        serial.to_bits(),
        "the loop's result is unchanged"
    );

    // A worker of some other pool is already a pool thread: the call runs in place.
    let wide = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .expect("local pool");
    let (outer, inner) = wide.install(|| {
        let outer = std::thread::current().id();
        (
            outer,
            run_on_single_worker_pool(|| std::thread::current().id()),
        )
    });
    assert_eq!(outer, inner, "a caller inside a pool is never moved");
}
