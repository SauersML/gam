#![cfg(test)]
//! Producer/consumer interleavings on [`ProjectedFactorCache`]'s in-progress
//! slots, observed through the cache's own synchronization internals.

use super::*;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Handles held on the in-progress slot for `key`: the cache's own, the
/// producer's, and one per consumer that has subscribed to it.
fn in_progress_handles(cache: &ProjectedFactorCache, key: ProjectedFactorKey) -> usize {
    let inner = cache
        .inner
        .lock()
        .expect("projected factor cache lock poisoned");
    inner.in_progress.get(&key).map_or(0, Arc::strong_count)
}

#[test]
fn projected_factor_cache_waiters_wake_when_producer_panics() {
    let cache = Arc::new(ProjectedFactorCache::with_budget(0));
    let key = ProjectedFactorKey::synthetic(42);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();

    let producer_cache = Arc::clone(&cache);
    let producer = thread::spawn(move || {
        catch_unwind(AssertUnwindSafe(|| {
            producer_cache.get_or_insert_with(key, || {
                started_tx.send(()).expect("send producer-start signal");
                release_rx.recv().expect("receive producer-release signal");
                panic!("simulated projected-factor panic");
            });
        }))
        .is_err()
    });
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("producer started computing");

    let waiter_cache = Arc::clone(&cache);
    let waiter = thread::spawn(move || {
        catch_unwind(AssertUnwindSafe(|| {
            waiter_cache.get_or_insert_with(key, || Array2::from_elem((1, 1), 7.0));
        }))
        .is_err()
    });

    // A consumer has subscribed once it holds its own handle on the slot. Only
    // then is the producer released, so its panic must reach a Wait branch.
    let deadline = Instant::now() + Duration::from_secs(5);
    while in_progress_handles(&cache, key) < 3 {
        assert!(
            Instant::now() < deadline,
            "waiter never subscribed to the in-progress slot"
        );
        thread::sleep(Duration::from_millis(1));
    }
    release_tx.send(()).expect("release producer");

    assert!(producer.join().expect("producer thread joined"));
    assert!(waiter.join().expect("waiter thread joined"));

    let recovered = cache.get_or_insert_with(key, || Array2::from_elem((1, 1), 9.0));
    assert_eq!(recovered[[0, 0]], 9.0);
}
