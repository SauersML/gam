use gam::warm_start::{EntryKind, Fingerprint, Fingerprinter, StoreOptions, WarmStartStore};
use std::fs;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

fn key(tag: &str) -> Fingerprint {
    let mut h = Fingerprinter::new();
    h.absorb_str(b"k", tag);
    h.finalize()
}

#[test]
fn fingerprint_is_deterministic_and_distinguishes_typical_inputs() {
    let mut a1 = Fingerprinter::new();
    a1.absorb_bytes(b"payload", b"same-input");
    let fp_a1 = a1.finalize();

    let mut a2 = Fingerprinter::new();
    a2.absorb_bytes(b"payload", b"same-input");
    let fp_a2 = a2.finalize();

    let mut b = Fingerprinter::new();
    b.absorb_bytes(b"payload", b"different-input");
    let fp_b = b.finalize();

    assert_eq!(
        fp_a1, fp_a2,
        "Fingerprint must be deterministic for the same absorbed bytes."
    );
    assert_ne!(
        fp_a1, fp_b,
        "Fingerprint should not collide for distinct typical inputs."
    );
}

#[test]
fn fingerprint_hex_roundtrip_preserves_bits() {
    let mut h = Fingerprinter::new();
    h.absorb_str(b"model", "cache-roundtrip");
    h.absorb_f64_slice(b"x", &[1.5, -0.0, 9.25]);
    let fp = h.finalize();

    let parsed =
        Fingerprint::from_hex(&fp.to_hex()).expect("from_hex must parse the exact to_hex output.");
    assert_eq!(
        parsed, fp,
        "from_hex(to_hex(fp)) must reconstruct the original fingerprint."
    );
}

#[test]
fn store_lookup_returns_inserted_payload_at_same_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = WarmStartStore::open(
        dir.path().to_path_buf(),
        StoreOptions {
            size_budget_bytes: 1 << 20,
            ttl: Duration::from_secs(60),
        },
    )
    .expect("open store");

    let k = key("insert-lookup");
    store
        .save(&k, b"payload-v1", Some(1.0), Some(7), EntryKind::Checkpoint)
        .expect("save should succeed");

    let got = store
        .lookup(&k)
        .expect("lookup should not error")
        .expect("entry should exist");
    assert_eq!(
        got.payload, b"payload-v1",
        "Lookup at the same key must return the inserted payload bytes."
    );
}

#[test]
fn eviction_prefers_recently_hit_entries_under_budget_pressure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = WarmStartStore::open(
        dir.path().to_path_buf(),
        StoreOptions {
            size_budget_bytes: 1600,
            ttl: Duration::from_secs(60),
        },
    )
    .expect("open store");

    let a = key("a");
    let b = key("b");
    let c = key("c");
    let payload = vec![9u8; 600];

    store
        .save(&a, &payload, Some(3.0), None, EntryKind::Checkpoint)
        .expect("save a");
    thread::sleep(Duration::from_millis(5));
    store
        .save(&b, &payload, Some(2.0), None, EntryKind::Checkpoint)
        .expect("save b");

    store.lookup(&a).expect("lookup a");

    thread::sleep(Duration::from_millis(5));
    store
        .save(&c, &payload, Some(1.0), None, EntryKind::Checkpoint)
        .expect("save c");
    store.evict_overflow().expect("evict");

    assert!(
        store.lookup(&a).expect("lookup a after evict").is_some(),
        "A reservation/hit should refresh recency so key A survives LRU eviction."
    );
    assert!(
        store.lookup(&b).expect("lookup b after evict").is_none(),
        "Least recently used key should be evicted first once budget is exceeded."
    );
}

#[test]
fn concurrent_writes_do_not_lose_entries_and_budget_is_respected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(
        WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 4096,
                ttl: Duration::from_secs(60),
            },
        )
        .expect("open store"),
    );

    let threads = 8usize;
    let barrier = Arc::new(Barrier::new(threads));
    let mut joins = Vec::new();

    for i in 0..threads {
        let s = Arc::clone(&store);
        let b = Arc::clone(&barrier);
        joins.push(thread::spawn(move || {
            let k = key(&format!("thread-{i}"));
            let payload = vec![i as u8; 700];
            b.wait();
            s.save(
                &k,
                &payload,
                Some(i as f64),
                Some(i as u64),
                EntryKind::Checkpoint,
            )
            .expect("concurrent save should succeed");
        }));
    }

    for j in joins {
        j.join().expect("writer thread should join cleanly");
    }

    store
        .evict_overflow()
        .expect("evict after concurrent writes");

    let mut total = 0u64;
    for key_dir in fs::read_dir(dir.path()).expect("read root") {
        let key_dir = key_dir.expect("dir entry").path();
        if !key_dir.is_dir() {
            continue;
        }
        for f in fs::read_dir(&key_dir).expect("read key dir") {
            total += f.expect("file").metadata().expect("metadata").len();
        }
    }

    assert!(
        total <= 4096,
        "Final on-disk cache size must not exceed the configured budget after concurrent insertion."
    );
}
