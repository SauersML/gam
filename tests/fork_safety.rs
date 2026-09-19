//! Fits never depend on rayon's global pool, so they keep working in a
//! `fork()`ed process.
//!
//! Rayon's global pool is built once per address space and cannot be rebuilt.
//! A forked child inherits it without any of its threads, so work queued there
//! waits forever. Every wait here is under a test-only time bound, so a
//! regression fails instead of hanging the suite.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::os::fd::FromRawFd;
use std::sync::mpsc;
use std::time::Duration;

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// A test-only bound on one fit scenario. The fixed small fits finish in
/// seconds; the regression this guards against never finishes.
const HANG_BOUND: Duration = Duration::from_secs(300);

/// Enough rows that the fit's dense products are large enough for a library
/// to split them over threads of its own.
const ROWS: usize = 2000;

fn gaussian_fixture() -> gam::data::EncodedDataset {
    let headers = vec!["y".to_owned(), "x".to_owned()];
    let rows = (0..ROWS)
        .map(|i| {
            let x = -2.4 + 4.8 * i as f64 / (ROWS - 1) as f64;
            let y = (2.1 * x).sin() + 0.07 * ((i * 37 % 17) as f64 - 8.0);
            StringRecord::from(vec![format!("{y:.17e}"), format!("{x:.17e}")])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode fixture")
}

/// The exact bits of one fit's coefficients and smoothing parameters.
fn fit_bits() -> String {
    gam::init_parallelism();
    let config = FitConfig {
        family: Some("gaussian".into()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=10)", &gaussian_fixture(), &config).expect("fit converges");
    let FitResult::Standard(result) = &result else {
        panic!("unexpected fit result");
    };
    let words = |values: &mut dyn Iterator<Item = f64>| {
        values
            .map(|v| format!("{:016x}", v.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "coefficients={} lambdas={}",
        words(&mut result.fit.blocks.iter().flat_map(|b| b.beta.iter().copied())),
        words(&mut result.fit.lambdas.iter().copied()),
    )
}

/// `fit_bits` on its own thread, failing the test if it does not return
/// within [`HANG_BOUND`].
fn bounded_fit_bits(what: &str) -> String {
    let (sender, receiver) = mpsc::channel();
    let fit = std::thread::spawn(move || drop(sender.send(fit_bits())));
    let bits = receiver
        .recv_timeout(HANG_BOUND)
        .unwrap_or_else(|err| panic!("{what}: fit did not finish in {HANG_BOUND:?}: {err}"));
    fit.join().expect("the fit thread returned");
    bits
}

/// This process's threads, by thread id.
fn thread_names() -> BTreeMap<String, String> {
    std::fs::read_dir("/proc/self/task")
        .expect("list threads")
        .filter_map(|entry| {
            let tid = entry.ok()?.file_name().into_string().ok()?;
            let comm = std::fs::read_to_string(format!("/proc/self/task/{tid}/comm")).ok()?;
            Some((tid, comm.trim().to_owned()))
        })
        .collect()
}

/// The threads started since `before` that are not gam's own pool workers.
///
/// The pool gives each worker its own `gam-` name. An unnamed thread (rayon's
/// global pool, a library's private pool) inherits the name of the thread that
/// started it, so it appears under a foreign name or as a second copy of a
/// worker's name.
fn foreign_threads_since(before: &BTreeMap<String, String>) -> Vec<String> {
    let after = thread_names();
    let mut counts = BTreeMap::<&str, usize>::new();
    for name in after.values() {
        *counts.entry(name).or_default() += 1;
    }
    after
        .iter()
        .filter(|(tid, _)| !before.contains_key(*tid))
        .filter(|(_, name)| !name.starts_with("gam-") || counts[name.as_str()] > 1)
        .map(|(_, name)| name.clone())
        .collect()
}

/// A fit in a forked child produces the parent's fit bit for bit, and every
/// thread either process starts for a fit is one of gam's own workers.
///
/// The parent builds rayon's global pool first, as any host program that uses
/// rayon itself would. The forked child inherits that pool without its
/// workers, so a computation that reached it there would wait forever.
#[test]
fn a_forked_child_fits_exactly_like_its_parent() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .thread_name(|index| format!("host-global-{index}"))
        .build_global()
        .expect("nothing in this process built the global pool before");

    let before = thread_names();
    let parent = bounded_fit_bits("parent");
    assert_eq!(
        foreign_threads_since(&before),
        Vec::<String>::new(),
        "the parent's fit started threads outside gam's pool"
    );

    let mut fds = [0; 2];
    // SAFETY: `fds` has room for the two descriptors `pipe` writes.
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "pipe");
    let [read_fd, write_fd] = fds;
    // SAFETY: the child only fits, reports through the pipe and `_exit`s.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork");
    if pid == 0 {
        let outcome = std::panic::catch_unwind(|| {
            let before = thread_names();
            let bits = fit_bits();
            format!("{bits}\nforeign_threads={:?}", foreign_threads_since(&before))
        });
        let status = match outcome {
            Ok(report) => {
                // SAFETY: `write_fd` is this process's write end of the pipe.
                let mut out = unsafe { std::fs::File::from_raw_fd(write_fd) };
                i32::from(out.write_all(report.as_bytes()).is_err())
            }
            Err(payload) => {
                drop(payload);
                1
            }
        };
        // SAFETY: leave the child without running the harness's teardown.
        unsafe { libc::_exit(status) };
    }
    // SAFETY: `write_fd` belongs to this process and is closed once, here.
    unsafe { libc::close(write_fd) };
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut report = String::new();
        // SAFETY: `read_fd` is this process's read end of the pipe.
        let read = unsafe { std::fs::File::from_raw_fd(read_fd) }.read_to_string(&mut report);
        let mut status = 0;
        // SAFETY: `pid` is this process's child.
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        drop(sender.send((read.map(|len| (len, report)), waited, status)));
    });
    let (read, waited, status) = receiver.recv_timeout(HANG_BOUND).unwrap_or_else(|err| {
        // SAFETY: `pid` is this process's child, still unreaped.
        unsafe { libc::kill(pid, libc::SIGKILL) };
        panic!("the forked child's fit did not finish in {HANG_BOUND:?}: {err}")
    });
    let (len, report) = read.expect("read the child's report");
    assert_eq!(waited, pid, "waitpid");
    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "forked child failed after writing {len} bytes: {report}"
    );
    assert_eq!(
        report,
        format!("{parent}\nforeign_threads=[]"),
        "the child's fit or threads differ from the parent's"
    );
}
