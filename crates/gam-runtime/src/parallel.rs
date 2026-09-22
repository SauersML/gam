//! The process's worker pool, and the top-level test every parallel gate asks.
//!
//! # Why gam owns its pool
//!
//! Rayon's global pool is built once per address space and can never be
//! rebuilt. A `fork()`ed child inherits the registry but none of its threads,
//! so the child's first parallel operation queues work for workers that do not
//! exist and waits forever. That is every fork-based caller of an imported
//! `gamfit` (a `multiprocessing` fork pool, joblib's fork backend, a bare
//! `os.fork()`) once the parent has fitted anything.
//!
//! gam therefore never touches the global pool. [`install`] runs a computation
//! on a pool that belongs to the current *process*: it is built on first use
//! and keyed by the process id, so a forked child sees a pool built by another
//! process and builds its own. The parent's pool object is leaked in the child,
//! never dropped: its threads do not exist there, and its teardown would wait
//! on synchronisation state whose owners stopped existing at the fork.
//!
//! Every parallel entry point — the Python FFI boundary, the CLI, the Rust fit
//! entry points — runs its computation through [`install`], so every
//! `par_iter`, `rayon::join`, faer `Par::rayon` product and ndarray parallel
//! zip inside it lands on this pool.
//!
//! # Top level
//!
//! Many kernels fan out only when they are not already running inside a
//! parallel region, so a row reduction nested in a per-block `par_iter` stays
//! serial instead of oversubscribing the pool. That used to be asked as
//! `rayon::current_thread_index().is_none()`: the calling thread was never a
//! pool worker. Under [`install`] the whole computation runs on a pool worker,
//! so that question no longer separates the two cases. [`at_top_level`] asks
//! the question that was always meant: the current thread is running
//! computation that no enclosing parallel operation has handed out.
//!
//! [`install`] marks the worker it runs on as a root. A kernel that fans out
//! wraps its parallel operation in [`fan_out`], which clears the mark for the
//! operation's duration: the pieces the root worker runs inline and the work
//! it steals while it waits are nested work. [`as_root`] marks a task spawned
//! into the pool as a root of its own, for callers that run independent whole
//! computations (multistart seeds, topology candidates) as peers.
//!
//! A thread that is not a pool worker at all is at top level, as before, and a
//! worker of some other rayon pool is nested, as before.
//!
//! # Threads outside rayon
//!
//! ndarray's dense products run through `matrixmultiply`, which keeps a
//! thread tree of its own; [`confine_matrixmultiply_to_the_pool`] keeps that
//! tree empty, so no gam computation starts a thread outside the pool.

use std::any::Any;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// Stack reserved for each worker of the process pool.
///
/// Every worker can run a whole fit, not only the deep kernels a fit fans
/// out: [`install`] runs the computation itself on a worker. The fit drivers
/// keep large fixed-size structures on the call stack — the survival
/// location-scale row kernel contracts a `Tower4<9>` jet program (9⁴
/// fourth-order entries, ≈59 KiB per scalar held by value, several towers live
/// at once), and the SAE outer-ρ per-row jet loop holds multi-megabyte
/// `Tower4<16>` frames. gam#2967 bounds those frames at their cause: every
/// split-level primitive calls its caller's closure only through an
/// `#[inline(never)]` leaf owner, so each large row program is live at most
/// once per stack, and the root `build.rs` refuses a regression. The worker
/// therefore carries the 64 MiB stack rayon's global-pool workers had, not a
/// wider one that would hide the next deep frame instead of exposing it: a
/// path that overflows here is a frame defect under gam#2967's rule.
pub const WORKER_STACK_SIZE: usize = 64 << 20;

/// The pool, together with the process that built it.
struct ProcessPool {
    pid: u32,
    pool: rayon::ThreadPool,
}

/// The current process's pool. A lock-free cell: a lock held by a parent
/// thread at the moment of a `fork()` would stay held forever in the child.
static POOL: AtomicPtr<ProcessPool> = AtomicPtr::new(std::ptr::null_mut());

thread_local! {
    /// Whether this thread is running a root computation that no enclosing
    /// parallel operation handed out.
    static ROOT: Cell<bool> = const { Cell::new(false) };
}

fn build_pool() -> rayon::ThreadPool {
    confine_matrixmultiply_to_the_pool();
    // The worker count is rayon's own default: `RAYON_NUM_THREADS` when set,
    // otherwise the available parallelism.
    rayon::ThreadPoolBuilder::new()
        .thread_name(|index| format!("gam-worker-{index}"))
        .stack_size(WORKER_STACK_SIZE)
        .build()
        // SAFETY: building fails only when the OS refuses to start threads;
        // no computation can run without them, and rayon's own global pool
        // panics the same way in that case.
        .unwrap_or_else(|err| panic!("gam: cannot start the worker pool: {err}"))
}

/// Whether this process has told `matrixmultiply` to keep every product on
/// its calling thread. A plain flag rather than a `Once`: a `Once` caught
/// mid-call by a `fork()` would block the child forever.
static MATRIXMULTIPLY_CONFINED: AtomicBool = AtomicBool::new(false);

/// Keep every `matrixmultiply` product on the thread that issues it.
///
/// ndarray's dense products (`.dot`, `general_mat_mul`) run through
/// `matrixmultiply`, and the dependency graph builds it with its `threading`
/// feature: burn's ndarray backend enables it, and burn comes in through
/// general-mcmc. With that feature the first product large enough to split
/// starts a private tree of up to three threads, sized from the physical core
/// count or `MATMUL_NUM_THREADS`, outside the process pool. Those threads are
/// the hazard the pool exists to remove: a forked child inherits the tree but
/// not its threads, and its first split product hands half its work to a
/// thread that does not exist and waits for it forever. They also oversubscribe
/// the pool, because a product issued from a worker adds threads of its own.
///
/// The tree is sized once, from `MATMUL_NUM_THREADS`, when the process runs
/// its first product. Setting the variable to 1 before then builds a tree with
/// no threads: each product runs whole on its calling thread — a pool worker
/// under [`install`] — and all of gam's parallelism is the pool's. The number
/// of threads is still chosen by the pool alone. `matrixmultiply` splits a
/// product only over blocks of its output, never over the summed dimension, so
/// the products' results are the same bits either way.
///
/// `gam::init_parallelism` calls this at every entry point's start (the Python
/// module init, the CLI's `main`), and the first build of the process pool
/// calls it for Rust callers that never ran `init_parallelism`.
pub fn confine_matrixmultiply_to_the_pool() {
    if MATRIXMULTIPLY_CONFINED.swap(true, Ordering::AcqRel) {
        return;
    }
    // SAFETY: `set_var` is unsound only while another thread reads or writes
    // the environment through libc at the same moment; Rust's own environment
    // access is serialised by std's lock. This one-time write runs before gam
    // has started any thread: from `init_parallelism` at the start of the
    // Python module init and the CLI, or just before the process pool's
    // workers are first spawned.
    unsafe { std::env::set_var("MATMUL_NUM_THREADS", "1") };
}

/// The pool that belongs to the current process, built on first use.
pub fn pool() -> &'static rayon::ThreadPool {
    let pid = std::process::id();
    loop {
        let current = POOL.load(Ordering::Acquire);
        // SAFETY: a non-null pointer in `POOL` came from `Box::into_raw` and is
        // never freed (a replaced pool is leaked on purpose), so it is valid
        // for the rest of the process.
        if let Some(entry) = unsafe { current.as_ref() }
            && entry.pid == pid
        {
            return &entry.pool;
        }
        let fresh = Box::into_raw(Box::new(ProcessPool {
            pid,
            pool: build_pool(),
        }));
        match POOL.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire) {
            // A pool built by the parent of a forked process stays where it is:
            // leaked, because its threads are gone and dropping it would wait
            // on state they owned.
            Ok(_) => {
                // SAFETY: `fresh` was just published and is never freed.
                return unsafe { &(*fresh).pool };
            }
            Err(_) => {
                // Another thread of this process published first. Ours has never
                // been shared, so it is torn down normally; the loop returns the
                // published one.
                // SAFETY: `fresh` came from `Box::into_raw` above and was not
                // published.
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
    }
}

/// The number of workers in the process pool.
pub fn num_threads() -> usize {
    pool().current_num_threads()
}

/// Restores a thread's root mark when the scope that set it ends, including by
/// unwinding.
struct RootMark {
    previous: bool,
}

impl RootMark {
    fn set(root: bool) -> Self {
        Self {
            previous: ROOT.replace(root),
        }
    }
}

impl Drop for RootMark {
    fn drop(&mut self) {
        ROOT.set(self.previous);
    }
}

/// Moves a carried thread-local's content in and out of the current thread's
/// slot: stores the argument and returns what the slot held.
pub type CarriedLocalSwap = fn(Option<Box<dyn Any + Send>>) -> Option<Box<dyn Any + Send>>;

/// One thread-local that follows a computation across [`install`]'s hop.
struct CarriedLocal {
    swap: CarriedLocalSwap,
    next: AtomicPtr<CarriedLocal>,
}

/// Every carried thread-local, as an append-only list. Lock-free for the same
/// reason [`POOL`] is: a lock a parent thread held at a `fork()` stays held in
/// the child. Entries are leaked, never freed.
static CARRIED_LOCALS: AtomicPtr<CarriedLocal> = AtomicPtr::new(std::ptr::null_mut());

/// Make one thread-local follow every computation [`install`] moves onto the
/// pool.
///
/// # Why
///
/// A thread-local a caller sets on its own thread is invisible to the worker
/// [`install`] runs the computation on. The outer-evidence channels are
/// exactly that: a test arms a probe on its own thread, calls a fit entry
/// point, and reads what the fit published. Since the entry points began
/// hopping onto the pool, a probe armed that way was never lent and an
/// audit was never written, and nothing reported it (gam#4566). A registered
/// channel is taken from the caller before the hop, swapped into the worker
/// for the computation's duration, and handed back afterwards. What the
/// computation left in it is returned to the caller, and the worker keeps
/// what it held.
///
/// The worker's previous content is restored rather than cleared because a
/// worker waiting inside one root computation can steal another's, so the two
/// nest on its stack.
///
/// This is a registration hook rather than a fixed list only because the
/// channels belong to crates above this one; the registry holds exactly what
/// they register. Today that is two channels, both in
/// `gam_solve::estimate::outer_eval_capture`: the outer-seed observer and the
/// ρ-block audit.
///
/// Register each channel once per process. `swap` must store its argument in
/// the channel and return what the channel held.
pub fn carry_across_install(swap: CarriedLocalSwap) {
    let entry = Box::into_raw(Box::new(CarriedLocal {
        swap,
        next: AtomicPtr::new(std::ptr::null_mut()),
    }));
    loop {
        let head = CARRIED_LOCALS.load(Ordering::Acquire);
        // SAFETY: `entry` came from `Box::into_raw` above and is not published
        // until the exchange below succeeds.
        unsafe { (*entry).next.store(head, Ordering::Relaxed) };
        if CARRIED_LOCALS
            .compare_exchange(head, entry, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return;
        }
    }
}

/// The registered channels, newest first.
fn carried_locals() -> impl Iterator<Item = &'static CarriedLocal> {
    let mut cursor = CARRIED_LOCALS.load(Ordering::Acquire);
    std::iter::from_fn(move || {
        // SAFETY: every published entry came from `Box::into_raw` and is never
        // freed, so it is valid for the rest of the process.
        let entry = unsafe { cursor.as_ref() }?;
        cursor = entry.next.load(Ordering::Acquire);
        Some(entry)
    })
}

/// A carried channel's content, paired with its channel.
type CarriedContent = Vec<(&'static CarriedLocal, Option<Box<dyn Any + Send>>)>;

/// Swap each channel's content into this thread and return what it held.
fn swap_carried(content: CarriedContent) -> CarriedContent {
    content
        .into_iter()
        .map(|(local, value)| (local, (local.swap)(value)))
        .collect()
}

/// Puts a worker's own channel content back if the computation unwinds.
struct WorkerCarried {
    previous: Option<CarriedContent>,
}

impl Drop for WorkerCarried {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            swap_carried(previous);
        }
    }
}

/// Run `op` on the process pool, as a root computation.
///
/// From a thread outside every rayon pool, `op` runs on a worker of the
/// process pool and the caller blocks until it returns (a panic in `op` is
/// resumed on the caller). From a thread that is already a rayon worker, `op`
/// runs inline: the computation is already on a pool, and its parallel
/// operations stay there.
///
/// # Invariant: a registered thread-local reads the same on either side of the hop
///
/// For every channel registered with [`carry_across_install`]:
/// - `op` sees exactly what the caller's thread held when `install` was called;
/// - the caller sees exactly what `op` left when `install` returns;
/// - the worker holds afterwards exactly what it held before, including when
///   `op` unwinds, so nothing one computation armed can reach the next
///   computation that worker runs.
///
/// Every other thread-local is the worker's own, as for any rayon job. The
/// inline path needs nothing: caller and computation share a thread.
pub fn install<R: Send>(op: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return op();
    }
    // Every channel travels, an empty one included: a worker waiting inside one
    // root computation can steal this one, and must not lend it the other's.
    let outgoing: CarriedContent = carried_locals()
        .map(|local| (local, (local.swap)(None)))
        .collect();
    let (out, returning) = pool().install(|| {
        let mut worker = WorkerCarried {
            previous: Some(swap_carried(outgoing)),
        };
        let root = RootMark::set(true);
        let out = op();
        drop(root);
        let previous = worker
            .previous
            .take()
            .expect("the worker's own channel content is held until the computation returns");
        let returning = swap_carried(previous);
        (out, returning)
    });
    swap_carried(returning);
    out
}

/// Whether the current thread runs computation that no enclosing parallel
/// operation handed out, so a kernel here may fan out over the pool.
///
/// A thread outside every rayon pool is at top level. A pool worker is at top
/// level while it runs a root computation ([`install`], [`as_root`]) outside
/// every [`fan_out`] scope.
#[inline]
pub fn at_top_level() -> bool {
    rayon::current_thread_index().is_none() || ROOT.get()
}

/// Run `op`, a parallel operation issued from top level, with the current
/// thread marked as nested for its duration: the pieces of `op` this thread
/// runs inline, and the work it steals while it waits, are nested work.
#[inline]
pub fn fan_out<R>(op: impl FnOnce() -> R) -> R {
    with_top_level(false, op)
}

/// Run `op` as a root computation of its own on the current pool worker:
/// kernels inside it fan out exactly as they would under [`install`].
///
/// For independent whole computations spawned into the pool as peers (for
/// example one task per multistart seed), each of which should parallelise
/// internally the way a lone computation would.
#[inline]
pub fn as_root<R>(op: impl FnOnce() -> R) -> R {
    with_top_level(true, op)
}

/// Run `op` on the current pool worker standing where its spawner stood:
/// a root when `top_level` (the spawner's [`at_top_level`], read before it
/// fanned out), nested otherwise.
///
/// For peer computations a caller hands to the pool, each of which should
/// parallelise exactly as the caller itself would have: a multistart issued
/// from a root runs every seed as a root, one issued from nested work runs
/// every seed nested.
#[inline]
pub fn with_top_level<R>(top_level: bool, op: impl FnOnce() -> R) -> R {
    let mark = RootMark::set(top_level);
    let out = op();
    drop(mark);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn install_runs_on_the_process_pool() {
        let (index, name) = install(|| {
            (
                rayon::current_thread_index(),
                std::thread::current().name().map(str::to_string),
            )
        });
        assert!(index.is_some());
        assert!(name.is_some_and(|name| name.starts_with("gam-worker-")));
        assert_eq!(install(rayon::current_num_threads), num_threads());
    }

    #[test]
    fn install_marks_the_root_and_fan_out_marks_its_pieces_nested() {
        assert!(at_top_level(), "a thread outside every pool is at top level");
        let (root, pieces, after) = install(|| {
            let root = at_top_level();
            let pieces: Vec<bool> = fan_out(|| {
                (0..64)
                    .into_par_iter()
                    .map(|_| at_top_level())
                    .collect()
            });
            (root, pieces, at_top_level())
        });
        assert!(root);
        assert!(pieces.iter().all(|&top| !top), "fanned-out pieces are nested");
        assert!(after, "the root mark is restored after the fan-out");
    }

    #[test]
    fn as_root_tasks_are_roots_and_a_foreign_pool_is_nested() {
        let peers: Vec<bool> = install(|| {
            fan_out(|| {
                (0..8)
                    .into_par_iter()
                    .map(|_| as_root(at_top_level))
                    .collect()
            })
        });
        assert!(peers.iter().all(|&top| top));

        let foreign = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool");
        let (outer, inner) = foreign.install(|| (at_top_level(), install(at_top_level)));
        assert!(!outer && !inner, "a worker of another pool stays nested");
    }

    thread_local! {
        static CARRIED_PROBE: std::cell::RefCell<Option<u64>> = const { std::cell::RefCell::new(None) };
        static UNCARRIED_PROBE: std::cell::RefCell<Option<u64>> = const { std::cell::RefCell::new(None) };
    }

    fn swap_carried_probe(value: Option<Box<dyn Any + Send>>) -> Option<Box<dyn Any + Send>> {
        let incoming = value.map(|value| {
            *value
                .downcast::<u64>()
                .expect("the probe carrier only ever carries its own u64")
        });
        CARRIED_PROBE
            .with(|slot| slot.replace(incoming))
            .map(|held| Box::new(held) as Box<dyn Any + Send>)
    }

    /// gam#4566: a channel set on the caller's thread reaches the worker the
    /// computation runs on, and what the computation writes comes back. The
    /// uncarried twin is the control: it shows the hop is real, so the carried
    /// one is not passing because the computation stayed on the caller.
    #[test]
    fn a_carried_thread_local_crosses_install_both_ways_4566() {
        carry_across_install(swap_carried_probe);
        CARRIED_PROBE.with(|slot| *slot.borrow_mut() = Some(7));
        UNCARRIED_PROBE.with(|slot| *slot.borrow_mut() = Some(7));
        let (carried_seen, uncarried_seen) = install(|| {
            let seen = (
                CARRIED_PROBE.with(|slot| *slot.borrow()),
                UNCARRIED_PROBE.with(|slot| *slot.borrow()),
            );
            CARRIED_PROBE.with(|slot| *slot.borrow_mut() = Some(11));
            seen
        });
        assert_eq!(uncarried_seen, None, "the computation must run off the caller's thread");
        assert_eq!(carried_seen, Some(7), "the carried channel reaches the worker");
        assert_eq!(
            CARRIED_PROBE.with(|slot| *slot.borrow()),
            Some(11),
            "what the computation wrote comes back to the caller"
        );
        let worker_after = install(|| CARRIED_PROBE.with(|slot| *slot.borrow()));
        assert_eq!(
            worker_after,
            Some(11),
            "the next computation is handed the caller's current content, not a worker leftover"
        );
    }

    #[test]
    fn a_panic_in_install_reaches_the_caller() {
        let caught = std::panic::catch_unwind(|| install(|| panic!("boom")));
        assert!(caught.is_err());
        assert!(install(at_top_level));
    }
}
