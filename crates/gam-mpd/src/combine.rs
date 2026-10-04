//! Many threads' products with the same large matrix, run as one (#2951).
//!
//! Pricing a library (`super::describe`) multiplies every block's thin factors (a few columns) by
//! the same few large matrices: the site's metric, its roots, a chart's Gram. Block by block each
//! is a matrix-vector product that reads the whole matrix from memory for a handful of
//! multiply-adds, so a few threads saturate the memory bus and the rest wait: a site's pricing ran
//! at about two cores of eighteen, nearly all of it in those products.
//!
//! [`map`] runs a function over many items on more threads than cores, each thread a participant.
//! A participant's [`product`] of a large matrix with a thin operand (on either side) is queued;
//! once every participant has queued one (or finished), the queue runs as one matrix-matrix
//! product per large matrix, its thin operands side by side, on the rayon pool, and each
//! participant takes its columns back. A matrix is the same matrix when its memory is: every
//! queued request holds its operands borrowed until its result comes back, so two requests naming
//! the same address and layout read the same values. Each column of a matrix-matrix product is the
//! matrix-vector product it replaces, up to the summation order of its dot products.
//!
//! Outside [`map`] (or below the size at which it pays), [`product`] is the plain product.

use gam_linalg::faer_ndarray::{fast_ab, with_nested_parallel};
use ndarray::{Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, ShapeBuilder};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// The least entries a matrix has before its products are combined.
const LARGE: usize = 1 << 16;

/// The most columns (or rows) an operand has to count as thin.
const THIN: usize = 16;

/// Participant threads per core.
const PER_CORE: usize = 4;

/// Which operand is the large one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Side {
    /// `A x`: the large matrix on the left, a thin operand of few columns on the right.
    Left,
    /// `x A`: a thin operand of few rows on the left, the large matrix on the right.
    Right,
}

/// A large matrix by its memory: address, shape and strides (in elements).
type Key = (usize, usize, usize, isize, isize, Side);

/// A large matrix's memory as a request holds it (module note).
#[derive(Clone, Copy)]
struct Raw {
    ptr: *const f64,
    rows: usize,
    cols: usize,
    strides: (isize, isize),
}

// SAFETY: a `Raw` names memory its requesting thread keeps borrowed, unmutated, until that request's
// result is handed back (`Combiner::product` blocks until then); other threads only read it.
unsafe impl Send for Raw {}

impl Raw {
    fn of(a: &ArrayView2<'_, f64>) -> Self {
        let strides = a.strides();
        Self { ptr: a.as_ptr(), rows: a.nrows(), cols: a.ncols(), strides: (strides[0], strides[1]) }
    }

    fn key(&self, side: Side) -> Key {
        (self.ptr as usize, self.rows, self.cols, self.strides.0, self.strides.1, side)
    }

    /// The view.
    ///
    /// # Safety
    /// The request that holds this `Raw` must still be pending (module note).
    // SAFETY: an unsafe fn: its callers keep the requesting thread waiting (module note).
    unsafe fn view<'a>(&self) -> ArrayView2<'a, f64> {
        let shape = (self.rows, self.cols).strides((self.strides.0 as usize, self.strides.1 as usize));
        // SAFETY: the memory and layout were a live view's (`Raw::of`, non-negative strides only),
        // and it stays borrowed while the request is pending (the caller's contract).
        unsafe { ArrayView2::from_shape_ptr(shape, self.ptr) }
    }
}

struct Request {
    id: u64,
    side: Side,
    large: Raw,
    /// The thin operand, copied: columns for `Side::Left`, rows for `Side::Right`.
    thin: Array2<f64>,
}

#[derive(Default)]
struct State {
    participants: usize,
    pending: Vec<Request>,
    done: BTreeMap<u64, Array2<f64>>,
    firing: bool,
    next: u64,
}

/// One [`map`]'s queue of products (module note).
#[derive(Default)]
struct Combiner {
    state: Mutex<State>,
    ready: Condvar,
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<Combiner>>> = const { RefCell::new(None) };
}

/// The pool combined products run on (a core a thread), whose workers are free while the
/// participants (threads of their own) wait; `None` when it could not be built.
fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
    match POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().thread_name(|i| format!("combine-{i}")).build().map_err(|e| e.to_string())) {
        Ok(pool) => Some(pool),
        Err(e) => {
            log::warn!("combine: no pool ({e}); combined products run on the thread that runs them");
            None
        }
    }
}

impl Combiner {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Run every queued request (the caller set `firing` and took them).
    fn fire(requests: Vec<Request>) -> Vec<(u64, Array2<f64>)> {
        let mut groups: BTreeMap<Key, Vec<Request>> = BTreeMap::new();
        for r in requests {
            groups.entry(r.large.key(r.side)).or_default().push(r);
        }
        let mut out = Vec::new();
        for (_, group) in groups {
            let side = group[0].side;
            // SAFETY: every request of the group is pending until `out` is handed back.
            let large = unsafe { group[0].large.view() };
            let axis = if side == Side::Left { Axis(1) } else { Axis(0) };
            let views: Vec<_> = group.iter().map(|r| r.thin.view()).collect();
            let joined = match ndarray::concatenate(axis, &views) {
                Ok(joined) => joined,
                Err(e) => {
                    // Operands of one large matrix always join; were they not, each runs alone.
                    log::warn!("combine: {e}; {} products run one by one", group.len());
                    for r in &group {
                        out.push((r.id, if side == Side::Left { fast_ab(&large, &r.thin) } else { fast_ab(&r.thin, &large) }));
                    }
                    continue;
                }
            };
            let multiply = || if side == Side::Left { fast_ab(&large, &joined) } else { fast_ab(&joined, &large) };
            let product = match pool() {
                Some(pool) => pool.install(multiply),
                None => multiply(),
            };
            let mut at = 0;
            for r in &group {
                let width = r.thin.len_of(axis);
                let part = if side == Side::Left { product.slice(ndarray::s![.., at..at + width]) } else { product.slice(ndarray::s![at..at + width, ..]) };
                out.push((r.id, part.to_owned()));
                at += width;
            }
        }
        out
    }

    fn product(&self, side: Side, large: Raw, thin: Array2<f64>) -> Array2<f64> {
        let mut state = self.lock();
        let id = state.next;
        state.next += 1;
        state.pending.push(Request { id, side, large, thin });
        loop {
            if let Some(result) = state.done.remove(&id) {
                return result;
            }
            if !state.firing && !state.pending.is_empty() && state.pending.len() >= state.participants {
                state.firing = true;
                let requests = std::mem::take(&mut state.pending);
                drop(state);
                let results = Self::fire(requests);
                state = self.lock();
                state.done.extend(results);
                state.firing = false;
                self.ready.notify_all();
                continue;
            }
            state = self.ready.wait(state).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn leave(&self) {
        let mut state = self.lock();
        state.participants -= 1;
        if !state.pending.is_empty() && state.pending.len() >= state.participants {
            self.ready.notify_all();
        }
    }
}

/// `a b`, combined with the other participants' products of the same large matrix inside
/// [`map`] (module note), else the plain product.
pub fn product<A: Data<Elem = f64>, B: Data<Elem = f64>>(a: &ArrayBase<A, Ix2>, b: &ArrayBase<B, Ix2>) -> Array2<f64> {
    let combiner = CURRENT.with(|c| c.borrow().clone());
    let (av, bv) = (a.view(), b.view());
    let positive = |v: &ArrayView2<'_, f64>| v.strides().iter().all(|s| *s >= 0);
    if let Some(combiner) = combiner
        && positive(&av)
        && positive(&bv)
    {
        if av.len() >= LARGE && bv.ncols() <= THIN {
            return combiner.product(Side::Left, Raw::of(&av), bv.to_owned());
        }
        if bv.len() >= LARGE && av.nrows() <= THIN {
            return combiner.product(Side::Right, Raw::of(&bv), av.to_owned());
        }
    }
    fast_ab(a, b)
}

/// A participant of a [`map`]: while it lives its thread's products queue on the combiner, and
/// when it goes (its items done, or a panic) the others stop waiting for it.
struct Participant(Arc<Combiner>);

impl Participant {
    fn join(combiner: &Arc<Combiner>) -> Self {
        CURRENT.with(|c| *c.borrow_mut() = Some(Arc::clone(combiner)));
        Self(Arc::clone(combiner))
    }
}

impl Drop for Participant {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = None);
        self.0.leave();
    }
}

/// `f` of every item, on [`PER_CORE`] threads a core (at most one an item), every large product
/// of theirs combined (module note); results in item order.
pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = (PER_CORE * rayon::current_num_threads()).clamp(1, items.len().max(1));
    let combiner = Arc::new(Combiner::default());
    combiner.lock().participants = threads;
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    let results: Vec<Vec<(usize, R)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (combiner, next, f) = (&combiner, &next, &f);
                scope.spawn(move || {
                    let participant = Participant::join(combiner);
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        mine.push((i, with_nested_parallel(|| f(item))));
                    }
                    drop(participant);
                    mine
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    });
    for (i, r) in results.into_iter().flatten() {
        out[i] = Some(r);
    }
    out.into_iter().flatten().collect()
}
