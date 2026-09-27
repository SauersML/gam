//! Host-wide memory pool: one [`MemoryGovernor`](crate::resource::MemoryGovernor)
//! budget shared by every gam process of one user on a host that has no binding
//! cgroup ceiling.
//!
//! # Why
//!
//! The governor's budget is 3/4 of this process's stationary *capacity*, not of
//! free memory, so that a reservation verdict never depends on what the rest of
//! the box is doing (#2684, #2702). Under a job scheduler that is exactly right:
//! the job's cgroup hard limit is the capacity, and the cgroup bounds the *sum*
//! of every process in the job. With no cgroup ceiling (macOS always, bare Linux
//! workstations) capacity is the whole host, and nothing bounds the sum: each
//! of N parallel processes (nextest runs up to `num_cpus` test binaries) may
//! reserve 3/4 of RAM. Measured: three test processes on a 64 GiB Mac held
//! 8.1 + 5.5 + 4.2 GB beside a build and the machine froze.
//!
//! # What
//!
//! When capacity is the host's own total, the governor keeps its per-process
//! ledger (which alone decides verdicts, exactly as before) and additionally
//! *covers* its live reservations with a lease on one pool shared by all gam
//! processes of the user: the same 3/4-of-capacity budget, held in a small JSON
//! ledger next to an OS file lock in the user's runtime/temp directory. Each
//! holder record is `(pid, start time, ticket, held, pending, label)`.
//!
//! * **Verdicts do not move (#2702).** A request is refused iff this process's
//!   own live reservations plus the request exceed the budget — the per-process
//!   ledger's verdict, computed before the pool is consulted. An admissible
//!   request therefore fits an *empty* pool (the lease target is clamped to the
//!   budget), and if the pool is currently full it **waits** instead of being
//!   refused. Neighbours change only when a reservation is granted, never
//!   whether.
//! * **Order, not timeouts (SPEC 20).** A process entry gets a monotone ticket
//!   when it first holds or waits, and keeps it until it holds nothing and waits
//!   for nothing. A request is granted by fit only if it also leaves room for
//!   every *more senior* waiter's pending bytes (conservative backfill), so a
//!   large senior request cannot be starved by a stream of small junior ones.
//!   Nothing sleeps or polls: a waiter blocks on its own datagram "bell",
//!   which every transaction that changes the ledger rings for every other
//!   waiter, and which a watcher thread rings when a holder the waiter is
//!   stuck behind ends its record or dies. The bell is bound before the waiter
//!   is marked pending, so no ring that could admit it is lost.
//! * **Crashed holders are reclaimed.** A process holds an exclusive OS lock
//!   on its own presence file (named by pid *and* start time) for as long as
//!   its record exists. The kernel drops that lock when the process dies, so
//!   whenever a request does not fit, a record whose presence lock can be
//!   taken shared is dead and is removed; a recycled pid has another start
//!   time, hence another presence file, and never inherits a dead holder's
//!   bytes. The ledger lock is the same kind of lock, so a crash never wedges
//!   the ledger either. Unix only (the bell is a Unix datagram socket);
//!   elsewhere the governor stays process-local.
//!
//! # Deadlock freedom
//!
//! Waiting while holding is the Coffman condition that makes a shared pool
//! deadlock: process P holds `a` and waits for `b`, Q holds `c` and waits for
//! `d`, and neither fits until the other releases. The alternatives are:
//!
//! 1. **Refuse the holding waiter.** Its per-process ledger would have granted
//!    the request, so a refusal here makes the verdict a function of the
//!    neighbours — precisely the #2702 defect (kilobyte refusals that pass in
//!    one test subset and fail in another). Rejected.
//! 2. **Banker's algorithm.** Every process's maximum claim is the whole
//!    budget, so the only safe states hold one process at a time: full
//!    host-wide serialization of every governed allocation. Rejected.
//! 3. **Detect the stall and break it by ordering.** Adopted.
//!
//! A holder record is *blocked* if it is waiting (`pending > 0`) or it is an
//! OS ancestor of a waiting process (a test or driver waiting on the `gam`
//! child it spawned cannot release until that child finishes). The pool is
//! **stalled** when every record that holds bytes is blocked: nothing will ever
//! be released, so waiting longer cannot help. In a stall, and only then:
//!
//! * the most senior waiter whose request fits the pool as it stands (ignoring
//!   seniority reservations) is granted — this resolves the common case where a
//!   senior waits for a junior holder's release while the junior waits behind
//!   the senior's reservation;
//! * if no waiter fits even that, the most senior waiter is granted **beyond
//!   the pool** and a warning names the holders.
//!
//! Every waiter evaluates the same predicate on the same locked ledger, and the
//! first grant clears its `pending`, which ends the stall, so exactly one waiter
//! breaks it. The most senior waiter always progresses (it fits, or a stall
//! grants it), so every waiter eventually becomes the most senior: no deadlock
//! and no starvation, with no deadline anywhere.
//!
//! The price is a bounded overdraw that occurs only in a true hold-and-wait
//! cycle. Every grant outside a stall-overdraw leaves the pool within budget,
//! so juniors' leases sum to at most the budget; without ancestor holders the
//! overdraw recipient is the most senior record (every holder is waiting, so
//! the most senior waiter is the most senior holder), whose own lease is at most
//! the budget. Total held is therefore at most twice the budget, and equal to
//! the budget whenever no stall is in progress — against `N` times the budget
//! with independent per-process ledgers.
//!
//! What the pool cannot see is a holder that is alive but idle (an interactive
//! session keeping a cache warm). Its bytes are really resident, so a request
//! that needs them waits for them; the wait is announced with the dominant
//! holders so the cause is visible.
//!
//! # Cost
//!
//! Governed allocations include small, hot cache entries (cell-moment memos
//! reserve per entry inside `par_iter`), so a file transaction per reservation
//! is not affordable. The process instead leases from the pool in quanta of
//! 1/1024 of the budget: a reservation that stays within the current lease is
//! one atomic load, and the lease is returned (with one quantum of hysteresis)
//! as the process's own reservations shrink, entirely when they reach zero.
//! Unused lease is at most two quanta per process, i.e. N/512 of the budget for
//! N processes.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Bumped when the ledger's on-disk shape changes; a ledger of another version
/// is discarded and rebuilt by its live holders (each rewrites its own absolute
/// lease on its next transaction).
const LEDGER_FORMAT_VERSION: u32 = 1;

/// The one on-disk ledger. Every field of a [`HolderRecord`] is owned by the
/// process it describes and written as an absolute value, so a ledger rebuilt
/// from scratch heals as holders touch it; other processes only ever delete
/// records whose process is gone.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LedgerFile {
    version: u32,
    budget_bytes: u64,
    next_ticket: u64,
    holders: Vec<HolderRecord>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct HolderRecord {
    pid: u32,
    /// Seconds since the epoch, as the OS reports it: `(pid, start_time)` names
    /// one process, so a recycled pid is not mistaken for a dead holder.
    start_time: u64,
    /// Seniority. Assigned when the record is created and kept until the
    /// process holds nothing and waits for nothing.
    ticket: u64,
    held_bytes: u64,
    /// Bytes this process is waiting to add to `held_bytes`; zero when it is
    /// not waiting.
    pending_bytes: u64,
    label: String,
    /// While waiting: this process's OS ancestor chain. An ancestor that holds
    /// bytes is presumed to be waiting on this process and counts as blocked.
    waiting_ancestors: Vec<u32>,
}

/// Answers, for each `(pid, start_time)`, whether that exact process still
/// holds its record.
pub(crate) type LivenessProbe<'probe> = dyn FnMut(&[(u32, u64)]) -> Vec<bool> + 'probe;

/// One process, as the pool identifies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ProcessIdentity {
    pub(crate) pid: u32,
    pub(crate) start_time: u64,
}

/// Outcome of one ledger transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Settlement {
    /// The record now holds exactly the requested target.
    Settled {
        /// Granted by breaking a stall beyond the pool's budget.
        beyond_budget: bool,
    },
    /// The target does not fit yet; the record is marked pending.
    Waiting,
}

/// A read of the host-wide pool for diagnostics: who holds what, and who waits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPoolSnapshot {
    pub budget_bytes: u64,
    pub held_bytes: u64,
    /// Every record, largest holding first.
    pub holders: Vec<HostPoolHolder>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostPoolHolder {
    pub pid: u32,
    /// The holder's executable name.
    pub label: String,
    pub held_bytes: u64,
    pub pending_bytes: u64,
    pub ticket: u64,
}

impl HostPoolSnapshot {
    fn of(ledger: &LedgerFile) -> Self {
        let mut holders: Vec<HostPoolHolder> = ledger
            .holders
            .iter()
            .map(|record| HostPoolHolder {
                pid: record.pid,
                label: record.label.clone(),
                held_bytes: record.held_bytes,
                pending_bytes: record.pending_bytes,
                ticket: record.ticket,
            })
            .collect();
        holders.sort_by(|a, b| {
            b.held_bytes
                .cmp(&a.held_bytes)
                .then(a.ticket.cmp(&b.ticket))
        });
        Self {
            budget_bytes: ledger.budget_bytes,
            held_bytes: total_held(ledger),
            holders,
        }
    }
}

impl std::fmt::Display for HostPoolSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} of {} bytes held host-wide by {} gam process record(s)",
            self.held_bytes,
            self.budget_bytes,
            self.holders.len()
        )?;
        // The largest few holders are the ones whose release would change the
        // wait; the rest are summarized by the total above.
        for (index, holder) in self.holders.iter().take(4).enumerate() {
            f.write_str(if index == 0 { "; largest: " } else { ", " })?;
            write!(
                f,
                "pid {} ({}) holding {} bytes",
                holder.pid, holder.label, holder.held_bytes
            )?;
            if holder.pending_bytes > 0 {
                write!(f, " waiting for {} more", holder.pending_bytes)?;
            }
        }
        Ok(())
    }
}

fn total_held(ledger: &LedgerFile) -> u64 {
    ledger
        .holders
        .iter()
        .fold(0u64, |sum, record| sum.saturating_add(record.held_bytes))
}

/// Would `need` more bytes for record `index` fit the pool? With `seniority`,
/// every more-senior waiter's pending bytes are kept free as well.
fn fits(ledger: &LedgerFile, index: usize, need: u64, seniority: bool) -> bool {
    let ticket = ledger.holders[index].ticket;
    let seniors_pending = if seniority {
        ledger
            .holders
            .iter()
            .enumerate()
            .filter(|(other, record)| *other != index && record.ticket < ticket)
            .fold(0u64, |sum, (_, record)| {
                sum.saturating_add(record.pending_bytes)
            })
    } else {
        0
    };
    total_held(ledger)
        .saturating_add(need)
        .saturating_add(seniors_pending)
        <= ledger.budget_bytes
}

/// Every record that holds bytes is blocked: waiting itself, or an ancestor of
/// a waiter. Nothing held will be released, so no wait can end by release.
fn stalled(ledger: &LedgerFile) -> bool {
    let blocked = |record: &HolderRecord| {
        record.pending_bytes > 0
            || ledger.holders.iter().any(|waiter| {
                waiter.pending_bytes > 0 && waiter.waiting_ancestors.contains(&record.pid)
            })
    };
    ledger
        .holders
        .iter()
        .filter(|record| record.held_bytes > 0)
        .all(blocked)
}

fn grant(record: &mut HolderRecord, target: u64) {
    record.held_bytes = target;
    record.pending_bytes = 0;
    record.waiting_ancestors.clear();
}

/// One transaction: move `me`'s record to hold exactly `target` bytes if the
/// pool admits it now, otherwise mark it waiting. Pure over the ledger and the
/// liveness oracle, so the policy is unit-testable without processes.
///
/// `alive` answers, for each `(pid, start_time)`, whether that exact process
/// still exists; it is consulted only when a request does not fit.
pub(crate) fn settle(
    ledger: &mut LedgerFile,
    me: ProcessIdentity,
    label: &str,
    target: u64,
    ancestors: &[u32],
    alive: &mut LivenessProbe<'_>,
) -> Settlement {
    // A record with this pid but another start time belongs to a dead
    // predecessor whose pid was recycled to this process.
    ledger
        .holders
        .retain(|record| record.pid != me.pid || record.start_time == me.start_time);
    let position = |ledger: &LedgerFile| {
        ledger
            .holders
            .iter()
            .position(|record| record.pid == me.pid && record.start_time == me.start_time)
    };
    let mine = position(ledger);
    let held_now = mine.map_or(0, |index| ledger.holders[index].held_bytes);
    if target <= held_now {
        // Returning bytes never waits.
        if let Some(index) = mine {
            if target == 0 {
                ledger.holders.remove(index);
            } else {
                grant(&mut ledger.holders[index], target);
            }
        }
        return Settlement::Settled {
            beyond_budget: false,
        };
    }
    let mut index = match mine {
        Some(index) => index,
        None => {
            let ticket = ledger.next_ticket;
            ledger.next_ticket = ticket.saturating_add(1);
            ledger.holders.push(HolderRecord {
                pid: me.pid,
                start_time: me.start_time,
                ticket,
                held_bytes: 0,
                pending_bytes: 0,
                label: label.to_owned(),
                waiting_ancestors: Vec::new(),
            });
            ledger.holders.len() - 1
        }
    };
    let need = target - held_now;
    if !fits(ledger, index, need, true) {
        // Contention: reclaim the records of processes that are gone.
        let others: Vec<(u32, u64)> = ledger
            .holders
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, record)| (record.pid, record.start_time))
            .collect();
        if !others.is_empty() {
            let verdicts = alive(&others);
            let mut verdicts = verdicts.into_iter();
            let mut position_in_all = 0usize;
            ledger.holders.retain(|_| {
                let keep = position_in_all == index || verdicts.next().unwrap_or(true);
                position_in_all += 1;
                keep
            });
            index = position(ledger).expect("this process's record survives reclamation");
        }
    }
    if fits(ledger, index, need, true) {
        grant(&mut ledger.holders[index], target);
        return Settlement::Settled {
            beyond_budget: false,
        };
    }
    {
        let record = &mut ledger.holders[index];
        record.pending_bytes = need;
        record.waiting_ancestors = ancestors.to_vec();
    }
    if stalled(ledger) {
        let mut waiters: Vec<usize> = (0..ledger.holders.len())
            .filter(|&other| ledger.holders[other].pending_bytes > 0)
            .collect();
        waiters.sort_by_key(|&other| ledger.holders[other].ticket);
        let first_that_fits = waiters.iter().copied().find(|&other| {
            fits(ledger, other, ledger.holders[other].pending_bytes, false)
        });
        let chosen = first_that_fits.unwrap_or(waiters[0]);
        if chosen == index {
            grant(&mut ledger.holders[index], target);
            return Settlement::Settled {
                beyond_budget: first_that_fits.is_none(),
            };
        }
    }
    Settlement::Waiting
}

/// The pool this process attaches to: where its files live and who this
/// process is in it. Cheap to clone; all mutable state is on disk or in the
/// [`HostLease`] that owns the session.
#[derive(Clone, Debug)]
pub(crate) struct HostMemoryPool {
    directory: PathBuf,
    lock_path: PathBuf,
    ledger_path: PathBuf,
    budget_bytes: u64,
    me: ProcessIdentity,
    label: String,
}

impl HostMemoryPool {
    /// The pool of this user on this host for `budget_bytes`, in the temp
    /// directory (per-user on macOS; the file name carries the uid on a shared
    /// Linux `/tmp`, and a file another user squatted is refused).
    pub(crate) fn open_default(budget_bytes: u64) -> std::io::Result<Self> {
        let stem = format!("gam-host-memory-pool-uid{}-{budget_bytes}", this_user_id()?);
        Self::open_in(&std::env::temp_dir(), &stem, budget_bytes)
    }

    /// A pool whose ledger lives at `directory/stem.{lock,json}`.
    pub(crate) fn open_in(directory: &Path, stem: &str, budget_bytes: u64) -> std::io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        let lock_path = directory.join(format!("{stem}.lock"));
        let ledger_path = directory.join(format!("{stem}.json"));
        let lock = open_lock_file(&lock_path)?;
        ensure_owned_by_this_user(&lock)?;
        drop(lock);
        Ok(Self {
            directory: directory.to_path_buf(),
            lock_path,
            ledger_path,
            budget_bytes,
            me: this_process_identity()?,
            label: this_process_label(),
        })
    }

    pub(crate) fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// The file a live record's owner keeps exclusively locked for as long as
    /// its record exists. The kernel drops the lock when the owner dies, so a
    /// shared lock that can be taken means the record is dead; the name carries
    /// the start time, so a recycled pid never answers for a dead holder.
    fn presence_path(&self, who: ProcessIdentity) -> PathBuf {
        self.directory
            .join(format!("gmp{}-{}.presence", who.pid, who.start_time))
    }

    /// The datagram socket a waiting process blocks on. Short, because a Unix
    /// socket path is limited to about a hundred bytes.
    fn bell_path(&self, who: ProcessIdentity) -> PathBuf {
        self.directory
            .join(format!("gmp{}-{}.bell", who.pid, who.start_time))
    }

    /// Run `edit` on the ledger under the pool's exclusive lock, writing it back
    /// if it changed, and return what `edit` returned with every other record's
    /// identity. The ledger is replaced by rename, so a crash mid-write leaves
    /// the previous ledger intact.
    ///
    /// `presence` is this process's presence lock: taken before its record
    /// first becomes visible, released only after the record is gone. Every
    /// waiter but this process is rung after a change, so a waiter never sleeps
    /// through the transaction that could admit it.
    fn transact<R>(
        &self,
        presence: &mut Option<File>,
        edit: impl FnOnce(&mut LedgerFile, &mut LivenessProbe<'_>) -> R,
    ) -> std::io::Result<(R, Vec<ProcessIdentity>)> {
        let lock = open_lock_file(&self.lock_path)?;
        lock.lock()?;
        let before = self.read_ledger();
        let mut ledger = before.clone();
        let result = edit(&mut ledger, &mut |identities| self.presence_alive(identities));
        let identity = |record: &HolderRecord| ProcessIdentity {
            pid: record.pid,
            start_time: record.start_time,
        };
        let recorded = ledger.holders.iter().any(|record| identity(record) == self.me);
        if recorded && presence.is_none() {
            let file = open_lock_file(&self.presence_path(self.me))?;
            file.lock()?;
            *presence = Some(file);
        }
        let mut bells = Vec::new();
        if ledger != before {
            let staging = self
                .ledger_path
                .with_extension(format!("json.{}", self.me.pid));
            let bytes = serde_json::to_vec(&ledger).map_err(std::io::Error::other)?;
            {
                let mut file = File::create(&staging)?;
                file.write_all(&bytes)?;
            }
            std::fs::rename(&staging, &self.ledger_path)?;
            // A record of another process that this transaction removed was
            // reclaimed as dead: its files go with it.
            for gone in before
                .holders
                .iter()
                .map(identity)
                .filter(|who| *who != self.me)
                .filter(|who| !ledger.holders.iter().any(|record| identity(record) == *who))
            {
                remove_quietly(&self.presence_path(gone));
                remove_quietly(&self.bell_path(gone));
            }
            bells = ledger
                .holders
                .iter()
                .filter(|record| record.pending_bytes > 0 && identity(record) != self.me)
                .map(|record| self.bell_path(identity(record)))
                .collect();
        }
        if !recorded && let Some(file) = presence.take() {
            drop(file);
            remove_quietly(&self.presence_path(self.me));
        }
        let others = ledger
            .holders
            .iter()
            .map(identity)
            .filter(|who| *who != self.me)
            .collect();
        lock.unlock()?;
        for bell in &bells {
            ring(bell);
        }
        Ok((result, others))
    }

    /// Whether each `(pid, start_time)` still holds its presence lock.
    fn presence_alive(&self, identities: &[(u32, u64)]) -> Vec<bool> {
        identities
            .iter()
            .map(|&(pid, start_time)| {
                match File::open(self.presence_path(ProcessIdentity { pid, start_time })) {
                    Err(error) => error.kind() != std::io::ErrorKind::NotFound,
                    Ok(file) => match file.try_lock_shared() {
                        Ok(()) => false,
                        Err(std::fs::TryLockError::WouldBlock) => true,
                        // Unknown: presumed alive, so a live holder is never
                        // robbed; the next contender asks again.
                        Err(std::fs::TryLockError::Error(error)) => {
                            log::debug!("host memory pool presence probe for pid {pid}: {error}");
                            true
                        }
                    },
                }
            })
            .collect()
    }

    /// The ledger on disk, or an empty one when it is missing, unreadable, of
    /// another format version, or for another budget.
    fn read_ledger(&self) -> LedgerFile {
        let fresh = LedgerFile {
            version: LEDGER_FORMAT_VERSION,
            budget_bytes: self.budget_bytes,
            next_ticket: 0,
            holders: Vec::new(),
        };
        let mut text = String::new();
        let read = File::open(&self.ledger_path).and_then(|mut file| file.read_to_string(&mut text));
        if let Err(error) = read {
            if error.kind() != std::io::ErrorKind::NotFound {
                log::debug!("host memory pool ledger unreadable, rebuilding: {error}");
            }
            return fresh;
        }
        match serde_json::from_str::<LedgerFile>(&text) {
            Ok(ledger)
                if ledger.version == LEDGER_FORMAT_VERSION
                    && ledger.budget_bytes == self.budget_bytes =>
            {
                ledger
            }
            Ok(other) => {
                log::debug!(
                    "host memory pool ledger of version {} for {} bytes replaced",
                    other.version,
                    other.budget_bytes
                );
                fresh
            }
            Err(error) => {
                log::debug!("host memory pool ledger unparsable, rebuilding: {error}");
                fresh
            }
        }
    }

    /// One transaction moving this process's record to `target` bytes.
    fn settle(
        &self,
        presence: &mut Option<File>,
        target: u64,
        ancestors: &[u32],
    ) -> std::io::Result<(Settlement, Vec<ProcessIdentity>)> {
        self.transact(presence, |ledger, alive| {
            settle(ledger, self.me, &self.label, target, ancestors, alive)
        })
    }

    /// Who holds the pool now. Read-only: never writes, never rings.
    pub(crate) fn snapshot(&self) -> std::io::Result<HostPoolSnapshot> {
        let lock = open_lock_file(&self.lock_path)?;
        lock.lock_shared()?;
        let snapshot = HostPoolSnapshot::of(&self.read_ledger());
        lock.unlock()?;
        Ok(snapshot)
    }
}

/// A file this process no longer needs; its absence is the goal, so a missing
/// file is success and any other failure only litters the temp directory.
fn remove_quietly(path: &Path) {
    if let Err(error) = std::fs::remove_file(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        log::debug!("host memory pool could not remove {}: {error}", path.display());
    }
}

fn open_lock_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// Ring the bell at `path`: one datagram, never blocking. A bell that is gone
/// (its waiter was granted) or full (a wake is already queued) needs nothing.
fn ring(path: &Path) {
    if let Err(error) = bell::ring(path) {
        log::trace!("host memory pool bell {}: {error}", path.display());
    }
}

#[cfg(unix)]
mod bell {
    use std::os::unix::net::UnixDatagram;
    use std::path::{Path, PathBuf};

    /// A waiter's doorbell: a bound datagram socket that every ledger change
    /// and every holder's death rings.
    #[derive(Debug)]
    pub(super) struct Bell {
        socket: UnixDatagram,
        path: PathBuf,
    }

    impl Bell {
        pub(super) fn bind(path: &Path) -> std::io::Result<Self> {
            super::remove_quietly(path);
            Ok(Self {
                socket: UnixDatagram::bind(path)?,
                path: path.to_path_buf(),
            })
        }

        /// Block until rung at least once since the last wait, then drain.
        pub(super) fn wait(&self) -> std::io::Result<()> {
            let mut byte = [0u8; 1];
            self.socket.recv(&mut byte)?;
            self.socket.set_nonblocking(true)?;
            let drained = loop {
                match self.socket.recv(&mut byte) {
                    Ok(_) => continue,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break Ok(()),
                    Err(error) => break Err(error),
                }
            };
            self.socket.set_nonblocking(false)?;
            drained
        }
    }

    impl Drop for Bell {
        fn drop(&mut self) {
            super::remove_quietly(&self.path);
        }
    }

    pub(super) fn ring(path: &Path) -> std::io::Result<()> {
        let socket = UnixDatagram::unbound()?;
        socket.set_nonblocking(true)?;
        socket.send_to(b"!", path).map(drop)
    }
}

/// Without Unix datagram sockets there is no doorbell, so the pool is not
/// joined ([`this_user_id`] refuses first) and the ledger stays process-local.
#[cfg(not(unix))]
mod bell {
    use std::path::Path;

    #[derive(Debug)]
    pub(super) struct Bell;

    fn unsupported(path: &Path) -> std::io::Error {
        std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("no host memory pool doorbell at {} on this platform", path.display()),
        )
    }

    impl Bell {
        pub(super) fn bind(path: &Path) -> std::io::Result<Self> {
            Err(unsupported(path))
        }

        pub(super) fn wait(&self) -> std::io::Result<()> {
            Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
        }
    }

    pub(super) fn ring(path: &Path) -> std::io::Result<()> {
        Err(unsupported(path))
    }
}

#[cfg(unix)]
fn this_user_id() -> std::io::Result<u32> {
    // SAFETY: `getuid` has no preconditions and cannot fail.
    Ok(unsafe { libc::getuid() })
}

#[cfg(not(unix))]
fn this_user_id() -> std::io::Result<u32> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "the host-wide memory pool needs Unix datagram sockets",
    ))
}

/// A pool file another user created (a squatted name in a shared `/tmp`) must
/// not be joined: its owner could hold this user's budget hostage.
#[cfg(unix)]
fn ensure_owned_by_this_user(file: &File) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let uid = this_user_id()?;
    let owner = file.metadata()?.uid();
    if owner == uid {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("host memory pool lock file is owned by uid {owner}, not this user ({uid})"),
        ))
    }
}

#[cfg(not(unix))]
fn ensure_owned_by_this_user(file: &File) -> std::io::Result<()> {
    file.metadata().map(drop)
}

fn this_process_identity() -> std::io::Result<ProcessIdentity> {
    let pid = std::process::id();
    let sys_pid = sysinfo::Pid::from_u32(pid);
    let mut system = sysinfo::System::new();
    system.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[sys_pid]),
        true,
        sysinfo::ProcessRefreshKind::nothing(),
    );
    let start_time = system
        .process(sys_pid)
        .map(sysinfo::Process::start_time)
        .ok_or_else(|| std::io::Error::other("this process's start time is not observable"))?;
    Ok(ProcessIdentity { pid, start_time })
}

fn this_process_label() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.file_name().map(|name| name.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".to_owned())
        .chars()
        .take(64)
        .collect()
}

/// This process's OS ancestors, nearest first.
fn this_process_ancestors() -> Vec<u32> {
    let mut system = sysinfo::System::new();
    let mut ancestors = Vec::new();
    let mut current = sysinfo::Pid::from_u32(std::process::id());
    loop {
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[current]),
            false,
            sysinfo::ProcessRefreshKind::nothing(),
        );
        let Some(parent) = system.process(current).and_then(sysinfo::Process::parent) else {
            break;
        };
        let parent_pid = parent.as_u32();
        // A cycle or pid 0 ends the chain (init's parent, or a torn read).
        if parent_pid == 0 || ancestors.contains(&parent_pid) {
            break;
        }
        ancestors.push(parent_pid);
        current = parent;
    }
    ancestors
}

/// One warning per process when the pool cannot be used; the ledger then
/// proceeds process-locally, which is the pre-pool behaviour.
pub(crate) fn warn_host_pool_unusable(error: &std::io::Error) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        log::warn!(
            "host-wide gam memory pool unavailable ({error}); this process's governor \
             bounds only its own reservations"
        );
    }
}

/// The mutable side of this process's membership, one transaction at a time.
#[derive(Debug)]
struct PoolSession {
    presence: Option<File>,
    ancestors: Option<Vec<u32>>,
    /// Holders a watcher thread is already blocked on. Shared with those
    /// threads, which remove themselves when their holder's record ends.
    watched: Arc<Mutex<HashSet<ProcessIdentity>>>,
}

/// This process's lease on the host-wide pool: bytes it holds there, always at
/// least its own reserved bytes once a reservation has returned.
///
/// Leasing in quanta of 1/1024 of the budget keeps a reservation that stays
/// inside the lease at one atomic load (hot cache inserts reserve per entry),
/// while the unused lease stays under two quanta per process.
#[derive(Debug)]
pub(crate) struct HostLease {
    pool: HostMemoryPool,
    quantum: usize,
    budget_bytes: usize,
    leased: AtomicUsize,
    /// A thread of this process is blocked in the pool; a local release rings
    /// its bell so it can re-read what it still needs.
    waiting: AtomicBool,
    /// One pool transaction per process at a time, so this process has at most
    /// one pending request in the pool.
    session: Mutex<PoolSession>,
}

impl HostLease {
    pub(crate) fn new(pool: HostMemoryPool) -> Self {
        let budget_bytes = usize::try_from(pool.budget_bytes()).unwrap_or(usize::MAX);
        Self {
            quantum: (budget_bytes / 1024).max(1),
            budget_bytes,
            pool,
            leased: AtomicUsize::new(0),
            waiting: AtomicBool::new(false),
            session: Mutex::new(PoolSession {
                presence: None,
                ancestors: None,
                watched: Arc::new(Mutex::new(HashSet::new())),
            }),
        }
    }

    /// The lease covering `own` reserved bytes: rounded up to a quantum, never
    /// beyond the budget (an admitted `own` never exceeds it, so every target
    /// fits an empty pool).
    fn target_for(&self, own: usize) -> usize {
        own.div_ceil(self.quantum)
            .saturating_mul(self.quantum)
            .min(self.budget_bytes)
    }

    /// Block until the lease covers `own_after`, the ledger total this
    /// reservation produced. The verdict is already taken; this only waits.
    pub(crate) fn cover(&self, own_after: usize, reserved: &AtomicUsize, context: &str) {
        if own_after <= self.leased.load(Ordering::SeqCst) {
            return;
        }
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Bound before any transaction can mark this process pending, so no
        // ring that follows the mark is lost.
        let bell = match bell::Bell::bind(&self.pool.bell_path(self.pool.me)) {
            Ok(bell) => bell,
            Err(error) => {
                warn_host_pool_unusable(&error);
                self.leased
                    .store(self.target_for(reserved.load(Ordering::SeqCst)), Ordering::SeqCst);
                return;
            }
        };
        let mut waited = false;
        loop {
            let leased = self.leased.load(Ordering::SeqCst);
            let target = self.target_for(reserved.load(Ordering::SeqCst));
            if target <= leased {
                if waited {
                    // This process's own releases covered the request while it
                    // waited: withdraw the pending record.
                    if let Err(error) =
                        self.pool.settle(&mut session.presence, leased as u64, &[])
                    {
                        warn_host_pool_unusable(&error);
                    }
                    self.waiting.store(false, Ordering::SeqCst);
                }
                return;
            }
            let ancestors = session.ancestors.clone().unwrap_or_default();
            match self.pool.settle(&mut session.presence, target as u64, &ancestors) {
                Ok((Settlement::Settled { beyond_budget }, _)) => {
                    self.leased.store(target, Ordering::SeqCst);
                    self.waiting.store(false, Ordering::SeqCst);
                    if beyond_budget {
                        log::warn!(
                            "{context}: every gam process holding host memory is waiting on \
                             another, so this, the most senior waiter, proceeds beyond the \
                             host-wide pool; {}",
                            self.snapshot_text()
                        );
                    } else if waited {
                        log::info!("{context}: host-wide gam memory pool granted after waiting");
                    }
                    return;
                }
                Ok((Settlement::Waiting, others)) => {
                    if !waited {
                        waited = true;
                        self.waiting.store(true, Ordering::SeqCst);
                        log::warn!(
                            "{context}: waiting for {} bytes of the host-wide gam memory pool \
                             (this process already holds {leased}; its own budget admits the \
                             request, only the timing waits on other processes); {}",
                            target - leased,
                            self.snapshot_text()
                        );
                        if session.ancestors.is_none() {
                            // The ancestor chain lets the pool see a driver that
                            // waits on this process; retry at once with it.
                            session.ancestors = Some(this_process_ancestors());
                            continue;
                        }
                    }
                    self.watch_holders(&session, &others);
                    if let Err(error) = bell.wait() {
                        warn_host_pool_unusable(&error);
                        self.leased.store(target, Ordering::SeqCst);
                        self.waiting.store(false, Ordering::SeqCst);
                        return;
                    }
                }
                Err(error) => {
                    warn_host_pool_unusable(&error);
                    self.leased.store(target, Ordering::SeqCst);
                    self.waiting.store(false, Ordering::SeqCst);
                    return;
                }
            }
        }
    }

    /// Make sure a thread is blocked on each other holder's presence lock: it
    /// returns when that holder's record ends or the holder dies (the kernel
    /// drops the lock), and rings this process's bell. A holder that dies
    /// changes no ledger, so without this a waiter could sleep through the one
    /// event that admits it.
    fn watch_holders(&self, session: &PoolSession, others: &[ProcessIdentity]) {
        let mut watched = session
            .watched
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for &holder in others {
            if !watched.insert(holder) {
                continue;
            }
            let presence = self.pool.presence_path(holder);
            let bell = self.pool.bell_path(self.pool.me);
            let registry = Arc::clone(&session.watched);
            let spawned = std::thread::Builder::new()
                .name("gam-host-pool-watch".to_owned())
                .stack_size(64 * 1024)
                .spawn(move || {
                    match File::open(&presence) {
                        Ok(file) => {
                            if let Err(error) = file.lock_shared() {
                                log::debug!("host memory pool watcher: {error}");
                            }
                        }
                        Err(error) => {
                            log::trace!("host memory pool watcher: holder already gone: {error}");
                        }
                    }
                    registry
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .remove(&holder);
                    ring(&bell);
                });
            if let Err(error) = spawned {
                watched.remove(&holder);
                log::warn!("host memory pool: cannot watch holder pid {}: {error}", holder.pid);
            }
        }
    }

    /// Return lease this process no longer needs: all of it once nothing is
    /// reserved, otherwise whatever exceeds the covering target by more than a
    /// quantum. Never blocks: if a transaction is in flight it leaves the lease
    /// covering the ledger, and a later release trims the rest.
    pub(crate) fn release_excess(&self, reserved: &AtomicUsize) {
        if self.waiting.load(Ordering::SeqCst) {
            // A thread of this process waits in the pool; what it needs may
            // just have shrunk.
            ring(&self.pool.bell_path(self.pool.me));
        }
        let own = reserved.load(Ordering::SeqCst);
        let leased = self.leased.load(Ordering::SeqCst);
        let keep = self.target_for(own);
        let excessive = if own == 0 {
            leased > 0
        } else {
            leased > keep.saturating_add(self.quantum)
        };
        if !excessive {
            return;
        }
        let mut session = match self.session.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        let leased = self.leased.load(Ordering::SeqCst);
        let mut keep = self.target_for(reserved.load(Ordering::SeqCst)).min(leased);
        self.leased.store(keep, Ordering::SeqCst);
        // A reservation that read the old lease before the store is visible
        // here (both sides are SeqCst), so the lease never drops below it.
        let needed = self.target_for(reserved.load(Ordering::SeqCst)).min(leased);
        if needed > keep {
            keep = needed;
            self.leased.store(keep, Ordering::SeqCst);
        }
        if keep < leased
            && let Err(error) = self.pool.settle(&mut session.presence, keep as u64, &[])
        {
            warn_host_pool_unusable(&error);
        }
    }

    pub(crate) fn snapshot(&self) -> std::io::Result<HostPoolSnapshot> {
        self.pool.snapshot()
    }

    fn snapshot_text(&self) -> String {
        match self.pool.snapshot() {
            Ok(snapshot) => snapshot.to_string(),
            Err(error) => format!("host pool unreadable: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET: u64 = 1_000;

    fn ledger() -> LedgerFile {
        LedgerFile {
            version: LEDGER_FORMAT_VERSION,
            budget_bytes: BUDGET,
            next_ticket: 0,
            holders: Vec::new(),
        }
    }

    fn process(pid: u32) -> ProcessIdentity {
        ProcessIdentity {
            pid,
            start_time: 1_000 + u64::from(pid),
        }
    }

    fn everyone_alive(identities: &[(u32, u64)]) -> Vec<bool> {
        vec![true; identities.len()]
    }

    fn at(ledger: &mut LedgerFile, who: ProcessIdentity, target: u64) -> Settlement {
        settle(ledger, who, "test", target, &[], &mut everyone_alive)
    }

    fn held(ledger: &LedgerFile, who: ProcessIdentity) -> Option<(u64, u64)> {
        ledger
            .holders
            .iter()
            .find(|record| record.pid == who.pid && record.start_time == who.start_time)
            .map(|record| (record.held_bytes, record.pending_bytes))
    }

    const GRANTED: Settlement = Settlement::Settled {
        beyond_budget: false,
    };

    #[test]
    fn reserve_and_release_move_one_record_and_free_it_at_zero() {
        let mut ledger = ledger();
        let (a, b) = (process(1), process(2));
        assert_eq!(at(&mut ledger, a, 400), GRANTED);
        assert_eq!(at(&mut ledger, b, 600), GRANTED);
        assert_eq!(total_held(&ledger), 1_000);
        assert_eq!(at(&mut ledger, a, 100), GRANTED);
        assert_eq!(held(&ledger, a), Some((100, 0)));
        assert_eq!(at(&mut ledger, a, 0), GRANTED);
        assert_eq!(held(&ledger, a), None, "a record holding nothing is removed");
        assert_eq!(ledger.holders.len(), 1);
    }

    #[test]
    fn a_request_that_does_not_fit_waits_and_is_granted_after_release() {
        let mut ledger = ledger();
        let (a, b) = (process(1), process(2));
        assert_eq!(at(&mut ledger, a, 600), GRANTED);
        // b fits an empty pool, so it waits rather than being refused.
        assert_eq!(at(&mut ledger, b, 600), Settlement::Waiting);
        assert_eq!(held(&ledger, b), Some((0, 600)));
        assert_eq!(at(&mut ledger, b, 600), Settlement::Waiting);
        assert_eq!(at(&mut ledger, a, 0), GRANTED);
        assert_eq!(at(&mut ledger, b, 600), GRANTED);
        assert_eq!(held(&ledger, b), Some((600, 0)));
    }

    #[test]
    fn a_junior_does_not_backfill_into_a_senior_waiters_bytes() {
        let mut ledger = ledger();
        let (a, senior, junior) = (process(1), process(2), process(3));
        assert_eq!(at(&mut ledger, a, 500), GRANTED);
        assert_eq!(at(&mut ledger, senior, 600), Settlement::Waiting);
        assert_eq!(at(&mut ledger, junior, 500), Settlement::Waiting);
        assert_eq!(at(&mut ledger, a, 0), GRANTED);
        // 500 fits the empty pool, but only by taking what the senior waiter
        // needs: it waits behind it.
        assert_eq!(at(&mut ledger, junior, 500), Settlement::Waiting);
        assert_eq!(at(&mut ledger, senior, 600), GRANTED);
        assert_eq!(at(&mut ledger, junior, 500), Settlement::Waiting);
        assert_eq!(at(&mut ledger, senior, 0), GRANTED);
        assert_eq!(at(&mut ledger, junior, 500), GRANTED);
        // A junior that fits beside every senior waiter's bytes is not held back.
        let mut ledger = self::ledger();
        assert_eq!(at(&mut ledger, a, 500), GRANTED);
        assert_eq!(at(&mut ledger, senior, 600), Settlement::Waiting);
        assert_eq!(at(&mut ledger, a, 0), GRANTED);
        assert_eq!(at(&mut ledger, junior, 300), GRANTED);
        assert_eq!(at(&mut ledger, senior, 600), GRANTED);
    }

    #[test]
    fn a_senior_waiting_on_a_blocked_junior_lets_the_junior_through() {
        // The senior waits for the junior's bytes; the junior waits behind the
        // senior's reservation. Everything held is blocked, and the junior fits
        // the pool as it stands, so the stall is broken by the junior, in budget.
        let mut ledger = ledger();
        let (senior, junior) = (process(1), process(2));
        assert_eq!(at(&mut ledger, senior, 1), GRANTED);
        assert_eq!(at(&mut ledger, junior, 300), GRANTED);
        assert_eq!(at(&mut ledger, senior, 901), Settlement::Waiting);
        assert_eq!(at(&mut ledger, junior, 600), GRANTED);
        assert!(total_held(&ledger) <= BUDGET);
        assert_eq!(at(&mut ledger, junior, 0), GRANTED);
        assert_eq!(at(&mut ledger, senior, 901), GRANTED);
    }

    #[test]
    fn a_hold_and_wait_cycle_is_broken_by_the_most_senior_waiter() {
        let mut ledger = ledger();
        let (senior, junior) = (process(1), process(2));
        assert_eq!(at(&mut ledger, senior, 400), GRANTED);
        assert_eq!(at(&mut ledger, junior, 400), GRANTED);
        // The senior still holds and has not asked for more, so it will release:
        // the junior waits.
        assert_eq!(at(&mut ledger, junior, 800), Settlement::Waiting);
        assert_eq!(at(&mut ledger, junior, 800), Settlement::Waiting);
        // Now both hold and both wait, and neither fits the pool as it stands:
        // nothing will ever be released. The senior breaks it, beyond budget.
        assert_eq!(
            at(&mut ledger, senior, 900),
            Settlement::Settled {
                beyond_budget: true
            }
        );
        // The senior holds but no longer waits, so the junior's wait ends by
        // release, not by another overdraw.
        assert_eq!(at(&mut ledger, junior, 800), Settlement::Waiting);
        assert_eq!(at(&mut ledger, senior, 0), GRANTED);
        assert_eq!(at(&mut ledger, junior, 800), GRANTED);
    }

    #[test]
    fn in_a_cycle_the_junior_never_breaks_it() {
        let mut ledger = ledger();
        let (senior, junior) = (process(1), process(2));
        assert_eq!(at(&mut ledger, senior, 400), GRANTED);
        assert_eq!(at(&mut ledger, junior, 400), GRANTED);
        assert_eq!(at(&mut ledger, senior, 900), Settlement::Waiting);
        // The junior closes the cycle, but the grant belongs to the senior.
        assert_eq!(at(&mut ledger, junior, 800), Settlement::Waiting);
        assert_eq!(held(&ledger, junior), Some((400, 400)));
        assert_eq!(
            at(&mut ledger, senior, 900),
            Settlement::Settled {
                beyond_budget: true
            }
        );
    }

    #[test]
    fn a_holding_ancestor_counts_as_blocked_on_its_waiting_child() {
        // A driver holds bytes and waits for the gam child it spawned; the child
        // needs the driver's bytes. Without the ancestor rule the child would
        // wait for a release that cannot come.
        let mut ledger = ledger();
        let (parent, child) = (process(10), process(11));
        assert_eq!(at(&mut ledger, parent, 700), GRANTED);
        assert_eq!(
            settle(&mut ledger, child, "child", 600, &[99], &mut everyone_alive),
            Settlement::Waiting,
            "an unrelated holder is presumed to release"
        );
        assert_eq!(
            settle(&mut ledger, child, "child", 600, &[10, 1], &mut everyone_alive),
            Settlement::Settled {
                beyond_budget: true
            }
        );
    }

    #[test]
    fn a_dead_holder_is_reclaimed_on_contention() {
        let mut ledger = ledger();
        let (dead, live) = (process(1), process(2));
        assert_eq!(at(&mut ledger, dead, 900), GRANTED);
        let mut only_live = |identities: &[(u32, u64)]| {
            identities
                .iter()
                .map(|&(pid, _)| pid != dead.pid)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            settle(&mut ledger, live, "live", 900, &[], &mut only_live),
            GRANTED
        );
        assert_eq!(held(&ledger, dead), None);
    }

    #[test]
    fn a_recycled_pid_does_not_inherit_the_dead_holders_bytes() {
        let mut ledger = ledger();
        let predecessor = process(7);
        let successor = ProcessIdentity {
            pid: predecessor.pid,
            start_time: predecessor.start_time + 5,
        };
        assert_eq!(at(&mut ledger, predecessor, 800), GRANTED);
        // The successor shares the pid; its own record starts from nothing.
        assert_eq!(at(&mut ledger, successor, 900), GRANTED);
        assert_eq!(ledger.holders.len(), 1);
        assert_eq!(held(&ledger, successor), Some((900, 0)));
        // And another process judges the pid by its start time, not its number.
        let mut ledger = self::ledger();
        assert_eq!(at(&mut ledger, predecessor, 800), GRANTED);
        let mut recycled = |identities: &[(u32, u64)]| {
            identities
                .iter()
                .map(|&(pid, start)| pid == successor.pid && start == successor.start_time)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            settle(&mut ledger, process(3), "other", 900, &[], &mut recycled),
            GRANTED
        );
    }

    #[test]
    fn a_record_is_alive_exactly_while_its_presence_lock_is_held() {
        let directory = tempfile::tempdir().expect("tempdir");
        let pool = HostMemoryPool::open_in(directory.path(), "pool", BUDGET).expect("open");
        let me = (pool.me.pid, pool.me.start_time);
        let recycled = (pool.me.pid, pool.me.start_time.wrapping_sub(1));
        // No record, no presence file: dead (and a recycled pid is never alive).
        assert_eq!(pool.presence_alive(&[me, recycled]), vec![false, false]);
        let mut presence = None;
        let (settled, others) = pool.settle(&mut presence, 300, &[]).expect("settle");
        assert_eq!(settled, GRANTED);
        assert!(others.is_empty());
        assert!(presence.is_some(), "a record is published under its presence lock");
        assert_eq!(pool.presence_alive(&[me, recycled]), vec![true, false]);
        // Dropping the lock without the record, as a crash does, reads as dead.
        let crashed = presence.take();
        drop(crashed);
        assert_eq!(pool.presence_alive(&[me]), vec![false]);
        let (released, _) = pool.settle(&mut presence, 0, &[]).expect("settle");
        assert_eq!(released, GRANTED);
        assert!(presence.is_none(), "no record, no presence lock");
        assert!(!this_process_ancestors().is_empty(), "a test runs under some parent");
    }

    #[test]
    fn transactions_persist_across_pool_handles_and_a_corrupt_ledger_is_rebuilt() {
        let directory = tempfile::tempdir().expect("tempdir");
        let pool = HostMemoryPool::open_in(directory.path(), "pool", BUDGET).expect("open");
        let mut presence = None;
        let settle = |presence: &mut Option<File>, target| {
            pool.settle(presence, target, &[]).expect("settle").0
        };
        assert_eq!(settle(&mut presence, 300), GRANTED);
        let again = HostMemoryPool::open_in(directory.path(), "pool", BUDGET).expect("open");
        let snapshot = again.snapshot().expect("snapshot");
        assert_eq!(snapshot.held_bytes, 300);
        assert_eq!(snapshot.holders[0].pid, std::process::id());
        std::fs::write(directory.path().join("pool.json"), b"{not json").expect("corrupt");
        assert_eq!(pool.snapshot().expect("snapshot").held_bytes, 0);
        // The holder's next transaction rewrites its absolute lease.
        assert_eq!(settle(&mut presence, 301), GRANTED);
        assert_eq!(pool.snapshot().expect("snapshot").held_bytes, 301);
        assert!(format!("{}", pool.snapshot().expect("snapshot")).contains("holding 301 bytes"));
        assert_eq!(settle(&mut presence, 0), GRANTED);
    }
}
