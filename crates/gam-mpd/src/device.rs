//! Proposal products on the Apple GPU (#2951).
//!
//! A fit has two kinds of computation. A *proposal* only ranks or steers: the exact first
//! derivatives that rank mask flips, the piece-update directions, the Fisher diagonals. An
//! *acceptance* decides: the masked forward whose exact code keeps or refuses an input's flips,
//! and backtracking's test of a step. Acceptances always run in float64 on the CPU. A proposal
//! may run its dense products in f32 on the GPU (`gam_gpu::banded`, Metal Performance Shaders),
//! where each product carries the band `γ_{k+2}(2^-23)·‖x_i‖‖a_j‖`; a wrong proposal costs
//! only the acceptance test that refuses it, so the band never reaches a decision.
//!
//! [`proposing`] marks a computation on the calling thread as a proposal. Inside it, [`product`]
//! routes `x·A` or `x·Aᵀ` for a dense operator by `gam_gpu`'s policy: `off` is the CPU; `auto`
//! takes Metal when a device resolved and the product clears its size floor; `required` takes
//! Metal or fails. Outside it, [`product`] is the float64 CPU product, bit for bit. Each operator
//! is uploaded once as f32 and kept on the device while the program holds it (the cache is keyed
//! by the operator's `Arc` and drops an entry when the operator is gone).

use super::operator_program::{Operator, OperatorBody, ProgramError};
use gam_gpu::banded::{BandedArithmetic, Layout, ResidentOperand, banded_matmul, resident_operand};
use gam_gpu::global_policy;
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
use ndarray::Array2;
use std::cell::Cell;
use std::sync::{Arc, Mutex, Weak};

thread_local! {
    static PROPOSING: Cell<bool> = const { Cell::new(false) };
}

/// Restores the calling thread's previous mode, also when the body unwinds.
struct ModeGuard(bool);

impl Drop for ModeGuard {
    fn drop(&mut self) {
        PROPOSING.set(self.0);
    }
}

/// Run `body` as a proposal (module note).
pub fn proposing<T>(body: impl FnOnce() -> T) -> T {
    let restore = ModeGuard(PROPOSING.replace(true));
    let out = body();
    drop(restore);
    out
}

fn device_error(error: impl std::fmt::Display) -> ProgramError {
    ProgramError::Input(format!("device product: {error}"))
}

/// One resident operator: the operator it was uploaded from and its device copy (`None` when
/// the policy keeps this operator's products on the CPU).
type Entry = (Weak<Operator>, Option<Arc<ResidentOperand>>);

static RESIDENT: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

/// The device copy of `op`'s matrix, uploading it on first use.
fn resident(op: &Arc<Operator>, values: &Array2<f64>) -> Result<Option<Arc<ResidentOperand>>, ProgramError> {
    let mut cache = RESIDENT.lock().map_err(|_| device_error("resident cache poisoned"))?;
    cache.retain(|(weak, _)| weak.strong_count() > 0);
    if let Some((_, entry)) = cache.iter().find(|(weak, _)| std::ptr::eq(weak.as_ptr(), Arc::as_ptr(op))) {
        return Ok(entry.clone());
    }
    let entry = resident_operand(global_policy(), BandedArithmetic::F32, values.view()).map_err(device_error)?.map(Arc::new);
    cache.push((Arc::downgrade(op), entry.clone()));
    Ok(entry)
}

/// `x·A` ([`Layout::AsStored`]) or `x·Aᵀ` ([`Layout::Transposed`]) for operator `op`: inside
/// [`proposing`], on the device when the policy selects it; otherwise the float64 CPU product.
pub fn product(op: &Arc<Operator>, x: &Array2<f64>, layout: Layout) -> Result<Array2<f64>, ProgramError> {
    if PROPOSING.get()
        && let OperatorBody::Dense { values, .. } = &op.body
        && let Some(device) = resident(op, values)?
        && let Some(banded) = device.product(x.view(), layout).map_err(device_error)?
    {
        return Ok(banded.values);
    }
    let a = op.matrix_cow();
    Ok(match layout {
        Layout::AsStored => fast_ab(x, a.as_ref()),
        Layout::Transposed => fast_abt(x, a.as_ref()),
    })
}

/// `aᵀ·b`: inside [`proposing`], in f32 on the device when the policy selects it; otherwise the
/// float64 CPU product.
pub fn product_atb(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, ProgramError> {
    if PROPOSING.get() {
        return Ok(banded_matmul(global_policy(), BandedArithmetic::F32, a.t(), b.view()).map_err(device_error)?.values);
    }
    Ok(fast_atb(a, b))
}
