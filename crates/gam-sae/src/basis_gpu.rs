//! Device-resident SAE basis evaluation for external-tensor (torch) interop.
//!
//! The torch manifold-SAE lane routes every forward/backward through the CPU
//! `basis_with_jet` FFI: device→host copy of the coordinates, a rayon CPU
//! evaluation, and a host→device copy of `(phi, jet)` — measured at 0.07–1.9
//! optimizer steps/s on a B200 while the GPU idles (the whole fit is the
//! bridge). This module is the GPU-resident sibling: the caller (the pyffi
//! torch bridge) passes RAW CUDA device pointers — the coordinates tensor and
//! two preallocated output tensors that torch owns — and a single NVRTC kernel
//! writes `phi` and `jet` in place. Zero host round-trips, no ownership
//! transfer across the boundary, and the basis MATH stays single-sourced in
//! this crate (the kernel mirrors [`crate::basis::PeriodicHarmonicEvaluator`]
//! exactly; the parity test pins them together).
//!
//! Synchronization contract: the caller must ensure the input tensor's
//! producing stream has completed before calling (the torch bridge issues a
//! `torch.cuda.synchronize()`), and this function synchronizes its own stream
//! before returning, so the outputs are globally visible to any stream the
//! caller reads them on afterwards.
//!
//! Safety contract: the pointers must be valid CUDA device allocations on
//! `ordinal`'s primary context (torch and cudarc both use the device primary
//! context, so allocations from either side share one address space), with
//! `t: n` doubles, `phi: n·(2H+1)` doubles and `jet: n·(2H+1)` doubles of
//! writable space (row-major; the `d = 1` jet `(n, m, 1)` is contiguous as
//! `(n, m)`).

#[cfg(target_os = "linux")]
mod device {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaFunction, CudaModule, PushKernelArg};

    /// Mirrors `PeriodicHarmonicEvaluator::evaluate` (basis.rs): column 0 is
    /// the constant, columns `2h-1`/`2h` are `sin(2πht)`/`cos(2πht)`, and the
    /// jet is the exact `t`-derivative. `--fmad=false` (the shared NVRTC
    /// options in `gam_gpu::device_cache`) keeps device rounding aligned with
    /// the CPU oracle.
    const PERIODIC_KERNEL_SRC: &str = r#"
extern "C" __global__ void sae_periodic_basis_with_jet(
    const double* __restrict__ t,
    double* __restrict__ phi,
    double* __restrict__ jet,
    long long n,
    long long num_harmonics)
{
    long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (i >= n) return;
    const long long m = 2 * num_harmonics + 1;
    const double two_pi = 6.283185307179586476925286766559;
    const double ti = t[i];
    double* prow = phi + i * m;
    double* jrow = jet + i * m;
    prow[0] = 1.0;
    jrow[0] = 0.0;
    for (long long h = 1; h <= num_harmonics; ++h) {
        const double freq = two_pi * (double)h;
        const double angle = freq * ti;
        const double s = sin(angle);
        const double c = cos(angle);
        prow[2 * h - 1] = s;
        prow[2 * h] = c;
        jrow[2 * h - 1] = freq * c;
        jrow[2 * h] = -freq * s;
    }
}
"#;

    fn periodic_function(ordinal: usize) -> Result<(Arc<CudaModule>, CudaFunction), String> {
        static MODULES: OnceLock<Mutex<HashMap<usize, Arc<CudaModule>>>> = OnceLock::new();
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_periodic_basis_with_jet_device: no CUDA context for ordinal {ordinal}")
        })?;
        let cache = MODULES.get_or_init(|| Mutex::new(HashMap::new()));
        let module = {
            let mut guard = cache
                .lock()
                .map_err(|err| format!("sae basis-gpu module cache poisoned: {err}"))?;
            match guard.get(&ordinal) {
                Some(module) => module.clone(),
                None => {
                    let ptx = gam_gpu::device_cache::compile_ptx_arch(PERIODIC_KERNEL_SRC)
                        .map_err(|err| {
                            format!("sae_periodic_basis_with_jet_device: NVRTC compile: {err}")
                        })?;
                    let module = ctx.load_module(ptx).map_err(|err| {
                        format!("sae_periodic_basis_with_jet_device: module load: {err}")
                    })?;
                    guard.insert(ordinal, module.clone());
                    module
                }
            }
        };
        let func = module.load_function("sae_periodic_basis_with_jet").map_err(|err| {
            format!("sae_periodic_basis_with_jet_device: load_function: {err}")
        })?;
        Ok((module, func))
    }

    /// Launch the periodic basis+jet kernel over `n` coordinates already on
    /// device `ordinal`, writing into caller-owned device buffers. See the
    /// module docs for the pointer and synchronization contracts.
    pub fn sae_periodic_basis_with_jet_device(
        ordinal: usize,
        t_dev_ptr: u64,
        n: usize,
        num_harmonics: usize,
        phi_dev_ptr: u64,
        jet_dev_ptr: u64,
    ) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        if num_harmonics == 0 {
            return Err(
                "sae_periodic_basis_with_jet_device: num_harmonics must be >= 1".to_string()
            );
        }
        if t_dev_ptr == 0 || phi_dev_ptr == 0 || jet_dev_ptr == 0 {
            return Err(
                "sae_periodic_basis_with_jet_device: null device pointer".to_string()
            );
        }
        let (_module, func) = periodic_function(ordinal)?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_periodic_basis_with_jet_device: no CUDA context for ordinal {ordinal}")
        })?;
        let stream = ctx.default_stream();
        let n_ll = i64::try_from(n)
            .map_err(|_| "sae_periodic_basis_with_jet_device: n overflows i64".to_string())?;
        let h_ll = i64::try_from(num_harmonics)
            .map_err(|_| "sae_periodic_basis_with_jet_device: H overflows i64".to_string())?;
        let block: u32 = 256;
        let grid: u32 = u32::try_from(n.div_ceil(block as usize))
            .map_err(|_| "sae_periodic_basis_with_jet_device: grid overflow".to_string())?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        // Raw device addresses travel as 8-byte scalars; the kernel signature
        // reinterprets them as `double*` (driver-API param buffers are untyped).
        builder.arg(&t_dev_ptr);
        builder.arg(&phi_dev_ptr);
        builder.arg(&jet_dev_ptr);
        builder.arg(&n_ll);
        builder.arg(&h_ll);
        // SAFETY: pointer validity and extents are the documented caller
        // contract; grid covers all n rows exactly once.
        unsafe { builder.launch(cfg) }
            .map_err(|err| format!("sae_periodic_basis_with_jet_device: launch: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("sae_periodic_basis_with_jet_device: sync: {err}"))?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod device_duchon {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::sync::{Arc, Mutex, OnceLock};

    use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, PushKernelArg};
    use gam_terms::basis::{
        DuchonNullspaceOrder, DuchonSaeDeviceParams, duchon_sae_atom_device_params,
    };
    use ndarray::ArrayView2;

    /// Runtime caps for the single-kernel evaluator's shared-memory staging:
    /// `n_centers·(2 + dim)` doubles must fit comfortably in the per-block
    /// shared allowance. The SAE torch lane uses `n_basis`-sized center clouds
    /// (tens), so these are generous.
    const MAX_CENTERS: usize = 512;
    const MAX_DIM: usize = 8;

    /// Mirrors `duchon_sae_atom_basis_with_jet` (gam-terms radial_jets_nd.rs):
    /// `phi = [amp·(ρ(r)·Z) | P(t)]`, `jet` the exact input-location
    /// derivative, `ρ(r) = c·r^power·[ln r]` with the CPU origin/`r ≤ 1e-12`
    /// conventions. One block per row: phase 1 stages `ρ_k`, `ρ'(r)/r`, and
    /// the per-axis deltas in shared memory; phase 2 spans output columns.
    const DUCHON_KERNEL_SRC: &str = r#"
extern "C" __global__ void sae_duchon_basis_with_jet(
    const double* __restrict__ t,        // (n, dim) row-major
    const double* __restrict__ centers,  // (kc, dim) row-major
    const double* __restrict__ z,        // (kc, n_kernel) row-major
    const long long* __restrict__ exps,  // (n_poly, dim) row-major
    double* __restrict__ phi,            // (n, width) row-major
    double* __restrict__ jet,            // (n, width, dim) row-major
    long long n,
    long long dim,
    long long kc,
    long long n_kernel,
    long long n_poly,
    double amp,
    double coeff_c,
    double coeff_power,
    long long coeff_is_log,
    double origin_value)
{
    extern __shared__ double stage[];   // [kc] rho | [kc] dcoef | [kc*dim] delta
    double* rho = stage;
    double* dcoef = stage + kc;
    double* delta = stage + 2 * kc;
    const long long row = blockIdx.x;
    if (row >= n) return;
    const double* trow = t + row * dim;
    for (long long k = threadIdx.x; k < kc; k += blockDim.x) {
        double r2 = 0.0;
        for (long long a = 0; a < dim; ++a) {
            const double d = trow[a] - centers[k * dim + a];
            delta[k * dim + a] = d;
            r2 += d * d;
        }
        const double r = sqrt(r2);
        if (r <= 0.0) {
            rho[k] = origin_value;
        } else if (coeff_is_log) {
            rho[k] = coeff_c * pow(r, coeff_power) * log(fmax(r, 1e-300));
        } else {
            rho[k] = coeff_c * pow(r, coeff_power);
        }
        // radial_input_location_jet_nd skips r <= 1e-12 (contribution 0).
        if (r <= 1.0e-12) {
            dcoef[k] = 0.0;
        } else if (coeff_is_log) {
            // d/dr [c r^p ln r] = c r^{p-1} (p ln r + 1); dcoef = phi'(r)/r.
            dcoef[k] = coeff_c * pow(r, coeff_power - 2.0)
                * (coeff_power * log(r) + 1.0);
        } else {
            dcoef[k] = coeff_c * coeff_power * pow(r, coeff_power - 2.0);
        }
    }
    __syncthreads();
    const long long width = n_kernel + n_poly;
    double* prow = phi + row * width;
    double* jrow = jet + row * width * dim;
    for (long long j = threadIdx.x; j < width; j += blockDim.x) {
        if (j < n_kernel) {
            double acc = 0.0;
            for (long long k = 0; k < kc; ++k) {
                acc += rho[k] * z[k * n_kernel + j];
            }
            prow[j] = amp * acc;
            for (long long a = 0; a < dim; ++a) {
                double jacc = 0.0;
                for (long long k = 0; k < kc; ++k) {
                    jacc += dcoef[k] * delta[k * dim + a] * z[k * n_kernel + j];
                }
                jrow[j * dim + a] = amp * jacc;
            }
        } else {
            const long long jp = j - n_kernel;
            const long long* alpha = exps + jp * dim;
            double value = 1.0;
            for (long long a = 0; a < dim; ++a) {
                for (long long e = 0; e < alpha[a]; ++e) value *= trow[a];
            }
            prow[j] = value;
            for (long long axis = 0; axis < dim; ++axis) {
                const long long a_axis = alpha[axis];
                if (a_axis == 0) { jrow[j * dim + axis] = 0.0; continue; }
                double dval = (double)a_axis;
                for (long long a = 0; a < dim; ++a) {
                    const long long e_target = alpha[a] - (a == axis ? 1 : 0);
                    for (long long e = 0; e < e_target; ++e) dval *= trow[a];
                }
                jrow[j * dim + axis] = dval;
            }
        }
    }
}
"#;

    /// Device-resident, center-derived state for one Duchon atom family.
    struct DuchonDeviceAtoms {
        module: Arc<CudaModule>,
        centers_dev: CudaSlice<f64>,
        z_dev: CudaSlice<f64>,
        exps_dev: CudaSlice<i64>,
        kc: usize,
        dim: usize,
        n_kernel: usize,
        n_poly: usize,
        kernel_amp: f64,
        coeff_c: f64,
        coeff_power: f64,
        coeff_is_log: bool,
        origin_value: f64,
    }

    fn params_key(ordinal: usize, centers: ArrayView2<'_, f64>, m: usize) -> (usize, u64) {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        centers.shape().hash(&mut hasher);
        m.hash(&mut hasher);
        for value in centers.iter() {
            value.to_bits().hash(&mut hasher);
        }
        (ordinal, hasher.finish())
    }

    fn nullspace_from_m(m: usize) -> DuchonNullspaceOrder {
        // Same mapping the CPU FFI (`duchon_nullspace_from_m`) applies.
        match m {
            0 | 1 => DuchonNullspaceOrder::Zero,
            2 => DuchonNullspaceOrder::Linear,
            other => DuchonNullspaceOrder::Degree(other - 1),
        }
    }

    fn atoms_for(
        ordinal: usize,
        centers: ArrayView2<'_, f64>,
        m: usize,
    ) -> Result<Arc<DuchonDeviceAtoms>, String> {
        static CACHE: OnceLock<Mutex<HashMap<(usize, u64), Arc<DuchonDeviceAtoms>>>> =
            OnceLock::new();
        let key = params_key(ordinal, centers, m);
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(hit) = cache
            .lock()
            .map_err(|err| format!("sae duchon-gpu cache poisoned: {err}"))?
            .get(&key)
        {
            return Ok(hit.clone());
        }
        let kc = centers.nrows();
        let dim = centers.ncols();
        if kc == 0 || kc > MAX_CENTERS || dim == 0 || dim > MAX_DIM {
            return Err(format!(
                "sae_duchon_basis_with_jet_device: centers ({kc} x {dim}) outside the \
                 device caps (1..={MAX_CENTERS} centers, 1..={MAX_DIM} dims)"
            ));
        }
        let params: DuchonSaeDeviceParams =
            duchon_sae_atom_device_params(centers, nullspace_from_m(m))
                .map_err(|err| format!("sae_duchon_basis_with_jet_device: params: {err}"))?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_duchon_basis_with_jet_device: no CUDA context for ordinal {ordinal}")
        })?;
        let ptx = gam_gpu::device_cache::compile_ptx_arch(DUCHON_KERNEL_SRC)
            .map_err(|err| format!("sae_duchon_basis_with_jet_device: NVRTC: {err}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|err| format!("sae_duchon_basis_with_jet_device: module load: {err}"))?;
        let stream = ctx.default_stream();
        let centers_host: Vec<f64> = centers.iter().copied().collect();
        let z_host: Vec<f64> = params.z.iter().copied().collect();
        let n_kernel = params.z.ncols();
        let n_poly = params.exponents.len();
        let mut exps_host: Vec<i64> = Vec::with_capacity(n_poly * dim);
        for alpha in &params.exponents {
            for &e in alpha {
                exps_host.push(i64::try_from(e).map_err(|_| {
                    "sae_duchon_basis_with_jet_device: exponent overflows i64".to_string()
                })?);
            }
        }
        if exps_host.is_empty() {
            // cudarc refuses zero-length uploads; keep one inert slot.
            exps_host.push(0);
        }
        let atoms = Arc::new(DuchonDeviceAtoms {
            module,
            centers_dev: stream
                .clone_htod(&centers_host)
                .map_err(|err| format!("sae_duchon_basis_with_jet_device: centers htod: {err}"))?,
            z_dev: stream
                .clone_htod(&z_host)
                .map_err(|err| format!("sae_duchon_basis_with_jet_device: Z htod: {err}"))?,
            exps_dev: stream
                .clone_htod(&exps_host)
                .map_err(|err| format!("sae_duchon_basis_with_jet_device: exps htod: {err}"))?,
            kc,
            dim,
            n_kernel,
            n_poly,
            kernel_amp: params.kernel_amp,
            coeff_c: params.coeff_c,
            coeff_power: params.coeff_power,
            coeff_is_log: params.coeff_is_log,
            origin_value: params.origin_value,
        });
        stream
            .synchronize()
            .map_err(|err| format!("sae_duchon_basis_with_jet_device: upload sync: {err}"))?;
        cache
            .lock()
            .map_err(|err| format!("sae duchon-gpu cache poisoned: {err}"))?
            .insert(key, atoms.clone());
        Ok(atoms)
    }

    /// Width `(n_kernel + n_poly)` of the device Duchon basis for these
    /// centers — the caller allocates `phi`/`jet` with exactly this width.
    pub fn sae_duchon_device_basis_width(
        ordinal: usize,
        centers: ArrayView2<'_, f64>,
        m: usize,
    ) -> Result<usize, String> {
        let atoms = atoms_for(ordinal, centers, m)?;
        Ok(atoms.n_kernel + atoms.n_poly)
    }

    /// Launch the Duchon basis+jet kernel over `n` coordinate rows already on
    /// device `ordinal`. Pointer/synchronization contract identical to the
    /// periodic lane (module docs above): `t` is `(n, dim)` doubles, `phi` is
    /// `(n, width)` and `jet` is `(n, width, dim)`, all contiguous and
    /// caller-owned.
    pub fn sae_duchon_basis_with_jet_device(
        ordinal: usize,
        t_dev_ptr: u64,
        n: usize,
        centers: ArrayView2<'_, f64>,
        m: usize,
        phi_dev_ptr: u64,
        jet_dev_ptr: u64,
    ) -> Result<usize, String> {
        let atoms = atoms_for(ordinal, centers, m)?;
        let width = atoms.n_kernel + atoms.n_poly;
        if n == 0 {
            return Ok(width);
        }
        if t_dev_ptr == 0 || phi_dev_ptr == 0 || jet_dev_ptr == 0 {
            return Err("sae_duchon_basis_with_jet_device: null device pointer".to_string());
        }
        let func: CudaFunction = atoms
            .module
            .load_function("sae_duchon_basis_with_jet")
            .map_err(|err| format!("sae_duchon_basis_with_jet_device: load_function: {err}"))?;
        let ctx = gam_gpu::device_runtime::cuda_context_for(ordinal).ok_or_else(|| {
            format!("sae_duchon_basis_with_jet_device: no CUDA context for ordinal {ordinal}")
        })?;
        let stream = ctx.default_stream();
        let grid = u32::try_from(n)
            .map_err(|_| "sae_duchon_basis_with_jet_device: n overflows the grid".to_string())?;
        let shared = (atoms.kc * (2 + atoms.dim) * std::mem::size_of::<f64>()) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: shared,
        };
        let n_ll = n as i64;
        let dim_ll = atoms.dim as i64;
        let kc_ll = atoms.kc as i64;
        let nk_ll = atoms.n_kernel as i64;
        let np_ll = atoms.n_poly as i64;
        let is_log_ll: i64 = if atoms.coeff_is_log { 1 } else { 0 };
        let mut builder = stream.launch_builder(&func);
        builder.arg(&t_dev_ptr);
        builder.arg(&atoms.centers_dev);
        builder.arg(&atoms.z_dev);
        builder.arg(&atoms.exps_dev);
        builder.arg(&phi_dev_ptr);
        builder.arg(&jet_dev_ptr);
        builder.arg(&n_ll);
        builder.arg(&dim_ll);
        builder.arg(&kc_ll);
        builder.arg(&nk_ll);
        builder.arg(&np_ll);
        builder.arg(&atoms.kernel_amp);
        builder.arg(&atoms.coeff_c);
        builder.arg(&atoms.coeff_power);
        builder.arg(&is_log_ll);
        builder.arg(&atoms.origin_value);
        // SAFETY: pointer/extent contract documented above; one block per row.
        unsafe { builder.launch(cfg) }
            .map_err(|err| format!("sae_duchon_basis_with_jet_device: launch: {err}"))?;
        stream
            .synchronize()
            .map_err(|err| format!("sae_duchon_basis_with_jet_device: sync: {err}"))?;
        Ok(width)
    }
}

#[cfg(target_os = "linux")]
pub use device::sae_periodic_basis_with_jet_device;
#[cfg(target_os = "linux")]
pub use device_duchon::{sae_duchon_basis_with_jet_device, sae_duchon_device_basis_width};

/// Non-Linux hosts have no CUDA driver; the torch bridge falls back to the
/// CPU `basis_with_jet` path on the typed refusal, which echoes the request so
/// a misrouted call is diagnosable from the message alone.
#[cfg(not(target_os = "linux"))]
pub fn sae_periodic_basis_with_jet_device(
    ordinal: usize,
    t_dev_ptr: u64,
    n: usize,
    num_harmonics: usize,
    phi_dev_ptr: u64,
    jet_dev_ptr: u64,
) -> Result<(), String> {
    Err(format!(
        "sae_periodic_basis_with_jet_device: CUDA path is Linux-only (requested \
         ordinal {ordinal}, n {n}, H {num_harmonics}, t@0x{t_dev_ptr:x}, \
         phi@0x{phi_dev_ptr:x}, jet@0x{jet_dev_ptr:x})"
    ))
}

/// Non-Linux stub — see [`sae_periodic_basis_with_jet_device`]'s note.
#[cfg(not(target_os = "linux"))]
pub fn sae_duchon_device_basis_width(
    ordinal: usize,
    centers: ndarray::ArrayView2<'_, f64>,
    m: usize,
) -> Result<usize, String> {
    Err(format!(
        "sae_duchon_device_basis_width: CUDA path is Linux-only (requested \
         ordinal {ordinal}, centers {:?}, m {m})",
        centers.dim()
    ))
}

/// Non-Linux stub — see [`sae_periodic_basis_with_jet_device`]'s note.
#[cfg(not(target_os = "linux"))]
pub fn sae_duchon_basis_with_jet_device(
    ordinal: usize,
    t_dev_ptr: u64,
    n: usize,
    centers: ndarray::ArrayView2<'_, f64>,
    m: usize,
    phi_dev_ptr: u64,
    jet_dev_ptr: u64,
) -> Result<usize, String> {
    Err(format!(
        "sae_duchon_basis_with_jet_device: CUDA path is Linux-only (requested \
         ordinal {ordinal}, n {n}, centers {:?}, m {m}, t@0x{t_dev_ptr:x}, \
         phi@0x{phi_dev_ptr:x}, jet@0x{jet_dev_ptr:x})",
        centers.dim()
    ))
}
