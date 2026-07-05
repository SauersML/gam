//! Cross-model transport on real banked activation arrays.
//!
//! Usage:
//!
//! ```text
//! cargo run -p gam-sae --example cross_model_transport_real -- \
//!   <qwen3-8b-resid-L18.npy> <qwen3-35b-msae-L17.npy> [max_rows]
//! ```
//!
//! The coordinate transported here is the top-2 activation-plane angle. For the
//! Qwen3-8B residual stream, the known position-0/sink direction is peeled by
//! taking SVD scores 1 and 2 after the leading component; for the 35B MSAE
//! activation file, scores 0 and 1 are used. This handles the 4096-vs-2048
//! ambient dimension mismatch before transport: the fitted map only sees paired
//! circle angles.

use gam_sae::identifiability::thin_svd_scores;
use gam_sae::inference::cross_model_transport::{
    ModelCoordinate, UniversalityVerdict, fit_cross_model_transport,
};
use ndarray::Array2;
use std::path::Path;
use std::process::ExitCode;

const DEFAULT_MAX_ROWS: usize = 20_000;

fn main() -> ExitCode {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "cross_model_transport_real".to_string());
    let Some(path_a) = args.next() else {
        eprintln!("usage: {program} <qwen3-8b-resid-L18.npy> <qwen3-35b-msae-L17.npy> [max_rows]");
        return ExitCode::from(2);
    };
    let Some(path_b) = args.next() else {
        eprintln!("usage: {program} <qwen3-8b-resid-L18.npy> <qwen3-35b-msae-L17.npy> [max_rows]");
        return ExitCode::from(2);
    };
    let max_rows = match args.next() {
        Some(raw) => match raw.parse::<usize>() {
            Ok(v) if v > 0 => v,
            Ok(_) | Err(_) => {
                eprintln!("max_rows must be a positive integer");
                return ExitCode::from(2);
            }
        },
        None => DEFAULT_MAX_ROWS,
    };

    match run(Path::new(&path_a), Path::new(&path_b), max_rows) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[cross_model_transport_real] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(path_a: &Path, path_b: &Path, max_rows: usize) -> Result<(), String> {
    let (n_a_full, p_a, x_a) = read_npy_subsample_f64(path_a, max_rows)?;
    let (n_b_full, p_b, x_b) = read_npy_subsample_f64(path_b, max_rows)?;
    let n = x_a.nrows().min(x_b.nrows());
    if n < 16 {
        return Err(format!("need at least 16 paired rows, got {n}"));
    }

    let x_a = x_a.slice(ndarray::s![0..n, ..]).to_owned();
    let x_b = x_b.slice(ndarray::s![0..n, ..]).to_owned();
    let scores_a = thin_svd_scores(x_a.view(), 3)?;
    let scores_b = thin_svd_scores(x_b.view(), 2)?;
    let coord_a = ModelCoordinate::from_plane_scores(
        "Qwen3-8B resid L18 top-2 after sink peel",
        scores_a.view(),
        1,
        2,
    )?;
    let coord_b = ModelCoordinate::from_plane_scores(
        "Qwen3.6-35B-A3B MSAE L17 top-2",
        scores_b.view(),
        0,
        1,
    )?;
    let report = fit_cross_model_transport(&coord_a, &coord_b)?;

    println!("=== Cross-model transport: Qwen3-8B L18 → Qwen3.6-35B-A3B L17 ===");
    println!("arrays: A N_full={n_a_full} used={n} p={p_a}; B N_full={n_b_full} used={n} p={p_b}");
    println!(
        "coordinate: model-local top-2 plane angle; A uses SVD score columns 1,2 to peel the known top sink direction; B uses columns 0,1"
    );
    println!(
        "pairing caveat: rows are paired by file order; without token ids this is a row-aligned activation proxy, not an independently verified matched-token audit"
    );
    if let Some(circle) = &report.circle {
        println!(
            "O(2): class={:?} winding={:+} phase={:+.6} rad ({:+.3} deg) defect={:.6e} gauge_scale={:.6e}",
            circle.class,
            circle.winding,
            circle.phase,
            circle.phase_degrees(),
            circle.defect,
            report.gauge_defect_scale
        );
    }
    println!(
        "transport: degree={:?} degree_concentration={:?} topology_preserved={} min_directional_derivative={:.6e}",
        report.fit.degree,
        report.fit.degree_concentration,
        report.fit.topology_preserved,
        report.fit.min_directional_derivative
    );
    println!(
        "defect: isometry={:.6e} se={:.6e} residual_rms={:.6e} edf={:.3} lambda={:.6e} noise_var={:.6e}",
        report.fit.isometry_defect,
        report.fit.isometry_defect_se,
        report.fit.residual_rms,
        report.fit.edf,
        report.fit.smoothing_lambda,
        report.fit.noise_variance
    );
    println!("verdict: {}", report.verdict.label());
    if report.verdict == UniversalityVerdict::ConsistentWithSharedFeatureWithinNoise {
        println!(
            "claim: consistent with a shared feature within noise; not evidence of exact identity"
        );
    } else {
        println!(
            "claim: measured transport obstruction/reparameterization; not an exact shared feature claim"
        );
    }
    Ok(())
}

fn parse_npy_header(
    head: &[u8],
    path: &Path,
) -> Result<(usize, usize, usize, bool, usize), String> {
    if head.len() <= 12 || &head[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let major = head[6];
    let (header_len, data_off) = if major >= 2 {
        let header_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        (header_len, 12 + header_len)
    } else {
        let header_len = u16::from_le_bytes([head[8], head[9]]) as usize;
        (header_len, 10 + header_len)
    };
    if data_off > head.len() {
        return Err(format!(
            "{}: header exceeds initial read buffer",
            path.display()
        ));
    }
    let header = std::str::from_utf8(&head[data_off - header_len..data_off])
        .map_err(|err| format!("{}: header is not utf8: {err}", path.display()))?;
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    if !(is_f4 || is_f2) {
        return Err(format!(
            "{}: expected little-endian <f4 or <f2; header: {header}",
            path.display()
        ));
    }
    if !(header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false")) {
        return Err(format!(
            "{}: expected C-order; header: {header}",
            path.display()
        ));
    }
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| format!("{}: missing shape key", path.display()))?
        + "'shape':".len();
    let paren_open = header[shape_start..]
        .find('(')
        .ok_or_else(|| format!("{}: missing shape open paren", path.display()))?
        + shape_start
        + 1;
    let paren_close = header[paren_open..]
        .find(')')
        .ok_or_else(|| format!("{}: missing shape close paren", path.display()))?
        + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|token| token.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!(
            "{}: expected a 2-D array, got {dims:?}",
            path.display()
        ));
    }
    let elem = if is_f4 { 4 } else { 2 };
    Ok((dims[0], dims[1], elem, is_f4, data_off))
}

fn read_npy_subsample_f64(path: &Path, cap: usize) -> Result<(usize, usize, Array2<f64>), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file =
        std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut head = vec![0u8; 512];
    file.read_exact(&mut head)
        .map_err(|err| format!("read header {}: {err}", path.display()))?;
    let (n_full, p, elem, is_f4, data_off) = parse_npy_header(&head, path)?;
    let take = cap.min(n_full);
    let stride = (n_full / take).max(1);
    let row_bytes = p
        .checked_mul(elem)
        .ok_or_else(|| format!("{}: row byte size overflow", path.display()))?;
    let mut buf = vec![0u8; row_bytes];
    let mut out = Array2::<f64>::zeros((take, p));
    for i in 0..take {
        let src_row = i * stride;
        let off = data_off as u64 + (src_row as u64) * (row_bytes as u64);
        file.seek(SeekFrom::Start(off))
            .map_err(|err| format!("seek {}: {err}", path.display()))?;
        file.read_exact(&mut buf)
            .map_err(|err| format!("read row {src_row} {}: {err}", path.display()))?;
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] =
                    f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]) as f64;
            }
        } else {
            for c in 0..p {
                let b = c * 2;
                out[[i, c]] = f16_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]])) as f64;
            }
        }
    }
    Ok((n_full, p, out))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut m = mant as u32;
            let mut e: i32 = -1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + 1 + e) as u32;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}
