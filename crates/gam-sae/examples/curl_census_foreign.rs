//! Census a FOREIGN linear dictionary (a publicly released SAE) for the circles
//! its reconstruction objective is structurally blind to.
//!
//! `crate::manifold::curl` proves that a mean-zero circle's cone IS its 2-plane:
//! two linear atoms reconstruct it exactly, the residual is identically zero, and
//! a residual-driven trainer therefore never sees it. That is a claim about every
//! SAE anyone has trained, so it has to be tested on someone else's dictionary —
//! this binary runs the engine's own derived witness statistics
//! (`census_shattered_circles`) over an external decoder and its external
//! encoder's codes.
//!
//! Everything numeric happens here in Rust. The thin producer alongside this file
//! only downloads the artifact, runs its owner's encoder, and dumps raw arrays:
//!
//! ```text
//! <dir>/meta.json     {"n": …, "p": …, "k": …, "nnz": …}
//! <dir>/x.f32         n·p  row-major ambient activations
//! <dir>/w.f32         k·p  row-major decoder directions
//! <dir>/b.f32         p    decoder bias (subtracted from x before parsing)
//! <dir>/indptr.i64    n+1  CSR row pointers into the codes
//! <dir>/idx.i32       nnz  atom index per stored code
//! <dir>/val.f32       nnz  coefficient per stored code
//! ```
//!
//! Usage:
//! `curl_census_foreign <dir> <max_atoms> <min_cooccur> <subsample_rows> <permute_seed>
//!                      <coalesce_cos> <null_replicates> <fdr_alpha> <out.json>`
//!
//! `coalesce_cos` is the decoder cosine at or below which two rectified halves
//! merge into one signed direction (`-0.85` is the engine's own default). Passing
//! a value below `-1` disables coalescing entirely — the A/B control for the
//! launch-blocker claim, since that is precisely the screen a transcription
//! ships.
//!
//! `permute_seed = 0` censuses the dictionary as it is. Any other value runs the
//! CALIBRATION NULL: each atom's coefficient column is independently permuted
//! across rows, so every atom keeps its exact marginal amplitude law and firing
//! rate while the JOINT law across atoms is destroyed. That is the null the
//! witness statistics are about — κ = 1 is a claim that two coordinates are
//! jointly confined to a shell, which independent marginals cannot produce — and
//! it is the null a Gaussian-data arm cannot supply, because feeding a dictionary
//! data it was not trained on moves σ and with it the rate–distortion screen.
//! σ is measured on the UNPERMUTED codes and held fixed, so the null differs from
//! the real arm in the joint law and in nothing else.
//!
//! Alongside `<out.json>` the binary writes `<out.json>.planes.json`: for every
//! ACCEPTED pair, the ambient row indices and the in-plane parse `(α, β)`. That is
//! what a picture of a shattered circle is made of, and what lets a reader check
//! the ring against the tokens that drew it.
//!
//! `max_atoms` bounds the SEARCH — the census screens the `max_atoms` most
//! frequently firing atoms, because coalescing and co-occurrence are both
//! quadratic in the atom count. It does not touch the ACCEPTANCE rule, which is
//! `curl_verdict`'s derived conjunction and takes no configuration.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use gam_sae::manifold::{
    AtomFrame, AtomImage, CurlCensusConfig, RD_CROSSOVER_FACTOR, census_shattered_circles,
};
use ndarray::{Array1, Array2};

fn read_f32(path: &Path, expect: usize) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != expect * 4 {
        return Err(format!(
            "{}: expected {} f32 ({} bytes), found {} bytes",
            path.display(),
            expect,
            expect * 4,
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i64(path: &Path, expect: usize) -> Result<Vec<i64>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != expect * 8 {
        return Err(format!(
            "{}: expected {expect} i64, found {} bytes",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        })
        .collect())
}

fn read_i32(path: &Path, expect: usize) -> Result<Vec<i32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != expect * 4 {
        return Err(format!(
            "{}: expected {expect} i32, found {} bytes",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn meta_usize(meta: &serde_json::Value, key: &str) -> Result<usize, String> {
    meta.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("meta.json: missing unsigned integer field `{key}`"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 10 {
        return Err(format!(
            "usage: {} <dir> <max_atoms> <min_cooccur> <subsample_rows> <permute_seed> \
             <coalesce_cos> <null_replicates> <fdr_alpha> <out.json>",
            args.first().map(String::as_str).unwrap_or("curl_census_foreign")
        ));
    }
    let dir = Path::new(&args[1]);
    let max_atoms: usize = args[2]
        .parse()
        .map_err(|e| format!("max_atoms must be a positive integer: {e}"))?;
    let min_cooccur: usize = args[3]
        .parse()
        .map_err(|e| format!("min_cooccur must be a positive integer: {e}"))?;
    let subsample_rows: usize = args[4]
        .parse()
        .map_err(|e| format!("subsample_rows must be a positive integer: {e}"))?;
    let permute_seed: u64 = args[5]
        .parse()
        .map_err(|e| format!("permute_seed must be a non-negative integer: {e}"))?;
    let null_replicates: usize = args[7]
        .parse()
        .map_err(|e| format!("null_replicates must be a positive integer: {e}"))?;
    let fdr_alpha: f64 = args[8]
        .parse()
        .map_err(|e| format!("fdr_alpha must be a float in (0, 1): {e}"))?;
    let coalesce_cos: f64 = args[6]
        .parse()
        .map_err(|e| format!("coalesce_cos must be a float: {e}"))?;
    let out_path = Path::new(&args[9]);

    let meta_text = fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| format!("read meta.json: {e}"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&meta_text).map_err(|e| format!("parse meta.json: {e}"))?;
    let n = meta_usize(&meta, "n")?;
    let p = meta_usize(&meta, "p")?;
    let k = meta_usize(&meta, "k")?;
    let nnz = meta_usize(&meta, "nnz")?;
    eprintln!("[census] n={n} p={p} k={k} nnz={nnz}");

    let x = read_f32(&dir.join("x.f32"), n * p)?;
    let w = read_f32(&dir.join("w.f32"), k * p)?;
    let b = read_f32(&dir.join("b.f32"), p)?;
    let indptr = read_i64(&dir.join("indptr.i64"), n + 1)?;
    let idx = read_i32(&dir.join("idx.i32"), nnz)?;
    let val = read_f32(&dir.join("val.f32"), nnz)?;
    eprintln!("[census] arrays loaded");

    // Firing counts, so the search bound picks the most-used atoms rather than an
    // arbitrary prefix of the dictionary's index order.
    let mut fire_count = vec![0usize; k];
    for &a in &idx {
        let a = a as usize;
        if a < k {
            fire_count[a] += 1;
        }
    }
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| fire_count[j].cmp(&fire_count[i]).then(i.cmp(&j)));
    let kept: Vec<usize> = order
        .into_iter()
        .filter(|&a| fire_count[a] > 0)
        .take(max_atoms)
        .collect();
    let slot_of: HashMap<usize, usize> =
        kept.iter().enumerate().map(|(s, &a)| (a, s)).collect();
    eprintln!(
        "[census] screening {} atoms (of {} that ever fire); firing counts {}..{}",
        kept.len(),
        fire_count.iter().filter(|&&c| c > 0).count(),
        kept.last().map(|&a| fire_count[a]).unwrap_or(0),
        kept.first().map(|&a| fire_count[a]).unwrap_or(0)
    );

    // Dense per-atom coefficient columns for the kept atoms, and the full
    // reconstruction (over ALL atoms — the noise scale must be measured against
    // what the dictionary actually reconstructs, not against the search subset).
    let mut coefs = Array2::<f64>::zeros((kept.len(), n));
    let mut sse = 0.0_f64;
    let mut recon = vec![0.0_f64; p];
    for row in 0..n {
        for (j, slot) in recon.iter_mut().enumerate() {
            *slot = b[j] as f64;
        }
        let lo = indptr[row] as usize;
        let hi = indptr[row + 1] as usize;
        for e in lo..hi {
            let atom = idx[e] as usize;
            let c = val[e] as f64;
            if c == 0.0 {
                continue;
            }
            if let Some(&slot) = slot_of.get(&atom) {
                coefs[[slot, row]] = c;
            }
            let wo = atom * p;
            for (j, r) in recon.iter_mut().enumerate() {
                *r += c * w[wo + j] as f64;
            }
        }
        let xo = row * p;
        for (j, r) in recon.iter().enumerate() {
            let d = x[xo + j] as f64 - *r;
            sse += d * d;
        }
    }
    let sigma = (sse / (n as f64 * p as f64)).sqrt().max(1e-12);
    // Variance ABOUT THE MEAN: a residual stream carries a large constant
    // component, so an uncentred energy ratio flatters every dictionary.
    let mut col_mean = vec![0.0_f64; p];
    for row in 0..n {
        for (j, m) in col_mean.iter_mut().enumerate() {
            *m += x[row * p + j] as f64;
        }
    }
    for m in col_mean.iter_mut() {
        *m /= n as f64;
    }
    let mut x_var = 0.0_f64;
    for row in 0..n {
        for (j, m) in col_mean.iter().enumerate() {
            let d = x[row * p + j] as f64 - *m;
            x_var += d * d;
        }
    }
    x_var /= (n * p) as f64;
    let fvu = sigma * sigma / x_var;
    eprintln!(
        "[census] sigma (residual RMS per coordinate) = {sigma:.6}; \
         centred ambient sd = {:.6}; fraction of variance unexplained = {fvu:.4}",
        x_var.sqrt()
    );

    if permute_seed != 0 {
        // Fisher–Yates per atom with a splitmix64 stream, so the null is exactly
        // reproducible from `permute_seed` and independent across atoms.
        let mut state = permute_seed;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for slot in 0..kept.len() {
            for i in (1..n).rev() {
                let j = (next() % (i as u64 + 1)) as usize;
                let tmp = coefs[[slot, i]];
                coefs[[slot, i]] = coefs[[slot, j]];
                coefs[[slot, j]] = tmp;
            }
        }
        eprintln!("[census] CALIBRATION NULL: coefficient columns permuted (seed {permute_seed})");
    }

    // The parse must be taken on the ambient rows the atoms live in, so every
    // kept atom's image is the rank-one `coef[r] · w_a` its own decoder writes.
    let dirs: Vec<Array1<f64>> = kept
        .iter()
        .map(|&a| Array1::from_iter((0..p).map(|j| w[a * p + j] as f64)))
        .collect();
    let frames: Vec<AtomFrame<'_>> = kept
        .iter()
        .enumerate()
        .map(|(slot, &a)| AtomFrame {
            id: a,
            dir: dirs[slot].clone(),
            active: (0..n).map(|r| coefs[[slot, r]] > 0.0).collect(),
            image: AtomImage::RankOne {
                coef: coefs.row(slot),
                dir: dirs[slot].view(),
            },
        })
        .collect();

    let cfg = CurlCensusConfig {
        harmonics: 1,
        // The engine's own defaults for the antipodal merge (see `CurlConfig`).
        coalesce_cos_threshold: coalesce_cos,
        coalesce_max_overlap: 0.25,
        min_cooccurrence: min_cooccur,
        subsample_rows,
        null_replicates,
        fdr_alpha,
    };
    eprintln!("[census] running the screen …");
    let census = census_shattered_circles(&frames, n, p, sigma, &cfg)?;
    eprintln!(
        "[census] {} signed directions ({} of them antipodal merges) → {} screened pairs, \
         {} pass the per-pair screen, {} are e-BH discoveries at FDR {} \
         (ledger threshold e ≥ {:.0}, max attainable e = {})",
        census.n_signed,
        census.n_coalesced,
        census.pairs.len(),
        census.pairs.iter().filter(|pair| pair.verdict.geometry_ok).count(),
        census.accepted(),
        census.fdr_alpha,
        census.ebh_threshold,
        census.replicates_drawn + 1
    );
    eprintln!(
        "[census] {} pairs beat every surrogate; making those discoveries needs \
         {:.0} replicates (this run drew {})",
        census.n_max_e, census.replicates_required, census.replicates_drawn
    );

    let rows: Vec<serde_json::Value> = census
        .pairs
        .iter()
        .map(|pair| {
            serde_json::json!({
                "members_a": pair.members_a,
                "members_b": pair.members_b,
                "n_co_fire": pair.n_co_fire,
                "kappa": pair.verdict.kappa,
                "kappa_se": pair.verdict.kappa_se,
                "z_below_gaussian": pair.verdict.z_below_gaussian,
                "resultant1": pair.verdict.resultant1,
                "resultant2": pair.verdict.resultant2,
                "radius": pair.verdict.radius,
                "radius_over_sigma": pair.verdict.radius / census.sigma,
                "gain_nats_per_row": pair.verdict.gain_nats_per_row,
                "net_evidence_nats": pair.verdict.net_evidence_nats,
                "screen_accepted": pair.verdict.geometry_ok,
                "kappa_gate": pair.verdict.recommend_curl,
                "rank_rho": pair.rank_rho,
                "p_value": pair.p_value,
                "e_value": pair.e_value,
                "null_rho_mean": pair.null_rho_mean,
                "null_rho_sd": pair.null_rho_sd,
                "accepted": pair.fdr_discovery,
            })
        })
        .collect();
    let planes: Vec<serde_json::Value> = census
        .pairs
        .iter()
        .filter_map(|pair| {
            if !pair.fdr_discovery {
                return None;
            }
            let plane = pair.accepted_geometry.as_ref()?;
            Some(serde_json::json!({
                "members_a": pair.members_a,
                "members_b": pair.members_b,
                "kappa": pair.verdict.kappa,
                "z_below_gaussian": pair.verdict.z_below_gaussian,
                "radius_over_sigma": pair.verdict.radius / census.sigma,
                "net_evidence_nats": pair.verdict.net_evidence_nats,
                "p_value": pair.p_value,
                "e_value": pair.e_value,
                "rank_rho": pair.rank_rho,
                "rows": plane.rows,
                "alpha": plane.alpha.to_vec(),
                "beta": plane.beta.to_vec(),
                // The plane's orthonormal ambient frame, so a plane found in one
                // dictionary can be matched to a plane found in an independently
                // trained one by principal angles — replication across artifacts,
                // not merely across seeds of the same artifact.
                "e1": plane.e1.to_vec(),
                "e2": plane.e2.to_vec(),
            }))
        })
        .collect();
    let planes_path = out_path.with_extension("planes.json");
    fs::write(
        &planes_path,
        serde_json::to_string(&serde_json::json!({ "planes": planes }))
            .map_err(|e| format!("serialise planes: {e}"))?,
    )
    .map_err(|e| format!("write {}: {e}", planes_path.display()))?;
    eprintln!("[census] wrote {} accepted planes to {}", planes.len(), planes_path.display());

    let doc = serde_json::json!({
        "n_rows": n,
        "ambient_p": p,
        "dictionary_k": k,
        "screened_atoms": kept.len(),
        "min_cooccurrence": min_cooccur,
        "subsample_rows": subsample_rows,
        "sigma": census.sigma,
        "ambient_sd": x_var.sqrt(),
        "fvu": fvu,
        "rd_crossover_radius": census.sigma * RD_CROSSOVER_FACTOR,
        "n_signed": census.n_signed,
        "n_coalesced": census.n_coalesced,
        "n_pairs": census.pairs.len(),
        "n_screen_accepted": census.pairs.iter().filter(|pair| pair.verdict.geometry_ok).count(),
        "n_accepted": census.accepted(),
        "fdr_alpha": census.fdr_alpha,
        "ebh_threshold": census.ebh_threshold,
        "n_max_e": census.n_max_e,
        "replicates_required": census.replicates_required,
        "null_replicates": census.replicates_drawn,
        "permute_seed": permute_seed,
        "coalesce_cos_threshold": coalesce_cos,
        "pairs": rows,
    });
    let mut file =
        fs::File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;
    file.write_all(
        serde_json::to_string(&doc)
            .map_err(|e| format!("serialise census: {e}"))?
            .as_bytes(),
    )
    .map_err(|e| format!("write {}: {e}", out_path.display()))?;
    eprintln!("[census] wrote {}", out_path.display());
    Ok(())
}

fn main() -> Result<(), String> {
    // `std::process::exit` is banned by the root build script (it skips
    // destructors and makes the exit path untestable), and one hit there blocks
    // every wheel build in the repository. Returning the error is the sanctioned
    // idiom and keeps the same nonzero exit status.
    run().map_err(|error| format!("curl_census_foreign: {error}"))
}
