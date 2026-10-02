//! #2946 pins for the cached reader Gram: every packed entry against the full product within its derived rounding band,
//! across a tile boundary, with a corrupted entry the pin must reject; a streamed tile writes the cached rows' words;
//! and footprints the ledger cannot admit are refused typed without allocating.

use super::{ReaderGram, ReaderGramError, fill_upper_rows, packed_bytes, row_offset, upper_tile_rows};
use gam_linalg::faer_ndarray::{fast_ab, fast_atb};
use gam_linalg::roundoff::accumulation_band;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn uniform_matrix(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows, cols));
    for entry in out.iter_mut() {
        *entry = rng.random_range(-1.0..1.0);
    }
    out
}

/// A block of `width` writers in `outputs` dimensions under a symmetric positive definite metric, returning `U` and
/// `M U`.
fn writers_and_metric_writers(seed: u64, outputs: usize, width: usize) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let writers = uniform_matrix(&mut rng, outputs, width);
    let root = uniform_matrix(&mut rng, outputs, outputs);
    let metric = fast_ab(&root.t(), &root) + Array2::<f64>::eye(outputs);
    let metric_writers = fast_ab(&metric, &writers);
    (writers, metric_writers)
}

/// Entries of `gram` farther from the full product `Uᵀ M U` than both products' rounding: each entry is a `p`-term
/// inner product, so each route errs by at most `γ_p Σ_i |U_ij (MU)_ik|` (Higham, ASNA §3.1).
fn violations(gram: &ReaderGram, writers: &Array2<f64>, metric_writers: &Array2<f64>) -> usize {
    let full = fast_atb(writers, metric_writers);
    let outputs = writers.nrows();
    let width = gram.width();
    let mut count = 0;
    for unit in 0..width {
        let row = gram.upper_row(unit);
        for other in unit..width {
            let absolute: f64 = (0..outputs)
                .map(|output| (writers[[output, unit]] * metric_writers[[output, other]]).abs())
                .sum();
            let band = 2.0 * accumulation_band(outputs, absolute);
            if (row[other - unit] - full[[unit, other]]).abs() > band {
                count += 1;
            }
        }
    }
    count
}

/// The narrowest block whose Gram spans more than `tiles` tiles under the library tile rule. The rule keeps a tile near
/// a fixed byte target, so every width up to the target's square root is one tile; a fixture sized from the rule
/// crosses tile boundaries whatever target the rule carries.
fn width_spanning(tiles: usize) -> usize {
    (1..)
        .find(|&width| upper_tile_rows(width) * tiles < width)
        .expect("a width past every tile target exists")
}

#[test]
fn packed_rows_match_the_full_product_across_a_tile_boundary() {
    let (outputs, width) = (3, width_spanning(2));
    let (writers, metric_writers) = writers_and_metric_writers(0x2946_0DA7, outputs, width);
    let gram = ReaderGram::new(writers.view(), metric_writers.view()).expect("the fixture's Gram is admitted");
    assert_eq!(
        gram.resident_bytes(),
        width * (width + 1) / 2 * std::mem::size_of::<f64>(),
        "the ledger must hold exactly the packed upper triangle"
    );
    assert_eq!(row_offset(width, width), width * (width + 1) / 2, "the packed length is the triangle's size");
    assert_eq!(violations(&gram, &writers, &metric_writers), 0, "a packed entry left the full product's band");
    // Positive control: one entry past the second tile's first row, moved by far more than its band, must be seen.
    let mut corrupted = gram;
    let offset = row_offset(width, upper_tile_rows(width) + 8) + 7;
    corrupted.packed[offset] += 1.0;
    assert!(
        violations(&corrupted, &writers, &metric_writers) > 0,
        "a corrupted packed entry passed the pin, so the pin cannot fail"
    );
}

#[test]
fn a_streamed_tile_writes_the_cached_rows_words() {
    let (outputs, width) = (4, width_spanning(2));
    let (writers, metric_writers) = writers_and_metric_writers(0x2946_57EA, outputs, width);
    let gram = ReaderGram::new(writers.view(), metric_writers.view()).expect("the fixture's Gram is admitted");
    let tile = upper_tile_rows(width);
    let mut tiles = 0;
    for start in (0..width).step_by(tile) {
        let end = (start + tile).min(width);
        let mut streamed = vec![0.0; row_offset(width, end) - row_offset(width, start)];
        fill_upper_rows(writers.view(), metric_writers.view(), start, end, &mut streamed)
            .expect("a tile of the block is admitted");
        let mut cursor = 0;
        for unit in start..end {
            for (position, (&cached, &fresh)) in gram.upper_row(unit).iter().zip(&streamed[cursor..]).enumerate() {
                assert_eq!(
                    cached.to_bits(),
                    fresh.to_bits(),
                    "entry D[{unit}, {}] differs between the cache and a streamed tile",
                    unit + position
                );
            }
            cursor += width - unit;
        }
        assert_eq!(cursor, streamed.len(), "the streamed tile must fill exactly its packed rows");
        tiles += 1;
    }
    assert!(tiles > 1, "the fixture must stream more than one tile");
    let mut short = vec![0.0; 3];
    let refusal = fill_upper_rows(writers.view(), metric_writers.view(), 0, tile, &mut short)
        .expect_err("a tile buffer of the wrong length must be refused");
    assert!(
        matches!(refusal, ReaderGramError::DimensionMismatch { .. }),
        "a wrong tile buffer must be refused as a dimension mismatch, got {refusal}"
    );
}

#[test]
fn footprints_the_ledger_cannot_admit_are_refused_typed() {
    // Zero output rows, so no writer storage exists, but the width asks for a packed Gram of about 1.1 PB.
    let wide = Array2::<f64>::zeros((0, 1 << 24));
    let refusal = ReaderGram::new(wide.view(), wide.view())
        .expect_err("a petabyte reader Gram must not be admitted on any host that runs this test");
    assert!(
        matches!(refusal, ReaderGramError::Admission { .. }),
        "a footprint beyond the ledger must be refused as an admission failure, got {refusal}"
    );
    let overflow = Array2::<f64>::zeros((0, 1 << 33));
    assert_eq!(packed_bytes(1 << 33), None, "a 2^33-unit packed Gram has no representable byte count");
    let refusal = ReaderGram::new(overflow.view(), overflow.view())
        .expect_err("an unrepresentable footprint must be refused");
    assert!(
        matches!(refusal, ReaderGramError::SizeOverflow { width } if width == 1 << 33),
        "an unrepresentable footprint must be refused as a size overflow, got {refusal}"
    );
}
