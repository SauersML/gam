/*!
# I/O Utilities for Saving MCMC Data to Arrow

This module provides functions to save MCMC sample data to arrow files. Enable via the `arrow` feature.
*/

use ndarray::{Array3, Axis};
use std::error::Error;
use std::fs::File;
use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Float64Builder, UInt32Builder},
    datatypes::{DataType, Field, Schema},
    ipc::writer::FileWriter,
    record_batch::RecordBatch,
};

/**
Saves MCMC data (chain × observation × dimension) as an Apache Arrow (IPC) file.

# Arguments

* `data` - An `Array3<T>` object.
* `filename` - The path to the Arrow (IPC) file to create.

# Type Parameters

* `T` - Must implement `Into<f64> + Copy`. Each dimension value will be
  converted to `f64` in the Arrow output.

# Returns

Returns `Ok(())` if the file was successfully written. Otherwise, an error
is returned if any I/O or Arrow related error occurs.

# Example

```rust
use general_mcmc::io::arrow::save_arrow;
use ndarray::arr3;

// Suppose we have 2 chains, each with 2 observations, and 3 dimensions.
let data = arr3(&[[[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]],
                [[10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0]]]);

save_arrow(&data, "/tmp/output.arrow")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_arrow<T: Into<f64> + Copy>(
    data: &Array3<T>,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    // Compute dimensions (but don't return early if empty)
    let shape = data.shape();
    let (n_chains, n_dims) = (shape[0], shape[2]);

    // Define the schema: chain (UInt32), observation (UInt32), dim_0..dim_n (Float64)
    let mut fields = vec![
        Field::new("chain", DataType::UInt32, false),
        Field::new("observation", DataType::UInt32, false),
    ];
    for dim_idx in 0..n_dims {
        fields.push(Field::new(
            format!("dim_{}", dim_idx),
            DataType::Float64,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Create our Arrow builders for chain & observation (UInt32) + each dim (Float64).
    // Even if no data, we need them to create an empty batch.
    let mut chain_builder = UInt32Builder::new();
    let mut observation_builder = UInt32Builder::new();
    let mut dim_builders: Vec<Float64Builder> =
        (0..n_dims).map(|_| Float64Builder::new()).collect();

    // If there's actual data, fill the builders
    if n_chains > 0 {
        for (chain_idx, chain) in data.axis_iter(Axis(0)).enumerate() {
            for (observation_idx, observation) in chain.axis_iter(Axis(0)).enumerate() {
                chain_builder.append_value(chain_idx as u32);
                observation_builder.append_value(observation_idx as u32);

                for (dim_idx, val) in observation.iter().enumerate() {
                    dim_builders[dim_idx].append_value((*val).into());
                }
            }
        }
    }

    // Convert the builders into Arrow arrays
    let chain_array = Arc::new(chain_builder.finish()) as ArrayRef;
    let observation_array = Arc::new(observation_builder.finish()) as ArrayRef;

    let mut dim_arrays = Vec::with_capacity(n_dims);
    for mut builder in dim_builders {
        dim_arrays.push(Arc::new(builder.finish()) as ArrayRef);
    }

    // Combine into a single RecordBatch
    let mut arrays = vec![chain_array, observation_array];
    arrays.extend(dim_arrays);
    let record_batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // Write the RecordBatch (possibly zero rows) to an Arrow IPC file
    let file = File::create(filename)?;
    let mut writer = FileWriter::try_new(file, &schema)?;
    writer.write(&record_batch)?;
    writer.finish()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Float64Array, UInt32Array},
        ipc::reader::FileReader,
    };
    use ndarray::arr3;
    use std::fs;
    use std::{error::Error, fs::File};
    use tempfile::NamedTempFile;

    // --- Arrow Tests ---

    /// Test saving empty data to Arrow (zero chains).
    #[test]
    fn test_save_arrow_empty_data() {
        let data = arr3::<f32, 0, 0>(&[]); // no chains
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_arrow(&data, filename);
        assert!(
            result.is_ok(),
            "Saving empty data to Arrow failed: {:?}",
            result
        );

        // The file should exist, but there's effectively no data in it.
        let metadata = fs::metadata(filename).unwrap();
        assert!(metadata.len() > 0, "Arrow file is unexpectedly empty");

        // (Optional) We can verify that the file indeed has an empty RecordBatch.
        let file = File::open(filename).unwrap();
        let mut reader = FileReader::try_new(file, None).unwrap();
        // Should have exactly one batch, with 0 rows
        if let Some(Ok(batch)) = reader.next() {
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 2);
        } else {
            panic!("Expected an empty batch, found none or an error");
        }
        // No second batch
        assert!(reader.next().is_none());
    }

    /// Test saving a single chain/single observation (with single dimension) to Arrow using `f64`.
    #[test]
    fn test_save_arrow_single_chain_single_observation_f64() -> Result<(), Box<dyn Error>> {
        let data = arr3(&[[[42f64]]]);
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        // Write Arrow
        save_arrow(&data, filename)?;

        // Read back the file to verify
        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0, "Arrow file is unexpectedly empty");

        let file = File::open(filename)?;
        let mut reader = FileReader::try_new(file, None)?;
        let batch = reader.next().expect("No record batch found")?.clone(); // read first batch
        assert!(reader.next().is_none(), "Expected only one batch");

        // We expect 1 row, 1 dimension => columns = chain(0), observation(0), dim_0(42.0)
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 3);

        // Downcast columns
        let chain_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let observation_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(chain_array.value(0), 0);
        assert_eq!(observation_array.value(0), 0);
        assert!((dim0_array.value(0) - 42.0).abs() < f64::EPSILON);

        Ok(())
    }

    /// Test multiple chains, multiple observations, multiple dimensions with `f32`.
    #[test]
    fn test_save_arrow_multi_chain_f32() -> Result<(), Box<dyn Error>> {
        // 2 chains, 2 observations each, 2 dims => total 4 rows
        // chain=0, observation=0 => dims=[1.0, 2.5]
        // chain=0, observation=1 => dims=[3.0, 4.5]
        // chain=1, observation=0 => dims=[10.0, 20.5]
        // chain=1, observation=1 => dims=[30.0, 40.5]
        let data = arr3(&[
            [[1f32, 2.5f32], [3f32, 4.5f32]],
            [[10f32, 20.5f32], [30f32, 40.5f32]],
        ]);
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        save_arrow(&data, filename)?;

        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0);

        // Read back Arrow
        let file = File::open(filename)?;
        let mut reader = FileReader::try_new(file, None)?;
        let batch = reader.next().expect("No record batch found")?.clone();
        assert!(reader.next().is_none());

        // Check shape: 4 rows, columns = chain, observation, dim_0, dim_1 => total 4 columns
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 4);

        let chain_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let observation_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let dim1_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Row 0: chain=0, observation=0, dim0=1.0, dim1=2.5
        assert_eq!(chain_array.value(0), 0);
        assert_eq!(observation_array.value(0), 0);
        assert!((dim0_array.value(0) - 1.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(0) - 2.5).abs() < f64::EPSILON);

        // Row 1: chain=0, observation=1, dim0=3.0, dim1=4.5
        assert_eq!(chain_array.value(1), 0);
        assert_eq!(observation_array.value(1), 1);
        assert!((dim0_array.value(1) - 3.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(1) - 4.5).abs() < f64::EPSILON);

        // Row 2: chain=1, observation=0, dim0=10.0, dim1=20.5
        assert_eq!(chain_array.value(2), 1);
        assert_eq!(observation_array.value(2), 0);
        assert!((dim0_array.value(2) - 10.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(2) - 20.5).abs() < f64::EPSILON);

        // Row 3: chain=1, observation=1, dim0=30.0, dim1=40.5
        assert_eq!(chain_array.value(3), 1);
        assert_eq!(observation_array.value(3), 1);
        assert!((dim0_array.value(3) - 30.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(3) - 40.5).abs() < f64::EPSILON);

        Ok(())
    }

    /// Test saving data with an integer type (i32) to Arrow
    /// to ensure `T: Into<f64> + Copy` is satisfied with different numeric types.
    #[test]
    fn test_save_arrow_integer_data() {
        let data = arr3(&[
            [[100, 200, 300], [400, 500, 600]],
            [[700, 800, 900], [1000, 1100, 1200]],
        ]);
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_arrow(&data, filename);
        assert!(
            result.is_ok(),
            "Saving integer data to Arrow failed: {:?}",
            result
        );

        let metadata = fs::metadata(filename).unwrap();
        assert!(metadata.len() > 0);
    }
}
