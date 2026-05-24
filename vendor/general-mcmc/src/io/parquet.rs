/*!
# I/O Utilities for Saving MCMC Data to Parquet

This module provides functions to save MCMC sample data to Parquet files. Enable via the `parquet` feature.
*/

use arrow::array::{ArrayRef, Float64Builder, UInt32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use burn::prelude::*;
use ndarray::{Array3, Axis};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;

/**
Saves MCMC data (chain × observation × dimension) to a Parquet file.

# Arguments

* `data` - An `Array3<T>` object where.
* `filename` - The path to the Parquet file to create.

# Type Parameters

* `T` - Must implement `Into<f64> + Copy`. Each dimension value is converted
  to `f64` in the underlying Arrow arrays before Parquet encoding.

# Returns

Returns `Ok(())` if the file was written successfully, otherwise returns an
error wrapped in a `Box<dyn Error>`.

# Example

```rust
use general_mcmc::io::parquet::save_parquet;
use ndarray::arr3;

// 1 chain, 2 observations, 3 dimensions
let data = arr3(&[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);

save_parquet(&data, "/tmp/output.parquet")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_parquet<T: Into<f64> + Copy>(
    data: &Array3<T>,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    // Define the Arrow schema: chain (UInt32), observation (UInt32), then dim_0..dim_n (Float64)
    let mut fields = vec![
        Field::new("chain", DataType::UInt32, false),
        Field::new("observation", DataType::UInt32, false),
    ];
    let n_dims = data.shape()[2];
    for dim_idx in 0..n_dims {
        fields.push(Field::new(
            format!("dim_{}", dim_idx),
            DataType::Float64,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Create builders for each column
    let mut chain_builder = UInt32Builder::new();
    let mut observation_builder = UInt32Builder::new();
    let mut dim_builders: Vec<Float64Builder> =
        (0..n_dims).map(|_| Float64Builder::new()).collect();

    // Populate builders
    for (chain_idx, chain) in data.axis_iter(Axis(0)).enumerate() {
        for (observation_idx, observation) in chain.axis_iter(Axis(0)).enumerate() {
            chain_builder.append_value(chain_idx as u32);
            observation_builder.append_value(observation_idx as u32);

            for (dim_idx, val) in observation.iter().enumerate() {
                dim_builders[dim_idx].append_value((*val).into());
            }
        }
    }

    // Convert builders into Arrow arrays
    let chain_array = Arc::new(chain_builder.finish()) as ArrayRef;
    let observation_array = Arc::new(observation_builder.finish()) as ArrayRef;
    let mut dim_arrays = Vec::with_capacity(n_dims);
    for mut builder in dim_builders {
        dim_arrays.push(Arc::new(builder.finish()) as ArrayRef);
    }

    // Create a single RecordBatch
    let mut arrays = vec![chain_array, observation_array];
    arrays.extend(dim_arrays);
    let record_batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // Create the Parquet writer and write the batch
    let file = File::create(filename)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    writer.write(&record_batch)?;
    // Close the writer to ensure metadata is written
    writer.close()?;

    Ok(())
}

/**
Saves a 3D Burn tensor (observation × chain × dimension) to a Parquet file.

The tensor’s data is first converted into Apache Arrow arrays with the following schema:
  - `"observation"` (UInt32): observation index,
  - `"chain"` (UInt32): chain index,
  - `"dim_0"`, `"dim_1"`, … (Float64): dimension values.

All chains must have the same shape. The function writes a single RecordBatch containing all data,
where each row corresponds to one (observation, chain) combination. For example, the coordinate `d` of
data point `s` that chain `c` generated is assumed to be in `tensor[n][s][d]`.

# Arguments
* `tensor` - A reference to a Burn tensor with shape `[num_observations, num_chains, num_dimensions]`.
* `filename` - The file path where the Parquet data will be written.

# Type Parameters
* `B` - The backend type.
* `K` - The tensor kind.
* `T` - The scalar type; must implement `Into<f64>` and `burn::tensor::Element`.

# Returns
Returns `Ok(())` if the file was written successfully, or an error if any I/O or
Arrow/Parquet error occurs.

# Example
```rust
use burn::tensor::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use general_mcmc::io::parquet::save_parquet_tensor;
let tensor = Tensor::<NdArray, 3>::from_floats(
    [
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        [[1.01, 2.01], [1.11, 2.11], [1.21, 2.21]],
        [[1.02, 2.02], [1.12, 2.12], [1.22, 2.22]],
        [[1.03, 2.03], [1.13, 2.13], [1.23, 2.23]],
    ],
    &NdArrayDevice::Cpu,
);
save_parquet_tensor::<NdArray, _, f32>(&tensor, "/tmp/output.parquet")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_parquet_tensor<B, K, T>(
    tensor: &Tensor<B, 3, K>,
    filename: &str,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    K: burn::tensor::TensorKind<B>,
    T: Into<f64> + burn::tensor::Element,
    K: burn::tensor::BasicOps<B>,
{
    // Extract data and shape from the tensor.
    let shape = tensor.dims();
    let data = tensor.to_data();
    let (num_observations, num_chains, num_dims) = (shape[0], shape[1], shape[2]);
    let flat: Vec<T> = data.to_vec::<T>().map_err(|e| {
        format!("Conversion of data to Vec failed.\nData: {data:?}.\nError: {e:?}.",)
    })?;

    // Build Arrow schema: observation (UInt32), chain (UInt32), then dim_0...dim_{num_dims-1} (Float64)
    let mut fields = vec![
        Field::new("observation", DataType::UInt32, false),
        Field::new("chain", DataType::UInt32, false),
    ];
    for dim_idx in 0..num_dims {
        fields.push(Field::new(
            format!("dim_{}", dim_idx),
            DataType::Float64,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Create builders for each column.
    let mut observation_builder = UInt32Builder::new();
    let mut chain_builder = UInt32Builder::new();
    let mut dim_builders: Vec<Float64Builder> =
        (0..num_dims).map(|_| Float64Builder::new()).collect();

    // Populate the builders.
    for observation in 0..num_observations {
        for chain in 0..num_chains {
            observation_builder.append_value(observation as u32);
            chain_builder.append_value(chain as u32);
            let offset = observation * num_chains * num_dims + chain * num_dims;
            for (dim_idx, value) in flat[offset..offset + num_dims].iter().enumerate() {
                dim_builders[dim_idx].append_value((*value).into());
            }
        }
    }

    let observation_array = Arc::new(observation_builder.finish()) as ArrayRef;
    let chain_array = Arc::new(chain_builder.finish()) as ArrayRef;
    let mut arrays = vec![observation_array, chain_array];
    for mut builder in dim_builders {
        arrays.push(Arc::new(builder.finish()) as ArrayRef);
    }

    let record_batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // Write the record batch to a Parquet file.
    let file = File::create(filename)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&record_batch)?;
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt32Array};
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use ndarray::arr3;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
    use std::fs;
    use std::{error::Error, fs::File};
    use tempfile::NamedTempFile;

    /// Test saving empty data to Parquet (zero chains).
    #[test]
    fn test_save_parquet_empty_data() -> Result<(), Box<dyn Error>> {
        let data = arr3::<f32, 0, 0>(&[]); // no chains
        // let file = NamedTempFile::new()?;
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_parquet(&data, filename);
        assert!(
            result.is_ok(),
            "Saving empty data to Parquet failed: {:?}",
            result
        );

        let metadata = fs::metadata(filename)?;
        // Even though there's no chain data, the file shouldn't be completely empty
        assert!(metadata.len() > 0, "Parquet file is unexpectedly empty");

        // Check emptyness
        let file = File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        assert!(reader.next().is_none());
        Ok(())
    }

    /// Test saving a single chain, single observation (one dimension).
    #[test]
    fn test_save_parquet_single_chain_single_observation() -> Result<(), Box<dyn Error>> {
        let data = arr3(&[[[42f64]]]); // chain=0, observation=0, dim_0=42
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        save_parquet(&data, filename)?;

        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0, "Parquet file is unexpectedly empty");

        // Read back the batch
        let file = File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        let batch = reader.next().expect("Expected a record batch")?.clone();
        assert!(reader.next().is_none(), "Expected only one batch");

        // Should have 1 row and 3 columns (chain, observation, dim_0)
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 3);

        // Check values
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

    /// Test multiple chains, multiple observations, multiple dimensions to Parquet.
    #[test]
    fn test_save_parquet_multi_chain() -> Result<(), Box<dyn Error>> {
        // data[chain][observation][dim]
        // chain=0 => observation=0 => [1.0, 2.0], observation=1 => [3.0, 4.0]
        // chain=1 => observation=0 => [10.0, 20.0], observation=1 => [30.0, 40.0]
        let data = arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]);

        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        save_parquet(&data, filename)?;

        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0);

        // Read back
        let file = File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        let batch = reader.next().expect("No record batch found")?;
        assert!(reader.next().is_none(), "Expected only one batch");

        // We expect 4 rows total: 2 chains × 2 observations each
        // columns = chain, observation, dim_0, dim_1 => 4 columns
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

        // Row 0: chain=0, observation=0, (dim_0=1.0, dim_1=2.0)
        assert_eq!(chain_array.value(0), 0);
        assert_eq!(observation_array.value(0), 0);
        assert!((dim0_array.value(0) - 1.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(0) - 2.0).abs() < f64::EPSILON);

        // Row 1: chain=0, observation=1, (dim_0=3.0, dim_1=4.0)
        assert_eq!(chain_array.value(1), 0);
        assert_eq!(observation_array.value(1), 1);
        assert!((dim0_array.value(1) - 3.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(1) - 4.0).abs() < f64::EPSILON);

        // Row 2: chain=1, observation=0, (dim_0=10.0, dim_1=20.0)
        assert_eq!(chain_array.value(2), 1);
        assert_eq!(observation_array.value(2), 0);
        assert!((dim0_array.value(2) - 10.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(2) - 20.0).abs() < f64::EPSILON);

        // Row 3: chain=1, observation=1, (dim_0=30.0, dim_1=40.0)
        assert_eq!(chain_array.value(3), 1);
        assert_eq!(observation_array.value(3), 1);
        assert!((dim0_array.value(3) - 30.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(3) - 40.0).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_save_parquet_tensor_data() -> Result<(), Box<dyn std::error::Error>> {
        // Create a tensor with shape [2, 2, 2]: 2 observations, 2 chains, 2 dimensions.
        let tensor = Tensor::<NdArray, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]],
            &NdArrayDevice::Cpu,
        );
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();
        save_parquet_tensor::<NdArray, _, f32>(&tensor, filename)?;

        // Open the Parquet file using ParquetRecordBatchReader.
        let file = fs::File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        let batch = reader.next().expect("Expected a record batch")?;
        // We expect 2 observations * 2 chains = 4 rows and 2 (observation, chain) + 2 dims = 4 columns.
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 4);

        // Extract and check the data.
        let observation_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("Failed to downcast observation column");
        let chain_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("Failed to downcast chain column");
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Failed to downcast dim_0 column");
        let dim1_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Failed to downcast dim_1 column");

        // Expected ordering: observation-major order:
        // Row 0: observation=0, chain=0, dims: [1.0, 2.0]
        assert_eq!(observation_array.value(0), 0);
        assert_eq!(chain_array.value(0), 0);
        assert!((dim0_array.value(0) - 1.0).abs() < 1e-6);
        assert!((dim1_array.value(0) - 2.0).abs() < 1e-6);
        // Row 1: observation=0, chain=1, dims: [3.0, 4.0]
        assert_eq!(observation_array.value(1), 0);
        assert_eq!(chain_array.value(1), 1);
        assert!((dim0_array.value(1) - 3.0).abs() < 1e-6);
        assert!((dim1_array.value(1) - 4.0).abs() < 1e-6);
        // Row 2: observation=1, chain=0, dims: [1.1, 2.1]
        assert_eq!(observation_array.value(2), 1);
        assert_eq!(chain_array.value(2), 0);
        assert!((dim0_array.value(2) - 1.1).abs() < 1e-6);
        assert!((dim1_array.value(2) - 2.1).abs() < 1e-6);
        // Row 3: observation=1, chain=1, dims: [3.1, 4.1]
        assert_eq!(observation_array.value(3), 1);
        assert_eq!(chain_array.value(3), 1);
        assert!((dim0_array.value(3) - 3.1).abs() < 1e-6);
        assert!((dim1_array.value(3) - 4.1).abs() < 1e-6);

        Ok(())
    }
}
