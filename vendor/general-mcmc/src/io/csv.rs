/*!
# I/O Utilities for Saving MCMC Data to CSV

This module provides functions to save MCMC sample data to CSV files. Enable via the `csv` feature.
*/

use burn::prelude::*;
use ndarray::{Array3, Axis};
use std::error::Error;
use std::fs::File;

use csv::Writer;

/**
Saves MCMC sample as a CSV file.

The data is expected to be in a shape of **chain × observation × dimension**.

The resulting CSV file will have:
- A header row containing `"chain"`, `"observation"`, and one column per dimension
  named `"dim_0"`, `"dim_1"`, etc.
- Each subsequent row will correspond to a single observation of a specific chain.

# Arguments

* `data` - An Array3<T> object containing the MCMC data.
* * `filename` - The file path where the CSV data will be written.

# Returns

Returns `Ok(())` if successful, or an error if any I/O or CSV formatting
issue occurs.

# Examples

```rust
use general_mcmc::io::csv::save_csv;
use ndarray::arr3;

// This matrix has 2 rows (observations) and 4 columns (dimensions).
let data = arr3(&[[[1, 2, 3, 4], [5, 6, 7, 8]]]);

save_csv(&data, "/tmp/output.csv").expect("Expecting saving data to succeed");
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_csv<T: std::fmt::Display>(
    data: &Array3<T>,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_writer(File::create(filename)?);
    let n_dims = data.shape()[2];

    let mut header: Vec<String> = vec!["chain".to_string(), "observation".to_string()];
    header.extend((0..n_dims).map(|i| format!("dim_{}", i)));
    wtr.write_record(&header)?;

    // Flatten and write data
    for (chain_idx, chain) in data.axis_iter(Axis(0)).enumerate() {
        for (obs_idx, obs) in chain.axis_iter(Axis(0)).enumerate() {
            let mut row = vec![chain_idx.to_string(), obs_idx.to_string()];
            row.extend(obs.iter().map(|v| v.to_string()));
            wtr.write_record(&row)?;
        }
    }

    wtr.flush()?;
    Ok(())
}

/**
Saves a 3D Burn tensor (chain x observation × dimension) as a CSV file.

The CSV file will contain a header row with columns:
  - `"chain"`: the chain index,
  - `"observation"`: the observation index,
  - `"dim_0"`, `"dim_1"`, … for each dimension.

Each subsequent row corresponds to one data point from the tensor, with its chain and observation indices
followed by the dimension values. For example, the coordinate `d` of data point `n` that chain `c`
generated is assumed to be in `tensor[c][n][d]`.

# Arguments
* `tensor` - A Burn tensor with shape `[num_chains, num_observations, num_dimensions]`.
* `filename` - The file path where the CSV data will be written.

# Type Parameters
* `B` - The backend type.
# Returns
Returns `Ok(())` if successful or an error if any I/O or CSV formatting issue occurs.

# Example
```rust
use burn::tensor::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use general_mcmc::io::csv::save_csv_tensor;
let tensor = Tensor::<NdArray, 3>::from_floats(
    [
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        [[1.01, 2.01], [1.11, 2.11], [1.21, 2.21]],
        [[1.02, 2.02], [1.12, 2.12], [1.22, 2.22]],
        [[1.03, 2.03], [1.13, 2.13], [1.23, 2.23]],
    ],
    &NdArrayDevice::Cpu,
);
save_csv_tensor(tensor, "/tmp/output.csv")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_csv_tensor<B>(
    tensor: burn::tensor::Tensor<B, 3>,
    filename: &str,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
{
    use csv::Writer;
    use std::fs::File;
    // Extract data as TensorData and convert to a flat Vec<T>
    let shape = tensor.dims(); // expected to be [num_chains, num_obs, num_dimensions]
    let data = tensor.to_data();
    let (num_chains, num_obs, num_dims) = (shape[0], shape[1], shape[2]);
    let flat: Vec<f32> = data
        .to_vec()
        .map_err(|e| format!("Converting data to Vec failed.\nData: {data:?}.\nError: {e:?}"))?;

    let mut wtr = Writer::from_writer(File::create(filename)?);

    // Build header: "obs", "chain", "dim_0", "dim_1", ...
    let mut header = vec!["chain".to_string(), "observation".to_string()];
    header.extend((0..num_dims).map(|i| format!("dim_{}", i)));
    wtr.write_record(&header)?;

    // Iterate over observation and chain indices; each row corresponds to one data point.
    for chain_idx in 0..num_chains {
        for obs_idx in 0..num_obs {
            let mut row = vec![chain_idx.to_string(), obs_idx.to_string()];
            let offset = chain_idx * num_obs * num_dims + obs_idx * num_dims;
            let row_slice = &flat[offset..offset + num_dims];
            row.extend(row_slice.iter().map(|v| v.to_string()));
            wtr.write_record(&row)?;
        }
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    use csv::Reader;
    use ndarray::arr3;
    use std::fs;
    use tempfile::NamedTempFile;

    // --- CSV Tests ---

    /// Test saving empty data to CSV (zero chains).
    #[test]
    fn test_save_csv_empty_data() {
        let data = arr3::<f32, 0, 0>(&[]);
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_csv(&data, filename);
        assert!(
            result.is_ok(),
            "Saving empty data to CSV failed: {:?}",
            result
        );

        // Verify that the CSV file is created and only has a header row (or is empty).
        let contents = fs::read_to_string(filename).unwrap();
        // The function writes a header even if there's no data.
        // The header should be "chain,observation" only, because num_dimensions=0
        assert_eq!(contents.trim(), "chain,observation");
    }

    /// Test saving a single chain with a single observation (and single dimension) to CSV.
    #[test]
    fn test_save_csv_single_chain_single_obs() {
        let data = arr3(&[[[42.0]]]); // chain=0, obs=0, dim_0=42
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_csv(&data, filename);
        assert!(
            result.is_ok(),
            "Saving single chain with single obs to CSV failed: {:?}",
            result
        );

        let contents = fs::read_to_string(filename).unwrap();
        let expected = "chain,observation,dim_0\n0,0,42";
        assert_eq!(contents.trim(), expected);
    }

    /// Test multiple chains, multiple observations, multiple dimensions to CSV.
    #[test]
    fn test_save_csv_multi_chain() {
        // data[chain][observation][dim]
        let data = arr3(&[[[1, 2], [3, 4]], [[10, 20], [30, 40]]]);
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_csv(&data, filename);
        assert!(result.is_ok());

        let contents = fs::read_to_string(filename).unwrap();
        let expected = "\
chain,observation,dim_0,dim_1
0,0,1,2
0,1,3,4
1,0,10,20
1,1,30,40";
        assert_eq!(contents.trim(), expected);
    }

    #[test]
    fn test_save_csv_tensor_data() -> Result<(), Box<dyn std::error::Error>> {
        // Create a tensor with shape [2, 2, 2]: 2 chains, 2 observations, 2 dimensions.
        let tensor = Tensor::<NdArray, 3, burn::tensor::Float>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]],
            &NdArrayDevice::Cpu,
        );
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();
        save_csv_tensor(tensor, filename)?;
        let contents = fs::read_to_string(filename)?;

        // Use csv::Reader to parse the CSV file.
        let mut rdr = Reader::from_reader(contents.as_bytes());
        let headers = rdr.headers()?;
        assert_eq!(&headers[0], "chain");
        assert_eq!(&headers[1], "observation");
        assert_eq!(&headers[2], "dim_0");
        assert_eq!(&headers[3], "dim_1");

        let records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
        // There should be 2 observations * 2 chains = 4 records.
        assert_eq!(records.len(), 4);

        // Expected ordering: For each observation, for each chain.
        // Row 0: obs 0, chain 0, dims: [1.0, 2.0]
        // Row 1: obs 0, chain 1, dims: [3.0, 4.0]
        // Row 2: obs 1, chain 0, dims: [1.1, 2.1]
        // Row 3: obs 1, chain 1, dims: [3.1, 4.1]
        let expected = [
            vec!["0", "0", "1", "2"],
            vec!["0", "1", "3", "4"],
            vec!["1", "0", "1.1", "2.1"],
            vec!["1", "1", "3.1", "4.1"],
        ];
        for (record, exp) in records.iter().zip(expected.iter()) {
            for (field, &exp_field) in record.iter().zip(exp.iter()) {
                // Allow small differences in formatting for floating-point numbers.
                assert!(
                    field.contains(exp_field),
                    "Expected field '{}' to contain '{}'",
                    field,
                    exp_field
                );
            }
        }
        Ok(())
    }
}
