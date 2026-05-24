use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone)]
pub struct GatedSAEDecoder {
    pub w_gate: Array2<f64>,
    pub w_amp: Array2<f64>,
}

impl GatedSAEDecoder {
    #[must_use = "build error must be handled"]
    pub fn new(w_gate: Array2<f64>, w_amp: Array2<f64>) -> Result<Self, String> {
        if w_gate.nrows() != w_gate.ncols() {
            return Err(format!(
                "GatedSAEDecoder::new requires square W_gate; got {:?}",
                w_gate.dim()
            ));
        }
        if w_amp.ncols() != w_gate.ncols() {
            return Err(format!(
                "GatedSAEDecoder::new requires W_amp columns {} to match W_gate input {}",
                w_amp.ncols(),
                w_gate.ncols()
            ));
        }
        if !w_gate.iter().all(|v| v.is_finite()) || !w_amp.iter().all(|v| v.is_finite()) {
            return Err("GatedSAEDecoder::new requires finite weights".to_string());
        }
        Ok(Self { w_gate, w_amp })
    }

    pub fn input_dim(&self) -> usize {
        self.w_gate.ncols()
    }

    pub fn output_dim(&self) -> usize {
        self.w_amp.nrows()
    }

    pub fn decode_row(&self, x: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        if x.len() != self.input_dim() {
            return Err(format!(
                "GatedSAEDecoder::decode_row expected x len {}, got {}",
                self.input_dim(),
                x.len()
            ));
        }
        let mut gated = Array1::<f64>::zeros(x.len());
        for gate_row in 0..self.w_gate.nrows() {
            let mut logit = 0.0;
            for col in 0..self.w_gate.ncols() {
                logit += self.w_gate[[gate_row, col]] * x[col];
            }
            if logit > 0.0 {
                gated[gate_row] = x[gate_row];
            }
        }
        Ok(self.w_amp.dot(&gated))
    }

    pub fn decode_batch(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if x.ncols() != self.input_dim() {
            return Err(format!(
                "GatedSAEDecoder::decode_batch expected {} columns, got {}",
                self.input_dim(),
                x.ncols()
            ));
        }
        let mut out = Array2::<f64>::zeros((x.nrows(), self.output_dim()));
        for row in 0..x.nrows() {
            let decoded = self.decode_row(x.row(row))?;
            out.row_mut(row).assign(&decoded);
        }
        Ok(out)
    }
}
