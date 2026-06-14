use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};

use rayon::prelude::*;

use std::collections::HashMap;

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use std::sync::{Arc, Condvar, Mutex};


use crate::faer_ndarray::FaerEigh;

use crate::linalg::matrix::{
    DesignMatrix, LinearOperator, SignedWeightsView, upper_triangle_pair_from_index,
};
