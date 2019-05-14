use std::f64::consts::E;
use std::f64::consts::PI;

use ndarray as nd;
use ndarray_linalg as ndl;
use num_traits;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::solveh::{InverseH,DeterminantH};


pub struct Dirichlet {
    categories: Vec<u32>,
    a: u32,
    samples: u32,
}

impl Dirichlet {

    pub fn blank(categories:usize,dispersion:u32) -> Dirichlet {
        Dirichlet {
            categories: vec![dispersion;categories],
            a: dispersion,
            samples: 0
        }
    }

    pub fn log_likelihoods(&self) -> Vec<f64> {
        unimplemented!();
    }

}


#[cfg(test)]
mod dirichlet_tests {

    use super::*;
    use crate::MarkovNode;
    use crate::tree_braider_tests::iris_forest;

}
