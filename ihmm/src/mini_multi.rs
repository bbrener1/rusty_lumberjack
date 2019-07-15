use std::f64::consts::E;
use std::f64::consts::PI;
use std::f64::EPSILON;

use ndarray as nd;
use ndarray_linalg as ndl;
// use num_traits;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::error::LapackError;
use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::svd::SVD;
// use ndarray_linalg::solveh::{InverseH,DeterminantH};

#[derive(Debug,Clone)]
pub struct MVN {
    means: Array<f64,Ix1>,
    samples: usize,
}

impl MVN {

    pub fn identity_prior(samples:usize,features:usize) -> MVN {
        MVN {
            means: Array::zeros(features as usize),
            samples: samples,
        }
    }

    pub fn set_samples(&mut self,samples: usize) {
        self.samples = samples;
    }

    pub fn scaled_identity_prior(means:&ArrayView<f64,Ix1>,samples:usize) -> MVN {

        MVN {
            means: means.to_owned(),
            samples,
        }

    }

    pub fn estimate_against_identity(data:&ArrayView<f64,Ix2>,strength_option:Option<u32>) -> Result<MVN,LinalgError> {

        let (mut samples,features) = data.dim();

        let means = data.mean_axis(Axis(0));

        let mut prior = MVN::scaled_identity_prior(&means.view(), samples);

        Ok(prior)
    }

    pub fn estimate(&mut self,data:&ArrayView<f64,Ix2>) -> Result<&mut MVN,LinalgError> {

        let (samples,features) = data.dim();
        let (centered,sample_means) = center(data);

        self.means = sample_means;
        self.samples = self.samples + samples;

        // eprintln!("EFM:{:?}",self.means);

        Ok(self)

    }

    pub fn mini_estimate(&mut self,data:&ArrayView<f64,Ix2>) -> Result<&mut MVN,LinalgError> {

        self.estimate(data)

    }

    pub fn log_likelihood(&self,data:&ArrayView<f64,Ix1>) -> f64 {

        let centered_data = (data - &self.means);

        let log_likelihood = -0.5 * (centered_data.iter().map(|v| v.powi(2)).sum::<f64>().log2() + 1. + self.means.dim() as f64);

        log_likelihood

    }


    pub fn dim(&self) -> (usize,usize) {
        (self.samples as usize,self.means.dim())
    }

    pub fn means(&self) -> ArrayView<f64,Ix1> {
        self.means.view()
    }

    pub fn pdet(&self) -> &f64 {
        &1.
    }

}


pub fn outer_product(a:&ArrayView<f64,Ix1>,b:&ArrayView<f64,Ix1>) -> Array<f64,Ix2> {
    let ad = a.dim();
    let bd = b.dim();

    let mut out = Array::zeros((ad,bd));

    for (mut out_r,ai) in out.axis_iter_mut(Axis(0)).zip(a.iter()) {
        out_r.assign(&(*ai * b));
    }

    out
}


pub fn center(data:&ArrayView<f64,Ix2>) -> (Array<f64,Ix2>,Array<f64,Ix1>) {
    let means = data.mean_axis(Axis(0));
    let mut centered = data.to_owned();
    for mut row in centered.axis_iter_mut(Axis(0)) {
        row -= &means;
    }
    return (centered,means);
}
