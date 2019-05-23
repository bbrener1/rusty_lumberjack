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
    covariance: Array<f64,Ix2>,
    pseudo_precision: Array<f64,Ix2>,
    precision: Array<f64,Ix2>,
    cdet: f64,
    pdet: f64,
    samples: u32,
}

impl MVN {

    pub fn identity_prior(samples:u32,features:u32) -> MVN {
        MVN {
            means: Array::zeros(features as usize),
            covariance: Array::eye(features as usize),
            pseudo_precision: Array::eye(features as usize),
            precision: Array::eye(features as usize),
            pdet: 0.,
            cdet: 0.,
            samples: samples,
        }
    }

    pub fn scaled_identity_prior(means:&ArrayView<f64,Ix1>,variances:&ArrayView<f64,Ix1>,samples:u32) -> MVN {

        let f = variances.dim();
        let mut covariance = Array::eye(f);
        covariance.diag_mut().assign(variances);
        let precision = covariance.inv().unwrap();
        let (pseudo_precision,pdet) = pinv(&covariance.view()).unwrap();
        let cdet = covariance.sln_det().unwrap().1 * f64::log2(E);


        MVN {
            means: means.to_owned(),
            covariance,
            precision,
            pseudo_precision,
            pdet,
            cdet,
            samples,
        }
    }

    pub fn estimate_against_identity(data:&ArrayView<f64,Ix2>,mask: &ArrayView<bool,Ix2>,strength_option:Option<u32>) -> Result<MVN,LinalgError> {
        let (samples,features) = data.dim();
        let prior_strength = strength_option.unwrap_or(features as u32);
        // let rank_deficit = features as i32 - samples as i32;
        // let prior_strength = std::cmp::max(rank_deficit,1) as u32;
        // let prior_strength = samples as u32;
        // let mut prior = MVN::identity_prior(prior_strength as u32, features as u32);
        let (means,variances) = masked_array_properties(data, mask);

        eprintln!("Estimating against scaled identity");
        eprintln!("mean:{:?},var:{:?}",means,variances);

        let mut prior = MVN::scaled_identity_prior(&means.view(), &variances.view(), prior_strength);

        // eprintln!("{:?}",prior);

        prior.estimate_masked(data, mask)?;

        // eprintln!("Estimated:{:?}",prior);

        Ok(prior)
    }

    pub fn estimate_masked(&mut self,data:&ArrayView<f64,Ix2>,mask:&ArrayView<bool,Ix2>) -> Result<&mut MVN,LinalgError> {

        let (samples,features) = data.dim();

        let feature_sums = data.sum_axis(Axis(0));
        let float_mask = mask.map(|b| if *b {1.} else {0.});

        let mut posterior_covariance = Array::eye(features);
        let mut posterior_precision = Array::eye(features);
        let mut posterior_pseudo_precision = Array::eye(features);
        let mut posterior_pseudo_determinant = 1.;

        let feature_populations = float_mask.sum_axis(Axis(0)) + 1.0;
        let feature_means = feature_sums/feature_populations;

        // eprintln!("Estimated means: {:?}", feature_means);
        // eprintln!("Samples in estimate:{:?}",samples);

        let posterior_means = ((&self.means * self.samples as f64) + (&feature_means * samples as f64)) / (self.samples as usize + samples) as f64;

        // eprintln!("Posterior means: {:?}", posterior_means);

        let mut centered_data = data.to_owned();

        for i in 0..samples {
            let fm = feature_means.view();
            let m = mask.row(i);
            let mut c = centered_data.row_mut(i);
            azip!(mut c,fm,m in {*c = if m {*c-fm} else {0.} });
        }

        // eprintln!("Centered data:{:?}",centered_data);

        let mut outer_feature_sum: Array<f64,Ix2> = Array::zeros((features,features));

        for (i,sample) in centered_data.axis_iter(Axis(0)).enumerate() {
            outer_feature_sum += &outer_product(&sample, &sample);
        }

        let mut scale_factor: Array<f64,Ix2> = Array::ones((features,features));

        for (i,sample) in float_mask.axis_iter(Axis(0)).enumerate() {
            scale_factor += &outer_product(&sample, &sample);
        }

        scale_factor /= samples as f64;

        let s = outer_feature_sum / scale_factor;

        let mut covariance_estimate = &s / samples as f64;

        // eprintln!("Covariance estimate: {:?}", covariance_estimate);

        {

            let lo = &self.covariance * self.samples as f64;

            let prior_mean_delta = &feature_means - &self.means;
            let mean_delta_outer = outer_product(&prior_mean_delta.view(),&prior_mean_delta.view());

            let mean_delta_scale = ((self.samples as usize * samples) / (self.samples as usize + samples)) as f64;

            // let mut ln = lo + &s + &(mean_delta_scale * mean_delta_outer);
            let mut ln = &lo + &s + &(mean_delta_scale * mean_delta_outer);

            let inverse_wishart_scale = (self.samples as usize + samples - features - 1) as f64;

            // eprintln!("Inverse wishart scale: {:?}",inverse_wishart_scale);

            posterior_covariance.assign(&mut (ln / inverse_wishart_scale));
            posterior_precision.assign(&mut posterior_covariance.inv()?);

            let (mut p_i,mut p_d) = pinv(&posterior_covariance.view())?;

            posterior_pseudo_determinant = p_d;
            posterior_pseudo_precision.assign(&mut p_i);

        }


        let (sign,ldet):(f64,f64) = posterior_covariance.sln_det()?;

        if !(sign > 0.) {
            eprintln!("{:?}", sign);
            eprintln!("{:?}",ldet);
            eprintln!("{:?}",posterior_covariance);
            eprintln!("WARNING MATRIX MAY NOT BE POSITIVE DEFINITE");
        }

        let cdet: f64 = ldet * f64::log2(E);

        self.means = posterior_means;
        self.covariance = posterior_covariance;
        self.precision = posterior_precision;
        self.pseudo_precision = posterior_pseudo_precision;
        self.cdet = cdet;
        self.samples = self.samples + samples as u32;

        Ok(self)

    }

    pub fn log_likelihood(&self,data:&ArrayView<f64,Ix1>) -> f64 {

        let centered_data = data - &self.means;

        // -0.5 * (self.cdet + centered_data.dot(&self.precision).dot(&centered_data) + (self.dim().1 as f64 * f64::log2(2.*PI)))

        -0.5 * (self.pdet + centered_data.dot(&self.pseudo_precision).dot(&centered_data) + (self.dim().1 as f64 * f64::log2(2.*PI)))

    }

    pub fn masked_likelihood(&self,data:&ArrayView<f64,Ix1>,mask:&ArrayView<bool,Ix1>) -> f64 {
        // eprintln!("Deriving masked model");
        let masked_normal = self.derive_masked_MVN(mask);
        // eprintln!("{:?}",masked_normal);
        // eprintln!("{:?}",(data,mask));
        let masked_data = array_mask(data, mask);
        // eprintln!("{:?}",masked_data.to_vec());
        // eprintln!("Masked normal debug!");
        // eprintln!("{:?}",masked_normal.means.to_vec());
        // eprintln!("{:?}",masked_normal.covariance);
        // eprintln!("{:?}",masked_normal.precision);
        // eprintln!("Evaluating masked likelihood");
        let likelihood = masked_normal.log_likelihood(&masked_data.view());
        // eprintln!("Likelihood: {:?}", likelihood);
        likelihood
    }

    pub fn derive_masked_MVN(&self,mask:&ArrayView<bool,Ix1>) -> MVN {

        let means = array_mask(&self.means.view(), mask);
        let covariance = array_double_mask(&self.covariance.view(), mask);
        let precision = covariance.inv().unwrap();
        let (pseudo_precision,pdet) = pinv(&covariance.view()).unwrap();
        let cdet = covariance.sln_det().unwrap().1;

        let samples = means.dim() as u32;

        MVN{
            means,
            covariance,
            precision,
            pseudo_precision,
            pdet,
            cdet,
            samples,
        }
    }

    pub fn dim(&self) -> (usize,usize) {
        (self.samples as usize,self.means.dim())
    }

    pub fn means(&self) -> ArrayView<f64,Ix1> {
        self.means.view()
    }

    pub fn variances(&self) -> ArrayView<f64,Ix1> {
        self.covariance.diag()
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

pub fn array_mask<T: Copy>(data:&ArrayView<T,Ix1>,mask:&ArrayView<bool,Ix1>) -> Array<T,Ix1> {
    let index_mask: Vec<usize> = mask.iter().enumerate().filter_map(|(i,b)| if *b {Some(i)} else {None}).collect();
    data.select(Axis(0),&index_mask[..])
}

pub fn array_mask_axis<T: Copy,U: nd::RemoveAxis>(data:&ArrayView<T,U>,mask:&ArrayView<bool,Ix1>,axis:Axis) -> Array<T,U> {
    let index_mask: Vec<usize> = mask.iter().enumerate().filter_map(|(i,b)| if *b {Some(i)} else {None}).collect();
    data.select(axis,&index_mask[..])
}

pub fn array_double_mask<T: Copy>(data:&ArrayView<T,Ix2>,mask:&ArrayView<bool,Ix1>) -> Array<T,Ix2> {
    let singe_masked = array_mask_axis(data, mask, Axis(0));
    let double_masked = array_mask_axis(&singe_masked.view(), mask, Axis(1));
    double_masked
}

pub fn array_double_select<T:Copy>(data:&ArrayView<T,Ix2>,mask:&[usize]) -> Array<T,Ix2> {
    let single_selected = data.select(Axis(0),mask);
    let double_selected = single_selected.select(Axis(0),mask);
    double_selected
}

pub fn masked_array_properties(data:&ArrayView<f64,Ix2>,mask: &ArrayView<bool,Ix2>) -> (Array<f64,Ix1>,Array<f64,Ix1>) {

    let feature_sums = data.sum_axis(Axis(0));
    let float_mask = mask.map(|b| if *b {1.} else {0.});

    let feature_populations = float_mask.sum_axis(Axis(0)) + 1.0;
    let feature_means = &feature_sums/&feature_populations;

    let feature_square_sums = data.mapv(|x| x.powi(2)).sum_axis(Axis(0));

    let variances = (&feature_square_sums / &feature_populations) - &feature_means.mapv(|x| x.powi(2));

    (feature_means,variances)

}

pub fn pinv(mtx:&ArrayView<f64,Ix2>) -> Result<(Array<f64,Ix2>,f64),LinalgError> {
    let (r,c) = mtx.dim();
    // eprintln!("Inverting:");
    // eprintln!("{:?}",mtx);
    if let (Some(u),sig,Some(vt)) = mtx.svd(true,true)? {
        // eprintln!("{:?},{:?},{:?}",u,sig,vt);
        let lower_bound = (EPSILON * 10.);
        let i_sig = sig.mapv(|v| if v > lower_bound {1./v} else {0.} );
        let pdet = sig.iter().fold(0.,|acc,sv| if *sv > lower_bound {acc + sv.log2()} else {acc});
        let mut t_sig = Array::zeros((c,r));
        t_sig.diag_mut().assign(&i_sig);
        let p_i = vt.t().dot(&t_sig).dot(&u.t());
        Ok((p_i,pdet))
    }
    else {Err(LinalgError::Lapack{return_code:0})}

}

// pub fn array_mask_row<T: Copy + num_traits::Zero>(data:&ArrayView<T,Ix2>,mask:&ArrayView<bool,Ix1>) -> Array<T,Ix2> {
//     let d: usize = mask.mapv(|b| if b {1_u32} else {0_u32}).into_iter().sum::<u32>() as usize;
//     let (r,c) = data.dim();
//     let mut masked: Array<T,Ix2> = Array::zeros((d,c));
//     for (i,r) in mask.iter().zip(data.axis_iter(Axis(0))).filter(|(b,r)| **b).map(|(b,r)| r).enumerate() {
//         masked.row_mut(i).assign(&data.row(i));
//     }
//     masked
// }
//
// pub fn array_mask_col<T: Copy + num_traits::Zero>(data:&ArrayView<T,Ix2>,mask:&ArrayView<bool,Ix1>) -> Array<T,Ix2> {
//     let d: usize = mask.mapv(|b| if b {1_u32} else {0_u32}).into_iter().sum::<u32>() as usize;
//     let (r,c) = data.dim();
//     let mut masked: Array<T,Ix2> = Array::zeros((r,d));
//     for (i,c) in mask.iter().zip(data.axis_iter(Axis(1))).filter(|(b,c)| **b).map(|(b,c)| c).enumerate() {
//         masked.column_mut(i).assign(&data.column(i));
//     }
//     masked
// }

// pub fn array_double_mask<T: Copy + num_traits::Zero>(data:&ArrayView<T,Ix2>,mask:&ArrayView<bool,Ix1>) -> Array<T,Ix2> {
//     let d: usize = mask.mapv(|b| if b {1_u32} else {0_u32}).into_iter().sum::<u32>() as usize;
//     let mut masked: Array<T,Ix2> = Array::zeros((d,d));
//     for (i,r) in mask.iter().zip(data.axis_iter(Axis(0))).filter(|(b,r)| **b).map(|(b,r)| r).enumerate() {
//         for (j,v) in mask.iter().zip(r.iter()).filter(|(b,v)| **b).map(|(b,v)| v).enumerate() {
//             masked[[i,j]] = *v;
//         }
//     }
//     masked
// }


#[cfg(test)]
mod tree_braider_tests {

    extern crate intel_mkl_src;
    // extern crate openblas_src;
    // extern crate netlib_src;

    use super::*;
    use crate::MarkovNode;
    use crate::tree_braider_tests::iris_forest;

    #[test]
    fn test_mvn_array_mask() {
        let a = array![0,1,2,3,4];
        let m = array![true,false,true,false,true];
        assert_eq!(array_mask(&a.view(), &m.view()),array![0,2,4]);
    }

    #[test]
    fn test_mvn_array_double_mask() {
        let a = array![[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]];
        let m = array![true,false,true,false,true];
        assert_eq!(array_double_mask(&a.view(), &m.view()),array![[0,2,4],[0,2,4],[0,2,4]]);
    }

    #[test]
    fn test_mvn_masked_properties() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        eprintln!("{:?}",masked_array_properties(&data.view(), &mask.view()));
    }

    #[test]
    fn test_mvn_normal_blank() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        let normal = MVN::identity_prior(150,4);
    }

    #[test]
    fn test_mvn_normal_iris() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        let mut normal = MVN::identity_prior(100,4);
        normal.estimate_masked(&data.view(), &mask.view());
        eprintln!("{:?}",normal.means);
        eprintln!("{:?}",normal.covariance);
        eprintln!("{:?}",normal.precision);
        eprintln!("{:?}",normal.cdet);
        // panic!();
    }

    #[test]
    fn test_mvn_normal_id_prior_iris() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        let normal = MVN::estimate_against_identity(&data.view(), &mask.view(), None).unwrap();
        eprintln!("{:?}",normal.means);
        eprintln!("{:?}",normal.covariance);
        eprintln!("{:?}",normal.precision);
        eprintln!("{:?}",normal.cdet);
        // panic!();
    }

    #[test]
    fn test_mvn_mask() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        assert_eq!(array_mask(&data.row(0),&mask.row(0)).dim(),2);
    }

    #[test]
    fn test_mvn_normal_likelihood_iris() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        let normal = MVN::estimate_against_identity(&data.view(), &mask.view(), None).unwrap();
        eprintln!("{:?}",data.axis_iter(Axis(0)).zip(mask.axis_iter(Axis(0))).map(|(d,m)| normal.masked_likelihood(&d, &m)).collect::<Vec<f64>>());
        eprintln!("{:?}",normal.means);
        eprintln!("{:?}",normal.cdet);
        eprintln!("{:?}",normal.covariance);
    }

    #[test]
    fn test_pinv() {
        let a = array![[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]];
        eprintln!("{:?}",pinv(&a.view()));
        let b = array![[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
        eprintln!("{:?}",pinv(&b.view()));
        let c = array![[1.,0.],[0.,1.]];
        eprintln!("{:?}",pinv(&c.view()));
        // panic!();
    }

}
