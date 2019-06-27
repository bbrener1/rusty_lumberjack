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
    variances: Array<f64,Ix1>,
    covariance: Array<f64,Ix2>,
    pseudo_precision: Array<f64,Ix2>,
    pdet: f64,
    rank: f64,
    samples: u32,
}

impl MVN {

    pub fn identity_prior(samples:u32,features:u32) -> MVN {
        MVN {
            means: Array::zeros(features as usize),
            variances: Array::ones(features as usize),
            covariance: Array::eye(features as usize),
            pseudo_precision: Array::eye(features as usize),
            pdet: 0.,
            rank: features as f64,
            samples: samples,
        }
    }

    pub fn set_samples(&mut self,samples: usize) {
        self.samples = samples as u32;
    }

    pub fn scaled_identity_prior(means:&ArrayView<f64,Ix1>,variances:&ArrayView<f64,Ix1>,samples:u32) -> MVN {

        let f = variances.dim();
        let mut covariance = Array::eye(f);
        let (pseudo_precision,pdet,rank) = pinv_pdet(&covariance.view()).unwrap();


        MVN {
            means: means.to_owned(),
            variances: variances.to_owned(),
            covariance,
            pseudo_precision,
            pdet,
            rank,
            samples,
        }
    }

    pub fn estimate_against_identity(data:&ArrayView<f64,Ix2>,strength_option:Option<u32>) -> Result<MVN,LinalgError> {

        let (mut samples,features) = data.dim();

        let means = data.mean_axis(Axis(0));
        let variances = data.var_axis(Axis(0),0.);

        if let Some(strength) = strength_option {
            samples = strength as usize;
        }

        let mut prior = MVN::scaled_identity_prior(&means.view(), &variances.view(), samples as u32);

        // eprintln!("{:?}",prior);

        prior.estimate(data)?;

        // eprintln!("Estimated:{:?}",prior);

        Ok(prior)
    }

    pub fn uninformed_estimate(&mut self,data:&ArrayView<f64,Ix2>) -> Result<&mut MVN,LinalgError> {

        let (samples,features) = data.dim();

        let (scaled,feature_means,variances) = scale(data);

        let mut outer_feature_sum: Array<f64,Ix2> = Array::zeros((features,features));

        for (i,sample) in scaled.axis_iter(Axis(0)).enumerate() {
            outer_feature_sum += &outer_product(&sample, &sample);
        }

        let mut covariance_estimate = outer_feature_sum / samples as f64;

        let (mut precision_estimate,mut covariance_log_determinant, mut determinant_rank) = pinv_pdet(&covariance_estimate.view())?;

        self.means = feature_means;
        self.variances = variances;
        self.covariance = covariance_estimate;
        self.pseudo_precision = precision_estimate;
        self.pdet = covariance_log_determinant;
        self.rank = determinant_rank;
        self.samples = samples as u32;

        // eprintln!("Uninformed estimate");
        // eprintln!("{:?}",self);

        Ok(self)

    }


    pub fn estimate(&mut self,data:&ArrayView<f64,Ix2>) -> Result<&mut MVN,LinalgError> {

        let (samples,features) = data.dim();
        let (scaled,sample_means,sample_variances) = scale(data);

        let mut posterior_covariance = Array::eye(features);
        let mut posterior_pseudo_precision = Array::eye(features);
        let mut posterior_pseudo_determinant = 1.;
        let mut posterior_rank = 0.;

        eprintln!("====================");
        eprintln!("Estimated means: {:?}", sample_means);
        eprintln!("Samples in estimate:{:?}",samples);

        let posterior_means = ((&self.means * (self.samples as f64)) + (&sample_means * (samples as f64))) / (self.samples as usize + samples) as f64;
        let posterior_variances = ((&self.variances * (self.samples as f64)) + (&sample_variances * (samples as f64))) / (self.samples as usize + samples) as f64;
        // eprintln!("Posterior means: {:?}", posterior_means);

        eprintln!("====================");
        eprintln!("Posterior means: {:?}", posterior_means);


        let mut s: Array<f64,Ix2> = Array::zeros((features,features));

        for (i,sample) in scaled.axis_iter(Axis(0)).enumerate() {
            s += &outer_product(&sample, &sample);
        }

        let mut covariance_estimate = &s / (samples + 1) as f64;

        // eprintln!("Covariance estimate: {:?}", covariance_estimate);

        {

            let lo = &self.covariance * self.samples as f64;

            let prior_mean_delta = &sample_means - &self.means;
            let mean_delta_outer = outer_product(&prior_mean_delta.view(),&prior_mean_delta.view());

            let mean_delta_scale = (((self.samples + 1) as usize * (samples + 1)) / (self.samples as usize + samples + 1)) as f64;

            // let mut ln = lo + &s + &(mean_delta_scale * mean_delta_outer);
            let mut ln = &lo + &s + &(mean_delta_scale * mean_delta_outer);

            // let inverse_wishart_scale = std::cmp::max((self.samples as usize + samples - features - 1),1) as f64;
            let inverse_wishart_scale = (self.samples as usize + samples) as f64;


            // eprintln!("Inverse wishart scale: {:?}",inverse_wishart_scale);

            posterior_covariance.assign(&mut (ln / inverse_wishart_scale));

            let (mut p_i,mut p_d, mut p_r) = pinv_pdet(&posterior_covariance.view())?;

            posterior_pseudo_determinant = p_d;
            posterior_pseudo_precision.assign(&mut p_i);
            posterior_rank = p_r;
        }

        self.means = posterior_means;
        self.variances = posterior_variances;
        self.covariance = posterior_covariance;
        self.pseudo_precision = posterior_pseudo_precision;
        self.pdet = posterior_pseudo_determinant;
        self.rank = posterior_rank;
        self.samples = self.samples + samples as u32;

        eprintln!("EFM:{:?}",self.means);

        Ok(self)

    }

    pub fn log_likelihood(&self,data:&ArrayView<f64,Ix1>) -> f64 {


        let centered_data = (data - &self.means);

        // let scaled_data: Array<f64,Ix1> = centered_data.iter().zip(&self.variances).map(|(cd,v)| if *v > 0. {cd/(v.sqrt())} else {0.}).collect();
        let scaled_data: Array<f64,Ix1> = centered_data.iter().zip(&self.variances).map(|(cd,v)| if *v > 0. {cd/v} else {0.}).collect();

        let pd = self.pdet;
        let f = scaled_data.dot(&self.pseudo_precision).dot(&scaled_data) * f64::log2(2.*PI);
        let dn = self.rank * f64::log2(2.*PI);

        // eprintln!("D:{:?}",data);
        // eprintln!("M:{:?}",self.means);
        // eprintln!("V:{:?}",self.variances);
        //
        // eprintln!("S:{:?}",scaled_data);
        // eprintln!("P:{:?}",self.pseudo_precision);
        //

        let log_likelihood = -0.5 * (pd + f + dn);

        // eprintln!("PD,F,DN:{},{},{}",pd,f,dn);
        // eprintln!("LL:{:?}",log_likelihood);


        log_likelihood
    }

    pub fn dim(&self) -> (usize,usize) {
        (self.samples as usize,self.means.dim())
    }

    pub fn means(&self) -> ArrayView<f64,Ix1> {
        self.means.view()
    }

    pub fn pdet(&self) -> &f64 {
        &self.pdet
    }

    pub fn variances(&self) -> ArrayView<f64,Ix1> {
        self.variances.diag()
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

pub fn vec_pinv(v:&ArrayView<f64,Ix1>) -> Array<f64,Ix1> {
    let sq_magnitude = v.fold(0.,|acc,vv| acc + vv);
    v/sq_magnitude
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

pub fn pinv_pdet(mtx:&ArrayView<f64,Ix2>) -> Result<(Array<f64,Ix2>,f64,f64),LinalgError> {
    let (r,c) = mtx.dim();
    // eprintln!("Inverting:");
    // eprintln!("{:?}",mtx);
    if let (Some(u),sig,Some(vt)) = mtx.svd(true,true)? {
        // eprintln!("{:?},{:?},{:?}",u,sig,vt);
        let lower_bound = (EPSILON * 1000000.);
        let i_sig = sig.mapv(|v| if v > lower_bound {1./v} else {0.} );
        // let i_sig = sig.mapv(|v| (-(v+1.).log2()).exp2());
        // let i_sig = sig.mapv(|v| (1./(1.+v)));
        let rank = sig.mapv(|v| if v > lower_bound {1.} else {0.}).sum();
        let l_sig = sig.mapv(|v| if v > lower_bound {v.log2()} else {0.});
        // let l_sig = sig.mapv(|v| (v+1.).log2());
        // eprintln!("IS:{:?}",i_sig);
        // eprintln!("{:?}",l_sig);
        let pdet = l_sig.iter().sum();
        // let pdet = sig.iter().fold(0.,|acc,sv| if *sv > lower_bound {acc + sv.log2()} else {acc});
        let mut t_sig = Array::zeros((c,r));
        t_sig.diag_mut().assign(&i_sig);
        let p_i = vt.t().dot(&t_sig).dot(&u.t());
        Ok((p_i,pdet,rank))
    }
    // else {Err(LinalgError::from(LapackError::new(0)))}
    else {Err(LinalgError::Lapack{return_code:0})}
}

pub fn scale(data:&ArrayView<f64,Ix2>) -> (Array<f64,Ix2>,Array<f64,Ix1>,Array<f64,Ix1>) {
    let means = data.mean_axis(Axis(0));
    let variances = data.var_axis(Axis(0),0.);
    let mut scaled = data.to_owned();
    for mut row in scaled.axis_iter_mut(Axis(0)) {
        row -= &means;
        row /= &variances;
    }
    return (scaled,means,variances);
}

pub fn minor_inverse(inverse:&ArrayView<f64,Ix2>,mask:&ArrayView<bool,Ix1>) -> Array<f64,Ix2> {

    // eprintln!("Selecting minor of an inverse");
    //
    // eprintln!("Mask:{:?}",mask);

    let mut minor = array_double_mask(inverse, mask);

    // eprintln!("Minor:{:?}",minor.dim());

    let selected_indices: Vec<usize> = (0..mask.dim()).zip(mask.iter()).flat_map(|(i,&m)| if m {Some(i)} else {None}).collect();
    let omitted_indices: Vec<usize> = (0..mask.dim()).zip(mask.iter()).flat_map(|(i,&m)| if m {None} else {Some(i)}).collect();

    // eprintln!("Selected Indices:{:?}",selected_indices);
    // eprintln!("Omitted Indices:{:?}",omitted_indices);

    let left = inverse.select(Axis(0),&selected_indices).select(Axis(1),&omitted_indices);
    let right = inverse.select(Axis(0),&omitted_indices).select(Axis(1),&selected_indices);

    // eprintln!("L:{:?}",left.dim());
    // eprintln!("R:{:?}",right.dim());

    let augmentation = left.dot(&right);

    // eprintln!("A:{:?}",augmentation.dim());

    minor = minor + augmentation;

    minor

}

#[cfg(test)]
mod tree_braider_tests {

    // extern crate intel_mkl_src;
    // extern crate openblas_src;
    // extern crate netlib_src;
    extern crate blas_src;

    use super::*;
    use crate::read_matrix;
    use crate::MarkovNode;
    use crate::tree_braider_tests::iris_forest;

    pub fn iris_matrix() -> Array<f64,Ix2> {
        read_matrix("../testing/iris.trunc").unwrap()
    }

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
        let data = MarkovNode::encode(&nodes);
        eprintln!("{:?}",data.mean_axis(Axis(0)));
        eprintln!("{:?}",data.var_axis(Axis(0),0.));
    }

    #[test]
    fn test_mvn_normal_blank() {
        let nodes = iris_forest();
        let data = MarkovNode::encode(&nodes);
        let normal = MVN::identity_prior(150,4);
    }

    #[test]
    fn test_mvn_normal_iris() {
        let nodes = iris_forest();
        let data = MarkovNode::encode(&nodes);
        let mut normal = MVN::identity_prior(100,4);
        normal.estimate(&data.view()).unwrap();
        eprintln!("{:?}",normal.means);
        eprintln!("{:?}",normal.covariance);
        eprintln!("{:?}",normal.pseudo_precision);
        eprintln!("{:?}",normal.pdet);
        // panic!();
    }

    #[test]
    fn test_mvn_normal_id_prior_iris() {
        let nodes = iris_forest();
        let data = MarkovNode::encode(&nodes);
        let normal = MVN::estimate_against_identity(&data.view(), None).unwrap();
        eprintln!("{:?}",normal.means);
        eprintln!("{:?}",normal.covariance);
        eprintln!("{:?}",normal.pseudo_precision);
        eprintln!("{:?}",normal.pdet);
        // panic!();
    }


    #[test]
    fn test_mvn_normal_likelihood_iris() {
        let nodes = iris_forest();
        let data = MarkovNode::encode(&nodes);
        let normal = MVN::estimate_against_identity(&data.view(), None).unwrap();
        eprintln!("{:?}",data.axis_iter(Axis(0)).map(|d| normal.log_likelihood(&d)).collect::<Vec<f64>>());
        eprintln!("{:?}",normal.means);
        eprintln!("{:?}",normal.pdet);
        eprintln!("{:?}",normal.covariance);
    }

    #[test]
    fn test_mvn_pinv() {
        let a = array![[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]];
        eprintln!("{:?}",pinv_pdet(&a.view()));
        let b = array![[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]];
        eprintln!("{:?}",pinv_pdet(&b.view()));
        let c = array![[1.,0.],[0.,1.]];
        eprintln!("{:?}",pinv_pdet(&c.view()));
        // panic!();
    }

    #[test]
    fn test_mvn_scale() {
        let iris = iris_matrix();
        let mask = Array::from_shape_fn((150,4),|_| true);
        eprintln!("{:?}",scale(&iris.view()));
    }

    // #[test]
    // fn test_mvn_order_op() {
    //     let a: Array<f64,Ix1> = array![1.,2.,3.];
    //     let b: Array<f64,Ix2> = array![[1.,2.,3.]];
    //     let c: Array<f64,Ix2> = array![[1.],[2.],[3.]];
    //     let d: Array<f64,Ix2> = array![[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]];
    //     eprintln!("{:?}",a.dot(&d));
    //     eprintln!("{:?}",b.dot(&d));
    //     eprintln!("{:?}",c.dot(&d));
    //     eprintln!("{:?}",vec_pinv(&a.view()));
    //     panic!();
    // }


}
