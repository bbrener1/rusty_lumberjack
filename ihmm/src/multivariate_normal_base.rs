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
    samples: usize,
}

impl MVN {

    pub fn identity_prior(samples:usize,features:usize) -> MVN {
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
        self.samples = samples;
    }

    pub fn scaled_identity_prior(means:&ArrayView<f64,Ix1>,variances:&ArrayView<f64,Ix1>,samples:usize) -> MVN {

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
        let (samples,features) = data.dim();
        let prior_strength = strength_option.unwrap_or(features as u32);
        // let rank_deficit = features as i32 - samples as i32;
        // let prior_strength = std::cmp::max(rank_deficit,1) as u32;
        // let prior_strength = samples as u32;
        // let mut prior = MVN::identity_prior(prior_strength as u32, features as u32);
        let (scaled,means,variances) = scale(&data);

        eprintln!("Estimating against scaled identity");
        eprintln!("mean:{:?},var:{:?}",means,variances);

        let mut prior = MVN::scaled_identity_prior(&means.view(), &variances.view(), samples);

        // eprintln!("{:?}",prior);

        prior.estimate(data)?;

        // eprintln!("Estimated:{:?}",prior);

        Ok(prior)
    }

    pub fn uninformed_estimate_masked(&mut self,data:&ArrayView<f64,Ix2>,mask:&ArrayView<bool,Ix2>) -> Result<&mut MVN,LinalgError> {

        let (samples,features) = data.dim();

        let (scaled,feature_means,variances) = scale(data);

        let float_mask = mask.map(|b| if *b {1.} else {0.});

        let mut outer_feature_sum: Array<f64,Ix2> = Array::zeros((features,features));

        for (i,sample) in scaled.axis_iter(Axis(0)).enumerate() {
            outer_feature_sum += &outer_product(&sample, &sample);
        }

        let mut scale_factor: Array<f64,Ix2> = Array::ones((features,features));

        for (i,sample) in float_mask.axis_iter(Axis(0)).enumerate() {
            scale_factor += &outer_product(&sample, &sample);
        }

        scale_factor /= samples as f64;

        let s = outer_feature_sum / scale_factor;

        let mut covariance_estimate = &s / samples as f64;

        let (mut precision_estimate,mut covariance_log_determinant, mut determinant_rank) = pinv_pdet(&covariance_estimate.view())?;

        self.means = feature_means;
        self.variances = variances;
        self.covariance = covariance_estimate;
        self.pseudo_precision = precision_estimate;
        self.pdet = covariance_log_determinant;
        self.rank = determinant_rank;
        self.samples = samples;

        // eprintln!("Uninformed estimate");
        // eprintln!("{:?}",self);

        Ok(self)

    }


    pub fn estimate(&mut self,data:&ArrayView<f64,Ix2>) -> Result<&mut MVN,LinalgError> {


        let (samples,features) = data.dim();
        let (scaled,sample_means,sample_variances) = scale(data);

        let mut posterior_covariance = Array::eye(features);
        let mut posterior_variances = Array::ones(features);
        let mut posterior_pseudo_precision = Array::eye(features);
        let mut posterior_pseudo_determinant = self.pdet;
        let mut posterior_rank = self.rank;

        eprintln!("====================");
        eprintln!("Estimated means: {:?}", sample_means);
        eprintln!("Samples in estimate:{:?}",samples);

        let posterior_means = ((&self.means * (self.samples as f64)) + (&sample_means * (samples as f64))) / (self.samples as usize + samples) as f64;
        // eprintln!("Posterior means: {:?}", posterior_means);

        // eprintln!("====================");
        // eprintln!("Posterior means: {:?}", posterior_means);

        // eprintln!("Scaled:{:?}",scaled);

        let mut s: Array<f64,Ix2> = scaled.t().dot(&scaled);

        // let mut s: Array<f64,Ix2> = Array::zeros((features,features));

        // for (i,sample) in centered.axis_iter(Axis(0)).enumerate() {
        //     s += &outer_product(&sample, &sample);
        // }

        let mut covariance_estimate = &s / (samples + 1) as f64;

        // eprintln!("Covariance estimate: {:?}", covariance_estimate);

        {

            let prior_covariance = &self.covariance;

            let lo = prior_covariance * (self.samples as f64 + 1.);

            let prior_mean_delta = &sample_means - &self.means;
            let mean_delta_outer = outer_product(&prior_mean_delta.view(),&prior_mean_delta.view());

            let mean_delta_scale = (((self.samples + 1) as usize * (samples + 1)) / (self.samples as usize + samples + 1)) as f64;

            // let mut ln = lo + &s + &(mean_delta_scale * mean_delta_outer);
            let mut ln = &lo + &s + &(mean_delta_scale * mean_delta_outer);

            // let inverse_wishart_scale = std::cmp::max((self.samples as usize + samples - features - 1),1) as f64;
            let inverse_wishart_scale = (self.samples as usize + samples) as f64;

            // eprintln!("Inverse wishart scale: {:?}",inverse_wishart_scale);

            posterior_covariance.assign(&(ln / inverse_wishart_scale));

            eprintln!("Posterior covariance:{:?}",posterior_covariance);

            posterior_variances.assign(&posterior_covariance.diag());

            let (ppp,pdet,rank) = pinv_pdet(&posterior_covariance.view()).expect("Inverse failed");

            eprintln!("Pseudo precision:{:?}",ppp);

            posterior_pseudo_precision.assign(&ppp);
            posterior_pseudo_determinant = pdet;
            posterior_rank = rank;
        }

        self.means = posterior_means;
        self.variances = posterior_variances;
        self.covariance = posterior_covariance;
        self.pseudo_precision = posterior_pseudo_precision;
        self.pdet = posterior_pseudo_determinant;
        self.rank = posterior_rank;
        self.samples = self.samples + samples;

        Ok(self)

    }

    pub fn mini_estimate(&mut self,data:&ArrayView<f64,Ix2>) -> Result<&mut MVN,LinalgError> {

        let (samples,features) = data.dim();
        let (scaled,sample_means,sample_variances) = scale(data);

        let posterior_means = ((&self.means * (self.samples as f64)) + (&sample_means * (samples as f64))) / (self.samples as usize + samples) as f64;

        self.means = posterior_means;
        self.variances = sample_variances;
        self.samples = self.samples + samples;

        // eprintln!("EFM:{:?}",self.means);

        Ok(self)

    }

    pub fn log_likelihood(&self,data:&ArrayView<f64,Ix1>) -> f64 {


        let centered_data = (data - &self.means);

        // let scaled_data: Array<f64,Ix1> = centered_data.iter().zip(&self.variances).map(|(cd,v)| if *v > 0. {cd/(v.sqrt())} else {0.}).collect();
        let scaled_data: Array<f64,Ix1> = centered_data.iter().zip(&self.variances).map(|(cd,v)| if *v > 0. {cd/v} else {0.}).collect();

        let pd = self.pdet;
        let f = scaled_data.dot(&self.pseudo_precision).dot(&scaled_data);
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
        //
        // let log_odds = log_likelihood - ((1. - log_likelihood.exp2()).log2());
        //
        // eprintln!("LO:{:?}",log_odds);
        //
        // log_odds
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

pub fn pinv_pdet(mtx:&ArrayView<f64,Ix2>) -> Result<(Array<f64,Ix2>,f64,f64),LinalgError> {
    let (r,c) = mtx.dim();
    // eprintln!("Inverting:");
    // eprintln!("{:?}",mtx);
    if let (Some(u),sig,Some(vt)) = mtx.svd(true,true)? {

        let reduction = 3;
        let lower_bound = (EPSILON * 1000000.);

        let reduced_u = u.slice(s![..,..reduction]).to_owned();
        let mut reduced_sig: Array<f64,Ix2> = Array::zeros((reduction,reduction));
        reduced_sig.diag_mut().assign(&sig.iter().take(reduction).cloned().collect::<Array<f64,Ix1>>());
        let mut reduced_inverse_sig = Array::zeros((reduction,reduction));
        reduced_inverse_sig.diag_mut().assign(&reduced_sig.diag().mapv(|v| if v > lower_bound {1./v} else {0.} ));
        let reduced_vt = vt.slice(s![..reduction,..]).to_owned();
        let reduced_pdet = reduced_sig.mapv(|v| if v > lower_bound {v.log2()} else {0.}).iter().sum();

        let rank = reduced_sig.mapv(|v| if v > lower_bound {1.} else {0.}).sum();

        let reduced_precision = reduced_vt.t().dot(&reduced_inverse_sig).dot(&reduced_u.t());

        Ok((reduced_precision,reduced_pdet,rank))
    }
    // else {Err(LinalgError::from(LapackError::new(0)))}
    else {Err(LinalgError::Lapack{return_code:0})}
}

pub fn scale(data:&ArrayView<f64,Ix2>) -> (Array<f64,Ix2>,Array<f64,Ix1>,Array<f64,Ix1>) {
    let means = data.mean_axis(Axis(0));
    let variances = data.var_axis(Axis(0),0.);
    let variance_zero_mask = variances.mapv(|v| v == 0.);
    let mut inverse_variances = 1./&variances;
    for (mut element,mask) in inverse_variances.iter_mut().zip(variance_zero_mask.iter()) {
        if *mask {
            *element = 0.;
        }
    }
    let mut scaled = data.to_owned();
    for mut row in scaled.axis_iter_mut(Axis(0)) {
        row -= &means;
        row *= &inverse_variances;
    }
    return (scaled,means,variances);
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
