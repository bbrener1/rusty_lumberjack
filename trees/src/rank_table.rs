
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::mpsc;
extern crate rand;
use std::f64;

use crate::feature_thread_pool::FeatureMessage;
use crate::Feature;
use crate::Sample;
use crate::Prerequisite;
use crate::Split;
use crate::{l1_sum,l2_sum};
use crate::io::NormMode;
use crate::io::DispersionMode;
use crate::rank_vector::RankVector;
use crate::rank_vector::Node;
use crate::rank_vector::Stencil;
use crate::io::DropMode;
use crate::io::Parameters;


// #[derive(Debug,Clone,Serialize,Deserialize)]
#[derive(Debug,Clone)]
pub struct RankTable {
    meta_vector: Vec<Arc<RankVector<Vec<Node>>>>,
    features: Vec<Feature>,
    samples: Vec<Sample>,
    index: usize,
    pub dimensions: (usize,usize),
    dropout: DropMode,

    dispersion_mode: DispersionMode,
    norm_mode: NormMode,
    split_fraction_regularization: i32
}


impl RankTable {

    pub fn new<'a> (counts: &Vec<Vec<f64>>,features: &'a[Feature],samples: &'a[Sample], parameters:Arc<Parameters>) -> RankTable {

        let mut meta_vector = Vec::new();

        for (i,loc_counts) in counts.iter().enumerate() {
            if i%200 == 0 {
                println!("Initializing: {}",i);
            }
            let mut construct = RankVector::<Vec<Node>>::link(loc_counts);
            construct.drop_using_mode(parameters.dropout);
            meta_vector.push(Arc::new(construct));
        }

        let dim = (meta_vector.len(),meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        println!("Made rank table with {} features, {} samples:", dim.0,dim.1);

        RankTable {
            meta_vector:meta_vector,

            index:0,
            dimensions:dim,
            dropout:parameters.dropout,

            features: features.iter().cloned().collect(),
            samples: samples.iter().cloned().collect(),

            norm_mode: parameters.norm_mode,
            dispersion_mode: parameters.dispersion_mode,
            split_fraction_regularization: parameters.split_fraction_regularization as i32,
        }

    }

    pub fn empty() -> RankTable {
        RankTable {
            meta_vector:vec![],
            features:vec![],
            samples:vec![],
            index:0,
            dimensions:(0,0),
            // feature_dictionary: HashMap::with_capacity(0),
            dropout:DropMode::No,

            norm_mode: NormMode::L1,
            dispersion_mode: DispersionMode::MAD,
            split_fraction_regularization: 1,
        }

    }

    pub fn medians(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.median()).collect()
    }

    pub fn dispersions(&self) -> Vec<f64> {

        match self.dispersion_mode {
            DispersionMode::Variance => self.meta_vector.iter().map(|x| x.var()).collect(),
            DispersionMode::MAD => self.meta_vector.iter().map(|x| x.mad()).collect(),
            DispersionMode::SSME => self.meta_vector.iter().map(|x| x.ssme()).collect(),
            DispersionMode::SME => self.meta_vector.iter().map(|x| x.sme()).collect(),
            DispersionMode::Mixed => panic!("Mixed mode isn't a valid setting for dispersion calculation in individual trees")
        }
    }

    pub fn covs(&self) -> Vec<f64> {
        self.dispersions().into_iter().zip(self.dispersions().into_iter()).map(|x| x.0/x.1).map(|y| if y.is_nan() {0.} else {y}).collect()
    }

    pub fn mask_prerequisites(&self,prerequisites:&[Prerequisite]) -> Vec<bool> {
        let mut mask = vec![true;self.dimensions.1];
        for p in prerequisites {
            if let Some(fi) = self.features.iter().position(|x| x == &p.feature) {
                if p.orientation {
                    for (i,v) in self.meta_vector[fi].full_values().iter().enumerate() {
                        if v <= &p.split {
                            mask[i] = false;
                        }
                    }
                }
                else {
                    for (i,v) in self.meta_vector[fi].full_values().iter().enumerate() {
                        if v > &p.split {
                            mask[i] = false;
                        }
                    }
                }
            }
        }
        mask
    }

    pub fn samples_given_prerequisites(&self,prerequisites:&[Prerequisite]) -> Vec<(usize,&Sample)> {
        let mask = self.mask_prerequisites(prerequisites);
        self.samples.iter()
            .zip(mask.iter())
            .enumerate()
            .filter(|(i,(s,m))| **m )
            .map(|(i,(s,m))| (i,s))
            .collect()
    }

    pub fn sort_by_feature(&self, feature:usize) -> (Vec<usize>,HashSet<usize>) {
        self.meta_vector[feature].draw_and_drop()
    }

    pub fn feature_fetch(&self, feature: usize, sample: usize) -> f64 {
        self.meta_vector[feature].fetch(sample)
    }

    pub fn features(&self) -> &[Feature] {
        &self.features[..]
    }

    pub fn sample_name(&self, index:usize) -> String {
        self.samples[index].name.clone()
    }

    pub fn full_values(&self) -> Vec<Vec<f64>> {
        let mut values = Vec::new();
        for feature in &self.meta_vector {
            values.push(feature.full_values());
        }
        values
    }

    pub fn full_ordered_values(&self) -> Vec<Vec<f64>> {
        self.meta_vector.iter().map(|x| x.ordered_values()).collect()
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples[..]
    }

    pub fn dispersion_mode(&self) -> DispersionMode {
        self.dispersion_mode
    }

    pub fn set_dispersion_mode(&mut self, dispersion_mode: DispersionMode) {
        self.dispersion_mode = dispersion_mode;
    }

    pub fn derive(&self, samples:&[usize]) -> RankTable {

        let dummy_features: Vec<usize> = (0..self.features.len()).collect();

        self.derive_specified(&dummy_features[..], samples)

    }

    pub fn filter_prototype_by_prerequisites(&self,prerequisites:&Vec<Prerequisite>) -> Vec<&Sample> {



        let mut masks: Vec<Vec<bool>> = Vec::with_capacity(prerequisites.len());

        for Prerequisite{feature:feature,split:split,orientation:orientation} in prerequisites {

            // NOTE, VERY IMPORTANT, THIS PART WORKS ONLY IN THE PROTOTYPE
            if *orientation {
                masks.push(self.meta_vector[*feature.index()].full_values().iter().map(|v| v > split).collect())
            }
            else {
                masks.push(self.meta_vector[*feature.index()].full_values().iter().map(|v| v < split).collect())
            }

        }

        let mut filtered_samples = Vec::with_capacity(self.samples.len());

        for sample in self.samples.iter() {
            if masks.iter_mut().flat_map(|m| m.pop()).any(|x| x) {
                filtered_samples.push(sample)
            }
        }

        filtered_samples

    }

    pub fn derive_specified(&self, features:&[usize],samples:&[usize]) -> RankTable {

        let mut new_meta_vector: Vec<Arc<RankVector<Vec<Node>>>> = Vec::with_capacity(features.len());

        let mut new_samples = samples.iter().map(|i| self.samples[*i].clone()).collect();

        let mut new_features = features.iter().map(|i| self.features[*i].clone()).collect();

        let sample_stencil = Stencil::from_slice(samples);

        let mut new_meta_vector: Vec<Arc<RankVector<Vec<Node>>>> = features.iter().map(|i| Arc::new(self.meta_vector[*i].derive_stencil(&sample_stencil))).collect();

        let new_draw_order: Vec<usize> = (0..samples.len()).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        RankTable {

            meta_vector: new_meta_vector,
            features: new_features,
            samples: new_samples,
            index: 0,
            dimensions: dimensions,
            dropout: self.dropout,
            norm_mode: self.norm_mode,
            dispersion_mode: self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization

        }

    }

    pub fn order_dispersions(&self,draw_order:&[usize],drop_set:&HashSet<usize>,feature_weights:&[f64]) -> Option<Vec<f64>> {
        let full_dispersion = self.full_dispersion(draw_order,drop_set);

        match self.norm_mode {
            NormMode::L1 => { Some(l1_sum(full_dispersion.as_ref()?,feature_weights))},
            NormMode::L2 => { Some(l2_sum(full_dispersion.as_ref()?,feature_weights))},
        }
    }

    pub fn full_dispersion(&self,draw_order:&[usize], drop_set: &HashSet<usize>) -> Option<Vec<Vec<f64>>> {

        if draw_order.len() < 6 {
            return None
        }

        let mut forward_dispersions: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];draw_order.len()+1];
        let mut reverse_dispersions: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];draw_order.len()+1];

        let mut worker_vec = RankVector::empty_sv();

        for (i,v) in self.meta_vector.iter().enumerate() {
            worker_vec.clone_from_prototype(v);
            let fd = match self.dispersion_mode {
                DispersionMode::Variance => worker_vec.ordered_variance(&draw_order,&drop_set),
                DispersionMode::MAD => worker_vec.ordered_mads(&draw_order,&drop_set),
                DispersionMode::SSME => worker_vec.ordered_ssme(&draw_order, &drop_set),
                DispersionMode::SME => worker_vec.ordered_sme(&draw_order,&drop_set),
                DispersionMode::Mixed => panic!("Mixed mode not a valid split setting for individual trees!"),
            };
            for (j,fr) in fd.into_iter().enumerate() {
                forward_dispersions[j][i] = fr;
            }
        }

        let mut reverse_draw_order:Vec<usize> = draw_order.to_owned();
        reverse_draw_order.reverse();

        for (i,v) in self.meta_vector.iter().enumerate() {
            worker_vec.clone_from_prototype(v);
            let mut rd = match self.dispersion_mode {
                DispersionMode::Variance => worker_vec.ordered_variance(&reverse_draw_order,&drop_set),
                DispersionMode::MAD => worker_vec.ordered_mads(&reverse_draw_order,&drop_set),
                DispersionMode::SSME => worker_vec.ordered_ssme(&reverse_draw_order, &drop_set),
                DispersionMode::SME => worker_vec.ordered_sme(&reverse_draw_order,&drop_set),
                DispersionMode::Mixed => panic!("Mixed mode not a valid split setting for individual trees!"),
            };
            for (j,rr) in rd.into_iter().enumerate() {
                reverse_dispersions[reverse_draw_order.len() - j][i] = rr;
            }
        }

        assert_eq!(draw_order.len(),7);
        assert_eq!(forward_dispersions.len(),8);
        assert_eq!(reverse_dispersions.len(),8);

        let len = forward_dispersions.len();
        let mut dispersions: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];len];

        for (i,(f_s,r_s)) in forward_dispersions.into_iter().zip(reverse_dispersions.into_iter()).enumerate() {
            for (j,(gf,gr)) in f_s.into_iter().zip(r_s.into_iter()).enumerate() {
                dispersions[i][j] = (gf * ((len - i) as f64 / len as f64).powi(self.split_fraction_regularization)) + (gr * ((i+1) as f64/ len as f64).powi(self.split_fraction_regularization));
            }
        }

        Some(dispersions)

    }

    pub fn wrap_consume(self) -> RankTableWrapper{
        RankTableWrapper {
            meta_vector:self.meta_vector.into_iter().map(|x| Arc::try_unwrap(x).expect("Failed to unwrap value during serialization")).collect(),
            features:self.features,
            samples:self.samples,
            index:0,
            dimensions:self.dimensions,
            dropout:self.dropout,

            norm_mode:self.norm_mode,
            dispersion_mode:self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization
        }

    }
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct RankTableWrapper {
    meta_vector: Vec<RankVector<Vec<Node>>>,
    features: Vec<Feature>,
    samples: Vec<Sample>,
    index: usize,
    pub dimensions: (usize,usize),
    dropout: DropMode,

    dispersion_mode: DispersionMode,
    norm_mode: NormMode,
    split_fraction_regularization: i32,
}

impl RankTableWrapper {
    pub fn unwrap(self) -> RankTable {
        RankTable {
            meta_vector:self.meta_vector.into_iter().map(|x| Arc::new(x)).collect(),
            features:self.features,
            samples:self.samples,
            index:0,
            dimensions:self.dimensions,
            dropout:self.dropout,

            norm_mode:self.norm_mode,
            dispersion_mode:self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization,
        }
    }
}





#[cfg(test)]
mod rank_table_tests {

    use super::*;
    use smallvec::SmallVec;
    use crate::feature_thread_pool::FeatureThreadPool;

    fn blank_parameter() -> Arc<Parameters> {
        let mut parameters = Parameters::empty();

        parameters.dropout = DropMode::Zeros;

        Arc::new(parameters)
    }

    #[test]
    fn rank_table_general_test() {
        let table = RankTable::new(&vec![vec![1.,2.,3.],vec![4.,5.,6.],vec![7.,8.,9.]], &Feature::vec(vec![1,2,3])[..],&Sample::vec(vec![1,2,3])[..],blank_parameter());
        assert_eq!(table.medians(),vec![2.,5.,8.]);

        // This assertion tests a default dispersion of SSME
        assert_eq!(table.dispersions(),vec![2.,2.,2.]);

        // This assertion tests a default dispersion of Variance
        // assert_eq!(table.dispersions(),vec![1.,1.,1.]);
    }

    #[test]
    fn rank_table_trivial_test() {
        let mut params = Parameters::empty();
        params.dropout = DropMode::No;
        let table = RankTable::new(&Vec::new(), &Vec::new(),&Vec::new(),Arc::new(params));
        let empty: Vec<f64> = Vec::new();
        assert_eq!(table.medians(),empty);
        assert_eq!(table.dispersions(),empty);
    }

    #[test]
    pub fn rank_table_simple_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &Feature::vec(vec![1])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        // let draw_order = table.sort_by_feature("one");
        // println!("{:?}",draw_order);
        // let mad_order = table.meta_vector[*table.feature_index("one").unwrap()].clone_to_container(SmallVec::new()).ordered_meds_mads(&draw_order.0,draw_order.1);
        // assert_eq!(mad_order, vec![(5.0,7.0),(7.5,8.),(10.,5.),(12.5,5.),(15.,5.),(17.5,2.5),(20.,0.),(0.,0.)]);
    }

    #[test]
    pub fn split() {
        let mut table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &Feature::vec(vec![1])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        let pool = FeatureThreadPool::new(1);
        // let mut draw_order = {(table.sort_by_feature("one").0.clone(),table.sort_by_feature("one").1.clone())};

        // println!("{:?}", table.sort_by_feature("one"));
        // println!("{:?}", table.clone().parallel_dispersion(&table.sort_by_feature("one").0,&table.sort_by_feature("one").1,FeatureThreadPool::new(1)));
        // println!("{:?}", table.clone().parallel_dispersion(&table.sort_by_feature("one").0,&table.sort_by_feature("one").1,FeatureThreadPool::new(1)));
        // assert_eq!(table.parallel_split_order_min(&draw_order.0, &draw_order.1, Some(&vec![1.]), pool).unwrap().0,3)

    }

    #[test]
    pub fn rank_table_derive_test() {
        let mut table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&Feature::vec(vec![1])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
        println!("{:?}",kid1);
        println!("{:?}",kid2);
        assert_eq!(kid1.medians(),vec![10.]);
        assert_eq!(kid2.medians(),vec![2.]);

        // These assertions test ssme as a dispersion
        assert_eq!(kid1.dispersions(),vec![169.]);
        assert_eq!(kid2.dispersions(),vec![367.]);

        // // These assertions test variance as a dispersion
        // assert_eq!(kid1.dispersions(),vec![5.]);
        // assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_table_derive_feature_twice() {
        let mut table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&Feature::vec(vec![1])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        let kid = table.derive_specified(&vec![0,0],&vec![0,2,4,6]);
        println!("{:?}",kid);
        assert_eq!(kid.medians(),vec![10.,10.]);

        // These assertions test ssme as a dispersion
        assert_eq!(kid.dispersions(),vec![169.,169.]);

        // // These assertions test variance as a dispersion
        // assert_eq!(kid1.dispersions(),vec![5.]);
        // assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_table_derive_sample_twice() {
        let mut table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&Feature::vec(vec![1])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        let kid = table.derive_specified(&vec![0],&vec![0,2,4,6,6]);
        println!("{:?}",kid);
        assert_eq!(kid.medians(),vec![12.5]);

        // These assertions test ssme as a dispersion
        assert_eq!(kid.dispersions(),vec![229.]);

        // // These assertions test variance as a dispersion
        // assert_eq!(kid1.dispersions(),vec![5.]);
        // assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_table_derive_empty_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],vec![0.,1.,0.,1.,0.,1.,0.,1.]], &Feature::vec(vec![1,2])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
    }


}
