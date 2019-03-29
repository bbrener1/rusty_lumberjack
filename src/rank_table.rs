
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::mpsc;
use feature_thread_pool::FeatureMessage;
extern crate rand;
use std::f64;
use io::NormMode;
use io::DispersionMode;
use rank_vector::RankVector;
use rank_vector::Node;
use rank_vector::Stencil;
use io::DropMode;
use io::Parameters;


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


    pub fn parallel_split_order_min(&self,draw_order:&Vec<usize>, drop_set: &HashSet<usize>,feature_weights:Option<&Vec<f64>>, pool:mpsc::Sender<FeatureMessage>) -> Option<(usize,usize,f64)> {

        if draw_order.len() < 6 {
            return None;
        };

        let disp_mtx_opt: Option<Vec<Vec<f64>>> = self.parallel_dispersion(draw_order,drop_set,pool);

        if let Some(disp_mtx) = disp_mtx_opt {

            if let Some((split_index, raw_split_dispersion)) = match self.norm_mode {
                NormMode::L1 => l1_minimum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.features.len()])),
                NormMode::L2 => l2_minimum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.features.len()])),
            } {

                // println!("Raw dispersion: {}", raw_split_dispersion);

                let split_sample_index = draw_order[split_index];

                let minimum = (split_index,split_sample_index,raw_split_dispersion * ((self.samples.len() - draw_order.len() + 1) as f64));

                // println!("Minimum: {:?}", minimum);

                Some(minimum)

            }
            else { None }

        }
        else { None }
    }

    pub fn parallel_split_order_max(&mut self,draw_order:&Vec<usize>, drop_set: &HashSet<usize>,feature_weights:Option<&Vec<f64>>, pool:mpsc::Sender<FeatureMessage>) -> Option<(usize,f64)> {

        if draw_order.len() < 6 {
            return None;
        };

        let disp_mtx_opt: Option<Vec<Vec<f64>>> = self.parallel_dispersion(draw_order,drop_set,pool);

        if let Some(disp_mtx) = disp_mtx_opt {

            let mut maximum = match self.norm_mode {
                NormMode::L1 => l1_maximum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.features.len()])),
                NormMode::L2 => l2_maximum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.features.len()])),
            };

            maximum = maximum.map(|z| (z.0, z.1 * draw_order.len() as f64));

            maximum

        }
        else { None }
    }

    pub fn parallel_dispersion(&self,draw_order:&Vec<usize>, drop_set: &HashSet<usize>, pool:mpsc::Sender<FeatureMessage>) -> Option<Vec<Vec<f64>>> {

        let forward_draw_arc = Arc::new(draw_order.clone());
        let reverse_draw_arc: Arc<Vec<usize>> = Arc::new(draw_order.iter().cloned().rev().collect());

        let drop_arc = Arc::new(drop_set.clone());

        if forward_draw_arc.len() < 6 {
            return None
        }

        let mut forward_dispersions: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];forward_draw_arc.len()+1];
        let mut reverse_dispersions: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];reverse_draw_arc.len()+1];

        let mut forward_receivers = Vec::with_capacity(self.dimensions.0);
        let mut reverse_receivers = Vec::with_capacity(self.dimensions.0);

        // let cd = self.dispersions();
        // let cm = self.medians();

        for feature in self.meta_vector.iter().cloned() {
            let (tx,rx) = mpsc::channel();
            pool.send(FeatureMessage::Message((feature,forward_draw_arc.clone(),drop_arc.clone(),self.dispersion_mode),tx));
            forward_receivers.push(rx);
        }

        for (i,fr) in forward_receivers.iter().enumerate() {
            if let Ok(disp) = fr.recv() {
                for (j,g) in disp.into_iter().enumerate() {
                    // println!("{:?}", g);
                    forward_dispersions[j][i] = g;
                }
            }
            else {
                panic!("Parellelization error!")
            }

        }

        for feature in self.meta_vector.iter().cloned() {
            let (tx,rx) = mpsc::channel();
            pool.send(FeatureMessage::Message((feature,reverse_draw_arc.clone(),drop_arc.clone(),self.dispersion_mode),tx));
            reverse_receivers.push(rx);
        }

        for (i,rr) in reverse_receivers.iter().enumerate() {
            if let Ok(disp) = rr.recv() {
                for (j,g) in disp.into_iter().enumerate() {
                    reverse_dispersions[reverse_draw_arc.len() - j][i] = g;
                }
            }
            else {
                panic!("Parellelization error!")
            }

        }

        let len = forward_dispersions.len();

        // println!("{:?}",forward_dispersions);
        // println!("{:?}",reverse_dispersions);

        let mut dispersions: Vec<Vec<f64>> = vec![vec![0.;self.dimensions.0];len];

        // match self.dispersion_mode {
        //     DispersionMode::SSME | DispersionMode::SME => {
        //         for (i,(f_s,r_s)) in forward_dispersions.into_iter().zip(reverse_dispersions.into_iter()).enumerate() {
        //             for (j,(gf,gr)) in f_s.into_iter().zip(r_s.into_iter()).enumerate() {
        //                 dispersions[i][j] = gf + gr;
        //             }
        //         }
        //     }
        //     _ => {
        //         for (i,(f_s,r_s)) in forward_dispersions.into_iter().zip(reverse_dispersions.into_iter()).enumerate() {
        //             for (j,(gf,gr)) in f_s.into_iter().zip(r_s.into_iter()).enumerate() {
        //                 dispersions[i][j] = (gf * ((len - i) as f64 / len as f64).powi(self.split_fraction_regularization)) + (gr * ((i+1) as f64/ len as f64).powi(self.split_fraction_regularization));
        //             }
        //         }
        //     }
        // }

        for (i,(f_s,r_s)) in forward_dispersions.into_iter().zip(reverse_dispersions.into_iter()).enumerate() {
            for (j,(gf,gr)) in f_s.into_iter().zip(r_s.into_iter()).enumerate() {
                dispersions[i][j] = (gf * ((len - i) as f64 / len as f64).powi(self.split_fraction_regularization)) + (gr * ((i+1) as f64/ len as f64).powi(self.split_fraction_regularization));
            }
        }


        // println!("{:?}", covs);
        //
        // println!("___________________________________________________________________________");
        // println!("___________________________________________________________________________");
        // println!("___________________________________________________________________________");


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

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct Feature {
    name: String,
    index: usize,
}

impl Feature {

    pub fn vec(input: Vec<usize>) -> Vec<Feature> {
        input.iter().map(|x| Feature::q(x)).collect()
    }

    pub fn nvec(input: &Vec<String>) -> Vec<Feature> {
        input.iter().enumerate().map(|(i,f)| Feature::new(f,&i)).collect()
    }

    pub fn q(index:&usize) -> Feature {
        Feature {name: index.to_string(),index:*index}
    }

    pub fn new(name:&str,index:&usize) -> Feature {
        Feature {name: name.to_owned(),index:*index}
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn index(&self) -> &usize {
        &self.index
    }
}

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct Sample {
    name: String,
    index: usize,
}

impl Sample {

    pub fn vec(input: Vec<usize>) -> Vec<Sample> {
        input.iter().map(|x| Sample::q(x)).collect()
    }

    pub fn nvec(input: &Vec<String>) -> Vec<Sample> {
        input.iter().enumerate().map(|(i,s)| Sample::new(s,&i)).collect()
    }

    pub fn q(index:&usize) -> Sample {
        Sample {name: index.to_string(),index:*index}
    }

    pub fn new(name:&str,index:&usize) -> Sample {
        Sample {name: name.to_owned(),index:*index}
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn index(&self) -> &usize {
        &self.index
    }
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Prerequisite {
    feature: Feature,
    split: f64,
    orientation: bool
}

impl Prerequisite {
    pub fn new(feature:Feature,split:f64,orientation:bool) -> Prerequisite {
        Prerequisite {feature,split,orientation}
    }
}


pub fn l2_minimum(mtx_in:&Vec<Vec<f64>>, weights: &Vec<f64>) -> Option<(usize,f64)> {

    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature.powi(2) * weights[i]).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY});

    // println!("Scoring:");
    // println!("{:?}", mtx_in.iter().map(|sample| {
    //     sample.iter().enumerate().map(|(i,feature)| feature.powi(2) * weights[i]).sum::<f64>() / weight_sum
    // }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY}).enumerate().collect::<Vec<(usize,f64)>>());

    sample_sums.enumerate().skip(3).rev().skip(3).min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater))

}

pub fn l1_minimum(mtx_in:&Vec<Vec<f64>>, weights: &Vec<f64>) -> Option<(usize,f64)> {

    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature * weights[i] ).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY});

    // println!("Scoring:");
    // println!("{:?}", mtx_in.iter().map(|sample| {
    //     sample.iter().enumerate().map(|(i,feature)| feature * weights[i]).sum::<f64>() / weight_sum
    // }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY}).enumerate().collect::<Vec<(usize,f64)>>());

    sample_sums.enumerate().skip(3).rev().skip(3).min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater))

}

pub fn l2_maximum(mtx_in:&Vec<Vec<f64>>, weights: &Vec<f64>) -> Option<(usize,f64)> {

    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature.powi(2) * weights[i]).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {0.});

    sample_sums.enumerate().skip(3).rev().skip(3).max_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater))

}

pub fn l1_maximum(mtx_in:&Vec<Vec<f64>>, weights: &Vec<f64>) -> Option<(usize,f64)> {

    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature * weights[i] ).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {0.});

    // println!("Scoring:");
    // println!("{:?}", mtx_in.iter().map(|sample| {
    //     sample.iter().enumerate().map(|(i,feature)| feature * weights[i]).sum::<f64>() / weight_sum
    // }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {0.}).enumerate().collect::<Vec<(usize,f64)>>());

    sample_sums.enumerate().skip(3).rev().skip(3).max_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater))

}

#[cfg(test)]
mod rank_table_tests {

    use super::*;
    use smallvec::SmallVec;
    use feature_thread_pool::FeatureThreadPool;

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
    pub fn rank_table_derive_empty_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],vec![0.,1.,0.,1.,0.,1.,0.,1.]], &Feature::vec(vec![1,2])[..],&Sample::vec(vec![0,1,2,3,4,5,6,7])[..],blank_parameter());
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
    }


}
