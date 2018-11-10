
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
use io::DropMode;
use io::Parameters;


// #[derive(Debug,Clone,Serialize,Deserialize)]
#[derive(Debug,Clone)]
pub struct RankTable {
    meta_vector: Vec<Arc<RankVector<Vec<Node>>>>,
    pub feature_names: Vec<String>,
    pub sample_names: Vec<String>,
    feature_dictionary: HashMap<String,usize>,
    sample_dictionary: HashMap<String,usize>,
    draw_order: Vec<usize>,
    index: usize,
    pub dimensions: (usize,usize),
    dropout: DropMode,

    dispersion_mode: DispersionMode,
    norm_mode: NormMode,
    split_fraction_regularization: i32
}


impl RankTable {

    pub fn new<'a> (counts: &Vec<Vec<f64>>,feature_names:&'a [String],sample_names:&'a [String], parameters:Arc<Parameters>) -> RankTable {

        let mut meta_vector = Vec::new();

        let mut feature_dictionary: HashMap<String,usize> = HashMap::with_capacity(feature_names.len());

        let sample_dictionary: HashMap<String,usize> = sample_names.iter().cloned().enumerate().map(|x| (x.1,x.0)).collect();

        // println!("{:?}", counts);
        // println!("{:?}", feature_names);

        for (i,(name,loc_counts)) in feature_names.iter().cloned().zip(counts.iter()).enumerate() {
            if i%200 == 0 {
                println!("Initializing: {}",i);
            }
            // println!("Starting to iterate");
            feature_dictionary.insert(name.clone(),i);
            // println!("Updated feature dict");
            // println!("{:?}", loc_counts);
            let mut construct = RankVector::<Vec<Node>>::link(loc_counts);
            // println!("Made a rank vector");
            construct.drop_using_mode(parameters.dropout);
            meta_vector.push(Arc::new(construct));
        }

        let draw_order = (0..counts.get(0).unwrap_or(&vec![]).len()).collect::<Vec<usize>>();

        let dim = (meta_vector.len(),meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        println!("Made rank table with {} features, {} samples:", dim.0,dim.1);

        RankTable {
            meta_vector:meta_vector,
            feature_names:feature_names.iter().cloned().collect(),
            sample_names:sample_names.iter().cloned().collect(),
            draw_order:draw_order,
            index:0,
            dimensions:dim,
            feature_dictionary: feature_dictionary,
            sample_dictionary: sample_dictionary,
            dropout:parameters.dropout,

            norm_mode: parameters.norm_mode,
            dispersion_mode: parameters.dispersion_mode,
            split_fraction_regularization: parameters.split_fraction_regularization as i32,
        }

    }

    pub fn empty() -> RankTable {
        RankTable {
            meta_vector:vec![],
            feature_names:vec![],
            sample_names:vec![],
            draw_order:vec![],
            index:0,
            dimensions:(0,0),
            feature_dictionary: HashMap::with_capacity(0),
            sample_dictionary: HashMap::with_capacity(0),
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

    pub fn sort_by_feature(& self, feature: &str) -> (Vec<usize>,HashSet<usize>) {

        self.meta_vector[self.feature_dictionary[feature]].draw_and_drop()

    }

    pub fn split_indecies_by_feature(&self, feature: &str, split: &f64) -> (Vec<usize>,Vec<usize>){
        self.meta_vector[self.feature_dictionary[feature]].split_indecies(split)
    }

    pub fn feature_name(&self, feature_index: usize) -> Option<&String> {
        self.feature_names.get(feature_index)
    }

    pub fn feature_index(&self, feature_name: &str) -> Option<&usize> {
        self.feature_dictionary.get(feature_name)
    }

    pub fn feature_fetch(&self, feature: &str, index: usize) -> f64 {
        self.meta_vector[self.feature_dictionary[feature]].fetch(index)
    }

    pub fn features(&self) -> &Vec<String> {
        &self.feature_names
    }

    pub fn sample_name(&self, index:usize) -> String {
        self.sample_names[index].clone()
    }

    pub fn sample_index(&self, sample_name: &str) -> usize {
        self.sample_dictionary[sample_name]
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

    pub fn samples(&self) -> &[String] {
        &self.sample_names[..]
    }

    pub fn dispersion_mode(&self) -> DispersionMode {
        self.dispersion_mode
    }

    pub fn set_dispersion_mode(&mut self, dispersion_mode: DispersionMode) {
        self.dispersion_mode = dispersion_mode;
    }

    pub fn derive(&self, indecies:&[usize]) -> RankTable {

        let mut new_meta_vector: Vec<Arc<RankVector<Vec<Node>>>> = Vec::with_capacity(indecies.len());

        let index_set: HashSet<&usize> = indecies.iter().collect();

        let mut new_sample_dictionary: HashMap<String,usize> = HashMap::with_capacity(indecies.len());

        let mut new_sample_names: Vec<String> = Vec::with_capacity(indecies.len());

        for (i,sample_name) in self.sample_names.iter().enumerate() {
            if index_set.contains(&i) {
                new_sample_names.push(sample_name.clone());
                new_sample_dictionary.insert(sample_name.clone(),new_sample_names.len()-1);
            }
        }

        for feature in &self.meta_vector {
            new_meta_vector.push(Arc::new(feature.derive(indecies)));
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (self.meta_vector.len(),self.meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        let child = RankTable {

            meta_vector: new_meta_vector,
            feature_names: self.feature_names.clone(),
            sample_names: new_sample_names,
            feature_dictionary: self.feature_dictionary.clone(),
            sample_dictionary: new_sample_dictionary,
            draw_order: new_draw_order,
            index: 0,
            dimensions: dimensions,
            dropout: self.dropout,
            norm_mode: self.norm_mode,
            dispersion_mode: self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization,
        };

        // println!("{:?}",child.samples());
        // println!("{:?}",child.features());
        // println!("{:?}",child.full_ordered_values());

        child

    }

    //
    // pub fn arc_features(&mut self) -> Vec<RankVector<Vec<Node>>> {
    //
    //     let mut out = Vec::with_capacity(0);
    //     swap(&mut self.meta_vector,&mut out);
    //     out
    // }
    //
    // pub fn return_features(&mut self, returned: Vec<RankVector<Vec<Node>>>) {
    //     self.meta_vector = returned;
    // }

    pub fn derive_specified(&self, features:&Vec<&String>,samples:&Vec<&String>) -> RankTable {

        let indecies: Vec<usize> = samples.iter().map(|x| self.sample_index(x)).collect();
        let index_set: HashSet<&usize> = indecies.iter().collect();

        let mut new_meta_vector: Vec<Arc<RankVector<Vec<Node>>>> = Vec::with_capacity(features.len());

        let mut new_sample_names = Vec::with_capacity(samples.len());
        let mut new_sample_dictionary = HashMap::with_capacity(samples.len());

        for (i,sample_name) in self.sample_names.iter().enumerate() {
            if index_set.contains(&i) {
                new_sample_names.push(sample_name.clone());
                new_sample_dictionary.insert(sample_name.clone(),new_sample_names.len()-1);
            }
        }

        let mut new_feature_dictionary = HashMap::with_capacity(features.len());
        let mut new_feature_names = Vec::with_capacity(features.len());

        for (i,feature) in features.iter().cloned().enumerate() {
            new_meta_vector.push(Arc::new(self.meta_vector[self.feature_dictionary[feature]].derive(&indecies)));
            new_feature_names.push(feature.clone());
            new_feature_dictionary.insert(feature.clone(),new_feature_names.len()-1);
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        RankTable {

            meta_vector: new_meta_vector,
            feature_names: new_feature_names,
            sample_names: new_sample_names,
            feature_dictionary: new_feature_dictionary,
            sample_dictionary: new_sample_dictionary,
            draw_order: new_draw_order,
            index: 0,
            dimensions: dimensions,
            dropout: self.dropout,
            norm_mode: self.norm_mode,
            dispersion_mode: self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization

        }

    }

    pub fn derive_random(&self, features:usize,samples:usize) -> RankTable {

        let mut rng = rand::thread_rng();

        let indecies = rand::seq::sample_iter(&mut rng, 0..self.sample_names.len(), samples).expect("Couldn't generate sample subset");

        let index_set: HashSet<&usize> = indecies.iter().collect();

        // println!("Derive debug {},{}", samples, indecies.len());

        let mut new_meta_vector: Vec<Arc<RankVector<Vec<Node>>>> = Vec::with_capacity(features);

        let new_sample_names: Vec<String> = self.sample_names.iter().enumerate().filter(|x| index_set.contains(&x.0)).map(|x| x.1).cloned().collect();
        let new_sample_dictionary : HashMap<String,usize> = new_sample_names.iter().enumerate().map(|(count,sample)| (sample.clone(),count)).collect();

        let mut new_feature_dictionary = HashMap::with_capacity(features);
        let mut new_feature_names = Vec::with_capacity(features);

        for (i,feature) in rand::seq::sample_iter(&mut rng, self.feature_names.iter().enumerate(), features).expect("Couldn't process feature during subsampling") {
            new_meta_vector.push(Arc::new(self.meta_vector[i].derive(&indecies)));
            new_feature_names.push(feature.clone());
            new_feature_dictionary.insert(feature.clone(),new_feature_names.len()-1);
        }

        let new_draw_order: Vec<usize> = (0..indecies.len()).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        //
        // println!("Feature dict {:?}", new_feature_dictionary.clone());
        // println!("New sample dict {:?}", new_sample_dictionary.clone());

        RankTable {

            meta_vector: new_meta_vector,
            feature_names: new_feature_names,
            sample_names: new_sample_names,
            feature_dictionary: new_feature_dictionary,
            sample_dictionary: new_sample_dictionary,
            draw_order: new_draw_order,
            index: 0,
            dimensions: dimensions,
            dropout: self.dropout,
            norm_mode: self.norm_mode,
            dispersion_mode: self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization,
        }
    }

    pub fn parallel_split_order_min(&self,draw_order:&Vec<usize>, drop_set: &HashSet<usize>,feature_weights:Option<&Vec<f64>>, pool:mpsc::Sender<FeatureMessage>) -> Option<(usize,usize,f64)> {

        if draw_order.len() < 6 {
            return None;
        };

        let disp_mtx_opt: Option<Vec<Vec<f64>>> = self.parallel_dispersion(draw_order,drop_set,pool);

        if let Some(disp_mtx) = disp_mtx_opt {

            if let Some((split_index, raw_split_dispersion)) = match self.norm_mode {
                NormMode::L1 => l1_minimum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.feature_names.len()])),
                NormMode::L2 => l2_minimum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.feature_names.len()])),
            } {

                // println!("Raw dispersion: {}", raw_split_dispersion);

                let split_sample_index = draw_order[split_index];

                let minimum = (split_index,split_sample_index,raw_split_dispersion * ((self.sample_names.len() - draw_order.len() + 1) as f64));

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
                NormMode::L1 => l1_maximum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.feature_names.len()])),
                NormMode::L2 => l2_maximum(&disp_mtx, feature_weights.unwrap_or(&vec![1.;self.feature_names.len()])),
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
            feature_names:self.feature_names,
            sample_names:self.sample_names,
            draw_order:self.draw_order,
            index:0,
            dimensions:self.dimensions,
            feature_dictionary:self.feature_dictionary,
            sample_dictionary:self.sample_dictionary,
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
    pub feature_names: Vec<String>,
    pub sample_names: Vec<String>,
    feature_dictionary: HashMap<String,usize>,
    sample_dictionary: HashMap<String,usize>,
    draw_order: Vec<usize>,
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
            feature_names:self.feature_names,
            sample_names:self.sample_names,
            draw_order:self.draw_order,
            index:0,
            dimensions:self.dimensions,
            feature_dictionary:self.feature_dictionary,
            sample_dictionary:self.sample_dictionary,
            dropout:self.dropout,

            norm_mode:self.norm_mode,
            dispersion_mode:self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization,
        }
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
        let table = RankTable::new(&vec![vec![1.,2.,3.],vec![4.,5.,6.],vec![7.,8.,9.]], &vec!["one".to_string(),"two".to_string(),"three".to_string()], &vec!["0".to_string(),"1".to_string(),"2".to_string()],blank_parameter());
        assert_eq!(table.medians(),vec![2.,5.,8.]);
        assert_eq!(table.dispersions(),vec![1.,1.,1.]);
        assert_eq!(*table.feature_index("one").unwrap(),0);
    }

    #[test]
    fn rank_table_trivial_test() {
        let mut params = Parameters::empty();
        params.dropout = DropMode::No;
        let table = RankTable::new(&Vec::new(), &Vec::new(), &Vec::new(),Arc::new(params));
        let empty: Vec<f64> = Vec::new();
        assert_eq!(table.medians(),empty);
        assert_eq!(table.dispersions(),empty);
    }

    #[test]
    pub fn rank_table_simple_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],blank_parameter());
        let draw_order = table.sort_by_feature("one");
        println!("{:?}",draw_order);
        let mad_order = table.meta_vector[*table.feature_index("one").unwrap()].clone_to_container(SmallVec::new()).ordered_meds_mads(&draw_order.0,draw_order.1);
        assert_eq!(mad_order, vec![(5.0,7.0),(7.5,8.),(10.,5.),(12.5,5.),(15.,5.),(17.5,2.5),(20.,0.),(0.,0.)]);
    }

    #[test]
    pub fn split() {
        let mut table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],blank_parameter());
        let pool = FeatureThreadPool::new(1);
        let mut draw_order = {(table.sort_by_feature("one").0.clone(),table.sort_by_feature("one").1.clone())};

        println!("{:?}", table.sort_by_feature("one"));
        println!("{:?}", table.clone().parallel_dispersion(&table.sort_by_feature("one").0,&table.sort_by_feature("one").1,FeatureThreadPool::new(1)));
        println!("{:?}", table.clone().parallel_dispersion(&table.sort_by_feature("one").0,&table.sort_by_feature("one").1,FeatureThreadPool::new(1)));
        assert_eq!(table.parallel_split_order_min(&draw_order.0, &draw_order.1, Some(&vec![1.]), pool).unwrap().0,3)

    }

    #[test]
    pub fn rank_table_derive_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]], &vec!["one".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],blank_parameter());
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
        println!("{:?}",kid1);
        println!("{:?}",kid2);
        assert_eq!(kid1.medians(),vec![10.]);
        assert_eq!(kid1.dispersions(),vec![5.]);
        assert_eq!(kid2.medians(),vec![2.]);
        assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_table_derive_empty_test() {
        let table = RankTable::new(&vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],vec![0.,1.,0.,1.,0.,1.,0.,1.]], &vec!["one".to_string(),"two".to_string()], &(0..8).map(|x| x.to_string()).collect::<Vec<String>>()[..],blank_parameter());
        let kid1 = table.derive(&vec![0,2,4,6]);
        let kid2 = table.derive(&vec![1,3,5,7]);
    }


}
