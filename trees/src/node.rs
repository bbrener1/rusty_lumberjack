
use std::sync::Arc;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::sync::mpsc;
use std::f64;
use std::mem::replace;
use std::collections::{HashMap,HashSet};
use serde_json;

extern crate rand;
use rand::Rng;

use rayon::prelude::*;

use crate::rank_table::RankTable;
use crate::Feature;
use crate::Sample;
use crate::Prerequisite;
use crate::Split;
use crate::Braid;
use crate::io::DropMode;
use crate::io::PredictionMode;
use crate::io::Parameters;
use crate::io::DispersionMode;
use crate::argmin;

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;

use rayon::prelude::*;


#[derive(Clone,Serialize,Deserialize)]
pub struct PrototypeNode {

    input_table: RankTable,
    output_table: RankTable,
    dropout: DropMode,

    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    prerequisites: Vec<Prerequisite>,
    braids: Vec<Braid>,

    depth: usize,

    split: Option<Split>,


    pub medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,

}

#[derive(Clone,Serialize,Deserialize)]
pub struct ComputeNode {

    input_table: RankTable,
    output_table: RankTable,
    dropout: DropMode,

    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    depth: usize,

    split: Option<Split>,

    pub medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,

}

#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct CompactNode {

    dropout: DropMode,

    pub children: Vec<CompactNode>,

    split: Option<Split>,

    prerequisites: Vec<Prerequisite>,
    braids: Vec<Braid>,

    features: Vec<Feature>,
    samples: Vec<Sample>,
    medians: Vec<f64>,
    dispersions: Vec<f64>,
    weights: Vec<f64>,


    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>,
}

//
// pub trait Node {
//
//     fn set_dispersion_mode(&mut self, dispersion_mode : DispersionMode);
//     fn dispersion_mode(&self) -> DispersionMode;
//     fn drop_mode(&self) -> DropMode;
//     fn id(&self) -> &str;
//
//     fn output_rank_table(&self) -> &RankTable;
//     fn input_rank_table(&self) -> &RankTable;
//
//     fn samples(&self) -> &[Sample];
//     fn sample_names(&self) -> Vec<String>;
//     fn input_features(&self) -> &[Feature];
//     fn input_feature_names(&self) -> Vec<String>;
//     fn output_features(&self) -> &[Feature];
//     fn output_feature_names(&self) -> Vec<String>;
//
//     fn depth(&self) -> usize;
//     fn split(&self) -> &Option<Split>;
//
//     fn medians(&self) -> &Vec<f64>;
//     fn dispersions(&self) -> &Vec<f64>;
//     fn dimensions(&self) -> (usize,usize);
//     fn absolute_gains(&self) -> Option<&Vec<f64>>;
//     fn local_gains(&self) -> Option<&Vec<f64>>;
//
//     fn set_weights(&mut self, weights:Vec<f64>);
//
// }

impl PrototypeNode {

    pub fn prototype<'a>(input_counts:&Vec<Vec<f64>>,output_counts:&Vec<Vec<f64>>,input_features:&'a[Feature],output_features:&'a[Feature],samples:&'a[Sample], parameters: Arc<Parameters> , feature_weight_option: Option<Vec<f64>>) -> PrototypeNode {

        let input_table = RankTable::new(input_counts,parameters.clone());
        let output_table = RankTable::new(output_counts,parameters.clone());
        let feature_weights = feature_weight_option.unwrap_or(vec![1.;output_features.len()]);
        let medians = output_table.medians();
        let dispersions = output_table.dispersions();
        let local_gains = vec![0.;dispersions.len()];

        let new_node = PrototypeNode {

            input_table: input_table,
            output_table: output_table,
            dropout: parameters.dropout,

            input_features: input_features.to_owned(),
            output_features: output_features.to_owned(),
            samples: samples.to_owned(),

            depth: 0,

            split: None,

            prerequisites: vec![],
            braids: vec![],

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
        };

        new_node

    }

    pub fn derive_specified_compute(&self, samples: &[Sample], input_features: &[Feature], output_features: &[Feature]) -> ComputeNode {

        let sample_indices: Vec<usize> = samples.iter().map(|s| s.index).collect();
        let input_feature_indices: Vec<usize> = input_features.iter().map(|f| f.index).collect();
        let output_feature_indices: Vec<usize> = output_features.iter().map(|f| f.index).collect();

        let mut new_input_table = self.input_table.derive_specified(&input_feature_indices,&sample_indices);
        let mut new_output_table = self.output_table.derive_specified(&output_feature_indices,&sample_indices);

        let new_input_features = input_features.to_owned();
        let new_output_features = output_features.to_owned();
        let new_samples = samples.to_owned();

        let medians = new_output_table.medians();
        let dispersions = new_output_table.dispersions();
        let feature_weights = output_feature_indices.iter().map(|y| self.feature_weights[*y]).collect();

        // let local_gains = Some(self.dispersions().iter().zip(dispersions.iter()).map(|(p,c)| (p/(self.samples().len() as f64)) - (c/((new_samples.len() + 1) as f64))).collect());

        let child = ComputeNode {

            input_table: new_input_table,
            output_table: new_output_table,
            dropout: self.dropout,

            input_features: new_input_features,
            output_features: new_output_features,
            samples: new_samples,

            depth: self.depth + 1,

            split: None,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
        };

        child
    }

    pub fn derive_specified_compact(&self, samples: &[Sample], input_features: &[Feature], output_features: &[Feature]) -> ComputeNode {

        let sample_indices: Vec<usize> = samples.iter().map(|s| s.index).collect();
        let input_feature_indices: Vec<usize> = input_features.iter().map(|f| f.index).collect();
        let output_feature_indices: Vec<usize> = output_features.iter().map(|f| f.index).collect();

        let mut new_input_table = self.input_table.derive_specified(&input_feature_indices,&sample_indices);
        let mut new_output_table = self.output_table.derive_specified(&output_feature_indices,&sample_indices);

        let new_input_features = input_features.to_owned();
        let new_output_features = output_features.to_owned();
        let new_samples = samples.to_owned();

        let medians = new_output_table.medians();
        let dispersions = new_output_table.dispersions();
        let feature_weights = output_feature_indices.iter().map(|y| self.feature_weights[*y]).collect();

        let child = CompactNode {

            input_table: new_input_table,
            output_table: new_output_table,
            dropout: self.dropout,

            input_features: new_input_features,
            output_features: new_output_features,
            samples: new_samples,

            depth: self.depth + 1,

            split: None,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
        };

        child
    }


    pub fn derive_prototype_by_prerequsites(&self, prerequisites:&[Prerequisite]) -> PrototypeNode {

        let sample_indices: Vec<usize> = self.indices_given_prerequisites(prerequisites);
        let input_feature_indices: Vec<usize> = self.input_features.iter().map(|f| f.index).collect();
        let output_feature_indices: Vec<usize> = self.output_features.iter().map(|f| f.index).collect();

        let mut new_input_table = self.input_table.derive_specified(&input_feature_indices,&sample_indices);
        let mut new_output_table = self.output_table.derive_specified(&output_feature_indices,&sample_indices);

        let new_input_features = self.input_features.clone();
        let new_output_features = self.input_features.clone();
        let new_samples = sample_indices.iter().map(|i| self.samples[*i].clone()).collect();

        let medians = new_output_table.medians();
        let dispersions = new_output_table.dispersions();
        let feature_weights = output_feature_indices.iter().map(|y| self.feature_weights[*y]).collect();

        // let local_gains = Some(self.dispersions().iter().zip(dispersions.iter()).map(|(p,c)| (p/(self.samples().len() as f64)) - (c/((new_samples.len() + 1) as f64))).collect());

        let child = PrototypeNode {

            input_table: new_input_table,
            output_table: new_output_table,
            dropout: self.dropout,

            input_features: new_input_features,
            output_features: new_output_features,
            samples: new_samples,

            depth: prerequisites.len(),

            split: None,

            braids: self.braids.clone(),
            prerequisites: prerequisites.to_owned(),

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
        };

        child
    }

    pub fn derive_compact_by_braid(&self,braid:Braid) -> CompactNode {

        let samples: Vec<usize> = (0..self.samples.len()).collect()
        let input_features: Vec<usize> = (0..self.samples.len()).collect()
        let output_features: Vec<usize> = (0..self.samples.len()).collect()

        let Braid{draw_order,drop_set,..} = braid;

        let mut compute = self.derive_specified_compute(samples, input_features, output_features);

        let dispersions = compute.output_table.order_dispersions(draw_order,drop_set);
        let (split_index,split_value) = argmin(dispersions);
        let left_samples = draw_order[..split_index].iter().map(|i| self.samples[*i]).collect()
        let right_samples = draw_order[split_index..].iter().map(|i| self.samples[*i]).collect();

    }

    fn indices_given_prerequisites(&self,prerequisites:&[Prerequisite]) -> Vec<usize> {

        // First we create a default-true mask.

        let mut mask = vec![true;self.samples.len()];

        // We then cycle through each prerequisite and figure out the local index of the feature it encodes.

        for prerequisite in prerequisites {
            if let Some(feature_index) = self.input_features.iter().position(|x| x == &prerequisite.feature) {

                // If the feature is present, we create a mask of samples that fulfil that prerequiste
                // We then update the total mask by the feature mask

                let Prerequisite{feature,split,orientation} = prerequisite;
                let feature_mask = self.input_table.feature_value_mask(feature_index,*split,*orientation);
                for (i,m) in feature_mask.into_iter().enumerate() {
                    if !m {
                        mask[i] = false;
                    }
                }
            }
        };

        // In this function we iterate through prerequisites and select only ones that either were dropped in a given feature
        // or fulfil a prerequisite

        mask.into_iter().zip(0..self.samples.len()).filter(|(m,s)| *m).map(|(m,s)| s).collect()

    }


    pub fn draw_random_features(&self,input_features: usize, output_features: usize) -> (Vec<Feature>,Vec<Feature>) {

        let mut rng = rand::thread_rng();

        let drawn_input: Vec<Feature> = (0..input_features).map(|_| rng.gen_range(0,self.input_table.dimensions.0)).map(|i| self.input_features[i].clone()).collect();
        let drawn_output: Vec<Feature> = (0..output_features).map(|_| rng.gen_range(0,self.output_table.dimensions.0)).map(|i| self.output_features[i].clone()).collect();

        (drawn_input,drawn_output)
    }

    pub fn draw_random_samples(&self,samples:usize) -> Vec<Sample> {
        let mut rng = rand::thread_rng();
        (0..samples).map(|_| rng.gen_range(0,self.input_table.dimensions.1)).map(|i| self.samples[i].clone()).collect()
    }

    pub fn subsample(&self,samples:usize,input_features:usize,output_features:usize) -> ComputeNode {
        let (input_features,output_features) = self.draw_random_features(input_features, output_features);
        let samples = self.draw_random_samples(samples);
        self.derive_specified_compute(&samples,&input_features,&output_features)
    }

    pub fn braid_split_node(&mut self,samples:usize,input_features:usize,output_features:usize) -> Option<Vec<CompactNode>> {

        let thickness = 4;
        let mut features = Vec::with_capacity(thickness);
        for i in 0..thickness {
            let mut compute = self.subsample(samples,input_features,output_features);
            if let Some(Split{feature,..}) = compute.rayon_best_split() {
                features.push(feature);
            }
        }
        let rvs: Vec<_> = features.iter().map(|f| self.input_table.rv_fetch(f.index).clone()).collect();

        let braid = Braid::from_rvs(features, &rvs);

        // eprintln!("Braid split:{:?}",braid);

        self.derive_complete_by_braid(braid)
    }

}

impl ComputeNode {


    pub fn rayon_best_split(&self) -> Option<Split> {

        let splits: Vec<Split> =
            (0..self.input_features().len())
            .into_par_iter()
            .flat_map(|i| self.feature_index_split(i))
            .collect();
        let dispersions: Vec<f64> = splits.iter().map(|s| s.dispersion).collect();
        Some(splits[argmin(&dispersions)?.0].clone())

    }

    pub fn feature_index_split(&self,feature_index:usize) -> Option<Split> {
        let feature = self.input_features()[feature_index].clone();
        let (draw_order,drop_set) = self.input_table.sort_by_feature(feature_index);
        let dispersions = self.output_table.order_dispersions(&draw_order,&drop_set,&self.feature_weights)?;
        let (split_index,minimum_dispersion) = argmin(&dispersions[1..])?;
        let split_sample_index = draw_order[split_index];
        let split_value = self.input_table.feature_fetch(feature_index,split_sample_index);
        Some(Split::new(feature,split_value,minimum_dispersion))

    }

}

impl CompactNode {

    fn crawl_children(&self) -> Vec<&CompactNode> {
        let mut output = Vec::new();
        for child in &self.children {
            output.extend(child.crawl_children());
        }
        output.push(&self);
        output
    }

    fn compute_absolute_gains(&mut self,root_dispersions: &Vec<f64>) {

        let mut absolute_gains = Vec::with_capacity(root_dispersions.len());

        for (nd,od) in self.dispersions.iter().zip(root_dispersions.iter()) {
            absolute_gains.push(od-nd)
        }
        self.absolute_gains = Some(absolute_gains);

        for child in self.children.iter_mut() {
            child.compute_absolute_gains(root_dispersions);
        }
    }

    fn root_absolute_gains(&mut self) {
        for child in self.children.iter_mut() {
            child.compute_absolute_gains(&self.dispersions);
        }
    }

    fn crawl_leaves(&self) -> Vec<&CompactNode> {
        let mut output = Vec::new();
        if self.children.len() < 1 {
            return vec![&self]
        }
        else {
            for child in &self.children {
                output.extend(child.crawl_leaves());
            }
        };
        output
    }

    pub fn to_string(self) -> Result<String,serde_json::Error> {
        serde_json::to_string(&self)
    }

    pub fn from_str(input:&str) -> Result<CompactNode,serde_json::Error> {
        serde_json::from_str(input)
    }


    pub fn predict_leaves(&self,vector: &Vec<f64>, header: &HashMap<String,usize>,drop_mode: &DropMode, prediction_mode:&PredictionMode) -> Vec<&CompactNode> {

        let mut leaves = vec![];

        if let Some(Split{feature,value:split, ..}) = self.split() {
            if *vector.get(*header.get(feature.name()).unwrap_or(&(vector.len()+1))).unwrap_or(&drop_mode.cmp()) != drop_mode.cmp() {
                if vector[header[feature.name()]] > *split {
                    leaves.extend(self.children[1].predict_leaves(vector, header, drop_mode, prediction_mode));
                }
                else {
                    leaves.extend(self.children[0].predict_leaves(vector, header, drop_mode, prediction_mode));
                }
            }
            else {
                match prediction_mode {
                    &PredictionMode::Branch => {
                        // println!("Mode is branching");
                        leaves.extend(self.children[1].predict_leaves(vector, header, drop_mode, prediction_mode));
                        leaves.extend(self.children[0].predict_leaves(vector, header, drop_mode, prediction_mode));
                        // println!("{}", leaves.len());
                    },
                    &PredictionMode::Truncate => {
                        leaves.push(&self)
                    },
                    &PredictionMode::Abort => {},
                    &PredictionMode::Auto => {
                        leaves.extend(self.predict_leaves(vector, header, drop_mode, prediction_mode));
                    }
                }
            }
        }
        else {
            // println!("Found a leaf");
            leaves.push(&self);
        }

        leaves

    }


    pub fn from_json(input:&str) -> Result<CompactNode,serde_json::Error> {
        serde_json::from_str(input)
    }

    pub fn from_location(location:&str) -> Result<Vec<CompactNode>,Box<Error>> {
        eprintln!("Current directory: {:?}",env::current_dir()?);
        eprintln!("Attempting to read from {:?}",location);
        let mut lc_pth = Path::new(location);
        if !lc_pth.is_dir() {
            lc_pth = lc_pth.parent().unwrap();
        }
        let tree_locations: Vec<String> =
            lc_pth.read_dir()?
            .flat_map(|t| t)
            .map(|t| t.path())
            .filter(|t| t.extension() == Some(OsStr::new("compact")))
            .map(|t| t.to_str().map(|ts| ts.to_owned()))
            .flat_map(|t| t.into_iter())
            .collect();
        eprintln!("Reading from locations:");
        eprintln!("{:?}",tree_locations);
        let mut nodes = vec![];
        for tl in tree_locations {
            nodes.push(CompactNode::from_file(&tl)?);
        };
        Ok(nodes)
    }

    pub fn from_file(location:&str) -> Result<CompactNode,Box<Error>> {
        let mut json_file = File::open(location)?;
        let mut json_string = String::new();
        json_file.read_to_string(&mut json_string)?;
        Ok(CompactNode::from_json(&json_string)?)
    }

}

#[cfg(test)]
mod node_testing {

    use super::*;
    // use ndarray_linalg;

    fn blank_parameter() -> Arc<Parameters> {
        let mut parameters = Parameters::empty();

        parameters.dropout = DropMode::Zeros;

        Arc::new(parameters)
    }

    fn blank_compact() -> CompactNode {
        CompactNode {
            dropout: DropMode::No,

            children: vec![],

            split: None,

            features: vec![],
            samples: vec![],

            prerequisites: vec![],
            braids: vec![],

            medians: vec![],
            dispersions: vec![],
            weights: vec![],

            local_gains: None,
            absolute_gains: None,
        }
    }

    fn blank_prototype() -> PrototypeNode {
        let input_counts = &vec![];
        let output_counts = &vec![];
        let input_features = &vec![][..];
        let output_features = &vec![][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        PrototypeNode::prototype(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option)
    }

    fn trivial_prototype() -> PrototypeNode {
        let input_counts = &vec![vec![]];
        let output_counts = &vec![vec![]];
        let input_features = &vec![Feature::q(&1)][..];
        let output_features = &vec![Feature::q(&2)][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        PrototypeNode::prototype(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option)
    }

    fn simple_prototype() -> PrototypeNode {
        let input_counts = &vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let output_counts = &vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let input_features = &vec![Feature::q(&1)][..];
        let output_features = &vec![Feature::q(&2)][..];
        let samples = &Sample::vec(vec![0,1,2,3,4,5,6,7])[..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        PrototypeNode::prototype(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option)
    }

    #[test]
    fn node_test_blank() {
        let mut root = blank_prototype()();
        root.medians();
    }

    #[test]
    fn node_test_trivial() {
        let mut root = trivial_prototype();
        root.medians();
    }

    //
    // #[test]
    // fn node_test_dispersions() {
    //
    //     let mut root = simple_node();
    //
    //     let split0 = root.feature_index_split(0).unwrap();
    //
    //     println!("{:?}",root.samples());
    //     println!("{:?}",root.output_table.full_values());
    //     println!("{:?}",split0);
    //
    //     // panic!();
    // }
    //
    // #[test]
    // fn node_test_subsample() {
    //
    //     let mut root = simple_node();
    //
    //
    //     for i in 0..1000 {
    //         let sub = root.subsample(8, 2, 2);
    //         let split_option = sub.rayon_best_split();
    //         eprintln!("{:?}",sub.strip_clone());
    //         let (draw_order,drop_set) = sub.input_rank_table().sort_by_feature(0);
    //         eprintln!("{:?}",(&draw_order,&drop_set));
    //         eprintln!("{:?}",sub.output_rank_table().order_dispersions(&draw_order,&drop_set,&sub.feature_weights));
    //         eprintln!("{:?}",split_option.unwrap());
    //         // if let Some(split) = split_option {
    //         //     root.clone().derive_complete_by_split(&split,None);
    //         // }
    //     }
    //
    // }
    //
    //
    // #[test]
    // fn node_test_split() {
    //
    //     let mut root = simple_node();
    //
    //     let split = root.rayon_best_split().unwrap();
    //
    //     println!("{:?}",split);
    //
    //     assert_eq!(split.dispersion,2822.265625);
    //     assert_eq!(split.value, 5.);
    // }
    //
    // #[test]
    // fn node_test_simple() {
    //
    //     let mut root = simple_node();
    //
    //     root.split_node();
    //
    //     println!("sample_order:{:?}",root.children[0].output_table.full_values());
    //
    //     // assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
    //     // assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"6".to_string(),"7".to_string()]);
    //
    //     // assert_eq!(root.children[0].samples(),&vec!["1".to_string(),"4".to_string(),"5".to_string()]);
    //     // assert_eq!(root.children[1].samples(),&vec!["0".to_string(),"3".to_string(),"6".to_string(),"7".to_string()]);
    //
    //     assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"2".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
    //     assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"2".to_string(),"6".to_string(),"7".to_string()]);
    //
    //     assert_eq!(&root.children[0].output_table.full_values(),&vec![vec![-3.,0.,5.,-2.,-1.]]);
    //     assert_eq!(&root.children[1].output_table.full_values(),&vec![vec![10.,0.,15.,20.]]);
    //
    // }
    //
    //
    // #[test]
    // fn node_test_stripped_file() {
    //     let mut root = StrippedNode::from_file("../testing/iris_forest/run.50.compact").unwrap();
    // }
    //
    // #[test]
    // fn node_test_stripped_location() {
    //     let mut roots = StrippedNode::from_location("../testing/iris_forest/").unwrap();
    // }
    //
    // #[test]
    // fn node_test_json() {
    //     let n = blank_stripped();
    //     let ns = n.to_string();
    //     let r = StrippedNode::from_json(&ns).unwrap();
    // }

}
