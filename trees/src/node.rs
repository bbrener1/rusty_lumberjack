
use std::sync::Arc;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::sync::mpsc;
use std::f64;
use std::mem::replace;
use std::collections::HashMap;
use serde_json;

extern crate rand;
use rand::Rng;

use crate::rank_table::RankTable;
use crate::rank_table::RankTableWrapper;
use crate::Feature;
use crate::Sample;
use crate::Prerequisite;
use crate::split_thread_pool::SplitMessage;
use crate::io::DropMode;
use crate::io::PredictionMode;
use crate::io::Parameters;
use crate::io::DispersionMode;

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;


#[derive(Clone)]
pub struct Node {

    // pool: mpsc::Sender<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>),mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>,
    split_thread_pool: mpsc::Sender<SplitMessage>,

    input_table: RankTable,
    output_table: RankTable,
    dropout: DropMode,

    pub parent_id: String,
    pub id: String,
    pub depth: usize,
    pub children: Vec<Node>,

    feature: Option<Feature>,
    split: Option<f64>,

    prerequisites: Vec<Prerequisite>,

    pub medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,
    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>
}


impl Node {

    pub fn feature_root<'a>(input_counts:&Vec<Vec<f64>>,output_counts:&Vec<Vec<f64>>,input_features:&'a[Feature],output_features:&'a[Feature],samples:&'a[Sample], parameters: Arc<Parameters> , feature_weight_option: Option<Vec<f64>>, split_thread_pool: mpsc::Sender<SplitMessage>) -> Node {

        let input_table = RankTable::new(input_counts,input_features,samples,parameters.clone());

        let output_table = RankTable::new(output_counts,output_features,samples,parameters.clone());

        let feature_weights = feature_weight_option.unwrap_or(vec![1.;output_features.len()]);

        let medians = output_table.medians();

        let dispersions = output_table.dispersions();

        let local_gains = vec![0.;dispersions.len()];

        let new_node = Node {
            split_thread_pool: split_thread_pool,

            input_table: input_table,
            output_table: output_table,
            dropout: parameters.dropout,

            id: "RT".to_string(),
            parent_id: "RT".to_string(),
            depth: 0,
            children: Vec::new(),

            feature: None,
            split: None,

            prerequisites: vec![],

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
            local_gains: Some(local_gains),
            absolute_gains: None
        };

        // assert_eq!(new_node.input_table.features(),new_node.output_table.features());
        assert_eq!(new_node.input_table.samples(),new_node.output_table.samples());
        // println!("Preliminary: {:?}",new_node.output_table.full_ordered_values());

        new_node
    }

    pub fn feature_parallel_best_split(&mut self) -> Option<(Feature,f64,f64,Vec<usize>,Vec<usize>)> {

        if self.input_features().len() < 1 {
            panic!("Tried to split with no input features");
        };

        let mut minimum_receivers = Vec::with_capacity(self.input_features().len());

        let reference_table = Arc::new(replace(&mut self.output_table,RankTable::empty()));
        //
        // println!("Starting to split");

        for input_feature in 0..self.input_features().len() {

            let (tx,rx) = mpsc::channel();

            let (draw_order,drop_set) = self.input_table.sort_by_feature(input_feature);
            //
            // println!("Passed to thread pool");

            self.split_thread_pool.send(SplitMessage::Message((reference_table.clone(),draw_order,drop_set,self.feature_weights.clone()),tx));

            minimum_receivers.push((input_feature,rx));

        };

        let mut minima = Vec::with_capacity(self.input_features().len());
        //
        // println!("Receiving");

        for (input_feature,receiver) in minimum_receivers.into_iter() {
            if let Some((split_index,split_sample_index,split_dispersion)) = receiver.recv().expect("Split thread pool error") {
                minima.push((input_feature,split_index,split_sample_index,split_dispersion))
            }
        }

        replace(&mut self.output_table, Arc::try_unwrap(reference_table).expect("Couldn't unwrap the reference table"));

        // println!("Replaced");

        let (best_feature_local_index,split_index,split_sample_index,split_dispersion) = minima.iter().min_by(|a,b| (a.3).partial_cmp(&b.3).unwrap_or(Ordering::Greater)).unwrap().clone();

        let split_order = self.input_table.sort_by_feature(best_feature_local_index);

        assert_eq!(split_sample_index,split_order.0[split_index]);

        let split_value = self.input_table.feature_fetch(best_feature_local_index,split_sample_index);

        let best_feature = self.input_table.features()[best_feature_local_index].clone();

        self.feature = Some(best_feature.clone());
        self.split = Some(split_value.clone());

        // println!("Best split: {:?}", (best_feature.clone(),split_index, split_value,split_dispersion));

        // println!("{:?}",self.output_table.full_ordered_values());
        // println!("{:?}",self.medians());

        let left_indices = split_order.0[..split_index].iter().cloned().collect();
        let right_indices = split_order.0[split_index..].iter().cloned().collect();

        Some((best_feature,split_dispersion,split_value,left_indices,right_indices))

    }



    pub fn feature_parallel_derive(&mut self,prototype_opt:Option<&Node>) -> Option<()> {

        // println!("Feature parallel derive:");
        // println!("{},{},{},{}", self.input_features().len(),self.output_features().len(),self.input_table.samples().len(),self.output_table.samples().len());

        if let Some((feature,_dispersion,split_value, left_indecies,right_indecies)) = self.feature_parallel_best_split() {

            let mut left_child_id = self.id.clone();
            let mut right_child_id = self.id.clone();
            left_child_id.push_str(&format!("!F{}:S{}L",feature.name(),split_value));
            right_child_id.push_str(&format!("!F{}:S{}R",feature.name(),split_value));

            // println!("{:?}",&left_indecies);
            // println!("{:?}",&right_indecies);

            let left_child;
            let right_child;

            if let Some(prototype) = prototype_opt {
                // left_child = self.derive_resampled(prototype, self.input_features().len(), self.output_features().len(), &left_indecies, &left_child_id,Some(false));
                // right_child = self.derive_resampled(prototype, self.input_features().len(), self.output_features().len(), &right_indecies, &right_child_id,Some(true));

                let (left_prerequisite,right_prerequisite) = (Prerequisite::new(feature.clone(),split_value.clone(),false),Prerequisite::new(feature.clone(),split_value.clone(),true));

                let mut left_prerequisites = self.prerequisites.clone();
                let mut right_prerequisites = self.prerequisites.clone();

                left_prerequisites.push(left_prerequisite);
                right_prerequisites.push(right_prerequisite);

                left_child = self.derive_by_prerequisites(prototype, &left_prerequisites, self.input_features().len(), self.output_features().len(), self.samples().len() , &left_child_id,Some(false));
                right_child = self.derive_by_prerequisites(prototype, &right_prerequisites, self.input_features().len(), self.output_features().len(), self.samples().len() , &right_child_id,Some(true));

            }
            else {
                left_child = self.derive(&left_indecies, &left_child_id,false);
                right_child = self.derive(&right_indecies, &right_child_id,true);
            }

            self.children.push(left_child);
            self.children.push(right_child);

            // println!("Derived children!");

            let ls = self.children[0].samples().len() as f64;
            let rs = self.children[1].samples().len() as f64;
            let os = self.samples().len() as f64;

            let left_dispersions = self.children[0].dispersions.iter().map(|&x| x * ls);
            let right_dispersions = self.children[1].dispersions.iter().map(|&x| x * rs);

            let own_dispersions = self.dispersions.iter().map(|x| x * os);

            let mut local_gains: Vec<f64> = own_dispersions.zip(left_dispersions.zip(right_dispersions)).map(|(o,(l,r))| o - (l+r)).collect();

            local_gains = local_gains.iter().map(|g| g/(self.samples().len() as f64)).collect();

            self.local_gains = Some(local_gains);

            Some(())
        }
        else {
            None
        }


    }

    pub fn derive(&self, indecies: &[usize],new_id:&str,orientation:bool) -> Node {

            let new_input_table = self.input_table.derive(indecies);
            let new_output_table = self.output_table.derive(indecies);

            let mut new_prerequisites = self.prerequisites.clone();
            new_prerequisites.push(Prerequisite::new(self.feature().clone().unwrap(),self.split().clone().unwrap(),orientation));

            let medians = new_output_table.medians();
            let dispersions = new_output_table.dispersions();
            let feature_weights = self.feature_weights.clone();

            for ((nd,nm),(od,om)) in dispersions.iter().zip(medians.iter()).zip(self.dispersions.iter().zip(self.medians.iter())) {
                let mut old_cov = od/om;
                if !old_cov.is_normal() {
                    old_cov = 0.;
                }
                let mut new_cov = nd/nm;
                if !new_cov.is_normal() {
                    new_cov = 0.;
                }
            }

            let child = Node {
                // pool: self.pool.clone(),
                split_thread_pool: self.split_thread_pool.clone(),

                input_table: new_input_table,
                output_table: new_output_table,
                dropout: self.dropout,

                parent_id: self.id.clone(),
                id: new_id.to_string(),
                depth: self.depth + 1,
                children: Vec::new(),

                feature: None,
                split: None,

                prerequisites: new_prerequisites,

                medians: medians,
                feature_weights: feature_weights,
                dispersions: dispersions,
                local_gains: None,
                absolute_gains: None
            };

            // assert_eq!(child.input_table.features(),child.output_table.features());
            assert_eq!(child.input_table.samples(),child.output_table.samples());

            child
        }

    pub fn derive_resampled(&self,prototype:&Node, input_features: usize, output_features: usize, indecies:&Vec<usize>, new_id:&str, orientation:Option<bool>) -> Node {

        let mut rng = rand::thread_rng();

        let new_input_features = (0..input_features).map(|_| rng.gen_range(0, prototype.input_table.dimensions.0)).collect();

        let new_output_features = (0..output_features).map(|_| rng.gen_range(0, prototype.output_table.dimensions.0)).collect();

        let prototype_indecies = indecies.iter().map(|&i| *self.input_table.samples()[i].index()).collect();

        prototype.derive_specified(&prototype_indecies,&new_input_features,&new_output_features,new_id, orientation)
    }

    pub fn derive_by_prerequisites(&self,prototype:&Node, prerequisites: &Vec<Prerequisite>, input_features: usize, output_features: usize, samples: usize, new_id:&str, orientation:Option<bool>) -> Node {

        let mut rng = rand::thread_rng();

        let new_input_features = (0..input_features).map(|_| rng.gen_range(0, prototype.input_table.dimensions.0)).collect();

        let new_output_features = (0..output_features).map(|_| rng.gen_range(0, prototype.output_table.dimensions.0)).collect();

        let potential_sample_indices: Vec<usize> = prototype.samples_given_prerequisites(&prerequisites).iter().map(|(i,s)| *i).collect();

        let prototype_indecies = (0..samples).map(|_| rng.gen_range(0, potential_sample_indices.len())).collect();

        prototype.derive_specified(&prototype_indecies,&new_input_features,&new_output_features,new_id, orientation)
    }

    pub fn derive_random(&self, samples: usize, input_features: usize, output_features: usize, new_id:&str, orientation: Option<bool>) -> Node {

        let mut rng = rand::thread_rng();

        let new_input_features = (0..input_features).map(|_| rng.gen_range(0,self.input_table.dimensions.0)).collect();

        let new_output_features = (0..output_features).map(|_| rng.gen_range(0,self.output_table.dimensions.0)).collect();

        let new_samples = (0..samples).map(|_| rng.gen_range(0,self.output_table.dimensions.1)).collect();

        self.derive_specified(&new_samples,&new_input_features,&new_output_features,new_id,orientation)
    }

    pub fn derive_specified(&self, samples: &Vec<usize>, input_features: &Vec<usize>, output_features: &Vec<usize>, new_id: &str,orientation_option:Option<bool>) -> Node {

        let mut new_input_table = self.input_table.derive_specified(&input_features,samples);
        let mut new_output_table = self.output_table.derive_specified(&output_features,samples);

        let mut new_prerequisites = self.prerequisites.clone();
        if let Some(orientation) = orientation_option {
            new_prerequisites.push(Prerequisite::new(self.feature().clone().unwrap_or(Feature::new("Prototype",&0)),self.split().clone().unwrap_or(0.),orientation));
        }

        let medians = new_output_table.medians();
        let dispersions = new_output_table.dispersions();
        let feature_weights = output_features.iter().map(|y| self.feature_weights[*y]).collect();


        let child = Node {
            // pool: self.pool.clone(),
            split_thread_pool: self.split_thread_pool.clone(),

            input_table: new_input_table,
            output_table: new_output_table,
            dropout: self.dropout,

            parent_id: self.id.clone(),
            id: new_id.to_string(),
            depth: self.depth + 1,
            children: Vec::new(),

            feature: None,
            split: None,

            prerequisites: new_prerequisites,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,
            local_gains: None,
            absolute_gains: None
        };

        // assert_eq!(child.input_table.features(),child.output_table.features());
        assert_eq!(child.input_table.samples(),child.output_table.samples());

        child
    }



    pub fn report(&self,verbose:bool) {
        println!("Node reporting:");
        println!("Feature:{:?}",self.feature);
        println!("Split:{:?}", self.split);
        println!("Output features:{}",self.output_features().len());
        if verbose {
            println!("{:?}",self.output_features());
            println!("{:?}",self.medians);
            println!("{:?}",self.dispersions);
            println!("{:?}",self.feature_weights);
        }
        println!("Samples: {}", self.samples().len());
        if verbose {
            println!("{:?}", self.samples());
            println!("Counts: {:?}", self.output_table.full_ordered_values());
            println!("Ordered counts: {:?}", self.output_table.full_values());
        }

    }


    pub fn summary(&self) -> String {
        let mut report_string = "".to_string();
        if self.children.len() > 1 {
            report_string.push_str(&format!("!ID:{}\n",self.id));
            report_string.push_str(&format!("F:{}\n",self.feature.as_ref().map(|f| f.name()).unwrap_or(&"".to_string())));
            report_string.push_str(&format!("S:{}\n",self.split.unwrap_or(0.)));
        }

        report_string
    }

    pub fn data_dump(&self) -> String {
        let mut report_string = String::new();
        report_string.push_str(&format!("!ID:{}\n",self.id));
        report_string.push_str(&format!("Children:"));
        for child in &self.children {
            report_string.push_str(&format!("!C:{}",child.id));
        }
        report_string.push_str("\n");
        report_string.push_str(&format!("ParentID:{}\n",self.parent_id));
        report_string.push_str(&format!("Feature:{:?}\n", self.feature));
        report_string.push_str(&format!("Split:{:?}\n",self.split));
        report_string.push_str(&format!("Output features:{:?}\n",self.output_features().len()));
        report_string.push_str(&format!("{:?}\n",self.output_features()));
        report_string.push_str(&format!("Medians:{:?}\n",self.medians));
        report_string.push_str(&format!("Dispersions:{:?}\n",self.dispersions));
        report_string.push_str(&format!("Local gains:{:?}\n",self.local_gains));
        report_string.push_str(&format!("Absolute gains:{:?}\n",self.absolute_gains));
        report_string.push_str(&format!("Feature weights:{:?}\n",self.feature_weights));
        report_string.push_str(&format!("Samples:{:?}\n",self.samples().len()));
        report_string.push_str(&format!("{:?}\n",self.samples()));
        report_string.push_str(&format!("Full:{:?}\n",self.output_table.full_ordered_values()));
        report_string
    }


    pub fn set_weights(&mut self, weights:Vec<f64>) {
        self.feature_weights = weights;
    }

    pub fn set_pool(&mut self, pool: &mpsc::Sender<SplitMessage>) {
        self.split_thread_pool = pool.clone()
    }

    pub fn set_dispersion_mode(&mut self, dispersion_mode : DispersionMode) {
        self.output_table.set_dispersion_mode(dispersion_mode);
    }

    pub fn dispersion_mode(&self) -> DispersionMode {
        self.output_table.dispersion_mode()
    }

    pub fn wrap_consume(self) -> NodeWrapper {

        // let mut children: Vec<String> = Vec::with_capacity(self.children.len());
        let mut children: Vec<NodeWrapper> = Vec::with_capacity(self.children.len());

        for child in self.children {
            // children.push(child.wrap_consume().to_string())
            children.push(child.wrap_consume())
        }

        NodeWrapper {
            input_table: self.input_table.wrap_consume(),
            output_table: self.output_table.wrap_consume(),
            dropout: self.dropout,

            parent_id: self.parent_id,
            id: self.id,
            depth: self.depth,
            children: children,

            feature: self.feature,
            split: self.split,

            prerequisites: self.prerequisites,

            medians: self.medians,
            feature_weights: self.feature_weights,
            dispersions: self.dispersions,
            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains

        }

    }

    pub fn strip_consume(self) -> StrippedNode {

        let features = self.output_features().iter().cloned().collect();
        let samples = self.samples().iter().cloned().collect();

        let mut stripped_children = Vec::new();

        for child in self.children.into_iter() {
            stripped_children.push(child.strip_consume())
        }


        StrippedNode {
            dropout: self.dropout,

            children: stripped_children,

            feature: self.feature,
            split: self.split,

            features: features,
            samples: samples,

            prerequisites: self.prerequisites,

            medians: self.medians,
            dispersions: self.dispersions,
            weights: self.feature_weights,

            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains,
        }
    }

    pub fn strip_clone(&self) -> StrippedNode {

        let mut stripped_children = Vec::new();

        for child in &self.children {
            stripped_children.push(child.strip_clone())
        }

        StrippedNode {
            dropout: self.dropout,

            children: stripped_children,

            feature: self.feature.clone(),
            split: self.split.clone(),

            features: self.output_features().iter().cloned().collect(),
            samples: self.samples().iter().cloned().collect(),

            prerequisites: self.prerequisites.clone(),

            medians: self.medians.clone(),
            dispersions: self.dispersions.clone(),
            weights: self.feature_weights.clone(),

            local_gains: self.local_gains.clone(),
            absolute_gains: self.absolute_gains.clone(),
        }

    }

    pub fn output_rank_table(&self) -> &RankTable {
        &self.output_table
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn samples(&self) -> &[Sample] {
        self.output_table.samples()
    }

    pub fn sample_names(&self) -> Vec<String> {
        self.output_table.samples().iter().map(|s| s.name().clone()).collect()
    }

    pub fn input_features(&self) -> &[Feature] {
        self.input_table.features()
    }

    pub fn input_feature_names(&self) -> Vec<String> {
        self.input_table.features().iter().map(|f| f.name().clone()).collect()
    }

    pub fn output_features(&self) -> &[Feature] {
        self.output_table.features()
    }

    pub fn output_feature_names(&self) -> Vec<String> {
        self.output_table.features().iter().map(|f| f.name().clone()).collect()
    }

    pub fn feature(&self) -> &Option<Feature> {
        &self.feature
    }

    pub fn feature_name(&self) -> Option<&String> {
        self.feature.as_ref().map(|f| f.name())
    }

    pub fn samples_given_prerequisites(&self,prerequisites:&Vec<Prerequisite>) -> Vec<(usize,&Sample)> {
        self.input_table.samples_given_prerequisites(prerequisites)
    }

    pub fn split(&self) -> &Option<f64> {
        &self.split
    }

    pub fn medians(&self) -> &Vec<f64> {
        &self.medians
    }

    pub fn dispersions(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn mads(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.output_table.dimensions
    }

    pub fn dropout(&self) -> DropMode {
        self.dropout
    }

    pub fn absolute_gains(&self) -> &Option<Vec<f64>> {
        &self.absolute_gains
    }

    pub fn local_gains(&self) -> Option<&Vec<f64>> {
        self.local_gains.as_ref()
    }

    pub fn covs(&self) -> Vec<f64> {
        self.dispersions.iter().zip(self.mads().iter()).map(|(d,m)| d/m).map(|x| if x.is_normal() {x} else {0.}).collect()
    }

    pub fn wrap_clone(&self) -> NodeWrapper {
        self.clone().wrap_consume()
    }

    pub fn crawl_children(&self) -> Vec<&Node> {
        let mut output = Vec::new();
        for child in &self.children {
            output.extend(child.crawl_children());
        }
        output.push(&self);
        output
    }

    pub fn compute_absolute_gains(&mut self,root_dispersions: &Vec<f64>) {

        let mut absolute_gains = Vec::with_capacity(root_dispersions.len());

        for (nd,od) in self.dispersions.iter().zip(root_dispersions.iter()) {
            absolute_gains.push(od-nd)
        }
        self.absolute_gains = Some(absolute_gains);

        for child in self.children.iter_mut() {
            child.compute_absolute_gains(root_dispersions);
        }
    }

    pub fn root_absolute_gains(&mut self) {
        for child in self.children.iter_mut() {
            child.compute_absolute_gains(&self.dispersions);
        }
    }

    pub fn crawl_leaves(&self) -> Vec<&Node> {
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

}


impl NodeWrapper {
    pub fn to_string(self) -> Result<String,serde_json::Error> {
        serde_json::to_string(&self)
    }

    pub fn unwrap(self,split_thread_pool: mpsc::Sender<SplitMessage>) -> Node {
        let mut children: Vec<Node> = Vec::with_capacity(self.children.len());
        for child in self.children {
            children.push(child.unwrap(split_thread_pool.clone()));
        }

        println!("Recursive unwrap finished!");

        Node {

            split_thread_pool: split_thread_pool,

            input_table: self.input_table.unwrap(),
            output_table: self.output_table.unwrap(),
            dropout: self.dropout,

            parent_id: self.parent_id,
            id: self.id,
            depth: self.depth,
            children: children,

            feature: self.feature,
            split: self.split,

            prerequisites:self.prerequisites,

            medians: self.medians,
            feature_weights: self.feature_weights,
            dispersions: self.dispersions,
            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains
        }

    }

}

#[derive(Serialize,Deserialize)]
pub struct NodeWrapper {

    pub input_table: RankTableWrapper,
    pub output_table: RankTableWrapper,
    pub dropout: DropMode,

    pub parent_id: String,
    pub id: String,
    pub depth: usize,
    pub children: Vec<NodeWrapper>,

    pub feature: Option<Feature>,
    pub split: Option<f64>,

    pub prerequisites: Vec<Prerequisite>,

    pub medians: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,
    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>

}



#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct StrippedNode {

    dropout: DropMode,

    pub children: Vec<StrippedNode>,

    feature: Option<Feature>,
    split: Option<f64>,

    prerequisites: Vec<Prerequisite>,

    features: Vec<Feature>,
    samples: Vec<Sample>,
    medians: Vec<f64>,
    dispersions: Vec<f64>,
    weights: Vec<f64>,


    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>,
}

impl StrippedNode {

    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn feature(&self) -> &Option<Feature> {
        &self.feature
    }

    pub fn feature_name(&self) -> Option<&String> {
        self.feature.as_ref().map(|f| f.name())
    }

    pub fn features(&self) -> &[Feature] {
        &self.features[..]
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples[..]
    }

    pub fn split(&self) -> &Option<f64> {
        &self.split
    }

    pub fn medians(&self) -> &Vec<f64> {
        &self.medians
    }

    pub fn mads(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn covs(&self) -> Vec<f64> {
        self.mads().iter().zip(self.medians().iter()).map(|(d,m)| d/m).map(|x| if x.is_normal() {x} else {0.}).collect()
    }

    pub fn absolute_gains(&self) -> &Option<Vec<f64>> {
        &self.absolute_gains
    }

    pub fn local_gains(&self) -> Option<&Vec<f64>> {
        self.local_gains.as_ref()
    }

    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn dropout(&self) -> DropMode {
        self.dropout
    }

    pub fn crawl_leaves(&self) -> Vec<&StrippedNode> {
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

    pub fn mut_crawl_to_leaves<'a>(&'a mut self) -> Vec<&'a mut StrippedNode> {
        let mut output = Vec::new();
        if self.children.len() < 1 {
            return vec![self]
        }
        else {
            for child in self.children.iter_mut() {
                output.extend(child.mut_crawl_to_leaves());
            }
        };
        output
    }

    pub fn crawl_children(&self) -> Vec<&StrippedNode> {
        let mut output = Vec::new();
        for child in &self.children {
            output.extend(child.crawl_children());
        }
        output.push(&self);
        output
    }

    pub fn predict_leaves(&self,vector: &Vec<f64>, header: &HashMap<String,usize>,drop_mode: &DropMode, prediction_mode:&PredictionMode) -> Vec<&StrippedNode> {

        let mut leaves = vec![];

        if let (Some(feature),Some(split)) = (self.feature_name(),self.split()) {
            if *vector.get(*header.get(feature).unwrap_or(&(vector.len()+1))).unwrap_or(&drop_mode.cmp()) != drop_mode.cmp() {
                if vector[header[feature]] > *split {
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

    pub fn node_sample_encoding(&self,header: &HashMap<String,usize>) -> Vec<bool> {
        let mut encoding = vec![false; header.len()];
        for sample in self.samples() {
            if let Some(sample_index) = header.get(sample.name()) {
                encoding[*sample_index] = true;
            }
        }
        encoding
    }

    pub fn from_json(input:&str) -> Result<StrippedNode,serde_json::Error> {
        // eprintln!("{:?}", input);
        // let v = serde_json::from_str(input)?;
        //
        // let dropout = v["dropout"];
        // let children = v["children"];
        // let feature = v["feature"];
        // let split = v["split"];
        // let features = v["features"];
        // let samples = v["samples"];
        // let prerequisites = v["prerequisites"];
        // let medians = v["medians"];
        // let dispersions = v["dispersions"];
        // let weights = v["weights"];
        // let local_gains = v["local_gains"];
        // let absolute_gains = v["absolute_gains"];
        //
        // let sn = StrippedNode {
        //     dropout,
        //     children,
        //     feature,
        //     split,
        //     features,
        //     samples,
        //     prerequisites,
        //     medians,
        //     dispersions,
        //     weights,
        //     local_gains,
        //     absolute_gains,
        // };
        // eprintln!("{:?}",v);
        // Ok(sn)

        serde_json::from_str(input)
    }

    pub fn from_location(location:&str) -> Result<Vec<StrippedNode>,Box<Error>> {
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
            nodes.push(StrippedNode::from_file(&tl)?);
        };
        Ok(nodes)
    }

    pub fn from_file(location:&str) -> Result<StrippedNode,Box<Error>> {
        let mut json_file = File::open(location)?;
        let mut json_string = String::new();
        json_file.read_to_string(&mut json_string)?;
        Ok(StrippedNode::from_json(&json_string)?)
    }

}

#[cfg(test)]
mod node_testing {

    use super::*;
    // use ndarray_linalg;
    use crate::feature_thread_pool::FeatureThreadPool;
    use crate::split_thread_pool::SplitThreadPool;

    fn blank_parameter() -> Arc<Parameters> {
        let mut parameters = Parameters::empty();

        parameters.dropout = DropMode::Zeros;

        Arc::new(parameters)
    }

    fn blank_stripped() -> StrippedNode {
        StrippedNode {
            dropout: DropMode::No,

            children: vec![],

            feature: None,
            split: None,

            features: vec![],
            samples: vec![],

            prerequisites: vec![],

            medians: vec![],
            dispersions: vec![],
            weights: vec![],

            local_gains: None,
            absolute_gains: None,
        }
    }

    fn blank_node() -> Node {
        let input_counts = &vec![];
        let output_counts = &vec![];
        let input_features = &vec![][..];
        let output_features = &vec![][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        let pool = SplitThreadPool::new(1);
        Node::feature_root(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option,pool)
    }

    fn trivial_node() -> Node {
        let input_counts = &vec![vec![]];
        let output_counts = &vec![vec![]];
        let input_features = &vec![Feature::q(&1)][..];
        let output_features = &vec![Feature::q(&2)][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        let pool = SplitThreadPool::new(1);
        Node::feature_root(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option,pool)
    }

    fn simple_node() -> Node {
        let input_counts = &vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let output_counts = &vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let input_features = &vec![Feature::q(&1)][..];
        let output_features = &vec![Feature::q(&2)][..];
        let samples = &Sample::vec(vec![0,1,2,3,4,5,6,7])[..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        let pool = SplitThreadPool::new(1);
        Node::feature_root(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option,pool)
    }

    #[test]
    fn node_test_blank() {
        let mut root = blank_node();
        root.mads();
        root.medians();
    }

    #[test]
    fn node_test_trivial() {
        let mut root = trivial_node();
        root.mads();
        root.medians();
    }

    #[test]
    fn node_test_simple() {

        let mut root = simple_node();

        root.feature_parallel_derive(None);
        //
        // println!("{:?}", root.output_table.sort_by_feature("two"));
        // println!("{:?}", root.clone().output_table.parallel_dispersion(&root.output_table.sort_by_feature("two").0,&root.output_table.sort_by_feature("two").1,FeatureThreadPool::new(1)));

        println!("sample_order:{:?}",root.children[0].output_table.full_values());

        // assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
        // assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"6".to_string(),"7".to_string()]);

        // assert_eq!(root.children[0].samples(),&vec!["1".to_string(),"4".to_string(),"5".to_string()]);
        // assert_eq!(root.children[1].samples(),&vec!["0".to_string(),"3".to_string(),"6".to_string(),"7".to_string()]);

        assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"4".to_string(),"5".to_string(),"3".to_string()]);
        assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"6".to_string(),"7".to_string()]);

    }

    #[test]
    fn node_test_stripped_file() {
        let mut root = StrippedNode::from_file("../testing/iris_forest/run.50.compact").unwrap();
    }

    #[test]
    fn node_test_stripped_location() {
        let mut roots = StrippedNode::from_location("../testing/iris_forest/").unwrap();
    }

    #[test]
    fn node_test_json() {
        let n = blank_stripped();
        let ns = n.to_string();
        let r = StrippedNode::from_json(&ns).unwrap();
    }

}
