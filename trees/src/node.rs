
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
use crate::argsort;
use crate::jaccard;

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;

use rayon::prelude::*;

// Prefer braid thickness to to be odd to make consensus braids work well
// const BRAID_THICKNESS: usize = 3;

#[derive(Clone,Serialize,Deserialize,Debug)]
pub struct Node {

    pub prototype: bool,

    input_table: RankTable,
    output_table: RankTable,
    dropout: DropMode,

    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    pub parent_id: String,
    pub id: String,
    pub depth: usize,
    pub children: Vec<Node>,

    split: Option<Split>,

    prerequisites: Vec<Prerequisite>,
    braids: Vec<Braid>,
    braid_thickness: usize,

    pub medians: Vec<f64>,
    pub additive: Vec<f64>,
    pub feature_weights: Vec<f64>,
    pub dispersions: Vec<f64>,
    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>

}


impl Node {

    pub fn prototype<'a>(input_counts:&Vec<Vec<f64>>,output_counts:&Vec<Vec<f64>>,input_features:&'a[Feature],output_features:&'a[Feature],samples:&'a[Sample], parameters: Arc<Parameters> , feature_weight_option: Option<Vec<f64>>) -> Node {

        let input_table = RankTable::new(input_counts,parameters.clone());
        let output_table = RankTable::new(output_counts,parameters.clone());
        let feature_weights = feature_weight_option.unwrap_or(vec![1.;output_features.len()]);
        let medians = output_table.medians();
        let additive = vec![0.;medians.len()];
        let dispersions = output_table.dispersions();
        let local_gains = vec![0.;dispersions.len()];

        let new_node = Node {

            prototype: true,

            input_table: input_table,
            output_table: output_table,
            dropout: parameters.dropout,

            input_features: input_features.to_owned(),
            output_features: output_features.to_owned(),
            samples: samples.to_owned(),

            id: "RT".to_string(),
            parent_id: "RT".to_string(),
            depth: 0,
            children: Vec::new(),

            split: None,

            prerequisites: vec![],
            braids: vec![],
            braid_thickness: parameters.braid_thickness,

            medians: medians,
            additive: additive,
            feature_weights: feature_weights,
            dispersions: dispersions,
            local_gains: Some(local_gains),
            absolute_gains: None
        };

        new_node

    }

    pub fn split_node(&mut self) -> Option<Vec<Node>> {
        if let Some(split) = self.rayon_best_split() {
            self.split = Some(split);
            // eprintln!("Deriving split:{:?}",self.split);
            Some(self.derive_complete_by_split(self.split.as_ref().unwrap(), None))
        }
        else { None }
    }

    pub fn sub_split_node(&mut self,samples:usize,input_features:usize,output_features:usize) -> Option<Vec<Node>> {
        let mut compact = self.subsample(samples,input_features,output_features);
        if let Some(split) = compact.rayon_best_split() {
            self.split = Some(split);
            // eprintln!("Deriving split:{:?}",self.split);
            Some(self.derive_complete_by_split(self.split.as_ref().unwrap(), None))
        }
        else { None }
    }

    pub fn braid_split_node(&mut self,samples:usize,input_features:usize,output_features:usize) -> Option<Vec<Node>> {

        if !self.prototype { panic!("Attempted to take a braid off an incomplete node") };

        let thickness = self.braid_thickness;

        // Here we can either resample the compute node multiple times or simply take the top 4 features. I am leaning towards the latter as the approach

        // This block represents complete resampling:

        // let mut features = Vec::with_capacity(thickness);
        //
        // for i in 0..thickness {
        //     let mut compact = self.subsample(samples,input_features,output_features);
        //     if let Some(Split{feature,..}) = compact.rayon_best_split() {
        //         features.push(feature);
        //     }
        // }
        //
        // This block represents one subsampling and taking the top 4 features:

        // let mut features: Vec<Feature> =
        //     self.subsample(samples,input_features,output_features)
        //     .rayon_best_n_split(thickness)
        //     .into_iter()
        //     .map(|s| s.feature)
        //     .collect();

        // This block represents one subsampling and taking the top feature, and n of its most correlated features:

        let mut splits: Vec<Split> = self.subsample(samples,input_features,output_features).rayon_best_plus_n_splits(thickness);

        let mut features: Vec<Feature> = splits.iter().map(|s| s.feature.clone()).collect();

        // This block represents resampling the input features and samples, but not the output features:

        // let mut features = Vec::with_capacity(thickness);
        //
        // let sisters = self.subsample_n_sisters(samples,input_features,output_features,thickness);
        //
        // for sister in sisters {
        //     if let Some(Split{feature,..}) = sister.rayon_best_split() {
        //         features.push(feature);
        //     }
        // }

        // One of the three above alternatives has to be picked.

        let samples = self.samples.clone();

        let rvs: Vec<_> = features.iter().map(|f| self.input_table.rv_fetch(f.index).clone()).collect();

        let braid = Braid::from_splits(features, samples, &rvs, &splits);

        self.derive_complete_by_braid(braid)
    }

    pub fn rayon_best_split(&self) -> Option<Split> {

        let splits: Vec<Split> =
            (0..self.input_features().len())
            .into_par_iter()
            .flat_map(|i| self.feature_index_split(i))
            .collect();
        let dispersions: Vec<f64> = splits.iter().map(|s| s.dispersion).collect();
        Some(splits[argmin(&dispersions)?.0].clone())

    }

    pub fn rayon_best_n_split(&self,n:usize) -> Vec<Split> {

        let splits: Vec<Split> =
            (0..self.input_features().len())
            .into_par_iter()
            .flat_map(|i| self.feature_index_split(i))
            .collect();
        let mut n_dispersions: Vec<(usize,f64)> = splits.iter().map(|s| s.dispersion).enumerate().collect();
        n_dispersions.sort_by(|a,b| a.1.partial_cmp(&b.1).expect("Nan dispersion"));
        let top_n = n_dispersions.iter().take(n).map(|(i,d)| splits[*i].clone()).collect();
        top_n

    }

    pub fn rayon_best_plus_n_splits(&self,n:usize) -> Vec<Split> {

        let splits: Vec<Split> =
            (0..self.input_features().len())
            .into_par_iter()
            .flat_map(|i| self.feature_index_split(i))
            .collect();

        let dispersions: Vec<f64> = splits.iter().map(|s| s.dispersion).collect();
        let best_split_index = argmin(&dispersions).unwrap().0;

        let split_encodings: Vec<Vec<bool>> = splits.par_iter().map(|s| self.split_encoding(s)).collect();
        let best_encoding = &split_encodings[best_split_index];
        let similarities = split_encodings.par_iter().map(|e| jaccard(best_encoding,e)).collect();
        let ranked_similarties = argsort(&similarities);
        let n_most_similar_indices: Vec<usize> = ranked_similarties.into_iter().enumerate().filter(|(index,(rank,value))| rank < &n).map(|(i,_)| i).collect();
        n_most_similar_indices.into_iter().map(|i| splits[i].clone()).collect()

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


    pub fn derive_complete_by_braid(&mut self,mut braid:Braid) -> Option<Vec<Node>> {
        // let (draw_order,drop_set) = braid.draw_order();
        // let dispersions = self.output_table.order_dispersions(&draw_order,&drop_set,&self.feature_weights)?;
        // let (split_index,minimum_dispersion) = argmin(&dispersions[1..])?;

        let left_indices = braid.left_indices();

        let right_indices = braid.right_indices();

        if left_indices.len() < 2 || right_indices.len() < 2 {
            return None
        }

        braid.compound_split = Some(0.);

        self.braids.push(braid.clone());

        let mut left_child_id = self.id.clone();
        let mut right_child_id = self.id.clone();
        left_child_id.push_str(&format!("!F{:?}:L",braid.features));
        right_child_id.push_str(&format!("!F{:?}:R",braid.features));

        let input_features: Vec<usize> = (0..self.input_features().len()).collect();
        let output_features: Vec<usize> = (0..self.output_features().len()).collect();

        let mut new_braids = self.braids.clone();

        let mut left_child = self.derive_specified(&left_indices, &input_features, &output_features, None, Some(new_braids.clone()), &left_child_id);
        let mut right_child = self.derive_specified(&right_indices, &input_features, &output_features, None, Some(new_braids), &right_child_id);

        left_child.prototype = true;
        right_child.prototype = true;

        left_child.braids.last_mut().unwrap().orientation = Some(false);
        right_child.braids.last_mut().unwrap().orientation = Some(true);

        Some(vec![left_child,right_child])
    }

    //
    // pub fn subsample_by_braids(&mut self,mut braids:&[Braid]) -> Option<Vec<Node>> {
    //     // let (draw_order,drop_set) = braid.draw_order();
    //     // let dispersions = self.output_table.order_dispersions(&draw_order,&drop_set,&self.feature_weights)?;
    //     // let (split_index,minimum_dispersion) = argmin(&dispersions[1..])?;
    //
    //     let left_indices = braid.left_indices();
    //
    //     let right_indices = braid.right_indices();
    //
    //     if left_indices.len() < 2 || right_indices.len() < 2 {
    //         return None
    //     }
    //
    //     braid.compound_split = Some(0.);
    //
    //     self.braids.push(braid.clone());
    //
    //     let mut left_child_id = self.id.clone();
    //     let mut right_child_id = self.id.clone();
    //     left_child_id.push_str(&format!("!F{:?}:L",braid.features));
    //     right_child_id.push_str(&format!("!F{:?}:R",braid.features));
    //
    //     let input_features: Vec<usize> = (0..self.input_features().len()).collect();
    //     let output_features: Vec<usize> = (0..self.output_features().len()).collect();
    //
    //     let mut new_braids = self.braids.clone();
    //
    //     let mut left_child = self.derive_specified(&left_indices, &input_features, &output_features, None, Some(new_braids.clone()), &left_child_id);
    //     let mut right_child = self.derive_specified(&right_indices, &input_features, &output_features, None, Some(new_braids), &right_child_id);
    //
    //     left_child.prototype = true;
    //     right_child.prototype = true;
    //
    //     Some(vec![left_child,right_child])
    // }

    pub fn derive_complete_by_split(&self,split:&Split,prototype:Option<&Node>) -> Vec<Node> {

        let mut left_prerequisites = self.prerequisites.clone();
        let mut right_prerequisites = self.prerequisites.clone();
        left_prerequisites.push(split.left());
        right_prerequisites.push(split.right());
        let mut left_child_id = self.id.clone();
        let mut right_child_id = self.id.clone();
        left_child_id.push_str(&format!("!F{:?}:S{:?}L",split.feature,split.value));
        right_child_id.push_str(&format!("!F{:?}:S{:?}R",split.feature,split.value));
        vec![prototype.unwrap_or(self).derive_complete_by_prerequisites(&left_prerequisites,&left_child_id),prototype.unwrap_or(self).derive_complete_by_prerequisites(&right_prerequisites,&right_child_id)]
    }


    pub fn derive_complete_by_prerequisites(&self, prerequisites: &[Prerequisite], new_id:&str) -> Node {

        let input_features: Vec<usize> = (0..self.input_features().len()).collect();
        let output_features: Vec<usize> = (0..self.output_features().len()).collect();

        let samples: Vec<usize> = self.indices_given_prerequisites(&prerequisites);

        let mut child = self.derive_specified(&samples,&input_features,&output_features, Some(prerequisites.to_owned()),None,new_id);

        child.prototype = true;

        child
    }

    pub fn derive_specified(&self, samples: &[usize], input_features: &[usize], output_features: &[usize], prerequisite_opt: Option<Vec<Prerequisite>>, braid_opt: Option<Vec<Braid>>, new_id: &str) -> Node {

        // This is the base derivation of a node from self.
        // Other derivations produce a node specification, then call this method

        // Features and samples in the child node are to be specified using LOCAL INDEXING.

        // DESIGN DECISION: For the moment I am allowing derivation from non-prototype nodes.
        // I may get rid of this later. As a result the global indices of a feature may not match
        // the indices provided above.

        // Derived nodes are guaranteed have matching features/tables
        // Derived nodes are not guaranteed to be a prototype, set this elsewhere

        let mut new_input_table = self.input_table.derive_specified(&input_features,samples);
        let mut new_output_table = self.output_table.derive_specified(&output_features,samples);

        // eprintln!("spec_derivation debug, tables done");

        let new_input_features = input_features.iter().map(|i| self.input_features[*i].clone()).collect();
        let new_output_features = output_features.iter().map(|i| self.output_features[*i].clone()).collect();
        let new_samples: Vec<Sample> = samples.iter().map(|i| self.samples[*i].clone()).collect();

        let mut new_prerequisites =
            if let Some(prerequisites) = prerequisite_opt {
                prerequisites.to_owned()
            }
            else {self.prerequisites.clone()};

        let mut new_braids = braid_opt.unwrap_or(self.braids.clone());

        let medians = new_output_table.medians();

        let additive = self.medians.iter().zip(medians.iter()).map(|(pm,cm)| cm - pm).collect();

        let dispersions = new_output_table.dispersions();
        let feature_weights = output_features.iter().map(|y| self.feature_weights[*y]).collect();

        let local_gains = Some(self.dispersions().iter().zip(dispersions.iter()).map(|(p,c)| (p/(self.samples().len() as f64)) - (c/((new_samples.len() + 1) as f64))).collect());

        let child = Node {

            prototype: false,

            input_table: new_input_table,
            output_table: new_output_table,
            dropout: self.dropout,

            input_features: new_input_features,
            output_features: new_output_features,
            samples: new_samples,

            parent_id: self.id.clone(),
            id: new_id.to_string(),
            depth: self.depth + 1,
            children: Vec::new(),

            split: None,

            prerequisites: new_prerequisites,
            braids: new_braids,
            braid_thickness: self.braid_thickness,

            medians: medians,
            additive: additive,
            feature_weights: feature_weights,
            dispersions: dispersions,
            local_gains: local_gains,
            absolute_gains: None
        };

        child
    }

    pub fn draw_random_features(&self,input_features: usize, output_features: usize) -> (Vec<(usize,Feature)>,Vec<(usize,Feature)>) {

        let mut rng = rand::thread_rng();

        let input_fvec = self.input_features();
        let output_fvec = self.output_features();

        let drawn_input: Vec<(usize,Feature)> = (0..input_features).map(|_| rng.gen_range(0,self.input_table.dimensions.0)).map(|i| (i,input_fvec[i].clone())).collect();
        let drawn_output: Vec<(usize,Feature)> = (0..output_features).map(|_| rng.gen_range(0,self.output_table.dimensions.0)).map(|i| (i,output_fvec[i].clone())).collect();

        (drawn_input,drawn_output)
    }

    pub fn draw_random_samples(&self,samples:usize) -> Vec<(usize,&Sample)> {
        let mut rng = rand::thread_rng();
        (0..samples).map(|_| rng.gen_range(0,self.input_table.dimensions.1)).map(|i| (i,&self.samples()[i])).collect()
    }

    pub fn subsample(&self,samples:usize,input_features:usize,output_features:usize) -> Node {
        let (input_features,output_features) = self.draw_random_features(input_features, output_features);
        let samples = self.draw_random_samples(samples);
        let ifi: Vec<usize> = input_features.into_iter().map(|(i,f)| i).collect();
        let ofi: Vec<usize> = output_features.into_iter().map(|(i,f)| i).collect();
        let si: Vec<usize> = samples.into_iter().map(|(i,s)| i).collect();
        let id = format!("{}!SSS",self.id,).to_string();
        self.derive_specified(&si,&ifi,&ofi,None,None,&id)
    }

    pub fn subsample_n_sisters(&self,samples:usize,input_features:usize,output_features:usize,n:usize) -> Vec<Node> {

        let mut sisters: Vec<Node> = Vec::with_capacity(n);

        let mut rng = rand::thread_rng();

        let output_fvec = self.output_features();
        let output_features: Vec<(usize,Feature)> = (0..output_features).map(|_| rng.gen_range(0,self.output_table.dimensions.0)).map(|i| (i,output_fvec[i].clone())).collect();

        let input_fvec = self.input_features();

        for sister in 0..n {
            let input_features: Vec<(usize,Feature)> = (0..input_features).map(|_| rng.gen_range(0,self.input_table.dimensions.0)).map(|i| (i,input_fvec[i].clone())).collect();
            let samples = self.draw_random_samples(samples);
            let ifi: Vec<usize> = input_features.into_iter().map(|(i,f)| i).collect();
            let ofi: Vec<usize> = output_features.iter().map(|(i,f)| *i).collect();
            let si: Vec<usize> = samples.into_iter().map(|(i,s)| i).collect();
            let id = format!("{}!SSS",self.id,).to_string();
            sisters.push(self.derive_specified(&si,&ifi,&ofi,None,None,&id));
        }

        sisters
    }

    pub fn report(&self,verbose:bool) {
        println!("Node reporting:");
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
            report_string.push_str(&format!("S:{:?}\n",self.split));
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

    pub fn set_dispersion_mode(&mut self, dispersion_mode : DispersionMode) {
        self.output_table.set_dispersion_mode(dispersion_mode);
    }

    pub fn dispersion_mode(&self) -> DispersionMode {
        self.output_table.dispersion_mode()
    }

    pub fn strip_consume(self) -> StrippedNode {

        let features = self.output_features().iter().cloned().collect();
        let samples = self.samples().iter().cloned().collect();

        let mut stripped_children = Vec::new();

        for child in self.children.into_iter().rev() {
            stripped_children.push(child.strip_consume())
        }


        StrippedNode {
            dropout: self.dropout,

            children: stripped_children,

            split: self.split,

            features: features,
            samples: samples,

            prerequisites: self.prerequisites,
            braids: self.braids,

            depth: self.depth,

            medians: self.medians,
            additive: self.additive,
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

            split: self.split.clone(),

            features: self.output_features().iter().cloned().collect(),
            samples: self.samples().iter().cloned().collect(),

            prerequisites: self.prerequisites.clone(),
            braids: self.braids.clone(),

            depth: self.depth,

            medians: self.medians.clone(),
            additive: self.additive.clone(),
            dispersions: self.dispersions.clone(),
            weights: self.feature_weights.clone(),

            local_gains: self.local_gains.clone(),
            absolute_gains: self.absolute_gains.clone(),
        }

    }

    pub fn set_children(&mut self, children: Vec<Node>) {
        self.children = children;
    }

    pub fn output_rank_table(&self) -> &RankTable {
        &self.output_table
    }

    pub fn input_rank_table(&self) -> &RankTable {
        &self.input_table
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    pub fn sample_names(&self) -> Vec<String> {
        self.samples().iter().map(|s| s.name().to_owned()).collect()
    }

    pub fn input_features(&self) -> &[Feature] {
        &self.input_features
    }

    pub fn input_feature_names(&self) -> Vec<String> {
        self.input_features().iter().map(|f| f.name().to_owned()).collect()
    }

    pub fn output_features(&self) -> &[Feature] {
        &self.output_features
    }

    pub fn output_feature_names(&self) -> Vec<String> {
        self.output_features().iter().map(|f| f.name().to_owned()).collect()
    }

    // In this function we iterate through prerequisites and select only ones that either were dropped in a given feature
    // or fulfil a prerequisite

    pub fn indices_given_prerequisites(&self,prerequisites:&[Prerequisite]) -> Vec<usize> {

        // First we create a default-true mask.

        let mut mask = vec![true;self.samples.len()];

        // We then cycle through each prerequisite and figure out the local index of the feature it encodes.

        for prerequisite in prerequisites {
            if let Some(feature_index) = self.input_features().iter().position(|x| x == &prerequisite.feature) {

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


        mask.into_iter().zip(0..self.samples.len()).filter(|(m,s)| *m).map(|(m,s)| s).collect()

    }

    pub fn samples_given_prerequisites(&self,prerequisites:&[Prerequisite]) -> Vec<&Sample> {
        self.indices_given_prerequisites(prerequisites).into_iter().map(|i| &self.samples[i]).collect()
    }

    pub fn split(&self) -> &Option<Split> {
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

        for child in self.children.iter_mut().rev() {
            child.compute_absolute_gains(root_dispersions);
        }
    }

    pub fn root_absolute_gains(&mut self) {
        for child in self.children.iter_mut().rev() {
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

    pub fn split_encoding(&self,split:&Split) -> Vec<bool> {
        let Split{ref feature, ref value,..} = split;
        let feature_index = self.input_features.iter().position(|e| e == feature).unwrap();
        let feature_values = self.input_table.full_feature_values(feature_index);
        feature_values.iter().map(|fv| if fv > value {true} else {false}).collect()
    }

    pub fn sample_encoding(&self,max_sample_option:Option<usize>) -> Vec<bool> {

        let max_sample =
            if self.prototype {self.samples.iter().map(|s| s.index).max().unwrap_or(0)}
            else {max_sample_option.unwrap_or(0)};

        let mut sample_encoding = vec![false; max_sample+1];
        for sample in self.samples() {
            sample_encoding[sample.index] = true;
        }
        return sample_encoding
    }

    pub fn feature_encoding(&self,max_feature_option:Option<usize>) -> Vec<bool> {

        let max_feature =
            if self.prototype {self.output_features().iter().map(|f| f.index).max().unwrap_or(0)}
            else {max_feature_option.unwrap_or(0)};

        let mut feature_encoding = vec![false; max_feature+1];
        for feature in self.output_features() {
            feature_encoding[feature.index] = true;
        }
        return feature_encoding
    }

    pub fn compact(self) -> UltraCompact {

        let feature_encoding = self.feature_encoding(None).into_iter().map(|b| if b {1} else {0}).collect();
        let sample_encoding = self.sample_encoding(None).into_iter().map(|b| if b {1} else {0}).collect();
        let children = self.children.into_iter().rev().map(|c| c.compact()).collect();
        let mut braid = if self.braids.len() == (self.depth+1) {self.braids.last().map(|b| b.clone())} else {None};
        if braid.is_some() {
            braid.as_mut().unwrap().samples = vec![];
        }

        UltraCompact {
            children: children,

            split: self.split,

            prerequisites: self.prerequisites,
            braid: braid,

            features: feature_encoding,
            samples: sample_encoding,
            medians: self.medians,
            additive: self.additive,
            dispersions: self.dispersions,

            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains,
        }
    }

    pub fn nano_compact(self) -> NanoCompact {

        let sample_encoding = self.sample_encoding(None).into_iter().map(|b| if b {1} else {0}).collect();
        let children = self.children.into_iter().rev().map(|c| c.nano_compact()).collect();
        let mut braid = if self.braids.len() == (self.depth+1) {self.braids.last().map(|b| b.clone())} else {None};
        if braid.is_some() {
            braid.as_mut().unwrap().samples = vec![];
        }

        NanoCompact {
            children: children,

            split: self.split,

            prerequisites: self.prerequisites,
            braid: braid,

            samples: sample_encoding,
        }
    }

    // pub fn assert_integrity(&self) {
    //
    //     // Here we check assumptions that must remain true of each node: that each sample in the node fulfills the requirements
    //     // of that node.
    //
    // }

}


impl Node {
    pub fn to_string(self) -> Result<String,serde_json::Error> {
        serde_json::to_string(&self)
    }

    pub fn from_str(input:&str) -> Result<Node,serde_json::Error> {
        serde_json::from_str(input)
    }
}



#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct StrippedNode {

    dropout: DropMode,

    pub children: Vec<StrippedNode>,

    split: Option<Split>,

    prerequisites: Vec<Prerequisite>,
    braids: Vec<Braid>,

    depth: usize,

    features: Vec<Feature>,
    samples: Vec<Sample>,
    medians: Vec<f64>,
    additive: Vec<f64>,
    dispersions: Vec<f64>,
    weights: Vec<f64>,


    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>,
}

impl StrippedNode {

    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn features(&self) -> &[Feature] {
        &self.features[..]
    }

    pub fn feature_names(&self) -> Vec<String> {
        self.features().iter().map(|f| f.name().to_owned()).collect()
    }

    pub fn dimensions(&self) -> (usize,usize) {
        (self.features.len(),self.samples.len())
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples[..]
    }

    pub fn split(&self) -> &Option<Split> {
        &self.split
    }

    pub fn medians(&self) -> &Vec<f64> {
        &self.medians
    }

    pub fn set_children(&mut self, children: Vec<StrippedNode>) {
        self.children = children;
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
            for child in self.children.iter_mut().rev() {
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

    pub fn compute_absolute_gains(&mut self,root_dispersions: &Vec<f64>) {

        let mut absolute_gains = Vec::with_capacity(root_dispersions.len());

        for (nd,od) in self.dispersions.iter().zip(root_dispersions.iter()) {
            absolute_gains.push(od-nd)
        }
        self.absolute_gains = Some(absolute_gains);

        for child in self.children.iter_mut().rev() {
            child.compute_absolute_gains(root_dispersions);
        }
    }

    pub fn root_absolute_gains(&mut self) {
        for child in self.children.iter_mut().rev() {
            child.compute_absolute_gains(&self.dispersions);
        }
    }
    //
    // pub fn predict_leaves(&self,vector: &Vec<f64>, header: &HashMap<String,usize>,drop_mode: &DropMode, prediction_mode:&PredictionMode) -> Vec<&StrippedNode> {
    //
    //     let mut leaves = vec![];
    //
    //     if let Some(Split{feature,value:split, ..}) = self.split() {
    //         if *vector.get(*header.get(feature.name()).unwrap_or(&(vector.len()+1))).unwrap_or(&drop_mode.cmp()) != drop_mode.cmp() {
    //             if vector[header[feature.name()]] > *split {
    //                 leaves.extend(self.children[1].predict_leaves(vector, header, drop_mode, prediction_mode));
    //             }
    //             else {
    //                 leaves.extend(self.children[0].predict_leaves(vector, header, drop_mode, prediction_mode));
    //             }
    //         }
    //         else {
    //             match prediction_mode {
    //                 &PredictionMode::Branch => {
    //                     // println!("Mode is branching");
    //                     leaves.extend(self.children[1].predict_leaves(vector, header, drop_mode, prediction_mode));
    //                     leaves.extend(self.children[0].predict_leaves(vector, header, drop_mode, prediction_mode));
    //                     // println!("{}", leaves.len());
    //                 },
    //                 &PredictionMode::Truncate => {
    //                     leaves.push(&self)
    //                 },
    //                 &PredictionMode::Abort => {},
    //                 &PredictionMode::Auto => {
    //                     leaves.extend(self.predict_leaves(vector, header, drop_mode, prediction_mode));
    //                 }
    //             }
    //         }
    //     }
    //     else {
    //         // println!("Found a leaf");
    //         leaves.push(&self);
    //     }
    //
    //     leaves
    //
    // }


    pub fn from_json(input:&str) -> Result<StrippedNode,serde_json::Error> {
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

    pub fn sample_encoding(&self,max_sample_option:Option<usize>) -> Vec<bool> {

        let max_sample =
            if let Some(max) = max_sample_option {max}
            else {self.samples.iter().map(|s| s.index).max().unwrap_or(0)};

        let mut sample_encoding = vec![false; max_sample+1];
        for sample in self.samples() {
            sample_encoding[sample.index] = true;
        }
        return sample_encoding
    }

    pub fn feature_encoding(&self,max_feature_option:Option<usize>) -> Vec<bool> {

        let max_feature =
            if let Some(max) = max_feature_option {max}
            else {self.features.iter().map(|s| s.index).max().unwrap_or(0)};

        let mut feature_encoding = vec![false; max_feature+1];
        for feature in self.features.iter() {
            feature_encoding[feature.index] = true;
        }
        return feature_encoding
    }

    pub fn compact(self) -> UltraCompact {

        let feature_encoding = self.feature_encoding(None).into_iter().map(|b| if b {1} else {0}).collect();
        let sample_encoding = self.sample_encoding(None).into_iter().map(|b| if b {1} else {0}).collect();
        let children = self.children.into_iter().rev().map(|c| c.compact()).collect();
        let mut braid = if self.braids.len() == (self.depth+1) {self.braids.last().map(|b| b.clone())} else {None};
        if braid.is_some() {
            braid.as_mut().unwrap().samples = vec![];
        }

        UltraCompact {
            children: children,

            split: self.split,

            prerequisites: self.prerequisites,
            braid: braid,

            features: feature_encoding,
            samples: sample_encoding,
            medians: self.medians,
            additive: self.additive,
            dispersions: self.dispersions,

            local_gains: self.local_gains,
            absolute_gains: self.absolute_gains,
        }
    }

    pub fn nano_compact(self) -> NanoCompact {

        let sample_encoding = self.sample_encoding(None).into_iter().map(|b| if b {1} else {0}).collect();
        let children = self.children.into_iter().rev().map(|c| c.nano_compact()).collect();
        let mut braid = if self.braids.len() == (self.depth+1) {self.braids.last().map(|b| b.clone())} else {None};
        if braid.is_some() {
            braid.as_mut().unwrap().samples = vec![];
        }

        NanoCompact {
            children: children,

            split: self.split,

            prerequisites: self.prerequisites,
            braid: braid,

            samples: sample_encoding,
        }
    }


}



#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct UltraCompact {

    pub children: Vec<UltraCompact>,

    split: Option<Split>,

    prerequisites: Vec<Prerequisite>,
    braid: Option<Braid>,

    features: Vec<i8>,
    samples: Vec<i8>,
    medians: Vec<f64>,
    additive: Vec<f64>,
    dispersions: Vec<f64>,

    pub local_gains: Option<Vec<f64>>,
    pub absolute_gains: Option<Vec<f64>>,
}

impl UltraCompact {
    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Serialize,Deserialize,Clone,Debug)]
pub struct NanoCompact {

    pub children: Vec<NanoCompact>,

    split: Option<Split>,

    prerequisites: Vec<Prerequisite>,
    braid: Option<Braid>,

    samples: Vec<i8>,
}

impl NanoCompact {
    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
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

    fn blank_stripped() -> StrippedNode {
        StrippedNode {
            dropout: DropMode::No,

            children: vec![],

            split: None,

            features: vec![],
            samples: vec![],

            prerequisites: vec![],
            braids: vec![],

            depth: 0,

            medians: vec![],
            additive: vec![],
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
        Node::prototype(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option)
    }

    fn trivial_node() -> Node {
        let input_counts = &vec![vec![]];
        let output_counts = &vec![vec![]];
        let input_features = &vec![Feature::q(&1)][..];
        let output_features = &vec![Feature::q(&2)][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        Node::prototype(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option)
    }

    fn simple_node() -> Node {
        let input_counts = &vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let output_counts = &vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let input_features = &vec![Feature::q(&0)][..];
        let output_features = &vec![Feature::q(&0)][..];
        let samples = &Sample::vec(vec![0,1,2,3,4,5,6,7])[..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        Node::prototype(input_counts,output_counts,input_features,output_features,samples,parameters,feature_weight_option)
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
    fn node_test_dispersions() {

        let mut root = simple_node();

        let split0 = root.feature_index_split(0).unwrap();

        println!("{:?}",root.samples());
        println!("{:?}",root.output_table.full_values());
        println!("{:?}",split0);

        // panic!();
    }

    #[test]
    fn node_test_subsample() {

        let mut root = simple_node();


        for i in 0..1000 {
            let sub = root.subsample(8, 2, 2);
            let split_option = sub.rayon_best_split();
            eprintln!("{:?}",sub.strip_clone());
            let (draw_order,drop_set) = sub.input_rank_table().sort_by_feature(0);
            eprintln!("{:?}",(&draw_order,&drop_set));
            eprintln!("{:?}",sub.output_rank_table().order_dispersions(&draw_order,&drop_set,&sub.feature_weights));
            eprintln!("{:?}",split_option.unwrap());
            // if let Some(split) = split_option {
            //     root.clone().derive_complete_by_split(&split,None);
            // }
        }

    }


    #[test]
    fn node_test_split() {

        let mut root = simple_node();

        let split = root.rayon_best_split().unwrap();

        println!("{:?}",split);

        assert_eq!(split.dispersion,2822.265625);
        assert_eq!(split.value, 5.);
    }

    #[test]
    fn node_test_simple() {

        let mut root = simple_node();

        println!("Created node");
        // root.split_node();
        let children = root.braid_split_node(8,1,1);
        root.children = children.unwrap();

        eprintln!("{:?}",root.braids[0]);

        assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
        assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"6".to_string(),"7".to_string()]);

        // assert_eq!(root.children[0].samples(),&vec!["1".to_string(),"4".to_string(),"5".to_string()]);
        // assert_eq!(root.children[1].samples(),&vec!["0".to_string(),"3".to_string(),"6".to_string(),"7".to_string()]);


        // assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"2".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
        // assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"2".to_string(),"6".to_string(),"7".to_string()]);
        //
        // assert_eq!(&root.children[0].output_table.full_values(),&vec![vec![-3.,0.,5.,-2.,-1.]]);
        // assert_eq!(&root.children[1].output_table.full_values(),&vec![vec![10.,0.,15.,20.]]);

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
