use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::io::Error;

use std::sync::mpsc;
use std::fs::OpenOptions;
use std::iter::repeat;
use std::collections::HashMap;
use serde_json;
use rayon::join;


extern crate rand;


use crate::node::{Node,StrippedNode};
use crate::{Feature,Sample};
use crate::io::DispersionMode;
use crate::io::DropMode;
use crate::io::PredictionMode;
use crate::io::Parameters;

#[derive(Clone,Serialize,Deserialize,Debug)]
pub struct Tree {
    pub prototype: Option<Node>,
    pub root: StrippedNode,
    dropout: DropMode,
    weights: Option<Vec<f64>>,
    size_limit: usize,
    depth_limit: usize,
    pub report_address: String,
}

impl<'a> Tree {

    pub fn prototype_tree(inputs:&Vec<Vec<f64>>,outputs:&Vec<Vec<f64>>,input_features:&[Feature],output_features:&[Feature],samples:&[Sample], feature_weight_option: Option<Vec<f64>>, parameters: Arc<Parameters> ,report_address: String) -> Tree {
        let processor_limit = parameters.processor_limit;
        let prototype = Node::prototype(inputs,outputs,input_features,output_features,samples, parameters.clone() , feature_weight_option.clone());
        let root = prototype.strip_clone();
        let weights = feature_weight_option;

        Tree {
            prototype: Some(prototype),
            root: root,
            dropout: parameters.dropout,
            weights: weights,
            size_limit: parameters.leaf_size_cutoff,
            depth_limit: parameters.depth_cutoff,
            report_address: report_address
        }
    }

    pub fn reload(location: &str, size_limit: usize, depth_limit: usize ) -> Result<Tree,Error> {

        println!("Reloading!");

        let mut json_file = File::open(location)?;
        let mut json_string = String::new();
        json_file.read_to_string(&mut json_string)?;

        // println!("{}",json_string);

        let root = StrippedNode::from_json(&json_string)?;

        println!("Deserialized root wrapper");

        println!("Finished recursive unwrapping and obtained a Node tree");

        Ok(Tree {
            dropout: root.dropout(),
            prototype: None,
            root: root,
            weights: None,
            size_limit: size_limit,
            depth_limit: depth_limit,
            report_address: location.to_string(),
        })

    }


    pub fn grow_branches(&mut self,parameters:Arc<Parameters>) {
        self.root = grow_branches(self.prototype.clone().expect("Tree without prototype"), parameters,0);
        self.root.root_absolute_gains();
    }

    pub fn derive_specified(&self,samples:&Vec<usize>,input_features:&Vec<usize>,output_features:&Vec<usize>,iteration: usize) -> Tree {

        let mut new_prototype = self.prototype.as_ref().expect("Tree without prototype").derive_specified(samples,input_features,output_features,"RT");

        new_prototype.prototype = true;

        let new_root = new_prototype.strip_clone();

        let mut address: Vec<String> = self.report_address.split('.').map(|x| x.to_string()).collect();
        *address.last_mut().unwrap() = iteration.to_string();
        let mut address_string: String = address.iter().zip(repeat(".")).fold(String::new(),|mut acc,x| {acc.push_str(x.0); acc.push_str(x.1); acc});
        address_string.pop();

        Tree{
            prototype: Some(new_prototype),
            root: new_root,
            dropout: self.dropout,
            weights: self.weights.clone(),
            size_limit: self.size_limit,
            depth_limit: self.depth_limit,
            report_address: address_string,
        }

    }

    pub fn set_scoring_weights(&mut self, weights: Vec<f64>) {
        self.root.set_weights(weights);
    }

    pub fn nodes(&self) -> Vec<&StrippedNode> {
        self.root.crawl_children()
    }

    pub fn root(&self) -> &StrippedNode {
        &self.root
    }

    pub fn dropout(&self) -> DropMode {
        self.dropout
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.root.dimensions()
    }

    // pub fn mut_crawl_to_leaves_target(&'a self, target: &'a mut Node) -> Vec<&'a mut Node> {
    //     let mut output = Vec::new();
    //     if target.children.len() < 1 {
    //         return vec![target]
    //     }
    //     else {
    //         for child in target.children.iter_mut() {
    //             output.extend(self.mut_crawl_to_leaves(child));
    //         }
    //     };
    //     output
    // }

    pub fn crawl_to_leaves(&self) -> Vec<& StrippedNode> {
        self.root.crawl_leaves()
    }

    pub fn crawl_nodes(&self) -> Vec<& StrippedNode> {
        self.root.crawl_children()
    }

    pub fn input_feature_names(&self) -> Vec<String> {
        self.prototype.as_ref().expect("Missing prototype").input_feature_names()
    }

    pub fn output_feature_names(&self) -> Vec<String> {
        self.prototype.as_ref().expect("Missing prototype").output_feature_names()
    }

    pub fn serialize_compact(&self) -> Result<(),Error> {
        println!("Serializing to:");
        println!("{}",self.report_address);
        let mut tree_dump = OpenOptions::new().write(true).truncate(true).create(true).open(&self.report_address)?;
        tree_dump.write(self.root.clone().to_string().as_bytes())?;
        tree_dump.write(b"\n")?;
        Ok(())
    }

    pub fn serialize_ultra_compact(&self) -> Result<(),Error> {
        println!("Serializing to:");
        println!("{}",self.report_address);
        let mut tree_dump = OpenOptions::new().write(true).truncate(true).create(true).open(&self.report_address)?;
        tree_dump.write(self.root.clone().compact().to_string().as_bytes())?;
        tree_dump.write(b"\n")?;
        Ok(())
    }


    pub fn serialize_compact_consume(self) -> Result<(),Error> {
        println!("Serializing to:");
        println!("{}",self.report_address);
        let mut tree_dump = OpenOptions::new().write(true).truncate(true).create(true).open(&self.report_address)?;
        tree_dump.write(self.root.to_string().as_bytes())?;
        tree_dump.write(b"\n")?;
        Ok(())
    }

    pub fn serialize(self) -> Result<(),Error> {

        println!("Serializing to:");
        println!("{}",self.report_address);

        let mut tree_dump = OpenOptions::new().write(true).truncate(true).create(true).open(self.report_address)?;
        tree_dump.write(self.root.to_string().as_bytes())?;
        tree_dump.write(b"\n")?;

        Ok(())
    }

    pub fn serialize_clone(&self) -> Result<(),Error> {
        self.clone().serialize()
    }

    // pub fn predict_leaves(&self,vector:&Vec<f64>, header: &HashMap<String,usize>, prediction_mode:&PredictionMode, drop_mode: &DropMode) -> Vec<&StrippedNode> {
    //     self.root.predict_leaves(vector,header,drop_mode,prediction_mode)
    // }




}

pub fn grow_branches(mut target:Node, parameters: Arc<Parameters>,level:usize) -> StrippedNode {
    if target.samples().len() > parameters.leaf_size_cutoff && level < parameters.depth_cutoff {
        // if let Some(mut cs) = target.sub_split_node(parameters.sample_subsample,parameters.input_features,parameters.output_features) {
        if let Some(mut cs) = target.braid_split_node(parameters.sample_subsample,parameters.input_features,parameters.output_features) {
            let mut stripped = target.strip_consume();
            let c1o = cs.pop();
            let c2o = cs.pop();
            if let (Some(mut c1),Some(mut c2)) = (c1o,c2o) {
                let (c1s,c2s) = join(
                    || {
                        grow_branches(c1, parameters.clone(), level + 1)
                    },
                    || {
                        grow_branches(c2, parameters.clone(), level + 1)
                    }
                );
                stripped.set_children(vec![c1s,c2s]);
            }
            stripped
        }
        else {target.strip_consume()}
    }
    else {target.strip_consume()}
}


// #[cfg(test)]
// mod tree_tests {
//     fn test_reconstitution()
// }

// pub fn test_splits(&mut self) {
//     self.root.derive_children();
//     for child in self.root.children.iter_mut() {
//         child.derive_children();
//         for second_children in child.children.iter_mut() {
//             if second_children.internal_report().len() > 20 {
//                 second_children.derive_children();
//             }
//         }
//     }
// }
//
// pub fn test_parallel_splits(&mut self) {
//     self.root.feature_parallel_derive();
//     for child in self.root.children.iter_mut() {
//         child.feature_parallel_derive();
//     }
// }


//
// pub fn node_predict_leaves<'a>(node: &'a Node, vector: &Vec<f64>, header: &HashMap<String,usize>, prediction_mode: &PredictionMode) -> Vec<&'a Node> {
//
//     let mut leaves: Vec<&Node> = Vec::new();
//
//     if let (&Some(ref feature),&Some(ref split)) = (&node.feature,&node.split) {
//         if header.contains_key(feature) {
//             if vector[header[feature]] > split.clone() {
//                 leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode));
//             }
//             else {
//                 leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode));
//             }
//         }
//         else {
//             match prediction_mode {
//                 &PredictionMode::Branch => {
//                     leaves.append(&mut node_predict_leaves(&node.children[0], vector, header, prediction_mode));
//                     leaves.append(&mut node_predict_leaves(&node.children[1], vector, header, prediction_mode));
//                 },
//                 &PredictionMode::Truncate => {
//                     leaves.push(&node)
//                 },
//                 &PredictionMode::Abort => {},
//                 &PredictionMode::Auto => {
//                     leaves.append(&mut node_predict_leaves(&node, vector, header, &PredictionMode::Branch));
//                 }
//             }
//         }
//     }
//     else {
//         leaves.push(&node);
//     }
//
//     return leaves
//
// }

// pub fn sum_leaves(leaves: Vec<&Node>) -> (Vec<f64>,Vec<f64>,Vec<usize>) {
//
//     median
//
//     for
// }
//
// pub fn interval_stack(intervals: Vec<(&f64,&f64,&f64)>) -> Vec<(f64,f64,f64)> {
//     let mut aggregate_intervals: Vec<f64> = intervals.iter().fold(Vec::with_capacity(intervals.len()*2), |mut acc,x| {acc.push(*x.0); acc.push(*x.1); acc});
//     aggregate_intervals.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
//     let mut aggregate_scores = vec![0.;aggregate_intervals.len()-1];
//     for (s_start,s_end,score) in intervals {
//         for (i,(w_start,w_end)) in aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).enumerate() {
//             if (*w_start >= *s_start) && (*w_end <= *s_end) {
//                 aggregate_scores[i] += score;
//             }
//             else {
//                 aggregate_scores[i] -= score;
//             }
//         }
//     }
//     let scored = aggregate_intervals.iter().zip(aggregate_intervals.iter().skip(1)).zip(aggregate_scores.into_iter()).map(|((begin,end),score)| (*begin,*end,score)).collect();
//     scored
// }


// pub fn grow_recursively(&mut self, target: ) {
//     if target.upgrade().unwrap().internal_report().len() < self.size_limit {
//         target.parallel_derive();
//         for child in target.children.iter_mut() {
//             self.grow_recursively(child);
//         }
//     }
// }
//
//
// pub fn crawl_leaves<'a>(&'a mut self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
//     let mut output = Vec::new();
//     if target.children.len() < 1 {
//         return vec![target]
//     }
//     else {
//         for child in target.children.iter_mut() {
//             output.extend(self.crawl_to_leaves(child));
//             output.push(&mut target);
//         }
//     };
//     output
// }
//
// pub fn weigh_leaves(&mut self) {
//     let root_dispersions = self.root.dispersions;
//     for leaf in self.crawl_leaves(&mut self.root) {
//
//         let leaf_weights = Vec::with_capacity(root_dispersions.len());
//
//         for (rv,lv) in leaf.dispersions.iter().zip(root_dispersions.iter()) {
//             if *lv != 0. && *rv != 0. {
//                 leaf_weights.push(rv)
//             }
//         }
//     }
// }


// impl<'a, U:Clone + std::cmp::Eq + std::hash::Hash + Debug,T:Clone + std::cmp::Eq + std::hash::Hash + Debug> LeafCrawler<'a, U, T> {
//
//     pub fn new(target:&'a mut Node<U,T>) -> LeafCrawler<'a,U,T> {
//         LeafCrawler{root: target}
//     }
//
//     pub fn crawl_leaves(&'a self, target: &'a mut Node<U,T>) -> Vec<&'a mut Node<U,T>> {
//         let mut output = Vec::new();
//         if target.children.len() < 1 {
//             return vec![target]
//         }
//         else {
//             for child in target.children.iter_mut() {
//                 output.extend(self.crawl_leaves(child));
//                 // output.push(&'a mut target);
//             }
//         };
//         output
//     }
//
// }
//
// pub struct LeafCrawler<'a, U:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug,T:'a + Clone + std::cmp::Eq + std::hash::Hash + Debug> {
//     root: &'a mut Node<U,T>,
// }
