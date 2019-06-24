use std::fs::File;
use std::io::Write;
use std::io::Error;
use std::io::BufRead;
use std::io;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use std::fs::OpenOptions;


extern crate rand;
use rand::seq;

use crate::tree::Tree;
use crate::tree::PredictiveTree;
use crate::io::DropMode;
use crate::io::Parameters;
use crate::io::TreeBackups;
use crate::Feature;
use crate::Sample;
use crate::split_thread_pool::SplitThreadPool;
// use crate::tree_thread_pool::TreeThreadPool;
// use predictor::predict;
// use compact_predictor::compact_predict;
// use weigh_leaves::weigh_leaves;
use crate::node::StrippedNode;
// use compact_predictor::node_sample_encoding;

impl Forest {
    pub fn initialize(input_array: &Vec<Vec<f64>>,output_array: &Vec<Vec<f64>>, parameters: Arc<Parameters>, report_address:&str) -> Forest {

        let report_string = format!("{}.0",report_address).to_string();

        let samples = Sample::nvec_global(&parameters.sample_names);

        let input_features = Feature::nvec_global(&parameters.input_feature_names);

        let output_features = Feature::nvec_global(&parameters.output_feature_names);

        let prototype_tree = Tree::prototype_tree(&input_array,&output_array,&input_features,&output_features,&samples,None, parameters.clone() ,report_string);

        prototype_tree.serialize_compact();

        let tree_limit = parameters.tree_limit;
        let processor_limit = parameters.processor_limit;

        Forest {
            trees: Vec::new(),
            predictive_trees: Vec::new(),
            size: tree_limit,
            prototype_tree: Some(prototype_tree),
            processor_limit: processor_limit,
            parameters: parameters.clone(),
        }
    }

    pub fn generate(&mut self, parameters:Arc<Parameters>, remember: bool) -> Result<(),Error> {

        if let Some(ref prototype) = self.prototype_tree {

            for tree in 1..self.size+1 {

                let mut new_tree = self.prototype_tree.as_ref().expect("No prototype tree").clone();
                new_tree.grow_branches(parameters.clone());
                if remember {
                    if let Ok(compact) = new_tree.serialize_compact_consume() {
                        self.predictive_trees.push(compact);
                    }
                }
            }


            let mut output_header_dump = OpenOptions::new().create(true).append(false).open([&self.parameters.report_address.clone(),".ifh"].join(""))?;
            output_header_dump.write(self.prototype_tree.as_ref().unwrap().input_feature_names().join("\n").as_bytes())?;
            output_header_dump.write(b"\n")?;

            let mut output_header_dump = OpenOptions::new().create(true).append(false).open([&self.parameters.report_address.clone(),".ofh"].join(""))?;
            output_header_dump.write(self.prototype_tree.as_ref().unwrap().output_feature_names().join("\n").as_bytes())?;
            output_header_dump.write(b"\n")?;

            Ok(())
        }
        else {
            panic!("Attempted to generate a forest without a prototype tree. Are you trying to do predictions after reloading from compact backups?")
        }

        // self.set_leaf_weights();

    }



    // pub fn set_leaf_weights(&mut self) {
    //     let truth = self.counts.clone();
    //     let sample_header = self.sample_map();
    //     let mut leaves = self.mut_leaves();
    //     let encoding = node_sample_encoding(&{leaves.iter().map(|x| &**x).collect()},&sample_header);
    //     let leaf_weights = weigh_leaves(&{leaves.iter().map(|x| &**x).collect()},&encoding,&truth);
    //     for (i,weights) in leaf_weights.into_iter().enumerate() {
    //         leaves[i].set_weights(weights)
    //     }
    // }

    pub fn compact_reconstitute(tree_locations: TreeBackups, feature_option: Option<Vec<String>>,sample_option:Option<Vec<String>>,processor_option: Option<usize>, report_address:&str) -> Result<Forest,Error> {

        let mut predictive_trees: Vec<PredictiveTree>;

        let processor_limit = processor_option.unwrap_or(1);

        let split_thread_pool = SplitThreadPool::new(processor_limit);


        match tree_locations {
            TreeBackups::File(location) => {
                let tree_file = File::open(location)?;
                let mut tree_locations: Vec<String> = io::BufReader::new(&tree_file).lines().map(|x| x.expect("Tree location error!")).collect();
                predictive_trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    predictive_trees.push(PredictiveTree::reload(&loc,1,"".to_string())?);
                }
            }
            TreeBackups::Vector(tree_locations) => {
                predictive_trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    predictive_trees.push(PredictiveTree::reload(&loc,1,"".to_string())?);
                }
            }
            TreeBackups::Trees(backup_trees) => {
                predictive_trees = backup_trees;
            }
        }

        let parameters = Arc::new(Parameters::empty());

        let prototype_tree = predictive_trees.remove(0);

        let dimensions = (0,0);

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.reconstituted.0",report_address).to_string();

        Ok (Forest {
            size: predictive_trees.len(),
            prototype_tree: None,
            processor_limit: processor_option.unwrap_or(1),
            trees: Vec::new(),
            predictive_trees: predictive_trees,
            parameters:parameters
        })

    }

    pub fn reconstitute(tree_locations: TreeBackups, feature_option: Option<Vec<String>>,sample_option:Option<Vec<String>>,processor_option: Option<usize>, report_address:&str) -> Result<Forest,Error> {

        let mut trees: Vec<Tree>;

        let processor_limit = processor_option.unwrap_or(1);

        let split_thread_pool = SplitThreadPool::new(processor_limit);

        match tree_locations {
            TreeBackups::File(location) => {
                let tree_file = File::open(location)?;
                let mut tree_locations: Vec<String> = io::BufReader::new(&tree_file).lines().map(|x| x.expect("Tree location error!")).collect();
                trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    trees.push(Tree::reload(&loc,split_thread_pool.clone(),1,1,"".to_string())?);
                }
            }
            TreeBackups::Vector(tree_locations) => {
                trees = Vec::with_capacity(tree_locations.len());
                for loc in tree_locations {
                    trees.push(Tree::reload(&loc,split_thread_pool.clone(),1,1,"".to_string())?);
                }
            }
            TreeBackups::Trees(backup_trees) => {
                trees = vec![];
                unimplemented!("Tree vectors for reconsitition are coming later")
            }
        }

        let prototype_tree = trees.remove(0);

        let dimensions = prototype_tree.dimensions();

        let feature_names = feature_option.unwrap_or((0..dimensions.0).map(|x| x.to_string()).collect());

        let sample_names = sample_option.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());

        let report_string = format!("{}.reconstituted.0",report_address).to_string();

        let parameters = Arc::new(Parameters::empty());

        Ok (Forest {
            size: trees.len(),
            prototype_tree: Some(prototype_tree),
            processor_limit: processor_option.unwrap_or(1),
            trees: trees,
            predictive_trees: Vec::new(),
            parameters: parameters,
        })
    }

    // pub fn compact_predict(&self,counts:&Vec<Vec<f64>>,feature_map: &HashMap<String,usize>,parameters: Arc<Parameters> ,report_address: &str) -> Result<Vec<Vec<f64>>,Error> {
    //
    //     println!("Predicting:");
    //
    //     let predictions = compact_predict(&self.predictive_trees,&matrix_flip(counts),feature_map, parameters);
    //
    //     let mut prediction_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction"].join("")).unwrap();
    //     prediction_dump.write(&tsv_format(&predictions).as_bytes())?;
    //     prediction_dump.write(b"\n")?;
    //
    //     let mut truth_dump = OpenOptions::new().create(true).append(true).open([report_address,".prediction_truth"].join("")).unwrap();
    //     truth_dump.write(&tsv_format(&matrix_flip(&self.counts)).as_bytes())?;
    //     truth_dump.write(b"\n")?;
    //
    //     let mut header_vec = vec!["";feature_map.len()];
    //     for (f,i) in feature_map { header_vec[*i] = f; };
    //     let header = tsv_format(&vec![header_vec]);
    //
    //     let mut prediction_header = OpenOptions::new().create(true).append(true).open([report_address,".prediction_header"].join("")).unwrap();
    //     prediction_header.write(&header.as_bytes())?;
    //     prediction_header.write(b"\n")?;
    //
    //     Ok(predictions)
    // }

    pub fn trees(&self) -> &Vec<Tree> {
        &self.trees
    }

    pub fn predictive_trees(&self) -> &Vec<PredictiveTree> {
        &self.predictive_trees
    }

    pub fn leaves(&self) -> Vec<&StrippedNode> {

        let mut leaves = vec![];

        for tree in &self.predictive_trees {
            leaves.extend(tree.crawl_to_leaves());
        }
        leaves
    }

    pub fn mut_leaves(&mut self) -> Vec<&mut StrippedNode> {

        let mut leaves = vec![];

        for tree in self.predictive_trees.iter_mut() {
            leaves.extend(tree.mut_crawl_to_leaves());
        }
        leaves
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.prototype_tree.as_ref().unwrap().dimensions()
    }

    pub fn input_features(&self) -> Option<Vec<String>>  {
        self.prototype_tree.as_ref().map(|x| x.input_feature_names())
    }

    pub fn output_features(&self) -> Option<Vec<String>> {
        self.prototype_tree.as_ref().map(|x| x.output_feature_names())
    }

    // pub fn feature_map(&self) -> Option<HashMap<String,usize>> {
    //     self.output_features().map(|x| x.clone().into_iter().enumerate().map(|x| (x.1,x.0)).collect())
    // }
    //
    // pub fn sample_map(&self) -> HashMap<String,usize> {
    //     self.prototype_tree.as_ref().unwrap().samples().iter().map(|s| (s.name().clone(),s.index().clone())).collect()
    // }

    // pub fn random_features(&self, n_features) -> &Vec<&String> {
    //
    // }

}

pub enum SampleMode {
    Map(Vec<HashMap<String,f64>>),
    VectorHeader(Vec<Vec<f64>>,HashMap<String,usize>),
}

pub struct Forest {
    trees: Vec<Tree>,
    predictive_trees: Vec<PredictiveTree>,
    size: usize,
    prototype_tree: Option<Tree>,
    processor_limit: usize,
    parameters: Arc<Parameters>,
}

fn split_shuffle<T>(source_vector: Vec<T>, pieces: usize) -> Vec<Vec<T>> {

    let piece_length = source_vector.len()/pieces;
    let mut len = source_vector.len();

    let mut rng = rand::thread_rng();

    let mut shuffled_source = seq::sample_iter(&mut rng, source_vector.into_iter(), len).unwrap_or(vec![]);

    if shuffled_source.len() < 1 {
        panic!("Failed to shuffle features correctly!")
    }

    let mut vector_pieces: Vec<Vec<T>> = Vec::with_capacity(pieces);

    for _ in 0..pieces {

        len -= piece_length;

        vector_pieces.push(shuffled_source.split_off(len))
    }

    vector_pieces
}

#[cfg(test)]
mod random_forest_tests {

    use super::*;
    use super::super::io::{read_matrix,read_header};
    use std::fs::remove_file;

    #[test]
    fn test_forest_initialization_trivial() {
        Forest::initialize(&vec![],&vec![], Arc::new(Parameters::empty()) , "../testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_simple() {
        let counts = vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        Forest::initialize(&counts,&counts, Arc::new(Parameters::empty()), "../testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_iris() {
        let counts = read_matrix("../testing/iris.drop");
        let features = read_header("../testing/iris.features");
        Forest::initialize(&counts,&counts, Arc::new(Parameters::empty()),"../testing/err");
    }

    #[test]
    fn test_forest_initialization_iris_nan() {
        let mut params = Parameters::empty();
        params.dropout = DropMode::NaNs;
        let counts = read_matrix("../testing/iris.nan");
        let features = read_header("../testing/iris.features");
        Forest::initialize(&counts,&counts, Arc::new(params),"./testing/err");
    }


    // #[test]
    // fn test_forest_reconstitution_simple() {
    //     let params = Parameters::empty();
    //     params.backup_vec = Some(vec!["./testing/precomputed_trees/simple.0.compact".to_string(), "./testing/precomputed_trees/simple.1.compact".to_string()]);
    //     let new_forest = Forest::compact_reconstitute(Arc::new(params),"./testing/").expect("Reconstitution test");
    //
    //     println!("Reconstitution successful");
    //
    //     let reconstituted_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_features: Vec<String> = vec!["0","0","0","0","0","0"].iter().map(|x| x.to_string()).collect();
    //     assert_eq!(reconstituted_features,correct_features);
    //
    //
    //     let correct_splits: Vec<f64> = vec![-1.,-2.,20.,10.,10.,5.];
    //     let reconstituted_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     assert_eq!(reconstituted_splits,correct_splits);
    // }
    //
    //
    // #[test]
    // fn test_forest_reconstitution() {
    //     let new_forest = Forest::compact_reconstitute(TreeBackups::Vector(vec!["./testing/precomputed_trees/iris.0.compact".to_string(),"./testing/precomputed_trees/iris.1.compact".to_string()]), None, None, Some(1), "./testing/").expect("Reconstitution test");
    //
    //     println!("Reconstitution successful");
    //
    //     let reconstituted_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
    //     assert_eq!(reconstituted_features,correct_features);
    //
    //
    //     let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
    //     let reconstituted_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    // }

    // #[test]
    // fn test_forest_generation() {
    //
    //     let counts = read_counts("./testing/iris.drop");
    //     let features = read_header("./testing/iris.features");
    //
    //     let mut params = Parameters::empty();
    //
    //     params.leaf_size_cutoff = Some(10);
    //
    //
    //     params.input_features = Some(4);
    //     params.output_features = Some(4);
    //     params.sample_subsample = Some(150);
    //     params.processor_limit = Some(1);
    //     params.counts = Some(counts.clone());
    //     params.feature_names = Some(features);
    //     params.tree_limit = Some(1);
    //     params.auto();
    //
    //     let arc_params = Arc::new(params);
    //
    //     let mut new_forest = Forest::initialize(&counts,arc_params.clone(), "./testing/tmp_test");
    //     new_forest.generate(arc_params.clone(), true);
    //
    //
    //     let computed_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
    //     assert_eq!(computed_features,correct_features);
    //
    //
    //     let computed_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
    //     assert_eq!(computed_splits,correct_splits);
    //
    //
    //     remove_file("./testing/tmp_test.0");
    //     remove_file("./testing/tmp_test.0.summary");
    //     remove_file("./testing/tmp_test.0.dump");
    //     remove_file("./testing/tmp_test.1");
    //     remove_file("./testing/tmp_test.1.summary");
    //     remove_file("./testing/tmp_test.1.dump");
    // }
}
