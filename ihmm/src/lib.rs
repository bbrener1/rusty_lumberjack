#[macro_use(array,azip)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate trees;
extern crate num_traits;
extern crate serde_json;
extern crate rand;

mod dirichlet;
mod multivariate_normal;

use trees::node::StrippedNode;
use trees::{Feature,Sample,Prerequisite};

use std::collections::{HashMap,HashSet};

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;

use std::f64::consts::E;
use std::f64::consts::PI;

use ndarray as nd;
use ndarray_linalg as ndl;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::solveh::{InverseH,DeterminantH};

use rand::{thread_rng,Rng};

use multivariate_normal::MVN;
use multivariate_normal::{array_mask,array_mask_axis,array_double_select,array_double_mask};
use dirichlet::Dirichlet;

pub struct MarkovNode {
    index: usize,
    hidden_state: Option<usize>,
    parent: Option<usize>,
    children: Option<(usize,usize)>,
    samples: Vec<Sample>,
    features: Vec<Feature>,
    emissions: Vec<f64>,
}

struct HiddenState {
    nodes: Vec<usize>,
    emission_model: MVN,
}

impl HiddenState {
    fn new(indices:Vec<usize>,encoding:(Array<f64,Ix2>,Array<bool,Ix2>)) -> HiddenState {
        let emission_model = MVN::estimate_against_identity(&encoding.0.view(),&encoding.1.view()).expect("Failed to estimate ");
        HiddenState{
            nodes:indices,
            emission_model
        }
    }
}

pub struct IHMM {
    beta: f64,
    gamma: f64,
    beta_e: f64,
    hidden_states: Vec<HiddenState>,
    nodes: Vec<MarkovNode>,
    encoding: (Array<f64,Ix2>,Array<bool,Ix2>),
    mixture_model: Dirichlet,
}

impl IHMM {
    fn new(nodes:Vec<MarkovNode>) -> IHMM {

        let encoding = MarkovNode::encode(&nodes);

        IHMM {
            beta: 0.,
            gamma: 0.,
            beta_e: 0.,
            hidden_states: vec![],
            nodes: nodes,
            encoding: encoding,
            mixture_model: Dirichlet::blank(1, 1),
        }
    }

    fn initialize(&mut self,states:usize) {

        for node in &mut self.nodes {
            if node.parent.is_some() && node.children.is_some() {
                node.hidden_state = Some(thread_rng().gen_range(0,states));
            }
        }
    }

    fn nodes_by_index(&self,indices:&[usize]) -> Vec<&MarkovNode> {
        indices.iter().map(|i| &self.nodes[*i]).collect()
    }

    fn select_encoding(&self,indices:&[usize]) -> (Array<f64,Ix2>,Array<bool,Ix2>) {
        let (data,mask) = &self.encoding;
        let selected_data = array_double_select(&data.view(),indices);
        let selected_mask = array_double_select(&mask.view(),indices);
        (selected_data,selected_mask)
    }

    fn repartition_hidden_states(&mut self) {
        let represented_states: HashSet<usize> = self.nodes.iter().flat_map(|n| n.hidden_state).collect();
        let mut sorted_represented_states: Vec<usize> = represented_states.into_iter().collect();
        sorted_represented_states.sort();
        let mut state_node_collections = vec![];
        for (old_state,new_state) in sorted_represented_states.iter().enumerate() {
            let state_node_indices: Vec<usize> = self.nodes.iter().filter(|n| n.hidden_state == Some(old_state)).map(|n| n.index).collect();
            for ni in &state_node_indices {
                self.nodes[*ni].hidden_state = Some(*new_state);
            }
            state_node_collections.push(state_node_indices);
        }
        let mut hidden_states = vec![];
        for hidden_state_node_indices in state_node_collections {
            let encoding = self.select_encoding(&hidden_state_node_indices);
            hidden_states.push(HiddenState::new(hidden_state_node_indices,encoding));
        }
        self.hidden_states = hidden_states;
    }
}


impl MarkovNode {

    pub fn encode(nodes:&Vec<MarkovNode>) -> (Array<f64,Ix2>,Array<bool,Ix2>) {
        let mut features = HashSet::new();
        for node in nodes {
            for feature in &node.features {
                features.insert(feature);
            }
        }
        if features.iter().any(|f| *f.index() > features.len() + 1) {
            panic!("Not all features read correctly, mising indices");
        }

        let mut data: Array<f64,Ix2> = Array::zeros((nodes.len(),features.len()));
        let mut mask: Array<bool,Ix2> = Array::from_shape_fn((nodes.len(),features.len()),|_| false);

        for (i,node) in nodes.iter().enumerate() {
            for (feature,value) in node.features.iter().zip(node.emissions.iter()) {
                data[[i,*feature.index()]] = *value;
                mask[[i,*feature.index()]] = true;
            }
        }

        (data,mask)
    }

    pub fn from_stripped_vec(stripped:&Vec<StrippedNode>) -> Vec<MarkovNode> {
        let mut markov = vec![];
        for root in stripped {
            let ci = markov.len();
            markov.append(&mut MarkovNode::from_stripped_node(root, ci));
        }
        markov
    }

    pub fn from_stripped_node(original:&StrippedNode,passed_index:usize) -> Vec<MarkovNode> {

        let mut nodes = vec![];
        let mut children = None;
        let mut index = passed_index;
        if let [ref lc,ref rc] = original.children[..] {
            let mut left_children = MarkovNode::from_stripped_node(lc,index);
            let lci = left_children.last().unwrap().index;
            let mut right_children = MarkovNode::from_stripped_node(rc,lci+1);
            let rci = right_children.last().unwrap().index;
            index = rci + 1;
            left_children.last_mut().unwrap().parent = Some(index);
            right_children.last_mut().unwrap().parent = Some(index);
            children = Some((lci,rci));
            nodes.append(&mut left_children);
            nodes.append(&mut right_children);
        }

        let parent = None;
        let samples = original.samples().to_vec();
        let features = original.features().to_vec();
        let emissions = original.medians().to_vec();


        let wrapped = MarkovNode{
            index,
            parent,
            children,
            samples,
            features,
            emissions,
            hidden_state: None,

        };

        nodes.push(wrapped);

        nodes
    }
}

impl IHMM {


}




#[cfg(test)]
pub mod tree_braider_tests {

    use super::*;

    pub fn iris_forest() -> Vec<MarkovNode> {
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/iris_forest/").unwrap())
    }

    pub fn iris_model() -> IHMM {
        let forest = iris_forest();
        let model = IHMM::new(forest);
        model
    }

    #[test]
    fn test_markov_import() {
        eprintln!("Test readout");
        let stripped_nodes = StrippedNode::from_location("../testing/iris_forest/").unwrap();
        eprintln!("Stripped nodes read");
        MarkovNode::from_stripped_vec(&stripped_nodes);
    }

    #[test]
    fn test_markov_encoding_iris() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(&nodes);
        eprintln!("{:?}",data.dim());
        eprintln!("{:?}",mask.dim());
        assert_eq!(data.dim().1,4);
        assert_eq!(mask.dim().1,4);
        assert_eq!(data.dim(),mask.dim());
    }

    fn test_markov_partition_iris() {
        let mut model = iris_model();
        model.initialize(5);
        model.repartition_hidden_states();
        eprintln!("{:?}",model.hidden_states.len());
        assert_eq!(model.hidden_states.len(),5);
        panic!();
    }
}




















//
