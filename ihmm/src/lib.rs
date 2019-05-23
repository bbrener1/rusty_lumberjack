
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

use std::cmp::PartialEq;

use std::f64::consts::E;
use std::f64::consts::PI;

use std::num::NonZeroUsize;

use ndarray as nd;
use ndarray_linalg as ndl;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::solveh::{InverseH,DeterminantH};

use rand::{thread_rng,Rng};

use multivariate_normal::MVN;
use multivariate_normal::{array_mask,array_mask_axis,array_double_select,array_double_mask,masked_array_properties};
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

#[derive(Clone,Debug)]
struct HiddenState {
    nodes: Vec<usize>,
    transition_model: Dirichlet<Option<usize>>,
    emission_model: MVN,
}

impl HiddenState {

    fn blank(features:usize) -> HiddenState {
        let emission_model = MVN::identity_prior(1, features as u32);
        let transition_model = Dirichlet::blank(NonZeroUsize::new(1).unwrap());
        let nodes = vec![];
        HiddenState{
            nodes,
            transition_model,
            emission_model,
        }
    }

    fn feature_log_odds(&self,data:&ArrayView<f64,Ix1>,mask:&ArrayView<bool,Ix1>) -> f64 {
        self.emission_model.masked_likelihood(data, mask)
    }

    fn mixture_log_odds(&self,parent_state:Option<usize>) -> f64 {
        self.transition_model.log_odds(&parent_state).unwrap_or(0.)
    }

}

pub struct IHMM {
    beta: usize,
    gamma: usize,
    beta_e: f64,
    hidden_states: Vec<HiddenState>,
    data_prior: HiddenState,
    nodes: Vec<MarkovNode>,
    encoding: (Array<f64,Ix2>,Array<bool,Ix2>),
}

impl IHMM {
    fn new(nodes:Vec<MarkovNode>) -> IHMM {

        let encoding = MarkovNode::encode(&nodes);

        let features = encoding.0.dim().1;

        IHMM {
            beta: 1,
            gamma: 1,
            beta_e: 0.,
            hidden_states: vec![],
            data_prior: HiddenState::blank(features),
            nodes: nodes,
            encoding: encoding,
        }
    }

    fn initialize(&mut self,states:usize) {
        eprintln!("Initializing prior");
        let features = self.encoding.0.dim().1;
        for _ in 0..states {
            self.hidden_states.push(HiddenState::blank(features));
        }
        let live_indices: Vec<usize> = self.nodes.iter().filter(|n| n.parent.is_some()).map(|n| n.index).collect();
        eprintln!("Estimating prior");
        self.data_prior = self.estimate_state_from_indices(&live_indices);
        eprintln!("Resampling states");
        self.resample_states();
    }

    fn sample_node_state(&self, node_index: usize) -> Option<usize> {
        let (features,mask) = self.node_encoding(node_index);
        // eprintln!("Computing feature log likelihoods");
        let mut feature_log_odds: Vec<f64> = self.hidden_states.iter().map(|s| s.feature_log_odds(&features,&mask)).collect();
        let mut mixture_log_odds: Vec<f64> = self.hidden_states.iter().map(|s| 0.).collect();
        let mut log_odds: Vec<f64> = feature_log_odds.iter().zip(mixture_log_odds.iter()).map(|(f,m)| f+m).collect();
        let log_max: f64 = log_odds.iter().fold(std::f64::NEG_INFINITY,|acc,o| f64::max(acc,*o));
        assert!(log_max.is_finite());
        log_odds = log_odds.iter().map(|o| o - log_max).collect();
        let state = sample_log_odds(log_odds);
        state
    }


    fn resample_states(&mut self) {
        let hidden_states: Vec<Option<usize>> = self.nodes.iter().map(|n| {
            // eprintln!("Resampling node {:?}",n.index);
            if n.parent.is_some() { self.sample_node_state(n.index) }
            else { None }
        }).collect();
        for (node,state) in self.nodes.iter_mut().zip(hidden_states) {
            node.hidden_state = state;
        }
    }

    fn nodes_by_index(&self,indices:&[usize]) -> Vec<&MarkovNode> {
        indices.iter().map(|i| &self.nodes[*i]).collect()
    }

    fn select_encoding(&self,indices:&[usize]) -> (Array<f64,Ix2>,Array<bool,Ix2>) {
        let (data,mask) = &self.encoding;
        let selected_data = data.select(Axis(0),indices);
        let selected_mask = mask.select(Axis(0),indices);
        (selected_data,selected_mask)
    }

    fn node_encoding(&self,index:usize) -> (ArrayView<f64,Ix1>,ArrayView<bool,Ix1>) {
        (self.encoding.0.row(index),self.encoding.1.row(index))
    }

    fn repartition_hidden_states(&mut self) {
        let represented_states: HashSet<usize> = self.nodes.iter().flat_map(|n| n.hidden_state).collect();
        let mut sorted_represented_states: Vec<usize> = represented_states.into_iter().collect();
        sorted_represented_states.sort();
        // eprintln!("Sorted Represented States: {:?}", sorted_represented_states);
        let mut state_node_collections = vec![];
        for node in self.nodes.iter_mut() {
            if let Some(old_state) = node.hidden_state {
                node.hidden_state = Some(sorted_represented_states[old_state]);
            }
        }
        for new_state in sorted_represented_states.iter() {
            let state_node_indices: Vec<usize> = self.nodes.iter().filter(|n| n.hidden_state == Some(*new_state)).map(|n| n.index).collect();
            eprintln!("Collecting state {:?}",new_state);
            eprintln!("{:?}",state_node_indices);
            state_node_collections.push(state_node_indices);
        }
        let mut hidden_states = vec![];
        for nodes in state_node_collections {
            hidden_states.push(self.estimate_state_from_indices(&nodes[..]));
            // hidden_states.push(HiddenState::new(hidden_state_node_indices,encoding));
        }
        self.hidden_states = hidden_states;
    }

    fn estimate_state_from_indices(&self,indices:&[usize]) -> HiddenState {
        let (data,mask) = self.select_encoding(&indices);
        let (prior_means,prior_variances) = (self.data_prior.emission_model.means(),self.data_prior.emission_model.variances());
        let features = prior_means.dim();
        let samples = indices.len();
        let parent_states = self.get_node_parent_states(indices);

        let mut emission_model = MVN::scaled_identity_prior(&prior_means, &prior_variances, (features+1) as u32);
        emission_model.estimate_masked(&data.view(), &mask.view());
        let transition_model = Dirichlet::estimate(&parent_states, NonZeroUsize::new(self.beta).unwrap());

        let mut state = HiddenState{nodes:indices.to_vec(), transition_model, emission_model};

        state
    }

    fn generate_transition_matrix(&self) -> Array<i32,Ix2> {
        let s = self.hidden_states.len();
        let mut transitions = Array::zeros((s,s));
        for node in &self.nodes {
            let ni = node.index;
            if let Some(pi) = node.parent {
                let parent = &self.nodes[pi];
                if let (Some(ns),Some(ps)) = (node.hidden_state,parent.hidden_state) {
                    transitions[[ns,ps]] += 1;
                }
            }
        }
        transitions
    }

    fn get_node_parent_states(&self,indices:&[usize]) -> Vec<Option<usize>> {
        let mut parent_states = Vec::with_capacity(indices.len());
        for ni in indices {
            let node = &self.nodes[*ni];
            if let Some(pi) = node.parent {
                let parent = &self.nodes[pi];
                parent_states.push(parent.hidden_state)
            }
        }
        parent_states
    }

    // fn generate_transition_model(transitions: &[i32]) -> Dirichlet {
    // }

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


pub fn sample_log_odds(odds:Vec<f64>) -> Option<usize> {
    let mut sorted_raw_odds: Vec<(usize,f64)> = odds.iter().cloned().map(f64::exp2).enumerate().collect();
    assert!(!sorted_raw_odds.iter().any(|(i,o)| o.is_nan()));
    // eprintln!("{:?}",sorted_raw_odds);
    sorted_raw_odds.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    sorted_raw_odds.reverse();
    // eprintln!("{:?}",sorted_raw_odds);
    let exponential_sum: f64 = sorted_raw_odds.iter().map(|(i,o)| o).sum();
    // eprintln!("{:?}",exponential_sum);
    let mut range_selection: f64 = rand::thread_rng().gen_range(0.,exponential_sum);
    // eprintln!("{:?}",range_selection);
    for (i,o) in sorted_raw_odds {
        range_selection -= o;
        if range_selection < 0. {
            return Some(i)
        }
    }
    return None
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

    // #[test]
    // fn test_markov_partition_iris() {
    //     let mut model = iris_model();
    //     model.initialize(5);
    //     model.repartition_hidden_states();
    //     eprintln!("{:?}",model.hidden_states.len());
    //     assert_eq!(model.hidden_states.len(),5);
    //     panic!()
    // }

    #[test]
    fn test_markov_log_select() {
        let mut draws = vec![];
        for _ in 0..100 {
            draws.push(sample_log_odds(vec![0.,-1.,-1.,-2.,-3.]));
            eprintln!("{:?}",draws.last());
        }
        for i in 0..5 {
            eprintln!("{:?}",draws.iter().filter(|x| x == &&Some(i)).count());
        }
        // panic!();
    }

    #[test]
    fn test_markov_multipart() {
        let mut model = iris_model();
        model.initialize(5);
        model.repartition_hidden_states();
        for i in 0..1000 {
            eprintln!("###############################");
            eprintln!("###############################");
            eprintln!("############   {:?}   #############",i);
            eprintln!("###############################");
            eprintln!("###############################");
            model.resample_states();
            model.repartition_hidden_states();
            for state in &model.hidden_states {
                eprintln!("MEANS");
                eprintln!("{:?}",state.emission_model.means())
            }
        }
        for state in &model.hidden_states {
            eprintln!("{:?}", state);
        }
        eprintln!("{:?}", model.generate_transition_matrix());
        // assert_eq!(model.hidden_states.len(),5);
        panic!();
    }

}




















//
