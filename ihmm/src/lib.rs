
#[macro_use(array,azip)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate trees;
extern crate num_traits;
extern crate serde_json;
extern crate rand;
extern crate rayon;

use rayon::prelude::*;

mod dirichlet;
mod multivariate_normal;

use trees::node::StrippedNode;
use trees::{Feature,Sample,Prerequisite};

use std::collections::{HashMap,HashSet};

use std::io;
use std::io::Write;
use std::io::prelude::*;

use std::io::Read;
use std::error::Error;

use std::fs;
use std::fs::File;
use std::path::Path;
use std::ffi::OsStr;
use std::env;


use std::cmp::PartialEq;

use std::f64;
use std::f64::consts::E;
use std::f64::consts::PI;

use std::mem::{swap,replace};

use std::num::NonZeroUsize;

use ndarray as nd;
use ndarray_linalg as ndl;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::solveh::{InverseH,DeterminantH};

use rand::{thread_rng,Rng};
use rand::distributions::{Distribution,Binomial};

use multivariate_normal::MVN;
use multivariate_normal::{array_mask,array_mask_axis,array_double_select,array_double_mask,masked_array_properties};
use dirichlet::{SymmetricDirichlet,Categorical};


pub struct MarkovNode {
    index: usize,
    oracle: bool,
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
    direct_transition_model: SymmetricDirichlet<Option<usize>>,
    oracle_transition_model: SymmetricDirichlet<Option<usize>>,
    emission_model: MVN,
}

impl HiddenState {

    fn blank(features:usize,states:usize) -> HiddenState {
        let emission_model = MVN::identity_prior(1, features as u32);
        let mut potential_states = vec![None];
        potential_states.extend((0..states).map(|i| Some(i)));
        let direct_transition_model = SymmetricDirichlet::blank_categories(&potential_states, NonZeroUsize::new(1).unwrap());
        let oracle_transition_model = SymmetricDirichlet::blank_categories(&potential_states, NonZeroUsize::new(1).unwrap());
        let nodes = vec![];
        HiddenState{
            nodes,
            direct_transition_model,
            oracle_transition_model,
            emission_model,
        }
    }

    fn feature_log_odds(&self,data:&ArrayView<f64,Ix1>,mask:&ArrayView<bool,Ix1>) -> f64 {
        self.emission_model.masked_likelihood(data, mask)
    }

    fn mixture_log_odds(&self,state:Option<usize>) -> f64 {
        let dto = self.direct_transition_odds(state);
        let oo = self.oracle_odds();
        let oto = self.oracle_transition_odds(state);
        (dto + (oo * oto)).log2()
    }

    fn oracle_odds(&self) -> f64 {
        self.direct_transition_model.oracle_odds()
    }

    fn oracle_transition_odds(&self,state:Option<usize>) -> f64 {
        self.oracle_transition_model.odds(&state).unwrap_or(0.)
    }

    fn direct_transition_odds(&self,state:Option<usize>) -> f64 {
        self.direct_transition_model.odds(&state).unwrap_or(0.)
    }

    fn oracle_transition_probability(&self, state:Option<usize>) -> f64 {
        let oracle_odds = self.oracle_odds();
        let direct_transition_odds = self.direct_transition_odds(state);
        let oracle_transition_odds = self.oracle_transition_odds(state);

        // eprintln!("Oracle Sampling Debug");
        // eprintln!("{:?}",state);
        // eprintln!("{:?}",self);
        // eprintln!("{:?}",oracle_odds);
        // eprintln!("{:?}",direct_transition_odds);
        // eprintln!("{:?}",oracle_transition_odds);

        let oracle_transition_probability = (oracle_odds * oracle_transition_odds) / ((oracle_odds * oracle_transition_odds) + direct_transition_odds);

        // eprintln!("{:?}",oracle_transition_probability);

        return oracle_transition_probability
    }


    fn set_emission_model(&mut self, model:MVN) {
        self.emission_model = model;
    }

    fn set_direct_transition_model(&mut self, model: SymmetricDirichlet<Option<usize>>) {
        self.direct_transition_model = model;
    }

    fn set_oracle_transition_model(&mut self, model: SymmetricDirichlet<Option<usize>>) {
        self.oracle_transition_model = model;
    }

}

pub struct IHMM {
    nodes: Vec<MarkovNode>,
    encoding: (Array<f64,Ix2>,Array<bool,Ix2>),
    beta: NonZeroUsize,
    gamma: NonZeroUsize,
    beta_e: f64,
    oracle_transition_model: SymmetricDirichlet<Option<usize>>,
    dp_transition_model: SymmetricDirichlet<Option<usize>>,
    prior_emission_model: MVN,
    hidden_states: Vec<HiddenState>,
}

impl IHMM {
    fn new(nodes:Vec<MarkovNode>) -> IHMM {

        let encoding = MarkovNode::encode(&nodes);

        let features = encoding.0.dim().1;

        IHMM {
            beta: NonZeroUsize::new(20).unwrap(),
            gamma: NonZeroUsize::new(1).unwrap(),
            beta_e: 0.,
            hidden_states: vec![],
            prior_emission_model: MVN::identity_prior(1, features as u32),
            oracle_transition_model: SymmetricDirichlet::blank(NonZeroUsize::new(1).unwrap()),
            dp_transition_model: SymmetricDirichlet::blank(NonZeroUsize::new(1).unwrap()),
            nodes: nodes,
            encoding: encoding,
        }
    }

    fn initialize(&mut self,states:usize) {
        eprintln!("Initializing");
        let features = self.encoding.0.dim().1;
        for _ in 0..states {
            self.hidden_states.push(HiddenState::blank(features,states));
        }
        eprintln!("Estimating prior");
        self.prior_emission_model = self.estimate_prior_features();
        for ni in self.live_indices() {
            self.nodes[ni].hidden_state = Some(thread_rng().gen_range(0,states));
            self.nodes[ni].oracle = rand::random::<f64>() < (1./(states as f64));
        }
        self.oracle_transition_model = SymmetricDirichlet::blank_categories(&self.current_states(), self.gamma);
        self.estimate_states();
    }

    fn sample_node_state(&self, node_index: usize) -> Option<usize> {

        let (features,mask) = self.node_encoding(node_index);
        let node = &self.nodes[node_index];
        // let parent_state = self.nodes[node.parent?].hidden_state;
        let cls = self.nodes[node.children?.0].hidden_state;
        let crs = self.nodes[node.children?.1].hidden_state;
        // eprintln!("Computing feature log likelihoods");

        let mut state_log_odds = Vec::with_capacity(self.hidden_states.len() + 1);

        eprint!("{:?}:[",node_index);
        for (si,state) in self.hidden_states.iter().enumerate() {
            let feature_log_odds = state.feature_log_odds(&features,&mask);
            // let mixture_log_odds = state.mixture_log_odds(cls) + state.mixture_log_odds(crs);
            let mixture_log_odds = self.dp_transition_model.log_odds(&Some(si)).unwrap();
            // let mixture_log_odds = 0.;
            eprint!("({:?},",feature_log_odds);
            eprint!("{:?}),",mixture_log_odds);
            state_log_odds.push(feature_log_odds + mixture_log_odds);
        }
        eprint!("]\n");

        // let new_state_log_odds = {
        //     let new_state_feature_log_odds = self.new_state_feature_log_odds(&features, &mask);
        //     let new_state_mixture_log_odds = self.new_state_mixture_log_odds(cls) + self.new_state_mixture_log_odds(crs);
        //     new_state_feature_log_odds + new_state_mixture_log_odds
        // };

        // state_log_odds.push(new_state_log_odds);

        let log_max: f64 = state_log_odds.iter().fold(std::f64::NEG_INFINITY,|acc,o| f64::max(acc,*o));
        state_log_odds = state_log_odds.iter().map(|o| o - log_max).collect();
        // state_log_odds = state_log_odds.iter().map(|o| o * 0.3).collect();
        assert!(log_max.is_finite());
        eprintln!("LOG_ODDS:{:?}",state_log_odds);
        let state = sample_log_odds(state_log_odds);
        eprintln!("S:{:?}",state);
        state
    }

    pub fn sample_oracle_transition(&self, s1:Option<usize>,s2:Option<usize>) -> bool {
        if let Some(s1i) = s1 {
            if let Some(s1s) = self.hidden_states.get(s1i) {
                let oracle_transition_probability = s1s.oracle_transition_probability(s2);
                return rand::random::<f64>() < oracle_transition_probability
            }
            else { true }
        }
        else { true }
    }


    fn resample_states(&mut self) {
        let hidden_states: Vec<Option<usize>> = self.live_indices().into_iter().map(|ni| {
        // let hidden_states: Vec<Option<usize>> = self.live_indices().into_par_iter().map(|ni| {
            self.sample_node_state(ni)
        }).collect();
        for (ni,state) in self.live_indices().into_iter().zip(hidden_states) {
            self.nodes[ni].hidden_state = state;
        }
    }

    fn resample_oracles(&mut self) {
        let node_states = self.nodes.iter().map(|n| n.hidden_state);
        let parent_states: Vec<Option<usize>> = self.get_node_parent_states(&(0..self.nodes.len()).collect::<Vec<usize>>());
        let transitions = node_states.zip(parent_states.into_iter());
        let oracles: Vec<bool> = transitions.map(|(s1,s2)| self.sample_oracle_transition(s1, s2)).collect();
        for (ni,oracle) in oracles.into_iter().enumerate() {
            self.nodes[ni].oracle = oracle;
        }
    }

    fn sweep(&mut self) {
        self.resample_states();
        self.resample_oracles();
        self.remove_unrepresented_states();
        self.estimate_states();
    }

    fn remove_unrepresented_states(&mut self) {
        eprintln!("Cleaning up unrepresented states");
        // eprintln!("OHS:{:?}",self.nodes.iter().map(|n| n.hidden_state).collect::<Vec<Option<usize>>>());
        let mut current_states: Vec<Option<usize>> = self.hidden_states.iter().map(|_| None).collect();
        let represented_states: Vec<usize> = self.represented_states().iter().flat_map(|rs| *rs).collect();
        current_states.push(Some(represented_states.len()));
        eprintln!("CS:{:?}",current_states);
        eprintln!("RS:{:?}",represented_states);
        for (new_state, old_state) in represented_states.iter().enumerate() {
            current_states[*old_state] = Some(new_state)
        }
        eprintln!("Represented States: {:?}", represented_states);
        for node in self.nodes.iter_mut() {
            if let Some(old_state) = node.hidden_state {
                node.hidden_state = current_states[old_state];
            }
        }
        // eprintln!("NHS:{:?}",self.nodes.iter().map(|n| n.hidden_state).collect::<Vec<Option<usize>>>());

    }


    fn estimate_states(&mut self) {

        let mut new_states = vec![];

        let represented_states: Vec<Option<usize>> = self.represented_states().into_iter().filter(|s| s.is_some()).collect();

        eprintln!("Estimating states:{:?}",represented_states);

        for (i,s) in self.hidden_states.iter().enumerate(){
            eprintln!("FM{:?}:{:?}",i,s.emission_model.means());
        }

        let oracle_transition_model = self.estimate_oracle_transitions();
        self.oracle_transition_model = oracle_transition_model.clone();

        let population_model = self.estimate_population_model();
        self.dp_transition_model = population_model;

        eprintln!("DPM:{:?}",self.dp_transition_model);

        for state in represented_states {
            let indices = self.state_indices(state);
            let state_emission_model = self.estimate_emissions(&indices);
            let state_transition_model = self.estimate_direct_transitions(&indices);
            let state = HiddenState {
                nodes:indices,
                direct_transition_model: state_transition_model,
                oracle_transition_model: oracle_transition_model.clone(),
                emission_model: state_emission_model,
            };
            new_states.push(state)
        }

        self.hidden_states = new_states;

        for (i,s) in self.hidden_states.iter().enumerate(){
            eprintln!("NM{:?}:{:?}",i,s.emission_model.means());
        }

    }

    fn estimate_emissions(&self, indices:&[usize]) -> MVN {

        let (data,mask) = self.select_encoding(&indices);

        let mut emission_model = self.prior_emission_model.clone();
        emission_model.set_samples(1);
        emission_model.estimate_masked(&data.view(), &mask.view());

        emission_model
    }

    fn estimate_population_model(&self) -> SymmetricDirichlet<Option<usize>> {
        let mut state_populations:HashMap<Option<usize>,usize> = self.represented_states().into_iter().filter(|rs| rs.is_some()).map(|s| (s,0)).collect();
        for (state,count) in state_populations.iter_mut() {
            *count = self.state_indices(*state).len();
        }
        let population_model = SymmetricDirichlet::from_map(state_populations, self.beta);
        population_model
    }


    fn estimate_direct_transitions(&self, indices:&[usize]) -> SymmetricDirichlet<Option<usize>> {
        let transitions = self.get_transitions(indices);
        let mut transition_map: HashMap<Option<usize>,usize> = self.current_states().into_iter().map(|cs| (cs,0)).collect();
        for (_,state,oracle) in transitions {
            if !oracle {
                *transition_map.entry(state).or_insert(0) += 1;
            }
        }
        let model = SymmetricDirichlet::from_map(transition_map, self.beta);
        model
    }

    fn estimate_oracle_transitions(&self) -> SymmetricDirichlet<Option<usize>> {
        let transitions = self.get_transitions(&self.live_indices());
        let mut transition_map: HashMap<Option<usize>,usize> = self.represented_states().into_iter().map(|cs| (cs,0)).collect();
        for (_,state,oracle) in transitions {
            if oracle {
                *transition_map.entry(state).or_insert(0) += 1;
            }
        }
        let model = SymmetricDirichlet::from_map(transition_map, self.gamma);
        model
    }

    fn estimate_prior_features(&mut self) -> MVN {

        let indices = self.live_indices();
        self.estimate_emissions(&indices)
    }

    fn new_state_feature_log_odds(&self,data:&ArrayView<f64,Ix1>,mask:&ArrayView<bool,Ix1>) -> f64 {
        self.prior_emission_model.masked_likelihood(data,mask)
    }

    fn new_state_mixture_log_odds(&self,adjacent_state: Option<usize>) -> f64 {
        let oracle_odds = self.oracle_transition_model.odds(&adjacent_state).unwrap_or(1.);
        let double_oracle_odds = self.oracle_transition_model.oracle_odds();
        (oracle_odds * double_oracle_odds).log2()
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

    fn get_node_child_states(&self,indices:&[usize]) -> Vec<Option<usize>> {
        let mut child_states = Vec::with_capacity(indices.len()*2);
        for ni in indices {
            let node = &self.nodes[*ni];
            if let Some((cli,cri)) = node.children {
                let left_child = &self.nodes[cli];
                let right_child = &self.nodes[cri];
                child_states.push(left_child.hidden_state);
                child_states.push(right_child.hidden_state);

            }
        }
        child_states
    }

    fn get_transition_counts(&self,indices:&[usize]) -> HashMap<Option<usize>,usize> {
        let child_states = self.get_node_child_states(&indices);
        let mut state_map: HashMap<Option<usize>,usize> = self.current_states().into_iter().map(|s| (s,0)).collect();
        for cs in child_states {
            *state_map.entry(cs).or_insert(0) += 1
        }
        return state_map
    }

    fn get_transitions(&self, indices:&[usize]) -> Vec<(Option<usize>,Option<usize>,bool)> {
        let mut transitions = Vec::with_capacity(indices.len());
        for ni in indices {
            let node = &self.nodes[*ni];
            let node_state = node.hidden_state;
            if let Some((cli,cri)) = node.children {
                let left_child = &self.nodes[cli];
                let right_child = &self.nodes[cri];
                transitions.push((node_state,left_child.hidden_state,left_child.oracle));
                transitions.push((node_state,right_child.hidden_state,right_child.oracle));
            }
        }
        transitions
    }

    fn represented_states(&self) -> Vec<Option<usize>> {
        let state_set: HashSet<Option<usize>> = self.nodes.iter().map(|n| n.hidden_state).collect();
        let mut state_vec: Vec<Option<usize>> = state_set.into_iter().collect();
        state_vec.sort();
        state_vec
    }

    fn current_states(&self) -> Vec<Option<usize>> {
        let mut current_states = vec![None];
        current_states.extend(self.hidden_states.iter().enumerate().map(|(si,_)| Some(si)));
        current_states
    }

    fn nodes_by_index(&self,indices:&[usize]) -> Vec<&MarkovNode> {
        indices.iter().map(|i| &self.nodes[*i]).collect()
    }

    fn state_indices(&self, state:Option<usize>) -> Vec<usize> {
        self.nodes.iter().filter(|n| n.hidden_state == state).map(|n| n.index).collect()
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

    pub fn live_indices(&self) -> Vec<usize> {
        self.nodes.iter().filter(|n| n.children.is_some()).map(|n| n.index).collect()
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
        // let emissions = original.local_gains().unwrap_or(&vec![0.;features.len()]).to_vec();


        let wrapped = MarkovNode{
            index,
            parent,
            children,
            samples,
            features,
            emissions,
            hidden_state: None,
            oracle: false,

        };

        nodes.push(wrapped);

        nodes
    }
}

impl IHMM {


}


pub fn sample_log_odds(odds:Vec<f64>) -> Option<usize> {
    // eprintln!("{:?}",odds);
    let mut sorted_odds: Vec<(usize,f64)> = odds.into_iter().enumerate().collect();
    sorted_odds.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    // eprintln!("{:?}",sorted_odds);
    let mut sorted_raw_odds: Vec<(usize,f64)> = sorted_odds.iter().cloned().map(|(i,lo)| (i,lo.exp2())).collect();
    sorted_raw_odds = sorted_raw_odds.into_iter().map(|(i,o)| if o.is_nan() {(i,0.)} else {(i,o)}).collect();
    assert!(!sorted_raw_odds.iter().any(|(i,o)| o.is_nan()));
    // eprintln!("{:?}",sorted_raw_odds);
    // sorted_raw_odds.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
    sorted_raw_odds.reverse();
    // eprintln!("{:?}",sorted_raw_odds);
    let exponential_sum: f64 = sorted_raw_odds.iter().map(|(i,o)| o).sum();
    // eprintln!("{:?}",exponential_sum);
    let mut range_selection: f64 = rand::thread_rng().gen_range(0.,exponential_sum);
    // eprintln!("{:?}",range_selection);
    for (i,o) in &sorted_raw_odds {
        range_selection -= o;
        if range_selection < 0. {
            return Some(*i)
        }
    }
    return sorted_raw_odds.get(0).map(|(i,o)| *i)
}

fn read_matrix(location:&str) -> Result<Array<f64,Ix2>,Box<Error>> {

    let element_array_file = File::open(location)?;
    let mut element_array_lines = io::BufReader::new(&element_array_file).lines();

    let mut outer_vector: Vec<Vec<f64>> = Vec::new();
    for (i,line) in element_array_lines.by_ref().enumerate() {
        let mut element_vector = Vec::new();
        let element_line = line?;
        for (j,e) in element_line.split_whitespace().enumerate() {
            match e.parse::<f64>() {
                Ok(exp_val) => {
                    element_vector.push(exp_val);
                },
                Err(msg) => {
                    if e != "nan" && e != "NAN" {
                        println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        println!("Cell content: {:?}", e);
                    }
                    element_vector.push(f64::NAN);
                }
            }
        }
        outer_vector.push(element_vector);
        // if i % 100 == 0 {
        //     println!("{}", i);
        // }

    };
    let (r,c) = (outer_vector.len(),outer_vector.get(0).unwrap_or(&vec![]).len());
    println!("===========");
    println!("Read {} lines, first line has {} elements",r,c);
    let mut array = Array::zeros((r,c));
    for (mut row,vector) in array.axis_iter_mut(Axis(0)).zip(outer_vector) {
        row.assign(&Array::from_vec(vector));
    }
    Ok(array)
}

pub struct Transition {
    state:(Option<usize>,Option<usize>),
    node:(usize,usize),
    oracle:bool
}


#[cfg(test)]
pub mod tree_braider_tests {

    use super::*;

    pub fn iris_matrix() -> Array<f64,Ix2> {
        read_matrix("../testing/iris.trunc").unwrap()
    }

    pub fn iris_forest() -> Vec<MarkovNode> {
        // MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/small_iris_forest/").unwrap())
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/iris_forest/").unwrap())
    }

    pub fn gene_forest() -> Vec<MarkovNode> {
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/johnston_forest/").unwrap())
    }

    pub fn iris_model() -> IHMM {
        let forest = iris_forest();
        let model = IHMM::new(forest);
        model
    }

    pub fn gene_model() -> IHMM {
        let forest = gene_forest();
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

    #[test]
    fn test_markov_partition_iris() {
        let mut model = iris_model();
        model.initialize(5);
        model.sweep();
        eprintln!("{:?}",model.hidden_states.len());
        assert_eq!(model.hidden_states.len(),5);
        panic!()
    }

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
        panic!();
    }
    //
    #[test]
    fn test_markov_multipart() {
        let mut model = iris_model();
        // let mut model = gene_model();
        model.initialize(20);
        for state in &model.hidden_states {
            eprintln!("Population: {:?}",state.nodes.len());
            eprintln!("MEANS");
            eprintln!("{:?}",state.emission_model.means());
            eprintln!("PDET");
            eprintln!("{:?}",state.emission_model.pdet());
        }
        for i in 0..100000 {
            model.sweep();
            for state in &model.hidden_states {
                eprintln!("{:?}",state);
            //     eprintln!("Population: {:?}",state.nodes.len());
            //     eprintln!("PDET:{:?}",state.emission_model.pdet());
            //     eprintln!("MEANS");
            //     eprintln!("{:?}",state.emission_model.means());
            //     eprintln!("VARIANCES");
            //     eprintln!("{:?}",state.emission_model.variances());
            }
            eprintln!("###############################");
            eprintln!("###############################");
            eprintln!("############   {:?}   #############",i);
            eprintln!("###############################");
            eprintln!("###############################");
            eprintln!("Populations:{:?}",model.hidden_states.iter().map(|hs| hs.nodes.len()).collect::<Vec<usize>>());
            eprintln!("PDET:{:?}",model.hidden_states.iter().map(|hs| *hs.emission_model.pdet()).collect::<Vec<f64>>());
            eprintln!("ORACLE MIXTURE:{:?}",model.oracle_transition_model);
        }
        // for state in &model.hidden_states {
        //     eprintln!("{:?}", state);
        // }
        // assert_eq!(model.hidden_states.len(),5);
        panic!();
    }

}




















//
