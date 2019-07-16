
extern crate blas_src;

#[macro_use(array,azip,s)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate trees;
extern crate num_traits;
extern crate serde_json;
extern crate rand;
extern crate rayon;

use rayon::prelude::*;

mod dirichlet;
mod mini_multi;
mod multivariate_normal;
pub mod io;

use trees::node::StrippedNode;
use trees::{Feature,Sample,Prerequisite,gn_argmax};

use std::collections::{HashMap,HashSet};

use std::io::Write;
use std::io::prelude::*;

use std::io::Read;
use std::error::Error;
use std::fmt::Debug;


use std::fs;
use std::fs::OpenOptions;
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
use ndarray_linalg::svd::SVD;

use std::f64::EPSILON;

use rand::{thread_rng,Rng};
use rand::distributions::{Distribution,Binomial};

// use multivariate_normal::MVN;
// NOTE: Minimulti is only appropriate when working on PCA-type data. Data must be orthonormal.
use mini_multi::MVN;
use multivariate_normal::{array_mask,array_mask_axis,array_double_select,array_double_mask};
use dirichlet::{SymmetricDirichlet,Categorical};

const G_REDUCTION: usize = 3;

pub struct MarkovNode {
    index: usize,
    oracle: bool,
    hidden_state: Option<usize>,
    parent: Option<usize>,
    sister: Option<usize>,
    children: Option<(usize,usize)>,
    samples: Vec<Sample>,
    features: Vec<Feature>,
    emissions: Vec<f64>,
}

#[derive(Clone,Debug)]
struct HiddenState {
    nodes: Vec<usize>,
    emission_model: MVN,
}

impl HiddenState {

    fn blank(features:usize,states:usize) -> HiddenState {
        let emission_model = MVN::identity_prior(1, features);
        let mut potential_states = vec![None];
        potential_states.extend((0..states).map(|i| Some(i)));
        let nodes = vec![];
        HiddenState{
            nodes,
            emission_model,
        }
    }

    fn feature_log_likelihood(&self,data:&ArrayView<f64,Ix1>) -> f64 {
        self.emission_model.log_likelihood(data)
    }

    fn set_emission_model(&mut self, model:MVN) {
        self.emission_model = model;
    }

}

pub struct IHMM {
    nodes: Vec<MarkovNode>,
    emissions: Array<f64,Ix2>,
    beta: NonZeroUsize,
    gamma: NonZeroUsize,
    beta_e: f64,
    parent_transition_log_odds: Array<f64,Ix2>,
    child_transition_log_odds: Array<f64,Ix2>,
    parent_oracle_log_odds: Array<f64,Ix2>,
    child_oracle_log_odds: Array<f64,Ix2>,
    prior_emission_model: MVN,
    hidden_states: Vec<HiddenState>,
    reduction: usize,
}

impl IHMM {
    fn new(nodes:Vec<MarkovNode>) -> IHMM {

        // let emissions = MarkovNode::reduced_encode(&nodes);
        let emissions = MarkovNode::sample_encode(&nodes);

        let features = emissions.dim().1;

        IHMM {
            beta: NonZeroUsize::new(5).unwrap(),
            gamma: NonZeroUsize::new(1).unwrap(),
            beta_e: 0.,
            hidden_states: vec![],
            prior_emission_model: MVN::identity_prior(1, features),
            parent_transition_log_odds: Array::zeros((0,0)),
            child_transition_log_odds: Array::zeros((0,0)),
            parent_oracle_log_odds: Array::ones((0,0)),
            child_oracle_log_odds: Array::ones((0,0)),
            nodes: nodes,
            emissions: emissions,
            reduction: G_REDUCTION,
        }
    }

    fn from_stripped(nodes:&[StrippedNode],gain:bool) -> IHMM {

        let m_nodes = MarkovNode::from_stripped_vec(nodes,gain);
        IHMM::new(m_nodes)
    }

    fn from_location(location:&str,gain:bool) -> IHMM {
        let s_nodes = StrippedNode::from_location(location).expect("Failed to read in nodes");
        IHMM::from_stripped(&s_nodes,gain)
    }

    fn cluster(nodes:&[StrippedNode],sweeps:Option<usize>,states:Option<usize>,gain:bool) -> Vec<usize> {
        let mut ihmm = IHMM::from_stripped(nodes,gain);
        ihmm.initialize(states.unwrap_or(10));
        for i in 0..sweeps.unwrap_or(1000) {
            eprintln!("Sweep {}",i);
            ihmm.sweep();
        };
        let represented_states = ihmm.represented_states();
        ihmm.node_states().into_iter().map(|s| s.unwrap_or(represented_states.len())).collect()
    }

    fn cluster_to_location(nodes:&[StrippedNode],location:&str,sweeps:Option<usize>,states:Option<usize>,gain:bool) -> Result<(),std::io::Error> {
        let mut ihmm = IHMM::from_stripped(nodes,gain);
        ihmm.initialize(states.unwrap_or(10));
        for i in 0..sweeps.unwrap_or(1000) {
            eprintln!("Sweep {}",i);
            ihmm.sweep();
        };
        let represented_states = ihmm.represented_states();
        let node_states = ihmm.node_states().into_iter().map(|s| s.unwrap_or(represented_states.len())).collect();
        let transitions = ihmm.compute_transition_matrix(&ihmm.get_transitions(&ihmm.live_indices()),false) + ihmm.compute_transition_matrix(&ihmm.get_transitions(&ihmm.live_indices()),true);
        write_array(&node_states, &format!("{}.cluster",location))?;
        write_array(&transitions, &format!("{}.transitions",location))
    }

    fn initialize(&mut self,states:usize) {
        eprintln!("Initializing");
        let features = self.emissions.dim().1;
        self.reduction = features.min(self.reduction);
        for _ in 0..states {
            self.hidden_states.push(HiddenState::blank(features,states));
        }
        eprintln!("Estimating prior");
        self.prior_emission_model = self.estimate_prior_features();
        for ni in self.live_indices() {
            self.nodes[ni].hidden_state = Some(thread_rng().gen_range(0,states));
            self.nodes[ni].oracle = rand::random::<f64>() < (1./(states as f64));
        }
        eprintln!("Estimating transitions");
        self.compute_transition_model();
        self.estimate_states();
    }

    fn sample_node_state(&self, node_index: usize) -> Option<usize> {

        let emissions = self.emissions.row(node_index);
        let node = &self.nodes[node_index];
        let ps = node.parent.map(|pi| self.nodes[pi].hidden_state).unwrap_or(None);
        let cls = self.nodes[node.children?.0].hidden_state;// PARENT XX CHILD SWITCH
        let crs = self.nodes[node.children?.1].hidden_state;
        // eprintln!("Computing feature log likelihoods");

        let mut state_log_odds = Vec::with_capacity(self.hidden_states.len() + 1);

        eprint!("{:?}:[",node_index);
        for (si,state) in self.hidden_states.iter().enumerate() {

            let feature_log_odds = state.feature_log_likelihood(&emissions);

            let mut mixture_log_odds = 0.;

            mixture_log_odds += self.parent_transition_log_odds[[ps.unwrap_or(self.hidden_states.len()),si]]; // PARENT XX CHILD SWITCH


            mixture_log_odds += self.child_transition_log_odds[[cls.unwrap_or(self.hidden_states.len()),si]];
            mixture_log_odds += self.child_transition_log_odds[[crs.unwrap_or(self.hidden_states.len()),si]];

            mixture_log_odds /= 2.;

            eprint!("({:?},",feature_log_odds);
            eprint!("{:?}),",mixture_log_odds);
            state_log_odds.push(feature_log_odds + mixture_log_odds);
        }
        eprint!("]\n");

        let new_state_log_odds = {
            let new_state_feature_log_odds = self.new_state_feature_log_odds(&emissions);
            // let new_state_mixture_log_odds = self.new_state_mixture_log_odds(ps); // PARENT XX CHILD SWITCH
            let new_state_mixture_log_odds = self.new_state_mixture_log_odds(node_index) + self.new_state_mixture_log_odds(node_index);
            new_state_feature_log_odds + new_state_mixture_log_odds
        };

        state_log_odds.push(new_state_log_odds);

        let log_max: f64 = state_log_odds.iter().fold(std::f64::NEG_INFINITY,|acc,o| f64::max(acc,*o));
        state_log_odds = state_log_odds.iter().map(|o| o - log_max).collect();
        // state_log_odds = state_log_odds.iter().map(|o| o * 0.5).collect();
        assert!(log_max.is_finite());
        eprintln!("LOG_ODDS:{:?}",state_log_odds);
        let state = sample_log_odds(state_log_odds);
        eprintln!("S:{:?}",state);
        state
    }

    pub fn sample_oracle_transition(&self, index:usize) -> bool {

        let mut log_odds = 0.;
        let node = &self.nodes[index];
        let null_index = self.hidden_states.len();
        let node_state = node.hidden_state.unwrap_or(null_index);
        if let Some(parent_index) = node.parent {
            let parent = &self.nodes[parent_index];
            let parent_state = parent.hidden_state.unwrap_or(null_index);
            log_odds += self.parent_oracle_log_odds[[parent_state,node_state]];
        }
        if let Some((c1i,c2i)) = node.children {
            let c1 = &self.nodes[c1i];
            let c2 = &self.nodes[c2i];
            let c1_state = c1.hidden_state.unwrap_or(null_index);
            let c2_state = c2.hidden_state.unwrap_or(null_index);
            log_odds += self.child_oracle_log_odds[[node_state,c1_state]];
            log_odds += self.child_oracle_log_odds[[node_state,c2_state]];
        }
        let oracle_transition_probability = log_odds.exp2() / (1. + log_odds.exp2());
        rand::random::<f64>() < oracle_transition_probability

    }


    fn resample_states(&mut self) {
        // let hidden_states: Vec<Option<usize>> = self.live_indices().into_iter().map(|ni| {
        let hidden_states: Vec<Option<usize>> = self.live_indices().into_par_iter().map(|ni| {
            self.sample_node_state(ni)
        }).collect();
        for (ni,state) in self.live_indices().into_iter().zip(hidden_states) {
            self.nodes[ni].hidden_state = state;
        }
    }


    fn resample_hyperparameters(&mut self) {
        self.beta = NonZeroUsize::new(self.resample_beta(None)).unwrap();
        self.gamma = NonZeroUsize::new(self.resample_gamma(None)).unwrap();
    }

    fn resample_beta(&mut self,maximum_option:Option<usize>) -> usize {
        let maximum = maximum_option.unwrap_or(10);
        let ln_beta: Vec<f64> = (1..(self.nodes.len()*2)).map(|v| (v as f64).ln()).collect();
        let ln_beta_cmsm: Vec<f64> = ln_beta.iter().scan(0., |acc,v| {*acc += v; Some(*acc)}).collect();
        let transition_matrix = self.compute_transition_matrix(&self.get_transitions(&self.live_indices()), false);

        let ki = transition_matrix.mapv(|v| if v > 0 {1} else {0}).sum_axis(Axis(0));
        let ni = transition_matrix.sum_axis(Axis(0));
        let potential_states = self.hidden_states.len() + 1;

        let likelihood = |beta| {
            let mut cml = 0.;
            for i in 0..potential_states {
                cml += ln_beta[beta] * ki[i] as f64 + (ln_beta_cmsm[ni[i]] - ln_beta_cmsm[beta])
            }
            cml -= beta as f64;
            cml
        };

        let likelihoods = (1..maximum).map(|beta| likelihood(beta));

        // eprintln!("LIKELIHOODS:{:?}",likelihoods.clone().collect::<Vec<f64>>());

        let mut maximum_beta = gn_argmax(likelihoods).unwrap() + 1;

        if maximum_beta == maximum && maximum < self.nodes.len() {
            maximum_beta = self.resample_beta(Some(maximum * 10));
        }

        eprintln!("BETA:{}",maximum_beta);

        maximum_beta
    }

    fn resample_gamma(&mut self,maximum_option:Option<usize>) -> usize {
        let maximum = maximum_option.unwrap_or(10);
        let ln_gamma: Vec<f64> = (1..(self.nodes.len()*2)).map(|v| (v as f64).ln()).collect();
        let ln_gamma_cmsm: Vec<f64> = ln_gamma.iter().scan(0., |acc,v| {*acc += v; Some(*acc)}).collect();
        let transition_matrix = self.compute_transition_matrix(&self.get_transitions(&self.live_indices()), true);

        let k = self.hidden_states.len() + 1;
        let to: usize = transition_matrix.iter().sum();

        let likelihood = |gamma| {
            (k*gamma) as f64 + (ln_gamma_cmsm[to + gamma] - ln_gamma_cmsm[gamma]) - gamma as f64
        };

        let likelihoods = (1..maximum).map(|gamma| likelihood(gamma));

        // eprintln!("LIKELIHOODS:{:?}",likelihoods.clone().collect::<Vec<f64>>());

        let mut maximum_gamma = gn_argmax(likelihoods).unwrap() + 1;

        if maximum_gamma == maximum && maximum < self.nodes.len() {
            maximum_gamma = self.resample_gamma(Some(maximum * 10));
        }

        eprintln!("GAMMA:{}",maximum_gamma);

        maximum_gamma

    }


    fn resample_oracles(&mut self) {
        let node_states = self.nodes.iter().map(|n| n.hidden_state);
        let parent_states: Vec<Option<usize>> = self.get_node_parent_states(&(0..self.nodes.len()).collect::<Vec<usize>>());
        let transitions = node_states.zip(parent_states.into_iter());
        let oracles: Vec<bool> = self.live_indices().into_iter().map(|i| self.sample_oracle_transition(i)).collect();
        for (ni,oracle) in oracles.into_iter().enumerate() {
            self.nodes[ni].oracle = oracle;
        }
    }

    fn sweep(&mut self) {
        self.resample_states();
        self.resample_oracles();
        self.remove_unrepresented_states();
        self.estimate_states();
        self.resample_hyperparameters();
        self.report();
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

        let represented_states: Vec<Option<usize>> = self.represented_states().into_iter().filter(|s| s.is_some()).collect();

        // eprintln!("Estimating states:{:?}",represented_states);

        // let new_states: Vec<HiddenState> = represented_states.par_iter().map(|state| {
        let new_states: Vec<HiddenState> = represented_states.iter().map(|state| {
            let indices = self.state_indices(*state);
            let state_emission_model = self.estimate_emissions(&indices).unwrap();
            let state = HiddenState {
                nodes:indices,
                emission_model: state_emission_model,
            };
            state
        }).collect();

        self.hidden_states = new_states;

        self.compute_transition_model();

        for (i,s) in self.hidden_states.iter().enumerate(){
            eprintln!("NM{:?}:{:?}",i,s.emission_model.means());
        }

    }

    fn estimate_emissions(&self, indices:&[usize]) -> Result<MVN,LinalgError> {

        // eprintln!("Estimating emissions");

        let data = self.emissions.select(Axis(0),indices);

        // eprintln!("Data selected");

        let mut emission_model = self.prior_emission_model.clone();
        emission_model.set_samples(1);
        for t in 0..10 {
            if let Err(lapak_err) = emission_model.estimate(&data.view()) {
            // if let Err(lapak_err) = emission_model.uninformed_estimate(&data.view()) {
                eprintln!("EST_ERR:{:?}",lapak_err);
            }
            else { break }
            if t > 3 {
                eprintln!("WARNING: Failed to estimate");
                emission_model.mini_estimate(&data.view());
            }
            // emission_model.uninformed_estimate(&data.view())?;
        }
        // eprintln!("EME:{:?}",emission_model.means());

        Ok(emission_model)
    }

    fn estimate_population_model(&self) -> SymmetricDirichlet<Option<usize>> {
        let mut state_populations:HashMap<Option<usize>,usize> = self.represented_states().into_iter().map(|s| (s,0)).collect();
        for (state,count) in state_populations.iter_mut() {
            *count = self.state_indices(*state).len();
        }
        let population_model = SymmetricDirichlet::from_map(state_populations, self.beta);
        population_model
    }


    fn estimate_prior_features(&mut self) -> MVN {

        let indices = self.live_indices();
        self.estimate_emissions(&indices).unwrap()
    }

    fn new_state_feature_log_odds(&self,data:&ArrayView<f64,Ix1>) -> f64 {
        self.prior_emission_model.log_likelihood(data)
    }

    fn new_state_mixture_log_odds(&self,index:usize) -> f64 {
        let mut log_odds = 0.;
        let node = &self.nodes[index];
        let null_index = self.hidden_states.len();
        let new_state = self.hidden_states.len();

        if let Some(parent_index) = node.parent {
            let parent = &self.nodes[parent_index];
            let parent_state = parent.hidden_state.unwrap_or(null_index);
            log_odds += self.parent_transition_log_odds[[parent_state,new_state]];
        }
        if let Some((c1i,c2i)) = node.children {
            let c1 = &self.nodes[c1i];
            let c2 = &self.nodes[c2i];
            let c1_state = c1.hidden_state.unwrap_or(null_index);
            let c2_state = c2.hidden_state.unwrap_or(null_index);
            log_odds += self.child_transition_log_odds[[new_state,c1_state]];
            log_odds += self.child_transition_log_odds[[new_state,c2_state]];
        }
        log_odds
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

        // Here we return a list of transitions by state from parent to both children
        // usize 1 is parent state, usize 2 is child state

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
            else {
                transitions.push((node_state,None,false));
            }
            if let Some(pi) = node.parent {
                let parent = &self.nodes[pi];
                transitions.push((parent.hidden_state,node_state,node.oracle));
            }
            else {
                transitions.push((None,node_state,node.oracle));
            }
        }
        transitions
    }


    fn compute_transition_matrix(&self,transitions:&[(Option<usize>,Option<usize>,bool)],oracle:bool) -> Array<usize,Ix2> {

        // Returns a matrix of transition counts i,j where i is parent state, j is child state,
        // Null state is last row and last column.

        let hidden_states = self.hidden_states.len();
        let mut transition_matrix: Array<usize,Ix2> = Array::zeros((hidden_states+1,hidden_states+1));
        for (s1,s2,o) in transitions {
            if *o == oracle {

                let s1i = s1.unwrap_or(hidden_states);
                let s2i = s2.unwrap_or(hidden_states);
                transition_matrix[[s1i,s2i]] += 1;

            }

        }
        transition_matrix
    }


    fn compute_transition_model(&mut self) {

        // We have to compute two things to completely represent a transition model:
        // The odds of a given state given that we know the states of the child nodes,
        // The odds of a given state given that we know the states of the parent node,

        // For each type of transition, we have to know:

        // The odds of transitioning from state i to state j directly,
        // The odds of transitioning from state i to the first oracle,
        // The odds of transitioning from the oracle to state j,

        // Complicating this fact is the existence of the null state,
        // And the fact that a new state is not represented.
        // Live nodes cannot transition to the null state

        // Direct transition matrices are organized as
        // Row: Child state
        // Column: Destination state

        // As such:
        // Rows: represented states, plus null state
        // Columns: represented states: plus new state

        // First we establish the currently represented states:
        // NB self.represented states includes the null state

        eprintln!("Computing transitions");

        let represented_states = self.represented_states();

        // Now we initialize the matrix containing state-state transition log odds.

        let mut direct_transition_odds: Array<f64,Ix2> = Array::ones((represented_states.len(),represented_states.len()));
        let mut oracle_transition_odds: Array<f64,Ix1> = Array::ones(represented_states.len());

        // We expect the input transition matrices to have the null state as the last colum and the last row

        let mut direct_transition_matrix = self.compute_transition_matrix(&self.get_transitions(&self.live_indices()), false);
        let mut oracle_transition_matrix = self.compute_transition_matrix(&self.get_transitions(&self.live_indices()), true);

        // eprintln!("DTM:{:?}",direct_transition_matrix.dim());
        // eprintln!("OTM:{:?}",oracle_transition_matrix.dim());
        // eprintln!("RS:{:?}",represented_states);

        // We set the last column to beta in order to analyze the oracle probabilities,
        // Previously it represented transitions to the null state, but these are prohibited
        // However we have to set the last column of the oracle transition matrix to zero in order
        // to avoid counting gamma multiple times per represented state.

        direct_transition_matrix.slice_mut(s![..,-1]).assign(&(Array::ones(represented_states.len()) * self.beta.get()));
        oracle_transition_matrix.slice_mut(s![..,-1]).assign(&Array::zeros(represented_states.len()));

        eprintln!("Direct transition counts:");
        eprintln!("{:?}",direct_transition_matrix);
        eprintln!("Oracle transition counts:");
        eprintln!("{:?}",oracle_transition_matrix);

        // Now we need to compute the total transitions that each child state undergoes

        let direct_transition_totals = direct_transition_matrix.sum_axis(Axis(1));

        // eprintln!("Direct transition totals:");
        // eprintln!("{:?}",direct_transition_totals);

        // Now we can compute the direct transition odds for each state, as well as the oracle odds
        // Oracle odds are represented in the last column.

        for i in 0..represented_states.len() {
            for j in 0..represented_states.len() {
                direct_transition_odds[[i,j]] = direct_transition_matrix[[i,j]] as f64 / direct_transition_totals[i] as f64;
            }
        }

        // eprintln!("Direct transitions computed!");
        // eprintln!("{:?}",direct_transition_odds);

        let oracle_odds = direct_transition_odds.slice(s![..,-1]);

        // Now we have to evaluate the second layer of the transition model.

        // let mut oracle_transitions = oracle_transition_matrix.sum_axis(Axis(1)) + (Array::ones(oracle_transition_matrix.dim().1) * self.beta.get());
        let mut oracle_transition_totals = oracle_transition_matrix.sum_axis(Axis(1)) + Array::ones(oracle_transition_matrix.dim().1);
        let last_index = (oracle_transition_totals.dim() as i32 - 1).max(0) as usize;
        oracle_transition_totals[[last_index]] += self.gamma.get();
        let oracle_transition_total = oracle_transition_totals.sum();

        // eprintln!("Direct transition totals:");
        // eprintln!("{:?}",oracle_transitions);

        // Here we have a slight hack. We are representing the transition to a novel state in the last column of the matrix
        // This is because we cannot transition to a null state anyway

        let oracle_transition_odds: Array<f64,Ix1> = oracle_transition_totals.mapv(|t| t as f64 / (oracle_transition_total + self.gamma.get()) as f64);

        // eprintln!("Oracle transitions computed!");
        // eprintln!("{:?}",oracle_transition_odds);

        // Now, we can have a matrix of complete odds of transition from state i to state j.
        // Odds of direct transition + odds of oracle transition * odds of reaching the oracle.

        let mut oracle_product_matrix: Array<f64,Ix2> = Array::ones((represented_states.len(),represented_states.len()));

        for i in 0..represented_states.len() {
            oracle_product_matrix.row_mut(i).fill(oracle_odds[i]);
        };

        for i in 0..represented_states.len() {
            oracle_product_matrix.column_mut(i).mapv_inplace(|v| v * oracle_transition_odds[i]);
        }

        let log_odds = (&direct_transition_odds + &oracle_product_matrix).mapv(|v| v.log2());

        eprintln!("Log odds computed!");
        eprintln!("{:?}",log_odds);

        let oracle_log_odds = (&oracle_product_matrix / &direct_transition_odds).mapv(|v| v.log2());

        // eprintln!("Oracle probability computed!");
        // eprintln!("{:?}",oracle_probability);

        assert!(!log_odds.iter().any(|ll| ll.is_nan()));
        assert!(!oracle_log_odds.iter().any(|ll| ll.is_nan()));

        // Unfortunately at this stage, we have only half of the picture.
        // We also wish to know what the odds are of a given state, if we know the state of the child

        let mut child_direct_transition_odds: Array<f64,Ix2> = Array::ones((represented_states.len(),represented_states.len()));
        let mut child_oracle_transition_odds: Array<f64,Ix2> = Array::ones((represented_states.len(),represented_states.len()));

        // We would like to keep the lookup orientation the same for consistency
        // Eg, P(State i | Child State j) is located at i,j
        // However, this means that the location of the probabilities of a new state must be altered.
        // A child cannot be in an unrepresented state at any given sweep,
        // And a node we are sampling cannot be in the null state.

        // Given Bayes: P(A|B) = P(B|A)P(A) / P(B) => O(A|B) = O(A) * (P(B|A) / P(B|A')
        // P(State | Child State) = P(Child State | State) * P(Child State) / P(State)
        // P(Child State| Parent State) / P(Child State | Not Parent State) is already given by the transition matrices

        // let mut child_direct_transition_matrix = self.compute_transition_matrix(&self.get_transitions(&self.live_indices()), false);
        // child_direct_transition_matrix.slice_mut(s![-1,..]).assign(&(Array::ones(represented_states.len()) * self.beta.get()));
        //
        // eprintln!("Direct transition counts:");
        // eprintln!("{:?}",child_direct_transition_matrix);
        //
        // let child_transition_totals = child_direct_transition_matrix.sum_axis(Axis(0)) + Array::ones(direct_transition_matrix.dim().0);
        // let total_direct_transitions = child_transition_totals.sum();
        //
        // let child_transition_totals = &direct_transition_totals + &oracle_transition_totals;
        // let total_transitions = child_transition_totals.sum();

        for child_state in 0..represented_states.len() {
            for parent_state in 0..represented_states.len() {

                // O(Parent State | Child State) = O(Parent State) * (P(Child|Parent) / P(Child | Not Parent))

                let op = direct_transition_totals[[parent_state]] as f64 / (direct_transition_totals.sum() - direct_transition_totals[[parent_state]]) as f64;
                let on = direct_transition_matrix[[parent_state,child_state]] as f64 / (direct_transition_matrix.slice(s![..,child_state]).sum() - direct_transition_matrix[[parent_state,child_state]]) as f64;

                child_direct_transition_odds[[parent_state,child_state]] = op * on;
                // // P(Child State | Parent State)
                // let pcp = direct_transition_matrix[[parent_state,child_state]] as f64 / direct_transition_totals[[parent_state]] as f64;
                // // P(Child State)
                // let pc = child_transition_totals[[child_state]] as f64 / total_transitions as f64;
                // // P(State)
                // let ps = child_transition_totals[[parent_state]] as f64 / total_transitions as f64;
                // let ppc = (pcp * pc) / ps;
                // child_direct_transition_odds[[parent_state,child_state]] = ppc / (1. - ppc);
            }
        };

        eprintln!("Child transfer odds:{:?}",child_direct_transition_odds);

        let child_oracle_probability = child_direct_transition_odds.slice(s![-1,..]).mapv(|v| v/(1.+v));

        eprintln!("Child oracle probability:{:?}",child_oracle_probability);

        // Given Bayes: P(A|B) = P(B|A)P(A) / P(B) => O(A|B) = O(A) * (P(B|A) / P(B|A')
        // O(Oracle | Child State) = O(Oracle) * (P(Child State | Oracle) / P(Child State | Not Oracle))
        // P(Oracle | Child State) = P(Child State | Oracle) * P(Child State) / P(Oracle)

        for child_state in 0..represented_states.len() {
            for parent_state in 0..represented_states.len() {
                let oracle_index = (direct_transition_totals.dim() as i32 - 1).max(0) as usize;
                let oo = direct_transition_totals[[oracle_index]] as f64 / (direct_transition_totals.sum() - direct_transition_totals[oracle_index]) as f64;
                let co = oracle_transition_totals[[child_state]] as f64 / oracle_transition_total as f64;
                let cno = direct_transition_totals[[child_state]] as f64 / direct_transition_totals.sum() as f64;

                child_oracle_transition_odds[[parent_state,child_state]] = oo * (co/cno);

                // // P(Child State | Oracle)
                // let pco = oracle_transition_totals[[child_state]] as f64 / oracle_transition_total as f64;
                // // P(Child State)
                // let pc = child_transition_totals[[child_state]] as f64 / total_transitions as f64;
                // // P(Oracle)
                // let po = child_oracle_probability[[child_state]];
                // let ppc = (pco * pc) / po;
                // child_oracle_transition_odds[[parent_state,child_state]] = ppc / (1. - ppc);
            }
        };

        let child_oracle_log_odds = child_oracle_transition_odds.mapv(|v| v.log2());
        
        let mut child_oracle_product_matrix: Array<f64,Ix2> = Array::ones((represented_states.len(),represented_states.len()));

        for child_state in 0..represented_states.len() {
            for parent_state in 0..represented_states.len() {
                child_oracle_product_matrix[[parent_state,child_state]] = child_oracle_transition_odds[[parent_state,child_state]] * oracle_odds[[parent_state]]
            }
        };

        eprintln!("Child oracle product matrix{:?}", child_oracle_product_matrix);

        eprintln!("Sum:{:?}",(&child_direct_transition_odds + &child_oracle_product_matrix));

        let child_log_odds = (child_direct_transition_odds + child_oracle_product_matrix).mapv(|v| v.log2());

        self.parent_transition_log_odds = log_odds;
        self.parent_oracle_log_odds = oracle_log_odds;
        self.child_transition_log_odds = child_log_odds;
        self.child_oracle_log_odds = child_oracle_log_odds;

        eprintln!("Transitions:{:?}",self.parent_transition_log_odds);
        eprintln!("{:?}",self.child_transition_log_odds);

    }

    fn represented_states(&self) -> Vec<Option<usize>> {
        let mut state_set: HashSet<Option<usize>> = self.nodes.iter().map(|n| n.hidden_state).collect();
        state_set.insert(None);
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

    pub fn node_states(&self) -> Vec<Option<usize>> {
        self.nodes.iter().map(|node| node.hidden_state).collect()
    }

    fn state_indices(&self, state:Option<usize>) -> Vec<usize> {
        self.nodes.iter().filter(|n| n.hidden_state == state).map(|n| n.index).collect()
    }


    pub fn live_indices(&self) -> Vec<usize> {
        // This function should be altered based on whether or not you are using child or
        // parent inheritance

        // self.nodes.iter().filter(|n| n.children.is_some()).map(|n| n.index).collect()
        self.nodes.iter().map(|n| n.index).collect()
    }

    pub fn report(&self) {
        eprintln!("######################");
        for (i,state) in self.hidden_states.iter().enumerate() {
            eprintln!("HS{} Means:{:?}",i,state.emission_model.means());
        }
        eprintln!("Populations:{:?}",self.hidden_states.iter().map(|hs| hs.nodes.len()).collect::<Vec<usize>>());
        eprintln!("Beta:{}",self.beta.get());
        eprintln!("Gamma:{}",self.gamma.get());
        eprintln!("")
    }


}


impl MarkovNode {

    pub fn encode(nodes:&Vec<MarkovNode>) -> Array<f64,Ix2> {
        let mut features = HashSet::new();
        for node in nodes {
            for feature in &node.features {
                features.insert(feature);
            }
        }
        if features.iter().any(|f| *f.index() > features.len() + 1) {
            panic!("Not all features read correctly, missing indices");
        }

        let mut data: Array<f64,Ix2> = Array::zeros((nodes.len(),features.len()));

        for (i,node) in nodes.iter().enumerate() {
            for (feature,value) in node.features.iter().zip(node.emissions.iter()) {
                data[[i,*feature.index()]] = *value;
            }
        }

        data
    }

    pub fn reduced_encode(nodes:&Vec<MarkovNode>) -> Array<f64,Ix2> {
        let mut features = HashSet::new();
        for node in nodes {
            for feature in &node.features {
                features.insert(feature);
            }
        }
        if features.iter().any(|f| *f.index() > features.len() + 1) {
            panic!("Not all features read correctly, missing indices");
        }

        let mut data: Array<f64,Ix2> = Array::zeros((nodes.len(),features.len()));

        for (i,node) in nodes.iter().enumerate() {
            for (feature,value) in node.features.iter().zip(node.emissions.iter()) {
                data[[i,*feature.index()]] = *value;
            }
        }

        if let Ok((Some(u),sig_v,Some(vt))) = data.svd(true,true) {

            let reduction = features.len().min(G_REDUCTION);

            let mut sig = Array::zeros((sig_v.dim(),sig_v.dim()));
            sig.diag_mut().assign(&sig_v);

            let lower_bound = EPSILON * 1000.;

            let mut i_sig = Array::zeros((sig_v.dim(),sig_v.dim()));
            i_sig.diag_mut().assign(&sig_v.mapv(|v| if v > lower_bound {1./v} else {0.} ));

            let reduced_u = u.slice(s![..,..reduction]).to_owned();
            let mut reduced_sig: Array<f64,Ix2> = Array::zeros((reduction,reduction));
            reduced_sig.diag_mut().assign(&sig_v.iter().take(reduction).cloned().collect::<Array<f64,Ix1>>());
            let mut reduced_i_sig = Array::zeros((reduction,reduction));
            reduced_i_sig.diag_mut().assign(&reduced_sig.diag().mapv(|v| if v > lower_bound {1./v} else {0.} ));
            let reduced_vt = vt.slice(s![..reduction,..]).to_owned();

            eprintln!("Reduced SVD:{:?},{:?},{:?}",reduced_u.shape(),reduced_sig.dim(),reduced_vt.dim());

            data = data.dot(&reduced_vt.t());

            data
        }

        else {panic!();}
    }

    pub fn sample_encode(nodes:&Vec<MarkovNode>) -> Array<f64,Ix2> {
        let mut samples = HashSet::new();
        for node in nodes {
            for sample in &node.samples {
                samples.insert(sample);
            }
        }
        if samples.iter().any(|f| *f.index() > samples.len() + 1) {
            panic!("Not all samples read correctly, missing indices");
        }

        let mut data: Array<f64,Ix2> = Array::zeros((nodes.len(),samples.len()));

        for (i,node) in nodes.iter().enumerate() {
            for sample in node.samples.iter() {
                data[[i,*sample.index()]] += 1.;
            }
        }

        for node in nodes.iter() {
            if let Some(sister) = node.sister {
                for sample in node.samples.iter() {
                    data[[sister,*sample.index()]] -= 1.;
                }
            }
        }

        if let Ok((Some(u),sig_v,Some(vt))) = data.svd(true,true) {

            let reduction = samples.len().min(G_REDUCTION);

            let mut sig = Array::zeros((sig_v.dim(),sig_v.dim()));
            sig.diag_mut().assign(&sig_v);

            let lower_bound = EPSILON * 1000.;

            let mut i_sig = Array::zeros((sig_v.dim(),sig_v.dim()));
            i_sig.diag_mut().assign(&sig_v.mapv(|v| if v > lower_bound {1./v} else {0.} ));

            let reduced_u = u.slice(s![..,..reduction]).to_owned();
            let mut reduced_sig: Array<f64,Ix2> = Array::zeros((reduction,reduction));
            reduced_sig.diag_mut().assign(&sig_v.iter().take(reduction).cloned().collect::<Array<f64,Ix1>>());
            let mut reduced_i_sig = Array::zeros((reduction,reduction));
            reduced_i_sig.diag_mut().assign(&reduced_sig.diag().mapv(|v| if v > lower_bound {1./v} else {0.} ));
            let reduced_vt = vt.slice(s![..reduction,..]).to_owned();

            eprintln!("Reduced SVD:{:?},{:?},{:?}",reduced_u.shape(),reduced_sig.dim(),reduced_vt.dim());

            data = data.dot(&reduced_vt.t());

            data
        }

        else {panic!();}
    }


    pub fn from_stripped_vec(stripped:&[StrippedNode],gain:bool) -> Vec<MarkovNode> {
        let mut markov = vec![];
        for root in stripped {
            let ci = markov.len();
            markov.append(&mut MarkovNode::from_stripped_node(root, ci,gain));
        }
        markov
    }

    pub fn from_stripped_node(original:&StrippedNode,passed_index:usize,gain:bool) -> Vec<MarkovNode> {

        let mut nodes = vec![];
        let mut children = None;
        let mut index = passed_index;
        if let [ref lc,ref rc] = original.children[..] {
            let mut left_children = MarkovNode::from_stripped_node(lc,index,gain);
            let lci = left_children.last().unwrap().index;
            let mut right_children = MarkovNode::from_stripped_node(rc,lci+1,gain);
            let rci = right_children.last().unwrap().index;
            index = rci + 1;
            left_children.last_mut().unwrap().parent = Some(index);
            left_children.last_mut().unwrap().sister = Some(rci);
            right_children.last_mut().unwrap().parent = Some(index);
            right_children.last_mut().unwrap().sister = Some(lci);
            children = Some((lci,rci));
            nodes.append(&mut left_children);
            nodes.append(&mut right_children);
        }

        let parent = None;
        let sister = None;
        let samples = original.samples().to_vec();
        let features = original.features().to_vec();
        let emissions =
            if gain {
                original.local_gains().unwrap_or(&vec![0.;features.len()]).to_vec()
            }
            else {
                original.medians().to_vec()
            };

        let wrapped = MarkovNode{
            index,
            parent,
            sister,
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
    let mut element_array_lines = std::io::BufReader::new(&element_array_file).lines();

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

pub fn tsv_format<T:Debug,D:ndarray::RemoveAxis>(input:&Array<T,D>) -> String {

    input.axis_iter(Axis(0)).map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")

}

pub fn write_array<T:Debug,D:ndarray::RemoveAxis>(input:&Array<T,D>,location:&str) -> Result<(),std::io::Error> {
    let mut handle = OpenOptions::new().create(true).append(true).open(location)?;
    handle.write(tsv_format(input).as_bytes())?;

    Ok(())

}

#[cfg(test)]
pub mod tree_braider_tests {

    extern crate blas_src;
    use super::*;

    pub fn iris_matrix() -> Array<f64,Ix2> {
        read_matrix("../testing/iris.trunc").unwrap()
    }

    pub fn iris_forest() -> Vec<MarkovNode> {
        // MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/small_iris_forest/").unwrap())
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/iris_forest/").unwrap(),true)
    }

    pub fn gene_forest() -> Vec<MarkovNode> {
        // MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/johnston_forest/").unwrap())
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("../testing/nesterowa_forest/").unwrap(),true)
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
        MarkovNode::from_stripped_vec(&stripped_nodes,true);
    }

    #[test]
    fn test_markov_encoding_iris() {
        let nodes = iris_forest();
        let data = MarkovNode::encode(&nodes);
        eprintln!("{:?}",data.dim());
        assert_eq!(data.dim().1,4);
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
        model.initialize(3);
        for state in &model.hidden_states {
            eprintln!("Population: {:?}",state.nodes.len());
            eprintln!("MEANS");
            eprintln!("{:?}",state.emission_model.means());
            eprintln!("PDET");
            eprintln!("{:?}",state.emission_model.pdet());
        }
        for i in 0..1 {
            model.sweep();
            for state in &model.hidden_states {
                // eprintln!("{:?}",state);
            //     eprintln!("Population: {:?}",state.nodes.len());
            //     eprintln!("PDET:{:?}",state.emission_model.pdet());
                eprintln!("MEANS");
                eprintln!("{:?}",state.emission_model.means());
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
            eprintln!("BETA:{:?}",model.beta);
            eprintln!("Transitions:{:?}",model.parent_transition_log_odds);
            eprintln!("{:?}",model.child_transition_log_odds);
        }
        // for state in &model.hidden_states {
        //     eprintln!("{:?}", state);
        // }
        // assert_eq!(model.hidden_states.len(),5);
        panic!();
    }

}




















//
