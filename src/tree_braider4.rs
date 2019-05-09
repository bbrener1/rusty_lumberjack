use std::collections::{HashMap,HashSet};
use node::StrippedNode;
use rank_table::{Feature,Sample};
use serde_json;

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;

use f64::consts::E;
use f64::consts::PI;

use ndarray as nd;
use ndarray_linalg as ndl;
use num_traits;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::solveh::{InverseH,DeterminantH};

use multivariate_normal::MVN;
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

struct IHMM {
    beta: f64,
    gamma: f64,
    hidden_states: Vec<HiddenState>,
    nodes: Vec<MarkovNode>,
    mixture_model: Dirichlet,
}


impl MarkovNode {

    pub fn encode(nodes:Vec<MarkovNode>) -> Result<(Array<f64,Ix2>,Array<bool,Ix2>),std::io::Error> {
        let mut features = HashSet::new();
        for node in &nodes {
            for feature in &node.features {
                features.insert(feature);
            }
        }
        if features.iter().any(|f| *f.index() > features.len() + 1) {
            return Err(std::io::Error::new(std::io::ErrorKind::Other,"Failed to read/write all features"));
        }

        let mut data: Array<f64,Ix2> = Array::zeros((nodes.len(),features.len()));
        let mut mask: Array<bool,Ix2> = Array::from_shape_fn((nodes.len(),features.len()),|_| false);

        for (i,node) in nodes.iter().enumerate() {
            for (feature,value) in node.features.iter().zip(node.emissions.iter()) {
                data[[i,*feature.index()]] = *value;
                mask[[i,*feature.index()]] = true;
            }
        }

        Ok((data,mask))
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
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("./testing/iris_forest/").unwrap())
    }

    #[test]
    fn test_markov_import() {
        eprintln!("Test readout");
        MarkovNode::from_stripped_vec(&StrippedNode::from_location("./testing/iris_forest/").unwrap());
    }

    #[test]
    fn test_markov_encoding_iris() {
        let nodes = iris_forest();
        let (data,mask) = MarkovNode::encode(nodes).unwrap();
        eprintln!("{:?}",data.dim());
        eprintln!("{:?}",mask.dim());
        assert_eq!(data.dim().1,4);
        assert_eq!(mask.dim().1,4);
        assert_eq!(data.dim(),mask.dim());
    }
}




















//
