use std::collections::{HashMap,HashSet};
use node::StrippedNode;
use rank_table::{Feature,Sample};
use serde_json;

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

struct MarkovNode {
    index: u32,
    hidden_state: Option<u32>,
    parent: Option<u32>,
    children: Option<(u32,u32)>,
    samples: Vec<Sample>,
    features: Vec<Feature>,
    emissions: Vec<f64>,
}

struct HiddenState {
    transitions: Option<Vec<u32>>,
    emission_log_odds: Option<Vec<f64>>,
    nodes: HashSet<u32>,

}

struct TransitionOracle {
    transitions: Vec<u32>,
    nodes: Vec<u32>,
}

struct IHMM {
    beta: f64,
    gamma: f64,
    hidden_states: Vec<HiddenState>,
    nodes: Vec<MarkovNode>,
}

impl MarkovNode {

    pub fn from_json(input:&str,index:Option<u32>) -> Result<Vec<MarkovNode>,Box<Error>> {
        let stripped: StrippedNode = serde_json::from_str(input)?;
        MarkovNode::from_stripped_node(&stripped, index.unwrap_or(0))
    }

    pub fn from_location(location:&str)

    pub fn from_file(location:&str,index:Option<u32>) -> Result<Vec<MarkovNode>,Box<Error>> {
        let mut json_file = File::open(location)?;
        let mut json_string = String::new();
        json_file.read_to_string(&mut json_string)?;
        MarkovNode::from_json(&json_string, index)
    }


    pub fn from_quick_encoding(index:u32,parent:Option<u32>,children:Option<(u32,u32)>,samples:Vec<Sample>,features:Vec<Feature>,emissions:Vec<f64>) -> MarkovNode {
        MarkovNode{
            index,
            parent,
            children,
            samples,
            features,
            emissions,
            hidden_state: None,

        }
    }

    pub fn from_stripped_node(original:&StrippedNode,passed_index:u32) -> Result<Vec<MarkovNode>,Box<Error>> {

        let mut nodes = vec![];
        let mut children = None;
        let mut index = passed_index;
        if let [ref lc,ref rc] = original.children[..] {
            let mut left_children = MarkovNode::from_stripped_node(lc,index)?;
            let lci = left_children.last().unwrap().index;
            let mut right_children = MarkovNode::from_stripped_node(rc,lci+1 as u32)?;
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

        Ok(nodes)
    }
}

impl IHMM {
}


#[cfg(test)]
mod tree_braider_tests {



}
