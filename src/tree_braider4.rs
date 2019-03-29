use std::collections::{HashMap,HashSet};
use node::StrippedNode;
use rank_table::{Feature,Sample};
use serde_json;


struct MarkovNode {
    index: u32,
    hidden_state: Option<u32>,
    parent: Option<u32>,
    children: Option<(u32,u32)>,
    samples: Vec<Sample>
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
    pub fn from_quick_encoding(index:u32,parent:Option<u32>,children:Option<(u32,u32)>,samples:Vec<Sample>) -> MarkovNode {
        MarkovNode{
            index,
            parent,
            children,
            samples,
            hidden_state: None,
        }
    }
}

impl IHMM {
    pub fn node_from_json(input:&str) {
        let json: serde_json::Value = serde_json::from_str(input).unwrap();

    }
}
