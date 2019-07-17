use std::io;
use std::f64;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::stdin;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;
use rayon::prelude::*;

use trees::node::StrippedNode;
use crate::MarkovNode;

pub struct Parameters {

        auto: bool,
        report_address: String,
        processor_limit: usize,

        nodes: Vec<StrippedNode>,
        node_locations: Vec<String>,

        gain_clustering: bool,

        sweeps: Option<usize>,
        initial_states: Option<usize>,

    }

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {

            auto: false,

            report_address: "./".to_string(),
            processor_limit: 1,

            nodes: vec![],
            node_locations: vec![],

            gain_clustering: false,

            sweeps: None,
            initial_states: None,

        };
        arg_struct
    }

    pub fn read<T: Iterator<Item = String>>(args: &mut T) -> Parameters {

        let mut arg_struct = Parameters::empty();

        let mut supress_warnings = false;
        let mut continuation_flag = false;
        let mut continuation_argument: String = "".to_string();

        while let Some((i,arg)) = args.enumerate().next() {
            if arg.clone().chars().next().unwrap_or('_') == '-' {
                continuation_flag = false;

            }
            match &arg[..] {
                "-sw" | "-suppress_warnings" => {
                    if i!=1 {
                        println!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                    supress_warnings = true;
                },
                // "-auto" | "-a"=> {
                //     arg_struct.auto = true;
                //     arg_struct.auto()
                // },
                // "-stdin" => {
                //     let single_array = Some(read_standard_in());
                // }
                "-n" | "-nodes" | "-f" | "-forest" | "-t" | "-trees" => {
                    continuation_flag = true;
                    continuation_argument = arg.clone();
                },
                "-p" | "-processors" | "-threads" => {
                    arg_struct.processor_limit = args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit");
                    rayon::ThreadPoolBuilder::new().num_threads(arg_struct.processor_limit).build_global().unwrap();
                    std::env::set_var("OMP_NUM_THREADS",format!("{}",arg_struct.processor_limit));
                },
                "-s" | "-sweeps" => {
                    arg_struct.sweeps = Some(args.next().expect("Error parsing sweep limit").parse::<usize>().expect("Error parsing sweep limit"));
                },
                "-is" | "-initial_states" | "-states" => {
                    arg_struct.initial_states = Some(args.next().expect("Error parsing initial states").parse::<usize>().expect("Error parsing initial states"));
                },
                "-o" | "-output" => {
                    arg_struct.report_address = args.next().expect("Error processing output destination")
                },
                "-gain" | "-gain_clustering" => {
                    arg_struct.gain_clustering = true;
                }


                &_ => {
                    if continuation_flag {
                        match &continuation_argument[..] {
                            "-n" | "-nodes" | "-f" | "-forest" | "-t" | "-trees" => {
                                arg_struct.node_locations.push(arg);
                            }
                            &_ => {
                                panic!("Continuation flag set but invalid continuation argument, debug arg parse!");
                            }
                        }
                    }
                    else if !supress_warnings {
                        eprintln!("Warning, detected unexpected argument:{}. Ignoring, press enter to continue, or CTRL-C to stop. Were you trying to input multiple arguments? Only some options take multiple arguments. Watch out for globs(*, also known as wild cards), these count as multiple arguments!",arg);
                        stdin().read_line(&mut String::new());
                    }
                }

            }

        }

        arg_struct.node_locations.sort();
        let s_nodes: Vec<StrippedNode> = arg_struct.node_locations.iter().map(|l| StrippedNode::from_file(l).expect("Failed to read node")).collect();
        arg_struct.nodes = s_nodes;

        arg_struct

    }

}

pub fn interpret(arg_iter: &mut std::env::Args) {

    let parameters = Parameters::read(arg_iter);

    let sweeps = parameters.sweeps;
    let states = parameters.initial_states;
    let output = parameters.report_address;

    let nodes = parameters.nodes;

    let gain = parameters.gain_clustering;

    crate::IHMM::cluster_to_location(&nodes[..],&output,sweeps,states,gain);

}












//
