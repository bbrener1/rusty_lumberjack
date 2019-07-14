// #![feature(test)]
//
// extern crate test;
// use test::Bencher;

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
// extern crate intel_mkl_src;
// extern crate openblas_src;
extern crate blas_src;
extern crate trees;
extern crate ihmm;

use std::env;

fn main() {

    // manual_testing::test_command_predict_full();


    let mut arg_iter = env::args();

    let command_literal = arg_iter.next().unwrap();

    let command_top = arg_iter.next().expect("Empty command?");

    match command_literal.as_str() {

        "construct" | "predict" | "combined" => {
            trees::io::interpret(&command_literal,&mut arg_iter);
        },
        "analyze" => {
            ihmm::io::interpret(&mut arg_iter);
        },
        _ => {
            panic!("Invalid top level command, please use 'construct','predict','combined', or 'analyze'");
        },
    }
}
