// #![feature(test)]
//
// extern crate test;
// use test::Bencher;

#[macro_use]
extern crate ndarray_linalg;
// extern crate intel_mkl_src;
// extern crate openblas_src;
extern crate trees;
extern crate ihmm;

use std::env;
use trees::io::{Command,construct,predict,combined,Parameters};

fn main() {

    // manual_testing::test_command_predict_full();


    let mut arg_iter = env::args();

    let command_literal = arg_iter.next();

    let command = Command::parse(&arg_iter.next().unwrap());

    let mut parameters = Parameters::read(&mut arg_iter);

    parameters.command = command;

    match parameters.command {
        Command::Construct => construct(parameters),
        Command::Predict => predict(parameters),
        Command::Combined => combined(parameters),
        Command::Analyze => unimplemented!(),
    }

}
