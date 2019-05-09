// #![feature(test)]
//
// extern crate test;
// use test::Bencher;

#[macro_use]
extern crate serde_derive;


extern crate serde;
extern crate serde_json;
extern crate num_cpus;
extern crate rand;
extern crate time;
extern crate smallvec;

#[macro_use(array,azip)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate num_traits;


mod rank_vector;
mod rank_table;
mod io;
mod node;
mod tree;
mod random_forest;
mod tree_thread_pool;
mod feature_thread_pool;
mod split_thread_pool;
mod randutils;
mod tree_braider4;
mod multivariate_normal;
mod dirichlet;

use io::Parameters;
use io::construct;
use io::predict;
use io::combined;
use io::Command;


use std::env;
use std::io as sio;
use std::f64;
use std::collections::HashMap;
use std::fs::File;
use std::io::stdin;
use std::io::prelude::*;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;


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
    }

}



fn read_header(location: &str) -> Vec<String> {

    println!("Reading header: {}", location);

    let mut header_map = HashMap::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = sio::BufReader::new(&header_file).lines();

    for (i,line) in header_file_iterator.by_ref().enumerate() {
        let feature = line.unwrap_or("error".to_string());
        let mut renamed = feature.clone();
        let mut j = 1;
        while header_map.contains_key(&renamed) {
            renamed = [feature.clone(),j.to_string()].join("");
            eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
            j += 1;
        }
        header_map.insert(renamed,i);
    };

    let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
    header_inter.sort_unstable_by_key(|x| x.1);
    let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

    println!("Read {} lines", header_vector.len());

    header_vector
}

fn read_sample_names(location: &str) -> Vec<String> {

    let mut header_vector = Vec::new();

    let sample_name_file = File::open(location).expect("Sample name file error!");
    let mut sample_name_lines = sio::BufReader::new(&sample_name_file).lines();

    for line in sample_name_lines.by_ref() {
        header_vector.push(line.expect("Error reading header line!").trim().to_string())
    }

    header_vector
}


fn argmin(in_vec: &Vec<f64>) -> (usize,f64) {
    let mut min_ind = 0;
    let mut min_val: f64 = 1./0.;
    for (i,val) in in_vec.iter().enumerate() {
        // println!("Argmin debug:{},{},{}",i,val,min_val);
        // match val.partial_cmp(&min_val).unwrap_or(Ordering::Less) {
        //     Ordering::Less => println!("Less"),
        //     Ordering::Equal => println!("Equal"),
        //     Ordering::Greater => println!("Greater")
        // }
        match val.partial_cmp(&min_val).unwrap_or(Ordering::Less) {
            Ordering::Less => {min_val = val.clone(); min_ind = i.clone()},
            Ordering::Equal => {},
            Ordering::Greater => {}
        }
    }
    (min_ind,min_val)
}



fn matrix_flip<T:Clone>(in_mat: &Vec<Vec<T>>) -> Vec<Vec<T>> {

    let dim = mtx_dim(in_mat);

    let mut out = vec![Vec::with_capacity(dim.0);dim.1];

    for (i,iv) in in_mat.iter().enumerate() {
        for (j,jv) in iv.iter().enumerate() {
            out[j].push(jv.clone());
        }
    }

    out
}

fn mtx_dim<T>(in_mat: &Vec<Vec<T>>) -> (usize,usize) {
    (in_mat.len(),in_mat.get(0).unwrap_or(&vec![]).len())
}

fn add_mtx(mat1: &Vec<Vec<f64>>,mat2:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(mat1) != mtx_dim(mat2) {
        panic!("Attempted to add matrices of unequal dimensions: {:?},{:?}", mtx_dim(mat1),mtx_dim(mat2));
    }

    let dim = mtx_dim(mat1);

    let mut output = vec![vec![0.;dim.1];dim.0];

    for i in 0..dim.0 {
        for j in 0..dim.1{
            output[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    output

}

fn sub_mtx(mat1: &Vec<Vec<f64>>,mat2:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(mat1) != mtx_dim(mat2) {
        panic!("Attempted to subtract matrices of unequal dimensions: {:?},{:?}", mtx_dim(mat1),mtx_dim(mat2));
    }

    let dim = mtx_dim(mat1);

    let mut output = vec![vec![0.;dim.1];dim.0];

    for i in 0..dim.0 {
        for j in 0..dim.1{
            output[i][j] = mat1[i][j] - mat2[i][j];
        }
    }

    output
}

fn add_mtx_ip(mut mtx1: Vec<Vec<f64>>, mtx2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(&mtx1) != mtx_dim(&mtx2) {
        panic!("Attempted to add matrices of unequal dimensions: {:?},{:?}", mtx_dim(&mtx1),mtx_dim(&mtx2));
    }

    let dim = mtx_dim(&mtx1);

    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx1[i][j] += mtx2[i][j];
        }
    }

    mtx1
}

fn sub_mtx_ip(mut mtx1: Vec<Vec<f64>>, mtx2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(&mtx1) != mtx_dim(&mtx2) {
        panic!("Attempted to subtract matrices of unequal dimensions: {:?},{:?}", mtx_dim(&mtx1),mtx_dim(&mtx2));
    }

    let dim = mtx_dim(&mtx1);

    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx1[i][j] += mtx2[i][j];
        }
    }

    mtx1
}

fn abs_mtx_ip(mtx: &mut Vec<Vec<f64>>) {
    let dim = mtx_dim(&mtx);
    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx[i][j] = mtx[i][j].abs();
        }
    }
}

fn square_mtx_ip(mut mtx: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let dim = mtx_dim(&mtx);
    for i in 0..dim.0 {
        for j in 0..dim.1{
            mtx[i][j] = mtx[i][j].powi(2);
        }
    }
    mtx
}


fn multiply_matrix(mat1: &Vec<Vec<f64>>,mat2:&Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    if mtx_dim(mat1) != mtx_dim(mat2) {
        panic!("Attempted to multiply matrices of unequal dimensions: {:?},{:?}", mtx_dim(mat1),mtx_dim(mat2));
    }

    let dim = mtx_dim(mat1);

    let mut output = vec![vec![0.;dim.1];dim.0];

    for i in 0..dim.0 {
        for j in 0..dim.1{
            output[i][j] = mat1[i][j] * mat2[i][j];
        }
    }

    output
}

fn zero_matrix(x:usize,y:usize) -> Vec<Vec<f64>> {
    vec![vec![0.;y];x]
}

fn float_matrix(x:usize,y:usize,float:f64) -> Vec<Vec<f64>> {
    vec![vec![float;y];x]
}

fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
    out
}

// fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
//     let mut out = input.iter().cloned().enumerate().collect::<Vec<(usize,f64)>>();
//     out.sort_unstable_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//     out
// }

fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {

    input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")

}

fn median(input: &Vec<f64>) -> (usize,f64) {
    let mut index = 0;
    let mut value = 0.;

    let mut sorted_input = input.clone();
    sorted_input.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    if sorted_input.len() % 2 == 0 {
        index = sorted_input.len()/2;
        value = (sorted_input[index-1] + sorted_input[index]) / 2.
    }
    else {
        if sorted_input.len() % 2 == 1 {
            index = (sorted_input.len()-1)/2;
            value = sorted_input[index]
        }
        else {
            panic!("Median failed!");
        }
    }
    (index,value)
}

fn mean(input: &Vec<f64>) -> f64 {
    input.iter().sum::<f64>() / (input.len() as f64)
}

fn covariance(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {

    if vec1.len() != vec2.len() {
        panic!("Tried to compute covariance for unequal length vectors: {}, {}",vec1.len(),vec2.len());
    }

    let mean1: f64 = mean(vec1);
    let mean2: f64 = mean(vec1);

    let covariance = vec1.iter().zip(vec2.iter()).map(|(x,y)| (x - mean1) * (y - mean2)).sum::<f64>() / (vec1.len() as f64 - 1.);

    if covariance.is_nan() {0.} else {covariance}

}

pub fn variance(input: &Vec<f64>) -> f64 {

    let mean = mean(input);

    let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() as f64 - 1.).max(1.);

    if var.is_nan() {0.} else {var}
}


pub fn std_dev(input: &Vec<f64>) -> f64 {

    let mean = mean(input);

    let std_dev = (input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() as f64 - 1.).max(1.)).sqrt();

    if std_dev.is_nan() {0.} else {std_dev}
}

fn pearsonr(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {

    if vec1.len() != vec2.len() {
        panic!("Tried to compute correlation for unequal length vectors: {}, {}",vec1.len(),vec2.len());
    }

    let mean1: f64 = mean(vec1);
    let mean2: f64 = mean(vec2);

    let dev1: Vec<f64> = vec1.iter().map(|x| (x - mean1)).collect();
    let dev2: Vec<f64> = vec2.iter().map(|x| (x - mean2)).collect();

    let covariance = dev1.iter().zip(dev2.iter()).map(|(x,y)| x * y).sum::<f64>() / (vec1.len() as f64 - 1.);

    let std_dev1 = (dev1.iter().map(|x| x.powi(2)).sum::<f64>() / (vec1.len() as f64 - 1.).max(1.)).sqrt();
    let std_dev2 = (dev2.iter().map(|x| x.powi(2)).sum::<f64>() / (vec2.len() as f64 - 1.).max(1.)).sqrt();

    // println!("{},{}", std_dev1,std_dev2);

    let r = covariance / (std_dev1*std_dev2);

    if r.is_nan() {0.} else {r}

}
