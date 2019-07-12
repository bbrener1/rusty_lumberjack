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
extern crate rayon;

extern crate num_traits;

mod rank_vector;
mod rank_table;
pub mod io;
pub mod node;
mod tree;
mod random_forest;
mod randutils;


use std::env;
use std::io as sio;
use std::f64;
use std::collections::{HashMap,HashSet};
use std::fs::File;
use std::io::stdin;
use std::io::prelude::*;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;

use io::DispersionMode;
use rank_vector::{RankVector,Node};


#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq,Hash)]
pub struct Feature {
    name: String,
    index: usize,
}

impl Feature {

    pub fn vec(input: Vec<usize>) -> Vec<Feature> {
        input.iter().map(|x| Feature::q(x)).collect()
    }

    pub fn nvec(input: &Vec<String>) -> Vec<Feature> {
        input.iter().enumerate().map(|(i,f)| Feature::new(f,&i)).collect()
    }

    pub fn q(index:&usize) -> Feature {
        Feature {name: index.to_string(),index:*index}
    }

    pub fn new(name:&str,index:&usize) -> Feature {
        Feature {name: name.to_owned(),index:*index}
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn index(&self) -> &usize {
        &self.index
    }
}

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq,Hash)]
pub struct Sample {
    name: String,
    index: usize,
}

impl Sample {

    pub fn vec(input: Vec<usize>) -> Vec<Sample> {
        input.iter().map(|x| Sample::q(x)).collect()
    }

    pub fn nvec(input: &Vec<String>) -> Vec<Sample> {
        input.iter().enumerate().map(|(i,s)| Sample::new(s,&i)).collect()
    }

    pub fn q(index:&usize) -> Sample {
        Sample {name: index.to_string(),index:*index}
    }

    pub fn new(name:&str,index:&usize) -> Sample {
        Sample {name: name.to_owned(),index:*index}
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn index(&self) -> &usize {
        &self.index
    }

}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Prerequisite {
    feature: Feature,
    split: f64,
    orientation: bool
}

impl Prerequisite {
    pub fn new(feature:Feature,split:f64,orientation:bool) -> Prerequisite {
        Prerequisite {feature,split,orientation}
    }
}

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Split {
    feature: Feature,
    value: f64,
    dispersion: f64,
}

impl Split {
    pub fn new(feature: Feature,value:f64,dispersion:f64) -> Split {
        Split {
            feature,
            value,
            dispersion,
        }
    }

    pub fn left(&self) -> Prerequisite {
        Prerequisite::new(self.feature.clone(), self.value, false)
    }

    pub fn right(&self) -> Prerequisite {
        Prerequisite::new(self.feature.clone(), self.value, true)
    }

}

#[derive(Clone,Serialize,Deserialize,Debug)]
pub struct Braid {
    features: Vec<Feature>,
    samples: Vec<Sample>,
    compound_values: Vec<f64>,
    draw_order: Vec<usize>,
    drop_set: HashSet<usize>,
    compound_split: Option<f64>,
}

impl Braid {
    fn from_rvs(features: Vec<Feature>,samples:Vec<Sample>,rvs: &[RankVector<Vec<Node>>]) -> Braid {

        let len = rvs.get(0).unwrap_or(&RankVector::<Vec<Node>>::empty()).raw_len();

        assert!(!rvs.iter().any(|rv| rv.raw_len() != len));

        // We would like to convert all values to rank-values using modified competition ranking

        // There are two advantages to using rank values: scaling, eg all features scaled identically,
        // and converting sparse 0s to 1s, allowing us to use geometric averaging without headaches.

        // Modified competition ranking is used to preserve extreme values as extreme. Eg, for a
        // sparse dataset, [0,0,0,0,0,0,10], conventional ranking would rank 10 as 1. This is
        // undesirable when comparing to ex [1,2,3,3,4,4,10], where conventional ranking would
        // rank 10 as 5.

        let ranked_values: Vec<Vec<usize>> = rvs.iter().map(|rv| modified_competition_ranking(&rv.full_values())).collect();

        let mut compound_values: Vec<f64> = vec![0.;len];

        for i in 0..len {
            for vec in ranked_values.iter() {

                // Here we can guarantee that all values are above 0 because we are using
                // 1-indexed ranking instead of raw values for geometric averaging.
                // Therefore we can safely take a ln.

                compound_values[i] += (vec[i] as f64).ln();
            }
            compound_values[i] /= ranked_values.len() as f64;
            compound_values[i] = compound_values[i].exp();
        }

        let compound_vector = RankVector::<Vec<Node>>::link(&compound_values);
        let (draw_order,drop_set) = compound_vector.draw_and_drop();

        Braid {
            features,
            samples,
            // compound_vector,
            compound_values,
            draw_order,
            drop_set,
            compound_split: None,
        }
    }

    fn draw_order(&self) -> (&[usize],&HashSet<usize>) {
        (&self.draw_order,&self.drop_set)
    }

    fn set_split(&mut self, split:f64) {
        self.compound_split = Some(split)
    }

    fn set_split_by_index(&mut self, split:usize) {
        let value = self.compound_values[split];
        self.compound_split = Some(value);
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


fn argmin(in_vec: &[f64]) -> Option<(usize,f64)> {
    let mut minimum = None;
    for (j,&val) in in_vec.iter().enumerate() {
        let check = if let Some((i,m)) = minimum.take() {
            match val.partial_cmp(&m).unwrap_or(Ordering::Greater) {
                Ordering::Less => {Some((j,val))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((i,m))},
            }
        }
        else {
            if !val.is_nan() {
                Some((j,val))
            }
            else {
                None
            }
        };
        minimum = check;

    };
    minimum
}

fn argmax(in_vec: &[f64]) -> Option<(usize,f64)> {
    let mut maximum = None;
    for (j,&val) in in_vec.iter().enumerate() {
        let check = if let Some((i,m)) = maximum.take() {
            match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                Ordering::Less => {Some((i,m))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((j,val))},
            }
        }
        else {
            if !val.is_nan() {
                Some((j,val))
            }
            else {
                None
            }
        };
        maximum = check;

    };
    maximum
}

pub fn gn_argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check = if let Some((i,m)) = maximum.take() {
            match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                Ordering::Less => {Some((i,m))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((j,val))},
            }
        }
        else {
            if val.partial_cmp(&val).is_some() { Some((j,val)) }
            else { None }
        };
        maximum = check;

    };
    maximum.map(|(i,m)| i)
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

fn modified_competition_ranking(input:&[f64]) -> Vec<usize> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().map(|(rank,(position,value))| (rank+1,(*position,*value))).collect::<Vec<(usize,(usize,&f64))>>();
    for i in 0..(intermediate2.len().max(1)-1) {
        let (r1,(_,v1)) = intermediate2[i];
        let (_,(_,v2)) = intermediate2[i+1];
        if v1 == v2 {
            intermediate2[i+1].0 = r1;
        }
    }
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    intermediate2.into_iter().map(|(rank,(position,value))| rank).collect()
}

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

pub fn l1_sum(mtx_in:&Vec<Vec<f64>>, weights: &[f64]) -> Vec<f64> {
    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature * weights[i] ).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY}).collect();
    sample_sums
}

pub fn l2_sum(mtx_in:&Vec<Vec<f64>>, weights: &[f64]) -> Vec<f64> {
    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature.powi(2) * weights[i] ).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY}).collect();
    sample_sums
}

#[cfg(test)]
pub mod tree_lib_tests {

    use super::*;

    #[test]
    fn test_argmin() {

        let na = std::f64::NAN;

        assert_eq!(argmin(&vec![1.]),Some((0,1.)));
        assert_eq!(argmin(&vec![1.,2.]),Some((0,1.)));
        assert_eq!(argmin(&vec![]),None);
        assert_eq!(argmin(&vec![1.,na]),Some((0,1.)));
        assert_eq!(argmin(&vec![na,1.]),Some((1,1.)));
        assert_eq!(argmin(&vec![na,na]),None);
        assert_eq!(argmin(&vec![1.,1.]),Some((0,1.)));

    }

    #[test]
    fn test_argmax() {

        let na = std::f64::NAN;

        assert_eq!(argmax(&vec![1.]),Some((0,1.)));
        assert_eq!(argmax(&vec![1.,2.]),Some((1,2.)));
        assert_eq!(argmax(&vec![]),None);
        assert_eq!(argmax(&vec![1.,na]),Some((0,1.)));
        assert_eq!(argmax(&vec![na,1.]),Some((1,1.)));
        assert_eq!(argmax(&vec![na,na]),None);
        assert_eq!(argmax(&vec![1.,1.]),Some((0,1.)));


    }

    #[test]
    fn test_gn_argmax() {

        let na = std::f64::NAN;

        assert_eq!(gn_argmax(vec![1.].iter()),Some((0)));
        assert_eq!(gn_argmax(vec![1.,2.].iter()),Some((1)));
        assert_eq!(gn_argmax(Vec::<f64>::new().iter()),None);
        assert_eq!(gn_argmax(vec![1.,na].iter()),Some((0)));
        assert_eq!(gn_argmax(vec![na,1.].iter()),Some((1)));
        assert_eq!(gn_argmax(vec![na,na].iter()),None);
        assert_eq!(gn_argmax(vec![1.,1.].iter()),Some((0)));


    }

    #[test]
    fn test_modified_competition_ranking() {

        let a = vec![2.,0.,1.,2.,0.,3.];
        let b = modified_competition_ranking(&a);
        let c = vec![4,1,3,4,1,6];

        let d: Vec<f64> = Vec::new();
        let e: Vec<usize> = Vec::new();

        assert_eq!(b,c);
        assert_eq!(e,modified_competition_ranking(&d));

    }

}
