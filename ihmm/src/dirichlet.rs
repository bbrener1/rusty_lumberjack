use std::f64::consts::E;
use std::f64::consts::PI;

use std::fmt::Debug;

use std::hash::Hash;
use std::cmp::Eq;
use std::collections::{HashSet,HashMap};

use std::num::NonZeroUsize;

use ndarray as nd;
use ndarray_linalg as ndl;
use num_traits;

use ndarray::{Ix1,Ix2,Axis};
use ndarray::{Array,ArrayView};
use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::solve::{Inverse,Determinant};
use ndarray_linalg::solveh::{InverseH,DeterminantH};

#[derive(Debug,Clone)]
pub struct Dirichlet<T: Hash + Eq + Copy + Debug> {
    categories: HashMap<T,usize>,
    log_odds: HashMap<T,f64>,
    a: NonZeroUsize,
    samples: usize,
}

impl<T: Hash + Eq + Copy + Debug> Dirichlet<T> {

    pub fn blank(dispersion:NonZeroUsize) -> Dirichlet<T> {
        Dirichlet {
            categories: HashMap::new(),
            log_odds: HashMap::new(),
            a: dispersion,
            samples: 0
        }
    }

    pub fn estimate(elements:&[T],dispersion:NonZeroUsize) -> Dirichlet<T> {
        let mut categories: HashMap<T,usize> = HashMap::new();
        for element in elements {
            *categories.entry(*element).or_insert(0) += 1;
        }
        let samples = elements.len();

        let mut log_odds = HashMap::new();

        let total = (samples + (categories.len() * dispersion.get())) as f64;

        for (key,key_count) in categories.iter() {
            let odds = (key_count + dispersion.get()) as f64 / (total+1.);
            log_odds.insert(*key, odds.log2());
        }

        Dirichlet {
            categories,
            log_odds,
            a: dispersion,
            samples,
        }


    }

    pub fn log_odds(&self,key:&T) -> Option<f64> {
        self.log_odds.get(key).map(|v| *v)
    }
}


#[cfg(test)]
mod dirichlet_tests {

    use super::*;
    use crate::MarkovNode;
    use crate::tree_braider_tests::iris_forest;

    #[test]
    fn test_dirichlet_count() {
        let items = vec!["eggs","bacon","eggs","milk"];
        let model = Dirichlet::estimate(&items, NonZeroUsize::new(1).unwrap());
        eprintln!("{:?}", model);
        for item in items {
            eprintln!("{:?}",model.log_odds(&item));
        }
        panic!();
    }

}
