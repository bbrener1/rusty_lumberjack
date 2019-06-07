use std::f64::consts::E;
use std::f64::consts::PI;
use std::f64;

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
pub struct SymmetricDirichlet<T: Hash + Eq + Copy + Debug> {
    categories: HashMap<T,usize>,
    odds: HashMap<T,f64>,
    log_odds: HashMap<T,f64>,
    a: NonZeroUsize,
    samples: usize,
}

#[derive(Debug,Clone)]
pub struct Categorical<T: Hash + Eq + Copy + Debug> {
    categories: HashMap<T,usize>,
    odds: HashMap<T,f64>,
    log_odds: HashMap<T,f64>,
    samples: usize,
}

impl<T: Hash + Eq + Copy + Debug> SymmetricDirichlet<T> {

    pub fn blank(dispersion:NonZeroUsize) -> SymmetricDirichlet<T> {
        SymmetricDirichlet {
            categories: HashMap::new(),
            log_odds: HashMap::new(),
            odds: HashMap::new(),
            a: dispersion,
            samples: 0
        }
    }

    pub fn blank_categories(categories:&[T],dispersion:NonZeroUsize) -> SymmetricDirichlet<T> {
        let categories: HashMap<T,usize> = categories.iter().map(|c| (*c,0)).collect();
        let odds: HashMap<T,f64> = categories.iter().map(|(category,count)| (*category, 1. / (categories.len() as f64))).collect();
        let log_odds: HashMap<T,f64> = odds.iter().map(|(c,o)| (*c,o.log2())).collect();
        SymmetricDirichlet {
            categories,
            odds,
            log_odds,
            a: dispersion,
            samples: 0
        }
    }


    pub fn estimate(&mut self, elements:&[T]) -> &mut SymmetricDirichlet<T> {
        for element in elements {
            *self.categories.entry(*element).or_insert(0) += 1;
        };

        self.samples += elements.len();

        let mut log_odds = HashMap::new();
        let mut odds = HashMap::new();

        let total = (self.samples + (self.categories.len() * self.a.get())) as f64;

        for (key,key_count) in self.categories.iter() {
            if self.categories.len() == 1 {
                odds.insert(*key,1.);
                log_odds.insert(*key,f64::INFINITY);
            }
            else {
                let category_odds = (key_count + self.a.get()) as f64 / (total - (key_count + self.a.get()) as f64);
                if !category_odds.is_normal() {
                    eprintln!("{:?}",(key,key_count,self.a));
                    panic!("Non-normal log odds in a dirichlet distribution, sure you did this right?");
                }
                odds.insert(*key,category_odds);
                log_odds.insert(*key, category_odds.log2());
            }
        }

        self.odds = odds;
        self.log_odds = log_odds;

        eprintln!("ESTIMATED MIXTURE");
        eprintln!("{:?}",self);
        self
    }

    pub fn from_map(categories:HashMap<T,usize>,dispersion: NonZeroUsize) -> SymmetricDirichlet<T> {

        let mut samples: usize = 0;

        for (category,category_count) in &categories {
            samples += category_count;
        };

        let mut log_odds = HashMap::new();
        let mut odds = HashMap::new();

        let total = (samples + (categories.len() * dispersion.get())) as f64;

        for (key,key_count) in categories.iter() {
            if categories.len() == 1 {
                odds.insert(*key,1.);
                log_odds.insert(*key,f64::INFINITY);
            }
            else {
                let category_odds = (key_count + dispersion.get()) as f64 / (total - (key_count + dispersion.get()) as f64);
                if !category_odds.is_normal() {
                    eprintln!("{:?}",(key,key_count,dispersion));
                    panic!("Non-normal log odds in a dirichlet distribution, sure you did this right?");
                }
                odds.insert(*key,category_odds);
                log_odds.insert(*key, category_odds.log2());
            }
        }

        SymmetricDirichlet {
            categories,
            odds,
            log_odds,
            a: dispersion,
            samples,
         }

    }

    pub fn log_odds(&self,key:&T) -> Option<f64> {
        self.log_odds.get(key).map(|v| *v)
    }

    pub fn odds(&self,key:&T) -> Option<f64> {
        self.odds.get(key).map(|v| *v)
    }

    pub fn probability(&self,key:&T) -> Option<f64> {
        self.categories.get(key).map(|&v| v as f64 / self.total() as f64)
    }

    pub fn total(&self) -> usize {
        self.samples + (self.categories.len() * self.a.get())
    }

    pub fn len(&self) -> usize {
        self.categories.len()
    }

}

impl<T: Hash + Eq + Copy + Debug> Categorical<T> {

    pub fn blank() -> Categorical<T> {
        Categorical {
            categories: HashMap::new(),
            log_odds: HashMap::new(),
            odds: HashMap::new(),
            samples: 0
        }
    }

    pub fn blank_categories(categories:&[T]) -> Categorical<T> {
        let categories: HashMap<T,usize> = categories.iter().map(|c| (*c,0)).collect();
        let odds: HashMap<T,f64> = categories.iter().map(|(category,count)| (*category, 1. / (categories.len() as f64))).collect();
        let log_odds: HashMap<T,f64> = odds.iter().map(|(c,o)| (*c,o.log2())).collect();
        Categorical {
            categories,
            odds,
            log_odds,
            samples: 0
        }
    }

    pub fn estimate(&mut self, elements:&[T]) -> &mut Categorical<T> {
        for element in elements {
            *self.categories.entry(*element).or_insert(0) += 1;
        };

        self.samples += elements.len();

        let mut log_odds = HashMap::new();
        let mut odds = HashMap::new();

        let total = self.samples as f64;

        for (&key,&key_count) in self.categories.iter() {
            if key_count as f64 == total {
                odds.insert(key,1.);
                log_odds.insert(key,f64::INFINITY);
            }
            else {
                let category_odds = key_count as f64 / (total - key_count as f64);
                if !category_odds.is_normal() {
                    eprintln!("{:?}",(key,key_count));
                    panic!("Non-normal log odds in a dirichlet distribution, sure you did this right?");
                }
                odds.insert(key,category_odds);
                log_odds.insert(key, category_odds.log2());
            }
        }

        self.odds = odds;
        self.log_odds = log_odds;

        eprintln!("ESTIMATED MIXTURE");
        eprintln!("{:?}",self);
        self
    }

    pub fn log_odds(&self,key:&T) -> Option<f64> {
        self.log_odds.get(key).map(|&v| v)
    }

    pub fn odds(&self,key:&T) -> Option<f64> {
        self.odds.get(key).map(|&v| v)
    }

    pub fn probability(&self,key:&T) -> Option<f64> {
        self.categories.get(key).map(|&v| v as f64 / self.samples as f64)
    }

    pub fn len(&self) -> usize {
        self.categories.len()
    }
}

impl<T: Hash + Eq + Copy + Debug> std::convert::From<SymmetricDirichlet<T>> for Categorical<T> {
    fn from(sd: SymmetricDirichlet<T>) -> Categorical<T> {
        let a = sd.a.get();
        let categories: HashMap<T,usize> = sd.categories.into_iter().map(|(category,count)| (category,count+a)).collect();
        let samples: usize = categories.iter().map(|(category,count)| count).sum();
        let odds: HashMap<T,f64> = categories.iter().map(|(category,count)| (*category, *count as f64 / (samples - count) as f64)).collect();
        let log_odds: HashMap<T,f64> = odds.iter().map(|(category,odds)| (*category,odds.log2())).collect() ;

        Categorical {
            categories,
            odds,
            log_odds,
            samples,
        }
    }
}



#[cfg(test)]
mod dirichlet_tests {

    use super::*;
    use crate::MarkovNode;
    use crate::tree_braider_tests::iris_forest;
    use std::f64;

    #[test]
    fn test_dirichlet() {
        let items = vec!["eggs","bacon","eggs","milk"];
        let mut dirichlet = SymmetricDirichlet::blank(NonZeroUsize::new(1).unwrap());
        dirichlet.estimate(&items);
        eprintln!("{:?}", dirichlet);
        let dirichlet_log_odds: Vec<Option<f64>> = items.iter().map(|item| dirichlet.log_odds(item)).collect();
        assert_eq!(vec![0.75,0.4,0.75,0.4].iter().map(|v| Some(f64::log2(*v))).collect::<Vec<Option<f64>>>(),dirichlet_log_odds);
        let dirichlet_odds: Vec<Option<f64>> = items.iter().map(|item| dirichlet.odds(item)).collect();
        assert_eq!(vec![0.75,0.4,0.75,0.4].iter().map(|v| Some(*v)).collect::<Vec<Option<f64>>>(),dirichlet_odds);
        let categorical = Categorical::from(dirichlet);
        eprintln!("{:?}", categorical);
        let categorical_log_odds: Vec<Option<f64>> = items.iter().map(|item| categorical.log_odds(item)).collect();
        assert_eq!(vec![0.75,0.4,0.75,0.4].iter().map(|v| Some(f64::log2(*v))).collect::<Vec<Option<f64>>>(),categorical_log_odds);
        let categorical_odds: Vec<Option<f64>> = items.iter().map(|item| categorical.odds(item)).collect();
        assert_eq!(vec![0.75,0.4,0.75,0.4].iter().map(|v| Some(*v)).collect::<Vec<Option<f64>>>(),categorical_odds);
    }

}
