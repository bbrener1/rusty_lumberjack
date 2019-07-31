// use rand::{Rng,ThreadRng,thread_rng};
// use ndarray::prelude::*;
// use std::cmp::Ordering;
// use std::f64;
// use std::mem::swap;
// use std::ops::Add;
//
// pub fn logit(p:f64) -> f64 {
//     (p/(1.-p)).ln()
// }
//
// pub fn weighted_choice<T: Rng>(weights: &Vec<f64>, rng: &mut T) -> usize {
//
//     let mut descending_weight:f64 = weights.iter().sum();
//
//     let choice = rng.gen::<f64>() * descending_weight;
//
//     for (i,weight) in weights.iter().enumerate() {
//         descending_weight -= *weight;
//         // println!("descending:{}",descending_weight);
//         if choice > descending_weight {
//             // println!("choice:{}",choice);
//             return i
//         }
//     }
//
//     0
// }
//
// pub fn weighted_sampling_with_replacement(draws: usize,weights: &Vec<f64>) -> Vec<usize> {
//     let mut rng = thread_rng();
//     let mut drawn_indecies: Vec<usize> = Vec::with_capacity(draws);
//     let mut weight_sum: f64 = weights.iter().sum();
//
//     let mut weighted_choices: Vec<f64> = (0..draws).map(|_| rng.gen_range::<f64,f64,f64>(0.,weight_sum)).collect();
//     weighted_choices.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
//
//     let mut current_choice = weighted_choices.pop().unwrap_or(-1.);
//
//     'f_loop: for (i,element) in weights.iter().enumerate() {
//         weight_sum -= *element;
//         // println!("descending:{}",descending_weight);
//         'w_loop: while weight_sum < current_choice {
//
//                 // println!("choice:{}",choice);
//
//                 // if weighted_choices.len()%1000 == 0 {
//                 //     if weighted_choices.len() > 0 {
//                 //         // println!("{}",weighted_choices.len());
//                 //     }
//                 // }
//
//             drawn_indecies.push(i);
//             if let Some(new_choice) = weighted_choices.pop() {
//                 current_choice = new_choice;
//             }
//             else {
//                 break 'f_loop;
//             }
//         }
//
//     }
//     drawn_indecies
// }
//
//
//
// pub fn weighted_sampling_with_increasing_similarity(draws:usize,weights:Option<&Vec<f64>>,similarity:&Array<f64,Ix2>) -> Vec<usize> {
//     let n = similarity.shape().0
//     let mut log_odds = Array::zeros(n);
//     log_odds += &weights.unwrap_or(vec![1.;n]).iter().map(|v| v.ln()).collect();
//     let mut selected_indecies = Vec::with_capacity(draws);
//     let mut rng = thread_rng();
//
//     for _ in 0..draws {
//
//         let local_odds = local_log_odds.iter().map(|x| x.exp()).collect();
//
//         let selection = weighted_choice(&local_odds, &mut rng);
//
//         let log_odds = &similarity[selection];
//
//         for (sim,log_odds) in selection_similarity.iter().zip(local_log_odds.iter_mut()) {
//             *log_odds += sim.ln();
//         }
//         local_log_odds[selection] = f64::NEG_INFINITY;
//
//         selected_indecies.push(selection);
//     }
//
//     selected_indecies
// }
//
// #[cfg(test)]
// pub mod randutil_test {
//
//     use super::*;
//
//     #[test]
//     fn test_argmin() {
//
//     }
