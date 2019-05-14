use std;
use std::sync::Arc;

use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;

use std::thread;

extern crate rand;
use rand::thread_rng;
use rand::Rng;

use crate::tree::Tree;
use crate::tree::PredictiveTree;
use crate::io::Parameters;
use crate::io::DispersionMode;

impl TreeThreadPool{
    pub fn new(prototype:&Tree, parameters: Arc<Parameters>) -> Sender<(usize, mpsc::Sender<PredictiveTree>)> {

        println!("Prototype tree: {},{},{}", prototype.input_features().len(), prototype.output_features().len(),prototype.root.samples().len());
        println!("Parameters:{:?},{:?},{:?}", parameters.input_features,parameters.output_features,parameters.sample_subsample);

        let processors = parameters.processor_limit;
        let samples_per_tree = parameters.sample_subsample;
        let input_features = parameters.input_features;
        let output_features = parameters.output_features;
        let dispersion_mode = parameters.dispersion_mode;

        if processors < 1 {
            panic!("Warning, no processors were allocated to the pool, quitting!");
        }

        let (tx,rx) = mpsc::channel();

        let worker_receiver_channel = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(processors);

        if processors > 20 {
            for i in 0..(processors/20) {

                println!("Spawning tree pool worker");
                println!("Prototype tree has {} threads", processors/(processors/30));

                workers.push(Worker::new(i,prototype.pool_switch_clone(processors/(processors/30)),samples_per_tree,input_features,output_features, dispersion_mode, worker_receiver_channel.clone()))

            }
        }
        else {
            workers.push(Worker::new(0,prototype.pool_switch_clone(processors),samples_per_tree,input_features,output_features, dispersion_mode, worker_receiver_channel.clone()))
        }

        tx
    }

    pub fn terminate(channel: &mut Sender<(usize, mpsc::Sender<PredictiveTree>)>) {
        while let Ok(()) = channel.send((0,mpsc::channel().0)) {}
    }

}

pub struct TreeThreadPool {
    workers: Vec<Worker>,
    worker_receiver_channel: Arc<Mutex<Receiver<(Tree, mpsc::Sender<PredictiveTree>)>>>,
}


impl Worker{

    pub fn new(id:usize,mut prototype:Tree,samples_per_tree:usize,input_features:usize,output_features:usize, dispersion_mode: DispersionMode, channel:Arc<Mutex<Receiver<(usize, mpsc::Sender<PredictiveTree>)>>>) -> Worker {
        Worker{
            id: id,
            thread: std::thread::spawn(move || {
                let mut rng = thread_rng();
                loop{
                    let message = channel.lock().unwrap().recv().ok();
                    if let Some((tree_iter,sender)) = message {
                        if tree_iter == 0 {
                            println!("Termination request");
                            prototype.terminate_pool();
                            break
                        }
                        println!("Tree Pool: Request for tree: {}",tree_iter);
                        println!("Tree Pool: Deriving {}", tree_iter);
                        let mut tree = prototype.derive_from_prototype(samples_per_tree,input_features,output_features,tree_iter);
                        match dispersion_mode {
                            DispersionMode::Mixed => {
                                if rng.gen::<f32>() > 0.4 { tree.set_dispersion_mode(DispersionMode::MAD) }
                                else { tree.set_dispersion_mode(DispersionMode::Variance) }
                            }
                            _ => {}
                        };
                        println!("Tree Pool: Growing {}", tree_iter);
                        println!("{:?}",tree.dispersion_mode());
                        tree.grow_branches(&prototype);
                        println!("Tree Pool: Sending {}", tree_iter);
                        let p_tree = tree.strip_consume();
                        sender.send(p_tree).expect("Tree worker thread error");
                    }
                }
            }),
        }
    }
}


struct Worker {
    id: usize,
    thread: thread::JoinHandle<()>,
    // worker_receiver_channel: Arc<Mutex<Receiver<((usize,(RankTableSplitter,RankTableSplitter,Vec<usize>),Vec<f64>), mpsc::Sender<(usize,usize,f64,Vec<usize>)>)>>>,
}
